#include "dart/server/GUIWebsocketServer.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

#include <assimp/scene.h>
#include <boost/filesystem.hpp>

#include "dart/collision/CollisionResult.hpp"
#include "dart/common/Aspect.hpp"
#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/server/RawJsonUtils.hpp"
#include "dart/server/external/base64/base64.h"
#include "dart/simulation/World.hpp"

namespace dart {
namespace server {

GUIWebsocketServer::GUIWebsocketServer()
  : mPort(-1),
    mServing(false),
    mStartingServer(false),
    mScreenSize(Eigen::Vector2i(680, 420))
{
}

GUIWebsocketServer::~GUIWebsocketServer()
{
  {
    const std::unique_lock<std::mutex> lock(this->mServingMutex);
    if (!mServing)
      return;
  }
  dterr << "GUIWebsocketServer is being deallocated while it's still "
           "serving! The server will now terminate, and attempt to clean up. "
           "If this was not intended "
           "behavior, please keep a reference to the GUIWebsocketServer to "
           "keep the server alive. If this was intended behavior, please "
           "call "
           "stopServing() on "
           "the server before deallocating it."
        << std::endl;
  stopServing();
}

/// This is a non-blocking call to start a websocket server on a given port
void GUIWebsocketServer::serve(int port)
{
  mPort = port;
  // Register signal and signal handler
  {
    const std::unique_lock<std::mutex> lock(this->mServingMutex);
    if (mServing || mStartingServer)
    {
      std::cout << "Errer in GUIWebsocketServer::serve()! Already serving. "
                   "Ignoring request."
                << std::endl;
      return;
    }
    // We're not serving yet, but we are starting the server
    mServing = false;
    mStartingServer = true;
  }
  mServer = new WebsocketServer();

  // Register our network callbacks, ensuring the logic is run on the main
  // thread's event loop
  mServer->connect([this](ClientConnection conn) {
    {
      // We don't need high throughput, so run everything through a global mutex
      // to avoid data races
      const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

      // Send a hello message to the client
      // mServer->send(conn) seems to break, cause conn appears to get cleaned
      // up in race conditions (it's a weak pointer)

      std::string jsonStr = getCurrentStateAsJson();
      try
      {
        mServer->send(conn, base64_encode(jsonStr));
      }
      catch (...)
      {
        dterr << "GUIWebsocketServer caught an error broadcasting message \""
              << jsonStr << "\"" << std::endl;
      }
      // mServer->broadcast("{\"type\": 1}");
      /*
      mServer->broadcast(
          "{\"type\": \"init\", \"world\": " + mWorld->toJson() + "}");
      */
    }

    // Don't hold the globalMutex when calling connection listeners, because
    // that can lead to deadlocks if the connection listeners call out to Python
    // (which tries to grab the GIL) while other Python code (holding the GIL)
    // tries to grab the globalMutex.

    for (auto listener : mConnectionListeners)
    {
      listener();
    }
  });

  mServer->disconnect([this](ClientConnection /* conn */) {
    std::clog << "Connection closed." << std::endl;
    std::clog << "There are now " << mServer->numConnections()
              << " open connections." << std::endl;
  });
  mServer->message([this](
                       ClientConnection /* conn */, const Json::Value& args) {
    if (args["type"].asString() == "keydown")
    {
      std::string key = args["key"].asString();
      {
        const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);
        this->mKeysDown.insert(key);
      }
      for (auto listener : this->mKeydownListeners)
      {
        listener(key);
      }
    }
    else if (args["type"].asString() == "keyup")
    {
      std::string key = args["key"].asString();
      {
        const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);
        this->mKeysDown.erase(key);
      }
      for (auto listener : this->mKeyupListeners)
      {
        listener(key);
      }
    }
    else if (args["type"].asString() == "button_click")
    {
      std::string key = this->getCodeString(args["key"].asInt());
      if (mButtons.find(key) != mButtons.end())
      {
        mButtons[key].onClick();
      }
    }
    else if (args["type"].asString() == "slider_set_value")
    {
      std::string key = this->getCodeString(args["key"].asInt());
      s_t value = static_cast<s_t>(args["value"].asDouble());
      if (mSliders.find(key) != mSliders.end())
      {
        mSliders[key].value = value;
        mSliders[key].onChange(value);
      }
    }
    else if (args["type"].asString() == "screen_resize")
    {
      Eigen::Vector2i size
          = Eigen::Vector2i(args["size"][0].asInt(), args["size"][1].asInt());
      mScreenSize = size;

      for (auto handler : mScreenResizeListeners)
      {
        handler(size);
      }
    }
    else if (args["type"].asString() == "drag")
    {
      std::string key = this->getCodeString(args["key"].asInt());
      Eigen::Vector3s pos = Eigen::Vector3s(
          static_cast<s_t>(args["pos"][0].asDouble()),
          static_cast<s_t>(args["pos"][1].asDouble()),
          static_cast<s_t>(args["pos"][2].asDouble()));

      for (auto handler : mDragListeners[key])
      {
        handler(pos);
      }
    }
    else if (args["type"].asString() == "drag_end")
    {
      std::string key = this->getCodeString(args["key"].asInt());
      for (auto handler : mDragEndListeners[key])
      {
        handler();
      }
    }
    else if (args["type"].asString() == "edit_tooltip")
    {
      std::string key = this->getCodeString(args["key"].asInt());
      std::string tooltip = args["tooltip"].asString();

      for (auto handler : mTooltipChangeListeners[key])
      {
        handler(tooltip);
      }
    }
  });

  // unblock signals in this thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_UNBLOCK, &sigset, nullptr);

  /*
  // The signal set is used to register termination notifications
  mSignalSet = new asio::signal_set(mServerEventLoop, SIGINT, SIGTERM);
  // register the handle_stop callback
  mSignalSet->async_wait([&](asio::error_code const& error, int signal_number) {
    if (error == asio::error::operation_aborted)
    {
      std::cout << "Signal listener was terminated by asio" << std::endl;
    }
    else if (error)
    {
      std::cout << "Got an error registering termination signals: " << error
                << std::endl;
    }
    else if (
        signal_number == SIGINT || signal_number == SIGTERM
        || signal_number == SIGQUIT)
    {
      std::cout << "Shutting down the server..." << std::endl;
      stopServing();
      mServerEventLoop.stop();
      exit(signal_number);
    }
  });
  */

  // Start the networking thread
  mServerThread = new std::thread([this, port]() {
    /*
    // block signals in this thread and subsequently
    // spawned threads so they're guaranteed to go to the main thread
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, nullptr);
    */

    std::cout << "GUIWebsocketServer will start serving a WebSocket server on "
                 "ws://localhost:"
              << port << std::endl;

    // Note that we've started, but do it from within the server's event loop
    // once the server has _actually_ started.
    mServer->eventLoop.post([&]() {
      {
        const std::unique_lock<std::mutex> lock(this->mServingMutex);
        mStartingServer = false;
        mServing = true;
        mServingConditionValue.notify_all();
      }

      // Start the flush thread
      mFlushThread = new std::thread([this]() { this->flushThread(); });
    });

    bool success = mServer->run(port);
    if (!success)
    {
      // This means we failed to bind to the port
      stopServing();
    }
  });
}

/// This kills the server, if one was running
void GUIWebsocketServer::stopServing()
{
  {
    std::unique_lock<std::mutex> lock(this->mServingMutex);
    if (mStartingServer)
    {
      std::cout << "GUIWebsocketServer called stopServing() while we're in the "
                   "middle of booting "
                   "the server. Waiting until booting finished..."
                << std::endl;
      mServingConditionValue.wait(lock, [&]() { return !mStartingServer; });
      std::cout << "GUIWebsocketServer finished booting server, will now "
                   "resume stopServing()."
                << std::endl;
    }
    if (!mServing)
      return;
    mServing = false;
  }
  std::cout << "GUIWebsocketServer is shutting down the WebSocket server on "
               "ws://localhost:"
            << mPort << std::endl;
  assert(mServer != nullptr);
  mServer->stop();
  assert(mServerThread != nullptr);
  mServerThread->join();
  delete mServer;
  delete mServerThread;
  assert(mFlushThread != nullptr);
  mFlushThread->join();
  delete mFlushThread;
  mServer = nullptr;
  mServerThread = nullptr;
  mServingConditionValue.notify_all();
  mFlushThread = nullptr;
}

/// Returns true if we're serving
bool GUIWebsocketServer::isServing()
{
  return mServing;
}

/// This flushes at a fixed framerate, not too fast to overwhelm the web GUI
void GUIWebsocketServer::flushThread()
{
  while (mServing)
  {
    flush();
    // limit to sending updates at 50fps
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

/// This sleeps until we're done serving, without busy-waiting in a loop. It
/// wakes up occassionally to call the `checkForSignals` callback, where you
/// can throw an exception to shut down the program.
void GUIWebsocketServer::blockWhileServing(
    std::function<void()> checkForSignals)
{
  std::unique_lock<std::mutex> lock(this->mServingMutex);
  if (!mServing && !mStartingServer)
    return;
  while (true)
  {
    if (mServingConditionValue.wait_for(
            lock, std::chrono::milliseconds(1000), [&]() {
              return !mServing && !mStartingServer;
            }))
    {
      // Our condition was met!
      return;
    }
    else
    {
      // Wake up and check for signals
      checkForSignals();
    }
  }
}

/// This adds a listener that will get called when someone connects to the
/// server
void GUIWebsocketServer::registerConnectionListener(
    std::function<void()> listener)
{
  mConnectionListeners.push_back(listener);
}

/// This adds a listener that will get called when ctrl+C is pressed
void GUIWebsocketServer::registerShutdownListener(
    std::function<void()> listener)
{
  mShutdownListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-down event
/// on the web client
void GUIWebsocketServer::registerKeydownListener(
    std::function<void(std::string)> listener)
{
  mKeydownListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-up event
/// on the web client
void GUIWebsocketServer::registerKeyupListener(
    std::function<void(std::string)> listener)
{
  mKeyupListeners.push_back(listener);
}

/// Gets the set of all the keys currently being pressed
const std::unordered_set<std::string>& GUIWebsocketServer::getKeysDown() const
{
  return mKeysDown;
}

/// Returns true if a key is currently being pressed
bool GUIWebsocketServer::isKeyDown(const std::string& key) const
{
  return mKeysDown.find(key) != mKeysDown.end();
}

/// This sends the current list of commands to the web GUI
void GUIWebsocketServer::flush()
{
  if (mServing && mMessagesQueued > 0)
  {
    std::string json = flushJson();
    try
    {
      mServer->broadcast(base64_encode(json));
    }
    catch (...)
    {
      dterr << "GUIWebsocketServer caught an error broadcasting message \""
            << json << "\"" << std::endl;
    }
  }
}

/// This completely resets the web GUI, deleting all objects, UI elements, and
/// listeners
void GUIWebsocketServer::clear()
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  GUIStateMachine::clear();
  mScreenResizeListeners.clear();
  mKeydownListeners.clear();
  mShutdownListeners.clear();
}

/// This enables mouse events on an object (if they're not already), and calls
/// "listener" whenever the object is dragged with the desired drag
/// coordinates
GUIWebsocketServer& GUIWebsocketServer::registerDragListener(
    const std::string& key,
    std::function<void(Eigen::Vector3s)> listener,
    std::function<void()> endDrag)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  setObjectDragEnabled(key);
  mDragListeners[key].push_back(listener);
  mDragEndListeners[key].push_back(endDrag);
  return *this;
}

/// This enables the user to edit the tooltip on an object, and calls this
/// listener when the tooltip changes.
GUIWebsocketServer& GUIWebsocketServer::registerTooltipChangeListener(
    const std::string& key, std::function<void(std::string)> listener)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  setObjectTooltipEditable(key);
  mTooltipChangeListeners[key].push_back(listener);
  return *this;
}

/// This gets the current screen size
Eigen::Vector2i GUIWebsocketServer::getScreenSize()
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  return mScreenSize;
}

/// This registers a callback to get called whenever the screen size changes.
void GUIWebsocketServer::registerScreenResizeListener(
    std::function<void(Eigen::Vector2i)> listener)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mScreenResizeListeners.push_back(listener);
}

} // namespace server
} // namespace dart