#include "dart/server/GUIWebsocketServer.hpp"

#include "dart/server/RawJsonUtils.hpp"

namespace dart {
namespace server {

GUIWebsocketServer::GUIWebsocketServer()
  : mServing(false), mMessagesQueued(0), mAutoflush(true)
{
  mJson << "[";
}

/// This is a non-blocking call to start a websocket server on a given port
void GUIWebsocketServer::serve(int port)
{
  // Register signal and signal handler
  if (mServing)
  {
    std::cout << "Errer in GUIWebsocketServer::serve()! Already serving. "
                 "Ignoring request."
              << std::endl;
    return;
  }
  mServing = true;
  mServer = new WebsocketServer();

  // Register our network callbacks, ensuring the logic is run on the main
  // thread's event loop
  mServer->connect([&](ClientConnection conn) {
    mServerEventLoop.post([&, conn]() {
      std::clog << "Connection opened." << std::endl;
      std::clog << "There are now " << mServer->numConnections()
                << " open connections." << std::endl;
      // Send a hello message to the client
      // mServer->send(conn) seems to break, cause conn appears to get cleaned
      // up in race conditions (it's a weak pointer)
      std::stringstream json;
      json << "[";
      bool isFirst = true;
      for (auto pair : mBoxes)
      {
        if (isFirst)
          isFirst = false;
        else
          json << ",";
        encodeCreateBox(json, pair.second);
      }
      for (auto pair : mSpheres)
      {
        if (isFirst)
          isFirst = false;
        else
          json << ",";
        encodeCreateSphere(json, pair.second);
      }
      for (auto pair : mLines)
      {
        if (isFirst)
          isFirst = false;
        else
          json << ",";
        encodeCreateLine(json, pair.second);
      }
      for (auto name : mMouseInteractionEnabled)
      {
        if (isFirst)
          isFirst = false;
        else
          json << ",";
        encodeEnableMouseInteraction(json, name);
      }

      json << "]";

      std::cout << json.str() << std::endl;

      mServer->send(conn, json.str());
      // mServer->broadcast("{\"type\": 1}");
      /*
      mServer->broadcast(
          "{\"type\": \"init\", \"world\": " + mWorld->toJson() + "}");

      for (auto listener : mConnectionListeners)
      {
        listener();
      }
      */
    });
  });
  mServer->disconnect([&](ClientConnection /* conn */) {
    mServerEventLoop.post([&]() {
      std::clog << "Connection closed." << std::endl;
      std::clog << "There are now " << mServer->numConnections()
                << " open connections." << std::endl;
    });
  });
  mServer->message([&](ClientConnection /* conn */, const Json::Value& args) {
    mServerEventLoop.post([args, this]() {
      if (args["type"].asString() == "keydown")
      {
        std::string key = args["key"].asString();
        this->mKeysDown.insert(key);
        for (auto listener : this->mKeydownListeners)
        {
          listener(key);
        }
      }
      else if (args["type"].asString() == "keyup")
      {
        std::string key = args["key"].asString();
        this->mKeysDown.erase(key);
        for (auto listener : this->mKeyupListeners)
        {
          listener(key);
        }
      }
      else if (args["type"].asString() == "drag")
      {
        std::string name = args["name"].asString();
        Eigen::Vector3d pos = Eigen::Vector3d(
            args["pos"][0].asDouble(),
            args["pos"][1].asDouble(),
            args["pos"][2].asDouble());

        for (auto handler : mDragListeners[name])
        {
          handler(pos);
        }
      }
    });
  });

  // Start the networking thread
  mServerThread = new std::thread([&]() {
    // block signals in this thread and subsequently
    // spawned threads so they're guaranteed to go to the main thread
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

    mServer->run(port);
  });

  // Start the event loop for the main thread
  mWork = new asio::io_service::work(mServerEventLoop);

  // unblock signals in this thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_UNBLOCK, &sigset, nullptr);

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
    }
  });

  // Start a thread to
  mAsioThread = new std::thread([&] { mServerEventLoop.run(); });
}

/// This kills the server, if one was running
void GUIWebsocketServer::stopServing()
{
  if (!mServing)
    return;
  delete mWork;
  mServer->stop();
  mServerThread->join();
  delete mServer;
  delete mServerThread;
  mServing = false;
}

/// Returns true if we're serving
bool GUIWebsocketServer::isServing()
{
  return mServing;
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

/// This tells us whether or not to automatically flush after each command
void GUIWebsocketServer::setAutoflush(bool autoflush)
{
  mAutoflush = autoflush;
}

/// This sends the current list of commands to the web GUI
void GUIWebsocketServer::flush()
{
  mJson << "]";
  std::string json = mJson.str();
  std::cout << json << std::endl;
  if (mServing)
  {
    mServerEventLoop.post([json, this]() { mServer->broadcast(json); });
  }

  // Reset
  mMessagesQueued = 0;
  mJson.str(std::string());
  mJson.clear();
  mJson << "[";
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
GUIWebsocketServer& GUIWebsocketServer::renderWorld(
    std::shared_ptr<simulation::World> world)
{
  // TODO
  return *this;
}

/// This is a high-level command that renders a given trajectory as a bunch of
/// lines in the world, one per body
GUIWebsocketServer& GUIWebsocketServer::renderTrajectory(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::TrajectoryRollout> rollout)
{
  // TODO
  return *this;
}

/// This completely resets the web GUI, deleting all objects, UI elements, and
/// listeners
GUIWebsocketServer& GUIWebsocketServer::clear()
{
  // TODO
  return *this;
}

/// This creates a box in the web GUI under a specified name
GUIWebsocketServer& GUIWebsocketServer::createBox(
    const std::string& name,
    const Eigen::Vector3d& size,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& color)
{
  Box box;
  box.name = name;
  box.size = size;
  box.pos = pos;
  box.euler = euler;
  box.color = color;
  mBoxes[name] = box;

  queueCommand([&](std::stringstream& json) { encodeCreateBox(json, box); });

  return *this;
}

/// This creates a sphere in the web GUI under a specified name
GUIWebsocketServer& GUIWebsocketServer::createSphere(
    const std::string& name,
    double radius,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& color)
{
  Sphere sphere;
  sphere.name = name;
  sphere.radius = radius;
  sphere.pos = pos;
  sphere.color = color;
  mSpheres[name] = sphere;

  queueCommand(
      [&](std::stringstream& json) { encodeCreateSphere(json, sphere); });

  return *this;
}

/// This creates a line in the web GUI under a specified name
GUIWebsocketServer& GUIWebsocketServer::createLine(
    const std::string& name,
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Vector3d& color)
{
  Line line;
  line.name = name;
  line.points = points;
  line.color = color;
  mLines[name] = line;

  queueCommand([&](std::stringstream& json) { encodeCreateLine(json, line); });

  return *this;
}

/// This moves an object (e.g. box, sphere, line) to a specified position
GUIWebsocketServer& GUIWebsocketServer::setObjectPosition(
    const std::string& name, const Eigen::Vector3d& pos)
{
  if (mBoxes.find(name) != mBoxes.end())
  {
    mBoxes[name].pos = pos;
  }
  if (mSpheres.find(name) != mSpheres.end())
  {
    mSpheres[name].pos = pos;
  }

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_pos\", \"name\": \"" << name
         << "\", \"pos\": ";
    vec3ToJson(json, pos);
    json << "}";
  });

  return *this;
}

/// This moves an object (e.g. box, sphere, line) to a specified orientation
GUIWebsocketServer& GUIWebsocketServer::setObjectRotation(
    const std::string& name, const Eigen::Vector3d& euler)
{
  if (mBoxes.find(name) != mBoxes.end())
  {
    mBoxes[name].euler = euler;
  }

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_rotation\", \"name\": " << name
         << "\", \"euler\": ";
    vec3ToJson(json, euler);
    json << "}";
  });

  return *this;
}

/// This changes an object (e.g. box, sphere, line) color
GUIWebsocketServer& GUIWebsocketServer::setObjectColor(
    const std::string& name, const Eigen::Vector3d& color)
{
  if (mBoxes.find(name) != mBoxes.end())
  {
    mBoxes[name].color = color;
  }
  if (mSpheres.find(name) != mSpheres.end())
  {
    mSpheres[name].color = color;
  }
  if (mLines.find(name) != mLines.end())
  {
    mLines[name].color = color;
  }

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_color\", \"name\": " << name
         << "\", \"color\": ";
    vec3ToJson(json, color);
    json << "}";
  });

  return *this;
}

/// This enables mouse events on an object (if they're not already), and calls
/// "listener" whenever the object is dragged with the desired drag
/// coordinates
GUIWebsocketServer& GUIWebsocketServer::registerDragListener(
    const std::string& name, std::function<void(Eigen::Vector3d)> listener)
{
  mMouseInteractionEnabled.emplace(name);
  queueCommand([&](std::stringstream& json) {
    encodeEnableMouseInteraction(json, name);
  });
  mDragListeners[name].push_back(listener);
  return *this;
}

/// This deletes an object by name
GUIWebsocketServer& GUIWebsocketServer::deleteObject(const std::string& name)
{
  mBoxes.erase(name);
  mSpheres.erase(name);
  mLines.erase(name);

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"delete_object\", \"name\": \"" << name << "\" }";
  });

  return *this;
}

/// This places some text on the screen at the specified coordinates
GUIWebsocketServer& GUIWebsocketServer::createText(
    const std::string& name,
    const std::string& contents,
    const Eigen::Vector2d& fromTopRight)
{
  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setTextContents(
    const std::string& name, const std::string& newContents)
{
  return *this;
}

/// This deletes a UI element by name
GUIWebsocketServer& GUIWebsocketServer::deleteUIElement(const std::string& name)
{
  return *this;
}

void GUIWebsocketServer::queueCommand(
    std::function<void(std::stringstream&)> writeCommand)
{
  if (mMessagesQueued > 0)
  {
    mJson << ",";
  }
  mMessagesQueued++;
  writeCommand(mJson);
  if (mAutoflush)
    flush();
}

void GUIWebsocketServer::encodeCreateBox(std::stringstream& json, Box& box)
{
  json << "{ \"type\": \"create_box\", \"name\": \"" << box.name
       << "\", \"size\": ";
  vec3ToJson(json, box.size);
  json << ", \"pos\": ";
  vec3ToJson(json, box.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, box.euler);
  json << ", \"color\": ";
  vec3ToJson(json, box.color);
  json << "}";
}

void GUIWebsocketServer::encodeCreateSphere(
    std::stringstream& json, Sphere& sphere)
{
  json << "{ \"type\": \"create_sphere\", \"name\": \"" << sphere.name
       << "\", \"radius\": " << sphere.radius;
  json << ", \"pos\": ";
  vec3ToJson(json, sphere.pos);
  json << ", \"color\": ";
  vec3ToJson(json, sphere.color);
  json << "}";
}

void GUIWebsocketServer::encodeCreateLine(std::stringstream& json, Line& line)
{
  json << "{ \"type\": \"create_line\", \"name\": \"" << line.name;
  json << "\", \"points\": [";
  bool firstPoint = true;
  for (Eigen::Vector3d& point : line.points)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, point);
  }
  json << "], \"color\": ";
  vec3ToJson(json, line.color);
  json << "}";
}

void GUIWebsocketServer::encodeEnableMouseInteraction(
    std::stringstream& json, const std::string& name)
{
  json << "{ \"type\": \"enable_mouse\", \"name\": \"" << name << "\" }";
}

} // namespace server
} // namespace dart