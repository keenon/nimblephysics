#ifndef DART_GUI_SERVER
#define DART_GUI_SERVER

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dart/include_eigen.hpp"
#include <asio/io_service.hpp>
#include <assimp/Importer.hpp>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/server/GUIStateMachine.hpp"
#include "dart/server/WebsocketServer.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
}

namespace trajectory {
class TrajectoryRollout;
}

namespace server {

class GUIWebsocketServer : public GUIStateMachine
{
public:
  GUIWebsocketServer();

  ~GUIWebsocketServer();

  /// This is a non-blocking call to start a websocket server on a given
  /// port
  void serve(int port);

  /// This kills the server, if one was running
  void stopServing();

  /// Returns true if we're serving
  bool isServing();

  /// This flushes at a fixed framerate, not too fast to overwhelm the web GUI
  void flushThread();

  /// This sleeps until we're done serving, without busy-waiting in a loop. It
  /// wakes up occassionally to call the `checkForSignals` callback, where you
  /// can throw an exception to shut down the program.
  void blockWhileServing(std::function<void()> checkForSignals = []() {});

  /// This adds a listener that will get called when someone connects to the
  /// server
  void registerConnectionListener(std::function<void()> listener);

  /// This adds a listener that will get called when ctrl+C is pressed
  void registerShutdownListener(std::function<void()> listener);

  /// This adds a listener that will get called when there is a key-down
  /// event on the web client
  void registerKeydownListener(std::function<void(std::string)> listener);

  /// This adds a listener that will get called when there is a key-up event
  /// on the web client
  void registerKeyupListener(std::function<void(std::string)> listener);

  /// Gets the set of all the keys currently being pressed
  const std::unordered_set<std::string>& getKeysDown() const;

  /// Returns true if a key is currently being pressed
  bool isKeyDown(const std::string& key) const;

  /// This sends the current list of commands to the web GUI
  void flush();

  /// This completely resets the web GUI, deleting all objects, UI elements, and
  /// listeners
  void clear() override;

  /// This enables mouse events on an object (if they're not already), and
  /// calls "listener" whenever the object is dragged with the desired drag
  /// coordinates
  GUIWebsocketServer& registerDragListener(
      const std::string& key, std::function<void(Eigen::Vector3s)> listener);

  /// This gets the current screen size
  Eigen::Vector2i getScreenSize();

  /// This registers a callback to get called whenever the screen size changes.
  void registerScreenResizeListener(
      std::function<void(Eigen::Vector2i)> listener);

protected:
  int mPort;
  bool mServing;
  bool mStartingServer;
  Eigen::Vector2i mScreenSize;
  asio::signal_set* mSignalSet;
  std::thread* mServerThread;
  std::thread* mFlushThread;
  WebsocketServer* mServer;
  std::mutex mServingMutex;
  std::condition_variable mServingConditionValue;

  // Listeners
  std::vector<std::function<void()>> mConnectionListeners;
  std::vector<std::function<void()>> mShutdownListeners;
  std::vector<std::function<void(std::string)>> mKeydownListeners;
  std::vector<std::function<void(std::string)>> mKeyupListeners;
  std::unordered_set<std::string> mKeysDown;
  std::unordered_map<
      std::string,
      std::vector<std::function<void(Eigen::Vector3s)>>>
      mDragListeners;
  std::vector<std::function<void(Eigen::Vector2i)>> mScreenResizeListeners;
  // This is a list of all the objects with mouse interaction enabled
  std::unordered_set<std::string> mMouseInteractionEnabled;
};

} // namespace server
} // namespace dart

#endif