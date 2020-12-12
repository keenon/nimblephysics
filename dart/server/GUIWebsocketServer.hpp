#ifndef DART_GUI_SERVER
#define DART_GUI_SERVER

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <asio/io_service.hpp>

#include "dart/server/WebsocketServer.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {
class TrajectoryRollout;
}

namespace server {

class GUIWebsocketServer
{
public:
  GUIWebsocketServer();

  /// This is a non-blocking call to start a websocket server on a given port
  void serve(int port);

  /// This kills the server, if one was running
  void stopServing();

  /// Returns true if we're serving
  bool isServing();

  /// This adds a listener that will get called when someone connects to the
  /// server
  void registerConnectionListener(std::function<void()> listener);

  /// This adds a listener that will get called when ctrl+C is pressed
  void registerShutdownListener(std::function<void()> listener);

  /// This adds a listener that will get called when there is a key-down event
  /// on the web client
  void registerKeydownListener(std::function<void(std::string)> listener);

  /// This adds a listener that will get called when there is a key-up event
  /// on the web client
  void registerKeyupListener(std::function<void(std::string)> listener);

  /// This tells us whether or not to automatically flush after each command
  void setAutoflush(bool autoflush);

  /// This sends the current list of commands to the web GUI
  void flush();

  /// This is a high-level command that creates/updates all the shapes in a
  /// world by calling the lower-level commands
  GUIWebsocketServer& renderWorld(std::shared_ptr<simulation::World> world);

  /// This is a high-level command that renders a given trajectory as a bunch of
  /// lines in the world, one per body
  GUIWebsocketServer& renderTrajectory(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<trajectory::TrajectoryRollout> rollout);

  /// This completely resets the web GUI, deleting all objects, UI elements, and
  /// listeners
  GUIWebsocketServer& clear();

  /// This creates a box in the web GUI under a specified name
  GUIWebsocketServer& createBox(
      const std::string& name,
      const Eigen::Vector3d& size,
      const Eigen::Vector3d& pos,
      const Eigen::Vector3d& euler,
      const Eigen::Vector3d& color = Eigen::Vector3d::Zero());

  /// This creates a sphere in the web GUI under a specified name
  GUIWebsocketServer& createSphere(
      const std::string& name,
      double radius,
      const Eigen::Vector3d& pos,
      const Eigen::Vector3d& color = Eigen::Vector3d::Zero());

  /// This creates a line in the web GUI under a specified name
  GUIWebsocketServer& createLine(
      const std::string& name,
      const std::vector<Eigen::Vector3d>& points,
      const Eigen::Vector3d& color = Eigen::Vector3d::Zero());

  /// This moves an object (e.g. box, sphere, line) to a specified position
  GUIWebsocketServer& setObjectPosition(
      const std::string& name, const Eigen::Vector3d& pos);

  /// This moves an object (e.g. box, sphere, line) to a specified orientation
  GUIWebsocketServer& setObjectRotation(
      const std::string& name, const Eigen::Vector3d& euler);

  /// This changes an object (e.g. box, sphere, line) color
  GUIWebsocketServer& setObjectColor(
      const std::string& name, const Eigen::Vector3d& color);

  /// This enables mouse events on an object (if they're not already), and calls
  /// "listener" whenever the object is dragged with the desired drag
  /// coordinates
  GUIWebsocketServer& registerDragListener(
      const std::string& name, std::function<void(Eigen::Vector3d)> listener);

  /// This deletes an object by name
  GUIWebsocketServer& deleteObject(const std::string& name);

  /// This places some text on the screen at the specified coordinates
  GUIWebsocketServer& createText(
      const std::string& name,
      const std::string& contents,
      const Eigen::Vector2d& fromTopRight);

  /// This changes the contents of text on the screen
  GUIWebsocketServer& setTextContents(
      const std::string& name, const std::string& newContents);

  /// This deletes a UI element by name
  GUIWebsocketServer& deleteUIElement(const std::string& name);

protected:
  bool mServing;
  asio::io_service mServerEventLoop;
  asio::io_service::work* mWork;
  asio::signal_set* mSignalSet;
  std::thread* mAsioThread;
  std::thread* mServerThread;
  WebsocketServer* mServer;
  bool mAutoflush;

  // Listeners
  std::vector<std::function<void()>> mConnectionListeners;
  std::vector<std::function<void()>> mShutdownListeners;
  std::vector<std::function<void(std::string)>> mKeydownListeners;
  std::vector<std::function<void(std::string)>> mKeyupListeners;
  std::unordered_set<std::string> mKeysDown;
  std::unordered_map<
      std::string,
      std::vector<std::function<void(Eigen::Vector3d)>>>
      mDragListeners;
  // This is a list of all the objects with mouse interaction enabled
  std::unordered_set<std::string> mMouseInteractionEnabled;

  struct Box
  {
    std::string name;
    Eigen::Vector3d size;
    Eigen::Vector3d pos;
    Eigen::Vector3d euler;
    Eigen::Vector3d color;
  };
  std::unordered_map<std::string, Box> mBoxes;
  struct Sphere
  {
    std::string name;
    double radius;
    Eigen::Vector3d pos;
    Eigen::Vector3d color;
  };
  std::unordered_map<std::string, Sphere> mSpheres;
  struct Line
  {
    std::string name;
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d color;
  };
  std::unordered_map<std::string, Line> mLines;
  struct Text
  {
    std::string name;
    std::string contents;
    Eigen::Vector2d fromTopRight;
  };
  std::unordered_map<std::string, Line> mText;

  int mMessagesQueued;
  std::stringstream mJson;

  void queueCommand(std::function<void(std::stringstream&)> writeCommand);
  void encodeCreateBox(std::stringstream& json, Box& box);
  void encodeCreateSphere(std::stringstream& json, Sphere& sphere);
  void encodeCreateLine(std::stringstream& json, Line& line);
  void encodeEnableMouseInteraction(
      std::stringstream& json, const std::string& name);
};

} // namespace server
} // namespace dart

#endif