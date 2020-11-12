#ifndef DART_REALTIME_WORLD
#define DART_REALTIME_WORLD

#include <functional>
#include <memory>
#include <thread>
#include <unordered_set>

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

namespace realtime {

class RealtimeWorld
{
public:
  RealtimeWorld(
      std::shared_ptr<simulation::World> world,
      std::function<Eigen::VectorXd()> getForces,
      std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
          recordState);

  ~RealtimeWorld();

  void start();

  void stop();

  /// This runs a local websocket server that you can connect to to receive live
  /// updates
  void serve(int port);

  /// This kills the server, if one was running
  void stopServing();

  /// This adds a listener that will get called when someone connects to the
  /// server
  void registerConnectionListener(std::function<void()> listener);

  /// This adds a listener that will get called when there is a key-down event
  /// on the web client
  void registerKeydownListener(std::function<void(std::string)> listener);

  /// This adds a listener that will get called when there is a key-up event
  /// on the web client
  void registerKeyupListener(std::function<void(std::string)> listener);

  /// This adds a listener that will get called right before every step(), that
  /// can do whatever it likes to the world
  void registerPreStepListener(std::function<void(
                                   int,
                                   std::shared_ptr<simulation::World>,
                                   std::unordered_set<std::string>)> listener);

  /// This sends a plan out to any web clients that may be watching, so that
  /// they can render the MPC plan as it evolves.
  void displayMPCPlan(const trajectory::TrajectoryRollout* rollout);

protected:
  void mainLoop();

  int mIterCount;
  std::shared_ptr<simulation::World> mWorld;
  bool mRunning;
  std::function<Eigen::VectorXd()> mGetForces;
  std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
      mRecordState;
  std::thread* mMainThread;
  std::vector<std::function<void()>> mConnectionListeners;

  bool mServing;
  asio::io_service* mServerEventLoop;
  std::thread* mServerThread;
  WebsocketServer* mServer;
  asio::io_service::work* mServerWork;
  std::thread* mServerMainThread;

  // This is a scratch world copy to use when writing MPC plans to JSON
  std::shared_ptr<simulation::World> mMPCPlanWorld;

  std::vector<std::function<void(std::string)>> mKeydownListeners;
  std::vector<std::function<void(std::string)>> mKeyupListeners;
  std::unordered_set<std::string> mKeysDown;
  std::vector<std::function<void(
      int,
      std::shared_ptr<simulation::World>,
      std::unordered_set<std::string>)>>
      mPreStepListeners;
};

} // namespace realtime
} // namespace dart

#endif