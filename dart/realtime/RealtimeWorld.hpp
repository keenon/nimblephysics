#ifndef DART_REALTIME_WORLD
#define DART_REALTIME_WORLD

#include <functional>
#include <memory>
#include <thread>

#include <Eigen/Dense>

namespace dart {

namespace simulation {
class World;
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

  void start();

  void stop();

protected:
  void mainLoop();

  int mIterCount;
  std::shared_ptr<simulation::World> mWorld;
  bool mRunning;
  std::function<Eigen::VectorXd()> mGetForces;
  std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
      mRecordState;
  std::thread mMainThread;
};

} // namespace realtime
} // namespace dart

#endif