#include "dart/realtime/RealtimeWorld.hpp"

#include <chrono>

#include "dart/simulation/World.hpp"

namespace dart {
namespace realtime {

RealtimeWorld::RealtimeWorld(
    std::shared_ptr<simulation::World> world,
    std::function<Eigen::VectorXd()> getForces,
    std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
        recordState)
  : mWorld(world),
    mGetForces(getForces),
    mRecordState(recordState),
    mRunning(false),
    mIterCount(0)
{
}

void RealtimeWorld::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mMainThread = std::thread(&RealtimeWorld::mainLoop, this);
}

void RealtimeWorld::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mMainThread.join();
}

void RealtimeWorld::mainLoop()
{
  while (mRunning)
  {
    mIterCount++;

    int interval = (int)(mWorld->getTimeStep() * 1000);
    auto x = std::chrono::steady_clock::now()
             + std::chrono::milliseconds(interval);

    mWorld->setForces(mGetForces());

    mWorld->step();

    mRecordState(
        mWorld->getPositions(), mWorld->getVelocities(), mWorld->getMasses());

    std::this_thread::sleep_until(x);
  }
}

} // namespace realtime
} // namespace dart