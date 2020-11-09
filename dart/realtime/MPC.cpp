#include "dart/realtime/MPC.hpp"

#include "dart/performance/PerformanceLog.hpp"
#include "dart/realtime/Millis.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/OptimizationRecord.hpp"

namespace dart {

using namespace trajectory;

namespace realtime {

MPC::MPC(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::LossFn> loss,
    int planningHorizonMillis)
  : mWorld(world),
    mLoss(loss),
    mPlanningHorizonMillis(planningHorizonMillis),
    mMillisPerStep(1000 * world->getTimeStep()),
    mSteps((int)ceil((double)planningHorizonMillis / mMillisPerStep)),
    mBuffer(RealTimeControlBuffer(world->getNumDofs(), mSteps, mMillisPerStep)),
    mShotLength(ceil((double)mSteps / 16)),
    mRunning(false),
    mObservationLog(
        timeSinceEpochMillis(),
        world->getPositions(),
        world->getVelocities(),
        world->getMasses())
{
}

/// Copy constructor
MPC::MPC(const MPC& mpc)
  : mWorld(mpc.mWorld),
    mLoss(mpc.mLoss),
    mPlanningHorizonMillis(mpc.mPlanningHorizonMillis),
    mMillisPerStep(mpc.mMillisPerStep),
    mSteps(mpc.mSteps),
    mBuffer(mpc.mBuffer),
    mShotLength(mpc.mShotLength),
    mRunning(mpc.mRunning),
    mObservationLog(mpc.mObservationLog)
{
}

/// This updates the loss function that we're going to move in real time to
/// minimize. This can happen quite frequently, for example if our loss
/// function is to track a mouse pointer in a simulated environment, we may
/// reset the loss function every time the mouse moves.
void MPC::setLoss(std::shared_ptr<trajectory::LossFn> loss)
{
  mLoss = loss;
}

/// This gets the force to apply to the world at this instant. If we haven't
/// computed anything for this instant yet, this just returns 0s.
Eigen::VectorXd MPC::getForce(long now)
{
  return mBuffer.getPlannedForce(now);
}

/// This calls getForce() with the current system clock as the time parameter
Eigen::VectorXd MPC::getForceNow()
{
  return getForce(timeSinceEpochMillis());
}

/// This records the current state of the world based on some external sensing
/// and inference. This resets the error in our model just assuming the world
/// is exactly following our simulation.
void MPC::recordGroundTruthState(
    long time, Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass)
{
  mObservationLog.observe(time, pos, vel, mass);
}

/// This calls recordGroundTruthState() with the current system clock as the
/// time parameter
void MPC::recordGroundTruthStateNow(
    Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass)
{
  recordGroundTruthState(timeSinceEpochMillis(), pos, vel, mass);
}

/// This optimizes a block of the plan, starting at `startTime`
void MPC::optimizePlan(long startTime)
{
  if (mOptimizationRecord == nullptr)
  {
    PerformanceLog::initialize();
    PerformanceLog* log = PerformanceLog::startRoot("MPC loop");

    PerformanceLog* createOpt = log->startRun("Create IPOPT");

    IPOptOptimizer optimizer = IPOptOptimizer();
    optimizer.setCheckDerivatives(false);
    optimizer.setSuppressOutput(true);
    optimizer.setRecoverBest(false);
    optimizer.setIterationLimit(4);

    createOpt->end();

    PerformanceLog* estimateState = log->startRun("Estimate State");
    mObservationLog.discardBefore(startTime);
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    mBuffer.estimateWorldStateAt(worldClone, &mObservationLog, startTime);
    estimateState->end();

    PerformanceLog* optimizeTrack = log->startRun("Optimize");
    mShot = std::make_shared<MultiShot>(
        worldClone, *mLoss.get(), mSteps, mShotLength, false);
    Eigen::MatrixXd forces
        = Eigen::MatrixXd::Zero(worldClone->getNumDofs(), mSteps);
    mBuffer.getPlannedForcesStartingAt(startTime, forces);
    mShot->setForces(forces);
    mOptimizationRecord = optimizer.optimize(mShot.get());
    optimizeTrack->end();

    mBuffer.setForcePlan(
        startTime, mShot->getRolloutCache(worldClone)->getForcesConst());

    log->end();

    std::cout << PerformanceLog::finalize()["MPC loop"]->prettyPrint()
              << std::endl;
  }
  else
  {
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    mBuffer.estimateWorldStateAt(worldClone, &mObservationLog, startTime);
    Eigen::MatrixXd forces
        = Eigen::MatrixXd::Zero(worldClone->getNumDofs(), mSteps);
    mBuffer.getPlannedForcesStartingAt(startTime, forces);
    mShot->setForces(forces);
    mOptimizationRecord->reoptimize();
    mBuffer.setForcePlan(
        startTime, mShot->getRolloutCache(worldClone)->getForcesConst());
  }
}

/// This adjusts parameters to make sure we're keeping up with real time. We
/// can compute how many (ms / step) it takes us to optimize plans. Sometimes
/// we can decrease (ms / step) by increasing the length of the optimization
/// and increasing the parallelism. We can also change the step size in the
/// physics engine to produce less accurate results, but keep up with the
/// world in fewer steps.
void MPC::adjustPerformance(long lastOptimizeTimeMillis)
{
  double millisToComputeEachStep = (double)lastOptimizeTimeMillis / mSteps;
  // Our safety margin is 3x, we want to be at least 3 times as fast as real
  // time
  long desiredMillisPerStep = 3 * millisToComputeEachStep;

  // This means our simulation step is too small, and we risk overflowing our
  // buffer before optimization finishes
  if (desiredMillisPerStep > mMillisPerStep)
  {
    std::cout << "Detected we're going too slow! Increasing timestep size from "
              << mMillisPerStep << "ms -> " << desiredMillisPerStep << "ms"
              << std::endl;

    mBuffer.setMillisPerStep(mMillisPerStep);
    mMillisPerStep = desiredMillisPerStep;
  }
}

/// This starts our main thread and begins running optimizations
void MPC::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&MPC::optimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void MPC::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This is the function for the optimization thread to run when we're live
void MPC::optimizationThreadLoop()
{
  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    optimizePlan(startTime);
    long endTime = timeSinceEpochMillis();
    adjustPerformance(endTime - startTime);
  }
}

} // namespace realtime
} // namespace dart