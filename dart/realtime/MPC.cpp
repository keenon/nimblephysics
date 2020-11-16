#include "dart/realtime/MPC.hpp"

#include "dart/performance/PerformanceLog.hpp"
#include "dart/realtime/Millis.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Solution.hpp"

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
    mShotLength(50),
    mRunning(false),
    mObservationLog(
        timeSinceEpochMillis(),
        world->getPositions(),
        world->getVelocities(),
        world->getMasses()),
    mLastOptimizedTime(0L),
    mSilent(false),
    mMaxIterations(5),
    mEnableLinesearch(true),
    mEnableOptimizationGuards(false),
    mRecordIterations(false),
    mMillisInAdvanceToPlan(0)
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
    mObservationLog(mpc.mObservationLog),
    mLastOptimizedTime(mpc.mLastOptimizedTime),
    mSilent(mpc.mSilent),
    mMaxIterations(mpc.mMaxIterations),
    mEnableLinesearch(mpc.mEnableLinesearch),
    mEnableOptimizationGuards(mpc.mEnableOptimizationGuards),
    mRecordIterations(mpc.mRecordIterations)
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

/// This returns how many millis we have left until we've run out of plan.
/// This can be a negative number, if we've run past our plan.
long MPC::getRemainingPlanBufferMillis()
{
  return mBuffer.getPlanBufferMillisAfter(timeSinceEpochMillis());
}

/// This can completely silence log output
void MPC::setSilent(bool silent)
{
  mSilent = silent;
}

/// This enables linesearch on the IPOPT sub-problems. Defaults to true. This
/// increases the stability of solutions, but can lead to spikes in solution
/// times.
void MPC::setEnableLineSearch(bool enabled)
{
  mEnableLinesearch = enabled;
}

/// This enables "guards" on the IPOPT sub-problems. Defaults to false. This
/// means that every IPOPT sub-problem always returns the best explored
/// trajectory, even if it subsequently explored other states. This increases
/// the stability of solutions, but can lead to getting stuck in local minima.
void MPC::setEnableOptimizationGuards(bool enabled)
{
  mEnableOptimizationGuards = enabled;
}

/// Defaults to false. This records every iteration of IPOPT in the log, so we
/// can debug it. This should only be used on MPC that's running for a short
/// time. Otherwise the log will grow without bound.
void MPC::setRecordIterations(bool enabled)
{
  mRecordIterations = enabled;
}

/// This gets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization.
int MPC::getMaxIterations()
{
  return mMaxIterations;
}

/// This sets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization. MPC reserves the right to change
/// this value during runtime depending on timing and performance values
/// observed during running.
void MPC::setMaxIterations(int maxIters)
{
  mMaxIterations = maxIters;
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
  // We don't allow time to go backwards, because that leads to all sorts of
  // issues. We can get called for a time before a time we already optimized
  // for, because of dilating buffers in front of our current time. If that
  // happens, just pretent like we were asked for the latest time we were
  // optimizing for.
  if (startTime < mLastOptimizedTime)
  {
    startTime = mLastOptimizedTime;
  }

  if (mSolution == nullptr)
  {
    PerformanceLog::initialize();
    PerformanceLog* log = PerformanceLog::startRoot("MPC loop");

    PerformanceLog* createOpt = log->startRun("Create IPOPT");

    IPOptOptimizer optimizer = IPOptOptimizer();
    optimizer.setCheckDerivatives(false);
    optimizer.setSuppressOutput(true);
    optimizer.setRecoverBest(mEnableOptimizationGuards);
    optimizer.setTolerance(1e-3);
    optimizer.setIterationLimit(mMaxIterations);
    optimizer.setDisableLinesearch(!mEnableLinesearch);
    optimizer.setRecordFullDebugInfo(false);
    optimizer.setRecordIterations(false);
    if (mSilent)
    {
      optimizer.setSilenceOutput(true);
    }

    createOpt->end();

    PerformanceLog* estimateState = log->startRun("Estimate State");
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    mBuffer.estimateWorldStateAt(worldClone, &mObservationLog, startTime);
    estimateState->end();

    PerformanceLog* optimizeTrack = log->startRun("Optimize");
    mShot = std::make_shared<MultiShot>(
        worldClone, *mLoss.get(), mSteps, mShotLength, false);
    mShot->setParallelOperationsEnabled(true);

    mSolution = optimizer.optimize(mShot.get());
    optimizeTrack->end();

    mLastOptimizedTime = startTime;

    mBuffer.setForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mShot->getRolloutCache(worldClone)->getForcesConst());

    log->end();

    std::cout << PerformanceLog::finalize()["MPC loop"]->prettyPrint()
              << std::endl;
  }
  else
  {
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();

    int diff = startTime - mLastOptimizedTime;
    int steps = floor((double)diff / mMillisPerStep);
    int roundedDiff = steps * mMillisPerStep;
    long roundedStartTime = mLastOptimizedTime + roundedDiff;
    long totalPlanTime = mSteps * mMillisPerStep;
    double percentage = (double)roundedDiff * 100.0 / totalPlanTime;

    if (!mSilent)
    {
      std::cout << "Advancing plan by " << roundedDiff << "ms = " << steps
                << " steps, " << (percentage) << "% of total " << totalPlanTime
                << "ms plan time" << std::endl;
    }

    long startComputeWallTime = timeSinceEpochMillis();

    mBuffer.estimateWorldStateAt(
        worldClone, &mObservationLog, roundedStartTime);

    mShot->advanceSteps(
        worldClone,
        worldClone->getPositions(),
        worldClone->getVelocities(),
        steps);

    mSolution->reoptimize();

    mBuffer.setForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mShot->getRolloutCache(worldClone)->getForcesConst());

    long computeDurationWallTime
        = timeSinceEpochMillis() - startComputeWallTime;

    // Call any listeners that might be waiting on us
    for (auto listener : mReplannedListeners)
    {
      listener(mShot->getRolloutCache(worldClone), computeDurationWallTime);
    }

    if (!mSilent)
    {
      double factorOfSafety = 0.5;
      std::cout << " -> We were allowed "
                << (int)floor(roundedDiff * factorOfSafety)
                << "ms to solve this problem (" << roundedDiff
                << "ms new planning * " << factorOfSafety
                << " factor of safety), and it took us "
                << computeDurationWallTime << "ms" << std::endl;
    }

    mLastOptimizedTime = roundedStartTime;
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
  // This ensures that we don't "optimize our way out of sync", by letting the
  // optimizer change forces that already happened by the time the optimization
  // finishes, leading to us getting out of sync. Better to make our plans start
  // into the future.
  mMillisInAdvanceToPlan = 1.2 * lastOptimizeTimeMillis;
  // Don't go more than 200ms into the future, cause then errors have a chance
  // to propagate
  if (mMillisInAdvanceToPlan > 200)
    mMillisInAdvanceToPlan = 200;

  /*
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
  */
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

/// This returns the main record we've been keeping of our optimization up to
/// this point
std::shared_ptr<trajectory::Solution> MPC::getCurrentSolution()
{
  return mSolution;
}

/// This registers a listener to get called when we finish replanning
void MPC::registerReplanningListener(
    std::function<void(const trajectory::TrajectoryRollout*, long)>
        replanListener)
{
  mReplannedListeners.push_back(replanListener);
}

/// This is the function for the optimization thread to run when we're live
void MPC::optimizationThreadLoop()
{
  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    optimizePlan(startTime + mMillisInAdvanceToPlan);
    long endTime = timeSinceEpochMillis();
    adjustPerformance(endTime - startTime);
  }
}

} // namespace realtime
} // namespace dart