#include "dart/realtime/SSID.hpp"

#include <thread>

#include "dart/realtime/Millis.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "signal.h"

namespace dart {

using namespace trajectory;

namespace realtime {

SSID::SSID(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::LossFn> loss,
    int planningHistoryMillis,
    Eigen::VectorXs sensorDims,
    int steps)
  : mRunning(false),
    mWorld(world),
    mLoss(loss),
    mPlanningHistoryMillis(planningHistoryMillis),
    mSensorDims(sensorDims),
    mControlLog(VectorLog(world->getNumDofs())),
    mPlanningSteps(steps)
{
  for(int i=0;i<mSensorDims.size();i++)
  {
    mSensorLogs.push_back(VectorLog(mSensorDims(i)));
  }
  int dofs = world->getNumDofs();
  mInitialPosEstimator
      = [dofs](Eigen::MatrixXs /* sensors */, long /* time */) {
          return Eigen::VectorXs::Zero(dofs);
        };
  mInitialVelEstimator
      = [dofs](Eigen::MatrixXs /* sensors */, long /* time */) {
          return Eigen::VectorXs::Zero(dofs);
        };

  std::shared_ptr<IPOptOptimizer> ipoptOptimizer
      = std::make_shared<IPOptOptimizer>();
  ipoptOptimizer->setCheckDerivatives(false);
  ipoptOptimizer->setSuppressOutput(true);
  ipoptOptimizer->setTolerance(1e-30);
  ipoptOptimizer->setIterationLimit(100);
  ipoptOptimizer->setRecordFullDebugInfo(false);
  ipoptOptimizer->setRecordIterations(false);
  ipoptOptimizer->setLBFGSHistoryLength(5);
  ipoptOptimizer->setSilenceOutput(true);
  mOptimizer = ipoptOptimizer;
}

/// This updates the loss function that we're going to move in real time to
/// minimize. This can happen quite frequently, for example if our loss
/// function is to track a mouse pointer in a simulated environment, we may
/// reset the loss function every time the mouse moves.
void SSID::setLoss(std::shared_ptr<trajectory::LossFn> loss)
{
  mLoss = loss;
}

/// This sets the optimizer that MPC will use. This will override the default
/// optimizer. This should be called before start().
void SSID::setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer)
{
  mOptimizer = optimizer;
}

/// This returns the current optimizer that MPC is using
std::shared_ptr<trajectory::Optimizer> SSID::getOptimizer()
{
  return mOptimizer;
}

/// This sets the problem that MPC will use. This will override the default
/// problem. This should be called before start().
void SSID::setProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblem = problem;
}

/// This registers a function that can be used to estimate the initial state
/// for the inference system from recent sensor history and the timestamp
void SSID::setInitialPosEstimator(
    std::function<Eigen::VectorXs(Eigen::MatrixXs, long)> initialPosEstimator)
{
  mInitialPosEstimator = initialPosEstimator;
}

void SSID::setInitialVelEstimator(
    std::function<Eigen::VectorXs(Eigen::MatrixXs, long)> initialVelEstimator)
{
  mInitialVelEstimator = initialVelEstimator;
}

/// This returns the current problem definition that MPC is using
std::shared_ptr<trajectory::Problem> SSID::getProblem()
{
  return mProblem;
}

/// This logs that the sensor output is a specific vector now
void SSID::registerSensorsNow(Eigen::VectorXs sensors,int sensor_id)
{
  return registerSensors(timeSinceEpochMillis(), sensors, sensor_id);
}

/// This logs that the controls are a specific vector now
void SSID::registerControlsNow(Eigen::VectorXs controls)
{
  return registerControls(timeSinceEpochMillis(), controls);
}

/// This logs that the sensor output was a specific vector at a specific
/// moment
void SSID::registerSensors(long now, Eigen::VectorXs sensors, int sensor_id)
{
  mSensorLogs[sensor_id].record(now, sensors);
}

/// This logs that our controls were this value at this time
void SSID::registerControls(long now, Eigen::VectorXs controls)
{
  mControlLog.record(now, controls);
}

/// This starts our main thread and begins running optimizations
void SSID::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&SSID::optimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void SSID::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This runs inference to find mutable values, starting at `startTime`
void SSID::runInference(long startTime)
{
  // registerLock();
  long startComputeWallTime = timeSinceEpochMillis();

  int millisPerStep = static_cast<int>(ceil(mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, false);
    //multishot->setParallelOperationsEnabled(true);
    mProblem = singleshot;
  }
  //std::cout<<"Problem Created"<<std::endl;
  // Every turn, we need to pin all the forces
  registerLock();
  //Eigen::MatrixXs forceHistory = mControlLog.getValues(
  //    startTime - mPlanningHistoryMillis, steps, millisPerStep);
  Eigen::MatrixXs forceHistory = mControlLog.getRecentValuesBefore(
    startTime, steps+1);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }
  //std::cout<<"ForcePinned"<<std::endl;
  // We also need to set all the sensor history into metadata

  //Eigen::MatrixXs poseHistory = mSensorLogs[0].getValues(
  //    startTime - mPlanningHistoryMillis, steps, millisPerStep);
  //Eigen::MatrixXs velHistory = mSensorLogs[1].getValues(
  //  startTime - mPlanningHistoryMillis, steps, millisPerStep
  // );
  Eigen::MatrixXs poseHistory = mSensorLogs[0].getRecentValuesBefore(startTime,steps+1);
  Eigen::MatrixXs velHistory = mSensorLogs[1].getRecentValuesBefore(startTime,steps+1);
  registerUnlock();
  //std::cout<<"In SSID Force Hist: \n"<<forceHistory<<"\nPos Hist: \n"<<poseHistory<<"\nVel Hist: \n"<<velHistory<<std::endl;
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities",velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  // TODO: Set initial velocity
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));
  // Then actually run the optimization
  //std::cout<<"Ready to Optimize"<<std::endl;
  mSolution = mOptimizer->optimize(mProblem.get());
  //std::cout<<"Optimization End"<<std::endl;

  long computeDurationWallTime = timeSinceEpochMillis() - startComputeWallTime;

  const trajectory::TrajectoryRollout* cache
      = mProblem->getRolloutCache(mWorld);

  Eigen::VectorXs pos = cache->getPosesConst().col(steps - 1);
  Eigen::VectorXs vel = cache->getVelsConst().col(steps - 1);
  Eigen::VectorXs mass = mWorld->getMasses();

  for (auto listener : mInferListeners)
  {
    listener(startTime, pos, vel, mass, computeDurationWallTime);
  }
}

Eigen::VectorXs SSID::runPlotting(long startTime, s_t upper, s_t lower,int samples)
{
  int millisPerStep = static_cast<int>(ceil(mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, false);
    mProblem = singleshot;
  }
  
  registerLock();
  Eigen::MatrixXs forceHistory = mControlLog.getRecentValuesBefore(
    startTime, steps+1);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }
  
  Eigen::MatrixXs poseHistory = mSensorLogs[0].getRecentValuesBefore(startTime,steps+1);
  Eigen::MatrixXs velHistory = mSensorLogs[1].getRecentValuesBefore(startTime,steps+1);
  //std::cout<<"In SSID Force Hist: \n"<<forceHistory<<"\nPos Hist: \n"<<poseHistory<<"\nVel Hist: \n"<<velHistory<<std::endl;
  registerUnlock();
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities",velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));

  Eigen::VectorXs losses;
  if(upper != lower)
  {
    losses = Eigen::VectorXs::Zero(samples);
    s_t epsilon = (upper-lower)/samples;
    s_t probe = lower;
    for(int i=0;i<samples;i++)
    {
      mWorld->setMasses(Eigen::Vector1s(probe));
      mProblem->resetDirty();
      s_t loss = mProblem->getLoss(mWorld);
      losses(i) = loss;
      probe += epsilon;
    }
  }
  else
  {
    losses = Eigen::VectorXs::Zero(1);
    mWorld->setMasses(Eigen::Vector1s(lower));
    mProblem->resetDirty();
    s_t loss = mProblem->getLoss(mWorld);
    losses(0) = loss;
  }
  
  return losses;
}

/// This registers a listener to get called when we finish replanning
void SSID::registerInferListener(
    std::function<
        void(long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)>
        inferListener)
{
  mInferListeners.push_back(inferListener);
}

/// This is the function for the optimization thread to run when we're live
void SSID::optimizationThreadLoop()
{
  // block signals in this thread and subsequently
  // spawned threads, so they're guaranteed to go to the server thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);
  /*
  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    if (mControlLog.availableHistoryBefore(startTime) > mPlanningHistoryMillis)
    {
      runInference(startTime);
      std::cout<<"Gap Time:"<<startTime - init_time<<std::endl;
    }
  }
  */
 while(mRunning)
 {
   long startTime = timeSinceEpochMillis();
   if(mControlLog.availableStepsBefore(startTime)>mPlanningSteps+1)
   {
     runInference(startTime);
   }
 }
}

void SSID::attachMutex(std::mutex &mutex_lock)
{
  mRegisterMutex = &mutex_lock;
  mLockRegistered = true;
}

void SSID::registerLock()
{
  if(mLockRegistered)
  {
    mRegisterMutex->lock();
  }
}

void SSID::registerUnlock()
{
  if(mLockRegistered)
  {
    mRegisterMutex->unlock();
  }
}

} // namespace realtime
} // namespace dart