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
    int steps,
    s_t scale)
  : mRunning(false),
    mRunningSlow(false),
    mParamChanged(true),
    mResultFromSlowIsReady(false),
    mWorld(world),
    // wrt mass should be safe
    mWorldSlow(world->clone()),
    mLoss(loss),
    mPlanningHistoryMillis(planningHistoryMillis),
    mPlanningHistoryMillisSlow(planningHistoryMillis * 20),
    mSensorDims(sensorDims),
    mControlLog(VectorLog(world->getNumDofs())),
    mPlanningSteps(steps),
    mPlanningStepsSlow(steps * 20),
    mScale(scale)
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

  std::shared_ptr<IPOptOptimizer> ipoptOptimizerSlow
    = std::make_shared<IPOptOptimizer>();
  ipoptOptimizerSlow->setCheckDerivatives(false);
  ipoptOptimizerSlow->setSuppressOutput(true);
  ipoptOptimizerSlow->setTolerance(1e-30);
  ipoptOptimizerSlow->setIterationLimit(200);
  ipoptOptimizerSlow->setRecordFullDebugInfo(false);
  ipoptOptimizerSlow->setRecordIterations(false);
  ipoptOptimizerSlow->setLBFGSHistoryLength(5);
  ipoptOptimizerSlow->setSilenceOutput(true);
  mOptimizerSlow = ipoptOptimizerSlow;
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

void SSID::setSlowOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer)
{
  mOptimizerSlow = optimizer;
}

/// This returns the current optimizer that MPC is using
std::shared_ptr<trajectory::Optimizer> SSID::getOptimizer()
{
  return mOptimizer;
}

std::shared_ptr<trajectory::Optimizer> SSID::getSlowOptimizer()
{
  return mOptimizerSlow;
}

/// This sets the problem that MPC will use. This will override the default
/// problem. This should be called before start().
void SSID::setProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblem = problem;
}

void SSID::setSlowProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblemSlow = problem;
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
  std::cout << "Start Fast Thread!" << std::endl;
  mOptimizationThread = std::thread(&SSID::optimizationThreadLoop, this);
}

void SSID::startSlow()
{
  if(mRunningSlow)
    return;
  mRunningSlow = true;
  std::cout << "Start Slow Thread" << std::endl;
  mOptimizationThreadSlow = std::thread(&SSID::slowOptimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void SSID::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

void SSID::stopSlow()
{
  if(!mRunningSlow)
    return;
  mRunningSlow = false;
  mOptimizationThreadSlow.join();

}

/// This runs inference to find mutable values, starting at `startTime`
void SSID::runInference(long startTime)
{
  long startComputeWallTime = timeSinceEpochMillis();
  int millisPerStep = static_cast<int>(ceil(mScale*mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, true);
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
  registerUnlock();
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities",velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  // TODO: Set initial velocity
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));
  // Then actually run the optimization
  mSolution = mOptimizer->optimize(mProblem.get());

  long computeDurationWallTime = timeSinceEpochMillis() - startComputeWallTime;

  const trajectory::TrajectoryRollout* cache
      = mProblem->getRolloutCache(mWorld);

  Eigen::VectorXs pos = cache->getPosesConst().col(steps - 1);
  Eigen::VectorXs vel = cache->getVelsConst().col(steps - 1);
  // Here the masses should be the concatenation of all registered mass nodes
  // Eigen::VectorXs mass = mWorld->getMasses();
  paramMutexLock();
  Eigen::VectorXs mass = mParam_Solution;
  paramMutexUnlock();
  mValue = getTrajConditionNumbers(poseHistory, velHistory);
  
  for (auto listener : mInferListeners)
  {
    listener(startTime, pos, vel, mass, computeDurationWallTime);
  }
}

void SSID::runSlowInference(long startTime)
{
  long startComputeWallTime = timeSinceEpochMillis();
  int millisPerStep = static_cast<int>(ceil(mScale*mWorldSlow->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillisSlow) / millisPerStep));

  if (!mProblemSlow)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorldSlow, *mLoss.get(), steps, true);
    mProblemSlow = singleshot;
    // Perhaps need multishot problem but let's see
  }
  registerLock();
  Eigen::MatrixXs forceHistory = mControlLog.getRecentValuesBefore(
    startTime, steps+1);
  for (int i = 0; i < steps; i++)
  {
    mProblemSlow->pinForce(i, forceHistory.col(i));
  }

  Eigen::MatrixXs poseHistory = mSensorLogs[0].getRecentValuesBefore(startTime,steps+1);
  Eigen::MatrixXs velHistory = mSensorLogs[1].getRecentValuesBefore(startTime,steps+1);
  registerUnlock();
  mProblemSlow->setMetadata("forces", forceHistory);
  mProblemSlow->setMetadata("sensors", poseHistory);
  mProblemSlow->setMetadata("velocities",velHistory);
  mProblemSlow->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  // TODO: Set initial velocity
  mProblemSlow->setStartVel(mInitialVelEstimator(velHistory, startTime));
  // Then actually run the optimization
  mSolutionSlow = mOptimizerSlow->optimize(mProblemSlow.get());

  long computeDurationWallTime = timeSinceEpochMillis() - startComputeWallTime;

  const trajectory::TrajectoryRollout* cache
      = mProblemSlow->getRolloutCache(mWorldSlow);

  Eigen::VectorXs pos = cache->getPosesConst().col(steps - 1);
  Eigen::VectorXs vel = cache->getVelsConst().col(steps - 1);
  // Here the masses should be the concatenation of all registered mass nodes
  Eigen::VectorXs mass = mWorldSlow->getMasses();

  // TODO: Listeners may be different
  for (auto listener : mInferListenersSlow)
  {
    listener(startTime, pos, vel, mass, computeDurationWallTime);
  }
}

// Run plotting function should be idenpotent
// Here it only support 1 body node plotting, but that make sense
std::pair<Eigen::VectorXs, Eigen::MatrixXs> SSID::runPlotting(long startTime, s_t upper, s_t lower,int samples)
{
  int millisPerStep = static_cast<int>(ceil(mScale*mWorld->getTimeStep() * 1000.0));
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

  std::pair<Eigen::VectorXs, Eigen::MatrixXs> result;
  result.first = losses;
  result.second = poseHistory;
  return result;
}


Eigen::MatrixXs SSID::runPlotting2D(long startTime, Eigen::Vector3s upper, Eigen::Vector3s lower, int x_samples,int y_samples, size_t rest_dim)
{
  int millisPerStep = static_cast<int>(ceil(mScale * mWorld->getTimeStep() * 1000.0));
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
  
  registerUnlock();
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities",velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));

  Eigen::MatrixXs losses = Eigen::MatrixXs::Zero(x_samples,y_samples);
  
  
  
  Eigen::Vector3s probe = lower;
  assert(rest_dim < 3);
  size_t probe_dim_1;
  size_t probe_dim_2;
  if(rest_dim==0)
  {
    probe_dim_1 = 1;
    probe_dim_2 = 2;
  }
  else if(rest_dim == 1)
  {
    probe_dim_1 = 0;
    probe_dim_2 = 2;
  }
  else
  {
    probe_dim_1 = 0;
    probe_dim_2 = 1;
  }
  assert(lower(probe_dim_1) < upper(probe_dim_1) && lower(probe_dim_2) < upper(probe_dim_2));
  s_t x_epsilon = (upper(probe_dim_1)-lower(probe_dim_1))/x_samples;
  s_t y_epsilon = (upper(probe_dim_2)-lower(probe_dim_2))/y_samples;

  for(int x_i=0;x_i<x_samples;x_i++)
  {
    probe(probe_dim_2) = lower(probe_dim_2);
    for(int y_i=0; y_i < y_samples; y_i++)
    {
      mWorld->setMasses(probe);
      mProblem->resetDirty();
      losses(x_i,y_i) = mProblem->getLoss(mWorld);
      probe(probe_dim_2) += y_epsilon;
    }
    probe(probe_dim_1) += x_epsilon;
  }
  return losses;
}

void SSID::saveCSVMatrix(std::string filename, Eigen::MatrixXs matrix)
{
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(filename);
  if (file.is_open())
  {
      file << matrix.format(CSVFormat);
      file.close();
  }
}

/// This registers a listener to get called when we finish replanning
void SSID::registerInferListener(
    std::function<
        void(long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)>
        inferListener)
{
  mInferListeners.push_back(inferListener);
}

void SSID::updateFastThreadBuffer(Eigen::VectorXs new_solution, Eigen::VectorXs new_value)
{
  mPrev_solutions.push_back(new_solution);
  mPrev_values.push_back(new_value);
  if(mPrev_solutions.size() > mPrev_Length)
  {
    mPrev_solutions.erase(mPrev_solutions.begin());
    mPrev_values.erase(mPrev_values.begin());
  }
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
  while(mRunning)
  {
    long startTime = timeSinceEpochMillis();
    if(mControlLog.availableStepsBefore(startTime)>mPlanningSteps+1)
    {
      
      paramMutexLock();
      if(mInitialize)
      {
        mParam_Solution = mWorld->getMasses();
      }
      mWorld->setMasses(mParam_Solution);
      paramMutexUnlock();
      runInference(startTime);
      // Update mPrev_result buffer
      updateFastThreadBuffer(mWorld->getMasses(), mValue);
      if(mResultFromSlowIsReady)
      {
        if(detectChangeParams())
        {
          mParamChanged = true;
          paramMutexLock();
          mParam_Solution = estimateSolution();
          paramMutexUnlock();
        }
      }
      // Update the mParam_Solution with current solution
      if(mParamChanged==true || mResultFromSlowIsReady==false)
      {
        // Estimate the result by weighted average
        paramMutexLock();
        std::cout << "Param Changed Use Fast Result!" << mPrev_solutions[mPrev_solutions.size()-1](0) 
                  <<" " << mPrev_solutions[mPrev_solutions.size()-1](1) << std::endl;
        Eigen::VectorXs conf = estimateConfidence();
        std::cout << "Confidence :" << conf(0) << " " << conf(1) << std::endl;
        if(!mInitialize)
        {
          
          mParam_Solution = estimateSolution();
          //std::cout <<"Weight: "<< mValue << std::endl;
        }
        else
        {
          mParam_Solution = mPrev_solutions[mPrev_solutions.size()-1];
        }
        paramMutexUnlock();
      }
      if(mInitialize)
      {
        mInitialize = false;
      }
    }
  }
}

void SSID::slowOptimizationThreadLoop()
{
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);
  while (mRunningSlow)
  {
    long startTime = timeSinceEpochMillis();
    // Need to be sensitive to change
    if(mParamChanged)
    {
      // std::cout << "Slow Thread is Running" << std::endl;
      registerLock();
      mResultFromSlowIsReady = false;
      mParamChanged = false;
      mControlLog.discardBefore(startTime);
      for(int i = 0; i < mSensorLogs.size(); i++)
      {
        mSensorLogs[i].discardBefore(startTime);
      }
      registerUnlock();
    }
    if(mControlLog.availableStepsBefore(startTime) > mPlanningStepsSlow+1 && !mResultFromSlowIsReady)
    {
      // TODO: Implement inference for slow thread
      // Which ideally need to solve a separate optimization problem with different settings
      // It may be need to receive some flags from other thread
      runSlowInference(startTime);
      // Assume runSlowInference Given long enough trajectory can definitely solve the trajectory
      paramMutexLock();
      mParam_Solution = mWorldSlow->getMasses();
      mParam_Slow = mWorldSlow->getMasses();
      std::cout << "Result From Slow Thread: "
                << mParam_Solution(0) << " " << mParam_Solution(1) << std::endl;
      paramMutexUnlock();
      mParamChanged = false;
      mResultFromSlowIsReady = true;
    }
    
  }
}

/// The Change of Parameter should be handled independently for each degree of freedom
bool SSID::detectChangeParams()
{
  Eigen::VectorXs mean_params = estimateSolution();
  Eigen::VectorXs confidence = estimateConfidence();
  paramMutexLock();
  if((mean_params - mParam_Solution).cwiseAbs().maxCoeff() > mParam_change_thresh &&
    confidence.minCoeff() > mConfidence_thresh)
  {
    paramMutexUnlock();
    return true;
  }
  paramMutexUnlock();
  return false;
}

/// The condition number is with respect to a particular node
s_t SSID::getTrajConditionNumberIndex(Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index)
{
  size_t steps = poses.cols();
  s_t cond = 0;
  s_t dt = mWorld->getTimeStep();
  Eigen::VectorXs init_pose = mWorld->getPositions();
  Eigen::VectorXs init_vel = mWorld->getVelocities();
  for(int i = 1; i < steps; i++)
  {
    mWorld->setPositions(poses.col(i));
    mWorld->setVelocities(vels.col(i));
    Eigen::VectorXs acc = (vels.col(i) - vels.col(i-1)) / dt;
    Eigen::MatrixXs Ak = mWorld->getSkeleton(mRobotSkelIndex)->getLinkAkMatrixIndex(index);
    cond += (Ak * acc).norm();
  }
  mWorld->setPositions(init_pose);
  mWorld->setVelocities(init_vel);
  return cond;
}

Eigen::VectorXs SSID::getTrajConditionNumbers(Eigen::MatrixXs poses, Eigen::MatrixXs vels)
{
  Eigen::VectorXs conds = Eigen::VectorXs::Zero(mSSIDNodeIndices.size());
  for(int i = 0; i < mSSIDNodeIndices.size(); i++)
  {
    conds(i) = getTrajConditionNumberIndex(poses, vels, mSSIDNodeIndices(i));
  }
  return conds;
}

void SSID::attachMutex(std::mutex &mutex_lock)
{
  mRegisterMutex = &mutex_lock;
  mLockRegistered = true;
}

void SSID::attachParamMutex(std::mutex &mutex_lock)
{
  mParamMutex = &mutex_lock;
  mParamLockRegistered = true;
}

Eigen::VectorXs SSID::estimateSolution()
{
  Eigen::VectorXs solution = Eigen::VectorXs::Zero(mParam_Solution.size());
  Eigen::VectorXs totalValue = Eigen::VectorXs::Zero(mParam_Solution.size());
  for(int i = 0; i < mPrev_values.size(); i++)
  {
    solution += mPrev_solutions[i].cwiseProduct(mPrev_values[i]);
    totalValue += mPrev_values[i];
  }
  solution = solution.cwiseProduct(totalValue.cwiseInverse());
  return solution;
}

Eigen::VectorXs SSID::estimateConfidence()
{
  Eigen::VectorXs totalValue = Eigen::VectorXs::Zero(mParam_Solution.size());
  for(int i = 0; i < mPrev_values.size(); i++)
  {
    totalValue += mPrev_values[i];
  }
  for(int j = 0; j < mParam_Solution.size(); j++)
  {
    totalValue(j) = tanh(totalValue(j)/mTemperature);
  }
  return totalValue;
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

void SSID::paramMutexLock()
{
  if(mParamLockRegistered)
    mParamMutex->lock();
}

void SSID::paramMutexUnlock()
{
  if(mParamLockRegistered)
    mParamMutex->unlock();
}

void SSID::setBufferLength(int length)
{
  mPrev_Length = length;
}

void SSID::setSSIDIndex(Eigen::VectorXi indices)
{
  mSSIDNodeIndices = indices;
}

void SSID::setTemperature(s_t temp)
{
  mTemperature = temp;
}

s_t SSID::getTemperature()
{
  return mTemperature;
}

void SSID::setThreshs(s_t param_change, s_t conf)
{
  mParam_change_thresh = param_change;
  mConfidence_thresh = conf;
}

} // namespace realtime
} // namespace dart