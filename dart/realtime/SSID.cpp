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
    mVerbose(false),
    mParamChanged(true),
    mSteadySolutionFound(false),
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
    mScale(scale),
    mMassDim(world->getMassDims()),
    mDampingDim(world->getDampingDims()),
    mSpringDim(world->getSpringDims()),
    mParam_Solution(Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim)),
    mValue(Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim)),
    mCumValue(Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim)),
    mTemperature(Eigen::VectorXs::Ones(mMassDim + mDampingDim + mSpringDim))
{
  for (int i = 0; i < mSensorDims.size(); i++)
  {
    mSensorLogs.push_back(VectorLog((double)mSensorDims(i)));
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
  ipoptOptimizer->setTolerance(1e-20);
  ipoptOptimizer->setIterationLimit(20);
  ipoptOptimizer->setRecordFullDebugInfo(false);
  ipoptOptimizer->setRecordIterations(false);
  ipoptOptimizer->setLBFGSHistoryLength(5);
  ipoptOptimizer->setSilenceOutput(true);
  mOptimizer = ipoptOptimizer;

  std::shared_ptr<IPOptOptimizer> ipoptOptimizerSlow
      = std::make_shared<IPOptOptimizer>();
  ipoptOptimizerSlow->setCheckDerivatives(false);
  ipoptOptimizerSlow->setSuppressOutput(true);
  ipoptOptimizerSlow->setTolerance(1e-15);
  ipoptOptimizerSlow->setIterationLimit(50);
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
void SSID::registerSensorsNow(Eigen::VectorXs sensors, int sensor_id)
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
  long startComputeWallTime = timeSinceEpochMillis();
  int millisPerStep
      = static_cast<int>(ceil(mScale * mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, false);
    // multishot->setParallelOperationsEnabled(true);
    mProblem = singleshot;
  }
  // std::cout<<"Problem Created"<<std::endl;
  //  Every turn, we need to pin all the forces
  registerLock();
  // Eigen::MatrixXs forceHistory = mControlLog.getValues(
  //     startTime - mPlanningHistoryMillis, steps, millisPerStep);
  Eigen::MatrixXs forceHistory
      = mControlLog.getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs forceHistory
      = mControlLog.getRecentValuesBefore(startTime, steps + 1);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }
  // std::cout<<"ForcePinned"<<std::endl;
  //  We also need to set all the sensor history into metadata

  // Eigen::MatrixXs poseHistory = mSensorLogs[0].getValues(
  //     startTime - mPlanningHistoryMillis, steps, millisPerStep);
  // Eigen::MatrixXs velHistory = mSensorLogs[1].getValues(
  //   startTime - mPlanningHistoryMillis, steps, millisPerStep
  //  );
  Eigen::MatrixXs poseHistory
      = mSensorLogs[0].getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs velHistory
      = mSensorLogs[1].getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs poseHistory
      = mSensorLogs[0].getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs velHistory
      = mSensorLogs[1].getRecentValuesBefore(startTime, steps + 1);
  registerUnlock();
  // std::cout<<"In SSID Force Hist: \n"<<forceHistory<<"\nPos Hist:
  // \n"<<poseHistory<<"\nVel Hist: \n"<<velHistory<<std::endl;
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities", velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));
  // Then actually run the optimization
  // std::cout<<"Ready to Optimize"<<std::endl;
  mSolution = mOptimizer->optimize(mProblem.get());
  // std::cout<<"Optimization End"<<std::endl;

  long computeDurationWallTime = timeSinceEpochMillis() - startComputeWallTime;

  const trajectory::TrajectoryRollout* cache
      = mProblem->getRolloutCache(mWorld);

  Eigen::VectorXs pos = cache->getPosesConst().col(steps - 1);
  Eigen::VectorXs vel = cache->getVelsConst().col(steps - 1);
  // Here the masses should be the concatenation of all registered mass nodes
  paramMutexLock();
  Eigen::VectorXs param = mParam_Solution;
  paramMutexUnlock();
  mValue = getTrajConditionNumbers(
      poseHistory, velHistory); // Should allow different value for each dofs
                                // should be identified independently

  for (auto listener : mInferListeners)
  {
    listener(
        startTime,
        pos,
        vel,
        param,
        computeDurationWallTime,
        mSteadySolutionFound);
  }
}

Eigen::VectorXs SSID::runPlotting(
    long startTime, s_t upper, s_t lower, int samples)
    // Run plotting function should be idenpotent
    // Here it only support 1 body node plotting, but that make sense
    std::pair<Eigen::VectorXs, Eigen::MatrixXs> SSID::runPlotting(
        long startTime, s_t upper, s_t lower, int samples)
{
  int millisPerStep
      = static_cast<int>(ceil(mScale * mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, false);
    mProblem = singleshot;
  }

  registerLock();
  Eigen::MatrixXs forceHistory
      = mControlLog.getRecentValuesBefore(startTime, steps + 1);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }

  Eigen::MatrixXs poseHistory
      = mSensorLogs[0].getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs velHistory
      = mSensorLogs[1].getRecentValuesBefore(startTime, steps + 1);

  registerUnlock();
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities", velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));

  Eigen::VectorXs losses;
  if (upper != lower)
  {
    losses = Eigen::VectorXs::Zero(samples);
    s_t epsilon = (upper - lower) / samples;
    s_t probe = lower;
    for (int i = 0; i < samples; i++)
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

  std::pair<Eigen::VectorXs, Eigen::MatrixXs> result;
  result.first = losses;
  result.second = poseHistory;
  return result;
}

Eigen::MatrixXs SSID::runPlotting2D(
    long startTime,
    Eigen::Vector3s upper,
    Eigen::Vector3s lower,
    int x_samples,
    int y_samples,
    size_t rest_dim)
{
  int millisPerStep
      = static_cast<int>(ceil(mScale * mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<SingleShot> singleshot
        = std::make_shared<SingleShot>(mWorld, *mLoss.get(), steps, false);
    mProblem = singleshot;
  }

  registerLock();
  Eigen::MatrixXs forceHistory
      = mControlLog.getRecentValuesBefore(startTime, steps + 1);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }

  Eigen::MatrixXs poseHistory
      = mSensorLogs[0].getRecentValuesBefore(startTime, steps + 1);
  Eigen::MatrixXs velHistory
      = mSensorLogs[1].getRecentValuesBefore(startTime, steps + 1);

  registerUnlock();
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", poseHistory);
  mProblem->setMetadata("velocities", velHistory);
  mProblem->setStartPos(mInitialPosEstimator(poseHistory, startTime));
  mProblem->setStartVel(mInitialVelEstimator(velHistory, startTime));

  Eigen::MatrixXs losses = Eigen::MatrixXs::Zero(x_samples, y_samples);

  Eigen::Vector3s probe = lower;
  assert(rest_dim < 3);
  size_t probe_dim_1;
  size_t probe_dim_2;
  if (rest_dim == 0)
  {
    probe_dim_1 = 1;
    probe_dim_2 = 2;
  }
  else if (rest_dim == 1)
  {
    probe_dim_1 = 0;
    probe_dim_2 = 2;
  }
  else
  {
    probe_dim_1 = 0;
    probe_dim_2 = 1;
  }
  assert(
      lower(probe_dim_1) < upper(probe_dim_1)
      && lower(probe_dim_2) < upper(probe_dim_2));
  s_t x_epsilon = (upper(probe_dim_1) - lower(probe_dim_1)) / x_samples;
  s_t y_epsilon = (upper(probe_dim_2) - lower(probe_dim_2)) / y_samples;

  for (int x_i = 0; x_i < x_samples; x_i++)
  {
    probe(probe_dim_2) = lower(probe_dim_2);
    for (int y_i = 0; y_i < y_samples; y_i++)
    {
      mWorld->setMasses(probe);
      mProblem->resetDirty();
      losses(x_i, y_i) = mProblem->getLoss(mWorld);
      probe(probe_dim_2) += y_epsilon;
    }
    probe(probe_dim_1) += x_epsilon;
  }
  return losses;
}

void SSID::saveCSVMatrix(std::string filename, Eigen::MatrixXs matrix)
{
  const static Eigen::IOFormat CSVFormat(
      Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(filename);
  if (file.is_open())
  {
    file << matrix.format(CSVFormat);
    file.close();
  }
}

/// This registers a listener to get called when we finish replanning
void SSID::registerInferListener(
    std::function<void(
        long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long, bool)>
        inferListener)
{
  mInferListeners.push_back(inferListener);
}

void SSID::updateFastThreadBuffer(
    Eigen::VectorXs new_solution, Eigen::VectorXs new_value)
{
  mPrev_solutions.push_back(new_solution);
  mPrev_values.push_back(new_value);
  if (mPrev_solutions.size() > mPrev_Length)
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
  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    if (mControlLog.availableStepsBefore(startTime) > mPlanningSteps + 1)
    {

      paramMutexLock();
      if (mInitialize)
      {
        // NOTE: Parameter solution by design support different types of
        // parameters
        mParam_Solution.segment(0, mMassDim) = mWorld->getMasses();
        mParam_Solution.segment(mMassDim, mDampingDim) = mWorld->getDampings();
        mParam_Solution.segment(mMassDim + mDampingDim, mSpringDim)
            = mWorld->getSprings();
      }
      mWorld->setMasses(mParam_Solution.segment(0, mMassDim));
      mWorld->setDampings(mParam_Solution.segment(mMassDim, mDampingDim));
      mWorld->setSprings(
          mParam_Solution.segment(mMassDim + mDampingDim, mSpringDim));

      paramMutexUnlock();
      runInference(startTime);
      // Update mPrev_result buffer
      Eigen::VectorXs new_solution
          = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
      new_solution.segment(0, mMassDim) = mWorld->getMasses();
      new_solution.segment(mMassDim, mDampingDim) = mWorld->getDampings();
      new_solution.segment(mMassDim + mDampingDim, mSpringDim)
          = mWorld->getSprings();

      if (!mUseSmoothing)
      {
        mParam_Solution = new_solution;
        // std::cout << "New Solution: " << new_solution(0) << new_solution(1)
        // << std::endl;
        continue;
      }
      // We have a stable solution, then detect changes
      if (mSteadySolutionFound && mUseConfidence)
      {
        if (detectEWiseChangeParams().second)
        {

          std::cout << "+++++++++++++++" << std::endl;
          std::cout << "Change Detected" << std::endl;
          std::cout << "+++++++++++++++" << std::endl;

          mParamChanged = true;
          mSteadySolutionFound = false;
          paramMutexLock();
          // Flush all the solutions in the buffer
          if (mUseConfidence)
          {
            mControlLog.discardBefore(startTime);
            for (int i = 0; i < mSensorLogs.size(); i++)
            {
              mSensorLogs[i].discardBefore(startTime);
            }
            mPrev_solutions.clear();
            mPrev_values.clear();
            mCumValue
                = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
          }
          mInitialize = true;
          paramMutexUnlock();
        }
      }
      updateFastThreadBuffer(new_solution, mValue);
      // We don't have a stable solution, we hope to find one
      if (mParamChanged == true || mSteadySolutionFound == false)
      {
        // Estimate the result by weighted average
        paramMutexLock();
        if (mVerbose)
        {
          std::cout << "Param Changed Use Fast Result!"
                    << mPrev_solutions[mPrev_solutions.size() - 1](0) << " "
                    << mPrev_solutions[mPrev_solutions.size() - 1](1) << " "
                    << mPrev_solutions[mPrev_solutions.size() - 1](2)
                    << std::endl;
        }
        if (!mInitialize)
        {
          // Sliding weighted average
          Eigen::VectorXs conf = computeConfidenceFromValue(mValue);
          if (mVerbose || true) // TODO: remove the override
          {
            std::cout << "Confidence: \n" << conf << std::endl;
          }
          Eigen::VectorXs prev_solution = mParam_Solution;
          if (mUseConfidence)
          {
            if (conf.mean()
                > mConfidence_thresh) // Confidence score is not a sensitive
                                      // hyper-param, we can use one for all
                                      // parameters
            {
              mParam_Solution
                  = (mValue.cwiseProduct((mCumValue + mValue).cwiseInverse()))
                        .cwiseProduct(new_solution)
                    + (mCumValue.cwiseProduct(
                           (mCumValue + mValue).cwiseInverse()))
                          .cwiseProduct(mParam_Solution);
              mCumValue += mValue;
            }
            // TODO : Need to use different mParam_change_thresh
            // If a parameter is detected to change, the other parameter is in
            // its low confidence area. Then we should change the parameter that
            // detected to change with high confidence score while maintain the
            // same other parameters
            // if((mParam_Solution-prev_solution).cwiseAbs().maxCoeff() <
            // mParam_change_thresh && conf.mean() > mConfidence_thresh)
            if (detectEWiseStable(mParam_Solution, prev_solution, conf).second)
            {
              mSteadySolutionFound = true;
              mParamChanged = false;
              if (mVerbose)
              {
                std::cout << "=====================" << std::endl;
                std::cout << "Steady Solution Found" << std::endl;
                std::cout << "=====================" << std::endl;
              }
            }
          }

          if (!mUseConfidence)
          {
            mParam_Solution = estimateSolution();
          }
          // How to determine a stable solution has been found
          // May be if we reject the low confidence result we don't need steady
          // solution, in that case we need to define confidence of a particular
          // solution, need to rescale the confidence
        }
        else
        {
          mParam_Solution = new_solution;
        }
        paramMutexUnlock();
      }
      if (mInitialize)
      {
        mInitialize = false;
      }
    }
  }
}

void SSID::setVerbose(bool verbose)
{
  mVerbose = verbose;
}

/// The Change of Parameter should be handled independently for each degree of
/// freedom For simplicity it is a global function right now
bool SSID::detectChangeParams()
{
  Eigen::VectorXs mean_params = estimateSolution();
  Eigen::VectorXs confidence = estimateConfidence();
  if (mVerbose)
  {
    std::cout << "Confidence: " << confidence.mean() << std::endl;
  }
  paramMutexLock();
  if ((mean_params - mParam_Solution).cwiseAbs().maxCoeff()
          > mParam_change_thresh
      && (confidence.mean() > mConfidence_thresh || !mUseConfidence))
  {
    paramMutexUnlock();
    return true;
  }
  paramMutexUnlock();
  return false;
}

std::pair<std::vector<bool>, bool> SSID::detectEWiseChangeParams()
{
  Eigen::VectorXs mean_params = estimateSolution();
  Eigen::VectorXs confidence = estimateConfidence();
  if (mVerbose)
  {
    std::cout << "Confidences:\n" << confidence << std::endl;
  }
  paramMutexLock();
  Eigen::VectorXs params_diff = (mean_params - mParam_Solution).cwiseAbs();
  std::vector<bool> params_change_flag;
  bool param_changed = false;
  for (int i = 0; i < params_diff.size(); i++)
  {
    // NOTE: Since confience score is relatively not sensitive, we use one
    // threshold for all
    if (params_diff(i) > mEwise_params_change_thresh(i)
        && (confidence(i) > mConfidence_thresh || !mUseConfidence))
    {
      params_change_flag.push_back(true);
      param_changed = true;
    }
    else
    {
      params_change_flag.push_back(false);
    }
  }
  paramMutexUnlock();
  return std::pair<std::vector<bool>, bool>(params_change_flag, param_changed);
}

std::pair<std::vector<bool>, bool> SSID::detectEWiseStable(
    Eigen::VectorXs current,
    Eigen::VectorXs reference,
    Eigen::VectorXs confidence)
{
  Eigen::VectorXs diff = (current - reference).cwiseAbs();
  std::vector<bool> stable_flags;
  bool stable_flag = true;
  for (int i = 0; i < diff.size(); i++)
  {
    // NOTE: There should have one confidence threshld at approximately 0.5
    if (diff(i) > mEwise_params_change_thresh(i)
        || confidence(i) <= mConfidence_thresh)
    {
      stable_flags.push_back(false);
      stable_flag = false;
    }
    else
    {
      stable_flags.push_back(true);
    }
  }
  return std::pair<std::vector<bool>, bool>(stable_flags, stable_flag);
}

/// The condition number is with respect to a particular node
s_t SSID::getTrajConditionNumberOfMassIndex(
    Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index)
{
  size_t steps = poses.cols();
  s_t cond = 0;
  s_t dt = mWorld->getTimeStep();
  Eigen::VectorXs init_pose = mWorld->getPositions();
  Eigen::VectorXs init_vel = mWorld->getVelocities();
  for (int i = 1; i < steps; i++)
  {
    mWorld->setPositions(poses.col(i));
    mWorld->setVelocities(vels.col(i));
    Eigen::VectorXs acc = (vels.col(i) - vels.col(i - 1)) / dt;
    // Eigen::VectorXs acc = Eigen::VectorXs::Ones(vels.rows()) * dt / dt;
    Eigen::MatrixXs Ak
        = mWorld->getSkeleton(mRobotSkelIndex)->getLinkAkMatrixIndex(index);
    Eigen::MatrixXs Jvk
        = mWorld->getSkeleton(mRobotSkelIndex)->getLinkJvkMatrixIndex(index);
    cond += (Ak * acc + Jvk.transpose() * mWorld->getGravity()).norm();
  }
  mWorld->setPositions(init_pose);
  mWorld->setVelocities(init_vel);
  return cond / steps;
}

Eigen::Vector3s SSID::getTrajConditionNumberOfCOMIndex(
    Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index)
{
  size_t steps = poses.cols();
  Eigen::Vector3s cond = Eigen::Vector3s::Zero();
  s_t dt = mWorld->getTimeStep();
  Eigen::VectorXs init_state = mWorld->getState();
  mWorld->setPositions(poses.col(0));
  mWorld->setVelocities(vels.col(0));
  Eigen::MatrixXs hat_Jw
      = mWorld->getSkeleton(mRobotSkelIndex)->getLinkLocalJwkMatrixIndex(index);
  Eigen::Matrix3s R
      = mWorld->getSkeleton(mRobotSkelIndex)->getLinkRMatrixIndex(index);
  for (int i = 1; i < steps; i++)
  {
    mWorld->setPositions(poses.col(i));
    mWorld->setVelocities(vels.col(i));
    Eigen::VectorXs acc = (vels.col(i) - vels.col(i - 1)) / dt;
    Eigen::MatrixXs new_R
        = mWorld->getSkeleton(mRobotSkelIndex)->getLinkRMatrixIndex(index);
    Eigen::MatrixXs new_hat_Jw = mWorld->getSkeleton(mRobotSkelIndex)
                                     ->getLinkLocalJwkMatrixIndex(index);
    Eigen::MatrixXs dR = (new_R - R) / dt;
    Eigen::MatrixXs d_hat_Jw = (new_hat_Jw - hat_Jw) / dt;
    R = new_R;
    hat_Jw = new_hat_Jw;
    // This should be: 3 x 3 matrix
    Eigen::MatrixXs S = R * vector2skew(hat_Jw * acc)
                        + dR * vector2skew(hat_Jw * vels.col(i))
                        + R * vector2skew(d_hat_Jw * vels.col(i));

    // This should be: N x 3 matrix
    Eigen::MatrixXs G = hat_Jw.transpose()
                        * vector2skew(R.transpose() * mWorld->getGravity());
    // Eigen::VectorXs G =

    cond(0) += S.col(0).norm() + G.col(0).norm();
    cond(1) += S.col(1).norm() + G.col(1).norm();
    cond(2) += S.col(2).norm() + G.col(2).norm();
  }
  mWorld->setState(init_state); // Idempotent
  return cond / steps;
}

// Should implement condition number of trajectory
Eigen::Vector3s SSID::getTrajConditionNumberOfMOIIndex(
    Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index)
{
  // Whether is is well conditioned is determined by the average of three
  // diagonal terms Compute Diagonal Matrix:
  size_t steps = poses.cols();
  Eigen::Vector3s cond = Eigen::Vector3s::Zero();
  s_t dt = mWorld->getTimeStep();
  Eigen::VectorXs init_state = mWorld->getState();
  for (int i = 1; i < steps; i++)
  {
    mWorld->setPositions(poses.col(i));
    mWorld->setVelocities(vels.col(i));
    Eigen::VectorXs acc = (vels.col(i) - vels.col(i - 1)) / dt;
    Eigen::MatrixXs Jw
        = mWorld->getSkeleton(mRobotSkelIndex)->getLinkJwkMatrixIndex(index);
    Eigen::MatrixXs Jv
        = mWorld->getSkeleton(mRobotSkelIndex)->getLinkJvkMatrixIndex(index);
    Eigen::MatrixXs diag = (Jw * acc).asDiagonal();
    Eigen::MatrixXs Ck = Jw.transpose() * diag;
    cond(0) += Ck.col(0).norm();
    cond(1) += Ck.col(1).norm();
    cond(2) += Ck.col(2).norm();
  }
  mWorld->setState(init_state); // Idempotent
  return cond / steps;
}

/// Here index represent joint index
s_t SSID::getTrajConditionNumberOfDampingIndex(
    Eigen::MatrixXs vels, size_t index)
{
  size_t steps = vels.cols();
  s_t cond = 0;
  for (int i = 0; i < steps; i++)
  {
    cond += abs(vels(index, i));
  }
  return cond / steps;
}

s_t SSID::getTrajConditionNumberOfSpringIndex(
    Eigen::MatrixXs poses, size_t index)
{
  size_t steps = poses.cols();
  s_t cond = 0;
  for (int i = 0; i < steps; i++)
  {
    cond += abs(poses(index, i) - mWorld->getRestPositionIndex(index));
  }
  return cond / steps;
}

// NOTE: This function can adapt to elementwise different type of parameters,
// with fixed order
Eigen::VectorXs SSID::getTrajConditionNumbers(
    Eigen::MatrixXs poses, Eigen::MatrixXs vels)
{
  Eigen::VectorXs conds
      = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
  int cur = 0;
  for (int i = 0; i < mSSIDMassNodeIndices.size(); i++)
  {
    conds(cur) = getTrajConditionNumberOfMassIndex(
        poses, vels, mSSIDMassNodeIndices(i));
    cur += 1;
  }
  for (int i = 0; i < mSSIDCOMNodeIndices.size(); i++)
  {
    conds.segment(cur, 3)
        = getTrajConditionNumberOfCOMIndex(poses, vels, mSSIDCOMNodeIndices(i));
    cur += 3;
  }
  for (int i = 0; i < mSSIDMOINodeIndices.size(); i++)
  {
    conds.segment(cur, 3)
        = getTrajConditionNumberOfMOIIndex(poses, vels, mSSIDMOINodeIndices(i));
    cur += 3;
  }
  for (int i = 0; i < mSSIDDampingJointIndices.size(); i++)
  {
    conds(cur) = getTrajConditionNumberOfDampingIndex(
        vels, mSSIDDampingJointIndices(i));
    cur += 1;
  }
  for (int i = 0; i < mSSIDSpringJointIndices.size(); i++)
  {
    conds(cur) = getTrajConditionNumberOfSpringIndex(
        poses, mSSIDSpringJointIndices(i));
    cur += 1;
  }
  return conds;
}

void SSID::attachMutex(std::mutex& mutex_lock)
{
  mRegisterMutex = &mutex_lock;
  mLockRegistered = true;
}

void SSID::attachParamMutex(std::mutex& mutex_lock)
{
  mParamMutex = &mutex_lock;
  mParamLockRegistered = true;
}

// This will use previous solutions to detect the change
// NOTE: Since mPrev_values is evaluated element-wisely it should be fine
Eigen::VectorXs SSID::estimateSolution()
{
  Eigen::VectorXs solution
      = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
  Eigen::VectorXs totalValue
      = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
  Eigen::VectorXs uniformValue
      = Eigen::VectorXs::Ones(mMassDim + mDampingDim + mSpringDim);
  // Eigen::VectorXs totalConfidence = estimateConfidence();

  for (int i = 0; i < mPrev_values.size(); i++)
  {
    if (mUseHeuristicWeight)
    {
      solution += mPrev_solutions[i].cwiseProduct(mPrev_values[i]);
      totalValue += mPrev_values[i];
    }
    else
    {
      solution += mPrev_solutions[i].cwiseProduct(uniformValue);
      totalValue += uniformValue;
    }
  }
  solution = solution.cwiseProduct(totalValue.cwiseInverse());
  // if(totalConfidence.minCoeff() > mConfidence_thresh && mUseConfidence &&
  // findStable)
  // {
  //   mSteadySolutionFound = true;
  //   mParamChanged = false;
  //   std::cout << "Steady Solution Found!!" << std::endl;
  // }
  return solution;
}

// Here confidence should be not related to trajectory length since it is a
// relative value Should be group-wise
Eigen::VectorXs SSID::estimateConfidence()
{
  Eigen::VectorXs totalValue
      = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
  for (int i = 0; i < mPrev_values.size(); i++)
  {
    totalValue += mPrev_values[i];
  }
  return computeConfidenceFromValue(totalValue / mPrev_values.size());
}

// NOTE: This is already element-wise, need to customize on temperature side
Eigen::VectorXs SSID::computeConfidenceFromValue(Eigen::VectorXs value)
{
  Eigen::VectorXs confidence
      = Eigen::VectorXs::Zero(mMassDim + mDampingDim + mSpringDim);
  for (int i = 0; i < mMassDim + mDampingDim + mSpringDim; i++)
  {
    confidence(i) = tanh(value(i) / mTemperature(i));
  }
  return confidence;
}

void SSID::registerLock()
{
  if (mLockRegistered)
  {
    mRegisterMutex->lock();
  }
}

void SSID::registerUnlock()
{
  if (mLockRegistered)
  {
    mRegisterMutex->unlock();
  }
}

void SSID::paramMutexLock()
{
  if (mParamLockRegistered)
    mParamMutex->lock();
}

void SSID::paramMutexUnlock()
{
  if (mParamLockRegistered)
    mParamMutex->unlock();
}

void SSID::setBufferLength(int length)
{
  mPrev_Length = length;
}

void SSID::setSSIDMassIndex(Eigen::VectorXi indices)
{
  mSSIDMassNodeIndices = indices;
}

void SSID::setSSIDCOMIndex(Eigen::VectorXi indices)
{
  mSSIDCOMNodeIndices = indices;
}

void SSID::setSSIDMOIIndex(Eigen::VectorXi indices)
{
  mSSIDMOINodeIndices = indices;
}

void SSID::setSSIDDampIndex(Eigen::VectorXi indices)
{
  mSSIDDampingJointIndices = indices;
}

void SSID::setSSIDSpringIndex(Eigen::VectorXi indices)
{
  mSSIDSpringJointIndices = indices;
}

void SSID::setTemperature(Eigen::VectorXs temp)
{
  assert(temp.size() == mMassDim + mDampingDim + mSpringDim);
  mTemperature = temp;
}

Eigen::VectorXs SSID::getTemperature()
{
  return mTemperature;
}

void SSID::setThreshs(s_t param_change, s_t conf)
{
  mParam_change_thresh = param_change;
  mConfidence_thresh = conf;
}

void SSID::setEWiseThreshs(Eigen::VectorXs param_changes, s_t conf)
{
  mEwise_params_change_thresh = param_changes;
  mConfidence_thresh = conf;
}

void SSID::useConfidence()
{
  mUseConfidence = true;
}

void SSID::useHeuristicWeight()
{
  mUseHeuristicWeight = true;
}

void SSID::useSmoothing()
{
  mUseSmoothing = true;
}

Eigen::Matrix3s SSID::vector2skew(Eigen::Vector3s vec)
{
  Eigen::Matrix3s skew = Eigen::Matrix3s::Zero();
  skew(2, 1) = vec(0);
  skew(1, 2) = -vec(0);
  skew(2, 0) = -vec(1);
  skew(0, 2) = vec(1);
  skew(1, 0) = vec(2);
  skew(0, 1) = -vec(2);
  return skew;
}

} // namespace realtime
} // namespace dart