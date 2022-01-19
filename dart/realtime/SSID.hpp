#ifndef DART_REALTIME_SSID
#define DART_REALTIME_SSID

#include <memory>
#include <thread>
#include <mutex>

#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include "dart/math/MathTypes.hpp"
#include "dart/realtime/VectorLog.hpp"

namespace dart {
namespace simulation {
class World;
}

namespace trajectory {
class LossFn;
class Problem;
class Solution;
class Optimizer;
} // namespace trajectory

namespace realtime {

// SSID = System + State IDentification
class SSID
{
public:
  SSID(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<trajectory::LossFn> loss,
      int planningHistoryMillis,
      Eigen::VectorXs sensorDims,
      int steps,
      s_t scale);

  /// This updates the loss function that we're going to move in real time to
  /// minimize. This can happen quite frequently, for example if our loss
  /// function is to track a mouse pointer in a simulated environment, we may
  /// reset the loss function every time the mouse moves.
  void setLoss(std::shared_ptr<trajectory::LossFn> loss);

  /// This sets the optimizer that MPC will use. This will override the default
  /// optimizer. This should be called before start().
  void setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This sets slow optimizer
  void setSlowOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This returns the current optimizer that MPC is using
  std::shared_ptr<trajectory::Optimizer> getOptimizer();

  /// This returns the slow optimizer
  std::shared_ptr<trajectory::Optimizer> getSlowOptimizer();

  /// This sets the problem that MPC will use. This will override the default
  /// problem. This should be called before start().
  void setProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This set slow problem
  void setSlowProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This registers a function that can be used to estimate the initial state
  /// for the inference system from recent sensor history and the timestamp
  void setInitialPosEstimator(
      std::function<Eigen::VectorXs(Eigen::MatrixXs, long)>
          initialPosEstimator);

  void setInitialVelEstimator(
    std::function<Eigen::VectorXs(Eigen::MatrixXs, long)>
    initialVelEstimator);

  /// This returns the current problem definition that MPC is using
  std::shared_ptr<trajectory::Problem> getProblem();

  /// This returns slow problem
  std::shared_ptr<trajectory::Problem> getSlowProblem();

  /// This logs that the sensor output is a specific vector now
  void registerSensorsNow(Eigen::VectorXs sensors, int sensor_id);

  /// This logs that the controls are a specific vector now
  void registerControlsNow(Eigen::VectorXs sensors);

  /// This logs that the sensor output was a specific vector at a specific
  /// moment
  void registerSensors(long now, Eigen::VectorXs sensors, int sensor_id);

  /// This logs that our controls were this value at this time
  void registerControls(long now, Eigen::VectorXs controls);

  /// This determine the condition number of trajectory to a node
  s_t getTrajConditionNumberOfMassIndex(Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index);

  s_t getTrajConditionNumberOfCOMIndex(Eigen::MatrixXs poses, Eigen::MatrixXs vels, size_t index);

  s_t getTrajConditionNumberOfDampingIndex(Eigen::MatrixXs vels, size_t index);

  s_t getTrajConditionNumberOfSpringIndex(Eigen::MatrixXs poses, size_t index);

  /// This determine the condition number of trajectory to all id node
  Eigen::VectorXs getTrajConditionNumbers(Eigen::MatrixXs poses, Eigen::MatrixXs vels);

  /// This set the index of param
  void setSSIDMassIndex(Eigen::VectorXi indices);

  void setSSIDCOMIndex(Eigen::VectorXi indices);

  void setSSIDDampIndex(Eigen::VectorXi indices);

  void setSSIDSpringIndex(Eigen::VectorXi indices);

  /// This starts our main thread and begins running optimizations
  void start();

  /// This starts the secondary thread of slow ssid which uses long trajectory
  void startSlow();

  /// This stops our main thread, waits for it to finish, and then returns
  void stop();

  /// This stops the secondary thread of slow ssid which uses long trajectory
  void stopSlow();

  /// This runs inference to find mutable values, starting at `startTime`
  void runInference(long startTime);

  /// This runs slow inference
  void runSlowInference(long startTime);

  std::pair<Eigen::VectorXs, Eigen::MatrixXs> runPlotting(long startTime, s_t upper, s_t lower, int samples);

  Eigen::MatrixXs runPlotting2D(long startTime, Eigen::Vector3s upper, Eigen::Vector3s lower, int x_samples, int y_samples, size_t rest_dim);

  void saveCSVMatrix(std::string filename, Eigen::MatrixXs matrix);

  /// This registers a listener to get called when we finish replanning
  void registerInferListener(
      std::function<
          void(long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)>
          inferListener);

  // wrapper for locking the buffer
  void attachMutex(std::mutex &mutex_lock);
  void attachParamMutex(std::mutex &mutex_lock);

  void registerLock();
  void registerUnlock();

  void paramMutexLock();

  void paramMutexUnlock();

  void setBufferLength(int length);

  void setTemperature(Eigen::VectorXs temp);

  Eigen::VectorXs getTemperature();

  // This is a big question
  // We assume that the current solution is common among two thread
  // We need to compare mean of recent fast thread against slow thread
  // Also, for fast thread, the initial guess can be provided by slow thread
  bool detectChangeParams();

  void updateFastThreadBuffer(Eigen::VectorXs new_solution, Eigen::VectorXs new_weight);

  Eigen::VectorXs estimateSolution();

  Eigen::VectorXs estimateConfidence();

  Eigen::VectorXs computeConfidenceFromValue(Eigen::VectorXs value);

  void setThreshs(s_t param_change, s_t conf);

  void useConfidence();

  void useHeuristicWeight();

  void useSmoothing();

protected:
  /// This is the function for the optimization thread to run when we're live
  void optimizationThreadLoop();

  void slowOptimizationThreadLoop();

  bool mRunning;
  bool mRunningSlow;
  
  std::shared_ptr<simulation::World> mWorld;
  // The World for Slow SSID may be there are some thread safety problems
  std::shared_ptr<simulation::World> mWorldSlow;
  std::shared_ptr<trajectory::LossFn> mLoss;
  int mPlanningHistoryMillis;
  int mPlanningSteps;

  // For Slow thread
  int mPlanningHistoryMillisSlow;
  int mPlanningStepsSlow;

  s_t mScale;
  // For dimension of different  system parameters
  size_t mMassDim;
  size_t mDampingDim;
  size_t mSpringDim;
  
  // For fast thread
  int mPrev_Length = 5;
  std::vector<Eigen::VectorXs> mPrev_solutions;
  std::vector<Eigen::VectorXs> mPrev_values;
  Eigen::VectorXs mParam_Solution;
  Eigen::VectorXs mParam_Steady;
  s_t mParam_change_thresh = 0.05;
  s_t mConfidence_thresh = 0.5;
  Eigen::VectorXs mTemperature;
  size_t mRobotSkelIndex = 0;
  Eigen::VectorXi mSSIDMassNodeIndices;
  Eigen::VectorXi mSSIDCOMNodeIndices;
  Eigen::VectorXi mSSIDDampingJointIndices;
  Eigen::VectorXi mSSIDSpringJointIndices;
  Eigen::VectorXs mValue;
  Eigen::VectorXs mCumValue;

  // Some Flags for control indication
  bool mSteadySolutionFound;
  bool mParamChanged;
  bool mInitialize = true;
  bool mSlowInit = true;
  bool mUseConfidence = false;
  bool mUseHeuristicWeight = false;
  bool mUseSmoothing = false;

  // Vector Logs for control
  Eigen::VectorXs mSensorDims;
  std::vector<VectorLog> mSensorLogs;
  VectorLog mControlLog;

  std::shared_ptr<trajectory::Optimizer> mOptimizer;
  std::shared_ptr<trajectory::Problem> mProblem;
  std::shared_ptr<trajectory::Solution> mSolution;
  std::thread mOptimizationThread;

  std::shared_ptr<trajectory::Optimizer> mOptimizerSlow;
  std::shared_ptr<trajectory::Problem> mProblemSlow;
  std::shared_ptr<trajectory::Solution> mSolutionSlow;
  std::thread mOptimizationThreadSlow;
  std::mutex* mRegisterMutex;
  std::mutex* mParamMutex;
  bool mLockRegistered = false;
  bool mParamLockRegistered = false;
  Eigen::VectorXs mParameters; // TODO : Should involve both mass and damping

  // These are listeners that get called when we finish replanning
  std::vector<std::function<void(
      long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)> >
      mInferListeners;

  // These are listeners that get called when we finish the slow ssid thread
  std::vector<std::function<void(
    long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)>>
    mInferListenersSlow;

  // This is the function that estimates our initial state before launching
  // learning
  std::function<Eigen::VectorXs(Eigen::MatrixXs, long)> mInitialPosEstimator;
  std::function<Eigen::VectorXs(Eigen::MatrixXs, long)> mInitialVelEstimator;
};

} // namespace realtime
} // namespace dart

#endif