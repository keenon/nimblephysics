#ifndef DART_REALTIME_SSID
#define DART_REALTIME_SSID

#include <memory>
#include <thread>

#include <Eigen/Dense>

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
      int sensorDim);

  /// This updates the loss function that we're going to move in real time to
  /// minimize. This can happen quite frequently, for example if our loss
  /// function is to track a mouse pointer in a simulated environment, we may
  /// reset the loss function every time the mouse moves.
  void setLoss(std::shared_ptr<trajectory::LossFn> loss);

  /// This sets the optimizer that MPC will use. This will override the default
  /// optimizer. This should be called before start().
  void setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This returns the current optimizer that MPC is using
  std::shared_ptr<trajectory::Optimizer> getOptimizer();

  /// This sets the problem that MPC will use. This will override the default
  /// problem. This should be called before start().
  void setProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This registers a function that can be used to estimate the initial state
  /// for the inference system from recent sensor history and the timestamp
  void setInitialPosEstimator(
      std::function<Eigen::VectorXd(Eigen::MatrixXd, long)>
          initialPosEstimator);

  /// This returns the current problem definition that MPC is using
  std::shared_ptr<trajectory::Problem> getProblem();

  /// This logs that the sensor output is a specific vector now
  void registerSensorsNow(Eigen::VectorXd sensors);

  /// This logs that the controls are a specific vector now
  void registerControlsNow(Eigen::VectorXd sensors);

  /// This logs that the sensor output was a specific vector at a specific
  /// moment
  void registerSensors(long now, Eigen::VectorXd sensors);

  /// This logs that our controls were this value at this time
  void registerControls(long now, Eigen::VectorXd controls);

  /// This starts our main thread and begins running optimizations
  void start();

  /// This stops our main thread, waits for it to finish, and then returns
  void stop();

  /// This runs inference to find mutable values, starting at `startTime`
  void runInference(long startTime);

  /// This registers a listener to get called when we finish replanning
  void registerInferListener(
      std::function<
          void(long, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, long)>
          inferListener);

protected:
  /// This is the function for the optimization thread to run when we're live
  void optimizationThreadLoop();

  bool mRunning;
  std::shared_ptr<simulation::World> mWorld;
  std::shared_ptr<trajectory::LossFn> mLoss;
  std::shared_ptr<trajectory::Optimizer> mOptimizer;
  std::shared_ptr<trajectory::Problem> mProblem;
  std::shared_ptr<trajectory::Solution> mSolution;
  int mSensorDim;
  int mPlanningHistoryMillis;
  std::thread mOptimizationThread;

  VectorLog mSensorLog;
  VectorLog mControlLog;

  // These are listeners that get called when we finish replanning
  std::vector<std::function<void(
      long, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, long)> >
      mInferListeners;

  // This is the function that estimates our initial state before launching
  // learning
  std::function<Eigen::VectorXd(Eigen::MatrixXd, long)> mInitialPosEstimator;
};

} // namespace realtime
} // namespace dart

#endif