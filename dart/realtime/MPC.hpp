#ifndef DART_REALTIME_MPC
#define DART_REALTIME_MPC

#include <memory>
#include <thread>

#include <Eigen/Dense>

#include "dart/realtime/RealTimeControlBuffer.hpp"

namespace dart {
namespace simulation {
class World;
}

namespace trajectory {
class LossFn;
class OptimizationRecord;
class MultiShot;
} // namespace trajectory

namespace realtime {

class MPC
{
public:
  MPC(std::shared_ptr<simulation::World> world,
      std::shared_ptr<trajectory::LossFn> loss,
      int planningHorizonMillis);

  /// Copy constructor
  MPC(const MPC& mpc);

  /// This updates the loss function that we're going to move in real time to
  /// minimize. This can happen quite frequently, for example if our loss
  /// function is to track a mouse pointer in a simulated environment, we may
  /// reset the loss function every time the mouse moves.
  void setLoss(std::shared_ptr<trajectory::LossFn> loss);

  /// This gets the force to apply to the world at this instant. If we haven't
  /// computed anything for this instant yet, this just returns 0s.
  Eigen::VectorXd getForce(long now);

  /// This calls getForce() with the current system clock as the time parameter
  Eigen::VectorXd getForceNow();

  /// This records the current state of the world based on some external sensing
  /// and inference. This resets the error in our model just assuming the world
  /// is exactly following our simulation.
  void recordGroundTruthState(
      long time,
      Eigen::VectorXd pos,
      Eigen::VectorXd vel,
      Eigen::VectorXd mass);

  /// This calls recordGroundTruthState() with the current system clock as the
  /// time parameter
  void recordGroundTruthStateNow(
      Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass);

  /// This optimizes a block of the plan, starting at `startTime`
  void optimizePlan(long startTime);

  /// This adjusts parameters to make sure we're keeping up with real time. We
  /// can compute how many (ms / step) it takes us to optimize plans. Sometimes
  /// we can decrease (ms / step) by increasing the length of the optimization
  /// and increasing the parallelism. We can also change the step size in the
  /// physics engine to produce less accurate results, but keep up with the
  /// world in fewer steps.
  void adjustPerformance(long lastOptimizeTimeMillis);

  /// This starts our main thread and begins running optimizations
  void start();

  /// This stops our main thread, waits for it to finish, and then returns
  void stop();

protected:
  /// This is the function for the optimization thread to run when we're live
  void optimizationThreadLoop();

  bool mRunning;
  std::shared_ptr<simulation::World> mWorld;
  std::shared_ptr<trajectory::LossFn> mLoss;
  ObservationLog mObservationLog;
  int mPlanningHorizonMillis;
  int mMillisPerStep;
  int mSteps;
  int mShotLength;
  RealTimeControlBuffer mBuffer;
  std::thread mOptimizationThread;
  // This is saved info so that we can reoptimize rather than create a fresh
  // problem each time
  std::shared_ptr<trajectory::OptimizationRecord> mOptimizationRecord;
  std::shared_ptr<trajectory::MultiShot> mShot;
};

} // namespace realtime
} // namespace dart

#endif