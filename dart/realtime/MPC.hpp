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
class Solution;
class Problem;
class Optimizer;
class TrajectoryRollout;
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

  /// This sets the optimizer that MPC will use. This will override the default
  /// optimizer. This should be called before start().
  void setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This returns the current optimizer that MPC is using
  std::shared_ptr<trajectory::Optimizer> getOptimizer();

  /// This sets the problem that MPC will use. This will override the default
  /// problem. This should be called before start().
  void setProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This returns the current problem definition that MPC is using
  std::shared_ptr<trajectory::Problem> getProblem();

  /// This gets the force to apply to the world at this instant. If we haven't
  /// computed anything for this instant yet, this just returns 0s.
  Eigen::VectorXd getForce(long now);

  /// This calls getForce() with the current system clock as the time parameter
  Eigen::VectorXd getForceNow();

  /// This returns how many millis we have left until we've run out of plan.
  /// This can be a negative number, if we've run past our plan.
  long getRemainingPlanBufferMillis();

  /// This can completely silence log output
  void setSilent(bool silent);

  /// This enables linesearch on the IPOPT sub-problems. Defaults to true. This
  /// increases the stability of solutions, but can lead to spikes in solution
  /// times.
  void setEnableLineSearch(bool enabled);

  /// This enables "guards" on the IPOPT sub-problems. Defaults to false. This
  /// means that every IPOPT sub-problem always returns the best explored
  /// trajectory, even if it subsequently explored other states. This increases
  /// the stability of solutions, but can lead to getting stuck in local minima.
  void setEnableOptimizationGuards(bool enabled);

  /// Defaults to false. This records every iteration of IPOPT in the log, so we
  /// can debug it. This should only be used on MPC that's running for a short
  /// time. Otherwise the log will grow without bound.
  void setRecordIterations(bool enabled);

  /// This gets the current maximum number of iterations that IPOPT will be
  /// allowed to run during an optimization.
  int getMaxIterations();

  /// This sets the current maximum number of iterations that IPOPT will be
  /// allowed to run during an optimization. MPC reserves the right to change
  /// this value during runtime depending on timing and performance values
  /// observed during running.
  void setMaxIterations(int maxIters);

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

  /// This returns the main record we've been keeping of our optimization up to
  /// this point
  std::shared_ptr<trajectory::Solution> getCurrentSolution();

  /// This registers a listener to get called when we finish replanning
  void registerReplanningListener(
      std::function<void(const trajectory::TrajectoryRollout*, long)>
          replanListener);

protected:
  /// This is the function for the optimization thread to run when we're live
  void optimizationThreadLoop();

  bool mRunning;
  std::shared_ptr<simulation::World> mWorld;
  std::shared_ptr<trajectory::LossFn> mLoss;
  ObservationLog mObservationLog;

  // Meta config
  bool mEnableLinesearch;
  bool mEnableOptimizationGuards;
  bool mRecordIterations;

  int mPlanningHorizonMillis;
  int mMillisPerStep;
  int mSteps;
  int mShotLength;
  int mMaxIterations;
  int mMillisInAdvanceToPlan;
  long mLastOptimizedTime;
  RealTimeControlBuffer mBuffer;
  std::thread mOptimizationThread;
  bool mSilent;

  std::shared_ptr<trajectory::Optimizer> mOptimizer;
  std::shared_ptr<trajectory::Solution> mSolution;
  std::shared_ptr<trajectory::Problem> mProblem;

  // These are listeners that get called when we finish replanning
  std::vector<std::function<void(const trajectory::TrajectoryRollout*, long)>>
      mReplannedListeners;
};

} // namespace realtime
} // namespace dart

#endif