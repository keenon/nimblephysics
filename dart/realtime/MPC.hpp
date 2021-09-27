#ifndef DART_MPC_INTERFACE
#define DART_MPC_INTERFACE

#include <functional>

#include <Eigen/Dense>

#include "dart/trajectory/TrajectoryRollout.hpp"

namespace dart {
namespace realtime {

class MPC
{
public:
  virtual ~MPC() = default;

  /// This gets the force to apply to the world at this instant. If we haven't
  /// computed anything for this instant yet, this just returns 0s.
  virtual Eigen::VectorXs getControlForce(long now) = 0;

  /// This calls getControlForce() with the current system clock as the time parameter
  virtual Eigen::VectorXs getControlForceNow();

  /// This returns how many millis we have left until we've run out of plan.
  /// This can be a negative number, if we've run past our plan.
  virtual long getRemainingPlanBufferMillis() = 0;

  /// This records the current state of the world based on some external sensing
  /// and inference. This resets the error in our model just assuming the world
  /// is exactly following our simulation.
  virtual void recordGroundTruthState(
      long time, Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass)
      = 0;

  /// This calls recordGroundTruthState() with the current system clock as the
  /// time parameter
  virtual void recordGroundTruthStateNow(
      Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass);

  /// This starts our main thread and begins running optimizations
  virtual void start() = 0;

  /// This stops our main thread, waits for it to finish, and then returns
  virtual void stop() = 0;

  /// This registers a listener to get called when we finish replanning
  virtual void registerReplanningListener(
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>
          replanListener)
      = 0;
};

} // namespace realtime
} // namespace dart

#endif