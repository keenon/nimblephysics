#ifndef DART_REALTIME_BUFFER
#define DART_REALTIME_BUFFER

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/realtime/ControlLog.hpp"
#include "dart/realtime/ObservationLog.hpp"

namespace dart {
namespace simulation {
class World;
}

namespace realtime {

enum BufferSwitchEnum
{
  UNINITIALIZED,
  BUF_A,
  BUF_B
};

class RealTimeControlBuffer
{
public:
  RealTimeControlBuffer(int forceDim, int steps, int millisPerStep);

  /// Gets the force at a given timestep. This HAS SIDE EFFECTS! We actually
  /// keep track of what forces were read, and assume that they're "immediately"
  /// applied to the real world after they're read.
  Eigen::VectorXd getPlannedForce(long time);

  /// This gets planned forces starting at `start`, and continuing for the
  /// length of our buffer size `mSteps`. This is useful for initializing MPC
  /// runs. It supports walking off the end of known future, and assumes 0
  /// forces in all extrapolation.
  void getPlannedForcesStartingAt(
      long start, Eigen::Ref<Eigen::MatrixXd> forcesOut);

  /// This swaps in a new buffer of forces. If "startAt" is after "now", this
  /// will copy enough of the current buffer into our updated buffer to keep the
  /// current trajectory.
  void setForcePlan(long startAt, long now, Eigen::MatrixXd forces);

  /// This retrieves the state of the world at a given time, assuming that we've
  /// been applying forces from the buffer since the last state that we fully
  /// observed.
  void estimateWorldStateAt(
      std::shared_ptr<simulation::World> world, ObservationLog* log, long time);

  /// This rescales the timestep size. This is useful because larger timesteps
  /// mean fewer time steps per real unit of time, and thus we can run our
  /// optimization slower and still keep up with real life.
  void setMillisPerStep(int millisPerStep);

  /// This changes the number of steps. Fewer steps mean we can compute a buffer
  /// faster, but it also means we have less time to compute the buffer. This
  /// probably has a nonlinear effect on runtime.
  void setNumSteps(int numSteps);

  /// This returns the number of millis we have left in the plan after `time`.
  /// This can be a negative number.
  long getPlanBufferMillisAfter(long time);

protected:
  int mForceDim;
  int mNumSteps;
  int mMillisPerStep;

  /// This is a helper to rescale the timestep size of a buffer while leaving
  /// the data otherwise unchanged.
  void rescaleBuffer(
      Eigen::MatrixXd& buf, int oldMillisPerStep, int newMillisPerStep);

  /// This controls which of our buffers is currently active
  BufferSwitchEnum mActiveBuffer;

  /// This is the A buffer of forces
  Eigen::MatrixXd mBufA;

  /// This is the B buffer of forces
  Eigen::MatrixXd mBufB;

  /// This is the time when the last buffer was written to
  long mLastWroteBufferAt;

  /// This keeps a log of all the control outputs we send, so that we can get
  /// the current state on request, even if we last had an observation a while
  /// ago.
  ControlLog mControlLog;
};

} // namespace realtime
} // namespace dart

#endif