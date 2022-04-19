#ifndef DART_REALTIME_BUFFER
#define DART_REALTIME_BUFFER

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"
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
  RealTimeControlBuffer(int forceDim, int steps, int millisPerStep, int stateDim=1);

  /// Gets the force at a given timestep. This HAS SIDE EFFECTS! We actually
  /// keep track of what forces were read, and assume that they're "immediately"
  /// applied to the real world after they're read.
  Eigen::VectorXs getPlannedForce(long time, bool dontLog = false);

  Eigen::VectorXs getPlannedk(long time, bool dontLog = false);

  Eigen::MatrixXs getPlannedK(long time, bool dontLog = false);

  Eigen::VectorXs getPlannedState(long time, bool dontLog = false);

  s_t getPlannedAlpha(long time, bool dontLog = false);

  /// This gets planned forces starting at `start`, and continuing for the
  /// length of our buffer size `mSteps`. This is useful for initializing MPC
  /// runs. It supports walking off the end of known future, and assumes 0
  /// forces in all extrapolation.
  void getPlannedForcesStartingAt(
      long start, Eigen::Ref<Eigen::MatrixXs> forcesOut);

  void getPlannedkStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXs> kOut);

  void getPlannedKStartingAt(
    long start, std::vector<Eigen::MatrixXs> &KOut);

  void getPlannedStateStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXs> stateOut);

  void getPlannedAlphaStartingAt(long start, Eigen::Ref<Eigen::VectorXs> alphaOut);
  
  size_t getRemainSteps(long start);

  /// This swaps in a new buffer of forces. If "startAt" is after "now", this
  /// will copy enough of the current buffer into our updated buffer to keep the
  /// current trajectory.
  void setControlForcePlan(long startAt, long now, Eigen::MatrixXs forces);

  void setControlLawPlan(long startAt, 
                         long now, 
                         std::vector<Eigen::VectorXs> ks,
                         std::vector<Eigen::MatrixXs> Ks, 
                         std::vector<Eigen::VectorXs> states,
                         std::vector<s_t> alphas);

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

  /// This is useful when we're replicating a log across a network boundary,
  /// which comes up in distributed MPC.
  void manuallyRecordObservedForce(long time, Eigen::VectorXs observation);

  /// For debug only
  long getLastWriteBufferTime();

  long getLastWriteBufferLawTime();

  void setiLQRFlag(bool ilqr_flag);

  void setActionBound(s_t bound);

protected:
  int mForceDim;
  int mStateDim;
  int mNumSteps;
  int mMillisPerStep;

  /// This is a helper to rescale the timestep size of a buffer while leaving
  /// the data otherwise unchanged.
  void rescaleBuffer(
      Eigen::MatrixXs& buf, int oldMillisPerStep, int newMillisPerStep);

  /// This controls which of our buffers is currently active
  BufferSwitchEnum mActiveBuffer;

  /// This controls which of our k K Buffers is currently active
  BufferSwitchEnum mActiveBufferLaw;

  /// This is the A buffer of forces
  Eigen::MatrixXs mBufA;

  /// This is the B buffer of forces
  Eigen::MatrixXs mBufB;

  // This is the A buffer of k, which is feed forward gain in iLQR
  Eigen::MatrixXs mkBufA;
  
  // This is the B buffer of k, which is feed forward gain in iLQR
  Eigen::MatrixXs mkBufB;

  // This is the A buffer of K, which is feedback gain in iLQR
  std::vector<Eigen::MatrixXs> mKBufA;

  // This is the B buffer of K, which is feedback gain in iLQR
  std::vector<Eigen::MatrixXs> mKBufB;

  // This is the A buffer of State

  Eigen::MatrixXs mxBufA;

  // This is the B buffer of state

  Eigen::MatrixXs mxBufB;

  // This is the A buffer of Alpha, which is line search learning rate in iLQR
  Eigen::VectorXs mAlphaBufA;

  // This is the B buffer of Alpha, , which is line search learning rate in iLQR
  Eigen::VectorXs mAlphaBufB;

  /// This is the time when the last buffer was written to
  long mLastWroteBufferAt;

  /// This is the time when the last buffer of control law was written to
  long mLastWroteLawBufferAt;

  /// This keeps a log of all the control outputs we send, so that we can get
  /// the current state on request, even if we last had an observation a while
  /// ago.
  ControlLog mControlLog;

  bool mUseiLQR = false;

  s_t mActionBound = 1000;
};

} // namespace realtime
} // namespace dart

#endif