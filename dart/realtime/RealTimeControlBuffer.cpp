#include "dart/realtime/RealTimeControlBuffer.hpp"

#include <iostream>

#include "dart/simulation/World.hpp"

namespace dart {
namespace realtime {

RealTimeControlBuffer::RealTimeControlBuffer(
    int forceDim, int steps, int millisPerStep)
  : mForceDim(forceDim),
    mNumSteps(steps),
    mMillisPerStep(millisPerStep),
    mActiveBuffer(UNINITIALIZED),
    mBufA(Eigen::MatrixXd::Zero(forceDim, steps)),
    mBufB(Eigen::MatrixXd::Zero(forceDim, steps)),
    mControlLog(ControlLog(forceDim, millisPerStep))
{
}

/// Gets the force at a given timestep
Eigen::VectorXd RealTimeControlBuffer::getPlannedForce(long time, bool dontLog)
{
  if (mActiveBuffer == UNINITIALIZED)
  {
    // Unitialized, default to no force
    return Eigen::VectorXd::Zero(mForceDim);
  }
  int elapsed = time - mLastWroteBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    return Eigen::VectorXd::Zero(mForceDim);
  }

  int step = (int)floor((double)elapsed / mMillisPerStep);
  if (step < mNumSteps)
  {
    if (mActiveBuffer == BUF_A)
    {
      if (!dontLog)
        mControlLog.record(time, mBufA.col(step));
      return mBufA.col(step);
    }
    else if (mActiveBuffer == BUF_B)
    {
      if (!dontLog)
        mControlLog.record(time, mBufB.col(step));
      return mBufB.col(step);
    }
    else
      assert(false && "Should never reach this point");
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    Eigen::VectorXd oob = Eigen::VectorXd::Zero(mForceDim);
    if (!dontLog)
      mControlLog.record(time, oob);
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}

/// This gets planned forces starting at `start`, and continuing for the
/// length of our buffer size `mSteps`. This is useful for initializing MPC
/// runs. It supports walking off the end of known future, and assumes 0
/// forces in all extrapolation.
void RealTimeControlBuffer::getPlannedForcesStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXd> forcesOut)
{
  if (mActiveBuffer == UNINITIALIZED)
  {
    // Unitialized, default to 0
    forcesOut.setZero();
    return;
  }
  int elapsed = start - mLastWroteBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to 0
    forcesOut.setZero();
    return;
  }
  int startStep = (int)floor((double)elapsed / mMillisPerStep);
  if (startStep < mNumSteps)
  {
    // Copy the appropriate block of our active buffer to the forcesOut block
    if (mActiveBuffer == BUF_A)
    {
      forcesOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mBufA.block(0, startStep, mForceDim, mNumSteps - startStep);
    }
    else if (mActiveBuffer == BUF_B)
    {
      forcesOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mBufB.block(0, startStep, mForceDim, mNumSteps - startStep);
    }
    else
      assert(false && "Should never reach this point");
    // Zero out the remainder of the forcesOut block
    forcesOut.block(0, mNumSteps - startStep, mForceDim, startStep).setZero();
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    forcesOut.setZero();
  }
}

/// This swaps in a new buffer of forces. The assumption is that "startAt" is
/// before "now", because we'll erase old data in this process.
void RealTimeControlBuffer::setForcePlan(
    long startAt, long now, Eigen::MatrixXd forces)
{
  if (startAt > now)
  {
    long padMillis = startAt - now;
    int padSteps = (int)floor((double)padMillis / mMillisPerStep);
    // If we're trying to set the force plan too far out in the future, this
    // whole exercise is a no-op
    if (padSteps >= mNumSteps)
    {
      return;
    }
    // Otherwise, we're going to copy part of the existing plan
    int currentStep
        = (int)floor((double)(now - mLastWroteBufferAt) / mMillisPerStep);
    int remainingSteps = mNumSteps - currentStep;
    mLastWroteBufferAt = now;

    // If we've overflowed our old buffer, this is bad, but recoverable. We'll
    // just not copy anything from our old plan, since it's all in the past now
    // anyways.
    if (remainingSteps < 0)
    {
      mBufA = forces;
      mActiveBuffer = BUF_A;
      return;
    }

    int copySteps = padSteps;
    int zeroSteps = 0;
    int useSteps = mNumSteps - padSteps;
    if (padSteps > remainingSteps)
    {
      copySteps = remainingSteps;
      zeroSteps = padSteps - remainingSteps;
      useSteps = mNumSteps - padSteps;
    }
    assert(copySteps + zeroSteps + useSteps == mNumSteps);

    if (mActiveBuffer == UNINITIALIZED)
    {
      mBufA.block(0, 0, mForceDim, copySteps).setZero();
      mBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufA.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_A;
    }
    else if (mActiveBuffer == BUF_B)
    {
      mBufA.block(0, 0, mForceDim, copySteps)
          = mBufB.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufA.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_A;
    }
    else if (mActiveBuffer == BUF_A)
    {
      mBufB.block(0, 0, mForceDim, copySteps)
          = mBufA.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mBufB.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufB.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_B;
    }
  }
  else
  {
    mLastWroteBufferAt = startAt;
    if (mActiveBuffer == UNINITIALIZED || mActiveBuffer == BUF_B)
    {
      mBufA = forces;
      // Crucial for lock-free behavior: copy the buffer BEFORE setting the
      // active buffer. Not a huge deal if we're a bit off here in the
      // optimizer, but not ideal.
      mActiveBuffer = BUF_A;
    }
    else
    {
      mBufB = forces;
      // Crucial for lock-free behavior: copy the buffer BEFORE setting the
      // active buffer. Not a huge deal if we're a bit off here in the
      // optimizer, but not ideal.
      mActiveBuffer = BUF_B;
    }
  }
}

/// This retrieves the state of the world at a given time, assuming that we've
/// been applying forces from the buffer since the last state that we fully
/// observed.
void RealTimeControlBuffer::estimateWorldStateAt(
    std::shared_ptr<simulation::World> world, ObservationLog* log, long time)
{
  Observation obs = log->getClosestObservationBefore(time);
  int elapsedSinceObservation = time - obs.time;
  if (elapsedSinceObservation < 0)
  {
    assert(
        elapsedSinceObservation >= 0
        && "estimateWorldStateAt() cannot ask far a time before the earliest available observation.");
  }
  int stepsSinceObservation
      = (int)floor((double)elapsedSinceObservation / mMillisPerStep);
  /*
  std::cout << "RealTimeControlBuffer time: " << time << std::endl;
  std::cout << "RealTimeControlBuffer obs.time: " << obs.time << std::endl;
  std::cout << "RealTimeControlBuffer elapsedSinceObservation: "
            << elapsedSinceObservation << std::endl;
  std::cout << "RealTimeControlBuffer stepsSinceObservation: "
            << stepsSinceObservation << std::endl;
  */

  world->setPositions(obs.pos);
  world->setVelocities(obs.vel);
  world->setMasses(log->getMass());
  for (int i = 0; i < stepsSinceObservation; i++)
  {
    long at = obs.time + i * mMillisPerStep;
    // In the future, project assuming planned forces
    if (at > mControlLog.last())
    {
      world->setExternalForces(getPlannedForce(at, true));
    }
    // In the past, project using known forces read from the buffer
    else
    {
      world->setExternalForces(mControlLog.get(at));
    }
    world->step();
  }
}

/// This rescales the timestep size. This is useful because larger timesteps
/// mean fewer time steps per real unit of time, and thus we can run our
/// optimization slower and still keep up with real life.
void RealTimeControlBuffer::setMillisPerStep(int newMillisPerStep)
{
  mControlLog.setMillisPerStep(newMillisPerStep);
  if (mActiveBuffer == BUF_A)
  {
    rescaleBuffer(mBufA, mMillisPerStep, newMillisPerStep);
  }
  else if (mActiveBuffer == BUF_B)
  {
    rescaleBuffer(mBufB, mMillisPerStep, newMillisPerStep);
  }
  mMillisPerStep = newMillisPerStep;
}

/// This changes the number of steps. Fewer steps mean we can compute a buffer
/// faster, but it also means we have less time to compute the buffer. This
/// probably has a nonlinear effect on runtime.
void RealTimeControlBuffer::setNumSteps(int newNumSteps)
{
  Eigen::MatrixXd newBuf = Eigen::MatrixXd::Zero(mForceDim, newNumSteps);

  int minLen = newNumSteps;
  if (mNumSteps < minLen)
    minLen = mNumSteps;

  if (mActiveBuffer == BUF_A)
  {
    newBuf.block(0, 0, mForceDim, minLen)
        = mBufA.block(0, 0, mForceDim, minLen);
  }
  else if (mActiveBuffer == BUF_B)
  {
    newBuf.block(0, 0, mForceDim, minLen)
        = mBufB.block(0, 0, mForceDim, minLen);
  }

  mBufA = newBuf;
  mBufB = newBuf;
}

/// This returns the number of millis we have left in the plan after `time`.
/// This can be a negative number.
long RealTimeControlBuffer::getPlanBufferMillisAfter(long time)
{
  long planEnd = mLastWroteBufferAt + (mNumSteps * mMillisPerStep);
  return planEnd - time;
}

/// This is useful when we're replicating a log across a network boundary,
/// which comes up in distributed MPC.
void RealTimeControlBuffer::manuallyRecordObservedForce(
    long time, Eigen::VectorXd observation)
{
  mControlLog.record(time, observation);
}

/// This is a helper to rescale the timestep size of a buffer while leaving
/// the data otherwise unchanged.
void RealTimeControlBuffer::rescaleBuffer(
    Eigen::MatrixXd& buf, int oldMillisPerStep, int newMillisPerStep)
{
  Eigen::MatrixXd newBuf = Eigen::MatrixXd::Zero(buf.rows(), buf.cols());

  for (int i = mNumSteps - 1; i >= 0; i--)
  {
    if (newMillisPerStep > oldMillisPerStep)
    {
      // If we're increasing the step size, there's more than one old column per
      // new column, so map from old to new
      int newCol = floor((double)(i * oldMillisPerStep) / newMillisPerStep);
      newBuf.col(newCol) = buf.col(i);
    }
    else
    {
      // If we're increasing the step size, there's more than one new column per
      // old column, so map from new to old
      int oldCol = floor((double)(i * newMillisPerStep) / oldMillisPerStep);
      newBuf.col(i) = buf.col(oldCol);
    }
  }

  buf = newBuf;
}

} // namespace realtime
} // namespace dart