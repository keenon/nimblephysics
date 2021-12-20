#include "dart/realtime/RealTimeControlBuffer.hpp"

#include <iostream>

#include "dart/simulation/World.hpp"

namespace dart {
namespace realtime {

RealTimeControlBuffer::RealTimeControlBuffer(
    int forceDim, int steps, int millisPerStep, int stateDim)
  : mForceDim(forceDim),
    mStateDim(stateDim),
    mNumSteps(steps),
    mMillisPerStep(millisPerStep),
    mActiveBuffer(UNINITIALIZED),
    mActiveBufferLaw(UNINITIALIZED),
    mBufA(Eigen::MatrixXs::Zero(forceDim, steps)),
    mBufB(Eigen::MatrixXs::Zero(forceDim, steps)),
    mkBufA(Eigen::MatrixXs::Zero(forceDim, steps)),
    mkBufB(Eigen::MatrixXs::Zero(forceDim, steps)),
    mxBufA(Eigen::MatrixXs::Zero(stateDim, steps)),
    mxBufB(Eigen::MatrixXs::Zero(stateDim, steps)),
    mAlphaBufA(Eigen::VectorXs::Zero(steps)),
    mAlphaBufB(Eigen::VectorXs::Zero(steps)),
    mControlLog(ControlLog(forceDim, millisPerStep))
{
  // std::cout<<"Initializing Buffer..." << std::endl;
  for(int i=0; i < steps; i++)
  {
    mKBufA.push_back(Eigen::MatrixXs::Zero(forceDim, stateDim));
    mKBufB.push_back(Eigen::MatrixXs::Zero(forceDim, stateDim));
  }
  // std::cout << "Initialization Complete" <<std::endl;
}

/// Gets the force at a given timestep
Eigen::VectorXs RealTimeControlBuffer::getPlannedForce(long time, bool dontLog)
{
  if (mActiveBuffer == UNINITIALIZED)
  {
    // Unitialized, default to no force
    return Eigen::VectorXs::Zero(mForceDim);
  }
  int elapsed = time - mLastWroteBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    // Which means 
    std::cout << "Ask for force time in the past" << std::endl;
    return Eigen::VectorXs::Zero(mForceDim);
  }

  int step = (int)floor((s_t)elapsed / mMillisPerStep);
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
    Eigen::VectorXs oob = Eigen::VectorXs::Zero(mForceDim);
    if (!dontLog)
      mControlLog.record(time, oob);
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}

Eigen::VectorXs RealTimeControlBuffer::getPlannedk(long time, bool dontLog)
{
  if(dontLog)
  {
    std::cout << "Don't Use Log" << std::endl;
  }
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to no force
    return Eigen::VectorXs::Zero(mForceDim);
  }
  int elapsed = time - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    std::cout << "Ask for k time in the past" << std::endl;
    return Eigen::VectorXs::Zero(mForceDim);
  }

  int step = (int)floor((s_t)elapsed / mMillisPerStep);
  if (step < mNumSteps)
  {
    if (mActiveBufferLaw == BUF_A)
    {
      return mkBufA.col(step);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      return mkBufB.col(step);
    }
    else
      assert(false && "Should never reach this point");
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    Eigen::VectorXs oob = Eigen::VectorXs::Zero(mForceDim);
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}


Eigen::VectorXs RealTimeControlBuffer::getPlannedState(long time, bool dontLog)
{
  if(dontLog)
  {
    std::cout << "Don't use Log" << std::endl;
  }
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to no force
    std::cout << "Buffer Not Initialized from state!" << std::endl;
    return Eigen::VectorXs::Zero(mStateDim);
  }
  int elapsed = time - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    std::cout << "Ask for state time in the past" << std::endl;
    return Eigen::VectorXs::Zero(mStateDim);
  }

  int step = (int)floor((s_t)elapsed / mMillisPerStep);
  if (step < mNumSteps)
  {
    if (mActiveBufferLaw == BUF_A)
    {
      return mxBufA.col(step);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      return mxBufB.col(step);
    }
    else
      assert(false && "Should never reach this point");
  }
  else
  {
    std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    Eigen::VectorXs oob = Eigen::VectorXs::Zero(mStateDim);
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}

s_t RealTimeControlBuffer::getPlannedAlpha(long time, bool dontLog)
{
  if(dontLog)
  {
    std::cout << "Don't use Log" << std::endl;
  }
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to no force
    return 0.0;
  }
  int elapsed = time - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    std::cout << "Ask for alpha time in the past" << std::endl;
    return 0.0;
  }

  int step = (int)floor((s_t)elapsed / mMillisPerStep);
  if (step < mNumSteps)
  {
    if (mActiveBufferLaw == BUF_A)
    {
      return mAlphaBufA(step);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      return mAlphaBufB(step);
    }
    else
      assert(false && "Should never reach this point");
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    s_t oob = 0.0;
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}


Eigen::MatrixXs RealTimeControlBuffer::getPlannedK(long time, bool dontLog)
{
  if(dontLog)
  {
    std::cout << "Don't Use Log" << std::endl;
  }
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to no force
    return Eigen::MatrixXs::Zero(mForceDim, mStateDim);
  }
  int elapsed = time - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to no force
    std::cout << "Ask for K time in the past" <<std::endl;
    return Eigen::MatrixXs::Zero(mForceDim, mStateDim);
  }

  int step = (int)floor((s_t)elapsed / mMillisPerStep);
  if (step < mNumSteps)
  {
    if (mActiveBufferLaw == BUF_A)
    {
      return mKBufA[step];
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      return mKBufB[step];
    }
    else
      assert(false && "Should never reach this point");
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    Eigen::MatrixXs oob = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
    return oob;
  }
  // The code should never reach here, but it's here to keep the compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}

/// This gets planned forces starting at `start`, and continuing for the
/// length of our buffer size `mSteps`. This is useful for initializing MPC
/// runs. It supports walking off the end of known future, and assumes 0
/// forces in all extrapolation.
/// Assume forcesOut has 50 steps
void RealTimeControlBuffer::getPlannedForcesStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXs> forcesOut)
{
  assert(forcesOut.rows() == mForceDim);
  assert(forcesOut.cols() == mNumSteps);
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
  int startStep = (int)floor((s_t)elapsed / mMillisPerStep);
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

void RealTimeControlBuffer::getPlannedkStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXs> kOut)
{
  assert(kOut.rows() == mForceDim);
  assert(kOut.cols() == mNumSteps);
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to 0
    kOut.setZero();
    return;
  }
  int elapsed = start - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to 0
    kOut.setZero();
    return;
  }
  int startStep = (int)floor((s_t)elapsed / mMillisPerStep);
  if (startStep < mNumSteps)
  {
    // Copy the appropriate block of our active buffer to the forcesOut block
    if (mActiveBufferLaw == BUF_A)
    {
      kOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mkBufA.block(0, startStep, mForceDim, mNumSteps - startStep);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      kOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mkBufB.block(0, startStep, mForceDim, mNumSteps - startStep);
    }
    else
      assert(false && "Should never reach this point");
    // Zero out the remainder of the forcesOut block
    kOut.block(0, mNumSteps - startStep, mForceDim, startStep).setZero();
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    kOut.setZero();
  }
}

void RealTimeControlBuffer::getPlannedStateStartingAt(
    long start, Eigen::Ref<Eigen::MatrixXs> stateOut)
{
  assert(stateOut.rows() == mStateDim);
  assert(stateOut.cols() == mNumSteps);
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to 0
    stateOut.setZero();
    return;
  }
  int elapsed = start - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to 0
    stateOut.setZero();
    return;
  }
  int startStep = (int)floor((s_t)elapsed / mMillisPerStep);
  if (startStep < mNumSteps)
  {
    // Copy the appropriate block of our active buffer to the forcesOut block
    if (mActiveBufferLaw == BUF_A)
    {
      stateOut.block(0, 0, mStateDim, mNumSteps - startStep)
          = mxBufA.block(0, startStep, mStateDim, mNumSteps - startStep);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      stateOut.block(0, 0, mStateDim, mNumSteps - startStep)
          = mxBufB.block(0, startStep, mStateDim, mNumSteps - startStep);
    }
    else
      assert(false && "Should never reach this point");
    // Zero out the remainder of the forcesOut block
    stateOut.block(0, mNumSteps - startStep, mStateDim, startStep).setZero();
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    stateOut.setZero();
  }
}

void RealTimeControlBuffer::getPlannedAlphaStartingAt(long start,
                                                      Eigen::Ref<Eigen::VectorXs> alphaOut)
{
  assert(alphaOut.size() == mNumSteps);
  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to 0
    alphaOut.setZero();
    return;
  }
  int elapsed = start - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to 0
    alphaOut.setZero();
    return;
  }
  int startStep = (int)floor((s_t)elapsed / mMillisPerStep);
  if (startStep < mNumSteps)
  {
    // Copy the appropriate block of our active buffer to the forcesOut block
    if (mActiveBufferLaw == BUF_A)
    {
      alphaOut.segment(0,mNumSteps - startStep)
          = mAlphaBufA.segment(startStep, mNumSteps - startStep);
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      alphaOut.segment(0, mNumSteps - startStep)
          = mAlphaBufB.segment(startStep, mNumSteps - startStep);
    }
    else
      assert(false && "Should never reach this point");
    // Zero out the remainder of the forcesOut block
    alphaOut.segment(mNumSteps - startStep, startStep).setZero();
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    alphaOut.setZero();
  }
}

void RealTimeControlBuffer::getPlannedKStartingAt(long start, 
  std::vector<Eigen::MatrixXs> &KOut)
{
  // Need to make sure that empty and filed KOut will be treated similarly
  assert(KOut.size() == mNumSteps);
  assert(KOut[0].rows() == mForceDim);
  assert(KOut[0].cols() == mStateDim);

  if (mActiveBufferLaw == UNINITIALIZED)
  {
    // Unitialized, default to 0
    for(int i=0; i < KOut.size(); i++)
    {
      KOut[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
    }
    return;
  }
  int elapsed = start - mLastWroteLawBufferAt;
  if (elapsed < 0)
  {
    // Asking for some time in the past, default to 0
    for(int i = 0; i < KOut.size(); i++)
    {
      KOut[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
    }
    return;
  }
  int startStep = (int)floor((s_t)elapsed / mMillisPerStep);
  if (startStep < mNumSteps)
  {
    // Copy the appropriate block of our active buffer to the forcesOut block
    if (mActiveBufferLaw == BUF_A)
    {
      /*
      kOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mkBufA.block(0, startStep, mForceDim, mNumSteps - startStep);
      */
      for(int i = 0; i < mNumSteps- startStep; i++)
      {
        KOut[i] = mKBufA[i];
      }
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      /*
      kOut.block(0, 0, mForceDim, mNumSteps - startStep)
          = mkBufB.block(0, startStep, mForceDim, mNumSteps - startStep);
      */
      for(int i = 0; i < mNumSteps- startStep; i++)
      {
        KOut[i] = mKBufB[i];
      }
    }
    else
    {
      assert(false && "Should never reach this point");
    }
    // Zero out the remainder of the forcesOut block
    
    // KOut.block(0, mNumSteps - startStep, mForceDim, startStep).setZero();
    for(int i = 0; i < mNumSteps- startStep; i++)
    {
      KOut[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
    }
  }
  else
  {
    // std::cout << "WARNING: MPC isn't keeping up!" << std::endl;
    for(int i = 0; i < KOut.size(); i++)
    {
      KOut[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
    }
  }
}

size_t RealTimeControlBuffer::getRemainSteps(long start)
{
  int elapsed = start - mLastWroteBufferAt;
  if(elapsed < 0)
  {
    return 0;
  }
  size_t elapsed_steps = (size_t)(((s_t)elapsed / mMillisPerStep));
  size_t remain_steps;
  if(mNumSteps > elapsed_steps)
    remain_steps = mNumSteps - elapsed_steps;
  else
    remain_steps = 0;
  return remain_steps;
}

/// This swaps in a new buffer of forces. The assumption is that "startAt" is
/// before "now", because we'll erase old data in this process.
void RealTimeControlBuffer::setControlForcePlan(
    long startAt, long now, Eigen::MatrixXs forces)
{
  // The solve is too fast that problem is solved before expected time
  if (startAt > now)
  {
    long padMillis = startAt - now;
    int padSteps = (int)floor((s_t)padMillis / mMillisPerStep);
    // If we're trying to set the force plan too far out in the future, this
    // whole exercise is a not allowed
    if (padSteps >= mNumSteps)
    {
      return;
    }
    // Otherwise, we're going to copy part of the existing plan
    // Since the first time pointer of control force plan should
    // always be earlier than actual time that get the buffer
    int currentStep
        = (int)floor((s_t)(now - mLastWroteBufferAt) / mMillisPerStep);
    int remainingSteps = mNumSteps - currentStep;
    mLastWroteBufferAt = now;

    // If we've overflowed our old buffer, this is bad, but recoverable. We'll
    // just not copy anything from our old plan, since it's all in the past now
    // anyways.
    // Previous time the MPC doesn't catch up
    if (remainingSteps < 0)
    {
      mBufA = forces;
      mActiveBuffer = BUF_A;
      return;
    }

    // With in pad step uses previous force until now, use current force
    int copySteps = padSteps;
    int zeroSteps = 0;
    // usestep will use new force
    int useSteps = mNumSteps - padSteps;
    if (padSteps > remainingSteps)
    {
      copySteps = remainingSteps;
      // There are some forces in the middle that need to set to zero since no information
      // can be provided either from incoming force or original plan
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

// Big Bug fixed hopefully the performance will be better
void RealTimeControlBuffer::setControlLawPlan(long startAt, long now,
                                              std::vector<Eigen::VectorXs> ks,
                                              std::vector<Eigen::MatrixXs> Ks,
                                              std::vector<Eigen::VectorXs> states,
                                              std::vector<s_t> alphas)
{
  // remove the last state
  states.pop_back();
  assert(ks.size() == Ks.size() && states.size() == Ks.size() && Ks.size() == alphas.size());
  if (startAt > now)
  {
    long padMillis = startAt - now;
    int padSteps = (int)floor((s_t)padMillis / mMillisPerStep);
    // If we're trying to set the force plan too far out in the future, this
    // whole exercise is a no-op
    if (padSteps >= mNumSteps)
    {
      return;
    }
    // Otherwise, we're going to copy part of the existing plan
    int currentStep
        = (int)floor((s_t)(now - mLastWroteLawBufferAt) / mMillisPerStep);
    int remainingSteps = mNumSteps - currentStep;
    mLastWroteLawBufferAt = now;

    // If we've overflowed our old buffer, this is bad, but recoverable. We'll
    // just not copy anything from our old plan, since it's all in the past now
    // anyways.
    if (remainingSteps < 0)
    {
      mkBufA = Eigen::MatrixXs::Zero(mForceDim, ks.size());
      mxBufA = Eigen::MatrixXs::Zero(mStateDim, states.size());
      mAlphaBufA = Eigen::VectorXs::Zero(alphas.size());
      mKBufA = Ks;
      for(int i = 0; i < ks.size(); i++)
      {
        mkBufA.col(i) = ks[i];
        mxBufA.col(i) = states[i];
        mAlphaBufA(i) = alphas[i];
      }

      mActiveBufferLaw = BUF_A;
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

    if (mActiveBufferLaw == UNINITIALIZED)
    {
      /*
      mBufA.block(0, 0, mForceDim, copySteps).setZero();
      mBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufA.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_A;
      */
     // For k
     mkBufA.block(0, 0, mForceDim, copySteps).setZero();
     mkBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
     for(int i = 0; i < useSteps; i++)
     {
       mkBufA.col(i + copySteps + zeroSteps) = ks[i];
     }
     // For states
     mxBufA.block(0, 0, mStateDim, copySteps).setZero();
     mxBufA.block(0, copySteps, mStateDim, zeroSteps).setZero();
     for(int i = 0; i < useSteps; i++)
     {
       mxBufA.col(i + copySteps + zeroSteps) = states[i];
     }

     // For Alpha
     mAlphaBufA.segment(0, copySteps).setZero();
     mAlphaBufA.segment(copySteps,zeroSteps).setZero();
     for(int i = 0; i < useSteps;i++)
     {
       mAlphaBufA(i + copySteps + zeroSteps) = alphas[i];
     }
     // For K
     for(int i = 0; i < copySteps + zeroSteps; i++)
     {
       mKBufA[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
     }
     for(int i = 0; i < useSteps; i++)
     {
       mKBufA[i + copySteps + zeroSteps] = Ks[i];
     }
     mActiveBufferLaw = BUF_A;
    }
    else if (mActiveBufferLaw == BUF_B)
    {
      /*
      mBufA.block(0, 0, mForceDim, copySteps)
          = mBufB.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufA.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_A;
      */
      // For k
      mkBufA.block(0, 0, mForceDim, copySteps)
          = mkBufB.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mkBufA.block(0, copySteps, mForceDim, zeroSteps).setZero();
      for(int i = 0; i < useSteps;i++)
      {
        mkBufA.col(i + copySteps + zeroSteps) = ks[i];
      }
      // For state
      mxBufA.block(0, 0, mStateDim, copySteps)
          = mxBufB.block(0, mNumSteps - copySteps, mStateDim, copySteps);
      mxBufA.block(0, copySteps, mStateDim, zeroSteps).setZero();
      for(int i = 0; i < useSteps;i++)
      {
        mxBufA.col(i + copySteps + zeroSteps) = states[i];
      }
      // For Alpha
      mAlphaBufA.segment(0, copySteps) = mAlphaBufB.segment(mNumSteps - copySteps, copySteps);
      mAlphaBufA.segment(copySteps, zeroSteps).setZero();
      for(int i = 0; i < useSteps; i++)
      {
        mAlphaBufA(i + copySteps + zeroSteps) = alphas[i];
      }
      // For K
      for(int i = 0;i < copySteps + zeroSteps; i++)
      {
        if(i < copySteps)
        {
          mKBufA[i] = mKBufB[mNumSteps - copySteps + i];
        }
        else
        {
          mKBufA[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
        }
      }
      for(int i = 0; i < useSteps;i++)
      {
        mKBufA[i + copySteps + zeroSteps] = Ks[i];
      }
      // change active buffer
      mActiveBufferLaw = BUF_A;
      
    }
    else if (mActiveBufferLaw == BUF_A)
    {
      /*
      mBufB.block(0, 0, mForceDim, copySteps)
          = mBufA.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mBufB.block(0, copySteps, mForceDim, zeroSteps).setZero();
      mBufB.block(0, copySteps + zeroSteps, mForceDim, useSteps)
          = forces.block(0, 0, mForceDim, useSteps);
      mActiveBuffer = BUF_B;
      */
      // For k
      mkBufB.block(0, 0, mForceDim, copySteps)
          = mkBufA.block(0, mNumSteps - copySteps, mForceDim, copySteps);
      mkBufB.block(0, copySteps, mForceDim, zeroSteps).setZero();
      for(int i = 0; i < useSteps;i++)
      {
        mkBufB.col(i + copySteps + zeroSteps) = ks[i];
      }

      // For state
      mxBufB.block(0, 0, mStateDim, copySteps)
          = mxBufA.block(0, mNumSteps - copySteps, mStateDim, copySteps);
      mxBufB.block(0, copySteps, mStateDim, zeroSteps).setZero();
      for(int i = 0; i < useSteps;i++)
      {
        mxBufB.col(i + copySteps + zeroSteps) = states[i];
      }
      // For Alpha
      mAlphaBufB.segment(0, copySteps) = mAlphaBufA.segment(mNumSteps - copySteps, copySteps);
      mAlphaBufB.segment(copySteps, zeroSteps).setZero();
      for(int i = 0; i < useSteps; i++)
      {
        mAlphaBufB(i + copySteps + zeroSteps) = alphas[i];
      }
      // For K
      for(int i = 0;i < copySteps + zeroSteps; i++)
      {
        if(i < copySteps)
        {
          mKBufB[i] = mKBufA[mNumSteps - copySteps + i];
        }
        else
        {
          mKBufB[i] = Eigen::MatrixXs::Zero(mForceDim, mStateDim);
        }
      }
      for(int i = 0; i < useSteps;i++)
      {
        mKBufB[i + copySteps + zeroSteps] = Ks[i];
      }
      // change active buffer
      mActiveBufferLaw = BUF_B;
    }
  }
  else
  {
    mLastWroteLawBufferAt = startAt;
    if (mActiveBufferLaw == UNINITIALIZED || mActiveBufferLaw == BUF_B)
    {
      mKBufA = Ks;
      for(int i = 0; i < ks.size(); i++)
      {
        mkBufA.col(i) = ks[i];
        mxBufA.col(i) = states[i];
        mAlphaBufA(i) = alphas[i]; 
      }
      // Crucial for lock-free behavior: copy the buffer BEFORE setting the
      // active buffer. Not a huge deal if we're a bit off here in the
      // optimizer, but not ideal.
      mActiveBufferLaw = BUF_A;
    }
    else
    {
      mKBufB = Ks;
      for(int i = 0; i < ks.size(); i++)
      {
        mkBufB.col(i) = ks[i];
        mxBufB.col(i) = states[i];
        mAlphaBufB(i) = alphas[i];
      }
      // Crucial for lock-free behavior: copy the buffer BEFORE setting the
      // active buffer. Not a huge deal if we're a bit off here in the
      // optimizer, but not ideal.
      mActiveBufferLaw = BUF_B;
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
      = (int)floor((s_t)elapsedSinceObservation / mMillisPerStep);
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
      if(!mUseiLQR)
      {
        Eigen::VectorXs action = getPlannedForce(at, true);
        if(action.size() == world->getNumDofs())
          world->setControlForces(action);
        else
          world->setAction(action);
      }
      else
      {
        // TODO: Eric This may be quite problematic since it report that we require time from past..
        // Which means the 'at' time is before the last time the control force was written to the buffer
        
        Eigen::VectorXs action = getPlannedForce(at, true);
        Eigen::VectorXs k = getPlannedk(at);
        Eigen::VectorXs x = getPlannedState(at);
        Eigen::MatrixXs K = getPlannedK(at);
        Eigen::VectorXs stateErr = world->getState() - x;
        s_t alpha = getPlannedAlpha(at);
        if(stateErr.norm() < 0.1)
        {
          action = (action + alpha * k
                   + K*(world->getState() - x).cwiseMin(mActionBound)).cwiseMax(-mActionBound);
        }
        world->setAction(action);
      }
      
    }
    // In the past, project using known forces read from the buffer
    // Here no need to differentiate iLQR and Trajectory Opt since they 
    // all set the executed force here
    else
    {
      Eigen::VectorXs action = mControlLog.get(at);
      if(action.size() == world->getNumDofs())
        world->setControlForces(action);
      else
        world->setAction(action);
    }
    world->step();
  }
}
// TODO: Need to add a version that estimate the world state using control law

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
  Eigen::MatrixXs newBuf = Eigen::MatrixXs::Zero(mForceDim, newNumSteps);

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
    long time, Eigen::VectorXs observation)
{
  mControlLog.record(time, observation);
}

/// This is a helper to rescale the timestep size of a buffer while leaving
/// the data otherwise unchanged.
/// TODO: Eric Enable rescale of K buffers
void RealTimeControlBuffer::rescaleBuffer(
    Eigen::MatrixXs& buf, int oldMillisPerStep, int newMillisPerStep)
{
  Eigen::MatrixXs newBuf = Eigen::MatrixXs::Zero(buf.rows(), buf.cols());

  for (int i = mNumSteps - 1; i >= 0; i--)
  {
    if (newMillisPerStep > oldMillisPerStep)
    {
      // If we're increasing the step size, there's more than one old column per
      // new column, so map from old to new
      int newCol = static_cast<int>(
          floor(static_cast<s_t>(i * oldMillisPerStep) / newMillisPerStep));
      newBuf.col(newCol) = buf.col(i);
    }
    else
    {
      // If we're increasing the step size, there's more than one new column per
      // old column, so map from new to old
      int oldCol = static_cast<int>(
          floor(static_cast<s_t>(i * newMillisPerStep) / oldMillisPerStep));
      newBuf.col(i) = buf.col(oldCol);
    }
  }

  buf = newBuf;
}

long RealTimeControlBuffer::getLastWriteBufferTime()
{
  return mLastWroteBufferAt;
}

long RealTimeControlBuffer::getLastWriteBufferLawTime()
{
  return mLastWroteLawBufferAt;
}

void RealTimeControlBuffer::setiLQRFlag(bool ilqr_flag)
{
  mUseiLQR = ilqr_flag;
}

void RealTimeControlBuffer::setActionBound(s_t bound)
{
  mActionBound = bound;
}

} // namespace realtime
} // namespace dart