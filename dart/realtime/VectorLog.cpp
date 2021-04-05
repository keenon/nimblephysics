#include "dart/realtime/VectorLog.hpp"

namespace dart {
namespace realtime {

VectorObservation::VectorObservation(long time, Eigen::VectorXs value)
  : time(time), value(value)
{
}

VectorLog::VectorLog(int dim) : mDim(dim), mStartTime(0L)
{
}

void VectorLog::record(long time, Eigen::VectorXs val)
{
  if (mObservations.size() == 0)
    mStartTime = time;
  assert(val.size() == mDim);
  mObservations.emplace_back(time, val);
}

Eigen::MatrixXs VectorLog::getValues(long start, int steps, long millisPerStep)
{
  Eigen::MatrixXs observations = Eigen::MatrixXs::Zero(mDim, steps);

  Eigen::VectorXs cursorValue = Eigen::VectorXs::Zero(mDim);
  int cursorStep = 0;
  for (const VectorObservation& obs : mObservations)
  {
    int step = static_cast<int>(
        ceil(static_cast<s_t>(obs.time - start) / millisPerStep));
    if (step > steps - 1)
      break;
    if (step >= cursorStep)
    {
      // Sweep the last cursor value forward to the current step
      while (cursorStep < step)
      {
        observations.col(cursorStep) = cursorValue;
        cursorStep++;
      }
      // Set the current value to the current state
      cursorValue = obs.value;
      observations.col(step) = cursorValue;
      assert(cursorStep == step);
    }
    else
    {
      cursorValue = obs.value;
    }
  }
  // Sweep the last cursor value forward to the end of the block
  while (cursorStep < steps)
  {
    observations.col(cursorStep) = cursorValue;
    cursorStep++;
  }

  return observations;
}

long VectorLog::availableHistoryBefore(long time)
{
  return time - mStartTime;
}

void VectorLog::discardBefore(long time)
{
  int discardBeforeIndex = -1;
  for (int i = mObservations.size() - 1; i >= 0; i--)
  {
    if (mObservations[i].time < time)
    {
      discardBeforeIndex = i;
      break;
    }
  }
  if (discardBeforeIndex == -1)
    return;

  std::vector<VectorObservation> newObservations;
  for (int i = discardBeforeIndex + 1; i < mObservations.size(); i++)
  {
    newObservations.push_back(mObservations[i]);
  }
  mObservations = newObservations;
}

} // namespace realtime
} // namespace dart