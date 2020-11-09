#include "dart/realtime/ControlLog.hpp"

namespace dart {
namespace realtime {

ControlLog::ControlLog(int dim, int millisPerStep)
  : mDim(dim), mMillisPerStep(millisPerStep)
{
}

void ControlLog::record(long time, Eigen::VectorXd control)
{
  if (mLog.size() == 0)
  {
    mLogStart = time;
    mLog.push_back(control);
  }
  else
  {
    long logEnd = mLogStart + ((mLog.size() - 1) * mMillisPerStep);
    int steps = (int)floor((double)(time - logEnd) / mMillisPerStep);
    // This means we're recording backwards in time, which shouldn't be allowed.
    if (steps < 0)
    {
      assert(
          false
          && "ControlLog::record() expects time to monotonically increase");
      return;
    }
    // This means we're overwriting the last element of the log, cause we
    // haven't had time to run a full timestep since our last recorded value
    if (steps == 0)
    {
      mLog[mLog.size() - 1] = control;
      return;
    }
    // Otherwise, we need to extend the last recorded force until just before
    // this timestep, on the assumption that the motors have been executing that
    // command until they were updated.
    Eigen::VectorXd last = mLog[mLog.size() - 1];
    for (int i = 0; i < steps - 1; i++)
    {
      mLog.push_back(last);
    }
    mLog.push_back(control);
  }
}

Eigen::VectorXd ControlLog::get(long time)
{
  // If we haven't recorded anything yet, default to 0
  if (mLog.size() == 0)
  {
    return Eigen::VectorXd::Zero(mDim);
  }

  int steps = (int)floor((double)(time - mLogStart) / mMillisPerStep);
  // If we're out of bounds in the past, extend our initial force
  if (steps <= 0)
    return mLog[0];
  // If we're out of bounds in the future, extend our last force
  if (steps >= mLog.size())
    return mLog[mLog.size() - 1];
  // Otherwise return the recorded force
  return mLog[steps];
}

void ControlLog::discardBefore(long time)
{
  if (time <= mLogStart || mLog.size() == 0)
    return;
  int discardSteps = (int)ceil((double)(time - mLogStart) / mMillisPerStep);
  // This means we're throwing out the whole log, just extrapolate the last
  // known force
  if (discardSteps >= mLog.size())
  {
    Eigen::VectorXd last = mLog[mLog.size() - 1];
    mLog.clear();
    mLog.push_back(last);
    mLogStart = time;
    return;
  }
  // Otherwise we're just snipping part of the log, so copy just the bit we care
  // about, and throw away the rest
  std::vector<Eigen::VectorXd> trimmedLog;
  for (int i = discardSteps; i < mLog.size(); i++)
  {
    trimmedLog.push_back(mLog[i]);
  }
  mLog = trimmedLog;
  mLogStart += discardSteps * mMillisPerStep;
}

void ControlLog::setMillisPerStep(int newMillisPerStep)
{
  int duration = mLog.size() * mMillisPerStep;
  int newSteps = (int)ceil((double)duration / newMillisPerStep);

  std::vector<Eigen::VectorXd> newLog;
  for (int i = 0; i < newSteps; i++)
  {
    newLog.push_back(get(mLogStart + i * newMillisPerStep));
  }

  mMillisPerStep = newMillisPerStep;
  mLog = newLog;
}

} // namespace realtime
} // namespace dart