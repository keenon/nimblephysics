#include "dart/realtime/ObservationLog.hpp"

#include <iostream>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace realtime {

Observation::Observation(long time, Eigen::VectorXs pos, Eigen::VectorXs vel)
  : time(time), pos(pos), vel(vel)
{
}

ObservationLog::ObservationLog(
    long startTime,
    Eigen::VectorXs initialPos,
    Eigen::VectorXs initialVel,
    Eigen::VectorXs initialMass)
  : mDofs(initialPos.size()), mMassDim(initialMass.size()), mMass(initialMass)
{
  mObservations.emplace_back(startTime, initialPos, initialVel);
}

void ObservationLog::observe(
    long time,
    Eigen::VectorXs pos,
    Eigen::VectorXs vel,
    // TODO(keenon): Support mass observations
    Eigen::VectorXs /* mass */)
{
  mObservations.emplace_back(time, pos, vel);
}

Observation ObservationLog::getClosestObservationBefore(long time)
{
  for (int i = mObservations.size() - 1; i >= 0; i--)
  {
    if (mObservations[i].time <= time)
      return mObservations[i];
  }
  std::cout << "WARNING: Asked for an observation before our initialization. "
               "Returning our initialization"
            << std::endl;
  return mObservations[0];
}

Eigen::VectorXs ObservationLog::getMass()
{
  return mMass;
}

void ObservationLog::discardBefore(long time)
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

  std::vector<Observation> newObservations;
  for (int i = discardBeforeIndex + 1; i < mObservations.size(); i++)
  {
    newObservations.push_back(mObservations[i]);
  }
  mObservations = newObservations;
}

} // namespace realtime
} // namespace dart