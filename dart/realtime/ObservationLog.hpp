#ifndef DART_REALTIME_OBS_LOG
#define DART_REALTIME_OBS_LOG

#include <vector>

#include "dart/include_eigen.hpp"

#include "dart/math/MathTypes.hpp"
namespace dart {
namespace realtime {

struct Observation
{
  long time;
  Eigen::VectorXs pos;
  Eigen::VectorXs vel;

  Observation(long time, Eigen::VectorXs pos, Eigen::VectorXs vel);
};

class ObservationLog
{
public:
  ObservationLog(
      long startTime,
      Eigen::VectorXs initialPos,
      Eigen::VectorXs initialVel,
      Eigen::VectorXs initialMass);

  void observe(
      long time,
      Eigen::VectorXs pos,
      Eigen::VectorXs vel,
      Eigen::VectorXs mass);

  Observation getClosestObservationBefore(long time);

  Eigen::VectorXs getMass();

  void discardBefore(long time);

protected:
  int mDofs;
  int mMassDim;
  std::vector<Observation> mObservations;
  Eigen::VectorXs mMass;
};

} // namespace realtime
} // namespace dart

#endif