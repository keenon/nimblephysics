#ifndef DART_REALTIME_OBS_LOG
#define DART_REALTIME_OBS_LOG

#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace realtime {

struct Observation
{
  long time;
  Eigen::VectorXd pos;
  Eigen::VectorXd vel;

  Observation(long time, Eigen::VectorXd pos, Eigen::VectorXd vel);
};

class ObservationLog
{
public:
  ObservationLog(
      long startTime,
      Eigen::VectorXd initialPos,
      Eigen::VectorXd initialVel,
      Eigen::VectorXd initialMass);

  void observe(
      long time,
      Eigen::VectorXd pos,
      Eigen::VectorXd vel,
      Eigen::VectorXd mass);

  Observation getClosestObservationBefore(long time);

  Eigen::VectorXd getMass();

  void discardBefore(long time);

protected:
  int mDofs;
  int mMassDim;
  std::vector<Observation> mObservations;
  Eigen::VectorXd mMass;
};

} // namespace realtime
} // namespace dart

#endif