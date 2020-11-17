#ifndef DART_REALTIME_VECTOR_LOG
#define DART_REALTIME_VECTOR_LOG

#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace realtime {

struct VectorObservation
{
  long time;
  Eigen::VectorXd value;

  VectorObservation(long time, Eigen::VectorXd value);
};

class VectorLog
{
public:
  VectorLog(int dim);

  void record(long time, Eigen::VectorXd val);

  Eigen::MatrixXd getValues(long start, int steps, long millisPerStep);

  void discardBefore(long time);

  long availableHistoryBefore(long time);

protected:
  int mDim;
  long mStartTime;
  std::vector<VectorObservation> mObservations;
};

} // namespace realtime
} // namespace dart

#endif