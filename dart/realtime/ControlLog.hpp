#ifndef DART_REALTIME_LOG
#define DART_REALTIME_LOG

#include <vector>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace realtime {

class ControlLog
{
public:
  ControlLog(int dim, int millisPerStep);

  void record(long time, Eigen::VectorXs control);

  long last();

  Eigen::VectorXs get(long time);

  void discardBefore(long time);

  void setMillisPerStep(int millisPerStep);

protected:
  int mDim;
  int mMillisPerStep;
  long mLogStart;
  long mLogEnd;
  std::vector<Eigen::VectorXs> mLog;
};

} // namespace realtime
} // namespace dart

#endif