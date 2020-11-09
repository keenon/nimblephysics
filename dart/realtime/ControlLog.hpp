#ifndef DART_REALTIME_LOG
#define DART_REALTIME_LOG

#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace realtime {

class ControlLog
{
public:
  ControlLog(int dim, int millisPerStep);

  void record(long time, Eigen::VectorXd control);

  Eigen::VectorXd get(long time);

  void discardBefore(long time);

  void setMillisPerStep(int millisPerStep);

protected:
  int mDim;
  int mMillisPerStep;
  long mLogStart;
  std::vector<Eigen::VectorXd> mLog;
};

} // namespace realtime
} // namespace dart

#endif