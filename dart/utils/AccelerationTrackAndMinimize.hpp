#ifndef UTILS_ACC_TRACK_AND_MINIMIZE
#define UTILS_ACC_TRACK_AND_MINIMIZE

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

typedef struct AccelerationTrackingResult
{
  Eigen::VectorXs series;
  s_t accelerationOffset;
} AccelerationTrackingResult;

class AccelerationTrackAndMinimize
{
public:
  AccelerationTrackAndMinimize(
      int numTimesteps,
      std::vector<bool> trackAccelerationAtTimesteps,
      s_t zeroUnobservedAccWeight = 1.0,
      s_t trackObservedAccWeight = 1.0,
      s_t regularizationWeight = 0.01,
      s_t dt = 1.0,
      int numIterations = 10000);

  AccelerationTrackingResult minimize(
      Eigen::VectorXs series, Eigen::VectorXs trackAcc);

  void setDebugIterationBackoff(bool debug);

  void setNumIterationsBackoff(int numIterations);

  void setConvergenceTolerance(s_t tolerance);

protected:
  int mTimesteps;
  s_t mDt;
  s_t mZeroUnobservedAccWeight;
  s_t mTrackObservedAccWeight;
  s_t mRegularizationWeight;
  std::vector<bool> mTrackAccelerationAtTimesteps;

  int mNumIterations;
  int mNumIterationsBackoff;
  bool mDebugIterationBackoff;
  s_t mConvergenceTolerance;
  Eigen::SparseMatrix<s_t> mB_sparse;
  Eigen::SparseQR<Eigen::SparseMatrix<s_t>, Eigen::NaturalOrdering<int>>
      mB_sparseSolver;
};

} // namespace utils
} // namespace dart

#endif