#ifndef UTILS_ACC_MINIMIZER
#define UTILS_ACC_MINIMIZER

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

class AccelerationMinimizer
{
public:
  AccelerationMinimizer(
      int numTimesteps,
      s_t smoothingWeight = 1.0,
      s_t regularizationWeight = 0.01,
      s_t startVelocityZeroWeight = 0.0,
      s_t endVelocityZeroWeight = 0.0,
      int numIterations = 10000);

  Eigen::VectorXs minimize(Eigen::VectorXs series);

  void setDebugIterationBackoff(bool debug);

  void setNumIterationsBackoff(int numIterations);

  void setConvergenceTolerance(s_t tolerance);

protected:
  int mTimesteps;
  s_t mSmoothingWeight;
  s_t mRegularizationWeight;
  s_t mStartVelocityZeroWeight;
  s_t mEndVelocityZeroWeight;
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