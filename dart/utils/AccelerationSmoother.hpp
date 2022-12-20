#ifndef UTILS_PATH_SMOOTHER
#define UTILS_PATH_SMOOTHER

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

class AccelerationSmoother
{
public:
  /**
   * Create (and pre-factor) a smoother that can remove the "jerk" from a time
   * seriese of data.
   */
  AccelerationSmoother(
      int timesteps,
      s_t smoothingWeight,
      s_t regularizationWeight,
      bool useSparse = true,
      bool useIterativeSolver = true);

  /**
   * Adjust a time series of points to minimize the jerk (d/dt of acceleration)
   * implied by the position data. This will return a shorter time series,
   * missing the last 3 entries, because those cannot be smoothed by this
   * technique.
   *
   * This method assumes that the `series` matrix has `mTimesteps` number of
   * columns, and each column represents a complete joint configuration at that
   * timestep.
   */
  Eigen::MatrixXs smooth(Eigen::MatrixXs series);

  /**
   * This computes the squared loss for this smoother, given a time series and a
   * set of perturbations `delta` to the time series.
   */
  s_t getLoss(
      Eigen::MatrixXs series,
      Eigen::MatrixXs originalSeries,
      bool debug = false);

  /**
   * This prints the stats for a time-series of data, with pos, vel, accel, and
   * jerk
   */
  void debugTimeSeries(Eigen::VectorXs series);

private:
  int mTimesteps;
  int mSmoothedTimesteps;
  s_t mSmoothingWeight;
  s_t mRegularizationWeight;
  bool mUseSparse;
  bool mUseIterativeSolver;
  Eigen::MatrixXs mB;
  Eigen::HouseholderQR<Eigen::MatrixXs> mFactoredB;

  Eigen::SparseMatrix<s_t> mB_sparse;
  Eigen::SparseQR<Eigen::SparseMatrix<s_t>, Eigen::NaturalOrdering<int>>
      mB_sparseSolver;
};

} // namespace utils
} // namespace dart

#endif