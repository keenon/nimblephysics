#ifndef UTILS_PATH_SMOOTHER
#define UTILS_PATH_SMOOTHER

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

class AccelerationSmoother
{
public:
  /**
   * Create (and pre-factor) a smoother that can remove the "jerk" from a time
   * seriese of data.
   *
   * The alpha value will determine how much smoothing to apply. A value of 0
   * corresponds to no smoothing.
   */
  AccelerationSmoother(int timesteps, s_t alpha);

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
      Eigen::VectorXs series, Eigen::VectorXs deltas, bool debug = false);

  /**
   * This prints the stats for a time-series of data, with pos, vel, accel, and
   * jerk
   */
  void debugTimeSeries(Eigen::VectorXs series);

private:
  int mTimesteps;
  s_t mAlpha;
  Eigen::Matrix4s mPosMap;
  Eigen::MatrixXs mB;
  Eigen::HouseholderQR<Eigen::MatrixXs> mFactoredB;
};

} // namespace utils
} // namespace dart

#endif