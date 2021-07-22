#include "AccelerationSmoother.hpp"

#include <iostream>

namespace dart {
namespace utils {

/**
 * Create (and pre-factor) a smoother that can remove the "jerk" from a time
 * seriese of data.
 *
 * The alpha value will determine how much smoothing to apply. A value of 0
 * corresponds to no smoothing.
 */
AccelerationSmoother::AccelerationSmoother(int timesteps, s_t alpha)
  : mTimesteps(timesteps), mAlpha(alpha)
{
  Eigen::Matrix4s stamp;
  // clang-format off
  stamp <<  1, -3,  3, -1,
           -3,  9, -9,  3,
            3, -9,  9, -3,
           -1,  3, -3,  1;
  // clang-format on
  stamp *= mAlpha;
  mPosMap = stamp * 2;

  mB = Eigen::MatrixXs::Zero(mTimesteps, mTimesteps);
  for (int i = 0; i < mTimesteps - 3; i++)
  {
    mB.block<4, 4>(i, i) += stamp;
  }
  mB += Eigen::MatrixXs::Identity(mTimesteps, mTimesteps);

  mFactoredB = Eigen::HouseholderQR<Eigen::MatrixXs>(mB);
};

/**
 * Adjust a time series of points to minimize the jerk (d/dt of acceleration)
 * implied by the position data. This will return a shorter time series, missing
 * the last 3 entries, because those cannot be smoothed by this technique.
 *
 * This method assumes that the `series` matrix has `mTimesteps` number of
 * columns, and each column represents a complete joint configuration at that
 * timestep.
 */
Eigen::MatrixXs AccelerationSmoother::smooth(Eigen::MatrixXs series)
{
  assert(series.cols() == mTimesteps);

  Eigen::MatrixXs smoothed
      = Eigen::MatrixXs::Zero(series.rows(), mTimesteps - 3);

  for (int row = 0; row < series.rows(); row++)
  {
    Eigen::VectorXs c = Eigen::VectorXs::Zero(mTimesteps);
    for (int i = 0; i < mTimesteps - 3; i++)
    {
      c.segment<4>(i) += mPosMap * series.block<1, 4>(row, i).transpose();
    }
    // Eigen::VectorXs deltas = mB.completeOrthogonalDecomposition().solve(c);
    Eigen::VectorXs deltas = mFactoredB.solve(c);
    smoothed.row(row) = series.block(row, 0, 1, mTimesteps - 3)
                        - deltas.segment(0, mTimesteps - 3).transpose();
  }

  return smoothed;
};

/**
 * This computes the squared loss for this smoother, given a time series and a
 * set of perturbations `delta` to the time series.
 */
s_t AccelerationSmoother::getLoss(
    Eigen::VectorXs series, Eigen::VectorXs deltas, bool debug)
{
  (void)series;
  (void)deltas;
  ///////////////////////////////////////////////////////////////////
  // Compute matrix version
  ///////////////////////////////////////////////////////////////////

  Eigen::Vector4s jMask;
  jMask << -1, 3, -3, 1;

  Eigen::VectorXs c = Eigen::VectorXs::Zero(mTimesteps);
  for (int i = 0; i < mTimesteps - 3; i++)
  {
    c.segment<4>(i) += mPosMap * series.segment<4>(i);
  }
  Eigen::MatrixXs BminusI
      = mB - Eigen::MatrixXs::Identity(mTimesteps, mTimesteps);
  s_t matrix_score = (deltas.transpose() * mB * deltas)(0)
                     + (series.transpose() * BminusI * series)(0)
                     + c.dot(deltas);
  // s_t matrix_score = series.transpose() * BminusI * series;

  ///////////////////////////////////////////////////////////////////
  // Compute manual version
  ///////////////////////////////////////////////////////////////////
  s_t manual_score = 0.0;
  for (int i = 0; i < mTimesteps - 3; i++)
  {
    /*
    s_t vt = series(i + 1) - series(i);
    s_t vt_1 = series(i + 2) - series(i + 1);
    s_t vt_2 = series(i + 3) - series(i + 2);
    */
    s_t vt = (series(i + 1) + deltas(i + 1)) - (series(i) + deltas(i));
    s_t vt_1
        = (series(i + 2) + deltas(i + 2)) - (series(i + 1) + deltas(i + 1));
    s_t vt_2
        = (series(i + 3) + deltas(i + 3)) - (series(i + 2) + deltas(i + 2));
    s_t at = vt_1 - vt;
    s_t at_1 = vt_2 - vt_1;
    s_t jt = at_1 - at;

    if (debug)
    {
      std::cout << "Jerk " << i << ": " << jt << std::endl;
      std::cout << "Mask " << i << ": "
                << (series.segment<4>(i) + deltas.segment<4>(i)).dot(jMask)
                << std::endl;
      Eigen::Matrix4s squareMask = jMask * jMask.transpose();
      std::cout << "Square mask: " << squareMask << std::endl;
      s_t sq = (series.segment<4>(i) + deltas.segment<4>(i)).transpose()
               * squareMask * (series.segment<4>(i) + deltas.segment<4>(i));
      std::cout << "Squared on mask: " << sq << std::endl;
      std::cout << "Manual: " << jt * jt << std::endl;
    }

    manual_score += mAlpha * jt * jt;
  }
  for (int i = 0; i < mTimesteps; i++)
  {
    manual_score += deltas(i) * deltas(i);
  }

  if (debug)
  {
    std::cout << "Matrix score: " << matrix_score << std::endl;
    // std::cout << "c: " << c << std::endl;
    std::cout << "Manual score: " << manual_score << std::endl;
  }

  return manual_score;
}

/**
 * This prints the stats for a time-series of data, with pos, vel, accel, and
 * jerk
 */
void AccelerationSmoother::debugTimeSeries(Eigen::VectorXs series)
{
  Eigen::MatrixXs cols = Eigen::MatrixXs::Zero(series.size() - 3, 4);
  for (int i = 0; i < series.size() - 3; i++)
  {
    s_t pt = series(i);
    s_t vt = series(i + 1) - series(i);
    s_t vt_1 = series(i + 2) - series(i + 1);
    s_t vt_2 = series(i + 3) - series(i + 2);
    s_t at = vt_1 - vt;
    s_t at_1 = vt_2 - vt_1;
    s_t jt = at - at_1;
    cols(i, 0) = pt;
    cols(i, 1) = vt;
    cols(i, 2) = at;
    cols(i, 3) = jt;
  }

  std::cout << "pos - vel - acc - jerk" << std::endl << cols << std::endl;
}

} // namespace utils
} // namespace dart