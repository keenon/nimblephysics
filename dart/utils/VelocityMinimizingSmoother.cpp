#include "VelocityMinimizingSmoother.hpp"

#include <iostream>

#include <Eigen/IterativeLinearSolvers>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

/**
 * Create (and pre-factor) a smoother that can remove the "jerk" from a time
 * seriese of data.
 *
 * The alpha value will determine how much smoothing to apply. A value of 0
 * corresponds to no smoothing.
 */
VelocityMinimizingSmoother::VelocityMinimizingSmoother(
    int timesteps,
    s_t smoothingWeight,
    s_t regularizationWeight,
    bool useSparse,
    bool useIterativeSolver)
  : mTimesteps(timesteps),
    mSmoothingWeight(smoothingWeight),
    mRegularizationWeight(regularizationWeight),
    mUseSparse(useSparse),
    mUseIterativeSolver(useIterativeSolver)
{
  Eigen::Vector2s stamp;
  stamp << -1, 1;
  stamp *= mSmoothingWeight;
  mSmoothedTimesteps = max(0, mTimesteps - 1);

  if (useSparse)
  {
    typedef Eigen::Triplet<s_t> T;
    std::vector<T> tripletList;
    for (int i = 0; i < mSmoothedTimesteps; i++)
    {
      for (int j = 0; j < 2; j++)
      {
        tripletList.push_back(T(i, i + j, stamp(j)));
      }
    }
    for (int i = 0; i < mTimesteps; i++)
    {
      tripletList.push_back(T(mSmoothedTimesteps + i, i, 1));
    }
    mB_sparse
        = Eigen::SparseMatrix<s_t>(mSmoothedTimesteps + mTimesteps, mTimesteps);
    mB_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
    mB_sparse.makeCompressed();
    if (!mUseIterativeSolver)
    {
      mB_sparseSolver.analyzePattern(mB_sparse);
      mB_sparseSolver.factorize(mB_sparse);
      if (mB_sparseSolver.info() != Eigen::Success)
      {
        std::cout << "mB_sparseSolver.factorize(mB_sparse) error: "
                  << mB_sparseSolver.lastErrorMessage() << std::endl;
      }
      assert(mB_sparseSolver.info() == Eigen::Success);
    }
  }
  else
  {
    mB = Eigen::MatrixXs::Zero(mSmoothedTimesteps + mTimesteps, mTimesteps);
    for (int i = 0; i < mSmoothedTimesteps; i++)
    {
      mB.block<1, 2>(i, i) = stamp;
    }
    mB.block(mSmoothedTimesteps, 0, mTimesteps, mTimesteps)
        = Eigen::MatrixXs::Identity(mTimesteps, mTimesteps);

    if (!mUseIterativeSolver)
    {
      mFactoredB = Eigen::HouseholderQR<Eigen::MatrixXs>(mB);
    }
  }
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
Eigen::MatrixXs VelocityMinimizingSmoother::smooth(Eigen::MatrixXs series)
{
  assert(series.cols() == mTimesteps);

  Eigen::MatrixXs smoothed = Eigen::MatrixXs::Zero(series.rows(), mTimesteps);

  for (int row = 0; row < series.rows(); row++)
  {
    Eigen::VectorXs c = Eigen::VectorXs::Zero(mSmoothedTimesteps + mTimesteps);
    c.segment(mSmoothedTimesteps, mTimesteps)
        = mRegularizationWeight * series.row(row);
    if (mUseIterativeSolver)
    {
      if (mUseSparse)
      {
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<s_t>> solver;
        solver.compute(mB_sparse);
        solver.setTolerance(1e-15);
        solver.setMaxIterations(100000);
        smoothed.row(row) = solver.solveWithGuess(c, series.row(row))
                            * (1.0 / mRegularizationWeight);
      }
      else
      {
        Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXs> cg;
        cg.compute(mB);
        cg.setTolerance(1e-15);
        cg.setMaxIterations(100000);
        smoothed.row(row) = cg.solveWithGuess(c, series.row(row))
                            * (1.0 / mRegularizationWeight);
      }
    }
    else
    {
      // Eigen::VectorXs deltas = mB.completeOrthogonalDecomposition().solve(c);
      if (mUseSparse)
      {
        smoothed.row(row)
            = mB_sparseSolver.solve(c) * (1.0 / mRegularizationWeight);
        assert(mB_sparseSolver.info() == Eigen::Success);
      }
      else
      {
        smoothed.row(row) = mFactoredB.solve(c) * (1.0 / mRegularizationWeight);
      }
    }
  }

  return smoothed;
};

/**
 * This computes the squared loss for this smoother, given a time series and a
 * set of perturbations `delta` to the time series.
 */
s_t VelocityMinimizingSmoother::getLoss(
    Eigen::MatrixXs series, Eigen::MatrixXs originalSeries, bool debug)
{
  s_t manual_score = 0.0;
  for (int row = 0; row < series.rows(); row++)
  {
    for (int i = 0; i < mSmoothedTimesteps; i++)
    {
      /*
      s_t vt = series(i + 1) - series(i);
      s_t vt_1 = series(i + 2) - series(i + 1);
      s_t vt_2 = series(i + 3) - series(i + 2);
      */
      s_t vt = series(row, i + 1) - series(row, i);
      s_t vtScaled = mSmoothingWeight * vt;

      if (debug)
      {
        std::cout << "Velocity " << i << ": " << vt << std::endl;
        std::cout << "Manual: " << vtScaled * vtScaled << std::endl;
      }

      manual_score += vtScaled * vtScaled;
    }

    for (int i = 0; i < mTimesteps; i++)
    {
      s_t diff = series(row, i) - originalSeries(row, i);
      diff *= mRegularizationWeight;
      manual_score += diff * diff;
    }

    if (debug)
    {
      std::cout << "Manual score: " << manual_score << std::endl;
    }
  }

  return manual_score;
}

/**
 * This prints the stats for a time-series of data, with pos, vel, accel, and
 * jerk
 */
void VelocityMinimizingSmoother::debugTimeSeries(Eigen::VectorXs series)
{
  Eigen::MatrixXs cols = Eigen::MatrixXs::Zero(series.size() - 1, 2);
  for (int i = 0; i < series.size() - 1; i++)
  {
    s_t pt = series(i);
    s_t vt = series(i + 1) - series(i);
    cols(i, 0) = pt;
    cols(i, 1) = vt;
  }

  std::cout << "pos - vel" << std::endl << cols << std::endl;
}

} // namespace utils
} // namespace dart