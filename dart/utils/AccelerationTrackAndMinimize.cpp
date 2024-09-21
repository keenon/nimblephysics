#include "AccelerationTrackAndMinimize.hpp"

#include <iostream>

// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
// #include <unsupported/Eigen/IterativeSolvers>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

AccelerationTrackAndMinimize::AccelerationTrackAndMinimize(
    int numTimesteps,
    std::vector<bool> trackAccelerationAtTimesteps,
    s_t zeroUnobservedAccWeight,
    s_t trackObservedAccWeight,
    s_t regularizationWeight,
    s_t dt,
    int numIterations)
  : mTimesteps(numTimesteps),
    mDt(dt),
    mTrackAccelerationAtTimesteps(trackAccelerationAtTimesteps),
    mTrackObservedAccWeight(trackObservedAccWeight),
    mZeroUnobservedAccWeight(zeroUnobservedAccWeight),
    mRegularizationWeight(regularizationWeight),
    mNumIterations(numIterations),
    mNumIterationsBackoff(6),
    mDebugIterationBackoff(false),
    mConvergenceTolerance(1e-10)
{
  if (mTimesteps < 3)
  {
    std::cout << "AccelerationTrackAndMinimize requires at least 3 timesteps"
              << std::endl;
    return;
  }
  if (mTrackAccelerationAtTimesteps.size() != mTimesteps)
  {
    std::cout << "trackAccelerationAtTimesteps.size() != mTimesteps"
              << std::endl;
    return;
  }

  Eigen::Vector3s stamp;
  stamp << 1, -2, 1;
  stamp *= 1.0 / (mDt * mDt);

  typedef Eigen::Triplet<s_t> T;
  std::vector<T> tripletList;
  const int accTimesteps = mTimesteps - 2;
  for (int i = 0; i < accTimesteps; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (trackAccelerationAtTimesteps[i + 1])
      {
        tripletList.push_back(T(i, i + j, stamp(j) * mTrackObservedAccWeight));
      }
      else
      {
        tripletList.push_back(T(i, i + j, stamp(j) * mZeroUnobservedAccWeight));
      }
    }
  }
  for (int i = 0; i < mTimesteps; i++)
  {
    tripletList.push_back(T(accTimesteps + i, i, mRegularizationWeight));
  }
  // Add one extra input (to solve for) which applies a linear offset to the
  // tracked accelerations
  for (int i = 0; i < accTimesteps; i++)
  {
    if (trackAccelerationAtTimesteps[i + 1])
    {
      tripletList.push_back(T(i, mTimesteps, 1.0));
    }
  }
  mB_sparse
      = Eigen::SparseMatrix<s_t>(accTimesteps + mTimesteps, mTimesteps + 1);
  mB_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
  mB_sparse.makeCompressed();
  mB_sparseSolver.analyzePattern(mB_sparse);
  mB_sparseSolver.factorize(mB_sparse);
  if (mB_sparseSolver.info() != Eigen::Success)
  {
    std::cout << "mB_sparseSolver.factorize(mB_sparse) error: "
              << mB_sparseSolver.lastErrorMessage() << std::endl;
  }
  assert(mB_sparseSolver.info() == Eigen::Success);
}

AccelerationTrackingResult AccelerationTrackAndMinimize::minimize(
    Eigen::VectorXs series, Eigen::VectorXs trackAcc)
{
  AccelerationTrackingResult result;
  result.series = series;
  result.accelerationOffset = 0.0;
  if (series.size() != mTimesteps)
  {
    std::cout << "series.size() != mTimesteps" << std::endl;
    return result;
  }
  if (trackAcc.size() != mTimesteps)
  {
    std::cout << "trackAcc.size() != mTimesteps" << std::endl;
    return result;
  }

  const int accTimesteps = mTimesteps - 2;
  Eigen::VectorXs b = Eigen::VectorXs(accTimesteps + mTimesteps);
  b.segment(0, accTimesteps) = trackAcc.segment(1, accTimesteps);
  for (int i = 0; i < accTimesteps; i++)
  {
    if (!mTrackAccelerationAtTimesteps[i + 1])
    {
      if (b(i) != 0)
      {
        std::cout << "Warning: trackAcc[" << i << "] is non-zero, but we're "
                  << "not tracking acceleration at this timestep. Setting it "
                  << "to zero. Check how you are formatting your input array!"
                  << std::endl;
      }
      b(i) = 0;
    }
    else
    {
      b(i) *= mTrackObservedAccWeight;
    }
  }
  b.segment(accTimesteps, mTimesteps) = series * mRegularizationWeight;

  Eigen::VectorXs paddedSeries = Eigen::VectorXs::Zero(mTimesteps + 1);
  paddedSeries.head(mTimesteps) = series;

  int iterations = mNumIterations;
  for (int i = 0; i < mNumIterationsBackoff; i++)
  {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<s_t>> solver;
    solver.compute(mB_sparse);
    solver.setTolerance(mConvergenceTolerance);
    solver.setMaxIterations(iterations);
    paddedSeries = solver.solveWithGuess(b, paddedSeries);
    // Check convergence
    if (solver.info() == Eigen::Success)
    {
      // Converged
      break;
    }
    else
    {
      if (mDebugIterationBackoff)
      {
        std::cout << "[AccelerationTrackAndMinimize] "
                     "LeastSquaresConjugateGradient did "
                     "not converge in "
                  << iterations << ", with error " << solver.error()
                  << " so doubling iteration count and trying again."
                  << std::endl;
      }
      iterations *= 2;
    }
  }

  result.series = paddedSeries.head(mTimesteps);
  result.accelerationOffset
      = paddedSeries(mTimesteps) / mTrackObservedAccWeight;

  return result;
}

void AccelerationTrackAndMinimize::setDebugIterationBackoff(bool debug)
{
  mDebugIterationBackoff = debug;
}

void AccelerationTrackAndMinimize::setNumIterationsBackoff(int numIterations)
{
  mNumIterationsBackoff = numIterations;
}

void AccelerationTrackAndMinimize::setConvergenceTolerance(s_t tolerance)
{
  mConvergenceTolerance = tolerance;
}

} // namespace utils
} // namespace dart