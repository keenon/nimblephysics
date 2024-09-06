#include "AccelerationMinimizer.hpp"

#include <iostream>

// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
// #include <unsupported/Eigen/IterativeSolvers>

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
AccelerationMinimizer::AccelerationMinimizer(
    int timesteps,
    s_t smoothingWeight,
    s_t regularizationWeight,
    int numIterations)
  : mTimesteps(timesteps),
    mSmoothingWeight(smoothingWeight),
    mRegularizationWeight(regularizationWeight),
    mNumIterations(numIterations),
    mNumIterationsBackoff(6),
    mDebugIterationBackoff(false),
    mConvergenceTolerance(1e-10)
{
  Eigen::Vector3s stamp;
  stamp << -1, 2, -1;
  stamp *= mSmoothingWeight;

  typedef Eigen::Triplet<s_t> T;
  std::vector<T> tripletList;
  const int accTimesteps = mTimesteps - 2;
  for (int i = 0; i < accTimesteps; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      tripletList.push_back(T(i, i + j, stamp(j)));
    }
  }
  for (int i = 0; i < mTimesteps; i++)
  {
    tripletList.push_back(T(accTimesteps + i, i, mRegularizationWeight));
  }
  mB_sparse = Eigen::SparseMatrix<s_t>(accTimesteps + mTimesteps, mTimesteps);
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

Eigen::VectorXs AccelerationMinimizer::minimize(Eigen::VectorXs series)
{
  const int accTimesteps = mTimesteps - 2;
  Eigen::VectorXs b = Eigen::VectorXs(accTimesteps + mTimesteps);
  b.segment(0, accTimesteps).setZero();
  b.segment(accTimesteps, mTimesteps) = series * mRegularizationWeight;

  Eigen::VectorXs x = series;

  int iterations = mNumIterations;
  for (int i = 0; i < mNumIterationsBackoff; i++)
  {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<s_t>> solver;
    solver.compute(mB_sparse);
    solver.setTolerance(mConvergenceTolerance);
    solver.setMaxIterations(iterations);
    x = solver.solveWithGuess(b, series);
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
        std::cout
            << "[AccelerationMinimizer] LeastSquaresConjugateGradient did "
               "not converge in "
            << iterations << ", with error " << solver.error()
            << " so doubling iteration count and trying again." << std::endl;
      }
      iterations *= 2;
    }
  }

  return x;
}

void AccelerationMinimizer::setDebugIterationBackoff(bool debug)
{
  mDebugIterationBackoff = debug;
}

void AccelerationMinimizer::setNumIterationsBackoff(int numIterations)
{
  mNumIterationsBackoff = numIterations;
}

void AccelerationMinimizer::setConvergenceTolerance(s_t tolerance)
{
  mConvergenceTolerance = tolerance;
}

} // namespace utils
} // namespace dart