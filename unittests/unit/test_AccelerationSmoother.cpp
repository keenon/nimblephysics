#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/math/MathTypes.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(ACCEL_SMOOTHER, SCORING)
{
  int dofs = 1;
  int timesteps = 5;
  AccelerationSmoother smoother(timesteps, 1.0, 1.0);

  Eigen::MatrixXs series = Eigen::MatrixXs::Random(dofs, timesteps);
  Eigen::MatrixXs perturb = Eigen::MatrixXs::Random(dofs, timesteps) * 0.01;

  smoother.getLoss(series, perturb, true);
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_SMOOTHER, BASIC)
{
  int dofs = 1;
  int timesteps = 10;
  Eigen::MatrixXs data = Eigen::MatrixXs::Random(dofs, timesteps);

  AccelerationSmoother smoother(timesteps, 1, 0.05);
  Eigen::MatrixXs smoothed = smoother.smooth(data);
  std::cout << "Raw data: " << std::endl;
  smoother.debugTimeSeries(data.row(0));
  std::cout << "Smoothed data: " << std::endl;
  smoother.debugTimeSeries(smoothed.row(0));

  EXPECT_EQ(smoothed.cols(), timesteps);
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_SMOOTHER, SPARSE_V_DENSE)
{
  int dofs = 1;
  int timesteps = 10;
  Eigen::MatrixXs data = Eigen::MatrixXs::Random(dofs, timesteps);

  AccelerationSmoother smootherSparse(timesteps, 1, 0.05, true);
  AccelerationSmoother smootherDense(timesteps, 1, 0.05, false);
  Eigen::MatrixXs smoothedSparse = smootherSparse.smooth(data);
  Eigen::MatrixXs smoothedDense = smootherDense.smooth(data);
  if (!equals(smoothedDense, smoothedSparse, 1e-8))
  {
    std::cout << "Smoothed sparse vs. Smoothed dense produce different results!"
              << std::endl;
    std::cout << "Smoothed dense:" << std::endl << smoothedDense << std::endl;
    std::cout << "Smoothed sparse:" << std::endl << smoothedSparse << std::endl;
    std::cout << "Diff:" << std::endl
              << (smoothedDense - smoothedSparse) << std::endl;
    EXPECT_TRUE(equals(smoothedDense, smoothedSparse, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_SMOOTHER, HUGE_TRAJECTORY)
{
  int dofs = 1;
  int timesteps = 300;
  Eigen::MatrixXs data = Eigen::MatrixXs::Random(dofs, timesteps);

  AccelerationSmoother smootherIterative(timesteps, 1, 0.05, true, true);
  Eigen::MatrixXs smoothedIterative = smootherIterative.smooth(data);

  AccelerationSmoother smootherSparse(timesteps, 1, 0.05, true, false);
  Eigen::MatrixXs smoothedSparse = smootherSparse.smooth(data);

  std::cout << "Diff: " << (smoothedIterative - smoothedSparse).squaredNorm()
            << std::endl;
}
#endif