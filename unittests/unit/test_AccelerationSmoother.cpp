#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/utils/AccelerationSmoother.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

TEST(ACCEL_SMOOTHER, SCORING)
{
  int timesteps = 5;
  AccelerationSmoother smoother(timesteps, 1.0);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);
  Eigen::VectorXs perturb = Eigen::VectorXs::Random(timesteps) * 0.01;

  smoother.getLoss(series, perturb, true);
}

TEST(ACCEL_SMOOTHER, BASIC)
{
  int dofs = 1;
  int timesteps = 10;
  Eigen::MatrixXs data = Eigen::MatrixXs::Random(dofs, timesteps);

  AccelerationSmoother smoother(timesteps, 0.05);
  Eigen::MatrixXs smoothed = smoother.smooth(data);
  std::cout << "Raw data: " << std::endl;
  smoother.debugTimeSeries(data.row(0));
  std::cout << "Smoothed data: " << std::endl;
  smoother.debugTimeSeries(smoothed.row(0));

  EXPECT_EQ(smoothed.cols(), timesteps - 3);
}