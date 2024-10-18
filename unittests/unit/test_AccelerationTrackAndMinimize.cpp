#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/math/MathTypes.hpp"
#include "dart/utils/AccelerationMinimizer.hpp"
#include "dart/utils/AccelerationTrackAndMinimize.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(ACCEL_TRACK_AND_MINIMIZE, DOES_NOT_CRASH)
{
  int timesteps = 50;
  std::vector<bool> trackAcc = std::vector<bool>(timesteps, false);
  for (int i = 0; i < timesteps; i++)
  {
    if (i % 2 == 0)
    {
      trackAcc[i] = true;
    }
  }
  AccelerationTrackAndMinimize minimizer(timesteps, trackAcc, 1.0, 1.0);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);
  Eigen::VectorXs track = Eigen::VectorXs::Random(timesteps);
  for (int i = 0; i < timesteps; i++)
  {
    if (!trackAcc[i])
    {
      track[i] = 0.0;
    }
  }

  Eigen::VectorXs x = minimizer.minimize(series, track).series;

  std::cout << "Finished" << std::endl;
  std::cout << x << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_TRACK_AND_MINIMIZE, REDUCES_TO_ACC_MIN)
{
  int timesteps = 50;
  std::vector<bool> trackAcc = std::vector<bool>(timesteps, false);
  s_t smoothingWeight = 1.0;
  s_t regularizationWeight = 0.01;
  s_t dt = 1.0;
  AccelerationTrackAndMinimize minimizer(
      timesteps, trackAcc, smoothingWeight, 0.0, regularizationWeight, dt);
  AccelerationMinimizer minimizer2(timesteps, 1.0, 0.01);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);
  Eigen::VectorXs track = Eigen::VectorXs::Random(timesteps);
  for (int i = 0; i < timesteps; i++)
  {
    if (!trackAcc[i])
    {
      track[i] = 0.0;
    }
  }

  Eigen::VectorXs x = minimizer.minimize(series, track).series;
  Eigen::VectorXs x2 = minimizer2.minimize(series);

  s_t dist = (x - x2).norm();
  if (dist > 1e-8)
  {
    std::cout << "x: " << x << std::endl;
    std::cout << "x2: " << x2 << std::endl;
  }
  EXPECT_TRUE(dist < 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_TRACK_AND_MINIMIZE, PERFECTLY_TRACKS_ACC)
{
  int timesteps = 10;
  std::vector<bool> trackAcc = std::vector<bool>(timesteps, true);
  s_t dt = 0.1;
  AccelerationTrackAndMinimize minimizer(
      timesteps, trackAcc, 0.0, 1.0, 0.0, dt);
  minimizer.setDebugIterationBackoff(true);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);
  Eigen::VectorXs acc = Eigen::VectorXs::Zero(timesteps);
  for (int i = 1; i < timesteps - 1; i++)
  {
    acc[i] = (series[i + 1] - 2 * series[i] + series[i - 1]) / (dt * dt);
  }
  acc[0] = acc[1];
  acc[timesteps - 1] = acc[timesteps - 2];

  Eigen::VectorXs initialization = Eigen::VectorXs::Random(timesteps);

  Eigen::VectorXs x = minimizer.minimize(initialization, acc).series;

  Eigen::VectorXs recoveredAcc = Eigen::VectorXs::Zero(timesteps);
  for (int i = 1; i < timesteps - 1; i++)
  {
    recoveredAcc[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (dt * dt);
  }
  recoveredAcc[0] = recoveredAcc[1];
  recoveredAcc[timesteps - 1] = recoveredAcc[timesteps - 2];
  s_t dist = (recoveredAcc - acc).norm();
  if (dist > 1e-8)
  {
    Eigen::MatrixXs compare = Eigen::MatrixXs(timesteps, 3);
    compare.col(0) = acc;
    compare.col(1) = recoveredAcc;
    compare.col(2) = acc - recoveredAcc;
    std::cout << "acc - recovered acc - diff: " << std::endl;
    std::cout << compare << std::endl;
  }
  EXPECT_TRUE(dist < 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(ACCEL_TRACK_AND_MINIMIZE, TRACKS_OFFSET_ACCS)
{
  int timesteps = 10;
  std::vector<bool> trackAcc = std::vector<bool>(timesteps, true);
  s_t dt = 0.1;
  AccelerationTrackAndMinimize minimizer(
      timesteps, trackAcc, 1.0, 1.0, 1.0, dt);
  minimizer.setDebugIterationBackoff(true);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);
  Eigen::VectorXs acc = Eigen::VectorXs::Zero(timesteps);
  for (int i = 1; i < timesteps - 1; i++)
  {
    acc[i] = (series[i + 1] - 2 * series[i] + series[i - 1]) / (dt * dt);
  }
  acc[0] = acc[1];
  acc[timesteps - 1] = acc[timesteps - 2];

  Eigen::VectorXs offsetAcc = acc + Eigen::VectorXs::Ones(timesteps) * 0.5;

  AccelerationTrackingResult result = minimizer.minimize(series, offsetAcc);
  Eigen::VectorXs x = result.series;
  s_t offset = result.accelerationOffset;
  EXPECT_NEAR(offset, 0.5, 1e-6);

  Eigen::VectorXs recoveredAcc = Eigen::VectorXs::Zero(timesteps);
  for (int i = 1; i < timesteps - 1; i++)
  {
    recoveredAcc[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (dt * dt);
  }
  recoveredAcc[0] = recoveredAcc[1];
  recoveredAcc[timesteps - 1] = recoveredAcc[timesteps - 2];
  s_t dist = (recoveredAcc - acc).norm();
  if (dist > 1e-6)
  {
    Eigen::MatrixXs compare = Eigen::MatrixXs(timesteps, 3);
    compare.col(0) = acc;
    compare.col(1) = recoveredAcc;
    compare.col(2) = acc - recoveredAcc;
    std::cout << "acc - recovered acc - diff: " << std::endl;
    std::cout << compare << std::endl;
  }
  EXPECT_TRUE(dist < 1e-6);
}
#endif