#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/math/MathTypes.hpp"
#include "dart/utils/AccelerationMinimizer.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(ACCEL_MINIMIZER, DOES_NOT_CRASH)
{
  int timesteps = 50;
  AccelerationMinimizer minimizer(timesteps, 1.0, 1.0);

  Eigen::VectorXs series = Eigen::VectorXs::Random(timesteps);

  Eigen::VectorXs x = minimizer.minimize(series);

  std::cout << "Finished" << std::endl;
  std::cout << x << std::endl;
}
#endif