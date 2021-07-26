#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/SimmSpline.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

//==============================================================================
TEST(SimmSpline, BASIC)
{
  std::vector<s_t> xs;
  std::vector<s_t> ys;

  xs.push_back(0.0);
  xs.push_back(1.0);
  xs.push_back(2.0);

  ys.push_back(0.0);
  ys.push_back(1.0);
  ys.push_back(0.0);

  SimmSpline spline(xs, ys);

  EXPECT_EQ(spline.calcValue(0.0), 0.0);
  EXPECT_EQ(spline.calcValue(1.0), 1.0);
  EXPECT_EQ(spline.calcValue(2.0), 0.0);

  EXPECT_EQ(spline.calcDerivative(1, 1.0), 0.0);
}