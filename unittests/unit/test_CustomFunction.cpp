#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/SimmSpline.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

//==============================================================================
TEST(CustomFunction, BASIC)
{
  std::vector<s_t> xs;
  std::vector<s_t> ys;

  xs.push_back(0.0);
  xs.push_back(1.0);
  xs.push_back(2.0);

  ys.push_back(0.0);
  ys.push_back(1.0);
  ys.push_back(0.0);

  CustomFunction* fn = new SimmSpline(xs, ys);

  EXPECT_EQ(fn->calcValue(0.0), 0.0);
  EXPECT_EQ(fn->calcValue(1.0), 1.0);
  EXPECT_EQ(fn->calcValue(2.0), 0.0);

  EXPECT_EQ(fn->calcDerivative(1, 1.0), 0.0);

  fn = new LinearFunction(1.0, 0.0);

  EXPECT_EQ(fn->calcValue(0.0), 0.0);
  EXPECT_EQ(fn->calcValue(1.0), 1.0);
  EXPECT_EQ(fn->calcValue(2.0), 2.0);
}