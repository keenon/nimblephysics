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

//==============================================================================
TEST(SimmSpline, GRAD_AT_EDGES)
{
  std::vector<s_t> rotZ_x;
  rotZ_x.push_back(0);
  rotZ_x.push_back(0.174533);
  rotZ_x.push_back(0.349066);
  rotZ_x.push_back(0.523599);
  rotZ_x.push_back(0.698132);
  rotZ_x.push_back(0.872665);
  rotZ_x.push_back(1.0472);
  rotZ_x.push_back(1.22173);
  rotZ_x.push_back(1.39626);
  rotZ_x.push_back(1.5708);
  rotZ_x.push_back(1.74533);
  rotZ_x.push_back(1.91986);
  rotZ_x.push_back(2.0944);
  std::vector<s_t> rotZ_y;
  rotZ_y.push_back(0.0);
  rotZ_y.push_back(0.0126809);
  rotZ_y.push_back(0.0226969);
  rotZ_y.push_back(0.0296054);
  rotZ_y.push_back(0.0332049);
  rotZ_y.push_back(0.0335354);
  rotZ_y.push_back(0.0308779);
  rotZ_y.push_back(0.0257548);
  rotZ_y.push_back(0.0189295);
  rotZ_y.push_back(0.011407);
  rotZ_y.push_back(0.00443314);
  rotZ_y.push_back(-0.00050475);
  rotZ_y.push_back(-0.0016782);

  math::SimmSpline rotZ(rotZ_x, rotZ_y);

  s_t dx = rotZ.calcDerivative(1, 0.0);
  s_t dx_fd = rotZ.finiteDifferenceFirstDerivative(0.0);
  s_t err = std::abs(dx - dx_fd);

  if (err > 1e-10)
  {
    std::cout << "SimmSpline at 0 (left edge of spline): " << std::endl;
    std::cout << "Analytical Grad: " << dx << std::endl;
    std::cout << "FD Grad: " << dx_fd << std::endl;
    std::cout << "Diff: " << dx - dx_fd << std::endl;
    EXPECT_LE(err, 1e-10);
  }
}