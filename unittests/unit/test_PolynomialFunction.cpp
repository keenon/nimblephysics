#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/PolynomialFunction.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

//==============================================================================
TEST(Polynomial, BASIC_LINEAR)
{
  std::vector<s_t> coeffs;

  // 2 + x

  coeffs.push_back(2.0);
  coeffs.push_back(1.0);

  PolynomialFunction poly(coeffs);

  EXPECT_EQ(poly.calcValue(0.0), 2.0);
  EXPECT_EQ(poly.calcValue(1.0), 3.0);
  EXPECT_EQ(poly.calcValue(2.0), 4.0);

  // 1

  EXPECT_EQ(poly.calcDerivative(1, 1.0), 1.0);
  EXPECT_EQ(poly.calcDerivative(1, 2.0), 1.0);

  // 0

  EXPECT_EQ(poly.calcDerivative(2, 1.0), 0.0);
  EXPECT_EQ(poly.calcDerivative(2, 2.0), 0.0);
}

//==============================================================================
TEST(Polynomial, SIMPLE_POLY)
{
  std::vector<s_t> coeffs;

  // x + 2*x^2

  coeffs.push_back(0.0);
  coeffs.push_back(1.0);
  coeffs.push_back(2.0);

  PolynomialFunction poly(coeffs);

  EXPECT_EQ(poly.calcValue(0.0), 0.0);
  EXPECT_EQ(poly.calcValue(1.0), 3.0);
  EXPECT_EQ(poly.calcValue(2.0), 10.0);

  // 1 + 4x

  EXPECT_EQ(poly.calcDerivative(1, 1.0), 5.0);
  EXPECT_EQ(poly.calcDerivative(1, 2.0), 9.0);

  // 4

  EXPECT_EQ(poly.calcDerivative(2, 1.0), 4.0);
  EXPECT_EQ(poly.calcDerivative(2, 2.0), 4.0);
}

//==============================================================================
TEST(Polynomial, COMPLEX_POLY)
{
  std::vector<s_t> coeffs;
  Eigen::VectorXs coeffsVec = Eigen::VectorXs::Random(4) * 0.1;
  for (int i = 0; i < coeffsVec.size(); i++)
  {
    coeffs.push_back(coeffsVec(i));
  }

  PolynomialFunction poly(coeffs);

  s_t THRESHOLD = 3e-5;

  for (s_t x = -1.0; x < 1.0; x += 0.027)
  {
    s_t dx = poly.calcDerivative(1, x);
    s_t dx_fd = poly.finiteDifferenceDerivative(1, x);
    const s_t dx_threshold = max(1.0, abs(dx)) * THRESHOLD;
    EXPECT_NEAR(dx, dx_fd, dx_threshold);

    s_t ddx = poly.calcDerivative(2, x);
    s_t ddx_fd = poly.finiteDifferenceDerivative(2, x);
    const s_t ddx_threshold = max(1.0, abs(ddx)) * THRESHOLD;
    if (abs(ddx - ddx_fd) > ddx_threshold)
    {
      std::cout << "Error at " << x << " on ddx." << std::endl;
      EXPECT_NEAR(ddx, ddx_fd, ddx_threshold);
      return;
    }

    s_t dddx = poly.calcDerivative(3, x);
    s_t dddx_fd = poly.finiteDifferenceDerivative(3, x);
    const s_t dddx_threshold = max(1.0, abs(dddx)) * THRESHOLD;
    if (abs(dddx - dddx_fd) > dddx_threshold)
    {
      std::cout << "Error at " << x << " on dddx." << std::endl;
      EXPECT_NEAR(dddx, dddx_fd, dddx_threshold);
      return;
    }
  }

  /*
  for (s_t x = -10; x < 10; x++)
  {
    s_t dx = poly.calcDerivative(1, x);
    s_t dx_fd = poly.finiteDifferenceDerivative(1, x);
    EXPECT_NEAR(dx, dx_fd, abs(dx) * 1e-8);

    s_t ddx = poly.calcDerivative(2, x);
    s_t ddx_fd = poly.finiteDifferenceDerivative(2, x);
    EXPECT_NEAR(ddx, ddx_fd, abs(ddx) * 1e-8);

    s_t dddx = poly.calcDerivative(3, x);
    s_t dddx_fd = poly.finiteDifferenceDerivative(3, x);
    EXPECT_NEAR(dddx, dddx_fd, abs(dddx) * 1e-8);
  }
  */
}