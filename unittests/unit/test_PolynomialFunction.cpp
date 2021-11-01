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