#include <vector>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/PolynomialFitter.hpp"
#include "dart/math/PolynomialFunction.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

//==============================================================================
TEST(PolynomialFitter, BASIC_LINEAR)
{
  std::vector<s_t> coeffs;

  coeffs.push_back(2.0);
  coeffs.push_back(1.0);
  PolynomialFunction poly(coeffs);
  Eigen::VectorXs timestamps = Eigen::VectorXs::Random(4);
  Eigen::VectorXs values = Eigen::VectorXs::Zero(4);
  for (int i = 0; i < 4; i++)
  {
    values(i) = poly.calcValue(timestamps[i]);
  }

  PolynomialFitter fitter(timestamps, coeffs.size() - 1);
  Eigen::VectorXs recoveredCoeffs = fitter.calcCoeffs(values);

  for (int i = 0; i < coeffs.size(); i++)
  {
    EXPECT_NEAR(coeffs[i], recoveredCoeffs(i), 1e-6);
  }
}

//==============================================================================
TEST(PolynomialFitter, HIGHER_ORDER)
{
  Eigen::VectorXs originalCoeffsVector = Eigen::VectorXs::Random(10);
  std::vector<s_t> coeffs;
  for (int i = 0; i < originalCoeffsVector.size(); i++)
  {
    coeffs.push_back(originalCoeffsVector(i));
  }

  PolynomialFunction poly(coeffs);
  Eigen::VectorXs timestamps = Eigen::VectorXs::Random(20);
  Eigen::VectorXs values = Eigen::VectorXs::Zero(timestamps.size());
  for (int i = 0; i < timestamps.size(); i++)
  {
    values(i) = poly.calcValue(timestamps[i]);
  }

  PolynomialFitter fitter(timestamps, coeffs.size() - 1);
  Eigen::VectorXs recoveredCoeffs = fitter.calcCoeffs(values);

  for (int i = 0; i < coeffs.size(); i++)
  {
    EXPECT_NEAR(coeffs[i], recoveredCoeffs(i), 1e-6);
  }
}

//==============================================================================
TEST(PolynomialFitter, UNDERCONSTRAINED_RECOVERS_POINTS)
{
  Eigen::VectorXs originalCoeffsVector = Eigen::VectorXs::Random(10);
  std::vector<s_t> coeffs;
  for (int i = 0; i < originalCoeffsVector.size(); i++)
  {
    coeffs.push_back(originalCoeffsVector(i));
  }

  PolynomialFunction poly(coeffs);
  Eigen::VectorXs timestamps = Eigen::VectorXs::Random(5);
  Eigen::VectorXs values = Eigen::VectorXs::Zero(timestamps.size());
  for (int i = 0; i < timestamps.size(); i++)
  {
    values(i) = poly.calcValue(timestamps[i]);
  }

  PolynomialFitter fitter(timestamps, coeffs.size() - 1);
  Eigen::VectorXs recoveredCoeffs = fitter.calcCoeffs(values);

  std::vector<s_t> recoveredCoeffsVector;
  for (int i = 0; i < recoveredCoeffs.size(); i++)
  {
    recoveredCoeffsVector.push_back(recoveredCoeffs(i));
  }
  PolynomialFunction recoveredPoly(recoveredCoeffsVector);
  for (int i = 0; i < timestamps.size(); i++)
  {
    EXPECT_NEAR(recoveredPoly.calcValue(timestamps[i]), values(i), 1e-6);
  }
}

//==============================================================================
TEST(PolynomialFitter, PROJECT_DERIVATIVE)
{
  Eigen::VectorXs originalCoeffsVector = Eigen::VectorXs::Random(10);
  std::vector<s_t> coeffs;
  for (int i = 0; i < originalCoeffsVector.size(); i++)
  {
    coeffs.push_back(originalCoeffsVector(i));
  }

  PolynomialFunction poly(coeffs);
  Eigen::VectorXs timestamps = Eigen::VectorXs::Random(20);
  Eigen::VectorXs values = Eigen::VectorXs::Zero(timestamps.size());
  for (int i = 0; i < timestamps.size(); i++)
  {
    values(i) = poly.calcValue(timestamps[i]);
  }

  PolynomialFitter fitter(timestamps, coeffs.size() - 1);

  Eigen::Vector3s posVelAcc
      = fitter.projectPosVelAccAtTime(timestamps[0], values);
  EXPECT_TRUE(abs(values(0) - posVelAcc(0)) < 1e-8);
}