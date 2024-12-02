#include "dart/math/PolynomialFunction.hpp"

namespace dart {
namespace math {

PolynomialFunction::PolynomialFunction(std::vector<s_t> coeffs)
  : mCoeffs(coeffs)
{
}

s_t PolynomialFunction::calcValue(s_t x) const
{
  s_t sum = 0.0;
  s_t x_pow = 1.0;
  for (int i = 0; i < mCoeffs.size(); i++)
  {
    sum += mCoeffs[i] * x_pow;
    x_pow *= x;
  }
  return sum;
}

s_t PolynomialFunction::calcDerivative(int order, s_t x) const
{
  s_t sum = 0.0;
  s_t x_pow = 1.0;
  for (int i = order; i < mCoeffs.size(); i++)
  {
    s_t multiple = i; // order 1
    // for order >1
    for (int j = 1; j < order; j++)
    {
      multiple *= (i - j);
    }
    sum += multiple * mCoeffs[i] * x_pow;
    x_pow *= x;
  }
  return sum;
}

std::shared_ptr<CustomFunction> PolynomialFunction::offsetBy(s_t y) const
{
  std::vector<s_t> newCoeffs = mCoeffs;
  newCoeffs[0] += y;
  return std::make_shared<PolynomialFunction>(newCoeffs);
}

} // namespace math
} // namespace dart