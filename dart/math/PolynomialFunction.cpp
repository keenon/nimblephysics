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
    s_t multiple = 0.0;
    if (order == 1)
    {
      multiple = i;
    }
    else if (order == 2)
    {
      multiple = (i - 1) * i;
    }
    sum += multiple * mCoeffs[i] * x_pow;
    x_pow *= x;
  }
  return sum;
}

} // namespace math
} // namespace dart