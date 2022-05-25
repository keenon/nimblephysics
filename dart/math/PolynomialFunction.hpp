#ifndef MATH_POLYFN_H_
#define MATH_POLYFN_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class PolynomialFunction : public CustomFunction
{
public:
  PolynomialFunction(std::vector<s_t> coeffs);

  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;
  std::shared_ptr<CustomFunction> offsetBy(s_t y) const override;

public:
  std::vector<s_t> mCoeffs;
};

} // namespace math
} // namespace dart

#endif