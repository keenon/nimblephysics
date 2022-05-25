#ifndef MATH_LINEARFN_H_
#define MATH_LINEARFN_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class LinearFunction : public CustomFunction
{
public:
  LinearFunction(s_t slope, s_t yIntercept);

  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;
  std::shared_ptr<CustomFunction> offsetBy(s_t y) const override;

public:
  s_t mSlope;
  s_t mYIntercept;
};

} // namespace math
} // namespace dart

#endif