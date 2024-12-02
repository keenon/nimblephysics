#ifndef MATH_CONSTANTFN_H_
#define MATH_CONSTANTFN_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class ConstantFunction : public CustomFunction
{
public:
  ConstantFunction(s_t value);

  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;
  std::shared_ptr<CustomFunction> offsetBy(s_t y) const override;

public:
  s_t mValue;
};

} // namespace math
} // namespace dart

#endif