#include "dart/math/ConstantFunction.hpp"

#include <memory>

namespace dart {
namespace math {

ConstantFunction::ConstantFunction(s_t value) : mValue(value)
{
}

s_t ConstantFunction::calcValue(s_t /* x*/) const
{
  return mValue;
}

s_t ConstantFunction::calcDerivative(int /* order */, s_t /* x */) const
{
  return 0.0;
}

std::shared_ptr<CustomFunction> ConstantFunction::offsetBy(s_t y) const
{
  return std::make_shared<ConstantFunction>(mValue + y);
}

} // namespace math
} // namespace dart