#include "dart/math/ConstantFunction.hpp"

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

} // namespace math
} // namespace dart