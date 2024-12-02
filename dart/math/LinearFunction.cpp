#include "dart/math/LinearFunction.hpp"

#include <memory>

namespace dart {
namespace math {

LinearFunction::LinearFunction(s_t slope, s_t yIntercept)
  : mSlope(slope), mYIntercept(yIntercept)
{
}

s_t LinearFunction::calcValue(s_t x) const
{
  return (mSlope * x) + mYIntercept;
}

s_t LinearFunction::calcDerivative(int order, s_t /* x */) const
{
  if (order == 1)
    return mSlope;
  return 0.0;
}

std::shared_ptr<CustomFunction> LinearFunction::offsetBy(s_t y) const
{
  return std::make_shared<LinearFunction>(mSlope, mYIntercept + y);
}

} // namespace math
} // namespace dart