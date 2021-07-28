#include "dart/math/LinearFunction.hpp"

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

} // namespace math
} // namespace dart