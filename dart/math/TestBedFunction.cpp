#include "dart/math/TestBedFunction.hpp"

namespace dart {
namespace math {

TestBedFunction::TestBedFunction(s_t y, s_t dy, s_t ddy)
  : y(y), dy(dy), ddy(ddy)
{
}

s_t TestBedFunction::calcValue(s_t x) const
{
  return y + dy * x + 0.5 * ddy * x * x;
}

s_t TestBedFunction::calcDerivative(int order, s_t x) const
{
  if (order == 1)
    return dy + x * ddy;
  if (order == 2)
    return ddy;
  return 0.0;
}

std::shared_ptr<CustomFunction> TestBedFunction::offsetBy(s_t offset) const
{
  return std::make_shared<TestBedFunction>(y + offset, dy, ddy);
}

} // namespace math
} // namespace dart