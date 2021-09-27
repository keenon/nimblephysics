#ifndef MATH_TESTBENCH_FN_H_
#define MATH_TESTBENCH_FN_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

/**
 * This class exists to make (fuzz) testing CustomJoint easier. It can be
 * constructed to return any `y` and its first and second derivatives.
 */
class TestBedFunction : public CustomFunction
{
public:
  TestBedFunction(s_t y, s_t dy, s_t ddy);

  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;

protected:
  s_t y;
  s_t dy;
  s_t ddy;
};

} // namespace math
} // namespace dart

#endif