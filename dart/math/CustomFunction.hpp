#ifndef MATH_CUSTOMFN_H_
#define MATH_CUSTOMFN_H_

#include "dart/math/MathTypes.hpp"

//=============================================================================
//=============================================================================
namespace dart {
namespace math {

class CustomFunction
{
public:
  virtual ~CustomFunction(){};

  virtual s_t calcValue(s_t x) const = 0;
  virtual s_t calcDerivative(int order, s_t x) const = 0;
  s_t finiteDifferenceDerivative(int order, s_t x) const;
};

} // namespace math
} // namespace dart

#endif