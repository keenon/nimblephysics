#include "dart/math/CustomFunction.hpp"

#include "dart/math/FiniteDifference.hpp"

//=============================================================================
//=============================================================================
namespace dart {
namespace math {

s_t CustomFunction::finiteDifferenceDerivative(int order, s_t x) const
{
  s_t EPS = 1e-3;
  s_t result = 0.0;

  dart::math::finiteDifference(
      // this should return if the perturbation was valid
      [&](
          /* in*/ s_t eps,
          /*out*/ s_t& perturbed) {
        if (order == 1)
        {
          perturbed = calcValue(x + eps);
        }
        else
        {
          perturbed = finiteDifferenceDerivative(order - 1, x + eps);
        }
        return true;
      },
      result,
      EPS,
      true);

  return result;
}

} // namespace math
} // namespace dart