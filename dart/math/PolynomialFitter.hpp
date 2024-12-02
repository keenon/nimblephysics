#ifndef MATH_POLYFIT_H_
#define MATH_POLYFIT_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class PolynomialFitter
{
public:
  PolynomialFitter(Eigen::VectorXs timesteps, int order);
  Eigen::VectorXs calcCoeffs(Eigen::VectorXs values) const;
  std::vector<int> getOutlierIndices(
      Eigen::VectorXs values, int maxOutlierCount = 2) const;
  Eigen::Vector3s projectPosVelAccAtTime(
      s_t timestep, Eigen::VectorXs pastValues) const;

protected:
  Eigen::VectorXs mTimesteps;
  Eigen::MatrixXs mForwardCoeffsMatrix;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> mFactored;
  int mOrder;
};

} // namespace math
} // namespace dart

#endif