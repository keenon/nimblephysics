#include "dart/math/PolynomialFitter.hpp"

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

//=============================================================================
PolynomialFitter::PolynomialFitter(Eigen::VectorXs timesteps, int order)
{
  mForwardCoeffsMatrix = Eigen::MatrixXs::Zero(timesteps.size(), order + 1);
  for (int t = 0; t < timesteps.size(); t++)
  {
    s_t timestep = timesteps[t];
    s_t timestep_pow = 1.0;
    for (int i = 0; i < order + 1; i++)
    {
      mForwardCoeffsMatrix(t, i) = timestep_pow;
      timestep_pow *= timestep;
    }
  }
  mFactored = mForwardCoeffsMatrix.completeOrthogonalDecomposition();
}

//=============================================================================
Eigen::VectorXs PolynomialFitter::calcCoeffs(Eigen::VectorXs values) const
{
  return mFactored.solve(values);
}

//=============================================================================
Eigen::Vector3s PolynomialFitter::projectPosVelAccAtTime(
    s_t timestep, Eigen::VectorXs pastValues) const
{
  Eigen::VectorXs coeffs = calcCoeffs(pastValues);
  s_t pos = 0.0;
  s_t vel = 0.0;
  s_t acc = 0.0;

  for (int i = 0; i < coeffs.size(); i++)
  {
    pos += coeffs[i] * std::pow(timestep, i);
    if (i > 0)
    {
      vel += i * coeffs[i] * std::pow(timestep, i - 1);
    }
    if (i > 1)
    {
      acc += i * (i - 1) * coeffs[i] * std::pow(timestep, i - 2);
    }
  }

  return Eigen::Vector3s(pos, vel, acc);
}

} // namespace math
} // namespace dart