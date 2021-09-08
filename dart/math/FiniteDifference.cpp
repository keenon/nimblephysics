#include "dart/math/FiniteDifference.hpp"

#include <array>
#include <iostream>
#include <exception>

using namespace dart;

namespace dart {
namespace math {


//==============================================================================
void centralDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed)> getPerturbed,
    Eigen::MatrixXs &result,
    s_t eps)
{
  if (result.size() == 0)
    return;

  // Run central differences for every column of the result separately
  for (std::size_t dof = 0; dof < result.cols(); dof++)
  {
    s_t epsPos = eps;
    Eigen::VectorXs perturbedPlus;
    // Get perturbed result with smaller and smaller eps until valid
    while (!getPerturbed(epsPos, dof, perturbedPlus))
    {
      epsPos *= 0.5;
      if (abs(epsPos) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    s_t epsNeg = eps;
    Eigen::VectorXs perturbedMinus;
    while (!getPerturbed(-epsNeg, dof, perturbedMinus))
    {
      epsNeg *= 0.5;
      if (abs(epsPos) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    // if this point is reached, getPerturbed should have produced valid results
    Eigen::VectorXs grad = (perturbedPlus - perturbedMinus) / (epsPos + epsNeg);
    result.col(dof).noalias() = grad;
  }

  return;
}

//==============================================================================
template <class T>
void centralDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ s_t& perturbed)> getPerturbed,
    T &result,
    s_t eps)
{
  if (result.size() == 0)
    return;

  // Run central differences for every column of the result separately
  for (std::size_t dof = 0; dof < result.size(); dof++)
  {
    s_t epsPos = eps;
    s_t perturbedPlus;
    // Get perturbed result with smaller and smaller eps until valid
    while (!getPerturbed(epsPos, dof, perturbedPlus))
    {
      epsPos *= 0.5;
      if (abs(epsPos) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    s_t epsNeg = eps;
    s_t perturbedMinus;
    while (!getPerturbed(-epsNeg, dof, perturbedMinus))
    {
      epsNeg *= 0.5;
      if (abs(epsPos) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    // if this point is reached, getPerturbed should have produced valid results
    result(dof) = (perturbedPlus - perturbedMinus) / (epsPos + epsNeg);
  }

  return;
}

//==============================================================================
template <class T>
void centralDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ T& perturbed)> getPerturbed,
    T &result,
    s_t eps)
{
  if (result.size() == 0)
    return;

  s_t epsPos = eps;
  T perturbedPlus;
  // Get perturbed result with smaller and smaller eps until valid
  while (!getPerturbed(epsPos, perturbedPlus))
  {
    epsPos *= 0.5;
    if (abs(epsPos) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  s_t epsNeg = eps;
  T perturbedMinus;
  while (!getPerturbed(-epsNeg, perturbedMinus))
  {
    epsNeg *= 0.5;
    if (abs(epsPos) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  // if this point is reached, getPerturbed should have produced valid results
  result = (perturbedPlus - perturbedMinus) / (epsPos + epsNeg);

  return;
}

//==============================================================================
void centralDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ s_t& perturbed)> getPerturbed,
    s_t &result,
    s_t eps)
{
  s_t epsPos = eps;
  s_t perturbedPlus;
  // Get perturbed result with smaller and smaller eps until valid
  while (!getPerturbed(epsPos, perturbedPlus))
  {
    epsPos *= 0.5;
    if (abs(epsPos) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  s_t epsNeg = eps;
  s_t perturbedMinus;
  while (!getPerturbed(-epsNeg, perturbedMinus))
  {
    epsNeg *= 0.5;
    if (abs(epsPos) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  // if this point is reached, getPerturbed should have produced valid results
  result = (perturbedPlus - perturbedMinus) / (epsPos + epsNeg);

  return;
}

//==============================================================================
void riddersMethod(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed)> getPerturbed,
    Eigen::MatrixXs &result,
    s_t eps)
{
  if (result.size() == 0)
    return;
  
  s_t originalStepSize = eps;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  // Run central differences for every column of the result separately
  for (std::size_t dof = 0; dof < result.cols(); dof++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    // Get perturbed result with smaller and smaller eps until valid
    // For Ridders we want the pos and neg epsilons to be the same.
    Eigen::VectorXs perturbedPlus, perturbedMinus;
    while (!getPerturbed(originalStepSize, dof, perturbedPlus)
           || !getPerturbed(-originalStepSize, dof, perturbedMinus))
    {
      originalStepSize *= 0.5;
      if (abs(originalStepSize) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    // if this point is reached, getPerturbed should have produced valid results
    tab[0][0] = (perturbedPlus - perturbedMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      if (!getPerturbed(stepSize, dof, perturbedPlus)
           || !getPerturbed(-stepSize, dof, perturbedMinus))
      {
        throw ridders_invalid_state_exception();
      }

      tab[0][iTab] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          result.col(dof).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  return;
}

//==============================================================================
template <class T>
void riddersMethod(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ s_t& perturbed)> getPerturbed,
    T &result,
    s_t eps)
{
  if (result.size() == 0)
    return;
  
  s_t originalStepSize = eps;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  // Run central differences for every column of the result separately
  for (std::size_t dof = 0; dof < result.size(); dof++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<s_t, tabSize>, tabSize> tab;

    // Get perturbed result with smaller and smaller eps until valid
    // For Ridders we want the pos and neg epsilons to be the same.
    s_t perturbedPlus, perturbedMinus;
    while (!getPerturbed(originalStepSize, dof, perturbedPlus)
           || !getPerturbed(-originalStepSize, dof, perturbedMinus))
    {
      originalStepSize *= 0.5;
      if (abs(originalStepSize) <= 1e-20)
        throw non_differentiable_point_exception();
    }

    // if this point is reached, getPerturbed should have produced valid results
    tab[0][0] = (perturbedPlus - perturbedMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      if (!getPerturbed(stepSize, dof, perturbedPlus)
           || !getPerturbed(-stepSize, dof, perturbedMinus))
      {
        throw ridders_invalid_state_exception();
      }

      tab[0][iTab] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]));
        if (currError < bestError)
        {
          bestError = currError;
          result(dof) = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1])
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  return;
}

//==============================================================================
template <class T>
void riddersMethod(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ T& perturbed)> getPerturbed,
    T &result,
    s_t eps)
{
  if (result.size() == 0)
    return;
  
  s_t originalStepSize = eps;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  // Neville tableau of finite difference results
  std::array<std::array<T, tabSize>, tabSize> tab;

  // Get perturbed result with smaller and smaller eps until valid
  // For Ridders we want the pos and neg epsilons to be the same.
  T perturbedPlus, perturbedMinus;
  while (!getPerturbed(originalStepSize, perturbedPlus)
          || !getPerturbed(-originalStepSize, perturbedMinus))
  {
    originalStepSize *= 0.5;
    if (abs(originalStepSize) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  // if this point is reached, getPerturbed should have produced valid results
  tab[0][0] = (perturbedPlus - perturbedMinus) / (2 * originalStepSize);

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    if (!getPerturbed(stepSize, perturbedPlus)
          || !getPerturbed(-stepSize, perturbedMinus))
    {
      throw ridders_invalid_state_exception();
    }

    tab[0][iTab] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                        / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
              .array()
              .abs()
              .maxCoeff());
      if (currError < bestError)
      {
        bestError = currError;
        result = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return;
}

//==============================================================================
void riddersMethod(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ s_t& perturbed)> getPerturbed,
    s_t &result,
    s_t eps)
{
  s_t originalStepSize = eps;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  // Neville tableau of finite difference results
  std::array<std::array<s_t, tabSize>, tabSize> tab;

  // Get perturbed result with smaller and smaller eps until valid
  // For Ridders we want the pos and neg epsilons to be the same.
  s_t perturbedPlus, perturbedMinus;
  while (!getPerturbed(originalStepSize, perturbedPlus)
          || !getPerturbed(-originalStepSize, perturbedMinus))
  {
    originalStepSize *= 0.5;
    if (abs(originalStepSize) <= 1e-20)
      throw non_differentiable_point_exception();
  }

  // if this point is reached, getPerturbed should have produced valid results
  tab[0][0] = (perturbedPlus - perturbedMinus) / (2 * originalStepSize);

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    if (!getPerturbed(stepSize, perturbedPlus)
          || !getPerturbed(-stepSize, perturbedMinus))
    {
      throw ridders_invalid_state_exception();
    }

    tab[0][iTab] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                        / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]));
      if (currError < bestError)
      {
        bestError = currError;
        result = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1])
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return;
}

//==============================================================================
void finiteDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed)> getPerturbed,
    Eigen::MatrixXs &result,
    s_t eps,
    bool useRidders)
{
  if (useRidders)
  {
    riddersMethod(getPerturbed, result, eps);
  }
  else
  {
    centralDifference(getPerturbed, result, eps);
  }

  return;
}

//==============================================================================
template <class T>
void finiteDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ s_t& perturbed)> getPerturbed,
    T &result,
    s_t eps,
    bool useRidders)
{
  if (useRidders)
  {
    riddersMethod(getPerturbed, result, eps);
  }
  else
  {
    centralDifference(getPerturbed, result, eps);
  }

  return;
}

//==============================================================================
template <class T>
void finiteDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ T& perturbed)> getPerturbed,
    T &result,
    s_t eps,
    bool useRidders)
{
  if (useRidders)
  {
    riddersMethod(getPerturbed, result, eps);
  }
  else
  {
    centralDifference(getPerturbed, result, eps);
  }

  return;
}

//==============================================================================
void finiteDifference(
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ s_t& perturbed)> getPerturbed,
    s_t &result,
    s_t eps,
    bool useRidders)
{
  if (useRidders)
  {
    riddersMethod(getPerturbed, result, eps);
  }
  else
  {
    centralDifference(getPerturbed, result, eps);
  }

  return;
}

//==============================================================================
// Explicit instantiations
template void finiteDifference<Eigen::MatrixXs>(
    std::function<bool(s_t, Eigen::MatrixXs&)>, Eigen::MatrixXs&, s_t, bool);
template void finiteDifference<Eigen::Matrix6s>(
    std::function<bool(s_t, Eigen::Matrix6s&)>, Eigen::Matrix6s&, s_t, bool);
template void finiteDifference<Eigen::Matrix3s>(
    std::function<bool(s_t, Eigen::Matrix3s&)>, Eigen::Matrix3s&, s_t, bool);
template void finiteDifference<Eigen::Matrix<s_t, 6, 2>>(
    std::function<bool(s_t, Eigen::Matrix<s_t, 6, 2>&)>, 
    Eigen::Matrix<s_t, 6, 2>&, s_t, bool);
template void finiteDifference<Eigen::VectorXs>(
    std::function<bool(s_t, Eigen::VectorXs&)>, Eigen::VectorXs&, s_t, bool);
template void finiteDifference<Eigen::Vector6s>(
    std::function<bool(s_t, Eigen::Vector6s&)>, Eigen::Vector6s&, s_t, bool);
template void finiteDifference<Eigen::Vector3s>(
    std::function<bool(s_t, Eigen::Vector3s&)>, Eigen::Vector3s&, s_t, bool);
template void finiteDifference<math::Jacobian>(
    std::function<bool(s_t, math::Jacobian&)>, math::Jacobian&, s_t, bool);

template void finiteDifference<Eigen::VectorXs>(
    std::function<bool(s_t, int, s_t&)>, Eigen::VectorXs&, s_t, bool);  
template void finiteDifference<Eigen::Vector6s>(
    std::function<bool(s_t, int, s_t&)>, Eigen::Vector6s&, s_t, bool);

} // namespace math
} // namespace dart
