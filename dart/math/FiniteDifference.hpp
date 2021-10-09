#ifndef DART_MATH_FINITE_DIFFERENCE_HPP_
#define DART_MATH_FINITE_DIFFERENCE_HPP_

#include <functional>

#include <Eigen/Dense>

#include "dart/common/Deprecated.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/MathTypes.hpp"

using namespace dart;
namespace dart {
namespace math {

/// Finite differences a vector function, iterating and perturbing
/// the partial derivatives w.r.t the input DOFs one by one.
/// Note that if using Ridders, epsilon should be very large, >=1e-4
void finiteDifference(
    // this should return if the perturbation was valid
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed)> getPerturbed,
    Eigen::MatrixXs& result,
    s_t eps = 1e-7,
    bool useRidders = false);

/// Finite differences a scalar function, iterating and perturbing
/// the partial derivatives w.r.t the input DOFs one by one.
/// Note that if using Ridders, epsilon should be very large, >=1e-4
/// T must be an Eigen type (instantiate new types in the .cpp)
template <class T>
void finiteDifference(
    // this should return if the perturbation was valid
    std::function<bool(
        /* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ s_t& perturbed)> getPerturbed,
    T& result,
    s_t eps = 1e-7,
    bool useRidders = false);

/// Static version, does not iterate over DOFs. T must be an
/// Eigen type (instantiate new types in the .cpp)
template <class T>
void finiteDifference(
    // this should return if the perturbation was valid
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ T& perturbed)> getPerturbed,
    T& result,
    s_t eps = 1e-7,
    bool useRidders = false);

/// Finite differences a scalar function of scalar input.
void finiteDifference(
    // this should return if the perturbation was valid
    std::function<bool(
        /* in*/ s_t eps,
        /*out*/ s_t& perturbed)> getPerturbed,
    s_t& result,
    s_t eps = 1e-7,
    bool useRidders = false);

struct non_differentiable_point_exception : public std::exception
{
  const char* what() const throw()
  {
    return "Found a non-differentiable point while determining EPS.";
  }
};

struct ridders_invalid_state_exception : public std::exception
{
  const char* what() const throw()
  {
    return "Ridders' Method reached an 'invalid' state by lowering EPS.";
  }
};

} // namespace math
} // namespace dart

#endif