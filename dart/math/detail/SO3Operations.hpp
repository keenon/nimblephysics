/*
 * Copyright (c) 2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DART_MATH_DETAIL_SO3OPERATIONS_HPP_
#define DART_MATH_DETAIL_SO3OPERATIONS_HPP_

#include <Eigen/Eigen>
#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Constants.hpp"

namespace dart {
namespace math {

struct SO3Representation;

struct RotationMatrixRep;
struct AxisAngleRep;
struct QuaternionRep;
struct RotationVectorRep;

using DefaultSO3CanonicalRep = RotationMatrixRep;

// Forward declarations
template <typename, typename> class SO3;

namespace detail {

//==============================================================================
// traits:
// Traits for all SO3 classes that is agnostic to representation types
//==============================================================================

//==============================================================================
template <typename S_, typename Rep_>
struct traits<SO3<S_, Rep_>>
{
  using S = S_;
  using Rep = Rep_;

  using SO3Canonical = SO3<S, DefaultSO3CanonicalRep>;
};

namespace SO3 {

//==============================================================================
// rep_traits:
// Traits for different SO3 representations
//==============================================================================

//==============================================================================
template <typename S, typename Rep>
struct rep_traits;

//==============================================================================
template <typename S>
struct rep_traits<S, RotationMatrixRep>
{
  using RepDataType = Eigen::Matrix<S, 3, 3>;
};

//==============================================================================
template <typename S>
struct rep_traits<S, AxisAngleRep>
{
  using RepDataType = Eigen::AngleAxis<S>;
};

//==============================================================================
template <typename S>
struct rep_traits<S, QuaternionRep>
{
  using RepDataType = Eigen::Quaternion<S>;
};

//==============================================================================
template <typename S>
struct rep_traits<S, RotationVectorRep>
{
  using RepDataType = Eigen::Matrix<S, 3, 1>;
};

//==============================================================================
// rep_is_eigen_rotation_impl:
//==============================================================================

//==============================================================================
template <typename S, typename Rep, typename Enable = void>
struct rep_is_eigen_rotation_impl : std::false_type {};

//==============================================================================
template <typename S, typename Rep>
struct rep_is_eigen_rotation_impl<
    S,
    Rep,
    typename std::enable_if<
        std::is_base_of<
            Eigen::RotationBase<typename rep_traits<S, Rep>::RepDataType, 3>,
            typename rep_traits<S, Rep>::RepDataType
        >::value
    >::type>
    : std::true_type {};

//==============================================================================
// rep_is_eigen_matrix_impl:
//==============================================================================

//==============================================================================
template <typename S, typename Rep, typename Enable = void>
struct rep_is_eigen_matrix_impl : std::false_type {};

//==============================================================================
template <typename S, typename Rep>
struct rep_is_eigen_matrix_impl<
    S,
    Rep,
    typename std::enable_if<
        std::is_base_of<
            Eigen::MatrixBase<typename rep_traits<S, Rep>::RepDataType>,
            typename rep_traits<S, Rep>::RepDataType
        >::value
    >::type>
    : std::true_type {};

//==============================================================================
// exp:
//==============================================================================

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 3> exp(const Eigen::Matrix<S, 3, 1>& w)
{
  using Matrix3 = Eigen::Matrix<S, 3, 3>;

  Matrix3 res;

  S s2[] = { w[0]*w[0], w[1]*w[1], w[2]*w[2] };
  S s3[] = { w[0]*w[1], w[1]*w[2], w[2]*w[0] };
  S theta = std::sqrt(s2[0] + s2[1] + s2[2]);
  S cos_t = std::cos(theta), alpha, beta;

  if (theta > constants<S>::eps())
  {
    S sin_t = std::sin(theta);
    alpha = sin_t / theta;
    beta = (1.0 - cos_t) / theta / theta;
  }
  else
  {
    alpha = 1.0 - theta*theta/6.0;
    beta = 0.5 - theta*theta/24.0;
  }

  res(0, 0) = beta*s2[0] + cos_t;
  res(1, 0) = beta*s3[0] + alpha*w[2];
  res(2, 0) = beta*s3[2] - alpha*w[1];

  res(0, 1) = beta*s3[0] - alpha*w[2];
  res(1, 1) = beta*s2[1] + cos_t;
  res(2, 1) = beta*s3[1] + alpha*w[0];

  res(0, 2) = beta*s3[2] + alpha*w[1];
  res(1, 2) = beta*s3[1] - alpha*w[0];
  res(2, 2) = beta*s2[2] + cos_t;

  return res;
}

//==============================================================================
// log:
//==============================================================================

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 1> log(const Eigen::Matrix<S, 3, 3>& R)
{
  Eigen::AngleAxis<S> aa(R);

  return aa.angle()*aa.axis();
}

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 1> log(Eigen::Matrix<S, 3, 3>&& R)
{
  Eigen::AngleAxis<S> aa(std::move(R));

  return aa.angle()*aa.axis();
}

//==============================================================================
// rep_convert_to_canonical_impl:
//==============================================================================

//==============================================================================
template <typename S,
          typename RepFrom,
          typename SO3CanonicalRep = DefaultSO3CanonicalRep>
struct rep_convert_to_canonical_impl
{
  using RepDataFrom = typename rep_traits<S, RepFrom>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};
// Eigen::Matrix has constructors for rotation types, AngleAxis, Quaternion,
// so we don't need to specialize for them.

//==============================================================================
template <typename S, typename SO3CanonicalRep>
struct rep_convert_to_canonical_impl<S, SO3CanonicalRep, SO3CanonicalRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataFrom& run(const RepDataFrom& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct rep_convert_to_canonical_impl<S, RotationVectorRep, RotationMatrixRep>
{
  using RepDataFrom = typename rep_traits<S, RotationVectorRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RotationMatrixRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return exp(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return exp(std::move(data));
  }
};

//==============================================================================
// rep_convert_from_canonical_impl:
//==============================================================================

//==============================================================================
template <typename S,
          typename RepTo,
          typename SO3CanonicalRep = DefaultSO3CanonicalRep>
struct rep_convert_from_canonical_impl
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RepTo>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return RepDataTo(canonicalData);
  }

  static const RepDataTo run(RepDataFrom&& canonicalData)
  {
    return RepDataTo(std::move(canonicalData));
  }
};
// Eigen rotation types (AngleAxis, Quaternion) have constructors for
// Eigen::MatrixBase so we don't need to specialize for them.

//==============================================================================
template <typename S, typename SO3CanonicalRep>
struct rep_convert_from_canonical_impl<S, SO3CanonicalRep, SO3CanonicalRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataFrom& run(const RepDataFrom& canonicalData)
  {
    return canonicalData;
  }
};

//==============================================================================
template <typename S>
struct rep_convert_from_canonical_impl<S, RotationVectorRep, RotationMatrixRep>
{
  using RepDataFrom = typename rep_traits<S, RotationMatrixRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RotationVectorRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return log(canonicalData);
  }

  static const RepDataTo run(RepDataFrom&& canonicalData)
  {
    return log(std::move(canonicalData));
  }
};

//==============================================================================
// rep_convert_impl:
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   1   |   1   |   1   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   1   |   0   |   1   |   2   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   1   |   1   |   0   |   1   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   1   |   2   |   1   |   0   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |       |       |       |       |       |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: zero conversion; return input as const reference
// 1: single conversion; from -> canonical, or canonical -> to
// 2: double conversion; from -> canonical -> to

//==============================================================================
template <typename S,
          typename RepFrom,
          typename RepTo,
          typename SO3CanonicalRep = DefaultSO3CanonicalRep,
          typename Enable = void>
struct rep_convert_impl
{
  using RepDataFrom = typename rep_traits<S, RepFrom>::RepDataType;
  using RepDataTo = typename rep_traits<S, RepTo>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return rep_convert_from_canonical_impl<S, RepTo, SO3CanonicalRep>::run(
        rep_convert_to_canonical_impl<S, RepFrom, SO3CanonicalRep>::run(data));
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return rep_convert_from_canonical_impl<S, RepTo, SO3CanonicalRep>::run(
        rep_convert_to_canonical_impl<S, RepFrom, SO3CanonicalRep>::run(
            std::move(data)));
  }
};

//==============================================================================
// For the same representations, simply return the data without conversion
template <typename S, typename Rep>
struct rep_convert_impl<S, Rep, Rep>
{
  using RepData = typename rep_traits<S, Rep>::RepDataType;

  static const RepData& run(const RepData& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct rep_convert_impl<S, RotationVectorRep, AxisAngleRep>
{
  using RepDataFrom = typename rep_traits<S, RotationVectorRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, AxisAngleRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    const S norm = data.norm();

    if (norm > static_cast<S>(0))
      return RepDataTo(norm, data/norm);
    else
      return RepDataTo(static_cast<S>(0), Eigen::Matrix<S, 3, 1>::UnitX());
  }
};

//==============================================================================
//template <typename S>
//struct rep_convert_impl<S, RotationVectorRep, QuaternionRep>
//{
//  using RepDataFrom = typename rep_traits<S, RotationVectorRep>::RepDataType;
//  using RepDataTo = typename rep_traits<S, QuaternionRep>::RepDataType;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct rep_convert_impl<S, AxisAngleRep, RotationVectorRep>
{
  using RepDataFrom = typename rep_traits<S, AxisAngleRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RotationVectorRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data.angle() * data.axis();
  }
};

//==============================================================================
template <typename S>
struct rep_convert_impl<S, AxisAngleRep, QuaternionRep>
{
  using RepDataFrom = typename rep_traits<S, AxisAngleRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, QuaternionRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
//template <typename S, typename SO3CanonicalRep>
//struct rep_convert_impl<S, QuaternionRep, RotationVectorRep, SO3Canonical>
//{
//  using RepDataFrom = typename rep_traits<S, QuaternionRep>::RepDataType;
//  using RepDataTo = typename rep_traits<S, RotationVectorRep>::RepDataType;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    return ;
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct rep_convert_impl<S, QuaternionRep, AxisAngleRep>
{
  using RepDataFrom = typename rep_traits<S, QuaternionRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, AxisAngleRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
// group_is_approx_impl:
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   1   |   1   |   1   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   1   |   0   |   2   |   2   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   1   |   2   |   0   |   2   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   1   |   2   |   2   |   0   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |       |       |       |       |       |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: zero conversion; compare in the given representation
// 2: double conversion; repA -> canonical rep (compare) <- repB

//==============================================================================
template <typename S,
          typename RepA,
          typename RepB,
          typename SO3CanonicalRep = DefaultSO3CanonicalRep,
          typename Enable = void>
struct rep_is_approx_impl
{
  using RepDataTypeA = typename rep_traits<S, RepA>::RepDataType;
  using RepDataTypeB = typename rep_traits<S, RepB>::RepDataType;

  static bool run(const RepDataTypeA& dataA, const RepDataTypeB& dataB, S tol)
  {
    return rep_convert_to_canonical_impl<S, RepA, SO3CanonicalRep>::run(dataA)
        .isApprox(rep_convert_to_canonical_impl<S, RepB, SO3CanonicalRep>
                  ::run(dataB), tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two point on the manifolds rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename S, typename Rep>
struct rep_is_approx_impl<S, Rep, Rep>
{
  using RepDataType = typename rep_traits<S, Rep>::RepDataType;

  static bool run(const RepDataType& dataA, const RepDataType& dataB, S tol)
  {
    return dataA.isApprox(dataB, tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two point on the manifolds rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
// rep_canonical_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3CanonicalRep = DefaultSO3CanonicalRep>
struct rep_canonical_multiplication_impl
{
  using CanonicalRepDataType
      = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const CanonicalRepDataType run(
      const CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    return data * otherData;
  }
};

//==============================================================================
template <typename S>
struct rep_canonical_multiplication_impl<S, RotationVectorRep>
{
  using CanonicalRepDataType
      = typename rep_traits<S, RotationMatrixRep>::RepDataType;

  static const CanonicalRepDataType run(
      const CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    return log(exp(data) * exp(otherData));
  }
};

//==============================================================================
// rep_canonical_inplace_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3CanonicalRep = DefaultSO3CanonicalRep>
struct rep_canonical_inplace_multiplication_impl
{
  using CanonicalRepDataType
      = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static void run(
      CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    data *= otherData;
  }
};

//==============================================================================
template <typename S>
struct rep_canonical_inplace_multiplication_impl<S, RotationVectorRep>
{
  using CanonicalRepDataType
      = typename rep_traits<S, RotationVectorRep>::RepDataType;

  static void run(
      CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    data = log(exp(data) * exp(otherData));
  }
};

//==============================================================================
// group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepA, typename RepB>
struct rep_multiplication_impl
{
  using RepDataTypeA = typename rep_traits<S, RepA>::RepDataType;
  using RepDataType = typename rep_traits<S, RepB>::RepDataType;

  static const auto run(
      const RepDataTypeA& dataA, const RepDataType& dataB)
      -> decltype(dataA * dataB)
  {
    return dataA * dataB;
  }
};

//==============================================================================
template <typename S>
struct rep_multiplication_impl<S, RotationVectorRep, RotationVectorRep>
{
  using RepDataType = typename rep_traits<S, RotationVectorRep>::RepDataType;
  using AxisAngleType = typename rep_traits<S, AxisAngleRep>::RepDataType;

  static const auto run(
      const RepDataType& dataA, const RepDataType& dataB)
      -> decltype(std::declval<AxisAngleType>() * std::declval<AxisAngleType>())
  {
    return AxisAngleType(dataA) * AxisAngleType(dataB);
  }
};

//==============================================================================
//template <typename S, typename RepB>
//struct rep_multiplication_impl<S, RotationVectorRep, RepB>
//{
//  using RepDataTypeA = typename rep_traits<S, RotationVectorRep>::RepDataType;
//  using RepDataType = typename rep_traits<S, RepB>::RepDataType;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepDataType& dataB)
//      -> decltype(std::declval<RepDataType>() * dataB)
//  {
//    return RepDataType(dataA) * dataB;
//  }
//};

////==============================================================================
//template <typename S, typename RepA>
//struct rep_multiplication_impl<S, RepA, RotationVectorRep>
//{
//  using RepDataTypeA = typename rep_traits<S, RepA>::RepDataType;
//  using RepDataType = typename rep_traits<S, RotationVectorRep>::RepDataType;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepDataType& dataB)
//      -> decltype(dataA * std::declval<RepDataTypeA>())
//  {
//    return dataA * RepDataTypeA(dataB);
//  }
//};

} // namespace SO3
} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3OPERATIONS_HPP_
