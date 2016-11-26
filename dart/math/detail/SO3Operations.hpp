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

template <typename>
class SO3Base;

template <typename>
class SO3RotationMatrix;

template <typename>
class SO3RotationVector;

template <typename>
class SO3AngleAxis;

template <typename>
class SO3Quaternion;

template <typename S>
using DefaultSO3Canonical = SO3RotationMatrix<S>;

// Forward declarations
template <typename, typename> class SO3;

namespace detail {

//==============================================================================
// traits:
// Traits for all SO3 classes that is agnostic to representation types
//==============================================================================

//==============================================================================
template <typename S_>
struct traits<SO3RotationMatrix<S_>>
{
  using S = S_;
  using RepData = Eigen::Matrix<S, 3, 3>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_>
struct traits<SO3AngleAxis<S_>>
{
  using S = S_;
  using RepData = Eigen::AngleAxis<S>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_>
struct traits<SO3Quaternion<S_>>
{
  using S = S_;
  using RepData = Eigen::Quaternion<S>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_>
struct traits<SO3RotationVector<S_>>
{
  using S = S_;
  using RepData = Eigen::Matrix<S, 3, 1>;
  static constexpr bool IsCoordinates = true;
};

namespace so3_operations {

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
            Eigen::RotationBase<typename traits<Rep>::RepData, 3>,
            typename traits<Rep>::RepData
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
            Eigen::MatrixBase<typename traits<Rep>::RepData>,
            typename traits<Rep>::RepData
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
template <typename SO3From,
          typename SO3Canonical = DefaultSO3Canonical<typename SO3From::S>>
struct so3_convert_to_canonical_impl
{
  using RepDataFrom = typename traits<SO3From>::RepData;
  using RepDataTo = typename traits<SO3Canonical>::RepData;

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
template <typename SO3Canonical>
struct so3_convert_to_canonical_impl<SO3Canonical, SO3Canonical>
{
  using RepDataFrom = typename traits<SO3Canonical>::RepData;
  using RepDataTo = typename traits<SO3Canonical>::RepData;

  static const RepDataFrom& run(const RepDataFrom& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct so3_convert_to_canonical_impl<SO3RotationVector<S>, SO3RotationMatrix<S>>
{
  using RepDataFrom = typename traits<SO3RotationVector<S>>::RepData;
  using RepDataTo = typename traits<SO3RotationMatrix<S>>::RepData;

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
          typename SO3RepTo,
          typename SO3Canonical = DefaultSO3Canonical<S>>
struct so3_convert_from_canonical_impl
{
  using RepDataFrom = typename traits<SO3Canonical>::RepData;
  using RepDataTo = typename traits<SO3RepTo>::RepData;

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
template <typename S, typename SO3Canonical>
struct so3_convert_from_canonical_impl<S, SO3Canonical, SO3Canonical>
{
  using RepDataFrom = typename traits<SO3Canonical>::RepData;

  static const RepDataFrom& run(const RepDataFrom& canonicalData)
  {
    return canonicalData;
  }
};

//==============================================================================
template <typename S>
struct so3_convert_from_canonical_impl<S, SO3RotationVector<S>, SO3RotationMatrix<S>>
{
  using RepDataFrom = typename traits<SO3RotationMatrix<S>>::RepData;
  using RepDataTo = typename traits<SO3RotationVector<S>>::RepData;

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
          typename SO3Canonical = DefaultSO3Canonical<S>,
          typename Enable = void>
struct so3_convert_impl
{
  using RepDataFrom = typename traits<RepFrom>::RepData;
  using RepDataTo = typename traits<RepTo>::RepData;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return so3_convert_from_canonical_impl<S, RepTo, SO3Canonical>::run(
        so3_convert_to_canonical_impl<RepFrom, SO3Canonical>::run(data));
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return so3_convert_from_canonical_impl<S, RepTo, SO3Canonical>::run(
        so3_convert_to_canonical_impl<RepFrom, SO3Canonical>::run(
            std::move(data)));
  }
};

//==============================================================================
// For the same representations, simply return the data without conversion
template <typename S, typename SO3Type>
struct so3_convert_impl<S, SO3Type, SO3Type>
{
  using RepData = typename traits<SO3Type>::RepData;

  static const RepData& run(const RepData& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct so3_convert_impl<S, SO3RotationVector<S>, SO3AngleAxis<S>>
{
  using RepDataFrom = typename traits<SO3RotationVector<S>>::RepData;
  using RepDataTo = typename traits<SO3AngleAxis<S>>::RepData;

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
//struct rep_convert_impl<S, SO3RotationVector, SO3Quaternion>
//{
//  using RepDataFrom = typename traits<SO3RotationVector<S>>::RepData;
//  using RepDataTo = typename traits<SO3Quaternion<S>>::RepData;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct so3_convert_impl<S, SO3AngleAxis<S>, SO3RotationVector<S>>
{
  using RepDataFrom = typename traits<SO3AngleAxis<S>>::RepData;
  using RepDataTo = typename traits<SO3RotationVector<S>>::RepData;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data.angle() * data.axis();
  }
};

//==============================================================================
template <typename S>
struct so3_convert_impl<S, SO3AngleAxis<S>, SO3Quaternion<S>>
{
  using RepDataFrom = typename traits<SO3AngleAxis<S>>::RepData;
  using RepDataTo = typename traits<SO3Quaternion<S>>::RepData;

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
//template <typename S, typename SO3Canonical>
//struct rep_convert_impl<S, SO3Quaternion, SO3RotationVector, SO3Canonical>
//{
//  using RepDataFrom = typename traits<SO3Quaternion<S>>::RepData;
//  using RepDataTo = typename traits<SO3RotationVector<S>>::RepData;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    return ;
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct so3_convert_impl<S, SO3Quaternion<S>, SO3AngleAxis<S>>
{
  using RepDataFrom = typename traits<SO3Quaternion<S>>::RepData;
  using RepDataTo = typename traits<SO3AngleAxis<S>>::RepData;

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
          typename SO3A,
          typename SO3B,
          typename SO3Canonical = DefaultSO3Canonical<S>,
          typename Enable = void>
struct so3_is_approx_impl
{
  using RepDataTypeA = typename traits<SO3A>::RepData;
  using RepDataTypeB = typename traits<SO3B>::RepData;

  static bool run(const RepDataTypeA& dataA, const RepDataTypeB& dataB, S tol)
  {
    return so3_convert_to_canonical_impl<SO3A, SO3Canonical>::run(dataA)
        .isApprox(so3_convert_to_canonical_impl<SO3B, SO3Canonical>
                  ::run(dataB), tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two point on the manifolds rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename S, typename SO3Type>
struct so3_is_approx_impl<S, SO3Type, SO3Type>
{
  using RepData = typename traits<SO3Type>::RepData;

  static bool run(const RepData& dataA, const RepData& dataB, S tol)
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
template <typename S, typename SO3Canonical = DefaultSO3Canonical<S>>
struct so3_canonical_multiplication_impl
{
  using CanonicalRepData
      = typename traits<SO3Canonical>::RepData;

  static const CanonicalRepData run(
      const CanonicalRepData& data, const CanonicalRepData& otherData)
  {
    return data * otherData;
  }
};

//==============================================================================
template <typename S>
struct so3_canonical_multiplication_impl<S, SO3RotationVector<S>>
{
  using CanonicalRepData
      = typename traits<SO3RotationMatrix<S>>::RepData;

  static const CanonicalRepData run(
      const CanonicalRepData& data, const CanonicalRepData& otherData)
  {
    return log(exp(data) * exp(otherData));
  }
};

//==============================================================================
// rep_canonical_inplace_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3Canonical = DefaultSO3Canonical<S>>
struct so3_canonical_inplace_multiplication_impl
{
  using CanonicalRepData
      = typename traits<SO3Canonical>::RepData;

  static void run(
      CanonicalRepData& data, const CanonicalRepData& otherData)
  {
    data *= otherData;
  }
};

//==============================================================================
template <typename S>
struct so3_canonical_inplace_multiplication_impl<S, SO3RotationVector<S>>
{
  using CanonicalRepData
      = typename traits<SO3RotationVector<S>>::RepData;

  static void run(
      CanonicalRepData& data, const CanonicalRepData& otherData)
  {
    data = log(exp(data) * exp(otherData));
  }
};

//==============================================================================
// rep_multiplication_impl:
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   -   |   -   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   X   |   3   |   -   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   X   |   X   |   0   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   X   |   X   |   X   |   0   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |       |       |       |       |       |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: zero conversion
// 3: triple conversions; [(rep -> canonical) * (rep -> canonical)] -> rep

//==============================================================================
template <typename SO3A, typename SO3B>
struct so3_multiplication_impl
{
  using RepDataTypeA = typename traits<SO3A>::RepData;
  using RepDataTypeB = typename traits<SO3B>::RepData;

  static auto run(const RepDataTypeA& dataA, const RepDataTypeB& dataB)
  -> decltype(dataA * dataB)
  {
    return dataA * dataB;
  }
};

//==============================================================================
template <typename S>
struct so3_multiplication_impl<SO3RotationVector<S>, SO3RotationVector<S>>
{
  using RepData = typename traits<SO3RotationVector<S>>::RepData;
  using SO3CanonicalRep = SO3Quaternion<S>; // TODO(JS): find best canonical for vec * vec
  using CanonicalRepData = typename traits<SO3CanonicalRep>::RepData;

  static auto run(const RepData& dataA, const RepData& dataB)
  -> decltype(std::declval<CanonicalRepData>() * std::declval<CanonicalRepData>())
  {
    return so3_convert_impl<S, SO3RotationVector<S>, SO3CanonicalRep>::run(dataA)
        * so3_convert_impl<S, SO3RotationVector<S>, SO3CanonicalRep>::run(dataB);
    // TODO(JS): improve; super slow
  }
};

// TODO(JS): Heterogeneous multiplications are not implemented yet.

//==============================================================================
//template <typename S, typename RepB>
//struct rep_multiplication_impl<S, SO3RotationVector, RepB>
//{
//  using RepDataTypeA = typename traits<SO3RotationVector<S>>::RepData;
//  using RepData = typename traits<SO3B>::RepData;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepData& dataB)
//      -> decltype(std::declval<RepData>() * dataB)
//  {
//    return RepData(dataA) * dataB;
//  }
//};

////==============================================================================
//template <typename S, typename RepA>
//struct rep_multiplication_impl<S, RepA, SO3RotationVector>
//{
//  using RepDataTypeA = typename traits<SO3A>::RepData;
//  using RepData = typename traits<SO3RotationVector<S>>::RepData;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepData& dataB)
//      -> decltype(dataA * std::declval<RepDataTypeA>())
//  {
//    return dataA * RepDataTypeA(dataB);
//  }
//};

//==============================================================================
// group_is_canonical:
//==============================================================================

//==============================================================================
template <typename SO3Type,
          typename SO3Canonical = DefaultSO3Canonical<typename SO3Type::S>,
          typename Enable = void>
struct group_is_canonical : std::false_type {};

template <typename SO3Type, typename SO3Canonical>
struct group_is_canonical<
    SO3Type,
    SO3Canonical,
    typename std::enable_if
        <std::is_same<typename SO3Type::Rep, SO3Canonical>::value>::type>
    : std::true_type {};

//==============================================================================
// assign_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3To, typename SO3From>
struct group_assign_impl
{
  static void run(SO3To& to, const SO3From& from)
  {
    to.setRepData(so3_convert_impl<S, SO3From, SO3To>::run(from.getRepData()));
  }
};

//==============================================================================
// group_multiplication_impl:
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (i.e., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A, typename SO3B, typename Enable = void>
struct group_multiplication_impl
{
  using S = typename SO3A::S;

  static const SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(so3_convert_from_canonical_impl<S, SO3A>::run(
          so3_canonical_multiplication_impl<S>::run(
            so3_convert_to_canonical_impl<SO3A>::run(Ra.getRepData()),
            so3_convert_to_canonical_impl<SO3B>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between ones supported by Eigen (i.e., 3x3 matrix,
// AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B>
struct group_multiplication_impl<
    SO3A,
    SO3B,
    typename std::enable_if<
//        !std::is_same<typename SO3A::RepData,  typename SO3B::RepData>::value
        true
        && (std::is_same<typename SO3A::RepData, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3A::RepData, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3A::RepData, Eigen::Quaternion<typename SO3A::S>>::value)
        && (std::is_same<typename SO3B::RepData, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3B::RepData, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3B::RepData, Eigen::Quaternion<typename SO3A::S>>::value)
    >::type>
{
  using S = typename SO3A::S;

  using RepDataTypeA = typename traits<SO3A>::RepData;
  using RepDataTypeB = typename traits<SO3B>::RepData;

  static const SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(Ra.getRepData() * Rb.getRepData());
  }
};

//==============================================================================
// group_inplace_multiplication_impl:
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (i.e., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A, typename SO3B, typename Enable = void>
struct group_inplace_multiplication_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& Ra, const SO3B& Rb)
  {
    Ra.setRepData(so3_convert_from_canonical_impl<S, RepA>::run(
          so3_canonical_multiplication_impl<S>::run(
            so3_convert_to_canonical_impl<S, RepA>::run(Ra.getRepData()),
            so3_convert_to_canonical_impl<S, RepB>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between ones supported by Eigen (i.e., 3x3 matrix,
// AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B>
struct group_inplace_multiplication_impl<
    SO3A,
    SO3B,
    typename std::enable_if<
        (std::is_same<typename SO3A::RepData, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3A::RepData, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3A::RepData, Eigen::Quaternion<typename SO3A::S>>::value)
        && (std::is_same<typename SO3B::RepData, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3B::RepData, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3B::RepData, Eigen::Quaternion<typename SO3A::S>>::value)
    >::type>
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& so3A, const SO3B& so3B)
  {
    so3A.getRepData() *= so3B.getRepData();
  }
};

//==============================================================================
// group_is_approx_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B>
struct group_is_approx_impl
{
  using S = typename SO3A::S;

  static bool run(const SO3A& Ra, const SO3B& Rb, S tol)
  {
    return so3_is_approx_impl<S, SO3A, SO3B>::run(
          Ra.getRepData(), Rb.getRepData(), tol);
  }
};

} // namespace so3_operations

//==============================================================================
template <typename S, typename RepFrom, typename RepTo, typename Enable = void>
struct to_impl {};

//==============================================================================
// Converting to the raw data type from given SO3 representation type
template <typename S, typename RepFrom, typename RepTo>
struct to_impl<
    S,
    RepFrom,
    RepTo,
    typename std::enable_if<
        std::is_base_of<SO3Base<RepTo>, RepTo>::value>::type
    >
{
  using RepData = typename detail::traits<RepFrom>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::so3_operations::so3_convert_impl<S, RepFrom, RepTo>::run(
      std::declval<RepData>()))
  {
    return detail::so3_operations::so3_convert_impl<S, RepFrom, RepTo>::run(
          repData);
  }
};

//==============================================================================
// Converting to the raw data type from given raw data type
template <typename S, typename RepFrom, typename RepTo>
struct to_impl<
    S,
    RepFrom,
    RepTo,
    typename std::enable_if<
        std::is_same<
            typename detail::traits<SO3RotationMatrix<S>>::RepData,
            RepTo>::value
        >::type
    >
{
  using RepData = typename detail::traits<RepFrom>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::so3_operations::so3_convert_impl<S, RepFrom, SO3RotationMatrix<S>>::run(
      std::declval<RepData>()))
  {
    return detail::so3_operations::so3_convert_impl<S, RepFrom, SO3RotationMatrix<S>>::run(
          repData);
  }
};

//==============================================================================
// Converting to the raw data type from given raw data type
template <typename S, typename RepFrom, typename RepTo>
struct to_impl<
    S,
    RepFrom,
    RepTo,
    typename std::enable_if<
        std::is_same<
            typename detail::traits<SO3RotationVector<S>>::RepData,
            RepTo>::value
        >::type
    >
{
  using RepData = typename detail::traits<RepFrom>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::so3_operations::so3_convert_impl<S, RepFrom, SO3RotationVector<S>>::run(
      std::declval<RepData>()))
  {
    return detail::so3_operations::so3_convert_impl<S, RepFrom, SO3RotationVector<S>>::run(
          repData);
  }
};

//==============================================================================
// Converting to the raw data type from given raw data type
template <typename S, typename RepFrom, typename RepTo>
struct to_impl<
    S,
    RepFrom,
    RepTo,
    typename std::enable_if<
        std::is_same<
            typename detail::traits<SO3AngleAxis<S>>::RepData,
            RepTo>::value
        >::type
    >
{
  using RepData = typename detail::traits<RepFrom>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::so3_operations::so3_convert_impl<S, RepFrom, SO3AngleAxis<S>>::run(
      std::declval<RepData>()))
  {
    return detail::so3_operations::so3_convert_impl<S, RepFrom, SO3AngleAxis<S>>::run(
          repData);
  }
};


//==============================================================================
// Converting to the raw data type from given raw data type
template <typename S, typename RepFrom, typename RepTo>
struct to_impl<
    S,
    RepFrom,
    RepTo,
    typename std::enable_if<
        std::is_same<
            typename detail::traits<SO3Quaternion<S>>::RepData,
            RepTo>::value
        >::type
    >
{
  using RepData = typename detail::traits<RepFrom>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::so3_operations::so3_convert_impl<S, RepFrom, SO3Quaternion<S>>::run(
      std::declval<RepData>()))
  {
    return detail::so3_operations::so3_convert_impl<S, RepFrom, SO3Quaternion<S>>::run(
          repData);
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3OPERATIONS_HPP_
