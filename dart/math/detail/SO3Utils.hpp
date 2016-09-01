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

#ifndef DART_MATH_DETAIL_SO3UTILS_HPP_
#define DART_MATH_DETAIL_SO3UTILS_HPP_

#include <Eigen/Eigen>
#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"

namespace dart {
namespace math {

struct SO3Representation {};

struct SO3RotationMatrix : SO3Representation {};
struct SO3AxisAngle : SO3Representation {};
struct SO3Quaternion : SO3Representation {};

enum class EulerAngle
{
  INTRINSIC_XYZ,
  INTRINSIC_ZYX,

  EXTRINSIC_XYZ = INTRINSIC_ZYX
};

template <EulerAngle Angles>
struct SO3EulerAngle : SO3Representation {};

using SO3Canonical = SO3RotationMatrix;

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

  using MatrixType = Eigen::Matrix<S, 3, 3>;
  using RotationMatrixType = Eigen::Matrix<S, 3, 3>;

  using Canonical = SO3<S, SO3Canonical>;
};

//==============================================================================
// SO3_rep_traits:
// Traits for different SO3 representations
//==============================================================================

//==============================================================================
template <typename S, typename Rep_>
struct SO3_rep_traits;

//==============================================================================
template <typename S_>
struct SO3_rep_traits<S_, SO3RotationMatrix>
{
  using S = S_;
  using Rep = SO3RotationMatrix;
  using DataType = Eigen::Matrix<S, 3, 3>;
};

//==============================================================================
template <typename S_>
struct SO3_rep_traits<S_, SO3AxisAngle>
{
  using S = S_;
  using Rep = SO3AxisAngle;
  using DataType = Eigen::Matrix<S, 3, 1>;
};

//==============================================================================
// SO3_is_canonical:
//==============================================================================

//==============================================================================
template <typename SO3Type, typename Enable = void>
struct SO3_is_canonical : std::false_type {};

template <typename SO3Type>
struct SO3_is_canonical<
    SO3Type,
    typename std::enable_if<std::is_same<typename SO3Type::Rep, SO3Canonical>::value>::type>
    : std::true_type {};

//==============================================================================
// SO3_convert_to_canonical_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepFrom>
struct SO3_convert_to_canonical_impl;

//==============================================================================
template <typename S>
struct SO3_convert_to_canonical_impl<S, SO3Canonical>
{
  using RepDataFrom = typename SO3_rep_traits<S, SO3Canonical>::DataType;
  using RepDataTo = typename SO3_rep_traits<S, SO3Canonical>::DataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct SO3_convert_to_canonical_impl<S, SO3AxisAngle>
{
  using RepDataFrom = typename SO3_rep_traits<S, SO3AxisAngle>::DataType;
  using RepDataTo = typename SO3_rep_traits<S, SO3Canonical>::DataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return ::dart::math::expMapRot(data);
  }
};

//==============================================================================
// SO3_convert_to_noncanonical_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepTo>
struct SO3_convert_to_noncanonical_impl;

//==============================================================================
template <typename S>
struct SO3_convert_to_noncanonical_impl<S, SO3Canonical>
{
  using RepDataFrom = typename SO3_rep_traits<S, SO3Canonical>::DataType;
  using RepDataTo = typename SO3_rep_traits<S, SO3Canonical>::DataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return canonicalData;
  }
};

//==============================================================================
template <typename S>
struct SO3_convert_to_noncanonical_impl<S, SO3AxisAngle>
{
  using RepDataFrom = typename SO3_rep_traits<S, SO3Canonical>::DataType;
  using RepDataTo = typename SO3_rep_traits<S, SO3AxisAngle>::DataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return ::dart::math::logMap(canonicalData);
  }
};

//==============================================================================
// SO3_rep_traits:
//==============================================================================

//==============================================================================
template <typename S, typename RepFrom, typename RepTo>
struct SO3_convert_impl
{
  using RepDataFrom = typename SO3_rep_traits<S, RepFrom>::DataType;
  using RepDataTo = typename SO3_rep_traits<S, RepTo>::DataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return SO3_convert_to_noncanonical_impl<S, RepTo>::run(
          SO3_convert_to_canonical_impl<S, RepFrom>::run(data));
  }
};

//==============================================================================
// SO3_canonical_inplace_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S>
struct SO3_canonical_group_multiplication_impl
{
  using CanonicalRepDataType
      = typename SO3_rep_traits<S, SO3Canonical>::DataType;

  static const CanonicalRepDataType run(
      const CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    return data * otherData;
  }
};

//==============================================================================
// SO3_canonical_inplace_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S>
struct SO3_canonical_inplace_group_multiplication_impl
{
  using CanonicalRepDataType
      = typename SO3_rep_traits<S, SO3Canonical>::DataType;

  static void run(
      CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    data *= otherData;
  }
};

//==============================================================================
// SO3_convert_rep_to_param:
//==============================================================================

//==============================================================================
template <typename S, typename RepFrom, typename RepTo>
struct SO3_convert_rep_to_param
{
  static typename SO3_rep_traits<S, RepTo>::RepType
  run(const typename SO3_rep_traits<S, RepFrom>::RepType& /*data*/)
  {
    // Generic version:
    // 1. convert data to the canonical data type
    // 2. convert the canonical data type to the canonical parameter type
  }
};

//==============================================================================
//template <typename S, typename Rep1, typename Rep2>
//struct SO3ConvertImpl
//{
//  static
//  void run(traits<)
//};


















//==============================================================================
// SO3_assign_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3To, typename SO3From>
struct SO3_assign_impl
{
  using RepFrom = typename SO3From::Rep;
  using RepTo = typename SO3To::Rep;

  static void run(SO3To& dataTo, const SO3From& dataFrom)
  {
    dataTo.matrix() = SO3_convert_to_noncanonical_impl<S, RepTo>::run(
          SO3_convert_to_canonical_impl<S, RepFrom>::run(dataFrom.matrix()));
  }
};

//==============================================================================
// SO3_canonical_inplace_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B, typename Enable = void>
struct SO3_inplace_group_multiplication_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& data, const SO3B& otherData)
  {
    data.matrix() = SO3_convert_to_noncanonical_impl<S, RepA>::run(
          SO3_canonical_group_multiplication_impl<S>::run(
            SO3_convert_to_canonical_impl<S, RepA>::run(data.matrix()),
            SO3_convert_to_canonical_impl<S, RepB>::run(otherData.matrix())));
  }
};

//==============================================================================
template <typename SO3A, typename SO3B>
struct SO3_inplace_group_multiplication_impl<
    SO3A,
    SO3B,
    //std::enable_if<SO3_is_canonical<SO3A>::value>>
    typename std::enable_if<SO3_is_canonical<SO3A>::value>::type>
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& data, const SO3B& otherData)
  {
    data.matrix() *= SO3_convert_to_canonical_impl<S, RepB>::run(
          otherData.matrix());
  }
};

//==============================================================================
// SO3_assign_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B>
struct SO3_is_approx_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static bool run(const SO3A& dataA, const SO3B& dataB, S tol)
  {
    return SO3_convert_to_canonical_impl<S, RepA>::run(dataA.matrix())
        .isApprox(
          SO3_convert_to_canonical_impl<S, RepB>::run(dataB.matrix()),
          tol);
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3UTILS_HPP_
