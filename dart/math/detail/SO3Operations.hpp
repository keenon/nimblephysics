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

namespace dart {
namespace math {

struct SO3Representation;

struct RotationMatrixRep;
struct AxisAngleRep;
struct QuaternionRep;
struct RotationVectorRep;

using SO3CanonicalRep = RotationMatrixRep;

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

  using Canonical = SO3<S, SO3CanonicalRep>;
};

namespace SO3 {

//==============================================================================
// rep_traits:
// Traits for different SO3 representations
//==============================================================================

//==============================================================================
template <typename S, typename Rep_>
struct rep_traits;

//==============================================================================
template <typename S_>
struct rep_traits<S_, RotationMatrixRep>
{
  using S = S_;
  using Rep = RotationMatrixRep;
  using RepDataType = Eigen::Matrix<S, 3, 3>;
};

//==============================================================================
template <typename S_>
struct rep_traits<S_, AxisAngleRep>
{
  using S = S_;
  using Rep = AxisAngleRep;
  using RepDataType = Eigen::Matrix<S, 3, 1>;
  // TODO(JS): Change to Eigen::AngleAxis<S>
};

//==============================================================================
template <typename S_>
struct rep_traits<S_, QuaternionRep>
{
  using S = S_;
  using Rep = QuaternionRep;
  using RepDataType = Eigen::Quaternion<S>;
};

//==============================================================================
template <typename S_>
struct rep_traits<S_, RotationVectorRep>
{
  using S = S_;
  using Rep = QuaternionRep;
  using RepDataType = Eigen::Matrix<S, 3, 1>;
};

//==============================================================================
// convert_to_canonical_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepFrom>
struct convert_to_canonical_impl;

//==============================================================================
template <typename S>
struct convert_to_canonical_impl<S, SO3CanonicalRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct convert_to_canonical_impl<S, AxisAngleRep>
{
  using RepDataFrom = typename rep_traits<S, AxisAngleRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return ::dart::math::expMapRot(data);
  }
};

//==============================================================================
template <typename S>
struct convert_to_canonical_impl<S, QuaternionRep>
{
  using RepDataFrom = typename rep_traits<S, QuaternionRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data.toRotationMatrix();
  }
};

//==============================================================================
template <typename S>
struct convert_to_canonical_impl<S, RotationVectorRep>
{
  using RepDataFrom = typename rep_traits<S, RotationVectorRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return ::dart::math::expMapRot(data);
  }
};

//==============================================================================
// convert_to_noncanonical_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepTo>
struct convert_to_noncanonical_impl;

//==============================================================================
template <typename S>
struct convert_to_noncanonical_impl<S, SO3CanonicalRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return canonicalData;
  }
};

//==============================================================================
template <typename S>
struct convert_to_noncanonical_impl<S, AxisAngleRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, AxisAngleRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return ::dart::math::logMap(canonicalData);
  }
};

//==============================================================================
template <typename S>
struct convert_to_noncanonical_impl<S, QuaternionRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, QuaternionRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return RepDataTo(canonicalData);
  }
};

//==============================================================================
template <typename S>
struct convert_to_noncanonical_impl<S, RotationVectorRep>
{
  using RepDataFrom = typename rep_traits<S, SO3CanonicalRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RotationVectorRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& canonicalData)
  {
    return ::dart::math::logMap(canonicalData);
  }
};

//==============================================================================
// convert_impl:
//==============================================================================

//==============================================================================
template <typename S, typename RepFrom, typename RepTo>
struct convert_impl
{
  using RepDataFrom = typename rep_traits<S, RepFrom>::RepDataType;
  using RepDataTo = typename rep_traits<S, RepTo>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return convert_to_noncanonical_impl<S, RepTo>::run(
          convert_to_canonical_impl<S, RepFrom>::run(data));
  }
};

//==============================================================================
// canonical_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S>
struct canonical_group_multiplication_impl
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
// canonical_inplace_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename S>
struct canonical_inplace_group_multiplication_impl
{
  using CanonicalRepDataType
      = typename rep_traits<S, SO3CanonicalRep>::RepDataType;

  static void run(
      CanonicalRepDataType& data, const CanonicalRepDataType& otherData)
  {
    data *= otherData;
  }
};

} // namespace SO3
} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3OPERATIONS_HPP_
