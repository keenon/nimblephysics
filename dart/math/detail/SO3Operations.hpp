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
  using RepDataType = Eigen::AngleAxis<S>;
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
//
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
//
//==============================================================================

//==============================================================================

template <typename S>
Eigen::Matrix<S, 3, 1> log(const Eigen::Matrix<S, 3, 3>& R)
{
  Eigen::AngleAxis<S> aa(R);

  return aa.angle()*aa.axis();
}

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
    return data.toRotationMatrix();
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
    return exp(data);
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
    return RepDataTo(canonicalData);
    // Above is identical to:
    // return Eigen::AngleAxis<S>(canonicalData);
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
    // Above is identical to:
    // return Eigen::Quaternion<S>(canonicalData);
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
    return log(canonicalData);
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
template <typename S, typename Rep>
struct convert_impl<S, Rep, Rep>
{
  using RepData = typename rep_traits<S, Rep>::RepDataType;

  static const RepData run(const RepData& data)
  {
    return data;
  }
};

//==============================================================================
template <typename S>
struct convert_impl<S, RotationVectorRep, AxisAngleRep>
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
template <typename S>
struct convert_impl<S, AxisAngleRep, RotationVectorRep>
{
  using RepDataFrom = typename rep_traits<S, AxisAngleRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, RotationVectorRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data.angle() * data.axis();
  }
};

//==============================================================================
//template <typename S>
//struct convert_impl<S, RotationVectorRep, QuaternionRep>
//{
//  using RepDataFrom = typename rep_traits<S, RotationVectorRep>::RepDataType;
//  using RepDataTo = typename rep_traits<S, QuaternionRep>::RepDataType;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
//template <typename S>
//struct convert_impl<S, QuaternionRep, RotationVectorRep>
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
struct convert_impl<S, QuaternionRep, AxisAngleRep>
{
  using RepDataFrom = typename rep_traits<S, QuaternionRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, AxisAngleRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }
};

//==============================================================================
template <typename S>
struct convert_impl<S, AxisAngleRep, QuaternionRep>
{
  using RepDataFrom = typename rep_traits<S, AxisAngleRep>::RepDataType;
  using RepDataTo = typename rep_traits<S, QuaternionRep>::RepDataType;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
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
