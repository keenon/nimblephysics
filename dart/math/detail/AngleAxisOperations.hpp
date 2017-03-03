/*
 * Copyright (c) 2017, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2017, Personal Robotics Lab, Carnegie Mellon University
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

#ifndef DART_MATH_DETAIL_ANGLEAXISOPERATIONS_HPP_
#define DART_MATH_DETAIL_ANGLEAXISOPERATIONS_HPP_

#include "dart/math/AngleAxis.hpp"

#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {
namespace detail {

//==============================================================================
// SO3AssignEigenToSO3
//==============================================================================

template <typename EigenFrom>
struct SO3AssignEigenToSO3<EigenFrom, AngleAxis<typename EigenFrom::Scalar>>
{
  static void run(const EigenFrom& from, AngleAxis<typename EigenFrom::Scalar>& to)
  {
    to.getRepData() = from;
  }
};

//==============================================================================
// SO3AssignEigenToSO3
//==============================================================================

//==============================================================================
// Specializations for SO3Vector --> AngleAxis
template <typename S>
struct SO3AssignSO3ToSO3<SO3Vector<S>, AngleAxis<S>>
{
  static constexpr bool IsSpecialized = true;

  static void run(const SO3Vector<S>& from, AngleAxis<S>& to)
  {
    const auto& axis = from.getRepData();
    const S norm = axis.norm();

    if (norm > static_cast<S>(0))
      to.setAngleAxis(axis/norm, norm);
    else
      to.setAngleAxis(Eigen::Matrix<S, 3, 1>::UnitX(), static_cast<S>(0));
  }
};

//==============================================================================
// Specializations for AngleAxis --> SO3Vector
template <typename S>
struct SO3AssignSO3ToSO3<AngleAxis<S>, SO3Vector<S>>
{
  static constexpr bool IsSpecialized = true;

  static void run(const AngleAxis<S>& from, SO3Vector<S>& to)
  {
    const Eigen::AngleAxis<S>& aa = from.getRepData();

    to.getRepData() = aa.angle() * aa.axis();
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_ANGLEAXISOPERATIONS_HPP_
