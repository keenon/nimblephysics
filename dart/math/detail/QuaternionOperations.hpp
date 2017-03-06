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

#ifndef DART_MATH_DETAIL_QUATERNIONOPERATIONS_HPP_
#define DART_MATH_DETAIL_QUATERNIONOPERATIONS_HPP_

#include "dart/math/Quaternion.hpp"

#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {
namespace detail {

//==============================================================================
// SO3AssignSO3ToSO3
//==============================================================================

//==============================================================================
template <typename S>
struct SO3AssignSO3ToSO3<SO3Vector<S>, Quaternion<S>>
{
  static void run(const SO3Vector<S>& from, Quaternion<S>& to)
  {
    // TODO(JS): improve
    const auto& fromData = from.getRepData();
    const auto& norm = fromData.norm();
    const Eigen::AngleAxis<S> eigAa(norm, fromData/norm);

    to.getRepData() = std::move(eigAa);
  }
};

//==============================================================================
template <typename S>
struct SO3AssignSO3ToSO3<Quaternion<S>, SO3Vector<S>>
{
  static void run(const Quaternion<S>& from, SO3Vector<S>& to)
  {
    // TODO(JS): improve
    const auto& fromData = from.getRepData();
    const Eigen::AngleAxis<S> eigAa(fromData);

    to.getRepData() = eigAa.angle() * eigAa.axis();
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_QUATERNIONOPERATIONS_HPP_
