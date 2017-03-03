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

#ifndef DART_MATH_DETAIL_EULERANGLESOPERATIONS_HPP_
#define DART_MATH_DETAIL_EULERANGLESOPERATIONS_HPP_

#include "dart/math/EulerAngles.hpp"

#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {
namespace detail {

// Proper Euler angles (x-y-x, x-z-x, y-x-y, y-z-y, z-x-z, z-y-z) and
// Tait–Bryan angles (x-y-z, x-z-y, y-x-z, y-z-x, z-x-y, z-y-x)
#define DART_EULER_ANGLES_EXPAND_FOR_PREDEFINED_ANGLE_TYPES(_)\
  _(0, 1, 0)\
  _(0, 2, 0)\
  _(1, 0, 1)\
  _(1, 2, 1)\
  _(2, 0, 2)\
  _(2, 1, 2)\
  _(0, 1, 2)\
  _(0, 2, 1)\
  _(1, 0, 2)\
  _(1, 2, 0)\
  _(2, 0, 1)\
  _(2, 1, 0)

//==============================================================================
// SO3AssignEigenToSO3
//==============================================================================

//==============================================================================
#define DART_SO3_ASSIGN_EIGEN_MATRIX_TO_EULER_ANGLES(id0, id1, id2)\
template <typename EigenFrom>\
struct SO3AssignEigenToSO3<\
  EigenFrom,\
  EulerAngles<typename EigenFrom::Scalar, id0, id1, id2>,\
  typename std::enable_if<\
      SO3IsEigenMatrixBase<EigenFrom>::value\
  >::type>\
{\
  using S = typename EigenFrom::Scalar;\
  static void run(const EigenFrom& from, EulerAngles<S, id0, id1, id2>& to)\
  {\
    to.getRepData() = math::matrixToEulerAngles<S, id0, id1, id2>(from);\
  }\
};

//==============================================================================
DART_EULER_ANGLES_EXPAND_FOR_PREDEFINED_ANGLE_TYPES(
    DART_SO3_ASSIGN_EIGEN_MATRIX_TO_EULER_ANGLES)

//==============================================================================
#define DART_SO3_ASSIGN_EIGEN_ROTATIONBASE_TO_EULER_ANGLES(id0, id1, id2)\
  template <typename EigenFrom>\
  struct SO3AssignEigenToSO3<\
    EigenFrom,\
    EulerAngles<typename EigenFrom::Scalar, id0, id1, id2>,\
    typename std::enable_if<\
        SO3IsEigenRotationBase<EigenFrom>::value\
    >::type>\
{\
  using S = typename EigenFrom::Scalar;\
  static void run(const EigenFrom& from, EulerAngles<S, id0, id1, id2>& to)\
  {\
    to.getRepData() = math::matrixToEulerAngles<S, id0, id1, id2>(from.toRotationMatrix());\
  }\
};

//==============================================================================
DART_EULER_ANGLES_EXPAND_FOR_PREDEFINED_ANGLE_TYPES(
    DART_SO3_ASSIGN_EIGEN_ROTATIONBASE_TO_EULER_ANGLES)

//==============================================================================
// SO3AssignSO3ToEigen
//==============================================================================

#define DART_SO3_ASSIGN_EULER_ANGLES_TO_EIGEN(id0, id1, id2)\
  template <typename EigenTo>\
  struct SO3AssignSO3ToEigen<\
    EulerAngles<typename EigenTo::Scalar, id0, id1, id2>,\
    EigenTo,\
    typename std::enable_if<\
        SO3IsEigen<EigenTo>::value\
    >::type>\
{\
  using S = typename EigenTo::Scalar;\
  static void run(const EulerAngles<S, id0, id1, id2>& from, EigenTo& to)\
  {\
    to = math::eulerAnglesToMatrix<S, id0, id1, id2>(from.getRepData());\
  }\
};

//==============================================================================
DART_EULER_ANGLES_EXPAND_FOR_PREDEFINED_ANGLE_TYPES(
    DART_SO3_ASSIGN_EULER_ANGLES_TO_EIGEN)

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_EULERANGLESOPERATIONS_HPP_
