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

#ifndef DART_MATH_SO3_HPP_
#define DART_MATH_SO3_HPP_

#include "dart/math/SO3AxisAngle.hpp"
#include "dart/math/SO3Quaternion.hpp"
#include "dart/math/SO3RotationMatrix.hpp"
#include "dart/math/SO3RotationVector.hpp"
#include "dart/math/SO3AxisAngle.hpp"

namespace dart {
namespace math {

template <typename S> using SO3Matrix     = SO3<S, RotationMatrixRep>;
template <typename S> using SO3Vector     = SO3<S, RotationVectorRep>;
template <typename S> using SO3AxisAngle  = SO3<S, AxisAngleRep>;
template <typename S> using SO3Quaternion = SO3<S, QuaternionRep>;

using SO3Matrixf = SO3Matrix<float>;
using SO3Matrixd = SO3Matrix<double>;

using SO3Vectorf = SO3Vector<float>;
using SO3Vectord = SO3Vector<double>;

using SO3AxisAnglef = SO3AxisAngle<float>;
using SO3AxisAngled = SO3AxisAngle<double>;

using SO3Quaternionf = SO3Quaternion<float>;
using SO3Quaterniond = SO3Quaternion<double>;

using SO3f = SO3<float, DefaultSO3CanonicalRep>;
using SO3d = SO3<double, DefaultSO3CanonicalRep>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3_HPP_
