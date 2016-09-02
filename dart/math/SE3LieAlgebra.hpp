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

#ifndef DART_MATH_SE3LIEALGEBRA_HPP_
#define DART_MATH_SE3LIEALGEBRA_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
//#include "dart/math/SE3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class se3
{
public:

  using S = S_;

  using LinearMotionType = Eigen::Matrix<S, 3, 1>;
  using AngularMotionType = Eigen::Matrix<S, 3, 1>;

  AngularMotionType& angular()
  {
    return mAngular;
  }

  const AngularMotionType& angular() const
  {
    return mAngular;
  }

  LinearMotionType& linear()
  {
    return mLinear;
  }

  const LinearMotionType& linear() const
  {
    return mLinear;
  }

protected:

  template <typename>
  friend class SE3Base;

  Eigen::Matrix<S, 3, 1> mAngular;
  Eigen::Matrix<S, 3, 1> mLinear;
};

template <typename S>
using SpatialMotion = se3<S>;

template <typename S_>
class dse3
{
public:

  using S = S_;

  using LinearForceType = Eigen::Matrix<S, 3, 1>;
  using AngularForceType = Eigen::Matrix<S, 3, 1>;

  AngularForceType& angular()
  {
    return mAngular;
  }

  const AngularForceType& angular() const
  {
    return mAngular;
  }

  LinearForceType& linear()
  {
    return mLinear;
  }

  const LinearForceType& linear() const
  {
    return mLinear;
  }

protected:

  template <typename>
  friend class SE3Base;

  Eigen::Matrix<S, 3, 1> mAngular;
  Eigen::Matrix<S, 3, 1> mLinear;
};

template <typename S>
using SpatialForce = dse3<S>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SE3LIEALGEBRA_HPP_
