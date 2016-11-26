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

#ifndef DART_MATH_EULERANGLES_HPP_
#define DART_MATH_EULERANGLES_HPP_

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class EulerAngles : public SO3Base<EulerAngles<S_>>
{
public:

  using This = EulerAngles;
  using Base = SO3Base<EulerAngles<S_>>;
  using S = S_;

  using RotationMatrix = typename Base::RotationMatrix;

  using RepData = typename Base::RepData;
  // TODO(JS): Rename to Data

  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *;
  using Base::operator *=;

//  using Base::getCoordinates;
  using Base::setRepData;
  using Base::getRepData;

  using Base::Exp;
  using Base::setExp;
  using Base::Log;
  using Base::getLog;

  /// \{ \name Constructors

  /// Default constructor. By default, the constructed SO(3) is not identity.
  EulerAngles() : Base()
  {
    // Do nothing
  }

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using EulerAnglesf = EulerAngles<float>;
using EulerAnglesd = EulerAngles<double>;

extern template
class EulerAngles<double>;

} // namespace math
} // namespace dart

#endif // DART_MATH_EULERANGLES_HPP_
