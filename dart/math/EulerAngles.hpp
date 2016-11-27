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

template <typename S_, int index0, int index1, int index2>
class EulerAngles : public SO3Base<EulerAngles<S_, index0, index1, index2>>
{
public:

  using This = EulerAngles;
  using Base = SO3Base<EulerAngles<S_, index0, index1, index2>>;
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

  //----------------------------------------------------------------------------
  /// \{ \name Constructors
  //----------------------------------------------------------------------------

  /// Default constructor. By default, the constructed SO(3) is not identity.
  EulerAngles() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  EulerAngles(const EulerAngles& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  /// Move constructor.
  EulerAngles(EulerAngles&& other) : Base(), mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  EulerAngles(const SO3Base<Derived>& other)
    : Base(),
      mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
               other.getRepData()))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  EulerAngles(SO3Base<Derived>&& other)
    : Base(),
      mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
               other.getRepData()))
  {
    // Do nothing
  }

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  void setAngles(const Eigen::Matrix<S, 3, 1>& angles)
  {
    mRepData = angles;
  }

  Eigen::Matrix<S, 3, 1> getAngles() const
  {
    return mRepData;
  }

  void setAngles(S angle0, S angle1, S angle2)
  {
    mRepData << angle0, angle1, angle2;
  }

  template <int index>
  void setAngle(S angle)
  {
    static_assert(0 <= index && index <= 2, "Invalid index");
    mRepData[index] = angle;
  }

  template <int index>
  S getAngle() const
  {
    static_assert(0 <= index && index <= 2, "Invalid index");
    return mRepData[index];
  }

  void setRandom()
  {
    mRepData.setRandom();
  }

  /// \} // Representation properties

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

//  template <typename OtherDerived>
//  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const;

  void setIdentity()
  {
    mRepData.setZero();
  }

  bool isIdentity()
  {
    return mRepData == RepData::Zero();
  }

  void invert()
  {
    mRepData.reverseInPlace();
  }

  const EulerAngles getInverse() const
  {
    return EulerAngles(RepData(-mRepData.reverse()));
  }

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

template <int index0, int index1, int index2>
using EulerAnglesf = EulerAngles<float, index0, index1, index2>;

template <int index0, int index1, int index2>
using EulerAnglesd = EulerAngles<double, index0, index1, index2>;

// Proper Euler angles (x-y-x, x-z-x, y-x-y, y-z-y, z-x-z, z-y-z)

using EulerXYXf = EulerAngles<float, 0, 1, 0>;
using EulerXYXd = EulerAngles<double, 0, 1, 0>;

using EulerXZXf = EulerAngles<float, 0, 2, 0>;
using EulerXZXd = EulerAngles<double, 0, 2, 0>;

using EulerYXYf = EulerAngles<float, 1, 0, 1>;
using EulerYXYd = EulerAngles<double, 1, 0, 1>;

using EulerYZYf = EulerAngles<float, 1, 2, 1>;
using EulerYZYd = EulerAngles<double, 1, 2, 1>;

using EulerZXZf = EulerAngles<float, 2, 0, 2>;
using EulerZXZd = EulerAngles<double, 2, 0, 2>;

using EulerZYZf = EulerAngles<float, 2, 1, 2>;
using EulerZYZd = EulerAngles<double, 2, 1, 2>;

// Tait–Bryan angles (x-y-z, x-z-y, y-x-z, y-z-x, z-x-y, z-y-x)

using EulerXYZf = EulerAngles<float, 0, 1, 2>;
using EulerXYZd = EulerAngles<double, 0, 1, 2>;

using EulerXZYf = EulerAngles<float, 0, 2, 1>;
using EulerXZYd = EulerAngles<double, 0, 2, 1>;

using EulerYXZf = EulerAngles<float, 1, 0, 2>;
using EulerYXZd = EulerAngles<double, 1, 0, 2>;

using EulerYZXf = EulerAngles<float, 1, 2, 0>;
using EulerYZXd = EulerAngles<double, 1, 2, 0>;

using EulerZXYf = EulerAngles<float, 2, 0, 1>;
using EulerZXYd = EulerAngles<double, 2, 0, 1>;

using EulerZYXf = EulerAngles<float, 2, 1, 0>;
using EulerZYXd = EulerAngles<double, 2, 1, 0>;

//extern template
//class EulerAngles<double>;

} // namespace math
} // namespace dart

#endif // DART_MATH_EULERANGLES_HPP_
