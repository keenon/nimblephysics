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

#ifndef DART_MATH_SO3AXISANGLE_HPP_
#define DART_MATH_SO3AXISANGLE_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class AngleAxis : public SO3Base<AngleAxis<S_>>
{
public:

  using This = AngleAxis<S_>;
  using Base = SO3Base<This>;
  using S = typename Base::S;

  using RotationMatrix = typename Base::RotationMatrix;
  using RotationVector = typename Base::RotationVector;

  using RepData = typename Base::RepData;

  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  //using Base::operator =;
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
  AngleAxis() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  AngleAxis(const AngleAxis& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  /// Move constructor.
  AngleAxis(AngleAxis&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  AngleAxis(const SO3Base<Derived>& other)
    : Base(),
      mRepData(detail::so3_operations::so3_convert_impl<S, Derived, This>::run(
              other.getRepData()))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  AngleAxis(SO3Base<Derived>&& other)
    : Base(),
      mRepData(detail::so3_operations::so3_convert_impl<S, Derived, This>::run(
              std::move(other.getRepData())))
  {
    // Do nothing
  }

  /// Construct from Eigen::AngleAxis.
  explicit AngleAxis(const Eigen::AngleAxis<S>& angleAxis)
    : Base(), mRepData(angleAxis)
  {
    // Do nothing
  }

  /// Construct from Eigen::AngleAxis.
  explicit AngleAxis(Eigen::AngleAxis<S>&& angleAxis)
    : Base(), mRepData(std::move(angleAxis))
  {
    // Do nothing
  }

  /// Construct from axis and angle
  explicit AngleAxis(const RotationVector& axis, S angle)
    : Base(), mRepData(angle, axis)
  {
    // Do nothing
  }

  /// Construct from axis and angle
  explicit AngleAxis(RotationVector&& axis, S angle)
    : Base(), mRepData(std::move(angle), axis)
  {
    // Do nothing
  }

  /// Construct from quaternion
  template <typename QuatDerived>
  explicit AngleAxis(const Eigen::QuaternionBase<QuatDerived>& q)
  {
    mRepData = q;
  }

  /// \} // Constructors

  /// \{ \name Operators

  /// Assign a SO3 with the same representation.
  AngleAxis& operator=(const AngleAxis& other)
  {
    mRepData = other.mRepData;
    return *this;
  }

  /// Move in a SO3 with the same representation.
  AngleAxis& operator=(AngleAxis&& other)
  {
    mRepData = std::move(other.mRepData);
    return *this;
  }

  AngleAxis& operator=(const Eigen::AngleAxis<S>& quat)
  {
    mRepData = quat;
    return *this;
  }

  AngleAxis& operator=(Eigen::AngleAxis<S>&& quat)
  {
    mRepData = std::move(quat);
    return *this;
  }

  template <typename QuatDerived>
  AngleAxis& operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
  {
    mRepData = quat;
    return *this;
  }

  template <typename QuatDerived>
  AngleAxis& operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
  {
    mRepData = std::move(quat);
    return *this;
  }

  template <typename Derived>
  AngleAxis& operator=(const Eigen::MatrixBase<Derived>& matrix)
  {
    mRepData = matrix;
    return *this;
  }

  template <typename Derived>
  AngleAxis& operator=(Eigen::MatrixBase<Derived>&& matrix)
  {
    mRepData = std::move(matrix);
    return *this;
  }

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const AngleAxis& other)
  {
    if (mRepData.angle() == static_cast<S>(0)
        && other.getRepData().angle() == static_cast<S>(0))
      return true;

    return mRepData.isApprox(other.mRepData, static_cast<S>(0));
  }

  /// \} // Operators

  /// \{ \name Representation properties

  void setAngleAxis(const RotationVector& axis, S angle)
  {
    mRepData.axis() = axis;
    mRepData.angle() = angle;
  }

  void setAxis(const RotationVector& axis)
  {
    mRepData.axis() = axis;
  }

  const RotationVector& getAxis() const
  {
    return mRepData.axis();
  }

  void setAngle(const S angle)
  {
    mRepData.angle() = angle;
  }

  S getAngle() const
  {
    return mRepData.angle();
  }

  void setRandom()
  {
    mRepData.axis().setRandom().normalize();
    mRepData.angle() = math::random(-1.0, 1.0);
    // TODO(JS): improve
  }

  /// \} // Representation properties

  /// \{ \name SO3 group operations

//  template <typename OtherDerived>
//  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const
//  {
//    return detail::SO3::group_is_approx_impl<Derived, OtherDerived>::run(
//          derived(), other.derived(), tol);
//  }

  void setIdentity()
  {
    mRepData.angle() = static_cast<S>(0);
  }

  bool isIdentity()
  {
    return mRepData.angle() == static_cast<S>(0);
  }

  void invert()
  {
    mRepData.angle() *= static_cast<S>(-1);
  }

  const AngleAxis getInverse() const
  {
    return AngleAxis(RepData(-mRepData.angle(), mRepData.axis()));
  }

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using AngleAxisf = AngleAxis<float>;
using AngleAxisd = AngleAxis<double>;

extern template
class AngleAxis<double>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3AXISANGLE_HPP_
