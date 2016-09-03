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

struct AxisAngleRep : SO3Representation {};

template <typename S_>
class SO3<S_, AxisAngleRep> : public SO3Base<SO3<S_, AxisAngleRep>>
{
public:

  using This = SO3<S_, AxisAngleRep>;
  using Base = SO3Base<This>;
  using S = typename Base::S;
  using Rep = typename Base::Rep;

  using RotationMatrixType = typename Base::RotationMatrixType;
  using VectorType = typename Base::VectorType;

  using RepDataType = typename Base::RepDataType;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *;
  using Base::operator *=;

  using Base::getCoordinates;
  using Base::setRepData;
  using Base::getRepData;

  /// \{ \name Constructors

  /// Default constructor. By default, the constructed SO(3) is not identity.
  SO3() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  SO3(const SO3& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  /// Move constructor.
  SO3(SO3&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  /// Construct from Eigen::AngleAxis.
  explicit SO3(const Eigen::AngleAxis<S>& angleAxis)
    : Base(), mRepData(angleAxis)
  {
    // Do nothing
  }

  /// Construct from Eigen::AngleAxis.
  explicit SO3(Eigen::AngleAxis<S>&& angleAxis)
    : Base(), mRepData(std::move(angleAxis))
  {
    // Do nothing
  }

  explicit SO3(const VectorType& axis, S angle)
    : Base(), mRepData(angle, axis)
  {
    // Do nothing
  }

  explicit SO3(VectorType&& axis, S angle)
    : Base(), mRepData(std::move(angle), axis)
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3(const SO3Base<Derived>& other)
    : Base(),
      mRepData(detail::SO3::convert_impl<S, typename Derived::Rep, Rep>::run(
              other.derived().getRepData()))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3(SO3Base<Derived>&& other)
    : Base(),
      mRepData(detail::SO3::convert_impl<S, typename Derived::Rep, Rep>::run(
              std::move(other.derived().getRepData())))
  {
    // Do nothing
  }

  /// \} // Constructors

  /// \{ \name Operators

  /// Assign a SO3 with the same representation.
  SO3& operator=(const SO3& other)
  {
    Base::operator =(other);
    return *this;
  }

  /// Move in a SO3 with the same representation.
  SO3& operator=(SO3&& other)
  {
    Base::operator=(std::move(other));
    return *this;
  }

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const SO3& other)
  {
    if (mRepData.angle() == static_cast<S>(0)
        && other.getRepData().angle() == static_cast<S>(0))
      return true;

    return mRepData.isApprox(other.mRepData, static_cast<S>(0));
  }

  /// \} // Operators

  /// \{ \name Representation properties

  void setAxisAngle(const VectorType& axis, S angle)
  {
    mRepData.axis() = axis;
    mRepData.angle() = angle;
  }

  void setAxis(const VectorType& axis)
  {
    mRepData.axis() = axis;
  }

  const VectorType& getAxis() const
  {
    return mRepData.axis();
  }

  VectorType& getAxis()
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

  S& getAngle()
  {
    return mRepData.angle();
  }

  void setRandom()
  {
    mRepData.axis().setRandom().normalize();
    mRepData.angle() = math::random<S>();
    // TODO(JS): improve
  }

  /// \} // Representation properties

  /// \{ \name \f$SO3\f$ group properties

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

  const SO3 getInverse() const
  {
    return SO3(RepDataType(-mRepData.angle(), mRepData.axis()));
  }

  /// \} // \f$SO3\f$ group properties

  static SO3 Exp(const so3& tangent)
  {
    const S norm = tangent.norm();

    if (norm > static_cast<S>(0))
      return SO3(RepDataType(norm, tangent/norm));
    else
      return SO3(RepDataType(0, VectorType::UnitX()));
  }

  static so3 Log(const SO3& point)
  {
    return point.mRepData.angle() * point.mRepData.axis();
  }

protected:
  template <typename>
  friend class SO3Base;

  RepDataType mRepData{RepDataType()};
};

extern template
class SO3<double, AxisAngleRep>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3AXISANGLE_HPP_
