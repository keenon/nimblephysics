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
#include "dart/math/detail/AngleAxisOperations.hpp"

namespace dart {
namespace math {

template <typename S_>
class AngleAxis : public SO3Base<AngleAxis<S_>>
{
public:

  using This = AngleAxis<S_>;
  using Base = SO3Base<This>;
  using S = typename Base::S;

  using Matrix3 = typename Base::Matrix3;
  using Axis = Eigen::Matrix<double, 3, 1>;

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

  //----------------------------------------------------------------------------
  /// \{ \name Constructors
  //----------------------------------------------------------------------------

  /// Default constructor. By default, the constructed SO(3) is not identity.
  AngleAxis();

  /// Copy constructor.
  AngleAxis(const AngleAxis& other);

  /// Move constructor.
  AngleAxis(AngleAxis&& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  AngleAxis(const SO3Base<Derived>& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  AngleAxis(SO3Base<Derived>&& other);

  /// Construct from Eigen::AngleAxis.
  explicit AngleAxis(const Eigen::AngleAxis<S>& angleAxis);

  /// Construct from Eigen::AngleAxis.
  explicit AngleAxis(Eigen::AngleAxis<S>&& angleAxis);

  /// Construct from axis and angle
  explicit AngleAxis(const Axis& axis, S angle);

  /// Construct from axis and angle
  explicit AngleAxis(Axis&& axis, S angle);

  /// Construct from quaternion
  template <typename QuatDerived>
  explicit AngleAxis(const Eigen::QuaternionBase<QuatDerived>& q);

  /// \} // Constructors

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Assign a SO3 with the same representation.
  AngleAxis& operator=(const AngleAxis& other);

  /// Move in a SO3 with the same representation.
  AngleAxis& operator=(AngleAxis&& other);

  AngleAxis& operator=(const Eigen::AngleAxis<S>& quat);

  AngleAxis& operator=(Eigen::AngleAxis<S>&& quat);

  template <typename QuatDerived>
  AngleAxis& operator=(const Eigen::QuaternionBase<QuatDerived>& quat);

  template <typename QuatDerived>
  AngleAxis& operator=(Eigen::QuaternionBase<QuatDerived>&& quat);

  template <typename Derived>
  AngleAxis& operator=(const Eigen::MatrixBase<Derived>& matrix);

  template <typename Derived>
  AngleAxis& operator=(Eigen::MatrixBase<Derived>&& matrix);

  /// Whether \b exactly equal to a SO3.
  bool operator==(const AngleAxis& other);

  /// \} // Operators

  //----------------------------------------------------------------------------
  /// \{ \name Conversions
  //----------------------------------------------------------------------------

  /// Set from the canonical type
  void fromCanonical(const SO3Matrix<S>& mat);

  /// Convert to the canonical type
  SO3Matrix<S_> toCanonical() const;

  /// \} // Conversions

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  void setAngleAxis(const Axis& axis, S angle);

  void setAxis(const Axis& axis);

  const Axis& getAxis() const;

  void setAngle(const S angle);

  S getAngle() const;

  void setRandom();

  /// \} // Representation properties

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

//  template <typename OtherDerived>
//  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const;

  void setIdentity();

  bool isIdentity();

  void invert();

  const AngleAxis getInverse() const;

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

#include "dart/math/detail/AngleAxis-impl.hpp"

#endif // DART_MATH_SO3AXISANGLE_HPP_
