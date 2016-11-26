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

#ifndef DART_MATH_QUATERNION_HPP_
#define DART_MATH_QUATERNION_HPP_

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class Quaternion : public SO3Base<Quaternion<S_>>
{
public:

  using This = Quaternion;
  using Base = SO3Base<Quaternion<S_>>;
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
  Quaternion();

  /// Copy constructor.
  Quaternion(const Quaternion& other);

  /// Move constructor.
  Quaternion(Quaternion&& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  Quaternion(const SO3Base<Derived>& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  Quaternion(SO3Base<Derived>&& other);

  /// Construct from Eigen::Quaternion.
  Quaternion(const Eigen::Quaternion<S>& quat);

  /// Construct from Eigen::Quaternion.
  Quaternion(Eigen::Quaternion<S>&& quat);

  // TODO(JS): Add more constructs that takes raw components of quaternions

  /// \} // Constructors

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Assign a SO3 with the same representation.
  Quaternion& operator=(const Quaternion& other);

  /// Move in a SO3 with the same representation.
  Quaternion& operator=(Quaternion&& other);

  Quaternion& operator=(const Eigen::AngleAxis<S>& quat);

  Quaternion& operator=(Eigen::AngleAxis<S>&& quat);

  template <typename QuatDerived>
  Quaternion& operator=(const Eigen::QuaternionBase<QuatDerived>& quat);

  template <typename QuatDerived>
  Quaternion& operator=(Eigen::QuaternionBase<QuatDerived>&& quat);

  template <typename Derived>
  Quaternion& operator=(const Eigen::MatrixBase<Derived>& matrix);

  template <typename Derived>
  Quaternion& operator=(Eigen::MatrixBase<Derived>&& matrix);

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const Quaternion& other);

  /// \} // Operators

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  void setQuaternion(const RepData& quat);
  // TODO(JS): Rename

  const RepData& getQuaternion() const;
  // TODO(JS): Rename

  void setQuaternion(S w, S x, S y, S z);
  // TODO(JS): Rename

//  void fromVector(const Vector& vector);

//  Vector toVector() const;

  void setW(S w);

  void setX(S x);

  void setY(S y);

  void setZ(S z);

  S getW() const;

  S getX() const;

  S getY() const;

  S getZ() const;

  void setRandom();

  /// \} // Representation properties

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

  void setIdentity();

  bool isIdentity();

  void invert();

  const Quaternion getInverse() const;

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

extern template
class Quaternion<double>;

} // namespace math
} // namespace dart

#include "dart/math/detail/Quaternion-impl.hpp"

#endif // DART_MATH_QUATERNION_HPP_
