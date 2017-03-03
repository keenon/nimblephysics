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

#ifndef DART_MATH_SO3TROTATIONVECTOR_HPP_
#define DART_MATH_SO3TROTATIONVECTOR_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"
#include "dart/math/detail/SO3VectorOperations.hpp"

namespace dart {
namespace math {

template <typename S_>
class SO3Vector : public SO3Base<SO3Vector<S_>>
{
public:

  using This = SO3Vector<S_>;
  using Base = SO3Base<This>;
  using S = typename Base::S;

  using Matrix3 = typename Base::Matrix3;
  using Vector3 = typename Base::Vector3;

  using RepData = typename Base::RepData;
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
  SO3Vector();

  /// Copy constructor.
  SO3Vector(const SO3Vector& other);

  /// Move constructor.
  SO3Vector(SO3Vector&& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Vector(const SO3Base<Derived>& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Vector(SO3Base<Derived>&& other);

  /// Construct from a raw rotation vector where the dimension is 3x1.
  template <typename Derived>
  SO3Vector(const Eigen::MatrixBase<Derived>& matrix);

  /// Construct from a raw rotation matrix where the dimension is 3x3.
  template <typename Derived>
  SO3Vector(Eigen::MatrixBase<Derived>&& matrix);

  /// \} // Constructors

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Assign a SO3 with the same representation.
  SO3Vector& operator=(const SO3Vector& other);

  /// Move in a SO3 with the same representation.
  SO3Vector& operator=(SO3Vector&& other);

  SO3Vector& operator=(const Eigen::AngleAxis<S>& quat);

  SO3Vector& operator=(Eigen::AngleAxis<S>&& quat);

  template <typename QuatDerived>
  SO3Vector& operator=(const Eigen::QuaternionBase<QuatDerived>& quat);

  template <typename QuatDerived>
  SO3Vector& operator=(Eigen::QuaternionBase<QuatDerived>&& quat);

  template <typename Derived>
  SO3Vector& operator=(const Eigen::MatrixBase<Derived>& matrix);
  // TODO(JS): take rotation vector as well using a template struct

  template <typename Derived>
  SO3Vector& operator=(Eigen::MatrixBase<Derived>&& matrix);

  /// Whether \b exactly equal to a SO3.
  bool operator==(const SO3Vector& other);

  /// \} // Operators

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  template <typename MatrixDerived>
  void fromRotationVector(const Eigen::MatrixBase<MatrixDerived>& vector);

  template <typename MatrixDerived>
  void fromRotationVector(Eigen::MatrixBase<MatrixDerived>&& vector);

  Vector3 toRotationVector() const;

  void setRotationVector(const Vector3& axisAngle);

  const Vector3& getRotationVector() const;

  void setRandom();

  /// \} // Representation properties

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

  void setIdentity();

  bool isIdentity();

  void invert();

  const SO3Vector getInverse() const;

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using SO3Vectorf = SO3Vector<float>;
using SO3Vectord = SO3Vector<double>;

extern template
class SO3Vector<double>;

} // namespace math
} // namespace dart

#include "dart/math/detail/SO3Vector-impl.hpp"

#endif // DART_MATH_SO3TROTATIONVECTOR_HPP_
