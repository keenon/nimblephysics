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
  EulerAngles();

  /// Copy constructor.
  EulerAngles(const EulerAngles& other);

  /// Move constructor.
  EulerAngles(EulerAngles&& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  EulerAngles(const SO3Base<Derived>& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  EulerAngles(SO3Base<Derived>&& other);

  /// Construct from a raw rotation vector where the dimension is 3x1.
  template <typename Derived>
  EulerAngles(const Eigen::MatrixBase<Derived>& angles);

  /// Construct from a raw rotation matrix where the dimension is 3x3.
  template <typename Derived>
  EulerAngles(Eigen::MatrixBase<Derived>&& angles);

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Assign a SO3 with the same representation.
  EulerAngles& operator=(const EulerAngles& other);

  /// Move in a SO3 with the same representation.
  EulerAngles& operator=(EulerAngles&& other);

  EulerAngles& operator=(const Eigen::AngleAxis<S>& aa);

  EulerAngles& operator=(Eigen::AngleAxis<S>&& aa);

  template <typename QuatDerived>
  EulerAngles& operator=(const Eigen::QuaternionBase<QuatDerived>& quat);

  template <typename QuatDerived>
  EulerAngles& operator=(Eigen::QuaternionBase<QuatDerived>&& quat);

  template <typename Derived>
  EulerAngles& operator=(const Eigen::MatrixBase<Derived>& matrix);

  template <typename Derived>
  EulerAngles& operator=(Eigen::MatrixBase<Derived>&& matrix);

  /// Whether \b exactly equal to a SO3.
  bool operator==(const EulerAngles& other);

  /// \} // Operators

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  void setAngles(const Eigen::Matrix<S, 3, 1>& angles);

  Eigen::Matrix<S, 3, 1> getAngles() const;

  void setAngles(S angle0, S angle1, S angle2);

  template <int index>
  void setAngle(S angle);

  template <int index>
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

  const EulerAngles getInverse() const;

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

#define DART_TYPEDEF_EULER_ANGLES(XYZ, id0, id1, id2)\
  using Euler##XYZ##f = EulerAngles<float, id0, id1, id2>;\
  using Euler##XYZ##d = EulerAngles<double, id0, id1, id2>;

// Proper Euler angles (x-y-x, x-z-x, y-x-y, y-z-y, z-x-z, z-y-z)
DART_TYPEDEF_EULER_ANGLES(XYX, 0, 1, 0)
DART_TYPEDEF_EULER_ANGLES(XZX, 0, 2, 0)
DART_TYPEDEF_EULER_ANGLES(YXY, 1, 0, 1)
DART_TYPEDEF_EULER_ANGLES(YZY, 1, 2, 1)
DART_TYPEDEF_EULER_ANGLES(ZXZ, 2, 0, 2)
DART_TYPEDEF_EULER_ANGLES(ZYZ, 2, 1, 2)

// Tait–Bryan angles (x-y-z, x-z-y, y-x-z, y-z-x, z-x-y, z-y-x)
DART_TYPEDEF_EULER_ANGLES(XYZ, 0, 1, 2)
DART_TYPEDEF_EULER_ANGLES(XZY, 0, 2, 1)
DART_TYPEDEF_EULER_ANGLES(YXZ, 1, 0, 2)
DART_TYPEDEF_EULER_ANGLES(YZX, 1, 2, 0)
DART_TYPEDEF_EULER_ANGLES(ZXY, 2, 0, 1)
DART_TYPEDEF_EULER_ANGLES(ZYX, 2, 1, 0)

#define DART_EXPLICITLY_INSTANTIATE_EULER_ANGLES(XYZ, id0, id1, id2)\
  extern template\
  class EulerAngles<double, id0, id1, id2>;

DART_EXPLICITLY_INSTANTIATE_EULER_ANGLES(XYZ, 0, 1, 2)
DART_EXPLICITLY_INSTANTIATE_EULER_ANGLES(ZYX, 2, 1, 0)

} // namespace math
} // namespace dart

#include "dart/math/detail/EulerAngles-impl.hpp"

#endif // DART_MATH_EULERANGLES_HPP_
