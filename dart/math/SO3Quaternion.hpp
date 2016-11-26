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

#ifndef DART_MATH_SO3QUATERNION_HPP_
#define DART_MATH_SO3QUATERNION_HPP_

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class SO3Quaternion : public SO3Base<SO3Quaternion<S_>>
{
public:

  using This = SO3Quaternion;
  using Base = SO3Base<SO3Quaternion<S_>>;
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
  SO3Quaternion() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  SO3Quaternion(const SO3Quaternion& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  /// Move constructor.
  SO3Quaternion(SO3Quaternion&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Quaternion(const SO3Base<Derived>& other)
    : Base(),
      mRepData(other.getRepData())
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Quaternion(SO3Base<Derived>&& other)
    : Base(),
      mRepData(detail::so3_operations::so3_convert_impl<
          S, Derived, This>::run(std::move(other.getRepData())))
  {
    // Do nothing
  }

  /// Construct from Eigen::Quaternion.
  SO3Quaternion(const Eigen::Quaternion<S>& quat) : Base(), mRepData(quat)
  {
    // Do nothing
  }

  /// Construct from Eigen::Quaternion.
  SO3Quaternion(Eigen::Quaternion<S>&& quat) : Base(), mRepData(std::move(quat))
  {
    // Do nothing
  }

  // TODO(JS): Add more constructs that takes raw components of quaternions

  /// \} // Constructors

  /// \{ \name Operators

  /// Assign a SO3 with the same representation.
  SO3Quaternion& operator=(const SO3Quaternion& other)
  {
    mRepData = other.mRepData;
    return *this;
  }

  /// Move in a SO3 with the same representation.
  SO3Quaternion& operator=(SO3Quaternion&& other)
  {
    mRepData = std::move(other.mRepData);
    return *this;
  }

  SO3Quaternion& operator=(const Eigen::AngleAxis<S>& quat)
  {
    mRepData = quat;
    return *this;
  }

  SO3Quaternion& operator=(Eigen::AngleAxis<S>&& quat)
  {
    mRepData = std::move(quat);
    return *this;
  }

  template <typename QuatDerived>
  SO3Quaternion& operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
  {
    mRepData = quat;
    return *this;
  }

  template <typename QuatDerived>
  SO3Quaternion& operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
  {
    mRepData = std::move(quat);
    return *this;
  }

  template <typename Derived>
  SO3Quaternion& operator=(const Eigen::MatrixBase<Derived>& matrix)
  {
    mRepData = matrix;
    return *this;
  }

  template <typename Derived>
  SO3Quaternion& operator=(Eigen::MatrixBase<Derived>&& matrix)
  {
    mRepData = std::move(matrix);
    return *this;
  }

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const SO3Quaternion& other)
  {
    return mRepData.isApprox(other.mRepData, static_cast<S>(0));
  }

  /// \} // Operators

  /// \{ \name Representation properties

  void setQuaternion(const RepData& quat)
  {
    mRepData = quat;
  }

  const RepData& getQuaternion() const
  {
    return mRepData;
  }

  void setQuaternion(S w, S x, S y, S z)
  {
    mRepData.w() = w;
    mRepData.x() = x;
    mRepData.y() = y;
    mRepData.z() = z;
  }

//  void fromVector(const Vector& vector)
//  {
//    mRepData.vec() = vector;
//  }

//  Vector toVector() const
//  {
//    return mRepData.vec();
//  }

  void setW(S w)
  {
    mRepData.w() = w;
  }

  void setX(S x)
  {
    mRepData.x() = x;
  }

  void setY(S y)
  {
    mRepData.y() = y;
  }

  void setZ(S z)
  {
    mRepData.z() = z;
  }

  S getW() const
  {
    return mRepData.w();
  }

  S getX() const
  {
    return mRepData.x();
  }

  S getY() const
  {
    return mRepData.y();
  }

  S getZ() const
  {
    return mRepData.z();
  }

  void setRandom()
  {
    // TODO(JS): This code was copied from
    // https://bitbucket.org/eigen/eigen/commits/5d78b569eac3/#LEigen/src/Geometry/Quaternion.hT621
    // This should be replaced to:
    // mRepData = RepData::UnitRandom() once the commit is released
    using std::sqrt;
    using std::sin;
    using std::cos;

    const S u1 = Eigen::internal::random<S>(0, 1);
    const S u2 = Eigen::internal::random<S>(0, 2*constants<S>::pi());
    const S u3 = Eigen::internal::random<S>(0, 2*constants<S>::pi());
    const S a = sqrt(1 - u1);
    const S b = sqrt(u1);
    mRepData = RepData(a * sin(u2), a * cos(u2), b * sin(u3), b * cos(u3));
  }

  /// \} // Representation properties

  /// \{ \name SO3 group operations

  void setIdentity()
  {
    mRepData.setIdentity();
  }

  bool isIdentity()
  {
    return mRepData.coeffs() == Eigen::Matrix<S, 4, 1>::Zero();
    // TODO(JS): double-check if this is correct
  }

  void invert()
  {
    mRepData = mRepData.conjugate();
  }

  const SO3Quaternion getInverse() const
  {
    return SO3Quaternion(mRepData.conjugate());
  }

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using SO3Quaternionf = SO3Quaternion<float>;
using SO3Quaterniond = SO3Quaternion<double>;

extern template
class SO3Quaternion<double>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3QUATERNION_HPP_
