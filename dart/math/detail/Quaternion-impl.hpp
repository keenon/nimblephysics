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

#ifndef DART_MATH_DETAIL_QUATERNION_IMPL_HPP_
#define DART_MATH_DETAIL_QUATERNION_IMPL_HPP_

#include "dart/math/Quaternion.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename S>
Quaternion<S>::Quaternion() : Base()
{
  // Do nothing
}

//==============================================================================
template <typename S>
Quaternion<S>::Quaternion(const Quaternion& other) : Base(), mRepData(other.mRepData)
{
  // Do nothing
}

//==============================================================================
template <typename S>
Quaternion<S>::Quaternion(Quaternion&& other) : mRepData(std::move(other.mRepData))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
Quaternion<S>::Quaternion(const SO3Base<Derived>& other)
  : Base(),
    mRepData(other.getRepData())
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
Quaternion<S>::Quaternion(SO3Base<Derived>&& other)
  : Base(),
    mRepData(detail::so3_operations::SO3RepDataConvertImpl<
        Derived, This>::run(std::move(other.getRepData())))
{
  // Do nothing
}

//==============================================================================
template <typename S>
Quaternion<S>::Quaternion(const Eigen::Quaternion<S>& quat) : Base(), mRepData(quat)
{
  // Do nothing
}

//==============================================================================
template <typename S>
Quaternion<S>::Quaternion(Eigen::Quaternion<S>&& quat) : Base(), mRepData(std::move(quat))
{
  // Do nothing
}

//==============================================================================
template <typename S>
Quaternion<S>& Quaternion<S>::operator=(const Quaternion<S>& other)
{
  mRepData = other.mRepData;
  return *this;
}

//==============================================================================
template <typename S>
Quaternion<S>& Quaternion<S>::operator=(Quaternion<S>&& other)
{
  mRepData = std::move(other.mRepData);
  return *this;
}

//==============================================================================
template <typename S>
Quaternion<S>& Quaternion<S>::operator=(const Eigen::AngleAxis<S>& quat)
{
  mRepData = quat;
  return *this;
}

//==============================================================================
template <typename S>
Quaternion<S>& Quaternion<S>::operator=(Eigen::AngleAxis<S>&& quat)
{
  mRepData = std::move(quat);
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
Quaternion<S>& Quaternion<S>::operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
{
  mRepData = quat;
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
Quaternion<S>& Quaternion<S>::operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
{
  mRepData = std::move(quat);
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
Quaternion<S>& Quaternion<S>::operator=(const Eigen::MatrixBase<Derived>& matrix)
{
  mRepData = matrix;
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
Quaternion<S>& Quaternion<S>::operator=(Eigen::MatrixBase<Derived>&& matrix)
{
  mRepData = std::move(matrix);
  return *this;
}

//==============================================================================
template <typename S>
bool Quaternion<S>::operator ==(const Quaternion& other)
{
  return mRepData.isApprox(other.mRepData, static_cast<S>(0));
}

//==============================================================================
template <typename S>
void Quaternion<S>::setQuaternion(const RepData& quat)
{
  mRepData = quat;
}

//==============================================================================
template <typename S>
const typename Quaternion<S>::RepData& Quaternion<S>::getQuaternion() const
{
  return mRepData;
}

//==============================================================================
template <typename S>
void Quaternion<S>::setQuaternion(S w, S x, S y, S z)
{
  mRepData.w() = w;
  mRepData.x() = x;
  mRepData.y() = y;
  mRepData.z() = z;
}

//==============================================================================
//  template <typename S>
//  void fromVector(const Vector& vector)
//  {
//    mRepData.vec() = vector;
//  }

//==============================================================================
//  template <typename S>
//  Vector toVector() const
//  {
//    return mRepData.vec();
//  }

//==============================================================================
template <typename S>
void Quaternion<S>::setW(S w)
{
  mRepData.w() = w;
}

//==============================================================================
template <typename S>
void Quaternion<S>::setX(S x)
{
  mRepData.x() = x;
}

//==============================================================================
template <typename S>
void Quaternion<S>::setY(S y)
{
  mRepData.y() = y;
}

//==============================================================================
template <typename S>
void Quaternion<S>::setZ(S z)
{
  mRepData.z() = z;
}

//==============================================================================
template <typename S>
S Quaternion<S>::getW() const
{
  return mRepData.w();
}

//==============================================================================
template <typename S>
S Quaternion<S>::getX() const
{
  return mRepData.x();
}

//==============================================================================
template <typename S>
S Quaternion<S>::getY() const
{
  return mRepData.y();
}

//==============================================================================
template <typename S>
S Quaternion<S>::getZ() const
{
  return mRepData.z();
}

//==============================================================================
template <typename S>
void Quaternion<S>::setRandom()
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

//==============================================================================
template <typename S>
void Quaternion<S>::setIdentity()
{
  mRepData.setIdentity();
}

//==============================================================================
template <typename S>
bool Quaternion<S>::isIdentity()
{
  return mRepData.coeffs() == Eigen::Matrix<S, 4, 1>::Zero();
  // TODO(JS): double-check if this is correct
}

//==============================================================================
template <typename S>
void Quaternion<S>::invert()
{
  mRepData = mRepData.conjugate();
}

//==============================================================================
template <typename S>
const Quaternion<S> Quaternion<S>::getInverse() const
{
  return Quaternion<S>(mRepData.conjugate());
}

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_QUATERNION_IMPL_HPP_
