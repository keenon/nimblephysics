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

#ifndef DART_MATH_DETAIL_EULERANGLES_IMPL_HPP_
#define DART_MATH_DETAIL_EULERANGLES_IMPL_HPP_

#include "dart/math/EulerAngles.hpp"

#include "dart/math/SO3Matrix.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>::EulerAngles() : Base()
{
  // Do nothing
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>::EulerAngles(const EulerAngles& other) : Base(), mRepData(other.mRepData)
{
  // Do nothing
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>::EulerAngles(EulerAngles&& other) : Base(), mRepData(std::move(other.mRepData))
{
  // Do nothing
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>::EulerAngles(const SO3Base<Derived>& other)
  : Base(), mRepData()
{
  *this = other;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>::EulerAngles(SO3Base<Derived>&& other)
  : Base(), mRepData()
{
  *this = std::move(other);
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>::EulerAngles(const Eigen::MatrixBase<Derived>& angles)
  : Base(), mRepData(angles)
{
  // Do nothing
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>::EulerAngles(Eigen::MatrixBase<Derived>&& angles)
  : Base(), mRepData(std::move(angles))
{
  // Do nothing
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(const EulerAngles& other)
{
  mRepData = other.mRepData;
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(EulerAngles&& other)
{
  mRepData = std::move(other.mRepData);
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(const Eigen::AngleAxis<S>& aa)
{
  detail::SO3Assign<This, Eigen::AngleAxis<S>>::run(*this, aa);
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(Eigen::AngleAxis<S>&& aa)
{
  detail::SO3Assign<This, Eigen::AngleAxis<S>>::run(*this, std::move(aa));
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename QuatDerived>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(
    const Eigen::QuaternionBase<QuatDerived>& quat)
{
  detail::SO3Assign<This, Quaternion<S>>::run(*this, quat);
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename QuatDerived>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(
    Eigen::QuaternionBase<QuatDerived>&& quat)
{
  detail::SO3Assign<This, Quaternion<S>>::run(*this, std::move(quat));
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(
    const Eigen::MatrixBase<Derived>& matrix)
{
  detail::SO3Assign<This, Eigen::MatrixBase<Derived>>::run(*this, matrix);
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <typename Derived>
EulerAngles<S, index0, index1, index2>&
EulerAngles<S, index0, index1, index2>::operator=(
    Eigen::MatrixBase<Derived>&& matrix)
{
  detail::SO3Assign<This, Eigen::MatrixBase<Derived>>::run(*this, std::move(matrix));
  return *this;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
bool EulerAngles<S, index0, index1, index2>::operator==(const EulerAngles& other)
{
  return (mRepData == other.mRepData);
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::fromCanonical(const SO3Matrix<S>& mat)
{
  mRepData = math::matrixToEulerAngles<S, index0, index1, index2>(
      mat.getRepData());
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
SO3Matrix<S> EulerAngles<S, index0, index1, index2>::toCanonical() const
{
  return SO3Matrix<S>(
      math::eulerAnglesToMatrix<S, index0, index1, index2>(mRepData));
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::setAngles(const Eigen::Matrix<S, 3, 1>& angles)
{
  mRepData = angles;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
Eigen::Matrix<S, 3, 1> EulerAngles<S, index0, index1, index2>::getAngles() const
{
  return mRepData;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::setAngles(S angle0, S angle1, S angle2)
{
  mRepData << angle0, angle1, angle2;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <int index>
void EulerAngles<S, index0, index1, index2>::setAngle(S angle)
{
  static_assert(0 <= index && index <= 2, "Invalid index");
  mRepData[index] = angle;
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
template <int index>
S EulerAngles<S, index0, index1, index2>::getAngle() const
{
  static_assert(0 <= index && index <= 2, "Invalid index");
  return mRepData[index];
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::setRandom()
{
  mRepData.setRandom();
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::setIdentity()
{
  mRepData.setZero();
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
bool EulerAngles<S, index0, index1, index2>::isIdentity()
{
  return mRepData == RepData::Zero();
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
void EulerAngles<S, index0, index1, index2>::invert()
{
  mRepData.reverseInPlace();
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
const EulerAngles<S, index0, index1, index2> EulerAngles<S, index0, index1, index2>::getInverse() const
{
  return EulerAngles(RepData(-mRepData.reverse()));
}

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_EULERANGLES_IMPL_HPP_
