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

#ifndef DART_MATH_DETAIL_ANGLEAXIS_IMPLE_HPP_
#define DART_MATH_DETAIL_ANGLEAXIS_IMPLE_HPP_

#include "dart/math/AngleAxis.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis() : Base()
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(const AngleAxis& other) : Base(), mRepData(other.mRepData)
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(AngleAxis&& other) : mRepData(std::move(other.mRepData))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
AngleAxis<S>::AngleAxis(const SO3Base<Derived>& other)
  : Base(),
    mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
             other.getRepData()))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
AngleAxis<S>::AngleAxis(SO3Base<Derived>&& other)
  : Base(),
    mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
             std::move(other.getRepData())))
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(const Eigen::AngleAxis<S>& angleAxis)
  : Base(), mRepData(angleAxis)
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(Eigen::AngleAxis<S>&& angleAxis)
  : Base(), mRepData(std::move(angleAxis))
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(const Axis& axis, S angle)
  : Base(), mRepData(angle, axis)
{
  // Do nothing
}

//==============================================================================
template <typename S>
AngleAxis<S>::AngleAxis(Axis&& axis, S angle)
  : Base(), mRepData(std::move(angle), axis)
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
AngleAxis<S>::AngleAxis(const Eigen::QuaternionBase<QuatDerived>& q)
{
  mRepData = q;
}

//==============================================================================
template <typename S>
AngleAxis<S>& AngleAxis<S>::operator=(const AngleAxis& other)
{
  mRepData = other.mRepData;
  return *this;
}

//==============================================================================
template <typename S>
AngleAxis<S>& AngleAxis<S>::operator=(AngleAxis&& other)
{
  mRepData = std::move(other.mRepData);
  return *this;
}

//==============================================================================
template <typename S>
AngleAxis<S>& AngleAxis<S>::operator=(const Eigen::AngleAxis<S>& quat)
{
  mRepData = quat;
  return *this;
}

//==============================================================================
template <typename S>
AngleAxis<S>& AngleAxis<S>::operator=(Eigen::AngleAxis<S>&& quat)
{
  mRepData = std::move(quat);
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
AngleAxis<S>& AngleAxis<S>::operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
{
  mRepData = quat;
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
AngleAxis<S>& AngleAxis<S>::operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
{
  mRepData = std::move(quat);
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
AngleAxis<S>& AngleAxis<S>::operator=(const Eigen::MatrixBase<Derived>& matrix)
{
  mRepData = matrix;
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
AngleAxis<S>& AngleAxis<S>::operator=(Eigen::MatrixBase<Derived>&& matrix)
{
  mRepData = std::move(matrix);
  return *this;
}

//==============================================================================
template <typename S>
bool AngleAxis<S>::operator==(const AngleAxis& other)
{
  if (mRepData.angle() == static_cast<S>(0)
      && other.getRepData().angle() == static_cast<S>(0))
    return true;

  return mRepData.isApprox(other.mRepData, static_cast<S>(0));
}

//==============================================================================
template <typename S>
void AngleAxis<S>::setAngleAxis(const Axis& axis, S angle)
{
  mRepData.axis() = axis;
  mRepData.angle() = angle;
}

//==============================================================================
template <typename S>
void AngleAxis<S>::setAxis(const Axis& axis)
{
  mRepData.axis() = axis;
}

//==============================================================================
template <typename S>
const typename AngleAxis<S>::Axis& AngleAxis<S>::getAxis() const
{
  return mRepData.axis();
}

//==============================================================================
template <typename S>
void AngleAxis<S>::setAngle(const S angle)
{
  mRepData.angle() = angle;
}

//==============================================================================
template <typename S>
typename AngleAxis<S>::S AngleAxis<S>::getAngle() const
{
  return mRepData.angle();
}

//==============================================================================
template <typename S>
void AngleAxis<S>::setRandom()
{
  mRepData.axis().setRandom().normalize();
  mRepData.angle() = math::random(-1.0, 1.0);
  // TODO(JS): improve
}

//==============================================================================
//  template <typename S>
//  template <typename OtherDerived>
//  bool AngleAxis<S>::isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const
//  {
//    return detail::SO3::group_is_approx_impl<Derived, OtherDerived>::run(
//          derived(), other.derived(), tol);
//  }

//==============================================================================
template <typename S>
void AngleAxis<S>::setIdentity()
{
  mRepData.angle() = static_cast<S>(0);
}

//==============================================================================
template <typename S>
bool AngleAxis<S>::isIdentity()
{
  return mRepData.angle() == static_cast<S>(0);
}

//==============================================================================
template <typename S>
void AngleAxis<S>::invert()
{
  mRepData.angle() *= static_cast<S>(-1);
}

//==============================================================================
template <typename S>
const AngleAxis<S> AngleAxis<S>::getInverse() const
{
  return AngleAxis(RepData(-mRepData.angle(), mRepData.axis()));
}

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_ANGLEAXIS_IMPLE_HPP_
