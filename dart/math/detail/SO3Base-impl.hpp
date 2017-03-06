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

#ifndef DART_MATH_DETAIL_SO3BASE_IMPL_HPP_
#define DART_MATH_DETAIL_SO3BASE_IMPL_HPP_

#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename Derived>
const Derived& SO3Base<Derived>::derived() const
{
  return *static_cast<const Derived*>(this);
}

//==============================================================================
template <typename Derived>
Derived& SO3Base<Derived>::derived()
{
  return *static_cast<Derived*>(this);
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
Derived& SO3Base<Derived>::operator=(const SO3Base<OtherDerived>& other)
{
  detail::SO3Assign<OtherDerived, Derived>::run(other.derived(), derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
Derived& SO3Base<Derived>::operator=(SO3Base<OtherDerived>&& other)
{
  detail::SO3Assign<OtherDerived, Derived>::run(
      std::move(other.derived()), derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename RotationDerived>
Derived& SO3Base<Derived>::operator=(const Eigen::RotationBase<RotationDerived, Dim>& rot)
{
  detail::SO3Assign<Eigen::RotationBase<RotationDerived, Dim>, Derived>::run(
        rot, derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename RotationDerived>
Derived& SO3Base<Derived>::operator=(Eigen::RotationBase<RotationDerived, Dim>&& rot)
{
  detail::SO3Assign<Eigen::RotationBase<RotationDerived, Dim>, Derived>::run(
        std::move(rot), derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename MatrixDerived>
Derived& SO3Base<Derived>::operator=(const Eigen::MatrixBase<MatrixDerived>& matrix)
{
  detail::SO3Assign<Eigen::MatrixBase<MatrixDerived>, Derived>::run(
        matrix, derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename MatrixDerived>
Derived& SO3Base<Derived>::operator=(Eigen::MatrixBase<MatrixDerived>&& matrix)
{
  detail::SO3Assign<Eigen::MatrixBase<MatrixDerived>, Derived>::run(
        std::move(matrix), derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
//auto
Derived
SO3Base<Derived>::operator*(const SO3Base<OtherDerived>& other) const
//-> decltype(detail::SO3MultiplicationImpl<
//    Derived, OtherDerived>::run(
//              std::declval<RepData>(),
//              std::declval<typename OtherDerived::RepData>()))
{
  return detail::SO3MultiplicationImpl<
      Derived, OtherDerived>::run(derived(), other.derived());
}

//==============================================================================
template <typename Derived>
const typename SO3Base<Derived>::Vector3
SO3Base<Derived>::operator*(const Vector3& vector)
{
  return derived().operator*(vector);
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
void SO3Base<Derived>::operator*=(const SO3Base<OtherDerived>& other)
{
  derived() = detail::SO3MultiplicationImpl<
      Derived, OtherDerived>::run(derived(), other.derived());

//    detail::SO3::SO3InplaceMultiplicationImpl<Derived, OtherDerived>::run(
//          derived(), other.derived());
}

//==============================================================================
template <typename Derived>
bool SO3Base<Derived>::operator==(const SO3Base& other)
{
  return derived() == other.derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
bool SO3Base<Derived>::operator==(const SO3Base<OtherDerived>& other)
{
  return toRotationMatrix() == other.toRotationMatrix();
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setRandom()
{
  derived().setRandom();
}

//==============================================================================
template <typename Derived>
Derived SO3Base<Derived>::Random()
{
  Derived R;
  R.setRandom();

  return R;
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setIdentity()
{
  derived().setIdentity();
}

//==============================================================================
template <typename Derived>
bool SO3Base<Derived>::isIdentity()
{
  return derived().isIdentity();
}

//==============================================================================
template <typename Derived>
Derived SO3Base<Derived>::Identity()
{
  Derived I;
  I.setIdentity();

  return I;
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::invert()
{
  return derived().invert();
}

//==============================================================================
template <typename Derived>
Derived SO3Base<Derived>::getInverse() const
{
  return derived().inverse();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
bool SO3Base<Derived>::isApprox(const SO3Base<OtherDerived>& other, S tol) const
{
  return detail::SO3IsApprox<Derived, OtherDerived>::run(
        derived(), other.derived(), tol);
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setExp(const so3& tangent)
{
  derived().setExp(tangent);
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setExp(so3&& tangent)
{
  derived().setExp(std::move(tangent));
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::Log(const Derived& point)
{
//  return detail::SO3RepDataConvertImpl<Derived, SO3Vector<S>>::run(
//        point.getRepData());
  return SO3Base<Derived>::so3();
  // TODO(JS):
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::Log(Derived&& point)
{
//  return detail::SO3RepDataConvertImpl<Derived, SO3Vector<S>>::run(
//        std::move(point.getRepData()));
  return SO3Base<Derived>::so3();
  // TODO(JS):
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::getLog() const
{
  return Log(derived());
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::Matrix3 SO3Base<Derived>::Hat(const Tangent& angleAxis)
{
  Matrix3 res;
  res <<  static_cast<S>(0),     -angleAxis(2),      angleAxis(1),
               angleAxis(2), static_cast<S>(0),     -angleAxis(0),
              -angleAxis(1),      angleAxis(0), static_cast<S>(0);

  return res;
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::Tangent SO3Base<Derived>::Vee(const Matrix3& mat)
{
  // TODO(JS): Add validity check if mat is skew-symmetric for debug mode
  return Tangent(mat(2, 1), mat(0, 2), mat(1, 0));
}

//==============================================================================
template <typename Derived>
template <typename RepTo>
auto SO3Base<Derived>::to() const
-> decltype(detail::SO3ToImpl<Derived, RepTo>::run(std::declval<Derived>()))
{
  return detail::SO3ToImpl<Derived, RepTo>::run(derived());
}

//==============================================================================
template <typename Derived>
auto SO3Base<Derived>::toRotationMatrix() const
-> typename detail::Traits<SO3Matrix<S>>::RepData
{
  // The return type could be either of const and const reference depending on
  // the Derived type. So the trailing return type deduction is used here.

  return to<SO3Matrix<S>>().getRepData();
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::fromRotationMatrix(const Matrix3& rotMat)
{
  detail::SO3Assign<Matrix3, Derived>::run(rotMat, derived());
}

//==============================================================================
//template <typename Derived>
//template <typename RepTo>
//auto SO3Base<Derived>::getCoordinates() const
//-> decltype(detail::SO3ConvertImpl<Derived, RepTo>::run(
//    std::declval<RepData>()))
//{
//  // TODO(JS): Change return type to Eigen::Matrix<S, Dim, 1> or
//  // check if the raw data of RepTo is a vector type.
//  static_assert(detail::traits<RepTo>::IsCoordinates,
//                "Attempting to get invalid coordinate type.");

//  return to<RepTo>();
//}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setRepData(const RepData& data)
{
  derived().mRepData = data;
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setRepData(RepData&& data)
{
  derived().mRepData = std::move(data);
}

//==============================================================================
template <typename Derived>
const typename SO3Base<Derived>::RepData& SO3Base<Derived>::getRepData() const
{
  return derived().mRepData;
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::RepData& SO3Base<Derived>::getRepData()
{
  return derived().mRepData;
}

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3BASE_IMPL_HPP_
