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
  detail::SO3_::group_assign_impl<S, Derived, OtherDerived>::run(
        derived(), other.derived());

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
Derived& SO3Base<Derived>::operator=(SO3Base<OtherDerived>&& other)
{
  detail::SO3_::group_assign_impl<S, Derived, OtherDerived>::run(
        derived(), std::move(other.derived()));

  return derived();
}

//==============================================================================
template <typename Derived>
template <typename RotationDerived>
Derived& SO3Base<Derived>::operator=(const Eigen::RotationBase<RotationDerived, Dim>& rot)
{
  derived() = rot;
  return derived();
}

//==============================================================================
template <typename Derived>
template <typename RotationDerived>
Derived& SO3Base<Derived>::operator=(Eigen::RotationBase<RotationDerived, Dim>&& rot)
{
  derived() = std::move(rot);
  return derived();
}

//==============================================================================
template <typename Derived>
template <typename MatrixDerived>
Derived& SO3Base<Derived>::operator=(const Eigen::MatrixBase<MatrixDerived>& matrix)
{
  derived() = matrix;
  return derived();
}

//==============================================================================
template <typename Derived>
template <typename MatrixDerived>
Derived& SO3Base<Derived>::operator=(Eigen::MatrixBase<MatrixDerived>&& matrix)
{
  derived() = std::move(matrix);
  return derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
auto
SO3Base<Derived>::operator*(const SO3Base<OtherDerived>& other) const
-> decltype(detail::SO3_::rep_multiplication_impl<
    S, Rep, typename OtherDerived::Rep>::run(
              std::declval<RepData>(),
              std::declval<typename OtherDerived::RepData>()))
{
  return detail::SO3_::rep_multiplication_impl<
      S, Rep, typename OtherDerived::Rep>::run(
        getRepData(), other.getRepData());
}

//==============================================================================
template <typename Derived>
const typename SO3Base<Derived>::RotationVector
SO3Base<Derived>::operator*(const RotationVector& vector)
{
  return derived().operator*(vector);
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
void SO3Base<Derived>::operator*=(const SO3Base<OtherDerived>& other)
{
  derived() = detail::SO3_::group_multiplication_impl<Derived, OtherDerived>::run(
            derived(), other.derived());

//    detail::SO3::group_inplace_multiplication_impl<Derived, OtherDerived>::run(
//          derived(), other.derived());
}

//==============================================================================
template <typename Derived>
bool SO3Base<Derived>::operator ==(const SO3Base& other)
{
  return derived() == other.derived();
}

//==============================================================================
template <typename Derived>
template <typename OtherDerived>
bool SO3Base<Derived>::operator ==(const SO3Base<OtherDerived>& other)
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
  return detail::SO3_::group_is_approx_impl<Derived, OtherDerived>::run(
        derived(), other.derived(), tol);
}

//==============================================================================
template <typename Derived>
bool SO3Base<Derived>::isApprox(const Eigen::AngleAxis<S>& aa, S tol) const
{
  return detail::SO3_::rep_is_approx_impl<S, Rep, AxisAngleRep>::run(
        getRepData(), aa, tol);
  // TODO(JS): improve; Eigen::AngleAxis and AxisAngleRep are in weak
  // connection..
}

//==============================================================================
template <typename Derived>
template <typename QuatDerived>
bool SO3Base<Derived>::isApprox(const Eigen::QuaternionBase<QuatDerived>& quat, S tol) const
{
  return detail::SO3_::rep_is_approx_impl<S, Rep, QuaternionRep>::run(
        getRepData(), quat, tol);
  // TODO(JS): improve; Eigen::QuaternionBase and QuaternionRep are in weak
  // connection..
}

//==============================================================================
template <typename Derived>
template <typename MatrixDerived>
bool SO3Base<Derived>::isApprox(const Eigen::MatrixBase<MatrixDerived>& matrix, S tol) const
{
  // We assume matrix is 3x3 rotation matrix
  return detail::SO3_::rep_is_approx_impl<S, Rep, RotationMatrixRep>::run(
        getRepData(), matrix, tol);
  // TODO(JS): improve; Eigen::QuaternionBase and QuaternionRep are in weak
  // connection..
}

//==============================================================================
template <typename Derived>
Derived SO3Base<Derived>::Exp(const so3& tangent)
{
  return Derived(
        detail::SO3_::rep_convert_impl<S, RotationVectorRep, Rep>::run(
          tangent));
}

//==============================================================================
template <typename Derived>
Derived SO3Base<Derived>::Exp(so3&& tangent)
{
  return Derived(
        detail::SO3_::rep_convert_impl<S, RotationVectorRep, Rep>::run(
          std::move(tangent)));
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setExp(const so3& tangent)
{
  derived() = Exp(tangent);
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::setExp(so3&& tangent)
{
  derived() = Exp(std::move(tangent));
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::Log(const Derived& point)
{
  return detail::SO3_::rep_convert_impl<S, Rep, RotationVectorRep>::run(
        point.getRepData());
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::Log(Derived&& point)
{
  return detail::SO3_::rep_convert_impl<S, Rep, RotationVectorRep>::run(
        std::move(point.getRepData()));
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::so3 SO3Base<Derived>::getLog() const
{
  return Log(derived());
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::RotationMatrix SO3Base<Derived>::Hat(const Tangent& angleAxis)
{
  RotationMatrix res;
  res <<  static_cast<S>(0),     -angleAxis(2),      angleAxis(1),
               angleAxis(2), static_cast<S>(0),     -angleAxis(0),
              -angleAxis(1),      angleAxis(0), static_cast<S>(0);

  return res;
}

//==============================================================================
template <typename Derived>
typename SO3Base<Derived>::Tangent SO3Base<Derived>::Vee(const RotationMatrix& mat)
{
  // TODO(JS): Add validity check if mat is skew-symmetric for debug mode
  return Tangent(mat(2, 1), mat(0, 2), mat(1, 0));
}

//==============================================================================
template <typename Derived>
template <typename RepTo>
auto SO3Base<Derived>::to() const
-> decltype(detail::SO3_::rep_convert_impl<S, Rep, RepTo>::run(
    std::declval<RepData>()))
{
  return detail::SO3_::rep_convert_impl<S, Rep, RepTo>::run(
        getRepData());
}

//==============================================================================
template <typename Derived>
auto SO3Base<Derived>::toRotationMatrix() const
-> decltype(detail::SO3_::rep_convert_impl<S, Rep, RotationMatrixRep>::run(
    std::declval<RepData>()))
{
  // The return type could be either of const and const reference depending on
  // the Derived type. So the trailing return type deduction is used here.

  return to<RotationMatrixRep>();
}

//==============================================================================
template <typename Derived>
void SO3Base<Derived>::fromRotationMatrix(const RotationMatrix& rotMat)
{
  // We assume the canonical representation is the rotation matrix
  setRepData(
        detail::SO3_::rep_convert_from_canonical_impl<S, Rep>::run(rotMat));
}

//==============================================================================
template <typename Derived>
template <typename RepTo>
auto SO3Base<Derived>::getCoordinates() const
-> decltype(detail::SO3_::rep_convert_impl<S, Rep, RepTo>::run(
    std::declval<RepData>()))
{
  // TODO(JS): Change return type to Eigen::Matrix<S, Dim, 1> or
  // check if the raw data of RepTo is a vector type.
  static_assert(detail::traits<SO3<S, RepTo>>::CanBeCoordinates,
                "Attempting to get invalid coordinate type.");

  return to<RepTo>();
}

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

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3BASE_IMPL_HPP_
