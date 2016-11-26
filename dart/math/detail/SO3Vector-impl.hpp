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

#ifndef DART_MATH_DETAIL_SO3VECTOR_IMPL_HPP_
#define DART_MATH_DETAIL_SO3VECTOR_IMPL_HPP_

#include "dart/math/SO3Vector.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename S>
SO3Vector<S>::SO3Vector() : Base()
{
  // Do nothing
}

//==============================================================================
template <typename S>
SO3Vector<S>::SO3Vector(const SO3Vector& other) : Base(), mRepData(other.mRepData)
{
  // Do nothing
}

//==============================================================================
template <typename S>
SO3Vector<S>::SO3Vector(SO3Vector&& other) : mRepData(std::move(other.mRepData))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>::SO3Vector(const SO3Base<Derived>& other)
  : Base(),
    mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
            other.getRepData()))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>::SO3Vector(SO3Base<Derived>&& other)
  : Base(),
    mRepData(detail::so3_operations::SO3RepDataConvertImpl<Derived, This>::run(
            std::move(other.getRepData())))
{
  // Do nothing
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>::SO3Vector(const Eigen::MatrixBase<Derived>& matrix) : Base(), mRepData(matrix)
{
  assert(matrix.rows() == 3);
  assert(matrix.cols() == 1);
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>::SO3Vector(Eigen::MatrixBase<Derived>&& matrix) : Base(), mRepData(std::move(matrix))
{
  assert(matrix.rows() == 3);
  assert(matrix.cols() == 1);
}

//==============================================================================
template <typename S>
SO3Vector<S>& SO3Vector<S>::operator=(const SO3Vector& other)
{
  mRepData = other.mRepData;
  return *this;
}

//==============================================================================
template <typename S>
SO3Vector<S>& SO3Vector<S>::operator=(SO3Vector&& other)
{
  mRepData = std::move(other.mRepData);
  return *this;
}

//==============================================================================
template <typename S>
SO3Vector<S>& SO3Vector<S>::operator=(const Eigen::AngleAxis<S>& quat)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<
      AngleAxis<S>, SO3Vector<S>>::run(quat);
  // TODO(JS): improve; need a way to deduce representation type from Eigen
  // data type
  return *this;
}

//==============================================================================
template <typename S>
SO3Vector<S>& SO3Vector<S>::operator=(Eigen::AngleAxis<S>&& quat)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<
      AngleAxis<S>, SO3Vector<S>>::run(std::move(quat));
  // TODO(JS): improve; need a way to deduce representation type from Eigen
  // data type
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
SO3Vector<S>& SO3Vector<S>::operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<Quaternion<S>, SO3Vector<S>>::run(quat);
  // TODO(JS): improve; need a way to deduce representation type from Eigen
  // data type
  return *this;
}

//==============================================================================
template <typename S>
template <typename QuatDerived>
SO3Vector<S>& SO3Vector<S>::operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<
      Quaternion<S>, SO3Vector<S>>::run(
        std::move(quat));
  // TODO(JS): improve; need a way to deduce representation type from Eigen
  // data type
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>& SO3Vector<S>::operator=(const Eigen::MatrixBase<Derived>& matrix)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<
      SO3Matrix<S>, SO3Vector<S>>::run(
        matrix);
  return *this;
}

//==============================================================================
template <typename S>
template <typename Derived>
SO3Vector<S>& SO3Vector<S>::operator=(Eigen::MatrixBase<Derived>&& matrix)
{
  mRepData = detail::so3_operations::SO3RepDataConvertImpl<
      SO3Matrix<S>, SO3Vector<S>>::run(
        std::move(matrix));
  return *this;
}

//==============================================================================
template <typename S>
bool SO3Vector<S>::operator ==(const SO3Vector& other)
{
  return mRepData == other.mRepData;
}

//==============================================================================
template <typename S>
template <typename MatrixDerived>
void SO3Vector<S>::fromRotationVector(const Eigen::MatrixBase<MatrixDerived>& vector)
{
  assert(vector.rows() == 3);
  assert(vector.cols() == 1);

  mRepData = vector;
}

//==============================================================================
template <typename S>
template <typename MatrixDerived>
void SO3Vector<S>::fromRotationVector(Eigen::MatrixBase<MatrixDerived>&& vector)
{
  assert(vector.rows() == 3);
  assert(vector.cols() == 1);

  mRepData = std::move(vector);
}

//==============================================================================
template <typename S>
typename SO3Vector<S>::RotationVector SO3Vector<S>::toRotationVector() const
{
  return mRepData;
}

//==============================================================================
template <typename S>
void SO3Vector<S>::setRotationVector(const RotationVector& axisAngle)
{
  mRepData = axisAngle;
}

//==============================================================================
template <typename S>
const typename SO3Vector<S>::RotationVector& SO3Vector<S>::getRotationVector() const
{
  return mRepData;
}

//==============================================================================
template <typename S>
void SO3Vector<S>::setRandom()
{
  mRepData.setRandom();
}

//==============================================================================
template <typename S>
void SO3Vector<S>::setIdentity()
{
  mRepData.setZero();
}

//==============================================================================
template <typename S>
bool SO3Vector<S>::isIdentity()
{
  return mRepData == RepData::Zero();
}

//==============================================================================
template <typename S>
void SO3Vector<S>::invert()
{
  mRepData *= static_cast<S>(-1);
}

//==============================================================================
template <typename S>
const SO3Vector<S> SO3Vector<S>::getInverse() const
{
  return SO3Vector(-mRepData);
}

} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3VECTOR_IMPL_HPP_
