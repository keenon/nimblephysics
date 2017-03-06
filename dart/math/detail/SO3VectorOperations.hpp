/*
 * Copyright (c) 2017, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2017, Personal Robotics Lab, Carnegie Mellon University
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

#ifndef DART_MATH_DETAIL_SO3VECTOROPERATIONS_HPP_
#define DART_MATH_DETAIL_SO3VECTOROPERATIONS_HPP_

#include "dart/math/SO3Vector.hpp"

#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {
namespace detail {

//==============================================================================
// SO3AssignEigenToSO3
//==============================================================================

//==============================================================================
template <typename EigenFrom>
struct SO3AssignEigenToSO3<
    EigenFrom,
    SO3Vector<typename EigenFrom::Scalar>,
    typename std::enable_if<SO3IsEigenMatrixBase<EigenFrom>::value>::type
    >
{
  static void run(const EigenFrom& from, SO3Vector<typename EigenFrom::Scalar>& to)
  {
    SO3Log(to.getRepData(), from);
  }

  static void run(EigenFrom&& from, SO3Vector<typename EigenFrom::Scalar>& to)
  {
    SO3Log(to.getRepData(), std::move(from));
  }
};

//==============================================================================
template <typename EigenFrom>
struct SO3AssignEigenToSO3<
    EigenFrom,
    SO3Vector<typename EigenFrom::Scalar>,
    typename std::enable_if<SO3IsEigenRotationBase<EigenFrom>::value>::type
    >
{
  static void run(const EigenFrom& from, SO3Vector<typename EigenFrom::Scalar>& to)
  {
    SO3Log(to.getRepData(), from.toRotationMatrix());
  }
};

//==============================================================================
// SO3AssignSO3ToEigen
//==============================================================================

//==============================================================================
template <typename EigenTo>
struct SO3AssignSO3ToEigen<
    SO3Vector<typename EigenTo::Scalar>,
    EigenTo,
    typename std::enable_if<SO3IsEigenMatrixBase<EigenTo>::value>::type
    >
{
  static void run(const SO3Vector<typename EigenTo::Scalar>& from, EigenTo& to)
  {
    SO3Exp(to, from.getRepData());
  }

  static void run(SO3Vector<typename EigenTo::Scalar>&& from, EigenTo& to)
  {
    SO3Exp(to, std::move(from.getRepData()));
  }
};

//==============================================================================
template <typename EigenTo>
struct SO3AssignSO3ToEigen<
    SO3Vector<typename EigenTo::Scalar>,
    EigenTo,
    typename std::enable_if<SO3IsEigenRotationBase<EigenTo>::value>::type
    >
{
  static void run(const SO3Vector<typename EigenTo::Scalar>& from, EigenTo& to)
  {
    Eigen::Matrix<typename EigenTo::Scalar, 3, 3> rotMat;
    SO3Exp(rotMat, from.getRepData());
    to = rotMat;
  }

  static void run(SO3Vector<typename EigenTo::Scalar>&& from, EigenTo& to)
  {
    Eigen::Matrix<typename EigenTo::Scalar, 3, 3> rotMat;
    SO3Exp(rotMat, std::move(from.getRepData()));
    to = rotMat;
  }
};

//==============================================================================
// SO3AssignSO3ToSO3
//==============================================================================

//==============================================================================
template <typename S>
struct SO3AssignSO3ToSO3<SO3Matrix<S>, SO3Vector<S>>
{
  static void run(const SO3Matrix<S>& from, SO3Vector<S>& to)
  {
    SO3Log(to.getRepData(), from.getRepData());
  }

  static void run(SO3Matrix<S>&& from, SO3Vector<S>& to)
  {
    SO3Log(to.getRepData(), std::move(from.getRepData()));
  }
};

//==============================================================================
template <typename S>
struct SO3AssignSO3ToSO3<SO3Vector<S>, SO3Matrix<S>>
{
  static void run(const SO3Vector<S>& from, SO3Matrix<S>& to)
  {
    SO3Exp(to.getRepData(), from.getRepData());
  }

  static void run(SO3Vector<S>&& from, SO3Matrix<S>& to)
  {
    SO3Exp(to.getRepData(), std::move(from.getRepData()));
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3VECTOROPERATIONS_HPP_
