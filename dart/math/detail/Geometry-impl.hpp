/*
 * Copyright (c) 2011-2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2011-2016, Humanoid Lab, Georgia Tech Research Corporation
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

#ifndef DART_MATH_DETAIL_GEOMETRY_IMPL_HPP_
#define DART_MATH_DETAIL_GEOMETRY_IMPL_HPP_

#include "dart/math/Geometry.hpp"

namespace dart {
namespace math {

namespace detail {

//==============================================================================
// eulerAnglesToMatrixImpl
//==============================================================================

//==============================================================================
template <typename S, int index0, int index1, int index2>
struct eulerAnglesToMatrixImpl
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    Eigen::Matrix<S, 3, 1> axis0 = Eigen::Matrix<S, 3, 1>::Zero();
    Eigen::Matrix<S, 3, 1> axis1 = Eigen::Matrix<S, 3, 1>::Zero();
    Eigen::Matrix<S, 3, 1> axis2 = Eigen::Matrix<S, 3, 1>::Zero();

    axis0[index0] = S(1.0);
    axis1[index1] = S(1.0);
    axis2[index2] = S(1.0);

    return Eigen::Matrix<S, 3, 3>(
        Eigen::AngleAxis<S>(angles[0], axis0)
        * Eigen::AngleAxis<S>(angles[1], axis1)
        * Eigen::AngleAxis<S>(angles[2], axis2));
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 0, 1, 0>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerXYXToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 0, 1, 2>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerXYZToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 0, 2, 0>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerXZXToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 0, 2, 1>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerXZYToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 1, 0, 1>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerYXYToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 1, 0, 2>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerYXZToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 1, 2, 0>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerYZXToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 1, 2, 1>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerYZYToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 2, 0, 1>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerZXYToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 2, 1, 0>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerZYXToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 2, 0, 2>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerZXZToMatrix(angles);
  }
};

//==============================================================================
template <typename S>
struct eulerAnglesToMatrixImpl<S, 2, 1, 2>
{
  static Eigen::Matrix<S, 3, 3> run(const Eigen::Matrix<S, 3, 1>& angles)
  {
    return eulerZYZToMatrix(angles);
  }
};

//==============================================================================
// matrixToEulerAnglesImpl
//==============================================================================

//==============================================================================
template <typename S, int index0, int index1, int index2>
struct matrixToEulerAnglesImpl
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrix.eulerAngles(index0, index1, index2);
  }
};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 0, 1, 0>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerXYX(matrix);
  }
};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 0, 1, 2>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerXYZ(matrix);
  }
};

//==============================================================================
//template <typename S>
//struct matrixToEulerAnglesImpl<S, 0, 2, 0>
//{
//  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
//  {
//    return matrixToEulerXZX(matrix);
//  }
//};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 0, 2, 1>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerXZY(matrix);
  }
};

//==============================================================================
//template <typename S>
//struct matrixToEulerAnglesImpl<S, 1, 0, 1>
//{
//  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
//  {
//    return matrixToEulerYXY(matrix);
//  }
//};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 1, 0, 2>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerYXZ(matrix);
  }
};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 1, 2, 0>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerYZX(matrix);
  }
};

//==============================================================================
//template <typename S>
//struct matrixToEulerAnglesImpl<S, 1, 2, 1>
//{
//  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
//  {
//    return matrixToEulerYZY(matrix);
//  }
//};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 2, 0, 1>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerZXY(matrix);
  }
};

//==============================================================================
template <typename S>
struct matrixToEulerAnglesImpl<S, 2, 1, 0>
{
  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
  {
    return matrixToEulerZYX(matrix);
  }
};

//==============================================================================
//template <typename S>
//struct matrixToEulerAnglesImpl<S, 2, 0, 2>
//{
//  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
//  {
//    return matrixToEulerZXZ(matrix);
//  }
//};

//==============================================================================
//template <typename S>
//struct matrixToEulerAnglesImpl<S, 2, 1, 2>
//{
//  static Eigen::Matrix<S, 3, 1> run(const Eigen::Matrix<S, 3, 3>& matrix)
//  {
//    return matrixToEulerZYZ(matrix);
//  }
//};

} // namespace detail

//==============================================================================
template <typename S, int index0, int index1, int index2>
Eigen::Matrix<S, 3, 3> eulerAnglesToMatrix(const Eigen::Matrix<S, 3, 1>& angles)
{
  return detail::eulerAnglesToMatrixImpl<S, index0, index1, index2>::run(angles);
}

//==============================================================================
template <typename S, int index0, int index1, int index2>
Eigen::Matrix<S, 3, 1> matrixToEulerAngles(const Eigen::Matrix<S, 3, 3>& matrix)
{
  return detail::matrixToEulerAnglesImpl<S, index0, index1, index2>::run(matrix);
}

} // namespace math
} // namespace dart

#endif  // DART_MATH_DETAIL_GEOMETRY_IMPL_HPP_
