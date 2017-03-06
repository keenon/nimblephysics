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

#ifndef DART_MATH_DETAIL_SO3OPERATIONS_HPP_
#define DART_MATH_DETAIL_SO3OPERATIONS_HPP_

#include <Eigen/Eigen>
#include "dart/common/Sfinae.hpp"
#include "dart/common/StlHelpers.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Constants.hpp"

//------------------------------------------------------------------------------
// Contents of this file:
//
// - Traits of SO(3) classes
//
// - SO3Exp
// - SO3Log
//
// - SO3RepDataIsEigenRotationBase3Impl
// - SO3RepDataIsEigenMatrixBase
// - SO3RepDataDirectConvertImpl
// - SO3RepDataConvertImpl
// - SO3RepDataMultiplicationImpl
// - SO3RepDataInplaceMultiplicationImpl
//
// - SO3IsCanonicalImpl
// - SO3MultiplicationImpl
// - SO3InplaceMultiplicationImpl
// - SO3ToImpl
//
// Naming convention is SO3[RepData](OperationName). 'RepData' is used if the
// fuction takes the representation data types. Otherwise, the function takes
// SO3 class types.
//------------------------------------------------------------------------------

namespace dart {
namespace math {

template <typename>
class SO3Base;

template <typename>
class SO3Matrix;

template <typename>
class SO3Vector;

template <typename>
class AngleAxis;

template <typename>
class Quaternion;

template <typename, int, int, int>
class EulerAngles;

template <typename S>
using DefaultSO3Canonical = SO3Matrix<S>;

// Forward declarations
template <typename, typename> class SO3;

namespace detail {

//==============================================================================
// Traits for all SO3 classes that is agnostic to the representation types
//==============================================================================

//==============================================================================
template <typename S_>
struct Traits<SO3Matrix<S_>>
{
  using S = S_;
  using RepData = Eigen::Matrix<S, 3, 3>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_>
struct Traits<SO3Vector<S_>>
{
  using S = S_;
  using RepData = Eigen::Matrix<S, 3, 1>;
  static constexpr bool IsCoordinates = true;
};

//==============================================================================
template <typename S_>
struct Traits<AngleAxis<S_>>
{
  using S = S_;
  using RepData = Eigen::AngleAxis<S>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_>
struct Traits<Quaternion<S_>>
{
  using S = S_;
  using RepData = Eigen::Quaternion<S>;
  static constexpr bool IsCoordinates = false;
};

//==============================================================================
template <typename S_, int index0, int index1, int index2>
struct Traits<EulerAngles<S_, index0, index1, index2>>
{
  using S = S_;
  using RepData = Eigen::Matrix<S, 3, 1>;
  static constexpr bool IsCoordinates = true;
};

//==============================================================================
// Exponential map for SO(3)
//==============================================================================

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 3> SO3Exp(const Eigen::Matrix<S, 3, 1>& w)
{
  using Matrix3 = Eigen::Matrix<S, 3, 3>;

  Matrix3 res;

  S s2[] = { w[0]*w[0], w[1]*w[1], w[2]*w[2] };
  S s3[] = { w[0]*w[1], w[1]*w[2], w[2]*w[0] };
  S theta = std::sqrt(s2[0] + s2[1] + s2[2]);
  S cos_t = std::cos(theta), alpha, beta;

  if (theta > constants<S>::eps())
  {
    S sin_t = std::sin(theta);
    alpha = sin_t / theta;
    beta = (1.0 - cos_t) / theta / theta;
  }
  else
  {
    alpha = 1.0 - theta*theta/6.0;
    beta = 0.5 - theta*theta/24.0;
  }

  res(0, 0) = beta*s2[0] + cos_t;
  res(1, 0) = beta*s3[0] + alpha*w[2];
  res(2, 0) = beta*s3[2] - alpha*w[1];

  res(0, 1) = beta*s3[0] - alpha*w[2];
  res(1, 1) = beta*s2[1] + cos_t;
  res(2, 1) = beta*s3[1] + alpha*w[0];

  res(0, 2) = beta*s3[2] + alpha*w[1];
  res(1, 2) = beta*s3[1] - alpha*w[0];
  res(2, 2) = beta*s2[2] + cos_t;

  return res;
}

//==============================================================================
template <typename S>
void SO3Exp(Eigen::Matrix<S, 3, 3>& res, const Eigen::Matrix<S, 3, 1>& w)
{
  S s2[] = { w[0]*w[0], w[1]*w[1], w[2]*w[2] };
  S s3[] = { w[0]*w[1], w[1]*w[2], w[2]*w[0] };
  S theta = std::sqrt(s2[0] + s2[1] + s2[2]);
  S cos_t = std::cos(theta), alpha, beta;

  if (theta > constants<S>::eps())
  {
    S sin_t = std::sin(theta);
    alpha = sin_t / theta;
    beta = (1.0 - cos_t) / theta / theta;
  }
  else
  {
    alpha = 1.0 - theta*theta/6.0;
    beta = 0.5 - theta*theta/24.0;
  }

  res(0, 0) = beta*s2[0] + cos_t;
  res(1, 0) = beta*s3[0] + alpha*w[2];
  res(2, 0) = beta*s3[2] - alpha*w[1];

  res(0, 1) = beta*s3[0] - alpha*w[2];
  res(1, 1) = beta*s2[1] + cos_t;
  res(2, 1) = beta*s3[1] + alpha*w[0];

  res(0, 2) = beta*s3[2] + alpha*w[1];
  res(1, 2) = beta*s3[1] - alpha*w[0];
  res(2, 2) = beta*s2[2] + cos_t;
}

//==============================================================================
// Logarithm map for SO(3)
//==============================================================================

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 1> SO3Log(const Eigen::Matrix<S, 3, 3>& R)
{
  Eigen::AngleAxis<S> aa(R);

  return aa.angle()*aa.axis();
}

//==============================================================================
template <typename S>
Eigen::Matrix<S, 3, 1> SO3Log(Eigen::Matrix<S, 3, 3>&& R)
{
  Eigen::AngleAxis<S> aa(std::move(R));

  return aa.angle()*aa.axis();
}

//==============================================================================
template <typename S>
void SO3Log(Eigen::Matrix<S, 3, 1>& vec, const Eigen::Matrix<S, 3, 3>& R)
{
  Eigen::AngleAxis<S> aa(R);
  vec = aa.angle()*aa.axis();
}

//==============================================================================
template <typename S>
void SO3Log(Eigen::Matrix<S, 3, 1>& vec, Eigen::Matrix<S, 3, 3>&& R)
{
  Eigen::AngleAxis<S> aa(std::move(R));
  vec = aa.angle()*aa.axis();
}

//==============================================================================
// SO3RepDataIsEigenRotationBase3Impl
//==============================================================================

//==============================================================================
template <typename SO3T, typename Enable = void>
struct SO3RepDataIsEigenRotationBase3Impl : std::false_type {};

//==============================================================================
template <typename SO3T>
struct SO3RepDataIsEigenRotationBase3Impl<
    SO3T,
    typename std::enable_if<
        std::is_base_of<
            Eigen::RotationBase<typename Traits<SO3T>::RepData, 3>,
            typename Traits<SO3T>::RepData
        >::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3IsEigenRotationBase3
//==============================================================================

//==============================================================================
template <typename T, typename Enable = void>
struct SO3IsEigenRotationBase : std::false_type {};

//==============================================================================
template <typename T>
struct SO3IsEigenRotationBase<
    T,
    typename std::enable_if<
        std::is_base_of<
            Eigen::RotationBase<T, 3>,
            T
        >::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3RepDataIsEigenMatrixBase is specialized to std::true_type if the given
// template parameter (SO3 class) is Eigen::MatrixBase, otherwise
// std::false_type.
//==============================================================================

//==============================================================================
template <typename SO3T, typename Enable = void>
struct SO3RepDataIsEigenMatrixBase : std::false_type {};

//==============================================================================
template <typename SO3T>
struct SO3RepDataIsEigenMatrixBase<
    SO3T,
    typename std::enable_if<
        std::is_base_of<
            Eigen::MatrixBase<typename Traits<SO3T>::RepData>,
            typename Traits<SO3T>::RepData
        >::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3RepDataIsEigenMatrixBase is specialized to std::true_type if the given
// template parameter (SO3 class) is Eigen::MatrixBase, otherwise
// std::false_type.
//==============================================================================

//==============================================================================
template <typename T, typename Enable = void>
struct SO3IsEigenMatrixBase : std::false_type {};

//==============================================================================
template <typename T>
struct SO3IsEigenMatrixBase<
    T,
    typename std::enable_if<
        std::is_base_of<Eigen::MatrixBase<T>, T>::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3IsEigen
//==============================================================================

//==============================================================================
template <typename T, typename Enable = void>
struct SO3IsEigen : std::false_type {};

//==============================================================================
template <typename T>
struct SO3IsEigen<
    T,
    typename std::enable_if<
        SO3IsEigenMatrixBase<T>::value || SO3IsEigenRotationBase<T>::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3IsEigen2
//==============================================================================

//==============================================================================
template <typename EigenA, typename EigenB, typename Enable = void>
struct SO3IsEigen2 : std::false_type {};

//==============================================================================
template <typename A, typename B>
struct SO3IsEigen2<
    A,
    B,
    typename std::enable_if<SO3IsEigen<A>::value && SO3IsEigen<B>::value>::type>
    : std::true_type {};

//==============================================================================
// SO3IsSO3
//==============================================================================

//==============================================================================
template <typename T, typename Enable = void>
struct SO3IsSO3 : std::false_type {};

//==============================================================================
template <typename T>
struct SO3IsSO3<
    T,
    typename std::enable_if<
        dart::common::is_base_of_template<SO3Base, T>::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3IsSO3_2
//==============================================================================

//==============================================================================
template <typename A, typename B, typename Enable = void>
struct SO3IsSO3_2 : std::false_type {};

//==============================================================================
template <typename A, typename B>
struct SO3IsSO3_2<
    A,
    B,
    typename std::enable_if<SO3IsSO3<A>::value && SO3IsSO3<B>::value>::type>
    : std::true_type {};

//==============================================================================
// SO3RepDataIsSupportedByEigenImpl
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B, typename Enable = void>
struct SO3RepDataIsSupportedByEigenImpl : std::false_type {};

//==============================================================================
template <typename SO3A, typename SO3B>
struct SO3RepDataIsSupportedByEigenImpl<
    SO3A,
    SO3B,
    typename std::enable_if<
        SO3IsEigen<typename SO3A::RepData>::value
        &&
        SO3IsEigen<typename SO3B::RepData>::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3RepDataConvertEigenToSO3Impl
//==============================================================================

//==============================================================================
template <typename SO3From, typename SO3To, typename Enable = void>
struct SO3RepDataConvertEigenToSO3Impl
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static constexpr bool IsSpecialized = false;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
template <typename SO3From, typename SO3To, typename Enable = void>
struct SO3RepDataConvertSO3ToEigenImpl
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static constexpr bool IsSpecialized = false;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
// SO3RepDataDirectConvertImpl
//==============================================================================

//==============================================================================
template <typename SO3From, typename SO3To, typename Enable = void>
struct SO3RepDataDirectConvertImpl
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static constexpr bool IsSpecialized = false;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
template <typename SO3From, typename SO3To>
struct SO3RepDataDirectConvertImpl<
    SO3From,
    SO3To,
    typename std::enable_if<common::HasAssignmentOperator<SO3From, SO3To>::value>::type>
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static constexpr bool IsSpecialized = false;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
// For the same representations, simply return the data without conversion
template <typename SO3T>
struct SO3RepDataDirectConvertImpl<SO3T, SO3T>
{
  using RepData = typename Traits<SO3T>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepData& run(const RepData& data)
  {
    return data;
  }

  static const RepData& run(RepData&& data)
  {
    return std::move(data);
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<SO3Matrix<S>, SO3Vector<S>>
{
  using RepDataFrom = typename Traits<SO3Matrix<S>>::RepData;
  using RepDataTo = typename Traits<SO3Vector<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return SO3Log(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return SO3Log(std::move(data));
  }
};

//==============================================================================
#define DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(id0, id1, id2)\
  template <typename S>\
  struct SO3RepDataDirectConvertImpl<SO3Matrix<S>, EulerAngles<S, id0, id1, id2>>\
  {\
    using RepDataFrom = typename Traits<SO3Matrix<S>>::RepData;\
    using RepDataTo = typename Traits<EulerAngles<S, id0, id1, id2>>::RepData;\
  \
    static constexpr bool IsSpecialized = true;\
  \
    static const RepDataTo run(const RepDataFrom& data)\
    {\
      return math::matrixToEulerAngles<S, id0, id1, id2>(data);\
    }\
  \
    static const RepDataTo run(RepDataFrom&& data)\
    {\
      return math::matrixToEulerAngles<S, id0, id1, id2>(std::move(data));\
    }\
  };

// Proper Euler angles (x-y-x, x-z-x, y-x-y, y-z-y, z-x-z, z-y-z)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(0, 1, 0)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(0, 2, 0)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(1, 0, 1)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(1, 2, 1)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(2, 0, 2)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(2, 1, 2)

// Tait–Bryan angles (x-y-z, x-z-y, y-x-z, y-z-x, z-x-y, z-y-x)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(0, 1, 2)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(0, 2, 1)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(1, 0, 2)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(1, 2, 0)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(2, 0, 1)
DART_SO3REPDATA_CONVERT_IMPL_TO_EULER_ANGLES(2, 1, 0)

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<SO3Vector<S>, SO3Matrix<S>>
{
  using RepDataFrom = typename Traits<SO3Vector<S>>::RepData;
  using RepDataTo = typename Traits<SO3Matrix<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return SO3Exp(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return SO3Exp(std::move(data));
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<SO3Vector<S>, AngleAxis<S>>
{
  using RepDataFrom = typename Traits<SO3Vector<S>>::RepData;
  using RepDataTo = typename Traits<AngleAxis<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    const S norm = data.norm();

    if (norm > static_cast<S>(0))
      return RepDataTo(norm, data/norm);
    else
      return RepDataTo(static_cast<S>(0), Eigen::Matrix<S, 3, 1>::UnitX());
  }
};

//==============================================================================
//template <typename S>
//struct SO3RepDataDirectConvertImpl<SO3Vector, Quaternion>
//{
//  using RepDataFrom = typename traits<SO3Vector<S>>::RepData;
//  using RepDataTo = typename traits<Quaternion<S>>::RepData;

//static constexpr bool IsSpecialized = true;

//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<AngleAxis<S>, SO3Vector<S>>
{
  using RepDataFrom = typename Traits<AngleAxis<S>>::RepData;
  using RepDataTo = typename Traits<SO3Vector<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return data.angle() * data.axis();
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<AngleAxis<S>, Quaternion<S>>
{
  using RepDataFrom = typename Traits<AngleAxis<S>>::RepData;
  using RepDataTo = typename Traits<Quaternion<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
//template <typename S, typename SO3Canonical>
//struct so3_convert2_impl<S, Quaternion, SO3Vector, SO3Canonical>
//{
//  using RepDataFrom = typename traits<Quaternion<S>>::RepData;
//  using RepDataTo = typename traits<SO3Vector<S>>::RepData;
//  static constexpr bool IsSpecialized = true;
//  static const RepDataTo run(const RepDataFrom& data)
//  {
//    return ;
//    // TODO(JS): Not implemented
//  }
//};

//==============================================================================
template <typename S>
struct SO3RepDataDirectConvertImpl<Quaternion<S>, AngleAxis<S>>
{
  using RepDataFrom = typename Traits<Quaternion<S>>::RepData;
  using RepDataTo = typename Traits<AngleAxis<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return RepDataTo(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return RepDataTo(std::move(data));
  }
};

//==============================================================================
#define DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(id0, id1, id2)\
  template <typename S>\
  struct SO3RepDataDirectConvertImpl<EulerAngles<S, id0, id1, id2>, SO3Matrix<S>>\
  {\
    using RepDataFrom = typename Traits<EulerAngles<S, id0, id1, id2>>::RepData;\
    using RepDataTo = typename Traits<SO3Matrix<S>>::RepData;\
  \
    static constexpr bool IsSpecialized = true;\
  \
    static const RepDataTo run(const RepDataFrom& data)\
    {\
      return math::eulerAnglesToMatrix<S, id0, id1, id2>(data);\
    }\
  \
    static const RepDataTo run(RepDataFrom&& data)\
    {\
      return math::eulerAnglesToMatrix<S, id0, id1, id2>(std::move(data));\
    }\
  };

// Proper Euler angles (x-y-x, x-z-x, y-x-y, y-z-y, z-x-z, z-y-z)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(0, 1, 0)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(0, 2, 0)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(1, 0, 1)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(1, 2, 1)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(2, 0, 2)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(2, 1, 2)

// Tait–Bryan angles (x-y-z, x-z-y, y-x-z, y-z-x, z-x-y, z-y-x)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(0, 1, 2)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(0, 2, 1)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(1, 0, 2)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(1, 2, 0)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(2, 0, 1)
DART_SO3REPDATA_CONVERT_IMPL_FROM_EULER_ANGLES(2, 1, 0)

//==============================================================================
// SO3RepDataConvertImpl
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   1   |   1   |   1   |   1   |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   1   |   0   |   1   |   2   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   1   |   1   |   0   |   1   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   1   |   2   |   1   |   0   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |   1   |   2   |   2   |   2   |   0   |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: no conversion; return input as const reference
// 1: convert once; from -> canonical, or canonical -> to
// 2: convert twice; from -> canonical -> to

//==============================================================================
template <typename SO3From,
          typename SO3To,
          typename SO3Via = DefaultSO3Canonical<typename Traits<SO3From>::S>,
          typename Enable = void>
struct SO3RepDataAssignOriginal
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return SO3RepDataDirectConvertImpl<SO3Via, SO3To>::run(
        SO3RepDataDirectConvertImpl<SO3From, SO3Via>::run(data));
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return SO3RepDataDirectConvertImpl<SO3Via, SO3To>::run(
        SO3RepDataDirectConvertImpl<SO3From, SO3Via>::run(std::move(data)));
  }
};

//==============================================================================
template <typename SO3From,
          typename SO3To,
          typename SO3Via>
struct SO3RepDataAssignOriginal<
    SO3From,
    SO3To,
    SO3Via,
    typename std::enable_if<
        SO3RepDataDirectConvertImpl<SO3From, SO3To>::IsSpecialized>::type
    >
{
  using RepDataFrom = typename Traits<SO3From>::RepData;
  using RepDataTo = typename Traits<SO3To>::RepData;

  static const RepDataTo run(const RepDataFrom& data)
  {
    return SO3RepDataDirectConvertImpl<SO3From, SO3To>::run(data);
  }

  static const RepDataTo run(RepDataFrom&& data)
  {
    return SO3RepDataDirectConvertImpl<SO3From, SO3To>::run(std::move(data));
  }
};

////==============================================================================
//// SO3Assign
////==============================================================================

////==============================================================================
//template <typename To, typename From, typename Enable = void>
//struct SO3RepDataAssign {};

////==============================================================================
//template <typename To, typename From>
//struct SO3RepDataAssign<To, From, typename std::enable_if<SO3IsEigen<To>::value>::type>
//{
//  static void run(To& to, const From& from)
//  {
//    SO3RepDataSetEigenFromX<To, From>::run(to, from);
//  }

//  static void run(To& to, From&& from)
//  {
//    SO3RepDataSetEigenFromX<To, From>::run(to, std::move(from));
//  }
//};

////==============================================================================
//template <typename To, typename From>
//struct SO3RepDataAssign<To, From, typename std::enable_if<SO3IsSO3<To>::value>::type>
//{
//  static void run(To& to, const From& from)
//  {
//    SO3RepDataSetSO3FromX<To, From>::run(to, from);
//  }

//  static void run(To& to, From&& from)
//  {
//    SO3RepDataSetSO3FromX<To, From>::run(to, std::move(from));
//  }
//};

















//==============================================================================
// SO3AssignEigenToEigen
//==============================================================================

template <typename EigenFrom, typename EigenTo>
struct SO3AssignEigenToEigen
{
  static void run(const EigenFrom& from, EigenTo& to)
  {
    static_assert(
        common::HasAssignmentOperator<EigenTo, EigenFrom>::value,
        "The assignment operator from EigenFrom to EigenTo is not define. "
        "Please make sure if the Eigen classes are relevant.");

    to = from;
  }

  static void run(EigenFrom&& from, EigenTo& to)
  {
    static_assert(
        common::HasMoveAssignmentOperator<EigenTo, EigenFrom>::value,
        "The move assignment operator from EigenFrom to EigenTo is not define. "
        "Please make sure if the Eigen classes are relevant.");

    to = std::move(from);
  }
};

//==============================================================================
// SO3AssignEigenToSO3
//==============================================================================

template <typename EigenFrom, typename SO3To, typename Enable = void>
struct SO3AssignEigenToSO3
{
  static void run(const EigenFrom& from, SO3To& to)
  {
    to.getRepData() = from;
  }

  static void run(EigenFrom&& from, SO3To& to)
  {
    to.getRepData() = std::move(from);
  }
};

//==============================================================================
// SO3AssignSO3ToEigen
//==============================================================================

template <typename SO3From, typename EigenTo, typename Enable = void>
struct SO3AssignSO3ToEigen
{
  static void run(const SO3From& from, EigenTo& to)
  {
    to = from.getRepData();
  }

  static void run(SO3From&& from, EigenTo& to)
  {
    to = std::move(from.getRepData());
  }
};

//==============================================================================
// SO3AssignSO3ToSO3
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   1   |   1   |   1   |   1   |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   1   |   0   |   1   |   2   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   1   |   1   |   0   |   1   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   1   |   2   |   1   |   0   |   2   |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |   1   |   2   |   2   |   2   |   0   |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: no conversion; return input as const reference
// 1: convert once; from -> canonical, or canonical -> to
// 2: convert twice; from -> canonical -> to

template <typename SO3From, typename SO3To, typename Enable = void>
struct SO3AssignSO3ToSO3
{
  static void run(const SO3From& from, SO3To& to)
  {
    to.fromCanonical(from.toCanonical());
  }

  static void run(SO3From&& from, SO3To& to)
  {
    to.fromCanonical(std::move(from.toCanonical()));
  }
};

//==============================================================================
template <typename SO3T>
struct SO3AssignSO3ToSO3<SO3T, SO3T>
{
  static void run(const SO3T& from, SO3T& to)
  {
    to.getRepData() = from.getRepData();
  }

  static void run(SO3T&& from, SO3T& to)
  {
    to.getRepData() = std::move(from.getRepData());
  }
};

//==============================================================================
template <typename SO3From, typename SO3To>
struct SO3AssignSO3ToSO3<
    SO3From,
    SO3To,
    typename std::enable_if<
        !std::is_same<SO3From, SO3To>::value
        && SO3IsEigen<typename Traits<SO3From>::RepData>::value
        && SO3IsEigen<typename Traits<SO3To>::RepData>::value
    >::type>
{
  static void run(const SO3From& from, SO3To& to)
  {
    to.getRepData() = from.getRepData();
  }

  static void run(SO3From&& from, SO3To& to)
  {
    to.getRepData() = std::move(from.getRepData());
  }
};

//==============================================================================
// SO3Assign
//==============================================================================

//==============================================================================
template <typename From, typename To, typename Enable = void>
struct SO3Assign {};

//==============================================================================
template <typename From, typename To>
struct SO3Assign<
    From,
    To,
    typename std::enable_if<SO3IsEigen2<From, To>::value>::type>
{
  static void run(const From& from, To& to)
  {
    SO3AssignEigenToEigen<From, To>::run(from, to);
  }

  static void run(From&& from, To& to)
  {
    SO3AssignEigenToEigen<From, To>::run(std::move(from), to);
  }
};

//==============================================================================
template <typename From, typename To>
struct SO3Assign<
    From,
    To,
    typename std::enable_if<SO3IsEigen<From>::value && SO3IsSO3<To>::value
    >::type>
{
  static void run(const From& from, To& to)
  {
    SO3AssignEigenToSO3<From, To>::run(from, to);
  }

  static void run(From&& from, To& to)
  {
    SO3AssignEigenToSO3<From, To>::run(std::move(from), to);
  }
};

//==============================================================================
template <typename From, typename To>
struct SO3Assign<
    From,
    To,
    typename std::enable_if<SO3IsSO3<From>::value && SO3IsEigen<To>::value
    >::type>
{
  static void run(const From& from, To& to)
  {
    SO3AssignSO3ToEigen<From, To>::run(from, to);
  }

  static void run(From&& from, To& to)
  {
    SO3AssignSO3ToEigen<From, To>::run(std::move(from), to);
  }
};

//==============================================================================
template <typename From, typename To>
struct SO3Assign<
    From,
    To,
    typename std::enable_if<SO3IsSO3_2<From, To>::value>::type>
{
  static void run(const From& from, To& to)
  {
    SO3AssignSO3ToSO3<From, To>::run(from, to);
  }

  static void run(From&& from, To& to)
  {
    SO3AssignSO3ToSO3<From, To>::run(std::move(from), to);
  }
};
































//==============================================================================
// SO3ToImpl
//==============================================================================

template <typename SO3From, typename To, typename Enable = void>
struct SO3ToImpl
{
  static To run(const SO3From& from)
  {
    To res;
    SO3Assign<SO3From, To>::run(from, res);

    return res;
  }
};

//==============================================================================
template <typename SO3T>
struct SO3ToImpl<SO3T, SO3T>
{
  static const SO3T& run(const SO3T& from)
  {
    return from;
  }
};

//==============================================================================
template <typename SO3From, typename To>
struct SO3ToImpl<
    SO3From,
    To,
    typename std::enable_if<
        std::is_same<
            typename Traits<SO3From>::RepData,
            To
        >::value
    >::type>
{
  static const To& run(const SO3From& from)
  {
    return from.getRepData();
  }
};

//==============================================================================
// SO3IsApprox
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   1   |   1   |   1   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   1   |   0   |   2   |   2   |       |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   1   |   2   |   0   |   2   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   1   |   2   |   2   |   0   |       |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |       |       |       |       |       |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: zero conversion; compare in the given representation
// 2: double conversion; repA -> canonical rep (compare) <- repB

//==============================================================================
template <typename SO3A,
          typename SO3B,
          typename SO3ToPerform = DefaultSO3Canonical<typename Traits<SO3A>::S>,
          typename Enable = void>
struct SO3IsApprox
{
  using S = typename Traits<SO3A>::S;

  static bool run(const SO3A& so3A, const SO3A& so3B, S tol)
  {
    const auto a = SO3ToImpl<SO3A, SO3ToPerform>::run(so3A);
    const auto b = SO3ToImpl<SO3B, SO3ToPerform>::run(so3B);

    return a.isApprox(b, tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifold rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename SO3A,
          typename SO3B,
          typename SO3ToPerform>
struct SO3IsApprox<
    SO3A,
    SO3B,
    SO3ToPerform,
    typename std::enable_if<
        std::is_same<SO3A, SO3ToPerform>::value
        && !std::is_same<SO3B, SO3ToPerform>::value
    >::type>
{
  using S = typename Traits<SO3A>::S;

  static bool run(const SO3A& so3A, const SO3A& so3B, S tol)
  {
    const auto b = SO3ToImpl<SO3B, SO3ToPerform>::run(so3B);

    return so3A.isApprox(b, tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifold rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename SO3A,
          typename SO3B,
          typename SO3ToPerform>
struct SO3IsApprox<
    SO3A,
    SO3B,
    SO3ToPerform,
    typename std::enable_if<
        !std::is_same<SO3A, SO3ToPerform>::value
        && std::is_same<SO3B, SO3ToPerform>::value
    >::type>
{
  using S = typename Traits<SO3A>::S;

  static bool run(const SO3A& so3A, const SO3A& so3B, S tol)
  {
    const auto a = SO3ToImpl<SO3A, SO3ToPerform>::run(so3B);

    return a.isApprox(so3B, tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifold rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename SO3A,
          typename SO3B,
          typename SO3ToPerform>
struct SO3IsApprox<
    SO3A,
    SO3B,
    SO3ToPerform,
    typename std::enable_if<
        std::is_same<SO3A, SO3ToPerform>::value
        && std::is_same<SO3B, SO3ToPerform>::value
    >::type>
{
  using S = typename SO3ToPerform::S;

  static bool run(const SO3ToPerform& so3A, const SO3ToPerform& so3B, S tol)
  {
    return so3A.getRepData().isApprox(so3B.getRepData(), tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifold rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
// SO3RepDataHomogeneousMultiplicationImpl
//==============================================================================

//==============================================================================
template <typename S, typename SO3T>
struct SO3RepDataHomogeneousMultiplicationImpl
{
  using RepData = typename Traits<SO3T>::RepData;

  static const RepData run(const RepData& data, const RepData& otherData)
  {
    return data * otherData;
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataHomogeneousMultiplicationImpl<S, SO3Vector<S>>
{
  using RepData = typename Traits<SO3Matrix<S>>::RepData;

  static const RepData run(const RepData& data, const RepData& otherData)
  {
    return SO3Log(SO3Exp(data) * SO3Exp(otherData));
  }
};

// TODO(JS): EulerAngles, AngleAxis(?)

//==============================================================================
// SO3RepDataHomogeneousInplaceMultiplicationImpl
//==============================================================================

//==============================================================================
template <typename S, typename SO3Canonical = DefaultSO3Canonical<S>>
struct SO3RepDataHomogeneousInplaceMultiplicationImpl
{
  using RepData = typename Traits<SO3Canonical>::RepData;

  static void run(RepData& data, const RepData& otherData)
  {
    data *= otherData;
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataHomogeneousInplaceMultiplicationImpl<S, SO3Vector<S>>
{
  using RepData = typename Traits<SO3Vector<S>>::RepData;

  static void run(RepData& data, const RepData& otherData)
  {
    data = SO3Log(SO3Exp(data) * SO3Exp(otherData));
  }
};

//==============================================================================
// SO3RepDataMultiplicationImpl
//==============================================================================

// +-------+ ------+-------+-------+-------+-------+
// |from\to|  Mat  |  Vec  |  Aa   | Quat  | Euler |
// +-------+ ------+-------+-------+-------+-------+
// |  Mat  |   0   |   -   |   -   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// |  Vec  |   X   |   3   |   -   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// |  Aa   |   X   |   X   |   0   |   -   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// | Quat  |   X   |   X   |   X   |   0   |   -   |
// +-------+ ------+-------+-------+-------+-------+
// | Euler |       |       |       |       |       |
// +-------+ ------+-------+-------+-------+-------+
//
// 0: zero conversion
// 3: triple conversions; [(rep -> canonical) * (rep -> canonical)] -> rep

//==============================================================================
template <typename SO3A, typename SO3B>
struct SO3RepDataMultiplicationImpl
{
  using RepDataTypeA = typename Traits<SO3A>::RepData;
  using RepDataTypeB = typename Traits<SO3B>::RepData;

  static auto run(const RepDataTypeA& dataA, const RepDataTypeB& dataB)
  -> decltype(dataA * dataB)
  {
    return dataA * dataB;
  }
};

//==============================================================================
template <typename S>
struct SO3RepDataMultiplicationImpl<SO3Vector<S>, SO3Vector<S>>
{
  using RepData = typename Traits<SO3Vector<S>>::RepData;
  using SO3CanonicalRep = Quaternion<S>; // TODO(JS): find best canonical for vec * vec
  using CanonicalRepData = typename Traits<SO3CanonicalRep>::RepData;

  static auto run(const RepData& dataA, const RepData& dataB)
  -> decltype(std::declval<CanonicalRepData>() * std::declval<CanonicalRepData>())
  {
    return SO3RepDataAssignOriginal<SO3Vector<S>, SO3CanonicalRep>::run(dataA)
        * SO3RepDataAssignOriginal<SO3Vector<S>, SO3CanonicalRep>::run(dataB);
    // TODO(JS): improve; super slow
  }
};

// TODO(JS): Heterogeneous multiplications are not implemented yet.

//==============================================================================
//template <typename S, typename RepB>
//struct SO3RepDataMultiplicationImpl<S, SO3Vector, RepB>
//{
//  using RepDataTypeA = typename traits<SO3Vector<S>>::RepData;
//  using RepData = typename traits<SO3B>::RepData;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepData& dataB)
//      -> decltype(std::declval<RepData>() * dataB)
//  {
//    return RepData(dataA) * dataB;
//  }
//};

////==============================================================================
//template <typename S, typename RepA>
//struct SO3RepDataMultiplicationImpl<S, RepA, SO3Vector>
//{
//  using RepDataTypeA = typename traits<SO3A>::RepData;
//  using RepData = typename traits<SO3Vector<S>>::RepData;

//  static const auto run(
//      const RepDataTypeA& dataA, const RepData& dataB)
//      -> decltype(dataA * std::declval<RepDataTypeA>())
//  {
//    return dataA * RepDataTypeA(dataB);
//  }
//};

//==============================================================================
// SO3IsCanonical
//==============================================================================

//==============================================================================
template <typename SO3Type,
          typename SO3Canonical
              = DefaultSO3Canonical<typename Traits<SO3Type>::S>,
          typename Enable = void>
struct SO3IsCanonical : std::false_type {};

//==============================================================================
template <typename SO3Type, typename SO3Canonical>
struct SO3IsCanonical<
    SO3Type,
    SO3Canonical,
    typename std::enable_if
        <std::is_same<typename SO3Type::Rep, SO3Canonical>::value>::type>
    : std::true_type {};

//==============================================================================
// SO3MultiplicationImpl
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (e.g., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A,
          typename SO3B,
          typename SO3ToPerform = DefaultSO3Canonical<typename Traits<SO3A>::S>,
          typename Enable = void>
struct SO3MultiplicationImpl
{
  using S = typename Traits<SO3A>::S;

  static SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(SO3RepDataAssignOriginal<SO3ToPerform, SO3A>::run(
          SO3RepDataHomogeneousMultiplicationImpl<S, SO3ToPerform>::run( // TODO(JS): Remove _canonical_
            SO3RepDataAssignOriginal<SO3A, SO3ToPerform>::run(Ra.getRepData()),
            SO3RepDataAssignOriginal<SO3B, SO3ToPerform>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between representation types supported by Eigen (i.e.,
// rotation matrix, AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B, typename SO3Canonical>
struct SO3MultiplicationImpl<
    SO3A,
    SO3B,
    SO3Canonical,
    typename std::enable_if<
        (SO3RepDataIsEigenRotationBase3Impl<SO3A>::value
            || std::is_same<typename Traits<SO3A>::RepData, Eigen::Matrix<typename Traits<SO3A>::S, 3, 3>>::value)
        &&
        (SO3RepDataIsEigenRotationBase3Impl<SO3B>::value
            || std::is_same<typename Traits<SO3B>::RepData, Eigen::Matrix<typename Traits<SO3B>::S, 3, 3>>::value)
    >::type>
{
  using S = typename Traits<SO3A>::S;

  using RepDataA = typename Traits<SO3A>::RepData;
  using RepDataB = typename Traits<SO3B>::RepData;

  static SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(Ra.getRepData() * Rb.getRepData());
  }
};

//==============================================================================
// SO3InplaceMultiplicationImpl
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (i.e., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A,
          typename SO3B,
          typename SO3Canonical = DefaultSO3Canonical<typename SO3A::S>,
          typename Enable = void>
struct SO3InplaceMultiplicationImpl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& Ra, const SO3B& Rb)
  {
    Ra.setRepData(SO3RepDataAssignOriginal<SO3Canonical, RepA>::run(
          SO3RepDataHomogeneousMultiplicationImpl<S, SO3Canonical>::run(
            SO3RepDataAssignOriginal<RepA, SO3Canonical>::run(Ra.getRepData()),
            SO3RepDataAssignOriginal<RepB, SO3Canonical>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between ones supported by Eigen (i.e., 3x3 matrix,
// AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B, typename SO3Canonical>
struct SO3InplaceMultiplicationImpl<
    SO3A,
    SO3B,
    SO3Canonical,
    typename std::enable_if<SO3RepDataIsSupportedByEigenImpl<SO3A, SO3B>::value>::type
    >
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& so3A, const SO3B& so3B)
  {
    so3A.getRepData() *= so3B.getRepData();
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3OPERATIONS_HPP_
