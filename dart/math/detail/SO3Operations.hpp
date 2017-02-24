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
        std::is_same<T, Eigen::Matrix<typename T::Scalar, 3, 3>>::value
            || std::is_same<T, Eigen::AngleAxis<typename T::Scalar>>::value
            || std::is_same<T, Eigen::Quaternion<typename T::Scalar>>::value
    >::type>
    : std::true_type {};

//==============================================================================
// SO3IsSupportedEigenType2
//==============================================================================

//==============================================================================
template <typename EigenA, typename EigenB, typename Enable = void>
struct SO3IsEigen2 : std::false_type {};

//==============================================================================
template <typename EigenA, typename EigenB>
struct SO3IsEigen2<
    EigenA,
    EigenB,
    typename std::enable_if<
        SO3IsEigen<EigenA>::value
        && SO3IsEigen<EigenB>::value
    >::type>
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
        dart::common::is_base_of_template<T, SO3Base>::value
    >::type>
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
// 0: zero conversion; return input as const reference
// 1: single conversion; from -> canonical, or canonical -> to
// 2: double conversion; from -> canonical -> to

//==============================================================================
template <typename SO3From,
          typename SO3To,
          typename SO3Via = DefaultSO3Canonical<typename Traits<SO3From>::S>,
          typename Enable = void>
struct SO3RepDataConvertImpl
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
struct SO3RepDataConvertImpl<
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



















//==============================================================================
// SO3SetEigenFromEigen
//==============================================================================

template <typename EigenTo, typename EigenFrom>
struct SO3SetEigenFromEigen
{
  static void run(EigenTo& to, const EigenFrom& from)
  {
    static_assert(
        common::HasAssignmentOperator<EigenTo, EigenFrom>::value,
        "The assignment operator from EigenFrom to EigenTo is not define. "
        "Please make sure if the Eigen classes are relevant.");

    to = from;
  }

  static void run(EigenTo& to, EigenFrom&& from)
  {
    static_assert(
        common::HasMoveAssignmentOperator<EigenTo, EigenFrom>::value,
        "The move assignment operator from EigenFrom to EigenTo is not define. "
        "Please make sure if the Eigen classes are relevant.");

    to = std::move(from);
  }
};

//==============================================================================
// SO3SetEigenFromSO3
//==============================================================================

template <typename SO3To, typename SO3From, typename Enable = void>
struct SO3SetEigenFromSO3 {};

//==============================================================================
// SO3SetSO3FromSO3
//==============================================================================

template <typename SO3To, typename SO3From, typename Enable = void>
struct SO3SetSO3FromSO3
{
  using RepDataTo = typename Traits<SO3To>::RepData;
  using RepDataFrom = typename Traits<SO3From>::RepData;

  static constexpr bool IsSpecialized = false;

  static void run(SO3To& to, const SO3From& from)
  {
    to.getRepData() = from.getRepData();
  }

  static void run(SO3To& to, SO3From&& from)
  {
    to.getRepData() = std::move(from.getRepData());
  }
};

//==============================================================================
template <typename SO3T>
struct SO3SetSO3FromSO3<SO3T, SO3T>
{
  using RepData = typename Traits<SO3T>::RepData;

  static constexpr bool IsSpecialized = true;

  static void run(SO3T& to, const SO3T& from)
  {
    to.getRepData() = from.getRepData();
  }

  static void run(SO3T& to, SO3T&& from)
  {
    to.getRepData() = std::move(from.getRepData());
  }
};

//==============================================================================
template <typename S>
struct SO3SetSO3FromSO3<SO3Matrix<S>, SO3Vector<S>>
{
  using RepDataTo = typename Traits<SO3Matrix<S>>::RepData;
  using RepDataFrom = typename Traits<SO3Vector<S>>::RepData;

  static constexpr bool IsSpecialized = true;

  static void run(SO3Matrix<S>& to, const SO3Vector<S>& from)
  {
    SO3Exp(to.getRepData(), from.getRepData());
  }

  static void run(SO3Matrix<S>& to, SO3Vector<S>&& from)
  {
    SO3Exp(to.getRepData(), std::move(from.getRepData()));
  }
};

//==============================================================================
// SO3SetSO3FromEigen
//==============================================================================

template <typename SO3To, typename EigenFrom, typename Enable = void>
struct SO3SetSO3FromEigen
{
  static void run(SO3To& to, const EigenFrom& from)
  {
//    to = from;
  }

  static void run(SO3To& to, EigenFrom&& from)
  {
//    to = std::move(from);
  }
};

//==============================================================================
template <typename SO3To, typename EigenFrom>
struct SO3SetSO3FromEigen<
    SO3To,
    EigenFrom,
    int>
//    typename std::enable_if<
//        SO3IsSO3<EigenTo, EigenFrom>::value
//    >::type>
{
  static void run(SO3To& to, const EigenFrom& from)
  {
//    to = from;
  }

  static void run(SO3To& to, EigenFrom&& from)
  {
//    to = std::move(from);
  }
};

//==============================================================================
// SO3SetEigenFromX
//==============================================================================

template <typename EigenTo, typename From, typename Enable = void>
struct SO3SetEigenFromX {};

//==============================================================================
template <typename EigenTo, typename From>
struct SO3SetEigenFromX<
    EigenTo,
    From,
    typename std::enable_if<SO3IsEigen<From>::value>::type
    >
{
  static void run(EigenTo& to, const From& from)
  {
    SO3SetEigenFromEigen<EigenTo, From>::run(to, from);
  }

  static void run(EigenTo& to, From&& from)
  {
    SO3SetEigenFromEigen<EigenTo, From>::run(to, std::move(from));
  }
};

//==============================================================================
// SO3SetSO3FromX
//==============================================================================

template <typename SO3To, typename From, typename Enable = void>
struct SO3SetSO3FromX {};

//==============================================================================
template <typename SO3To, typename From>
struct SO3SetSO3FromX<
    SO3To,
    From,
    typename std::enable_if<SO3IsSO3<From>::value>::type
    >
{
  static void run(SO3To& to, const From& from)
  {
    SO3SetSO3FromSO3<SO3To, From>::run(to, from);
  }

  static void run(SO3To& to, From&& from)
  {
    SO3SetSO3FromSO3<SO3To, From>::run(to, std::move(from));
  }
};

//==============================================================================
template <typename SO3To, typename From>
struct SO3SetSO3FromX<
    SO3To,
    From,
    typename std::enable_if<SO3IsEigen<From>::value>::type
    >
{
  static void run(SO3To& to, const From& from)
  {
    SO3SetSO3FromEigen<SO3To, From>::run(to, from);
  }

  static void run(SO3To& to, From&& from)
  {
    SO3SetSO3FromEigen<SO3To, From>::run(to, std::move(from));
  }
};

//==============================================================================
template <typename To, typename From, typename Enable = void>
struct SO3Assign {};

//==============================================================================
template <typename To, typename From>
struct SO3Assign<To, From, typename std::enable_if<SO3IsEigen<To>::value>::type>
{
  static void run(To& to, const From& from)
  {
    SO3SetEigenFromX<To, From>::run(to, from);
  }

  static void run(To& to, From&& from)
  {
    SO3SetEigenFromX<To, From>::run(to, std::move(from));
  }
};

//==============================================================================
template <typename To, typename From>
struct SO3Assign<To, From, typename std::enable_if<SO3IsSO3<To>::value>::type>
{
  static void run(To& to, const From& from)
  {
    SO3SetSO3FromX<To, From>::run(to, from);
  }

  static void run(To& to, From&& from)
  {
    SO3SetSO3FromX<To, From>::run(to, std::move(from));
  }
};

































//==============================================================================
// SO3ConvertEigenToEigen
//==============================================================================

//==============================================================================
template <typename EigenFrom, typename EigenTo, typename Enable = void>
struct SO3ConvertEigenToEigen {};

//==============================================================================
template <typename EigenFrom, typename EigenTo>
struct SO3ConvertEigenToEigen<
    EigenFrom,
    EigenTo,
    typename std::enable_if<
        SO3IsEigen2<EigenFrom, EigenTo>::value
    >::type>
{
  static const EigenTo run(const EigenFrom& data)
  {
    return EigenTo(data);
  }

  static const EigenTo run(EigenFrom&& data)
  {
    return EigenTo(std::move(data));
  }
};

//==============================================================================
// SO3ConvertEigenToDart
//==============================================================================

template <typename EigenFrom, typename To, typename Enable = void>
struct SO3ConvertEigenToDart {};

//==============================================================================
// SO3ConvertDartToEigen
//==============================================================================

template <typename EigenFrom, typename To, typename Enable = void>
struct SO3ConvertDartToEigen {};

//==============================================================================
// SO3ConvertDartToDart
//==============================================================================

template <typename EigenFrom, typename To, typename Enable = void>
struct SO3ConvertDartToDart {};

//==============================================================================
// SO3ConvertEigenToX
//==============================================================================

template <typename EigenFrom, typename To, typename Enable = void>
struct SO3ConvertEigenToX {};

//==============================================================================
template <typename EigenFrom, typename SO3To>
struct SO3ConvertEigenToX<
    EigenFrom,
    SO3To,
    typename std::enable_if<SO3IsEigen<SO3To>::value>::type
    >
{
  static const SO3To run(const EigenFrom& data)
  {
    return SO3ConvertEigenToEigen<EigenFrom, SO3To>::run(data);
  }

  static const SO3To run(EigenFrom&& data)
  {
    return SO3ConvertEigenToEigen<EigenFrom, SO3To>::run(std::move(data));
  }
};

////==============================================================================
//template <typename DartSO3From, typename To, typename Enable = void>
//struct SO3ConvertDartToX
//{
////  using RepDataFrom = typename Traits<SO3From>::RepData;
////  using RepDataTo = typename Traits<SO3To>::RepData;

//  static const SO3To run(const SO3From& data)
//  {
//  }

//  static const SO3To run(SO3From&& data)
//  {
//  }
//};

//==============================================================================
template <typename SO3From, typename SO3To, typename Enable = void>
struct SO3Convert_ {};

//==============================================================================
template <typename SO3From, typename SO3To>
struct SO3Convert_<
    SO3From,
    SO3To,
    typename std::enable_if<SO3IsEigen<SO3From>::value>::type
    >
{
  static const SO3To run(const SO3From& data)
  {
    return SO3ConvertEigenToX<SO3From, SO3To>::run(data);
  }

  static const SO3To run(SO3From&& data)
  {
    return SO3ConvertEigenToX<SO3From, SO3To>::run(std::move(data));
  }
};

//==============================================================================
// SO3RepDataIsApproxImpl
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
          typename SO3ToPerform = DefaultSO3Canonical<typename Traits<SO3A>::S>>
struct SO3RepDataIsApproxImpl
{
  using RepDataTypeA = typename Traits<SO3A>::RepData;
  using RepDataTypeB = typename Traits<SO3B>::RepData;

  using S = typename Traits<SO3A>::S;

  static bool run(const RepDataTypeA& dataA, const RepDataTypeB& dataB, S tol)
  {
    return SO3RepDataDirectConvertImpl<SO3A, SO3ToPerform>::run(dataA)
        .isApprox(SO3RepDataDirectConvertImpl<SO3B, SO3ToPerform>::run(dataB), tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifolds rather than one provided
    // by Eigen that might be the Euclidean distance metric (not sure).
  }
};

//==============================================================================
template <typename SO3Type>
struct SO3RepDataIsApproxImpl<SO3Type, SO3Type>
{
  using RepData = typename Traits<SO3Type>::RepData;

  using S = typename Traits<SO3Type>::S;

  static bool run(const RepData& dataA, const RepData& dataB, S tol)
  {
    return dataA.isApprox(dataB, tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // discrepancy between two points on the manifolds rather than one provided
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
    return SO3RepDataConvertImpl<SO3Vector<S>, SO3CanonicalRep>::run(dataA)
        * SO3RepDataConvertImpl<SO3Vector<S>, SO3CanonicalRep>::run(dataB);
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
    return SO3A(SO3RepDataConvertImpl<SO3ToPerform, SO3A>::run(
          SO3RepDataHomogeneousMultiplicationImpl<S, SO3ToPerform>::run( // TODO(JS): Remove _canonical_
            SO3RepDataConvertImpl<SO3A, SO3ToPerform>::run(Ra.getRepData()),
            SO3RepDataConvertImpl<SO3B, SO3ToPerform>::run(Rb.getRepData()))));
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
    Ra.setRepData(SO3RepDataConvertImpl<SO3Canonical, RepA>::run(
          SO3RepDataHomogeneousMultiplicationImpl<S, SO3Canonical>::run(
            SO3RepDataConvertImpl<RepA, SO3Canonical>::run(Ra.getRepData()),
            SO3RepDataConvertImpl<RepB, SO3Canonical>::run(Rb.getRepData()))));
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

//==============================================================================
// SO3ToImpl
//==============================================================================

template <typename SO3From, typename SO3OrEigenObject, typename Enable = void>
struct SO3ToImpl {};

//==============================================================================
// Converting to the raw data type from given SO3 representation type
template <typename SO3From, typename SO3OrEigenObject>
struct SO3ToImpl<
    SO3From,
    SO3OrEigenObject,
    typename std::enable_if<
        std::is_base_of<SO3Base<SO3OrEigenObject>, SO3OrEigenObject>::value>::type
    >
{
  using RepData = typename detail::Traits<SO3From>::RepData;

  static SO3OrEigenObject run(const RepData& data)
  {
    return SO3OrEigenObject(detail::SO3RepDataConvertImpl<SO3From, SO3OrEigenObject>::run(
          data));
  }
};

//==============================================================================
// Converting to the raw data type from given raw data type
template <typename SO3From, typename SO3OrEigenObject>
struct SO3ToImpl<
    SO3From,
    SO3OrEigenObject,
    typename std::enable_if<
        std::is_same<
            typename detail::Traits<SO3Matrix<typename Traits<SO3From>::S>>::RepData,
            SO3OrEigenObject>::value
        >::type
    >
{
  using S = typename Traits<SO3From>::S;
  using RepData = typename detail::Traits<SO3From>::RepData;

  static auto run(const RepData& data)
  -> decltype(detail::SO3RepDataConvertImpl<SO3From, SO3Matrix<S>>::run(
      std::declval<RepData>()))
  {
    return detail::SO3RepDataConvertImpl<SO3From, SO3Matrix<S>>::run(data);
  }
};

//==============================================================================
// Converting to the raw data type from given raw data type
template <typename SO3From, typename SO3OrEigenObject>
struct SO3ToImpl<
    SO3From,
    SO3OrEigenObject,
    typename std::enable_if<
        std::is_same<
            typename detail::Traits<AngleAxis<typename Traits<SO3From>::S>>::RepData,
            SO3OrEigenObject>::value
        >::type
    >
{
  using S = typename Traits<SO3From>::S;
  using RepData = typename detail::Traits<SO3From>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::SO3RepDataConvertImpl<SO3From, AngleAxis<S>>::run(
      std::declval<RepData>()))
  {
    return detail::SO3RepDataConvertImpl<SO3From, AngleAxis<S>>::run(repData);
  }
};


//==============================================================================
// Converting to the raw data type from given raw data type
template <typename SO3From, typename SO3OrEigenObject>
struct SO3ToImpl<
    SO3From,
    SO3OrEigenObject,
    typename std::enable_if<
        std::is_same<
            typename detail::Traits<Quaternion<typename Traits<SO3From>::S>>::RepData,
            SO3OrEigenObject>::value
        >::type
    >
{
  using S = typename Traits<SO3From>::S;
  using RepData = typename detail::Traits<SO3From>::RepData;

  static auto run(const RepData& repData)
  -> decltype(detail::SO3RepDataConvertImpl<SO3From, Quaternion<S>>::run(
      std::declval<RepData>()))
  {
    return detail::SO3RepDataConvertImpl<SO3From, Quaternion<S>>::run(
          repData);
  }
};

} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3OPERATIONS_HPP_
