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

#ifndef DART_MATH_DETAIL_SO3BASE_HPP_
#define DART_MATH_DETAIL_SO3BASE_HPP_

#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {
namespace detail {
namespace SO3 {

//==============================================================================
// group_is_canonical:
//==============================================================================

//==============================================================================
template <typename SO3Type,
          typename SO3CanonicalRep = DefaultSO3CanonicalRep,
          typename Enable = void>
struct group_is_canonical : std::false_type {};

template <typename SO3Type, typename SO3CanonicalRep>
struct group_is_canonical<
    SO3Type,
    SO3CanonicalRep,
    typename std::enable_if
        <std::is_same<typename SO3Type::Rep, SO3CanonicalRep>::value>::type>
    : std::true_type {};

//==============================================================================
// assign_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3To, typename SO3From>
struct group_assign_impl
{
  using RepFrom = typename SO3From::Rep;
  using RepTo = typename SO3To::Rep;

  static void run(SO3To& to, const SO3From& from)
  {
    to.setRepData(rep_convert_impl<S, RepFrom, RepTo>::run(from.getRepData()));
  }
};

//==============================================================================
// group_multiplication_impl:
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (i.e., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A, typename SO3B, typename Enable = void>
struct group_multiplication_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static const SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(rep_convert_from_canonical_impl<S, RepA>::run(
          rep_canonical_multiplication_impl<S>::run(
            rep_convert_to_canonical_impl<S, RepA>::run(Ra.getRepData()),
            rep_convert_to_canonical_impl<S, RepB>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between ones supported by Eigen (i.e., 3x3 matrix,
// AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B>
struct group_multiplication_impl<
    SO3A,
    SO3B,
    typename std::enable_if<
//        !std::is_same<typename SO3A::RepDataType,  typename SO3B::RepDataType>::value
        true
        && (std::is_same<typename SO3A::RepDataType, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3A::RepDataType, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3A::RepDataType, Eigen::Quaternion<typename SO3A::S>>::value)
        && (std::is_same<typename SO3B::RepDataType, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3B::RepDataType, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3B::RepDataType, Eigen::Quaternion<typename SO3A::S>>::value)
    >::type>
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  using RepDataTypeA = typename rep_traits<S, RepA>::RepDataType;
  using RepDataTypeB = typename rep_traits<S, RepB>::RepDataType;

  static const SO3A run(const SO3A& Ra, const SO3B& Rb)
  {
    return SO3A(Ra.getRepData() * Rb.getRepData());
  }
};

//==============================================================================
// group_inplace_multiplication_impl:
//==============================================================================

//==============================================================================
// Generic version. Convert the input representation to canonical representation
// (i.e., 3x3 rotation matrix), perform group multiplication for those converted
// 3x3 rotation matrices, then finally convert the result to the output
// representation.
template <typename SO3A, typename SO3B, typename Enable = void>
struct group_inplace_multiplication_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& Ra, const SO3B& Rb)
  {
    Ra.setRepData(rep_convert_from_canonical_impl<S, RepA>::run(
          rep_canonical_multiplication_impl<S>::run(
            rep_convert_to_canonical_impl<S, RepA>::run(Ra.getRepData()),
            rep_convert_to_canonical_impl<S, RepB>::run(Rb.getRepData()))));
  }
};

//==============================================================================
// Data conversions between ones supported by Eigen (i.e., 3x3 matrix,
// AngleAxis, and Quaternion)
template <typename SO3A, typename SO3B>
struct group_inplace_multiplication_impl<
    SO3A,
    SO3B,
    typename std::enable_if<
//        !std::is_same<typename SO3A::RepDataType,  typename SO3B::RepDataType>::value
        true
        && (std::is_same<typename SO3A::RepDataType, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3A::RepDataType, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3A::RepDataType, Eigen::Quaternion<typename SO3A::S>>::value)
        && (std::is_same<typename SO3B::RepDataType, Eigen::Matrix<typename SO3A::S, 3, 3>>::value
            || std::is_same<typename SO3B::RepDataType, Eigen::AngleAxis<typename SO3A::S>>::value
            || std::is_same<typename SO3B::RepDataType, Eigen::Quaternion<typename SO3A::S>>::value)
    >::type>
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& so3A, const SO3B& so3B)
  {
//    so3A.getRepData() *= so3B.getRepData();
  }
};

//==============================================================================
// group_is_approx_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B>
struct group_is_approx_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static bool run(const SO3A& Ra, const SO3B& Rb, S tol)
  {
    return rep_is_approx_impl<S, RepA, RepB>::run(
          Ra.getRepData(), Rb.getRepData(), tol);
  }
};

} // namespace SO3
} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3UTILS_HPP_
