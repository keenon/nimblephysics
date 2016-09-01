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
// is_canonical:
//==============================================================================

//==============================================================================
template <typename SO3Type, typename Enable = void>
struct is_canonical : std::false_type {};

template <typename SO3Type>
struct is_canonical<
    SO3Type,
    typename std::enable_if
        <std::is_same<typename SO3Type::Rep, SO3CanonicalRep>::value>::type>
    : std::true_type {};

//==============================================================================
// assign_impl:
//==============================================================================

//==============================================================================
template <typename S, typename SO3To, typename SO3From>
struct assign_impl
{
  using RepFrom = typename SO3From::Rep;
  using RepTo = typename SO3To::Rep;

  static void run(SO3To& dataTo, const SO3From& dataFrom)
  {
    dataTo.matrix() = convert_to_noncanonical_impl<S, RepTo>::run(
          convert_to_canonical_impl<S, RepFrom>::run(dataFrom.matrix()));
  }
};

//==============================================================================
// canonical_inplace_group_multiplication_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B, typename Enable = void>
struct inplace_group_multiplication_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& data, const SO3B& otherData)
  {
    data.matrix() = convert_to_noncanonical_impl<S, RepA>::run(
          canonical_group_multiplication_impl<S>::run(
            convert_to_canonical_impl<S, RepA>::run(data.matrix()),
            convert_to_canonical_impl<S, RepB>::run(otherData.matrix())));
  }
};

//==============================================================================
template <typename SO3A, typename SO3B>
struct inplace_group_multiplication_impl<
    SO3A,
    SO3B,
    typename std::enable_if<is_canonical<SO3A>::value>::type>
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static void run(SO3A& data, const SO3B& otherData)
  {
    data.matrix() *= convert_to_canonical_impl<S, RepB>::run(
          otherData.matrix());
  }
};

//==============================================================================
// assign_impl:
//==============================================================================

//==============================================================================
template <typename SO3A, typename SO3B>
struct is_approx_impl
{
  using S = typename SO3A::S;

  using RepA = typename SO3A::Rep;
  using RepB = typename SO3B::Rep;

  static bool run(const SO3A& dataA, const SO3B& dataB, S tol)
  {
    return convert_to_canonical_impl<S, RepA>::run(dataA.matrix())
        .isApprox(
          convert_to_canonical_impl<S, RepB>::run(dataB.matrix()),
          tol);
  }
};

} // namespace SO3
} // namespace detail
} // namespace math
} // namespace dart

#endif // DART_MATH_DETAIL_SO3UTILS_HPP_
