/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
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

#ifndef DART_UTILS_VSKPARSER_HPP_
#define DART_UTILS_VSKPARSER_HPP_

#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/math/Constants.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace utils {

namespace VskParser
{
  /// Options struct is additional information that helps building a skeleton
  /// that can be used in kinematics or dynamics simulation. VSK file format
  /// itself doesn't provide essential properties for it such as body's shape,
  /// mass, and inertia.
  struct Options
  {
    /// Resource retriever. LocalResourceRetriever is used if it's nullptr.
    common::ResourceRetrieverPtr retrieverOrNullptr;

    /// The default shape for body node is ellipsoid. The size of ellipsoid of
    /// each body node are determined by the relative transformation from a body
    /// node and its child body node. defaultEllipsoidSize is used for body
    /// nodes that don't have child body node.
    Eigen::Vector3s defaultEllipsoidSize;

    /// Ratio of shorter radii of each ellipsoid to the longest radius where
    /// the longest radius is the distance between a body and its child body
    /// node.
    s_t thicknessRatio;

    /// Density of each ellipsoid that are used to compute mass.
    s_t density;

    /// Lower limit of joint position
    s_t jointPositionLowerLimit;

    /// Upper limit of joint position
    s_t jointPositionUpperLimit;

    /// Joint damping coefficient
    s_t jointDampingCoefficient;

    /// Joint Coulomb friction
    s_t jointFriction;

    /// Remove end BodyNodes with no Shape segment
    bool removeEndBodyNodes;

    /// Constructor
    Options(const common::ResourceRetrieverPtr& retrieverOrNullptr = nullptr,
            const Eigen::Vector3s& defaultEllipsoidSize
                = Eigen::Vector3s::Constant(0.05),
            s_t thicknessRatio = 0.35,
            s_t density = 1e+3,
            s_t jointPositionLowerLimit = -math::constantsd::pi(),
            s_t jointPositionUpperLimit = +math::constantsd::pi(),
            s_t jointDampingCoefficient = 0.1,
            s_t jointFriction = 0.0,
            bool removeEndBodyNodes = false);
  };

  /// Read Skeleton from VSK file
  dynamics::SkeletonPtr readSkeleton(const common::Uri& fileUri,
                                     Options options = Options());

} // namespace VskParser

} // namespace utils
} // namespace dart

#endif // #ifndef DART_UTILS_VSKPARSER_HPP_

