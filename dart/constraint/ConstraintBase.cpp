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

#include "dart/constraint/ConstraintBase.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "dart/dynamics/Skeleton.hpp"

namespace dart {
namespace constraint {

//==============================================================================
ConstraintBase::ConstraintBase() : mDim(0)
{
}

//==============================================================================
ConstraintBase::~ConstraintBase()
{
}

//==============================================================================
std::size_t ConstraintBase::getDimension() const
{
  return mDim;
}

//==============================================================================
void ConstraintBase::uniteSkeletons()
{
  // Do nothing
}

//==============================================================================
std::vector<dynamics::SkeletonPtr> ConstraintBase::getSkeletons() const
{
  std::vector<dynamics::SkeletonPtr> skeletons;
  // Return an empty vector
  return skeletons;
}

//==============================================================================
dynamics::SkeletonPtr ConstraintBase::compressPath(
    dynamics::SkeletonPtr _skeleton)
{
  while (_skeleton->mUnionRootSkeleton.lock() != _skeleton)
  {
    _skeleton->mUnionRootSkeleton
        = _skeleton->mUnionRootSkeleton.lock()->mUnionRootSkeleton.lock();
    _skeleton = _skeleton->mUnionRootSkeleton.lock();
  }
  // TODO(keenon): It's possible that mUnionRootSkeleton doesn't actually
  // point to the root skeleton for every skeleton at this point in the code.
  // Consider a chain of 4 nodes in a graph.
  //
  //    0  <--- 1 <--- 2 <--- 3
  //
  // After this code runs, we'd have
  //
  //            +-------------+
  //            |             |
  //            v             |
  //    0  <--- 1 <--- 2      3
  //
  // Other code seems to assume that every node would point to 0 after this
  // code. Perhaps we should change that if it's not intended behavior?

  return _skeleton;
}

//==============================================================================
dynamics::SkeletonPtr ConstraintBase::getRootSkeleton(
    dynamics::SkeletonPtr _skeleton)
{
  while (_skeleton->mUnionRootSkeleton.lock() != _skeleton)
    _skeleton = _skeleton->mUnionRootSkeleton.lock();

  return _skeleton;
}

//==============================================================================
double ConstraintBase::getCoefficientOfRestitution()
{
  return 0;
}

//==============================================================================
double ConstraintBase::getPenetrationCorrectionVelocity()
{
  return 0;
}

} // namespace constraint
} // namespace dart
