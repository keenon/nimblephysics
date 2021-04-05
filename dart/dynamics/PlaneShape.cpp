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

#include "dart/dynamics/PlaneShape.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
PlaneShape::PlaneShape(const Eigen::Vector3s& _normal, s_t _offset)
  : Shape(PLANE),
    mNormal(_normal.normalized()),
    mOffset(_offset)
{
}

//==============================================================================
PlaneShape::PlaneShape(const Eigen::Vector3s& _normal,
                       const Eigen::Vector3s& _point)
  : Shape(),
    mNormal(_normal.normalized()),
    mOffset(mNormal.dot(_point))
{
}

//==============================================================================
const std::string& PlaneShape::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& PlaneShape::getStaticType()
{
  static const std::string type("PlaneShape");
  return type;
}

//==============================================================================
Eigen::Matrix3s PlaneShape::computeInertia(s_t /*mass*/) const
{
  return Eigen::Matrix3s::Zero();
}

//==============================================================================
void PlaneShape::setNormal(const Eigen::Vector3s& _normal)
{
  mNormal = _normal.normalized();

  incrementVersion();
}

//==============================================================================
const Eigen::Vector3s& PlaneShape::getNormal() const
{
  return mNormal;
}

//==============================================================================
void PlaneShape::setOffset(s_t _offset)
{
  mOffset = _offset;

  incrementVersion();
}

//==============================================================================
s_t PlaneShape::getOffset() const
{
  return mOffset;
}

//==============================================================================
void PlaneShape::setNormalAndOffset(const Eigen::Vector3s& _normal,
                                    s_t _offset)
{
  setNormal(_normal);
  setOffset(_offset);
}

//==============================================================================
void PlaneShape::setNormalAndPoint(const Eigen::Vector3s& _normal,
                                   const Eigen::Vector3s& _point)
{
  setNormal(_normal);
  setOffset(mNormal.dot(_point));
}

//==============================================================================
s_t PlaneShape::computeDistance(const Eigen::Vector3s& _point) const
{
  return abs(computeSignedDistance(_point));
}

//==============================================================================
s_t PlaneShape::computeSignedDistance(const Eigen::Vector3s& _point) const
{
  return mNormal.dot(_point) - mOffset;
}

//==============================================================================
void PlaneShape::updateBoundingBox() const
{
  mBoundingBox.setMin(
      Eigen::Vector3s::Constant(-std::numeric_limits<s_t>::infinity()));
  mBoundingBox.setMax(
      Eigen::Vector3s::Constant(std::numeric_limits<s_t>::infinity()));

  mIsBoundingBoxDirty = false;
}

//==============================================================================
void PlaneShape::updateVolume() const
{
  mVolume = 0.0;
  mIsVolumeDirty = false;
}

}  // namespace dynamics
}  // namespace dart
