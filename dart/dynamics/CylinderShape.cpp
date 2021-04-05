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

#include "dart/dynamics/CylinderShape.hpp"

#include <cmath>
#include "dart/math/Helpers.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
CylinderShape::CylinderShape(s_t _radius, s_t _height)
  : Shape(CYLINDER),
    mRadius(_radius),
    mHeight(_height)
{
  assert(0.0 < _radius);
  assert(0.0 < _height);
}

//==============================================================================
const std::string& CylinderShape::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& CylinderShape::getStaticType()
{
  static const std::string type("CylinderShape");
  return type;
}

//==============================================================================
s_t CylinderShape::getRadius() const
{
  return mRadius;
}

//==============================================================================
void CylinderShape::setRadius(s_t _radius)
{
  assert(0.0 < _radius);
  mRadius = _radius;
  mIsBoundingBoxDirty = true;
  mIsVolumeDirty = true;

  incrementVersion();
}

//==============================================================================
s_t CylinderShape::getHeight() const
{
  return mHeight;
}

//==============================================================================
void CylinderShape::setHeight(s_t _height)
{
  assert(0.0 < _height);
  mHeight = _height;
  mIsBoundingBoxDirty = true;
  mIsVolumeDirty = true;

  incrementVersion();
}

//==============================================================================
s_t CylinderShape::computeVolume(s_t radius, s_t height)
{
  return math::constantsd::pi() * pow(radius, 2) * height;
}

//==============================================================================
Eigen::Matrix3s CylinderShape::computeInertia(
    s_t radius, s_t height, s_t mass)
{
  Eigen::Matrix3s inertia = Eigen::Matrix3s::Zero();

  inertia(0, 0) = mass * (3.0 * pow(radius, 2) + pow(height, 2))
      / 12.0;
  inertia(1, 1) = inertia(0, 0);
  inertia(2, 2) = 0.5 * mass * radius * radius;

  return inertia;
}

//==============================================================================
void CylinderShape::updateBoundingBox() const
{
  mBoundingBox.setMin(Eigen::Vector3s(-mRadius, -mRadius, -mHeight * 0.5));
  mBoundingBox.setMax(Eigen::Vector3s(mRadius, mRadius, mHeight * 0.5));
  mIsBoundingBoxDirty = false;
}

//==============================================================================
void CylinderShape::updateVolume() const
{
  mVolume = computeVolume(mRadius, mHeight);
  mIsVolumeDirty = false;
}

//==============================================================================
Eigen::Matrix3s CylinderShape::computeInertia(s_t mass) const
{
  return computeInertia(mRadius, mHeight, mass);
}

}  // namespace dynamics
}  // namespace dart
