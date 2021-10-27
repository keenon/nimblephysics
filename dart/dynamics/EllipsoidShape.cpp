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

#include "dart/dynamics/EllipsoidShape.hpp"

#include "dart/math/Helpers.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
EllipsoidShape::EllipsoidShape(const Eigen::Vector3s& diameters)
  : Shape(ELLIPSOID)
{
  setDiameters(diameters);
}

//==============================================================================
EllipsoidShape::~EllipsoidShape()
{
  // Do nothing
}

//==============================================================================
const std::string& EllipsoidShape::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& EllipsoidShape::getStaticType()
{
  static const std::string type("EllipsoidShape");
  return type;
}

//==============================================================================
void EllipsoidShape::setSize(const Eigen::Vector3s& diameters)
{
  setDiameters(diameters);
}

//==============================================================================
const Eigen::Vector3s& EllipsoidShape::getSize() const
{
  return getDiameters();
}

//==============================================================================
void EllipsoidShape::setDiameters(const Eigen::Vector3s& diameters)
{
  assert(diameters[0] > 0.0);
  assert(diameters[1] > 0.0);
  assert(diameters[2] > 0.0);

  mDiameters = diameters;

  mIsBoundingBoxDirty = true;
  mIsVolumeDirty = true;

  incrementVersion();
}

//==============================================================================
const Eigen::Vector3s& EllipsoidShape::getDiameters() const
{
  return mDiameters;
}

//==============================================================================
void EllipsoidShape::setRadii(const Eigen::Vector3s& radii)
{
  mDiameters = radii * 2.0;

  mIsBoundingBoxDirty = true;
  mIsVolumeDirty = true;

  incrementVersion();
}

//==============================================================================
const Eigen::Vector3s EllipsoidShape::getRadii() const
{
  return mDiameters / 2.0;
}

//==============================================================================
s_t EllipsoidShape::computeVolume(const Eigen::Vector3s& diameters)
{
  // 4/3* Pi* a/2* b/2* c/2
  return math::constantsd::pi() * diameters[0] * diameters[1] * diameters[2]
         / 6.0;
}

//==============================================================================
Eigen::Matrix3s EllipsoidShape::computeInertia(
    const Eigen::Vector3s& diameters, s_t mass)
{
  Eigen::Matrix3s inertia = Eigen::Matrix3s::Identity();

  const auto coeff = mass / 20.0;
  const auto AA = pow(diameters[0], 2);
  const auto BB = pow(diameters[1], 2);
  const auto CC = pow(diameters[2], 2);

  inertia(0, 0) = coeff * (BB + CC);
  inertia(1, 1) = coeff * (AA + CC);
  inertia(2, 2) = coeff * (AA + BB);

  return inertia;
}

//==============================================================================
Eigen::Matrix3s EllipsoidShape::computeInertia(s_t mass) const
{
  return computeInertia(mDiameters, mass);
}

//==============================================================================
bool EllipsoidShape::isSphere() const
{
  if (mDiameters[0] == mDiameters[1] && mDiameters[1] == mDiameters[2])
    return true;
  else
    return false;
}

//==============================================================================
/// Allow us to clone shapes, to avoid race conditions when scaling shapes
/// belonging to different skeletons
ShapePtr EllipsoidShape::clone() const
{
  return std::make_shared<EllipsoidShape>(mDiameters);
}

//==============================================================================
void EllipsoidShape::updateBoundingBox() const
{
  mBoundingBox.setMin(-mDiameters * 0.5);
  mBoundingBox.setMax(mDiameters * 0.5);
  mIsBoundingBoxDirty = false;
}

//==============================================================================
void EllipsoidShape::updateVolume() const
{
  mVolume = computeVolume(mDiameters);
  mIsVolumeDirty = false;
}

} // namespace dynamics
} // namespace dart
