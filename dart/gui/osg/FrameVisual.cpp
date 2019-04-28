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

#include "dart/gui/osg/FrameVisual.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/SimpleFrame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/gui/osg/Utils.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {
namespace gui {
namespace osg {

//==============================================================================
FrameVisual::FrameVisual(dynamics::WeakBodyNodePtr frame)
  : mFrame(std::move(frame))
{
  initialize();
}

//==============================================================================
void FrameVisual::display(bool display)
{
  if (mDisplay == display)
    return;

  mDisplay = display;

  if (mDisplay)
    addChild(mGeode);
  else
    removeChild(mGeode);
}

//==============================================================================
bool FrameVisual::isDisplayed() const
{
  return mDisplay;
}

//==============================================================================
static void setAxisColor(
    ::osg::Vec4Array* axisColors,
    ::osg::Geometry* geom,
    std::size_t index,
    const Eigen::Vector4d& color)
{
  assert(axisColors->size() == 3);
  axisColors->at(index) = ::osg::Vec4(
      static_cast<float>(color[0]),
      static_cast<float>(color[1]),
      static_cast<float>(color[2]),
      static_cast<float>(color[3]));
  geom->setColorArray(axisColors, ::osg::Array::BIND_PER_PRIMITIVE_SET);
}

//==============================================================================
static Eigen::Vector4d getAxisColor(
    ::osg::Vec4Array* axisColors, std::size_t index)
{
  const ::osg::Vec4& c = axisColors->at(index);
  return Eigen::Vector4f(c[0], c[1], c[2], c[3]).cast<double>();
}

//==============================================================================
void FrameVisual::seAxisXColor(const Eigen::Vector4d& color)
{
  setAxisColor(mAxisColors, mGeom, 0, color);
}

//==============================================================================
Eigen::Vector4d FrameVisual::getAxisXColor() const
{
  return getAxisColor(mAxisColors, 0);
}

//==============================================================================
void FrameVisual::setAxisYColor(const Eigen::Vector4d& color)
{
  setAxisColor(mAxisColors, mGeom, 1, color);
}

//==============================================================================
Eigen::Vector4d FrameVisual::getAxisYColor() const
{
  return getAxisColor(mAxisColors, 1);
}

//==============================================================================
void FrameVisual::setAxisZColor(const Eigen::Vector4d& color)
{
  setAxisColor(mAxisColors, mGeom, 2, color);
}

//==============================================================================
Eigen::Vector4d FrameVisual::getAxisZColor() const
{
  return getAxisColor(mAxisColors, 2);
}

//==============================================================================
void FrameVisual::updateGeometry()
{
//  ::osg::Vec3Array* minorLineVertices;
}

//==============================================================================
void FrameVisual::refresh()
{
  std::cout << "haha" << std::endl;
  if (!mNeedUpdate)
    return;

  if (mDisplay)
  {
    updateGeometry();
    //    mGeom->setVertexArray(mAxisYVertices);
    //    mGeom->setPrimitiveSet(0, mAxisXFaces);
    //    mGeom->setPrimitiveSet(1, mAxisYFaces);
    //    mGeom->setPrimitiveSet(2, mAxisZFaces);

    //    mGeom->setColorArray(mAxisXColor);
  }

  mNeedUpdate = false;
}

//==============================================================================
void FrameVisual::initialize()
{
  mNeedUpdate = true;

  mDisplay = true;

  mGeode = new ::osg::Geode;
  mGeode->getOrCreateStateSet()->setMode(
      GL_LIGHTING, ::osg::StateAttribute::OFF);
  addChild(mGeode);

  mGeom = new ::osg::Geometry;
  mGeode->addDrawable(mGeom);

  mAxisXVertices = new ::osg::Vec3Array;
  mGeom->setVertexArray(mAxisXVertices);
  mGeom->setDataVariance(::osg::Object::STATIC);

  mAxisYVertices = new ::osg::Vec3Array;
  mGeom->setVertexArray(mAxisYVertices);
  mGeom->setDataVariance(::osg::Object::STATIC);

  mAxisZVertices = new ::osg::Vec3Array;
  mGeom->setVertexArray(mAxisZVertices);
  mGeom->setDataVariance(::osg::Object::STATIC);

  // Set grid color
  static const ::osg::Vec4 axisXColor(0.9f, 0.1f, 0.1f, 1.0f);
  static const ::osg::Vec4 axisYColor(0.1f, 0.9f, 0.1f, 1.0f);
  static const ::osg::Vec4 axisZColor(0.1f, 0.1f, 0.9f, 1.0f);

  mAxisColors = new ::osg::Vec4Array;
  mAxisColors->resize(3);
  mAxisColors->at(0) = axisXColor;
  mAxisColors->at(1) = axisYColor;
  mAxisColors->at(2) = axisZColor;
  mGeom->setColorArray(mAxisColors);
  mGeom->setColorBinding(::osg::Geometry::BIND_PER_PRIMITIVE_SET);

  mAxisXFaces = new ::osg::DrawElementsUInt(::osg::PrimitiveSet::LINES, 0);
  mAxisYFaces = new ::osg::DrawElementsUInt(::osg::PrimitiveSet::LINES, 0);
  mAxisZFaces = new ::osg::DrawElementsUInt(::osg::PrimitiveSet::LINES, 0);
  mGeom->addPrimitiveSet(mAxisXFaces);
  mGeom->addPrimitiveSet(mAxisYFaces);
  mGeom->addPrimitiveSet(mAxisZFaces);
}

} // namespace osg
} // namespace gui
} // namespace dart
