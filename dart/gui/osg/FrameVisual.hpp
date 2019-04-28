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

#ifndef DART_GUI_OSG_FRAMEVISUAL_HPP_
#define DART_GUI_OSG_FRAMEVISUAL_HPP_

#include <osg/Geode>
#include <osg/LineWidth>

#include "dart/dynamics/Frame.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/gui/osg/ShapeFrameNode.hpp"
#include "dart/gui/osg/Viewer.hpp"

namespace dart {
namespace gui {
namespace osg {

/// Attach this to a Viewer in order to visualize grid.
class FrameVisual : public ViewerAttachment
{
public:
  /// Default constructor
  explicit FrameVisual(dynamics::WeakBodyNodePtr frame);

  /// Displays the support polygon
  void display(bool display);

  /// Returns true if the support polygon is being displayed
  bool isDisplayed() const;

  void seAxisXColor(const Eigen::Vector4d& color);

  Eigen::Vector4d getAxisXColor() const;

  void setAxisYColor(const Eigen::Vector4d& color);

  Eigen::Vector4d getAxisYColor() const;

  void setAxisZColor(const Eigen::Vector4d& color);

  Eigen::Vector4d getAxisZColor() const;

  /// Updates the support polygon visual
  void refresh() override final;

protected:
  /// Initializes the memory used by this visual
  void initialize();

  void updateGeometry();

  dynamics::WeakBodyNodePtr mFrame;

  /// Whether to display the grid
  bool mDisplay;

  std::size_t mResolution{16};

  /// Color for axis lines
  ::osg::ref_ptr<::osg::Vec4Array> mAxisColors;

  /// Geode to hold the grid
  ::osg::ref_ptr<::osg::Geode> mGeode;

  /// Geometry to describe axis lines
  ::osg::ref_ptr<::osg::Geometry> mGeom;

  /// Vertices of axis lines
  ::osg::ref_ptr<::osg::Vec3Array> mAxisXVertices;

  /// Vertices of major lines
  ::osg::ref_ptr<::osg::Vec3Array> mAxisYVertices;

  /// Vertices of minor lines
  ::osg::ref_ptr<::osg::Vec3Array> mAxisZVertices;

  /// Faces of the first axis positive line
  ::osg::ref_ptr<::osg::DrawElementsUInt> mAxisXFaces;

  /// Faces of the first axis negative line
  ::osg::ref_ptr<::osg::DrawElementsUInt> mAxisYFaces;

  /// Faces of the second axis positive line
  ::osg::ref_ptr<::osg::DrawElementsUInt> mAxisZFaces;

  /// Dirty flag to notify this grid needs to be updated
  bool mNeedUpdate;
};

} // namespace osg
} // namespace gui
} // namespace dart

#endif // DART_GUI_OSG_FRAMEVISUAL_HPP_
