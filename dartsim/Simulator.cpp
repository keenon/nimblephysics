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
 *   * This code incorporates portions of Open Dynamics Engine
 *     (Copyright (c) 2001-2004, Russell L. Smith. All rights
 *     reserved.) and portions of FCL (Copyright (c) 2011, Willow
 *     Garage, Inc. All rights reserved.), which were released under
 *     the same BSD license as below
 *
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

#include "Simulator.hpp"

#include <dart/utils/urdf/urdf.hpp>
#include "widgets/MainMenuWidget.hpp"
#include "widgets/PropertyGridWidget.hpp"
#include "widgets/TestWidget.hpp"

using namespace dart::common;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::utils;
using namespace dart::math;

namespace dart {
namespace simulator {

//==============================================================================
class CustomWorldNode : public dart::gui::osg::WorldNode
{
public:
  CustomWorldNode(const dart::simulation::WorldPtr& world = nullptr)
    : dart::gui::osg::WorldNode(world)
  {
    // Set up the customized WorldNode
  }

  void customPreRefresh()
  {
    // Use this function to execute custom code before each time that the
    // window is rendered. This function can be deleted if it does not need
    // to be used.
  }

  void customPostRefresh()
  {
    // Use this function to execute custom code after each time that the
    // window is rendered. This function can be deleted if it does not need
    // to be used.
  }

  void customPreStep()
  {
    // Use this function to execute custom code before each simulation time
    // step is performed. This function can be deleted if it does not need
    // to be used.
  }

  void customPostStep()
  {
    // Use this function to execute custom code after each simulation time
    // step is performed. This function can be deleted if it does not need
    // to be used.
  }
};

//==============================================================================
class CustomEventHandler : public osgGA::GUIEventHandler
{
public:
  CustomEventHandler(/*Pass in any necessary arguments*/)
  {
    // Set up the customized event handler
  }

  virtual bool handle(
      const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&) override
  {
    if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
    {
      if (ea.getKey() == 'q')
      {
        std::cout << "Lowercase q pressed" << std::endl;
        return true;
      }
      else if (ea.getKey() == 'Q')
      {
        std::cout << "Capital Q pressed" << std::endl;
        return true;
      }
      else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_Left)
      {
        std::cout << "Left arrow key pressed" << std::endl;
        return true;
      }
      else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_Right)
      {
        std::cout << "Right arrow key pressed" << std::endl;
        return true;
      }
    }
    else if (ea.getEventType() == osgGA::GUIEventAdapter::KEYUP)
    {
      if (ea.getKey() == 'q')
      {
        std::cout << "Lowercase q released" << std::endl;
        return true;
      }
      else if (ea.getKey() == 'Q')
      {
        std::cout << "Capital Q released" << std::endl;
        return true;
      }
      else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_Left)
      {
        std::cout << "Left arrow key released" << std::endl;
        return true;
      }
      else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_Right)
      {
        std::cout << "Right arrow key released" << std::endl;
        return true;
      }
    }

    // The return value should be 'true' if the input has been fully handled
    // and should not be visible to any remaining event handlers. It should be
    // false if the input has not been fully handled and should be viewed by
    // any remaining event handlers.
    return false;
  }
};

//==============================================================================
SkeletonPtr createAtlas()
{
  // Parse in the atlas model
  DartLoader urdf;
  SkeletonPtr atlas
      = urdf.parseSkeleton("dart://sample/sdf/atlas/atlas_v3_no_head.urdf");

  // Add a box to the root node to make it easier to click and drag
  double scale = 0.25;
  ShapePtr boxShape
      = std::make_shared<BoxShape>(scale * Eigen::Vector3d(1.0, 1.0, 0.5));

  Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());
  tf.translation() = Eigen::Vector3d(0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));

  auto shapeNode
      = atlas->getBodyNode(0)->createShapeNodeWith<VisualAspect>(boxShape);
  shapeNode->getVisualAspect()->setColor(dart::Color::Black());
  shapeNode->setRelativeTransform(tf);

  return atlas;
}

//==============================================================================
Simulator::Simulator()
{
  // Create a world
  mWorld = simulation::World::create();

  // Add a target object to the world
  //  auto target
  //      =
  //      std::make_shared<gui::osg::InteractiveFrame>(dynamics::Frame::World());
  //  mWorld->addSimpleFrame(target);

  dynamics::SkeletonPtr atlas = createAtlas();
  mWorld->addSkeleton(atlas);

  // Wrap a WorldNode around it
  ::osg::ref_ptr<CustomWorldNode> node = new CustomWorldNode(mWorld);

  // Create a Viewer and set it up with the WorldNode
  mViewer.addWorldNode(node);

  // Add control widgets
  mViewer.getImGuiHandler()->addWidget(
      std::make_shared<TestWidget>(&mViewer, mWorld));
  mViewer.getImGuiHandler()->addWidget(std::make_shared<MainMenuWidget>());
  auto propertyGridWidget = std::make_shared<PropertyGridWidget>();
  propertyGridWidget->setWorld(mWorld);
  mViewer.getImGuiHandler()->addWidget(propertyGridWidget);

  // Active the drag-and-drop feature for the target
  //  mViewer.enableDragAndDrop(target.get());

  // Pass in the custom event handler
  mViewer.addEventHandler(new CustomEventHandler);

  // Set up the window to be 640x480
  mViewer.setUpViewInWindow(0, 0, 1280, 960);

  // Adjust the viewpoint of the Viewer
  mViewer.getCameraManipulator()->setHomePosition(
      ::osg::Vec3(2.57f, 3.14f, 1.64f),
      ::osg::Vec3(0.00f, 0.00f, 0.00f),
      ::osg::Vec3(-0.24f, -0.25f, 0.94f));

  // We need to re-dirty the CameraManipulator by passing it into the viewer
  // again, so that the viewer knows to update its HomePosition setting
  mViewer.setCameraManipulator(mViewer.getCameraManipulator());
}

//==============================================================================
void Simulator::run()
{
  mViewer.run();
}

} // namespace simulator
} // namespace dart
