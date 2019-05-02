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

#include "BoxStackingProject.hpp"

#include <sstream>

#include "dart/utils/utils.hpp"

namespace dart {
namespace examples {

//==============================================================================
dynamics::SkeletonPtr createBox(const Eigen::Vector3d& position)
{
  dynamics::SkeletonPtr boxSkel = dynamics::Skeleton::create("box");

  // Give the floor a body
  dynamics::BodyNodePtr boxBody
      = boxSkel->createJointAndBodyNodePair<dynamics::FreeJoint>(nullptr)
            .second;

  // Give the body a shape
  double boxWidth = 1.0;
  double boxDepth = 1.0;
  double boxHeight = 0.5;
  auto boxShape = std::make_shared<dynamics::BoxShape>(
      Eigen::Vector3d(boxWidth, boxDepth, boxHeight));
  dynamics::ShapeNode* shapeNode = boxBody->createShapeNodeWith<
      dynamics::VisualAspect,
      dynamics::CollisionAspect,
      dynamics::DynamicsAspect>(boxShape);
  shapeNode->getVisualAspect()->setColor(
      dart::math::Random::uniform<Eigen::Vector3d>(0.0, 1.0));

  // Put the body into position
  Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
  tf.translation() = position;
  boxBody->getParentJoint()->setTransformFromParentBodyNode(tf);

  return boxSkel;
}

//==============================================================================
std::vector<dynamics::SkeletonPtr> createBoxStack(
    std::size_t numBoxes, double heightFromGround = 0.5)
{
  std::vector<dynamics::SkeletonPtr> boxSkels(numBoxes);

  for (auto i = 0u; i < numBoxes; ++i)
    boxSkels[i] = createBox(
        Eigen::Vector3d(0.0, 0.0, heightFromGround + 0.25 + i * 0.5));

  return boxSkels;
}

//==============================================================================
dynamics::SkeletonPtr createFloor()
{
  dynamics::SkeletonPtr floor = dynamics::Skeleton::create("floor");

  // Give the floor a body
  dynamics::BodyNodePtr body
      = floor->createJointAndBodyNodePair<dynamics::WeldJoint>(nullptr).second;

  // Give the body a shape
  double floorWidth = 10.0;
  double floorHeight = 0.01;
  auto box = std::make_shared<dynamics::BoxShape>(
      Eigen::Vector3d(floorWidth, floorWidth, floorHeight));
  dynamics::ShapeNode* shapeNode = body->createShapeNodeWith<
      dynamics::VisualAspect,
      dynamics::CollisionAspect,
      dynamics::DynamicsAspect>(box);
  shapeNode->getVisualAspect()->setColor(dart::Color::LightGray());

  // Put the body into position
  Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
  tf.translation() = Eigen::Vector3d(0.0, 0.0, -floorHeight / 2.0);
  body->getParentJoint()->setTransformFromParentBodyNode(tf);

  return floor;
}

//==============================================================================
void BoxStackingProject::initialize()
{
  // Create and initialize the world
  mWorld = simulation::World::create("Box Stacking World");

  mWorld->addSkeleton(createFloor());

  auto boxSkels = createBoxStack(5);
  for (const auto& boxSkel : boxSkels)
    mWorld->addSkeleton(boxSkel);
}

//==============================================================================
void BoxStackingProject::reset()
{
  // Do nothing
}

//==============================================================================
void BoxStackingProject::prestep()
{
  mWorld->step();
}

//==============================================================================
void BoxStackingProject::finalize()
{
  // Do nothing
}

//==============================================================================
void BoxStackingProject::render()
{
  // Do nothing
}

//==============================================================================
std::string BoxStackingProject::getNameStatic()
{
  return "Box Stacking";
}

//==============================================================================
std::string BoxStackingProject::getName() const
{
  return getNameStatic();
}

//==============================================================================
std::string BoxStackingProject::getUsage() const
{
  std::stringstream ss;
  // TODO(JS): Outdated
  ss << "\n";

  return ss.str();
}

//==============================================================================
std::string BoxStackingProject::getDiscriptionStatic()
{
  return "No descriptions";
}

} // namespace examples
} // namespace dart
