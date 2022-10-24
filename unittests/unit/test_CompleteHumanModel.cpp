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

#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dart.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/EllipsoidJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/ScapulathoracicJoint.hpp"
#include "dart/dynamics/detail/BodyNodePtr.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/utils.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace utils;

//==============================================================================
TEST(CompleteHumanModel, LOAD_SHOULDER_OPENSIM)
{
  auto osim = biomechanics::OpenSimParser::parseOsim(
      "dart://sample/osim/CompleteHumanModel/CompleteHumanModel.osim");
  for (int i = 0; i < osim.skeleton->getBodyNode("torso")->getNumShapeNodes();
       i++)
  {
    auto* shapeNode = osim.skeleton->getBodyNode("torso")->getShapeNode(i);
    auto shape = shapeNode->getShape();
    std::cout << "Found shape: " << shapeNode->getName() << " - "
              << shape->getType() << std::endl;
  }
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerList;
  for (auto pair : osim.markersMap)
  {
    markerList.push_back(pair.second);
  }
  // osim.skeleton->getBodyNode("thorax")->setScale(
  //     Eigen::Vector3s(0.8, 1.1, 1.4));

  bool markersVerified
      = verifySkeletonMarkerJacobians(osim.skeleton, markerList);
  EXPECT_TRUE(markersVerified);
  if (!markersVerified)
  {
    return;
  }
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim.skeleton);
  bool jacobiansVerified = verifyFeatherstoneJacobians(world);
  EXPECT_TRUE(jacobiansVerified);
  if (!jacobiansVerified)
  {
    return;
  }

  server::GUIRecording server;
  server.setFramesPerSecond(20);
  server.renderBasis();
  server.renderSkeleton(osim.skeleton);

  auto locations = osim.skeleton->getMarkerMapWorldPositions(osim.markersMap);
  for (auto& pair : locations)
  {
    server.createSphere(pair.first, 0.01, pair.second);
    server.setObjectTooltip(pair.first, pair.first);
  }

  /*
  auto sprinter = biomechanics::OpenSimParser::parseOsim(
      "dart://sample/osim/Sprinter/sprinter.osim");

  server.renderSkeleton(sprinter.skeleton);
  auto sprinterLocation
      = sprinter.skeleton->getMarkerMapWorldPositions(sprinter.markersMap);
  for (auto& pair : sprinterLocation)
  {
    server.createSphere(
        pair.first, 0.01, pair.second, Eigen::Vector4s(1, 0, 0, 1));
    server.setObjectTooltip(pair.first, pair.first);
  }
  */

  // TODO: Transfer markers to the thorax / scapula / lumbar / neck, instead of
  // the torso

  /*
  for (int i = 0; i < osim.skeleton->getNumJoints(); i++)
  {
    auto* joint = osim.skeleton->getJoint(i);
    if (joint->getType() == EllipsoidJoint::getStaticType())
    {
      dynamics::EllipsoidJoint* scap
          = static_cast<dynamics::EllipsoidJoint*>(joint);
      Eigen::Vector3s radii
          = scap->getEllipsoidRadii().cwiseProduct(joint->getParentScale());
      Eigen::Isometry3s worldT = scap->getParentBodyNode()->getWorldTransform()
                                 * scap->getTransformFromParentBodyNode();
      server.createSphere(
          "joint" + std::to_string(i),
          radii,
          worldT.translation(),
          Eigen::Vector4s(1, 0, 0, 0.3));
      server.setObjectRotation(
          "joint" + std::to_string(i), math::matrixToEulerXYZ(worldT.linear()));
      server.renderBasis(
          0.1,
          "joint" + std::to_string(i),
          worldT.translation(),
          math::matrixToEulerXYZ(worldT.linear()));
    }
    if (joint->getType() == EulerJoint::getStaticType())
    {
      Eigen::Isometry3s worldT = joint->getParentBodyNode()->getWorldTransform()
                                 * joint->getTransformFromParentBodyNode();
      server.renderBasis(
          0.1,
          "joint" + std::to_string(i),
          worldT.translation(),
          math::matrixToEulerXYZ(worldT.linear()));
    }
  }
  */

  server.saveFrame();

  /*
  //////////////////////////////////////////////////////////////
  Eigen::Vector3s originalScale
      = osim.skeleton->getBodyNode("thorax")->getScale();
  osim.skeleton->getJoint("scapulothoracic_r")
      ->setPositions(osim.skeleton->getJoint("scapulothoracic_r")
                         ->getPositionLowerLimits());
  osim.skeleton->getJoint("scapulothoracic_l")
      ->setPositions(osim.skeleton->getJoint("scapulothoracic_l")
                         ->getPositionUpperLimits());
  for (int axis = 0; axis < 3; axis++)
  {
    for (s_t scale = 0.8; scale < 1.2; scale += 0.01)
    {
      Eigen::Vector3s newScale = originalScale;
      newScale(axis) = scale;
      osim.skeleton->getBodyNode("thorax")->setScale(newScale);

      server.renderSkeleton(osim.skeleton);
      for (int i = 0; i < osim.skeleton->getNumJoints(); i++)
      {
        auto* joint = osim.skeleton->getJoint(i);
        if (joint->getType() == EllipsoidJoint::getStaticType())
        {
          dynamics::EllipsoidJoint* scap
              = static_cast<dynamics::EllipsoidJoint*>(joint);
          Eigen::Vector3s radii
              = scap->getEllipsoidRadii().cwiseProduct(joint->getParentScale());
          Eigen::Isometry3s worldT
              = scap->getParentBodyNode()->getWorldTransform()
                * scap->getTransformFromParentBodyNode();
          server.createSphere(
              "joint" + std::to_string(i),
              radii,
              worldT.translation(),
              Eigen::Vector4s(1, 0, 0, 0.3));
          server.setObjectRotation(
              "joint" + std::to_string(i),
              math::matrixToEulerXYZ(worldT.linear()));
          server.renderBasis(
              0.1,
              "joint" + std::to_string(i),
              worldT.translation(),
              math::matrixToEulerXYZ(worldT.linear()));
        }
        if (joint->getType() == EulerJoint::getStaticType())
        {
          Eigen::Isometry3s worldT
              = joint->getParentBodyNode()->getWorldTransform()
                * joint->getTransformFromParentBodyNode();
          server.renderBasis(
              0.1,
              "joint" + std::to_string(i),
              worldT.translation(),
              math::matrixToEulerXYZ(worldT.linear()));
        }
      }
      server.saveFrame();
    }
  }
  //////////////////////////////////////////////////////////////
  */

  /*
  std::vector<std::string> names;
  names.push_back("scapula_abduction_r");
  names.push_back("scapula_elevation_r");
  names.push_back("scapula_upward_rot_r");
  names.push_back("scapula_abduction_l");
  names.push_back("scapula_elevation_l");
  names.push_back("scapula_upward_rot_l");

  // names.push_back("lumbar_bending");
  // names.push_back("lumbar_extension");
  // names.push_back("lumbar_twist");
  // names.push_back("thorax_bending");
  // names.push_back("thorax_extension");
  // names.push_back("thorax_twist");
  // names.push_back("head_bending");
  // names.push_back("head_extension");
  // names.push_back("head_twist");

  for (auto& name : names)
  {
    const int steps = 50;
    for (int i = 0; i < steps; i++)
    {
      auto* dof = osim.skeleton->getDof(name);
      dof->setPosition(
          ((s_t)i / steps)
              * (dof->getPositionUpperLimit() - dof->getPositionLowerLimit())
          + dof->getPositionLowerLimit());
      server.renderSkeleton(osim.skeleton);
      auto locations
          = osim.skeleton->getMarkerMapWorldPositions(osim.markersMap);
      for (auto& pair : locations)
      {
        server.setObjectPosition(pair.first, pair.second);
      }
      server.saveFrame();
    }
    osim.skeleton->getDof(name)->setPosition(
        osim.skeleton->getDof(name)->getInitialPosition());
  }

  */
  server.writeFramesJson("../../../javascript/src/data/movement2.bin");
}