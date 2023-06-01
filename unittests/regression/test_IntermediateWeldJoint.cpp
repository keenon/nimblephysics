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

#include <GradientTestUtils.hpp>
#include <TestHelpers.hpp>
#include <dart/server/GUIWebsocketServer.hpp>
#include <dart/utils/UniversalLoader.hpp>
#include <gtest/gtest.h>

using namespace dart;

#define ALL_TESTS

//==============================================================================
TEST(IntermediateWeldJoint, SIMPLE)
{
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  T.translation() = Eigen::Vector3s::UnitX();

  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto rootPair = skel->createJointAndBodyNodePair<dynamics::RevoluteJoint>();
  auto weldPair
      = rootPair.second->createChildJointAndBodyNodePair<dynamics::WeldJoint>();
  weldPair.first->setTransformFromChildBodyNode(T);
  auto tailPair
      = weldPair.second
            ->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>();
  tailPair.first->setTransformFromChildBodyNode(T);

  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);

  Eigen::VectorXs worldVel = Eigen::VectorXs::Random(world->getNumDofs());

  Eigen::MatrixXs M = skel->getMassMatrix();

  Eigen::MatrixXs bruteMinv = M.inverse();

  Eigen::MatrixXs Minv = skel->getInvMassMatrix();

  EXPECT_TRUE(verifyAnalyticalA_c(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world, true));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

//==============================================================================
#ifdef ALL_TESTS
TEST(IntermediateWeldJoint, AMASS_FD)
{
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/test/amass_fd/ground.skel");
  std::shared_ptr<dynamics::Skeleton> amass
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/test/amass_fd/amass_nimble.urdf");

  world->setState(Eigen::VectorXs::Zero(world->getStateSize()));

  // Recompute all inertia matrices

  /*
  for (int i = 0; i < amass->getNumBodyNodes(); i++)
  {
    auto bodyNode = amass->getBodyNode(i);
    for (int j = 0; j < bodyNode->getNumShapeNodes(); j++)
    {
      auto shapeNode = bodyNode->getShapeNode(j);
      auto shape = shapeNode->getShape();
      Eigen::Matrix3s inertia = shape->computeInertia(bodyNode->getMass());
      bodyNode->setInertia(dart::dynamics::Inertia(
          bodyNode->getMass(), Eigen::Vector3s::Zero(), inertia));
      // if (!shapeNode->hasVisualAspect())
      // {
      //   shapeNode->createVisualAspect();
      //   shapeNode->getVisualAspect()->setRGBA(
      //       Eigen::Vector4s(0.5, 0.5, 0.5, 1.0));
      //   shapeNode->getVisualAspect()->show();
      // }
    }
  }
  */

  VectorXs worldVel = world->getVelocities();

  /*
  // Make all the colliders visible

  for (int i = 0; i < amass->getNumBodyNodes(); i++)
  {
    auto bodyNode = amass->getBodyNode(i);
    for (int j = 0; j < bodyNode->getNumShapeNodes(); j++)
    {
      auto shapeNode = bodyNode->getShapeNode(j);
      if (!shapeNode->hasVisualAspect())
      {
        shapeNode->createVisualAspect();
        shapeNode->getVisualAspect()->setRGBA(
            Eigen::Vector4s(0.5, 0.5, 0.5, 1.0));
        shapeNode->getVisualAspect()->show();
      }
    }
  }

  // Run the GUI server

  dart::server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  server.blockWhileServing();
  */

  // Test the classic formulation
  EXPECT_TRUE(verifyAnalyticalA_c(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}
#endif
