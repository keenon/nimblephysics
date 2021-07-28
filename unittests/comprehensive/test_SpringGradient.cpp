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

#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

#ifdef DART_USE_ARBITRARY_PRECISION
#include "mpreal.h"
#endif

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

// This is the margin so that finite-differencing over position doesn't break
// contacts
const s_t CONTACT_MARGIN = 1e-5;

/******************************************************************************

This test sets up a configuration that looks like this:

          Force
            |
            v
          +---+
Force --> |   |
          +---+
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground has configurable friction in this setup.

*/


void testBlockWithFrictionCoeff(s_t frictionCoeff, s_t mass)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = box->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setSpringStiffness(0,1);
  boxJoint->setSpringStiffness(1,2);
  boxJoint->setRestPosition(0,0.2);
  boxJoint->setRestPosition(1,0.1);


  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3s floorPosition = Eigen::Isometry3s::Identity();
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - CONTACT_MARGIN), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);  


  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXs timestepVel = box->getVelocities();
  VectorXs timestepWorldVel = world->getVelocities();

  VectorXs worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_1_MASS)
{
  testBlockWithFrictionCoeff(0, 1);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_2_MASS)
{
  testBlockWithFrictionCoeff(0, 2);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_4_MASS)
{
  testBlockWithFrictionCoeff(0, 4);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_STATIC_FRICTION)
{
  testBlockWithFrictionCoeff(1e7, 1);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_SLIPPING_FRICTION)
{
  testBlockWithFrictionCoeff(0.5, 1);
}

#endif


/******************************************************************************

This test sets up a configuration that looks like this:

      Large Velocity
            |
            v
          +---+
Force --> |   |
          +---+

There's a box with two DOFs, x and y axis, with a force driving it down. 
The ground has been removed to create non contact condition.

*/
void testFreeBlockPosGradients(s_t frictionCoeff, s_t spring_stiff_x,s_t spring_stiff_y)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s::Zero());

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = box->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setSpringStiffness(0,spring_stiff_x);
  boxJoint->setSpringStiffness(1,spring_stiff_y);
  boxJoint->setRestPosition(0,0.05);
  boxJoint->setRestPosition(1,0.01);

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3s(1, 0, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(1.0);
  // Set the 1th dof index to -1.0
  box->setVelocity(1, -1.0);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////
  // Test the forward behavior against 
  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXs timestepVel = box->getVelocities();
  VectorXs timestepWorldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyAnalyticalJacobians(world,true));
  EXPECT_TRUE(verifyVelGradients(world, timestepWorldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS

TEST(GRADIENTS, POS_BLOCK_FREE_NO_FRICTION_1_0_DAMP)
{
  testFreeBlockPosGradients(0, 1.0, 0.0);
}


TEST(GRADIENTS, POS_BLOCK_FREE_NO_FRICTION_0_1_DAMP)
{
  testFreeBlockPosGradients(0, 0.0, 1.0);
}

TEST(GRADIENTS, POS_BLOCK_FREE_NO_FRICTION_1_1_DAMP)
{
  testFreeBlockPosGradients(0, 1.0, 1.0);
}
#endif

void testRotateBlockPosGradients(s_t frictionCoeff, s_t spring_stiff)
{
  // World
  WorldPtr world = World::create();
  //world->setGravity(Eigen::Vector3s::Zero());
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<RevoluteJoint*, BodyNode*> pair
      = box->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  RevoluteJoint* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  //boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  //boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setSpringStiffness(0,spring_stiff);
  boxJoint->setRestPosition(0,0.2);

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  //boxBody->addExtForce(Eigen::Vector3s(1, 0, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(1.0);
  // Set the 1th dof index to -1.0
  box->setVelocity(0, -1.0);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////
  // Test the forward behavior against 
  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  //world->step();
  VectorXs timestepVel = box->getVelocities();
  VectorXs timestepWorldVel = world->getVelocities();
  // Test the classic formulation
  //EXPECT_TRUE(verifyNextV(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world,true));
  EXPECT_TRUE(verifyVelGradients(world, timestepWorldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, POS_BLOCK_ROT_NO_FRICTION_1_0_DAMP)
{
  testRotateBlockPosGradients(0, 1.0);
}

#endif

/******************************************************************************

This test sets up a configuration that looks something like this:

                        |
                  _____ |
                O _____O| < Fixed wall
              / /       |
             / /        |
            / /         |
            O           |
           | |          |
           | |          |
           | |          |
            O
            ^             ^
      Rotating base

It's a robot arm, with a rotating base, with "numLinks" links and
"rotationDegree" position at each link. There's also a fixed plane at the end
of the robot arm that it intersects with.
*/
void testRobotArm(
    std::size_t numLinks, s_t rotationRadians, int attachPoint = -1, s_t spring_stiff = 1.0)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));
  world->setContactClippingDepth(1.);

  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));

  std::vector<BodyNode*> nodes;

  for (std::size_t i = 0; i < numLinks; i++)
  {
    RevoluteJoint::Properties jointProps;
    jointProps.mName = "revolute_" + std::to_string(i);
    BodyNode::Properties bodyProps;
    bodyProps.mName = "arm_" + std::to_string(i);
    std::pair<RevoluteJoint*, BodyNode*> jointPair
        = arm->createJointAndBodyNodePair<RevoluteJoint>(
            parent, jointProps, bodyProps);
    nodes.push_back(jointPair.second);
    if (parent != nullptr)
    {
      Eigen::Isometry3s armOffset = Eigen::Isometry3s::Identity();
      armOffset.translation() = Eigen::Vector3s(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
      jointPair.first->setSpringStiffness(0,spring_stiff);
      jointPair.first->setRestPosition(0,1);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
    if ((attachPoint == -1 && i < numLinks - 1)
        || (attachPoint != -1 && i != attachPoint))
    {
      // ShapeNode* visual =
      parent->createShapeNodeWith<VisualAspect>(boxShape);
    }
  }

  if (attachPoint != -1)
  {
    parent = nodes[attachPoint];
  }

  std::shared_ptr<BoxShape> endShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0) * 1. / sqrt(2.0)));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);

  arm->setPositions(Eigen::VectorXs::Ones(arm->getNumDofs()) * rotationRadians);
  world->addSkeleton(arm);

  Eigen::Isometry3s worldTransform = parent->getWorldTransform();
  Eigen::Matrix3s rotation
      = math::eulerXYZToMatrix(Eigen::Vector3s(0, 45, -45) * 3.1415 / 180)
        * worldTransform.linear().transpose();
  endNode->setRelativeRotation(rotation);

  SkeletonPtr wall = Skeleton::create("wall");
  std::pair<WeldJoint*, BodyNode*> jointPair
      = wall->createJointAndBodyNodePair<WeldJoint>(nullptr);
  std::shared_ptr<BoxShape> wallShape(
      new BoxShape(Eigen::Vector3s(1.0, 10.0, 10.0)));
  
  jointPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      wallShape);
  world->addSkeleton(wall);

  Eigen::Isometry3s wallLocalOffset = Eigen::Isometry3s::Identity();
  wallLocalOffset.translation() = parent->getWorldTransform().translation()
                                  + Eigen::Vector3s(-(1.0 - 1e-2), 0.0, 0.0);
  jointPair.first->setTransformFromParentBodyNode(wallLocalOffset);


  if (numLinks == 5)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * -0.05);
  }
  if (numLinks == 6 || numLinks == 3)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * 0.05);
  }


  Eigen::VectorXs worldVel = world->getVelocities();

  EXPECT_TRUE(verifyNextV(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));

}

#ifdef ALL_TESTS

TEST(GRADIENTS, ARM_3_LINK_30_DEG)
{
  testRobotArm(3, 30.0 / 180 * 3.1415);
}


TEST(GRADIENTS, ARM_5_LINK_40_DEG)
{
  // This one penetrates much more deeply than the others
  testRobotArm(5, 40.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_6_LINK_15_DEG)
{
  testRobotArm(6, 15.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_3_LINK_30_DEG_MIDDLE_ATTACH)
{
  testRobotArm(3, 30.0 / 180 * 3.1415, 1);
}

#endif

void testRobotArmNoContact(
    std::size_t numLinks, s_t rotationRadians, int attachPoint = -1, s_t spring_stiff = 1.0)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));
  world->setContactClippingDepth(1.);

  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));

  std::vector<BodyNode*> nodes;

  for (std::size_t i = 0; i < numLinks; i++)
  {
    RevoluteJoint::Properties jointProps;
    jointProps.mName = "revolute_" + std::to_string(i);
    BodyNode::Properties bodyProps;
    bodyProps.mName = "arm_" + std::to_string(i);
    std::pair<RevoluteJoint*, BodyNode*> jointPair
        = arm->createJointAndBodyNodePair<RevoluteJoint>(
            parent, jointProps, bodyProps);
    nodes.push_back(jointPair.second);
    if (parent != nullptr)
    {
      Eigen::Isometry3s armOffset = Eigen::Isometry3s::Identity();
      armOffset.translation() = Eigen::Vector3s(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
      jointPair.first->setSpringStiffness(0,spring_stiff);
      jointPair.first->setRestPosition(0,1);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
    if ((attachPoint == -1 && i < numLinks - 1)
        || (attachPoint != -1 && i != attachPoint))
    {
      // ShapeNode* visual =
      parent->createShapeNodeWith<VisualAspect>(boxShape);
    }
  }

  if (attachPoint != -1)
  {
    parent = nodes[attachPoint];
  }

  std::shared_ptr<BoxShape> endShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0) * 1. / sqrt(2.0)));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);

  arm->setPositions(Eigen::VectorXs::Ones(arm->getNumDofs()) * rotationRadians);
  world->addSkeleton(arm);

  Eigen::Isometry3s worldTransform = parent->getWorldTransform();
  Eigen::Matrix3s rotation
      = math::eulerXYZToMatrix(Eigen::Vector3s(0, 45, -45) * 3.1415 / 180)
        * worldTransform.linear().transpose();
  endNode->setRelativeRotation(rotation);

  if (numLinks == 5)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * -0.05);
  }
  if (numLinks == 6 || numLinks == 3)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * 0.05);
  }


  Eigen::VectorXs worldVel = world->getVelocities();

  EXPECT_TRUE(verifyNextV(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalJacobians(world,true));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));

}

#ifdef ALL_TESTS

TEST(GRADIENTS, ARM_3_LINK_NO_CONTACT_30_DEG)
{
  testRobotArmNoContact(3, 30.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_5_LINK_NO_CONTACT_40_DEG)
{
  // This one penetrates much more deeply than the others
  testRobotArmNoContact(5, 40.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_6_LINK_NO_CONTACT_15_DEG)
{
  testRobotArmNoContact(6, 15.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_3_LINK_NO_CONTACT_30_DEG_MIDDLE_ATTACH)
{
  testRobotArmNoContact(3, 30.0 / 180 * 3.1415, 1);
}

#endif

void testFreeBlockWithFrictionCoeff(
    s_t frictionCoeff, s_t mass, bool largeInitialVelocity)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s::UnitY() * -9.81);
  world->setPenetrationCorrectionEnabled(false);

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<FreeJoint*, BodyNode*> pair
      = box->createJointAndBodyNodePair<FreeJoint>(nullptr);
  FreeJoint* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
  fromParent.translation() = Eigen::Vector3s::UnitX();
  boxJoint->setTransformFromParentBodyNode(fromParent);

  Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();
  fromChild.translation() = Eigen::Vector3s::UnitX() * -2;
  fromChild = fromChild.rotate(
      Eigen::AngleAxis<s_t>(M_PI_2, Eigen::Vector3s::UnitX()));
  boxJoint->setTransformFromChildBodyNode(fromChild);
  // Add damping coeffs
  boxJoint->setSpringStiffness(0,1);
  boxJoint->setRestPosition(0,1);
  boxJoint->setSpringStiffness(1,1);
  boxJoint->setRestPosition(1,1);
  boxJoint->setSpringStiffness(2,1);
  boxJoint->setRestPosition(2,1);
  boxJoint->setSpringStiffness(3,1);
  boxJoint->setSpringStiffness(4,1);
  boxJoint->setSpringStiffness(5,1);
  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);

  // Move off the origin to X=1, rotate 90deg on the Y axis
  box->setPosition(3, 1.0);
  box->setPosition(1, 90 * M_PI / 180);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3s floorPosition = Eigen::Isometry3s::Identity();
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - 1e-2), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXs timestepVel = box->getVelocities();
  VectorXs timestepWorldVel = world->getVelocities();

  world->step(true);

  if (largeInitialVelocity)
  {
    box->setVelocity(3, 0.01);
    box->setVelocity(5, 0.01);
  }

  VectorXs worldVel = world->getVelocities();
  EXPECT_TRUE(verifyNextV(world));

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
  EXPECT_TRUE(verifyPosGradients(world, 1, 1e-8));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_NO_FRICTION)
{
  testFreeBlockWithFrictionCoeff(0, 1, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_STATIC_FRICTION)
{
  testFreeBlockWithFrictionCoeff(1e7, 1, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_SLIPPING_FRICTION)
{
  testFreeBlockWithFrictionCoeff(0.5, 1, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_NO_FRICTION_INITIAL_VELOCITY)
{
  testFreeBlockWithFrictionCoeff(0, 1, true);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_STATIC_FRICTION_INITIAL_VELOCITY)
{
  testFreeBlockWithFrictionCoeff(1e-7, 1, true);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK_ON_GROUND_SLIPPING_FRICTION_INITIAL_VELOCITY)
{
  testFreeBlockWithFrictionCoeff(0.5, 1, true);
}
#endif
