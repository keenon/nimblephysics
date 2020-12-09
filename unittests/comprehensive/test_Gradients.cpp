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

#include <dart/gui/gui.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
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
#include "dart/simulation/World.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

// This test is ugly and difficult to interpret, but it broke our system early
// on. It now passes, and is here to detect regression.
/******************************************************************************

This test sets up a configuration that looks like this:

          O
          |
          | +
Force --> O | <-- Fixed
          | +
          |
          O

There's a 3 link pendulum, with a force driving the middle link into a fixed
block, creating a contact.

*/
/*
TEST(GRADIENTS, PENDULUM_BLOCK)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the pendulum
  ///////////////////////////////////////////////

  SkeletonPtr pendulum = Skeleton::create("pendulum");

  std::pair<RevoluteJoint*, BodyNode*> pair;
  BodyNode *body1, *body2, *body3;
  RevoluteJoint *joint1, *joint2, *joint3;

  RevoluteJoint::Properties jointProps;
  BodyNode::Properties bodyProps;

  jointProps.mName = 'Joint_1';
  bodyProps.mName = 'Body_1';
  pair = pendulum->createJointAndBodyNodePair<RevoluteJoint>(
      nullptr, jointProps, bodyProps);
  joint1 = pair.first;
  body1 = pair.second;

  jointProps.mName = 'Joint_2';
  bodyProps.mName = 'Body_2';
  pair = body1->createChildJointAndBodyNodePair<RevoluteJoint>(
      jointProps, bodyProps);
  joint2 = pair.first;
  body2 = pair.second;

  jointProps.mName = 'Joint_3';
  bodyProps.mName = 'Body_3';
  pair = body2->createChildJointAndBodyNodePair<RevoluteJoint>(
      jointProps, bodyProps);
  joint3 = pair.first;
  body3 = pair.second;

  Eigen::Isometry3d offset(Eigen::Isometry3d::Identity());
  offset.translation().noalias() = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d axis = Eigen::Vector3d(0.0, 1.0, 0.0);

  // Joints
  joint1->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  joint1->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint1->setAxis(axis);

  joint2->setTransformFromParentBodyNode(offset);
  joint2->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint2->setAxis(axis);

  joint3->setTransformFromParentBodyNode(offset);
  joint3->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint3->setAxis(axis);

  // Add collisions to the last node of the chain
  std::shared_ptr<BoxShape> pendulumBox(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  body1->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body1->setFrictionCoeff(0);
  body2->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body2->setFrictionCoeff(0);
  body3->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body3->setFrictionCoeff(0);

  // The block is to the right, drive the chain into the block
  body2->setExtForce(Eigen::Vector3d(5.0, 0, 0));
  world->addSkeleton(pendulum);

  ///////////////////////////////////////////////
  // Create the block
  ///////////////////////////////////////////////

  SkeletonPtr block = Skeleton::create("block");

  // Give the floor a body
  BodyNodePtr body
      = block->createJointAndBodyNodePair<WeldJoint>(nullptr).second;

  // Give the body a shape
  std::shared_ptr<BoxShape> box(new BoxShape(Eigen::Vector3d(1.0, 0.5, 0.5)));
  auto shapeNode
      = body->createShapeNodeWith<VisualAspect, CollisionAspect>(box);
  shapeNode->getVisualAspect()->setColor(dart::Color::Black());

  // Put the body into position
  Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());
  tf.translation() = Eigen::Vector3d(0.55, 0.0, -1.0);
  body->getParentJoint()->setTransformFromParentBodyNode(tf);

  world->addSkeleton(block);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  pendulum->computeForwardDynamics();
  pendulum->integrateVelocities(world->getTimeStep());
  VectorXd timestepVel = pendulum->getVelocities();

  VectorXd worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyWorldGradients(world, worldVel));
}
*/

// This is the margin so that finite-differencing over position doesn't break
// contacts
const double CONTACT_MARGIN = 1e-5;

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
void testBlockWithFrictionCoeff(double frictionCoeff, double mass)
{
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
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
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

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - CONTACT_MARGIN), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXd timestepVel = box->getVelocities();
  VectorXd timestepWorldVel = world->getVelocities();

  VectorXd worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_1_MASS)
{
  testBlockWithFrictionCoeff(0, 1);
}

#ifdef ALL_TESTS
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

                +---+
          +---+ |   |
Force --> |   | |   | <-- Force
          +---+ |   |
                +---+

There are two blocks, each with two DOFs (X, Y). The force pushing them
together (or apart, if negative) is configurable.

The right box is larger, to prevent exact vertex-vertex collisions, which are
hard for the engine to handle.

*/
void testTwoBlocks(
    double leftPressingForce,
    double rightPressingForce,
    double frictionCoeff,
    double leftMass,
    double rightMass)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the left box
  ///////////////////////////////////////////////

  SkeletonPtr leftBox = Skeleton::create("left box");

  std::pair<PrismaticJoint*, BodyNode*> leftBoxPair
      = leftBox->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* leftBoxJoint = leftBoxPair.first;
  BodyNode* leftBoxBody = leftBoxPair.second;

  leftBoxJoint->setAxis(Eigen::Vector3d::UnitX());
  Eigen::Isometry3d leftBoxPosition = Eigen::Isometry3d::Identity();
  leftBoxPosition.translation() = Eigen::Vector3d(-0.5, 0, 0);
  leftBoxJoint->setTransformFromParentBodyNode(leftBoxPosition);
  leftBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> leftBoxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* leftBoxShapeNode
      = leftBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          leftBoxShape);
  leftBoxShapeNode->setName("Left box shape");
  leftBoxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the right
  leftBoxBody->addExtForce(Eigen::Vector3d(leftPressingForce, -1, 0));
  // Prevent the mass matrix from being Identity
  leftBoxBody->setMass(leftMass);

  world->addSkeleton(leftBox);

  ///////////////////////////////////////////////
  // Create the right box
  ///////////////////////////////////////////////

  SkeletonPtr rightBox = Skeleton::create("right box");

  std::pair<PrismaticJoint*, BodyNode*> rightBoxPair
      = rightBox->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* rightBoxJoint = rightBoxPair.first;
  BodyNode* rightBoxBody = rightBoxPair.second;

  rightBoxJoint->setAxis(Eigen::Vector3d::UnitX());
  Eigen::Isometry3d rightBoxPosition = Eigen::Isometry3d::Identity();
  rightBoxPosition.translation() = Eigen::Vector3d(0.5 - CONTACT_MARGIN, 0, 0);
  rightBoxJoint->setTransformFromParentBodyNode(rightBoxPosition);
  rightBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> rightBoxShape(
      new BoxShape(Eigen::Vector3d(1.0, 2.0, 2.0)));
  ShapeNode* rightBoxShapeNode
      = rightBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          rightBoxShape);
  rightBoxBody->setFrictionCoeff(frictionCoeff);
  rightBoxShapeNode->setName("Right box shape");

  // Add a force driving the box down into the floor, and to the left
  rightBoxBody->addExtForce(Eigen::Vector3d(-rightPressingForce, -1, 0));
  // Prevent the mass matrix from being Identity
  rightBoxBody->setMass(rightMass);

  world->addSkeleton(rightBox);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  leftBox->computeForwardDynamics();
  leftBox->integrateVelocities(world->getTimeStep());
  VectorXd leftBoxVel = leftBox->getVelocities();

  rightBox->computeForwardDynamics();
  rightBox->integrateVelocities(world->getTimeStep());
  VectorXd rightBoxVel = rightBox->getVelocities();

  VectorXd worldVel = world->getVelocities();

  /*
  world->getConstraintSolver()->solve();
  std::cout << "Contacts: " <<
  world->getLastCollisionResult().getNumContacts()
            << std::endl;
  for (std::size_t i = 0; i <
  world->getLastCollisionResult().getNumContacts(); i++)
  {
    collision::Contact contact =
  world->getLastCollisionResult().getContact(i); std::cout << "Contact " << i
  << " "
              << contact.collisionObject1->getShapeFrame()->getName() << "<->"
              << contact.collisionObject2->getShapeFrame()->getName() << ": "
              << contact.point << " at depth " << contact.penetrationDepth
              << std::endl;
  }
  */

  // Test the classic formulation

  world->getConstraintSolver()->setGradientEnabled(true);
  world->getConstraintSolver()->solve();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, TWO_BLOCKS_1_1_MASS)
{
  testTwoBlocks(1, 1, 0, 1, 1);
}

TEST(GRADIENTS, TWO_BLOCKS_1_2_MASS)
{
  testTwoBlocks(2, 1, 0, 1, 2);
}

TEST(GRADIENTS, TWO_BLOCKS_3_5_MASS)
{
  testTwoBlocks(2, 1, 0, 3, 5);
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
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground and the block both have coefficients of restitution of 0.5.
The ground has configurable friction in this setup.

*/
void testBouncingBlockWithFrictionCoeff(double frictionCoeff, double mass)
{
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
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - CONTACT_MARGIN), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockWithFrictionCoeff(0, 1);
}
#endif

/******************************************************************************

This test sets up a configuration that looks like this:

          +---+
          | x |
          +-|-+
            |
            |
          +-|-+
Force --> | O |
          +---+
      -------------
            ^
       Fixed ground

There's a reverse pendulum sled with three DOFs, x and y axis, and angle of
the reverse pendulum, with a force driving it into the ground. The ground has
configurable friction in this setup.

*/
void testReversePendulumSledWithFrictionCoeff(double frictionCoeff)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr reversePendulumSled = Skeleton::create("reversePendulumSled");

  TranslationalJoint2D::Properties jointProps;
  BodyNode::Properties bodyProps;
  jointProps.mName = "2D Sled Translation";
  bodyProps.mName = "Sled";
  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = reversePendulumSled->createJointAndBodyNodePair<TranslationalJoint2D>(
          nullptr, jointProps, bodyProps);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));

  // Create the reverse pendulum portion

  RevoluteJoint::Properties pendulumJointProps;
  pendulumJointProps.mName = "Reverse Pendulum Joint";
  bodyProps.mName = "Reverse Pendulum Body";

  std::pair<RevoluteJoint*, BodyNode*> pendulumPair
      = boxBody->createChildJointAndBodyNodePair<RevoluteJoint>(
          pendulumJointProps, bodyProps);
  RevoluteJoint* pendulumJoint = pendulumPair.first;
  BodyNode* pendulumBody = pendulumPair.second;

  pendulumJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  Eigen::Isometry3d pendulumBodyPosition = Eigen::Isometry3d::Identity();
  pendulumBodyPosition.translation() = Eigen::Vector3d(0, -1, 0);
  pendulumJoint->setTransformFromChildBodyNode(pendulumBodyPosition);
  pendulumJoint->setAxis(Eigen::Vector3d(0, 0, 1.0));
  pendulumBody->setMass(200);
  std::shared_ptr<BoxShape> pendulumBodyShape(
      new BoxShape(Eigen::Vector3d(0.5, 2.0, 0.5)));
  pendulumBody->createShapeNodeWith<VisualAspect>(pendulumBodyShape);

  world->addSkeleton(reversePendulumSled);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - 1e-6), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  pendulumJoint->setPosition(0, 30 * 3.1415 / 180);
  reversePendulumSled->computeForwardDynamics();
  reversePendulumSled->integrateVelocities(world->getTimeStep());
  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SLIDING_REVERSE_PENDULUM_NO_FRICTION)
{
  testReversePendulumSledWithFrictionCoeff(0);
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
              <-- some small air gap
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground and the block both have coefficients of restitution of 0.5.
The ground has configurable friction in this setup.

*/
void testBouncingBlockPosGradients(double frictionCoeff, double mass)
{
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
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -1.0, 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  EXPECT_TRUE(verifyPosGradients(world, 300, 5e-3));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, POS_BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockPosGradients(0, 1);
}
#endif

/******************************************************************************

This test sets up a configuration that looks like this:

        Velocity      Velocity
            |             |
            v             v
          +---+         +---+
          |   |         |   |        * * *
          +---+         +---+
        +-------+     +-------+
        |       |     |       |
        +-------+     +-------+
            ^             ^
        Velocity      Velocity

There are "numGroups" pairs of boxes, each with force driving them together.

*/
void testMultigroup(int numGroups)
{
  // World
  WorldPtr world = World::create();

  std::vector<SkeletonPtr> topBoxes;
  std::vector<SkeletonPtr> bottomBoxes;

  for (std::size_t i = 0; i < numGroups; i++)
  {
    // This is where this group is going to be positioned along the x axis
    double xOffset = i * 10;

    // Create the top box in the pair

    SkeletonPtr topBox = Skeleton::create("topBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> topBoxPair
        = topBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* topBoxJoint = topBoxPair.first;
    BodyNode* topBoxBody = topBoxPair.second;

    topBoxJoint->setXYPlane();
    Eigen::Isometry3d topBoxPosition = Eigen::Isometry3d::Identity();
    topBoxPosition.translation()
        = Eigen::Vector3d(xOffset, 0.5 - CONTACT_MARGIN, 0);
    topBoxJoint->setTransformFromParentBodyNode(topBoxPosition);
    topBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

    std::shared_ptr<BoxShape> topBoxShape(
        new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
    topBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(topBoxShape);
    topBoxBody->setFrictionCoeff(0.5);
    topBoxBody->setExtForce(Eigen::Vector3d(0, -1.0, 0));

    topBoxes.push_back(topBox);

    // Create the bottom box in the pair

    SkeletonPtr bottomBox = Skeleton::create("bottomBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> bottomBoxPair
        = bottomBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* bottomBoxJoint = bottomBoxPair.first;
    BodyNode* bottomBoxBody = bottomBoxPair.second;

    bottomBoxJoint->setXYPlane();
    Eigen::Isometry3d bottomBoxPosition = Eigen::Isometry3d::Identity();
    bottomBoxPosition.translation() = Eigen::Vector3d(xOffset, -0.5, 0);
    bottomBoxJoint->setTransformFromParentBodyNode(bottomBoxPosition);
    bottomBoxJoint->setTransformFromChildBodyNode(
        Eigen::Isometry3d::Identity());

    std::shared_ptr<BoxShape> bottomBoxShape(
        new BoxShape(Eigen::Vector3d(2.0, 1.0, 2.0)));
    bottomBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
        bottomBoxShape);
    bottomBoxBody->setFrictionCoeff(1);
    bottomBoxBody->setExtForce(Eigen::Vector3d(0, 1.0, 0));
    // Make each group less symmetric
    bottomBoxBody->setMass(1.0 / (i + 1));

    bottomBoxes.push_back(bottomBox);

    // Add a tiny bit of velocity to the boxes
    topBox->computeForwardDynamics();
    topBox->integrateVelocities(world->getTimeStep());
    bottomBox->computeForwardDynamics();
    bottomBox->integrateVelocities(world->getTimeStep());
  }

  // Add all the top boxes first, then all the bottom boxes. This ensures that
  // our constraint group ordering doesn't match our world ordering, which
  // will help us catch bugs in matrix layout.
  for (SkeletonPtr topBox : topBoxes)
    world->addSkeleton(topBox);
  for (SkeletonPtr bottomBox : bottomBoxes)
    world->addSkeleton(bottomBox);

  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, MULTIGROUP_2)
{
  testMultigroup(2);
}

TEST(GRADIENTS, MULTIGROUP_4)
{
  testMultigroup(4);
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
    std::size_t numLinks, double rotationRadians, int attachPoint = -1)
{
  // World
  WorldPtr world = World::create();
  /*
  for (auto key : collision::CollisionDetector::getFactory()->getKeys())
  {
    std::cout << "Option: " << key << std::endl;
  }
  */
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));

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
      Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
      armOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
    if ((attachPoint == -1 && i < numLinks - 1) || i != attachPoint)
    {
      ShapeNode* visual = parent->createShapeNodeWith<VisualAspect>(boxShape);
    }
  }

  if (attachPoint != -1)
  {
    parent = nodes[attachPoint];
  }

  std::shared_ptr<BoxShape> endShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0) * sqrt(1.0 / 3.0)));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);

  arm->setPositions(Eigen::VectorXd::Ones(arm->getNumDofs()) * rotationRadians);
  world->addSkeleton(arm);

  Eigen::Isometry3d worldTransform = parent->getWorldTransform();
  Eigen::Matrix3d rotation
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, -45) * 3.1415 / 180)
        * worldTransform.linear().transpose();
  endNode->setRelativeRotation(rotation);

  SkeletonPtr wall = Skeleton::create("wall");
  std::pair<WeldJoint*, BodyNode*> jointPair
      = wall->createJointAndBodyNodePair<WeldJoint>(nullptr);
  std::shared_ptr<BoxShape> wallShape(
      new BoxShape(Eigen::Vector3d(1.0, 10.0, 10.0)));
  ShapeNode* wallNode
      = jointPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          wallShape);
  world->addSkeleton(wall);
  // jointPair.second->setFrictionCoeff(0.0);

  Eigen::Isometry3d wallLocalOffset = Eigen::Isometry3d::Identity();
  wallLocalOffset.translation() = parent->getWorldTransform().translation()
                                  + Eigen::Vector3d(-(1.0 - 1e-2), 0.0, 0);
  jointPair.first->setTransformFromParentBodyNode(wallLocalOffset);

  /*
  // Run collision detection
  world->getConstraintSolver()->solve();

  // Check
  auto result = world->getLastCollisionResult();
  if (result.getNumContacts() > 0)
  {
    std::cout << "Num contacts: " << result.getNumContacts() << std::endl;
    std::cout << "end affector offset: " << std::endl
              << endNode->getWorldTransform().matrix() << std::endl;
    std::cout << "wall node position: " << std::endl
              << wallNode->getWorldTransform().matrix() << std::endl;
  }
  */

  // arm->computeForwardDynamics();
  // arm->integrateVelocities(world->getTimeStep());
  // -0.029 at 0.5
  // -0.323 at 5.0
  arm->setVelocities(Eigen::VectorXd::Ones(arm->getNumDofs()) * 0.05);

  VectorXd worldVel = world->getVelocities();

  /*
  VectorXd pos = world->getPositions();
  pos(0) += 1e-4;
  world->setPositions(pos);
  */

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
  // This test wraps an arm around, and it's actually breaking contact, so this
  // tests unconstrained free-motion
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

/******************************************************************************

This test sets up a configuration that looks something like this:

           | |
           | |
           | |
    ======= O =======
            ^
      Rotating base

It's a robot arm, with a rotating base, with "numLinks" links and
"rotationDegree" position at each link. There's also a fixed plane at the end
of the robot arm that it intersects with.
*/
void testCartpole(double rotationRadians)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 0, 0));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
  armOffset.translation() = Eigen::Vector3d(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  /*
  cartpole = dart.dynamics.Skeleton()
  cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
  cartRail.setAxis([1, 0, 0])
  cartShape = cart.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
  cartVisual = cartShape.createVisualAspect()
  cartVisual.setColor([0, 0, 0])

  poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
  poleJoint.setAxis([0, 0, 1])
  poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
  poleVisual = poleShape.createVisualAspect()
  poleVisual.setColor([0, 0, 0])

  poleOffset = dart.math.Isometry3()
  poleOffset.set_translation([0, -0.5, 0])
  poleJoint.setTransformFromChildBodyNode(poleOffset)
  */

  world->addSkeleton(cartpole);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, rotationRadians);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyWrtMass(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyGradientBackprop(world, 20, [](WorldPtr world) {
    Eigen::VectorXd pos = world->getPositions();
    Eigen::VectorXd vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  }));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, CARTPOLE_15_DEG)
{
  testCartpole(15.0 / 180.0 * 3.1415);
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Just idiot checking that the code doesn't crash on silly edge cases.
///////////////////////////////////////////////////////////////////////////////

#ifdef ALL_TESTS
TEST(GRADIENTS, EMPTY_WORLD)
{
  WorldPtr world = World::create();
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

TEST(GRADIENTS, EMPTY_SKELETON)
{
  WorldPtr world = World::create();
  SkeletonPtr empty = Skeleton::create("empty");
  world->addSkeleton(empty);
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Checking the trajectory optimizations
///////////////////////////////////////////////////////////////////////////////

BodyNode* createTailSegment(BodyNode* parent, Eigen::Vector3d color)
{
  std::pair<RevoluteJoint*, BodyNode*> poleJointPair
      = parent->createChildJointAndBodyNodePair<RevoluteJoint>();
  RevoluteJoint* poleJoint = poleJointPair.first;
  BodyNode* pole = poleJointPair.second;
  poleJoint->setAxis(Eigen::Vector3d::UnitZ());

  std::shared_ptr<BoxShape> shape(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* poleShape
      = pole->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  poleShape->getVisualAspect()->setColor(color);
  poleJoint->setForceUpperLimit(0, 100.0);
  poleJoint->setForceLowerLimit(0, -100.0);
  poleJoint->setVelocityUpperLimit(0, 10000.0);
  poleJoint->setVelocityLowerLimit(0, -10000.0);

  Eigen::Isometry3d poleOffset = Eigen::Isometry3d::Identity();
  poleOffset.translation() = Eigen::Vector3d(0, -0.125, 0);
  poleJoint->setTransformFromChildBodyNode(poleOffset);
  poleJoint->setPosition(0, 90 * 3.1415 / 180);

  if (parent->getParentBodyNode() != nullptr)
  {
    Eigen::Isometry3d childOffset = Eigen::Isometry3d::Identity();
    childOffset.translation() = Eigen::Vector3d(0, 0.125, 0);
    poleJoint->setTransformFromParentBodyNode(childOffset);
  }

  return pole;
}

void testJumpWorm(bool offGround, bool interpenetration)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr jumpworm = Skeleton::create("jumpworm");

  std::pair<TranslationalJoint2D*, BodyNode*> rootJointPair
      = jumpworm->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* rootJoint = rootJointPair.first;
  BodyNode* root = rootJointPair.second;

  std::shared_ptr<BoxShape> shape(new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* rootVisual
      = root->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  Eigen::Vector3d black = Eigen::Vector3d::Zero();
  rootVisual->getVisualAspect()->setColor(black);
  rootJoint->setForceUpperLimit(0, 0);
  rootJoint->setForceLowerLimit(0, 0);
  rootJoint->setForceUpperLimit(1, 0);
  rootJoint->setForceLowerLimit(1, 0);
  rootJoint->setVelocityUpperLimit(0, 1000.0);
  rootJoint->setVelocityLowerLimit(0, -1000.0);
  rootJoint->setVelocityUpperLimit(1, 1000.0);
  rootJoint->setVelocityLowerLimit(1, -1000.0);

  BodyNode* tail1 = createTailSegment(
      root, Eigen::Vector3d(182.0 / 255, 223.0 / 255, 144.0 / 255));
  BodyNode* tail2 = createTailSegment(
      tail1, Eigen::Vector3d(223.0 / 255, 228.0 / 255, 163.0 / 255));
  BodyNode* tail3 = createTailSegment(
      tail2, Eigen::Vector3d(221.0 / 255, 193.0 / 255, 121.0 / 255));

  Eigen::VectorXd pos = Eigen::VectorXd(5);
  pos << 0, 0, 90, 90, 45;
  jumpworm->setPositions(pos * 3.1415 / 180);

  world->addSkeleton(jumpworm);

  // Floor

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorJointPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorJointPair.first;
  BodyNode* floorBody = floorJointPair.second;
  Eigen::Isometry3d floorOffset = Eigen::Isometry3d::Identity();
  floorOffset.translation() = Eigen::Vector3d(0, offGround ? -0.7 : -0.56, 0);
  floorJoint->setTransformFromParentBodyNode(floorOffset);
  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(2.5, 0.25, 0.5)));
  ShapeNode* floorVisual
      = floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          floorShape);
  floorBody->setFrictionCoeff(0);

  world->addSkeleton(floor);

  rootJoint->setVelocity(1, -0.1);

  // world->setTimeStep(1e-1);
  // world->step();
  // world->step();

  if (interpenetration)
  {
    Eigen::VectorXd initialPos = Eigen::VectorXd(5);
    initialPos << 0.96352, -0.5623, -0.0912082, 0.037308, 0.147683;
    // Initial vel
    Eigen::VectorXd initialVel = Eigen::VectorXd(5);
    initialVel << 0.110462, 0.457093, 0.257748, 0.592256, 0.167432;

    world->setPositions(initialPos);
    world->setVelocities(initialVel);

    /*
    Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
    brokenPos << -0.0223332, -0.345524, 1.15215, 1.99026, 1.49591;
    Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
    brokenVel << -0.0635003, -2.1615, -1.19201, 1.19774, 2.11499;
    Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
    brokenForce << 0, 0, 0.00564396, -0.0037863, -0.00587224;
    world->setPositions(brokenPos);
    world->setVelocities(brokenVel);
    world->setForces(brokenForce);
    */
  }

  Eigen::VectorXd vels = world->getVelocities();

  // renderWorld(world);

  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyNoMultistepIntereference(world, 10));
  // EXPECT_TRUE(verifyAnalyticalBackprop(world));

  /*
  std::function<double(WorldPtr)> loss = [](WorldPtr world) {
    Eigen::VectorXd pos = world->getPositions();
    Eigen::VectorXd vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  };
  // Test to make sure the loss lambda doesn't crash
  double l = loss(world);
  EXPECT_TRUE(verifyGradientBackprop(world, 50, loss));
  */
}

#ifdef ALL_TESTS
TEST(GRADIENTS, JUMP_WORM)
{
  testJumpWorm(false, false);
}

TEST(GRADIENTS, JUMP_WORM_OFF_GROUND)
{
  testJumpWorm(true, false);
}
#endif

/*
TEST(GRADIENTS, JUMP_WORM_INTER_PENETRATE)
{
  testJumpWorm(false, true);
}
*/