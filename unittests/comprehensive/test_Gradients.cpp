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

  Eigen::Isometry3s offset(Eigen::Isometry3s::Identity());
  offset.translation().noalias() = Eigen::Vector3s(0.0, 0.0, -1.0);
  Eigen::Vector3s axis = Eigen::Vector3s(0.0, 1.0, 0.0);

  // Joints
  joint1->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  joint1->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  joint1->setAxis(axis);

  joint2->setTransformFromParentBodyNode(offset);
  joint2->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  joint2->setAxis(axis);

  joint3->setTransformFromParentBodyNode(offset);
  joint3->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());
  joint3->setAxis(axis);

  // Add collisions to the last node of the chain
  std::shared_ptr<BoxShape> pendulumBox(
      new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));
  body1->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body1->setFrictionCoeff(0);
  body2->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body2->setFrictionCoeff(0);
  body3->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body3->setFrictionCoeff(0);

  // The block is to the right, drive the chain into the block
  body2->setExtForce(Eigen::Vector3s(5.0, 0, 0));
  world->addSkeleton(pendulum);

  ///////////////////////////////////////////////
  // Create the block
  ///////////////////////////////////////////////

  SkeletonPtr block = Skeleton::create("block");

  // Give the floor a body
  BodyNodePtr body
      = block->createJointAndBodyNodePair<WeldJoint>(nullptr).second;

  // Give the body a shape
  std::shared_ptr<BoxShape> box(new BoxShape(Eigen::Vector3s(1.0, 0.5, 0.5)));
  auto shapeNode
      = body->createShapeNodeWith<VisualAspect, CollisionAspect>(box);
  shapeNode->getVisualAspect()->setColor(dart::Color::Black());

  // Put the body into position
  Eigen::Isometry3s tf(Eigen::Isometry3s::Identity());
  tf.translation() = Eigen::Vector3s(0.55, 0.0, -1.0);
  body->getParentJoint()->setTransformFromParentBodyNode(tf);

  world->addSkeleton(block);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  pendulum->computeForwardDynamics();
  pendulum->integrateVelocities(world->getTimeStep());
  VectorXs timestepVel = pendulum->getVelocities();

  VectorXs worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyWorldGradients(world, worldVel));
}
*/

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

          +---+
          |   |
          +---+
            * contact
          +---+
          |   |
          +---+

There are a pair of spheres each on a single linear DOF, with the bottom sphere
pushing the top sphere up.

*/
void testSphereStack()
{
  // World
  WorldPtr world = World::create();

  std::shared_ptr<SphereShape> sphereShape(new SphereShape(0.5));

  ///////////////////////////////////////////////
  // Create the sphere A
  ///////////////////////////////////////////////

  SkeletonPtr sphereA = Skeleton::create("sphereA");
  std::pair<PrismaticJoint*, BodyNode*> pairA
      = sphereA->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* jointA = pairA.first;
  BodyNode* bodyA = pairA.second;
  jointA->setAxis(Eigen::Vector3s::UnitY());
  bodyA->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  bodyA->setFrictionCoeff(0.0);

  world->addSkeleton(sphereA);

  ///////////////////////////////////////////////
  // Create the sphere B (on top)
  ///////////////////////////////////////////////

  SkeletonPtr sphereB = Skeleton::create("sphereB");
  std::pair<PrismaticJoint*, BodyNode*> pairB
      = sphereB->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* jointB = pairB.first;
  BodyNode* bodyB = pairB.second;
  jointB->setAxis(Eigen::Vector3s::UnitY());
  bodyB->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  bodyB->setFrictionCoeff(0.0);

  sphereB->setControlForceUpperLimit(0, 0.0);
  sphereB->setControlForceLowerLimit(0, 0.0);
  sphereB->setPosition(0, 1.0 - CONTACT_MARGIN);

  world->addSkeleton(sphereB);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  VectorXs worldVel = world->getVelocities();
  worldVel(0) = 0.01;
  worldVel(1) = -0.005;
  world->setVelocities(worldVel);

  /*
  std::shared_ptr<neural::BackpropSnapshot> snapshot =
  neural::forwardPass(world, true); Eigen::MatrixXs forceVel =
  snapshot->getControlForceVelJacobian(world); std::cout << "force-vel" <<
  std::endl << forceVel << std::endl; Eigen::MatrixXs velVel =
  snapshot->getVelVelJacobian(world); std::cout << "vel-vel" << std::endl <<
  velVel << std::endl; Eigen::MatrixXs A_c =
  snapshot->getClampingConstraintMatrix(world); std::cout << "A_c" << std::endl
  << A_c << std::endl; Eigen::MatrixXs A_cc = snapshot->getClampingAMatrix();
  std::cout << "A_cc" << std::endl << A_cc << std::endl;
  Eigen::MatrixXs Minv = snapshot->getInvMassMatrix(world);
  std::cout << "Minv" << std::endl << Minv << std::endl;
  Eigen::MatrixXs rel = Minv * A_c *
  A_cc.completeOrthogonalDecomposition().pseudoInverse() * A_c.transpose();
  std::cout << "rel" << std::endl << rel << std::endl;
  // We want to push up the top sphere
  Eigen::VectorXs lossWrtNextVel = Eigen::VectorXs::Zero(2);
  lossWrtNextVel(1) = 1.0;
  std::cout << "loss wrt v_t+1" << std::endl << lossWrtNextVel << std::endl;
  // Here are the resulting other losses
  Eigen::VectorXs lossWrtForces = A_c.transpose() * lossWrtNextVel;
  std::cout << "loss wrt f_t" << std::endl << lossWrtForces << std::endl;
  Eigen::VectorXs lossWrtVel = velVel.transpose() * lossWrtNextVel;
  std::cout << "loss wrt v_t" << std::endl << lossWrtVel << std::endl;
  Eigen::VectorXs lossWrtControl = forceVel.transpose() * lossWrtNextVel;
  std::cout << "loss wrt tau_t" << std::endl << lossWrtControl << std::endl;
  lossWrtControl(1) = 0.0;
  std::cout << "clipped loss wrt tau_t" << std::endl << lossWrtControl <<
  std::endl; Eigen::VectorXs lossWrtNextVelRecovered = forceVel *
  lossWrtControl; std::cout << "loss wrt v_t+1" << std::endl <<
  lossWrtNextVelRecovered << std::endl; Eigen::VectorXs lossThroughLCP =
  rel.transpose() * lossWrtNextVel; std::cout << "loss wrt v_t through LCP" <<
  std::endl << lossThroughLCP << std::endl;
  */

  // Test the classic formulation
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_STACK)
{
  testSphereStack();
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
    s_t leftPressingForce,
    s_t rightPressingForce,
    s_t frictionCoeff,
    s_t leftMass,
    s_t rightMass)
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

  leftBoxJoint->setAxis(Eigen::Vector3s::UnitX());
  Eigen::Isometry3s leftBoxPosition = Eigen::Isometry3s::Identity();
  leftBoxPosition.translation() = Eigen::Vector3s(-0.5, 0, 0);
  leftBoxJoint->setTransformFromParentBodyNode(leftBoxPosition);
  leftBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> leftBoxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  ShapeNode* leftBoxShapeNode
      = leftBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          leftBoxShape);
  leftBoxShapeNode->setName("Left box shape");
  leftBoxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the right
  leftBoxBody->addExtForce(Eigen::Vector3s(leftPressingForce, -1, 0));
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

  rightBoxJoint->setAxis(Eigen::Vector3s::UnitX());
  Eigen::Isometry3s rightBoxPosition = Eigen::Isometry3s::Identity();
  rightBoxPosition.translation() = Eigen::Vector3s(0.5 - CONTACT_MARGIN, 0, 0);
  rightBoxJoint->setTransformFromParentBodyNode(rightBoxPosition);
  rightBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> rightBoxShape(
      new BoxShape(Eigen::Vector3s(1.0, 2.0, 2.0)));
  ShapeNode* rightBoxShapeNode
      = rightBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          rightBoxShape);
  rightBoxBody->setFrictionCoeff(frictionCoeff);
  rightBoxShapeNode->setName("Right box shape");

  // Add a force driving the box down into the floor, and to the left
  rightBoxBody->addExtForce(Eigen::Vector3s(-rightPressingForce, -1, 0));
  // Prevent the mass matrix from being Identity
  rightBoxBody->setMass(rightMass);

  world->addSkeleton(rightBox);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  leftBox->computeForwardDynamics();
  leftBox->integrateVelocities(world->getTimeStep());
  VectorXs leftBoxVel = leftBox->getVelocities();

  rightBox->computeForwardDynamics();
  rightBox->integrateVelocities(world->getTimeStep());
  VectorXs rightBoxVel = rightBox->getVelocities();

  VectorXs worldVel = world->getVelocities();

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
  world->getConstraintSolver()->runEnforceContactAndJointAndCustomConstraintsFn();

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
void testBouncingBlockWithFrictionCoeff(s_t frictionCoeff, s_t mass)
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
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));
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

  Eigen::Isometry3s floorPosition = Eigen::Isometry3s::Identity();
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - CONTACT_MARGIN), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXs worldVel = world->getVelocities();

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
void testReversePendulumSledWithFrictionCoeff(s_t frictionCoeff)
{
  // World
  WorldPtr world = World::create();
  world->setPenetrationCorrectionEnabled(false);

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
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3s(1, -10, 0));

  // Create the reverse pendulum portion

  RevoluteJoint::Properties pendulumJointProps;
  pendulumJointProps.mName = "Reverse Pendulum Joint";
  bodyProps.mName = "Reverse Pendulum Body";

  std::pair<RevoluteJoint*, BodyNode*> pendulumPair
      = boxBody->createChildJointAndBodyNodePair<RevoluteJoint>(
          pendulumJointProps, bodyProps);
  RevoluteJoint* pendulumJoint = pendulumPair.first;
  BodyNode* pendulumBody = pendulumPair.second;

  pendulumJoint->setTransformFromParentBodyNode(Eigen::Isometry3s::Identity());
  Eigen::Isometry3s pendulumBodyPosition = Eigen::Isometry3s::Identity();
  pendulumBodyPosition.translation() = Eigen::Vector3s(0, -1, 0);
  pendulumJoint->setTransformFromChildBodyNode(pendulumBodyPosition);
  pendulumJoint->setAxis(Eigen::Vector3s(0, 0, 1.0));
  pendulumBody->setMass(200);
  std::shared_ptr<BoxShape> pendulumBodyShape(
      new BoxShape(Eigen::Vector3s(0.5, 2.0, 0.5)));
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

  pendulumJoint->setPosition(0, 30 * 3.1415 / 180);
  reversePendulumSled->computeForwardDynamics();
  reversePendulumSled->integrateVelocities(world->getTimeStep());
  VectorXs worldVel = world->getVelocities();

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
void testBouncingBlockPosGradients(s_t frictionCoeff, s_t mass)
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

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  // boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));
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

  Eigen::Isometry3s floorPosition = Eigen::Isometry3s::Identity();
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - 1e-5), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  // The analytical Jacobian needs us to be in actual contact
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - 1e-5), 0);
  neural::BackpropSnapshotPtr analyticalPtr = neural::forwardPass(world, true);
  MatrixXs analytical = analyticalPtr->getPosPosJacobian(world);

  // The FD Jacobian needs us to be just out of contact range, so the bounce can
  // occur on some step in computation
  Eigen::MatrixXs bruteForce = Eigen::MatrixXs::Zero(2, 2);
  // clang-format off
  bruteForce << 1, 0, 
                0, -0.5;
  // clang-format on

  if (!equals(analytical, bruteForce, 5e-3))
  {
    std::cout << "Brute force pos-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical pos-pos Jacobian: " << std::endl
              << analytical << std::endl;
    EXPECT_TRUE(false);
  }
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

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  std::vector<SkeletonPtr> topBoxes;
  std::vector<SkeletonPtr> bottomBoxes;

  for (std::size_t i = 0; i < numGroups; i++)
  {
    // This is where this group is going to be positioned along the x axis
    s_t xOffset = i * 10;

    // Create the top box in the pair

    SkeletonPtr topBox = Skeleton::create("topBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> topBoxPair
        = topBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* topBoxJoint = topBoxPair.first;
    BodyNode* topBoxBody = topBoxPair.second;

    topBoxJoint->setXYPlane();
    Eigen::Isometry3s topBoxPosition = Eigen::Isometry3s::Identity();
    topBoxPosition.translation()
        = Eigen::Vector3s(xOffset, 0.5 - CONTACT_MARGIN, 0);
    topBoxJoint->setTransformFromParentBodyNode(topBoxPosition);
    topBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

    std::shared_ptr<BoxShape> topBoxShape(
        new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
    topBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(topBoxShape);
    topBoxBody->setFrictionCoeff(0.5);
    topBoxBody->setExtForce(Eigen::Vector3s(0, -1.0, 0));

    topBoxes.push_back(topBox);

    // Create the bottom box in the pair

    SkeletonPtr bottomBox = Skeleton::create("bottomBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> bottomBoxPair
        = bottomBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* bottomBoxJoint = bottomBoxPair.first;
    BodyNode* bottomBoxBody = bottomBoxPair.second;

    bottomBoxJoint->setXYPlane();
    Eigen::Isometry3s bottomBoxPosition = Eigen::Isometry3s::Identity();
    bottomBoxPosition.translation() = Eigen::Vector3s(xOffset, -0.5, 0);
    bottomBoxJoint->setTransformFromParentBodyNode(bottomBoxPosition);
    bottomBoxJoint->setTransformFromChildBodyNode(
        Eigen::Isometry3s::Identity());

    std::shared_ptr<BoxShape> bottomBoxShape(
        new BoxShape(Eigen::Vector3s(2.0, 1.0, 2.0)));
    bottomBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
        bottomBoxShape);
    bottomBoxBody->setFrictionCoeff(1);
    bottomBoxBody->setExtForce(Eigen::Vector3s(0, 1.0, 0));
    // Make each group less symmetric
    bottomBoxBody->setMass(1.0 / (i + 1));

    bottomBoxes.push_back(bottomBox);

    // Add a tiny bit of velocity to the boxes pushing them into each other
    topBox->setVelocity(1, -0.01);
    bottomBox->setVelocity(1, 0.01);
    // Add a non-zero shear velocity, so that we're not at a non-differentiable
    // point for x-axis forces
    topBox->setVelocity(0, 0.01);
  }

  // Add all the top boxes first, then all the bottom boxes. This ensures that
  // our constraint group ordering doesn't match our world ordering, which
  // will help us catch bugs in matrix layout.
  for (SkeletonPtr topBox : topBoxes)
    world->addSkeleton(topBox);
  for (SkeletonPtr bottomBox : bottomBoxes)
    world->addSkeleton(bottomBox);

  VectorXs worldVel = world->getVelocities();

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
    std::size_t numLinks, s_t rotationRadians, int attachPoint = -1)
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
  // ShapeNode* wallNode =
  jointPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      wallShape);
  world->addSkeleton(wall);
  // jointPair.second->setFrictionCoeff(0.0);

  Eigen::Isometry3s wallLocalOffset = Eigen::Isometry3s::Identity();
  wallLocalOffset.translation() = parent->getWorldTransform().translation()
                                  + Eigen::Vector3s(-(1.0 - 1e-2), 0.0, 0.0);
  jointPair.first->setTransformFromParentBodyNode(wallLocalOffset);

  // arm->computeForwardDynamics();
  // arm->integrateVelocities(world->getTimeStep());
  // -0.029 at 0.5
  // -0.323 at 5.0
  if (numLinks == 5)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * -0.05);
  }
  if (numLinks == 6 || numLinks == 3)
  {
    arm->setVelocities(Eigen::VectorXs::Ones(arm->getNumDofs()) * 0.05);
  }

  // // Run collision detection
  // world->getConstraintSolver()->solve(world.get());

  // // Check
  // auto result = world->getLastCollisionResult();
  // if (result.getNumContacts() > 0)
  // {
  //   std::cout << "Num contacts: " << result.getNumContacts() << std::endl;
  //   std::cout << "end affector offset: " << std::endl
  //             << endNode->getWorldTransform().matrix() << std::endl;
  //   std::cout << "wall node position: " << std::endl
  //             << wallNode->getWorldTransform().matrix() << std::endl;
  // }

  Eigen::VectorXs worldVel = world->getVelocities();

  // // visual inspection code
  // Eigen::VectorXs worldPos = world->getPositions();
  // server::GUIWebsocketServer server;
  // server.serve(8070);
  // server.renderWorld(world);
  // Eigen::VectorXs animatePos = worldPos;
  // int i = 0;
  // realtime::Ticker ticker(0.01);
  // ticker.registerTickListener([&](long /* time */) {
  //   world->setPositions(animatePos);
  //   animatePos += worldVel * 0.001;
  //   i++;
  //   if (i >= 100)
  //   {
  //     animatePos = worldPos;
  //     i = 0;
  //   }
  //   server.renderWorld(world);
  // });

  // server.registerConnectionListener([&]() { ticker.start(); });

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));

  // while (server.isServing())
  // {
  //   // spin
  // }
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
void testCartpole(s_t rotationRadians)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3s(1, 0, 0));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3s(0, 0, 1));

  Eigen::Isometry3s armOffset = Eigen::Isometry3s::Identity();
  armOffset.translation() = Eigen::Vector3s(0, -0.5, 0);
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

  VectorXs worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyWrtMass(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyGradientBackprop(world, 20, [](WorldPtr world) {
    Eigen::VectorXs pos = world->getPositions();
    Eigen::VectorXs vel = world->getVelocities();
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
  VectorXs worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

TEST(GRADIENTS, EMPTY_SKELETON)
{
  WorldPtr world = World::create();
  SkeletonPtr empty = Skeleton::create("empty");
  world->addSkeleton(empty);
  VectorXs worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Checking the trajectory optimizations
///////////////////////////////////////////////////////////////////////////////

BodyNode* createTailSegment(BodyNode* parent, Eigen::Vector3s color)
{
  std::pair<RevoluteJoint*, BodyNode*> poleJointPair
      = parent->createChildJointAndBodyNodePair<RevoluteJoint>();
  RevoluteJoint* poleJoint = poleJointPair.first;
  BodyNode* pole = poleJointPair.second;
  poleJoint->setAxis(Eigen::Vector3s::UnitZ());

  std::shared_ptr<BoxShape> shape(
      new BoxShape(Eigen::Vector3s(0.05, 0.25, 0.05)));
  ShapeNode* poleShape
      = pole->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  poleShape->getVisualAspect()->setColor(color);
  poleJoint->setControlForceUpperLimit(0, 100.0);
  poleJoint->setControlForceLowerLimit(0, -100.0);
  poleJoint->setVelocityUpperLimit(0, 10000.0);
  poleJoint->setVelocityLowerLimit(0, -10000.0);

  Eigen::Isometry3s poleOffset = Eigen::Isometry3s::Identity();
  poleOffset.translation() = Eigen::Vector3s(0, -0.125, 0);
  poleJoint->setTransformFromChildBodyNode(poleOffset);
  poleJoint->setPosition(0, 90 * 3.1415 / 180);

  if (parent->getParentBodyNode() != nullptr)
  {
    Eigen::Isometry3s childOffset = Eigen::Isometry3s::Identity();
    childOffset.translation() = Eigen::Vector3s(0, 0.125, 0);
    poleJoint->setTransformFromParentBodyNode(childOffset);
  }

  return pole;
}

void testJumpWorm(bool offGround, bool interpenetration)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  SkeletonPtr jumpworm = Skeleton::create("jumpworm");

  std::pair<TranslationalJoint2D*, BodyNode*> rootJointPair
      = jumpworm->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* rootJoint = rootJointPair.first;
  BodyNode* root = rootJointPair.second;

  std::shared_ptr<BoxShape> shape(new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));
  ShapeNode* rootVisual
      = root->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  Eigen::Vector3s black = Eigen::Vector3s::Zero();
  rootVisual->getVisualAspect()->setColor(black);
  rootJoint->setControlForceUpperLimit(0, 0);
  rootJoint->setControlForceLowerLimit(0, 0);
  rootJoint->setControlForceUpperLimit(1, 0);
  rootJoint->setControlForceLowerLimit(1, 0);
  rootJoint->setVelocityUpperLimit(0, 1000.0);
  rootJoint->setVelocityLowerLimit(0, -1000.0);
  rootJoint->setVelocityUpperLimit(1, 1000.0);
  rootJoint->setVelocityLowerLimit(1, -1000.0);

  BodyNode* tail1 = createTailSegment(
      root, Eigen::Vector3s(182.0 / 255, 223.0 / 255, 144.0 / 255));
  BodyNode* tail2 = createTailSegment(
      tail1, Eigen::Vector3s(223.0 / 255, 228.0 / 255, 163.0 / 255));
  // BodyNode* tail3 =
  createTailSegment(
      tail2, Eigen::Vector3s(221.0 / 255, 193.0 / 255, 121.0 / 255));

  Eigen::VectorXs pos = Eigen::VectorXs(5);
  pos << 0, 0, 90, 90, 45;
  jumpworm->setPositions(pos * 3.1415 / 180);

  world->addSkeleton(jumpworm);

  // Floor

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorJointPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorJointPair.first;
  BodyNode* floorBody = floorJointPair.second;
  Eigen::Isometry3s floorOffset = Eigen::Isometry3s::Identity();
  floorOffset.translation() = Eigen::Vector3s(0, offGround ? -0.7 : -0.56, 0);
  floorJoint->setTransformFromParentBodyNode(floorOffset);
  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(2.5, 0.25, 0.5)));
  // ShapeNode* floorVisual =
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(0);

  world->addSkeleton(floor);

  rootJoint->setVelocity(1, -0.1);

  // world->setTimeStep(1e-1);
  // world->step();
  // world->step();

  if (interpenetration)
  {
    Eigen::VectorXs initialPos = Eigen::VectorXs(5);
    initialPos << 0.96352, -0.5623, -0.0912082, 0.037308, 0.147683;
    // Initial vel
    Eigen::VectorXs initialVel = Eigen::VectorXs(5);
    initialVel << 0.110462, 0.457093, 0.257748, 0.592256, 0.167432;

    world->setPositions(initialPos);
    world->setVelocities(initialVel);

    /*
    Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
    brokenPos << -0.0223332, -0.345524, 1.15215, 1.99026, 1.49591;
    Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
    brokenVel << -0.0635003, -2.1615, -1.19201, 1.19774, 2.11499;
    Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
    brokenForce << 0, 0, 0.00564396, -0.0037863, -0.00587224;
    world->setPositions(brokenPos);
    world->setVelocities(brokenVel);
    world->setControlForces(brokenForce);
    */
  }

  Eigen::VectorXs vels = world->getVelocities();

  // renderWorld(world);

  EXPECT_TRUE(verifyAnalyticalJacobians(world, offGround));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyPosGradients(world, 1, 1e-8));
  EXPECT_TRUE(verifyWrtMass(world));
  // EXPECT_TRUE(verifyNoMultistepIntereference(world, 10));
  // EXPECT_TRUE(verifyAnalyticalBackprop(world));

  /*
  std::function<s_t(WorldPtr)> loss = [](WorldPtr world) {
    Eigen::VectorXs pos = world->getPositions();
    Eigen::VectorXs vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  };
  // Test to make sure the loss lambda doesn't crash
  s_t l = loss(world);
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

////////////////////////////////////////////////////////////////////
// All Atlas robot tests have been moved to test_AtlasGradients.cpp, since
// they're so slow.
////////////////////////////////////////////////////////////////////

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

There's a box with six DOFs, a full FreeJoint, with a force driving it into the
ground. The ground has configurable friction in this setup.

*/
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

  /*
  Eigen::Matrix4d realTransform = boxBody->getWorldTransform().matrix();
  Eigen::Matrix4d expTransform = math::expMapDart(box->getPositions()).matrix();
  if (realTransform != expTransform)
  {
    std::cout << "Transforms don't match!" << std::endl;
    std::cout << "Real: " << std::endl << realTransform << std::endl;
    std::cout << "Exp: " << std::endl << expTransform << std::endl;
    std::cout << "Diff: " << std::endl
              << (realTransform - expTransform) << std::endl;
    return;
  }
  */

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

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  VectorXs worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyNextV(world));

  // EXPECT_TRUE(verifyF_c(world));

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