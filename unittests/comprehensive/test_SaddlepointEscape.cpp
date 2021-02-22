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

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

#ifdef ALL_TESTS
TEST(SADDLEPOINTS, BALL_ON_FIXED_GROUND)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the ball
  ///////////////////////////////////////////////

  SkeletonPtr ball = Skeleton::create("ball");
  std::pair<PrismaticJoint*, BodyNode*> pairBall
      = ball->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* ballJoint = pairBall.first;
  BodyNode* ballBody = pairBall.second;
  ballJoint->setAxis(Eigen::Vector3d::UnitY());
  std::shared_ptr<SphereShape> sphereShape(
      new SphereShape(0.5));
  ballBody->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  ballBody->setFrictionCoeff(0.0);

  world->addSkeleton(ball);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");
  std::pair<WeldJoint*, BodyNode*> pairFloor
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  // WeldJoint* floorJoint = pairFloor.first;
  BodyNode* floorBody = pairFloor.second;
  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(5.0,1.0,5.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(0.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Try a loss function
  ///////////////////////////////////////////////

  ball->setPosition(0, 1.0 - 1e-4);
  // Set the ball to be clamping
  ball->setVelocity(0, - 1e-4);

  std::shared_ptr<neural::BackpropSnapshot> snapshot = neural::forwardPass(world, true);
  EXPECT_EQ(1, snapshot->getNumClamping());

  // Do plain vanilla loss, should block all progress
  
  neural::LossGradient nextTimestepLoss;
  nextTimestepLoss.lossWrtPosition = Eigen::VectorXd::Zero(1);
  nextTimestepLoss.lossWrtVelocity = Eigen::VectorXd::Ones(1) * -0.1; // points downwards, which means we want to move upwards
  neural::LossGradient thisTimestepLoss;
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss);
  EXPECT_EQ(0, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(0, thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(0, thisTimestepLoss.lossWrtTorque(0));

  // Try exploratory loss

  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss, nullptr, true);
  EXPECT_EQ(0.0, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(-0.1, thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(-0.1 * world->getTimeStep(), thisTimestepLoss.lossWrtTorque(0));

  /*
  ///////////////////////////////////////////////
  // Display a GUI
  ///////////////////////////////////////////////

  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  while (server.isServing()) {
    // do nothing
  }
  // server.blockWhileServing();
  */
}
#endif

#ifdef ALL_TESTS
TEST(SADDLEPOINTS, UNCONTROLLED_BALL_ON_PADDLE)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the ball
  ///////////////////////////////////////////////

  SkeletonPtr ball = Skeleton::create("ball");
  std::pair<PrismaticJoint*, BodyNode*> pairBall
      = ball->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* ballJoint = pairBall.first;
  BodyNode* ballBody = pairBall.second;
  ballJoint->setAxis(Eigen::Vector3d::UnitY());
  std::shared_ptr<SphereShape> sphereShape(
      new SphereShape(0.5));
  ballBody->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  ballBody->setFrictionCoeff(0.0);
  ballJoint->setForceUpperLimit(0, 0.0);
  ballJoint->setForceLowerLimit(0, 0.0);

  world->addSkeleton(ball);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr paddle = Skeleton::create("paddle");
  std::pair<PrismaticJoint*, BodyNode*> pairPaddle
      = paddle->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* paddleJoint = pairPaddle.first;
  paddleJoint->setAxis(Eigen::Vector3d::UnitY());
  BodyNode* paddleBody = pairPaddle.second;
  std::shared_ptr<BoxShape> paddleShape(
      new BoxShape(Eigen::Vector3d(5.0,1.0,5.0)));
  paddleBody->createShapeNodeWith<VisualAspect, CollisionAspect>(paddleShape);
  paddleBody->setFrictionCoeff(0.0);

  world->addSkeleton(paddle);

  ///////////////////////////////////////////////
  // Try a loss function
  ///////////////////////////////////////////////

  ball->setPosition(0, 1.0 - 1e-4);
  ball->setVelocity(0, - 1e-4);

  std::shared_ptr<neural::BackpropSnapshot> snapshot = neural::forwardPass(world, true);
  EXPECT_EQ(1, snapshot->getNumClamping());

  // Do plain vanilla loss, should block all progress
  
  neural::LossGradient nextTimestepLoss;
  nextTimestepLoss.lossWrtPosition = Eigen::VectorXd::Zero(2);
  nextTimestepLoss.lossWrtVelocity = Eigen::Vector2d(-0.1, 0.0); // points downwards for ball, which means we want to move upwards
  neural::LossGradient thisTimestepLoss;
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss);
  // We're interested in the paddle's loss
  EXPECT_EQ(0, thisTimestepLoss.lossWrtPosition(1));
  EXPECT_EQ(-0.05, thisTimestepLoss.lossWrtVelocity(1));
  EXPECT_EQ(-0.05 * world->getTimeStep(), thisTimestepLoss.lossWrtTorque(1));

  // Try exploratory loss, should explore not-clamping, decide clamping is better

  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss, nullptr, true);
  // We're interested in the paddle's loss
  EXPECT_EQ(0.0, thisTimestepLoss.lossWrtPosition(1));
  EXPECT_EQ(-0.05, thisTimestepLoss.lossWrtVelocity(1));
  EXPECT_EQ(-0.05 * world->getTimeStep(), thisTimestepLoss.lossWrtTorque(1));

  /*
  ///////////////////////////////////////////////
  // Display a GUI
  ///////////////////////////////////////////////

  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  while (server.isServing()) {
    // do nothing
  }
  // server.blockWhileServing();
  */
}
#endif