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
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/SGDOptimizer.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;

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
  ballJoint->setAxis(Eigen::Vector3s::UnitY());
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
      new BoxShape(Eigen::Vector3s(5.0,1.0,5.0)));
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

  // Do plain vanilla loss, should block all progress to torque for both timesteps
  
  neural::LossGradient nextTimestepLoss;
  nextTimestepLoss.lossWrtPosition = Eigen::VectorXs::Ones(1) * -0.1; // points downwards, which means we want to move upwards
  nextTimestepLoss.lossWrtVelocity = Eigen::VectorXs::Zero(1);
  neural::LossGradient thisTimestepLoss;
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss);
  EXPECT_EQ(-0.1, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(-0.1 * world->getTimeStep(), thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(0, thisTimestepLoss.lossWrtTorque(0));
  nextTimestepLoss.lossWrtPosition = thisTimestepLoss.lossWrtPosition;
  nextTimestepLoss.lossWrtVelocity = thisTimestepLoss.lossWrtVelocity;
  nextTimestepLoss.lossWrtTorque.setZero();
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss);
  EXPECT_EQ(-0.1, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(-0.1 * world->getTimeStep(), thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(0, thisTimestepLoss.lossWrtTorque(0));

  // Try exploratory loss, should allow progress on torque

  nextTimestepLoss.lossWrtPosition = Eigen::VectorXs::Ones(1) * -0.1; // points downwards, which means we want to move upwards
  nextTimestepLoss.lossWrtVelocity = Eigen::VectorXs::Zero(1);
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss, nullptr, true);
  EXPECT_EQ(-0.1, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(-0.1 * world->getTimeStep(), thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(0, thisTimestepLoss.lossWrtTorque(0));
  nextTimestepLoss.lossWrtPosition = thisTimestepLoss.lossWrtPosition;
  nextTimestepLoss.lossWrtVelocity = thisTimestepLoss.lossWrtVelocity;
  nextTimestepLoss.lossWrtTorque.setZero();
  snapshot->backprop(world, thisTimestepLoss, nextTimestepLoss, nullptr, true);
  EXPECT_EQ(-0.1, thisTimestepLoss.lossWrtPosition(0));
  EXPECT_EQ(-0.2 * world->getTimeStep(), thisTimestepLoss.lossWrtVelocity(0));
  EXPECT_EQ(-0.1 * world->getTimeStep() * world->getTimeStep(), thisTimestepLoss.lossWrtTorque(0));

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
  ballJoint->setAxis(Eigen::Vector3s::UnitY());
  std::shared_ptr<SphereShape> sphereShape(
      new SphereShape(0.5));
  ballBody->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  ballBody->setFrictionCoeff(0.0);
  ballJoint->setControlForceUpperLimit(0, 0.0);
  ballJoint->setControlForceLowerLimit(0, 0.0);

  world->addSkeleton(ball);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr paddle = Skeleton::create("paddle");
  std::pair<PrismaticJoint*, BodyNode*> pairPaddle
      = paddle->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* paddleJoint = pairPaddle.first;
  paddleJoint->setAxis(Eigen::Vector3s::UnitY());
  BodyNode* paddleBody = pairPaddle.second;
  std::shared_ptr<BoxShape> paddleShape(
      new BoxShape(Eigen::Vector3s(5.0,1.0,5.0)));
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
  nextTimestepLoss.lossWrtPosition = Eigen::VectorXs::Zero(2);
  nextTimestepLoss.lossWrtVelocity = Eigen::Vector2s(-0.1, 0.0); // points downwards for ball, which means we want to move upwards
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

// #ifdef ALL_TESTS
TEST(SADDLEPOINTS, BALL_ON_FIXED_GROUND_TRAJECTORY)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s::UnitY() * -9.81);

  ///////////////////////////////////////////////
  // Create the ball
  ///////////////////////////////////////////////

  SkeletonPtr ball = Skeleton::create("ball");
  std::pair<PrismaticJoint*, BodyNode*> pairBall
      = ball->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* ballJoint = pairBall.first;
  BodyNode* ballBody = pairBall.second;
  ballJoint->setAxis(Eigen::Vector3s::UnitY());
  std::shared_ptr<SphereShape> sphereShape(
      new SphereShape(0.5));
  ballBody->createShapeNodeWith<VisualAspect, CollisionAspect>(sphereShape);
  ballBody->setFrictionCoeff(0.0);

  ballJoint->setControlForceLowerLimit(0, -10);
  ballJoint->setControlForceUpperLimit(0, 10);

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
      new BoxShape(Eigen::Vector3s(5.0,1.0,5.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(0.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Try a loss function
  ///////////////////////////////////////////////

  ball->setPosition(0, 1.0 - 1e-4);
  // Set the ball to be clamping
  ball->setVelocity(0, - 1e-4);
  ballBody->setMass(0.1);

  std::shared_ptr<neural::BackpropSnapshot> snapshot = neural::forwardPass(world, true);
  EXPECT_EQ(1, snapshot->getNumClamping());

  ///////////////////////////////////////////////
  // Build an actual trajectory
  ///////////////////////////////////////////////

  s_t goalPos = 5.0;

  TrajectoryLossFn loss = [goalPos](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst().cols();
    s_t lastPos = rollout->getPosesConst()(0, steps - 1);
    s_t diff = goalPos - lastPos;
    return diff * diff;
  };

  TrajectoryLossFnAndGrad lossGrad = [goalPos](const TrajectoryRollout* rollout,
                                        TrajectoryRollout* gradWrtRollout // OUT
                                     ) {
    gradWrtRollout->getPoses().setZero();
    gradWrtRollout->getVels().setZero();
    gradWrtRollout->getControlForces().setZero();

    int steps = rollout->getPosesConst().cols();
    s_t lastPos = rollout->getPosesConst()(0, steps - 1);
    s_t diff = lastPos - goalPos;
    gradWrtRollout->getPoses()(0, steps - 1) = 2 * diff;
    return diff * diff;
  };

  LossFn lossFn = LossFn(loss, lossGrad);

  SingleShot trajectory = SingleShot(world, lossFn, 50, false);
  trajectory.setExploreAlternateStrategies(true);

  world->setTimeStep(0.01);

  SGDOptimizer optimizer;
  optimizer.setLearningRate(1.0);
  optimizer.setIterationLimit(500);

  std::shared_ptr<Solution> solution = optimizer.optimize(&trajectory);
  std::cout << "Found forces: " << trajectory.getRolloutCache(world)->getControlForcesConst() << std::endl;
  std::cout << "Found vels: " << trajectory.getRolloutCache(world)->getVelsConst() << std::endl;
  std::cout << "Found poses: " << trajectory.getRolloutCache(world)->getPosesConst() << std::endl;

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
// #endif