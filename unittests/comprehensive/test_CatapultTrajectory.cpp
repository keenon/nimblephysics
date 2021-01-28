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

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/collision/fcl/FCLCollisionDetector.hpp"
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
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "TrajectoryTestUtils.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

//////////////////////////////////////////////////////////////////////
// This is meant to replicate the catapult.py example in C++. Here's the
// original Python file:
// clang-format off
/**

world = dart.simulation.World()
world.setGravity([0, -9.81, 0])

# Set up the projectile

projectile = dart.dynamics.Skeleton()

projectileJoint, projectileNode = projectile.createTranslationalJoint2DAndBodyNodePair()
projectileJoint.setXYPlane()
projectileShape = projectileNode.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
projectileVisual = projectileShape.createVisualAspect()
projectileShape.createCollisionAspect()
projectileVisual.setColor([0.7, 0.7, 0.7])
projectileVisual.setCastShadows(False)
projectileJoint.setForceUpperLimit(0, 0)
projectileJoint.setForceLowerLimit(0, 0)
projectileJoint.setForceUpperLimit(1, 0)
projectileJoint.setForceLowerLimit(1, 0)
projectileJoint.setVelocityUpperLimit(0, 1000.0)
projectileJoint.setVelocityLowerLimit(0, -1000.0)
projectileJoint.setVelocityUpperLimit(1, 1000.0)
projectileJoint.setVelocityLowerLimit(1, -1000.0)

projectile.setPositions(np.array([0, 0.1]))

world.addSkeleton(projectile)

# Set up catapult

catapult = dart.dynamics.Skeleton()

rootJoint, root = catapult.createWeldJointAndBodyNodePair()
rootOffset = dart.math.Isometry3()
rootOffset.set_translation([0.5, -0.45, 0])
rootJoint.setTransformFromParentBodyNode(rootOffset)

def createTailSegment(parent, color):
    poleJoint, pole = catapult.createRevoluteJointAndBodyNodePair(parent)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor(color)
    poleJoint.setForceUpperLimit(0, 1000.0)
    poleJoint.setForceLowerLimit(0, -1000.0)
    poleJoint.setVelocityUpperLimit(0, 10000.0)
    poleJoint.setVelocityLowerLimit(0, -10000.0)

    poleOffset = dart.math.Isometry3()
    poleOffset.set_translation([0, -0.125, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    poleJoint.setPosition(0, 90 * 3.1415 / 180)
    poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
    poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

    poleShape.createCollisionAspect()

    if parent != root:
        childOffset = dart.math.Isometry3()
        childOffset.set_translation([0, 0.125, 0])
        poleJoint.setTransformFromParentBodyNode(childOffset)
    return pole

tail1 = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255])
tail2 = createTailSegment(tail1, [223.0/255, 228.0/255, 163.0/255])
tail3 = createTailSegment(tail2, [221.0/255, 193.0/255, 121.0/255])

catapult.setPositions(np.array([45, 0, 45]) * 3.1415 / 180)

world.addSkeleton(catapult)

# Floor

floor = dart.dynamics.Skeleton()
floor.setName('floor')  # important for rendering shadows

floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
floorOffset = dart.math.Isometry3()
floorOffset.set_translation([1.2, -0.7, 0])
floorJoint.setTransformFromParentBodyNode(floorOffset)
floorShape: dart.dynamics.ShapeNode = floorBody.createShapeNode(dart.dynamics.BoxShape(
    [3.5, 0.25, .5]))
floorVisual: dart.dynamics.VisualAspect = floorShape.createVisualAspect()
floorVisual.setColor([0.5, 0.5, 0.5])
floorVisual.setCastShadows(False)
floorShape.createCollisionAspect()

world.addSkeleton(floor)

# Target

target_x = 2.2
target_y = 2.2

target = dart.dynamics.Skeleton()
target.setName('target')  # important for rendering shadows

targetJoint, targetBody = floor.createWeldJointAndBodyNodePair()
targetOffset = dart.math.Isometry3()
targetOffset.set_translation([target_x, target_y, 0])
targetJoint.setTransformFromParentBodyNode(targetOffset)
targetShape = targetBody.createShapeNode(dart.dynamics.BoxShape([0.1, 0.1, 0.1]))
targetVisual = targetShape.createVisualAspect()
targetVisual.setColor([0.8, 0.5, 0.5])

world.addSkeleton(target)

# Set up the view

def loss(rollout: DartTorchTrajectoryRollout):
    last_pos = rollout.getPoses('identity')[:, -1]
    last_x = last_pos[0]
    last_y = last_pos[1]
    final_loss = (target_x - last_x)**2 + (target_y - last_y)**2
    return final_loss
dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

trajectory = dart.trajectory.MultiShot(world, dartLoss, 500, 20, False)
# trajectory.setParallelOperationsEnabled(True)

def callback(problem: dart.trajectory.MultiShot, iter: int, loss: float, infeas: float):
    print('From Python, iter='+str(iter)+", loss="+str(loss))
    rollout: dart.trajectory.TrajectoryRollout = problem.getRolloutCache(world)
    currentPoses = rollout.getPoses()
    return True

optimizer = dart.trajectory.IPOptOptimizer()
optimizer.setLBFGSHistoryLength(5)
optimizer.setTolerance(1e-4)
optimizer.setCheckDerivatives(False)
optimizer.setIterationLimit(500)
optimizer.registerIntermediateCallback(callback)
result: dart.trajectory.Solution = optimizer.optimize(trajectory)

*/
// clang-format on

BodyNode* createTailSegment(BodyNode* parent, Eigen::Vector3d color)
{
  /*
  def createTailSegment(parent, color):
      poleJoint, pole = catapult.createRevoluteJointAndBodyNodePair(parent)
      poleJoint.setAxis([0, 0, 1])
      poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
      poleVisual = poleShape.createVisualAspect()
      poleVisual.setColor(color)
      poleJoint.setForceUpperLimit(0, 1000.0)
      poleJoint.setForceLowerLimit(0, -1000.0)
      poleJoint.setVelocityUpperLimit(0, 10000.0)
      poleJoint.setVelocityLowerLimit(0, -10000.0)

      poleOffset = dart.math.Isometry3()
      poleOffset.set_translation([0, -0.125, 0])
      poleJoint.setTransformFromChildBodyNode(poleOffset)

      poleJoint.setPosition(0, 90 * 3.1415 / 180)
      poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
      poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

      poleShape.createCollisionAspect()

      if parent != root:
          childOffset = dart.math.Isometry3()
          childOffset.set_translation([0, 0.125, 0])
          poleJoint.setTransformFromParentBodyNode(childOffset)
      return pole
  */
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
  poleJoint->setForceUpperLimit(0, 1000.0);
  poleJoint->setForceLowerLimit(0, -1000.0);
  poleJoint->setVelocityUpperLimit(0, 10000.0);
  poleJoint->setVelocityLowerLimit(0, -10000.0);

  Eigen::Isometry3d poleOffset = Eigen::Isometry3d::Identity();
  poleOffset.translation() = Eigen::Vector3d(0, -0.125, 0);
  poleJoint->setTransformFromChildBodyNode(poleOffset);
  poleJoint->setPosition(0, 90 * 3.1415 / 180);
  poleJoint->setPositionUpperLimit(0, 180 * 3.1415 / 180);
  poleJoint->setPositionLowerLimit(0, 0 * 3.1415 / 180);

  if (parent->getParentBodyNode() != nullptr)
  {
    Eigen::Isometry3d childOffset = Eigen::Isometry3d::Identity();
    childOffset.translation() = Eigen::Vector3d(0, 0.125, 0);
    poleJoint->setTransformFromParentBodyNode(childOffset);
  }

  return pole;
}

std::shared_ptr<simulation::World> createWorld(double target_x, double target_y)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  // world->setConstraintForceMixingEnabled(true);
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));

  // Create projectile

  SkeletonPtr projectile = Skeleton::create("projectile");
  std::pair<TranslationalJoint2D*, BodyNode*> projectileJointPair
      = projectile->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* projectileJoint = projectileJointPair.first;
  BodyNode* projectileBody = projectileJointPair.second;

  std::shared_ptr<BoxShape> projectileShape(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* projectileVisual
      = projectileBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          projectileShape);
  projectileVisual->getVisualAspect()->setColor(Eigen::Vector3d(0.7, 0.7, 0.7));
  projectileVisual->getVisualAspect()->setCastShadows(false);
  projectileJoint->setForceUpperLimit(0, 0);
  projectileJoint->setForceLowerLimit(0, 0);
  projectileJoint->setForceUpperLimit(1, 0);
  projectileJoint->setForceLowerLimit(1, 0);
  projectileJoint->setVelocityUpperLimit(0, 1000.0);
  projectileJoint->setVelocityLowerLimit(0, -1000.0);
  projectileJoint->setVelocityUpperLimit(1, 1000.0);
  projectileJoint->setVelocityLowerLimit(1, -1000.0);
  projectile->setPositions(Eigen::Vector2d(0, 0.1));
  world->addSkeleton(projectile);

  // Create catapult

  SkeletonPtr catapult = Skeleton::create("catapult");
  std::pair<WeldJoint*, BodyNode*> rootJointPair
      = catapult->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* rootJoint = rootJointPair.first;
  BodyNode* root = rootJointPair.second;

  Eigen::Isometry3d rootOffset = Eigen::Isometry3d::Identity();
  rootOffset.translation() = Eigen::Vector3d(0.5, -0.45, 0);
  rootJoint->setTransformFromParentBodyNode(rootOffset);

  BodyNode* tail1 = createTailSegment(
      root, Eigen::Vector3d(182.0 / 255, 223.0 / 255, 144.0 / 255));
  BodyNode* tail2 = createTailSegment(
      tail1, Eigen::Vector3d(223.0 / 255, 228.0 / 255, 163.0 / 255));
  BodyNode* tail3 = createTailSegment(
      tail2, Eigen::Vector3d(221.0 / 255, 193.0 / 255, 121.0 / 255));

  catapult->setPositions(Eigen::Vector3d(45, 0, 45) * 3.1415 / 180);

  world->addSkeleton(catapult);

  // Create floor

  SkeletonPtr floor = Skeleton::create("floor");
  std::pair<WeldJoint*, BodyNode*> floorJointPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorJointPair.first;
  BodyNode* floorBody = floorJointPair.second;
  Eigen::Isometry3d floorOffset = Eigen::Isometry3d::Identity();
  floorOffset.translation() = Eigen::Vector3d(1.2, -0.7, 0);
  floorJoint->setTransformFromParentBodyNode(floorOffset);
  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(3.5, 0.25, 0.5)));
  ShapeNode* floorVisual
      = floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          floorShape);
  floorVisual->getVisualAspect()->setColor(Eigen::Vector3d(0.5, 0.5, 0.5));
  floorVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(floor);

  // Create target

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3d targetOffset = Eigen::Isometry3d::Identity();
  targetOffset.translation() = Eigen::Vector3d(target_x, target_y, 0);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3d(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  return world;
}

#ifdef ALL_TESTS
TEST(CATAPULT_EXAMPLE, BROKEN_POINT)
{
  double target_x = 2.2;
  double target_y = 2.2;

  // Create a world
  std::shared_ptr<simulation::World> world = createWorld(target_x, target_y);

  /*
  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
  brokenPos << -7.4747, 9.43449, 2.12166, 2.98394, 2.34673;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
  brokenVel << -2.84978, 1.03633, 0, 9.16668, 6.99675;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
  brokenForce << 0, 0, -2.11163, -2.06504, -1.3781;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(12);
  brokenLCPCache << 0.0173545, 0.0132076, 0, 0.0173545, 0.0132076, 0, 0, 0, 0,
      0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */
  /*
  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
  brokenPos << 8.75823, -1.33554, 1.60919, 0.367526, 1.09027;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
  brokenVel << 4.48639, -5.53436, 1.73472e-18, -1.03812e-17, -0.472044;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
  brokenForce << 0, 0, 9.428, -1.14176, 0.947147;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(6);
  brokenLCPCache << 0.0491903, 0.00921924, 0, 0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */
  /*
  ///
  /// This used to fail to standardize the LCP properly
  ///

  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
  brokenPos << 8.75828, -1.33554, 1.6092, 0.367528, 1.09028;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
  brokenVel << 4.48642, -5.53436, -1.73472e-18, -2.35814e-18, -0.472011;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
  brokenForce << 0, 0, 9.428, -1.14176, 0.947147;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(6);
  brokenLCPCache << 0.0245947, 0.00461058, 0, 0.0245947, 0.00461058, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */

  /*
  ///
  /// This used to fail on edge-edge gradients being computed incorrectly
  ///
  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
  brokenPos << -0.13334, -0.178891, 1.07272, 0.130007, 0.436478;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
  brokenVel << -0.433131, -0.847734, 2.55373, -1.13021, -1.61568;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
  brokenForce << 0, 0, -0.17232, 6.83192, -0.275112;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(12);
  brokenLCPCache << 0, 0, 0, 0, 0, 0, 1.0778, 0.330749, 0, 1.0778, 0.330749, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */

  ///
  /// This used to fail on CLAMPING_THRESHOLD being too large in
  /// ConstraintGroupGradientMatrices.cpp
  ///
  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(5);
  brokenPos << -0.000646825, -0.0351094, 0.759088, 0.102786, 0.731049;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(5);
  brokenVel << -0.216819, -0.25626, 0.256483, 0.758835, -0.794271;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(5);
  brokenForce << 0, 0, 0.136721, 1.88135, 7.45379;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(12);
  brokenLCPCache << 0.00454883, -5.55535e-05, 0, 0.00454883, -5.55535e-05, 0, 0,
      0, 0, 0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  // EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  // EXPECT_TRUE(verifyF_c(world));

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXd animatePos = brokenPos;
  int i = 0;
  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    world->setPositions(animatePos);
    animatePos += brokenVel * 0.001;

    i++;
    if (i >= 100)
    {
      animatePos = brokenPos;
      i = 0;
    }
    // world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });
  while (server.isServing())
  {
    // spin
  }

  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = neural::forwardPass(world, true);
  EXPECT_TRUE(snapshot->areResultsStandardized());
}
#endif

// #ifdef ALL_TESTS
TEST(CATAPULT_EXAMPLE, FULL_TEST)
{
  double target_x = 2.2;
  double target_y = 2.2;

  // Create a world
  std::shared_ptr<simulation::World> world = createWorld(target_x, target_y);
  world->setSlowDebugResultsAgainstFD(true);

  TrajectoryLossFn loss =
      [target_x, target_y](const trajectory::TrajectoryRollout* rollout) {
        const Eigen::VectorXd lastPos
            = rollout->getPosesConst().col(rollout->getPosesConst().cols() - 1);

        double diffX = lastPos(0) - target_x;
        double diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      };

  TrajectoryLossFnAndGrad lossGrad
      = [target_x, target_y](
            const TrajectoryRollout* rollout,
            TrajectoryRollout* gradWrtRollout // OUT
        ) {
          int lastCol = rollout->getPosesConst().cols() - 1;
          const Eigen::VectorXd lastPos = rollout->getPosesConst().col(lastCol);

          double diffX = lastPos(0) - target_x;
          double diffY = lastPos(1) - target_y;

          gradWrtRollout->getPoses().setZero();
          gradWrtRollout->getVels().setZero();
          gradWrtRollout->getForces().setZero();
          gradWrtRollout->getPoses()(0, lastCol) = 2 * diffX;
          gradWrtRollout->getPoses()(1, lastCol) = 2 * diffY;

          return diffX * diffX + diffY * diffY;
        };

  trajectory::LossFn lossObj(loss);

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, lossObj, 500, 20, false);
  trajectory->setParallelOperationsEnabled(false);

  /*
  // EXPECT_TRUE(verifyTrajectory(world, trajectory));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyGradientBackprop(
      world,
      500,
      [target_x, target_y](std::shared_ptr<simulation::World> world) {
        const Eigen::VectorXd lastPos = world->getPositions();

        double diffX = lastPos(0) - target_x;
        double diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      }));

  EXPECT_TRUE(verifySingleStep(world, 5e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 5e-7, false, nullptr));
  EXPECT_TRUE(verifyShotJacobian(world, 4, nullptr));
  EXPECT_TRUE(verifyShotGradient(world, 7, loss, lossGrad));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 6, 2, nullptr));
  EXPECT_TRUE(verifySparseJacobian(world, 8, 2, nullptr));
  EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss, lossGrad));
  EXPECT_TRUE(verifyMultiShotJacobianCustomConstraint(
      world, 8, 4, loss, lossGrad, 3.0));
  */

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(1);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(true);
  optimizer.setIterationLimit(500);
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  int i = 0;
  const Eigen::MatrixXd poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXd vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  Ticker ticker(world->getTimeStep());
  ticker.registerTickListener([&](long time) {
    world->setPositions(poses.col(i));
    // world->setVelocities(vels.col(i));

    i++;
    if (i >= poses.cols())
    {
      i = 0;
    }
    // world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
    // spin
  }
  */
}
// #endif
