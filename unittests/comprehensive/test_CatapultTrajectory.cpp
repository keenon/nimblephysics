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

#include "TestHelpers.hpp"
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

// #ifdef ALL_TESTS
TEST(CATAPULT_EXAMPLE, FULL_TEST)
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

  double target_x = 2.2;
  double target_y = 2.2;

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

  trajectory::LossFn loss(
      [target_x, target_y](const trajectory::TrajectoryRollout* rollout) {
        const Eigen::VectorXd lastPos
            = rollout->getPosesConst().col(rollout->getPosesConst().cols() - 1);

        double diffX = lastPos(0) - target_x;
        double diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      });

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 500, 20, false);

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(false);
  optimizer.setIterationLimit(500);
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  int i = 0;
  const Eigen::MatrixXd poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXd vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  /*
  world->setPositions(poses.col(0));
  world->setVelocities(vels.col(0));

  for (int i = 0; i < 175; i++)
  {
    world->step();
    server.renderWorld(world);
  }

  world->step();
  */
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
}
// #endif
