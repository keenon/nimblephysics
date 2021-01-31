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
#include <iostream>
#include <thread>

#include <dart/gui/gui.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/gui/glut/TrajectoryReplayWindow.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;
using namespace performance;

bool equals(TimestepJacobians a, TimestepJacobians b, double threshold)
{
  return equals(a.forcePos, b.forcePos, threshold)
         && equals(a.forceVel, b.forceVel, threshold)
         && equals(a.posPos, b.posPos, threshold)
         && equals(a.posVel, b.posVel, threshold)
         && equals(a.velPos, b.velPos, threshold)
         && equals(a.velVel, b.velVel, threshold);
}

bool verifyMultiShotOptimization(WorldPtr world, MultiShot shot)
{
  IPOptOptimizer optimizer = IPOptOptimizer();

  optimizer.setIterationLimit(100);
  optimizer.setRecordPerformanceLog(true);
  std::shared_ptr<Solution> record = optimizer.optimize(&shot);
  EXPECT_TRUE(record->getNumSteps() <= 101);
  EXPECT_EQ(record->getStep(0).index, 0);
  EXPECT_EQ(record->getStep(1).index, 1);
  EXPECT_TRUE(record->getStep(1).rollout != record->getStep(0).rollout);

  // Playback the trajectory

  TrajectoryRolloutReal withKnots = TrajectoryRolloutReal(&shot);
  TrajectoryRolloutReal withoutKnots = TrajectoryRolloutReal(&shot);

  // Get the version with knots
  shot.getStates(world, &withKnots, nullptr, true);
  // Get the version without knots next, so that they can play in a loop
  shot.getStates(world, &withoutKnots, nullptr, false);

  auto results = PerformanceLog::finalize();
  for (auto pair : results)
  {
    std::cout << pair.second->prettyPrint() << std::endl;
  }

  // Create a window for rendering the world and handling user input
  // dart::gui::glut::displayTrajectoryInGUI(world, &shot);
  return true;
}

/*
TEST(TRAJECTORY, UNCONSTRAINED_BOX)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

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
  boxBody->setFrictionCoeff(0.0);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(1.0);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  // Passes
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false, nullptr));
  EXPECT_TRUE(verifyShotJacobian(world, 40, nullptr));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, nullptr));

  // Verify using the IK mapping as the representation
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addLinearBodyNode(boxBody);
  EXPECT_TRUE(verifyChangeRepresentationToIK(world, 10, 5, ikMap, true, true));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false, ikMap));
  EXPECT_TRUE(verifyShotJacobian(world, 40, ikMap));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, ikMap));
}

TEST(TRAJECTORY, REVOLUTE_JOINT)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr spinner = Skeleton::create("spinner");

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = spinner->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  world->addSkeleton(spinner);

  spinner->setPosition(0, 15.0 / 180.0 * 3.1415);
  spinner->computeForwardDynamics();
  spinner->integrateVelocities(world->getTimeStep());

  // Passes
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false, nullptr));
  EXPECT_TRUE(verifyShotJacobian(world, 40, nullptr));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, nullptr));

  // Verify using the IK mapping as the representation
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addAngularBodyNode(armPair.second);
  EXPECT_TRUE(verifyChangeRepresentationToIK(world, 10, 5, ikMap, true, true));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false, ikMap));
  EXPECT_TRUE(verifyShotJacobian(world, 40, ikMap));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, ikMap));
}

TEST(TRAJECTORY, TWO_LINK)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  // Add child arm

  std::pair<RevoluteJoint*, BodyNode*> elbowPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(armPair.second);
  Eigen::Isometry3d elbowOffset = Eigen::Isometry3d::Identity();
  elbowOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
  elbowPair.first->setTransformFromParentBodyNode(elbowOffset);

  world->addSkeleton(arm);

  arm->setPosition(0, 15.0 / 180.0 * 3.1415);
  arm->computeForwardDynamics();
  arm->integrateVelocities(world->getTimeStep());

  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  // This passes, but is very slow, so we'll skip it for now
  // EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40, nullptr));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, nullptr));

  // Verify using the IK mapping as the representation
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addSpatialBodyNode(armPair.second);
  ikMap->addSpatialBodyNode(elbowPair.second);
  EXPECT_TRUE(verifyChangeRepresentationToIK(world, 10, 5, ikMap, true, true));
  EXPECT_TRUE(verifyShotJacobian(world, 40, ikMap));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, ikMap));
}

TEST(TRAJECTORY, PRISMATIC)
{

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 1, 0));

  world->addSkeleton(cartpole);

  cartpole->setPosition(0, 0);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  // Passes
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false, nullptr));
  EXPECT_TRUE(verifyShotJacobian(world, 40, nullptr));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, nullptr));

  // EXPECT_TRUE(verifyShotGradient(world, 7, loss));
  // EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss));
  // EXPECT_TRUE(verifyMultiShotJacobianCustomConstraint(
  // world, 8, 4, loss, lossGrad, 3.0));

  // Verify using the IK mapping as the representation
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addSpatialBodyNode(sledPair.second);
  EXPECT_TRUE(verifyChangeRepresentationToIK(world, 10, 5, ikMap, true, true));
  EXPECT_TRUE(verifyShotJacobian(world, 40, ikMap));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2, ikMap));
}

TEST(TRAJECTORY, CARTPOLE)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 0, 0));
  std::shared_ptr<BoxShape> sledShapeBox(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* sledShape
      = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));
  std::shared_ptr<BoxShape> armShapeBox(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* armShape
      = armPair.second->createShapeNodeWith<VisualAspect>(armShapeBox);

  Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
  armOffset.translation() = Eigen::Vector3d(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  world->addSkeleton(cartpole);

  cartpole->setForceUpperLimit(0, 0);
  cartpole->setForceLowerLimit(0, 0);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);

  cartpole->setForceLowerLimit(1, -1000);
  cartpole->setForceUpperLimit(1, 1000);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst("identity").cols();
    Eigen::VectorXd lastPos = rollout->getPosesConst("identity").col(steps - 1);
    return rollout->getVelsConst("identity").col(steps - 1).squaredNorm()
           + lastPos.squaredNorm()
           + rollout->getForcesConst("identity").squaredNorm();
  };

  TrajectoryLossFnAndGrad lossGrad = [](const TrajectoryRollout* rollout,
                                        TrajectoryRollout* gradWrtRollout // OUT
                                     ) {
    gradWrtRollout->getPoses("identity").setZero();
    gradWrtRollout->getVels("identity").setZero();
    gradWrtRollout->getForces("identity").setZero();
    int steps = rollout->getPosesConst("identity").cols();
    gradWrtRollout->getPoses("identity").col(steps - 1)
        = 2 * rollout->getPosesConst("identity").col(steps - 1);
    gradWrtRollout->getVels("identity").col(steps - 1)
        = 2 * rollout->getVelsConst("identity").col(steps - 1);
    for (int i = 0; i < steps; i++)
    {
      gradWrtRollout->getForces("identity").col(i)
          = 2 * rollout->getForcesConst("identity").col(i);
    }
    Eigen::VectorXd lastPos = rollout->getPosesConst("identity").col(steps - 1);
    return rollout->getVelsConst("identity").col(steps - 1).squaredNorm()
           + lastPos.squaredNorm()
           + rollout->getForcesConst("identity").squaredNorm();
  };

  LossFn lossFn(loss);
  MultiShot shot(world, lossFn, 50, 10, false);
  EXPECT_TRUE(verifyMultiShotOptimization(world, shot));
}
*/

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
  poleJoint->setVelocityUpperLimit(0, 100.0);
  poleJoint->setVelocityLowerLimit(0, -100.0);
  poleJoint->setPositionUpperLimit(0, 270 * 3.1415 / 180);
  poleJoint->setPositionLowerLimit(0, -270 * 3.1415 / 180);

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

TEST(TRAJECTORY, JUMP_WORM)
{
  bool offGround = false;

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  world->setPenetrationCorrectionEnabled(false);
  world->setConstraintForceMixingEnabled(false);

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
  rootJoint->setPositionUpperLimit(0, 5);
  rootJoint->setPositionLowerLimit(0, -5);
  rootJoint->setPositionUpperLimit(1, 5);
  rootJoint->setPositionLowerLimit(1, -5);

  BodyNode* tail1 = createTailSegment(
      root, Eigen::Vector3d(182.0 / 255, 223.0 / 255, 144.0 / 255));
  BodyNode* tail2 = createTailSegment(
      tail1, Eigen::Vector3d(223.0 / 255, 228.0 / 255, 163.0 / 255));
  // BodyNode* tail3 =
  createTailSegment(
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
  // ShapeNode* floorVisual =
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(0);

  world->addSkeleton(floor);

  rootJoint->setVelocity(1, -0.1);
  Eigen::VectorXd vels = world->getVelocities();

  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    const Eigen::Ref<const Eigen::MatrixXd> poses
        = rollout->getPosesConst("identity");
    const Eigen::Ref<const Eigen::MatrixXd> vels
        = rollout->getVelsConst("identity");
    const Eigen::Ref<const Eigen::MatrixXd> forces
        = rollout->getForcesConst("identity");

    double maxPos = -1000;
    double minPos = 1000;
    for (int i = 0; i < poses.cols(); i++)
    {
      if (poses(1, i) > maxPos)
      {
        maxPos = poses(1, i);
      }
      if (poses(1, i) < minPos)
      {
        minPos = poses(1, i);
      }
    }
    // double peakPosLoss = -(maxPos * maxPos) * (maxPos > 0 ? 1.0 : -1.0);
    // double minPosLoss = -(minPos * minPos) * (minPos > 0 ? 1.0 : -1.0);
    double endPos = poses(1, poses.cols() - 1);
    double endPosLoss = -(endPos * endPos) * (endPos > 0 ? 1.0 : -1.0);

    // double forceLoss = forces.squaredNorm();

    // return endPosLoss * 100 + forceLoss * 1e-3;
    // return forceLoss;
    return endPosLoss; // + forceLoss;
    // return (100 * peakPosLoss) + (20 * minPosLoss) + endPosLoss;

    /*
    Eigen::VectorXd midVel = vels.col(vels.cols() / 2);
    double midVelSquaredSigned
        = -(midVel[1] * midVel[1]) * (midVel[1] > 0 ? 1.0 : -1.0);

    return posSquaredSigned + midVelSquaredSigned;
    */
    /*
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
    */
  };

  TrajectoryLossFnAndGrad lossGrad
      = [](const TrajectoryRollout* rollout,
           /* OUT */ TrajectoryRollout* gradWrtRollout) {
          gradWrtRollout->getPoses("identity").setZero();
          gradWrtRollout->getVels("identity").setZero();
          gradWrtRollout->getForces("identity").setZero();
          const Eigen::Ref<const Eigen::MatrixXd> poses
              = rollout->getPosesConst("identity");
          const Eigen::Ref<const Eigen::MatrixXd> vels
              = rollout->getVelsConst("identity");
          const Eigen::Ref<const Eigen::MatrixXd> forces
              = rollout->getForcesConst("identity");

          gradWrtRollout->getPoses("identity")(1, poses.cols() - 1)
              = 2 * poses(1, poses.cols() - 1);
          double endPos = poses(1, poses.cols() - 1);
          double endPosLoss = -(endPos * endPos) * (endPos > 0 ? 1.0 : -1.0);
          return endPosLoss;
        };

  // Make a huge timestep, to try to make the gradients easier to get exactly
  // for finite differencing
  world->setTimeStep(1e-3);

  world->setPenetrationCorrectionEnabled(false);
  world->setConstraintForceMixingEnabled(false);

  /*
  // Initial pos that creates deep inter-penetration and generates larger
  // gradient errors
  Eigen::VectorXd initialPos = Eigen::VectorXd(5);
  initialPos << 0.96352, -0.5623, -0.0912082, 0.037308, 0.147683;
  // Initial vel
  Eigen::VectorXd initialVel = Eigen::VectorXd(5);
  initialVel << 0.110462, 0.457093, 0.257748, 0.592256, 0.167432;

  world->setPositions(initialPos);
  world->setVelocities(initialVel);
  */

  /*
  EXPECT_TRUE(verifyVelGradients(world, world->getVelocities()));
  EXPECT_TRUE(verifyNoMultistepIntereference(world, 10));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  */
  // renderWorld(world);

  LossFn lossFn(loss);
  MultiShot shot(world, lossFn, 100, 20, false);
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addLinearBodyNode(root);
  shot.addMapping("ik", ikMap);

  shot.setParallelOperationsEnabled(true);

  EXPECT_TRUE(verifyMultiShotOptimization(world, shot));
}