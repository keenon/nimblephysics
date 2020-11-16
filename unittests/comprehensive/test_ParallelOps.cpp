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
#include <thread>

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
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;

bool checkOutOfOrderBackprop(WorldPtr world)
{
  WorldPtr world1 = world->clone();
  WorldPtr world2 = world->clone();

  const int STEPS = 200;

  std::vector<std::shared_ptr<BackpropSnapshot>> snapshots1;
  for (int i = 0; i < STEPS; i++)
  {
    snapshots1.push_back(neural::forwardPass(world1));
  }

  std::vector<std::shared_ptr<BackpropSnapshot>> snapshots2;
  for (int i = 0; i < STEPS; i++)
  {
    snapshots2.push_back(neural::forwardPass(world2));
  }

  // Run the first set forward

  for (int i = 0; i < STEPS; i++)
  {
    snapshots1[i]->getPosPosJacobian(world1);
    snapshots1[i]->getVelPosJacobian(world1);
    snapshots1[i]->getPosVelJacobian(world1);
    snapshots1[i]->getForceVelJacobian(world1);
    snapshots1[i]->getVelVelJacobian(world1);
  }

  // Run the second set backwards

  for (int i = STEPS - 1; i >= 0; i--)
  {
    snapshots2[i]->getPosPosJacobian(world2);
    snapshots2[i]->getVelPosJacobian(world2);
    snapshots2[i]->getPosVelJacobian(world2);
    snapshots2[i]->getForceVelJacobian(world2);
    snapshots2[i]->getVelVelJacobian(world2);
  }

  ///////////////////////////////////////////////
  // Test that everything is EXACTLY equal
  ///////////////////////////////////////////////

  for (int i = 0; i < STEPS; i++)
  {
    if (!equals(
            snapshots1[i]->getPosPosJacobian(world1),
            snapshots2[i]->getPosPosJacobian(world2),
            0.0))
    {
      std::cout << "Off on pos-pos Jac at step " << i << std::endl;
      return false;
    }
    if (!equals(
            snapshots1[i]->getVelPosJacobian(world1),
            snapshots2[i]->getVelPosJacobian(world2),
            0.0))
    {
      std::cout << "Off on vel-pos Jac at step " << i << std::endl;
      return false;
    }
    if (!equals(
            snapshots1[i]->getPosVelJacobian(world1),
            snapshots2[i]->getPosVelJacobian(world2),
            0.0))
    {
      std::cout << "Off on pos-vel Jac at step " << i << std::endl;
      return false;
    }
    if (!equals(
            snapshots1[i]->getVelVelJacobian(world1),
            snapshots2[i]->getVelVelJacobian(world2),
            0.0))
    {
      std::cout << "Off on vel-vel Jac at step " << i << std::endl;
      return false;
    }
    if (!equals(
            snapshots1[i]->getForceVelJacobian(world1),
            snapshots2[i]->getForceVelJacobian(world2),
            0.0))
    {
      std::cout << "Off on force-vel Jac at step " << i << std::endl;
      return false;
    }
  }
  return true;
}

TEST(WEB, SIMPLE_BOX)
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

  ///////////////////////////////////////////////////
  // Run the forward passes
  ///////////////////////////////////////////////////

  EXPECT_TRUE(checkOutOfOrderBackprop(world));
}

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
    double peakPosLoss = -(maxPos * maxPos) * (maxPos > 0 ? 1.0 : -1.0);
    double minPosLoss = -(minPos * minPos) * (minPos > 0 ? 1.0 : -1.0);
    double endPos = poses(1, poses.cols() - 1);
    double endPosLoss = -(endPos * endPos) * (endPos > 0 ? 1.0 : -1.0);

    double forceLoss = forces.squaredNorm();

    // return endPosLoss * 100 + forceLoss * 1e-3;
    // return forceLoss;
    return endPosLoss; // + forceLoss;
    // return (100 * peakPosLoss) + (20 * minPosLoss) + endPosLoss;
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

  EXPECT_TRUE(checkOutOfOrderBackprop(world));

  LossFn lossFn(loss);
  MultiShot shot(world, lossFn, 200, 20, false);
  shot.setParallelOperationsEnabled(false);
  std::shared_ptr<IKMapping> ikMap = std::make_shared<IKMapping>(world);
  ikMap->addLinearBodyNode(root);
  shot.addMapping("ik", ikMap);

  MultiShot shot2(world, lossFn, 200, 20, false);
  shot2.setParallelOperationsEnabled(true);
  shot2.addMapping("ik", ikMap);

  IPOptOptimizer optimizer = IPOptOptimizer();

  optimizer.setIterationLimit(50);
  std::shared_ptr<Solution> record = optimizer.optimize(&shot);

  // shot.setParallelOperationsEnabled(true);
  std::shared_ptr<Solution> record2 = optimizer.optimize(&shot2);

  double loss1 = shot.getLoss(world);
  double loss2 = shot2.getLoss(world);

  EXPECT_DOUBLE_EQ(loss1, loss2);

  std::vector<MappedBackpropSnapshotPtr> snapshots1 = shot.getSnapshots(world);
  std::vector<MappedBackpropSnapshotPtr> snapshots2 = shot2.getSnapshots(world);

  for (int i = 0; i < 102; i++)
  {
    BackpropSnapshotPtr b1 = snapshots1[i]->getUnderlyingSnapshot();
    BackpropSnapshotPtr b2 = snapshots2[i]->getUnderlyingSnapshot();
    if (!equals(b1->getPreStepPosition(), b2->getPreStepPosition(), 0.0))
    {
      std::cout << "Off on pre-step pos at step " << i << std::endl;
    }
    if (!equals(b1->getPreStepVelocity(), b2->getPreStepVelocity(), 0.0))
    {
      std::cout << "Off on pre-step vel at step " << i << std::endl;
    }
    if (!equals(b1->getPreStepTorques(), b2->getPreStepTorques(), 0.0))
    {
      std::cout << "Off on pre-step torques at step " << i << std::endl;
    }
    if (!equals(b1->getPostStepPosition(), b2->getPostStepPosition(), 0.0))
    {
      std::cout << "Off on post-step pos at step " << i << std::endl;
    }
    if (!equals(b1->getPostStepVelocity(), b2->getPostStepVelocity(), 0.0))
    {
      std::cout << "Off on post-step vel at step " << i << std::endl;
    }
    if (!equals(b1->getPostStepTorques(), b2->getPostStepTorques(), 0.0))
    {
      std::cout << "Off on post-step torques at step " << i << std::endl;
    }
    if (!equals(
            b1->getClampingConstraintImpulses(),
            b2->getClampingConstraintImpulses(),
            0.0))
    {
      std::cout << "Off on clamping constraint impulses at step " << i
                << std::endl;
    }
    if (!equals(
            b1->getPosPosJacobian(world), b2->getPosPosJacobian(world), 0.0))
    {
      std::cout << "Off on pos-pos Jac at step " << i << std::endl;
    }
    if (!equals(
            b1->getVelPosJacobian(world), b2->getVelPosJacobian(world), 0.0))
    {
      std::cout << "Off on vel-pos Jac at step " << i << std::endl;
    }
    if (!equals(
            snapshots1[i]->getUnderlyingSnapshot()->getPosVelJacobian(world),
            snapshots2[i]->getUnderlyingSnapshot()->getPosVelJacobian(world),
            0.0))
    {
      std::cout << "Off on pos-vel Jac at step " << i << std::endl;

      RestorableSnapshot snapshot(world);
      world->setPositions(b1->getPreStepPosition());
      world->setVelocities(b1->getPreStepVelocity());
      world->setExternalForces(b1->getPreStepTorques());

      Eigen::MatrixXd A_c1 = b1->getClampingConstraintMatrix(world);
      Eigen::MatrixXd A_ub1 = b1->getUpperBoundConstraintMatrix(world);
      Eigen::MatrixXd E1 = b1->getUpperBoundMappingMatrix();
      Eigen::MatrixXd A_c_ub_E1 = A_c1 + A_ub1 * E1;

      Eigen::MatrixXd A_c2 = b2->getClampingConstraintMatrix(world);
      Eigen::MatrixXd A_ub2 = b2->getUpperBoundConstraintMatrix(world);
      Eigen::MatrixXd E2 = b2->getUpperBoundMappingMatrix();
      Eigen::MatrixXd A_c_ub_E2 = A_c2 + A_ub2 * E2;

      if (!equals(A_c_ub_E1, A_c_ub_E2, 0.0))
      {
        std::cout << "   A_c_ub_E off" << std::endl;
      }

      Eigen::VectorXd tau = world->getExternalForces();
      Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
      Eigen::VectorXd f_c1 = b1->getClampingConstraintImpulses();
      Eigen::VectorXd f_c2 = b2->getClampingConstraintImpulses();
      double dt = world->getTimeStep();

      Eigen::MatrixXd dM1 = b1->getJacobianOfMinv(
          world, dt * (tau - C) + A_c_ub_E1 * f_c1, WithRespectTo::POSITION);
      Eigen::MatrixXd dM2 = b2->getJacobianOfMinv(
          world, dt * (tau - C) + A_c_ub_E2 * f_c2, WithRespectTo::POSITION);

      if (!equals(dM1, dM2, 0.0))
      {
        std::cout << "   dM off" << std::endl;
      }

      Eigen::MatrixXd Minv = world->getInvMassMatrix();
      Eigen::MatrixXd dC1 = b1->getJacobianOfC(world, WithRespectTo::POSITION);
      Eigen::MatrixXd dC2 = b2->getJacobianOfC(world, WithRespectTo::POSITION);

      if (!equals(dC1, dC2, 0.0))
      {
        std::cout << "   dC off" << std::endl;
      }

      Eigen::MatrixXd dF_c1
          = b1->getJacobianOfConstraintForce(world, WithRespectTo::POSITION);
      Eigen::MatrixXd dF_c2
          = b2->getJacobianOfConstraintForce(world, WithRespectTo::POSITION);

      std::vector<std::shared_ptr<DifferentiableContactConstraint>> contacts1
          = b1->getDifferentiableConstraints();
      std::vector<std::shared_ptr<DifferentiableContactConstraint>> contacts2
          = b2->getDifferentiableConstraints();

      // Check LCP setup is the same
      if (contacts1.size() != contacts2.size())
      {
        std::cout << "   contact num off: " << contacts1.size() << " vs "
                  << contacts2.size() << std::endl;
      }
      Eigen::MatrixXd mA1 = b1->mGradientMatrices[0]->mA;
      Eigen::MatrixXd mA2 = b2->mGradientMatrices[0]->mA;
      if (!equals(mA1, mA2, 0.0))
      {
        std::cout << "   mA off" << std::endl;
        std::cout << "mA1: " << mA1 << std::endl;
        std::cout << "mA2: " << mA2 << std::endl;
      }
      Eigen::MatrixXd mB1 = b1->mGradientMatrices[0]->mB;
      Eigen::MatrixXd mB2 = b2->mGradientMatrices[0]->mB;
      if (!equals(mB1, mB2, 0.0))
      {
        std::cout << "   mB off" << std::endl;
        std::cout << "mB1: " << mB1 << std::endl;
        std::cout << "mB2: " << mB2 << std::endl;
      }

      // Check LCP solution is the same
      Eigen::VectorXd mX1 = b1->getContactConstraintImpluses();
      Eigen::VectorXd mX2 = b2->getContactConstraintImpluses();
      if (!equals(mX1, mX2, 0.0))
      {
        std::cout << "   mX off" << std::endl;
        std::cout << "mX1: " << mX1 << std::endl;
        std::cout << "mX2: " << mX2 << std::endl;
      }

      if (!equals(dF_c1, dF_c2, 0.0))
      {
        std::cout << "   dF_c off" << std::endl;

        Eigen::MatrixXd Q1 = b1->getClampingAMatrix();
        Eigen::MatrixXd Q2 = b2->getClampingAMatrix();
        if (!equals(Q1, Q2, 0.0))
        {
          std::cout << "      Q off" << std::endl;
        }

        Eigen::MatrixXd dB1 = b1->getJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::POSITION);
        Eigen::MatrixXd dB2 = b2->getJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::POSITION);
        if (!equals(dB1, dB2, 0.0))
        {
          std::cout << "      dB off" << std::endl;

          Eigen::MatrixXd Minv1 = b1->getInvMassMatrix(world);
          Eigen::MatrixXd Minv2 = b2->getInvMassMatrix(world);
          if (!equals(Minv1, Minv2, 0.0))
          {
            std::cout << "         Minv off" << std::endl;
          }

          Eigen::MatrixXd dC1
              = b1->getJacobianOfC(world, WithRespectTo::POSITION);
          Eigen::MatrixXd dC2
              = b2->getJacobianOfC(world, WithRespectTo::POSITION);
          if (!equals(dC1, dC2, 0.0))
          {
            std::cout << "         dC off" << std::endl;
          }

          Eigen::VectorXd C1 = b1->getCoriolisAndGravityForces();
          Eigen::VectorXd C2 = b2->getCoriolisAndGravityForces();
          if (!equals(C1, C2, 0.0))
          {
            std::cout << "         C off" << std::endl;
          }

          Eigen::VectorXd f1 = b1->getPreStepTorques() - C1;
          Eigen::VectorXd f2 = b1->getPreStepTorques() - C2;
          if (!equals(f1, f2, 0.0))
          {
            std::cout << "         f off" << std::endl;
          }

          Eigen::MatrixXd dMinv_f1
              = b1->getJacobianOfMinv(world, f1, WithRespectTo::POSITION);
          Eigen::MatrixXd dMinv_f2
              = b2->getJacobianOfMinv(world, f2, WithRespectTo::POSITION);
          if (!equals(dMinv_f1, dMinv_f2, 0.0))
          {
            std::cout << "         dMinv_f off" << std::endl;
          }

          Eigen::VectorXd v_f1 = b1->getPreConstraintVelocity();
          Eigen::VectorXd v_f2 = b2->getPreConstraintVelocity();
          if (!equals(v_f1, v_f2, 0.0))
          {
            std::cout << "         v_f off" << std::endl;
          }

          Eigen::MatrixXd dA_c_f1
              = b1->getJacobianOfClampingConstraintsTranspose(world, v_f1);
          Eigen::MatrixXd dA_c_f2
              = b2->getJacobianOfClampingConstraintsTranspose(world, v_f2);
          if (!equals(dA_c_f1, dA_c_f2, 0.0))
          {
            std::cout << "         dA_c_f off" << std::endl;

            std::vector<std::shared_ptr<DifferentiableContactConstraint>> con1
                = b1->getClampingConstraints();
            std::vector<std::shared_ptr<DifferentiableContactConstraint>> con2
                = b2->getClampingConstraints();
            if (con1.size() != con2.size())
            {
              std::cout << "            different num clamping! " << con1.size()
                        << " vs " << con2.size() << std::endl;
            }

            for (int i = 0; i < con1.size(); i++)
            {
              Eigen::VectorXd col1
                  = con1[i]->getConstraintForcesJacobian(world).transpose()
                    * v_f1;
              Eigen::VectorXd col2
                  = con2[i]->getConstraintForcesJacobian(world).transpose()
                    * v_f2;
              if (!equals(col1, col2, 0.0))
              {
                std::cout << "            constraint forces jac off at row "
                          << i << std::endl;

                int dim = world->getNumDofs();

                math::Jacobian forceJac1
                    = con1[i]->getContactForceJacobian(world);
                math::Jacobian forceJac2
                    = con2[i]->getContactForceJacobian(world);
                if (!equals(forceJac1, forceJac2, 0.0))
                {
                  std::cout << "               forceJac off" << std::endl;
                  std::cout << "forceJac1: " << forceJac1 << std::endl;
                  std::cout << "forceJac2: " << forceJac2 << std::endl;

                  math::LinearJacobian posJac1
                      = con1[i]->getContactPositionJacobian(world);
                  math::LinearJacobian posJac2
                      = con2[i]->getContactPositionJacobian(world);
                  if (!equals(posJac1, posJac2, 0.0))
                  {
                    std::cout << "               posJac off" << std::endl;
                    std::cout << "posJac1: " << posJac1 << std::endl;
                    std::cout << "posJac2: " << posJac2 << std::endl;
                  }

                  math::LinearJacobian dirJac1
                      = con1[i]->getContactForceDirectionJacobian(world);
                  math::LinearJacobian dirJac2
                      = con2[i]->getContactForceDirectionJacobian(world);
                  if (!equals(dirJac1, dirJac2, 0.0))
                  {
                    std::cout << "               dirJac off" << std::endl;
                    std::cout << "dirJac1: " << dirJac1 << std::endl;
                    std::cout << "dirJac2: " << dirJac2 << std::endl;
                  }
                }

                Eigen::Vector6d force1 = con1[i]->getWorldForce();
                Eigen::Vector6d force2 = con2[i]->getWorldForce();
                if (!equals(force1, force2, 0.0))
                {
                  std::cout << "               force off" << std::endl;
                }

                Eigen::MatrixXd result1 = Eigen::MatrixXd::Zero(dim, dim);
                Eigen::MatrixXd result2 = Eigen::MatrixXd::Zero(dim, dim);

                std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
                for (int row = 0; row < dim; row++)
                {
                  Eigen::Vector6d axis1 = con1[i]->getWorldScrewAxis(dofs[row]);
                  Eigen::Vector6d axis2 = con2[i]->getWorldScrewAxis(dofs[row]);
                  if (!equals(axis1, axis2, 0.0))
                  {
                    std::cout << "               axis @ " << row << " off"
                              << std::endl;
                  }

                  for (int wrt = 0; wrt < dim; wrt++)
                  {
                    Eigen::Vector6d screwAxisGradient1
                        = con1[i]->getScrewAxisGradient(dofs[row], dofs[wrt]);
                    Eigen::Vector6d screwAxisGradient2
                        = con2[i]->getScrewAxisGradient(dofs[row], dofs[wrt]);
                    if (!equals(screwAxisGradient1, screwAxisGradient2, 0.0))
                    {
                      std::cout << "               axis grad @ " << row << ", "
                                << wrt << " off" << std::endl;
                    }

                    Eigen::Vector6d forceGradient1 = forceJac1.col(wrt);
                    Eigen::Vector6d forceGradient2 = forceJac2.col(wrt);

                    if (!equals(forceGradient1, forceGradient2, 0.0))
                    {
                      std::cout << "               force grad @ " << row << ", "
                                << wrt << " off" << std::endl;
                    }

                    double multiple1 = con1[i]->getForceMultiple(dofs[row]);
                    double multiple2 = con2[i]->getForceMultiple(dofs[row]);

                    if (multiple1 != multiple2)
                    {
                      std::cout << "               multiple @ " << row << ", "
                                << wrt << " off" << std::endl;
                    }

                    result1(row, wrt) = multiple1
                                        * (screwAxisGradient1.dot(force1)
                                           + axis1.dot(forceGradient1));
                    result2(row, wrt) = multiple2
                                        * (screwAxisGradient2.dot(force2)
                                           + axis2.dot(forceGradient2));
                  }
                }
              }
            }
          }
        }

        Eigen::VectorXd b_1 = b1->getClampingConstraintRelativeVels();
        Eigen::VectorXd b_2 = b2->getClampingConstraintRelativeVels();
        if (!equals(b_1, b_2, 0.0))
        {
          std::cout << "      b off" << std::endl;
        }

        Eigen::MatrixXd dQ_b1
            = b1->getJacobianOfLCPConstraintMatrixClampingSubset(
                world, b_1, WithRespectTo::POSITION);
        Eigen::MatrixXd dQ_b2
            = b2->getJacobianOfLCPConstraintMatrixClampingSubset(
                world, b_2, WithRespectTo::POSITION);
        if (!equals(dQ_b1, dQ_b2, 0.0))
        {
          std::cout << "      dQ_b off" << std::endl;
        }
      }
    }
    if (!equals(
            snapshots1[i]->getUnderlyingSnapshot()->getVelVelJacobian(world),
            snapshots2[i]->getUnderlyingSnapshot()->getVelVelJacobian(world),
            0.0))
    {
      std::cout << "Off on vel-vel Jac at step " << i << std::endl;
    }
    if (!equals(
            snapshots1[i]->getUnderlyingSnapshot()->getForceVelJacobian(world),
            snapshots2[i]->getUnderlyingSnapshot()->getForceVelJacobian(world),
            0.0))
    {
      std::cout << "Off on force-vel Jac at step " << i << std::endl;
    }
  }
}