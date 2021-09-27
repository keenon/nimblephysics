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
#include "dart/utils/UniversalLoader.hpp"
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

std::shared_ptr<simulation::World> createWorld(
    s_t target_x, s_t target_y, s_t target_z)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0.0));

  std::shared_ptr<dynamics::Skeleton> KR5
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/urdf/KR5/KR5 sixx R650.urdf");

  world->setPositionUpperLimits(Eigen::VectorXs::Ones(world->getNumDofs()) * 5);
  world->setPositionLowerLimits(
      Eigen::VectorXs::Ones(world->getNumDofs()) * -5);

  world->setControlForceUpperLimits(
      Eigen::VectorXs::Ones(world->getNumDofs()) * 20);
  world->setControlForceLowerLimits(
      Eigen::VectorXs::Ones(world->getNumDofs()) * -20);

  world->setVelocityUpperLimits(
      Eigen::VectorXs::Ones(world->getNumDofs()) * 20);
  world->setVelocityLowerLimits(
      Eigen::VectorXs::Ones(world->getNumDofs()) * -20);

  // Create target

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3s targetOffset = Eigen::Isometry3s::Identity();
  targetOffset.translation() = Eigen::Vector3s(target_x, target_y, target_z);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3s(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  return world;
}

#ifdef ALL_TESTS
TEST(KR5_EXAMPLE, BROKEN_POINT)
{
  s_t target_x = 2.2;
  s_t target_y = 2.2;
  s_t target_z = 2.2;

  // Create a world
  std::shared_ptr<simulation::World> world
      = createWorld(target_x, target_y, target_z);

  /*
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
  brokenPos << -7.4747, 9.43449, 2.12166, 2.98394, 2.34673;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
  brokenVel << -2.84978, 1.03633, 0, 9.16668, 6.99675;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
  brokenForce << 0, 0, -2.11163, -2.06504, -1.3781;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(12);
  brokenLCPCache << 0.0173545, 0.0132076, 0, 0.0173545, 0.0132076, 0, 0, 0, 0,
      0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */
  /*
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
  brokenPos << 8.75823, -1.33554, 1.60919, 0.367526, 1.09027;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
  brokenVel << 4.48639, -5.53436, 1.73472e-18, -1.03812e-17, -0.472044;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
  brokenForce << 0, 0, 9.428, -1.14176, 0.947147;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(6);
  brokenLCPCache << 0.0491903, 0.00921924, 0, 0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */
  /*
  ///
  /// This used to fail to standardize the LCP properly
  ///

  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
  brokenPos << 8.75828, -1.33554, 1.6092, 0.367528, 1.09028;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
  brokenVel << 4.48642, -5.53436, -1.73472e-18, -2.35814e-18, -0.472011;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
  brokenForce << 0, 0, 9.428, -1.14176, 0.947147;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(6);
  brokenLCPCache << 0.0245947, 0.00461058, 0, 0.0245947, 0.00461058, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */

  /*
  ///
  /// This used to fail on edge-edge gradients being computed incorrectly
  ///
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
  brokenPos << -0.13334, -0.178891, 1.07272, 0.130007, 0.436478;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
  brokenVel << -0.433131, -0.847734, 2.55373, -1.13021, -1.61568;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
  brokenForce << 0, 0, -0.17232, 6.83192, -0.275112;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(12);
  brokenLCPCache << 0, 0, 0, 0, 0, 0, 1.0778, 0.330749, 0, 1.0778, 0.330749, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);
  */

  ///
  /// This used to fail on CLAMPING_THRESHOLD being too large in
  /// ConstraintGroupGradientMatrices.cpp
  ///
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(5);
  brokenPos << -0.000646825, -0.0351094, 0.759088, 0.102786, 0.731049;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(5);
  brokenVel << -0.216819, -0.25626, 0.256483, 0.758835, -0.794271;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(5);
  brokenForce << 0, 0, 0.136721, 1.88135, 7.45379;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(12);
  brokenLCPCache << 0.00454883, -5.55535e-05, 0, 0.00454883, -5.55535e-05, 0, 0,
      0, 0, 0, 0, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  // EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  // EXPECT_TRUE(verifyF_c(world));

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXs animatePos = brokenPos;
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
TEST(KR5_EXAMPLE, FULL_TEST)
{
  s_t target_x = 2.2;
  s_t target_y = 2.2;
  s_t target_z = 2.2;

  // Create a world
  std::shared_ptr<simulation::World> world
      = createWorld(target_x, target_y, target_z);
  world->setSlowDebugResultsAgainstFD(true);
  std::shared_ptr<dynamics::Skeleton> KR5 = world->getSkeleton(0);

  TrajectoryLossFn loss = [target_x, target_y, target_z](
                              const trajectory::TrajectoryRollout* rollout) {
    const Eigen::VectorXs lastPos
        = rollout->getPosesConst("ik").col(rollout->getPosesConst().cols() - 1);
    const Eigen::VectorXs lastVel
        = rollout->getVelsConst("ik").col(rollout->getPosesConst().cols() - 1);

    s_t diffX = lastPos(0) - target_x;
    s_t diffY = lastPos(1) - target_y;
    s_t diffZ = lastPos(2) - target_z;

    s_t ikPosLoss = diffX * diffX + diffY * diffY + diffZ * diffZ;
    s_t ikVelLoss = lastVel(0) * lastVel(0) + lastVel(1) * lastVel(1)
                    + lastVel(2) * lastVel(2);

    s_t forcesLoss = 0.0;
    /*
    Eigen::MatrixXs forces = rollout->getVelsConst();
    for (int i = 0; i < forces.cols(); i++)
    {
      for (int j = 0; j < forces.rows(); j++)
      {
        s_t scaling = 0.5 / (forces.rows() * forces.cols());
        forcesLoss += forces(j, i) * forces(j, i) * scaling;
      }
    }
    */

    return ikPosLoss + ikVelLoss + forcesLoss;
  };

  TrajectoryLossFnAndGrad lossGrad
      = [target_x, target_y, target_z](
            const TrajectoryRollout* rollout,
            TrajectoryRollout* gradWrtRollout // OUT
        ) {
          int lastCol = rollout->getPosesConst("ik").cols() - 1;
          const Eigen::VectorXs lastPos = rollout->getPosesConst().col(lastCol);

          // Do the IK loss and gradient

          s_t diffX = lastPos(0) - target_x;
          s_t diffY = lastPos(1) - target_y;
          s_t diffZ = lastPos(2) - target_z;

          gradWrtRollout->getPoses("ik").setZero();
          gradWrtRollout->getPoses("ik")(0, lastCol) = 2 * diffX;
          gradWrtRollout->getPoses("ik")(1, lastCol) = 2 * diffY;
          gradWrtRollout->getPoses("ik")(2, lastCol) = 2 * diffZ;
          gradWrtRollout->getVels("ik").setZero();
          gradWrtRollout->getControlForces("ik").setZero();

          s_t ikLoss = diffX * diffX + diffY * diffY + diffZ * diffZ;

          // Do the ordinary force loss and gradient

          gradWrtRollout->getPoses().setZero();
          gradWrtRollout->getVels().setZero();
          gradWrtRollout->getControlForces().setZero();

          Eigen::MatrixXs forces = rollout->getVelsConst();
          s_t forcesLoss = 0.0;
          for (int i = 0; i < forces.cols(); i++)
          {
            for (int j = 0; j < forces.rows(); j++)
            {
              s_t scaling = 0.5 / (forces.rows() * forces.cols());
              forcesLoss += forces(j, i) * forces(j, i) * scaling;
              gradWrtRollout->getVels()(j, i) = 2 * forces(j, i) * scaling;
            }
          }

          return ikLoss;
        };

  trajectory::LossFn lossObj(loss);

  std::shared_ptr<neural::IKMapping> ikMapping
      = std::make_shared<neural::IKMapping>(world);
  // ikMapping->addLinearBodyNode(atlas->getBodyNode(0));
  ikMapping->addLinearBodyNode(KR5->getBodyNode("palm"));
  // atlas->getBodyNode("l_hand")

  // world->setTimeStep(0.01);

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, lossObj, 300, 10, false);
  trajectory->addMapping("ik", ikMapping);
  // trajectory->setParallelOperationsEnabled(false);
  // int flatProblemDim = trajectory->getFlatProblemDim(world);
  // int staticDim = trajectory->getFlatStaticProblemDim(world);
  // int dynamicDim = trajectory->getFlatDynamicProblemDim(world);
  // trajectory->backpropGradient(world, grad)
  // TrajectoryRollout* rollout = trajectory->getGradientWrtRolloutCache(world);

  /*
  // EXPECT_TRUE(verifyTrajectory(world, trajectory));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyGradientBackprop(
      world,
      500,
      [target_x, target_y](std::shared_ptr<simulation::World> world) {
        const Eigen::VectorXs lastPos = world->getPositions();

        s_t diffX = lastPos(0) - target_x;
        s_t diffY = lastPos(1) - target_y;

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

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(1);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(false);
  optimizer.setIterationLimit(20);
  optimizer.registerIntermediateCallback([&](trajectory::Problem* problem,
                                             int /* step */,
                                             s_t /* primal */,
                                             s_t /* dual */) {
    const Eigen::MatrixXs poses
        = problem->getRolloutCache(world)->getPosesConst();
    const Eigen::MatrixXs vels
        = problem->getRolloutCache(world)->getVelsConst();
    std::cout << "Rendering trajectory lines" << std::endl;
    server.renderTrajectoryLines(world, poses);
    world->setPositions(poses.col(0));
    server.renderWorld(world);
    return true;
  });
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  const Eigen::MatrixXs poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXs vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  /*
  int i = 0;
  Ticker ticker(world->getTimeStep());
  ticker.registerTickListener([&](
    long // time
  ) {
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
