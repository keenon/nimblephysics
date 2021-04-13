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
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
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

#ifdef DART_USE_ARBITRARY_PRECISION
#include "mpreal.h"
#endif

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, NUMERICAL_INSTABILITY)
{
  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(9);
  brokenPos << -5.15992, -0.210083, 8.27897, -0.00318367, 0.513758, -0.0286844,
      0.587853, 0.0282165, 0.486934;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(9);
  brokenVel << -2.22286, -6.07728, 0.890211, 9.22385, 3.00743, 1.34837, 9.61029,
      -2.97553, 9.96726;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(9);
  brokenForce << 0, 0, 3.15531, -8.41413, -6.37498, -5.82503, -7.75906, 9.98554,
      7.19786;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(0);
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  EXPECT_TRUE(verifyF_c(world));
  EXPECT_TRUE(verifyIdentityMapping(world));
}
#endif

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, BROKEN_POINT)
{
  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(9);
  brokenPos << 3.21866, -0.465303, 5.9565, -0.487295, -0.969093, 0.0792724,
      -0.235988, -0.109183, 0.0769134;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(9);
  brokenVel << 4.32527, -4.96436, 12.677, 6.0086, -21.5535, 6.76575, 25.5498,
      -86.1003, 38.0762;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(9);
  brokenForce << -0, -0, -0.398301, -2.15491, -6.51447, 23.0772, 7.68141,
      -30.2488, 9.71529;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(3);
  brokenLCPCache << 11.9108, 11.8521, 0;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  EXPECT_TRUE(verifyF_c(world));
  EXPECT_TRUE(verifyIdentityMapping(world));
}
#endif

// #ifdef ALL_TESTS
TEST(HALF_CHEETAH, CAPSULE_INTER_PENETRATION)
{
  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(9);
  brokenPos << -8.71924, -0.0564965, -7.58459, 0.291652, -0.587514, 0.200843,
      0.255774, 0.802592, -0.201628;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(9);
  brokenVel << 2.85676, 8.95689, -1.45436, -8.77701, -3.59597, -2.06459,
      -7.10815, 3.59627, -20.2623;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(9);
  brokenForce << 0, 0, 0.193621, -9.13741, 9.79953, 5.09491, -1.46957, -8.3705,
      -9.74503;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(0);
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  // world->step();
  // world->setPositions(brokenPos);

  // EXPECT_TRUE(verifyAnalyticalA_ubJacobian(world));
  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyF_c(world));

  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  server.blockWhileServing();
  */

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  EXPECT_TRUE(verifyF_c(world));
  EXPECT_TRUE(verifyIdentityMapping(world));
}
// #endif

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, POS_VEL_ERRORS)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(9);
  brokenPos << -8.71924, -0.0564965, -7.58459, 0.291652, -0.587514, 0.200843,
      0.255774, 0.802592, -0.201628;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(9);
  brokenVel << 2.85676, 8.95689, -1.45436, -8.77701, -3.59597, -2.06459,
      -7.10815, 3.59627, -20.2623;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(9);
  brokenForce << 0, 0, 0.193621, -9.13741, 9.79953, 5.09491, -1.46957, -8.3705,
      -9.74503;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(0);
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  // world->step();
  // world->setPositions(brokenPos);

  // EXPECT_TRUE(verifyAnalyticalA_ubJacobian(world));
  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyF_c(world));

  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  server.blockWhileServing();
  */

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  EXPECT_TRUE(verifyF_c(world));
  EXPECT_TRUE(verifyIdentityMapping(world));
}
#endif

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, POS_VEL_ERRORS_2)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(9);
  brokenPos << -0.0442559, -0.204541, -0.00676443, -0.0591177, -0.0841678,
      -0.360202, -0.224704, -0.102217, -0.0377656;
  Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(9);
  brokenVel << -0.119657, -0.571126, -0.0161144, -0.257376, -0.0404545,
      -0.431694, -0.418564, -0.279473, -0.24886;
  Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero(9);
  brokenForce << -0, -0, -0.257804, -0.352182, -0.114889, -0.0418478, -0.261154,
      -0.208821, -0.0516623;
  Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero(0);
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setControlForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  // world->step();
  // world->setPositions(brokenPos);

  // EXPECT_TRUE(verifyAnalyticalA_ubJacobian(world));
  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyF_c(world));

  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);
  server.blockWhileServing();
  */

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  EXPECT_TRUE(verifyF_c(world));
  EXPECT_TRUE(verifyIdentityMapping(world));
}
#endif

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, FULL_TEST)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);

  for (auto* dof : world->getDofs())
  {
    std::cout << "DOF: " << dof->getName() << std::endl;
  }

  Eigen::VectorXs forceLimits
      = Eigen::VectorXs::Ones(world->getNumDofs()) * 100;
  forceLimits(0) = 0;
  forceLimits(1) = 0;
  world->setControlForceUpperLimits(forceLimits);
  world->setControlForceLowerLimits(-1 * forceLimits);

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  // Create target

  s_t target_x = 0.5;
  s_t target_y = 0.5;

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3s targetOffset = Eigen::Isometry3s::Identity();
  targetOffset.translation() = Eigen::Vector3s(target_x, target_y, 0.0);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3s(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  trajectory::LossFn loss(
      [target_x, target_y](const trajectory::TrajectoryRollout* rollout) {
        const Eigen::VectorXs lastPos
            = rollout->getPosesConst().col(rollout->getPosesConst().cols() - 1);

        s_t diffX = lastPos(0) - target_x;
        s_t diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      });

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 400, 10, false);
  trajectory->setParallelOperationsEnabled(false);

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  // optimizer.setCheckDerivatives(true);
  optimizer.setIterationLimit(500);
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

  int i = 0;
  const Eigen::MatrixXs poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXs vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  Ticker ticker(0.1);
  ticker.registerTickListener([&](long /* time */) {
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

  /*
  Eigen::VectorXs forceLimits = Eigen::VectorXs::Ones(atlas->getNumDofs()) * 30;
  forceLimits.segment<6>(0).setZero();
  atlas->setControlForceUpperLimits(forceLimits);
  atlas->setControlForceLowerLimits(forceLimits * -1);
  Eigen::VectorXs posLimits = Eigen::VectorXs::Ones(atlas->getNumDofs()) * 200;
  atlas->setPositionUpperLimits(posLimits);
  atlas->setPositionLowerLimits(posLimits * -1);
  Eigen::VectorXs velLimits = Eigen::VectorXs::Ones(atlas->getNumDofs()) * 1000;
  atlas->setVelocityUpperLimits(velLimits);
  atlas->setVelocityLowerLimits(velLimits * -1);

  // Create target

  s_t target_x = 0.5;
  s_t target_y = 0.5;
  s_t target_z = 0.0;

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

  trajectory::LossFn loss([target_x, target_y, target_z](
                              const trajectory::TrajectoryRollout* rollout) {
    const Eigen::VectorXs lastPos = rollout->getPosesConst("ik").col(
        rollout->getPosesConst("ik").cols() - 1);

    s_t diffX = lastPos(0) - target_x;
    s_t diffY = lastPos(1) - target_y;
    s_t diffZ = lastPos(2) - target_z;

    return diffX * diffX + diffY * diffY + diffZ * diffZ;
  });

  std::shared_ptr<neural::IKMapping> ikMapping
      = std::make_shared<neural::IKMapping>(world);
  // ikMapping->addLinearBodyNode(atlas->getBodyNode(0));
  ikMapping->addLinearBodyNode(atlas->getBodyNode("l_hand"));
  // atlas->getBodyNode("l_hand")

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 100, 10, false);
  trajectory->addMapping("ik", ikMapping);
  trajectory->setParallelOperationsEnabled(true);

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXs atlasPos = world->getPositions();
  */
  /*
  for (int i = 0; i < 1000; i++)
  {
    Eigen::VectorXs pos = ikMapping->getPositions(world);
    Eigen::MatrixXs jac = ikMapping->getMappedPosToRealPosJac(world);
    s_t diffX = pos(0) - target_x;
    s_t diffY = pos(1) - target_y;

    Eigen::Vector3s grad = Eigen::Vector3s(2.0 * diffX, 2.0 * diffY, 0.0);
    s_t loss = diffX * diffX + diffY * diffY;

    Eigen::VectorXs posDiff = jac * grad;
    atlasPos -= posDiff * 0.001;
    world->setPositions(atlasPos);
    server.renderWorld(world);
  }
  */

  /*
  server.renderBasis(
      10.0, "basis", Eigen::Vector3s::Zero(), Eigen::Vector3s::Zero());
      */

  /*
  Ticker ticker(0.1);
  ticker.registerTickListener([&](long time) {
    Eigen::VectorXs pos = ikMapping->getPositions(world);
    Eigen::MatrixXs jac = ikMapping->getMappedPosToRealPosJac(world);
    s_t diffX = pos(0) - target_x;
    s_t diffY = pos(1) - target_y;
    s_t diffZ = pos(2) - target_z;

    Eigen::Vector3s grad
        = Eigen::Vector3s(2.0 * diffX, 2.0 * diffY, 2.0 * diffZ);
    s_t loss = diffX * diffX + diffY * diffY + diffZ * diffZ;

    Eigen::VectorXs posDiff = jac * grad;
    posDiff.segment(0, 6).setZero();
    atlasPos -= posDiff * 0.01;
    world->setPositions(atlasPos);
    server.renderWorld(world);
  });
  */

  // server.registerConnectionListener([&]() { ticker.start(); });

  /*
  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(false);
  optimizer.setIterationLimit(500);
  optimizer.registerIntermediateCallback(
      [&](trajectory::Problem* problem, int step, s_t primal, s_t dual) {
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

  int i = 0;
  const Eigen::MatrixXs poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXs vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  Ticker ticker(0.1);
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
#endif
