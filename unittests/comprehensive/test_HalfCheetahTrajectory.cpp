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

#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

#ifdef ALL_TESTS
TEST(HALF_CHEETAH, FULL_TEST)
{
  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  world->setSlowDebugResultsAgainstFD(true);

  for (auto* dof : world->getDofs())
  {
    std::cout << "DOF: " << dof->getName() << std::endl;
  }

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  // Create target

  double target_x = 0.5;
  double target_y = 0.5;

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3d targetOffset = Eigen::Isometry3d::Identity();
  targetOffset.translation() = Eigen::Vector3d(target_x, target_y, 0.0);
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

        double diffX = lastPos(3) - target_x;
        double diffY = lastPos(4) - target_y;

        return diffX * diffX + diffY * diffY;
      });

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 100, 10, false);
  trajectory->setParallelOperationsEnabled(true);

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(true);
  optimizer.setIterationLimit(500);
  optimizer.registerIntermediateCallback([&](trajectory::Problem* problem,
                                             int /* step */,
                                             double /* primal */,
                                             double /* dual */) {
    const Eigen::MatrixXd poses
        = problem->getRolloutCache(world)->getPosesConst();
    const Eigen::MatrixXd vels
        = problem->getRolloutCache(world)->getVelsConst();
    std::cout << "Rendering trajectory lines" << std::endl;
    server.renderTrajectoryLines(world, poses);
    world->setPositions(poses.col(0));
    server.renderWorld(world);
    return true;
  });
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  /*
  while (server.isServing())
  {
  }
  */

  /*
  Eigen::VectorXd forceLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 30;
  forceLimits.segment<6>(0).setZero();
  atlas->setForceUpperLimits(forceLimits);
  atlas->setForceLowerLimits(forceLimits * -1);
  Eigen::VectorXd posLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 200;
  atlas->setPositionUpperLimits(posLimits);
  atlas->setPositionLowerLimits(posLimits * -1);
  Eigen::VectorXd velLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 1000;
  atlas->setVelocityUpperLimits(velLimits);
  atlas->setVelocityLowerLimits(velLimits * -1);

  // Create target

  double target_x = 0.5;
  double target_y = 0.5;
  double target_z = 0.0;

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3d targetOffset = Eigen::Isometry3d::Identity();
  targetOffset.translation() = Eigen::Vector3d(target_x, target_y, target_z);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3d(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  trajectory::LossFn loss([target_x, target_y, target_z](
                              const trajectory::TrajectoryRollout* rollout) {
    const Eigen::VectorXd lastPos = rollout->getPosesConst("ik").col(
        rollout->getPosesConst("ik").cols() - 1);

    double diffX = lastPos(0) - target_x;
    double diffY = lastPos(1) - target_y;
    double diffZ = lastPos(2) - target_z;

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

  Eigen::VectorXd atlasPos = world->getPositions();
  */
  /*
  for (int i = 0; i < 1000; i++)
  {
    Eigen::VectorXd pos = ikMapping->getPositions(world);
    Eigen::MatrixXd jac = ikMapping->getMappedPosToRealPosJac(world);
    double diffX = pos(0) - target_x;
    double diffY = pos(1) - target_y;

    Eigen::Vector3d grad = Eigen::Vector3d(2.0 * diffX, 2.0 * diffY, 0.0);
    double loss = diffX * diffX + diffY * diffY;

    Eigen::VectorXd posDiff = jac * grad;
    atlasPos -= posDiff * 0.001;
    world->setPositions(atlasPos);
    server.renderWorld(world);
  }
  */

  /*
  server.renderBasis(
      10.0, "basis", Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      */

  /*
  Ticker ticker(0.1);
  ticker.registerTickListener([&](long time) {
    Eigen::VectorXd pos = ikMapping->getPositions(world);
    Eigen::MatrixXd jac = ikMapping->getMappedPosToRealPosJac(world);
    double diffX = pos(0) - target_x;
    double diffY = pos(1) - target_y;
    double diffZ = pos(2) - target_z;

    Eigen::Vector3d grad
        = Eigen::Vector3d(2.0 * diffX, 2.0 * diffY, 2.0 * diffZ);
    double loss = diffX * diffX + diffY * diffY + diffZ * diffZ;

    Eigen::VectorXd posDiff = jac * grad;
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
      [&](trajectory::Problem* problem, int step, double primal, double dual) {
        const Eigen::MatrixXd poses
            = problem->getRolloutCache(world)->getPosesConst();
        const Eigen::MatrixXd vels
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
  const Eigen::MatrixXd poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXd vels
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
