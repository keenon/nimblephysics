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
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "dart/realtime/MPC.hpp"
#include "dart/realtime/RealtimeWorld.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/LossFn.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;

#ifdef ALL_TESTS
TEST(REALTIME, REALTIME_CARTPOLE)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

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

  std::shared_ptr<LossFn> lossFn = std::make_shared<LossFn>(loss, lossGrad);

  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 20 fps, in the hope that this helps us keep up
  world->setTimeStep(1.0 / 20);

  int planningHorizonMillis = 1000;

  MPC mpc = MPC(world, lossFn, planningHorizonMillis);

  std::function<Eigen::VectorXd()> getForces
      = [&]() { return mpc.getForceNow(); };
  std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
      recordState
      = [&](Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass) {
          mpc.recordGroundTruthStateNow(pos, vel, mass);
        };
  RealtimeWorld realtimeWorld = RealtimeWorld(world, getForces, recordState);

  realtimeWorld.start();
  mpc.start();

  while (true)
  {
    // spin
  }
}
#endif
