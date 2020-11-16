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

#include <gtest/gtest.h>

#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/RealtimeWorld.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"

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
      new BoxShape(Eigen::Vector3d(0.5, 0.1, 0.1)));
  ShapeNode* sledShape
      = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);
  sledShape->getVisualAspect()->setColor(Eigen::Vector3d(0.5, 0.5, 0.5));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));
  std::shared_ptr<BoxShape> armShapeBox(
      new BoxShape(Eigen::Vector3d(0.1, 1.0, 0.1)));
  ShapeNode* armShape
      = armPair.second->createShapeNodeWith<VisualAspect>(armShapeBox);
  armShape->getVisualAspect()->setColor(Eigen::Vector3d(0.7, 0.7, 0.7));

  Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
  armOffset.translation() = Eigen::Vector3d(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  world->addSkeleton(cartpole);

  cartpole->setForceUpperLimit(0, 15);
  cartpole->setForceLowerLimit(0, -15);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);

  cartpole->setForceUpperLimit(1, 0);
  cartpole->setForceLowerLimit(1, 0);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());
  // cartpole->getDof(1)->setCoulombFriction(0.1);

  double weightPose = 1.0;

  TrajectoryLossFn loss = [weightPose](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst("identity").cols();
    double sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      // rollout->getVelsConst().col(i).squaredNorm()
      sum += rollout->getPosesConst().col(i).squaredNorm() * weightPose;
    }
    return sum;
  };

  TrajectoryLossFnAndGrad lossGrad
      = [weightPose](
            const TrajectoryRollout* rollout,
            TrajectoryRollout* gradWrtRollout // OUT
        ) {
          gradWrtRollout->getPoses().setZero();
          gradWrtRollout->getVels().setZero();
          gradWrtRollout->getForces().setZero();
          int steps = rollout->getPosesConst().cols();
          for (int i = 0; i < steps; i++)
          {
            gradWrtRollout->getPoses().col(i)
                = 2 * rollout->getPosesConst().col(i) * weightPose;
            // gradWrtRollout->getVels().col(i) = 2 *
            // rollout->getVelsConst().col(i);
          }
          /*
          for (int i = 0; i < steps; i++)
          {
            gradWrtRollout->getForces("identity").col(i)
                = 2 * rollout->getForcesConst("identity").col(i);
          }
          */
          double sum = 0.0;
          for (int i = 0; i < steps; i++)
          {
            // rollout->getVelsConst().col(i).squaredNorm()
            sum += rollout->getPosesConst().col(i).squaredNorm() * weightPose;
          }
          return sum;
        };

  std::shared_ptr<LossFn> lossFn = std::make_shared<LossFn>(loss, lossGrad);

  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int planningHorizonMillis = 300 * world->getTimeStep() * 1000;
  int advanceSteps = 70;

  /*
  RestorableSnapshot snapshot(world);

  std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl
            << "<<<<<<<<<<< RAW <<<<<<<<<<<<<<<<<<<" << std::endl
            << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

  MultiShot shot = MultiShot(world, *lossFn.get(), 300, 50, false);
  shot.setParallelOperationsEnabled(true);
  IPOptOptimizer optimizer = IPOptOptimizer();
  optimizer.setIterationLimit(20);
  optimizer.setCheckDerivatives(false);
  optimizer.setSuppressOutput(true);
  optimizer.setRecoverBest(false);
  optimizer.setTolerance(1e-3);
  // optimizer.setDisableLinesearch(true);

  std::shared_ptr<OptimizationRecord> record = optimizer.optimize(&shot);

  for (int i = 0; i < 1; i++)
  {
    std::cout << "Restarting!" << std::endl;
    const TrajectoryRollout* cache = shot.getRolloutCache(world);
    world->setPositions(shot.getStartPos());
    world->setVelocities(shot.getStartVel());
    std::cout << "Initial pos: " << world->getPositions() << std::endl;
    std::cout << "Initial vel: " << world->getVelocities() << std::endl;
    for (int j = 0; j < advanceSteps; j++)
    {
      world->setForces(cache->getForcesConst().col(j));
      world->step();
      std::cout << "Step " << j << std::endl;
      std::cout << "Force: " << cache->getForcesConst().col(j) << std::endl;
      std::cout << "World pos: " << world->getPositions() << std::endl;
      std::cout << "World vel: " << world->getVelocities() << std::endl;
    }
    if (i == 10)
    {
      Eigen::VectorXd perturbedVelocities = world->getVelocities();
      perturbedVelocities(0) += 0.2;
      world->setVelocities(perturbedVelocities);
    }
    std::cout << "Final world pos: " << world->getPositions() << std::endl;
    std::cout << "Final world vel: " << world->getVelocities() << std::endl;

    shot.advanceSteps(
        world, world->getPositions(), world->getVelocities(), advanceSteps);
    record->reoptimize();
  }

  std::ofstream out("realtime.txt");
  out << record->toJson(world);
  out.close();

  snapshot.restore();

  std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl
            << "<<<<<<<<<<< MPC <<<<<<<<<<<<<<<<<<<" << std::endl
            << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  */

  MPC mpc = MPC(world, lossFn, planningHorizonMillis);
  mpc.recordGroundTruthState(
      0L, world->getPositions(), world->getVelocities(), world->getMasses());
  mpc.setSilent(true);

  /*
  long stepSize
      = advanceSteps * world->getTimeStep() * 1000; // timesteps to millis
  for (int i = 0; i < 50; i++)
  {
    std::cout << "Reoptimize" << std::endl;
    mpc.optimizePlan(i * stepSize);
    // Read forces up to the next iteration, so that they get used in projecting
    // the future
    for (int j = i * stepSize; j < (i + 1) * stepSize; j++)
    {
      mpc.getForce(j);
    }
  }

  std::ofstream out2("realtime.txt");
  out2 << mpc.getOptimizationRecord()->toJson(world);
  out2.close();
  */

  std::function<Eigen::VectorXd()> getForces
      = [&]() { return mpc.getForceNow(); };
  std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
      recordState
      = [&](Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass) {
          mpc.recordGroundTruthStateNow(pos, vel, mass);
        };
  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = world->clone();
  RealtimeWorld realtimeWorld
      = RealtimeWorld(realtimeUnderlyingWorld, getForces, recordState);

  mpc.registerReplanningListener(
      [&](const trajectory::TrajectoryRollout* rollout, long duration) {
        realtimeWorld.displayMPCPlan(rollout);
        realtimeWorld.registerTiming("replanning", duration, "ms");
      });

  mpc.setMaxIterations(7);

  mpc.setEnableLineSearch(false);
  mpc.setEnableOptimizationGuards(true);

  realtimeWorld.serve(8070);

  // Start everything up when someone connects for the first time
  realtimeWorld.registerConnectionListener([&]() {
    realtimeWorld.start();
    mpc.start();
  });

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3d originalColor = sledBodyVisual->getColor();
  realtimeWorld.registerPreStepListener(
      [sledBodyVisual, originalColor, &realtimeWorld, &mpc](
          int step,
          std::shared_ptr<simulation::World> world,
          std::unordered_set<std::string> keysDown) {
        realtimeWorld.registerTiming(
            "buffer", mpc.getRemainingPlanBufferMillis(), "ms");
        if (keysDown.count("a"))
        {
          Eigen::VectorXd perturbedForces = world->getForces();
          perturbedForces(0) = -15.0;
          world->setForces(perturbedForces);
          sledBodyVisual->setColor(Eigen::Vector3d(1, 0, 0));
        }
        else if (keysDown.count("e"))
        {
          Eigen::VectorXd perturbedForces = world->getForces();
          perturbedForces(0) = 15.0;
          world->setForces(perturbedForces);
          sledBodyVisual->setColor(Eigen::Vector3d(0, 1, 0));
        }
        else
        {
          sledBodyVisual->setColor(originalColor);
        }
      });

  while (true)
  {
    // spin
    // cartpole->setPosition(0, 0.0);
    // cartpole->setForces(Eigen::VectorXd::Zero(cartpole->getNumDofs()));
    // cartpole->setPositions(Eigen::VectorXd::Zero(cartpole->getNumDofs()));
  }
}
#endif
