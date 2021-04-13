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
#include "dart/realtime/MPCLocal.hpp"
#include "dart/realtime/MPCRemote.hpp"
#include "dart/realtime/SSID.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;
using namespace server;

std::shared_ptr<LossFn> getMPCLoss()
{
  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst("identity").cols();
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      // rollout->getVelsConst().col(i).squaredNorm()
      sum += rollout->getPosesConst().col(i).squaredNorm();
    }
    return sum;
  };

  TrajectoryLossFnAndGrad lossGrad = [](const TrajectoryRollout* rollout,
                                        TrajectoryRollout* gradWrtRollout // OUT
                                     ) {
    gradWrtRollout->getPoses().setZero();
    gradWrtRollout->getVels().setZero();
    gradWrtRollout->getControlForces().setZero();
    int steps = rollout->getPosesConst().cols();
    for (int i = 0; i < steps; i++)
    {
      gradWrtRollout->getPoses().col(i) = 2 * rollout->getPosesConst().col(i);
      // gradWrtRollout->getVels().col(i) = 2 *
      // rollout->getVelsConst().col(i);
    }
    /*
    for (int i = 0; i < steps; i++)
    {
      gradWrtRollout->getControlForces("identity").col(i)
          = 2 * rollout->getControlForcesConst("identity").col(i);
    }
    */
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      // rollout->getVelsConst().col(i).squaredNorm()
      sum += rollout->getPosesConst().col(i).squaredNorm();
    }
    return sum;
  };

  return std::make_shared<LossFn>(loss, lossGrad);
}

std::shared_ptr<LossFn> getSSIDLoss()
{
  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    Eigen::MatrixXs sensorPositions = rollout->getMetadata("sensors");
    int steps = rollout->getPosesConst().cols();

    Eigen::MatrixXs posError = rollout->getPosesConst() - sensorPositions;

    // std::cout << "Pos Error: " << std::endl << posError << std::endl;

    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += posError.col(i).squaredNorm();
    }
    return sum;
  };

  TrajectoryLossFnAndGrad lossGrad = [](const TrajectoryRollout* rollout,
                                        TrajectoryRollout* gradWrtRollout // OUT
                                     ) {
    gradWrtRollout->getPoses().setZero();
    gradWrtRollout->getVels().setZero();
    gradWrtRollout->getControlForces().setZero();
    int steps = rollout->getPosesConst().cols();

    Eigen::MatrixXs sensorPositions = rollout->getMetadata("sensors");

    Eigen::MatrixXs posError = rollout->getPosesConst() - sensorPositions;

    gradWrtRollout->getPoses() = 2 * posError;
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += posError.col(i).squaredNorm();
    }
    return sum;
  };

  return std::make_shared<LossFn>(loss, lossGrad);
}

#ifdef ALL_TESTS
TEST(REALTIME, CARTPOLE_MPC)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3s(1, 0, 0));
  std::shared_ptr<BoxShape> sledShapeBox(
      new BoxShape(Eigen::Vector3s(0.5, 0.1, 0.1)));
  ShapeNode* sledShape
      = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);
  sledShape->getVisualAspect()->setColor(Eigen::Vector3s(0.5, 0.5, 0.5));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3s(0, 0, 1));
  std::shared_ptr<BoxShape> armShapeBox(
      new BoxShape(Eigen::Vector3s(0.1, 1.0, 0.1)));
  ShapeNode* armShape
      = armPair.second->createShapeNodeWith<VisualAspect>(armShapeBox);
  armShape->getVisualAspect()->setColor(Eigen::Vector3s(0.7, 0.7, 0.7));

  Eigen::Isometry3s armOffset = Eigen::Isometry3s::Identity();
  armOffset.translation() = Eigen::Vector3s(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  world->addSkeleton(cartpole);

  cartpole->setControlForceUpperLimit(0, 15);
  cartpole->setControlForceLowerLimit(0, -15);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);

  cartpole->setControlForceUpperLimit(1, 0);
  cartpole->setControlForceLowerLimit(1, 0);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());
  // cartpole->getDof(1)->setCoulombFriction(0.1);

  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 300 * millisPerTimestep;
  int advanceSteps = 70;

  s_t goalX = 1.0;

  TrajectoryLossFn loss = [&goalX](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst("identity").cols();
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      // rollout->getVelsConst().col(i).squaredNorm()
      s_t xPos = rollout->getPosesConst()(0, i);
      s_t theta = rollout->getPosesConst()(1, i);
      sum += (goalX - xPos) * (goalX - xPos) + theta * theta;
    }
    return sum;
  };

  TrajectoryLossFnAndGrad lossGrad =
      [&goalX](
          const TrajectoryRollout* rollout,
          TrajectoryRollout* gradWrtRollout // OUT
      ) {
        gradWrtRollout->getPoses().setZero();
        gradWrtRollout->getVels().setZero();
        gradWrtRollout->getControlForces().setZero();
        int steps = rollout->getPosesConst().cols();
        for (int i = 0; i < steps; i++)
        {
          gradWrtRollout->getPoses()(0, i)
              = 2 * (rollout->getPosesConst()(0, i) - goalX);
          gradWrtRollout->getPoses()(1, i) = 2 * rollout->getPosesConst()(1, i);
          // gradWrtRollout->getVels().col(i) = 2 *
          // rollout->getVelsConst().col(i);
        }
        /*
        for (int i = 0; i < steps; i++)
        {
          gradWrtRollout->getControlForces("identity").col(i)
              = 2 * rollout->getControlForcesConst("identity").col(i);
        }
        */
        s_t sum = 0.0;
        for (int i = 0; i < steps; i++)
        {
          // rollout->getVelsConst().col(i).squaredNorm()
          s_t xPos = rollout->getPosesConst()(0, i);
          s_t theta = rollout->getPosesConst()(1, i);
          sum += (goalX - xPos) * (goalX - xPos) + theta * theta;
        }
        return sum;
      };

  MPCLocal mpcLocal = MPCLocal(
      world, std::make_shared<LossFn>(loss, lossGrad), planningHorizonMillis);
  mpcLocal.setSilent(true);

  int inferenceHistoryMillis = 10 * millisPerTimestep;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  SSID ssid = SSID(
      ssidWorld, getSSIDLoss(), inferenceHistoryMillis, world->getNumDofs());

  mpcLocal.setMaxIterations(7);

  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  // This should fork off mpcLocal to another process
  // MPCRemote mpcRemote = MPCRemote(mpcLocal);
  MPCLocal& mpcRemote = mpcLocal;

  std::function<Eigen::VectorXs()> getControlForces = [&]() {
    Eigen::VectorXs forces = mpcRemote.getControlForceNow();
    // ssid.registerControlsNow(forces);
    return forces;
  };
  std::function<void(Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs)>
      recordState
      = [&](Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass) {
          mpcRemote.recordGroundTruthStateNow(pos, vel, mass);
          // ssid.registerSensorsNow(pos);
        };
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mass,
                                 long duration) {
    mpcRemote.recordGroundTruthState(time, pos, vel, mass);
    world->setMasses(mass);
  });

  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = world->clone();
  GUIWebsocketServer server;

  server.createSphere(
      "goal",
      0.1,
      Eigen::Vector3s(goalX, 1.0, 0),
      Eigen::Vector3s(1.0, 0.0, 0.0));
  server.registerDragListener("goal", [&](Eigen::Vector3s dragTo) {
    goalX = dragTo(0);
    dragTo(1) = 1.0;
    dragTo(2) = 0.0;
    server.setObjectPosition("goal", dragTo);
  });

  Ticker ticker = Ticker(realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();

  ticker.registerTickListener([&](long now) {
    realtimeUnderlyingWorld->setControlForces(mpcRemote.getControlForce(now));

    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getExternalForces();
      perturbedForces(0) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
      sledBodyVisual->setColor(Eigen::Vector3s(1, 0, 0));
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getExternalForces();
      perturbedForces(0) = 15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
      sledBodyVisual->setColor(Eigen::Vector3s(0, 1, 0));
    }
    else
    {
      sledBodyVisual->setColor(originalColor);
    }

    if (server.getKeysDown().count(","))
    {
      // Increase mass
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
    }

    /*
    realtimeWorld.registerTiming(
        "buffer", mpcRemote.getRemainingPlanBufferMillis(), "ms");
        */

    realtimeUnderlyingWorld->step();
    mpcRemote.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());

    server.renderWorld(realtimeUnderlyingWorld);
  });

  mpcRemote.registerReplanningListener(
      [&](long time,
          const trajectory::TrajectoryRollout* rollout,
          long duration) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
        // realtimeWorld.registerTiming("replanning", duration, "ms");
      });

  // Start everything up when someone connects for the first time
  server.registerConnectionListener([&]() {
    ticker.start();
    mpcRemote.start();

    // TODO: turns out IPOPT isn't threadsafe to run in multiple instances in
    // parallel, because MUMPS isn't threadsafe. So we need to spawn child
    // processes to handle identifying state.
    //
    // https://github.com/coin-or/Ipopt/issues/298

    // ssid.start();
  });

  server.registerShutdownListener([&]() { mpcRemote.stop(); });

  server.serve(8070);
  while (server.isServing())
  {
    // spin
    // cartpole->setPosition(0, 0.0);
    // cartpole->setControlForces(Eigen::VectorXs::Zero(cartpole->getNumDofs()));
    // cartpole->setPositions(Eigen::VectorXs::Zero(cartpole->getNumDofs()));
  }
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CARTPOLE_SSID)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3s(1, 0, 0));
  std::shared_ptr<BoxShape> sledShapeBox(
      new BoxShape(Eigen::Vector3s(0.5, 0.1, 0.1)));
  ShapeNode* sledShape
      = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);
  sledShape->getVisualAspect()->setColor(Eigen::Vector3s(0.5, 0.5, 0.5));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3s(0, 0, 1));
  std::shared_ptr<BoxShape> armShapeBox(
      new BoxShape(Eigen::Vector3s(0.1, 1.0, 0.1)));
  ShapeNode* armShape
      = armPair.second->createShapeNodeWith<VisualAspect>(armShapeBox);
  armShape->getVisualAspect()->setColor(Eigen::Vector3s(0.7, 0.7, 0.7));

  Eigen::Isometry3s armOffset = Eigen::Isometry3s::Identity();
  armOffset.translation() = Eigen::Vector3s(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  world->addSkeleton(cartpole);

  cartpole->setControlForceUpperLimit(0, 15);
  cartpole->setControlForceLowerLimit(0, -15);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);

  cartpole->setControlForceUpperLimit(1, 0);
  cartpole->setControlForceLowerLimit(1, 0);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());
  // cartpole->getDof(1)->setCoulombFriction(0.1);

  world->tuneMass(
      armPair.second,
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 3.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDLoss();

  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int inferenceHistoryMillis = 5 * millisPerTimestep;
  int advanceSteps = 70;

  SSID ssid = SSID(world, lossFn, inferenceHistoryMillis, world->getNumDofs());

  armPair.second->setMass(2.0);
  for (int i = 0; i < 50; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    world->setControlForces(forces);
    world->step();
    ssid.registerControls(time, forces);
    ssid.registerSensors(time, world->getPositions());
  }
  armPair.second->setMass(1.0);

  ssid.setInitialPosEstimator([](Eigen::MatrixXs sensors, long timestamp) {
    // Use the first column of sensor data as an approximate starting point
    return sensors.col(0);
  });

  ssid.runInference(30 * millisPerTimestep);

  std::cout << "Recovered mass after 1st iteration: "
            << armPair.second->getMass() << std::endl;

  ssid.runInference(50 * millisPerTimestep);

  std::cout << "Recovered mass after 2nd iteration: "
            << armPair.second->getMass() << std::endl;
}
#endif
