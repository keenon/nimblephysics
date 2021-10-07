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
#include <mutex>

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

#define ALL_TESTS

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
    }
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += rollout->getPosesConst().col(i).squaredNorm();
    }
    return sum;
  };

  return std::make_shared<LossFn>(loss, lossGrad);
}

std::shared_ptr<LossFn> getSSIDPosLoss()
{
  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    Eigen::MatrixXs rawPos = rollout->getMetadata("sensors");
    Eigen::MatrixXs sensorPositions = rawPos.block(0,1,rawPos.rows(),rawPos.cols()-1);
    int steps = rollout->getPosesConst().cols();

    Eigen::MatrixXs posError = rollout->getPosesConst() - sensorPositions;

    //std::cout << "Pos In Buffer: " << std::endl << sensorPositions << std::endl;
    //std::cout << "Pos In Traj  : " << std::endl << rollout->getPosesConst() << std::endl;

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

    Eigen::MatrixXs rawPos = rollout->getMetadata("sensors");
    Eigen::MatrixXs sensorPositions = rawPos.block(0,1,rawPos.rows(),rawPos.cols()-1);

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

std::shared_ptr<LossFn> getSSIDVelPosLoss()
{
  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {

    Eigen::MatrixXs rawPos = rollout->getMetadata("sensors");
    Eigen::MatrixXs rawVel = rollout->getMetadata("velocities");
    Eigen::MatrixXs sensorPositions = rawPos.block(0,1,rawPos.rows(),rawPos.cols()-1);
    Eigen::MatrixXs sensorVelocities = rawVel.block(0,1,rawVel.rows(),rawVel.cols()-1);
    int steps = rollout->getPosesConst().cols();

    Eigen::MatrixXs posError = rollout->getPosesConst() - sensorPositions;
    Eigen::MatrixXs velError = rollout->getVelsConst() - sensorVelocities;

    // std::cout << "Pos Error: " << std::endl << posError << std::endl;

    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += posError.col(i).squaredNorm();
      sum += velError.col(i).squaredNorm();
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

    Eigen::MatrixXs rawPos = rollout->getMetadata("sensors");
    Eigen::MatrixXs rawVel = rollout->getMetadata("velocities");
    Eigen::MatrixXs sensorPositions = rawPos.block(0,1,rawPos.rows(),rawPos.cols()-1);
    Eigen::MatrixXs sensorVelocities = rawVel.block(0,1,rawVel.rows(),rawVel.cols()-1);

    Eigen::MatrixXs posError = rollout->getPosesConst() - sensorPositions;
    Eigen::MatrixXs velError = rollout->getVelsConst() - sensorVelocities;

    gradWrtRollout->getPoses() = 2 * posError;
    gradWrtRollout->getVels() = 2 * velError;
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += posError.col(i).squaredNorm();
      sum += velError.col(i).squaredNorm();
    }
    return sum;
  };

  return std::make_shared<LossFn>(loss, lossGrad);
}

std::shared_ptr<LossFn> getSSIDVelLoss()
{
  TrajectoryLossFn loss = [](const TrajectoryRollout* rollout) {
    Eigen::MatrixXs rawVel = rollout->getMetadata("velocities");
    Eigen::MatrixXs sensorVelocities = rawVel.block(0,1,rawVel.rows(),rawVel.cols()-1);
    int steps = rollout->getVelsConst().cols();

    Eigen::MatrixXs velError = rollout->getVelsConst() - sensorVelocities;

    // std::cout << "Pos Error: " << std::endl << posError << std::endl;

    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += velError.col(i).squaredNorm();
    }
    return sum;
  };

  TrajectoryLossFnAndGrad lossGrad = [](const TrajectoryRollout* rollout,
                                        TrajectoryRollout* gradWrtRollout // OUT
                                     ) {
    gradWrtRollout->getPoses().setZero();
    gradWrtRollout->getVels().setZero();
    gradWrtRollout->getControlForces().setZero();
    int steps = rollout->getVelsConst().cols();

    Eigen::MatrixXs rawVel = rollout->getMetadata("velocities");
    Eigen::MatrixXs sensorVelocities = rawVel.block(0,1,rawVel.rows(),rawVel.cols()-1);

    Eigen::MatrixXs velError = rollout->getVelsConst() - sensorVelocities;

    gradWrtRollout->getVels() = 2 * velError;
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
      sum += velError.col(i).squaredNorm();
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
  cartpole->setPosition(1, 30.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());
  
  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 300 * millisPerTimestep;

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

  int inferenceSteps = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  
  ssidWorld->tuneMass(
    armPair.second, // Also feasible since only name is used
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);
  
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(
      ssidWorld, getSSIDPosLoss(), inferenceHistoryMillis, sensorDims,inferenceSteps);
  
  std::mutex lock;
  ssid.attachMutex(lock);
  
  ssid.setInitialPosEstimator(
      [](Eigen::MatrixXs sensors, long) {
        return sensors.col(0);
      });
  
  ssid.setInitialVelEstimator(
    [](Eigen::MatrixXs sensors, long){
      return sensors.col(0);
    });

  world->clearTunableMassThisInstance();
  MPCLocal mpcLocal = MPCLocal(
      world, std::make_shared<LossFn>(loss, lossGrad), planningHorizonMillis);
  mpcLocal.setSilent(true);
  
  mpcLocal.setMaxIterations(7);

  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  MPCLocal& mpcRemote = mpcLocal;
  
  bool init_flag = true;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mass,
                                 long) {
    mpcRemote.recordGroundTruthState(time, pos, vel, mass);
    mpcRemote.setMasschange(mass(0));
    if(!init_flag)
    {
      s_t old_mass = world->getLinkMassIndex(1);
      world->setLinkMassIndex(0.9*old_mass+0.1*mass(0),1);
    }
    else
    {
      world->setLinkMassIndex(mass(0),1);
    }
    
  });

  std::function<Eigen::VectorXs()> getControlForces = [&]() {
    Eigen::VectorXs forces = mpcRemote.getControlForceNow();
    // ssid.registerControlsNow(forces);
    return forces;
  };
  std::function<void(Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs)>
      recordState
      = [&](Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass) {
          mpcRemote.recordGroundTruthStateNow(pos, vel, mass);
        };
  
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
  std::string key = "mass";

  Ticker ticker = Ticker(0.5*realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  float mass = 2.0;
  float id_mass = 1.0;
  size_t total_step = 0;
  realtimeUnderlyingWorld->setLinkMassIndex(mass,1);
  ssidWorld->setLinkMassIndex(id_mass,1);
  world->setLinkMassIndex(id_mass,1);

  ticker.registerTickListener([&](long now) {
    Eigen::VectorXs mpcforces = mpcRemote.getControlForce(now);
    realtimeUnderlyingWorld->setControlForces(mpcforces);

    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
      sledBodyVisual->setColor(Eigen::Vector3s(1, 0, 0));
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
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
      mass = 3.0;
      realtimeUnderlyingWorld->getSkeleton("cartpole")->getBodyNode(1)->setMass(mass);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      mass = 1.0;
      realtimeUnderlyingWorld->getSkeleton("cartpole")->getBodyNode(1)->setMass(mass);
    }
    
    ssid.registerLock();
    ssid.registerControls(now, realtimeUnderlyingWorld->getControlForces());
    ssid.registerSensors(now, realtimeUnderlyingWorld->getPositions(),0);
    ssid.registerSensors(now, realtimeUnderlyingWorld->getVelocities(),1);
    ssid.registerUnlock();
    realtimeUnderlyingWorld->step();
    id_mass = world->getLinkMassIndex(1);
    mpcRemote.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());
    if(total_step % 5 == 0)
    {
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,"Current Masses: "+std::to_string(id_mass),Eigen::Vector2i(100,100),Eigen::Vector2i(200,200));
      total_step = 0;
    }
    total_step += 1;
  });

  mpcRemote.registerReplanningListener(
      [&](long /* time */,
          const trajectory::TrajectoryRollout* rollout,
          long /* duration */) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });

  server.registerConnectionListener([&]() {
    ticker.start();
    mpcRemote.start();
    ssid.start();
  });
  server.registerShutdownListener([&]() { mpcRemote.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif


#ifdef ALL_TESTS
TEST(REALTIME, CARTPOLE_MPC_COM)
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
  cartpole->setPosition(1, 30.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());
  
  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 300 * millisPerTimestep;

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

  int inferenceSteps = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  
  ssidWorld->tuneMass(
    armPair.second, // Also feasible since only name is used
    WrtMassBodyNodeEntryType::INERTIA_COM,
    Eigen::VectorXs::Ones(3) * 5.0,
    Eigen::VectorXs::Ones(3) * 0.2);
  
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(
      ssidWorld, getSSIDPosLoss(), inferenceHistoryMillis, sensorDims,inferenceSteps);
  
  std::mutex lock;
  ssid.attachMutex(lock);
  
  ssid.setInitialPosEstimator(
      [](Eigen::MatrixXs sensors, long) {
        return sensors.col(0);
      });
  
  ssid.setInitialVelEstimator(
    [](Eigen::MatrixXs sensors, long){
      return sensors.col(0);
    });

  world->clearTunableMassThisInstance();
  MPCLocal mpcLocal = MPCLocal(
      world, std::make_shared<LossFn>(loss, lossGrad), planningHorizonMillis);
  mpcLocal.setSilent(true);
  
  mpcLocal.setMaxIterations(7);

  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  MPCLocal& mpcRemote = mpcLocal;
  
  bool init_flag = true;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs com,
                                 long) {
    mpcRemote.recordGroundTruthState(time, pos, vel, com);
    mpcRemote.setCOMchange(com);
    if(!init_flag)
    {
      Eigen::Vector3s old_com = world->getLinkCOMIndex(1);
      world->setLinkCOMIndex(0.9*old_com+0.1*com,1);
    }
    else
    {
      world->setLinkCOMIndex(com,1);
    }
    
  });

  std::function<Eigen::VectorXs()> getControlForces = [&]() {
    Eigen::VectorXs forces = mpcRemote.getControlForceNow();
    // ssid.registerControlsNow(forces);
    return forces;
  };
  std::function<void(Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs)>
      recordState
      = [&](Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs com) {
          mpcRemote.recordGroundTruthStateNow(pos, vel, com);
        };
  
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
  std::string key = "mass";

  Ticker ticker = Ticker(0.1*realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  float mass = 2.0;
  float id_mass = 1.0;
  size_t total_step = 0;
  realtimeUnderlyingWorld->setLinkMassIndex(mass,1);
  ssidWorld->setLinkMassIndex(id_mass,1);
  world->setLinkMassIndex(id_mass,1);

  ticker.registerTickListener([&](long now) {
    Eigen::VectorXs mpcforces = mpcRemote.getControlForce(now);
    realtimeUnderlyingWorld->setControlForces(mpcforces);

    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
      sledBodyVisual->setColor(Eigen::Vector3s(1, 0, 0));
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
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
      mass = 3.0;
      realtimeUnderlyingWorld->getSkeleton("cartpole")->getBodyNode(1)->setMass(mass);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      mass = 1.0;
      realtimeUnderlyingWorld->getSkeleton("cartpole")->getBodyNode(1)->setMass(mass);
    }
    
    ssid.registerLock();
    ssid.registerControls(now, realtimeUnderlyingWorld->getControlForces());
    ssid.registerSensors(now, realtimeUnderlyingWorld->getPositions(),0);
    ssid.registerSensors(now, realtimeUnderlyingWorld->getVelocities(),1);
    ssid.registerUnlock();
    realtimeUnderlyingWorld->step();
    id_mass = world->getLinkMassIndex(1);
    mpcRemote.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());

    if(total_step%100==0)
    {
      std::cout<<"World Rendered"<<std::endl;
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,"Current Masses: "+std::to_string(id_mass),Eigen::Vector2i(100,100),Eigen::Vector2i(200,200));
      total_step = 0;
    }
    total_step += 1;
  });

  mpcRemote.registerReplanningListener(
      [&](long ,
          const trajectory::TrajectoryRollout* rollout,
          long ) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });

  server.registerConnectionListener([&]() {
    ticker.start();
    mpcRemote.start();
    ssid.start();
  });
  server.registerShutdownListener([&]() { mpcRemote.stop(); });
  server.serve(8070);
  server.blockWhileServing();
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

  world->tuneMass(
      sledPair.second,
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 5.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();
  ////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int steps = 5;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  //int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, lossFn, inferenceHistoryMillis, sensorDims,steps);
  ssid.setInitialPosEstimator(
      [](Eigen::MatrixXs sensors, long /* timestamp */) {
        return sensors.col(0);
      });
  
  ssid.setInitialVelEstimator(
    [](Eigen::MatrixXs sensors, long)
    {
      return sensors.col(0);
    }
  );
  
  sledPair.second->setMass(2.0);
  float init_mass = 1.0;
  for (int i = 0; i < 1500; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Zero(world->getNumDofs());
    world->setControlForces(forces);
    world->step();
    ssid.registerControls(time, forces);
    ssid.registerSensors(time, world->getPositions(),0);
    ssid.registerSensors(time, world->getVelocities(),1);
    
    if(i%50==0 && i!=0 && i!=50)
    {
      sledPair.second->setMass(init_mass);
      for(int j=0;j<1;j++)
      {
        ssid.runInference(time);
      }
      init_mass = sledPair.second->getMass();
      std::cout << "Recovered mass after iteration "<<i<<": "
                << init_mass << std::endl;
      sledPair.second->setMass(2.0);
    }
  }
}
#endif

TEST(REALTIME, CARTPOLE_PLOT)
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

  world->tuneMass(
      //sledPair.second,
      armPair.second,
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 5.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();
  ////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 5;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  // int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, lossFn, inferenceHistoryMillis, sensorDims,steps);
  ssid.setInitialPosEstimator(
      [](Eigen::MatrixXs sensors, long /* timestamp */) {
        return sensors.col(0);
      });
  
  ssid.setInitialVelEstimator(
    [](Eigen::MatrixXs sensors, long)
    {
      return sensors.col(0);
    }
  );
  
  //sledPair.second->setMass(2.0);
  armPair.second->setMass(2.0);
  float init_mass = 1.0;
  std::vector<Eigen::VectorXs> multi_loss;
  //std::vector<Eigen::VectorXs> multi_loss2;
  std::vector<s_t> solutions;
  for (int i = 0; i < 200; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    world->setControlForces(forces);
    world->step();
    ssid.registerControls(time, forces);
    ssid.registerSensors(time, world->getPositions(),0);
    ssid.registerSensors(time, world->getVelocities(),1);
    //std::cout<<"Registered Pos: \n"<<world->getPositions()<<"\nVel: \n"<<world->getVelocities()<<"\nMass: "<<armPair.second->getMass()<<std::endl;
    
    if(i%5==0 && i!=0)
    {
      multi_loss.push_back(ssid.runPlotting(time,5.0,0.2,200));
      //multi_loss2.push_back(ssid.runPlotting(time,5.0,0.2,200));
      //sledPair.second->setMass(init_mass);
      armPair.second->setMass(init_mass);
      ssid.runInference(time);
      //init_mass = sledPair.second->getMass();
      init_mass = armPair.second->getMass();
      std::cout << "Recovered mass after iteration "<<i<<": "
                << init_mass << std::endl;
      solutions.push_back(init_mass);
      //sledPair.second->setMass(2.0); 
      armPair.second->setMass(2.0);
    }
  }
  Eigen::MatrixXs lossMatrix = Eigen::MatrixXs::Zero(200,multi_loss.size());
  //Eigen::MatrixXs lossMatrix2 = Eigen::MatrixXs::Zero(200,multi_loss.size());
  Eigen::VectorXs solutionVec = Eigen::VectorXs::Zero(solutions.size());
  for(int i=0;i<multi_loss.size();i++)
  {
    lossMatrix.col(i) = multi_loss[i];
    //lossMatrix2.col(i) = multi_loss2[i];
    solutionVec(i) = solutions[i];
  }
  std::ofstream file;
  file.open("/workspaces/nimblephysics/saved_data/raw_data/Losses.txt");
  file<<lossMatrix.transpose();
  file.close();
  /*
  std::ofstream file2;
  file2.open("/workspaces/nimblephysics/saved_data/raw_data/Losses2.txt");
  file2<<lossMatrix2.transpose();
  file2.close();
  */
  std::ofstream sfile;
  sfile.open("/workspaces/nimblephysics/saved_data/raw_data/Solutions.txt");
  sfile<<solutionVec;
  sfile.close();
}
