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
#include <sstream>
#include <memory>
#include <thread>
#include <mutex>

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <filesystem>

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

// #define COM_SSID
#define MASS_SSID
// #define MASS_PLOT
#define WITH_NOISE

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;
using namespace server;

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

WorldPtr createWorld(s_t timestep)
{
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));
  SkeletonPtr cartpole = Skeleton::create("cartpole");
  // Create Cart
  std::pair<PrismaticJoint*, BodyNode*> sledPair
    = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3s(1, 0, 0));
  std::shared_ptr<BoxShape> sledShapeBox(
      new BoxShape(Eigen::Vector3s(0.5, 0.1, 0.1)));
  ShapeNode* sledShape
    = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);
  sledShape->getVisualAspect()->setColor(Eigen::Vector3s(0.5, 0.5, 0.5));
  
  // Create Pole
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
  cartpole->setControlForceUpperLimit(1, 0);
  cartpole->setControlForceLowerLimit(1, 0);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);
  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 30.0 / 180.0 * 3.1415);
  world->setTimeStep(timestep);
  
  return world;
}

std::mt19937 initializeRandom()
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  return gen;
}

Eigen::VectorXs rand_normal(size_t length, s_t mean, s_t stddev, std::mt19937 random_gen)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(length);
  std::normal_distribution<> dist{mean, stddev};

  for(int i = 0; i < length; i++)
  {
    result(i) += (s_t)(dist(random_gen));
  }
  return result;
}

void recordObs(size_t now, SSID* ssid, WorldPtr realtimeWorld)
{
  ssid->registerControls(now, realtimeWorld->getControlForces());
  ssid->registerSensors(now, realtimeWorld->getPositions(), 0);
  ssid->registerSensors(now, realtimeWorld->getVelocities(), 1);
}

void recordObsWithNoise(size_t now, SSID* ssid, WorldPtr realtimeWorld, s_t noise_scale, std::mt19937 random_gen)
{
  Eigen::VectorXs control_force = realtimeWorld->getControlForces();
  ssid->registerControls(now, control_force);

  Eigen::VectorXs position = realtimeWorld->getPositions();
  Eigen::VectorXs position_eps = rand_normal(position.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, position + position_eps, 0);

  Eigen::VectorXs velocity = realtimeWorld->getVelocities();
  Eigen::VectorXs velocity_eps = rand_normal(velocity.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, velocity + velocity_eps, 1);
}

#ifdef COM_SSID
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

  Eigen::Vector3s beta;
  beta << 0.1, 1, 0;
  world->setLinkBetaIndex(beta,1);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 200 * millisPerTimestep;

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
  s_t scale = 1.0;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  
  ssidWorld->tuneMass(
    armPair.second, // Also feasible since only name is used
    WrtMassBodyNodeEntryType::INERTIA_COM_MU,
    Eigen::VectorXs::Ones(1) * 0.5,
    Eigen::VectorXs::Ones(1) * -0.5);
  
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(ssidWorld, 
                   getSSIDVelPosLoss(), 
                   inferenceHistoryMillis, 
                   sensorDims,
                   inferenceSteps,
                   scale);
  
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
  MPCLocal mpcLocal = MPCLocal(world, 
                               std::make_shared<LossFn>(loss, lossGrad), 
                               planningHorizonMillis,
                               scale);
  mpcLocal.setSilent(true);
  
  mpcLocal.setMaxIterations(7);

  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  MPCLocal& mpcRemote = mpcLocal;
  
  bool init_flag = true;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mu,
                                 long) {
    mpcRemote.recordGroundTruthState(time, pos, vel, mu);
    mpcRemote.setMUchange(mu(0));
    if(!init_flag)
    {
      s_t old_mu = world->getLinkMUIndex(1);
      world->setLinkMUIndex(0.9*old_mu+0.1*mu(0),1);
    }
    else
    {
      world->setLinkMUIndex(mu(0),1);
    }
    
  });

  std::function<Eigen::VectorXs()> getControlForces = [&]() {
    Eigen::VectorXs forces = mpcRemote.getControlForceNow();
    // ssid.registerControlsNow(forces);
    return forces;
  };
  std::function<void(Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs)>
      recordState
      = [&](Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mu) {
          mpcRemote.recordGroundTruthStateNow(pos, vel, mu);
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
  std::string key = "com mu";

  Ticker ticker = Ticker(1*realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  float mu = 0.05;
  float id_mu = -0.2;
  size_t total_step = 0;
  realtimeUnderlyingWorld->setLinkMUIndex(mu,1);
  ssidWorld->setLinkMUIndex(id_mu,1);
  world->setLinkMUIndex(id_mu,1);
  // Sanity Check
  std::cout<<world->getLinkBetas()<<std::endl;
  std::cout<<ssidWorld->getLinkBetas()<<std::endl;
  std::cout<<realtimeUnderlyingWorld->getLinkBetas()<<std::endl;
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
      // Increase mu
      mu = 1.0;
      realtimeUnderlyingWorld->setLinkMUIndex(mu,1);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      mu = 0;
      realtimeUnderlyingWorld->setLinkMUIndex(mu,1);
    }
    
    ssid.registerLock();
    ssid.registerControls(now, realtimeUnderlyingWorld->getControlForces());
    ssid.registerSensors(now, realtimeUnderlyingWorld->getPositions(),0);
    ssid.registerSensors(now, realtimeUnderlyingWorld->getVelocities(),1);
    ssid.registerUnlock();
    realtimeUnderlyingWorld->step();
    id_mu = world->getLinkMUIndex(1);
    mpcRemote.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses()); // May be a problem?

    if(total_step%5==0)
    {
      //std::cout<<"World Rendered"<<std::endl;
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,"Current MU: "+std::to_string(id_mu),Eigen::Vector2i(100,100),Eigen::Vector2i(200,200));
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

#ifdef MASS_SSID
TEST(REALTIME, CARTPOLE_SSID)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

  // World
  WorldPtr world = createWorld(1.0/100);
  #ifdef WITH_NOISE
  std::mt19937 random_gen = initializeRandom();
  s_t noise_scale = 0.01;
  #endif
  
  std::cout << "A_k Matrix of cartpole\n"
            << world->getSkeleton(0)->getLinkAkMatrixIndex(1)
            << std::endl;

  world->tuneMass(
      world->getBodyNodeIndex(1),
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 5.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 200;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  // int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, 
                   lossFn, 
                   inferenceHistoryMillis, 
                   sensorDims,
                   steps,
                   scale);
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
  float realmass = 2.0;
  world->setLinkMassIndex(realmass, 1);
  float init_mass = 1.0;
  for (int i = 0; i < 500; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    #ifdef WITH_NOISE
    // Eigen::VectorXs forces_eps = rand_normal(forces.size(), 0, 0.01, random_gen);
    Eigen::VectorXs forces_eps = Eigen::VectorXs::Zero(forces.size());
    world->setControlForces(forces + forces_eps);
    #else
    world->setControlForces(forces);
    #endif

    #ifndef WITH_NOISE
    recordObs(time, &ssid, world);
    #else
    recordObsWithNoise(time, &ssid, world, noise_scale, random_gen);
    #endif
    world->step();
    
    if(i%5==0 && i >= steps+10)
    {
      world->setLinkMassIndex(init_mass, 1);
      ssid.runInference(time);
      init_mass = world->getLinkMassIndex(1);
      std::cout << "Recovered mass after iteration "<<i<<": "
                << init_mass << std::endl;
      world->setLinkMassIndex(realmass, 1);
    }
  }
}
#endif

#ifdef MASS_PLOT
TEST(REALTIME, CARTPOLE_PLOT)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

  // World
  WorldPtr world = createWorld(1.0/100);
  #ifdef WITH_NOISE
  std::mt19937 random_gen = initializeRandom();
  s_t noise_scale = 0.01;
  #endif
  
  world->tuneMass(
      world->getBodyNodeIndex(1),
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 5.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 100;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  // int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, 
                   lossFn, 
                   inferenceHistoryMillis, 
                   sensorDims,
                   steps,
                   scale);
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
  float realmass = 2.0;
  world->setLinkMassIndex(realmass, 1);
  float init_mass = 1.0;
  std::vector<Eigen::VectorXs> multi_loss;
  std::vector<s_t> solutions;
  int data_cnt = 0;
  std::string exp_name = "noise_100";
  std::stringstream folder_stream;
  folder_stream << "mkdir -p /workspaces/nimblephysics/dart/realtime/saved_data/raw_data/States/" << exp_name;
  std::string folder_cmd = folder_stream.str();
  std::system(folder_cmd.c_str());
  for (int i = 0; i < 500; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    #ifdef WITH_NOISE
    Eigen::VectorXs forces_eps = rand_normal(forces.size(), 0, 0.01, random_gen);
    world->setControlForces(forces + forces_eps);
    #else
    world->setControlForces(forces);
    #endif

    #ifndef WITH_NOISE
    recordObs(time, &ssid, world);
    #else
    recordObsWithNoise(time, &ssid, world, noise_scale, random_gen);
    #endif
    world->step();
    
    if(i%5==0 && i >= steps-1)
    {
      std::stringstream ss;
      ss << "/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/States/" << exp_name <<"/state_"<< data_cnt++<<".csv";
      std::string fname = ss.str();
      std::pair<Eigen::VectorXs, Eigen::MatrixXs> result 
        = ssid.runPlotting(time,5.0,0.2,200);
      multi_loss.push_back(result.first);
      ssid.saveCSVMatrix(fname, result.second);
      world->setLinkMassIndex(init_mass, 1);
      ssid.runInference(time);
      init_mass = world->getLinkMassIndex(1);
      std::cout << "Recovered mass after iteration "<<i<<": "
                << init_mass << std::endl;
      solutions.push_back(init_mass);
      world->setLinkMassIndex(realmass, 1);
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
  file.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Losses.txt");
  file<<lossMatrix.transpose();
  file.close();
  
  std::ofstream sfile;
  sfile.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Solutions.txt");
  sfile<<solutionVec;
  sfile.close();
}
#endif

#ifdef MU_PLOT
TEST(REALTIME, CARTPOLE_MU_PLOT)
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

  Eigen::Vector3s beta;
  beta << 0.1, 1, 0;
  world->setLinkBetaIndex(beta,1);

  s_t upper_bound = 0.5;
  s_t lower_bound = -0.5;

  world->tuneMass(
      world->getBodyNodeIndex(1),
      WrtMassBodyNodeEntryType::INERTIA_COM_MU,
      Eigen::VectorXs::Ones(1) *  upper_bound,
      Eigen::VectorXs::Ones(1) *  lower_bound);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();
  ////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 5;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  // int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, 
                   lossFn, 
                   inferenceHistoryMillis, 
                   sensorDims,
                   steps,
                   scale);
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
  
  world->setLinkMUIndex(0.2,1);
  float init_mu = -0.4;
  std::vector<Eigen::VectorXs> multi_loss;
  std::vector<s_t> solutions;

  for (int i = 0; i < 300; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    world->setControlForces(forces);
    ssid.registerControls(time, forces);
    ssid.registerSensors(time, world->getPositions(),0);
    ssid.registerSensors(time, world->getVelocities(),1);
    world->step();
    
    if(i%5==0 && i!=0 && i!=5)
    {
      multi_loss.push_back(ssid.runPlotting(time,lower_bound,upper_bound,200));
      world->setLinkMUIndex(init_mu,1);
      ssid.runInference(time);
      init_mu = world->getLinkMUIndex(1);
      std::cout << "Recovered mu after iteration "<<i<<": "
                << init_mu << std::endl;
      solutions.push_back(init_mu);
      world->setLinkMUIndex(0.2,1);
    }
  }
  Eigen::MatrixXs lossMatrix = Eigen::MatrixXs::Zero(200,multi_loss.size());
  Eigen::VectorXs solutionVec = Eigen::VectorXs::Zero(solutions.size());
  for(int i=0;i<multi_loss.size();i++)
  {
    lossMatrix.col(i) = multi_loss[i];
    solutionVec(i) = solutions[i];
  }
  std::ofstream file;
  file.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Losses.txt");
  file<<lossMatrix.transpose();
  file.close();

  std::ofstream sfile;
  sfile.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Solutions.txt");
  sfile<<solutionVec;
  sfile.close();
}
#endif

#ifdef COM_PLOT
TEST(REALTIME, CARTPOLE_COM_PLOT)
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

  Eigen::Vector3s upper_bound;
  Eigen::Vector3s lower_bound;
  upper_bound <<  0.05,  0.5,  0;
  lower_bound << -0.05, -0.5,  0;
  

  world->tuneMass(
      world->getBodyNodeIndex(1),
      WrtMassBodyNodeEntryType::INERTIA_COM,
      upper_bound,
      lower_bound);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();
  ////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 5;
  int inferenceHistoryMillis = steps * millisPerTimestep;
  // int advanceSteps = 70;
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(world, 
                   lossFn, 
                   inferenceHistoryMillis, 
                   sensorDims,
                   steps,
                   scale);
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
  
  Eigen::Vector3s real_com;
  real_com << 0.02, 0.2, 0;
  world->setLinkCOMIndex(real_com,1);
  std::cout<<"Initial COM\n"<<world->getLinkCOMIndex(1)<<std::endl;
  Eigen::Vector3s init_com;
  Eigen::Vector3s solution;
  init_com << 0.03, 0.3, 0;
  std::vector<Eigen::MatrixXs> multi_loss;
  std::vector<Eigen::Vector3s> solutions;
  size_t data_cnt = 0;
  for (int i = 0; i < 300; i++)
  {
    long time = i * millisPerTimestep;
    Eigen::VectorXs forces = Eigen::VectorXs::Ones(world->getNumDofs());
    world->setControlForces(forces);
    ssid.registerControls(time, forces);
    ssid.registerSensors(time, world->getPositions(),0);
    ssid.registerSensors(time, world->getVelocities(),1);
    world->step();
    
    if(i%5==0 && i!=0 && i!=5)
    {
      std::stringstream ss;
      ss << "/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Losses/loss_"<< data_cnt++<<".csv";
      std::string fname = ss.str();
      Eigen::MatrixXs losses = ssid.runPlotting2D(time,upper_bound,lower_bound,10,100,2);
      ssid.saveCSVMatrix(fname,losses);
      world->setLinkCOMIndex(init_com,1);
      ssid.runInference(time);
      solution = world->getLinkCOMIndex(1);
      std::cout << "Recovered mu after iteration "<<i<<": \n"
                << solution << std::endl;
      if((solution-real_com).norm()>1e-4)
      {
        std::cout<<"Bad Init Guess\n"<<init_com<<std::endl;
      }
      solutions.push_back(solution);
      //init_com.segment(0,1) = Eigen::Vector1s::Random()*0.05;
      //init_com.segment(1,1) = Eigen::Vector1s::Random()*0.5;
      init_com(0) = 0.03;
      init_com(1) = 0.3;


      // solutions.push_back(init_mu);
      world->setLinkCOMIndex(real_com,1);
    }
  }
  
  Eigen::MatrixXs solutionMat = Eigen::MatrixXs::Zero(solutions.size(),3);
  for(int i=0;i<solutions.size();i++)
  {
    solutionMat.row(i) = solutions[i];
  }
  ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Solutions.csv",solutionMat);
}
#endif