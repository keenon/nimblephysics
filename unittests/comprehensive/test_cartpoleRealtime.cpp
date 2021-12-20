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

#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/MPCLocal.hpp"
#include "dart/realtime/SSID.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

//#define COM_SSID
#define MASS_SSID

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
  ssid->registerLock();
  ssid->registerControls(now, realtimeWorld->getControlForces());
  ssid->registerSensors(now, realtimeWorld->getPositions(),0);
  ssid->registerSensors(now, realtimeWorld->getVelocities(),1);
  ssid->registerUnlock();
}

void recordObsWithNoise(size_t now, SSID* ssid, WorldPtr realtimeWorld, s_t noise_scale, std::mt19937 random_gen)
{
  ssid->registerLock();
  Eigen::VectorXs control_force = realtimeWorld->getControlForces();
  Eigen::VectorXs force_eps = rand_normal(control_force.size(), 0, noise_scale, random_gen);
  ssid->registerControls(now, control_force + force_eps);

  Eigen::VectorXs position = realtimeWorld->getPositions();
  Eigen::VectorXs position_eps = rand_normal(position.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, position + position_eps, 0);

  Eigen::VectorXs velocity = realtimeWorld->getVelocities();
  Eigen::VectorXs velocity_eps = rand_normal(velocity.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, velocity + velocity_eps, 1);
  ssid->registerUnlock();
}

#ifdef COM_SSID
TEST(REALTIME, CARTPOLE_REALTIME_COM)
{
  WorldPtr world = createWorld(1.0 / 100);

  // Initialize Hyperparameters
  Eigen::Vector3s beta(0.1, 1, 0);
  world->setLinkBetaIndex(beta,1);

  Eigen::Vector2s sensorDims(world->getNumDofs(), world->getNumDofs());

  int steps = 300;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  int inferenceSteps = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;
  s_t scale = 1.0;

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

  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  ssidWorld->tuneMass(
    world->getBodyNodeIndex(1),
    WrtMassBodyNodeEntryType::INERTIA_COM_MU,
    Eigen::VectorXs::Ones(1) * 0.5,
    Eigen::VectorXs::Ones(1) * -0.5);
  
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
  
  bool init_flag = true;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mu,
                                 long) {
    mpcLocal.recordGroundTruthState(time, pos, vel, mu);
    mpcLocal.setMUchange(mu(0));
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
    Eigen::VectorXs forces = mpcLocal.getControlForceNow();
    // ssid.registerControlsNow(forces);
    return forces;
  };
  std::function<void(Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs)>
      recordState
      = [&](Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mu) {
          mpcLocal.recordGroundTruthStateNow(pos, vel, mu);
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
    Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
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
    
    recordObs(now, &ssid, realtimeUnderlyingWorld);
    
    realtimeUnderlyingWorld->step();
    
    mpcLocal.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses()); // May be a problem?

    if(total_step%5==0)
    {
      id_mu = world->getLinkMUIndex(1);
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,"Current MU: "+std::to_string(id_mu),Eigen::Vector2i(100,100),Eigen::Vector2i(200,200));
      total_step = 0;
    }
    total_step += 1;
  });

  mpcLocal.registerReplanningListener(
      [&](long ,
          const trajectory::TrajectoryRollout* rollout,
          long ) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });

  server.registerConnectionListener([&]() {
    ticker.start();
    mpcLocal.start();
    ssid.start();
  });
  server.registerShutdownListener([&]() { mpcLocal.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif

#ifdef MASS_SSID
TEST(REALTIME, CARTPOLE_MPC_MASS)
{
  WorldPtr world = createWorld(1.0/100);

  // Initialize Hyperparameters
  // std::mt19937 rand_gen = initializeRandom();
  // s_t noise_stddev = 0.01;

  // planning parameters
  int steps = 300;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  // SSID Parameters
  s_t scale = 1.0;
  int inferenceSteps = 20;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;

  s_t goalX = 1.0;

  TrajectoryLossFn loss = [&goalX](const TrajectoryRollout* rollout) {
    int steps = rollout->getPosesConst("identity").cols();
    s_t sum = 0.0;
    for (int i = 0; i < steps; i++)
    {
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
        }
        s_t sum = 0.0;
        for (int i = 0; i < steps; i++)
        {
          s_t xPos = rollout->getPosesConst()(0, i);
          s_t theta = rollout->getPosesConst()(1, i);
          sum += (goalX - xPos) * (goalX - xPos) + theta * theta;
        }
        return sum;
      };
  
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  ssidWorld->tuneMass(
    world->getBodyNodeIndex(0),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);

  ssidWorld->tuneMass(
    world->getBodyNodeIndex(1),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);
  
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(ssidWorld, 
                   getSSIDPosLoss(), 
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
  
  bool init_flag = true;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mass,
                                 long) {
    mpcLocal.recordGroundTruthState(time, pos, vel, mass);
    // If we need to do SSID with multiple node this detection need to work with vector
    mpcLocal.setParameterChange(mass);
    if(!init_flag)
    {
      s_t old_mass_0 = world->getLinkMassIndex(0);
      world->setLinkMassIndex(0.9*old_mass_0+0.1*mass(0),0);
      s_t old_mass_1 = world->getLinkMassIndex(1);
      world->setLinkMassIndex(0.9*old_mass_1+0.1*mass(1),1);
    }
    else
    {
      world->setLinkMassIndex(mass(0), 0);
      world->setLinkMassIndex(mass(1), 1);
      init_flag = false;
    }
    
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
  std::string key = "mass";

  Ticker ticker = Ticker(scale*realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  Eigen::Vector2s masses(2.0, 1.5);
  Eigen::Vector2s id_masses(1.0, 1.0);
  size_t total_step = 0;
  realtimeUnderlyingWorld->setLinkMassIndex(masses(0), 0);
  realtimeUnderlyingWorld->setLinkMassIndex(masses(1), 1);
  ssidWorld->setLinkMassIndex(id_masses(0), 0);
  ssidWorld->setLinkMassIndex(id_masses(1), 1);
  world->setLinkMassIndex(id_masses(0), 0);
  world->setLinkMassIndex(id_masses(1), 1);

  ticker.registerTickListener([&](long now) {
    Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
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
      masses(0) = 3.0;
      masses(1) = 2.5;
      realtimeUnderlyingWorld->setLinkMassIndex(masses(0), 0);
      realtimeUnderlyingWorld->setLinkMassIndex(masses(1), 1);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      masses(0) = 1.0;
      masses(1) = 0.5;
      realtimeUnderlyingWorld->setLinkMassIndex(masses(0), 0);
      realtimeUnderlyingWorld->setLinkMassIndex(masses(1), 1);
    }
    
    recordObs(now, &ssid, realtimeUnderlyingWorld);
    // recordObsWithNoise(now, &ssid, realtimeUnderlyingWorld, noise_stddev, rand_gen);
    realtimeUnderlyingWorld->step();
    id_masses(0) = world->getLinkMassIndex(0);
    id_masses(1) = world->getLinkMassIndex(1);
    mpcLocal.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());
    if(total_step % 5 == 0)
    {
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,
                        "Current Masses: "+std::to_string(id_masses(0))+" "+std::to_string(id_masses(1)),
                        Eigen::Vector2i(100,100),
                        Eigen::Vector2i(200,200));
      total_step = 0;
    }
    total_step += 1;
  });

  mpcLocal.registerReplanningListener(
      [&](long /* time */,
          const trajectory::TrajectoryRollout* rollout,
          long /* duration */) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });

  server.registerConnectionListener([&]() {
    ticker.start();
    mpcLocal.start();
    ssid.start();
  });
  server.registerShutdownListener([&]() { mpcLocal.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif
