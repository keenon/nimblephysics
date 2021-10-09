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

#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>
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
  // Create a half Cheetah example
  ////////////////////////////////////////////////////////////

  // World
  WorldPtr world = dart::utils::UniversalLoader::loadWorld(
      "dart://sample/skel/half_cheetah.skel");

  Eigen::VectorXs forceLimits
    = Eigen::VectorXs::Ones(world->getNumDofs()) * 100;
  forceLimits(0) = 0;
  forceLimits(1) = 0;
  world->setControlForceUpperLimits(forceLimits);
  world->setControlForceLowerLimits(-1 * forceLimits);
  
  ////////////////////////////////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////

  // 100 fps
  world->setTimeStep(1.0 / 100);

  // 300 timesteps
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 300 * millisPerTimestep;

  // Create target

  s_t target_x = 0.5;
  s_t target_y = 0.5;

  // BUG: Target pass by value vs pass by reference. 
  TrajectoryLossFn loss = [&target_x, &target_y](const trajectory::TrajectoryRollout* rollout) {
        const Eigen::VectorXs lastPos
            = rollout->getPosesConst().col(rollout->getPosesConst().cols() - 1);

        s_t diffX = lastPos(0) - target_x;
        s_t diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      };

  TrajectoryLossFnAndGrad lossGrad =
      [&target_x,&target_y](
          const TrajectoryRollout* rollout,
          TrajectoryRollout* gradWrtRollout // OUT
      ) {
        gradWrtRollout->getPoses().setZero();
        gradWrtRollout->getVels().setZero();
        gradWrtRollout->getControlForces().setZero();
        int steps = rollout->getPosesConst().cols();
        const Eigen::VectorXs lastPos
            = rollout->getPosesConst().col(steps-1);

        gradWrtRollout->getPoses()(0, steps-1)
              = 2 * (rollout->getPosesConst()(0, steps-1) - target_x);
        
        gradWrtRollout->getPoses()(1, steps-1) 
              = 2 * (rollout->getPosesConst()(1, steps-1) - target_y);
        
        
        s_t sum = 0.0;
        sum += (lastPos(0) - target_x)*(lastPos(0) - target_x);
        sum += (lastPos(1) - target_y)*(lastPos(1) - target_y);
        return sum;
      };

  int inferenceSteps = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  
  // Need to get a body node which is useful for SSID
  ssidWorld->tuneMass(
    world->getBodyNodeIndex(1),
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
      Eigen::Vector3s(target_x, target_y, 0),
      Eigen::Vector3s(1.0, 0.0, 0.0));
  server.registerDragListener("goal", [&](Eigen::Vector3s dragTo) {
    target_x = dragTo(0);
    dragTo(1) = 1.0;
    dragTo(2) = 0.0;
    server.setObjectPosition("goal", dragTo);
  });
  std::string key = "mass";

  Ticker ticker = Ticker(realtimeUnderlyingWorld->getTimeStep());

  float mass = 2.0;
  float id_mass = 1.0;
  size_t total_step = 0;
  //realtimeUnderlyingWorld->setLinkMassIndex(mass,1);
  //ssidWorld->setLinkMassIndex(id_mass,1);
  //world->setLinkMassIndex(id_mass,1);

  ticker.registerTickListener([&](long now) {
    //Eigen::VectorXs mpcforces = mpcRemote.getControlForce(now);
    Eigen::VectorXs mpcforces = Eigen::VectorXs::Zero(world->getNumDofs());
    realtimeUnderlyingWorld->setControlForces(mpcforces);

    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = 15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }
    if (server.getKeysDown().count(","))
    {
      // Increase mass
      mass = 3.0;
      realtimeUnderlyingWorld->setLinkMassIndex(mass,1);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      mass = 1.0;
      realtimeUnderlyingWorld->setLinkMassIndex(mass,1);
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
    //mpcRemote.start();
    //ssid.start();
  });
  server.registerShutdownListener([&]() { mpcRemote.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif