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

#define MASS_SSID
#define COM_SSID
#define MOI_SSID
#define DAMP_SSID
#define SPRING_SSID

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

#ifdef MASS_SSID
TEST(REALTIME, CARTPOLE_MASS_SSID)
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
  int steps = 10;
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
  Eigen::VectorXi ssid_index = Eigen::VectorXi::Ones(1);
  ssid.setSSIDMassIndex(ssid_index);
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
  for (int i = 0; i < 100; i++)
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
      EXPECT_TRUE(abs(init_mass - realmass) <= 1e-3);
      world->setLinkMassIndex(realmass, 1);
    }
  }
}
#endif

#ifdef COM_SSID
TEST(REALTIME, CARTPOLE_COM_SSID)
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
      WrtMassBodyNodeEntryType::INERTIA_COM,
      Eigen::VectorXs::Ones(3) * 0.2,
      Eigen::VectorXs::Ones(3) * -0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 10;
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

  Eigen::VectorXi ssid_index = Eigen::VectorXi::Ones(1);
  ssid.setSSIDCOMIndex(ssid_index);
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
  Eigen::VectorXs real_com = world->getLinkCOMIndex(1);
  Eigen::VectorXs init_com = Eigen::Vector3s(0.1, 0.1,real_com(2));
  
  for (int i = 0; i < 100; i++)
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
    
    if(i%10==0 && i >= steps+10)
    {
      world->setLinkCOMIndex(init_com, 1);
      ssid.runInference(time);
      init_com = world->getLinkCOMIndex(1);
      EXPECT_TRUE((init_com - real_com).norm() <= 1e-3);
      world->setLinkCOMIndex(real_com, 1);
    }
  }
}
#endif

#ifdef MOI_SSID
TEST(REALTIME, CARTPOLE_MOI_SSID)
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
      WrtMassBodyNodeEntryType::INERTIA_DIAGONAL_NOMASS,
      Eigen::VectorXs::Ones(3) * 0.1,
      Eigen::VectorXs::Zero(3));

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 20;
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

  Eigen::VectorXi ssid_index = Eigen::VectorXi::Ones(1);
  ssid.setSSIDMOIIndex(ssid_index);
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
  Eigen::VectorXs real_moi = Eigen::Vector3s(0.02, 0.02, 0.05);
  world->setLinkDiagIIndex(real_moi, 1);
  Eigen::VectorXs init_moi = Eigen::Vector3s(0.02, 0.02, 0.02);
  
  for (int i = 0; i < 100; i++)
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
    
    if(i%10==0 && i >= steps+10)
    {
      world->setLinkDiagIIndex(init_moi, 1);
      ssid.runInference(time);
      init_moi = world->getLinkDiagIIndex(1);
      EXPECT_TRUE(abs(init_moi(2) - real_moi(2)) <= 1e-3);
      world->setLinkDiagIIndex(real_moi, 1);
    }
  }
}
#endif

#ifdef DAMP_SSID
TEST(REALTIME, CARTPOLE_DAMPING_SSID)
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
  Eigen::VectorXi dofs_index = Eigen::VectorXi::Zero(1);
  dofs_index(0) = 1;

  world->tuneDamping(
      world->getJointIndex(1),
      WrtDampingJointEntryType::DAMPING,
      dofs_index,
      Eigen::VectorXs::Ones(1) * 2.0,
      Eigen::VectorXs::Ones(1) * 0.01);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 10;
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
  ssid.setSSIDDampIndex(dofs_index);
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
  Eigen::Vector1s realdamp(0.3);
  world->setJointDampingCoeffIndex(realdamp, 1);
  Eigen::Vector1s init_damp(0.1);
  for (int i = 0; i < 100; i++)
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
      world->setJointDampingCoeffIndex(init_damp, 1);
      ssid.runInference(time);
      init_damp = world->getJointDampingCoeffIndex(1);
      EXPECT_TRUE(abs(init_damp(0) - realdamp(0)) <= 1e-3);
      world->setJointDampingCoeffIndex(realdamp, 1);
    }
  }
}
#endif

#ifdef SPRING_SSID
TEST(REALTIME, CARTPOLE_SPRING_SSID)
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
  Eigen::VectorXi dofs_index = Eigen::VectorXi::Zero(1);
  dofs_index(0) = 1;

  world->tuneSpring(
      world->getJointIndex(1),
      WrtSpringJointEntryType::SPRING,
      dofs_index,
      Eigen::VectorXs::Ones(1) * 2.0,
      Eigen::VectorXs::Ones(1) * 0.01);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();

  // 300 timesteps
  s_t scale = 1.0;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 10;
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
  ssid.setSSIDSpringIndex(dofs_index);
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
  Eigen::Vector1s realspring(0.3);
  world->setJointSpringStiffIndex(realspring, 1);
  Eigen::Vector1s init_spring(0.1);
  for (int i = 0; i < 100; i++)
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
      world->setJointSpringStiffIndex(init_spring, 1);
      ssid.runInference(time);
      init_spring = world->getJointSpringStiffIndex(1);
      EXPECT_TRUE(abs(init_spring(0)-realspring(0)) <= 1e-3);
      world->setJointSpringStiffIndex(realspring, 1);
    }
  }
}
#endif