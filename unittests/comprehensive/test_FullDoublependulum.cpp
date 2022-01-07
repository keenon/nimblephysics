#include <chrono>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <memory>
#include <thread>
#include <mutex>
#include <Eigen/Dense>


#include <gtest/gtest.h>
#include <random>
#include <cmath>

#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/iLQRLocal.hpp"
#include "dart/realtime/SSID.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

#define iLQR_MPC_TEST

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
  std::shared_ptr<simulation::World> world = dart::utils::UniversalLoader::loadWorld(
    "dart://sample/skel/inverted_double_pendulum.skel");
  world->setTimeStep(timestep);
  world->removeDofFromActionSpace(1);
  world->removeDofFromActionSpace(2);
  SkeletonPtr skel = world->getSkeleton(0);
  for(int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    for( dart::dynamics::ShapeNode* shapenode :skel->getBodyNode(i)->getShapeNodes())
    {
      // Collision handling may crash current iLQR
      shapenode->removeCollisionAspect();
    }
  }
  return world;
}

void clearShapeNodes(WorldPtr world)
{
  SkeletonPtr skel = world->getSkeleton(0);
  for(int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    skel->getBodyNode(i)->removeAllShapeNodes();
  }
}

WorldPtr cloneWorld(WorldPtr world, bool keep_shapeNode)
{
  WorldPtr cloneWorld = world->clone();
  // Default the robot skeleton is the first one
  SkeletonPtr skel = cloneWorld->getSkeleton(0);
  if(!keep_shapeNode)
  {
    clearShapeNodes(cloneWorld);
  }
  return cloneWorld;
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
  // Eigen::VectorXs force_eps = rand_normal(control_force.size(), 0, noise_scale, random_gen);
  ssid->registerControls(now, control_force);

  Eigen::VectorXs position = realtimeWorld->getPositions();
  Eigen::VectorXs position_eps = rand_normal(position.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, position + position_eps, 0);

  Eigen::VectorXs velocity = realtimeWorld->getVelocities();
  Eigen::VectorXs velocity_eps = rand_normal(velocity.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, velocity + velocity_eps, 1);
  ssid->registerUnlock();
}

#ifdef iLQR_MPC_TEST
TEST(REALTIME, CARTPOLE_MPC_MASS)
{
  WorldPtr world = createWorld(1.0 / 100);

  // Initialize Hyper Parameters
  // TODO: Need to find out
  int steps = 100;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  // For add noise in measurement
  // std::mt19937 rand_gen = initializeRandom();
  // s_t noise_scale = 0.01;

  // For SSID
  s_t scale = 1.0;
  size_t ssid_index = 1;
  size_t ssid_index2 = 2;
  int inferenceSteps = 10;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;

  std::shared_ptr<simulation::World> ssidWorld = cloneWorld(world, false);
  ssidWorld->tuneMass(
    world->getBodyNodeIndex(ssid_index),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);

  ssidWorld->tuneMass(
    world->getBodyNodeIndex(ssid_index2),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);

  Eigen::Vector2s sensorDims(world->getNumDofs(), world->getNumDofs());
  std::vector<size_t> ssid_idx{ssid_index};//, ssid_index2};

  SSID ssid = SSID(ssidWorld,
                   getSSIDPosLoss(),
                   inferenceHistoryMillis,
                   sensorDims,
                   inferenceSteps,
                   scale);
  std::mutex lock;
  std::mutex param_lock;
  ssid.attachMutex(lock);
  ssid.attachParamMutex(param_lock);
  // ssid.setSSIDIndex(Eigen::Vector2i(ssid_index, ssid_index2));
  Eigen::VectorXi index;
  index = Eigen::VectorXi::Zero(1);
  index(0) = ssid_index;
  ssid.setSSIDIndex(index);  

  ssid.setInitialPosEstimator(
    [](Eigen::MatrixXs sensors, long)
    {
      return sensors.col(0);
    });

  ssid.setInitialVelEstimator(
    [](Eigen::MatrixXs sensors, long)
    {
      return sensors.col(0);
    });

  world->clearTunableMassThisInstance();
  // Create Goal
  int dofs = 3;
  Eigen::VectorXs runningStateWeight = Eigen::VectorXs::Zero(2 * dofs);
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Ones(1) * 0.01;
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Zero(2 * dofs);
  finalStateWeight << 10.0, 50, 10, 10, 10 ,10;

  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = cloneWorld(world,true);

  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight,
                                           world);

  costFn->setSSIDNodeIndex(ssid_idx);
  // costFn->enableSSIDLoss(0.01);
  costFn->setTimeStep(world->getTimeStep());
  Eigen::Vector6s goal;
  goal << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  costFn->setTarget(goal);
  std::cout << "Goal: " << goal << std::endl;
  iLQRLocal mpcLocal = iLQRLocal(
    world, 1, planningHorizonMillis, 1.0);
  mpcLocal.setCostFn(costFn);
  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(5);
  mpcLocal.setPatience(1);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);
  mpcLocal.setActionBound(40.0);
  mpcLocal.setAlpha(1.0);

  ssid.registerInferListener([&](long,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs mass,
                                 long){
    // mpcLocal.recordGroundTruthState(time, pos, vel, mass); //TODO: This will cause problem ... But Why
    mpcLocal.setParameterChange(mass);
    world->setLinkMassIndex(mass(0), ssid_index);
    world->setLinkMassIndex(mass(1), ssid_index2);
  });


  
  GUIWebsocketServer server;

  /*
  server.createSphere("goal", 0.1,
                      Eigen::Vector3s(goal(0),1.0,0),
                      Eigen::Vector3s(1.0, 0.0, 0.0));
  server.registerDragListener("goal", [&](Eigen::Vector3s dragTo){
    goal(0) = dragTo(0);
    dragTo(1) = 1.0;
    dragTo(2) = 0.0;
    costFn->setTarget(goal);
    server.setObjectPosition("goal", dragTo);
  });
  */

  std::string key = "mass";

  Ticker ticker = Ticker(scale * realtimeUnderlyingWorld->getTimeStep());
  long total_steps = 0;

  Eigen::Vector2s masses;
  masses(0) = world->getLinkMassIndex(ssid_index);
  // masses(1) = 0;
  masses(1) = world->getLinkMassIndex(ssid_index2);
  Eigen::Vector2s id_masses(0.5, 0.5);

  // realtimeUnderlyingWorld->setLinkMassIndex(masses(0), ssid_index);
  // realtimeUnderlyingWorld->setLinkMassIndex(masses(1), ssid_index2);
  // ssidWorld->setLinkMassIndex(id_masses(0), ssid_index);
  // ssidWorld->setLinkMassIndex(id_masses(1), ssid_index2);
  // world->setLinkMassIndex(id_masses(0), ssid_index);
  // world->setLinkMassIndex(id_masses(1), ssid_index2);
  // Preload visualization
  bool renderIsReady = false;
  ticker.registerTickListener([&](long now) {
    // Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
    Eigen::VectorXs mpcforces;
    if(renderIsReady)
    {
      mpcforces = mpcLocal.computeForce(realtimeUnderlyingWorld->getState(), now);
      //mpcforces = Eigen::VectorXs::Ones(dofs);
    }
    else
    {
      mpcforces = Eigen::VectorXs::Zero(dofs);
    }
    //std::cout << "MPC Force: \n" << mpcforces 
    //          << "\nRef forces: \n" << feedback_forces << std::endl;
    // TODO: Currently the forces are almost zero need to figure out why
    // std::cout <<"Force:\n" << mpcforces << std::endl;
    //Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
    // Eigen::VectorXs force_eps = rand_normal(mpcforces.size(), 0, noise_scale, rand_gen);
    realtimeUnderlyingWorld->setControlForces(mpcforces);
    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = -15.0;
      // perturbedForces(6) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = 15.0;
      // perturbedForces(6) = 15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }

    if (server.getKeysDown().count(","))
    {
      // Increase mass
      masses(0) = 3.0;
      masses(1) = 2.5;
      realtimeUnderlyingWorld->setLinkMassIndex(masses(0), ssid_index);
      realtimeUnderlyingWorld->setLinkMassIndex(masses(1), ssid_index2);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      masses(0) = 1.0;
      masses(1) = 0.5;
      realtimeUnderlyingWorld->setLinkMassIndex(masses(0), ssid_index);
      realtimeUnderlyingWorld->setLinkMassIndex(masses(1), ssid_index2);
    }
    else if(server.getKeysDown().count("s"))
    {
      renderIsReady = true;
      //ssid.start();
      //ssid.startSlow();
      mpcLocal.setPredictUsingFeedback(true);
      mpcLocal.ilqrstart();
    }
    // recordObs(now, &ssid, realtimeUnderlyingWorld);
    if(renderIsReady)
    {
      // recordObsWithNoise(now, &ssid, realtimeUnderlyingWorld, noise_scale, rand_gen);
      recordObs(now, &ssid, realtimeUnderlyingWorld);
      realtimeUnderlyingWorld->step();
    }
    id_masses(0) = world->getLinkMassIndex(ssid_index);
    id_masses(1) = world->getLinkMassIndex(ssid_index2);
    if(renderIsReady)
    {
      mpcLocal.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());
    }

    if(total_steps % 5 == 0)
    {
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,
                        "Current Masses: "+std::to_string(id_masses(0))+" "+std::to_string(id_masses(1))+" "+
                        "Real Masses: "+std::to_string(masses(0))+" "+std::to_string(masses(1)),
                        Eigen::Vector2i(100,100),
                        Eigen::Vector2i(200,200));
      // server.createText(key,
      //                   "Current Masses: "+std::to_string(id_masses(0))+" "+"Real Masses: " + std::to_string(masses(0)),
      //                   Eigen::Vector2i(100, 100),
      //                   Eigen::Vector2i(400, 400));
      total_steps = 0;
    }
    total_steps ++;
  });

  // Should only work when trajectory opt
  // We can always feed the trajectory used in forward pass here
  mpcLocal.registerReplanningListener(
      [&](long ,
          const trajectory::TrajectoryRollout* rollout,
          long ) {
        // std::cout << "Reached Here!" << std::endl;
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });
  
  server.registerConnectionListener([&](){
    ticker.start();
    
    
  });
  server.registerShutdownListener([&]() {mpcLocal.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif