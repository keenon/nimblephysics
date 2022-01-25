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
#include "dart/neural/IKMapping.hpp"
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
#define USE_NOISE

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
    "dart://sample/skel/whip2d.skel");
  world->setTimeStep(timestep);
  world->removeDofFromActionSpace(1);
  world->removeDofFromActionSpace(2);
  world->removeDofFromActionSpace(3);
  Eigen::VectorXs init_state = Eigen::VectorXs::Zero(8);
  // s_t pi = 3.14159;
  // init_state << 0.0, 30.0/180 * pi, 30.0/180 * pi, 30.0 /180 * pi, 0, 0, 0, 0;
  world->setState(init_state);
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
  ssid->registerControls(now, control_force);

  Eigen::VectorXs position = realtimeWorld->getPositions();
  Eigen::VectorXs position_eps = rand_normal(position.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, position + position_eps, 0);

  Eigen::VectorXs velocity = realtimeWorld->getVelocities();
  Eigen::VectorXs velocity_eps = rand_normal(velocity.size(), 0, noise_scale, random_gen);
  ssid->registerSensors(now, velocity + velocity_eps, 1);
  ssid->registerUnlock();
}

std::vector<s_t> convert2stdvec(Eigen::VectorXs vec)
{
  std::vector<s_t> stdvec;
  for(int i = 0; i < vec.size(); i++)
  {
    stdvec.push_back(vec(i));
  }
  return stdvec;
}

Eigen::MatrixXs std2eigen(std::vector<Eigen::VectorXs> record)
{
  size_t num_record = record.size();
  Eigen::MatrixXs eigen_record = Eigen::MatrixXs::Zero(num_record, record[0].size());
  for(int i = 0 ; i < num_record; i++)
  {
    eigen_record.row(i) = record[i];
  }
  return eigen_record;
}

#ifdef iLQR_MPC_TEST
TEST(REALTIME, CARTPOLE_MPC_MASS)
{
  WorldPtr world = createWorld(2.0 / 1000);

  // Initialize Hyper Parameters
  // TODO: Need to find out
  int steps = 300;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  // For add noise in measurement
  #ifdef USE_NOISE
  std::mt19937 rand_gen = initializeRandom();
  s_t noise_scale = 0.01;
  #endif

  // For SSID
  s_t scale = 1.0;
  size_t ssid_index = 1;
  size_t ssid_index2 = 2;
  size_t ssid_index3 = 3;
  int inferenceSteps = 10;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;

  std::shared_ptr<simulation::World> ssidWorld = cloneWorld(world, false);
  Eigen::VectorXi index = Eigen::VectorXi::Ones(1);

  ssidWorld->tuneSpring(
    world->getJointIndex(ssid_index),
    WrtSpringJointEntryType::SPRING,
    index,
    Eigen::VectorXs::Ones(1) * 15.0,
    Eigen::VectorXs::Ones(1) * 0.1);

  index(0) = 2;
  ssidWorld->tuneSpring(
    world->getJointIndex(ssid_index2),
    WrtSpringJointEntryType::SPRING,
    index,
    Eigen::VectorXs::Ones(1) * 15.0,
    Eigen::VectorXs::Ones(1) * 0.1);
  

  index(0) = 3;
  ssidWorld->tuneSpring(
    world->getJointIndex(ssid_index3),
    WrtSpringJointEntryType::SPRING,
    index,
    Eigen::VectorXs::Ones(1) * 15.0,
    Eigen::VectorXs::Ones(1) * 0.1);
  
  Eigen::Vector2s sensorDims(world->getNumDofs(), world->getNumDofs());

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

  ssid.useSmoothing();
  ssid.useHeuristicWeight();
  ssid.useConfidence();
  ssid.setTemperature(Eigen::Vector3s(0.1, 0.1, 0.1));
  ssid.setThreshs(0.3, 0.5);



  Eigen::Vector3i id_index(ssid_index, ssid_index2, ssid_index3);
  ssid.setSSIDSpringIndex(id_index);

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

  world->clearTunableSpringThisInstance();
  // Create Goal
  int dofs = 4;
  Eigen::VectorXs runningStateWeight = Eigen::VectorXs::Ones(2 * dofs) * 0.01; // Need it to be as fast as possible
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Ones(1) * 0.001;
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Zero(2 * dofs);
  // Requires IK which is not implemented
  finalStateWeight << 100, 100, 100, 100, 50, 100, 100, 100;

  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = cloneWorld(world,true);

  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight,
                                           world);

  // The objective is the elastic rod vibrate between the two goal
  //s_t pi = 3.14159265;
  Eigen::VectorXs goal = Eigen::VectorXs::Zero(2 * dofs);
  goal << 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // Up right pose

  costFn->setTarget(goal);
  // Need to change the coefficient on the fly
  //costFn->setSSIDSpringJointIndex(Eigen::Vector3i(1, 2, 3));
  //costFn->enableSSIDLoss(1);
  iLQRLocal mpcLocal = iLQRLocal(
    world, 1, planningHorizonMillis, 1.0);
  
  mpcLocal.setCostFn(costFn);
  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(5);
  mpcLocal.setPatience(3);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);
  mpcLocal.setActionBound(40.0);
  mpcLocal.setAlpha(1.0);

  ssid.registerInferListener([&](long,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs confidence,
                                 Eigen::VectorXs spring,
                                 long){
    // mpcLocal.recordGroundTruthState(time, pos, vel, mass); //TODO: This will cause problem ... But Why
    mpcLocal.setParameterChange(spring);
    world->setJointSpringStiffIndex(spring.segment(0, 1), ssid_index);
    world->setJointSpringStiffIndex(spring.segment(1, 1), ssid_index2);
    world->setJointSpringStiffIndex(spring.segment(2, 1), ssid_index3);
    // Should interface here to change the weight of the system
    if(confidence.mean() > 0.2)
    {
      costFn->setSSIDHeuristicWeight(0);
    }
  });


  
  GUIWebsocketServer server;


  // May need to create the sphere for better visualization
  
  server.createSphere("goal1", 0.1,
                      Eigen::Vector3s(goal(0), 0.0, 0),
                      Eigen::Vector3s(1.0, 0.0, 0.0));
  server.registerDragListener("goal1", [&](Eigen::Vector3s dragTo){
    goal(0) = dragTo(0);
    dragTo(1) = 0.0;
    dragTo(2) = 0.0;
    costFn->setTarget(goal);
    server.setObjectPosition("goal1", dragTo);
  });
  

  std::string key = "spring";

  Ticker ticker = Ticker(scale * realtimeUnderlyingWorld->getTimeStep());
  long total_steps = 0;

  Eigen::Vector3s spring_stiffs;
  spring_stiffs(0) = world->getJointSpringStiffIndex(ssid_index)(0);
  spring_stiffs(1) = world->getJointSpringStiffIndex(ssid_index2)(0);
  spring_stiffs(2) = world->getJointSpringStiffIndex(ssid_index3)(0);
  Eigen::Vector3s id_spring_stiffs(5.0, 3.0, 2.0);

  realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(0, 1), ssid_index);
  realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(1, 1), ssid_index2);
  realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(2, 1), ssid_index3);
  ssidWorld->setJointSpringStiffIndex(id_spring_stiffs.segment(0, 1), ssid_index);
  ssidWorld->setJointSpringStiffIndex(id_spring_stiffs.segment(1, 1), ssid_index2);
  ssidWorld->setJointSpringStiffIndex(id_spring_stiffs.segment(2, 1), ssid_index3);
  world->setJointSpringStiffIndex(id_spring_stiffs.segment(0, 1), ssid_index);
  world->setJointSpringStiffIndex(id_spring_stiffs.segment(1, 1), ssid_index2);
  world->setJointSpringStiffIndex(id_spring_stiffs.segment(2, 1), ssid_index3);
  // Preload visualization
  bool renderIsReady = false;
  int filecnt = 0;
  int cnt = 0;
  bool record = false;
  std::vector<Eigen::VectorXs> real_record;
  std::vector<Eigen::VectorXs> id_record;

  ticker.registerTickListener([&](long now) {
    // Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
    Eigen::VectorXs mpcforces;
    if(renderIsReady)
    {
      mpcforces = mpcLocal.computeForce(realtimeUnderlyingWorld->getState(), now);
      #ifdef USE_NOISE
      Eigen::VectorXs force_eps = rand_normal(mpcforces.size(), 0, noise_scale, rand_gen);
      mpcforces += force_eps;
      #endif
    }
    else
    {
      mpcforces = Eigen::VectorXs::Zero(dofs);
    }
    

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

    if (server.getKeysDown().count(",") || cnt == 300)
    {
      // Increase mass
      spring_stiffs(0) = 8.0;
      spring_stiffs(1) = 8.0;
      spring_stiffs(2) = 8.0;
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(0, 1), ssid_index);
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(1, 1), ssid_index2);
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(2, 1), ssid_index3);
    }
    else if (server.getKeysDown().count("o") || cnt == 600)
    {
      // Decrease mass
      spring_stiffs(0) = 8.0;
      spring_stiffs(1) = 8.0;
      spring_stiffs(2) = 8.0;
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(0, 1), ssid_index);
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(1, 1), ssid_index2);
      realtimeUnderlyingWorld->setJointSpringStiffIndex(spring_stiffs.segment(2, 1), ssid_index3);
    }
    else if(server.getKeysDown().count("s"))
    {
      renderIsReady = true;
      record = true;
      ssid.start();
      mpcLocal.setPredictUsingFeedback(false);
      mpcLocal.ilqrstart();
    }
    else if(server.getKeysDown().count("r"))
    {
      record = true;
    }
    else if(server.getKeysDown().count("f"))
    {
      renderIsReady = false;
      ssid.stop();
      mpcLocal.ilqrstop();
    }
    else if(server.getKeysDown().count("p") || cnt == 1000)
    {
      if(record)
      {
        Eigen::MatrixXs real_params = std2eigen(real_record);
        Eigen::MatrixXs sysid_params = std2eigen(id_record);
        std::cout << "Converted!" << std::endl;
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/elastic_real_"
                           +std::to_string(filecnt)+".csv", real_params);
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/elastic_identified_"
                           +std::to_string(filecnt)+".csv", sysid_params);
        filecnt++;
      }
      record = false;
    }
    
    if(renderIsReady)
    {
      Eigen::VectorXs pos = realtimeUnderlyingWorld->getPositions();
      s_t err = abs(pos(0) - goal(0));
      if(err < 0.1)
      {
        std::cout << "Goal Reached in: "<< cnt << "Steps" << std::endl;
        exit(0);
      }
    }
    
    if(renderIsReady)
    {
      #ifdef USE_NOISE
      recordObsWithNoise(now, &ssid, realtimeUnderlyingWorld, noise_scale, rand_gen);
      #else
      recordObs(now, &ssid, realtimeUnderlyingWorld);
      #endif
      realtimeUnderlyingWorld->step();
      cnt++;
    }
    id_spring_stiffs(0) = world->getJointSpringStiffIndex(ssid_index)(0);
    id_spring_stiffs(1) = world->getJointSpringStiffIndex(ssid_index2)(0);
    id_spring_stiffs(2) = world->getJointSpringStiffIndex(ssid_index3)(0);
    
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
                        "Current Spring: "+std::to_string(id_spring_stiffs(0))+" "+std::to_string(id_spring_stiffs(1))+" "+std::to_string(id_spring_stiffs(2))+
                        "Real Spring: "+std::to_string(spring_stiffs(0))+" "+std::to_string(spring_stiffs(1))+" "+std::to_string(spring_stiffs(2)),
                        Eigen::Vector2i(100,100),
                        Eigen::Vector2i(200,200));
      if(record && renderIsReady)
      {
        id_record.push_back(id_spring_stiffs);
        real_record.push_back(spring_stiffs);
      }
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