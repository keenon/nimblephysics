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

//#define iLQR_MPC_TEST
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
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, 0, -9.81));
  std::shared_ptr<dynamics::Skeleton> xmate3p 
    = dart::utils::UniversalLoader::loadSkeleton(
        world.get(), 
        "/workspaces/nimblephysics/data/urdf/xmate3p/xmate3p.urdf");
  world->setTimeStep(timestep);
  for(int i = 0; i < xmate3p->getNumDofs(); i++)
  {
    xmate3p->getJoint(i)->setDampingCoefficient(0, 0.01);
  }
  for(int i = 0; i < xmate3p->getNumBodyNodes(); i++)
  {
    // xmate3p->getBodyNode(i)->removeAllShapeNodes();
    for( dart::dynamics::ShapeNode* shapenode :xmate3p->getBodyNode(i)->getShapeNodes())
    {
      // Collision handling may crash current iLQR
      shapenode->removeCollisionAspect();
    }
    
  }
  Eigen::Vector6s init_state;
  init_state << 130.0 / 180.0 * 3.1415, 90.0 / 180.0 * 3.1415, 90.0 / 180.0 * 3.1415, 0, 0, 0;
  world->setState(init_state);
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
  for(int i = 0; i < num_record; i++)
  {
    eigen_record.row(i) = record[i];
  }
  return eigen_record;
}

#ifdef iLQR_MPC_TEST
TEST(REALTIME, CARTPOLE_MPC_MASS)
{
  WorldPtr world = createWorld(0.2 / 100);

  // Initialize Hyper Parameters
  int steps = 100;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  // For add noise in measurement
  #ifdef USE_NOISE
  std::mt19937 rand_gen = initializeRandom();
  s_t noise_scale = 0.005;
  #endif

  // For SSID
  s_t scale = 1.0;
  size_t ssid_index = 6;
  int inferenceSteps = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;

  std::shared_ptr<simulation::World> ssidWorld = cloneWorld(world, false);
  ssidWorld->tuneMass(
    world->getBodyNodeByIndex(ssid_index),
    WrtMassBodyNodeEntryType::INERTIA_COM,
    Eigen::Vector3s(0.2, 0.2, 0.2),
    Eigen::Vector3s(0., 0., 0.));

  Eigen::Vector2s sensorDims(world->getNumDofs(), world->getNumDofs());
  std::vector<size_t> ssid_idx{ssid_index};//, ssid_index2};

  SSID ssid = SSID(ssidWorld,
                   getSSIDVelLoss(),
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
  ssid.setTemperature(Eigen::Vector3s(4, 4, 4));
  ssid.setThreshs(0.002, 0.5);
  
  Eigen::VectorXi index;
  index = Eigen::VectorXi::Zero(1);
  index(0) = ssid_index;
  ssid.setSSIDCOMIndex(index);

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
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Ones(dofs) * 0.2;
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Ones(2 * dofs) * 100.0;

  runningStateWeight.segment(0, dofs) = Eigen::VectorXs::Ones(dofs) * 0.01;
  finalStateWeight.segment(dofs, dofs) *= 0.5;
  finalStateWeight(2) *= 5;

  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = cloneWorld(world,true);

  clearShapeNodes(world);
  
  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight,
                                           world);

  costFn->setSSIDNodeIndex(ssid_idx);
  // costFn->enableSSIDLoss(0.01);
  costFn->setTimeStep(world->getTimeStep());
  Eigen::Vector6s goal;
  goal << 60.0 / 180.0 * 3.1415, 90.0 / 180.0 * 3.1415, 0.0 / 180.0 * 3.1415, 0, 0, 0;
  Eigen::Vector6s goal2;
  goal2 << 0.0/180.0, 0, 0, 0, 0, 0;
  

  costFn->setTarget(goal2);
  std::cout << "Goal: " << goal << std::endl;
  iLQRLocal mpcLocal = iLQRLocal(
    world, dofs, planningHorizonMillis, 1.0);
  mpcLocal.setCostFn(costFn);
  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(5);
  mpcLocal.setPatience(1);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);
  mpcLocal.setActionBound(30.0);
  mpcLocal.setAlpha(1);

  ssid.registerInferListener([&](long,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs diag_I,
                                 long){
    // mpcLocal.recordGroundTruthState(time, pos, vel, mass); //TODO: This will cause problem ... But Why
    mpcLocal.setParameterChange(diag_I);
    world->setLinkCOMIndex(diag_I, ssid_index);
  });


  
  GUIWebsocketServer server;

  std::string key = "mass";

  Ticker ticker = Ticker(scale * realtimeUnderlyingWorld->getTimeStep());
  long total_steps = 0;

  Eigen::Vector3s diag_Is;
  diag_Is = world->getLinkCOMIndex(ssid_index);
  // masses(1) = 0;
  Eigen::Vector3s id_diag_Is(0.01, 0.02, 0.021);

  realtimeUnderlyingWorld->setLinkCOMIndex(diag_Is, ssid_index);
  ssidWorld->setLinkCOMIndex(id_diag_Is, ssid_index);
  world->setLinkCOMIndex(id_diag_Is, ssid_index);
  // Preload visualization
  bool renderIsReady = false;
  int filecnt = 0;
  int cnt = 0;
  bool record = false;
  std::vector<Eigen::VectorXs> real_record;
  std::vector<Eigen::VectorXs> id_record;
  ticker.registerTickListener([&](long now) {
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
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }
    else if (server.getKeysDown().count("e"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = 15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
    }
    else if (server.getKeysDown().count(",") || cnt == 300)
    {
      // Increase mass
      diag_Is(1) = 0.025;
      diag_Is(2) = 0.030;
      realtimeUnderlyingWorld->setLinkCOMIndex(diag_Is, ssid_index);
    }
    else if (server.getKeysDown().count("c") || cnt == 500)
    {
      costFn->setTarget(goal2);
    }
    else if (server.getKeysDown().count("o") || cnt == 600)
    {
      // Decrease mass
      diag_Is(0) = 0.050;
      diag_Is(1) = 0.050;
      diag_Is(2) = 0.020;
      realtimeUnderlyingWorld->setLinkCOMIndex(diag_Is, ssid_index);
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
    else if(server.getKeysDown().count("p") || cnt == 1000)
    {
      if(record)
      {
        Eigen::MatrixXs real_params = std2eigen(real_record);
        Eigen::MatrixXs sysid_params = std2eigen(id_record);
        std::cout << "Converted!" << std::endl;
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/arm_real_"
                           +std::to_string(filecnt)+".csv", real_params);
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/arm_identified_"
                           +std::to_string(filecnt)+".csv", sysid_params);
        filecnt++;
      }
      record = false;
    }
    if(renderIsReady)
    {
      s_t err = (realtimeUnderlyingWorld->getState() - goal2).norm();
      if( err < 0.2)
      {
        std::cout << "--------------------------------------------"<< std::endl;
        std::cout << "--------------------------------------------"<< std::endl;
        std::cout << "Target Reached in: " << cnt << std::endl;
        std::cout << "--------------------------------------------"<< std::endl;
        std::cout << "--------------------------------------------"<< std::endl;
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
    
    id_diag_Is = world->getLinkCOMIndex(ssid_index);
    
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
      Eigen::Vector3s moi = realtimeUnderlyingWorld->getLinkCOMIndex(ssid_index);
      server.createText(key,
                        "Current MOIs: "+std::to_string(id_diag_Is(0))+" "+std::to_string(id_diag_Is(1))+" "+std::to_string(id_diag_Is(2))+
                        "Real MOIs: "+std::to_string(moi(0))+" "+std::to_string(moi(1))+" "+std::to_string(moi(2)),
                        Eigen::Vector2i(100,100),
                        Eigen::Vector2i(400,400));
      if(record && renderIsReady)
      {
        id_record.push_back(id_diag_Is);
        real_record.push_back(diag_Is);
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