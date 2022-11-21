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
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/UniversalLoader.hpp"

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
  WorldPtr world = dart::utils::UniversalLoader::loadWorld("dart://sample/skel/cartpole.skel");
  world->setState(Eigen::Vector4s(0, 175.0 / 180 * 3.14, 0, 0));
  world->removeDofFromActionSpace(1);
  
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

std::vector<s_t> convert2stdvec(Eigen::VectorXs vec)
{
  std::vector<s_t> stdvec;
  for(int i = 0; i < vec.size(); i++)
  {
    stdvec.push_back(vec(i));
  }
  return stdvec;
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

Eigen::MatrixXs std2eigen(std::vector<Eigen::VectorXs> record)
{
  size_t num_record = record.size();
  Eigen::MatrixXs eigen_record = Eigen::MatrixXs::Zero(num_record, record[0].size());
  std::cout << "Size: " << eigen_record.cols() << " " << eigen_record.rows() << std::endl; 
  for(int i = 0; i < num_record; i++)
  {
    eigen_record.row(i) = record[i];
  }
  return eigen_record;
}

#ifdef iLQR_MPC_TEST
TEST(REALTIME, CARTPOLE_MPC_MASS)
{
  WorldPtr world = createWorld(1.0 / 100);

  // Initialize Hyper Parameters
  int steps = 100;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  // For add noise in measurement
  std::mt19937 rand_gen = initializeRandom();
  s_t noise_scale = 1. / 180 * 3.14;

  // For SSID
  s_t scale = 1.0;
  size_t ssid_index = 0;
  size_t ssid_index2 = 1;
  int inferenceSteps = 10;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;

  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  ssidWorld->tuneMass(
    world->getBodyNodeByIndex(ssid_index),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);

  ssidWorld->tuneMass(
    world->getBodyNodeByIndex(ssid_index2),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);
  
  ssidWorld->tuneDamping(
    world->getJointIndex(1),
    WrtDampingJointEntryType::DAMPING,
    Eigen::VectorXi::Ones(1),
    Eigen::VectorXs::Ones(1),
    Eigen::VectorXs::Ones(1) * 0.01);

  Eigen::Vector2s sensorDims(world->getNumDofs(), world->getNumDofs());
  Eigen::Vector2i ssid_idx(0,1);

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
  ssid.setSSIDMassIndex(Eigen::Vector2i(0, 1)); 
  ssid.setSSIDDampIndex(Eigen::VectorXi::Ones(1));
  ssid.useConfidence();
  ssid.useHeuristicWeight();
  ssid.useSmoothing();
  ssid.setTemperature(Eigen::Vector3s(0.5, 5, 2));
  // ssid.setThreshs(0.1, 0.5); // As in paper
  ssid.setEWiseThreshs(Eigen::Vector3s(0.1, 0.1, 0.02), 0.5);
  

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
  Eigen::VectorXs runningStateWeight = Eigen::VectorXs::Zero(2 * 2);
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Zero(1);
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Zero(2 * 2);
  finalStateWeight(0) = 10.0;
  finalStateWeight(1) = 50.0;
  finalStateWeight(2) = 10.0;
  finalStateWeight(3) = 10.0;
  runningActionWeight(0) = 0.01;

  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight,
                                           world);

  costFn->setSSIDMassNodeIndex(ssid_idx);
  // costFn->enableSSIDLoss(0.01);
  costFn->setTimeStep(world->getTimeStep());
  Eigen::VectorXs goal = Eigen::VectorXs::Zero(4);
  goal(0) = 1.0;
  costFn->setTarget(goal);
  std::cout << "Before MPC Local Initialization" << std::endl;
  iLQRLocal mpcLocal = iLQRLocal(
    world, 1, planningHorizonMillis, 1.0);
  mpcLocal.setCostFn(costFn);

  std::cout << "mpcLocal Created Successfully" << std::endl;

  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(5);
  mpcLocal.setPatience(3);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);
  mpcLocal.setActionBound(20.0);
  mpcLocal.setAlpha(1);

  //mpcLocal.disableAdaptiveTime();

  // bool init_flag = true;

  ssid.registerInferListener([&](long,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs,
                                 Eigen::VectorXs params,
                                 long, 
                                 bool steadyFound){
    // mpcLocal.recordGroundTruthState(time, pos, vel, mass); //TODO: This will cause problem ... But Why
    // If there are more than one type of parameters will this work?
    if(steadyFound)
    {
      mpcLocal.setCandidateHorizon(planningHorizonMillis);
    }
    else
    {
      mpcLocal.setCandidateHorizon((size_t)(1*planningHorizonMillis));
    }
    world->setLinkMassIndex(params(0), ssid_index);
    world->setLinkMassIndex(params(1), ssid_index2);
    // The rest of segment should be damping
    world->setJointDampingCoeffIndex(params.segment(2,1), 1);
  });


  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = world->clone();
  GUIWebsocketServer server;

  server.createSphere("goal", 0.1,
                      Eigen::Vector3s(goal(0), 0.7, 0),
                      Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
  server.registerDragListener("goal", [&](Eigen::Vector3s dragTo){
    goal(0) = dragTo(0);
    dragTo(1) = 0.3;
    dragTo(2) = 0.0;
    costFn->setTarget(goal);
    server.setObjectPosition("goal", dragTo);
  });

  std::string key = "mass";

  Ticker ticker = Ticker(scale * realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton(0)
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  long total_steps = 0;

  Eigen::Vector3s params(1.0, 0.5, 0.2);
  Eigen::Vector3s id_params(1.5, 1.0, 0.4);

  realtimeUnderlyingWorld->setLinkMassIndex(params(0), ssid_index);
  realtimeUnderlyingWorld->setLinkMassIndex(params(1), ssid_index2);
  realtimeUnderlyingWorld->setJointDampingCoeffIndex(params.segment(2,1), 1);
  ssidWorld->setLinkMassIndex(id_params(0), ssid_index);
  ssidWorld->setLinkMassIndex(id_params(1), ssid_index2);
  ssidWorld->setJointDampingCoeffIndex(id_params.segment(2,1),1);
  world->setLinkMassIndex(id_params(0), ssid_index);
  world->setLinkMassIndex(id_params(1), ssid_index2);
  world->setJointDampingCoeffIndex(id_params.segment(2,1),1);

  int cnt = 0;
  int filecnt = 0;
  bool renderIsReady = false;
  bool record = false;
  std::vector<Eigen::VectorXs> real_record;
  std::vector<Eigen::VectorXs> id_record; 
  ticker.registerTickListener([&](long now) {
    if(renderIsReady)
    {
      Eigen::VectorXs mpcforces = mpcLocal.computeForce(realtimeUnderlyingWorld->getState(), now);
      Eigen::VectorXs force_eps = rand_normal(mpcforces.size(), 0, noise_scale, rand_gen);
      realtimeUnderlyingWorld->setControlForces(mpcforces + force_eps);
    }
    if (server.getKeysDown().count("a"))
    {
      Eigen::VectorXs perturbedForces
          = realtimeUnderlyingWorld->getControlForces();
      perturbedForces(0) = -15.0;
      realtimeUnderlyingWorld->setControlForces(perturbedForces);
      sledBodyVisual->setColor(Eigen::Vector3s(1, 0, 0));
    }
    else if (server.getKeysDown().count("d"))
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
    if (server.getKeysDown().count(",") || cnt == 150)
    {
      // Increase mass
      params(0) = 3.0;
      params(1) = 2.5;
      realtimeUnderlyingWorld->setLinkMassIndex(params(0), ssid_index);
      realtimeUnderlyingWorld->setLinkMassIndex(params(1), ssid_index2);
    }
    else if (server.getKeysDown().count("o") || cnt == 600)
    {
      // Decrease mass
      params(0) = 1.0;
      params(2) = 0.1;
      realtimeUnderlyingWorld->setLinkMassIndex(params(0), ssid_index);
      realtimeUnderlyingWorld->setJointDampingCoeffIndex(params.segment(2,1), 1);
    }
    else if(server.getKeysDown().count("s"))
    {
      renderIsReady = true;
      record = true;
      ssid.start();
      //ssid.startSlow(); // Which should be useless
      mpcLocal.setPredictUsingFeedback(false);
      mpcLocal.ilqrstart();
    }
    else if(server.getKeysDown().count("r"))
    {
      record = true;
    }
    else if(server.getKeysDown().count("p") || cnt == 940)
    {
      if(record)
      {
        Eigen::MatrixXs real_params = std2eigen(real_record);
        Eigen::MatrixXs sysid_params = std2eigen(id_record);
        std::cout << "Converted!" << std::endl;
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/cartpole_10_real_"+std::to_string(filecnt)+".csv", real_params);
        ssid.saveCSVMatrix("/workspaces/nimblephysics/dart/realtime/saved_data/timeplots/cartpole_10_identified_"+std::to_string(filecnt)+".csv", sysid_params);
        filecnt ++;
      }
      record = false;
    }
    // recordObs(now, &ssid, realtimeUnderlyingWorld);
    if(renderIsReady)
    {
      // if((realtimeUnderlyingWorld->getState()-goal).norm()<0.2)
      // {
      //   std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
      //   std::cout << "Target Reached in: " << cnt << " Steps" << std::endl;
      //   std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
      //   //ssid.stop();
      //   //mpcLocal.stop();
      //   //exit(1);
      // }
      recordObsWithNoise(now, &ssid, realtimeUnderlyingWorld, noise_scale, rand_gen);
      realtimeUnderlyingWorld->step();
      cnt++;
    }
    id_params(0) = world->getLinkMassIndex(ssid_index);
    id_params(1) = world->getLinkMassIndex(ssid_index2);
    id_params(2) = world->getJointDampingCoeffIndex(1)(0);
    if(renderIsReady)
    {
      mpcLocal.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses()); // TODO: Why only mass
    }
  
    if(total_steps % 5 == 0)
    {
      server.renderWorld(realtimeUnderlyingWorld);
      server.createText(key,
                        "Cur Params: "+std::to_string(id_params(0))+" "+std::to_string(id_params(1))+" "+std::to_string(id_params(2))
                        +"Real Params: "+std::to_string(params(0))+" "+std::to_string(params(1))+" "+std::to_string(params(2)),
                        Eigen::Vector2i(100,100),
                        Eigen::Vector2i(200,200));
      if(record && renderIsReady)
      {
        id_record.push_back(id_params);
        real_record.push_back(params);
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