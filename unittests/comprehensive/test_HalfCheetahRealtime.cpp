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
#include <math.h>

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

//#define ALL_TESTS
//#define NO_VIS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;
using namespace server;

enum BODY_NODE
{
  H_PELVIS = 1,
  H_PELVIS_AUX,
  H_PELVIS_AUX2,
  H_HEAD,
  B_THIGH,
  B_SHIN,
  B_FOOT,
  F_THIGH,
  F_SHIN,
  F_FOOT
};

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

#ifdef NO_VIS

TEST(HALF_CHEETAH, FIRST_DEMO)
{
  std::shared_ptr<simulation::World> world = dart::utils::UniversalLoader::loadWorld(
    "dart://sample/skel/half_cheetah.skel");
  world->setPositions(Eigen::VectorXs::Zero(world->getNumDofs()));
  world->setVelocities(Eigen::VectorXs::Zero(world->getNumDofs()));
  std::cout <<"Time Step:" << world->getTimeStep() << std::endl;

  std::shared_ptr<dynamics::Skeleton> halfCheetah
      = world->getSkeleton("half_cheetah");
  std::cout << "Springs: " << halfCheetah->getSpringStiffVector() << std::endl;
  std::cout << "Damping: " << halfCheetah->getDampingCoeffVector() << std::endl;
  
  for(int i=0; i < 3; i++)
  {
    world->step();
    std::cout << i << ":" <<std::endl;
    std::cout << "Pos:" << std::endl << world->getPositions() << std::endl;
    std::cout << "Vel:" << std::endl << world->getVelocities() << std::endl;
    if(world->getPositions().hasNaN())
    {
      std::cout << "Got a NaN on timestep " << i << "!!" <<std::endl;
      EXPECT_FALSE(world->getPositions().hasNaN());
      break;
    }
  }
}


TEST(REALTIME, HALF_CHEETAH_PLOT)
{
  ////////////////////////////////////////////////////////////
  // Create a cartpole example
  ////////////////////////////////////////////////////////////

  // World
  std::shared_ptr<simulation::World> world = dart::utils::UniversalLoader::loadWorld(
      "dart://sample/skel/half_cheetah.skel");
  world->setPositions(Eigen::VectorXs::Zero(world->getNumDofs()));
  world->setVelocities(Eigen::VectorXs::Zero(world->getNumDofs()));

  
  Eigen::VectorXs forceLimits
    = Eigen::VectorXs::Ones(world->getNumDofs()) * 50;
  forceLimits(0) = 0;
  forceLimits(1) = 0;
  world->setControlForceUpperLimits(forceLimits);
  world->setControlForceLowerLimits(-1 * forceLimits);
  world->setTimeStep(1.0 / 1000);
  
  size_t mass_idx = 10;

  world->tuneMass(
      world->getBodyNodeByIndex(mass_idx),
      WrtMassBodyNodeEntryType::INERTIA_MASS,
      Eigen::VectorXs::Ones(1) * 5.0,
      Eigen::VectorXs::Ones(1) * 0.2);

  std::shared_ptr<LossFn> lossFn = getSSIDPosLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelLoss();
  //std::shared_ptr<LossFn> lossFn = getSSIDVelPosLoss();
  ////////////////////////////////
  // Set up a realtime world and controller
  ////////////////////////////////////////////////////////////


  int millisPerTimestep = world->getTimeStep() * 1000;
  int steps = 5;
  int inferenceHistoryMillis = steps * millisPerTimestep;

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
  
  
  world->setLinkMassIndex(2.0,mass_idx);
  float init_mass = 1.0;
  std::vector<Eigen::VectorXs> multi_loss;
  
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
    
    if(i%5==0 && i!=0)
    {
      multi_loss.push_back(ssid.runPlotting(time,5.0,0.2,200));
      world->setLinkMassIndex(init_mass,mass_idx);
      ssid.runInference(time);
      init_mass = world->getLinkMassIndex(mass_idx);
      std::cout << "Recovered mass after iteration "<<i<<": "
                << init_mass << std::endl;
      solutions.push_back(init_mass);
      world->setLinkMassIndex(2.0,mass_idx);
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
  file.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Losses_half_cheetah.txt");
  file<<lossMatrix.transpose();
  file.close();
  
  std::ofstream sfile;
  sfile.open("/workspaces/nimblephysics/dart/realtime/saved_data/raw_data/Solutions_half_cheetah.txt");
  sfile<<solutionVec;
  sfile.close();
}
#endif

#ifdef ALL_TESTS


TEST(HALF_CHEETAH, FULL_TEST)
{
  // set precision to 256 bits (double has only 53 bits)
// #ifdef DART_USE_ARBITRARY_PRECISION
//   mpfr::mpreal::set_default_prec(256);
// #endif

  // Create a world
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/half_cheetah.skel");
  // world->setSlowDebugResultsAgainstFD(true);
  // world->setTimeStep(2.0/1000);

  for (auto* dof : world->getDofs())
  {
    std::cout << "DOF: " << dof->getName() << std::endl;
  }

  Eigen::VectorXs forceLimits
      = Eigen::VectorXs::Ones(world->getNumDofs()) * 100;
  forceLimits(0) = 0;
  forceLimits(1) = 0;
  forceLimits(2) = 0;
  world->setControlForceUpperLimits(forceLimits);
  world->setControlForceLowerLimits(-1 * forceLimits);

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  // Create target

  s_t target_x = 3.5;
  s_t target_y = 0.5;

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3s targetOffset = Eigen::Isometry3s::Identity();
  targetOffset.translation() = Eigen::Vector3s(target_x, target_y, 0.0);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3s(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  trajectory::LossFn loss(
      [target_x, target_y](const trajectory::TrajectoryRollout* rollout) {
        const Eigen::VectorXs lastPos
            = rollout->getPosesConst().col(rollout->getPosesConst().cols() - 1);

        s_t diffX = lastPos(0) - target_x;
        s_t diffY = lastPos(1) - target_y;

        return diffX * diffX + diffY * diffY;
      });

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 200, 10, false);
  trajectory->setParallelOperationsEnabled(false);

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  // optimizer.setCheckDerivatives(true);
  optimizer.setIterationLimit(300);
  optimizer.registerIntermediateCallback([&](trajectory::Problem* problem,
                                             int /* step */,
                                             s_t /* primal */,
                                             s_t /* dual */) {
    const Eigen::MatrixXs poses
        = problem->getRolloutCache(world)->getPosesConst();
    const Eigen::MatrixXs vels
        = problem->getRolloutCache(world)->getVelsConst();
    std::cout << "Rendering trajectory lines" << std::endl;
    server.renderTrajectoryLines(world, poses);
    world->setPositions(poses.col(0));
    server.renderWorld(world);
    return true;
  });
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  int i = 0;
  const Eigen::MatrixXs poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXs vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long /* time */) {
    world->setPositions(poses.col(i));

    i++;
    if (i >= poses.cols())
    {
      i = 0;
    }
    // world->step();
    std::cout << "Frame: " << i << std::endl;
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {}
}

TEST(REALTIME, CARTPOLE_MPC)
{
  ////////////////////////////////////////////////////////////
  // Create a half Cheetah example
  ////////////////////////////////////////////////////////////

  // World
  std::shared_ptr<simulation::World> world = dart::utils::UniversalLoader::loadWorld(
      "dart://sample/skel/half_cheetah.skel");
  world->setPositions(Eigen::VectorXs::Zero(world->getNumDofs()));
  world->setVelocities(Eigen::VectorXs::Zero(world->getNumDofs()));

  
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
  world->setTimeStep(2.0 / 1000);
  s_t scale = 10.0;

  // 300 timesteps
  int millisPerTimestep = scale * world->getTimeStep() * 1000;
  int planningHorizonMillis = 100 * millisPerTimestep;

  // Create target

  s_t target_x = 2.5;
  s_t target_y = 0.5;

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
  size_t mass_idx = 5;
  int inferenceHistoryMillis = inferenceSteps * millisPerTimestep;
  std::shared_ptr<simulation::World> ssidWorld = world->clone();
  
  // Need to get a body node which is useful for SSID
  ssidWorld->tuneMass(
    world->getBodyNodeByIndex(mass_idx),
    WrtMassBodyNodeEntryType::INERTIA_MASS,
    Eigen::VectorXs::Ones(1) * 5.0,
    Eigen::VectorXs::Ones(1) * 0.2);
  
  Eigen::VectorXs sensorDims = Eigen::VectorXs::Zero(2);
  sensorDims(0) = world->getNumDofs();
  sensorDims(1) = world->getNumDofs();
  SSID ssid = SSID(
      ssidWorld, getSSIDPosLoss(), inferenceHistoryMillis, sensorDims,inferenceSteps, scale);
  
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
      world, std::make_shared<LossFn>(loss, lossGrad), planningHorizonMillis, scale);
  mpcLocal.setSilent(true);
  
  mpcLocal.setMaxIterations(20);

  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  MPCLocal& mpcRemote = mpcLocal;
  
  bool init_flag = true;
  float weight = 0;
  float stddev = 10;
  float gamma  = 0.9;
  ssid.registerInferListener([&](long time,
                                 Eigen::VectorXs pos,
                                 Eigen::VectorXs vel,
                                 Eigen::VectorXs mass,
                                 long) {
    mass(0) = 1.0;
    mpcRemote.recordGroundTruthState(time, pos, vel, mass);
    mpcRemote.setMasschange(mass(0));
    if(!init_flag)
    {
      s_t old_mass = world->getLinkMassIndex(mass_idx);
      weight = exp(-(mass(0)-old_mass)*(mass(0)-old_mass)/stddev);
      world->setLinkMassIndex((1-weight)*old_mass+weight*mass(0),mass_idx);
    }
    else
    {
      world->setLinkMassIndex(mass(0),mass_idx);
      init_flag = false;
    }
    if(stddev > 1)
      stddev*= gamma;
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
    dragTo(1) = 0.5;
    dragTo(2) = 0.0;
    server.setObjectPosition("goal", dragTo);
  });
  std::string key = "mass";

  Ticker ticker = Ticker(scale*realtimeUnderlyingWorld->getTimeStep());

  float mass = 1.0;
  float id_mass = 1.0;
  size_t total_step = 0;
  realtimeUnderlyingWorld->setLinkMassIndex(mass,mass_idx);
  ssidWorld->setLinkMassIndex(id_mass,mass_idx);
  world->setLinkMassIndex(id_mass,mass_idx);

  ticker.registerTickListener([&](long now) {
    Eigen::VectorXs mpcforces = mpcRemote.getControlForce(now);
    //Eigen::VectorXs mpcforces = 20*Eigen::VectorXs::Random(world->getNumDofs());
    //mpcforces -= 10*Eigen::VectorXs::Ones(world->getNumDofs());
    // std::cout<<"Norm of control force: \n"<<mpcforces<<std::endl;
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
      realtimeUnderlyingWorld->setLinkMassIndex(mass,mass_idx);
    }
    else if (server.getKeysDown().count("o"))
    {
      // Decrease mass
      mass = 1.0;
      realtimeUnderlyingWorld->setLinkMassIndex(mass,mass_idx);
    }
    
    ssid.registerLock();
    ssid.registerControls(now, realtimeUnderlyingWorld->getControlForces());
    ssid.registerSensors(now, realtimeUnderlyingWorld->getPositions(),0);
    ssid.registerSensors(now, realtimeUnderlyingWorld->getVelocities(),1);
    ssid.registerUnlock();
    realtimeUnderlyingWorld->step();
    id_mass = world->getLinkMassIndex(mass_idx);
    mpcRemote.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());
    if(total_step % 10 == 0)
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
    //ssid.start();
  });
  server.registerShutdownListener([&]() { mpcRemote.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}

#endif