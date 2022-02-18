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

#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/iLQRLocal.hpp"
#include "dart/realtime/SSID.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/UniversalLoader.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

#define iLQR_TEST

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;

WorldPtr createWorld(s_t timestep)
{
  WorldPtr world = dart::utils::UniversalLoader::loadWorld("dart://sample/skel/cartpole.skel");
  world->setState(Eigen::Vector4s(0, 170.0/180 * 3.14, 0, 0));
  world->removeDofFromActionSpace(1);
  world->setTimeStep(timestep);
  return world;
}

#ifdef iLQR_TEST
TEST(REALTIME, CARTPOLE_ILQR)
{
  // Create the world
  WorldPtr world = createWorld(1.0/100);

  // Create iLQR instance
  int steps = 400;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

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
  costFn->setTimeStep(world->getTimeStep());
  Eigen::VectorXs goal = Eigen::VectorXs::Zero(4);
  goal(0) = 0.5;
  costFn->setTarget(goal);
  std::cout << "Before MPC Local Initialization" << std::endl;
  std::cout << "Planning Millis: " << planningHorizonMillis << std::endl;
  iLQRLocal ilqr = iLQRLocal(
    world, 1, planningHorizonMillis, 1.0);
  ilqr.setCostFn(costFn);

  Eigen::VectorXs init_state = world->getState();

  std::cout << "mpcLocal Created Successfully" << std::endl;
  int maxIter = 10;
  ilqr.setSilent(true);
  ilqr.setMaxIterations(maxIter);
  ilqr.setAlpha(0.5);
  ilqr.setPatience(1);
  ilqr.setActionBound(100.0);

  // Initialize a fresh rollout for loss computation
  TrajectoryRolloutReal rollout = ilqr.createRollout(steps, world->getNumDofs(), 
                                                     world->getMassDims(), 
                                                     world->getDampingDims(), 
                                                     world->getSpringDims());

  // Define a lambda function for simulate traj
  auto simulate_traj = [&](std::vector<Eigen::VectorXs> X, std::vector<Eigen::VectorXs> U, bool render)
  {
    world->setState(init_state);
    if(render)
    {
      for(int i = 0; i < steps-1; i++)
      {
        rollout.getPoses().col(i) = world->getPositions();
        rollout.getVels().col(i) = world->getVelocities();
        rollout.getControlForces().col(i) = world->mapToForceSpaceVector(U[i]);
        world->setAction(U[i]);
        world->step();
        X[i+1] = world->getState();
        usleep(10000);
      }
      rollout.getPoses().col(steps-1) = world->getPositions();
      rollout.getVels().col(steps-1) = world->getVelocities();
      s_t loss = costFn->computeLoss(&rollout);
      return loss;
    }
    else
    {
      for(int i = 0; i < steps-1; i++)
      {
        rollout.getPoses().col(i) = world->getPositions();
        rollout.getVels().col(i) = world->getVelocities();
        rollout.getControlForces().col(i) = world->mapToForceSpaceVector(U[i]);
        world->setAction(U[i]);
        world->step();
        X[i+1] = world->getState();
      }
      rollout.getPoses().col(steps-1) = world->getPositions();
      rollout.getVels().col(steps-1) = world->getVelocities();
      s_t loss = costFn->computeLoss(&rollout);
      return loss;
    }
  };
  // Instead of starting a single thread, iLQR Trajectory optimization from starting state
  s_t init_cost = simulate_traj(ilqr.getStatesFromiLQRBuffer(), ilqr.getActionsFromiLQRBuffer(), false);
  std::cout << "Initial Cost: " << init_cost << std::endl;
  ilqr.setCurrentCost(init_cost);
  s_t prev_cost = 1e10;
  s_t threshold = 0.01;
  
  // Print out current parameters settings
  std::cout << "Alpha: " << ilqr.getAlpha() << "\n"
            << "MU: " << ilqr.getMU() << std::endl;
  
  int iter = 0;
  while(iter < maxIter)
  {
    // Set the world to initial state
    world->setState(init_state);
    
    bool forwardFlag = ilqr.ilqrForward(world);
    bool backwardFlag = false;
    if(!forwardFlag)
    {
      std::cout << "Optimization Terminated, Exiting ..." <<std::endl;
      break;
    }
    else
    {
      backwardFlag = ilqr.ilqrBackward();
    }
    std::cout << "Iteration: " << iter+1 << " Cost: " << ilqr.getCurrentCost() << std::endl;
    if(!backwardFlag)
    {
      std::cout << "Backward Terminated, Exiting ..." << std::endl;
      break; 
    }
    if(abs(prev_cost-ilqr.getCurrentCost()) < threshold)
    {
      std::cout << "Optimization Converged, Existing ..." << std::endl;
      break;
    }
    prev_cost = ilqr.getCurrentCost();
    iter++;
  }

  // Demonstrate the performance
  s_t final_cost = 0;
  for(int i = 0; i < 5; i++)
  {
    final_cost = simulate_traj(ilqr.getStatesFromiLQRBuffer(), ilqr.getActionsFromiLQRBuffer(), true);
  }
  std::cout << "Final Cost: " << final_cost << std::endl;
  EXPECT_TRUE(final_cost < 5);
}
#endif