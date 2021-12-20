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
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

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
using namespace server;


#ifdef iLQR_TEST
TEST(REALTIME, CARTPOLE_ILQR)
{
  // Create the world
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0.0));
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

  cartpole->setControlForceUpperLimit(0, 20);
  cartpole->setControlForceLowerLimit(0, -20);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);
  // The second DOF cannot be controlled
  cartpole->setControlForceUpperLimit(1, 0);
  cartpole->setControlForceLowerLimit(1, 0);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 1.0);
  cartpole->setPosition(1, 3.1415);

  world->setTimeStep(1.0 / 100);

  // Create iLQR instance
  int steps = 400;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  world->removeDofFromActionSpace(1);
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
    world, costFn, 1, planningHorizonMillis, 1.0);

  Eigen::VectorXs init_state = world->getState();

  std::cout << "mpcLocal Created Successfully" << std::endl;
  int maxIter = 10;
  ilqr.setSilent(true);
  ilqr.setMaxIterations(maxIter);
  ilqr.setAlpha(0.5);
  ilqr.setPatience(1);
  ilqr.setActionBound(100.0);
  // Create Server for rendering
  GUIWebsocketServer server;
  server.serve(8070);
  // Initialize a fresh rollout for loss computation
  TrajectoryRolloutReal rollout = ilqr.createRollout(steps, world->getNumDofs(), world->getMassDims());

  // Define a lambda function for simulate traj
  auto simulate_traj = [&](std::vector<Eigen::VectorXs> X, std::vector<Eigen::VectorXs> U, bool render)
  {
    world->setState(init_state);
    if(render)
    {
      server.renderWorld(world);
      for(int i = 0; i < steps-1; i++)
      {
        rollout.getPoses().col(i) = world->getPositions();
        rollout.getVels().col(i) = world->getVelocities();
        rollout.getControlForces().col(i) = world->mapToForceSpaceVector(U[i]);
        world->setAction(U[i]);
        world->step();
        X[i+1] = world->getState();
        server.renderWorld(world);
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
}
#endif