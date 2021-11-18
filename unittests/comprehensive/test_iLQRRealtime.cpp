#include <chrono>
#include <fstream>
#include <iostream>
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

#ifdef ALL_TESTS
TEST(REALTIME, CARTPOLE_MPC)
{
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // Create World of cartpole
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

  cartpole->setControlForceUpperLimit(0, 15);
  cartpole->setControlForceLowerLimit(0, -15);
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

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 30.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  world->setTimeStep(1.0 / 100);

  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = 200 * millisPerTimestep;

  // Create Goal
  Eigen::VectorXs runningStateWeight = Eigen::VectorXs::Zero(2 * 2);
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Zero(1);
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Zero(2 * 2);
  Eigen::VectorXi actuatedJoint = Eigen::VectorXi::Zero(1);
  finalStateWeight(0) = 1.0;
  finalStateWeight(1) = 1.0;

  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight, 
                                           actuatedJoint);

  Eigen::VectorXs goal = Eigen::VectorXs::Zero(4);
  goal(0) = 1.0;
  costFn->setTarget(goal);
  std::cout << "Before MPC Local Initialization" << std::endl;
  iLQRLocal mpcLocal = iLQRLocal(
    world, costFn, actuatedJoint, planningHorizonMillis, 1.0);

  std::cout << "mpcLocal Created Successfully" << std::endl;

  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(5);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);

  std::shared_ptr<simulation::World> realtimeUnderlyingWorld = world->clone();
  GUIWebsocketServer server;

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
  std::cout << "Reach Here Before Ticker" << std::endl;
  Ticker ticker = Ticker(1*realtimeUnderlyingWorld->getTimeStep());

  auto sledBodyVisual = realtimeUnderlyingWorld->getSkeleton("cartpole")
                            ->getBodyNodes()[0]
                            ->getShapeNodesWith<VisualAspect>()[0]
                            ->getVisualAspect();
  Eigen::Vector3s originalColor = sledBodyVisual->getColor();
  long total_steps = 0;
  ticker.registerTickListener([&](long now) {
    //Eigen::VectorXs mpcforces = mpcLocal.computeForce(realtimeUnderlyingWorld->getState(), now);
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
    realtimeUnderlyingWorld->step();
    mpcLocal.recordGroundTruthState(
        now,
        realtimeUnderlyingWorld->getPositions(),
        realtimeUnderlyingWorld->getVelocities(),
        realtimeUnderlyingWorld->getMasses());

    if(total_steps % 5 == 0)
    {
      server.renderWorld(realtimeUnderlyingWorld);
      total_steps = 0;
    }
    total_steps ++;
  });

  // Should only work when trajectory opt
  mpcLocal.registerReplanningListener(
      [&](long ,
          const trajectory::TrajectoryRollout* rollout,
          long ) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });
  
  server.registerConnectionListener([&](){
    ticker.start();
    mpcLocal.start();
    // mpcLocal.ilqrstart();
  });
  server.registerShutdownListener([&]() {mpcLocal.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif