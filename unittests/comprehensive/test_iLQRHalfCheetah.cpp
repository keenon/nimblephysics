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
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

//#define iLQR_MPC_TEST

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;
using namespace trajectory;
using namespace server;

#ifdef iLQR_MPC_TEST
TEST(REALTIME, CARTPOLE_MPC)
{
  std::shared_ptr<simulation::World> world = dart::utils::UniversalLoader::loadWorld(
      "dart://sample/skel/half_cheetah.skel");
  world->setPositions(Eigen::VectorXs::Zero(world->getNumDofs()));
  world->setVelocities(Eigen::VectorXs::Zero(world->getNumDofs()));

  
  Eigen::VectorXs forceLimits
    = Eigen::VectorXs::Ones(world->getNumDofs()) * 100;
  forceLimits(0) = 0;
  forceLimits(1) = 0;
  forceLimits(2) = 0;
  world->setControlForceUpperLimits(forceLimits);
  world->setControlForceLowerLimits(-1 * forceLimits);

  world->setTimeStep(2.0 / 1000);

  int steps = 200;
  int millisPerTimestep = world->getTimeStep() * 1000;
  int planningHorizonMillis = steps * millisPerTimestep;

  world->removeDofFromActionSpace(0);
  world->removeDofFromActionSpace(1);
  world->removeDofFromActionSpace(2);
  // Create Goal
  Eigen::VectorXs runningStateWeight = Eigen::VectorXs::Ones(2 * world->getNumDofs())*5;
  Eigen::VectorXs runningActionWeight = Eigen::VectorXs::Ones(world->getNumDofs()-3)*0.1;
  Eigen::VectorXs finalStateWeight = Eigen::VectorXs::Ones(2 * world->getNumDofs())*1;
  finalStateWeight(0) = 20.0;
  finalStateWeight(1) = 20.0;
  // Speed must be zero
  finalStateWeight(world->getNumDofs()) = 20.0;
  finalStateWeight(world->getNumDofs() + 1) = 20.0;
  // Running State need to have some weight
  runningStateWeight(0) = 10;
  runningStateWeight(1) = 10;
  
  std::shared_ptr<TargetReachingCost> costFn
    = std::make_shared<TargetReachingCost>(runningStateWeight,
                                           runningActionWeight, 
                                           finalStateWeight,
                                           world);
  costFn->setTimeStep(world->getTimeStep());
  Eigen::VectorXs goal = world->getState();
  goal(0) += 2.5;
  goal(1) = 0.5;
  goal(2) += 2.5;
  costFn->setTarget(goal);
  std::cout << "Before MPC Local Initialization" << std::endl;
  iLQRLocal mpcLocal = iLQRLocal(
    world, costFn, world->getNumDofs()-3, planningHorizonMillis, 1.0);

  std::cout << "mpcLocal Created Successfully" << std::endl;

  mpcLocal.setSilent(true);
  mpcLocal.setMaxIterations(3);
  mpcLocal.setPatience(1);
  mpcLocal.setEnableLineSearch(false);
  mpcLocal.setEnableOptimizationGuards(true);
  mpcLocal.setActionBound(20.0);
  mpcLocal.setAlpha(1);

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

  long total_steps = 0;
  ticker.registerTickListener([&](long now) {
    // Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
    Eigen::VectorXs mpcforces = mpcLocal.computeForce(realtimeUnderlyingWorld->getState(), now);
    //std::cout << "MPC Force: \n" << mpcforces 
    //          << "\nRef forces: \n" << feedback_forces << std::endl;
    // TODO: Currently the forces are almost zero need to figure out why
    // std::cout <<"Force:\n" << mpcforces << std::endl;
    //Eigen::VectorXs mpcforces = mpcLocal.getControlForce(now);
    realtimeUnderlyingWorld->setControlForces(mpcforces);
    
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
  // We can always feed the trajectory used in forward pass here
  mpcLocal.registerReplanningListener(
      [&](long ,
          const trajectory::TrajectoryRollout* rollout,
          long ) {
        server.renderTrajectoryLines(world, rollout->getPosesConst());
      });
  
  server.registerConnectionListener([&](){
    ticker.start();
    // mpcLocal.start();
    mpcLocal.setPredictUsingFeedback(true);
    mpcLocal.ilqrstart();
  });
  server.registerShutdownListener([&]() {mpcLocal.stop(); });
  server.serve(8070);
  server.blockWhileServing();
}
#endif