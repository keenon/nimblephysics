#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

TEST(OpenSimParser, RAJAGOPAL)
{
  std::shared_ptr<dynamics::Skeleton> skel = OpenSimParser::readSkeleton(
      "dart://sample/osim/FullBodyModel-4.0/Rajagopal2015.osim");
  (void)skel;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);
  // verifyFeatherstoneJacobians(world);

  // Uncomment this for local testing
  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);

  Ticker ticker = Ticker(0.01);
  ticker.registerTickListener([&](long now) {
    double progress = (now % 2000) / 2000.0;
    skel->getDof("knee_angle_r")
        ->setPosition(
            progress * skel->getDof("knee_angle_r")->getPositionUpperLimit());
    skel->getDof("knee_angle_l")
        ->setPosition(
            progress * skel->getDof("knee_angle_l")->getPositionUpperLimit());
    // skel->getDof("knee_angle_r_beta")->setPosition(progress);
    // skel->getDof("knee_angle_l_beta")->setPosition(progress);
    server.renderSkeleton(skel);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  server.blockWhileServing();
  */
}