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

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(OpenSimParser, RAJAGOPAL_v3)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  EXPECT_TRUE(skel->getNumDofs() > 0);
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);
  verifyFeatherstoneJacobians(world);

  EXPECT_TRUE(file.markersMap.size() > 0);
  for (auto pair : file.markersMap)
  {
    EXPECT_TRUE(pair.second.first != nullptr);
  }

  /*
  // Uncomment this for local testing
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
    server.renderSkeleton(skel);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  server.blockWhileServing();
  */
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, RAJAGOPAL_v4)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  (void)skel;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);
  verifyFeatherstoneJacobians(world);

  EXPECT_TRUE(file.markersMap.size() > 0);
  for (auto pair : file.markersMap)
  {
    EXPECT_TRUE(pair.second.first != nullptr);
  }

  /*
  // Uncomment this for local testing
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
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, DELP_1990)
{
  OpenSimFile file
      = OpenSimParser::parseOsim("dart://sample/osim/NoArms_v3/Delp1990.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  (void)skel;
  EXPECT_TRUE(skel->getNumDofs() > 0);
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);
  verifyFeatherstoneJacobians(world);

  /*
  // Uncomment this for local testing
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
    server.renderSkeleton(skel);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  server.blockWhileServing();
  */
}
#endif
