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
TEST(OpenSimParser, LOAD_TRC)
{
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");
  EXPECT_TRUE(markerTrajectories.markerTimesteps.size() > 0);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, RAJAGOPAL_GET_CONFIGURATION)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  OpenSimFile modded = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");

  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(modded.markersMap);
  standard.markersMap = convertedMarkers;

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimScaleAndMarkerOffsets config
      = OpenSimParser::getScaleAndMarkerOffsets(standard, scaled);
  EXPECT_TRUE(config.success);

  standard.skeleton->setBodyScales(config.bodyScales);
  std::map<std::string, Eigen::Vector3s> markerWorldPositions
      = standard.skeleton->getMarkerMapWorldPositions(config.markers);
  std::map<std::string, Eigen::Vector3s> scaledWorldPositions
      = scaled.skeleton->getMarkerMapWorldPositions(scaled.markersMap);
  for (auto pair : markerWorldPositions)
  {
    Eigen::Vector3s absoluteMarker = markerWorldPositions[pair.first];
    Eigen::Vector3s scaledMarker = scaledWorldPositions[pair.first];
    if (!equals(absoluteMarker, scaledMarker, 1e-10))
    {
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 3);
      compare.col(0) = absoluteMarker;
      compare.col(1) = scaledMarker;
      compare.col(2) = scaledMarker - absoluteMarker;
      std::cout << "Got an error on marker \"" << pair.first
                << "\":" << std::endl
                << "absolute world - scaled world - diff world" << std::endl
                << compare << std::endl;
      compare = Eigen::MatrixXs::Zero(3, 5);
      compare.col(0) = config.markers[pair.first].second;
      compare.col(1) = config.markers[pair.first].first->getScale();
      compare.col(2) = config.markers[pair.first].second.cwiseProduct(
          config.markers[pair.first].first->getScale());
      compare.col(3) = scaled.markersMap[pair.first].second;
      compare.col(4) = config.markers[pair.first].second.cwiseProduct(
                           config.markers[pair.first].first->getScale())
                       - scaled.markersMap[pair.first].second;
      std::cout << "config local - body scale - config local * body scale - "
                   "scaled local - diff"
                << std::endl
                << compare << std::endl;
      EXPECT_TRUE(equals(absoluteMarker, scaledMarker, 1e-10));
    }
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