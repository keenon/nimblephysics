#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/MJCFExporter.hpp"
#include "dart/utils/sdf/SdfParser.hpp"
#include "dart/utils/urdf/DartLoader.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

#define ALL_TESTS

/*
// This leads to a hard exit
TEST(OpenSimParser, UNSUPPORTED_JOINT_TYPE)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Bugs/79597a1/unscaled_generic.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  EXPECT_TRUE(skel->getNumDofs() > 0);
}
*/

#ifdef ALL_TESTS
TEST(OpenSimParser, CONVERT_TO_SDF)
{
  auto file = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest/results/Models/autoscaled.osim");

  std::map<std::string, std::string> mergeBodiesInto;
  mergeBodiesInto["ulna_r"] = "radius_r";
  mergeBodiesInto["ulna_l"] = "radius_l";
  OpenSimParser::convertOsimToSDF(
      "dart://sample/osim/MichaelTest/results/Models/autoscaled.osim",
      "../../../data/osim/MichaelTest/results/Models/model.sdf",
      mergeBodiesInto);

  std::shared_ptr<dynamics::Skeleton> skel = SdfParser::readSkeleton(
      "dart://sample/osim/MichaelTest/results/Models/model.sdf");

  /*
  // skel->getDof("walker_knee_r")->setPosition(1.0);
  // skel->getDof("walker_knee_l")->setPosition(0.5);
  GUIWebsocketServer server;
  server.renderSkeleton(skel);
  server.renderSkeleton(file.skeleton, "osim");
  server.serve(8070);
  server.blockWhileServing();
  */

  /*
  SkeletonConverter converter(skel, file.skeleton);
  // Set the root orientation to be the same
  for (int i = 0; i < 6; i++)
  {
    skel->setPosition(i, file.skeleton->getPosition(i));
  }
  // Link joints
  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    auto* sourceJoint = skel->getJoint(i);
    if (file.skeleton->getJoint(sourceJoint->getName()) != nullptr)
    {
      converter.linkJoints(
          sourceJoint, file.skeleton->getJoint(sourceJoint->getName()));
    }
  }
  converter.createVirtualMarkers();

  auto mot = OpenSimParser::loadMot(
      file.skeleton,
      "dart://sample/osim/MichaelTest/results/IK/S02DN101_ik.mot");
  Eigen::MatrixXs convertedPoses = converter.convertMotion(mot.poses);
  OpenSimParser::saveMot(
      skel,
      "../../../data/osim/MichaelTest/results/S02DN101.mot",
      mot.timestamps,
      convertedPoses);

  GUIWebsocketServer server;
  server.renderSkeleton(skel);

  // Render the converted poses over time
  int timestep = 0;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(1.0 / 50);
  ticker->registerTickListener([&](long) {
    skel->setPositions(convertedPoses.col(timestep));
    server.renderSkeleton(skel);
    timestep++;
    if (timestep >= convertedPoses.cols())
      timestep = 0;
  });
  server.registerConnectionListener([&]() { ticker->start(); });

  server.serve(8070);
  server.blockWhileServing();
  */
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, CONVERT_TO_SDF_2)
{
  auto file = OpenSimParser::parseOsim(
      "dart://sample/osim/Bugs/tmp187o1np6/unscaled_generic.osim");

  std::map<std::string, std::string> mergeBodiesInto;
  mergeBodiesInto["ulna_r"] = "radius_r";
  mergeBodiesInto["ulna_l"] = "radius_l";
  OpenSimParser::convertOsimToSDF(
      "dart://sample/osim/Bugs/tmp187o1np6/unscaled_generic.osim",
      "../../../data/osim/Bugs/tmp187o1np6/model.sdf",
      mergeBodiesInto);

  std::shared_ptr<dynamics::Skeleton> skel = SdfParser::readSkeleton(
      "dart://sample/osim/Bugs/tmp187o1np6/model.sdf");

  /*
  // skel->getDof("walker_knee_r")->setPosition(1.0);
  // skel->getDof("walker_knee_l")->setPosition(0.5);
  GUIWebsocketServer server;
  server.renderSkeleton(skel);
  server.renderSkeleton(file.skeleton, "osim");
  server.serve(8070);
  server.blockWhileServing();
  */

  /*
  SkeletonConverter converter(skel, file.skeleton);
  // Set the root orientation to be the same
  for (int i = 0; i < 6; i++)
  {
    skel->setPosition(i, file.skeleton->getPosition(i));
  }
  // Link joints
  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    auto* sourceJoint = skel->getJoint(i);
    if (file.skeleton->getJoint(sourceJoint->getName()) != nullptr)
    {
      converter.linkJoints(
          sourceJoint, file.skeleton->getJoint(sourceJoint->getName()));
    }
  }
  converter.createVirtualMarkers();

  auto mot = OpenSimParser::loadMot(
      file.skeleton,
      "dart://sample/osim/MichaelTest/results/IK/S02DN101_ik.mot");
  Eigen::MatrixXs convertedPoses = converter.convertMotion(mot.poses);
  OpenSimParser::saveMot(
      skel,
      "../../../data/osim/MichaelTest/results/S02DN101.mot",
      mot.timestamps,
      convertedPoses);

  GUIWebsocketServer server;
  server.renderSkeleton(skel);

  // Render the converted poses over time
  int timestep = 0;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(1.0 / 50);
  ticker->registerTickListener([&](long) {
    skel->setPositions(convertedPoses.col(timestep));
    server.renderSkeleton(skel);
    timestep++;
    if (timestep >= convertedPoses.cols())
      timestep = 0;
  });
  server.registerConnectionListener([&]() { ticker->start(); });

  server.serve(8070);
  server.blockWhileServing();
  */
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, CONVERT_TO_MJCF)
{
  auto file = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest4/Models/"
      "optimized_scale_and_markers.osim");

  std::map<std::string, std::string> mergeBodiesInto;
  mergeBodiesInto["ulna_r"] = "radius_r";
  mergeBodiesInto["ulna_l"] = "radius_l";
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton->simplifySkeleton(
      file.skeleton->getName(), mergeBodiesInto);

  MJCFExporter::writeSkeleton(
      "../../../data/osim/MichaelTest4/Models/model.mjcf", skel);
  /*
  OpenSimParser::convertOsimToMJCF(
      "dart://sample/osim/MichaelTest/results/Models/autoscaled.osim",
      "../../../data/osim/MichaelTest/results/Models/model.mjcf",
      mergeBodiesInto);
  */

  SkeletonConverter converter(skel, file.skeleton);
  // Set the root orientation to be the same
  for (int i = 0; i < 6; i++)
  {
    skel->setPosition(i, file.skeleton->getPosition(i));
  }
  // Link joints
  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    auto* sourceJoint = skel->getJoint(i);
    if (file.skeleton->getJoint(sourceJoint->getName()) != nullptr)
    {
      converter.linkJoints(
          sourceJoint, file.skeleton->getJoint(sourceJoint->getName()));
    }
  }
  converter.createVirtualMarkers();

  auto mot = OpenSimParser::loadMot(
      file.skeleton, "dart://sample/osim/MichaelTest4/IK/S02DN101_ik.mot");
  Eigen::MatrixXs convertedPoses = converter.convertMotion(mot.poses);
  OpenSimParser::saveMot(
      skel,
      "../../../data/osim/MichaelTest4/S02DN101.mot",
      mot.timestamps,
      convertedPoses);

  GUIWebsocketServer server;
  server.renderSkeleton(skel);

  // Render the converted poses over time
  int timestep = 900;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(1.0 / 50);
  ticker->registerTickListener([&](long) {
    skel->setPositions(convertedPoses.col(timestep));
    server.renderSkeleton(skel);
    timestep++;
    if (timestep >= 1300)
      timestep = 900;
  });
  server.registerConnectionListener([&]() { ticker->start(); });

  server.serve(8070);
  server.blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, CONVERT_TO_MJCF_2)
{
  auto file = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest5/Models/"
      "optimized_scale_and_markers.osim");
  dynamics::EulerFreeJoint* joint
      = static_cast<dynamics::EulerFreeJoint*>(file.skeleton->getJoint(0));
  std::cout << "Axis order: " << (int)joint->getAxisOrder() << std::endl;

  std::map<std::string, std::string> mergeBodiesInto;
  mergeBodiesInto["ulna_r"] = "radius_r";
  mergeBodiesInto["ulna_l"] = "radius_l";
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton->simplifySkeleton(
      file.skeleton->getName(), mergeBodiesInto);

  MJCFExporter::writeSkeleton(
      "../../../data/osim/MichaelTest5/Models/model.mjcf", skel);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, PIECEWISE_LINEAR)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Bugs/ee8cdcfd/unscaled_generic.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  EXPECT_TRUE(skel->getNumDofs() > 0);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, OLDER_FORMAT)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Bugs/tmpc9bpl7gs/unscaled_generic_raw.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  EXPECT_TRUE(skel->getNumDofs() > 0);
}
#endif

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
TEST(OpenSimParser, LOAD_GRF)
{
  std::vector<ForcePlate> grf = OpenSimParser::loadGRF(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_grf.mot",
      10);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, LOAD_NORMAL_GRF)
{
  std::vector<ForcePlate> forcePlates = OpenSimParser::loadGRF(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_grf.mot",
      10);

  EXPECT_EQ(forcePlates.size(), 2);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, LOAD_WEIRD_GRF)
{
  std::vector<ForcePlate> forcePlates = OpenSimParser::loadGRF(
      "dart://sample/osim/WeirdGRF/"
      "weird.mot",
      10);

  EXPECT_EQ(forcePlates.size(), 2);
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

#ifdef ALL_TESTS
TEST(OpenSimParser, SCALING)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");

  OpenSimParser::saveOsimScalingXMLFile(
      "Rajagopal2015",
      standard.skeleton,
      68.0,
      1.8,
      "Rajagopal2015.osim",
      "Rajagopal2015_markers.osim",
      "Rajagopal2015_rescaled.osim",
      "../../../data/osim/Rajagopal2015/ScalingInstructions.xml");
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, SAVE_TRC)
{
  auto c3d = C3DLoader::loadC3D("dart://sample/osim/Test_Output/JA1Gait35.c3d");
  OpenSimParser::saveTRC(
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Test_Output/JA1Gait35.trc",
      c3d.timestamps,
      c3d.markerTimesteps);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, SAVE_GRF)
{
  auto c3d = C3DLoader::loadC3D("dart://sample/osim/Test_Output/JA1Gait35.c3d");
  OpenSimParser::saveGRFMot(
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Test_Output/JA1Gait35_grf.mot",
      c3d.timestamps,
      c3d.forcePlates);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, MOVE_OUTPUT_MARKERS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::map<std::string, Eigen::Vector3s> bodyScales;
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>> markerOffsets;
  for (std::pair<
           const std::string,
           std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& pair :
       standard.markersMap)
  {
    markerOffsets[pair.first] = std::make_pair<std::string, Eigen::Vector3s>(
        std::string(pair.second.first->getName()),
        Eigen::Vector3s(pair.second.second));
  }

  OpenSimParser::moveOsimMarkers(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim",
      bodyScales,
      markerOffsets,
      "../../../data/osim/Rajagopal2015/Rajagopal2015_markersMoved.osim");

  OpenSimParser::filterJustMarkers(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015_markersMoved.osim",
      "../../../data/osim/Rajagopal2015/justMarkers.osim");
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, OVERWRITE_OUTPUT_MARKERS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>> markerOffsets;
  markerOffsets["TEST1"] = std::make_pair("femur_l", Eigen::Vector3s::Ones());
  markerOffsets["TEST2"] = std::make_pair("femur_r", Eigen::Vector3s::Zero());
  markerOffsets["TEST3"]
      = std::make_pair("pelvis", -1 * Eigen::Vector3s::Ones());
  std::map<std::string, bool> anatomical;
  anatomical["TEST1"] = true;
  anatomical["TEST2"] = false;
  // anatomical["TEST3"] = false;

  OpenSimParser::replaceOsimMarkers(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim",
      markerOffsets,
      anatomical,
      "../../../data/osim/Rajagopal2015/Rajagopal2015_markersReplaced.osim");

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015_markersReplaced.osim");
  EXPECT_EQ(file.markersMap.size(), 3);
  EXPECT_EQ(file.markersMap.count("TEST1"), 1);
  EXPECT_EQ(file.markersMap.count("TEST2"), 1);
  EXPECT_EQ(file.markersMap.count("TEST3"), 1);
  EXPECT_EQ(file.markersMap.count("FOOBAR"), 0);
  EXPECT_EQ(file.anatomicalMarkers.size(), 1);
  EXPECT_EQ(file.anatomicalMarkers[0], "TEST1");
  EXPECT_EQ(file.trackingMarkers.size(), 2);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, RATIONALIZE_CUSTOM_JOINTS)
{
  OpenSimParser::rationalizeJoints(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim",
      "../../../data/osim/ComplexKnee/"
      "gait2392_frontHingeKnee_dem_rational.osim");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim");
  OpenSimFile rational = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/"
      "gait2392_frontHingeKnee_dem_rational.osim");
  standard.skeleton->zeroTranslationInCustomFunctions();
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    EXPECT_TRUE(
        standard.skeleton->getBodyNode(i)->getWorldTransform().translation()
        == rational.skeleton->getBodyNode(i)
               ->getWorldTransform()
               .translation());
  }
  for (int i = 0; i < standard.skeleton->getNumJoints(); i++)
  {
    Eigen::Vector3s original
        = standard.skeleton->getJoint(i)->getRelativeTransform().translation();
    Eigen::Vector3s recovered
        = rational.skeleton->getJoint(i)->getRelativeTransform().translation();
    if ((original - recovered).norm() > 1e-14)
    {
      std::cout << "Doesn't match on joint "
                << standard.skeleton->getJoint(i)->getName() << std::endl;
      EXPECT_TRUE(original == recovered);
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, SAVE_MOT)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/LaiArnoldSubject6/"
      "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      standard.skeleton, "dart://sample/osim/LaiArnoldSubject6/walking1.mot");

  OpenSimParser::saveMot(
      standard.skeleton,
      "../../../data/osim/LaiArnoldSubject6/recovered.mot",
      mot.timestamps,
      mot.poses);
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, SAVE_IK_FILE)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/JA1GaitResults/rescaled.osim");

  std::vector<std::string> markerNames;
  for (auto& pair : standard.markersMap)
  {
    markerNames.push_back(pair.first);
  }

  OpenSimParser::saveOsimInverseKinematicsXMLFile(
      "JA1Gait35",
      markerNames,
      "rescaled.osim",
      "JA1Gait35.trc",
      "JA1Gait35_ik_by_opensim.mot",
      "../../../data/osim/JA1GaitResults/JA1Gait35_ik_setup.xml");
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, SAVE_ID_FILE)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/JA1GaitResults/rescaled.osim");
  auto c3d
      = C3DLoader::loadC3D("dart://sample/osim/JA1GaitResults/original.c3d");
  auto mot = OpenSimParser::loadMot(
      standard.skeleton, "dart://sample/osim/JA1GaitResults/JA1Gait35_ik.mot");

  OpenSimParser::saveOsimInverseDynamicsForcesXMLFile(
      "JA1Gait35",
      standard.skeleton,
      mot.poses,
      c3d.forcePlates,
      "JA1Gait35_grf.mot",
      "../../../data/osim/JA1GaitResults/JA1Gait35_external_forces.xml");

  OpenSimParser::saveOsimInverseDynamicsXMLFile(
      "JA1Gait35",
      "rescaled.osim",
      "JA1Gait35_ik.mot",
      "JA1Gait35_external_forces.xml",
      "JA1Gait35_id_output.sto",
      "JA1Gait35_id_body_forces.sto",
      "../../../data/osim/JA1GaitResults/JA1Gait35_id_setup.xml");
}
#endif

#ifdef ALL_TESTS
TEST(OpenSimParser, COMPLEX_KNEE)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  EXPECT_TRUE(skel->getNumDofs() > 0);
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(skel);

  EXPECT_TRUE(file.markersMap.size() > 0);
  for (auto pair : file.markersMap)
  {
    EXPECT_TRUE(pair.second.first != nullptr);
  }

  // Uncomment this for local testing
  /*
  GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);

  Ticker ticker = Ticker(0.01);
  ticker.registerTickListener([&](long now) {
    double progress = (now % 2000) / 2000.0;
    // Also test: subtalar, intercond (toes)
    std::string jointName = "ankle_angle_l";
    double jointUpperLimit = skel->getDof(jointName)->getPositionUpperLimit();
    double jointLowerLimit = skel->getDof(jointName)->getPositionLowerLimit();

    double jointPos
        = progress * (jointUpperLimit - jointLowerLimit) + jointLowerLimit;
    skel->getDof(jointName)->setPosition(jointPos);

    server.renderSkeleton(skel);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  server.blockWhileServing();

  verifyFeatherstoneJacobians(world);
  */
}
#endif