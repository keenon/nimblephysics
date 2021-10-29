#include <gtest/gtest.h>

#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

void debugGRFToGUI(
    std::shared_ptr<dynamics::Skeleton>& skel,
    Eigen::MatrixXs poses,
    OpenSimGRF& grf)
{
  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);
  server.renderBasis();
  server.setAutoflush(false);

  int timestep = 0;
  Ticker ticker(1.0 / 50);
  ticker.registerTickListener([&](long) {
    skel->setPositions(poses.col(timestep));
    server.renderSkeleton(skel);

    server.deleteObjectsByPrefix("grf_");
    for (int i = 0; i < grf.plateCOPs.size(); i++)
    {
      // Eigen::Isometry3s T = skel->getRootBodyNode()->getWorldTransform();

      Eigen::Vector3s cop = grf.plateCOPs[i].col(timestep) * 0.2;
      Eigen::Vector6s wrench = grf.plateGRFs[i].col(timestep) * 0.001;

      std::vector<Eigen::Vector3s> points;
      points.push_back(cop);
      points.push_back(cop + wrench.tail<3>());
      server.createLine("grf_" + i, points, Eigen::Vector3s::UnitX());
    }

    server.flush();

    timestep++;
    if (timestep >= poses.cols())
    {
      timestep = 0;
    }
  });
  server.registerConnectionListener([&]() { ticker.start(); });
  server.blockWhileServing();
}

// #ifdef ALL_TESTS
TEST(MarkerFitter, DYNAMICS_OFF_PRE_SCALED)
{
  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  OpenSimGRF grf = OpenSimParser::loadGRF(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_grf.mot",
      10);

  // Create a marker fitter

  MarkerFitter fitter(scaled.skeleton, scaled.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  MarkerInitialization init;
  init.poses = mot.poses;
  init.groupScales = scaled.skeleton->getGroupScales();

  fitter.initializeMasses(init);

  // Target markers
  // debugGRFToGUI(scaled.skeleton, init.poses, grf);
}
// #endif

// #ifdef FULL_EVAL
#ifdef ALL_TESTS
TEST(MarkerFitter, EVAL_PERFORMANCE)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  // Get the gold data scales in `config`
  OpenSimFile moddedBase = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(moddedBase.markersMap);
  standard.markersMap = convertedMarkers;
  OpenSimScaleAndMarkerOffsets config
      = OpenSimParser::getScaleAndMarkerOffsets(standard, scaled);
  EXPECT_TRUE(config.success);

  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;
  (void)poses;

  // Check our marker maps

  std::vector<
      std::pair<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>
      moddedMarkerOffsets;
  for (auto pair : standard.markersMap)
  {
    moddedMarkerOffsets.push_back(pair);
  }
  for (int i = 0; i < moddedMarkerOffsets.size(); i++)
  {
    for (int j = 0; j < moddedMarkerOffsets.size(); j++)
    {
      if (i == j)
        continue;
      // Just don't have duplicate markers
      if (moddedMarkerOffsets[i].second.first
              == moddedMarkerOffsets[j].second.first
          && moddedMarkerOffsets[i].second.second
                 == moddedMarkerOffsets[j].second.second)
      {
        std::cout << "Found duplicate markers, " << i << " and " << j << ": "
                  << moddedMarkerOffsets[i].first << " and "
                  << moddedMarkerOffsets[j].first << " on "
                  << moddedMarkerOffsets[i].second.first->getName()
                  << std::endl;
      }
    }
  }

  // Check that timestamps match up
  if (mot.timestamps.size() != markerTrajectories.timestamps.size())
  {
    std::cout << "Got a different number of timestamps. Mot: "
              << mot.timestamps.size()
              << " != Trc: " << markerTrajectories.timestamps.size()
              << std::endl;
    EXPECT_EQ(mot.timestamps.size(), markerTrajectories.timestamps.size());
    std::cout << "First 10 timesteps:" << std::endl;
    for (int k = 0; k < 10; k++)
    {
      std::cout << k << ": Mot=" << mot.timestamps[k]
                << " != Trc=" << markerTrajectories.timestamps[k] << std::endl;
    }
  }
  else
  {
    for (int i = 0; i < mot.timestamps.size(); i++)
    {
      if (abs(mot.timestamps[i] - markerTrajectories.timestamps[i]) > 1e-10)
      {
        std::cout << "Different timestamps at step " << i
                  << ": Mot: " << mot.timestamps[i]
                  << " != Trc: " << markerTrajectories.timestamps[i]
                  << std::endl;
        EXPECT_NEAR(mot.timestamps[i], markerTrajectories.timestamps[i], 1e-10);
        break;
      }
    }
  }

  /*
  scaled.skeleton->setPositions(poses.col(0));
  std::cout << scaled.skeleton->getJoint(0)->getRelativeTransform().matrix()
            << std::endl;

  std::shared_ptr<dynamics::Skeleton> clone = scaled.skeleton->clone();
  clone->setPositions(poses.col(0));
  std::cout << clone->getJoint(0)->getRelativeTransform().matrix() << std::endl;

  // Target markers
  debugTrajectoryAndMarkersToGUI(
      scaled.skeleton,
      scaled.markersMap,
      poses,
      markerTrajectories.markerTimesteps);
  */

  int timestep = 1264; // 103, 1851

  std::map<std::string, Eigen::Vector3s> goldMarkers
      = markerTrajectories.markerTimesteps[timestep];
  Eigen::VectorXs goldPose = poses.col(timestep);

  /*
  // Try to convert the goldPose to the standard skeleton

  Eigen::VectorXs standardGoldPose
      = Eigen::VectorXs::Zero(standard.skeleton->getNumDofs());
  for (int i = 0; i < standard.skeleton->getNumDofs(); i++)
  {
    standardGoldPose(i) = goldPose(
        scaled.skeleton->getDof(standard.skeleton->getDof(i)->getName())
            ->getIndexInSkeleton());
  }
  std::cout << "Eigen::VectorXs standardGoldPose = Eigen::VectorXs::Zero("
            << standardGoldPose.size() << ");" << std::endl;
  std::cout << "standardGoldPose << ";
  for (int i = 0; i < standardGoldPose.size(); i++)
  {
    if (i > 0)
    {
      std::cout << ", ";
    }
    std::cout << standardGoldPose(i);
  }
  std::cout << ";" << std::endl;

  Eigen::VectorXs standardBodyScales = config.bodyScales;
  std::cout << "Eigen::VectorXs standardBodyScales = Eigen::VectorXs::Zero("
            << standardBodyScales.size() << ");" << std::endl;
  std::cout << "standardBodyScales << ";
  for (int i = 0; i < standardBodyScales.size(); i++)
  {
    if (i > 0)
    {
      std::cout << ", ";
    }
    std::cout << standardBodyScales(i);
  }
  std::cout << ";" << std::endl;
  */

  // Try to fit the skeleton to

  /*
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  Eigen::VectorXs targetPoses = Eigen::VectorXs::Zero(goldMarkers.size() * 3);
  for (auto pair : goldMarkers)
  {
    std::cout << "Marker: " << pair.first << std::endl;
    targetPoses.segment<3>(markers.size() * 3) = pair.second;
    markers.push_back(standard.markersMap[pair.first]);
  }
  Eigen::VectorXs markerWeights = Eigen::VectorXs::Ones(markers.size());
  debugFitToGUI(
      standard.skeleton, markers, targetPoses, scaled.skeleton, goldPose);
  */

  // Get a random subset of the data

  srand(25);
  /*
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations
      = MarkerFitter::pickSubset(markerTrajectories.markerTimesteps, 40);
  */

  /*
  std::vector<unsigned int> indices(markerTrajectories.markerTimesteps.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  for (int i = 0; i < 2; i++)
  {
    std::cout << "Shuffled " << i << "->" << indices[i] << std::endl;
    markerObservations.push_back(
        markerTrajectories.markerTimesteps[indices[i]]);
    for (auto pair : markerTrajectories.markerTimesteps[indices[i]])
    {
      std::cout << pair.first << ": " << pair.second << std::endl;
    }
  }
  */

  std::cout << "Original skel pos: " << standard.skeleton->getPositions()
            << std::endl;

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  markerObservations.push_back(markerTrajectories.markerTimesteps[0]);
  for (auto pair : markerObservations[0])
  {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }

  std::cout << "Marker map:" << std::endl;
  for (auto pair : standard.markersMap)
  {
    std::cout << pair.first << ": (" << pair.second.first->getName() << ", "
              << pair.second.second << ")" << std::endl;
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::shared_ptr<BilevelFitResult> result
      = fitter.optimize(markerObservations);
  standard.skeleton->setGroupScales(result->groupScales);
  Eigen::VectorXs bodyScales = standard.skeleton->getBodyScales();

  std::cout << "Result scales: " << bodyScales << std::endl;

  Eigen::VectorXs groupScaleError = bodyScales - config.bodyScales;
  Eigen::MatrixXs groupScaleCols = Eigen::MatrixXs(groupScaleError.size(), 4);
  groupScaleCols.col(0) = config.bodyScales;
  groupScaleCols.col(1) = bodyScales;
  groupScaleCols.col(2) = groupScaleError;
  groupScaleCols.col(3) = groupScaleError.cwiseQuotient(config.bodyScales);
  std::cout << "gold scales - result scales - error - error %" << std::endl
            << groupScaleCols << std::endl;
}
#endif
// #endif