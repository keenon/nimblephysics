#include <gtest/gtest.h>

#include "dart/biomechanics/IKErrorReport.hpp"
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

// #ifdef FULL_EVAL
// #ifdef ALL_TESTS
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
      = standard.skeleton->convertMarkerMap(scaled.markersMap);
  standard.markersMap = convertedMarkers;
  OpenSimScaleAndMarkerOffsets config
      = OpenSimParser::getScaleAndMarkerOffsets(standard, scaled);
  EXPECT_TRUE(config.success);

  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;

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

  std::vector<std::string> activeMarkers;
  for (auto pair : scaled.markersMap)
  {
    std::string name = pair.first;
    if (name.find("_") == std::string::npos
        && std::all_of(name.begin(), name.end(), [](unsigned char c) {
             return std::isupper(c);
           }))
    {
      activeMarkers.push_back(name);
    }
  }

  biomechanics::IKErrorReport report = biomechanics::IKErrorReport(
      scaled.skeleton,
      scaled.markersMap,
      mot.poses,
      markerTrajectories.markerTimesteps);
  report.printReport(10);
}
// #endif
// #endif