#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/StreamingIK.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;

// #ifdef ALL_TESTS
TEST(StreamingIK, LIMIT_TEST)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/IncompleteIK/Models/"
      "optimized_scale_and_markers.osim");

  OpenSimTRC trc = OpenSimParser::loadTRC(
      "dart://sample/osim/IncompleteIK/MarkerData/markers_smpl.trc");

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  std::map<std::string, int> markerNameToIndex;
  for (auto& pair : standard.markersMap)
  {
    markerNameToIndex[pair.first]
        = standard.skeleton->getNumBodyNodes() + markers.size();
    markers.push_back(pair.second);
  }

  std::shared_ptr<Anthropometrics> anthro = Anthropometrics::loadFromFile(
      "dart://sample/osim/ANSUR/ANSUR_metrics.xml");

  StreamingIK ik(standard.skeleton, markers);
  ik.setAnthropometricPrior(anthro);
  ik.startSolverThread();

  std::shared_ptr<server::GUIWebsocketServer> gui
      = std::make_shared<server::GUIWebsocketServer>();
  gui->serve(8070);
  ik.startGUIThread(gui);

  // while (true)
  // {
  for (int i = 0; i < trc.markerTimesteps.size(); i++)
  {
    std::vector<Eigen::Vector3s> frameMarkers;
    std::vector<int> classes;
    for (auto& marker : trc.markerTimesteps[i])
    {
      frameMarkers.push_back(marker.second);
      classes.push_back(markerNameToIndex[marker.first]);
    }
    std::vector<Eigen::Vector9s> copTorqueForces;
    ik.observeMarkers(frameMarkers, classes, i, copTorqueForces);
    // Sleep this thread for 1ms to simulate real-time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // }
}
// #endif