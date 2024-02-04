#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/StreamingMocapLab.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;

// #ifdef ALL_TESTS
TEST(StreamingMocapLab, LIMIT_TEST)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/IncompleteIK/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricPrefixes();
  standard.skeleton->autogroupSymmetricSuffixes();

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

  int numWindows = 5;
  int stride = 10;
  int maxMarkersPerTimestep = 50;

  StreamingMocapLab lab(
      standard.skeleton, markers, numWindows, stride, maxMarkersPerTimestep);
  lab.setAnthropometricPrior(anthro);
  lab.startSolverThread();

  std::shared_ptr<server::GUIWebsocketServer> gui
      = std::make_shared<server::GUIWebsocketServer>();
  gui->serve(8070);
  lab.startGUIThread(gui);

  long time = 10;
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
    lab.manuallyObserveMarkers(frameMarkers, time);
    auto featurePair = lab.getTraceFeatures();
    time += 10;
    Eigen::MatrixXs logits = Eigen::MatrixXs::Zero(
        standard.skeleton->getNumBodyNodes() + markers.size() + 1,
        classes.size());
    Eigen::VectorXi traceIDs = featurePair.second;
    // std::cout << "Trace IDs: " << traceIDs.transpose() << std::endl;
    for (int j = 0; j < classes.size(); j++)
    {
      logits(classes[j], j) = 1.0;
    }
    lab.observeTraceLogits(logits, traceIDs);
    // Sleep this thread for 1ms to simulate real-time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // }
}
// #endif