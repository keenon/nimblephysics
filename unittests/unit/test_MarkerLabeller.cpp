#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerLabeller.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace server;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(LABELLER, RECREATE_LABELS)
{
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;

  std::vector<std::map<std::string, Eigen::Vector3s>> markersOverTime;
  std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime;

  for (int i = 0; i < 250; i++)
  {
    scaled.skeleton->setPositions(poses.col(i));
    markersOverTime.push_back(
        scaled.skeleton->getMarkerMapWorldPositions(scaled.markersMap));
    // TODO: add dropout to markers
    jointsOverTime.push_back(scaled.skeleton->getJointWorldPositionsMap());
    // TODO: add noise to joints
  }

  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  labeller.setSkeleton(scaled.skeleton);
  for (int i = 0; i < scaled.skeleton->getNumJoints(); i++)
  {
    std::string jointName = scaled.skeleton->getJoint(i)->getName();
    labeller.matchUpJointToSkeletonJoint(jointName, jointName);
  }
  labeller.setMockJointLocations(jointsOverTime);

  std::map<std::string, std::pair<std::string, Eigen::Vector3s>>
      markerStringMap;
  for (auto& pair : scaled.markersMap)
  {
    markerStringMap[pair.first] = std::pair<std::string, Eigen::Vector3s>(
        pair.second.first->getName(), pair.second.second);
  }

  labeller.evaluate(markerStringMap, markersOverTime);
}
#endif