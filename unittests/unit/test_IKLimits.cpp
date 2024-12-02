#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

// #ifdef ALL_TESTS
TEST(IKLimits, LIMIT_TEST)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/IncompleteIK/Models/"
      "optimized_scale_and_markers.osim");

  OpenSimMot mot = OpenSimParser::loadMot(
      standard.skeleton,
      "dart://sample/osim/IncompleteIK/IK/markers_smpl_ik.mot");
  OpenSimTRC trc = OpenSimParser::loadTRC(
      "dart://sample/osim/IncompleteIK/MarkerData/markers_smpl.trc");

  // Worst timestep = 387

  for (int i = 0; i < mot.poses.cols(); i++)
  {
    standard.skeleton->setPositions(mot.poses.col(i));
    auto modelMarkers
        = standard.skeleton->getMarkerMapWorldPositions(standard.markersMap);
    auto observedMarkers = trc.markerTimesteps[i];
    s_t rmse = 0.0;
    for (auto& pair : modelMarkers)
    {
      if (observedMarkers.count(pair.first))
      {
        rmse += (pair.second - observedMarkers.at(pair.first)).squaredNorm();
      }
    }
    std::cout << "Step " << i << ": " << rmse << std::endl;
  }

  int worstTimestep = 387;
  standard.skeleton->setPositions(mot.poses.col(worstTimestep));

  auto ballJoints = standard.skeleton->convertSkeletonToBallJoints();
  ballJoints->setPositions(standard.skeleton->convertPositionsToBallSpace(
      standard.skeleton->getPositions()));
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> linearMarkers;
  Eigen::VectorXs linearTargetPositions
      = Eigen::VectorXs::Zero(trc.markerTimesteps[worstTimestep].size() * 3);
  int cursor = 0;
  for (auto& pair : trc.markerTimesteps[worstTimestep])
  {
    auto& marker = standard.markersMap.at(pair.first);
    linearMarkers.emplace_back(
        ballJoints->getBodyNode(marker.first->getName()), marker.second);
    linearTargetPositions.segment<3>(cursor) = pair.second;
    cursor += 3;
  }
  Eigen::VectorXs markerWeights
      = Eigen::VectorXs::Ones(trc.markerTimesteps[worstTimestep].size());

  ballJoints->fitMarkersToWorldPositions(
      linearMarkers,
      linearTargetPositions,
      markerWeights,
      false,
      math::IKConfig()
          .setConvergenceThreshold(1e-9)
          .setMaxStepCount(2000)
          .setMaxRestarts(1)
          .setLogOutput(true));
  standard.skeleton->setPositions(
      standard.skeleton->convertPositionsFromBallSpace(
          ballJoints->getPositions()));

  auto modelMarkers
      = standard.skeleton->getMarkerMapWorldPositions(standard.markersMap);
  auto observedMarkers = trc.markerTimesteps[worstTimestep];
  s_t rmse = 0.0;
  for (auto& pair : modelMarkers)
  {
    if (observedMarkers.count(pair.first))
    {
      rmse += (pair.second - observedMarkers.at(pair.first)).squaredNorm();
    }
  }
  std::cout << "Adjusted RMSE: " << rmse << std::endl;
}
// #endif