#include <gtest/gtest.h>

#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

bool testFitterGradients(
    MarkerFitter& fitter,
    std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& observedMarkers)
{
  const s_t THRESHOLD = 5e-8;

  Eigen::VectorXs gradWrtJoints = fitter.getLossGradientWrtJoints(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtJoints_fd
      = fitter.finiteDifferenceLossGradientWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(gradWrtJoints, gradWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt joints" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJoints << std::endl
              << "FD:" << std::endl
              << gradWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJoints - gradWrtJoints_fd << std::endl;
    return false;
  }

  Eigen::VectorXs gradWrtScales = fitter.getLossGradientWrtGroupScales(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtScales_fd
      = fitter.finiteDifferenceLossGradientWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(gradWrtScales, gradWrtScales_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtScales << std::endl
              << "FD:" << std::endl
              << gradWrtScales_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtScales - gradWrtScales_fd << std::endl;
    return false;
  }

  Eigen::VectorXs gradWrtMarkerOffsets = fitter.getLossGradientWrtMarkerOffsets(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtMarkerOffsets_fd
      = fitter.finiteDifferenceLossGradientWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(gradWrtMarkerOffsets, gradWrtMarkerOffsets_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt marker offsets" << std::endl
              << "Analytical:" << std::endl
              << gradWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << gradWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtMarkerOffsets - gradWrtMarkerOffsets_fd << std::endl;
    return false;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Second order Jacobians (and their input components)
  /////////////////////////////////////////////////////////////////////////////

  std::vector<int> sparsityMap
      = fitter.getSparsityMap(markers, observedMarkers);

  Eigen::MatrixXs markerErrorJacWrtJoints
      = fitter.getMarkerErrorJacobianWrtJoints(skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtJoints_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(markerErrorJacWrtJoints, markerErrorJacWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt joints" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtJoints << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtJoints - markerErrorJacWrtJoints_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtJoints
      = fitter.getLossGradientWrtJointsJacobianWrtJoints(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtJoints_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtJoints, gradWrtJointsJacWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt joints" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtJoints << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtJoints - gradWrtJointsJacWrtJoints_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs markerErrorJacWrtGroupScales
      = fitter.getMarkerErrorJacobianWrtGroupScales(skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtGroupScales_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(
          markerErrorJacWrtGroupScales,
          markerErrorJacWrtGroupScales_fd,
          THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtGroupScales << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtGroupScales_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtGroupScales - markerErrorJacWrtGroupScales_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtGroupScales
      = fitter.getLossGradientWrtJointsJacobianWrtGroupScales(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtGroupScales_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtGroupScales,
          gradWrtJointsJacWrtGroupScales_fd,
          THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtGroupScales << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtGroupScales_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtGroupScales
                     - gradWrtJointsJacWrtGroupScales_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs markerErrorJacWrtMarkerOffsets
      = fitter.getMarkerErrorJacobianWrtMarkerOffsets(
          skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtMarkerOffsets_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(
          markerErrorJacWrtMarkerOffsets,
          markerErrorJacWrtMarkerOffsets_fd,
          THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt marker offsets" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtMarkerOffsets
                     - markerErrorJacWrtMarkerOffsets_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtMarkerOffsets
      = fitter.getLossGradientWrtJointsJacobianWrtMarkerOffsets(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtMarkerOffsets_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtMarkerOffsets,
          gradWrtJointsJacWrtMarkerOffsets_fd,
          THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets
                     - gradWrtJointsJacWrtMarkerOffsets_fd
              << std::endl;
    return false;
  }

  return true;
}

#ifdef ALL_TESTS
TEST(MarkerFitter, DERIVATIVES)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(
      std::make_pair(osim->getBodyNode("radius_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("radius_r"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("tibia_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("tibia_r"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("ulna_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("ulna_r"), Eigen::Vector3s::Random()));

  MarkerFitter fitter(osim, markers);

  std::vector<std::pair<int, Eigen::Vector3s>> observedMarkers;
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(0, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(1, Eigen::Vector3s::Random()));
  // Skip 2
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(3, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(4, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(5, Eigen::Vector3s::Random()));

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, DERIVATIVES_BALL_JOINTS)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);
  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("radius_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("radius_r"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("tibia_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("tibia_r"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("ulna_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("ulna_r"), Eigen::Vector3s::Random()));

  MarkerFitter fitter(osimBallJoints, markers);

  std::vector<std::pair<int, Eigen::Vector3s>> observedMarkers;
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(0, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(1, Eigen::Vector3s::Random()));
  // Skip 2
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(3, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(4, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(5, Eigen::Vector3s::Random()));

  EXPECT_TRUE(
      testFitterGradients(fitter, osimBallJoints, markers, observedMarkers));
}
#endif