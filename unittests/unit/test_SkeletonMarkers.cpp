#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
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

#ifdef ALL_TESTS
TEST(SkeletonConverter, IK_JACOBIANS)
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

  osim->mergeScaleGroups(
      osim->getBodyNode("radius_l"), osim->getBodyNode("radius_r"));
  osim->setScaleGroupUniformScaling(osim->getBodyNode("tibia_r"));

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
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

  EXPECT_TRUE(verifySkeletonMarkerJacobians(osim, markers));

  // Try some non-zero joint configurations
  for (int i = 0; i < 3; i++)
  {
    osim->setPositions(Eigen::VectorXs::Random(osim->getNumDofs()));
    EXPECT_TRUE(verifySkeletonMarkerJacobians(osim, markers));
  }
}
#endif

#ifdef ALL_TESTS
TEST(SkeletonConverter, IK_JACOBIANS_BALL_JOINTS)
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

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
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

  EXPECT_TRUE(verifySkeletonMarkerJacobians(osimBallJoints, markers));

  // Try some non-zero joint configurations
  for (int i = 0; i < 3; i++)
  {
    osimBallJoints->setPositions(
        Eigen::VectorXs::Random(osimBallJoints->getNumDofs()));
    EXPECT_TRUE(verifySkeletonMarkerJacobians(osimBallJoints, markers));
  }
}
#endif

#ifdef ALL_TESTS
TEST(SkeletonConverter, MARKER_DISTANCE)
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

  osim->mergeScaleGroups(
      osim->getBodyNode("radius_l"), osim->getBodyNode("radius_r"));
  osim->setScaleGroupUniformScaling(osim->getBodyNode("tibia_r"));

  std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA = std::make_pair(
      osim->getBodyNode("radius_l"), Eigen::Vector3s::Random());
  std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB
      = std::make_pair(osim->getBodyNode("tibia_l"), Eigen::Vector3s::Random());

  s_t dist = osim->getDistanceInWorldSpace(markerA, markerB);
  EXPECT_TRUE(dist > 0);

  Eigen::VectorXs bodyGrad
      = osim->getGradientOfDistanceWrtBodyScales(markerA, markerB);
  Eigen::VectorXs bodyGrad_fd
      = osim->finiteDifferenceGradientOfDistanceWrtBodyScales(markerA, markerB);

  if (!equals(bodyGrad, bodyGrad_fd, 1e-10))
  {
    std::cout << "Grad wrt body scales disagrees!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(bodyGrad.size(), 3);
    compare.col(0) = bodyGrad;
    compare.col(1) = bodyGrad_fd;
    compare.col(2) = bodyGrad - bodyGrad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(bodyGrad, bodyGrad_fd, 1e-10));
  }

  Eigen::VectorXs groupGrad
      = osim->getGradientOfDistanceWrtGroupScales(markerA, markerB);
  Eigen::VectorXs groupGrad_fd
      = osim->finiteDifferenceGradientOfDistanceWrtGroupScales(
          markerA, markerB);

  if (!equals(groupGrad, groupGrad_fd, 1e-10))
  {
    std::cout << "Grad wrt group scales disagrees!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(groupGrad.size(), 3);
    compare.col(0) = groupGrad;
    compare.col(1) = groupGrad_fd;
    compare.col(2) = groupGrad - groupGrad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(groupGrad, groupGrad_fd, 1e-10));
  }

  Eigen::Vector3s axis = Eigen::Vector3s::Random();
  Eigen::VectorXs axisBodyGrad
      = osim->getGradientOfDistanceAlongAxisWrtBodyScales(
          markerA, markerB, axis);
  Eigen::VectorXs axisBodyGrad_fd
      = osim->finiteDifferenceGradientOfDistanceAlongAxisWrtBodyScales(
          markerA, markerB, axis);

  if (!equals(axisBodyGrad, axisBodyGrad_fd, 1e-10))
  {
    std::cout << "Grad of axis distance wrt body scales disagrees!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(axisBodyGrad.size(), 3);
    compare.col(0) = axisBodyGrad;
    compare.col(1) = axisBodyGrad_fd;
    compare.col(2) = axisBodyGrad - axisBodyGrad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(axisBodyGrad, axisBodyGrad_fd, 1e-10));
  }

  Eigen::VectorXs axisGroupGrad
      = osim->getGradientOfDistanceAlongAxisWrtGroupScales(
          markerA, markerB, axis);
  Eigen::VectorXs axisGroupGrad_fd
      = osim->finiteDifferenceGradientOfDistanceAlongAxisWrtGroupScales(
          markerA, markerB, axis);

  if (!equals(axisGroupGrad, axisGroupGrad_fd, 1e-10))
  {
    std::cout << "Grad of axis distance wrt group scales disagrees!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(axisGroupGrad.size(), 3);
    compare.col(0) = axisGroupGrad;
    compare.col(1) = axisGroupGrad_fd;
    compare.col(2) = axisGroupGrad - axisGroupGrad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(axisGroupGrad, axisGroupGrad_fd, 1e-10));
  }
}
#endif