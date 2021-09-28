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