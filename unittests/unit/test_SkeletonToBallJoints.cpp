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

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

TEST(SkeletonConverter, CONVERT_OSIM)
{
  std::shared_ptr<dynamics::Skeleton> osim = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim").skeleton;
  osim->getBodyNode("tibia_l")->setScale(1.2);
  std::shared_ptr<dynamics::Skeleton> converted
      = osim->convertSkeletonToBallJoints();

  assert(
      osim->getBodyNode("tibia_l")->getScale()
      == converted->getBodyNode("tibia_l")->getScale());
  assert(
      osim->getBodyNode("tibia_l")->getParentJoint()->getChildScale()
      == converted->getBodyNode("tibia_l")->getParentJoint()->getChildScale());
  assert(
      osim->getBodyNode("tibia_l")->getChildJoint(0)->getParentScale()
      == converted->getBodyNode("tibia_l")->getChildJoint(0)->getParentScale());

  osim->getBodyNode("tibia_l")->setScale(1.4);
  converted->getBodyNode("tibia_l")->setScale(1.4);

#ifndef NDEBUG
  assert(osim->getNumJoints() == converted->getNumJoints());
  for (int i = 0; i < osim->getNumJoints(); i++)
  {
    dynamics::Joint* joint = converted->getJoint(i);
    assert(joint != nullptr);
    assert(joint != osim->getJoint(i));
    assert(joint->getName() == osim->getJoint(i)->getName());
    assert(
        joint->getTransformFromChildBodyNode().matrix()
        == osim->getJoint(i)->getTransformFromChildBodyNode().matrix());
    assert(
        joint->getTransformFromParentBodyNode().matrix()
        == osim->getJoint(i)->getTransformFromParentBodyNode().matrix());
  }
#endif

  EXPECT_EQ(osim->getNumDofs(), converted->getNumDofs());
  std::cout << "Converted skeleton DOFs: " << converted->getNumDofs()
            << std::endl;

  for (int i = 0; i < 100; i++)
  {
    Eigen::VectorXs positions = Eigen::VectorXs::Random(osim->getNumDofs());
    Eigen::VectorXs ballSpace = osim->convertPositionsToBallSpace(positions);
    Eigen::VectorXs recovered = osim->convertPositionsFromBallSpace(ballSpace);
    EXPECT_TRUE(equals(positions, recovered, 1e-12));

    osim->setPositions(positions);
    converted->setPositions(ballSpace);

    for (int j = 0; j < osim->getNumBodyNodes(); j++)
    {
      Eigen::Matrix4s Teuler
          = osim->getBodyNode(j)->getWorldTransform().matrix();
      Eigen::Matrix4s Tball
          = converted->getBodyNode(j)->getWorldTransform().matrix();
      if (!equals(Teuler, Tball, 1e-8))
      {
        std::cout << "Error on Body[" << j
                  << "]: " << osim->getBodyNode(j)->getName() << std::endl;
        std::cout << "Teuler: " << std::endl << Teuler << std::endl;
        std::cout << "Tball: " << std::endl << Tball << std::endl;
        std::cout << "Diff: " << std::endl << Teuler - Tball << std::endl;
        EXPECT_TRUE(equals(Teuler, Tball, 1e-12));
        return;
      }
    }
  }
}