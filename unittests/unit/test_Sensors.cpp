#include <gtest/gtest.h>

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

using namespace dart;
using namespace biomechanics;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(Sensors, HEIGHT)
{
  std::shared_ptr<dynamics::Skeleton> standard
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  EXPECT_NE(0, standard->getHeight(standard->getPositions()));

  srand(30);
  Eigen::VectorXs pos = standard->getRandomPose();

  Eigen::VectorXs analytical = standard->getGradientOfHeightWrtBodyScales(pos);
  Eigen::VectorXs bruteForce
      = standard->finiteDifferenceGradientOfHeightWrtBodyScales(pos);
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Gradient of height wrt body scales not equal!" << std::endl;
    Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(analytical.size(), 3);
    diff.col(0) = analytical;
    diff.col(1) = bruteForce;
    diff.col(2) = analytical - bruteForce;
    std::cout << "Analytical - Brute Force - Diff" << std::endl
              << diff << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(Sensors, LOWEST_POINT)
{
  std::shared_ptr<dynamics::Skeleton> standard
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;

  srand(30);
  Eigen::VectorXs pos = standard->getRandomPose();
  standard->setPositions(pos);

  EXPECT_NE(0, standard->getLowestPoint());

  Eigen::VectorXs analytical
      = standard->getGradientOfLowestPointWrtBodyScales();
  Eigen::VectorXs bruteForce
      = standard->finiteDifferenceGradientOfLowestPointWrtBodyScales();
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Gradient of lowest point wrt body scales not equal!"
              << std::endl;
    Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(analytical.size(), 3);
    diff.col(0) = analytical;
    diff.col(1) = bruteForce;
    diff.col(2) = analytical - bruteForce;
    std::cout << "Analytical - Brute Force - Diff" << std::endl
              << diff << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
  }

  analytical = standard->getGradientOfLowestPointWrtJoints();
  bruteForce = standard->finiteDifferenceGradientOfLowestPointWrtJoints();
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Gradient of lowest point wrt joints not equal!" << std::endl;
    Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(analytical.size(), 3);
    diff.col(0) = analytical;
    diff.col(1) = bruteForce;
    diff.col(2) = analytical - bruteForce;
    std::cout << "Analytical - Brute Force - Diff" << std::endl
              << diff << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(Sensors, MARKERS_WRT_MARKERS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> standard = file.skeleton;

  srand(30);
  Eigen::VectorXs pos = standard->getRandomPose();
  standard->setPositions(pos);

  for (auto pair : file.markersMap)
  {
    Eigen::Vector3s analytical
        = pair.second.first
              ->getGradientOfDistToClosestVerticesToMarkerWrtMarker(
                  pair.second.second);
    Eigen::Vector3s bruteForce
        = pair.second.first
              ->finiteDifferenceGradientOfDistToClosestVerticesToMarkerWrtMarker(
                  pair.second.second);
    if (!equals(analytical, bruteForce, 1e-8))
    {
      std::cout << "Gradient of dist(marker, closest vertex) wrt marker!"
                << std::endl;
      Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(analytical.size(), 3);
      diff.col(0) = analytical;
      diff.col(1) = bruteForce;
      diff.col(2) = analytical - bruteForce;
      std::cout << "Analytical - Brute Force - Diff" << std::endl
                << diff << std::endl;
      EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(Sensors, MARKERS_WRT_SCALE)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> standard = file.skeleton;

  srand(30);
  Eigen::VectorXs pos = standard->getRandomPose();
  standard->setPositions(pos);

  Eigen::VectorXs bodyScales
      = Eigen::VectorXs::Ones(standard->getNumBodyNodes() * 3)
        + 0.1 * Eigen::VectorXs::Random(standard->getNumBodyNodes() * 3);
  standard->setBodyScales(bodyScales);

  for (auto pair : file.markersMap)
  {
    Eigen::Vector3s analytical
        = pair.second.first
              ->getGradientOfDistToClosestVerticesToMarkerWrtBodyScale(
                  pair.second.second);
    Eigen::Vector3s bruteForce
        = pair.second.first
              ->finiteDifferenceGradientOfDistToClosestVerticesToMarkerWrtBodyScale(
                  pair.second.second);
    if (!equals(analytical, bruteForce, 1e-8))
    {
      std::cout << "Gradient of dist(marker, closest vertex) wrt marker!"
                << std::endl;
      Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(analytical.size(), 3);
      diff.col(0) = analytical;
      diff.col(1) = bruteForce;
      diff.col(2) = analytical - bruteForce;
      std::cout << "Analytical - Brute Force - Diff" << std::endl
                << diff << std::endl;
      EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
    }
  }
}
#endif