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

TEST(Anthropometrics, HEIGHT_AND_GRAD)
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