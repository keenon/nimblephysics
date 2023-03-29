#include <vector>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
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
TEST(JointForceFields, FORCE_FIELD_JACS_GRAD)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;

  std::vector<dynamics::Joint*> joints;
  for (int i = 0; i < osim->getNumJoints(); i++)
  {
    joints.push_back(osim->getJoint(i));
  }

  std::vector<neural::WithRespectTo*> wrts;
  wrts.push_back(neural::WithRespectTo::POSITION);
  wrts.push_back(neural::WithRespectTo::VELOCITY);
  wrts.push_back(neural::WithRespectTo::ACCELERATION);
  wrts.push_back(neural::WithRespectTo::GROUP_SCALES);
  wrts.push_back(neural::WithRespectTo::GROUP_COMS);

  for (int i = 0; i < joints.size(); i++)
  {
    std::cout << "  Testing Joint " << i << "/" << joints.size() << std::endl;
    for (neural::WithRespectTo* wrt : wrts)
    {
      std::cout << "Testing WRT " << wrt->name() << std::endl;
      Eigen::MatrixXs analyticalJac
          = osim->getJointDistanceToOtherJointsJacobianWrt(joints, i, wrt);
      Eigen::MatrixXs bruteForceJac
          = osim->finiteDifferenceJointDistanceToOtherJointsJacobianWrt(
              joints, i, wrt);
      if (!equals(analyticalJac, bruteForceJac, 1e-8))
      {
        std::cout << "Joint distance jacobian wrt " << wrt->name()
                  << " disagrees!" << std::endl;
        std::cout << "Analytical: " << std::endl << analyticalJac << std::endl;
        std::cout << "Brute force: " << std::endl << bruteForceJac << std::endl;
        std::cout << "Diff: " << std::endl
                  << analyticalJac - bruteForceJac << std::endl;
        EXPECT_TRUE(equals(analyticalJac, bruteForceJac, 1e-8));
      }

      Eigen::MatrixXs analyticalGrad
          = osim->getJointForceFieldToOtherJointsGradient(
              joints, i, 10.0, 0.2, wrt);
      Eigen::MatrixXs bruteForceGrad
          = osim->finiteDifferenceJointForceFieldToOtherJointsGradient(
              joints, i, 10.0, 0.2, wrt);
      if (!equals(analyticalGrad, bruteForceGrad, 1e-8))
      {
        std::cout << "Joint force field grad wrt " << wrt->name()
                  << " disagrees!" << std::endl;
        std::cout << "Analytical: " << std::endl << analyticalGrad << std::endl;
        std::cout << "Brute force: " << std::endl
                  << bruteForceGrad << std::endl;
        std::cout << "Diff: " << std::endl
                  << analyticalGrad - bruteForceGrad << std::endl;
        EXPECT_TRUE(equals(analyticalGrad, bruteForceGrad, 1e-8));
      }
    }
    std::cout << "Testing WRT body scales" << std::endl;
    Eigen::MatrixXs analyticalJac
        = osim->getJointDistanceToOtherJointsJacobianWrtBodyScales(joints, i);
    Eigen::MatrixXs bruteForceJac
        = osim->finiteDifferenceJointDistanceToOtherJointsJacobianWrtBodyScales(
            joints, i);
    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Joint distance jacobian wrt body scales disagrees!"
                << std::endl;
      std::cout << "Analytical: " << std::endl << analyticalJac << std::endl;
      std::cout << "Brute force: " << std::endl << bruteForceJac << std::endl;
      std::cout << "Diff: " << std::endl
                << analyticalJac - bruteForceJac << std::endl;
      EXPECT_TRUE(equals(analyticalJac, bruteForceJac, 1e-8));
    }

    Eigen::MatrixXs analyticalGrad
        = osim->getJointForceFieldToOtherJointsGradientWrtBodyScales(
            joints, i, 10.0, 0.2);
    Eigen::MatrixXs bruteForceGrad
        = osim->finiteDifferenceJointForceFieldToOtherJointsGradientWrtBodyScales(
            joints, i, 10.0, 0.2);
    if (!equals(analyticalGrad, bruteForceGrad, 1e-8))
    {
      std::cout << "Joint force field grad wrt body scales disagrees!"
                << std::endl;
      std::cout << "Analytical: " << std::endl << analyticalGrad << std::endl;
      std::cout << "Brute force: " << std::endl << bruteForceGrad << std::endl;
      std::cout << "Diff: " << std::endl
                << analyticalGrad - bruteForceGrad << std::endl;
      EXPECT_TRUE(equals(analyticalGrad, bruteForceGrad, 1e-8));
    }
  }
}
#endif