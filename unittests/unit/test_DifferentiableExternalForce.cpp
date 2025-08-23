#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
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

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define JACOBIAN_TESTS
// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

bool testWorldWrenchJacobian(
    std::shared_ptr<dynamics::Skeleton> skel,
    int body,
    Eigen::Vector6s worldWrench)
{
  DifferentiableExternalForce force(skel, body);

  Eigen::MatrixXs analytical
      = force.getJacobianOfTauWrtWorldWrench(worldWrench);
  Eigen::MatrixXs fd
      = force.finiteDifferenceJacobianOfTauWrtWorldWrench(worldWrench);

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Jacobian of tau wrt world wrench not equal!" << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "FD:" << std::endl << fd << std::endl;
    std::cout << "Diff:" << std::endl << fd - analytical << std::endl;
    return false;
  }

  return true;
}

bool testScrewAxisGradients(
    std::shared_ptr<dynamics::Skeleton> skel,
    int body,
    Eigen::Vector6s worldWrench)
{
  (void)worldWrench;
  DifferentiableExternalForce force(skel, body);

  for (int col = 0; col < skel->getNumDofs(); col++)
  {
    for (int row = 0; row < skel->getNumDofs(); row++)
    {
      dynamics::DegreeOfFreedom* moveDof = skel->getDof(col);

      s_t originalPos = moveDof->getPosition();
      Eigen::Vector6s fd;
      math::finiteDifference<Eigen::Vector6s>(
          [&](
              /* in*/ s_t eps,
              /*out*/ Eigen::Vector6s& perturbed) {
            moveDof->setPosition(originalPos + eps);
            perturbed
                = DifferentiableContactConstraint::getWorldScrewAxisForForce(
                    skel->getDof(row));
            return true;
          },
          fd,
          1e-2,
          true);
      moveDof->setPosition(originalPos);

      Eigen::Vector6s normal
          = DifferentiableContactConstraint::getScrewAxisForForceGradient(
              skel->getDof(row), skel->getDof(col));

      if (!equals(fd, normal, 1e-8))
      {
        std::cout << "Gradient of Jac force on DOF " << row << " ("
                  << skel->getDof(row)->getJoint()->getName() << " index "
                  << skel->getDof(row)->getIndexInJoint()
                  << ") with respect to DOF " << col << " ("
                  << skel->getDof(col)->getJoint()->getName() << " index "
                  << skel->getDof(col)->getIndexInJoint()
                  << ") didn't equal analytical!" << std::endl;
        std::cout << "Analytical: " << normal << std::endl;
        std::cout << "FD: " << fd << std::endl;
        std::cout << "Diff: " << fd - normal << std::endl;
        return false;
      }
    }
  }

  return true;
}

bool testGroupScaleMoveBody(std::shared_ptr<dynamics::Skeleton> skel, int body)
{
  for (int col = 0; col < skel->getGroupScaleDim(); col++)
  {
    Eigen::Vector3s fd
        = skel->finiteDifferenceGroupScaleMovementOnBodyInWorldSpace(col, body);
    Eigen::Vector3s normal
        = skel->getGroupScaleMovementOnBodyInWorldSpace(col, body);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of world pos of body "
                << skel->getBodyNode(body)->getName()
                << " with respect to group scale DOF "
                << skel->debugGroupScaleIndex(col)
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << normal << std::endl;
      std::cout << "FD: " << fd << std::endl;
      std::cout << "Diff: " << fd - normal << std::endl;
      return false;
    }
  }

  return true;
}

bool testBodyScaleJointJacobians(
    std::shared_ptr<dynamics::Skeleton> skel, int joint)
{
  for (int col = 0; col < skel->getGroupScaleDim(); col++)
  {
    Eigen::Vector3s fd
        = skel->finiteDifferenceGroupScaleMovementOnJointInWorldSpace(
            col, joint);
    Eigen::Vector3s normal
        = skel->getGroupScaleMovementOnJointInWorldSpace(col, joint);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of world pos of joint "
                << skel->getJoint(joint)->getName()
                << " with respect to group scale DOF "
                << skel->debugGroupScaleIndex(col)
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << normal << std::endl;
      std::cout << "FD: " << fd << std::endl;
      std::cout << "Diff: " << fd - normal << std::endl;
      return false;
    }
  }

  return true;
}

bool testJacobianWrt(
    std::shared_ptr<dynamics::Skeleton> skel,
    int body,
    Eigen::Vector6s worldWrench,
    neural::WithRespectTo* wrt)
{
  if (wrt == WithRespectTo::POSITION)
  {
    if (!testScrewAxisGradients(skel, body, worldWrench))
    {
      return false;
    }
  }

  DifferentiableExternalForce force(skel, body);

  Eigen::MatrixXs analytical = force.getJacobianOfTauWrt(worldWrench, wrt);
  Eigen::MatrixXs fd = force.finiteDifferenceJacobianOfTauWrt(worldWrench, wrt);

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Jacobian of tau wrt " << wrt->name() << " not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "FD:" << std::endl << fd << std::endl;
    std::cout << "Diff (" << (fd - analytical).minCoeff() << " - "
              << (fd - analytical).maxCoeff() << "):" << std::endl
              << (fd - analytical) << std::endl;
    return false;
  }

  return true;
}

bool testAllJacobians(
    std::shared_ptr<dynamics::Skeleton> skel,
    int body,
    Eigen::Vector6s worldWrench)
{
  if (!testWorldWrenchJacobian(skel, body, worldWrench))
  {
    return false;
  }

  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    if (!testGroupScaleMoveBody(skel, i))
    {
      return false;
    }
  }
  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    if (!testBodyScaleJointJacobians(skel, i))
    {
      return false;
    }
  }

  std::vector<neural::WithRespectTo*> wrts;
  wrts.push_back(WithRespectTo::POSITION);
  wrts.push_back(WithRespectTo::VELOCITY);
  wrts.push_back(WithRespectTo::FORCE);
  wrts.push_back(WithRespectTo::GROUP_SCALES);
  wrts.push_back(WithRespectTo::GROUP_MASSES);
  wrts.push_back(WithRespectTo::GROUP_COMS);
  wrts.push_back(WithRespectTo::GROUP_INERTIAS);

  for (auto* wrt : wrts)
  {
    if (!testJacobianWrt(skel, body, worldWrench, wrt))
    {
      return false;
    }
  }

  return true;
}

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, TEST_OPENSIM_JACOBIANS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);
  file.skeleton->setPositions(file.skeleton->getRandomPose());
  file.skeleton->setVelocities(file.skeleton->getRandomVelocity());

  std::map<int, Eigen::Vector6s> worldForces;
  EXPECT_TRUE(testAllJacobians(
      file.skeleton,
      file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector6s::Random()));
  EXPECT_TRUE(testAllJacobians(
      file.skeleton,
      file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton(),
      Eigen::Vector6s::Random()));
}
#endif