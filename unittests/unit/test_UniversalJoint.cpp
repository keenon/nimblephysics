#include <gtest/gtest.h>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/UniversalJoint.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;

void testUniversalJoint(dart::dynamics::UniversalJoint* joint)
{
  Eigen::Vector2s pos = joint->getPositionsStatic();
  Eigen::Vector2s vel = joint->getVelocitiesStatic();

  const s_t THRESHOLD = 1e-9;

  Eigen::Matrix<s_t, 6, 2> dJ
      = joint->getRelativeJacobianTimeDerivStatic(pos, vel);
  Eigen::Matrix<s_t, 6, 2> dJ_fd
      = joint->finiteDifferenceRelativeJacobianTimeDerivStatic(pos, vel);
  if (!equals(dJ, dJ_fd, THRESHOLD))
  {
    std::cout << "Analytical dJ: " << std::endl << dJ << std::endl;
    std::cout << "FD dJ: " << std::endl << dJ_fd << std::endl;
    std::cout << "Diff: " << std::endl << dJ - dJ_fd << std::endl;
    EXPECT_TRUE(equals(dJ, dJ_fd, THRESHOLD));
    return;
  }

  // Test all the 2-dof derivatives of
  // Jacobians
  for (int j = 0; j < 2; j++)
  {
    Eigen::Matrix<s_t, 6, 2> dpos_J
        = joint->getRelativeJacobianDerivWrtPosition(j);
    Eigen::Matrix<s_t, 6, 2> dpos_J_fd
        = joint->finiteDifferenceRelativeJacobianDerivWrtPos(pos, j);
    if (!equals(dpos_J, dpos_J_fd, THRESHOLD))
    {
      std::cout << "Wrt position: " << j << std::endl;
      std::cout << "Analytical d_J: " << std::endl << dpos_J << std::endl;
      std::cout << "FD d_J: " << std::endl << dpos_J_fd << std::endl;
      std::cout << "Diff: " << std::endl << dpos_J - dpos_J_fd << std::endl;
      EXPECT_TRUE(equals(dpos_J, dpos_J_fd, THRESHOLD));
      return;
    }

    Eigen::Matrix<s_t, 6, 2> dpos_dJ
        = joint->getRelativeJacobianTimeDerivDerivWrtPosition(j);
    Eigen::Matrix<s_t, 6, 2> dpos_dJ_fd
        = joint->finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
            pos, vel, j);
    if (!equals(dpos_dJ, dpos_dJ_fd, THRESHOLD))
    {
      std::cout << "Wrt position: " << j << std::endl;
      std::cout << "Analytical d_dJ: " << std::endl << dpos_dJ << std::endl;
      std::cout << "FD d_dJ: " << std::endl << dpos_dJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << dpos_dJ - dpos_dJ_fd << std::endl;
      EXPECT_TRUE(equals(dpos_dJ, dpos_dJ_fd, THRESHOLD));
      return;
    }

    Eigen::Matrix<s_t, 6, 2> dvel_dJ
        = joint->getRelativeJacobianTimeDerivDerivWrtVelocity(j);
    Eigen::Matrix<s_t, 6, 2> dvel_dJ_fd
        = joint->finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
            pos, vel, j);
    if (!equals(dvel_dJ, dvel_dJ_fd, THRESHOLD))
    {
      std::cout << "Wrt velocity: " << j << std::endl;
      std::cout << "Analytical dvel_dJ: " << std::endl << dvel_dJ << std::endl;
      std::cout << "FD dvel_dJ: " << std::endl << dvel_dJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << dvel_dJ - dvel_dJ_fd << std::endl;
      EXPECT_TRUE(equals(dvel_dJ, dvel_dJ_fd, THRESHOLD));
      return;
    }
  }
}

TEST(UniversalJoint, JACOBIANS)
{
  // Create single-body skeleton with a universal joint
  auto skel = dynamics::Skeleton::create();
  auto pair
      = skel->createJointAndBodyNodePair<dart::dynamics::UniversalJoint>();
  dart::dynamics::UniversalJoint* joint = pair.first;
  auto body = pair.second;

  (void)joint;
  (void)body;

  testUniversalJoint(joint);

  // Test a bunch of random configurations
  for (int i = 0; i < 100; i++)
  {
    Eigen::Vector3s axis1 = Eigen::Vector3s::Random();
    Eigen::Vector3s axis2 = Eigen::Vector3s::Random();
    Eigen::Vector2s pos = Eigen::Vector2s::Random();
    Eigen::Vector2s vel = Eigen::Vector2s::Random();
    Eigen::Vector2s acc = Eigen::Vector2s::Random();

    Eigen::Matrix3s R = math::expMapRot(Eigen::Vector3s::Random());
    Eigen::Vector3s p = Eigen::Vector3s::Random();
    Eigen::Isometry3s child = Eigen::Isometry3s::Identity();
    child.linear() = R;
    child.translation() = p;

    joint->setAxis1(axis1);
    joint->setAxis2(axis2);
    joint->setPositions(pos);
    joint->setVelocities(vel);
    joint->setAccelerations(acc);
    joint->setTransformFromChildBodyNode(child);

    testUniversalJoint(joint);
  }
}