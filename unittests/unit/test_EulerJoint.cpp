#include <gtest/gtest.h>

#include "dart/dart.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;

//==============================================================================
TEST(EulerJoint, JacobianXYZ)
{
  // Create single-body skeleton with a screw joint
  auto skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dart::dynamics::EulerJoint>();
  auto eulerJoint = pair.first;

  eulerJoint->setAxisOrder(EulerJoint::AxisOrder::XYZ);

  std::vector<Eigen::Vector3s> flips;
  flips.push_back(Eigen::Vector3s::Ones());
  flips.push_back(Eigen::Vector3s(-1.0, 1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, -1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, 1.0, -1.0));
  flips.push_back(Eigen::Vector3s::Ones() * -1);

  for (Eigen::Vector3s& flip : flips)
  {
    std::cout << "Testing flip " << flip << std::endl;
    eulerJoint->setFlipAxisMap(flip);
    for (int i = 0; i < 100; i++)
    {
      eulerJoint->setPositions(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      eulerJoint->setVelocities(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      Eigen::MatrixXs jac = eulerJoint->getRelativeJacobian();
      Eigen::MatrixXs jac_fd
          = eulerJoint->finiteDifferenceRelativeJacobianInPositionSpace();
      if (!equals(jac, jac_fd, 1e-8))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Euler XYZ Jac: " << std::endl << jac << std::endl;
        std::cout << "Euler XYZ Jac FD: " << std::endl << jac_fd << std::endl;
        std::cout << "Diff: " << std::endl << jac - jac_fd << std::endl;
        EXPECT_TRUE(equals(jac, jac_fd, 1e-8));
        return;
      }

      Eigen::MatrixXs dJdt = eulerJoint->getRelativeJacobianTimeDeriv();
      Eigen::MatrixXs dJdt_fd
          = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
              eulerJoint->getPositions(),
              eulerJoint->getVelocities(),
              eulerJoint->getAxisOrder(),
              eulerJoint->getFlipAxisMap(),
              eulerJoint->getTransformFromChildBodyNode());
      if (!equals(dJdt, dJdt_fd, 1e-7))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Velocities: " << std::endl
                  << eulerJoint->getVelocities() << std::endl;
        std::cout << "Euler XYZ dJac/dt: " << std::endl << dJdt << std::endl;
        std::cout << "Euler XYZ dJac/dt FD: " << std::endl
                  << dJdt_fd << std::endl;
        std::cout << "Diff: " << std::endl << dJdt - dJdt_fd << std::endl;
        EXPECT_TRUE(equals(dJdt, dJdt_fd, 1e-7));
        return;
      }
    }
  }
}

//==============================================================================
TEST(EulerJoint, JacobianZYX)
{
  // Create single-body skeleton with a screw joint
  auto skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dart::dynamics::EulerJoint>();
  auto eulerJoint = pair.first;

  eulerJoint->setAxisOrder(EulerJoint::AxisOrder::ZYX);

  std::vector<Eigen::Vector3s> flips;
  flips.push_back(Eigen::Vector3s::Ones());
  flips.push_back(Eigen::Vector3s(-1.0, 1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, -1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, 1.0, -1.0));
  flips.push_back(Eigen::Vector3s::Ones() * -1);

  for (Eigen::Vector3s& flip : flips)
  {
    std::cout << "Testing flip " << flip << std::endl;
    eulerJoint->setFlipAxisMap(flip);
    for (int i = 0; i < 100; i++)
    {
      eulerJoint->setPositions(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      eulerJoint->setVelocities(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      Eigen::MatrixXs jac = eulerJoint->getRelativeJacobian();
      Eigen::MatrixXs jac_fd
          = eulerJoint->finiteDifferenceRelativeJacobianInPositionSpace();
      if (!equals(jac, jac_fd, 1e-8))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Euler ZYX Jac: " << std::endl << jac << std::endl;
        std::cout << "Euler ZYX Jac FD: " << std::endl << jac_fd << std::endl;
        std::cout << "Diff: " << std::endl << jac - jac_fd << std::endl;
        EXPECT_TRUE(equals(jac, jac_fd, 1e-8));
        return;
      }

      Eigen::MatrixXs dJdt = eulerJoint->getRelativeJacobianTimeDeriv();
      Eigen::MatrixXs dJdt_fd
          = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
              eulerJoint->getPositions(),
              eulerJoint->getVelocities(),
              eulerJoint->getAxisOrder(),
              eulerJoint->getFlipAxisMap(),
              eulerJoint->getTransformFromChildBodyNode());
      if (!equals(dJdt, dJdt_fd, 1e-7))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Velocities: " << std::endl
                  << eulerJoint->getVelocities() << std::endl;
        std::cout << "Euler ZYX dJac/dt: " << std::endl << dJdt << std::endl;
        std::cout << "Euler ZYX dJac/dt FD: " << std::endl
                  << dJdt_fd << std::endl;
        std::cout << "Diff: " << std::endl << dJdt - dJdt_fd << std::endl;
        EXPECT_TRUE(equals(dJdt, dJdt_fd, 1e-7));
        return;
      }

      for (int j = 0; j < 3; j++)
      {
        Eigen::MatrixXs dJdP
            = eulerJoint->getRelativeJacobianDerivWrtPosition(j);
        Eigen::MatrixXs dJdP_fd
            = EulerJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
                eulerJoint->getPositions(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP, dJdP_fd, 1e-8))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZYX dJac/dq: " << std::endl << dJdP << std::endl;
          std::cout << "Euler ZYX dJac/dq FD: " << std::endl
                    << dJdP_fd << std::endl;
          std::cout << "Diff: " << std::endl << dJdP - dJdP_fd << std::endl;
          EXPECT_TRUE(equals(dJdP, dJdP_fd, 1e-8));
          return;
        }

        Eigen::MatrixXs dJdP_dq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_dq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_dq, dJdP_dq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZYX dJac/dP dq: " << std::endl
                    << dJdP_dq << std::endl;
          std::cout << "Euler ZYX dJac/dP dq FD: " << std::endl
                    << dJdP_dq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_dq - dJdP_dq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_dq, dJdP_dq_fd, 1e-7));
          return;
        }

        Eigen::MatrixXs dJdP_ddq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_ddq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_ddq, dJdP_ddq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZYX dJac/dP ddq: " << std::endl
                    << dJdP_ddq << std::endl;
          std::cout << "Euler ZYX dJac/dP ddq FD: " << std::endl
                    << dJdP_ddq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_ddq - dJdP_ddq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_ddq, dJdP_ddq_fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(EulerJoint, JacobianXZY)
{
  // Create single-body skeleton with a screw joint
  auto skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dart::dynamics::EulerJoint>();
  auto eulerJoint = pair.first;

  std::vector<Eigen::Vector3s> flips;
  flips.push_back(Eigen::Vector3s::Ones());
  flips.push_back(Eigen::Vector3s(-1.0, 1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, -1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, 1.0, -1.0));
  flips.push_back(Eigen::Vector3s::Ones() * -1);

  for (Eigen::Vector3s& flip : flips)
  {
    std::cout << "Testing flip " << flip << std::endl;
    eulerJoint->setFlipAxisMap(flip);
    eulerJoint->setAxisOrder(EulerJoint::AxisOrder::XZY);
    for (int i = 0; i < 100; i++)
    {
      eulerJoint->setPositions(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      eulerJoint->setVelocities(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      Eigen::MatrixXs jac = eulerJoint->getRelativeJacobian();
      Eigen::MatrixXs jac_fd
          = eulerJoint->finiteDifferenceRelativeJacobianInPositionSpace();
      if (!equals(jac, jac_fd, 1e-8))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Euler XZY Jac: " << std::endl << jac << std::endl;
        std::cout << "Euler XZY Jac FD: " << std::endl << jac_fd << std::endl;
        std::cout << "s0: " << sin(eulerJoint->getPosition(0)) << std::endl;
        std::cout << "c0: " << cos(eulerJoint->getPosition(0)) << std::endl;
        std::cout << "s1: " << sin(eulerJoint->getPosition(1)) << std::endl;
        std::cout << "c1: " << cos(eulerJoint->getPosition(1)) << std::endl;
        std::cout << "s2: " << sin(eulerJoint->getPosition(2)) << std::endl;
        std::cout << "c2: " << cos(eulerJoint->getPosition(2)) << std::endl;
        std::cout << "s1*s2: "
                  << sin(eulerJoint->getPosition(1))
                         * sin(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "s1*c2: "
                  << sin(eulerJoint->getPosition(1))
                         * cos(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "c1*s2: "
                  << cos(eulerJoint->getPosition(1))
                         * sin(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "c1*c2: "
                  << cos(eulerJoint->getPosition(1))
                         * cos(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "Diff: " << std::endl << jac - jac_fd << std::endl;
        EXPECT_TRUE(equals(jac, jac_fd, 1e-8));
        return;
      }

      Eigen::MatrixXs dJdt = eulerJoint->getRelativeJacobianTimeDeriv();
      Eigen::MatrixXs dJdt_fd
          = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
              eulerJoint->getPositions(),
              eulerJoint->getVelocities(),
              eulerJoint->getAxisOrder(),
              eulerJoint->getFlipAxisMap(),
              eulerJoint->getTransformFromChildBodyNode());
      if (!equals(dJdt, dJdt_fd, 1e-7))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Velocities: " << std::endl
                  << eulerJoint->getVelocities() << std::endl;
        std::cout << "Euler XZY dJac/dt: " << std::endl << dJdt << std::endl;
        std::cout << "Euler XZY dJac/dt FD: " << std::endl
                  << dJdt_fd << std::endl;
        std::cout << "Diff: " << std::endl << dJdt - dJdt_fd << std::endl;
        EXPECT_TRUE(equals(dJdt, dJdt_fd, 1e-7));
        return;
      }

      for (int j = 0; j < 3; j++)
      {
        Eigen::MatrixXs dJdP
            = eulerJoint->getRelativeJacobianDerivWrtPosition(j);
        Eigen::MatrixXs dJdP_fd
            = EulerJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
                eulerJoint->getPositions(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP, dJdP_fd, 1e-8))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler XZY dJac/dq: " << std::endl << dJdP << std::endl;
          std::cout << "Euler XZY dJac/dq FD: " << std::endl
                    << dJdP_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "s0: " << sin(eulerJoint->getPosition(0)) << std::endl;
          std::cout << "c0: " << cos(eulerJoint->getPosition(0)) << std::endl;
          std::cout << "s1: " << sin(eulerJoint->getPosition(1)) << std::endl;
          std::cout << "c1: " << cos(eulerJoint->getPosition(1)) << std::endl;
          std::cout << "s2: " << sin(eulerJoint->getPosition(2)) << std::endl;
          std::cout << "c2: " << cos(eulerJoint->getPosition(2)) << std::endl;
          std::cout << "s1*s2: "
                    << sin(eulerJoint->getPosition(1))
                           * sin(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "s1*c2: "
                    << sin(eulerJoint->getPosition(1))
                           * cos(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "c1*s2: "
                    << cos(eulerJoint->getPosition(1))
                           * sin(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "c1*c2: "
                    << cos(eulerJoint->getPosition(1))
                           * cos(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "Diff: " << std::endl << dJdP - dJdP_fd << std::endl;
          EXPECT_TRUE(equals(dJdP, dJdP_fd, 1e-8));
          return;
        }

        Eigen::MatrixXs dJdP_dq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_dq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_dq, dJdP_dq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler XZY dJac/dP dq: " << std::endl
                    << dJdP_dq << std::endl;
          std::cout << "Euler XZY dJac/dP dq FD: " << std::endl
                    << dJdP_dq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_dq - dJdP_dq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_dq, dJdP_dq_fd, 1e-7));
          return;
        }

        Eigen::MatrixXs dJdP_ddq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_ddq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_ddq, dJdP_ddq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler XZY dJac/dP ddq: " << std::endl
                    << dJdP_ddq << std::endl;
          std::cout << "Euler XZY dJac/dP ddq FD: " << std::endl
                    << dJdP_ddq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_ddq - dJdP_ddq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_ddq, dJdP_ddq_fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(EulerJoint, JacobianZXY)
{
  // Create single-body skeleton with a screw joint
  auto skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dart::dynamics::EulerJoint>();
  auto eulerJoint = pair.first;

  std::vector<Eigen::Vector3s> flips;
  flips.push_back(Eigen::Vector3s::Ones());
  flips.push_back(Eigen::Vector3s(-1.0, 1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, -1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, 1.0, -1.0));
  flips.push_back(Eigen::Vector3s::Ones() * -1);

  for (Eigen::Vector3s& flip : flips)
  {
    std::cout << "Testing flip " << flip << std::endl;
    eulerJoint->setFlipAxisMap(flip);
    eulerJoint->setAxisOrder(EulerJoint::AxisOrder::ZXY);
    for (int i = 0; i < 100; i++)
    {
      eulerJoint->setPositions(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      eulerJoint->setVelocities(
          Eigen::VectorXs::Random(eulerJoint->getNumDofs()));
      Eigen::MatrixXs jac = eulerJoint->getRelativeJacobian();
      Eigen::MatrixXs jac_fd
          = eulerJoint->finiteDifferenceRelativeJacobianInPositionSpace();
      if (!equals(jac, jac_fd, 1e-8))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Euler ZXY Jac: " << std::endl << jac << std::endl;
        std::cout << "Euler ZXY Jac FD: " << std::endl << jac_fd << std::endl;
        std::cout << "s0: " << sin(eulerJoint->getPosition(0)) << std::endl;
        std::cout << "c0: " << cos(eulerJoint->getPosition(0)) << std::endl;
        std::cout << "s1: " << sin(eulerJoint->getPosition(1)) << std::endl;
        std::cout << "c1: " << cos(eulerJoint->getPosition(1)) << std::endl;
        std::cout << "s2: " << sin(eulerJoint->getPosition(2)) << std::endl;
        std::cout << "c2: " << cos(eulerJoint->getPosition(2)) << std::endl;
        std::cout << "s1*s2: "
                  << sin(eulerJoint->getPosition(1))
                         * sin(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "s1*c2: "
                  << sin(eulerJoint->getPosition(1))
                         * cos(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "c1*s2: "
                  << cos(eulerJoint->getPosition(1))
                         * sin(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "c1*c2: "
                  << cos(eulerJoint->getPosition(1))
                         * cos(eulerJoint->getPosition(2))
                  << std::endl;
        std::cout << "Diff: " << std::endl << jac - jac_fd << std::endl;
        EXPECT_TRUE(equals(jac, jac_fd, 1e-8));
        return;
      }

      Eigen::MatrixXs dJdt = eulerJoint->getRelativeJacobianTimeDeriv();
      Eigen::MatrixXs dJdt_fd
          = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
              eulerJoint->getPositions(),
              eulerJoint->getVelocities(),
              eulerJoint->getAxisOrder(),
              eulerJoint->getFlipAxisMap(),
              eulerJoint->getTransformFromChildBodyNode());
      if (!equals(dJdt, dJdt_fd, 1e-7))
      {
        std::cout << "Positions: " << std::endl
                  << eulerJoint->getPositions() << std::endl;
        std::cout << "Velocities: " << std::endl
                  << eulerJoint->getVelocities() << std::endl;
        std::cout << "Euler ZXY dJac/dt: " << std::endl << dJdt << std::endl;
        std::cout << "Euler ZXY dJac/dt FD: " << std::endl
                  << dJdt_fd << std::endl;
        std::cout << "Diff: " << std::endl << dJdt - dJdt_fd << std::endl;
        EXPECT_TRUE(equals(dJdt, dJdt_fd, 1e-7));
        return;
      }

      for (int j = 0; j < 3; j++)
      {
        Eigen::MatrixXs dJdP
            = eulerJoint->getRelativeJacobianDerivWrtPosition(j);
        Eigen::MatrixXs dJdP_fd
            = EulerJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
                eulerJoint->getPositions(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP, dJdP_fd, 1e-8))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZXY dJac/dq: " << std::endl << dJdP << std::endl;
          std::cout << "Euler ZXY dJac/dq FD: " << std::endl
                    << dJdP_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "s0: " << sin(eulerJoint->getPosition(0)) << std::endl;
          std::cout << "c0: " << cos(eulerJoint->getPosition(0)) << std::endl;
          std::cout << "s1: " << sin(eulerJoint->getPosition(1)) << std::endl;
          std::cout << "c1: " << cos(eulerJoint->getPosition(1)) << std::endl;
          std::cout << "s2: " << sin(eulerJoint->getPosition(2)) << std::endl;
          std::cout << "c2: " << cos(eulerJoint->getPosition(2)) << std::endl;
          std::cout << "s1*s2: "
                    << sin(eulerJoint->getPosition(1))
                           * sin(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "s1*c2: "
                    << sin(eulerJoint->getPosition(1))
                           * cos(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "c1*s2: "
                    << cos(eulerJoint->getPosition(1))
                           * sin(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "c1*c2: "
                    << cos(eulerJoint->getPosition(1))
                           * cos(eulerJoint->getPosition(2))
                    << std::endl;
          std::cout << "Diff: " << std::endl << dJdP - dJdP_fd << std::endl;
          EXPECT_TRUE(equals(dJdP, dJdP_fd, 1e-8));
          return;
        }
        Eigen::MatrixXs dJdP_dq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_dq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_dq, dJdP_dq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZXY dJac/dP dq: " << std::endl
                    << dJdP_dq << std::endl;
          std::cout << "Euler ZXY dJac/dP dq FD: " << std::endl
                    << dJdP_dq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_dq - dJdP_dq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_dq, dJdP_dq_fd, 1e-7));
          return;
        }

        Eigen::MatrixXs dJdP_ddq
            = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
                j,
                eulerJoint->getPositions(),
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        Eigen::MatrixXs dJdP_ddq_fd
            = EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
                eulerJoint->getPositions(),
                eulerJoint->getVelocities(),
                j,
                eulerJoint->getAxisOrder(),
                eulerJoint->getFlipAxisMap(),
                eulerJoint->getTransformFromChildBodyNode());
        if (!equals(dJdP_ddq, dJdP_ddq_fd, 1e-7))
        {
          std::cout << "Positions: " << std::endl
                    << eulerJoint->getPositions() << std::endl;
          std::cout << "Euler ZXY dJac/dP ddq: " << std::endl
                    << dJdP_ddq << std::endl;
          std::cout << "Euler ZXY dJac/dP ddq FD: " << std::endl
                    << dJdP_ddq_fd << std::endl;
          std::cout << "Index = " << j << std::endl;
          std::cout << "Diff: " << std::endl
                    << dJdP_ddq - dJdP_ddq_fd << std::endl;
          EXPECT_TRUE(equals(dJdP_ddq, dJdP_ddq_fd, 1e-7));
          return;
        }
      }
    }
  }
}