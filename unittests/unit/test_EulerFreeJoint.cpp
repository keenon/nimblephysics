#include <gtest/gtest.h>

#include "dart/dart.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;

//==============================================================================
TEST(Geometry, EULER_XYZ_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      Eigen::MatrixXs grad = math::eulerXYZToMatrixGrad(rand, j);
      Eigen::MatrixXs fd = math::eulerXYZToMatrixFiniteDifference(rand, j);
      if (!equals(grad, fd, 1e-7))
      {
        std::cout << "Euler XYZ Grad[" << j << "]: " << std::endl
                  << grad << std::endl;
        std::cout << "Euler XYZ FD[" << j << "]: " << std::endl
                  << fd << std::endl;
        std::cout << "Diff: " << std::endl << grad - fd << std::endl;
        EXPECT_TRUE(equals(grad, fd, 1e-7));
        return;
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_XYZ_SECOND_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        Eigen::MatrixXs grad = math::eulerXYZToMatrixSecondGrad(rand, j, k);
        Eigen::MatrixXs fd
            = math::eulerXYZToMatrixSecondFiniteDifference(rand, j, k);
        if (!equals(grad, fd, 1e-7))
        {
          std::cout << "Euler XYZ Grad[" << j << "," << k << "]: " << std::endl
                    << grad << std::endl;
          std::cout << "Euler XYZ FD[" << j << "," << k << "]: " << std::endl
                    << fd << std::endl;
          std::cout << "Diff: " << std::endl << grad - fd << std::endl;
          EXPECT_TRUE(equals(grad, fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_XZY_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      Eigen::MatrixXs grad = math::eulerXZYToMatrixGrad(rand, j);
      Eigen::MatrixXs fd = math::eulerXZYToMatrixFiniteDifference(rand, j);
      if (!equals(grad, fd, 1e-7))
      {
        std::cout << "Euler XZY Grad[" << j << "]: " << std::endl
                  << grad << std::endl;
        std::cout << "Euler XZY FD[" << j << "]: " << std::endl
                  << fd << std::endl;
        std::cout << "Diff: " << std::endl << grad - fd << std::endl;
        EXPECT_TRUE(equals(grad, fd, 1e-7));
        return;
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_XZY_SECOND_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        Eigen::MatrixXs grad = math::eulerXZYToMatrixSecondGrad(rand, j, k);
        Eigen::MatrixXs fd
            = math::eulerXZYToMatrixSecondFiniteDifference(rand, j, k);
        if (!equals(grad, fd, 1e-7))
        {
          std::cout << "Euler XZY Grad[" << j << "," << k << "]: " << std::endl
                    << grad << std::endl;
          std::cout << "Euler XZY FD[" << j << "," << k << "]: " << std::endl
                    << fd << std::endl;
          std::cout << "Diff: " << std::endl << grad - fd << std::endl;
          EXPECT_TRUE(equals(grad, fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_ZYX_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      Eigen::MatrixXs grad = math::eulerZYXToMatrixGrad(rand, j);
      Eigen::MatrixXs fd = math::eulerZYXToMatrixFiniteDifference(rand, j);
      if (!equals(grad, fd, 1e-7))
      {
        std::cout << "Euler ZYX Grad[" << j << "]: " << std::endl
                  << grad << std::endl;
        std::cout << "Euler ZYX FD[" << j << "]: " << std::endl
                  << fd << std::endl;
        std::cout << "Diff: " << std::endl << grad - fd << std::endl;
        EXPECT_TRUE(equals(grad, fd, 1e-7));
        return;
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_ZYX_SECOND_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        Eigen::MatrixXs grad = math::eulerZYXToMatrixSecondGrad(rand, j, k);
        Eigen::MatrixXs fd
            = math::eulerZYXToMatrixSecondFiniteDifference(rand, j, k);
        if (!equals(grad, fd, 1e-7))
        {
          std::cout << "Euler ZYX Grad[" << j << "," << k << "]: " << std::endl
                    << grad << std::endl;
          std::cout << "Euler ZYX FD[" << j << "," << k << "]: " << std::endl
                    << fd << std::endl;
          std::cout << "Diff: " << std::endl << grad - fd << std::endl;
          EXPECT_TRUE(equals(grad, fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_ZXY_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      Eigen::MatrixXs grad = math::eulerZXYToMatrixGrad(rand, j);
      Eigen::MatrixXs fd = math::eulerZXYToMatrixFiniteDifference(rand, j);
      if (!equals(grad, fd, 1e-7))
      {
        std::cout << "Euler ZXY Grad[" << j << "]: " << std::endl
                  << grad << std::endl;
        std::cout << "Euler ZXY FD[" << j << "]: " << std::endl
                  << fd << std::endl;
        std::cout << "Diff: " << std::endl << grad - fd << std::endl;
        EXPECT_TRUE(equals(grad, fd, 1e-7));
        return;
      }
    }
  }
}

//==============================================================================
TEST(Geometry, EULER_ZXY_SECOND_GRAD)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s rand = Eigen::Vector3s::Random();

    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        Eigen::MatrixXs grad = math::eulerZXYToMatrixSecondGrad(rand, j, k);
        Eigen::MatrixXs fd
            = math::eulerZXYToMatrixSecondFiniteDifference(rand, j, k);
        if (!equals(grad, fd, 1e-7))
        {
          std::cout << "Euler ZXY Grad[" << j << "," << k << "]: " << std::endl
                    << grad << std::endl;
          std::cout << "Euler ZXY FD[" << j << "," << k << "]: " << std::endl
                    << fd << std::endl;
          std::cout << "Diff: " << std::endl << grad - fd << std::endl;
          EXPECT_TRUE(equals(grad, fd, 1e-7));
          return;
        }
      }
    }
  }
}

//==============================================================================
TEST(EulerFreeJoint, Construct)
{
  // Create single-body skeleton with a screw joint
  auto skelA = dynamics::Skeleton::create();
  auto pair
      = skelA->createJointAndBodyNodePair<dart::dynamics::EulerFreeJoint>();
  auto freeJoint = pair.first;
  auto bodyA = pair.second;

  auto skelB = dynamics::Skeleton::create();
  auto transPair
      = skelB->createJointAndBodyNodePair<dart::dynamics::TranslationalJoint>();
  auto transBody = transPair.second;
  auto eulerPair
      = transBody
            ->createChildJointAndBodyNodePair<dart::dynamics::EulerJoint>();
  auto euler = eulerPair.first;
  euler->setAxisOrder(dynamics::EulerJoint::AxisOrder::XYZ);
  auto bodyB = eulerPair.second;

  // Set a child transform

  Eigen::Isometry3s childToEuler = Eigen::Isometry3s::Identity();
  childToEuler.linear() = math::eulerXYZToMatrix(Eigen::Vector3s::Random());
  childToEuler.translation() = Eigen::Vector3s::Random();
  freeJoint->setTransformFromChildBodyNode(childToEuler);
  euler->setTransformFromChildBodyNode(childToEuler);

  // Do a bunch of randomized trials
  for (int i = 0; i < 100; i++)
  {
    Eigen::Vector3s eulerPos = Eigen::Vector3s::Random();
    Eigen::Vector3s transPos = Eigen::Vector3s::Random();
    Eigen::Vector3s eulerVel = Eigen::Vector3s::Random();
    Eigen::Vector3s transVel = Eigen::Vector3s::Random();
    Eigen::Vector3s eulerAcc = Eigen::Vector3s::Random();
    Eigen::Vector3s transAcc = Eigen::Vector3s::Random();
    /*
    if (i < 20)
    {
      transPos.setZero();
      transVel.setZero();
    }
    */

    Eigen::Vector6s skelAPos;
    skelAPos.head<3>() = eulerPos;
    skelAPos.tail<3>() = transPos;
    Eigen::Vector6s skelAVel;
    skelAVel.head<3>() = eulerVel;
    skelAVel.tail<3>() = transVel;
    Eigen::Vector6s skelAAcc;
    skelAAcc.head<3>() = eulerAcc;
    skelAAcc.tail<3>() = transAcc;

    Eigen::Vector6s skelBPos;
    skelBPos.head<3>() = transPos;
    skelBPos.tail<3>() = eulerPos;
    Eigen::Vector6s skelBVel;
    skelBVel.head<3>() = transVel;
    skelBVel.tail<3>() = eulerVel;
    Eigen::Vector6s skelBAcc;
    skelBAcc.head<3>() = transAcc;
    skelBAcc.tail<3>() = eulerAcc;

    skelA->setPositions(skelAPos);
    skelA->setVelocities(skelAVel);
    skelA->setAccelerations(skelAAcc);

    skelB->setPositions(skelBPos);
    skelB->setVelocities(skelBVel);
    skelB->setAccelerations(skelBAcc);

    // Verify updateRelativeTransform()

    Eigen::Matrix4s posA = bodyA->getWorldTransform().matrix();
    Eigen::Matrix4s posB = bodyB->getWorldTransform().matrix();
    if (!equals(posA, posB, 1e-8))
    {
      std::cout << "Testing euler positions: " << eulerPos << std::endl;
      std::cout << "Testing euler velocities: " << eulerVel << std::endl;
      std::cout << "Testing trans positions: " << transPos << std::endl;
      std::cout << "Testing trans velocities: " << transVel << std::endl;

      std::cout << "Pos A: " << std::endl << posA << std::endl;
      std::cout << "Pos B: " << std::endl << posB << std::endl;
      std::cout << "Diff: " << std::endl << posA - posB << std::endl;
      EXPECT_TRUE(equals(posA, posB, 1e-8));
      return;
    }

    // Verify updateRelativeJacobian()

    Eigen::Vector6s velA = bodyA->getSpatialVelocity();
    Eigen::Vector6s velB = bodyB->getSpatialVelocity();
    if (!equals(velA, velB, 1e-8))
    {
      std::cout << "Testing euler positions: " << eulerPos << std::endl;
      std::cout << "Testing euler velocities: " << eulerVel << std::endl;
      std::cout << "Testing trans positions: " << transPos << std::endl;
      std::cout << "Testing trans velocities: " << transVel << std::endl;

      std::cout << "Vel A: " << std::endl << velA << std::endl;
      std::cout << "Vel B: " << std::endl << velB << std::endl;
      std::cout << "Diff: " << std::endl << velA - velB << std::endl;
      EXPECT_TRUE(equals(velA, velB, 1e-8));
      return;
    }

    // Directly verify updateRelativeJacobianTimeDeriv()

    Eigen::Matrix6s dJ = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
        skelAPos, skelAVel, EulerJoint::AxisOrder::XYZ, childToEuler);
    Eigen::Matrix6s dJ_fd
        = EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
            skelAPos, skelAVel, EulerJoint::AxisOrder::XYZ, childToEuler);
    if (!equals(dJ, dJ_fd, 1e-7))
    {
      std::cout << "Testing euler positions: " << eulerPos << std::endl;
      std::cout << "Testing euler velocities: " << eulerVel << std::endl;
      std::cout << "Testing euler acc: " << eulerAcc << std::endl;
      std::cout << "Testing trans positions: " << transPos << std::endl;
      std::cout << "Testing trans velocities: " << transVel << std::endl;
      std::cout << "Testing trans acc: " << transAcc << std::endl;

      std::cout << "Analytical dJ: " << std::endl << dJ << std::endl;
      std::cout << "FD dJ: " << std::endl << dJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << dJ - dJ_fd << std::endl;
      EXPECT_TRUE(equals(dJ, dJ_fd, 1e-7));
      return;
    }

    // Indirectly verify updateRelativeJacobianTimeDeriv()

    Eigen::Vector6s accA = bodyA->getSpatialAcceleration();
    Eigen::Vector6s accB = bodyB->getSpatialAcceleration();
    if (!equals(accA, accB, 1e-8))
    {
      std::cout << "Testing euler positions: " << eulerPos << std::endl;
      std::cout << "Testing euler velocities: " << eulerVel << std::endl;
      std::cout << "Testing euler acc: " << eulerAcc << std::endl;
      std::cout << "Testing trans positions: " << transPos << std::endl;
      std::cout << "Testing trans velocities: " << transVel << std::endl;
      std::cout << "Testing trans acc: " << transAcc << std::endl;

      std::cout << "Acc A: " << std::endl << accA << std::endl;
      std::cout << "Acc B: " << std::endl << accB << std::endl;
      std::cout << "Diff: " << std::endl << accA - accB << std::endl;
      EXPECT_TRUE(equals(accA, accB, 1e-8));
      return;
    }

    // Test all the spatial (6-dof euler + translation) derivatives of Jacobians
    for (int j = 0; j < 6; j++)
    {
      Eigen::Matrix6s dpos_J
          = EulerFreeJoint::computeRelativeJacobianStaticDerivWrtPos(
              skelAPos, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      Eigen::Matrix6s dpos_J_fd
          = EulerFreeJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
              skelAPos, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      if (!equals(dpos_J, dpos_J_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "Wrt position: " << j << std::endl;
        std::cout << "Analytical d_J: " << std::endl << dpos_J << std::endl;
        std::cout << "FD d_J: " << std::endl << dpos_J_fd << std::endl;
        std::cout << "Diff: " << std::endl << dpos_J - dpos_J_fd << std::endl;
        EXPECT_TRUE(equals(dpos_J, dpos_J_fd, 1e-7));
        return;
      }

      Eigen::Matrix6s dpos_dJ
          = EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
              skelAPos, skelAVel, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      Eigen::Matrix6s dpos_dJ_fd = EulerFreeJoint::
          finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
              skelAPos, skelAVel, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      if (!equals(dpos_dJ, dpos_dJ_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "Wrt position: " << j << std::endl;
        std::cout << "Analytical d_dJ: " << std::endl << dpos_dJ << std::endl;
        std::cout << "FD d_dJ: " << std::endl << dpos_dJ_fd << std::endl;
        std::cout << "Diff: " << std::endl << dpos_dJ - dpos_dJ_fd << std::endl;
        EXPECT_TRUE(equals(dpos_dJ, dpos_dJ_fd, 1e-7));
        return;
      }

      Eigen::Matrix6s dvel_dJ
          = EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
              skelAPos, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      Eigen::Matrix6s dvel_dJ_fd = EulerFreeJoint::
          finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
              skelAPos, skelAVel, j, EulerJoint::AxisOrder::XYZ, childToEuler);
      if (!equals(dvel_dJ, dvel_dJ_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "Wrt velocity: " << j << std::endl;
        std::cout << "Analytical d_dJ: " << std::endl << dvel_dJ << std::endl;
        std::cout << "FD d_dJ: " << std::endl << dvel_dJ_fd << std::endl;
        std::cout << "Diff: " << std::endl << dvel_dJ - dvel_dJ_fd << std::endl;
        EXPECT_TRUE(equals(dvel_dJ, dvel_dJ_fd, 1e-7));
        return;
      }
    }
  }
}
