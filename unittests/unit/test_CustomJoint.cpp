#include <gtest/gtest.h>

#include "dart/dart.hpp"

#include "TestHelpers.hpp"

using namespace dart;

//==============================================================================
TEST(CustomJoint, Construct)
{
  // Create single-body skeleton with a screw joint
  auto skelA = dynamics::Skeleton::create();
  auto pair = skelA->createJointAndBodyNodePair<dart::dynamics::CustomJoint>();
  auto custom = pair.first;
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
  custom->setTransformFromChildBodyNode(childToEuler);
  euler->setTransformFromChildBodyNode(childToEuler);

  // Do a bunch of randomized trials
  std::vector<Eigen::Vector3s> flips;
  flips.push_back(Eigen::Vector3s::Ones());
  flips.push_back(Eigen::Vector3s(-1.0, 1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, -1.0, 1.0));
  flips.push_back(Eigen::Vector3s(1.0, 1.0, -1.0));
  flips.push_back(Eigen::Vector3s::Ones() * -1);

  for (Eigen::Vector3s& flip : flips) {
    std::cout << "Testing flip " << flip << std::endl;
    custom->setFlipAxisMap(flip);
    euler->setFlipAxisMap(flip);
    for (int i = 0; i < 20; i++)
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

      for (int j = 0; j < 6; j++)
      {
        custom->setCustomFunction(
            j,
            std::make_shared<math::TestBedFunction>(
                skelAPos(j), skelAVel(j), skelAAcc(j)));
      }
      skelA->setPositions(Eigen::VectorXs::Zero(1));
      skelA->setVelocities(Eigen::VectorXs::Ones(1));
      skelA->setAccelerations(Eigen::VectorXs::Zero(1));

      ////////////////////////////////////////////////////////////////////////////////
      // Check custom function mappings and various derivatives
      ////////////////////////////////////////////////////////////////////////////////

      EXPECT_TRUE(
          equals(custom->getCustomFunctionPositions(0.0), skelAPos, 1e-12));
      EXPECT_TRUE(
          equals(custom->getCustomFunctionVelocities(0.0, 1.0), skelAVel, 1e-12));

      Eigen::Vector6s customAcc
          = custom->getCustomFunctionAccelerations(0.0, 1.0, 0.0);
      if (!equals(customAcc, skelAAcc, 1e-12))
      {
        std::cout << "Custom Acc: " << std::endl << customAcc << std::endl;
        std::cout << "Acc A: " << std::endl << skelAAcc << std::endl;
        std::cout << "Diff: " << std::endl << customAcc - skelAAcc << std::endl;
        EXPECT_TRUE(equals(customAcc, skelAAcc, 1e-12));
        return;
      }

      Eigen::Vector6s customV_dp
          = custom->getCustomFunctionVelocitiesDerivativeWrtPos(0.0, 1.0);
      Eigen::Vector6s customV_dp_fd
          = custom->finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
              0.0, 1.0);
      if (!equals(customV_dp, customV_dp_fd, 1e-9))
      {
        std::cout << "Custom dV/dp: " << std::endl << customV_dp << std::endl;
        std::cout << "FD Custom dV/dp: " << std::endl
                  << customV_dp_fd << std::endl;
        std::cout << "Diff: " << std::endl
                  << customV_dp - customV_dp_fd << std::endl;
        EXPECT_TRUE(equals(customV_dp, customV_dp_fd, 1e-9));
        return;
      }

      Eigen::Vector6s customAcc_dp
          = custom->getCustomFunctionAccelerationsDerivativeWrtPos(0.0, 1.0, 0.0);
      Eigen::Vector6s customAcc_dp_fd
          = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
              0.0, 1.0, 0.0);
      if (!equals(customAcc_dp, customAcc_dp_fd, 1e-12))
      {
        std::cout << "Custom dAcc/dp: " << std::endl << customAcc_dp << std::endl;
        std::cout << "FD Custom dAcc/dp: " << std::endl
                  << customAcc_dp_fd << std::endl;
        std::cout << "Diff: " << std::endl
                  << customAcc_dp - customAcc_dp_fd << std::endl;
        EXPECT_TRUE(equals(customAcc_dp, customAcc_dp_fd, 1e-12));
        return;
      }

      Eigen::Vector6s customAcc_dv
          = custom->getCustomFunctionAccelerationsDerivativeWrtVel(0.0);
      Eigen::Vector6s customAcc_dv_fd
          = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
              0.0, 1.0, 0.0);
      if (!equals(customAcc_dv, customAcc_dv_fd, 1e-9))
      {
        std::cout << "Custom dAcc/dv: " << std::endl << customAcc_dv << std::endl;
        std::cout << "FD Custom dAcc/dv: " << std::endl
                  << customAcc_dv_fd << std::endl;
        std::cout << "Diff: " << std::endl
                  << customAcc_dv - customAcc_dv_fd << std::endl;
        EXPECT_TRUE(equals(customAcc_dv, customAcc_dv_fd, 1e-9));
        return;
      }

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

      Eigen::Matrix6s dsJ = custom->getSpatialJacobianStaticDerivWrtInput(0);
      Eigen::Matrix6s dsJ_fd
          = custom->finiteDifferenceSpatialJacobianStaticDerivWrtInput(0);

      if (!equals(dsJ, dsJ_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "getSpatialJacobianDerivWrtInput(): " << std::endl;
        std::cout << "Analytical dsJ: " << std::endl << dsJ << std::endl;
        std::cout << "FD dsJ: " << std::endl << dsJ_fd << std::endl;
        std::cout << "Diff: " << std::endl << dsJ - dsJ_fd << std::endl;
        EXPECT_TRUE(equals(dsJ, dsJ_fd, 1e-7));
        return;
      }

      Eigen::Vector6s dj = custom->getRelativeJacobianDeriv(0);
      Eigen::Vector6s dj_fd = custom->finiteDifferenceRelativeJacobianDeriv(0);

      if (!equals(dj, dj_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "relativeJacobianDeriv(): " << std::endl;
        std::cout << "Analytical dj: " << std::endl << dj << std::endl;
        std::cout << "FD dj: " << std::endl << dj_fd << std::endl;
        std::cout << "Diff: " << std::endl << dj - dj_fd << std::endl;
        EXPECT_TRUE(equals(dj, dj_fd, 1e-7));
        return;
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Check d/dt d/dx of relative Jacobians
      ////////////////////////////////////////////////////////////////////////////////

      Eigen::Vector6s dj_dt_dp
          = custom->getRelativeJacobianTimeDerivDerivWrtPosition(0);
      Eigen::Vector6s dj_dt_dp_fd
          = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(0);

      if (!equals(dj_dt_dp, dj_dt_dp_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "getRelativeJacobianTimeDerivDerivWrtPosition(): "
                  << std::endl;
        std::cout << "Analytical dj dt dp: " << std::endl
                  << dj_dt_dp << std::endl;
        std::cout << "FD dj dt dp: " << std::endl << dj_dt_dp_fd << std::endl;
        std::cout << "Diff: " << std::endl << dj_dt_dp - dj_dt_dp_fd << std::endl;
        EXPECT_TRUE(equals(dj_dt_dp, dj_dt_dp_fd, 1e-7));
        return;
      }

      Eigen::Vector6s dj_dt_dv
          = custom->getRelativeJacobianTimeDerivDerivWrtVelocity(0);
      Eigen::Vector6s dj_dt_dv_fd
          = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(0);

      if (!equals(dj_dt_dv, dj_dt_dv_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "getRelativeJacobianTimeDerivDerivWrtVelocity(): "
                  << std::endl;
        std::cout << "Analytical dj dt dv: " << std::endl
                  << dj_dt_dv << std::endl;
        std::cout << "FD dj dt dv: " << std::endl << dj_dt_dv_fd << std::endl;
        std::cout << "Diff: " << std::endl << dj_dt_dv - dj_dt_dv_fd << std::endl;
        EXPECT_TRUE(equals(dj_dt_dv, dj_dt_dv_fd, 1e-7));
        return;
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Check d/dt d/dx of spatial Jacobians
      ////////////////////////////////////////////////////////////////////////////////

      Eigen::Matrix6s j3
          = custom->getSpatialJacobianTimeDerivDerivWrtInputPos(0.0, 1.0);
      Eigen::Matrix6s j3_fd
          = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
              0.0, 1.0);
      if (!equals(j3, j3_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "getSpatialJacobianTimeDerivDerivWrtInput(): " << std::endl;
        std::cout << "Analytical dj dt dx: " << std::endl << j3 << std::endl;
        std::cout << "FD dj dt dx: " << std::endl << j3_fd << std::endl;
        std::cout << "Diff: " << std::endl << j3 - j3_fd << std::endl;
        EXPECT_TRUE(equals(j3, j3_fd, 1e-7));
        return;
      }

      Eigen::Matrix6s j4
          = custom->getSpatialJacobianTimeDerivDerivWrtInputVel(0.0);
      Eigen::Matrix6s j4_fd
          = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
              0.0, 1.0);
      if (!equals(j4, j4_fd, 1e-7))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing euler acc: " << eulerAcc << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;
        std::cout << "Testing trans acc: " << transAcc << std::endl;

        std::cout << "getSpatialJacobianTimeDerivDerivWrtInputVel(): "
                  << std::endl;
        std::cout << "Analytical dj dt dx: " << std::endl << j4 << std::endl;
        std::cout << "FD dj dt dx: " << std::endl << j4_fd << std::endl;
        std::cout << "Diff: " << std::endl << j4 - j4_fd << std::endl;
        EXPECT_TRUE(equals(j4, j4_fd, 1e-7));
        return;
      }

      Eigen::Vector6s scratch = custom->scratchAnalytical();
      Eigen::Vector6s scratch_fd = custom->scratchFd();
      if (!equals(scratch, scratch_fd, 1e-8))
      {
        std::cout << "Scratch: " << std::endl << scratch << std::endl;
        std::cout << "FD Scratch: " << std::endl << scratch_fd << std::endl;
        std::cout << "Diff: " << std::endl << scratch - scratch_fd << std::endl;
        EXPECT_TRUE(equals(scratch, scratch_fd, 1e-8));
        return;
      }
    }
  }
}
