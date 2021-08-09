#include <gtest/gtest.h>

#include "dart/dart.hpp"

#include "TestHelpers.hpp"

using namespace dart;

// #define ALL_TESTS

//==============================================================================
bool verifyCustomJoint(CustomJoint* custom, s_t TEST_THRESHOLD) {
  s_t pos = custom->getPosition(0);
  s_t vel = custom->getVelocity(0);

  Eigen::Vector6s customP_dp
      = custom->getCustomFunctionGradientAt(pos);
  Eigen::Vector6s customP_dp_fd
      = custom->finiteDifferenceCustomFunctionGradientAt(
          pos);
  if (!equals(customP_dp, customP_dp_fd, TEST_THRESHOLD))
  {
    std::cout << "Custom dP/dx: " << std::endl << customP_dp << std::endl;
    std::cout << "FD Custom dP/dx: " << std::endl
              << customP_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customP_dp - customP_dp_fd << std::endl;
    EXPECT_TRUE(equals(customP_dp, customP_dp_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Vector6s customV_dp
      = custom->getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel);
  Eigen::Vector6s customV_dp_fd
      = custom->finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
          pos, vel);
  if (!equals(customV_dp, customV_dp_fd, TEST_THRESHOLD))
  {
    std::cout << "Custom dV/dp: " << std::endl << customV_dp << std::endl;
    std::cout << "FD Custom dV/dp: " << std::endl
              << customV_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customV_dp - customV_dp_fd << std::endl;
    EXPECT_TRUE(equals(customV_dp, customV_dp_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Vector6s customAcc_dp
      = custom->getCustomFunctionAccelerationsDerivativeWrtPos(pos, vel, 0.0);
  Eigen::Vector6s customAcc_dp_fd
      = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
          pos, vel, 0.0);
  if (!equals(customAcc_dp, customAcc_dp_fd, 1e-10))
  {
    std::cout << "Custom dAcc/dp: " << std::endl << customAcc_dp << std::endl;
    std::cout << "FD Custom dAcc/dp: " << std::endl
              << customAcc_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customAcc_dp - customAcc_dp_fd << std::endl;
    EXPECT_TRUE(equals(customAcc_dp, customAcc_dp_fd, 1e-10));
    return false;
  }

  Eigen::Vector6s customAcc_dv
      = custom->getCustomFunctionAccelerationsDerivativeWrtVel(pos);
  Eigen::Vector6s customAcc_dv_fd
      = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
          pos, vel, 0.0);
  if (!equals(customAcc_dv, customAcc_dv_fd, TEST_THRESHOLD))
  {
    std::cout << "Custom dAcc/dv: " << std::endl << customAcc_dv << std::endl;
    std::cout << "FD Custom dAcc/dv: " << std::endl
              << customAcc_dv_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customAcc_dv - customAcc_dv_fd << std::endl;
    EXPECT_TRUE(equals(customAcc_dv, customAcc_dv_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Matrix6s dsJ = custom->getSpatialJacobianStaticDerivWrtInput(0);
  Eigen::Matrix6s dsJ_fd
      = custom->finiteDifferenceSpatialJacobianStaticDerivWrtInput(0);

  if (!equals(dsJ, dsJ_fd, TEST_THRESHOLD))
  {
    std::cout << "getSpatialJacobianDerivWrtInput(): " << std::endl;
    std::cout << "Analytical dsJ: " << std::endl << dsJ << std::endl;
    std::cout << "FD dsJ: " << std::endl << dsJ_fd << std::endl;
    std::cout << "Diff: " << std::endl << dsJ - dsJ_fd << std::endl;
    EXPECT_TRUE(equals(dsJ, dsJ_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Vector6s dj = custom->getRelativeJacobianDeriv(0);
  Eigen::Vector6s dj_fd = custom->finiteDifferenceRelativeJacobianDeriv(0);

  if (!equals(dj, dj_fd, TEST_THRESHOLD))
  {
    std::cout << "relativeJacobianDeriv(): " << std::endl;
    std::cout << "Analytical dj: " << std::endl << dj << std::endl;
    std::cout << "FD dj: " << std::endl << dj_fd << std::endl;
    std::cout << "Diff: " << std::endl << dj - dj_fd << std::endl;
    EXPECT_TRUE(equals(dj, dj_fd, TEST_THRESHOLD));
    return false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Check d/dt d/dx of relative Jacobians
  ////////////////////////////////////////////////////////////////////////////////

  Eigen::Vector6s dj_dt_dp
      = custom->getRelativeJacobianTimeDerivDerivWrtPosition(0);
  Eigen::Vector6s dj_dt_dp_fd
      = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(0);

  if (!equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD))
  {
    std::cout << "getRelativeJacobianTimeDerivDerivWrtPosition(): "
              << std::endl;
    std::cout << "Analytical dj dt dp: " << std::endl
              << dj_dt_dp << std::endl;
    std::cout << "FD dj dt dp: " << std::endl << dj_dt_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl << dj_dt_dp - dj_dt_dp_fd << std::endl;
    EXPECT_TRUE(equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Vector6s dj_dt_dv
      = custom->getRelativeJacobianTimeDerivDerivWrtVelocity(0);
  Eigen::Vector6s dj_dt_dv_fd
      = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(0);

  if (!equals(dj_dt_dv, dj_dt_dv_fd, TEST_THRESHOLD))
  {
    std::cout << "getRelativeJacobianTimeDerivDerivWrtVelocity(): "
              << std::endl;
    std::cout << "Analytical dj dt dv: " << std::endl
              << dj_dt_dv << std::endl;
    std::cout << "FD dj dt dv: " << std::endl << dj_dt_dv_fd << std::endl;
    std::cout << "Diff: " << std::endl << dj_dt_dv - dj_dt_dv_fd << std::endl;
    EXPECT_TRUE(equals(dj_dt_dv, dj_dt_dv_fd, TEST_THRESHOLD));
    return false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Check d/dt d/dx of spatial Jacobians
  ////////////////////////////////////////////////////////////////////////////////

  Eigen::Matrix6s j3
      = custom->getSpatialJacobianTimeDerivDerivWrtInputPos(0.0, 1.0);
  Eigen::Matrix6s j3_fd
      = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
          0.0, 1.0);
  if (!equals(j3, j3_fd, TEST_THRESHOLD))
  {
    std::cout << "getSpatialJacobianTimeDerivDerivWrtInput(): " << std::endl;
    std::cout << "Analytical dj dt dx: " << std::endl << j3 << std::endl;
    std::cout << "FD dj dt dx: " << std::endl << j3_fd << std::endl;
    std::cout << "Diff: " << std::endl << j3 - j3_fd << std::endl;
    EXPECT_TRUE(equals(j3, j3_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Matrix6s j4
      = custom->getSpatialJacobianTimeDerivDerivWrtInputVel(0.0);
  Eigen::Matrix6s j4_fd
      = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
          0.0, 1.0);
  if (!equals(j4, j4_fd, TEST_THRESHOLD))
  {
    std::cout << "getSpatialJacobianTimeDerivDerivWrtInputVel(): "
              << std::endl;
    std::cout << "Analytical dj dt dx: " << std::endl << j4 << std::endl;
    std::cout << "FD dj dt dx: " << std::endl << j4_fd << std::endl;
    std::cout << "Diff: " << std::endl << j4 - j4_fd << std::endl;
    EXPECT_TRUE(equals(j4, j4_fd, TEST_THRESHOLD));
    return false;
  }

  Eigen::Vector6s scratch = custom->scratchAnalytical();
  Eigen::Vector6s scratch_fd = custom->scratchFd();
  if (!equals(scratch, scratch_fd, 1e-8))
  {
    std::cout << "Scratch: " << std::endl << scratch << std::endl;
    std::cout << "FD Scratch: " << std::endl << scratch_fd << std::endl;
    std::cout << "Diff: " << std::endl << scratch - scratch_fd << std::endl;
    EXPECT_TRUE(equals(scratch, scratch_fd, 1e-8));
    return false;
  }
  return true;
}

//==============================================================================
#ifdef ALL_TESTS
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

      s_t TEST_THRESHOLD = 1e-9;

      skelB->setPositions(skelBPos);
      skelB->setVelocities(skelBVel);
      skelB->setAccelerations(skelBAcc);

      // Verify updateRelativeTransform()

      Eigen::Matrix4s posA = bodyA->getWorldTransform().matrix();
      Eigen::Matrix4s posB = bodyB->getWorldTransform().matrix();
      if (!equals(posA, posB, TEST_THRESHOLD))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;

        std::cout << "Pos A: " << std::endl << posA << std::endl;
        std::cout << "Pos B: " << std::endl << posB << std::endl;
        std::cout << "Diff: " << std::endl << posA - posB << std::endl;
        EXPECT_TRUE(equals(posA, posB, TEST_THRESHOLD));
        return;
      }

      // Verify updateRelativeJacobian()

      Eigen::Vector6s velA = bodyA->getSpatialVelocity();
      Eigen::Vector6s velB = bodyB->getSpatialVelocity();
      if (!equals(velA, velB, TEST_THRESHOLD))
      {
        std::cout << "Testing euler positions: " << eulerPos << std::endl;
        std::cout << "Testing euler velocities: " << eulerVel << std::endl;
        std::cout << "Testing trans positions: " << transPos << std::endl;
        std::cout << "Testing trans velocities: " << transVel << std::endl;

        std::cout << "Vel A: " << std::endl << velA << std::endl;
        std::cout << "Vel B: " << std::endl << velB << std::endl;
        std::cout << "Diff: " << std::endl << velA - velB << std::endl;
        EXPECT_TRUE(equals(velA, velB, TEST_THRESHOLD));
        return;
      }

      // Indirectly verify updateRelativeJacobianTimeDeriv()

      Eigen::Vector6s accA = bodyA->getSpatialAcceleration();
      Eigen::Vector6s accB = bodyB->getSpatialAcceleration();
      if (!equals(accA, accB, TEST_THRESHOLD))
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
        EXPECT_TRUE(equals(accA, accB, TEST_THRESHOLD));
        return;
      }

      if (!verifyCustomJoint(custom, TEST_THRESHOLD)) {return;}
    }
  }
}
#endif

//==============================================================================
TEST(CustomJoint, OpenSim_Knee)
{
  CustomJoint::Properties props;
  CustomJoint joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);
  std::shared_ptr<math::LinearFunction> linear = std::make_shared<math::LinearFunction>(1.0, 0.0);
  joint.setCustomFunction(0, linear);

  std::vector<s_t> rotZ_x;
  rotZ_x.push_back(0);
  rotZ_x.push_back(0.174533);
  rotZ_x.push_back(0.349066);
  rotZ_x.push_back(0.523599);
  rotZ_x.push_back(0.698132);
  rotZ_x.push_back(0.872665);
  rotZ_x.push_back(1.0472);
  rotZ_x.push_back(1.22173);
  rotZ_x.push_back(1.39626);
  rotZ_x.push_back(1.5708);
  rotZ_x.push_back(1.74533);
  rotZ_x.push_back(1.91986);
  rotZ_x.push_back(2.0944);
  std::vector<s_t> rotZ_y;
  rotZ_y.push_back(0.0);
  rotZ_y.push_back(0.0126809);
  rotZ_y.push_back(0.0226969);
  rotZ_y.push_back(0.0296054);
  rotZ_y.push_back(0.0332049);
  rotZ_y.push_back(0.0335354);
  rotZ_y.push_back(0.0308779);
  rotZ_y.push_back(0.0257548);
  rotZ_y.push_back(0.0189295);
  rotZ_y.push_back(0.011407);
  rotZ_y.push_back(0.00443314);
  rotZ_y.push_back(-0.00050475);
  rotZ_y.push_back(-0.0016782);
  std::shared_ptr<math::SimmSpline> rotZ = std::make_shared<math::SimmSpline>(rotZ_x, rotZ_y);
  joint.setCustomFunction(1, rotZ);

  std::vector<s_t> rotY_x;
  rotY_x.push_back(0.0);
  rotY_x.push_back(0.174533);
  rotY_x.push_back(0.349066);
  rotY_x.push_back(0.523599);
  rotY_x.push_back(0.698132);
  rotY_x.push_back(0.872665);
  rotY_x.push_back(1.0472);
  rotY_x.push_back(1.22173);
  rotY_x.push_back(1.39626);
  rotY_x.push_back(1.5708);
  rotY_x.push_back(1.74533);
  rotY_x.push_back(1.91986);
  rotY_x.push_back(2.0944);

  std::vector<s_t> rotY_y;
  rotY_y.push_back(0.0);
  rotY_y.push_back(0.059461);
  rotY_y.push_back(0.109399);
  rotY_y.push_back(0.150618);
  rotY_y.push_back(0.18392);
  rotY_y.push_back(0.210107);
  rotY_y.push_back(0.229983);
  rotY_y.push_back(0.24435);
  rotY_y.push_back(0.254012);
  rotY_y.push_back(0.25977);
  rotY_y.push_back(0.262428);
  rotY_y.push_back(0.262788);
  rotY_y.push_back(0.261654);

  std::shared_ptr<math::SimmSpline> rotY = std::make_shared<math::SimmSpline>(rotY_x, rotY_y);
  (void)rotY;
  joint.setCustomFunction(2, rotY);

  std::shared_ptr<math::LinearFunction> zero = std::make_shared<math::LinearFunction>(0.0, 0.0);
  joint.setCustomFunction(5, zero);
  /*
							<SpatialTransform>
								<!--3 Axes for rotations are listed first.-->
								<TransformAxis name="rotation1">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates>knee_angle_r</coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>1 0 0</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<LinearFunction>
											<coefficients> 1 0</coefficients>
										</LinearFunction>
									</function>
								</TransformAxis>
								<TransformAxis name="rotation2">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates>knee_angle_r</coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>0 0 1</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<SimmSpline>
											<x> 0 0.174533 0.349066 0.523599 0.698132 0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x>
											<y> 0 0.0126809 0.0226969 0.0296054 0.0332049 0.0335354 0.0308779 0.0257548 0.0189295 0.011407 0.00443314 -0.00050475 -0.0016782</y>
										</SimmSpline>
									</function>
								</TransformAxis>
								<TransformAxis name="rotation3">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates>knee_angle_r</coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>0 1 0</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<SimmSpline>
											<x> 0 0.174533 0.349066 0.523599 0.698132 0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x>
											<y> 0 0.059461 0.109399 0.150618 0.18392 0.210107 0.229983 0.24435 0.254012 0.25977 0.262428 0.262788 0.261654</y>
										</SimmSpline>
									</function>
								</TransformAxis>
								<!--3 Axes for translations are listed next.-->
								<TransformAxis name="translation1">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates>knee_angle_r</coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>0 1 0</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<SimmSpline>
											<x> 0 0.174533 0.349066 0.523599 0.698132 0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x>
											<y> 0 0.000479 0.000835 0.001086 0.001251 0.001346 0.001391 0.001403 0.0014 0.0014 0.001421 0.001481 0.001599</y>
										</SimmSpline>
									</function>
								</TransformAxis>
								<TransformAxis name="translation2">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates>knee_angle_r</coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>0 0 1</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<SimmSpline>
											<x> 0 0.174533 0.349066 0.523599 0.698132 0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x>
											<y> 0 0.000988 0.001899 0.002734 0.003492 0.004173 0.004777 0.005305 0.005756 0.00613 0.006427 0.006648 0.006792</y>
										</SimmSpline>
									</function>
								</TransformAxis>
								<TransformAxis name="translation3">
									<!--Names of the coordinates that serve as the independent variables         of the transform function.-->
									<coordinates></coordinates>
									<!--Rotation or translation axis for the transform.-->
									<axis>1 0 0</axis>
									<!--Transform function of the generalized coordinates used to        represent the amount of transformation along a specified axis.-->
									<function>
										<Constant>
											<value>0</value>
										</Constant>
									</function>
								</TransformAxis>
							</SpatialTransform>
  */
  for (int i = 0; i < 10; i++) {
    joint.setPositions(Eigen::Vector1s::Random());
    joint.setVelocities(Eigen::Vector1s::Random());
    // joint.setPosition(0, 0.396927);
    // joint.setVelocity(0, 0.535337);
    std::cout << "Testing (" << joint.getPosition(0) << "," << joint.getVelocity(0) << ")" << std::endl;
    if (!verifyCustomJoint(&joint, 1e-9)) { return; }
  }
}
