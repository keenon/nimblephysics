#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

#include "TestHelpers.hpp"

using namespace dart;

// #define ALL_TESTS

//==============================================================================
template <std::size_t Dimension>
bool verifyCustomJoint(CustomJoint<Dimension>* custom, s_t TEST_THRESHOLD)
{
  Eigen::VectorXs pos = custom->getPositions();
  Eigen::VectorXs vel = custom->getVelocities();

  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customP_dp
      = custom->getCustomFunctionGradientAt(pos);
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customP_dp_fd
      = custom->finiteDifferenceCustomFunctionGradientAt(pos);
  if (!equals(customP_dp, customP_dp_fd, 1e-10))
  {
    std::cout << "Custom dP/dx: " << std::endl << customP_dp << std::endl;
    std::cout << "FD Custom dP/dx: " << std::endl << customP_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customP_dp - customP_dp_fd << std::endl;
    EXPECT_TRUE(equals(customP_dp, customP_dp_fd, 1e-10));
    return false;
  }

  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customV_dp
      = custom->getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel);
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customV_dp_fd
      = custom->finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
          pos, vel);
  if (!equals(customV_dp, customV_dp_fd, TEST_THRESHOLD))
  {
    std::cout << "Custom dV/dp: " << std::endl << customV_dp << std::endl;
    std::cout << "FD Custom dV/dp: " << std::endl << customV_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customV_dp - customV_dp_fd << std::endl;
    EXPECT_TRUE(equals(customV_dp, customV_dp_fd, TEST_THRESHOLD));

    for (int i = 0; i < 6; i++)
    {
      std::cout << "Custom function " << i << " (driven by DOF "
                << custom->getCustomFunctionDrivenByDof(i) << "):" << std::endl;
      std::cout << "     ddx:"
                << custom->getCustomFunction(i)->calcDerivative(
                       2, pos(custom->getCustomFunctionDrivenByDof(i)))
                << std::endl;
      std::cout << "  fd_ddx:"
                << custom->getCustomFunction(i)->finiteDifferenceDerivative(
                       2, pos(custom->getCustomFunctionDrivenByDof(i)))
                << std::endl;
    }
    return false;
  }

  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customAcc_dp
      = custom->getCustomFunctionAccelerationsDerivativeWrtPos(
          pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()));
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customAcc_dp_fd
      = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
          pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()));
  if (!equals(customAcc_dp, customAcc_dp_fd, 1e-9))
  {
    std::cout << "Custom dAcc/dp: " << std::endl << customAcc_dp << std::endl;
    std::cout << "FD Custom dAcc/dp: " << std::endl
              << customAcc_dp_fd << std::endl;
    std::cout << "Diff: " << std::endl
              << customAcc_dp - customAcc_dp_fd << std::endl;
    EXPECT_TRUE(equals(customAcc_dp, customAcc_dp_fd, 3e-9));
    return false;
  }

  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customAcc_dv
      = custom->getCustomFunctionAccelerationsDerivativeWrtVel(pos);
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> customAcc_dv_fd
      = custom->finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
          pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()));
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

  math::Jacobian j = custom->getRelativeJacobian();
  math::Jacobian j_fd = custom->finiteDifferenceRelativeJacobian();

  if (!equals(j, j_fd, TEST_THRESHOLD))
  {
    std::cout << "relativeJacobian: " << std::endl;
    std::cout << "Analytical j: " << std::endl << j << std::endl;
    std::cout << "FD j: " << std::endl << j_fd << std::endl;
    std::cout << "Diff: " << std::endl << j - j_fd << std::endl;
    EXPECT_TRUE(equals(j, j_fd, TEST_THRESHOLD));
    return false;
  }

  math::Jacobian dc_dt = custom->getCustomFunctionGradientAtTimeDeriv(pos, vel);
  math::Jacobian dc_dt_fd
      = custom->finiteDifferenceCustomFunctionGradientAtTimeDeriv(pos, vel);

  if (!equals(dc_dt, dc_dt_fd, TEST_THRESHOLD))
  {
    std::cout << "customFunctionGradientAtTimeDeriv: " << std::endl;
    std::cout << "Analytical dc_dt: " << std::endl << dc_dt << std::endl;
    std::cout << "FD dc_dt: " << std::endl << dc_dt_fd << std::endl;
    std::cout << "Diff: " << std::endl << dc_dt - dc_dt_fd << std::endl;
    EXPECT_TRUE(equals(dc_dt, dc_dt_fd, TEST_THRESHOLD));
    return false;
  }

  math::Jacobian dj_dt = custom->getRelativeJacobianTimeDeriv();
  math::Jacobian dj_dt_fd = custom->finiteDifferenceRelativeJacobianTimeDeriv();

  if (!equals(dj_dt, dj_dt_fd, TEST_THRESHOLD))
  {
    std::cout << "relativeJacobianTimeDeriv: " << std::endl;
    std::cout << "Analytical dj_dt: " << std::endl << dj_dt << std::endl;
    std::cout << "FD dj_dt: " << std::endl << dj_dt_fd << std::endl;
    std::cout << "Diff: " << std::endl << dj_dt - dj_dt_fd << std::endl;
    EXPECT_TRUE(equals(dj_dt, dj_dt_fd, TEST_THRESHOLD));
    return false;
  }

  for (int i = 0; i < custom->getNumDofs(); i++)
  {
    Eigen::Matrix6s dsJ = custom->getSpatialJacobianStaticDerivWrtInput(pos, i);
    Eigen::Matrix6s dsJ_fd
        = custom->finiteDifferenceSpatialJacobianStaticDerivWrtInput(pos, i);

    if (!equals(dsJ, dsJ_fd, TEST_THRESHOLD))
    {

      std::cout << "getSpatialJacobianDerivWrtInput(index=" << i
                << "): " << std::endl;

      ////////////////////////////
      /*

      std::cout << "Gradient: " << std::endl
                << custom->getCustomFunctionGradientAt(pos) << std::endl;
      std::cout << "Gradient Diff: " << std::endl
                << custom->finiteDifferenceCustomFunctionGradientAt(pos)
                      - custom->getCustomFunctionGradientAt(pos)
                << std::endl;
      Eigen::Vector6s positions = custom->getCustomFunctionPositions(pos);
      Eigen::Matrix6s jac
          = EulerFreeJoint::computeRelativeJacobianStaticDerivWrtPos(
              positions,
              1,
              custom->getAxisOrder(),
              custom->getFlipAxisMap(),
              custom->getTransformFromChildBodyNode());
      Eigen::Matrix6s jacFD
          = EulerFreeJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
              positions,
              1,
              custom->getAxisOrder(),
              custom->getFlipAxisMap(),
              custom->getTransformFromChildBodyNode());
      std::cout << "Jac: " << std::endl << jac << std::endl;
      std::cout << "Jac Diff: " << std::endl << jac - jacFD << std::endl;

      */
      ////////////////////////////

      std::cout << "Analytical dsJ: " << std::endl << dsJ << std::endl;
      std::cout << "FD dsJ: " << std::endl << dsJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << dsJ - dsJ_fd << std::endl;
      EXPECT_TRUE(equals(dsJ, dsJ_fd, TEST_THRESHOLD));
      return false;
    }

    math::Jacobian dj = custom->getRelativeJacobianDeriv(i);
    math::Jacobian dj_fd = custom->finiteDifferenceRelativeJacobianDeriv(i);

    if (!equals(dj, dj_fd, TEST_THRESHOLD))
    {
      std::cout << "relativeJacobianDeriv(index=" << i << "): " << std::endl;
      std::cout << "Analytical dj: " << std::endl << dj << std::endl;
      std::cout << "FD dj: " << std::endl << dj_fd << std::endl;
      std::cout << "Diff: " << std::endl << dj - dj_fd << std::endl;
      EXPECT_TRUE(equals(dj, dj_fd, TEST_THRESHOLD));
      return false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Check d/dt d/dx of relative Jacobians
    ////////////////////////////////////////////////////////////////////////////////

    math::Jacobian dc_dt_dp
        = custom->getCustomFunctionGradientAtTimeDerivPosDeriv(
            pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()), i);
    math::Jacobian dc_dt_dp_fd
        = custom->finiteDifferenceCustomFunctionGradientAtTimeDerivPosDeriv(
            pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()), i);

    if (!equals(dc_dt_dp, dc_dt_dp_fd, TEST_THRESHOLD))
    {
      std::cout << "getCustomFunctionGradientAtTimeDerivPosDeriv: "
                << std::endl;
      std::cout << "Analytical dc_dt_dp: " << std::endl
                << dc_dt_dp << std::endl;
      std::cout << "FD dc_dt_dp: " << std::endl << dc_dt_dp_fd << std::endl;
      std::cout << "Diff: " << std::endl << dc_dt_dp - dc_dt_dp_fd << std::endl;
      EXPECT_TRUE(equals(dc_dt_dp, dc_dt_dp_fd, TEST_THRESHOLD));
      return false;
    }

    math::Jacobian dc_dt_dv
        = custom->getCustomFunctionGradientAtTimeDerivVelDeriv(
            pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()), i);
    math::Jacobian dc_dt_dv_fd
        = custom->finiteDifferenceCustomFunctionGradientAtTimeDerivVelDeriv(
            pos, vel, Eigen::VectorXs::Zero(custom->getNumDofs()), i);

    if (!equals(dc_dt_dv, dc_dt_dv_fd, TEST_THRESHOLD))
    {
      std::cout << "getCustomFunctionGradientAtTimeDerivVelDeriv: "
                << std::endl;
      std::cout << "Analytical dc_dt_dp: " << std::endl
                << dc_dt_dv << std::endl;
      std::cout << "FD dc_dt_dp: " << std::endl << dc_dt_dv_fd << std::endl;
      std::cout << "Diff: " << std::endl << dc_dt_dv - dc_dt_dv_fd << std::endl;
      EXPECT_TRUE(equals(dc_dt_dv, dc_dt_dv_fd, TEST_THRESHOLD));
      return false;
    }

    math::Jacobian dj_dt_dp
        = custom->getRelativeJacobianTimeDerivDerivWrtPosition(i);
    math::Jacobian dj_dt_dp_fd
        = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(i);

    if (!equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD))
    {
      std::cout << "getRelativeJacobianTimeDerivDerivWrtPosition(index=" << i
                << "): " << std::endl;
      std::cout << "Analytical dj dt dp: " << std::endl
                << dj_dt_dp << std::endl;
      std::cout << "FD dj dt dp: " << std::endl << dj_dt_dp_fd << std::endl;
      std::cout << "Diff: " << std::endl << dj_dt_dp - dj_dt_dp_fd << std::endl;
      EXPECT_TRUE(equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD));
      return false;
    }

    math::Jacobian dj_dt_dv
        = custom->getRelativeJacobianTimeDerivDerivWrtVelocity(i);
    math::Jacobian dj_dt_dv_fd
        = custom->finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(i);

    if (!equals(dj_dt_dv, dj_dt_dv_fd, TEST_THRESHOLD))
    {
      std::cout << "getRelativeJacobianTimeDerivDerivWrtVelocity(index=" << i
                << "): " << std::endl;
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
        = custom->getSpatialJacobianTimeDerivDerivWrtInputPos(pos, vel, i);
    Eigen::Matrix6s j3_fd
        = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
            pos, vel, i);
    if (!equals(j3, j3_fd, TEST_THRESHOLD))
    {
      std::cout << "getSpatialJacobianTimeDerivDerivWrtInput(index=" << i
                << "): " << std::endl;
      std::cout << "Analytical dj dt dx: " << std::endl << j3 << std::endl;
      std::cout << "FD dj dt dx: " << std::endl << j3_fd << std::endl;
      std::cout << "Diff: " << std::endl << j3 - j3_fd << std::endl;
      EXPECT_TRUE(equals(j3, j3_fd, TEST_THRESHOLD));
      return false;
    }

    Eigen::Matrix6s j4
        = custom->getSpatialJacobianTimeDerivDerivWrtInputVel(pos, i);
    Eigen::Matrix6s j4_fd
        = custom->finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
            pos, vel, i);
    if (!equals(j4, j4_fd, TEST_THRESHOLD))
    {
      std::cout << "getSpatialJacobianTimeDerivDerivWrtInputVel(index=" << i
                << "): " << std::endl;
      std::cout << "Analytical dj dt dx: " << std::endl << j4 << std::endl;
      std::cout << "FD dj dt dx: " << std::endl << j4_fd << std::endl;
      std::cout << "Diff: " << std::endl << j4 - j4_fd << std::endl;
      EXPECT_TRUE(equals(j4, j4_fd, TEST_THRESHOLD));
      return false;
    }
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

  for (Eigen::Vector3s& flip : flips)
  {
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
      EXPECT_TRUE(equals(
          custom->getCustomFunctionVelocities(0.0, 1.0), skelAVel, 1e-12));

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

      if (!verifyCustomJoint(custom, TEST_THRESHOLD))
      {
        return;
      }
    }
  }
}
#endif

//==============================================================================
TEST(CustomJoint, OpenSim_Knee)
{
  CustomJoint<1>::Properties props;
  CustomJoint<1> joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);
  std::shared_ptr<math::LinearFunction> linear
      = std::make_shared<math::LinearFunction>(1.0, 0.0);
  // joint.setCustomFunction(0, linear);

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
  std::shared_ptr<math::SimmSpline> rotZ
      = std::make_shared<math::SimmSpline>(rotZ_x, rotZ_y);
  (void)rotZ;
  joint.setCustomFunction(1, rotZ, 0);

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

  std::shared_ptr<math::SimmSpline> rotY
      = std::make_shared<math::SimmSpline>(rotY_x, rotY_y);
  (void)rotY;
  joint.setCustomFunction(2, rotY, 0);

  std::shared_ptr<math::ConstantFunction> zero
      = std::make_shared<math::ConstantFunction>(0.0);
  (void)zero;
  joint.setCustomFunction(5, zero, 0);
  /*
              <SpatialTransform>
                <!--3 Axes for rotations are listed first.-->
                <TransformAxis name="rotation1">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.-->
                  <coordinates>knee_angle_r</coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>1 0 0</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <LinearFunction>
                      <coefficients> 1 0</coefficients>
                    </LinearFunction>
                  </function>
                </TransformAxis>
                <TransformAxis name="rotation2">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.-->
                  <coordinates>knee_angle_r</coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>0 0 1</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <SimmSpline>
                      <x> 0 0.174533 0.349066 0.523599 0.698132
     0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x> <y> 0
     0.0126809 0.0226969 0.0296054 0.0332049 0.0335354 0.0308779 0.0257548
     0.0189295 0.011407 0.00443314 -0.00050475 -0.0016782</y>
                    </SimmSpline>
                  </function>
                </TransformAxis>
                <TransformAxis name="rotation3">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.-->
                  <coordinates>knee_angle_r</coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>0 1 0</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <SimmSpline>
                      <x> 0 0.174533 0.349066 0.523599 0.698132
     0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x> <y> 0
     0.059461 0.109399 0.150618 0.18392 0.210107 0.229983 0.24435 0.254012
     0.25977 0.262428 0.262788 0.261654</y>
                    </SimmSpline>
                  </function>
                </TransformAxis>
                <!--3 Axes for translations are listed next.-->
                <TransformAxis name="translation1">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.-->
                  <coordinates>knee_angle_r</coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>0 1 0</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <SimmSpline>
                      <x> 0 0.174533 0.349066 0.523599 0.698132
     0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x> <y> 0
     0.000479 0.000835 0.001086 0.001251 0.001346 0.001391 0.001403 0.0014
     0.0014 0.001421 0.001481 0.001599</y>
                    </SimmSpline>
                  </function>
                </TransformAxis>
                <TransformAxis name="translation2">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.-->
                  <coordinates>knee_angle_r</coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>0 0 1</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <SimmSpline>
                      <x> 0 0.174533 0.349066 0.523599 0.698132
     0.872665 1.0472 1.22173 1.39626 1.5708 1.74533 1.91986 2.0944</x> <y> 0
     0.000988 0.001899 0.002734 0.003492 0.004173 0.004777 0.005305 0.005756
     0.00613 0.006427 0.006648 0.006792</y>
                    </SimmSpline>
                  </function>
                </TransformAxis>
                <TransformAxis name="translation3">
                  <!--Names of the coordinates that serve as the independent
     variables         of the transform function.--> <coordinates></coordinates>
                  <!--Rotation or translation axis for the transform.-->
                  <axis>1 0 0</axis>
                  <!--Transform function of the generalized coordinates used to
     represent the amount of transformation along a specified axis.-->
                  <function>
                    <Constant>
                      <value>0</value>
                    </Constant>
                  </function>
                </TransformAxis>
              </SpatialTransform>
  */
  for (int i = 0; i < 10; i++)
  {
    joint.setPositions(Eigen::Vector1s::Random());
    joint.setVelocities(Eigen::Vector1s::Random());

    /*
    joint.setPosition(0, 0.0);
    joint.setVelocity(0, 1.0);
    */
    std::cout << "Testing (" << joint.getPosition(0) << ","
              << joint.getVelocity(0) << ")" << std::endl;
    if (!verifyCustomJoint(&joint, 1e-9))
    {
      return;
    }
  }
}

//==============================================================================
TEST(CustomJoint, Polynomial)
{
  CustomJoint<1>::Properties props;
  CustomJoint<1> joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);

  // Make 6 random polynomials
  srand(42);
  for (int i = 0; i < 6; i++)
  {
    std::vector<s_t> coeffs;
    for (int j = 0; j < 10; j++)
    {
      coeffs.push_back((((double)rand() / RAND_MAX) * 0.1) - 0.05);
    }

    std::shared_ptr<PolynomialFunction> poly
        = std::make_shared<PolynomialFunction>(coeffs);
    joint.setCustomFunction(i, poly, 0);
  }

  // Check at random positions
  for (int i = 0; i < 10; i++)
  {
    joint.setPositions(Eigen::Vector1s::Random());
    joint.setVelocities(Eigen::Vector1s::Random());

    /*
    joint.setPosition(0, 0.0);
    joint.setVelocity(0, 1.0);
    */
    std::cout << "Testing (" << joint.getPosition(0) << ","
              << joint.getVelocity(0) << ")" << std::endl;
    if (!verifyCustomJoint(&joint, 1e-9))
    {
      return;
    }
  }
}

//==============================================================================
TEST(CustomJoint, Polynomial2)
{
  CustomJoint<2>::Properties props;
  CustomJoint<2> joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);

  // Make 6 random polynomials
  srand(42);
  for (int i = 0; i < 6; i++)
  {
    std::vector<s_t> coeffs;
    for (int j = 0; j < 10; j++)
    {
      coeffs.push_back((((double)rand() / RAND_MAX) * 0.1) - 0.05);
    }

    std::shared_ptr<PolynomialFunction> poly
        = std::make_shared<PolynomialFunction>(coeffs);
    joint.setCustomFunction(i, poly, i % 2);
  }

  // Check at random positions
  for (int i = 0; i < 10; i++)
  {
    joint.setPositions(Eigen::Vector2s::Random());
    joint.setVelocities(Eigen::Vector2s::Random());

    /*
    joint.setPosition(0, 0.0);
    joint.setVelocity(0, 1.0);
    */
    std::cout << "Testing: " << joint.getPositions() << ".."
              << joint.getVelocities() << std::endl;

    if (!verifyCustomJoint(&joint, 1e-9))
    {
      return;
    }
  }
}
