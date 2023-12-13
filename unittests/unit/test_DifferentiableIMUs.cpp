#include <algorithm> // std::sort
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;

#define ALL_TESTS

bool verifyIMUJacobians(
    std::shared_ptr<dynamics::Skeleton> skel,
    neural::WithRespectTo* wrt,
    bool isComplexSkeleton = false)
{
  srand(42);

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> sensors;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    sensors.emplace_back(skel->getBodyNode(i), Eigen::Isometry3s::Identity());
    Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
    T.translation() = Eigen::Vector3s::Ones();
    sensors.emplace_back(skel->getBodyNode(i), T);
    Eigen::Isometry3s T2 = Eigen::Isometry3s::Identity();
    T2.linear() = math::expMapRot(Eigen::Vector3s::Ones());
    sensors.emplace_back(skel->getBodyNode(i), T2);
    Eigen::Isometry3s T3 = Eigen::Isometry3s::Identity();
    T3.translation() = Eigen::Vector3s::Random();
    T3.linear() = math::expMapRot(Eigen::Vector3s::Random());
    sensors.emplace_back(skel->getBodyNode(i), T3);
  }

  Eigen::Vector3s oldGravity = skel->getGravity();
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s gravity = Eigen::Vector3s::Random();
    skel->setGravity(gravity);
    skel->setPositions(Eigen::VectorXs::Zero(skel->getNumDofs()));
    skel->setVelocities(Eigen::VectorXs::Zero(skel->getNumDofs()));
    skel->setAccelerations(Eigen::VectorXs::Zero(skel->getNumDofs()));

    Eigen::VectorXs gyroReadings = skel->getGyroReadings(sensors);
    Eigen::VectorXs expectedGyroReadings
        = Eigen::VectorXs::Zero(sensors.size() * 3);
    if (!equals(gyroReadings, expectedGyroReadings, 1e-8))
    {
      std::cout << "Gyro reading error!" << std::endl;
      std::cout << "Expected: " << std::endl
                << Eigen::VectorXs::Zero(sensors.size() * 3) << std::endl;
      std::cout << "Actual: " << std::endl << gyroReadings << std::endl;
      std::cout << "Diff: " << std::endl
                << Eigen::VectorXs::Zero(sensors.size() * 3) - gyroReadings
                << std::endl;
      return false;
    }

    Eigen::VectorXs accReadings = skel->getAccelerometerReadings(sensors);
    for (int b = 0; b < sensors.size(); b++)
    {
      Eigen::Isometry3s T_wb = sensors[b].first->getWorldTransform();
      Eigen::Isometry3s T_ba = sensors[b].second;
      Eigen::Isometry3s T_wa = T_wb * T_ba;
      Eigen::Vector3s expected = T_wa.linear().transpose() * -gravity;
      Eigen::Vector3s actual = accReadings.segment<3>(b * 3);

      if (!equals(expected, actual, 1e-8))
      {
        std::cout << "Acc reading error!" << std::endl;
        std::cout << "Expected: " << std::endl << expected << std::endl;
        std::cout << "Actual: " << std::endl << actual << std::endl;
        std::cout << "Diff: " << std::endl << expected - actual << std::endl;
        return false;
      }
    }
  }
  skel->setGravity(oldGravity);

  for (int i = 0; i < 10; i++)
  {
    skel->setPositions(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setVelocities(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setAccelerations(Eigen::VectorXs::Random(skel->getNumDofs()));

    Eigen::MatrixXs velJ_fd
        = skel->finiteDifferenceBodyLocalVelocitiesJacobian(wrt);
    Eigen::MatrixXs velJ = skel->getBodyLocalVelocitiesJacobian(wrt);

    if (!equals(velJ, velJ_fd, 1e-8))
    {
      std::cout << "Vel wrt " << wrt->name() << " error!" << std::endl;
      for (int body = 0; body < velJ.rows() / 6; body++)
      {
        for (int dof = 0; dof < velJ.cols(); dof++)
        {
          Eigen::Vector6s analytical = velJ.block<6, 1>(body * 6, dof);
          Eigen::Vector6s fd = velJ_fd.block<6, 1>(body * 6, dof);
          if (!equals(analytical, fd, 1e-8))
          {
            std::cout << "Body \"" << skel->getBodyNode(body)->getName()
                      << "\" disagrees on DOF \""
                      << skel->getDof(dof)->getName() << "\"" << std::endl;
            Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(6, 3);
            compare.col(0) = fd;
            compare.col(1) = analytical;
            compare.col(2) = analytical - fd;
            std::cout << "FD - Analytical - Diff:" << std::endl
                      << compare << std::endl;
          }
        }
      }
      return false;
    }
    if (isComplexSkeleton
        && (wrt == neural::WithRespectTo::VELOCITY
            || wrt == neural::WithRespectTo::GROUP_SCALES
            || wrt == neural::WithRespectTo::POSITION))
    {
      if (velJ.isZero())
      {
        std::cout << "Vel wrt " << wrt->name() << " is zero!" << std::endl;
        return false;
      }
    }

    Eigen::MatrixXs localAccJ = skel->getBodyLocalAccelerationsJacobian(wrt);
    Eigen::MatrixXs localAccJ_fd
        = skel->finiteDifferenceBodyLocalAccelerationsJacobian(wrt);
    if (!equals(localAccJ, localAccJ_fd, 1e-8))
    {
      std::cout << "Acc wrt " << wrt->name() << " error!" << std::endl;
      std::cout << "Analytical: " << std::endl << localAccJ << std::endl;
      std::cout << "FD: " << std::endl << localAccJ_fd << std::endl;
      std::cout << "Diff: " << std::endl
                << localAccJ - localAccJ_fd << std::endl;
      return false;
    }
    if (isComplexSkeleton
        && (wrt == neural::WithRespectTo::VELOCITY
            || wrt == neural::WithRespectTo::GROUP_SCALES
            || wrt == neural::WithRespectTo::POSITION))
    {
      if (localAccJ.isZero())
      {
        std::cout << "Local acc wrt " << wrt->name() << " is zero!"
                  << std::endl;
        return false;
      }
    }

    Eigen::MatrixXs localGravJ = skel->getBodyLocalGravityVectorsJacobian(wrt);
    Eigen::MatrixXs localGravJ_fd
        = skel->finiteDifferenceBodyLocalGravityVectorsJacobian(wrt);
    if (!equals(localGravJ, localGravJ_fd, 1e-8))
    {
      std::cout << "Grav wrt " << wrt->name() << " error!" << std::endl;
      std::cout << "Analytical: " << std::endl << localGravJ << std::endl;
      std::cout << "FD: " << std::endl << localGravJ_fd << std::endl;
      std::cout << "Diff: " << std::endl
                << localGravJ - localGravJ_fd << std::endl;
      return false;
    }
    if (isComplexSkeleton && (wrt == neural::WithRespectTo::POSITION))
    {
      if (localGravJ.isZero())
      {
        std::cout << "Local grav wrt " << wrt->name() << " is zero!"
                  << std::endl;
        return false;
      }
    }

    Eigen::MatrixXs gyroJ = skel->getGyroReadingsJacobianWrt(sensors, wrt);
    Eigen::MatrixXs gyroJ_fd
        = skel->finiteDifferenceGyroReadingsJacobianWrt(sensors, wrt);
    if (!equals(gyroJ, gyroJ_fd, 1e-8))
    {
      std::cout << "Gyro wrt " << wrt->name() << " error!" << std::endl;
      for (int i = 0; i < sensors.size(); i++)
      {
        for (int dof = 0; dof < gyroJ.cols(); dof++)
        {
          Eigen::Vector3s analytical = gyroJ.block<3, 1>(i * 3, dof);
          Eigen::Vector3s fd = gyroJ_fd.block<3, 1>(i * 3, dof);
          if (!equals(analytical, fd, 1e-8))
          {
            std::cout << "Sensor on body \"" << sensors[i].first->getName()
                      << "\" @ [" << sensors[i].second.translation()(0) << ", "
                      << sensors[i].second.translation()(1) << ", "
                      << sensors[i].second.translation()(2)
                      << "] disagrees WRT " << wrt->name() << "[" << i << "]"
                      << std::endl;
            Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 3);
            compare.col(0) = fd;
            compare.col(1) = analytical;
            compare.col(2) = analytical - fd;
            std::cout << "FD - Analytical - Diff:" << std::endl
                      << compare << std::endl;
          }
        }
      }
      return false;
    }

    Eigen::MatrixXs accJ
        = skel->getAccelerometerReadingsJacobianWrt(sensors, wrt);
    Eigen::MatrixXs accJ_fd
        = skel->finiteDifferenceAccelerometerReadingsJacobianWrt(sensors, wrt);
    if (!equals(accJ, accJ_fd, 1e-8))
    {
      std::cout << "Accelerometer wrt " << wrt->name() << " error!"
                << std::endl;
      for (int i = 0; i < sensors.size(); i++)
      {
        for (int dof = 0; dof < accJ.cols(); dof++)
        {
          Eigen::Vector3s analytical = accJ.block<3, 1>(i * 3, dof);
          Eigen::Vector3s fd = accJ_fd.block<3, 1>(i * 3, dof);
          if (!equals(analytical, fd, 1e-8))
          {
            std::cout << "Sensor on body \"" << sensors[i].first->getName()
                      << "\" @ [" << sensors[i].second.translation()(0) << ", "
                      << sensors[i].second.translation()(1) << ", "
                      << sensors[i].second.translation()(2)
                      << "] disagrees WRT " << wrt->name() << "[" << i << "]"
                      << std::endl;
            Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 3);
            compare.col(0) = fd;
            compare.col(1) = analytical;
            compare.col(2) = analytical - fd;
            std::cout << "FD - Analytical - Diff:" << std::endl
                      << compare << std::endl;
          }
        }
      }
      return false;
    }

    Eigen::Vector3s magField = Eigen::Vector3s::Random();

    Eigen::MatrixXs magJ
        = skel->getMagnetometerReadingsJacobianWrt(sensors, magField, wrt);
    Eigen::MatrixXs magJ_fd
        = skel->finiteDifferenceMagnetometerReadingsJacobianWrt(
            sensors, magField, wrt);
    if (!equals(magJ, magJ_fd, 1e-8))
    {
      std::cout << "Magnetometer wrt " << wrt->name() << " error!" << std::endl;
      for (int i = 0; i < sensors.size(); i++)
      {
        for (int dof = 0; dof < magJ.cols(); dof++)
        {
          Eigen::Vector3s analytical = magJ.block<3, 1>(i * 3, dof);
          Eigen::Vector3s fd = magJ_fd.block<3, 1>(i * 3, dof);
          if (!equals(analytical, fd, 1e-8))
          {
            std::cout << "Sensor on body \"" << sensors[i].first->getName()
                      << "\" @ [" << sensors[i].second.translation()(0) << ", "
                      << sensors[i].second.translation()(1) << ", "
                      << sensors[i].second.translation()(2)
                      << "] disagrees WRT " << wrt->name() << "[" << i << "]"
                      << std::endl;
            Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 3);
            compare.col(0) = fd;
            compare.col(1) = analytical;
            compare.col(2) = analytical - fd;
            std::cout << "FD - Analytical - Diff:" << std::endl
                      << compare << std::endl;
          }
        }
      }
      return false;
    }

    Eigen::MatrixXs magJWrtField
        = skel->getMagnetometerReadingsJacobianWrtMagneticField(
            sensors, magField);
    Eigen::MatrixXs magJWrtField_fd
        = skel->finiteDifferenceMagnetometerReadingsJacobianWrtMagneticField(
            sensors, magField);
    if (!equals(magJWrtField, magJWrtField_fd, 1e-8))
    {
      std::cout << "Magnetometer wrt field error!" << std::endl;
      for (int i = 0; i < sensors.size(); i++)
      {
        Eigen::Matrix3s analytical = magJ.block<3, 3>(i * 3, 0);
        Eigen::Matrix3s fd = magJ_fd.block<3, 3>(i * 3, 0);
        if (!equals(analytical, fd, 1e-8))
        {
          std::cout << "Sensor on body \"" << sensors[i].first->getName()
                    << "\" @ [" << sensors[i].second.translation()(0) << ", "
                    << sensors[i].second.translation()(1) << ", "
                    << sensors[i].second.translation()(2) << "] disagrees WRT "
                    << wrt->name() << "[" << i << "]" << std::endl;
          std::cout << "FD: " << std::endl << fd << std::endl;
          std::cout << "Analytical: " << std::endl << analytical << std::endl;
          std::cout << "Diff: " << std::endl << analytical - fd << std::endl;
        }
      }
      return false;
    }
  }
  return true;
}

bool verifySpatialJacobians(
    std::shared_ptr<dynamics::Skeleton> skel, bool isComplexSkeleton = false)
{
  if (!verifyIMUJacobians(
          skel, neural::WithRespectTo::VELOCITY, isComplexSkeleton))
  {
    return false;
  }
  std::cout << "Passed VELOCITY" << std::endl;
  if (!verifyIMUJacobians(
          skel, neural::WithRespectTo::GROUP_SCALES, isComplexSkeleton))
  {
    return false;
  }
  std::cout << "Passed GROUP_SCALES" << std::endl;
  if (!verifyIMUJacobians(
          skel, neural::WithRespectTo::POSITION, isComplexSkeleton))
  {
    return false;
  }
  std::cout << "Passed POSITION" << std::endl;
  if (!verifyIMUJacobians(
          skel, neural::WithRespectTo::ACCELERATION, isComplexSkeleton))
  {
    return false;
  }
  std::cout << "Passed ACCELERATION" << std::endl;
  if (!verifyIMUJacobians(
          skel, neural::WithRespectTo::GROUP_COMS, isComplexSkeleton))
  {
    return false;
  }
  std::cout << "Passed GROUP_COMS" << std::endl;
  return true;
}

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, BOX)
{
  std::shared_ptr<dynamics::Skeleton> skel = createBox(Eigen::Vector3s::Ones());
  skel->getBodyNode(0)->setLocalCOM(Eigen::Vector3s::Random());
  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, BOX_WITH_CHILD_TRANSFORM)
{
  std::shared_ptr<dynamics::Skeleton> skel = createBox(Eigen::Vector3s::Ones());
  Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();
  fromChild.translation() = Eigen::Vector3s::Random();
  skel->getJoint(0)->setTransformFromChildBodyNode(fromChild);
  skel->getBodyNode(0)->setLocalCOM(Eigen::Vector3s::Random());
  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, CARTPOLE)
{
  std::shared_ptr<dynamics::Skeleton> skel = createCartpole();
  skel->getBodyNode(0)->setLocalCOM(Eigen::Vector3s::Random());
  skel->getBodyNode(1)->setLocalCOM(Eigen::Vector3s::Random());
  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, TWO_LINK)
{
  std::shared_ptr<dynamics::Skeleton> skel = createTwoLinkRobot(
      Eigen::Vector3s::Random().cwiseAbs(),
      TypeOfDOF::DOF_ROLL,
      Eigen::Vector3s::Random().cwiseAbs(),
      TypeOfDOF::DOF_PITCH);
  skel->getBodyNode(0)->setLocalCOM(Eigen::Vector3s::Random());
  skel->getBodyNode(1)->setLocalCOM(Eigen::Vector3s::Random());
  skel->getBodyNode(2)->setLocalCOM(Eigen::Vector3s::Random());
  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, THREE_LINK)
{
  std::shared_ptr<dynamics::Skeleton> skel = createThreeLinkRobot(
      Eigen::Vector3s::Random().cwiseAbs(),
      TypeOfDOF::DOF_ROLL,
      Eigen::Vector3s::Random().cwiseAbs(),
      TypeOfDOF::DOF_ROLL,
      Eigen::Vector3s::Random().cwiseAbs(),
      TypeOfDOF::DOF_PITCH);
  skel->getBodyNode(0)->setLocalCOM(Eigen::Vector3s::Random());
  skel->getBodyNode(1)->setLocalCOM(Eigen::Vector3s::Random());
  skel->getBodyNode(2)->setLocalCOM(Eigen::Vector3s::Random());
  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, EULER_FREE_JOINT)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto eulerJointPair
      = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>();
  dynamics::Joint* eulerJoint = eulerJointPair.first;
  (void)eulerJoint;
  dynamics::BodyNode* rootBody = eulerJointPair.second;
  rootBody->setName("root");
  rootBody->setLocalCOM(Eigen::Vector3s::Random());

  auto revoluteJointPair
      = rootBody->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>();
  dynamics::Joint* childJoint = revoluteJointPair.first;
  dynamics::BodyNode* childBody = revoluteJointPair.second;
  childBody->setName("child");
  childBody->setLocalCOM(Eigen::Vector3s::Random());

  Eigen::Isometry3s T_rc = Eigen::Isometry3s::Identity();
  T_rc.translation() = Eigen::Vector3s::UnitX();
  childJoint->setTransformFromParentBodyNode(T_rc.inverse());
  childJoint->setTransformFromChildBodyNode(T_rc);

  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, RAJAGOPAL)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  skel->autogroupSymmetricSuffixes();

  EXPECT_TRUE(verifySpatialJacobians(skel, true));
}
#endif