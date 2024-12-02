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

bool verifySpatialJacobians(
    std::shared_ptr<dynamics::Skeleton> skel, neural::WithRespectTo* wrt)
{
  srand(42);
  for (int i = 0; i < 10; i++)
  {
    skel->setPositions(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setVelocities(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setAccelerations(Eigen::VectorXs::Random(skel->getNumDofs()));

    Eigen::MatrixXs comVelJ_fd
        = skel->finiteDifferenceCOMWorldVelocitiesJacobian(wrt);
    Eigen::MatrixXs comVelJ = skel->getCOMWorldVelocitiesJacobian(wrt);

    if (!equals(comVelJ, comVelJ_fd, 1e-8))
    {
      std::cout << "COM Vel wrt " << wrt->name() << " error!" << std::endl;
      for (int body = 0; body < comVelJ.rows() / 6; body++)
      {
        for (int dof = 0; dof < comVelJ.cols(); dof++)
        {
          Eigen::Vector6s analytical = comVelJ.block<6, 1>(body * 6, dof);
          Eigen::Vector6s fd = comVelJ_fd.block<6, 1>(body * 6, dof);
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

    Eigen::MatrixXs comAccJ = skel->getCOMWorldAccelerationsJacobian(wrt);
    Eigen::MatrixXs comAccJ_fd
        = skel->finiteDifferenceCOMWorldAccelerationsJacobian(wrt);
    if (!equals(comAccJ, comAccJ_fd, 1e-8))
    {
      std::cout << "COM acc wrt " << wrt->name() << " error!" << std::endl;
      std::cout << "Analytical: " << std::endl << comAccJ << std::endl;
      std::cout << "FD: " << std::endl << comAccJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << comAccJ - comAccJ_fd << std::endl;
      return false;
    }

    Eigen::MatrixXs velJ_fd
        = skel->finiteDifferenceBodyWorldVelocitiesJacobian(wrt);
    Eigen::MatrixXs velJ = skel->getBodyWorldVelocitiesJacobian(wrt);

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

    Eigen::MatrixXs accJ = skel->getBodyWorldAccelerationsJacobian(wrt);
    Eigen::MatrixXs accJ_fd
        = skel->finiteDifferenceBodyWorldAccelerationsJacobian(wrt);
    if (!equals(accJ, accJ_fd, 1e-8))
    {
      std::cout << "Acc wrt " << wrt->name() << " error!" << std::endl;
      std::cout << "Analytical: " << std::endl << accJ << std::endl;
      std::cout << "FD: " << std::endl << accJ_fd << std::endl;
      std::cout << "Diff: " << std::endl << accJ - accJ_fd << std::endl;
      return false;
    }

    Eigen::MatrixXs comLinAccJ
        = skel->getCOMWorldLinearAccelerationsJacobian(wrt);
    Eigen::MatrixXs comLinAccJ_fd
        = skel->finiteDifferenceCOMWorldLinearAccelerationsJacobian(wrt);
    if (!equals(comLinAccJ, comLinAccJ_fd, 1e-8))
    {
      std::cout << "COM linear acc wrt " << wrt->name() << " error!"
                << std::endl;
      std::cout << "Analytical: " << std::endl << comLinAccJ << std::endl;
      std::cout << "FD: " << std::endl << comLinAccJ_fd << std::endl;
      std::cout << "Diff: " << std::endl
                << comLinAccJ - comLinAccJ_fd << std::endl;
      return false;
    }
  }
  return true;
}

bool verifySpatialJacobians(std::shared_ptr<dynamics::Skeleton> skel)
{
  if (!verifySpatialJacobians(skel, neural::WithRespectTo::VELOCITY))
  {
    return false;
  }
  std::cout << "Passed VELOCITY" << std::endl;
  if (!verifySpatialJacobians(skel, neural::WithRespectTo::GROUP_SCALES))
  {
    return false;
  }
  std::cout << "Passed GROUP_SCALES" << std::endl;
  if (!verifySpatialJacobians(skel, neural::WithRespectTo::POSITION))
  {
    return false;
  }
  std::cout << "Passed POSITION" << std::endl;
  if (!verifySpatialJacobians(skel, neural::WithRespectTo::ACCELERATION))
  {
    return false;
  }
  std::cout << "Passed ACCELERATION" << std::endl;
  if (!verifySpatialJacobians(skel, neural::WithRespectTo::GROUP_COMS))
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

// #ifdef ALL_TESTS
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
// #endif

#ifdef ALL_TESTS
TEST(BODY_SPATIAL_TRANSLATION, RAJAGOPAL)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  skel->autogroupSymmetricSuffixes();

  EXPECT_TRUE(verifySpatialJacobians(skel));
}
#endif