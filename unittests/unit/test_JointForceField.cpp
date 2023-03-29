#include <memory>
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

bool testJacobians(std::shared_ptr<dynamics::Skeleton> osim)
{
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

  Eigen::MatrixXs analyticalJointsWorldWrtPos
      = osim->getJointWorldPositionsJacobianWrtJointPositions(joints);
  Eigen::MatrixXs bruteForceJointsWorldWrtPos
      = osim->finiteDifferenceJointWorldPositionsJacobianWrtJointPositions(
          joints);
  if (!equals(analyticalJointsWorldWrtPos, bruteForceJointsWorldWrtPos, 1e-8))
  {
    std::cout << "Joint world pos jacobian disagrees!" << std::endl;
    for (int worldJoint = 0; worldJoint < joints.size(); worldJoint++)
    {
      Eigen::MatrixXs analyticalWorldJac = analyticalJointsWorldWrtPos.block(
          3 * worldJoint, 0, 3, analyticalJointsWorldWrtPos.cols());
      Eigen::MatrixXs bruteForceWorldJac = bruteForceJointsWorldWrtPos.block(
          3 * worldJoint, 0, 3, bruteForceJointsWorldWrtPos.cols());
      if (!equals(analyticalWorldJac, bruteForceWorldJac, 1e-8))
      {
        assert(analyticalWorldJac.cols() == osim->getNumDofs());
        assert(bruteForceWorldJac.cols() == osim->getNumDofs());
        for (int dof = 0; dof < osim->getNumDofs(); dof++)
        {
          Eigen::Vector3s analyticalJointWorldGradWrtDof
              = analyticalWorldJac.col(dof);
          Eigen::Vector3s bruteForceJointWorldGradWrtDof
              = bruteForceWorldJac.col(dof);
          if (!equals(
                  analyticalJointWorldGradWrtDof,
                  bruteForceJointWorldGradWrtDof,
                  1e-8))
          {
            std::cout << "Joint[" << worldJoint << "] "
                      << joints[worldJoint]->getName()
                      << " world pos grad wrt dof[" << dof << "] "
                      << osim->getDof(dof)->getName() << " in joint "
                      << osim->getDof(dof)->getJoint()->getName()
                      << " disagrees!" << std::endl;
            dynamics::Joint* childJoint = joints[worldJoint];
            dynamics::Joint* parentJoint = osim->getDof(dof)->getJoint();
            std::cout << "Parent joint " << parentJoint->getName()
                      << " is parent of child joint " << childJoint->getName()
                      << "? "
                      << osim->getJointParentMap()(
                             parentJoint->getJointIndexInSkeleton(),
                             childJoint->getJointIndexInSkeleton())
                      << std::endl;
            Eigen::Matrix3s compare;
            compare.col(0) = analyticalJointWorldGradWrtDof;
            compare.col(1) = bruteForceJointWorldGradWrtDof;
            compare.col(2) = analyticalJointWorldGradWrtDof
                             - bruteForceJointWorldGradWrtDof;
            std::cout << "Analytical - Brute force - Diff:" << std::endl
                      << compare << std::endl;

            /////////////////////////////////////////////////
            // This is the key code we want to check
            const BodyNode* bodyNode = childJoint->getChildBodyNode();
            Eigen::Vector3s originalRotation
                = math::logMap(bodyNode->getWorldTransform().linear());
            const Eigen::Vector3s _localOffset
                = childJoint->getTransformFromChildBodyNode().translation();
            const DegreeOfFreedom* dofPtr = osim->getDof(dof);
            const Joint* joint = dofPtr->getJoint();

            bool isParent = false;
            const BodyNode* cursor = childJoint->getChildBodyNode();
            while (cursor != nullptr)
            {
              if (cursor->getParentJoint() == joint)
              {
                isParent = true;
                break;
              }
              if (cursor->getParentJoint() != nullptr)
              {
                cursor = cursor->getParentJoint()->getParentBodyNode();
              }
            }

            if (isParent)
            {
              Eigen::Vector6s screw = joint->getWorldAxisScrewForPosition(
                  dofPtr->getIndexInJoint());
              std::cout << "Screw for dof: " << std::endl << screw << std::endl;
              std::cout << "Body world pos: " << std::endl
                        << (bodyNode->getWorldTransform() * _localOffset)
                        << std::endl;
              screw.tail<3>() += screw.head<3>().cross(
                  bodyNode->getWorldTransform() * _localOffset);
              std::cout << "Screw after tail modification: " << std::endl
                        << screw << std::endl;
              // This is key so we get an actual gradient of the angle (as a
              // screw), rather than just a screw representing a rotation.
              screw.head<3>() = math::expMapNestedGradient(
                  originalRotation, screw.head<3>());

              std::cout << "Screw after head modification: " << std::endl
                        << screw << std::endl;
              assert(screw.tail<3>() == analyticalJointWorldGradWrtDof);
            }
            else
            {
              assert(Eigen::Vector3s::Zero() == analyticalJointWorldGradWrtDof);
            }
            /////////////////////////////////////////////////
          }
        }
      }
    }
    return false;
  }

  for (int i = 0; i < joints.size(); i++)
  {
    std::cout << "  Testing Joint " << i << "/" << joints.size() << std::endl;
    Eigen::MatrixXs analyticalJointsJac
        = osim->getJointDistanceToOtherJointsJacobianWrtJointWorldPositions(
            joints, i);
    Eigen::MatrixXs bruteForceJointsJac
        = osim->finiteDifferenceJointDistanceToOtherJointsJacobianWrtJointWorldPositions(
            joints, i);
    if (!equals(analyticalJointsJac, bruteForceJointsJac, 1e-8))
    {
      std::cout << "Joint distances wrt joint world position disagrees!"
                << std::endl;
      std::cout << "Analytical: " << std::endl
                << analyticalJointsJac << std::endl;
      std::cout << "Brute force: " << std::endl
                << bruteForceJointsJac << std::endl;
      std::cout << "Diff: " << std::endl
                << analyticalJointsJac - bruteForceJointsJac << std::endl;
      return false;
    }

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
        for (int distTo = 0; distTo < joints.size(); distTo++)
        {
          Eigen::VectorXs analyticalJacRow = analyticalJac.row(distTo);
          Eigen::VectorXs bruteForceJacRow = bruteForceJac.row(distTo);
          if (!equals(analyticalJacRow, bruteForceJacRow, 1e-8))
          {
            std::cout << "Joint distance jacobian wrt " << wrt->name()
                      << " disagrees on distance to " << distTo
                      << " from joint " << i << "!" << std::endl;
            Eigen::MatrixXs compare
                = Eigen::MatrixXs::Zero(analyticalJacRow.size(), 3);
            compare.col(0) = analyticalJacRow;
            compare.col(1) = bruteForceJacRow;
            compare.col(2) = analyticalJacRow - bruteForceJacRow;
            std::cout << "Analytical - Brute Force - Diff: " << std::endl
                      << compare << std::endl;
            for (int i = 0; i < compare.rows(); i++)
            {
              if (abs(compare(i, 2)) > 1e-8)
              {
                if (wrt == neural::WithRespectTo::POSITION)
                {
                  std::cout << "Error wrt " << wrt->name() << "[" << i << "] "
                            << osim->getDof(i)->getName() << std::endl;
                }
                else
                {
                  std::cout << "Error wrt " << wrt->name() << "[" << i << "]"
                            << std::endl;
                }
              }
            }
          }
        }
        return false;
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
        return false;
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
      return false;
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
      return false;
    }
  }
  return true;
}

#ifdef ALL_TESTS
TEST(JointForceFields, RAJAGOPAL_FORCE_FIELD_GRAD)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  EXPECT_TRUE(testJacobians(osim));
}
#endif

// #ifdef ALL_TESTS
TEST(JointForceFields, LAI_ARNOLD_BALL_JOINTS_FORCE_FIELD_GRAD)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/LaiArnoldSubject6/"
            "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim")
            .skeleton;
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);
  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();
  EXPECT_TRUE(testJacobians(osimBallJoints));
}
// #endif