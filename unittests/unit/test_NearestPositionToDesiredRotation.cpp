#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/ConstantCurveIncompressibleJoint.hpp"
#include "dart/dynamics/ConstantCurveJoint.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/EllipsoidJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ScapulathoracicJoint.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/detail/BodyNodePtr.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;

#define ALL_TESTS

template <typename JointType>
bool testNearestPosition(
    std::function<void(JointType* joint)> randomizeJoint = [](JointType*) {},
    std::function<void(JointType* joint)> initJoint = [](JointType*) {})
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  std::pair<JointType*, dynamics::BodyNode*> pair
      = skel->createJointAndBodyNodePair<JointType>();
  JointType* joint = pair.first;
  initJoint(joint);

  for (int k = 0; k < 5; k++)
  {
    for (int i = 0; i < 10; i++)
    {
      Eigen::Vector3s axisAngle = Eigen::Vector3s::Random();
      Eigen::Matrix3s desiredRotation = math::expMapRot(axisAngle);
      Eigen::VectorXs pos
          = joint->getNearestPositionToDesiredRotation(desiredRotation);
      joint->setPositions(pos);
      Eigen::Matrix3s nearestRot = joint->getRelativeTransform().linear();
      s_t nearestDist = (nearestRot - desiredRotation).squaredNorm();

      for (int j = 0; j < 10; j++)
      {
        Eigen::VectorXs perturbed
            = pos + Eigen::VectorXs::Random(pos.size()) * 0.001;
        joint->setPositions(perturbed);
        Eigen::Matrix3s perturbedRot = joint->getRelativeTransform().linear();
        s_t perturbedDist = (perturbedRot - desiredRotation).squaredNorm();
        // We add some numerical tolerance here, to not require _exact_
        // solutions.
        if (perturbedDist <= nearestDist + 1e-9)
        {
          std::cout << "On joint " << joint->getStaticType()
                    << " got a bad nearest rotation!" << std::endl;
          std::cout << "Our desired rotation was " << (axisAngle.norm() / M_PI)
                    << "pi about:" << std::endl
                    << axisAngle.normalized() << std::endl;
          std::cout << "With matrix:" << std::endl
                    << desiredRotation << std::endl;
          std::cout << "Found position (dist = " << nearestDist
                    << "): " << std::endl
                    << pos << std::endl;
          Eigen::Vector3s nearestAxisAngle = math::logMap(nearestRot);
          std::cout << "The resulting rotation was "
                    << (nearestAxisAngle.norm() / M_PI)
                    << "pi about:" << std::endl
                    << nearestAxisAngle.normalized() << std::endl;
          std::cout << "Matrix was: " << std::endl << nearestRot << std::endl;
          std::cout << "But found closer position (dist = " << perturbedDist
                    << "): " << std::endl
                    << perturbed << std::endl;
          Eigen::Vector3s perturbedAxisAngle = math::logMap(perturbedRot);
          std::cout << "The resulting rotation was "
                    << (perturbedAxisAngle.norm() / M_PI)
                    << "pi about:" << std::endl
                    << perturbedAxisAngle.normalized() << std::endl;
          std::cout << "Matrix was: " << std::endl << perturbedRot << std::endl;
          return false;
        }
      }
    }
    randomizeJoint(joint);
    Eigen::Isometry3s childTransform = Eigen::Isometry3s::Identity();
    childTransform.linear() = math::expMapRot(Eigen::Vector3s::Random());
    childTransform.translation() = Eigen::Vector3s::Random();
    joint->setTransformFromChildBodyNode(childTransform);
    Eigen::Isometry3s parentTransform = Eigen::Isometry3s::Identity();
    parentTransform.linear() = math::expMapRot(Eigen::Vector3s::Random());
    parentTransform.translation() = Eigen::Vector3s::Random();
    joint->setTransformFromParentBodyNode(parentTransform);
  }

  return true;
}

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, CLOSEST_ROTATION_APPROX_SAME_AXIS)
{
  Eigen::Vector3s axis = Eigen::Vector3s::UnitX();
  s_t targetAngle = -0.5;
  s_t recovered = math::getClosestRotationalApproximation(
      axis, math::expMapRot(axis * targetAngle));
  EXPECT_NEAR(targetAngle, recovered, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, CLOSEST_ROTATION_APPROX_PERP_AXIS)
{
  s_t targetAngle = -0.5;
  s_t recovered = math::getClosestRotationalApproximation(
      Eigen::Vector3s::UnitZ(),
      math::expMapRot(Eigen::Vector3s::UnitX() * targetAngle));
  EXPECT_NEAR(0.0, recovered, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, EULER_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::EulerJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, EULER_FREE_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::EulerFreeJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, BALL_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::BallJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, FREE_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::FreeJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, UNIVERSAL_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::UniversalJoint>(
      [](dynamics::UniversalJoint* joint) {
        Eigen::Vector3s axis1 = Eigen::Vector3s::Random().normalized();
        joint->setAxis1(axis1);
        joint->setAxis2(
            (axis1.unitOrthogonal() + Eigen::Vector3s::Random() * 0.2)
                .normalized());
      }));
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, REVOLUTE_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::RevoluteJoint>(
      [](dynamics::RevoluteJoint* joint) {
        joint->setAxis(Eigen::Vector3s::Random().normalized());
      }));
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, ELLIPSOID_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::EllipsoidJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, SCAPULA_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::ScapulathoracicJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, CONSTANT_CURVE_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::ConstantCurveJoint>());
}
#endif

#ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, CONSTANT_CURVE_INCOMPRESSIBLE_JOINT)
{
  EXPECT_TRUE(
      testNearestPosition<dynamics::ConstantCurveIncompressibleJoint>());
}
#endif

/*
// #ifdef ALL_TESTS
TEST(NEAREST_POSITION_TO_ROTATION, CUSTOM_JOINT)
{
  EXPECT_TRUE(testNearestPosition<dynamics::CustomJoint<1>>(
      [](dynamics::CustomJoint<1>* joint) {
        (void)joint;
        // Do nothing to randomize the joint
      },
      [](dynamics::CustomJoint<1>* joint) {
        std::shared_ptr<math::LinearFunction> lin
            = std::make_shared<math::LinearFunction>(1.0, 0.0);
        joint->setCustomFunction(0, lin, 0);
        joint->setCustomFunction(1, lin, 0);
        joint->setCustomFunction(2, lin, 0);
      }));
}
// #endif
*/