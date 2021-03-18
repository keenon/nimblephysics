#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"

#include "TestHelpers.hpp"
#include "GradientTestUtils.hpp"

using namespace dart;
using namespace dart::dynamics;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(JOINT_JACOBIANS, REVOLUTE_JOINT)
{
  std::shared_ptr<World> world = World::create();
  SkeletonPtr box = Skeleton::create("box");
  std::pair<RevoluteJoint*, BodyNode*> pair
      = box->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  RevoluteJoint* boxJoint = pair.first;
  boxJoint->setAxis(Eigen::Vector3d::Random());

  Eigen::Isometry3d fromParent = Eigen::Isometry3d::Identity();
  fromParent.translation() = Eigen::Vector3d::UnitX();
  boxJoint->setTransformFromParentBodyNode(fromParent);

  Eigen::Isometry3d fromChild = Eigen::Isometry3d::Identity();
  fromChild.translation() = Eigen::Vector3d::UnitX() * -3;
  fromChild = fromChild.rotate(Eigen::AngleAxis<double>(M_PI_2, Eigen::Vector3d::UnitX()));
  boxJoint->setTransformFromChildBodyNode(fromChild);
  world->addSkeleton(box);

  EXPECT_TRUE(verifyJointVelocityJacobians(world));
  EXPECT_TRUE(verifyJointPositionJacobians(world));
}
#endif

#ifdef ALL_TESTS
TEST(JOINT_JACOBIANS, FREE_JOINT)
{
  std::shared_ptr<World> world = World::create();
  SkeletonPtr box = Skeleton::create("box");
  std::pair<FreeJoint*, BodyNode*> pair
      = box->createJointAndBodyNodePair<FreeJoint>(nullptr);
  FreeJoint* boxJoint = pair.first;

  Eigen::Isometry3d fromParent = Eigen::Isometry3d::Identity();
  // fromParent.translation() = Eigen::Vector3d::UnitX();
  boxJoint->setTransformFromParentBodyNode(fromParent);

  Eigen::Isometry3d fromChild = Eigen::Isometry3d::Identity();
  // fromChild.translation() = Eigen::Vector3d::UnitX() * -3;
  // fromChild = fromChild.rotate(Eigen::AngleAxis<double>(M_PI_2, Eigen::Vector3d::UnitX()));
  boxJoint->setTransformFromChildBodyNode(fromChild);
  world->addSkeleton(box);

  // box->setPositions(Eigen::Vector6d::Random());

  EXPECT_TRUE(verifyJointVelocityJacobians(world));
  EXPECT_TRUE(verifyJointPositionJacobians(world));
}
#endif