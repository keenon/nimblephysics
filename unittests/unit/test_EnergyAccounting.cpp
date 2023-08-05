#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/detail/BodyNodePtr.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, LINEAR_PERPENDICULAR_TO_GRAVITY)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>();
  pair.first->setAxis(Eigen::Vector3s::UnitX());
  pair.second->setMass(1.0);

  skel->setPositions(Eigen::Vector1s(0.0));
  skel->setVelocities(Eigen::Vector1s(1.0));
  skel->setAccelerations(Eigen::Vector1s(1.0));

  auto accounting = skel->getEnergyAccounting();
  std::cout << "Body kinetic energy: " << accounting.bodyKineticEnergy
            << std::endl;
  std::cout << "Body potential energy: " << accounting.bodyPotentialEnergy
            << std::endl;
  EXPECT_EQ(accounting.bodyKineticEnergy(0), 0.5);
  EXPECT_EQ(accounting.bodyPotentialEnergy(0), 0.0);

  for (int i = 0; i < accounting.joints.size(); i++)
  {
    auto& joint = accounting.joints[i];
    EXPECT_EQ(joint.powerToChild, 1.0);
    EXPECT_EQ(joint.powerToParent, 0.0);
    std::cout << "Power to child: " << joint.powerToChild << std::endl;
    std::cout << "Power to parent: " << joint.powerToParent << std::endl;
  }
}
#endif

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, LINEAR_PERPENDICULAR_TO_GRAVITY_REFERENCE_VEL)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>();
  pair.first->setAxis(Eigen::Vector3s::UnitX());
  pair.second->setMass(1.0);

  skel->setPositions(Eigen::Vector1s(0.0));
  skel->setVelocities(Eigen::Vector1s(1.0));
  skel->setAccelerations(Eigen::Vector1s(1.0));

  auto accounting = skel->getEnergyAccounting(0.0, Eigen::Vector3s::UnitX());
  std::cout << "Body kinetic energy: " << accounting.bodyKineticEnergy
            << std::endl;
  std::cout << "Body potential energy: " << accounting.bodyPotentialEnergy
            << std::endl;
  EXPECT_EQ(accounting.bodyKineticEnergy(0), 2.0);
  EXPECT_EQ(accounting.bodyPotentialEnergy(0), 0.0);

  for (int i = 0; i < accounting.joints.size(); i++)
  {
    auto& joint = accounting.joints[i];
    EXPECT_EQ(joint.powerToChild, 2.0);
    EXPECT_EQ(joint.powerToParent, 0.0);
    std::cout << "Power to child: " << joint.powerToChild << std::endl;
    std::cout << "Power to parent: " << joint.powerToParent << std::endl;
  }
}
#endif

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, LINEAR_PARALLEL_TO_GRAVITY)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>();
  skel->setGravity(Eigen::Vector3s::UnitY() * -9.81);
  pair.first->setAxis(Eigen::Vector3s::UnitY());
  pair.second->setMass(1.0);

  skel->setPositions(Eigen::Vector1s(1.0));
  skel->setVelocities(Eigen::Vector1s(0.0));
  skel->setAccelerations(Eigen::Vector1s(-9.81));

  auto accounting = skel->getEnergyAccounting();
  std::cout << "Body kinetic energy: " << accounting.bodyKineticEnergy
            << std::endl;
  std::cout << "Body potential energy: " << accounting.bodyPotentialEnergy
            << std::endl;
  EXPECT_EQ(accounting.bodyKineticEnergy(0), 0.0);
  EXPECT_EQ(accounting.bodyPotentialEnergy(0), 9.81);

  for (int i = 0; i < accounting.joints.size(); i++)
  {
    auto& joint = accounting.joints[i];
    EXPECT_EQ(joint.powerToChild, 0.0);
    EXPECT_EQ(joint.powerToParent, 0.0);
    std::cout << "Power to child: " << joint.powerToChild << std::endl;
    std::cout << "Power to parent: " << joint.powerToParent << std::endl;
  }
}
#endif

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, LINEAR_PARALLEL_TO_GRAVITY_REFERENCE_VEL)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>();
  skel->setGravity(Eigen::Vector3s::UnitY() * -9.81);
  pair.first->setAxis(Eigen::Vector3s::UnitY());
  pair.second->setMass(1.0);

  skel->setPositions(Eigen::Vector1s(1.0));
  skel->setVelocities(Eigen::Vector1s(0.0));
  skel->setAccelerations(Eigen::Vector1s(-9.81));

  auto accounting = skel->getEnergyAccounting(0, Eigen::Vector3s::UnitY());
  std::cout << "Body kinetic energy: " << accounting.bodyKineticEnergy
            << std::endl;
  std::cout << "Body potential energy: " << accounting.bodyPotentialEnergy
            << std::endl;
  EXPECT_EQ(accounting.bodyKineticEnergy(0), 0.5);
  EXPECT_EQ(accounting.bodyPotentialEnergy(0), 9.81);

  for (int i = 0; i < accounting.joints.size(); i++)
  {
    auto& joint = accounting.joints[i];
    EXPECT_EQ(joint.powerToChild, 0.0);
    EXPECT_EQ(joint.powerToParent, 0.0);
    std::cout << "Power to child: " << joint.powerToChild << std::endl;
    std::cout << "Power to parent: " << joint.powerToParent << std::endl;
  }
}
#endif

#ifdef ALL_TESTS
TEST(
    ENERGY_ACCOUNTING,
    LINEAR_PERPENDICULAR_TO_GRAVITY_THEN_SECOND_PERPENDICULAR)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  skel->setGravity(Eigen::Vector3s::UnitY());
  auto pair = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>();
  pair.first->setAxis(Eigen::Vector3s::UnitX());
  pair.first->setName("joint0");
  pair.second->setMass(1.0);
  auto pair2
      = pair.second
            ->createChildJointAndBodyNodePair<dynamics::PrismaticJoint>();
  pair2.first->setName("joint1");
  pair2.first->setAxis(Eigen::Vector3s::UnitZ());
  pair2.second->setMass(1.0);

  skel->setPositions(Eigen::Vector2s(0.0, 0.0));
  skel->setVelocities(Eigen::Vector2s(1.0, 0.0));
  skel->setAccelerations(Eigen::Vector2s(1.0, 0.0));

  auto accounting = skel->getEnergyAccounting();
  std::cout << "Body kinetic energy: " << accounting.bodyKineticEnergy
            << std::endl;
  std::cout << "Body potential energy: " << accounting.bodyPotentialEnergy
            << std::endl;
  EXPECT_EQ(accounting.bodyKineticEnergy(0), 0.5);
  EXPECT_EQ(accounting.bodyKineticEnergy(1), 0.5);
  EXPECT_EQ(accounting.bodyPotentialEnergy(0), 0.0);
  EXPECT_EQ(accounting.bodyPotentialEnergy(1), 0.0);

  std::cout << accounting.joints[0].name << std::endl;
  EXPECT_EQ(accounting.joints[0].powerToChild, 2.0);
  EXPECT_EQ(accounting.joints[0].powerToParent, 0.0);
  std::cout << "Joint 0 power to child: " << accounting.joints[0].powerToChild
            << std::endl;
  std::cout << "Joint 0 power to parent: " << accounting.joints[0].powerToParent
            << std::endl;

  EXPECT_EQ(accounting.joints[1].powerToChild, 1.0);
  EXPECT_EQ(accounting.joints[1].powerToParent, -1.0);
  std::cout << "Joint 1 power to child: " << accounting.joints[1].powerToChild
            << std::endl;
  std::cout << "Joint 1 power to parent: " << accounting.joints[1].powerToParent
            << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, ENERGY_GRADIENTS)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  skel->setGravity(Eigen::Vector3s::UnitY());
  auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>();
  pair.first->setName("joint0");
  pair.second->setMass(1.0);
  pair.second->setMomentOfInertia(0.5, 0.7, 0.9, 0.1, 0.2, 0.3);

  for (int i = 0; i < 20; i++)
  {
    skel->setPositions(Eigen::Vector6s::Random());
    skel->setVelocities(Eigen::Vector6s::Random());
    skel->setAccelerations(Eigen::Vector6s::Random());
    auto accounting1 = skel->getEnergyAccounting();
    s_t dt = 1e-4;
    skel->integrateVelocities(dt);
    skel->integratePositions(dt);
    auto accounting2 = skel->getEnergyAccounting();

    s_t fdKinetic
        = (accounting2.bodyKineticEnergy(0) - accounting1.bodyKineticEnergy(0))
          / dt;
    s_t gradKinetic = accounting1.bodyKineticEnergyDeriv(0);
    EXPECT_NEAR(fdKinetic, gradKinetic, 5e-4);

    s_t fdPotential = (accounting2.bodyPotentialEnergy(0)
                       - accounting1.bodyPotentialEnergy(0))
                      / dt;
    s_t gradPotential = accounting1.bodyPotentialEnergyDeriv(0);
    EXPECT_NEAR(fdPotential, gradPotential, 5e-4);
  }
}
#endif

#ifdef ALL_TESTS
TEST(ENERGY_ACCOUNTING, ENERGY_GRADIENTS_WITH_REFERENCE_VEL)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  skel->setGravity(Eigen::Vector3s::UnitY());
  auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>();
  pair.first->setName("joint0");
  pair.second->setMass(1.0);
  pair.second->setMomentOfInertia(0.5, 0.7, 0.9, 0.1, 0.2, 0.3);

  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector3s referenceVel = Eigen::Vector3s::Random();
    skel->setPositions(Eigen::Vector6s::Random());
    skel->setVelocities(Eigen::Vector6s::Random());
    skel->setAccelerations(Eigen::Vector6s::Random());
    auto accounting1 = skel->getEnergyAccounting(0.0, referenceVel);
    s_t dt = 1e-4;
    skel->integrateVelocities(dt);
    skel->integratePositions(dt);
    auto accounting2 = skel->getEnergyAccounting(0.0, referenceVel);

    s_t fdKinetic
        = (accounting2.bodyKineticEnergy(0) - accounting1.bodyKineticEnergy(0))
          / dt;
    s_t gradKinetic = accounting1.bodyKineticEnergyDeriv(0);
    EXPECT_NEAR(fdKinetic, gradKinetic, 5e-4);

    s_t fdPotential = (accounting2.bodyPotentialEnergy(0)
                       - accounting1.bodyPotentialEnergy(0))
                      / dt;
    s_t gradPotential = accounting1.bodyPotentialEnergyDeriv(0);
    EXPECT_NEAR(fdPotential, gradPotential, 5e-4);
  }
}
#endif