/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <array>
#include <iostream>

#include <gtest/gtest.h>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/PlanarJoint.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ScrewJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/TranslationalJoint2D.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/SkelParser.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dart::math;
using namespace dart::dynamics;
using namespace dart::simulation;

#define JOINT_TOL 0.01

#define ALL_TESTS

//==============================================================================
class JOINTS : public testing::Test
{
public:
  // Get reference frames
  const std::vector<SimpleFrame*>& getFrames() const;

  // Randomize the properties of all the reference frames
  void randomizeRefFrames();

#ifdef _WIN32
  template <typename JointType>
  static typename JointType::Properties createJointProperties()
  {
    return typename JointType::Properties();
  }
#endif

  template <typename JointType>
  void kinematicsTest(
#ifdef _WIN32
      const typename JointType::Properties& _joint
      = BodyNode::createJointProperties<JointType>());
#else
      const typename JointType::Properties& _joint
      = typename JointType::Properties());
#endif

protected:
  // Sets up the test fixture.
  void SetUp() override;

  std::vector<SimpleFrame*> frames;
};

//==============================================================================
void JOINTS::SetUp()
{
  // Create a list of reference frames to use during tests
  frames.push_back(new SimpleFrame(Frame::World(), "refFrame1"));
  frames.push_back(new SimpleFrame(frames.back(), "refFrame2"));
  frames.push_back(new SimpleFrame(frames.back(), "refFrame3"));
  frames.push_back(new SimpleFrame(frames.back(), "refFrame4"));
  frames.push_back(new SimpleFrame(Frame::World(), "refFrame5"));
  frames.push_back(new SimpleFrame(frames.back(), "refFrame6"));
}

//==============================================================================
const std::vector<SimpleFrame*>& JOINTS::getFrames() const
{
  return frames;
}

//==============================================================================
void JOINTS::randomizeRefFrames()
{
  for (std::size_t i = 0; i < frames.size(); ++i)
  {
    SimpleFrame* F = frames[i];

    Eigen::Vector3s p = Random::uniform<Eigen::Vector3s>(-100, 100);
    Eigen::Vector3s theta
        = Random::uniform<Eigen::Vector3s>(-2 * M_PI, 2 * M_PI);

    Eigen::Isometry3s tf(Eigen::Isometry3s::Identity());
    tf.translate(p);
    tf.linear() = math::eulerXYZToMatrix(theta);

    F->setRelativeTransform(tf);
    F->setRelativeSpatialVelocity(Random::uniform<Eigen::Vector6s>(-100, 100));
    F->setRelativeSpatialAcceleration(
        Random::uniform<Eigen::Vector6s>(-100, 100));
  }
}

//==============================================================================
template <typename JointType>
void JOINTS::kinematicsTest(const typename JointType::Properties& _properties)
{
#ifdef NDEBUG
  int numTests = 5;
#else
  int numTests = 2;
#endif

  SkeletonPtr skeleton = Skeleton::create();
  Joint* joint
      = skeleton->createJointAndBodyNodePair<JointType>(nullptr, _properties)
            .first;
  joint->setTransformFromChildBodyNode(math::expMap(Eigen::Vector6s::Random()));
  joint->setTransformFromParentBodyNode(
      math::expMap(Eigen::Vector6s::Random()));

  int dof = joint->getNumDofs();

  //--------------------------------------------------------------------------
  //
  //--------------------------------------------------------------------------
  VectorXs q = VectorXs::Zero(dof);
  VectorXs dq = VectorXs::Zero(dof);

  for (int idxTest = 0; idxTest < numTests; ++idxTest)
  {
    s_t q_delta = 1e-6;

    for (int i = 0; i < dof; ++i)
    {
      q(i) = Random::uniform(-constantsd::pi() * 1.0, constantsd::pi() * 1.0);
      dq(i) = Random::uniform(-constantsd::pi() * 1.0, constantsd::pi() * 1.0);
    }

    joint->setPositions(q);
    joint->setVelocities(dq);

    if (dof == 0)
      return;

    Eigen::Isometry3s T = joint->getRelativeTransform();
    Jacobian J = joint->getRelativeJacobian();
    Jacobian dJ = joint->getRelativeJacobianTimeDeriv();

    //--------------------------------------------------------------------------
    // Test T
    //--------------------------------------------------------------------------
    EXPECT_TRUE(math::verifyTransform(T));

    //--------------------------------------------------------------------------
    // Test analytic Jacobian and numerical Jacobian
    // J == numericalJ
    //--------------------------------------------------------------------------
    Jacobian numericJ = Jacobian::Zero(6, dof);
    for (int i = 0; i < dof; ++i)
    {
      // a
      Eigen::VectorXs q_a = q;
      joint->setPositions(q_a);
      Eigen::Isometry3s T_a = joint->getRelativeTransform();

      // b
      Eigen::VectorXs q_b = q;
      q_b(i) += q_delta;
      joint->setPositions(q_b);
      Eigen::Isometry3s T_b = joint->getRelativeTransform();

      //
      Eigen::Isometry3s Tinv_a = T_a.inverse();
      Eigen::Matrix4s Tinv_a_eigen = Tinv_a.matrix();

      // dTdq
      Eigen::Matrix4s T_a_eigen = T_a.matrix();
      Eigen::Matrix4s T_b_eigen = T_b.matrix();
      Eigen::Matrix4s dTdq_eigen = (T_b_eigen - T_a_eigen) / q_delta;
      // Matrix4s dTdq_eigen = (T_b_eigen * T_a_eigen.inverse()) / dt;

      // J(i)
      Eigen::Matrix4s Ji_4x4matrix_eigen = Tinv_a_eigen * dTdq_eigen;
      Eigen::Vector6s Ji;
      Ji[0] = Ji_4x4matrix_eigen(2, 1);
      Ji[1] = Ji_4x4matrix_eigen(0, 2);
      Ji[2] = Ji_4x4matrix_eigen(1, 0);
      Ji[3] = Ji_4x4matrix_eigen(0, 3);
      Ji[4] = Ji_4x4matrix_eigen(1, 3);
      Ji[5] = Ji_4x4matrix_eigen(2, 3);
      numericJ.col(i) = Ji;
    }

    auto res = equals(J, numericJ, JOINT_TOL);
    EXPECT_TRUE(res);
    if (!res)
    {
      std::cout << "J       : \n" << J << std::endl;
      std::cout << "numericJ: \n" << numericJ << std::endl;
    }

    //--------------------------------------------------------------------------
    // Test first time derivative of analytic Jacobian and numerical Jacobian
    // dJ == numerical_dJ
    //--------------------------------------------------------------------------
    Jacobian numeric_dJ = Jacobian::Zero(6, dof);
    for (int i = 0; i < dof; ++i)
    {
      // a
      Eigen::VectorXs q_a = q;
      joint->setPositions(q_a);
      Jacobian J_a = joint->getRelativeJacobian();

      // b
      Eigen::VectorXs q_b = q;
      q_b(i) += q_delta;
      joint->setPositions(q_b);
      Jacobian J_b = joint->getRelativeJacobian();

      //
      Jacobian dJ_dq = (J_b - J_a) / q_delta;

      // J(i)
      numeric_dJ += dJ_dq * dq(i);
    }

    for (int i = 0; i < dof; ++i)
    {
      for (int j = 0; j < 6; ++j)
        EXPECT_NEAR(
            static_cast<double>(dJ.col(i)(j)),
            static_cast<double>(numeric_dJ.col(i)(j)),
            static_cast<double>(JOINT_TOL));
    }
  }

  // Forward kinematics test with high joint position
  s_t posMin = -1e+64;
  s_t posMax = +1e+64;

  for (int idxTest = 0; idxTest < numTests; ++idxTest)
  {
    for (int i = 0; i < dof; ++i)
      q(i) = Random::uniform(posMin, posMax);

    skeleton->setPositions(q);

    if (joint->getNumDofs() == 0)
      return;

    Eigen::Isometry3s T = joint->getRelativeTransform();
    EXPECT_TRUE(math::verifyTransform(T));
  }
}

// 0-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, WELD_JOINT)
{
  kinematicsTest<WeldJoint>();
}
#endif

// 1-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, REVOLUTE_JOINT)
{
  kinematicsTest<RevoluteJoint>();
}
#endif

// 1-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, PRISMATIC_JOINT)
{
  kinematicsTest<PrismaticJoint>();
}
#endif

// 1-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, SCREW_JOINT)
{
  kinematicsTest<ScrewJoint>();
}
#endif

// 2-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, UNIVERSAL_JOINT)
{
  kinematicsTest<UniversalJoint>();
}
#endif

// 2-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, TRANSLATIONAL_JOINT_2D)
{
  kinematicsTest<TranslationalJoint2D>();
}
#endif

// 3-dof joint
#ifdef ALL_TESTS
#ifndef DART_USE_IDENTITY_JACOBIAN
TEST_F(JOINTS, BALL_JOINT)
{
  kinematicsTest<BallJoint>();
}
#endif
#endif

// 3-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, EULER_JOINT)
{
  EulerJoint::Properties properties;

  properties.mAxisOrder = EulerJoint::AxisOrder::XYZ;
  kinematicsTest<EulerJoint>(properties);

  properties.mAxisOrder = EulerJoint::AxisOrder::ZYX;
  kinematicsTest<EulerJoint>(properties);
}
#endif

// 3-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, TRANSLATIONAL_JOINT)
{
  kinematicsTest<TranslationalJoint>();
}
#endif

// 3-dof joint
#ifdef ALL_TESTS
TEST_F(JOINTS, PLANAR_JOINT)
{
  kinematicsTest<PlanarJoint>();
}
#endif

// 6-dof joint
#ifdef ALL_TESTS
#ifndef DART_USE_IDENTITY_JACOBIAN
TEST_F(JOINTS, FREE_JOINT)
{
  kinematicsTest<FreeJoint>();
}
#endif
#endif

//==============================================================================
template <
    void (Joint::*setX)(std::size_t, s_t),
    void (Joint::*setXLowerLimit)(std::size_t, s_t),
    void (Joint::*setXUpperLimit)(std::size_t, s_t)>
void testCommandLimits(dynamics::Joint* joint)
{
  const s_t lower = -5.0;
  const s_t upper = +5.0;
  const s_t mid = 0.5 * (lower + upper);
  const s_t lessThanLower = -10.0;
  const s_t greaterThanUpper = +10.0;

  for (std::size_t i = 0; i < joint->getNumDofs(); ++i)
  {
    (joint->*setXLowerLimit)(i, lower);
    (joint->*setXUpperLimit)(i, upper);

    joint->setCommand(i, mid);
    EXPECT_EQ(joint->getCommand(i), mid);
    (joint->*setX)(i, mid);
    EXPECT_EQ(joint->getCommand(i), mid);

    joint->setCommand(i, lessThanLower);
    EXPECT_EQ(joint->getCommand(i), lower);
    (joint->*setX)(i, lessThanLower);
    EXPECT_EQ(joint->getCommand(i), lessThanLower);

    joint->setCommand(i, greaterThanUpper);
    EXPECT_EQ(joint->getCommand(i), upper);
    (joint->*setX)(i, greaterThanUpper);
    EXPECT_EQ(joint->getCommand(i), greaterThanUpper);
  }
}

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, COMMAND_LIMIT)
{
  simulation::WorldPtr myWorld = utils::SkelParser::readWorld(
      "dart://sample/skel/test/joint_limit_test.skel");
  EXPECT_TRUE(myWorld != nullptr);

  dynamics::SkeletonPtr pendulum = myWorld->getSkeleton("double_pendulum");
  EXPECT_TRUE(pendulum != nullptr);

  auto bodyNodes = pendulum->getBodyNodes();

  for (auto bodyNode : bodyNodes)
  {
    Joint* joint = bodyNode->getParentJoint();

    joint->setActuatorType(Joint::FORCE);
    EXPECT_EQ(joint->getActuatorType(), Joint::FORCE);
    testCommandLimits<
        &Joint::setForce,
        &Joint::setForceLowerLimit,
        &Joint::setForceUpperLimit>(joint);

    joint->setActuatorType(Joint::ACCELERATION);
    EXPECT_EQ(joint->getActuatorType(), Joint::ACCELERATION);
    testCommandLimits<
        &Joint::setAcceleration,
        &Joint::setAccelerationLowerLimit,
        &Joint::setAccelerationUpperLimit>(joint);

    joint->setActuatorType(Joint::VELOCITY);
    EXPECT_EQ(joint->getActuatorType(), Joint::VELOCITY);
    testCommandLimits<
        &Joint::setVelocity,
        &Joint::setVelocityLowerLimit,
        &Joint::setVelocityUpperLimit>(joint);
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, POSITION_LIMIT)
{
  s_t tol = 1e-3;

  simulation::WorldPtr myWorld = utils::SkelParser::readWorld(
      "dart://sample/skel/test/joint_limit_test.skel");
  EXPECT_TRUE(myWorld != nullptr);

  myWorld->setGravity(Eigen::Vector3s(0.0, 0.0, 0.0));

  dynamics::SkeletonPtr pendulum = myWorld->getSkeleton("double_pendulum");
  EXPECT_TRUE(pendulum != nullptr);

  dynamics::Joint* joint0 = pendulum->getJoint("joint0");
  dynamics::Joint* joint1 = pendulum->getJoint("joint1");

  EXPECT_TRUE(joint0 != nullptr);
  EXPECT_TRUE(joint1 != nullptr);

  s_t limit0 = constantsd::pi() / 6.0;
  s_t limit1 = constantsd::pi() / 6.0;

  joint0->setPositionLimitEnforced(true);
  joint0->setPositionLowerLimit(0, -limit0);
  joint0->setPositionUpperLimit(0, limit0);

  joint1->setPositionLimitEnforced(true);
  joint1->setPositionLowerLimit(0, -limit1);
  joint1->setPositionUpperLimit(0, limit1);

#ifndef NDEBUG // Debug mode
  s_t simTime = 0.2;
#else
  s_t simTime = 2.0;
#endif // ------- Debug mode
  s_t timeStep = myWorld->getTimeStep();
  int nSteps = static_cast<int>(simTime / timeStep);

  // Two seconds with positive control forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, 0.1);
    joint1->setForce(0, 0.1);
    myWorld->step();

    s_t jointPos0 = joint0->getPosition(0);
    s_t jointPos1 = joint1->getPosition(0);

    EXPECT_GE(jointPos0, -limit0 - tol);
    EXPECT_GE(jointPos1, -limit1 - tol);

    EXPECT_LE(jointPos0, limit0 + tol);
    EXPECT_LE(jointPos1, limit1 + tol);
  }

  // Two more seconds with negative control forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, -0.1);
    joint1->setForce(0, -0.1);
    myWorld->step();

    s_t jointPos0 = joint0->getPosition(0);
    s_t jointPos1 = joint1->getPosition(0);

    EXPECT_GE(jointPos0, -limit0 - tol);
    EXPECT_GE(jointPos1, -limit1 - tol);

    EXPECT_LE(jointPos0, limit0 + tol);
    EXPECT_LE(jointPos1, limit1 + tol);
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, JOINT_LIMITS)
{
  simulation::WorldPtr myWorld = utils::SkelParser::readWorld(
      "dart://sample/skel/test/joint_limit_test.skel");
  EXPECT_TRUE(myWorld != nullptr);

  myWorld->setGravity(Eigen::Vector3s(0.0, 0.0, 0.0));

  dynamics::SkeletonPtr pendulum = myWorld->getSkeleton("double_pendulum");
  EXPECT_TRUE(pendulum != nullptr);

  dynamics::Joint* joint0 = pendulum->getJoint("joint0");

  EXPECT_TRUE(joint0 != nullptr);

  s_t limit = constantsd::pi() / 6.0;
  Eigen::VectorXs limits = Eigen::VectorXs::Constant(1, constantsd::pi() / 2.0);

  joint0->setPositionLowerLimit(0, -limit);
  joint0->setPositionUpperLimit(0, limit);
  EXPECT_EQ(
      joint0->getPositionLowerLimits(), Eigen::VectorXs::Constant(1, -limit));
  EXPECT_EQ(
      joint0->getPositionUpperLimits(), Eigen::VectorXs::Constant(1, limit));

  joint0->setPositionLowerLimits(-limits);
  joint0->setPositionUpperLimits(limits);
  EXPECT_EQ(joint0->getPositionLowerLimits(), -limits);
  EXPECT_EQ(joint0->getPositionUpperLimits(), limits);

  joint0->setVelocityLowerLimit(0, -limit);
  joint0->setVelocityUpperLimit(0, limit);
  EXPECT_EQ(
      joint0->getVelocityLowerLimits(), Eigen::VectorXs::Constant(1, -limit));
  EXPECT_EQ(
      joint0->getVelocityUpperLimits(), Eigen::VectorXs::Constant(1, limit));

  joint0->setVelocityLowerLimits(-limits);
  joint0->setVelocityUpperLimits(limits);
  EXPECT_EQ(joint0->getVelocityLowerLimits(), -limits);
  EXPECT_EQ(joint0->getVelocityUpperLimits(), limits);

  joint0->setAccelerationLowerLimit(0, -limit);
  joint0->setAccelerationUpperLimit(0, limit);
  EXPECT_EQ(
      joint0->getAccelerationLowerLimits(),
      Eigen::VectorXs::Constant(1, -limit));
  EXPECT_EQ(
      joint0->getAccelerationUpperLimits(),
      Eigen::VectorXs::Constant(1, limit));

  joint0->setAccelerationLowerLimits(-limits);
  joint0->setAccelerationUpperLimits(limits);
  EXPECT_EQ(joint0->getAccelerationLowerLimits(), -limits);
  EXPECT_EQ(joint0->getAccelerationUpperLimits(), limits);

  joint0->setForceLowerLimit(0, -limit);
  joint0->setForceUpperLimit(0, limit);
  EXPECT_EQ(
      joint0->getForceLowerLimits(), Eigen::VectorXs::Constant(1, -limit));
  EXPECT_EQ(joint0->getForceUpperLimits(), Eigen::VectorXs::Constant(1, limit));

  joint0->setForceLowerLimits(-limits);
  joint0->setForceUpperLimits(limits);
  EXPECT_EQ(joint0->getForceLowerLimits(), -limits);
  EXPECT_EQ(joint0->getForceUpperLimits(), limits);
}
#endif

//==============================================================================
void testJointCoulombFrictionForce(s_t _timeStep)
{
  s_t tol = 1e-9;

  simulation::WorldPtr myWorld = utils::SkelParser::readWorld(
      "dart://sample/skel/test/joint_friction_test.skel");
  EXPECT_TRUE(myWorld != nullptr);

  myWorld->setGravity(Eigen::Vector3s(0.0, 0.0, 0.0));
  myWorld->setTimeStep(_timeStep);

  dynamics::SkeletonPtr pendulum = myWorld->getSkeleton("double_pendulum");
  EXPECT_TRUE(pendulum != nullptr);
  pendulum->disableSelfCollisionCheck();

  dynamics::Joint* joint0 = pendulum->getJoint("joint0");
  dynamics::Joint* joint1 = pendulum->getJoint("joint1");

  EXPECT_TRUE(joint0 != nullptr);
  EXPECT_TRUE(joint1 != nullptr);

  s_t frictionForce = 5.0;

  joint0->setPositionLimitEnforced(false);
  joint1->setPositionLimitEnforced(false);

  joint0->setCoulombFriction(0, frictionForce);
  joint1->setCoulombFriction(0, frictionForce);

  EXPECT_EQ(joint0->getCoulombFriction(0), frictionForce);
  EXPECT_EQ(joint1->getCoulombFriction(0), frictionForce);

#ifndef NDEBUG // Debug mode
  s_t simTime = 0.2;
#else
  s_t simTime = 2.0;
#endif // ------- Debug mode
  s_t timeStep = myWorld->getTimeStep();
  int nSteps = static_cast<int>(simTime / timeStep);

  // Two seconds with lower control forces than the friction
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, +4.9);
    joint1->setForce(0, +4.9);
    myWorld->step();

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_NEAR(
        static_cast<double>(jointVel0),
        static_cast<double>(0.0),
        static_cast<double>(tol));
    EXPECT_NEAR(
        static_cast<double>(jointVel1),
        static_cast<double>(0.0),
        static_cast<double>(tol));
  }

  // Another two seconds with lower control forces than the friction forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, -4.9);
    joint1->setForce(0, -4.9);
    myWorld->step();

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_NEAR(
        static_cast<double>(jointVel0),
        static_cast<double>(0.0),
        static_cast<double>(tol));
    EXPECT_NEAR(
        static_cast<double>(jointVel1),
        static_cast<double>(0.0),
        static_cast<double>(tol));
  }

  // Another two seconds with higher control forces than the friction forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, 10.0);
    joint1->setForce(0, 10.0);
    myWorld->step();

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_GE(abs(jointVel0), 0.0);
    EXPECT_GE(abs(jointVel1), 0.0);
  }

  // Spend 20 sec waiting the joints to stop
  for (int i = 0; i < nSteps * 10; i++)
  {
    myWorld->step();
  }
  s_t jointVel0 = joint0->getVelocity(0);
  s_t jointVel1 = joint1->getVelocity(0);

  EXPECT_NEAR(
      static_cast<double>(jointVel0),
      static_cast<double>(0.0),
      static_cast<double>(tol));
  EXPECT_NEAR(
      static_cast<double>(jointVel1),
      static_cast<double>(0.0),
      static_cast<double>(tol));

  // Another two seconds with lower control forces than the friction forces
  // and expect the joints to stop
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, 4.9);
    joint1->setForce(0, 4.9);
    myWorld->step();

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_NEAR(
        static_cast<double>(jointVel0),
        static_cast<double>(0.0),
        static_cast<double>(tol));
    EXPECT_NEAR(
        static_cast<double>(jointVel1),
        static_cast<double>(0.0),
        static_cast<double>(tol));
  }
}

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, JOINT_COULOMB_FRICTION)
{
  std::array<s_t, 3> timeSteps;
  timeSteps[0] = 1e-2;
  timeSteps[1] = 1e-3;
  timeSteps[2] = 1e-4;

  for (auto timeStep : timeSteps)
    testJointCoulombFrictionForce(timeStep);
}
#endif

//==============================================================================
SkeletonPtr createPendulum(Joint::ActuatorType actType)
{
  using namespace dart::math::suffixes;

  Vector3s dim(1, 1, 1);
  Vector3s offset(0, 0, 2);

  SkeletonPtr pendulum = createNLinkPendulum(1, dim, DOF_ROLL, offset);
  EXPECT_NE(pendulum, nullptr);

  pendulum->disableSelfCollisionCheck();

  for (std::size_t i = 0; i < pendulum->getNumBodyNodes(); ++i)
  {
    auto bodyNode = pendulum->getBodyNode(i);
    bodyNode->removeAllShapeNodesWith<CollisionAspect>();
  }

  // Joint common setting
  dynamics::Joint* joint = pendulum->getJoint(0);
  EXPECT_NE(joint, nullptr);

  joint->setActuatorType(actType);
  joint->setPosition(0, toRadian(static_cast<s_t>(90.0)));
  joint->setDampingCoefficient(0, 0.0);
  joint->setSpringStiffness(0, 0.0);
  joint->setPositionLimitEnforced(true);
  joint->setCoulombFriction(0, 0.0);

  return pendulum;
}

//==============================================================================
void testServoMotor()
{
  using namespace dart::math::suffixes;

  std::size_t numPendulums = 7;
  s_t timestep = 1e-3;
  s_t tol = 1e-9;
  s_t posUpperLimit = toRadian(static_cast<s_t>(90.0));
  s_t posLowerLimit = toRadian(static_cast<s_t>(45.0));
  s_t sufficient_force = 1e+5;
  s_t insufficient_force = 1e-1;

  // World
  simulation::WorldPtr world = simulation::World::create();
  EXPECT_TRUE(world != nullptr);

  world->setGravity(Eigen::Vector3s(0, 0, -9.81));
  world->setTimeStep(timestep);

  // Each pendulum has servo motor in the joint but also have different
  // expectations depending on the property values.
  //
  // Pendulum0:
  //  - Condition: Zero desired velocity and sufficient servo motor force limits
  //  - Expectation: The desired velocity should be achieved.
  // Pendulum 1:
  //  - Condition: Nonzero desired velocity and sufficient servo motor force
  //               limits
  //  - Expectation: The desired velocity should be achieved.
  // Pendulum 2:
  //  - Condition: Nonzero desired velocity and insufficient servo motor force
  //               limits
  //  - Expectation: The desired velocity can be achieve or not. But when it's
  //                 not achieved, the servo motor force is reached to the lower
  //                 or upper limit.
  // Pendulum 3:
  //  - Condition: Nonzero desired velocity, finite servo motor force limits,
  //               and position limits
  //  - Expectation: The desired velocity should be achieved unless joint
  //                 position is reached to lower or upper position limit.
  // Pendulum 4:
  //  - Condition: Nonzero desired velocity, infinite servo motor force limits,
  //               and position limits
  //  - Expectation: Same as the Pendulum 3's expectation.
  // Pendulum 5:
  //  - Condition: Nonzero desired velocity, finite servo motor force limits,
  //               and infinite Coulomb friction
  //  - Expectation: The the pendulum shouldn't move at all due to the infinite
  //                 friction.
  // Pendulum 6:
  //  - Condition: Nonzero desired velocity, infinite servo motor force limits,
  //               and infinite Coulomb friction
  //  - Expectation: The the pendulum shouldn't move at all due to the friction.
  //    TODO(JS): Should a servo motor dominent Coulomb friction in this case?

  std::vector<SkeletonPtr> pendulums(numPendulums);
  std::vector<JointPtr> joints(numPendulums);
  for (std::size_t i = 0; i < numPendulums; ++i)
  {
    pendulums[i] = createPendulum(Joint::SERVO);
    joints[i] = pendulums[i]->getJoint(0);
  }

  joints[0]->setForceUpperLimit(0, sufficient_force);
  joints[0]->setForceLowerLimit(0, -sufficient_force);

  joints[1]->setForceUpperLimit(0, sufficient_force);
  joints[1]->setForceLowerLimit(0, -sufficient_force);

  joints[2]->setForceUpperLimit(0, insufficient_force);
  joints[2]->setForceLowerLimit(0, -insufficient_force);

  joints[3]->setForceUpperLimit(0, sufficient_force);
  joints[3]->setForceLowerLimit(0, -sufficient_force);
  joints[3]->setPositionUpperLimit(0, posUpperLimit);
  joints[3]->setPositionLowerLimit(0, posLowerLimit);

  joints[4]->setForceUpperLimit(0, constantsd::inf());
  joints[4]->setForceLowerLimit(0, -constantsd::inf());
  joints[4]->setPositionUpperLimit(0, posUpperLimit);
  joints[4]->setPositionLowerLimit(0, posLowerLimit);

  joints[5]->setForceUpperLimit(0, sufficient_force);
  joints[5]->setForceLowerLimit(0, -sufficient_force);
  joints[5]->setCoulombFriction(0, constantsd::inf());

  joints[6]->setForceUpperLimit(0, constantsd::inf());
  joints[6]->setForceLowerLimit(0, -constantsd::inf());
  joints[6]->setCoulombFriction(0, constantsd::inf());

  for (auto pendulum : pendulums)
    world->addSkeleton(pendulum);

#ifndef NDEBUG // Debug mode
  s_t simTime = 0.2;
#else
  s_t simTime = 2.0;
#endif // ------- Debug mode
  s_t timeStep = world->getTimeStep();
  int nSteps = static_cast<int>(simTime / timeStep);

  // Two seconds with lower control forces than the friction
  for (int i = 0; i < nSteps; i++)
  {
    const s_t expected_vel = sin(world->getTime());

    joints[0]->setCommand(0, 0.0);
    joints[1]->setCommand(0, expected_vel);
    joints[2]->setCommand(0, expected_vel);
    joints[3]->setCommand(0, expected_vel);
    joints[4]->setCommand(0, expected_vel);
    joints[5]->setCommand(0, expected_vel);
    joints[6]->setCommand(0, expected_vel);

    world->step();

    std::vector<s_t> jointVels(numPendulums);
    for (std::size_t j = 0; j < numPendulums; ++j)
      jointVels[j] = joints[j]->getVelocity(0);

    EXPECT_NEAR(
        static_cast<double>(jointVels[0]),
        static_cast<double>(0.0),
        static_cast<double>(tol));
    EXPECT_NEAR(
        static_cast<double>(jointVels[1]),
        static_cast<double>(expected_vel),
        static_cast<double>(tol));
    bool result2 = abs(jointVels[2] - expected_vel) < tol
                   || abs(joints[2]->getConstraintImpulse(0) / timeStep
                          - insufficient_force)
                          < tol
                   || abs(joints[2]->getConstraintImpulse(0) / timeStep
                          + insufficient_force)
                          < tol;
    EXPECT_TRUE(result2);
    EXPECT_LE(
        joints[3]->getPosition(0), posUpperLimit + expected_vel * timeStep);
    EXPECT_GE(
        joints[3]->getPosition(0), posLowerLimit - expected_vel * timeStep);
    // EXPECT_LE(joints[4]->getPosition(0),
    //     posUpperLimit + expected_vel * timeStep);
    // EXPECT_GE(joints[4]->getPosition(0),
    //     posLowerLimit - expected_vel * timeStep);
    // TODO(JS): Position limits and servo motor with infinite force limits
    // doesn't work together because they compete against each other to achieve
    // different joint velocities with their infinit force limits. In this case,
    // the position limit constraint should dominent the servo motor constraint.
    EXPECT_NEAR(
        static_cast<double>(jointVels[5]),
        static_cast<double>(0.0),
        static_cast<double>(tol * 1e+2));
    // EXPECT_NEAR(jointVels[6], 0.0, tol * 1e+2);
    // TODO(JS): Servo motor with infinite force limits and infinite Coulomb
    // friction doesn't work because they compete against each other to achieve
    // different joint velocities with their infinit force limits. In this case,
    // the friction constraints should dominent the servo motor constraints.
  }
}

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, SERVO_MOTOR)
{
  testServoMotor();
}
#endif

//==============================================================================
void testMimicJoint()
{
  using namespace dart::math::suffixes;

  s_t timestep = 1e-3;
  s_t tol = 1e-9;
  s_t tolPos = 1e-3;
  s_t sufficient_force = 1e+5;

  // World
  simulation::WorldPtr world = simulation::World::create();
  EXPECT_TRUE(world != nullptr);

  world->setGravity(Eigen::Vector3s(0, 0, -9.81));
  world->setTimeStep(timestep);

  Vector3s dim(1, 1, 1);
  Vector3s offset(0, 0, 2);

  SkeletonPtr pendulum = createNLinkPendulum(2, dim, DOF_ROLL, offset);
  EXPECT_NE(pendulum, nullptr);

  pendulum->disableSelfCollisionCheck();

  for (std::size_t i = 0; i < pendulum->getNumBodyNodes(); ++i)
  {
    auto bodyNode = pendulum->getBodyNode(i);
    bodyNode->removeAllShapeNodesWith<CollisionAspect>();
  }

  std::vector<JointPtr> joints(2);

  for (std::size_t i = 0; i < pendulum->getNumBodyNodes(); ++i)
  {
    dynamics::Joint* joint = pendulum->getJoint(i);
    EXPECT_NE(joint, nullptr);

    joint->setActuatorType(Joint::SERVO);
    joint->setPosition(0, toRadian(static_cast<s_t>(90.0)));
    joint->setDampingCoefficient(0, 0.0);
    joint->setSpringStiffness(0, 0.0);
    joint->setPositionLimitEnforced(true);
    joint->setCoulombFriction(0, 0.0);

    joints[i] = joint;
  }

  joints[0]->setForceUpperLimit(0, sufficient_force);
  joints[0]->setForceLowerLimit(0, -sufficient_force);

  joints[1]->setForceUpperLimit(0, sufficient_force);
  joints[1]->setForceLowerLimit(0, -sufficient_force);

  // Second joint mimics the first one
  joints[1]->setActuatorType(Joint::MIMIC);
  joints[1]->setMimicJoint(joints[0], 1., 0.);

  world->addSkeleton(pendulum, true);
  // We need to use implicit integration here to keep the results stable enough
  // for the numerical tolerances to be satisfied.
  world->setParallelVelocityAndPositionUpdates(false);

#ifndef NDEBUG // Debug mode
  s_t simTime = 0.2;
#else
  s_t simTime = 2.0;
#endif // ------- Debug mode
  s_t timeStep = world->getTimeStep();
  int nSteps = static_cast<int>(simTime / timeStep);

  // Two seconds with lower control forces than the friction
  for (int i = 0; i < nSteps; i++)
  {
    const s_t expected_vel = sin(world->getTime());

    joints[0]->setCommand(0, expected_vel);

    world->step();

    // Check if the first joint achieved the velocity at each time-step
    EXPECT_NEAR(
        static_cast<double>(joints[0]->getVelocity(0)),
        static_cast<double>(expected_vel),
        static_cast<double>(tol));

    // Check if the mimic joint follows the "master" joint
    EXPECT_NEAR(
        static_cast<double>(joints[0]->getPosition(0)),
        static_cast<double>(joints[1]->getPosition(0)),
        static_cast<double>(tolPos));
  }

  // In the end, check once more if the mimic joint followed the "master" joint
  EXPECT_NEAR(
      static_cast<double>(joints[0]->getPosition(0)),
      static_cast<double>(joints[1]->getPosition(0)),
      static_cast<double>(tolPos));
}

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, MIMIC_JOINT)
{
  testMimicJoint();
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, JOINT_COULOMB_FRICTION_AND_POSITION_LIMIT)
{
  const s_t timeStep = 1e-3;
  const s_t tol = 1e-2;

  simulation::WorldPtr myWorld = utils::SkelParser::readWorld(
      "dart://sample/skel/test/joint_friction_test.skel");
  EXPECT_TRUE(myWorld != nullptr);

  myWorld->setGravity(Eigen::Vector3s(0.0, 0.0, 0.0));
  myWorld->setTimeStep(timeStep);
  // Use implicit integration for precision for this test
  myWorld->setParallelVelocityAndPositionUpdates(false);

  dynamics::SkeletonPtr pendulum = myWorld->getSkeleton("double_pendulum");
  EXPECT_TRUE(pendulum != nullptr);
  pendulum->disableSelfCollisionCheck();

  dynamics::Joint* joint0 = pendulum->getJoint("joint0");
  dynamics::Joint* joint1 = pendulum->getJoint("joint1");

  EXPECT_TRUE(joint0 != nullptr);
  EXPECT_TRUE(joint1 != nullptr);

  s_t frictionForce = 5.0;

  joint0->setPositionLimitEnforced(true);
  joint1->setPositionLimitEnforced(true);

  const s_t ll = -constantsd::pi() / 12.0; // -15 degree
  const s_t ul = +constantsd::pi() / 12.0; // +15 degree

  std::size_t dof0 = joint0->getNumDofs();
  for (std::size_t i = 0; i < dof0; ++i)
  {
    joint0->setPosition(i, 0.0);
    joint0->setPosition(i, 0.0);
    joint0->setPositionLowerLimit(i, ll);
    joint0->setPositionUpperLimit(i, ul);
  }

  std::size_t dof1 = joint1->getNumDofs();
  for (std::size_t i = 0; i < dof1; ++i)
  {
    joint1->setPosition(i, 0.0);
    joint1->setPosition(i, 0.0);
    joint1->setPositionLowerLimit(i, ll);
    joint1->setPositionUpperLimit(i, ul);
  }

  joint0->setCoulombFriction(0, frictionForce);
  joint1->setCoulombFriction(0, frictionForce);

  EXPECT_EQ(joint0->getCoulombFriction(0), frictionForce);
  EXPECT_EQ(joint1->getCoulombFriction(0), frictionForce);

#ifndef NDEBUG // Debug mode
  s_t simTime = 0.2;
#else
  s_t simTime = 2.0;
#endif // ------- Debug mode
  int nSteps = static_cast<int>(simTime / timeStep);

  // First two seconds rotating in positive direction with higher control forces
  // than the friction forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, 100.0);
    joint1->setForce(0, 100.0);
    myWorld->step();

    s_t jointPos0 = joint0->getPosition(0);
    s_t jointPos1 = joint1->getPosition(0);

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_GE(abs(jointVel0), 0.0);
    EXPECT_GE(abs(jointVel1), 0.0);

    EXPECT_GE(jointPos0, ll - tol);
    EXPECT_LE(jointPos0, ul + tol);

    EXPECT_GE(jointPos1, ll - tol);
    EXPECT_LE(jointPos1, ul + tol);
  }

  // Another two seconds rotating in negative direction with higher control
  // forces than the friction forces
  for (int i = 0; i < nSteps; i++)
  {
    joint0->setForce(0, -100.0);
    joint1->setForce(0, -100.0);
    myWorld->step();

    s_t jointPos0 = joint0->getPosition(0);
    s_t jointPos1 = joint1->getPosition(0);

    s_t jointVel0 = joint0->getVelocity(0);
    s_t jointVel1 = joint1->getVelocity(0);

    EXPECT_GE(abs(jointVel0), 0.0);
    EXPECT_GE(abs(jointVel1), 0.0);

    EXPECT_GE(jointPos0, ll - tol);
    EXPECT_LE(jointPos0, ul + tol);

    EXPECT_GE(jointPos1, ll - tol);
    EXPECT_LE(jointPos1, ul + tol);
  }
}
#endif

//==============================================================================
template <int N>
Eigen::Matrix<s_t, N, 1> random_vec(s_t limit = 100)
{
  Eigen::Matrix<s_t, N, 1> v;
  for (std::size_t i = 0; i < N; ++i)
    v[i] = math::Random::uniform(-abs(limit), abs(limit));
  return v;
}

//==============================================================================
Eigen::Isometry3s random_transform(
    s_t translation_limit = 100, s_t rotation_limit = 2 * M_PI)
{
  Eigen::Vector3s r = random_vec<3>(translation_limit);
  Eigen::Vector3s theta = random_vec<3>(rotation_limit);

  Eigen::Isometry3s tf;
  tf.setIdentity();
  tf.translate(r);

  if (theta.norm() > 0)
    tf.rotate(Eigen::AngleAxis_s(theta.norm(), theta.normalized()));

  return tf;
}

//==============================================================================
Eigen::Isometry3s predict_joint_transform(
    Joint* joint, const Eigen::Isometry3s& joint_tf)
{
  return joint->getTransformFromParentBodyNode() * joint_tf
         * joint->getTransformFromChildBodyNode().inverse();
}

//==============================================================================
Eigen::Isometry3s get_relative_transform(BodyNode* bn, BodyNode* relativeTo)
{
  return relativeTo->getTransform().inverse() * bn->getTransform();
}

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, CONVENIENCE_FUNCTIONS)
{
  SkeletonPtr skel = Skeleton::create();

  std::pair<Joint*, BodyNode*> pair;

  pair = skel->createJointAndBodyNodePair<WeldJoint>();
  BodyNode* root = pair.second;

  // -- set up the FreeJoint
  std::pair<FreeJoint*, BodyNode*> freepair
      = root->createChildJointAndBodyNodePair<FreeJoint>();
  FreeJoint* freejoint = freepair.first;
  BodyNode* freejoint_bn = freepair.second;

  freejoint->setTransformFromParentBodyNode(random_transform());
  freejoint->setTransformFromChildBodyNode(random_transform());

  // -- set up the EulerJoint
  std::pair<EulerJoint*, BodyNode*> eulerpair
      = root->createChildJointAndBodyNodePair<EulerJoint>();
  EulerJoint* eulerjoint = eulerpair.first;
  BodyNode* eulerjoint_bn = eulerpair.second;

  eulerjoint->setTransformFromParentBodyNode(random_transform());
  eulerjoint->setTransformFromChildBodyNode(random_transform());

  // -- set up the BallJoint
  std::pair<BallJoint*, BodyNode*> ballpair
      = root->createChildJointAndBodyNodePair<BallJoint>();
  BallJoint* balljoint = ballpair.first;
  BodyNode* balljoint_bn = ballpair.second;

  balljoint->setTransformFromParentBodyNode(random_transform());
  balljoint->setTransformFromChildBodyNode(random_transform());

  // Test a hundred times
  for (std::size_t n = 0; n < 100; ++n)
  {
    // -- convert transforms to positions and then positions back to transforms
    Eigen::Isometry3s desired_freejoint_tf = random_transform();
    freejoint->setPositions(
        FreeJoint::convertToPositions(desired_freejoint_tf));
    Eigen::Isometry3s actual_freejoint_tf
        = FreeJoint::convertToTransform(freejoint->getPositions());

    Eigen::Isometry3s desired_eulerjoint_tf = random_transform();
    desired_eulerjoint_tf.translation() = Eigen::Vector3s::Zero();
    eulerjoint->setPositions(
        eulerjoint->convertToPositions(desired_eulerjoint_tf.linear()));
    Eigen::Isometry3s actual_eulerjoint_tf
        = eulerjoint->convertToTransform(eulerjoint->getPositions());

    Eigen::Isometry3s desired_balljoint_tf = random_transform();
    desired_balljoint_tf.translation() = Eigen::Vector3s::Zero();
    balljoint->setPositions(
        BallJoint::convertToPositions(desired_balljoint_tf.linear()));
    Eigen::Isometry3s actual_balljoint_tf
        = BallJoint::convertToTransform(balljoint->getPositions());

    // -- collect everything so we can cycle through the tests
    std::vector<Joint*> joints;
    std::vector<BodyNode*> bns;
    common::aligned_vector<Eigen::Isometry3s> desired_tfs;
    common::aligned_vector<Eigen::Isometry3s> actual_tfs;

    joints.push_back(freejoint);
    bns.push_back(freejoint_bn);
    desired_tfs.push_back(desired_freejoint_tf);
    actual_tfs.push_back(actual_freejoint_tf);

    joints.push_back(eulerjoint);
    bns.push_back(eulerjoint_bn);
    desired_tfs.push_back(desired_eulerjoint_tf);
    actual_tfs.push_back(actual_eulerjoint_tf);

    joints.push_back(balljoint);
    bns.push_back(balljoint_bn);
    desired_tfs.push_back(desired_balljoint_tf);
    actual_tfs.push_back(actual_balljoint_tf);

    for (std::size_t i = 0; i < joints.size(); ++i)
    {
      Joint* joint = joints[i];
      BodyNode* bn = bns[i];
      Eigen::Isometry3s tf = desired_tfs[i];

      bool check_transform_conversion = equals(
          predict_joint_transform(joint, tf).matrix(),
          get_relative_transform(bn, bn->getParentBodyNode()).matrix());
      EXPECT_TRUE(check_transform_conversion);

      if (!check_transform_conversion)
      {
        std::cout << "[" << joint->getName() << " Failed]\n";
        std::cout
            << "Predicted:\n"
            << predict_joint_transform(joint, tf).matrix() << "\n\nActual:\n"
            << get_relative_transform(bn, bn->getParentBodyNode()).matrix()
            << "\n\n";
      }

      bool check_full_cycle
          = equals(desired_tfs[i].matrix(), actual_tfs[i].matrix());
      EXPECT_TRUE(check_full_cycle);

      if (!check_full_cycle)
      {
        std::cout << "[" << joint->getName() << " Failed]\n";
        std::cout << "Desired:\n"
                  << desired_tfs[i].matrix() << "\n\nActual:\n"
                  << actual_tfs[i].matrix() << "\n\n";
      }
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST_F(JOINTS, FREE_JOINT_RELATIVE_TRANSFORM_VELOCITY_ACCELERATION)
{
  const std::size_t numTests = 50;

  // Generate random reference frames
  randomizeRefFrames();
  auto refFrames = getFrames();

  // Generate random relative frames
  randomizeRefFrames();
  auto relFrames = getFrames();

  //-- Build a skeleton that contains two FreeJoints and two BodyNodes
  SkeletonPtr skel = Skeleton::create();

  auto pair = skel->createJointAndBodyNodePair<FreeJoint>();
  FreeJoint* rootJoint = pair.first;
  BodyNode* rootBodyNode = pair.second;
  rootJoint->setRelativeTransform(random_transform());
  rootJoint->setRelativeSpatialVelocity(random_vec<6>());
  rootJoint->setRelativeSpatialAcceleration(random_vec<6>());
  rootJoint->setTransformFromParentBodyNode(random_transform());
  rootJoint->setTransformFromChildBodyNode(random_transform());

  pair = rootBodyNode->createChildJointAndBodyNodePair<FreeJoint>();
  FreeJoint* joint1 = pair.first;
  BodyNode* bodyNode1 = pair.second;
  joint1->setTransformFromParentBodyNode(random_transform());
  joint1->setTransformFromChildBodyNode(random_transform());

  //-- Actual terms
  Eigen::Isometry3s actualTf;
  Eigen::Vector6s actualVel;
  Eigen::Vector6s actualAcc;

  Eigen::Vector3s actualLinVel;
  Eigen::Vector3s actualAngVel;
  Eigen::Vector3s actualLinAcc;
  Eigen::Vector3s actualAngAcc;

  Eigen::Vector3s oldLinVel;
  Eigen::Vector3s oldAngVel;
  Eigen::Vector3s oldLinAcc;
  Eigen::Vector3s oldAngAcc;

  //-- Test
  for (std::size_t i = 0; i < numTests; ++i)
  {
    const Eigen::Isometry3s desiredTf = random_transform();
    const Eigen::Vector6s desiredVel = random_vec<6>();
    const Eigen::Vector6s desiredAcc = random_vec<6>();
    const Eigen::Vector3s desiredLinVel = random_vec<3>();
    const Eigen::Vector3s desiredAngVel = random_vec<3>();
    const Eigen::Vector3s desiredLinAcc = random_vec<3>();
    const Eigen::Vector3s desiredAngAcc = random_vec<3>();

    //-- Relative transformation

    joint1->setRelativeTransform(desiredTf);
    actualTf = bodyNode1->getTransform(bodyNode1->getParentBodyNode());
    EXPECT_TRUE(equals(desiredTf.matrix(), actualTf.matrix()));

    for (auto relativeTo : relFrames)
    {
      joint1->setTransform(desiredTf, relativeTo);
      actualTf = bodyNode1->getTransform(relativeTo);
      EXPECT_TRUE(equals(desiredTf.matrix(), actualTf.matrix()));
    }

    //-- Relative spatial velocity

    joint1->setRelativeSpatialVelocity(desiredVel);
    actualVel = bodyNode1->getSpatialVelocity(
        bodyNode1->getParentBodyNode(), bodyNode1);

    EXPECT_TRUE(equals(desiredVel, actualVel));

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialVelocity(desiredVel, relativeTo, inCoordinatesOf);
        actualVel = bodyNode1->getSpatialVelocity(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(desiredVel, actualVel));
      }
    }

    //-- Relative classic linear velocity

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialVelocity(desiredVel, relativeTo, inCoordinatesOf);
        oldAngVel = bodyNode1->getAngularVelocity(relativeTo, inCoordinatesOf);
        joint1->setLinearVelocity(desiredLinVel, relativeTo, inCoordinatesOf);

        actualLinVel
            = bodyNode1->getLinearVelocity(relativeTo, inCoordinatesOf);
        actualAngVel
            = bodyNode1->getAngularVelocity(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(desiredLinVel, actualLinVel));
        EXPECT_TRUE(equals(oldAngVel, actualAngVel));
      }
    }

    //-- Relative classic angular velocity

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialVelocity(desiredVel, relativeTo, inCoordinatesOf);
        oldLinVel = bodyNode1->getLinearVelocity(relativeTo, inCoordinatesOf);
        joint1->setAngularVelocity(desiredAngVel, relativeTo, inCoordinatesOf);

        actualLinVel
            = bodyNode1->getLinearVelocity(relativeTo, inCoordinatesOf);
        actualAngVel
            = bodyNode1->getAngularVelocity(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(oldLinVel, actualLinVel));
        EXPECT_TRUE(equals(desiredAngVel, actualAngVel));
      }
    }

    //-- Relative spatial acceleration

    joint1->setRelativeSpatialAcceleration(desiredAcc);
    actualAcc = bodyNode1->getSpatialAcceleration(
        bodyNode1->getParentBodyNode(), bodyNode1);

    EXPECT_TRUE(equals(desiredAcc, actualAcc));

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialAcceleration(desiredAcc, relativeTo, inCoordinatesOf);
        actualAcc
            = bodyNode1->getSpatialAcceleration(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(desiredAcc, actualAcc));
      }
    }

    //-- Relative transform, spatial velocity, and spatial acceleration
    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialMotion(
            &desiredTf,
            relativeTo,
            &desiredVel,
            relativeTo,
            inCoordinatesOf,
            &desiredAcc,
            relativeTo,
            inCoordinatesOf);
        actualTf = bodyNode1->getTransform(relativeTo);
        actualVel = bodyNode1->getSpatialVelocity(relativeTo, inCoordinatesOf);
        actualAcc
            = bodyNode1->getSpatialAcceleration(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(desiredTf.matrix(), actualTf.matrix()));
        EXPECT_TRUE(equals(desiredVel, actualVel));
        EXPECT_TRUE(equals(desiredAcc, actualAcc));
      }
    }

    //-- Relative classic linear acceleration

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialAcceleration(desiredAcc, relativeTo, inCoordinatesOf);
        oldAngAcc
            = bodyNode1->getAngularAcceleration(relativeTo, inCoordinatesOf);
        joint1->setLinearAcceleration(
            desiredLinAcc, relativeTo, inCoordinatesOf);

        actualLinAcc
            = bodyNode1->getLinearAcceleration(relativeTo, inCoordinatesOf);
        actualAngAcc
            = bodyNode1->getAngularAcceleration(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(desiredLinAcc, actualLinAcc));
        EXPECT_TRUE(equals(oldAngAcc, actualAngAcc));
      }
    }

    //-- Relative classic angular acceleration

    for (auto relativeTo : relFrames)
    {
      for (auto inCoordinatesOf : refFrames)
      {
        joint1->setSpatialAcceleration(desiredAcc, relativeTo, inCoordinatesOf);
        oldLinAcc
            = bodyNode1->getLinearAcceleration(relativeTo, inCoordinatesOf);
        joint1->setAngularAcceleration(
            desiredAngAcc, relativeTo, inCoordinatesOf);

        actualLinAcc
            = bodyNode1->getLinearAcceleration(relativeTo, inCoordinatesOf);
        actualAngAcc
            = bodyNode1->getAngularAcceleration(relativeTo, inCoordinatesOf);

        EXPECT_TRUE(equals(oldLinAcc, actualLinAcc));
        EXPECT_TRUE(equals(desiredAngAcc, actualAngAcc));
      }
    }
  }
}
#endif
