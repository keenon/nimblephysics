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

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "dart/realtime/ControlLog.hpp"
#include "dart/realtime/ObservationLog.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"
#include "dart/realtime/VectorLog.hpp"
#include "dart/simulation/World.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace realtime;

#ifdef ALL_TESTS
TEST(REALTIME, VECTOR_LOG)
{
  int dim = 2;
  VectorLog log = VectorLog(dim);

  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);

  Eigen::MatrixXs expected = Eigen::MatrixXs::Ones(dim, 20);
  expected.block(0, 10, 2, 10) *= 2;
  Eigen::MatrixXs actual = log.getValues(0L, 20, 1L);

  if (!equals(expected, actual))
  {
    std::cout << "Expected: " << std::endl << expected << std::endl;
    std::cout << "Actual: " << std::endl << actual << std::endl;
  }

  EXPECT_TRUE(equals(expected, actual));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, VECTOR_LOG_EXTEND)
{
  int dim = 2;
  VectorLog log = VectorLog(dim);

  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);

  Eigen::MatrixXs expected = Eigen::MatrixXs::Ones(dim, 20);
  expected.block(0, 2, 2, 18) *= 2;
  Eigen::MatrixXs actual = log.getValues(8L, 20, 1L);

  if (!equals(expected, actual))
  {
    std::cout << "Expected: " << std::endl << expected << std::endl;
    std::cout << "Actual: " << std::endl << actual << std::endl;
  }

  EXPECT_TRUE(equals(expected, actual));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, VECTOR_LOG_AFTER)
{
  int dim = 2;
  VectorLog log = VectorLog(dim);

  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);

  Eigen::MatrixXs expected = Eigen::MatrixXs::Ones(dim, 20);
  expected *= 2;
  Eigen::MatrixXs actual = log.getValues(18L, 20, 1L);

  if (!equals(expected, actual))
  {
    std::cout << "Expected: " << std::endl << expected << std::endl;
    std::cout << "Actual: " << std::endl << actual << std::endl;
  }

  EXPECT_TRUE(equals(expected, actual));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_LOG)
{
  int dim = 2;
  int dt = 5;
  ControlLog log = ControlLog(dim, dt);
  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);

  EXPECT_DOUBLE_EQ(1.0, static_cast<double>(log.get(-3L)(0)));
  EXPECT_DOUBLE_EQ(1.0, static_cast<double>(log.get(3L)(0)));
  EXPECT_DOUBLE_EQ(1.0, static_cast<double>(log.get(7L)(0)));
  EXPECT_DOUBLE_EQ(2.0, static_cast<double>(log.get(10L)(0)));
  EXPECT_DOUBLE_EQ(2.0, static_cast<double>(log.get(24L)(0)));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_LOG_DISCARD_BEFORE)
{
  int dim = 2;
  int dt = 5;
  ControlLog log = ControlLog(dim, dt);
  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(5L, Eigen::VectorXs::Ones(dim) * 3);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);
  log.discardBefore(5L);

  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(-3L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(3L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(7L)(0)));
  EXPECT_DOUBLE_EQ(2.0, static_cast<double>(log.get(24L)(0)));
  EXPECT_DOUBLE_EQ(2.0, static_cast<double>(log.get(10L)(0)));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_LOG_DISCARD_BEFORE_OFF_TIMESTEP)
{
  int dim = 2;
  int dt = 5;
  ControlLog log = ControlLog(dim, dt);
  log.record(0L, Eigen::VectorXs::Ones(dim) * 1);
  log.record(5L, Eigen::VectorXs::Ones(dim) * 3);
  log.record(10L, Eigen::VectorXs::Ones(dim) * 2);
  // This should discard up through 5L, and set last observed to 10L
  log.discardBefore(7L);
  // This should overwrite the 10L slot, because it's not far enough to hit the
  // 15L slot. That won't happen if the discardBefore() didn't set the last
  // observed point correctly.
  log.record(14L, Eigen::VectorXs::Ones(dim) * 3);

  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(-3L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(3L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(7L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(24L)(0)));
  EXPECT_DOUBLE_EQ(3.0, static_cast<double>(log.get(10L)(0)));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_LOG_GET_EMPTY)
{
  int dim = 2;
  int dt = 5;
  ControlLog log = ControlLog(dim, dt);
  EXPECT_DOUBLE_EQ(0.0, static_cast<double>(log.get(-3L)(0)));
  EXPECT_DOUBLE_EQ(0.0, static_cast<double>(log.get(3L)(0)));
  EXPECT_DOUBLE_EQ(0.0, static_cast<double>(log.get(7L)(0)));
  EXPECT_DOUBLE_EQ(0.0, static_cast<double>(log.get(24L)(0)));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  buffer.setControlForcePlan(0L, 0L, Eigen::MatrixXs::Ones(forceDim, steps) * 2);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(25L)(0)), 2.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(49L)(0)), 2.0);
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_OOB)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  buffer.setControlForcePlan(0L, 0L, Eigen::MatrixXs::Ones(forceDim, steps) * 2);
  // This reads off the end, should print a warning and return 0s
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(50L)(0)), 0.0);
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_SCALE_DOWN)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < steps; i++)
  {
    plan.col(i) *= i;
  }
  buffer.setControlForcePlan(0L, 0L, plan);
  buffer.setMillisPerStep(10);
  buffer.setNumSteps(5);

  // Read off the lower resolution, should now jump by whole numbers
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(0 * dt)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(1 * dt)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(2 * dt)(0)), 2);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(3 * dt)(0)), 2);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(4 * dt)(0)), 4);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(5 * dt)(0)), 4);
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_SCALE_UP)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < steps; i++)
  {
    plan.col(i) *= i;
  }
  buffer.setControlForcePlan(0L, 0L, plan);
  buffer.setMillisPerStep(1);

  // Read off the lower resolution, should now jump by whole numbers
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(0)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(1)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(2)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(3)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(4)(0)), 0);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(5)(0)), 1);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(6)(0)), 1);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(7)(0)), 1);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(8)(0)), 1);
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(9)(0)), 1);

  // This is OOB, throws a warning and return 0
  EXPECT_DOUBLE_EQ(static_cast<double>(buffer.getPlannedForce(10)(0)), 0);
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_MIX)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < steps; i++)
  {
    plan.col(i) *= i;
  }
  buffer.setControlForcePlan(0L, 0L, plan);

  Eigen::MatrixXs plan2 = Eigen::MatrixXs::Ones(forceDim, steps) * 2;
  for (int i = 0; i < steps; i++)
  {
    plan2.col(i) *= i;
  }
  buffer.setControlForcePlan(25L, 0L, plan2);

  Eigen::MatrixXs planOut = Eigen::MatrixXs::Random(forceDim, steps);
  buffer.getPlannedForcesStartingAt(0L, planOut);

  Eigen::MatrixXs expectedPlan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < 5; i++)
  {
    expectedPlan.col(i) *= (i + 5);
  }
  for (int i = 5; i < 10; i++)
  {
    expectedPlan.col(i) *= (i - 5) * 2;
  }

  if (!equals(planOut, expectedPlan))
  {
    std::cout << "Expected plan: " << std::endl << expectedPlan << std::endl;
    std::cout << "Actual plan: " << std::endl << planOut << std::endl;
  }
  EXPECT_TRUE(equals(planOut, expectedPlan));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_GAP)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < steps; i++)
  {
    plan.col(i) *= i;
  }
  buffer.setControlForcePlan(0L, 0L, plan);

  Eigen::MatrixXs plan2 = Eigen::MatrixXs::Ones(forceDim, steps) * 2;
  for (int i = 0; i < steps; i++)
  {
    plan2.col(i) *= i;
  }
  buffer.setControlForcePlan(12 * dt, 8 * dt, plan2);

  Eigen::MatrixXs planOut = Eigen::MatrixXs::Random(forceDim, steps);
  buffer.getPlannedForcesStartingAt(8 * dt, planOut);

  Eigen::MatrixXs expectedPlan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < 2; i++)
  {
    expectedPlan.col(i) *= (i + 8);
  }
  for (int i = 2; i < 4; i++)
  {
    expectedPlan.col(i) *= 0;
  }
  for (int i = 4; i < 10; i++)
  {
    expectedPlan.col(i) *= (i - 4) * 2;
  }

  if (!equals(planOut, expectedPlan))
  {
    std::cout << "Expected plan: " << std::endl << expectedPlan << std::endl;
    std::cout << "Actual plan: " << std::endl << planOut << std::endl;
  }
  EXPECT_TRUE(equals(planOut, expectedPlan));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_MERGE_OOB)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Ones(forceDim, steps);
  for (int i = 0; i < steps; i++)
  {
    plan.col(i) *= i;
  }
  buffer.setControlForcePlan(0L, 0L, plan);

  Eigen::MatrixXs plan2 = Eigen::MatrixXs::Ones(forceDim, steps) * 2;
  for (int i = 0; i < steps; i++)
  {
    plan2.col(i) *= i;
  }

  // This is out of bounds, so should be discarded
  buffer.setControlForcePlan(12 * dt, 0L, plan2);

  Eigen::MatrixXs planOut = Eigen::MatrixXs::Random(forceDim, steps);
  buffer.getPlannedForcesStartingAt(0L, planOut);

  if (!equals(planOut, plan))
  {
    std::cout << "Expected plan: " << std::endl << plan << std::endl;
    std::cout << "Actual plan: " << std::endl << planOut << std::endl;
  }
  EXPECT_TRUE(equals(planOut, plan));
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_PLAN)
{
  int forceDim = 3;
  int steps = 10;
  int dt = 5;
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);

  buffer.setControlForcePlan(0L, 0L, Eigen::MatrixXs::Ones(forceDim, steps) * 2);

  Eigen::MatrixXs plan = Eigen::MatrixXs::Random(forceDim, steps);
  buffer.getPlannedForcesStartingAt(25L, plan);

  // the beginning of the buffer should be the old plan
  for (int i = 0; i < 5; i++)
  {
    EXPECT_DOUBLE_EQ(static_cast<double>(plan.col(i).sum()), 6.0);
  }
  // the end of the buffer should be 0s
  for (int i = 5; i < 10; i++)
  {
    EXPECT_DOUBLE_EQ(static_cast<double>(plan.col(i).sum()), 0.0);
  }
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, CONTROL_BUFFER_ESTIMATE)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  world->setPenetrationCorrectionEnabled(false);

  /////////////////////////////////////////////////////////////////////
  // Create the skeleton with a single prismatic joint
  /////////////////////////////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<PrismaticJoint*, BodyNode*> boxJointPair
      = box->createJointAndBodyNodePair<PrismaticJoint>();
  // PrismaticJoint* boxJoint = boxJointPair.first;
  BodyNode* boxBody = boxJointPair.second;

  std::shared_ptr<BoxShape> shape(
      new BoxShape(Eigen::Vector3s(0.05, 0.25, 0.05)));
  // ShapeNode* boxShape =
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);

  // We're going to tune the full inertia properties of the swinging object
  Eigen::VectorXs upperBounds = Eigen::VectorXs::Ones(1) * 5.0;
  Eigen::VectorXs lowerBounds = Eigen::VectorXs::Ones(1) * 0.1;
  world->getWrtMass()->registerNode(
      boxBody,
      neural::WrtMassBodyNodeEntryType::INERTIA_MASS,
      upperBounds,
      lowerBounds);

  world->addSkeleton(box);

  assert(world->getNumDofs() == 1);

  /////////////////////////////////////////////////////////////////////
  // Set up a buffer
  /////////////////////////////////////////////////////////////////////

  int forceDim = world->getNumDofs();
  int steps = 100;
  int dt = static_cast<int>(world->getTimeStep() * 1000);
  RealTimeControlBuffer buffer = RealTimeControlBuffer(forceDim, steps, dt);
  ObservationLog log = ObservationLog(
      0L, world->getPositions(), world->getVelocities(), world->getMasses());

  buffer.setControlForcePlan(0L, 0L, Eigen::MatrixXs::Ones(forceDim, steps) * 2);

  log.observe(
      0L,
      Eigen::VectorXs::Zero(1),
      Eigen::VectorXs::Zero(1),
      Eigen::VectorXs::Ones(1));

  /////////////////////////////////////////////////////////////////////
  // Get ground truth behavior
  /////////////////////////////////////////////////////////////////////

  for (int i = 0; i < steps; i++)
  {
    world->setControlForces(buffer.getPlannedForce(i * dt));
    world->step();
  }
  Eigen::VectorXs truePos = world->getPositions();
  Eigen::VectorXs trueVel = world->getVelocities();

  /////////////////////////////////////////////////////////////////////
  // Get ground truth behavior
  /////////////////////////////////////////////////////////////////////

  // Scramble state
  world->setPositions(Eigen::VectorXs::Random(1));
  world->setVelocities(Eigen::VectorXs::Random(1));
  // Estimate state
  buffer.estimateWorldStateAt(world, &log, steps * dt);

  EXPECT_TRUE(equals(truePos, world->getPositions()));
  EXPECT_TRUE(equals(trueVel, world->getVelocities()));
}
#endif