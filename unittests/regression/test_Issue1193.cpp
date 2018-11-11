/*
 * Copyright (c) 2011-2018, The DART development contributors
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

#include <gtest/gtest.h>
#include <TestHelpers.hpp>
#include <dart/dart.hpp>

//==============================================================================
class Issue1193 : public testing::Test
{
public:
  void testAngularVelAdd();
};

//==============================================================================
void Issue1193::testAngularVelAdd()
{
  WorldPtr world = World::create();
  EXPECT_TRUE(world != nullptr);
  world->setGravity(Vector3d(0.0, -10.0, 0.0));
  world->setTimeStep(0.001);

  SkeletonPtr sphereSkel = createSphere(0.05, Vector3d(0.0, 1.0, 0.0));
  BodyNode* sphere = sphereSkel->getBodyNode(0);
  Joint* sphereJoint = sphere->getParentJoint();
  world->addSkeleton(sphereSkel);

  Vector3d linearVelBefore = sphere->getLinearVelocity(); //Get linear velocity
  EXPECT_TRUE(equals(linearVelBefore, Vector3d(0, 0, 0)));

  const int maxSteps = 500;

  // Case 1. Zero velocity

  sphereJoint->setVelocity(0, 0.0);
  sphereJoint->setVelocity(1, 0.0);
  sphereJoint->setVelocity(2, 0.0);

  for (int i = 0; i < maxSteps; i++)
  {
    // std::cout << "linear Z is: " << sphere->getLinearVelocity()[2] << std::endl;
    world->step();
  }

  // Get linear velocity
  Vector3d linearVelAfter1 = sphere->getLinearVelocity();
  std::cout << "Linear Velocity after running is " << linearVelAfter1.transpose()
            << std::endl;
  double lx1 = linearVelAfter1[0];
  //double ly1 = linearVelAfter1[1];
  double lz1 = linearVelAfter1[2];

  EXPECT_DOUBLE_EQ(lx1, 0.0);
  //EXPECT_DOUBLE_EQ(ly1, 0.0);
  EXPECT_DOUBLE_EQ(lz1, 0.0);

  // Case 2. Non-zero velocity

  sphereJoint->setVelocity(0, 10.0);  // ang_x -> Affect lz and ly
  sphereJoint->setVelocity(1, 10.0);  // ang_y -> No effect
  sphereJoint->setVelocity(2, 10.0);  // ang_z -> Affect lx and ly

  for (int i = 0; i < maxSteps; i++)
  {
    // std::cout << "linear Z is: " << sphere->getLinearVelocity()[2] << std::endl;
    world->step();
  }

  // Get linear velocity
  Vector3d linearVelAfter2 = sphere->getLinearVelocity();
  std::cout << "Linear Velocity after running is " << linearVelAfter2.transpose()
            << std::endl;
  double lx2 = linearVelAfter2[0];
  //double ly2 = linearVelAfter2[1];
  double lz2 = linearVelAfter2[2];

  EXPECT_DOUBLE_EQ(lx2, 0.0);
  //EXPECT_DOUBLE_EQ(ly2, 0.0);
  EXPECT_DOUBLE_EQ(lz2, 0.0);
}

//==============================================================================
TEST_F(Issue1193, SymTest_AngularVelAdd)
{
  testAngularVelAdd();
}
