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

#include <iostream>

#include <gtest/gtest.h>

// #include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/UniversalLoader.hpp"

using namespace dart;

TEST(ConstraintSolver, SIMPLE)
{
  // Load a world where a cube is colliding with the ground.
  std::shared_ptr<simulation::World> world
      = dart::utils::UniversalLoader::loadWorld(
          "dart://sample/skel/test/colliding_cube.skel");
  auto skel = world->getSkeleton("box skeleton");
  auto box = skel->getBodyNode("box");

  // Initially velocity is zero.
  Eigen::Vector6s dq0 = Eigen::Vector6s::Zero();
  EXPECT_EQ(box->getRelativeSpatialVelocity(), dq0);

  // Integrate non-constraint forces, velocity should be negative
  skel->computeForwardDynamics();
  skel->integrateVelocities(world->getTimeStep());
  // 0, 0, 0, 0, -0.00981, 0
  EXPECT_TRUE(box->getRelativeSpatialVelocity()[4] < 0);

  // Collision detection.
  auto solver = world->getConstraintSolver();
  solver->updateConstraints();
  solver->buildConstrainedGroups();
  EXPECT_TRUE(solver->getLastCollisionResult().getNumContacts() > 0);
  EXPECT_TRUE(solver->getNumConstrainedGroups() > 0);

  // Solve constraint impulses.
  for (auto constraintGroup : solver->getConstrainedGroups())
  {
    std::vector<s_t*> impulses = solver->solveConstrainedGroup(constraintGroup);
    EXPECT_TRUE(impulses.size() > 0);
    solver->applyConstraintImpulses(constraintGroup.getConstraints(), impulses);
  }

  // Integrate velocities from solved impulses. Should have non-negative normal
  // velocity after integration.
  EXPECT_TRUE(skel->isImpulseApplied());
  skel->computeImpulseForwardDynamics();
  EXPECT_TRUE(box->getRelativeSpatialVelocity()[4] >= 0);
}
