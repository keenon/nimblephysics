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
#include <thread>

#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/proto/SerializeEigen.hpp"
#include "dart/realtime/MPCLocal.hpp"
#include "dart/realtime/MPCRemote.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;
using namespace performance;
using namespace proto;

TEST(PROTO, SERIALIZE_VECTOR)
{
  Eigen::VectorXs original = Eigen::VectorXs::Random(10);
  proto::VectorXs proto;
  serializeVector(proto, original);
  Eigen::VectorXs recovered = deserializeVector(proto);

  EXPECT_TRUE(equals(original, recovered, 0.0));
}

TEST(PROTO, SERIALIZE_MATRIX)
{
  Eigen::MatrixXs original = Eigen::MatrixXs::Random(10, 5);
  proto::MatrixXs proto;
  serializeMatrix(proto, original);
  Eigen::MatrixXs recovered = deserializeMatrix(proto);

  EXPECT_TRUE(equals(original, recovered, 0.0));
}

TEST(PROTO, SERIALIZE_ROLLOUT)
{
  int dofs = 5;
  int steps = 10;

  std::string representationMapping = "identity";
  std::unordered_map<std::string, Eigen::MatrixXs> pos;
  std::unordered_map<std::string, Eigen::MatrixXs> vel;
  std::unordered_map<std::string, Eigen::MatrixXs> force;
  Eigen::VectorXs mass = Eigen::VectorXs::Random(dofs);
  std::unordered_map<std::string, Eigen::MatrixXs> metadata;

  pos["identity"] = Eigen::MatrixXs::Random(dofs, steps);
  pos["mapped"] = Eigen::MatrixXs::Random(dofs, steps);
  vel["identity"] = Eigen::MatrixXs::Random(dofs, steps);
  vel["mapped"] = Eigen::MatrixXs::Random(dofs, steps);
  force["identity"] = Eigen::MatrixXs::Random(dofs, steps);
  force["mapped"] = Eigen::MatrixXs::Random(dofs, steps);

  metadata["1"] = Eigen::MatrixXs::Random(dofs, steps);
  metadata["2"] = Eigen::MatrixXs::Random(dofs, steps);
  metadata["3"] = Eigen::MatrixXs::Random(dofs, steps);

  TrajectoryRolloutReal rollout = TrajectoryRolloutReal(
      representationMapping, pos, vel, force, mass, metadata);

  proto::TrajectoryRollout proto;
  rollout.serialize(proto);

  TrajectoryRolloutReal recovered
      = trajectory::TrajectoryRollout::deserialize(proto);

  EXPECT_EQ(
      rollout.getRepresentationMapping(), recovered.getRepresentationMapping());
  EXPECT_TRUE(equals(rollout.getMassesConst(), recovered.getMassesConst()));
  EXPECT_TRUE(equals(
      rollout.getPosesConst("identity"),
      recovered.getPosesConst("identity"),
      0.0));
  EXPECT_TRUE(equals(
      rollout.getPosesConst("mapped"), recovered.getPosesConst("mapped"), 0.0));
  EXPECT_TRUE(equals(
      rollout.getVelsConst("identity"),
      recovered.getVelsConst("identity"),
      0.0));
  EXPECT_TRUE(equals(
      rollout.getVelsConst("mapped"), recovered.getVelsConst("mapped"), 0.0));
  EXPECT_TRUE(equals(
      rollout.getForcesConst("identity"),
      recovered.getForcesConst("identity"),
      0.0));
  EXPECT_TRUE(equals(
      rollout.getForcesConst("mapped"),
      recovered.getForcesConst("mapped"),
      0.0));
  EXPECT_TRUE(
      equals(rollout.getMetadata("1"), recovered.getMetadata("1"), 0.0));
  EXPECT_TRUE(
      equals(rollout.getMetadata("2"), recovered.getMetadata("2"), 0.0));
  EXPECT_TRUE(
      equals(rollout.getMetadata("3"), recovered.getMetadata("3"), 0.0));
}