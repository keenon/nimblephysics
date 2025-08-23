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
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include <dart/utils/utils.hpp>
#include <gtest/gtest.h>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

// #ifdef ALL_TESTS
TEST(PARALLEL_POS_AND_VEL, s_t_PENDULUM)
{
  // Create a world
  std::shared_ptr<simulation::World> world = dart::utils::SkelParser::readWorld(
      "dart://sample/skel/inverted_double_pendulum.skel");
  world->setPositions(Eigen::Vector3s(0, M_PI, 0));
  // TODO: if we set this to "true", then the model velocity explodes over time
  // with no input from us. This is a simple and obvious example of a bad
  // result.
  world->setParallelVelocityAndPositionUpdates(false);

  for (int i = 0; i < 1000; i++)
  {
    world->step();
  }

  EXPECT_NEAR(static_cast<double>(world->getVelocities().norm()), 3.8100562316867368, 1e-7);

  /*
  // Uncomment this code to get a visualization server that you can perturb with
  // your keyboard.

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Ticker ticker(world->getTimeStep());
  ticker.registerTickListener([&](long time) {
    if (server.isKeyDown("a"))
    {
      world->setControlForces(Eigen::Vector3s(-1, 0, 0));
    }
    else if (server.isKeyDown("e"))
    {
      world->setControlForces(Eigen::Vector3s(1, 0, 0));
    }
    else
    {
      world->setControlForces(Eigen::Vector3s(0, 0, 0));
    }
    world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
    // spin
  }
  */
}
// #endif
