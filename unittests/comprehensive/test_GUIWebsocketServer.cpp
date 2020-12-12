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

#include <gtest/gtest.h>

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

// #ifdef ALL_TESTS
TEST(REALTIME, GUI_SERVER)
{
  GUIWebsocketServer server;
  server
      .createBox(
          "box1",
          Eigen::Vector3d(1, 1, 1),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d(0.5, 0.7, 0.5))
      .setObjectPosition("box1", Eigen::Vector3d(2, 2, 2))
      .flush();
  server.serve(8070);
  server.createSphere(
      "ball1", 0.5, Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.5, 0.5, 0.7));

  std::vector<Eigen::Vector3d> linePoints;
  linePoints.push_back(Eigen::Vector3d::Zero());
  linePoints.push_back(Eigen::Vector3d(0.5, 0.5, 0.7));
  linePoints.push_back(Eigen::Vector3d(1, 1, 1));

  server.createLine("line", linePoints, Eigen::Vector3d(0.7, 0.5, 0.5)).flush();

  server.registerDragListener("box1", [&](Eigen::Vector3d pos) {
    server.setObjectPosition("box1", pos).flush();
  });

  server.registerKeydownListener([&](std::string key) {
    // double v = ((double)rand() / RAND_MAX) * 3;
    // server.setObjectPosition("box1", Eigen::Vector3d(v, v, v)).flush();
  });

  while (server.isServing())
  {
    // spin
    // cartpole->setPosition(0, 0.0);
    // cartpole->setForces(Eigen::VectorXd::Zero(cartpole->getNumDofs()));
    // cartpole->setPositions(Eigen::VectorXd::Zero(cartpole->getNumDofs()));
  }
}
// #endif
