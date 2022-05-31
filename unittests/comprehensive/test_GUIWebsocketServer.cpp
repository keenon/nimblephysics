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

#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>
#include <gtest/gtest.h>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
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

#ifdef ALL_TESTS
TEST(REALTIME, GUI_SERVER)
{
  // Create a world
  /*
  std::shared_ptr<simulation::World> world
      = dart::utils::SkelParser::readWorld("dart://sample/skel/cartpole.skel");
      */

  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0.0));

  // Load ground and Atlas robot and add them to the world
  dart::utils::DartLoader urdfLoader;
  std::shared_ptr<dynamics::Skeleton> ground
      = urdfLoader.parseSkeleton("dart://sample/sdf/atlas/ground.urdf");

  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  /*
  std::shared_ptr<dynamics::Skeleton> kr5
      = urdfLoader.parseSkeleton("dart://sample/urdf/KR5/KR5 sixx R650.urdf");
  */

  // world->addSkeleton(ground);
  // world->addSkeleton(atlas);
  // world->addSkeleton(kr5);

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  atlas->setPosition(3, 0.75);

  // Disable the ground from casting its own shadows
  ground->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  /*
  while (true)
  {
    world->step();
  }
  */

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  server.createRichPlot(
      "test_rich",
      Eigen::Vector2i(20, 20),
      Eigen::Vector2i(300, 300),
      -5.0,
      20.0,
      -5.0,
      20.0,
      "Test Plot",
      "X Axis",
      "Y Axis");

  std::vector<s_t> xs;
  std::vector<s_t> y1;
  std::vector<s_t> y2;

  for (int i = 0; i < 40; i++)
  {
    xs.push_back(((double)i / 40) * 25 - 5);
    y1.push_back(
        ((double)i / 40) * 25 - 5 + (((double)rand() / RAND_MAX) * 5.0 - 2.5));
    y2.push_back(
        ((double)i / 40) * 25 - 5 + (((double)rand() / RAND_MAX) * 5.0 - 2.5));
  }

  server.setRichPlotData("test_rich", "Series 1", "blue", "line", xs, y1);
  server.setRichPlotData("test_rich", "Series 2", "green", "line", xs, y2);

  std::vector<s_t> x_now;
  std::vector<s_t> y_now;
  x_now.push_back(0);
  y_now.push_back(-5);
  x_now.push_back(0);
  y_now.push_back(25);
  server.setRichPlotData("test_rich", "Now", "red", "line", x_now, y_now);

  server.setRichPlotBounds("test_rich", -2.5, 20.0, 0, 25.0);

  Ticker ticker(world->getTimeStep());
  ticker.registerTickListener([&](long time) {
    world->step();
    server.renderWorld(world);

    std::vector<s_t> x_now;
    std::vector<s_t> y_now;
    x_now.push_back(time % 40);
    y_now.push_back(-5);
    x_now.push_back(time % 40);
    y_now.push_back(25);
    server.setRichPlotData("test_rich", "Now", "red", "line", x_now, y_now);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  /*
  server
      .createBox(
          "box1",
          Eigen::Vector3s(1, 1, 1),
          Eigen::Vector3s::Zero(),
          Eigen::Vector3s::Zero(),
          Eigen::Vector3s(0.5, 0.7, 0.5))
      .setObjectPosition("box1", Eigen::Vector3s(2, 2, 2))
      .flush();
  server.createSphere(
      "ball1", 0.5, Eigen::Vector3s(1, 1, 1), Eigen::Vector3s(0.5, 0.5, 0.7));

  std::vector<Eigen::Vector3s> linePoints;
  linePoints.push_back(Eigen::Vector3s::Zero());
  linePoints.push_back(Eigen::Vector3s(0.5, 0.5, 0.7));
  linePoints.push_back(Eigen::Vector3s(1, 1, 1));

  server.createLine("line", linePoints, Eigen::Vector3s(0.7, 0.5, 0.5)).flush();

  server.registerDragListener("box1", [&](Eigen::Vector3s pos) {
    server.setObjectPosition("box1", pos).flush();
  });

  server.createText(
      "text", "Hello world", Eigen::Vector2i(20, 20), Eigen::Vector2i(80, 20));

  server.createButton(
      "button",
      "Click me!",
      Eigen::Vector2i(20, 60),
      Eigen::Vector2i(80, 20),
      [&]() { std::cout << "Button was clicked!" << std::endl; });

  server.createSlider(
      "slider",
      Eigen::Vector2i(80, 80),
      Eigen::Vector2i(120, 40),
      0,
      100,
      0,
      true,
      true,
      [&](s_t val) { std::cout << "Slider moved to " << val << std::endl; });

  server.createSlider(
      "vert_slider",
      Eigen::Vector2i(80, 140),
      Eigen::Vector2i(40, 120),
      0,
      100,
      0,
      true,
      false,
      [&](s_t val) {
        std::cout << "Vertical slider moved to " << val << std::endl;
      });

  std::vector<s_t> xs;
  std::vector<s_t> ys;
  for (int i = 0; i < 10; i++)
  {
    xs.push_back(i);
    ys.push_back(log(i));
  }
  server.createPlot(
      "plot",
      Eigen::Vector2i(60, 270),
      Eigen::Vector2i(80, 80),
      xs,
      0,
      10,
      ys,
      0,
      10,
      "line");

  server.registerScreenResizeListener([&](Eigen::Vector2i size) {
    std::cout << "Screen resized: " << size << std::endl;
  });

  server.registerKeydownListener([&](std::string key) {
    // s_t v = ((s_t)rand() / RAND_MAX) * 3;
    // server.setObjectPosition("box1", Eigen::Vector3s(v, v, v)).flush();
    std::cout << "Pressed key " << key << std::endl;
  });
  */

  /*
  Ticker ticker(1.0);
  ticker.registerTickListener(
      [&](long ms) { std::cout << "Tick: " << ms << std::endl; });
  */

  server.blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(REALTIME, GUI_SERVER_2)
{
  GUIWebsocketServer server;
  server.serve(8070);

  server.createLayer("Default on", Eigen::Vector4s(1, 0, 0, 1), true);
  server.createLayer("Default off", Eigen::Vector4s(0, 0, 1, 1), false);
  server.createBox(
      "test",
      Eigen::Vector3s::Ones(),
      Eigen::Vector3s::Zero(),
      Eigen::Vector3s::Zero(),
      Eigen::Vector4s(1, 0, 0, 1),
      "Default on");

  server.createBox(
      "test",
      Eigen::Vector3s::Ones(),
      Eigen::Vector3s::Ones(),
      Eigen::Vector3s::Zero(),
      Eigen::Vector4s(0, 0, 1, 1),
      "Default off");

  server.createText(
      "key2",
      "Testing text",
      Eigen::Vector2i(150, 150),
      Eigen::Vector2i(100, 20));

  server.createButton(
      "key3",
      "Button",
      Eigen::Vector2i(200, 150),
      Eigen::Vector2i(100, 20),
      []() { std::cout << "Clicked!" << std::endl; });

  server.createSlider(
      "key4",
      Eigen::Vector2i(250, 150),
      Eigen::Vector2i(100, 20),
      0.0,
      1.0,
      0.5,
      false,
      true,
      [](s_t value) { std::cout << "New value: " << value << std::endl; });

  server.blockWhileServing();
}
#endif