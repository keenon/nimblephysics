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
 *   * This code incorporates portions of Open Dynamics Engine
 *     (Copyright (c) 2001-2004, Russell L. Smith. All rights
 *     reserved.) and portions of FCL (Copyright (c) 2011, Willow
 *     Garage, Inc. All rights reserved.), which were released under
 *     the same BSD license as below
 *
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

#include "dart/gui/glut/TrajectoryReplayWindow.hpp"

#include <cstdio>
#include <iostream>
#include <string>

#include "dart/gui/glut/glut.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

namespace dart {
namespace gui {
namespace glut {

TrajectoryReplayWindow::TrajectoryReplayWindow(
    std::shared_ptr<simulation::World> world, trajectory::AbstractShot* shot)
  : SimWindow()
{
  mCounter = 0;
  mShot = shot;
  mRollout = nullptr;
  mUseKnots = true;
  setWorld(world);
}

//==============================================================================
void TrajectoryReplayWindow::timeStepping()
{
  if (mRollout == nullptr)
  {
    mRollout = new trajectory::TrajectoryRolloutReal(mShot);
    mShot->getStates(mWorld, mRollout, mUseKnots);
  }
  // std::cout << "Time stepping " << mCounter << std::endl;
  int cols = mShot->getNumSteps();
  if (mCounter < cols)
  {
    mWorld->setPositions(mRollout->getPoses("identity").col(mCounter));
  }

  mCounter++;

  if (mCounter >= 2 * cols)
  {
    mCounter = 0;
    delete mRollout;
    mRollout = nullptr;
    mUseKnots = !mUseKnots;
  }

  // Step the simulation forward
  SimWindow::draw();
  // SimWindow::timeStepping();
}

//==============================================================================
void displayTrajectoryInGUI(
    std::shared_ptr<simulation::World> world, trajectory::AbstractShot* shot)
{
  // Create a window for rendering the world and handling user input
  TrajectoryReplayWindow window(world, shot);

  // Initialize glut, initialize the window, and begin the glut event loop
  int argc = 0;
  glutInit(&argc, nullptr);
  window.initWindow(640, 480, "Test");
  glutMainLoop();
}

} // namespace glut
} // namespace gui
} // namespace dart
