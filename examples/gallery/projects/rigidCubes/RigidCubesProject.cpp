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

#include "RigidCubesProject.hpp"

#include <sstream>

#include "dart/utils/utils.hpp"

namespace dart {
namespace examples {

//==============================================================================
void RigidCubesProject::initialize()
{
  // Create and initialize the world
  mWorld = dart::utils::SkelParser::readWorld("dart://sample/skel/cubes.skel");
  if (!mWorld)
  {
    dterr << "Failed to load world.\n";
    exit(EXIT_FAILURE);
  }
  mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));

  mForce.setZero();
}

//==============================================================================
void RigidCubesProject::reset()
{
  // Do nothing
}

//==============================================================================
void RigidCubesProject::prestep()
{
  mWorld->getSkeleton(1)->getBodyNode(0)->addExtForce(mForce);
  mWorld->step();
  mForce /= 2.0;
}

//==============================================================================
void RigidCubesProject::finalize()
{
  // Do nothing
}

//==============================================================================
void RigidCubesProject::render()
{
  // Do nothing
}

//==============================================================================
std::string RigidCubesProject::getNameStatic()
{
  return "Rigid Cubes";
}

//==============================================================================
std::string RigidCubesProject::getName() const
{
  return getNameStatic();
}

//==============================================================================
std::string RigidCubesProject::getUsage() const
{
  std::stringstream ss;
  // TODO(JS): Outdated
  ss << "space bar: simulation on/off\n";
  ss << "'p': playback/stop\n";
  ss << "'[' and ']': play one frame backward and forward\n";
  ss << "'v': visualization on/off\n";
  ss << "'1'--'4': programmed interaction\n";

  return ss.str();
}

//==============================================================================
std::string RigidCubesProject::getDiscriptionStatic()
{
  return "No descriptions";
}

} // namespace examples
} // namespace dart
