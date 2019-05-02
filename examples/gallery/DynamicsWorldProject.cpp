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

#include "DynamicsWorldProject.hpp"

#include "dart/utils/utils.hpp"

namespace dart {
namespace examples {

//==============================================================================
DynamicsWorldProject::DynamicsWorldProject()
{
  mWorld = dart::simulation::World::create();
}

//==============================================================================
void DynamicsWorldProject::initialize()
{
  // Do nothing
}

//==============================================================================
void DynamicsWorldProject::reset()
{
  mWorld->reset();
}

//==============================================================================
void DynamicsWorldProject::step()
{
  mWorld->step();
}

//==============================================================================
void DynamicsWorldProject::finalize()
{
  // Do nothing
}

//==============================================================================
void DynamicsWorldProject::render()
{
  // Do nothing
}

//==============================================================================
void DynamicsWorldProject::drawToProjectWidget()
{
  Project::drawToProjectWidget();

  if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
  {
//    int e = mGallery->isSimulating() ? 0 : 1;
//    if (mViewer->isAllowingSimulation())
//    {
//      if (ImGui::RadioButton("Play", &e, 0) && !mViewer->isSimulating())
//        mViewer->simulate(true);
//      ImGui::SameLine();
//      if (ImGui::RadioButton("Pause", &e, 1) && mViewer->isSimulating())
//        mViewer->simulate(false);
//    }
  }
}

//==============================================================================
void DynamicsWorldProject::drawToMainMenuWidget()
{

}

} // namespace examples
} // namespace dart
