/*
 * Copyright (c) 2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
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

#include "SimulationWidget.hpp"

#include "dart/external/imgui/imgui.h"

#include "AtlasSimbiconWorldNode.hpp"

//==============================================================================
SimulationWidget::SimulationWidget(
    dart::gui::osg::ImGuiViewer* viewer,
    AtlasSimbiconWorldNode* node)
  : mViewer(viewer),
    mNode(node)
{
  // Do nothing
}

//==============================================================================
void SimulationWidget::render()
{
  const auto w = mViewer->getWidth();
  const auto h = mViewer->getHeight();

  ImGui::SetNextWindowPos(ImVec2(mMargin, h - (mWidgetHeight + mMargin)));
  ImGui::SetNextWindowSize(ImVec2(w - 2.0f * mMargin, mWidgetHeight));
  if (!ImGui::Begin("Simulation Control", &mIsVisible,
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove))
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  auto simFrames = mNode->getWorld()->getSimFrames();
  auto simTime = mNode->getWorld()->getTime();

  // Play or pause
  const auto simulating = mViewer->isSimulating();
  const auto playOrPause = simulating ? "Pause" : "Play ";
  const auto buttonPushed = ImGui::Button(playOrPause);
  if (buttonPushed)
  {
    if (simulating)
      mViewer->simulate(false);
    else
      mViewer->simulate(true);
  }

  // Manual steps
  if (!simulating)
  {
    ImGui::SameLine();
    const auto buttonPushed = ImGui::Button("Simulate");
    if (buttonPushed)
    {
      std::cout << "steps!" << std::endl;
    }
  }
  ImGui::SameLine();
  ImGui::Text("Steps: ");
  ImGui::SameLine();
  ImGui::PushItemWidth(100);
  ImGui::InputInt("", &mSteps);
  mSteps = std::max(1, mSteps);
  ImGui::PopItemWidth();

  char simTimeBuff[16];
  std::size_t simTimeBuffSize = 16;

  // Simulation time
  std::sprintf(simTimeBuff, "%.3f", simTime);
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  ImGui::Text("Simulation Time: ");
  ImGui::SameLine();
  ImGui::PushItemWidth(100);
  ImGui::InputText("sec", simTimeBuff, simTimeBuffSize,
                   ImGuiInputTextFlags_ReadOnly |
                   ImGuiInputTextFlags_AutoSelectAll);
  ImGui::PopItemWidth();

  // Simulation frames
  std::sprintf(simTimeBuff, "%d", simFrames);
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  ImGui::Text("| Simulation Frames: ");
  ImGui::SameLine();
  ImGui::PushItemWidth(100);
  ImGui::InputText("", simTimeBuff, simTimeBuffSize,
                   ImGuiInputTextFlags_ReadOnly |
                   ImGuiInputTextFlags_AutoSelectAll);
  ImGui::PopItemWidth();

  // FPS
  ImGui::SameLine();
  ImGui::Text("| %.1f FPS", ImGui::GetIO().Framerate);

  ImGui::End();
}
