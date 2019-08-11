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

#include "ProjectWidget.hpp"

#include "../Gallery.hpp"

namespace dart {
namespace examples {

//==============================================================================
ProjectWidget::ProjectWidget(
    Gallery* gallery, dart::gui::osg::ImGuiViewer* viewer)
  : mGallery(gallery), mViewer(viewer)
{
  // Do nothing
}

//==============================================================================
void ProjectWidget::render()
{
  const auto& io = ImGui::GetIO();
  const auto& displaySize = io.DisplaySize;

  auto widgetSize = displaySize;
  widgetSize.x = 240;
  widgetSize.y = displaySize.y - 170;

  auto widgetPos = io.DisplaySize;
  widgetPos.x = displaySize.x - widgetSize.x - 10;
  widgetPos.y = 25;

  ImGui::SetNextWindowPos(widgetPos);
  ImGui::SetNextWindowSize(widgetSize, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.75f);

  if (!ImGui::Begin(
          "Project Control",
          &mIsVisible,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings
              | ImGuiWindowFlags_NoCollapse
              | ImGuiWindowFlags_NoFocusOnAppearing
              | ImGuiWindowFlags_HorizontalScrollbar))
  {
    ImGui::End();
    return;
  }

  if (mGallery)
  {
    auto currentProject = mGallery->getCurrentProject();
    if (currentProject)
    {
      currentProject->drawToProjectWidget();
    }
  }

  ImGui::End();
}

} // namespace examples
} // namespace dart
