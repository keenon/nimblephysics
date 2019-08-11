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

#include "MainMenuWidget.hpp"

#include "OutputWidget.hpp"
#include "ProjectExplorerWidget.hpp"
#include "ProjectWidget.hpp"

namespace dart {
namespace examples {

//==============================================================================
MainMenuWidget::MainMenuWidget(
    Gallery* gallery, dart::gui::osg::ImGuiViewer* viewer)
  : mGallery(gallery), mViewer(viewer)
{
  // Do nothing
}

//==============================================================================
void MainMenuWidget::render()
{
  if (ImGui::BeginMainMenuBar())
  {
    if (ImGui::BeginMenu("Gallery"))
    {
      if (ImGui::MenuItem("Exit"))
        mViewer->setDone(true);
      ImGui::EndMenu();
    }

    if (mGallery)
    {
      auto currentProject = mGallery->getCurrentProject();
      if (currentProject)
      {
        currentProject->drawToMainMenuWidget();
      }
    }

    if (ImGui::BeginMenu("View"))
    {
      bool isVisible;

      isVisible = mGallery->getProjectExplorerWidget()->isVisible();
      if (ImGui::MenuItem("Project Explorer Window", "Ctrl+E", &isVisible))
        mGallery->getProjectExplorerWidget()->setVisible(isVisible);

      isVisible = mGallery->getProjectWidget()->isVisible();
      if (ImGui::MenuItem("Project Control Window", "Ctrl+P", &isVisible))
        mGallery->getProjectWidget()->setVisible(isVisible);

      isVisible = mGallery->getOutputWidget()->isVisible();
      if (ImGui::MenuItem("Output Window", "Ctrl+L", &isVisible))
        mGallery->getOutputWidget()->setVisible(isVisible);

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help"))
    {
      if (ImGui::MenuItem("About DART"))
        mViewer->showAbout();
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

} // namespace examples
} // namespace dart
