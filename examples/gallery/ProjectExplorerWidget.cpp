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

#include "ProjectExplorerWidget.hpp"

namespace dart {
namespace examples {

//==============================================================================
ProjectExplorerWidget::ProjectExplorerWidget(
    Gallery* gallery, dart::gui::osg::ImGuiViewer* viewer)
  : mGallery(gallery), mViewer(viewer)
{
  // Do nothing
}

//==============================================================================
void ProjectExplorerWidget::render()
{
  const auto& io = ImGui::GetIO();
  const auto& displaySize = io.DisplaySize;

  auto widgetSize = displaySize;
  widgetSize.x = 240;
  widgetSize.y = displaySize.y - 170;

  ImGui::SetNextWindowPos(ImVec2(10, 25));
  ImGui::SetNextWindowSize(widgetSize);
  ImGui::SetNextWindowBgAlpha(0.75f);

  if (!ImGui::Begin(
          "Project Explorer",
          &mIsVisible,
          ImGuiWindowFlags_NoResize
              | ImGuiWindowFlags_HorizontalScrollbar))
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  drawProjectTree();

  ImGui::End();
}

//==============================================================================
void ProjectExplorerWidget::drawProjectTree()
{
  nodeClicked = nullptr;
  nodeDoubleClicked = nullptr;

  auto root = mGallery->getProjectTreeRoot();

  if (auto group = root->asGroup())
  {
    for (const auto& child : *group)
      drawProjectTreeRecurse(child.get());
  }
  else
  {
    drawProjectTreeRecurse(root);
  }

  if (nodeDoubleClicked && !nodeDoubleClicked->asGroup())
    selectProject(nodeClicked);
}

//==============================================================================
void ProjectExplorerWidget::drawProjectTreeRecurse(const ProjectNode* node)
{
  if (!node)
    return;

  ImGuiTreeNodeFlags treeNodeFlags
      = ImGuiTreeNodeFlags_None | ImGuiTreeNodeFlags_DefaultOpen;

  if (!node->asGroup())
    treeNodeFlags |= ImGuiTreeNodeFlags_Leaf;

  if (node == nodeSelected)
    treeNodeFlags |= ImGuiTreeNodeFlags_Selected;

  auto nodeOpen = ImGui::TreeNodeEx(node->getName().c_str(), treeNodeFlags);
  if (ImGui::IsItemClicked())
    nodeClicked = node;
  if (ImGui::IsMouseDoubleClicked(0))
    nodeDoubleClicked = node;
  if (nodeOpen)
  {
    if (auto group = node->asGroup())
    {
      for (const auto& child : *group)
        drawProjectTreeRecurse(child.get());
    }

    ImGui::TreePop();
  }
}

//==============================================================================
void ProjectExplorerWidget::selectProject(const ProjectNode* node)
{
  if (nodeSelected == node)
    return;

  nodeSelected = node;
  mGallery->selectProject(nodeSelected);
}

} // namespace examples
} // namespace dart
