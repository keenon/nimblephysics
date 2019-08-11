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

#include "OutputWidget.hpp"

namespace dart {
namespace examples {

//==============================================================================
OutputWidget::OutputWidget(
    Gallery* gallery, dart::gui::osg::ImGuiViewer* viewer)
  : mGallery(gallery), mViewer(viewer)
{
  AutoScroll = true;
  ScrollToBottom = false;
  clear();

  addLog("Welcome to DART Gallery!\n");
}

//==============================================================================
void OutputWidget::addLog(const std::string& str)
{
  Buf.append(str.c_str());
  if (AutoScroll)
    ScrollToBottom = true;
}

//==============================================================================
void OutputWidget::clear()
{
  Buf.clear();
  LineOffsets.clear();
  LineOffsets.push_back(0);
}

//==============================================================================
void OutputWidget::render()
{
  const auto& io = ImGui::GetIO();

  const auto& displaySize = io.DisplaySize;

  auto widgetSize = displaySize;
  widgetSize.x = displaySize.x - 20;
  widgetSize.y = 120;

  auto widgetPos = io.DisplaySize;
  widgetPos.x = 10;
  widgetPos.y = displaySize.y - widgetSize.y - 10;

  ImGui::SetNextWindowPos(widgetPos);
  ImGui::SetNextWindowSize(widgetSize, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.75f);

  if (!ImGui::Begin(
          "Output",
          &mIsVisible,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings
              | ImGuiWindowFlags_NoCollapse
              | ImGuiWindowFlags_NoFocusOnAppearing
              | ImGuiWindowFlags_HorizontalScrollbar))
  {
    ImGui::End();
    return;
  }

  // Options menu
  if (ImGui::BeginPopup("Options"))
  {
    if (ImGui::Checkbox("Auto-scroll", &AutoScroll))
      if (AutoScroll)
        ScrollToBottom = true;
    ImGui::EndPopup();
  }

  // Main window
  if (ImGui::Button("Options"))
    ImGui::OpenPopup("Options");

  ImGui::SameLine();
  const bool clearClicked = ImGui::Button("Clear");

  ImGui::Separator();
  ImGui::BeginChild(
      "scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

  if (clearClicked)
    clear();

  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
  const char* buf = Buf.begin();
  const char* buf_end = Buf.end();

  ImGuiListClipper clipper;
  clipper.Begin(LineOffsets.Size);
  while (clipper.Step())
  {
    for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd;
         line_no++)
    {
      const char* line_start = buf + LineOffsets[line_no];
      const char* line_end = (line_no + 1 < LineOffsets.Size)
                                 ? (buf + LineOffsets[line_no + 1] - 1)
                                 : buf_end;
      ImGui::TextUnformatted(line_start, line_end);
    }
  }
  clipper.End();

  ImGui::PopStyleVar();

  if (ScrollToBottom)
    ImGui::SetScrollHereY(1.0f);
  ScrollToBottom = false;
  ImGui::EndChild();

  ImGui::End();
}

} // namespace examples
} // namespace dart
