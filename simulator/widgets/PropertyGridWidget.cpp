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

#include "PropertyGridWidget.hpp"

namespace dart {
namespace simulator {

namespace {

static void ShowDummyObject(const char* prefix, int uid)
{
  ImGui::PushID(uid); // Use object uid as identifier. Most commonly you
                      // could also use the object pointer as a base ID.
  ImGui::AlignTextToFramePadding(); // Text and Tree nodes are less high
                                    // than regular widgets, here we add
                                    // vertical spacing to make the tree
                                    // lines equal high.
  bool node_open = ImGui::TreeNode("Object", "%s_%u", prefix, uid);
  ImGui::NextColumn();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("my sailor is rich");
  ImGui::NextColumn();
  if (node_open)
  {
    static float dummy_members[8] = {0.0f, 0.0f, 1.0f, 3.1416f, 100.0f, 999.0f};
    for (int i = 0; i < 8; i++)
    {
      ImGui::PushID(i); // Use field index as identifier.
      if (i < 2)
      {
        ShowDummyObject("Child", 424242);
      }
      else
      {
        // Here we use a TreeNode to highlight on hover (we could use e.g.
        // Selectable as well)
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx(
            "Field",
            ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen
                | ImGuiTreeNodeFlags_Bullet,
            "Field_%d",
            i);
        ImGui::NextColumn();
        ImGui::PushItemWidth(-1);
        if (i >= 5)
          ImGui::InputFloat("##value", &dummy_members[i], 1.0f);
        else
          ImGui::DragFloat("##value", &dummy_members[i], 0.01f);
        ImGui::PopItemWidth();
        ImGui::NextColumn();
      }
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

//==============================================================================
void showSimpleFrameTreeNode(const dynamics::SimpleFrame& simpleFrame)
{
  ImGui::PushID(&simpleFrame);
  ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_Leaf;
  bool nodeOpen = ImGui::TreeNodeEx(
      "BodyNode", node_flags, "%s", simpleFrame.getName().c_str());
  if (nodeOpen)
  {

    ImGui::TreePop();
  }
  ImGui::PopID();
}

//==============================================================================
void showBodyNodeTreeNode(const dynamics::BodyNode& bodyNode)
{
  ImGui::PushID(&bodyNode);
  ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_Leaf;
  bool nodeOpen = ImGui::TreeNodeEx(
      "BodyNode", node_flags, "%s", bodyNode.getName().c_str());
  if (nodeOpen)
  {

    ImGui::TreePop();
  }
  ImGui::PopID();
}

//==============================================================================
void showJointTreeNode(const dynamics::Joint& joint)
{
  ImGui::PushID(&joint);
  ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_Leaf;
  bool nodeOpen = ImGui::TreeNodeEx(
      "BodyNode", node_flags, "%s", joint.getName().c_str());
  if (nodeOpen)
  {
    //    for (auto j = 0u; j < skel.getNumBodyNodes(); ++j)
    //    {
    //      auto bodyNode = skel.getBodyNode(j);
    //      if (bodyNode)
    //        showBodyNodeSubtree(*bodyNode);
    //    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

//==============================================================================
void showSkeletonTreeNode(const dynamics::Skeleton& skel)
{
  int nodeClicked = -1;

  ImGui::PushID(&skel); // Use object uid as identifier. Most commonly you
                        // could also use the object pointer as a base ID.
  ImGui::AlignTextToFramePadding(); // Text and Tree nodes are less high
                                    // than regular widgets, here we add
                                    // vertical spacing to make the tree
                                    // lines equal high.

  bool nodeOpen = ImGui::TreeNode("Skeleton", "%s", skel.getName().c_str());
  if (ImGui::IsItemClicked())
  {
    //
    // TODO: Notify the select manager that this node is selected
    // To do so, this function should take select manager or engine, or even
    // make this a private method
    //
    nodeClicked = 1;

  }
  if (nodeOpen)
  {
    auto linksNodeOpen = ImGui::TreeNode("Links", "[Links]");
    if (linksNodeOpen)
    {
      for (auto j = 0u; j < skel.getNumBodyNodes(); ++j)
      {
        auto bodyNode = skel.getBodyNode(j);
        if (bodyNode)
        {
          showBodyNodeTreeNode(*bodyNode);
        }
      }
      ImGui::TreePop();
    }

    auto jointsOpen = ImGui::TreeNode("Joints", "[Joints]");
    if (jointsOpen)
    {
      for (auto j = 0u; j < skel.getNumJoints(); ++j)
      {
        auto joint = skel.getJoint(j);
        if (joint)
        {
          showJointTreeNode(*joint);
        }
      }
      ImGui::TreePop();
    }

    ImGui::TreePop();
  }
  ImGui::PopID();
}

//==============================================================================
void showWorldTreeNode(const simulation::World& world)
{
  ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_None;

  // Skeletons
  treeNodeFlags = ImGuiTreeNodeFlags_None;
  const auto numSkeletons = world.getNumSkeletons();
  if (numSkeletons == 0)
    treeNodeFlags |= ImGuiTreeNodeFlags_Leaf;
  auto skeletonsOpen = ImGui::TreeNode("Skeletons", "[Skeletons]");
  if (skeletonsOpen)
  {
    for (auto i = 0u; i < numSkeletons; ++i)
    {
      auto skel = world.getSkeleton(i);
      if (skel)
        showSkeletonTreeNode(*skel);
    }
    ImGui::TreePop();
  }

  // SimpleFrames
  const auto numSimpleFrames = world.getNumSimpleFrames();
  treeNodeFlags = ImGuiTreeNodeFlags_None;
  if (numSimpleFrames == 0)
    treeNodeFlags |= ImGuiTreeNodeFlags_Leaf;
  auto simpleFramesOpen
      = ImGui::TreeNodeEx("SimpleFrames", treeNodeFlags, "[SimpleFrames]");
  if (simpleFramesOpen)
  {
    for (auto i = 0u; i < numSimpleFrames; ++i)
    {
      auto simpleFrame = world.getSimpleFrame(i);
      if (simpleFrame)
        showSimpleFrameTreeNode(*simpleFrame);
    }
    ImGui::TreePop();
  }
}

} // namespace

//==============================================================================
void PropertyGridWidget::setWorld(simulation::WorldPtr world)
{
  {
    std::lock_guard<std::mutex> lock(mWorldMutex);
    mWorld = std::move(world);
  }
}

//==============================================================================
void PropertyGridWidget::render()
{
  ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Example: Property editor", nullptr))
  {
    ImGui::End();
    return;
  }

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
  ImGui::Columns(1);
  ImGui::Separator();

  std::lock_guard<std::mutex> lock(mWorldMutex);
  {
    if (mWorld)
    {
      showWorldTreeNode(*mWorld);
    }
  }

  ImGui::Columns(1);
  ImGui::Separator();
  ImGui::PopStyleVar();
  ImGui::End();
}

} // namespace simulator
} // namespace dart
