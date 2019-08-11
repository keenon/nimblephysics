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

#pragma once

#include <memory>

#include <dart/dart.hpp>
#include <dart/gui/osg/osg.hpp>

#include "Project.hpp"
#include "ProjectNode.hpp"

namespace dart {
namespace examples {

class MainMenuWidget;
class ProjectExplorerWidget;
class ProjectWidget;
class OutputWidget;

class Gallery final
{
public:
  /// Constructor
  Gallery();

  /// Destructor
  ~Gallery() = default;

  /// Runs the main loop
  void run();

  /// Selects the current project
  void selectProject(const ProjectNode* node);

  /// Returns the current project being executed
  std::shared_ptr<Project> getCurrentProject();

  /// Returns the current project being executed
  std::shared_ptr<const Project> getCurrentProject() const;

  /// Returns the root of the project tree
  const ProjectGroup* getProjectTreeRoot() const;

  /// Returns the main menu widget that display the main menu
  MainMenuWidget* getMainMenuWidget();

  /// Returns the project explorer widget that display the project tree
  ProjectExplorerWidget* getProjectExplorerWidget();

  /// Returns the project widget that display project specific GUI
  ProjectWidget* getProjectWidget();

  /// Returns the output widget that prints the logs
  OutputWidget* getOutputWidget();

  /// Returns the grid
  gui::osg::GridVisual* getGrid();

  /// Sets the visibility of the grid
  void setGridVisibility(bool show);

  /// Returns the visibility of the grid
  bool isGridVisible() const;

protected:
  ::osg::ref_ptr<gui::osg::ImGuiViewer> mViewer;

  std::shared_ptr<MainMenuWidget> mMainMenuWidget;
  std::shared_ptr<ProjectExplorerWidget> mProjectExplorerWidget;
  std::shared_ptr<ProjectWidget> mProjectWidget;
  std::shared_ptr<OutputWidget> mOutputWidget;

  ProjectGroup mProjectTreeRoot;

  ::osg::ref_ptr<OsgProjectNode> mOsgNode{nullptr};
  std::vector<::osg::ref_ptr<OsgProjectNode>> mPrevOsgNode;

  std::shared_ptr<Project> mCurrentProject;

  std::vector<std::shared_ptr<Project>> mPrevProject;

  ::osg::ref_ptr<gui::osg::GridVisual> mGrid;
};

} // namespace examples
} // namespace dart
