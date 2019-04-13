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

#include <vector>

#include <dart/dart.hpp>
#include <dart/external/imgui/imgui.h>
#include <dart/gui/osg/osg.hpp>

#include "Project.hpp"

namespace dart {
namespace examples {

class Project;
class ProjectGroup;

class ProjectNode
{
public:
  virtual ~ProjectNode();

  virtual std::function<std::unique_ptr<Project>()> getCreateFunction() const;

  virtual std::string getName() const;

  virtual std::string getDiscription() const;

  virtual ProjectGroup* asGroup()
  {
    return nullptr;
  }
  virtual const ProjectGroup* asGroup() const
  {
    return nullptr;
  }
};

class ProjectGroup : public ProjectNode
{
public:
  using NodeList = std::vector<std::shared_ptr<ProjectNode>>;

  static std::shared_ptr<ProjectGroup> create(const std::string& name);

  explicit ProjectGroup(const std::string& name);

  ~ProjectGroup() override;

  void addChild(std::shared_ptr<ProjectNode> child);

  std::size_t getNumChildren() const;

  NodeList::iterator begin();

  NodeList::const_iterator begin() const;

  NodeList::iterator end();

  NodeList::const_iterator end() const;

  std::string getName() const override;

  ProjectGroup* asGroup() override
  {
    return this;
  }
  const ProjectGroup* asGroup() const override
  {
    return this;
  }

protected:
  std::string mName;

  NodeList mChildren;
};

template <typename T>
class TProjectNote : public ProjectNode
{
public:
  using ProjectType = T;

  static std::shared_ptr<TProjectNote> create()
  {
    return std::make_shared<TProjectNote>();
  }

  std::function<std::unique_ptr<Project>()> getCreateFunction() const override
  {
    return []() { return dart::common::make_unique<T>(); };
  }

  std::string getName() const override
  {
    return T::getNameStatic();
  }

  std::string getDiscription() const override
  {
    return T::getDiscriptionStatic();
  }
};

} // namespace examples
} // namespace dart
