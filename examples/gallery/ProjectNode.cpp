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

#include "ProjectNode.hpp"

namespace dart {
namespace examples {

//==============================================================================
ProjectNode::~ProjectNode()
{
  // Do nothing
}

//==============================================================================
auto ProjectNode::getCreateFunction() const -> CreateFunction
{
  return nullptr;
}

//==============================================================================
std::string ProjectNode::getName() const
{
  return std::string();
}

//==============================================================================
std::string ProjectNode::getDiscription() const
{
  return std::string();
}

//==============================================================================
ProjectGroup* ProjectNode::asGroup()
{
  return nullptr;
}

//==============================================================================
const ProjectGroup* ProjectNode::asGroup() const
{
  return nullptr;
}

//==============================================================================
std::shared_ptr<ProjectGroup> ProjectGroup::create(const std::string& name)
{
  return std::make_shared<ProjectGroup>(name);
}

//==============================================================================
ProjectGroup::ProjectGroup(const std::string& name) : mName(name)
{
  // Do nothing
}

//==============================================================================
ProjectGroup::~ProjectGroup()
{
  // Do nothing
}

//==============================================================================
void ProjectGroup::addChild(std::shared_ptr<ProjectNode> child)
{
  mChildren.push_back(child);
}

//==============================================================================
std::size_t ProjectGroup::getNumChildren() const
{
  return mChildren.size();
}

//==============================================================================
auto ProjectGroup::begin() -> NodeList::iterator
{
  return mChildren.begin();
}

//==============================================================================
auto ProjectGroup::begin() const -> NodeList::const_iterator
{
  return mChildren.begin();
}

//==============================================================================
auto ProjectGroup::end() -> NodeList::iterator
{
  return mChildren.end();
}

//==============================================================================
auto ProjectGroup::end() const -> NodeList::const_iterator
{
  return mChildren.end();
}

//==============================================================================
std::string ProjectGroup::getName() const
{
  return mName;
}

//==============================================================================
ProjectGroup* ProjectGroup::asGroup()
{
  return this;
}

//==============================================================================
const ProjectGroup* ProjectGroup::asGroup() const
{
  return this;
}

} // namespace examples
} // namespace dart
