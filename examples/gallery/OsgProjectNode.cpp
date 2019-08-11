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

#include "OsgProjectNode.hpp"

#include "Project.hpp"
// TODO(JS): For testing
#include "dart/gui/osg/WorldNode.hpp"
#include "projects/rigidCubes/RigidCubesProject.hpp"

namespace dart {
namespace examples {

//==============================================================================
OsgProjectNode::OsgProjectNode(std::shared_ptr<Project> project)
  : dart::gui::osg::RealTimeWorldNode(), mProject(std::move(project))
{
  setWorld(mProject->getWorld());
}

//==============================================================================
void OsgProjectNode::refresh()
{
  customPreRefresh();

  clearChildUtilizationFlags();

  if (mSimulating)
  {
    for (auto i = 0u; i < mNumStepsPerCycle; ++i)
    {
      customPreStep();

      mProject->step();

      customPostStep();
    }
  }

  refreshSkeletons();
  refreshSimpleFrames();

  clearUnusedNodes();

  customPostRefresh();
}

//==============================================================================
void OsgProjectNode::customPreRefresh()
{
  // Use this function to execute custom code before each time that the
  // window is rendered. This function can be deleted if it does not need
  // to be used.
}

//==============================================================================
void OsgProjectNode::customPostRefresh()
{
  // Use this function to execute custom code after each time that the
  // window is rendered. This function can be deleted if it does not need
  // to be used.
}

//==============================================================================
void OsgProjectNode::customPreStep()
{
  mProject->prestep();
}

//==============================================================================
void OsgProjectNode::customPostStep()
{
  mProject->poststep();
}

} // namespace examples
} // namespace dart
