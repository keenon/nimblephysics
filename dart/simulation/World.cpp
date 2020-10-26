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

#include "dart/simulation/World.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "dart/collision/CollisionGroup.hpp"
#include "dart/common/Console.hpp"
#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/integration/SemiImplicitEulerIntegrator.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

namespace dart {
namespace simulation {

//==============================================================================
std::shared_ptr<World> World::create(const std::string& name)
{
  return std::make_shared<World>(name);
}

//==============================================================================
World::World(const std::string& _name)
  : mName(_name),
    mNameMgrForSkeletons("World::Skeleton | " + _name, "skeleton"),
    mNameMgrForSimpleFrames("World::SimpleFrame | " + _name, "frame"),
    mGravity(0.0, 0.0, -9.81),
    mTimeStep(0.001),
    mTime(0.0),
    mFrame(0),
    mDofs(0),
    mRecording(new Recording(mSkeletons)),
    onNameChanged(mNameChangedSignal),
    mConstraintForceMixingEnabled(false),
    mPenetrationCorrectionEnabled(false)
{
  mIndices.push_back(0);

  auto solver = std::make_unique<constraint::BoxedLcpConstraintSolver>();
  setConstraintSolver(std::move(solver));
}

//==============================================================================
World::~World()
{
  delete mRecording;

  for (common::Connection& connection : mNameConnectionsForSkeletons)
    connection.disconnect();

  for (common::Connection& connection : mNameConnectionsForSimpleFrames)
    connection.disconnect();
}

//==============================================================================
WorldPtr World::clone() const
{
  WorldPtr worldClone = World::create(mName);

  worldClone->setGravity(mGravity);
  worldClone->setTimeStep(mTimeStep);
  worldClone->setConstraintForceMixingEnabled(mConstraintForceMixingEnabled);
  worldClone->setPenetrationCorrectionEnabled(mPenetrationCorrectionEnabled);

  auto cd = getConstraintSolver()->getCollisionDetector();
  worldClone->getConstraintSolver()->setCollisionDetector(
      cd->cloneWithoutCollisionObjects());

  // Clone and add each Skeleton
  for (std::size_t i = 0; i < mSkeletons.size(); ++i)
  {
    worldClone->addSkeleton(mSkeletons[i]->cloneSkeleton());
  }

  // Clone and add each SimpleFrame
  for (std::size_t i = 0; i < mSimpleFrames.size(); ++i)
  {
    worldClone->addSimpleFrame(
        mSimpleFrames[i]->clone(mSimpleFrames[i]->getParentFrame()));
  }

  // For each newly cloned SimpleFrame, try to make its parent Frame be one of
  // the new clones if there is a match. This is meant to minimize any possible
  // interdependencies between the kinematics of different worlds.
  for (std::size_t i = 0; i < worldClone->getNumSimpleFrames(); ++i)
  {
    dynamics::Frame* current_parent
        = worldClone->getSimpleFrame(i)->getParentFrame();

    dynamics::SimpleFramePtr parent_candidate
        = worldClone->getSimpleFrame(current_parent->getName());

    if (parent_candidate)
      worldClone->getSimpleFrame(i)->setParentFrame(parent_candidate.get());
  }

  return worldClone;
}

//==============================================================================
void World::setTimeStep(double _timeStep)
{
  if (_timeStep <= 0.0)
  {
    dtwarn << "[World] Attempting to set negative timestep. Ignoring this "
           << "request because it can lead to undefined behavior.\n";
    return;
  }

  mTimeStep = _timeStep;
  assert(mConstraintSolver);
  mConstraintSolver->setTimeStep(_timeStep);
  for (auto& skel : mSkeletons)
    skel->setTimeStep(_timeStep);
}

//==============================================================================
double World::getTimeStep() const
{
  return mTimeStep;
}

//==============================================================================
void World::reset()
{
  mTime = 0.0;
  mFrame = 0;
  mRecording->clear();
  mConstraintSolver->clearLastCollisionResult();
}

//==============================================================================
void World::integrateVelocities()
{
  // Integrate velocity for unconstrained skeletons
  for (auto& skel : mSkeletons)
  {
    if (!skel->isMobile())
      continue;

    skel->computeForwardDynamics();
    skel->integrateVelocities(mTimeStep);
  }
}

//==============================================================================
void World::step(bool _resetCommand)
{
  Eigen::VectorXd initialVelocity = getVelocities();
  bool _parallelVelocityAndPositionUpdates = true;

  // Integrate velocity for unconstrained skeletons
  for (auto& skel : mSkeletons)
  {
    if (!skel->isMobile())
      continue;

    skel->computeForwardDynamics();
    skel->integrateVelocities(mTimeStep);
  }

  // Record the unconstrained velocities, cause we need them for backprop
  if (mConstraintSolver->getGradientEnabled())
  {
    mLastPreConstraintVelocity = getVelocities();
  }

  // Detect activated constraints and compute constraint impulses
  mConstraintSolver->solve();
  mConstraintSolver->setPenetrationCorrectionEnabled(
      mPenetrationCorrectionEnabled);
  mConstraintSolver->setConstraintForceMixingEnabled(
      mConstraintForceMixingEnabled);

  // Compute velocity changes given constraint impulses
  for (auto& skel : mSkeletons)
  {
    if (!skel->isMobile())
      continue;

    if (skel->isImpulseApplied())
    {
      skel->computeImpulseForwardDynamics();
      skel->setImpulseApplied(false);
    }

    // <DiffDART>: This is the original way integration happened, right after
    // velocity updates
    if (!_parallelVelocityAndPositionUpdates)
      skel->integratePositions(mTimeStep);
    // </DiffDART>

    if (_resetCommand)
    {
      skel->clearInternalForces();
      skel->clearExternalForces();
      skel->resetCommands();
    }
  }

  // <DiffDART>: This is an easier way to compute gradients for. We update p_t+1
  // using v_t, instead of v_t+1
  if (_parallelVelocityAndPositionUpdates)
  {
    Eigen::VectorXd pos = getPositions();
    pos += initialVelocity * mTimeStep;
    setPositions(pos);
  }
  // </DiffDART>: Integrate positions before velocity changes, instead of after

  mTime += mTimeStep;
  mFrame++;
}

//==============================================================================
void World::setTime(double _time)
{
  mTime = _time;
}

//==============================================================================
double World::getTime() const
{
  return mTime;
}

//==============================================================================
void World::setPenetrationCorrectionEnabled(bool enable)
{
  mPenetrationCorrectionEnabled = enable;
}

//==============================================================================
bool World::getPenetrationCorrectionEnabled()
{
  return mPenetrationCorrectionEnabled;
}

//==============================================================================
void World::setConstraintForceMixingEnabled(bool enable)
{
  mConstraintForceMixingEnabled = enable;
}

//==============================================================================
bool World::getConstraintForceMixingEnabled()
{
  return mConstraintForceMixingEnabled;
}

//==============================================================================
int World::getSimFrames() const
{
  return mFrame;
}

//==============================================================================
const std::string& World::setName(const std::string& _newName)
{
  if (_newName == mName)
    return mName;

  const std::string oldName = mName;
  mName = _newName;

  mNameChangedSignal.raise(oldName, mName);

  mNameMgrForSkeletons.setManagerName("World::Skeleton | " + mName);
  mNameMgrForSimpleFrames.setManagerName("World::SimpleFrame | " + mName);

  return mName;
}

//==============================================================================
const std::string& World::getName() const
{
  return mName;
}

//==============================================================================
void World::setGravity(const Eigen::Vector3d& _gravity)
{
  mGravity = _gravity;
  for (std::vector<dynamics::SkeletonPtr>::iterator it = mSkeletons.begin();
       it != mSkeletons.end();
       ++it)
  {
    (*it)->setGravity(_gravity);
  }
}

//==============================================================================
const Eigen::Vector3d& World::getGravity() const
{
  return mGravity;
}

//==============================================================================
dynamics::SkeletonPtr World::getSkeleton(std::size_t _index) const
{
  if (_index < mSkeletons.size())
    return mSkeletons[_index];

  return nullptr;
}

//==============================================================================
dynamics::SkeletonPtr World::getSkeleton(const std::string& _name) const
{
  return mNameMgrForSkeletons.getObject(_name);
}

//==============================================================================
std::size_t World::getSkeletonDofOffset(
    const dynamics::SkeletonPtr& _skeleton) const
{
  std::size_t dofCursor = 0;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    if (skel == _skeleton)
      return dofCursor;
    dofCursor += skel->getNumDofs();
  }
  assert(false && "You asked for an world DOF offset for a skeleton that isn't in the world");
}

//==============================================================================
std::vector<dynamics::BodyNode*> World::getAllBodyNodes()
{
  std::vector<dynamics::BodyNode*> nodes;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    for (dynamics::BodyNode* body : skel->getBodyNodes())
    {
      nodes.push_back(body);
    }
  }
  return nodes;
}

//==============================================================================
std::size_t World::getNumSkeletons() const
{
  return mSkeletons.size();
}

//==============================================================================
std::string World::addSkeleton(const dynamics::SkeletonPtr& _skeleton)
{
  if (nullptr == _skeleton)
  {
    dtwarn << "[World::addSkeleton] Attempting to add a nullptr Skeleton to "
           << "the world!\n";
    return "";
  }

  // If mSkeletons already has _skeleton, then we do nothing.
  if (find(mSkeletons.begin(), mSkeletons.end(), _skeleton) != mSkeletons.end())
  {
    dtwarn << "[World::addSkeleton] Skeleton named [" << _skeleton->getName()
           << "] is already in the world." << std::endl;
    return _skeleton->getName();
  }

  mSkeletons.push_back(_skeleton);
  mMapForSkeletons[_skeleton] = _skeleton;

  mNameConnectionsForSkeletons.push_back(_skeleton->onNameChanged.connect(
      [=](dynamics::ConstMetaSkeletonPtr skel,
          const std::string&,
          const std::string&) { this->handleSkeletonNameChange(skel); }));

  _skeleton->setName(
      mNameMgrForSkeletons.issueNewNameAndAdd(_skeleton->getName(), _skeleton));

  _skeleton->setTimeStep(mTimeStep);
  _skeleton->setGravity(mGravity);

  mIndices.push_back(mIndices.back() + _skeleton->getNumDofs());
  mDofs += _skeleton->getNumDofs();
  mConstraintSolver->addSkeleton(_skeleton);

  // Update recording
  mRecording->updateNumGenCoords(mSkeletons);

  return _skeleton->getName();
}

//==============================================================================
void World::removeSkeleton(const dynamics::SkeletonPtr& _skeleton)
{
  assert(
      _skeleton != nullptr
      && "Attempted to remove nullptr Skeleton from world");

  if (nullptr == _skeleton)
  {
    dtwarn << "[World::removeSkeleton] Attempting to remove a nullptr Skeleton "
           << "from the world!\n";
    return;
  }

  // Find index of _skeleton in mSkeleton.
  std::size_t index = 0;
  for (; index < mSkeletons.size(); ++index)
  {
    if (mSkeletons[index] == _skeleton)
      break;
  }

  // If i is equal to the number of skeletons, then _skeleton is not in
  // mSkeleton. We do nothing.
  if (index == mSkeletons.size())
  {
    dtwarn << "[World::removeSkeleton] Skeleton [" << _skeleton->getName()
           << "] is not in the world.\n";
    return;
  }

  // Update mIndices.
  for (std::size_t i = index + 1; i < mSkeletons.size() - 1; ++i)
    mIndices[i] = mIndices[i + 1] - _skeleton->getNumDofs();
  mIndices.pop_back();
  mDofs -= _skeleton->getNumDofs();

  // Remove _skeleton from constraint handler.
  mConstraintSolver->removeSkeleton(_skeleton);

  // Remove _skeleton from mSkeletons
  mSkeletons.erase(
      remove(mSkeletons.begin(), mSkeletons.end(), _skeleton),
      mSkeletons.end());

  // Disconnect the name change monitor
  mNameConnectionsForSkeletons[index].disconnect();
  mNameConnectionsForSkeletons.erase(
      mNameConnectionsForSkeletons.begin() + index);

  // Update recording
  mRecording->updateNumGenCoords(mSkeletons);

  // Remove from NameManager
  mNameMgrForSkeletons.removeName(_skeleton->getName());

  // Remove from the pointer map
  mMapForSkeletons.erase(_skeleton);
}

//==============================================================================
std::set<dynamics::SkeletonPtr> World::removeAllSkeletons()
{
  std::set<dynamics::SkeletonPtr> ptrs;
  for (std::vector<dynamics::SkeletonPtr>::iterator it = mSkeletons.begin(),
                                                    end = mSkeletons.end();
       it != end;
       ++it)
    ptrs.insert(*it);

  while (getNumSkeletons() > 0)
    removeSkeleton(getSkeleton(0));

  return ptrs;
}

//==============================================================================
bool World::hasSkeleton(const dynamics::ConstSkeletonPtr& skeleton) const
{
  return std::find(mSkeletons.begin(), mSkeletons.end(), skeleton)
         != mSkeletons.end();
}

//==============================================================================
int World::getIndex(int _index) const
{
  return mIndices[_index];
}

//==============================================================================
dynamics::SimpleFramePtr World::getSimpleFrame(std::size_t _index) const
{
  if (_index < mSimpleFrames.size())
    return mSimpleFrames[_index];

  return nullptr;
}

//==============================================================================
dynamics::SimpleFramePtr World::getSimpleFrame(const std::string& _name) const
{
  return mNameMgrForSimpleFrames.getObject(_name);
}

//==============================================================================
std::size_t World::getNumSimpleFrames() const
{
  return mSimpleFrames.size();
}

//==============================================================================
std::string World::addSimpleFrame(const dynamics::SimpleFramePtr& _frame)
{
  assert(_frame != nullptr && "Attempted to add nullptr SimpleFrame to world");

  if (nullptr == _frame)
  {
    dtwarn << "[World::addFrame] Attempting to add a nullptr SimpleFrame to "
              "the world!\n";
    return "";
  }

  if (find(mSimpleFrames.begin(), mSimpleFrames.end(), _frame)
      != mSimpleFrames.end())
  {
    dtwarn << "[World::addFrame] SimpleFrame named [" << _frame->getName()
           << "] is already in the world.\n";
    return _frame->getName();
  }

  mSimpleFrames.push_back(_frame);
  mSimpleFrameToShared[_frame.get()] = _frame;

  mNameConnectionsForSimpleFrames.push_back(_frame->onNameChanged.connect(
      [=](const dynamics::Entity* _entity,
          const std::string&,
          const std::string&) { this->handleSimpleFrameNameChange(_entity); }));

  _frame->setName(
      mNameMgrForSimpleFrames.issueNewNameAndAdd(_frame->getName(), _frame));

  return _frame->getName();
}

//==============================================================================
void World::removeSimpleFrame(const dynamics::SimpleFramePtr& _frame)
{
  assert(
      _frame != nullptr
      && "Attempted to remove nullptr SimpleFrame from world");

  std::vector<dynamics::SimpleFramePtr>::iterator it
      = find(mSimpleFrames.begin(), mSimpleFrames.end(), _frame);

  if (it == mSimpleFrames.end())
  {
    dtwarn << "[World::removeFrame] Frame named [" << _frame->getName()
           << "] is not in the world.\n";
    return;
  }

  std::size_t index = it - mSimpleFrames.begin();

  // Remove the frame
  mSimpleFrames.erase(mSimpleFrames.begin() + index);

  // Disconnect the name change monitor
  mNameConnectionsForSimpleFrames[index].disconnect();
  mNameConnectionsForSimpleFrames.erase(
      mNameConnectionsForSimpleFrames.begin() + index);

  // Remove from NameManager
  mNameMgrForSimpleFrames.removeName(_frame->getName());

  // Remove from the pointer map
  mSimpleFrameToShared.erase(_frame.get());
}

//==============================================================================
std::set<dynamics::SimpleFramePtr> World::removeAllSimpleFrames()
{
  std::set<dynamics::SimpleFramePtr> ptrs;
  for (std::vector<dynamics::SimpleFramePtr>::iterator it
       = mSimpleFrames.begin(),
       end = mSimpleFrames.end();
       it != end;
       ++it)
    ptrs.insert(*it);

  while (getNumSimpleFrames() > 0)
    removeSimpleFrame(getSimpleFrame(0));

  return ptrs;
}

//==============================================================================
std::size_t World::getNumDofs()
{
  return mDofs;
}

//==============================================================================
std::vector<dynamics::DegreeOfFreedom*> World::getDofs()
{
  std::vector<dynamics::DegreeOfFreedom*> vec;
  vec.reserve(mDofs);
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    for (int i = 0; i < skel->getNumDofs(); i++)
    {
      vec.push_back(skel->getDof(i));
    }
  }
  assert(vec.size() == mDofs);
  return vec;
}

//==============================================================================
std::size_t World::getLinkCOMDims()
{
  std::size_t count = 0;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    count += skel->getLinkCOMDims();
  }
  return count;
}

//==============================================================================
std::size_t World::getLinkMOIDims()
{
  std::size_t count = 0;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    count += skel->getLinkMOIDims();
  }
  return count;
}

//==============================================================================
std::size_t World::getLinkMassesDims()
{
  std::size_t count = 0;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    count += skel->getLinkMassesDims();
  }
  return count;
}

//==============================================================================
std::size_t World::getNumBodyNodes()
{
  std::size_t count = 0;
  for (dynamics::SkeletonPtr skel : mSkeletons)
  {
    count += skel->getNumBodyNodes();
  }
  return count;
}

//==============================================================================
Eigen::VectorXd World::getLinkCOMs()
{
  Eigen::VectorXd coms = Eigen::VectorXd(getLinkCOMDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkCOMDims();
    coms.segment(cursor, dims) = mSkeletons[i]->getLinkCOMs();
    cursor += dims;
  }
  return coms;
}

//==============================================================================
Eigen::VectorXd World::getLinkMOIs()
{
  Eigen::VectorXd mois = Eigen::VectorXd(getLinkMOIDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkMOIDims();
    mois.segment(cursor, dims) = mSkeletons[i]->getLinkMOIs();
    cursor += dims;
  }
  return mois;
}

//==============================================================================
Eigen::VectorXd World::getLinkMasses()
{
  Eigen::VectorXd inertias = Eigen::VectorXd(getLinkMassesDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkMassesDims();
    inertias.segment(cursor, dims) = mSkeletons[i]->getLinkMasses();
    cursor += dims;
  }
  return inertias;
}

//==============================================================================
Eigen::VectorXd World::getPositions()
{
  Eigen::VectorXd positions = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    positions.segment(cursor, dofs) = mSkeletons[i]->getPositions();
    cursor += dofs;
  }
  return positions;
}

//==============================================================================
Eigen::VectorXd World::getVelocities()
{
  Eigen::VectorXd velocities = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    velocities.segment(cursor, dofs) = mSkeletons[i]->getVelocities();
    cursor += dofs;
  }
  return velocities;
}

//==============================================================================
Eigen::VectorXd World::getAccelerations()
{
  Eigen::VectorXd velocities = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    velocities.segment(cursor, dofs) = mSkeletons[i]->getAccelerations();
    cursor += dofs;
  }
  return velocities;
}

//==============================================================================
Eigen::VectorXd World::getForces()
{
  Eigen::VectorXd forces = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    forces.segment(cursor, dofs) = mSkeletons[i]->getForces();
    cursor += dofs;
  }
  return forces;
}

//==============================================================================
Eigen::VectorXd World::getForceUpperLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getForceUpperLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXd World::getForceLowerLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getForceLowerLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXd World::getPositionUpperLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getPositionUpperLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXd World::getPositionLowerLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getPositionLowerLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXd World::getVelocityUpperLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getVelocityUpperLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXd World::getVelocityLowerLimits()
{
  Eigen::VectorXd limits = Eigen::VectorXd(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getVelocityLowerLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
void World::setPositions(Eigen::VectorXd position)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setPositions(position.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setVelocities(Eigen::VectorXd velocity)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setVelocities(velocity.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setAccelerations(Eigen::VectorXd accelerations)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setAccelerations(accelerations.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setForces(Eigen::VectorXd forces)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setForces(forces.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setForceUpperLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setForceUpperLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setForceLowerLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setForceLowerLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setPositionUpperLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setPositionUpperLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setPositionLowerLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setPositionLowerLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setVelocityUpperLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setVelocityUpperLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setVelocityLowerLimits(Eigen::VectorXd limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setVelocityLowerLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setLinkCOMs(Eigen::VectorXd coms)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkCOMDims();
    mSkeletons[i]->setLinkCOMs(coms.segment(cursor, dims));
    cursor += dims;
  }
}

//==============================================================================
void World::setLinkMOIs(Eigen::VectorXd mois)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkMOIDims();
    mSkeletons[i]->setLinkMOIs(mois.segment(cursor, dims));
    cursor += dims;
  }
}

//==============================================================================
void World::setLinkMasses(Eigen::VectorXd masses)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getLinkMassesDims();
    mSkeletons[i]->setLinkMasses(masses.segment(cursor, dims));
    cursor += dims;
  }
}

//==============================================================================
/// This gives the C(pos, vel) vector for all the skeletons in the world,
/// without accounting for the external forces
Eigen::VectorXd World::getCoriolisAndGravityForces()
{
  Eigen::VectorXd result = Eigen::VectorXd::Zero(getNumDofs());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = getSkeleton(i);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs) = skel->getCoriolisAndGravityForces();
    cursor += dofs;
  }
  return result;
}

//==============================================================================
Eigen::VectorXd World::getCoriolisAndGravityAndExternalForces()
{
  Eigen::VectorXd result = Eigen::VectorXd::Zero(getNumDofs());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = getSkeleton(i);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->getCoriolisAndGravityForces() - skel->getExternalForces();
    cursor += dofs;
  }
  return result;
}

//==============================================================================
Eigen::MatrixXd World::getMassMatrix()
{
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd::Zero(mDofs, mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    massMatrix.block(cursor, cursor, dofs, dofs)
        = mSkeletons[i]->getMassMatrix();
    cursor += dofs;
  }
  return massMatrix;
}

//==============================================================================
Eigen::MatrixXd World::getInvMassMatrix()
{
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd::Zero(mDofs, mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    invMassMatrix.block(cursor, cursor, dofs, dofs)
        = mSkeletons[i]->getInvMassMatrix();
    cursor += dofs;
  }
  return invMassMatrix;
}

//==============================================================================
bool World::checkCollision(bool checkAllCollisions)
{
  collision::CollisionOption option;

  if (checkAllCollisions)
    option.maxNumContacts = 1e+3;
  else
    option.maxNumContacts = 1u;

  return checkCollision(option);
}

//==============================================================================
bool World::checkCollision(
    const collision::CollisionOption& option,
    collision::CollisionResult* result)
{
  return mConstraintSolver->getCollisionGroup()->collide(option, result);
}

//==============================================================================
const collision::CollisionResult& World::getLastCollisionResult() const
{
  return mConstraintSolver->getLastCollisionResult();
}

//==============================================================================
void World::setConstraintSolver(constraint::UniqueConstraintSolverPtr solver)
{
  if (!solver)
  {
    dtwarn << "[World::setConstraintSolver] nullptr for constraint solver is "
           << "not allowed. Doing nothing.";
    return;
  }

  if (mConstraintSolver)
    solver->setFromOtherConstraintSolver(*mConstraintSolver);

  mConstraintSolver = std::move(solver);
  mConstraintSolver->setTimeStep(mTimeStep);
}

//==============================================================================
constraint::ConstraintSolver* World::getConstraintSolver()
{
  return mConstraintSolver.get();
}

//==============================================================================
const constraint::ConstraintSolver* World::getConstraintSolver() const
{
  return mConstraintSolver.get();
}

//==============================================================================
void World::bake()
{
  const auto collisionResult = getConstraintSolver()->getLastCollisionResult();
  const auto nContacts = static_cast<int>(collisionResult.getNumContacts());
  const auto nSkeletons = getNumSkeletons();

  Eigen::VectorXd state(getIndex(nSkeletons) + 6 * nContacts);
  for (auto i = 0u; i < getNumSkeletons(); ++i)
  {
    state.segment(getIndex(i), getSkeleton(i)->getNumDofs())
        = getSkeleton(i)->getPositions();
  }

  for (auto i = 0; i < nContacts; ++i)
  {
    auto begin = getIndex(nSkeletons) + i * 6;
    state.segment(begin, 3) = collisionResult.getContact(i).point;
    state.segment(begin + 3, 3) = collisionResult.getContact(i).force;
  }

  mRecording->addState(state);
}

//==============================================================================
Recording* World::getRecording()
{
  return mRecording;
}

//==============================================================================
const Eigen::VectorXd& World::getLastPreConstraintVelocity() const
{
  return mLastPreConstraintVelocity;
}

//==============================================================================
void World::handleSkeletonNameChange(
    const dynamics::ConstMetaSkeletonPtr& _skeleton)
{
  if (nullptr == _skeleton)
  {
    dterr << "[World::handleSkeletonNameChange] Received a name change "
          << "callback for a nullptr Skeleton. This is most likely a bug. "
          << "Please report this!\n";
    assert(false);
    return;
  }

  // Get the new name of the Skeleton
  const std::string& newName = _skeleton->getName();

  // Find the shared version of the Skeleton
  std::map<dynamics::ConstMetaSkeletonPtr, dynamics::SkeletonPtr>::iterator it
      = mMapForSkeletons.find(_skeleton);
  if (it == mMapForSkeletons.end())
  {
    dterr << "[World::handleSkeletonNameChange] Could not find Skeleton named ["
          << _skeleton->getName() << "] in the shared_ptr map of World ["
          << getName() << "]. This is most likely a bug. Please report this!\n";
    assert(false);
    return;
  }
  dynamics::SkeletonPtr sharedSkel = it->second;

  // Inform the NameManager of the change
  std::string issuedName
      = mNameMgrForSkeletons.changeObjectName(sharedSkel, newName);

  // If the name issued by the NameManger does not match, reset the name of the
  // Skeleton to match the newly issued name.
  if ((!issuedName.empty()) && (newName != issuedName))
  {
    sharedSkel->setName(issuedName);
  }
  else if (issuedName.empty())
  {
    dterr << "[World::handleSkeletonNameChange] Skeleton named ["
          << sharedSkel->getName() << "] (" << sharedSkel << ") does not exist "
          << "in the NameManager of World [" << getName() << "]. This is most "
          << "likely a bug. Please report this!\n";
    assert(false);
    return;
  }
}

//==============================================================================
void World::handleSimpleFrameNameChange(const dynamics::Entity* _entity)
{
  // Check that this is actually a SimpleFrame
  const dynamics::SimpleFrame* frame
      = dynamic_cast<const dynamics::SimpleFrame*>(_entity);

  if (nullptr == frame)
  {
    dterr << "[World::handleFrameNameChange] Received a callback for a nullptr "
          << "enity. This is most likely a bug. Please report this!\n";
    assert(false);
    return;
  }

  // Get the new name of the Frame
  const std::string& newName = frame->getName();

  // Find the shared version of the Frame
  std::map<const dynamics::SimpleFrame*, dynamics::SimpleFramePtr>::iterator it
      = mSimpleFrameToShared.find(frame);
  if (it == mSimpleFrameToShared.end())
  {
    dterr << "[World::handleFrameNameChange] Could not find SimpleFrame named ["
          << frame->getName() << "] in the shared_ptr map of World ["
          << getName() << "]. This is most likely a bug. Please report this!\n";
    assert(false);
    return;
  }
  dynamics::SimpleFramePtr sharedFrame = it->second;

  std::string issuedName
      = mNameMgrForSimpleFrames.changeObjectName(sharedFrame, newName);

  if ((!issuedName.empty()) && (newName != issuedName))
  {
    sharedFrame->setName(issuedName);
  }
  else if (issuedName.empty())
  {
    dterr << "[World::handleFrameNameChange] SimpleFrame named ["
          << frame->getName() << "] (" << frame << ") does not exist in the "
          << "NameManager of World [" << getName() << "]. This is most likely "
          << "a bug. Please report this!\n";
    assert(false);
    return;
  }
}

} // namespace simulation
} // namespace dart
