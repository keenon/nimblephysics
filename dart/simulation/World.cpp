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
#include <sstream>
#include <string>
#include <vector>

#include "dart/collision/CollisionGroup.hpp"
#include "dart/common/Console.hpp"
#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/server/RawJsonUtils.hpp"

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
    mConstraintForceMixingEnabled(
        false), // TODO(keenon): We should updated gradients to support this,
                // and re-enable it by default
    mPenetrationCorrectionEnabled(false),
    mWrtMass(std::make_shared<neural::WithRespectToMass>())
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

  // Copy the WithRespectToMass pointer, so we have the same object
  worldClone->mWrtMass = mWrtMass;

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
  mConstraintSolver->setPenetrationCorrectionEnabled(
      mPenetrationCorrectionEnabled);
  mConstraintSolver->setConstraintForceMixingEnabled(
      mConstraintForceMixingEnabled);
  mConstraintSolver->solve();

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
    int cursor = 0;
    for (auto& skel : mSkeletons)
    {
      int dofs = skel->getNumDofs();
      skel->setPositions(skel->integratePositionsExplicit(
          skel->getPositions(),
          initialVelocity.segment(cursor, dofs),
          mTimeStep));
      cursor += dofs;
    }
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
std::shared_ptr<neural::WithRespectToMass> World::getWrtMass()
{
  return mWrtMass;
}

//==============================================================================
/// This returns the world state as a JSON blob that we can render
std::string World::toJson()
{
  std::stringstream json;

  json << "[";
  std::vector<dynamics::BodyNode*> bodies = getAllBodyNodes();
  for (int i = 0; i < bodies.size(); i++)
  {
    auto bodyNode = bodies[i];
    auto skel = bodyNode->getSkeleton();
    /*
    {
      name: "skel.node1",
      shapes: [
        {
          type: "box",
          size: [1, 2, 3],
          color: [1, 2, 3],
          pos: [0, 0, 0],
          angle: [0, 0, 0]
        }
      ],
      pos: [0, 0, 0],
      angle: [0, 0, 0]
    }
    */
    json << "{";
    std::string name = skel->getName() + "." + bodyNode->getName();
    json << "\"name\": \"" << name << "\",";
    json << "\"shapes\": [";
    const std::vector<dynamics::ShapeNode*> visualShapeNodes
        = bodyNode->getShapeNodesWith<dynamics::VisualAspect>();
    for (int j = 0; j < visualShapeNodes.size(); j++)
    {
      json << "{";
      auto shape = visualShapeNodes[j];
      dynamics::ShapePtr shapePtr = shape->getShape();

      if (shapePtr->is<dynamics::BoxShape>())
      {
        const auto box = static_cast<const dynamics::BoxShape*>(shapePtr.get());
        json << "\"type\": \"box\",";
        const Eigen::Vector3d& size = box->getSize();
        json << "\"size\": ";
        vec3ToJson(json, size);
        json << ",";
      }

      dynamics::VisualAspect* visual = shape->getVisualAspect(false);
      json << "\"color\": ";
      vec3ToJson(json, visual->getColor());
      json << ",";

      Eigen::Vector3d relativePos = shape->getRelativeTranslation();
      json << "\"pos\": ";
      vec3ToJson(json, relativePos);
      json << ",";

      Eigen::Vector3d relativeAngle
          = math::matrixToEulerXYZ(shape->getRelativeRotation());
      json << "\"angle\": ";
      vec3ToJson(json, relativeAngle);

      json << "}";
      if (j < visualShapeNodes.size() - 1)
      {
        json << ",";
      }
    }
    json << "],";
    const Eigen::Isometry3d& bodyTransform = bodyNode->getWorldTransform();
    json << "\"pos\":";
    vec3ToJson(json, bodyTransform.translation());
    json << ",";
    json << "\"angle\":";
    vec3ToJson(json, math::matrixToEulerXYZ(bodyTransform.linear()));
    json << "}";
    if (i < bodies.size() - 1)
    {
      json << ",";
    }
  }

  json << "]";

  return json.str();
}

//==============================================================================
/// This returns just the positions as a JSON blob that can be rendered if we
/// already have the original world loaded. Good for real-time viewing.
std::string World::positionsToJson()
{
  std::stringstream json;

  json << "{";

  std::vector<dynamics::BodyNode*> bodies = getAllBodyNodes();
  for (int i = 0; i < bodies.size(); i++)
  {
    auto bodyNode = bodies[i];
    auto skel = bodyNode->getSkeleton();
    /*
    {
      "skel.node1": {
        pos: [0, 0, 0],
        angle: [0, 0, 0]
      }
    }
    */
    std::string name = skel->getName() + "." + bodyNode->getName();
    json << "\"" << name << "\": {";
    const Eigen::Isometry3d& bodyTransform = bodyNode->getWorldTransform();
    json << "\"pos\":";
    vec3ToJson(json, bodyTransform.translation());
    json << ",";
    json << "\"angle\":";
    vec3ToJson(json, math::matrixToEulerXYZ(bodyTransform.linear()));
    json << "}";
    if (i < bodies.size() - 1)
    {
      json << ",";
    }
  }

  json << "}";

  return json.str();
}

//==============================================================================
/// This returns the colors as a JSON blob that can be rendered if we
/// already have the original world loaded. Good for real-time viewing.
std::string World::colorsToJson()
{
  std::stringstream json;

  json << "{";

  std::vector<dynamics::BodyNode*> bodies = getAllBodyNodes();
  for (int i = 0; i < bodies.size(); i++)
  {
    auto bodyNode = bodies[i];
    auto skel = bodyNode->getSkeleton();
    /*
    // A BodyNode with two child shapes gets rendered like this:
    {
      "skel.node1": [
        [0, 0, 0],
        [1, 1, 1]
      ]
    }
    */
    std::string name = skel->getName() + "." + bodyNode->getName();
    json << "\"" << name << "\": [";

    const std::vector<dynamics::ShapeNode*> visualShapeNodes
        = bodyNode->getShapeNodesWith<dynamics::VisualAspect>();
    for (int j = 0; j < visualShapeNodes.size(); j++)
    {
      auto shape = visualShapeNodes[j];
      dynamics::VisualAspect* visual = shape->getVisualAspect(false);
      if (j > 0)
        json << ",";
      vec3ToJson(json, visual->getColor());
    }

    json << "]";

    if (i < bodies.size() - 1)
    {
      json << ",";
    }
  }

  json << "}";

  return json.str();
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
/// Get the indexed skeleton
const dynamics::SkeletonPtr& World::getSkeletonRef(std::size_t _index) const
{
  return mSkeletons[_index];
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

  // TODO(keenon): Add support for springs and damping coefficients
  bool warnedSpringStiffness = false;
  bool warnedDampingCoefficient = false;
  for (auto* dof : _skeleton->getDofs())
  {
    if (dof->getSpringStiffness() != 0)
    {
      if (!warnedSpringStiffness)
      {
        warnedSpringStiffness = true;
        dtwarn << "[World::addSkeleton] Attempting to add a Skeleton \""
               << _skeleton->getName() << "\" to "
               << "the world with non-zero spring stiffness! This version of "
                  "DiffDART doesn't support spring stiffness. It will be "
                  "automatically set to zero.\n";
      }
    }
    dof->setSpringStiffness(0);
    if (dof->getDampingCoefficient() != 0)
    {
      if (!warnedDampingCoefficient)
      {
        warnedDampingCoefficient = true;
        dtwarn
            << "[World::addSkeleton] Attempting to add a Skeleton \""
            << _skeleton->getName() << "\" to "
            << "the world with non-zero damping coefficient! This version of "
               "DiffDART doesn't support damping coefficients. It will be "
               "automatically set to zero.\n";
      }
    }
    dof->setDampingCoefficient(0);
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
std::size_t World::getNumDofs() const
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
/// Returns the size of the getMasses() vector
std::size_t World::getMassDims()
{
  return mWrtMass->dim(this);
}

//==============================================================================
/// This will prevent mass from being tuned
void World::clearTunableMassThisInstance()
{
  mWrtMass = std::make_shared<neural::WithRespectToMass>();
}

//==============================================================================
/// This registers that we'd like to keep track of this BodyNode's mass in a
/// specified way in differentiation
void World::tuneMass(
    dynamics::BodyNode* node,
    neural::WrtMassBodyNodeEntryType type,
    Eigen::VectorXd upperBound,
    Eigen::VectorXd lowerBound)
{
  mWrtMass->registerNode(node, type, upperBound, lowerBound);
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
Eigen::VectorXd World::getMasses()
{
  return mWrtMass->get(this);
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
Eigen::VectorXd World::getExternalForces()
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
Eigen::VectorXd World::getExternalForceUpperLimits()
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
Eigen::VectorXd World::getExternalForceLowerLimits()
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
// This gives the vector of mass upper limits for all the registered bodies in
// this world
Eigen::VectorXd World::getMassUpperLimits()
{
  return mWrtMass->upperBound(this);
}

//==============================================================================
// This gives the vector of mass lower limits for all the registered bodies in
// this world
Eigen::VectorXd World::getMassLowerLimits()
{
  return mWrtMass->lowerBound(this);
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
void World::setExternalForces(Eigen::VectorXd forces)
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
void World::setExternalForceUpperLimits(Eigen::VectorXd limits)
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
void World::setExternalForceLowerLimits(Eigen::VectorXd limits)
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
// This sets all the masses for all the registered bodies in the world
void World::setMasses(Eigen::VectorXd masses)
{
  mWrtMass->set(this, masses);
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

/// This gets the Jacobian relating how changing our current position will
/// change our next position after a step. Intuitively, you'd expect this to
/// just be an identity matrix, and often it is, but if we have any FreeJoints
/// or BallJoints things get more complicated, because they actually use a
/// complicated function to integrate to the next position.
Eigen::MatrixXd World::getPosPosJacobian() const
{
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mDofs, mDofs);
  int cursor = 0;
  for (auto& skel : mSkeletons)
  {
    int dofs = skel->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = skel->getPosPosJac(
        skel->getPositions(), skel->getVelocities(), mTimeStep);
    cursor += dofs;
  }
  return jac;
}

/// This gets the Jacobian relating how changing our current velocity will
/// change our next position after a step. Intuitively, you'd expect this to
/// just be an identity matrix * dt, and often it is, but if we have any
/// FreeJoints or BallJoints things get more complicated, because they
/// actually use a complicated function to integrate to the next position.
Eigen::MatrixXd World::getVelPosJacobian() const
{
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mDofs, mDofs);
  int cursor = 0;
  for (auto& skel : mSkeletons)
  {
    int dofs = skel->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = skel->getVelPosJac(
        skel->getPositions(), skel->getVelocities(), mTimeStep);
    cursor += dofs;
  }
  return jac;
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
