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

#include <algorithm>
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
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
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
    mParallelVelocityAndPositionUpdates(
        true), // TODO(keenon): We should fix our backprop to somehow achieve
               // the best of both worlds here
    mFallbackConstraintForceMixingConstant(1e-4),
    mContactClippingDepth(0.03),
    mPenetrationCorrectionEnabled(false),
    mWrtMass(std::make_shared<neural::WithRespectToMass>()),
    mUseFDOverride(false),
    mSlowDebugResultsAgainstFD(false),
    mConstraintEngine([this](simulation::World* world, bool _resetCommand) {
      return lcpConstraintEngine(world, _resetCommand);
    })
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
  worldClone->setFallbackConstraintForceMixingConstant(
      mFallbackConstraintForceMixingConstant);
  worldClone->setContactClippingDepth(mContactClippingDepth);
  worldClone->setPenetrationCorrectionEnabled(mPenetrationCorrectionEnabled);
  worldClone->setParallelVelocityAndPositionUpdates(
      mParallelVelocityAndPositionUpdates);

  // Copy the WithRespectToMass pointer, so we have the same object
  worldClone->mWrtMass = mWrtMass;

  auto cd = getConstraintSolver()->getCollisionDetector();
  worldClone->getConstraintSolver()->setCollisionDetector(
      cd->cloneWithoutCollisionObjects());

  // Clone and add each Skeleton
  for (std::size_t i = 0; i < mSkeletons.size(); ++i)
  {

    dart::dynamics::SkeletonPtr cloned_skel = mSkeletons[i]->cloneSkeleton();
    cloned_skel->setLinkMasses(mSkeletons[i]->getLinkMasses());
    cloned_skel->setLinkCOMs(mSkeletons[i]->getLinkCOMs());
    cloned_skel->setLinkMOIs(mSkeletons[i]->getLinkMOIs());
    cloned_skel->setLinkBetas(mSkeletons[i]->getLinkBetas());
    worldClone->addSkeleton(cloned_skel);
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

  // Ensure that the action mapping for the RL-style API is preserved
  worldClone->setActionSpace(mActionSpace);

  return worldClone;
}

//==============================================================================
void World::setTimeStep(s_t _timeStep)
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
s_t World::getTimeStep() const
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
  Eigen::VectorXs initialVelocity = getVelocities();

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
  mConstraintSolver->setContactClippingDepth(mContactClippingDepth);
  mConstraintSolver->setFallbackConstraintForceMixingConstant(
      mFallbackConstraintForceMixingConstant);
  runConstraintEngine(_resetCommand);
  integratePositions(initialVelocity);

  mTime += mTimeStep;
  mFrame++;
}

//==============================================================================
void World::runConstraintEngine(bool _resetCommand)
{
  mConstraintEngine(this, _resetCommand);
}

//==============================================================================
void World::lcpConstraintEngine(simulation::World* world, bool _resetCommand)
{
  mConstraintSolver->solve();
  integrateVelocitiesFromImpulses(_resetCommand);
}

//==============================================================================
void World::replaceConstraintEngine(const constraintEngine& engine)
{
  mConstraintEngine = engine;
}

//==============================================================================
void World::integrateVelocitiesFromImpulses(bool _resetCommand)
{
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

    if (_resetCommand)
    {
      skel->clearInternalForces();
      skel->clearExternalForces();
      skel->resetCommands();
    }
  }
}

//==============================================================================
void World::integratePositions(Eigen::VectorXs initialVelocity)
{
  int cursor = 0;
  for (auto& skel : mSkeletons)
  {
    if (mParallelVelocityAndPositionUpdates)
    {
      // <Nimble>: This is an easier way to compute gradients for. We update
      // p_t+1 using v_t, instead of v_t+1
      int dofs = skel->getNumDofs();
      skel->setPositions(skel->integratePositionsExplicit(
          skel->getPositions(),
          initialVelocity.segment(cursor, dofs),
          mTimeStep));
      cursor += dofs;
      // </Nimble>: Integrate positions before velocity changes, instead of
      // after
    }
    else
    {
      // <Nimble>: This is the original way integration happened, right after
      // velocity updates
      skel->integratePositions(mTimeStep);
      // </Nimble>
    }
  }
}

//==============================================================================
void World::integrateVelocitiesFromImpulses(bool _resetCommand)
{
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

    if (_resetCommand)
    {
      skel->clearInternalForces();
      skel->clearExternalForces();
      skel->resetCommands();
    }
  }
}

//==============================================================================
void World::integratePositions(Eigen::VectorXs initialVelocity)
{
  int cursor = 0;
  for (auto& skel : mSkeletons)
  {
    if (mParallelVelocityAndPositionUpdates)
    {
      // <Nimble>: This is an easier way to compute gradients for. We update
      // p_t+1 using v_t, instead of v_t+1
      int dofs = skel->getNumDofs();
      skel->setPositions(skel->integratePositionsExplicit(
          skel->getPositions(),
          initialVelocity.segment(cursor, dofs),
          mTimeStep));
      cursor += dofs;
      // </Nimble>: Integrate positions before velocity changes, instead of
      // after
    }
    else
    {
      // <Nimble>: This is the original way integration happened, right after
      // velocity updates
      skel->integratePositions(mTimeStep);
      // </Nimble>
    }
  }
}

//==============================================================================
void World::setTime(s_t _time)
{
  mTime = _time;
}

//==============================================================================
s_t World::getTime() const
{
  return mTime;
}

//==============================================================================
void World::setParallelVelocityAndPositionUpdates(bool enable)
{
  mParallelVelocityAndPositionUpdates = enable;
}

//==============================================================================
bool World::getParallelVelocityAndPositionUpdates()
{
  return mParallelVelocityAndPositionUpdates;
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
void World::setFallbackConstraintForceMixingConstant(s_t constant)
{
  mFallbackConstraintForceMixingConstant = constant;
}

//==============================================================================
s_t World::getFallbackConstraintForceMixingConstant()
{
  return mFallbackConstraintForceMixingConstant;
}

//==============================================================================
void World::setContactClippingDepth(s_t depth)
{
  mContactClippingDepth = depth;
}

//==============================================================================
s_t World::getContactClippingDepth()
{
  return mContactClippingDepth;
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
        const Eigen::Vector3s& size = box->getSize();
        json << "\"size\": ";
        vec3ToJson(json, size);
        json << ",";
      }

      dynamics::VisualAspect* visual = shape->getVisualAspect(false);
      json << "\"color\": ";
      vec3ToJson(json, visual->getColor());
      json << ",";

      Eigen::Vector3s relativePos = shape->getRelativeTranslation();
      json << "\"pos\": ";
      vec3ToJson(json, relativePos);
      json << ",";

      Eigen::Vector3s relativeAngle
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
    const Eigen::Isometry3s& bodyTransform = bodyNode->getWorldTransform();
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
    const Eigen::Isometry3s& bodyTransform = bodyNode->getWorldTransform();
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
/// This gets the cached LCP solution, which is useful to be able to get/set
/// because it can effect the forward solutions of physics problems because of
/// our optimistic LCP-stabilization-to-acceptance approach.
Eigen::VectorXs World::getCachedLCPSolution()
{
  return mConstraintSolver->getCachedLCPSolution();
}

//==============================================================================
/// This gets the cached LCP solution, which is useful to be able to get/set
/// because it can effect the forward solutions of physics problems because of
/// our optimistic LCP-stabilization-to-acceptance approach.
void World::setCachedLCPSolution(Eigen::VectorXs X)
{
  mConstraintSolver->setCachedLCPSolution(X);
}

//==============================================================================
/// If this is true, we use finite-differencing to compute all of the
/// requested Jacobians. This override can be useful to verify if there's a
/// bug in the analytical Jacobians that's causing learning to not converge.
void World::setUseFDOverride(bool fdOverride)
{
  mUseFDOverride = fdOverride;
}

//==============================================================================
bool World::getUseFDOverride()
{
  return mUseFDOverride;
}

//==============================================================================
/// If this is true, we check all Jacobians against their finite-differencing
/// counterparts at runtime. If they aren't sufficiently close, we immediately
/// crash the program and print what went wrong and some simple replication
/// instructions.
void World::setSlowDebugResultsAgainstFD(bool slowDebug)
{
  mSlowDebugResultsAgainstFD = slowDebug;
}

//==============================================================================
bool World::getSlowDebugResultsAgainstFD()
{
  return mSlowDebugResultsAgainstFD;
}

void World::DisableWrtMass()
{
  mWrtMass = nullptr;
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
void World::setGravity(const Eigen::Vector3s& _gravity)
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
const Eigen::Vector3s& World::getGravity() const
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
  return 0;
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

dynamics::BodyNode* World::getBodyNodeIndex(size_t index)
{
  std::vector<dynamics::BodyNode*> nodes = getAllBodyNodes();
  assert(index < nodes.size());
  return nodes[index];
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
  // Add all this skeletons DOFs to the RL "action space" mapping by default
  for (int i = 0; i < _skeleton->getNumDofs(); i++)
  {
    mActionSpace.push_back(mDofs + i);
  }
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
    Eigen::VectorXs upperBound,
    Eigen::VectorXs lowerBound)
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
Eigen::VectorXs World::getMasses()
{
  return mWrtMass->get(this);
}

size_t World::getLinkMassesDims()
{
  size_t mass_dim = 0;
  for (int i = 0; i < mSkeletons.size(); i++)
  {
    mass_dim += mSkeletons[i]->getLinkMassesDims();
  }
  return mass_dim;
}

Eigen::VectorXs World::getLinkMasses()
{
  Eigen::VectorXs masses = Eigen::VectorXs::Zero(getLinkMassesDims());
  size_t cur = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t mdim = mSkeletons[i]->getLinkMassesDims();
    masses.segment(cur, mdim) = mSkeletons[i]->getLinkMasses();
    cur += mdim;
  }
  return masses;
}

Eigen::VectorXs World::getLinkMUs()
{
  Eigen::VectorXs mus = Eigen::VectorXs::Zero(getLinkMassesDims());
  size_t cur = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t mdim = mSkeletons[i]->getLinkMassesDims();
    mus.segment(cur, mdim) = mSkeletons[i]->getLinkMUs();
    cur += mdim;
  }
  return mus;
}

Eigen::VectorXs World::getLinkCOMs()
{
  Eigen::VectorXs coms = Eigen::VectorXs::Zero(3 * getLinkMassesDims());
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    Eigen::VectorXs skel_coms = mSkeletons[i]->getLinkCOMs();
    coms.segment(cursor, skel_coms.size());
    cursor += skel_coms.size();
  }
  return coms;
}

Eigen::VectorXs World::getLinkMOIs()
{
  Eigen::VectorXs mois = Eigen::VectorXs::Zero(6 * getLinkMassesDims());
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    Eigen::VectorXs skel_mois = mSkeletons[i]->getLinkMOIs();
    mois.segment(cursor, skel_mois.size());
    cursor += skel_mois.size();
  }
  return mois;
}

s_t World::getLinkMassIndex(size_t index)
{
  Eigen::VectorXs masses = getLinkMasses();
  return masses(index);
}

s_t World::getLinkMUIndex(size_t index)
{
  Eigen::VectorXs mus = getLinkMUs();
  return mus(index);
}

Eigen::VectorXs World::getLinkBetas()
{
  Eigen::VectorXs betas = Eigen::VectorXs::Zero(3 * getLinkMassesDims());
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t dim = 3 * mSkeletons[i]->getLinkMassesDims();
    betas.segment(cursor, dim) = mSkeletons[i]->getLinkBetas();
    cursor += dim;
  }
  return betas;
}

Eigen::Vector3s World::getLinkBetaIndex(size_t index)
{
  Eigen::VectorXs betas = getLinkBetas();
  return betas.segment(index * 3, 3);
}

Eigen::Vector3s World::getLinkCOMIndex(size_t index)
{
  size_t probe = 0;
  size_t skeleton_id = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    probe += mSkeletons[i]->getNumBodyNodes();
    if (index < probe)
    {
      skeleton_id = i;
      probe -= mSkeletons[i]->getNumBodyNodes();
      break;
    }
  }
  return mSkeletons[skeleton_id]->getLinkCOMIndex(index - probe);
}

Eigen::Vector6s World::getLinkMOIIndex(size_t index)
{
  size_t probe = 0;
  size_t skeleton_id = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    probe += mSkeletons[i]->getNumBodyNodes();
    if (index < probe)
    {
      skeleton_id = i;
      probe -= mSkeletons[i]->getNumBodyNodes();
      break;
    }
  }
  return mSkeletons[skeleton_id]->getLinkMOIIndex(index - probe);
}

//==============================================================================
Eigen::VectorXs World::getPositions()
{
  Eigen::VectorXs positions = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getVelocities()
{
  Eigen::VectorXs velocities = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getAccelerations()
{
  Eigen::VectorXs velocities = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getControlForces()
{
  Eigen::VectorXs forces = Eigen::VectorXs(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    forces.segment(cursor, dofs) = mSkeletons[i]->getControlForces();
    cursor += dofs;
  }
  return forces;
}

//==============================================================================
Eigen::VectorXs World::getControlForceUpperLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getControlForceUpperLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs World::getControlForceLowerLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    limits.segment(cursor, dofs) = mSkeletons[i]->getControlForceLowerLimits();
    cursor += dofs;
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs World::getPositionUpperLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getPositionLowerLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getVelocityUpperLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getVelocityLowerLimits()
{
  Eigen::VectorXs limits = Eigen::VectorXs(mDofs);
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
Eigen::VectorXs World::getMassUpperLimits()
{
  return mWrtMass->upperBound(this);
}

//==============================================================================
// This gives the vector of mass lower limits for all the registered bodies in
// this world
Eigen::VectorXs World::getMassLowerLimits()
{
  return mWrtMass->lowerBound(this);
}

//==============================================================================
void World::setPositions(Eigen::VectorXs position)
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
void World::setVelocities(Eigen::VectorXs velocity)
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
void World::setAccelerations(Eigen::VectorXs accelerations)
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
void World::setControlForces(Eigen::VectorXs forces)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setControlForces(forces.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setControlForceUpperLimits(Eigen::VectorXs limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setControlForceUpperLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setControlForceLowerLimits(Eigen::VectorXs limits)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    mSkeletons[i]->setControlForceLowerLimits(limits.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
void World::setPositionUpperLimits(Eigen::VectorXs limits)
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
void World::setPositionLowerLimits(Eigen::VectorXs limits)
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
void World::setVelocityUpperLimits(Eigen::VectorXs limits)
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
void World::setVelocityLowerLimits(Eigen::VectorXs limits)
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
void World::setMasses(Eigen::VectorXs masses)
{
  mWrtMass->set(this, masses);
}

void World::setLinkMasses(Eigen::VectorXs masses)
{
  assert(masses.size() == getLinkMassesDims());
  size_t cur = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t mdim = mSkeletons[i]->getLinkMassesDims();
    mSkeletons[i]->setLinkMasses(masses.segment(cur, mdim));
    cur += mdim;
  }
}

void World::setLinkMassIndex(s_t mass, size_t index)
{
  Eigen::VectorXs masses = getLinkMasses();
  masses(index) = mass;
  setLinkMasses(masses);
}

void World::setLinkMUs(Eigen::VectorXs mus)
{
  assert(mus.size() == getLinkMassesDims());
  size_t cur = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t mdim = mSkeletons[i]->getLinkMassesDims();
    mSkeletons[i]->setLinkMUs(mus.segment(cur, mdim));
    cur += mdim;
  }
}

void World::setLinkMUIndex(s_t mu, size_t index)
{
  Eigen::VectorXs mus = getLinkMUs();
  mus(index) = mu;
  setLinkMUs(mus);
}

void World::setLinkBetas(Eigen::VectorXs betas)
{
  assert(betas.size() == getLinkMassesDims() * 3);
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t dim = mSkeletons[i]->getLinkMassesDims() * 3;
    mSkeletons[i]->setLinkBetas(betas.segment(cursor, dim));
    cursor += dim;
  }
}

void World::setLinkBetaIndex(Eigen::Vector3s beta, size_t index)
{
  Eigen::VectorXs betas = getLinkBetas();
  betas.segment(index * 3, 3) = beta;
  setLinkBetas(betas);
}

void World::setLinkCOMIndex(Eigen::Vector3s com, size_t index)
{
  size_t probe = 0;
  size_t skeleton_id = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    probe += mSkeletons[i]->getNumBodyNodes();
    if (index < probe)
    {
      skeleton_id = i;
      probe -= mSkeletons[i]->getNumBodyNodes();
      break;
    }
  }
  mSkeletons[skeleton_id]->setLinkCOMIndex(com, index - probe);
}

void World::setLinkCOMs(Eigen::VectorXs coms)
{
  assert(coms.size() == getLinkMassesDims() * 3);
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t dim = mSkeletons[i]->getLinkMassesDims() * 3;
    mSkeletons[i]->setLinkCOMs(coms.segment(cursor, dim));
    cursor += dim;
  }
}

void World::setLinkMOIIndex(Eigen::Vector6s moi, size_t index)
{
  size_t probe = 0;
  size_t skeleton_id = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    probe += mSkeletons[i]->getNumBodyNodes();
    if (index < probe)
    {
      skeleton_id = i;
      probe -= mSkeletons[i]->getNumBodyNodes();
      break;
    }
  }
  mSkeletons[skeleton_id]->setLinkMOIIndex(moi, index - probe);
}

void World::setLinkMOIs(Eigen::VectorXs mois)
{
  size_t cursor = 0;
  for (size_t i = 0; i < mSkeletons.size(); i++)
  {
    size_t dim = 6 * mSkeletons[i]->getLinkMassesDims();
    mSkeletons[i]->setLinkMOIs(mois.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This gives the C(pos, vel) vector for all the skeletons in the world,
/// without accounting for the external forces
Eigen::VectorXs World::getCoriolisAndGravityForces()
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getNumDofs());
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
Eigen::VectorXs World::getCoriolisAndGravityAndExternalForces()
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getNumDofs());
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
Eigen::MatrixXs World::getMassMatrix()
{
  Eigen::MatrixXs massMatrix = Eigen::MatrixXs::Zero(mDofs, mDofs);
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
Eigen::MatrixXs World::getInvMassMatrix()
{
  Eigen::MatrixXs invMassMatrix = Eigen::MatrixXs::Zero(mDofs, mDofs);
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
// This sets all the positions of the joints to within their limit range, if
// they're currently outside it.
void World::clampPositionsToLimits()
{
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    mSkeletons[i]->clampPositionsToLimits();
  }
}

//==============================================================================
// The state is [pos, vel] concatenated, so this return 2*getNumDofs()
int World::getStateSize()
{
  return 2 * getNumDofs();
}

//==============================================================================
// This takes a single state vector and calls setPositions() and setVelocities()
// on the head and tail, respectively
void World::setState(Eigen::VectorXs state)
{
  int dofs = getNumDofs();
  if (state.size() != 2 * dofs)
  {
    std::cerr << "World::setState() called with a vector of incorrect size ("
              << state.size() << ") instead of getStateSize() ("
              << getStateSize() << "). Ignoring call." << std::endl;
    return;
  }
  setPositions(state.head(dofs));
  setVelocities(state.tail(dofs));
}

//==============================================================================
// This return the concatenation of [pos, vel]
Eigen::VectorXs World::getState()
{
  int dofs = getNumDofs();
  Eigen::VectorXs state = Eigen::VectorXs::Zero(dofs * 2);
  state.head(dofs) = getPositions();
  state.tail(dofs) = getVelocities();
  return state;
}

//==============================================================================
// The action dim is given by the size of the action mapping. This defaults to a
// 1-1 map onto control forces, but can be configured to be just a subset of the
// control forces, if there are several DOFs that are uncontrolled.
int World::getActionSize()
{
  return mActionSpace.size();
}

//==============================================================================
// This sets the control forces, using the action mapping to decide how to map
// the passed in vector to control forces. Unmapped control forces are set to 0.
void World::setAction(Eigen::VectorXs action)
{
  if (action.size() != mActionSpace.size())
  {
    std::cerr << "World::setAction() got an action vector of incorrect size. "
                 "Expected "
              << mActionSpace.size() << " but got " << action.size()
              << ". Ignoring call." << std::endl;
    return;
  }
  Eigen::VectorXs forces = Eigen::VectorXs::Zero(getNumDofs());
  for (int i = 0; i < mActionSpace.size(); i++)
  {
    int mapping = mActionSpace[i];
    if (mapping < 0 || mapping >= forces.size())
    {
      std::cerr << "World::setAction() discovered out-of-bounds action "
                   "mapping. Index "
                << i << " -> " << mapping << ", out of bounds of [0,"
                << forces.size() << "). Ignoring call." << std::endl;
      return;
    }
    forces(mapping) = action(i);
  }
  setControlForces(forces);
}

//==============================================================================
// This reads the control forces and runs them through the action mapping to
// construct a vector for the currently set action.
Eigen::VectorXs World::getAction()
{
  Eigen::VectorXs action = Eigen::VectorXs::Zero(mActionSpace.size());
  Eigen::VectorXs forces = getControlForces();
  for (int i = 0; i < mActionSpace.size(); i++)
  {
    int mapping = mActionSpace[i];
    if (mapping < 0 || mapping >= forces.size())
    {
      std::cerr << "World::getAction() discovered out-of-bounds action "
                   "mapping. Index "
                << i << " -> " << mapping << ", out of bounds of [0,"
                << forces.size() << "). Returning 0s from call." << std::endl;
      return action;
    }
    action(i) = forces(mapping);
  }
  return action;
}

//==============================================================================
// This sets the mapping that will be used for the action. Each index of
// `mapping` is an integer corresponding to an index in the control forces
// vector
void World::setActionSpace(std::vector<int> mapping)
{
  int dofs = getNumDofs();
  for (int i = 0; i < mapping.size(); i++)
  {
    int m = mapping[i];
    if (m < 0 || m >= dofs)
    {
      std::cerr << "World::setActionMapping() discovered out-of-bounds action "
                   "mapping. Index "
                << i << " -> " << m << ", out of bounds of [0," << dofs
                << "). Ignoring call." << std::endl;
      return;
    }
    for (int j = 0; j < mapping.size(); j++)
    {
      if (j == i)
        continue;
      if (mapping[i] == mapping[j])
      {
        std::cerr << "World::setActionMapping() discovered duplicate action "
                     "mapping. Index "
                  << i << " -> " << mapping[i] << ", but we also get index "
                  << j << " -> " << mapping[j] << ". Ignoring call."
                  << std::endl;
        return;
      }
    }
  }
  mActionSpace = mapping;
}

//==============================================================================
// This returns the action mapping set by `setActionMapping()`. Each index of
// the returned mapping is an integer corresponding to an index in the control
// forces vector.
std::vector<int> World::getActionSpace()
{
  return mActionSpace;
}

//==============================================================================
// This is a shorthand method to remove a DOF from the action vector. No-op if
// the dof is already not in the action vector.
void World::removeDofFromActionSpace(int index)
{
  mActionSpace.erase(
      std::remove(mActionSpace.begin(), mActionSpace.end(), index),
      mActionSpace.end());
}

//==============================================================================
// This is a shorthand method to add a DOF from the action vector, at the end of
// the mapping space. No-op if the dof is already in the action vector.
void World::addDofToActionSpace(int index)
{
  int dofs = getNumDofs();
  if (index < 0 || index >= dofs)
  {
    std::cerr << "World::addDofToActionSpace() attempting to add out-of-bounds "
                 "action mapping. Attempting to add "
              << index << ", out of bounds of [0," << dofs
              << "). Ignoring call." << std::endl;
    return;
  }
  if (std::find(mActionSpace.begin(), mActionSpace.end(), index)
      == mActionSpace.end())
  {
    mActionSpace.push_back(index);
  }
}

//==============================================================================
/// This gets a backprop snapshot for the current state, (re)computing if
/// necessary
std::shared_ptr<neural::BackpropSnapshot> World::getCachedBackpropSnapshot()
{
  int dofs = getNumDofs();
  if (mCachedSnapshotPtr == nullptr || mCachedSnapshotPos.size() != dofs
      || mCachedSnapshotVel.size() != dofs
      || mCachedSnapshotForce.size() != dofs
      || mCachedSnapshotPos != getPositions()
      || mCachedSnapshotVel != getVelocities()
      || mCachedSnapshotForce != getControlForces())
  {
    mCachedSnapshotPos = getPositions();
    mCachedSnapshotVel = getVelocities();
    mCachedSnapshotForce = getControlForces();
    mCachedSnapshotPtr = neural::forwardPass(shared_from_this(), true);
  }
  return mCachedSnapshotPtr;
}

//==============================================================================
// This returns the Jacobian for state_t -> state_{t+1}.
Eigen::MatrixXs World::getStateJacobian()
{
  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = getCachedBackpropSnapshot();
  int dofs = getNumDofs();
  Eigen::MatrixXs stateJac = Eigen::MatrixXs::Zero(2 * dofs, 2 * dofs);
  WorldPtr sharedThis = shared_from_this();
  stateJac.block(0, 0, dofs, dofs) = snapshot->getPosPosJacobian(sharedThis);
  stateJac.block(dofs, 0, dofs, dofs) = snapshot->getPosVelJacobian(sharedThis);
  stateJac.block(0, dofs, dofs, dofs) = snapshot->getVelPosJacobian(sharedThis);
  stateJac.block(dofs, dofs, dofs, dofs)
      = snapshot->getVelVelJacobian(sharedThis);
  return stateJac;
}

//==============================================================================
// This returns the Jacobian for action_t -> state_{t+1}.
Eigen::MatrixXs World::getActionJacobian()
{
  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = getCachedBackpropSnapshot();
  int dofs = getNumDofs();
  WorldPtr sharedThis = shared_from_this();
  const Eigen::MatrixXs& forceVelJac
      = snapshot->getControlForceVelJacobian(sharedThis);

  int actionDim = mActionSpace.size();
  Eigen::MatrixXs actionJac = Eigen::MatrixXs::Zero(2 * dofs, actionDim);
  for (int i = 0; i < actionDim; i++)
  {
    actionJac.block(dofs, i, dofs, 1) = forceVelJac.col(mActionSpace[i]);
  }
  return actionJac;
}

//==============================================================================
Eigen::MatrixXs World::finiteDifferenceStateJacobian()
{
  WorldPtr sharedThis = shared_from_this();
  neural::RestorableSnapshot snapshot(sharedThis);

  int stateDim = getStateSize();
  Eigen::VectorXs originalState = getState();
  Eigen::MatrixXs stateJac = Eigen::MatrixXs::Zero(stateDim, stateDim);

  s_t EPS = 1e-6;

  for (int i = 0; i < stateDim; i++)
  {
    Eigen::VectorXs perturbedState = originalState;
    perturbedState(i) += EPS;
    setState(perturbedState);
    step(false);
    Eigen::VectorXs statePos = getState();
    snapshot.restore();

    perturbedState = originalState;
    perturbedState(i) -= EPS;
    setState(perturbedState);
    step(false);
    Eigen::VectorXs stateNeg = getState();
    snapshot.restore();

    stateJac.col(i) = (statePos - stateNeg) / (2 * EPS);
  }

  snapshot.restore();

  return stateJac;
}

//==============================================================================
Eigen::MatrixXs World::finiteDifferenceActionJacobian()
{
  WorldPtr sharedThis = shared_from_this();
  neural::RestorableSnapshot snapshot(sharedThis);

  int dofs = getNumDofs();
  int actionDim = mActionSpace.size();
  Eigen::VectorXs originalAction = getAction();
  Eigen::MatrixXs actionJac = Eigen::MatrixXs::Zero(2 * dofs, actionDim);

  s_t EPS = 1e-6;

  for (int i = 0; i < actionDim; i++)
  {
    Eigen::VectorXs perturbedAction = originalAction;
    perturbedAction(i) += EPS;
    setAction(perturbedAction);
    step(false);
    Eigen::VectorXs statePos = getState();
    snapshot.restore();

    perturbedAction = originalAction;
    perturbedAction(i) -= EPS;
    setAction(perturbedAction);
    step(false);
    Eigen::VectorXs stateNeg = getState();
    snapshot.restore();

    actionJac.col(i) = (statePos - stateNeg) / (2 * EPS);
  }

  snapshot.restore();

  return actionJac;
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

  Eigen::VectorXs state(getIndex(nSkeletons) + 6 * nContacts);
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
const Eigen::VectorXs& World::getLastPreConstraintVelocity() const
{
  return mLastPreConstraintVelocity;
}

/// This gets the Jacobian relating how changing our current position will
/// change our next position after a step. Intuitively, you'd expect this to
/// just be an identity matrix, and often it is, but if we have any FreeJoints
/// or BallJoints things get more complicated, because they actually use a
/// complicated function to integrate to the next position.
Eigen::MatrixXs World::getPosPosJacobian() const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mDofs, mDofs);
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
Eigen::MatrixXs World::getVelPosJacobian() const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mDofs, mDofs);
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
