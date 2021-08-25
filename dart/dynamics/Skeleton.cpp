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

#include "dart/dynamics/Skeleton.hpp"

#include <algorithm>
#include <array>
#include <queue>
#include <string>
#include <vector>

#include "dart/common/Console.hpp"
#include "dart/common/Deprecated.hpp"
#include "dart/common/StlHelpers.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/EndEffector.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Marker.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PointMass.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/SoftBodyNode.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

#define SET_ALL_FLAGS(X)                                                       \
  for (auto& cache : mTreeCache)                                               \
    cache.mDirty.X = true;                                                     \
  mSkelCache.mDirty.X = true;

#define SET_FLAG(Y, X)                                                         \
  mTreeCache[Y].mDirty.X = true;                                               \
  mSkelCache.mDirty.X = true;

#define ON_ALL_TREES(X)                                                        \
  for (std::size_t i = 0; i < mTreeCache.size(); ++i)                          \
    X(i);

#define CHECK_CONFIG_VECTOR_SIZE(V)                                            \
  if (V.size() > 0)                                                            \
  {                                                                            \
    if (nonzero_size != INVALID_INDEX                                          \
        && V.size() != static_cast<int>(nonzero_size))                         \
    {                                                                          \
      dterr << "[Skeleton::Configuration] Mismatch in size of vector [" << #V  \
            << "] (expected " << nonzero_size << " | found " << V.size()       \
            << "\n";                                                           \
      assert(false);                                                           \
    }                                                                          \
    else if (nonzero_size == INVALID_INDEX)                                    \
      nonzero_size = V.size();                                                 \
  }

namespace dart {
namespace dynamics {

namespace detail {

//==============================================================================
/// Templated function for passing each entry in a std::vector<Data> into each
/// member of an array of Objects belonging to some Owner class.
///
/// The ObjectBase argument should be the base class of Object in which the
/// setData function is defined. In many cases, ObjectBase may be the same as
/// Object, but it is not always.
//
// TODO(MXG): Consider putting this in an accessible header if it might be
// useful in other places.
template <
    class Owner,
    class Object,
    class ObjectBase,
    class Data,
    std::size_t (Owner::*getNumObjects)() const,
    Object* (Owner::*getObject)(std::size_t),
    void (ObjectBase::*setData)(const Data&)>
void setAllMemberObjectData(Owner* owner, const std::vector<Data>& data)
{
  if (!owner)
  {
    dterr << "[setAllMemberObjectData] Attempting to set ["
          << typeid(Data).name() << "] of every [" << typeid(Object).name()
          << "] in a nullptr [" << typeid(Owner).name() << "]. Please report "
          << "this as a bug!\n";
    assert(false);
    return;
  }

  std::size_t numObjects = (owner->*getNumObjects)();

  if (data.size() != numObjects)
  {
    dtwarn << "[setAllMemberObjectData] Mismatch between the number of ["
           << typeid(Object).name() << "] member objects (" << numObjects
           << ") in the [" << typeid(Owner).name() << "] named ["
           << owner->getName() << "] (" << owner << ") and the number of ["
           << typeid(Object).name() << "] which is (" << data.size()
           << ") while setting [" << typeid(Data).name() << "]\n"
           << " -- We will set (" << std::min(numObjects, data.size())
           << ") of them.\n";
    numObjects = std::min(numObjects, data.size());
  }

  for (std::size_t i = 0; i < numObjects; ++i)
    ((owner->*getObject)(i)->*setData)(data[i]);
}

//==============================================================================
/// Templated function for aggregating a std::vector<Data> out of each member of
/// an array of Objects belonging to some Owner class.
///
/// The ObjectBase argument should be the base class of Object in which the
/// getData function is defined. In many cases, ObjectBase may be the same as
/// Object, but it is not always.
//
// TODO(MXG): Consider putting this in an accessible header if it might be
// useful in other places.
template <
    class Owner,
    class Object,
    class ObjectBase,
    class Data,
    std::size_t (Owner::*getNumObjects)() const,
    const Object* (Owner::*getObject)(std::size_t) const,
    Data (ObjectBase::*getData)() const>
std::vector<Data> getAllMemberObjectData(const Owner* owner)
{
  if (!owner)
  {
    dterr << "[getAllMemberObjectData] Attempting to get the ["
          << typeid(Data).name() << "] from every [" << typeid(Object).name()
          << "] in a nullptr [" << typeid(Owner).name() << "]. Please report "
          << "this as a bug!\n";
    assert(false);
    return std::vector<Data>();
  }

  const std::size_t numObjects = (owner->*getNumObjects)();
  std::vector<Data> data;
  data.reserve(numObjects);

  for (std::size_t i = 0; i < numObjects; ++i)
    data.push_back(((owner->*getObject)(i)->*getData)());

  return data;
}

//==============================================================================
SkeletonAspectProperties::SkeletonAspectProperties(
    const std::string& _name,
    bool _isMobile,
    const Eigen::Vector3s& _gravity,
    s_t _timeStep,
    bool _enabledSelfCollisionCheck,
    bool _enableAdjacentBodyCheck)
  : mName(_name),
    mIsMobile(_isMobile),
    mGravity(_gravity),
    mTimeStep(_timeStep),
    mEnabledSelfCollisionCheck(_enabledSelfCollisionCheck),
    mEnabledAdjacentBodyCheck(_enableAdjacentBodyCheck)
{
  // Do nothing
}

//==============================================================================
void setAllBodyNodeStates(Skeleton* skel, const BodyNodeStateVector& states)
{
  setAllMemberObjectData<
      Skeleton,
      BodyNode,
      common::Composite,
      common::Composite::State,
      &Skeleton::getNumBodyNodes,
      &Skeleton::getBodyNode,
      &common::Composite::setCompositeState>(skel, states);
}

//==============================================================================
BodyNodeStateVector getAllBodyNodeStates(const Skeleton* skel)
{
  return getAllMemberObjectData<
      Skeleton,
      BodyNode,
      common::Composite,
      common::Composite::State,
      &Skeleton::getNumBodyNodes,
      &Skeleton::getBodyNode,
      &common::Composite::getCompositeState>(skel);
}

//==============================================================================
void setAllBodyNodeProperties(
    Skeleton* skel, const BodyNodePropertiesVector& properties)
{
  setAllMemberObjectData<
      Skeleton,
      BodyNode,
      common::Composite,
      common::Composite::Properties,
      &Skeleton::getNumBodyNodes,
      &Skeleton::getBodyNode,
      &common::Composite::setCompositeProperties>(skel, properties);
}

//==============================================================================
BodyNodePropertiesVector getAllBodyNodeProperties(const Skeleton* skel)
{
  return getAllMemberObjectData<
      Skeleton,
      BodyNode,
      common::Composite,
      common::Composite::Properties,
      &Skeleton::getNumBodyNodes,
      &Skeleton::getBodyNode,
      &common::Composite::getCompositeProperties>(skel);
}

//==============================================================================
void setAllJointStates(Skeleton* skel, const BodyNodeStateVector& states)
{
  setAllMemberObjectData<
      Skeleton,
      Joint,
      common::Composite,
      common::Composite::State,
      &Skeleton::getNumJoints,
      &Skeleton::getJoint,
      &common::Composite::setCompositeState>(skel, states);
}

//==============================================================================
BodyNodeStateVector getAllJointStates(const Skeleton* skel)
{
  return getAllMemberObjectData<
      Skeleton,
      Joint,
      common::Composite,
      common::Composite::State,
      &Skeleton::getNumJoints,
      &Skeleton::getJoint,
      &common::Composite::getCompositeState>(skel);
}

//==============================================================================
void setAllJointProperties(
    Skeleton* skel, const BodyNodePropertiesVector& properties)
{
  setAllMemberObjectData<
      Skeleton,
      Joint,
      common::Composite,
      common::Composite::Properties,
      &Skeleton::getNumJoints,
      &Skeleton::getJoint,
      &common::Composite::setCompositeProperties>(skel, properties);
}

//==============================================================================
BodyNodePropertiesVector getAllJointProperties(const Skeleton* skel)
{
  return getAllMemberObjectData<
      Skeleton,
      Joint,
      common::Composite,
      common::Composite::Properties,
      &Skeleton::getNumJoints,
      &Skeleton::getJoint,
      &common::Composite::getCompositeProperties>(skel);
}

} // namespace detail

//==============================================================================
Skeleton::Configuration::Configuration(
    const Eigen::VectorXs& positions,
    const Eigen::VectorXs& velocities,
    const Eigen::VectorXs& accelerations,
    const Eigen::VectorXs& forces,
    const Eigen::VectorXs& commands)
  : mPositions(positions),
    mVelocities(velocities),
    mAccelerations(accelerations),
    mControlForces(forces),
    mCommands(commands)
{
  std::size_t nonzero_size = INVALID_INDEX;

  CHECK_CONFIG_VECTOR_SIZE(positions);
  CHECK_CONFIG_VECTOR_SIZE(velocities);
  CHECK_CONFIG_VECTOR_SIZE(accelerations);
  CHECK_CONFIG_VECTOR_SIZE(forces);
  CHECK_CONFIG_VECTOR_SIZE(commands);

  if (nonzero_size != INVALID_INDEX)
  {
    for (std::size_t i = 0; i < nonzero_size; ++i)
      mIndices.push_back(i);
  }
}

//==============================================================================
Skeleton::Configuration::Configuration(
    const std::vector<std::size_t>& indices,
    const Eigen::VectorXs& positions,
    const Eigen::VectorXs& velocities,
    const Eigen::VectorXs& accelerations,
    const Eigen::VectorXs& forces,
    const Eigen::VectorXs& commands)
  : mIndices(indices),
    mPositions(positions),
    mVelocities(velocities),
    mAccelerations(accelerations),
    mControlForces(forces),
    mCommands(commands)
{
  std::size_t nonzero_size = indices.size();

  CHECK_CONFIG_VECTOR_SIZE(positions);
  CHECK_CONFIG_VECTOR_SIZE(velocities);
  CHECK_CONFIG_VECTOR_SIZE(accelerations);
  CHECK_CONFIG_VECTOR_SIZE(forces);
  CHECK_CONFIG_VECTOR_SIZE(commands);
}

//==============================================================================
#define RETURN_IF_CONFIG_VECTOR_IS_INEQ(V)                                     \
  if (V.size() != other.V.size())                                              \
    return false;                                                              \
  if (V != other.V)                                                            \
    return false;

//==============================================================================
bool Skeleton::Configuration::operator==(const Configuration& other) const
{
  if (mIndices != other.mIndices)
    return false;

  RETURN_IF_CONFIG_VECTOR_IS_INEQ(mPositions);
  RETURN_IF_CONFIG_VECTOR_IS_INEQ(mVelocities);
  RETURN_IF_CONFIG_VECTOR_IS_INEQ(mAccelerations);
  RETURN_IF_CONFIG_VECTOR_IS_INEQ(mControlForces);
  RETURN_IF_CONFIG_VECTOR_IS_INEQ(mCommands);

  return true;
}

//==============================================================================
bool Skeleton::Configuration::operator!=(const Configuration& other) const
{
  return !(*this == other);
}

//==============================================================================
SkeletonPtr Skeleton::create(const std::string& _name)
{
  return create(AspectPropertiesData(_name));
}

//==============================================================================
SkeletonPtr Skeleton::create(const AspectPropertiesData& properties)
{
  SkeletonPtr skel(new Skeleton(properties));
  skel->setPtr(skel);
  return skel;
}

//==============================================================================
SkeletonPtr Skeleton::getPtr()
{
  return mPtr.lock();
}

//==============================================================================
ConstSkeletonPtr Skeleton::getPtr() const
{
  return mPtr.lock();
}

//==============================================================================
SkeletonPtr Skeleton::getSkeleton()
{
  return mPtr.lock();
}

//==============================================================================
ConstSkeletonPtr Skeleton::getSkeleton() const
{
  return mPtr.lock();
}

//==============================================================================
std::mutex& Skeleton::getMutex() const
{
  return mMutex;
}

//==============================================================================
std::unique_ptr<common::LockableReference> Skeleton::getLockableReference()
    const
{
  return std::make_unique<common::SingleLockableReference<std::mutex>>(
      mPtr, mMutex);
}

//==============================================================================
Skeleton::~Skeleton()
{
  for (BodyNode* bn : mSkelCache.mBodyNodes)
    delete bn;
}

//==============================================================================
SkeletonPtr Skeleton::clone() const
{
  return cloneSkeleton(getName());
}

//==============================================================================
SkeletonPtr Skeleton::clone(const std::string& cloneName) const
{
  return cloneSkeleton(cloneName);
}

//==============================================================================
SkeletonPtr Skeleton::cloneSkeleton() const
{
  return cloneSkeleton(getName());
}

//==============================================================================
SkeletonPtr Skeleton::cloneSkeleton(const std::string& cloneName) const
{
  SkeletonPtr skelClone = Skeleton::create(cloneName);

  for (std::size_t i = 0; i < getNumBodyNodes(); ++i)
  {
    // Create a clone of the parent Joint
    Joint* joint = getJoint(i)->clone();

    // Identify the original parent BodyNode
    const BodyNode* originalParent = getBodyNode(i)->getParentBodyNode();

    // Grab the parent BodyNode clone (using its name, which is guaranteed to be
    // unique), or use nullptr if this is a root BodyNode
    BodyNode* parentClone
        = (originalParent == nullptr)
              ? nullptr
              : skelClone->getBodyNode(originalParent->getName());

    if ((nullptr != originalParent) && (nullptr == parentClone))
    {
      dterr << "[Skeleton::clone] Failed to find a clone of BodyNode named ["
            << originalParent->getName() << "] which is needed as the parent "
            << "of the BodyNode named [" << getBodyNode(i)->getName()
            << "] and should already have been created. Please report this as "
            << "a bug!\n";
    }

    BodyNode* newBody = getBodyNode(i)->clone(parentClone, joint, false);

    skelClone->registerBodyNode(newBody);
  }

  // Clone over the nodes in such a way that their indexing will match up with
  // the original
  for (const auto& nodeType : mNodeMap)
  {
    for (const auto& node : nodeType.second)
    {
      const BodyNode* originalBn = node->getBodyNodePtr();
      BodyNode* newBn = skelClone->getBodyNode(originalBn->getName());
      node->cloneNode(newBn)->attach();
    }
  }

  skelClone->setProperties(getAspectProperties());
  skelClone->setName(cloneName);
  skelClone->setState(getState());

  // Fix mimic joint references
  for (std::size_t i = 0; i < getNumJoints(); ++i)
  {
    Joint* joint = skelClone->getJoint(i);
    if (joint->getActuatorType() == Joint::MIMIC)
    {
      const Joint* mimicJoint
          = skelClone->getJoint(joint->getMimicJoint()->getName());
      if (mimicJoint)
      {
        joint->setMimicJoint(
            mimicJoint, joint->getMimicMultiplier(), joint->getMimicOffset());
      }
      else
      {
        dterr << "[Skeleton::clone] Failed to clone mimic joint successfully: "
              << "Unable to find the mimic joint ["
              << joint->getMimicJoint()->getName()
              << "] in the cloned Skeleton. Please report this as a bug!\n";
      }
    }
  }

  return skelClone;
}

//==============================================================================
MetaSkeletonPtr Skeleton::cloneMetaSkeleton(const std::string& cloneName) const
{
  return cloneSkeleton(cloneName);
}

//==============================================================================
#define SET_CONFIG_VECTOR(V)                                                   \
  if (configuration.m##V.size() > 0)                                           \
  {                                                                            \
    if (static_cast<int>(configuration.mIndices.size())                        \
        != configuration.m##V.size())                                          \
    {                                                                          \
      dterr << "[Skeleton::setConfiguration] Mismatch in size of vector ["     \
            << #V << "] (expected " << configuration.mIndices.size()           \
            << " | found " << configuration.m##V.size() << "\n";               \
      assert(false);                                                           \
    }                                                                          \
    else                                                                       \
      set##V(configuration.mIndices, configuration.m##V);                      \
  }

//==============================================================================
void Skeleton::setConfiguration(const Configuration& configuration)
{
  SET_CONFIG_VECTOR(Positions);
  SET_CONFIG_VECTOR(Velocities);
  SET_CONFIG_VECTOR(Accelerations);
  SET_CONFIG_VECTOR(ControlForces);
  SET_CONFIG_VECTOR(Commands);
}

//==============================================================================
Skeleton::Configuration Skeleton::getConfiguration(int flags) const
{
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < getNumDofs(); ++i)
    indices.push_back(i);

  return getConfiguration(indices, flags);
}

//==============================================================================
Skeleton::Configuration Skeleton::getConfiguration(
    const std::vector<std::size_t>& indices, int flags) const
{
  Configuration config(indices);
  if (flags == CONFIG_NOTHING)
    return config;

  if ((flags & CONFIG_POSITIONS) == CONFIG_POSITIONS)
    config.mPositions = getPositions(indices);

  if ((flags & CONFIG_VELOCITIES) == CONFIG_VELOCITIES)
    config.mVelocities = getVelocities(indices);

  if ((flags & CONFIG_ACCELERATIONS) == CONFIG_ACCELERATIONS)
    config.mAccelerations = getAccelerations(indices);

  if ((flags & CONFIG_FORCES) == CONFIG_FORCES)
    config.mControlForces = getControlForces(indices);

  if ((flags & CONFIG_COMMANDS) == CONFIG_COMMANDS)
    config.mCommands = getCommands(indices);

  return config;
}

//==============================================================================
void Skeleton::setState(const State& state)
{
  setCompositeState(state);
}

//==============================================================================
Skeleton::State Skeleton::getState() const
{
  return getCompositeState();
}

//==============================================================================
void Skeleton::setProperties(const Properties& properties)
{
  setCompositeProperties(properties);
}

//==============================================================================
Skeleton::Properties Skeleton::getProperties() const
{
  return getCompositeProperties();
}

//==============================================================================
void Skeleton::setProperties(const AspectProperties& properties)
{
  setAspectProperties(properties);
}

//==============================================================================
void Skeleton::setAspectProperties(const AspectProperties& properties)
{
  setName(properties.mName);
  setMobile(properties.mIsMobile);
  setGravity(properties.mGravity);
  setTimeStep(properties.mTimeStep);
  setSelfCollisionCheck(properties.mEnabledSelfCollisionCheck);
  setAdjacentBodyCheck(properties.mEnabledAdjacentBodyCheck);
}

//==============================================================================
const Skeleton::AspectProperties& Skeleton::getSkeletonProperties() const
{
  return mAspectProperties;
}

//==============================================================================
const std::string& Skeleton::setName(const std::string& _name)
{
  if (_name == mAspectProperties.mName && !_name.empty())
    return mAspectProperties.mName;

  const std::string oldName = mAspectProperties.mName;
  mAspectProperties.mName = _name;

  mNameMgrForBodyNodes.setManagerName(
      "Skeleton::BodyNode | " + mAspectProperties.mName);
  mNameMgrForSoftBodyNodes.setManagerName(
      "Skeleton::SoftBodyNode | " + mAspectProperties.mName);
  mNameMgrForJoints.setManagerName(
      "Skeleton::Joint | " + mAspectProperties.mName);
  mNameMgrForDofs.setManagerName(
      "Skeleton::DegreeOfFreedom | " + mAspectProperties.mName);

  for (auto& mgr : mNodeNameMgrMap)
    mgr.second.setManagerName(
        std::string("Skeleton::") + mgr.first.name() + " | "
        + mAspectProperties.mName);

  ConstMetaSkeletonPtr me = mPtr.lock();
  mNameChangedSignal.raise(me, oldName, mAspectProperties.mName);

  return mAspectProperties.mName;
}

//==============================================================================
const std::string& Skeleton::getName() const
{
  return mAspectProperties.mName;
}

//==============================================================================
const std::string& Skeleton::addEntryToBodyNodeNameMgr(BodyNode* _newNode)
{
  _newNode->BodyNode::mAspectProperties.mName
      = mNameMgrForBodyNodes.issueNewNameAndAdd(_newNode->getName(), _newNode);

  return _newNode->BodyNode::mAspectProperties.mName;
}

//==============================================================================
const std::string& Skeleton::addEntryToJointNameMgr(
    Joint* _newJoint, bool _updateDofNames)
{
  _newJoint->mAspectProperties.mName
      = mNameMgrForJoints.issueNewNameAndAdd(_newJoint->getName(), _newJoint);

  if (_updateDofNames)
    _newJoint->updateDegreeOfFreedomNames();

  return _newJoint->mAspectProperties.mName;
}

//==============================================================================
void Skeleton::addEntryToSoftBodyNodeNameMgr(SoftBodyNode* _newNode)
{
  // Note: This doesn't need the same checks as BodyNode and Joint, because
  // its name has already been resolved against all the BodyNodes, which
  // includes all SoftBodyNodes.
  mNameMgrForSoftBodyNodes.addName(_newNode->getName(), _newNode);
}

//==============================================================================
void Skeleton::enableSelfCollision(bool enableAdjacentBodyCheck)
{
  enableSelfCollisionCheck();
  setAdjacentBodyCheck(enableAdjacentBodyCheck);
}

//==============================================================================
void Skeleton::disableSelfCollision()
{
  disableSelfCollisionCheck();
  setAdjacentBodyCheck(false);
}

//==============================================================================
void Skeleton::setSelfCollisionCheck(bool enable)
{
  mAspectProperties.mEnabledSelfCollisionCheck = enable;
}

//==============================================================================
bool Skeleton::getSelfCollisionCheck() const
{
  return mAspectProperties.mEnabledSelfCollisionCheck;
}

//==============================================================================
void Skeleton::enableSelfCollisionCheck()
{
  setSelfCollisionCheck(true);
}

//==============================================================================
void Skeleton::disableSelfCollisionCheck()
{
  setSelfCollisionCheck(false);
}

//==============================================================================
bool Skeleton::isEnabledSelfCollisionCheck() const
{
  return getSelfCollisionCheck();
}

//==============================================================================
void Skeleton::setAdjacentBodyCheck(bool enable)
{
  mAspectProperties.mEnabledAdjacentBodyCheck = enable;
}

//==============================================================================
bool Skeleton::getAdjacentBodyCheck() const
{
  return mAspectProperties.mEnabledAdjacentBodyCheck;
}

//==============================================================================
void Skeleton::enableAdjacentBodyCheck()
{
  setAdjacentBodyCheck(true);
}

//==============================================================================
void Skeleton::disableAdjacentBodyCheck()
{
  setAdjacentBodyCheck(false);
}

//==============================================================================
bool Skeleton::isEnabledAdjacentBodyCheck() const
{
  return getAdjacentBodyCheck();
}

//==============================================================================
void Skeleton::setMobile(bool _isMobile)
{
  mAspectProperties.mIsMobile = _isMobile;
}

//==============================================================================
bool Skeleton::isMobile() const
{
  return mAspectProperties.mIsMobile;
}

//==============================================================================
void Skeleton::setTimeStep(s_t _timeStep)
{
  assert(_timeStep > 0.0);
  mAspectProperties.mTimeStep = _timeStep;

  for (std::size_t i = 0; i < mTreeCache.size(); ++i)
    dirtyArticulatedInertia(i);
}

//==============================================================================
s_t Skeleton::getTimeStep() const
{
  return mAspectProperties.mTimeStep;
}

//==============================================================================
void Skeleton::setGravity(const Eigen::Vector3s& _gravity)
{
  mAspectProperties.mGravity = _gravity;
  SET_ALL_FLAGS(mGravityForces);
  SET_ALL_FLAGS(mCoriolisAndGravityForces);
  ON_ALL_TREES(dirtySupportPolygon);
}

//==============================================================================
const Eigen::Vector3s& Skeleton::getGravity() const
{
  return mAspectProperties.mGravity;
}

//==============================================================================
std::size_t Skeleton::getNumBodyNodes() const
{
  return mSkelCache.mBodyNodes.size();
}

//==============================================================================
std::size_t Skeleton::getNumRigidBodyNodes() const
{
  return mSkelCache.mBodyNodes.size() - mSoftBodyNodes.size();
}

//==============================================================================
std::size_t Skeleton::getNumSoftBodyNodes() const
{
  return mSoftBodyNodes.size();
}

//==============================================================================
std::size_t Skeleton::getNumTrees() const
{
  return mTreeCache.size();
}

//==============================================================================
BodyNode* Skeleton::getRootBodyNode(std::size_t _treeIdx)
{
  if (mTreeCache.size() > _treeIdx)
    return mTreeCache[_treeIdx].mBodyNodes[0];

  if (mTreeCache.size() == 0)
  {
    dterr << "[Skeleton::getRootBodyNode] Requested a root BodyNode from a "
          << "Skeleton with no BodyNodes!\n";
    assert(false);
  }
  else
  {
    dterr << "[Skeleton::getRootBodyNode] Requested invalid root BodyNode "
          << "index (" << _treeIdx << ")! Must be less than "
          << mTreeCache.size() << ".\n";
    assert(false);
  }

  return nullptr;
}

//==============================================================================
const BodyNode* Skeleton::getRootBodyNode(std::size_t _treeIdx) const
{
  return const_cast<Skeleton*>(this)->getRootBodyNode(_treeIdx);
}

//==============================================================================
Joint* Skeleton::getRootJoint(std::size_t treeIdx)
{
  auto rootBodyNode = getRootBodyNode(treeIdx);

  if (rootBodyNode)
    return rootBodyNode->getParentJoint();

  return nullptr;
}

//==============================================================================
const Joint* Skeleton::getRootJoint(std::size_t treeIdx) const
{
  return const_cast<Skeleton*>(this)->getRootJoint(treeIdx);
}

//==============================================================================
BodyNode* Skeleton::getBodyNode(std::size_t _idx)
{
  return common::getVectorObjectIfAvailable<BodyNode*>(
      _idx, mSkelCache.mBodyNodes);
}

//==============================================================================
const BodyNode* Skeleton::getBodyNode(std::size_t _idx) const
{
  return common::getVectorObjectIfAvailable<BodyNode*>(
      _idx, mSkelCache.mBodyNodes);
}

//==============================================================================
SoftBodyNode* Skeleton::getSoftBodyNode(std::size_t _idx)
{
  return common::getVectorObjectIfAvailable<SoftBodyNode*>(
      _idx, mSoftBodyNodes);
}

//==============================================================================
const SoftBodyNode* Skeleton::getSoftBodyNode(std::size_t _idx) const
{
  return common::getVectorObjectIfAvailable<SoftBodyNode*>(
      _idx, mSoftBodyNodes);
}

//==============================================================================
BodyNode* Skeleton::getBodyNode(const std::string& _name)
{
  return mNameMgrForBodyNodes.getObject(_name);
}

//==============================================================================
const BodyNode* Skeleton::getBodyNode(const std::string& _name) const
{
  return mNameMgrForBodyNodes.getObject(_name);
}

//==============================================================================
SoftBodyNode* Skeleton::getSoftBodyNode(const std::string& _name)
{
  return mNameMgrForSoftBodyNodes.getObject(_name);
}

//==============================================================================
const SoftBodyNode* Skeleton::getSoftBodyNode(const std::string& _name) const
{
  return mNameMgrForSoftBodyNodes.getObject(_name);
}

//==============================================================================
template <class T>
static std::vector<const T*>& convertToConstPtrVector(
    const std::vector<T*>& vec, std::vector<const T*>& const_vec)
{
  const_vec.resize(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i)
    const_vec[i] = vec[i];
  return const_vec;
}

//==============================================================================
const std::vector<BodyNode*>& Skeleton::getBodyNodes()
{
  return mSkelCache.mBodyNodes;
}

//==============================================================================
const std::vector<const BodyNode*>& Skeleton::getBodyNodes() const
{
  return convertToConstPtrVector<BodyNode>(
      mSkelCache.mBodyNodes, mSkelCache.mConstBodyNodes);
}

//==============================================================================
std::vector<BodyNode*> Skeleton::getBodyNodes(const std::string& name)
{
  auto bodyNode = getBodyNode(name);

  if (bodyNode)
    return {bodyNode};
  else
    return std::vector<BodyNode*>();
}

//==============================================================================
std::vector<const BodyNode*> Skeleton::getBodyNodes(
    const std::string& name) const
{
  const auto bodyNode = getBodyNode(name);

  if (bodyNode)
    return {bodyNode};
  else
    return std::vector<const BodyNode*>();
}

//==============================================================================
bool Skeleton::hasBodyNode(const BodyNode* bodyNode) const
{
  return std::find(
             mSkelCache.mBodyNodes.begin(),
             mSkelCache.mBodyNodes.end(),
             bodyNode)
         != mSkelCache.mBodyNodes.end();
}

//==============================================================================
template <class ObjectT, std::size_t (ObjectT::*getIndexInSkeleton)() const>
static std::size_t templatedGetIndexOf(
    const Skeleton* _skel,
    const ObjectT* _obj,
    const std::string& _type,
    bool _warning)
{
  if (nullptr == _obj)
  {
    if (_warning)
    {
      dterr << "[Skeleton::getIndexOf] Requesting the index of a nullptr "
            << _type << " within the Skeleton [" << _skel->getName() << "] ("
            << _skel << ")!\n";
      assert(false);
    }
    return INVALID_INDEX;
  }

  if (_skel == _obj->getSkeleton().get())
    return (_obj->*getIndexInSkeleton)();

  if (_warning)
  {
    dterr << "[Skeleton::getIndexOf] Requesting the index of a " << _type
          << " [" << _obj->getName() << "] (" << _obj
          << ") from a Skeleton that it does "
          << "not belong to!\n";
    assert(false);
  }

  return INVALID_INDEX;
}

//==============================================================================
std::size_t Skeleton::getIndexOf(const BodyNode* _bn, bool _warning) const
{
  return templatedGetIndexOf<BodyNode, &BodyNode::getIndexInSkeleton>(
      this, _bn, "BodyNode", _warning);
}

//==============================================================================
const std::vector<BodyNode*>& Skeleton::getTreeBodyNodes(std::size_t _treeIdx)
{
  if (_treeIdx >= mTreeCache.size())
  {
    std::size_t count = mTreeCache.size();
    dterr << "[Skeleton::getTreeBodyNodes] Requesting an invalid tree ("
          << _treeIdx << ") "
          << (count > 0
                  ? (std::string("when the max tree index is (")
                     + std::to_string(count - 1) + ")\n")
                  : std::string("when there are no trees in this Skeleton\n"));
    assert(false);
  }

  return mTreeCache[_treeIdx].mBodyNodes;
}

//==============================================================================
std::vector<const BodyNode*> Skeleton::getTreeBodyNodes(
    std::size_t _treeIdx) const
{
  return convertToConstPtrVector<BodyNode>(
      mTreeCache[_treeIdx].mBodyNodes, mTreeCache[_treeIdx].mConstBodyNodes);
}

//==============================================================================
std::size_t Skeleton::getNumJoints() const
{
  // The number of joints and body nodes are identical
  return getNumBodyNodes();
}

//==============================================================================
Joint* Skeleton::getJoint(std::size_t _idx)
{
  BodyNode* bn = common::getVectorObjectIfAvailable<BodyNode*>(
      _idx, mSkelCache.mBodyNodes);
  if (bn)
    return bn->getParentJoint();

  return nullptr;
}

//==============================================================================
const Joint* Skeleton::getJoint(std::size_t _idx) const
{
  return const_cast<Skeleton*>(this)->getJoint(_idx);
}

//==============================================================================
Joint* Skeleton::getJoint(const std::string& name)
{
  return mNameMgrForJoints.getObject(name);
}

//==============================================================================
const Joint* Skeleton::getJoint(const std::string& name) const
{
  return mNameMgrForJoints.getObject(name);
}

//==============================================================================
std::vector<Joint*> Skeleton::getJoints()
{
  const auto& bodyNodes = getBodyNodes();

  std::vector<Joint*> joints;
  joints.reserve(bodyNodes.size());
  for (const auto& bodyNode : bodyNodes)
    joints.emplace_back(bodyNode->getParentJoint());

  return joints;
}

//==============================================================================
std::vector<const Joint*> Skeleton::getJoints() const
{
  const auto& bodyNodes = getBodyNodes();

  std::vector<const Joint*> joints;
  joints.reserve(bodyNodes.size());
  for (const auto& bodyNode : bodyNodes)
    joints.emplace_back(bodyNode->getParentJoint());

  return joints;
}

//==============================================================================
std::vector<Joint*> Skeleton::getJoints(const std::string& name)
{
  auto joint = getJoint(name);

  if (joint)
    return {joint};
  else
    return std::vector<Joint*>();
}

//==============================================================================
std::vector<const Joint*> Skeleton::getJoints(const std::string& name) const
{
  const auto joint = getJoint(name);

  if (joint)
    return {joint};
  else
    return std::vector<const Joint*>();
}

//==============================================================================
bool Skeleton::hasJoint(const Joint* joint) const
{
  return std::find_if(
             mSkelCache.mBodyNodes.begin(),
             mSkelCache.mBodyNodes.end(),
             [&joint](const BodyNode* bodyNode) {
               return bodyNode->getParentJoint() == joint;
             })
         != mSkelCache.mBodyNodes.end();
}

//==============================================================================
std::size_t Skeleton::getIndexOf(const Joint* _joint, bool _warning) const
{
  return templatedGetIndexOf<Joint, &Joint::getJointIndexInSkeleton>(
      this, _joint, "Joint", _warning);
}

//==============================================================================
std::size_t Skeleton::getNumDofs() const
{
  return mSkelCache.mDofs.size();
}

//==============================================================================
std::size_t Skeleton::getNumDofs(std::size_t treeIndex) const
{
  return mTreeCache[treeIndex].mDofs.size();
}

//==============================================================================
DegreeOfFreedom* Skeleton::getDof(std::size_t _idx)
{
  return common::getVectorObjectIfAvailable<DegreeOfFreedom*>(
      _idx, mSkelCache.mDofs);
}

//==============================================================================
const DegreeOfFreedom* Skeleton::getDof(std::size_t _idx) const
{
  return common::getVectorObjectIfAvailable<DegreeOfFreedom*>(
      _idx, mSkelCache.mDofs);
}

//==============================================================================
DegreeOfFreedom* Skeleton::getDof(const std::string& _name)
{
  return mNameMgrForDofs.getObject(_name);
}

//==============================================================================
const DegreeOfFreedom* Skeleton::getDof(const std::string& _name) const
{
  return mNameMgrForDofs.getObject(_name);
}

//==============================================================================
const std::vector<DegreeOfFreedom*>& Skeleton::getDofs()
{
  return mSkelCache.mDofs;
}

//==============================================================================
std::vector<const DegreeOfFreedom*> Skeleton::getDofs() const
{
  return convertToConstPtrVector<DegreeOfFreedom>(
      mSkelCache.mDofs, mSkelCache.mConstDofs);
}

//==============================================================================
std::size_t Skeleton::getIndexOf(
    const DegreeOfFreedom* _dof, bool _warning) const
{
  return templatedGetIndexOf<
      DegreeOfFreedom,
      &DegreeOfFreedom::getIndexInSkeleton>(
      this, _dof, "DegreeOfFreedom", _warning);
}

//==============================================================================
const std::vector<DegreeOfFreedom*>& Skeleton::getTreeDofs(std::size_t _treeIdx)
{
  return mTreeCache[_treeIdx].mDofs;
}

//==============================================================================
const std::vector<const DegreeOfFreedom*>& Skeleton::getTreeDofs(
    std::size_t _treeIdx) const
{
  return convertToConstPtrVector<DegreeOfFreedom>(
      mTreeCache[_treeIdx].mDofs, mTreeCache[_treeIdx].mConstDofs);
}

//==============================================================================
bool Skeleton::checkIndexingConsistency() const
{
  bool consistent = true;

  // Check each BodyNode in the Skeleton cache
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    const BodyNode* bn = mSkelCache.mBodyNodes[i];
    if (bn->mIndexInSkeleton != i)
    {
      dterr << "[Skeleton::checkIndexingConsistency] BodyNode named ["
            << bn->getName() << "] in Skeleton [" << getName() << "] is "
            << "mistaken about its index in the Skeleton (" << i << " | "
            << bn->mIndexInSkeleton << "). Please report this as a bug!"
            << std::endl;
      consistent = false;
      assert(false);
    }

    const BodyNode* nameEntryForBodyNode = getBodyNode(bn->getName());
    if (nameEntryForBodyNode != bn)
    {
      dterr << "[Skeleton::checkIndexingConsistency] Skeleton named ["
            << getName() << "] (" << this << ") is mistaken about the name of "
            << "BodyNode [" << bn->getName() << "] (" << bn << "). The name "
            << "instead maps to [" << nameEntryForBodyNode->getName() << "] ("
            << nameEntryForBodyNode << "). Please report this as a bug!"
            << std::endl;
      consistent = false;
      assert(false);
    }

    const Joint* joint = bn->getParentJoint();
    const Joint* nameEntryForJoint = getJoint(joint->getName());
    if (nameEntryForJoint != joint)
    {
      dterr << "[Skeleton::checkIndexingConsistency] Skeleton named ["
            << getName() << "] (" << this << ") is mistaken about the name of "
            << "Joint [" << joint->getName() << "] (" << joint << "). The name "
            << "instead maps to [" << nameEntryForJoint->getName() << "] ("
            << nameEntryForJoint << "). Please report this as a bug!"
            << std::endl;
      consistent = false;
      assert(false);
    }

    const BodyNode::NodeMap& nodeMap = bn->mNodeMap;
    for (const auto& nodeType : nodeMap)
    {
      const std::vector<Node*>& nodes = nodeType.second;
      for (std::size_t k = 0; k < nodes.size(); ++k)
      {
        const Node* node = nodes[k];
        if (node->getBodyNodePtr() != bn)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "BodyNode [" << node->getBodyNodePtr()->getName() << "] ("
                << node->getBodyNodePtr() << "). Please report this as a bug!"
                << std::endl;
          consistent = false;
          assert(false);
        }

        if (node->mIndexInBodyNode != k)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "index in its BodyNode (" << k << "|"
                << node->mIndexInBodyNode << "). Please report this as a bug!"
                << std::endl;
          consistent = false;
          assert(false);
        }

        // TODO(MXG): Consider checking Node names here
      }
    }
  }

  // Check DegreesOfFreedom indexing
  for (std::size_t i = 0; i < getNumDofs(); ++i)
  {
    const DegreeOfFreedom* dof = getDof(i);
    if (dof->getIndexInSkeleton() != i)
    {
      dterr << "[Skeleton::checkIndexingConsistency] DegreeOfFreedom named ["
            << dof->getName() << "] (" << dof << ") in Skeleton [" << getName()
            << "] (" << this << ") is mistaken about its index "
            << "in its Skeleton (" << i << "|" << dof->getIndexInSkeleton()
            << "). Please report this as a bug!" << std::endl;
      consistent = false;
      assert(false);
    }

    const DegreeOfFreedom* nameEntryForDof = getDof(dof->getName());
    if (nameEntryForDof != dof)
    {
      dterr << "[Skeleton::checkIndexingConsistency] Skeleton named ["
            << getName() << "] (" << this << ") is mistaken about the name of "
            << "DegreeOfFreedom [" << dof->getName() << "] (" << dof << "). "
            << "The name instead maps to [" << nameEntryForDof->getName()
            << "] (" << nameEntryForDof << "). Please report this as a bug!"
            << std::endl;
      consistent = false;
      assert(false);
    }
  }

  // Check each Node in the Skeleton-scope NodeMap
  {
    const Skeleton::NodeMap& nodeMap = mNodeMap;
    for (const auto& nodeType : nodeMap)
    {
      const std::vector<Node*>& nodes = nodeType.second;
      for (std::size_t k = 0; k < nodes.size(); ++k)
      {
        const Node* node = nodes[k];
        if (node->getSkeleton().get() != this)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "Skeleton [" << node->getSkeleton()->getName() << "] ("
                << node->getSkeleton() << "). Please report this as a bug!"
                << std::endl;
          consistent = false;
          assert(false);
        }

        if (node->mIndexInSkeleton != k)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "index in its Skeleton (" << k << "|"
                << node->mIndexInSkeleton << "). Please report this as a bug!"
                << std::endl;
          consistent = false;
          assert(false);
        }
      }
    }
  }

  // Check each BodyNode in each Tree cache
  for (std::size_t i = 0; i < mTreeCache.size(); ++i)
  {
    const DataCache& cache = mTreeCache[i];
    for (std::size_t j = 0; j < cache.mBodyNodes.size(); ++j)
    {
      const BodyNode* bn = cache.mBodyNodes[j];
      if (bn->mTreeIndex != i)
      {
        dterr << "[Skeleton::checkIndexingConsistency] BodyNode named ["
              << bn->getName() << "] in Skeleton [" << getName() << "] is "
              << "mistaken about its tree's index (" << i << "|"
              << bn->mTreeIndex << "). Please report this as a bug!"
              << std::endl;
        consistent = false;
        assert(false);
      }

      if (bn->mIndexInTree != j)
      {
        dterr << "[Skeleton::checkIndexingConsistency] BodyNode named ["
              << bn->getName() << "] (" << bn << ") in Skeleton [" << getName()
              << "] (" << this << ") is mistaken about its index "
              << "in the tree (" << j << "|" << bn->mIndexInTree << "). Please "
              << "report this as a bug!" << std::endl;
        consistent = false;
        assert(false);
      }
    }

    for (std::size_t j = 0; j < cache.mDofs.size(); ++j)
    {
      const DegreeOfFreedom* dof = cache.mDofs[j];
      if (dof->getTreeIndex() != i)
      {
        dterr << "[Skeleton::checkIndexingConsistency] DegreeOfFreedom named ["
              << dof->getName() << "] (" << dof << ") in Skeleton ["
              << getName() << "] (" << this << ") is mistaken about its tree's "
              << "index (" << i << "|" << dof->getTreeIndex() << "). Please "
              << "report this as a bug!" << std::endl;
        consistent = false;
        assert(false);
      }
    }
  }

  // Check that the Tree cache and the number of Tree NodeMaps match up
  if (mTreeCache.size() != mTreeNodeMaps.size())
  {
    consistent = false;
    dterr << "[Skeleton::checkIndexingConsistency] Skeleton named ["
          << getName() << "] (" << this << ") has inconsistent tree cache "
          << " and tree Node map sizes (" << mTreeCache.size() << "|"
          << mTreeNodeMaps.size() << "). Please report this as a bug!"
          << std::endl;
    assert(false);
  }

  // Check each Node in the NodeMap of each Tree
  for (std::size_t i = 0; i < mTreeNodeMaps.size(); ++i)
  {
    const NodeMap& nodeMap = mTreeNodeMaps[i];

    for (const auto& nodeType : nodeMap)
    {
      const std::vector<Node*>& nodes = nodeType.second;
      for (std::size_t k = 0; k < nodes.size(); ++k)
      {
        const Node* node = nodes[k];
        if (node->getBodyNodePtr()->mTreeIndex != i)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "Tree Index (" << i << "|"
                << node->getBodyNodePtr()->mTreeIndex << "). Please report "
                << "this as a bug!" << std::endl;
          consistent = false;
          assert(false);
        }

        if (node->mIndexInTree != k)
        {
          dterr << "[Skeleton::checkIndexingConsistency] Node named ["
                << node->getName() << "] (" << node << ") in Skeleton ["
                << getName() << "] (" << this << ") is mistaken about its "
                << "index in its tree (" << k << "|" << node->mIndexInTree
                << "). Please report this as a bug!" << std::endl;
          consistent = false;
          assert(false);
        }
      }
    }
  }

  return consistent;
}

//==============================================================================
/// This returns a square (N x N) matrix, filled with 1s and 0s. This can be
/// interpreted as:
///
/// getParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
/// getParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
///
/// This is computed in bulk, and cached in the skeleton.
const Eigen::MatrixXi& Skeleton::getParentMap()
{
  if (mSkelCache.mDirty.mParentMap)
  {
    mSkelCache.mParentMap = Eigen::MatrixXi::Zero(getNumDofs(), getNumDofs());
    for (int row = 0; row < getNumDofs(); row++)
    {
      /*
      dynamics::DegreeOfFreedom* rowDof = getDof(row);
      for (int col = 0; col < getNumDofs(); col++) {
        dynamics::DegreeOfFreedom* colDof = getDof(col);
        if (rowDof->isParentOf(colDof)) {
          mSkelCache.mParentMap(row, col) = 1;
        }
      }
      */
      dynamics::DegreeOfFreedom* dof = getDof(row);
      dynamics::Joint* joint = dof->getJoint();
      std::vector<dynamics::Joint*> visit;
      visit.push_back(joint);
      while (visit.size() > 0)
      {
        dynamics::Joint* cursor = visit.back();
        visit.pop_back();

        dynamics::BodyNode* cursorChildBodyNode = cursor->getChildBodyNode();
        for (int i = 0; i < cursorChildBodyNode->getNumChildJoints(); i++)
        {
          dynamics::Joint* childJoint = cursorChildBodyNode->getChildJoint(i);
          visit.push_back(childJoint);

          for (int j = 0; j < childJoint->getNumDofs(); j++)
          {
            mSkelCache.mParentMap(row, childJoint->getIndexInSkeleton(j)) = 1;
          }
        }
      }
    }
    mSkelCache.mDirty.mParentMap = false;
  }
  return mSkelCache.mParentMap;
}

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_SKEL_DEFINITIONS(Skeleton, Marker)

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_SKEL_DEFINITIONS(Skeleton, ShapeNode)

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_SKEL_DEFINITIONS(Skeleton, EndEffector)

//==============================================================================
void Skeleton::clearGradientConstraintMatrices()
{
  mSkelCache.mGradientConstraintMatrices = nullptr;
}

//==============================================================================
std::shared_ptr<neural::ConstrainedGroupGradientMatrices>
Skeleton::getGradientConstraintMatrices()
{
  return mSkelCache.mGradientConstraintMatrices;
}

//==============================================================================
void Skeleton::setGradientConstraintMatrices(
    std::shared_ptr<neural::ConstrainedGroupGradientMatrices> gradientMatrices)
{
  mSkelCache.mGradientConstraintMatrices = gradientMatrices;
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfC(neural::WithRespectTo* wrt)
{
  const int dofs = static_cast<int>(getNumDofs());
  Eigen::MatrixXs DCg_Dp = Eigen::MatrixXs::Zero(dofs, dofs);

  if (wrt == neural::WithRespectTo::FORCE)
  {
    return DCg_Dp;
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

#ifdef DART_DEBUG_ANALYTICAL_DERIV
    mDiffC.init(bodyNodes.size(), getNumDofs());
#endif

    for (BodyNode* bodyNode : bodyNodes)
    {
      bodyNode->computeJacobianOfCForward(wrt);
    }

    for (int i = bodyNodes.size() - 1; i >= 0; i--)
    {
      BodyNode* bodyNode = bodyNodes[i];
      bodyNode->computeJacobianOfCBackward(
          wrt, DCg_Dp, mAspectProperties.mGravity);
    }

#ifdef DART_DEBUG_ANALYTICAL_DERIV
    mDiffC.print();
#endif

    return DCg_Dp;
  }
  else
  {
    return finiteDifferenceJacobianOfC(wrt);
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfM(
    const Eigen::VectorXs& x, neural::WithRespectTo* wrt)
{
  const int dofs = static_cast<int>(getNumDofs());
  Eigen::MatrixXs DM_Dq = Eigen::MatrixXs::Zero(dofs, dofs);

  if (wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::FORCE)
  {
    return DM_Dq;
  }
  else if (wrt == neural::WithRespectTo::POSITION)
  {
    const auto old_ddq = getAccelerations();
    setAccelerations(x);

    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (BodyNode* bodyNode : bodyNodes)
    {
      bodyNode->computeJacobianOfMForward(wrt);
    }

    for (int i = bodyNodes.size() - 1; i >= 0; i--)
    {
      BodyNode* bodyNode = bodyNodes[i];
      bodyNode->computeJacobianOfMBackward(wrt, DM_Dq);
    }

    setAccelerations(old_ddq);

    return DM_Dq;
  }
  else
  {
    // other than pos/vel/force such as mass
    return finiteDifferenceJacobianOfM(x, wrt);
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfID(
    const Eigen::VectorXs& x, neural::WithRespectTo* wrt)
{
  const auto old_ddq = getAccelerations();
  setAccelerations(x);

  Eigen::MatrixXs DID_Dq = getJacobianOfM(x, wrt) + getJacobianOfC(wrt);

  setAccelerations(old_ddq);

  return DID_Dq;
}

#ifdef DART_DEBUG_ANALYTICAL_DERIV

//==============================================================================
void Skeleton::DiffMinv::Data::init()
{
  AI.setZero();
  AB.setZero();
  psi.setZero(0, 0);
}

//==============================================================================
void Skeleton::DiffMinv::init(size_t numBodies, size_t numDofs)
{
  nodes.resize(numBodies);
  for (auto& node : nodes)
  {
    node.data.init();
    node.derivs.resize(numDofs);
    for (auto& deriv : node.derivs)
    {
      deriv.init();
    }
  }

  nodes_numeric.resize(numBodies);
  for (auto& node : nodes_numeric)
  {
    node.data.init();
    node.derivs.resize(numDofs);
    for (auto& deriv : node.derivs)
    {
      deriv.init();
    }
  }
}

//==============================================================================
void Skeleton::DiffMinv::print()
{
  std::cout << "[Diff DMinv_Dq]\n\n";

  std::cout << "<<<< BACKWARD >>>>\n\n";

  if (nodes.empty())
  {
    return;
  }

  for (int i = static_cast<int>(nodes.size() - 1); i >= 0; --i)
  {
    const auto& node = nodes[static_cast<size_t>(i)];
    const auto& data = node.data;

    const auto& node_numeric = nodes_numeric[static_cast<size_t>(i)];
    // const auto& data_numeric = node_numeric.data;

    std::cout << "<<< i: " << i + 1 << ">>>\n\n";

    std::cout << "S[" << i + 1 << "]    : " << data.S.transpose() << "\n";
    //    std::cout << "AI[" << i+1 << "]:\n" << data.AI << "\n";
    std::cout << "AIS[" << i + 1 << "]  : " << (data.AI * data.S).transpose()
              << "\n";
    std::cout << "AB[" << i + 1 << "]   : " << data.AB.transpose() << "\n";
    std::cout << "psi[" << i + 1 << "]  : " << data.psi << "\n";
    //    std::cout << "Pi[" << i+1 << "]:\n" << data.Pi << "\n";
    std::cout << "alpha[" << i + 1 << "]: " << data.alpha.transpose() << "\n";
    std::cout << "beta[" << i + 1 << "] : " << data.beta.transpose() << "\n";

    std::cout << "\n";

    for (auto j = 0u; j < node.derivs.size(); ++j)
    {
      const auto& deriv = node.derivs[j];
      const auto& deriv_numeric = node_numeric.derivs[j];
      std::cout << "DAI[" << i + 1 << "," << j + 1 << "]   :\n"
                << deriv.AI << "\n";
      std::cout << "DAI_num[" << i + 1 << "," << j + 1 << "]   :\n"
                << deriv_numeric.AI << "\n";
      std::cout << "DAB[" << i + 1 << "," << j + 1
                << "]   : " << deriv.AB.transpose() << "\n";
      std::cout << "DAB_num[" << i + 1 << "," << j + 1
                << "]   : " << deriv_numeric.AB.transpose() << "\n";
      std::cout << "Dpsi[" << i + 1 << "," << j + 1 << "]  : " << deriv.psi
                << "\n";
      std::cout << "Dpsi_num[" << i + 1 << "," << j + 1
                << "]  : " << deriv_numeric.psi << "\n";
      std::cout << "DPi[" << i + 1 << "," << j + 1 << "]   :\n"
                << deriv.Pi << "\n";
      std::cout << "Dalpha[" << i + 1 << "," << j + 1
                << "]: " << deriv.alpha.transpose() << "\n";
      std::cout << "Dalpha_num[" << i + 1 << "," << j + 1
                << "]: " << deriv_numeric.alpha.transpose() << "\n";
      std::cout << "Dbeta[" << i + 1 << "," << j + 1
                << "] : " << deriv.beta.transpose() << "\n";
      std::cout << "Dbeta_num[" << i + 1 << "," << j + 1
                << "] : " << deriv_numeric.beta.transpose() << "\n";
      std::cout << "\n";
    }

    std::cout << "\n";
  }

  std::cout << "<<<< FORWARD >>>>\n\n";

  for (auto i = 0u; i < nodes.size(); ++i)
  {
    const auto& node = nodes[i];
    const auto& data = node.data;

    std::cout << "<<< i: " << i + 1 << ">>>\n\n";

    std::cout << "ddq[" << i + 1 << "]:" << data.ddq.transpose() << "\n";
    std::cout << "dV[" << i + 1 << "]:" << data.dV.transpose() << "\n";

    //    for (auto j = 0u; j < node.derivs.size(); ++j)
    //    {
    //      const auto& deriv = node.derivs[j];
    //    }
  }

  std::cout << std::endl << std::endl;
}

#endif

//==============================================================================
/// This gives the unconstrained Jacobian of M^{-1}f
Eigen::MatrixXs Skeleton::getJacobianOfMinv(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt)
{
  return getJacobianOfMinv_ID(f, wrt);
  // We no longer use the direct Jacobian, because it's both incorrect _and_
  // slower than doing the inverse-dynamics approach, so probably not worth
  // fixing.
  /*
  if (useID)
    return getJacobianOfMinv_ID(f, wrt);
  else
    return getJacobianOfMinv_Direct(f, wrt);
  */
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfMinv_ID(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt)
{
  if (getNumDofs() == 0)
  {
    return Eigen::MatrixXs::Zero(0, wrt->dim(this));
  }
  if (wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::FORCE)
  {
    const int dofs = static_cast<int>(getNumDofs());
    return Eigen::MatrixXs::Zero(dofs, dofs);
  }
  else if (wrt == neural::WithRespectTo::POSITION)
  {
    const Eigen::MatrixXs& Minv = getInvMassMatrix();
    const Eigen::MatrixXs& DMddq_Dq = getJacobianOfM(Minv * f, wrt);
    return -Minv * DMddq_Dq;
  }
  else
  {
    return finiteDifferenceJacobianOfMinv(f, wrt);
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfMinv_Direct(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt)
{
  // TODO: explore correcting and debugging this method.
  assert(
      false
      && "We should never be calling this method, it's not completely "
         "correct.");

  const int dofs = static_cast<int>(getNumDofs());
  Eigen::MatrixXs DMinvX_Dp = Eigen::MatrixXs::Zero(dofs, dofs);

  if (wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::FORCE)
  {
    return DMinvX_Dp;
  }
  else if (wrt == neural::WithRespectTo::POSITION)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

#ifdef DART_DEBUG_ANALYTICAL_DERIV
    mDiffMinv.init(bodyNodes.size(), getNumDofs());
#endif

    const Eigen::VectorXs oldForces = getControlForces();
    setControlForces(f);

    // Backward iteration
    for (auto it = bodyNodes.rbegin(); it != bodyNodes.rend(); ++it)
    {
      BodyNode* bodyNode = *it;
      bodyNode->computeJacobianOfMinvXInit();
      bodyNode->computeJacobianOfMinvXBackward();
    }

    // Forward iteration
    for (BodyNode* bodyNode : bodyNodes)
    {
      bodyNode->computeJacobianOfMinvXForward(DMinvX_Dp);
    }

#ifdef DART_DEBUG_ANALYTICAL_DERIV
    // Verification
    if (!bodyNodes.empty())
    {
      const s_t EPS = 1e-7;
      const size_t numDofs = getNumDofs();
      Eigen::VectorXs start = getPositions();
      for (size_t i = 0; i < numDofs; ++i)
      {
        Eigen::VectorXs tweaked = start;
        tweaked[static_cast<int>(i)] += EPS;
        setPositions(tweaked);

        for (int j = static_cast<int>(bodyNodes.size()) - 1; j >= 0; --j)
        {
          auto& node = mDiffMinv.nodes_numeric[j];

          BodyNode* bodyNode = bodyNodes[static_cast<size_t>(j)];
          Joint* joint = bodyNode->getParentJoint();
          const math::Jacobian S = joint->getRelativeJacobian();
          bodyNode->updateInvMassMatrix();
          const math::Inertia& AI = bodyNode->getArticulatedInertia();
          const Eigen::Vector6s& AB = bodyNode->mInvM_c;
          const Eigen::VectorXs& alpha = joint->getAlpha();
          const Eigen::Vector6s& beta = joint->computeBeta(AI, AB);
          Eigen::MatrixXs psi = (S.transpose() * AI * S).inverse();
          node.derivs[i].AI = AI;
          node.derivs[i].AB = AB;
          node.derivs[i].alpha = alpha;
          node.derivs[i].beta = beta;
          node.derivs[i].psi = psi;
        }

        tweaked = start;
        tweaked[static_cast<int>(i)] -= EPS;
        setPositions(tweaked);

        for (int j = static_cast<int>(bodyNodes.size()) - 1; j >= 0; --j)
        {
          auto& node = mDiffMinv.nodes_numeric[j];

          BodyNode* bodyNode = bodyNodes[static_cast<size_t>(j)];
          Joint* joint = bodyNode->getParentJoint();
          const math::Jacobian S = joint->getRelativeJacobian();
          bodyNode->updateInvMassMatrix();
          const math::Inertia& AI = bodyNode->getArticulatedInertia();
          const Eigen::Vector6s& AB = bodyNode->mInvM_c;
          const Eigen::VectorXs& alpha = joint->getAlpha();
          const Eigen::Vector6s& beta = joint->computeBeta(AI, AB);
          Eigen::MatrixXs psi = (S.transpose() * AI * S).inverse();
          node.derivs[i].AI = (node.derivs[i].AI - AI) / (2 * EPS);
          node.derivs[i].AB = (node.derivs[i].AB - AB) / (2 * EPS);
          node.derivs[i].alpha = (node.derivs[i].alpha - alpha) / (2 * EPS);
          node.derivs[i].beta = (node.derivs[i].beta - beta) / (2 * EPS);
          node.derivs[i].psi = (node.derivs[i].psi - psi) / (2 * EPS);
        }
      }

      setPositions(start);
    }
    //  mDiffMinv.print();
#endif

    setControlForces(oldForces);

    return DMinvX_Dp;
  }
  else
  {
    return finiteDifferenceJacobianOfMinv(f, wrt);
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfDampSpring(neural::WithRespectTo* wrt)
{
  s_t dt = getTimeStep();
  size_t nDofs = getNumDofs();
  Eigen::MatrixXs damp_coeff = getDampingCoeffVector().asDiagonal();
  Eigen::MatrixXs spring_stiff = getSpringStiffVector().asDiagonal();
  if (wrt == neural::WithRespectTo::VELOCITY)
  {
    Eigen::MatrixXs jacobian = damp_coeff + dt * spring_stiff;
    return jacobian;
  }
  else if (wrt == neural::WithRespectTo::POSITION)
  {
    Eigen::MatrixXs jacobian = spring_stiff;
    return jacobian;
  }
  else
  {
    Eigen::MatrixXs jacobian = Eigen::MatrixXs::Zero(nDofs, nDofs);
    return jacobian;
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getJacobianOfFD(neural::WithRespectTo* wrt)
{
  const auto& tau = getControlForces();
  const auto& Cg = getCoriolisAndGravityForces();
  const auto& Minv = getInvMassMatrix();
  const auto& spring_force = getSpringForce();
  const auto& damping_force = getDampingForce();

  const auto& DMinv_Dp
      = getJacobianOfMinv(tau - Cg - damping_force - spring_force, wrt);
  const auto& DC_Dp = getJacobianOfC(wrt);
  const auto& D_damp_spring = getJacobianOfDampSpring(wrt);

  return DMinv_Dp - Minv * DC_Dp - Minv * D_damp_spring;
}

//==============================================================================
Eigen::MatrixXs Skeleton::getUnconstrainedVelJacobianWrt(
    s_t dt, neural::WithRespectTo* wrt)
{
  Eigen::VectorXs tau = getControlForces();
  Eigen::VectorXs C = getCoriolisAndGravityForces() - getExternalForces();

  Eigen::MatrixXs Minv = getInvMassMatrix();
  Eigen::MatrixXs dC = getJacobianOfC(wrt);

  if (wrt == neural::WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dM = getJacobianOfMinv(
        dt * (tau - C - getDampingForce() - getSpringForce()), wrt);
    return dM - Minv * dt * dC;
  }
  else
  {
    return -Minv * dt * dC;
  }
}

//==============================================================================
Eigen::MatrixXs Skeleton::getVelCJacobian()
{
  // TOOD(keenon): replace with the GEAR approach
  // return finiteDifferenceVelCJacobian();
  return getJacobianOfC(neural::WithRespectTo::VELOCITY);
}

#ifdef DART_DEBUG_ANALYTICAL_DERIV
//==============================================================================
void Skeleton::DiffC::Data::init()
{
  // Do nothing
}

//==============================================================================
void Skeleton::DiffC::init(size_t numBodies, size_t numDofs)
{
  nodes.resize(numBodies);
  for (auto& node : nodes)
  {
    node.data.init();
    node.derivs.resize(numDofs);
    for (auto& deriv : node.derivs)
    {
      deriv.init();
    }
  }

  nodes_numeric.resize(numBodies);
  for (auto& node : nodes_numeric)
  {
    node.data.init();
    node.derivs.resize(numDofs);
    for (auto& deriv : node.derivs)
    {
      deriv.init();
    }
  }
}

//==============================================================================
void Skeleton::DiffC::print()
{
  std::cout << "[Diff DC_Dq]\n\n";

  std::cout << "<<<< FORWARD >>>>\n\n";

  if (nodes.empty())
  {
    return;
  }

  for (auto i = 0u; i < nodes.size(); ++i)
  {
    const auto& node = nodes[static_cast<size_t>(i)];
    const auto& data = node.data;

    const auto& node_numeric = nodes_numeric[static_cast<size_t>(i)];
    const auto& data_numeric = node_numeric.data;

    std::cout << "<<< i: " << i + 1 << ">>>\n\n";

    std::cout << "S[" << i + 1 << "]    : " << data.S.transpose() << "\n";
    std::cout << "dV[" << i + 1 << "] : " << data.dV.transpose() << "\n";
    std::cout << "F[" << i + 1 << "] : " << data.F.transpose() << "\n";
    std::cout << "tau[" << i + 1 << "] : " << data.tau.transpose() << "\n";

    std::cout << "\n";

    for (auto j = 0u; j < node.derivs.size(); ++j)
    {
      const auto& deriv = node.derivs[j];
      const auto& deriv_numeric = node_numeric.derivs[j];
      std::cout << "DdV[" << i + 1 << "," << j + 1
                << "] : " << deriv.dV.transpose() << "\n";
      std::cout << "DdV_num[" << i + 1 << "," << j + 1
                << "] : " << deriv_numeric.dV.transpose() << "\n";
      std::cout << "\n";
    }

    std::cout << "\n";
  }

  std::cout << "<<<< BACKWARD >>>>\n\n";

  for (int i = static_cast<int>(nodes.size() - 1); i >= 0; --i)
  {
    const auto& node = nodes[i];
    const auto& data = node.data;

    std::cout << "<<< i: " << i + 1 << ">>>\n\n";

    std::cout << "F[" << i + 1 << "]:" << data.F.transpose() << "\n";

    //    for (auto j = 0u; j < node.derivs.size(); ++j)
    //    {
    //      const auto& deriv = node.derivs[j];
    //    }
  }

  std::cout << std::endl << std::endl;
}
#endif

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfM(
    const Eigen::VectorXs& x, neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfM(x, wrt);

  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  // Get baseline C(pos, vel)
  Eigen::VectorXs baseline = getMassMatrix() * x;

  s_t EPS = 5e-7;

  for (std::size_t i = 0; i < m; i++)
  {
    Eigen::VectorXs tweaked = start;
    tweaked(i) += EPS;
    wrt->set(this, tweaked);
    mSkelCache.mDirty.mMassMatrix = true;
    Eigen::VectorXs plus = getMassMatrix() * x;
    tweaked = start;
    tweaked(i) -= EPS;
    wrt->set(this, tweaked);
    mSkelCache.mDirty.mMassMatrix = true;
    Eigen::VectorXs minus = getMassMatrix() * x;

    J.col(i) = (plus - minus) / (2 * EPS);
  }

  // Reset everything how we left it
  wrt->set(this, start);
  mSkelCache.mDirty.mMassMatrix = true;
  getMassMatrix();

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersJacobianOfM(
    const Eigen::VectorXs& x, neural::WithRespectTo* wrt)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < m; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(this, perturbedPlus);
    mSkelCache.mDirty.mMassMatrix = true;
    Eigen::MatrixXs plus = getMassMatrix() * x;

    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(this, perturbedMinus);
    mSkelCache.mDirty.mMassMatrix = true;
    Eigen::MatrixXs minus = getMassMatrix() * x;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(this, perturbedPlus);
      mSkelCache.mDirty.mMassMatrix = true;
      plus = getMassMatrix() * x;

      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(this, perturbedMinus);
      mSkelCache.mDirty.mMassMatrix = true;
      minus = getMassMatrix() * x;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }
  wrt->set(this, originalWrt);
  mSkelCache.mDirty.mMassMatrix = true;
  getMassMatrix();

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfC(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfC(wrt);

  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  // Get baseline C(pos, vel)
  Eigen::VectorXs baseline
      = getCoriolisAndGravityForces() - getExternalForces();

  s_t EPS = 1e-7;

  for (std::size_t i = 0; i < m; i++)
  {
    Eigen::VectorXs tweaked = start;
    tweaked(i) += EPS;
    wrt->set(this, tweaked);
    Eigen::VectorXs perturbedPos
        = getCoriolisAndGravityForces() - getExternalForces();
    tweaked = start;
    tweaked(i) -= EPS;
    wrt->set(this, tweaked);
    Eigen::VectorXs perturbedNeg
        = getCoriolisAndGravityForces() - getExternalForces();

    J.col(i) = (perturbedPos - perturbedNeg) / (2 * EPS);
  }

  // Reset everything how we left it
  wrt->set(this, start);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfID(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfID(f, wrt);

  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  const Eigen::VectorXs old_ddq = getAccelerations();
  setAccelerations(f);

  s_t EPS = 5e-7;

  for (std::size_t i = 0; i < m; i++)
  {
    Eigen::VectorXs tweaked = start;
    tweaked(i) += EPS;
    wrt->set(this, tweaked);
    computeInverseDynamics();
    const Eigen::VectorXs plus = getControlForces();
    tweaked = start;
    tweaked(i) -= EPS;
    wrt->set(this, tweaked);
    computeInverseDynamics();
    const Eigen::VectorXs minus = getControlForces();

    J.col(i) = (plus - minus) / (2 * EPS);
  }

  // Reset everything how we left it
  wrt->set(this, start);

  setAccelerations(old_ddq);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersJacobianOfID(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  const Eigen::VectorXs old_ddq = getAccelerations();
  setAccelerations(f);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < m; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(this, perturbedPlus);
    computeInverseDynamics();
    Eigen::VectorXs plus = getControlForces();
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(this, perturbedMinus);
    computeInverseDynamics();
    Eigen::VectorXs minus = getControlForces();

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(this, perturbedPlus);
      computeInverseDynamics();
      plus = getControlForces();
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(this, perturbedMinus);
      computeInverseDynamics();
      minus = getControlForces();

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }
  wrt->set(this, originalWrt);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersJacobianOfC(
    neural::WithRespectTo* wrt)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < m; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(this, perturbedPlus);
    Eigen::MatrixXs tauPlus
        = getCoriolisAndGravityForces() - getExternalForces();
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(this, perturbedMinus);
    Eigen::MatrixXs tauMinus
        = getCoriolisAndGravityForces() - getExternalForces();

    tab[0][0] = (tauPlus - tauMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(this, perturbedPlus);
      tauPlus = getCoriolisAndGravityForces() - getExternalForces();
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(this, perturbedMinus);
      tauMinus = getCoriolisAndGravityForces() - getExternalForces();

      tab[0][iTab] = (tauPlus - tauMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }
  wrt->set(this, originalWrt);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfMinv(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMinv(f, wrt);

  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  // Get baseline C(pos, vel)
  Eigen::VectorXs baseline = multiplyByImplicitInvMassMatrix(f);

  s_t EPS = 5e-7;

  for (std::size_t i = 0; i < m; i++)
  {
    Eigen::VectorXs tweaked = start;
    tweaked(i) += EPS;
    wrt->set(this, tweaked);
    Eigen::VectorXs plus = multiplyByImplicitInvMassMatrix(f);
    tweaked = start;
    tweaked(i) -= EPS;
    wrt->set(this, tweaked);
    Eigen::VectorXs minus = multiplyByImplicitInvMassMatrix(f);

    J.col(i) = (plus - minus) / (2 * EPS);
  }

  // Reset everything how we left it
  wrt->set(this, start);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersJacobianOfMinv(
    Eigen::VectorXs f, neural::WithRespectTo* wrt)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < m; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(this, perturbedPlus);
    Eigen::MatrixXs MinvFPlus = multiplyByImplicitInvMassMatrix(f);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(this, perturbedMinus);
    Eigen::MatrixXs MinvFMinus = multiplyByImplicitInvMassMatrix(f);

    tab[0][0] = (MinvFPlus - MinvFMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(this, perturbedPlus);
      MinvFPlus = multiplyByImplicitInvMassMatrix(f);
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(this, perturbedMinus);
      MinvFMinus = multiplyByImplicitInvMassMatrix(f);

      tab[0][iTab] = (MinvFPlus - MinvFMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }
  wrt->set(this, originalWrt);

  return J;
}

//==============================================================================
Eigen::VectorXs Skeleton::getDynamicsForces()
{
  computeForwardDynamics();
  std::size_t n = getNumDofs();
  Eigen::VectorXs forces = Eigen::VectorXs(n);
  int cursor = 0;
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    Eigen::VectorXs jointForces = mSkelCache.mBodyNodes[i]
                                      ->getParentJoint()
                                      ->getRelativeJacobian()
                                      .transpose()
                                  * mSkelCache.mBodyNodes[i]->getBodyForce();
    forces.segment(cursor, jointForces.size()) = jointForces;
    cursor += jointForces.size();
  }
  return forces;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceVelCJacobian(bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersVelCJacobian();

  std::size_t n = getNumDofs();
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, n);
  Eigen::VectorXs vel = getVelocities();

  // Get baseline C(pos, vel)
  Eigen::VectorXs baseline = getCoriolisAndGravityForces();

  s_t EPS = 1e-6;

  for (std::size_t i = 0; i < n; i++)
  {
    Eigen::VectorXs tweakedVel = vel;
    tweakedVel(i) += EPS;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedPos = getCoriolisAndGravityForces();
    tweakedVel = vel;
    tweakedVel(i) -= EPS;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedNeg = getCoriolisAndGravityForces();

#ifndef NDEBUG
    if (perturbedPos == perturbedNeg && perturbedPos != baseline)
    {
      // std::cout << "Got a mysteriously broken coriolis force result" <<
      // std::endl;

      // Set positive vel change

      tweakedVel = vel;
      tweakedVel(i) += EPS;
      setVelocities(tweakedVel);

      DataCache& cache = mTreeCache[0];
      std::size_t dof = cache.mDofs.size();
      Eigen::VectorXs mCg = Eigen::VectorXs::Zero(dof);
      assert(static_cast<std::size_t>(mCg.size()) == dof);

      mCg.setZero();

      std::vector<Eigen::Vector6s> posVecs;

      for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
           it != cache.mBodyNodes.end();
           ++it)
      {
        (*it)->updateCombinedVector();
      }

      for (std::vector<BodyNode*>::const_reverse_iterator it
           = cache.mBodyNodes.rbegin();
           it != cache.mBodyNodes.rend();
           ++it)
      {
        Eigen::Vector6s V = (*it)->getSpatialVelocity();
        const Eigen::Matrix6s& mI
            = (*it)->mAspectProperties.mInertia.getSpatialTensor();
        posVecs.push_back(math::dad(V, mI * V));
        (*it)->aggregateCombinedVector(mCg, mAspectProperties.mGravity);
      }

      // Set negative vel change

      tweakedVel = vel;
      tweakedVel(i) -= EPS;
      setVelocities(tweakedVel);

      Eigen::VectorXs mCg2 = Eigen::VectorXs::Zero(dof);
      assert(static_cast<std::size_t>(mCg2.size()) == dof);

      mCg2.setZero();

      std::vector<Eigen::Vector6s> negVecs;

      for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
           it != cache.mBodyNodes.end();
           ++it)
      {
        (*it)->updateCombinedVector();
      }

      for (std::vector<BodyNode*>::const_reverse_iterator it
           = cache.mBodyNodes.rbegin();
           it != cache.mBodyNodes.rend();
           ++it)
      {
        Eigen::Vector6s V = (*it)->getSpatialVelocity();
        const Eigen::Matrix6s& mI
            = (*it)->mAspectProperties.mInertia.getSpatialTensor();
        negVecs.push_back(math::dad(V, mI * V));
        (*it)->aggregateCombinedVector(mCg2, mAspectProperties.mGravity);
      }

      // TODO: negVecs and posVecs are the same, cause math::dad() is *=-1
      // idempotent in V std::cout << "mCg pos: " << std::endl << mCg <<
      // std::endl; std::cout << "mCg neg: " << std::endl << mCg2 << std::endl;
    }
#endif

    J.col(i) = (perturbedPos - perturbedNeg) / (2 * EPS);
  }

  // Reset everything how we left it
  setVelocities(vel);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersVelCJacobian()
{
  std::size_t n = getNumDofs();
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, n);
  Eigen::VectorXs vel = getVelocities();

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < n; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs tweakedVel = vel;
    tweakedVel(i) += stepSize;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedPos = getCoriolisAndGravityForces();
    tweakedVel = vel;
    tweakedVel(i) -= stepSize;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedNeg = getCoriolisAndGravityForces();

    tab[0][0] = (perturbedPos - perturbedNeg) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      tweakedVel = vel;
      tweakedVel(i) += stepSize;
      setVelocities(tweakedVel);
      perturbedPos = getCoriolisAndGravityForces();
      tweakedVel = vel;
      tweakedVel(i) -= stepSize;
      setVelocities(tweakedVel);
      perturbedNeg = getCoriolisAndGravityForces();

      tab[0][iTab] = (perturbedPos - perturbedNeg) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i) = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  // Reset everything how we left it
  setVelocities(vel);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfFD(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfFD(wrt);

  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  s_t EPS = 5e-7;

  for (std::size_t i = 0; i < m; i++)
  {
    Eigen::VectorXs tweaked = start;
    tweaked(i) += EPS;
    wrt->set(this, tweaked);
    computeForwardDynamics();
    Eigen::VectorXs plus = getAccelerations();
    tweaked = start;
    tweaked(i) -= EPS;
    wrt->set(this, tweaked);
    computeForwardDynamics();
    Eigen::VectorXs minus = getAccelerations();

    J.col(i) = (plus - minus) / (2 * EPS);
  }

  // Reset everything how we left it
  wrt->set(this, start);

  return J;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceRiddersJacobianOfFD(
    neural::WithRespectTo* wrt)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(n, m);
  Eigen::VectorXs start = wrt->get(this);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < n; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs tweaked = start;
    tweaked(i) += stepSize;
    wrt->set(this, tweaked);
    computeForwardDynamics();
    Eigen::VectorXs plus = getAccelerations();
    tweaked = start;
    tweaked(i) -= stepSize;
    wrt->set(this, tweaked);
    computeForwardDynamics();
    Eigen::VectorXs minus = getAccelerations();

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      tweaked = start;
      tweaked(i) += stepSize;
      wrt->set(this, tweaked);
      computeForwardDynamics();
      plus = getAccelerations();
      tweaked = start;
      tweaked(i) -= stepSize;
      wrt->set(this, tweaked);
      computeForwardDynamics();
      minus = getAccelerations();

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i) = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  // Reset everything how we left it
  wrt->set(this, start);
  computeForwardDynamics();

  return J;
}

//==============================================================================
Eigen::VectorXs Skeleton::getControlForceUpperLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getControlForceUpperLimit();
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs Skeleton::getControlForceLowerLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getControlForceLowerLimit();
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs Skeleton::getPositionUpperLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getPositionUpperLimit();
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs Skeleton::getPositionLowerLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getPositionLowerLimit();
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs Skeleton::getVelocityUpperLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getVelocityUpperLimit();
  }
  return limits;
}

//==============================================================================
Eigen::VectorXs Skeleton::getVelocityLowerLimits()
{
  std::size_t n = getNumDofs();
  Eigen::VectorXs limits(n);
  for (std::size_t i = 0; i < n; i++)
  {
    auto dof = getDof(i);
    limits[i] = dof->getVelocityLowerLimit();
  }
  return limits;
}

//==============================================================================
std::size_t Skeleton::getLinkCOMDims()
{
  return 3 * getNumBodyNodes();
}

//==============================================================================
std::size_t Skeleton::getLinkMOIDims()
{
  return 6 * getNumBodyNodes();
}

//==============================================================================
std::size_t Skeleton::getLinkMassesDims()
{
  return getNumBodyNodes();
}

//==============================================================================
Eigen::VectorXs Skeleton::getLinkCOMs()
{
  Eigen::VectorXs inertias = Eigen::VectorXs::Zero(getLinkCOMDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    inertias(cursor++) = inertia.COM_X;
    inertias(cursor++) = inertia.COM_Y;
    inertias(cursor++) = inertia.COM_Z;
  }
  return inertias;
}

//==============================================================================
Eigen::VectorXs Skeleton::getLinkMOIs()
{
  Eigen::VectorXs inertias = Eigen::VectorXs::Zero(getLinkMOIDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    inertias(cursor++) = inertia.I_XX;
    inertias(cursor++) = inertia.I_YY;
    inertias(cursor++) = inertia.I_ZZ;
    inertias(cursor++) = inertia.I_XY;
    inertias(cursor++) = inertia.I_XZ;
    inertias(cursor++) = inertia.I_YZ;
  }
  return inertias;
}

//==============================================================================
Eigen::VectorXs Skeleton::getLinkMasses()
{
  Eigen::VectorXs masses = Eigen::VectorXs::Zero(getLinkMassesDims());
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    masses(i) = getBodyNode(i)->getMass();
  }
  return masses;
}

//==============================================================================
void Skeleton::setControlForceUpperLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setControlForceUpperLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setControlForceLowerLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setControlForceLowerLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setPositionUpperLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setPositionUpperLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setPositionLowerLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setPositionLowerLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setVelocityUpperLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setVelocityUpperLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setVelocityLowerLimits(Eigen::VectorXs limits)
{
  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    getDof(i)->setVelocityLowerLimit(limits[i]);
  }
}

//==============================================================================
void Skeleton::setLinkCOMs(Eigen::VectorXs coms)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    s_t COM_X = coms(cursor++);
    s_t COM_Y = coms(cursor++);
    s_t COM_Z = coms(cursor++);
    Inertia newInertia(
        inertia.MASS,
        COM_X,
        COM_Y,
        COM_Z,
        inertia.I_XX,
        inertia.I_YY,
        inertia.I_ZZ,
        inertia.I_XY,
        inertia.I_XZ,
        inertia.I_YZ);
    getBodyNode(i)->setInertia(newInertia);
  }
}

//==============================================================================
void Skeleton::setLinkMOIs(Eigen::VectorXs mois)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    s_t I_XX = mois(cursor++);
    s_t I_YY = mois(cursor++);
    s_t I_ZZ = mois(cursor++);
    s_t I_XY = mois(cursor++);
    s_t I_XZ = mois(cursor++);
    s_t I_YZ = mois(cursor++);
    Inertia newInertia(
        inertia.MASS,
        inertia.COM_X,
        inertia.COM_Y,
        inertia.COM_Z,
        I_XX,
        I_YY,
        I_ZZ,
        I_XY,
        I_XZ,
        I_YZ);
    getBodyNode(i)->setInertia(newInertia);
  }
}

//==============================================================================
void Skeleton::setLinkMasses(Eigen::VectorXs masses)
{
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    getBodyNode(i)->setMass(masses(i));
  }
}

//==============================================================================
// This returns a vector of all the link scales for all the links in the
// skeleton concatenated into a flat vector
Eigen::VectorXs Skeleton::getLinkScales()
{
  Eigen::VectorXs scales = Eigen::VectorXs::Zero(getNumBodyNodes());
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    scales(i) = getBodyNode(i)->getScale();
  }
  return scales;
}

//==============================================================================
// Sets all the link scales for the skeleton, from a flat vector
void Skeleton::setLinkScales(Eigen::VectorXs scales)
{
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    getBodyNode(i)->setScale(scales(i));
  }
}

//==============================================================================
// This sets all the positions of the joints to within their limit range, if
// they're currently outside it.
void Skeleton::clampPositionsToLimits()
{
  for (int i = 0; i < getNumDofs(); i++)
  {
    auto* dof = getDof(i);
    if (dof->getPosition() > dof->getPositionUpperLimit())
    {
      dof->setPosition(dof->getPositionUpperLimit());
    }
    if (dof->getPosition() < dof->getPositionLowerLimit())
    {
      dof->setPosition(dof->getPositionLowerLimit());
    }
  }

  for (int i = 0; i < getNumJoints(); i++)
  {
    // Wrap all the exponential coordinates around to a minimum set
    if (getJoint(i)->getType() == FreeJoint::getStaticType())
    {
      Eigen::Vector6s pos = getJoint(i)->getPositions();
      pos.head<3>() = math::logMap(math::expMapRot(pos.head<3>()));
      getJoint(i)->setPositions(pos);
    }
    if (getJoint(i)->getType() == BallJoint::getStaticType())
    {
      Eigen::Vector3s pos = getJoint(i)->getPositions();
      pos = math::logMap(math::expMapRot(pos));
      getJoint(i)->setPositions(pos);
    }
  }
}

//==============================================================================
const std::vector<std::vector<dynamics::BodyNode*>>&
Skeleton::getBodyScaleGroups() const
{
  return mBodyScaleGroups;
}

//==============================================================================
const std::vector<dynamics::BodyNode*>& Skeleton::getBodyScaleGroup(
    int index) const
{
  return mBodyScaleGroups[index];
}

//==============================================================================
/// This creates scale groups for any body nodes that may've been added since
/// we last interacted with the body scale group APIs
void Skeleton::ensureBodyScaleGroups()
{
  if (mBodyScaleGroups.size() == 0)
  {
    // Add scale groups, one per body
    mBodyScaleGroups.reserve(getNumBodyNodes());
    for (int i = 0; i < getNumBodyNodes(); i++)
    {
      std::vector<dynamics::BodyNode*> singleGroup;
      singleGroup.push_back(getBodyNode(i));
      mBodyScaleGroups.push_back(singleGroup);
    }
  }
}

//==============================================================================
/// This returns the index of the group that this body node corresponds to
int Skeleton::getScaleGroupIndex(dynamics::BodyNode* bodyNode)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (int j = 0; j < mBodyScaleGroups[i].size(); j++)
    {
      if (mBodyScaleGroups[i][j]->getName() == bodyNode->getName())
        return i;
    }
  }
  return -1;
}

//==============================================================================
/// This takes two scale groups and merges their contents into a single group.
/// After this operation, there is one fewer scale group.
void Skeleton::mergeScaleGroups(dynamics::BodyNode* a, dynamics::BodyNode* b)
{
  mergeScaleGroupsByIndex(getScaleGroupIndex(a), getScaleGroupIndex(b));
}

//==============================================================================
/// This gets the scale upper bound for the first body in a group, by index
s_t Skeleton::getScaleGroupUpperBound(int groupIndex)
{
  return mBodyScaleGroups[groupIndex][0]->getScaleUpperBound();
}

//==============================================================================
/// This gets the scale lower bound for the first body in a group, by index
s_t Skeleton::getScaleGroupLowerBound(int groupIndex)
{
  return mBodyScaleGroups[groupIndex][0]->getScaleLowerBound();
}

//==============================================================================
/// This takes two scale groups and merges their contents into a single group.
/// After this operation, there is one fewer scale group.
void Skeleton::mergeScaleGroupsByIndex(int a, int b)
{
  // This is a no-op if both groups are already the same
  if (a == b)
    return;
  assert(a != -1);
  assert(b != -1);

  ensureBodyScaleGroups();
  std::vector<dynamics::BodyNode*>& groupA = mBodyScaleGroups[a];
  std::vector<dynamics::BodyNode*>& groupB = mBodyScaleGroups[b];
  // Remove the element further back in the array
  if (a > b)
  {
    // Transfer all elems from A to B
    for (dynamics::BodyNode*& node : groupA)
      groupB.push_back(node);
    // Then erase A
    mBodyScaleGroups.erase(mBodyScaleGroups.begin() + a);
  }
  else
  {
    // Transfer all elems from B to A
    for (dynamics::BodyNode*& node : groupB)
      groupA.push_back(node);
    // Then erase B
    mBodyScaleGroups.erase(mBodyScaleGroups.begin() + b);
  }
}

//==============================================================================
/// This returns the number of scaling groups (groups with an equal-scale
/// constraint) that there are in the model.
int Skeleton::getNumScaleGroups()
{
  ensureBodyScaleGroups();
  return mBodyScaleGroups.size();
}

//==============================================================================
/// This sets the scales of all the body nodes according to their group
/// membership. The `scale` vector is expected to be the same size as the
/// number of groups.
void Skeleton::setGroupScales(Eigen::VectorXs scale)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < scale.size(); i++)
  {
    for (dynamics::BodyNode* node : mBodyScaleGroups[i])
    {
      node->setScale(scale(i));
    }
  }
}

//==============================================================================
/// This gets the scales of the first body in each scale group.
Eigen::VectorXs Skeleton::getGroupScales()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups());
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].size() > 0);
    groups(i) = mBodyScaleGroups[i][0]->getScale();
  }
  return groups;
}

//==============================================================================
/// This returns the Jacobian of the joint positions wrt the scales of the
/// groups
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtGroupScales(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs individualBodiesJac
      = getJointWorldPositionsJacobianWrtBodyScales(joints);
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(individualBodiesJac.rows(), getNumScaleGroups());
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (dynamics::BodyNode* node : mBodyScaleGroups[i])
    {
      J.col(i) += individualBodiesJac.col(node->getIndexInSkeleton());
    }
  }
  return J;
}

//==============================================================================
/// This returns the Jacobian of the joint positions wrt the scales of the
/// groups
Eigen::MatrixXs
Skeleton::finiteDifferenceJointWorldPositionsJacobianWrtGroupScales(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(joints.size() * 3, getNumScaleGroups());

  Eigen::VectorXs original = getGroupScales();

  const double EPS = 1e-7;
  for (int i = 0; i < getNumScaleGroups(); i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    setGroupScales(perturbed);
    Eigen::VectorXs plus = getJointWorldPositions(joints);

    perturbed = original;
    perturbed(i) -= EPS;
    setGroupScales(perturbed);
    Eigen::VectorXs minus = getJointWorldPositions(joints);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  setGroupScales(original);

  return jac;
}

//==============================================================================
// This creates a fresh skeleton, which is a copy of this one EXCEPT that
// EulerJoints are BallJoints, and EulerFreeJoints are FreeJoints. This means
// the configuration spaces are different, so you need to use
// `convertPositionsToBallSpace()` and `convertPositionsFromBallSpace()` to
// transform positions to and from the new skeleton's configuration.
std::shared_ptr<dynamics::Skeleton> Skeleton::convertSkeletonToBallJoints()
{
  std::shared_ptr<dynamics::Skeleton> copy = dynamics::Skeleton::create();

  for (int i = 0; i < getNumJoints(); i++)
  {
    dynamics::Joint* joint = getJoint(i);
    dynamics::BodyNode* body = joint->getChildBodyNode();
    dynamics::BodyNode* parent
        = joint->getParentBodyNode() == nullptr
              ? nullptr
              : copy->getBodyNode(joint->getParentBodyNode()->getName());

    if (joint->getType() == EulerFreeJoint::getStaticType())
    {
      dynamics::FreeJoint::Properties props;
      props.mName = joint->getName();
      auto pair = cloneBodyNodeTree<dynamics::FreeJoint>(
          body, copy, parent, props, false);
      pair.first->copyTransformsFrom(joint);
    }
    else if (joint->getType() == EulerJoint::getStaticType())
    {
      dynamics::BallJoint::Properties props;
      props.mName = joint->getName();
      auto pair = cloneBodyNodeTree<dynamics::BallJoint>(
          body, copy, parent, props, false);
      pair.first->copyTransformsFrom(joint);
    }
    else
    {
      // Copy nodes to the new tree
      auto pair = cloneBodyNodeTree(nullptr, body, copy, parent, false);
      (void)pair;
      assert(
          pair.first->getParentScale()
          == body->getParentJoint()->getParentScale());
      assert(
          pair.first->getChildScale()
          == body->getParentJoint()->getChildScale());
    }
  }

  // Verify the memory structure of the copy is correct

#ifndef NDEBUG
  assert(copy->getNumBodyNodes() == getNumBodyNodes());
  for (int i = 0; i < copy->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* body = copy->getBodyNode(i);
    assert(body != getBodyNode(i));
    if (i > 0)
    {
      dynamics::Joint* parentJoint = body->getParentJoint();
      assert(parentJoint != nullptr);
    }
  }
  assert(copy->getNumJoints() == getNumJoints());
  for (int i = 0; i < copy->getNumJoints(); i++)
  {
    dynamics::Joint* joint = copy->getJoint(i);
    assert(joint != nullptr);
    assert(joint != getJoint(i));
    assert(joint->getName() == getJoint(i)->getName());
    assert(
        joint->getTransformFromChildBodyNode().matrix()
        == getJoint(i)->getTransformFromChildBodyNode().matrix());
    assert(
        joint->getTransformFromParentBodyNode().matrix()
        == getJoint(i)->getTransformFromParentBodyNode().matrix());
  }
#endif

  return copy;
}

//==============================================================================
// This converts the position vector from Euler space to Ball space for any
// joints that need to be converted. This needs to be called on a skeleton
// with EulerJoints and/or EulerFreeJoints or it will just return the passed
// in vector unchanged.
Eigen::VectorXs Skeleton::convertPositionsToBallSpace(Eigen::VectorXs pos)
{
  Eigen::VectorXs translated = pos;
  int cursor = 0;
  for (int i = 0; i < getNumJoints(); i++)
  {
    dynamics::Joint* joint = getJoint(i);
    if (joint->getType() == EulerFreeJoint::getStaticType())
    {
      dynamics::EulerFreeJoint* eulerFreeJoint
          = static_cast<dynamics::EulerFreeJoint*>(joint);
      Eigen::Isometry3s T = EulerJoint::convertToTransform(
          translated.segment<3>(cursor),
          eulerFreeJoint->getAxisOrder(),
          eulerFreeJoint->getFlipAxisMap());
      translated.segment<3>(cursor) = math::logMap(T.linear());
    }
    else if (joint->getType() == EulerJoint::getStaticType())
    {
      dynamics::EulerJoint* eulerJoint
          = static_cast<dynamics::EulerJoint*>(joint);
      Eigen::Isometry3s T = EulerJoint::convertToTransform(
          translated.segment<3>(cursor),
          eulerJoint->getAxisOrder(),
          eulerJoint->getFlipAxisMap());
      translated.segment<3>(cursor) = math::logMap(T.linear());
    }
    cursor += joint->getNumDofs();
  }
  assert(cursor == translated.size());
  return translated;
}

//==============================================================================
// This converts the position vector from Ball space to Euler space for any
// joints that need to be converted. This needs to be called on a skeleton
// with EulerJoints and/or EulerFreeJoints or it will just return the passed
// in vector unchanged.
Eigen::VectorXs Skeleton::convertPositionsFromBallSpace(Eigen::VectorXs pos)
{
  Eigen::VectorXs translated = pos;
  int cursor = 0;
  for (int i = 0; i < getNumJoints(); i++)
  {
    dynamics::Joint* joint = getJoint(i);
    if (joint->getType() == EulerFreeJoint::getStaticType())
    {
      dynamics::EulerFreeJoint* eulerFreeJoint
          = static_cast<dynamics::EulerFreeJoint*>(joint);
      Eigen::Matrix3s R = math::expMapRot(translated.segment<3>(cursor));
      if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::XYZ)
      {
        translated.segment<3>(cursor) = math::matrixToEulerXYZ(R).cwiseProduct(
            eulerFreeJoint->getFlipAxisMap());
      }
      else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::XZY)
      {
        translated.segment<3>(cursor) = math::matrixToEulerXZY(R).cwiseProduct(
            eulerFreeJoint->getFlipAxisMap());
      }
      else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
      {
        translated.segment<3>(cursor) = math::matrixToEulerZXY(R).cwiseProduct(
            eulerFreeJoint->getFlipAxisMap());
      }
      else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::ZYX)
      {
        translated.segment<3>(cursor) = math::matrixToEulerZYX(R).cwiseProduct(
            eulerFreeJoint->getFlipAxisMap());
      }
      else
      {
        assert(false && "Unsupported AxisOrder when decoding EulerFreeJoint");
      }
    }
    else if (joint->getType() == EulerJoint::getStaticType())
    {
      dynamics::EulerJoint* eulerJoint
          = static_cast<dynamics::EulerJoint*>(joint);
      Eigen::Matrix3s R = math::expMapRot(translated.segment<3>(cursor));
      if (eulerJoint->getAxisOrder() == EulerJoint::AxisOrder::XYZ)
      {
        translated.segment<3>(cursor) = math::matrixToEulerXYZ(R).cwiseProduct(
            eulerJoint->getFlipAxisMap());
      }
      else if (eulerJoint->getAxisOrder() == EulerJoint::AxisOrder::XZY)
      {
        translated.segment<3>(cursor) = math::matrixToEulerXZY(R).cwiseProduct(
            eulerJoint->getFlipAxisMap());
      }
      else if (eulerJoint->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
      {
        translated.segment<3>(cursor) = math::matrixToEulerZXY(R).cwiseProduct(
            eulerJoint->getFlipAxisMap());
      }
      else if (eulerJoint->getAxisOrder() == EulerJoint::AxisOrder::ZYX)
      {
        translated.segment<3>(cursor) = math::matrixToEulerZYX(R).cwiseProduct(
            eulerJoint->getFlipAxisMap());
      }
      else
      {
        assert(false && "Unsupported AxisOrder when decoding EulerJoint");
      }
    }
    cursor += joint->getNumDofs();
  }
  return translated;
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered source joints.
Eigen::VectorXs Skeleton::getJointWorldPositions(
    const std::vector<const dynamics::Joint*>& joints) const
{
  Eigen::VectorXs sourcePositions = Eigen::VectorXs::Zero(joints.size() * 3);
  for (int i = 0; i < joints.size(); i++)
  {
    sourcePositions.segment<3>(i * 3)
        = (joints[i]->getChildBodyNode()->getWorldTransform()
           * joints[i]->getTransformFromChildBodyNode())
              .translation();
  }
  return sourcePositions;
}

//==============================================================================
/// This returns the concatenated 3-vectors for world angle of each joint's
/// child space in 3D world space, for the registered joints.
Eigen::VectorXs Skeleton::getJointWorldAngles(
    const std::vector<const dynamics::Joint*>& joints) const
{
  Eigen::VectorXs sourceAngles = Eigen::VectorXs::Zero(joints.size() * 3);
  for (int i = 0; i < joints.size(); i++)
  {
    sourceAngles.segment<3>(i * 3) = math::logMap(
        joints[i]->getChildBodyNode()->getWorldTransform().linear());
  }
  return sourceAngles;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtJointPositions(
    const std::vector<const dynamics::Joint*>& joints) const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(joints.size() * 3, getNumDofs());

  for (int i = 0; i < joints.size(); i++)
  {
    math::Jacobian bodyJac = getWorldPositionJacobian(
        joints[i]->getChildBodyNode(),
        joints[i]->getTransformFromChildBodyNode().translation());
    jac.block(3 * i, 0, 3, bodyJac.cols() - 1)
        = bodyJac.block(3, 0, 3, bodyJac.cols() - 1);
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceJointWorldPositionsJacobianWrtJointPositions(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(joints.size() * 3, getNumDofs());

  Eigen::VectorXs originalPos = getPositions();
  const double EPS = 1e-7;
  for (int i = 0; i < getNumDofs(); i++)
  {
    Eigen::VectorXs perturbed = originalPos;
    perturbed(i) += EPS;
    setPositions(perturbed);
    Eigen::VectorXs plus = getJointWorldPositions(joints);

    perturbed = originalPos;
    perturbed(i) -= EPS;
    setPositions(perturbed);
    Eigen::VectorXs minus = getJointWorldPositions(joints);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtJointChildAngles(
    const std::vector<const dynamics::Joint*>& joints) const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(joints.size() * 3, getNumDofs());

  for (int i = 0; i < joints.size(); i++)
  {
    math::Jacobian bodyJac
        = getWorldPositionJacobian(joints[i]->getChildBodyNode());
    jac.block(3 * i, 0, 3, bodyJac.cols())
        = bodyJac.block(0, 0, 3, bodyJac.cols());
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceJointWorldPositionsJacobianWrtJointChildAngles(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(joints.size() * 3, getNumDofs());

  Eigen::VectorXs originalPos = getPositions();
  const double EPS = 1e-7;
  for (int i = 0; i < getNumDofs(); i++)
  {
    Eigen::VectorXs perturbed = originalPos;
    perturbed(i) += EPS;
    setPositions(perturbed);
    Eigen::VectorXs plus = getJointWorldAngles(joints);

    perturbed = originalPos;
    perturbed(i) -= EPS;
    setPositions(perturbed);
    Eigen::VectorXs minus = getJointWorldAngles(joints);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton body scales
/// to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtBodyScales(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(joints.size() * 3, getNumBodyNodes());

  const Eigen::MatrixXi& parentMap = getParentMap();
  // Scaling a body will cause the joint offsets to scale, which will move the
  // downstream joint positions by those vectors
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* bodyNode = getBodyNode(i);
    Eigen::Matrix3s R = bodyNode->getWorldTransform().linear();

    Eigen::Vector3s parentOffset = bodyNode->getParentJoint()
                                       ->getTransformFromChildBodyNode()
                                       .translation();

    Eigen::Vector3s worldParentOffset = -R * parentOffset;

    for (int j = 0; j < joints.size(); j++)
    {
      int sourceJointDof = joints[j]->getDof(0)->getIndexInSkeleton();
      for (int k = 0; k < bodyNode->getNumChildJoints(); k++)
      {
        dynamics::Joint* childJoint = bodyNode->getChildJoint(k);
        if (childJoint == joints[j]
            || parentMap(
                childJoint->getDof(0)->getIndexInSkeleton(), sourceJointDof))
        {
          // This is the child joint

          Eigen::Vector3s childOffset
              = childJoint->getTransformFromParentBodyNode().translation();
          Eigen::Vector3s worldChildOffset = R * childOffset;
          jac.block(j * 3, i, 3, 1)
              = (worldParentOffset + worldChildOffset) / bodyNode->getScale();

          break;
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton body scales
/// to changes in source joint world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceJointWorldPositionsJacobianWrtBodyScales(
    const std::vector<const dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(joints.size() * 3, getNumBodyNodes());

  const double EPS = 1e-7;
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    s_t originalScale = getBodyNode(i)->getScale();

    getBodyNode(i)->setScale(originalScale + EPS);
    Eigen::VectorXs plus = getJointWorldPositions(joints);

    getBodyNode(i)->setScale(originalScale - EPS);
    Eigen::VectorXs minus = getJointWorldPositions(joints);

    getBodyNode(i)->setScale(originalScale);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  return jac;
}

//==============================================================================
/// These are a set of bodies, and offsets in local body space where markers
/// are mounted on the body
Eigen::VectorXs Skeleton::getMarkerWorldPositions(
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers)
{
  Eigen::VectorXs positions = Eigen::VectorXs::Zero(markers.size() * 3);
  for (int i = 0; i < markers.size(); i++)
  {
    positions.segment<3>(i * 3)
        = markers[i].first->getWorldTransform() * markers[i].second;
  }
  return positions;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getMarkerWorldPositionsJacobianWrtJointPositions(
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers) const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  for (int i = 0; i < markers.size(); i++)
  {
    math::Jacobian bodyJac
        = getWorldPositionJacobian(markers[i].first, markers[i].second);
    jac.block(3 * i, 0, 3, bodyJac.cols())
        = bodyJac.block(3, 0, 3, bodyJac.cols());
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceMarkerWorldPositionsJacobianWrtJointPositions(
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  Eigen::VectorXs originalPos = getPositions();
  const double EPS = 1e-7;
  for (int i = 0; i < getNumDofs(); i++)
  {
    Eigen::VectorXs perturbed = originalPos;
    perturbed(i) += EPS;
    setPositions(perturbed);
    Eigen::VectorXs plus = getMarkerWorldPositions(markers);

    perturbed = originalPos;
    perturbed(i) -= EPS;
    setPositions(perturbed);
    Eigen::VectorXs minus = getMarkerWorldPositions(markers);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This runs IK, attempting to fit the world positions of the passed in
/// joints to the vector of (concatenated) target positions. This can
/// optionally also rescale the skeleton.
#define DART_SKEL_LOG_IK_OUTPUT
s_t Skeleton::fitJointsToWorldPositions(
    const std::vector<const dynamics::Joint*>& positionJoints,
    Eigen::VectorXs targetPositions,
    bool scaleBodies,
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  if (scaleBodies)
  {
    Eigen::VectorXs initialPos
        = Eigen::VectorXs::Zero(getNumDofs() + getNumScaleGroups());
    initialPos.segment(0, getNumDofs()) = getPositions();
    initialPos.segment(getNumDofs(), getNumScaleGroups()) = getGroupScales();

    return math::solveIK(
        initialPos,
        positionJoints.size() * 3,
        [this](/* in*/ const Eigen::VectorXs pos) {
          // Set positions
          setPositions(pos.segment(0, getNumDofs()));
          clampPositionsToLimits();

          // Set scales
          Eigen::VectorXs newScales
              = pos.segment(getNumDofs(), getNumScaleGroups());
          for (int i = 0; i < getNumScaleGroups(); i++)
          {
            if (newScales(i) > getScaleGroupUpperBound(i))
            {
              newScales(i) = getScaleGroupUpperBound(i);
            }
            if (newScales(i) < getScaleGroupLowerBound(i))
            {
              newScales(i) = getScaleGroupLowerBound(i);
            }
          }
          setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, getNumDofs()) = getPositions();
          clampedPos.segment(getNumDofs(), getNumScaleGroups()) = newScales;
          return clampedPos;
        },
        [this, targetPositions, positionJoints](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = targetPositions - getJointWorldPositions(positionJoints);
          assert(jac.cols() == getNumDofs() + getNumScaleGroups());
          assert(jac.rows() == positionJoints.size() * 3);
          jac.setZero();
          jac.block(0, 0, positionJoints.size() * 3, getNumDofs())
              = getJointWorldPositionsJacobianWrtJointPositions(positionJoints);
          jac.block(
              0, getNumDofs(), positionJoints.size() * 3, getNumScaleGroups())
              = getJointWorldPositionsJacobianWrtGroupScales(positionJoints);
        },
        convergenceThreshold,
        maxStepCount,
        leastSquaresDamping,
        lineSearch,
        logOutput);
  }
  else
  {
    return math::solveIK(
        getPositions(),
        positionJoints.size() * 3,
        [this](/* in*/ Eigen::VectorXs pos) {
          setPositions(pos);
          clampPositionsToLimits();
          return getPositions();
        },
        [this, targetPositions, positionJoints](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = targetPositions - getJointWorldPositions(positionJoints);
          jac = getJointWorldPositionsJacobianWrtJointPositions(positionJoints);
        },
        convergenceThreshold,
        maxStepCount,
        leastSquaresDamping,
        lineSearch,
        logOutput);
  }
}

//==============================================================================
/// This runs IK, attempting to fit the world positions of the passed in
/// markers to the vector of (concatenated) target positions.
s_t Skeleton::fitMarkersToWorldPositions(
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    Eigen::VectorXs targetPositions,
    Eigen::VectorXs markerWeights,
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  return math::solveIK(
      getPositions(),
      markers.size() * 3,
      [this](/* in*/ Eigen::VectorXs pos) {
        setPositions(pos);
        clampPositionsToLimits();
        return getPositions();
      },
      [this, targetPositions, markers, markerWeights](
          /*out*/ Eigen::VectorXs& diff,
          /*out*/ Eigen::MatrixXs& jac) {
        diff = targetPositions - getMarkerWorldPositions(markers);
        for (int j = 0; j < markerWeights.size(); j++)
        {
          diff.segment<3>(j * 3) *= markerWeights(j);
        }
        jac = getMarkerWorldPositionsJacobianWrtJointPositions(markers);
      },
      convergenceThreshold,
      maxStepCount,
      leastSquaresDamping,
      lineSearch,
      logOutput);
}

//==============================================================================
void Skeleton::integratePositions(s_t _dt)
{
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    mSkelCache.mBodyNodes[i]->getParentJoint()->integratePositions(_dt);

  for (std::size_t i = 0; i < mSoftBodyNodes.size(); ++i)
  {
    for (std::size_t j = 0; j < mSoftBodyNodes[i]->getNumPointMasses(); ++j)
      mSoftBodyNodes[i]->getPointMass(j)->integratePositions(_dt);
  }
}

//==============================================================================
Eigen::VectorXs Skeleton::integratePositionsExplicit(
    Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt)
{
  Eigen::VectorXs nextPos = Eigen::VectorXs::Zero(pos.size());

  int cursor = 0;
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    Joint* joint = mSkelCache.mBodyNodes[i]->getParentJoint();
    int dofs = joint->getNumDofs();
    nextPos.segment(cursor, dofs) = joint->integratePositionsExplicit(
        pos.segment(cursor, dofs), vel.segment(cursor, dofs), dt);
    cursor += dofs;
  }

  return nextPos;
}

//==============================================================================
Eigen::MatrixXs Skeleton::getPosPosJac(
    Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(pos.size(), pos.size());

  int cursor = 0;
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    Joint* joint = mSkelCache.mBodyNodes[i]->getParentJoint();
    int dofs = joint->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = joint->getPosPosJacobian(
        pos.segment(cursor, dofs), vel.segment(cursor, dofs), dt);
    cursor += dofs;
  }

  return jac;
}

//==============================================================================
Eigen::MatrixXs Skeleton::getVelPosJac(
    Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(pos.size(), pos.size());

  int cursor = 0;
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    Joint* joint = mSkelCache.mBodyNodes[i]->getParentJoint();
    int dofs = joint->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = joint->getVelPosJacobian(
        pos.segment(cursor, dofs), vel.segment(cursor, dofs), dt);
    cursor += dofs;
  }

  return jac;
}

//==============================================================================
void Skeleton::integrateVelocities(s_t _dt)
{
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    mSkelCache.mBodyNodes[i]->getParentJoint()->integrateVelocities(_dt);

  for (std::size_t i = 0; i < mSoftBodyNodes.size(); ++i)
  {
    for (std::size_t j = 0; j < mSoftBodyNodes[i]->getNumPointMasses(); ++j)
      mSoftBodyNodes[i]->getPointMass(j)->integrateVelocities(_dt);
  }
}

//==============================================================================
Eigen::VectorXs Skeleton::getPositionDifferences(
    const Eigen::VectorXs& _q2, const Eigen::VectorXs& _q1) const
{
  if (static_cast<std::size_t>(_q2.size()) != getNumDofs()
      || static_cast<std::size_t>(_q1.size()) != getNumDofs())
  {
    dterr << "Skeleton::getPositionsDifference: q1's size[" << _q1.size()
          << "] or q2's size[" << _q2.size() << "is different with the dof ["
          << getNumDofs() << "]." << std::endl;
    return Eigen::VectorXs::Zero(getNumDofs());
  }

  Eigen::VectorXs dq(getNumDofs());

  for (const auto& bodyNode : mSkelCache.mBodyNodes)
  {
    const Joint* joint = bodyNode->getParentJoint();
    const std::size_t dof = joint->getNumDofs();

    if (dof)
    {
      std::size_t index = joint->getDof(0)->getIndexInSkeleton();
      const Eigen::VectorXs& q2Seg = _q2.segment(index, dof);
      const Eigen::VectorXs& q1Seg = _q1.segment(index, dof);
      dq.segment(index, dof) = joint->getPositionDifferences(q2Seg, q1Seg);
    }
  }

  return dq;
}

//==============================================================================
Eigen::VectorXs Skeleton::getVelocityDifferences(
    const Eigen::VectorXs& _dq2, const Eigen::VectorXs& _dq1) const
{
  if (static_cast<std::size_t>(_dq2.size()) != getNumDofs()
      || static_cast<std::size_t>(_dq1.size()) != getNumDofs())
  {
    dterr << "Skeleton::getPositionsDifference: dq1's size[" << _dq1.size()
          << "] or dq2's size[" << _dq2.size() << "is different with the dof ["
          << getNumDofs() << "]." << std::endl;
    return Eigen::VectorXs::Zero(getNumDofs());
  }

  // All the tangent spaces of Joint's configuration spaces are vector spaces.
  return _dq2 - _dq1;
}

//==============================================================================
/// This quantifies how much error the inverse dynamics result ended up with.
s_t Skeleton::ContactInverseDynamicsResult::sumError()
{
  Eigen::VectorXs oldPos = skel->getPositions();
  Eigen::VectorXs oldVel = skel->getVelocities();
  Eigen::VectorXs oldControl = skel->getControlForces();
  Eigen::Vector6s oldExtForce = contactBody->getExternalForceLocal();

  // Set to the initial conditions of the problem
  skel->setPositions(pos);
  skel->setVelocities(vel);
  const_cast<dynamics::BodyNode*>(contactBody)->setExtWrench(contactWrench);
  skel->setControlForces(jointTorques);

  // Compute one timestep
  skel->computeForwardDynamics();
  skel->integrateVelocities(skel->getTimeStep());
  Eigen::VectorXs realNextVel = skel->getVelocities();

  /*
  Eigen::MatrixXs velCompare = Eigen::MatrixXs::Zero(realNextVel.size(), 2);
  velCompare.col(0) = realNextVel;
  velCompare.col(1) = nextVel;
  std::cout << "real next vel - next vel (" << (realNextVel - nextVel).norm()
            << "): " << std::endl
            << velCompare << std::endl;
  */

  // Compute the error
  s_t error = (realNextVel - nextVel).norm();

  // Reset to old forces
  skel->setPositions(oldPos);
  skel->setVelocities(oldVel);
  skel->setControlForces(oldControl);
  const_cast<dynamics::BodyNode*>(contactBody)->setExtWrench(oldExtForce);

  return error;
}

//==============================================================================
/// This solves a simple inverse dynamics problem to get forces we need to
/// apply to arrive at "nextVel" at the next timestep.
Eigen::VectorXs Skeleton::getInverseDynamics(const Eigen::VectorXs& nextVel)
{
  Eigen::VectorXs accel
      = getVelocityDifferences(nextVel, getVelocities()) / getTimeStep();
  Eigen::VectorXs massTorques = multiplyByImplicitMassMatrix(accel);
  Eigen::VectorXs coriolisAndGravity = getCoriolisAndGravityForces()
                                       - getExternalForces() + getDampingForce()
                                       + getSpringForce();
  return massTorques + coriolisAndGravity;
}

//==============================================================================
/// This solves the inverse dynamics problem to figure out what forces we
/// would need to apply (in our _current state_) in order to get the desired
/// next velocity. This includes arbitrary forces and moments at the
/// `contactBody`, which can be post-processed down to individual contact
/// results.
Skeleton::ContactInverseDynamicsResult Skeleton::getContactInverseDynamics(
    const Eigen::VectorXs& nextVel, const dynamics::BodyNode* contactBody)
{
  ContactInverseDynamicsResult result;
  result.skel = this;
  result.contactBody = contactBody;
  result.pos = getPositions();
  result.vel = getVelocities();
  result.nextVel = nextVel;

  const dynamics::Joint* joint = getRootJoint();
  const dynamics::FreeJoint* freeJoint
      = dynamic_cast<const dynamics::FreeJoint*>(joint);
  const dynamics::EulerFreeJoint* eulerFreeJoint
      = dynamic_cast<const dynamics::EulerFreeJoint*>(joint);
  if (freeJoint == nullptr && eulerFreeJoint == nullptr)
  {
    std::cout
        << "Error: Skeleton::getContactInverseDynamics() assumes that the root "
           "joint of the skeleton is a FreeJoint or an EulerFreeJoint. Since "
           "it's neither, this function won't work and we're returning zeros."
        << std::endl;
    result.contactWrench.setZero();
    result.jointTorques = Eigen::VectorXs::Zero(getNumDofs());
    return result;
  }

  // This is the Jacobian in local body space. We're going to end up applying
  // our contact force in local body space, so this works out.
  math::Jacobian jac = getJacobian(contactBody);
  Eigen::Matrix6s jacBlock = jac.block<6, 6>(0, 0).transpose();

  Eigen::VectorXs accel
      = getVelocityDifferences(nextVel, getVelocities()) / getTimeStep();
  Eigen::VectorXs massTorques = multiplyByImplicitMassMatrix(accel);

  Eigen::VectorXs coriolisAndGravity = getCoriolisAndGravityForces()
                                       - getExternalForces() + getDampingForce()
                                       + getSpringForce();

  Eigen::Vector6s rootTorque
      = massTorques.head<6>() + coriolisAndGravity.head<6>();
  result.contactWrench
      = jacBlock.completeOrthogonalDecomposition().solve(rootTorque);
  Eigen::VectorXs contactTorques = jac.transpose() * result.contactWrench;
  result.jointTorques = massTorques + coriolisAndGravity - contactTorques;
  result.jointTorques.head<6>().setZero();

  return result;
}

//==============================================================================
/// This quantifies how much error the inverse dynamics result ended up with.
s_t Skeleton::MultipleContactInverseDynamicsResult::sumError()
{
  Eigen::VectorXs oldPos = skel->getPositions();
  Eigen::VectorXs oldVel = skel->getVelocities();
  Eigen::VectorXs oldControl = skel->getControlForces();
  std::vector<Eigen::Vector6s> oldExtForces;
  for (int i = 0; i < contactBodies.size(); i++)
  {
    oldExtForces.push_back(contactBodies[i]->getExternalForceLocal());
  }

  // Set to the initial conditions of the problem
  skel->setPositions(pos);
  skel->setVelocities(vel);
  for (int i = 0; i < contactBodies.size(); i++)
  {
    /*
    std::cout << "Contact wrench " << i << ": " << contactWrenches[i]
              << std::endl;
    */
    const_cast<dynamics::BodyNode*>(contactBodies[i])
        ->setExtWrench(contactWrenches[i]);
  }
  // std::cout << "Control torques: " << jointTorques << std::endl;
  skel->setControlForces(jointTorques);

  // Compute one timestep
  skel->computeForwardDynamics();
  skel->integrateVelocities(skel->getTimeStep());
  Eigen::VectorXs realNextVel = skel->getVelocities();

  /*
  Eigen::MatrixXs velCompare = Eigen::MatrixXs::Zero(realNextVel.size(), 2);
  velCompare.col(0) = realNextVel;
  velCompare.col(1) = nextVel;
  std::cout << "real next vel - next vel (" << (realNextVel - nextVel).norm()
            << "): " << std::endl
            << velCompare << std::endl;
  */

  // Compute the error
  s_t error = (realNextVel - nextVel).norm();

  // Reset to old forces
  skel->setPositions(oldPos);
  skel->setVelocities(oldVel);
  skel->setControlForces(oldControl);
  for (int i = 0; i < contactBodies.size(); i++)
  {
    const_cast<dynamics::BodyNode*>(contactBodies[i])
        ->setExtWrench(oldExtForces[i]);
  }

  return error;
}

//==============================================================================
/// This computes the difference between the guess and the closest valid
/// solution
s_t Skeleton::MultipleContactInverseDynamicsResult::computeGuessLoss()
{
  s_t loss = 0.0;
  for (int i = 0; i < contactBodies.size(); i++)
  {
    loss += (contactWrenchGuesses[i] - contactWrenches[i]).squaredNorm();
  }
  return loss;
}

//==============================================================================
/// If you pass in multiple simultaneous contacts, with guesses about the
/// contact wrenches for each body, this method will find the least-squares
/// closest solution for contact wrenches on each body that will satisfying
/// the next velocity constraint. This is intended to be useful for EM loops
/// for learning rich contact models. Without initial guesses, the solution is
/// not unique, so in order to use this method to get useful inverse dynamics
/// you'll need good initial guesses.
Skeleton::MultipleContactInverseDynamicsResult
Skeleton::getMultipleContactInverseDynamics(
    const Eigen::VectorXs& nextVel,
    std::vector<const dynamics::BodyNode*> bodies,
    std::vector<Eigen::Vector6s> bodyWrenchGuesses)
{
  MultipleContactInverseDynamicsResult result;
  result.skel = this;
  result.contactBodies = bodies;
  result.pos = getPositions();
  result.vel = getVelocities();
  result.nextVel = nextVel;

  const dynamics::Joint* joint = getRootJoint();
  const dynamics::FreeJoint* freeJoint
      = dynamic_cast<const dynamics::FreeJoint*>(joint);
  const dynamics::EulerFreeJoint* eulerFreeJoint
      = dynamic_cast<const dynamics::EulerFreeJoint*>(joint);
  if (freeJoint == nullptr && eulerFreeJoint == nullptr)
  {
    std::cout
        << "Error: Skeleton::getContactInverseDynamics() assumes that the root "
           "joint of the skeleton is a FreeJoint or an EulerFreeJoint. Since "
           "it's neither, this function won't work and we're returning zeros."
        << std::endl;
    result.contactWrenches = std::vector<Eigen::Vector6s>();
    for (int i = 0; i < bodies.size(); i++)
      result.contactWrenches.push_back(Eigen::Vector6s::Zero());
    result.jointTorques = Eigen::VectorXs::Zero(getNumDofs());
    return result;
  }

  // This is the Jacobian in local body space. We're going to end up applying
  // our contact force in local body space, so this works out.
  Eigen::MatrixXs jacs = Eigen::MatrixXs::Zero(6 * bodies.size(), getNumDofs());
  assert(bodies.size() > 0);

  for (int i = 0; i < bodies.size(); i++)
  {
    jacs.block(6 * i, 0, 6, getNumDofs()) = getJacobian(bodies[i]);
  }
  Eigen::MatrixXs jacBlock = jacs.block(0, 0, 6 * bodies.size(), 6).transpose();

  Eigen::VectorXs massTorques = multiplyByImplicitMassMatrix(
      (nextVel - getVelocities()) / getTimeStep());

  Eigen::VectorXs coriolisAndGravity = getCoriolisAndGravityForces()
                                       - getExternalForces() + getDampingForce()
                                       + getSpringForce();

  Eigen::Vector6s rootTorque
      = massTorques.head<6>() + coriolisAndGravity.head<6>();

  Eigen::VectorXs correctedForces = Eigen::VectorXs::Zero(6 * bodies.size());

  // If no guesses are passed in to us, we'll do our best to construct something
  // sensible using the heuristic that we'd like a guess that minimizes the
  // torques required at the bodies. That amounts to minimizing the moment arm
  // between each body and its center of pressure. We can construct and solve a
  // linearly constrained QP in closed form.
  if (bodyWrenchGuesses.size() == 0)
  {
    // We want to take a minimum of torques.
    int n = 6 * bodies.size();
    int m = 6;

    // Create a diagonal weight matrix for our QP.
    s_t eps = 0.01;
    Eigen::MatrixXs B = Eigen::MatrixXs::Identity(n, n);
    for (int i = 0; i < bodies.size(); i++)
    {
      B(i * 6 + 3, i * 6 + 3) = eps;
      B(i * 6 + 4, i * 6 + 4) = eps;
      B(i * 6 + 5, i * 6 + 5) = eps;
    }

    // Create the KKT matrix for our QP
    Eigen::MatrixXs KKT = Eigen::MatrixXs::Zero(n + m, n + m);
    KKT.block(0, 0, n, n) = B;
    KKT.block(n, 0, m, n) = jacBlock;
    KKT.block(0, n, n, m) = jacBlock.transpose();

    Eigen::VectorXs KKTeq = Eigen::VectorXs::Zero(n + m);
    KKTeq.segment(n, m) = rootTorque;

    Eigen::VectorXs KKTSolution = KKT.householderQr().solve(KKTeq);

    // Read the forces off of the solution to the KKT conditions
    correctedForces = KKTSolution.segment(0, n);
  }
  // If we were handed guesses, just find the least-squares nearest values for
  // contact force that still satisfy inverse dynamics.
  else
  {
    result.contactWrenchGuesses = bodyWrenchGuesses;

    Eigen::VectorXs forces = Eigen::VectorXs::Zero(6 * bodies.size());
    for (int i = 0; i < bodies.size(); i++)
    {
      forces.segment(6 * i, 6) = bodyWrenchGuesses[i];
    }
    correctedForces = jacBlock.completeOrthogonalDecomposition().solve(
                          rootTorque - jacBlock * forces)
                      + forces;
  }

  result.contactWrenches = std::vector<Eigen::Vector6s>();
  for (int i = 0; i < bodies.size(); i++)
  {
    result.contactWrenches.push_back(Eigen::Vector6s::Zero());
    result.contactWrenches[i] = correctedForces.segment(i * 6, 6);
  }
  Eigen::VectorXs contactTorques = jacs.transpose() * correctedForces;
  result.jointTorques = massTorques + coriolisAndGravity - contactTorques;
  result.jointTorques.head<6>().setZero();

  return result;
}

//==============================================================================
/// This computes how much the actual dynamics we get when we apply this
/// solution differ from the goal solution.
s_t Skeleton::MultipleContactInverseDynamicsOverTimeResult::sumError()
{
  Eigen::VectorXs oldPos = skel->getPositions();
  Eigen::VectorXs oldVel = skel->getVelocities();
  Eigen::VectorXs oldControl = skel->getControlForces();
  std::vector<Eigen::Vector6s> oldExtForces;
  for (int i = 0; i < contactBodies.size(); i++)
  {
    oldExtForces.push_back(contactBodies[i]->getExternalForceLocal());
  }

  s_t error = 0.0;
  for (int i = 0; i < timesteps; i++)
  {
    // Set to the initial conditions of the problem
    skel->setPositions(positions.col(i));
    skel->setVelocities(velocities.col(i));
    for (int j = 0; j < contactBodies.size(); j++)
    {
      const_cast<dynamics::BodyNode*>(contactBodies[j])
          ->setExtWrench(contactWrenches[i][j]);
    }
    skel->setControlForces(jointTorques.col(i));

    // Compute one timestep
    skel->computeForwardDynamics();
    skel->integrateVelocities(skel->getTimeStep());
    Eigen::VectorXs realNextVel = skel->getVelocities();

    // Compute the error
    error += (realNextVel - nextVelocities.col(i)).norm();
  }

  // Reset to old forces
  skel->setPositions(oldPos);
  skel->setVelocities(oldVel);
  skel->setControlForces(oldControl);
  for (int i = 0; i < contactBodies.size(); i++)
  {
    const_cast<dynamics::BodyNode*>(contactBodies[i])
        ->setExtWrench(oldExtForces[i]);
  }

  return error;
}

//==============================================================================
s_t Skeleton::MultipleContactInverseDynamicsOverTimeResult::
    computeSmoothnessLoss()
{
  s_t loss = 0.0;
  for (int i = 1; i < timesteps; i++)
  {
    for (int j = 0; j < contactBodies.size(); j++)
    {
      loss += (contactWrenches[i][j] - contactWrenches[i - 1][j]).squaredNorm();
    }
  }

  return loss;
}

//==============================================================================
s_t Skeleton::MultipleContactInverseDynamicsOverTimeResult::
    computePrevForceLoss()
{
  s_t loss = 0.0;
  for (int j = 0; j < contactBodies.size(); j++)
  {
    loss += (contactWrenches[0][j] - prevContactForces[j]).squaredNorm();
  }
  return loss;
}

Eigen::MatrixXs Skeleton::EMPTY = Eigen::MatrixXs::Zero(0, 0);

//==============================================================================
/// This sets up and solves a QP that tracks multiple contacts over a
/// time-series of positions. This has two blending factors to control the
/// solution, a `smoothingWeight` and a `minTorqueWeight`. Increasing the
/// smoothing weight will prioritize a smoother (less time varying) set of
/// contact forces. Increasing the minimize torques weight will prioritize
/// solutions at each timestep that minimize the torque-component of the
/// contact forces at each body.
///
/// This will not provide a contact solution at the last two timesteps passed
/// in, because it cannot compute a velocity and acceleration at those
/// timesteps.
Skeleton::MultipleContactInverseDynamicsOverTimeResult
Skeleton::getMultipleContactInverseDynamicsOverTime(
    const Eigen::MatrixXs& positions,
    std::vector<const dynamics::BodyNode*> bodies,
    // This allows us to penalize non-smooth GRFs
    s_t smoothingWeight,
    // This allows us to penalize large torques in our GRFs
    s_t minTorqueWeight,
    // This allows us to penalize GRFs on rapidly moving bodies
    std::function<s_t(s_t)> velocityPenalty,
    // This allows us to specify exactly what we want the initial forces to be
    std::vector<Eigen::Vector6s> prevContactForces,
    s_t prevContactWeight,
    // This allows us to specify how we'd like to penalize magnitudes of
    // different contact forces frame-by-frame
    Eigen::MatrixXs magnitudeCosts)
{
  MultipleContactInverseDynamicsOverTimeResult result;
  result.skel = this;
  result.contactBodies = bodies;

  Eigen::VectorXs oldPos = getPositions();
  Eigen::VectorXs oldVel = getVelocities();
  Eigen::VectorXs oldControl = getControlForces();

  int fDim = 6 * bodies.size();
  int dofs = getNumDofs();

  int timesteps = positions.cols() - 2;
  result.timesteps = timesteps;
  Eigen::MatrixXs B = Eigen::MatrixXs::Zero(fDim * timesteps, fDim * timesteps);
  Eigen::VectorXs b = Eigen::VectorXs::Zero(fDim * timesteps);
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(6 * timesteps, fDim * timesteps);
  Eigen::VectorXs c = Eigen::VectorXs::Zero(6 * timesteps);

  result.positions = Eigen::MatrixXs::Zero(dofs, timesteps);
  result.velocities = Eigen::MatrixXs::Zero(dofs, timesteps);
  result.nextVelocities = Eigen::MatrixXs::Zero(dofs, timesteps);
  result.jointTorques = Eigen::MatrixXs::Zero(dofs, timesteps);

  std::vector<Eigen::MatrixXs> timestepJacs;
  std::vector<Eigen::VectorXs> timestepJointTorques;

  Eigen::MatrixXs torqueStamp = Eigen::MatrixXs::Identity(fDim, fDim);
  s_t eps = 0.01;
  for (int j = 0; j < bodies.size(); j++)
  {
    torqueStamp(j * 6 + 3, j * 6 + 3) = eps;
    torqueStamp(j * 6 + 4, j * 6 + 4) = eps;
    torqueStamp(j * 6 + 5, j * 6 + 5) = eps;
  }

  B.block(0, 0, fDim, fDim)
      += prevContactWeight * Eigen::MatrixXs::Identity(fDim, fDim);
  if (prevContactWeight > 0)
  {
    result.prevContactForces = prevContactForces;
    assert(prevContactForces.size() == bodies.size());
    for (int i = 0; i < bodies.size(); i++)
    {
      b.segment<6>(i * 6) = -2 * prevContactWeight * prevContactForces[i];
    }
  }
  for (int i = 0; i < timesteps; i++)
  {
    B.block(fDim * i, fDim * i, fDim, fDim) += minTorqueWeight * torqueStamp;
    if (i + 1 < timesteps)
    {
      B.block(fDim * i, fDim * i, fDim, fDim)
          += smoothingWeight * Eigen::MatrixXs::Identity(fDim, fDim);
      B.block(fDim * (i + 1), fDim * i, fDim, fDim)
          -= smoothingWeight * Eigen::MatrixXs::Identity(fDim, fDim);
      B.block(fDim * i, fDim * (i + 1), fDim, fDim)
          -= smoothingWeight * Eigen::MatrixXs::Identity(fDim, fDim);
      B.block(fDim * (i + 1), fDim * (i + 1), fDim, fDim)
          += smoothingWeight * Eigen::MatrixXs::Identity(fDim, fDim);
    }
    if (magnitudeCosts.rows() == bodies.size())
    {
      for (int j = 0; j < bodies.size(); j++)
      {
        B.block<6, 6>(fDim * i + j * 6, fDim * i + j * 6)
            += Eigen::Matrix6s::Identity() * magnitudeCosts(j, i);
      }
    }

    Eigen::VectorXs vel
        = getPositionDifferences(positions.col(i + 1), positions.col(i))
          / getTimeStep();
    Eigen::VectorXs nextVel
        = getPositionDifferences(positions.col(i + 2), positions.col(i + 1))
          / getTimeStep();
    Eigen::VectorXs accel
        = getVelocityDifferences(nextVel, vel) / getTimeStep();

    result.positions.col(i) = positions.col(i);
    result.velocities.col(i) = vel;
    result.nextVelocities.col(i) = nextVel;

    setPositions(positions.col(i));
    setVelocities(vel);

    // Get the spatial velocities of each of the contact bodies, and construct a
    // weighted cost for applying contact wrenches.

    Eigen::VectorXs velCosts = Eigen::VectorXs::Ones(fDim);
    for (int j = 0; j < bodies.size(); j++)
    {
      Eigen::Vector3s worldVel = bodies[j]->getLinearVelocity();
      s_t velNorm = worldVel.squaredNorm();
      velCosts.segment<6>(j * 6) *= velocityPenalty(velNorm);
    }
    B.block(fDim * i, fDim * i, fDim, fDim) += velCosts.asDiagonal();

    // This is the Jacobian in local body space. We're going to end up applying
    // our contact force in local body space, so this works out.
    Eigen::MatrixXs jacs = Eigen::MatrixXs::Zero(fDim, getNumDofs());
    assert(bodies.size() > 0);

    for (int i = 0; i < bodies.size(); i++)
    {
      jacs.block(6 * i, 0, 6, getNumDofs()) = getJacobian(bodies[i]);
    }
    timestepJacs.push_back(jacs);

    A.block(6 * i, fDim * i, 6, fDim) = jacs.block(0, 0, fDim, 6).transpose();

    Eigen::VectorXs jointTorques
        = (multiplyByImplicitMassMatrix(accel) + getCoriolisAndGravityForces()
           - getExternalForces() + getDampingForce() + getSpringForce());
    timestepJointTorques.push_back(jointTorques);
    c.segment<6>(6 * i) = jointTorques.head<6>();
  }

  // We now have B, b, A, and c, so build the KKT matrix

  Eigen::MatrixXs kktMatrix
      = Eigen::MatrixXs::Zero(B.rows() + A.rows(), B.rows() + A.rows());
  kktMatrix.block(0, 0, B.rows(), B.cols()) = 2 * B;
  kktMatrix.block(B.rows(), 0, A.rows(), A.cols()) = A;
  kktMatrix.block(0, B.cols(), A.cols(), A.rows()) = A.transpose();
  Eigen::VectorXs kktVector = Eigen::VectorXs::Zero(b.size() + c.size());
  kktVector.segment(0, b.size()) = -b;
  kktVector.segment(b.size(), c.size()) = c;

  // Now factor and solve:

  Eigen::VectorXs kktSolution = kktMatrix.householderQr().solve(kktVector);

  // And we can read the solution off of the result:
  for (int i = 0; i < timesteps; i++)
  {
    std::vector<Eigen::Vector6s> timestepContactWrenches;
    for (int j = 0; j < bodies.size(); j++)
    {
      timestepContactWrenches.push_back(
          kktSolution.segment<6>(i * fDim + j * 6));
    }
    result.contactWrenches.push_back(timestepContactWrenches);

    Eigen::VectorXs contactTorques
        = timestepJacs[i].transpose() * kktSolution.segment(i * fDim, fDim);
    result.jointTorques.col(i) = timestepJointTorques[i] - contactTorques;
    result.jointTorques.col(i).head<6>().setZero();
  }

  setPositions(oldPos);
  setVelocities(oldVel);
  setControlForces(oldControl);
  return result;
}

//==============================================================================
static bool isValidBodyNode(
    const Skeleton* _skeleton,
    const JacobianNode* _node,
    const std::string& _fname)
{
  if (nullptr == _node)
  {
    dtwarn << "[Skeleton::" << _fname << "] Invalid BodyNode pointer: "
           << "nullptr. Returning zero Jacobian.\n";
    assert(false);
    return false;
  }

  // The given BodyNode should be in the Skeleton
  if (_node->getSkeleton().get() != _skeleton)
  {
    dtwarn << "[Skeleton::" << _fname
           << "] Attempting to get a Jacobian for a "
              "BodyNode ["
           << _node->getName() << "] (" << _node
           << ") that is not in this Skeleton [" << _skeleton->getName()
           << "] (" << _skeleton << "). Returning zero Jacobian.\n";
    assert(false);
    return false;
  }

  return true;
}

//==============================================================================
template <typename JacobianType>
void assignJacobian(
    JacobianType& _J, const JacobianNode* _node, const JacobianType& _JBodyNode)
{
  // Assign the BodyNode's Jacobian to the result Jacobian.
  std::size_t localIndex = 0;
  const auto& indices = _node->getDependentGenCoordIndices();
  for (const auto& index : indices)
  {
    // Each index should be less than the number of dofs of this Skeleton.
    assert(index < _node->getSkeleton()->getNumDofs());

    _J.col(index) = _JBodyNode.col(localIndex++);
  }
}

//==============================================================================
template <typename... Args>
math::Jacobian variadicGetJacobian(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::Jacobian J = math::Jacobian::Zero(6, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getJacobian"))
    return J;

  const math::Jacobian JBodyNode = _node->getJacobian(args...);

  assignJacobian<math::Jacobian>(J, _node, JBodyNode);

  return J;
}

//==============================================================================
template <typename... Args>
math::Jacobian variadicGetJacobianInPositionSpace(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::Jacobian J = math::Jacobian::Zero(6, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getJacobian"))
    return J;

  const math::Jacobian JBodyNode = _node->getJacobianInPositionSpace(args...);

  assignJacobian<math::Jacobian>(J, _node, JBodyNode);

  return J;
}

//==============================================================================
math::Jacobian Skeleton::getJacobian(const JacobianNode* _node) const
{
  return variadicGetJacobian(this, _node);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianInPositionSpace(
    const JacobianNode* _node) const
{
  return variadicGetJacobianInPositionSpace(this, _node);
}

//==============================================================================
math::Jacobian Skeleton::getJacobian(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobian(this, _node, _inCoordinatesOf);
}

//==============================================================================
math::Jacobian Skeleton::getJacobian(
    const JacobianNode* _node, const Eigen::Vector3s& _localOffset) const
{
  return variadicGetJacobian(this, _node, _localOffset);
}

//==============================================================================
math::Jacobian Skeleton::getJacobian(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobian(this, _node, _localOffset, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::Jacobian variadicGetWorldJacobian(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::Jacobian J = math::Jacobian::Zero(6, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getWorldJacobian"))
    return J;

  const math::Jacobian JBodyNode = _node->getWorldJacobian(args...);

  assignJacobian<math::Jacobian>(J, _node, JBodyNode);

  return J;
}

//==============================================================================
math::Jacobian Skeleton::getWorldPositionJacobian(
    const JacobianNode* _node) const
{
  return getWorldPositionJacobian(_node, Eigen::Vector3s::Zero());
}

//==============================================================================
math::Jacobian Skeleton::getWorldPositionJacobian(
    const JacobianNode* _node, const Eigen::Vector3s& _localOffset) const
{
  math::Jacobian J = math::Jacobian::Zero(6, getNumDofs());

  if (!isValidBodyNode(this, _node, "getWorldJacobian"))
    return J;

  const BodyNode* bodyNode = static_cast<const BodyNode*>(_node);
  Eigen::Vector3s originalRotation
      = math::logMap(bodyNode->getWorldTransform().linear());

  for (int i = 0; i < getNumDofs(); i++)
  {
    const DegreeOfFreedom* dof = getDof(i);
    const Joint* joint = dof->getJoint();

    bool isParent = false;
    const BodyNode* cursor = bodyNode;
    while (cursor != nullptr)
    {
      if (cursor->getParentJoint() == joint)
      {
        isParent = true;
        break;
      }
      if (cursor->getParentJoint() != nullptr)
      {
        cursor = cursor->getParentJoint()->getParentBodyNode();
      }
    }

    if (isParent)
    {
      Eigen::Vector6s screw
          = joint->getWorldAxisScrewForPosition(dof->getIndexInJoint());
      screw.tail<3>() += screw.head<3>().cross(
          bodyNode->getWorldTransform() * _localOffset);
      // This is key so we get an actual gradient of the angle (as a screw),
      // rather than just a screw representing a rotation.
      screw.head<3>()
          = math::expMapNestedGradient(originalRotation, screw.head<3>());

      J.col(i) = screw;
    }
    // else leave J.col(i) as zeros
  }

  return J;
}

//==============================================================================
math::Jacobian Skeleton::finiteDifferenceWorldPositionJacobian(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersWorldPositionJacobian(_node, _localOffset);
  }
  math::Jacobian J = math::Jacobian::Zero(6, getNumDofs());
  s_t EPS = 1e-5;
  for (int i = 0; i < getNumDofs(); i++)
  {
    s_t original = getPosition(i);
    setPosition(i, original + EPS);
    Eigen::Vector6s plus = Eigen::Vector6s::Zero();
    plus.head<3>() = math::logMap(_node->getWorldTransform().linear());
    plus.tail<3>() = _node->getWorldTransform() * _localOffset;
    setPosition(i, original - EPS);
    Eigen::Vector6s minus = Eigen::Vector6s::Zero();
    minus.head<3>() = math::logMap(_node->getWorldTransform().linear());
    minus.tail<3>() = _node->getWorldTransform() * _localOffset;
    J.col(i) = (plus - minus) / (2 * EPS);
    setPosition(i, original);
  }
  return J;
}

//==============================================================================
math::Jacobian Skeleton::finiteDifferenceRiddersWorldPositionJacobian(
    const JacobianNode* _node, const Eigen::Vector3s& _localOffset)
{
  std::size_t n = getNumDofs();
  Eigen::MatrixXs J = math::Jacobian::Zero(6, n);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < n; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::Vector6s, tabSize>, tabSize> tab;

    s_t original = getPosition(i);

    setPosition(i, original + stepSize);
    Eigen::Vector6s plus = Eigen::Vector6s::Zero();
    plus.head<3>() = math::logMap(_node->getWorldTransform().linear());
    plus.tail<3>() = _node->getWorldTransform() * _localOffset;

    setPosition(i, original - stepSize);
    Eigen::Vector6s minus = Eigen::Vector6s::Zero();
    minus.head<3>() = math::logMap(_node->getWorldTransform().linear());
    minus.tail<3>() = _node->getWorldTransform() * _localOffset;

    setPosition(i, original);

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      s_t original = getPosition(i);

      setPosition(i, original + stepSize);
      Eigen::Vector6s plus = Eigen::Vector6s::Zero();
      plus.head<3>() = math::logMap(_node->getWorldTransform().linear());
      plus.tail<3>() = _node->getWorldTransform() * _localOffset;

      setPosition(i, original - stepSize);
      Eigen::Vector6s minus = Eigen::Vector6s::Zero();
      minus.head<3>() = math::logMap(_node->getWorldTransform().linear());
      minus.tail<3>() = _node->getWorldTransform() * _localOffset;

      setPosition(i, original);

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  return J;
}

//==============================================================================
math::Jacobian Skeleton::getWorldJacobian(const JacobianNode* _node) const
{
  return variadicGetWorldJacobian(this, _node);
}

//==============================================================================
math::Jacobian Skeleton::getWorldJacobian(
    const JacobianNode* _node, const Eigen::Vector3s& _localOffset) const
{
  return variadicGetWorldJacobian(this, _node, _localOffset);
}

//==============================================================================
template <typename... Args>
math::LinearJacobian variadicGetLinearJacobian(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::LinearJacobian J = math::LinearJacobian::Zero(3, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getLinearJacobian"))
    return J;

  const math::LinearJacobian JBodyNode = _node->getLinearJacobian(args...);

  assignJacobian<math::LinearJacobian>(J, _node, JBodyNode);

  return J;
}

//==============================================================================
math::LinearJacobian Skeleton::getLinearJacobian(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetLinearJacobian(this, _node, _inCoordinatesOf);
}

//==============================================================================
math::LinearJacobian Skeleton::getLinearJacobian(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const Frame* _inCoordinatesOf) const
{
  return variadicGetLinearJacobian(this, _node, _localOffset, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::AngularJacobian variadicGetAngularJacobian(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::AngularJacobian J = math::AngularJacobian::Zero(3, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getAngularJacobian"))
    return J;

  const math::AngularJacobian JBodyNode = _node->getAngularJacobian(args...);

  assignJacobian<math::AngularJacobian>(J, _node, JBodyNode);

  return J;
}

//==============================================================================
math::AngularJacobian Skeleton::getAngularJacobian(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetAngularJacobian(this, _node, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::Jacobian variadicGetJacobianSpatialDeriv(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::Jacobian dJ = math::Jacobian::Zero(6, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getJacobianSpatialDeriv"))
    return dJ;

  const math::Jacobian dJBodyNode = _node->getJacobianSpatialDeriv(args...);

  assignJacobian<math::Jacobian>(dJ, _node, dJBodyNode);

  return dJ;
}

//==============================================================================
math::Jacobian Skeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node) const
{
  return variadicGetJacobianSpatialDeriv(this, _node);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobianSpatialDeriv(this, _node, _inCoordinatesOf);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node, const Eigen::Vector3s& _localOffset) const
{
  return variadicGetJacobianSpatialDeriv(this, _node, _localOffset);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobianSpatialDeriv(
      this, _node, _localOffset, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::Jacobian variadicGetJacobianClassicDeriv(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::Jacobian dJ = math::Jacobian::Zero(6, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getJacobianClassicDeriv"))
    return dJ;

  const math::Jacobian dJBodyNode = _node->getJacobianClassicDeriv(args...);

  assignJacobian<math::Jacobian>(dJ, _node, dJBodyNode);

  return dJ;
}

//==============================================================================
math::Jacobian Skeleton::getJacobianClassicDeriv(
    const JacobianNode* _node) const
{
  return variadicGetJacobianClassicDeriv(this, _node);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianClassicDeriv(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobianClassicDeriv(this, _node, _inCoordinatesOf);
}

//==============================================================================
math::Jacobian Skeleton::getJacobianClassicDeriv(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const Frame* _inCoordinatesOf) const
{
  return variadicGetJacobianClassicDeriv(
      this, _node, _localOffset, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::LinearJacobian variadicGetLinearJacobianDeriv(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::LinearJacobian dJv = math::LinearJacobian::Zero(3, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getLinearJacobianDeriv"))
    return dJv;

  const math::LinearJacobian dJvBodyNode
      = _node->getLinearJacobianDeriv(args...);

  assignJacobian<math::LinearJacobian>(dJv, _node, dJvBodyNode);

  return dJv;
}

//==============================================================================
math::LinearJacobian Skeleton::getLinearJacobianDeriv(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetLinearJacobianDeriv(this, _node, _inCoordinatesOf);
}

//==============================================================================
math::LinearJacobian Skeleton::getLinearJacobianDeriv(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const Frame* _inCoordinatesOf) const
{
  return variadicGetLinearJacobianDeriv(
      this, _node, _localOffset, _inCoordinatesOf);
}

//==============================================================================
template <typename... Args>
math::AngularJacobian variadicGetAngularJacobianDeriv(
    const Skeleton* _skel, const JacobianNode* _node, Args... args)
{
  math::AngularJacobian dJw
      = math::AngularJacobian::Zero(3, _skel->getNumDofs());

  if (!isValidBodyNode(_skel, _node, "getAngularJacobianDeriv"))
    return dJw;

  const math::AngularJacobian dJwBodyNode
      = _node->getAngularJacobianDeriv(args...);

  assignJacobian<math::AngularJacobian>(dJw, _node, dJwBodyNode);

  return dJw;
}

//==============================================================================
math::AngularJacobian Skeleton::getAngularJacobianDeriv(
    const JacobianNode* _node, const Frame* _inCoordinatesOf) const
{
  return variadicGetAngularJacobianDeriv(this, _node, _inCoordinatesOf);
}

//==============================================================================
s_t Skeleton::getMass() const
{
  return mTotalMass;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getMassMatrix(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mMassMatrix)
    updateMassMatrix(_treeIdx);
  return mTreeCache[_treeIdx].mM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getMassMatrix() const
{
  if (mSkelCache.mDirty.mMassMatrix)
    updateMassMatrix();
  return mSkelCache.mM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getAugMassMatrix(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mAugMassMatrix)
    updateAugMassMatrix(_treeIdx);

  return mTreeCache[_treeIdx].mAugM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getAugMassMatrix() const
{
  if (mSkelCache.mDirty.mAugMassMatrix)
    updateAugMassMatrix();

  return mSkelCache.mAugM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getInvMassMatrix(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mInvMassMatrix)
    updateInvMassMatrix(_treeIdx);

  return mTreeCache[_treeIdx].mInvM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getInvMassMatrix() const
{
  if (mSkelCache.mDirty.mInvMassMatrix)
    updateInvMassMatrix();

  return mSkelCache.mInvM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getInvAugMassMatrix(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mInvAugMassMatrix)
    updateInvAugMassMatrix(_treeIdx);

  return mTreeCache[_treeIdx].mInvAugM;
}

//==============================================================================
const Eigen::MatrixXs& Skeleton::getInvAugMassMatrix() const
{
  if (mSkelCache.mDirty.mInvAugMassMatrix)
    updateInvAugMassMatrix();

  return mSkelCache.mInvAugM;
}

//==============================================================================
Eigen::VectorXs Skeleton::multiplyByImplicitMassMatrix(Eigen::VectorXs x)
{
  // The trick here is to treat x as delta acceleration, and measure delta force
  std::size_t dof = mSkelCache.mDofs.size();
  if (dof == 0)
  {
    return x;
  }

  // Backup the original internal force
  Eigen::VectorXs originalGenAcceleration = getAccelerations();

  // Set the acceleration the DOFs to x, which will allow us to compute M*x
  // through Featherstone
  setAccelerations(x);

  // We don't need to set this to 0 if the below is correct
  Eigen::VectorXs finalResult = Eigen::VectorXs(dof);

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    DataCache& cache = mTreeCache[tree];
    std::size_t dof = cache.mDofs.size();
    if (dof == 0)
    {
      continue;
    }

    // Prepare cache data
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateMassMatrix();
    }

    // Collect the result of (M * x) for this tree
    Eigen::MatrixXs treeMulResult
        = Eigen::MatrixXs::Zero(cache.mDofs.size(), 1);
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->aggregateMassMatrix(treeMulResult, 0);
    }

    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t ki = treeDofs[i]->getIndexInSkeleton();

      finalResult(ki) = treeMulResult(i, 0);
    }
  }

  // Restore the original generalized accelerations
  const_cast<Skeleton*>(this)->setAccelerations(originalGenAcceleration);

  return finalResult;
}

//==============================================================================
Eigen::VectorXs Skeleton::multiplyByImplicitInvMassMatrix(Eigen::VectorXs x)
{
  // The trick here is to treat x as delta force, and measure delta acceleration

  std::size_t dof = mSkelCache.mDofs.size();
  assert(
      static_cast<std::size_t>(mSkelCache.mInvM.cols()) == dof
      && static_cast<std::size_t>(mSkelCache.mInvM.rows()) == dof);
  if (dof == 0)
  {
    return x;
  }

  // Backup the origianl internal force
  Eigen::VectorXs originalInternalForce = getControlForces();

  // Set the forces on the DOFs to x, which will allow us to compute Minv*x
  // through Featherstone
  setControlForces(x);

  // We don't need to set this to 0 if the below is correct
  Eigen::VectorXs finalResult = Eigen::VectorXs(dof);

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    DataCache& cache = mTreeCache[tree];
    std::size_t dof = cache.mDofs.size();
    if (dof == 0)
    {
      continue;
    }

    // Prepare cache data
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->updateInvMassMatrix();
    }

    // Collect the result of (Minv * x) for this tree
    Eigen::MatrixXs treeMulResult
        = Eigen::MatrixXs::Zero(cache.mDofs.size(), 1);
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->aggregateInvMassMatrix(treeMulResult, 0);
    }

    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t ki = treeDofs[i]->getIndexInSkeleton();

      finalResult(ki) = treeMulResult(i, 0);
    }
  }

  // Restore the original internal force
  const_cast<Skeleton*>(this)->setControlForces(originalInternalForce);

  return finalResult;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getCoriolisForces(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mCoriolisForces)
    updateCoriolisForces(_treeIdx);

  return mTreeCache[_treeIdx].mCvec;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getCoriolisForces() const
{
  if (mSkelCache.mDirty.mCoriolisForces)
    updateCoriolisForces();

  return mSkelCache.mCvec;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getGravityForces(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mGravityForces)
    updateGravityForces(_treeIdx);

  return mTreeCache[_treeIdx].mG;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getGravityForces() const
{
  if (mSkelCache.mDirty.mGravityForces)
    updateGravityForces();

  return mSkelCache.mG;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getCoriolisAndGravityForces(
    std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mCoriolisAndGravityForces)
    updateCoriolisAndGravityForces(_treeIdx);

  return mTreeCache[_treeIdx].mCg;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getCoriolisAndGravityForces() const
{
  if (mSkelCache.mDirty.mCoriolisAndGravityForces)
    updateCoriolisAndGravityForces();

  return mSkelCache.mCg;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getExternalForces(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mExternalForces)
    updateExternalForces(_treeIdx);

  return mTreeCache[_treeIdx].mFext;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getExternalForces() const
{
  if (mSkelCache.mDirty.mExternalForces)
    updateExternalForces();

  return mSkelCache.mFext;
}
//==============================================================================
Eigen::VectorXs Skeleton::getDampingCoeffVector()
{
  std::vector<dynamics::DegreeOfFreedom*> dofs = getDofs();
  size_t nDofs = getNumDofs();
  Eigen::VectorXs damp_coeffs = Eigen::VectorXs::Zero(nDofs);
  for (int i = 0; i < nDofs; i++)
  {
    damp_coeffs(i) = dofs[i]->getDampingCoefficient();
  }
  return damp_coeffs;
}

Eigen::VectorXs Skeleton::getDampingForce()
{
  Eigen::VectorXs velocities = getVelocities();
  Eigen::VectorXs damp_coeffs = getDampingCoeffVector();
  Eigen::VectorXs damp_force = damp_coeffs.asDiagonal() * velocities;
  return damp_force;
}

//==============================================================================
Eigen::VectorXs Skeleton::getSpringStiffVector()
{
  std::vector<dynamics::DegreeOfFreedom*> dofs = getDofs();
  size_t nDofs = getNumDofs();
  Eigen::VectorXs spring_stiffs = Eigen::VectorXs::Zero(nDofs);
  for (int i = 0; i < nDofs; i++)
  {
    spring_stiffs(i) = dofs[i]->getSpringStiffness();
  }
  return spring_stiffs;
}

Eigen::VectorXs Skeleton::getRestPositions()
{
  std::vector<dynamics::DegreeOfFreedom*> dofs = getDofs();
  size_t nDofs = getNumDofs();
  Eigen::VectorXs rest_pose = Eigen::VectorXs::Zero(nDofs);
  for (int i = 0; i < nDofs; i++)
  {
    rest_pose(i) = dofs[i]->getRestPosition();
  }
  return rest_pose;
}

Eigen::VectorXs Skeleton::getSpringForce()
{
  Eigen::VectorXs spring_stiffs = getSpringStiffVector();
  Eigen::VectorXs rest_pose = getRestPositions();
  Eigen::VectorXs velocities = getVelocities();
  Eigen::VectorXs pose = getPositions();
  s_t dt = getTimeStep();
  Eigen::VectorXs spring_force
      = spring_stiffs.asDiagonal() * (pose - rest_pose + dt * velocities);
  return spring_force;
}
//==============================================================================
const Eigen::VectorXs& Skeleton::getConstraintForces(std::size_t _treeIdx) const
{
  return computeConstraintForces(mTreeCache[_treeIdx]);
}

//==============================================================================
const Eigen::VectorXs& Skeleton::getConstraintForces() const
{
  return computeConstraintForces(mSkelCache);
}

//==============================================================================
// const Eigen::VectorXs& Skeleton::getDampingForceVector() {
//  if (mIsDampingForceVectorDirty)
//    updateDampingForceVector();
//  return mFd;
//}

//==============================================================================
Skeleton::Skeleton(const AspectPropertiesData& properties)
  : mTotalMass(0.0), mIsImpulseApplied(false), mUnionSize(1)
{
  createAspect<Aspect>(properties);
  createAspect<detail::BodyNodeVectorProxyAspect>();
  createAspect<detail::JointVectorProxyAspect>();
}

//==============================================================================
void Skeleton::setPtr(const SkeletonPtr& _ptr)
{
  mPtr = _ptr;
  resetUnion();
}

//==============================================================================
void Skeleton::constructNewTree()
{
  mTreeCache.push_back(DataCache());

  mTreeNodeMaps.push_back(NodeMap());
  NodeMap& nodeMap = mTreeNodeMaps.back();

  // Create the machinery needed to directly call on specialized node types
  for (auto& nodeType : mSpecializedTreeNodes)
  {
    const std::type_index& index = nodeType.first;
    nodeMap[index] = std::vector<Node*>();

    std::vector<NodeMap::iterator>* nodeVec = nodeType.second;
    nodeVec->push_back(nodeMap.find(index));

    assert(nodeVec->size() == mTreeCache.size());
  }
}

//==============================================================================
void Skeleton::registerBodyNode(BodyNode* _newBodyNode)
{
#ifndef NDEBUG // Debug mode
  std::vector<BodyNode*>::iterator repeat = std::find(
      mSkelCache.mBodyNodes.begin(), mSkelCache.mBodyNodes.end(), _newBodyNode);
  if (repeat != mSkelCache.mBodyNodes.end())
  {
    dterr << "[Skeleton::registerBodyNode] Attempting to s_t-register the "
          << "BodyNode named [" << _newBodyNode->getName() << "] in the "
          << "Skeleton named [" << getName() << "]. Please report this as a "
          << "bug!\n";
    assert(false);
    return;
  }
#endif // -------- Debug mode

  mSkelCache.mBodyNodes.push_back(_newBodyNode);
  if (nullptr == _newBodyNode->getParentBodyNode())
  {
    // Create a new tree and add the new BodyNode to it
    _newBodyNode->mIndexInTree = 0;
    constructNewTree();
    mTreeCache.back().mBodyNodes.push_back(_newBodyNode);
    _newBodyNode->mTreeIndex = mTreeCache.size() - 1;
  }
  else
  {
    std::size_t tree = _newBodyNode->getParentBodyNode()->getTreeIndex();
    _newBodyNode->mTreeIndex = tree;
    DataCache& cache = mTreeCache[tree];
    cache.mBodyNodes.push_back(_newBodyNode);
    _newBodyNode->mIndexInTree = cache.mBodyNodes.size() - 1;
  }

  _newBodyNode->mSkeleton = getPtr();
  _newBodyNode->mIndexInSkeleton = mSkelCache.mBodyNodes.size() - 1;
  addEntryToBodyNodeNameMgr(_newBodyNode);
  registerJoint(_newBodyNode->getParentJoint());

  SoftBodyNode* softBodyNode = dynamic_cast<SoftBodyNode*>(_newBodyNode);
  if (softBodyNode)
  {
    mSoftBodyNodes.push_back(softBodyNode);
    addEntryToSoftBodyNodeNameMgr(softBodyNode);
  }

  _newBodyNode->init(getPtr());

  BodyNode::NodeMap& nodeMap = _newBodyNode->mNodeMap;
  for (auto& nodeType : nodeMap)
    for (auto& node : nodeType.second)
      registerNode(node);

  updateTotalMass();
  updateCacheDimensions(_newBodyNode->mTreeIndex);

#ifndef NDEBUG // Debug mode
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    if (mSkelCache.mBodyNodes[i]->mIndexInSkeleton != i)
    {
      dterr << "[Skeleton::registerBodyNode] BodyNode named ["
            << mSkelCache.mBodyNodes[i]->getName() << "] in Skeleton ["
            << getName() << "] is mistaken about its index in the Skeleton ( "
            << i << " : " << mSkelCache.mBodyNodes[i]->mIndexInSkeleton
            << "). Please report this as a bug!\n";
      assert(false);
    }
  }

  for (std::size_t i = 0; i < mTreeCache.size(); ++i)
  {
    const DataCache& cache = mTreeCache[i];
    for (std::size_t j = 0; j < cache.mBodyNodes.size(); ++j)
    {
      BodyNode* bn = cache.mBodyNodes[j];
      if (bn->mTreeIndex != i)
      {
        dterr << "[Skeleton::registerBodyNode] BodyNode named ["
              << bn->getName() << "] in Skeleton [" << getName() << "] is "
              << "mistaken about its tree's index (" << i << " : "
              << bn->mTreeIndex << "). Please report this as a bug!\n";
        assert(false);
      }

      if (bn->mIndexInTree != j)
      {
        dterr << "[Skeleton::registerBodyNode] BodyNode named ["
              << bn->getName() << "] in Skeleton [" << getName() << "] is "
              << "mistaken about its index in the tree (" << j << " : "
              << bn->mIndexInTree << "). Please report this as a bug!\n";
        assert(false);
      }
    }
  }
#endif // ------- Debug mode

  _newBodyNode->mStructuralChangeSignal.raise(_newBodyNode);
}

//==============================================================================
void Skeleton::registerJoint(Joint* _newJoint)
{
  if (nullptr == _newJoint)
  {
    dterr << "[Skeleton::registerJoint] Error: Attempting to add a nullptr "
             "Joint to the Skeleton named ["
          << mAspectProperties.mName
          << "]. Report "
             "this as a bug!\n";
    assert(false);
    return;
  }

  addEntryToJointNameMgr(_newJoint);
  _newJoint->registerDofs();

  std::size_t tree = _newJoint->getChildBodyNode()->getTreeIndex();
  std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
  for (std::size_t i = 0; i < _newJoint->getNumDofs(); ++i)
  {
    mSkelCache.mDofs.push_back(_newJoint->getDof(i));
    _newJoint->getDof(i)->mIndexInSkeleton = mSkelCache.mDofs.size() - 1;

    treeDofs.push_back(_newJoint->getDof(i));
    _newJoint->getDof(i)->mIndexInTree = treeDofs.size() - 1;
  }
}

//==============================================================================
void Skeleton::registerNode(
    NodeMap& nodeMap, Node* _newNode, std::size_t& _index)
{
  NodeMap::iterator it = nodeMap.find(typeid(*_newNode));

  if (nodeMap.end() == it)
  {
    nodeMap[typeid(*_newNode)] = std::vector<Node*>();
    it = nodeMap.find(typeid(*_newNode));
  }

  std::vector<Node*>& nodes = it->second;

  if (INVALID_INDEX == _index)
  {
    // If this Node believes its index is invalid, then it should not exist
    // anywhere in the vector
    assert(std::find(nodes.begin(), nodes.end(), _newNode) == nodes.end());

    nodes.push_back(_newNode);
    _index = nodes.size() - 1;
  }

  assert(std::find(nodes.begin(), nodes.end(), _newNode) != nodes.end());
}

//==============================================================================
void Skeleton::registerNode(Node* _newNode)
{
  registerNode(mNodeMap, _newNode, _newNode->mIndexInSkeleton);

  registerNode(
      mTreeNodeMaps[_newNode->getBodyNodePtr()->getTreeIndex()],
      _newNode,
      _newNode->mIndexInTree);

  const std::type_info& info = typeid(*_newNode);
  NodeNameMgrMap::iterator it = mNodeNameMgrMap.find(info);
  if (mNodeNameMgrMap.end() == it)
  {
    mNodeNameMgrMap[info] = common::NameManager<Node*>(
        std::string("Skeleton::") + info.name() + " | "
            + mAspectProperties.mName,
        info.name());

    it = mNodeNameMgrMap.find(info);
  }

  common::NameManager<Node*>& mgr = it->second;
  _newNode->setName(mgr.issueNewNameAndAdd(_newNode->getName(), _newNode));
}

//==============================================================================
void Skeleton::destructOldTree(std::size_t tree)
{
  mTreeCache.erase(mTreeCache.begin() + tree);
  mTreeNodeMaps.erase(mTreeNodeMaps.begin() + tree);

  // Decrease the tree index of every BodyNode whose tree index is higher than
  // the one which is being removed. None of the BodyNodes that predate the
  // current one can have a higher tree index, so they can be ignored.
  for (std::size_t i = tree; i < mTreeCache.size(); ++i)
  {
    DataCache& loweredTree = mTreeCache[i];
    for (std::size_t j = 0; j < loweredTree.mBodyNodes.size(); ++j)
      loweredTree.mBodyNodes[j]->mTreeIndex = i;
  }

  for (auto& nodeType : mSpecializedTreeNodes)
  {
    std::vector<NodeMap::iterator>* nodeRepo = nodeType.second;
    nodeRepo->erase(nodeRepo->begin() + tree);
  }
}

//==============================================================================
void Skeleton::unregisterBodyNode(BodyNode* _oldBodyNode)
{
  unregisterJoint(_oldBodyNode->getParentJoint());

  BodyNode::NodeMap& nodeMap = _oldBodyNode->mNodeMap;
  for (auto& nodeType : nodeMap)
    for (auto& node : nodeType.second)
      unregisterNode(node);

  mNameMgrForBodyNodes.removeName(_oldBodyNode->getName());

  std::size_t index = _oldBodyNode->getIndexInSkeleton();
  assert(mSkelCache.mBodyNodes[index] == _oldBodyNode);
  mSkelCache.mBodyNodes.erase(mSkelCache.mBodyNodes.begin() + index);
  for (std::size_t i = index; i < mSkelCache.mBodyNodes.size(); ++i)
  {
    BodyNode* bn = mSkelCache.mBodyNodes[i];
    bn->mIndexInSkeleton = i;
  }

  if (nullptr == _oldBodyNode->getParentBodyNode())
  {
    // If the parent of this BodyNode is a nullptr, then this is the root of its
    // tree. If the root of the tree is being removed, then the tree itself
    // should be destroyed.

    // There is no way that any child BodyNodes of this root BodyNode are still
    // registered, because the BodyNodes always get unregistered from leaf to
    // root.

    std::size_t tree = _oldBodyNode->getTreeIndex();
    assert(mTreeCache[tree].mBodyNodes.size() == 1);
    assert(mTreeCache[tree].mBodyNodes[0] == _oldBodyNode);

    destructOldTree(tree);
    updateCacheDimensions(mSkelCache);
  }
  else
  {
    std::size_t tree = _oldBodyNode->getTreeIndex();
    std::size_t indexInTree = _oldBodyNode->getIndexInTree();
    assert(mTreeCache[tree].mBodyNodes[indexInTree] == _oldBodyNode);
    mTreeCache[tree].mBodyNodes.erase(
        mTreeCache[tree].mBodyNodes.begin() + indexInTree);

    for (std::size_t i = indexInTree; i < mTreeCache[tree].mBodyNodes.size();
         ++i)
      mTreeCache[tree].mBodyNodes[i]->mIndexInTree = i;

    updateCacheDimensions(tree);
  }

  SoftBodyNode* soft = dynamic_cast<SoftBodyNode*>(_oldBodyNode);
  if (soft)
  {
    mNameMgrForSoftBodyNodes.removeName(soft->getName());

    mSoftBodyNodes.erase(
        std::remove(mSoftBodyNodes.begin(), mSoftBodyNodes.end(), soft),
        mSoftBodyNodes.end());
  }

  updateTotalMass();
}

//==============================================================================
void Skeleton::unregisterJoint(Joint* _oldJoint)
{
  if (nullptr == _oldJoint)
  {
    dterr << "[Skeleton::unregisterJoint] Attempting to unregister nullptr "
          << "Joint from Skeleton named [" << getName() << "]. Report this as "
          << "a bug!\n";
    assert(false);
    return;
  }

  mNameMgrForJoints.removeName(_oldJoint->getName());

  std::size_t tree = _oldJoint->getChildBodyNode()->getTreeIndex();
  std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
  std::vector<DegreeOfFreedom*>& skelDofs = mSkelCache.mDofs;

  std::size_t firstSkelIndex = INVALID_INDEX;
  std::size_t firstTreeIndex = INVALID_INDEX;
  for (std::size_t i = 0; i < _oldJoint->getNumDofs(); ++i)
  {
    DegreeOfFreedom* dof = _oldJoint->getDof(i);
    mNameMgrForDofs.removeObject(dof);

    firstSkelIndex = std::min(firstSkelIndex, dof->getIndexInSkeleton());
    skelDofs.erase(
        std::remove(skelDofs.begin(), skelDofs.end(), dof), skelDofs.end());

    firstTreeIndex = std::min(firstTreeIndex, dof->getIndexInTree());
    treeDofs.erase(
        std::remove(treeDofs.begin(), treeDofs.end(), dof), treeDofs.end());
  }

  for (std::size_t i = firstSkelIndex; i < skelDofs.size(); ++i)
  {
    DegreeOfFreedom* dof = skelDofs[i];
    dof->mIndexInSkeleton = i;
  }

  for (std::size_t i = firstTreeIndex; i < treeDofs.size(); ++i)
  {
    DegreeOfFreedom* dof = treeDofs[i];
    dof->mIndexInTree = i;
  }
}

//==============================================================================
void Skeleton::unregisterNode(
    NodeMap& nodeMap, Node* _oldNode, std::size_t& _index)
{
  NodeMap::iterator it = nodeMap.find(typeid(*_oldNode));

  if (nodeMap.end() == it)
  {
    // If the Node was not in the map, then its index should be invalid
    assert(INVALID_INDEX == _index);
    return;
  }

  std::vector<Node*>& nodes = it->second;

  // This Node's index in the vector should be referring to this Node
  assert(nodes[_index] == _oldNode);
  nodes.erase(nodes.begin() + _index);

  _index = INVALID_INDEX;
}

//==============================================================================
void Skeleton::unregisterNode(Node* _oldNode)
{
  const std::size_t indexInSkel = _oldNode->mIndexInSkeleton;
  unregisterNode(mNodeMap, _oldNode, _oldNode->mIndexInSkeleton);

  NodeMap::iterator node_it = mNodeMap.find(typeid(*_oldNode));
  assert(mNodeMap.end() != node_it);

  const std::vector<Node*>& skelNodes = node_it->second;
  for (std::size_t i = indexInSkel; i < skelNodes.size(); ++i)
    skelNodes[i]->mIndexInSkeleton = i;

  const std::size_t indexInTree = _oldNode->mIndexInTree;
  const std::size_t treeIndex = _oldNode->getBodyNodePtr()->getTreeIndex();
  NodeMap& treeNodeMap = mTreeNodeMaps[treeIndex];
  unregisterNode(treeNodeMap, _oldNode, _oldNode->mIndexInTree);

  node_it = treeNodeMap.find(typeid(*_oldNode));
  assert(treeNodeMap.end() != node_it);

  const std::vector<Node*>& treeNodes = node_it->second;
  for (std::size_t i = indexInTree; i < treeNodes.size(); ++i)
    treeNodes[i]->mIndexInTree = i;

  // Remove it from the NameManager, if a NameManager is being used for this
  // type.
  NodeNameMgrMap::iterator name_it = mNodeNameMgrMap.find(typeid(*_oldNode));
  if (mNodeNameMgrMap.end() != name_it)
  {
    common::NameManager<Node*>& mgr = name_it->second;
    mgr.removeObject(_oldNode);
  }
}

//==============================================================================
bool Skeleton::moveBodyNodeTree(
    Joint* _parentJoint,
    BodyNode* _bodyNode,
    SkeletonPtr _newSkeleton,
    BodyNode* _parentNode)
{
  if (nullptr == _bodyNode)
  {
    dterr << "[Skeleton::moveBodyNodeTree] Skeleton named [" << getName()
          << "] (" << this << ") is attempting to move a nullptr BodyNode. "
          << "Please report this as a bug!\n";
    assert(false);
    return false;
  }

  if (this != _bodyNode->getSkeleton().get())
  {
    dterr << "[Skeleton::moveBodyNodeTree] Skeleton named [" << getName()
          << "] (" << this << ") is attempting to move a BodyNode named ["
          << _bodyNode->getName() << "] even though it belongs to another "
          << "Skeleton [" << _bodyNode->getSkeleton()->getName() << "] ("
          << _bodyNode->getSkeleton() << "). Please report this as a bug!\n";
    assert(false);
    return false;
  }

  if ((nullptr == _parentJoint)
      && (_bodyNode->getParentBodyNode() == _parentNode)
      && (this == _newSkeleton.get()))
  {
    // Short-circuit if the BodyNode is already in the requested place, and its
    // Joint does not need to be changed
    return false;
  }

  if (_bodyNode == _parentNode)
  {
    dterr << "[Skeleton::moveBodyNodeTree] Attempting to move BodyNode named ["
          << _bodyNode->getName() << "] (" << _bodyNode << ") to be its own "
          << "parent. This is not permitted!\n";
    return false;
  }

  if (_parentNode && _parentNode->descendsFrom(_bodyNode))
  {
    dterr << "[Skeleton::moveBodyNodeTree] Attempting to move BodyNode named ["
          << _bodyNode->getName() << "] of Skeleton [" << getName() << "] ("
          << this << ") to be a child of BodyNode [" << _parentNode->getName()
          << "] in Skeleton [" << _newSkeleton->getName() << "] ("
          << _newSkeleton << "), but that would create a closed kinematic "
          << "chain, which is not permitted! Nothing will be moved.\n";
    return false;
  }

  if (nullptr == _newSkeleton)
  {
    if (nullptr == _parentNode)
    {
      dterr << "[Skeleton::moveBodyNodeTree] Attempting to move a BodyNode "
            << "tree starting from [" << _bodyNode->getName() << "] in "
            << "Skeleton [" << getName() << "] into a nullptr Skeleton. This "
            << "is not permitted!\n";
      return false;
    }

    _newSkeleton = _parentNode->getSkeleton();
  }

  if (_parentNode && _newSkeleton != _parentNode->getSkeleton())
  {
    dterr << "[Skeleton::moveBodyNodeTree] Mismatch between the specified "
          << "Skeleton [" << _newSkeleton->getName() << "] (" << _newSkeleton
          << ") and the specified new parent BodyNode ["
          << _parentNode->getName() << "] whose actual Skeleton is named ["
          << _parentNode->getSkeleton()->getName() << "] ("
          << _parentNode->getSkeleton() << ") while attempting to move a "
          << "BodyNode tree starting from [" << _bodyNode->getName() << "] in "
          << "Skeleton [" << getName() << "] (" << this << ")\n";
    return false;
  }

  std::vector<BodyNode*> tree = extractBodyNodeTree(_bodyNode);

  Joint* originalParent = _bodyNode->getParentJoint();
  if (originalParent != _parentJoint)
  {
    _bodyNode->mParentJoint = _parentJoint;
    _parentJoint->mChildBodyNode = _bodyNode;
    delete originalParent;
  }

  if (_parentNode != _bodyNode->getParentBodyNode())
  {
    _bodyNode->mParentBodyNode = _parentNode;
    if (_parentNode)
    {
      _parentNode->mChildBodyNodes.push_back(_bodyNode);
      _bodyNode->changeParentFrame(_parentNode);
    }
    else
    {
      _bodyNode->changeParentFrame(Frame::World());
    }
  }
  _newSkeleton->receiveBodyNodeTree(tree);

  return true;
}

//==============================================================================
std::pair<Joint*, BodyNode*> Skeleton::cloneBodyNodeTree(
    Joint* _parentJoint,
    const BodyNode* _bodyNode,
    const SkeletonPtr& _newSkeleton,
    BodyNode* _parentNode,
    bool _recursive) const
{
  std::pair<Joint*, BodyNode*> root(nullptr, nullptr);
  std::vector<const BodyNode*> tree;
  if (_recursive)
    tree = constructBodyNodeTree(_bodyNode);
  else
    tree.push_back(_bodyNode);

  std::map<std::string, BodyNode*> nameMap;
  std::vector<BodyNode*> clones;
  clones.reserve(tree.size());

  for (std::size_t i = 0; i < tree.size(); ++i)
  {
    const BodyNode* original = tree[i];
    // If this is the root of the tree, and the user has requested a change in
    // its parent Joint, use the specified parent Joint instead of created a
    // clone
    Joint* joint;
    if (i == 0 && _parentJoint != nullptr)
    {
      joint = _parentJoint;
    }
    else
    {
      joint = original->getParentJoint()->clone();
      joint->copyTransformsFrom(original->getParentJoint());
    }

    BodyNode* newParent
        = i == 0 ? _parentNode
                 : nameMap[original->getParentBodyNode()->getName()];

    BodyNode* clone = original->clone(newParent, joint, true);
    clones.push_back(clone);
    nameMap[clone->getName()] = clone;

    if (0 == i)
    {
      root.first = joint;
      root.second = clone;
    }
  }

  _newSkeleton->receiveBodyNodeTree(clones);
  return root;
}

//==============================================================================
template <typename BodyNodeT>
static void recursiveConstructBodyNodeTree(
    std::vector<BodyNodeT*>& tree, BodyNodeT* _currentBodyNode)
{
  tree.push_back(_currentBodyNode);
  for (std::size_t i = 0; i < _currentBodyNode->getNumChildBodyNodes(); ++i)
    recursiveConstructBodyNodeTree(tree, _currentBodyNode->getChildBodyNode(i));
}

//==============================================================================
std::vector<const BodyNode*> Skeleton::constructBodyNodeTree(
    const BodyNode* _bodyNode) const
{
  std::vector<const BodyNode*> tree;
  recursiveConstructBodyNodeTree<const BodyNode>(tree, _bodyNode);

  return tree;
}

//==============================================================================
std::vector<BodyNode*> Skeleton::constructBodyNodeTree(BodyNode* _bodyNode)
{
  std::vector<BodyNode*> tree;
  recursiveConstructBodyNodeTree<BodyNode>(tree, _bodyNode);

  return tree;
}

//==============================================================================
std::vector<BodyNode*> Skeleton::extractBodyNodeTree(BodyNode* _bodyNode)
{
  std::vector<BodyNode*> tree = constructBodyNodeTree(_bodyNode);

  // Go backwards to minimize the number of shifts needed
  std::vector<BodyNode*>::reverse_iterator rit;
  // Go backwards to minimize the amount of element shifting in the vectors
  for (rit = tree.rbegin(); rit != tree.rend(); ++rit)
    unregisterBodyNode(*rit);

  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    mSkelCache.mBodyNodes[i]->init(getPtr());

  return tree;
}

//==============================================================================
void Skeleton::receiveBodyNodeTree(const std::vector<BodyNode*>& _tree)
{
  for (BodyNode* bn : _tree)
    registerBodyNode(bn);
}

//==============================================================================
void Skeleton::updateTotalMass()
{
  mTotalMass = 0.0;
  for (std::size_t i = 0; i < getNumBodyNodes(); ++i)
    mTotalMass += getBodyNode(i)->getMass();
}

//==============================================================================
void Skeleton::updateCacheDimensions(Skeleton::DataCache& _cache)
{
  std::size_t dof = _cache.mDofs.size();
  _cache.mM = Eigen::MatrixXs::Zero(dof, dof);
  _cache.mAugM = Eigen::MatrixXs::Zero(dof, dof);
  _cache.mInvM = Eigen::MatrixXs::Zero(dof, dof);
  _cache.mInvAugM = Eigen::MatrixXs::Zero(dof, dof);
  _cache.mCvec = Eigen::VectorXs::Zero(dof);
  _cache.mG = Eigen::VectorXs::Zero(dof);
  _cache.mCg = Eigen::VectorXs::Zero(dof);
  _cache.mFext = Eigen::VectorXs::Zero(dof);
  _cache.mFc = Eigen::VectorXs::Zero(dof);
}

//==============================================================================
void Skeleton::updateCacheDimensions(std::size_t _treeIdx)
{
  updateCacheDimensions(mTreeCache[_treeIdx]);
  updateCacheDimensions(mSkelCache);

  dirtyArticulatedInertia(_treeIdx);
}

//==============================================================================
void Skeleton::updateArticulatedInertia(std::size_t _tree) const
{
  DataCache& cache = mTreeCache[_tree];
  for (std::vector<BodyNode*>::const_reverse_iterator it
       = cache.mBodyNodes.rbegin();
       it != cache.mBodyNodes.rend();
       ++it)
  {
    (*it)->updateArtInertia(mAspectProperties.mTimeStep);
  }

  cache.mDirty.mArticulatedInertia = false;
}

//==============================================================================
void Skeleton::updateArticulatedInertia() const
{
  for (std::size_t i = 0; i < mTreeCache.size(); ++i)
  {
    DataCache& cache = mTreeCache[i];
    if (cache.mDirty.mArticulatedInertia)
      updateArticulatedInertia(i);
  }

  mSkelCache.mDirty.mArticulatedInertia = false;
}

//==============================================================================
void Skeleton::updateMassMatrix(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(
      static_cast<std::size_t>(cache.mM.cols()) == dof
      && static_cast<std::size_t>(cache.mM.rows()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mMassMatrix = false;
    return;
  }

  cache.mM.setZero();

  // Backup the original internal force
  Eigen::VectorXs originalGenAcceleration = getAccelerations();

  // Clear out the accelerations of the dofs in this tree so that we can set
  // them to 1.0 one at a time to build up the mass matrix
  for (std::size_t i = 0; i < dof; ++i)
    cache.mDofs[i]->setAcceleration(0.0);

  for (std::size_t j = 0; j < dof; ++j)
  {
    // Set the acceleration of this DOF to 1.0 while all the rest are 0.0
    cache.mDofs[j]->setAcceleration(1.0);

    // Prepare cache data
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateMassMatrix();
    }

    // Mass matrix
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->aggregateMassMatrix(cache.mM, j);
      std::size_t localDof = (*it)->mParentJoint->getNumDofs();
      if (localDof > 0)
      {
        std::size_t iStart = (*it)->mParentJoint->getIndexInTree(0);

        if (iStart + localDof < j)
          break;
      }
    }

    // Set the acceleration of this DOF back to 0.0
    cache.mDofs[j]->setAcceleration(0.0);
  }
  cache.mM.triangularView<Eigen::StrictlyUpper>() = cache.mM.transpose();

  // Restore the original generalized accelerations
  const_cast<Skeleton*>(this)->setAccelerations(originalGenAcceleration);

  cache.mDirty.mMassMatrix = false;
}

//==============================================================================
void Skeleton::updateMassMatrix() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(
      static_cast<std::size_t>(mSkelCache.mM.cols()) == dof
      && static_cast<std::size_t>(mSkelCache.mM.rows()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mMassMatrix = false;
    return;
  }

  mSkelCache.mM.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::MatrixXs& treeM = getMassMatrix(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      for (std::size_t j = 0; j < nTreeDofs; ++j)
      {
        std::size_t ki = treeDofs[i]->getIndexInSkeleton();
        std::size_t kj = treeDofs[j]->getIndexInSkeleton();

        mSkelCache.mM(ki, kj) = treeM(i, j);
      }
    }
  }

  mSkelCache.mDirty.mMassMatrix = false;
}

//==============================================================================
void Skeleton::updateAugMassMatrix(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(
      static_cast<std::size_t>(cache.mAugM.cols()) == dof
      && static_cast<std::size_t>(cache.mAugM.rows()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mAugMassMatrix = false;
    return;
  }

  cache.mAugM.setZero();

  // Backup the origianl internal force
  Eigen::VectorXs originalGenAcceleration = getAccelerations();

  // Clear out the accelerations of the DOFs in this tree so that we can set
  // them to 1.0 one at a time to build up the augmented mass matrix
  for (std::size_t i = 0; i < dof; ++i)
    cache.mDofs[i]->setAcceleration(0.0);

  for (std::size_t j = 0; j < dof; ++j)
  {
    // Set the acceleration of this DOF to 1.0 while all the rest are 0.0
    cache.mDofs[j]->setAcceleration(1.0);

    // Prepare cache data
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateMassMatrix();
    }

    // Augmented Mass matrix
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->aggregateAugMassMatrix(
          cache.mAugM, j, mAspectProperties.mTimeStep);
      std::size_t localDof = (*it)->mParentJoint->getNumDofs();
      if (localDof > 0)
      {
        std::size_t iStart = (*it)->mParentJoint->getIndexInTree(0);

        if (iStart + localDof < j)
          break;
      }
    }

    // Set the acceleration of this DOF back to 0.0
    cache.mDofs[j]->setAcceleration(0.0);
  }
  cache.mAugM.triangularView<Eigen::StrictlyUpper>() = cache.mAugM.transpose();

  // Restore the origianl internal force
  const_cast<Skeleton*>(this)->setAccelerations(originalGenAcceleration);

  cache.mDirty.mAugMassMatrix = false;
}

//==============================================================================
void Skeleton::updateAugMassMatrix() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(
      static_cast<std::size_t>(mSkelCache.mAugM.cols()) == dof
      && static_cast<std::size_t>(mSkelCache.mAugM.rows()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mMassMatrix = false;
    return;
  }

  mSkelCache.mAugM.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::MatrixXs& treeAugM = getAugMassMatrix(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      for (std::size_t j = 0; j < nTreeDofs; ++j)
      {
        std::size_t ki = treeDofs[i]->getIndexInSkeleton();
        std::size_t kj = treeDofs[j]->getIndexInSkeleton();

        mSkelCache.mAugM(ki, kj) = treeAugM(i, j);
      }
    }
  }

  mSkelCache.mDirty.mAugMassMatrix = false;
}

//==============================================================================
void Skeleton::updateInvMassMatrix(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(
      static_cast<std::size_t>(cache.mInvM.cols()) == dof
      && static_cast<std::size_t>(cache.mInvM.rows()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mInvMassMatrix = false;
    return;
  }

  // We don't need to set mInvM as zero matrix as long as the below is correct
  // cache.mInvM.setZero();

  // Backup the origianl internal force
  Eigen::VectorXs originalInternalForce = getControlForces();

  // Clear out the forces of the dofs in this tree so that we can set them to
  // 1.0 one at a time to build up the inverse mass matrix
  for (std::size_t i = 0; i < dof; ++i)
    cache.mDofs[i]->setControlForce(0.0);

  for (std::size_t j = 0; j < dof; ++j)
  {
    // Set the force of this DOF to 1.0 while all the rest are 0.0
    cache.mDofs[j]->setControlForce(1.0);

    // Prepare cache data
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->updateInvMassMatrix();
    }

    // Inverse of mass matrix
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->aggregateInvMassMatrix(cache.mInvM, j);
      std::size_t localDof = (*it)->mParentJoint->getNumDofs();
      if (localDof > 0)
      {
        std::size_t iStart = (*it)->mParentJoint->getIndexInTree(0);

        if (iStart + localDof > j)
          break;
      }
    }

    // Set the force of this DOF back to 0.0
    cache.mDofs[j]->setControlForce(0.0);
  }
  cache.mInvM.triangularView<Eigen::StrictlyLower>() = cache.mInvM.transpose();

  // Restore the original internal force
  const_cast<Skeleton*>(this)->setControlForces(originalInternalForce);

  cache.mDirty.mInvMassMatrix = false;
}

//==============================================================================
void Skeleton::updateInvMassMatrix() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(
      static_cast<std::size_t>(mSkelCache.mInvM.cols()) == dof
      && static_cast<std::size_t>(mSkelCache.mInvM.rows()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mInvMassMatrix = false;
    return;
  }

  mSkelCache.mInvM.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::MatrixXs& treeInvM = getInvMassMatrix(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      for (std::size_t j = 0; j < nTreeDofs; ++j)
      {
        std::size_t ki = treeDofs[i]->getIndexInSkeleton();
        std::size_t kj = treeDofs[j]->getIndexInSkeleton();

        mSkelCache.mInvM(ki, kj) = treeInvM(i, j);
      }
    }
  }

  mSkelCache.mDirty.mInvMassMatrix = false;
}

//==============================================================================
void Skeleton::updateInvAugMassMatrix(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(
      static_cast<std::size_t>(cache.mInvAugM.cols()) == dof
      && static_cast<std::size_t>(cache.mInvAugM.rows()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mInvAugMassMatrix = false;
    return;
  }

  // We don't need to set mInvM as zero matrix as long as the below is correct
  // mInvM.setZero();

  // Backup the origianl internal force
  Eigen::VectorXs originalInternalForce = getControlForces();

  // Clear out the forces of the dofs in this tree so that we can set them to
  // 1.0 one at a time to build up the inverse augmented mass matrix
  for (std::size_t i = 0; i < dof; ++i)
    cache.mDofs[i]->setControlForce(0.0);

  for (std::size_t j = 0; j < dof; ++j)
  {
    // Set the force of this DOF to 1.0 while all the rest are 0.0
    cache.mDofs[j]->setControlForce(1.0);

    // Prepare cache data
    for (std::vector<BodyNode*>::const_reverse_iterator it
         = cache.mBodyNodes.rbegin();
         it != cache.mBodyNodes.rend();
         ++it)
    {
      (*it)->updateInvAugMassMatrix();
    }

    // Inverse of augmented mass matrix
    for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
         it != cache.mBodyNodes.end();
         ++it)
    {
      (*it)->aggregateInvAugMassMatrix(
          cache.mInvAugM, j, mAspectProperties.mTimeStep);
      std::size_t localDof = (*it)->mParentJoint->getNumDofs();
      if (localDof > 0)
      {
        std::size_t iStart = (*it)->mParentJoint->getIndexInTree(0);

        if (iStart + localDof > j)
          break;
      }
    }

    // Set the force of this DOF back to 0.0
    cache.mDofs[j]->setControlForce(0.0);
  }
  cache.mInvAugM.triangularView<Eigen::StrictlyLower>()
      = cache.mInvAugM.transpose();

  // Restore the original internal force
  const_cast<Skeleton*>(this)->setControlForces(originalInternalForce);

  cache.mDirty.mInvAugMassMatrix = false;
}

//==============================================================================
void Skeleton::updateInvAugMassMatrix() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(
      static_cast<std::size_t>(mSkelCache.mInvAugM.cols()) == dof
      && static_cast<std::size_t>(mSkelCache.mInvAugM.rows()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mInvAugMassMatrix = false;
    return;
  }

  mSkelCache.mInvAugM.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::MatrixXs& treeInvAugM = getInvAugMassMatrix(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      for (std::size_t j = 0; j < nTreeDofs; ++j)
      {
        std::size_t ki = treeDofs[i]->getIndexInSkeleton();
        std::size_t kj = treeDofs[j]->getIndexInSkeleton();

        mSkelCache.mInvAugM(ki, kj) = treeInvAugM(i, j);
      }
    }
  }

  mSkelCache.mDirty.mInvAugMassMatrix = false;
}

//==============================================================================
void Skeleton::updateCoriolisForces(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(static_cast<std::size_t>(cache.mCvec.size()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mCoriolisForces = false;
    return;
  }

  cache.mCvec.setZero();

  for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
       it != cache.mBodyNodes.end();
       ++it)
  {
    (*it)->updateCombinedVector();
  }

  for (std::vector<BodyNode*>::const_reverse_iterator it
       = cache.mBodyNodes.rbegin();
       it != cache.mBodyNodes.rend();
       ++it)
  {
    (*it)->aggregateCoriolisForceVector(cache.mCvec);
  }

  cache.mDirty.mCoriolisForces = false;
}

//==============================================================================
void Skeleton::updateCoriolisForces() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(static_cast<std::size_t>(mSkelCache.mCvec.size()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mCoriolisForces = false;
    return;
  }

  mSkelCache.mCvec.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::VectorXs& treeCvec = getCoriolisForces(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t k = treeDofs[i]->getIndexInSkeleton();
      mSkelCache.mCvec[k] = treeCvec[i];
    }
  }

  mSkelCache.mDirty.mCoriolisForces = false;
}

//==============================================================================
void Skeleton::updateGravityForces(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(static_cast<std::size_t>(cache.mG.size()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mGravityForces = false;
    return;
  }

  cache.mG.setZero();

  for (std::vector<BodyNode*>::const_reverse_iterator it
       = cache.mBodyNodes.rbegin();
       it != cache.mBodyNodes.rend();
       ++it)
  {
    (*it)->aggregateGravityForceVector(cache.mG, mAspectProperties.mGravity);
  }

  cache.mDirty.mGravityForces = false;
}

//==============================================================================
void Skeleton::updateGravityForces() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(static_cast<std::size_t>(mSkelCache.mG.size()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mGravityForces = false;
    return;
  }

  mSkelCache.mG.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::VectorXs& treeG = getGravityForces(tree);
    std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t k = treeDofs[i]->getIndexInSkeleton();
      mSkelCache.mG[k] = treeG[i];
    }
  }

  mSkelCache.mDirty.mGravityForces = false;
}

//==============================================================================
void Skeleton::updateCoriolisAndGravityForces(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(static_cast<std::size_t>(cache.mCg.size()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mCoriolisAndGravityForces = false;
    return;
  }

  cache.mCg.setZero();

  for (std::vector<BodyNode*>::const_iterator it = cache.mBodyNodes.begin();
       it != cache.mBodyNodes.end();
       ++it)
  {
    (*it)->updateCombinedVector();
  }

  for (std::vector<BodyNode*>::const_reverse_iterator it
       = cache.mBodyNodes.rbegin();
       it != cache.mBodyNodes.rend();
       ++it)
  {
    (*it)->aggregateCombinedVector(cache.mCg, mAspectProperties.mGravity);
  }

  cache.mDirty.mCoriolisAndGravityForces = false;
}

//==============================================================================
void Skeleton::updateCoriolisAndGravityForces() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(static_cast<std::size_t>(mSkelCache.mCg.size()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mCoriolisAndGravityForces = false;
    return;
  }

  mSkelCache.mCg.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::VectorXs& treeCg = getCoriolisAndGravityForces(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t k = treeDofs[i]->getIndexInSkeleton();
      mSkelCache.mCg[k] = treeCg[i];
    }
  }

  mSkelCache.mDirty.mCoriolisAndGravityForces = false;
}

//==============================================================================
void Skeleton::updateExternalForces(std::size_t _treeIdx) const
{
  DataCache& cache = mTreeCache[_treeIdx];
  std::size_t dof = cache.mDofs.size();
  assert(static_cast<std::size_t>(cache.mFext.size()) == dof);
  if (dof == 0)
  {
    cache.mDirty.mExternalForces = false;
    return;
  }

  // Clear external force.
  cache.mFext.setZero();

  for (std::vector<BodyNode*>::const_reverse_iterator itr
       = cache.mBodyNodes.rbegin();
       itr != cache.mBodyNodes.rend();
       ++itr)
  {
    (*itr)->aggregateExternalForces(cache.mFext);
  }

  // TODO(JS): Not implemented yet
  //  for (std::vector<SoftBodyNode*>::iterator it = mSoftBodyNodes.begin();
  //       it != mSoftBodyNodes.end(); ++it)
  //  {
  //    s_t kv = (*it)->getVertexSpringStiffness();
  //    s_t ke = (*it)->getEdgeSpringStiffness();

  //    for (int i = 0; i < (*it)->getNumPointMasses(); ++i)
  //    {
  //      PointMass* pm = (*it)->getPointMass(i);
  //      int nN = pm->getNumConnectedPointMasses();

  //      // Vertex restoring force
  //      Eigen::Vector3s Fext = -(kv + nN * ke) * pm->getPositions()
  //                             - (mTimeStep * (kv + nN*ke)) *
  //                             pm->getVelocities();

  //      // Edge restoring force
  //      for (int j = 0; j < nN; ++j)
  //      {
  //        Fext += ke * (pm->getConnectedPointMass(j)->getPositions()
  //                      + mTimeStep
  //                        * pm->getConnectedPointMass(j)->getVelocities());
  //      }

  //      // Assign
  //      int iStart = pm->getIndexInSkeleton(0);
  //      mFext.segment<3>(iStart) = Fext;
  //    }
  //  }

  cache.mDirty.mExternalForces = false;
}

//==============================================================================
void Skeleton::updateExternalForces() const
{
  std::size_t dof = mSkelCache.mDofs.size();
  assert(static_cast<std::size_t>(mSkelCache.mFext.size()) == dof);
  if (dof == 0)
  {
    mSkelCache.mDirty.mExternalForces = false;
    return;
  }

  mSkelCache.mFext.setZero();

  for (std::size_t tree = 0; tree < mTreeCache.size(); ++tree)
  {
    const Eigen::VectorXs& treeFext = getExternalForces(tree);
    const std::vector<DegreeOfFreedom*>& treeDofs = mTreeCache[tree].mDofs;
    std::size_t nTreeDofs = treeDofs.size();
    for (std::size_t i = 0; i < nTreeDofs; ++i)
    {
      std::size_t k = treeDofs[i]->getIndexInSkeleton();
      mSkelCache.mFext[k] = treeFext[i];
    }
  }

  mSkelCache.mDirty.mExternalForces = false;
}

//==============================================================================
const Eigen::VectorXs& Skeleton::computeConstraintForces(DataCache& cache) const
{
  const std::size_t dof = cache.mDofs.size();
  assert(static_cast<std::size_t>(cache.mFc.size()) == dof);

  // Body constraint impulses
  for (std::vector<BodyNode*>::reverse_iterator it = cache.mBodyNodes.rbegin();
       it != cache.mBodyNodes.rend();
       ++it)
  {
    (*it)->aggregateSpatialToGeneralized(
        cache.mFc, (*it)->getConstraintImpulse());
  }

  // Joint constraint impulses
  for (std::size_t i = 0; i < dof; ++i)
    cache.mFc[i] += cache.mDofs[i]->getConstraintImpulse();

  // Get force by dividing the impulse by the time step
  cache.mFc = cache.mFc / mAspectProperties.mTimeStep;

  return cache.mFc;
}

//==============================================================================
static void computeSupportPolygon(
    const Skeleton* skel,
    math::SupportPolygon& polygon,
    math::SupportGeometry& geometry,
    std::vector<std::size_t>& ee_indices,
    Eigen::Vector3s& axis1,
    Eigen::Vector3s& axis2,
    Eigen::Vector2s& centroid,
    std::size_t treeIndex)
{
  polygon.clear();
  geometry.clear();
  ee_indices.clear();

  const Eigen::Vector3s& up = -skel->getGravity();
  if (up.norm() == 0.0)
  {
    dtwarn << "[computeSupportPolygon] Requesting support polygon of a "
           << "Skeleton with no gravity. The result will only be an empty "
           << "set!\n";
    axis1.setZero();
    axis2.setZero();
    centroid = Eigen::Vector2s::Constant(std::nan(""));
    return;
  }

  std::vector<std::size_t> originalEE_map;
  originalEE_map.reserve(skel->getNumEndEffectors());
  for (std::size_t i = 0; i < skel->getNumEndEffectors(); ++i)
  {
    const EndEffector* ee = skel->getEndEffector(i);
    if (ee->getSupport() && ee->getSupport()->isActive()
        && (INVALID_INDEX == treeIndex || ee->getTreeIndex() == treeIndex))
    {
      const math::SupportGeometry& eeGeom = ee->getSupport()->getGeometry();
      for (const Eigen::Vector3s& v : eeGeom)
      {
        geometry.push_back(ee->getWorldTransform() * v);
        originalEE_map.push_back(ee->getIndexInSkeleton());
      }
    }
  }

  axis1 = (up - Eigen::Vector3s::UnitX()).norm() > 1e-6
              ? Eigen::Vector3s::UnitX()
              : Eigen::Vector3s::UnitY();

  axis1 = axis1 - up.dot(axis1) * up / up.dot(up);
  axis1.normalize();

  axis2 = up.normalized().cross(axis1);

  std::vector<std::size_t> vertex_indices;
  polygon = math::computeSupportPolgyon(vertex_indices, geometry, axis1, axis2);

  ee_indices.reserve(vertex_indices.size());
  for (std::size_t i = 0; i < vertex_indices.size(); ++i)
    ee_indices[i] = originalEE_map[vertex_indices[i]];

  if (polygon.size() > 0)
    centroid = math::computeCentroidOfHull(polygon);
  else
    centroid = Eigen::Vector2s::Constant(std::nan(""));
}

//==============================================================================
const math::SupportPolygon& Skeleton::getSupportPolygon() const
{
  math::SupportPolygon& polygon = mSkelCache.mSupportPolygon;

  if (!mSkelCache.mDirty.mSupport)
    return polygon;

  computeSupportPolygon(
      this,
      polygon,
      mSkelCache.mSupportGeometry,
      mSkelCache.mSupportIndices,
      mSkelCache.mSupportAxes.first,
      mSkelCache.mSupportAxes.second,
      mSkelCache.mSupportCentroid,
      INVALID_INDEX);

  mSkelCache.mDirty.mSupport = false;
  ++mSkelCache.mDirty.mSupportVersion;
  return polygon;
}

//==============================================================================
const math::SupportPolygon& Skeleton::getSupportPolygon(
    std::size_t _treeIdx) const
{
  math::SupportPolygon& polygon = mTreeCache[_treeIdx].mSupportPolygon;

  if (!mTreeCache[_treeIdx].mDirty.mSupport)
    return polygon;

  computeSupportPolygon(
      this,
      polygon,
      mTreeCache[_treeIdx].mSupportGeometry,
      mTreeCache[_treeIdx].mSupportIndices,
      mTreeCache[_treeIdx].mSupportAxes.first,
      mTreeCache[_treeIdx].mSupportAxes.second,
      mTreeCache[_treeIdx].mSupportCentroid,
      _treeIdx);

  mTreeCache[_treeIdx].mDirty.mSupport = false;
  ++mTreeCache[_treeIdx].mDirty.mSupportVersion;
  return polygon;
}

//==============================================================================
const std::vector<std::size_t>& Skeleton::getSupportIndices() const
{
  getSupportPolygon();
  return mSkelCache.mSupportIndices;
}

//==============================================================================
const std::vector<std::size_t>& Skeleton::getSupportIndices(
    std::size_t _treeIdx) const
{
  getSupportPolygon(_treeIdx);
  return mTreeCache[_treeIdx].mSupportIndices;
}

//==============================================================================
const std::pair<Eigen::Vector3s, Eigen::Vector3s>& Skeleton::getSupportAxes()
    const
{
  getSupportPolygon();
  return mSkelCache.mSupportAxes;
}

//==============================================================================
const std::pair<Eigen::Vector3s, Eigen::Vector3s>& Skeleton::getSupportAxes(
    std::size_t _treeIdx) const
{
  getSupportPolygon(_treeIdx);
  return mTreeCache[_treeIdx].mSupportAxes;
}

//==============================================================================
const Eigen::Vector2s& Skeleton::getSupportCentroid() const
{
  getSupportPolygon();
  return mSkelCache.mSupportCentroid;
}

//==============================================================================
const Eigen::Vector2s& Skeleton::getSupportCentroid(std::size_t _treeIdx) const
{
  getSupportPolygon(_treeIdx);
  return mTreeCache[_treeIdx].mSupportCentroid;
}

//==============================================================================
std::size_t Skeleton::getSupportVersion() const
{
  if (mSkelCache.mDirty.mSupport)
    return mSkelCache.mDirty.mSupportVersion + 1;

  return mSkelCache.mDirty.mSupportVersion;
}

//==============================================================================
std::size_t Skeleton::getSupportVersion(std::size_t _treeIdx) const
{
  if (mTreeCache[_treeIdx].mDirty.mSupport)
    return mTreeCache[_treeIdx].mDirty.mSupportVersion + 1;

  return mTreeCache[_treeIdx].mDirty.mSupportVersion;
}

//==============================================================================
void Skeleton::computeForwardKinematics(
    bool _updateTransforms, bool _updateVels, bool _updateAccs)
{
  if (_updateTransforms)
  {
    for (std::vector<BodyNode*>::iterator it = mSkelCache.mBodyNodes.begin();
         it != mSkelCache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateTransform();
    }
  }

  if (_updateVels)
  {
    for (std::vector<BodyNode*>::iterator it = mSkelCache.mBodyNodes.begin();
         it != mSkelCache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateVelocity();
      (*it)->updatePartialAcceleration();
    }
  }

  if (_updateAccs)
  {
    for (std::vector<BodyNode*>::iterator it = mSkelCache.mBodyNodes.begin();
         it != mSkelCache.mBodyNodes.end();
         ++it)
    {
      (*it)->updateAccelerationID();
    }
  }
}

//==============================================================================
void Skeleton::computeForwardDynamics()
{
  // Note: Articulated Inertias will be updated automatically when
  // getArtInertiaImplicit() is called in BodyNode::updateBiasForce()

  for (auto it = mSkelCache.mBodyNodes.rbegin();
       it != mSkelCache.mBodyNodes.rend();
       ++it)
    (*it)->updateBiasForce(
        mAspectProperties.mGravity, mAspectProperties.mTimeStep);

  // Forward recursion
  for (auto& bodyNode : mSkelCache.mBodyNodes)
  {
    bodyNode->updateAccelerationFD();
    bodyNode->updateTransmittedForceFD();
    bodyNode->updateJointForceFD(mAspectProperties.mTimeStep, true, true);
  }
}

//==============================================================================
void Skeleton::computeInverseDynamics(
    bool _withExternalForces, bool _withDampingForces, bool _withSpringForces)
{
  // Skip immobile or 0-dof skeleton
  if (getNumDofs() == 0)
    return;

  // Backward recursion
  for (auto it = mSkelCache.mBodyNodes.rbegin();
       it != mSkelCache.mBodyNodes.rend();
       ++it)
  {
    (*it)->updateTransmittedForceID(
        mAspectProperties.mGravity, _withExternalForces);
    (*it)->updateJointForceID(
        mAspectProperties.mTimeStep, _withDampingForces, _withSpringForces);
  }
}

//==============================================================================
void Skeleton::clearExternalForces()
{
  for (auto& bodyNode : mSkelCache.mBodyNodes)
    bodyNode->clearExternalForces();
}

//==============================================================================
void Skeleton::clearInternalForces()
{
  for (auto& bodyNode : mSkelCache.mBodyNodes)
    bodyNode->clearInternalForces();
}

//==============================================================================
void Skeleton::notifyArticulatedInertiaUpdate(std::size_t _treeIdx)
{
  dirtyArticulatedInertia(_treeIdx);
}

//==============================================================================
void Skeleton::dirtyArticulatedInertia(std::size_t _treeIdx)
{
  SET_FLAG(_treeIdx, mArticulatedInertia);
  SET_FLAG(_treeIdx, mMassMatrix);
  SET_FLAG(_treeIdx, mAugMassMatrix);
  SET_FLAG(_treeIdx, mInvMassMatrix);
  SET_FLAG(_treeIdx, mInvAugMassMatrix);
  SET_FLAG(_treeIdx, mCoriolisForces);
  SET_FLAG(_treeIdx, mGravityForces);
  SET_FLAG(_treeIdx, mCoriolisAndGravityForces);
}

//==============================================================================
void Skeleton::notifySupportUpdate(std::size_t _treeIdx)
{
  dirtySupportPolygon(_treeIdx);
}

//==============================================================================
void Skeleton::dirtySupportPolygon(std::size_t _treeIdx)
{
  SET_FLAG(_treeIdx, mSupport);
}

//==============================================================================
void Skeleton::clearConstraintImpulses()
{
  for (auto& bodyNode : mSkelCache.mBodyNodes)
    bodyNode->clearConstraintImpulse();
}

//==============================================================================
void Skeleton::updateBiasImpulse(BodyNode* _bodyNode)
{
  if (nullptr == _bodyNode)
  {
    dterr << "[Skeleton::updateBiasImpulse] Passed in a nullptr!\n";
    assert(false);
    return;
  }

  assert(getNumDofs() > 0);

  // This skeleton should contain _bodyNode
  assert(_bodyNode->getSkeleton().get() == this);

#ifndef NDEBUG
  // All the constraint impulse should be zero
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    assert(
        mSkelCache.mBodyNodes[i]->mConstraintImpulse
        == Eigen::Vector6s::Zero());
#endif

  // Prepare cache data
  BodyNode* it = _bodyNode;
  while (it != nullptr)
  {
    it->updateBiasImpulse();
    it = it->getParentBodyNode();
  }
}

//==============================================================================
void Skeleton::updateBiasImpulse(
    BodyNode* _bodyNode, const Eigen::Vector6s& _imp)
{
  if (nullptr == _bodyNode)
  {
    dterr << "[Skeleton::updateBiasImpulse] Passed in a nullptr!\n";
    assert(false);
    return;
  }

  assert(getNumDofs() > 0);

  // This skeleton should contain _bodyNode
  assert(_bodyNode->getSkeleton().get() == this);

#ifndef NDEBUG
  // All the constraint impulse should be zero
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    assert(
        mSkelCache.mBodyNodes[i]->mConstraintImpulse
        == Eigen::Vector6s::Zero());
#endif

  // Set impulse of _bodyNode
  _bodyNode->mConstraintImpulse = _imp;

  // Prepare cache data
  BodyNode* it = _bodyNode;
  while (it != nullptr)
  {
    it->updateBiasImpulse();
    it = it->getParentBodyNode();
  }

  _bodyNode->mConstraintImpulse.setZero();
}

//==============================================================================
void Skeleton::updateBiasImpulse(
    BodyNode* _bodyNode1,
    const Eigen::Vector6s& _imp1,
    BodyNode* _bodyNode2,
    const Eigen::Vector6s& _imp2)
{
  // Assertions
  if (nullptr == _bodyNode1)
  {
    dterr << "[Skeleton::updateBiasImpulse] Passed in nullptr for BodyNode1!\n";
    assert(false);
    return;
  }

  if (nullptr == _bodyNode2)
  {
    dterr << "[Skeleton::updateBiasImpulse] Passed in nullptr for BodyNode2!\n";
    assert(false);
    return;
  }

  assert(getNumDofs() > 0);

  // This skeleton should contain _bodyNode
  assert(_bodyNode1->getSkeleton().get() == this);
  assert(_bodyNode2->getSkeleton().get() == this);

#ifndef NDEBUG
  // All the constraint impulse should be zero
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    assert(
        mSkelCache.mBodyNodes[i]->mConstraintImpulse
        == Eigen::Vector6s::Zero());
#endif

  // Set impulse to _bodyNode
  _bodyNode1->mConstraintImpulse = _imp1;
  _bodyNode2->mConstraintImpulse = _imp2;

  // Find which body is placed later in the list of body nodes in this skeleton
  std::size_t index1 = _bodyNode1->getIndexInSkeleton();
  std::size_t index2 = _bodyNode2->getIndexInSkeleton();

  std::size_t index = std::max(index1, index2);

  // Prepare cache data
  for (int i = index; 0 <= i; --i)
    mSkelCache.mBodyNodes[i]->updateBiasImpulse();

  _bodyNode1->mConstraintImpulse.setZero();
  _bodyNode2->mConstraintImpulse.setZero();
}

//==============================================================================
void Skeleton::updateBiasImpulse(
    SoftBodyNode* _softBodyNode,
    PointMass* _pointMass,
    const Eigen::Vector3s& _imp)
{
  // Assertions
  assert(_softBodyNode != nullptr);
  assert(getNumDofs() > 0);

  // This skeleton should contain _bodyNode
  assert(
      std::find(mSoftBodyNodes.begin(), mSoftBodyNodes.end(), _softBodyNode)
      != mSoftBodyNodes.end());

#ifndef NDEBUG
  // All the constraint impulse should be zero
  for (std::size_t i = 0; i < mSkelCache.mBodyNodes.size(); ++i)
    assert(
        mSkelCache.mBodyNodes[i]->mConstraintImpulse
        == Eigen::Vector6s::Zero());
#endif

  // Set impulse to _bodyNode
  Eigen::Vector3s oldConstraintImpulse = _pointMass->getConstraintImpulses();
  _pointMass->setConstraintImpulse(_imp, true);

  // Prepare cache data
  BodyNode* it = _softBodyNode;
  while (it != nullptr)
  {
    it->updateBiasImpulse();
    it = it->getParentBodyNode();
  }

  // TODO(JS): Do we need to backup and restore the original value?
  _pointMass->setConstraintImpulse(oldConstraintImpulse);
}

//==============================================================================
void Skeleton::updateVelocityChange()
{
  for (auto& bodyNode : mSkelCache.mBodyNodes)
    bodyNode->updateVelocityChangeFD();
}

//==============================================================================
void Skeleton::setImpulseApplied(bool _val)
{
  mIsImpulseApplied = _val;
}

//==============================================================================
bool Skeleton::isImpulseApplied() const
{
  return mIsImpulseApplied;
}

//==============================================================================
void Skeleton::computeImpulseForwardDynamics()
{
  // Skip immobile or 0-dof skeleton
  if (!isMobile() || getNumDofs() == 0)
    return;

  // Note: we do not need to update articulated inertias here, because they will
  // be updated when BodyNode::updateBiasImpulse() calls
  // BodyNode::getArticulatedInertia()

  // Backward recursion
  for (auto it = mSkelCache.mBodyNodes.rbegin();
       it != mSkelCache.mBodyNodes.rend();
       ++it)
    (*it)->updateBiasImpulse();

  // Forward recursion
  for (auto& bodyNode : mSkelCache.mBodyNodes)
  {
    bodyNode->updateVelocityChangeFD();
    bodyNode->updateTransmittedImpulse();
    bodyNode->updateJointImpulseFD();
    bodyNode->updateConstrainedTerms(mAspectProperties.mTimeStep);
  }
}

//==============================================================================
s_t Skeleton::computeKineticEnergy() const
{
  s_t KE = 0.0;

  for (auto* bodyNode : mSkelCache.mBodyNodes)
    KE += bodyNode->computeKineticEnergy();

  assert(KE >= 0.0 && "Kinetic energy should be positive value.");
  return KE;
}

//==============================================================================
s_t Skeleton::computePotentialEnergy() const
{
  s_t PE = 0.0;

  for (auto* bodyNode : mSkelCache.mBodyNodes)
  {
    PE += bodyNode->computePotentialEnergy(mAspectProperties.mGravity);
    PE += bodyNode->getParentJoint()->computePotentialEnergy();
  }

  return PE;
}

//==============================================================================
void Skeleton::clearCollidingBodies()
{
  for (auto i = 0u; i < getNumBodyNodes(); ++i)
  {
    auto bodyNode = getBodyNode(i);
    DART_SUPPRESS_DEPRECATED_BEGIN
    bodyNode->setColliding(false);
    DART_SUPPRESS_DEPRECATED_END

    auto softBodyNode = bodyNode->asSoftBodyNode();
    if (softBodyNode)
    {
      auto& pointMasses = softBodyNode->getPointMasses();

      for (auto pointMass : pointMasses)
        pointMass->setColliding(false);
    }
  }
}

//==============================================================================
Eigen::Vector3s Skeleton::getCOM(const Frame* _withRespectTo) const
{
  Eigen::Vector3s com = Eigen::Vector3s::Zero();

  const std::size_t numBodies = getNumBodyNodes();
  for (std::size_t i = 0; i < numBodies; ++i)
  {
    const BodyNode* bodyNode = getBodyNode(i);
    com += bodyNode->getMass() * bodyNode->getCOM(_withRespectTo);
  }

  assert(mTotalMass != 0.0);
  return com / mTotalMass;
}

//==============================================================================
// Templated function for computing different kinds of COM properties, like
// velocities and accelerations
template <
    typename PropertyType,
    PropertyType (BodyNode::*getPropertyFn)(const Frame*, const Frame*) const>
PropertyType getCOMPropertyTemplate(
    const Skeleton* _skel,
    const Frame* _relativeTo,
    const Frame* _inCoordinatesOf)
{
  PropertyType result(PropertyType::Zero());

  const std::size_t numBodies = _skel->getNumBodyNodes();
  for (std::size_t i = 0; i < numBodies; ++i)
  {
    const BodyNode* bodyNode = _skel->getBodyNode(i);
    result += bodyNode->getMass()
              * (bodyNode->*getPropertyFn)(_relativeTo, _inCoordinatesOf);
  }

  assert(_skel->getMass() != 0.0);
  return result / _skel->getMass();
}

//==============================================================================
Eigen::Vector6s Skeleton::getCOMSpatialVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getCOMPropertyTemplate<
      Eigen::Vector6s,
      &BodyNode::getCOMSpatialVelocity>(this, _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector3s Skeleton::getCOMLinearVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getCOMPropertyTemplate<
      Eigen::Vector3s,
      &BodyNode::getCOMLinearVelocity>(this, _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector6s Skeleton::getCOMSpatialAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getCOMPropertyTemplate<
      Eigen::Vector6s,
      &BodyNode::getCOMSpatialAcceleration>(
      this, _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector3s Skeleton::getCOMLinearAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getCOMPropertyTemplate<
      Eigen::Vector3s,
      &BodyNode::getCOMLinearAcceleration>(this, _relativeTo, _inCoordinatesOf);
}

//==============================================================================
// Templated function for computing different kinds of COM Jacobians and their
// derivatives
template <
    typename JacType, // JacType is the type of Jacobian we're computing
    JacType (TemplatedJacobianNode<BodyNode>::*getJacFn)(
        const Eigen::Vector3s&, const Frame*) const>
JacType getCOMJacobianTemplate(
    const Skeleton* _skel, const Frame* _inCoordinatesOf)
{
  // Initialize the Jacobian to zero
  JacType J = JacType::Zero(JacType::RowsAtCompileTime, _skel->getNumDofs());

  // Iterate through each of the Skeleton's BodyNodes
  const std::size_t numBodies = _skel->getNumBodyNodes();
  for (std::size_t i = 0; i < numBodies; ++i)
  {
    const BodyNode* bn = _skel->getBodyNode(i);

    // (bn->*getJacFn) is a function pointer to the function that gives us the
    // kind of Jacobian we want from the BodyNodes. Calling it will give us the
    // relevant Jacobian for this BodyNode
    JacType bnJ
        = bn->getMass() * (bn->*getJacFn)(bn->getLocalCOM(), _inCoordinatesOf);

    // For each column in the Jacobian of this BodyNode, we add it to the
    // appropriate column of the overall BodyNode
    for (std::size_t j = 0, end = bn->getNumDependentGenCoords(); j < end; ++j)
    {
      std::size_t idx = bn->getDependentGenCoordIndex(j);
      J.col(idx) += bnJ.col(j);
    }
  }

  assert(_skel->getMass() != 0.0);
  return J / _skel->getMass();
}

//==============================================================================
math::Jacobian Skeleton::getCOMJacobian(const Frame* _inCoordinatesOf) const
{
  return getCOMJacobianTemplate<
      math::Jacobian,
      &TemplatedJacobianNode<BodyNode>::getJacobian>(this, _inCoordinatesOf);
}

//==============================================================================
math::Jacobian Skeleton::getCOMPositionJacobian() const
{
  math::Jacobian J = math::Jacobian::Zero(6, getNumDofs());
  s_t totalMass = 0.0;
  for (const BodyNode* node : getBodyNodes())
  {
    totalMass += node->getMass();
    J += getWorldPositionJacobian(node) * node->getMass();
  }
  J /= totalMass;
  return J;
}

//==============================================================================
math::LinearJacobian Skeleton::getCOMLinearJacobian(
    const Frame* _inCoordinatesOf) const
{
  return getCOMJacobianTemplate<
      math::LinearJacobian,
      &TemplatedJacobianNode<BodyNode>::getLinearJacobian>(
      this, _inCoordinatesOf);
}

//==============================================================================
math::Jacobian Skeleton::getCOMJacobianSpatialDeriv(
    const Frame* _inCoordinatesOf) const
{
  return getCOMJacobianTemplate<
      math::Jacobian,
      &TemplatedJacobianNode<BodyNode>::getJacobianSpatialDeriv>(
      this, _inCoordinatesOf);
}

//==============================================================================
math::LinearJacobian Skeleton::getCOMLinearJacobianDeriv(
    const Frame* _inCoordinatesOf) const
{
  return getCOMJacobianTemplate<
      math::LinearJacobian,
      &TemplatedJacobianNode<BodyNode>::getLinearJacobianDeriv>(
      this, _inCoordinatesOf);
}

//==============================================================================
Skeleton::DirtyFlags::DirtyFlags()
  : mArticulatedInertia(true),
    mMassMatrix(true),
    mAugMassMatrix(true),
    mInvMassMatrix(true),
    mInvAugMassMatrix(true),
    mGravityForces(true),
    mCoriolisForces(true),
    mCoriolisAndGravityForces(true),
    mExternalForces(true),
    mDampingForces(true),
    mSupport(true),
    mParentMap(true),
    mSupportVersion(0)
{
  // Do nothing
}

} // namespace dynamics
} // namespace dart
