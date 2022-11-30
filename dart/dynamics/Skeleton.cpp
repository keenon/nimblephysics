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
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "dart/common/Console.hpp"
#include "dart/common/Deprecated.hpp"
#include "dart/common/StlHelpers.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/EndEffector.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/Frame.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Marker.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PointMass.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/SoftBodyNode.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/WithRespectTo.hpp"

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

  // Copy the scale groups
  skelClone->mBodyScaleGroups.clear();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    skelClone->mBodyScaleGroups.emplace_back();
    auto& thisGroup = mBodyScaleGroups[i];
    auto& cloneGroup = skelClone->mBodyScaleGroups[i];

    cloneGroup.flipAxis = thisGroup.flipAxis;
    cloneGroup.uniformScaling = thisGroup.uniformScaling;
    for (auto& node : thisGroup.nodes)
    {
      cloneGroup.nodes.push_back(skelClone->getBodyNode(node->getName()));
    }
  }
  skelClone->updateGroupScaleIndices();

  return skelClone;
}

//==============================================================================
/// Creates and returns a clone of this Skeleton, where we merge the provided
/// bodies together and approximate the CustomJoints with simpler joint types.
SkeletonPtr Skeleton::simplifySkeleton(
    const std::string& cloneName,
    std::map<std::string, std::string> mergeBodiesInto) const
{
  (void)mergeBodiesInto;

  SkeletonPtr skelClone = Skeleton::create(cloneName);

  for (std::size_t i = 0; i < getNumBodyNodes(); ++i)
  {
    // Create a clone of the parent Joint
    Joint* joint = getJoint(i)->simplifiedClone();
    if (joint == nullptr)
    {
      std::cout << "WARNING: Skeleton " << getName() << " cannot be simplified"
                << std::endl;
      return nullptr;
    }

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

  // Fix the scale groups
  for (auto group : mBodyScaleGroups)
  {
    if (group.nodes.size() > 0)
    {
      for (int i = 1; i < group.nodes.size(); i++)
      {
        skelClone->mergeScaleGroups(
            skelClone->getBodyNode(group.nodes[0]->getName()),
            skelClone->getBodyNode(group.nodes[i]->getName()));
      }
      skelClone->setScaleGroupUniformScaling(
          skelClone->getBodyNode(group.nodes[0]->getName()),
          group.uniformScaling);
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
// Gets the index in the skeleton where this joint lives
int Skeleton::getJointIndex(const Joint* joint)
{
  for (int i = 0; i < getNumJoints(); i++)
  {
    if (getJoint(i) == joint)
      return i;
  }
  return -1;
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
/// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
/// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
///
/// This is computed in bulk, and cached in the skeleton.
const Eigen::MatrixXi& Skeleton::getDofParentMap()
{
  if (mSkelCache.mDirty.mDofParentMap)
  {
    mSkelCache.mDofParentMap
        = Eigen::MatrixXi::Zero(getNumDofs(), getNumDofs());
    for (int row = 0; row < getNumDofs(); row++)
    {
      /*
      dynamics::DegreeOfFreedom* rowDof = getDof(row);
      for (int col = 0; col < getNumDofs(); col++) {
        dynamics::DegreeOfFreedom* colDof = getDof(col);
        if (rowDof->isParentOf(colDof)) {
          mSkelCache.mDofParentMap(row, col) = 1;
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
            mSkelCache.mDofParentMap(row, childJoint->getIndexInSkeleton(j))
                = 1;
          }
        }
      }
    }
    mSkelCache.mDirty.mDofParentMap = false;
  }
  return mSkelCache.mDofParentMap;
}

/// This returns a square (N x N) matrix, filled with 1s and 0s. This can be
/// interpreted as:
///
/// getJointParentMap(i,j) == 1: Joint[i] is a parent of Joint[j]
/// getJointParentMap(i,j) == 0: Joint[i] is NOT a parent of Joint[j]
///
/// This is computed in bulk, and cached in the skeleton.
const Eigen::MatrixXi& Skeleton::getJointParentMap()
{
  if (mSkelCache.mDirty.mJointParentMap)
  {
    mSkelCache.mJointParentMap
        = Eigen::MatrixXi::Zero(getNumJoints(), getNumJoints());
    for (int row = 0; row < getNumJoints(); row++)
    {
      dynamics::Joint* joint = getJoint(row);

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

          // Find which index the childJoint is at in the skeleton
          std::size_t index = getJointIndex(childJoint);

          assert(index != -1);

          mSkelCache.mJointParentMap(row, index) = 1;
        }
      }
    }
    mSkelCache.mDirty.mJointParentMap = false;
  }
  return mSkelCache.mJointParentMap;
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
  Eigen::MatrixXs DCg_Dp = Eigen::MatrixXs::Zero(dofs, wrt->dim(this));

  if (wrt == neural::WithRespectTo::FORCE)
  {
    return DCg_Dp;
  }
  else if (wrt == neural::WithRespectTo::LINEARIZED_MASSES)
  {
    Eigen::MatrixXs groupFromLin = getGroupMassesJacobianWrtLinearizedMasses();
    return getJacobianOfC(neural::WithRespectTo::GROUP_MASSES) * groupFromLin;
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::GROUP_SCALES
      || wrt == neural::WithRespectTo::GROUP_MASSES
      || wrt == neural::WithRespectTo::GROUP_COMS
      || wrt == neural::WithRespectTo::GROUP_INERTIAS)
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
  Eigen::MatrixXs DM_Dp = Eigen::MatrixXs::Zero(dofs, wrt->dim(this));

  if (wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::FORCE)
  {
    return DM_Dp;
  }
  else if (wrt == neural::WithRespectTo::LINEARIZED_MASSES)
  {
    Eigen::MatrixXs groupFromLin = getGroupMassesJacobianWrtLinearizedMasses();
    return getJacobianOfM(x, neural::WithRespectTo::GROUP_MASSES)
           * groupFromLin;
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::GROUP_SCALES
      || wrt == neural::WithRespectTo::GROUP_COMS
      || wrt == neural::WithRespectTo::GROUP_INERTIAS
      || wrt == neural::WithRespectTo::GROUP_MASSES)
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
      bodyNode->computeJacobianOfMBackward(wrt, DM_Dp);
    }

    setAccelerations(old_ddq);

    return DM_Dp;
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
  bool hasAnyZeroDof = false;
  for (int i = 0; i < getNumJoints(); i++)
  {
    if (getJoint(i)->getNumDofs() == 0)
    {
      hasAnyZeroDof = true;
      break;
    }
  }

  // Our analytical method doesn't yet support weld joints
  if (hasAnyZeroDof)
  {
    Eigen::MatrixXs Minv = getInvMassMatrix();
    // Use the traditional formula, d/dx A^{-1} = -A^{-1} (d/dx A) A^{-1}
    return -1 * Minv * getJacobianOfM(Minv * f, wrt);
  }

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
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(this, tweakedWrt);
        mSkelCache.mDirty.mMassMatrix = true;
        perturbed = getMassMatrix() * x;
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, originalWrt);
  mSkelCache.mDirty.mMassMatrix = true;
  getMassMatrix();

  return result;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfC(
    neural::WithRespectTo* wrt, bool useRidders)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(this, tweakedWrt);
        perturbed = getCoriolisAndGravityForces() - getExternalForces();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, originalWrt);

  return result;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfID(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt, bool useRidders)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  const Eigen::VectorXs old_ddq = getAccelerations();
  setAccelerations(f);

  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(this, tweakedWrt);
        computeInverseDynamics();
        perturbed = getControlForces();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, originalWrt);
  setAccelerations(old_ddq);

  return result;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfMinv(
    const Eigen::VectorXs& f, neural::WithRespectTo* wrt, bool useRidders)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(this, tweakedWrt);
        perturbed = multiplyByImplicitInvMassMatrix(f);
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, originalWrt);

  return result;
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
/// This computes the distance (along the `up` vector) from the highest vertex
/// to the lowest vertex on the model, when positioned at `pose`
s_t Skeleton::getHeight(Eigen::VectorXs pose, Eigen::Vector3s up)
{
  Eigen::VectorXs originalPose = getPositions();
  setPositions(pose);

  s_t maxUp = -1 * std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s maxVertex = Eigen::Vector3s::Zero();
  dynamics::BodyNode* maxBodyNode = nullptr;
  s_t minUp = std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s minVertex = Eigen::Vector3s::Zero();
  dynamics::BodyNode* minBodyNode = nullptr;

  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* node = getBodyNode(i);
    for (int j = 0; j < node->getNumShapeNodes(); j++)
    {
      dynamics::ShapeNode* shapeNode = node->getShapeNode(j);
      std::shared_ptr<dynamics::Shape> shape = shapeNode->getShape();
      if (shape->getType() == dynamics::MeshShape::getStaticType())
      {
        dynamics::MeshShape* mesh
            = static_cast<dynamics::MeshShape*>(shape.get());
        for (Eigen::Vector3s rawVertex : mesh->getVertices())
        {
          Eigen::Vector3s vertex = node->getScale().cwiseProduct(rawVertex);
          Eigen::Vector3s worldVertex = shapeNode->getWorldTransform() * vertex;
          s_t upDist = up.dot(worldVertex);
          if (upDist > maxUp)
          {
            maxUp = upDist;
            maxBodyNode = node;
            maxVertex = worldVertex;
          }
          if (upDist < minUp)
          {
            minUp = upDist;
            minBodyNode = node;
            minVertex = worldVertex;
          }
        }
      }
      else
      {
        std::cout << "WARNING: getHeight() currently only supports Skeletons "
                     "with Mesh shapes. Instead we got a shape of type \""
                  << shape->getType()
                  << "\". This shape will be ignored in computing height."
                  << std::endl;
      }
    }
  }

  (void)maxVertex;
  (void)maxBodyNode;
  (void)minVertex;
  (void)minBodyNode;

  setPositions(originalPose);

  if (isfinite(maxUp) && isfinite(minUp))
  {
    return maxUp - minUp;
  }
  // Fallback
  return 0.0;
}

//==============================================================================
/// This computes the gradient of the height
Eigen::VectorXs Skeleton::getGradientOfHeightWrtBodyScales(
    Eigen::VectorXs pose, Eigen::Vector3s up)
{
  Eigen::VectorXs originalPose = getPositions();
  setPositions(pose);

  s_t maxUp = -1 * std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s maxVertex = Eigen::Vector3s::Zero();
  dynamics::BodyNode* maxBodyNode = nullptr;
  s_t minUp = std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s minVertex = Eigen::Vector3s::Zero();
  dynamics::BodyNode* minBodyNode = nullptr;

  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* node = getBodyNode(i);
    for (int j = 0; j < node->getNumShapeNodes(); j++)
    {
      dynamics::ShapeNode* shapeNode = node->getShapeNode(j);
      std::shared_ptr<dynamics::Shape> shape = shapeNode->getShape();
      if (shape->getType() == dynamics::MeshShape::getStaticType())
      {
        dynamics::MeshShape* mesh
            = static_cast<dynamics::MeshShape*>(shape.get());
        for (Eigen::Vector3s vertex : mesh->getVertices())
        {
          Eigen::Vector3s worldVertex = shapeNode->getWorldTransform() * vertex;
          s_t upDist = up.dot(worldVertex);
          if (upDist > maxUp)
          {
            maxUp = upDist;
            maxBodyNode = node;
            maxVertex = node->getWorldTransform().inverse() * worldVertex;
          }
          if (upDist < minUp)
          {
            minUp = upDist;
            minBodyNode = node;
            minVertex = node->getWorldTransform().inverse() * worldVertex;
          }
        }
      }
      else
      {
        std::cout << "WARNING: getGradientOfHeightWrtBodyScales() currently "
                     "only supports Skeletons "
                     "with Mesh shapes. Instead we got a shape of type \""
                  << shape->getType()
                  << "\". This shape will be ignored in computing the gradient."
                  << std::endl;
      }
    }
  }

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.emplace_back(minBodyNode, minVertex);
  markers.emplace_back(maxBodyNode, maxVertex);
  Eigen::Vector6s markerToHeight = Eigen::Vector6s::Zero();
  markerToHeight.head<3>() = -up;
  markerToHeight.tail<3>() = up;

  Eigen::VectorXs grad
      = getMarkerWorldPositionsJacobianWrtBodyScales(markers).transpose()
        * markerToHeight;

  setPositions(originalPose);
  return grad;
}

//==============================================================================
/// This computes the gradient of the height
Eigen::VectorXs Skeleton::finiteDifferenceGradientOfHeightWrtBodyScales(
    Eigen::VectorXs pose, Eigen::Vector3s up)
{
  Eigen::VectorXs originalPose = getPositions();
  setPositions(pose);

  std::size_t n = getNumBodyNodes() * 3;
  Eigen::VectorXs result(n);
  Eigen::VectorXs originalScales = getBodyScales();

  s_t eps = 1e-6;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& height) {
        Eigen::VectorXs tweaked = originalScales;
        tweaked(i) += eps;
        setBodyScales(tweaked);
        height = getHeight(pose, up);
        return true;
      },
      result,
      eps,
      false);

  setBodyScales(originalScales);
  setPositions(originalPose);

  return result;
}

//==============================================================================
/// This returns a marker set with at least one marker in it, that each
/// represents the lowest point on the body, measure by the `up` vector, in
/// the specified position. If there are no ties, this will be of length 1. If
/// there are more than one tied lowest point, then this is of length > 1.
std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
Skeleton::getLowestPointMarkers(Eigen::Vector3s up)
{
  s_t minUp = std::numeric_limits<s_t>::infinity();
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> minMarkers;

  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* node = getBodyNode(i);
    for (int j = 0; j < node->getNumShapeNodes(); j++)
    {
      dynamics::ShapeNode* shapeNode = node->getShapeNode(j);
      std::shared_ptr<dynamics::Shape> shape = shapeNode->getShape();
      if (shape->getType() == dynamics::MeshShape::getStaticType())
      {
        dynamics::MeshShape* mesh
            = static_cast<dynamics::MeshShape*>(shape.get());
        for (Eigen::Vector3s rawVertex : mesh->getVertices())
        {
          Eigen::Vector3s vertex = node->getScale().cwiseProduct(rawVertex);
          Eigen::Vector3s worldVertex = shapeNode->getWorldTransform() * vertex;
          s_t upDist = up.dot(worldVertex);
          if (upDist < minUp)
          {
            minUp = upDist;
            minMarkers.clear();
          }
          if (upDist <= minUp)
          {
            minMarkers.emplace_back(
                node, node->getWorldTransform().inverse() * worldVertex);
          }
        }
      }
      else
      {
        std::cout << "WARNING: getGradientOfHeightWrtBodyScales() currently "
                     "only supports Skeletons "
                     "with Mesh shapes. Instead we got a shape of type \""
                  << shape->getType()
                  << "\". This shape will be ignored in computing the gradient."
                  << std::endl;
      }
    }
  }

  return minMarkers;
}

//==============================================================================
/// This computes the lowest point on the colliders, as measured by the `up`
/// vector. This is useful in order to apply constraints that a model can't
/// penetrate the ground.
s_t Skeleton::getLowestPoint(Eigen::Vector3s up)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> minMarkers
      = getLowestPointMarkers(up);

  s_t lowestPoint
      = up.dot(minMarkers[0].first->getWorldTransform() * minMarkers[0].second);

  return lowestPoint;
}

//==============================================================================
/// This computes the gradient of the lowest point wrt body scales
Eigen::VectorXs Skeleton::getGradientOfLowestPointWrtBodyScales(
    Eigen::Vector3s up)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> minMarkers
      = getLowestPointMarkers(up);

  Eigen::VectorXs markersToLowestPoint
      = Eigen::VectorXs::Zero(minMarkers.size() * 3);
  for (int i = 0; i < minMarkers.size(); i++)
  {
    markersToLowestPoint.segment<3>(i * 3) = up;
  }

  Eigen::VectorXs grad
      = getMarkerWorldPositionsJacobianWrtBodyScales(minMarkers).transpose()
        * markersToLowestPoint;

  return grad;
}

//==============================================================================
/// This computes the gradient of the lowest point wrt body scales
Eigen::VectorXs Skeleton::finiteDifferenceGradientOfLowestPointWrtBodyScales(
    Eigen::Vector3s up)
{
  std::size_t n = getNumBodyNodes() * 3;
  Eigen::VectorXs result(n);
  Eigen::VectorXs originalScales = getBodyScales();

  s_t eps = 1e-6;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs tweaked = originalScales;
        tweaked(i) += eps;
        setBodyScales(tweaked);
        out = getLowestPoint(up);
        return true;
      },
      result,
      eps,
      false);

  setBodyScales(originalScales);

  return result;
}

//==============================================================================
/// This computes the gradient of the lowest point wrt body scales
Eigen::VectorXs Skeleton::getGradientOfLowestPointWrtJoints(Eigen::Vector3s up)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> minMarkers
      = getLowestPointMarkers(up);

  Eigen::VectorXs markersToLowestPoint
      = Eigen::VectorXs::Zero(minMarkers.size() * 3);
  for (int i = 0; i < minMarkers.size(); i++)
  {
    markersToLowestPoint.segment<3>(i * 3) = up;
  }

  Eigen::VectorXs grad
      = getMarkerWorldPositionsJacobianWrtJointPositions(minMarkers).transpose()
        * markersToLowestPoint;

  return grad;
}

//==============================================================================
/// This computes the gradient of the lowest point wrt body scales
Eigen::VectorXs Skeleton::finiteDifferenceGradientOfLowestPointWrtJoints(
    Eigen::Vector3s up)
{
  Eigen::VectorXs originalPose = getPositions();

  std::size_t n = getNumDofs();
  Eigen::VectorXs result(n);

  s_t eps = 1e-6;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs tweaked = originalPose;
        tweaked(i) += eps;
        setPositions(tweaked);
        out = getLowestPoint(up);
        return true;
      },
      result,
      eps,
      false);

  setPositions(originalPose);

  return result;
}

//==============================================================================
/// This gets a random pose that's valid within joint limits
Eigen::VectorXs Skeleton::getRandomPose()
{
  Eigen::VectorXs pose = Eigen::VectorXs::Random(getNumDofs());
  for (int i = 0; i < getNumDofs(); i++)
  {
    s_t upperLimit = getDof(i)->getPositionUpperLimit() - 0.02;
    if (upperLimit == std::numeric_limits<s_t>::infinity())
    {
      upperLimit = 5.0;
    }
    s_t lowerLimit = getDof(i)->getPositionLowerLimit() + 0.02;
    if (lowerLimit == -1 * std::numeric_limits<s_t>::infinity())
    {
      lowerLimit = -5.0;
    }
    // If there's no space in the bounds:
    if (upperLimit < lowerLimit)
    {
      pose(i) = getDof(i)->getPositionUpperLimit();
    }
    else
    {
      s_t withinBounds
          = (((abs(pose(i)) + 1.0) / 2.0) * (upperLimit - lowerLimit))
            + lowerLimit;
      pose(i) = withinBounds;
    }
  }

  /*
#ifndef NDEBUG
  Eigen::VectorXs oldPose = getPositions();
  setPositions(pose);
  clampPositionsToLimits();
  Eigen::VectorXs clampedPos = getPositions();
  assert(clampedPos == pose);
  setPositions(oldPose);
#endif
  */

  return pose;
}

//==============================================================================
/// This gets a random pose that's valid within joint limits
Eigen::VectorXs Skeleton::getRandomVelocity()
{
  Eigen::VectorXs pose = Eigen::VectorXs::Random(getNumDofs());
  for (int i = 0; i < getNumDofs(); i++)
  {
    s_t upperLimit = getDof(i)->getVelocityUpperLimit() - 0.015;
    if (upperLimit == std::numeric_limits<s_t>::infinity())
    {
      upperLimit = 5.0;
    }
    s_t lowerLimit = getDof(i)->getVelocityLowerLimit() + 0.015;
    if (lowerLimit == -1 * std::numeric_limits<s_t>::infinity())
    {
      lowerLimit = -5.0;
    }
    if (upperLimit < lowerLimit)
    {
      pose(i) = getDof(i)->getVelocityUpperLimit();
    }
    else
    {
      s_t withinBounds
          = (((abs(pose(i)) + 1.0) / 2.0) * (upperLimit - lowerLimit))
            + lowerLimit;
      pose(i) = withinBounds;
    }
  }

  /*
#ifndef NDEBUG
  Eigen::VectorXs oldPose = getPositions();
  setPositions(pose);
  clampPositionsToLimits();
  Eigen::VectorXs clampedPos = getPositions();
  assert(clampedPos == pose);
  setPositions(oldPose);
#endif
  */

  return pose;
}

//==============================================================================
/// This gets a random pose that's valid within joint limits, but only changes
/// the specified joints. All unspecified joints are left as 0.
Eigen::VectorXs Skeleton::getRandomPoseForJoints(
    std::vector<dynamics::Joint*> joints)
{
  Eigen::VectorXs randomPose = getRandomPose();
  Eigen::VectorXs pose = Eigen::VectorXs::Zero(getNumDofs());

  for (int i = 0; i < getNumDofs(); i++)
  {
    pose(i) = getDof(i)->getInitialPosition();
  }

  for (auto joint : joints)
  {
    if (joint->getNumDofs() > 0)
    {
      pose.segment(joint->getDof(0)->getIndexInSkeleton(), joint->getNumDofs())
          = randomPose.segment(
              joint->getDof(0)->getIndexInSkeleton(), joint->getNumDofs());
    }
  }

  return pose;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceVelCJacobian(bool useRidders)
{
  std::size_t n = getNumDofs();
  Eigen::MatrixXs result(n, n);
  Eigen::VectorXs originalVel = getVelocities();

  s_t eps = useRidders ? 1e-3 : 1e-6;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedVel = originalVel;
        tweakedVel(dof) += eps;
        setVelocities(tweakedVel);
        perturbed = getCoriolisAndGravityForces();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  setVelocities(originalVel);

  // DEBUGGING
  // TODO: delete?
#ifndef NDEBUG
  // Get baseline C(pos, vel)
  Eigen::VectorXs baseline = getCoriolisAndGravityForces();
  s_t EPS = 1e-6;
  for (std::size_t i = 0; i < n; i++)
  {
    Eigen::VectorXs tweakedVel = originalVel;
    tweakedVel(i) += EPS;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedPos = getCoriolisAndGravityForces();
    tweakedVel = originalVel;
    tweakedVel(i) -= EPS;
    setVelocities(tweakedVel);
    Eigen::VectorXs perturbedNeg = getCoriolisAndGravityForces();

    if (perturbedPos == perturbedNeg && perturbedPos != baseline)
    {
      // std::cout << "Got a mysteriously broken coriolis force result" <<
      // std::endl;

      // Set positive vel change

      tweakedVel = originalVel;
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

      tweakedVel = originalVel;
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
  }
  // Reset everything how we left it
  setVelocities(originalVel);
#endif

  return result;
}

//==============================================================================
Eigen::MatrixXs Skeleton::finiteDifferenceJacobianOfFD(
    neural::WithRespectTo* wrt, bool useRidders)
{
  std::size_t n = getNumDofs();
  std::size_t m = wrt->dim(this);
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(this);

  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(this, tweakedWrt);
        computeForwardDynamics();
        perturbed = getAccelerations();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, originalWrt);
  computeForwardDynamics();

  return result;
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
s_t Skeleton::getLinkMUIndex(size_t index)
{
  Eigen::Vector3s com = getLinkCOMIndex(index);
  Eigen::Vector3s beta = getBodyNode(index)->getBeta();
  if (beta(0) != 0)
    return com(0) / beta(0);
  else if (beta(1) != 0)
    return com(1) / beta(1);
  else
    return com(2) / beta(2);
  // Code should not reach here only to please the compiler
  return 0;
}

Eigen::VectorXs Skeleton::getLinkMUs()
{
  Eigen::VectorXs mus = Eigen::VectorXs::Zero(getNumBodyNodes());
  for (size_t i = 0; i < getNumBodyNodes(); i++)
  {
    mus(i) = getLinkMUIndex(i);
  }
  return mus;
}

//==============================================================================
Eigen::Vector3s Skeleton::getLinkBetaIndex(size_t index)
{
  Eigen::Vector3s beta = getBodyNode(index)->getBeta();
  return beta;
}

Eigen::VectorXs Skeleton::getLinkBetas()
{
  Eigen::VectorXs betas = Eigen::VectorXs::Zero(3 * getNumBodyNodes());
  size_t cursor = 0;
  for (size_t i = 0; i < getNumBodyNodes(); i++)
  {
    Eigen::Vector3s beta = getBodyNode(i)->getBeta();
    betas.segment(cursor, 3) = beta;
    cursor += 3;
  }
  return betas;
}
//==============================================================================
Eigen::VectorXs Skeleton::getLinkCOMs()
{
  Eigen::VectorXs inertias = Eigen::VectorXs::Zero(getLinkCOMDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::COM_X);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::COM_Y);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::COM_Z);
  }
  return inertias;
}

Eigen::Vector3s Skeleton::getLinkCOMIndex(size_t index)
{
  Eigen::Vector3s mass_center = Eigen::Vector3s::Zero();
  const Inertia& node_inertia = getBodyNode(index)->getInertia();
  mass_center(0) = node_inertia.getParameter(dynamics::Inertia::Param::COM_X);
  mass_center(1) = node_inertia.getParameter(dynamics::Inertia::Param::COM_Y);
  mass_center(2) = node_inertia.getParameter(dynamics::Inertia::Param::COM_Z);
  return mass_center;
}

//==============================================================================
Eigen::VectorXs Skeleton::getLinkMOIs()
{
  Eigen::VectorXs inertias = Eigen::VectorXs::Zero(getLinkMOIDims());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_XX);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_YY);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_ZZ);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_XY);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_XZ);
    inertias(cursor++) = inertia.getParameter(dynamics::Inertia::Param::I_YZ);
  }
  return inertias;
}

Eigen::Vector6s Skeleton::getLinkMOIIndex(size_t index)
{
  Eigen::Vector6s inertia = Eigen::Vector6s::Zero();
  const Inertia& node_inertia = getBodyNode(index)->getInertia();
  inertia(0) = node_inertia.getParameter(dynamics::Inertia::Param::I_XX);
  inertia(1) = node_inertia.getParameter(dynamics::Inertia::Param::I_YY);
  inertia(2) = node_inertia.getParameter(dynamics::Inertia::Param::I_ZZ);
  inertia(3) = node_inertia.getParameter(dynamics::Inertia::Param::I_XY);
  inertia(4) = node_inertia.getParameter(dynamics::Inertia::Param::I_XZ);
  inertia(5) = node_inertia.getParameter(dynamics::Inertia::Param::I_YZ);
  return inertia;
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
void Skeleton::setLinkMUIndex(s_t mu, size_t index)
{
  Eigen::Vector3s com = Eigen::Vector3s::Zero();
  Eigen::Vector3s node_beta = getBodyNode(index)->getBeta();
  com(0) = node_beta(0) * mu;
  com(1) = node_beta(1) * mu;
  com(2) = node_beta(2) * mu;
  setLinkCOMIndex(com, index);
}

void Skeleton::setLinkMUs(Eigen::VectorXs mus)
{
  assert(mus.size() == getNumBodyNodes());
  for (size_t i = 0; i < getNumBodyNodes(); i++)
  {
    Eigen::Vector3s com = Eigen::Vector3s::Zero();
    Eigen::Vector3s beta = getBodyNode(i)->getBeta();
    com(0) = beta(0) * mus(i);
    com(1) = beta(1) * mus(i);
    com(2) = beta(2) * mus(i);
    setLinkCOMIndex(com, i);
  }
}

//==============================================================================

void Skeleton::setLinkBetaIndex(Eigen::Vector3s beta, size_t index)
{
  getBodyNode(index)->setBeta(beta);
}

void Skeleton::setLinkBetas(Eigen::VectorXs betas)
{
  size_t cursor = 0;
  for (size_t i = 0; i < getNumBodyNodes(); i++)
  {
    getBodyNode(i)->setBeta(betas.segment(cursor, 3));
    cursor += 3;
  }
}

//==============================================================================
void Skeleton::setLinkCOMs(Eigen::VectorXs coms)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < getNumBodyNodes(); i++)
  {
    const Inertia& inertia = getBodyNode(i)->getInertia();
    s_t com_x = coms(cursor++);
    s_t com_y = coms(cursor++);
    s_t com_z = coms(cursor++);
    Inertia newInertia(
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        com_x,
        com_y,
        com_z,
        inertia.getParameter(dynamics::Inertia::Param::I_XX),
        inertia.getParameter(dynamics::Inertia::Param::I_YY),
        inertia.getParameter(dynamics::Inertia::Param::I_ZZ),
        inertia.getParameter(dynamics::Inertia::Param::I_XY),
        inertia.getParameter(dynamics::Inertia::Param::I_XZ),
        inertia.getParameter(dynamics::Inertia::Param::I_YZ));
    getBodyNode(i)->setInertia(newInertia);
  }
}

void Skeleton::setLinkCOMIndex(Eigen::Vector3s com, size_t index)
{
  const Inertia& inertia = getBodyNode(index)->getInertia();
  s_t com_x = com(0);
  s_t com_y = com(1);
  s_t com_z = com(2);
  Inertia newInertia(
      inertia.getParameter(dynamics::Inertia::Param::MASS),
      com_x,
      com_y,
      com_z,
      inertia.getParameter(dynamics::Inertia::Param::I_XX),
      inertia.getParameter(dynamics::Inertia::Param::I_YY),
      inertia.getParameter(dynamics::Inertia::Param::I_ZZ),
      inertia.getParameter(dynamics::Inertia::Param::I_XY),
      inertia.getParameter(dynamics::Inertia::Param::I_XZ),
      inertia.getParameter(dynamics::Inertia::Param::I_YZ));
  getBodyNode(index)->setInertia(newInertia);
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
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        inertia.getParameter(dynamics::Inertia::Param::COM_X),
        inertia.getParameter(dynamics::Inertia::Param::COM_Y),
        inertia.getParameter(dynamics::Inertia::Param::COM_Z),
        I_XX,
        I_YY,
        I_ZZ,
        I_XY,
        I_XZ,
        I_YZ);
    getBodyNode(i)->setInertia(newInertia);
  }
}

void Skeleton::setLinkMOIIndex(Eigen::Vector6s moi, size_t index)
{
  const Inertia& inertia = getBodyNode(index)->getInertia();
  s_t I_XX = moi(0);
  s_t I_YY = moi(1);
  s_t I_ZZ = moi(2);
  s_t I_XY = moi(3);
  s_t I_XZ = moi(4);
  s_t I_YZ = moi(5);
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
  getBodyNode(index)->setInertia(newInertia);
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
Eigen::VectorXs Skeleton::getBodyScales()
{
  Eigen::VectorXs scales = Eigen::VectorXs::Zero(getNumBodyNodes() * 3);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    scales.segment<3>(i * 3) = getBodyNode(i)->getScale();
  }
  return scales;
}

//==============================================================================
// Sets all the link scales for the skeleton, from a flat vector
void Skeleton::setBodyScales(Eigen::VectorXs scales)
{
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    getBodyNode(i)->setScale(scales.segment<3>(i * 3));
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

    // Special case for rotational joints and euler joints, we're interested in
    // wrapping by 2*PI
    bool wrapByTwoPi = false;
    if (dof->getJoint()->getType() == RevoluteJoint::getStaticType()
        || dof->getJoint()->getType() == EulerJoint::getStaticType())
    {
      wrapByTwoPi = true;
    }
    if (dof->getJoint()->getType() == EulerFreeJoint::getStaticType()
        && dof->getIndexInJoint() < 3)
    {
      wrapByTwoPi = true;
    }

    if (wrapByTwoPi)
    {
      std::vector<s_t> posesToTry;
      s_t clampedPos = dof->getPosition();
      posesToTry.push_back(clampedPos);
      while (clampedPos > dof->getPositionUpperLimit())
      {
        clampedPos -= 2 * M_PI;
        posesToTry.push_back(clampedPos);
      }
      while (clampedPos < dof->getPositionLowerLimit())
      {
        clampedPos += 2 * M_PI;
        posesToTry.push_back(clampedPos);
      }

      // Choose the value that results in the smallest constraint violation
      s_t lowestViolation = std::numeric_limits<double>::infinity();
      for (s_t pos : posesToTry)
      {
        // Set the value if we ended up in bounds
        if (clampedPos >= dof->getPositionLowerLimit()
            && clampedPos <= dof->getPositionUpperLimit())
        {
          dof->setPosition(pos);
          break;
        }
        else if (clampedPos > dof->getPositionUpperLimit())
        {
          s_t violation = clampedPos - dof->getPositionUpperLimit();
          if (violation < lowestViolation)
          {
            dof->setPosition(pos);
            lowestViolation = violation;
          }
        }
        else if (clampedPos < dof->getPositionLowerLimit())
        {
          s_t violation = dof->getPositionLowerLimit() - pos;
          if (violation < lowestViolation)
          {
            dof->setPosition(pos);
            lowestViolation = violation;
          }
        }
      }
    }

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
/// There is an annoying tendency for custom joints to encode the linear
/// offset of the bone in their custom functions. We don't want that, so we
/// want to move any relative transform caused by custom functions into the
/// parent transform.
void Skeleton::zeroTranslationInCustomFunctions()
{
  for (int i = 0; i < getNumJoints(); i++)
  {
    if (getJoint(i)->getType() == CustomJoint<1>::getStaticType())
    {
      static_cast<CustomJoint<1>*>(getJoint(i))
          ->zeroTranslationInCustomFunctions();
    }
    if (getJoint(i)->getType() == CustomJoint<2>::getStaticType())
    {
      static_cast<CustomJoint<2>*>(getJoint(i))
          ->zeroTranslationInCustomFunctions();
    }
    if (getJoint(i)->getType() == CustomJoint<3>::getStaticType())
    {
      static_cast<CustomJoint<3>*>(getJoint(i))
          ->zeroTranslationInCustomFunctions();
    }
  }
}

//==============================================================================
const std::vector<BodyScaleGroup>& Skeleton::getBodyScaleGroups() const
{
  return mBodyScaleGroups;
}

//==============================================================================
const BodyScaleGroup& Skeleton::getBodyScaleGroup(int index) const
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
      mBodyScaleGroups.emplace_back();
      BodyScaleGroup& group = mBodyScaleGroups.back();
      group.nodes.push_back(getBodyNode(i));
      group.flipAxis.push_back(Eigen::Vector3s::Ones());
      group.uniformScaling = false;
    }
    updateGroupScaleIndices();
  }
}

//==============================================================================
/// This returns the index of the group that this body node corresponds to
int Skeleton::getScaleGroupIndex(dynamics::BodyNode* bodyNode)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (int j = 0; j < mBodyScaleGroups[i].nodes.size(); j++)
    {
      if (mBodyScaleGroups[i].nodes[j]->getName() == bodyNode->getName())
        return i;
    }
  }
  return -1;
}

//==============================================================================
/// This returns the axis flips of this body in the scale group that this body
/// node corresponds to
Eigen::Vector3s Skeleton::getScaleGroupFlips(dynamics::BodyNode* bodyNode)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (int j = 0; j < mBodyScaleGroups[i].nodes.size(); j++)
    {
      if (mBodyScaleGroups[i].nodes[j]->getName() == bodyNode->getName())
        return mBodyScaleGroups[i].flipAxis[j];
    }
  }
  return Eigen::Vector3s::Ones();
}

//==============================================================================
/// This takes two scale groups and merges their contents into a single group.
/// After this operation, there is one fewer scale group.
void Skeleton::mergeScaleGroups(dynamics::BodyNode* a, dynamics::BodyNode* b)
{
  mergeScaleGroupsByIndex(getScaleGroupIndex(a), getScaleGroupIndex(b));
}

//==============================================================================
/// The scale group axis flips
void Skeleton::autodetectScaleGroupAxisFlips(int symmetryAxis)
{
  for (BodyScaleGroup& group : mBodyScaleGroups)
  {
    s_t midPoint = 0.0;
    for (auto* node : group.nodes)
    {
      midPoint += node->getWorldTransform().translation()(symmetryAxis);
    }
    midPoint /= group.nodes.size();
    for (int j = 0; j < group.nodes.size(); j++)
    {
      bool positive
          = midPoint - 1e-8
            <= group.nodes[j]->getWorldTransform().translation()(symmetryAxis);
      group.flipAxis[j](symmetryAxis) = positive ? 1.0 : -1.0;
    }
  }
}

//==============================================================================
/// This finds all the pairs of bodies that share the same prefix, and
/// different suffixes (for example "a_body_l" and "a_body_r", sharing "_l"
/// and "_r")
void Skeleton::autogroupSymmetricSuffixes(
    std::string leftSuffix, std::string rightSuffix)
{
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* leftBody = getBodyNode(i);
    std::string leftBodyName = leftBody->getName();
    if (leftBodyName.size() < leftSuffix.size())
      continue;
    std::string leftBodySuffix = leftBodyName.substr(
        leftBodyName.size() - leftSuffix.size(), leftSuffix.size());
    if (leftBodySuffix != leftSuffix)
      continue;
    std::string leftBodyPrefix
        = leftBodyName.substr(0, leftBodyName.size() - leftSuffix.size());
    for (int j = 0; j < getNumBodyNodes(); j++)
    {
      if (i == j)
        continue;
      dynamics::BodyNode* rightBody = getBodyNode(j);
      std::string rightBodyName = rightBody->getName();
      if (rightBodyName.size() < rightSuffix.size())
        continue;
      std::string rightBodySuffix = rightBodyName.substr(
          rightBodyName.size() - rightSuffix.size(), rightSuffix.size());
      if (rightBodySuffix != rightSuffix)
        continue;
      std::string rightBodyPrefix
          = rightBodyName.substr(0, rightBodyName.size() - rightSuffix.size());
      if (leftBodyPrefix == rightBodyPrefix)
      {
        // If we make it here, then we have a genuine match, so merge the bodies
        mergeScaleGroups(leftBody, rightBody);
      }
    }
  }
}

//==============================================================================
/// This finds all the pairs of bodies that share the same suffix, and
/// different prefixes (for example "ulna_l" and "radius_l", sharing "_l")
void Skeleton::autogroupSymmetricPrefixes(
    std::string firstPrefix, std::string secondPrefix)
{
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* firstBody = getBodyNode(i);
    std::string firstBodyName = firstBody->getName();
    if (firstBodyName.size() < firstPrefix.size())
      continue;
    std::string firstBodyPrefix = firstBodyName.substr(0, firstPrefix.size());
    if (firstBodyPrefix != firstPrefix)
      continue;
    std::string firstBodySuffix = firstBodyName.substr(
        firstPrefix.size(), firstBodyName.size() - firstPrefix.size());
    for (int j = 0; j < getNumBodyNodes(); j++)
    {
      if (i == j)
        continue;
      dynamics::BodyNode* secondBody = getBodyNode(j);
      std::string secondBodyName = secondBody->getName();
      if (secondBodyName.size() < secondPrefix.size())
        continue;
      std::string secondBodyPrefix
          = secondBodyName.substr(0, secondPrefix.size());
      if (secondBodyPrefix != secondPrefix)
        continue;
      std::string secondBodySuffix = secondBodyName.substr(
          secondPrefix.size(), secondBodyName.size() - secondPrefix.size());
      if (firstBodySuffix == secondBodySuffix)
      {
        // If we make it here, then we have a genuine match, so merge the bodies
        mergeScaleGroups(firstBody, secondBody);
      }
    }
  }
}

//==============================================================================
/// This means that we'll scale a group along all three axis equally. This
/// constrains scaling.
void Skeleton::setScaleGroupUniformScaling(dynamics::BodyNode* a, bool uniform)
{
  mBodyScaleGroups[getScaleGroupIndex(a)].uniformScaling = uniform;
  updateGroupScaleIndices();
}

//==============================================================================
/// This returns the number of scale groups
int Skeleton::getNumScaleGroups()
{
  ensureBodyScaleGroups();
  return mBodyScaleGroups.size();
}

//==============================================================================
/// This returns the dimension of the scale group
int Skeleton::getScaleGroupDim(int groupIndex)
{
  if (mBodyScaleGroups[groupIndex].uniformScaling)
    return 1;
  return 3;
}

//==============================================================================
/// This gets the scale upper bound for the first body in a group, by index
Eigen::VectorXs Skeleton::getScaleGroupUpperBound(int groupIndex)
{
  Eigen::Vector3s result
      = mBodyScaleGroups[groupIndex].nodes[0]->getScaleUpperBound();
  for (int i = 0; i < mBodyScaleGroups[groupIndex].nodes.size(); i++)
  {
    Eigen::Vector3s localBound
        = mBodyScaleGroups[groupIndex].nodes[i]->getScaleUpperBound();
    result = result.cwiseMin(localBound);
  }
  if (mBodyScaleGroups[groupIndex].uniformScaling)
  {
    return Eigen::Vector1s(result.minCoeff());
  }
  else
  {
    return result;
  }
}

//==============================================================================
/// This gets the scale lower bound for the first body in a group, by index
Eigen::VectorXs Skeleton::getScaleGroupLowerBound(int groupIndex)
{
  Eigen::Vector3s result
      = mBodyScaleGroups[groupIndex].nodes[0]->getScaleLowerBound();
  for (int i = 0; i < mBodyScaleGroups[groupIndex].nodes.size(); i++)
  {
    Eigen::Vector3s localBound
        = mBodyScaleGroups[groupIndex].nodes[i]->getScaleLowerBound();
    result = result.cwiseMax(localBound);
  }
  if (mBodyScaleGroups[groupIndex].uniformScaling)
  {
    return Eigen::Vector1s(result.maxCoeff());
  }
  else
  {
    return result;
  }
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
  BodyScaleGroup& groupA = mBodyScaleGroups[a];
  BodyScaleGroup& groupB = mBodyScaleGroups[b];
  // Remove the element further back in the array
  if (a > b)
  {
    // Transfer all elems from A to B
    for (dynamics::BodyNode*& node : groupA.nodes)
      groupB.nodes.push_back(node);
    for (Eigen::Vector3s& flips : groupA.flipAxis)
      groupB.flipAxis.push_back(flips);
    // If either group has uniform scaling, merged group has it
    groupB.uniformScaling = groupA.uniformScaling || groupB.uniformScaling;
    // Then erase A
    mBodyScaleGroups.erase(mBodyScaleGroups.begin() + a);
  }
  else
  {
    // Transfer all elems from B to A
    for (dynamics::BodyNode*& node : groupB.nodes)
      groupA.nodes.push_back(node);
    for (Eigen::Vector3s& flips : groupB.flipAxis)
      groupA.flipAxis.push_back(flips);
    // If either group has uniform scaling, merged group has it
    groupA.uniformScaling = groupA.uniformScaling || groupB.uniformScaling;
    // Then erase B
    mBodyScaleGroups.erase(mBodyScaleGroups.begin() + b);
  }
  updateGroupScaleIndices();
}

//==============================================================================
/// This returns the dimensions of the grouped scale vector.
int Skeleton::getGroupScaleDim()
{
  ensureBodyScaleGroups();
  int sum = 0;
  for (auto group : mBodyScaleGroups)
  {
    if (group.uniformScaling)
    {
      sum += 1;
    }
    else
    {
      sum += 3;
    }
  }
  return sum;
}

//==============================================================================
BodyScaleGroupAndIndex::BodyScaleGroupAndIndex(BodyScaleGroup& group, int axis)
  : group(group), axis(axis)
{
}

//==============================================================================
/// This precomputes the array of group scale indices that we need for
/// getGroupScaleIndexDetails()
void Skeleton::updateGroupScaleIndices()
{
  mGroupScaleIndices.clear();
  // Find the group and axis we're talking about
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    if (mBodyScaleGroups[i].uniformScaling)
    {
      mGroupScaleIndices.emplace_back(mBodyScaleGroups.at(i), -1);
    }
    else
    {
      mGroupScaleIndices.emplace_back(mBodyScaleGroups.at(i), 0);
      mGroupScaleIndices.emplace_back(mBodyScaleGroups.at(i), 1);
      mGroupScaleIndices.emplace_back(mBodyScaleGroups.at(i), 2);
    }
  }
}

//==============================================================================
/// This grabs the details for what a group scale index corresponds to
const BodyScaleGroupAndIndex& Skeleton::getGroupScaleIndexDetails(
    int index) const
{
  return mGroupScaleIndices.at(index);
}

//==============================================================================
/// This produces a human-readable description of the group scale vector index
std::string Skeleton::debugGroupScaleIndex(int groupIdx)
{
  BodyScaleGroup* group = nullptr;
  int axis = 0;

  std::string result = std::to_string(groupIdx) + " ";
  // Find the group and axis we're talking about
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    Eigen::Vector3s localScale;
    if (mBodyScaleGroups[i].uniformScaling)
    {
      if (groupIdx == 0)
      {
        group = &mBodyScaleGroups[i];
        break;
      }
      else
      {
        groupIdx--;
      }
    }
    else
    {
      if (groupIdx < 3)
      {
        group = &mBodyScaleGroups[i];
        axis = groupIdx;
        break;
      }
      else
      {
        groupIdx -= 3;
      }
    }
  }
  assert(group != nullptr);

  if (group->uniformScaling)
  {
    result += "[uniform]";
  }
  else if (axis == 0)
  {
    result += "[X]";
  }
  else if (axis == 1)
  {
    result += "[Y]";
  }
  else if (axis == 2)
  {
    result += "[Z]";
  }
  for (dynamics::BodyNode* node : group->nodes)
  {
    result += " " + node->getName();
  }
  return result;
}

//==============================================================================
/// This returns the vector relating changing a group scale parameter to the
/// location of the center of a body.
Eigen::Vector3s Skeleton::getGroupScaleMovementOnBodyInWorldSpace(
    int groupIdx, int bodyIdx)
{
  BodyScaleGroup* group = nullptr;
  int axis = 0;

  // Find the group and axis we're talking about
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    Eigen::Vector3s localScale;
    if (mBodyScaleGroups[i].uniformScaling)
    {
      if (groupIdx == 0)
      {
        group = &mBodyScaleGroups[i];
        break;
      }
      else
      {
        groupIdx--;
      }
    }
    else
    {
      if (groupIdx < 3)
      {
        group = &mBodyScaleGroups[i];
        axis = groupIdx;
        break;
      }
      else
      {
        groupIdx -= 3;
      }
    }
  }
  assert(group != nullptr);

  Eigen::Vector3s motionSum = Eigen::Vector3s::Zero();

  const Eigen::MatrixXi& parents = getJointParentMap();

  BodyNode* target = getBodyNode(bodyIdx);
  int targetJointIdx = target->getParentJoint()->getJointIndexInSkeleton();

  // //==============================================================================
  // /// Set the scale of the child body
  // void Joint::setChildScale(Eigen::Vector3s scale)
  // {
  //   mAspectProperties.mChildScale = scale;
  //   mAspectProperties.mT_ChildBodyToJoint.translation()
  //       = mAspectProperties.mOriginalChildTranslation.cwiseProduct(scale);
  //   updateRelativeJacobian();
  //   notifyPositionUpdated();
  // }

  // //==============================================================================
  // /// Set the scale of the parent body
  // void Joint::setParentScale(Eigen::Vector3s scale)
  // {
  //   mAspectProperties.mParentScale = scale;
  //   mAspectProperties.mT_ParentBodyToJoint.translation()
  //       = mAspectProperties.mOriginalParentTranslation.cwiseProduct(scale);
  //   notifyPositionUpdated();
  // }

  // //==============================================================================
  // void Joint::updateRelativeTransform()
  // {
  //    mT = Joint::mAspectProperties.mT_ParentBodyToJoint * mQ
  //         * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
  // }

  for (int i = 0; i < group->nodes.size(); i++)
  {
    BodyNode* source = group->nodes[i];
    int sourceJointIdx = source->getParentJoint()->getJointIndexInSkeleton();

    if (source == target || parents(sourceJointIdx, targetJointIdx) == 1)
    {
      Eigen::Vector3s localT
          = source->getParentJoint()->getOriginalTransformFromChildBodyNode();
      if (!group->uniformScaling)
      {
        localT = localT.cwiseProduct(Eigen::Vector3s::Unit(axis));
      }

      Eigen::Vector3s worldParentChildT
          = source->getWorldTransform().linear() * -localT;
      motionSum += worldParentChildT;
    }

    if (parents(sourceJointIdx, targetJointIdx) == 1)
    {
      for (int j = 0; j < source->getNumChildJoints(); j++)
      {
        dynamics::Joint* childJoint = source->getChildJoint(j);
        if (childJoint == target->getParentJoint()
            || parents(childJoint->getJointIndexInSkeleton(), targetJointIdx))
        {
          Eigen::Vector3s localT
              = childJoint->getOriginalTransformFromParentBodyNode();
          if (!group->uniformScaling)
          {
            localT = localT.cwiseProduct(Eigen::Vector3s::Unit(axis));
          }
          Eigen::Vector3s worldChildParentT
              = source->getWorldTransform().linear() * localT;
          motionSum += worldChildParentT;
        }
      }
    }
  }

  return motionSum;
}

//==============================================================================
/// This returns the vector relating changing a group scale parameter to the
/// location of the center of a body.
Eigen::Vector3s Skeleton::finiteDifferenceGroupScaleMovementOnBodyInWorldSpace(
    int groupIdx, int bodyIdx)
{
  Eigen::VectorXs original = getGroupScales();
  Eigen::Vector3s result;
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector3s& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(groupIdx) += eps;
        setGroupScales(tweaked);
        perturbed = getBodyNode(bodyIdx)->getWorldTransform().translation();
        return true;
      },
      result,
      1e-3,
      true);
  return result;
}

//==============================================================================
/// This returns the vector relating changing a group scale parameter to the
/// location of the center of a body.
Eigen::Vector3s Skeleton::getGroupScaleMovementOnJointInWorldSpace(
    int groupIdx, int jointIdx)
{
  BodyScaleGroup* group = nullptr;
  int axis = 0;

  // Find the group and axis we're talking about
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    Eigen::Vector3s localScale;
    if (mBodyScaleGroups[i].uniformScaling)
    {
      if (groupIdx == 0)
      {
        group = &mBodyScaleGroups[i];
        break;
      }
      else
      {
        groupIdx--;
      }
    }
    else
    {
      if (groupIdx < 3)
      {
        group = &mBodyScaleGroups[i];
        axis = groupIdx;
        break;
      }
      else
      {
        groupIdx -= 3;
      }
    }
  }
  assert(group != nullptr);

  Eigen::Vector3s motionSum = Eigen::Vector3s::Zero();

  const Eigen::MatrixXi& parents = getJointParentMap();

  // //==============================================================================
  // /// Set the scale of the child body
  // void Joint::setChildScale(Eigen::Vector3s scale)
  // {
  //   mAspectProperties.mChildScale = scale;
  //   mAspectProperties.mT_ChildBodyToJoint.translation()
  //       = mAspectProperties.mOriginalChildTranslation.cwiseProduct(scale);
  //   updateRelativeJacobian();
  //   notifyPositionUpdated();
  // }

  // //==============================================================================
  // /// Set the scale of the parent body
  // void Joint::setParentScale(Eigen::Vector3s scale)
  // {
  //   mAspectProperties.mParentScale = scale;
  //   mAspectProperties.mT_ParentBodyToJoint.translation()
  //       = mAspectProperties.mOriginalParentTranslation.cwiseProduct(scale);
  //   notifyPositionUpdated();
  // }

  // //==============================================================================
  // void Joint::updateRelativeTransform()
  // {
  //    mT = Joint::mAspectProperties.mT_ParentBodyToJoint * mQ
  //         * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
  // }

  for (int i = 0; i < group->nodes.size(); i++)
  {
    BodyNode* source = group->nodes[i];
    int sourceJointIdx = source->getParentJoint()->getJointIndexInSkeleton();

    if (parents(sourceJointIdx, jointIdx) == 1)
    {
      Eigen::Vector3s localT
          = source->getParentJoint()->getOriginalTransformFromChildBodyNode();
      if (!group->uniformScaling)
      {
        localT = localT.cwiseProduct(Eigen::Vector3s::Unit(axis));
      }

      Eigen::Vector3s worldParentChildT
          = source->getWorldTransform().linear() * -localT;
      motionSum += worldParentChildT;
    }

    for (int j = 0; j < source->getNumChildJoints(); j++)
    {
      dynamics::Joint* childJoint = source->getChildJoint(j);
      if (childJoint->getJointIndexInSkeleton() == jointIdx
          || parents(childJoint->getJointIndexInSkeleton(), jointIdx))
      {
        Eigen::Vector3s localT
            = childJoint->getOriginalTransformFromParentBodyNode();
        if (!group->uniformScaling)
        {
          localT = localT.cwiseProduct(Eigen::Vector3s::Unit(axis));
        }
        Eigen::Vector3s worldChildParentT
            = source->getWorldTransform().linear() * localT;
        motionSum += worldChildParentT;
      }
    }
  }

  return motionSum;
}

//==============================================================================
/// This returns the vector relating changing a group scale parameter to the
/// location of the center of a body.
Eigen::Vector3s Skeleton::finiteDifferenceGroupScaleMovementOnJointInWorldSpace(
    int groupIdx, int jointIdx)
{
  Eigen::VectorXs original = getGroupScales();
  Eigen::Vector3s result;
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector3s& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(groupIdx) += eps;
        setGroupScales(tweaked);
        perturbed = getJointWorldPosition(jointIdx);
        return true;
      },
      result,
      1e-3,
      true);
  return result;
}

//==============================================================================
/// This sets the scales of all the body nodes according to their group
/// membership. The `scale` vector is expected to be 3 times the size of the
/// number of groups.
void Skeleton::setGroupScales(Eigen::VectorXs scale, bool silentlyClamp)
{
  ensureBodyScaleGroups();
  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    Eigen::Vector3s localScale;
    if (mBodyScaleGroups[i].uniformScaling)
    {
      localScale.setConstant(scale[cursor]);
      cursor++;
    }
    else
    {
      localScale = scale.segment<3>(cursor);
      cursor += 3;
    }
    for (dynamics::BodyNode* node : mBodyScaleGroups[i].nodes)
    {
      node->setScale(localScale, silentlyClamp);
    }
  }
  assert(cursor == scale.size());
}

//==============================================================================
/// This gets the scales of the first body in each scale group.
Eigen::VectorXs Skeleton::getGroupScales()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getGroupScaleDim());
  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    if (mBodyScaleGroups[i].uniformScaling)
    {
      groups(cursor) = mBodyScaleGroups[i].nodes[0]->getScale()(0);
      cursor++;
    }
    else
    {
      groups.segment<3>(cursor) = mBodyScaleGroups[i].nodes[0]->getScale();
      cursor += 3;
    }
  }
  assert(cursor == groups.size());
  return groups;
}

//==============================================================================
/// This converts a map of body scales back into group scales, interpreting
/// everything as gradients.
Eigen::VectorXs Skeleton::getGroupScaleGradientsFromMap(
    std::map<std::string, Eigen::Vector3s> bodyScales)
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getGroupScaleDim());
  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    if (mBodyScaleGroups[i].uniformScaling)
    {
      for (auto node : mBodyScaleGroups[i].nodes)
      {
        groups(cursor) += bodyScales[node->getName()].sum();
      }
      cursor++;
    }
    else
    {
      for (auto node : mBodyScaleGroups[i].nodes)
      {
        groups.segment<3>(cursor) += bodyScales[node->getName()];
      }
      cursor += 3;
    }
  }
  assert(cursor == groups.size());
  return groups;
}

//==============================================================================
/// This returns the upper bound values for each index in the group scales
/// vector
Eigen::VectorXs Skeleton::getGroupScalesUpperBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getGroupScaleDim());
  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    if (mBodyScaleGroups[i].uniformScaling)
    {
      groups(cursor) = getScaleGroupUpperBound(i)(0);
      cursor++;
    }
    else
    {
      groups.segment<3>(cursor) = getScaleGroupUpperBound(i);
      cursor += 3;
    }
  }
  assert(cursor == groups.size());
  return groups;
}

//==============================================================================
/// This returns the upper bound values for each index in the group scales
/// vector
Eigen::VectorXs Skeleton::getGroupScalesLowerBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getGroupScaleDim());
  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    if (mBodyScaleGroups[i].uniformScaling)
    {
      groups(cursor) = getScaleGroupLowerBound(i)(0);
      cursor++;
    }
    else
    {
      groups.segment<3>(cursor) = getScaleGroupLowerBound(i);
      cursor += 3;
    }
  }
  assert(cursor == groups.size());
  return groups;
}

//==============================================================================
/// This gets the masses of each scale group, concatenated
Eigen::VectorXs Skeleton::getGroupMasses()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups());
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups(i) = mBodyScaleGroups[i].nodes[0]->getMass();
  }
  return groups;
}

//==============================================================================
/// This sets the masses of each scale group, concatenated
void Skeleton::setGroupMasses(Eigen::VectorXs masses)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    for (auto* node : mBodyScaleGroups[i].nodes)
    {
      node->setMass(masses(i));
    }
  }
}

//==============================================================================
/// This gets the upper bound for each group's mass, concatenated
Eigen::VectorXs Skeleton::getGroupMassesUpperBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups());
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups(i) = mBodyScaleGroups[i].nodes[0]->getInertia().getMassUpperBound();
  }
  return groups;
}

//==============================================================================
/// This gets the lower bound for each group's mass, concatenated
Eigen::VectorXs Skeleton::getGroupMassesLowerBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups());
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups(i) = mBodyScaleGroups[i].nodes[0]->getInertia().getMassLowerBound();
  }
  return groups;
}

//==============================================================================
/// This gets a vector of [(1/m), p_0, ..., p_n], where the first entry is the
/// inverse of the total mass of the skeleton, and all subsequent entries are
/// the percentages of the total for each link mass.
Eigen::VectorXs Skeleton::getLinearizedMasses()
{
  ensureBodyScaleGroups();
  s_t totalMass = getMass();
  Eigen::VectorXs linearized
      = Eigen::VectorXs::Zero(mBodyScaleGroups.size() + 1);
  linearized(0) = 1.0 / totalMass;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    s_t totalGroupMass = mBodyScaleGroups[i].nodes[0]->getMass()
                         * mBodyScaleGroups[i].nodes.size();
    linearized(i + 1) = totalGroupMass / totalMass;
  }
  return linearized;
}

//==============================================================================
/// This sets a vector of [(1/m), p_0, ..., p_n], where the first entry is the
/// inverse of the total mass of the skeleton, and all subsequent entries are
/// the percentages of the total for each link mass, and maps back into the
/// mass list.
void Skeleton::setLinearizedMasses(Eigen::VectorXs masses)
{
  ensureBodyScaleGroups();
  s_t totalMass = 1.0 / masses(0);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    s_t totalGroupMass = masses(i + 1) * totalMass;
    for (auto* body : mBodyScaleGroups[i].nodes)
    {
      body->setMass(totalGroupMass / mBodyScaleGroups[i].nodes.size());
    }
  }
}

//==============================================================================
/// This gets the upper bound for the linearized mass, concatenated
Eigen::VectorXs Skeleton::getLinearizedMassesUpperBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs linearized
      = Eigen::VectorXs::Ones(mBodyScaleGroups.size() + 1);
  s_t minimumMass = 0.0;
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    minimumMass += getBodyNode(i)->getInertia().getMassLowerBound();
  }
  linearized(0) = 1.0 / minimumMass;
  return linearized;
}

//==============================================================================
/// This gets the lower bound for the linearized mass, concatenated
Eigen::VectorXs Skeleton::getLinearizedMassesLowerBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs linearized
      = Eigen::VectorXs::Zero(mBodyScaleGroups.size() + 1);
  s_t maxMass = 0.0;
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    maxMass += getBodyNode(i)->getInertia().getMassUpperBound();
  }
  linearized(0) = 1.0 / maxMass;
  return linearized;
}

//==============================================================================
/// This maps a linearized masses vector to an "unnormalized COM." This means
/// that the percentages in the linearized mass vector need not add to 1.
/// Relaxing this constraint, and then applying it later in the optimizer, can
/// make certain dynamics problems into linear.
Eigen::Vector3s Skeleton::getUnnormalizedCOM(Eigen::VectorXs linearizedMasses)
{
  Eigen::Vector3s unnormalizedCOM = Eigen::Vector3s::Zero();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    s_t percentage = linearizedMasses(i + 1);
    s_t bodyPercentage = percentage / mBodyScaleGroups[i].nodes.size();
    for (auto* body : mBodyScaleGroups[i].nodes)
    {
      unnormalizedCOM += bodyPercentage * body->getCOM();
    }
  }
  return unnormalizedCOM;
}

//==============================================================================
/// This maps a linearized masses vector to an "unnormalized COM
/// acceleration." This means that the percentages in the linearized mass
/// vector need not add to 1. Relaxing this constraint, and then applying it
/// later in the optimizer, can make certain dynamics problems into linear.
Eigen::Vector3s Skeleton::getUnnormalizedCOMAcceleration(
    Eigen::VectorXs linearizedMasses)
{
  Eigen::Vector3s unnormalizedCOMAcc = Eigen::Vector3s::Zero();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    s_t percentage = linearizedMasses(i + 1);
    s_t bodyPercentage = percentage / mBodyScaleGroups[i].nodes.size();
    for (auto* body : mBodyScaleGroups[i].nodes)
    {
      unnormalizedCOMAcc += bodyPercentage * body->getCOMLinearAcceleration();
    }
  }
  return unnormalizedCOMAcc;
}

//==============================================================================
/// This maps a linearized masses vector to an "unnormalized COM
/// acceleration." This means that the percentages in the linearized mass
/// vector need not add to 1. Relaxing this constraint, and then applying it
/// later in the optimizer, can make certain dynamics problems into linear.
Eigen::Vector3s Skeleton::getUnnormalizedCOMFDAcceleration(
    Eigen::VectorXs linearizedMasses)
{
  Eigen::VectorXs p = getPositions();
  Eigen::VectorXs v = getVelocities();
  Eigen::VectorXs a = getAccelerations();
  s_t dt = getTimeStep();
  Eigen::VectorXs p_last = p - (dt * v);
  Eigen::VectorXs p_next = (dt * dt * a) - p_last + (2 * p);

  setPositions(p);
  Eigen::Vector3s com = getUnnormalizedCOM(linearizedMasses);
  setPositions(p_next);
  Eigen::Vector3s com_next = getUnnormalizedCOM(linearizedMasses);
  setPositions(p_last);
  Eigen::Vector3s com_last = getUnnormalizedCOM(linearizedMasses);
  setPositions(p);

  return (com_last - 2 * com + com_next) / (dt * dt);
}

//==============================================================================
/// This maps to the difference for unnormalized (analytical COM acc - fd COM
/// acc)
Eigen::Vector3s Skeleton::getUnnormalizedCOMAccelerationOffset(
    Eigen::VectorXs linearizedMasses)
{
  return getUnnormalizedCOMAcceleration(linearizedMasses)
         - getUnnormalizedCOMFDAcceleration(linearizedMasses);
}

//==============================================================================
/// This gets the analytical jacobian relating the linearized masses to link
/// masses
Eigen::MatrixXs Skeleton::getGroupMassesJacobianWrtLinearizedMasses()
{
  ensureBodyScaleGroups();
  int numGroups = mBodyScaleGroups.size();
  s_t totalMass = getMass();
  s_t inverseMass = 1.0 / totalMass;
  (void)inverseMass;
  Eigen::VectorXs linearized = getLinearizedMasses();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(numGroups, 1 + numGroups);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    result(i, 0) = -(linearized(i + 1) / mBodyScaleGroups[i].nodes.size())
                   / (inverseMass * inverseMass);
    result(i, i + 1) = totalMass / mBodyScaleGroups[i].nodes.size();
  }
  return result;
}

//==============================================================================
/// This gets the finite difference'd jacobian relating the linearized masses
/// to link masses
Eigen::MatrixXs
Skeleton::finiteDifferenceGroupMassesJacobianWrtLinearizedMasses()
{
  ensureBodyScaleGroups();
  int numGroups = mBodyScaleGroups.size();
  Eigen::MatrixXs result(numGroups, 1 + numGroups);
  Eigen::VectorXs original = getLinearizedMasses();

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        setLinearizedMasses(tweaked);
        perturbed = getGroupMasses();
        return true;
      },
      result,
      eps,
      false);

  // Reset everything how we left it
  setLinearizedMasses(original);

  return result;
}

//==============================================================================
/// This gets the analytical jacobian relating changes in the linearized
/// masses to changes in the COM position
Eigen::MatrixXs Skeleton::getUnnormalizedCOMJacobianWrtLinearizedMasses()
{
  ensureBodyScaleGroups();
  int numGroups = mBodyScaleGroups.size();
  Eigen::VectorXs linearized = getLinearizedMasses();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(3, 1 + numGroups);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    // s_t percentage = linearized(i + 1);
    // s_t bodyPercentage = percentage / mBodyScaleGroups[i].nodes.size();
    for (auto* body : mBodyScaleGroups[i].nodes)
    {
      result.col(i + 1) += body->getCOM() / mBodyScaleGroups[i].nodes.size();
    }
  }
  return result;
}

//==============================================================================
/// This gets the analytical jacobian relating changes in the linearized
/// masses to changes in the COM position
Eigen::MatrixXs
Skeleton::finiteDifferenceUnnormalizedCOMJacobianWrtLinearizedMasses()
{
  ensureBodyScaleGroups();
  int numGroups = mBodyScaleGroups.size();
  Eigen::MatrixXs result(3, 1 + numGroups);
  Eigen::VectorXs original = getLinearizedMasses();

  s_t eps = 1e-3;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        perturbed = getUnnormalizedCOM(tweaked);
        return true;
      },
      result,
      eps,
      true);

  // Reset everything how we left it
  setLinearizedMasses(original);

  return result;
}

//==============================================================================
/// This relates changes to the linearized masses to changes to the finite
/// difference'd formula for COM acceleration, (c[t-1] - 2*c[t] +
/// c[t+1])/(dt*dt)
Eigen::MatrixXs Skeleton::getUnnormalizedCOMFDAccJacobianWrtLinearizedMasses()
{
  Eigen::VectorXs p = getPositions();
  Eigen::VectorXs v = getVelocities();
  Eigen::VectorXs a = getAccelerations();
  s_t dt = getTimeStep();
  // v[t] = (p[t] - p[t-1])/dt
  // p[t-1] = p[t] - dt*v[t]
  // a[t] = (p[t-1] - 2*p[t] + p[t+1])/(dt*dt)
  // dt*dt*a[t] = (p[t-1] - 2*p[t] + p[t+1])
  // dt*dt*a[t] - p[t-1] + 2*p[t] = p[t+1]
  Eigen::VectorXs p_last = p - (dt * v);
  Eigen::VectorXs p_next = (dt * dt * a) - p_last + (2 * p);

  Eigen::MatrixXs comJ = getUnnormalizedCOMJacobianWrtLinearizedMasses();
  setPositions(p_last);
  Eigen::MatrixXs comJ_last = getUnnormalizedCOMJacobianWrtLinearizedMasses();
  setPositions(p_next);
  Eigen::MatrixXs comJ_next = getUnnormalizedCOMJacobianWrtLinearizedMasses();
  setPositions(p);

  return (comJ_last - 2 * comJ + comJ_next) / (dt * dt);
}

//==============================================================================
/// This relates changes to the linearized masses to changes to the finite
/// difference'd formula for COM acceleration, (c[t-1] - 2*c[t] +
/// c[t+1])/(dt*dt)
Eigen::MatrixXs
Skeleton::finiteDifferenceUnnormalizedCOMFDAccJacobianWrtLinearizedMasses()
{
  int numGroups = mBodyScaleGroups.size();
  Eigen::MatrixXs result(3, 1 + numGroups);
  Eigen::VectorXs original = getLinearizedMasses();

  s_t eps = 1e-3;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;

        perturbed = getUnnormalizedCOMFDAcceleration(tweaked);

        return true;
      },
      result,
      eps,
      true);

  // Reset everything how we left it
  setLinearizedMasses(original);

  return result;
}

//==============================================================================
/// This relates changes to the linearized masses to changes to the analytical
/// formula for COM acceleration
Eigen::MatrixXs
Skeleton::getUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses()
{
  int numGroups = mBodyScaleGroups.size();
  Eigen::VectorXs linearized = getLinearizedMasses();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(3, 1 + numGroups);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    // s_t percentage = linearized(i + 1);
    // s_t bodyPercentage = percentage / mBodyScaleGroups[i].nodes.size();
    for (auto* body : mBodyScaleGroups[i].nodes)
    {
      result.col(i + 1) += body->getCOMLinearAcceleration()
                           / mBodyScaleGroups[i].nodes.size();
    }
  }
  return result;
}

//==============================================================================
/// This relates changes to the linearized masses to changes to the analytical
/// formula for COM acceleration
Eigen::MatrixXs Skeleton::
    finiteDifferenceUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses()
{
  int numGroups = mBodyScaleGroups.size();
  Eigen::MatrixXs result(3, 1 + numGroups);
  Eigen::VectorXs original = getLinearizedMasses();

  s_t eps = 1e-3;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;

        perturbed = getUnnormalizedCOMAcceleration(tweaked);

        return true;
      },
      result,
      eps,
      true);

  return result;
}

//==============================================================================
/// This relates changes to the linearized masses to changes in the
/// unnormalized (analytical COM acc - fd COM acc) quantity.
Eigen::MatrixXs
Skeleton::getUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses()
{
  return getUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses()
         - getUnnormalizedCOMFDAccJacobianWrtLinearizedMasses();
}

//==============================================================================
/// This relates changes to the linearized masses to changes in the
/// unnormalized (analytical COM acc - fd COM acc) quantity.
Eigen::MatrixXs Skeleton::
    finiteDifferenceUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses()
{
  int numGroups = mBodyScaleGroups.size();
  Eigen::MatrixXs result(3, 1 + numGroups);
  Eigen::VectorXs original = getLinearizedMasses();

  s_t eps = 1e-3;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;

        perturbed = getUnnormalizedCOMAccelerationOffset(tweaked);

        return true;
      },
      result,
      eps,
      true);

  return result;
}

//==============================================================================
/// This gets the COMs of each scale group, concatenated
Eigen::VectorXs Skeleton::getGroupCOMs()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 3);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups.segment<3>(i * 3)
        = mBodyScaleGroups[i].nodes[0]->getInertia().getLocalCOM().cwiseProduct(
            mBodyScaleGroups[i].flipAxis[0]);
  }
  return groups;
}

//==============================================================================
/// This gets the upper bound for each axis of each group's COM, concatenated
Eigen::VectorXs Skeleton::getGroupCOMUpperBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 3);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups.segment<3>(i * 3)
        = mBodyScaleGroups[i].nodes[0]->getInertia().getLocalCOMUpperBound();
  }
  return groups;
}

//==============================================================================
/// This gets the lower bound for each axis of each group's COM, concatenated
Eigen::VectorXs Skeleton::getGroupCOMLowerBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 3);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups.segment<3>(i * 3)
        = mBodyScaleGroups[i].nodes[0]->getInertia().getLocalCOMLowerBound();
  }
  return groups;
}

//==============================================================================
/// This sets the COMs of each scale group, concatenated
void Skeleton::setGroupCOMs(Eigen::VectorXs coms)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    for (int j = 0; j < mBodyScaleGroups[i].nodes.size(); j++)
    {
      auto* node = mBodyScaleGroups[i].nodes[j];
      node->setLocalCOM(
          coms.segment<3>(i * 3).cwiseProduct(mBodyScaleGroups[i].flipAxis[j]));
    }
  }
}

//==============================================================================
/// This gets the Inertias of each scale group (the 6 vector), concatenated
Eigen::VectorXs Skeleton::getGroupInertias()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 6);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    Eigen::Vector3s flips = mBodyScaleGroups[i].flipAxis[0];
    Eigen::Vector6s dimsAndEulers
        = mBodyScaleGroups[i].nodes[0]->getInertia().getDimsAndEulerVector();
    // If we are flipping the Z axis, we want to negate the X and Y rotations,
    // but leave Z the same
    if (flips(2) < 0)
    {
      dimsAndEulers(3 + 0) *= -1;
      dimsAndEulers(3 + 1) *= -1;
    }
    groups.segment<6>(i * 6) = dimsAndEulers;
  }
  return groups;
}

//==============================================================================
/// This sets the Inertias of each scale group (the 6 vector), concatenated
void Skeleton::setGroupInertias(Eigen::VectorXs inertias)
{
  ensureBodyScaleGroups();
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    for (int j = 0; j < mBodyScaleGroups[i].nodes.size(); j++)
    {
      auto* node = mBodyScaleGroups[i].nodes[j];
      Eigen::Vector3s flips = mBodyScaleGroups[i].flipAxis[j];
      Eigen::Vector6s dimsAndEulers = inertias.segment<6>(i * 6);
      // If we are flipping the Z axis, we want to negate the X and Y rotations,
      // but leave Z the same
      if (flips(2) < 0)
      {
        dimsAndEulers(3 + 0) *= -1;
        dimsAndEulers(3 + 1) *= -1;
      }
      node->setDimsAndEulersVector(dimsAndEulers);
    }
  }
}

//==============================================================================
/// This gets the upper bound for each axis of each group's inertias,
/// concatenated
Eigen::VectorXs Skeleton::getGroupInertiasUpperBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 6);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups.segment<6>(i * 6) = mBodyScaleGroups[i]
                                   .nodes[0]
                                   ->getInertia()
                                   .getDimsAndEulerUpperBound();
  }
  return groups;
}

//==============================================================================
/// This gets the lower bound for each axis of each group's inertias,
/// concatenated
Eigen::VectorXs Skeleton::getGroupInertiasLowerBound()
{
  ensureBodyScaleGroups();
  Eigen::VectorXs groups = Eigen::VectorXs::Zero(getNumScaleGroups() * 6);
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    assert(mBodyScaleGroups[i].nodes.size() > 0);
    groups.segment<6>(i * 6) = mBodyScaleGroups[i]
                                   .nodes[0]
                                   ->getInertia()
                                   .getDimsAndEulerLowerBound();
  }
  return groups;
}

//==============================================================================
/// This is a general purpose utility to convert a Gradient wrt Body scales to
/// one wrt Group scales
Eigen::VectorXs Skeleton::convertBodyScalesGradientToGroupScales(
    Eigen::VectorXs bodyScalesGrad)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(getGroupScaleDim());

  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (dynamics::BodyNode* node : mBodyScaleGroups[i].nodes)
    {
      if (mBodyScaleGroups[i].uniformScaling)
      {
        grad(cursor) += bodyScalesGrad(node->getIndexInSkeleton() * 3)
                        + bodyScalesGrad(node->getIndexInSkeleton() * 3 + 1)
                        + bodyScalesGrad(node->getIndexInSkeleton() * 3 + 2);
      }
      else
      {
        grad.segment<3>(cursor)
            += bodyScalesGrad.segment<3>(node->getIndexInSkeleton() * 3);
      }
    }
    if (mBodyScaleGroups[i].uniformScaling)
    {
      cursor++;
    }
    else
    {
      cursor += 3;
    }
  }

  assert(cursor == grad.size());

  return grad;
}

//==============================================================================
/// This is a general purpose utility to convert a Jacobian wrt Body scales to
/// one wrt Group scales
Eigen::MatrixXs Skeleton::convertBodyScalesJacobianToGroupScales(
    Eigen::MatrixXs individualBodiesJac)
{
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(individualBodiesJac.rows(), getGroupScaleDim());

  int cursor = 0;
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    for (dynamics::BodyNode* node : mBodyScaleGroups[i].nodes)
    {
      if (mBodyScaleGroups[i].uniformScaling)
      {
        J.col(cursor)
            += individualBodiesJac.col(node->getIndexInSkeleton() * 3)
               + individualBodiesJac.col(node->getIndexInSkeleton() * 3 + 1)
               + individualBodiesJac.col(node->getIndexInSkeleton() * 3 + 2);
      }
      else
      {
        J.block(0, cursor, J.rows(), 3) += individualBodiesJac.block(
            0, node->getIndexInSkeleton() * 3, J.rows(), 3);
      }
    }
    if (mBodyScaleGroups[i].uniformScaling)
    {
      cursor++;
    }
    else
    {
      cursor += 3;
    }
  }

  assert(cursor == J.cols());

  return J;
}

//==============================================================================
/// This returns the Jacobian of the joint positions wrt the scales of the
/// groups
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtGroupScales(
    const std::vector<dynamics::Joint*>& joints)
{
  return convertBodyScalesJacobianToGroupScales(
      getJointWorldPositionsJacobianWrtBodyScales(joints));
}

//==============================================================================
/// This returns the Jacobian of the joint positions wrt the scales of the
/// groups
Eigen::MatrixXs
Skeleton::finiteDifferenceJointWorldPositionsJacobianWrtGroupScales(
    const std::vector<dynamics::Joint*>& joints)
{
  Eigen::MatrixXs result(joints.size() * 3, getGroupScaleDim());
  Eigen::VectorXs original = getGroupScales();

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        setGroupScales(tweaked);
        perturbed = getJointWorldPositions(joints);
        return true;
      },
      result,
      eps,
      false);

  // Reset everything how we left it
  setGroupScales(original);

  return result;
}

//==============================================================================
/// This returns the Jacobian relating changes in body scales to changes in
/// marker world positions.
Eigen::MatrixXs Skeleton::getMarkerWorldPositionsJacobianWrtGroupScales(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  return convertBodyScalesJacobianToGroupScales(
      getMarkerWorldPositionsJacobianWrtBodyScales(markers));
}

//==============================================================================
/// This returns the Jacobian relating changes in body scales to changes in
/// marker world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceMarkerWorldPositionsJacobianWrtGroupScales(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, getGroupScaleDim());

  Eigen::VectorXs original = getGroupScales();

  const double EPS = 1e-7;
  for (int i = 0; i < getGroupScaleDim(); i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    setGroupScales(perturbed);
    Eigen::VectorXs plus = getMarkerWorldPositions(markers);

    perturbed = original;
    perturbed(i) -= EPS;
    setGroupScales(perturbed);
    Eigen::VectorXs minus = getMarkerWorldPositions(markers);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  setGroupScales(original);

  return jac;
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to group
/// scales
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsSecondJacobianWrtJointWrtGroupScales(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs leftMultiply)
{
  return convertBodyScalesJacobianToGroupScales(
      getMarkerWorldPositionsSecondJacobianWrtJointWrtBodyScale(
          markers, leftMultiply));
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to group
/// scales
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtGroupScales(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(getNumDofs(), getGroupScaleDim());

  const s_t EPS = 1e-7;

  Eigen::VectorXs original = getGroupScales();
  for (int i = 0; i < getGroupScaleDim(); i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    setGroupScales(perturbed);

    Eigen::VectorXs plus
        = leftMultiply.transpose()
          * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

    perturbed = original;
    perturbed(i) -= EPS;
    setGroupScales(perturbed);
    Eigen::VectorXs minus
        = leftMultiply.transpose()
          * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

    result.col(i) = (plus - minus) / (2 * EPS);
  }

  setGroupScales(original);

  return result;
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// group scales
Eigen::VectorXs Skeleton::getGradientOfDistanceWrtGroupScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB)
{
  return convertBodyScalesGradientToGroupScales(
      getGradientOfDistanceWrtBodyScales(markerA, markerB));
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// group scales
Eigen::VectorXs Skeleton::finiteDifferenceGradientOfDistanceWrtGroupScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB)
{
  Eigen::VectorXs originalScales = getGroupScales();

  Eigen::VectorXs result = Eigen::VectorXs::Zero(originalScales.size());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs perturbed = originalScales;
        perturbed(i) += eps;
        setGroupScales(perturbed);

        out = getDistanceInWorldSpace(markerA, markerB);

        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// group scales
Eigen::VectorXs Skeleton::getGradientOfDistanceAlongAxisWrtGroupScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
    Eigen::Vector3s axis)
{
  return convertBodyScalesGradientToGroupScales(
      getGradientOfDistanceAlongAxisWrtBodyScales(markerA, markerB, axis));
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// group scales
Eigen::VectorXs
Skeleton::finiteDifferenceGradientOfDistanceAlongAxisWrtGroupScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
    Eigen::Vector3s axis)
{
  Eigen::VectorXs originalScales = getGroupScales();

  Eigen::VectorXs result = Eigen::VectorXs::Zero(originalScales.size());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs perturbed = originalScales;
        perturbed(i) += eps;
        setGroupScales(perturbed);

        out = getDistanceAlongAxis(markerA, markerB, axis);

        return true;
      },
      result,
      eps,
      useRidders);

  return result;
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
      for (int i = 3; i < 6; i++)
      {
        pair.first->setPositionUpperLimit(i, joint->getPositionUpperLimit(i));
        pair.first->setPositionLowerLimit(i, joint->getPositionLowerLimit(i));
      }
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
      pair.first->setPositionUpperLimits(
          body->getParentJoint()->getPositionUpperLimits());
      pair.first->setPositionLowerLimits(
          body->getParentJoint()->getPositionLowerLimits());
      assert(
          pair.first->getPositionUpperLimits()
          == body->getParentJoint()->getPositionUpperLimits());
      assert(
          pair.first->getPositionLowerLimits()
          == body->getParentJoint()->getPositionLowerLimits());
    }
  }

  // Copy the groups
  for (int i = 0; i < mBodyScaleGroups.size(); i++)
  {
    copy->mBodyScaleGroups.emplace_back();
    BodyScaleGroup& group = copy->mBodyScaleGroups.back();
    group.uniformScaling = mBodyScaleGroups[i].uniformScaling;
    for (auto body : mBodyScaleGroups[i].nodes)
    {
      group.nodes.push_back(copy->getBodyNode(body->getName()));
    }
    for (auto flips : mBodyScaleGroups[i].flipAxis)
    {
      group.flipAxis.push_back(flips);
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
    if (getJoint(i)->getType() == RevoluteJoint::getStaticType())
    {
      dynamics::RevoluteJoint* ourJoint
          = static_cast<dynamics::RevoluteJoint*>(getJoint(i));
      dynamics::RevoluteJoint* copyJoint
          = static_cast<dynamics::RevoluteJoint*>(joint);
#ifndef NDEBUG
      Eigen::Vector3s ourAxis = ourJoint->getAxis();
      Eigen::Vector3s copyAxis = copyJoint->getAxis();
      assert((ourAxis - copyAxis).norm() < 1e-15);
#endif
    }
    if (getJoint(i)->getType() == CustomJoint<2>::getStaticType())
    {
      dynamics::CustomJoint<2>* ourJoint
          = static_cast<dynamics::CustomJoint<2>*>(getJoint(i));
      dynamics::CustomJoint<2>* copyJoint
          = static_cast<dynamics::CustomJoint<2>*>(joint);
      assert(ourJoint->getFlipAxisMap() == copyJoint->getFlipAxisMap());
      for (int j = 0; j < 6; j++)
      {
        assert(
            ourJoint->getCustomFunctionDrivenByDof(j)
            == copyJoint->getCustomFunctionDrivenByDof(j));
      }
    }
    // All the joint limits must match except the eulerian joints
    if (getJoint(i)->getType() != EulerFreeJoint::getStaticType()
        && getJoint(i)->getType() != EulerJoint::getStaticType())
    {
      std::string jointName = joint->getName();
      (void)jointName;
      Eigen::VectorXs copyUpperLimits = joint->getPositionUpperLimits();
      Eigen::VectorXs ourUpperLimits = getJoint(i)->getPositionUpperLimits();
      assert(copyUpperLimits == ourUpperLimits);
      assert(
          joint->getPositionLowerLimits()
          == getJoint(i)->getPositionLowerLimits());
    }
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
      // Do our best to pick an equivalent set of EulerAngles that's within
      // joint bounds, if one exists
      translated.segment<3>(cursor) = math::attemptToClampEulerAnglesToBounds(
          translated.segment<3>(cursor),
          eulerFreeJoint->getPositionUpperLimits().head<3>(),
          eulerFreeJoint->getPositionLowerLimits().head<3>());
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
      // Do our best to pick an equivalent set of EulerAngles that's within
      // joint bounds, if one exists
      translated.segment<3>(cursor) = math::attemptToClampEulerAnglesToBounds(
          translated.segment<3>(cursor),
          eulerJoint->getPositionUpperLimits(),
          eulerJoint->getPositionLowerLimits());
    }
    cursor += joint->getNumDofs();
  }
  return translated;
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered source joints.
Eigen::VectorXs Skeleton::getJointWorldPositions(
    const std::vector<dynamics::Joint*>& joints) const
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
/// This returns the world position of a joint
Eigen::Vector3s Skeleton::getJointWorldPosition(int idx) const
{
  return (getJoint(idx)->getChildBodyNode()->getWorldTransform()
          * getJoint(idx)->getTransformFromChildBodyNode())
      .translation();
}

//==============================================================================
/// This returns a map with the world positions of each joint, keyed by joint
/// name
std::map<std::string, Eigen::Vector3s> Skeleton::getJointWorldPositionsMap()
    const
{
  std::map<std::string, Eigen::Vector3s> result;
  for (int i = 0; i < getNumJoints(); i++)
  {
    result[getJoint(i)->getName()] = getJointWorldPosition(i);
  }
  return result;
}

//==============================================================================
/// This returns the concatenated 3-vectors for world angle of each joint's
/// child space in 3D world space, for the registered joints.
Eigen::VectorXs Skeleton::getJointWorldAngles(
    const std::vector<dynamics::Joint*>& joints) const
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
    const std::vector<dynamics::Joint*>& joints) const
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
    const std::vector<dynamics::Joint*>& joints)
{
  Eigen::MatrixXs result(joints.size() * 3, getNumDofs());
  Eigen::VectorXs originalPos = getPositions();

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedPos = originalPos;
        tweakedPos(dof) += eps;
        setPositions(tweakedPos);
        perturbed = getJointWorldPositions(joints);
        return true;
      },
      result,
      eps,
      false);

  // Reset everything how we left it
  setPositions(originalPos);

  return result;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtJointChildAngles(
    const std::vector<dynamics::Joint*>& joints) const
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
    const std::vector<dynamics::Joint*>& joints)
{
  Eigen::MatrixXs result(joints.size() * 3, getNumDofs());
  Eigen::VectorXs originalPos = getPositions();

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedPos = originalPos;
        tweakedPos(dof) += eps;
        setPositions(tweakedPos);
        perturbed = getJointWorldAngles(joints);
        return true;
      },
      result,
      eps,
      false);

  // Reset everything how we left it
  setPositions(originalPos);

  return result;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton body scales
/// to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getJointWorldPositionsJacobianWrtBodyScales(
    const std::vector<dynamics::Joint*>& joints)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(joints.size() * 3, getNumBodyNodes() * 3);

  const Eigen::MatrixXi& parentMap = getJointParentMap();
  // Scaling a body will cause the joint offsets to scale, which will move the
  // downstream joint positions by those vectors
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* bodyNode = getBodyNode(i);

    // Each body can scale along 3 distinct axis
    for (int axis = 0; axis < 3; axis++)
    {
      Eigen::Vector3s worldOffsetDueToParentJoint
          = bodyNode->getParentJoint()
                ->getWorldTranslationOfChildBodyWrtChildScale(axis);

      // Now begin iterating rows, for each joint
      for (int j = 0; j < joints.size(); j++)
      {
        int sourceJointIndex = getJointIndex(joints[j]);
        if (joints[j]->getName() == bodyNode->getParentJoint()->getName())
        {
          jac.block(j * 3, i * 3 + axis, 3, 1)
              = worldOffsetDueToParentJoint
                // Ignore any translation of the child body due to simply
                // scaling the child offset, since the joint isn't attached to
                // the child offset
                - bodyNode->getParentJoint()
                      ->Joint::getWorldTranslationOfChildBodyWrtChildScale(
                          axis);
        }
        else
        {
          for (int k = 0; k < bodyNode->getNumChildJoints(); k++)
          {
            dynamics::Joint* childJoint = bodyNode->getChildJoint(k);
            if (childJoint->getName() == joints[j]->getName()
                || parentMap(getJointIndex(childJoint), sourceJointIndex))
            {
              Eigen::Vector3s worldOffsetDueToChildJoint
                  = childJoint->getWorldTranslationOfChildBodyWrtParentScale(
                      axis);

              jac.block(j * 3, i * 3 + axis, 3, 1)
                  = worldOffsetDueToParentJoint + worldOffsetDueToChildJoint;

              break;
            }
          }
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
    const std::vector<dynamics::Joint*>& joints)
{
  Eigen::MatrixXs result(joints.size() * 3, getNumBodyNodes() * 3);

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        int body = dof / 3;
        int axis = dof - (body * 3);
        Eigen::Vector3s originalScale = getBodyNode(body)->getScale();
        Eigen::Vector3s perturbedScale = originalScale;
        perturbedScale(axis) += eps;
        getBodyNode(body)->setScale(perturbedScale);
        perturbed = getJointWorldPositions(joints);
        getBodyNode(body)->setScale(originalScale);
        return true;
      },
      result,
      eps,
      false);

  return result;
}

//==============================================================================
/// These are a set of bodies, and offsets in local body space where markers
/// are mounted on the body
std::map<std::string, Eigen::Vector3s> Skeleton::getMarkerMapWorldPositions(
    const MarkerMap& markers)
{
  std::map<std::string, Eigen::Vector3s> returnMap;
  for (auto markerPair : markers)
  {
    returnMap[markerPair.first]
        = markerPair.second.first->getWorldTransform()
          * markerPair.second.first->getScale().cwiseProduct(
              markerPair.second.second);
  }
  return returnMap;
}

//==============================================================================
/// This converts markers from a source skeleton to the current, doing a
/// simple mapping based on body node names. Any markers that don't find a
/// body node in the current skeleton with the same name are dropped.
std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
Skeleton::convertMarkerMap(
    const std::map<
        std::string,
        std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markerMap,
    bool warnOnDrop)
{
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>> result;
  for (auto pair : markerMap)
  {
    dynamics::BodyNode* node = getBodyNode(pair.second.first->getName());
    if (node != nullptr)
    {
      result[pair.first] = std::make_pair(
          node,
          Eigen::Vector3s(pair.second.second)
              .cwiseQuotient(pair.second.first->getScale()));
    }
    else if (warnOnDrop)
    {
      std::cout
          << "WARNING: marker \"" << pair.first
          << "\" attaches to a node named \"" << pair.second.first->getName()
          << "\", for which there is no equivalent in the current skeleton. "
             "This marker will be dropped in the converted markers map."
          << std::endl;
    }
  }
  return result;
}

//==============================================================================
/// These are a set of bodies, and offsets in local body space where markers
/// are mounted on the body
Eigen::VectorXs Skeleton::getMarkerWorldPositions(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::VectorXs positions = Eigen::VectorXs::Zero(markers.size() * 3);
  for (int i = 0; i < markers.size(); i++)
  {
    positions.segment<3>(i * 3)
        = markers[i].first->getWorldTransform()
          * markers[i].first->getScale().cwiseProduct(markers[i].second);
  }
  return positions;
}

//==============================================================================
/// This returns the Jacobian relating changes in source skeleton joint
/// positions to changes in source joint world positions.
Eigen::MatrixXs Skeleton::getMarkerWorldPositionsJacobianWrtJointPositions(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
    const
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  for (int i = 0; i < markers.size(); i++)
  {
    math::Jacobian bodyJac = getWorldPositionJacobian(
        markers[i].first,
        markers[i].first->getScale().cwiseProduct(markers[i].second));
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
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::MatrixXs result(markers.size() * 3, getNumDofs());
  Eigen::VectorXs originalPos = getPositions();

  s_t eps = 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedPos = originalPos;
        tweakedPos(dof) += eps;
        setPositions(tweakedPos);
        perturbed = getMarkerWorldPositions(markers);
        return true;
      },
      result,
      eps,
      false);

  // Reset everything how we left it
  setPositions(originalPos);

  return result;
}

//==============================================================================
/// This returns the Jacobian relating changes in body scales to changes in
/// marker world positions.
Eigen::MatrixXs Skeleton::getMarkerWorldPositionsJacobianWrtBodyScales(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, getNumBodyNodes() * 3);

  const Eigen::MatrixXi& parentMap = getJointParentMap();
  // Scaling a body will cause the joint offsets to scale, which will move the
  // downstream joint positions by those vectors
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* bodyNode = getBodyNode(i);
    Eigen::Matrix3s R = bodyNode->getWorldTransform().linear();

    // Each body can scale along 3 distinct axis
    for (int axis = 0; axis < 3; axis++)
    {
      Eigen::Vector3s worldOffsetDueToParentJoint
          = bodyNode->getParentJoint()
                ->getWorldTranslationOfChildBodyWrtChildScale(axis);

      // Now begin iterating rows, for each marker
      for (int j = 0; j < markers.size(); j++)
      {
        int sourceJointIndex
            = getJointIndex(markers[j].first->getParentJoint());

        // If this marker is directly attached to the body node we're scaling,
        // we need to account for that
        if (markers[j].first == bodyNode)
        {
          jac.block(j * 3, i * 3 + axis, 3, 1)
              = (R.col(axis) * markers[j].second(axis))
                + worldOffsetDueToParentJoint;
        }
        else
        {
          // Check if this marker is a child of the body we're scaling
          for (int k = 0; k < bodyNode->getNumChildJoints(); k++)
          {
            dynamics::Joint* childJoint = bodyNode->getChildJoint(k);
            if (childJoint == markers[j].first->getParentJoint()
                || parentMap(getJointIndex(childJoint), sourceJointIndex))
            {
              Eigen::Vector3s worldOffsetDueToChildJoint
                  = childJoint->getWorldTranslationOfChildBodyWrtParentScale(
                      axis);

              jac.block(j * 3, i * 3 + axis, 3, 1)
                  = worldOffsetDueToParentJoint + worldOffsetDueToChildJoint;

              break;
            }
          }
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in body scales to changes in
/// marker world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceMarkerWorldPositionsJacobianWrtBodyScales(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, getNumBodyNodes() * 3);

  const double EPS = 1e-7;
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    Eigen::Vector3s originalScale = getBodyNode(i)->getScale();

    for (int axis = 0; axis < 3; axis++)
    {
      Eigen::Vector3s perturbed = originalScale;

      perturbed(axis) += EPS;
      getBodyNode(i)->setScale(perturbed);
      Eigen::VectorXs plus = getMarkerWorldPositions(markers);

      perturbed = originalScale;
      perturbed(axis) -= EPS;
      getBodyNode(i)->setScale(perturbed);
      Eigen::VectorXs minus = getMarkerWorldPositions(markers);

      getBodyNode(i)->setScale(originalScale);

      jac.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in marker offsets to changes in
/// marker world positions.
Eigen::MatrixXs Skeleton::getMarkerWorldPositionsJacobianWrtMarkerOffsets(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
    const
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, markers.size() * 3);

  for (int i = 0; i < markers.size(); i++)
  {
    jac.block<3, 3>(i * 3, i * 3)
        = markers[i].first->getWorldTransform().linear();
    jac.block<3, 1>(i * 3, i * 3) *= markers[i].first->getScale()(0);
    jac.block<3, 1>(i * 3, i * 3 + 1) *= markers[i].first->getScale()(1);
    jac.block<3, 1>(i * 3, i * 3 + 2) *= markers[i].first->getScale()(2);
  }

  return jac;
}

//==============================================================================
/// This returns the Jacobian relating changes in marker offsets to changes in
/// marker world positions.
Eigen::MatrixXs
Skeleton::finiteDifferenceMarkerWorldPositionsJacobianWrtMarkerOffsets(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, markers.size() * 3);

  const double EPS = 1e-7;
  for (int i = 0; i < markers.size(); i++)
  {
    for (int j = 0; j < 3; j++)
    {
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
          perturbedMarkers;
      for (auto pair : markers)
      {
        perturbedMarkers.emplace_back(pair.first, pair.second);
      }
      perturbedMarkers[i].second(j) = markers[i].second(j) + EPS;
      Eigen::VectorXs plus = getMarkerWorldPositions(perturbedMarkers);

      perturbedMarkers[i].second(j) = markers[i].second(j) - EPS;
      Eigen::VectorXs minus = getMarkerWorldPositions(perturbedMarkers);

      jac.col(i * 3 + j) = (plus - minus) / (2 * EPS);
    }
  }

  return jac;
}

//==============================================================================
/// This gets the gradient of the ||f(q) - x|| function with respect to q
Eigen::VectorXs Skeleton::getMarkerWorldPositionDiffToGoalGradientWrtJointPos(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs goal)
{
  Eigen::VectorXs worldPos = getMarkerWorldPositions(markers);
  Eigen::VectorXs diff = worldPos - goal;
  Eigen::MatrixXs J = getMarkerWorldPositionsJacobianWrtJointPositions(markers);
  return 2 * J.transpose() * diff;
}

//==============================================================================
/// This gets the gradient of the ||f(q) - x|| function with respect to q
Eigen::VectorXs
Skeleton::finiteDifferenceMarkerWorldPositionDiffToGoalGradientWrtJointPos(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs goal)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getNumDofs());

  const s_t EPS = 1e-7;
  Eigen::VectorXs originalPos = getPositions();

  for (int i = 0; i < getNumDofs(); i++)
  {
    Eigen::VectorXs perturbed = originalPos;
    perturbed(i) += EPS;
    setPositions(perturbed);

    s_t plus = (getMarkerWorldPositions(markers) - goal).squaredNorm();

    perturbed = originalPos;
    perturbed(i) -= EPS;
    setPositions(perturbed);

    s_t minus = (getMarkerWorldPositions(markers) - goal).squaredNorm();

    result(i) = (plus - minus) / (2 * EPS);
  }

  setPositions(originalPos);

  return result;
}

//==============================================================================
/// This should be equivalent to
/// `getMarkerWorldPositionsJacobianWrtJointPositions`, just slower. This is
/// here so there's a simple non-recursive formula for the Jacobian to take
/// derivatives against.
Eigen::MatrixXs
Skeleton::getScrewsMarkerWorldPositionsJacobianWrtJointPositions(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  // HERE
  //
  //
  //
  //
  //
  //
  //
  //
  //
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
  const Eigen::MatrixXi& parentMap = getJointParentMap();

  for (int j = 0; j < getNumDofs(); j++)
  {
    dynamics::Joint* parentJoint = getDof(j)->getJoint();
    Eigen::Vector6s screw = parentJoint->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint());
    int parentJointIndex = getJointIndex(parentJoint);
    for (int i = 0; i < markers.size(); i++)
    {
      dynamics::Joint* sourceJoint = markers[i].first->getParentJoint();
      int sourceJointIndex = getJointIndex(sourceJoint);

      /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
      /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
      if (parentMap(parentJointIndex, sourceJointIndex) == 1
          || sourceJoint == parentJoint)
      {
        jac.block<3, 1>(i * 3, j) = math::gradientWrtTheta(
            screw, worldMarkers.segment<3>(i * 3), 0.0);
      }
    }
  }

  return jac;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single joint index
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    int index)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
  const Eigen::MatrixXi& parentMap = getJointParentMap();

  // Differentiating the whole mess wrt this joint
  dynamics::Joint* rootJoint = getDof(index)->getJoint();
  Eigen::Vector6s rootScrew = rootJoint->getWorldAxisScrewForPosition(
      getDof(index)->getIndexInJoint());
  int rootJointIndex = getJointIndex(rootJoint);

  for (int j = 0; j < getNumDofs(); j++)
  {
    dynamics::Joint* parentJoint = getDof(j)->getJoint();
    Eigen::Vector6s screw = parentJoint->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint());
    int parentJointIndex = getJointIndex(parentJoint);
    for (int i = 0; i < markers.size(); i++)
    {
      dynamics::Joint* sourceJoint = markers[i].first->getParentJoint();
      int sourceJointIndex = getJointIndex(sourceJoint);

      /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
      /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
      if (parentMap(parentJointIndex, sourceJointIndex) == 1
          || sourceJoint == parentJoint)
      {
        // The original value in this cell is the following

        /*
        jac.block<3, 1>(i * 3, j) = math::gradientWrtTheta(
            screw, worldMarkers.segment<3>(i * 3), 0.0);
        */

        // There's a special case if the root is the parent of both the DOF for
        // this column of the Jac, _and_ of the marker. That means that all
        // we're doing is rotating (and translating, but that's irrelevant) the
        // joint-marker system. So all we need is the gradient of the rotation.
        if (parentMap(rootJointIndex, parentJointIndex) == 1)
        {
          Eigen::Vector3s originalJac = math::gradientWrtTheta(
              screw, worldMarkers.segment<3>(i * 3), 0.0);
          jac.block<3, 1>(i * 3, j) = math::gradientWrtThetaPureRotation(
              rootScrew.head<3>(), originalJac, 0);
        }
        else
        {
          // We'll use the sum-product rule, so we need to individually
          // differentiate both terms (`screw` and `markerPos`) wrt the root
          // joint's theta term.

          // Make `screwGrad` hold the gradient of the screw with respect to
          // root

          Eigen::Vector6s screwGrad = Eigen::Vector6s::Zero();

          if (rootJoint == parentJoint)
          {
            int axisIndex = getDof(j)->getIndexInJoint();
            int rotateIndex = getDof(index)->getIndexInJoint();
            screwGrad = parentJoint->getScrewAxisGradientForPosition(
                axisIndex, rotateIndex);
          }
          else
          {
            // Otherwise rotating the root joint doesn't effect parentJoint's
            // screw
            screwGrad.setZero();
          }

          // `screwGrad` now holds the gradient of the screw with respect to
          // root.

          // Now we need the marker position's gradient wrt the root axis

          Eigen::Vector3s markerGradWrtRoot = Eigen::Vector3s::Zero();

          if (parentMap(rootJointIndex, sourceJointIndex) == 1
              || (rootJoint == sourceJoint))
          {
            markerGradWrtRoot = math::gradientWrtTheta(
                rootScrew, worldMarkers.segment<3>(i * 3), 0.0);
          }

          // Now we just need to apply the product rule to get the final
          // result

          Eigen::Vector3s partA = math::gradientWrtTheta(
              screwGrad, worldMarkers.segment<3>(i * 3), 0.0);
          Eigen::Vector3s partB = math::gradientWrtThetaPureRotation(
              screw.head<3>(), markerGradWrtRoot, 0.0);

          jac.block<3, 1>(i * 3, j) = partA + partB;
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single joint index
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        int index)
{
  s_t originalPos = getPosition(index);
  const s_t EPS = 1e-7;

  setPosition(index, originalPos + EPS);
  Eigen::MatrixXs plus
      = getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  setPosition(index, originalPos - EPS);
  Eigen::MatrixXs minus
      = getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  setPosition(index, originalPos);
  return (plus - minus) / (2 * EPS);
}

//==============================================================================
/// This gets the Jacobian of J.transpose()*leftMultiply with respect to joint
/// positions, holding leftMultiply constant
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsSecondJacobianWrtJointWrtJointPositions(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(getNumDofs(), getNumDofs());

  // We're creating these caches outside of the inner loops, to avoid calling
  // these getter's a bazillion times in the hot inner loops
  std::vector<dynamics::Joint*> dofJoints;
  std::vector<Eigen::Vector6s> dofScrews;
  std::vector<int> dofJointIndices;
  std::vector<int> dofIndexInJoint;
  for (int j = 0; j < getNumDofs(); j++)
  {
    dofJoints.push_back(getDof(j)->getJoint());
    dofIndexInJoint.push_back(getDof(j)->getIndexInJoint());
    dofScrews.push_back(
        dofJoints[j]->getWorldAxisScrewForPosition(dofIndexInJoint[j]));
    dofJointIndices.push_back(getJointIndex(dofJoints[j]));
  }

  std::vector<dynamics::Joint*> markerSourceJoints;
  std::vector<int> markerSourceJointIndices;
  for (int i = 0; i < markers.size(); i++)
  {
    markerSourceJoints.push_back(markers[i].first->getParentJoint());
    markerSourceJointIndices.push_back(getJointIndex(markerSourceJoints[i]));
  }

  const Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
  const Eigen::MatrixXi& parentMap = getJointParentMap();

  // The left multiply means we're taking a weighted combination of rows, to get
  // a single row. That's our new vector. Then we're treating that vector as a
  // column vector, and building a Jacobian of how that vector changes as we
  // change joint positions.

  for (int col = 0; col < getNumDofs(); col++)
  {
    const int index = col;

    // Here's the original version, not optimized

    // result.col(col)
    //     = getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints(
    //           markers, col)
    //           .transpose()
    //       * leftMultiply;

    // It turns out that calling
    // getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints()
    // in a tight inner loop actually wastes a lot of inefficient index
    // lookups in Skeleton, so instead we can cache so data outside of the
    // loop

    Eigen::MatrixXs jac
        = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

    // Differentiating the whole mess wrt this joint
    const dynamics::Joint* rootJoint = dofJoints[index];
    const Eigen::Vector6s rootScrew = dofScrews[index];
    const int rootJointIndex = dofJointIndices[index];

    for (int j = 0; j < getNumDofs(); j++)
    {
      dynamics::Joint* parentJoint = dofJoints[j];
      const Eigen::Vector6s screw = dofScrews[j];
      const int parentJointIndex = dofJointIndices[j];
      for (int i = 0; i < markers.size(); i++)
      {
        dynamics::Joint* sourceJoint = markerSourceJoints[i];
        int sourceJointIndex = markerSourceJointIndices[i];

        /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
        /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
        if (parentMap(parentJointIndex, sourceJointIndex) == 1
            || sourceJoint == parentJoint)
        {
          // The original value in this cell is the following

          /*
          jac.block<3, 1>(i * 3, j) = math::gradientWrtTheta(
              screw, worldMarkers.segment<3>(i * 3), 0.0);
          */

          // There's a special case if the root is the parent of both the DOF
          // for this column of the Jac, _and_ of the marker. That means that
          // all we're doing is rotating (and translating, but that's
          // irrelevant) the joint-marker system. So all we need is the gradient
          // of the rotation.
          if (parentMap(rootJointIndex, parentJointIndex) == 1)
          {
            Eigen::Vector3s originalJac = math::gradientWrtTheta(
                screw, worldMarkers.segment<3>(i * 3), 0.0);
            jac.block<3, 1>(i * 3, j) = math::gradientWrtThetaPureRotation(
                rootScrew.head<3>(), originalJac, 0);
          }
          else
          {
            // We'll use the sum-product rule, so we need to individually
            // differentiate both terms (`screw` and `markerPos`) wrt the root
            // joint's theta term.

            // Make `screwGrad` hold the gradient of the screw with respect to
            // root

            Eigen::Vector6s screwGrad = Eigen::Vector6s::Zero();

            if (rootJoint == parentJoint)
            {
              int axisIndex
                  = dofIndexInJoint[j]; // getDof(j)->getIndexInJoint();
              int rotateIndex
                  = dofIndexInJoint[index]; // getDof(index)->getIndexInJoint();
              screwGrad = parentJoint->getScrewAxisGradientForPosition(
                  axisIndex, rotateIndex);
            }
            else
            {
              // Otherwise rotating the root joint doesn't effect parentJoint's
              // screw
              screwGrad.setZero();
            }

            // `screwGrad` now holds the gradient of the screw with respect to
            // root.

            // Now we need the marker position's gradient wrt the root axis

            Eigen::Vector3s markerGradWrtRoot = Eigen::Vector3s::Zero();

            if (parentMap(rootJointIndex, sourceJointIndex) == 1
                || (rootJoint == sourceJoint))
            {
              markerGradWrtRoot = math::gradientWrtTheta(
                  rootScrew, worldMarkers.segment<3>(i * 3), 0.0);
            }

            // Now we just need to apply the product rule to get the final
            // result

            Eigen::Vector3s partA = math::gradientWrtTheta(
                screwGrad, worldMarkers.segment<3>(i * 3), 0.0);
            Eigen::Vector3s partB = math::gradientWrtThetaPureRotation(
                screw.head<3>(), markerGradWrtRoot, 0.0);

            jac.block<3, 1>(i * 3, j) = partA + partB;
          }
        }
      }
    }

    result.col(col) = jac.transpose() * leftMultiply;
  }

  return result;
}

//==============================================================================
/// This gets the Jacobian of J.transpose()*leftMultiply with respect to joint
/// positions, holding leftMultiply constant
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtJointPositions(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(getNumDofs(), getNumDofs());

  const s_t EPS = 1e-7;
  Eigen::VectorXs originalPos = getPositions();

  for (int i = 0; i < getNumDofs(); i++)
  {
    Eigen::VectorXs perturbed = originalPos;
    perturbed(i) += EPS;
    setPositions(perturbed);

    Eigen::VectorXs plus
        = leftMultiply.transpose()
          * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

    perturbed = originalPos;
    perturbed(i) -= EPS;
    setPositions(perturbed);

    Eigen::VectorXs minus
        = leftMultiply.transpose()
          * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

    result.col(i) = (plus - minus) / (2 * EPS);
  }

  setPositions(originalPos);

  return result;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single body scaling
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtBodyScale(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    int index,
    int axis,
    const Eigen::MatrixXs& markerWrtScaleJac)
{
  dynamics::BodyNode* scaleBody = getBodyNode(index);
  dynamics::Joint* scaleBodyParentJoint = scaleBody->getParentJoint();
  int scaleBodyParentJointIndex = getJointIndex(scaleBodyParentJoint);
  (void)scaleBodyParentJointIndex;
  int gradCol = index * 3 + axis;

  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
  const Eigen::MatrixXi& parentMap = getJointParentMap();

  Eigen::MatrixXs jointChangeInPos
      = getJointWorldPositionsJacobianWrtBodyScales(getJoints());

  for (int j = 0; j < getNumDofs(); j++)
  {
    dynamics::Joint* parentJoint = getDof(j)->getJoint();
    Eigen::Vector6s screw = parentJoint->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint());

    // Find the change in the velocity due to the joint
    Eigen::Vector6s dScrew = Eigen::Vector6s::Zero();
    if (parentJoint->getParentBodyNode() != nullptr
        && parentJoint->getParentBodyNode()->getName() == scaleBody->getName())
    {
      dScrew = parentJoint->getRelativeJacobianDerivWrtParentScale(axis).col(
          getDof(j)->getIndexInJoint());
    }
    if (parentJoint->getChildBodyNode()->getName() == scaleBody->getName())
    {
      // Get only the portion of the derivative that is not due to scaling the
      // child transform
      dScrew = parentJoint->getRelativeJacobianDerivWrtChildScale(axis).col(
                   getDof(j)->getIndexInJoint())
               - parentJoint->Joint::getRelativeJacobianDerivWrtChildScale(axis)
                     .col(getDof(j)->getIndexInJoint());
    }
    assert(dScrew.head<3>().norm() == 0 && "This optimized formula does not support joints where scaling an attached body can change rotational speed! Either update this function, or consider re-parameterizing your custom joint definition.");
    Eigen::Isometry3s childT
        = parentJoint->getChildBodyNode()->getWorldTransform();
    dScrew = math::AdT(childT, dScrew);

    int parentJointIndex = getJointIndex(parentJoint);
    for (int i = 0; i < markers.size(); i++)
    {
      dynamics::Joint* sourceJoint = markers[i].first->getParentJoint();
      int sourceJointIndex = getJointIndex(sourceJoint);

      /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
      /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
      if (parentMap(parentJointIndex, sourceJointIndex) == 1
          || sourceJoint == parentJoint)
      {
        Eigen::Vector3s markerGrad
            = markerWrtScaleJac.block<3, 1>(i * 3, gradCol);
        Eigen::Vector3s jointGrad
            = jointChangeInPos.block<3, 1>(parentJointIndex * 3, gradCol);
        // We're interested in relative motion of the marker wrt the joint
        jac.block<3, 1>(i * 3, j)
            = screw.head<3>().cross(markerGrad - jointGrad) + dScrew.tail<3>();
      }
    }
  }

  return jac;
}

//==============================================================================
Eigen::MatrixXs Skeleton::scratch(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers)
{
  (void)markers;
  Eigen::VectorXs worldJoints = getJointWorldPositions(getJoints());

  return worldJoints;
}

//==============================================================================
Eigen::MatrixXs Skeleton::scratchAnalytical(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    int index,
    int axis)
{
  dynamics::BodyNode* scaleBody = getBodyNode(index);
  (void)scaleBody;

  const Eigen::MatrixXs markerWrtScaleJac
      = getMarkerWorldPositionsJacobianWrtBodyScales(markers);
  const Eigen::MatrixXs jointsWrtScaleJac
      = getJointWorldPositionsJacobianWrtBodyScales(getJoints());
  int gradCol = index * 3 + axis;
  (void)gradCol;

  return jointsWrtScaleJac.col(gradCol);
}

//==============================================================================
Eigen::MatrixXs Skeleton::scratchFd(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    int index,
    int axis)
{
  Eigen::Vector3s originalScale = getBodyNode(index)->getScale();
  const s_t EPS = 1e-3;

  Eigen::MatrixXs result(getJoints().size() * 3, getNumDofs());

  math::finiteDifference<Eigen::MatrixXs>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::MatrixXs& perturbed) {
        Eigen::VectorXs tweakedScale = originalScale;
        if (axis == -1)
        {
          tweakedScale += Eigen::Vector3s::Ones() * eps;
        }
        else
        {
          tweakedScale(axis) += eps;
        }
        getBodyNode(index)->setScale(tweakedScale);
        perturbed = scratch(markers);
        return true;
      },
      result,
      EPS,
      false);

  // Reset everything how we left it
  getBodyNode(index)->setScale(originalScale);

  return result;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single body scaling
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtBodyScale(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        int index,
        int axis)
{
  Eigen::Vector3s originalScale = getBodyNode(index)->getScale();
  const s_t EPS = 1e-3;

  Eigen::MatrixXs result(markers.size() * 3, getNumDofs());

  math::finiteDifference<Eigen::MatrixXs>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::MatrixXs& perturbed) {
        Eigen::VectorXs tweakedScale = originalScale;
        if (axis == -1)
        {
          tweakedScale += Eigen::Vector3s::Ones() * eps;
        }
        else
        {
          tweakedScale(axis) += eps;
        }
        getBodyNode(index)->setScale(tweakedScale);
        perturbed = getMarkerWorldPositionsJacobianWrtJointPositions(markers);
        return true;
      },
      result,
      EPS,
      false);

  // Reset everything how we left it
  getBodyNode(index)->setScale(originalScale);

  return result;
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to body
/// scales
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsSecondJacobianWrtJointWrtBodyScale(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(getNumDofs(), getNumBodyNodes() * 3);

  // The left multiply means we're taking a weighted combination of rows, to get
  // a single row. That's our new vector. Then we're treating that vector as a
  // column vector, and building a Jacobian of how that vector changes as we
  // change body scales.

  Eigen::MatrixXs markerJacWrtScales
      = getMarkerWorldPositionsJacobianWrtBodyScales(markers);
  Eigen::MatrixXs jointJacWrtScales
      = getJointWorldPositionsJacobianWrtBodyScales(getJoints());

  // We're creating these caches outside of the inner loops, to avoid calling
  // these getter's a bazillion times in the hot inner loops
  std::vector<dynamics::Joint*> parentJoints;
  std::vector<Eigen::Vector6s> screws;
  std::vector<int> parentJointIndices;
  for (int j = 0; j < getNumDofs(); j++)
  {
    parentJoints.push_back(getDof(j)->getJoint());
    screws.push_back(parentJoints[j]->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint()));
    parentJointIndices.push_back(getJointIndex(parentJoints[j]));
  }

  std::vector<dynamics::Joint*> markerSourceJoints;
  std::vector<int> markerSourceJointIndices;
  for (int i = 0; i < markers.size(); i++)
  {
    markerSourceJoints.push_back(markers[i].first->getParentJoint());
    markerSourceJointIndices.push_back(getJointIndex(markerSourceJoints[i]));
  }

  for (int body = 0; body < getNumBodyNodes(); body++)
  {
    // We also cache these results in this outer loop to avoid re-creating them
    // in the inner axis loop
    const int index = body;
    dynamics::BodyNode* scaleBody = getBodyNode(index);

    for (int axis = 0; axis < 3; axis++)
    {
      // Here's the original version, not optimized

      // result.col(body * 3 + axis)
      //     = getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtBodyScale(
      //           markers, body, axis, scaleJac)
      //           .transpose()
      //       * leftMultiply;

      // It turns out that calling
      // getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets()
      // in a tight inner loop actually wastes a lot of inefficient index
      // lookups in Skeleton, so instead we can cache so data outside of the
      // loop

      int gradCol = index * 3 + axis;

      Eigen::MatrixXs jac
          = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

      Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
      const Eigen::MatrixXi& parentMap = getJointParentMap();

      for (int j = 0; j < getNumDofs(); j++)
      {
        dynamics::Joint* parentJoint = parentJoints[j];
        Eigen::Vector6s screw = screws[j];
        int parentJointIndex = parentJointIndices[j];

        // Find the change in the velocity due to the joint
        Eigen::Vector6s dScrew = Eigen::Vector6s::Zero();
        if (parentJoint->getParentBodyNode() != nullptr
            && parentJoint->getParentBodyNode()->getName()
                   == scaleBody->getName())
        {
          dScrew
              = parentJoint->getRelativeJacobianDerivWrtParentScale(axis).col(
                  getDof(j)->getIndexInJoint());
        }
        if (parentJoint->getChildBodyNode()->getName() == scaleBody->getName())
        {
          // Get only the portion of the derivative that is not due to scaling
          // the child transform
          dScrew = parentJoint->getRelativeJacobianDerivWrtChildScale(axis).col(
                       getDof(j)->getIndexInJoint())
                   - parentJoint
                         ->Joint::getRelativeJacobianDerivWrtChildScale(axis)
                         .col(getDof(j)->getIndexInJoint());
        }
        assert(dScrew.head<3>().norm() == 0 && "This optimized formula does not support joints where scaling an attached body can change rotational speed! Either update this function, or consider re-parameterizing your custom joint definition.");
        Eigen::Isometry3s childT
            = parentJoint->getChildBodyNode()->getWorldTransform();
        dScrew = math::AdT(childT, dScrew);

        for (int i = 0; i < markers.size(); i++)
        {
          dynamics::Joint* sourceJoint = markerSourceJoints[i];
          int sourceJointIndex = markerSourceJointIndices[i];

          /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
          /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
          if (parentMap(parentJointIndex, sourceJointIndex) == 1
              || sourceJoint == parentJoint)
          {
            Eigen::Vector3s markerGrad
                = markerJacWrtScales.block<3, 1>(i * 3, gradCol);
            Eigen::Vector3s jointGrad
                = jointJacWrtScales.block<3, 1>(parentJointIndex * 3, gradCol);
            // We're interested in relative motion of the marker wrt the joint
            jac.block<3, 1>(i * 3, j)
                = screw.head<3>().cross(markerGrad - jointGrad)
                  + dScrew.tail<3>();
          }
        }
      }

      result.col(body * 3 + axis) = jac.transpose() * leftMultiply;
    }
  }

  return result;
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to body
/// scales
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtBodyScale(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(getNumDofs(), getNumBodyNodes() * 3);

  const s_t EPS = 1e-7;

  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    Eigen::VectorXs originalScale = getBodyNode(i)->getScale();

    for (int axis = 0; axis < 3; axis++)
    {
      Eigen::VectorXs perturbed = originalScale;
      perturbed(axis) += EPS;
      getBodyNode(i)->setScale(perturbed);

      Eigen::VectorXs plus
          = leftMultiply.transpose()
            * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

      perturbed = originalScale;
      perturbed(axis) -= EPS;
      getBodyNode(i)->setScale(perturbed);

      Eigen::VectorXs minus
          = leftMultiply.transpose()
            * getMarkerWorldPositionsJacobianWrtJointPositions(markers);

      getBodyNode(i)->setScale(originalScale);

      result.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return result;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single marker offset
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    int marker,
    int axis,
    const Eigen::MatrixXs& markerWrtMarkerJac)
{
  int markerGradCol = marker * 3 + axis;
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

  // Eigen::VectorXs worldMarkers = getMarkerWorldPositions(markers);
  const Eigen::MatrixXi& parentMap = getJointParentMap();

  for (int j = 0; j < getNumDofs(); j++)
  {
    dynamics::Joint* parentJoint = getDof(j)->getJoint();
    Eigen::Vector6s screw = parentJoint->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint());
    int parentJointIndex = getJointIndex(parentJoint);
    for (int i = 0; i < markers.size(); i++)
    {
      dynamics::Joint* sourceJoint = markers[i].first->getParentJoint();
      int sourceJointIndex = getJointIndex(sourceJoint);

      /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
      /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
      if (parentMap(parentJointIndex, sourceJointIndex) == 1
          || sourceJoint == parentJoint)
      {
        Eigen::Vector3s markerGrad
            = markerWrtMarkerJac.block<3, 1>(i * 3, markerGradCol);
        jac.block<3, 1>(i * 3, j) = math::gradientWrtThetaPureRotation(
            screw.head<3>(), markerGrad, 0.0);
      }
    }
  }

  return jac;
}

//==============================================================================
/// This gets the derivative of the Jacobian of the markers wrt joint
/// positions, with respect to a single marker offset
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        int marker,
        int axis)
{
  const s_t EPS = 1e-7;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersCopy;
  for (auto marker : markers)
    markersCopy.push_back(std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
        &(*marker.first), Eigen::Vector3s(marker.second)));

  s_t originalOffset = markersCopy[marker].second(axis);

  markersCopy[marker].second(axis) = originalOffset + EPS;
  Eigen::MatrixXs plus
      = getMarkerWorldPositionsJacobianWrtJointPositions(markersCopy);

  markersCopy[marker].second(axis) = originalOffset - EPS;
  Eigen::MatrixXs minus
      = getMarkerWorldPositionsJacobianWrtJointPositions(markersCopy);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to marker
/// offsets
Eigen::MatrixXs
Skeleton::getMarkerWorldPositionsSecondJacobianWrtJointWrtMarkerOffsets(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs leftMultiply)
{
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(getNumDofs(), markers.size() * 3);

  // The left multiply means we're taking a weighted combination of rows, to get
  // a single row. That's our new vector. Then we're treating that vector as a
  // column vector, and building a Jacobian of how that vector changes as we
  // change body scales.

  Eigen::MatrixXs markerJac
      = getMarkerWorldPositionsJacobianWrtMarkerOffsets(markers);

  // We're creating these caches outside of the inner loops, to avoid calling
  // these getter's a bazillion times in the hot inner loops
  std::vector<dynamics::Joint*> parentJoints;
  std::vector<Eigen::Vector6s> screws;
  std::vector<int> parentJointIndices;
  for (int j = 0; j < getNumDofs(); j++)
  {
    parentJoints.push_back(getDof(j)->getJoint());
    screws.push_back(parentJoints[j]->getWorldAxisScrewForPosition(
        getDof(j)->getIndexInJoint()));
    parentJointIndices.push_back(getJointIndex(parentJoints[j]));
  }

  std::vector<dynamics::Joint*> markerSourceJoints;
  std::vector<int> markerSourceJointIndices;
  for (int i = 0; i < markers.size(); i++)
  {
    markerSourceJoints.push_back(markers[i].first->getParentJoint());
    markerSourceJointIndices.push_back(getJointIndex(markerSourceJoints[i]));
  }

  const Eigen::MatrixXi& parentMap = getJointParentMap();

  for (int marker = 0; marker < markers.size(); marker++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      // Here's the original version, not optimized

      // result.col(marker * 3 + axis)
      //     =
      //     getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets(
      //           markers, marker, axis, markerJac)
      //           .transpose()
      //       * leftMultiply;

      // It turns out that calling
      // getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets()
      // in a tight inner loop actually wastes a lot of inefficient index
      // lookups in Skeleton, so instead we can cache so data outside of the
      // loop

      int markerGradCol = marker * 3 + axis;
      Eigen::MatrixXs jac
          = Eigen::MatrixXs::Zero(markers.size() * 3, getNumDofs());

      for (int j = 0; j < getNumDofs(); j++)
      {
        dynamics::Joint* parentJoint = parentJoints[j];
        Eigen::Vector6s screw = screws[j];
        int parentJointIndex = parentJointIndices[j];

        for (int i = 0; i < markers.size(); i++)
        {
          dynamics::Joint* sourceJoint = markerSourceJoints[i];
          int sourceJointIndex = markerSourceJointIndices[i];

          /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
          /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
          if (parentMap(parentJointIndex, sourceJointIndex) == 1
              || sourceJoint == parentJoint)
          {
            Eigen::Vector3s markerGrad
                = markerJac.block<3, 1>(i * 3, markerGradCol);
            jac.block<3, 1>(i * 3, j) = math::gradientWrtThetaPureRotation(
                screw.head<3>(), markerGrad, 0.0);
          }
        }
      }

      result.col(marker * 3 + axis) = jac.transpose() * leftMultiply;
    }
  }

  return result;
}

//==============================================================================
/// This gets the Jacobian of leftMultiply.transpose()*J with respect to marker
/// offsets
Eigen::MatrixXs Skeleton::
    finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtMarkerOffsets(
        const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
            markers,
        Eigen::VectorXs leftMultiply)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersCopy;
  for (auto marker : markers)
    markersCopy.push_back(std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
        &(*marker.first), Eigen::Vector3s(marker.second)));

  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(getNumDofs(), markers.size() * 3);

  const s_t EPS = 1e-7;

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      s_t originalOffset = markersCopy[i].second(axis);

      markersCopy[i].second(axis) = originalOffset + EPS;
      Eigen::VectorXs plus
          = leftMultiply.transpose()
            * getMarkerWorldPositionsJacobianWrtJointPositions(markersCopy);

      markersCopy[i].second(axis) = originalOffset - EPS;
      Eigen::VectorXs minus
          = leftMultiply.transpose()
            * getMarkerWorldPositionsJacobianWrtJointPositions(markersCopy);

      markersCopy[i].second(axis) = originalOffset;

      result.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return result;
}

//==============================================================================
/// This runs IK, attempting to fit the world positions of the passed in
/// joints to the vector of (concatenated) target positions. This can
/// optionally also rescale the skeleton.
#define DART_SKEL_LOG_IK_OUTPUT
s_t Skeleton::fitJointsToWorldPositions(
    const std::vector<dynamics::Joint*>& positionJoints,
    Eigen::VectorXs targetPositions,
    bool scaleBodies,
    math::IKConfig config)
{
  if (scaleBodies)
  {
    Eigen::VectorXs initialPos
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    Eigen::VectorXs lowerBound
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    Eigen::VectorXs upperBound
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    initialPos.segment(0, getNumDofs()) = getPositions();
    lowerBound.segment(0, getNumDofs()) = getPositionLowerLimits();
    upperBound.segment(0, getNumDofs()) = getPositionUpperLimits();
    initialPos.segment(getNumDofs(), getGroupScaleDim()) = getGroupScales();
    lowerBound.segment(getNumDofs(), getGroupScaleDim())
        = getGroupScalesLowerBound();
    upperBound.segment(getNumDofs(), getGroupScaleDim())
        = getGroupScalesUpperBound();

    return math::solveIK(
        initialPos,
        upperBound,
        lowerBound,
        positionJoints.size() * 3,
        [this](/* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          setPositions(pos.segment(0, getNumDofs()));
          if (clamp)
          {
            clampPositionsToLimits();
          }

          // Set scales
          Eigen::VectorXs newScales
              = pos.segment(getNumDofs(), getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound = getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound = getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, getNumDofs()) = getPositions();
          clampedPos.segment(getNumDofs(), getGroupScaleDim()) = newScales;
          return clampedPos;
        },
        [this, targetPositions, positionJoints](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff = getJointWorldPositions(positionJoints) - targetPositions;
          assert(jac.cols() == getNumDofs() + getGroupScaleDim());
          assert(jac.rows() == positionJoints.size() * 3);
          jac.setZero();
          jac.block(0, 0, positionJoints.size() * 3, getNumDofs())
              = getJointWorldPositionsJacobianWrtJointPositions(positionJoints);
          jac.block(
              0, getNumDofs(), positionJoints.size() * 3, getGroupScaleDim())
              = getJointWorldPositionsJacobianWrtGroupScales(positionJoints);
        },
        [this](Eigen::Ref<Eigen::VectorXs> val) {
          val.segment(0, getNumDofs()) = getRandomPose();
          val.segment(getNumDofs(), getGroupScaleDim()).setConstant(1.0);
        },
        config);
  }
  else
  {
    return math::solveIK(
        getPositions(),
        getPositionUpperLimits(),
        getPositionLowerLimits(),
        positionJoints.size() * 3,
        [this](/* in*/ Eigen::VectorXs pos, bool clamp) {
          setPositions(pos);
          if (clamp)
          {
            clampPositionsToLimits();
            return getPositions();
          }
          return pos;
        },
        [this, targetPositions, positionJoints](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff = getJointWorldPositions(positionJoints) - targetPositions;
          jac = getJointWorldPositionsJacobianWrtJointPositions(positionJoints);
        },
        [this](Eigen::Ref<Eigen::VectorXs> val) { val = getRandomPose(); },
        config);
  }
}

//==============================================================================
/// This runs IK, attempting to fit the world positions of the passed in
/// markers to the vector of (concatenated) target positions.
s_t Skeleton::fitMarkersToWorldPositions(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs targetPositions,
    Eigen::VectorXs markerWeights,
    bool scaleBodies,
    math::IKConfig config)
{
  if (scaleBodies)
  {
    Eigen::VectorXs initialPos
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    Eigen::VectorXs lowerBound
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    Eigen::VectorXs upperBound
        = Eigen::VectorXs::Zero(getNumDofs() + getGroupScaleDim());
    initialPos.segment(0, getNumDofs()) = getPositions();
    lowerBound.segment(0, getNumDofs()) = getPositionLowerLimits();
    upperBound.segment(0, getNumDofs()) = getPositionUpperLimits();
    initialPos.segment(getNumDofs(), getGroupScaleDim()) = getGroupScales();
    lowerBound.segment(getNumDofs(), getGroupScaleDim())
        = getGroupScalesLowerBound();
    upperBound.segment(getNumDofs(), getGroupScaleDim())
        = getGroupScalesUpperBound();

    return math::solveIK(
        initialPos,
        upperBound,
        lowerBound,
        markers.size() * 3,
        [this](/* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          setPositions(pos.segment(0, getNumDofs()));
          if (clamp)
          {
            clampPositionsToLimits();
          }

          // Set scales
          Eigen::VectorXs newScales
              = pos.segment(getNumDofs(), getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound = getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound = getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, getNumDofs()) = getPositions();
          clampedPos.segment(getNumDofs(), getGroupScaleDim()) = newScales;
          return clampedPos;
        },
        [this, targetPositions, markers](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff = getMarkerWorldPositions(markers) - targetPositions;
          assert(jac.cols() == getNumDofs() + getGroupScaleDim());
          assert(jac.rows() == markers.size() * 3);
          jac.setZero();
          jac.block(0, 0, markers.size() * 3, getNumDofs())
              = getMarkerWorldPositionsJacobianWrtJointPositions(markers);
          jac.block(0, getNumDofs(), markers.size() * 3, getGroupScaleDim())
              = getMarkerWorldPositionsJacobianWrtGroupScales(markers);
        },
        [this](Eigen::Ref<Eigen::VectorXs> val) {
          val.segment(0, getNumDofs()) = getRandomPose();
          val.segment(getNumDofs(), getGroupScaleDim()).setConstant(1.0);
        },
        config);
  }
  else
  {
    return math::solveIK(
        getPositions(),
        getPositionUpperLimits(),
        getPositionLowerLimits(),
        markers.size() * 3,
        [this](/* in*/ Eigen::VectorXs pos, bool clamp) {
          setPositions(pos);
          if (clamp)
          {
            clampPositionsToLimits();
            return getPositions();
          }
          return pos;
        },
        [this, targetPositions, markers, markerWeights](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff = getMarkerWorldPositions(markers) - targetPositions;
          for (int j = 0; j < markerWeights.size(); j++)
          {
            diff.segment<3>(j * 3) *= markerWeights(j);
          }
          jac = getMarkerWorldPositionsJacobianWrtJointPositions(markers);
        },
        [this](Eigen::Ref<Eigen::VectorXs> val) { val = getRandomPose(); },
        config);
  }
}

//==============================================================================
/// This measures the distance between two markers in world space, at the
/// current configuration and scales.
s_t Skeleton::getDistanceInWorldSpace(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(markerA);
  markers.push_back(markerB);
  Eigen::VectorXs poses = getMarkerWorldPositions(markers);
  Eigen::Vector3s poseA = poses.head<3>();
  Eigen::Vector3s poseB = poses.tail<3>();

  return (poseA - poseB).norm();
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// body scales
Eigen::VectorXs Skeleton::getGradientOfDistanceWrtBodyScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(markerA);
  markers.push_back(markerB);
  Eigen::VectorXs poses = getMarkerWorldPositions(markers);
  Eigen::Vector3s poseA = poses.head<3>();
  Eigen::Vector3s poseB = poses.tail<3>();

  s_t norm = (poseA - poseB).norm();

  Eigen::MatrixXs poseJac
      = getMarkerWorldPositionsJacobianWrtBodyScales(markers);

  Eigen::VectorXs result = Eigen::VectorXs::Zero(poseJac.cols());
  for (int i = 0; i < poseJac.cols(); i++)
  {
    Eigen::Vector3s d_poseA = poseJac.col(i).head<3>();
    Eigen::Vector3s d_poseB = poseJac.col(i).tail<3>();

    result(i) = (1.0 / norm) * (poseA - poseB).dot(d_poseA - d_poseB);
  }

  return result;
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// body scales
Eigen::VectorXs Skeleton::finiteDifferenceGradientOfDistanceWrtBodyScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB)
{
  Eigen::VectorXs originalScales = getBodyScales();

  Eigen::VectorXs result = Eigen::VectorXs::Zero(originalScales.size());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs perturbed = originalScales;
        perturbed(i) += eps;
        setBodyScales(perturbed);

        out = getDistanceInWorldSpace(markerA, markerB);

        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
/// This measures the distance between two markers in world space **along a
/// specific axis**, at the current configuration and scales. For example, if
/// the axis is the Y axis, we're just measuring the Y distance between
/// markers.
s_t Skeleton::getDistanceAlongAxis(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
    Eigen::Vector3s axis)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(markerA);
  markers.push_back(markerB);
  Eigen::VectorXs poses = getMarkerWorldPositions(markers);
  Eigen::Vector3s poseA = poses.head<3>();
  Eigen::Vector3s poseB = poses.tail<3>();

  return (poseA - poseB).dot(axis);
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// body scales
Eigen::VectorXs Skeleton::getGradientOfDistanceAlongAxisWrtBodyScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
    Eigen::Vector3s axis)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(markerA);
  markers.push_back(markerB);

  Eigen::MatrixXs poseJac
      = getMarkerWorldPositionsJacobianWrtBodyScales(markers);

  Eigen::VectorXs result = Eigen::VectorXs::Zero(poseJac.cols());
  for (int i = 0; i < poseJac.cols(); i++)
  {
    Eigen::Vector3s d_poseA = poseJac.col(i).head<3>();
    Eigen::Vector3s d_poseB = poseJac.col(i).tail<3>();

    result(i) = axis.dot(d_poseA - d_poseB);
  }

  return result;
}

//==============================================================================
/// This returns the gradient of the distance measurement, with respect to
/// body scales
Eigen::VectorXs
Skeleton::finiteDifferenceGradientOfDistanceAlongAxisWrtBodyScales(
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
    std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
    Eigen::Vector3s axis)
{
  Eigen::VectorXs originalScales = getBodyScales();

  Eigen::VectorXs result = Eigen::VectorXs::Zero(originalScales.size());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs perturbed = originalScales;
        perturbed(i) += eps;
        setBodyScales(perturbed);

        out = getDistanceAlongAxis(markerA, markerB, axis);

        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
/// This returns the spatial velocities (6 vecs) of the bodies in world space,
/// concatenated
Eigen::VectorXs Skeleton::getBodyWorldVelocities()
{
  Eigen::VectorXs vels = Eigen::VectorXs(getNumBodyNodes() * 6);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    vels.segment<6>(i * 6)
        = getBodyNode(i)->getSpatialVelocity(Frame::World(), Frame::World());
  }
  return vels;
}

//==============================================================================
/// This returns the spatial accelerations (6 vecs) of the bodies in world
/// space, concatenated
Eigen::VectorXs Skeleton::getBodyWorldAccelerations()
{
  Eigen::VectorXs accs = Eigen::VectorXs(getNumBodyNodes() * 6);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    accs.segment<6>(i * 6) = getBodyNode(i)->getSpatialAcceleration(
        Frame::World(), Frame::World());
  }
  return accs;
}

//==============================================================================
/// This computes the jacobian of the world velocities for each body with
/// respect to `wrt`
Eigen::MatrixXs Skeleton::getBodyWorldVelocitiesJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(getNumBodyNodes() * 6, dim);

  if (wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::GROUP_SCALES)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      bodyNodes[i]->computeJacobianOfCForward(wrt, true);
      jac.block(i * 6, 0, 6, dim) = math::AdRJac(
          bodyNodes[i]->getWorldTransform(), bodyNodes[i]->mCg_V_p);
    }
    // We need to account for the relative rotation, wrt the world frame,
    // because that's an extra step that isn't accounted for in the featherstone
    // algo. We can use the product rule, and add the cross product of the
    // current vel and the rotation axis for each joint.
    if (wrt == neural::WithRespectTo::POSITION)
    {
      const Eigen::MatrixXi& jointParent = getJointParentMap();
      for (int i = 0; i < getNumDofs(); i++)
      {
        int jointIndex = getDof(i)->getJoint()->getJointIndexInSkeleton();
        Eigen::Vector6s worldScrew
            = getDof(i)->getJoint()->getWorldAxisScrewForPosition(
                getDof(i)->getIndexInJoint());
        Eigen::Vector3s worldRot = worldScrew.head<3>();
        for (int b = 0; b < getNumBodyNodes(); b++)
        {
          int bodyJointIndex
              = getBodyNode(b)->getParentJoint()->getJointIndexInSkeleton();
          if (bodyJointIndex == jointIndex
              || jointParent(jointIndex, bodyJointIndex) == 1)
          {
            Eigen::Vector6s spatialVel = getBodyNode(b)->getSpatialVelocity(
                Frame::World(), Frame::World());
            jac.block<3, 1>(b * 6, i) -= spatialVel.head<3>().cross(worldRot);
            jac.block<3, 1>(b * 6 + 3, i)
                -= spatialVel.tail<3>().cross(worldRot);
          }
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This brute forces our world velocities jacobian
Eigen::MatrixXs Skeleton::finiteDifferenceBodyWorldVelocitiesJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs result(getNumBodyNodes() * 6, dim);
  Eigen::VectorXs original = wrt->get(this);

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        wrt->set(this, tweaked);
        perturbed = getBodyWorldVelocities();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, original);

  return result;
}

//==============================================================================
/// This computes the jacobian of the world accelerations for each body with
/// respect to `wrt`
Eigen::MatrixXs Skeleton::getBodyWorldAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(getNumBodyNodes() * 6, dim);

  if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      jac.block(i * 6, 0, 6, dim) = getJacobian(bodyNodes[i], Frame::World());
    }
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::GROUP_SCALES)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      bodyNodes[i]->computeJacobianOfCForward(wrt, true);
      jac.block(i * 6, 0, 6, dim) = math::AdRJac(
          bodyNodes[i]->getWorldTransform(), bodyNodes[i]->mCg_dV_p);
    }
  }
  // We need to account for the relative rotation, wrt the world frame,
  // because that's an extra step that isn't accounted for in the featherstone
  // algo. We can use the product rule, and add the cross product of the
  // current acc and the rotation axis for each joint.
  if (wrt == neural::WithRespectTo::POSITION)
  {
    const Eigen::MatrixXi& jointParent = getJointParentMap();
    for (int i = 0; i < getNumDofs(); i++)
    {
      int jointIndex = getDof(i)->getJoint()->getJointIndexInSkeleton();
      Eigen::Vector6s worldScrew
          = getDof(i)->getJoint()->getWorldAxisScrewForPosition(
              getDof(i)->getIndexInJoint());
      Eigen::Vector3s worldRot = worldScrew.head<3>();
      for (int b = 0; b < getNumBodyNodes(); b++)
      {
        int bodyJointIndex
            = getBodyNode(b)->getParentJoint()->getJointIndexInSkeleton();
        if (bodyJointIndex == jointIndex
            || jointParent(jointIndex, bodyJointIndex) == 1)
        {
          Eigen::Vector6s spatialAcc = getBodyNode(b)->getSpatialAcceleration(
              Frame::World(), Frame::World());
          jac.block<3, 1>(b * 6, i) -= spatialAcc.head<3>().cross(worldRot);
          jac.block<3, 1>(b * 6 + 3, i) -= spatialAcc.tail<3>().cross(worldRot);
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This brute forces our world accelerations jacobian
Eigen::MatrixXs Skeleton::finiteDifferenceBodyWorldAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs result(getNumBodyNodes() * 6, dim);
  Eigen::VectorXs original = wrt->get(this);

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        wrt->set(this, tweaked);
        perturbed = getBodyWorldAccelerations();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, original);

  return result;
}

//==============================================================================
/// This returns the spatial velocities (6 vecs) of the COMs of each body in
/// world space, concatenated
Eigen::VectorXs Skeleton::getCOMWorldVelocities()
{
  Eigen::VectorXs vels = Eigen::VectorXs(getNumBodyNodes() * 6);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    vels.segment<6>(i * 6)
        = getBodyNode(i)->getCOMSpatialVelocity(Frame::World(), Frame::World());
  }
  return vels;
}

//==============================================================================
/// This returns the spatial accelerations (6 vecs) of the COMs of each body
/// in world space, concatenated
Eigen::VectorXs Skeleton::getCOMWorldAccelerations()
{
  Eigen::VectorXs accs = Eigen::VectorXs(getNumBodyNodes() * 6);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    accs.segment<6>(i * 6) = getBodyNode(i)->getCOMSpatialAcceleration(
        Frame::World(), Frame::World());
  }
  return accs;
}

//==============================================================================
/// This returns the linear accelerations (3 vecs) of the COMs of each body in
/// world space.
Eigen::VectorXs Skeleton::getCOMWorldLinearAccelerations()
{
  Eigen::VectorXs accs = Eigen::VectorXs(getNumBodyNodes() * 3);
  for (int i = 0; i < getNumBodyNodes(); i++)
  {
    accs.segment<3>(i * 3) = getBodyNode(i)->getCOMLinearAcceleration(
        Frame::World(), Frame::World());
  }
  return accs;
}

//==============================================================================
/// This computes the jacobian of the world velocities for each body with
/// respect to `wrt`
Eigen::MatrixXs Skeleton::getCOMWorldVelocitiesJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(getNumBodyNodes() * 6, dim);

  if (wrt == neural::WithRespectTo::GROUP_COMS)
  {
    for (int i = 0; i < getNumScaleGroups(); i++)
    {
      BodyScaleGroup scaleGroup = getBodyScaleGroup(i);
      for (int b = 0; b < scaleGroup.nodes.size(); b++)
      {
        Eigen::Vector6s localV = scaleGroup.nodes[b]->getSpatialVelocity();
        for (int axis = 0; axis < 3; axis++)
        {
          jac.block<3, 1>(
              scaleGroup.nodes[b]->getIndexInSkeleton() * 6 + 3, i * 3 + axis)
              = scaleGroup.nodes[b]->getWorldTransform().linear()
                * (localV.head<3>().cross(
                    Eigen::Vector3s::Unit(axis)
                    * scaleGroup.flipAxis[b](axis)));
        }
      }
    }
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::GROUP_SCALES)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      bodyNodes[i]->computeJacobianOfCForward(wrt, true);
      jac.block(i * 6, 0, 6, dim) = bodyNodes[i]->mCg_V_p;
      for (int j = 0; j < dim; j++)
      {
        jac.block<3, 1>(i * 6 + 3, j)
            += jac.block<3, 1>(i * 6, j).cross(bodyNodes[i]->getLocalCOM());
      }
      jac.block(i * 6, 0, 6, dim) = math::AdRJac(
          bodyNodes[i]->getWorldTransform(),
          jac.block<6, Eigen::Dynamic>(i * 6, 0, 6, dim));
    }
    // We need to account for the relative rotation, wrt the world frame,
    // because that's an extra step that isn't accounted for in the featherstone
    // algo. We can use the product rule, and add the cross product of the
    // current vel and the rotation axis for each joint.
    if (wrt == neural::WithRespectTo::POSITION)
    {
      const Eigen::MatrixXi& jointParent = getJointParentMap();
      for (int i = 0; i < getNumDofs(); i++)
      {
        int jointIndex = getDof(i)->getJoint()->getJointIndexInSkeleton();
        Eigen::Vector6s worldScrew
            = getDof(i)->getJoint()->getWorldAxisScrewForPosition(
                getDof(i)->getIndexInJoint());
        Eigen::Vector3s worldRot = worldScrew.head<3>();
        for (int b = 0; b < getNumBodyNodes(); b++)
        {
          int bodyJointIndex
              = getBodyNode(b)->getParentJoint()->getJointIndexInSkeleton();
          if (bodyJointIndex == jointIndex
              || jointParent(jointIndex, bodyJointIndex) == 1)
          {
            Eigen::Vector6s spatialVel = getBodyNode(b)->getCOMSpatialVelocity(
                Frame::World(), Frame::World());
            jac.block<3, 1>(b * 6, i) -= spatialVel.head<3>().cross(worldRot);
            jac.block<3, 1>(b * 6 + 3, i)
                -= spatialVel.tail<3>().cross(worldRot);
          }
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This brute forces our world velocities jacobian
Eigen::MatrixXs Skeleton::finiteDifferenceCOMWorldVelocitiesJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs result(getNumBodyNodes() * 6, dim);
  Eigen::VectorXs original = wrt->get(this);

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        wrt->set(this, tweaked);
        perturbed = getCOMWorldVelocities();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, original);

  return result;
}

//==============================================================================
/// This computes the jacobian of the world accelerations for each body with
/// respect to `wrt`
Eigen::MatrixXs Skeleton::getCOMWorldAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(getNumBodyNodes() * 6, dim);

  if (wrt == neural::WithRespectTo::GROUP_COMS)
  {
    for (int i = 0; i < getNumScaleGroups(); i++)
    {
      BodyScaleGroup scaleGroup = getBodyScaleGroup(i);
      for (int b = 0; b < scaleGroup.nodes.size(); b++)
      {
        Eigen::Vector6s localV = scaleGroup.nodes[b]->getSpatialAcceleration();
        for (int axis = 0; axis < 3; axis++)
        {
          jac.block<3, 1>(
              scaleGroup.nodes[b]->getIndexInSkeleton() * 6 + 3, i * 3 + axis)
              = scaleGroup.nodes[b]->getWorldTransform().linear()
                * (localV.head<3>().cross(
                    Eigen::Vector3s::Unit(axis)
                    * scaleGroup.flipAxis[b](axis)));
        }
      }
    }
  }
  else if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      jac.block(i * 6, 0, 6, dim) = getJacobian(bodyNodes[i], Frame::World());
      for (int j = 0; j < getNumDofs(); j++)
      {
        jac.block<3, 1>(i * 6 + 3, j) += jac.block<3, 1>(i * 6, j).cross(
            bodyNodes[i]->getWorldTransform().linear()
            * bodyNodes[i]->getLocalCOM());
      }
    }
  }
  else if (
      wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::GROUP_SCALES)
  {
    std::vector<BodyNode*>& bodyNodes = mSkelCache.mBodyNodes;

    for (int i = 0; i < bodyNodes.size(); i++)
    {
      bodyNodes[i]->computeJacobianOfCForward(wrt, true);
      jac.block(i * 6, 0, 6, dim) = bodyNodes[i]->mCg_dV_p;
      for (int j = 0; j < dim; j++)
      {
        jac.block<3, 1>(i * 6 + 3, j)
            += jac.block<3, 1>(i * 6, j).cross(bodyNodes[i]->getLocalCOM());
      }
      jac.block(i * 6, 0, 6, dim) = math::AdRJac(
          bodyNodes[i]->getWorldTransform(),
          jac.block<6, Eigen::Dynamic>(i * 6, 0, 6, dim));
    }
    // We need to account for the relative rotation, wrt the world frame,
    // because that's an extra step that isn't accounted for in the featherstone
    // algo. We can use the product rule, and add the cross product of the
    // current acc and the rotation axis for each joint.
    if (wrt == neural::WithRespectTo::POSITION)
    {
      const Eigen::MatrixXi& jointParent = getJointParentMap();
      for (int i = 0; i < getNumDofs(); i++)
      {
        int jointIndex = getDof(i)->getJoint()->getJointIndexInSkeleton();
        Eigen::Vector6s worldScrew
            = getDof(i)->getJoint()->getWorldAxisScrewForPosition(
                getDof(i)->getIndexInJoint());
        Eigen::Vector3s worldRot = worldScrew.head<3>();
        for (int b = 0; b < getNumBodyNodes(); b++)
        {
          int bodyJointIndex
              = getBodyNode(b)->getParentJoint()->getJointIndexInSkeleton();
          if (bodyJointIndex == jointIndex
              || jointParent(jointIndex, bodyJointIndex) == 1)
          {
            Eigen::Vector6s spatialAcc
                = getBodyNode(b)->getCOMSpatialAcceleration(
                    Frame::World(), Frame::World());
            jac.block<3, 1>(b * 6, i) -= spatialAcc.head<3>().cross(worldRot);
            jac.block<3, 1>(b * 6 + 3, i)
                -= spatialAcc.tail<3>().cross(worldRot);
          }
        }
      }
    }
  }

  return jac;
}

//==============================================================================
/// This brute forces our world accelerations jacobian
Eigen::MatrixXs Skeleton::finiteDifferenceCOMWorldAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs result(getNumBodyNodes() * 6, dim);
  Eigen::VectorXs original = wrt->get(this);

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        wrt->set(this, tweaked);
        perturbed = getCOMWorldAccelerations();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, original);

  return result;
}

//==============================================================================
Eigen::MatrixXs Skeleton::getCOMWorldLinearAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);

  Eigen::MatrixXs spatialAccJac = getCOMWorldAccelerationsJacobian(wrt);
  Eigen::MatrixXs spatialVelJac = getCOMWorldVelocitiesJacobian(wrt);
  Eigen::VectorXs spatialVel = getCOMWorldVelocities();

  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(getNumBodyNodes() * 3, dim);
  for (int i = 0; i < dim; i++)
  {
    for (int b = 0; b < getNumBodyNodes(); b++)
    {
      Eigen::Vector6s vel = spatialVel.segment<6>(b * 6);
      jac.block<3, 1>(b * 3, i)
          = spatialAccJac.block<3, 1>(b * 6 + 3, i)
            + (vel.head<3>().cross(spatialVelJac.block<3, 1>(b * 6 + 3, i))
               - vel.tail<3>().cross(spatialVelJac.block<3, 1>(b * 6, i)));
    }
  }

  return jac;
}

//==============================================================================
/// This brute forces our world linear accelerations jacobian
Eigen::MatrixXs Skeleton::finiteDifferenceCOMWorldLinearAccelerationsJacobian(
    neural::WithRespectTo* wrt)
{
  int dim = wrt->dim(this);
  Eigen::MatrixXs result(getNumBodyNodes() * 3, dim);
  Eigen::VectorXs original = wrt->get(this);

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        wrt->set(this, tweaked);
        perturbed = getCOMWorldLinearAccelerations();
        return true;
      },
      result,
      eps,
      useRidders);

  // Reset everything how we left it
  wrt->set(this, original);

  return result;
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
/// This solves a simple inverse dynamics problem to get force we need to
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

  dynamics::Joint* joint = getRootJoint();
  const dynamics::FreeJoint* freeJoint
      = dynamic_cast<const dynamics::FreeJoint*>(joint);
  const dynamics::EulerFreeJoint* eulerFreeJoint
      = dynamic_cast<const dynamics::EulerFreeJoint*>(joint);
  if (freeJoint == nullptr && eulerFreeJoint == nullptr)
  {
    std::cout
        << "Error: Skeleton::getContactInverseDynamics() assumes that the root "
           "joint of the skeleton is a FreeJoint ro an EulerFreeJoint. Since"
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

  dynamics::Joint* joint = getRootJoint();
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

Skeleton::MultipleContactCoPProblem
Skeleton::createMultipleContactInverseDynamicsNearCoPProblem(
    const Eigen::VectorXs& nextVel,
    std::vector<const dynamics::BodyNode*> bodies,
    std::vector<Eigen::Vector9s> copWrenchGuesses,
    s_t groundHeight,
    int verticalAxis)
{
  // This is the Jacobian in local body space. We're going to end up applying
  // our contact force in local body space, so this works out.
  Eigen::MatrixXs jacs = Eigen::MatrixXs::Zero(6 * bodies.size(), getNumDofs());
  assert(bodies.size() > 0);

  for (int i = 0; i < bodies.size(); i++)
  {
    // jacs.block(6 * i, 0, 6, getNumDofs())
    //     = getJacobian(bodies[i], Frame::World());

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

  // Precompute matrix factorization
  Eigen::FullPivLU<Eigen::MatrixXs> lu(jacBlock);
  assert(lu.rank() <= 6);
  Eigen::MatrixXs J_null_space = lu.kernel();
  for (int i = 0; i < J_null_space.cols(); i++)
  {
    J_null_space.col(i).normalize();
  }

  Skeleton::MultipleContactCoPProblem problem;
  problem.massTorques = massTorques;
  problem.coriolisAndGravity = coriolisAndGravity;
  problem.copWrenchGuesses = copWrenchGuesses;
  problem.jacs = jacs;
  problem.jacBlock = jacBlock;
  problem.lu = lu;
  problem.J_null_space = J_null_space;
  problem.rootTorque = rootTorque;
  problem.groundHeight = groundHeight;
  problem.verticalAxis = verticalAxis;
  problem.bodies = bodies;
  problem.weightForceToMeters = 0.001;

  return problem;
}

//==============================================================================
Eigen::VectorXs Skeleton::MultipleContactCoPProblem::getInitialGuess()
{
  return jacBlock.completeOrthogonalDecomposition().solve(rootTorque);
  /*
  Eigen::VectorXs xLocal
      = jacBlock.completeOrthogonalDecomposition().solve(rootTorque);
  // std::cout << "Initial guess: " << std::endl << x << std::endl;
  Eigen::VectorXs xGlobal = Eigen::VectorXs::Zero(xLocal.size());
  for (int i = 0; i < xLocal.size(); i++)
  {
    xGlobal.segment<6>(i * 6) = math::dAdInvT(
        bodies[i]->getWorldTransform(), xLocal.segment<6>(i * 6));
  }
  return xGlobal;
  */
}

//==============================================================================
s_t Skeleton::MultipleContactCoPProblem::getLoss(const Eigen::VectorXs& x)
{
  s_t loss = 0.0;
  for (int i = 0; i < copWrenchGuesses.size(); i++)
  {
    assert(!copWrenchGuesses[i].hasNaN());
    assert(!x.segment<6>(i * 6).hasNaN());

    Eigen::Vector9s copWrench = math::projectWrenchToCoP(
        math::dAdInvT(bodies[i]->getWorldTransform(), x.segment<6>(i * 6)),
        groundHeight,
        verticalAxis);
    assert(!copWrench.hasNaN());
    Eigen::Vector3s cop = copWrench.head<3>();
    Eigen::Vector3s copError = copWrenchGuesses[i].head<3>() - cop;
    loss += copError.norm();

    Eigen::Vector3s f = copWrench.tail<3>();
    Eigen::Vector3s fError = copWrenchGuesses[i].tail<3>() - f;
    loss += fError.norm() * weightForceToMeters;
  }
  return loss;
}

//==============================================================================
s_t Skeleton::MultipleContactCoPProblem::getAvgCoPDistance(
    const Eigen::VectorXs& x)
{
  s_t loss = 0.0;
  for (int i = 0; i < copWrenchGuesses.size(); i++)
  {
    assert(!copWrenchGuesses[i].hasNaN());
    assert(!x.segment<6>(i * 6).hasNaN());
    Eigen::Vector9s copWrench = math::projectWrenchToCoP(
        math::dAdInvT(bodies[i]->getWorldTransform(), x.segment<6>(i * 6)),
        groundHeight,
        verticalAxis);
    assert(!copWrench.hasNaN());
    Eigen::Vector3s cop = copWrench.head<3>();
    Eigen::Vector3s copError = copWrenchGuesses[i].head<3>() - cop;
    loss += copError.norm();
  }
  return loss;
}

//==============================================================================
Eigen::Vector6s Skeleton::MultipleContactCoPProblem::getConstraintErrors(
    const Eigen::VectorXs& x)
{
  Eigen::VectorXs contactTorques = jacs.transpose() * x;
  Eigen::VectorXs jointTorques
      = massTorques + coriolisAndGravity - contactTorques;
  return jointTorques.head<6>();
}

//==============================================================================
Eigen::VectorXs Skeleton::MultipleContactCoPProblem::getUnconstrainedGradient(
    const Eigen::VectorXs& x)
{
  Eigen::VectorXs unconstrained = Eigen::VectorXs::Zero(x.size());
  for (int i = 0; i < copWrenchGuesses.size(); i++)
  {
    // Eigen::Matrix6s invT = math::dAdTMatrix(bodies[i]->getWorldTransform());
    Eigen::Vector9s copWrench = math::projectWrenchToCoP(
        math::dAdInvT(bodies[i]->getWorldTransform(), x.segment<6>(i * 6)),
        groundHeight,
        verticalAxis);
    Eigen::Vector3s cop = copWrench.head<3>();
    Eigen::Vector3s copError = cop - copWrenchGuesses[i].head<3>();
    Eigen::Vector3s f = copWrench.tail<3>();
    Eigen::Vector3s fError = f - copWrenchGuesses[i].tail<3>();

    Eigen::Vector9s copWrenchGrad = Eigen::Vector9s::Zero();
    copWrenchGrad.head<3>() = copError.normalized();
    copWrenchGrad.tail<3>() = fError.normalized() * weightForceToMeters;

    Eigen::Vector6s globalWrenchGrad
        = math::getProjectWrenchToCoPJacobian(
              math::dAdInvT(
                  bodies[i]->getWorldTransform(), x.segment<6>(i * 6)),
              groundHeight,
              verticalAxis)
              .transpose()
          * copWrenchGrad;
    Eigen::Vector6s wrenchGrad
        = math::dAdInvTMatrix(bodies[i]->getWorldTransform()).transpose()
          * globalWrenchGrad;
    unconstrained.segment<6>(i * 6) = wrenchGrad;
  }
  return unconstrained;
}

//==============================================================================
Eigen::VectorXs
Skeleton::MultipleContactCoPProblem::finiteDifferenceUnconstrainedGradient(
    const Eigen::VectorXs& x)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(x.size());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs perturbed = x;
        perturbed(i) += eps;

        out = getLoss(perturbed);

        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::VectorXs Skeleton::MultipleContactCoPProblem::projectToNullSpace(
    const Eigen::VectorXs& x)
{
  Eigen::VectorXs dx = Eigen::VectorXs::Zero(x.size());
  for (int i = 0; i < J_null_space.cols(); i++)
  {
    dx += J_null_space.col(i) * J_null_space.col(i).dot(x);
  }
  return dx;
}

//==============================================================================
Eigen::VectorXs Skeleton::MultipleContactCoPProblem::clampToNearestLegalValues(
    const Eigen::VectorXs& x)
{
  return lu.solve(rootTorque - jacBlock * x) + x;
}

//==============================================================================
/// This performs a similar task to getMultipleContactInverseDynamics(), but
/// it resolves ambiguity by attempting to find contact forces that are as
/// closes as possible to the center-of-pressure (CoP) guesses.
Skeleton::MultipleContactInverseDynamicsResult
Skeleton::getMultipleContactInverseDynamicsNearCoP(
    const Eigen::VectorXs& nextVel,
    std::vector<const dynamics::BodyNode*> bodies,
    std::vector<Eigen::Vector6s> bodyWrenchGuesses,
    s_t groundHeight,
    int verticalAxis,
    s_t weightForceToMeters,
    bool logOutput)
{
  MultipleContactInverseDynamicsResult result;
  result.skel = this;
  result.contactBodies = bodies;
  result.pos = getPositions();
  result.vel = getVelocities();
  result.nextVel = nextVel;

  dynamics::Joint* joint = getRootJoint();
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
  std::vector<Eigen::Vector9s> copWrenchGuesses;
  for (int i = 0; i < bodyWrenchGuesses.size(); i++)
  {
    Eigen::Vector9s copWrench = math::projectWrenchToCoP(
        math::dAdInvT(bodies[i]->getWorldTransform(), bodyWrenchGuesses[i]),
        groundHeight,
        verticalAxis);
    /*
#ifndef NDEBUG
    if ((copWrench.head<3>() - copGuesses[i]).norm() > 1e-8)
    {
      std::cout << "Failed to recover CoP complete wrench!";
      std::cout << "Original guess:" << std::endl << copGuesses[i] << std::endl;
      std::cout << "Wrench guess:" << std::endl
                << copWrench.head<3>() << std::endl;
      std::cout << "Diff:" << std::endl
                << copWrench.head<3>() - copGuesses[i] << std::endl;
      assert((copWrench.head<3>() - copGuesses[i]).norm() < 1e-8);
    }
#endif
    */
    copWrenchGuesses.push_back(copWrench);
  }

  MultipleContactCoPProblem problem
      = createMultipleContactInverseDynamicsNearCoPProblem(
          nextVel, bodies, copWrenchGuesses, groundHeight, verticalAxis);
  problem.weightForceToMeters = weightForceToMeters;

  // Eigen::VectorXs x = problem.getInitialGuess();

  auto initialResult
      = getMultipleContactInverseDynamics(nextVel, bodies, bodyWrenchGuesses);

  // Find an initial guess that's legal, and we'll only take legal steps from
  // here
  Eigen::VectorXs x = Eigen::VectorXs::Zero(bodies.size() * 6);
  for (int i = 0; i < bodies.size(); i++)
  {
    x.segment<6>(i * 6) = initialResult.contactWrenches[i];
  }

#ifndef NDEBUG
  Eigen::VectorXs constraintErrors = problem.getConstraintErrors(x);
  if (constraintErrors.norm() > 1e-10)
  {
    std::cout << "Got an initial constraint error!" << std::endl
              << constraintErrors << std::endl;
    assert(constraintErrors.norm() <= 1e-10);
  }
#endif

  // 1.5. Compute initial loss
  s_t lastLoss = problem.getLoss(x);
  assert(!isnan(lastLoss));

#ifndef NDEBUG
  std::cout << "Initial dist " << problem.getAvgCoPDistance(x) << std::endl;
#endif

#ifndef NDEBUG
  s_t lastDist = problem.getAvgCoPDistance(x);
#endif

  s_t alpha = 0.01;
  for (int t = 0; t < 400; t++)
  {
    if (logOutput)
      std::cout << "Iteration " << t << ": " << lastLoss << std::endl;
    // 1.2. Arive at an unconstrained grad x
    Eigen::VectorXs unconstrained = problem.getUnconstrainedGradient(x);

    // 1.3. Constrain grad x to only operate in the null space of J
    Eigen::VectorXs dx = problem.projectToNullSpace(unconstrained);

    // 1.4. Update x
    Eigen::VectorXs proposedX = x - dx * alpha;

#ifndef NDEBUG
    Eigen::VectorXs constraintErrors = problem.getConstraintErrors(proposedX);
    if (constraintErrors.norm() > 1e-10)
    {
      std::cout << "Got a constraint violation!" << std::endl
                << constraintErrors << std::endl;
      assert(constraintErrors.norm() <= 1e-10);
    }
#endif

    /*
    Eigen::VectorXs proposedX
        = problem.clampToNearestLegalValues(x - unconstrained * alpha);
        */

    // 1.5. Compute loss
    s_t loss = problem.getLoss(proposedX);
    assert(!isnan(loss));

    if (loss < lastLoss)
    {
      if (logOutput)
        std::cout << "Decreased loss by " << (lastLoss - loss) << std::endl;
      lastLoss = loss;
      x = proposedX;
      alpha *= 1.2;
#ifndef NDEBUG
      if (logOutput)
      {
        s_t newDist = problem.getAvgCoPDistance(x);
        std::cout << "New dist " << newDist << ", decreased by "
                  << (lastDist - newDist) << std::endl;
        lastDist = newDist;
      }
#endif
    }
    else
    {
      if (logOutput)
        std::cout << "   Cutting back step size to " << alpha << std::endl;
      alpha *= 0.5;
      if (alpha < 1e-15)
      {
        if (logOutput)
        {
          std::cout << "Step size got below 1e-15! This suggests we stalled! "
                       "Exiting optimizer."
                    << std::endl;
          std::cout << "Latest x: " << std::endl << x << std::endl;
          std::cout << "Latest grad: " << std::endl
                    << unconstrained << std::endl;
        }
        break;
      }
    }
  }

  // Polish the final solution back into legal space (take a least-squares
  // closest approx)
  Eigen::VectorXs clampedX = problem.clampToNearestLegalValues(x);
#ifndef NDEBUG
  if ((clampedX - x).squaredNorm() > 1e-10)
  {
    std::cout << "Ended up with a very different final x!" << std::endl;
    assert(false);
  }
#endif
  x = clampedX;

#ifndef NDEBUG
  std::cout << "Final dist " << problem.getAvgCoPDistance(x) << std::endl;
#endif

  /////////////////////////////////////////////////////
  // 2. Write out the results
  result.contactWrenches = std::vector<Eigen::Vector6s>();
  for (int i = 0; i < bodies.size(); i++)
  {
    // result.contactWrenches.push_back(
    //     math::dAdT(bodies[i]->getWorldTransform(), x.segment<6>(i * 6)));
    result.contactWrenches.push_back(x.segment<6>(i * 6));
  }
  Eigen::VectorXs contactTorques = problem.jacs.transpose() * x;
  result.jointTorques
      = problem.massTorques + problem.coriolisAndGravity - contactTorques;
  assert(result.jointTorques.head<6>().norm() < 1e-10);
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
  Eigen::MatrixXs result(6, getNumDofs());

  s_t eps = useRidders ? 1e-3 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        s_t original = getPosition(dof);
        setPosition(dof, original + eps);
        perturbed = Eigen::Vector6s::Zero();
        perturbed.head<3>() = math::logMap(_node->getWorldTransform().linear());
        perturbed.tail<3>() = _node->getWorldTransform() * _localOffset;
        setPosition(dof, original);
        return true;
      },
      result,
      eps,
      useRidders);

  return math::Jacobian(result);
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
  bool hasAnyZeroDof = false;
  for (int i = 0; i < getNumJoints(); i++)
  {
    if (getJoint(i)->getNumDofs() == 0)
    {
      hasAnyZeroDof = true;
      break;
    }
  }

  if (hasAnyZeroDof)
  {
    return getInvMassMatrix() * x;
  }

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

  // TODO: cache this so we don't have to check every time

  bool hasZeroDofJoint = false;
  for (auto node : cache.mBodyNodes)
  {
    if (node->getParentJoint()->getNumDofs() == 0)
    {
      hasZeroDofJoint = true;
      break;
    }
  }

  // Implicitly computing the inverse mass matrix currently doesn't work if you
  // have any WeldJoints in the hierarchy. To get around this, we can just
  // invert the mass matrix, which could be slower, but works fine.

  if (hasZeroDofJoint)
  {
    updateMassMatrix(_treeIdx);
    cache.mInvM = cache.mM.llt().solve(Eigen::MatrixXs::Identity(dof, dof));
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
    mDofParentMap(true),
    mJointParentMap(true),
    mSupportVersion(0)
{
  // Do nothing
}

} // namespace dynamics
} // namespace dart
