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

#include "dart/dynamics/BodyNode.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "dart/common/Console.hpp"
#include "dart/common/StlHelpers.hpp"
#include "dart/dynamics/Chain.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/EndEffector.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Marker.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SoftBodyNode.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
template <
    class DataType,
    std::unique_ptr<DataType> (Node::*getData)() const,
    typename VectorType = common::CloneableVector<std::unique_ptr<DataType> >,
    typename DataMap = std::map<std::type_index, std::unique_ptr<VectorType> > >
static void extractDataFromNodeTypeMap(
    DataMap& dataMap, const BodyNode::NodeMap& nodeMap)
{
  for (const auto& node_it : nodeMap)
  {
    const std::vector<Node*>& nodes = node_it.second;

    std::pair<typename DataMap::iterator, bool> insertion
        = dataMap.insert(typename DataMap::value_type(node_it.first, nullptr));

    typename DataMap::iterator& it = insertion.first;

    std::unique_ptr<VectorType>& data = it->second;
    if (!data)
      data = std::make_unique<VectorType>();

    data->getVector().resize(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
      std::unique_ptr<DataType>& datum = data->getVector()[i];
      datum = (nodes[i]->*getData)();
    }
  }
}

//==============================================================================
template <
    class DataType,
    void (Node::*setData)(const DataType&),
    typename VectorType = common::CloneableVector<std::unique_ptr<DataType> >,
    typename DataMap = std::map<std::type_index, std::unique_ptr<VectorType> > >
static void setNodesFromDataTypeMap(
    BodyNode::NodeMap& nodeMap, const DataMap& dataMap)
{
  typename BodyNode::NodeMap::iterator node_it = nodeMap.begin();
  typename DataMap::const_iterator data_it = dataMap.begin();

  while (nodeMap.end() != node_it && dataMap.end() != data_it)
  {
    if (node_it->first == data_it->first)
    {
      const std::vector<Node*>& node_vec = node_it->second;
      const std::vector<std::unique_ptr<DataType> >& data_vec
          = data_it->second->getVector();

      // TODO(MXG): Should we report if the dimensions are mismatched?
      std::size_t stop = std::min(node_vec.size(), data_vec.size());
      for (std::size_t i = 0; i < stop; ++i)
      {
        if (data_vec[i])
          (node_vec[i]->*setData)(*data_vec[i]);
      }

      ++node_it;
      ++data_it;
    }
    else if (node_it->first < data_it->first)
    {
      ++node_it;
    }
    else
    {
      ++data_it;
    }
  }
}

//==============================================================================
SkeletonRefCountingBase::SkeletonRefCountingBase()
  : mReferenceCount(0),
    mLockedSkeleton(std::make_shared<MutexedWeakSkeletonPtr>())
{
  // Do nothing
}

//==============================================================================
void SkeletonRefCountingBase::incrementReferenceCount() const
{
  int previous = std::atomic_fetch_add(&mReferenceCount, 1);
  if (0 == previous)
    mReferenceSkeleton = mSkeleton.lock();
}

//==============================================================================
void SkeletonRefCountingBase::decrementReferenceCount() const
{
  int previous = std::atomic_fetch_sub(&mReferenceCount, 1);
  if (1 == previous)
    mReferenceSkeleton = nullptr;
}

//==============================================================================
SkeletonPtr SkeletonRefCountingBase::getSkeleton()
{
  return mSkeleton.lock();
}

//==============================================================================
ConstSkeletonPtr SkeletonRefCountingBase::getSkeleton() const
{
  return mSkeleton.lock();
}

/// SKEL_SET_FLAGS : Lock a Skeleton pointer and activate dirty flags of X for
/// the tree that this BodyNode belongs to, as well as the flag for the Skeleton
/// overall
#define SKEL_SET_FLAGS(X)                                                      \
  {                                                                            \
    SkeletonPtr skel = getSkeleton();                                          \
    if (skel)                                                                  \
    {                                                                          \
      skel->mTreeCache[mTreeIndex].mDirty.X = true;                            \
      skel->mSkelCache.mDirty.X = true;                                        \
    }                                                                          \
  }

/// SET_FLAGS : A version of SKEL_SET_FLAGS that assumes a SkeletonPtr named
/// 'skel' has already been locked
#define SET_FLAGS(X)                                                           \
  skel->mTreeCache[mTreeIndex].mDirty.X = true;                                \
  skel->mSkelCache.mDirty.X = true;

/// CHECK_FLAG : Check if the dirty flag X for the tree of this BodyNode is
/// active
#define CHECK_FLAG(X) skel->mTreeCache[mTreeIndex].mDirty.X

//==============================================================================
typedef std::set<Entity*> EntityPtrSet;

//==============================================================================
std::size_t BodyNode::msBodyNodeCount = 0;

namespace detail {

//==============================================================================
void setAllNodeStates(BodyNode* bodyNode, const AllNodeStates& states)
{
  bodyNode->setAllNodeStates(states);
}

//==============================================================================
AllNodeStates getAllNodeStates(const BodyNode* bodyNode)
{
  return bodyNode->getAllNodeStates();
}

//==============================================================================
void setAllNodeProperties(
    BodyNode* bodyNode, const AllNodeProperties& properties)
{
  bodyNode->setAllNodeProperties(properties);
}

//==============================================================================
AllNodeProperties getAllNodeProperties(const BodyNode* bodyNode)
{
  return bodyNode->getAllNodeProperties();
}

//==============================================================================
BodyNodeState::BodyNodeState(const Eigen::Vector6s& Fext) : mFext(Fext)
{
  // Do nothing
}

//==============================================================================
BodyNodeAspectProperties::BodyNodeAspectProperties(
    const std::string& name,
    const Inertia& _inertia,
    bool _isCollidable,
    s_t _frictionCoeff,
    s_t _restitutionCoeff,
    bool _gravityMode)
  : mName(name),
    mInertia(_inertia),
    mIsCollidable(_isCollidable),
    mFrictionCoeff(_frictionCoeff),
    mRestitutionCoeff(_restitutionCoeff),
    mGravityMode(_gravityMode)
{
  // Do nothing
}

} // namespace detail

//==============================================================================
BodyNode::~BodyNode()
{
  // Delete all Nodes
  mNodeMap.clear();
  mNodeDestructors.clear();

  delete mParentJoint;
}

//==============================================================================
SoftBodyNode* BodyNode::asSoftBodyNode()
{
  return nullptr;
}

//==============================================================================
const SoftBodyNode* BodyNode::asSoftBodyNode() const
{
  return nullptr;
}

//==============================================================================
void BodyNode::setAllNodeStates(const AllNodeStates& states)
{
  setNodesFromDataTypeMap<Node::State, &Node::setNodeState>(
      mNodeMap, states.getMap());
}

//==============================================================================
BodyNode::AllNodeStates BodyNode::getAllNodeStates() const
{
  detail::NodeStateMap nodeStates;
  extractDataFromNodeTypeMap<Node::State, &Node::getNodeState>(
      nodeStates, mNodeMap);
  return nodeStates;
}

//==============================================================================
void BodyNode::setAllNodeProperties(const AllNodeProperties& properties)
{
  setNodesFromDataTypeMap<Node::Properties, &Node::setNodeProperties>(
      mNodeMap, properties.getMap());
}

//==============================================================================
BodyNode::AllNodeProperties BodyNode::getAllNodeProperties() const
{
  // TODO(MXG): Make a version of this function that will fill in a
  // NodeProperties instance instead of creating a new one
  detail::NodePropertiesMap nodeProperties;
  extractDataFromNodeTypeMap<Node::Properties, &Node::getNodeProperties>(
      nodeProperties, mNodeMap);
  return nodeProperties;
}

//==============================================================================
void BodyNode::setProperties(const CompositeProperties& _properties)
{
  setCompositeProperties(_properties);
}

//==============================================================================
void BodyNode::setProperties(const AspectProperties& _properties)
{
  setAspectProperties(_properties);
}

//==============================================================================
void BodyNode::setAspectState(const AspectState& state)
{
  if (mAspectState.mFext != state.mFext)
  {
    mAspectState.mFext = state.mFext;
    SKEL_SET_FLAGS(mExternalForces);
  }
}

//==============================================================================
void BodyNode::setAspectProperties(const AspectProperties& properties)
{
  setName(properties.mName);
  setInertia(properties.mInertia);
  setGravityMode(properties.mGravityMode);
  setFrictionCoeff(properties.mFrictionCoeff);
  setRestitutionCoeff(properties.mRestitutionCoeff);
}

//==============================================================================
BodyNode::Properties BodyNode::getBodyNodeProperties() const
{
  return getCompositeProperties();
}

//==============================================================================
void BodyNode::copy(const BodyNode& otherBodyNode)
{
  if (this == &otherBodyNode)
    return;

  setCompositeProperties(otherBodyNode.getCompositeProperties());
}

//==============================================================================
void BodyNode::copy(const BodyNode* otherBodyNode)
{
  if (nullptr == otherBodyNode)
    return;

  copy(*otherBodyNode);
}

//==============================================================================
BodyNode& BodyNode::operator=(const BodyNode& otherBodyNode)
{
  copy(otherBodyNode);
  return *this;
}

//==============================================================================
void BodyNode::duplicateNodes(const BodyNode* otherBodyNode)
{
  if (nullptr == otherBodyNode)
  {
    dterr << "[BodyNode::duplicateNodes] You have asked to duplicate the Nodes "
          << "of a nullptr, which is not allowed!\n";
    assert(false);
    return;
  }

  const NodeMap& otherMap = otherBodyNode->mNodeMap;
  for (const auto& vec : otherMap)
  {
    for (const auto& node : vec.second)
      node->cloneNode(this)->attach();
  }
}

//==============================================================================
void BodyNode::matchNodes(const BodyNode* otherBodyNode)
{
  if (nullptr == otherBodyNode)
  {
    dterr << "[BodyNode::matchNodes] You have asked to match the Nodes of a "
          << "nullptr, which is not allowed!\n";
    assert(false);
    return;
  }

  for (auto& cleaner : mNodeDestructors)
    cleaner->getNode()->stageForRemoval();

  duplicateNodes(otherBodyNode);
}

//==============================================================================
const std::string& BodyNode::setName(const std::string& _name)
{
  // If it already has the requested name, do nothing
  if (mAspectProperties.mName == _name)
    return mAspectProperties.mName;

  const std::string oldName = mAspectProperties.mName;

  // If the BodyNode belongs to a Skeleton, consult the Skeleton's NameManager
  const SkeletonPtr& skel = getSkeleton();
  if (skel)
  {
    skel->mNameMgrForBodyNodes.removeName(mAspectProperties.mName);
    SoftBodyNode* softnode = dynamic_cast<SoftBodyNode*>(this);
    if (softnode)
      skel->mNameMgrForSoftBodyNodes.removeName(mAspectProperties.mName);

    mAspectProperties.mName = _name;
    skel->addEntryToBodyNodeNameMgr(this);

    if (softnode)
      skel->addEntryToSoftBodyNodeNameMgr(softnode);
  }
  else
  {
    mAspectProperties.mName = _name;
  }

  incrementVersion();
  Entity::mNameChangedSignal.raise(this, oldName, mAspectProperties.mName);

  // Return the final name (which might have been altered by the Skeleton's
  // NameManager)
  return mAspectProperties.mName;
}

//==============================================================================
const std::string& BodyNode::getName() const
{
  return mAspectProperties.mName;
}

//==============================================================================
void BodyNode::setGravityMode(bool _gravityMode)
{
  if (mAspectProperties.mGravityMode == _gravityMode)
    return;

  mAspectProperties.mGravityMode = _gravityMode;

  SKEL_SET_FLAGS(mGravityForces);
  SKEL_SET_FLAGS(mCoriolisAndGravityForces);

  incrementVersion();
}

//==============================================================================
bool BodyNode::getGravityMode() const
{
  return mAspectProperties.mGravityMode;
}

//==============================================================================
bool BodyNode::isCollidable() const
{
  return mAspectProperties.mIsCollidable;
}

//==============================================================================
void BodyNode::setCollidable(bool _isCollidable)
{
  mAspectProperties.mIsCollidable = _isCollidable;
}

//==============================================================================
void checkMass(const BodyNode& bodyNode, const s_t mass)
{
  if (mass <= 0.0)
  {
    dtwarn << "[BodyNode] A negative or zero mass [" << mass
           << "] is set to BodyNode [" << bodyNode.getName()
           << "], which can cause invalid physical behavior or segfault. "
           << "Consider setting positive value instead.\n";
  }
}

//==============================================================================
void BodyNode::setMass(const s_t mass)
{
  checkMass(*this, mass);

  mAspectProperties.mInertia.setMass(mass);

  dirtyArticulatedInertia();
  const SkeletonPtr& skel = getSkeleton();
  if (skel)
    skel->updateTotalMass();
}

//==============================================================================
s_t BodyNode::getMass() const
{
  return mAspectProperties.mInertia.getMass();
}

//==============================================================================
void BodyNode::setMomentOfInertia(
    s_t _Ixx, s_t _Iyy, s_t _Izz, s_t _Ixy, s_t _Ixz, s_t _Iyz)
{
  mAspectProperties.mInertia.setMoment(_Ixx, _Iyy, _Izz, _Ixy, _Ixz, _Iyz);

  dirtyArticulatedInertia();
}

//==============================================================================
void BodyNode::getMomentOfInertia(
    s_t& _Ixx, s_t& _Iyy, s_t& _Izz, s_t& _Ixy, s_t& _Ixz, s_t& _Iyz) const
{
  _Ixx = mAspectProperties.mInertia.getParameter(Inertia::I_XX);
  _Iyy = mAspectProperties.mInertia.getParameter(Inertia::I_YY);
  _Izz = mAspectProperties.mInertia.getParameter(Inertia::I_ZZ);

  _Ixy = mAspectProperties.mInertia.getParameter(Inertia::I_XY);
  _Ixz = mAspectProperties.mInertia.getParameter(Inertia::I_XZ);
  _Iyz = mAspectProperties.mInertia.getParameter(Inertia::I_YZ);
}

//==============================================================================
const Eigen::Matrix6s& BodyNode::getSpatialInertia() const
{
  return mAspectProperties.mInertia.getSpatialTensor();
}

//==============================================================================
void BodyNode::setInertia(const Inertia& inertia)
{
  if (inertia == mAspectProperties.mInertia)
    return;

  checkMass(*this, inertia.getMass());

  mAspectProperties.mInertia = inertia;

  dirtyArticulatedInertia();
  const SkeletonPtr& skel = getSkeleton();
  if (skel)
    skel->updateTotalMass();

  incrementVersion();
}

//==============================================================================
const Inertia& BodyNode::getInertia() const
{
  return mAspectProperties.mInertia;
}

//==============================================================================
const math::Inertia& BodyNode::getArticulatedInertia() const
{
  const ConstSkeletonPtr& skel = getSkeleton();
  if (skel && CHECK_FLAG(mArticulatedInertia))
    skel->updateArticulatedInertia(mTreeIndex);

  return mArtInertia;
}

//==============================================================================
const math::Inertia& BodyNode::getArticulatedInertiaImplicit() const
{
  const ConstSkeletonPtr& skel = getSkeleton();
  if (skel && CHECK_FLAG(mArticulatedInertia))
    skel->updateArticulatedInertia(mTreeIndex);

  return mArtInertiaImplicit;
}

//==============================================================================
void BodyNode::setLocalCOM(const Eigen::Vector3s& _com)
{
  mAspectProperties.mInertia.setLocalCOM(_com);

  dirtyArticulatedInertia();
}

//==============================================================================
const Eigen::Vector3s& BodyNode::getLocalCOM() const
{
  return mAspectProperties.mInertia.getLocalCOM();
}

//==============================================================================
Eigen::Vector3s BodyNode::getCOM(const Frame* _withRespectTo) const
{
  return getTransform(_withRespectTo) * getLocalCOM();
}

//==============================================================================
Eigen::Vector3s BodyNode::getCOMLinearVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getLinearVelocity(getLocalCOM(), _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector6s BodyNode::getCOMSpatialVelocity() const
{
  return getSpatialVelocity(getLocalCOM());
}

//==============================================================================
Eigen::Vector6s BodyNode::getCOMSpatialVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getSpatialVelocity(getLocalCOM(), _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector3s BodyNode::getCOMLinearAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getLinearAcceleration(getLocalCOM(), _relativeTo, _inCoordinatesOf);
}

//==============================================================================
Eigen::Vector6s BodyNode::getCOMSpatialAcceleration() const
{
  return getSpatialAcceleration(getLocalCOM());
}

//==============================================================================
Eigen::Vector6s BodyNode::getCOMSpatialAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getSpatialAcceleration(getLocalCOM(), _relativeTo, _inCoordinatesOf);
}

//==============================================================================
void BodyNode::setFrictionCoeff(s_t _coeff)
{
  if (mAspectProperties.mFrictionCoeff == _coeff)
    return;

  assert(
      0.0 <= _coeff && "Coefficient of friction should be non-negative value.");
  mAspectProperties.mFrictionCoeff = _coeff;

  incrementVersion();
}

//==============================================================================
s_t BodyNode::getFrictionCoeff() const
{
  return mAspectProperties.mFrictionCoeff;
}

//==============================================================================
void BodyNode::setRestitutionCoeff(s_t _coeff)
{
  if (_coeff == mAspectProperties.mRestitutionCoeff)
    return;

  assert(
      0.0 <= _coeff && _coeff <= 1.0
      && "Coefficient of restitution should be in range of [0, 1].");
  mAspectProperties.mRestitutionCoeff = _coeff;

  incrementVersion();
}

//==============================================================================
s_t BodyNode::getRestitutionCoeff() const
{
  return mAspectProperties.mRestitutionCoeff;
}

//==============================================================================
std::size_t BodyNode::getIndexInSkeleton() const
{
  return mIndexInSkeleton;
}

//==============================================================================
std::size_t BodyNode::getIndexInTree() const
{
  return mIndexInTree;
}

//==============================================================================
std::size_t BodyNode::getTreeIndex() const
{
  return mTreeIndex;
}

//==============================================================================
static bool checkSkeletonNodeAgreement(
    const BodyNode* _bodyNode,
    const ConstSkeletonPtr& _newSkeleton,
    const BodyNode* _newParent,
    const std::string& _function,
    const std::string& _operation)
{
  if (nullptr == _newSkeleton)
  {
    dterr << "[BodyNode::" << _function << "] Attempting to " << _operation
          << " a BodyNode tree starting "
          << "from [" << _bodyNode->getName() << "] in the Skeleton named ["
          << _bodyNode->getSkeleton()->getName()
          << "] into a nullptr Skeleton.\n";
    return false;
  }

  if (_newParent && _newSkeleton != _newParent->getSkeleton())
  {
    dterr << "[BodyNode::" << _function << "] Mismatch between the specified "
          << "Skeleton [" << _newSkeleton->getName() << "] (" << _newSkeleton
          << ") and the specified new parent BodyNode ["
          << _newParent->getName() << "] whose actual Skeleton is named ["
          << _newParent->getSkeleton()->getName() << "] ("
          << _newParent->getSkeleton() << ") while attempting to " << _operation
          << " the BodyNode [" << _bodyNode->getName() << "] from the "
          << "Skeleton named [" << _bodyNode->getSkeleton()->getName() << "] ("
          << _bodyNode->getSkeleton() << ").\n";
    return false;
  }

  return true;
}

//==============================================================================
SkeletonPtr BodyNode::remove(const std::string& _name)
{
  return split(_name);
}

//==============================================================================
bool BodyNode::moveTo(BodyNode* _newParent)
{
  if (nullptr == _newParent)
    return getSkeleton()->moveBodyNodeTree(
        getParentJoint(), this, getSkeleton(), nullptr);
  else
    return getSkeleton()->moveBodyNodeTree(
        getParentJoint(), this, _newParent->getSkeleton(), _newParent);
}

//==============================================================================
bool BodyNode::moveTo(const SkeletonPtr& _newSkeleton, BodyNode* _newParent)
{
  if (checkSkeletonNodeAgreement(
          this, _newSkeleton, _newParent, "moveTo", "move"))
  {
    return getSkeleton()->moveBodyNodeTree(
        getParentJoint(), this, _newSkeleton, _newParent);
  }

  return false;
}

//==============================================================================
SkeletonPtr BodyNode::split(const std::string& _skeletonName)
{
  const SkeletonPtr& skel
      = Skeleton::create(getSkeleton()->getAspectProperties());
  skel->setName(_skeletonName);
  moveTo(skel, nullptr);
  return skel;
}

//==============================================================================
std::pair<Joint*, BodyNode*> BodyNode::copyTo(
    BodyNode* _newParent, bool _recursive)
{
  if (nullptr == _newParent)
    return getSkeleton()->cloneBodyNodeTree(
        nullptr, this, getSkeleton(), nullptr, _recursive);
  else
    return getSkeleton()->cloneBodyNodeTree(
        nullptr, this, _newParent->getSkeleton(), _newParent, _recursive);
}

//==============================================================================
std::pair<Joint*, BodyNode*> BodyNode::copyTo(
    const SkeletonPtr& _newSkeleton,
    BodyNode* _newParent,
    bool _recursive) const
{
  if (checkSkeletonNodeAgreement(
          this, _newSkeleton, _newParent, "copyTo", "copy"))
  {
    return getSkeleton()->cloneBodyNodeTree(
        nullptr, this, _newSkeleton, _newParent, _recursive);
  }

  return std::pair<Joint*, BodyNode*>(nullptr, nullptr);
}

//==============================================================================
SkeletonPtr BodyNode::copyAs(
    const std::string& _skeletonName, bool _recursive) const
{
  const SkeletonPtr& skel
      = Skeleton::create(getSkeleton()->getAspectProperties());
  skel->setName(_skeletonName);
  copyTo(skel, nullptr, _recursive);
  return skel;
}

//==============================================================================
SkeletonPtr BodyNode::getSkeleton()
{
  return mSkeleton.lock();
}

//==============================================================================
ConstSkeletonPtr BodyNode::getSkeleton() const
{
  return mSkeleton.lock();
}

//==============================================================================
Joint* BodyNode::getParentJoint()
{
  return mParentJoint;
}

//==============================================================================
const Joint* BodyNode::getParentJoint() const
{
  return mParentJoint;
}

//==============================================================================
BodyNode* BodyNode::getParentBodyNode()
{
  return mParentBodyNode;
}

//==============================================================================
const BodyNode* BodyNode::getParentBodyNode() const
{
  return mParentBodyNode;
}

//==============================================================================
void BodyNode::addChildBodyNode(BodyNode* _body)
{
  assert(_body != nullptr);

  if (std::find(mChildBodyNodes.begin(), mChildBodyNodes.end(), _body)
      != mChildBodyNodes.end())
  {
    dtwarn << "[BodyNode::addChildBodyNode] Attempting to add a BodyNode '"
           << _body->getName() << "' as a child BodyNode of '" << getName()
           << "', which is already its parent." << std::endl;
    return;
  }

  mChildBodyNodes.push_back(_body);
  _body->mParentBodyNode = this;
  _body->changeParentFrame(this);
}

//==============================================================================
std::size_t BodyNode::getNumChildBodyNodes() const
{
  return mChildBodyNodes.size();
}

//==============================================================================
BodyNode* BodyNode::getChildBodyNode(std::size_t _index)
{
  return common::getVectorObjectIfAvailable<BodyNode*>(_index, mChildBodyNodes);
}

//==============================================================================
const BodyNode* BodyNode::getChildBodyNode(std::size_t _index) const
{
  return common::getVectorObjectIfAvailable<BodyNode*>(_index, mChildBodyNodes);
}

//==============================================================================
std::size_t BodyNode::getNumChildJoints() const
{
  return mChildBodyNodes.size();
}

//==============================================================================
Joint* BodyNode::getChildJoint(std::size_t _index)
{
  BodyNode* childBodyNode = getChildBodyNode(_index);

  if (childBodyNode)
    return childBodyNode->getParentJoint();
  else
    return nullptr;
}

//==============================================================================
const Joint* BodyNode::getChildJoint(std::size_t _index) const
{
  return const_cast<BodyNode*>(this)->getChildJoint(_index);
}

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_DEFINITIONS(BodyNode, ShapeNode)

//==============================================================================
const std::vector<ShapeNode*> BodyNode::getShapeNodes()
{
  const auto numShapeNodes = getNumShapeNodes();

  std::vector<ShapeNode*> shapeNodes(numShapeNodes);

  for (auto i = 0u; i < numShapeNodes; ++i)
    shapeNodes[i] = getShapeNode(i);

  return shapeNodes;
}

//==============================================================================
const std::vector<const ShapeNode*> BodyNode::getShapeNodes() const
{
  const auto numShapeNodes = getNumShapeNodes();

  std::vector<const ShapeNode*> shapeNodes(numShapeNodes);

  for (auto i = 0u; i < numShapeNodes; ++i)
    shapeNodes[i] = getShapeNode(i);

  return shapeNodes;
}

//==============================================================================
void BodyNode::removeAllShapeNodes()
{
  auto shapeNodes = getShapeNodes();
  for (auto shapeNode : shapeNodes)
    shapeNode->remove();
}

//==============================================================================
/// This gets all the vertices from any mesh colliders, in local space
std::vector<Eigen::Vector3s> BodyNode::getLocalVertices() const
{
  std::vector<Eigen::Vector3s> verts;
  for (int k = 0; k < getNumShapeNodes(); k++)
  {
    if (getShapeNode(k)->getShape()->getType()
        == dynamics::MeshShape::getStaticType())
    {
      const dynamics::MeshShape* meshShape
          = dynamic_cast<const dynamics::MeshShape*>(
              getShapeNode(k)->getShape().get());
      std::vector<Eigen::Vector3s> meshVerts = meshShape->getVertices();
      verts.insert(verts.end(), meshVerts.begin(), meshVerts.end());
    }
  }
  return verts;
}

//==============================================================================
BodyNode::MovingVertex::MovingVertex(
    Eigen::Vector3s pos,
    Eigen::Vector3s vel,
    Eigen::Vector3s accel,
    const BodyNode* bodyNode,
    int timestep)
  : pos(pos), vel(vel), accel(accel), bodyNode(bodyNode), timestep(timestep)
{
}

//==============================================================================
/// This is useful for doing contact inference. You can get all the vertices
/// on all of the moving bodies, translated into world space.
std::vector<BodyNode::MovingVertex> BodyNode::getMovingVerticesInWorldSpace(
    int timestep) const
{
  std::vector<BodyNode::MovingVertex> movingVertices;
  std::vector<Eigen::Vector3s> vertices = getLocalVertices();

  for (int i = 0; i < vertices.size(); i++)
  {
    Eigen::Vector3s vert = vertices[i];
    Eigen::Vector3s worldPos = getWorldTransform() * vert;
    Eigen::Vector3s worldVel
        = getLinearVelocity(vert, Frame::World(), Frame::World());
    Eigen::Vector3s worldAccel
        = getLinearAcceleration(vert, Frame::World(), Frame::World());
    movingVertices.emplace_back(worldPos, worldVel, worldAccel, this, timestep);
  }

  return movingVertices;
}

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_DEFINITIONS(BodyNode, EndEffector)

//==============================================================================
EndEffector* BodyNode::createEndEffector(
    const EndEffector::BasicProperties& _properties)
{
  return createNode<EndEffector>(_properties);
}

//==============================================================================
EndEffector* BodyNode::createEndEffector(const std::string& _name)
{
  EndEffector::BasicProperties properties;
  properties.mName = _name;

  return createNode<EndEffector>(properties);
}

//==============================================================================
EndEffector* BodyNode::createEndEffector(const char* _name)
{
  return createEndEffector(std::string(_name));
}

//==============================================================================
DART_BAKE_SPECIALIZED_NODE_DEFINITIONS(BodyNode, Marker)

//==============================================================================
Marker* BodyNode::createMarker(
    const std::string& name,
    const Eigen::Vector3s& position,
    const Eigen::Vector4s& color)
{
  Marker::BasicProperties properties;
  properties.mName = name;
  properties.mRelativeTf.translation() = position;
  properties.mColor = color;

  return createNode<Marker>(properties);
}

//==============================================================================
Marker* BodyNode::createMarker(const Marker::BasicProperties& properties)
{
  return createNode<Marker>(properties);
}

//==============================================================================
bool BodyNode::dependsOn(std::size_t _genCoordIndex) const
{
  return std::binary_search(
      mDependentGenCoordIndices.begin(),
      mDependentGenCoordIndices.end(),
      _genCoordIndex);
}

//==============================================================================
std::size_t BodyNode::getNumDependentGenCoords() const
{
  return mDependentGenCoordIndices.size();
}

//==============================================================================
std::size_t BodyNode::getDependentGenCoordIndex(std::size_t _arrayIndex) const
{
  assert(_arrayIndex < mDependentGenCoordIndices.size());

  return mDependentGenCoordIndices[_arrayIndex];
}

//==============================================================================
const std::vector<std::size_t>& BodyNode::getDependentGenCoordIndices() const
{
  return mDependentGenCoordIndices;
}

//==============================================================================
std::size_t BodyNode::getNumDependentDofs() const
{
  return mDependentDofs.size();
}

//==============================================================================
DegreeOfFreedom* BodyNode::getDependentDof(std::size_t _index)
{
  return common::getVectorObjectIfAvailable<DegreeOfFreedom*>(
      _index, mDependentDofs);
}

//==============================================================================
const DegreeOfFreedom* BodyNode::getDependentDof(std::size_t _index) const
{
  return common::getVectorObjectIfAvailable<DegreeOfFreedom*>(
      _index, mDependentDofs);
}

//==============================================================================
const std::vector<DegreeOfFreedom*>& BodyNode::getDependentDofs()
{
  return mDependentDofs;
}

//==============================================================================
const std::vector<const DegreeOfFreedom*>& BodyNode::getDependentDofs() const
{
  return mConstDependentDofs;
}

//==============================================================================
const std::vector<const DegreeOfFreedom*> BodyNode::getChainDofs() const
{
  // TODO(MXG): Consider templating the Criteria for const BodyNodes so that we
  // don't need a const_cast here. That said, the const_cast isn't hurting
  // anything, because the Criteria function would work just as well operating
  // on const BodyNodes.
  Chain::Criteria criteria(const_cast<BodyNode*>(this), nullptr);
  std::vector<BodyNode*> bn_chain = criteria.satisfy();
  std::vector<const DegreeOfFreedom*> dofs;
  dofs.reserve(getNumDependentGenCoords());
  for (std::vector<BodyNode*>::reverse_iterator rit = bn_chain.rbegin();
       rit != bn_chain.rend();
       ++rit)
  {
    std::size_t nDofs = (*rit)->getParentJoint()->getNumDofs();
    for (std::size_t i = 0; i < nDofs; ++i)
      dofs.push_back((*rit)->getParentJoint()->getDof(i));
  }

  return dofs;
}

//==============================================================================
const Eigen::Isometry3s& BodyNode::getRelativeTransform() const
{
  return mParentJoint->getRelativeTransform();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getRelativeSpatialVelocity() const
{
  return mParentJoint->getRelativeSpatialVelocity();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getRelativeSpatialAcceleration() const
{
  return mParentJoint->getRelativeSpatialAcceleration();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getPrimaryRelativeAcceleration() const
{
  return mParentJoint->getRelativePrimaryAcceleration();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getPartialAcceleration() const
{
  if (mIsPartialAccelerationDirty)
    updatePartialAcceleration();

  return mPartialAcceleration;
}

//==============================================================================
const math::Jacobian& BodyNode::getJacobian() const
{
  if (mIsBodyJacobianDirty)
    updateBodyJacobian();

  return mBodyJacobian;
}

//==============================================================================
const math::Jacobian& BodyNode::getJacobianInPositionSpace() const
{
  if (mIsBodyJacobianInPositionSpaceDirty)
    updateBodyJacobianInPositionSpace();

  return mBodyJacobianInPositionSpace;
}

//==============================================================================
const math::Jacobian& BodyNode::getWorldJacobian() const
{
  if (mIsWorldJacobianDirty)
    updateWorldJacobian();

  return mWorldJacobian;
}

//==============================================================================
const math::Jacobian& BodyNode::getJacobianSpatialDeriv() const
{
  if (mIsBodyJacobianSpatialDerivDirty)
    updateBodyJacobianSpatialDeriv();

  return mBodyJacobianSpatialDeriv;
}

//==============================================================================
const math::Jacobian& BodyNode::getJacobianClassicDeriv() const
{
  if (mIsWorldJacobianClassicDerivDirty)
    updateWorldJacobianClassicDeriv();

  return mWorldJacobianClassicDeriv;
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getBodyVelocityChange() const
{
  return mDelV;
}

//==============================================================================
void BodyNode::setColliding(bool _isColliding)
{
  mIsColliding = _isColliding;
}

//==============================================================================
bool BodyNode::isColliding()
{
  return mIsColliding;
}

//==============================================================================
void BodyNode::addExtForce(
    const Eigen::Vector3s& _force,
    const Eigen::Vector3s& _offset,
    bool _isForceLocal,
    bool _isOffsetLocal)
{
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  Eigen::Vector6s F = Eigen::Vector6s::Zero();
  const Eigen::Isometry3s& W = getWorldTransform();

  if (_isOffsetLocal)
    T.translation() = _offset;
  else
    T.translation() = W.inverse() * _offset;

  if (_isForceLocal)
    F.tail<3>() = _force;
  else
    F.tail<3>() = W.linear().transpose() * _force;

  mAspectState.mFext += math::dAdInvT(T, F);

  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
void BodyNode::setExtForce(
    const Eigen::Vector3s& _force,
    const Eigen::Vector3s& _offset,
    bool _isForceLocal,
    bool _isOffsetLocal)
{
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  Eigen::Vector6s F = Eigen::Vector6s::Zero();
  const Eigen::Isometry3s& W = getWorldTransform();

  if (_isOffsetLocal)
    T.translation() = _offset;
  else
    T.translation() = W.inverse() * _offset;

  if (_isForceLocal)
    F.tail<3>() = _force;
  else
    F.tail<3>() = W.linear().transpose() * _force;

  mAspectState.mFext = math::dAdInvT(T, F);

  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
/// Set the 6vec wrench directly
void BodyNode::setExtWrench(const Eigen::Vector6s& wrench)
{
  mAspectState.mFext = wrench;

  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
void BodyNode::addExtTorque(const Eigen::Vector3s& _torque, bool _isLocal)
{
  if (_isLocal)
    mAspectState.mFext.head<3>() += _torque;
  else
    mAspectState.mFext.head<3>()
        += getWorldTransform().linear().transpose() * _torque;

  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
void BodyNode::setExtTorque(const Eigen::Vector3s& _torque, bool _isLocal)
{
  if (_isLocal)
    mAspectState.mFext.head<3>() = _torque;
  else
    mAspectState.mFext.head<3>()
        = getWorldTransform().linear().transpose() * _torque;

  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
BodyNode::BodyNode(
    BodyNode* _parentBodyNode,
    Joint* _parentJoint,
    const Properties& _properties)
  : Entity(ConstructFrame),
    Frame(Frame::World()),
    TemplatedJacobianNode<BodyNode>(this),
    mID(BodyNode::msBodyNodeCount++),
    mIsColliding(false),
    mParentJoint(_parentJoint),
    mParentBodyNode(nullptr),
    mPartialAcceleration(Eigen::Vector6s::Zero()),
    mIsPartialAccelerationDirty(true),
    mF(Eigen::Vector6s::Zero()),
    mFgravity(Eigen::Vector6s::Zero()),
    mArtInertia(Eigen::Matrix6s::Identity()),
    mArtInertiaImplicit(Eigen::Matrix6s::Identity()),
    mBiasForce(Eigen::Vector6s::Zero()),
    mCg_dV(Eigen::Vector6s::Zero()),
    mCg_F(Eigen::Vector6s::Zero()),
    mG_F(Eigen::Vector6s::Zero()),
    mFext_F(Eigen::Vector6s::Zero()),
    mM_dV(Eigen::Vector6s::Zero()),
    mM_F(Eigen::Vector6s::Zero()),
    mInvM_c(Eigen::Vector6s::Zero()),
    mInvM_U(Eigen::Vector6s::Zero()),
    mArbitrarySpatial(Eigen::Vector6s::Zero()),
    mDelV(Eigen::Vector6s::Zero()),
    mBiasImpulse(Eigen::Vector6s::Zero()),
    mConstraintImpulse(Eigen::Vector6s::Zero()),
    mImpF(Eigen::Vector6s::Zero()),
    onColShapeAdded(mColShapeAddedSignal),
    onColShapeRemoved(mColShapeRemovedSignal),
    onStructuralChange(mStructuralChangeSignal)
{
  // Generate an inert destructor to make sure that it will not try to
  // s_t-delete this BodyNode when it gets destroyed.
  mSelfDestructor
      = std::shared_ptr<NodeDestructor>(new NodeDestructor(nullptr));
  mDestructor = mSelfDestructor;
  mAmAttached = true;

  mParentJoint->mChildBodyNode = this;
  setProperties(_properties);

  if (_parentBodyNode)
    _parentBodyNode->addChildBodyNode(this);

  createAspect<Aspect>();
  createAspect<detail::NodeVectorProxyAspect>();
}

//==============================================================================
BodyNode::BodyNode(const std::tuple<BodyNode*, Joint*, Properties>& args)
  : BodyNode(std::get<0>(args), std::get<1>(args), std::get<2>(args))
{
  // The initializer list is delegating the construction
}

//==============================================================================
BodyNode* BodyNode::clone(
    BodyNode* _parentBodyNode, Joint* _parentJoint, bool cloneNodes) const
{
  BodyNode* clonedBn
      = new BodyNode(_parentBodyNode, _parentJoint, getBodyNodeProperties());

  clonedBn->matchAspects(this);

  if (cloneNodes)
    clonedBn->matchNodes(this);

  return clonedBn;
}

//==============================================================================
Node* BodyNode::cloneNode(BodyNode* /*bn*/) const
{
  dterr << "[BodyNode::cloneNode] This function should never be called! Please "
        << "report this as an error!\n";
  assert(false);
  return nullptr;
}

//==============================================================================
void BodyNode::init(const SkeletonPtr& _skeleton)
{
  mSkeleton = _skeleton;
  assert(_skeleton);
  if (mReferenceCount > 0)
  {
    mReferenceSkeleton = mSkeleton.lock();
  }

  setVersionDependentObject(
      dynamic_cast<common::VersionCounter*>(mSkeleton.lock().get()));
  mParentJoint->setVersionDependentObject(
      dynamic_cast<common::VersionCounter*>(mSkeleton.lock().get()));

  // Put the scope around this so that 'lock' releases the mutex immediately
  // after we're done with it
  {
    std::lock_guard<std::mutex> lock(mLockedSkeleton->mMutex);
    mLockedSkeleton->mSkeleton = mSkeleton;
  }

  //--------------------------------------------------------------------------
  // Fill the list of generalized coordinates this node depends on, and sort
  // it.
  //--------------------------------------------------------------------------
  if (mParentBodyNode)
    mDependentGenCoordIndices = mParentBodyNode->mDependentGenCoordIndices;
  else
    mDependentGenCoordIndices.clear();

  for (std::size_t i = 0; i < mParentJoint->getNumDofs(); i++)
    mDependentGenCoordIndices.push_back(mParentJoint->getIndexInSkeleton(i));

  // Sort
  std::sort(mDependentGenCoordIndices.begin(), mDependentGenCoordIndices.end());

  mDependentDofs.clear();
  mDependentDofs.reserve(mDependentGenCoordIndices.size());
  mConstDependentDofs.clear();
  mConstDependentDofs.reserve(mDependentGenCoordIndices.size());
  for (const std::size_t& index : mDependentGenCoordIndices)
  {
    mDependentDofs.push_back(_skeleton->getDof(index));
    mConstDependentDofs.push_back(_skeleton->getDof(index));
  }

#ifndef NDEBUG
  // Check whether there is duplicated indices.
  std::size_t nDepGenCoordIndices = mDependentGenCoordIndices.size();
  for (std::size_t i = 0; i < nDepGenCoordIndices; ++i)
  {
    for (std::size_t j = i + 1; j < nDepGenCoordIndices; ++j)
    {
      assert(
          mDependentGenCoordIndices[i] != mDependentGenCoordIndices[j]
          && "Duplicated index is found in mDependentGenCoordIndices.");
    }
  }
#endif // NDEBUG

  //--------------------------------------------------------------------------
  // Set dimensions of dynamics matrices and vectors.
  //--------------------------------------------------------------------------
  std::size_t numDepGenCoords = getNumDependentGenCoords();
  mBodyJacobian.setZero(6, numDepGenCoords);
  mBodyJacobianInPositionSpace.setZero(6, numDepGenCoords);
  mWorldJacobian.setZero(6, numDepGenCoords);
  mBodyJacobianSpatialDeriv.setZero(6, numDepGenCoords);
  mWorldJacobianClassicDeriv.setZero(6, numDepGenCoords);
  dirtyTransform();
}

//==============================================================================
void BodyNode::processNewEntity(Entity* _newChildEntity)
{
  // If the Entity is a JacobianNode, add it to the list of JacobianNodes

  // Dev Note (MXG): There are two places where child JacobianNodes get added.
  // This is one place, and the constructor of the JacobianNode class is another
  // place. They get added in two different places because:
  // 1. This location only works for child BodyNodes. When a non-BodyNode gets
  //    constructed, its Entity becomes a child of this BodyNode frame during
  //    the Entity construction, so it cannot be dynamically cast to a
  //    JacobianNode at that time. But this is not an issue for BodyNodes,
  //    because BodyNodes become children of this Frame after construction is
  //    finished.
  // 2. The JacobianNode constructor only works for non-BodyNodes. When a
  //    JacobianNode is being used as a base for a BodyNode, it does not know
  //    the parent BodyNode.
  //
  // We should consider doing something to unify these two pipelines that are
  // currently independent of each other.
  if (JacobianNode* node = dynamic_cast<JacobianNode*>(_newChildEntity))
    mChildJacobianNodes.insert(node);

  // Here we want to sort out whether the Entity that has been added is a child
  // BodyNode or not

  // Check if it's a child BodyNode (if not, then it's just some other arbitrary
  // type of Entity)
  if (std::find(mChildBodyNodes.begin(), mChildBodyNodes.end(), _newChildEntity)
      != mChildBodyNodes.end())
    return;

  // Check if it's already accounted for in our Non-BodyNode Entities
  if (mNonBodyNodeEntities.find(_newChildEntity) != mNonBodyNodeEntities.end())
  {
    dtwarn << "[BodyNode::processNewEntity] Attempting to add an Entity ["
           << _newChildEntity->getName() << "] as a child Entity of ["
           << getName() << "], which is already its parent." << std::endl;
    return;
  }

  // Add it to the Non-BodyNode Entities
  mNonBodyNodeEntities.insert(_newChildEntity);
}

//==============================================================================
void BodyNode::processRemovedEntity(Entity* _oldChildEntity)
{
  std::vector<BodyNode*>::iterator it = std::find(
      mChildBodyNodes.begin(), mChildBodyNodes.end(), _oldChildEntity);
  if (it != mChildBodyNodes.end())
    mChildBodyNodes.erase(it);

  if (JacobianNode* node = dynamic_cast<JacobianNode*>(_oldChildEntity))
    mChildJacobianNodes.erase(node);

  if (std::find(
          mNonBodyNodeEntities.begin(),
          mNonBodyNodeEntities.end(),
          _oldChildEntity)
      != mNonBodyNodeEntities.end())
    mNonBodyNodeEntities.erase(_oldChildEntity);
}

//==============================================================================
void BodyNode::dirtyTransform()
{
  dirtyVelocity(); // Global Velocity depends on the Global Transform

  if (mNeedTransformUpdate)
    return;

  mNeedTransformUpdate = true;

  const SkeletonPtr& skel = getSkeleton();
  if (skel)
  {
    // All of these depend on the world transform of this BodyNode, so they must
    // be dirtied whenever mNeedTransformUpdate is dirtied, and if
    // mTransformUpdate is already dirty, then these must already be dirty as
    // well
    SET_FLAGS(mCoriolisForces);
    SET_FLAGS(mGravityForces);
    SET_FLAGS(mCoriolisAndGravityForces);
    SET_FLAGS(mExternalForces);
  }

  // Child BodyNodes and other generic Entities are notified separately to allow
  // some optimizations
  for (std::size_t i = 0; i < mChildBodyNodes.size(); ++i)
    mChildBodyNodes[i]->dirtyTransform();

  for (Entity* entity : mNonBodyNodeEntities)
    entity->dirtyTransform();
}

//==============================================================================
void BodyNode::dirtyVelocity()
{
  dirtyAcceleration(); // Global Acceleration depends on Global Velocity

  if (mNeedVelocityUpdate)
    return;

  mNeedVelocityUpdate = true;
  mIsPartialAccelerationDirty = true;

  const SkeletonPtr& skel = getSkeleton();
  if (skel)
  {
    SET_FLAGS(mCoriolisForces);
    SET_FLAGS(mCoriolisAndGravityForces);
  }

  // Child BodyNodes and other generic Entities are notified separately to allow
  // some optimizations
  for (std::size_t i = 0; i < mChildBodyNodes.size(); ++i)
    mChildBodyNodes[i]->dirtyVelocity();

  for (Entity* entity : mNonBodyNodeEntities)
    entity->dirtyVelocity();
}

//==============================================================================
void BodyNode::dirtyAcceleration()
{
  // If we already know we need to update, just quit
  if (mNeedAccelerationUpdate)
    return;

  mNeedAccelerationUpdate = true;

  for (std::size_t i = 0; i < mChildBodyNodes.size(); ++i)
    mChildBodyNodes[i]->dirtyAcceleration();

  for (Entity* entity : mNonBodyNodeEntities)
    entity->dirtyAcceleration();
}

//==============================================================================
void BodyNode::notifyArticulatedInertiaUpdate()
{
  dirtyArticulatedInertia();
}

//==============================================================================
void BodyNode::dirtyArticulatedInertia()
{
  const SkeletonPtr& skel = getSkeleton();
  if (skel)
    skel->dirtyArticulatedInertia(mTreeIndex);
}

//==============================================================================
void BodyNode::notifyExternalForcesUpdate()
{
  dirtyExternalForces();
}

//==============================================================================
void BodyNode::dirtyExternalForces()
{
  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
void BodyNode::notifyCoriolisUpdate()
{
  dirtyCoriolisForces();
}

//==============================================================================
void BodyNode::dirtyCoriolisForces()
{
  SKEL_SET_FLAGS(mCoriolisForces);
  SKEL_SET_FLAGS(mCoriolisAndGravityForces);
}

//==============================================================================
void BodyNode::updateTransform()
{
  // Calling getWorldTransform will update the transform if an update is needed
  getWorldTransform();
  assert(math::verifyTransform(mWorldTransform));
}

//==============================================================================
void BodyNode::updateVelocity()
{
  // Calling getSpatialVelocity will update the velocity if an update is needed
  getSpatialVelocity();
  assert(!math::isNan(mVelocity));
}

//==============================================================================
void BodyNode::updatePartialAcceleration() const
{
  // Compute partial acceleration
  mParentJoint->setPartialAccelerationTo(
      mPartialAcceleration, getSpatialVelocity());
  mIsPartialAccelerationDirty = false;
}

//==============================================================================
void BodyNode::updateAccelerationID()
{
  // Note: auto-updating has replaced this function
  getSpatialAcceleration();
  // Verification
  assert(!math::isNan(mAcceleration));
}

//==============================================================================
void BodyNode::updateTransmittedForceID(
    const Eigen::Vector3s& _gravity, bool _withExternalForces)
{
  // Gravity force
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  if (mAspectProperties.mGravityMode == true)
    mFgravity.noalias()
        = mI * math::AdInvRLinear(getWorldTransform(), _gravity);
  else
    mFgravity.setZero();

  // Inertial force
  mF.noalias() = mI * getSpatialAcceleration();

  // External force
  if (_withExternalForces)
    mF -= mAspectState.mFext;

  // Verification
  assert(!math::isNan(mF));

  // Gravity force
  mF -= mFgravity;

  // Coriolis force
  const Eigen::Vector6s& V = getSpatialVelocity();
  mF -= math::dad(V, mI * V);

  //
  for (const auto& childBodyNode : mChildBodyNodes)
  {
    Joint* childJoint = childBodyNode->getParentJoint();
    assert(childJoint != nullptr);

    mF += math::dAdInvT(
        childJoint->getRelativeTransform(), childBodyNode->getBodyForce());
  }

  // Verification
  assert(!math::isNan(mF));
}

//==============================================================================
void BodyNode::updateArtInertia(s_t _timeStep) const
{
  // Set spatial inertia to the articulated body inertia
  mArtInertia = mAspectProperties.mInertia.getSpatialTensor();
  mArtInertiaImplicit = mArtInertia;

  // and add child articulated body inertia
  for (const auto& child : mChildBodyNodes)
  {
    Joint* childJoint = child->getParentJoint();

    childJoint->addChildArtInertiaTo(mArtInertia, child->mArtInertia);
    childJoint->addChildArtInertiaImplicitTo(
        mArtInertiaImplicit, child->mArtInertiaImplicit);
  }

  // Verification
  //  assert(!math::isNan(mArtInertia));
  assert(!math::isNan(mArtInertiaImplicit));

  // Update parent joint's inverse of projected articulated body inertia
  mParentJoint->updateInvProjArtInertia(mArtInertia);
  mParentJoint->updateInvProjArtInertiaImplicit(mArtInertiaImplicit, _timeStep);

  // Verification
  //  assert(!math::isNan(mArtInertia));
  assert(!math::isNan(mArtInertiaImplicit));
}

//==============================================================================
void BodyNode::updateBiasForce(const Eigen::Vector3s& _gravity, s_t _timeStep)
{
  // Gravity force
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  if (mAspectProperties.mGravityMode == true)
    mFgravity.noalias()
        = mI * math::AdInvRLinear(getWorldTransform(), _gravity);
  else
    mFgravity.setZero();

  // Set bias force
  const Eigen::Vector6s& V = getSpatialVelocity();
  mBiasForce = -math::dad(V, mI * V) - mAspectState.mFext - mFgravity;

  // Verification
  assert(!math::isNan(mBiasForce));

  // And add child bias force
  for (const auto& childBodyNode : mChildBodyNodes)
  {
    Joint* childJoint = childBodyNode->getParentJoint();

    childJoint->addChildBiasForceTo(
        mBiasForce,
        childBodyNode->getArticulatedInertia(),
        childBodyNode->mBiasForce,
        childBodyNode->getPartialAcceleration());
  }

  // Verification
  assert(!math::isNan(mBiasForce));

  // Update parent joint's total force with implicit joint damping and spring
  // forces
  mParentJoint->updateTotalForce(
      getArticulatedInertia() * getPartialAcceleration() + mBiasForce,
      _timeStep);
}

//==============================================================================
void BodyNode::updateBiasImpulse()
{
  // Update impulsive bias force
  mBiasImpulse = -mConstraintImpulse;

  // And add child bias impulse
  for (auto& childBodyNode : mChildBodyNodes)
  {
    Joint* childJoint = childBodyNode->getParentJoint();

    childJoint->addChildBiasImpulseTo(
        mBiasImpulse,
        childBodyNode->getArticulatedInertia(),
        childBodyNode->mBiasImpulse);
  }

  // Verification
  assert(!math::isNan(mBiasImpulse));

  // Update parent joint's total force
  mParentJoint->updateTotalImpulse(mBiasImpulse);
}

//==============================================================================
void BodyNode::updateTransmittedForceFD()
{
  mF = mBiasForce;
  mF.noalias() += getArticulatedInertia() * getSpatialAcceleration();

  assert(!math::isNan(mF));
}

//==============================================================================
void BodyNode::updateTransmittedImpulse()
{
  mImpF = mBiasImpulse;
  mImpF.noalias() += getArticulatedInertia() * mDelV;

  assert(!math::isNan(mImpF));
}

//==============================================================================
void BodyNode::updateAccelerationFD()
{
  if (mParentBodyNode)
  {
    // Update joint acceleration
    /*
    mParentJoint->updateAcceleration(
        getArticulatedInertiaImplicit(),
        mParentBodyNode->getSpatialAcceleration());
    */
    mParentJoint->updateAcceleration(
        getArticulatedInertia(),
        mParentBodyNode->getSpatialAcceleration());
  }
  else
  {
    // Update joint acceleration
    /*
    mParentJoint->updateAcceleration(
        getArticulatedInertiaImplicit(), Eigen::Vector6s::Zero());
    */
    mParentJoint->updateAcceleration(
        getArticulatedInertia(), Eigen::Vector6s::Zero());
  }

  // Verify the spatial acceleration of this body
  assert(!math::isNan(mAcceleration));
}

//==============================================================================
void BodyNode::updateVelocityChangeFD()
{
  if (mParentBodyNode)
  {
    // Update joint velocity change
    mParentJoint->updateVelocityChange(
        getArticulatedInertia(), mParentBodyNode->mDelV);

    // Transmit spatial acceleration of parent body to this body
    mDelV = math::AdInvT(
        mParentJoint->getRelativeTransform(), mParentBodyNode->mDelV);
  }
  else
  {
    // Update joint velocity change
    mParentJoint->updateVelocityChange(
        getArticulatedInertia(), Eigen::Vector6s::Zero());

    // Transmit spatial acceleration of parent body to this body
    mDelV.setZero();
  }

  // Add parent joint's acceleration to this body
  mParentJoint->addVelocityChangeTo(mDelV);

  // Verify the spatial velocity change of this body
  assert(!math::isNan(mDelV));
}

//==============================================================================
void BodyNode::updateJointForceID(
    s_t _timeStep, bool _withDampingForces, bool _withSpringForces)
{
  assert(mParentJoint != nullptr);
  mParentJoint->updateForceID(
      mF, _timeStep, _withDampingForces, _withSpringForces);
}

//==============================================================================
void BodyNode::updateJointForceFD(
    s_t _timeStep, bool _withDampingForces, bool _withSpringForces)
{
  assert(mParentJoint != nullptr);
  mParentJoint->updateForceFD(
      mF, _timeStep, _withDampingForces, _withSpringForces);
}

//==============================================================================
void BodyNode::updateJointImpulseFD()
{
  assert(mParentJoint != nullptr);
  mParentJoint->updateImpulseFD(mF);
}

//==============================================================================
void BodyNode::updateConstrainedTerms(s_t _timeStep)
{
  // 1. dq = dq + del_dq
  // 2. ddq = ddq + del_dq / dt
  // 3. tau = tau + imp / dt
  mParentJoint->updateConstrainedTerms(_timeStep);

  //
  mF += mImpF / _timeStep;
}

//==============================================================================
void BodyNode::clearExternalForces()
{
  mAspectState.mFext.setZero();
  SKEL_SET_FLAGS(mExternalForces);
}

//==============================================================================
void BodyNode::clearInternalForces()
{
  mParentJoint->resetControlForces();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getExternalForceLocal() const
{
  return mAspectState.mFext;
}

//==============================================================================
Eigen::Vector6s BodyNode::getExternalForceGlobal() const
{
  return math::dAdInvT(getWorldTransform(), mAspectState.mFext);
}

//==============================================================================
void BodyNode::addConstraintImpulse(
    const Eigen::Vector3s& _constImp,
    const Eigen::Vector3s& _offset,
    bool _isImpulseLocal,
    bool _isOffsetLocal)
{
  // TODO(JS): Add contact sensor data here (DART 4.1)

  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  Eigen::Vector6s F = Eigen::Vector6s::Zero();
  const Eigen::Isometry3s& W = getWorldTransform();

  if (_isOffsetLocal)
    T.translation() = _offset;
  else
    T.translation() = W.inverse() * _offset;

  if (_isImpulseLocal)
    F.tail<3>() = _constImp;
  else
    F.tail<3>() = W.linear().transpose() * _constImp;

  mConstraintImpulse += math::dAdInvT(T, F);
}

//==============================================================================
void BodyNode::clearConstraintImpulse()
{
  mDelV.setZero();
  mBiasImpulse.setZero();
  mConstraintImpulse.setZero();
  mImpF.setZero();

  mParentJoint->resetConstraintImpulses();
  mParentJoint->resetTotalImpulses();
  mParentJoint->resetVelocityChanges();
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getBodyForce() const
{
  return mF;
}

//==============================================================================
void BodyNode::setConstraintImpulse(const Eigen::Vector6s& _constImp)
{
  assert(!math::isNan(_constImp));
  mConstraintImpulse = _constImp;
}

//==============================================================================
void BodyNode::addConstraintImpulse(const Eigen::Vector6s& _constImp)
{
  assert(!math::isNan(_constImp));
  mConstraintImpulse += _constImp;
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getConstraintImpulse() const
{
  return mConstraintImpulse;
}

//==============================================================================
s_t BodyNode::computeLagrangian(const Eigen::Vector3s& gravity) const
{
  return computeKineticEnergy() - computePotentialEnergy(gravity);
}

//==============================================================================
s_t BodyNode::getKineticEnergy() const
{
  return computeKineticEnergy();
}

//==============================================================================
s_t BodyNode::computeKineticEnergy() const
{
  const Eigen::Vector6s& V = getSpatialVelocity();
  const Eigen::Matrix6s& G = mAspectProperties.mInertia.getSpatialTensor();

  return 0.5 * V.dot(G * V);
}

//==============================================================================
s_t BodyNode::getPotentialEnergy(const Eigen::Vector3s& _gravity) const
{
  return computePotentialEnergy(_gravity);
}

//==============================================================================
s_t BodyNode::computePotentialEnergy(const Eigen::Vector3s& gravity) const
{
  return -getMass() * getWorldTransform().translation().dot(gravity);
}

//==============================================================================
Eigen::Vector3s BodyNode::getLinearMomentum() const
{
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  return (mI * getSpatialVelocity()).tail<3>();
}

//==============================================================================
Eigen::Vector3s BodyNode::getAngularMomentum(const Eigen::Vector3s& _pivot)
{
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  T.translation() = _pivot;
  return math::dAdT(T, mI * getSpatialVelocity()).head<3>();
}

//==============================================================================
bool BodyNode::isReactive() const
{
  const ConstSkeletonPtr& skel = getSkeleton();
  if (skel && skel->isMobile() && getNumDependentGenCoords() > 0)
  {
    // Check if all the ancestor joints are motion prescribed.
    const BodyNode* body = this;
    while (body != nullptr)
    {
      if (body->mParentJoint->isDynamic())
        return true;

      body = body->mParentBodyNode;
    }
    // TODO: Checking if all the ancestor joints are motion prescribed is
    // expensive. It would be good to evaluate this in advance and update only
    // when necessary.

    return false;
  }
  else
  {
    return false;
  }
}

//==============================================================================
void BodyNode::aggregateCoriolisForceVector(Eigen::VectorXs& _C)
{
  aggregateCombinedVector(_C, Eigen::Vector3s::Zero());
}

//==============================================================================
void BodyNode::aggregateGravityForceVector(
    Eigen::VectorXs& _g, const Eigen::Vector3s& _gravity)
{
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  if (mAspectProperties.mGravityMode == true)
    mG_F = mI * math::AdInvRLinear(getWorldTransform(), _gravity);
  else
    mG_F.setZero();

  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mG_F += math::dAdInvT(
        (*it)->mParentJoint->getRelativeTransform(), (*it)->mG_F);
  }

  std::size_t nGenCoords = mParentJoint->getNumDofs();
  if (nGenCoords > 0)
  {
    Eigen::VectorXs g
        = -(mParentJoint->getRelativeJacobian().transpose() * mG_F);
    std::size_t iStart = mParentJoint->getIndexInTree(0);
    _g.segment(iStart, nGenCoords) = g;
  }
}

//==============================================================================
void BodyNode::updateCombinedVector()
{
  if (mParentBodyNode)
  {
    mCg_dV = math::AdInvT(
                 mParentJoint->getRelativeTransform(), mParentBodyNode->mCg_dV)
             + getPartialAcceleration();
  }
  else
  {
    mCg_dV = getPartialAcceleration();
  }
}

//==============================================================================
void BodyNode::aggregateCombinedVector(
    Eigen::VectorXs& _Cg, const Eigen::Vector3s& _gravity)
{
  // H(i) = I(i) * W(i) -
  //        dad{V}(I(i) * V(i)) + sum(k \in children) dAd_{T(i,j)^{-1}}(H(k))
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  if (mAspectProperties.mGravityMode == true)
    mFgravity = mI * math::AdInvRLinear(getWorldTransform(), _gravity);
  else
    mFgravity.setZero();

  const Eigen::Vector6s& V = getSpatialVelocity();
  mCg_F = mI * mCg_dV;
  mCg_F -= mFgravity;
  mCg_F -= math::dad(V, mI * V);

  for (std::vector<BodyNode*>::iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mCg_F += math::dAdInvT((*it)->getParentJoint()->mT, (*it)->mCg_F);
  }

  std::size_t nGenCoords = mParentJoint->getNumDofs();
  if (nGenCoords > 0)
  {
    Eigen::VectorXs Cg
        = mParentJoint->getRelativeJacobian().transpose() * mCg_F;
    std::size_t iStart = mParentJoint->getIndexInTree(0);
    _Cg.segment(iStart, nGenCoords) = Cg;
  }
}

//==============================================================================
void BodyNode::aggregateExternalForces(Eigen::VectorXs& _Fext)
{
  mFext_F = mAspectState.mFext;

  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mFext_F += math::dAdInvT(
        (*it)->mParentJoint->getRelativeTransform(), (*it)->mFext_F);
  }

  std::size_t nGenCoords = mParentJoint->getNumDofs();
  if (nGenCoords > 0)
  {
    Eigen::VectorXs Fext
        = mParentJoint->getRelativeJacobian().transpose() * mFext_F;
    std::size_t iStart = mParentJoint->getIndexInTree(0);
    _Fext.segment(iStart, nGenCoords) = Fext;
  }
}

//==============================================================================
void BodyNode::aggregateSpatialToGeneralized(
    Eigen::VectorXs& _generalized, const Eigen::Vector6s& _spatial)
{
  //
  mArbitrarySpatial = _spatial;

  //
  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mArbitrarySpatial += math::dAdInvT(
        (*it)->mParentJoint->getRelativeTransform(), (*it)->mArbitrarySpatial);
  }

  // Project the spatial quantity to generalized coordinates
  const auto numDofs = mParentJoint->getNumDofs();
  if (numDofs > 0u)
  {
    const std::size_t iStart = mParentJoint->getIndexInTree(0);
    _generalized.segment(iStart, numDofs)
        = mParentJoint->getSpatialToGeneralized(mArbitrarySpatial);
  }
}

//==============================================================================
void BodyNode::updateMassMatrix()
{
  mM_dV.setZero();
  std::size_t dof = mParentJoint->getNumDofs();
  if (dof > 0)
  {
    mM_dV.noalias() += mParentJoint->getRelativeJacobian()
                       * mParentJoint->getAccelerations();
    assert(!math::isNan(mM_dV));
  }
  if (mParentBodyNode)
    mM_dV += math::AdInvT(
        mParentJoint->getRelativeTransform(), mParentBodyNode->mM_dV);
  assert(!math::isNan(mM_dV));
}

//==============================================================================
void BodyNode::aggregateMassMatrix(Eigen::MatrixXs& _MCol, std::size_t _col)
{
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
  //
  mM_F.noalias() = mI * mM_dV;

  // Verification
  assert(!math::isNan(mM_F));

  //
  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mM_F += math::dAdInvT(
        (*it)->getParentJoint()->getRelativeTransform(), (*it)->mM_F);
  }

  // Verification
  assert(!math::isNan(mM_F));

  //
  std::size_t dof = mParentJoint->getNumDofs();
  if (dof > 0)
  {
    std::size_t iStart = mParentJoint->getIndexInTree(0);
    _MCol.block(iStart, _col, dof, 1).noalias()
        = mParentJoint->getRelativeJacobian().transpose() * mM_F;
  }
}

//==============================================================================
void BodyNode::aggregateAugMassMatrix(
    Eigen::MatrixXs& _MCol, std::size_t _col, s_t _timeStep)
{
  // TODO(JS): Need to be reimplemented
  const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();

  //
  mM_F.noalias() = mI * mM_dV;

  // Verification
  assert(!math::isNan(mM_F));

  //
  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    mM_F += math::dAdInvT(
        (*it)->getParentJoint()->getRelativeTransform(), (*it)->mM_F);
  }

  // Verification
  assert(!math::isNan(mM_F));

  //
  std::size_t dof = mParentJoint->getNumDofs();
  if (dof > 0)
  {
    Eigen::MatrixXs K = Eigen::MatrixXs::Zero(dof, dof);
    Eigen::MatrixXs D = Eigen::MatrixXs::Zero(dof, dof);
    for (std::size_t i = 0; i < dof; ++i)
    {
      K(i, i) = mParentJoint->getSpringStiffness(i);
      D(i, i) = mParentJoint->getDampingCoefficient(i);
    }

    std::size_t iStart = mParentJoint->getIndexInTree(0);

    _MCol.block(iStart, _col, dof, 1).noalias()
        = mParentJoint->getRelativeJacobian().transpose() * mM_F
          + D * (_timeStep * mParentJoint->getAccelerations())
          + K * (_timeStep * _timeStep * mParentJoint->getAccelerations());
  }
}

//==============================================================================
void BodyNode::updateInvMassMatrix()
{
  //
  mInvM_c.setZero();

  //
  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    (*it)->getParentJoint()->addChildBiasForceForInvMassMatrix(
        mInvM_c, (*it)->getArticulatedInertia(), (*it)->mInvM_c);
  }

  // Verification
  assert(!math::isNan(mInvM_c));

  // Update parent joint's total force for inverse mass matrix
  mParentJoint->updateTotalForceForInvMassMatrix(mInvM_c);
}

//==============================================================================
void BodyNode::updateInvAugMassMatrix()
{
  //
  mInvM_c.setZero();

  //
  for (std::vector<BodyNode*>::const_iterator it = mChildBodyNodes.begin();
       it != mChildBodyNodes.end();
       ++it)
  {
    (*it)->getParentJoint()->addChildBiasForceForInvAugMassMatrix(
        mInvM_c, (*it)->getArticulatedInertiaImplicit(), (*it)->mInvM_c);
  }

  // Verification
  assert(!math::isNan(mInvM_c));

  // Update parent joint's total force for inverse mass matrix
  mParentJoint->updateTotalForceForInvMassMatrix(mInvM_c);
}

//==============================================================================
void BodyNode::aggregateInvMassMatrix(
    Eigen::MatrixXs& _InvMCol, std::size_t _col)
{
  if (mParentBodyNode)
  {
    //
    mParentJoint->getInvMassMatrixSegment(
        _InvMCol, _col, getArticulatedInertia(), mParentBodyNode->mInvM_U);

    //
    mInvM_U = math::AdInvT(
        mParentJoint->getRelativeTransform(), mParentBodyNode->mInvM_U);
  }
  else
  {
    //
    mParentJoint->getInvMassMatrixSegment(
        _InvMCol, _col, getArticulatedInertia(), Eigen::Vector6s::Zero());

    //
    mInvM_U.setZero();
  }

  //
  mParentJoint->addInvMassMatrixSegmentTo(mInvM_U);
}

//==============================================================================
void BodyNode::aggregateInvAugMassMatrix(
    Eigen::MatrixXs& _InvMCol, std::size_t _col, s_t /*_timeStep*/)
{
  if (mParentBodyNode)
  {
    //
    mParentJoint->getInvAugMassMatrixSegment(
        _InvMCol,
        _col,
        getArticulatedInertiaImplicit(),
        mParentBodyNode->mInvM_U);

    //
    mInvM_U = math::AdInvT(
        mParentJoint->getRelativeTransform(), mParentBodyNode->mInvM_U);
  }
  else
  {
    //
    mParentJoint->getInvAugMassMatrixSegment(
        _InvMCol,
        _col,
        getArticulatedInertiaImplicit(),
        Eigen::Vector6s::Zero());

    //
    mInvM_U.setZero();
  }

  //
  mParentJoint->addInvMassMatrixSegmentTo(mInvM_U);
}

//==============================================================================
void BodyNode::updateBodyJacobian() const
{
  //--------------------------------------------------------------------------
  // Jacobian update
  //
  // J = | J1 J2 ... Jn |
  //   = | Ad(T(i,i-1), J_parent) J_local |
  //
  //   J_parent: (6 x parentDOF)
  //    J_local: (6 x localDOF)
  //         Ji: (6 x 1) se3
  //          n: number of dependent coordinates
  //--------------------------------------------------------------------------

  if (nullptr == mParentJoint)
    return;

  const std::size_t localDof = mParentJoint->getNumDofs();
  assert(getNumDependentGenCoords() >= localDof);
  const std::size_t ascendantDof = getNumDependentGenCoords() - localDof;

  // Parent Jacobian
  if (mParentBodyNode)
  {
    assert(
        static_cast<std::size_t>(mParentBodyNode->getJacobian().cols())
            + mParentJoint->getNumDofs()
        == static_cast<std::size_t>(mBodyJacobian.cols()));

    assert(mParentJoint);
    mBodyJacobian.leftCols(ascendantDof) = math::AdInvTJac(
        mParentJoint->getRelativeTransform(), mParentBodyNode->getJacobian());
  }

  // Local Jacobian
  mBodyJacobian.rightCols(localDof) = mParentJoint->getRelativeJacobian();

  mIsBodyJacobianDirty = false;
}

//==============================================================================
/// Update body Jacobian. getJacobianInPositionSpace() calls this function if
/// mIsBodyJacobianInPositionSpaceDirty is true.
void BodyNode::updateBodyJacobianInPositionSpace() const
{
  //--------------------------------------------------------------------------
  // Jacobian update
  //
  // J = | J1 J2 ... Jn |
  //   = | Ad(T(i,i-1), J_parent) J_local |
  //
  //   J_parent: (6 x parentDOF)
  //    J_local: (6 x localDOF)
  //         Ji: (6 x 1) se3
  //          n: number of dependent coordinates
  //--------------------------------------------------------------------------

  if (nullptr == mParentJoint)
    return;

  const std::size_t localDof = mParentJoint->getNumDofs();
  assert(getNumDependentGenCoords() >= localDof);
  const std::size_t ascendantDof = getNumDependentGenCoords() - localDof;

  // Parent Jacobian
  if (mParentBodyNode)
  {
    assert(
        static_cast<std::size_t>(
            mParentBodyNode->getJacobianInPositionSpace().cols())
            + mParentJoint->getNumDofs()
        == static_cast<std::size_t>(mBodyJacobianInPositionSpace.cols()));

    assert(mParentJoint);
    mBodyJacobianInPositionSpace.leftCols(ascendantDof) = math::AdInvTJac(
        mParentJoint->getRelativeTransform(),
        mParentBodyNode->getJacobianInPositionSpace());
  }

  // Local Jacobian
  mBodyJacobianInPositionSpace.rightCols(localDof)
      = mParentJoint->getRelativeJacobianInPositionSpace();

  mIsBodyJacobianInPositionSpaceDirty = false;
}

//==============================================================================
void BodyNode::updateWorldJacobian() const
{
  mWorldJacobian = math::AdRJac(getWorldTransform(), getJacobian());

  mIsWorldJacobianDirty = false;
}

//==============================================================================
void BodyNode::updateBodyJacobianSpatialDeriv() const
{
  //--------------------------------------------------------------------------
  // Body Jacobian first spatial derivative update
  //
  // dJ = | Ad(T(i, parent(i)), dJ_parent(i))    ad(V(i), S(i)) + dS(i) |
  //
  // T(i, parent(i)): Transformation from this BodyNode to the parent BodyNode
  // dJ             : Spatial Jacobian derivative (6 x dependentDOF)
  // dJ_parent      : Parent Jacobian derivative (6 x (dependentDOF - localDOF))
  // V(i)           : Spatial velocity (6 x 1)
  // S(i)           : Local spatial Jacobian (6 x localDOF)
  // dS(i)          : Local spatial Jacobian deriavative (6 x localDOF)
  // Ad(T(1,2), V)  : Transformation a spatial motion from frame 2 to frame 1
  // ad(V, W)       : Spatial cross product for spatial motions
  //--------------------------------------------------------------------------

  if (nullptr == mParentJoint)
    return;

  const auto numLocalDOFs = mParentJoint->getNumDofs();
  assert(getNumDependentGenCoords() >= numLocalDOFs);
  const auto numParentDOFs = getNumDependentGenCoords() - numLocalDOFs;

  // Parent Jacobian: Ad(T(i, parent(i)), dJ_parent(i))
  if (mParentBodyNode)
  {
    const auto& dJ_parent = mParentBodyNode->getJacobianSpatialDeriv();

    assert(
        static_cast<std::size_t>(dJ_parent.cols()) + mParentJoint->getNumDofs()
        == static_cast<std::size_t>(mBodyJacobianSpatialDeriv.cols()));

    mBodyJacobianSpatialDeriv.leftCols(numParentDOFs)
        = math::AdInvTJac(mParentJoint->getRelativeTransform(), dJ_parent);
  }

  // Local Jacobian: ad(V(i), S(i)) + dS(i)
  mBodyJacobianSpatialDeriv.rightCols(numLocalDOFs)
      = math::adJac(getSpatialVelocity(), mParentJoint->getRelativeJacobian())
        + mParentJoint->getRelativeJacobianTimeDeriv();

  mIsBodyJacobianSpatialDerivDirty = false;
}

//==============================================================================
void BodyNode::updateWorldJacobianClassicDeriv() const
{
  //----------------------------------------------------------------------------
  // World Jacobian first classic deriv update
  //
  // dJr = |                   dJr_parent dJr_local - Jr_local x w |
  //
  // dJl = | dJl_parent + Jr_parent x (v_local + w_parent x p) + dJr_parent x p
  // dJl_local - Jl_local x w |
  //
  // dJr: Rotational portion of Jacobian derivative
  // dJl: Linear portion of Jacobian derivative
  // dJr_parent: Parent rotational Jacobian derivative
  // dJl_parent: Parent linear Jacobian derivative
  // dJr_local: Local rotational Jacobian derivative (in World coordinates)
  // dJl_local: Local linear Jacobian derivative (in World coordinates)
  // v_local: Linear velocity relative to parent Frame
  // w_parent: Total angular velocity of the parent Frame
  // w: Total angular velocity of this Frame
  // p: Offset from origin of parent Frame

  if (nullptr == mParentJoint)
    return;

  const std::size_t numLocalDOFs = mParentJoint->getNumDofs();
  assert(getNumDependentGenCoords() >= numLocalDOFs);
  const std::size_t numParentDOFs = getNumDependentGenCoords() - numLocalDOFs;

  if (mParentBodyNode)
  {
    const math::Jacobian& dJ_parent
        = mParentBodyNode->getJacobianClassicDeriv();
    const math::Jacobian& J_parent = mParentBodyNode->getWorldJacobian();

    const Eigen::Vector3s& v_local
        = getLinearVelocity(mParentBodyNode, Frame::World());
    const Eigen::Vector3s& w_parent = mParentFrame->getAngularVelocity();
    const Eigen::Vector3s& p
        = (getWorldTransform().translation()
           - mParentBodyNode->getWorldTransform().translation())
              .eval();

    assert(
        static_cast<std::size_t>(dJ_parent.cols()) + mParentJoint->getNumDofs()
        == static_cast<std::size_t>(mWorldJacobianClassicDeriv.cols()));

    // dJr
    mWorldJacobianClassicDeriv.block(0, 0, 3, numParentDOFs)
        = dJ_parent.topRows<3>();
    mWorldJacobianClassicDeriv.block(3, 0, 3, numParentDOFs)
        = dJ_parent.bottomRows<3>()
          + J_parent.topRows<3>().colwise().cross(v_local + w_parent.cross(p))
          + dJ_parent.topRows<3>().colwise().cross(p);
  }

  const math::Jacobian& dJ_local = mParentJoint->getRelativeJacobianTimeDeriv();
  const math::Jacobian& J_local = mParentJoint->getRelativeJacobian();
  const Eigen::Isometry3s& T = getWorldTransform();
  const Eigen::Vector3s& w = getAngularVelocity();

  mWorldJacobianClassicDeriv.block(0, numParentDOFs, 3, numLocalDOFs)
      = T.linear() * dJ_local.topRows<3>()
        - (T.linear() * J_local.topRows<3>()).colwise().cross(w);

  mWorldJacobianClassicDeriv.block(3, numParentDOFs, 3, numLocalDOFs)
      = T.linear() * dJ_local.bottomRows<3>()
        - (T.linear() * J_local.bottomRows<3>()).colwise().cross(w);

  mIsWorldJacobianClassicDerivDirty = false;
}

//==============================================================================
void BodyNode::computeJacobianOfMForward(neural::WithRespectTo* wrt)
{
  using math::ad;
  using math::AdInvT;
  using math::Jacobian;

  (void)wrt;
#ifndef NDEBUG
  if (wrt != neural::WithRespectTo::POSITION)
  {
    std::cout << "wrt: " << wrt << std::endl;
    std::cout << "when:\n"
              << "- neural::WithRespectTo::POSITION: "
              << neural::WithRespectTo::POSITION << "\n"
              << "- neural::WithRespectTo::VELOCITY: "
              << neural::WithRespectTo::VELOCITY << "\n"
              << "- neural::WithRespectTo::FORCE   : "
              << neural::WithRespectTo::FORCE << "\n"
              << std::endl;
    assert(wrt == neural::WithRespectTo::POSITION);
  }
#endif

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();

  mMddq_dV_p.resize(6, static_cast<int>(numDofs));

  const Eigen::Isometry3s& T = mParentJoint->getRelativeTransform();
  const Jacobian& S = mParentJoint->getRelativeJacobian();
  const Jacobian& H = mParentJoint->getRelativeJacobianInPositionSpace();
  const Eigen::VectorXs& ddq = mParentJoint->getAccelerations();

  // Update mMddq_dV
  if (mParentBodyNode)
    mMddq_dV = math::AdInvT(T, mParentBodyNode->mMddq_dV) + S * ddq;
  else
    mMddq_dV.noalias() = S * ddq;

  // TODO(JS): iterate joints instead for vectorization
  for (auto i = 0u; i < numDofs; ++i)
  {
    const DegreeOfFreedom* dof = skel->getDof(i);
    if (mParentJoint->hasDof(dof))
    {
      const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());
      const Eigen::Vector6s Hcol = H.col(dofIndexInJoint);

      const Jacobian DS_Dq
          = mParentJoint->getRelativeJacobianDeriv(dof->getIndexInJoint());
      const Jacobian DdS_Dq
          = mParentJoint->getRelativeJacobianTimeDerivDerivWrtPosition(
              dof->getIndexInJoint());

      if (mParentBodyNode)
      {
        const Eigen::Vector6s& parent_dV = mParentBodyNode->mMddq_dV;
        mMddq_dV_p.col(i) = AdInvT(T, mParentBodyNode->mMddq_dV_p.col(i))
                            - ad(Hcol, AdInvT(T, parent_dV)) + DS_Dq * ddq;
      }
      else
      {
        mMddq_dV_p.col(i).noalias() = DS_Dq * ddq;
      }
    }
    else
    {
      if (mParentBodyNode)
      {
        mMddq_dV_p.col(i) = AdInvT(T, mParentBodyNode->mMddq_dV_p.col(i));
      }
      else
      {
        mMddq_dV_p.col(i).setZero();
      }
    }
  }
}

//==============================================================================
void BodyNode::computeJacobianOfMBackward(
    neural::WithRespectTo* wrt, Eigen::MatrixXs& dMddq)
{
  using math::AdInvRLinear;
  using math::dad;
  using math::dAdInvT;
  using math::Jacobian;

  if (wrt != neural::WithRespectTo::POSITION)
  {
    // Shouldn't reach here
    assert(false);
  }

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();

  mMddq_F_p.resize(6, static_cast<int>(numDofs));

  const Eigen::Matrix6s& G = mAspectProperties.mInertia.getSpatialTensor();
  const Jacobian J = skel->getJacobian(this);
  const int jointNumDofs = static_cast<int>(mParentJoint->getNumDofs());

  // Update mMddq_dV
  mMddq_F = G * mMddq_dV + mAspectState.mFext;

  for (BodyNode* childBody : mChildBodyNodes)
  {
    const Joint* childJoint = childBody->getParentJoint();
    const Eigen::Isometry3s& childT = childJoint->getRelativeTransform();
    mMddq_F += dAdInvT(childT, childBody->mMddq_F);
  }

  // TODO(JS): iterate joints instead for vectorization
  for (auto i = 0u; i < numDofs; ++i)
  {
    const DegreeOfFreedom* dof = skel->getDof(i);
    const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());

    mMddq_F_p.col(i) = G * mMddq_dV_p.col(i);
    // TODO(JS): Add -(D F_{ext} / D q^k) to handle external forces other than
    // gravity

    for (BodyNode* childBody : mChildBodyNodes)
    {
      const Joint* childJoint = childBody->getParentJoint();
      const Eigen::Isometry3s& childT = childJoint->getRelativeTransform();

      if (childJoint->hasDof(dof))
      {
        const math::Jacobian& H
            = childJoint->getRelativeJacobianInPositionSpace();
        const Eigen::Vector6s Hcol = H.col(dofIndexInJoint);

        mMddq_F_p.col(i) += dAdInvT(
            childT,
            childBody->mMddq_F_p.col(i) - dad(Hcol, childBody->mMddq_F));
      }
      else
      {
        mMddq_F_p.col(i) += dAdInvT(childT, childBody->mMddq_F_p.col(i));
      }
    }

    // TODO: benchmark taking this outside of the for loop and just doing the
    // matrix multiply once
    if (jointNumDofs > 0)
    {
      const Jacobian& S = mParentJoint->getRelativeJacobian();

      const int iStart
          = static_cast<int>(mParentJoint->getDof(0)->getIndexInSkeleton());
      const int jStart = static_cast<int>(dof->getIndexInSkeleton());

      /*
      std::cout << "====" << std::endl
                << S.transpose() << std::endl
                << "*" << std::endl
                << mMddq_F_p.col(i) << std::endl
                << "=" << std::endl
                << S.transpose() * mMddq_F_p.col(i) << std::endl;
      */

      dMddq.block(iStart, jStart, jointNumDofs, 1)
          = S.transpose() * mMddq_F_p.col(i); // m x 1

      if (mParentJoint->hasDof(dof))
      {
        const Jacobian DS_Dq
            = mParentJoint->getRelativeJacobianDeriv(dof->getIndexInJoint());

        dMddq.block(iStart, jStart, jointNumDofs, 1)
            += DS_Dq.transpose() * mMddq_F;
      }
    }
  }
}

//==============================================================================
void BodyNode::computeJacobianOfCForward(neural::WithRespectTo* wrt)
{
  using math::ad;
  using math::AdInvT;
  using math::Jacobian;

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();

  // To update mCg_dV
  updateCombinedVector();

  mCg_V_p.resize(6, static_cast<int>(numDofs));
  mCg_dV_p.resize(6, static_cast<int>(numDofs));

  const Eigen::Isometry3s& T = mParentJoint->getRelativeTransform();
  const Jacobian& H = mParentJoint->getRelativeJacobianInPositionSpace();
  const Jacobian& S = mParentJoint->getRelativeJacobian();
  const Jacobian& dS = mParentJoint->getRelativeJacobianTimeDeriv();
  const Eigen::VectorXs& dq = mParentJoint->getVelocities();
  const Eigen::Vector6s& V = getSpatialVelocity();

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const auto bodyNodeIndex = getIndexInSkeleton();
  auto& data = skel->mDiffC.nodes[bodyNodeIndex].data;
#endif

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  data.V = V;
  data.dV = mCg_dV;
#endif

  if (wrt == neural::WithRespectTo::POSITION)
  {
    // TODO(JS): iterate joints instead for vectorization
    for (auto i = 0u; i < numDofs; ++i)
    {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
      auto& deriv = skel->mDiffC.nodes[bodyNodeIndex].derivs[i];
#endif

      const DegreeOfFreedom* dof = skel->getDof(i);
      if (mParentJoint->hasDof(dof))
      {
        const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());
        const Eigen::Vector6s Hcol = H.col(dofIndexInJoint);

        const Jacobian DS_Dq
            = mParentJoint->getRelativeJacobianDeriv(dof->getIndexInJoint());
        const Jacobian DdS_Dq
            = mParentJoint->getRelativeJacobianTimeDerivDerivWrtPosition(
                dof->getIndexInJoint());

        if (mParentBodyNode)
        {
          const Eigen::Vector6s& parent_V
              = mParentBodyNode->getSpatialVelocity();
          mCg_V_p.col(i) = AdInvT(T, mParentBodyNode->mCg_V_p.col(i))
                           - ad(Hcol, AdInvT(T, parent_V)) + DS_Dq * dq;

          const Eigen::Vector6s& parent_dV = mParentBodyNode->mCg_dV;
          mCg_dV_p.col(i) = AdInvT(T, mParentBodyNode->mCg_dV_p.col(i))
                            - ad(Hcol, AdInvT(T, parent_dV))
                            + ad(mCg_V_p.col(i), S * dq) + ad(V, DS_Dq * dq)
                            + DdS_Dq * dq;
        }
        else
        {
          mCg_V_p.col(i) = DS_Dq * dq;
          mCg_dV_p.col(i)
              = ad(mCg_V_p.col(i), S * dq) + ad(V, DS_Dq * dq) + DdS_Dq * dq;
        }
      }
      else
      {
        if (mParentBodyNode)
        {
          mCg_V_p.col(i) = AdInvT(T, mParentBodyNode->mCg_V_p.col(i));
          mCg_dV_p.col(i) = AdInvT(T, mParentBodyNode->mCg_dV_p.col(i))
                            + ad(mCg_V_p.col(i), S * dq);
        }
        else
        {
          mCg_V_p.col(i).setZero();
          mCg_dV_p.col(i) = ad(mCg_V_p.col(i), S * dq);
        }
      }

#ifdef DART_DEBUG_ANALYTICAL_DERIV
      deriv.V = mCg_V_p.col(i);
      deriv.dV = mCg_dV_p.col(i);
#endif
    }
  }
  else if (wrt == neural::WithRespectTo::VELOCITY)
  {
    for (auto i = 0u; i < numDofs; ++i)
    {
      const DegreeOfFreedom* dof = skel->getDof(i);
      if (mParentJoint->hasDof(dof))
      {
        const Jacobian DS_Dq
            = mParentJoint->getRelativeJacobianDeriv(dof->getIndexInJoint());
        const Jacobian DdS_Dq
            = mParentJoint->getRelativeJacobianTimeDerivDerivWrtPosition(
                dof->getIndexInJoint());
        const Jacobian DdS_Ddq
            = mParentJoint->getRelativeJacobianTimeDerivDerivWrtVelocity(
                dof->getIndexInJoint());

        const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());
        const Eigen::Vector6s Scol = S.col(dofIndexInJoint);
        const Eigen::Vector6s dScol = dS.col(dofIndexInJoint);

        if (mParentBodyNode)
        {
          mCg_V_p.col(i) = AdInvT(T, mParentBodyNode->mCg_V_p.col(i)) + Scol;
          mCg_dV_p.col(i) = AdInvT(T, mParentBodyNode->mCg_dV_p.col(i))
                            + ad(mCg_V_p.col(i), S * dq) + ad(V, Scol)
                            + DdS_Ddq * dq + dScol;
        }
        else
        {
          mCg_V_p.col(i) = Scol;
          mCg_dV_p.col(i)
              = ad(mCg_V_p.col(i), S * dq) + ad(V, Scol) + DdS_Ddq * dq + dScol;
        }
      }
      else
      {
        if (mParentBodyNode)
        {
          mCg_V_p.col(i) = AdInvT(T, mParentBodyNode->mCg_V_p.col(i));
          mCg_dV_p.col(i) = AdInvT(T, mParentBodyNode->mCg_dV_p.col(i))
                            + ad(mCg_V_p.col(i), S * dq);
        }
        else
        {
          mCg_V_p.col(i).setZero();
          mCg_dV_p.col(i) = ad(mCg_V_p.col(i), S * dq);
        }
      }
    }
  }
  else
  {
    // Shouldn't reach here
    assert(false);
  }
}

//============================================================================
void BodyNode::debugJacobianOfCForward(neural::WithRespectTo* wrt)
{
  computeJacobianOfCForward(wrt);
  const s_t threshold = 1e-9;
  Eigen::MatrixXs mCg_V_p_fd = finiteDifferenceJacobianOfSpatialVelocity(wrt);
  if (((mCg_V_p_fd - mCg_V_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mCg_V_p disagrees on body node " << getIndexInSkeleton()
              << "!" << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_V_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_V_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mCg_V_p - mCg_V_p_fd << std::endl;
  }
  Eigen::MatrixXs mCg_dV_p_fd
      = finiteDifferenceJacobianOfSpatialCoriolisAcceleration(wrt);
  if (((mCg_dV_p_fd - mCg_dV_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mCg_dV_p disagrees on body node " << getIndexInSkeleton()
              << "!" << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_dV_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_dV_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mCg_dV_p - mCg_dV_p_fd << std::endl;
  }
}

//============================================================================
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfSpatialVelocity(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfSpatialVelocity(wrt);
  }
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  const s_t EPS = 1e-6;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s plus = getSpatialVelocity();

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s minus = getSpatialVelocity();

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);

  return jac;
}

//============================================================================
/// This computes the Jacobian of spatial velocity with respect
/// to wrt using Ridders method
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfSpatialVelocity(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s plus = getSpatialVelocity();

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s minus = getSpatialVelocity();

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      Eigen::Vector6s plus = getSpatialVelocity();

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      Eigen::Vector6s minus = getSpatialVelocity();

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();

  return jac;
}

//============================================================================
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfSpatialCoriolisAcceleration(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfSpatialCoriolisAcceleration(wrt);
  }
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  const s_t EPS = 1e-6;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    Eigen::Vector6s plus = mCg_dV;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    Eigen::Vector6s minus = mCg_dV;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);

  return jac;
}

//==============================================================================
/// This computes the Jacobian of spatial acceleration (mCg_dV) with respect
/// to wrt using Ridders method
Eigen::MatrixXs
BodyNode::finiteDifferenceRiddersJacobianOfSpatialCoriolisAcceleration(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    Eigen::Vector6s plus = mCg_dV;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    Eigen::Vector6s minus = mCg_dV;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      Eigen::Vector6s plus = mCg_dV;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      Eigen::Vector6s minus = mCg_dV;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
  }

  return jac;
}

//==============================================================================
void BodyNode::computeJacobianOfCBackward(
    neural::WithRespectTo* wrt,
    Eigen::MatrixXs& dCg,
    const Eigen::Vector3s& gravity)
{
  using math::AdInvRLinear;
  using math::dad;
  using math::dAdInvT;
  using math::Jacobian;

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(numDofs));
  aggregateCombinedVector(tmp, gravity);

  mCg_F_p.resize(6, static_cast<int>(numDofs));
  mCg_V_ad_IV_p.resize(6, static_cast<int>(numDofs));
  mCg_IdV_p.resize(6, static_cast<int>(numDofs));
  mCg_g_p.resize(6, static_cast<int>(numDofs));
  mCg_g_p.setZero();

  const Eigen::Matrix6s& G = mAspectProperties.mInertia.getSpatialTensor();
  const Eigen::Isometry3s& Tworld = getWorldTransform();
  const Jacobian J = skel->getJacobian(this);
  const Jacobian& H = skel->getJacobianInPositionSpace(this);
  const Eigen::Vector6s& V = getSpatialVelocity();
  const int jointNumDofs = static_cast<int>(mParentJoint->getNumDofs());

  if (wrt == neural::WithRespectTo::POSITION)
  {
    // TODO(JS): iterate joints instead for vectorization
    for (auto i = 0u; i < numDofs; ++i)
    {
      const DegreeOfFreedom* dof = skel->getDof(i);
      const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());
      const int dofIndexInSkeleton
          = static_cast<int>(dof->getIndexInSkeleton());

      // Derivative of gravity force
      mCg_g_p.col(i).tail<3>().noalias()
          = -1 * math::makeSkewSymmetric(H.col(dofIndexInSkeleton).head<3>())
            * Tworld.rotation().transpose() * gravity;
      mCg_g_p.col(i) = G * mCg_g_p.col(i);
      mCg_V_ad_IV_p.col(i)
          = dad(mCg_V_p.col(i), G * V) + dad(V, G * mCg_V_p.col(i));
      mCg_IdV_p.col(i) = G * mCg_dV_p.col(i);
      mCg_F_p.col(i) = mCg_IdV_p.col(i) - mCg_V_ad_IV_p.col(i) - mCg_g_p.col(i);
      // TODO(JS): Add -(D F_{ext} / D q^k) to handle external forces other than
      // gravity

      for (BodyNode* childBody : mChildBodyNodes)
      {
        const Joint* childJoint = childBody->getParentJoint();
        const Eigen::Isometry3s& childT = childJoint->getRelativeTransform();

        // mCg_F += math::dAdInvT((*it)->getParentJoint()->mT, (*it)->mCg_F);
        if (childJoint->hasDof(dof))
        {
          const Jacobian& H = childJoint->getRelativeJacobianInPositionSpace();
          const Eigen::Vector6s Hcol = H.col(dofIndexInJoint);

          mCg_F_p.col(i) += dAdInvT(
              childT, childBody->mCg_F_p.col(i) - dad(Hcol, childBody->mCg_F));
        }
        else
        {
          mCg_F_p.col(i) += dAdInvT(childT, childBody->mCg_F_p.col(i));
        }
      }

      if (jointNumDofs > 0)
      {
        const Jacobian& S = mParentJoint->getRelativeJacobian();

        Eigen::MatrixXs dCg_seg = S.transpose() * mCg_F_p.col(i); // m x 1
        if (mParentJoint->hasDof(dof))
        {
          const Jacobian DS_Dq
              = mParentJoint->getRelativeJacobianDeriv(dof->getIndexInJoint());

          dCg_seg += DS_Dq.transpose() * mCg_F;
        }

        const int iStart
            = static_cast<int>(mParentJoint->getDof(0)->getIndexInSkeleton());
        const int jStart = static_cast<int>(dof->getIndexInSkeleton());
        dCg.block(iStart, jStart, jointNumDofs, 1) = dCg_seg;
      }
    }
  }
  else if (wrt == neural::WithRespectTo::VELOCITY)
  {
    // TODO(JS): Vectorize instead of iterating DOFs
    for (auto i = 0u; i < numDofs; ++i)
    {
      const DegreeOfFreedom* dof = skel->getDof(i);

      mCg_V_ad_IV_p.col(i)
          = dad(mCg_V_p.col(i), G * V) + dad(V, G * mCg_V_p.col(i));
      mCg_IdV_p.col(i) = G * mCg_dV_p.col(i);
      mCg_F_p.col(i) = mCg_IdV_p.col(i) - mCg_V_ad_IV_p.col(i);
      // TODO(JS): Add -(D F_{ext} / D q^k) for the case that F_ext (other than
      // gravity force) is a function of q

      for (const BodyNode* childBody : mChildBodyNodes)
      {
        const Joint* childJoint = childBody->getParentJoint();
        const Eigen::Isometry3s& childT = childJoint->getRelativeTransform();
        mCg_F_p.col(i) += dAdInvT(childT, childBody->mCg_F_p.col(i));
      }

      if (jointNumDofs > 0)
      {
        const Jacobian& S = mParentJoint->getRelativeJacobian();
        const int iStart
            = static_cast<int>(mParentJoint->getDof(0)->getIndexInSkeleton());
        const int jStart = static_cast<int>(dof->getIndexInSkeleton());
        dCg.block(iStart, jStart, jointNumDofs, 1).noalias()
            = S.transpose() * mCg_F_p.col(i);
      }
    }
  }
  else
  {
    // Shouldn't reach here
    assert(false);
  }
}

//==============================================================================
/// This checks the intermediate analytical results of
/// computeJacobianOfCBackword() against the finite differencing equivalents.
void BodyNode::debugJacobianOfCBackward(neural::WithRespectTo* wrt)
{
  Eigen::MatrixXs dC = Eigen::MatrixXs::Zero(
      getSkeleton()->getNumDofs(), getSkeleton()->getNumDofs());
  computeJacobianOfCBackward(
      wrt, dC, getSkeleton()->mAspectProperties.mGravity);

  const s_t threshold = 1e-9;
  // mCg_g_p holds an inverse of the gradient of gravity wrt position
  Eigen::MatrixXs mCg_g_p_fd = finiteDifferenceJacobianOfGravityForce(wrt);
  if (((mCg_g_p_fd - mCg_g_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mCg_g_p disagrees on body node " << getIndexInSkeleton()
              << "! " << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_g_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_g_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mCg_g_p - mCg_g_p_fd << std::endl;
  }
  Eigen::MatrixXs mCg_F_p_fd = finiteDifferenceJacobianOfBodyForce(wrt);
  if (((mCg_F_p_fd - mCg_F_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mCg_F_p disagrees on body node " << getIndexInSkeleton()
              << "! num children = " << mChildBodyNodes.size() << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_F_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_F_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mCg_F_p - mCg_F_p_fd << std::endl;
  }
  Eigen::MatrixXs mCg_V_ad_IV_p_fd
      = finiteDifferenceJacobianOfBodyForceAdVIV(wrt);
  if (((mCg_V_ad_IV_p - mCg_V_ad_IV_p_fd).cwiseAbs().array() > threshold).any())
  {
    std::cout << "ad(V, I*V) disagrees on body node " << getIndexInSkeleton()
              << "!" << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_V_ad_IV_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_V_ad_IV_p_fd << std::endl;
    std::cout << "Diff:" << std::endl
              << mCg_V_ad_IV_p - mCg_V_ad_IV_p_fd << std::endl;
  }
  Eigen::MatrixXs mCg_IdV_p_fd = finiteDifferenceJacobianOfBodyForceIdV(wrt);
  if (((mCg_IdV_p - mCg_IdV_p_fd).cwiseAbs().array() > threshold).any())
  {
    std::cout << "I*dV disagrees on body node " << getIndexInSkeleton() << "!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << mCg_IdV_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mCg_IdV_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mCg_IdV_p - mCg_IdV_p_fd << std::endl;
  }
}

//==============================================================================
/// This computes the Jacobian of gravity force (mFgravity) with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfGravityForce(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfGravityForce(wrt);
  }
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  const s_t EPS = 1e-6;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s plus = mFgravity;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s minus = mFgravity;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  skel->computeForwardDynamics();

  return jac;
}

//==============================================================================
/// This computes the Jacobian of gravity force (mFgravity) with respect to
/// wrt using Ridders method
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfGravityForce(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s plus = mFgravity;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s minus = mFgravity;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      Eigen::Vector6s plus = mFgravity;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      Eigen::Vector6s minus = mFgravity;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of body force (mCg_F) with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfBodyForce(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfBodyForce(wrt);
  }

  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t EPS = 1e-4;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
    }
    Eigen::Vector6s plus = mCg_F;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
    }
    Eigen::Vector6s minus = mCg_F;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of body force (mCg_F) with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfBodyForce(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-2;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
    }
    Eigen::Vector6s plus = mCg_F;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
    }
    Eigen::Vector6s minus = mCg_F;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
      }
      Eigen::Vector6s plus = mCg_F;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
      }
      Eigen::Vector6s minus = mCg_F;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateCombinedVector(tmp, skel->getGravity());
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the Ad(V,IV) subexpression of body force
/// (mCg_F) with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfBodyForceAdVIV(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfBodyForceAdVIV(wrt);

  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t EPS = 1e-4;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    // mCg_F = mI * mCg_dV;
    // mCg_F -= mFgravity;
    // mCg_F -= math::dad(V, mI * V);
    const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
    const Eigen::Vector6s& V = getSpatialVelocity();
    Eigen::Vector6s plus = math::dad(V, mI * V);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI_minus
        = mAspectProperties.mInertia.getSpatialTensor();
    const Eigen::Vector6s& V_minus = getSpatialVelocity();
    Eigen::Vector6s minus = math::dad(V_minus, mI_minus * V_minus);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the Ad(V,IV) subexpression of body force
/// (mCg_F) with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfBodyForceAdVIV(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
    const Eigen::Vector6s& V = getSpatialVelocity();
    Eigen::Vector6s plus = math::dad(V, mI * V);

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI_minus
        = mAspectProperties.mInertia.getSpatialTensor();
    const Eigen::Vector6s& V_minus = getSpatialVelocity();
    Eigen::Vector6s minus = math::dad(V_minus, mI_minus * V_minus);

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
      const Eigen::Vector6s& V = getSpatialVelocity();
      plus = math::dad(V, mI * V);

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      const Eigen::Matrix6s& mI_minus
          = mAspectProperties.mInertia.getSpatialTensor();
      const Eigen::Vector6s& V_minus = getSpatialVelocity();
      Eigen::Vector6s minus = math::dad(V_minus, mI_minus * V_minus);

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the I*dV subexpression of body force (mCg_F)
/// with respect to wrt
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfBodyForceIdV(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfBodyForceIdV(wrt);

  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t EPS = 1e-4;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    // mCg_F = mI * mCg_dV;
    // mCg_F -= mFgravity;
    // mCg_F -= math::dad(V, mI * V);
    const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
    Eigen::Vector6s plus = mI * mCg_dV;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI_minus
        = mAspectProperties.mInertia.getSpatialTensor();
    Eigen::Vector6s minus = mI_minus * mCg_dV;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the I*dV subexpression of body force (mCg_F)
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfBodyForceIdV(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
    Eigen::Vector6s plus = mI * mCg_dV;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateCombinedVector();
    }
    const Eigen::Matrix6s& mI_minus
        = mAspectProperties.mInertia.getSpatialTensor();
    Eigen::Vector6s minus = mI_minus * mCg_dV;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      const Eigen::Matrix6s& mI = mAspectProperties.mInertia.getSpatialTensor();
      Eigen::Vector6s plus = mI * mCg_dV;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      skel->computeForwardDynamics();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateCombinedVector();
      }
      const Eigen::Matrix6s& mI_minus
          = mAspectProperties.mInertia.getSpatialTensor();
      Eigen::Vector6s minus = mI_minus * mCg_dV;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  skel->computeForwardDynamics();
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateCombinedVector();
  }

  return jac;
}

//==============================================================================
/// This checks the intermediate analytical results of
/// computeJacobianOfMForward() against the finite differencing equivalents.
void BodyNode::debugJacobianOfMForward(
    neural::WithRespectTo* wrt, Eigen::VectorXs x)
{
  auto skel = getSkeleton();
  Eigen::VectorXs oldAccel = skel->getAccelerations();
  skel->setAccelerations(x);
  Eigen::MatrixXs tmp
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), skel->getNumDofs());
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->computeJacobianOfMForward(wrt);
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->computeJacobianOfMBackward(wrt, tmp);
  }

  const s_t threshold = 1e-8;
  // mM_dV_p holds the gradient of spatial acceleration with respect to position
  Eigen::MatrixXs mMddq_dV_p_fd
      = finiteDifferenceJacobianOfMassSpatialAcceleration(wrt);
  if (((mMddq_dV_p_fd - mMddq_dV_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mMddq_dV_p disagrees on body node " << getIndexInSkeleton()
              << "! " << std::endl;
    std::cout << "Analytical:" << std::endl << mMddq_dV_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mMddq_dV_p_fd << std::endl;
    std::cout << "Diff:" << std::endl
              << mMddq_dV_p - mMddq_dV_p_fd << std::endl;
  }
  getSkeleton()->setAccelerations(oldAccel);
}

//==============================================================================
/// This computes the Jacobian of the dV (mM_dV) as it's computed in the M
/// computation for some given acceleration
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfMassSpatialAcceleration(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMassSpatialAcceleration(wrt);

  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t EPS = 1e-4;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    Eigen::Vector6s plus = mM_dV;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    Eigen::Vector6s minus = mM_dV;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  // This isn't strictly necessary, because mM_dV is an intermediate variable
  // not used elsewhere without being recomputed first, but just to keep things
  // visually consistent across finite differencing code (and to future proof
  // against copy-pasta errors), we recalculate the state after a reset
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateMassMatrix();
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the dV (mM_dV) as it's computed in the M
/// computation
Eigen::MatrixXs
BodyNode::finiteDifferenceRiddersJacobianOfMassSpatialAcceleration(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::VectorXs tmp;
  tmp.resize(static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    Eigen::Vector6s plus = mM_dV;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    Eigen::Vector6s minus = mM_dV;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateMassMatrix();
      }
      Eigen::Vector6s plus = mM_dV;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateMassMatrix();
      }
      Eigen::Vector6s minus = mM_dV;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  // This isn't strictly necessary, because mM_dV is an intermediate variable
  // not used elsewhere without being recomputed first, but just to keep things
  // visually consistent across finite differencing code (and to future proof
  // against copy-pasta errors), we recalculate the state after a reset
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateMassMatrix();
  }

  return jac;
}

//==============================================================================
/// This checks the intermediate analytical results of
/// computeJacobianOfMForward() against the finite differencing equivalents.
void BodyNode::debugJacobianOfMBackward(
    neural::WithRespectTo* wrt, Eigen::VectorXs x, Eigen::MatrixXs dM)
{
  auto skel = getSkeleton();
  Eigen::VectorXs oldAccel = skel->getAccelerations();
  skel->setAccelerations(x);
  Eigen::MatrixXs tmp
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), skel->getNumDofs());
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->computeJacobianOfMForward(wrt);
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->computeJacobianOfMBackward(wrt, tmp);
  }

  const s_t threshold = 5e-10;
  // mM_dV_p holds the gradient of spatial acceleration with respect to position
  Eigen::MatrixXs mMddq_F_p_fd = finiteDifferenceJacobianOfMassBodyForce(wrt);
  if (((mMddq_F_p_fd - mMddq_F_p).cwiseAbs().array() > threshold).any())
  {
    std::cout << "mMddq_F_p disagrees on body node " << getIndexInSkeleton()
              << "! " << std::endl;
    std::cout << "Analytical:" << std::endl << mMddq_F_p << std::endl;
    std::cout << "Brute Force:" << std::endl << mMddq_F_p_fd << std::endl;
    std::cout << "Diff:" << std::endl << mMddq_F_p - mMddq_F_p_fd << std::endl;
  }

  int parentDofs = mParentJoint->getNumDofs();
  if (parentDofs > 0)
  {
    const Eigen::MatrixXs& S = mParentJoint->getRelativeJacobian();

    const int iStart
        = static_cast<int>(mParentJoint->getDof(0)->getIndexInSkeleton());
    Eigen::MatrixXs correct = dM.block(iStart, 0, parentDofs, dM.cols());

    Eigen::MatrixXs analytical = S.transpose() * mMddq_F_p; // m x 1
    for (int i = 0; i < parentDofs; i++)
    {
      const Eigen::MatrixXs& DS_Dq = mParentJoint->getRelativeJacobianDeriv(i);
      analytical.col(i) += DS_Dq.transpose() * mMddq_F;
    }

    if (((correct - analytical).cwiseAbs().array() > threshold).any())
    {
      std::cout << "mMddq disagrees on body node " << getIndexInSkeleton()
                << "! " << std::endl;
      std::cout << "Analytical:" << std::endl << analytical << std::endl;
      std::cout << "Brute Force:" << std::endl << correct << std::endl;
      std::cout << "Diff:" << std::endl << analytical - correct << std::endl;
      std::cout << "S:" << std::endl << S << std::endl;
      std::cout << "mMddq_F_p:" << std::endl << mMddq_F_p << std::endl;
    }
  }

  getSkeleton()->setAccelerations(oldAccel);
}

//==============================================================================
/// This computes the Jacobian of the F (mM_F) as it's computed in the M
/// computation
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfMassBodyForce(
    neural::WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMassBodyForce(wrt);
  // void BodyNode::updateMassMatrix();
  // void BodyNode::aggregateMassMatrix(Eigen::MatrixXs& _MCol, std::size_t
  // _col);
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  const s_t EPS = 1e-4;
  Eigen::VectorXs original = wrt->get(skel);
  Eigen::MatrixXs tmp = Eigen::MatrixXs::Zero(
      static_cast<int>(skel->getNumDofs()),
      static_cast<int>(skel->getNumDofs()));
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
    }
    Eigen::Vector6s plus = mM_F;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
    }
    Eigen::Vector6s minus = mM_F;

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  // This isn't strictly necessary, because mM_F is an intermediate variable
  // not used elsewhere without being recomputed first, but just to keep things
  // visually consistent across finite differencing code (and to future proof
  // against copy-pasta errors), we recalculate the state after a reset
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateMassMatrix();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
  }

  return jac;
}

//==============================================================================
/// This computes the Jacobian of the F (mM_F) as it's computed in the M
/// computation
Eigen::MatrixXs BodyNode::finiteDifferenceRiddersJacobianOfMassBodyForce(
    neural::WithRespectTo* wrt)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  Eigen::MatrixXs tmp = Eigen::MatrixXs::Zero(
      static_cast<int>(skel->getNumDofs()),
      static_cast<int>(skel->getNumDofs()));

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
    }
    Eigen::Vector6s plus = mM_F;

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(skel, perturbed);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->updateMassMatrix();
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
    }
    Eigen::Vector6s minus = mM_F;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(skel, perturbed);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateMassMatrix();
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
      }
      Eigen::Vector6s plus = mM_F;

      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(skel, perturbed);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->updateMassMatrix();
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
      }
      Eigen::Vector6s minus = mM_F;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i).noalias() = tab[jTab][iTab];
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
  wrt->set(skel, original);
  // This isn't strictly necessary, because mM_F is an intermediate variable
  // not used elsewhere without being recomputed first, but just to keep things
  // visually consistent across finite differencing code (and to future proof
  // against copy-pasta errors), we recalculate the state after a reset
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    skel->getBodyNode(j)->updateMassMatrix();
  }
  for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
  {
    skel->getBodyNode(j)->aggregateMassMatrix(tmp, 0);
  }

  return jac;
}

//==============================================================================
void BodyNode::computeJacobianOfMinvXInit()
{
  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();

  mInvM_DAI_Dq.resize(numDofs);
  mInvM_DAB_Dq.setZero(6, static_cast<int>(numDofs));
  mInvM_DPi_Dq.resize(numDofs);
  mInvM_Dbeta_Dq.setZero(6, static_cast<int>(numDofs));
  mInvM_dV_q.setZero(6, static_cast<int>(numDofs));

  mParentJoint->computeJacobianOfMinvX_init();
}

//==============================================================================
void BodyNode::computeJacobianOfMinvXBackward()
{
  using math::ad;
  using math::AdInvRLinear;
  using math::adMatrix;
  using math::AdTMatrix;
  using math::dad;
  using math::dAdInvT;
  using math::dAdTMatrix;
  using math::Jacobian;

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();
#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const auto bodyNodeIndex = getIndexInSkeleton();
  auto& data = skel->mDiffMinv.nodes[bodyNodeIndex].data;
#endif

  // Compute articulated bias force (mInvM_c) and alpha (mInvM_a)...
  updateInvMassMatrix();

  const math::Inertia& AI = getArticulatedInertia();
  const Eigen::Vector6s& AB = mInvM_c;

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  data.AI = AI;
  data.AB = AB;
#endif

  math::Jacobian& DAB_Dq = mInvM_DAB_Dq;

  for (auto i = 0u; i < numDofs; ++i)
  {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
    auto& deriv = skel->mDiffMinv.nodes[bodyNodeIndex].derivs[i];
#endif
    const DegreeOfFreedom* dof = skel->getDof(i);
    const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());

    mInvM_DAI_Dq[i].setZero();
    DAB_Dq.col(i).setZero();
    for (BodyNode* childBody : mChildBodyNodes)
    {
      const Joint* childJoint = childBody->getParentJoint();
      const Eigen::Isometry3s& childT = childJoint->getRelativeTransform();
      const math::Inertia& childAI = childBody->getArticulatedInertia();
      const Eigen::Vector6s& childAB = childBody->mInvM_c;

      if (childJoint->hasDof(dof))
      {
        const math::Jacobian& H
            = childJoint->getRelativeJacobianInPositionSpace();
        const Eigen::Vector6s Hcol = H.col(dofIndexInJoint);
        const math::Inertia Pi = childJoint->computePi(childAI);
        const Eigen::Matrix6s adH = adMatrix(Hcol);
        const Eigen::Vector6s child_beta
            = childJoint->computeBeta(childAI, childAB);
        const math::Inertia tmp = Pi * adH;

        mInvM_DAI_Dq[i]
            += math::dAdInvTMatrix(childT)
               * (childBody->mInvM_DPi_Dq[i] - tmp - tmp.transpose())
               * math::AdInvTMatrix(childT);

        DAB_Dq.col(i) += dAdInvT(
            childT, childBody->mInvM_Dbeta_Dq.col(i) - dad(Hcol, child_beta));
      }
      else
      {
        mInvM_DAI_Dq[i] += math::dAdInvTMatrix(childT)
                           * childBody->mInvM_DPi_Dq[i]
                           * math::AdInvTMatrix(childT);

        DAB_Dq.col(i) += dAdInvT(childT, childBody->mInvM_Dbeta_Dq.col(i));
      }
    }
#ifdef DART_DEBUG_ANALYTICAL_DERIV
    deriv.AI = mInvM_DAI_Dq[i];
    deriv.AB = DAB_Dq.col(i);
#endif
  }

  mInvM_Dbeta_Dq = DAB_Dq;
  mParentJoint->computeJacobianOfMinvX_A(AI, AB);
}

//==============================================================================
void BodyNode::computeJacobianOfMinvXForward(Eigen::MatrixXs& DinvMx_Dq)
{
  using math::ad;
  using math::AdInvRLinear;
  using math::AdInvT;
  using math::dad;
  using math::dAdInvT;
  using math::Jacobian;

  const auto skel = getSkeleton();
  const auto numDofs = skel->getNumDofs();
#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const auto bodyNodeIndex = getIndexInSkeleton();
  auto& data = skel->mDiffMinv.nodes[bodyNodeIndex].data;
#endif

  const Joint* joint = mParentJoint;
  const int jointNumDofs = static_cast<int>(joint->getNumDofs());
  const BodyNode* parentBody = mParentBodyNode;
  const Eigen::Isometry3s& T = mParentJoint->getRelativeTransform();

  const math::Inertia& AI = getArticulatedInertia();

  Eigen::MatrixXs invMx = Eigen::MatrixXs::Zero(static_cast<int>(numDofs), 1);
#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const Jacobian& S = mParentJoint->getRelativeJacobian();
  data.S = S;
#endif
  if (getNumChildBodyNodes() > 0)
  {
    if (parentBody)
    {
      mParentJoint->getInvMassMatrixSegment(invMx, 0, AI, parentBody->mInvM_U);
      mInvM_U = math::AdInvT(T, parentBody->mInvM_U);
    }
    else
    {
      mParentJoint->getInvMassMatrixSegment(
          invMx, 0, AI, Eigen::Vector6s::Zero());
      mInvM_U.setZero();
    }
    mParentJoint->addInvMassMatrixSegmentTo(mInvM_U);
  }

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  Eigen::VectorXs ddq = Eigen::VectorXs::Zero(0);
  if (jointNumDofs > 0)
  {
    const int indexI = static_cast<int>(joint->getDof(0)->getIndexInSkeleton());
    ddq = invMx.block(indexI, 0, jointNumDofs, 1);
  }
  data.ddq = ddq;
  data.dV = mInvM_U;
#endif

  if (jointNumDofs > 0)
  {
    const int iStart
        = static_cast<int>(mParentJoint->getDof(0)->getIndexInSkeleton());
    const Eigen::MatrixXs block = mParentJoint->computeJacobianOfMinvX_B(AI);
    DinvMx_Dq.block(iStart, 0, jointNumDofs, static_cast<int>(numDofs)) = block;
  }
}

//==============================================================================
/// This checks the intermediate analytical results of
/// computeJacobianOfMinvXBackward() against the finite differencing
/// equivalents.
void BodyNode::debugJacobianOfMinvXBackward(
    neural::WithRespectTo* wrt, Eigen::VectorXs x)
{
  auto skel = getSkeleton();
  const Eigen::VectorXs oldForces = skel->getControlForces();
  skel->setControlForces(x);

  const s_t threshold = 5e-10;

  computeJacobianOfMinvXBackward();
  std::vector<Eigen::MatrixXs> mInvM_DAI_Dq_fd
      = finiteDifferenceJacobianOfInvMassArtInertia(wrt);

  int dofs = wrt->dim(skel.get());
  for (int i = 0; i < dofs; i++)
  {
    Eigen::Matrix6s bruteForce = mInvM_DAI_Dq_fd[i];
    Eigen::Matrix6s analytical = mInvM_DAI_Dq[i];
    if (((bruteForce - analytical).cwiseAbs().array() > threshold).any())
    {
      std::cout << "Articulated Inertia disagrees on dof " << i << "! "
                << std::endl;
      std::cout << "Analytical:" << std::endl << analytical << std::endl;
      std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
      std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    }
  }

  Eigen::MatrixXs mInvM_DAB_Dq_fd
      = finiteDifferenceJacobianOfInvMassArtBias(wrt);
  if (((mInvM_DAB_Dq_fd - mInvM_Dbeta_Dq).cwiseAbs().array() > threshold).any())
  {
    std::cout << "Articulated bias force disagrees on node "
              << getIndexInSkeleton() << "! " << std::endl;
    std::cout << "Analytical:" << std::endl << mInvM_Dbeta_Dq << std::endl;
    std::cout << "Brute Force:" << std::endl << mInvM_DAB_Dq_fd << std::endl;
    std::cout << "Diff:" << std::endl
              << mInvM_Dbeta_Dq - mInvM_DAB_Dq_fd << std::endl;
  }

  skel->setControlForces(oldForces);
}

//==============================================================================
/// This computes the Jacobian of the articulate inertia (mArtInertia) as it's
/// computed in the Minv computation
std::vector<Eigen::MatrixXs>
BodyNode::finiteDifferenceJacobianOfInvMassArtInertia(
    neural::WithRespectTo* wrt, bool /* useRidders */)
{
  /*
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMassBodyForce(wrt);
  */

  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  std::vector<Eigen::MatrixXs> jacs;
  jacs.resize(dofs);

  const s_t EPS = 1e-5;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    Eigen::Matrix6s plus = getArticulatedInertia();

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    Eigen::Matrix6s minus = getArticulatedInertia();
    jacs[i] = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  // This isn't strictly necessary, because inertia is an intermediate variable
  // not used elsewhere without being recomputed first, but just to keep things
  // visually consistent across finite differencing code (and to future proof
  // against copy-pasta errors), we recalculate the state after a reset
  getArticulatedInertia();

  return jacs;
}

//==============================================================================
/// This computes the Jacobian (tensor) of the articulate bias force (mInvM_c)
/// as it's computed in the Minv computation
Eigen::MatrixXs BodyNode::finiteDifferenceJacobianOfInvMassArtBias(
    neural::WithRespectTo* wrt, bool /* useRidders */)
{
  auto skel = getSkeleton().get();
  int dofs = wrt->dim(skel);
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(6, dofs);

  const s_t EPS = 1e-5;
  Eigen::VectorXs original = wrt->get(skel);
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s plus = getBiasForce();

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(skel, perturbed);
    skel->computeForwardDynamics();
    Eigen::Vector6s minus = getBiasForce();

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  wrt->set(skel, original);
  skel->computeForwardDynamics();

  return jac;
}

//==============================================================================
/// This checks the intermediate analytical results of
/// computeJacobianOfMinvXForward() against the finite differencing equivalents.
void BodyNode::debugJacobianOfMinvXForward(
    neural::WithRespectTo* wrt, Eigen::VectorXs x)
{
  (void)wrt;
  (void)x;
}

//==============================================================================
const Eigen::Vector6s& BodyNode::getBiasForce() const
{
  return mBiasForce;
}

} // namespace dynamics
} // namespace dart
