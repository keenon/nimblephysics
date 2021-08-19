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

#include "dart/dynamics/Joint.hpp"

#include <string>

#include "dart/common/Console.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
const Joint::ActuatorType Joint::DefaultActuatorType
    = detail::DefaultActuatorType;
// These declarations are needed for linking to work
constexpr Joint::ActuatorType Joint::FORCE;
constexpr Joint::ActuatorType Joint::PASSIVE;
constexpr Joint::ActuatorType Joint::SERVO;
constexpr Joint::ActuatorType Joint::MIMIC;
constexpr Joint::ActuatorType Joint::ACCELERATION;
constexpr Joint::ActuatorType Joint::VELOCITY;
constexpr Joint::ActuatorType Joint::LOCKED;

namespace detail {

//==============================================================================
JointProperties::JointProperties(
    const std::string& _name,
    const Eigen::Isometry3s& _T_ParentBodyToJoint,
    const Eigen::Isometry3s& _T_ChildBodyToJoint,
    bool _isPositionLimitEnforced,
    ActuatorType _actuatorType,
    const Joint* _mimicJoint,
    s_t _mimicMultiplier,
    s_t _mimicOffset)
  : mName(_name),
    mT_ParentBodyToJoint(_T_ParentBodyToJoint),
    mT_ChildBodyToJoint(_T_ChildBodyToJoint),
    mParentScale(1.0),
    mChildScale(1.0),
    mOriginalParentTranslation(_T_ParentBodyToJoint.translation()),
    mOriginalChildTranslation(_T_ChildBodyToJoint.translation()),
    mIsPositionLimitEnforced(_isPositionLimitEnforced),
    mActuatorType(_actuatorType),
    mMimicJoint(_mimicJoint),
    mMimicMultiplier(_mimicMultiplier),
    mMimicOffset(_mimicOffset)
{
  // Do nothing
}

} // namespace detail

//==============================================================================
Joint::ExtendedProperties::ExtendedProperties(
    const Properties& standardProperties,
    const CompositeProperties& aspectProperties)
  : Properties(standardProperties), mCompositeProperties(aspectProperties)
{
  // Do nothing
}

//==============================================================================
Joint::ExtendedProperties::ExtendedProperties(
    Properties&& standardProperties, CompositeProperties&& aspectProperties)
  : Properties(std::move(standardProperties)),
    mCompositeProperties(std::move(aspectProperties))
{
  // Do nothing
}

//==============================================================================
Joint::~Joint()
{
  // Do nothing
}

//==============================================================================
void Joint::setProperties(const Properties& properties)
{
  setAspectProperties(properties);
}

//==============================================================================
void Joint::setAspectProperties(const AspectProperties& properties)
{
  setName(properties.mName);
  setTransformFromParentBodyNode(properties.mT_ParentBodyToJoint);
  mAspectProperties.mParentScale = properties.mParentScale;
  mAspectProperties.mOriginalParentTranslation
      = properties.mOriginalParentTranslation;
  setTransformFromChildBodyNode(properties.mT_ChildBodyToJoint);
  mAspectProperties.mChildScale = properties.mChildScale;
  mAspectProperties.mOriginalChildTranslation
      = properties.mOriginalChildTranslation;
  setPositionLimitEnforced(properties.mIsPositionLimitEnforced);
  setActuatorType(properties.mActuatorType);
  setMimicJoint(
      properties.mMimicJoint,
      properties.mMimicMultiplier,
      properties.mMimicOffset);
}

//==============================================================================
const Joint::Properties& Joint::getJointProperties() const
{
  return mAspectProperties;
}

//==============================================================================
void Joint::copy(const Joint& _otherJoint)
{
  if (this == &_otherJoint)
    return;

  setProperties(_otherJoint.getJointProperties());
}

//==============================================================================
void Joint::copy(const Joint* _otherJoint)
{
  if (nullptr == _otherJoint)
    return;

  copy(*_otherJoint);
}

//==============================================================================
Joint& Joint::operator=(const Joint& _otherJoint)
{
  copy(_otherJoint);
  return *this;
}

//==============================================================================
const std::string& Joint::setName(const std::string& _name, bool _renameDofs)
{
  if (mAspectProperties.mName == _name)
  {
    if (_renameDofs)
      updateDegreeOfFreedomNames();
    return mAspectProperties.mName;
  }

  const SkeletonPtr& skel
      = mChildBodyNode ? mChildBodyNode->getSkeleton() : nullptr;
  if (skel)
  {
    skel->mNameMgrForJoints.removeName(mAspectProperties.mName);
    mAspectProperties.mName = _name;

    skel->addEntryToJointNameMgr(this, _renameDofs);
  }
  else
  {
    mAspectProperties.mName = _name;

    if (_renameDofs)
      updateDegreeOfFreedomNames();
  }

  return mAspectProperties.mName;
}

//==============================================================================
const std::string& Joint::getName() const
{
  return mAspectProperties.mName;
}

//==============================================================================
void Joint::setActuatorType(Joint::ActuatorType _actuatorType)
{
  mAspectProperties.mActuatorType = _actuatorType;
}

//==============================================================================
Joint::ActuatorType Joint::getActuatorType() const
{
  return mAspectProperties.mActuatorType;
}

//==============================================================================
void Joint::setMimicJoint(
    const Joint* _mimicJoint, s_t _mimicMultiplier, s_t _mimicOffset)
{
  mAspectProperties.mMimicJoint = _mimicJoint;
  mAspectProperties.mMimicMultiplier = _mimicMultiplier;
  mAspectProperties.mMimicOffset = _mimicOffset;
}

//==============================================================================
const Joint* Joint::getMimicJoint() const
{
  return mAspectProperties.mMimicJoint;
}

//==============================================================================
s_t Joint::getMimicMultiplier() const
{
  return mAspectProperties.mMimicMultiplier;
}

//==============================================================================
s_t Joint::getMimicOffset() const
{
  return mAspectProperties.mMimicOffset;
}

//==============================================================================
bool Joint::isKinematic() const
{
  switch (mAspectProperties.mActuatorType)
  {
    case FORCE:
    case PASSIVE:
    case SERVO:
    case MIMIC:
      return false;
    case ACCELERATION:
    case VELOCITY:
    case LOCKED:
      return true;
    default: {
      dterr << "Unsupported actuator type." << std::endl;
      return false;
    }
  }
}

//==============================================================================
bool Joint::isDynamic() const
{
  return !isKinematic();
}

//==============================================================================
BodyNode* Joint::getChildBodyNode()
{
  return mChildBodyNode;
}

//==============================================================================
const BodyNode* Joint::getChildBodyNode() const
{
  return mChildBodyNode;
}

//==============================================================================
BodyNode* Joint::getParentBodyNode()
{
  if (mChildBodyNode)
    return mChildBodyNode->getParentBodyNode();

  return nullptr;
}

//==============================================================================
const BodyNode* Joint::getParentBodyNode() const
{
  return const_cast<Joint*>(this)->getParentBodyNode();
}

//==============================================================================
SkeletonPtr Joint::getSkeleton()
{
  return mChildBodyNode ? mChildBodyNode->getSkeleton() : nullptr;
}

//==============================================================================
std::shared_ptr<const Skeleton> Joint::getSkeleton() const
{
  return mChildBodyNode ? mChildBodyNode->getSkeleton() : nullptr;
}

//==============================================================================
const Eigen::Isometry3s& Joint::getLocalTransform() const
{
  return getRelativeTransform();
}

//==============================================================================
const Eigen::Vector6s& Joint::getLocalSpatialVelocity() const
{
  return getRelativeSpatialVelocity();
}

//==============================================================================
const Eigen::Vector6s& Joint::getLocalSpatialAcceleration() const
{
  return getRelativeSpatialAcceleration();
}

//==============================================================================
const Eigen::Vector6s& Joint::getLocalPrimaryAcceleration() const
{
  return getRelativePrimaryAcceleration();
}

//==============================================================================
const math::Jacobian Joint::getLocalJacobian() const
{
  return getRelativeJacobian();
}

//==============================================================================
math::Jacobian Joint::getLocalJacobian(const Eigen::VectorXs& positions) const
{
  return getRelativeJacobian(positions);
}

//==============================================================================
const math::Jacobian Joint::getLocalJacobianTimeDeriv() const
{
  return getRelativeJacobianTimeDeriv();
}

//==============================================================================
const Eigen::Isometry3s& Joint::getRelativeTransform() const
{
  if (mNeedTransformUpdate)
  {
    updateRelativeTransform();
    mNeedTransformUpdate = false;
  }

  return mT;
}

//==============================================================================
const Eigen::Vector6s& Joint::getRelativeSpatialVelocity() const
{
  if (mNeedSpatialVelocityUpdate)
  {
    updateRelativeSpatialVelocity();
    mNeedSpatialVelocityUpdate = false;
  }

  return mSpatialVelocity;
}

//==============================================================================
const Eigen::Vector6s& Joint::getRelativeSpatialAcceleration() const
{
  if (mNeedSpatialAccelerationUpdate)
  {
    updateRelativeSpatialAcceleration();
    mNeedSpatialAccelerationUpdate = false;
  }

  return mSpatialAcceleration;
}

//==============================================================================
const Eigen::Vector6s& Joint::getRelativePrimaryAcceleration() const
{
  if (mNeedPrimaryAccelerationUpdate)
  {
    updateRelativePrimaryAcceleration();
    mNeedPrimaryAccelerationUpdate = false;
  }

  return mPrimaryAcceleration;
}

//==============================================================================
Eigen::MatrixXs Joint::finiteDifferenceRelativeJacobian()
{
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> J
      = Eigen::MatrixXs::Zero(6, getNumDofs());
  const s_t EPS = 1e-5;

  for (int i = 0; i < getNumDofs(); i++)
  {
    s_t original = getVelocity(i);
    setVelocity(i, original + EPS);
    Eigen::Vector6s Vplus = getRelativeSpatialVelocity();
    setVelocity(i, original - EPS);
    Eigen::Vector6s Vminus = getRelativeSpatialVelocity();
    setVelocity(i, original);

    J.col(i) = (Vplus - Vminus) / (2 * EPS);
  }

  return J;
}

//==============================================================================
Eigen::MatrixXs Joint::finiteDifferenceRelativeJacobianInPositionSpace(
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRelativeJacobianInPositionSpace();
  }
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> J
      = Eigen::MatrixXs::Zero(6, getNumDofs());
  const s_t EPS = 1e-5;

  Eigen::Isometry3s T = getRelativeTransform();

  for (int i = 0; i < getNumDofs(); i++)
  {
    s_t original = getPosition(i);
    setPosition(i, original + EPS);
    Eigen::Vector6s Tplus = math::logMap(T.inverse() * getRelativeTransform());
    setPosition(i, original - EPS);
    Eigen::Vector6s Tminus = math::logMap(T.inverse() * getRelativeTransform());
    setPosition(i, original);

    J.col(i) = (Tplus - Tminus) / (2 * EPS);
  }

  return J;
}

//==============================================================================
Eigen::MatrixXs Joint::finiteDifferenceRiddersRelativeJacobianInPositionSpace()
{
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> J
      = Eigen::MatrixXs::Zero(6, getNumDofs());

  Eigen::Isometry3s T = getRelativeTransform();

  s_t originalStepSize = 1e-2;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 14;

  for (std::size_t i = 0; i < getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    s_t original = getPosition(i);
    setPosition(i, original + originalStepSize);
    Eigen::Vector6s Tplus = math::logMap(T.inverse() * getRelativeTransform());
    setPosition(i, original - originalStepSize);
    Eigen::Vector6s Tminus = math::logMap(T.inverse() * getRelativeTransform());
    setPosition(i, original);

    tab[0][0] = (Tplus - Tminus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      setPosition(i, original + stepSize);
      Eigen::Vector6s Tplus
          = math::logMap(T.inverse() * getRelativeTransform());
      setPosition(i, original - stepSize);
      Eigen::Vector6s Tminus
          = math::logMap(T.inverse() * getRelativeTransform());
      setPosition(i, original);

      tab[0][iTab] = (Tplus - Tminus) / (2 * stepSize);

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
void Joint::debugRelativeJacobianInPositionSpace()
{
  Eigen::MatrixXs bruteForce
      = finiteDifferenceRelativeJacobianInPositionSpace();
  Eigen::MatrixXs analytical = getRelativeJacobianInPositionSpace();
  const s_t threshold = 1e-9;
  if (((bruteForce - analytical).cwiseAbs().array() > threshold).any())
  {
    std::cout << "Relative Jacobian (in position space) disagrees on joint \""
              << getName() << "\" of type \"" << getType() << "\"!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff (" << (analytical - bruteForce).minCoeff() << ","
              << (analytical - bruteForce).maxCoeff() << "):" << std::endl
              << analytical - bruteForce << std::endl;
  }
}

//==============================================================================
Eigen::Vector6s Joint::getWorldAxisScrewForPosition(int dof) const
{
  assert(dof >= 0 && dof < getNumDofs());
  return math::AdT(
      getChildBodyNode()->getWorldTransform(),
      getRelativeJacobianInPositionSpace().col(dof));
}

//==============================================================================
Eigen::Vector6s Joint::getWorldAxisScrewForVelocity(int dof) const
{
  assert(dof >= 0 && dof < getNumDofs());
  return math::AdT(
      getChildBodyNode()->getWorldTransform(), getRelativeJacobian().col(dof));
}

//==============================================================================
void Joint::setPositionLimitEnforced(bool _isPositionLimitEnforced)
{
  mAspectProperties.mIsPositionLimitEnforced = _isPositionLimitEnforced;
}

//==============================================================================
bool Joint::isPositionLimitEnforced() const
{
  return mAspectProperties.mIsPositionLimitEnforced;
}

//==============================================================================
std::size_t Joint::getJointIndexInSkeleton() const
{
  return mChildBodyNode->getIndexInSkeleton();
}

//==============================================================================
std::size_t Joint::getJointIndexInTree() const
{
  return mChildBodyNode->getIndexInTree();
}

//==============================================================================
std::size_t Joint::getTreeIndex() const
{
  return mChildBodyNode->getTreeIndex();
}

//==============================================================================
bool Joint::checkSanity(bool _printWarnings) const
{
  bool sane = true;
  for (std::size_t i = 0; i < getNumDofs(); ++i)
  {
    if (getInitialPosition(i) < getPositionLowerLimit(i)
        || getPositionUpperLimit(i) < getInitialPosition(i))
    {
      if (_printWarnings)
      {
        dtwarn << "[Joint::checkSanity] Initial position of index " << i << " ["
               << getDofName(i) << "] in Joint [" << getName() << "] is "
               << "outside of its position limits\n"
               << " -- Initial Position: " << getInitialPosition(i) << "\n"
               << " -- Limits: [" << getPositionLowerLimit(i) << ", "
               << getPositionUpperLimit(i) << "]\n";
      }
      else
      {
        return false;
      }

      sane = false;
    }

    if (getInitialVelocity(i) < getVelocityLowerLimit(i)
        || getVelocityUpperLimit(i) < getInitialVelocity(i))
    {
      if (_printWarnings)
      {
        dtwarn << "[Joint::checkSanity] Initial velocity of index " << i << " ["
               << getDofName(i) << "] is Joint [" << getName() << "] is "
               << "outside of its velocity limits\n"
               << " -- Initial Velocity: " << getInitialVelocity(i) << "\n"
               << " -- Limits: [" << getVelocityLowerLimit(i) << ", "
               << getVelocityUpperLimit(i) << "]\n";
      }
      else
      {
        return false;
      }

      sane = false;
    }
  }

  return sane;
}

//==============================================================================
s_t Joint::getPotentialEnergy() const
{
  return computePotentialEnergy();
}

//==============================================================================
void Joint::setTransformFromParentBodyNode(const Eigen::Isometry3s& _T)
{
  assert(math::verifyTransform(_T));
  mAspectProperties.mT_ParentBodyToJoint = _T;
  mAspectProperties.mParentScale = 1.0;
  mAspectProperties.mOriginalParentTranslation = _T.translation();
  notifyPositionUpdated();
}

//==============================================================================
void Joint::setTransformFromChildBodyNode(const Eigen::Isometry3s& _T)
{
  assert(math::verifyTransform(_T));
  mAspectProperties.mT_ChildBodyToJoint = _T;
  mAspectProperties.mChildScale = 1.0;
  mAspectProperties.mOriginalChildTranslation = _T.translation();
  updateRelativeJacobian();
  notifyPositionUpdated();
}

//==============================================================================
const Eigen::Isometry3s& Joint::getTransformFromParentBodyNode() const
{
  return mAspectProperties.mT_ParentBodyToJoint;
}

//==============================================================================
const Eigen::Isometry3s& Joint::getTransformFromChildBodyNode() const
{
  return mAspectProperties.mT_ChildBodyToJoint;
}

//==============================================================================
/// Copy the transfromFromParentNode and transfromFromChildNode, and their
/// scales, from another joint
void Joint::copyTransformsFrom(const dynamics::Joint* other)
{
  mAspectProperties.mChildScale = other->mAspectProperties.mChildScale;
  mAspectProperties.mT_ChildBodyToJoint
      = other->mAspectProperties.mT_ChildBodyToJoint;
  mAspectProperties.mOriginalChildTranslation
      = other->mAspectProperties.mOriginalChildTranslation;
  mAspectProperties.mParentScale = other->mAspectProperties.mParentScale;
  mAspectProperties.mT_ParentBodyToJoint
      = other->mAspectProperties.mT_ParentBodyToJoint;
  mAspectProperties.mOriginalParentTranslation
      = other->mAspectProperties.mOriginalParentTranslation;
}

//==============================================================================
/// Set the scale of the child body
void Joint::setChildScale(s_t scale)
{
  mAspectProperties.mChildScale = scale;
  mAspectProperties.mT_ChildBodyToJoint.translation()
      = mAspectProperties.mOriginalChildTranslation * scale;
  updateRelativeJacobian();
  notifyPositionUpdated();
}

//==============================================================================
/// Set the scale of the parent body
void Joint::setParentScale(s_t scale)
{
  mAspectProperties.mParentScale = scale;
  mAspectProperties.mT_ParentBodyToJoint.translation()
      = mAspectProperties.mOriginalParentTranslation * scale;
  notifyPositionUpdated();
}

//==============================================================================
/// Get the scale of the child body
s_t Joint::getChildScale() const
{
  return mAspectProperties.mChildScale;
}

//==============================================================================
/// Get the scale of the parent body
s_t Joint::getParentScale() const
{
  return mAspectProperties.mParentScale;
}

//==============================================================================
Joint::Joint()
  : mChildBodyNode(nullptr),
    mT(Eigen::Isometry3s::Identity()),
    mSpatialVelocity(Eigen::Vector6s::Zero()),
    mSpatialAcceleration(Eigen::Vector6s::Zero()),
    mPrimaryAcceleration(Eigen::Vector6s::Zero()),
    mNeedTransformUpdate(true),
    mNeedSpatialVelocityUpdate(true),
    mNeedSpatialAccelerationUpdate(true),
    mNeedPrimaryAccelerationUpdate(true),
    mIsRelativeJacobianDirty(true),
    mIsRelativeJacobianInPositionSpaceDirty(true),
    mIsRelativeJacobianTimeDerivDirty(true)
{
  // Do nothing. The Joint::Aspect must be created by a derived class.
}

//==============================================================================
DegreeOfFreedom* Joint::createDofPointer(std::size_t _indexInJoint)
{
  return new DegreeOfFreedom(this, _indexInJoint);
}

//==============================================================================
void Joint::updateLocalTransform() const
{
  updateRelativeTransform();
}

//==============================================================================
void Joint::updateLocalSpatialVelocity() const
{
  updateRelativeSpatialVelocity();
}

//==============================================================================
void Joint::updateLocalSpatialAcceleration() const
{
  updateRelativeSpatialAcceleration();
}

//==============================================================================
void Joint::updateLocalPrimaryAcceleration() const
{
  updateRelativePrimaryAcceleration();
}

//==============================================================================
void Joint::updateLocalJacobian(bool mandatory) const
{
  updateRelativeJacobian(mandatory);
}

//==============================================================================
void Joint::updateLocalJacobianTimeDeriv() const
{
  updateRelativeJacobianTimeDeriv();
}

//==============================================================================
void Joint::updateArticulatedInertia() const
{
  mChildBodyNode->getArticulatedInertia();
}

//==============================================================================
// Eigen::VectorXs Joint::getDampingForces() const
//{
//  int numDofs = getNumDofs();
//  Eigen::VectorXs dampingForce(numDofs);

//  for (int i = 0; i < numDofs; ++i)
//    dampingForce(i) = -mDampingCoefficient[i] * getGenCoord(i)->getVel();

//  return dampingForce;
//}

//==============================================================================
// Eigen::VectorXs Joint::getSpringForces(s_t _timeStep) const
//{
//  int dof = getNumDofs();
//  Eigen::VectorXs springForce(dof);
//  for (int i = 0; i < dof; ++i)
//  {
//    springForce(i) =
//        -mSpringStiffness[i] * (getGenCoord(i)->getPos()
//                                + getGenCoord(i)->getVel() * _timeStep
//                                - mRestPosition[i]);
//  }
//  assert(!math::isNan(springForce));
//  return springForce;
//}

//==============================================================================
void Joint::notifyPositionUpdate()
{
  notifyPositionUpdated();
}

//==============================================================================
void Joint::notifyPositionUpdated()
{
  if (mChildBodyNode)
  {
    mChildBodyNode->dirtyTransform();
    mChildBodyNode->dirtyJacobian();
    mChildBodyNode->dirtyJacobianDeriv();
  }

  mIsRelativeJacobianDirty = true;
  mIsRelativeJacobianInPositionSpaceDirty = true;
  mIsRelativeJacobianTimeDerivDirty = true;
  mNeedPrimaryAccelerationUpdate = true;

  mNeedTransformUpdate = true;
  mNeedSpatialVelocityUpdate = true;
  mNeedSpatialAccelerationUpdate = true;

  SkeletonPtr skel = getSkeleton();
  if (skel)
  {
    std::size_t tree = mChildBodyNode->mTreeIndex;
    skel->dirtyArticulatedInertia(tree);
    skel->mTreeCache[tree].mDirty.mExternalForces = true;
    skel->mSkelCache.mDirty.mExternalForces = true;
  }
}

//==============================================================================
void Joint::notifyVelocityUpdate()
{
  notifyVelocityUpdated();
}

//==============================================================================
void Joint::notifyVelocityUpdated()
{
  if (mChildBodyNode)
  {
    mChildBodyNode->dirtyVelocity();
    mChildBodyNode->dirtyJacobianDeriv();
  }

  mIsRelativeJacobianTimeDerivDirty = true;

  mNeedSpatialVelocityUpdate = true;
  mNeedSpatialAccelerationUpdate = true;
}

//==============================================================================
void Joint::notifyAccelerationUpdate()
{
  notifyAccelerationUpdated();
}

//==============================================================================
void Joint::notifyAccelerationUpdated()
{
  if (mChildBodyNode)
    mChildBodyNode->dirtyAcceleration();

  mNeedSpatialAccelerationUpdate = true;
  mNeedPrimaryAccelerationUpdate = true;
}

} // namespace dynamics
} // namespace dart
