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

#include "dart/dynamics/DegreeOfFreedom.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
const std::string& DegreeOfFreedom::setName(
    const std::string& _name, bool _preserveName)
{
  return mJoint->setDofName(mIndexInJoint, _name, _preserveName);
}

//==============================================================================
const std::string& DegreeOfFreedom::getName() const
{
  return mJoint->getDofName(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::preserveName(bool _preserve)
{
  mJoint->preserveDofName(mIndexInJoint, _preserve);
}

//==============================================================================
bool DegreeOfFreedom::isNamePreserved() const
{
  return mJoint->isDofNamePreserved(mIndexInJoint);
}

//==============================================================================
std::size_t DegreeOfFreedom::getIndexInSkeleton() const
{
  return mIndexInSkeleton;
}

//==============================================================================
std::size_t DegreeOfFreedom::getIndexInTree() const
{
  return mIndexInTree;
}

//==============================================================================
std::size_t DegreeOfFreedom::getIndexInJoint() const
{
  return mIndexInJoint;
}

//==============================================================================
std::size_t DegreeOfFreedom::getTreeIndex() const
{
  return mJoint->getTreeIndex();
}

//==============================================================================
void DegreeOfFreedom::setCommand(s_t _command)
{
  mJoint->setCommand(mIndexInJoint, _command);
}

//==============================================================================
s_t DegreeOfFreedom::getCommand() const
{
  return mJoint->getCommand(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetCommand()
{
  setCommand(0.0);
}

//==============================================================================
void DegreeOfFreedom::setPosition(s_t _position)
{
  mJoint->setPosition(mIndexInJoint, _position);
}

//==============================================================================
s_t DegreeOfFreedom::getPosition() const
{
  return mJoint->getPosition(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setPositionLimits(s_t _lowerLimit, s_t _upperLimit)
{
  setPositionLowerLimit(_lowerLimit);
  setPositionUpperLimit(_upperLimit);
}

//==============================================================================
void DegreeOfFreedom::setPositionLimits(const std::pair<s_t, s_t>& _limits)
{
  setPositionLimits(_limits.first, _limits.second);
}

//==============================================================================
std::pair<s_t, s_t> DegreeOfFreedom::getPositionLimits() const
{
  return std::pair<s_t, s_t>(getPositionLowerLimit(), getPositionUpperLimit());
}

//==============================================================================

void DegreeOfFreedom::setPositionLowerLimit(s_t _limit)
{
  mJoint->setPositionLowerLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getPositionLowerLimit() const
{
  return mJoint->getPositionLowerLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setPositionUpperLimit(s_t _limit)
{
  mJoint->setPositionUpperLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getPositionUpperLimit() const
{
  return mJoint->getPositionUpperLimit(mIndexInJoint);
}

//==============================================================================
bool DegreeOfFreedom::hasPositionLimit() const
{
  return mJoint->hasPositionLimit(mIndexInJoint);
}

//==============================================================================
bool DegreeOfFreedom::isCyclic() const
{
  return mJoint->isCyclic(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetPosition()
{
  mJoint->resetPosition(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setInitialPosition(s_t _initial)
{
  mJoint->setInitialPosition(mIndexInJoint, _initial);
}

//==============================================================================
s_t DegreeOfFreedom::getInitialPosition() const
{
  return mJoint->getInitialPosition(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setVelocity(s_t _velocity)
{
  mJoint->setVelocity(mIndexInJoint, _velocity);
}

//==============================================================================
s_t DegreeOfFreedom::getVelocity() const
{
  return mJoint->getVelocity(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetVelocity()
{
  mJoint->resetVelocity(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setVelocityLimits(s_t _lowerLimit, s_t _upperLimit)
{
  setVelocityLowerLimit(_lowerLimit);
  setVelocityUpperLimit(_upperLimit);
}

//==============================================================================
void DegreeOfFreedom::setVelocityLimits(const std::pair<s_t, s_t>& _limits)
{
  setVelocityLimits(_limits.first, _limits.second);
}

//==============================================================================
std::pair<s_t, s_t> DegreeOfFreedom::getVelocityLimits() const
{
  return std::pair<s_t, s_t>(getVelocityLowerLimit(), getVelocityUpperLimit());
}

//==============================================================================
void DegreeOfFreedom::setVelocityLowerLimit(s_t _limit)
{
  mJoint->setVelocityLowerLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getVelocityLowerLimit() const
{
  return mJoint->getVelocityLowerLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setVelocityUpperLimit(s_t _limit)
{
  mJoint->setVelocityUpperLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getVelocityUpperLimit() const
{
  return mJoint->getVelocityUpperLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setInitialVelocity(s_t _initial)
{
  mJoint->setInitialVelocity(mIndexInJoint, _initial);
}

//==============================================================================
s_t DegreeOfFreedom::getInitialVelocity() const
{
  return mJoint->getInitialVelocity(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setAcceleration(s_t _acceleration)
{
  mJoint->setAcceleration(mIndexInJoint, _acceleration);
}

//==============================================================================
s_t DegreeOfFreedom::getAcceleration() const
{
  return mJoint->getAcceleration(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetAcceleration()
{
  setAcceleration(0.0);
}

//==============================================================================
void DegreeOfFreedom::setAccelerationLimits(s_t _lowerLimit, s_t _upperLimit)
{
  setAccelerationLowerLimit(_lowerLimit);
  setAccelerationUpperLimit(_upperLimit);
}

//==============================================================================
void DegreeOfFreedom::setAccelerationLimits(const std::pair<s_t, s_t>& _limits)
{
  setAccelerationLimits(_limits.first, _limits.second);
}

//==============================================================================
std::pair<s_t, s_t> DegreeOfFreedom::getAccelerationLimits() const
{
  return std::pair<s_t, s_t>(
      getAccelerationLowerLimit(), getAccelerationUpperLimit());
}

//==============================================================================
void DegreeOfFreedom::setAccelerationLowerLimit(s_t _limit)
{
  mJoint->setAccelerationLowerLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getAccelerationLowerLimit() const
{
  return mJoint->getAccelerationLowerLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setAccelerationUpperLimit(s_t _limit)
{
  mJoint->setAccelerationUpperLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getAccelerationUpperLimit() const
{
  return mJoint->getAccelerationUpperLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setControlForce(s_t _force)
{
  mJoint->setControlForce(mIndexInJoint, _force);
}

//==============================================================================
s_t DegreeOfFreedom::getControlForce() const
{
  return mJoint->getControlForce(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetControlForce()
{
  setControlForce(0.0);
}

//==============================================================================
void DegreeOfFreedom::setControlForceLimits(s_t _lowerLimit, s_t _upperLimit)
{
  setControlForceLowerLimit(_lowerLimit);
  setControlForceUpperLimit(_upperLimit);
}

//==============================================================================
void DegreeOfFreedom::setControlForceLimits(const std::pair<s_t, s_t>& _limits)
{
  setControlForceLimits(_limits.first, _limits.second);
}

//==============================================================================
std::pair<s_t, s_t> DegreeOfFreedom::getControlForceLimits() const
{
  return std::pair<s_t, s_t>(
      getControlForceLowerLimit(), getControlForceUpperLimit());
}

//==============================================================================
void DegreeOfFreedom::setControlForceLowerLimit(s_t _limit)
{
  mJoint->setControlForceLowerLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getControlForceLowerLimit() const
{
  return mJoint->getControlForceLowerLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setControlForceUpperLimit(s_t _limit)
{
  mJoint->setControlForceUpperLimit(mIndexInJoint, _limit);
}

//==============================================================================
s_t DegreeOfFreedom::getControlForceUpperLimit() const
{
  return mJoint->getControlForceUpperLimit(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setVelocityChange(s_t _velocityChange)
{
  mJoint->setVelocityChange(mIndexInJoint, _velocityChange);
}

//==============================================================================
s_t DegreeOfFreedom::getVelocityChange() const
{
  return mJoint->getVelocityChange(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetVelocityChange()
{
  setVelocityChange(0.0);
}

//==============================================================================
void DegreeOfFreedom::setConstraintImpulse(s_t _impulse)
{
  mJoint->setConstraintImpulse(mIndexInJoint, _impulse);
}

//==============================================================================
s_t DegreeOfFreedom::getConstraintImpulse() const
{
  return mJoint->getConstraintImpulse(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::resetConstraintImpulse()
{
  setConstraintImpulse(0.0);
}

//==============================================================================
void DegreeOfFreedom::setSpringStiffness(s_t _k)
{
  mJoint->setSpringStiffness(mIndexInJoint, _k);
}

//==============================================================================
s_t DegreeOfFreedom::getSpringStiffness() const
{
  return mJoint->getSpringStiffness(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setRestPosition(s_t _q0)
{
  mJoint->setRestPosition(mIndexInJoint, _q0);
}

//==============================================================================
s_t DegreeOfFreedom::getRestPosition() const
{
  return mJoint->getRestPosition(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setDampingCoefficient(s_t _coeff)
{
  mJoint->setDampingCoefficient(mIndexInJoint, _coeff);
}

//==============================================================================
s_t DegreeOfFreedom::getDampingCoefficient() const
{
  return mJoint->getDampingCoefficient(mIndexInJoint);
}

//==============================================================================
void DegreeOfFreedom::setCoulombFriction(s_t _friction)
{
  mJoint->setCoulombFriction(mIndexInJoint, _friction);
}

//==============================================================================
s_t DegreeOfFreedom::getCoulombFriction() const
{
  return mJoint->getCoulombFriction(mIndexInJoint);
}

//==============================================================================
Joint* DegreeOfFreedom::getJoint()
{
  return mJoint;
}

//==============================================================================
const Joint* DegreeOfFreedom::getJoint() const
{
  return mJoint;
}

//==============================================================================
SkeletonPtr DegreeOfFreedom::getSkeleton()
{
  return mJoint->getSkeleton();
}

//==============================================================================
ConstSkeletonPtr DegreeOfFreedom::getSkeleton() const
{
  return mJoint->getSkeleton();
}

//==============================================================================
BodyNode* DegreeOfFreedom::getChildBodyNode()
{
  return mJoint->getChildBodyNode();
}

//==============================================================================
const BodyNode* DegreeOfFreedom::getChildBodyNode() const
{
  return mJoint->getChildBodyNode();
}

//==============================================================================
BodyNode* DegreeOfFreedom::getParentBodyNode()
{
  return mJoint->getParentBodyNode();
}

//==============================================================================
const BodyNode* DegreeOfFreedom::getParentBodyNode() const
{
  return mJoint->getParentBodyNode();
}

//==============================================================================
bool DegreeOfFreedom::isParentOf(const DegreeOfFreedom* target) const
{
  const dynamics::Joint* parentJoint = getJoint();
  const dynamics::Joint* childJoint = target->getJoint();
  if (parentJoint == childJoint)
  {
    // For multi-DOF joints, each axis affects all the others.
    return target->getIndexInJoint() != getIndexInJoint();
  }
  // If these joints aren't in the same skeleton, or aren't in the same tree
  // within that skeleton, this is trivially false
  if (parentJoint->getSkeleton()->getName()
          != childJoint->getSkeleton()->getName()
      || parentJoint->getTreeIndex() != childJoint->getTreeIndex())
    return false;
  // If the dof joint is after the node parent joint in the skeleton, this is
  // also false
  if (parentJoint->getIndexInTree(0) > childJoint->getIndexInTree(0))
    return false;
  // Now this may be true, if the node is a direct child of the dof
  while (true)
  {
    if (parentJoint == childJoint)
      return true;
    if (childJoint->getParentBodyNode() == nullptr
        || childJoint->getParentBodyNode()->getParentJoint() == nullptr)
      return false;
    childJoint = childJoint->getParentBodyNode()->getParentJoint();
  }
}

//==============================================================================
/// This uses the cached version, stored on the parent Skeleton, to return the
/// same value as isParentOf()
bool DegreeOfFreedom::isParentOfFast(const DegreeOfFreedom* target) const
{
  const dynamics::Joint* parentJoint = getJoint();
  const dynamics::Joint* childJoint = target->getJoint();
  dynamics::Skeleton* parentSkel
      = const_cast<dynamics::Skeleton*>(getSkeleton().get());
  if (parentJoint == childJoint)
  {
    // For multi-DOF joints, each axis affects all the others.
    return target->getIndexInJoint() != getIndexInJoint();
  }
  // If these joints aren't in the same skeleton, or aren't in the same tree
  // within that skeleton, this is trivially false
  if (parentSkel->getName() != childJoint->getSkeleton()->getName()
      || parentJoint->getTreeIndex() != childJoint->getTreeIndex())
    return false;

  bool result = parentSkel->getDofParentMap()(
                    getIndexInSkeleton(), target->getIndexInSkeleton())
                == 1;
#ifndef NDEBUG
  bool slowResult = isParentOf(target);
  assert(result == slowResult);
#endif
  return result;
}

//==============================================================================
bool DegreeOfFreedom::isParentOf(const BodyNode* target) const
{
  const dynamics::Joint* dofJoint = getJoint();
  const dynamics::Joint* nodeParentJoint = target->getParentJoint();

  // If our immediate parent is a weld joint, keep walking up the tree until we
  // find a normal joint. If there are none, then return false.
  while (nodeParentJoint->getNumDofs() == 0)
  {
    if (nodeParentJoint->getParentBodyNode() != nullptr
        && nodeParentJoint->getParentBodyNode()->getParentJoint() != nullptr)
    {
      nodeParentJoint = nodeParentJoint->getParentBodyNode()->getParentJoint();
    }
    else
    {
      return false;
    }
  }
  // Edge cases
  if (nodeParentJoint == nullptr || dofJoint->getSkeleton() == nullptr
      || nodeParentJoint->getSkeleton() == nullptr
      || dofJoint->getNumDofs() == 0)
  {
    return false;
  }
  // If these joints aren't in the same skeleton, or aren't in the same tree
  // within that skeleton, this is trivially false
  if (dofJoint->getSkeleton()->getName()
          != nodeParentJoint->getSkeleton()->getName()
      || dofJoint->getTreeIndex() != nodeParentJoint->getTreeIndex())
    return false;
  // If the dof joint is after the node parent joint in the skeleton, this is
  // also false
  if (dofJoint->getIndexInTree(0) > nodeParentJoint->getIndexInTree(0))
    return false;
  // Now this may be true, if the node is a direct child of the dof
  while (true)
  {
    if (nodeParentJoint->getName() == dofJoint->getName())
      return true;
    if (nodeParentJoint->getParentBodyNode() == nullptr
        || nodeParentJoint->getParentBodyNode()->getParentJoint() == nullptr)
      return false;
    nodeParentJoint = nodeParentJoint->getParentBodyNode()->getParentJoint();
  }
}

//==============================================================================
/// This uses the cached version, stored on the parent Skeleton, to return the
/// same value as isParentOf()
bool DegreeOfFreedom::isParentOfFast(const BodyNode* target) const
{
  const dynamics::Joint* dofJoint = getJoint();
  const dynamics::Joint* nodeParentJoint = target->getParentJoint();

  // If our immediate parent is a weld joint, keep walking up the tree until we
  // find a normal joint. If there are none, then return false.
  while (nodeParentJoint->getNumDofs() == 0)
  {
    if (nodeParentJoint->getParentBodyNode() != nullptr
        && nodeParentJoint->getParentBodyNode()->getParentJoint() != nullptr)
    {
      nodeParentJoint = nodeParentJoint->getParentBodyNode()->getParentJoint();
    }
    else
    {
      return false;
    }
  }
  // Edge cases
  if (nodeParentJoint == nullptr || dofJoint->getSkeleton() == nullptr
      || nodeParentJoint->getSkeleton() == nullptr
      || dofJoint->getNumDofs() == 0)
  {
    return false;
  }
  // If these joints aren't in the same skeleton, or aren't in the same tree
  // within that skeleton, this is trivially false
  if (dofJoint->getSkeleton()->getName()
          != nodeParentJoint->getSkeleton()->getName()
      || dofJoint->getTreeIndex() != nodeParentJoint->getTreeIndex())
    return false;
  if (nodeParentJoint->getName() == dofJoint->getName())
    return true;

  bool result
      = const_cast<dynamics::Skeleton*>(nodeParentJoint->getSkeleton().get())
            ->getDofParentMap()(
                getIndexInSkeleton(), nodeParentJoint->getIndexInSkeleton(0))
        == 1;

#ifndef NDEBUG
  bool slowResult = isParentOf(target);
  if (result != slowResult)
  {
    Eigen::MatrixXi parentMap
        = const_cast<dynamics::Skeleton*>(nodeParentJoint->getSkeleton().get())
              ->getDofParentMap();
    int myIndex = getIndexInSkeleton();
    int childIndex = nodeParentJoint->getIndexInSkeleton(0);
    std::cout << "Parent map: " << std::endl << parentMap << std::endl;
    std::cout << "My index: " << myIndex << std::endl;
    std::cout << "Child index: " << childIndex << std::endl;
    std::cout << "Slow result: " << slowResult << std::endl;
    std::cout << "Fast result: " << result << std::endl;
    slowResult = isParentOf(target);
  }
  assert(result == slowResult);
#endif
  return result;
}

//==============================================================================
DegreeOfFreedom::DegreeOfFreedom(Joint* _joint, std::size_t _indexInJoint)
  : mIndexInJoint(_indexInJoint),
    mIndexInSkeleton(0),
    mIndexInTree(0),
    mJoint(_joint)
{
  // Do nothing
}

} // namespace dynamics
} // namespace dart
