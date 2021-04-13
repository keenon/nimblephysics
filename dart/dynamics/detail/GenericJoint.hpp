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

#ifndef DART_DYNAMICS_DETAIL_GenericJoint_HPP_
#define DART_DYNAMICS_DETAIL_GenericJoint_HPP_

#include "dart/config.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/GenericJoint.hpp"
#include "dart/math/ConfigurationSpace.hpp"
#include "dart/math/Helpers.hpp"

#define GenericJoint_REPORT_DIM_MISMATCH(func, arg)                            \
  {                                                                            \
    dterr << "[GenericJoint::" #func "] Mismatch beteween size of "            \
          << #arg " [" << arg.size() << "] and the number of "                 \
          << "DOFs [" << getNumDofs() << "] for Joint named ["                 \
          << this->getName() << "].\n";                                        \
    assert(false);                                                             \
  }

#define GenericJoint_REPORT_OUT_OF_RANGE(func, index)                          \
  {                                                                            \
    dterr << "[GenericJoint::" << #func << "] The index [" << index            \
          << "] is out of range for Joint named [" << this->getName()          \
          << "] which has " << this->getNumDofs() << " DOFs.\n";               \
    assert(false);                                                             \
  }

#define GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(func)                         \
  {                                                                            \
    dterr << "[GenericJoint::" #func "] Unsupported actuator type ("           \
          << Joint::mAspectProperties.mActuatorType << ") for Joint ["         \
          << this->getName() << "].\n";                                        \
    assert(false);                                                             \
  }

#define GenericJoint_SET_IF_DIFFERENT(mField, value)                           \
  if (value == Base::mAspectProperties.mField)                                 \
    return;                                                                    \
  Base::mAspectProperties.mField = value;                                      \
  Joint::incrementVersion();

namespace dart {
namespace dynamics {

//==============================================================================
//
// These namespace-level definitions are required to enable ODR-use of static
// constexpr member variables.
//
// See this StackOverflow answer: http://stackoverflow.com/a/14396189/111426
//
template <class ConfigSpaceT>
constexpr size_t GenericJoint<ConfigSpaceT>::NumDofs;

//==============================================================================
template <class ConfigSpaceT>
GenericJoint<ConfigSpaceT>::~GenericJoint()
{
  for (auto i = 0u; i < NumDofs; ++i)
    delete mDofs[i];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setProperties(const Properties& properties)
{
  Joint::setProperties(static_cast<const Joint::Properties&>(properties));
  setProperties(static_cast<const UniqueProperties&>(properties));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setProperties(
    const UniqueProperties& properties)
{
  setAspectProperties(properties);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAspectState(const AspectState& state)
{
  setCommands(state.mCommands);
  setPositionsStatic(state.mPositions);
  setVelocitiesStatic(state.mVelocities);
  setAccelerationsStatic(state.mAccelerations);
  setControlForces(state.mForces);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAspectProperties(
    const AspectProperties& properties)
{
  for (auto i = 0u; i < NumDofs; ++i)
  {
    setDofName(i, properties.mDofNames[i], properties.mPreserveDofNames[i]);
    setPositionLowerLimit(i, properties.mPositionLowerLimits[i]);
    setPositionUpperLimit(i, properties.mPositionUpperLimits[i]);
    setInitialPosition(i, properties.mInitialPositions[i]);
    setVelocityLowerLimit(i, properties.mVelocityLowerLimits[i]);
    setVelocityUpperLimit(i, properties.mVelocityUpperLimits[i]);
    setInitialVelocity(i, properties.mInitialVelocities[i]);
    setAccelerationLowerLimit(i, properties.mAccelerationLowerLimits[i]);
    setAccelerationUpperLimit(i, properties.mAccelerationUpperLimits[i]);
    setControlForceLowerLimit(i, properties.mForceLowerLimits[i]);
    setControlForceUpperLimit(i, properties.mForceUpperLimits[i]);
    setSpringStiffness(i, properties.mSpringStiffnesses[i]);
    setRestPosition(i, properties.mRestPositions[i]);
    setDampingCoefficient(i, properties.mDampingCoefficients[i]);
    setCoulombFriction(i, properties.mFrictions[i]);
  }
}

//==============================================================================
template <class ConfigSpaceT>
typename GenericJoint<ConfigSpaceT>::Properties
GenericJoint<ConfigSpaceT>::getGenericJointProperties() const
{
  return GenericJoint<ConfigSpaceT>::Properties(
      Joint::mAspectProperties, Base::mAspectProperties);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::copy(const GenericJoint<ConfigSpaceT>& other)
{
  if (this == &other)
    return;

  setProperties(other.getGenericJointProperties());
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::copy(const GenericJoint<ConfigSpaceT>* other)
{
  if (nullptr == other)
    return;

  copy(*other);
}

//==============================================================================
template <class ConfigSpaceT>
GenericJoint<ConfigSpaceT>& GenericJoint<ConfigSpaceT>::operator=(
    const GenericJoint<ConfigSpaceT>& other)
{
  copy(other);
  return *this;
}

//==============================================================================
template <class ConfigSpaceT>
bool GenericJoint<ConfigSpaceT>::hasDof(const DegreeOfFreedom* dof) const
{
  for (const DegreeOfFreedom* internalDof : mDofs)
  {
    if (dof == internalDof)
    {
      return true;
    }
  }

  return false;
}

//==============================================================================
template <class ConfigSpaceT>
DegreeOfFreedom* GenericJoint<ConfigSpaceT>::getDof(size_t index)
{
  if (index < NumDofs)
    return mDofs[index];

  GenericJoint_REPORT_OUT_OF_RANGE(getDof, index);

  return nullptr;
}

//==============================================================================
template <class ConfigSpaceT>
const DegreeOfFreedom* GenericJoint<ConfigSpaceT>::getDof(size_t index) const
{
  if (index < NumDofs)
    return mDofs[index];

  GenericJoint_REPORT_OUT_OF_RANGE(getDof, index);

  return nullptr;
}

//==============================================================================
template <class ConfigSpaceT>
size_t GenericJoint<ConfigSpaceT>::getNumDofs() const
{
  return NumDofs;
}

//==============================================================================
template <class ConfigSpaceT>
const std::string& GenericJoint<ConfigSpaceT>::setDofName(
    size_t index, const std::string& name, bool preserveName)
{
  if (NumDofs <= index)
  {
    dterr << "[GenericJoint::setDofName] Attempting to set the name of DOF "
          << "index " << index << ", which is out of bounds for the Joint ["
          << this->getName()
          << "]. We will set the name of DOF index 0 instead.\n";
    assert(false);
    index = 0u;
  }

  preserveDofName(index, preserveName);

  std::string& dofName = Base::mAspectProperties.mDofNames[index];

  if (name == dofName)
    return dofName;

  const SkeletonPtr& skel
      = this->mChildBodyNode ? this->mChildBodyNode->getSkeleton() : nullptr;
  if (skel)
    dofName = skel->mNameMgrForDofs.changeObjectName(mDofs[index], name);
  else
    dofName = name;

  return dofName;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::preserveDofName(size_t index, bool preserve)
{
  if (NumDofs <= index)
  {
    GenericJoint_REPORT_OUT_OF_RANGE(preserveDofName, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mPreserveDofNames[index], preserve);
}

//==============================================================================
template <class ConfigSpaceT>
bool GenericJoint<ConfigSpaceT>::isDofNamePreserved(size_t index) const
{
  if (NumDofs <= index)
  {
    GenericJoint_REPORT_OUT_OF_RANGE(isDofNamePreserved, index);
    index = 0;
  }

  return Base::mAspectProperties.mPreserveDofNames[index];
}

//==============================================================================
template <class ConfigSpaceT>
const std::string& GenericJoint<ConfigSpaceT>::getDofName(size_t index) const
{
  if (NumDofs <= index)
  {
    dterr << "[GenericJoint::getDofName] Requested name of DOF index [" << index
          << "] in Joint [" << this->getName() << "], but that is "
          << "out of bounds (max " << NumDofs - 1
          << "). Returning name of DOF 0.\n";
    assert(false);
    return Base::mAspectProperties.mDofNames[0];
  }

  return Base::mAspectProperties.mDofNames[index];
}

//==============================================================================
template <class ConfigSpaceT>
size_t GenericJoint<ConfigSpaceT>::getIndexInSkeleton(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getIndexInSkeleton, index);
    return 0;
  }

  return mDofs[index]->mIndexInSkeleton;
}

//==============================================================================
template <class ConfigSpaceT>
size_t GenericJoint<ConfigSpaceT>::getIndexInTree(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getIndexInTree, index);
    return 0;
  }

  return mDofs[index]->mIndexInTree;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setCommand(size_t index, s_t command)
{
  if (index >= getNumDofs())
    GenericJoint_REPORT_OUT_OF_RANGE(setCommand, index);

  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
      this->mAspectState.mCommands[index] = math::clip(
          command,
          Base::mAspectProperties.mForceLowerLimits[index],
          Base::mAspectProperties.mForceUpperLimits[index]);
      break;
    case Joint::PASSIVE:
      if (0.0 != command)
      {
        dtwarn << "[GenericJoint::setCommand] Attempting to set a non-zero ("
               << command << ") command for a PASSIVE joint ["
               << this->getName() << "].\n";
      }
      this->mAspectState.mCommands[index] = command;
      break;
    case Joint::SERVO:
      this->mAspectState.mCommands[index] = math::clip(
          command,
          Base::mAspectProperties.mVelocityLowerLimits[index],
          Base::mAspectProperties.mVelocityUpperLimits[index]);
      break;
    case Joint::MIMIC:
      if (0.0 != command)
      {
        dtwarn << "[GenericJoint::setCommand] Attempting to set a non-zero ("
               << command << ") command for a MIMIC joint [" << this->getName()
               << "].\n";
      }
      this->mAspectState.mCommands[index] = math::clip(
          command,
          Base::mAspectProperties.mVelocityLowerLimits[index],
          Base::mAspectProperties.mVelocityUpperLimits[index]);
      break;
    case Joint::ACCELERATION:
      this->mAspectState.mCommands[index] = math::clip(
          command,
          Base::mAspectProperties.mAccelerationLowerLimits[index],
          Base::mAspectProperties.mAccelerationUpperLimits[index]);
      break;
    case Joint::VELOCITY:
      this->mAspectState.mCommands[index] = math::clip(
          command,
          Base::mAspectProperties.mVelocityLowerLimits[index],
          Base::mAspectProperties.mVelocityUpperLimits[index]);
      // TODO: This possibly makes the acceleration to exceed the limits.
      break;
    case Joint::LOCKED:
      if (0.0 != command)
      {
        dtwarn << "[GenericJoint::setCommand] Attempting to set a non-zero ("
               << command << ") command for a LOCKED joint [" << this->getName()
               << "].\n";
      }
      this->mAspectState.mCommands[index] = command;
      break;
    default:
      assert(false);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getCommand(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getCommand, index);
    return 0.0;
  }

  return this->mAspectState.mCommands[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setCommands(const Eigen::VectorXs& commands)
{
  if (static_cast<size_t>(commands.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setCommands, commands);
    return;
  }

  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
      this->mAspectState.mCommands = math::clip(
          commands,
          Base::mAspectProperties.mForceLowerLimits,
          Base::mAspectProperties.mForceUpperLimits);
      break;
    case Joint::PASSIVE:
      if (Vector::Zero() != commands)
      {
        dtwarn << "[GenericJoint::setCommands] Attempting to set a non-zero ("
               << commands.transpose() << ") command for a PASSIVE joint ["
               << this->getName() << "].\n";
      }
      this->mAspectState.mCommands = commands;
      break;
    case Joint::SERVO:
      this->mAspectState.mCommands = math::clip(
          commands,
          Base::mAspectProperties.mVelocityLowerLimits,
          Base::mAspectProperties.mVelocityUpperLimits);
      break;
    case Joint::MIMIC:
      if (Vector::Zero() != commands)
      {
        dtwarn << "[GenericJoint::setCommands] Attempting to set a non-zero ("
               << commands.transpose() << ") command for a MIMIC joint ["
               << this->getName() << "].\n";
      }
      this->mAspectState.mCommands = math::clip(
          commands,
          Base::mAspectProperties.mVelocityLowerLimits,
          Base::mAspectProperties.mVelocityUpperLimits);
      break;
    case Joint::ACCELERATION:
      this->mAspectState.mCommands = math::clip(
          commands,
          Base::mAspectProperties.mAccelerationLowerLimits,
          Base::mAspectProperties.mAccelerationUpperLimits);
      break;
    case Joint::VELOCITY:
      this->mAspectState.mCommands = math::clip(
          commands,
          Base::mAspectProperties.mVelocityLowerLimits,
          Base::mAspectProperties.mVelocityUpperLimits);
      // TODO: This possibly makes the acceleration to exceed the limits.
      break;
    case Joint::LOCKED:
      if (Vector::Zero() != commands)
      {
        dtwarn << "[GenericJoint::setCommands] Attempting to set a non-zero ("
               << commands.transpose() << ") command for a LOCKED joint ["
               << this->getName() << "].\n";
      }
      this->mAspectState.mCommands = commands;
      break;
    default:
      assert(false);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getCommands() const
{
  return this->mAspectState.mCommands;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetCommands()
{
  this->mAspectState.mCommands.setZero();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPosition(size_t index, s_t position)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setPosition, index);
    return;
  }

  if (this->mAspectState.mPositions[index] == position)
    return;
  // TODO(JS): Above code should be changed something like:
  //  if (ConfigSpaceT::getEuclideanPoint(mPositions, index) == position)
  //    return;

  // Note: It would not make much sense to use setPositionsStatic() here
  this->mAspectState.mPositions[index] = position;
  this->notifyPositionUpdated();
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getPosition(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getPosition, index);
    return 0.0;
  }

  return getPositionsStatic()[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositions(const Eigen::VectorXs& positions)
{
  if (static_cast<size_t>(positions.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setPositions, positions);
    return;
  }

  setPositionsStatic(positions);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getPositions() const
{
  return getPositionsStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositionLowerLimit(
    size_t index, s_t position)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setPositionLowerLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mPositionLowerLimits[index], position);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getPositionLowerLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getPositionLowerLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mPositionLowerLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositionLowerLimits(
    const Eigen::VectorXs& lowerLimits)
{
  if (static_cast<size_t>(lowerLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setPositionLowerLimits, lowerLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mPositionLowerLimits, lowerLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getPositionLowerLimits() const
{
  return Base::mAspectProperties.mPositionLowerLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositionUpperLimit(
    size_t index, s_t position)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setPositionUpperLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mPositionUpperLimits[index], position);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getPositionUpperLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getPositionUpperLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mPositionUpperLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositionUpperLimits(
    const Eigen::VectorXs& upperLimits)
{
  if (static_cast<size_t>(upperLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setPositionUpperLimits, upperLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mPositionUpperLimits, upperLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getPositionUpperLimits() const
{
  return Base::mAspectProperties.mPositionUpperLimits;
}

//==============================================================================
template <class ConfigSpaceT>
bool GenericJoint<ConfigSpaceT>::hasPositionLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(hasPositionLimit, index);
    return true;
  }

  return isfinite(Base::mAspectProperties.mPositionUpperLimits[index])
         || isfinite(Base::mAspectProperties.mPositionLowerLimits[index]);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetPosition(size_t index)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(resetPosition, index);
    return;
  }

  setPosition(index, Base::mAspectProperties.mInitialPositions[index]);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetPositions()
{
  setPositionsStatic(Base::mAspectProperties.mInitialPositions);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setInitialPosition(size_t index, s_t initial)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setInitialPosition, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mInitialPositions[index], initial);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getInitialPosition(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getInitialPosition, index);
    return 0.0;
  }

  return Base::mAspectProperties.mInitialPositions[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setInitialPositions(
    const Eigen::VectorXs& initial)
{
  if (static_cast<size_t>(initial.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setInitialPositions, initial);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mInitialPositions, initial);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getInitialPositions() const
{
  return Base::mAspectProperties.mInitialPositions;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPositionsStatic(const Vector& positions)
{
  if (this->mAspectState.mPositions == positions)
    return;

  this->mAspectState.mPositions = positions;
  this->notifyPositionUpdated();
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::Vector&
GenericJoint<ConfigSpaceT>::getPositionsStatic() const
{
  return this->mAspectState.mPositions;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocitiesStatic(const Vector& velocities)
{
  if (this->mAspectState.mVelocities == velocities)
    return;

  this->mAspectState.mVelocities = velocities;
  this->notifyVelocityUpdated();
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::Vector&
GenericJoint<ConfigSpaceT>::getVelocitiesStatic() const
{
  return this->mAspectState.mVelocities;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerationsStatic(const Vector& accels)
{
  if (this->mAspectState.mAccelerations == accels)
    return;

  this->mAspectState.mAccelerations = accels;
  this->notifyAccelerationUpdated();
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::Vector&
GenericJoint<ConfigSpaceT>::getAccelerationsStatic() const
{
  return this->mAspectState.mAccelerations;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocity(size_t index, s_t velocity)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setVelocity, index);
    return;
  }

  if (this->mAspectState.mVelocities[index] == velocity)
    return;

  // Note: It would not make much sense to use setVelocitiesStatic() here
  this->mAspectState.mVelocities[index] = velocity;
  this->notifyVelocityUpdated();

  if (Joint::mAspectProperties.mActuatorType == Joint::VELOCITY)
    this->mAspectState.mCommands[index] = this->getVelocitiesStatic()[index];
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getVelocity(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getVelocity, index);
    return 0.0;
  }

  return getVelocitiesStatic()[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocities(
    const Eigen::VectorXs& velocities)
{
  if (static_cast<size_t>(velocities.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setVelocities, velocities);
    return;
  }

  setVelocitiesStatic(velocities);

  if (Joint::mAspectProperties.mActuatorType == Joint::VELOCITY)
    this->mAspectState.mCommands = this->getVelocitiesStatic();
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getVelocities() const
{
  return getVelocitiesStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocityLowerLimit(
    size_t index, s_t velocity)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setVelocityLowerLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mVelocityLowerLimits[index], velocity);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getVelocityLowerLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getVelocityLowerLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mVelocityLowerLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocityLowerLimits(
    const Eigen::VectorXs& lowerLimits)
{
  if (static_cast<size_t>(lowerLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setVelocityLowerLimits, lowerLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mVelocityLowerLimits, lowerLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getVelocityLowerLimits() const
{
  return Base::mAspectProperties.mVelocityLowerLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocityUpperLimit(
    size_t index, s_t velocity)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setVelocityUpperLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mVelocityUpperLimits[index], velocity);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getVelocityUpperLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getVelocityUpperLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mVelocityUpperLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocityUpperLimits(
    const Eigen::VectorXs& upperLimits)
{
  if (static_cast<size_t>(upperLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setVelocityUpperLimits, upperLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mVelocityUpperLimits, upperLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getVelocityUpperLimits() const
{
  return Base::mAspectProperties.mVelocityUpperLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetVelocity(size_t index)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(resetVelocity, index);
    return;
  }

  setVelocity(index, Base::mAspectProperties.mInitialVelocities[index]);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetVelocities()
{
  setVelocitiesStatic(Base::mAspectProperties.mInitialVelocities);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setInitialVelocity(size_t index, s_t initial)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setInitialVelocity, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mInitialVelocities[index], initial);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getInitialVelocity(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getInitialVelocity, index);
    return 0.0;
  }

  return Base::mAspectProperties.mInitialVelocities[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setInitialVelocities(
    const Eigen::VectorXs& initial)
{
  if (static_cast<size_t>(initial.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setInitialVelocities, initial);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mInitialVelocities, initial);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getInitialVelocities() const
{
  return Base::mAspectProperties.mInitialVelocities;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAcceleration(size_t index, s_t acceleration)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setAcceleration, index);
    return;
  }

  if (this->mAspectState.mAccelerations[index] == acceleration)
    return;

  // Note: It would not make much sense to use setAccelerationsStatic() here
  this->mAspectState.mAccelerations[index] = acceleration;
  this->notifyAccelerationUpdated();

  if (Joint::mAspectProperties.mActuatorType == Joint::ACCELERATION)
    this->mAspectState.mCommands[index] = this->getAccelerationsStatic()[index];
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getAcceleration(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getAcceleration, index);
    return 0.0;
  }

  return getAccelerationsStatic()[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerations(
    const Eigen::VectorXs& accelerations)
{
  if (static_cast<size_t>(accelerations.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setAccelerations, accelerations);
    return;
  }

  setAccelerationsStatic(accelerations);

  if (Joint::mAspectProperties.mActuatorType == Joint::ACCELERATION)
    this->mAspectState.mCommands = this->getAccelerationsStatic();
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getAccelerations() const
{
  return getAccelerationsStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerationLowerLimit(
    size_t index, s_t acceleration)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setAccelerationLowerLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mAccelerationLowerLimits[index], acceleration);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getAccelerationLowerLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getAccelerationLowerLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mAccelerationLowerLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerationLowerLimits(
    const Eigen::VectorXs& lowerLimits)
{
  if (static_cast<size_t>(lowerLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setAccelerationLowerLimits, lowerLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mAccelerationLowerLimits, lowerLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getAccelerationLowerLimits() const
{
  return Base::mAspectProperties.mAccelerationLowerLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerationUpperLimit(
    size_t index, s_t acceleration)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setAccelerationUpperLimit, index) return;
  }

  GenericJoint_SET_IF_DIFFERENT(mAccelerationUpperLimits[index], acceleration);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getAccelerationUpperLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getAccelerationUpperLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mAccelerationUpperLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setAccelerationUpperLimits(
    const Eigen::VectorXs& upperLimits)
{
  if (static_cast<size_t>(upperLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setAccelerationUpperLimits, upperLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mAccelerationUpperLimits, upperLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getAccelerationUpperLimits() const
{
  return Base::mAspectProperties.mAccelerationUpperLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetAccelerations()
{
  setAccelerationsStatic(Vector::Zero());
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForce(size_t index, s_t force)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setControlForce, index);
    return;
  }

  this->mAspectState.mForces[index] = force;

  if (Joint::mAspectProperties.mActuatorType == Joint::FORCE)
    this->mAspectState.mCommands[index] = this->mAspectState.mForces[index];
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getControlForce(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getControlForce, index);
    return 0.0;
  }

  return this->mAspectState.mForces[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForces(const Eigen::VectorXs& forces)
{
  if (static_cast<size_t>(forces.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setControlForces, forces);
    return;
  }

  this->mAspectState.mForces = forces;

  if (Joint::mAspectProperties.mActuatorType == Joint::FORCE)
    this->mAspectState.mCommands = this->mAspectState.mForces;
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getControlForces() const
{
  return this->mAspectState.mForces;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForceLowerLimit(size_t index, s_t force)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setControlForceLowerLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mForceLowerLimits[index], force);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getControlForceLowerLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getControlForceLowerLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mForceLowerLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForceLowerLimits(
    const Eigen::VectorXs& lowerLimits)
{
  if (static_cast<size_t>(lowerLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setControlForceLowerLimits, lowerLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mForceLowerLimits, lowerLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getControlForceLowerLimits() const
{
  return Base::mAspectProperties.mForceLowerLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForceUpperLimit(size_t index, s_t force)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setControlForceUpperLimit, index);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mForceUpperLimits[index], force);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getControlForceUpperLimit(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getControlForceUpperLimit, index);
    return 0.0;
  }

  return Base::mAspectProperties.mForceUpperLimits[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setControlForceUpperLimits(
    const Eigen::VectorXs& upperLimits)
{
  if (static_cast<size_t>(upperLimits.size()) != getNumDofs())
  {
    GenericJoint_REPORT_DIM_MISMATCH(setControlForceUpperLimits, upperLimits);
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mForceUpperLimits, upperLimits);
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getControlForceUpperLimits() const
{
  return Base::mAspectProperties.mForceUpperLimits;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetControlForces()
{
  this->mAspectState.mForces.setZero();

  if (Joint::mAspectProperties.mActuatorType == Joint::FORCE)
    this->mAspectState.mCommands = this->mAspectState.mForces;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setVelocityChange(
    size_t index, s_t velocityChange)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setVelocityChange, index);
    return;
  }

  mVelocityChanges[index] = velocityChange;
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getVelocityChange(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getVelocityChange, index);
    return 0.0;
  }

  return mVelocityChanges[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetVelocityChanges()
{
  mVelocityChanges.setZero();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setConstraintImpulse(size_t index, s_t impulse)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setConstraintImpulse, index);
    return;
  }

  mConstraintImpulses[index] = impulse;
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getConstraintImpulse(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getConstraintImpulse, index);
    return 0.0;
  }

  return mConstraintImpulses[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetConstraintImpulses()
{
  mConstraintImpulses.setZero();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::integratePositions(s_t dt)
{
  const Point& point = math::integratePosition<ConfigSpaceT>(
      math::toManifoldPoint<ConfigSpaceT>(getPositionsStatic()),
      getVelocitiesStatic(),
      dt);

  setPositionsStatic(math::toEuclideanPoint<ConfigSpaceT>(point));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::integrateVelocities(s_t dt)
{
  setVelocitiesStatic(math::integrateVelocity<ConfigSpaceT>(
      getVelocitiesStatic(), getAccelerationsStatic(), dt));
}

//==============================================================================
// Documentation inherited
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::integratePositionsExplicit(
    const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t dt)
{
  const Point& point = math::integratePosition<ConfigSpaceT>(
      math::toManifoldPoint<ConfigSpaceT>(pos), vel, dt);

  return math::toEuclideanPoint<ConfigSpaceT>(point);
}

//==============================================================================
/// Returns d/dpos of integratePositionsExplicit()
template <class ConfigSpaceT>
Eigen::MatrixXs GenericJoint<ConfigSpaceT>::getPosPosJacobian(
    const Eigen::VectorXs& pos, const Eigen::VectorXs& /* vel */, s_t /* _dt */)
{
  return Eigen::MatrixXs::Identity(pos.size(), pos.size());
}

//==============================================================================
/// Returns d/dvel of integratePositionsExplicit()
template <class ConfigSpaceT>
Eigen::MatrixXs GenericJoint<ConfigSpaceT>::getVelPosJacobian(
    const Eigen::VectorXs& pos, const Eigen::VectorXs& /* vel */, s_t _dt)
{
  return _dt * Eigen::MatrixXs::Identity(pos.size(), pos.size());
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getPositionDifferences(
    const Eigen::VectorXs& q2, const Eigen::VectorXs& q1) const
{
  if (static_cast<size_t>(q1.size()) != getNumDofs()
      || static_cast<size_t>(q2.size()) != getNumDofs())
  {
    dterr << "[GenericJoint::getPositionsDifference] q1's size [" << q1.size()
          << "] or q2's size [" << q2.size() << "] must both equal the dof ["
          << this->getNumDofs() << "] for Joint [" << this->getName() << "].\n";
    assert(false);
    return Eigen::VectorXs::Zero(getNumDofs());
  }

  return getPositionDifferencesStatic(q2, q1);
}

//==============================================================================
template <class ConfigSpaceT>
typename ConfigSpaceT::Vector
GenericJoint<ConfigSpaceT>::getPositionDifferencesStatic(
    const Vector& q2, const Vector& q1) const
{
  return q2 - q1;
  // TODO(JS): Move this implementation to each configuration space classes.
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setSpringStiffness(size_t index, s_t k)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setSpringStiffness, index);
    return;
  }

  assert(k >= 0.0);

  GenericJoint_SET_IF_DIFFERENT(mSpringStiffnesses[index], k);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getSpringStiffness(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getSpringStiffness, index);
    return 0.0;
  }

  return Base::mAspectProperties.mSpringStiffnesses[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setRestPosition(size_t index, s_t q0)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setRestPosition, index);
    return;
  }

  if (Base::mAspectProperties.mPositionLowerLimits[index] > q0
      || Base::mAspectProperties.mPositionUpperLimits[index] < q0)
  {
    dtwarn << "[GenericJoint::setRestPosition] Value of _q0 [" << q0
           << "], is out of the limit range ["
           << Base::mAspectProperties.mPositionLowerLimits[index] << ", "
           << Base::mAspectProperties.mPositionUpperLimits[index]
           << "] for index [" << index << "] of Joint [" << this->getName()
           << "].\n";
    return;
  }

  GenericJoint_SET_IF_DIFFERENT(mRestPositions[index], q0);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getRestPosition(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getRestPosition, index);
    return 0.0;
  }

  return Base::mAspectProperties.mRestPositions[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setDampingCoefficient(size_t index, s_t d)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setDampingCoefficient, index);
    return;
  }

  assert(d >= 0.0);

  GenericJoint_SET_IF_DIFFERENT(mDampingCoefficients[index], d);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getDampingCoefficient(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getDampingCoefficient, index);
    return 0.0;
  }

  return Base::mAspectProperties.mDampingCoefficients[index];
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setCoulombFriction(size_t index, s_t friction)
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(setCoulombFriction, index);
    return;
  }

  assert(friction >= 0.0);

  GenericJoint_SET_IF_DIFFERENT(mFrictions[index], friction);
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::getCoulombFriction(size_t index) const
{
  if (index >= getNumDofs())
  {
    GenericJoint_REPORT_OUT_OF_RANGE(getCoulombFriction, index);
    return 0.0;
  }

  return Base::mAspectProperties.mFrictions[index];
}

//==============================================================================
template <class ConfigSpaceT>
s_t GenericJoint<ConfigSpaceT>::computePotentialEnergy() const
{
  // Spring energy
  Vector displacement
      = getPositionsStatic() - Base::mAspectProperties.mRestPositions;

  const s_t pe = 0.5
                 * displacement.dot(
                     Base::mAspectProperties.mSpringStiffnesses.cwiseProduct(
                         displacement));

  return pe;
}

//==============================================================================
template <class ConfigSpaceT>
const math::Jacobian GenericJoint<ConfigSpaceT>::getRelativeJacobian() const
{
  return getRelativeJacobianStatic();
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::JacobianMatrix&
GenericJoint<ConfigSpaceT>::getRelativeJacobianStatic() const
{
  if (this->mIsRelativeJacobianDirty)
  {
    this->updateRelativeJacobian(false);
    this->mIsRelativeJacobianDirty = false;
  }

  return mJacobian;
}

//==============================================================================
template <class ConfigSpaceT>
math::Jacobian GenericJoint<ConfigSpaceT>::getRelativeJacobian(
    const Eigen::VectorXs& positions) const
{
  return getRelativeJacobianStatic(positions);
}

//==============================================================================
template <class ConfigSpaceT>
const math::Jacobian GenericJoint<ConfigSpaceT>::getRelativeJacobianTimeDeriv()
    const
{
  return getRelativeJacobianTimeDerivStatic();
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::JacobianMatrix&
GenericJoint<ConfigSpaceT>::getRelativeJacobianTimeDerivStatic() const
{
  if (this->mIsRelativeJacobianTimeDerivDirty)
  {
    this->updateRelativeJacobianTimeDeriv();
    this->mIsRelativeJacobianTimeDerivDirty = false;
  }

  return mJacobianDeriv;
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::JacobianMatrix&
GenericJoint<ConfigSpaceT>::getRelativeJacobianInPositionSpaceStatic() const
{
  if (this->mIsRelativeJacobianInPositionSpaceDirty)
  {
    this->updateRelativeJacobianInPositionSpace(false);
    this->mIsRelativeJacobianInPositionSpaceDirty = false;
  }

  return mJacobianInPositionSpace;
}

//==============================================================================
template <class ConfigSpaceT>
const math::Jacobian
GenericJoint<ConfigSpaceT>::getRelativeJacobianInPositionSpace() const
{
  return getRelativeJacobianInPositionSpaceStatic();
}

//==============================================================================
template <class ConfigSpaceT>
typename GenericJoint<ConfigSpaceT>::JacobianMatrix
GenericJoint<ConfigSpaceT>::getRelativeJacobianInPositionSpaceStatic(
    const Vector& positions) const
{
  // Default to just returning the ordinary Jacobian, which is defined in
  // velocity space. In most joints, except FreeJoint and BallJoint, these are
  // the same quantity.
  return getRelativeJacobianStatic(positions);
}

//==============================================================================
template <class ConfigSpaceT>
math::Jacobian GenericJoint<ConfigSpaceT>::getRelativeJacobianInPositionSpace(
    const Eigen::VectorXs& positions) const
{
  return getRelativeJacobianInPositionSpaceStatic(positions);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateRelativeJacobianInPositionSpace(
    bool /* mandatory */) const
{
  mJacobianInPositionSpace
      = getRelativeJacobianInPositionSpaceStatic(getPositionsStatic());
}

//==============================================================================
template <class ConfigSpaceT>
GenericJoint<ConfigSpaceT>::GenericJoint(const Properties& properties)
  : mVelocityChanges(Vector::Zero()),
    mImpulses(Vector::Zero()),
    mConstraintImpulses(Vector::Zero()),
    mJacobian(JacobianMatrix::Zero()),
    mJacobianDeriv(JacobianMatrix::Zero()),
    mInvProjArtInertia(Matrix::Zero()),
    mInvProjArtInertiaImplicit(Matrix::Zero()),
    mTotalForce(Vector::Zero()),
    mTotalImpulse(Vector::Zero())
{
  for (auto i = 0u; i < NumDofs; ++i)
    mDofs[i] = this->createDofPointer(i);

  // Joint and GenericJoint Aspects must be created by the most derived class.
  this->mAspectState.mPositions = properties.mInitialPositions;
  this->mAspectState.mVelocities = properties.mInitialVelocities;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::registerDofs()
{
  const SkeletonPtr& skel = this->mChildBodyNode->getSkeleton();
  for (auto i = 0u; i < NumDofs; ++i)
  {
    Base::mAspectProperties.mDofNames[i]
        = skel->mNameMgrForDofs.issueNewNameAndAdd(
            mDofs[i]->getName(), mDofs[i]);
  }
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::Vector6s GenericJoint<ConfigSpaceT>::getBodyConstraintWrench() const
{
  assert(this->mChildBodyNode);
  return this->mChildBodyNode->getBodyForce()
         - this->getRelativeJacobianStatic() * this->mAspectState.mForces;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateRelativeSpatialVelocity() const
{
  this->mSpatialVelocity
      = this->getRelativeJacobianStatic() * this->getVelocitiesStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateRelativeSpatialAcceleration() const
{
  this->mSpatialAcceleration = this->getRelativePrimaryAcceleration()
                               + this->getRelativeJacobianTimeDerivStatic()
                                     * this->getVelocitiesStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateRelativePrimaryAcceleration() const
{
  this->mPrimaryAcceleration
      = this->getRelativeJacobianStatic() * this->getAccelerationsStatic();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addVelocityTo(Eigen::Vector6s& vel)
{
  // Add joint velocity to _vel
  vel.noalias() += getRelativeJacobianStatic() * getVelocitiesStatic();

  // Verification
  assert(!math::isNan(vel));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::setPartialAccelerationTo(
    Eigen::Vector6s& partialAcceleration, const Eigen::Vector6s& childVelocity)
{
  // ad(V, S * dq) + dS * dq
  partialAcceleration
      = math::ad(
            childVelocity, getRelativeJacobianStatic() * getVelocitiesStatic())
        + getRelativeJacobianTimeDerivStatic() * getVelocitiesStatic();
  // Verification
  assert(!math::isNan(partialAcceleration));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addAccelerationTo(Eigen::Vector6s& acc)
{
  // Add joint acceleration to _acc
  acc.noalias() += getRelativeJacobianStatic() * getAccelerationsStatic();

  // Verification
  assert(!math::isNan(acc));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addVelocityChangeTo(
    Eigen::Vector6s& velocityChange)
{
  // Add joint velocity change to velocityChange
  velocityChange.noalias() += getRelativeJacobianStatic() * mVelocityChanges;

  // Verification
  assert(!math::isNan(velocityChange));
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::Matrix&
GenericJoint<ConfigSpaceT>::getInvProjArtInertia() const
{
  Joint::updateArticulatedInertia();

  return mInvProjArtInertia;
}

//==============================================================================
template <class ConfigSpaceT>
const typename GenericJoint<ConfigSpaceT>::Matrix&
GenericJoint<ConfigSpaceT>::getInvProjArtInertiaImplicit() const
{
  Joint::updateArticulatedInertia();

  return mInvProjArtInertiaImplicit;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaTo(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      addChildArtInertiaToDynamic(parentArtInertia, childArtInertia);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      addChildArtInertiaToKinematic(parentArtInertia, childArtInertia);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildArtInertiaTo);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getAlpha() const
{
  return mInvM_a;
}

//==============================================================================
template <class ConfigSpaceT>
math::Inertia GenericJoint<ConfigSpaceT>::computePi(
    const math::Inertia& AI) const
{
  const Matrix& psi = getInvProjArtInertia();
  const math::Jacobian& S = getRelativeJacobianStatic();
  const Eigen::MatrixXs AIS = AI * S;

  return AI - AIS * psi * AIS.transpose();
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::Vector6s GenericJoint<ConfigSpaceT>::computeBeta(
    const math::Inertia& AI, const Eigen::Vector6s& AB) const
{
  const Matrix& psi = getInvProjArtInertia();
  const math::Jacobian& S = getRelativeJacobianStatic();
  const Vector& alpha = mInvM_a;

  return AB + AI * S * psi * alpha;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::computeJacobianOfMinvX_init()
{
  mNumSkeletonDofs = this->getSkeleton()->getNumDofs();
  mInvM_Dpsi_Dq.resize(mNumSkeletonDofs);
  mInvM_DInvM_Dq.setZero(NumDofs, mNumSkeletonDofs);
  mInvM_Dalpha_Dq.setZero(NumDofs, mNumSkeletonDofs);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::computeJacobianOfMinvX_A(
    const math::Inertia& AI, const Eigen::Vector6s& AB)
{
  using math::ad;
  using math::AdInvT;
  using math::Jacobian;

  assert(mInvM_DInvM_Dq.cols() == static_cast<int>(mNumSkeletonDofs));

  const auto& skel = this->getSkeleton();
  BodyNode* childBody = this->getChildBodyNode();
  const BodyNode* parentBody = this->getParentBodyNode();

  std::vector<math::Inertia>& DPi_Dq = childBody->mInvM_DPi_Dq;
  math::Jacobian& Dbeta_Dq = childBody->mInvM_Dbeta_Dq;
  const std::vector<math::Inertia>& DAI_Dq = childBody->mInvM_DAI_Dq;
  const math::Jacobian& DAB_Dq = childBody->mInvM_DAB_Dq;

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const auto bodyNodeIndex = childBody->getIndexInSkeleton();
  auto& data = skel->mDiffMinv.nodes[bodyNodeIndex].data;
#endif

  //  const Eigen::Isometry3s& T = this->getRelativeTransform();
  const Jacobian& S = getRelativeJacobianStatic();
  const Jacobian AIS = AI * S;

  const Vector& alpha = mInvM_a;
  const Matrix& psi = getInvProjArtInertia();
#ifdef DART_DEBUG_ANALYTICAL_DERIV
  data.psi = psi;
  data.Pi = this->computePi(AI);
  data.alpha = alpha;
  data.beta = this->computeBeta(AI, AB);
#endif

  DPi_Dq = DAI_Dq;

  for (auto i = 0u; i < mNumSkeletonDofs; ++i)
  {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
    auto& deriv = skel->mDiffMinv.nodes[bodyNodeIndex].derivs[i];
#endif
    const DegreeOfFreedom* dof = skel->getDof(i);
    Matrix& Dpsi_Dq = mInvM_Dpsi_Dq[i];

    if (hasDof(dof))
    {
      const Jacobian DS_Dq
          = this->getRelativeJacobianDeriv(dof->getIndexInJoint());
      const Jacobian DdS_Dq
          = this->getRelativeJacobianTimeDerivDeriv(dof->getIndexInJoint());

      const Matrix tmp0 = DS_Dq.transpose() * AIS;
      Dpsi_Dq = -psi * (S.transpose() * DAI_Dq[i] * S + tmp0 + tmp0.transpose())
                * psi;

      mInvM_Dalpha_Dq.col(i).noalias() = -(DS_Dq.transpose() * AB);
      mInvM_Dalpha_Dq.col(i).noalias() -= S.transpose() * DAB_Dq.col(i);

      if (parentBody)
      {
        DPi_Dq[i].noalias() -= AIS * Dpsi_Dq * AIS.transpose();
        const Eigen::MatrixXs psiAIS_T = psi * AIS.transpose();
        const math::Inertia tmp1 = DAI_Dq[i] * S * psiAIS_T;
        DPi_Dq[i] -= tmp1;
        DPi_Dq[i] -= tmp1.transpose();
        const math::Inertia tmp2 = AI * DS_Dq * psiAIS_T;
        DPi_Dq[i] -= tmp2;
        DPi_Dq[i] -= tmp2.transpose();

        //      Dbeta_Dq.col(i).noalias() += DAI_Dq[i] * S * psi * alpha;
        //      Dbeta_Dq.col(i).noalias() += AI * DS_Dq * psi * alpha;
        Dbeta_Dq.col(i).noalias() += (DAI_Dq[i] * S + AI * DS_Dq) * psi * alpha;
        //      Dbeta_Dq.col(i).noalias() += AIS * Dpsi_Dq * alpha;
        //      Dbeta_Dq.col(i).noalias() += AIS * psi * mInvM_Dalpha_Dq.col(i);
        Dbeta_Dq.col(i).noalias()
            += AIS * (Dpsi_Dq * alpha + psi * mInvM_Dalpha_Dq.col(i));
      }
    }
    else
    {
      Dpsi_Dq = -psi * S.transpose() * DAI_Dq[i] * S * psi;

      mInvM_Dalpha_Dq.col(i).noalias() = -(S.transpose() * DAB_Dq.col(i));

      if (parentBody)
      {
        DPi_Dq[i].noalias() -= AIS * Dpsi_Dq * AIS.transpose();
        const math::Inertia tmp1 = DAI_Dq[i] * S * psi * AIS.transpose();
        DPi_Dq[i] -= tmp1;
        DPi_Dq[i] -= tmp1.transpose();

        Dbeta_Dq.col(i).noalias() += DAI_Dq[i] * S * psi * alpha;
        //      Dbeta_Dq.col(i).noalias() += AIS * Dpsi_Dq * alpha;
        //      Dbeta_Dq.col(i).noalias() += AIS * psi * mInvM_Dalpha_Dq.col(i);
        Dbeta_Dq.col(i).noalias()
            += AIS * (Dpsi_Dq * alpha + psi * mInvM_Dalpha_Dq.col(i));
      }
    }

#ifdef DART_DEBUG_ANALYTICAL_DERIV
    deriv.psi = Dpsi_Dq;
    deriv.Pi = DPi_Dq[i];
    deriv.alpha = mInvM_Dalpha_Dq.col(i);
    deriv.beta = Dbeta_Dq.col(i);
#endif
  }
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::MatrixXs GenericJoint<ConfigSpaceT>::computeJacobianOfMinvX_B(
    const math::Inertia& AI)
{
  using math::ad;
  using math::AdInvT;
  using math::AdInvTJac;
  using math::adJac;
  using math::Jacobian;

  assert(mInvM_DInvM_Dq.cols() == static_cast<int>(mNumSkeletonDofs));

  const auto& skel = this->getSkeleton();
  const BodyNode* parentBody = this->getParentBodyNode();
  BodyNode* childBody = this->getChildBodyNode();
  const std::vector<math::Inertia>& DAI_Dq = childBody->mInvM_DAI_Dq;

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  const auto bodyNodeIndex = childBody->getIndexInSkeleton();
  auto& data = skel->mDiffMinv.nodes[bodyNodeIndex].data;
#endif

  const Eigen::Isometry3s& T = this->getRelativeTransform();
  const Jacobian& S = getRelativeJacobianStatic();
  // TODO(JS): Assuming mInvM_a is already updated
  const Vector& alpha = mInvM_a;
  const Matrix& psi = getInvProjArtInertia();
  const Vector& ddq = mInvMassMatrixSegment;

  Eigen::MatrixXs& DInvM_Dq = mInvM_DInvM_Dq;
  Eigen::MatrixXs& Dalpha_Dq = mInvM_Dalpha_Dq;

  for (auto i = 0u; i < mNumSkeletonDofs; ++i)
  {
    const DegreeOfFreedom* dof = skel->getDof(i);
    const Matrix& Dpsi_Dq = mInvM_Dpsi_Dq[i];
#ifdef DART_DEBUG_ANALYTICAL_DERIV
    auto& deriv = skel->mDiffMinv.nodes[bodyNodeIndex].derivs[i];
#endif

    if (hasDof(dof))
    {
      const int dofIndexInJoint = static_cast<int>(dof->getIndexInJoint());
      const Eigen::Vector6s Scol = S.col(dofIndexInJoint);

      const Jacobian DS_Dq
          = this->getRelativeJacobianDeriv(dof->getIndexInJoint());
      const Jacobian DdS_Dq
          = this->getRelativeJacobianTimeDerivDeriv(dof->getIndexInJoint());

      if (parentBody)
      {
        const Eigen::Vector6s parent_dV2 = AdInvT(T, parentBody->mInvM_U);
        const math::Jacobian& parent_DdV_Dq = parentBody->mInvM_dV_q;
        const Eigen::Vector6s parent_DdV_Dq2 = AdInvT(T, parent_DdV_Dq.col(i));

        DInvM_Dq.col(i).noalias()
            = Dpsi_Dq * (alpha - S.transpose() * AI * parent_dV2);
        DInvM_Dq.col(i).noalias()
            += psi
               * (Dalpha_Dq.col(i)
                  - (DS_Dq.transpose() * AI + S.transpose() * DAI_Dq[i])
                        * parent_dV2
                  - S.transpose() * AI
                        * (parent_DdV_Dq2 - ad(Scol, parent_dV2)));

        childBody->mInvM_dV_q.col(i).noalias()
            = AdInvT(T, parentBody->mInvM_dV_q.col(i))
              - ad(Scol, AdInvT(T, parentBody->mInvM_U)) + DS_Dq * ddq
              + S * DInvM_Dq.col(i);
      }
      else
      {
        DInvM_Dq.col(i).noalias() = Dpsi_Dq * alpha;
        DInvM_Dq.col(i).noalias() += psi * Dalpha_Dq.col(i);

        childBody->mInvM_dV_q.col(i).noalias()
            = DS_Dq * ddq + S * DInvM_Dq.col(i);
      }
    }
    else
    {
      if (parentBody)
      {
        const Eigen::Vector6s parent_dV2 = AdInvT(T, parentBody->mInvM_U);
        const math::Jacobian& parent_DdV_Dq = parentBody->mInvM_dV_q;
        const Eigen::Vector6s parent_DdV_Dq2
            = AdInvTJac(T, parent_DdV_Dq.col(i));

        DInvM_Dq.col(i).noalias()
            = Dpsi_Dq * (alpha - S.transpose() * AI * parent_dV2);
        DInvM_Dq.col(i).noalias()
            += psi
               * (Dalpha_Dq.col(i) - S.transpose() * DAI_Dq[i] * parent_dV2
                  - S.transpose() * AI * parent_DdV_Dq2);

        childBody->mInvM_dV_q.col(i).noalias()
            = AdInvT(T, parentBody->mInvM_dV_q.col(i)) + S * DInvM_Dq.col(i);
      }
      else
      {
        DInvM_Dq.col(i).noalias() = Dpsi_Dq * alpha;
        DInvM_Dq.col(i).noalias() += psi * Dalpha_Dq.col(i);

        childBody->mInvM_dV_q.col(i).noalias() = S * DInvM_Dq.col(i);
      }
    }
#ifdef DART_DEBUG_ANALYTICAL_DERIV
    deriv.ddq = DInvM_Dq.col(i);
    deriv.dV = childBody->mInvM_dV_q.col(i);
#endif
  }

  return DInvM_Dq;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaToDynamic(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  // Child body's articulated inertia
  JacobianMatrix AIS = childArtInertia * getRelativeJacobianStatic();
  Eigen::Matrix6s PI = childArtInertia;
  PI.noalias() -= AIS * mInvProjArtInertia * AIS.transpose();
  assert(!math::isNan(PI));

  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia
      += math::transformInertia(this->getRelativeTransform().inverse(), PI);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaToKinematic(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia += math::transformInertia(
      this->getRelativeTransform().inverse(), childArtInertia);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaImplicitTo(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      addChildArtInertiaImplicitToDynamic(parentArtInertia, childArtInertia);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      addChildArtInertiaImplicitToKinematic(parentArtInertia, childArtInertia);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildArtInertiaImplicitTo);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaImplicitToDynamic(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  // Child body's articulated inertia
  JacobianMatrix AIS = childArtInertia * getRelativeJacobianStatic();
  Eigen::Matrix6s PI = childArtInertia;
  PI.noalias() -= AIS * mInvProjArtInertiaImplicit * AIS.transpose();
  assert(!math::isNan(PI));

  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia
      += math::transformInertia(this->getRelativeTransform().inverse(), PI);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaImplicitToKinematic(
    Eigen::Matrix6s& parentArtInertia, const Eigen::Matrix6s& childArtInertia)
{
  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia += math::transformInertia(
      this->getRelativeTransform().inverse(), childArtInertia);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertia(
    const Eigen::Matrix6s& artInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateInvProjArtInertiaDynamic(artInertia);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateInvProjArtInertiaKinematic(artInertia);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateInvProjArtInertia);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaDynamic(
    const Eigen::Matrix6s& artInertia)
{
  // Projected articulated inertia
  const JacobianMatrix& Jacobian = getRelativeJacobianStatic();
  const Matrix projAI = Jacobian.transpose() * artInertia * Jacobian;

  // Inversion of projected articulated inertia
  mInvProjArtInertia = math::inverse<ConfigSpaceT>(projAI);

  // Verification
  assert(!math::isNan(mInvProjArtInertia));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaKinematic(
    const Eigen::Matrix6s& /*_artInertia*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaImplicit(
    const Eigen::Matrix6s& artInertia, s_t timeStep)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateInvProjArtInertiaImplicitDynamic(artInertia, timeStep);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateInvProjArtInertiaImplicitKinematic(artInertia, timeStep);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateInvProjArtInertiaImplicit);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaImplicitDynamic(
    const Eigen::Matrix6s& artInertia, s_t timeStep)
{
  // Projected articulated inertia
  const JacobianMatrix& Jacobian = getRelativeJacobianStatic();
  Matrix projAI = Jacobian.transpose() * artInertia * Jacobian;

  // Add additional inertia for implicit damping and spring force
  projAI += (timeStep * Base::mAspectProperties.mDampingCoefficients
             + timeStep * timeStep * Base::mAspectProperties.mSpringStiffnesses)
                .asDiagonal();

  // Inversion of projected articulated inertia
  mInvProjArtInertiaImplicit = math::inverse<ConfigSpaceT>(projAI);

  // Verification
  assert(!math::isNan(mInvProjArtInertiaImplicit));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaImplicitKinematic(
    const Eigen::Matrix6s& /*artInertia*/, s_t /*timeStep*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasForceTo(
    Eigen::Vector6s& parentBiasForce,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasForce,
    const Eigen::Vector6s& childPartialAcc)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      addChildBiasForceToDynamic(
          parentBiasForce, childArtInertia, childBiasForce, childPartialAcc);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      addChildBiasForceToKinematic(
          parentBiasForce, childArtInertia, childBiasForce, childPartialAcc);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildBiasForceTo);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasForceToDynamic(
    Eigen::Vector6s& parentBiasForce,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasForce,
    const Eigen::Vector6s& childPartialAcc)
{
  // Compute beta
  const Eigen::Vector6s beta
      = childBiasForce
        + childArtInertia
              * (childPartialAcc
                 + getRelativeJacobianStatic() * getInvProjArtInertiaImplicit()
                       * mTotalForce);

  //    Eigen::Vector6s beta
  //        = _childBiasForce;
  //    beta.noalias() += _childArtInertia * _childPartialAcc;
  //    beta.noalias() += _childArtInertia *  mJacobian *
  //    getInvProjArtInertiaImplicit() * mTotalForce;

  // Verification
  assert(!math::isNan(beta));

  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  parentBiasForce += math::dAdInvT(this->getRelativeTransform(), beta);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasForceToKinematic(
    Eigen::Vector6s& _parentBiasForce,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasForce,
    const Eigen::Vector6s& childPartialAcc)
{
  // Compute beta
  const Eigen::Vector6s beta
      = childBiasForce
        + childArtInertia
              * (childPartialAcc
                 + getRelativeJacobianStatic() * getAccelerationsStatic());

  //    Eigen::Vector6s beta
  //        = _childBiasForce;
  //    beta.noalias() += _childArtInertia * _childPartialAcc;
  //    beta.noalias() += _childArtInertia *  mJacobian *
  //    getInvProjArtInertiaImplicit() * mTotalForce;

  // Verification
  assert(!math::isNan(beta));

  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  _parentBiasForce += math::dAdInvT(this->getRelativeTransform(), beta);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasImpulseTo(
    Eigen::Vector6s& parentBiasImpulse,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasImpulse)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      addChildBiasImpulseToDynamic(
          parentBiasImpulse, childArtInertia, childBiasImpulse);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      addChildBiasImpulseToKinematic(
          parentBiasImpulse, childArtInertia, childBiasImpulse);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildBiasImpulseTo);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasImpulseToDynamic(
    Eigen::Vector6s& _parentBiasImpulse,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasImpulse)
{
  // Compute beta
  const Eigen::Vector6s beta = childBiasImpulse
                               + childArtInertia * getRelativeJacobianStatic()
                                     * getInvProjArtInertia() * mTotalImpulse;

  // Verification
  assert(!math::isNan(beta));

  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  _parentBiasImpulse += math::dAdInvT(this->getRelativeTransform(), beta);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasImpulseToKinematic(
    Eigen::Vector6s& parentBiasImpulse,
    const Eigen::Matrix6s& /*childArtInertia*/,
    const Eigen::Vector6s& childBiasImpulse)
{
  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  parentBiasImpulse
      += math::dAdInvT(this->getRelativeTransform(), childBiasImpulse);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalForce(
    const Eigen::Vector6s& bodyForce, s_t timeStep)
{
  assert(timeStep > 0.0);

  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
      this->mAspectState.mForces = this->mAspectState.mCommands;
      updateTotalForceDynamic(bodyForce, timeStep);
      break;
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      this->mAspectState.mForces.setZero();
      updateTotalForceDynamic(bodyForce, timeStep);
      break;
    case Joint::ACCELERATION:
      setAccelerationsStatic(this->mAspectState.mCommands);
      updateTotalForceKinematic(bodyForce, timeStep);
      break;
    case Joint::VELOCITY:
      setAccelerationsStatic(
          (this->mAspectState.mCommands - getVelocitiesStatic()) / timeStep);
      updateTotalForceKinematic(bodyForce, timeStep);
      break;
    case Joint::LOCKED:
      setVelocitiesStatic(Vector::Zero());
      setAccelerationsStatic(Vector::Zero());
      updateTotalForceKinematic(bodyForce, timeStep);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateTotalForce);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalForceDynamic(
    const Eigen::Vector6s& bodyForce, s_t timeStep)
{
  // Spring force
  const Vector springForce
      = -Base::mAspectProperties.mSpringStiffnesses.cwiseProduct(
          getPositionsStatic() - Base::mAspectProperties.mRestPositions
          + getVelocitiesStatic() * timeStep);

  // Damping force
  const Vector dampingForce
      = -Base::mAspectProperties.mDampingCoefficients.cwiseProduct(
          getVelocitiesStatic());

  //
  mTotalForce = this->mAspectState.mForces + springForce + dampingForce
                - getRelativeJacobianStatic().transpose() * bodyForce;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalForceKinematic(
    const Eigen::Vector6s& /*bodyForce*/, s_t /*timeStep*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalImpulse(
    const Eigen::Vector6s& bodyImpulse)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateTotalImpulseDynamic(bodyImpulse);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateTotalImpulseKinematic(bodyImpulse);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateTotalImpulse);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalImpulseDynamic(
    const Eigen::Vector6s& bodyImpulse)
{
  //
  mTotalImpulse = mConstraintImpulses
                  - getRelativeJacobianStatic().transpose() * bodyImpulse;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalImpulseKinematic(
    const Eigen::Vector6s& /*bodyImpulse*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::resetTotalImpulses()
{
  mTotalImpulse.setZero();
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateAcceleration(
    const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& spatialAcc)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateAccelerationDynamic(artInertia, spatialAcc);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateAccelerationKinematic(artInertia, spatialAcc);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateAcceleration);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateAccelerationDynamic(
    const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& spatialAcc)
{
  //
  setAccelerationsStatic(
      getInvProjArtInertiaImplicit()
      * (mTotalForce
         - getRelativeJacobianStatic().transpose() * artInertia
               * math::AdInvT(this->getRelativeTransform(), spatialAcc)));

  // Verification
  assert(!math::isNan(getAccelerationsStatic()));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateAccelerationKinematic(
    const Eigen::Matrix6s& /*artInertia*/,
    const Eigen::Vector6s& /*spatialAcc*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateVelocityChange(
    const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& velocityChange)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateVelocityChangeDynamic(artInertia, velocityChange);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateVelocityChangeKinematic(artInertia, velocityChange);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateVelocityChange);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateVelocityChangeDynamic(
    const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& velocityChange)
{
  //
  mVelocityChanges
      = getInvProjArtInertia()
        * (mTotalImpulse
           - getRelativeJacobianStatic().transpose() * artInertia
                 * math::AdInvT(this->getRelativeTransform(), velocityChange));

  // Verification
  assert(!math::isNan(mVelocityChanges));
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateVelocityChangeKinematic(
    const Eigen::Matrix6s& /*artInertia*/,
    const Eigen::Vector6s& /*velocityChange*/)
{
  // Do nothing
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateForceID(
    const Eigen::Vector6s& bodyForce,
    s_t timeStep,
    bool withDampingForces,
    bool withSpringForces)
{
  this->mAspectState.mForces
      = getRelativeJacobianStatic().transpose() * bodyForce;

  // Damping force
  if (withDampingForces)
  {
    const typename ConfigSpaceT::Vector dampingForces
        = -Base::mAspectProperties.mDampingCoefficients.cwiseProduct(
            getVelocitiesStatic());
    this->mAspectState.mForces -= dampingForces;
  }

  // Spring force
  if (withSpringForces)
  {
    const typename ConfigSpaceT::Vector springForces
        = -Base::mAspectProperties.mSpringStiffnesses.cwiseProduct(
            getPositionsStatic() - Base::mAspectProperties.mRestPositions
            + getVelocitiesStatic() * timeStep);
    this->mAspectState.mForces -= springForces;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateForceFD(
    const Eigen::Vector6s& bodyForce,
    s_t timeStep,
    bool withDampingForces,
    bool withSpringForces)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateForceID(bodyForce, timeStep, withDampingForces, withSpringForces);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateForceFD);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateImpulseID(
    const Eigen::Vector6s& bodyImpulse)
{
  mImpulses = getRelativeJacobianStatic().transpose() * bodyImpulse;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateImpulseFD(
    const Eigen::Vector6s& bodyImpulse)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateImpulseID(bodyImpulse);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateImpulseFD);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateConstrainedTerms(s_t timeStep)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
      updateConstrainedTermsDynamic(timeStep);
      break;
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      updateConstrainedTermsKinematic(timeStep);
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateConstrainedTerms);
      break;
  }
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateConstrainedTermsDynamic(s_t timeStep)
{
  const s_t invTimeStep = 1.0 / timeStep;

  setVelocitiesStatic(getVelocitiesStatic() + mVelocityChanges);
  setAccelerationsStatic(
      getAccelerationsStatic() + mVelocityChanges * invTimeStep);
  this->mAspectState.mForces.noalias() += mImpulses * invTimeStep;
  // Note: As long as this is only called from BodyNode::updateConstrainedTerms
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateConstrainedTermsKinematic(s_t timeStep)
{
  this->mAspectState.mForces.noalias() += mImpulses / timeStep;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasForceForInvMassMatrix(
    Eigen::Vector6s& parentBiasForce,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasForce)
{
  // Compute beta
  Eigen::Vector6s beta = childBiasForce;
  beta.noalias() += childArtInertia * getRelativeJacobianStatic()
                    * getInvProjArtInertia() * mInvM_a;

  // Verification
  assert(!math::isNan(beta));

  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  parentBiasForce += math::dAdInvT(this->getRelativeTransform(), beta);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildBiasForceForInvAugMassMatrix(
    Eigen::Vector6s& parentBiasForce,
    const Eigen::Matrix6s& childArtInertia,
    const Eigen::Vector6s& childBiasForce)
{
  // Compute beta
  Eigen::Vector6s beta = childBiasForce;
  beta.noalias() += childArtInertia * getRelativeJacobianStatic()
                    * getInvProjArtInertiaImplicit() * mInvM_a;

  // Verification
  assert(!math::isNan(beta));

  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  parentBiasForce += math::dAdInvT(this->getRelativeTransform(), beta);
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateTotalForceForInvMassMatrix(
    const Eigen::Vector6s& bodyForce)
{
  // Compute alpha
  mInvM_a = this->mAspectState.mForces
            - getRelativeJacobianStatic().transpose() * bodyForce;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::getInvMassMatrixSegment(
    Eigen::MatrixXs& _invMassMat,
    const size_t _col,
    const Eigen::Matrix6s& artInertia,
    const Eigen::Vector6s& spatialAcc)
{
  //
  mInvMassMatrixSegment
      = getInvProjArtInertia()
        * (mInvM_a
           - getRelativeJacobianStatic().transpose() * artInertia
                 * math::AdInvT(this->getRelativeTransform(), spatialAcc));

  // Verification
  assert(!math::isNan(mInvMassMatrixSegment));

  // Index
  size_t iStart = mDofs[0]->mIndexInTree;

  // Assign
  _invMassMat.block<NumDofs, 1>(iStart, _col) = mInvMassMatrixSegment;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::getInvAugMassMatrixSegment(
    Eigen::MatrixXs& invMassMat,
    const size_t col,
    const Eigen::Matrix6s& artInertia,
    const Eigen::Vector6s& spatialAcc)
{
  //
  mInvMassMatrixSegment
      = getInvProjArtInertiaImplicit()
        * (mInvM_a
           - getRelativeJacobianStatic().transpose() * artInertia
                 * math::AdInvT(this->getRelativeTransform(), spatialAcc));

  // Verification
  assert(!math::isNan(mInvMassMatrixSegment));

  // Index
  size_t iStart = mDofs[0]->mIndexInTree;

  // Assign
  invMassMat.block<NumDofs, 1>(iStart, col) = mInvMassMatrixSegment;
}

//==============================================================================
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addInvMassMatrixSegmentTo(Eigen::Vector6s& acc)
{
  //
  acc += getRelativeJacobianStatic() * mInvMassMatrixSegment;
}

//==============================================================================
template <class ConfigSpaceT>
Eigen::VectorXs GenericJoint<ConfigSpaceT>::getSpatialToGeneralized(
    const Eigen::Vector6s& spatial)
{
  return getRelativeJacobianStatic().transpose() * spatial;
}

} // namespace dynamics
} // namespace dart

#endif // DART_DYNAMICS_DETAIL_GenericJoint_HPP_
