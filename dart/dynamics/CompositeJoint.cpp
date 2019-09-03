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

#include "dart/dynamics/CompositeJoint.hpp"

#include "dart/dynamics/RevoluteJoint.hpp"

//==============================================================================
#define COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(func, index)                       \
  {                                                                            \
    dterr << "[CompositeJoint::" << #func << "] The index [" << index          \
          << "] is out of range for Joint named [" << this->getName()          \
          << "] which has " << this->getNumDofs() << " DOFs. "                 \
          << "Returning a default value that may not what you expect.\n";      \
    assert(false);                                                             \
  }

//==============================================================================
#define COMPOSITE_JOINT_GET_VALUE(func, index)                                 \
  if (index < mNumDofs)                                                        \
  {                                                                            \
    const auto& subJointIndex = mJointIndexMap[index];                         \
    const auto& localIndex = mLocalDofIndexMap[index];                         \
    return mJoints[subJointIndex]->func(localIndex);                           \
  }                                                                            \
                                                                               \
  COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(#func, index);

//==============================================================================
#define COMPOSITE_JOINT_GET_VECTOR(func)                                       \
  Eigen::Vector3d values_(mNumDofs);                                           \
                                                                               \
  int index_ = 0;                                                              \
  for (const std::unique_ptr<Joint>& joint : mJoints)                          \
  {                                                                            \
    const int size = static_cast<int>(joint->getNumDofs());                    \
    values_.segment(index_, size) = joint->func();                             \
    index_ += size;                                                            \
  }                                                                            \
                                                                               \
  return values_;

//==============================================================================
#define COMPOSITE_JOINT_GET_VECTOR_2(func, param1, param2)                     \
  Eigen::Vector3d values_(mNumDofs);                                           \
                                                                               \
  int index_ = 0;                                                              \
  for (const std::unique_ptr<Joint>& joint : mJoints)                          \
  {                                                                            \
    const int size = static_cast<int>(joint->getNumDofs());                    \
    values_.segment(index_, size) = joint->func(                               \
        param1.segment(index_, size), param2.segment(index_, size));           \
    index_ += size;                                                            \
  }                                                                            \
                                                                               \
  return values_;

//==============================================================================
#define COMPOSITE_JOINT_SET_VALUE(func, index, val)                            \
  if (index < mNumDofs)                                                        \
  {                                                                            \
    const auto& subJointIndex_ = mJointIndexMap[index];                        \
    const auto& localIndex_ = mLocalDofIndexMap[index];                        \
    return mJoints[subJointIndex_]->func(localIndex_, val);                    \
  }                                                                            \
                                                                               \
  COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(#func, index);

//==============================================================================
#define COMPOSITE_JOINT_SET_VECTOR(func, vec)                                  \
  int index_ = 0;                                                              \
  for (const std::unique_ptr<Joint>& joint : mJoints)                          \
  {                                                                            \
    const int size = static_cast<int>(joint->getNumDofs());                    \
    joint->func(vec.segment(index_, size));                                    \
    index_ += size;                                                            \
  }

//==============================================================================
#define COMPOSITE_JOINT_CALL_FUNCTION(func, index)                             \
  if (index < mNumDofs)                                                        \
  {                                                                            \
    const auto& subJointIndex = mJointIndexMap[index];                         \
    const auto& localIndex = mLocalDofIndexMap[index];                         \
    mJoints[subJointIndex]->func(localIndex);                                  \
  }                                                                            \
                                                                               \
  COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(#func, index);

//==============================================================================
#define COMPOSITE_JOINT_CALL_FUNCTIONS(func)                                   \
  for (const std::unique_ptr<Joint>& joint : mJoints)                          \
    joint->func();

//==============================================================================
#define COMPOSITE_JOINT_CALL_FUNCTIONS_1(func, val1)                           \
  for (const std::unique_ptr<Joint>& joint : mJoints)                          \
    joint->func(val1);

//==============================================================================
namespace dart {
namespace dynamics {

//==============================================================================
CompositeJoint::Properties::Properties(const Joint::Properties& properties)
  : Joint::Properties(properties)
{
  // Do nothing
}

//==============================================================================
CompositeJoint::Properties CompositeJoint::getCompositeJointProperties() const
{
  return getJointProperties();
}

//==============================================================================
const std::string& CompositeJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& CompositeJoint::getStaticType()
{
  static const std::string name = "CompositeJoint";
  return name;
}

//==============================================================================
std::size_t CompositeJoint::getNumJoints() const
{
  return mJoints.size();
}

//==============================================================================
Joint* CompositeJoint::getJoint(std::size_t index)
{
  if (index < mNumDofs)
    return mJoints[index].get();

  COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(getJoint, index);

  return nullptr;
}

//==============================================================================
const Joint* CompositeJoint::getJoint(std::size_t index) const
{
  if (index < mNumDofs)
    return mJoints[index].get();

  COMPOSITE_JOINT_REPORT_OUT_OF_RANGE(getJoint, index);

  return nullptr;
}

//==============================================================================
bool CompositeJoint::isCyclic(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(isCyclic, index);

  return false;
}

//==============================================================================
DegreeOfFreedom* CompositeJoint::getDof(std::size_t index)
{
  COMPOSITE_JOINT_GET_VALUE(getDof, index);

  return nullptr;
}

//==============================================================================
const DegreeOfFreedom* CompositeJoint::getDof(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getDof, index);

  return nullptr;
}

//==============================================================================
const std::string& CompositeJoint::setDofName(
    std::size_t index, const std::string& name, bool preserveName)
{
}

//==============================================================================
void CompositeJoint::preserveDofName(std::size_t index, bool preserve)
{
}

//==============================================================================
bool CompositeJoint::isDofNamePreserved(std::size_t index) const
{
}

//==============================================================================
const std::string& CompositeJoint::getDofName(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getDofName, index);

  static std::string empty;
  return empty;
}

//==============================================================================
std::size_t CompositeJoint::getNumDofs() const
{
  return mNumDofs;
}

//==============================================================================
std::size_t CompositeJoint::getIndexInSkeleton(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getIndexInSkeleton, index);

  return 0u;
}

//==============================================================================
std::size_t CompositeJoint::getIndexInTree(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getIndexInTree, index);

  return 0;
}

//==============================================================================
void CompositeJoint::setCommand(std::size_t index, double command)
{
  COMPOSITE_JOINT_SET_VALUE(setCommand, index, command);
}

//==============================================================================
double CompositeJoint::getCommand(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getCommand, index);

  return 0;
}

//==============================================================================
void CompositeJoint::setCommands(const Eigen::VectorXd& commands)
{
  COMPOSITE_JOINT_SET_VECTOR(setCommands, commands);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getCommands() const
{
  COMPOSITE_JOINT_GET_VECTOR(getCommands);
}

//==============================================================================
void CompositeJoint::resetCommands()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetCommands)
}

//==============================================================================
void CompositeJoint::setPosition(std::size_t index, double position)
{
  COMPOSITE_JOINT_SET_VALUE(setPosition, index, position);
}

//==============================================================================
double CompositeJoint::getPosition(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getPosition, index);
}

//==============================================================================
void CompositeJoint::setPositions(const Eigen::VectorXd& positions)
{
  COMPOSITE_JOINT_SET_VECTOR(setPositions, positions);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getPositions() const
{
  COMPOSITE_JOINT_GET_VECTOR(getPositions);
}

//==============================================================================
void CompositeJoint::setPositionLowerLimit(std::size_t index, double position)
{
  COMPOSITE_JOINT_SET_VALUE(setPositionLowerLimit, index, position);
}

//==============================================================================
double CompositeJoint::getPositionLowerLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getPositionLowerLimit, index);
}

//==============================================================================
void CompositeJoint::setPositionLowerLimits(const Eigen::VectorXd& lowerLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setPositionLowerLimits, lowerLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getPositionLowerLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getPositionLowerLimits);
}

//==============================================================================
void CompositeJoint::setPositionUpperLimit(std::size_t index, double position)
{
  COMPOSITE_JOINT_SET_VALUE(setPositionUpperLimit, index, position);
}

//==============================================================================
double CompositeJoint::getPositionUpperLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getPositionUpperLimit, index);
}

//==============================================================================
void CompositeJoint::setPositionUpperLimits(const Eigen::VectorXd& upperLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setPositionUpperLimits, upperLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getPositionUpperLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getPositionUpperLimits);
}

//==============================================================================
bool CompositeJoint::hasPositionLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(hasPositionLimit, index);
}

//==============================================================================
void CompositeJoint::resetPosition(std::size_t index)
{
  COMPOSITE_JOINT_CALL_FUNCTION(resetPosition, index);
}

//==============================================================================
void CompositeJoint::resetPositions()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetPositions)
}

//==============================================================================
void CompositeJoint::setInitialPosition(std::size_t index, double initial)
{
  COMPOSITE_JOINT_SET_VALUE(setInitialPosition, index, initial);
}

//==============================================================================
double CompositeJoint::getInitialPosition(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getInitialPosition, index);
}

//==============================================================================
void CompositeJoint::setInitialPositions(const Eigen::VectorXd& initial)
{
  COMPOSITE_JOINT_SET_VECTOR(setInitialPositions, initial);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getInitialPositions() const
{
  COMPOSITE_JOINT_GET_VECTOR(getInitialPositions);
}

//==============================================================================
void CompositeJoint::setVelocity(std::size_t index, double velocity)
{
  COMPOSITE_JOINT_SET_VALUE(setVelocity, index, velocity);
}

//==============================================================================
double CompositeJoint::getVelocity(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getVelocity, index);
}

//==============================================================================
void CompositeJoint::setVelocities(const Eigen::VectorXd& velocities)
{
  COMPOSITE_JOINT_SET_VECTOR(setVelocities, velocities);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getVelocities() const
{
  COMPOSITE_JOINT_GET_VECTOR(getVelocities);
}

//==============================================================================
void CompositeJoint::setVelocityLowerLimit(std::size_t index, double velocity)
{
  COMPOSITE_JOINT_SET_VALUE(setVelocityLowerLimit, index, velocity);
}

//==============================================================================
double CompositeJoint::getVelocityLowerLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getVelocityLowerLimit, index);
}

//==============================================================================
void CompositeJoint::setVelocityLowerLimits(const Eigen::VectorXd& lowerLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setVelocityLowerLimits, lowerLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getVelocityLowerLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getVelocityLowerLimits);
}

//==============================================================================
void CompositeJoint::setVelocityUpperLimit(std::size_t index, double velocity)
{
  COMPOSITE_JOINT_SET_VALUE(setVelocityUpperLimit, index, velocity);
}

//==============================================================================
double CompositeJoint::getVelocityUpperLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getVelocityUpperLimit, index);
}

//==============================================================================
void CompositeJoint::setVelocityUpperLimits(const Eigen::VectorXd& upperLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setVelocityUpperLimits, upperLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getVelocityUpperLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getVelocityUpperLimits);
}

//==============================================================================
void CompositeJoint::resetVelocity(std::size_t index)
{
  COMPOSITE_JOINT_CALL_FUNCTION(resetVelocity, index);
}

//==============================================================================
void CompositeJoint::resetVelocities()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetVelocities)
}

//==============================================================================
void CompositeJoint::setInitialVelocity(std::size_t index, double initial)
{
  COMPOSITE_JOINT_SET_VALUE(setInitialVelocity, index, initial);
}

//==============================================================================
double CompositeJoint::getInitialVelocity(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getInitialVelocity, index);
}

//==============================================================================
void CompositeJoint::setInitialVelocities(const Eigen::VectorXd& initial)
{
  COMPOSITE_JOINT_SET_VECTOR(setInitialVelocities, initial);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getInitialVelocities() const
{
  COMPOSITE_JOINT_GET_VECTOR(getInitialVelocities);
}

//==============================================================================
void CompositeJoint::setAcceleration(std::size_t index, double acceleration)
{
  COMPOSITE_JOINT_SET_VALUE(setAcceleration, index, acceleration);
}

//==============================================================================
double CompositeJoint::getAcceleration(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getAcceleration, index);
}

//==============================================================================
void CompositeJoint::setAccelerations(const Eigen::VectorXd& accelerations)
{
  COMPOSITE_JOINT_SET_VECTOR(setAccelerations, accelerations);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getAccelerations() const
{
  COMPOSITE_JOINT_GET_VECTOR(getAccelerations);
}

//==============================================================================
void CompositeJoint::setAccelerationLowerLimit(
    std::size_t index, double acceleration)
{
  COMPOSITE_JOINT_SET_VALUE(setAccelerationLowerLimit, index, acceleration);
}

//==============================================================================
double CompositeJoint::getAccelerationLowerLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getAccelerationLowerLimit, index);
}

//==============================================================================
void CompositeJoint::setAccelerationLowerLimits(
    const Eigen::VectorXd& lowerLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setAccelerationLowerLimits, lowerLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getAccelerationLowerLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getAccelerationLowerLimits);
}

//==============================================================================
void CompositeJoint::setAccelerationUpperLimit(
    std::size_t index, double acceleration)
{
  COMPOSITE_JOINT_SET_VALUE(setAccelerationUpperLimit, index, acceleration);
}

//==============================================================================
double CompositeJoint::getAccelerationUpperLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getAccelerationUpperLimit, index);
}

//==============================================================================
void CompositeJoint::setAccelerationUpperLimits(
    const Eigen::VectorXd& upperLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setAccelerationUpperLimits, upperLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getAccelerationUpperLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getAccelerationUpperLimits);
}

//==============================================================================
void CompositeJoint::resetAccelerations()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetAccelerations)
}

//==============================================================================
void CompositeJoint::setForce(std::size_t index, double force)
{
  COMPOSITE_JOINT_SET_VALUE(setForce, index, force);
}

//==============================================================================
double CompositeJoint::getForce(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getForce, index);
}

//==============================================================================
void CompositeJoint::setForces(const Eigen::VectorXd& forces)
{
  COMPOSITE_JOINT_SET_VECTOR(setForces, forces);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getForces() const
{
  COMPOSITE_JOINT_GET_VECTOR(getForces);
}

//==============================================================================
void CompositeJoint::setForceLowerLimit(std::size_t index, double force)
{
  COMPOSITE_JOINT_SET_VALUE(setForceLowerLimit, index, force);
}

//==============================================================================
double CompositeJoint::getForceLowerLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getForceLowerLimit, index);
}

//==============================================================================
void CompositeJoint::setForceLowerLimits(const Eigen::VectorXd& lowerLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setForceLowerLimits, lowerLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getForceLowerLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getForceLowerLimits);
}

//==============================================================================
void CompositeJoint::setForceUpperLimit(std::size_t index, double force)
{
  COMPOSITE_JOINT_SET_VALUE(setForceUpperLimit, index, force);
}

//==============================================================================
double CompositeJoint::getForceUpperLimit(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getForceUpperLimit, index);
}

//==============================================================================
void CompositeJoint::setForceUpperLimits(const Eigen::VectorXd& upperLimits)
{
  COMPOSITE_JOINT_SET_VECTOR(setForceUpperLimits, upperLimits);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getForceUpperLimits() const
{
  COMPOSITE_JOINT_GET_VECTOR(getForceUpperLimits);
}

//==============================================================================
void CompositeJoint::resetForces()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetForces)
}

//==============================================================================
void CompositeJoint::setVelocityChange(std::size_t index, double velocityChange)
{
  COMPOSITE_JOINT_SET_VALUE(setVelocityChange, index, velocityChange);
}

//==============================================================================
double CompositeJoint::getVelocityChange(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getVelocityChange, index);
}

//==============================================================================
void CompositeJoint::resetVelocityChanges()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetVelocityChanges)
}

//==============================================================================
void CompositeJoint::setConstraintImpulse(std::size_t index, double impulse)
{
  COMPOSITE_JOINT_SET_VALUE(setConstraintImpulse, index, impulse);
}

//==============================================================================
double CompositeJoint::getConstraintImpulse(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getConstraintImpulse, index);
}

//==============================================================================
void CompositeJoint::resetConstraintImpulses()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetConstraintImpulses)
}

//==============================================================================
void CompositeJoint::integratePositions(double dt)
{
  COMPOSITE_JOINT_CALL_FUNCTIONS_1(integratePositions, dt);
}

//==============================================================================
void CompositeJoint::integrateVelocities(double dt)
{
  COMPOSITE_JOINT_CALL_FUNCTIONS_1(integrateVelocities, dt);
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getPositionDifferences(
    const Eigen::VectorXd& q2, const Eigen::VectorXd& q1) const
{
  COMPOSITE_JOINT_GET_VECTOR_2(getPositionDifferences, q2, q1);
}

//==============================================================================
void CompositeJoint::setSpringStiffness(std::size_t index, double k)
{
  COMPOSITE_JOINT_SET_VALUE(setSpringStiffness, index, k);
}

//==============================================================================
double CompositeJoint::getSpringStiffness(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getSpringStiffness, index);
}

//==============================================================================
void CompositeJoint::setRestPosition(std::size_t index, double q0)
{
  COMPOSITE_JOINT_SET_VALUE(setRestPosition, index, q0);
}

//==============================================================================
double CompositeJoint::getRestPosition(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getRestPosition, index);
}

//==============================================================================
void CompositeJoint::setDampingCoefficient(std::size_t index, double d)
{
  COMPOSITE_JOINT_SET_VALUE(setSpringStiffness, index, d);
}

//==============================================================================
double CompositeJoint::getDampingCoefficient(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getDampingCoefficient, index);
}

//==============================================================================
void CompositeJoint::setCoulombFriction(std::size_t index, double friction)
{
  COMPOSITE_JOINT_SET_VALUE(setCoulombFriction, index, friction);
}

//==============================================================================
double CompositeJoint::getCoulombFriction(std::size_t index) const
{
  COMPOSITE_JOINT_GET_VALUE(getCoulombFriction, index);
}

//==============================================================================
double CompositeJoint::computePotentialEnergy() const
{
}

//==============================================================================
Eigen::Vector6d CompositeJoint::getBodyConstraintWrench() const
{
}

//==============================================================================
CompositeJoint::CompositeJoint(const CompositeJoint::Properties& properties)
{
  // Inherited Aspects must be created in the final joint class or else we
  // get pure virtual function calls
  createCompositeJointAspect(properties);
  createJointAspect(properties);
}

//==============================================================================
Joint* CompositeJoint::clone() const
{
  return new CompositeJoint(getCompositeJointProperties());
}

//==============================================================================
void CompositeJoint::updateRelativeTransform() const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateRelativeTransform);
}

//==============================================================================
void CompositeJoint::updateRelativeSpatialVelocity() const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateRelativeSpatialVelocity);
}

//==============================================================================
void CompositeJoint::updateRelativeSpatialAcceleration() const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateRelativeSpatialAcceleration);
}

//==============================================================================
void CompositeJoint::updateRelativePrimaryAcceleration() const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateRelativePrimaryAcceleration);
}

//==============================================================================
void CompositeJoint::updateRelativeJacobian(bool mandatory) const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS_1(updateRelativeJacobian, mandatory);
}

//==============================================================================
void CompositeJoint::updateRelativeJacobianTimeDeriv() const
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateRelativeJacobianTimeDeriv);
}

//==============================================================================
void CompositeJoint::registerDofs()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(registerDofs);
}

//==============================================================================
void CompositeJoint::updateDegreeOfFreedomNames()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(updateDegreeOfFreedomNames);
}

//==============================================================================
const math::Jacobian CompositeJoint::getRelativeJacobian() const
{
  //
  // Spatial relative Jacobian in the coordinates of child BodyNode
  //
  //     |
  // J = | Ad(Tc*T(k,1), J1) | Ad(Tc*T(k,2), J2) | ...
  //     |
  //
  //                                          |
  //       ... | Ad(Tc*T(k,k-1), J(k-1)) | Jk |
  //                                          |
  //
  // where
  //
  //   Tc       : transform from child body to com￣posite joint
  //   T(i-1, i): relative transform of i-th sub-joint
  //   T(i, j)  : transform products from i-th sub-joint to j-th sub-joint that
  //              is T(i, i+1) * T(i+1, i+2) * ... * T(j-1, j)
  //   Ji       : relative Jacobian of i-th sub-joint (6 x DOF(i))
  //    k       : number of sub-joints (scalar)
  //

  math::Jacobian J = math::Jacobian::Zero(6, static_cast<int>(mNumDofs));

  int index = static_cast<int>(mNumDofs);
  Eigen::Isometry3d tf = Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
  for (std::size_t i = mJoints.size(); i--; )
  {
    const std::unique_ptr<Joint>& joint = mJoints[i];
    const int size = static_cast<int>(joint->getNumDofs());
    if (size == 0)
      continue;

    index -= size;
    const math::Jacobian localJ = joint->getRelativeJacobian();

    if (i != mJoints.size() - 1u)
    {
      const std::unique_ptr<Joint>& childJoint = mJoints[i+1];
      const Eigen::Isometry3d& childRelTf = childJoint->getRelativeTransform();
      tf = childRelTf * tf;
    }

    J.block(0, index, 6, size) = math::AdInvT(tf, localJ);
  }

  return J;
}

//==============================================================================
math::Jacobian CompositeJoint::getRelativeJacobian(
    const Eigen::VectorXd& /*positions*/) const
{
  // TODO(JS): Not implemented
  dterr << "[CompositeJoint] Not implemented\n";
  return math::Jacobian::Zero(6, static_cast<int>(mNumDofs));
}

//==============================================================================
const math::Jacobian CompositeJoint::getRelativeJacobianTimeDeriv() const
{
  // TODO(JS): Not implemented
  dterr << "[CompositeJoint] Not implemented\n";
  return math::Jacobian::Zero(6, static_cast<int>(mNumDofs));
}

//==============================================================================
void CompositeJoint::addVelocityTo(Eigen::Vector6d& vel)
{
  // Add relative velocity to vel
  vel.noalias() += getRelativeJacobian() * getVelocities();

  assert(!math::isNan(vel));
}

//==============================================================================
void CompositeJoint::setPartialAccelerationTo(
    Eigen::Vector6d& partialAcceleration, const Eigen::Vector6d& childVelocity)
{
  // ad(V, S * dq) + dS * dq
  const Eigen::VectorXd velocities = getVelocities();
  partialAcceleration
      = math::ad(childVelocity, getRelativeJacobian() * velocities)
        + getRelativeJacobianTimeDeriv() * velocities;

  assert(!math::isNan(partialAcceleration));
}

//==============================================================================
void CompositeJoint::addAccelerationTo(Eigen::Vector6d& acc)
{
  // Add joint acceleration to _acc
  acc.noalias() += getRelativeJacobian() * getAccelerations();

  assert(!math::isNan(acc));
}

//==============================================================================
void CompositeJoint::addVelocityChangeTo(Eigen::Vector6d& velocityChange)
{
  // Add joint velocity change to velocityChange
  velocityChange.noalias() += getRelativeJacobian() * mVelocityChanges;

  assert(!math::isNan(velocityChange));
}

//==============================================================================
void CompositeJoint::addChildArtInertiaTo(
    Eigen::Matrix6d& parentArtInertia, const Eigen::Matrix6d& childArtInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
    {
      // Child body's articulated inertia
      math::Jacobian AIS = childArtInertia * getRelativeJacobian();
      Eigen::Matrix6d PI = childArtInertia;
      PI.noalias() -= AIS * mInvProjArtInertia * AIS.transpose();
      assert(!math::isNan(PI));

      // Add child body's articulated inertia to parent body's articulated
      // inertia. Note that mT should be updated.
      parentArtInertia
          += math::transformInertia(this->getRelativeTransform().inverse(), PI);
      break;
    }
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
    {
      // Add child body's articulated inertia to parent body's articulated
      // inertia. Note that mT should be updated.
      parentArtInertia += math::transformInertia(
          this->getRelativeTransform().inverse(), childArtInertia);
      break;
    }
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildArtInertiaTo);
      break;
  }
}

//==============================================================================
void CompositeJoint::addChildArtInertiaImplicitTo(
    Eigen::Matrix6d& parentArtInertia, const Eigen::Matrix6d& childArtInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
    {
      // Child body's articulated inertia
      math::Jacobian AIS = childArtInertia * getRelativeJacobian();
      Eigen::Matrix6d PI = childArtInertia;
      PI.noalias() -= AIS * mInvProjArtInertiaImplicit * AIS.transpose();
      assert(!math::isNan(PI));

      // Add child body's articulated inertia to parent body's articulated inertia.
      // Note that mT should be updated.
      parentArtInertia
          += math::transformInertia(this->getRelativeTransform().inverse(), PI);
      break;
    }
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
    {
      // Add child body's articulated inertia to parent body's articulated inertia.
      // Note that mT should be updated.
      parentArtInertia += math::transformInertia(
          this->getRelativeTransform().inverse(), childArtInertia);
      break;
    }
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(addChildArtInertiaImplicitTo);
      break;
  }
}

//==============================================================================
void CompositeJoint::updateInvProjArtInertia(const Eigen::Matrix6d& artInertia)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
    {
      // Projected articulated inertia
      const math::Jacobian& Jacobian = getRelativeJacobian();
      const Eigen::MatrixXd projAI = Jacobian.transpose() * artInertia * Jacobian;

      // Inversion of projected articulated inertia
      mInvProjArtInertia = projAI.inverse();

      // Verification
      assert(!math::isNan(mInvProjArtInertia));
      break;
    }
    case Joint::ACCELERATION:
    case Joint::VELOCITY:
    case Joint::LOCKED:
      // Do nothing
      break;
    default:
      GenericJoint_REPORT_UNSUPPORTED_ACTUATOR(updateInvProjArtInertia);
      break;
  }
}

//==============================================================================
void CompositeJoint::updateInvProjArtInertiaImplicit(
    const Eigen::Matrix6d& artInertia, double timeStep)
{
  switch (Joint::mAspectProperties.mActuatorType)
  {
    case Joint::FORCE:
    case Joint::PASSIVE:
    case Joint::SERVO:
    case Joint::MIMIC:
    {
      // Projected articulated inertia
      const math::Jacobian& Jacobian = getRelativeJacobian();
      Eigen::MatrixXd projAI = Jacobian.transpose() * artInertia * Jacobian;

      // Add additional inertia for implicit damping and spring force
      projAI += (timeStep * Base::mAspectProperties.mDampingCoefficients
                 + timeStep * timeStep * Base::mAspectProperties.mSpringStiffnesses)
                    .asDiagonal();

      // Inversion of projected articulated inertia
      mInvProjArtInertiaImplicit = projAI.inverse();

      // Verification
      assert(!math::isNan(mInvProjArtInertiaImplicit));
      break;
    }
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
void CompositeJoint::addChildBiasForceTo(
    Eigen::Vector6d& parentBiasForce,
    const Eigen::Matrix6d& childArtInertia,
    const Eigen::Vector6d& childBiasForce,
    const Eigen::Vector6d& childPartialAcc)
{
}

//==============================================================================
void CompositeJoint::addChildBiasImpulseTo(
    Eigen::Vector6d& parentBiasImpulse,
    const Eigen::Matrix6d& childArtInertia,
    const Eigen::Vector6d& childBiasImpulse)
{
}

//==============================================================================
void CompositeJoint::updateTotalForce(
    const Eigen::Vector6d& bodyForce, double timeStep)
{
}

//==============================================================================
void CompositeJoint::updateTotalImpulse(const Eigen::Vector6d& bodyImpulse)
{
}

//==============================================================================
void CompositeJoint::resetTotalImpulses()
{
  COMPOSITE_JOINT_CALL_FUNCTIONS(resetTotalImpulses);
}

//==============================================================================
void CompositeJoint::updateAcceleration(
    const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& spatialAcc)
{
}

//==============================================================================
void CompositeJoint::updateVelocityChange(
    const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& velocityChange)
{
}

//==============================================================================
void CompositeJoint::updateForceID(
    const Eigen::Vector6d& bodyForce,
    double timeStep,
    bool withDampingForces,
    bool withSpringForces)
{
}

//==============================================================================
void CompositeJoint::updateForceFD(
    const Eigen::Vector6d& bodyForce,
    double timeStep,
    bool withDampingForces,
    bool withSpringForces)
{
}

//==============================================================================
void CompositeJoint::updateImpulseID(const Eigen::Vector6d& bodyImpulse)
{
}

//==============================================================================
void CompositeJoint::updateImpulseFD(const Eigen::Vector6d& bodyImpulse)
{
}

//==============================================================================
void CompositeJoint::updateConstrainedTerms(double timeStep)
{
}

//==============================================================================
void CompositeJoint::addChildBiasForceForInvMassMatrix(
    Eigen::Vector6d& parentBiasForce,
    const Eigen::Matrix6d& childArtInertia,
    const Eigen::Vector6d& childBiasForce)
{
}

//==============================================================================
void CompositeJoint::addChildBiasForceForInvAugMassMatrix(
    Eigen::Vector6d& parentBiasForce,
    const Eigen::Matrix6d& childArtInertia,
    const Eigen::Vector6d& childBiasForce)
{
}

//==============================================================================
void CompositeJoint::updateTotalForceForInvMassMatrix(
    const Eigen::Vector6d& bodyForce)
{
}

//==============================================================================
void CompositeJoint::getInvMassMatrixSegment(
    Eigen::MatrixXd& invMassMat,
    const std::size_t col,
    const Eigen::Matrix6d& artInertia,
    const Eigen::Vector6d& spatialAcc)
{
}

//==============================================================================
void CompositeJoint::getInvAugMassMatrixSegment(
    Eigen::MatrixXd& invMassMat,
    const std::size_t col,
    const Eigen::Matrix6d& artInertia,
    const Eigen::Vector6d& spatialAcc)
{
}

//==============================================================================
void CompositeJoint::addInvMassMatrixSegmentTo(Eigen::Vector6d& acc)
{
}

//==============================================================================
Eigen::VectorXd CompositeJoint::getSpatialToGeneralized(
    const Eigen::Vector6d& spatial)
{
}

} // namespace dynamics
} // namespace dart
