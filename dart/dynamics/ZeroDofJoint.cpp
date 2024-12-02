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

#include "dart/dynamics/ZeroDofJoint.hpp"

#include "dart/common/Console.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
ZeroDofJoint::Properties::Properties(const Joint::Properties& _properties)
  : Joint::Properties(_properties)
{
  // Do nothing
}

//==============================================================================
ZeroDofJoint::~ZeroDofJoint()
{
  // Do nothing
}

//==============================================================================
ZeroDofJoint::Properties ZeroDofJoint::getZeroDofJointProperties() const
{
  return getJointProperties();
}

//==============================================================================
bool ZeroDofJoint::isFixed() const
{
  return true;
}

//==============================================================================
bool ZeroDofJoint::hasDof(const DegreeOfFreedom*) const
{
  return false;
}

//==============================================================================
DegreeOfFreedom* ZeroDofJoint::getDof(std::size_t)
{
  dterr << "[ZeroDofJoint::getDof] Attempting to get a DegreeOfFreedom from a "
        << "ZeroDofJoint. This is not allowed!\n";
  assert(false);
  return nullptr;
}

//==============================================================================
const DegreeOfFreedom* ZeroDofJoint::getDof(std::size_t) const
{
  dterr << "[ZeroDofJoint::getDof] Attempting to get a DegreeOfFreedom from a "
        << "ZeroDofJoint. This is not allowed!\n";
  assert(false);
  return nullptr;
}

//==============================================================================
const std::string& ZeroDofJoint::setDofName(
    std::size_t, const std::string&, bool)
{
  return emptyString;
}

//==============================================================================
void ZeroDofJoint::preserveDofName(std::size_t, bool)
{
  // Do nothing
}

//==============================================================================
bool ZeroDofJoint::isDofNamePreserved(std::size_t) const
{
  return false;
}

//==============================================================================
const std::string& ZeroDofJoint::getDofName(std::size_t) const
{
  return emptyString;
}

//==============================================================================
std::size_t ZeroDofJoint::getNumDofs() const
{
  return 0;
}

//==============================================================================
std::size_t ZeroDofJoint::getIndexInSkeleton(std::size_t _index) const
{
  dterr << "[ZeroDofJoint::getIndexInSkeleton] This function should never be "
        << "called (" << _index << ")!\n";
  assert(false);

  return 0;
}

//==============================================================================
std::size_t ZeroDofJoint::getIndexInTree(std::size_t _index) const
{
  dterr << "ZeroDofJoint::getIndexInTree] This function should never be "
        << "called (" << _index << ")!\n";
  assert(false);

  return 0;
}

//==============================================================================
void ZeroDofJoint::setCommand(std::size_t /*_index*/, s_t /*_command*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getCommand(std::size_t _index) const
{
  dterr << "[ZeroDofJoint::getCommand]: index[" << _index << "] out of range"
        << std::endl;

  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setCommands(const Eigen::VectorXs& /*_commands*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getCommands() const
{
  return Eigen::Matrix<s_t, 0, 1>();
}

//==============================================================================
void ZeroDofJoint::resetCommands()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setPosition(std::size_t, s_t)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getPosition(std::size_t _index) const
{
  dterr << "getPosition index[" << _index << "] out of range" << std::endl;

  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setPositions(const Eigen::VectorXs& /*_positions*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getPositions() const
{
  return Eigen::Matrix<s_t, 0, 1>();
}

//==============================================================================
void ZeroDofJoint::setPositionLowerLimit(
    std::size_t /*_index*/, s_t /*_position*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getPositionLowerLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setPositionLowerLimits(
    const Eigen::VectorXs& /*lowerLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getPositionLowerLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setPositionUpperLimit(
    std::size_t /*_index*/, s_t /*_position*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getPositionUpperLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setPositionUpperLimits(
    const Eigen::VectorXs& /*upperLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getPositionUpperLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
bool ZeroDofJoint::hasPositionLimit(std::size_t /*_index*/) const
{
  return true;
}

//==============================================================================
void ZeroDofJoint::resetPosition(std::size_t /*_index*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::resetPositions()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setInitialPosition(std::size_t /*_index*/, s_t /*_initial*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getInitialPosition(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setInitialPositions(const Eigen::VectorXs& /*_initial*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getInitialPositions() const
{
  return Eigen::VectorXs();
}

//==============================================================================
void ZeroDofJoint::setVelocity(std::size_t /*_index*/, s_t /*_velocity*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getVelocity(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setVelocities(const Eigen::VectorXs& /*_velocities*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getVelocities() const
{
  return Eigen::Matrix<s_t, 0, 1>();
}

//==============================================================================
void ZeroDofJoint::setVelocityLowerLimit(
    std::size_t /*_index*/, s_t /*_velocity*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getVelocityLowerLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setVelocityLowerLimits(
    const Eigen::VectorXs& /*lowerLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getVelocityLowerLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setVelocityUpperLimit(
    std::size_t /*_index*/, s_t /*_velocity*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getVelocityUpperLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setVelocityUpperLimits(
    const Eigen::VectorXs& /*upperLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getVelocityUpperLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::resetVelocity(std::size_t /*_index*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::resetVelocities()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setInitialVelocity(std::size_t /*_index*/, s_t /*_initial*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getInitialVelocity(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setInitialVelocities(const Eigen::VectorXs& /*_initial*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getInitialVelocities() const
{
  return Eigen::VectorXs();
}

//==============================================================================
void ZeroDofJoint::setAcceleration(
    std::size_t /*_index*/, s_t /*_acceleration*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getAcceleration(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setAccelerations(const Eigen::VectorXs& /*_accelerations*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getAccelerations() const
{
  return Eigen::Matrix<s_t, 0, 1>();
}

//==============================================================================
void ZeroDofJoint::resetAccelerations()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setAccelerationLowerLimit(
    std::size_t /*_index*/, s_t /*_acceleration*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getAccelerationLowerLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setAccelerationLowerLimits(
    const Eigen::VectorXs& /*lowerLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getAccelerationLowerLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setAccelerationUpperLimit(
    std::size_t /*_index*/, s_t /*_acceleration*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getAccelerationUpperLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setAccelerationUpperLimits(
    const Eigen::VectorXs& /*upperLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getAccelerationUpperLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setControlForce(std::size_t /*_index*/, s_t /*_force*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getControlForce(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setControlForces(const Eigen::VectorXs& /*_forces*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getControlForces() const
{
  return Eigen::Matrix<s_t, 0, 1>();
}

//==============================================================================
void ZeroDofJoint::resetControlForces()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setControlForceLowerLimit(
    std::size_t /*_index*/, s_t /*_force*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getControlForceLowerLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setControlForceLowerLimits(
    const Eigen::VectorXs& /*lowerLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getControlForceLowerLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setControlForceUpperLimit(
    std::size_t /*_index*/, s_t /*_force*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getControlForceUpperLimit(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setControlForceUpperLimits(
    const Eigen::VectorXs& /*upperLimits*/)
{
  // Do nothing
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getControlForceUpperLimits() const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setVelocityChange(
    std::size_t /*_index*/, s_t /*_velocityChange*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getVelocityChange(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::resetVelocityChanges()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setConstraintImpulse(
    std::size_t /*_index*/, s_t /*_impulse*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getConstraintImpulse(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::resetConstraintImpulses()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::integratePositions(s_t /*_dt*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::integrateVelocities(s_t /*_dt*/)
{
  // Do nothing
}

//==============================================================================
// Documentation inherited
Eigen::VectorXs ZeroDofJoint::integratePositionsExplicit(
    const Eigen::VectorXs& pos, const Eigen::VectorXs& /* vel */, s_t /* dt */)
{
  return pos;
}

//==============================================================================
/// Returns d/dpos of integratePositionsExplicit()
Eigen::MatrixXs ZeroDofJoint::getPosPosJacobian(
    const Eigen::VectorXs& /* pos */,
    const Eigen::VectorXs& /* vel */,
    s_t /* _dt */)
{
  return Eigen::MatrixXs::Zero(0, 0);
}

//==============================================================================
/// Returns d/dvel of integratePositionsExplicit()
Eigen::MatrixXs ZeroDofJoint::getVelPosJacobian(
    const Eigen::VectorXs& /* pos */,
    const Eigen::VectorXs& /* vel */,
    s_t /* _dt */)
{
  return Eigen::MatrixXs::Zero(0, 0);
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getPositionDifferences(
    const Eigen::VectorXs& /*_q2*/, const Eigen::VectorXs& /*_q1*/) const
{
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
void ZeroDofJoint::setSpringStiffness(std::size_t /*_index*/, s_t /*_k*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getSpringStiffness(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setRestPosition(std::size_t /*_index*/, s_t /*_q0*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getRestPosition(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setDampingCoefficient(std::size_t /*_index*/, s_t /*_d*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getDampingCoefficient(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
void ZeroDofJoint::setCoulombFriction(std::size_t /*_index*/, s_t /*_friction*/)
{
  // Do nothing
}

//==============================================================================
s_t ZeroDofJoint::getCoulombFriction(std::size_t /*_index*/) const
{
  return 0.0;
}

//==============================================================================
s_t ZeroDofJoint::computePotentialEnergy() const
{
  return 0.0;
}

//==============================================================================
ZeroDofJoint::ZeroDofJoint()
{
  // Do nothing. The Joint Aspect must be created by the most derived class.
}

//==============================================================================
void ZeroDofJoint::registerDofs()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateDegreeOfFreedomNames()
{
  // Do nothing
}

//==============================================================================
Eigen::Vector6s ZeroDofJoint::getBodyConstraintWrench() const
{
  assert(mChildBodyNode);
  return mChildBodyNode->getBodyForce();
}

//==============================================================================
const math::Jacobian ZeroDofJoint::getRelativeJacobian() const
{
  return Eigen::Matrix<s_t, 6, 0>();
}

//==============================================================================
math::Jacobian ZeroDofJoint::getRelativeJacobian(
    const Eigen::VectorXs& /*_positions*/) const
{
  return Eigen::Matrix<s_t, 6, 0>();
}

//==============================================================================
math::Jacobian ZeroDofJoint::getRelativeJacobianDerivWrtPosition(
    std::size_t /*index*/) const
{
  return Eigen::Matrix<s_t, 6, 0>();
}

//==============================================================================
const math::Jacobian ZeroDofJoint::getRelativeJacobianTimeDeriv() const
{
  return Eigen::Matrix<s_t, 6, 0>();
}

//==============================================================================
Eigen::Vector6s ZeroDofJoint::getWorldAxisScrewForPosition(int /* dof */) const
{
  assert(
      false
      && "You can't getWorldAxisScrew() on a joint with no degrees of freedom");
  return Eigen::Vector6s::Zero();
}

//==============================================================================
Eigen::Vector6s ZeroDofJoint::getWorldAxisScrewForVelocity(int /* dof */) const
{
  assert(
      false
      && "You can't getWorldAxisScrew() on a joint with no degrees of freedom");
  return Eigen::Vector6s::Zero();
}

//==============================================================================
const math::Jacobian ZeroDofJoint::getRelativeJacobianInPositionSpace() const
{
  return getRelativeJacobian();
}

//==============================================================================
math::Jacobian ZeroDofJoint::getRelativeJacobianInPositionSpace(
    const Eigen::VectorXs& _positions) const
{
  return getRelativeJacobian(_positions);
}

//==============================================================================
void ZeroDofJoint::updateRelativeJacobianInPositionSpace(
    bool /* mandatory */) const
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::addVelocityTo(Eigen::Vector6s& /*_vel*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::addVelocityChangeTo(Eigen::Vector6s& /*_velocityChange*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::setPartialAccelerationTo(
    Eigen::Vector6s& _partialAcceleration,
    const Eigen::Vector6s& /*_childVelocity*/)
{
  _partialAcceleration.setZero();
}

//==============================================================================
void ZeroDofJoint::addAccelerationTo(Eigen::Vector6s& /*_acc*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::addChildArtInertiaTo(
    Eigen::Matrix6s& _parentArtInertia, const Eigen::Matrix6s& _childArtInertia)
{
  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  _parentArtInertia += math::transformInertia(
      getRelativeTransform().inverse(), _childArtInertia);
}

//==============================================================================
void ZeroDofJoint::addChildArtInertiaImplicitTo(
    Eigen::Matrix6s& _parentArtInertia, const Eigen::Matrix6s& _childArtInertia)
{
  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  _parentArtInertia += math::transformInertia(
      getRelativeTransform().inverse(), _childArtInertia);
}

//==============================================================================
void ZeroDofJoint::updateInvProjArtInertia(
    const Eigen::Matrix6s& /*_artInertia*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateInvProjArtInertiaImplicit(
    const Eigen::Matrix6s& /*_artInertia*/, s_t /*_timeStep*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::addChildBiasForceTo(
    Eigen::Vector6s& _parentBiasForce,
    const Eigen::Matrix6s& _childArtInertia,
    const Eigen::Vector6s& _childBiasForce,
    const Eigen::Vector6s& _childPartialAcc)
{
  // Add child body's bias force to parent body's bias force. Note that mT
  // should be updated.
  _parentBiasForce += math::dAdInvT(
      getRelativeTransform(),
      _childBiasForce + _childArtInertia * _childPartialAcc);
}

//==============================================================================
void ZeroDofJoint::addChildBiasImpulseTo(
    Eigen::Vector6s& _parentBiasImpulse,
    const Eigen::Matrix6s& /*_childArtInertia*/,
    const Eigen::Vector6s& _childBiasImpulse)
{
  // Add child body's bias force to parent body's bias impulse. Note that mT
  // should be updated.
  _parentBiasImpulse
      += math::dAdInvT(getRelativeTransform(), _childBiasImpulse);
}

//==============================================================================
void ZeroDofJoint::updateTotalForce(
    const Eigen::Vector6s& /*_bodyForce*/, s_t /*_timeStep*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateTotalImpulse(const Eigen::Vector6s& /*_bodyImpulse*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::resetTotalImpulses()
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateAcceleration(
    const Eigen::Matrix6s& /*_artInertia*/,
    const Eigen::Vector6s& /*_spatialAcc*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateVelocityChange(
    const Eigen::Matrix6s& /*_artInertia*/,
    const Eigen::Vector6s& /*_velocityChange*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateForceID(
    const Eigen::Vector6s& /*_bodyForce*/,
    s_t /*_timeStep*/,
    bool /*_withDampingForces*/,
    bool /*_withSpringForces*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateForceFD(
    const Eigen::Vector6s& /*_bodyForce*/,
    s_t /*_timeStep*/,
    bool /*_withDampingForces*/,
    bool /*_withSpringForces*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateImpulseID(const Eigen::Vector6s& /*_bodyImpulse*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateImpulseFD(const Eigen::Vector6s& /*_bodyImpulse*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::updateConstrainedTerms(s_t /*_timeStep*/)
{
  // Do nothing
}

//==============================================================================
void ZeroDofJoint::addChildBiasForceForInvMassMatrix(
    Eigen::Vector6s& /*_parentBiasForce*/,
    const Eigen::Matrix6s& /*_childArtInertia*/,
    const Eigen::Vector6s& /*_childBiasForce*/)
{
  // TODO(JS)
}

//==============================================================================
void ZeroDofJoint::addChildBiasForceForInvAugMassMatrix(
    Eigen::Vector6s& /*_parentBiasForce*/,
    const Eigen::Matrix6s& /*_childArtInertia*/,
    const Eigen::Vector6s& /*_childBiasForce*/)
{
  // TODO(JS)
}

//==============================================================================
void ZeroDofJoint::updateTotalForceForInvMassMatrix(
    const Eigen::Vector6s& /*_bodyForce*/)
{
  // TODO(JS)
}

//==============================================================================
void ZeroDofJoint::getInvMassMatrixSegment(
    Eigen::MatrixXs& /*_invMassMat*/,
    const std::size_t /*_col*/,
    const Eigen::Matrix6s& /*_artInertia*/,
    const Eigen::Vector6s& /*_spatialAcc*/)
{
  // TODO(JS)
}

//==============================================================================
void ZeroDofJoint::getInvAugMassMatrixSegment(
    Eigen::MatrixXs& /*_invMassMat*/,
    const std::size_t /*_col*/,
    const Eigen::Matrix6s& /*_artInertia*/,
    const Eigen::Vector6s& /*_spatialAcc*/)
{
  // TODO(JS)
}

//==============================================================================
void ZeroDofJoint::addInvMassMatrixSegmentTo(Eigen::Vector6s& /*_acc*/)
{
  // TODO(JS)
}

//==============================================================================
Eigen::VectorXs ZeroDofJoint::getSpatialToGeneralized(
    const Eigen::Vector6s& /*_spatial*/)
{
  // Return zero size vector
  return Eigen::VectorXs::Zero(0);
}

//==============================================================================
/// Returns the value for q that produces the nearest rotation to
/// `relativeRotation` passed in.
Eigen::VectorXs ZeroDofJoint::getNearestPositionToDesiredRotation(
    const Eigen::Matrix3s& relativeRotation)
{
  (void)relativeRotation;
  // Return zero size vector
  return Eigen::VectorXs::Zero(0);
}

} // namespace dynamics
} // namespace dart
