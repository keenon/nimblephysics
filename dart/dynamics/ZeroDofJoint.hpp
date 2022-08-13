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

#ifndef DART_DYNAMICS_ZERODOFJOINT_HPP_
#define DART_DYNAMICS_ZERODOFJOINT_HPP_

#include <string>

#include "dart/dynamics/Joint.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

class BodyNode;
class Skeleton;

/// class ZeroDofJoint
class ZeroDofJoint : public Joint
{
public:
  struct Properties : Joint::Properties
  {
    Properties(const Joint::Properties& _properties = Joint::Properties());
    virtual ~Properties() = default;
  };

  ZeroDofJoint(const ZeroDofJoint&) = delete;

  /// Destructor
  virtual ~ZeroDofJoint();

  /// Get the Properties of this ZeroDofJoint
  Properties getZeroDofJointProperties() const;

  //----------------------------------------------------------------------------
  // Interface for generalized coordinates
  //----------------------------------------------------------------------------

  // Documentation inherited
  bool hasDof(const DegreeOfFreedom*) const override;

  // Documentation inherited
  DegreeOfFreedom* getDof(std::size_t) override;

  // Documentation inherited
  const DegreeOfFreedom* getDof(std::size_t) const override;

  // Documentation inherited
  const std::string& setDofName(std::size_t, const std::string&, bool) override;

  // Documentation inherited
  void preserveDofName(std::size_t, bool) override;

  // Documentation inherited
  bool isDofNamePreserved(std::size_t) const override;

  const std::string& getDofName(std::size_t) const override;

  // Documentation inherited
  std::size_t getNumDofs() const override;

  // Documentation inherited
  std::size_t getIndexInSkeleton(std::size_t _index) const override;

  // Documentation inherited
  std::size_t getIndexInTree(std::size_t _index) const override;

  //----------------------------------------------------------------------------
  // Command
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setCommand(std::size_t _index, s_t _command) override;

  // Documentation inherited
  s_t getCommand(std::size_t _index) const override;

  // Documentation inherited
  void setCommands(const Eigen::VectorXs& _commands) override;

  // Documentation inherited
  Eigen::VectorXs getCommands() const override;

  // Documentation inherited
  void resetCommands() override;

  //----------------------------------------------------------------------------
  // Position
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setPosition(std::size_t, s_t) override;

  // Documentation inherited
  s_t getPosition(std::size_t _index) const override;

  // Documentation inherited
  void setPositions(const Eigen::VectorXs& _positions) override;

  // Documentation inherited
  Eigen::VectorXs getPositions() const override;

  // Documentation inherited
  void setPositionLowerLimit(std::size_t _index, s_t _position) override;

  // Documentation inherited
  s_t getPositionLowerLimit(std::size_t _index) const override;

  // Documentation inherited
  void setPositionLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getPositionLowerLimits() const override;

  // Documentation inherited
  void setPositionUpperLimit(std::size_t index, s_t position) override;

  // Documentation inherited
  s_t getPositionUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setPositionUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getPositionUpperLimits() const override;

  // Documentation inherited
  bool hasPositionLimit(std::size_t _index) const override;

  // Documentation inherited
  void resetPosition(std::size_t _index) override;

  // Documentation inherited
  void resetPositions() override;

  // Documentation inherited
  void setInitialPosition(std::size_t _index, s_t _initial) override;

  // Documentation inherited
  s_t getInitialPosition(std::size_t _index) const override;

  // Documentation inherited
  void setInitialPositions(const Eigen::VectorXs& _initial) override;

  // Documentation inherited
  Eigen::VectorXs getInitialPositions() const override;

  //----------------------------------------------------------------------------
  // Velocity
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocity(std::size_t _index, s_t _velocity) override;

  // Documentation inherited
  s_t getVelocity(std::size_t _index) const override;

  // Documentation inherited
  void setVelocities(const Eigen::VectorXs& _velocities) override;

  // Documentation inherited
  Eigen::VectorXs getVelocities() const override;

  // Documentation inherited
  void setVelocityLowerLimit(std::size_t _index, s_t _velocity) override;

  // Documentation inherited
  s_t getVelocityLowerLimit(std::size_t _index) const override;

  // Documentation inherited
  void setVelocityLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getVelocityLowerLimits() const override;

  // Documentation inherited
  void setVelocityUpperLimit(std::size_t _index, s_t _velocity) override;

  // Documentation inherited
  s_t getVelocityUpperLimit(std::size_t _index) const override;

  // Documentation inherited
  void setVelocityUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getVelocityUpperLimits() const override;

  // Documentation inherited
  void resetVelocity(std::size_t _index) override;

  // Documentation inherited
  void resetVelocities() override;

  // Documentation inherited
  void setInitialVelocity(std::size_t _index, s_t _initial) override;

  // Documentation inherited
  s_t getInitialVelocity(std::size_t _index) const override;

  // Documentation inherited
  void setInitialVelocities(const Eigen::VectorXs& _initial) override;

  // Documentation inherited
  Eigen::VectorXs getInitialVelocities() const override;

  //----------------------------------------------------------------------------
  // Acceleration
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setAcceleration(std::size_t _index, s_t _acceleration) override;

  // Documentation inherited
  s_t getAcceleration(std::size_t _index) const override;

  // Documentation inherited
  void setAccelerations(const Eigen::VectorXs& _accelerations) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerations() const override;

  // Documentation inherited
  void resetAccelerations() override;

  // Documentation inherited
  void setAccelerationLowerLimit(
      std::size_t _index, s_t _acceleration) override;

  // Documentation inherited
  s_t getAccelerationLowerLimit(std::size_t _index) const override;

  // Documentation inherited
  void setAccelerationLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerationLowerLimits() const override;

  // Documentation inherited
  void setAccelerationUpperLimit(
      std::size_t _index, s_t _acceleration) override;

  // Documentation inherited
  s_t getAccelerationUpperLimit(std::size_t _index) const override;

  // Documentation inherited
  void setAccelerationUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerationUpperLimits() const override;

  //----------------------------------------------------------------------------
  // Force
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setControlForce(std::size_t _index, s_t _force) override;

  // Documentation inherited
  s_t getControlForce(std::size_t _index) const override;

  // Documentation inherited
  void setControlForces(const Eigen::VectorXs& _forces) override;

  // Documentation inherited
  Eigen::VectorXs getControlForces() const override;

  // Documentation inherited
  void resetControlForces() override;

  // Documentation inherited
  void setControlForceLowerLimit(std::size_t _index, s_t _force) override;

  // Documentation inherited
  s_t getControlForceLowerLimit(std::size_t _index) const override;

  // Documentation inherited
  void setControlForceLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getControlForceLowerLimits() const override;

  // Documentation inherited
  void setControlForceUpperLimit(std::size_t _index, s_t _force) override;

  // Documentation inherited
  s_t getControlForceUpperLimit(std::size_t _index) const override;

  // Documentation inherited
  void setControlForceUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getControlForceUpperLimits() const override;

  //----------------------------------------------------------------------------
  // Velocity change
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocityChange(std::size_t _index, s_t _velocityChange) override;

  // Documentation inherited
  s_t getVelocityChange(std::size_t _index) const override;

  // Documentation inherited
  void resetVelocityChanges() override;

  //----------------------------------------------------------------------------
  // Constraint impulse
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setConstraintImpulse(std::size_t _index, s_t _impulse) override;

  // Documentation inherited
  s_t getConstraintImpulse(std::size_t _index) const override;

  // Documentation inherited
  void resetConstraintImpulses() override;

  //----------------------------------------------------------------------------
  // Integration and finite difference
  //----------------------------------------------------------------------------

  // Documentation inherited
  void integratePositions(s_t _dt) override;

  // Documentation inherited
  void integrateVelocities(s_t _dt) override;

  // Documentation inherited
  Eigen::VectorXs integratePositionsExplicit(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t dt) override;

  /// Returns d/dpos of integratePositionsExplicit()
  Eigen::MatrixXs getPosPosJacobian(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t _dt) override;

  /// Returns d/dvel of integratePositionsExplicit()
  Eigen::MatrixXs getVelPosJacobian(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t _dt) override;

  // Documentation inherited
  Eigen::VectorXs getPositionDifferences(
      const Eigen::VectorXs& _q2, const Eigen::VectorXs& _q1) const override;

  //----------------------------------------------------------------------------
  /// \{ \name Passive forces - spring, viscous friction, Coulomb friction
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setSpringStiffness(std::size_t _index, s_t _k) override;

  // Documentation inherited
  s_t getSpringStiffness(std::size_t _index) const override;

  // Documentation inherited
  void setRestPosition(std::size_t _index, s_t _q0) override;

  // Documentation inherited
  s_t getRestPosition(std::size_t _index) const override;

  // Documentation inherited
  void setDampingCoefficient(std::size_t _index, s_t _d) override;

  // Documentation inherited
  s_t getDampingCoefficient(std::size_t _index) const override;

  // Documentation inherited
  void setCoulombFriction(std::size_t _index, s_t _friction) override;

  // Documentation inherited
  s_t getCoulombFriction(std::size_t _index) const override;

  /// \}

  //----------------------------------------------------------------------------

  // Documentation inherited
  s_t computePotentialEnergy() const override;

  // Documentation inherited
  Eigen::Vector6s getBodyConstraintWrench() const override;

protected:
  /// Constructor called by inheriting classes
  ZeroDofJoint();

  // Documentation inherited
  void registerDofs() override;

  // Documentation inherited
  void updateDegreeOfFreedomNames() override;

  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  // Documentation inherited
  const math::Jacobian getRelativeJacobian() const override;

  // Documentation inherited
  math::Jacobian getRelativeJacobian(
      const Eigen::VectorXs& _positions) const override;

  // Documentation inherited
  math::Jacobian getRelativeJacobianDerivWrtPosition(
      std::size_t index) const override;

  // Documentation inherited
  const math::Jacobian getRelativeJacobianTimeDeriv() const override;

  // Documentation inherited
  Eigen::Vector6s getWorldAxisScrewForPosition(int dof) const override;

  // Documentation inherited
  Eigen::Vector6s getWorldAxisScrewForVelocity(int dof) const override;

  // Documentation inherited
  const math::Jacobian getRelativeJacobianInPositionSpace() const override;

  // Documentation inherited
  math::Jacobian getRelativeJacobianInPositionSpace(
      const Eigen::VectorXs& _positions) const override;

  // Documentation inherited
  void updateRelativeJacobianInPositionSpace(
      bool mandatory = true) const override;

  // Documentation inherited
  void addVelocityTo(Eigen::Vector6s& _vel) override;

  // Documentation inherited
  void setPartialAccelerationTo(
      Eigen::Vector6s& _partialAcceleration,
      const Eigen::Vector6s& _childVelocity) override;

  // Documentation inherited
  void addAccelerationTo(Eigen::Vector6s& _acc) override;

  // Documentation inherited
  void addVelocityChangeTo(Eigen::Vector6s& _velocityChange) override;

  // Documentation inherited
  void addChildArtInertiaTo(
      Eigen::Matrix6s& _parentArtInertia,
      const Eigen::Matrix6s& _childArtInertia) override;

  // Documentation inherited
  void addChildArtInertiaImplicitTo(
      Eigen::Matrix6s& _parentArtInertia,
      const Eigen::Matrix6s& _childArtInertia) override;

  // Documentation inherited
  void updateInvProjArtInertia(const Eigen::Matrix6s& _artInertia) override;

  // Documentation inherited
  void updateInvProjArtInertiaImplicit(
      const Eigen::Matrix6s& _artInertia, s_t _timeStep) override;

  // Documentation inherited
  void addChildBiasForceTo(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce,
      const Eigen::Vector6s& _childPartialAcc) override;

  // Documentation inherited
  void addChildBiasImpulseTo(
      Eigen::Vector6s& _parentBiasImpulse,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasImpulse) override;

  // Documentation inherited
  void updateTotalForce(
      const Eigen::Vector6s& _bodyForce, s_t _timeStep) override;

  // Documentation inherited
  void updateTotalImpulse(const Eigen::Vector6s& _bodyImpulse) override;

  // Documentation inherited
  void resetTotalImpulses() override;

  // Documentation inherited
  void updateAcceleration(
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _spatialAcc) override;

  // Documentation inherited
  void updateVelocityChange(
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _velocityChange) override;

  // Documentation inherited
  void updateForceID(
      const Eigen::Vector6s& _bodyForce,
      s_t _timeStep,
      bool _withDampingForces,
      bool _withSpringForces) override;

  // Documentation inherited
  void updateForceFD(
      const Eigen::Vector6s& _bodyForce,
      s_t _timeStep,
      bool _withDampingForces,
      bool _withSpringForces) override;

  // Documentation inherited
  void updateImpulseID(const Eigen::Vector6s& _bodyImpulse) override;

  // Documentation inherited
  void updateImpulseFD(const Eigen::Vector6s& _bodyImpulse) override;

  // Documentation inherited
  void updateConstrainedTerms(s_t _timeStep) override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Recursive algorithm routines for equations of motion
  //----------------------------------------------------------------------------

  /// Add child's bias force to parent's one
  void addChildBiasForceForInvMassMatrix(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce) override;

  /// Add child's bias force to parent's one
  void addChildBiasForceForInvAugMassMatrix(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce) override;

  ///
  void updateTotalForceForInvMassMatrix(
      const Eigen::Vector6s& _bodyForce) override;

  // Documentation inherited
  void getInvMassMatrixSegment(
      Eigen::MatrixXs& _invMassMat,
      const std::size_t _col,
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _spatialAcc) override;

  // Documentation inherited
  void getInvAugMassMatrixSegment(
      Eigen::MatrixXs& _invMassMat,
      const std::size_t _col,
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _spatialAcc) override;

  // Documentation inherited
  void addInvMassMatrixSegmentTo(Eigen::Vector6s& _acc) override;

  // Documentation inherited
  Eigen::VectorXs getSpatialToGeneralized(
      const Eigen::Vector6s& _spatial) override;

  /// \}

private:
  /// Used by getDofName()
  const std::string emptyString;
};

} // namespace dynamics
} // namespace dart

#endif // DART_DYNAMICS_ZERODOFJOINT_HPP_
