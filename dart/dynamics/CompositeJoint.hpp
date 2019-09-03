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

#ifndef DART_DYNAMICS_COMPOSITEJOINT_HPP_
#define DART_DYNAMICS_COMPOSITEJOINT_HPP_

#include <iostream>
#include "dart/dynamics/Joint.hpp"

namespace dart {
namespace dynamics {

class CompositeJoint final : public Joint
{
public:
  struct Properties : Joint::Properties
  {
    Properties(const Joint::Properties& properties = Joint::Properties());
    ~Properties() override = default;

    std::vector<std::function<std::unique_ptr<Joint>()>> mJointCreators;

    template <typename JointPropertiesType>
    void addSubProperties(const JointPropertiesType& props)
    {
      mJointCreators
          .emplace_back([props]() -> std::unique_ptr<typename JointPropertiesType::JointType> {
            auto* newJoint = new typename JointPropertiesType::JointType(props);
            return std::unique_ptr<typename JointPropertiesType::JointType>(newJoint);
          });
    }
  };

  DART_BAKE_SPECIALIZED_ASPECT_IRREGULAR(Aspect, CompositeJointAspect)

  CompositeJoint(const CompositeJoint&) = delete;

  /// Destructor
  ~CompositeJoint() override = default;

  /// Returns the Properties of this WeldJoint
  Properties getCompositeJointProperties() const;

  // Documentation inherited
  const std::string& getType() const override;

  /// Returns joint type for this class
  static const std::string& getStaticType();

  //----------------------------------------------------------------------------
  // Joint composition
  //----------------------------------------------------------------------------

  template <class JointType>
  Joint* addJoint(const typename JointType::Properties& jointProps)
  {
    auto joint = std::make_unique<JointType>(jointProps);
    const std::size_t numDofs = joint->getNumDofs();

    // Update sub-joint indices and local indices
    mJointIndexMap.push_back(mJointIndexMap.size() + numDofs);
    mLocalDofIndexMap.reserve(mLocalDofIndexMap.size() + numDofs);
    for (auto i = 0u; i < numDofs; ++i)
    {
      mJointIndexMap.push_back(mJoints.size());
      mLocalDofIndexMap.push_back(i);
    }

    // Update total number of degrees-of-freedom
    mNumDofs += numDofs;

    mJoints.emplace_back(std::move(joint));

    return mJoints.back().get();
  }

  std::size_t getNumJoints() const;

  Joint* getJoint(std::size_t index);

  const Joint* getJoint(std::size_t index) const;

  //----------------------------------------------------------------------------

  // Documentation inherited
  bool isCyclic(std::size_t index) const override;

  //----------------------------------------------------------------------------
  // Interface for generalized coordinates
  //----------------------------------------------------------------------------

  // Documentation inherited
  DegreeOfFreedom* getDof(std::size_t index) override;

  // Documentation inherited
  const DegreeOfFreedom* getDof(std::size_t index) const override;

  // Documentation inherited
  const std::string& setDofName(std::size_t index, const std::string& name, bool preserveName = true) override;

  // Documentation inherited
  void preserveDofName(std::size_t index, bool preserve) override;

  // Documentation inherited
  bool isDofNamePreserved(std::size_t index) const override;

  // Documentation inherited
  const std::string& getDofName(std::size_t index) const override;

  // Documentation inherited
  std::size_t getNumDofs() const override;

  // Documentation inherited
  std::size_t getIndexInSkeleton(std::size_t index) const override;

  // Documentation inherited
  std::size_t getIndexInTree(std::size_t index) const override;

  //----------------------------------------------------------------------------
  // Command
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setCommand(std::size_t index, double command) override;

  // Documentation inherited
  double getCommand(std::size_t index) const override;

  // Documentation inherited
  void setCommands(const Eigen::VectorXd& commands) override;

  // Documentation inherited
  Eigen::VectorXd getCommands() const override;

  // Documentation inherited
  void resetCommands() override;

  //----------------------------------------------------------------------------
  // Position
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setPosition(std::size_t index, double position) override;

  // Documentation inherited
  double getPosition(std::size_t index) const override;

  // Documentation inherited
  void setPositions(const Eigen::VectorXd& positions) override;

  // Documentation inherited
  Eigen::VectorXd getPositions() const override;

  // Documentation inherited
  void setPositionLowerLimit(std::size_t index, double position) override;

  // Documentation inherited
  double getPositionLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setPositionLowerLimits(const Eigen::VectorXd& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXd getPositionLowerLimits() const override;

  // Documentation inherited
  void setPositionUpperLimit(std::size_t index, double position) override;

  // Documentation inherited
  double getPositionUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setPositionUpperLimits(const Eigen::VectorXd& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXd getPositionUpperLimits() const override;

  // Documentation inherited
  bool hasPositionLimit(std::size_t index) const override;

  // Documentation inherited
  void resetPosition(std::size_t index) override;

  // Documentation inherited
  void resetPositions() override;

  // Documentation inherited
  void setInitialPosition(std::size_t index, double initial) override;

  // Documentation inherited
  double getInitialPosition(std::size_t index) const override;

  // Documentation inherited
  void setInitialPositions(const Eigen::VectorXd& initial) override;

  // Documentation inherited
  Eigen::VectorXd getInitialPositions() const override;

  //----------------------------------------------------------------------------
  // Velocity
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocity(std::size_t index, double velocity) override;

  // Documentation inherited
  double getVelocity(std::size_t index) const override;

  // Documentation inherited
  void setVelocities(const Eigen::VectorXd& velocities) override;

  // Documentation inherited
  Eigen::VectorXd getVelocities() const override;

  // Documentation inherited
  void setVelocityLowerLimit(std::size_t index, double velocity) override;

  // Documentation inherited
  double getVelocityLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setVelocityLowerLimits(const Eigen::VectorXd& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXd getVelocityLowerLimits() const override;

  // Documentation inherited
  void setVelocityUpperLimit(std::size_t index, double velocity) override;

  // Documentation inherited
  double getVelocityUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setVelocityUpperLimits(const Eigen::VectorXd& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXd getVelocityUpperLimits() const override;

  // Documentation inherited
  void resetVelocity(std::size_t index) override;

  // Documentation inherited
  void resetVelocities() override;

  // Documentation inherited
  void setInitialVelocity(std::size_t index, double initial) override;

  // Documentation inherited
  double getInitialVelocity(std::size_t index) const override;

  // Documentation inherited
  void setInitialVelocities(const Eigen::VectorXd& initial) override;

  // Documentation inherited
  Eigen::VectorXd getInitialVelocities() const override;

  //----------------------------------------------------------------------------
  // Acceleration
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setAcceleration(std::size_t index, double acceleration) override;

  // Documentation inherited
  double getAcceleration(std::size_t index) const override;

  // Documentation inherited
  void setAccelerations(const Eigen::VectorXd& accelerations) override;

  // Documentation inherited
  Eigen::VectorXd getAccelerations() const override;

  // Documentation inherited
  void resetAccelerations() override;

  // Documentation inherited
  void setAccelerationLowerLimit(
      std::size_t index, double acceleration) override;

  // Documentation inherited
  double getAccelerationLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setAccelerationLowerLimits(const Eigen::VectorXd& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXd getAccelerationLowerLimits() const override;

  // Documentation inherited
  void setAccelerationUpperLimit(
      std::size_t index, double acceleration) override;

  // Documentation inherited
  double getAccelerationUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setAccelerationUpperLimits(const Eigen::VectorXd& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXd getAccelerationUpperLimits() const override;

  //----------------------------------------------------------------------------
  // Force
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setForce(std::size_t index, double force) override;

  // Documentation inherited
  double getForce(std::size_t index) const override;

  // Documentation inherited
  void setForces(const Eigen::VectorXd& forces) override;

  // Documentation inherited
  Eigen::VectorXd getForces() const override;

  // Documentation inherited
  void resetForces() override;

  // Documentation inherited
  void setForceLowerLimit(std::size_t index, double force) override;

  // Documentation inherited
  double getForceLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setForceLowerLimits(const Eigen::VectorXd& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXd getForceLowerLimits() const override;

  // Documentation inherited
  void setForceUpperLimit(std::size_t index, double force) override;

  // Documentation inherited
  double getForceUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setForceUpperLimits(const Eigen::VectorXd& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXd getForceUpperLimits() const override;

  //----------------------------------------------------------------------------
  // Velocity change
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocityChange(std::size_t index, double velocityChange) override;

  // Documentation inherited
  double getVelocityChange(std::size_t index) const override;

  // Documentation inherited
  void resetVelocityChanges() override;

  //----------------------------------------------------------------------------
  // Constraint impulse
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setConstraintImpulse(std::size_t index, double impulse) override;

  // Documentation inherited
  double getConstraintImpulse(std::size_t index) const override;

  // Documentation inherited
  void resetConstraintImpulses() override;

  //----------------------------------------------------------------------------
  // Integration and finite difference
  //----------------------------------------------------------------------------

  // Documentation inherited
  void integratePositions(double dt) override;

  // Documentation inherited
  void integrateVelocities(double dt) override;

  // Documentation inherited
  Eigen::VectorXd getPositionDifferences(
      const Eigen::VectorXd& q2, const Eigen::VectorXd& q1) const override;

  //----------------------------------------------------------------------------
  /// \{ \name Passive forces - spring, viscous friction, Coulomb friction
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setSpringStiffness(std::size_t index, double k) override;

  // Documentation inherited
  double getSpringStiffness(std::size_t index) const override;

  // Documentation inherited
  void setRestPosition(std::size_t index, double q0) override;

  // Documentation inherited
  double getRestPosition(std::size_t index) const override;

  // Documentation inherited
  void setDampingCoefficient(std::size_t index, double d) override;

  // Documentation inherited
  double getDampingCoefficient(std::size_t index) const override;

  // Documentation inherited
  void setCoulombFriction(std::size_t index, double friction) override;

  // Documentation inherited
  double getCoulombFriction(std::size_t index) const override;

  /// \}

  //----------------------------------------------------------------------------

  // Documentation inherited
  double computePotentialEnergy() const override;

  // Documentation inherited
  Eigen::Vector6d getBodyConstraintWrench() const override;

protected:
  /// Constructor called by inheriting classes
  CompositeJoint() = default;

  /// Constructor called by Skeleton class
  CompositeJoint(const CompositeJoint::Properties& properties);

  // Documentation inherited
  Joint* clone() const override;

  //----------------------------------------------------------------------------
  // Recursive algorithms
  //----------------------------------------------------------------------------

  // Documentation inherited
  void updateRelativeTransform() const override;

  // Documentation inherited
  void updateRelativeSpatialVelocity() const override;

  // Documentation inherited
  void updateRelativeSpatialAcceleration() const override;

  // Documentation inherited
  void updateRelativePrimaryAcceleration() const override;

  // Documentation inherited
  void updateRelativeJacobian(bool mandatory = true) const override;

  // Documentation inherited
  void updateRelativeJacobianTimeDeriv() const override;

  //----------------------------------------------------------------------------

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
      const Eigen::VectorXd& positions) const override;

  // Documentation inherited
  const math::Jacobian getRelativeJacobianTimeDeriv() const override;

  // Documentation inherited
  void addVelocityTo(Eigen::Vector6d& vel) override;

  // Documentation inherited
  void setPartialAccelerationTo(
      Eigen::Vector6d& partialAcceleration,
      const Eigen::Vector6d& childVelocity) override;

  // Documentation inherited
  void addAccelerationTo(Eigen::Vector6d& acc) override;

  // Documentation inherited
  void addVelocityChangeTo(Eigen::Vector6d& velocityChange) override;

  // Documentation inherited
  void addChildArtInertiaTo(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia) override;

  // Documentation inherited
  void addChildArtInertiaImplicitTo(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia) override;

  // Documentation inherited
  void updateInvProjArtInertia(const Eigen::Matrix6d& artInertia) override;

  // Documentation inherited
  void updateInvProjArtInertiaImplicit(
      const Eigen::Matrix6d& artInertia, double timeStep) override;

  // Documentation inherited
  void addChildBiasForceTo(
      Eigen::Vector6d& parentBiasForce,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasForce,
      const Eigen::Vector6d& childPartialAcc) override;

  // Documentation inherited
  void addChildBiasImpulseTo(
      Eigen::Vector6d& parentBiasImpulse,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasImpulse) override;

  // Documentation inherited
  void updateTotalForce(
      const Eigen::Vector6d& bodyForce, double timeStep) override;

  // Documentation inherited
  void updateTotalImpulse(const Eigen::Vector6d& bodyImpulse) override;

  // Documentation inherited
  void resetTotalImpulses() override;

  // Documentation inherited
  void updateAcceleration(
      const Eigen::Matrix6d& artInertia,
      const Eigen::Vector6d& spatialAcc) override;

  // Documentation inherited
  void updateVelocityChange(
      const Eigen::Matrix6d& artInertia,
      const Eigen::Vector6d& velocityChange) override;

  // Documentation inherited
  void updateForceID(
      const Eigen::Vector6d& bodyForce,
      double timeStep,
      bool withDampingForces,
      bool withSpringForces) override;

  // Documentation inherited
  void updateForceFD(
      const Eigen::Vector6d& bodyForce,
      double timeStep,
      bool withDampingForces,
      bool withSpringForces) override;

  // Documentation inherited
  void updateImpulseID(const Eigen::Vector6d& bodyImpulse) override;

  // Documentation inherited
  void updateImpulseFD(const Eigen::Vector6d& bodyImpulse) override;

  // Documentation inherited
  void updateConstrainedTerms(double timeStep) override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Recursive algorithm routines for equations of motion
  //----------------------------------------------------------------------------

  /// Adds child's bias force to parent's one
  void addChildBiasForceForInvMassMatrix(
      Eigen::Vector6d& parentBiasForce,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasForce) override;

  /// Adds child's bias force to parent's one
  void addChildBiasForceForInvAugMassMatrix(
      Eigen::Vector6d& parentBiasForce,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasForce) override;

  ///
  void updateTotalForceForInvMassMatrix(
      const Eigen::Vector6d& bodyForce) override;

  // Documentation inherited
  void getInvMassMatrixSegment(
      Eigen::MatrixXd& invMassMat,
      const std::size_t col,
      const Eigen::Matrix6d& artInertia,
      const Eigen::Vector6d& spatialAcc) override;

  // Documentation inherited
  void getInvAugMassMatrixSegment(
      Eigen::MatrixXd& invMassMat,
      const std::size_t col,
      const Eigen::Matrix6d& artInertia,
      const Eigen::Vector6d& spatialAcc) override;

  // Documentation inherited
  void addInvMassMatrixSegmentTo(Eigen::Vector6d& acc) override;

  // Documentation inherited
  Eigen::VectorXd getSpatialToGeneralized(
      const Eigen::Vector6d& spatial) override;

  /// \}

  Eigen::VectorXd mVelocityChanges;
  Eigen::VectorXd mImpulses;
  Eigen::VectorXd mConstraintImpulses;

private:
  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  void addChildArtInertiaToDynamic(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia);

  void addChildArtInertiaToKinematic(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia);

  void addChildArtInertiaImplicitToDynamic(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia);

  void addChildArtInertiaImplicitToKinematic(
      Eigen::Matrix6d& parentArtInertia,
      const Eigen::Matrix6d& childArtInertia);

  void updateInvProjArtInertiaDynamic(const Eigen::Matrix6d& artInertia);

  void updateInvProjArtInertiaKinematic(const Eigen::Matrix6d& artInertia);

  void updateInvProjArtInertiaImplicitDynamic(
      const Eigen::Matrix6d& artInertia, double timeStep);

  void updateInvProjArtInertiaImplicitKinematic(
      const Eigen::Matrix6d& artInertia, double timeStep);

  void addChildBiasForceToDynamic(
      Eigen::Vector6d& parentBiasForce,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasForce,
      const Eigen::Vector6d& childPartialAcc);

  void addChildBiasForceToKinematic(
      Eigen::Vector6d& parentBiasForce,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasForce,
      const Eigen::Vector6d& childPartialAcc);

  void addChildBiasImpulseToDynamic(
      Eigen::Vector6d& parentBiasImpulse,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasImpulse);

  void addChildBiasImpulseToKinematic(
      Eigen::Vector6d& parentBiasImpulse,
      const Eigen::Matrix6d& childArtInertia,
      const Eigen::Vector6d& childBiasImpulse);

  void updateTotalForceDynamic(
      const Eigen::Vector6d& bodyForce, double timeStep);

  void updateTotalForceKinematic(
      const Eigen::Vector6d& bodyForce, double timeStep);

  void updateTotalImpulseDynamic(const Eigen::Vector6d& bodyImpulse);

  void updateTotalImpulseKinematic(const Eigen::Vector6d& bodyImpulse);

  void updateAccelerationDynamic(
      const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& spatialAcc);

  void updateAccelerationKinematic(
      const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& spatialAcc);

  void updateVelocityChangeDynamic(
      const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& velocityChange);

  void updateVelocityChangeKinematic(
      const Eigen::Matrix6d& artInertia, const Eigen::Vector6d& velocityChange);

  void updateConstrainedTermsDynamic(double timeStep);

  void updateConstrainedTermsKinematic(double timeStep);

  /// \}

  std::vector<std::unique_ptr<Joint>> mJoints;
  std::size_t mNumDofs{0};
  std::vector<std::size_t> mJointIndexMap;
  std::vector<std::size_t> mLocalDofIndexMap;

  //----------------------------------------------------------------------------
  // For recursive dynamics algorithms
  //----------------------------------------------------------------------------

  /// Spatial Jacobian expressed in the child body frame
  ///
  /// Do not use directly! Use getRelativeJacobianStatic() to access this
  /// quantity
  mutable math::Jacobian mJacobian;

  /// Time derivative of spatial Jacobian expressed in the child body frame
  ///
  /// Do not use directly! Use getRelativeJacobianTimeDerivStatic() to access
  /// this quantity
  mutable math::Jacobian mJacobianDeriv;

  /// Inverse of projected articulated inertia
  ///
  /// Do not use directly! Use getInvProjArtInertia() to get this quantity
  mutable Eigen::MatrixXd mInvProjArtInertia;

  /// Inverse of projected articulated inertia for implicit joint damping and
  /// spring forces
  ///
  /// Do not use directly! Use getInvProjArtInertiaImplicit() to access this
  /// quantity
  mutable Eigen::MatrixXd mInvProjArtInertiaImplicit;

  /// Total force projected on joint space
  Eigen::VectorXd mTotalForce;

  /// Total impluse projected on joint space
  Eigen::VectorXd mTotalImpulse;
};

} // namespace dynamics
} // namespace dart

#endif // DART_DYNAMICS_COMPOSITEJOINT_HPP_
