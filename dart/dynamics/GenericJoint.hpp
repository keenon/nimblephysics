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

#ifndef DART_DYNAMICS_GENERICJOINT_HPP_
#define DART_DYNAMICS_GENERICJOINT_HPP_

#include <array>
#include <string>

#include "dart/dynamics/detail/GenericJointAspect.hpp"

namespace dart {
namespace dynamics {

class DegreeOfFreedom;

template <class ConfigSpaceT>
class GenericJoint
  : public detail::GenericJointBase<GenericJoint<ConfigSpaceT>, ConfigSpaceT>
{
public:
  static constexpr std::size_t NumDofs = ConfigSpaceT::NumDofs;

  using ThisClass = GenericJoint<ConfigSpaceT>;
  using Base = detail::GenericJointBase<ThisClass, ConfigSpaceT>;

  using Point = typename ConfigSpaceT::Point;
  using EuclideanPoint = typename ConfigSpaceT::EuclideanPoint;
  using Vector = typename ConfigSpaceT::Vector;
  using JacobianMatrix = typename ConfigSpaceT::JacobianMatrix;
  using Matrix = typename ConfigSpaceT::Matrix;

  using UniqueProperties = detail::GenericJointUniqueProperties<ConfigSpaceT>;
  using Properties = detail::GenericJointProperties<ConfigSpaceT>;
  using AspectState = typename Base::AspectState;
  using AspectProperties = typename Base::AspectProperties;

  DART_BAKE_SPECIALIZED_ASPECT_IRREGULAR(
      typename ThisClass::Aspect, GenericJointAspect)

  GenericJoint(const ThisClass&) = delete;

  /// Destructor
  virtual ~GenericJoint();

  /// Set the Properties of this GenericJoint
  void setProperties(const Properties& properties);

  /// Set the Properties of this GenericJoint
  void setProperties(const UniqueProperties& properties);

  /// Set the AspectState of this GenericJoint
  void setAspectState(const AspectState& state);

  /// Set the AspectProperties of this GenericJoint
  void setAspectProperties(const AspectProperties& properties);

  /// Get the Properties of this GenericJoint
  Properties getGenericJointProperties() const;

  /// Copy the Properties of another GenericJoint
  void copy(const ThisClass& otherJoint);

  /// Copy the Properties of another GenericJoint
  void copy(const ThisClass* otherJoint);

  /// Same as copy(const GenericJoint&)
  ThisClass& operator=(const ThisClass& other);

  //----------------------------------------------------------------------------
  /// \{ \name Interface for generalized coordinates
  //----------------------------------------------------------------------------

  // Documentation inherited
  bool hasDof(const DegreeOfFreedom*) const override;

  // Documentation inherited
  DegreeOfFreedom* getDof(std::size_t index) override;

  // Documentation inherited
  const DegreeOfFreedom* getDof(std::size_t _index) const override;

  // Documentation inherited
  std::size_t getNumDofs() const override;

  // Documentation inherited
  const std::string& setDofName(
      std::size_t index,
      const std::string& name,
      bool preserveName = true) override;

  // Documentation inherited
  void preserveDofName(size_t index, bool preserve) override;

  // Documentation inherited
  bool isDofNamePreserved(size_t index) const override;

  // Documentation inherited
  const std::string& getDofName(size_t index) const override;

  // Documentation inherited
  size_t getIndexInSkeleton(size_t index) const override;

  // Documentation inherited
  size_t getIndexInTree(size_t index) const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Command
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setCommand(std::size_t index, s_t command) override;

  // Documentation inherited
  s_t getCommand(std::size_t index) const override;

  // Documentation inherited
  void setCommands(const Eigen::VectorXs& commands) override;

  // Documentation inherited
  Eigen::VectorXs getCommands() const override;

  // Documentation inherited
  void resetCommands() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Position
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setPosition(std::size_t index, s_t position) override;

  // Documentation inherited
  s_t getPosition(std::size_t index) const override;

  // Documentation inherited
  void setPositions(const Eigen::VectorXs& positions) override;

  // Documentation inherited
  Eigen::VectorXs getPositions() const override;

  // Documentation inherited
  void setPositionLowerLimit(std::size_t index, s_t position) override;

  // Documentation inherited
  s_t getPositionLowerLimit(std::size_t index) const override;

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
  bool hasPositionLimit(std::size_t index) const override;

  // Documentation inherited
  void resetPosition(std::size_t index) override;

  // Documentation inherited
  void resetPositions() override;

  // Documentation inherited
  void setInitialPosition(std::size_t index, s_t initial) override;

  // Documentation inherited
  s_t getInitialPosition(std::size_t index) const override;

  // Documentation inherited
  void setInitialPositions(const Eigen::VectorXs& initial) override;

  // Documentation inherited
  Eigen::VectorXs getInitialPositions() const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Fixed-size mutators and accessors
  //----------------------------------------------------------------------------

  // Note: The fixed-size versions of these functions exist to make it easier
  // to comply with the auto-updating design. Use these functions to avoid
  // accessing mPosition directly, that way it is easier to ensure that the
  // auto-updating design assumptions are being satisfied when reviewing the
  // code.

  /// Fixed-size version of setPositions()
  void setPositionsStatic(const Vector& positions);

  /// Fixed-size version of getPositions()
  const Vector& getPositionsStatic() const;

  /// Fixed-size version of setVelocities()
  void setVelocitiesStatic(const Vector& velocities);

  /// Fixed-size version of getVelocities()
  const Vector& getVelocitiesStatic() const;

  /// Fixed-size version of setAccelerations()
  void setAccelerationsStatic(const Vector& accels);

  /// Fixed-size version of getAccelerations()
  const Vector& getAccelerationsStatic() const;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Velocity
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocity(std::size_t index, s_t velocity) override;

  // Documentation inherited
  s_t getVelocity(std::size_t index) const override;

  // Documentation inherited
  void setVelocities(const Eigen::VectorXs& velocities) override;

  // Documentation inherited
  Eigen::VectorXs getVelocities() const override;

  // Documentation inherited
  void setVelocityLowerLimit(std::size_t index, s_t velocity) override;

  // Documentation inherited
  s_t getVelocityLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setVelocityLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getVelocityLowerLimits() const override;

  // Documentation inherited
  void setVelocityUpperLimit(std::size_t index, s_t velocity) override;

  // Documentation inherited
  s_t getVelocityUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setVelocityUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getVelocityUpperLimits() const override;

  // Documentation inherited
  void resetVelocity(std::size_t index) override;

  // Documentation inherited
  void resetVelocities() override;

  // Documentation inherited
  void setInitialVelocity(std::size_t index, s_t initial) override;

  // Documentation inherited
  s_t getInitialVelocity(std::size_t index) const override;

  // Documentation inherited
  void setInitialVelocities(const Eigen::VectorXs& initial) override;

  // Documentation inherited
  Eigen::VectorXs getInitialVelocities() const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Acceleration
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setAcceleration(std::size_t index, s_t acceleration) override;

  // Documentation inherited
  s_t getAcceleration(std::size_t index) const override;

  // Documentation inherited
  void setAccelerations(const Eigen::VectorXs& accelerations) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerations() const override;

  // Documentation inherited
  void setAccelerationLowerLimit(size_t index, s_t acceleration) override;

  // Documentation inherited
  s_t getAccelerationLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setAccelerationLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerationLowerLimits() const override;

  // Documentation inherited
  void setAccelerationUpperLimit(std::size_t index, s_t acceleration) override;

  // Documentation inherited
  s_t getAccelerationUpperLimit(std::size_t index) const override;

  // Documentation inherited
  void setAccelerationUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getAccelerationUpperLimits() const override;

  // Documentation inherited
  void resetAccelerations() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Force
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setForce(std::size_t index, s_t force) override;

  // Documentation inherited
  s_t getForce(std::size_t index) const override;

  // Documentation inherited
  void setForces(const Eigen::VectorXs& forces) override;

  // Documentation inherited
  Eigen::VectorXs getForces() const override;

  // Documentation inherited
  void setForceLowerLimit(size_t index, s_t force) override;

  // Documentation inherited
  s_t getForceLowerLimit(std::size_t index) const override;

  // Documentation inherited
  void setForceLowerLimits(const Eigen::VectorXs& lowerLimits) override;

  // Documentation inherited
  Eigen::VectorXs getForceLowerLimits() const override;

  // Documentation inherited
  void setForceUpperLimit(size_t index, s_t force) override;

  // Documentation inherited
  s_t getForceUpperLimit(size_t index) const override;

  // Documentation inherited
  void setForceUpperLimits(const Eigen::VectorXs& upperLimits) override;

  // Documentation inherited
  Eigen::VectorXs getForceUpperLimits() const override;

  // Documentation inherited
  void resetForces() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Velocity change
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocityChange(std::size_t index, s_t velocityChange) override;

  // Documentation inherited
  s_t getVelocityChange(std::size_t index) const override;

  // Documentation inherited
  void resetVelocityChanges() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Constraint impulse
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setConstraintImpulse(std::size_t index, s_t impulse) override;

  // Documentation inherited
  s_t getConstraintImpulse(std::size_t index) const override;

  // Documentation inherited
  void resetConstraintImpulses() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Integration and finite difference
  //----------------------------------------------------------------------------

  // Documentation inherited
  void integratePositions(s_t dt) override;

  // Documentation inherited
  void integrateVelocities(s_t dt) override;

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
      const Eigen::VectorXs& q2, const Eigen::VectorXs& q1) const override;

  /// Fixed-size version of getPositionDifferences()
  virtual Vector getPositionDifferencesStatic(
      const Vector& q2, const Vector& q1) const;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Passive forces - spring, viscous friction, Coulomb friction
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setSpringStiffness(std::size_t index, s_t k) override;

  // Documentation inherited
  s_t getSpringStiffness(std::size_t index) const override;

  // Documentation inherited
  void setRestPosition(std::size_t index, s_t q0) override;

  // Documentation inherited
  s_t getRestPosition(std::size_t index) const override;

  // Documentation inherited
  void setDampingCoefficient(std::size_t index, s_t coeff) override;

  // Documentation inherited
  s_t getDampingCoefficient(std::size_t index) const override;

  // Documentation inherited
  void setCoulombFriction(std::size_t index, s_t friction) override;

  // Documentation inherited
  s_t getCoulombFriction(std::size_t index) const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Energy
  //----------------------------------------------------------------------------

  // Documentation inherited
  s_t computePotentialEnergy() const override;

  /// \}

  // Documentation inherited
  Eigen::Vector6s getBodyConstraintWrench() const override;

  //----------------------------------------------------------------------------
  /// \{ \name Joint Jacobians
  //----------------------------------------------------------------------------

  // Documentation inherited
  const math::Jacobian getRelativeJacobian() const override;

  /// Fixed-size version of getRelativeJacobian()
  const typename GenericJoint<ConfigSpaceT>::JacobianMatrix&
  getRelativeJacobianStatic() const;

  // Documentation inherited
  math::Jacobian getRelativeJacobian(
      const Eigen::VectorXs& _positions) const override;

  /// Fixed-size version of getRelativeJacobian(positions)
  virtual JacobianMatrix getRelativeJacobianStatic(
      const Vector& positions) const = 0;

  // Documentation inherited
  const math::Jacobian getRelativeJacobianTimeDeriv() const override;

  /// Fixed-size version of getRelativeJacobianTimeDeriv()
  const JacobianMatrix& getRelativeJacobianTimeDerivStatic() const;

  /// Fixed-size version of getRelativeJacobianInPositionSpace()
  const typename GenericJoint<ConfigSpaceT>::JacobianMatrix&
  getRelativeJacobianInPositionSpaceStatic() const;

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame, in the `q` vector space. This is
  /// generally the same as the getRelativeJacobian() for the `dq` vector space,
  /// because `q` and `dq` are generally in the same vector space. However, for
  /// BallJoint and FreeJoint these are different values.
  virtual const math::Jacobian getRelativeJacobianInPositionSpace()
      const override;

  /// Fixed-size version of getRelativeJacobianInPositionSpace(positions)
  virtual JacobianMatrix getRelativeJacobianInPositionSpaceStatic(
      const Vector& positions) const;

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame, in the `q` vector space. This is
  /// generally the same as the getRelativeJacobian() for the `dq` vector space,
  /// because `q` and `dq` are generally in the same vector space. However, for
  /// BallJoint and FreeJoint these are different values.
  virtual math::Jacobian getRelativeJacobianInPositionSpace(
      const Eigen::VectorXs& positions) const override;

  /// Provide a default implementation to update the relative Jacobian
  virtual void updateRelativeJacobianInPositionSpace(
      bool mandatory = true) const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Skeleton Jacobians
  //----------------------------------------------------------------------------

  const Eigen::MatrixXs getSpatialVelocityJacobianWrtPosition();

  const Eigen::MatrixXs getSpatialAccelerationJacobianWrtPosition();

  const Eigen::MatrixXs getSpatialVelocityJacobianWrtVelocity();

  const Eigen::MatrixXs getSpatialAccelerationJacobianWrtVelocity();

  /// \}

protected:
  GenericJoint(const Properties& properties);

  // Documentation inherited
  void registerDofs() override;

  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  /// Get the inverse of the projected articulated inertia
  const Matrix& getInvProjArtInertia() const;

  /// Get the inverse of projected articulated inertia for implicit joint
  /// damping and spring forces
  const Matrix& getInvProjArtInertiaImplicit() const;

  // Documentation inherited
  void updateRelativeSpatialVelocity() const override;

  // Documentation inherited
  void updateRelativeSpatialAcceleration() const override;

  // Documentation inherited
  void updateRelativePrimaryAcceleration() const override;

  // Documentation inherited
  void addVelocityTo(Eigen::Vector6s& vel) override;

  // Documentation inherited
  void setPartialAccelerationTo(
      Eigen::Vector6s& partialAcceleration,
      const Eigen::Vector6s& childVelocity) override;

  // Documentation inherited
  void addAccelerationTo(Eigen::Vector6s& acc) override;

  // Documentation inherited
  void addVelocityChangeTo(Eigen::Vector6s& velocityChange) override;

  // Documentation inherited
  void addChildArtInertiaTo(
      Eigen::Matrix6s& parentArtInertia,
      const Eigen::Matrix6s& childArtInertia) override;

  // Documentation inherited
  void addChildArtInertiaImplicitTo(
      Eigen::Matrix6s& parentArtInertiaImplicit,
      const Eigen::Matrix6s& childArtInertiaImplicit) override;

  // Documentation inherited
  void updateInvProjArtInertia(const Eigen::Matrix6s& artInertia) override;

  // Documentation inherited
  void updateInvProjArtInertiaImplicit(
      const Eigen::Matrix6s& artInertia, s_t timeStep) override;

  // Documentation inherited
  void addChildBiasForceTo(
      Eigen::Vector6s& parentBiasForce,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasForce,
      const Eigen::Vector6s& childPartialAcc) override;

  // Documentation inherited
  void addChildBiasImpulseTo(
      Eigen::Vector6s& parentBiasImpulse,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasImpulse) override;

  // Documentation inherited
  void updateTotalForce(
      const Eigen::Vector6s& bodyForce, s_t timeStep) override;

  // Documentation inherited
  void updateTotalImpulse(const Eigen::Vector6s& bodyImpulse) override;

  // Documentation inherited
  void resetTotalImpulses() override;

  // Documentation inherited
  void updateAcceleration(
      const Eigen::Matrix6s& artInertia,
      const Eigen::Vector6s& spatialAcc) override;

  // Documentation inherited
  void updateVelocityChange(
      const Eigen::Matrix6s& artInertia,
      const Eigen::Vector6s& velocityChange) override;

  // Documentation inherited
  void updateForceID(
      const Eigen::Vector6s& bodyForce,
      s_t timeStep,
      bool withDampingForces,
      bool withSpringForces) override;

  // Documentation inherited
  void updateForceFD(
      const Eigen::Vector6s& bodyForce,
      s_t timeStep,
      bool withDampingForcese,
      bool withSpringForces) override;

  // Documentation inherited
  void updateImpulseID(const Eigen::Vector6s& bodyImpulse) override;

  // Documentation inherited
  void updateImpulseFD(const Eigen::Vector6s& bodyImpulse) override;

  // Documentation inherited
  void updateConstrainedTerms(s_t timeStep) override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Recursive algorithm routines for equations of motion
  //----------------------------------------------------------------------------

  // Documentation inherited
  void addChildBiasForceForInvMassMatrix(
      Eigen::Vector6s& parentBiasForce,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasForce) override;

  // Documentation inherited
  void addChildBiasForceForInvAugMassMatrix(
      Eigen::Vector6s& parentBiasForce,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasForce) override;

  // Documentation inherited
  void updateTotalForceForInvMassMatrix(
      const Eigen::Vector6s& bodyForce) override;

  // Documentation inherited
  void getInvMassMatrixSegment(
      Eigen::MatrixXs& invMassMat,
      const size_t col,
      const Eigen::Matrix6s& artInertia,
      const Eigen::Vector6s& spatialAcc) override;

  // Documentation inherited
  void getInvAugMassMatrixSegment(
      Eigen::MatrixXs& invMassMat,
      const size_t col,
      const Eigen::Matrix6s& artInertia,
      const Eigen::Vector6s& spatialAcc) override;

  // Documentation inherited
  void addInvMassMatrixSegmentTo(Eigen::Vector6s& acc) override;

  // Documentation inherited
  Eigen::VectorXs getSpatialToGeneralized(
      const Eigen::Vector6s& spatial) override;

  /// \}

protected:
  /// Array of DegreeOfFreedom objects
  std::array<DegreeOfFreedom*, NumDofs> mDofs;

  //----------------------------------------------------------------------------
  // Impulse
  //----------------------------------------------------------------------------

  /// Change of generalized velocity
  Vector mVelocityChanges;

  /// Generalized impulse
  Vector mImpulses;

  /// Generalized constraint impulse
  Vector mConstraintImpulses;

  //----------------------------------------------------------------------------
  // For recursive dynamics algorithms
  //----------------------------------------------------------------------------

  /// Spatial Jacobian expressed in the child body frame
  ///
  /// Do not use directly! Use getRelativeJacobianStatic() to access this
  /// quantity
  mutable JacobianMatrix mJacobian;

  /// Spatial Jacobian expressed in the child body frame
  ///
  /// Do not use directly! Use getRelativeJacobianInPositionSpaceStatic() to
  /// access this quantity
  mutable JacobianMatrix mJacobianInPositionSpace;

  /// Time derivative of spatial Jacobian expressed in the child body frame
  ///
  /// Do not use directly! Use getRelativeJacobianTimeDerivStatic() to access
  /// this quantity
  mutable JacobianMatrix mJacobianDeriv;

  /// Inverse of projected articulated inertia
  ///
  /// Do not use directly! Use getInvProjArtInertia() to get this quantity
  mutable Matrix mInvProjArtInertia;

  /// Inverse of projected articulated inertia for implicit joint damping and
  /// spring forces
  ///
  /// Do not use directly! Use getInvProjArtInertiaImplicit() to access this
  /// quantity
  mutable Matrix mInvProjArtInertiaImplicit;

  /// Total force projected on joint space
  Vector mTotalForce;

  /// Total impluse projected on joint space
  Vector mTotalImpulse;

  //----------------------------------------------------------------------------
  // For equations of motion
  //----------------------------------------------------------------------------

  ///
  Vector mInvM_a;

  ///
  Vector mInvMassMatrixSegment;

  //----------------------------------------------------------------------------
  /// \{ \name Differential Dynamics
  //----------------------------------------------------------------------------

  Eigen::VectorXs getAlpha() const override;
  math::Inertia computePi(const math::Inertia& AI) const override;
  Eigen::Vector6s computeBeta(
      const math::Inertia& AI, const Eigen::Vector6s& AB) const override;

  void computeJacobianOfMinvX_init() override;
  void computeJacobianOfMinvX_A(
      const math::Inertia& AI, const Eigen::Vector6s& AB) override;
  Eigen::MatrixXs computeJacobianOfMinvX_B(const math::Inertia& AI) override;

  std::size_t mNumSkeletonDofs;
  std::vector<Matrix> mInvM_Dpsi_Dq;
  Eigen::MatrixXs mInvM_Dalpha_Dq;
  Eigen::MatrixXs mInvM_DInvM_Dq;

  /// \}

private:
  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  void addChildArtInertiaToDynamic(
      Eigen::Matrix6s& parentArtInertia,
      const Eigen::Matrix6s& childArtInertia);

  void addChildArtInertiaToKinematic(
      Eigen::Matrix6s& parentArtInertia,
      const Eigen::Matrix6s& childArtInertia);

  void addChildArtInertiaImplicitToDynamic(
      Eigen::Matrix6s& parentArtInertia,
      const Eigen::Matrix6s& childArtInertia);

  void addChildArtInertiaImplicitToKinematic(
      Eigen::Matrix6s& parentArtInertia,
      const Eigen::Matrix6s& childArtInertia);

  void updateInvProjArtInertiaDynamic(const Eigen::Matrix6s& artInertia);

  void updateInvProjArtInertiaKinematic(const Eigen::Matrix6s& artInertia);

  void updateInvProjArtInertiaImplicitDynamic(
      const Eigen::Matrix6s& artInertia, s_t timeStep);

  void updateInvProjArtInertiaImplicitKinematic(
      const Eigen::Matrix6s& artInertia, s_t timeStep);

  void addChildBiasForceToDynamic(
      Eigen::Vector6s& parentBiasForce,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasForce,
      const Eigen::Vector6s& childPartialAcc);

  void addChildBiasForceToKinematic(
      Eigen::Vector6s& parentBiasForce,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasForce,
      const Eigen::Vector6s& childPartialAcc);

  void addChildBiasImpulseToDynamic(
      Eigen::Vector6s& parentBiasImpulse,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasImpulse);

  void addChildBiasImpulseToKinematic(
      Eigen::Vector6s& parentBiasImpulse,
      const Eigen::Matrix6s& childArtInertia,
      const Eigen::Vector6s& childBiasImpulse);

  void updateTotalForceDynamic(const Eigen::Vector6s& bodyForce, s_t timeStep);

  void updateTotalForceKinematic(
      const Eigen::Vector6s& bodyForce, s_t timeStep);

  void updateTotalImpulseDynamic(const Eigen::Vector6s& bodyImpulse);

  void updateTotalImpulseKinematic(const Eigen::Vector6s& bodyImpulse);

  void updateAccelerationDynamic(
      const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& spatialAcc);

  void updateAccelerationKinematic(
      const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& spatialAcc);

  void updateVelocityChangeDynamic(
      const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& velocityChange);

  void updateVelocityChangeKinematic(
      const Eigen::Matrix6s& artInertia, const Eigen::Vector6s& velocityChange);

  void updateConstrainedTermsDynamic(s_t timeStep);

  void updateConstrainedTermsKinematic(s_t timeStep);

  /// \}
};

} // namespace dynamics
} // namespace dart

#include "dart/dynamics/detail/GenericJoint.hpp"

#endif // DART_DYNAMICS_GENERICJOINT_HPP_
