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

#ifndef DART_DYNAMICS_JOINT_HPP_
#define DART_DYNAMICS_JOINT_HPP_

#include <memory>
#include <string>
#include <vector>

#include "dart/common/Deprecated.hpp"
#include "dart/common/EmbeddedAspect.hpp"
#include "dart/common/Subject.hpp"
#include "dart/common/VersionCounter.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/dynamics/detail/JointAspect.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

class BodyNode;
class Skeleton;
class DegreeOfFreedom;

/// class Joint
class Joint : public virtual common::Subject,
              public virtual common::VersionCounter,
              public common::EmbedProperties<Joint, detail::JointProperties>
{
public:
  using CompositeProperties = common::Composite::Properties;
  using Properties = detail::JointProperties;

  typedef detail::ActuatorType ActuatorType;
  static constexpr ActuatorType FORCE = detail::FORCE;
  static constexpr ActuatorType PASSIVE = detail::PASSIVE;
  static constexpr ActuatorType SERVO = detail::SERVO;
  static constexpr ActuatorType MIMIC = detail::MIMIC;
  static constexpr ActuatorType ACCELERATION = detail::ACCELERATION;
  static constexpr ActuatorType VELOCITY = detail::VELOCITY;
  static constexpr ActuatorType LOCKED = detail::LOCKED;

  DART_BAKE_SPECIALIZED_ASPECT_IRREGULAR(Aspect, JointAspect)

  struct ExtendedProperties : Properties
  {
    /// Composed constructor
    ExtendedProperties(
        const Properties& standardProperties = Properties(),
        const CompositeProperties& aspectProperties = CompositeProperties());

    /// Composed move constructor
    ExtendedProperties(
        Properties&& standardProperties,
        CompositeProperties&& aspectProperties);

    /// Properties of all the Aspects attached to this Joint
    CompositeProperties mCompositeProperties;
  };

  /// Default actuator type
  static const ActuatorType DefaultActuatorType;

  Joint(const Joint&) = delete;

  /// Destructor
  virtual ~Joint();

  /// Set the Properties of this Joint
  void setProperties(const Properties& properties);

  /// Set the AspectProperties of this Joint
  void setAspectProperties(const AspectProperties& properties);

  /// Get the Properties of this Joint
  const Properties& getJointProperties() const;

  /// Copy the properties of another Joint
  void copy(const Joint& _otherJoint);

  /// Copy the properties of another Joint
  void copy(const Joint* _otherJoint);

  /// Same as copy(const Joint&)
  Joint& operator=(const Joint& _otherJoint);

  /// \brief Set joint name and return the name.
  /// \param[in] _name The specified joint name to be set.
  /// \param[in] _renameDofs If true, the names of the joint's degrees of
  /// freedom will be updated by calling updateDegreeOfFreedomNames().
  ///
  /// If the name is already taken, this will return an altered version which
  /// will be used by the Skeleton. Otherwise, return _name.
  const std::string& setName(const std::string& _name, bool _renameDofs = true);

  /// Get joint name
  const std::string& getName() const;

  /// Gets a string representing the joint type
  virtual const std::string& getType() const = 0;

  /// Set actuator type
  void setActuatorType(ActuatorType _actuatorType);

  /// Get actuator type
  ActuatorType getActuatorType() const;

  /// Set mimic joint
  void setMimicJoint(
      const Joint* _mimicJoint,
      s_t _mimicMultiplier = 1.0,
      s_t _mimicOffset = 0.0);

  /// Get mimic joint
  const Joint* getMimicJoint() const;

  /// Get mimic joint multiplier
  s_t getMimicMultiplier() const;

  /// Get mimic joint offset
  s_t getMimicOffset() const;

  /// Return true if this joint is kinematic joint.
  ///
  /// Kinematic joint means the motion is prescribed by position or velocity or
  /// acceleration, which is determined by the actuator type.
  /// ACCELERATION/VELOCITY/LOCKED are kinematic joints while
  /// FORCE/PASSIVE/SERVO are dynamic joints.
  bool isKinematic() const;

  /// Return true if this joint is dynamic joint.
  bool isDynamic() const;

  /// Get the child BodyNode of this Joint
  BodyNode* getChildBodyNode();

  /// Get the (const) child BodyNode of this Joint
  const BodyNode* getChildBodyNode() const;

  /// Get the parent BodyNode of this Joint
  BodyNode* getParentBodyNode();

  /// Get the (const) parent BodyNode of this Joint
  const BodyNode* getParentBodyNode() const;

  /// Get the Skeleton that this Joint belongs to
  SkeletonPtr getSkeleton();

  /// Get the (const) Skeleton that this Joint belongs to.
  std::shared_ptr<const Skeleton> getSkeleton() const;

  /// Set transformation from parent body node to this joint
  virtual void setTransformFromParentBodyNode(const Eigen::Isometry3s& _T);

  /// Set transformation from child body node to this joint
  virtual void setTransformFromChildBodyNode(const Eigen::Isometry3s& _T);

  /// Get transformation from parent body node to this joint
  const Eigen::Isometry3s& getTransformFromParentBodyNode() const;

  /// Get transformation from child body node to this joint
  const Eigen::Isometry3s& getTransformFromChildBodyNode() const;

  /// Copy the transfromFromParentNode and transfromFromChildNode, and their
  /// scales, from another joint
  void copyTransformsFrom(const dynamics::Joint* other);

  /// Set the scale of the child body
  void setChildScale(s_t scale);

  /// Set the scale of the parent body
  void setParentScale(s_t scale);

  /// Get the scale of the child body
  s_t getChildScale() const;

  /// Get the scale of the parent body
  s_t getParentScale() const;

  /// Set to enforce joint position limit
  ///
  /// The joint position limit is valid when the actutor type is one of
  /// PASSIVE/FORCE.
  ///
  /// \sa ActuatorType
  void setPositionLimitEnforced(bool _isPositionLimitEnforced);

  /// Get whether enforcing joint position limit
  ///
  /// The joint position limit is valid when the actutor type is one of
  /// PASSIVE/FORCE.
  ///
  /// \sa ActuatorType
  bool isPositionLimitEnforced() const;

  /// Get a unique index in skeleton of a generalized coordinate in this Joint
  virtual std::size_t getIndexInSkeleton(std::size_t _index) const = 0;

  /// Get a unique index in the kinematic tree of a generalized coordinate in
  /// this Joint
  virtual std::size_t getIndexInTree(std::size_t _index) const = 0;

  /// Get the index of this Joint within its Skeleton
  std::size_t getJointIndexInSkeleton() const;

  /// Get the index of this Joint within its tree
  std::size_t getJointIndexInTree() const;

  /// Get the index of the tree that this Joint belongs to
  std::size_t getTreeIndex() const;

  /// Returns whether the joint contains a DegreeOfFreedom
  virtual bool hasDof(const DegreeOfFreedom* dof) const = 0;

  /// Get an object to access the _index-th degree of freedom (generalized
  /// coordinate) of this Joint
  virtual DegreeOfFreedom* getDof(std::size_t _index) = 0;

  /// Get an object to access the _index-th degree of freedom (generalized
  /// coordinate) of this Joint
  virtual const DegreeOfFreedom* getDof(std::size_t _index) const = 0;

  /// Alternative to DegreeOfFreedom::setName()
  virtual const std::string& setDofName(
      std::size_t _index, const std::string& _name, bool _preserveName = true)
      = 0;

  /// Alternative to DegreeOfFreedom::preserveName()
  virtual void preserveDofName(std::size_t _index, bool _preserve) = 0;

  /// Alternative to DegreeOfFreedom::isNamePreserved()
  virtual bool isDofNamePreserved(std::size_t _index) const = 0;

  /// Alternative to DegreeOfFreedom::getName()
  virtual const std::string& getDofName(std::size_t _index) const = 0;

  /// Get number of generalized coordinates
  virtual std::size_t getNumDofs() const = 0;

  //----------------------------------------------------------------------------
  /// \{ \name Command
  //----------------------------------------------------------------------------

  /// Set a single command
  virtual void setCommand(std::size_t _index, s_t _command) = 0;

  /// Get a single command
  virtual s_t getCommand(std::size_t _index) const = 0;

  /// Set all commands for this Joint
  virtual void setCommands(const Eigen::VectorXs& _commands) = 0;

  /// Get all commands for this Joint
  virtual Eigen::VectorXs getCommands() const = 0;

  /// Set all the commands for this Joint to zero
  virtual void resetCommands() = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Position
  //----------------------------------------------------------------------------

  /// Set the position of a single generalized coordinate
  virtual void setPosition(std::size_t _index, s_t _position) = 0;

  /// Get the position of a single generalized coordinate
  virtual s_t getPosition(std::size_t _index) const = 0;

  /// Set the positions of all generalized coordinates in this Joint
  virtual void setPositions(const Eigen::VectorXs& _positions) = 0;

  /// Get the positions of all generalized coordinates in this Joint
  virtual Eigen::VectorXs getPositions() const = 0;

  /// Set lower limit for position
  virtual void setPositionLowerLimit(std::size_t _index, s_t _position) = 0;

  /// Get lower limit for position
  virtual s_t getPositionLowerLimit(std::size_t _index) const = 0;

  /// Set the position lower limits of all the generalized coordinates.
  virtual void setPositionLowerLimits(const Eigen::VectorXs& lowerLimits) = 0;

  /// Get the position lower limits of all the generalized coordinates.
  virtual Eigen::VectorXs getPositionLowerLimits() const = 0;

  /// Set upper limit for position
  virtual void setPositionUpperLimit(std::size_t _index, s_t _position) = 0;

  /// Get upper limit for position
  virtual s_t getPositionUpperLimit(std::size_t _index) const = 0;

  /// Set the position upper limits of all the generalized coordinates.
  virtual void setPositionUpperLimits(const Eigen::VectorXs& upperLimits) = 0;

  /// Get the position upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getPositionUpperLimits() const = 0;

  /// Get whether a generalized coordinate is cyclic. Return true if and only
  /// if this generalized coordinate has an infinite number of positions that
  /// produce the same relative transform. Note that, for a multi-DOF joint,
  /// producing a cycle may require altering the position of this Joint's other
  /// generalized coordinates.
  virtual bool isCyclic(std::size_t _index) const = 0;

  /// Get whether the position of a generalized coordinate has limits.
  virtual bool hasPositionLimit(std::size_t _index) const = 0;

  /// Set the position of this generalized coordinate to its initial position
  virtual void resetPosition(std::size_t _index) = 0;

  /// Set the positions of all generalized coordinates in this Joint to their
  /// initial positions
  virtual void resetPositions() = 0;

  /// Change the position that resetPositions() will give to this coordinate
  virtual void setInitialPosition(std::size_t _index, s_t _initial) = 0;

  /// Get the position that resetPosition() will give to this coordinate
  virtual s_t getInitialPosition(std::size_t _index) const = 0;

  /// Change the positions that resetPositions() will give to this Joint's
  /// coordinates
  virtual void setInitialPositions(const Eigen::VectorXs& _initial) = 0;

  /// Get the positions that resetPositions() will give to this Joint's
  /// coordinates
  virtual Eigen::VectorXs getInitialPositions() const = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Velocity
  //----------------------------------------------------------------------------

  /// Set the velocity of a single generalized coordinate
  virtual void setVelocity(std::size_t _index, s_t _velocity) = 0;

  /// Get the velocity of a single generalized coordinate
  virtual s_t getVelocity(std::size_t _index) const = 0;

  /// Set the velocities of all generalized coordinates in this Joint
  virtual void setVelocities(const Eigen::VectorXs& _velocities) = 0;

  /// Get the velocities of all generalized coordinates in this Joint
  virtual Eigen::VectorXs getVelocities() const = 0;

  /// Set lower limit for velocity
  virtual void setVelocityLowerLimit(std::size_t _index, s_t _velocity) = 0;

  /// Get lower limit for velocity
  virtual s_t getVelocityLowerLimit(std::size_t _index) const = 0;

  /// Set the velocity lower limits of all the generalized coordinates.
  virtual void setVelocityLowerLimits(const Eigen::VectorXs& lowerLimits) = 0;

  /// Get the velocity lower limits of all the generalized coordinates.
  virtual Eigen::VectorXs getVelocityLowerLimits() const = 0;

  /// Set upper limit for velocity
  virtual void setVelocityUpperLimit(std::size_t _index, s_t _velocity) = 0;

  /// Get upper limit for velocity
  virtual s_t getVelocityUpperLimit(std::size_t _index) const = 0;

  /// Set the velocity upper limits of all the generalized coordinates.
  virtual void setVelocityUpperLimits(const Eigen::VectorXs& upperLimits) = 0;

  /// Get the velocity upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getVelocityUpperLimits() const = 0;

  /// Set the velocity of a generalized coordinate in this Joint to its initial
  /// velocity
  virtual void resetVelocity(std::size_t _index) = 0;

  /// Set the velocities of all generalized coordinates in this Joint to their
  /// initial velocities
  virtual void resetVelocities() = 0;

  /// Change the velocity that resetVelocity() will give to this coordinate
  virtual void setInitialVelocity(std::size_t _index, s_t _initial) = 0;

  /// Get the velocity that resetVelocity() will give to this coordinate
  virtual s_t getInitialVelocity(std::size_t _index) const = 0;

  /// Change the velocities that resetVelocities() will give to this Joint's
  /// coordinates
  virtual void setInitialVelocities(const Eigen::VectorXs& _initial) = 0;

  /// Get the velocities that resetVelocities() will give to this Joint's
  /// coordinates
  virtual Eigen::VectorXs getInitialVelocities() const = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Acceleration
  //----------------------------------------------------------------------------

  /// Set the acceleration of a single generalized coordinate
  virtual void setAcceleration(std::size_t _index, s_t _acceleration) = 0;

  /// Get the acceleration of a single generalized coordinate
  virtual s_t getAcceleration(std::size_t _index) const = 0;

  /// Set the accelerations of all generalized coordinates in this Joint
  virtual void setAccelerations(const Eigen::VectorXs& _accelerations) = 0;

  /// Get the accelerations of all generalized coordinates in this Joint
  virtual Eigen::VectorXs getAccelerations() const = 0;

  /// Set the accelerations of all generalized coordinates in this Joint to zero
  virtual void resetAccelerations() = 0;

  /// Set lower limit for acceleration
  virtual void setAccelerationLowerLimit(std::size_t _index, s_t _acceleration)
      = 0;

  /// Get lower limit for acceleration
  virtual s_t getAccelerationLowerLimit(std::size_t _index) const = 0;

  /// Set the acceleration upper limits of all the generalized coordinates.
  virtual void setAccelerationLowerLimits(const Eigen::VectorXs& lowerLimits)
      = 0;

  /// Get the acceleration upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getAccelerationLowerLimits() const = 0;

  /// Set upper limit for acceleration
  virtual void setAccelerationUpperLimit(std::size_t _index, s_t _acceleration)
      = 0;

  /// Get upper limit for acceleration
  virtual s_t getAccelerationUpperLimit(std::size_t _index) const = 0;

  /// Set the acceleration upper limits of all the generalized coordinates.
  virtual void setAccelerationUpperLimits(const Eigen::VectorXs& upperLimits)
      = 0;

  /// Get the acceleration upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getAccelerationUpperLimits() const = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Force
  //----------------------------------------------------------------------------

  /// Set the force of a single generalized coordinate
  virtual void setControlForce(std::size_t _index, s_t _force) = 0;

  /// Get the force of a single generalized coordinate
  virtual s_t getControlForce(std::size_t _index) const = 0;

  /// Set the forces of all generalized coordinates in this Joint
  virtual void setControlForces(const Eigen::VectorXs& _forces) = 0;

  /// Get the forces of all generalized coordinates in this Joint
  virtual Eigen::VectorXs getControlForces() const = 0;

  /// Set the forces of all generalized coordinates in this Joint to zero
  virtual void resetControlForces() = 0;

  /// Set lower limit for force
  virtual void setControlForceLowerLimit(std::size_t _index, s_t _force) = 0;

  /// Get lower limit for force
  virtual s_t getControlForceLowerLimit(std::size_t _index) const = 0;

  /// Set the force upper limits of all the generalized coordinates.
  virtual void setControlForceLowerLimits(const Eigen::VectorXs& lowerLimits)
      = 0;

  /// Get the force upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getControlForceLowerLimits() const = 0;

  /// Set upper limit for force
  virtual void setControlForceUpperLimit(std::size_t _index, s_t _force) = 0;

  /// Get upper limit for force
  virtual s_t getControlForceUpperLimit(std::size_t _index) const = 0;

  /// Set the force upper limits of all the generalized coordinates.
  virtual void setControlForceUpperLimits(const Eigen::VectorXs& upperLimits)
      = 0;

  /// Get the force upper limits of all the generalized coordinates.
  virtual Eigen::VectorXs getControlForceUpperLimits() const = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Sanity Check
  //----------------------------------------------------------------------------

  /// Returns false if the initial position or initial velocity are outside of
  /// limits
  // TODO: Consider extending this to a more comprehensive sanity check
  bool checkSanity(bool _printWarnings = true) const;

  //----------------------------------------------------------------------------
  /// \{ \name Velocity change
  //----------------------------------------------------------------------------

  /// Set a single velocity change
  virtual void setVelocityChange(std::size_t _index, s_t _velocityChange) = 0;

  /// Get a single velocity change
  virtual s_t getVelocityChange(std::size_t _index) const = 0;

  /// Set zero all the velocity change
  virtual void resetVelocityChanges() = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Constraint impulse
  //----------------------------------------------------------------------------

  /// Set a single constraint impulse
  virtual void setConstraintImpulse(std::size_t _index, s_t _impulse) = 0;

  /// Get a single constraint impulse
  virtual s_t getConstraintImpulse(std::size_t _index) const = 0;

  /// Set zero all the constraint impulses
  virtual void resetConstraintImpulses() = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Integration and finite difference
  //----------------------------------------------------------------------------

  /// Integrate positions using Euler method
  virtual void integratePositions(s_t _dt) = 0;

  /// Integrate velocities using Euler method
  virtual void integrateVelocities(s_t _dt) = 0;

  /// Return the difference of two generalized coordinates which are measured in
  /// the configuration space of this Skeleton.
  virtual Eigen::VectorXs getPositionDifferences(
      const Eigen::VectorXs& _q2, const Eigen::VectorXs& _q1) const = 0;

  /// Returns the value of integrating a position, where it's possible to pass
  /// in an arbitrory position and velocity
  virtual Eigen::VectorXs integratePositionsExplicit(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t dt)
      = 0;

  /// Returns d/dpos of integratePositionsExplicit()
  virtual Eigen::MatrixXs getPosPosJacobian(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t _dt)
      = 0;

  /// Returns d/dvel of integratePositionsExplicit()
  virtual Eigen::MatrixXs getVelPosJacobian(
      const Eigen::VectorXs& pos, const Eigen::VectorXs& vel, s_t _dt)
      = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Passive forces - spring, viscous friction, Coulomb friction
  //----------------------------------------------------------------------------

  /// Set stiffness of joint spring force.
  /// \param[in] _index Index of joint axis.
  /// \param[in] _k Spring stiffness.
  virtual void setSpringStiffness(std::size_t _index, s_t _k) = 0;

  /// Get stiffness of joint spring force.
  /// \param[in] _index Index of joint axis.
  virtual s_t getSpringStiffness(std::size_t _index) const = 0;

  /// Set rest position of spring force.
  /// \param[in] _index Index of joint axis.
  /// \param[in] _q0 Rest position.
  virtual void setRestPosition(std::size_t _index, s_t _q0) = 0;

  /// Get rest position of spring force.
  /// \param[in] _index Index of joint axis.
  /// \return Rest position.
  virtual s_t getRestPosition(std::size_t _index) const = 0;

  /// Set coefficient of joint damping (viscous friction) force.
  /// \param[in] _index Index of joint axis.
  /// \param[in] _coeff Damping coefficient.
  virtual void setDampingCoefficient(std::size_t _index, s_t _coeff) = 0;

  /// Get coefficient of joint damping (viscous friction) force.
  /// \param[in] _index Index of joint axis.
  virtual s_t getDampingCoefficient(std::size_t _index) const = 0;

  /// Set joint Coulomb friction froce.
  /// \param[in] _index Index of joint axis.
  /// \param[in] _friction Joint Coulomb friction froce given index.
  virtual void setCoulombFriction(std::size_t _index, s_t _friction) = 0;

  /// Get joint Coulomb friction froce.
  /// \param[in] _index Index of joint axis.
  virtual s_t getCoulombFriction(std::size_t _index) const = 0;

  /// \}

  //----------------------------------------------------------------------------

  /// Get potential energy
  DART_DEPRECATED(6.1)
  s_t getPotentialEnergy() const;

  /// Compute and return the potential energy
  virtual s_t computePotentialEnergy() const = 0;

  //----------------------------------------------------------------------------

  /// Deprecated. Use getRelativeTransform() instead.
  DART_DEPRECATED(6.0)
  const Eigen::Isometry3s& getLocalTransform() const;

  /// Deprecated. Use getLocalSpatialVelocity() instead.
  DART_DEPRECATED(6.0)
  const Eigen::Vector6s& getLocalSpatialVelocity() const;

  /// Deprecated. Use getLocalSpatialAcceleration() instead.
  DART_DEPRECATED(6.0)
  const Eigen::Vector6s& getLocalSpatialAcceleration() const;

  /// Deprecated. Use getLocalPrimaryAcceleration() instead.
  DART_DEPRECATED(6.0)
  const Eigen::Vector6s& getLocalPrimaryAcceleration() const;

  /// Deprecated. Use getRelativeJacobian() instead.
  DART_DEPRECATED(6.0)
  const math::Jacobian getLocalJacobian() const;

  /// Deprecated. Use getRelativeJacobian() instead.
  DART_DEPRECATED(6.0)
  math::Jacobian getLocalJacobian(const Eigen::VectorXs& positions) const;

  /// Deprecated. Use getRelativeJacobianTimeDeriv() instead.
  DART_DEPRECATED(6.0)
  const math::Jacobian getLocalJacobianTimeDeriv() const;

  /// Get transform of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame
  const Eigen::Isometry3s& getRelativeTransform() const;

  /// Get spatial velocity of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame
  const Eigen::Vector6s& getRelativeSpatialVelocity() const;

  /// Get spatial acceleration of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame
  const Eigen::Vector6s& getRelativeSpatialAcceleration() const;

  /// Get J * \f$ \ddot{q} \f$ of this joint
  const Eigen::Vector6s& getRelativePrimaryAcceleration() const;

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame
  virtual const math::Jacobian getRelativeJacobian() const = 0;

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame
  virtual math::Jacobian getRelativeJacobian(
      const Eigen::VectorXs& positions) const = 0;

  // TODO(JS): Rename and add documentation
  virtual math::Jacobian getRelativeJacobianDeriv(std::size_t /*index*/) const
  {
    // TODO(JS): Remove
    return math::Jacobian::Zero(6, getNumDofs());
  }

  /// Get time derivative of spatial Jacobian of the child BodyNode relative to
  /// the parent BodyNode expressed in the child BodyNode frame
  virtual const math::Jacobian getRelativeJacobianTimeDeriv() const = 0;

  /// Computes derivative of time derivative of Jacobian w.r.t. position.
  virtual math::Jacobian getRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t /*index*/) const
  {
    // TODO(JS): Remove
    return math::Jacobian::Zero(6, getNumDofs());
  }

  /// Computes derivative of time derivative of Jacobian w.r.t. velocity.
  virtual math::Jacobian getRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t /*index*/) const
  {
    // TODO(JS): Remove
    return math::Jacobian::Zero(6, getNumDofs());
  }

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame, in the `q` vector space. This is
  /// generally the same as the getRelativeJacobian() for the `dq` vector space,
  /// because `q` and `dq` are generally in the same vector space. However, for
  /// BallJoint and FreeJoint these are different values.
  virtual const math::Jacobian getRelativeJacobianInPositionSpace() const = 0;

  /// Get spatial Jacobian of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame, in the `q` vector space. This is
  /// generally the same as the getRelativeJacobian() for the `dq` vector space,
  /// because `q` and `dq` are generally in the same vector space. However, for
  /// BallJoint and FreeJoint these are different values.
  virtual math::Jacobian getRelativeJacobianInPositionSpace(
      const Eigen::VectorXs& positions) const = 0;

  /// Update spatial Jacobian of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame, in the `q` vector space.
  /// This is generally the same as the updateRelativeJacobian() for the `dq`
  /// vector space, because `q` and `dq` are generally in the same vector space.
  /// However, for BallJoint and FreeJoint these are different values.
  ///
  /// \param[in] mandatory This argument can be set to false if the Jacobian
  /// update request is due to a change in Joint positions, because not all
  /// Joint types have a relative Jacobian that depends on their Joint
  /// positions, so a relative Jacobian update would not actually be required.
  virtual void updateRelativeJacobianInPositionSpace(
      bool mandatory = true) const = 0;

  /// This uses finite differencing to compute the relative Jacobian in velocity
  /// space
  Eigen::MatrixXs finiteDifferenceRelativeJacobian();

  /// This uses finite differencing to compute the relative Jacobian in position
  /// space
  Eigen::MatrixXs finiteDifferenceRelativeJacobianInPositionSpace(
      bool useRidders = true);

  /// This checks the intermediate analytical results of
  /// getRelativeJacobianInPositionSpace() against the finite differencing
  /// equivalents.
  void debugRelativeJacobianInPositionSpace();

  /// Returns the screw representation of a given dof at our current position.
  /// That is, if we increment dof by EPS, that's the same as left-multiplying
  /// our getWorldTransform() by math::expMap(screw * EPS).
  virtual Eigen::Vector6s getWorldAxisScrewForPosition(int dof) const;

  /// Returns the screw representation of a given dof at our current position.
  /// That is, if we increment the vel for dof by EPS, that increases all the
  /// child body spatial velocities by the returned screw.
  virtual Eigen::Vector6s getWorldAxisScrewForVelocity(int dof) const;

  /// Get constraint wrench expressed in body node frame
  virtual Eigen::Vector6s getBodyConstraintWrench() const = 0;
  // TODO: Need more informative name.

  /// Get spring force
  ///
  /// We apply spring force in implicit manner. The spring force is
  /// F = -(springStiffness * q(k+1)), where q(k+1) is approximated as
  /// q(k) + h * dq(k) * h^2 * ddq(k). Since, in the recursive forward dynamics
  /// algorithm, ddq(k) is unknown variable that we want to obtain as the
  /// result, the spring force here is just
  /// F = -springStiffness * (q(k) + h * dq(k)) and
  /// -springStiffness * h^2 * ddq(k) term is rearranged at the recursive
  /// forward dynamics algorithm, and it affects on the articulated inertia.
  /// \sa BodyNode::updateArticulatedInertia(s_t).
  ///
  /// \param[in] _timeStep Time step used for approximating q(k+1).
  //  Eigen::VectorXs getSpringForces(s_t _timeStep) const;

  /// Get damping force
  ///
  /// We apply the damping force in implicit manner. The damping force is
  /// F = -(dampingCoefficient * dq(k+1)), where dq(k+1) is approximated as
  /// dq(k) + h * ddq(k). Since, in the recursive forward dynamics algorithm,
  /// ddq(k) is unknown variable that we want to obtain as the result, the
  /// damping force here is just F = -(dampingCoefficient * dq(k)) and
  /// -dampingCoefficient * h * ddq(k) term is rearranged at the recursive
  /// forward dynamics algorithm, and it affects on the articulated inertia.
  /// \sa BodyNode::updateArticulatedInertia(s_t).
  //  Eigen::VectorXs getDampingForces() const;

  //----------------------------------------------------------------------------
  /// \{ \name Differential Dynamics
  //----------------------------------------------------------------------------

  // TODO(JS): Rename and add documentation
  virtual Eigen::VectorXs getAlpha() const
  {
    return Eigen::VectorXs::Zero(0);
  }

  // TODO(JS): Rename and add documentation
  virtual math::Inertia computePi(const math::Inertia& AI) const
  {
    return AI;
  }

  // TODO(JS): Rename and add documentation
  virtual Eigen::Vector6s computeBeta(
      const math::Inertia& /*AI*/, const Eigen::Vector6s& AB) const
  {
    return AB;
  }

  // TODO(JS): Rename and add documentation
  virtual void computeJacobianOfMinvX_init()
  {
    // Do nothing
  }

  // TODO(JS): Rename and add documentation
  virtual void computeJacobianOfMinvX_A(
      const math::Inertia& /*AI*/, const Eigen::Vector6s& /*AB*/)
  {
    // Do nothing
  }

  // TODO(JS): Rename and add documentation
  virtual Eigen::MatrixXs computeJacobianOfMinvX_B(const math::Inertia& /*AI*/)
  {
    return Eigen::MatrixXs::Zero(0, 0);
  }

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Update Notifiers
  //----------------------------------------------------------------------------

  /// Notify that a position has updated
  DART_DEPRECATED(6.2)
  void notifyPositionUpdate();

  /// Notify that a position has updated
  void notifyPositionUpdated();

  /// Notify that a velocity has updated
  DART_DEPRECATED(6.2)
  void notifyVelocityUpdate();

  /// Notify that a velocity has updated
  void notifyVelocityUpdated();

  /// Notify that an acceleration has updated
  DART_DEPRECATED(6.2)
  void notifyAccelerationUpdate();

  /// Notify that an acceleration has updated
  void notifyAccelerationUpdated();

  /// \}

  //----------------------------------------------------------------------------
  // Friendship
  //----------------------------------------------------------------------------

  friend class BodyNode;
  friend class SoftBodyNode;
  friend class Skeleton;

protected:
  /// Constructor called by inheriting class
  Joint();

  /// Create a clone of this Joint. This may only be called by the Skeleton
  /// class.
  virtual Joint* clone() const = 0;

  /// Called by the Skeleton class
  virtual void registerDofs() = 0;

  /// \brief Create a DegreeOfFreedom pointer.
  /// \param[in] _indexInJoint DegreeOfFreedom's index within the joint. Note
  /// that the index should be unique within the joint.
  ///
  /// DegreeOfFreedom should be created by the Joint because the DegreeOfFreedom
  /// class has a protected constructor, and the Joint is responsible for memory
  /// management of the pointer which is returned.
  DegreeOfFreedom* createDofPointer(std::size_t _indexInJoint);

  /// Update the names of the joint's degrees of freedom. Used when setName() is
  /// called with _renameDofs set to true.
  virtual void updateDegreeOfFreedomNames() = 0;

  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  /// Deprecated. Use updateRelativeTransform() instead.
  DART_DEPRECATED(6.0)
  void updateLocalTransform() const;

  /// Deprecated. Use updateRelativeSpatialVelocity() instead.
  DART_DEPRECATED(6.0)
  void updateLocalSpatialVelocity() const;

  /// Deprecated. Use updateRelativeSpatialAcceleration() instead.
  DART_DEPRECATED(6.0)
  void updateLocalSpatialAcceleration() const;

  /// Deprecated. Use updateRelativePrimaryAcceleration() instead.
  DART_DEPRECATED(6.0)
  void updateLocalPrimaryAcceleration() const;

  /// Deprecated. Use updateRelativeJacobian() instead.
  DART_DEPRECATED(6.0)
  void updateLocalJacobian(bool mandatory = true) const;

  /// Deprecated. Use updateRelativeJacobianTimeDeriv() instead.
  DART_DEPRECATED(6.0)
  void updateLocalJacobianTimeDeriv() const;

  /// Update transform of the child BodyNode relative to the parent BodyNode
  /// expressed in the child BodyNode frame
  virtual void updateRelativeTransform() const = 0;

  /// Update spatial velocity of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame
  virtual void updateRelativeSpatialVelocity() const = 0;

  /// Update spatial acceleration of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame
  virtual void updateRelativeSpatialAcceleration() const = 0;

  /// Update J * \f$ \ddot{q} \f$ of this joint
  virtual void updateRelativePrimaryAcceleration() const = 0;

  /// Update spatial Jacobian of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame
  ///
  /// \param[in] mandatory This argument can be set to false if the Jacobian
  /// update request is due to a change in Joint positions, because not all
  /// Joint types have a relative Jacobian that depends on their Joint
  /// positions, so a relative Jacobian update would not actually be required.
  virtual void updateRelativeJacobian(bool mandatory = true) const = 0;

  /// Update time derivative of spatial Jacobian of the child BodyNode relative
  /// to the parent BodyNode expressed in the child BodyNode frame
  ///
  /// If the relative Jacobian time derivative of this Joint is zero, then this
  /// function will be a no op.
  virtual void updateRelativeJacobianTimeDeriv() const = 0;

  /// Tells the Skeleton to update the articulated inertia if it needs updating
  void updateArticulatedInertia() const;

  /// Add joint velocity to _vel
  virtual void addVelocityTo(Eigen::Vector6s& _vel) = 0;

  /// Set joint partial acceleration to _partialAcceleration
  virtual void setPartialAccelerationTo(
      Eigen::Vector6s& _partialAcceleration,
      const Eigen::Vector6s& _childVelocity)
      = 0;
  // TODO(JS): Rename with more informative name

  /// Add joint acceleration to _acc
  virtual void addAccelerationTo(Eigen::Vector6s& _acc) = 0;

  /// Add joint velocity change to _velocityChange
  virtual void addVelocityChangeTo(Eigen::Vector6s& _velocityChange) = 0;

  /// Add child's articulated inertia to parent's one
  virtual void addChildArtInertiaTo(
      Eigen::Matrix6s& _parentArtInertia,
      const Eigen::Matrix6s& _childArtInertia)
      = 0;

  /// Add child's articulated inertia to parent's one. Forward dynamics routine.
  virtual void addChildArtInertiaImplicitTo(
      Eigen::Matrix6s& _parentArtInertiaImplicit,
      const Eigen::Matrix6s& _childArtInertiaImplicit)
      = 0;
  // TODO(JS): rename to updateAInertiaChildAInertia()

  /// Update inverse of projected articulated body inertia
  virtual void updateInvProjArtInertia(const Eigen::Matrix6s& _artInertia) = 0;

  /// Forward dynamics routine.
  virtual void updateInvProjArtInertiaImplicit(
      const Eigen::Matrix6s& _artInertia, s_t _timeStep)
      = 0;
  // TODO(JS): rename to updateAInertiaPsi()

  /// Add child's bias force to parent's one
  virtual void addChildBiasForceTo(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce,
      const Eigen::Vector6s& _childPartialAcc)
      = 0;

  /// Add child's bias impulse to parent's one
  virtual void addChildBiasImpulseTo(
      Eigen::Vector6s& _parentBiasImpulse,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasImpulse)
      = 0;

  /// Update joint total force
  virtual void updateTotalForce(
      const Eigen::Vector6s& _bodyForce, s_t _timeStep)
      = 0;
  // TODO: rename

  /// Update joint total impulse
  virtual void updateTotalImpulse(const Eigen::Vector6s& _bodyImpulse) = 0;
  // TODO: rename

  /// Set total impulses to zero
  virtual void resetTotalImpulses() = 0;

  /// Update joint acceleration
  virtual void updateAcceleration(
      const Eigen::Matrix6s& _artInertia, const Eigen::Vector6s& _spatialAcc)
      = 0;

  /// Update joint velocity change
  /// \param _artInertia
  /// \param _velocityChange
  virtual void updateVelocityChange(
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _velocityChange)
      = 0;

  /// Update joint force for inverse dynamics.
  /// \param[in] _bodyForce Transmitting spatial body force from the parent
  /// BodyNode to the child BodyNode. The spatial force is expressed in the
  /// child BodyNode's frame.
  /// \param[in] _timeStep
  /// \param[in] _withDampingForces
  /// \param[in] _withSpringForces
  virtual void updateForceID(
      const Eigen::Vector6s& _bodyForce,
      s_t _timeStep,
      bool _withDampingForces,
      bool _withSpringForces)
      = 0;

  /// Update joint force for forward dynamics.
  /// \param[in] _bodyForce Transmitting spatial body force from the parent
  /// BodyNode to the child BodyNode. The spatial force is expressed in the
  /// child BodyNode's frame.
  /// \param[in] _timeStep
  /// \param[in] _withDampingForces
  /// \param[in] _withSpringForces
  virtual void updateForceFD(
      const Eigen::Vector6s& _bodyForce,
      s_t _timeStep,
      bool _withDampingForces,
      bool _withSpringForces)
      = 0;

  /// Update joint impulses for inverse dynamics
  virtual void updateImpulseID(const Eigen::Vector6s& _bodyImpulse) = 0;

  /// Update joint impulses for forward dynamics
  virtual void updateImpulseFD(const Eigen::Vector6s& _bodyImpulse) = 0;

  /// Update constrained terms for forward dynamics
  virtual void updateConstrainedTerms(s_t _timeStep) = 0;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Recursive algorithm routines for equations of motion
  //----------------------------------------------------------------------------

  /// Add child's bias force to parent's one
  virtual void addChildBiasForceForInvMassMatrix(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce)
      = 0;

  /// Add child's bias force to parent's one
  virtual void addChildBiasForceForInvAugMassMatrix(
      Eigen::Vector6s& _parentBiasForce,
      const Eigen::Matrix6s& _childArtInertia,
      const Eigen::Vector6s& _childBiasForce)
      = 0;

  ///
  virtual void updateTotalForceForInvMassMatrix(
      const Eigen::Vector6s& _bodyForce)
      = 0;

  ///
  virtual void getInvMassMatrixSegment(
      Eigen::MatrixXs& _invMassMat,
      const std::size_t _col,
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _spatialAcc)
      = 0;

  ///
  virtual void getInvAugMassMatrixSegment(
      Eigen::MatrixXs& _invMassMat,
      const std::size_t _col,
      const Eigen::Matrix6s& _artInertia,
      const Eigen::Vector6s& _spatialAcc)
      = 0;

  ///
  virtual void addInvMassMatrixSegmentTo(Eigen::Vector6s& _acc) = 0;

  ///
  virtual Eigen::VectorXs getSpatialToGeneralized(
      const Eigen::Vector6s& _spatial)
      = 0;

  /// \}

protected:
  /// Child BodyNode pointer that this Joint belongs to
  BodyNode* mChildBodyNode;

  /// Relative transformation of the child BodyNode relative to the parent
  /// BodyNode expressed in the child BodyNode frame
  ///
  /// Do not use directly! Use getRelativeTransform() to access this
  mutable Eigen::Isometry3s mT;

  /// Relative spatial velocity from parent BodyNode to child BodyNode where the
  /// velocity is expressed in child body Frame
  ///
  /// Do not use directly! Use getRelativeSpatialVelocity() to access this
  mutable Eigen::Vector6s mSpatialVelocity;

  /// Relative spatial acceleration from parent BodyNode to child BodyNode where
  /// the acceleration is expressed in the child body Frame
  ///
  /// Do not use directly! Use getRelativeSpatialAcceleration() to access this
  mutable Eigen::Vector6s mSpatialAcceleration;

  /// J * q_dd
  ///
  /// Do not use directly! Use getRelativePrimaryAcceleration() to access this
  mutable Eigen::Vector6s mPrimaryAcceleration;

  /// True iff this joint's position has changed since the last call to
  /// getRelativeTransform()
  mutable bool mNeedTransformUpdate;
  // TODO(JS): Rename this to mIsTransformDirty in DART 7

  /// True iff this joint's position or velocity has changed since the last call
  /// to getRelativeSpatialVelocity()
  mutable bool mNeedSpatialVelocityUpdate;
  // TODO(JS): Rename this to mIsSpatialVelocityDirty in DART 7

  /// True iff this joint's position, velocity, or acceleration has changed
  /// since the last call to getRelativeSpatialAcceleration()
  mutable bool mNeedSpatialAccelerationUpdate;
  // TODO(JS): Rename this to mIsSpatialAccelerationDirty in DART 7

  /// True iff this joint's position, velocity, or acceleration has changed
  /// since the last call to getRelativePrimaryAcceleration()
  mutable bool mNeedPrimaryAccelerationUpdate;
  // TODO(JS): Rename this to mIsPrimaryAccelerationDirty in DART 7

  /// True iff this joint's relative Jacobian has not been updated since the
  /// last position change
  mutable bool mIsRelativeJacobianDirty;

  /// True iff this joint's relative Jacobian has not been updated since the
  /// last position change
  mutable bool mIsRelativeJacobianInPositionSpaceDirty;

  /// True iff this joint's relative Jacobian time derivative has not been
  /// updated since the last position or velocity change
  mutable bool mIsRelativeJacobianTimeDerivDirty;

public:
  // To get byte-aligned Eigen vectors
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace dynamics
} // namespace dart

#endif // DART_DYNAMICS_JOINT_HPP_
