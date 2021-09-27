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

#ifndef DART_DYNAMICS_POINTMASS_HPP_
#define DART_DYNAMICS_POINTMASS_HPP_

#include <vector>
#include <Eigen/Dense>
#include "dart/math/Helpers.hpp"
#include "dart/dynamics/Entity.hpp"

namespace dart {
namespace dynamics {

class EllipsoidShape;
class SoftBodyNode;

class PointMassNotifier;

///
class PointMass : public common::Subject
{
public:
  friend class SoftBodyNode;

  /// State for each PointMass
  struct State
  {
    /// Position
    Eigen::Vector3s mPositions;

    /// Generalized velocity
    Eigen::Vector3s mVelocities;

    /// Generalized acceleration
    Eigen::Vector3s mAccelerations;

    /// Generalized force
    Eigen::Vector3s mForces;

    /// Default constructor
    State(const Eigen::Vector3s& positions = Eigen::Vector3s::Zero(),
          const Eigen::Vector3s& velocities = Eigen::Vector3s::Zero(),
          const Eigen::Vector3s& accelerations = Eigen::Vector3s::Zero(),
          const Eigen::Vector3s& forces = Eigen::Vector3s::Zero());

    bool operator==(const State& other) const;

    virtual ~State() = default;
  };

  /// Properties for each PointMass
  struct Properties
  {
    /// Resting position viewed in the parent SoftBodyNode frame
    Eigen::Vector3s mX0;

    /// Mass.
    s_t mMass;

    /// Indices of connected Point Masses
    std::vector<std::size_t> mConnectedPointMassIndices;

    /// Lower limit of position
    Eigen::Vector3s mPositionLowerLimits; // Currently unused

    /// Upper limit of position
    Eigen::Vector3s mPositionUpperLimits; // Currently unused

    /// Min value allowed.
    Eigen::Vector3s mVelocityLowerLimits; // Currently unused

    /// Max value allowed.
    Eigen::Vector3s mVelocityUpperLimits; // Currently unused

    /// Min value allowed.
    Eigen::Vector3s mAccelerationLowerLimits; // Currently unused

    /// upper limit of generalized acceleration
    Eigen::Vector3s mAccelerationUpperLimits; // Currently unused

    /// Min value allowed.
    Eigen::Vector3s mForceLowerLimits; // Currently unused

    /// Max value allowed.
    Eigen::Vector3s mForceUpperLimits; // Currently unused

    Properties(const Eigen::Vector3s& _X0 = Eigen::Vector3s::Zero(),
               s_t _mass = 0.0005,
               const std::vector<std::size_t>& _connections = std::vector<std::size_t>(),
               const Eigen::Vector3s& _positionLowerLimits =
                                      Eigen::Vector3s::Constant(-math::constantsd::inf()),
               const Eigen::Vector3s& _positionUpperLimits =
                                      Eigen::Vector3s::Constant( math::constantsd::inf()),
               const Eigen::Vector3s& _velocityLowerLimits =
                                      Eigen::Vector3s::Constant(-math::constantsd::inf()),
               const Eigen::Vector3s& _velocityUpperLimits =
                                      Eigen::Vector3s::Constant( math::constantsd::inf()),
               const Eigen::Vector3s& _accelerationLowerLimits =
                                      Eigen::Vector3s::Constant(-math::constantsd::inf()),
               const Eigen::Vector3s& _accelerationUpperLimits =
                                      Eigen::Vector3s::Constant( math::constantsd::inf()),
               const Eigen::Vector3s& _forceLowerLimits =
                                      Eigen::Vector3s::Constant(-math::constantsd::inf()),
               const Eigen::Vector3s& _forceUpperLimits =
                                      Eigen::Vector3s::Constant( math::constantsd::inf()));

    void setRestingPosition(const Eigen::Vector3s& _x);

    void setMass(s_t _mass);

    bool operator==(const Properties& other) const;

    bool operator!=(const Properties& other) const;

    virtual ~Properties() = default;
  };

  //--------------------------------------------------------------------------
  // Constructor and Desctructor
  //--------------------------------------------------------------------------

  /// Default destructor
  virtual ~PointMass();

  /// State of this PointMass
  State& getState();

  /// State of this PointMass
  const State& getState() const;

  ///
  std::size_t getIndexInSoftBodyNode() const;

  ///
  void setMass(s_t _mass);

  ///
  s_t getMass() const;

  ///
  s_t getPsi() const;

  ///
  s_t getImplicitPsi() const;

  ///
  s_t getPi() const;

  ///
  s_t getImplicitPi() const;

  ///
  void addConnectedPointMass(PointMass* _pointMass);

  ///
  std::size_t getNumConnectedPointMasses() const;

  ///
  PointMass* getConnectedPointMass(std::size_t _idx);

  ///
  const PointMass* getConnectedPointMass(std::size_t _idx) const;


  /// Set whether this point mass is colliding with other objects. Note that
  /// this status is set by the constraint solver during dynamics simulation but
  /// not by collision detector.
  /// \param[in] _isColliding True if this point mass is colliding.
  void setColliding(bool _isColliding);

  /// Return whether this point mass is set to be colliding with other objects.
  /// \return True if this point mass is colliding.
  bool isColliding();

  //----------------------------------------------------------------------------

  // Documentation inherited
  std::size_t getNumDofs() const;

//  // Documentation inherited
//  void setIndexInSkeleton(std::size_t _index, std::size_t _indexInSkeleton);

//  // Documentation inherited
//  std::size_t getIndexInSkeleton(std::size_t _index) const;

  //----------------------------------------------------------------------------
  // Position
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setPosition(std::size_t _index, s_t _position);

  // Documentation inherited
  s_t getPosition(std::size_t _index) const;

  // Documentation inherited
  void setPositions(const Eigen::Vector3s& _positions);

  // Documentation inherited
  const Eigen::Vector3s& getPositions() const;

  // Documentation inherited
  void resetPositions();

  //----------------------------------------------------------------------------
  // Velocity
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocity(std::size_t _index, s_t _velocity);

  // Documentation inherited
  s_t getVelocity(std::size_t _index) const;

  // Documentation inherited
  void setVelocities(const Eigen::Vector3s& _velocities);

  // Documentation inherited
  const Eigen::Vector3s& getVelocities() const;

  // Documentation inherited
  void resetVelocities();

  //----------------------------------------------------------------------------
  // Acceleration
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setAcceleration(std::size_t _index, s_t _acceleration);

  // Documentation inherited
  s_t getAcceleration(std::size_t _index) const;

  // Documentation inherited
  void setAccelerations(const Eigen::Vector3s& _accelerations);

  // Documentation inherited
  const Eigen::Vector3s& getAccelerations() const;

  /// Get the Eta term of this PointMass
  const Eigen::Vector3s& getPartialAccelerations() const;

  // Documentation inherited
  void resetAccelerations();

  //----------------------------------------------------------------------------
  // Force
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setForce(std::size_t _index, s_t _force);

  // Documentation inherited
  s_t getControlForce(std::size_t _index);

  // Documentation inherited
  void setForces(const Eigen::Vector3s& _forces);

  // Documentation inherited
  const Eigen::Vector3s& getControlForces() const;

  // Documentation inherited
  void resetForces();

  //----------------------------------------------------------------------------
  // Velocity change
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setVelocityChange(std::size_t _index, s_t _velocityChange);

  // Documentation inherited
  s_t getVelocityChange(std::size_t _index);

  // Documentation inherited
  void resetVelocityChanges();

  //----------------------------------------------------------------------------
  // Constraint impulse
  //----------------------------------------------------------------------------

  // Documentation inherited
  void setConstraintImpulse(std::size_t _index, s_t _impulse);

  // Documentation inherited
  s_t getConstraintImpulse(std::size_t _index);

  // Documentation inherited
  void resetConstraintImpulses();

  //----------------------------------------------------------------------------
  // Integration
  //----------------------------------------------------------------------------

  // Documentation inherited
  void integratePositions(s_t _dt);

  // Documentation inherited
  void integrateVelocities(s_t _dt);

  //----------------------------------------------------------------------------

  /// Add linear Cartesian force to this node.
  /// \param[in] _force External force.
  /// \param[in] _isForceLocal True if _force's reference frame is of the parent
  ///                          soft body node. False if _force's reference frame
  ///                          is of the world.
  void addExtForce(const Eigen::Vector3s& _force, bool _isForceLocal = false);

  ///
  void clearExtForce();

  //----------------------------------------------------------------------------
  // Constraints
  //   - Following functions are managed by constraint solver.
  //----------------------------------------------------------------------------
  /// Set constraint impulse
  void setConstraintImpulse(const Eigen::Vector3s& _constImp,
                            bool _isLocal = false);

  /// Add constraint impulse
  void addConstraintImpulse(const Eigen::Vector3s& _constImp,
                            bool _isLocal = false);

  /// Clear constraint impulse
  void clearConstraintImpulse();

  /// Get constraint impulse
  Eigen::Vector3s getConstraintImpulses() const;

  //----------------------------------------------------------------------------
  ///
  void setRestingPosition(const Eigen::Vector3s& _p);

  ///
  const Eigen::Vector3s& getRestingPosition() const;

  ///
  const Eigen::Vector3s& getLocalPosition() const;

  ///
  const Eigen::Vector3s& getWorldPosition() const;

  /// \todo Temporary function.
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> getBodyJacobian();
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> getWorldJacobian();

  /// Return velocity change due to impulse
  const Eigen::Vector3s& getBodyVelocityChange() const;

  ///
  SoftBodyNode* getParentSoftBodyNode();

  ///
  const SoftBodyNode* getParentSoftBodyNode() const;

  /// The number of the generalized coordinates by which this node is
  ///        affected.
//  int getNumDependentGenCoords() const;

  /// Return a generalized coordinate index from the array index
  ///        (< getNumDependentDofs).
//  int getDependentGenCoord(int _arrayIndex) const;

  /// Get the generalized velocity at the position of this point mass
  ///        where the velocity is expressed in the parent soft body node frame.
  const Eigen::Vector3s& getBodyVelocity() const;

  /// Get the generalized velocity at the position of this point mass
  ///        where the velocity is expressed in the world frame.
  Eigen::Vector3s getWorldVelocity() const;

  /// Get the generalized acceleration at the position of this point mass
  ///        where the acceleration is expressed in the parent soft body node
  ///        frame.
  const Eigen::Vector3s& getBodyAcceleration() const;

  /// Get the generalized acceleration at the position of this point mass
  ///        where the acceleration is expressed in the world frame.
  Eigen::Vector3s getWorldAcceleration() const;

protected:
  /// Constructor used by SoftBodyNode
  explicit PointMass(SoftBodyNode* _softBodyNode);

  ///
  void init();

  //----------------------------------------------------------------------------
  /// \{ \name Recursive dynamics routines
  //----------------------------------------------------------------------------

  /// \brief Update transformation.
  void updateTransform() const;

  /// \brief Update body velocity.
  void updateVelocity() const;

  /// \brief Update partial body acceleration due to parent joint's velocity.
  void updatePartialAcceleration() const;

  /// \brief Update articulated body inertia. Forward dynamics routine.
  /// \param[in] _timeStep Rquired for implicit joint stiffness and damping.
  void updateArtInertiaFD(s_t _timeStep) const;

  /// \brief Update bias force associated with the articulated body inertia.
  /// Forward dynamics routine.
  /// \param[in] _dt Required for implicit joint stiffness and damping.
  /// \param[in] _gravity Vector of gravitational acceleration
  void updateBiasForceFD(s_t _dt, const Eigen::Vector3s& _gravity);

  /// \brief Update bias impulse associated with the articulated body inertia.
  /// Impulse-based forward dynamics routine.
  void updateBiasImpulseFD();

  /// \brief Update body acceleration with the partial body acceleration.
  void updateAccelerationID() const;

  /// \brief Update body acceleration. Forward dynamics routine.
  void updateAccelerationFD();

  /// \brief Update body velocity change. Impluse-based forward dynamics
  /// routine.
  void updateVelocityChangeFD();

  /// \brief Update body force. Inverse dynamics routine.
  void updateTransmittedForceID(const Eigen::Vector3s& _gravity,
                                bool _withExternalForces = false);

  /// \brief Update body force. Forward dynamics routine.
  void updateTransmittedForce();

  /// \brief Update body force. Impulse-based forward dynamics routine.
  void updateTransmittedImpulse();

  /// \brief Update the joint force. Inverse dynamics routine.
  void updateJointForceID(s_t _timeStep,
                          s_t _withDampingForces,
                          s_t _withSpringForces);

  /// \brief Update constrained terms due to the constraint impulses. Foward
  /// dynamics routine.
  void updateConstrainedTermsFD(s_t _timeStep);

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Equations of motion related routines
  //----------------------------------------------------------------------------

  ///
  void updateMassMatrix();

  ///
  void aggregateMassMatrix(Eigen::MatrixXs& _MCol, int _col);

  ///
  void aggregateAugMassMatrix(Eigen::MatrixXs& _MCol, int _col,
                              s_t _timeStep);

  ///
  void updateInvMassMatrix();

  ///
  void updateInvAugMassMatrix();

  ///
  void aggregateInvMassMatrix(Eigen::MatrixXs& _MInvCol, int _col);

  ///
  void aggregateInvAugMassMatrix(Eigen::MatrixXs& _MInvCol, int _col,
                                 s_t _timeStep);

  ///
  void aggregateGravityForceVector(Eigen::VectorXs& _g,
                                   const Eigen::Vector3s& _gravity);

  ///
  void updateCombinedVector();

  ///
  void aggregateCombinedVector(Eigen::VectorXs& _Cg,
                               const Eigen::Vector3s& _gravity);

  /// Aggregate the external forces mFext in the generalized
  ///        coordinates recursively.
  void aggregateExternalForces(Eigen::VectorXs& _Fext);

  /// \}

  //-------------------- Cache Data for Mass Matrix ----------------------------
  ///
  Eigen::Vector3s mM_dV;

  ///
  Eigen::Vector3s mM_F;

  //----------------- Cache Data for Mass Inverse Matrix -----------------------
  ///
  Eigen::Vector3s mBiasForceForInvMeta;

  //---------------- Cache Data for Gravity Force Vector -----------------------
  ///
  Eigen::Vector3s mG_F;

  //------------------- Cache Data for Combined Vector -------------------------
  ///
  Eigen::Vector3s mCg_dV;

  ///
  Eigen::Vector3s mCg_F;

protected:
  // TODO(JS): Need?
  ///
//  Eigen::Matrix<std::size_t, 3, 1> mIndexInSkeleton;

  /// SoftBodyNode that this PointMass belongs to
  SoftBodyNode* mParentSoftBodyNode;

  /// Index of this PointMass within the SoftBodyNode
  std::size_t mIndex;

  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------

  /// Derivatives w.r.t. an arbitrary scalr variable
  Eigen::Vector3s mPositionDeriv;

  //----------------------------------------------------------------------------
  // Velocity
  //----------------------------------------------------------------------------

  /// Derivatives w.r.t. an arbitrary scalr variable
  Eigen::Vector3s mVelocitiesDeriv;

  //----------------------------------------------------------------------------
  // Acceleration
  //----------------------------------------------------------------------------

  /// Derivatives w.r.t. an arbitrary scalr variable
  Eigen::Vector3s mAccelerationsDeriv;

  //----------------------------------------------------------------------------
  // Force
  //----------------------------------------------------------------------------


  /// Derivatives w.r.t. an arbitrary scalr variable
  Eigen::Vector3s mForcesDeriv;

  //----------------------------------------------------------------------------
  // Impulse
  //----------------------------------------------------------------------------

  /// Change of generalized velocity
  Eigen::Vector3s mVelocityChanges;

//  /// Generalized impulse
//  Eigen::Vector3s mImpulse;

  /// Generalized constraint impulse
  Eigen::Vector3s mConstraintImpulses;

  //----------------------------------------------------------------------------

  /// Current position viewed in world frame.
  mutable Eigen::Vector3s mW;

  /// Current position viewed in parent soft body node frame.
  mutable Eigen::Vector3s mX;

  /// Current velocity viewed in parent soft body node frame.
  mutable Eigen::Vector3s mV;

  /// Partial Acceleration of this PointMass
  mutable Eigen::Vector3s mEta;

  ///
  Eigen::Vector3s mAlpha;

  ///
  Eigen::Vector3s mBeta;

  /// Current acceleration viewed in parent body node frame.
  mutable Eigen::Vector3s mA;

  ///
  Eigen::Vector3s mF;

  ///
  mutable s_t mPsi;

  ///
  mutable s_t mImplicitPsi;

  ///
  mutable s_t mPi;

  ///
  mutable s_t mImplicitPi;

  /// Bias force
  Eigen::Vector3s mB;

  /// External force.
  Eigen::Vector3s mFext;

  /// A increasingly sorted list of dependent dof indices.
  std::vector<std::size_t> mDependentGenCoordIndices;

  /// Whether the node is currently in collision with another node.
  bool mIsColliding;

  //------------------------- Impulse-based Dyanmics ---------------------------
  /// Velocity change due to constraint impulse
  Eigen::Vector3s mDelV;

  /// Impulsive bias force due to external impulsive force exerted on
  ///        bodies of the parent skeleton.
  Eigen::Vector3s mImpB;

  /// Cache data for mImpB
  Eigen::Vector3s mImpAlpha;

  /// Cache data for mImpB
  Eigen::Vector3s mImpBeta;

  /// Generalized impulsive body force w.r.t. body frame.
  Eigen::Vector3s mImpF;

  PointMassNotifier* mNotifier;
};

//struct PointMassPair
//{
//  PointMass* pm1;
//  PointMass* pm2;
//};

class PointMassNotifier : public Entity
{
public:

  PointMassNotifier(SoftBodyNode* _parentSoftBody, const std::string& _name);

  bool needsPartialAccelerationUpdate() const;

  void clearTransformNotice();
  void clearVelocityNotice();
  void clearPartialAccelerationNotice();
  void clearAccelerationNotice();

  void dirtyTransform() override;
  void dirtyVelocity() override;
  void dirtyAcceleration() override;

  // Documentation inherited
  const std::string& setName(const std::string& _name) override;

  // Documentation inherited
  const std::string& getName() const override;

protected:

  std::string mName;

  bool mNeedPartialAccelerationUpdate;
  // TODO(JS): Rename this to mIsPartialAccelerationDirty in DART 7

  SoftBodyNode* mParentSoftBodyNode;

};

}  // namespace dynamics
}  // namespace dart

#endif  // DART_DYNAMICS_POINTMASS_HPP_
