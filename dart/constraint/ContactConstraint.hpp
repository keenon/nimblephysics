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

#ifndef DART_CONSTRAINT_CONTACTCONSTRAINT_HPP_
#define DART_CONSTRAINT_CONTACTCONSTRAINT_HPP_

#include "dart/collision/CollisionDetector.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {

namespace dynamics {
class BodyNode;
class Skeleton;
} // namespace dynamics

namespace constraint {

/// ContactConstraint represents a contact constraint between two bodies
class ContactConstraint : public ConstraintBase
{
public:
  /// Constructor
  ContactConstraint(
      collision::Contact& contact,
      s_t timeStep,
      bool penetrationCorrectionEnabled);

  /// Destructor
  ~ContactConstraint() override = default;

  //----------------------------------------------------------------------------
  // Property settings
  //----------------------------------------------------------------------------

  /// Set global error reduction parameter
  static void setErrorAllowance(s_t allowance);

  /// Get global error reduction parameter
  static s_t getErrorAllowance();

  /// Set global error reduction parameter
  static void setErrorReductionParameter(s_t erp);

  /// Get global error reduction parameter
  static s_t getErrorReductionParameter();

  /// Set global error reduction parameter
  static void setMaxErrorReductionVelocity(s_t erv);

  /// Get global error reduction parameter
  static s_t getMaxErrorReductionVelocity();

  /// Set global constraint force mixing parameter
  static void setConstraintForceMixing(s_t cfm);

  /// Get global constraint force mixing parameter
  static s_t getConstraintForceMixing();

  /// Set first frictional direction
  void setFrictionDirection(const Eigen::Vector3s& dir);

  /// Get first frictional direction
  const Eigen::Vector3s& getFrictionDirection1() const;

  // Returns the contact
  const collision::Contact& getContact() const;

  // Returns body node A
  const dynamics::BodyNode* getBodyNodeA() const;

  // Returns body node B
  const dynamics::BodyNode* getBodyNodeB() const;

  /// Get contact Jacobian for bodyNodeA
  const Eigen::MatrixXs getSpatialNormalA() const;

  /// Get contact Jacobian for bodyNodeB
  const Eigen::MatrixXs getSpatialNormalB() const;

  /// Check whether friction is on.
  bool isFrictionOn() const;

  //----------------------------------------------------------------------------
  // Friendship
  //----------------------------------------------------------------------------

  friend class ConstraintSolver;
  friend class ConstrainedGroup;

protected:
  //----------------------------------------------------------------------------
  // Constraint virtual functions
  //----------------------------------------------------------------------------

  // Documentation inherited
  void update() override;

  // Documentation inherited
  void getInformation(ConstraintInfo* info) override;

  // Documentation inherited
  void applyUnitImpulse(std::size_t index) override;

  // Documentation inherited
  void getVelocityChange(s_t* vel, bool withCfm) override;

  // Documentation inherited
  void excite() override;

  // Documentation inherited
  void unexcite() override;

  // Documentation inherited
  void applyImpulse(Eigen::VectorXs lambda) override;

  // Documentation inherited
  dynamics::SkeletonPtr getRootSkeleton() const override;

  // Documentation inherited
  void uniteSkeletons() override;

  // Documentation inherited
  std::vector<dynamics::SkeletonPtr> getSkeletons() const override;

  // Documentation inherited
  bool isActive() const override;

public:
  /// Returns true
  bool isContactConstraint() const override;

  /// Returns 0 if this constraint isn't bouncing, otherwise returns the
  /// coefficient of restitution
  s_t getCoefficientOfRestitution() override;

  /// Returns 0 if this constraint isn't using the "bouncing" hack to correct
  /// penetration. Otherwise, this returns the velocity being used by the
  /// penetration correction hack.
  s_t getPenetrationCorrectionVelocity() override;

  using TangentBasisMatrix = Eigen::Matrix<s_t, 3, 2>;

  /// Get change in relative velocity at contact point due to external impulse
  /// \param[out] relVel Change in relative velocity at contact point of the
  /// two colliding bodies.
  void getRelVelocity(s_t* relVel);

  ///
  void updateFirstFrictionalDirection();

  TangentBasisMatrix getTangentBasisMatrixODE(const Eigen::Vector3s& n);

  /// This returns the gradient of each element of the Tangent basis matrix, if
  /// `g` is the gradient of `n` with respect to whatever scalar we care about.
  TangentBasisMatrix getTangentBasisMatrixODEGradient(
      const Eigen::Vector3s& n, const Eigen::Vector3s& g);

private:
  /// Time step
  s_t mTimeStep;

  /// Fircst body node
  dynamics::BodyNodePtr mBodyNodeA;

  /// Second body node
  dynamics::BodyNodePtr mBodyNodeB;

  /// Contact between mBodyNode1 and mBodyNode2
  collision::Contact& mContact;

  /// First frictional direction
  Eigen::Vector3s mFirstFrictionalDirection;

  /// Coefficient of Friction
  s_t mFrictionCoeff;

  /// Coefficient of restitution
  s_t mRestitutionCoeff;

  /// This is the component of the contact force due to penetration correction
  s_t mPenetrationCorrectionVelocity;

  /// True if we are going fast enough to bounce, used for gradient computation
  bool mDidBounce;

  /// Whether this contact is self-collision.
  bool mIsSelfCollision;

  /// Whether to enable penetration correction forces.
  bool mPenetrationCorrectionEnabled;

  /// Local body jacobians for mBodyNode1
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> mSpatialNormalA;

  /// Local body jacobians for mBodyNode2
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> mSpatialNormalB;

  ///
  bool mIsFrictionOn;

  /// Index of applied impulse
  std::size_t mAppliedImpulseIndex;

  ///
  bool mIsBounceOn;

  ///
  bool mActive;

  /// Global constraint error allowance
  static s_t mErrorAllowance;

  /// Global constraint error redection parameter in the range of [0, 1]. The
  /// default is 0.01.
  static s_t mErrorReductionParameter;

  /// Maximum error reduction velocity
  static s_t mMaxErrorReductionVelocity;

  /// Global constraint force mixing parameter in the range of [1e-9, 1]. The
  /// default is 1e-5
  /// \sa http://www.ode.org/ode-latest-userguide.html#sec_3_8_0
  static s_t mConstraintForceMixing;
};
// TODO(JS): Create SelfContactConstraint.

} // namespace constraint
} // namespace dart

#endif // DART_CONSTRAINT_CONTACTCONSTRAINT_HPP_
