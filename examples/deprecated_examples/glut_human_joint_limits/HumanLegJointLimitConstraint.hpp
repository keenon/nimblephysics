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

#ifndef EXAMPLES_HUMANJOINTLIMITS_HUMANLEGJOINTLIMITCONSTRAINT_HPP_
#define EXAMPLES_HUMANJOINTLIMITS_HUMANLEGJOINTLIMITCONSTRAINT_HPP_

#include <dart/dart.hpp>
#include <tiny_dnn/tiny_dnn.h>

DART_COMMON_MAKE_SHARED_WEAK(HumanLegJointLimitConstraint)

/// HumanLegJointLimitConstraint handles joint position limits on human leg,
/// representing range of motion of hip, knee and ankle joints.
class HumanLegJointLimitConstraint : public dart::constraint::ConstraintBase
{
public:
  /// Constructor
  explicit HumanLegJointLimitConstraint(
      dart::dynamics::Joint* hipjoint,
      dart::dynamics::Joint* kneejoint,
      dart::dynamics::Joint* anklejoint,
      bool isMirror);

  /// Destructor
  virtual ~HumanLegJointLimitConstraint() = default;

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

  //----------------------------------------------------------------------------
  // Friendship
  //----------------------------------------------------------------------------

  friend class dart::constraint::ConstraintSolver;
  friend class dart::constraint::ConstrainedGroup;

protected:
  //----------------------------------------------------------------------------
  // Constraint virtual functions
  //----------------------------------------------------------------------------

  // Documentation inherited
  void update() override;

  // Documentation inherited
  void getInformation(dart::constraint::ConstraintInfo* lcp) override;

  // Documentation inherited
  void applyUnitImpulse(std::size_t index) override;

  // Documentation inherited
  void getVelocityChange(s_t* delVel, bool withCfm) override;

  // Documentation inherited
  void excite() override;

  // Documentation inherited
  void unexcite() override;

  // Documentation inherited
  void applyImpulse(s_t* lambda) override;

  // Documentation inherited
  dart::dynamics::SkeletonPtr getRootSkeleton() const override;

  // Documentation inherited
  bool isActive() const override;

private:
  /// leg joints involved
  dart::dynamics::Joint* mHipJoint;
  dart::dynamics::Joint* mKneeJoint;
  dart::dynamics::Joint* mAnkleJoint;

  /// leg body nodes involved
  dart::dynamics::BodyNode* mThighNode;
  dart::dynamics::BodyNode* mLowerLegNode;
  dart::dynamics::BodyNode* mFootNode;

  /// A workaround to have a de facto left-handed euler joint
  /// for right hip, so it could share same limits with left hip.
  /// left-leg set to false, right-leg set to true.
  bool mIsMirror;

  /// the neural network for calculating limits
  tiny_dnn::network<tiny_dnn::sequential> mNet;

  /// Gradient of the neural net function
  Eigen::Vector6s mJacobian;

  /// Index of applied impulse
  std::size_t mAppliedImpulseIndex;

  std::size_t mLifeTime;

  s_t mViolation;

  s_t mNegativeVel;

  s_t mOldX;

  s_t mUpperBound;

  s_t mLowerBound;

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

#endif // EXAMPLES_HUMANJOINTLIMITS_HUMANLEGJOINTLIMITCONSTRAINT_HPP_
