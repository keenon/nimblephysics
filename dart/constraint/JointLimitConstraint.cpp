/*
 * Copyright (c) 2011-2018, The DART development contributors
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

#include "dart/constraint/JointLimitConstraint.hpp"

#include <iostream>

#include "dart/external/odelcpsolver/lcp.h"

#include "dart/common/Console.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"

#define DART_ERROR_ALLOWANCE 0.0
#define DART_ERP     0.01
#define DART_MAX_ERV 1e+1
#define DART_CFM     1e-9

namespace dart {
namespace constraint {

double JointLimitConstraint::mErrorAllowance            = DART_ERROR_ALLOWANCE;
double JointLimitConstraint::mErrorReductionParameter   = DART_ERP;
double JointLimitConstraint::mMaxErrorReductionVelocity = DART_MAX_ERV;
double JointLimitConstraint::mConstraintForceMixing     = DART_CFM;

//==============================================================================
JointLimitConstraint::JointLimitConstraint(dynamics::Joint* joint)
  : ConstraintBase(),
    mJoint(joint),
    mBodyNode(joint->getChildBodyNode()),
    mAppliedImpulseIndex(0),
    mLifeTime(std::vector<std::size_t>(joint->getNumDofs(), 0u)),
    mActive(std::vector<bool>(joint->getNumDofs(), false))
{
  assert(joint);
  assert(mBodyNode);
}

//==============================================================================
void JointLimitConstraint::setErrorAllowance(double allowance)
{
  // Clamp error reduction parameter if it is out of the range
  if (allowance < 0.0)
  {
    dtwarn << "Error reduction parameter[" << allowance
           << "] is lower than 0.0. "
           << "It is set to 0.0." << std::endl;
    mErrorAllowance = 0.0;
  }

  mErrorAllowance = allowance;
}

//==============================================================================
double JointLimitConstraint::getErrorAllowance()
{
  return mErrorAllowance;
}

//==============================================================================
void JointLimitConstraint::setErrorReductionParameter(double erp)
{
  // Clamp error reduction parameter if it is out of the range [0, 1]
  if (erp < 0.0)
  {
    dtwarn << "Error reduction parameter[" << erp << "] is lower than 0.0. "
           << "It is set to 0.0." << std::endl;
    mErrorReductionParameter = 0.0;
  }
  if (erp > 1.0)
  {
    dtwarn << "Error reduction parameter[" << erp << "] is greater than 1.0. "
           << "It is set to 1.0." << std::endl;
    mErrorReductionParameter = 1.0;
  }

  mErrorReductionParameter = erp;
}

//==============================================================================
double JointLimitConstraint::getErrorReductionParameter()
{
  return mErrorReductionParameter;
}

//==============================================================================
void JointLimitConstraint::setMaxErrorReductionVelocity(double erv)
{
  // Clamp maximum error reduction velocity if it is out of the range
  if (erv < 0.0)
  {
    dtwarn << "Maximum error reduction velocity[" << erv
           << "] is lower than 0.0. "
           << "It is set to 0.0." << std::endl;
    mMaxErrorReductionVelocity = 0.0;
  }

  mMaxErrorReductionVelocity = erv;
}

//==============================================================================
double JointLimitConstraint::getMaxErrorReductionVelocity()
{
  return mMaxErrorReductionVelocity;
}

//==============================================================================
void JointLimitConstraint::setConstraintForceMixing(double cfm)
{
  // Clamp constraint force mixing parameter if it is out of the range
  if (cfm < 1e-9)
  {
    dtwarn << "Constraint force mixing parameter[" << cfm
           << "] is lower than 1e-9. " << "It is set to 1e-9." << std::endl;
    mConstraintForceMixing = 1e-9;
  }
  if (cfm > 1.0)
  {
    dtwarn << "Constraint force mixing parameter[" << cfm
           << "] is greater than 1.0. " << "It is set to 1.0." << std::endl;
    mConstraintForceMixing = 1.0;
  }

  mConstraintForceMixing = cfm;
}

//==============================================================================
double JointLimitConstraint::getConstraintForceMixing()
{
  return mConstraintForceMixing;
}

//==============================================================================
void JointLimitConstraint::update()
{
  const auto timeStep = mJoint->getSkeleton()->getTimeStep();
  const auto timeStepInv = 1.0 / timeStep;

  // Reset dimention
  mDim = 0;

  const auto dof = mJoint->getNumDofs();

  const Eigen::VectorXd positions = mJoint->getPositions();
  const Eigen::VectorXd positionLower = mJoint->getPositionLowerLimits();
  const Eigen::VectorXd positionUpper = mJoint->getPositionUpperLimits();

  const Eigen::VectorXd positionLowerDiff = positions - positionLower;
  const Eigen::VectorXd positionUpperDiff = positions - positionUpper;

  const Eigen::VectorXd velocities = mJoint->getVelocities();
  const Eigen::VectorXd velocityLower = mJoint->getVelocityLowerLimits();
  const Eigen::VectorXd velocityUpper = mJoint->getVelocityUpperLimits();

  const Eigen::VectorXd velocityLowerDiff = velocities - velocityLower;
  const Eigen::VectorXd velocityUpperDiff = velocities - velocityUpper;

  for (auto i = 0u; i < dof; ++i)
  {
#ifndef NDEBUG // Debug mode
    if (positionLower[i] > positionUpper[i])
    {
      dterr << "[JointLimitConstraint] The joint position limits are invalid; "
            << "The " << i << "-th upper limit '" << positionUpper[i]
            << "' is lower than the lower limit '" << positionLower[i]
            << "'.\n";
      mActive[i] = false;
      continue;
    }

    if (velocityLower[i] > velocityUpper[i])
    {
      dterr << "[JointLimitConstraint] The joint velocity limits are invalid; "
            << "The " << i << "-th upper limit '" << velocityUpper[i]
            << "' is lower than the lower limit '" << velocityLower[i]
            << "'.\n";
      mActive[i] = false;
      continue;
    }
#endif

    // Lower bound check
    if (positionLowerDiff[i] < 0.0)
    {
      mTargetVelocityChange[i] = -mJoint->getVelocity(i);

      mConstraintImpulseLowerBound[i] = 0.0;
      mConstraintImpulseUpperBound[i] = math::constantsd::inf();

      if (mActive[i])
      {
        ++(mLifeTime[i]);
      }
      else
      {
        mActive[i] = true;
        mLifeTime[i] = 0;
      }

      ++mDim;
      continue;
    }

    // Upper bound check
    if (positionUpperDiff[i] > 0.0)
    {
      mTargetVelocityChange[i] = -mJoint->getVelocity(i);

      auto bouncingVelocity = -positionLowerDiff[i] - mErrorAllowance;
      bouncingVelocity *= timeStepInv * mErrorReductionParameter;
      bouncingVelocity = std::min(bouncingVelocity, mMaxErrorReductionVelocity);

      mConstraintImpulseLowerBound[i] = -math::constantsd::inf();
      mConstraintImpulseUpperBound[i] = 0.0;

      if (mActive[i])
      {
        ++(mLifeTime[i]);
      }
      else
      {
        mActive[i] = true;
        mLifeTime[i] = 0;
      }

      ++mDim;
      continue;
    }

    auto bouncingVel = -mPositionViolation[i];

    if (bouncingVel > 0.0)
      bouncingVel = -mErrorAllowance;
    else
      bouncingVel = +mErrorAllowance;

    bouncingVel *= lcp->invTimeStep * mErrorReductionParameter;

    if (bouncingVel > mMaxErrorReductionVelocity)
      bouncingVel = mMaxErrorReductionVelocity;

    lcp->b[index] = mTargetVelocityChange[i] + bouncingVel;

    mActive[i] = false;
  }

  assert(mDim <= dof);
}

//==============================================================================
void JointLimitConstraint::getInformation(ConstraintInfo* lcp)
{
  auto index = 0u;
  const auto dof = mJoint->getNumDofs();
  for (auto i = 0u; i < dof; ++i)
  {
    if (!mActive[i])
      continue;

    assert(lcp->w[index] == 0.0);

    lcp->b[index] = mTargetVelocityChange[i];

    lcp->lo[index] = mConstraintImpulseLowerBound[i];
    lcp->hi[index] = mConstraintImpulseUpperBound[i];

    assert(lcp->findex[index] == -1);

    if (mLifeTime[i])
      lcp->x[index] = mPreviousConstraintImpulse[i];
    else
      lcp->x[index] = 0.0;

    index++;
  }

  assert(index <= dof);
}

//==============================================================================
void JointLimitConstraint::applyUnitImpulse(std::size_t index)
{
  assert(index < mDim && "Invalid Index.");

  auto localIndex = 0u;
  auto skeleton = mJoint->getSkeleton();

  const auto dof = mJoint->getNumDofs();
  for (auto i = 0u; i < dof; ++i)
  {
    if (!mActive[i])
      continue;

    if (localIndex == index)
    {
      skeleton->clearConstraintImpulses();
      mJoint->setConstraintImpulse(i, 1.0);
      skeleton->updateBiasImpulse(mBodyNode);
      skeleton->updateVelocityChange();
      mJoint->setConstraintImpulse(i, 0.0);
    }

    ++localIndex;
  }

  mAppliedImpulseIndex = index;
}

//==============================================================================
void JointLimitConstraint::getVelocityChange(double* delVel, bool withCfm)
{
  assert(delVel != nullptr && "Null pointer is not allowed.");

  auto localIndex = 0u;
  const auto dof = mJoint->getNumDofs();
  for (auto i = 0u; i < dof ; ++i)
  {
    if (!mActive[i])
      continue;

    if (mJoint->getSkeleton()->isImpulseApplied())
      delVel[localIndex] = mJoint->getVelocityChange(i);
    else
      delVel[localIndex] = 0.0;

    ++localIndex;
  }

  // Add small values to diagnal to keep it away from singular, similar to cfm
  // varaible in ODE
  if (withCfm)
  {
    delVel[mAppliedImpulseIndex] += delVel[mAppliedImpulseIndex]
                                     * mConstraintForceMixing;
  }

  assert(localIndex == mDim);
}

//==============================================================================
void JointLimitConstraint::excite()
{
  mJoint->getSkeleton()->setImpulseApplied(true);
}

//==============================================================================
void JointLimitConstraint::unexcite()
{
  mJoint->getSkeleton()->setImpulseApplied(false);
}

//==============================================================================
void JointLimitConstraint::applyImpulse(double* lambda)
{
  auto localIndex = 0u;
  auto dof = mJoint->getNumDofs();
  for (auto i = 0u; i < dof ; ++i)
  {
    if (!mActive[i])
      continue;

    mJoint->setConstraintImpulse(
          i, mJoint->getConstraintImpulse(i) + lambda[localIndex]);

    mPreviousConstraintImpulse[i] = lambda[localIndex];

    ++localIndex;
  }
}

//==============================================================================
dynamics::SkeletonPtr JointLimitConstraint::getRootSkeleton() const
{
  return mJoint->getSkeleton()->mUnionRootSkeleton.lock();
}

//==============================================================================
bool JointLimitConstraint::isActive() const
{
  for (std::size_t i = 0; i < 6; ++i)
  {
    if (mActive[i])
      return true;
  }

  return false;
}

} // namespace constraint
} // namespace dart
