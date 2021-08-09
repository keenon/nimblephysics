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

#ifndef DART_DYNAMICS_CUSTOMJOINT_HPP_
#define DART_DYNAMICS_CUSTOMJOINT_HPP_

#include <memory>
#include <string>
#include <vector>

#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/GenericJoint.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/SimmSpline.hpp"

namespace dart {
namespace dynamics {

/**
 * This class is an effort to reproduce enough of OpenSim's custom joint
 * behavior in Nimble to load the Rajagopal model, without sacrificing
 * performance or differentiability.
 */
class CustomJoint : public GenericJoint<math::R1Space>
{
public:
  friend class Skeleton;

  CustomJoint(const Properties& props);

  /// This sets a custom function to map our single input degree of freedom to
  /// the wrapped Euler joint's degree of freedom and index i
  void setCustomFunction(
      std::size_t i, std::shared_ptr<math::CustomFunction> fn);

  /// This gets the Jacobian of the mapping functions. That is, for every
  /// epsilon change in x, how does each custom function change?
  Eigen::Vector6s getCustomFunctionGradientAt(s_t x) const;

  Eigen::Vector6s finiteDifferenceCustomFunctionGradientAt(
      s_t x) const;

  /// This gets the array of 2nd order derivatives at x
  Eigen::Vector6s getCustomFunctionSecondGradientAt(s_t x) const;

  /// This produces the positions of each of the mapping functions, at a given
  /// point of input.
  Eigen::Vector6s getCustomFunctionPositions(s_t x) const;

  /// This produces the velocities of each of the mapping functions, at a given
  /// point with a specific velocity.
  Eigen::Vector6s getCustomFunctionVelocities(s_t x, s_t dx) const;

  /// This produces the accelerations of each of the mapping functions, at a
  /// given point with a specific acceleration.
  Eigen::Vector6s getCustomFunctionAccelerations(s_t x, s_t dx, s_t ddx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in position x
  Eigen::Vector6s getCustomFunctionVelocitiesDerivativeWrtPos(
      s_t x, s_t dx) const;

  Eigen::Vector6s finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
      s_t x, s_t dx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in position x
  Eigen::Vector6s getCustomFunctionAccelerationsDerivativeWrtPos(
      s_t x, s_t dx, s_t ddx) const;

  Eigen::Vector6s finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
      s_t x, s_t dx, s_t ddx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in velocity dx
  Eigen::Vector6s getCustomFunctionAccelerationsDerivativeWrtVel(s_t x) const;

  Eigen::Vector6s finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
      s_t x, s_t dx, s_t ddx) const;

  /// This returns the first 3 custom function outputs
  Eigen::Vector3s getEulerPositions(s_t x) const;

  /// This returns the first 3 custom function outputs's derivatives
  Eigen::Vector3s getEulerVelocities(s_t x, s_t dx) const;

  /// This returns the first 3 custom function outputs's second derivatives
  Eigen::Vector3s getEulerAccelerations(s_t x, s_t dx, s_t ddx) const;

  /// This returns the last 3 custom function outputs
  Eigen::Vector3s getTranslationPositions(s_t x) const;

  /// This returns the last 3 custom function outputs's derivatives
  Eigen::Vector3s getTranslationVelocities(s_t x, s_t dx) const;

  /// This returns the last 3 custom function outputs's second derivatives
  Eigen::Vector3s getTranslationAccelerations(s_t x, s_t dx, s_t ddx) const;

  const std::string& getType() const override;

  /// Get joint type for this class
  static const std::string& getStaticType();

  bool isCyclic(std::size_t) const override;

  /// Set the axis order
  /// \param[in] _order Axis order
  /// \param[in] _renameDofs If true, the names of dofs in this joint will be
  /// renmaed according to the axis order.
  void setAxisOrder(EulerJoint::AxisOrder _order, bool _renameDofs = true);

  /// Return the axis order
  EulerJoint::AxisOrder getAxisOrder() const;

  /// This takes a vector of 1's and -1's to indicate which entries to flip, if
  /// any
  void setFlipAxisMap(Eigen::Vector3s map);

  Eigen::Vector3s getFlipAxisMap();

  dart::dynamics::Joint* clone() const override;

  void updateDegreeOfFreedomNames() override;

  void updateRelativeTransform() const override;

  /// Fixed-size version of getRelativeJacobian(positions)
  JacobianMatrix getRelativeJacobianStatic(
      const Eigen::Vector1s& position) const override;

  Eigen::Matrix6s getSpatialJacobianStaticDerivWrtInput(s_t pos) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianStaticDerivWrtInput(
      s_t pos, bool useRidders = true) const;

  Eigen::Matrix6s finiteDifferenceRiddersSpatialJacobianStaticDerivWrtInput(
      s_t pos) const;

  math::Jacobian getRelativeJacobianDeriv(std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianDeriv(std::size_t index, bool useRidders = true);

  math::Jacobian finiteDifferenceRiddersRelativeJacobianDeriv(std::size_t index);

  void updateRelativeJacobian(bool) const override;

  void updateRelativeJacobianTimeDeriv() const override;

  Eigen::Matrix6s getSpatialJacobianTimeDerivDerivWrtInputPos(
      s_t pos, s_t vel) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
      s_t pos, s_t vel, bool useRidders = true) const;

  Eigen::Matrix6s finiteDifferenceRiddersSpatialJacobianTimeDerivDerivWrtInputPos(
      s_t pos, s_t vel) const;

  Eigen::Matrix6s getSpatialJacobianTimeDerivDerivWrtInputVel(s_t pos) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
      s_t pos, s_t vel, bool useRidders = true) const;

  Eigen::Matrix6s finiteDifferenceRiddersSpatialJacobianTimeDerivDerivWrtInputVel(
      s_t pos, s_t vel) const;

  /// Computes derivative of time derivative of Jacobian w.r.t. position.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index, bool useRidders = true);

  math::Jacobian finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index);

  /// Computes derivative of time derivative of Jacobian w.r.t. velocity.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index, bool useRidders = true);

  math::Jacobian finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index);

  ///////////////////////////////////////////////////////////////////////////
  // Only for use during development and testing of Jacobians.
  ///////////////////////////////////////////////////////////////////////////
  Eigen::Vector6s scratch();

  Eigen::Vector6s scratchFd();

  Eigen::Vector6s scratchAnalytical();

protected:
  dynamics::EulerJoint::AxisOrder mAxisOrder;

  /// This contains 1's and -1's to indicate whether we should flip a given
  /// input axis.
  Eigen::Vector3s mFlipAxisMap;

  // There should be 6 of these, one for each axis of the wrapped Euler joint
  std::vector<std::shared_ptr<math::CustomFunction>> mFunctions;
};

}; // namespace dynamics
}; // namespace dart

#endif