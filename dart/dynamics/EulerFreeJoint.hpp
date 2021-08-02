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

#ifndef DART_DYNAMICS_EULER_FREE_JOINT_HPP_
#define DART_DYNAMICS_EULER_FREE_JOINT_HPP_

#include <memory>
#include <string>
#include <vector>

#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/GenericJoint.hpp"

namespace dart {
namespace dynamics {

/**
 * This class replicates the behavior of a [<Translation>,<Euler>] joint
 * combination. The key difference is that: A) this is a single joint, rather
 * than two and B) the DOFs for the Euler joint come first
 */
class EulerFreeJoint : public GenericJoint<math::R6Space>
{
public:
  friend class Skeleton;

  EulerFreeJoint(const Properties& props);

  const std::string& getType() const override;

  /// Get joint type for this class
  static const std::string& getStaticType();

  bool isCyclic(std::size_t) const override;

  /// Set the axis order
  /// \param[in] _order Axis order
  /// \param[in] _renameDofs If true, the names of dofs in this joint will be
  /// renmaed according to the axis order.
  void setAxisOrder(EulerJoint::AxisOrder _order, bool _renameDofs = true);

  /// This takes a vector of 1's and -1's to indicate which entries to flip, if
  /// any
  void setFlipAxisMap(Eigen::Vector3s map);

  /// Return the axis order
  EulerJoint::AxisOrder getAxisOrder() const;

  dart::dynamics::Joint* clone() const override;

  void updateDegreeOfFreedomNames() override;

  void updateRelativeTransform() const override;

  /// Fixed-size version of getRelativeJacobian(positions)
  JacobianMatrix getRelativeJacobianStatic(
      const Eigen::Vector6s& position) const override;

  math::Jacobian getRelativeJacobianDeriv(std::size_t index) const override;

  void updateRelativeJacobian(bool) const override;

  void updateRelativeJacobianTimeDeriv() const override;

  /// Computes derivative of time derivative of Jacobian w.r.t. position.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index) const override;

  /// Computes derivative of time derivative of Jacobian w.r.t. velocity.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index) const override;

  ////////////////////////////////////////////////////////////////////////////
  // Public static helper methods, which are used here and in CustomJoint
  ////////////////////////////////////////////////////////////////////////////

  static Eigen::Matrix6s computeRelativeJacobianStatic(
      const Eigen::Vector6s& positions,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s computeRelativeJacobianStaticDerivWrtPos(
      const Eigen::Vector6s& positions,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s finiteDifferenceRelativeJacobianStaticDerivWrtPos(
      const Eigen::Vector6s& positions,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s computeRelativeJacobianTimeDerivStatic(
      const Eigen::Vector6s& positions,
      const Eigen::Vector6s& velocities,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s finiteDifferenceRelativeJacobianTimeDerivStatic(
      const Eigen::Vector6s& positions,
      const Eigen::Vector6s& velocities,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s computeRelativeJacobianTimeDerivDerivWrtPos(
      const Eigen::Vector6s& positions,
      const Eigen::Vector6s& velocities,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
      const Eigen::Vector6s& positions,
      const Eigen::Vector6s& velocities,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s computeRelativeJacobianTimeDerivDerivWrtVel(
      const Eigen::Vector6s& positions,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

  static Eigen::Matrix6s finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
      const Eigen::Vector6s& positions,
      const Eigen::Vector6s& velocities,
      std::size_t index,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Isometry3s childBodyToJoint);

protected:
  EulerJoint::AxisOrder mAxisOrder;
  Eigen::Vector3s mFlipAxisMap;
};

}; // namespace dynamics
}; // namespace dart

#endif