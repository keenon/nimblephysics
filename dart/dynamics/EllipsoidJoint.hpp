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

#ifndef DART_DYNAMICS_ELLIPSOIDJOINT_HPP_
#define DART_DYNAMICS_ELLIPSOIDJOINT_HPP_

#include <memory>
#include <string>
#include <vector>

#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/GenericJoint.hpp"
#include "dart/math/ConfigurationSpace.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/SimmSpline.hpp"

namespace dart {
namespace dynamics {

/**
 * This class is an effort to reproduce enough of OpenSim's custom joint
 * behavior in Nimble.
 */
class EllipsoidJoint : public GenericJoint<math::RealVectorSpace<3>>
{
  static constexpr std::size_t NumDofs = 3;
  static constexpr int NumDofsEigen = static_cast<int>(3);

  using TangentSpace = typename math::RealVectorSpace<3>::TangentSpace;

  using Point = typename math::RealVectorSpace<3>::Point;
  using EuclideanPoint = typename math::RealVectorSpace<3>::EuclideanPoint;
  using Vector = typename math::RealVectorSpace<3>::Vector;
  using Matrix = typename math::RealVectorSpace<3>::Matrix;
  using JacobianMatrix = typename math::RealVectorSpace<3>::JacobianMatrix;

public:
  friend class Skeleton;

  EllipsoidJoint(
      const detail::GenericJointProperties<math::RealVectorSpace<3>>& props);

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

  Eigen::Vector3s getFlipAxisMap() const;

  void setEllipsoidRadii(Eigen::Vector3s radii);

  Eigen::Vector3s getEllipsoidRadii() const;

  dart::dynamics::Joint* clone() const override;

  dart::dynamics::Joint* simplifiedClone() const override;

  void updateDegreeOfFreedomNames() override;

  void updateRelativeTransform() const override;

  static Eigen::Isometry3s getRelativeTransformStatic(
      Eigen::Isometry3s parentTransform,
      Eigen::Isometry3s childTransform,
      Eigen::Vector3s parentScale,
      Eigen::Vector3s ellipsoidScale,
      EulerJoint::AxisOrder axisOrder,
      Eigen::Vector3s position,
      Eigen::Vector3s flipAxisMap);

  /// Fixed-size version of getRelativeJacobian(positions)
  JacobianMatrix getRelativeJacobianStatic(
      const Vector& position) const override;

  JacobianMatrix getRelativeJacobianDerivWrtPositionStatic(
      std::size_t index) const override;

  JacobianMatrix getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(
      std::size_t firstIndex, std::size_t secondIndex) const;

  /// This gets the change in world translation of the child body, with respect
  /// to an axis of parent scaling. Use axis = -1 for uniform scaling of all the
  /// axis.
  Eigen::Vector3s getWorldTranslationOfChildBodyWrtParentScale(
      int axis) const override;

  /// Gets the derivative of the spatial Jacobian of the child BodyNode relative
  /// to the parent BodyNode expressed in the child BodyNode frame, with respect
  /// to the scaling of the parent body along a specific axis.
  ///
  /// Use axis = -1 for uniform scaling of all the axis.
  math::Jacobian getRelativeJacobianDerivWrtParentScale(
      int axis) const override;

  JacobianMatrix getRelativeJacobianDerivWrtPositionDerivWrtParentScale(
      std::size_t firstIndex, int axis) const;

  /// Gets the derivative of the spatial Jacobian of the child BodyNode relative
  /// to the parent BodyNode expressed in the child BodyNode frame, with respect
  /// to the scaling of the child body along a specific axis.
  ///
  /// Use axis = -1 for uniform scaling of all the axis.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtParentScale(
      int axis) const override;

  void updateRelativeJacobian(bool) const override;

  void updateRelativeJacobianTimeDeriv() const override;

  /// Computes derivative of time derivative of Jacobian w.r.t. position.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index) const override;

  /// Computes derivative of time derivative of Jacobian w.r.t. velocity.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index) const override;

  // Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6s getScrewAxisGradientForPosition(
      int axisDof, int rotateDof) override;

  // Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6s getScrewAxisGradientForForce(
      int axisDof, int rotateDof) override;

  // For testing
  Eigen::MatrixXs getScratch(int firstIndex);
  Eigen::MatrixXs analyticalScratch(int firstIndex, int secondIndex);
  Eigen::MatrixXs finiteDifferenceScratch(int firstIndex, int secondIndex);

  /// Returns the value for q that produces the nearest rotation to
  /// `relativeRotation` passed in.
  Eigen::VectorXs getNearestPositionToDesiredRotation(
      const Eigen::Matrix3s& relativeRotation) override;

protected:
  dynamics::EulerJoint::AxisOrder mAxisOrder;

  Eigen::Vector3s mEllipsoidRadii;

  /// This contains 1's and -1's to indicate whether we should flip a given
  /// input axis.
  Eigen::Vector3s mFlipAxisMap;
};

}; // namespace dynamics
}; // namespace dart

#endif