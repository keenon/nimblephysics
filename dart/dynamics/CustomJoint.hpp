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
#include "dart/math/ConfigurationSpace.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/SimmSpline.hpp"

namespace dart {
namespace dynamics {

/**
 * This class is an effort to reproduce enough of OpenSim's custom joint
 * behavior in Nimble to load the Rajagopal model, without sacrificing
 * performance or differentiability.
 */
template <std::size_t Dimension>
class CustomJoint : public GenericJoint<math::RealVectorSpace<Dimension>>
{
  static constexpr std::size_t NumDofs = Dimension;
  static constexpr int NumDofsEigen = static_cast<int>(Dimension);

  using TangentSpace = typename math::RealVectorSpace<Dimension>::TangentSpace;

  using Point = typename math::RealVectorSpace<Dimension>::Point;
  using EuclideanPoint =
      typename math::RealVectorSpace<Dimension>::EuclideanPoint;
  using Vector = typename math::RealVectorSpace<Dimension>::Vector;
  using Matrix = typename math::RealVectorSpace<Dimension>::Matrix;
  using JacobianMatrix =
      typename math::RealVectorSpace<Dimension>::JacobianMatrix;

public:
  friend class Skeleton;

  CustomJoint<Dimension>(
      const detail::GenericJointProperties<math::RealVectorSpace<Dimension>>&
          props);

  /// This sets a custom function to map our single input degree of freedom to
  /// the wrapped Euler joint's degree of freedom and index i
  void setCustomFunction(
      std::size_t i, std::shared_ptr<math::CustomFunction> fn, int drivenByDof);

  std::shared_ptr<math::CustomFunction> getCustomFunction(std::size_t i);

  /// There is an annoying tendency for custom joints to encode the linear
  /// offset of the bone in their custom functions. We don't want that, so we
  /// want to move any relative transform caused by custom functions into the
  /// parent transform.
  void zeroTranslationInCustomFunctions();

  int getCustomFunctionDrivenByDof(std::size_t i);

  /// This gets the Jacobian of the mapping functions. That is, for every
  /// epsilon change in dof=x, how does each custom function change?
  math::Jacobian getCustomFunctionGradientAt(const Eigen::VectorXs& x) const;

  /// This gets the time derivative of the Jacobian of the mapping functions.
  math::Jacobian getCustomFunctionGradientAtTimeDeriv(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  math::Jacobian finiteDifferenceCustomFunctionGradientAtTimeDeriv(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  math::Jacobian finiteDifferenceCustomFunctionGradientAt(
      const Eigen::VectorXs& x, bool useRidders = true) const;

  math::Jacobian getCustomFunctionGradientAtTimeDerivPosDeriv(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx,
      int index) const;

  math::Jacobian finiteDifferenceCustomFunctionGradientAtTimeDerivPosDeriv(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx,
      int index) const;

  math::Jacobian getCustomFunctionGradientAtTimeDerivVelDeriv(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx,
      int index) const;

  math::Jacobian finiteDifferenceCustomFunctionGradientAtTimeDerivVelDeriv(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx,
      int index) const;

  /// This gets the array of 2nd order derivatives at x
  math::Jacobian getCustomFunctionSecondGradientAt(
      const Eigen::VectorXs& x) const;

  /// This produces the positions of each of the mapping functions, at a given
  /// point of input.
  Eigen::Vector6s getCustomFunctionPositions(const Eigen::VectorXs& x) const;

  /// This produces the velocities of each of the mapping functions, at a given
  /// point with a specific velocity.
  Eigen::Vector6s getCustomFunctionVelocities(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  /// This produces the accelerations of each of the mapping functions, at a
  /// given point with a specific acceleration.
  Eigen::Vector6s getCustomFunctionAccelerations(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in position x
  math::Jacobian getCustomFunctionVelocitiesDerivativeWrtPos(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  math::Jacobian finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in position x
  math::Jacobian getCustomFunctionAccelerationsDerivativeWrtPos(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

  math::Jacobian finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

  /// This produces the derivative of the accelerations with respect to changes
  /// in velocity dx
  math::Jacobian getCustomFunctionAccelerationsDerivativeWrtVel(
      const Eigen::VectorXs& x) const;

  math::Jacobian finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

  /// This returns the first 3 custom function outputs
  Eigen::Vector3s getEulerPositions(const Eigen::VectorXs& x) const;

  /// This returns the first 3 custom function outputs's derivatives
  Eigen::Vector3s getEulerVelocities(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  /// This returns the first 3 custom function outputs's second derivatives
  Eigen::Vector3s getEulerAccelerations(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

  /// This returns the last 3 custom function outputs
  Eigen::Vector3s getTranslationPositions(const Eigen::VectorXs& x) const;

  /// This returns the last 3 custom function outputs's derivatives
  Eigen::Vector3s getTranslationVelocities(
      const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const;

  /// This returns the last 3 custom function outputs's second derivatives
  Eigen::Vector3s getTranslationAccelerations(
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& dx,
      const Eigen::VectorXs& ddx) const;

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

  dart::dynamics::Joint* clone() const override;

  dart::dynamics::Joint* simplifiedClone() const override;

  void updateDegreeOfFreedomNames() override;

  void updateRelativeTransform() const override;

  /// Fixed-size version of getRelativeJacobian(positions)
  JacobianMatrix getRelativeJacobianStatic(
      const Vector& position) const override;

  Eigen::Matrix6s getSpatialJacobianStaticDerivWrtInput(
      const Eigen::VectorXs& pos, std::size_t index) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianStaticDerivWrtInput(
      const Eigen::VectorXs& pos,
      std::size_t index,
      bool useRidders = true) const;

  JacobianMatrix getRelativeJacobianDerivWrtPositionStatic(
      std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianDeriv(
      std::size_t index, bool useRidders = true);

  void updateRelativeJacobian(bool) const override;

  void updateRelativeJacobianTimeDeriv() const override;

  Eigen::Matrix6s getSpatialJacobianTimeDerivDerivWrtInputPos(
      const Eigen::VectorXs& pos,
      const Eigen::VectorXs& vel,
      std::size_t index) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
      const Eigen::VectorXs& pos,
      const Eigen::VectorXs& vel,
      std::size_t index,
      bool useRidders = true) const;

  Eigen::Matrix6s getSpatialJacobianTimeDerivDerivWrtInputVel(
      const Eigen::VectorXs& pos, std::size_t index) const;

  Eigen::Matrix6s finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
      const Eigen::VectorXs& pos,
      const Eigen::VectorXs& vel,
      std::size_t index,
      bool useRidders = true) const;

  /// Computes derivative of time derivative of Jacobian w.r.t. position.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
      std::size_t index, bool useRidders = true);

  /// Computes derivative of time derivative of Jacobian w.r.t. velocity.
  math::Jacobian getRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index) const override;

  math::Jacobian finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
      std::size_t index, bool useRidders = true);

  // Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6s getScrewAxisGradientForPosition(
      int axisDof, int rotateDof) override;

  // Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6s getScrewAxisGradientForForce(
      int axisDof, int rotateDof) override;

  /// Returns the value for q that produces the nearest rotation to
  /// `relativeRotation` passed in.
  Eigen::VectorXs getNearestPositionToDesiredRotation(
      const Eigen::Matrix3s& relativeRotation) override;

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

  // Each function is driven by a single degree of freedom
  std::vector<int> mFunctionDrivenByDof;
};

}; // namespace dynamics
}; // namespace dart

#endif