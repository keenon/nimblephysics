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

#include "dart/dynamics/BallJoint.hpp"

#include <string>

#include "dart/math/Helpers.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
BallJoint::Properties::Properties(const Base::Properties& properties)
  : Base::Properties(properties)
{
  // Do nothing
}

//==============================================================================
BallJoint::~BallJoint()
{
  // Do nothing
}

//==============================================================================
const std::string& BallJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& BallJoint::getStaticType()
{
  static const std::string name = "BallJoint";
  return name;
}

//==============================================================================
bool BallJoint::isCyclic(std::size_t _index) const
{
  return _index < 3
      && !hasPositionLimit(0) && !hasPositionLimit(1) && !hasPositionLimit(2);
}

//==============================================================================
BallJoint::Properties BallJoint::getBallJointProperties() const
{
  return getGenericJointProperties();
}

//==============================================================================
Eigen::Isometry3d BallJoint::convertToTransform(
    const Eigen::Vector3d& _positions)
{
  return Eigen::Isometry3d(convertToRotation(_positions));
}

//==============================================================================
Eigen::Matrix3d BallJoint::convertToRotation(const Eigen::Vector3d& _positions)
{
  return math::expMapRot(_positions);
}

//==============================================================================
BallJoint::BallJoint(const Properties& properties)
#ifdef DART_USE_IDENTITY_JACOBIAN
  : Base(properties),
    mR(Eigen::Isometry3d::Identity())
#else
  : Base::Properties(properties)
#endif
{
  mJacobianDeriv = Eigen::Matrix<double, 6, 3>::Zero();

  // Inherited Aspects must be created in the final joint class in reverse order
  // or else we get pure virtual function calls
  createGenericJointAspect(properties);
  createJointAspect(properties);
}

//==============================================================================
Joint* BallJoint::clone() const
{
  return new BallJoint(getBallJointProperties());
}

//==============================================================================
Eigen::Matrix<double, 6, 3> BallJoint::getRelativeJacobianStatic(
    const Eigen::Vector3d& positions) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  (void)positions;
  return mJacobian;
#else
  Eigen::Matrix<double, 6, 3> J;

  const Eigen::Vector3d& q = positions;
  const Eigen::Isometry3d& T = Joint::mAspectProperties.mT_ChildBodyToJoint;

  J.topRows(3).noalias() = T.rotation() * math::so3RightJacobian(q);
  J.bottomRows(3).noalias()
      = math::makeSkewSymmetric(T.translation()) * J.topRows(3);

  return J;
#endif
}

//==============================================================================
math::Jacobian BallJoint::getRelativeJacobianDeriv(std::size_t index) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  //return finiteDifferenceRelativeJacobianDeriv(index);
  (void)index;
  return Eigen::Matrix<double, 6, 3>::Zero();
#else
  Eigen::Matrix<double, 6, 3> DS_Dq;

  const auto& q = getPositionsStatic();
  Eigen::Vector3d dq = Eigen::Vector3d::Zero();
  dq[static_cast<int>(index)] = 1;

  Eigen::Matrix3d S = math::so3RightJacobianTimeDeriv(q, dq);

  const Eigen::Isometry3d& T = Joint::mAspectProperties.mT_ChildBodyToJoint;

  DS_Dq.topRows(3).noalias() = T.rotation() * S;
  DS_Dq.bottomRows(3).noalias()
      = math::makeSkewSymmetric(T.translation()) * DS_Dq.topRows(3);

  return DS_Dq;
#endif
}

//==============================================================================
math::Jacobian BallJoint::finiteDifferenceRelativeJacobianDeriv(std::size_t index) const
{
  const auto& q = getPositionsStatic();

  const double EPS = 1e-6;
  Eigen::VectorXd tweaked = q;
  tweaked(index) += EPS;
  const_cast<BallJoint*>(this)->setPositions(tweaked);
  auto plus = getRelativeJacobian();
  tweaked = q;
  tweaked(index) -= EPS;
  const_cast<BallJoint*>(this)->setPositions(tweaked);
  auto minus = getRelativeJacobian();
  const Eigen::MatrixXd DS_Dq_num = (plus - minus) / (2 * EPS);
  const_cast<BallJoint*>(this)->setPositions(q);

  return DS_Dq_num;
}

//==============================================================================
math::Jacobian BallJoint::getRelativeJacobianTimeDerivDeriv(std::size_t index) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  //return finiteDifferenceRelativeJacobianTimeDerivDeriv(index);
  (void)index;
  return Eigen::Matrix<double, 6, 3>::Zero();
#else
  Eigen::Matrix<double, 6, 3> DdS_Dq;

  const auto& q = getPositionsStatic();
  const auto& dq = getVelocitiesStatic();

  const Eigen::Matrix3d S
      = math::so3RightJacobianTimeDerivDeriv(q, dq, static_cast<int>(index));
  const Eigen::Isometry3d& T = Joint::mAspectProperties.mT_ChildBodyToJoint;

  DdS_Dq.topRows(3).noalias() = T.rotation() * S;
  DdS_Dq.bottomRows(3).noalias()
      = math::makeSkewSymmetric(T.translation()) * DdS_Dq.topRows(3);

  return DdS_Dq;
#endif
}

//==============================================================================
math::Jacobian BallJoint::finiteDifferenceRelativeJacobianTimeDerivDeriv(std::size_t index) const
{
  const auto& q = getPositionsStatic();

  const double EPS = 1e-6;
  Eigen::VectorXd tweaked = q;
  tweaked(index) += EPS;
  const_cast<BallJoint*>(this)->setPositions(tweaked);
  auto plus = getRelativeJacobianTimeDeriv();
  tweaked = q;
  tweaked(index) -= EPS;
  const_cast<BallJoint*>(this)->setPositions(tweaked);
  auto minus = getRelativeJacobianTimeDeriv();
  const Eigen::MatrixXd DS_Dq_num = (plus - minus) / (2 * EPS);
  const_cast<BallJoint*>(this)->setPositions(q);

  return DS_Dq_num;
}

//==============================================================================
math::Jacobian BallJoint::getRelativeJacobianTimeDerivDeriv2(std::size_t index) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  // return finiteDifferenceRelativeJacobianTimeDerivDeriv2(index);
  (void)index;
  return Eigen::Matrix<double, 6, 3>::Zero();
#else
  Eigen::Matrix<double, 6, 3> DdS_Dq;

  const auto& q = getPositionsStatic();
  const auto& dq = getVelocitiesStatic();

  const Eigen::Matrix3d S
      = math::so3RightJacobianTimeDerivDeriv2(q, dq, static_cast<int>(index));
  const Eigen::Isometry3d& T = Joint::mAspectProperties.mT_ChildBodyToJoint;

  DdS_Dq.topRows(3).noalias() = T.rotation() * S;
  DdS_Dq.bottomRows(3).noalias()
      = math::makeSkewSymmetric(T.translation()) * DdS_Dq.topRows(3);

  return DdS_Dq;
#endif
}

//==============================================================================
math::Jacobian BallJoint::finiteDifferenceRelativeJacobianTimeDerivDeriv2(std::size_t index) const
{
  const auto& dq = getVelocitiesStatic();

  const double EPS = 1e-6;
  Eigen::VectorXd tweaked = dq;
  tweaked(index) += EPS;
  const_cast<BallJoint*>(this)->setVelocities(tweaked);
  auto plus = getRelativeJacobianTimeDeriv();
  tweaked = dq;
  tweaked(index) -= EPS;
  const_cast<BallJoint*>(this)->setVelocities(tweaked);
  auto minus = getRelativeJacobianTimeDeriv();
  const Eigen::MatrixXd DS_Dq_num = (plus - minus) / (2 * EPS);
  const_cast<BallJoint*>(this)->setVelocities(dq);

  return DS_Dq_num;
}

//==============================================================================
Eigen::Matrix<double, 6, 3> BallJoint::getRelativeJacobianInPositionSpaceStatic(
    const Eigen::Vector3d& _positions) const {
  Eigen::Matrix<double, 6, 3> J;
  J.topLeftCorner<3, 3>().noalias()
      = math::expMapJac(_positions).transpose();
  J.bottomLeftCorner<3, 3>().setZero();
  return math::AdTJacFixed(Joint::mAspectProperties.mT_ChildBodyToJoint, J);
}

//==============================================================================
Eigen::Vector3d BallJoint::getPositionDifferencesStatic(
    const Eigen::Vector3d& _q2, const Eigen::Vector3d& _q1) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  const Eigen::Matrix3d R1 = convertToRotation(_q1);
  const Eigen::Matrix3d R2 = convertToRotation(_q2);

  return convertToPositions(R1.transpose() * R2);
#else
  const Eigen::Matrix3d R1 = convertToRotation(_q1);
  const Eigen::Matrix3d R2 = convertToRotation(_q2);
  const Eigen::Matrix3d S = math::so3RightJacobian(_q1);

  return S.inverse() * convertToPositions(R1.transpose() * R2);
#endif
}

//==============================================================================
void BallJoint::integratePositions(double dt)
{
  const Eigen::Vector3d& q = getPositionsStatic();
  const Eigen::Vector3d& dq = getVelocitiesStatic();

  setPositionsStatic(integratePositionsExplicit(q, dq, dt));
}

//==============================================================================
#ifndef DART_USE_IDENTITY_JACOBIAN
void BallJoint::integrateVelocities(double dt)
{
  const Eigen::Vector3d& dq = getVelocitiesStatic();
  const Eigen::Vector3d& ddq = getAccelerationsStatic();

  const auto& S = getRelativeJacobian().topRows(3);
  const auto& dS = getRelativeJacobianTimeDeriv().topRows(3);

  setVelocitiesStatic(S.inverse() * (S * dq + dt * (dS * dq + S * ddq)));
}
#endif

//==============================================================================
Eigen::VectorXd BallJoint::integratePositionsExplicit(
    const Eigen::VectorXd& pos, const Eigen::VectorXd& vel, double dt) {
#ifdef DART_USE_IDENTITY_JACOBIAN
  const auto& q = pos;
  const auto& dq = vel;
  Eigen::Matrix3d Rnext
      = convertToRotation(q) * convertToRotation(dq * dt);
#else
  const auto& q = pos;
  const auto& dq = vel;
  const Eigen::Matrix3d S = math::so3RightJacobian(q);
  const Eigen::Matrix3d Rnext
      = convertToRotation(q) * convertToRotation(S * dq * dt);
#endif
  return convertToPositions(Rnext);
}

//==============================================================================
Eigen::MatrixXd BallJoint::getPosPosJacobian(
    const Eigen::VectorXd& pos, const Eigen::VectorXd& vel, double _dt) {
  // TODO
  return finiteDifferencePosPosJacobian(pos, vel, _dt);
}

//==============================================================================
Eigen::MatrixXd BallJoint::getVelPosJacobian(
    const Eigen::VectorXd& pos, const Eigen::VectorXd& vel, double _dt) {
  // TODO
  return finiteDifferenceVelPosJacobian(pos, vel, _dt);
}

//==============================================================================
/// Returns d/dpos of integratePositionsExplicit() by finite differencing
Eigen::MatrixXd BallJoint::finiteDifferencePosPosJacobian(
    const Eigen::VectorXd& pos, const Eigen::VectorXd& vel, double dt)
{
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3, 3);
  double EPS = 1e-6;
  for (int i = 0; i < 3; i++) {
    Eigen::VectorXd perturbed = pos;
    perturbed(i) += EPS;
    Eigen::VectorXd plus = integratePositionsExplicit(perturbed, vel, dt);

    perturbed = pos;
    perturbed(i) -= EPS;
    Eigen::VectorXd minus = integratePositionsExplicit(perturbed, vel, dt);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
/// Returns d/dvel of integratePositionsExplicit() by finite differencing
Eigen::MatrixXd BallJoint::finiteDifferenceVelPosJacobian(
    const Eigen::VectorXd& pos, const Eigen::VectorXd& vel, double dt)
{
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3, 3);
  double EPS = 1e-7;
  for (int i = 0; i < 3; i++) {
    Eigen::VectorXd perturbed = vel;
    perturbed(i) += EPS;
    Eigen::VectorXd plus = integratePositionsExplicit(pos, perturbed, dt);

    perturbed = vel;
    perturbed(i) -= EPS;
    Eigen::VectorXd minus = integratePositionsExplicit(pos, perturbed, dt);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
void BallJoint::updateDegreeOfFreedomNames()
{
  if(!mDofs[0]->isNamePreserved())
    mDofs[0]->setName(Joint::mAspectProperties.mName + "_x", false);
  if(!mDofs[1]->isNamePreserved())
    mDofs[1]->setName(Joint::mAspectProperties.mName + "_y", false);
  if(!mDofs[2]->isNamePreserved())
    mDofs[2]->setName(Joint::mAspectProperties.mName + "_z", false);
}

//==============================================================================
void BallJoint::updateRelativeTransform() const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  mR.linear() = convertToRotation(getPositionsStatic());

  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * mR
      * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
#else
  Eigen::Isometry3d R = Eigen::Isometry3d::Identity();
  R.linear() = convertToRotation(getPositionsStatic());

  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * R
      * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
#endif

  assert(math::verifyTransform(mT));
}

//==============================================================================
void BallJoint::updateRelativeJacobian(bool _mandatory) const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  if (_mandatory)
  {
    mJacobian = math::getAdTMatrix(
          Joint::mAspectProperties.mT_ChildBodyToJoint).leftCols<3>();
  }
#else
  (void)_mandatory;
  mJacobian = getRelativeJacobianStatic(getPositionsStatic());
#endif
}

//==============================================================================
void BallJoint::updateRelativeJacobianTimeDeriv() const
{
#ifdef DART_USE_IDENTITY_JACOBIAN
  assert(Eigen::Matrix6d::Zero().leftCols<3>() == mJacobianDeriv);
#else
  const auto& q = getPositionsStatic();
  const auto& dq = getVelocitiesStatic();
  const Eigen::Isometry3d& T = Joint::mAspectProperties.mT_ChildBodyToJoint;

  const Eigen::Matrix3d dJ = math::so3RightJacobianTimeDeriv(q, dq);

  mJacobianDeriv.topRows(3).noalias() = T.rotation() * dJ;
  mJacobianDeriv.bottomRows(3).noalias()
      = math::makeSkewSymmetric(T.translation()) * mJacobianDeriv.topRows(3);
#endif
}

/*
//==============================================================================
Eigen::Vector6d BallJoint::getWorldAxisScrewForPosition(int dof) const
{
  return getWorldAxisScrewAt(getPositionsStatic(), dof);
}
*/

//==============================================================================
// This computes the world axis screw at a given position, without moving the joint.
//
// We do this relative to the parent body, rather than the child body, because in
// moving the joint we also move the child body.
Eigen::Vector6d BallJoint::getWorldAxisScrewAt(Eigen::Vector3d pos, int dof) const
{
  Eigen::Vector6d grad = Eigen::Vector6d::Zero();
  grad.head<3>() = math::expMapJac(pos).col(dof);
  Eigen::Vector6d parentTwist = math::AdT(Joint::mAspectProperties.mT_ParentBodyToJoint, grad);

  Eigen::Isometry3d parentTransform = Eigen::Isometry3d::Identity();
  if (getParentBodyNode() != nullptr) {
    parentTransform = getParentBodyNode()->getWorldTransform();
  }
  return math::AdT(parentTransform, parentTwist);
}

//==============================================================================
// This estimates the new world screw axis at `axisDof` when we perturbe `rotateDof` by `eps`
Eigen::Vector6d BallJoint::estimatePerturbedScrewAxisForPosition(
  int axisDof,
  int rotateDof,
  double eps)
{
  Eigen::Vector3d pos = getPositionsStatic();
  pos(rotateDof) += eps;
  return getWorldAxisScrewAt(pos, axisDof);
}

//==============================================================================
// This estimates the new world screw axis at `axisDof` when we perturbe `rotateDof` by `eps`
Eigen::Vector6d BallJoint::estimatePerturbedScrewAxisForForce(
  int axisDof,
  int rotateDof,
  double eps)
{
  Eigen::Vector3d pos = getPositionsStatic();
  pos(rotateDof) += eps;

  Eigen::Isometry3d parentTransform = Eigen::Isometry3d::Identity();
  if (getParentBodyNode() != nullptr) {
    parentTransform = getParentBodyNode()->getWorldTransform();
  }
  return math::AdT(
    parentTransform * Joint::mAspectProperties.mT_ParentBodyToJoint * convertToTransform(pos)
        * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse(), getRelativeJacobian(pos).col(axisDof));
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6d BallJoint::getScrewAxisGradientForPosition(
  int axisDof,
  int rotateDof)
{
  double EPS = 1e-7;
  Eigen::Vector6d pos = estimatePerturbedScrewAxisForPosition(axisDof, rotateDof, EPS);
  Eigen::Vector6d neg = estimatePerturbedScrewAxisForPosition(axisDof, rotateDof, -EPS);
  return (pos - neg) / (2 * EPS);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6d BallJoint::getScrewAxisGradientForForce(
  int axisDof,
  int rotateDof)
{
  // getRelativeJacobian() is constant wrt position
  // toRotate is also constant wrt position
  // Eigen::Vector6d toRotate = Eigen::Vector6d::Unit(axisDof);
  Eigen::Vector6d toRotate = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint.inverse(), getRelativeJacobian().col(axisDof));
  Eigen::Vector6d grad = Eigen::Vector6d::Zero();
  Eigen::Matrix3d rotate = math::expMapRot(getPositionsStatic());
  Eigen::Vector3d screwAxis = math::expMapJac(getPositionsStatic()).row(rotateDof);
  grad.head<3>() = rotate * screwAxis.cross(toRotate.head<3>());
  grad.tail<3>() = rotate * screwAxis.cross(toRotate.tail<3>());

  Eigen::Isometry3d parentTransform = Eigen::Isometry3d::Identity();
  if (getParentBodyNode() != nullptr) {
    parentTransform = getParentBodyNode()->getWorldTransform();
  }
  return math::AdT(
    parentTransform * Joint::mAspectProperties.mT_ParentBodyToJoint, grad);
}

#ifdef DART_USE_IDENTITY_JACOBIAN
//==============================================================================
const Eigen::Isometry3d& BallJoint::getR() const
{
  if(mNeedTransformUpdate)
  {
    updateRelativeTransform();
    mNeedTransformUpdate = false;
  }

  return mR;
}
#endif

}  // namespace dynamics
}  // namespace dart
