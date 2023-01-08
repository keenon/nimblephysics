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

#include "dart/dynamics/UniversalJoint.hpp"

#include <limits>
#include <string>

#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
UniversalJoint::~UniversalJoint()
{
  // Do nothing
}

//==============================================================================
void UniversalJoint::setProperties(const Properties& _properties)
{
  GenericJoint<math::R2Space>::setProperties(
      static_cast<const GenericJoint<math::R2Space>::Properties&>(_properties));
  setProperties(static_cast<const UniqueProperties&>(_properties));
}

//==============================================================================
void UniversalJoint::setProperties(const UniqueProperties& _properties)
{
  setAspectProperties(_properties);
}

//==============================================================================
void UniversalJoint::setAspectProperties(const AspectProperties& properties)
{
  setAxis1(properties.mAxis[0]);
  setAxis2(properties.mAxis[1]);
}

//==============================================================================
UniversalJoint::Properties UniversalJoint::getUniversalJointProperties() const
{
  return Properties(getGenericJointProperties(), mAspectProperties);
}

//==============================================================================
void UniversalJoint::copy(const UniversalJoint& _otherJoint)
{
  if (this == &_otherJoint)
    return;

  setProperties(_otherJoint.getUniversalJointProperties());
}

//==============================================================================
void UniversalJoint::copy(const UniversalJoint* _otherJoint)
{
  if (nullptr == _otherJoint)
    return;

  copy(*this);
}

//==============================================================================
UniversalJoint& UniversalJoint::operator=(const UniversalJoint& _otherJoint)
{
  copy(_otherJoint);
  return *this;
}

//==============================================================================
const std::string& UniversalJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& UniversalJoint::getStaticType()
{
  static const std::string name = "UniversalJoint";
  return name;
}

//==============================================================================
bool UniversalJoint::isCyclic(std::size_t _index) const
{
  return !hasPositionLimit(_index);
}

//==============================================================================
void UniversalJoint::setAxis1(const Eigen::Vector3s& _axis)
{
  mAspectProperties.mAxis[0] = _axis;
  Joint::notifyPositionUpdated();
  Joint::incrementVersion();
}

//==============================================================================
void UniversalJoint::setAxis2(const Eigen::Vector3s& _axis)
{
  mAspectProperties.mAxis[1] = _axis;
  Joint::notifyPositionUpdated();
  Joint::incrementVersion();
}

//==============================================================================
const Eigen::Vector3s& UniversalJoint::getAxis1() const
{
  return mAspectProperties.mAxis[0];
}

//==============================================================================
const Eigen::Vector3s& UniversalJoint::getAxis2() const
{
  return mAspectProperties.mAxis[1];
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2> UniversalJoint::getRelativeJacobianStatic(
    const Eigen::Vector2s& _positions) const
{
  Eigen::Matrix<s_t, 6, 2> J;
  J.col(0) = math::AdTAngular(
      Joint::mAspectProperties.mT_ChildBodyToJoint
          * math::expAngular(-getAxis2() * _positions[1]),
      getAxis1());
  J.col(1) = math::AdTAngular(
      Joint::mAspectProperties.mT_ChildBodyToJoint, getAxis2());
  assert(!math::isNan(J));
  return J;
}

//==============================================================================
UniversalJoint::UniversalJoint(const Properties& properties)
  : detail::UniversalJointBase(properties)
{
  // Inherited Aspects must be created in the final joint class in reverse order
  // or else we get pure virtual function calls
  createUniversalJointAspect(properties);
  createGenericJointAspect(properties);
  createJointAspect(properties);
}

//==============================================================================
Joint* UniversalJoint::clone() const
{
  return new UniversalJoint(getUniversalJointProperties());
}

//==============================================================================
void UniversalJoint::updateDegreeOfFreedomNames()
{
  if (!mDofs[0]->isNamePreserved())
    mDofs[0]->setName(Joint::mAspectProperties.mName + "_1", false);
  if (!mDofs[1]->isNamePreserved())
    mDofs[1]->setName(Joint::mAspectProperties.mName + "_2", false);
}

//==============================================================================
void UniversalJoint::updateRelativeTransform() const
{
  const Eigen::Vector2s& positions = getPositionsStatic();
  mT = Joint::mAspectProperties.mT_ParentBodyToJoint
       * Eigen::AngleAxis_s(positions[0], getAxis1())
       * Eigen::AngleAxis_s(positions[1], getAxis2())
       * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
  assert(math::verifyTransform(mT));
}

//==============================================================================
void UniversalJoint::updateRelativeJacobian(bool) const
{
  mJacobian = getRelativeJacobianStatic(getPositionsStatic());
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2> UniversalJoint::getRelativeJacobianTimeDerivStatic(
    const Eigen::Vector2s& _positions, const Eigen::Vector2s& _velocities) const
{
  Eigen::Matrix<s_t, 6, 2> dJ = Eigen::Matrix<s_t, 6, 2>::Zero();

  /*

  /// Original formulation

  Eigen::Vector6s tmpV1
      = getRelativeJacobianStatic(_positions).col(1) * _velocities[1];

  Eigen::Isometry3s tmpT = math::expAngular(-getAxis2() * _positions[1]);

  Eigen::Vector6s tmpV2 = math::AdTAngular(
      Joint::mAspectProperties.mT_ChildBodyToJoint * tmpT, getAxis1());

  J.col(0) = -math::ad(tmpV1, tmpV2);
  */

  /// Easier to differentiate, equivalent formulation

  Eigen::Vector6s V1
      = getRelativeJacobianStatic(_positions).col(1) * _velocities[1];

  Eigen::Matrix3s R = math::expMapRot(-getAxis2() * _positions[1]);

  Eigen::Matrix3s cR
      = Joint::mAspectProperties.mT_ChildBodyToJoint.linear() * R;
  Eigen::Vector3s p
      = Joint::mAspectProperties.mT_ChildBodyToJoint.translation();

  Eigen::Vector6s V2;
  V2.head<3>().noalias() = cR * getAxis1();
  V2.tail<3>().noalias() = p.cross(cR * getAxis1());

  (void)V1;
  (void)V2;
  dJ.col(0) = -math::ad(V1, V2);

  assert(!math::isNan(dJ.col(0)));
  assert(dJ.col(1) == Eigen::Vector6s::Zero());
  return dJ;
}

//==============================================================================
void UniversalJoint::updateRelativeJacobianTimeDeriv() const
{
  mJacobianDeriv = getRelativeJacobianTimeDerivStatic(
      getPositionsStatic(), getVelocitiesStatic());
  /*
  Eigen::Vector6s tmpV1
      = getRelativeJacobianStatic().col(1) * getVelocitiesStatic()[1];

  Eigen::Isometry3s tmpT
      = math::expAngular(-getAxis2() * getPositionsStatic()[1]);

  Eigen::Vector6s tmpV2 = math::AdTAngular(
      Joint::mAspectProperties.mT_ChildBodyToJoint * tmpT, getAxis1());

  mJacobianDeriv.col(0) = -math::ad(tmpV1, tmpV2);

  assert(!math::isNan(mJacobianDeriv.col(0)));
  assert(mJacobianDeriv.col(1) == Eigen::Vector6s::Zero());
  */
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::getRelativeJacobianDerivWrtPositionStatic(size_t index) const
{
  Eigen::Matrix<s_t, 6, 2> J = Eigen::Matrix<s_t, 6, 2>::Zero();

  if (index == 1)
  {
    Eigen::Vector6s tmpV1 = getRelativeJacobianStatic().col(1);

    Eigen::Isometry3s tmpT
        = math::expAngular(-getAxis2() * getPositionsStatic()[1]);

    Eigen::Vector6s tmpV2 = math::AdTAngular(
        Joint::mAspectProperties.mT_ChildBodyToJoint * tmpT, getAxis1());

    J.col(0) = -math::ad(tmpV1, tmpV2);

    assert(!math::isNan(J.col(0)));
    assert(J.col(1) == Eigen::Vector6s::Zero());
  }

  return J;
}

//==============================================================================
math::Jacobian UniversalJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    size_t index) const
{
  Eigen::Matrix<s_t, 6, 2> J = Eigen::Matrix<s_t, 6, 2>::Zero();

  // Original code
  /*
  Eigen::Vector6s V1
      = getRelativeJacobianStatic(_positions).col(1) * _velocities[1];

  Eigen::Matrix3s R = math::expMapRot(-getAxis2() * _positions[1]);

  Eigen::Matrix3s cR
      = Joint::mAspectProperties.mT_ChildBodyToJoint.linear() * R;
  Eigen::Vector3s p
      = Joint::mAspectProperties.mT_ChildBodyToJoint.translation();

  Eigen::Vector6s V2;
  V2.head<3>().noalias() = cR * getAxis1();
  V2.tail<3>().noalias() = p.cross(cR * getAxis1());

  J.col(0) = -math::ad(V1, V2);
  */
  Eigen::Vector2s _positions = getPositionsStatic();
  Eigen::Vector2s _velocities = getVelocitiesStatic();

  Eigen::Vector6s V1
      = getRelativeJacobianStatic(_positions).col(1) * _velocities[1];
  Eigen::Vector6s dV1
      = getRelativeJacobianDerivWrtPosition(index).col(1) * _velocities[1];

  Eigen::Matrix3s R = math::expMapRot(-getAxis2() * _positions[1]);

  Eigen::Matrix3s cR
      = Joint::mAspectProperties.mT_ChildBodyToJoint.linear() * R;
  Eigen::Vector3s p
      = Joint::mAspectProperties.mT_ChildBodyToJoint.translation();

  Eigen::Vector6s V2;
  V2.head<3>().noalias() = cR * getAxis1();
  V2.tail<3>().noalias() = p.cross(cR * getAxis1());

  Eigen::Vector6s dV2;
  if (index == 1)
  {
    dV2.head<3>().noalias()
        = Joint::mAspectProperties.mT_ChildBodyToJoint.linear()
          * (R * getAxis1()).cross(getAxis2());
    dV2.tail<3>().noalias() = p.cross(dV2.head<3>());
  }
  else
  {
    dV2.setZero();
  }

  (void)V1;
  (void)dV1;
  (void)V2;
  (void)dV2;
  J.col(0) = -math::ad(dV1, V2) - math::ad(V1, dV2);

  assert(!math::isNan(J.col(0)));
  assert(J.col(1) == Eigen::Vector6s::Zero());

  return J;
}

//==============================================================================
math::Jacobian UniversalJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    size_t index) const
{
  Eigen::Matrix<s_t, 6, 2> J = Eigen::Matrix<s_t, 6, 2>::Zero();

  if (index == 1)
  {
    Eigen::Vector6s tmpV1
        = getRelativeJacobianStatic(getPositionsStatic()).col(1);

    Eigen::Isometry3s tmpT
        = math::expAngular(-getAxis2() * getPositionsStatic()[1]);

    Eigen::Vector6s tmpV2 = math::AdTAngular(
        Joint::mAspectProperties.mT_ChildBodyToJoint * tmpT, getAxis1());

    J.col(0) = -math::ad(tmpV1, tmpV2);

    assert(!math::isNan(J.col(0)));
    assert(J.col(1) == Eigen::Vector6s::Zero());
  }

  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
    const Eigen::Vector2s& positions,
    const Eigen::Vector2s& velocities,
    bool useRidders) const
{
  Eigen::Matrix<s_t, 6, 2> result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix<s_t, 6, 2>>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix<s_t, 6, 2>& perturbed) {
        perturbed = getRelativeJacobianStatic(positions + eps * velocities);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianDerivWrtPos(
    const Eigen::Vector2s& positions, int index, bool useRidders) const
{
  Eigen::Matrix<s_t, 6, 2> result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix<s_t, 6, 2>>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix<s_t, 6, 2>& perturbed) {
        perturbed = getRelativeJacobianStatic(
            positions + eps * Eigen::Vector2s::Unit(index));
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
    const Eigen::Vector2s& positions,
    const Eigen::Vector2s& velocities,
    int index,
    bool useRidders) const
{
  Eigen::Matrix<s_t, 6, 2> result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix<s_t, 6, 2>>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix<s_t, 6, 2>& perturbed) {
        perturbed = getRelativeJacobianTimeDerivStatic(
            positions + eps * Eigen::Vector2s::Unit(index), velocities);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
    const Eigen::Vector2s& positions,
    const Eigen::Vector2s& velocities,
    int index,
    bool useRidders) const
{
  Eigen::Matrix<s_t, 6, 2> result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix<s_t, 6, 2>>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix<s_t, 6, 2>& perturbed) {
        perturbed = getRelativeJacobianTimeDerivStatic(
            positions, velocities + eps * Eigen::Vector2s::Unit(index));
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
/// Returns the value for q that produces the nearest rotation to
/// `relativeRotation` passed in.
Eigen::VectorXs UniversalJoint::getNearestPositionToDesiredRotation(
    const Eigen::Matrix3s& relativeRotationGlobal)
{
  Eigen::Matrix3s relativeRotation
      = Joint::mAspectProperties.mT_ParentBodyToJoint.linear().transpose()
        * relativeRotationGlobal
        * Joint::mAspectProperties.mT_ChildBodyToJoint.linear();

  // mT = Joint::mAspectProperties.mT_ParentBodyToJoint
  //      * Eigen::AngleAxis_s(positions[0], getAxis1())
  //      * Eigen::AngleAxis_s(positions[1], getAxis2())
  //      * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();

  s_t ang1 = 0.0;
  s_t ang2 = 0.0;
  Eigen::Matrix3s remainingRotation = relativeRotation;

  s_t lastDist = std::numeric_limits<s_t>::infinity();
  for (int i = 0; i < 50; i++)
  {
    Eigen::Matrix3s rot1 = math::expMapRot(ang1 * getAxis1());
    remainingRotation = rot1.inverse() * relativeRotation;
    ang2 = math::getClosestRotationalApproximation(
        getAxis2(), remainingRotation);
    Eigen::Matrix3s rot2 = math::expMapRot(ang2 * getAxis2());
    remainingRotation = relativeRotation * rot2.inverse();
    ang1 = math::getClosestRotationalApproximation(
        getAxis1(), remainingRotation);

    Eigen::Matrix3s R = math::expMapRot(ang1 * getAxis1())
                        * math::expMapRot(ang2 * getAxis2());
    s_t dist = (R - relativeRotation).squaredNorm();
    s_t improvement = lastDist - dist;
    lastDist = dist;

    // #ifndef NDEBUG
    //     std::cout << "Improvement[" << i << "]: " << improvement <<
    //     std::endl;
    // #endif

    if (improvement == 0)
    {
      break;
    }
  }
  return Eigen::Vector2s(ang1, ang2);
}

} // namespace dynamics
} // namespace dart
