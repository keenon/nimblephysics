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

#include <string>

#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"

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
math::Jacobian UniversalJoint::getRelativeJacobianDeriv(size_t index) const
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
  Eigen::Vector6s dV1 = getRelativeJacobianDeriv(index).col(1) * _velocities[1];

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
  if (useRidders)
  {
    return finiteDifferenceRiddersRelativeJacobianTimeDerivStatic(
        positions, velocities);
  }

  const s_t EPS = 1e-7;

  Eigen::Matrix<s_t, 6, 2> plus
      = getRelativeJacobianStatic(positions + EPS * velocities);
  Eigen::Matrix<s_t, 6, 2> minus
      = getRelativeJacobianStatic(positions - EPS * velocities);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRiddersRelativeJacobianTimeDerivStatic(
    const Eigen::Vector2s& positions, const Eigen::Vector2s& velocities) const
{
  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Neville tableau of finite difference results
  std::array<std::array<Eigen::Matrix<s_t, 6, 2>, tabSize>, tabSize> tab;

  Eigen::Matrix<s_t, 6, 2> plus
      = getRelativeJacobianStatic(positions + stepSize * velocities);
  Eigen::Matrix<s_t, 6, 2> minus
      = getRelativeJacobianStatic(positions - stepSize * velocities);

  tab[0][0] = (plus - minus) / (2 * stepSize);
  Eigen::Matrix<s_t, 6, 2> jac = (plus - minus) / (2 * stepSize);

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    Eigen::Matrix<s_t, 6, 2> plus
        = getRelativeJacobianStatic(positions + stepSize * velocities);
    Eigen::Matrix<s_t, 6, 2> minus
        = getRelativeJacobianStatic(positions - stepSize * velocities);

    tab[0][iTab] = (plus - minus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab]
          = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1]) / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]).array().abs().maxCoeff());
      if (currError < bestError)
      {
        bestError = currError;
        jac.noalias() = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return jac;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianDerivWrtPos(
    const Eigen::Vector2s& positions, int index, bool useRidders) const
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRelativeJacobianDerivWrtPos(positions, index);
  }

  const s_t EPS = 1e-7;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianStatic(
      positions + EPS * Eigen::Vector2s::Unit(index));
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianStatic(
      positions - EPS * Eigen::Vector2s::Unit(index));

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRiddersRelativeJacobianDerivWrtPos(
    const Eigen::Vector2s& positions, int index) const
{
  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Neville tableau of finite difference results
  std::array<std::array<Eigen::Matrix<s_t, 6, 2>, tabSize>, tabSize> tab;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianStatic(
      positions + stepSize * Eigen::Vector2s::Unit(index));
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianStatic(
      positions - stepSize * Eigen::Vector2s::Unit(index));

  tab[0][0] = (plus - minus) / (2 * stepSize);
  Eigen::Matrix<s_t, 6, 2> jac = (plus - minus) / (2 * stepSize);

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianStatic(
        positions + stepSize * Eigen::Vector2s::Unit(index));
    Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianStatic(
        positions - stepSize * Eigen::Vector2s::Unit(index));

    tab[0][iTab] = (plus - minus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab]
          = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1]) / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]).array().abs().maxCoeff());
      if (currError < bestError)
      {
        bestError = currError;
        jac.noalias() = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return jac;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
    const Eigen::Vector2s& positions,
    const Eigen::Vector2s& velocities,
    int index,
    bool useRidders) const
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtPosition(
        positions, velocities, index);
  }

  const s_t EPS = 1e-7;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
      positions + EPS * Eigen::Vector2s::Unit(index), velocities);
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
      positions - EPS * Eigen::Vector2s::Unit(index), velocities);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2> UniversalJoint::
    finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtPosition(
        const Eigen::Vector2s& positions,
        const Eigen::Vector2s& velocities,
        int index) const
{
  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Neville tableau of finite difference results
  std::array<std::array<Eigen::Matrix<s_t, 6, 2>, tabSize>, tabSize> tab;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
      positions + stepSize * Eigen::Vector2s::Unit(index), velocities);
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
      positions - stepSize * Eigen::Vector2s::Unit(index), velocities);

  tab[0][0] = (plus - minus) / (2 * stepSize);
  Eigen::Matrix<s_t, 6, 2> jac = (plus - minus) / (2 * stepSize);

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
        positions + stepSize * Eigen::Vector2s::Unit(index), velocities);
    Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
        positions - stepSize * Eigen::Vector2s::Unit(index), velocities);

    tab[0][iTab] = (plus - minus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab]
          = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1]) / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]).array().abs().maxCoeff());
      if (currError < bestError)
      {
        bestError = currError;
        jac.noalias() = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return jac;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2>
UniversalJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
    const Eigen::Vector2s& positions,
    const Eigen::Vector2s& velocities,
    int index,
    bool useRidders) const
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtVelocity(
        positions, velocities, index);
  }

  const s_t EPS = 1e-7;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
      positions, velocities + EPS * Eigen::Vector2s::Unit(index));
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
      positions, velocities - EPS * Eigen::Vector2s::Unit(index));

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix<s_t, 6, 2> UniversalJoint::
    finiteDifferenceRiddersRelativeJacobianTimeDerivDerivWrtVelocity(
        const Eigen::Vector2s& positions,
        const Eigen::Vector2s& velocities,
        int index) const
{
  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  s_t stepSize = originalStepSize;
  s_t bestError = std::numeric_limits<s_t>::max();

  // Neville tableau of finite difference results
  std::array<std::array<Eigen::Matrix<s_t, 6, 2>, tabSize>, tabSize> tab;

  Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
      positions, velocities + stepSize * Eigen::Vector2s::Unit(index));
  Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
      positions, velocities - stepSize * Eigen::Vector2s::Unit(index));

  tab[0][0] = (plus - minus) / (2 * stepSize);
  Eigen::Matrix<s_t, 6, 2> jac = (plus - minus) / (2 * stepSize);

  // Iterate over smaller and smaller step sizes
  for (int iTab = 1; iTab < tabSize; iTab++)
  {
    stepSize /= con;

    Eigen::Matrix<s_t, 6, 2> plus = getRelativeJacobianTimeDerivStatic(
        positions, velocities + stepSize * Eigen::Vector2s::Unit(index));
    Eigen::Matrix<s_t, 6, 2> minus = getRelativeJacobianTimeDerivStatic(
        positions, velocities - stepSize * Eigen::Vector2s::Unit(index));

    tab[0][iTab] = (plus - minus) / (2 * stepSize);

    s_t fac = con2;
    // Compute extrapolations of increasing orders, requiring no new
    // evaluations
    for (int jTab = 1; jTab <= iTab; jTab++)
    {
      tab[jTab][iTab]
          = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1]) / (fac - 1.0);
      fac = con2 * fac;
      s_t currError = max(
          (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
          (tab[jTab][iTab] - tab[jTab - 1][iTab - 1]).array().abs().maxCoeff());
      if (currError < bestError)
      {
        bestError = currError;
        jac.noalias() = tab[jTab][iTab];
      }
    }

    // If higher order is worse by a significant factor, quit early.
    if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
        >= safeThreshold * bestError)
    {
      break;
    }
  }

  return jac;
}

} // namespace dynamics
} // namespace dart
