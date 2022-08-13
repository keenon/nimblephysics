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

#include "dart/dynamics/EulerJoint.hpp"

#include <string>

#include "dart/common/Console.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
EulerJoint::~EulerJoint()
{
  // Do nothing
}

//==============================================================================
void EulerJoint::setProperties(const Properties& _properties)
{
  Base::setProperties(static_cast<const Base::Properties&>(_properties));
  setProperties(static_cast<const UniqueProperties&>(_properties));
}

//==============================================================================
void EulerJoint::setProperties(const UniqueProperties& _properties)
{
  setAspectProperties(_properties);
}

//==============================================================================
void EulerJoint::setAspectProperties(const AspectProperties& properties)
{
  setAxisOrder(properties.mAxisOrder, false);
}

//==============================================================================
EulerJoint::Properties EulerJoint::getEulerJointProperties() const
{
  return EulerJoint::Properties(
      getGenericJointProperties(), getEulerJointAspect()->getProperties());
}

//==============================================================================
void EulerJoint::copy(const EulerJoint& _otherJoint)
{
  if (this == &_otherJoint)
    return;

  setProperties(_otherJoint.getEulerJointProperties());
}

//==============================================================================
void EulerJoint::copy(const EulerJoint* _otherJoint)
{
  if (nullptr == _otherJoint)
    return;

  copy(*_otherJoint);
}

//==============================================================================
EulerJoint& EulerJoint::operator=(const EulerJoint& _otherJoint)
{
  copy(_otherJoint);
  return *this;
}

//==============================================================================
const std::string& EulerJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& EulerJoint::getStaticType()
{
  static const std::string name = "EulerJoint";
  return name;
}

//==============================================================================
bool EulerJoint::isCyclic(std::size_t _index) const
{
  return !hasPositionLimit(_index);
}

//==============================================================================
void EulerJoint::setAxisOrder(EulerJoint::AxisOrder _order, bool _renameDofs)
{
  mAspectProperties.mAxisOrder = _order;
  if (_renameDofs)
    updateDegreeOfFreedomNames();

  Joint::notifyPositionUpdated();
  updateRelativeJacobian(true);
  Joint::incrementVersion();
}

//==============================================================================
EulerJoint::AxisOrder EulerJoint::getAxisOrder() const
{
  return mAspectProperties.mAxisOrder;
}

//==============================================================================
/// Returns the axis of rotation for DOF `index`, depending on the AxisOrder
Eigen::Vector3s EulerJoint::getAxis(int index) const
{
  if (getAxisOrder() == AxisOrder::XYZ)
  {
    if (index == 0)
    {
      return Eigen::Vector3s::UnitX() * getFlipAxisMap()(0);
    }
    if (index == 1)
    {
      return Eigen::Vector3s::UnitY() * getFlipAxisMap()(1);
    }
    if (index == 2)
    {
      return Eigen::Vector3s::UnitZ() * getFlipAxisMap()(2);
    }
  }
  if (getAxisOrder() == AxisOrder::XZY)
  {
    if (index == 0)
    {
      return Eigen::Vector3s::UnitX() * getFlipAxisMap()(0);
    }
    if (index == 1)
    {
      return Eigen::Vector3s::UnitZ() * getFlipAxisMap()(1);
    }
    if (index == 2)
    {
      return Eigen::Vector3s::UnitY() * getFlipAxisMap()(2);
    }
  }
  if (getAxisOrder() == AxisOrder::ZXY)
  {
    if (index == 0)
    {
      return Eigen::Vector3s::UnitZ() * getFlipAxisMap()(0);
    }
    if (index == 1)
    {
      return Eigen::Vector3s::UnitX() * getFlipAxisMap()(1);
    }
    if (index == 2)
    {
      return Eigen::Vector3s::UnitY() * getFlipAxisMap()(2);
    }
  }
  if (getAxisOrder() == AxisOrder::ZYX)
  {
    if (index == 0)
    {
      return Eigen::Vector3s::UnitZ() * getFlipAxisMap()(0);
    }
    if (index == 1)
    {
      return Eigen::Vector3s::UnitY() * getFlipAxisMap()(1);
    }
    if (index == 2)
    {
      return Eigen::Vector3s::UnitX() * getFlipAxisMap()(2);
    }
  }
  std::cout
      << "ERROR: EulerJoint is being asked for an axis that is out of bounds!"
      << std::endl;
  assert(false);
  return Eigen::Vector3s::UnitX();
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void EulerJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s EulerJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
Eigen::Isometry3s EulerJoint::convertToTransform(
    const Eigen::Vector3s& _positions,
    AxisOrder _ordering,
    Eigen::Vector3s _flipAxisMap)
{
  return Eigen::Isometry3s(
      convertToRotation(_positions, _ordering, _flipAxisMap));
}

//==============================================================================
Eigen::Isometry3s EulerJoint::convertToTransform(
    const Eigen::Vector3s& _positions) const
{
  return convertToTransform(_positions, getAxisOrder(), getFlipAxisMap());
}

//==============================================================================
Eigen::Matrix3s EulerJoint::convertToRotation(
    const Eigen::Vector3s& _positions,
    AxisOrder _ordering,
    const Eigen::Vector3s& _flipAxisMap)
{
  switch (_ordering)
  {
    case AxisOrder::XYZ:
      return math::eulerXYZToMatrix(_positions.cwiseProduct(_flipAxisMap));
    case AxisOrder::ZYX:
      return math::eulerZYXToMatrix(_positions.cwiseProduct(_flipAxisMap));
    case AxisOrder::ZXY:
      return math::eulerZXYToMatrix(_positions.cwiseProduct(_flipAxisMap));
    case AxisOrder::XZY:
      return math::eulerXZYToMatrix(_positions.cwiseProduct(_flipAxisMap));
    default: {
      dterr << "[EulerJoint::convertToRotation] Invalid AxisOrder specified ("
            << static_cast<int>(_ordering) << ")\n";
      return Eigen::Matrix3s::Identity();
    }
  }
}

//==============================================================================
Eigen::Matrix3s EulerJoint::convertToRotation(
    const Eigen::Vector3s& _positions) const
{
  return convertToRotation(_positions, getAxisOrder(), getFlipAxisMap());
}

//==============================================================================
/// This is a truly static method to compute the relative Jacobian, which gets
/// reused in CustomJoint
Eigen::Matrix<s_t, 6, 3> EulerJoint::computeRelativeJacobianStatic(
    const Eigen::Vector3s& _positions,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  (void)flipAxisMap;
  Eigen::Matrix<s_t, 6, 3> J;

  // s_t q0 = _positions[0];
  const s_t q1 = _positions[1] * flipAxisMap(1);
  const s_t q2 = _positions[2] * flipAxisMap(2);

  // s_t c0 = cos(q0);
  s_t c1 = cos(q1);
  s_t c2 = cos(q2);

  // s_t s0 = sin(q0);
  s_t s1 = sin(q1);
  s_t s2 = sin(q2);

  Eigen::Vector6s J0 = Eigen::Vector6s::Zero();
  Eigen::Vector6s J1 = Eigen::Vector6s::Zero();
  Eigen::Vector6s J2 = Eigen::Vector6s::Zero();

  switch (axisOrder)
  {
    case EulerJoint::AxisOrder::XYZ: {
      //------------------------------------------------------------------------
      // S = [    c1*c2, s2,  0
      //       -(c1*s2), c2,  0
      //             s1,  0,  1
      //              0,  0,  0
      //              0,  0,  0
      //              0,  0,  0 ];
      //------------------------------------------------------------------------
      J0 << c1 * c2, -(c1 * s2), s1, 0.0, 0.0, 0.0;
      J1 << s2, c2, 0.0, 0.0, 0.0, 0.0;
      J2 << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (abs(_positions[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in XYZ-euler joint. ("
                  << _positions[0] << ", " << _positions[1] << ", "
                  << _positions[2] << ")" << std::endl;
#endif

      break;
    }
    case EulerJoint::AxisOrder::ZYX: {
      //------------------------------------------------------------------------
      // S = [   -s1,    0,   1
      //       s2*c1,   c2,   0
      //       c1*c2,  -s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      J0 << -s1, s2 * c1, c1 * c2, 0.0, 0.0, 0.0;
      J1 << 0.0, c2, -s2, 0.0, 0.0, 0.0;
      J2 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (abs(_positions[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in ZYX-euler joint. ("
                  << _positions[0] << ", " << _positions[1] << ", "
                  << _positions[2] << ")" << std::endl;
#endif

      break;
    }

    case EulerJoint::AxisOrder::XZY: {
      //------------------------------------------------------------------------
      // S = [ c1*c2,  -s2,   0
      //         -s1,    0,   1
      //       c1*s2,   c2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      J0 << c1 * c2, -s1, c1 * s2, 0.0, 0.0, 0.0;
      J1 << -s2, 0, c2, 0.0, 0.0, 0.0;
      J2 << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (abs(_positions[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in XZY-euler joint. ("
                  << _positions[0] << ", " << _positions[1] << ", "
                  << _positions[2] << ")" << std::endl;
#endif

      break;
    }

    case EulerJoint::AxisOrder::ZXY: {
      //------------------------------------------------------------------------
      // S = [-c1*s2,   c2,   0
      //          s1,    0,   1
      //       c1*c2,   s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      J0 << -c1 * s2, s1, c1 * c2, 0.0, 0.0, 0.0;
      J1 << c2, 0, s2, 0.0, 0.0, 0.0;
      J2 << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (abs(_positions[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in ZXY-euler joint. ("
                  << _positions[0] << ", " << _positions[1] << ", "
                  << _positions[2] << ")" << std::endl;
#endif

      break;
    }
    default: {
      dterr << "Undefined Euler axis order in computeRelativeJacobianStatic\n";
      break;
    }
  }

  J.col(0) = math::AdT(childBodyToJoint, J0) * flipAxisMap(0);
  J.col(1) = math::AdT(childBodyToJoint, J1) * flipAxisMap(1);
  J.col(2) = math::AdT(childBodyToJoint, J2) * flipAxisMap(2);

  assert(!math::isNan(J));

#ifndef NDEBUG
  Eigen::MatrixXs JTJ = J.transpose() * J;
  Eigen::FullPivLU<Eigen::MatrixXs> luJTJ(JTJ);
  //    Eigen::FullPivLU<Eigen::MatrixXs> luS(mS);
  s_t det = luJTJ.determinant();
  if (det < 1e-8)
  {
    std::cout << "ill-conditioned Jacobian in joint."
              << " The determinant of the Jacobian is (" << det << ")."
              << std::endl;
    std::cout << "rank is (" << luJTJ.rank() << ")." << std::endl;
    std::cout << "det is (" << luJTJ.determinant() << ")." << std::endl;
    std::cout << "J is:" << std::endl << J << std::endl;
    std::cout << "pos is:" << std::endl << _positions << std::endl;
    //        std::cout << "mS: \n" << mS << std::endl;
  }
#endif

  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3> EulerJoint::getRelativeJacobianStatic(
    const Eigen::Vector3s& _positions) const
{
  return computeRelativeJacobianStatic(
      _positions,
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
math::Jacobian EulerJoint::computeRelativeJacobianDerivWrtPos(
    std::size_t index,
    const Eigen::Vector3s& positions,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  (void)flipAxisMap;
  assert(index < 3);

  Eigen::Matrix<s_t, 6, 3> DJ_Dq = Eigen::Matrix<s_t, 6, 3>::Zero();

  const Eigen::Vector3s& q = positions;

  const s_t q1 = q[1] * flipAxisMap(1);
  const s_t q2 = q[2] * flipAxisMap(2);

  // s_t c0 = cos(q0);
  const s_t c1 = cos(q1);
  const s_t c2 = cos(q2);

  // s_t s0 = sin(q0);
  const s_t s1 = sin(q1);
  const s_t s2 = sin(q2);

  switch (axisOrder)
  {
    case EulerJoint::AxisOrder::XYZ: {
      //------------------------------------------------------------------------
      // S = [    c1*c2, s2,  0
      //       -(c1*s2), c2,  0
      //             s1,  0,  1
      //              0,  0,  0
      //              0,  0,  0
      //              0,  0,  0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DS/Dq1 = [ -s1*c2,  0,  0
        //             s1*s2,  0,  0
        //                c1,  0,  0
        //                 0,  0,  0
        //                 0,  0,  0
        //                 0,  0,  0 ];

        DJ_Dq(0, 0) = -s1 * c2;
        DJ_Dq(1, 0) = s1 * s2;
        DJ_Dq(2, 0) = c1;

        DJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DS/Dq2 = [ -c1*s2,  c2,  0
        //            -c1*c2, -s2,  0
        //                 0,   0,  0
        //                 0,   0,  0
        //                 0,   0,  0
        //                 0,   0,  0 ];

        DJ_Dq(0, 0) = -c1 * s2;
        DJ_Dq(1, 0) = -c1 * c2;

        DJ_Dq(0, 1) = c2;
        DJ_Dq(1, 1) = -s2;

        DJ_Dq *= flipAxisMap(2);
      }
      break;
    }
    case EulerJoint::AxisOrder::ZYX: {
      //------------------------------------------------------------------------
      // S = [   -s1,    0,   1
      //       s2*c1,   c2,   0
      //       c1*c2,  -s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DS/Dq1 = [   -c1,  0,   0
        //           -s1*s2,  0,   0
        //           -s1*c2,  0,   0
        //                0,  0,   0
        //                0,  0,   0
        //                0,  0,   0 ];

        DJ_Dq(0, 0) = -c1;
        DJ_Dq(1, 0) = -s1 * s2;
        DJ_Dq(2, 0) = -s1 * c2;

        DJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DS/Dq2 = [     0,    0,   0
        //            c1*c2,  -s2,   0
        //           -c1*s2,  -c2,   0
        //                0,    0,   0
        //                0,    0,   0
        //                0,    0,   0 ];

        DJ_Dq(1, 0) = c1 * c2;
        DJ_Dq(2, 0) = -c1 * s2;

        DJ_Dq(1, 1) = -s2;
        DJ_Dq(2, 1) = -c2;

        DJ_Dq *= flipAxisMap(2);
      }

      break;
    }
    case EulerJoint::AxisOrder::XZY: {
      //------------------------------------------------------------------------
      // S = [ c1*c2,  -s2,   0
      //         -s1,    0,   1
      //       c1*s2,   c2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DS/Dq1 = [-s1*c2,  0,   0
        //              -c1,  0,   0
        //           -s1*s2,  0,   0
        //                0,  0,   0
        //                0,  0,   0
        //                0,  0,   0 ];

        DJ_Dq(0, 0) = -s1 * c2;
        DJ_Dq(1, 0) = -c1;
        DJ_Dq(2, 0) = -s1 * s2;

        DJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DS/Dq2 = [-c1*s2,  -c2,   0
        //                0,    0,   0
        //            c1*c2,  -s2,   0
        //                0,    0,   0
        //                0,    0,   0
        //                0,    0,   0 ];

        DJ_Dq(0, 0) = -c1 * s2;
        DJ_Dq(2, 0) = c1 * c2;

        DJ_Dq(0, 1) = -c2;
        DJ_Dq(2, 1) = -s2;

        DJ_Dq *= flipAxisMap(2);
      }

      break;
    }

    case EulerJoint::AxisOrder::ZXY: {
      //------------------------------------------------------------------------
      // S = [-c1*s2,   c2,   0
      //          s1,    0,   1
      //       c1*c2,   s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DS/Dq1 = [ s1*s2,  0,   0
        //               c1,  0,   0
        //           -s1*c2,  0,   0
        //                0,  0,   0
        //                0,  0,   0
        //                0,  0,   0 ];

        DJ_Dq(0, 0) = s1 * s2;
        DJ_Dq(1, 0) = c1;
        DJ_Dq(2, 0) = -s1 * c2;

        DJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DS/Dq2 = [-c1*c2,  -s2,   0
        //                0,    0,   0
        //           -c1*s2,   c2,   0
        //                0,    0,   0
        //                0,    0,   0
        //                0,    0,   0 ];

        DJ_Dq(0, 0) = -c1 * c2;
        DJ_Dq(2, 0) = -c1 * s2;

        DJ_Dq(0, 1) = -s2;
        DJ_Dq(2, 1) = c2;

        DJ_Dq *= flipAxisMap(2);
      }

      break;
    }
    default: {
      dterr << "Undefined Euler axis order in "
               "computeRelativeJacobianDerivWrtPos\n";
      break;
    }
  }

  DJ_Dq = math::AdTJac(childBodyToJoint, DJ_Dq);
  DJ_Dq.col(0) *= flipAxisMap(0);
  DJ_Dq.col(1) *= flipAxisMap(1);
  DJ_Dq.col(2) *= flipAxisMap(2);

  assert(!math::isNan(DJ_Dq));

  return DJ_Dq;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EulerJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
    const Eigen::Vector3s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPlus
      = positions + (EPS * Eigen::Vector3s::Unit(index));
  Eigen::Vector3s perturbedMinus
      = positions - (EPS * Eigen::Vector3s::Unit(index));

  Eigen::Matrix<s_t, 6, 3> plus = computeRelativeJacobianStatic(
      perturbedPlus, axisOrder, flipAxisMap, childBodyToJoint);
  Eigen::Matrix<s_t, 6, 3> minus = computeRelativeJacobianStatic(
      perturbedMinus, axisOrder, flipAxisMap, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3> EulerJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  return computeRelativeJacobianDerivWrtPos(
      index,
      getPositionsStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
math::Jacobian EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
    std::size_t index,
    const Eigen::Vector3s& positions,
    const Eigen::Vector3s& velocities,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  (void)flipAxisMap;
  assert(index < 3);

  Eigen::Matrix<s_t, 6, 3> DdJ_Dq = Eigen::Matrix<s_t, 6, 3>::Zero();

  const Eigen::Vector3s& q = positions;
  const s_t q1 = q[1] * flipAxisMap(1);
  const s_t q2 = q[2] * flipAxisMap(2);

  // s_t dq0 = mVelocities[0];
  const Eigen::Vector3s& dq = velocities;
  const s_t dq1 = dq[1] * flipAxisMap(1);
  const s_t dq2 = dq[2] * flipAxisMap(2);

  const s_t c1 = cos(q1);
  const s_t c2 = cos(q2);

  const s_t s1 = sin(q1);
  const s_t s2 = sin(q2);

  switch (axisOrder)
  {
    case EulerJoint::AxisOrder::XYZ: {
      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DdS/Dq1 = [ -c1*c2*dq1 + s1*s2*dq2, 0, 0
        //              c1*s2*dq1 + s1*c2*dq2, 0, 0
        //                            -s1*dq1, 0, 0
        //                                  0, 0, 0
        //                                  0, 0, 0
        //                                  0, 0, 0 ];

        DdJ_Dq(0, 0) = -c1 * c2 * dq1 + s1 * s2 * dq2;
        DdJ_Dq(1, 0) = c1 * s2 * dq1 + s1 * c2 * dq2;
        DdJ_Dq(2, 0) = -s1 * dq1;

        DdJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DdS/Dq2 = [ s1*s2*dq1 - c1*c2*dq2, -s2*dq2, 0
        //             s1*c2*dq1 + c1*s2*dq2, -c2*dq2, 0
        //                                 0,       0, 0
        //                                 0,       0, 0
        //                                 0,       0, 0
        //                                 0,       0, 0 ];

        DdJ_Dq(0, 0) = s1 * s2 * dq1 - c1 * c2 * dq2;
        DdJ_Dq(1, 0) = s1 * c2 * dq1 + c1 * s2 * dq2;

        DdJ_Dq(0, 1) = -s2 * dq2;
        DdJ_Dq(1, 1) = -c2 * dq2;

        DdJ_Dq *= flipAxisMap(2);
      }
      break;
    }
    case EulerJoint::AxisOrder::ZYX: {
      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        // DdS/Dq1 = [                 s1*dq1, 0, 0
        //             -c1*s2*dq1 - s1*c2*dq2, 0, 0
        //             -c1*c2*dq1 + s1*s2*dq2, 0, 0
        //                                  0, 0, 0
        //                                  0, 0, 0
        //                                  0, 0, 0 ];

        DdJ_Dq(0, 0) = s1 * dq1;
        DdJ_Dq(1, 0) = -c1 * s2 * dq1 - s1 * c2 * dq2;
        DdJ_Dq(2, 0) = -c1 * c2 * dq1 + s1 * s2 * dq2;

        DdJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DdS/Dq1 = [                      0,       0, 0
        //             -s1*c2*dq1 - c1*s2*dq2, -c2*dq2, 0
        //              s1*s2*dq1 - c1*c2*dq2,  s2*dq2, 0
        //                                  0,       0, 0
        //                                  0,       0, 0
        //                                  0,       0, 0 ];

        DdJ_Dq(1, 0) = -s1 * c2 * dq1 - c1 * s2 * dq2;
        DdJ_Dq(2, 0) = s1 * s2 * dq1 - c1 * c2 * dq2;

        DdJ_Dq(1, 1) = -c2 * dq2;
        DdJ_Dq(2, 1) = s2 * dq2;

        DdJ_Dq *= flipAxisMap(2);
      }
      break;
    }
    case EulerJoint::AxisOrder::XZY: {
      //------------------------------------------------------------------------
      // dS = [ -(c2 * s1*dq1) -(c1 * s2*dq2),   -c2*dq2,   0
      //                              -c1*dq1,    0,        0
      //        (c1 * c2*dq2) - (s2 * s1*dq1),   -s2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        //------------------------------------------------------------------------
        // dS = [ -(c2 * c1*dq1) +(s1 * s2*dq2),    0,   0
        //                               s1*dq1,    0,   0
        //        -(s1 * c2*dq2) -(s2 * c1*dq1),    0,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Dq(0, 0) = -(c2 * c1 * dq1) + (s1 * s2 * dq2);
        DdJ_Dq(1, 0) = s1 * dq1;
        DdJ_Dq(2, 0) = -(s1 * c2 * dq2) - (s2 * c1 * dq1);

        DdJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        //------------------------------------------------------------------------
        // dS = [ (s2 * s1*dq1) -(c1 * c2*dq2),   s2*dq2,   0
        //                                   0,        0,   0
        //        (c1 * -s2*dq2) - (c2 * s1*dq1), -c2*dq2,  0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Dq(0, 0) = (s2 * s1 * dq1) - (c1 * c2 * dq2);
        DdJ_Dq(0, 1) = s2 * dq2;
        DdJ_Dq(2, 0) = -(c1 * s2 * dq2) - (c2 * s1 * dq1);
        DdJ_Dq(2, 1) = -c2 * dq2;

        DdJ_Dq *= flipAxisMap(2);
      }
      break;
    }

    case EulerJoint::AxisOrder::ZXY: {
      //------------------------------------------------------------------------
      // dS = [(s1*s2*dq1) - (c1*c2*dq2),  -s2*dq2,   0
      //                          c1*dq1,        0,   0
      //      -(s1*c2*dq1) - (c1*s2*dq2),   c2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        //------------------------------------------------------------------------
        // dS = [(c1*s2*dq1) + (s1*c2*dq2),        0,   0
        //                         -s1*dq1,        0,   0
        //      -(c1*c2*dq1) + (s1*s2*dq2),        0,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Dq(0, 0) = (c1 * s2 * dq1) + (s1 * c2 * dq2);
        DdJ_Dq(1, 0) = -s1 * dq1;
        DdJ_Dq(2, 0) = -(c1 * c2 * dq1) + (s2 * s1 * dq2);

        DdJ_Dq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        //------------------------------------------------------------------------
        // dS = [(s1*c2*dq1) + (c1*s2*dq2),  -c2*dq2,   0
        //                               0,        0,   0
        //       (s1*s2*dq1) - (c1*c2*dq2),  -s2*dq2,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Dq(0, 0) = (s1 * c2 * dq1) + (c1 * s2 * dq2);
        DdJ_Dq(0, 1) = -c2 * dq2;
        DdJ_Dq(2, 0) = (s1 * s2 * dq1) - (c1 * c2 * dq2);
        DdJ_Dq(2, 1) = -s2 * dq2;

        DdJ_Dq *= flipAxisMap(2);
      }
      break;
    }
    default: {
      dterr << "Undefined Euler axis order in "
               "computeRelativeJacobianTimeDerivDerivWrtPos\n";
      break;
    }
  }

  DdJ_Dq = math::AdTJac(childBodyToJoint, DdJ_Dq);
  DdJ_Dq.col(0) *= flipAxisMap(0);
  DdJ_Dq.col(1) *= flipAxisMap(1);
  DdJ_Dq.col(2) *= flipAxisMap(2);

  assert(!math::isNan(DdJ_Dq));

  return DdJ_Dq;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
    const Eigen::Vector3s& positions,
    const Eigen::Vector3s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-8;
  Eigen::Vector3s perturbedPlus
      = positions + (EPS * Eigen::Vector3s::Unit(index));
  Eigen::Vector3s perturbedMinus
      = positions - (EPS * Eigen::Vector3s::Unit(index));

  Eigen::Matrix<s_t, 6, 3> plus = computeRelativeJacobianTimeDerivStatic(
      perturbedPlus, velocities, axisOrder, flipAxisMap, childBodyToJoint);
  Eigen::Matrix<s_t, 6, 3> minus = computeRelativeJacobianTimeDerivStatic(
      perturbedMinus, velocities, axisOrder, flipAxisMap, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
math::Jacobian EulerJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  assert(index < 3);
  return computeRelativeJacobianTimeDerivDerivWrtPos(
      index,
      getPositionsStatic(),
      getVelocitiesStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
math::Jacobian EulerJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
    std::size_t index,
    const Eigen::Vector3s& positions,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  (void)flipAxisMap;
  assert(index < 3);

  Eigen::Matrix<s_t, 6, 3> DdJ_Ddq = Eigen::Matrix<s_t, 6, 3>::Zero();

  const Eigen::Vector3s& q = positions;
  const s_t q1 = q[1] * flipAxisMap(1);
  const s_t q2 = q[2] * flipAxisMap(2);

  const s_t c1 = cos(q1);
  const s_t c2 = cos(q2);

  const s_t s1 = sin(q1);
  const s_t s2 = sin(q2);

  switch (axisOrder)
  {
    case EulerJoint::AxisOrder::XYZ: {
      //------------------------------------------------------------------------
      // dS = [  -(dq1*c2*s1) - dq2*c1*s2,    dq2*c2,  0
      //         -(dq2*c1*c2) + dq1*s1*s2, -(dq2*s2),  0
      //                           dq1*c1,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DdS/Ddq0 = 0;
      }
      else if (index == 1)
      {
        // DdS/Ddq1 = [ -s1*c2, 0, 0
        //               s1*s2, 0, 0
        //                  c1, 0, 0
        //                   0, 0, 0
        //                   0, 0, 0
        //                   0, 0, 0 ];

        DdJ_Ddq(0, 0) = -s1 * c2;
        DdJ_Ddq(1, 0) = s1 * s2;
        DdJ_Ddq(2, 0) = c1;

        DdJ_Ddq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DdS/Ddq2 = [ -c1*s2,  c2, 0
        //              -c1*c2, -s2, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0 ];

        DdJ_Ddq(0, 0) = -c1 * s2;
        DdJ_Ddq(1, 0) = -c1 * c2;

        DdJ_Ddq(0, 1) = c2;
        DdJ_Ddq(1, 1) = -s2;

        DdJ_Ddq *= flipAxisMap(2);
      }
      break;
    }
    case EulerJoint::AxisOrder::ZYX: {
      //------------------------------------------------------------------------
      // dS = [               -c1*dq1,        0,   0
      //          c2*c1*dq2-s2*s1*dq1,  -s2*dq2,   0
      //         -s1*c2*dq1-c1*s2*dq2,  -c2*dq2,   0
      //                            0,        0,   0
      //                            0,        0,   0
      //                            0,        0,   0 ];
      //------------------------------------------------------------------------
      if (index == 0)
      {
        // DdS/Ddq0 = 0;
      }
      else if (index == 1)
      {
        // DdS/Ddq1 = [    -c1, 0, 0
        //              -s1*s2, 0, 0
        //              -s1*c2, 0, 0
        //                   0, 0, 0
        //                   0, 0, 0
        //                   0, 0, 0 ];

        DdJ_Ddq(0, 0) = -c1;
        DdJ_Ddq(1, 0) = -s1 * s2;
        DdJ_Ddq(2, 0) = -s1 * c2;

        DdJ_Ddq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        // DdS/Ddq1 = [      0,   0, 0
        //               c1*c2, -s2, 0
        //              -c1*s2, -c2, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0 ];

        DdJ_Ddq(1, 0) = c1 * c2;
        DdJ_Ddq(2, 0) = -c1 * s2;

        DdJ_Ddq(1, 1) = -s2;
        DdJ_Ddq(2, 1) = -c2;

        DdJ_Ddq *= flipAxisMap(2);
      }
      break;
    }
    case EulerJoint::AxisOrder::XZY: {
      //------------------------------------------------------------------------
      // dS = [ -(c2 * s1*dq1) -(c1 * s2*dq2),   -c2*dq2,   0
      //                              -c1*dq1,    0,        0
      //        (c1 * c2*dq2) - (s2 * s1*dq1),   -s2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        //------------------------------------------------------------------------
        // dS = [ -(c2 * s1),   0,   0
        //               -c1,   0,   0
        //        -(s2 * s1),   0,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Ddq(0, 0) = -(c2 * s1);
        DdJ_Ddq(1, 0) = -c1;
        DdJ_Ddq(2, 0) = -(s2 * s1);

        DdJ_Ddq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        //------------------------------------------------------------------------
        // dS = [ -(c1 * s2), -c2,   0
        //                                   0,        0,   0
        //        (c1 * c2),  -s2,  0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Ddq(0, 0) = -(c1 * s2);
        DdJ_Ddq(0, 1) = -c2;
        DdJ_Ddq(2, 0) = (c1 * c2);
        DdJ_Ddq(2, 1) = -s2;

        DdJ_Ddq *= flipAxisMap(2);
      }
      break;
    }

    case EulerJoint::AxisOrder::ZXY: {
      //------------------------------------------------------------------------
      // dS = [(s1*s2*dq1) - (c1*c2*dq2),  -s2*dq2,   0
      //                          c1*dq1,        0,   0
      //      -(s1*c2*dq1) - (c1*s2*dq2),   c2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------

      if (index == 0)
      {
        // DdS/Dq0 = 0;
      }
      else if (index == 1)
      {
        //------------------------------------------------------------------------
        // dS = [(s1*s2),  0,   0
        //            c1,  0,   0
        //      -(s1*c2),  0,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Ddq(0, 0) = (s1 * s2);
        DdJ_Ddq(1, 0) = c1;
        DdJ_Ddq(2, 0) = -(s1 * c2);

        DdJ_Ddq *= flipAxisMap(1);
      }
      else if (index == 2)
      {
        //------------------------------------------------------------------------
        // dS = [- (c1*c2),  -s2,   0
        //               0,    0,   0
        //       - (c1*s2),   c2,   0
        //           0,    0,   0
        //           0,    0,   0
        //           0,    0,   0 ];
        //------------------------------------------------------------------------
        DdJ_Ddq(0, 0) = -(c1 * c2);
        DdJ_Ddq(0, 1) = -s2;
        DdJ_Ddq(2, 0) = -(c1 * s2);
        DdJ_Ddq(2, 1) = c2;

        DdJ_Ddq *= flipAxisMap(2);
      }
      break;
    }
    default: {
      dterr << "Undefined Euler axis order in "
               "computeRelativeJacobianTimeDerivDerivWrtVel\n";
      break;
    }
  }

  DdJ_Ddq = math::AdTJac(childBodyToJoint, DdJ_Ddq);
  DdJ_Ddq.col(0) *= flipAxisMap(0);
  DdJ_Ddq.col(1) *= flipAxisMap(1);
  DdJ_Ddq.col(2) *= flipAxisMap(2);

  assert(!math::isNan(DdJ_Ddq));

  return DdJ_Ddq;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EulerJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
    const Eigen::Vector3s& positions,
    const Eigen::Vector3s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-8;
  Eigen::Vector3s perturbedPlus
      = velocities + (EPS * Eigen::Vector3s::Unit(index));
  Eigen::Vector3s perturbedMinus
      = velocities - (EPS * Eigen::Vector3s::Unit(index));

  Eigen::Matrix<s_t, 6, 3> plus = computeRelativeJacobianTimeDerivStatic(
      positions, perturbedPlus, axisOrder, flipAxisMap, childBodyToJoint);
  Eigen::Matrix<s_t, 6, 3> minus = computeRelativeJacobianTimeDerivStatic(
      positions, perturbedMinus, axisOrder, flipAxisMap, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
math::Jacobian EulerJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return computeRelativeJacobianTimeDerivDerivWrtVel(
      index,
      getPositionsStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
EulerJoint::EulerJoint(const Properties& properties)
  : detail::EulerJointBase(properties), mFlipAxisMap(Eigen::Vector3s::Ones())
{
  // Inherited Aspects must be created in the final joint class in reverse
  // order or else we get pure virtual function calls
  createEulerJointAspect(properties);
  createGenericJointAspect(properties);
  createJointAspect(properties);
}

//==============================================================================
Joint* EulerJoint::clone() const
{
  EulerJoint* joint = new EulerJoint(getEulerJointProperties());
  joint->setFlipAxisMap(mFlipAxisMap);
  return joint;
}

//==============================================================================
void EulerJoint::updateDegreeOfFreedomNames()
{
  std::vector<std::string> affixes;
  switch (getAxisOrder())
  {
    case AxisOrder::ZYX:
      affixes.push_back("_z");
      affixes.push_back("_y");
      affixes.push_back("_x");
      break;
    case AxisOrder::XYZ:
      affixes.push_back("_x");
      affixes.push_back("_y");
      affixes.push_back("_z");
      break;
    case AxisOrder::XZY:
      affixes.push_back("_x");
      affixes.push_back("_z");
      affixes.push_back("_y");
      break;
    case AxisOrder::ZXY:
      affixes.push_back("_z");
      affixes.push_back("_x");
      affixes.push_back("_y");
      break;
    default:
      dterr << "Unsupported axis order in EulerJoint named '"
            << Joint::mAspectProperties.mName << "' ("
            << static_cast<int>(getAxisOrder()) << ")\n";
  }

  if (affixes.size() == 3)
  {
    for (std::size_t i = 0; i < 3; ++i)
    {
      if (!mDofs[i]->isNamePreserved())
        mDofs[i]->setName(Joint::mAspectProperties.mName + affixes[i], false);
    }
  }
}

//==============================================================================
void EulerJoint::updateRelativeTransform() const
{
  Eigen::Vector3s pos = getPositionsStatic();
  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * convertToTransform(pos)
       * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();

  assert(math::verifyTransform(mT));
}

//==============================================================================
void EulerJoint::updateRelativeJacobian(bool) const
{
  mJacobian = getRelativeJacobianStatic(getPositionsStatic());
}

//==============================================================================
/// This is a truly static method to compute the relative Jacobian, which gets
/// reused in CustomJoint
Eigen::Matrix<s_t, 6, 3> EulerJoint::computeRelativeJacobianTimeDerivStatic(
    const Eigen::Vector3s& positions,
    const Eigen::Vector3s& velocities,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  (void)flipAxisMap;
  Eigen::Matrix<s_t, 6, 3> dJ;

  // s_t q0 = mPositions[0];
  s_t q1 = positions[1] * flipAxisMap(1);
  s_t q2 = positions[2] * flipAxisMap(2);

  // s_t dq0 = mVelocities[0];
  s_t dq1 = velocities[1] * flipAxisMap(1);
  s_t dq2 = velocities[2] * flipAxisMap(2);

  // s_t c0 = cos(q0);
  s_t c1 = cos(q1);
  s_t c2 = cos(q2);

  // s_t s0 = sin(q0);
  s_t s1 = sin(q1);
  s_t s2 = sin(q2);

  Eigen::Vector6s dJ0 = Eigen::Vector6s::Zero();
  Eigen::Vector6s dJ1 = Eigen::Vector6s::Zero();
  Eigen::Vector6s dJ2 = Eigen::Vector6s::Zero();

  switch (axisOrder)
  {
    case EulerJoint::AxisOrder::XYZ: {
      //------------------------------------------------------------------------
      // S = [    c1*c2, s2,  0
      //       -(c1*s2), c2,  0
      //             s1,  0,  1
      //              0,  0,  0
      //              0,  0,  0
      //              0,  0,  0 ];
      //------------------------------------------------------------------------
      //------------------------------------------------------------------------
      // dS = [  -(dq1*c2*s1) - dq2*c1*s2,    dq2*c2,  0
      //         -(dq2*c1*c2) + dq1*s1*s2, -(dq2*s2),  0
      //                           dq1*c1,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0 ];
      //------------------------------------------------------------------------
      dJ0 << -(dq1 * c2 * s1) - dq2 * c1 * s2, -(dq2 * c1 * c2) + dq1 * s1 * s2,
          dq1 * c1, 0, 0, 0;
      dJ1 << dq2 * c2, -(dq2 * s2), 0.0, 0.0, 0.0, 0.0;
      dJ2.setZero();

      break;
    }
    case EulerJoint::AxisOrder::ZYX: {
      //------------------------------------------------------------------------
      // S = [   -s1,    0,   1
      //       s2*c1,   c2,   0
      //       c1*c2,  -s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      //------------------------------------------------------------------------
      // dS = [               -c1*dq1,        0,   0
      //          c2*c1*dq2-s2*s1*dq1,  -s2*dq2,   0
      //         -s1*c2*dq1-c1*s2*dq2,  -c2*dq2,   0
      //                            0,        0,   0
      //                            0,        0,   0
      //                            0,        0,   0 ];
      //------------------------------------------------------------------------
      dJ0 << -c1 * dq1, c2 * c1 * dq2 - s2 * s1 * dq1,
          -s1 * c2 * dq1 - c1 * s2 * dq2, 0.0, 0.0, 0.0;
      dJ1 << 0.0, -s2 * dq2, -c2 * dq2, 0.0, 0.0, 0.0;
      dJ2.setZero();
      break;
    }
    case EulerJoint::AxisOrder::XZY: {
      //------------------------------------------------------------------------
      // S = [ c1*c2,  -s2,   0
      //         -s1,    0,   1
      //       c1*s2,   c2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      //------------------------------------------------------------------------
      // dS = [ -(c2 * s1*dq1) -(c1 * s2*dq2),   -c2*dq2,   0
      //                              -c1*dq1,    0,        0
      //        (c1 * c2*dq2) - (s2 * s1*dq1),   -s2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      dJ0 << -(c2 * s1 * dq1) - (c1 * s2 * dq2), -c1 * dq1,
          (c1 * c2 * dq2) - (s2 * s1 * dq1), 0.0, 0.0, 0.0;
      dJ1 << -c2 * dq2, 0, -s2 * dq2, 0.0, 0.0, 0.0;
      dJ2.setZero();
      break;
    }

    case EulerJoint::AxisOrder::ZXY: {
      //------------------------------------------------------------------------
      // S = [-c1*s2,   c2,   0
      //          s1,    0,   1
      //       c1*c2,   s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      //------------------------------------------------------------------------
      // dS = [(s1*s2*dq1) - (c1*c2*dq2),  -s2*dq2,   0
      //                          c1*dq1,        0,   0
      //      -(s1*c2*dq1) - (c1*s2*dq2),   c2*dq2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      dJ0 << (s1 * s2 * dq1) - (c1 * c2 * dq2), c1 * dq1,
          -(s1 * c2 * dq1) - (c1 * s2 * dq2), 0.0, 0.0, 0.0;
      dJ1 << -s2 * dq2, 0, c2 * dq2, 0.0, 0.0, 0.0;
      dJ2.setZero();
      break;
    }
    default: {
      dterr << "Undefined Euler axis order in "
               "computeRelativeJacobianTimeDerivStatic\n";
      break;
    }
  }

  dJ.col(0) = math::AdT(childBodyToJoint, dJ0) * flipAxisMap(0);
  dJ.col(1) = math::AdT(childBodyToJoint, dJ1) * flipAxisMap(1);
  dJ.col(2) = math::AdT(childBodyToJoint, dJ2) * flipAxisMap(2);

  assert(!math::isNan(dJ));
  return dJ;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EulerJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
    const Eigen::Vector3s& positions,
    const Eigen::Vector3s& velocities,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  const s_t EPS = 1e-8;
  Eigen::Vector3s perturbedPlus = positions + (EPS * velocities);
  Eigen::Vector3s perturbedMinus = positions - (EPS * velocities);

  Eigen::Matrix<s_t, 6, 3> plus = computeRelativeJacobianStatic(
      perturbedPlus, axisOrder, flipAxisMap, childBodyToJoint);
  Eigen::Matrix<s_t, 6, 3> minus = computeRelativeJacobianStatic(
      perturbedMinus, axisOrder, flipAxisMap, childBodyToJoint);
  return (plus - minus) / (2 * EPS);
}

//==============================================================================
void EulerJoint::updateRelativeJacobianTimeDeriv() const
{
  mJacobianDeriv = computeRelativeJacobianTimeDerivStatic(
      getPositionsStatic(),
      getVelocitiesStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

} // namespace dynamics
} // namespace dart
