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
#include "dart/math/Geometry.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"

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
  Base::setProperties(
        static_cast<const Base::Properties&>(_properties));
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
  return EulerJoint::Properties(getGenericJointProperties(),
                                getEulerJointAspect()->getProperties());
}

//==============================================================================
void EulerJoint::copy(const EulerJoint& _otherJoint)
{
  if(this == &_otherJoint)
    return;

  setProperties(_otherJoint.getEulerJointProperties());
}

//==============================================================================
void EulerJoint::copy(const EulerJoint* _otherJoint)
{
  if(nullptr == _otherJoint)
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
Eigen::Isometry3d EulerJoint::convertToTransform(
    const Eigen::Vector3d& _positions, AxisOrder _ordering)
{
  return Eigen::Isometry3d(convertToRotation(_positions, _ordering));
}

//==============================================================================
Eigen::Isometry3d EulerJoint::convertToTransform(
    const Eigen::Vector3d &_positions) const
{
  return convertToTransform(_positions, getAxisOrder());
}

//==============================================================================
Eigen::Matrix3d EulerJoint::convertToRotation(
    const Eigen::Vector3d& _positions, AxisOrder _ordering)
{
  switch (_ordering)
  {
    case AxisOrder::XYZ:
      return math::eulerXYZToMatrix(_positions);
    case AxisOrder::ZYX:
      return math::eulerZYXToMatrix(_positions);
    default:
    {
      dterr << "[EulerJoint::convertToRotation] Invalid AxisOrder specified ("
            << static_cast<int>(_ordering) << ")\n";
      return Eigen::Matrix3d::Identity();
    }
  }
}

//==============================================================================
Eigen::Matrix3d EulerJoint::convertToRotation(const Eigen::Vector3d& _positions)
                                                                           const
{
  return convertToRotation(_positions, getAxisOrder());
}

//==============================================================================
Eigen::Matrix<double, 6, 3> EulerJoint::getRelativeJacobianStatic(
    const Eigen::Vector3d& _positions) const
{
  Eigen::Matrix<double, 6, 3> J;

  // double q0 = _positions[0];
  const double q1 = _positions[1];
  const double q2 = _positions[2];

  // double c0 = cos(q0);
  double c1 = cos(q1);
  double c2 = cos(q2);

  // double s0 = sin(q0);
  double s1 = sin(q1);
  double s2 = sin(q2);

  Eigen::Vector6d J0 = Eigen::Vector6d::Zero();
  Eigen::Vector6d J1 = Eigen::Vector6d::Zero();
  Eigen::Vector6d J2 = Eigen::Vector6d::Zero();

  switch (getAxisOrder())
  {
    case AxisOrder::XYZ:
    {
      //------------------------------------------------------------------------
      // S = [    c1*c2, s2,  0
      //       -(c1*s2), c2,  0
      //             s1,  0,  1
      //              0,  0,  0
      //              0,  0,  0
      //              0,  0,  0 ];
      //------------------------------------------------------------------------
      J0 << c1*c2, -(c1*s2),  s1, 0.0, 0.0, 0.0;
      J1 <<    s2,       c2, 0.0, 0.0, 0.0, 0.0;
      J2 <<   0.0,      0.0, 1.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (std::abs(getPositionsStatic()[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in ZYX-euler joint ["
                  << Joint::mAspectProperties.mName << "]. ("
                  << _positions[0] << ", "
                  << _positions[1] << ", "
                  << _positions[2] << ")"
                  << std::endl;
#endif

      break;
    }
    case AxisOrder::ZYX:
    {
      //------------------------------------------------------------------------
      // S = [   -s1,    0,   1
      //       s2*c1,   c2,   0
      //       c1*c2,  -s2,   0
      //           0,    0,   0
      //           0,    0,   0
      //           0,    0,   0 ];
      //------------------------------------------------------------------------
      J0 << -s1, s2*c1, c1*c2, 0.0, 0.0, 0.0;
      J1 << 0.0,    c2,   -s2, 0.0, 0.0, 0.0;
      J2 << 1.0,   0.0,   0.0, 0.0, 0.0, 0.0;

#ifndef NDEBUG
      if (std::abs(_positions[1]) == math::constantsd::pi() * 0.5)
        std::cout << "Singular configuration in ZYX-euler joint ["
                  << Joint::mAspectProperties.mName << "]. ("
                  << _positions[0] << ", "
                  << _positions[1] << ", "
                  << _positions[2] << ")"
                  << std::endl;
#endif

      break;
    }
    default:
    {
      dterr << "Undefined Euler axis order\n";
      break;
    }
  }

  J.col(0) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, J0);
  J.col(1) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, J1);
  J.col(2) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, J2);

  assert(!math::isNan(J));

#ifndef NDEBUG
  Eigen::MatrixXd JTJ = J.transpose() * J;
  Eigen::FullPivLU<Eigen::MatrixXd> luJTJ(JTJ);
  //    Eigen::FullPivLU<Eigen::MatrixXd> luS(mS);
  double det = luJTJ.determinant();
  if (det < 1e-5)
  {
    std::cout << "ill-conditioned Jacobian in joint [" << Joint::mAspectProperties.mName << "]."
              << " The determinant of the Jacobian is (" << det << ")."
              << std::endl;
    std::cout << "rank is (" << luJTJ.rank() << ")." << std::endl;
    std::cout << "det is (" << luJTJ.determinant() << ")." << std::endl;
    //        std::cout << "mS: \n" << mS << std::endl;
  }
#endif

  return J;
}

//==============================================================================
math::Jacobian EulerJoint::getRelativeJacobianDeriv(std::size_t index) const
{
  assert(index < 3);

  Eigen::Matrix<double, 6, 3> DJ_Dq = Eigen::Matrix<double, 6, 3>::Zero();

  const Eigen::Vector3d& q = getPositionsStatic();

  const double q1 = q[1];
  const double q2 = q[2];

  // double c0 = cos(q0);
  const double c1 = std::cos(q1);
  const double c2 = std::cos(q2);

  // double s0 = sin(q0);
  const double s1 = std::sin(q1);
  const double s2 = std::sin(q2);

  switch (getAxisOrder())
  {
    case AxisOrder::XYZ:
    {
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

        DJ_Dq(0, 0) = -s1*c2;
        DJ_Dq(1, 0) = -s1*s2;
        DJ_Dq(2, 0) = c1;
      }
      else if (index == 2)
      {
        // DS/Dq2 = [ -c1*s2,  c2,  0
        //            -c1*c2, -s2,  0
        //                s1,   0,  0
        //                 0,   0,  0
        //                 0,   0,  0
        //                 0,   0,  0 ];

        DJ_Dq(0, 0) = -c1*s2;
        DJ_Dq(1, 0) = -c1*c2;
        DJ_Dq(2, 0) = s1;

        DJ_Dq(0, 1) = c2;
        DJ_Dq(1, 1) = -s2;
      }
      break;
    }
    case AxisOrder::ZYX:
    {
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
        DJ_Dq(1, 0) = -s1*s2;
        DJ_Dq(2, 0) = -s1*c2;
      }
      else if (index == 2)
      {
        // DS/Dq2 = [     0,    0,   0
        //            c1*c2,  -s2,   0
        //           -c1*s2,  -c2,   0
        //                0,    0,   0
        //                0,    0,   0
        //                0,    0,   0 ];

        DJ_Dq(1, 0) = c1*c2;
        DJ_Dq(2, 0) = -c1*s2;

        DJ_Dq(1, 1) = -s2;
        DJ_Dq(2, 1) = -c2;
      }

      break;
    }
    default:
    {
      dterr << "Undefined Euler axis order\n";
      break;
    }
  }

  DJ_Dq = math::AdTJac(Joint::mAspectProperties.mT_ChildBodyToJoint, DJ_Dq);

  assert(!math::isNan(DJ_Dq));

  return DJ_Dq;
}

//==============================================================================
math::Jacobian EulerJoint::getRelativeJacobianTimeDerivDeriv(std::size_t index) const
{
  assert(index < 3);

  Eigen::Matrix<double, 6, 3> DdJ_Dq = Eigen::Matrix<double, 6, 3>::Zero();

  const Eigen::Vector3d& q = getPositionsStatic();
  const double q1 = q[1];
  const double q2 = q[2];

  // double dq0 = mVelocities[0];
  const Eigen::Vector3d& dq = getVelocitiesStatic();
  const double dq1 = dq[1];
  const double dq2 = dq[2];

  const double c1 = std::cos(q1);
  const double c2 = std::cos(q2);

  const double s1 = std::sin(q1);
  const double s2 = std::sin(q2);

  switch (getAxisOrder())
  {
    case AxisOrder::XYZ:
    {
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

        DdJ_Dq(0, 0) = -c1*c2*dq1 + s1*s2*dq2;
        DdJ_Dq(1, 0) = c1*s2*dq1 + s1*c2*dq2;
        DdJ_Dq(2, 0) = -s1*dq1;
      }
      else if (index == 2)
      {
        // DdS/Dq2 = [ s1*s2*dq1 - c1*c2*dq2, -s2*dq2, 0
        //             s1*c2*dq1 + c1*s2*dq2, -c2*dq2, 0
        //                                 0,       0, 0
        //                                 0,       0, 0
        //                                 0,       0, 0
        //                                 0,       0, 0 ];

        DdJ_Dq(0, 0) = s1*s2*dq1 - c1*c2*dq2;
        DdJ_Dq(1, 0) = s1*c2*dq1 + c1*s2*dq2;

        DdJ_Dq(0, 1) = -s2*dq2;
        DdJ_Dq(1, 1) = -c2*dq2;
      }
      break;
    }
    case AxisOrder::ZYX:
    {
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

        DdJ_Dq(0, 0) = s1*dq1;
        DdJ_Dq(1, 0) = -c1*s2*dq1 - s1*c2*dq2;
        DdJ_Dq(2, 0) = -c1*c2*dq1 + s1*s2*dq2;
      }
      else if (index == 2)
      {
        // DdS/Dq1 = [                      0,       0, 0
        //             -s1*c2*dq1 - c1*s2*dq2, -c2*dq2, 0
        //              s1*s2*dq1 - c1*c2*dq2,  s2*dq2, 0
        //                                  0,       0, 0
        //                                  0,       0, 0
        //                                  0,       0, 0 ];

        DdJ_Dq(1, 0) = -s1*c2*dq1 - c1*s2*dq2;
        DdJ_Dq(2, 0) = s1*s2*dq1 - c1*c2*dq2;

        DdJ_Dq(1, 1) = -c2*dq2;
        DdJ_Dq(2, 1) = s2*dq2;
      }
      break;
    }
    default:
    {
      dterr << "Undefined Euler axis order\n";
      break;
    }
  }

  DdJ_Dq = math::AdTJac(Joint::mAspectProperties.mT_ChildBodyToJoint, DdJ_Dq);

  assert(!math::isNan(DdJ_Dq));

  return DdJ_Dq;
}

//==============================================================================
math::Jacobian EulerJoint::getRelativeJacobianTimeDerivDeriv2(std::size_t index) const
{
  assert(index < 3);

  Eigen::Matrix<double, 6, 3> DdJ_Ddq = Eigen::Matrix<double, 6, 3>::Zero();

  const Eigen::Vector3d& q = getPositionsStatic();
  const double q1 = q[1];
  const double q2 = q[2];

  const Eigen::Vector3d& dq = getVelocitiesStatic();
  const double dq1 = dq[1];
  const double dq2 = dq[2];

  const double c1 = std::cos(q1);
  const double c2 = std::cos(q2);

  const double s1 = std::sin(q1);
  const double s2 = std::sin(q2);

  switch (getAxisOrder())
  {
    case AxisOrder::XYZ:
    {
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

        DdJ_Ddq(0, 0) = -s1*c2;
        DdJ_Ddq(1, 0) = s1*s2;
        DdJ_Ddq(2, 0) = c1;
      }
      else if (index == 2)
      {
        // DdS/Ddq2 = [ -c1*s2,  c2, 0
        //              -c1*c2, -s2, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0 ];

        DdJ_Ddq(0, 0) = -c1*s2;
        DdJ_Ddq(1, 0) = -c1*c2;

        DdJ_Ddq(0, 1) = c2;
        DdJ_Ddq(1, 1) = -s2;
      }
      break;
    }
    case AxisOrder::ZYX:
    {
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
        DdJ_Ddq(1, 0) = -s1*s2;
        DdJ_Ddq(2, 0) = -s1*c2;
      }
      else if (index == 2)
      {
        // DdS/Ddq1 = [      0,   0, 0
        //               c1*c2, -s2, 0
        //              -c1*s2, -c2, 0
        //                   0,   0, 0
        //                   0,   0, 0
        //                   0,   0, 0 ];

        DdJ_Ddq(1, 0) = c1*c2;
        DdJ_Ddq(2, 0) = -c1*s2;

        DdJ_Ddq(1, 1) = -s2;
        DdJ_Ddq(2, 1) = -c2;
      }
      break;
    }
    default:
    {
      dterr << "Undefined Euler axis order\n";
      break;
    }
  }

  DdJ_Ddq = math::AdTJac(Joint::mAspectProperties.mT_ChildBodyToJoint, DdJ_Ddq);

  assert(!math::isNan(DdJ_Ddq));

  return DdJ_Ddq;
}

//==============================================================================
EulerJoint::EulerJoint(const Properties& properties)
  : detail::EulerJointBase(properties)
{
  // Inherited Aspects must be created in the final joint class in reverse order
  // or else we get pure virtual function calls
  createEulerJointAspect(properties);
  createGenericJointAspect(properties);
  createJointAspect(properties);
}

//==============================================================================
Joint* EulerJoint::clone() const
{
  return new EulerJoint(getEulerJointProperties());
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
    default:
      dterr << "Unsupported axis order in EulerJoint named '" << Joint::mAspectProperties.mName
            << "' (" << static_cast<int>(getAxisOrder()) << ")\n";
  }

  if (affixes.size() == 3)
  {
    for (std::size_t i = 0; i < 3; ++i)
    {
      if(!mDofs[i]->isNamePreserved())
        mDofs[i]->setName(Joint::mAspectProperties.mName + affixes[i], false);
    }
  }
}

//==============================================================================
void EulerJoint::updateRelativeTransform() const
{
  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * convertToTransform(getPositionsStatic())
       * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();

  assert(math::verifyTransform(mT));
}

//==============================================================================
void EulerJoint::updateRelativeJacobian(bool) const
{
  mJacobian = getRelativeJacobianStatic(getPositionsStatic());
}

//==============================================================================
void EulerJoint::updateRelativeJacobianTimeDeriv() const
{
  // double q0 = mPositions[0];
  const Eigen::Vector3d& positions = getPositionsStatic();
  double q1 = positions[1];
  double q2 = positions[2];

  // double dq0 = mVelocities[0];
  const Eigen::Vector3d& velocities = getVelocitiesStatic();
  double dq1 = velocities[1];
  double dq2 = velocities[2];

  // double c0 = cos(q0);
  double c1 = cos(q1);
  double c2 = cos(q2);

  // double s0 = sin(q0);
  double s1 = sin(q1);
  double s2 = sin(q2);

  Eigen::Vector6d dJ0 = Eigen::Vector6d::Zero();
  Eigen::Vector6d dJ1 = Eigen::Vector6d::Zero();
  Eigen::Vector6d dJ2 = Eigen::Vector6d::Zero();

  switch (getAxisOrder())
  {
    case AxisOrder::XYZ:
    {
      //------------------------------------------------------------------------
      // dS = [  -(dq1*c2*s1) - dq2*c1*s2,    dq2*c2,  0
      //         -(dq2*c1*c2) + dq1*s1*s2, -(dq2*s2),  0
      //                           dq1*c1,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0
      //                                0,         0,  0 ];
      //------------------------------------------------------------------------
      dJ0 << -(dq1*c2*s1) - dq2*c1*s2, -(dq2*c1*c2) + dq1*s1*s2, dq1*c1,
             0, 0, 0;
      dJ1 << dq2*c2,                -(dq2*s2),    0.0, 0.0, 0.0, 0.0;
      dJ2.setZero();

      break;
    }
    case AxisOrder::ZYX:
    {
      //------------------------------------------------------------------------
      // dS = [               -c1*dq1,        0,   0
      //          c2*c1*dq2-s2*s1*dq1,  -s2*dq2,   0
      //         -s1*c2*dq1-c1*s2*dq2,  -c2*dq2,   0
      //                            0,        0,   0
      //                            0,        0,   0
      //                            0,        0,   0 ];
      //------------------------------------------------------------------------
      dJ0 << -c1*dq1, c2*c1*dq2 - s2*s1*dq1, -s1*c2*dq1 - c1*s2*dq2,
             0.0, 0.0, 0.0;
      dJ1 <<     0.0,               -s2*dq2,                -c2*dq2,
             0.0, 0.0, 0.0;
      dJ2.setZero();
      break;
    }
    default:
    {
      dterr << "Undefined Euler axis order\n";
      break;
    }
  }

  mJacobianDeriv.col(0) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, dJ0);
  mJacobianDeriv.col(1) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, dJ1);
  mJacobianDeriv.col(2) = math::AdT(Joint::mAspectProperties.mT_ChildBodyToJoint, dJ2);

  assert(!math::isNan(mJacobianDeriv));
}

}  // namespace dynamics
}  // namespace dart
