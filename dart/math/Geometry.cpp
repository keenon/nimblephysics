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

#include "dart/math/Geometry.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "dart/common/Console.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/MathTypes.hpp"

#define DART_EPSILON 1e-6

/***************************************************************
 * Here's a useful little Python script to generate second derivatives of the
 * euler transformations automatically:
 ***************************************************************

 code_block = """
  ret(0, 0) = cz * cy;
  ret(1, 0) = sz * cy;
  ret(2, 0) = -sy;

  ret(0, 1) = cz * sy * sx + -(sz * cx);
  ret(1, 1) = sz * sy * sx + cz * cx;
  ret(2, 1) = cy * sx;

  ret(0, 2) = cz * sy * cx + sz * sx;
  ret(1, 2) = sz * sy * cx + -(cz * sx);
  ret(2, 2) = cy * cx;
"""


def diff(code, dx):
  def process_clause(clause):
    clause = clause.replace('c'+dx, '!!!')
    clause = clause.replace('s'+dx, '(c'+dx+')')
    clause = clause.replace('!!!', '(-s'+dx+')')
    return clause

  def process_line(line):
    parts = line.split('=')
    if len(parts) == 1:
      return line

    eq = parts[1]
    body = eq.split(';')[0]
    clauses = body.split('+')
    clauses = [process_clause(clause) for clause in clauses if dx in clause]
    if len(clauses) == 0:
      clauses = [' 0']

    finished_eq = '+'.join(clauses)+';'

    return parts[0]+'='+finished_eq

  return '\n'.join(['  '+process_line(l) for l in code.split('\n')])


order = ['z', 'y', 'x']

for i in range(len(order)):
  inner_block = diff(code_block, order[i])
  lines = []
  lines.append('  if (firstIndex == '+str(i)+') {')
  lines.append('    if (secondIndex == 0) {')
  lines.append('      '+diff(inner_block, order[0]).strip())
  lines.append('    } else if (secondIndex == 1) {')
  lines.append('      '+diff(inner_block, order[1]).strip())
  lines.append('    } else if (secondIndex == 2) {')
  lines.append('      '+diff(inner_block, order[2]).strip())
  lines.append('    }')
  lines.append('  }')
  print('\n'.join(lines))

***************************************************/

namespace dart {
namespace math {

Eigen::Quaternion_s expToQuat(const Eigen::Vector3s& _v)
{
  s_t mag = _v.norm();

  if (mag > 1e-10)
  {
    Eigen::Quaternion_s q(Eigen::AngleAxis_s(mag, _v / mag));
    return q;
  }
  else
  {
    Eigen::Quaternion_s q(1, 0, 0, 0);
    return q;
  }
}

Eigen::Vector3s quatToExp(const Eigen::Quaternion_s& _q)
{
  Eigen::AngleAxis_s aa(_q);
  Eigen::Vector3s v = aa.axis();
  return v * aa.angle();
}

// Reference:
// http://www.geometrictools.com/LibMathematics/Algebra/Wm5Matrix3.inl
Eigen::Vector3s matrixToEulerXYX(const Eigen::Matrix3s& _R)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  // | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  // | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |
  // +-           -+   +-                                                -+

  if (_R(0, 0) < 1.0)
  {
    if (_R(0, 0) > -1.0)
    {
      // y_angle  = acos(r00)
      // x0_angle = atan2(r10,-r20)
      // x1_angle = atan2(r01,r02)
      s_t y = acos(_R(0, 0));
      s_t x0 = atan2(_R(1, 0), -_R(2, 0));
      s_t x1 = atan2(_R(0, 1), _R(0, 2));
      // return EA_UNIQUE;
      return Eigen::Vector3s(x0, y, x1);
    }
    else
    {
      // Not a unique solution:  x1_angle - x0_angle = atan2(-r12,r11)
      s_t y = constantsd::pi();
      s_t x0 = -atan2(-_R(1, 2), _R(1, 1));
      s_t x1 = 0.0;
      // return EA_NOT_UNIQUE_DIF;
      return Eigen::Vector3s(x0, y, x1);
    }
  }
  else
  {
    // Not a unique solution:  x1_angle + x0_angle = atan2(-r12,r11)
    s_t y = 0.0;
    s_t x0 = -atan2(-_R(1, 2), _R(1, 1));
    s_t x1 = 0.0;
    // return EA_NOT_UNIQUE_SUM;
    return Eigen::Vector3s(x0, y, x1);
  }
}

Eigen::Vector3s matrixToEulerXYZ(const Eigen::Matrix3s& _R)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+

  s_t x, y, z;

  if (_R(0, 2) > (1.0 - DART_EPSILON))
  {
    z = atan2(_R(1, 0), _R(1, 1));
    y = constantsd::half_pi();
    x = 0.0;
    return Eigen::Vector3s(x, y, z);
  }

  if (_R(0, 2) < -(1.0 - DART_EPSILON))
  {
    z = atan2(_R(1, 0), _R(1, 1));
    y = -constantsd::half_pi();
    x = 0.0;
    return Eigen::Vector3s(x, y, z);
  }

  z = -atan2(_R(0, 1), _R(0, 0));
  y = asin(_R(0, 2));
  x = -atan2(_R(1, 2), _R(2, 2));

  // order of return is the order of input
  return Eigen::Vector3s(x, y, z);
}

Eigen::Vector3s matrixToEulerZYX(const Eigen::Matrix3s& _R)
{
  s_t x, y, z;

  if (_R(2, 0) > (1.0 - DART_EPSILON))
  {
    x = atan2(_R(0, 1), _R(0, 2));
    y = -constantsd::half_pi();
    z = 0.0;
    return Eigen::Vector3s(z, y, x);
  }

  if (_R(2, 0) < -(1.0 - DART_EPSILON))
  {
    x = atan2(_R(0, 1), _R(0, 2));
    y = constantsd::half_pi();
    z = 0.0;
    return Eigen::Vector3s(z, y, x);
  }

  x = atan2(_R(2, 1), _R(2, 2));
  y = -asin(_R(2, 0));
  z = atan2(_R(1, 0), _R(0, 0));

  // order of return is the order of input
  return Eigen::Vector3s(z, y, x);
}

Eigen::Vector3s matrixToEulerXZY(const Eigen::Matrix3s& _R)
{
  s_t x, y, z;

  if (_R(0, 1) > (1.0 - DART_EPSILON))
  {
    y = atan2(_R(1, 2), _R(1, 0));
    z = -constantsd::half_pi();
    x = 0.0;
    return Eigen::Vector3s(x, z, y);
  }

  if (_R(0, 1) < -(1.0 - DART_EPSILON))
  {
    y = atan2(_R(1, 2), _R(1, 0));
    z = constantsd::half_pi();
    x = 0.0;
    return Eigen::Vector3s(x, z, y);
  }

  y = atan2(_R(0, 2), _R(0, 0));
  z = -asin(_R(0, 1));
  x = atan2(_R(2, 1), _R(1, 1));

  // order of return is the order of input
  return Eigen::Vector3s(x, z, y);
}

Eigen::Vector3s matrixToEulerYZX(const Eigen::Matrix3s& _R)
{
  s_t x, y, z;

  if (_R(1, 0) > (1.0 - DART_EPSILON))
  {
    x = -atan2(_R(0, 2), _R(0, 1));
    z = constantsd::half_pi();
    y = 0.0;
    return Eigen::Vector3s(y, z, x);
  }

  if (_R(1, 0) < -(1.0 - DART_EPSILON))
  {
    x = -atan2(_R(0, 2), _R(0, 1));
    z = -constantsd::half_pi();
    y = 0.0;
    return Eigen::Vector3s(y, z, x);
  }

  x = -atan2(_R(1, 2), _R(1, 1));
  z = asin(_R(1, 0));
  y = -atan2(_R(2, 0), _R(0, 0));

  // order of return is the order of input
  return Eigen::Vector3s(y, z, x);
}

Eigen::Vector3s matrixToEulerZXY(const Eigen::Matrix3s& _R)
{
  s_t x, y, z;

  if (_R(2, 1) > (1.0 - DART_EPSILON))
  {
    y = atan2(_R(0, 2), _R(0, 0));
    x = constantsd::half_pi();
    z = 0.0;
    return Eigen::Vector3s(z, x, y);
  }

  if (_R(2, 1) < -(1.0 - DART_EPSILON))
  {
    y = atan2(_R(0, 2), _R(0, 0));
    x = -constantsd::half_pi();
    z = 0.0;
    return Eigen::Vector3s(z, x, y);
  }

  y = -atan2(_R(2, 0), _R(2, 2));
  x = asin(_R(2, 1));
  z = -atan2(_R(0, 1), _R(1, 1));

  // order of return is the order of input
  return Eigen::Vector3s(z, x, y);
}

Eigen::Vector3s matrixToEulerYXZ(const Eigen::Matrix3s& _R)
{
  s_t x, y, z;

  if (_R(1, 2) > (1.0 - DART_EPSILON))
  {
    z = -atan2(_R(0, 1), _R(0, 0));
    x = -constantsd::half_pi();
    y = 0.0;
    return Eigen::Vector3s(y, x, z);
  }

  if (_R(1, 2) < -(1.0 - DART_EPSILON))
  {
    z = -atan2(_R(0, 1), _R(0, 0));
    x = constantsd::half_pi();
    y = 0.0;
    return Eigen::Vector3s(y, x, z);
  }

  z = atan2(_R(1, 0), _R(1, 1));
  x = -asin(_R(1, 2));
  y = atan2(_R(0, 2), _R(2, 2));

  // order of return is the order of input
  return Eigen::Vector3s(y, x, z);
}

// get the derivative of rotation matrix wrt el no.
Eigen::Matrix3s quatDeriv(const Eigen::Quaternion_s& _q, int _el)
{
  Eigen::Matrix3s mat = Eigen::Matrix3s::Zero();

  switch (_el)
  {
    case 0: // wrt w
      mat(0, 0) = _q.w();
      mat(1, 1) = _q.w();
      mat(2, 2) = _q.w();
      mat(0, 1) = -_q.z();
      mat(1, 0) = _q.z();
      mat(0, 2) = _q.y();
      mat(2, 0) = -_q.y();
      mat(1, 2) = -_q.x();
      mat(2, 1) = _q.x();
      break;
    case 1: // wrt x
      mat(0, 0) = _q.x();
      mat(1, 1) = -_q.x();
      mat(2, 2) = -_q.x();
      mat(0, 1) = _q.y();
      mat(1, 0) = _q.y();
      mat(0, 2) = _q.z();
      mat(2, 0) = _q.z();
      mat(1, 2) = -_q.w();
      mat(2, 1) = _q.w();
      break;
    case 2: // wrt y
      mat(0, 0) = -_q.y();
      mat(1, 1) = _q.y();
      mat(2, 2) = -_q.y();
      mat(0, 1) = _q.x();
      mat(1, 0) = _q.x();
      mat(0, 2) = _q.w();
      mat(2, 0) = -_q.w();
      mat(1, 2) = _q.z();
      mat(2, 1) = _q.z();
      break;
    case 3: // wrt z
      mat(0, 0) = -_q.z();
      mat(1, 1) = -_q.z();
      mat(2, 2) = _q.z();
      mat(0, 1) = -_q.w();
      mat(1, 0) = _q.w();
      mat(0, 2) = _q.x();
      mat(2, 0) = _q.x();
      mat(1, 2) = _q.y();
      mat(2, 1) = _q.y();
      break;
    default:
      break;
  }

  return 2 * mat;
}

Eigen::Matrix3s quatSecondDeriv(
    const Eigen::Quaternion_s& /*_q*/, int _el1, int _el2)
{
  Eigen::Matrix3s mat = Eigen::Matrix3s::Zero();

  if (_el1 == _el2)
  { // wrt same dof
    switch (_el1)
    {
      case 0: // wrt w
        mat(0, 0) = 1;
        mat(1, 1) = 1;
        mat(2, 2) = 1;
        break;
      case 1: // wrt x
        mat(0, 0) = 1;
        mat(1, 1) = -1;
        mat(2, 2) = -1;
        break;
      case 2: // wrt y
        mat(0, 0) = -1;
        mat(1, 1) = 1;
        mat(2, 2) = -1;
        break;
      case 3: // wrt z
        mat(0, 0) = -1;
        mat(1, 1) = -1;
        mat(2, 2) = 1;
        break;
    }
  }
  else
  { // wrt different dofs
    // arrange in increasing order
    if (_el1 > _el2)
    {
      int temp = _el2;
      _el2 = _el1;
      _el1 = temp;
    }

    switch (_el1)
    {
      case 0: // wrt w
        switch (_el2)
        {
          case 1: // wrt x
            mat(1, 2) = -1;
            mat(2, 1) = 1;
            break;
          case 2: // wrt y
            mat(0, 2) = 1;
            mat(2, 0) = -1;
            break;
          case 3: // wrt z
            mat(0, 1) = -1;
            mat(1, 0) = 1;
            break;
        }
        break;
      case 1: // wrt x
        switch (_el2)
        {
          case 2: // wrt y
            mat(0, 1) = 1;
            mat(1, 0) = 1;
            break;
          case 3: // wrt z
            mat(0, 2) = 1;
            mat(2, 0) = 1;
            break;
        }
        break;
      case 2: // wrt y
        switch (_el2)
        {
          case 3: // wrt z
            mat(1, 2) = 1;
            mat(2, 1) = 1;
            break;
        }
        break;
    }
  }

  return 2 * mat;
}

Eigen::Vector3s rotatePoint(
    const Eigen::Quaternion_s& _q, const Eigen::Vector3s& _pt)
{
  Eigen::Quaternion_s quat_pt(0, _pt[0], _pt[1], _pt[2]);
  Eigen::Quaternion_s qinv = _q.inverse();

  Eigen::Quaternion_s rot = _q * quat_pt * qinv;

  // check below - assuming same format of point achieved
  Eigen::Vector3s temp;
  //  VLOG(1) << "Point before: " << 0 << " "
  //          << pt.x << " " << pt.y << " " << pt.z << "\n";
  //  VLOG(1) << "Point after:  "
  //          << rot.x << " " << rot.y << " " << rot.z << " " << rot.w << "\n";
  temp[0] = rot.x();
  temp[1] = rot.y();
  temp[2] = rot.z();

  //  VLOG(1) << "Point after rotation: "
  //          << temp[0] << " " << temp[1] << " " << temp[2] << endl;
  return temp;
}

Eigen::Vector3s rotatePoint(
    const Eigen::Quaternion_s& _q, s_t _x, s_t _y, s_t _z)
{
  Eigen::Vector3s pt(_x, _y, _z);
  return rotatePoint(_q, pt);
}

// ----------- expmap computations -------------

#define EPSILON_EXPMAP_THETA 1.0e-3

Eigen::Matrix3s expMapRot(const Eigen::Vector3s& _q)
{
  s_t theta = _q.norm();

  Eigen::Matrix3s R = Eigen::Matrix3s::Zero();
  Eigen::Matrix3s qss = math::makeSkewSymmetric(_q);
  Eigen::Matrix3s qss2 = qss * qss;

  if (theta < EPSILON_EXPMAP_THETA)
    R = Eigen::Matrix3s::Identity() + qss + 0.5 * qss2;
  else
    R = Eigen::Matrix3s::Identity() + (sin(theta) / theta) * qss
        + ((1 - cos(theta)) / (theta * theta)) * qss2;

  return R;
}

Eigen::Matrix3s expMapJac(const Eigen::Vector3s& _q)
{
  s_t theta = _q.norm();

  Eigen::Matrix3s J = Eigen::Matrix3s::Zero();
  Eigen::Matrix3s qss = math::makeSkewSymmetric(_q);
  Eigen::Matrix3s qss2 = qss * qss;

  if (theta < EPSILON_EXPMAP_THETA)
    J = Eigen::Matrix3s::Identity() + 0.5 * qss + (1.0 / 6.0) * qss2;
  else
    J = Eigen::Matrix3s::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
        + ((theta - sin(theta)) / (theta * theta * theta)) * qss2;

  return J;
}

/// \brief Computes the Jacobian of the logMap(R * expMapRot(expMap))
Eigen::Matrix3s expMapJacAt(
    const Eigen::Vector3s& _expmap, const Eigen::Matrix3s& R)
{
  Eigen::Matrix3s J = Eigen::Matrix3s::Zero();
  const s_t EPS = 1e-5;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3s perturb = Eigen::Vector3s::Unit(i) * EPS;
    Eigen::Vector3s plus = logMap(R * expMapRot(_expmap + perturb));
    Eigen::Vector3s minus = logMap(R * expMapRot(_expmap - perturb));
    J.col(i) = (plus - minus) / (2 * EPS);
  }
  return J;
}

Eigen::Matrix3s expMapJacDot(
    const Eigen::Vector3s& _q, const Eigen::Vector3s& _qdot)
{
  s_t theta = _q.norm();

  Eigen::Matrix3s Jdot = Eigen::Matrix3s::Zero();
  Eigen::Matrix3s qss = math::makeSkewSymmetric(_q);
  Eigen::Matrix3s qss2 = qss * qss;
  Eigen::Matrix3s qdss = math::makeSkewSymmetric(_qdot);
  s_t ttdot = _q.dot(_qdot); // theta*thetaDot
  s_t st = sin(theta);
  s_t ct = cos(theta);
  s_t t2 = theta * theta;
  s_t t3 = t2 * theta;
  s_t t4 = t3 * theta;
  s_t t5 = t4 * theta;

  if (theta < EPSILON_EXPMAP_THETA)
  {
    Jdot = 0.5 * qdss + (1.0 / 6.0) * (qss * qdss + qdss * qss);
    Jdot += (-1.0 / 12) * ttdot * qss + (-1.0 / 60) * ttdot * qss2;
  }
  else
  {
    Jdot = ((1 - ct) / t2) * qdss
           + ((theta - st) / t3) * (qss * qdss + qdss * qss);
    Jdot += ((theta * st + 2 * ct - 2) / t4) * ttdot * qss
            + ((3 * st - theta * ct - 2 * theta) / t5) * ttdot * qss2;
  }

  return Jdot;
}

Eigen::Matrix3s expMapJacDeriv(const Eigen::Vector3s& _q, int _qi)
{
  assert(_qi >= 0 && _qi <= 2);

  Eigen::Vector3s qdot = Eigen::Vector3s::Zero();
  qdot[_qi] = 1.0;
  return expMapJacDot(_q, qdot);
}

Eigen::Vector3s expMapGradient(const Eigen::Vector3s& pos, int _qi)
{
  assert(_qi >= 0 && _qi <= 2);

  Eigen::MatrixXs original = expMapRot(pos);

  // TODO: maybe we can handle this with dLogMap?
  s_t EPS = 1e-7;
  Eigen::Vector3s perturbed = pos;
  perturbed(_qi) += EPS;
  Eigen::Vector3s plus = logMap(original.transpose() * expMapRot(perturbed));
  perturbed = pos;
  perturbed(_qi) -= EPS;
  Eigen::Vector3s minus = logMap(original.transpose() * expMapRot(perturbed));

  return (plus - minus) / (2 * EPS);
}

Eigen::Matrix3s expMapMagGradient(const Eigen::Vector3s& screw)
{
  return makeSkewSymmetric(screw);
}

Eigen::Matrix3s finiteDifferenceExpMapMagGradient(
    const Eigen::Vector3s& screw, bool useRidders)
{
  s_t eps = useRidders ? 1e-3 : 1e-7;
  Eigen::Matrix3s result;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        perturbed = expMapRot(screw * eps);
        return true;
      },
      result,
      eps,
      useRidders);
  return result;
}

Eigen::Vector3s expMapNestedGradient(
    const Eigen::Vector3s& original, const Eigen::Vector3s& screw)
{
  Eigen::MatrixXs R = expMapRot(original);
  Eigen::MatrixXs dR = makeSkewSymmetric(screw) * R;

  return dLogMap(R, dR);
}

Eigen::Vector3s finiteDifferenceExpMapNestedGradient(
    const Eigen::Vector3s& original,
    const Eigen::Vector3s& screw,
    bool useRidders)
{
  Eigen::MatrixXs R = expMapRot(original);
  Eigen::Vector3s result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector3s& perturbed) {
        perturbed = logMap(expMapRot(screw * eps) * R);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

// Vec3 AdInvTLinear(const SE3& T, const Vec3& v)
// {
//     return Vec3(T(0,0)*v[0] + T(1,0)*v[1] + T(2,0)*v[2],
//                 T(0,1)*v[0] + T(1,1)*v[1] + T(2,1)*v[2],
//                 T(0,2)*v[0] + T(1,2)*v[1] + T(2,2)*v[2]);
// }

// Vec3 ad_Vec3_se3(const Vec3& s1, const se3& s2)
// {
//     Vec3 ret;

//     ret << s2[2]*s1[1] - s2[1]*s1[2],
//            s2[0]*s1[2] - s2[2]*s1[0],
//            s2[1]*s1[0] - s2[0]*s1[1];

//     return ret;
// }

Eigen::Vector3s logMap(const Eigen::Matrix3s& _R)
{
  //--------------------------------------------------------------------------
  // T = (R, p) = exp([w, v]), t = ||w||
  // v = beta*p + gamma*w + 1 / 2*cross(p, w)
  //    , beta = t*(1 + cos(t)) / (2*sin(t)), gamma = <w, p>*(1 - beta) / t^2
  //--------------------------------------------------------------------------
  s_t theta
      = acos(max(min(0.5 * (_R(0, 0) + _R(1, 1) + _R(2, 2) - 1.0), 1.0), -1.0));

  if (theta > constantsd::pi() - DART_EPSILON)
  {
    s_t delta
        = 0.5 + 0.125 * (constantsd::pi() - theta) * (constantsd::pi() - theta);

    return Eigen::Vector3s(
        _R(2, 1) > _R(1, 2) ? theta * sqrt(1.0 + (_R(0, 0) - 1.0) * delta)
                            : -theta * sqrt(1.0 + (_R(0, 0) - 1.0) * delta),
        _R(0, 2) > _R(2, 0) ? theta * sqrt(1.0 + (_R(1, 1) - 1.0) * delta)
                            : -theta * sqrt(1.0 + (_R(1, 1) - 1.0) * delta),
        _R(1, 0) > _R(0, 1) ? theta * sqrt(1.0 + (_R(2, 2) - 1.0) * delta)
                            : -theta * sqrt(1.0 + (_R(2, 2) - 1.0) * delta));
  }
  else
  {
    s_t alpha = 0.0;

    if (theta > DART_EPSILON)
      alpha = 0.5 * theta / sin(theta);
    else
      alpha = 0.5 + (1.0 / 12.0) * theta * theta;

    return Eigen::Vector3s(
        alpha * (_R(2, 1) - _R(1, 2)),
        alpha * (_R(0, 2) - _R(2, 0)),
        alpha * (_R(1, 0) - _R(0, 1)));
  }

  // Eigen::AngleAxis_s aa(_R);
  // return aa.angle() * aa.axis();
}

/// \brief Log mapping
/// \note This gets the value of d/dt logMap(R), given R and d/dt R
Eigen::Vector3s dLogMap(const Eigen::Matrix3s& _R, const Eigen::Matrix3s& dR)
{
  (void)dR;
  //--------------------------------------------------------------------------
  // T = (R, p) = exp([w, v]), t = ||w||
  // v = beta*p + gamma*w + 1 / 2*cross(p, w)
  //    , beta = t*(1 + cos(t)) / (2*sin(t)), gamma = <w, p>*(1 - beta) / t^2
  //--------------------------------------------------------------------------
  s_t diagSum = 0.5 * (_R(0, 0) + _R(1, 1) + _R(2, 2) - 1.0);
  s_t d_diagSum = 0.5 * (dR(0, 0) + dR(1, 1) + dR(2, 2));
  s_t d_theta = 0.0;
  if (diagSum >= 1.0)
  {
    diagSum = 1.0;
    d_theta = 0.0;
  }
  else if (diagSum <= -1.0)
  {
    diagSum = -1.0;
    d_theta = 0.0;
  }
  else
  {
    d_theta = -d_diagSum / sqrt(1 - diagSum * diagSum);
  }
  s_t theta = acos(diagSum);

  if (theta > constantsd::pi() - DART_EPSILON)
  {
    s_t delta
        = 0.5 + 0.125 * (constantsd::pi() - theta) * (constantsd::pi() - theta);
    s_t d_delta = 0.25 * (constantsd::pi() - theta) * -d_theta;

    // return Eigen::Vector3s(
    //     _R(2, 1) > _R(1, 2) ? theta * sqrt(1.0 + (_R(0, 0) - 1.0) * delta)
    //                         : -theta * sqrt(1.0 + (_R(0, 0) - 1.0) * delta),
    //     _R(0, 2) > _R(2, 0) ? theta * sqrt(1.0 + (_R(1, 1) - 1.0) * delta)
    //                         : -theta * sqrt(1.0 + (_R(1, 1) - 1.0) * delta),
    //     _R(1, 0) > _R(0, 1) ? theta * sqrt(1.0 + (_R(2, 2) - 1.0) * delta)
    //                         : -theta * sqrt(1.0 + (_R(2, 2) - 1.0) * delta));

    s_t elem1 = theta * sqrt(1.0 + delta * _R(0, 0) - delta);
    (void)elem1;
    s_t d_elem1 = d_theta * sqrt(1.0 + delta * _R(0, 0) - delta)
                  + theta * 0.5 / sqrt(1.0 + delta * _R(0, 0) - delta)
                        * (d_delta * _R(0, 0) + delta * dR(0, 0) - d_delta);
    s_t elem2 = theta * sqrt(1.0 + (_R(1, 1) - 1.0) * delta);
    (void)elem2;
    s_t d_elem2 = d_theta * sqrt(1.0 + delta * _R(1, 1) - delta)
                  + theta * 0.5 / sqrt(1.0 + delta * _R(1, 1) - delta)
                        * (d_delta * _R(1, 1) + delta * dR(1, 1) - d_delta);
    s_t elem3 = theta * sqrt(1.0 + (_R(2, 2) - 1.0) * delta);
    (void)elem3;
    s_t d_elem3 = d_theta * sqrt(1.0 + delta * _R(2, 2) - delta)
                  + theta * 0.5 / sqrt(1.0 + delta * _R(2, 2) - delta)
                        * (d_delta * _R(2, 2) + delta * dR(2, 2) - d_delta);

    return Eigen::Vector3s(
        _R(2, 1) > _R(1, 2) ? d_elem1 : -d_elem1,
        _R(0, 2) > _R(2, 0) ? d_elem2 : -d_elem2,
        _R(1, 0) > _R(0, 1) ? d_elem3 : -d_elem3);
  }
  else
  {
    s_t alpha = 0.0;
    s_t d_alpha = 0.0;

    if (theta > DART_EPSILON)
    {
      alpha = 0.5 * theta / sin(theta);
      s_t csc = 1.0 / sin(theta);
      // s_t cot = 1.0 / tan(theta);
      s_t cot
          = cos(theta) / sin(theta); // -> alternative form to 1.0 / tan(theta),
                                     // perhaps more numerically stable
      d_alpha = 0.5 * (d_theta * csc - theta * cot * csc * d_theta);
      // d_alpha = 0.5 * d_theta / cos(theta) * theta / sin(d_theta);
    }
    else
    {
      alpha = 0.5 + (1.0 / 12.0) * theta * theta;
      d_alpha = (2.0 / 12.0) * theta * d_theta;
    }

    /*
    return Eigen::Vector3s(
        alpha * (_R(2, 1) - _R(1, 2)),
        alpha * (_R(0, 2) - _R(2, 0)),
        alpha * (_R(1, 0) - _R(0, 1)));
    */

    return Eigen::Vector3s(
        d_alpha * (_R(2, 1) - _R(1, 2)) + alpha * (dR(2, 1) - dR(1, 2)),
        d_alpha * (_R(0, 2) - _R(2, 0)) + alpha * (dR(0, 2) - dR(2, 0)),
        d_alpha * (_R(1, 0) - _R(0, 1)) + alpha * (dR(1, 0) - dR(0, 1)));
  }
}

Eigen::Vector3s finiteDifferenceDLogMap(
    const Eigen::Matrix3s& R, const Eigen::Matrix3s& dR, bool useRidders)
{
  Eigen::Vector3s result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector3s& perturbed) {
        perturbed = logMap(R + dR * eps);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

Eigen::Vector6s logMap(const Eigen::Isometry3s& _T)
{
  //--------------------------------------------------------------------------
  // T = (R, p) = exp([w, v]), t = ||w||
  // v = beta*p + gamma*w + 1 / 2*cross(p, w)
  //    , beta = t*(1 + cos(t)) / (2*sin(t)), gamma = <w, p>*(1 - beta) / t^2
  //--------------------------------------------------------------------------
  s_t theta
      = acos(max(min(0.5 * (_T(0, 0) + _T(1, 1) + _T(2, 2) - 1.0), 1.0), -1.0));
  s_t beta;
  s_t gamma;
  Eigen::Vector6s ret;

  s_t PI = constantsd::pi();
  if (theta > PI - DART_EPSILON)
  {
#ifdef DART_USE_ARBITRARY_PRECISION
    const s_t c1 = static_cast<s_t>(1.0) / (PI * PI); // 1 / pi^2
    const s_t c2
        = (static_cast<s_t>(1.0) / static_cast<s_t>(4.0)) / PI
          - static_cast<s_t>(2.0) / (PI * PI * PI); // 1 / 4 / pi - 2 / pi^3
    const s_t c3 = static_cast<s_t>(3.0) / (PI * PI * PI * PI)
                   - (static_cast<s_t>(1.0) / static_cast<s_t>(4.0))
                         / (PI * PI); // 3 / pi^4 - 1 / 4 / pi^2
#else
    const s_t c1 = 0.10132118364234; // 1 / pi^2
    const s_t c2 = 0.01507440267955; // 1 / 4 / pi - 2 / pi^3
    const s_t c3 = 0.00546765085347; // 3 / pi^4 - 1 / 4 / pi^2
#endif

    s_t phi = constantsd::pi() - theta;
    s_t delta = 0.5 + 0.125 * phi * phi;

    s_t w[]
        = {_T(2, 1) > _T(1, 2) ? theta * sqrt(1.0 + (_T(0, 0) - 1.0) * delta)
                               : -theta * sqrt(1.0 + (_T(0, 0) - 1.0) * delta),
           _T(0, 2) > _T(2, 0) ? theta * sqrt(1.0 + (_T(1, 1) - 1.0) * delta)
                               : -theta * sqrt(1.0 + (_T(1, 1) - 1.0) * delta),
           _T(1, 0) > _T(0, 1) ? theta * sqrt(1.0 + (_T(2, 2) - 1.0) * delta)
                               : -theta * sqrt(1.0 + (_T(2, 2) - 1.0) * delta)};

    beta = 0.25 * theta * (constantsd::pi() - theta);
    gamma = (w[0] * _T(0, 3) + w[1] * _T(1, 3) + w[2] * _T(2, 3))
            * (c1 - c2 * phi + c3 * phi * phi);

    ret << w[0], w[1], w[2],
        beta * _T(0, 3) - 0.5 * (w[1] * _T(2, 3) - w[2] * _T(1, 3))
            + gamma * w[0],
        beta * _T(1, 3) - 0.5 * (w[2] * _T(0, 3) - w[0] * _T(2, 3))
            + gamma * w[1],
        beta * _T(2, 3) - 0.5 * (w[0] * _T(1, 3) - w[1] * _T(0, 3))
            + gamma * w[2];
  }
  else
  {
    s_t alpha;
    if (theta > DART_EPSILON)
    {
      alpha = 0.5 * theta / sin(theta);
      beta = (1.0 + cos(theta)) * alpha;
      gamma = (1.0 - beta) / theta / theta;
    }
    else
    {
      alpha = 0.5 + 1.0 / 12.0 * theta * theta;
      beta = 1.0 - 1.0 / 12.0 * theta * theta;
      gamma = 1.0 / 12.0 + 1.0 / 720.0 * theta * theta;
    }

    s_t w[]
        = {alpha * (_T(2, 1) - _T(1, 2)),
           alpha * (_T(0, 2) - _T(2, 0)),
           alpha * (_T(1, 0) - _T(0, 1))};
    gamma *= w[0] * _T(0, 3) + w[1] * _T(1, 3) + w[2] * _T(2, 3);

    ret << w[0], w[1], w[2],
        beta * _T(0, 3) + 0.5 * (w[2] * _T(1, 3) - w[1] * _T(2, 3))
            + gamma * w[0],
        beta * _T(1, 3) + 0.5 * (w[0] * _T(2, 3) - w[2] * _T(0, 3))
            + gamma * w[1],
        beta * _T(2, 3) + 0.5 * (w[1] * _T(0, 3) - w[0] * _T(1, 3))
            + gamma * w[2];
  }

  return ret;
}

Eigen::Vector3s gradientWrtTheta(
    const Eigen::Vector6s& _S, const Eigen::Vector3s& point, s_t theta)
{
  const Eigen::Vector3s& w = _S.head<3>();
  const Eigen::Vector3s& v = _S.tail<3>();
  s_t normW = w.norm();
  if (normW > DART_EPSILON)
  {
    const s_t cos_t = cos(theta);
    const s_t sin_t = sin(theta);
    const Eigen::Vector3s wp = w.cross(point);
    const Eigen::Vector3s wwp = w.cross(wp);
    const Eigen::Vector3s wv = w.cross(v);
    const Eigen::Vector3s wwv = w.cross(wv);
    return cos_t * (wp - wwv) + sin_t * (wwp + wv) + v + wwv;
  }
  else
  {
    // This means we don't rotate, so it's pure linear motion
    return v;
  }
}

/// This takes a screw axis and a point, and gives us the direction that the
/// point will move if we increase theta by an infinitesimal amount.
Eigen::Vector3s gradientWrtThetaSecondGrad(
    const Eigen::Vector6s& _S,
    const Eigen::Vector6s& d_S,
    const Eigen::Vector3s& point,
    const Eigen::Vector3s& dPoint,
    s_t theta)
{

  const Eigen::Vector3s& w = _S.head<3>();
  const Eigen::Vector3s& dw = d_S.head<3>();
  const Eigen::Vector3s& v = _S.tail<3>();
  const Eigen::Vector3s& dv = d_S.tail<3>();

  s_t normW = w.norm();
  if (normW > DART_EPSILON)
  {
    const s_t cos_t = cos(theta);
    const s_t sin_t = sin(theta);
    const Eigen::Vector3s wp = w.cross(point);
    const Eigen::Vector3s dwp = dw.cross(point) + w.cross(dPoint);
    const Eigen::Vector3s dwwp = dw.cross(wp) + w.cross(dwp);
    const Eigen::Vector3s wv = w.cross(v);
    const Eigen::Vector3s dwv = dw.cross(v) + w.cross(dv);
    const Eigen::Vector3s dwwv = dw.cross(wv) + w.cross(dwv);
    return cos_t * (dwp - dwwv) + sin_t * (dwwp + dwv) + dv + dwwv;
  }
  else
  {
    // This means we don't rotate, so it's pure linear motion
    return dv;
  }
}

Eigen::Vector3s gradientWrtThetaPureRotation(
    const Eigen::Vector3s& w, const Eigen::Vector3s& point, s_t theta)
{
  const s_t cos_t = cos(theta);
  const s_t sin_t = sin(theta);
  const Eigen::Vector3s wp = w.cross(point);
  const Eigen::Vector3s wwp = w.cross(wp);
  return cos_t * wp + sin_t * wwp;
}

/// Copied and modified to work with Eigen from the version by the same name in
/// DARTCollide.cpp
/// pa: a point on line A
/// ua: a unit vector in the direction of line A
/// pb: a point on line B
/// ub: a unit vector in the direction of B
void dLineClosestApproach(
    const Eigen::Vector3s& pa,
    const Eigen::Vector3s& ua,
    const Eigen::Vector3s& pb,
    const Eigen::Vector3s& ub,
    s_t* alpha,
    s_t* beta)
{
  Eigen::Vector3s p;
  p[0] = pb[0] - pa[0];
  p[1] = pb[1] - pa[1];
  p[2] = pb[2] - pa[2];
  s_t uaub = ua.dot(ub);
  s_t q1 = ua.dot(p);
  s_t q2 = -ub.dot(p);
  s_t d = 1 - uaub * uaub;
  if (d <= 0)
  {
    // @@@ this needs to be made more robust
    *alpha = 0;
    *beta = 0;
  }
  else
  {
    d = 1.0 / d;
    *alpha = (q1 + uaub * q2) * d;
    *beta = (uaub * q1 + q2) * d;
  }
}

//==============================================================================
/// This returns the average of the points on edge A and edge B closest to each
/// other.
Eigen::Vector3s getContactPoint(
    const Eigen::Vector3s& edgeAPoint,
    const Eigen::Vector3s& edgeADir,
    const Eigen::Vector3s& edgeBPoint,
    const Eigen::Vector3s& edgeBDir,
    s_t radiusA,
    s_t radiusB)
{
  s_t alpha;
  s_t beta;
  dLineClosestApproach(
      edgeAPoint, edgeADir, edgeBPoint, edgeBDir, &alpha, &beta);
  Eigen::Vector3s closestPointA = edgeAPoint + alpha * edgeADir;
  Eigen::Vector3s closestPointB = edgeBPoint + beta * edgeBDir;
  Eigen::Vector3s resultA = (closestPointA * radiusB + closestPointB * radiusA)
                            / (radiusA + radiusB);

  /// TODO: this is the old way

  Eigen::Vector3s p = edgeBPoint - edgeAPoint;
  s_t uaub = edgeADir.dot(edgeBDir);
  s_t q1 = edgeADir.dot(p);
  s_t q2 = -edgeBDir.dot(p);
  s_t d = 1 - uaub * uaub;
  if (d <= 0)
  {
    // Comment from original code in DARTCollide.cpp: "@@@ this needs to be made
    // more robust" Don't try to find the nearest point, just average the points
    Eigen::Vector3s resultB
        = (edgeAPoint * radiusB + edgeBPoint * radiusA) / (radiusA + radiusB);
    if (resultA != resultB)
    {
      std::cout << "Error detected!" << std::endl;
    }
    return resultA;
  }
  else
  {
    d = 1.0 / d;
    s_t alpha = (q1 + uaub * q2) * d;
    s_t beta = (uaub * q1 + q2) * d;
    Eigen::Vector3s resultB = ((edgeAPoint + alpha * edgeADir) * radiusB
                               + (edgeBPoint + beta * edgeBDir) * radiusA)
                              / (radiusA + radiusB);
    if (resultA != resultB)
    {
      std::cout << "Error detected!" << std::endl;
    }
    return resultA;
  }
}

/// This returns gradient of the average of the points on edge A and edge B
/// closest to each other, allowing all the inputs to change.
Eigen::Vector3s getContactPointGradient(
    const Eigen::Vector3s& edgeAPoint,
    const Eigen::Vector3s& edgeAPointGradient,
    const Eigen::Vector3s& edgeADir,
    const Eigen::Vector3s& edgeADirGradient,
    const Eigen::Vector3s& edgeBPoint,
    const Eigen::Vector3s& edgeBPointGradient,
    const Eigen::Vector3s& edgeBDir,
    const Eigen::Vector3s& edgeBDirGradient,
    s_t radiusA,
    s_t radiusB)
{
  Eigen::Vector3s p = edgeBPoint - edgeAPoint;
  Eigen::Vector3s d_p = edgeBPointGradient - edgeAPointGradient;

  /*
  Eigen::Vector3s p_prime = (edgeBPoint + edgeBPointGradient * EPS)
                            - (edgeAPoint + edgeAPointGradient * EPS);
  */

  s_t uaub = edgeADir.dot(edgeBDir);
  s_t d_uaub = edgeADirGradient.dot(edgeBDir) + edgeADir.dot(edgeBDirGradient);

  /*
  s_t uaub_prime = (edgeADir + edgeADirGradient * EPS)
                          .dot(edgeBDir + edgeBDirGradient * EPS);
  s_t d_uaub_brute = (uaub_prime - uaub) / EPS;
  */

  s_t q1 = edgeADir.dot(p);
  s_t d_q1 = edgeADirGradient.dot(p) + edgeADir.dot(d_p);

  /*
  s_t q1_prime = (edgeADir + edgeADirGradient * EPS).dot(p_prime);
  s_t d_q1_brute = (q1_prime - q1) / EPS;
  */

  s_t q2 = -edgeBDir.dot(p);
  s_t d_q2 = -edgeBDirGradient.dot(p) - edgeBDir.dot(d_p);

  /*
  s_t q2_prime = -(edgeBDir + edgeBDirGradient * EPS).dot(p_prime);
  s_t d_q2_brute = (q2_prime - q2) / EPS;
  */

  s_t d = 1 - uaub * uaub;
  s_t d_d = -2 * d_uaub * uaub;

  /*
  s_t d_prime = 1 - uaub_prime * uaub_prime;
  s_t d_d_brute = (d_prime - d) / EPS;
  */

  if (d <= 0)
  {
    // Comment from original code in DARTCollide.cpp: "@@@ this needs to be made
    // more robust" Don't try to find the nearest point, just average the points
    return (edgeAPointGradient * radiusB + edgeBPointGradient * radiusA)
           / (radiusA + radiusB);
  }
  else
  {
    s_t e = 1.0 / d;
    s_t d_e = -(1.0 / (d * d)) * d_d;

    /*
    s_t e_prime = 1.0 / d_prime;
    s_t d_e_brute = (e_prime - e) / EPS;
    */

    s_t alpha = (q1 + uaub * q2) * e;
    s_t d_alpha
        = (q1 + uaub * q2) * d_e + (d_q1 + d_uaub * q2 + uaub * d_q2) * e;

    /*
    s_t alpha_prime = (q1_prime + uaub_prime * q2_prime) * e_prime;
    s_t d_alpha_brute = (alpha_prime - alpha) / EPS;
    */

    s_t beta = (uaub * q1 + q2) * e;
    s_t d_beta
        = (uaub * q1 + q2) * d_e + (d_uaub * q1 + uaub * d_q1 + d_q2) * e;

    /*
    s_t beta_prime = (uaub_prime * q1_prime + q2_prime) * e_prime;
    s_t d_beta_brute = (beta_prime - beta) / EPS;
    */

    /*
     Eigen::Vector3s d_offsetA = alpha * edgeADirGradient + d_alpha * edgeADir;
     Eigen::Vector3s offsetA = alpha * edgeADir;
     Eigen::Vector3s offsetA_prime
         = alpha_prime * (edgeADir + edgeADirGradient * EPS);
     Eigen::Vector3s d_offsetA_brute = (offsetA_prime - offsetA) / EPS;

     Eigen::Vector3s d_offsetB = beta * edgeBDirGradient + d_beta * edgeBDir;
     Eigen::Vector3s offsetB = beta * edgeBDir;
     Eigen::Vector3s offsetB_prime
         = beta_prime * (edgeBDir + edgeBDirGradient * EPS);
     Eigen::Vector3s d_offsetB_brute = (offsetB_prime - offsetB) / EPS;
     */

    return ((edgeAPointGradient + alpha * edgeADirGradient + d_alpha * edgeADir)
                * radiusB
            + (edgeBPointGradient + beta * edgeBDirGradient + d_beta * edgeBDir)
                  * radiusA)
           / (radiusA + radiusB);
  }
}

Eigen::VectorXs dampedPInv(
    const Eigen::MatrixXs& J, const Eigen::VectorXs& x, s_t damping)
{
  int rows = J.rows(), cols = J.cols();
  if (rows <= cols)
  {
    return J.transpose()
           * (pow(damping, 2) * Eigen::MatrixXs::Identity(rows, rows)
              + J * J.transpose())
                 .inverse()
           * x;
  }
  else
  {
    return (pow(damping, 2) * Eigen::MatrixXs::Identity(cols, cols)
            + J.transpose() * J)
               .inverse()
           * J.transpose() * x;
  }
}

bool hasTinySingularValues(const Eigen::MatrixXs& J, s_t clippingThreshold)
{
  Eigen::JacobiSVD<Eigen::MatrixXs> svd(
      J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXs singulars = svd.singularValues();
  for (int i = 0; i < singulars.size(); i++)
  {
    if (abs(singulars(i)) < clippingThreshold)
    {
      return true;
    }
  }
  return false;
}

Eigen::MatrixXs clippedSingularsPinv(
    const Eigen::MatrixXs& J, s_t clippingThreshold)
{
  Eigen::JacobiSVD<Eigen::MatrixXs> svd(
      J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXs singulars = svd.singularValues();
  Eigen::MatrixXs Ds = singulars.asDiagonal();
  Eigen::MatrixXs recovered = svd.matrixU() * Ds * svd.matrixV().transpose();
  // Kill any small singular values before inverting, to avoid numerical
  // instability issues.
  for (int i = 0; i < singulars.size(); i++)
  {
    if (abs(singulars(i)) < clippingThreshold)
    {
      singulars(i) = 0.0;
    }
    else
    {
      singulars(i) = 1.0 / singulars(i);
    }
  }
  return svd.matrixV() * singulars.asDiagonal() * svd.matrixU().transpose();
}

// res = T * s * Inv(T)
Eigen::Vector6s AdT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V)
{
  //--------------------------------------------------------------------------
  // w' = R*w
  // v' = p x R*w + R*v
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _V.head<3>();
  res.tail<3>().noalias()
      = _T.linear() * _V.tail<3>() + _T.translation().cross(res.head<3>());
  return res;
}

//==============================================================================
Eigen::Matrix6s getAdTMatrix(const Eigen::Isometry3s& T)
{
  Eigen::Matrix6s AdT;

  AdT.topLeftCorner<3, 3>() = T.linear();
  AdT.topRightCorner<3, 3>().setZero();
  AdT.bottomLeftCorner<3, 3>()
      = makeSkewSymmetric(T.translation()) * T.linear();
  AdT.bottomRightCorner<3, 3>() = T.linear();

  return AdT;
}

//==============================================================================
Eigen::Matrix6s AdTMatrix(const Eigen::Isometry3s& T)
{
  return getAdTMatrix(T);
}

//==============================================================================
Eigen::Matrix6s AdInvTMatrix(const Eigen::Isometry3s& T)
{
  Eigen::Matrix6s AdT;

  AdT.topRightCorner<3, 3>().setZero();

  AdT.topLeftCorner<3, 3>() = T.linear().transpose();
  AdT.bottomRightCorner<3, 3>() = AdT.topLeftCorner<3, 3>();

  AdT.bottomLeftCorner<3, 3>()
      = AdT.topLeftCorner<3, 3>() * makeSkewSymmetric(-T.translation());

  return AdT;
}

//==============================================================================
Eigen::Matrix6s dAdTMatrix(const Eigen::Isometry3s& T)
{
  Eigen::Matrix6s AdT;

  AdT.bottomLeftCorner<3, 3>().setZero();

  AdT.topLeftCorner<3, 3>() = T.linear();
  AdT.bottomRightCorner<3, 3>() = AdT.topLeftCorner<3, 3>();

  AdT.topRightCorner<3, 3>() = makeSkewSymmetric(T.translation()) * T.linear();

  return AdT;
}

//==============================================================================
Eigen::Matrix6s dAdInvTMatrix(const Eigen::Isometry3s& T)
{
  return AdInvTMatrix(T).transpose();
}

Eigen::Vector6s AdR(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V)
{
  //--------------------------------------------------------------------------
  // w' = R*w
  // v' = R*v
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _V.head<3>();
  res.tail<3>().noalias() = _T.linear() * _V.tail<3>();
  return res;
}

Eigen::Vector6s AdTAngular(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _w)
{
  //--------------------------------------------------------------------------
  // w' = R*w
  // v' = p x R*w
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _w;
  res.tail<3>() = _T.translation().cross(res.head<3>());
  return res;
}

Eigen::Vector6s AdTLinear(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _v)
{
  //--------------------------------------------------------------------------
  // w' = 0
  // v' = R*v
  //--------------------------------------------------------------------------
  Eigen::Vector6s res = Eigen::Vector6s::Zero();
  res.tail<3>().noalias() = _T.linear() * _v;
  return res;
}

// se3 AdP(const Vec3& p, const se3& s)
// {
//  //--------------------------------------------------------------------------
//  // w' = w
//  // v' = p x w + v
//  //--------------------------------------------------------------------------
//  se3 ret;
//  ret << s[0],
//      s[1],
//      s[2],
//      p[1]*s[2] - p[2]*s[1] + s[3],
//      p[2]*s[0] - p[0]*s[2] + s[4],
//      p[0]*s[1] - p[1]*s[0] + s[5];

//  return ret;
// }

// Jacobian AdRJac(const SE3& T, const Jacobian& J)
// {
//     Jacobian AdTJ(6,J.cols());

//     for (int i = 0; i < J.cols(); ++i)
//     {
//         AdTJ.col(i) = math::AdR(T, J.col(i));
//     }

//     return AdTJ;
// }

// re = Inv(T)*s*T
Eigen::Vector6s AdInvT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V)
{
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear().transpose() * _V.head<3>();
  res.tail<3>().noalias()
      = _T.linear().transpose()
        * (_V.tail<3>() + _V.head<3>().cross(_T.translation()));
  return res;
}

// se3 AdInvR(const SE3& T, const se3& s)
// {
//     se3 ret;

//     ret << T(0,0)*s[0] + T(1,0)*s[1] + T(2,0)*s[2],
//                 T(0,1)*s[0] + T(1,1)*s[1] + T(2,1)*s[2],
//                 T(0,2)*s[0] + T(1,2)*s[1] + T(2,2)*s[2],
//                 T(0,0)*s[3] + T(1,0)*s[4] + T(2,0)*s[5],
//                 T(0,1)*s[3] + T(1,1)*s[4] + T(2,1)*s[5],
//                 T(0,2)*s[3] + T(1,2)*s[4] + T(2,2)*s[5];

//     return ret;
// }

Eigen::Vector6s AdInvRLinear(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _v)
{
  Eigen::Vector6s res = Eigen::Vector6s::Zero();
  res.tail<3>().noalias() = _T.linear().transpose() * _v;
  return res;
}

Eigen::Vector6s ad(const Eigen::Vector6s& _X, const Eigen::Vector6s& _Y)
{
  //--------------------------------------------------------------------------
  // ad(s1, s2) = | [w1]    0 | | w2 |
  //              | [v1] [w1] | | v2 |
  //
  //            = |          [w1]w2 |
  //              | [v1]w2 + [w1]v2 |
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>() = _X.head<3>().cross(_Y.head<3>());
  res.tail<3>()
      = _X.head<3>().cross(_Y.tail<3>()) + _X.tail<3>().cross(_Y.head<3>());
  return res;
}

Eigen::Matrix6s adMatrix(const Eigen::Vector6s& X)
{
  //--------------------------------------------------------------------------
  // ad(s) = | [w1]    0 |
  //         | [v1] [w1] |
  //--------------------------------------------------------------------------

  Eigen::Matrix6s res;

  res.topRightCorner<3, 3>().setZero();

  res.topLeftCorner<3, 3>() = makeSkewSymmetric(X.head<3>());
  res.bottomRightCorner<3, 3>() = res.topLeftCorner<3, 3>();

  res.bottomLeftCorner<3, 3>() = makeSkewSymmetric(X.tail<3>());

  return res;
}

Eigen::Vector6s dAdT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F)
{
  Eigen::Vector6s res;
  res.head<3>().noalias()
      = _T.linear().transpose()
        * (_F.head<3>() + _F.tail<3>().cross(_T.translation()));
  res.tail<3>().noalias() = _T.linear().transpose() * _F.tail<3>();
  return res;
}

// dse3 dAdTLinear(const SE3& T, const Vec3& v)
// {
//     dse3 ret;
//     s_t tmp[3] = { - T(1,3)*v[2] + T(2,3)*v[1],
//                       - T(2,3)*v[0] + T(0,3)*v[2],
//                       - T(0,3)*v[1] + T(1,3)*v[0] };
//     ret << T(0,0)*tmp[0] + T(1,0)*tmp[1] + T(2,0)*tmp[2],
//                 T(0,1)*tmp[0] + T(1,1)*tmp[1] + T(2,1)*tmp[2],
//                 T(0,2)*tmp[0] + T(1,2)*tmp[1] + T(2,2)*tmp[2],
//                 T(0,0)*v[0] + T(1,0)*v[1] + T(2,0)*v[2],
//                 T(0,1)*v[0] + T(1,1)*v[1] + T(2,1)*v[2],
//                 T(0,2)*v[0] + T(1,2)*v[1] + T(2,2)*v[2];

//     return ret;
// }

Eigen::Vector6s dAdInvT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F)
{
  Eigen::Vector6s res;
  res.tail<3>().noalias() = _T.linear() * _F.tail<3>();
  res.head<3>().noalias() = _T.linear() * _F.head<3>();
  res.head<3>() += _T.translation().cross(res.tail<3>());
  return res;
}

Eigen::Vector6s dAdInvR(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F)
{
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _F.head<3>();
  res.tail<3>().noalias() = _T.linear() * _F.tail<3>();
  return res;
}

// dse3 dAdInvPLinear(const Vec3& p, const Vec3& f)
// {
//     dse3 ret;

//     ret << p[1]*f[2] - p[2]*f[1],
//                 p[2]*f[0] - p[0]*f[2],
//                 p[0]*f[1] - p[1]*f[0],
//                 f[0],
//                 f[1],
//                 f[2];

//     return ret;
// }

/// Best effort attempt to find an equivalent set of euler angles that fits
/// within bounds
Eigen::Vector3s attemptToClampEulerAnglesToBounds(
    const Eigen::Vector3s& angle,
    const Eigen::Vector3s& upperBounds,
    const Eigen::Vector3s& lowerBounds,
    dynamics::detail::AxisOrder axisOrder)
{
  Eigen::Vector3s clampedAngle = angle;
  bool allClamped = true;
  for (int i = 0; i < 3; i++)
  {
    while (clampedAngle(i) > upperBounds(i))
    {
      clampedAngle(i) -= M_PI * 2;
    }
    while (clampedAngle(i) < lowerBounds(i))
    {
      clampedAngle(i) += M_PI * 2;
    }
    // Check if we successfully got this index in-bounds
    if ((clampedAngle(i) > upperBounds(i))
        || (clampedAngle(i) < lowerBounds(i)))
    {
      allClamped = false;
      break;
    }
  }
  // If we succeeded, return our attempt
  if (allClamped)
  {
    return clampedAngle;
  }

  // Try the equivalent strategy, where we flip the first axis, then recover
  // with 2nd and 3nd rotations
  clampedAngle
      = Eigen::Vector3s(M_PI + angle(0), M_PI - angle(1), M_PI + angle(2));

  // Try to clamp all the angles in the alternate formulation too
  allClamped = true;
  for (int i = 0; i < 3; i++)
  {
    while (clampedAngle(i) > upperBounds(i))
    {
      clampedAngle(i) -= M_PI * 2;
    }
    while (clampedAngle(i) < lowerBounds(i))
    {
      clampedAngle(i) += M_PI * 2;
    }
    // Check if we successfully got this index in-bounds
    if (clampedAngle(i) > upperBounds(i) || clampedAngle(i) < lowerBounds(i))
    {
      allClamped = false;
      break;
    }
  }
  // If we succeeded, return our attempt
  if (allClamped)
  {
    return clampedAngle;
  }

  // If we're rotating by PI/2 (90 degrees) mod PI (180 degrees), then the first
  // and 3rd axis are colinear.
  s_t secondAngle = angle(1);
  s_t secondSign = 1.0;
  while (secondAngle > M_PI / 2)
  {
    secondAngle -= M_PI;
    secondSign = -secondSign;
  }
  while (secondAngle < -M_PI / 2)
  {
    secondAngle += M_PI;
    secondSign = -secondSign;
  }
  if (abs((double)secondAngle - (M_PI / 2)) < 1e-4)
  {
    if (axisOrder == dynamics::detail::AxisOrder::ZYX
        || axisOrder == dynamics::detail::AxisOrder::ZYX)
    {
      secondSign = -secondSign;
    }

    // We can now say that angle(0) + angle(2) = clampedAngle(0) +
    // secondSign*clampedAngle(2)

    s_t c = angle(0) + angle(2);
    clampedAngle = angle;
    clampedAngle(0) = c / 2;
    clampedAngle(2) = secondSign * c / 2;

    // Try to clamp all the angles in the alternate formulation too
    allClamped = true;
    for (int i = 0; i < 3; i++)
    {
      while (clampedAngle(i) > upperBounds(i))
      {
        clampedAngle(i) -= M_PI * 2;
      }
      while (clampedAngle(i) < lowerBounds(i))
      {
        clampedAngle(i) += M_PI * 2;
      }
      // Check if we successfully got this index in-bounds
      if (clampedAngle(i) > upperBounds(i) || clampedAngle(i) < lowerBounds(i))
      {
        allClamped = false;
        break;
      }
    }
    // If we succeeded, return our attempt
    if (allClamped)
    {
      return clampedAngle;
    }
  }

  // If we still weren't able to clamp this, the angle probably isn't
  // reachable from the given bounds.
  return angle;
}

/// This will find an equivalent set of euler angles that is closest in joint
/// space to `previousAngle`
Eigen::Vector3s roundEulerAnglesToNearest(
    const Eigen::Vector3s& angle,
    const Eigen::Vector3s& previousAngle,
    dynamics::detail::AxisOrder axisOrder)
{
  (void)axisOrder;

  Eigen::Vector3s closestAngle = angle;

  // Check if we can add/subtract 2PI from any combination of angles to get a
  // closer value
  Eigen::Vector3s roundedAngle = angle;
  for (int i = 0; i < 3; i++)
  {
    s_t diff = angle(i) - previousAngle(i);
    s_t diffMultipleOf2Pi = round(diff / (2 * M_PI)) * 2 * M_PI;
    s_t smallestDiff = diff - diffMultipleOf2Pi;
    roundedAngle(i) = previousAngle(i) + smallestDiff;
  }
  if ((roundedAngle - previousAngle).norm()
      < (closestAngle - previousAngle).norm())
  {
    closestAngle = roundedAngle;
  }

  // Try the equivalent strategy, where we flip the first axis, then recover
  // with 2nd and 3nd rotations
  roundedAngle
      = Eigen::Vector3s(M_PI + angle(0), M_PI - angle(1), M_PI + angle(2));
  for (int i = 0; i < 3; i++)
  {
    s_t diff = angle(i) - previousAngle(i);
    s_t diffMultipleOf2Pi = round(diff / (2 * M_PI)) * 2 * M_PI;
    s_t smallestDiff = diff - diffMultipleOf2Pi;
    roundedAngle(i) = previousAngle(i) + smallestDiff;
  }
  if ((roundedAngle - previousAngle).norm()
      < (closestAngle - previousAngle).norm())
  {
    closestAngle = roundedAngle;
  }

  return closestAngle;
}

// Reference:
// http://www.geometrictools.com/LibMathematics/Algebra/Wm5Matrix3.inl
Eigen::Matrix3s eulerXYXToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  // | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  // | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret;

  s_t cx0 = cos(_angle(0));
  s_t sx0 = sin(_angle(0));
  s_t cy = cos(_angle(1));
  s_t sy = sin(_angle(1));
  s_t cx1 = cos(_angle(2));
  s_t sx1 = sin(_angle(2));

  ret(0, 0) = cy;
  ret(1, 0) = sy * sx0;
  ret(2, 0) = -sy * cx0;

  ret(0, 1) = sy * sx1;
  ret(1, 1) = cx0 * cx1 - cy * sx0 * sx1;
  ret(2, 1) = cx1 * sx0 + cy * cx0 * sx1;

  ret(0, 2) = sy * cx1;
  ret(1, 2) = -cy * cx1 * sx0 - cx0 * sx1;
  ret(2, 2) = cy * cx0 * cx1 - sx0 * sx1;

  return ret;
}

Eigen::Matrix3s eulerXYZToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle[0]);
  s_t sx = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cz = cos(_angle[2]);
  s_t sz = sin(_angle[2]);

  ret(0, 0) = cy * cz;
  ret(1, 0) = cx * sz + cz * sx * sy;
  ret(2, 0) = sx * sz - cx * cz * sy;

  ret(0, 1) = -cy * sz;
  ret(1, 1) = cx * cz - sx * sy * sz;
  ret(2, 1) = cz * sx + cx * sy * sz;

  ret(0, 2) = sy;
  ret(1, 2) = -cy * sx;
  ret(2, 2) = cx * cy;

  return ret;
}

/// This gives the gradient of an XYZ rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerXYZToMatrixGrad(const Eigen::Vector3s& _angle, int index)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle[0]);
  s_t sx = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cz = cos(_angle[2]);
  s_t sz = sin(_angle[2]);

  // dx
  //
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  0                0                0      |
  // | r10 r11 r12 | = |  cx*cz*sy-sx*sz   -sx*cz-cx*sy*sz  -cy*cx |
  // | r20 r21 r22 |   |  sx*cz*sy+cx*sz   cz*cx-sx*sy*sz   -sx*cy |
  // +-           -+   +-                                        -+

  if (index == 0)
  {
    ret(0, 0) = 0;
    ret(1, 0) = (-sx) * sz + cz * (cx)*sy;
    ret(2, 0) = cx * sz + sx * cz * sy;

    ret(0, 1) = 0;
    ret(1, 1) = -sx * cz - cx * sy * sz;
    ret(2, 1) = cz * cx - sx * sy * sz;

    ret(0, 2) = 0;
    ret(1, 2) = -cy * cx;
    ret(2, 2) = -sx * cy;
  }
  if (index == 1)
  {
    ret(0, 0) = (-sy) * cz;
    ret(1, 0) = cz * sx * (cy);
    ret(2, 0) = -cx * cz * (cy);

    ret(0, 1) = -(-sy) * sz;
    ret(1, 1) = -sx * (cy)*sz;
    ret(2, 1) = cx * (cy)*sz;

    ret(0, 2) = (cy);
    ret(1, 2) = -(-sy) * sx;
    ret(2, 2) = cx * (-sy);
  }
  if (index == 2)
  {
    ret(0, 0) = cy * (-sz);
    ret(1, 0) = cx * (cz) + (-sz) * sx * sy;
    ret(2, 0) = sx * (cz)-cx * (-sz) * sy;

    ret(0, 1) = -cy * (cz);
    ret(1, 1) = cx * (-sz) - sx * sy * (cz);
    ret(2, 1) = (-sz) * sx + cx * sy * (cz);

    ret(0, 2) = 0;
    ret(1, 2) = 0;
    ret(2, 2) = 0;
  }

  return ret;
}

/// This gives the gradient of an XYZ rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerXYZToMatrixFiniteDifference(
    const Eigen::Vector3s& _angle, int index)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(index) += eps;
        perturbed = eulerXYZToMatrix(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

/// This gives the gradient of eulerXYZToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerXYZToMatrixSecondGrad(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+
  (void)secondIndex;

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle[0]);
  s_t sx = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cz = cos(_angle[2]);
  s_t sz = sin(_angle[2]);

  // dx
  //
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  0                0                0      |
  // | r10 r11 r12 | = |  cx*cz*sy-sx*sz   -sx*cz-cx*sy*sz  -cy*cx |
  // | r20 r21 r22 |   |  sx*cz*sy+cx*sz   cz*cx-sx*sy*sz   -sx*cy |
  // +-           -+   +-                                        -+

  if (firstIndex == 0)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (-cx) * sz + cz * (-sx) * sy;
      ret(2, 0) = (-sx) * sz + (cx)*cz * sy;

      ret(0, 1) = 0;
      ret(1, 1) = -(cx)*cz - (-sx) * sy * sz;
      ret(2, 1) = cz * (-sx) - (cx)*sy * sz;

      ret(0, 2) = 0;
      ret(1, 2) = -cy * (-sx);
      ret(2, 2) = -(cx)*cy;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = 0;
      ret(1, 0) = cz * (cx) * (cy);
      ret(2, 0) = sx * cz * (cy);

      ret(0, 1) = 0;
      ret(1, 1) = -cx * (cy)*sz;
      ret(2, 1) = -sx * (cy)*sz;

      ret(0, 2) = 0;
      ret(1, 2) = -(-sy) * cx;
      ret(2, 2) = -sx * (-sy);
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (-sx) * (cz) + (-sz) * (cx)*sy;
      ret(2, 0) = cx * (cz) + sx * (-sz) * sy;

      ret(0, 1) = 0;
      ret(1, 1) = -sx * (-sz) - cx * sy * (cz);
      ret(2, 1) = (-sz) * cx - sx * sy * (cz);

      ret(0, 2) = 0;
      ret(1, 2) = 0;
      ret(2, 2) = 0;
    }
  }
  if (firstIndex == 1)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = cz * (cx) * (cy);
      ret(2, 0) = -(-sx) * cz * (cy);

      ret(0, 1) = 0;
      ret(1, 1) = -(cx) * (cy)*sz;
      ret(2, 1) = (-sx) * (cy)*sz;

      ret(0, 2) = 0;
      ret(1, 2) = -(-sy) * (cx);
      ret(2, 2) = (-sx) * (-sy);
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = (-(cy)) * cz;
      ret(1, 0) = cz * sx * ((-sy));
      ret(2, 0) = -cx * cz * ((-sy));

      ret(0, 1) = -(-(cy)) * sz;
      ret(1, 1) = -sx * ((-sy)) * sz;
      ret(2, 1) = cx * ((-sy)) * sz;

      ret(0, 2) = ((-sy));
      ret(1, 2) = -(-(cy)) * sx;
      ret(2, 2) = cx * (-(cy));
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = (-sy) * (-sz);
      ret(1, 0) = (-sz) * sx * (cy);
      ret(2, 0) = -cx * (-sz) * (cy);

      ret(0, 1) = -(-sy) * (cz);
      ret(1, 1) = -sx * (cy) * (cz);
      ret(2, 1) = cx * (cy) * (cz);

      ret(0, 2) = 0;
      ret(1, 2) = 0;
      ret(2, 2) = 0;
    }
  }
  if (firstIndex == 2)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (-sx) * (cz) + (-sz) * (cx)*sy;
      ret(2, 0) = (cx) * (cz) - (-sx) * (-sz) * sy;

      ret(0, 1) = 0;
      ret(1, 1) = (-sx) * (-sz) - (cx)*sy * (cz);
      ret(2, 1) = (-sz) * (cx) + (-sx) * sy * (cz);

      ret(0, 2) = 0;
      ret(1, 2) = 0;
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = (-sy) * (-sz);
      ret(1, 0) = (-sz) * sx * (cy);
      ret(2, 0) = -cx * (-sz) * (cy);

      ret(0, 1) = -(-sy) * (cz);
      ret(1, 1) = -sx * (cy) * (cz);
      ret(2, 1) = cx * (cy) * (cz);

      ret(0, 2) = 0;
      ret(1, 2) = 0;
      ret(2, 2) = 0;
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = cy * (-(cz));
      ret(1, 0) = cx * ((-sz)) + (-(cz)) * sx * sy;
      ret(2, 0) = sx * ((-sz)) - cx * (-(cz)) * sy;

      ret(0, 1) = -cy * ((-sz));
      ret(1, 1) = cx * (-(cz)) - sx * sy * ((-sz));
      ret(2, 1) = (-(cz)) * sx + cx * sy * ((-sz));

      ret(0, 2) = 0;
      ret(1, 2) = 0;
      ret(2, 2) = 0;
    }
  }

  return ret;
}

/// This gives the gradient of eulerXYZToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerXYZToMatrixSecondFiniteDifference(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(secondIndex) += eps;
        perturbed = eulerXYZToMatrixGrad(tweaked, firstIndex);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

Eigen::Matrix3s eulerXZXToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   | cz      -sz*cx1               sz*sx1             |
  // | r10 r11 r12 | = | sz*cx0   cz*cx0*cx1-sx0*sx1  -cx1*sx0-cz*cx0*sx1 |
  // | r20 r21 r22 |   | sz*sx0   cz*cx1*sx0+cx0*sx1   cx0*cx1-cz*sx0*sx1 |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret;

  s_t cx0 = cos(_angle(0));
  s_t sx0 = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cx1 = cos(_angle(2));
  s_t sx1 = sin(_angle(2));

  ret(0, 0) = cz;
  ret(1, 0) = sz * cx0;
  ret(2, 0) = sz * sx0;

  ret(0, 1) = -sz * cx1;
  ret(1, 1) = cz * cx0 * cx1 - sx0 * sx1;
  ret(2, 1) = cz * cx1 * sx0 + cx0 * sx1;

  ret(0, 2) = sz * sx1;
  ret(1, 2) = -cx1 * sx0 - cz * cx0 * sx1;
  ret(2, 2) = cx0 * cx1 - cz * sx0 * sx1;

  return ret;
}

Eigen::Matrix3s eulerXZYToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -sz      cz*sy          |
  // | r10 r11 r12 | = |  sx*sy+cx*cy*sz   cx*cz  -cy*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -cx*sy+cy*sx*sz   cz*sx   cx*cy+sx*sy*sz |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle(0));
  s_t sx = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  ret(0, 0) = cy * cz;
  ret(1, 0) = sx * sy + cx * cy * sz;
  ret(2, 0) = -cx * sy + cy * sx * sz;

  ret(0, 1) = -sz;
  ret(1, 1) = cx * cz;
  ret(2, 1) = cz * sx;

  ret(0, 2) = cz * sy;
  ret(1, 2) = -cy * sx + cx * sy * sz;
  ret(2, 2) = cx * cy + sx * sy * sz;

  return ret;
}

/// This gives the gradient of an XZY rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerXZYToMatrixGrad(const Eigen::Vector3s& _angle, int index)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -sz      cz*sy          |
  // | r10 r11 r12 | = |  sx*sy+cx*cy*sz   cx*cz  -cy*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -cx*sy+cy*sx*sz   cz*sx   cx*cy+sx*sy*sz |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle(0));
  s_t sx = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  if (index == 0)
  {
    ret(0, 0) = 0;
    ret(1, 0) = (cx)*sy + (-sx) * cy * sz;
    ret(2, 0) = -(-sx) * sy + cy * (cx)*sz;

    ret(0, 1) = 0;
    ret(1, 1) = (-sx) * cz;
    ret(2, 1) = cz * (cx);

    ret(0, 2) = 0;
    ret(1, 2) = -cy * (cx) + (-sx) * sy * sz;
    ret(2, 2) = (-sx) * cy + (cx)*sy * sz;
  }
  else if (index == 1)
  {
    ret(0, 0) = cy * (-sz);
    ret(1, 0) = cx * cy * (cz);
    ret(2, 0) = cy * sx * (cz);

    ret(0, 1) = -(cz);
    ret(1, 1) = cx * (-sz);
    ret(2, 1) = (-sz) * sx;

    ret(0, 2) = (-sz) * sy;
    ret(1, 2) = cx * sy * (cz);
    ret(2, 2) = sx * sy * (cz);
  }
  else if (index == 2)
  {
    ret(0, 0) = (-sy) * cz;
    ret(1, 0) = sx * (cy) + cx * (-sy) * sz;
    ret(2, 0) = -cx * (cy) + (-sy) * sx * sz;

    ret(0, 1) = 0;
    ret(1, 1) = 0;
    ret(2, 1) = 0;

    ret(0, 2) = cz * (cy);
    ret(1, 2) = -(-sy) * sx + cx * (cy)*sz;
    ret(2, 2) = cx * (-sy) + sx * (cy)*sz;
  }
  else
  {
    assert(false);
  }

  return ret;
}

/// This gives the gradient of an XZY rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerXZYToMatrixFiniteDifference(
    const Eigen::Vector3s& _angle, int index)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(index) += eps;
        perturbed = eulerXZYToMatrix(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

/// This gives the gradient of eulerXZYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerXZYToMatrixSecondGrad(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -sz      cz*sy          |
  // | r10 r11 r12 | = |  sx*sy+cx*cy*sz   cx*cz  -cy*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -cx*sy+cy*sx*sz   cz*sx   cx*cy+sx*sy*sz |
  // +-           -+   +-                                        -+
  (void)secondIndex;

  Eigen::Matrix3s ret;

  s_t cx = cos(_angle(0));
  s_t sx = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  // dx
  //
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  0                0                0      |
  // | r10 r11 r12 | = |  cx*cz*sy-sx*sz   -sx*cz-cx*sy*sz  -cy*cx |
  // | r20 r21 r22 |   |  sx*cz*sy+cx*sz   cz*cx-sx*sy*sz   -sx*cy |
  // +-           -+   +-                                        -+

  if (firstIndex == 0)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = ((-sx)) * sy + (-(cx)) * cy * sz;
      ret(2, 0) = -(-(cx)) * sy + cy * ((-sx)) * sz;

      ret(0, 1) = 0;
      ret(1, 1) = (-(cx)) * cz;
      ret(2, 1) = cz * ((-sx));

      ret(0, 2) = 0;
      ret(1, 2) = -cy * ((-sx)) + (-(cx)) * sy * sz;
      ret(2, 2) = (-(cx)) * cy + ((-sx)) * sy * sz;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (-sx) * cy * (cz);
      ret(2, 0) = cy * (cx) * (cz);

      ret(0, 1) = 0;
      ret(1, 1) = (-sx) * (-sz);
      ret(2, 1) = (-sz) * (cx);

      ret(0, 2) = 0;
      ret(1, 2) = (-sx) * sy * (cz);
      ret(2, 2) = (cx)*sy * (cz);
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (cx) * (cy) + (-sx) * (-sy) * sz;
      ret(2, 0) = -(-sx) * (cy) + (-sy) * (cx)*sz;

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = 0;
      ret(1, 2) = -(-sy) * (cx) + (-sx) * (cy)*sz;
      ret(2, 2) = (-sx) * (-sy) + (cx) * (cy)*sz;
    }
  }
  if (firstIndex == 1)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (-sx) * cy * (cz);
      ret(2, 0) = cy * (cx) * (cz);

      ret(0, 1) = 0;
      ret(1, 1) = (-sx) * (-sz);
      ret(2, 1) = (-sz) * (cx);

      ret(0, 2) = 0;
      ret(1, 2) = (-sx) * sy * (cz);
      ret(2, 2) = (cx)*sy * (cz);
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = cy * (-(cz));
      ret(1, 0) = cx * cy * ((-sz));
      ret(2, 0) = cy * sx * ((-sz));

      ret(0, 1) = -((-sz));
      ret(1, 1) = cx * (-(cz));
      ret(2, 1) = (-(cz)) * sx;

      ret(0, 2) = (-(cz)) * sy;
      ret(1, 2) = cx * sy * ((-sz));
      ret(2, 2) = sx * sy * ((-sz));
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = (-sy) * (-sz);
      ret(1, 0) = cx * (-sy) * (cz);
      ret(2, 0) = (-sy) * sx * (cz);

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy);
      ret(1, 2) = cx * (cy) * (cz);
      ret(2, 2) = sx * (cy) * (cz);
    }
  }
  if (firstIndex == 2)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = (cx) * (cy) + (-sx) * (-sy) * sz;
      ret(2, 0) = -(-sx) * (cy) + (-sy) * (cx)*sz;

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = 0;
      ret(1, 2) = -(-sy) * (cx) + (-sx) * (cy)*sz;
      ret(2, 2) = (-sx) * (-sy) + (cx) * (cy)*sz;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = (-sy) * (-sz);
      ret(1, 0) = cx * (-sy) * (cz);
      ret(2, 0) = (-sy) * sx * (cz);

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy);
      ret(1, 2) = cx * (cy) * (cz);
      ret(2, 2) = sx * (cy) * (cz);
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = (-(cy)) * cz;
      ret(1, 0) = sx * ((-sy)) + cx * (-(cy)) * sz;
      ret(2, 0) = -cx * ((-sy)) + (-(cy)) * sx * sz;

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = cz * ((-sy));
      ret(1, 2) = -(-(cy)) * sx + cx * ((-sy)) * sz;
      ret(2, 2) = cx * (-(cy)) + sx * ((-sy)) * sz;
    }
  }

  return ret;
}

/// This gives the gradient of eulerXZYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerXZYToMatrixSecondFiniteDifference(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(secondIndex) += eps;
        perturbed = eulerXZYToMatrixGrad(tweaked, firstIndex);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

Eigen::Matrix3s eulerYXYToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   |  cy0*cy1-cx*sy0*sy1  sx*sy0   cx*cy1*sy0+cy0*sy1 |
  // | r10 r11 r12 | = |  sx*sy1              cx      -sx*cy1             |
  // | r20 r21 r22 |   | -cy1*sy0-cx*cy0*sy1  sx*cy0   cx*cy0*cy1-sy0*sy1 |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret;

  s_t cy0 = cos(_angle(0));
  s_t sy0 = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cy1 = cos(_angle(2));
  s_t sy1 = sin(_angle(2));

  ret(0, 0) = cy0 * cy1 - cx * sy0 * sy1;
  ret(1, 0) = sx * sy1;
  ret(2, 0) = -cy1 * sy0 - cx * cy0 * sy1;

  ret(0, 1) = sx * sy0;
  ret(1, 1) = cx;
  ret(2, 1) = sx * cy0;

  ret(0, 2) = cx * cy1 * sy0 + cy0 * sy1;
  ret(1, 2) = -sx * cy1;
  ret(2, 2) = cx * cy0 * cy1 - sy0 * sy1;

  return ret;
}

Eigen::Matrix3s eulerYXZToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                       -+
  // | r00 r01 r02 |   |  cy*cz+sx*sy*sz  cz*sx*sy-cy*sz   cx*sy |
  // | r10 r11 r12 | = |  cx*sz           cx*cz           -sx    |
  // | r20 r21 r22 |   | -cz*sy+cy*sx*sz  cy*cz*sx+sy*sz   cx*cy |
  // +-           -+   +-                                       -+

  Eigen::Matrix3s ret;

  s_t cy = cos(_angle(0));
  s_t sy = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cz = cos(_angle(2));
  s_t sz = sin(_angle(2));

  ret(0, 0) = cy * cz + sx * sy * sz;
  ret(0, 1) = cz * sx * sy - cy * sz;
  ret(0, 2) = cx * sy;
  ret(1, 0) = cx * sz;
  ret(1, 1) = cx * cz;
  ret(1, 2) = -sx;
  ret(2, 0) = -cz * sy + cy * sx * sz;
  ret(2, 1) = cy * cz * sx + sy * sz;
  ret(2, 2) = cx * cy;

  return ret;
}

Eigen::Matrix3s eulerYZXToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                       -+
  // | r00 r01 r02 |   |  cy*cz  sx*sy-cx*cy*sz   cx*sy+cy*sx*sz |
  // | r10 r11 r12 | = |  sz     cx*cz           -cz*sx          |
  // | r20 r21 r22 |   | -cz*sy  cy*sx+cx*sy*sz   cx*cy-sx*sy*sz |
  // +-           -+   +-                                       -+

  Eigen::Matrix3s ret;

  s_t cy = cos(_angle(0));
  s_t sy = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cx = cos(_angle(2));
  s_t sx = sin(_angle(2));

  ret(0, 0) = cy * cz;
  ret(0, 1) = sx * sy - cx * cy * sz;
  ret(0, 2) = cx * sy + cy * sx * sz;
  ret(1, 0) = sz;
  ret(1, 1) = cx * cz;
  ret(1, 2) = -cz * sx;
  ret(2, 0) = -cz * sy;
  ret(2, 1) = cy * sx + cx * sy * sz;
  ret(2, 2) = cx * cy - sx * sy * sz;

  return ret;
}

Eigen::Matrix3s eulerYZYToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   |  cz*cy0*cy1-sy0*sy1  -sz*cy0  cy1*sy0+cz*cy0*sy1 |
  // | r10 r11 r12 | = |  sz*cy1               cz      sz*sy1             |
  // | r20 r21 r22 |   | -cz*cy1*sy0-cy0*sy1   sz*sy0  cy0*cy1-cz*sy0*sy1 |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret;

  s_t cy0 = cos(_angle(0));
  s_t sy0 = sin(_angle(0));
  s_t cz = cos(_angle(1));
  s_t sz = sin(_angle(1));
  s_t cy1 = cos(_angle(2));
  s_t sy1 = sin(_angle(2));

  ret(0, 0) = cz * cy0 * cy1 - sy0 * sy1;
  ret(1, 0) = sz * cy1;
  ret(2, 0) = -cz * cy1 * sy0 - cy0 * sy1;

  ret(0, 1) = -sz * cy0;
  ret(1, 1) = cz;
  ret(2, 1) = sz * sy0;

  ret(0, 2) = cy1 * sy0 + cz * cy0 * sy1;
  ret(1, 2) = sz * sy1;
  ret(2, 2) = cy0 * cy1 - cz * sy0 * sy1;

  return ret;
}

Eigen::Matrix3s eulerZXYToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz-sx*sy*sz  -cx*sz   cz*sy+cy*sx*sz |
  // | r10 r11 r12 | = |  cz*sx*sy+cy*sz   cx*cz  -cy*cz*sx+sy*sz |
  // | r20 r21 r22 |   | -cx*sy            sx      cx*cy          |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle(0));
  s_t sz = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  ret(0, 0) = cy * cz - sx * sy * sz;
  ret(1, 0) = cz * sx * sy + cy * sz;
  ret(2, 0) = -cx * sy;

  ret(0, 1) = -cx * sz;
  ret(1, 1) = cx * cz;
  ret(2, 1) = sx;

  ret(0, 2) = cz * sy + cy * sx * sz;
  ret(1, 2) = -cy * cz * sx + sy * sz;
  ret(2, 2) = cx * cy;

  return ret;
}

/// This gives the gradient of an ZXY rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerZXYToMatrixGrad(const Eigen::Vector3s& _angle, int index)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz-sx*sy*sz  -cx*sz   cz*sy+cy*sx*sz |
  // | r10 r11 r12 | = |  cz*sx*sy+cy*sz   cx*cz  -cy*cz*sx+sy*sz |
  // | r20 r21 r22 |   | -cx*sy            sx      cx*cy          |
  // +-           -+   +-                                        -+

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle(0));
  s_t sz = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  if (index == 0)
  {
    ret(0, 0) = cy * (-sz) - sx * sy * (cz);
    ret(1, 0) = (-sz) * sx * sy + cy * (cz);
    ret(2, 0) = 0;

    ret(0, 1) = -cx * (cz);
    ret(1, 1) = cx * (-sz);
    ret(2, 1) = 0;

    ret(0, 2) = (-sz) * sy + cy * sx * (cz);
    ret(1, 2) = -cy * (-sz) * sx + sy * (cz);
    ret(2, 2) = 0;
  }
  if (index == 1)
  {
    ret(0, 0) = -(cx)*sy * sz;
    ret(1, 0) = cz * (cx)*sy;
    ret(2, 0) = -(-sx) * sy;

    ret(0, 1) = -(-sx) * sz;
    ret(1, 1) = (-sx) * cz;
    ret(2, 1) = (cx);

    ret(0, 2) = cy * (cx)*sz;
    ret(1, 2) = -cy * cz * (cx);
    ret(2, 2) = (-sx) * cy;
  }
  if (index == 2)
  {
    ret(0, 0) = (-sy) * cz - sx * (cy)*sz;
    ret(1, 0) = cz * sx * (cy) + (-sy) * sz;
    ret(2, 0) = -cx * (cy);

    ret(0, 1) = 0;
    ret(1, 1) = 0;
    ret(2, 1) = 0;

    ret(0, 2) = cz * (cy) + (-sy) * sx * sz;
    ret(1, 2) = -(-sy) * cz * sx + (cy)*sz;
    ret(2, 2) = cx * (-sy);
  }

  return ret;
}

/// This gives the gradient of an ZXY rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerZXYToMatrixFiniteDifference(
    const Eigen::Vector3s& _angle, int index)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(index) += eps;
        perturbed = eulerZXYToMatrix(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

/// This gives the gradient of eulerXZYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerZXYToMatrixSecondGrad(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  // Original
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -sz      cz*sy          |
  // | r10 r11 r12 | = |  sx*sy+cx*cy*sz   cx*cz  -cy*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -cx*sy+cy*sx*sz   cz*sx   cx*cy+sx*sy*sz |
  // +-           -+   +-                                        -+
  (void)secondIndex;

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle(0));
  s_t sz = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cy = cos(_angle(2));
  s_t sy = sin(_angle(2));

  // dx
  //
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  0                0                0      |
  // | r10 r11 r12 | = |  cx*cz*sy-sx*sz   -sx*cz-cx*sy*sz  -cy*cx |
  // | r20 r21 r22 |   |  sx*cz*sy+cx*sz   cz*cx-sx*sy*sz   -sx*cy |
  // +-           -+   +-                                        -+

  if (firstIndex == 0)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = cy * (-(cz)) + -(sx * sy * ((-sz)));
      ret(1, 0) = (-(cz)) * sx * sy + cy * ((-sz));
      ret(2, 0) = 0;

      ret(0, 1) = -(cx * ((-sz)));
      ret(1, 1) = cx * (-(cz));
      ret(2, 1) = 0;

      ret(0, 2) = (-(cz)) * sy + cy * sx * ((-sz));
      ret(1, 2) = -cy * (-(cz)) * sx + sy * ((-sz));
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = -((cx)*sy * (cz));
      ret(1, 0) = (-sz) * (cx)*sy;
      ret(2, 0) = 0;

      ret(0, 1) = -((-sx) * (cz));
      ret(1, 1) = (-sx) * (-sz);
      ret(2, 1) = 0;

      ret(0, 2) = cy * (cx) * (cz);
      ret(1, 2) = -cy * (-sz) * (cx);
      ret(2, 2) = 0;
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = (-sy) * (-sz) + -(sx * (cy) * (cz));
      ret(1, 0) = (-sz) * sx * (cy) + (-sy) * (cz);
      ret(2, 0) = 0;

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy) + (-sy) * sx * (cz);
      ret(1, 2) = -(-sy) * (-sz) * sx + (cy) * (cz);
      ret(2, 2) = 0;
    }
  }
  if (firstIndex == 1)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = -((cx)*sy * (cz));
      ret(1, 0) = (-sz) * (cx)*sy;
      ret(2, 0) = 0;

      ret(0, 1) = -((-sx) * (cz));
      ret(1, 1) = (-sx) * (-sz);
      ret(2, 1) = 0;

      ret(0, 2) = cy * (cx) * (cz);
      ret(1, 2) = -cy * (-sz) * (cx);
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = -(((-sx)) * sy * sz);
      ret(1, 0) = cz * ((-sx)) * sy;
      ret(2, 0) = -(-(cx)) * sy;

      ret(0, 1) = -((-(cx)) * sz);
      ret(1, 1) = (-(cx)) * cz;
      ret(2, 1) = ((-sx));

      ret(0, 2) = cy * ((-sx)) * sz;
      ret(1, 2) = -cy * cz * ((-sx));
      ret(2, 2) = (-(cx)) * cy;
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = -((cx) * (cy)*sz);
      ret(1, 0) = cz * (cx) * (cy);
      ret(2, 0) = -(-sx) * (cy);

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sy) * (cx)*sz;
      ret(1, 2) = -(-sy) * cz * (cx);
      ret(2, 2) = (-sx) * (-sy);
    }
  }
  if (firstIndex == 2)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = (-sy) * (-sz) + -(sx * (cy) * (cz));
      ret(1, 0) = (-sz) * sx * (cy) + (-sy) * (cz);
      ret(2, 0) = 0;

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy) + (-sy) * sx * (cz);
      ret(1, 2) = -(-sy) * (-sz) * sx + (cy) * (cz);
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = -((cx) * (cy)*sz);
      ret(1, 0) = cz * (cx) * (cy);
      ret(2, 0) = -(-sx) * (cy);

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = (-sy) * (cx)*sz;
      ret(1, 2) = -(-sy) * cz * (cx);
      ret(2, 2) = (-sx) * (-sy);
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = (-(cy)) * cz + -(sx * ((-sy)) * sz);
      ret(1, 0) = cz * sx * ((-sy)) + (-(cy)) * sz;
      ret(2, 0) = -cx * ((-sy));

      ret(0, 1) = 0;
      ret(1, 1) = 0;
      ret(2, 1) = 0;

      ret(0, 2) = cz * ((-sy)) + (-(cy)) * sx * sz;
      ret(1, 2) = -(-(cy)) * cz * sx + ((-sy)) * sz;
      ret(2, 2) = cx * (-(cy));
    }
  }

  return ret;
}

/// This gives the gradient of eulerZXYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerZXYToMatrixSecondFiniteDifference(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(secondIndex) += eps;
        perturbed = eulerZXYToMatrixGrad(tweaked, firstIndex);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

Eigen::Matrix3s eulerZYXToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                      -+
  // | r00 r01 r02 |   |  cy*cz  cz*sx*sy-cx*sz  cx*cz*sy+sx*sz |
  // | r10 r11 r12 | = |  cy*sz  cx*cz+sx*sy*sz -cz*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -sy     cy*sx           cx*cy          |
  // +-           -+   +-                                      -+

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle[0]);
  s_t sz = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cx = cos(_angle[2]);
  s_t sx = sin(_angle[2]);

  ret(0, 0) = cz * cy;
  ret(1, 0) = sz * cy;
  ret(2, 0) = -sy;

  ret(0, 1) = cz * sy * sx - sz * cx;
  ret(1, 1) = sz * sy * sx + cz * cx;
  ret(2, 1) = cy * sx;

  ret(0, 2) = cz * sy * cx + sz * sx;
  ret(1, 2) = sz * sy * cx - cz * sx;
  ret(2, 2) = cy * cx;

  return ret;
}

/// This gives the gradient of an ZYX rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerZYXToMatrixGrad(const Eigen::Vector3s& _angle, int index)
{
  // Original
  // +-           -+   +-                                      -+
  // | r00 r01 r02 |   |  cy*cz  cz*sx*sy-cx*sz  cx*cz*sy+sx*sz |
  // | r10 r11 r12 | = |  cy*sz  cx*cz+sx*sy*sz -cz*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -sy     cy*sx           cx*cy          |
  // +-           -+   +-                                      -+

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle[0]);
  s_t sz = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cx = cos(_angle[2]);
  s_t sx = sin(_angle[2]);

  if (index == 0)
  {
    ret(0, 0) = (-sz) * cy;
    ret(1, 0) = (cz)*cy;
    ret(2, 0) = 0;

    ret(0, 1) = (-sz) * sy * sx - (cz)*cx;
    ret(1, 1) = (cz)*sy * sx + (-sz) * cx;
    ret(2, 1) = 0;

    ret(0, 2) = (-sz) * sy * cx + (cz)*sx;
    ret(1, 2) = (cz)*sy * cx - (-sz) * sx;
    ret(2, 2) = 0;
  }
  if (index == 1)
  {
    ret(0, 0) = cz * (-sy);
    ret(1, 0) = sz * (-sy);
    ret(2, 0) = -(cy);

    ret(0, 1) = cz * (cy)*sx;
    ret(1, 1) = sz * (cy)*sx;
    ret(2, 1) = (-sy) * sx;

    ret(0, 2) = cz * (cy)*cx;
    ret(1, 2) = sz * (cy)*cx;
    ret(2, 2) = (-sy) * cx;
  }
  if (index == 2)
  {
    ret(0, 0) = 0;
    ret(1, 0) = 0;
    ret(2, 0) = 0;

    ret(0, 1) = cz * sy * (cx)-sz * (-sx);
    ret(1, 1) = sz * sy * (cx) + cz * (-sx);
    ret(2, 1) = cy * (cx);

    ret(0, 2) = cz * sy * (-sx) + sz * (cx);
    ret(1, 2) = sz * sy * (-sx) - cz * (cx);
    ret(2, 2) = cy * (-sx);
  }

  return ret;
}

/// This gives the gradient of an ZYX rotation matrix with respect to the
/// specific index (0, 1, or 2)
Eigen::Matrix3s eulerZYXToMatrixFiniteDifference(
    const Eigen::Vector3s& _angle, int index)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(index) += eps;
        perturbed = eulerZYXToMatrix(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

/// This gives the gradient of eulerXZYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerZYXToMatrixSecondGrad(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  // Original
  // +-           -+   +-                                      -+
  // | r00 r01 r02 |   |  cy*cz  cz*sx*sy-cx*sz  cx*cz*sy+sx*sz |
  // | r10 r11 r12 | = |  cy*sz  cx*cz+sx*sy*sz -cz*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -sy     cy*sx           cx*cy          |
  // +-           -+   +-                                      -+
  (void)secondIndex;

  Eigen::Matrix3s ret;

  s_t cz = cos(_angle[0]);
  s_t sz = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cx = cos(_angle[2]);
  s_t sx = sin(_angle[2]);

  if (firstIndex == 0)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = (-(cz)) * cy;
      ret(1, 0) = ((-sz)) * cy;
      ret(2, 0) = 0;

      ret(0, 1) = (-(cz)) * sy * sx + -(((-sz)) * cx);
      ret(1, 1) = ((-sz)) * sy * sx + (-(cz)) * cx;
      ret(2, 1) = 0;

      ret(0, 2) = (-(cz)) * sy * cx + ((-sz)) * sx;
      ret(1, 2) = ((-sz)) * sy * cx + -((-(cz)) * sx);
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = (-sz) * (-sy);
      ret(1, 0) = (cz) * (-sy);
      ret(2, 0) = 0;

      ret(0, 1) = (-sz) * (cy)*sx;
      ret(1, 1) = (cz) * (cy)*sx;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy)*cx;
      ret(1, 2) = (cz) * (cy)*cx;
      ret(2, 2) = 0;
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = 0;
      ret(1, 0) = 0;
      ret(2, 0) = 0;

      ret(0, 1) = (-sz) * sy * (cx) + -((cz) * (-sx));
      ret(1, 1) = (cz)*sy * (cx) + (-sz) * (-sx);
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * sy * (-sx) + (cz) * (cx);
      ret(1, 2) = (cz)*sy * (-sx) + -((-sz) * (cx));
      ret(2, 2) = 0;
    }
  }
  if (firstIndex == 1)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = (-sz) * (-sy);
      ret(1, 0) = (cz) * (-sy);
      ret(2, 0) = 0;

      ret(0, 1) = (-sz) * (cy)*sx;
      ret(1, 1) = (cz) * (cy)*sx;
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * (cy)*cx;
      ret(1, 2) = (cz) * (cy)*cx;
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = cz * (-(cy));
      ret(1, 0) = sz * (-(cy));
      ret(2, 0) = -((-sy));

      ret(0, 1) = cz * ((-sy)) * sx;
      ret(1, 1) = sz * ((-sy)) * sx;
      ret(2, 1) = (-(cy)) * sx;

      ret(0, 2) = cz * ((-sy)) * cx;
      ret(1, 2) = sz * ((-sy)) * cx;
      ret(2, 2) = (-(cy)) * cx;
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = 0;
      ret(1, 0) = 0;
      ret(2, 0) = 0;

      ret(0, 1) = cz * (cy) * (cx);
      ret(1, 1) = sz * (cy) * (cx);
      ret(2, 1) = (-sy) * (cx);

      ret(0, 2) = cz * (cy) * (-sx);
      ret(1, 2) = sz * (cy) * (-sx);
      ret(2, 2) = (-sy) * (-sx);
    }
  }
  if (firstIndex == 2)
  {
    if (secondIndex == 0)
    {
      ret(0, 0) = 0;
      ret(1, 0) = 0;
      ret(2, 0) = 0;

      ret(0, 1) = (-sz) * sy * (cx) + -((cz) * (-sx));
      ret(1, 1) = (cz)*sy * (cx) + (-sz) * (-sx);
      ret(2, 1) = 0;

      ret(0, 2) = (-sz) * sy * (-sx) + (cz) * (cx);
      ret(1, 2) = (cz)*sy * (-sx) + -((-sz) * (cx));
      ret(2, 2) = 0;
    }
    else if (secondIndex == 1)
    {
      ret(0, 0) = 0;
      ret(1, 0) = 0;
      ret(2, 0) = 0;

      ret(0, 1) = cz * (cy) * (cx);
      ret(1, 1) = sz * (cy) * (cx);
      ret(2, 1) = (-sy) * (cx);

      ret(0, 2) = cz * (cy) * (-sx);
      ret(1, 2) = sz * (cy) * (-sx);
      ret(2, 2) = (-sy) * (-sx);
    }
    else if (secondIndex == 2)
    {
      ret(0, 0) = 0;
      ret(1, 0) = 0;
      ret(2, 0) = 0;

      ret(0, 1) = cz * sy * ((-sx)) + -(sz * (-(cx)));
      ret(1, 1) = sz * sy * ((-sx)) + cz * (-(cx));
      ret(2, 1) = cy * ((-sx));

      ret(0, 2) = cz * sy * (-(cx)) + sz * ((-sx));
      ret(1, 2) = sz * sy * (-(cx)) + -(cz * ((-sx)));
      ret(2, 2) = cy * (-(cx));
    }
  }

  return ret;
}

/// This gives the gradient of eulerXZYToMatrixGrad(_angle, firstIndex) with
/// respect to secondIndex.
Eigen::Matrix3s eulerZYXToMatrixSecondFiniteDifference(
    const Eigen::Vector3s& _angle, int firstIndex, int secondIndex)
{
  Eigen::Matrix3s result;

  bool useRidders = false;
  s_t eps = 1e-8;
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        Eigen::Vector3s tweaked = _angle;
        tweaked(secondIndex) += eps;
        perturbed = eulerZYXToMatrixGrad(tweaked, firstIndex);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

Eigen::Matrix3s eulerZXZToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   | cz0*cz1-cx*sz0*sz1  -cx*cz1*sz0-cz0*sz1   sx*sz0 |
  // | r10 r11 r12 | = | cz1*sz0+cx*cz0*sz1   cx*cz0*cz1-sz0*sz1  -sz*cz0 |
  // | r20 r21 r22 |   | sx*sz1               sx*cz1               cx     |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret;

  s_t cz0 = cos(_angle(0));
  s_t sz0 = sin(_angle(0));
  s_t cx = cos(_angle(1));
  s_t sx = sin(_angle(1));
  s_t cz1 = cos(_angle(2));
  s_t sz1 = sin(_angle(2));

  ret(0, 0) = cz0 * cz1 - cx * sz0 * sz1;
  ret(1, 0) = cz1 * sz0 + cx * cz0 * sz1;
  ret(2, 0) = sx * sz1;

  ret(0, 1) = -cx * cz1 * sz0 - cz0 * sz1;
  ret(1, 1) = cx * cz0 * cz1 - sz0 * sz1;
  ret(2, 1) = sx * cz1;

  ret(0, 2) = sx * sz0;
  ret(1, 2) = -sx * cz0;
  ret(2, 2) = cx;

  return ret;
}

Eigen::Matrix3s eulerZYZToMatrix(const Eigen::Vector3s& _angle)
{
  // +-           -+   +-                                                -+
  // | r00 r01 r02 |   |  cy*cz0*cz1-sz0*sz1  -cz1*sz0-cy*cz0*sz1  sy*cz0 |
  // | r10 r11 r12 | = |  cy*cz1*sz0+cz0*sz1   cz0*cz1-cy*sz0*sz1  sy*sz0 |
  // | r20 r21 r22 |   | -sy*cz1               sy*sz1              cy     |
  // +-           -+   +-                                                -+

  Eigen::Matrix3s ret = Eigen::Matrix3s::Identity();

  s_t cz0 = cos(_angle[0]);
  s_t sz0 = sin(_angle[0]);
  s_t cy = cos(_angle[1]);
  s_t sy = sin(_angle[1]);
  s_t cz1 = cos(_angle[2]);
  s_t sz1 = sin(_angle[2]);

  ret(0, 0) = cz0 * cy * cz1 - sz0 * sz1;
  ret(1, 0) = sz0 * cy * cz1 + cz0 * sz1;
  ret(2, 0) = -sy * cz1;

  ret(0, 1) = -cz0 * cy * sz1 - sz0 * cz1;
  ret(1, 1) = cz0 * cz1 - sz0 * cy * sz1;
  ret(2, 1) = sy * sz1;

  ret(0, 2) = cz0 * sy;
  ret(1, 2) = sz0 * sy;
  ret(2, 2) = cy;

  return ret;
}

//==============================================================================
Eigen::Matrix3s so3LeftJacobian(const Eigen::Vector3s& w)
{
  return expMapJac(w);
}

//==============================================================================
Eigen::Matrix3s so3RightJacobian(const Eigen::Vector3s& w)
{
  const s_t theta = w.norm();

  Eigen::Matrix3s J = Eigen::Matrix3s::Zero();
  const Eigen::Matrix3s qss = math::makeSkewSymmetric(w);
  const Eigen::Matrix3s qss2 = qss * qss;

  if (theta < EPSILON_EXPMAP_THETA)
    J = Eigen::Matrix3s::Identity() - 0.5 * qss + (1.0 / 6.0) * qss2;
  else
    J = Eigen::Matrix3s::Identity() - ((1 - cos(theta)) / (theta * theta)) * qss
        + ((theta - sin(theta)) / (theta * theta * theta)) * qss2;

  return J;
}

//==============================================================================
Eigen::Matrix3s so3LeftJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq)
{
  return expMapJacDot(q, dq);
}

//==============================================================================
Eigen::Matrix3s so3RightJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq)
{
  const s_t theta = q.norm();

  Eigen::Matrix3s Jdot = Eigen::Matrix3s::Zero();
  const Eigen::Matrix3s qss = math::makeSkewSymmetric(q);
  const Eigen::Matrix3s qss2 = qss * qss;
  const Eigen::Matrix3s qdss = math::makeSkewSymmetric(dq);
  const s_t ttdot = q.dot(dq); // theta*thetaDot
  const s_t st = sin(theta);
  const s_t ct = cos(theta);
  const s_t t2 = theta * theta;
  const s_t t3 = t2 * theta;
  const s_t t4 = t3 * theta;
  const s_t t5 = t4 * theta;

  if (theta < EPSILON_EXPMAP_THETA)
  {
    Jdot = -0.5 * qdss + (1.0 / 6.0) * (qss * qdss + qdss * qss);
    Jdot += (1.0 / 12) * ttdot * qss + (-1.0 / 60) * ttdot * qss2;
  }
  else
  {
    Jdot = -((1 - ct) / t2) * qdss
           + ((theta - st) / t3) * (qss * qdss + qdss * qss);
    Jdot += -((theta * st + 2 * ct - 2) / t4) * ttdot * qss
            + ((3 * st - theta * ct - 2 * theta) / t5) * ttdot * qss2;
  }

  return Jdot;
}

//==============================================================================
Eigen::Matrix3s so3RightJacobianTimeDerivDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq, int index)
{
  // TODO(JS): Relplace with analytical method

  const s_t eps = 1e-7;

  Eigen::Vector3s perterb = q;
  perterb[index] += eps;
  const Eigen::Matrix3s mat1 = so3RightJacobianTimeDeriv(perterb, dq);

  perterb = q;
  perterb[index] -= eps;
  const Eigen::Matrix3s mat2 = so3RightJacobianTimeDeriv(perterb, dq);

  return (mat1 - mat2) / (eps * 2);
}

//==============================================================================
Eigen::Matrix3s so3RightJacobianTimeDerivDeriv2(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq, int index)
{
  // TODO(JS): Relplace with analytical method

  const s_t eps = 1e-7;

  Eigen::Vector3s perterb = dq;
  perterb[index] += eps;
  const Eigen::Matrix3s mat1 = so3RightJacobianTimeDeriv(q, perterb);

  perterb = dq;
  perterb[index] -= eps;
  const Eigen::Matrix3s mat2 = so3RightJacobianTimeDeriv(q, perterb);

  return (mat1 - mat2) / (eps * 2);
}

// R = Exp(w)
// p = sin(t) / t*v + (t - sin(t)) / t^3*<w, v>*w + (1 - cos(t)) / t^2*(w X v)
// , when S = (w, v), t = |w|
Eigen::Isometry3s expMap(const Eigen::Vector6s& _S)
{
  Eigen::Isometry3s ret = Eigen::Isometry3s::Identity();
  s_t s2[] = {_S[0] * _S[0], _S[1] * _S[1], _S[2] * _S[2]};
  s_t s3[] = {_S[0] * _S[1], _S[1] * _S[2], _S[2] * _S[0]};
  s_t theta = sqrt(s2[0] + s2[1] + s2[2]);
  s_t cos_t = cos(theta), alpha, beta, gamma;

  if (theta > DART_EPSILON)
  {
    s_t sin_t = sin(theta);
    alpha = sin_t / theta;
    beta = (1.0 - cos_t) / theta / theta;
    gamma = (_S[0] * _S[3] + _S[1] * _S[4] + _S[2] * _S[5]) * (theta - sin_t)
            / theta / theta / theta;
  }
  else
  {
    alpha = 1.0 - theta * theta / 6.0;
    beta = 0.5 - theta * theta / 24.0;
    gamma = (_S[0] * _S[3] + _S[1] * _S[4] + _S[2] * _S[5]) / 6.0
            - theta * theta / 120.0;
  }

  ret(0, 0) = beta * s2[0] + cos_t;
  ret(1, 0) = beta * s3[0] + alpha * _S[2];
  ret(2, 0) = beta * s3[2] - alpha * _S[1];

  ret(0, 1) = beta * s3[0] - alpha * _S[2];
  ret(1, 1) = beta * s2[1] + cos_t;
  ret(2, 1) = beta * s3[1] + alpha * _S[0];

  ret(0, 2) = beta * s3[2] + alpha * _S[1];
  ret(1, 2) = beta * s3[1] - alpha * _S[0];
  ret(2, 2) = beta * s2[2] + cos_t;

  ret(0, 3)
      = alpha * _S[3] + beta * (_S[1] * _S[5] - _S[2] * _S[4]) + gamma * _S[0];
  ret(1, 3)
      = alpha * _S[4] + beta * (_S[2] * _S[3] - _S[0] * _S[5]) + gamma * _S[1];
  ret(2, 3)
      = alpha * _S[5] + beta * (_S[0] * _S[4] - _S[1] * _S[3]) + gamma * _S[2];

  return ret;
}

/// \brief Exponential mapping, DART style. This treats the exponentiation
/// operation as a rotation, and then a translation, rather than an integration
/// of a screw.
Eigen::Isometry3s expMapDart(const Eigen::Vector6s& _S)
{
  Eigen::Isometry3s t = expAngular(_S.head<3>());
  t.translation() = _S.tail<3>();
  return t;
}

// I + sin(t) / t*[S] + (1 - cos(t)) / t^2*[S]^2, where t = |S|
Eigen::Isometry3s expAngular(const Eigen::Vector3s& _s)
{
  Eigen::Isometry3s ret = Eigen::Isometry3s::Identity();
  s_t s2[] = {_s[0] * _s[0], _s[1] * _s[1], _s[2] * _s[2]};
  s_t s3[] = {_s[0] * _s[1], _s[1] * _s[2], _s[2] * _s[0]};
  s_t theta = sqrt(s2[0] + s2[1] + s2[2]);
  s_t cos_t = cos(theta);
  s_t alpha = 0.0;
  s_t beta = 0.0;

  if (theta > DART_EPSILON)
  {
    alpha = sin(theta) / theta;
    beta = (1.0 - cos_t) / theta / theta;
  }
  else
  {
    alpha = 1.0 - theta * theta / 6.0;
    beta = 0.5 - theta * theta / 24.0;
  }

  ret(0, 0) = beta * s2[0] + cos_t;
  ret(1, 0) = beta * s3[0] + alpha * _s[2];
  ret(2, 0) = beta * s3[2] - alpha * _s[1];

  ret(0, 1) = beta * s3[0] - alpha * _s[2];
  ret(1, 1) = beta * s2[1] + cos_t;
  ret(2, 1) = beta * s3[1] + alpha * _s[0];

  ret(0, 2) = beta * s3[2] + alpha * _s[1];
  ret(1, 2) = beta * s3[1] - alpha * _s[0];
  ret(2, 2) = beta * s2[2] + cos_t;

  return ret;
}

// SE3 Normalize(const SE3& T)
// {
//    SE3 ret = SE3::Identity();
//    s_t idet = 1.0 / (T(0,0)*(T(1,1)*T(2,2) - T(2,1)*T(1,2)) +
//                              T(0,1)*(T(2,0)*T(1,2) - T(1,0)*T(2,2)) +
//                              T(0,2)*(T(1,0)*T(2,1) - T(2,0)*T(1,1)));

//    ret(0,0) = 1.0_2*(T(0,0) + idet*(T(1,1)*T(2,2) - T(2,1)*T(1,2)));
//    ret(0,1) = 1.0_2*(T(0,1) + idet*(T(2,0)*T(1,2) - T(1,0)*T(2,2)));
//    ret(0,2) = 1.0_2*(T(0,2) + idet*(T(1,0)*T(2,1) - T(2,0)*T(1,1)));
//    ret(0,3) = T(0,3);

//    ret(1,0) = 1.0_2*(T(1,0) + idet*(T(2,1)*T(0,2) - T(0,1)*T(2,2)));
//    ret(1,1) = 1.0_2*(T(1,1) + idet*(T(0,0)*T(2,2) - T(2,0)*T(0,2)));
//    ret(1,2) = 1.0_2*(T(1,2) + idet*(T(2,0)*T(0,1) - T(0,0)*T(2,1)));
//    ret(1,3) = T(1,3);

//    ret(2,0) = 1.0_2*(T(2,0) + idet*(T(0,1)*T(1,2) - T(1,1)*T(0,2)));
//    ret(2,1) = 1.0_2*(T(2,1) + idet*(T(1,0)*T(0,2) - T(0,0)*T(1,2)));
//    ret(2,2) = 1.0_2*(T(2,2) + idet*(T(0,0)*T(1,1) - T(1,0)*T(0,1)));
//    ret(2,3) = T(2,3);

//    return ret;
// }

// Axis Reparameterize(const Axis& s)
// {
//  s_t theta = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
//  s_t eta = theta < LIE_EPS
//               ? 1.0
//               : 1.0 - (s_t)((int)(theta / M_PI + 1.0) / 2)*M_2PI / theta;

//  return eta*s;
// }

// Axis AdInvTAngular(const SE3& T, const Axis& v)
// {
//    return Axis(T(0,0)*v[0] + T(1,0)*v[1] + T(2,0)*v[2],
//                T(0,1)*v[0] + T(1,1)*v[1] + T(2,1)*v[2],
//                T(0,2)*v[0] + T(1,2)*v[1] + T(2,2)*v[2]);
// }

// Axis ad(const Axis& s1, const se3& s2)
// {
//    return Axis(s2[2]*s1[1] - s2[1]*s1[2],
//                s2[0]*s1[2] - s2[2]*s1[0],
//                s2[1]*s1[0] - s2[0]*s1[1]);
// }

// Axis ad_Axis_Axis(const Axis& s1, const Axis& s2)
// {
//    return Axis(s2[2]*s1[1] - s2[1]*s1[2],
//                s2[0]*s1[2] - s2[2]*s1[0],
//                s2[1]*s1[0] - s2[0]*s1[1]);
// }

Eigen::Vector6s dad(const Eigen::Vector6s& _s, const Eigen::Vector6s& _t)
{
  Eigen::Vector6s res;
  res.head<3>()
      = _t.head<3>().cross(_s.head<3>()) + _t.tail<3>().cross(_s.tail<3>());
  res.tail<3>() = _t.tail<3>().cross(_s.head<3>());
  return res;
}

Inertia transformInertia(const Eigen::Isometry3s& _T, const Inertia& _I)
{
  // operation count: multiplication = 186, addition = 117, subtract = 21

  Inertia ret = Inertia::Identity();

  s_t d0 = _I(0, 3) + _T(2, 3) * _I(3, 4) - _T(1, 3) * _I(3, 5);
  s_t d1 = _I(1, 3) - _T(2, 3) * _I(3, 3) + _T(0, 3) * _I(3, 5);
  s_t d2 = _I(2, 3) + _T(1, 3) * _I(3, 3) - _T(0, 3) * _I(3, 4);
  s_t d3 = _I(0, 4) + _T(2, 3) * _I(4, 4) - _T(1, 3) * _I(4, 5);
  s_t d4 = _I(1, 4) - _T(2, 3) * _I(3, 4) + _T(0, 3) * _I(4, 5);
  s_t d5 = _I(2, 4) + _T(1, 3) * _I(3, 4) - _T(0, 3) * _I(4, 4);
  s_t d6 = _I(0, 5) + _T(2, 3) * _I(4, 5) - _T(1, 3) * _I(5, 5);
  s_t d7 = _I(1, 5) - _T(2, 3) * _I(3, 5) + _T(0, 3) * _I(5, 5);
  s_t d8 = _I(2, 5) + _T(1, 3) * _I(3, 5) - _T(0, 3) * _I(4, 5);
  s_t e0 = _I(0, 0) + _T(2, 3) * _I(0, 4) - _T(1, 3) * _I(0, 5) + d3 * _T(2, 3)
           - d6 * _T(1, 3);
  s_t e3 = _I(0, 1) + _T(2, 3) * _I(1, 4) - _T(1, 3) * _I(1, 5) - d0 * _T(2, 3)
           + d6 * _T(0, 3);
  s_t e4 = _I(1, 1) - _T(2, 3) * _I(1, 3) + _T(0, 3) * _I(1, 5) - d1 * _T(2, 3)
           + d7 * _T(0, 3);
  s_t e6 = _I(0, 2) + _T(2, 3) * _I(2, 4) - _T(1, 3) * _I(2, 5) + d0 * _T(1, 3)
           - d3 * _T(0, 3);
  s_t e7 = _I(1, 2) - _T(2, 3) * _I(2, 3) + _T(0, 3) * _I(2, 5) + d1 * _T(1, 3)
           - d4 * _T(0, 3);
  s_t e8 = _I(2, 2) + _T(1, 3) * _I(2, 3) - _T(0, 3) * _I(2, 4) + d2 * _T(1, 3)
           - d5 * _T(0, 3);
  s_t f0 = _T(0, 0) * e0 + _T(1, 0) * e3 + _T(2, 0) * e6;
  s_t f1 = _T(0, 0) * e3 + _T(1, 0) * e4 + _T(2, 0) * e7;
  s_t f2 = _T(0, 0) * e6 + _T(1, 0) * e7 + _T(2, 0) * e8;
  s_t f3 = _T(0, 0) * d0 + _T(1, 0) * d1 + _T(2, 0) * d2;
  s_t f4 = _T(0, 0) * d3 + _T(1, 0) * d4 + _T(2, 0) * d5;
  s_t f5 = _T(0, 0) * d6 + _T(1, 0) * d7 + _T(2, 0) * d8;
  s_t f6 = _T(0, 1) * e0 + _T(1, 1) * e3 + _T(2, 1) * e6;
  s_t f7 = _T(0, 1) * e3 + _T(1, 1) * e4 + _T(2, 1) * e7;
  s_t f8 = _T(0, 1) * e6 + _T(1, 1) * e7 + _T(2, 1) * e8;
  s_t g0 = _T(0, 1) * d0 + _T(1, 1) * d1 + _T(2, 1) * d2;
  s_t g1 = _T(0, 1) * d3 + _T(1, 1) * d4 + _T(2, 1) * d5;
  s_t g2 = _T(0, 1) * d6 + _T(1, 1) * d7 + _T(2, 1) * d8;
  s_t g3 = _T(0, 2) * d0 + _T(1, 2) * d1 + _T(2, 2) * d2;
  s_t g4 = _T(0, 2) * d3 + _T(1, 2) * d4 + _T(2, 2) * d5;
  s_t g5 = _T(0, 2) * d6 + _T(1, 2) * d7 + _T(2, 2) * d8;
  s_t h0 = _T(0, 0) * _I(3, 3) + _T(1, 0) * _I(3, 4) + _T(2, 0) * _I(3, 5);
  s_t h1 = _T(0, 0) * _I(3, 4) + _T(1, 0) * _I(4, 4) + _T(2, 0) * _I(4, 5);
  s_t h2 = _T(0, 0) * _I(3, 5) + _T(1, 0) * _I(4, 5) + _T(2, 0) * _I(5, 5);
  s_t h3 = _T(0, 1) * _I(3, 3) + _T(1, 1) * _I(3, 4) + _T(2, 1) * _I(3, 5);
  s_t h4 = _T(0, 1) * _I(3, 4) + _T(1, 1) * _I(4, 4) + _T(2, 1) * _I(4, 5);
  s_t h5 = _T(0, 1) * _I(3, 5) + _T(1, 1) * _I(4, 5) + _T(2, 1) * _I(5, 5);

  ret(0, 0) = f0 * _T(0, 0) + f1 * _T(1, 0) + f2 * _T(2, 0);
  ret(0, 1) = f0 * _T(0, 1) + f1 * _T(1, 1) + f2 * _T(2, 1);
  ret(0, 2) = f0 * _T(0, 2) + f1 * _T(1, 2) + f2 * _T(2, 2);
  ret(0, 3) = f3 * _T(0, 0) + f4 * _T(1, 0) + f5 * _T(2, 0);
  ret(0, 4) = f3 * _T(0, 1) + f4 * _T(1, 1) + f5 * _T(2, 1);
  ret(0, 5) = f3 * _T(0, 2) + f4 * _T(1, 2) + f5 * _T(2, 2);
  ret(1, 1) = f6 * _T(0, 1) + f7 * _T(1, 1) + f8 * _T(2, 1);
  ret(1, 2) = f6 * _T(0, 2) + f7 * _T(1, 2) + f8 * _T(2, 2);
  ret(1, 3) = g0 * _T(0, 0) + g1 * _T(1, 0) + g2 * _T(2, 0);
  ret(1, 4) = g0 * _T(0, 1) + g1 * _T(1, 1) + g2 * _T(2, 1);
  ret(1, 5) = g0 * _T(0, 2) + g1 * _T(1, 2) + g2 * _T(2, 2);
  ret(2, 2) = (_T(0, 2) * e0 + _T(1, 2) * e3 + _T(2, 2) * e6) * _T(0, 2)
              + (_T(0, 2) * e3 + _T(1, 2) * e4 + _T(2, 2) * e7) * _T(1, 2)
              + (_T(0, 2) * e6 + _T(1, 2) * e7 + _T(2, 2) * e8) * _T(2, 2);
  ret(2, 3) = g3 * _T(0, 0) + g4 * _T(1, 0) + g5 * _T(2, 0);
  ret(2, 4) = g3 * _T(0, 1) + g4 * _T(1, 1) + g5 * _T(2, 1);
  ret(2, 5) = g3 * _T(0, 2) + g4 * _T(1, 2) + g5 * _T(2, 2);
  ret(3, 3) = h0 * _T(0, 0) + h1 * _T(1, 0) + h2 * _T(2, 0);
  ret(3, 4) = h0 * _T(0, 1) + h1 * _T(1, 1) + h2 * _T(2, 1);
  ret(3, 5) = h0 * _T(0, 2) + h1 * _T(1, 2) + h2 * _T(2, 2);
  ret(4, 4) = h3 * _T(0, 1) + h4 * _T(1, 1) + h5 * _T(2, 1);
  ret(4, 5) = h3 * _T(0, 2) + h4 * _T(1, 2) + h5 * _T(2, 2);
  ret(5, 5)
      = (_T(0, 2) * _I(3, 3) + _T(1, 2) * _I(3, 4) + _T(2, 2) * _I(3, 5))
            * _T(0, 2)
        + (_T(0, 2) * _I(3, 4) + _T(1, 2) * _I(4, 4) + _T(2, 2) * _I(4, 5))
              * _T(1, 2)
        + (_T(0, 2) * _I(3, 5) + _T(1, 2) * _I(4, 5) + _T(2, 2) * _I(5, 5))
              * _T(2, 2);

  ret.triangularView<Eigen::StrictlyLower>() = ret.transpose();

  return ret;
}

Eigen::Matrix3s parallelAxisTheorem(
    const Eigen::Matrix3s& _original,
    const Eigen::Vector3s& _comShift,
    s_t _mass)
{
  const Eigen::Vector3s& p = _comShift;
  Eigen::Matrix3s result(_original);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      result(i, j) += _mass * (delta(i, j) * p.dot(p) - p(i) * p(j));

  return result;
}

bool verifyRotation(const Eigen::Matrix3s& _T)
{
  return !isNan(_T) && abs(_T.determinant() - 1.0) <= DART_EPSILON;
}

bool verifyTransform(const Eigen::Isometry3s& _T)
{
  return !isNan(_T.matrix().topRows<3>())
         && abs(_T.linear().determinant() - 1.0) <= DART_EPSILON;
}

/// This projects a global wrench to a [CoP, tau, f] vector
Eigen::Vector9s projectWrenchToCoP(
    Eigen::Vector6s worldWrench, s_t groundHeight, int verticalAxis)
{
  // To get COP, solve for k and p, where we know the y-coordinate of p in
  // advance:
  //
  // k * worldF = worldTau - worldF.cross(p);
  // k * worldF + crossF * p_unknown = worldTau - worldF.cross(p_known);
  // [worldF, crossF] * [k, p_unknown] = worldTau - worldF.cross(p_known);
  Eigen::Vector3s worldTau = worldWrench.head<3>();
  Eigen::Vector3s worldF = worldWrench.tail<3>();

  Eigen::Matrix3s crossF = math::makeSkewSymmetric(worldF);
  assert(!crossF.hasNaN());
  Eigen::Vector3s rightSide
      = worldTau + crossF.col(verticalAxis) * groundHeight;
  assert(!rightSide.hasNaN());
  Eigen::Matrix3s leftSide = -crossF;
  leftSide.col(verticalAxis) = worldF;
  assert(!leftSide.hasNaN());
  Eigen::Vector3s p
      = leftSide.completeOrthogonalDecomposition().solve(rightSide);
  assert(!p.hasNaN());
  s_t k = p(verticalAxis);
  p(verticalAxis) = groundHeight;
  Eigen::Vector3s expectedTau = worldF * k;
  Eigen::Vector3s cop = p;

#ifndef NDEBUG
  Eigen::Vector3s recoveredTau
      = worldWrench.head<3>() + worldWrench.tail<3>().cross(cop);
  if ((recoveredTau - expectedTau).norm() > 1e-8)
  {
    std::cout << "World wrench: " << std::endl << worldWrench << std::endl;
    std::cout << "Recovered tau doesn't match expected tau ("
              << (recoveredTau - expectedTau).norm() << ")!" << std::endl;
    std::cout << "Expected tau: " << std::endl << expectedTau << std::endl;
    std::cout << "Recovered tau: " << std::endl << recoveredTau << std::endl;
    assert((recoveredTau - expectedTau).norm() <= 1e-8);
  }
#endif

  Eigen::Vector9s result;
  result.segment<3>(0) = cop;
  result.segment<3>(3) = expectedTau;
  result.segment<3>(6) = worldF;

  return result;
}

/// This gets the relationship between changes in the world wrench and the
/// resulting [CoP, tau, f] vector
Eigen::Matrix<s_t, 9, 6> getProjectWrenchToCoPJacobian(
    Eigen::Vector6s worldWrench, s_t groundHeight, int verticalAxis)
{
  Eigen::Vector3s worldTau = worldWrench.head<3>();
  Eigen::Vector3s worldF = worldWrench.tail<3>();

  Eigen::Matrix3s crossF = math::makeSkewSymmetric(worldF);
  Eigen::Vector3s rightSide
      = worldTau + crossF.col(verticalAxis) * groundHeight;
  Eigen::Matrix3s leftSide = -crossF;
  leftSide.col(verticalAxis) = worldF;
  auto leftSideFactored = leftSide.completeOrthogonalDecomposition();
  Eigen::Vector3s p = leftSideFactored.solve(rightSide);
  s_t k = p(verticalAxis);

  Eigen::Matrix<s_t, 9, 6> J = Eigen::Matrix<s_t, 9, 6>::Zero();
  for (int i = 0; i < 6; i++)
  {
    if (i < 3)
    {
      Eigen::Vector3s dRightSide = Eigen::Vector3s::Unit(i);
      Eigen::Vector3s dp = leftSideFactored.solve(dRightSide);

      s_t dk = dp(verticalAxis);
      dp(verticalAxis) = 0;
      Eigen::Vector3s dExpectedTau = worldF * dk;
      Eigen::Vector3s dCop = dp;
      J.block<3, 1>(0, i) = dCop;
      J.block<3, 1>(3, i) = dExpectedTau;
    }
    else
    {
      Eigen::Vector3s dF = Eigen::Vector3s::Unit(i - 3);
      Eigen::Matrix3s dCrossF = math::makeSkewSymmetric(dF);
      Eigen::Vector3s dRightSide = dCrossF.col(verticalAxis) * groundHeight;
      Eigen::Matrix3s dLeftSide = -dCrossF;
      dLeftSide.col(verticalAxis) = dF;

      // d(A^{-1}) = -1 * A^{-1} * dA * A^{-1}
      Eigen::Vector3s dp = -(leftSideFactored.solve(
                               dLeftSide * leftSideFactored.solve(rightSide)))
                           + leftSideFactored.solve(dRightSide);
      s_t dk = dp(verticalAxis);
      dp(verticalAxis) = 0;
      Eigen::Vector3s dExpectedTau = worldF * dk + dF * k;
      Eigen::Vector3s dCop = dp;
      J.block<3, 1>(0, i) = dCop;
      J.block<3, 1>(3, i) = dExpectedTau;
      J.block<3, 1>(6, i) = dF;
    }
  }
  return J;
}

/// This gets the relationship between changes in the world wrench and the
/// resulting [CoP, tau, f] vector
Eigen::Matrix<s_t, 9, 6> finiteDifferenceProjectWrenchToCoPJacobian(
    Eigen::Vector6s worldWrench, s_t groundHeight, int verticalAxis)
{
  Eigen::Matrix<s_t, 9, 6> J = Eigen::Matrix<s_t, 9, 6>::Zero();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(9, 6);
  const bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-6;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::Vector6s tweaked = worldWrench;
        tweaked(dof) += eps;
        perturbed = projectWrenchToCoP(tweaked, groundHeight, verticalAxis);
        return true;
      },
      result,
      eps,
      useRidders);
  J = result;

  return J;
}

Eigen::Vector3s fromSkewSymmetric(const Eigen::Matrix3s& _m)
{
#ifndef NDEBUG
  if (abs(_m(0, 0)) > DART_EPSILON || abs(_m(1, 1)) > DART_EPSILON
      || abs(_m(2, 2)) > DART_EPSILON)
  {
    dtwarn << "[math::fromSkewSymmetric] Not skew a symmetric matrix:\n"
           << _m << "\n";
    return Eigen::Vector3s::Zero();
  }
#endif
  Eigen::Vector3s ret;
  ret << _m(2, 1), _m(0, 2), _m(1, 0);
  return ret;
}

/// This checks whether a 2D shape contains a point. This assumes that shape was
/// sorted using sortConvex2DShape().
///
/// Source:
/// https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
/// A better source:
/// https://inginious.org/course/competitive-programming/geometry-pointinconvex
bool convex2DShapeContains(
    const Eigen::Vector3s& point,
    const std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY)
{
  Eigen::Vector2s point2d = pointInPlane(point, origin, basis2dX, basis2dY);

  int side = 0;
  for (int i = 0; i < shape.size(); i++)
  {
    Eigen::Vector2s a = pointInPlane(shape[i], origin, basis2dX, basis2dY);
    Eigen::Vector2s b = pointInPlane(
        shape[(i + 1) % shape.size()], origin, basis2dX, basis2dY);
    int thisSide = crossProduct2D(point2d - a, b - a) > 0 ? 1 : -1;
    if (i == 0)
      side = thisSide;
    else if (thisSide == 0)
      continue;
    else if (side == 0 && thisSide != 0)
      side = thisSide;
    else if (side != thisSide && side != 0)
      return false;
  }

  return true;
}

/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape. This sorts the shape by angle from the
/// center, and trims out any points that lie inside the convex polygon.
void prepareConvex2DShape(
    std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY)
{
  // Sort the shape in clockwise order around some internal point (choose the
  // average).
  Eigen::Vector2s avg = Eigen::Vector2s::Zero();
  for (Eigen::Vector3s pt : shape)
  {
    avg += pointInPlane(pt, origin, basis2dX, basis2dY);
  }
  avg /= shape.size();
  std::sort(
      shape.begin(),
      shape.end(),
      [&avg, &origin, &basis2dX, &basis2dY](
          Eigen::Vector3s& a, Eigen::Vector3s& b) {
        return angle2D(avg, pointInPlane(a, origin, basis2dX, basis2dY))
               < angle2D(avg, pointInPlane(b, origin, basis2dX, basis2dY));
      });
}

s_t angle2D(const Eigen::Vector2s& from, const Eigen::Vector2s& to)
{
  return atan2(to(1) - from(1), to(0) - from(0));
}

/// This transforms a 3D point down to a 2D point in the given 3D plane
Eigen::Vector2s pointInPlane(
    const Eigen::Vector3s& point,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY)
{
  return Eigen::Vector2s(
      (point - origin).dot(basis2dX), (point - origin).dot(basis2dY));
}

// This implements the "2D cross product" as redefined here:
// https://stackoverflow.com/a/565282/13177487
inline s_t crossProduct2D(const Eigen::Vector2s& v, const Eigen::Vector2s& w)
{
  return v(0) * w(1) - v(1) * w(0);
}

Eigen::Matrix3s makeSkewSymmetric(const Eigen::Vector3s& _v)
{
  Eigen::Matrix3s result = Eigen::Matrix3s::Zero();

  result(0, 1) = -_v(2);
  result(1, 0) = _v(2);
  result(0, 2) = _v(1);
  result(2, 0) = -_v(1);
  result(1, 2) = -_v(0);
  result(2, 1) = _v(0);

  return result;
}

//==============================================================================
Eigen::Matrix3s computeRotation(
    const Eigen::Vector3s& axis, const AxisType axisType)
{
  assert(axis != Eigen::Vector3s::Zero());

  // First axis
  const Eigen::Vector3s axis0 = axis.normalized();

  // Second axis
  Eigen::Vector3s axis1 = axis0.cross(Eigen::Vector3s::UnitX());
  if (axis1.norm() < DART_EPSILON)
    axis1 = axis0.cross(Eigen::Vector3s::UnitY());
  axis1.normalize();

  // Third axis
  const Eigen::Vector3s axis2 = axis0.cross(axis1).normalized();

  // Assign the three axes
  Eigen::Matrix3s result;
  int index = axisType;
  result.col(index) = axis0;
  result.col(++index % 3) = axis1;
  result.col(++index % 3) = axis2;

  assert(verifyRotation(result));

  return result;
}

//==============================================================================
Eigen::Isometry3s computeTransform(
    const Eigen::Vector3s& axis,
    const Eigen::Vector3s& translation,
    AxisType axisType)
{
  Eigen::Isometry3s result = Eigen::Isometry3s::Identity();

  result.linear() = computeRotation(axis, axisType);
  result.translation() = translation;

  // Verification
  assert(verifyTransform(result));

  return result;
}

//==============================================================================
Eigen::Isometry3s getFrameOriginAxisZ(
    const Eigen::Vector3s& _origin, const Eigen::Vector3s& _axisZ)
{
  return computeTransform(_axisZ, _origin, AxisType::AXIS_Z);
}

//==============================================================================
SupportPolygon computeSupportPolgyon(
    const SupportGeometry& _geometry,
    const Eigen::Vector3s& _axis1,
    const Eigen::Vector3s& _axis2)
{
  std::vector<std::size_t> indices;
  indices.reserve(_geometry.size());
  return computeSupportPolgyon(indices, _geometry, _axis1, _axis2);
}

//==============================================================================
SupportPolygon computeSupportPolgyon(
    std::vector<std::size_t>& _originalIndices,
    const SupportGeometry& _geometry,
    const Eigen::Vector3s& _axis1,
    const Eigen::Vector3s& _axis2)
{
  SupportPolygon polygon;
  polygon.reserve(_geometry.size());
  for (const Eigen::Vector3s& v : _geometry)
    polygon.push_back(Eigen::Vector2s(v.dot(_axis1), v.dot(_axis2)));

  return computeConvexHull(_originalIndices, polygon);
}

//==============================================================================
// HullAngle is an internal struct used to facilitate the computation of 2D
// convex hulls
struct HullAngle
{
  HullAngle(s_t angle, s_t distance, std::size_t index)
    : mAngle(angle), mDistance(distance), mIndex(index)
  {
    // Do nothing
  }

  s_t mAngle;
  s_t mDistance;
  std::size_t mIndex;
};

//==============================================================================
// Comparison function to allow hull angles to be sorted
static bool HullAngleComparison(const HullAngle& a, const HullAngle& b)
{
  return a.mAngle < b.mAngle;
}

//==============================================================================
SupportPolygon computeConvexHull(const SupportPolygon& _points)
{
  std::vector<std::size_t> indices;
  indices.reserve(_points.size());
  return computeConvexHull(indices, _points);
}

//==============================================================================
Eigen::Vector2s computeCentroidOfHull(const SupportPolygon& _convexHull)
{
  if (_convexHull.size() == 0)
  {
    Eigen::Vector2s invalid = Eigen::Vector2s::Constant(std::nan(""));
    dtwarn << "[computeCentroidOfHull] Requesting the centroid of an empty set "
           << "of points! We will return <" << invalid.transpose() << ">.\n";
    return invalid;
  }

  if (_convexHull.size() == 1)
    return _convexHull[0];

  if (_convexHull.size() == 2)
    return (_convexHull[0] + _convexHull[1]) / 2.0;

  Eigen::Vector2s c(0, 0);
  Eigen::Vector2s intersect;
  s_t area = 0;
  s_t area_i;
  Eigen::Vector2s midp12, midp01;

  for (std::size_t i = 2; i < _convexHull.size(); ++i)
  {
    const Eigen::Vector2s& p0 = _convexHull[0];
    const Eigen::Vector2s& p1 = _convexHull[i - 1];
    const Eigen::Vector2s& p2 = _convexHull[i];

    area_i = 0.5
             * ((p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p1[1] - p0[1]) * (p2[0] - p0[0]));

    midp12 = 0.5 * (p1 + p2);
    midp01 = 0.5 * (p0 + p1);

    IntersectionResult result
        = computeIntersection(intersect, p0, midp12, p2, midp01);

    if (BEYOND_ENDPOINTS == result)
    {
      s_t a1 = atan2((p1 - p0)[1], (p1 - p0)[0]) * 180.0 / constantsd::pi();
      s_t a2 = atan2((p2 - p0)[1], (p2 - p0)[0]) * 180.0 / constantsd::pi();
      s_t diff = a1 - a2;
      dtwarn << "[computeCentroidOfHull] You have passed in a set of points "
             << "which is not a proper convex hull! The invalid segment "
             << "contains indices " << i - 1 << " -> " << i << ":\n"
             << i - 1 << ") " << p1.transpose() << " (" << a1 << " degrees)"
             << "\n"
             << i << ") " << p2.transpose() << " (" << a2 << " degrees)"
             << "\n"
             << "0) " << p0.transpose() << "\n"
             << "(" << result << ") "
             << (PARALLEL == result
                     ? "These segments are parallel!\n"
                     : "These segments are too short to intersect!\n")
             << "Difference in angle: " << diff << "\n\n";
      continue;
    }

    area += area_i;
    c += area_i * intersect;
  }

  // A negative area means we have a bug in the code
  assert(area >= 0.0);

  if (area == 0.0)
    return c;

  return c / area;
}

//==============================================================================
// Returns true if the path that goes from p1 -> p2 -> p3 turns left
static bool isLeftTurn(
    const Eigen::Vector2s& p1,
    const Eigen::Vector2s& p2,
    const Eigen::Vector2s& p3)
{
  return (cross(p2 - p1, p3 - p1) > 0);
}

//==============================================================================
SupportPolygon computeConvexHull(
    std::vector<std::size_t>& _originalIndices, const SupportPolygon& _points)
{
  _originalIndices.clear();

  if (_points.size() <= 3)
  {
    // Three or fewer points is already a convex hull
    for (std::size_t i = 0; i < _points.size(); ++i)
      _originalIndices.push_back(i);

    return _points;
  }

  // We'll use "Graham scan" to compute the convex hull in the general case
  std::size_t lowestIndex = static_cast<std::size_t>(-1);
  s_t lowestY = std::numeric_limits<s_t>::infinity();
  for (std::size_t i = 0; i < _points.size(); ++i)
  {
    if (_points[i][1] < lowestY)
    {
      lowestIndex = i;
      lowestY = _points[i][1];
    }
    else if (_points[i][1] == lowestY)
    {
      if (_points[i][0] < _points[lowestIndex][0])
      {
        lowestIndex = i;
      }
    }
  }

  std::vector<HullAngle> angles;
  const Eigen::Vector2s& bottom = _points[lowestIndex];
  for (std::size_t i = 0; i < _points.size(); ++i)
  {
    const Eigen::Vector2s& p = _points[i];
    if (p != bottom)
    {
      const Eigen::Vector2s& v = p - bottom;
      angles.push_back(HullAngle(atan2(v[1], v[0]), v.norm(), i));
    }
  }

  std::sort(angles.begin(), angles.end(), HullAngleComparison);

  if (angles.size() > 1)
  {
    for (std::size_t i = 0; i < angles.size() - 1; ++i)
    {
      if (abs(angles[i].mAngle - angles[i + 1].mAngle) < 1e-12)
      {
        // If two points have the same angle, throw out the one that is closer
        // to the corner
        std::size_t tossout
            = (angles[i].mDistance < angles[i + 1].mDistance) ? i : i + 1;
        angles.erase(angles.begin() + tossout);
        --i;
      }
    }
  }

  if (angles.size() <= 3)
  {
    // There were so many repeated points in the given set that we only have
    // three or fewer unique points
    _originalIndices.reserve(angles.size() + 1);
    _originalIndices.push_back(lowestIndex);
    for (std::size_t i = 0; i < angles.size(); ++i)
      _originalIndices.push_back(angles[i].mIndex);

    SupportPolygon polygon;
    polygon.reserve(_originalIndices.size());
    for (std::size_t index : _originalIndices)
      polygon.push_back(_points[index]);

    return polygon;
  }

  std::vector<std::size_t>& edge = _originalIndices;
  std::size_t lastIndex = lowestIndex;
  std::size_t secondToLastIndex = angles[0].mIndex;
  edge.reserve(angles.size() + 1);

  edge.push_back(lowestIndex);

  for (std::size_t i = 1; i < angles.size(); ++i)
  {
    std::size_t currentIndex = angles[i].mIndex;
    const Eigen::Vector2s& p1 = _points[lastIndex];
    const Eigen::Vector2s& p2 = _points[secondToLastIndex];
    const Eigen::Vector2s& p3 = _points[currentIndex];

    bool leftTurn = isLeftTurn(p1, p2, p3);

    if (leftTurn)
    {
      edge.push_back(secondToLastIndex);
      lastIndex = secondToLastIndex;
      secondToLastIndex = currentIndex;
    }
    else
    {
      secondToLastIndex = edge.back();
      edge.pop_back();
      lastIndex = edge.back();
      --i;
    }
  }

  const Eigen::Vector2s& p1 = _points[edge.back()];
  const Eigen::Vector2s& p2 = _points[angles.back().mIndex];
  const Eigen::Vector2s& p3 = _points[lowestIndex];
  if (isLeftTurn(p1, p2, p3))
    edge.push_back(angles.back().mIndex);

  SupportPolygon polygon;
  polygon.reserve(edge.size());
  for (std::size_t index : edge)
    polygon.push_back(_points[index]);

  // Note that we do not need to fill in _originalIndices, because "edge" is a
  // non-const reference to _originalIndices and it has been filled in with the
  // appropriate values.
  return polygon;
}

//==============================================================================
IntersectionResult computeIntersection(
    Eigen::Vector2s& _intersectionPoint,
    const Eigen::Vector2s& a1,
    const Eigen::Vector2s& a2,
    const Eigen::Vector2s& b1,
    const Eigen::Vector2s& b2)
{
  s_t dx_a = a2[0] - a1[0];
  s_t dy_a = a2[1] - a1[1];

  s_t dx_b = b2[0] - b1[0];
  s_t dy_b = b2[1] - b1[1];

  Eigen::Vector2s& point = _intersectionPoint;

  if (abs(dx_b * dy_a - dx_a * dy_b) < 1e-12)
  {
    // The line segments are parallel, so give back an average of all the points
    point = (a1 + a2 + b1 + b2) / 4.0;
    return PARALLEL;
  }

  point[0] = (dx_b * dy_a * a1[0] - dx_a * dy_b * b1[0]
              + dx_a * dx_b * (b1[1] - a1[1]))
             / (dx_b * dy_a - dx_a * dy_b);

  if (dx_a != 0.0)
    point[1] = dy_a / dx_a * (point[0] - a1[0]) + a1[1];
  else
    point[1] = dy_b / dx_b * (point[0] - b1[0]) + b1[1];

  for (std::size_t i = 0; i < 2; ++i)
  {
    if ((point[i] < std::min(a1[i], a2[i])) || (max(a1[i], a2[i]) < point[i]))
    {
      return BEYOND_ENDPOINTS;
    }

    if ((point[i] < std::min(b1[i], b2[i])) || (max(b1[i], b2[i]) < point[i]))
    {
      return BEYOND_ENDPOINTS;
    }
  }

  return INTERSECTING;
}

//==============================================================================
s_t cross(const Eigen::Vector2s& _v1, const Eigen::Vector2s& _v2)
{
  return _v1[0] * _v2[1] - _v1[1] * _v2[0];
}

//==============================================================================
bool isInsideSupportPolygon(
    const Eigen::Vector2s& _p,
    const SupportPolygon& _support,
    bool _includeEdge)
{
  if (_support.size() == 0)
    return false;

  if (_support.size() == 1)
  {
    if (!_includeEdge)
      return false;

    return (_support[0] == _p);
  }

  if (_support.size() == 2)
  {
    if (!_includeEdge)
      return false;

    const Eigen::Vector2s& p1 = _support[0];
    const Eigen::Vector2s& p2 = _support[1];
    const Eigen::Vector2s& p3 = _p;

    if (cross(p2 - p1, p3 - p1) == 0)
    {
      if (p3[0] < std::min(p1[0], p2[0]) || max(p1[0], p2[0]) < p3[0])
        return false;

      return true;
    }

    return false;
  }

  for (std::size_t i = 0; i < _support.size(); ++i)
  {
    const Eigen::Vector2s& p1 = (i == 0) ? _support.back() : _support[i - 1];
    const Eigen::Vector2s& p2 = _support[i];
    const Eigen::Vector2s& p3 = _p;

    s_t crossProduct = cross(p2 - p1, p3 - p1);
    if (crossProduct > 0.0)
      continue;

    if (crossProduct == 0)
    {
      if (!_includeEdge)
        return false;

      if (p3[0] < std::min(p1[0], p2[0]) || max(p1[0], p2[0]) < p3[0])
        return false;

      return true;
    }
    else
    {
      return false;
    }
  }

  return true;
}

//==============================================================================
Eigen::Vector2s computeClosestPointOnLineSegment(
    const Eigen::Vector2s& _p,
    const Eigen::Vector2s& _s1,
    const Eigen::Vector2s& _s2)
{
  Eigen::Vector2s result;

  if (_s1[0] - _s2[0] == 0)
  {
    result[0] = _s1[0];
    result[1] = _p[1];

    if (result[1] < std::min(_s1[1], _s2[1]) || max(_s1[1], _s2[1]) < result[1])
    {
      if (abs(_p[1] - _s2[1]) < abs(_p[1] - _s1[1]))
        result[1] = _s2[1];
      else
        result[1] = _s1[1];
    }
  }
  else
  {
    s_t m = (_s2[1] - _s1[1]) / (_s2[0] - _s1[0]);
    s_t k = _s1[1] - m * _s1[0];
    result[0] = (_p[0] + m * (_p[1] - k)) / (m * m + 1.0);
    result[1] = m * result[0] + k;

    if (result[0] < std::min(_s1[0], _s2[0]) || max(_s1[0], _s2[0]) < result[0])
    {
      if ((_p - _s2).norm() < (_p - _s1).norm())
        result = _s2;
      else
        result = _s1;
    }
  }

  return result;
}

//==============================================================================
Eigen::Vector2s computeClosestPointOnSupportPolygon(
    const Eigen::Vector2s& _p, const SupportPolygon& _support)
{
  std::size_t _index1;
  std::size_t _index2;
  return computeClosestPointOnSupportPolygon(_index1, _index2, _p, _support);
}

//==============================================================================
Eigen::Vector2s computeClosestPointOnSupportPolygon(
    std::size_t& _index1,
    std::size_t& _index2,
    const Eigen::Vector2s& _p,
    const SupportPolygon& _support)
{
  if (_support.size() == 0)
  {
    _index1 = static_cast<std::size_t>(-1);
    _index2 = _index1;
    return _p;
  }

  if (_support.size() == 1)
  {
    _index1 = 0;
    _index2 = 0;
    return _support[0];
  }

  if (_support.size() == 2)
  {
    _index1 = 0;
    _index2 = 1;
    return computeClosestPointOnLineSegment(_p, _support[0], _support[1]);
  }

  s_t best = std::numeric_limits<s_t>::infinity(), check;
  Eigen::Vector2s test, result;
  for (std::size_t i = 0; i < _support.size(); ++i)
  {
    const Eigen::Vector2s& p1 = (i == 0) ? _support.back() : _support[i - 1];
    const Eigen::Vector2s& p2 = _support[i];

    test = computeClosestPointOnLineSegment(_p, p1, p2);
    check = (test - _p).norm();
    if (check < best)
    {
      best = check;
      result = test;
      _index1 = (i == 0) ? _support.size() - 1 : i - 1;
      _index2 = i;
    }
  }

  return result;
}

//==============================================================================
Eigen::Vector3s closestPointOnLine(
    const Eigen::Vector3s& pointOnLine,
    const Eigen::Vector3s& lineDirection,
    const Eigen::Vector3s& goalPoint)
{
  s_t offset = lineDirection.dot(pointOnLine);
  s_t relative = lineDirection.dot(goalPoint) - offset;
  return pointOnLine + relative * lineDirection;
}

//==============================================================================
Eigen::Vector3s closestPointOnLineGradient(
    const Eigen::Vector3s& pointOnLine,
    const Eigen::Vector3s& pointOnLineGradient,
    const Eigen::Vector3s& lineDirection,
    const Eigen::Vector3s& lineDirectionGradient,
    const Eigen::Vector3s& goalPoint,
    const Eigen::Vector3s& goalPointGradient)
{
  s_t offset = lineDirection.dot(pointOnLine);
  s_t dOffset = lineDirectionGradient.dot(pointOnLine)
                + lineDirection.dot(pointOnLineGradient);
  s_t goalOffset = lineDirection.dot(goalPoint);
  s_t dGoalOffset = lineDirectionGradient.dot(goalPoint)
                    + lineDirection.dot(goalPointGradient);
  s_t relative = goalOffset - offset;
  s_t dRelative = dGoalOffset - dOffset;
  return pointOnLineGradient + relative * lineDirectionGradient
         + dRelative * lineDirection;
}

//==============================================================================
/// This computes and returns the distance to the closest point on a line
/// segment
s_t distanceToSegment(
    const Eigen::Vector3s& segmentPointA,
    const Eigen::Vector3s& segmentPointB,
    const Eigen::Vector3s& goalPoint)
{
  // From A to B
  Eigen::Vector3s dir = (segmentPointB - segmentPointA).normalized();

  s_t onDir = dir.dot(goalPoint);
  s_t aOnDir = dir.dot(segmentPointA);
  s_t bOnDir = dir.dot(segmentPointB);
  assert(aOnDir <= bOnDir);
  if (onDir < aOnDir)
  {
    return (segmentPointA - goalPoint).norm();
  }
  else if (onDir > bOnDir)
  {
    return (segmentPointB - goalPoint).norm();
  }
  else
  {
    Eigen::Vector3s closestPoint = segmentPointA + dir * (onDir - aOnDir);
    return (closestPoint - goalPoint).norm();
  }
}

//==============================================================================
/// This gets the closest approximation to `desiredRotation` that we can get,
/// rotating around `axis`
s_t getClosestRotationalApproximation(
    const Eigen::Vector3s& axis, const Eigen::Matrix3s& desiredRotation)
{
  // We can treat this like a 2D rotation, by establishing a basis around the
  // rotational axis, and then doing all our work projected onto the
  // orthogonal plane.
  Eigen::Vector3s arbitraryFirstAxis = Eigen::Vector3s::UnitZ();
  // If we're too close to parallel, pick another axis
  if (axis.dot(arbitraryFirstAxis) > 0.99
      || axis.dot(arbitraryFirstAxis) < -0.99)
  {
    arbitraryFirstAxis = Eigen::Vector3s::UnitX();
  }

  // Get an orthogonal plane to the rotation axis
  Eigen::Vector3s x = axis.cross(arbitraryFirstAxis);
  Eigen::Vector3s y = axis.cross(x);

  // A basis where the `axis` as always the Z axis
  Eigen::Matrix3s R_bw = Eigen::Matrix3s::Zero();
  R_bw.col(0) = x;
  R_bw.col(1) = y;
  R_bw.col(2) = axis;

  if (R_bw.determinant() < 0)
  {
    R_bw.col(2) *= -1;
  }

  // Get our desired rotation in the coordinate space of the `axis` basis
  Eigen::Matrix3s R_b = R_bw.transpose() * desiredRotation * R_bw;

  // Now the top left corner of R_b is our rotation matrix in 2D, about the
  // `axis` basis
  Eigen::Matrix2s twoDimensionalRotation = R_b.block<2, 2>(0, 0);
  Eigen::JacobiSVD<Eigen::Matrix2s> svd(
      twoDimensionalRotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2s U = svd.matrixU();
  Eigen::Matrix2s V = svd.matrixV();
  Eigen::Matrix2s normalizedTwoDimensional = U * V.transpose();
  if (normalizedTwoDimensional.determinant() < 0)
  {
    normalizedTwoDimensional.col(1) *= -1;
  }
  s_t angle
      = atan2(normalizedTwoDimensional(1, 0), normalizedTwoDimensional(0, 0));

  // And now, perform gradient descent using a finite-differenced gradient!

  s_t cost = (desiredRotation - math::expMapRot(axis * angle)).norm();
  const s_t eps = 1e-3;
  s_t stepSize = 1e-2;
  for (int iter = 0; iter < 100; iter++)
  {
    s_t plusCost
        = (desiredRotation - math::expMapRot(axis * (angle + eps))).norm();
    s_t minusCost
        = (desiredRotation - math::expMapRot(axis * (angle - eps))).norm();
    s_t grad = (plusCost - minusCost) / (2 * eps);

    while (stepSize > 1e-12)
    {
      s_t proposedAngle = angle - grad * stepSize;
      s_t proposedCost
          = (desiredRotation - math::expMapRot(axis * proposedAngle)).norm();
      if (proposedCost < cost)
      {
        cost = proposedCost;
        angle = proposedAngle;
        stepSize *= 1.2;
        break;
      }
      stepSize *= 0.5;
    }
  }

  return angle;
}

//==============================================================================
/// This will rotate and translate a point cloud to match the first N points
/// as closely as possible to the passed in matrix
Eigen::MatrixXs mapPointCloudToData(
    const Eigen::MatrixXs& pointCloud,
    std::vector<Eigen::Vector3s> firstNPoints)
{
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> targetPointCloud
      = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, firstNPoints.size());
  for (int i = 0; i < firstNPoints.size(); i++)
  {
    targetPointCloud.col(i) = firstNPoints[i];
  }
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> sourcePointCloud
      = pointCloud.block(0, 0, 3, firstNPoints.size());

  assert(sourcePointCloud.cols() == targetPointCloud.cols());

  // Compute the centroids of the source and target points
  Eigen::Vector3s sourceCentroid = sourcePointCloud.rowwise().mean();
  Eigen::Vector3s targetCentroid = targetPointCloud.rowwise().mean();

#ifndef NDEBUG
  Eigen::Vector3s sourceAvg = Eigen::Vector3s::Zero();
  Eigen::Vector3s targetAvg = Eigen::Vector3s::Zero();
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    sourceAvg += sourcePointCloud.col(i);
    targetAvg += targetPointCloud.col(i);
  }
  sourceAvg /= sourcePointCloud.cols();
  targetAvg /= targetPointCloud.cols();
  assert((sourceAvg - sourceCentroid).norm() < 1e-12);
  assert((targetAvg - targetCentroid).norm() < 1e-12);
#endif

  // Compute the centered source and target points
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredSourcePoints
      = sourcePointCloud.colwise() - sourceCentroid;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredTargetPoints
      = targetPointCloud.colwise() - targetCentroid;

#ifndef NDEBUG
  assert(std::abs(centeredSourcePoints.rowwise().mean().norm()) < 1e-8);
  assert(std::abs(centeredTargetPoints.rowwise().mean().norm()) < 1e-8);
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    Eigen::Vector3s expectedCenteredSourcePoints
        = sourcePointCloud.col(i) - sourceAvg;
    assert(
        (centeredSourcePoints.col(i) - expectedCenteredSourcePoints).norm()
        < 1e-12);
    Eigen::Vector3s expectedCenteredTargetPoints
        = targetPointCloud.col(i) - targetAvg;
    assert(
        (centeredTargetPoints.col(i) - expectedCenteredTargetPoints).norm()
        < 1e-12);
  }
#endif

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix
      = centeredTargetPoints * centeredSourcePoints.transpose();

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  // Normally, we would want to check the determinant of R here, to ensure that
  // we're only doing right-handed rotations. HOWEVER, because we're using a
  // point cloud, we may actually have to flip the data along an axis to get it
  // to match up, so we skip the determinant check.

  // Transform the source point cloud to the target point cloud
  Eigen::MatrixXs transformed = Eigen::MatrixXs::Zero(3, pointCloud.cols());
  for (int i = 0; i < pointCloud.cols(); i++)
  {
    transformed.col(i)
        = R * (pointCloud.col(i).head<3>() - sourceCentroid) + targetCentroid;
  }
  return transformed;
}

//==============================================================================
/// This will give the world transform necessary to apply to the local points
/// (worldT * p[i] for all localPoints) to get the local points to match the
/// world points as closely as possible.
Eigen::Isometry3s getPointCloudToPointCloudTransform(
    std::vector<Eigen::Vector3s> localPoints,
    std::vector<Eigen::Vector3s> worldPoints,
    std::vector<s_t> weights)
{
  assert(localPoints.size() > 0);
  assert(worldPoints.size() > 0);
  assert(localPoints.size() == worldPoints.size());

  // Compute the centroids of the local and world points
  Eigen::Vector3s localCentroid = Eigen::Vector3s::Zero();
  s_t sumWeights = 0.0;
  for (int i = 0; i < localPoints.size(); i++)
  {
    Eigen::Vector3s& point = localPoints[i];
    sumWeights += weights[i];
    localCentroid += point * weights[i];
  }
  localCentroid /= sumWeights;
  Eigen::Vector3s worldCentroid = Eigen::Vector3s::Zero();
  for (int i = 0; i < worldPoints.size(); i++)
  {
    Eigen::Vector3s& point = worldPoints[i];
    worldCentroid += point * weights[i];
  }
  worldCentroid /= sumWeights;

  // Compute the centered local and world points
  std::vector<Eigen::Vector3s> centeredLocalPoints;
  std::vector<Eigen::Vector3s> centeredWorldPoints;
  for (int i = 0; i < localPoints.size(); i++)
  {
    centeredLocalPoints.push_back(localPoints[i] - localCentroid);
    centeredWorldPoints.push_back(worldPoints[i] - worldCentroid);
  }

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix = Eigen::Matrix3s::Zero();
  for (int i = 0; i < localPoints.size(); i++)
  {
    covarianceMatrix += weights[i] * centeredWorldPoints[i]
                        * centeredLocalPoints[i].transpose();
  }

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  if (R.determinant() < 0)
  {
    Eigen::Matrix3s scales = Eigen::Matrix3s::Identity();
    scales(2, 2) = -1;
    R = U * scales * V.transpose();
  }
  Eigen::Vector3s translation = worldCentroid - R * localCentroid;

  Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
  transform.linear() = R;
  transform.translation() = translation;
  return transform;
}

//==============================================================================
/// This will give the world transform necessary to apply to the local points
/// (worldT * p[i] for all localPoints) to get the local points to match the
/// world points as closely as possible. This does not require any mapping
/// between the vertices, and instead just iteratively builds a correspondance.
Eigen::Isometry3s iterativeClosestPoint(
    std::vector<Eigen::Vector3s> localPoints,
    std::vector<Eigen::Vector3s> worldPoints,
    Eigen::Isometry3s transform,
    bool verbose)
{
  Eigen::Isometry3s T = transform;
  s_t lastAvgError = std::numeric_limits<s_t>::infinity();
  for (int i = 0; i < 10; i++)
  {
    // If the meshes don't match and the world is missing detail (for example,
    // if we're aligning skeleton meshes and localPoints have arms, and the
    // worldPoints does not), then we want to make sure that we're not using the
    // localPoints arms to match to the worldPoints torso. So we need to make
    // sure that we're only matching points that are close to each other.

    std::vector<Eigen::Vector3s> localPointsToMatch;
    std::vector<Eigen::Vector3s> worldPointsToMatch;
    std::vector<s_t> weights;

    s_t totalDist = 0.0;
    for (int j = 0; j < localPoints.size(); j++)
    {
      Eigen::Vector3s& localPoint = localPoints[j];
      Eigen::Vector3s worldPointGuess = T * localPoint;
      s_t bestDistance = std::numeric_limits<double>::infinity();
      Eigen::Vector3s bestWorldPoint = Eigen::Vector3s::Zero();

      for (int k = 0; k < worldPoints.size(); k++)
      {
        Eigen::Vector3s& worldPoint = worldPoints[k];
        s_t dist = (worldPoint - worldPointGuess).squaredNorm();
        if (dist < bestDistance)
        {
          bestDistance = dist;
          bestWorldPoint = worldPoint;
        }
      }

      if (bestDistance < 0.05)
      {
        localPointsToMatch.push_back(localPoint);
        worldPointsToMatch.push_back(bestWorldPoint);
        weights.push_back(1.0);
        totalDist += bestDistance;
      }
    }

    s_t avgError = (totalDist / localPointsToMatch.size());
    if (verbose)
    {
      std::cout << "ICP Iteration " << i << " with "
                << localPointsToMatch.size() << " points matched, avg dist "
                << avgError << std::endl;
    }

    if (avgError >= lastAvgError)
    {
      // We're either not getting better, or we're getting worse, so terminate
      break;
    }
    lastAvgError = avgError;

    Eigen::Isometry3s newT = getPointCloudToPointCloudTransform(
        localPointsToMatch, worldPointsToMatch, weights);
    T = newT;
  }
  return T;
}

BoundingBox::BoundingBox() : mMin(0, 0, 0), mMax(0, 0, 0)
{
}
BoundingBox::BoundingBox(const Eigen::Vector3s& min, const Eigen::Vector3s& max)
  : mMin(min), mMax(max)
{
}

} // namespace math
} // namespace dart
