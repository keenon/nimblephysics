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

#ifndef DART_MATH_GEOMETRY_HPP_
#define DART_MATH_GEOMETRY_HPP_

#include <Eigen/Dense>

#include "dart/common/Deprecated.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

/// \brief
Eigen::Matrix3s makeSkewSymmetric(const Eigen::Vector3s& _v);

/// \brief
Eigen::Vector3s fromSkewSymmetric(const Eigen::Matrix3s& _m);

//------------------------------------------------------------------------------
/// \brief
Eigen::Quaternion_s expToQuat(const Eigen::Vector3s& _v);

/// \brief
Eigen::Vector3s quatToExp(const Eigen::Quaternion_s& _q);

/// \brief
Eigen::Vector3s rotatePoint(
    const Eigen::Quaternion_s& _q, const Eigen::Vector3s& _pt);

/// \brief
Eigen::Vector3s rotatePoint(
    const Eigen::Quaternion_s& _q, s_t _x, s_t _y, s_t _z);

/// \brief
Eigen::Matrix3s quatDeriv(const Eigen::Quaternion_s& _q, int _el);

/// \brief
Eigen::Matrix3s quatSecondDeriv(
    const Eigen::Quaternion_s& _q, int _el1, int _el2);

//------------------------------------------------------------------------------
/// \brief Given Euler XYX angles, return a 3x3 rotation matrix, which is
/// equivalent to RotX(angle(0)) * RotY(angle(1)) * RotX(angle(2)).
Eigen::Matrix3s eulerXYXToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerXYZ angles, return a 3x3 rotation matrix, which is
/// equivalent to RotX(angle(0)) * RotY(angle(1)) * RotZ(angle(2)).
Eigen::Matrix3s eulerXYZToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerXZX angles, return a 3x3 rotation matrix, which is
/// equivalent to RotX(angle(0)) * RotZ(angle(1)) * RotX(angle(2)).
Eigen::Matrix3s eulerXZXToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerXZY angles, return a 3x3 rotation matrix, which is
/// equivalent to RotX(angle(0)) * RotZ(angle(1)) * RotY(angle(2)).
Eigen::Matrix3s eulerXZYToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerYXY angles, return a 3x3 rotation matrix, which is
/// equivalent to RotY(angle(0)) * RotX(angle(1)) * RotY(angle(2)).
Eigen::Matrix3s eulerYXYToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerYXZ angles, return a 3x3 rotation matrix, which is
/// equivalent to RotY(angle(0)) * RotX(angle(1)) * RotZ(angle(2)).
Eigen::Matrix3s eulerYXZToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerYZX angles, return a 3x3 rotation matrix, which is
/// equivalent to RotY(angle(0)) * RotZ(angle(1)) * RotX(angle(2)).
Eigen::Matrix3s eulerYZXToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerYZY angles, return a 3x3 rotation matrix, which is
/// equivalent to RotY(angle(0)) * RotZ(angle(1)) * RotY(angle(2)).
Eigen::Matrix3s eulerYZYToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerZXY angles, return a 3x3 rotation matrix, which is
/// equivalent to RotZ(angle(0)) * RotX(angle(1)) * RotY(angle(2)).
Eigen::Matrix3s eulerZXYToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerZYX angles, return a 3x3 rotation matrix, which is
/// equivalent to RotZ(angle(0)) * RotY(angle(1)) * RotX(angle(2)).
/// singularity : angle[1] = -+ 0.5*PI
Eigen::Matrix3s eulerZYXToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerZXZ angles, return a 3x3 rotation matrix, which is
/// equivalent to RotZ(angle(0)) * RotX(angle(1)) * RotZ(angle(2)).
Eigen::Matrix3s eulerZXZToMatrix(const Eigen::Vector3s& _angle);

/// \brief Given EulerZYZ angles, return a 3x3 rotation matrix, which is
/// equivalent to RotZ(angle(0)) * RotY(angle(1)) * RotZ(angle(2)).
/// singularity : angle[1] = 0, PI
Eigen::Matrix3s eulerZYZToMatrix(const Eigen::Vector3s& _angle);

//------------------------------------------------------------------------------
/// \brief get the Euler XYX angle from R
Eigen::Vector3s matrixToEulerXYX(const Eigen::Matrix3s& _R);

/// \brief get the Euler XYZ angle from R
Eigen::Vector3s matrixToEulerXYZ(const Eigen::Matrix3s& _R);

///// \brief get the Euler XZX angle from R
// Eigen::Vector3s matrixToEulerXZX(const Eigen::Matrix3s& R);

/// \brief get the Euler XZY angle from R
Eigen::Vector3s matrixToEulerXZY(const Eigen::Matrix3s& _R);

///// \brief get the Euler YXY angle from R
// Eigen::Vector3s matrixToEulerYXY(const Eigen::Matrix3s& R);

/// \brief get the Euler YXZ angle from R
Eigen::Vector3s matrixToEulerYXZ(const Eigen::Matrix3s& _R);

/// \brief get the Euler YZX angle from R
Eigen::Vector3s matrixToEulerYZX(const Eigen::Matrix3s& _R);

///// \brief get the Euler YZY angle from R
// Eigen::Vector3s matrixToEulerYZY(const Eigen::Matrix3s& R);

/// \brief get the Euler ZXY angle from R
Eigen::Vector3s matrixToEulerZXY(const Eigen::Matrix3s& _R);

/// \brief get the Euler ZYX angle from R
Eigen::Vector3s matrixToEulerZYX(const Eigen::Matrix3s& _R);

///// \brief get the Euler ZXZ angle from R
// Eigen::Vector3s matrixToEulerZXZ(const Eigen::Matrix3s& R);

///// \brief get the Euler ZYZ angle from R
// Eigen::Vector3s matrixToEulerZYZ(const Eigen::Matrix3s& R);

//------------------------------------------------------------------------------

/// Returns the Jacobian of an SO(3) element w.r.t. its exponential
/// coordinates where the Jacobian maps the time derivative of the exponential
/// coordinates to the angular velocity in the world frame.
Eigen::Matrix3s so3LeftJacobian(const Eigen::Vector3s& w);

/// Returns the Jacobian of an SO(3) element w.r.t. its exponential
/// coordinates where the Jacobian maps the time derivative of the exponential
/// coordinates to the angular velocity in the body frame.
Eigen::Matrix3s so3RightJacobian(const Eigen::Vector3s& w);

/// Returns the time derivative of the left Jacobian of SO(3).
Eigen::Matrix3s so3LeftJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq);

/// Returns the time derivative of the right Jacobian of SO(3).
Eigen::Matrix3s so3RightJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq);

/// \brief Exponential mapping
Eigen::Isometry3s expMap(const Eigen::Vector6s& _S);

/// \brief Exponential mapping, DART style. This treats the exponentiation
/// operation as a rotation, and then a translation, rather than an integration
/// of a screw.
Eigen::Isometry3s expMapDart(const Eigen::Vector6s& _S);

/// \brief fast version of Exp(se3(s, 0))
/// \todo This expAngular() can be replaced by Eigen::AngleAxis() but we need
/// to verify that they have exactly same functionality.
/// See: https://github.com/dartsim/dart/issues/88
Eigen::Isometry3s expAngular(const Eigen::Vector3s& _s);

/// \brief Computes the Rotation matrix from a given expmap vector.
Eigen::Matrix3s expMapRot(const Eigen::Vector3s& _expmap);

/// \brief Computes the Jacobian of the expmap
Eigen::Matrix3s expMapJac(const Eigen::Vector3s& _expmap);

/// Returns the Jacobian of an SO(3) element w.r.t. its exponential
/// coordinates where the Jacobian maps the time derivative of the exponential
/// coordinates to the angular velocity in the world frame.
Eigen::Matrix3s so3LeftJacobian(const Eigen::Vector3s& w);

/// Returns the Jacobian of an SO(3) element w.r.t. its exponential
/// coordinates where the Jacobian maps the time derivative of the exponential
/// coordinates to the angular velocity in the body frame.
Eigen::Matrix3s so3RightJacobian(const Eigen::Vector3s& w);

/// \brief Computes the Jacobian of the logMap(R * expMapRot(expMap))
Eigen::Matrix3s expMapJacAt(
    const Eigen::Vector3s& _expmap, const Eigen::Matrix3s& R);

/// \brief Computes the time derivative of the expmap Jacobian.
Eigen::Matrix3s expMapJacDot(
    const Eigen::Vector3s& _expmap, const Eigen::Vector3s& _qdot);

Eigen::Matrix3s so3LeftJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq);

Eigen::Matrix3s so3RightJacobianTimeDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq);

Eigen::Matrix3s so3RightJacobianTimeDerivDeriv(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq, int index);

Eigen::Matrix3s so3RightJacobianTimeDerivDeriv2(
    const Eigen::Vector3s& q, const Eigen::Vector3s& dq, int index);

/// \brief computes the derivative of the Jacobian of the expmap wrt to _qi
/// indexed dof; _qi \f$ \in \f$ {0,1,2}
Eigen::Matrix3s expMapJacDeriv(const Eigen::Vector3s& _expmap, int _qi);

/// \brief computes the gradient of logMap(expMapRot()) wrt to _qi
/// indexed dof; _qi \f$ \in \f$ {0,1,2}
Eigen::Vector3s expMapGradient(const Eigen::Vector3s& pos, int _qi);

/// \brief computes the gradient of logMap(expMapRot(screw * eps) *
/// expMapRot(original)) wrt to eps
Eigen::Vector3s expMapNestedGradient(
    const Eigen::Vector3s& original, const Eigen::Vector3s& screw);

/// \brief Log mapping
/// \note When @f$|Log(R)| = @pi@f$, Exp(LogR(R) = Exp(-Log(R)).
/// The implementation returns only the positive one.
Eigen::Vector3s logMap(const Eigen::Matrix3s& _R);

/// \brief Log mapping
Eigen::Vector6s logMap(const Eigen::Isometry3s& _T);

/// This takes a screw axis and a point, and gives us the direction that the
/// point will move if we increase theta by an infinitesimal amount.
Eigen::Vector3s gradientWrtTheta(
    const Eigen::Vector6s& screwAxis, const Eigen::Vector3s& point, s_t theta);

/// This takes a rotation axis and a point, and gives us the direction that the
/// point will move if we increase the theta by an infinitesimal amount.
Eigen::Vector3s gradientWrtThetaPureRotation(
    const Eigen::Vector3s& omega, const Eigen::Vector3s& point, s_t theta);

/// This returns the average of the points on edge A and edge B closest to each
/// other.
Eigen::Vector3s getContactPoint(
    const Eigen::Vector3s& edgeAPoint,
    const Eigen::Vector3s& edgeADir,
    const Eigen::Vector3s& edgeBPoint,
    const Eigen::Vector3s& edgeBDir,
    s_t radiusA = 1.0,
    s_t radiusB = 1.0);

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
    s_t radiusA = 1.0,
    s_t radiusB = 1.0);

Eigen::VectorXs dampedPInv(
    const Eigen::MatrixXs& J, const Eigen::VectorXs& x, s_t damping = 0.05);

bool hasTinySingularValues(
    const Eigen::MatrixXs& J, s_t clippingThreshold = 1e-4);

Eigen::MatrixXs clippedSingularsPinv(
    const Eigen::MatrixXs& J, s_t clippingThreshold = 1e-4);

//------------------------------------------------------------------------------
/// \brief Rectify the rotation part so as that it satifies the orthogonality
/// condition.
///
/// It is one step of @f$R_{i_1}=1/2(R_i + R_i^{-T})@f$.
/// Hence by calling this function iterativley, you can make the rotation part
/// closer to SO(3).
// SE3 Normalize(const SE3& T);

/// \brief reparameterize such as ||s'|| < M_PI and Exp(s) == Epx(s')
// Axis Reparameterize(const Axis& s);

//------------------------------------------------------------------------------
/// \brief adjoint mapping
/// \note @f$Ad_TV = ( Rw@,, ~p @times Rw + Rv)@f$,
/// where @f$T=(R,p)@in SE(3), @quad V=(w,v)@in se(3) @f$.
Eigen::Vector6s AdT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V);

/// \brief Get linear transformation matrix of Adjoint mapping
Eigen::Matrix6s getAdTMatrix(const Eigen::Isometry3s& T);

// TODO(JS): Rename and add documentation
Eigen::Matrix6s AdTMatrix(const Eigen::Isometry3s& T);
Eigen::Matrix6s AdInvTMatrix(const Eigen::Isometry3s& T);

// TODO(JS): Rename and add documentation
Eigen::Matrix6s dAdTMatrix(const Eigen::Isometry3s& T);
Eigen::Matrix6s dAdInvTMatrix(const Eigen::Isometry3s& T);

/// Adjoint mapping for dynamic size Jacobian
template <typename Derived>
typename Derived::PlainObject AdTJac(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  // Check the number of rows is 6 at compile time
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  // Compute AdT column by column
  for (int i = 0; i < _J.cols(); ++i)
    ret.col(i) = AdT(_T, _J.col(i));

  return ret;
}

/// Adjoint mapping for fixed size Jacobian
template <typename Derived>
typename Derived::PlainObject AdTJacFixed(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  // Check if _J is fixed size Jacobian
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);

  // Check the number of rows is 6 at compile time
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  // Compute AdT
  ret.template topRows<3>().noalias() = _T.linear() * _J.template topRows<3>();
  ret.template bottomRows<3>().noalias()
      = -ret.template topRows<3>().colwise().cross(_T.translation())
        + _T.linear() * _J.template bottomRows<3>();

  return ret;
}

/// \brief Fast version of Ad([R 0; 0 1], V)
Eigen::Vector6s AdR(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V);

/// \brief fast version of Ad(T, se3(w, 0))
Eigen::Vector6s AdTAngular(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _w);

/// \brief fast version of Ad(T, se3(0, v))
Eigen::Vector6s AdTLinear(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _v);

///// \brief fast version of Ad([I p; 0 1], V)
// se3 AdP(const Vec3& p, const se3& s);

/// \brief Change coordinate Frame of a Jacobian
template <typename Derived>
typename Derived::PlainObject AdRJac(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  ret.template topRows<3>().noalias() = _T.linear() * _J.template topRows<3>();

  ret.template bottomRows<3>().noalias()
      = _T.linear() * _J.template bottomRows<3>();

  return ret;
}

template <typename Derived>
typename Derived::PlainObject AdRInvJac(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  ret.template topRows<3>().noalias()
      = _T.linear().transpose() * _J.template topRows<3>();

  ret.template bottomRows<3>().noalias()
      = _T.linear().transpose() * _J.template bottomRows<3>();

  return ret;
}

template <typename Derived>
typename Derived::PlainObject adJac(
    const Eigen::Vector6s& _V, const Eigen::MatrixBase<Derived>& _J)
{
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  ret.template topRows<3>().noalias()
      = -_J.template topRows<3>().colwise().cross(_V.head<3>());

  ret.template bottomRows<3>().noalias()
      = -_J.template bottomRows<3>().colwise().cross(_V.head<3>())
        - _J.template topRows<3>().colwise().cross(_V.tail<3>());

  return ret;
}

/// \brief fast version of Ad(Inv(T), V)
Eigen::Vector6s AdInvT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _V);

/// Adjoint mapping for dynamic size Jacobian
template <typename Derived>
typename Derived::PlainObject AdInvTJac(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  // Check the number of rows is 6 at compile time
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  // Compute AdInvT column by column
  for (int i = 0; i < _J.cols(); ++i)
    ret.col(i) = AdInvT(_T, _J.col(i));

  return ret;
}

/// Adjoint mapping for fixed size Jacobian
template <typename Derived>
typename Derived::PlainObject AdInvTJacFixed(
    const Eigen::Isometry3s& _T, const Eigen::MatrixBase<Derived>& _J)
{
  // Check if _J is fixed size Jacobian
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);

  // Check the number of rows is 6 at compile time
  EIGEN_STATIC_ASSERT(
      Derived::RowsAtCompileTime == 6,
      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::PlainObject ret(_J.rows(), _J.cols());

  // Compute AdInvT
  ret.template topRows<3>().noalias()
      = _T.linear().transpose() * _J.template topRows<3>();
  ret.template bottomRows<3>().noalias()
      = _T.linear().transpose()
        * (_J.template bottomRows<3>()
           + _J.template topRows<3>().colwise().cross(_T.translation()));

  return ret;
}

///// \brief fast version of Ad(Inv(T), se3(Eigen_Vec3(0), v))
// Eigen::Vector3s AdInvTLinear(const Eigen::Isometry3s& T,
//                             const Eigen::Vector3s& v);

///// \brief fast version of Ad(Inv(T), se3(w, Eigen_Vec3(0)))
// Axis AdInvTAngular(const SE3& T, const Axis& w);

///// \brief Fast version of Ad(Inv([R 0; 0 1]), V)
// se3 AdInvR(const SE3& T, const se3& V);

/// \brief Fast version of Ad(Inv([R 0; 0 1]), se3(0, v))
Eigen::Vector6s AdInvRLinear(
    const Eigen::Isometry3s& _T, const Eigen::Vector3s& _v);

/// \brief dual adjoint mapping
/// \note @f$Ad^{@,*}_TF = ( R^T (m - p@times f)@,,~ R^T f)@f$,
/// where @f$T=(R,p)@in SE(3), F=(m,f)@in se(3)^*@f$.
Eigen::Vector6s dAdT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F);

///// \brief fast version of Ad(Inv(T), dse3(Eigen_Vec3(0), F))
// dse3 dAdTLinear(const SE3& T, const Vec3& F);

/// \brief fast version of dAd(Inv(T), F)
Eigen::Vector6s dAdInvT(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F);

/// \brief fast version of dAd(Inv([R 0; 0 1]), F)
Eigen::Vector6s dAdInvR(const Eigen::Isometry3s& _T, const Eigen::Vector6s& _F);

///// \brief fast version of dAd(Inv(SE3(p)), dse3(Eigen_Vec3(0), F))
// dse3 dAdInvPLinear(const Vec3& p, const Vec3& F);

/// \brief adjoint mapping
/// \note @f$ad_X Y = ( w_X @times w_Y@,,~w_X @times v_Y - w_Y @times v_X),@f$,
/// where @f$X=(w_X,v_X)@in se(3), @quad Y=(w_Y,v_Y)@in se(3) @f$.
Eigen::Vector6s ad(const Eigen::Vector6s& _X, const Eigen::Vector6s& _Y);

// TODO(JS): Rename and add documentation
Eigen::Matrix6s adMatrix(const Eigen::Vector6s& X);

/// \brief fast version of ad(se3(Eigen_Vec3(0), v), S)
// Vec3 ad_Vec3_se3(const Vec3& v, const se3& S);

/// \brief fast version of ad(se3(w, 0), se3(v, 0)) -> check
// Axis ad_Axis_Axis(const Axis& w, const Axis& v);

/// \brief dual adjoint mapping
/// \note @f$ad^{@,*}_V F = (m @times w + f @times v@,,~ f @times w),@f$
/// , where @f$F=(m,f)@in se^{@,*}(3), @quad V=(w,v)@in se(3) @f$.
Eigen::Vector6s dad(const Eigen::Vector6s& _s, const Eigen::Vector6s& _t);

/// \brief
Inertia transformInertia(const Eigen::Isometry3s& _T, const Inertia& _AI);

/// Use the Parallel Axis Theorem to compute the moment of inertia of a body
/// whose center of mass has been shifted from the origin
Eigen::Matrix3s parallelAxisTheorem(
    const Eigen::Matrix3s& _original,
    const Eigen::Vector3s& _comShift,
    s_t _mass);

enum AxisType
{
  AXIS_X = 0,
  AXIS_Y = 1,
  AXIS_Z = 2
};

/// Compute a rotation matrix from a vector. One axis of the rotated coordinates
/// by the rotation matrix matches the input axis where the axis is specified
/// by axisType.
Eigen::Matrix3s computeRotation(
    const Eigen::Vector3s& axis, AxisType axisType = AxisType::AXIS_X);

/// Compute a transform from a vector and a position. The rotation of the result
/// transform is computed by computeRotationMatrix(), and the translation is
/// just the input translation.
Eigen::Isometry3s computeTransform(
    const Eigen::Vector3s& axis,
    const Eigen::Vector3s& translation,
    AxisType axisType = AxisType::AXIS_X);

/// Generate frame given origin and z-axis
DART_DEPRECATED(6.0)
Eigen::Isometry3s getFrameOriginAxisZ(
    const Eigen::Vector3s& _origin, const Eigen::Vector3s& _axisZ);

/// \brief Check if determinant of _R is equat to 1 and all the elements are not
/// NaN values.
bool verifyRotation(const Eigen::Matrix3s& _R);

/// \brief Check if determinant of the rotational part of _T is equat to 1 and
/// all the elements are not NaN values.
bool verifyTransform(const Eigen::Isometry3s& _T);

/// Compute the angle (in the range of -pi to +pi) which ignores any full
/// rotations
#ifdef DART_USE_ARBITRARY_PRECISION
inline s_t wrapToPi(s_t angle)
{
  s_t pi = constantsd::pi();
  return fmod(angle + pi, 2 * pi) - pi;
}
#else
inline s_t wrapToPi(s_t angle)
{
  constexpr auto pi = constantsd::pi();

  return std::fmod(angle + pi, 2 * pi) - pi;
}
#endif

template <typename MatrixType, typename ReturnType>
void extractNullSpace(const Eigen::JacobiSVD<MatrixType>& _SVD, ReturnType& _NS)
{
  int rank = 0;
  // TODO(MXG): Replace this with _SVD.rank() once the latest Eigen is released
  if (_SVD.nonzeroSingularValues() > 0)
  {
    s_t thresh = max(
        _SVD.singularValues().coeff(0) * 1e-10,
        std::numeric_limits<s_t>::min());
    int i = _SVD.nonzeroSingularValues() - 1;
    while (i >= 0 && _SVD.singularValues().coeff(i) < thresh)
      --i;
    rank = i + 1;
  }

  int cols = _SVD.matrixV().cols(), rows = _SVD.matrixV().rows();
  _NS = _SVD.matrixV().block(0, rank, rows, cols - rank);
}

template <typename MatrixType, typename ReturnType>
void computeNullSpace(const MatrixType& _M, ReturnType& _NS)
{
  Eigen::JacobiSVD<MatrixType> svd(_M, Eigen::ComputeFullV);
  extractNullSpace(svd, _NS);
}

typedef std::vector<Eigen::Vector3s> SupportGeometry;

typedef common::aligned_vector<Eigen::Vector2s> SupportPolygon;

/// Project the support geometry points onto a plane with the given axes
/// and then compute their convex hull, which will take the form of a polgyon.
/// _axis1 and _axis2 must both have unit length for this function to work
/// correctly.
SupportPolygon computeSupportPolgyon(
    const SupportGeometry& _geometry,
    const Eigen::Vector3s& _axis1 = Eigen::Vector3s::UnitX(),
    const Eigen::Vector3s& _axis2 = Eigen::Vector3s::UnitY());

/// Same as computeSupportPolgyon, except you can pass in a
/// std::vector<std::size_t> which will have the same size as the returned
/// SupportPolygon, and each entry will contain the original index of each point
/// in the SupportPolygon
SupportPolygon computeSupportPolgyon(
    std::vector<std::size_t>& _originalIndices,
    const SupportGeometry& _geometry,
    const Eigen::Vector3s& _axis1 = Eigen::Vector3s::UnitX(),
    const Eigen::Vector3s& _axis2 = Eigen::Vector3s::UnitY());

/// Computes the convex hull of a set of 2D points
SupportPolygon computeConvexHull(const SupportPolygon& _points);

/// Computes the convex hull of a set of 2D points and fills in _originalIndices
/// with the original index of each entry in the returned SupportPolygon
SupportPolygon computeConvexHull(
    std::vector<std::size_t>& _originalIndices, const SupportPolygon& _points);

/// Compute the centroid of a polygon, assuming the polygon is a convex hull
Eigen::Vector2s computeCentroidOfHull(const SupportPolygon& _convexHull);

/// Intersection_t is returned by the computeIntersection() function to indicate
/// whether there was a valid intersection between the two line segments
enum IntersectionResult
{

  INTERSECTING = 0, ///< An intersection was found
  PARALLEL,         ///< The line segments are parallel
  BEYOND_ENDPOINTS  ///< There is no intersection because the end points do not
                    ///< expand far enough

};

/// Compute the intersection between a line segment that goes from a1 -> a2 and
/// a line segment that goes from b1 -> b2.
IntersectionResult computeIntersection(
    Eigen::Vector2s& _intersectionPoint,
    const Eigen::Vector2s& a1,
    const Eigen::Vector2s& a2,
    const Eigen::Vector2s& b1,
    const Eigen::Vector2s& b2);

/// Compute a 2D cross product
s_t cross(const Eigen::Vector2s& _v1, const Eigen::Vector2s& _v2);

/// Returns true if the point _p is inside the support polygon
bool isInsideSupportPolygon(
    const Eigen::Vector2s& _p,
    const SupportPolygon& _support,
    bool _includeEdge = true);

/// Returns the point which is closest to _p that also lays on the line segment
/// that goes from _s1 -> _s2
Eigen::Vector2s computeClosestPointOnLineSegment(
    const Eigen::Vector2s& _p,
    const Eigen::Vector2s& _s1,
    const Eigen::Vector2s& _s2);

/// Returns the point which is closest to _p that also lays on the edge of the
/// support polygon
Eigen::Vector2s computeClosestPointOnSupportPolygon(
    const Eigen::Vector2s& _p, const SupportPolygon& _support);

/// Same as closestPointOnSupportPolygon, but also fills in _index1 and _index2
/// with the indices of the line segment
Eigen::Vector2s computeClosestPointOnSupportPolygon(
    std::size_t& _index1,
    std::size_t& _index2,
    const Eigen::Vector2s& _p,
    const SupportPolygon& _support);

/// This computes and returns the closest point on a line, given by a point and
/// a direction, to a goal point
Eigen::Vector3s closestPointOnLine(
    const Eigen::Vector3s& pointOnLine,
    const Eigen::Vector3s& lineDirection,
    const Eigen::Vector3s& goalPoint);

/// This computes and returns the gradient of closestPointOnLine(), given the
/// gradients of its inputs
Eigen::Vector3s closestPointOnLineGradient(
    const Eigen::Vector3s& pointOnLine,
    const Eigen::Vector3s& pointOnLineGradient,
    const Eigen::Vector3s& lineDirection,
    const Eigen::Vector3s& lineDirectionGradient,
    const Eigen::Vector3s& goalPoint,
    const Eigen::Vector3s& goalPointGradient);

// Represents a bounding box with minimum and maximum coordinates.
class BoundingBox
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BoundingBox();
  BoundingBox(const Eigen::Vector3s& min, const Eigen::Vector3s& max);

  inline const Eigen::Vector3s& getMin() const
  {
    return mMin;
  }
  inline const Eigen::Vector3s& getMax() const
  {
    return mMax;
  }

  inline void setMin(const Eigen::Vector3s& min)
  {
    mMin = min;
  }
  inline void setMax(const Eigen::Vector3s& max)
  {
    mMax = max;
  }

  // \brief Centroid of the bounding box (i.e average of min and max)
  inline Eigen::Vector3s computeCenter() const
  {
    return (mMax + mMin) * 0.5;
  }
  // \brief Coordinates of the maximum corner with respect to the centroid.
  inline Eigen::Vector3s computeHalfExtents() const
  {
    return (mMax - mMin) * 0.5;
  }
  // \brief Length of each of the sides of the bounding box.
  inline Eigen::Vector3s computeFullExtents() const
  {
    return (mMax - mMin);
  }

protected:
  // \brief minimum coordinates of the bounding box
  Eigen::Vector3s mMin;
  // \brief maximum coordinates of the bounding box
  Eigen::Vector3s mMax;
};

} // namespace math
} // namespace dart

#endif // DART_MATH_GEOMETRY_HPP_
