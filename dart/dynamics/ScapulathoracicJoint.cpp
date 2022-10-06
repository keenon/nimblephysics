#include "dart/dynamics/ScapulathoracicJoint.hpp"

#include <memory>
#include <ostream>
#include <string>

#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

ScapulathoracicJoint::ScapulathoracicJoint(
    const detail::GenericJointProperties<math::RealVectorSpace<4>>& props)
  : GenericJoint<math::RealVectorSpace<4>>(props),
    mAxisOrder(EulerJoint::AxisOrder::XYZ), // we want YXZ
    mFlipAxisMap(Eigen::Vector4s::Ones()),
    mEllipsoidRadii(Eigen::Vector3s::Ones()),
    mWingingAxisOffset(Eigen::Vector2s::Zero()),
    mWingingAxisDirection(0.0)
{
}

//==============================================================================
const std::string& ScapulathoracicJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& ScapulathoracicJoint::getStaticType()
{
  static const std::string name = "EllipsoidJoint";
  return name;
}

//==============================================================================
bool ScapulathoracicJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
void ScapulathoracicJoint::setAxisOrder(
    EulerJoint::AxisOrder _order, bool _renameDofs)
{
  mAxisOrder = _order;
  if (_renameDofs)
    updateDegreeOfFreedomNames();

  Joint::notifyPositionUpdated();
  updateRelativeJacobian(true);
  Joint::incrementVersion();
}

//==============================================================================
EulerJoint::AxisOrder ScapulathoracicJoint::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void ScapulathoracicJoint::setFlipAxisMap(Eigen::Vector4s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector4s ScapulathoracicJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
void ScapulathoracicJoint::setEllipsoidRadii(Eigen::Vector3s radii)
{
  mEllipsoidRadii = radii;
}

//==============================================================================
Eigen::Vector3s ScapulathoracicJoint::getEllipsoidRadii() const
{
  return mEllipsoidRadii;
}

//==============================================================================
void ScapulathoracicJoint::setWingingAxisOffset(Eigen::Vector2s offset)
{
  mWingingAxisOffset = offset;
}

//==============================================================================
Eigen::Vector2s ScapulathoracicJoint::getWingingAxisOffset() const
{
  return mWingingAxisOffset;
}

//==============================================================================
void ScapulathoracicJoint::setWingingAxisDirection(s_t radians)
{
  mWingingAxisDirection = radians;
}

//==============================================================================
s_t ScapulathoracicJoint::getWingingAxisDirection() const
{
  return mWingingAxisDirection;
}

//==============================================================================
dart::dynamics::Joint* ScapulathoracicJoint::clone() const
{
  ScapulathoracicJoint* joint
      = new ScapulathoracicJoint(this->getJointProperties());
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(getFlipAxisMap());
  joint->setAxisOrder(getAxisOrder());
  joint->setName(this->getName());
  joint->setEllipsoidRadii(this->getEllipsoidRadii());
  joint->setWingingAxisDirection(this->getWingingAxisDirection());
  joint->setWingingAxisOffset(this->getWingingAxisOffset());
  joint->setPositionUpperLimits(this->getPositionUpperLimits());
  joint->setPositionLowerLimits(this->getPositionLowerLimits());
  joint->setVelocityUpperLimits(this->getVelocityUpperLimits());
  joint->setVelocityLowerLimits(this->getVelocityLowerLimits());
  return joint;
}

//==============================================================================
dart::dynamics::Joint* ScapulathoracicJoint::simplifiedClone() const
{
  // TOOD: we need to actually find a good simplification for this joint in
  // terms of simpler joint types, maybe a ball joint with an offset?
  assert(false);
  return clone();
}

//==============================================================================
void ScapulathoracicJoint::updateDegreeOfFreedomNames()
{
  if (!this->mDofs[0]->isNamePreserved())
    this->mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
void ScapulathoracicJoint::updateRelativeTransform() const
{
  Eigen::Vector4s pos = this->getPositionsStatic();

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  // 2. Rotation on the ellipsoid surface:
  // We have an euler XYZ ball, which we will use to rotate a unit vector. Then
  // after the unit vector has been rotated, we'll component-wise scale it to
  // get an ellipsoid.

  // 2.1. Do this XYZ rotation in +90Z space
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>());
  rot.linear() = eulerR.transpose() * rot.linear() * eulerR;

  // 2.2. Now that we have a rotation in `rot`, we need to rotate a unit vector.
  // We will end up with a translation on the surface of a unit sphere.
  Eigen::Isometry3s ballSurface = Eigen::Isometry3s::Identity();
  ballSurface.translation() = Eigen::Vector3s::UnitZ();
  ballSurface = rot * ballSurface;

  // 2.3. Scale the translation to make the sphere into an ellipsoid
  ballSurface.translation()
      = ballSurface.translation().cwiseProduct(mEllipsoidRadii);

  // 3. Situate relative to parent and child joints
  this->mT = Joint::mAspectProperties.mT_ParentBodyToJoint * ballSurface
             * winging * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4> ScapulathoracicJoint::getRelativeJacobianStatic(
    const Eigen::Vector4s& pos) const
{
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();

  // Think in terms of the child frame

  // 1. Compute the Jacobian of the Euler transformation
  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  J.block<3, 1>(3, 0) = J.block<3, 1>(0, 0).cross(localSphericalOffset);
  J.block<3, 1>(3, 1) = J.block<3, 1>(0, 1).cross(localSphericalOffset);
  J.block<3, 1>(3, 2) = J.block<3, 1>(0, 2).cross(localSphericalOffset);

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>());
  rot.linear() = eulerR.transpose() * rot.linear() * eulerR;
  Eigen::Matrix3s scaleInParentSpace
      = rot.linear().transpose() * mEllipsoidRadii.asDiagonal() * rot.linear();
  J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  J.block<3, 1>(0, 3) = wingDirection;
  J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
  J = math::AdTJacFixed(winging.inverse(), J);

  // Finally, take into account the transform to the child body node
  J = math::AdTJacFixed(getTransformFromChildBodyNode(), J);
  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4>
ScapulathoracicJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)index;
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ = Eigen::Matrix<s_t, 6, 4>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ;
  if (index < 3)
  {
    euler_dJ = EulerJoint::computeRelativeJacobianDerivWrtPos(
        index, pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  }
  else
  {
    euler_dJ.setZero();
  }
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  dJ.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  dJ.block<3, 1>(3, 0) = dJ.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ.block<3, 1>(3, 1) = dJ.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ.block<3, 1>(3, 2) = dJ.block<3, 1>(0, 2).cross(localSphericalOffset);
  J.block<3, 1>(3, 0) = J.block<3, 1>(0, 0).cross(localSphericalOffset);
  J.block<3, 1>(3, 1) = J.block<3, 1>(0, 1).cross(localSphericalOffset);
  J.block<3, 1>(3, 2) = J.block<3, 1>(0, 2).cross(localSphericalOffset);

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Matrix3s rot = EulerJoint::convertToTransform(
                            pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>())
                            .linear();
  rot = eulerR.transpose() * rot * eulerR;
  Eigen::Matrix3s scaleInParentSpace
      = rot.transpose() * mEllipsoidRadii.asDiagonal() * rot;

  if (index < 3)
  {
    Eigen::Matrix3s dRot
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, index));
    Eigen::Matrix3s dScaleInParentSpace
        = dRot.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace += dScaleInParentSpace.transpose().eval();
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0)
                           + dScaleInParentSpace * J.block<3, 3>(3, 0);
  }
  else
  {
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0);
    J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);
  }

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  if (index < 3)
  {
    dJ = math::AdTJacFixed(winging.inverse(), dJ);
  }
  else
  {
    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ.col(0) = math::ad(J.col(0), J.col(3));
    dJ.col(1) = math::ad(J.col(1), J.col(3));
    dJ.col(2) = math::ad(J.col(2), J.col(3));
  }

  // Finally, take into account the transform to the child body node
  dJ = math::AdTJacFixed(getTransformFromChildBodyNode(), dJ);

  return dJ;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4>
ScapulathoracicJoint::getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(
    std::size_t firstIndex, std::size_t secondIndex) const
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  (void)secondIndex;
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> ddJ_dFirst_dSecond
      = Eigen::Matrix<s_t, 6, 4>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dFirst;
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dSecond;
  Eigen::Matrix<s_t, 6, 3> euler_ddJ_dFirst_dSecond;
  if (firstIndex < 3)
  {
    euler_dJ_dFirst = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        mAxisOrder,
        mFlipAxisMap.head<3>(),
        identity);
  }
  else
  {
    euler_dJ_dFirst.setZero();
  }
  if (secondIndex < 3)
  {
    euler_dJ_dSecond = EulerJoint::computeRelativeJacobianDerivWrtPos(
        secondIndex,
        pos.head<3>(),
        mAxisOrder,
        mFlipAxisMap.head<3>(),
        identity);
  }
  else
  {
    euler_dJ_dSecond.setZero();
  }
  if (firstIndex < 3 && secondIndex < 3)
  {
    euler_ddJ_dFirst_dSecond
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            Eigen::Vector3s::Unit(firstIndex),
            mAxisOrder,
            mFlipAxisMap.head<3>(),
            identity);
  }
  else
  {
    euler_ddJ_dFirst_dSecond.setZero();
  }
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  ddJ_dFirst_dSecond.block<3, 3>(0, 0)
      = eulerR.transpose() * euler_ddJ_dFirst_dSecond.topRows(3);
  dJ_dSecond.block<3, 3>(0, 0)
      = eulerR.transpose() * euler_dJ_dSecond.topRows(3);
  dJ_dFirst.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ_dFirst.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  ddJ_dFirst_dSecond.block<3, 1>(3, 0)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 0).cross(localSphericalOffset);
  ddJ_dFirst_dSecond.block<3, 1>(3, 1)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 1).cross(localSphericalOffset);
  ddJ_dFirst_dSecond.block<3, 1>(3, 2)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 2).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 0)
      = dJ_dFirst.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 1)
      = dJ_dFirst.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 2)
      = dJ_dFirst.block<3, 1>(0, 2).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 0)
      = dJ_dSecond.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 1)
      = dJ_dSecond.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 2)
      = dJ_dSecond.block<3, 1>(0, 2).cross(localSphericalOffset);
  J.block<3, 1>(3, 0) = J.block<3, 1>(0, 0).cross(localSphericalOffset);
  J.block<3, 1>(3, 1) = J.block<3, 1>(0, 1).cross(localSphericalOffset);
  J.block<3, 1>(3, 2) = J.block<3, 1>(0, 2).cross(localSphericalOffset);

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Matrix3s rot = EulerJoint::convertToTransform(
                            pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>())
                            .linear();
  rot = eulerR.transpose() * rot * eulerR;
  Eigen::Matrix3s scaleInParentSpace
      = rot.transpose() * mEllipsoidRadii.asDiagonal() * rot;

  if (firstIndex < 3 && secondIndex < 3)
  {
    // euler wrt euler
    Eigen::Matrix3s dRot_dFirst
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
    Eigen::Matrix3s dScaleInParentSpace_dFirst
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();

    Eigen::Matrix3s dRot_dSecond
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex));
    Eigen::Matrix3s dScaleInParentSpace_dSecond
        = dRot_dSecond.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dSecond
        += dScaleInParentSpace_dSecond.transpose().eval();

    Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);

    Eigen::Matrix3s dRot_dFirst_dSecond
        = dRot_dFirst
              * math::makeSkewSymmetric(
                  eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex))
          + rot
                * math::makeSkewSymmetric(
                    eulerR.transpose()
                    * euler_dJ_dFirst.block<3, 1>(0, secondIndex));

    Eigen::Matrix3s ddScaleInParentSpace_dFirst_dSecond
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * dRot_dSecond
          + dRot_dFirst_dSecond.transpose() * mEllipsoidRadii.asDiagonal()
                * rot;
    ddScaleInParentSpace_dFirst_dSecond
        += ddScaleInParentSpace_dFirst_dSecond.transpose().eval();

    ddJ_dFirst_dSecond.block<3, 3>(3, 0)
        = (ddScaleInParentSpace_dFirst_dSecond * J.block<3, 3>(3, 0)
           + dScaleInParentSpace_dFirst * dJ_dSecond.block<3, 3>(3, 0))
          + (scaleInParentSpace * ddJ_dFirst_dSecond.block<3, 3>(3, 0)
             + dScaleInParentSpace_dSecond * dJ_dFirst.block<3, 3>(3, 0));

    dJ_dFirst.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dFirst.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);
    dJ_dSecond.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dSecond.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);

    ddJ_dFirst_dSecond
        = math::AdTJacFixed(winging.inverse(), ddJ_dFirst_dSecond);
  }
  else if (firstIndex < 3)
  {
    // euler wrt winging
    assert(secondIndex == 3);
    Eigen::Matrix3s dRot_dFirst
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
    Eigen::Matrix3s dScaleInParentSpace_dFirst
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();
    dJ_dFirst.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dFirst.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);

    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dFirst = math::AdTJacFixed(winging.inverse(), dJ_dFirst);
    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dFirst.col(0), J.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dFirst.col(1), J.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dFirst.col(2), J.col(3));
    ddJ_dFirst_dSecond.col(3) = math::ad(dJ_dFirst.col(3), J.col(3));
  }
  else if (secondIndex < 3)
  {
    // winging wrt euler
    assert(firstIndex == 3);
    ddJ_dFirst_dSecond.setZero();

    Eigen::Matrix3s dRot_dSecond
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex));
    Eigen::Matrix3s dScaleInParentSpace_dSecond
        = dRot_dSecond.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dSecond
        += dScaleInParentSpace_dSecond.transpose().eval();
    dJ_dSecond.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dSecond.block<3, 3>(3, 0)
          + dScaleInParentSpace_dSecond * J.block<3, 3>(3, 0);

    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dSecond = math::AdTJacFixed(winging.inverse(), dJ_dSecond);
    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dSecond.col(0), J.col(3))
                                + math::ad(J.col(0), dJ_dSecond.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dSecond.col(1), J.col(3))
                                + math::ad(J.col(1), dJ_dSecond.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dSecond.col(2), J.col(3))
                                + math::ad(J.col(2), dJ_dSecond.col(3));
  }
  else
  {
    // winging wrt winging
    assert(firstIndex == 3);
    assert(secondIndex == 3);

    J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);
    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dFirst.col(0) = math::ad(J.col(0), J.col(3));
    dJ_dFirst.col(1) = math::ad(J.col(1), J.col(3));
    dJ_dFirst.col(2) = math::ad(J.col(2), J.col(3));

    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dFirst.col(0), J.col(3))
                                + math::ad(J.col(0), dJ_dFirst.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dFirst.col(1), J.col(3))
                                + math::ad(J.col(1), dJ_dFirst.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dFirst.col(2), J.col(3))
                                + math::ad(J.col(2), dJ_dFirst.col(3));
  }

  // Finally, take into account the transform to the child body node
  ddJ_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), ddJ_dFirst_dSecond);

  return ddJ_dFirst_dSecond;
}

//==============================================================================
void ScapulathoracicJoint::updateRelativeJacobian(bool) const
{
  this->mJacobian = getRelativeJacobianStatic(this->getPositionsStatic());
}

//==============================================================================
void ScapulathoracicJoint::updateRelativeJacobianTimeDeriv() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 4> dJ = Eigen::Matrix<s_t, 6, 4>::Zero();
  for (int i = 0; i < pos.size(); i++)
  {
    dJ += getRelativeJacobianDerivWrtPositionStatic(i) * vel(i);
  }
  this->mJacobianDeriv = dJ;
}

//==============================================================================
math::Jacobian
ScapulathoracicJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 4> ddJ = Eigen::Matrix<s_t, 6, 4>::Zero();
  for (int i = 0; i < pos.size(); i++)
  {
    ddJ += getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(i, index)
           * vel(i);
  }
  return ddJ;
}

//==============================================================================
math::Jacobian
ScapulathoracicJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return getRelativeJacobianDerivWrtPositionStatic(index);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s ScapulathoracicJoint::getScrewAxisGradientForPosition(
    int axisDof, int rotateDof)
{
  // Defaults to Finite Differencing - this is slow, but at least it's
  // approximately correct. Child joints should override with a faster
  // implementation.
  return Joint::finiteDifferenceScrewAxisGradientForPosition(
      axisDof, rotateDof);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s ScapulathoracicJoint::getScrewAxisGradientForForce(
    int axisDof, int rotateDof)
{
  // Defaults to Finite Differencing - this is slow, but at least it's
  // approximately correct. Child joints should override with a faster
  // implementation.
  return Joint::finiteDifferenceScrewAxisGradientForForce(axisDof, rotateDof);
}

// For testing
Eigen::MatrixXs ScapulathoracicJoint::getScratch(int firstIndex)
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ = Eigen::Matrix<s_t, 6, 4>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ;
  if (firstIndex < 3)
  {
    euler_dJ = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        mAxisOrder,
        mFlipAxisMap.head<3>(),
        identity);
  }
  else
  {
    euler_dJ.setZero();
  }
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  dJ.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  dJ.block<3, 1>(3, 0) = dJ.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ.block<3, 1>(3, 1) = dJ.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ.block<3, 1>(3, 2) = dJ.block<3, 1>(0, 2).cross(localSphericalOffset);
  J.block<3, 1>(3, 0) = J.block<3, 1>(0, 0).cross(localSphericalOffset);
  J.block<3, 1>(3, 1) = J.block<3, 1>(0, 1).cross(localSphericalOffset);
  J.block<3, 1>(3, 2) = J.block<3, 1>(0, 2).cross(localSphericalOffset);

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Matrix3s rot = EulerJoint::convertToTransform(
                            pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>())
                            .linear();
  rot = eulerR.transpose() * rot * eulerR;
  Eigen::Matrix3s scaleInParentSpace
      = rot.transpose() * mEllipsoidRadii.asDiagonal() * rot;

  if (firstIndex < 3)
  {
    Eigen::Matrix3s dRot
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
    Eigen::Matrix3s dScaleInParentSpace
        = dRot.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace += dScaleInParentSpace.transpose().eval();
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0)
                           + dScaleInParentSpace * J.block<3, 3>(3, 0);
  }
  else
  {
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0);
    J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);
  }

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  if (firstIndex < 3)
  {
    dJ = math::AdTJacFixed(winging.inverse(), dJ);
    // return dJ;
  }
  else
  {
    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ.col(0) = math::ad(J.col(0), J.col(3));
    dJ.col(1) = math::ad(J.col(1), J.col(3));
    dJ.col(2) = math::ad(J.col(2), J.col(3));
  }

  // Finally, take into account the transform to the child body node
  dJ = math::AdTJacFixed(getTransformFromChildBodyNode(), dJ);

  return dJ;
}

Eigen::MatrixXs ScapulathoracicJoint::analyticalScratch(
    int firstIndex, int secondIndex)
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  (void)secondIndex;
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> dJ_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> ddJ_dFirst_dSecond
      = Eigen::Matrix<s_t, 6, 4>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dFirst;
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dSecond;
  Eigen::Matrix<s_t, 6, 3> euler_ddJ_dFirst_dSecond;
  if (firstIndex < 3)
  {
    euler_dJ_dFirst = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        mAxisOrder,
        mFlipAxisMap.head<3>(),
        identity);
  }
  else
  {
    euler_dJ_dFirst.setZero();
  }
  if (secondIndex < 3)
  {
    euler_dJ_dSecond = EulerJoint::computeRelativeJacobianDerivWrtPos(
        secondIndex,
        pos.head<3>(),
        mAxisOrder,
        mFlipAxisMap.head<3>(),
        identity);
  }
  else
  {
    euler_dJ_dSecond.setZero();
  }
  if (firstIndex < 3 && secondIndex < 3)
  {
    euler_ddJ_dFirst_dSecond
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            Eigen::Vector3s::Unit(firstIndex),
            mAxisOrder,
            mFlipAxisMap.head<3>(),
            identity);
  }
  else
  {
    euler_ddJ_dFirst_dSecond.setZero();
  }
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  ddJ_dFirst_dSecond.block<3, 3>(0, 0)
      = eulerR.transpose() * euler_ddJ_dFirst_dSecond.topRows(3);
  dJ_dSecond.block<3, 3>(0, 0)
      = eulerR.transpose() * euler_dJ_dSecond.topRows(3);
  dJ_dFirst.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ_dFirst.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  ddJ_dFirst_dSecond.block<3, 1>(3, 0)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 0).cross(localSphericalOffset);
  ddJ_dFirst_dSecond.block<3, 1>(3, 1)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 1).cross(localSphericalOffset);
  ddJ_dFirst_dSecond.block<3, 1>(3, 2)
      = ddJ_dFirst_dSecond.block<3, 1>(0, 2).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 0)
      = dJ_dFirst.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 1)
      = dJ_dFirst.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 2)
      = dJ_dFirst.block<3, 1>(0, 2).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 0)
      = dJ_dSecond.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 1)
      = dJ_dSecond.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dSecond.block<3, 1>(3, 2)
      = dJ_dSecond.block<3, 1>(0, 2).cross(localSphericalOffset);
  J.block<3, 1>(3, 0) = J.block<3, 1>(0, 0).cross(localSphericalOffset);
  J.block<3, 1>(3, 1) = J.block<3, 1>(0, 1).cross(localSphericalOffset);
  J.block<3, 1>(3, 2) = J.block<3, 1>(0, 2).cross(localSphericalOffset);

  // 1. Winging:
  // We need to rotate around a "winging axis", which is always perpendicular to
  // the z-axis, and may have an offset in the X,Y plane.
  Eigen::Vector3s wingDirection = Eigen::Vector3s(
      -sin(mWingingAxisDirection), cos(mWingingAxisDirection), 0);
  Eigen::Vector3s wingOriginInIntermediateFrame
      = Eigen::Vector3s(mWingingAxisOffset(0), mWingingAxisOffset(1), 0);
  Eigen::Isometry3s wingAxisT = Eigen::Isometry3s::Identity();
  wingAxisT.translation() = wingOriginInIntermediateFrame;
  Eigen::Isometry3s winging = Eigen::Isometry3s::Identity();
  winging.linear() = math::expMapRot(wingDirection * pos(3) * mFlipAxisMap(3));
  winging = wingAxisT * winging * wingAxisT.inverse();

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Matrix3s rot = EulerJoint::convertToTransform(
                            pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>())
                            .linear();
  rot = eulerR.transpose() * rot * eulerR;
  Eigen::Matrix3s scaleInParentSpace
      = rot.transpose() * mEllipsoidRadii.asDiagonal() * rot;

  if (firstIndex < 3 && secondIndex < 3)
  {
    // euler wrt euler
    Eigen::Matrix3s dRot_dFirst
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
    Eigen::Matrix3s dScaleInParentSpace_dFirst
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();

    Eigen::Matrix3s dRot_dSecond
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex));
    Eigen::Matrix3s dScaleInParentSpace_dSecond
        = dRot_dSecond.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dSecond
        += dScaleInParentSpace_dSecond.transpose().eval();

    Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);

    Eigen::Matrix3s dRot_dFirst_dSecond
        = dRot_dFirst
              * math::makeSkewSymmetric(
                  eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex))
          + rot
                * math::makeSkewSymmetric(
                    eulerR.transpose()
                    * euler_dJ_dFirst.block<3, 1>(0, secondIndex));

    Eigen::Matrix3s ddScaleInParentSpace_dFirst_dSecond
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * dRot_dSecond
          + dRot_dFirst_dSecond.transpose() * mEllipsoidRadii.asDiagonal()
                * rot;
    ddScaleInParentSpace_dFirst_dSecond
        += ddScaleInParentSpace_dFirst_dSecond.transpose().eval();

    ddJ_dFirst_dSecond.block<3, 3>(3, 0)
        = (ddScaleInParentSpace_dFirst_dSecond * J.block<3, 3>(3, 0)
           + dScaleInParentSpace_dFirst * dJ_dSecond.block<3, 3>(3, 0))
          + (scaleInParentSpace * ddJ_dFirst_dSecond.block<3, 3>(3, 0)
             + dScaleInParentSpace_dSecond * dJ_dFirst.block<3, 3>(3, 0));

    dJ_dFirst.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dFirst.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);
    dJ_dSecond.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dSecond.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);

    ddJ_dFirst_dSecond
        = math::AdTJacFixed(winging.inverse(), ddJ_dFirst_dSecond);
  }
  else if (firstIndex < 3)
  {
    // euler wrt winging
    assert(secondIndex == 3);
    Eigen::Matrix3s dRot_dFirst
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
    Eigen::Matrix3s dScaleInParentSpace_dFirst
        = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();
    dJ_dFirst.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dFirst.block<3, 3>(3, 0)
          + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);

    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dFirst = math::AdTJacFixed(winging.inverse(), dJ_dFirst);
    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dFirst.col(0), J.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dFirst.col(1), J.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dFirst.col(2), J.col(3));
    ddJ_dFirst_dSecond.col(3) = math::ad(dJ_dFirst.col(3), J.col(3));
  }
  else if (secondIndex < 3)
  {
    // winging wrt euler
    assert(firstIndex == 3);
    ddJ_dFirst_dSecond.setZero();

    Eigen::Matrix3s dRot_dSecond
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex));
    Eigen::Matrix3s dScaleInParentSpace_dSecond
        = dRot_dSecond.transpose() * mEllipsoidRadii.asDiagonal() * rot;
    dScaleInParentSpace_dSecond
        += dScaleInParentSpace_dSecond.transpose().eval();
    dJ_dSecond.block<3, 3>(3, 0)
        = scaleInParentSpace * dJ_dSecond.block<3, 3>(3, 0)
          + dScaleInParentSpace_dSecond * J.block<3, 3>(3, 0);

    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dSecond = math::AdTJacFixed(winging.inverse(), dJ_dSecond);
    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dSecond.col(0), J.col(3))
                                + math::ad(J.col(0), dJ_dSecond.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dSecond.col(1), J.col(3))
                                + math::ad(J.col(1), dJ_dSecond.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dSecond.col(2), J.col(3))
                                + math::ad(J.col(2), dJ_dSecond.col(3));
  }
  else
  {
    // winging wrt winging
    assert(firstIndex == 3);
    assert(secondIndex == 3);

    J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);
    // Finish computing J
    J.block<3, 1>(0, 3) = wingDirection;
    J.block<3, 1>(3, 3) = wingOriginInIntermediateFrame.cross(wingDirection);
    J = math::AdTJacFixed(winging.inverse(), J);

    dJ_dFirst.col(0) = math::ad(J.col(0), J.col(3));
    dJ_dFirst.col(1) = math::ad(J.col(1), J.col(3));
    dJ_dFirst.col(2) = math::ad(J.col(2), J.col(3));

    ddJ_dFirst_dSecond.col(0) = math::ad(dJ_dFirst.col(0), J.col(3))
                                + math::ad(J.col(0), dJ_dFirst.col(3));
    ddJ_dFirst_dSecond.col(1) = math::ad(dJ_dFirst.col(1), J.col(3))
                                + math::ad(J.col(1), dJ_dFirst.col(3));
    ddJ_dFirst_dSecond.col(2) = math::ad(dJ_dFirst.col(2), J.col(3))
                                + math::ad(J.col(2), dJ_dFirst.col(3));
  }

  // Finally, take into account the transform to the child body node
  ddJ_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), ddJ_dFirst_dSecond);

  return ddJ_dFirst_dSecond;
}

Eigen::MatrixXs ScapulathoracicJoint::finiteDifferenceScratch(
    int firstIndex, int secondIndex)
{
  Eigen::MatrixXs result = getScratch(firstIndex);

  Eigen::VectorXs originalPoses = getPositions();

  s_t eps = 1e-3;

  math::finiteDifference<Eigen::MatrixXs>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::MatrixXs& out) {
        Eigen::VectorXs tweaked = originalPoses;
        tweaked(secondIndex) += eps;
        setPositions(tweaked);
        out = getScratch(firstIndex);
        return true;
      },
      result,
      eps,
      true);

  setPositions(originalPoses);

  return result;
}

} // namespace dynamics
} // namespace dart