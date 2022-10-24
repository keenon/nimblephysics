#include "dart/dynamics/EllipsoidJoint.hpp"

#include <memory>
#include <ostream>
#include <string>

#include "dart/dynamics/EllipsoidJoint.hpp"
#include "dart/dynamics/EllipsoidShape.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

EllipsoidJoint::EllipsoidJoint(
    const detail::GenericJointProperties<math::RealVectorSpace<3>>& props)
  : GenericJoint<math::RealVectorSpace<3>>(props),
    mAxisOrder(EulerJoint::AxisOrder::XYZ), // we want YXZ
    mFlipAxisMap(Eigen::Vector3s::Ones()),
    mEllipsoidRadii(Eigen::Vector3s::Ones())
{
}

//==============================================================================
const std::string& EllipsoidJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& EllipsoidJoint::getStaticType()
{
  static const std::string name = "EllipsoidJoint";
  return name;
}

//==============================================================================
bool EllipsoidJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
void EllipsoidJoint::setAxisOrder(
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
EulerJoint::AxisOrder EllipsoidJoint::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void EllipsoidJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s EllipsoidJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
void EllipsoidJoint::setEllipsoidRadii(Eigen::Vector3s radii)
{
  mEllipsoidRadii = radii;
}

//==============================================================================
Eigen::Vector3s EllipsoidJoint::getEllipsoidRadii() const
{
  return mEllipsoidRadii;
}

//==============================================================================
dart::dynamics::Joint* EllipsoidJoint::clone() const
{
  EllipsoidJoint* joint = new EllipsoidJoint(this->getJointProperties());
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(getFlipAxisMap());
  joint->setAxisOrder(getAxisOrder());
  joint->setName(this->getName());
  joint->setEllipsoidRadii(this->getEllipsoidRadii());
  joint->setPositionsStatic(this->getPositionsStatic());
  joint->setInitialPositions(this->getInitialPositions());
  joint->setPositionUpperLimits(this->getPositionUpperLimits());
  joint->setPositionLowerLimits(this->getPositionLowerLimits());
  joint->setVelocityUpperLimits(this->getVelocityUpperLimits());
  joint->setVelocityLowerLimits(this->getVelocityLowerLimits());
  return joint;
}

//==============================================================================
dart::dynamics::Joint* EllipsoidJoint::simplifiedClone() const
{
  // TOOD: we need to actually find a good simplification for this joint in
  // terms of simpler joint types, maybe a ball joint with an offset?
  assert(false);
  return clone();
}

//==============================================================================
void EllipsoidJoint::updateDegreeOfFreedomNames()
{
  if (!this->mDofs[0]->isNamePreserved())
    this->mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
void EllipsoidJoint::updateRelativeTransform() const
{
  Eigen::Vector3s pos = this->getPositionsStatic();

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

  Eigen::Vector3s parentScale = getParentScale();

  // 2.3. Scale the translation to make the sphere into an ellipsoid
  ballSurface.translation() = ballSurface.translation()
                                  .cwiseProduct(mEllipsoidRadii)
                                  .cwiseProduct(parentScale);

  // 3. Situate relative to parent and child joints
  this->mT = Joint::mAspectProperties.mT_ParentBodyToJoint * ballSurface
             * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
Eigen::Isometry3s EllipsoidJoint::getRelativeTransformStatic(
    Eigen::Isometry3s parentTransform,
    Eigen::Isometry3s childTransform,
    Eigen::Vector3s parentScale,
    Eigen::Vector3s ellipsoidScale,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Vector3s position,
    Eigen::Vector3s flipAxisMap)
{
  // 2.1. Do this XYZ rotation in +90Z space
  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  Eigen::Isometry3s rot
      = EulerJoint::convertToTransform(position, axisOrder, flipAxisMap);
  rot.linear() = eulerR.transpose() * rot.linear() * eulerR;

  // 2.2. Now that we have a rotation in `rot`, we need to rotate a unit vector.
  // We will end up with a translation on the surface of a unit sphere.
  Eigen::Isometry3s ballSurface = Eigen::Isometry3s::Identity();
  ballSurface.translation() = Eigen::Vector3s::UnitZ();
  ballSurface = rot * ballSurface;

  // 2.3. Scale the translation to make the sphere into an ellipsoid
  ballSurface.translation() = ballSurface.translation()
                                  .cwiseProduct(ellipsoidScale)
                                  .cwiseProduct(parentScale);

  // 3. Situate relative to parent and child joints
  return parentTransform * ballSurface * childTransform.inverse();
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3> EllipsoidJoint::getRelativeJacobianStatic(
    const Eigen::Vector3s& pos) const
{
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();

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
  Eigen::Vector3s parentScale = getParentScale();
  Eigen::Matrix3s scaleInParentSpace
      = rot.linear().transpose() * mEllipsoidRadii.asDiagonal()
        * parentScale.asDiagonal() * rot.linear();
  J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);

  // Finally, take into account the transform to the child body node
  J = math::AdTJacFixed(getTransformFromChildBodyNode(), J);
  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EllipsoidJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)index;
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ = Eigen::Matrix<s_t, 6, 3>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          index, pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
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
  Eigen::Vector3s parentScale = getParentScale();
  Eigen::Matrix3s scaleInParentSpace = rot.transpose()
                                       * mEllipsoidRadii.asDiagonal()
                                       * parentScale.asDiagonal() * rot;

  if (index < 3)
  {
    Eigen::Matrix3s dRot
        = rot
          * math::makeSkewSymmetric(
              eulerR.transpose() * eulerJ.block<3, 1>(0, index));
    Eigen::Matrix3s dScaleInParentSpace = dRot.transpose()
                                          * mEllipsoidRadii.asDiagonal()
                                          * parentScale.asDiagonal() * rot;
    dScaleInParentSpace += dScaleInParentSpace.transpose().eval();
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0)
                           + dScaleInParentSpace * J.block<3, 3>(3, 0);
  }
  else
  {
    dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0);
    J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);
  }

  // Finally, take into account the transform to the child body node
  dJ = math::AdTJacFixed(getTransformFromChildBodyNode(), dJ);

  return dJ;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EllipsoidJoint::getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(
    std::size_t firstIndex, std::size_t secondIndex) const
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  (void)secondIndex;
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dFirst = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dSecond = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> ddJ_dFirst_dSecond
      = Eigen::Matrix<s_t, 6, 3>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dFirst;
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dSecond;
  Eigen::Matrix<s_t, 6, 3> euler_ddJ_dFirst_dSecond;
  euler_dJ_dFirst = EulerJoint::computeRelativeJacobianDerivWrtPos(
      firstIndex, pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  euler_dJ_dSecond = EulerJoint::computeRelativeJacobianDerivWrtPos(
      secondIndex, pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  euler_ddJ_dFirst_dSecond
      = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
          secondIndex,
          pos.head<3>(),
          Eigen::Vector3s::Unit(firstIndex),
          mAxisOrder,
          mFlipAxisMap.head<3>(),
          identity);
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

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Matrix3s rot = EulerJoint::convertToTransform(
                            pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>())
                            .linear();
  rot = eulerR.transpose() * rot * eulerR;

  Eigen::Vector3s parentScale = getParentScale();
  Eigen::Matrix3s scaleInParentSpace = rot.transpose()
                                       * mEllipsoidRadii.asDiagonal()
                                       * parentScale.asDiagonal() * rot;

  // euler wrt euler
  Eigen::Matrix3s dRot_dFirst
      = rot
        * math::makeSkewSymmetric(
            eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
  Eigen::Matrix3s dScaleInParentSpace_dFirst = dRot_dFirst.transpose()
                                               * mEllipsoidRadii.asDiagonal()
                                               * parentScale.asDiagonal() * rot;
  dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();

  Eigen::Matrix3s dRot_dSecond
      = rot
        * math::makeSkewSymmetric(
            eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex));
  Eigen::Matrix3s dScaleInParentSpace_dSecond
      = dRot_dSecond.transpose() * mEllipsoidRadii.asDiagonal()
        * parentScale.asDiagonal() * rot;
  dScaleInParentSpace_dSecond += dScaleInParentSpace_dSecond.transpose().eval();

  Eigen::Matrix3s dRot_dFirst_dSecond
      = dRot_dFirst
            * math::makeSkewSymmetric(
                eulerR.transpose() * eulerJ.block<3, 1>(0, secondIndex))
        + rot
              * math::makeSkewSymmetric(
                  eulerR.transpose()
                  * euler_dJ_dFirst.block<3, 1>(0, secondIndex));

  Eigen::Matrix3s ddScaleInParentSpace_dFirst_dSecond
      = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal()
            * parentScale.asDiagonal() * dRot_dSecond
        + dRot_dFirst_dSecond.transpose() * mEllipsoidRadii.asDiagonal()
              * parentScale.asDiagonal() * rot;
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

  // Finally, take into account the transform to the child body node
  ddJ_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), ddJ_dFirst_dSecond);

  return ddJ_dFirst_dSecond;
}

//==============================================================================
/// This gets the change in world translation of the child body, with respect
/// to an axis of parent scaling. Use axis = -1 for uniform scaling of all the
/// axis.
Eigen::Vector3s EllipsoidJoint::getWorldTranslationOfChildBodyWrtParentScale(
    int axis) const
{
  const dynamics::BodyNode* parentBody = getParentBodyNode();
  if (parentBody == nullptr)
  {
    return Eigen::Vector3s::Zero();
  }

  Eigen::Matrix3s R = parentBody->getWorldTransform().linear();
  Eigen::Isometry3s T_jj = getTransformFromParentBodyNode().inverse()
                           * getRelativeTransform()
                           * getTransformFromChildBodyNode();
  if (axis == -1)
  {
    Eigen::Vector3s dT_jj
        = getTransformFromParentBodyNode().linear()
          * T_jj.translation().cwiseQuotient(getParentScale());
    Eigen::Vector3s parentOffset
        = getTransformFromParentBodyNode().translation().cwiseQuotient(
            getParentScale());
    return R * (parentOffset + dT_jj);
  }
  else
  {
    Eigen::Vector3s dT_jj
        = getTransformFromParentBodyNode().linear()
          * (Eigen::Vector3s::Unit(axis) * T_jj.translation()(axis)
             / getParentScale()(axis));
    Eigen::Vector3s parentOffset
        = Eigen::Vector3s::Unit(axis)
          * (getTransformFromParentBodyNode().translation()(axis)
             / getParentScale()(axis));
    return R * (parentOffset + dT_jj);
  }
}

//==============================================================================
/// Gets the derivative of the spatial Jacobian of the child BodyNode relative
/// to the parent BodyNode expressed in the child BodyNode frame, with respect
/// to the scaling of the parent body along a specific axis.
///
/// Use axis = -1 for uniform scaling of all the axis.
math::Jacobian EllipsoidJoint::getRelativeJacobianDerivWrtParentScale(
    int axis) const
{
  math::Jacobian J = math::Jacobian::Zero(6, 3);
  Eigen::Vector3s pos = getPositionsStatic();

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

  J.block<3, 3>(0, 0).setZero();

  // 2. The euler transform will generate some linear velocities, based on the
  // offset that the ellipse generates. We can first compute the velocities we
  // would get if we were on a perfect sphere, and then scale those velocities.
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>());
  rot.linear() = eulerR.transpose() * rot.linear() * eulerR;

  Eigen::Vector3s dParentScale;
  if (axis == -1)
  {
    dParentScale = Eigen::Vector3s::Ones();
  }
  else
  {
    dParentScale = Eigen::Vector3s::Unit(axis);
  }

  Eigen::Matrix3s scaleInParentSpace
      = rot.linear().transpose() * mEllipsoidRadii.asDiagonal()
        * dParentScale.asDiagonal() * rot.linear();
  J.block<3, 3>(3, 0) = scaleInParentSpace * J.block<3, 3>(3, 0);

  // Finally, take into account the transform to the child body node
  J = math::AdTJac(getTransformFromChildBodyNode(), J);
  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
EllipsoidJoint::getRelativeJacobianDerivWrtPositionDerivWrtParentScale(
    std::size_t firstIndex, int axis) const
{
  Eigen::VectorXs pos = this->getPositions();
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dFirst = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dSecond = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> ddJ_dFirst_dSecond
      = Eigen::Matrix<s_t, 6, 3>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dFirst
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          firstIndex,
          pos.head<3>(),
          mAxisOrder,
          mFlipAxisMap.head<3>(),
          identity);

  Eigen::Vector3s parentScale = getParentScale();
  Eigen::Vector3s dParentScale;
  if (axis == -1)
  {
    dParentScale = Eigen::Vector3s::Ones();
  }
  else
  {
    dParentScale = Eigen::Vector3s::Unit(axis);
  }

  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  dJ_dFirst.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ_dFirst.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  dJ_dFirst.block<3, 1>(3, 0)
      = dJ_dFirst.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 1)
      = dJ_dFirst.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 2)
      = dJ_dFirst.block<3, 1>(0, 2).cross(localSphericalOffset);
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

  // euler wrt euler
  Eigen::Matrix3s dRot_dFirst
      = rot
        * math::makeSkewSymmetric(
            eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
  Eigen::Matrix3s dScaleInParentSpace_dFirst = dRot_dFirst.transpose()
                                               * mEllipsoidRadii.asDiagonal()
                                               * parentScale.asDiagonal() * rot;
  dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();

  Eigen::Matrix3s dScaleInParentSpace_dSecond
      = rot.transpose() * mEllipsoidRadii.asDiagonal()
        * dParentScale.asDiagonal() * rot;

  Eigen::Matrix3s ddScaleInParentSpace_dFirst_dSecond
      = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal()
        * dParentScale.asDiagonal() * rot;
  ddScaleInParentSpace_dFirst_dSecond
      += ddScaleInParentSpace_dFirst_dSecond.transpose().eval();

  ddJ_dFirst_dSecond.block<3, 3>(3, 0)
      = dScaleInParentSpace_dSecond * dJ_dFirst.block<3, 3>(3, 0)
        + scaleInParentSpace * ddJ_dFirst_dSecond.block<3, 3>(3, 0)
        + ddScaleInParentSpace_dFirst_dSecond * J.block<3, 3>(3, 0)
        + dScaleInParentSpace_dFirst * dJ_dSecond.block<3, 3>(3, 0);
  ddJ_dFirst_dSecond.block<3, 3>(0, 0).setZero();

  // Finally, take into account the transform to the child body node
  ddJ_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), ddJ_dFirst_dSecond);
  return ddJ_dFirst_dSecond;
}

//==============================================================================
/// Gets the derivative of the spatial Jacobian of the child BodyNode relative
/// to the parent BodyNode expressed in the child BodyNode frame, with respect
/// to the scaling of the child body along a specific axis.
///
/// Use axis = -1 for uniform scaling of all the axis.
math::Jacobian EllipsoidJoint::getRelativeJacobianTimeDerivDerivWrtParentScale(
    int axis) const
{
  (void)axis;
  math::Jacobian J = math::Jacobian::Zero(6, 3);
  Eigen::Vector3s v = getVelocitiesStatic();
  for (int i = 0; i < 3; i++)
  {
    J += v(i) * getRelativeJacobianDerivWrtPositionDerivWrtParentScale(i, axis);
  }
  return J;
}

//==============================================================================
void EllipsoidJoint::updateRelativeJacobian(bool) const
{
  this->mJacobian = getRelativeJacobianStatic(this->getPositionsStatic());
}

//==============================================================================
void EllipsoidJoint::updateRelativeJacobianTimeDeriv() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 3> dJ = Eigen::Matrix<s_t, 6, 3>::Zero();
  for (int i = 0; i < pos.size(); i++)
  {
    dJ += getRelativeJacobianDerivWrtPositionStatic(i) * vel(i);
  }
  this->mJacobianDeriv = dJ;
}

//==============================================================================
math::Jacobian EllipsoidJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 3> ddJ = Eigen::Matrix<s_t, 6, 3>::Zero();
  for (int i = 0; i < pos.size(); i++)
  {
    ddJ += getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(i, index)
           * vel(i);
  }
  return ddJ;
}

//==============================================================================
math::Jacobian EllipsoidJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return getRelativeJacobianDerivWrtPositionStatic(index);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s EllipsoidJoint::getScrewAxisGradientForPosition(
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
Eigen::Vector6s EllipsoidJoint::getScrewAxisGradientForForce(
    int axisDof, int rotateDof)
{
  // Defaults to Finite Differencing - this is slow, but at least it's
  // approximately correct. Child joints should override with a faster
  // implementation.
  return Joint::finiteDifferenceScrewAxisGradientForForce(axisDof, rotateDof);
}

// For testing
Eigen::MatrixXs EllipsoidJoint::getScratch(int firstIndex)
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ = Eigen::Matrix<s_t, 6, 3>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  Eigen::Matrix<s_t, 6, 3> euler_dJ
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          firstIndex,
          pos.head<3>(),
          mAxisOrder,
          mFlipAxisMap.head<3>(),
          identity);

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
  Eigen::Vector3s parentScale = getParentScale();

  Eigen::Matrix3s scaleInParentSpace = rot.transpose()
                                       * mEllipsoidRadii.asDiagonal()
                                       * parentScale.asDiagonal() * rot;

  Eigen::Matrix3s dRot
      = rot
        * math::makeSkewSymmetric(
            eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));

  Eigen::Matrix3s dScaleInParentSpace = dRot.transpose()
                                        * mEllipsoidRadii.asDiagonal()
                                        * parentScale.asDiagonal() * rot;
  dScaleInParentSpace += dScaleInParentSpace.transpose().eval();

  dJ.block<3, 3>(3, 0) = scaleInParentSpace * dJ.block<3, 3>(3, 0)
                         + dScaleInParentSpace * J.block<3, 3>(3, 0);

  // Finally, take into account the transform to the child body node
  dJ = math::AdTJacFixed(getTransformFromChildBodyNode(), dJ);

  return dJ;
}

Eigen::MatrixXs EllipsoidJoint::analyticalScratch(int firstIndex, int axis)
{
  Eigen::VectorXs pos = this->getPositions();
  (void)pos;
  (void)firstIndex;
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dFirst = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> dJ_dSecond = Eigen::Matrix<s_t, 6, 3>::Zero();
  Eigen::Matrix<s_t, 6, 3> ddJ_dFirst_dSecond
      = Eigen::Matrix<s_t, 6, 3>::Zero();

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  Eigen::Matrix<s_t, 6, 3> eulerJ = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(), mAxisOrder, mFlipAxisMap.head<3>(), identity);
  (void)eulerJ;
  Eigen::Matrix<s_t, 6, 3> euler_dJ_dFirst
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          firstIndex,
          pos.head<3>(),
          mAxisOrder,
          mFlipAxisMap.head<3>(),
          identity);

  Eigen::Vector3s parentScale = getParentScale();
  Eigen::Vector3s dParentScale;
  if (axis == -1)
  {
    dParentScale = Eigen::Vector3s::Ones();
  }
  else
  {
    dParentScale = Eigen::Vector3s::Unit(axis);
  }

  Eigen::Matrix3s eulerR = Eigen::Matrix3s::Zero();
  eulerR(1, 0) = -1.0;
  eulerR(0, 1) = 1.0;
  eulerR(2, 2) = 1.0;
  dJ_dFirst.block<3, 3>(0, 0) = eulerR.transpose() * euler_dJ_dFirst.topRows(3);
  J.block<3, 3>(0, 0) = eulerR.transpose() * eulerJ.topRows(3);

  // Get the spherical velocity from the euler joints in local space (this has
  // not yet been scaled by the ellipsoid radii)
  const Eigen::Vector3s localSphericalOffset = Eigen::Vector3s::UnitZ();
  dJ_dFirst.block<3, 1>(3, 0)
      = dJ_dFirst.block<3, 1>(0, 0).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 1)
      = dJ_dFirst.block<3, 1>(0, 1).cross(localSphericalOffset);
  dJ_dFirst.block<3, 1>(3, 2)
      = dJ_dFirst.block<3, 1>(0, 2).cross(localSphericalOffset);
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

  // euler wrt euler
  Eigen::Matrix3s dRot_dFirst
      = rot
        * math::makeSkewSymmetric(
            eulerR.transpose() * eulerJ.block<3, 1>(0, firstIndex));
  Eigen::Matrix3s dScaleInParentSpace_dFirst = dRot_dFirst.transpose()
                                               * mEllipsoidRadii.asDiagonal()
                                               * parentScale.asDiagonal() * rot;
  dScaleInParentSpace_dFirst += dScaleInParentSpace_dFirst.transpose().eval();

  Eigen::Matrix3s dScaleInParentSpace_dSecond
      = rot.transpose() * mEllipsoidRadii.asDiagonal()
        * dParentScale.asDiagonal() * rot;

  Eigen::Matrix3s ddScaleInParentSpace_dFirst_dSecond
      = dRot_dFirst.transpose() * mEllipsoidRadii.asDiagonal()
        * dParentScale.asDiagonal() * rot;
  ddScaleInParentSpace_dFirst_dSecond
      += ddScaleInParentSpace_dFirst_dSecond.transpose().eval();

  ddJ_dFirst_dSecond.block<3, 3>(3, 0)
      = dScaleInParentSpace_dSecond * dJ_dFirst.block<3, 3>(3, 0)
        + scaleInParentSpace * ddJ_dFirst_dSecond.block<3, 3>(3, 0)
        + ddScaleInParentSpace_dFirst_dSecond * J.block<3, 3>(3, 0)
        + dScaleInParentSpace_dFirst * dJ_dSecond.block<3, 3>(3, 0);
  ddJ_dFirst_dSecond.block<3, 3>(0, 0).setZero();

  dJ_dFirst.block<3, 3>(3, 0)
      = scaleInParentSpace * dJ_dFirst.block<3, 3>(3, 0)
        + dScaleInParentSpace_dFirst * J.block<3, 3>(3, 0);
  dJ_dSecond.block<3, 3>(3, 0)
      = scaleInParentSpace * dJ_dSecond.block<3, 3>(3, 0)
        + dScaleInParentSpace_dSecond * J.block<3, 3>(3, 0);

  // Finally, take into account the transform to the child body node
  ddJ_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), ddJ_dFirst_dSecond);
  return ddJ_dFirst_dSecond;
}

Eigen::MatrixXs EllipsoidJoint::finiteDifferenceScratch(
    int firstIndex, int axis)
{
  Eigen::MatrixXs result = getScratch(firstIndex);

  Eigen::Vector3s originalParentScale = getParentScale();

  s_t eps = 1e-3;

  math::finiteDifference<Eigen::MatrixXs>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::MatrixXs& out) {
        Eigen::VectorXs tweaked = originalParentScale;
        if (axis == -1)
        {
          tweaked += Eigen::Vector3s::Ones() * eps;
        }
        else
        {
          tweaked(axis) += eps;
        }
        setParentScale(tweaked);
        out = getScratch(firstIndex);
        return true;
      },
      result,
      eps,
      true);

  setParentScale(originalParentScale);

  return result;
}

} // namespace dynamics
} // namespace dart