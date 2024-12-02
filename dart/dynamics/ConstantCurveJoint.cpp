#include "dart/dynamics/ConstantCurveJoint.hpp"

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

ConstantCurveJoint::ConstantCurveJoint(
    const detail::GenericJointProperties<math::RealVectorSpace<4>>& props)
  : GenericJoint<math::RealVectorSpace<4>>(props),
    mFlipAxisMap(Eigen::Vector3s::Ones()),
    mNeutralPos(Eigen::Vector4s::Unit(3) * 0.6)
{
}

//==============================================================================
const std::string& ConstantCurveJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& ConstantCurveJoint::getStaticType()
{
  static const std::string name = "ConstantCurveJoint";
  return name;
}

//==============================================================================
bool ConstantCurveJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void ConstantCurveJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s ConstantCurveJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
void ConstantCurveJoint::setNeutralPos(Eigen::Vector4s pos)
{
  mNeutralPos = pos;
}

//==============================================================================
Eigen::Vector4s ConstantCurveJoint::getNeutralPos() const
{
  return mNeutralPos;
}

//==============================================================================
dart::dynamics::Joint* ConstantCurveJoint::clone() const
{
  ConstantCurveJoint* joint
      = new ConstantCurveJoint(this->getJointProperties());
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(getFlipAxisMap());
  joint->setName(this->getName());
  joint->setNeutralPos(this->getNeutralPos());
  joint->setPositionsStatic(this->getPositionsStatic());
  joint->setPositionUpperLimits(this->getPositionUpperLimits());
  joint->setPositionLowerLimits(this->getPositionLowerLimits());
  joint->setVelocityUpperLimits(this->getVelocityUpperLimits());
  joint->setVelocityLowerLimits(this->getVelocityLowerLimits());
  return joint;
}

//==============================================================================
dart::dynamics::Joint* ConstantCurveJoint::simplifiedClone() const
{
  // TOOD: we need to actually find a good simplification for this joint in
  // terms of simpler joint types, maybe a ball joint with an offset?
  assert(false);
  return clone();
}

//==============================================================================
void ConstantCurveJoint::updateDegreeOfFreedomNames()
{
  if (!this->mDofs[0]->isNamePreserved())
    this->mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
void ConstantCurveJoint::updateRelativeTransform() const
{
  Eigen::Vector4s pos = this->getPositionsStatic() + mNeutralPos;
  s_t d = pos(3) * this->getChildScale()(1);

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());

  // 2. Computing translation from vertical
  Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle
      = Eigen::Vector3s(-sz, cx * cz, cz * sx); // rot.linear().col(1);

  s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));
  if (sinTheta < 0.001)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;
  }
  else
  {
    // Compute the bend as a function of the angle from vertical
    s_t theta = asin(sinTheta);
    s_t r = (d / theta);
    s_t horizontalDist = r - r * cos(theta);
    s_t verticalDist = r * sinTheta;

    bentRod.translation() = Eigen::Vector3s(
        horizontalDist * (linearAngle(0) / sinTheta),
        verticalDist,
        horizontalDist * (linearAngle(2) / sinTheta));
    bentRod.linear() = rot.linear();
  }

  // 3. Situate relative to parent and child joints
  this->mT = Joint::mAspectProperties.mT_ParentBodyToJoint * bentRod
             * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4> ConstantCurveJoint::getRelativeJacobianStatic(
    const Eigen::Vector4s& rawPos) const
{
  // Think in terms of the child frame

  Eigen::Vector4s pos = rawPos + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(),
      EulerJoint::AxisOrder::XZY,
      mFlipAxisMap.head<3>(),
      identity);

  s_t d = pos(3) * this->getChildScale()(1);

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);
  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.001)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint

    // 1. Do the euler rotation
    Eigen::Isometry3s rot = EulerJoint::convertToTransform(
        pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    J.block<3, 1>(3, 0)
        = 0.5 * J.block<3, 1>(0, 0).cross(bentRod.translation());
    J.block<3, 1>(3, 1)
        = 0.5 * J.block<3, 1>(0, 1).cross(bentRod.translation());
    J.block<3, 1>(3, 2)
        = 0.5 * J.block<3, 1>(0, 2).cross(bentRod.translation());
    if (bentRod.translation().norm() > 0.003)
    {
      J.block<3, 1>(3, 3)
          = bentRod.translation().normalized() * this->getChildScale()(1);
    }
    else
    {
      J.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    // Compute the bend as a function of the angle from vertical
    Eigen::Vector4s dSinTheta;
    for (int i = 0; i < 3; i++)
    {
      dSinTheta(i) = (0.5
                      / sqrt(
                          linearAngle(0) * linearAngle(0)
                          + linearAngle(2) * linearAngle(2)))
                     * (2 * linearAngle(0) * dLinearAngle(0, i)
                        + 2 * linearAngle(2) * dLinearAngle(2, i));
    }
    dSinTheta(3) = 0;

    s_t theta = asin(sinTheta);
    Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;

    s_t r = (d / theta);
    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    s_t horizontalDist = r - r * cos(theta);

    Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    Eigen::Vector4s dVerticalDist = r * cos(theta) * dTheta + dR * sinTheta;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    J.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation.block<3, 1>(0, 0);
    J.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation.block<3, 1>(0, 1);
    J.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation.block<3, 1>(0, 2);
    J.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J = math::AdTJacFixed(getTransformFromChildBodyNode(), J);
  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4>
ConstantCurveJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  // Think in terms of the child frame

  Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  Eigen::Matrix3s rot_dFirst;
  if (index < 3)
  {
    rot_dFirst = math::eulerXZYToMatrixGrad(pos.head<3>(), index);
  }
  else
  {
    rot_dFirst.setZero();
  }

  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  if (index < 3)
  {
    J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
        index,
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);
  }

  s_t scale = this->getChildScale()(1);
  s_t d = pos(3) * scale;
  s_t d_dFirst = index == 3 ? scale : 0.0;

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  if (index == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    dLinearAngle_dFirst.col(2).setZero();
    dLinearAngle_dFirst.col(3).setZero();
  }
  else if (index == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dFirst.col(2).setZero();
    dLinearAngle_dFirst.col(3).setZero();
  }

  s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.001)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    Eigen::Vector3s translation_dFirst;
    if (index < 3)
    {
      translation_dFirst
          = rot.linear() * J.block<3, 1>(0, index).cross(bentRod.translation());
    }
    else
    {
      translation_dFirst
          = rot.linear() * bentRod.translation().normalized() * scale;
    }

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    if (index < 3)
    {
      if (translation_dFirst.norm() > 0.003)
      {
        J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized() * scale;
      }
      else
      {
        J_dFirst.block<3, 1>(3, 3).setZero();
      }
    }
    else
    {
      J_dFirst.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    Eigen::Vector4s dSinTheta;
    Eigen::Vector4s dSinTheta_dFirst;
    for (int i = 0; i < 3; i++)
    {
      s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;

    s_t sinTheta_dFirst = (0.5
                           / sqrt(
                               linearAngle(0) * linearAngle(0)
                               + linearAngle(2) * linearAngle(2)))
                          * (2 * linearAngle(0) * linearAngle_dFirst(0)
                             + 2 * linearAngle(2) * linearAngle_dFirst(2));

    // Compute the bend as a function of the angle from vertical
    s_t theta = asin(sinTheta);
    s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;

    Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;

    s_t r = (d / theta);
    s_t r_dFirst = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);
    (void)r_dFirst;

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);
    (void)dR_dFirst;

    s_t horizontalDist = r - r * cos(theta);
    s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;

    Eigen::Vector4s dVerticalDist = r * cos(theta) * dTheta + dR * sinTheta;
    Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst = math::AdTJacFixed(getTransformFromChildBodyNode(), J_dFirst);
  return J_dFirst;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4>
ConstantCurveJoint::getRelativeJacobianDerivWrtSegmentLengthStatic(
    s_t d,
    s_t d_dFirst,
    s_t scaleLen,
    Eigen::Vector3s inputPos,
    Eigen::Vector3s neutralPos,
    Eigen::Vector3s flipAxisMap,
    Eigen::Isometry3s childTransform)
{
  // Think in terms of the child frame

  Eigen::Vector3s pos = inputPos + neutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos, EulerJoint::AxisOrder::XZY, flipAxisMap);
  const Eigen::Matrix3s rot_dFirst = Eigen::Matrix3s::Zero();

  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  const Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  const Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  const Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  const Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.001)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos, EulerJoint::AxisOrder::XZY, flipAxisMap, identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    const Eigen::Vector3s translation_dFirst
        = rot.linear() * bentRod.translation().normalized() * (d / scaleLen);

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized();
  }
  else
  {
    Eigen::Vector4s dSinTheta;
    Eigen::Vector4s dSinTheta_dFirst;
    for (int i = 0; i < 3; i++)
    {
      const s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      const s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      const s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      const s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;

    s_t sinTheta_dFirst = (0.5
                           / sqrt(
                               linearAngle(0) * linearAngle(0)
                               + linearAngle(2) * linearAngle(2)))
                          * (2 * linearAngle(0) * linearAngle_dFirst(0)
                             + 2 * linearAngle(2) * linearAngle_dFirst(2));

    // Compute the bend as a function of the angle from vertical
    const s_t theta = asin(sinTheta);
    const s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;

    const Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;

    const s_t r = (d / theta);
    const s_t r_dFirst
        = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);

    const s_t horizontalDist = r - r * cos(theta);
    const s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);

    const Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);

    const Eigen::Vector4s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst = math::AdTJacFixed(childTransform, J_dFirst);
  return J_dFirst;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4>
ConstantCurveJoint::getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(
    std::size_t firstIndex, std::size_t secondIndex) const
{
  (void)secondIndex;

  // Think in terms of the child frame

  Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  Eigen::Matrix3s rot_dFirst;
  if (firstIndex < 3)
  {
    rot_dFirst = math::eulerXZYToMatrixGrad(pos.head<3>(), firstIndex);
  }
  else
  {
    rot_dFirst.setZero();
  }
  Eigen::Matrix3s rot_dSecond;
  if (secondIndex < 3)
  {
    rot_dSecond = math::eulerXZYToMatrixGrad(pos.head<3>(), secondIndex);
  }
  else
  {
    rot_dSecond.setZero();
  }
  Eigen::Matrix3s rot_dFirst_dSecond;
  if (firstIndex < 3 && secondIndex < 3)
  {
    rot_dFirst_dSecond = math::eulerXZYToMatrixSecondGrad(
        pos.head<3>(), firstIndex, secondIndex);
  }
  else
  {
    rot_dFirst_dSecond.setZero();
  }

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (firstIndex < 3)
  {
    J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);
  }
  Eigen::Matrix<s_t, 6, 4> J_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (secondIndex < 3)
  {
    J_dSecond.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);
  }
  Eigen::Matrix<s_t, 6, 4> J_dFirst_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (firstIndex < 3 && secondIndex < 3)
  {
    J_dFirst_dSecond.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            Eigen::Vector3s::Unit(firstIndex),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);
  }

  s_t scale = this->getChildScale()(1);
  s_t d = pos(3) * scale;
  s_t d_dFirst = firstIndex == 3 ? scale : 0.0;
  s_t d_dSecond = secondIndex == 3 ? scale : 0.0;
  (void)d_dSecond;
  s_t d_dFirst_dSecond = 0.0;
  (void)d_dFirst_dSecond;

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dSecond = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dFirst_dSecond = Eigen::Vector3s::Zero();
  if (firstIndex == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    if (secondIndex == 0)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -cx * cz, cz * -sx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, sx * cz, cz * -cx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, cx * sz, -sz * -sx);
    }
    else if (secondIndex == 1)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -cx * -sz, -sz * -sx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, sx * cz, -cz * cx);
    }
  }
  else if (firstIndex == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    if (secondIndex == 0)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -cx * -sz, -sz * -sx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, sx * cz, -cz * cx);
    }
    else if (secondIndex == 1)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(sz, cx * -cz, -cz * sx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -sx * -cz, -cz * cx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(cz, -cx * -sz, sz * sx);
    }
  }

  if (secondIndex == 0)
  {
    linearAngle_dSecond = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dSecond.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dSecond.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    dLinearAngle_dSecond.col(2).setZero();
    dLinearAngle_dSecond.col(3).setZero();
  }
  else if (secondIndex == 1)
  {
    linearAngle_dSecond = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dSecond.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dSecond.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dSecond.col(2).setZero();
    dLinearAngle_dSecond.col(3).setZero();
  }

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.003)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    Eigen::Vector3s translation_dSecond;
    Eigen::Vector3s translation_normalized_dSecond;
    if (secondIndex < 3)
    {
      translation_dSecond
          = rot.linear()
            * J.block<3, 1>(0, secondIndex).cross(bentRod.translation());
      translation_normalized_dSecond
          = rot.linear()
            * J.block<3, 1>(0, secondIndex)
                  .cross(bentRod.translation().normalized());
    }
    else
    {
      translation_dSecond = rot.linear() * bentRod.translation().normalized();
      translation_normalized_dSecond.setZero();
    }

    Eigen::Vector3s translation_dFirst;
    Eigen::Vector3s translation_dFirst_dSecond;
    if (firstIndex < 3)
    {
      translation_dFirst
          = rot.linear()
            * J.block<3, 1>(0, firstIndex).cross(bentRod.translation());
      translation_dFirst_dSecond
          = (rot_dSecond
             * J.block<3, 1>(0, firstIndex).cross(bentRod.translation()))
            + (rot.linear()
               * J_dSecond.block<3, 1>(0, firstIndex)
                     .cross(bentRod.translation()))
            + (rot.linear()
               * J.block<3, 1>(0, firstIndex).cross(translation_dSecond));
    }
    else
    {
      translation_dFirst = rot.linear() * bentRod.translation().normalized();
      translation_dFirst_dSecond
          = rot_dSecond * bentRod.translation().normalized()
            + rot.linear() * translation_normalized_dSecond;
    }

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    if (firstIndex < 3)
    {
      J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized();
    }
    else
    {
      J_dFirst.block<3, 1>(3, 3).setZero();
    }

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 0).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 0).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 0).cross(translation_dFirst)
             + J.block<3, 1>(0, 0).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 1).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 1).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 1).cross(translation_dFirst)
             + J.block<3, 1>(0, 1).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 2).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 2).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 2).cross(translation_dFirst)
             + J.block<3, 1>(0, 2).cross(translation_dFirst_dSecond));
    if (firstIndex < 3)
    {
      if (translation_dFirst.norm() > 0)
      {
        J_dFirst_dSecond.block<3, 1>(3, 3)
            = (translation_dFirst_dSecond
               - (translation_dFirst.dot(translation_dFirst_dSecond)
                  * translation_dFirst)
                     / (translation_dFirst.squaredNorm()))
              / (translation_dFirst.norm());
      }
    }
    else
    {
      J_dFirst_dSecond.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    Eigen::Vector4s dSinTheta = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst_dSecond = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dSecond = Eigen::Vector4s::Zero();
    for (int i = 0; i < 3; i++)
    {
      const s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      const s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      const s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      const s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;

      const s_t part1_dSecond
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2)));
      const s_t part2_dSecond
          = (2 * linearAngle(0) * dLinearAngle_dSecond(0, i)
             + 2 * linearAngle_dSecond(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dSecond(2, i)
             + 2 * linearAngle_dSecond(2) * dLinearAngle(2, i));

      dSinTheta_dSecond(i) = part1_dSecond * part2 + part1 * part2_dSecond;

      const s_t part1_dFirst_dSecond
          = ((0.375
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  2.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)))
            + ((-0.25
                / pow(
                    linearAngle(0) * linearAngle(0)
                        + linearAngle(2) * linearAngle(2),
                    1.5))
               * 2
               * ((linearAngle_dSecond(0) * linearAngle_dFirst(0))
                  + (linearAngle(0) * linearAngle_dFirst_dSecond(0))
                  + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                  + (linearAngle(2) * linearAngle_dFirst_dSecond(2))));
      s_t part2_dFirst_dSecond
          = 2
            * ((linearAngle_dSecond(0) * dLinearAngle_dFirst(0, i)
                + linearAngle(0) * dLinearAngle_dFirst_dSecond(0, i))
               + (linearAngle_dFirst_dSecond(0) * dLinearAngle(0, i))
               + (linearAngle_dFirst(0) * dLinearAngle_dSecond(0, i))
               + (linearAngle_dSecond(2) * dLinearAngle_dFirst(2, i))
               + (linearAngle(2) * dLinearAngle_dFirst_dSecond(2, i))
               + (linearAngle_dFirst_dSecond(2) * dLinearAngle(2, i))
               + (linearAngle_dFirst(2) * dLinearAngle_dSecond(2, i)));

      dSinTheta_dFirst_dSecond(i)
          = part1_dFirst_dSecond * part2 + part1_dFirst * part2_dSecond
            + part1_dSecond * part2_dFirst + part1 * part2_dFirst_dSecond;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;
    dSinTheta_dSecond(3) = 0;
    dSinTheta_dFirst_dSecond(3) = 0;

    const s_t sinTheta_dFirst
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dFirst(0)
             + 2 * linearAngle(2) * linearAngle_dFirst(2));
    const s_t sinTheta_dSecond
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dSecond(0)
             + 2 * linearAngle(2) * linearAngle_dSecond(2));
    (void)sinTheta_dSecond;
    const s_t sinTheta_dFirst_dSecond
        = (-0.25
           / pow(
               linearAngle(0) * linearAngle(0)
                   + linearAngle(2) * linearAngle(2),
               1.5))
              * 2
              * (linearAngle(0) * linearAngle_dSecond(0)
                 + linearAngle(2) * linearAngle_dSecond(2))
              * 2
              * (linearAngle(0) * linearAngle_dFirst(0)
                 + linearAngle(2) * linearAngle_dFirst(2))
          + (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)))
                * 2
                * (linearAngle_dSecond(0) * linearAngle_dFirst(0)
                   + linearAngle(0) * linearAngle_dFirst_dSecond(0)
                   + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                   + linearAngle(2) * linearAngle_dFirst_dSecond(2));
    (void)sinTheta_dFirst_dSecond;

    // Compute the bend as a function of the angle from vertical
    const s_t theta = asin(sinTheta);
    const s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;
    const s_t theta_dSecond
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dSecond;
    (void)theta_dSecond;
    const s_t theta_dFirst_dSecond
        = ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
           * sinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * sinTheta_dFirst_dSecond);
    (void)theta_dFirst_dSecond;

    const Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;
    const Eigen::Vector4s dTheta_dSecond
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dSecond * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dSecond;
    (void)dTheta_dSecond;

    const Eigen::Vector4s dTheta_dFirst_dSecond
        = ((3.0 / pow(1.0 - (sinTheta * sinTheta), 2.5)) * sinTheta
           * sinTheta_dSecond * sinTheta * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
             * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst_dSecond * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst * dSinTheta_dSecond)
          + ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5))
             * sinTheta_dSecond * dSinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * dSinTheta_dFirst_dSecond);
    (void)dTheta_dFirst_dSecond;

    const s_t r = (d / theta);
    const s_t r_dFirst
        = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);
    (void)r_dFirst;
    const s_t r_dSecond
        = (-d / (theta * theta)) * theta_dSecond + (d_dSecond / theta);
    (void)r_dSecond;
    const s_t r_dFirst_dSecond
        = ((-d_dSecond / (theta * theta)) * theta_dFirst)
          + ((2 * d / (theta * theta * theta)) * theta_dSecond * theta_dFirst)
          + ((-d / (theta * theta)) * theta_dFirst_dSecond)
          + (d_dFirst_dSecond / theta)
          + (-d_dFirst / (theta * theta)) * theta_dSecond;
    (void)r_dFirst_dSecond;

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);
    (void)dR_dFirst;

    Eigen::Vector4s dR_dSecond = Eigen::Vector4s::Zero();
    dR_dSecond.segment<3>(0)
        = (-d_dSecond / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dSecond
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dSecond.segment<3>(0);
    dR_dSecond(3) = -theta_dSecond / (theta * theta);
    (void)dR_dSecond;

    Eigen::Vector4s dR_dFirst_dSecond = Eigen::Vector4s::Zero();
    dR_dFirst_dSecond.segment<3>(0)
        = ((-d_dFirst_dSecond / (theta * theta)) * dTheta.segment<3>(0))
          + ((2 * d_dFirst / (theta * theta * theta)) * theta_dSecond
             * dTheta.segment<3>(0))
          + ((-d_dFirst / (theta * theta)) * dTheta_dSecond.segment<3>(0))
          + ((2 * d_dSecond / (theta * theta * theta)) * theta_dFirst
             * dTheta.segment<3>(0))
          + ((-6 * d / (theta * theta * theta * theta)) * theta_dSecond
             * theta_dFirst * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst_dSecond
             * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst
             * dTheta_dSecond.segment<3>(0))
          + ((-d_dSecond / (theta * theta)) * dTheta_dFirst.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dSecond
             * dTheta_dFirst.segment<3>(0))
          + ((-d / (theta * theta)) * dTheta_dFirst_dSecond.segment<3>(0));
    dR_dFirst_dSecond(3)
        = -theta_dFirst_dSecond / (theta * theta)
          + 2 * theta_dFirst / (theta * theta * theta) * theta_dSecond;
    (void)dR_dFirst_dSecond;

    s_t horizontalDist = r - r * cos(theta);
    (void)horizontalDist;
    s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    s_t horizontalDist_dSecond
        = r_dSecond - (r_dSecond * cos(theta) - r * sin(theta) * theta_dSecond);
    (void)horizontalDist_dSecond;

    s_t horizontalDist_dFirst_dSecond
        = r_dFirst_dSecond
          - (r_dFirst_dSecond * cos(theta)
             - r_dFirst * sin(theta) * theta_dSecond
             - (r_dSecond * sin(theta) * theta_dFirst)
             - (r * cos(theta) * theta_dSecond * theta_dFirst)
             - (r * sin(theta) * theta_dFirst_dSecond));
    (void)horizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;
    const Eigen::Vector4s dHorizontalDist_dSecond
        = dR_dSecond
          + (r_dSecond * sin(theta) * dTheta
             + r * cos(theta) * theta_dSecond * dTheta
             + r * sin(theta) * dTheta_dSecond)
          - (dR_dSecond * cos(theta) - dR * sin(theta) * theta_dSecond);
    (void)dHorizontalDist_dSecond;
    const Eigen::Vector4s dHorizontalDist_dFirst_dSecond
        = dR_dFirst_dSecond
          + ((r_dFirst_dSecond * sin(theta) * dTheta)
             + (r_dFirst * cos(theta) * theta_dSecond * dTheta)
             + (r_dFirst * sin(theta) * dTheta_dSecond))
          + ((r_dSecond * cos(theta) * theta_dFirst * dTheta)
             + (r * -sin(theta) * theta_dSecond * theta_dFirst * dTheta)
             + (r * cos(theta) * theta_dFirst_dSecond * dTheta)
             + (r * cos(theta) * theta_dFirst * dTheta_dSecond))
          + ((r_dSecond * sin(theta) * dTheta_dFirst)
             + (r * cos(theta) * theta_dSecond * dTheta_dFirst)
             + (r * sin(theta) * dTheta_dFirst_dSecond))
          - ((dR_dFirst_dSecond * cos(theta))
             + (dR_dFirst * -sin(theta) * theta_dSecond))
          + ((dR_dSecond * sin(theta) * theta_dFirst)
             + (dR * cos(theta) * theta_dSecond * theta_dFirst)
             + (dR * sin(theta) * theta_dFirst_dSecond));
    (void)dHorizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;
    const Eigen::Vector4s dVerticalDist_dSecond
        = (r_dSecond * cos(theta) * dTheta
           - r * sin(theta) * theta_dSecond * dTheta
           + r * cos(theta) * dTheta_dSecond)
          + (dR_dSecond * sinTheta + dR * sinTheta_dSecond);
    (void)dVerticalDist_dSecond;

    const Eigen::Vector4s dVerticalDist_dFirst_dSecond
        = (((r_dFirst_dSecond * cos(theta) * dTheta)
            + (r_dFirst * -sin(theta) * theta_dSecond * dTheta)
            + (r_dFirst * cos(theta) * dTheta_dSecond))
           - ((r_dSecond * sin(theta) * theta_dFirst * dTheta)
              + (r * cos(theta) * theta_dSecond * theta_dFirst * dTheta)
              + (r * sin(theta) * theta_dFirst_dSecond * dTheta)
              + (r * sin(theta) * theta_dFirst * dTheta_dSecond))
           + ((r_dSecond * cos(theta) * dTheta_dFirst)
              + (r * -sin(theta) * theta_dSecond * dTheta_dFirst)
              + (r * cos(theta) * dTheta_dFirst_dSecond)))
          + ((dR_dFirst_dSecond * sinTheta) + (dR_dFirst * sinTheta_dSecond))
          + ((dR_dSecond * sinTheta_dFirst) + (dR * sinTheta_dFirst_dSecond));
    (void)dVerticalDist_dFirst_dSecond;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    Eigen::Matrix<s_t, 3, 4> dTranslation_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dSecond.row(0)
        = ((linearAngle_dSecond(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(0))
          + ((horizontalDist_dSecond * linearAngle(0)
              + horizontalDist * linearAngle_dSecond(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());
    dTranslation_dSecond.row(1) = dVerticalDist_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dSecond.row(2)
        = ((linearAngle_dSecond(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(2))
          + ((horizontalDist_dSecond * linearAngle(2)
              + horizontalDist * linearAngle_dSecond(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());

    // return dHorizontalDist_dFirst_dSecond.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst_dSecond.row(0)
        = (((linearAngle_dFirst_dSecond(0) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(0) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(0) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(0) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(0) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(0) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(0) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(0))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(0))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(0)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(0))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(0))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(0)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(0))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(0))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(0)))
          // + ((horizontalDist_dFirst * linearAngle(0)
          //     + horizontalDist * linearAngle_dFirst(0))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(0)
               + horizontalDist_dFirst * linearAngle_dSecond(0)
               + horizontalDist_dSecond * linearAngle_dFirst(0)
               + horizontalDist * linearAngle_dFirst_dSecond(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(0))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));
    dTranslation_dFirst_dSecond.row(1) = dVerticalDist_dFirst_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst_dSecond.row(2)
        = (((linearAngle_dFirst_dSecond(2) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(2) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(2) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(2) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(2) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(2) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(2) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(2))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(2))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(2)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(2))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(2))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(2)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(2))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(2))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(2)))
          // + ((horizontalDist_dFirst * linearAngle(2)
          //     + horizontalDist * linearAngle_dFirst(2))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(2)
               + horizontalDist_dFirst * linearAngle_dSecond(2)
               + horizontalDist_dSecond * linearAngle_dFirst(2)
               + horizontalDist * linearAngle_dFirst_dSecond(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(2))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 0)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 0);
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 1)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 1);
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 2)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 2);
    J_dFirst_dSecond.block<3, 1>(3, 3)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 3)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), J_dFirst_dSecond);

  /////////// HERE - analytical
  return J_dFirst_dSecond;
}

//==============================================================================
/// This gets the change in world translation of the child body, with respect
/// to an axis of child scaling. Use axis = -1 for uniform scaling of all the
/// axis.
Eigen::Vector3s ConstantCurveJoint::getWorldTranslationOfChildBodyWrtChildScale(
    int axis) const
{
  if (axis == -1 || axis == 1)
  {
    Eigen::Matrix3s R_wc = getChildBodyNode()->getWorldTransform().linear();
    Eigen::Isometry3s T_jj = getTransformFromParentBodyNode().inverse()
                             * getRelativeTransform()
                             * getTransformFromChildBodyNode();
    Eigen::Vector3s p_cj = getTransformFromChildBodyNode().linear()
                           * T_jj.linear().transpose() * T_jj.translation();
    return ((R_wc * p_cj) / getChildScale()(1))
           + Joint::getWorldTranslationOfChildBodyWrtChildScale(axis);
  }
  else
  {
    return Joint::getWorldTranslationOfChildBodyWrtChildScale(axis);
  }
}

//==============================================================================
Eigen::Matrix<s_t, 6, 4> ConstantCurveJoint::
    getRelativeJacobianDerivWrtPositionDerivWrtSegmentLengthStatic(
        std::size_t firstIndex,
        s_t d,
        s_t d_dSecond,
        s_t scaleLen,
        Eigen::Vector3s rawPos,
        Eigen::Vector3s neutralPos,
        Eigen::Vector3s flipAxisMap,
        Eigen::Isometry3s childTransform)
{
  // Think in terms of the child frame

  Eigen::Vector3s pos = rawPos + neutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, flipAxisMap);
  Eigen::Matrix3s rot_dFirst;
  if (firstIndex < 3)
  {
    rot_dFirst = math::eulerXZYToMatrixGrad(pos.head<3>(), firstIndex);
  }
  else
  {
    rot_dFirst.setZero();
  }
  const Eigen::Matrix3s rot_dSecond = Eigen::Matrix3s::Zero();

  const Eigen::Matrix3s rot_dFirst_dSecond = Eigen::Matrix3s::Zero();

  const Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (firstIndex < 3)
  {
    J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        flipAxisMap,
        identity);
  }
  const Eigen::Matrix<s_t, 6, 4> J_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  Eigen::Matrix<s_t, 6, 4> J_dFirst_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();

  const s_t d_dFirst = firstIndex == 3 ? scaleLen : 0.0;
  (void)d_dSecond;
  const s_t d_dFirst_dSecond = firstIndex == 3 ? 1.0 : 0.0;
  (void)d_dFirst_dSecond;

  // Remember, this is X,*Z*,Y

  const s_t cx = cos(pos(0));
  const s_t sx = sin(pos(0));
  const s_t cz = cos(pos(1));
  const s_t sz = sin(pos(1));

  const Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  const Eigen::Matrix<s_t, 3, 4> dLinearAngle_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  const Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dSecond = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dFirst_dSecond = Eigen::Vector3s::Zero();
  if (firstIndex == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
  }
  else if (firstIndex == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
  }

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.003)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(), EulerJoint::AxisOrder::XZY, flipAxisMap, identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    const Eigen::Vector3s translation_dSecond
        = rot.linear() * bentRod.translation().normalized();
    const Eigen::Vector3s translation_normalized_dSecond
        = Eigen::Vector3s::Zero();

    Eigen::Vector3s translation_dFirst;
    Eigen::Vector3s translation_dFirst_dSecond;
    if (firstIndex < 3)
    {
      translation_dFirst
          = rot.linear()
            * J.block<3, 1>(0, firstIndex).cross(bentRod.translation());
      translation_dFirst_dSecond
          = (rot_dSecond
             * J.block<3, 1>(0, firstIndex).cross(bentRod.translation()))
            + (rot.linear()
               * J_dSecond.block<3, 1>(0, firstIndex)
                     .cross(bentRod.translation()))
            + (rot.linear()
               * J.block<3, 1>(0, firstIndex).cross(translation_dSecond));
    }
    else
    {
      translation_dFirst
          = rot.linear() * bentRod.translation(); // .normalized();
      translation_dFirst_dSecond
          = rot_dSecond * bentRod.translation() // .normalized()
            + rot.linear() * translation_normalized_dSecond;
    }

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    if (firstIndex < 3)
    {
      J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized();
    }
    else
    {
      J_dFirst.block<3, 1>(3, 3).setZero();
    }

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 0).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 0).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 0).cross(translation_dFirst)
             + J.block<3, 1>(0, 0).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 1).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 1).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 1).cross(translation_dFirst)
             + J.block<3, 1>(0, 1).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 2).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 2).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 2).cross(translation_dFirst)
             + J.block<3, 1>(0, 2).cross(translation_dFirst_dSecond));
    if (firstIndex < 3)
    {
      if (translation_dFirst.norm() > 0)
      {
        J_dFirst_dSecond.block<3, 1>(3, 3)
            = (translation_dFirst_dSecond
               - (translation_dFirst.dot(translation_dFirst_dSecond)
                  * translation_dFirst)
                     / (translation_dFirst.squaredNorm()))
              / (translation_dFirst.norm());
      }
    }
    else
    {
      J_dFirst_dSecond.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    Eigen::Vector4s dSinTheta = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst_dSecond = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dSecond = Eigen::Vector4s::Zero();
    for (int i = 0; i < 3; i++)
    {
      const s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      const s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      const s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      const s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;

      const s_t part1_dSecond
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2)));
      const s_t part2_dSecond
          = (2 * linearAngle(0) * dLinearAngle_dSecond(0, i)
             + 2 * linearAngle_dSecond(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dSecond(2, i)
             + 2 * linearAngle_dSecond(2) * dLinearAngle(2, i));

      dSinTheta_dSecond(i) = part1_dSecond * part2 + part1 * part2_dSecond;

      const s_t part1_dFirst_dSecond
          = ((0.375
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  2.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)))
            + ((-0.25
                / pow(
                    linearAngle(0) * linearAngle(0)
                        + linearAngle(2) * linearAngle(2),
                    1.5))
               * 2
               * ((linearAngle_dSecond(0) * linearAngle_dFirst(0))
                  + (linearAngle(0) * linearAngle_dFirst_dSecond(0))
                  + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                  + (linearAngle(2) * linearAngle_dFirst_dSecond(2))));
      s_t part2_dFirst_dSecond
          = 2
            * ((linearAngle_dSecond(0) * dLinearAngle_dFirst(0, i)
                + linearAngle(0) * dLinearAngle_dFirst_dSecond(0, i))
               + (linearAngle_dFirst_dSecond(0) * dLinearAngle(0, i))
               + (linearAngle_dFirst(0) * dLinearAngle_dSecond(0, i))
               + (linearAngle_dSecond(2) * dLinearAngle_dFirst(2, i))
               + (linearAngle(2) * dLinearAngle_dFirst_dSecond(2, i))
               + (linearAngle_dFirst_dSecond(2) * dLinearAngle(2, i))
               + (linearAngle_dFirst(2) * dLinearAngle_dSecond(2, i)));

      dSinTheta_dFirst_dSecond(i)
          = part1_dFirst_dSecond * part2 + part1_dFirst * part2_dSecond
            + part1_dSecond * part2_dFirst + part1 * part2_dFirst_dSecond;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;
    dSinTheta_dSecond(3) = 0;
    dSinTheta_dFirst_dSecond(3) = 0;

    const s_t sinTheta_dFirst
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dFirst(0)
             + 2 * linearAngle(2) * linearAngle_dFirst(2));
    const s_t sinTheta_dSecond
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dSecond(0)
             + 2 * linearAngle(2) * linearAngle_dSecond(2));
    (void)sinTheta_dSecond;
    const s_t sinTheta_dFirst_dSecond
        = (-0.25
           / pow(
               linearAngle(0) * linearAngle(0)
                   + linearAngle(2) * linearAngle(2),
               1.5))
              * 2
              * (linearAngle(0) * linearAngle_dSecond(0)
                 + linearAngle(2) * linearAngle_dSecond(2))
              * 2
              * (linearAngle(0) * linearAngle_dFirst(0)
                 + linearAngle(2) * linearAngle_dFirst(2))
          + (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)))
                * 2
                * (linearAngle_dSecond(0) * linearAngle_dFirst(0)
                   + linearAngle(0) * linearAngle_dFirst_dSecond(0)
                   + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                   + linearAngle(2) * linearAngle_dFirst_dSecond(2));
    (void)sinTheta_dFirst_dSecond;

    // Compute the bend as a function of the angle from vertical
    const s_t theta = asin(sinTheta);
    const s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;
    const s_t theta_dSecond
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dSecond;
    (void)theta_dSecond;
    const s_t theta_dFirst_dSecond
        = ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
           * sinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * sinTheta_dFirst_dSecond);
    (void)theta_dFirst_dSecond;

    const Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;
    const Eigen::Vector4s dTheta_dSecond
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dSecond * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dSecond;
    (void)dTheta_dSecond;

    const Eigen::Vector4s dTheta_dFirst_dSecond
        = ((3.0 / pow(1.0 - (sinTheta * sinTheta), 2.5)) * sinTheta
           * sinTheta_dSecond * sinTheta * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
             * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst_dSecond * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst * dSinTheta_dSecond)
          + ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5))
             * sinTheta_dSecond * dSinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * dSinTheta_dFirst_dSecond);
    (void)dTheta_dFirst_dSecond;

    const s_t r = (d / theta);
    const s_t r_dFirst
        = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);
    (void)r_dFirst;
    const s_t r_dSecond
        = (-d / (theta * theta)) * theta_dSecond + (d_dSecond / theta);
    (void)r_dSecond;
    const s_t r_dFirst_dSecond
        = ((-d_dSecond / (theta * theta)) * theta_dFirst)
          + ((2 * d / (theta * theta * theta)) * theta_dSecond * theta_dFirst)
          + ((-d / (theta * theta)) * theta_dFirst_dSecond)
          + (d_dFirst_dSecond / theta)
          + (-d_dFirst / (theta * theta)) * theta_dSecond;
    (void)r_dFirst_dSecond;

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);
    (void)dR_dFirst;

    Eigen::Vector4s dR_dSecond = Eigen::Vector4s::Zero();
    dR_dSecond.segment<3>(0)
        = (-d_dSecond / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dSecond
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dSecond.segment<3>(0);
    dR_dSecond(3) = -theta_dSecond / (theta * theta);
    (void)dR_dSecond;

    Eigen::Vector4s dR_dFirst_dSecond = Eigen::Vector4s::Zero();
    dR_dFirst_dSecond.segment<3>(0)
        = ((-d_dFirst_dSecond / (theta * theta)) * dTheta.segment<3>(0))
          + ((2 * d_dFirst / (theta * theta * theta)) * theta_dSecond
             * dTheta.segment<3>(0))
          + ((-d_dFirst / (theta * theta)) * dTheta_dSecond.segment<3>(0))
          + ((2 * d_dSecond / (theta * theta * theta)) * theta_dFirst
             * dTheta.segment<3>(0))
          + ((-6 * d / (theta * theta * theta * theta)) * theta_dSecond
             * theta_dFirst * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst_dSecond
             * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst
             * dTheta_dSecond.segment<3>(0))
          + ((-d_dSecond / (theta * theta)) * dTheta_dFirst.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dSecond
             * dTheta_dFirst.segment<3>(0))
          + ((-d / (theta * theta)) * dTheta_dFirst_dSecond.segment<3>(0));
    dR_dFirst_dSecond(3)
        = -theta_dFirst_dSecond / (theta * theta)
          + 2 * theta_dFirst / (theta * theta * theta) * theta_dSecond;
    (void)dR_dFirst_dSecond;

    s_t horizontalDist = r - r * cos(theta);
    (void)horizontalDist;
    s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    s_t horizontalDist_dSecond
        = r_dSecond - (r_dSecond * cos(theta) - r * sin(theta) * theta_dSecond);
    (void)horizontalDist_dSecond;

    s_t horizontalDist_dFirst_dSecond
        = r_dFirst_dSecond
          - (r_dFirst_dSecond * cos(theta)
             - r_dFirst * sin(theta) * theta_dSecond
             - (r_dSecond * sin(theta) * theta_dFirst)
             - (r * cos(theta) * theta_dSecond * theta_dFirst)
             - (r * sin(theta) * theta_dFirst_dSecond));
    (void)horizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;
    const Eigen::Vector4s dHorizontalDist_dSecond
        = dR_dSecond
          + (r_dSecond * sin(theta) * dTheta
             + r * cos(theta) * theta_dSecond * dTheta
             + r * sin(theta) * dTheta_dSecond)
          - (dR_dSecond * cos(theta) - dR * sin(theta) * theta_dSecond);
    (void)dHorizontalDist_dSecond;
    const Eigen::Vector4s dHorizontalDist_dFirst_dSecond
        = dR_dFirst_dSecond
          + ((r_dFirst_dSecond * sin(theta) * dTheta)
             + (r_dFirst * cos(theta) * theta_dSecond * dTheta)
             + (r_dFirst * sin(theta) * dTheta_dSecond))
          + ((r_dSecond * cos(theta) * theta_dFirst * dTheta)
             + (r * -sin(theta) * theta_dSecond * theta_dFirst * dTheta)
             + (r * cos(theta) * theta_dFirst_dSecond * dTheta)
             + (r * cos(theta) * theta_dFirst * dTheta_dSecond))
          + ((r_dSecond * sin(theta) * dTheta_dFirst)
             + (r * cos(theta) * theta_dSecond * dTheta_dFirst)
             + (r * sin(theta) * dTheta_dFirst_dSecond))
          - ((dR_dFirst_dSecond * cos(theta))
             + (dR_dFirst * -sin(theta) * theta_dSecond))
          + ((dR_dSecond * sin(theta) * theta_dFirst)
             + (dR * cos(theta) * theta_dSecond * theta_dFirst)
             + (dR * sin(theta) * theta_dFirst_dSecond));
    (void)dHorizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;
    const Eigen::Vector4s dVerticalDist_dSecond
        = (r_dSecond * cos(theta) * dTheta
           - r * sin(theta) * theta_dSecond * dTheta
           + r * cos(theta) * dTheta_dSecond)
          + (dR_dSecond * sinTheta + dR * sinTheta_dSecond);
    (void)dVerticalDist_dSecond;

    const Eigen::Vector4s dVerticalDist_dFirst_dSecond
        = (((r_dFirst_dSecond * cos(theta) * dTheta)
            + (r_dFirst * -sin(theta) * theta_dSecond * dTheta)
            + (r_dFirst * cos(theta) * dTheta_dSecond))
           - ((r_dSecond * sin(theta) * theta_dFirst * dTheta)
              + (r * cos(theta) * theta_dSecond * theta_dFirst * dTheta)
              + (r * sin(theta) * theta_dFirst_dSecond * dTheta)
              + (r * sin(theta) * theta_dFirst * dTheta_dSecond))
           + ((r_dSecond * cos(theta) * dTheta_dFirst)
              + (r * -sin(theta) * theta_dSecond * dTheta_dFirst)
              + (r * cos(theta) * dTheta_dFirst_dSecond)))
          + ((dR_dFirst_dSecond * sinTheta) + (dR_dFirst * sinTheta_dSecond))
          + ((dR_dSecond * sinTheta_dFirst) + (dR * sinTheta_dFirst_dSecond));
    (void)dVerticalDist_dFirst_dSecond;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    Eigen::Matrix<s_t, 3, 4> dTranslation_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dSecond.row(0)
        = ((linearAngle_dSecond(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(0))
          + ((horizontalDist_dSecond * linearAngle(0)
              + horizontalDist * linearAngle_dSecond(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());
    dTranslation_dSecond.row(1) = dVerticalDist_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dSecond.row(2)
        = ((linearAngle_dSecond(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(2))
          + ((horizontalDist_dSecond * linearAngle(2)
              + horizontalDist * linearAngle_dSecond(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());

    // return dHorizontalDist_dFirst_dSecond.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst_dSecond.row(0)
        = (((linearAngle_dFirst_dSecond(0) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(0) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(0) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(0) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(0) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(0) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(0) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(0))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(0))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(0)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(0))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(0))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(0)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(0))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(0))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(0)))
          // + ((horizontalDist_dFirst * linearAngle(0)
          //     + horizontalDist * linearAngle_dFirst(0))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(0)
               + horizontalDist_dFirst * linearAngle_dSecond(0)
               + horizontalDist_dSecond * linearAngle_dFirst(0)
               + horizontalDist * linearAngle_dFirst_dSecond(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(0))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));
    dTranslation_dFirst_dSecond.row(1) = dVerticalDist_dFirst_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst_dSecond.row(2)
        = (((linearAngle_dFirst_dSecond(2) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(2) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(2) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(2) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(2) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(2) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(2) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(2))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(2))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(2)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(2))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(2))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(2)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(2))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(2))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(2)))
          // + ((horizontalDist_dFirst * linearAngle(2)
          //     + horizontalDist * linearAngle_dFirst(2))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(2)
               + horizontalDist_dFirst * linearAngle_dSecond(2)
               + horizontalDist_dSecond * linearAngle_dFirst(2)
               + horizontalDist * linearAngle_dFirst_dSecond(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(2))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 0)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 0);
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 1)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 1);
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 2)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 2);
    J_dFirst_dSecond.block<3, 1>(3, 3)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 3)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst_dSecond = math::AdTJacFixed(childTransform, J_dFirst_dSecond);

  /////////// HERE - analytical
  return J_dFirst_dSecond;
}

//==============================================================================
/// Gets the derivative of the spatial Jacobian of the child BodyNode relative
/// to the parent BodyNode expressed in the child BodyNode frame, with respect
/// to the scaling of the child body along a specific axis.
///
/// Use axis = -1 for uniform scaling of all the axis.
math::Jacobian ConstantCurveJoint::getRelativeJacobianDerivWrtChildScale(
    int axis) const
{
  math::Jacobian J = getRelativeJacobian();

  /*
  //--------------------------------------------------------------------------
  // w' = R*w
  // v' = p x R*w + R*v
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _V.head<3>();
  res.tail<3>().noalias()
      = _T.linear() * _V.tail<3>() + _T.translation().cross(res.head<3>());
  */

  Eigen::Vector3s dTrans = Joint::getOriginalTransformFromChildBodyNode();
  if (axis != -1)
  {
    dTrans = dTrans.cwiseProduct(Eigen::Vector3s::Unit(axis));
  }

  for (int i = 0; i < J.cols(); i++)
  {
    J.block<3, 1>(3, i) = dTrans.cross(J.block<3, 1>(0, i));
    J.block<3, 1>(0, i).setZero();
  }

  if (axis == 1 || axis == -1)
  {
    Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;
    s_t originalD = pos(3);
    s_t scale = this->getChildScale()(1);
    s_t d = originalD * scale;
    math::Jacobian dJ = getRelativeJacobianDerivWrtSegmentLengthStatic(
        d,
        originalD,
        scale,
        getPositionsStatic().head<3>(),
        mNeutralPos.head<3>(),
        mFlipAxisMap,
        getTransformFromChildBodyNode());

    J += dJ;
  }

  return J;
}

//==============================================================================
/// Gets the derivative of the time derivative of the spatial Jacobian of the
/// child BodyNode relative to the parent BodyNode expressed in the child
/// BodyNode frame, with respect to the scaling of the child body along a
/// specific axis.
///
/// Use axis = -1 for uniform scaling of all the axis.
math::Jacobian
ConstantCurveJoint::getRelativeJacobianTimeDerivDerivWrtChildScale(
    int axis) const
{
  math::Jacobian J = getRelativeJacobianTimeDeriv();

  /*
  //--------------------------------------------------------------------------
  // w' = R*w
  // v' = p x R*w + R*v
  //--------------------------------------------------------------------------
  Eigen::Vector6s res;
  res.head<3>().noalias() = _T.linear() * _V.head<3>();
  res.tail<3>().noalias()
      = _T.linear() * _V.tail<3>() + _T.translation().cross(res.head<3>());
  */

  Eigen::Vector3s dTrans = Joint::getOriginalTransformFromChildBodyNode();
  if (axis != -1)
  {
    dTrans = dTrans.cwiseProduct(Eigen::Vector3s::Unit(axis));
  }

  for (int i = 0; i < J.cols(); i++)
  {
    J.block<3, 1>(3, i) = dTrans.cross(J.block<3, 1>(0, i));
    J.block<3, 1>(0, i).setZero();
  }

  if (axis == 1 || axis == -1)
  {
    Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;
    Eigen::Vector4s vel = getVelocitiesStatic();
    s_t originalD = pos(3);
    s_t scale = this->getChildScale()(1);
    s_t d = originalD * scale;
    math::Jacobian dJ = math::Jacobian::Zero(6, 4);
    for (int i = 0; i < 4; i++)
    {
      dJ += vel(i)
            * getRelativeJacobianDerivWrtPositionDerivWrtSegmentLengthStatic(
                i,
                d,
                originalD,
                scale,
                getPositionsStatic().head<3>(),
                mNeutralPos.head<3>(),
                mFlipAxisMap,
                getTransformFromChildBodyNode());
    }

    J += dJ;
  }

  return J;
}

//==============================================================================
void ConstantCurveJoint::updateRelativeJacobian(bool) const
{
  this->mJacobian = getRelativeJacobianStatic(this->getPositionsStatic());
}

//==============================================================================
void ConstantCurveJoint::updateRelativeJacobianTimeDeriv() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 4> dJ = Eigen::Matrix<s_t, 6, 4>::Zero();
  for (int i = 0; i < 4; i++)
  {
    // getRelativeJacobianDerivWrtPositionStatic
    Eigen::Matrix<s_t, 6, 4> J_di
        = getRelativeJacobianDerivWrtPositionStatic(i);
    dJ += J_di * vel(i);
  }
  this->mJacobianDeriv = dJ;
}

//==============================================================================
math::Jacobian ConstantCurveJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
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
math::Jacobian ConstantCurveJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return getRelativeJacobianDerivWrtPositionStatic(index);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s ConstantCurveJoint::getScrewAxisGradientForPosition(
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
Eigen::Vector6s ConstantCurveJoint::getScrewAxisGradientForForce(
    int axisDof, int rotateDof)
{
  // Defaults to Finite Differencing - this is slow, but at least it's
  // approximately correct. Child joints should override with a faster
  // implementation.
  return Joint::finiteDifferenceScrewAxisGradientForForce(axisDof, rotateDof);
}

//==============================================================================
/// Returns the value for q that produces the nearest rotation to
/// `relativeRotation` passed in.
Eigen::VectorXs ConstantCurveJoint::getNearestPositionToDesiredRotation(
    const Eigen::Matrix3s& relativeRotationGlobal)
{
  Eigen::Matrix3s relativeRotation
      = Joint::mAspectProperties.mT_ParentBodyToJoint.linear().transpose()
        * relativeRotationGlobal
        * Joint::mAspectProperties.mT_ChildBodyToJoint.linear();
  Eigen::VectorXs positions = getPositions();
  positions.head<3>() = EulerJoint::convertToPositions(
      relativeRotation, EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  return positions;
}

// For testing
Eigen::MatrixXs ConstantCurveJoint::getScratch(int firstIndex)
{
  // Think in terms of the child frame

  Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());

  Eigen::Matrix3s rot_dFirst;
  if (firstIndex < 3)
  {
    rot_dFirst = math::eulerXZYToMatrixGrad(pos.head<3>(), firstIndex);
  }
  else
  {
    rot_dFirst.setZero();
  }

  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  if (firstIndex < 3)
  {
    J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);
  }

  s_t d = pos(3);
  s_t d_dFirst = firstIndex == 3 ? 1.0 : 0.0;

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  if (firstIndex == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    dLinearAngle_dFirst.col(2).setZero();
    dLinearAngle_dFirst.col(3).setZero();
  }
  else if (firstIndex == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dFirst.col(2).setZero();
    dLinearAngle_dFirst.col(3).setZero();
  }

  s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.003)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    Eigen::Vector3s translation_dFirst;
    if (firstIndex < 3)
    {
      translation_dFirst
          = rot.linear()
            * J.block<3, 1>(0, firstIndex).cross(bentRod.translation());
    }
    else
    {
      translation_dFirst = rot.linear() * bentRod.translation().normalized();
    }

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    if (firstIndex < 3)
    {
      if (translation_dFirst.norm() > 0.003)
      {
        J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized();
      }
      else
      {
        J_dFirst.block<3, 1>(3, 3).setZero();
      }
    }
    else
    {
      J_dFirst.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    Eigen::Vector4s dSinTheta = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst = Eigen::Vector4s::Zero();
    for (int i = 0; i < 3; i++)
    {
      s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;

    s_t sinTheta_dFirst = (0.5
                           / sqrt(
                               linearAngle(0) * linearAngle(0)
                               + linearAngle(2) * linearAngle(2)))
                          * (2 * linearAngle(0) * linearAngle_dFirst(0)
                             + 2 * linearAngle(2) * linearAngle_dFirst(2));

    // Compute the bend as a function of the angle from vertical
    s_t theta = asin(sinTheta);
    s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;

    Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;

    s_t r = (d / theta);
    s_t r_dFirst = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);
    (void)r_dFirst;

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);
    (void)dR_dFirst;

    s_t horizontalDist = r - r * cos(theta);
    s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst + (r_dFirst * sin(theta) * dTheta)
          + (r * cos(theta) * theta_dFirst * dTheta)
          + (r * sin(theta) * dTheta_dFirst) - (dR_dFirst * cos(theta))
          + (dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;

    Eigen::Vector4s dVerticalDist = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose())
          - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
             * dHorizontalDist.transpose())
          + ((linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0))
          - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
             * dLinearAngle.row(0))
          + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
             * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
          + ((horizontalDist * linearAngle(0))
             * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
             * dSinTheta.transpose())
          + ((horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
             * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst = math::AdTJacFixed(getTransformFromChildBodyNode(), J_dFirst);

  /////////// HERE - scratch
  return J_dFirst;
}

Eigen::MatrixXs ConstantCurveJoint::analyticalScratch(
    int firstIndex, int secondIndex)
{
  (void)secondIndex;

  // Think in terms of the child frame

  Eigen::Vector4s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  Eigen::Matrix3s rot_dFirst;
  if (firstIndex < 3)
  {
    rot_dFirst = math::eulerXZYToMatrixGrad(pos.head<3>(), firstIndex);
  }
  else
  {
    rot_dFirst.setZero();
  }
  Eigen::Matrix3s rot_dSecond;
  if (secondIndex < 3)
  {
    rot_dSecond = math::eulerXZYToMatrixGrad(pos.head<3>(), secondIndex);
  }
  else
  {
    rot_dSecond.setZero();
  }
  Eigen::Matrix3s rot_dFirst_dSecond;
  if (firstIndex < 3 && secondIndex < 3)
  {
    rot_dFirst_dSecond = math::eulerXZYToMatrixSecondGrad(
        pos.head<3>(), firstIndex, secondIndex);
  }
  else
  {
    rot_dFirst_dSecond.setZero();
  }

  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Matrix<s_t, 6, 4> J_dFirst = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (firstIndex < 3)
  {
    J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
        firstIndex,
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);
  }
  Eigen::Matrix<s_t, 6, 4> J_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (secondIndex < 3)
  {
    J_dSecond.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);
  }
  Eigen::Matrix<s_t, 6, 4> J_dFirst_dSecond = Eigen::Matrix<s_t, 6, 4>::Zero();
  if (firstIndex < 3 && secondIndex < 3)
  {
    J_dFirst_dSecond.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
            secondIndex,
            pos.head<3>(),
            Eigen::Vector3s::Unit(firstIndex),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);
  }

  s_t d = pos(3);
  s_t d_dFirst = firstIndex == 3 ? 1.0 : 0.0;
  s_t d_dSecond = secondIndex == 3 ? 1.0 : 0.0;
  (void)d_dSecond;
  s_t d_dFirst_dSecond = 0.0;
  (void)d_dFirst_dSecond;

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 4> dLinearAngle = Eigen::Matrix<s_t, 3, 4>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();
  dLinearAngle.col(3).setZero();

  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Matrix<s_t, 3, 4> dLinearAngle_dFirst_dSecond
      = Eigen::Matrix<s_t, 3, 4>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dSecond = Eigen::Vector3s::Zero();
  Eigen::Vector3s linearAngle_dFirst_dSecond = Eigen::Vector3s::Zero();
  if (firstIndex == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    if (secondIndex == 0)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -cx * cz, cz * -sx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, sx * cz, cz * -cx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, cx * sz, -sz * -sx);
    }
    else if (secondIndex == 1)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -cx * -sz, -sz * -sx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, sx * cz, -cz * cx);
    }
  }
  else if (firstIndex == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    if (secondIndex == 0)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -cx * -sz, -sz * -sx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(0, sx * cz, -cz * cx);
    }
    else if (secondIndex == 1)
    {
      linearAngle_dFirst_dSecond = Eigen::Vector3s(sz, cx * -cz, -cz * sx);
      dLinearAngle_dFirst_dSecond.col(0)
          = Eigen::Vector3s(0, -sx * -cz, -cz * cx);
      dLinearAngle_dFirst_dSecond.col(1)
          = Eigen::Vector3s(cz, -cx * -sz, sz * sx);
    }
  }

  if (secondIndex == 0)
  {
    linearAngle_dSecond = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dSecond.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dSecond.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    dLinearAngle_dSecond.col(2).setZero();
    dLinearAngle_dSecond.col(3).setZero();
  }
  else if (secondIndex == 1)
  {
    linearAngle_dSecond = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dSecond.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dSecond.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dSecond.col(2).setZero();
    dLinearAngle_dSecond.col(3).setZero();
  }

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.003)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    Eigen::Matrix<s_t, 6, 4> J = Eigen::Matrix<s_t, 6, 4>::Zero();
    J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
        pos.head<3>(),
        EulerJoint::AxisOrder::XZY,
        mFlipAxisMap.head<3>(),
        identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    Eigen::Vector3s translation_dSecond;
    Eigen::Vector3s translation_normalized_dSecond;
    if (secondIndex < 3)
    {
      translation_dSecond
          = rot.linear()
            * J.block<3, 1>(0, secondIndex).cross(bentRod.translation());
      translation_normalized_dSecond
          = rot.linear()
            * J.block<3, 1>(0, secondIndex)
                  .cross(bentRod.translation().normalized());
    }
    else
    {
      translation_dSecond = rot.linear() * bentRod.translation().normalized();
      translation_normalized_dSecond.setZero();
    }

    Eigen::Vector3s translation_dFirst;
    Eigen::Vector3s translation_dFirst_dSecond;
    if (firstIndex < 3)
    {
      translation_dFirst
          = rot.linear()
            * J.block<3, 1>(0, firstIndex).cross(bentRod.translation());
      translation_dFirst_dSecond
          = (rot_dSecond
             * J.block<3, 1>(0, firstIndex).cross(bentRod.translation()))
            + (rot.linear()
               * J_dSecond.block<3, 1>(0, firstIndex)
                     .cross(bentRod.translation()))
            + (rot.linear()
               * J.block<3, 1>(0, firstIndex).cross(translation_dSecond));
    }
    else
    {
      translation_dFirst = rot.linear() * bentRod.translation().normalized();
      translation_dFirst_dSecond
          = rot_dSecond * bentRod.translation().normalized()
            + rot.linear() * translation_normalized_dSecond;
    }

    J_dFirst.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 0).cross(bentRod.translation())
             + J.block<3, 1>(0, 0).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 1).cross(bentRod.translation())
             + J.block<3, 1>(0, 1).cross(translation_dFirst));
    J_dFirst.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst.block<3, 1>(0, 2).cross(bentRod.translation())
             + J.block<3, 1>(0, 2).cross(translation_dFirst));
    if (firstIndex < 3)
    {
      J_dFirst.block<3, 1>(3, 3) = translation_dFirst.normalized();
    }
    else
    {
      J_dFirst.block<3, 1>(3, 3).setZero();
    }

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 0).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 0).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 0).cross(translation_dFirst)
             + J.block<3, 1>(0, 0).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 1).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 1).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 1).cross(translation_dFirst)
             + J.block<3, 1>(0, 1).cross(translation_dFirst_dSecond));
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = 0.5
          * (J_dFirst_dSecond.block<3, 1>(0, 2).cross(bentRod.translation())
             + J_dFirst.block<3, 1>(0, 2).cross(translation_dSecond)
             + J_dSecond.block<3, 1>(0, 2).cross(translation_dFirst)
             + J.block<3, 1>(0, 2).cross(translation_dFirst_dSecond));
    if (firstIndex < 3)
    {
      if (translation_dFirst.norm() > 0)
      {
        J_dFirst_dSecond.block<3, 1>(3, 3)
            = (translation_dFirst_dSecond
               - (translation_dFirst.dot(translation_dFirst_dSecond)
                  * translation_dFirst)
                     / (translation_dFirst.squaredNorm()))
              / (translation_dFirst.norm());
      }
    }
    else
    {
      J_dFirst_dSecond.block<3, 1>(3, 3).setZero();
    }
  }
  else
  {
    Eigen::Vector4s dSinTheta = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dFirst_dSecond = Eigen::Vector4s::Zero();
    Eigen::Vector4s dSinTheta_dSecond = Eigen::Vector4s::Zero();
    for (int i = 0; i < 3; i++)
    {
      const s_t part1
          = (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)));
      const s_t part2
          = (2 * linearAngle(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle(2, i));
      dSinTheta(i) = part1 * part2;

      const s_t part1_dFirst
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)));
      const s_t part2_dFirst
          = (2 * linearAngle(0) * dLinearAngle_dFirst(0, i)
             + 2 * linearAngle_dFirst(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dFirst(2, i)
             + 2 * linearAngle_dFirst(2) * dLinearAngle(2, i));

      dSinTheta_dFirst(i) = part1_dFirst * part2 + part1 * part2_dFirst;

      const s_t part1_dSecond
          = ((-0.25
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  1.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2)));
      const s_t part2_dSecond
          = (2 * linearAngle(0) * dLinearAngle_dSecond(0, i)
             + 2 * linearAngle_dSecond(0) * dLinearAngle(0, i)
             + 2 * linearAngle(2) * dLinearAngle_dSecond(2, i)
             + 2 * linearAngle_dSecond(2) * dLinearAngle(2, i));

      dSinTheta_dSecond(i) = part1_dSecond * part2 + part1 * part2_dSecond;

      const s_t part1_dFirst_dSecond
          = ((0.375
              / pow(
                  linearAngle(0) * linearAngle(0)
                      + linearAngle(2) * linearAngle(2),
                  2.5))
             * (2 * linearAngle(0) * linearAngle_dSecond(0)
                + 2 * linearAngle(2) * linearAngle_dSecond(2))
             * (2 * linearAngle(0) * linearAngle_dFirst(0)
                + 2 * linearAngle(2) * linearAngle_dFirst(2)))
            + ((-0.25
                / pow(
                    linearAngle(0) * linearAngle(0)
                        + linearAngle(2) * linearAngle(2),
                    1.5))
               * 2
               * ((linearAngle_dSecond(0) * linearAngle_dFirst(0))
                  + (linearAngle(0) * linearAngle_dFirst_dSecond(0))
                  + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                  + (linearAngle(2) * linearAngle_dFirst_dSecond(2))));
      s_t part2_dFirst_dSecond
          = 2
            * ((linearAngle_dSecond(0) * dLinearAngle_dFirst(0, i)
                + linearAngle(0) * dLinearAngle_dFirst_dSecond(0, i))
               + (linearAngle_dFirst_dSecond(0) * dLinearAngle(0, i))
               + (linearAngle_dFirst(0) * dLinearAngle_dSecond(0, i))
               + (linearAngle_dSecond(2) * dLinearAngle_dFirst(2, i))
               + (linearAngle(2) * dLinearAngle_dFirst_dSecond(2, i))
               + (linearAngle_dFirst_dSecond(2) * dLinearAngle(2, i))
               + (linearAngle_dFirst(2) * dLinearAngle_dSecond(2, i)));

      dSinTheta_dFirst_dSecond(i)
          = part1_dFirst_dSecond * part2 + part1_dFirst * part2_dSecond
            + part1_dSecond * part2_dFirst + part1 * part2_dFirst_dSecond;
    }
    dSinTheta(3) = 0;
    dSinTheta_dFirst(3) = 0;
    dSinTheta_dSecond(3) = 0;
    dSinTheta_dFirst_dSecond(3) = 0;

    const s_t sinTheta_dFirst
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dFirst(0)
             + 2 * linearAngle(2) * linearAngle_dFirst(2));
    const s_t sinTheta_dSecond
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dSecond(0)
             + 2 * linearAngle(2) * linearAngle_dSecond(2));
    (void)sinTheta_dSecond;
    const s_t sinTheta_dFirst_dSecond
        = (-0.25
           / pow(
               linearAngle(0) * linearAngle(0)
                   + linearAngle(2) * linearAngle(2),
               1.5))
              * 2
              * (linearAngle(0) * linearAngle_dSecond(0)
                 + linearAngle(2) * linearAngle_dSecond(2))
              * 2
              * (linearAngle(0) * linearAngle_dFirst(0)
                 + linearAngle(2) * linearAngle_dFirst(2))
          + (0.5
             / sqrt(
                 linearAngle(0) * linearAngle(0)
                 + linearAngle(2) * linearAngle(2)))
                * 2
                * (linearAngle_dSecond(0) * linearAngle_dFirst(0)
                   + linearAngle(0) * linearAngle_dFirst_dSecond(0)
                   + (linearAngle_dSecond(2) * linearAngle_dFirst(2))
                   + linearAngle(2) * linearAngle_dFirst_dSecond(2));
    (void)sinTheta_dFirst_dSecond;

    // Compute the bend as a function of the angle from vertical
    const s_t theta = asin(sinTheta);
    const s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;
    const s_t theta_dSecond
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dSecond;
    (void)theta_dSecond;
    const s_t theta_dFirst_dSecond
        = ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
           * sinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * sinTheta_dFirst_dSecond);
    (void)theta_dFirst_dSecond;

    const Eigen::Vector4s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector4s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;
    const Eigen::Vector4s dTheta_dSecond
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dSecond * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dSecond;
    (void)dTheta_dSecond;

    const Eigen::Vector4s dTheta_dFirst_dSecond
        = ((3.0 / pow(1.0 - (sinTheta * sinTheta), 2.5)) * sinTheta
           * sinTheta_dSecond * sinTheta * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta_dSecond
             * sinTheta_dFirst * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst_dSecond * dSinTheta)
          + ((1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
             * sinTheta_dFirst * dSinTheta_dSecond)
          + ((sinTheta / pow(1.0 - (sinTheta * sinTheta), 1.5))
             * sinTheta_dSecond * dSinTheta_dFirst)
          + ((1.0 / sqrt(1.0 - (sinTheta * sinTheta)))
             * dSinTheta_dFirst_dSecond);
    (void)dTheta_dFirst_dSecond;

    const s_t r = (d / theta);
    const s_t r_dFirst
        = (-d / (theta * theta)) * theta_dFirst + (d_dFirst / theta);
    (void)r_dFirst;
    const s_t r_dSecond
        = (-d / (theta * theta)) * theta_dSecond + (d_dSecond / theta);
    (void)r_dSecond;
    const s_t r_dFirst_dSecond
        = ((-d_dSecond / (theta * theta)) * theta_dFirst)
          + ((2 * d / (theta * theta * theta)) * theta_dSecond * theta_dFirst)
          + ((-d / (theta * theta)) * theta_dFirst_dSecond)
          + (d_dFirst_dSecond / theta)
          + (-d_dFirst / (theta * theta)) * theta_dSecond;
    (void)r_dFirst_dSecond;

    Eigen::Vector4s dR = Eigen::Vector4s::Zero();
    dR.segment<3>(0) = (-d / (theta * theta)) * dTheta.segment<3>(0);
    dR(3) = 1.0 / theta;

    Eigen::Vector4s dR_dFirst = Eigen::Vector4s::Zero();
    dR_dFirst.segment<3>(0)
        = (-d_dFirst / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dFirst
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);
    dR_dFirst(3) = -theta_dFirst / (theta * theta);
    (void)dR_dFirst;

    Eigen::Vector4s dR_dSecond = Eigen::Vector4s::Zero();
    dR_dSecond.segment<3>(0)
        = (-d_dSecond / (theta * theta)) * dTheta.segment<3>(0)
          + (2 * d / (theta * theta * theta)) * theta_dSecond
                * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dSecond.segment<3>(0);
    dR_dSecond(3) = -theta_dSecond / (theta * theta);
    (void)dR_dSecond;

    Eigen::Vector4s dR_dFirst_dSecond = Eigen::Vector4s::Zero();
    dR_dFirst_dSecond.segment<3>(0)
        = ((-d_dFirst_dSecond / (theta * theta)) * dTheta.segment<3>(0))
          + ((2 * d_dFirst / (theta * theta * theta)) * theta_dSecond
             * dTheta.segment<3>(0))
          + ((-d_dFirst / (theta * theta)) * dTheta_dSecond.segment<3>(0))
          + ((2 * d_dSecond / (theta * theta * theta)) * theta_dFirst
             * dTheta.segment<3>(0))
          + ((-6 * d / (theta * theta * theta * theta)) * theta_dSecond
             * theta_dFirst * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst_dSecond
             * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst
             * dTheta_dSecond.segment<3>(0))
          + ((-d_dSecond / (theta * theta)) * dTheta_dFirst.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dSecond
             * dTheta_dFirst.segment<3>(0))
          + ((-d / (theta * theta)) * dTheta_dFirst_dSecond.segment<3>(0));
    dR_dFirst_dSecond(3)
        = -theta_dFirst_dSecond / (theta * theta)
          + 2 * theta_dFirst / (theta * theta * theta) * theta_dSecond;
    (void)dR_dFirst_dSecond;

    s_t horizontalDist = r - r * cos(theta);
    (void)horizontalDist;
    s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    s_t horizontalDist_dSecond
        = r_dSecond - (r_dSecond * cos(theta) - r * sin(theta) * theta_dSecond);
    (void)horizontalDist_dSecond;

    s_t horizontalDist_dFirst_dSecond
        = r_dFirst_dSecond
          - (r_dFirst_dSecond * cos(theta)
             - r_dFirst * sin(theta) * theta_dSecond
             - (r_dSecond * sin(theta) * theta_dFirst)
             - (r * cos(theta) * theta_dSecond * theta_dFirst)
             - (r * sin(theta) * theta_dFirst_dSecond));
    (void)horizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector4s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;
    const Eigen::Vector4s dHorizontalDist_dSecond
        = dR_dSecond
          + (r_dSecond * sin(theta) * dTheta
             + r * cos(theta) * theta_dSecond * dTheta
             + r * sin(theta) * dTheta_dSecond)
          - (dR_dSecond * cos(theta) - dR * sin(theta) * theta_dSecond);
    (void)dHorizontalDist_dSecond;
    const Eigen::Vector4s dHorizontalDist_dFirst_dSecond
        = dR_dFirst_dSecond
          + ((r_dFirst_dSecond * sin(theta) * dTheta)
             + (r_dFirst * cos(theta) * theta_dSecond * dTheta)
             + (r_dFirst * sin(theta) * dTheta_dSecond))
          + ((r_dSecond * cos(theta) * theta_dFirst * dTheta)
             + (r * -sin(theta) * theta_dSecond * theta_dFirst * dTheta)
             + (r * cos(theta) * theta_dFirst_dSecond * dTheta)
             + (r * cos(theta) * theta_dFirst * dTheta_dSecond))
          + ((r_dSecond * sin(theta) * dTheta_dFirst)
             + (r * cos(theta) * theta_dSecond * dTheta_dFirst)
             + (r * sin(theta) * dTheta_dFirst_dSecond))
          - ((dR_dFirst_dSecond * cos(theta))
             + (dR_dFirst * -sin(theta) * theta_dSecond))
          + ((dR_dSecond * sin(theta) * theta_dFirst)
             + (dR * cos(theta) * theta_dSecond * theta_dFirst)
             + (dR * sin(theta) * theta_dFirst_dSecond));
    (void)dHorizontalDist_dFirst_dSecond;

    const Eigen::Vector4s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector4s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;
    const Eigen::Vector4s dVerticalDist_dSecond
        = (r_dSecond * cos(theta) * dTheta
           - r * sin(theta) * theta_dSecond * dTheta
           + r * cos(theta) * dTheta_dSecond)
          + (dR_dSecond * sinTheta + dR * sinTheta_dSecond);
    (void)dVerticalDist_dSecond;

    const Eigen::Vector4s dVerticalDist_dFirst_dSecond
        = (((r_dFirst_dSecond * cos(theta) * dTheta)
            + (r_dFirst * -sin(theta) * theta_dSecond * dTheta)
            + (r_dFirst * cos(theta) * dTheta_dSecond))
           - ((r_dSecond * sin(theta) * theta_dFirst * dTheta)
              + (r * cos(theta) * theta_dSecond * theta_dFirst * dTheta)
              + (r * sin(theta) * theta_dFirst_dSecond * dTheta)
              + (r * sin(theta) * theta_dFirst * dTheta_dSecond))
           + ((r_dSecond * cos(theta) * dTheta_dFirst)
              + (r * -sin(theta) * theta_dSecond * dTheta_dFirst)
              + (r * cos(theta) * dTheta_dFirst_dSecond)))
          + ((dR_dFirst_dSecond * sinTheta) + (dR_dFirst * sinTheta_dSecond))
          + ((dR_dSecond * sinTheta_dFirst) + (dR * sinTheta_dFirst_dSecond));
    (void)dVerticalDist_dFirst_dSecond;

    Eigen::Matrix<s_t, 3, 4> dTranslation = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation.row(0)
        = (linearAngle(0) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(0)
          + (horizontalDist * linearAngle(0)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();
    dTranslation.row(1) = dVerticalDist;
    dTranslation.row(2)
        = (linearAngle(2) / sinTheta) * dHorizontalDist.transpose()
          + (horizontalDist / sinTheta) * dLinearAngle.row(2)
          + (horizontalDist * linearAngle(2)) * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst.row(0)
        = ((linearAngle_dFirst(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + ((horizontalDist_dFirst * linearAngle(0)
              + horizontalDist * linearAngle_dFirst(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());
    dTranslation_dFirst.row(1) = dVerticalDist_dFirst;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst.row(2)
        = ((linearAngle_dFirst(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dFirst.transpose())
          + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + ((horizontalDist_dFirst * linearAngle(2)
              + horizontalDist * linearAngle_dFirst(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dFirst.transpose());

    Eigen::Matrix<s_t, 3, 4> dTranslation_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dSecond.row(0)
        = ((linearAngle_dSecond(0) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(0) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(0)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(0)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(0))
          + ((horizontalDist_dSecond * linearAngle(0)
              + horizontalDist * linearAngle_dSecond(0))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(0))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());
    dTranslation_dSecond.row(1) = dVerticalDist_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dSecond.row(2)
        = ((linearAngle_dSecond(2) / sinTheta) * dHorizontalDist.transpose()
           - (linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                 * dHorizontalDist.transpose()
           + (linearAngle(2) / sinTheta) * dHorizontalDist_dSecond.transpose())
          + ((horizontalDist_dSecond / sinTheta) * dLinearAngle.row(2)
             - (horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                   * dLinearAngle.row(2)
             + (horizontalDist / sinTheta) * dLinearAngle_dSecond.row(2))
          + ((horizontalDist_dSecond * linearAngle(2)
              + horizontalDist * linearAngle_dSecond(2))
                 * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (2.0 * sinTheta_dSecond / (sinTheta * sinTheta * sinTheta))
                   * dSinTheta.transpose()
             + (horizontalDist * linearAngle(2))
                   * (-1.0 / (sinTheta * sinTheta))
                   * dSinTheta_dSecond.transpose());

    // return dHorizontalDist_dFirst_dSecond.transpose();

    Eigen::Matrix<s_t, 3, 4> dTranslation_dFirst_dSecond
        = Eigen::Matrix<s_t, 3, 4>::Zero();
    dTranslation_dFirst_dSecond.row(0)
        = (((linearAngle_dFirst_dSecond(0) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(0) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(0) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(0) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(0) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(0) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(0) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(0) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(0))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(0))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(0))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(0)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(0))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(0))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(0))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(0)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(0))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(0))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(0))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(0)))
          // + ((horizontalDist_dFirst * linearAngle(0)
          //     + horizontalDist * linearAngle_dFirst(0))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(0)
               + horizontalDist_dFirst * linearAngle_dSecond(0)
               + horizontalDist_dSecond * linearAngle_dFirst(0)
               + horizontalDist * linearAngle_dFirst_dSecond(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(0)
                 + horizontalDist * linearAngle_dFirst(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(0))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(0))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(0))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(0))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(0))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));
    dTranslation_dFirst_dSecond.row(1) = dVerticalDist_dFirst_dSecond;
    // This looks like a whole new mess, but it's actually identical to row(0),
    // except with the indices changed to 2
    dTranslation_dFirst_dSecond.row(2)
        = (((linearAngle_dFirst_dSecond(2) / sinTheta)
            * dHorizontalDist.transpose())
           + ((-linearAngle_dFirst(2) / (sinTheta * sinTheta))
              * sinTheta_dSecond * dHorizontalDist.transpose())
           + ((linearAngle_dFirst(2) / sinTheta)
              * dHorizontalDist_dSecond.transpose()))
          // - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
          //    * dHorizontalDist.transpose())
          - (((linearAngle_dSecond(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dHorizontalDist.transpose())
             + ((-2 * linearAngle(2) / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst
                * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dHorizontalDist.transpose())
             + ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dHorizontalDist_dSecond.transpose()))
          // + ((linearAngle(2) / sinTheta) *
          // dHorizontalDist_dFirst.transpose());
          + (((linearAngle_dSecond(2) / sinTheta)
              * dHorizontalDist_dFirst.transpose())
             - ((linearAngle(2) / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dHorizontalDist_dFirst.transpose())
             + ((linearAngle(2) / sinTheta)
                * dHorizontalDist_dFirst_dSecond.transpose()))
          // + ((horizontalDist_dFirst / sinTheta) * dLinearAngle.row(2))
          + (((horizontalDist_dFirst_dSecond / sinTheta) * dLinearAngle.row(2))
             + ((-horizontalDist_dFirst / (sinTheta * sinTheta))
                * sinTheta_dSecond * dLinearAngle.row(2))
             + ((horizontalDist_dFirst / sinTheta)
                * dLinearAngle_dSecond.row(2)))
          //  - ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
          //        * dLinearAngle.row(2))
          - (((horizontalDist_dSecond / (sinTheta * sinTheta)) * sinTheta_dFirst
              * dLinearAngle.row(2))
             + ((-2 * horizontalDist / (sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * sinTheta_dFirst * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta))
                * sinTheta_dFirst_dSecond * dLinearAngle.row(2))
             + ((horizontalDist / (sinTheta * sinTheta)) * sinTheta_dFirst
                * dLinearAngle_dSecond.row(2)))
          //    + ((horizontalDist / sinTheta) * dLinearAngle_dFirst.row(2))
          + (((horizontalDist_dSecond / sinTheta) * dLinearAngle_dFirst.row(2))
             + ((-horizontalDist / (sinTheta * sinTheta)) * sinTheta_dSecond
                * dLinearAngle_dFirst.row(2))
             + ((horizontalDist / sinTheta)
                * dLinearAngle_dFirst_dSecond.row(2)))
          // + ((horizontalDist_dFirst * linearAngle(2)
          //     + horizontalDist * linearAngle_dFirst(2))
          //    * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose());
          + (((horizontalDist_dFirst_dSecond * linearAngle(2)
               + horizontalDist_dFirst * linearAngle_dSecond(2)
               + horizontalDist_dSecond * linearAngle_dFirst(2)
               + horizontalDist * linearAngle_dFirst_dSecond(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta.transpose())
             + ((horizontalDist_dFirst * linearAngle(2)
                 + horizontalDist * linearAngle_dFirst(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta *
          //          sinTheta))
          //          * dSinTheta.transpose()
          + (((horizontalDist_dSecond * linearAngle(2))
              * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
              * dSinTheta.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst_dSecond
                   / (sinTheta * sinTheta * sinTheta))
                * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-6.0 * sinTheta_dFirst
                   / (sinTheta * sinTheta * sinTheta * sinTheta))
                * sinTheta_dSecond * dSinTheta.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 * sinTheta_dFirst / (sinTheta * sinTheta * sinTheta))
                * dSinTheta_dSecond.transpose()))
          //    + (horizontalDist * linearAngle(2))
          //          * (-1.0 / (sinTheta * sinTheta))
          //          * dSinTheta_dFirst.transpose());
          + (((horizontalDist_dSecond * linearAngle(2))
              * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle_dSecond(2))
                * (-1.0 / (sinTheta * sinTheta)) * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (2.0 / (sinTheta * sinTheta * sinTheta)) * sinTheta_dSecond
                * dSinTheta_dFirst.transpose())
             + ((horizontalDist * linearAngle(2))
                * (-1.0 / (sinTheta * sinTheta))
                * dSinTheta_dFirst_dSecond.transpose()));

    J_dFirst.block<3, 1>(3, 0)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 0);
    J_dFirst.block<3, 1>(3, 1)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 1);
    J_dFirst.block<3, 1>(3, 2)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 2);
    J_dFirst.block<3, 1>(3, 3)
        = rot.linear().transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation.block<3, 1>(0, 3);

    J_dFirst_dSecond.block<3, 1>(3, 0)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 0)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 0)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 0)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 0);
    J_dFirst_dSecond.block<3, 1>(3, 1)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 1)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 1)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 1)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 1);
    J_dFirst_dSecond.block<3, 1>(3, 2)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 2)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 2)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 2)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 2);
    J_dFirst_dSecond.block<3, 1>(3, 3)
        = rot_dSecond.transpose() * dTranslation_dFirst.block<3, 1>(0, 3)
          + rot.linear().transpose()
                * dTranslation_dFirst_dSecond.block<3, 1>(0, 3)
          + rot_dFirst_dSecond.transpose() * dTranslation.block<3, 1>(0, 3)
          + rot_dFirst.transpose() * dTranslation_dSecond.block<3, 1>(0, 3);
  }

  // Finally, take into account the transform to the child body node
  J_dFirst_dSecond
      = math::AdTJacFixed(getTransformFromChildBodyNode(), J_dFirst_dSecond);

  /////////// HERE - analytical
  return J_dFirst_dSecond;
}

Eigen::MatrixXs ConstantCurveJoint::finiteDifferenceScratch(
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