#include "dart/dynamics/ConstantCurveIncompressibleJoint.hpp"

#include <memory>
#include <ostream>
#include <string>

#include "dart/dynamics/ConstantCurveJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

ConstantCurveIncompressibleJoint::ConstantCurveIncompressibleJoint(
    const detail::GenericJointProperties<math::RealVectorSpace<3>>& props)
  : GenericJoint<math::RealVectorSpace<3>>(props),
    mFlipAxisMap(Eigen::Vector3s::Ones()),
    mNeutralPos(Eigen::Vector3s::Zero()),
    mLength(1.0)
{
}

//==============================================================================
const std::string& ConstantCurveIncompressibleJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& ConstantCurveIncompressibleJoint::getStaticType()
{
  static const std::string name = "ConstantCurveIncompressibleJoint";
  return name;
}

//==============================================================================
bool ConstantCurveIncompressibleJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void ConstantCurveIncompressibleJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s ConstantCurveIncompressibleJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
void ConstantCurveIncompressibleJoint::setNeutralPos(Eigen::Vector3s pos)
{
  mNeutralPos = pos;
}

//==============================================================================
Eigen::Vector3s ConstantCurveIncompressibleJoint::getNeutralPos() const
{
  return mNeutralPos;
}

//==============================================================================
void ConstantCurveIncompressibleJoint::setLength(s_t len)
{
  mLength = len;
}

//==============================================================================
s_t ConstantCurveIncompressibleJoint::getLength() const
{
  return mLength;
}

//==============================================================================
dart::dynamics::Joint* ConstantCurveIncompressibleJoint::clone() const
{
  ConstantCurveIncompressibleJoint* joint
      = new ConstantCurveIncompressibleJoint(this->getJointProperties());
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(getFlipAxisMap());
  joint->setName(this->getName());
  joint->setNeutralPos(this->getNeutralPos());
  joint->setPositionUpperLimits(this->getPositionUpperLimits());
  joint->setPositionLowerLimits(this->getPositionLowerLimits());
  joint->setVelocityUpperLimits(this->getVelocityUpperLimits());
  joint->setVelocityLowerLimits(this->getVelocityLowerLimits());
  return joint;
}

//==============================================================================
dart::dynamics::Joint* ConstantCurveIncompressibleJoint::simplifiedClone() const
{
  // TOOD: we need to actually find a good simplification for this joint in
  // terms of simpler joint types, maybe a ball joint with an offset?
  assert(false);
  return clone();
}

//==============================================================================
void ConstantCurveIncompressibleJoint::updateDegreeOfFreedomNames()
{
  if (!this->mDofs[0]->isNamePreserved())
    this->mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
Eigen::Isometry3s ConstantCurveIncompressibleJoint::getRelativeTransformAt(
    Eigen::Vector3s inPos, s_t d)
{
  Eigen::Vector3s pos = inPos + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos, EulerJoint::AxisOrder::XZY, mFlipAxisMap);

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
  return Joint::mAspectProperties.mT_ParentBodyToJoint * bentRod
         * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
void ConstantCurveIncompressibleJoint::updateRelativeTransform() const
{
  Eigen::Vector3s pos = this->getPositionsStatic() + mNeutralPos;
  s_t scale = this->getChildScale()(1);
  s_t d = mLength * scale;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos, EulerJoint::AxisOrder::XZY, mFlipAxisMap);

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
Eigen::Matrix<s_t, 6, 3>
ConstantCurveIncompressibleJoint::getRelativeJacobianStatic(
    const Eigen::Vector3s& rawPos) const
{
  // Think in terms of the child frame

  Eigen::Vector3s pos = rawPos + mNeutralPos;

  // 1. Do the euler rotation
  Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  Eigen::Matrix<s_t, 6, 3> J = Eigen::Matrix<s_t, 6, 3>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  J.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
      pos.head<3>(),
      EulerJoint::AxisOrder::XZY,
      mFlipAxisMap.head<3>(),
      identity);

  s_t scale = this->getChildScale()(1);
  s_t d = mLength * scale;

  // Remember, this is X,*Z*,Y

  s_t cx = cos(pos(0));
  s_t sx = sin(pos(0));
  s_t cz = cos(pos(1));
  s_t sz = sin(pos(1));

  Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);
  Eigen::Matrix<s_t, 3, 3> dLinearAngle = Eigen::Matrix<s_t, 3, 3>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();

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
  }
  else
  {
    // Compute the bend as a function of the angle from vertical
    Eigen::Vector3s dSinTheta;
    for (int i = 0; i < 3; i++)
    {
      dSinTheta(i) = (0.5
                      / sqrt(
                          linearAngle(0) * linearAngle(0)
                          + linearAngle(2) * linearAngle(2)))
                     * (2 * linearAngle(0) * dLinearAngle(0, i)
                        + 2 * linearAngle(2) * dLinearAngle(2, i));
    }

    s_t theta = asin(sinTheta);
    Eigen::Vector3s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;

    s_t r = (d / theta);
    Eigen::Vector3s dR = (-d / (theta * theta)) * dTheta;

    s_t horizontalDist = r - r * cos(theta);

    Eigen::Vector3s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    Eigen::Vector3s dVerticalDist = r * cos(theta) * dTheta + dR * sinTheta;

    Eigen::Matrix<s_t, 3, 3> dTranslation = Eigen::Matrix<s_t, 3, 3>::Zero();
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
  }

  // Finally, take into account the transform to the child body node
  J = math::AdTJacFixed(getTransformFromChildBodyNode(), J);
  return J;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3>
ConstantCurveIncompressibleJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  // Think in terms of the child frame

  const Eigen::Vector3s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  const Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  const Eigen::Matrix3s rot_dFirst
      = math::eulerXZYToMatrixGrad(pos.head<3>(), index);

  Eigen::Matrix<s_t, 6, 3> J_dFirst = Eigen::Matrix<s_t, 6, 3>::Zero();

  // 2. Compute the Jacobian of the Euler transformation
  const Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();
  J_dFirst.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDerivWrtPos(
      index,
      pos.head<3>(),
      EulerJoint::AxisOrder::XZY,
      mFlipAxisMap.head<3>(),
      identity);

  s_t scale = this->getChildScale()(1);
  const s_t d = mLength * scale;

  // Remember, this is X,*Z*,Y

  const s_t cx = cos(pos(0));
  const s_t sx = sin(pos(0));
  const s_t cz = cos(pos(1));
  const s_t sz = sin(pos(1));

  const Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 3> dLinearAngle = Eigen::Matrix<s_t, 3, 3>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();

  Eigen::Matrix<s_t, 3, 3> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 3>::Zero();
  Eigen::Vector3s linearAngle_dFirst = Eigen::Vector3s::Zero();
  if (index == 0)
  {
    linearAngle_dFirst = Eigen::Vector3s(0, -sx * cz, cz * cx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -cx * cz, cz * -sx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(0, sx * sz, -sz * cx);
    dLinearAngle_dFirst.col(2).setZero();
  }
  else if (index == 1)
  {
    linearAngle_dFirst = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dFirst.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dFirst.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dFirst.col(2).setZero();
  }

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.001)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    const Eigen::Matrix<s_t, 6, 3> J
        = EulerJoint::computeRelativeJacobianStatic(
            pos.head<3>(),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    const Eigen::Vector3s translation_dFirst
        = rot.linear() * J.block<3, 1>(0, index).cross(bentRod.translation());

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
  }
  else
  {
    Eigen::Vector3s dSinTheta;
    Eigen::Vector3s dSinTheta_dFirst;
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

    const s_t sinTheta_dFirst
        = (0.5
           / sqrt(
               linearAngle(0) * linearAngle(0)
               + linearAngle(2) * linearAngle(2)))
          * (2 * linearAngle(0) * linearAngle_dFirst(0)
             + 2 * linearAngle(2) * linearAngle_dFirst(2));

    // Compute the bend as a function of the angle from vertical
    const s_t theta = asin(sinTheta);
    const s_t theta_dFirst
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * sinTheta_dFirst;
    (void)theta_dFirst;

    const Eigen::Vector3s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector3s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;

    const s_t r = (d / theta);
    const s_t r_dFirst = (-d / (theta * theta)) * theta_dFirst;
    (void)r_dFirst;

    const Eigen::Vector3s dR = (-d / (theta * theta)) * dTheta;
    const Eigen::Vector3s dR_dFirst
        = (2 * d / (theta * theta * theta)) * theta_dFirst * dTheta
          + (-d / (theta * theta)) * dTheta_dFirst;
    (void)dR_dFirst;

    const s_t horizontalDist = r - r * cos(theta);
    const s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    const Eigen::Vector3s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector3s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;

    const Eigen::Vector3s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector3s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;

    Eigen::Matrix<s_t, 3, 3> dTranslation = Eigen::Matrix<s_t, 3, 3>::Zero();
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

    Eigen::Matrix<s_t, 3, 3> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 3>::Zero();
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
  }

  // Finally, take into account the transform to the child body node
  J_dFirst = math::AdTJacFixed(getTransformFromChildBodyNode(), J_dFirst);
  return J_dFirst;
}

//==============================================================================
Eigen::Matrix<s_t, 6, 3> ConstantCurveIncompressibleJoint::
    getRelativeJacobianDerivWrtPositionDerivWrtPositionStatic(
        std::size_t firstIndex, std::size_t secondIndex) const
{
  (void)secondIndex;

  // Think in terms of the child frame

  const Eigen::Vector3s pos = getPositionsStatic() + mNeutralPos;

  // 1. Do the euler rotation
  const Eigen::Isometry3s rot = EulerJoint::convertToTransform(
      pos.head<3>(), EulerJoint::AxisOrder::XZY, mFlipAxisMap.head<3>());
  const Eigen::Matrix3s rot_dFirst
      = math::eulerXZYToMatrixGrad(pos.head<3>(), firstIndex);
  const Eigen::Matrix3s rot_dSecond
      = math::eulerXZYToMatrixGrad(pos.head<3>(), secondIndex);
  const Eigen::Matrix3s rot_dFirst_dSecond = math::eulerXZYToMatrixSecondGrad(
      pos.head<3>(), firstIndex, secondIndex);

  const Eigen::Isometry3s identity = Eigen::Isometry3s::Identity();

  // 2. Compute the Jacobian of the Euler transformation
  Eigen::Matrix<s_t, 6, 3> J_dFirst
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          firstIndex,
          pos.head<3>(),
          EulerJoint::AxisOrder::XZY,
          mFlipAxisMap.head<3>(),
          identity);
  Eigen::Matrix<s_t, 6, 3> J_dSecond
      = EulerJoint::computeRelativeJacobianDerivWrtPos(
          secondIndex,
          pos.head<3>(),
          EulerJoint::AxisOrder::XZY,
          mFlipAxisMap.head<3>(),
          identity);
  Eigen::Matrix<s_t, 6, 3> J_dFirst_dSecond
      = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
          secondIndex,
          pos.head<3>(),
          Eigen::Vector3s::Unit(firstIndex),
          EulerJoint::AxisOrder::XZY,
          mFlipAxisMap.head<3>(),
          identity);

  s_t scale = this->getChildScale()(1);
  const s_t d = mLength * scale;

  // Remember, this is X,*Z*,Y

  const s_t cx = cos(pos(0));
  const s_t sx = sin(pos(0));
  const s_t cz = cos(pos(1));
  const s_t sz = sin(pos(1));

  const Eigen::Vector3s linearAngle = Eigen::Vector3s(-sz, cx * cz, cz * sx);

  Eigen::Matrix<s_t, 3, 3> dLinearAngle = Eigen::Matrix<s_t, 3, 3>::Zero();
  dLinearAngle.col(0) = Eigen::Vector3s(0, -sx * cz, cz * cx);
  dLinearAngle.col(1) = Eigen::Vector3s(-cz, -cx * sz, -sz * sx);
  dLinearAngle.col(2).setZero();

  Eigen::Matrix<s_t, 3, 3> dLinearAngle_dFirst
      = Eigen::Matrix<s_t, 3, 3>::Zero();
  Eigen::Matrix<s_t, 3, 3> dLinearAngle_dSecond
      = Eigen::Matrix<s_t, 3, 3>::Zero();
  Eigen::Matrix<s_t, 3, 3> dLinearAngle_dFirst_dSecond
      = Eigen::Matrix<s_t, 3, 3>::Zero();
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
  }
  else if (secondIndex == 1)
  {
    linearAngle_dSecond = Eigen::Vector3s(-cz, cx * -sz, -sz * sx);
    dLinearAngle_dSecond.col(0) = Eigen::Vector3s(0, -sx * -sz, -sz * cx);
    dLinearAngle_dSecond.col(1) = Eigen::Vector3s(sz, -cx * cz, -cz * sx);
    dLinearAngle_dSecond.col(2).setZero();
  }

  const s_t sinTheta
      = sqrt(linearAngle(0) * linearAngle(0) + linearAngle(2) * linearAngle(2));

  if (sinTheta < 0.003)
  {
    // Near very vertical angles, don't worry about the bend, just approximate
    // with an euler joint
    const Eigen::Matrix<s_t, 6, 3> J
        = EulerJoint::computeRelativeJacobianStatic(
            pos.head<3>(),
            EulerJoint::AxisOrder::XZY,
            mFlipAxisMap.head<3>(),
            identity);

    // 2. Computing translation from vertical
    Eigen::Isometry3s bentRod = Eigen::Isometry3s::Identity();

    bentRod.translation() = Eigen::Vector3s::UnitY() * d;
    bentRod = rot * bentRod;

    const Eigen::Vector3s translation_dSecond
        = rot.linear()
          * J.block<3, 1>(0, secondIndex).cross(bentRod.translation());

    const Eigen::Vector3s translation_dFirst
        = rot.linear()
          * J.block<3, 1>(0, firstIndex).cross(bentRod.translation());
    const Eigen::Vector3s translation_dFirst_dSecond
        = (rot_dSecond
           * J.block<3, 1>(0, firstIndex).cross(bentRod.translation()))
          + (rot.linear()
             * J_dSecond.block<3, 1>(0, firstIndex)
                   .cross(bentRod.translation()))
          + (rot.linear()
             * J.block<3, 1>(0, firstIndex).cross(translation_dSecond));

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
  }
  else
  {
    Eigen::Vector3s dSinTheta = Eigen::Vector3s::Zero();
    Eigen::Vector3s dSinTheta_dFirst = Eigen::Vector3s::Zero();
    Eigen::Vector3s dSinTheta_dFirst_dSecond = Eigen::Vector3s::Zero();
    Eigen::Vector3s dSinTheta_dSecond = Eigen::Vector3s::Zero();
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

    const Eigen::Vector3s dTheta
        = (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta;
    const Eigen::Vector3s dTheta_dFirst
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dFirst * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dFirst;
    (void)dTheta_dFirst;
    const Eigen::Vector3s dTheta_dSecond
        = (1.0 / pow(1.0 - (sinTheta * sinTheta), 1.5)) * sinTheta
              * sinTheta_dSecond * dSinTheta
          + (1.0 / sqrt(1.0 - (sinTheta * sinTheta))) * dSinTheta_dSecond;
    (void)dTheta_dSecond;

    const Eigen::Vector3s dTheta_dFirst_dSecond
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
    const s_t r_dFirst = (-d / (theta * theta)) * theta_dFirst;
    (void)r_dFirst;
    const s_t r_dSecond = (-d / (theta * theta)) * theta_dSecond;
    (void)r_dSecond;
    const s_t r_dFirst_dSecond
        = ((2 * d / (theta * theta * theta)) * theta_dSecond * theta_dFirst)
          + ((-d / (theta * theta)) * theta_dFirst_dSecond);
    (void)r_dFirst_dSecond;

    const Eigen::Vector3s dR = (-d / (theta * theta)) * dTheta.segment<3>(0);

    const Eigen::Vector3s dR_dFirst
        = (2 * d / (theta * theta * theta)) * theta_dFirst
              * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dFirst.segment<3>(0);

    const Eigen::Vector3s dR_dSecond
        = (2 * d / (theta * theta * theta)) * theta_dSecond
              * dTheta.segment<3>(0)
          + (-d / (theta * theta)) * dTheta_dSecond.segment<3>(0);
    (void)dR_dSecond;

    const Eigen::Vector3s dR_dFirst_dSecond
        = ((-6 * d / (theta * theta * theta * theta)) * theta_dSecond
           * theta_dFirst * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst_dSecond
             * dTheta.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dFirst
             * dTheta_dSecond.segment<3>(0))
          + ((2 * d / (theta * theta * theta)) * theta_dSecond
             * dTheta_dFirst.segment<3>(0))
          + ((-d / (theta * theta)) * dTheta_dFirst_dSecond.segment<3>(0));
    (void)dR_dFirst_dSecond;

    const s_t horizontalDist = r - r * cos(theta);
    (void)horizontalDist;
    const s_t horizontalDist_dFirst
        = r_dFirst - (r_dFirst * cos(theta) - r * sin(theta) * theta_dFirst);
    (void)horizontalDist_dFirst;

    const s_t horizontalDist_dSecond
        = r_dSecond - (r_dSecond * cos(theta) - r * sin(theta) * theta_dSecond);
    (void)horizontalDist_dSecond;

    const s_t horizontalDist_dFirst_dSecond
        = r_dFirst_dSecond
          - (r_dFirst_dSecond * cos(theta)
             - r_dFirst * sin(theta) * theta_dSecond
             - (r_dSecond * sin(theta) * theta_dFirst)
             - (r * cos(theta) * theta_dSecond * theta_dFirst)
             - (r * sin(theta) * theta_dFirst_dSecond));
    (void)horizontalDist_dFirst_dSecond;

    const Eigen::Vector3s dHorizontalDist
        = dR + r * sin(theta) * dTheta - dR * cos(theta);
    const Eigen::Vector3s dHorizontalDist_dFirst
        = dR_dFirst
          + (r_dFirst * sin(theta) * dTheta
             + r * cos(theta) * theta_dFirst * dTheta
             + r * sin(theta) * dTheta_dFirst)
          - (dR_dFirst * cos(theta) - dR * sin(theta) * theta_dFirst);
    (void)dHorizontalDist_dFirst;
    const Eigen::Vector3s dHorizontalDist_dSecond
        = dR_dSecond
          + (r_dSecond * sin(theta) * dTheta
             + r * cos(theta) * theta_dSecond * dTheta
             + r * sin(theta) * dTheta_dSecond)
          - (dR_dSecond * cos(theta) - dR * sin(theta) * theta_dSecond);
    (void)dHorizontalDist_dSecond;
    const Eigen::Vector3s dHorizontalDist_dFirst_dSecond
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

    const Eigen::Vector3s dVerticalDist
        = r * cos(theta) * dTheta + dR * sinTheta;
    const Eigen::Vector3s dVerticalDist_dFirst
        = (r_dFirst * cos(theta) * dTheta
           - r * sin(theta) * theta_dFirst * dTheta
           + r * cos(theta) * dTheta_dFirst)
          + (dR_dFirst * sinTheta + dR * sinTheta_dFirst);
    (void)dVerticalDist_dFirst;
    const Eigen::Vector3s dVerticalDist_dSecond
        = (r_dSecond * cos(theta) * dTheta
           - r * sin(theta) * theta_dSecond * dTheta
           + r * cos(theta) * dTheta_dSecond)
          + (dR_dSecond * sinTheta + dR * sinTheta_dSecond);
    (void)dVerticalDist_dSecond;

    const Eigen::Vector3s dVerticalDist_dFirst_dSecond
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

    Eigen::Matrix<s_t, 3, 3> dTranslation = Eigen::Matrix<s_t, 3, 3>::Zero();
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

    Eigen::Matrix<s_t, 3, 3> dTranslation_dFirst
        = Eigen::Matrix<s_t, 3, 3>::Zero();
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

    Eigen::Matrix<s_t, 3, 3> dTranslation_dSecond
        = Eigen::Matrix<s_t, 3, 3>::Zero();
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

    Eigen::Matrix<s_t, 3, 3> dTranslation_dFirst_dSecond
        = Eigen::Matrix<s_t, 3, 3>::Zero();
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
Eigen::Vector3s
ConstantCurveIncompressibleJoint::getWorldTranslationOfChildBodyWrtChildScale(
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
/// Gets the derivative of the spatial Jacobian of the child BodyNode relative
/// to the parent BodyNode expressed in the child BodyNode frame, with respect
/// to the scaling of the child body along a specific axis.
///
/// Use axis = -1 for uniform scaling of all the axis.
math::Jacobian
ConstantCurveIncompressibleJoint::getRelativeJacobianDerivWrtChildScale(
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
    s_t scale = this->getChildScale()(1);
    s_t d = mLength * scale;
    J += ConstantCurveJoint::getRelativeJacobianDerivWrtSegmentLengthStatic(
             d,
             mLength,
             scale,
             getPositionsStatic().head<3>(),
             mNeutralPos.head<3>(),
             mFlipAxisMap,
             getTransformFromChildBodyNode())
             .block(0, 0, 6, 3);
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
math::Jacobian ConstantCurveIncompressibleJoint::
    getRelativeJacobianTimeDerivDerivWrtChildScale(int axis) const
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
    Eigen::Vector3s vel = getVelocitiesStatic();
    s_t originalD = mLength;
    s_t scale = this->getChildScale()(1);
    s_t d = originalD * scale;
    for (int i = 0; i < 3; i++)
    {
      J += vel(i)
           * ConstantCurveJoint::
                 getRelativeJacobianDerivWrtPositionDerivWrtSegmentLengthStatic(
                     i,
                     d,
                     originalD,
                     scale,
                     getPositionsStatic().head<3>(),
                     mNeutralPos.head<3>(),
                     mFlipAxisMap,
                     getTransformFromChildBodyNode())
                     .block(0, 0, 6, 3);
    }
  }

  return J;
}

//==============================================================================
void ConstantCurveIncompressibleJoint::updateRelativeJacobian(bool) const
{
  this->mJacobian = getRelativeJacobianStatic(this->getPositionsStatic());
}

//==============================================================================
void ConstantCurveIncompressibleJoint::updateRelativeJacobianTimeDeriv() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Matrix<s_t, 6, 3> dJ = Eigen::Matrix<s_t, 6, 3>::Zero();
  for (int i = 0; i < 3; i++)
  {
    // getRelativeJacobianDerivWrtPositionStatic
    Eigen::Matrix<s_t, 6, 3> J_di
        = getRelativeJacobianDerivWrtPositionStatic(i);
    dJ += J_di * vel(i);
  }
  this->mJacobianDeriv = dJ;
}

//==============================================================================
math::Jacobian
ConstantCurveIncompressibleJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
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
math::Jacobian
ConstantCurveIncompressibleJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return getRelativeJacobianDerivWrtPositionStatic(index);
}

//==============================================================================
// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s
ConstantCurveIncompressibleJoint::getScrewAxisGradientForPosition(
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
Eigen::Vector6s ConstantCurveIncompressibleJoint::getScrewAxisGradientForForce(
    int axisDof, int rotateDof)
{
  // Defaults to Finite Differencing - this is slow, but at least it's
  // approximately correct. Child joints should override with a faster
  // implementation.
  return Joint::finiteDifferenceScrewAxisGradientForForce(axisDof, rotateDof);
}

// For testing
Eigen::MatrixXs ConstantCurveIncompressibleJoint::getScratch(int firstIndex)
{
  (void)firstIndex;
  return Eigen::MatrixXs::Zero(1, 1);
}

Eigen::MatrixXs ConstantCurveIncompressibleJoint::analyticalScratch(
    int firstIndex, int secondIndex)
{
  (void)firstIndex;
  (void)secondIndex;
  return Eigen::MatrixXs::Zero(1, 1);
}

Eigen::MatrixXs ConstantCurveIncompressibleJoint::finiteDifferenceScratch(
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