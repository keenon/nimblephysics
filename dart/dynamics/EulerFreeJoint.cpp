#include "dart/dynamics/EulerFreeJoint.hpp"

#include <memory>

#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/LinearFunction.hpp"

namespace dart {
namespace dynamics {

EulerFreeJoint::EulerFreeJoint(const Properties& props)
  : GenericJoint(props),
    mAxisOrder(dynamics::EulerJoint::AxisOrder::XYZ),
    mFlipAxisMap(Eigen::Vector3s::Ones()){};

//==============================================================================
const std::string& EulerFreeJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& EulerFreeJoint::getStaticType()
{
  static const std::string name = "EulerFreeJoint";
  return name;
}

//==============================================================================
bool EulerFreeJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
/// Set the axis order
/// \param[in] _order Axis order
/// \param[in] _renameDofs If true, the names of dofs in this joint will be
/// renmaed according to the axis order.
void EulerFreeJoint::setAxisOrder(
    EulerJoint::AxisOrder _order, bool _renameDofs)
{
  mAxisOrder = _order;
  if (_renameDofs)
  {
    updateDegreeOfFreedomNames();
  }

  Joint::notifyPositionUpdated();
  updateRelativeJacobian(true);
  Joint::incrementVersion();
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void EulerFreeJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s EulerFreeJoint::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
/// Return the axis order
EulerJoint::AxisOrder EulerFreeJoint::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
/// Returns the axis of rotation for DOF `index`, depending on the AxisOrder
Eigen::Vector3s EulerFreeJoint::getAxis(int index) const
{
  if (getAxisOrder() == EulerJoint::AxisOrder::XYZ)
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
  if (getAxisOrder() == EulerJoint::AxisOrder::XZY)
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
  if (getAxisOrder() == EulerJoint::AxisOrder::ZXY)
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
  if (getAxisOrder() == EulerJoint::AxisOrder::ZYX)
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
  if (index == 3)
  {
    return Eigen::Vector3s::UnitX();
  }
  if (index == 4)
  {
    return Eigen::Vector3s::UnitY();
  }
  if (index == 5)
  {
    return Eigen::Vector3s::UnitZ();
  }
  std::cout << "ERROR: EulerFreeJoint is being asked for an axis that is out "
               "of bounds!"
            << std::endl;
  assert(false);
  return Eigen::Vector3s::UnitX();
}

//==============================================================================
dart::dynamics::Joint* EulerFreeJoint::clone() const
{
  EulerFreeJoint* joint = new EulerFreeJoint(this->getJointProperties());
  for (int i = 0; i < getNumDofs(); i++)
  {
    joint->setDofName(i, getDofName(i));
  }
  joint->setName(getName());
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(mFlipAxisMap);
  joint->setAxisOrder(mAxisOrder);
  joint->setPositionUpperLimits(getPositionUpperLimits());
  joint->setPositionLowerLimits(getPositionLowerLimits());
  joint->setVelocityUpperLimits(getVelocityUpperLimits());
  joint->setVelocityLowerLimits(getVelocityLowerLimits());
  return joint;
}

//==============================================================================
void EulerFreeJoint::updateDegreeOfFreedomNames()
{
  std::vector<std::string> affixes;
  switch (getAxisOrder())
  {
    case EulerJoint::AxisOrder::ZYX:
      affixes.push_back("_rot_z");
      affixes.push_back("_rot_y");
      affixes.push_back("_rot_x");
      break;
    case EulerJoint::AxisOrder::XYZ:
      affixes.push_back("_rot_x");
      affixes.push_back("_rot_y");
      affixes.push_back("_rot_z");
      break;
    case EulerJoint::AxisOrder::XZY:
      affixes.push_back("_rot_x");
      affixes.push_back("_rot_z");
      affixes.push_back("_rot_y");
      break;
    case EulerJoint::AxisOrder::ZXY:
      affixes.push_back("_rot_z");
      affixes.push_back("_rot_x");
      affixes.push_back("_rot_y");
      break;
    default:
      dterr << "Unsupported axis order in EulerFreeJoint named '"
            << Joint::mAspectProperties.mName << "' ("
            << static_cast<int>(getAxisOrder()) << ")\n";
  }
  affixes.push_back("_trans_x");
  affixes.push_back("_trans_y");
  affixes.push_back("_trans_z");

  if (affixes.size() == 6)
  {
    for (std::size_t i = 0; i < 6; ++i)
    {
      if (!mDofs[i]->isNamePreserved())
        mDofs[i]->setName(Joint::mAspectProperties.mName + affixes[i], false);
    }
  }
}

//==============================================================================
void EulerFreeJoint::updateRelativeTransform() const
{
  Eigen::Isometry3s T = EulerJoint::convertToTransform(
      getPositionsStatic().head<3>(), getAxisOrder(), getFlipAxisMap());
  T.translation() = getPositionsStatic().tail<3>();

  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * T
       * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
/// Fixed-size version of getRelativeJacobian(positions)
Eigen::Matrix6s EulerFreeJoint::getRelativeJacobianStatic(
    const Eigen::Vector6s& positions) const
{
  return computeRelativeJacobianStatic(
      positions,
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::getRelativeJacobianDerivWrtPositionStatic(
    std::size_t index) const
{
  return computeRelativeJacobianStaticDerivWrtPos(
      getPositionsStatic(),
      index,
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
void EulerFreeJoint::updateRelativeJacobian(bool) const
{
  mJacobian = computeRelativeJacobianStatic(
      getPositionsStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
void EulerFreeJoint::updateRelativeJacobianTimeDeriv() const
{
  mJacobianDeriv = computeRelativeJacobianTimeDerivStatic(
      getPositionsStatic(),
      getVelocitiesStatic(),
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
/// Computes derivative of time derivative of Jacobian w.r.t. position.
math::Jacobian EulerFreeJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  return computeRelativeJacobianTimeDerivDerivWrtPos(
      getPositionsStatic(),
      getVelocitiesStatic(),
      index,
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
/// Computes derivative of time derivative of Jacobian w.r.t. velocity.
math::Jacobian EulerFreeJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  return computeRelativeJacobianTimeDerivDerivWrtVel(
      getPositionsStatic(),
      index,
      getAxisOrder(),
      getFlipAxisMap(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianStatic(
    const Eigen::Vector6s& positions,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  Eigen::Vector3s euler = positions.head<3>();
  Eigen::Isometry3s T
      = EulerJoint::convertToTransform(euler, axisOrder, flipAxisMap)
        * childBodyToJoint.inverse();

  Eigen::Matrix6s spatialJac = Eigen::Matrix6s::Identity();
  spatialJac.block<3, 3>(3, 3) = T.linear().transpose(); // R^T
  spatialJac.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
      euler, axisOrder, flipAxisMap, childBodyToJoint);

  return spatialJac;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianStaticDerivWrtPos(
    const Eigen::Vector6s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();

    Eigen::Matrix6s spatialJac = Eigen::Matrix6s::Identity();
    if (axisOrder == EulerJoint::AxisOrder::XYZ)
    {
      spatialJac.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerXYZToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index); // R^T
    }
    if (axisOrder == EulerJoint::AxisOrder::XZY)
    {
      spatialJac.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerXZYToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index); // R^T
    }
    if (axisOrder == EulerJoint::AxisOrder::ZYX)
    {
      spatialJac.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerZYXToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index); // R^T
    }
    if (axisOrder == EulerJoint::AxisOrder::ZXY)
    {
      spatialJac.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerZXYToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index); // R^T
    }
    spatialJac.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianDerivWrtPos(
            index, euler, axisOrder, flipAxisMap, childBodyToJoint);

    return spatialJac;
  }
  else
  {
    return Eigen::Matrix6s::Zero();
  }
}

//==============================================================================
Eigen::Matrix6s
EulerFreeJoint::finiteDifferenceRelativeJacobianStaticDerivWrtPos(
    const Eigen::Vector6s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint,
    bool useRidders)
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::Vector6s tweaked
            = positions + (eps * Eigen::Vector6s::Unit(index));
        perturbed = computeRelativeJacobianStatic(
            tweaked, axisOrder, flipAxisMap, childBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  Eigen::Vector3s euler = positions.head<3>();
  Eigen::Vector3s eulerVel = velocities.head<3>();

  Eigen::Matrix6s spatialJacDeriv = Eigen::Matrix6s::Zero();
  spatialJacDeriv.block<6, 3>(0, 0)
      = EulerJoint::computeRelativeJacobianTimeDerivStatic(
          euler, eulerVel, axisOrder, flipAxisMap, childBodyToJoint);

  if (axisOrder == EulerJoint::AxisOrder::XYZ)
  {
    for (int i = 0; i < 3; i++)
    {
      spatialJacDeriv.block<3, 3>(3, 3)
          += childBodyToJoint.linear()
             * math::eulerXYZToMatrixGrad(euler.cwiseProduct(flipAxisMap), i)
                   .transpose()
             * eulerVel(i) * flipAxisMap(i);
    }
  }
  else if (axisOrder == EulerJoint::AxisOrder::XZY)
  {
    for (int i = 0; i < 3; i++)
    {
      spatialJacDeriv.block<3, 3>(3, 3)
          += childBodyToJoint.linear()
             * math::eulerXZYToMatrixGrad(euler.cwiseProduct(flipAxisMap), i)
                   .transpose()
             * eulerVel(i) * flipAxisMap(i);
    }
  }
  else if (axisOrder == EulerJoint::AxisOrder::ZXY)
  {
    for (int i = 0; i < 3; i++)
    {
      spatialJacDeriv.block<3, 3>(3, 3)
          += childBodyToJoint.linear()
             * math::eulerZXYToMatrixGrad(euler.cwiseProduct(flipAxisMap), i)
                   .transpose()
             * eulerVel(i) * flipAxisMap(i);
    }
  }
  else if (axisOrder == EulerJoint::AxisOrder::ZYX)
  {
    for (int i = 0; i < 3; i++)
    {
      spatialJacDeriv.block<3, 3>(3, 3)
          += childBodyToJoint.linear()
             * math::eulerZYXToMatrixGrad(euler.cwiseProduct(flipAxisMap), i)
                   .transpose()
             * eulerVel(i) * flipAxisMap(i);
    }
  }
  return spatialJacDeriv;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint,
    bool useRidders)
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-8;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::Vector6s tweaked = positions + (eps * velocities);
        perturbed = computeRelativeJacobianStatic(
            tweaked, axisOrder, flipAxisMap, childBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();
    Eigen::Vector3s eulerVel = velocities.head<3>();

    Eigen::Matrix6s d_dJ = Eigen::Matrix6s::Zero();
    d_dJ.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
            index, euler, eulerVel, axisOrder, flipAxisMap, childBodyToJoint);

    if (axisOrder == EulerJoint::AxisOrder::XYZ)
    {
      for (int i = 0; i < 3; i++)
      {
        d_dJ.block<3, 3>(3, 3)
            += childBodyToJoint.linear()
               * math::eulerXYZToMatrixSecondGrad(
                     euler.cwiseProduct(flipAxisMap), i, index)
                     .transpose()
               * eulerVel(i) * flipAxisMap(i);
      }
    }
    else if (axisOrder == EulerJoint::AxisOrder::XZY)
    {
      for (int i = 0; i < 3; i++)
      {
        d_dJ.block<3, 3>(3, 3)
            += childBodyToJoint.linear()
               * math::eulerXZYToMatrixSecondGrad(
                     euler.cwiseProduct(flipAxisMap), i, index)
                     .transpose()
               * eulerVel(i) * flipAxisMap(i);
      }
    }
    else if (axisOrder == EulerJoint::AxisOrder::ZXY)
    {
      for (int i = 0; i < 3; i++)
      {
        d_dJ.block<3, 3>(3, 3)
            += childBodyToJoint.linear()
               * math::eulerZXYToMatrixSecondGrad(
                     euler.cwiseProduct(flipAxisMap), i, index)
                     .transpose()
               * eulerVel(i) * flipAxisMap(i);
      }
    }
    else if (axisOrder == EulerJoint::AxisOrder::ZYX)
    {
      for (int i = 0; i < 3; i++)
      {
        d_dJ.block<3, 3>(3, 3)
            += childBodyToJoint.linear()
               * math::eulerZYXToMatrixSecondGrad(
                     euler.cwiseProduct(flipAxisMap), i, index)
                     .transpose()
               * eulerVel(i) * flipAxisMap(i);
      }
    }
    d_dJ.block<3, 3>(3, 3) *= flipAxisMap(index);
    return d_dJ;
  }
  else
  {
    return Eigen::Matrix6s::Zero();
  }
}

//==============================================================================
Eigen::Matrix6s
EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint,
    bool useRidders)
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-8;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::Vector6s tweaked
            = positions + (eps * Eigen::Vector6s::Unit(index));
        perturbed = computeRelativeJacobianTimeDerivStatic(
            tweaked, velocities, axisOrder, flipAxisMap, childBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
    const Eigen::Vector6s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint)
{
  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();

    Eigen::Matrix6s d_dJ = Eigen::Matrix6s::Zero();
    d_dJ.block<6, 3>(0, 0)
        = EulerJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
            index, euler, axisOrder, flipAxisMap, childBodyToJoint);

    if (axisOrder == EulerJoint::AxisOrder::XYZ)
    {
      d_dJ.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerXYZToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index);
    }
    else if (axisOrder == EulerJoint::AxisOrder::XZY)
    {
      d_dJ.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerXZYToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index);
    }
    else if (axisOrder == EulerJoint::AxisOrder::ZXY)
    {
      d_dJ.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerZXYToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index);
    }
    else if (axisOrder == EulerJoint::AxisOrder::ZYX)
    {
      d_dJ.block<3, 3>(3, 3)
          = childBodyToJoint.linear()
            * math::eulerZYXToMatrixGrad(euler.cwiseProduct(flipAxisMap), index)
                  .transpose()
            * flipAxisMap(index);
    }
    return d_dJ;
  }
  else
  {
    return Eigen::Matrix6s::Zero();
  }
}

//==============================================================================
Eigen::Matrix6s
EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVel(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    const Eigen::Vector3s& flipAxisMap,
    const Eigen::Isometry3s& childBodyToJoint,
    bool useRidders)
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-8;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::Vector6s tweaked
            = velocities + (eps * Eigen::Vector6s::Unit(index));
        perturbed = computeRelativeJacobianTimeDerivStatic(
            positions, tweaked, axisOrder, flipAxisMap, childBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

} // namespace dynamics
} // namespace dart