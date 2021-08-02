#include "dart/dynamics/EulerFreeJoint.hpp"

#include <memory>

#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/LinearFunction.hpp"

namespace dart {
namespace dynamics {

EulerFreeJoint::EulerFreeJoint(const Properties& props)
  : GenericJoint(props), mAxisOrder(dynamics::EulerJoint::AxisOrder::XYZ){};

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
/// Return the axis order
EulerJoint::AxisOrder EulerFreeJoint::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
dart::dynamics::Joint* EulerFreeJoint::clone() const
{
  EulerFreeJoint* joint = new EulerFreeJoint(this->getJointProperties());
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
      getPositionsStatic().head<3>(), getAxisOrder());
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
      positions, getAxisOrder(), Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
math::Jacobian EulerFreeJoint::getRelativeJacobianDeriv(std::size_t index) const
{
  return computeRelativeJacobianStaticDerivWrtPos(
      getPositionsStatic(),
      index,
      getAxisOrder(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
void EulerFreeJoint::updateRelativeJacobian(bool) const
{
  mJacobian = computeRelativeJacobianStatic(
      getPositionsStatic(),
      getAxisOrder(),
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
void EulerFreeJoint::updateRelativeJacobianTimeDeriv() const
{
  mJacobianDeriv = computeRelativeJacobianTimeDerivStatic(
      getPositionsStatic(),
      getVelocitiesStatic(),
      getAxisOrder(),
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
      Joint::mAspectProperties.mT_ChildBodyToJoint);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianStatic(
    const Eigen::Vector6s& positions,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  Eigen::Vector3s euler = positions.head<3>();
  Eigen::Isometry3s T = EulerJoint::convertToTransform(euler, axisOrder)
                        * childBodyToJoint.inverse();

  Eigen::Matrix6s spatialJac = Eigen::Matrix6s::Identity();
  spatialJac.block<3, 3>(3, 3) = T.linear().transpose(); // R^T
  spatialJac.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianStatic(
      euler, axisOrder, childBodyToJoint);

  return spatialJac;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianStaticDerivWrtPos(
    const Eigen::Vector6s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  assert(
      axisOrder == EulerJoint::AxisOrder::XYZ
      && "Only XYZ AxisOrder is currently supported in the EulerFreeJoint computeRelativeJacobianStaticDerivWrtPos");

  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();

    Eigen::Matrix6s spatialJac = Eigen::Matrix6s::Identity();
    spatialJac.block<3, 3>(3, 3)
        = childBodyToJoint.linear()
          * math::eulerXYZToMatrixGrad(euler, index).transpose(); // R^T
    spatialJac.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianDeriv(
        index, euler, axisOrder, childBodyToJoint);

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
    Eigen::Isometry3s childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector6s perturbedPlus
      = positions + (EPS * Eigen::Vector6s::Unit(index));
  Eigen::Vector6s perturbedMinus
      = positions - (EPS * Eigen::Vector6s::Unit(index));

  Eigen::Matrix6s plus = computeRelativeJacobianStatic(
      perturbedPlus, axisOrder, childBodyToJoint);
  Eigen::Matrix6s minus = computeRelativeJacobianStatic(
      perturbedMinus, axisOrder, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  Eigen::Vector3s euler = positions.head<3>();
  Eigen::Vector3s eulerVel = velocities.head<3>();

  Eigen::Matrix6s spatialJacDeriv = Eigen::Matrix6s::Zero();
  spatialJacDeriv.block<6, 3>(0, 0)
      = EulerJoint::computeRelativeJacobianTimeDerivStatic(
          euler, eulerVel, axisOrder, childBodyToJoint);

  for (int i = 0; i < 3; i++)
  {
    spatialJacDeriv.block<3, 3>(3, 3)
        += childBodyToJoint.linear()
           * math::eulerXYZToMatrixGrad(euler, i).transpose() * eulerVel(i);
  }
  return spatialJacDeriv;
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivStatic(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  const s_t EPS = 1e-8;
  Eigen::Vector6s perturbedPlus = positions + (EPS * velocities);
  Eigen::Vector6s perturbedMinus = positions - (EPS * velocities);

  Eigen::Matrix6s plus = computeRelativeJacobianStatic(
      perturbedPlus, axisOrder, childBodyToJoint);
  Eigen::Matrix6s minus = computeRelativeJacobianStatic(
      perturbedMinus, axisOrder, childBodyToJoint);
  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();
    Eigen::Vector3s eulerVel = velocities.head<3>();

    Eigen::Matrix6s d_dJ = Eigen::Matrix6s::Zero();
    d_dJ.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianTimeDerivDeriv(
        index, euler, eulerVel, axisOrder, childBodyToJoint);

    for (int i = 0; i < 3; i++)
    {
      d_dJ.block<3, 3>(3, 3)
          += childBodyToJoint.linear()
             * math::eulerXYZToMatrixSecondGrad(euler, i, index).transpose()
             * eulerVel(i);
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
EulerFreeJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPos(
    const Eigen::Vector6s& positions,
    const Eigen::Vector6s& velocities,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-8;
  Eigen::Vector6s perturbedPlus
      = positions + (EPS * Eigen::Vector6s::Unit(index));
  Eigen::Vector6s perturbedMinus
      = positions - (EPS * Eigen::Vector6s::Unit(index));

  Eigen::Matrix6s plus = computeRelativeJacobianTimeDerivStatic(
      perturbedPlus, velocities, axisOrder, childBodyToJoint);
  Eigen::Matrix6s minus = computeRelativeJacobianTimeDerivStatic(
      perturbedMinus, velocities, axisOrder, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix6s EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
    const Eigen::Vector6s& positions,
    std::size_t index,
    EulerJoint::AxisOrder axisOrder,
    Eigen::Isometry3s childBodyToJoint)
{
  if (index < 3)
  {
    Eigen::Vector3s euler = positions.head<3>();

    Eigen::Matrix6s d_dJ = Eigen::Matrix6s::Zero();
    d_dJ.block<6, 3>(0, 0) = EulerJoint::computeRelativeJacobianTimeDerivDeriv2(
        index, euler, axisOrder, childBodyToJoint);

    d_dJ.block<3, 3>(3, 3)
        = childBodyToJoint.linear()
          * math::eulerXYZToMatrixGrad(euler, index).transpose();
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
    Eigen::Isometry3s childBodyToJoint)
{
  // This is wrt position
  const s_t EPS = 1e-8;
  Eigen::Vector6s perturbedPlus
      = velocities + (EPS * Eigen::Vector6s::Unit(index));
  Eigen::Vector6s perturbedMinus
      = velocities - (EPS * Eigen::Vector6s::Unit(index));

  Eigen::Matrix6s plus = computeRelativeJacobianTimeDerivStatic(
      positions, perturbedPlus, axisOrder, childBodyToJoint);
  Eigen::Matrix6s minus = computeRelativeJacobianTimeDerivStatic(
      positions, perturbedMinus, axisOrder, childBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

} // namespace dynamics
} // namespace dart