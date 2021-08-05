#include "dart/dynamics/CustomJoint.hpp"

#include <memory>

#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/LinearFunction.hpp"

namespace dart {
namespace dynamics {

CustomJoint::CustomJoint(const Properties& props)
  : GenericJoint(props),
    mAxisOrder(EulerJoint::AxisOrder::XYZ),
    mFlipAxisMap(Eigen::Vector3s::Ones())
{
  mFunctions.reserve(6);
  for (int i = 0; i < 6; i++)
  {
    mFunctions.push_back(std::make_shared<math::ConstantFunction>(0));
  }
}

//==============================================================================
/// This sets a custom function to map our single input degree of freedom to
/// the wrapped Euler joint's degree of freedom and index i
void CustomJoint::setCustomFunction(
    std::size_t i, std::shared_ptr<math::CustomFunction> fn)
{
  mFunctions[i] = fn;
  notifyPositionUpdated();
}

//==============================================================================
/// This gets the Jacobian of the mapping functions. That is, for every
/// epsilon change in x, how does each custom function change?
Eigen::Vector6s CustomJoint::getCustomFunctionGradientAt(s_t x) const
{
  Eigen::Vector6s df = Eigen::Vector6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    df(i) = mFunctions[i]->calcDerivative(1, x);
  }
  return df;
}

//==============================================================================
/// This gets the array of 2nd order derivatives at x
Eigen::Vector6s CustomJoint::getCustomFunctionSecondGradientAt(s_t x) const
{
  Eigen::Vector6s ddf = Eigen::Vector6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    ddf(i) = mFunctions[i]->calcDerivative(2, x);
  }
  return ddf;
}

//==============================================================================
/// This produces the positions of each of the mapping functions, at a given
/// point of input.
Eigen::Vector6s CustomJoint::getCustomFunctionPositions(s_t x) const
{
  Eigen::Vector6s pos = Eigen::Vector6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    pos(i) = mFunctions[i]->calcValue(x);
  }
  return pos;
}

//==============================================================================
/// This produces the velocities of each of the mapping functions, at a given
/// point with a specific velocity.
Eigen::Vector6s CustomJoint::getCustomFunctionVelocities(s_t x, s_t dx) const
{
  return getCustomFunctionGradientAt(x) * dx;
}

//==============================================================================
/// This produces the accelerations of each of the mapping functions, at a
/// given point with a specific acceleration.
Eigen::Vector6s CustomJoint::getCustomFunctionAccelerations(
    s_t x, s_t dx, s_t ddx) const
{
  Eigen::Vector6s ddf = ddx * getCustomFunctionGradientAt(x);
  for (int i = 0; i < 6; i++)
  {
    ddf(i) += mFunctions[i]->calcDerivative(2, x) * dx;
  }
  return ddf;
}

//==============================================================================
/// This produces the derivative of the velocities with respect to changes
/// in position x
Eigen::Vector6s CustomJoint::getCustomFunctionVelocitiesDerivativeWrtPos(
    s_t x, s_t dx) const
{
  return getCustomFunctionSecondGradientAt(x) * dx;
}

//==============================================================================
Eigen::Vector6s
CustomJoint::finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
    s_t x, s_t dx) const
{
  double EPS = 1e-7;
  Eigen::Vector6s pos = getCustomFunctionVelocities(x + EPS, dx);
  Eigen::Vector6s neg = getCustomFunctionVelocities(x - EPS, dx);
  return (pos - neg) / (2 * EPS);
}

//==============================================================================
/// This produces the derivative of the accelerations with respect to changes
/// in position x
Eigen::Vector6s CustomJoint::getCustomFunctionAccelerationsDerivativeWrtPos(
    s_t x, s_t dx, s_t ddx) const
{
  Eigen::Vector6s ddf_dx = ddx * getCustomFunctionSecondGradientAt(x);
  for (int i = 0; i < 6; i++)
  {
    // Most custom functions will have a 0 third derivative, but this is here
    // just in case
    ddf_dx(i) += mFunctions[i]->calcDerivative(3, x) * dx;
  }
  return ddf_dx;
}

//==============================================================================
Eigen::Vector6s
CustomJoint::finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
    s_t x, s_t dx, s_t ddx) const
{
  double EPS = 1e-7;
  Eigen::Vector6s pos = getCustomFunctionAccelerations(x + EPS, dx, ddx);
  Eigen::Vector6s neg = getCustomFunctionAccelerations(x - EPS, dx, ddx);
  return (pos - neg) / (2 * EPS);
}

//==============================================================================
/// This produces the derivative of the accelerations with respect to changes
/// in velocity dx
Eigen::Vector6s CustomJoint::getCustomFunctionAccelerationsDerivativeWrtVel(
    s_t x) const
{
  Eigen::Vector6s d_dx = Eigen::Vector6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    d_dx(i) += mFunctions[i]->calcDerivative(2, x);
  }
  return d_dx;
}

//==============================================================================
Eigen::Vector6s
CustomJoint::finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
    s_t x, s_t dx, s_t ddx) const
{
  double EPS = 1e-7;
  Eigen::Vector6s pos = getCustomFunctionAccelerations(x, dx + EPS, ddx);
  Eigen::Vector6s neg = getCustomFunctionAccelerations(x, dx - EPS, ddx);
  return (pos - neg) / (2 * EPS);
}

//==============================================================================
/// This returns the first 3 custom function outputs
Eigen::Vector3s CustomJoint::getEulerPositions(s_t x) const
{
  Eigen::Vector3s pos;
  for (int i = 0; i < 3; i++)
  {
    pos(i) = mFunctions[i]->calcValue(x);
  }
  return pos;
}

//==============================================================================
/// This returns the first 3 custom function outputs's derivatives
Eigen::Vector3s CustomJoint::getEulerVelocities(s_t x, s_t dx) const
{
  Eigen::Vector3s vel;
  for (int i = 0; i < 3; i++)
  {
    vel(i) = mFunctions[i]->calcDerivative(1, x) * dx;
  }
  return vel;
}

//==============================================================================
/// This returns the first 3 custom function outputs's derivatives
Eigen::Vector3s CustomJoint::getEulerAccelerations(s_t x, s_t dx, s_t ddx) const
{
  Eigen::Vector3s acc;
  for (int i = 0; i < 3; i++)
  {
    acc(i) = mFunctions[i]->calcDerivative(1, x) * ddx
             + mFunctions[i]->calcDerivative(2, x) * dx;
  }
  return acc;
}

//==============================================================================
/// This returns the last 3 custom function outputs
Eigen::Vector3s CustomJoint::getTranslationPositions(s_t x) const
{
  Eigen::Vector3s pos;
  for (int i = 3; i < 6; i++)
  {
    pos(i - 3) = mFunctions[i]->calcValue(x);
  }
  return pos;
}

//==============================================================================
/// This returns the last 3 custom function outputs's derivatives
Eigen::Vector3s CustomJoint::getTranslationVelocities(s_t x, s_t dx) const
{
  Eigen::Vector3s vel;
  for (int i = 3; i < 6; i++)
  {
    vel(i - 3) = mFunctions[i]->calcDerivative(1, x) * dx;
  }
  return vel;
}

//==============================================================================
/// This returns the last 3 custom function outputs's second derivatives
Eigen::Vector3s CustomJoint::getTranslationAccelerations(
    s_t x, s_t dx, s_t ddx) const
{
  Eigen::Vector3s acc;
  for (int i = 3; i < 6; i++)
  {
    acc(i - 3) = mFunctions[i]->calcDerivative(1, x) * ddx
                 + mFunctions[i]->calcDerivative(2, x) * dx;
  }
  return acc;
}

//==============================================================================
const std::string& CustomJoint::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& CustomJoint::getStaticType()
{
  static const std::string name = "CustomJoint";
  return name;
}

//==============================================================================
bool CustomJoint::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
void CustomJoint::setAxisOrder(EulerJoint::AxisOrder _order, bool _renameDofs)
{
  mAxisOrder = _order;
  if (_renameDofs)
    updateDegreeOfFreedomNames();

  Joint::notifyPositionUpdated();
  updateRelativeJacobian(true);
  Joint::incrementVersion();
}

//==============================================================================
EulerJoint::AxisOrder CustomJoint::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void CustomJoint::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
Eigen::Vector3s CustomJoint::getFlipAxisMap()
{
  return mFlipAxisMap;
}

//==============================================================================
dart::dynamics::Joint* CustomJoint::clone() const
{
  CustomJoint* joint = new CustomJoint(this->getJointProperties());
  joint->mFunctions = mFunctions;
  return joint;
}

//==============================================================================
void CustomJoint::updateDegreeOfFreedomNames()
{
  if (!mDofs[0]->isNamePreserved())
    mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
void CustomJoint::updateRelativeTransform() const
{
  s_t pos = getPositionsStatic()(0);
  Eigen::Vector3s euler = getEulerPositions(pos);
  Eigen::Isometry3s T
      = EulerJoint::convertToTransform(euler, mAxisOrder, mFlipAxisMap);
  T.translation() = getTranslationPositions(pos);

  mT = Joint::mAspectProperties.mT_ParentBodyToJoint * T
       * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
Eigen::Matrix6s CustomJoint::getSpatialJacobianStaticDerivWrtInput(
    s_t pos) const
{
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s grad = getCustomFunctionGradientAt(pos);

  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    J += EulerFreeJoint::computeRelativeJacobianStaticDerivWrtPos(
             positions,
             i,
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * grad(i);
  }
  return J;
}

//==============================================================================
Eigen::Matrix6s CustomJoint::finiteDifferenceSpatialJacobianStaticDerivWrtInput(
    s_t pos) const
{
  const s_t EPS = 1e-7;

  Eigen::Matrix6s plus = EulerFreeJoint::computeRelativeJacobianStatic(
      getCustomFunctionPositions(pos + EPS),
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);
  Eigen::Matrix6s minus = EulerFreeJoint::computeRelativeJacobianStatic(
      getCustomFunctionPositions(pos - EPS),
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
math::Jacobian CustomJoint::getRelativeJacobianDeriv(std::size_t index) const
{
  (void)index;
  assert(index == 0);
  s_t pos = getPositionsStatic()(0);

  Eigen::Vector6s grad = getCustomFunctionGradientAt(pos);
  Eigen::Vector6s secondGrad = getCustomFunctionSecondGradientAt(pos);

  return getSpatialJacobianStaticDerivWrtInput(pos) * grad
         + EulerFreeJoint::computeRelativeJacobianStatic(
               getCustomFunctionPositions(pos),
               mAxisOrder,
              mFlipAxisMap,
               Joint::mAspectProperties.mT_ChildBodyToJoint)
               * secondGrad;
}

//==============================================================================
math::Jacobian CustomJoint::finiteDifferenceRelativeJacobianDeriv(
    std::size_t index)
{
  (void)index;
  assert(index == 0);

  Eigen::Vector1s original = getPositionsStatic();

  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector1s perturbedPlus = original + (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedPlus);
  Eigen::Vector6s plus = getRelativeJacobian();

  Eigen::Vector1s perturbedMinus = original - (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedMinus);
  Eigen::Vector6s minus = getRelativeJacobian();

  setPositionsStatic(original);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Vector6s CustomJoint::getRelativeJacobianStatic(
    const Eigen::Vector1s& position) const
{
  s_t pos = position(0);
  return EulerFreeJoint::computeRelativeJacobianStatic(
             getCustomFunctionPositions(pos),
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * getCustomFunctionGradientAt(pos);
}

//==============================================================================
void CustomJoint::updateRelativeJacobian(bool) const
{
  mJacobian = getRelativeJacobianStatic(getPositionsStatic());
}

//==============================================================================
void CustomJoint::updateRelativeJacobianTimeDeriv() const
{
  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);

  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  mJacobianDeriv = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
                       positions,
                       velocities,
                       mAxisOrder,
                       mFlipAxisMap,
                       Joint::mAspectProperties.mT_ChildBodyToJoint)
                       * velocities
                   + EulerFreeJoint::computeRelativeJacobianStatic(
                         positions,
                         mAxisOrder,
                         mFlipAxisMap,
                         Joint::mAspectProperties.mT_ChildBodyToJoint)
                         * getCustomFunctionAccelerations(pos, vel, 0.0);
}

//==============================================================================
Eigen::Matrix6s CustomJoint::getSpatialJacobianTimeDerivDerivWrtInputPos(
    s_t pos, s_t vel) const
{
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  Eigen::Vector6s posGrad = getCustomFunctionGradientAt(pos);
  Eigen::Vector6s velGrad
      = getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel);
  for (int i = 0; i < 6; i++)
  {
    J += EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtPos(
             positions,
             velocities,
             i,
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * posGrad(i);
    J += EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
             positions,
             i,
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * velGrad(i);
  }
  return J;
}

//==============================================================================
Eigen::Matrix6s
CustomJoint::finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
    s_t pos, s_t vel) const
{
  // This is wrt position
  const s_t EPS = 1e-8;

  Eigen::Matrix6s plus = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
      getCustomFunctionPositions(pos + EPS),
      getCustomFunctionVelocities(pos + EPS, vel),
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);
  Eigen::Matrix6s minus
      = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
          getCustomFunctionPositions(pos - EPS),
          getCustomFunctionVelocities(pos - EPS, vel),
          mAxisOrder,
          mFlipAxisMap,
          Joint::mAspectProperties.mT_ChildBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Matrix6s CustomJoint::getSpatialJacobianTimeDerivDerivWrtInputVel(
    s_t pos) const
{
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s grad = getCustomFunctionGradientAt(pos);
  for (int i = 0; i < 6; i++)
  {
    J += EulerFreeJoint::computeRelativeJacobianTimeDerivDerivWrtVel(
             positions,
             i,
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * grad(i);
  }
  return J;
}

//==============================================================================
Eigen::Matrix6s
CustomJoint::finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
    s_t pos, s_t vel) const
{
  // This is wrt position
  const s_t EPS = 1e-8;

  Eigen::Matrix6s plus = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
      getCustomFunctionPositions(pos),
      getCustomFunctionVelocities(pos, vel + EPS),
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);
  Eigen::Matrix6s minus
      = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
          getCustomFunctionPositions(pos),
          getCustomFunctionVelocities(pos, vel - EPS),
          mAxisOrder,
          mFlipAxisMap,
          Joint::mAspectProperties.mT_ChildBodyToJoint);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
math::Jacobian CustomJoint::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  /*
  // The original function we're differentiating, for reference:

  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  mJacobianDeriv = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
                       positions,
                       velocities,
                       mAxisOrder,
                       Joint::mAspectProperties.mT_ChildBodyToJoint)
                       * velocities
                   + EulerFreeJoint::computeRelativeJacobianStatic(
                         positions,
                         mAxisOrder,
                         Joint::mAspectProperties.mT_ChildBodyToJoint)
                         * getCustomFunctionAccelerations(pos, vel, 0.0);
  */

  (void)index;
  assert(index == 0);

  Eigen::Vector6s J = Eigen::Vector6s::Zero();
  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  J = getSpatialJacobianTimeDerivDerivWrtInputPos(pos, vel) * velocities
      + EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
            positions,
            velocities,
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint)
            * getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel)
      + getSpatialJacobianStaticDerivWrtInput(pos)
            * getCustomFunctionAccelerations(pos, vel, 0.0)
      + EulerFreeJoint::computeRelativeJacobianStatic(
            positions, mAxisOrder, mFlipAxisMap, Joint::mAspectProperties.mT_ChildBodyToJoint)
            * getCustomFunctionAccelerationsDerivativeWrtPos(pos, vel, 0.0);
  return J;
}

//==============================================================================
Eigen::Vector6s CustomJoint::scratch()
{
  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  return EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
             positions,
             velocities,
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * velocities;
}

//==============================================================================
Eigen::Vector6s CustomJoint::scratchFd()
{
  Eigen::Vector1s original = getPositionsStatic();

  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector1s perturbedPlus = original + (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedPlus);
  Eigen::Vector6s plus = scratch();

  Eigen::Vector1s perturbedMinus = original - (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedMinus);
  Eigen::Vector6s minus = scratch();

  setPositionsStatic(original);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
Eigen::Vector6s CustomJoint::scratchAnalytical()
{
  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  return getSpatialJacobianTimeDerivDerivWrtInputPos(pos, vel) * velocities
         + EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
               positions,
               velocities,
               mAxisOrder,
               mFlipAxisMap,
               Joint::mAspectProperties.mT_ChildBodyToJoint)
               * getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel);
}

//==============================================================================
math::Jacobian
CustomJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index)
{
  (void)index;
  assert(index == 0);

  Eigen::Vector1s original = getPositionsStatic();

  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector1s perturbedPlus = original + (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedPlus);
  Eigen::Vector6s plus = getRelativeJacobianTimeDeriv();

  Eigen::Vector1s perturbedMinus = original - (EPS * Eigen::Vector1s::Ones());
  setPositionsStatic(perturbedMinus);
  Eigen::Vector6s minus = getRelativeJacobianTimeDeriv();

  setPositionsStatic(original);

  return (plus - minus) / (2 * EPS);
}

//==============================================================================
math::Jacobian CustomJoint::getRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index) const
{
  /*
  // The original function we're differentiating, for reference:

  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  mJacobianDeriv = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
                       positions,
                       velocities,
                       mAxisOrder,
                       Joint::mAspectProperties.mT_ChildBodyToJoint)
                       * velocities
                   + EulerFreeJoint::computeRelativeJacobianStatic(
                         positions,
                         mAxisOrder,
                         Joint::mAspectProperties.mT_ChildBodyToJoint)
                         * getCustomFunctionAccelerations(pos, vel, 0.0);
  */

  (void)index;
  assert(index == 0);

  Eigen::Vector6s J = Eigen::Vector6s::Zero();
  s_t pos = getPositionsStatic()(0);
  s_t vel = getVelocitiesStatic()(0);
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  J = getSpatialJacobianTimeDerivDerivWrtInputVel(pos) * velocities
      + EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
            positions,
            velocities,
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint)
            * getCustomFunctionGradientAt(pos)
      + EulerFreeJoint::computeRelativeJacobianStatic(
            positions, mAxisOrder, mFlipAxisMap, Joint::mAspectProperties.mT_ChildBodyToJoint)
            * getCustomFunctionAccelerationsDerivativeWrtVel(pos);
  return J;
}

//==============================================================================
math::Jacobian
CustomJoint::finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
    std::size_t index)
{
  (void)index;
  assert(index == 0);

  Eigen::Vector1s original = getVelocitiesStatic();

  // This is wrt position
  const s_t EPS = 1e-7;
  Eigen::Vector1s perturbedPlus = original + (EPS * Eigen::Vector1s::Ones());
  setVelocitiesStatic(perturbedPlus);
  Eigen::Vector6s plus = getRelativeJacobianTimeDeriv();

  Eigen::Vector1s perturbedMinus = original - (EPS * Eigen::Vector1s::Ones());
  setVelocitiesStatic(perturbedMinus);
  Eigen::Vector6s minus = getRelativeJacobianTimeDeriv();

  setVelocitiesStatic(original);

  return (plus - minus) / (2 * EPS);
}

} // namespace dynamics
} // namespace dart