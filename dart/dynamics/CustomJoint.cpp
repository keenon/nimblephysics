#include "dart/dynamics/CustomJoint.hpp"

#include <memory>
#include <string>

#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"

#define PointDefine typename math::RealVectorSpace<Dimension>::Point;
#define JacMatrixDefine                                                        \
  typename math::RealVectorSpace<Dimension>::JacobianMatrix

namespace dart {
namespace dynamics {

template <std::size_t Dimension>
CustomJoint<Dimension>::CustomJoint(
    const detail::GenericJointProperties<math::RealVectorSpace<Dimension>>&
        props)
  : GenericJoint<math::RealVectorSpace<Dimension>>(props),
    mAxisOrder(EulerJoint::AxisOrder::XYZ),
    mFlipAxisMap(Eigen::Vector3s::Ones())
{
  mFunctions.reserve(6);
  for (int i = 0; i < 6; i++)
  {
    mFunctions.push_back(std::make_shared<math::ConstantFunction>(0));
    mFunctionDrivenByDof.push_back(0);
  }
}

//==============================================================================
/// This sets a custom function to map our single input degree of freedom to
/// the wrapped Euler joint's degree of freedom and index i
template <std::size_t Dimension>
void CustomJoint<Dimension>::setCustomFunction(
    std::size_t i, std::shared_ptr<math::CustomFunction> fn, int drivenByDof)
{
  assert(fn.get() != nullptr);
  mFunctions[i] = fn;
  mFunctionDrivenByDof[i] = drivenByDof;
  this->notifyPositionUpdated();
}

//==============================================================================
template <std::size_t Dimension>
std::shared_ptr<math::CustomFunction> CustomJoint<Dimension>::getCustomFunction(
    std::size_t i)
{
  assert(mFunctions[i].get() != nullptr);
  return mFunctions[i];
}

//==============================================================================
/// There is an annoying tendency for custom joints to encode the linear
/// offset of the bone in their custom functions. We don't want that, so we
/// want to move any relative transform caused by custom functions into the
/// parent transform.
template <std::size_t Dimension>
void CustomJoint<Dimension>::zeroTranslationInCustomFunctions()
{
  Eigen::Isometry3s parentT = this->getTransformFromParentBodyNode();
#ifndef NDEBUG
  Eigen::Vector3s originalParentTranslation = parentT.translation();
  Eigen::Vector3s originalT = this->getRelativeTransform().translation();
#endif

  Eigen::Vector3s defaultValues = Eigen::Vector3s::Zero();
  for (int i = 3; i < 6; i++)
  {
    defaultValues(i - 3)
        = mFunctions[i]->calcValue(this->getPosition(mFunctionDrivenByDof[i]));
    this->setCustomFunction(
        i,
        mFunctions[i]->offsetBy(-defaultValues(i - 3)),
        mFunctionDrivenByDof[i]);
    assert(
        mFunctions[i]->calcValue(this->getPosition(mFunctionDrivenByDof[i]))
        == 0);
  }
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::Vector3s euler = getEulerPositions(pos);
  Eigen::Isometry3s T
      = EulerJoint::convertToTransform(euler, mAxisOrder, mFlipAxisMap);
  T.translation() = getTranslationPositions(pos);
  Eigen::Matrix3s childRotation
      = T.linear() * Joint::mAspectProperties.mT_ChildBodyToJoint.linear();
  parentT.translation() += childRotation * defaultValues;
  this->setTransformFromParentBodyNode(parentT);

#ifndef NDEBUG
  Eigen::Vector3s finalT = this->getRelativeTransform().translation();
  if ((originalT - finalT).squaredNorm() > 1e-8)
  {
    std::cout << "Positions: " << std::endl
              << this->getPositions() << std::endl;
    std::cout << "Custom Functions: " << std::endl
              << this->getCustomFunctionPositions(this->getPositions())
              << std::endl;
    Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(3, 6);
    diff.col(0) = originalT;
    diff.col(1) = originalParentTranslation;
    diff.col(2) = defaultValues;
    diff.col(3) = parentT.translation();
    diff.col(4) = finalT;
    diff.col(5) = originalT - finalT;
    std::cout << "Original - Original Parent T - Custom T - New Parent T - "
                 "Final - Error"
              << std::endl
              << diff << std::endl;
    assert(originalT == finalT);
  }
#endif
}

//==============================================================================
template <std::size_t Dimension>
int CustomJoint<Dimension>::getCustomFunctionDrivenByDof(std::size_t i)
{
  return mFunctionDrivenByDof[i];
}

//==============================================================================
/// This gets the Jacobian of the mapping functions. That is, for every
/// epsilon change in x, how does each custom function change?
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::getCustomFunctionGradientAt(
    const Eigen::VectorXs& x) const
{
  math::Jacobian df = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = mFunctionDrivenByDof[i];
    df(i, drivenByDof) = mFunctions[i]->calcDerivative(1, x(drivenByDof));
  }
  return df;
}

//==============================================================================
/// This gets the time derivative of the Jacobian of the mapping functions.
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::getCustomFunctionGradientAtTimeDeriv(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  math::Jacobian dfdt = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = mFunctionDrivenByDof[i];
    dfdt(i, drivenByDof)
        = mFunctions[i]->calcDerivative(2, x(drivenByDof)) * dx(drivenByDof);
  }
  return dfdt;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::finiteDifferenceCustomFunctionGradientAtTimeDeriv(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  math::Jacobian result = math::Jacobian::Zero(6, this->getNumDofs());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](/* in*/ s_t eps,
          /*out*/ math::Jacobian& perturbed) {
        Eigen::VectorXs tweaked = x + eps * dx;
        perturbed = getCustomFunctionGradientAt(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return math::Jacobian(result);
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::finiteDifferenceCustomFunctionGradientAt(
    const Eigen::VectorXs& x, bool useRidders) const
{
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(6, this->getNumDofs());

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs perturbedPoint = x;
        perturbedPoint(dof) += eps;
        perturbed = getCustomFunctionPositions(perturbedPoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return math::Jacobian(result);
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getCustomFunctionGradientAtTimeDerivPosDeriv(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx,
    int index) const
{
  math::Jacobian dfdt = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = mFunctionDrivenByDof[i];
    if (drivenByDof == index)
    {
      dfdt(i, drivenByDof)
          = mFunctions[i]->calcDerivative(2, x(drivenByDof)) * ddx(drivenByDof)
            + mFunctions[i]->calcDerivative(3, x(drivenByDof))
                  * dx(drivenByDof);
    }
  }
  return dfdt;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceCustomFunctionGradientAtTimeDerivPosDeriv(
        const Eigen::VectorXs& x,
        const Eigen::VectorXs& dx,
        const Eigen::VectorXs& ddx,
        int index) const
{
  math::Jacobian result = math::Jacobian::Zero(6, this->getNumDofs());
  (void)ddx;

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](/* in*/ s_t eps,
          /*out*/ math::Jacobian& perturbed) {
        Eigen::VectorXs tweaked = x;
        tweaked(index) += eps;
        perturbed = getCustomFunctionGradientAtTimeDeriv(tweaked, dx);
        return true;
      },
      result,
      eps,
      useRidders);

  return math::Jacobian(result);
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getCustomFunctionGradientAtTimeDerivVelDeriv(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx,
    int index) const
{
  (void)dx;
  (void)ddx;
  math::Jacobian dfdt = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = mFunctionDrivenByDof[i];
    if (drivenByDof == index)
    {
      dfdt(i, drivenByDof) = mFunctions[i]->calcDerivative(2, x(drivenByDof));
      /*
      dfdt(i, drivenByDof)
          = mFunctions[i]->calcDerivative(2, x(drivenByDof)) * dx(drivenByDof);
          */
    }
  }
  return dfdt;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceCustomFunctionGradientAtTimeDerivVelDeriv(
        const Eigen::VectorXs& x,
        const Eigen::VectorXs& dx,
        const Eigen::VectorXs& ddx,
        int index) const
{
  math::Jacobian result = math::Jacobian::Zero(6, this->getNumDofs());
  (void)ddx;

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](/* in*/ s_t eps,
          /*out*/ math::Jacobian& perturbed) {
        Eigen::VectorXs tweaked = dx;
        tweaked(index) += eps;
        perturbed = getCustomFunctionGradientAtTimeDeriv(x, tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return math::Jacobian(result);
}

//==============================================================================
/// This gets the array of 2nd order derivatives at x
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::getCustomFunctionSecondGradientAt(
    const Eigen::VectorXs& x) const
{
  math::Jacobian ddf = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = mFunctionDrivenByDof[i];
    ddf(i, drivenByDof) = mFunctions[i]->calcDerivative(2, x(drivenByDof));
  }
  return ddf;
}

//==============================================================================
/// This produces the positions of each of the mapping functions, at a given
/// point of input.
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::getCustomFunctionPositions(
    const Eigen::VectorXs& x) const
{
  Eigen::Vector6s pos = Eigen::Vector6s::Zero();
  for (int i = 0; i < 6; i++)
  {
    pos(i) = mFunctions[i]->calcValue(x(mFunctionDrivenByDof[i]));
  }
  return pos;
}

//==============================================================================
/// This produces the velocities of each of the mapping functions, at a given
/// point with a specific velocity.
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::getCustomFunctionVelocities(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  Eigen::Vector6s vel = Eigen::Vector6s::Zero();
  math::Jacobian fnGrad = getCustomFunctionGradientAt(x);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = this->mFunctionDrivenByDof[i];
    vel(i) = fnGrad(i, drivenByDof) * dx(drivenByDof);
  }
  return vel;
}

//==============================================================================
/// This produces the accelerations of each of the mapping functions, at a
/// given point with a specific acceleration.
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::getCustomFunctionAccelerations(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx) const
{
  math::Jacobian ddf = getCustomFunctionGradientAt(x);
  Eigen::Vector6s acc;

  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = this->mFunctionDrivenByDof[i];
    acc(i)
        = ddf(i, drivenByDof) * ddx(drivenByDof)
          + mFunctions[i]->calcDerivative(2, x(drivenByDof)) * dx(drivenByDof);
  }

  return acc;
}

//==============================================================================
/// This produces the derivative of the velocities with respect to changes
/// in position x
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getCustomFunctionVelocitiesDerivativeWrtPos(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  math::Jacobian fnGrad = getCustomFunctionSecondGradientAt(x);
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);
  for (int i = 0; i < 6; i++)
  {
    int drivenByDof = this->mFunctionDrivenByDof[i];
    jac(i, drivenByDof) = fnGrad(i, drivenByDof) * dx(drivenByDof);
  }
  return jac;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceCustomFunctionVelocitiesDerivativeWrtPos(
        const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);

  const double EPS = 1e-7;

  for (int i = 0; i < Dimension; i++)
  {
    Eigen::VectorXs perturbed = x;
    perturbed(i) += EPS;
    Eigen::Vector6s pos = getCustomFunctionVelocities(perturbed, dx);
    perturbed = x;
    perturbed(i) -= EPS;
    Eigen::Vector6s neg = getCustomFunctionVelocities(perturbed, dx);
    jac.col(i) = (pos - neg) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
/// This produces the derivative of the accelerations with respect to changes
/// in position x
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getCustomFunctionAccelerationsDerivativeWrtPos(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx) const
{
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);

  math::Jacobian ddf_dx = getCustomFunctionSecondGradientAt(x);
  for (int i = 0; i < 6; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];

    jac(i, drivenBy)
        = ddx(drivenBy) * ddf_dx(i, drivenBy)
          // Most custom functions will have a 0 third derivative, but this is
          // here just in case
          + mFunctions[i]->calcDerivative(3, x(drivenBy)) * dx(drivenBy);
  }

  return jac;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceCustomFunctionAccelerationsDerivativeWrtPos(
        const Eigen::VectorXs& x,
        const Eigen::VectorXs& dx,
        const Eigen::VectorXs& ddx) const
{
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);

  const double EPS = 1e-7;
  for (int i = 0; i < Dimension; i++)
  {
    Eigen::VectorXs perturbed = x;
    perturbed(i) += EPS;
    Eigen::Vector6s pos = getCustomFunctionAccelerations(perturbed, dx, ddx);
    perturbed = x;
    perturbed(i) -= EPS;
    Eigen::Vector6s neg = getCustomFunctionAccelerations(perturbed, dx, ddx);
    jac.col(i) = (pos - neg) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
/// This produces the derivative of the accelerations with respect to changes
/// in velocity dx
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getCustomFunctionAccelerationsDerivativeWrtVel(
    const Eigen::VectorXs& x) const
{
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);

  for (int i = 0; i < 6; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    jac(i, drivenBy) = mFunctions[i]->calcDerivative(2, x(drivenBy));
  }
  return jac;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceCustomFunctionAccelerationsDerivativeWrtVel(
        const Eigen::VectorXs& x,
        const Eigen::VectorXs& dx,
        const Eigen::VectorXs& ddx) const
{
  math::Jacobian jac = math::Jacobian::Zero(6, Dimension);

  const double EPS = 1e-7;
  for (int i = 0; i < Dimension; i++)
  {
    Eigen::VectorXs perturbed = dx;
    perturbed(i) += EPS;
    Eigen::Vector6s pos = getCustomFunctionAccelerations(x, perturbed, ddx);
    perturbed = dx;
    perturbed(i) -= EPS;
    Eigen::Vector6s neg = getCustomFunctionAccelerations(x, perturbed, ddx);
    jac.col(i) = (pos - neg) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
/// This returns the first 3 custom function outputs
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getEulerPositions(
    const Eigen::VectorXs& x) const
{
  Eigen::Vector3s pos;
  for (int i = 0; i < 3; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    pos(i) = mFunctions[i]->calcValue(x(drivenBy));
  }
  return pos;
}

//==============================================================================
/// This returns the first 3 custom function outputs's derivatives
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getEulerVelocities(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  Eigen::Vector3s vel;
  for (int i = 0; i < 3; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    vel(i) = mFunctions[i]->calcDerivative(1, x(drivenBy)) * dx(drivenBy);
  }
  return vel;
}

//==============================================================================
/// This returns the first 3 custom function outputs's derivatives
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getEulerAccelerations(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx) const
{
  Eigen::Vector3s acc;
  for (int i = 0; i < 3; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    acc(i) = mFunctions[i]->calcDerivative(1, x(drivenBy)) * ddx(drivenBy)
             + mFunctions[i]->calcDerivative(2, x(drivenBy)) * dx(drivenBy);
  }
  return acc;
}

//==============================================================================
/// This returns the last 3 custom function outputs
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getTranslationPositions(
    const Eigen::VectorXs& x) const
{
  Eigen::Vector3s pos;
  for (int i = 3; i < 6; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    pos(i - 3) = mFunctions[i]->calcValue(x(drivenBy));
  }
  return pos;
}

//==============================================================================
/// This returns the last 3 custom function outputs's derivatives
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getTranslationVelocities(
    const Eigen::VectorXs& x, const Eigen::VectorXs& dx) const
{
  Eigen::Vector3s vel;
  for (int i = 3; i < 6; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    vel(i - 3) = mFunctions[i]->calcDerivative(1, x(drivenBy)) * dx(drivenBy);
  }
  return vel;
}

//==============================================================================
/// This returns the last 3 custom function outputs's second derivatives
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getTranslationAccelerations(
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& dx,
    const Eigen::VectorXs& ddx) const
{
  Eigen::Vector3s acc;
  for (int i = 3; i < 6; i++)
  {
    int drivenBy = this->mFunctionDrivenByDof[i];
    acc(i - 3) = mFunctions[i]->calcDerivative(1, x(drivenBy)) * ddx(drivenBy)
                 + mFunctions[i]->calcDerivative(2, x(drivenBy)) * dx(drivenBy);
  }
  return acc;
}

//==============================================================================
template <std::size_t Dimension>
const std::string& CustomJoint<Dimension>::getType() const
{
  return getStaticType();
}

//==============================================================================
template <std::size_t Dimension>
const std::string& CustomJoint<Dimension>::getStaticType()
{
  static const std::string name = "CustomJoint" + std::to_string(Dimension);
  return name;
}

//==============================================================================
template <std::size_t Dimension>
bool CustomJoint<Dimension>::isCyclic(std::size_t) const
{
  return false;
}

//==============================================================================
template <std::size_t Dimension>
void CustomJoint<Dimension>::setAxisOrder(
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
template <std::size_t Dimension>
EulerJoint::AxisOrder CustomJoint<Dimension>::getAxisOrder() const
{
  return mAxisOrder;
}

//==============================================================================
/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
template <std::size_t Dimension>
void CustomJoint<Dimension>::setFlipAxisMap(Eigen::Vector3s map)
{
  mFlipAxisMap = map;
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Vector3s CustomJoint<Dimension>::getFlipAxisMap() const
{
  return mFlipAxisMap;
}

//==============================================================================
template <std::size_t Dimension>
dart::dynamics::Joint* CustomJoint<Dimension>::clone() const
{
  CustomJoint<Dimension>* joint
      = new CustomJoint<Dimension>(this->getJointProperties());
  joint->mFunctions = mFunctions;
  joint->mFunctionDrivenByDof = mFunctionDrivenByDof;
  joint->copyTransformsFrom(this);
  joint->setFlipAxisMap(getFlipAxisMap());
  joint->setAxisOrder(getAxisOrder());
  joint->setName(this->getName());
  joint->setPositionUpperLimits(this->getPositionUpperLimits());
  joint->setPositionLowerLimits(this->getPositionLowerLimits());
  joint->setVelocityUpperLimits(this->getVelocityUpperLimits());
  joint->setVelocityLowerLimits(this->getVelocityLowerLimits());
  return joint;
}

//==============================================================================
template <std::size_t Dimension>
void CustomJoint<Dimension>::updateDegreeOfFreedomNames()
{
  if (!this->mDofs[0]->isNamePreserved())
    this->mDofs[0]->setName(Joint::mAspectProperties.mName, false);
}

//==============================================================================
template <std::size_t Dimension>
void CustomJoint<Dimension>::updateRelativeTransform() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::Vector3s euler = getEulerPositions(pos);
  Eigen::Isometry3s T
      = EulerJoint::convertToTransform(euler, mAxisOrder, mFlipAxisMap);
  T.translation() = getTranslationPositions(pos);

  this->mT = Joint::mAspectProperties.mT_ParentBodyToJoint * T
             * Joint::mAspectProperties.mT_ChildBodyToJoint.inverse();
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Matrix6s CustomJoint<Dimension>::getSpatialJacobianStaticDerivWrtInput(
    const Eigen::VectorXs& pos, std::size_t index) const
{
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s grad = getCustomFunctionGradientAt(pos).col(index);

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
template <std::size_t Dimension>
Eigen::Matrix6s
CustomJoint<Dimension>::finiteDifferenceSpatialJacobianStaticDerivWrtInput(
    const Eigen::VectorXs& pos, std::size_t index, bool useRidders) const
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::VectorXs perturbedPos = pos;
        perturbedPos(index) += eps;
        perturbed = EulerFreeJoint::computeRelativeJacobianStatic(
            getCustomFunctionPositions(perturbedPos),
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::getRelativeJacobianDeriv(
    std::size_t index) const
{
  Eigen::VectorXs pos = this->getPositions();

  math::Jacobian grad = getCustomFunctionGradientAt(pos);
  math::Jacobian secondGrad = getCustomFunctionSecondGradientAt(pos);

  for (int i = 0; i < Dimension; i++)
  {
    if (i != index)
    {
      // grad.col(i).setZero();
      secondGrad.col(i).setZero();
    }
  }

  return getSpatialJacobianStaticDerivWrtInput(pos, index) * grad
         + EulerFreeJoint::computeRelativeJacobianStatic(
               getCustomFunctionPositions(pos),
               mAxisOrder,
               mFlipAxisMap,
               Joint::mAspectProperties.mT_ChildBodyToJoint)
               * secondGrad;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::finiteDifferenceRelativeJacobianDeriv(
    std::size_t index, bool useRidders)
{
  math::Jacobian result = math::Jacobian::Zero(6, this->getNumDofs());

  s_t original = this->getPosition(index);

  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](/* in*/ s_t eps,
          /*out*/ math::Jacobian& perturbed) {
        s_t tweaked = original + eps;
        this->setPosition(index, tweaked);
        perturbed = this->getRelativeJacobian();
        return true;
      },
      result,
      eps,
      useRidders);

  this->setPosition(index, original);

  return result;
}

//==============================================================================
template <std::size_t Dimension>
typename math::RealVectorSpace<Dimension>::JacobianMatrix
CustomJoint<Dimension>::getRelativeJacobianStatic(
    const typename math::RealVectorSpace<Dimension>::Vector& pos) const
{
  // typename math::RealVectorSpace<Dimension>::JacobianMatrix jacobian;
  return EulerFreeJoint::computeRelativeJacobianStatic(
             getCustomFunctionPositions(pos),
             mAxisOrder,
             mFlipAxisMap,
             Joint::mAspectProperties.mT_ChildBodyToJoint)
         * getCustomFunctionGradientAt(pos);
}

//==============================================================================
template <std::size_t Dimension>
void CustomJoint<Dimension>::updateRelativeJacobian(bool) const
{
  this->mJacobian = getRelativeJacobianStatic(this->getPositionsStatic());
}

//==============================================================================
template <std::size_t Dimension>
void CustomJoint<Dimension>::updateRelativeJacobianTimeDeriv() const
{
  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  math::Jacobian customJac = getCustomFunctionGradientAt(pos);
  math::Jacobian customJacTimeDeriv
      = getCustomFunctionGradientAtTimeDeriv(pos, vel);

  Eigen::Matrix6s eulerJacTimeDeriv
      = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
          positions,
          velocities,
          mAxisOrder,
          mFlipAxisMap,
          Joint::mAspectProperties.mT_ChildBodyToJoint);
  Eigen::Matrix6s eulerJac = EulerFreeJoint::computeRelativeJacobianStatic(
      positions,
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);

  this->mJacobianDeriv
      = eulerJacTimeDeriv * customJac + eulerJac * customJacTimeDeriv;
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Matrix6s
CustomJoint<Dimension>::getSpatialJacobianTimeDerivDerivWrtInputPos(
    const Eigen::VectorXs& pos,
    const Eigen::VectorXs& vel,
    std::size_t index) const
{
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  Eigen::Vector6s posGrad = getCustomFunctionGradientAt(pos).col(index);
  Eigen::Vector6s velGrad
      = getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel).col(index);
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
template <std::size_t Dimension>
Eigen::Matrix6s CustomJoint<Dimension>::
    finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputPos(
        const Eigen::VectorXs& pos,
        const Eigen::VectorXs& vel,
        std::size_t index,
        bool useRidders) const
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-8;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::VectorXs perturbedPos = pos;
        perturbedPos(index) += eps;
        perturbed = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
            getCustomFunctionPositions(perturbedPos),
            getCustomFunctionVelocities(perturbedPos, vel),
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Matrix6s
CustomJoint<Dimension>::getSpatialJacobianTimeDerivDerivWrtInputVel(
    const Eigen::VectorXs& pos, std::size_t index) const
{
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s grad = getCustomFunctionGradientAt(pos).col(index);
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
template <std::size_t Dimension>
Eigen::Matrix6s CustomJoint<Dimension>::
    finiteDifferenceSpatialJacobianTimeDerivDerivWrtInputVel(
        const Eigen::VectorXs& pos,
        const Eigen::VectorXs& vel,
        std::size_t index,
        bool useRidders) const
{
  Eigen::Matrix6s result;

  s_t eps = useRidders ? 1e-3 : 1e-8;
  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& perturbed) {
        Eigen::VectorXs perturbedVel = vel;
        perturbedVel(index) += eps;
        perturbed = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
            getCustomFunctionPositions(pos),
            getCustomFunctionVelocities(pos, perturbedVel),
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getRelativeJacobianTimeDerivDerivWrtPosition(
    std::size_t index) const
{
  /*
  // The original function we're differentiating, for reference:

  Eigen::VectorXs pos = this->getPositionsStatic();
  Eigen::VectorXs vel = this->getVelocitiesStatic();

  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  math::Jacobian customJac = getCustomFunctionGradientAt(pos);
  math::Jacobian customJacTimeDeriv
      = getCustomFunctionGradientAtTimeDeriv(pos, vel);

  Eigen::Matrix6s eulerJacTimeDeriv
      = EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
          positions,
          velocities,
          mAxisOrder,
          mFlipAxisMap,
          Joint::mAspectProperties.mT_ChildBodyToJoint);
  Eigen::Matrix6s eulerJac = EulerFreeJoint::computeRelativeJacobianStatic(
      positions,
      mAxisOrder,
      mFlipAxisMap,
      Joint::mAspectProperties.mT_ChildBodyToJoint);

  this->mJacobianDeriv
      = eulerJacTimeDeriv * customJac + eulerJac * customJacTimeDeriv;
  */

  math::Jacobian J = math::Jacobian::Zero(6, this->getNumDofs());
  Eigen::VectorXs pos = this->getPositions();
  Eigen::VectorXs vel = this->getVelocities();
  Eigen::VectorXs acc = Eigen::VectorXs::Zero(this->getNumDofs());
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);

  math::Jacobian customJacTimeDeriv
      = getCustomFunctionGradientAtTimeDeriv(pos, vel);
  math::Jacobian dc_dt_dp
      = getCustomFunctionGradientAtTimeDerivPosDeriv(pos, vel, acc, index);
  math::Jacobian dc_dp = getCustomFunctionGradientAt(pos);
  math::Jacobian secondJacWrtIndex = getCustomFunctionSecondGradientAt(pos);
  for (int i = 0; i < Dimension; i++)
  {
    if (i != index)
    {
      secondJacWrtIndex.col(i).setZero();
    }
  }

  J = getSpatialJacobianTimeDerivDerivWrtInputPos(pos, vel, index) * dc_dp
      + EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
            positions,
            velocities,
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint)
            * secondJacWrtIndex
      + getSpatialJacobianStaticDerivWrtInput(pos, index) * customJacTimeDeriv
      + EulerFreeJoint::computeRelativeJacobianStatic(
            positions,
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint)
            * dc_dt_dp;
  return J;
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::scratch()
{
  Eigen::VectorXs pos = this->getPositions();
  Eigen::VectorXs vel = this->getVelocities();
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
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::scratchFd()
{
  Eigen::Vector6s result;
  Eigen::VectorXs original = this->getPositions();

  bool useRidders = false;
  s_t eps = 1e-7;
  math::finiteDifference<Eigen::Vector6s>(
      [&](s_t eps, Eigen::Vector6s& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(0) += eps;
        this->setPositions(tweaked);
        perturbed = scratch();
        return true;
      },
      result,
      eps,
      useRidders);

  this->setPositions(original);
  return result;
}

//==============================================================================
template <std::size_t Dimension>
Eigen::Vector6s CustomJoint<Dimension>::scratchAnalytical()
{
  Eigen::VectorXs pos = this->getPositions();
  Eigen::VectorXs vel = this->getVelocities();
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);
  Eigen::Vector6s velocities = getCustomFunctionVelocities(pos, vel);
  std::size_t index = 0;

  return getSpatialJacobianTimeDerivDerivWrtInputPos(pos, vel, index)
             * velocities
         + EulerFreeJoint::computeRelativeJacobianTimeDerivStatic(
               positions,
               velocities,
               mAxisOrder,
               mFlipAxisMap,
               Joint::mAspectProperties.mT_ChildBodyToJoint)
               * getCustomFunctionVelocitiesDerivativeWrtPos(pos, vel).col(
                   index);
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
        std::size_t index, bool useRidders)
{
  (void)useRidders;

  math::Jacobian result = math::Jacobian::Zero(6, this->getNumDofs());

  Eigen::VectorXs original = this->getPositions();
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](s_t eps, math::Jacobian& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(index) += eps;
        this->setPositions(tweaked);
        perturbed = this->getRelativeJacobianTimeDeriv();
        return true;
      },
      result,
      eps,
      useRidders);

  this->setPositions(original);
  return result;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian
CustomJoint<Dimension>::getRelativeJacobianTimeDerivDerivWrtVelocity(
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

  math::Jacobian J = math::Jacobian::Zero(6, this->getNumDofs());
  Eigen::VectorXs pos = this->getPositions();
  Eigen::VectorXs vel = this->getVelocities();
  Eigen::VectorXs acc = Eigen::VectorXs::Zero(this->getNumDofs());
  Eigen::Vector6s positions = getCustomFunctionPositions(pos);

  math::Jacobian dc_dt_dv
      = getCustomFunctionGradientAtTimeDerivVelDeriv(pos, vel, acc, index);
  math::Jacobian dc_dp = getCustomFunctionGradientAt(pos);
  J = getSpatialJacobianTimeDerivDerivWrtInputVel(pos, index) * dc_dp
      + EulerFreeJoint::computeRelativeJacobianStatic(
            positions,
            mAxisOrder,
            mFlipAxisMap,
            Joint::mAspectProperties.mT_ChildBodyToJoint)
            * dc_dt_dv;
  return J;
}

//==============================================================================
template <std::size_t Dimension>
math::Jacobian CustomJoint<Dimension>::
    finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
        std::size_t index, bool useRidders)
{
  (void)useRidders;

  math::Jacobian result = math::Jacobian::Zero(6, Dimension);

  Eigen::VectorXs original = this->getVelocities();
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference<math::Jacobian>(
      [&](s_t eps, math::Jacobian& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(index) += eps;
        this->setVelocities(tweaked);
        perturbed = this->getRelativeJacobianTimeDeriv();
        return true;
      },
      result,
      eps,
      useRidders);
  this->setVelocities(original);

  return result;
}

// Instantiate templates
template class dart::dynamics::CustomJoint<1>;
template class dart::dynamics::CustomJoint<2>;
template class dart::dynamics::CustomJoint<3>;

} // namespace dynamics
} // namespace dart