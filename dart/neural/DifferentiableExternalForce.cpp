#include "dart/neural/DifferentiableExternalForce.hpp"

#include "dart/collision/Contact.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/ContactConstraint.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace neural {

//==============================================================================
DifferentiableExternalForce::DifferentiableExternalForce(
    std::shared_ptr<dynamics::Skeleton> skel, int appliedToBody)
  : mSkel(skel), mBodyIndex(appliedToBody)
{
  /// Compute the `activeDofs` list of DOFs that will be torqued by this
  /// worldWrench
  const Eigen::MatrixXi& parents = skel->getDofParentMap();
  dynamics::BodyNode* body = skel->getBodyNode(appliedToBody);
  std::vector<int> parentDofs;
  for (int i = 0; i < body->getParentJoint()->getNumDofs(); i++)
  {
    parentDofs.push_back(
        body->getParentJoint()->getDof(i)->getIndexInSkeleton());
  }
  for (int i = 0; i < skel->getNumDofs(); i++)
  {
    bool isParent = false;
    for (int j : parentDofs)
    {
      if (parents(i, j) == 1 || i == j)
      {
        isParent = true;
        break;
      }
    }
    if (!isParent)
      continue;

    activeDofs.push_back(i);
  }
}

//==============================================================================
/// This analytically computes the torques that this world wrench applies to
/// this skeleton.
Eigen::VectorXs DifferentiableExternalForce::computeTau(
    Eigen::Vector6s worldWrench)
{
  Eigen::VectorXs taus = Eigen::VectorXs::Zero(mSkel->getNumDofs());
  for (int i : activeDofs)
  {
    taus(i) = DifferentiableContactConstraint::getWorldScrewAxisForForce(
                  mSkel->getDof(i))
                  .dot(worldWrench);
  }
  return taus;
}

//==============================================================================
/// This computes the Jacobian relating changes in `wrt` to changes in the
/// output of `computeTau()`.
Eigen::MatrixXs DifferentiableExternalForce::getJacobianOfTauWrt(
    Eigen::Vector6s worldWrench, neural::WithRespectTo* wrt)
{
  (void)worldWrench;
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(mSkel->getNumDofs(), wrt->dim(mSkel.get()));

  if (wrt == WithRespectTo::POSITION)
  {
    for (int col = 0; col < mSkel->getNumDofs(); col++)
    {
      for (int i : activeDofs)
      {
        result(i, col)
            = DifferentiableContactConstraint::getScrewAxisForForceGradient(
                  mSkel->getDof(i), mSkel->getDof(col))
                  .dot(worldWrench);
      }
    }
  }
  else if (wrt == WithRespectTo::GROUP_SCALES)
  {
    for (int col = 0; col < mSkel->getGroupScaleDim(); col++)
    {
      for (int i : activeDofs)
      {
        dynamics::DegreeOfFreedom* screwDof = mSkel->getDof(i);
        Eigen::Vector6s axisWorldTwist
            = DifferentiableContactConstraint::getWorldScrewAxisForForce(
                screwDof);
        Eigen::Vector6s rotateWorldTwist = Eigen::Vector6s::Zero();
        rotateWorldTwist.tail<3>()
            = mSkel->getGroupScaleMovementOnJointInWorldSpace(
                col, screwDof->getJoint()->getJointIndexInSkeleton());

        Eigen::Vector6s grad = math::ad(rotateWorldTwist, axisWorldTwist);
        result(i, col) = grad.dot(worldWrench);
      }
    }
  }
  else if (
      wrt == WithRespectTo::FORCE || wrt == WithRespectTo::VELOCITY
      || wrt == WithRespectTo::GROUP_MASSES || wrt == WithRespectTo::GROUP_COMS
      || wrt == WithRespectTo::GROUP_INERTIAS)
  {
    // Do nothing, these are zeros
  }
  else
  {
    // Fall back to finite differencing
    return finiteDifferenceJacobianOfTauWrt(worldWrench, wrt);
  }

  return result;
}

//==============================================================================
/// This computes the Jacobian relating changes in `wrt` to changes in the
/// output of `computeTau()`.
Eigen::MatrixXs DifferentiableExternalForce::finiteDifferenceJacobianOfTauWrt(
    Eigen::Vector6s worldWrench, neural::WithRespectTo* wrt)
{
  std::size_t n = mSkel->getNumDofs();
  std::size_t m = wrt->dim(mSkel.get());
  Eigen::MatrixXs result(n, m);
  Eigen::VectorXs originalWrt = wrt->get(mSkel.get());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedWrt = originalWrt;
        tweakedWrt(dof) += eps;
        wrt->set(mSkel.get(), tweakedWrt);
        perturbed = computeTau(worldWrench);
        return true;
      },
      result,
      eps,
      useRidders);

  wrt->set(mSkel.get(), originalWrt);
  return result;
}

//==============================================================================
/// This computes the Jacobian relating changes in world torques to changes in
/// the output of `computeTau()`.
Eigen::MatrixXs DifferentiableExternalForce::getJacobianOfTauWrtWorldWrench(
    Eigen::Vector6s worldWrench)
{
  (void)worldWrench;

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mSkel->getNumDofs(), 6);
  for (int i : activeDofs)
  {
    result.row(i) = DifferentiableContactConstraint::getWorldScrewAxisForForce(
        mSkel->getDof(i));
  }
  return result;
}

//==============================================================================
/// This computes the Jacobian relating changes in world torques to changes in
/// the output of `computeTau()`.
Eigen::MatrixXs
DifferentiableExternalForce::finiteDifferenceJacobianOfTauWrtWorldWrench(
    Eigen::Vector6s worldWrench)
{
  std::size_t n = mSkel->getNumDofs();
  Eigen::MatrixXs result(n, 6);

  bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 5e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::Vector6s tweakedWrench = worldWrench;
        tweakedWrench(dof) += eps;
        perturbed = computeTau(tweakedWrench);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

} // namespace neural
} // namespace dart