#include "dart/exo/ExoSolverPinnedContact.hpp"

#include <iostream>
#include <tuple>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace exo {

//==============================================================================
ExoSolverPinnedContact::ExoSolverPinnedContact(
    std::shared_ptr<dynamics::Skeleton> realSkel,
    std::shared_ptr<dynamics::Skeleton> virtualSkel)
  : mRealSkel(realSkel), mVirtualSkel(virtualSkel)
{
}

//==============================================================================
void ExoSolverPinnedContact::addMotorDof(int dofIndex)
{
  mMotorDofs.push_back(dofIndex);
}

//==============================================================================
void ExoSolverPinnedContact::setPositions(Eigen::VectorXs q)
{
  mRealSkel->setPositions(q);
  mVirtualSkel->setPositions(q);
}

//==============================================================================
Eigen::MatrixXs ExoSolverPinnedContact::getExoToJointTorquesJacobian()
{
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(mRealSkel->getNumDofs(), mMotorDofs.size());
  for (int i = 0; i < mMotorDofs.size(); i++)
  {
    J(mMotorDofs[i], i) = 1;
  }
  return J;
}

//==============================================================================
/// Set the contact points that we will use when solving inverse dynamics.
void ExoSolverPinnedContact::setContactPins(
    std::vector<std::pair<int, Eigen::Vector3s>> pins)
{
  mPins = pins;
}

//==============================================================================
/// Get the Jacobian relating world space velocity of the contact points to
/// joint velocities.
Eigen::MatrixXs ExoSolverPinnedContact::getContactJacobian()
{
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(mPins.size() * 3, mRealSkel->getNumDofs());
  for (int i = 0; i < mPins.size(); i++)
  {
    J.block(3 * i, 0, 3, J.cols()) = mRealSkel->getLinearJacobian(
        mRealSkel->getBodyNode(mPins[i].first),
        mPins[i].second,
        dynamics::Frame::World());
  }
  return J;
}

//==============================================================================
/// This is only used for testing: Get the Jacobian relating world space
/// velocity of the contact points to joint velocities, by finite
/// differencing.
Eigen::MatrixXs ExoSolverPinnedContact::finiteDifferenceContactJacobian()
{
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(mPins.size() * 3, mRealSkel->getNumDofs());
  Eigen::VectorXs originalVel = mRealSkel->getVelocities();

  s_t eps = 1e-3;
  bool useRidders = true;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ Eigen::VectorXs& out) {
        Eigen::VectorXs tweaked = originalVel;
        tweaked(i) += eps;
        mRealSkel->setVelocities(tweaked);
        out = Eigen::VectorXs::Zero(mPins.size() * 3);
        for (int j = 0; j < mPins.size(); j++)
        {
          out.segment<3>(3 * j)
              = mRealSkel->getBodyNode(mPins[j].first)
                    ->getLinearVelocity(
                        mPins[j].second, dynamics::Frame::World());
        }
        return true;
      },
      J,
      eps,
      useRidders);

  mRealSkel->setVelocities(originalVel);

  return J;
}

//==============================================================================
/// This is only used for testing, to allow us to compare the analytical
/// solution to the numerical solution.
Eigen::VectorXs ExoSolverPinnedContact::analyticalForwardDynamics(
    Eigen::VectorXs dq,
    Eigen::VectorXs tau,
    Eigen::VectorXs exoTorques,
    Eigen::VectorXs contactForces)
{
  mRealSkel->setVelocities(dq);
  Eigen::MatrixXs Minv = mRealSkel->getInvMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::VectorXs exoJointTorques = getExoToJointTorquesJacobian() * exoTorques;
  Eigen::VectorXs contactJointTorques
      = getContactJacobian().transpose() * contactForces;
  Eigen::VectorXs ddq
      = Minv * (tau - C + exoJointTorques + contactJointTorques);
  return ddq;
}

//==============================================================================
/// This is only used for testing, to allow us to compare the analytical
/// solution to the numerical solution.
Eigen::VectorXs ExoSolverPinnedContact::implicitForwardDynamics(
    Eigen::VectorXs dq,
    Eigen::VectorXs tau,
    Eigen::VectorXs exoTorques,
    Eigen::VectorXs contactForces)
{
  mRealSkel->setVelocities(dq);
  Eigen::VectorXs totalTorques = tau;
  for (int i = 0; i < mMotorDofs.size(); i++)
  {
    totalTorques(mMotorDofs[i]) += exoTorques(i);
  }
  mRealSkel->setControlForces(totalTorques);
  for (int i = 0; i < mPins.size(); i++)
  {
    mRealSkel->getBodyNode(mPins[i].first)
        ->setExtForce(
            contactForces.segment<3>(3 * i), mPins[i].second, false, true);
  }
  mRealSkel->computeForwardDynamics();
  Eigen::VectorXs ddq = mRealSkel->getAccelerations();
  for (int i = 0; i < mPins.size(); i++)
  {
    mRealSkel->getBodyNode(mPins[i].first)
        ->setExtForce(Eigen::Vector3s::Zero());
  }
  return ddq;
}

//==============================================================================
/// This is part of the main exoskeleton solver. It takes in the current
/// joint velocities and accelerations, and the last exoskeleton torques, and
/// returns the estimated human pilot joint torques.
Eigen::VectorXs ExoSolverPinnedContact::estimateHumanTorques(
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs contactForces,
    Eigen::VectorXs lastExoTorques)
{
  mRealSkel->setVelocities(dq);

  Eigen::VectorXs lastExoJointTorques
      = getExoToJointTorquesJacobian() * lastExoTorques;
  Eigen::MatrixXs M = mRealSkel->getMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::VectorXs contactJointTorques
      = getContactJacobian().transpose() * contactForces;
  Eigen::VectorXs tau = M * ddq + C - lastExoJointTorques - contactJointTorques;

  return tau;
}

//==============================================================================
/// This is part of the main exoskeleton solver. It takes in the current
/// joint velocities and accelerations, and returns the estimated total
/// joint torques for the human + exoskeleton system.
Eigen::VectorXs ExoSolverPinnedContact::estimateTotalTorques(
    Eigen::VectorXs dq, Eigen::VectorXs ddq, Eigen::VectorXs contactForces)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs M = mRealSkel->getMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::VectorXs contactJointTorques
      = getContactJacobian().transpose() * contactForces;
  Eigen::VectorXs tau = M * ddq + C - contactJointTorques;

  return tau;
}

//==============================================================================
/// This is part of the main exoskeleton solver. It takes in the current
/// estimated human pilot joint torques, and computes the accelerations we
/// would see on the virtual skeleton if we applied those same torques, with
/// the contacts pinned at the CoPs.
std::pair<Eigen::VectorXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedVirtualDynamics(
    Eigen::VectorXs dq, Eigen::VectorXs tau)
{
  mVirtualSkel->setVelocities(dq);

  Eigen::MatrixXs Minv = mVirtualSkel->getInvMassMatrix();
  Eigen::VectorXs C = mVirtualSkel->getCoriolisAndGravityForces()
                      - mVirtualSkel->getExternalForces();
  Eigen::MatrixXs J = getContactJacobian();
  Eigen::VectorXs ddqOffset = Minv * (tau - C);

  const int numDofs = mVirtualSkel->getNumDofs();

  Eigen::VectorXs b = Eigen::VectorXs::Zero(numDofs + J.rows());
  b.segment(0, numDofs) = ddqOffset;
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(b.size(), b.size());
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(0, numDofs, numDofs, J.rows()) = -Minv * J.transpose();
  A.block(numDofs, 0, J.rows(), numDofs) = J;

  Eigen::VectorXs result = A.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXs ddq = result.segment(0, numDofs);
  Eigen::VectorXs forces = result.segment(numDofs, J.rows());

  return std::make_pair(ddq, forces);
}

//==============================================================================
/// This does the same thing as getPinndVirtualDynamics, but returns the Ax +
/// b values A and b such that Ax + b = ddq, accounting for the pin
/// constraints.
std::pair<Eigen::MatrixXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedVirtualDynamicsLinearMap(Eigen::VectorXs dq)
{
  mVirtualSkel->setVelocities(dq);

  Eigen::MatrixXs Minv = mVirtualSkel->getInvMassMatrix();
  Eigen::VectorXs C = mVirtualSkel->getCoriolisAndGravityForces()
                      - mVirtualSkel->getExternalForces();
  Eigen::MatrixXs J = getContactJacobian();
  const int numDofs = mVirtualSkel->getNumDofs();

  int bDim = numDofs + J.rows();
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(bDim, bDim);
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(0, numDofs, numDofs, J.rows()) = -Minv * J.transpose();
  A.block(numDofs, 0, J.rows(), numDofs) = J;

  Eigen::MatrixXs A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXs pinnedMassMatrix = A_inv.block(0, 0, numDofs, numDofs);

  Eigen::VectorXs b = -pinnedMassMatrix * Minv * C;
  Eigen::MatrixXs A_out = pinnedMassMatrix * Minv;
  return std::make_pair(A_out, b);
}

//==============================================================================
/// This is not part of the main exoskeleton solver, but is useful for the
/// inverse problem of analyzing the human pilot's joint torques under
/// different assistance strategies.
std::pair<Eigen::VectorXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedRealDynamics(
    Eigen::VectorXs dq, Eigen::VectorXs tau)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs Minv = mRealSkel->getInvMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::MatrixXs J = getContactJacobian();
  Eigen::VectorXs ddqOffset = Minv * (tau - C);

  const int numDofs = mRealSkel->getNumDofs();

  Eigen::VectorXs b = Eigen::VectorXs::Zero(numDofs + J.rows());
  b.segment(0, numDofs) = ddqOffset;
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(b.size(), b.size());
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(0, numDofs, numDofs, J.rows()) = -Minv * J.transpose();
  A.block(numDofs, 0, J.rows(), numDofs) = J;

  Eigen::VectorXs result = A.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXs ddq = result.segment(0, numDofs);
  Eigen::VectorXs forces = result.segment(numDofs, J.rows());

  return std::make_pair(ddq, forces);
}

//==============================================================================
/// This does the same thing as getPinndRealDynamics, but returns the Ax +
/// b values A and b such that Ax + b = ddq, accounting for the pin
/// constraints.
std::pair<Eigen::MatrixXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedRealDynamicsLinearMap(Eigen::VectorXs dq)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs Minv = mRealSkel->getInvMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::MatrixXs J = getContactJacobian();
  const int numDofs = mRealSkel->getNumDofs();

  int bDim = numDofs + J.rows();
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(bDim, bDim);
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(0, numDofs, numDofs, J.rows()) = -Minv * J.transpose();
  A.block(numDofs, 0, J.rows(), numDofs) = J;

  Eigen::MatrixXs A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXs pinnedMassMatrix = A_inv.block(0, 0, numDofs, numDofs);

  Eigen::VectorXs b = -pinnedMassMatrix * Minv * C;
  Eigen::MatrixXs A_out = pinnedMassMatrix * Minv;
  return std::make_pair(A_out, b);
}

//==============================================================================
/// This is part of the main exoskeleton solver. It takes in how the digital
/// twin of the exo pilot is accelerating, and attempts to solve for the
/// torques that the exo needs to apply to get as close to that as possible.
/// It resolves ambiguities by minimizing the exo torques.
std::pair<Eigen::VectorXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedTotalTorques(
    Eigen::VectorXs dq,
    Eigen::VectorXs ddqDesired,
    Eigen::VectorXs centeringTau,
    Eigen::VectorXs centeringForces)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs M = mRealSkel->getMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::MatrixXs contactJointTorques = getContactJacobian();

  const int numDofs = mRealSkel->getNumDofs();

  Eigen::VectorXs b = Eigen::VectorXs::Zero(numDofs + 6);
  b.segment(0, numDofs) = M * ddqDesired + C;

  Eigen::MatrixXs A
      = Eigen::MatrixXs::Zero(b.size(), numDofs + contactJointTorques.rows());
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(numDofs, 0, 6, 6) = Eigen::MatrixXs::Identity(6, 6);
  A.block(0, numDofs, contactJointTorques.cols(), contactJointTorques.rows())
      = contactJointTorques.transpose();

  Eigen::VectorXs centering
      = Eigen::VectorXs::Zero(numDofs + contactJointTorques.rows());
  centering.segment(0, numDofs) = centeringTau;
  centering.segment(numDofs, contactJointTorques.rows()) = centeringForces;

  Eigen::VectorXs solution
      = A.completeOrthogonalDecomposition().solve(b - A * centering)
        + centering;

  Eigen::VectorXs tauTotal = solution.segment(0, numDofs);
  Eigen::VectorXs f = solution.segment(numDofs, contactJointTorques.rows());

  return std::make_pair(tauTotal, f);
}

//==============================================================================
/// This does the same thing as getPinnedTotalTorques, but returns the Ax +
/// b values A and b such that Ax + b = tau, accounting for the pin
/// constraints.
std::pair<Eigen::MatrixXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedTotalTorquesLinearMap(Eigen::VectorXs dq)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs M = mRealSkel->getMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::MatrixXs contactJointTorques = getContactJacobian();

  const int numDofs = mRealSkel->getNumDofs();

  int bDim = numDofs + 6;
  Eigen::MatrixXs A
      = Eigen::MatrixXs::Zero(bDim, numDofs + contactJointTorques.rows());
  A.block(0, 0, numDofs, numDofs) = Eigen::MatrixXs::Identity(numDofs, numDofs);
  A.block(numDofs, 0, 6, 6) = Eigen::MatrixXs::Identity(6, 6);
  A.block(0, numDofs, contactJointTorques.cols(), contactJointTorques.rows())
      = contactJointTorques.transpose();

  Eigen::MatrixXs A_pinv = A.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXs pinnedInvMassMatrix = A_pinv.block(0, 0, numDofs, numDofs);

  Eigen::VectorXs b = pinnedInvMassMatrix * C;
  Eigen::MatrixXs A_out = pinnedInvMassMatrix * M;

  return std::make_pair(A_out, b);
}

//==============================================================================
/// This is part of the main exoskeleton solver. It takes in the desired
/// torques for the exoskeleton, and returns the torques on the actuated
/// DOFs that can be used to drive the exoskeleton.
Eigen::VectorXs ExoSolverPinnedContact::projectTorquesToExoControlSpace(
    Eigen::VectorXs torques)
{
  Eigen::MatrixXs J = getExoToJointTorquesJacobian();
  return J.completeOrthogonalDecomposition().solve(torques);
}

//==============================================================================
/// This does the same thing as projectTorquesToExoControlSpace, but returns
/// the matrix to multiply by the torques to get the exo torques.
Eigen::MatrixXs
ExoSolverPinnedContact::projectTorquesToExoControlSpaceLinearMap()
{
  Eigen::MatrixXs J = getExoToJointTorquesJacobian();
  return J.completeOrthogonalDecomposition().pseudoInverse();
}

//==============================================================================
/// Often our estimates for `dq` and `ddq` violate the pin constraints. That
/// leads to exo torques that do not tend to zero as the virtual human exactly
/// matches the real human+exo system. To solve this problem, we can solve a
/// set of least-squares equations to find the best set of ddq values to
/// satisfy the constraint.
Eigen::VectorXs ExoSolverPinnedContact::
    getClosestRealAccelerationConsistentWithPinsAndContactForces(
        Eigen::VectorXs dq, Eigen::VectorXs ddq, Eigen::VectorXs contactForces)
{
  mRealSkel->setVelocities(dq);

  Eigen::MatrixXs J = getContactJacobian();
  Eigen::MatrixXs M = mRealSkel->getMassMatrix();
  Eigen::VectorXs C = mRealSkel->getCoriolisAndGravityForces()
                      - mRealSkel->getExternalForces();
  Eigen::VectorXs contactTau = J.transpose() * contactForces;

  const int numDofs = dq.size();

  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(6 + J.rows(), numDofs);
  A.block(0, 0, 6, numDofs) = M.block(0, 0, 6, numDofs);
  A.block(6, 0, J.rows(), numDofs) = J;

  Eigen::VectorXs b = Eigen::VectorXs::Zero(6 + J.rows());
  b.segment<6>(0) = contactTau.segment<6>(0) - C.segment<6>(0);

  // Solve, centered on ddq
  Eigen::VectorXs solution
      = A.completeOrthogonalDecomposition().solve(b - A * ddq) + ddq;

  return solution;
}

//==============================================================================
/// This runs the entire exoskeleton solver pipeline, spitting out the
/// torques to apply to the exoskeleton actuators.
Eigen::VectorXs ExoSolverPinnedContact::solveFromAccelerations(
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs lastExoTorques,
    Eigen::VectorXs contactForces)
{
  Eigen::VectorXs systemTau = estimateTotalTorques(dq, ddq, contactForces);
  Eigen::VectorXs humanTau
      = estimateHumanTorques(dq, ddq, contactForces, lastExoTorques);
  return solveFromBiologicalTorques(dq, humanTau, systemTau, contactForces);
}

//==============================================================================
/// This is a subset of the steps in solveFromAccelerations, which can take
/// the biological joint torques directly, and solve for the exo torques.
Eigen::VectorXs ExoSolverPinnedContact::solveFromBiologicalTorques(
    Eigen::VectorXs dq,
    Eigen::VectorXs humanTau,
    Eigen::VectorXs centeringTau,
    Eigen::VectorXs centeringForces)
{
  Eigen::VectorXs virtualDdq = getPinnedVirtualDynamics(dq, humanTau).first;
  Eigen::VectorXs totalRealTorques
      = getPinnedTotalTorques(dq, virtualDdq, centeringTau, centeringForces)
            .first;
  Eigen::VectorXs netTorques = totalRealTorques - humanTau;
  Eigen::VectorXs exoTorques = projectTorquesToExoControlSpace(netTorques);
  return exoTorques;
}

//==============================================================================
/// This is the same as solveFromBiologicalTorques, but returns the Ax + b
/// values A and b such that Ax + b = exo_tau, accounting for the pin
/// constraints.
std::pair<Eigen::MatrixXs, Eigen::VectorXs>
ExoSolverPinnedContact::getExoTorquesLinearMap(Eigen::VectorXs dq)
{
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> pinnedVirtualDynamics
      = getPinnedVirtualDynamicsLinearMap(dq);
  Eigen::MatrixXs A_1 = pinnedVirtualDynamics.first;
  Eigen::VectorXs b_1 = pinnedVirtualDynamics.second;
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> pinnedTotalTorques
      = getPinnedTotalTorquesLinearMap(dq);
  Eigen::MatrixXs A_2 = pinnedTotalTorques.first;
  Eigen::VectorXs b_2 = pinnedTotalTorques.second;
  Eigen::MatrixXs J_pinv = projectTorquesToExoControlSpaceLinearMap();

  Eigen::MatrixXs A_composite = A_2 * A_1;
  Eigen::VectorXs b_composite = A_2 * b_1 + b_2;
  const int numDofs = mRealSkel->getNumDofs();
  Eigen::MatrixXs A_net
      = A_composite - Eigen::MatrixXs::Identity(numDofs, numDofs);

  Eigen::MatrixXs A_out = J_pinv * A_net;
  Eigen::VectorXs b_out = J_pinv * b_composite;
  return std::make_pair(A_out, b_out);
}

//==============================================================================
/// This does a simple forward dynamics step, given the current human joint
/// torques, factoring in how the exoskeleton will respond to those torques.
std::tuple<Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedForwardDynamicsForExoAndHuman(
    Eigen::VectorXs dq, Eigen::VectorXs humanTau)
{
  Eigen::VectorXs exoTau = solveFromBiologicalTorques(
      dq, humanTau, humanTau, Eigen::VectorXs::Zero(3 * mPins.size()));
  Eigen::VectorXs exoTauOnJoints = getExoToJointTorquesJacobian() * exoTau;
  Eigen::VectorXs totalTau = humanTau + exoTauOnJoints;
  std::pair<Eigen::VectorXs, Eigen::VectorXs> ddqAndForces
      = getPinnedRealDynamics(dq, totalTau);
  Eigen::VectorXs pinnedDdq = ddqAndForces.first;
  Eigen::VectorXs pinnedForces = ddqAndForces.second;
  return std::make_tuple(pinnedDdq, pinnedForces, exoTau);
}

//==============================================================================
/// This does the same thing as getPinnedForwardDynamicsForExoAndHuman, but
/// returns the Ax + b values A and b such that Ax + b = ddq, accounting for
/// the pin constraints.
std::pair<Eigen::MatrixXs, Eigen::VectorXs>
ExoSolverPinnedContact::getPinnedForwardDynamicsForExoAndHumanLinearMap(
    Eigen::VectorXs dq)
{
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> exoTorquesLinearMap
      = getExoTorquesLinearMap(dq);
  const int numDofs = mRealSkel->getNumDofs();
  Eigen::MatrixXs A_1 = exoTorquesLinearMap.first;
  Eigen::VectorXs b_1 = exoTorquesLinearMap.second;
  Eigen::MatrixXs J = getExoToJointTorquesJacobian();
  // A_2, b_2 will now return the total torques on the real skeleton after the
  // exo has compensated, given as input the human biological torques.
  Eigen::MatrixXs A_2 = Eigen::MatrixXs::Identity(numDofs, numDofs) + J * A_1;
  Eigen::VectorXs b_2 = J * b_1;

  std::pair<Eigen::MatrixXs, Eigen::VectorXs> pinnedRealDynamics
      = getPinnedRealDynamicsLinearMap(dq);
  Eigen::MatrixXs A_3 = pinnedRealDynamics.first;
  Eigen::VectorXs b_3 = pinnedRealDynamics.second;

  Eigen::MatrixXs A_composite = A_3 * A_2;
  Eigen::VectorXs b_composite = A_3 * b_2 + b_3;

  return std::make_pair(A_composite, b_composite);
}

//==============================================================================
/// Given the desired end-kinematics, after the human and exoskeleton have
/// finished "negotiating" how they will collaborate, this computes the
/// resulting human and exoskeleton torques.
std::pair<Eigen::VectorXs, Eigen::VectorXs>
ExoSolverPinnedContact::getHumanAndExoTorques(
    Eigen::VectorXs dq, Eigen::VectorXs ddq)
{
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> humanTorquesLinearForwardDynamics
      = getPinnedForwardDynamicsForExoAndHumanLinearMap(dq);

  // Solve Ax+b = ddq for the human torques
  // Ax + b = ddq
  // Ax = ddq - b
  // x = A^-1 (ddq - b)
  Eigen::VectorXs humanTau
      = humanTorquesLinearForwardDynamics.first
            .completeOrthogonalDecomposition()
            .solve(ddq - humanTorquesLinearForwardDynamics.second);

  // Reconstruct the exo torque from the human torque
  Eigen::VectorXs exoTau = solveFromBiologicalTorques(
      dq, humanTau, humanTau, Eigen::VectorXs::Zero(3 * mPins.size()));

  return std::make_pair(humanTau, exoTau);
}

} // namespace exo
} // namespace dart