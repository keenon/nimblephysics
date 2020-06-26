/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>

#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

void debugDofs(SkeletonPtr skel)
{
  std::cout << "DOFs for skeleton '" << skel->getName() << "'" << std::endl;
  for (auto i = 0; i < skel->getNumDofs(); i++)
  {
    std::cout << "   [" << i << "]: '" << skel->getDof(i)->getName() << "'"
              << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
// World testing methods
////////////////////////////////////////////////////////////////////////////////

/**
 * This tests that A_c is being computed correctly, by checking that
 * mapper = A_c.pinv().transpose() * A_c.transpose() does the right thing to a
 * given set of joint velocities. Namely, it maps proposed joint velocities into
 * just the component of motion that's violating the constraints. If we subtract
 * out that components and re-run the solver, we should see no constraint
 * forces.
 *
 * This needs to be done at the world level, because otherwise contact points
 * between two free bodies will look to each body individually as though it's
 * being locked along that axis, which (while correct) is too aggressive a
 * condition, and would break downstream computations.
 */
bool verifyClassicClampingConstraintMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClampingConstraintMatrix forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  // Check that mapper = A_c.pinv().transpose() * A_c.transpose() does the right
  // thing to a given set of joint velocities. Namely, it maps proposed joint
  // velocities into just the component of motion that's violating the
  // constraints. If we subtract out that components and re-run the solver, we
  // should see no constraint forces.

  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  if (A_c.size() == 0)
  {
    // this means that there's no clamping contacts
    return true;
  }

  Eigen::MatrixXd A_cInv
      = A_c.size() > 0 ? A_c.completeOrthogonalDecomposition().pseudoInverse()
                       : Eigen::MatrixXd(0, 0);
  MatrixXd mapper = A_cInv.eval().transpose() * A_c.transpose();
  VectorXd violationVelocities = mapper * proposedVelocities;
  VectorXd cleanVelocities = proposedVelocities - violationVelocities;

  world->setVelocities(cleanVelocities);
  // Populate the constraint matrices, without taking a time step or integrating
  // velocities
  world->getConstraintSolver()->setGradientEnabled(true);
  world->getConstraintSolver()->solve();

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    std::shared_ptr<ConstrainedGroupGradientMatrices> grad
        = skel->getGradientConstraintMatrices();
    if (!grad)
      continue;

    VectorXd cleanContactImpulses = grad->getContactConstraintImpluses();
    VectorXd zero = VectorXd::Zero(cleanContactImpulses.size());
    if (!equals(cleanContactImpulses, zero, 1e-3))
    {
      std::cout << "world A_c: " << std::endl << A_c << std::endl;
      std::cout << "world A_cInv: " << std::endl << A_cInv << std::endl;
      std::cout << "A_c.Inv()^T * A_c^T:" << std::endl << mapper << std::endl;
      std::cout << "proposed Velocities:" << std::endl
                << proposedVelocities << std::endl;
      std::cout << "clean Velocities:" << std::endl
                << cleanVelocities << std::endl;
      std::cout << "Error skeleton " << world->getSkeleton(i)->getName()
                << std::endl
                << " pos: " << std::endl
                << world->getSkeleton(i)->getPositions() << std::endl
                << "vel: " << std::endl
                << world->getSkeleton(i)->getVelocities() << std::endl;
      debugDofs(skel);
      for (Contact contact : world->getLastCollisionResult().getContacts())
      {
        std::cout << "Contact depth " << contact.penetrationDepth << std::endl
                  << contact.point << std::endl;
      }
      std::cout << "actual constraint forces:" << std::endl
                << cleanContactImpulses << std::endl;
      return false;
    }
  }

  return true;
}

/**
 * This verifies the massed formulation by verifying its relationship to the
 * classic formulation.
 */
bool verifyMassedClampingConstraintMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXd V_c = classicPtr->getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXd M = classicPtr->getMassMatrix(world);
  Eigen::MatrixXd Minv = classicPtr->getInvMassMatrix(world);

  Eigen::MatrixXd A_c_recovered = M * V_c;
  Eigen::MatrixXd V_c_recovered = Minv * A_c;

  if (!equals(A_c, A_c_recovered, 1e-3) || !equals(V_c, V_c_recovered, 1e-3))
  {
    std::cout << "A_c massed check failed" << std::endl;
    return false;
  }

  return true;
}

/**
 * This verifies the massed formulation by verifying its relationship to the
 * classic formulation.
 */
bool verifyMassedUpperBoundConstraintMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd V_ub = classicPtr->getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd M = classicPtr->getMassMatrix(world);
  Eigen::MatrixXd Minv = classicPtr->getInvMassMatrix(world);

  Eigen::MatrixXd A_ub_recovered = M * V_ub;
  Eigen::MatrixXd V_ub_recovered = Minv * A_ub;

  if (!equals(A_ub, A_ub_recovered, 1e-3)
      || !equals(V_ub, V_ub_recovered, 1e-3))
  {
    std::cout << "A_ub massed check failed" << std::endl;
    return false;
  }

  return true;
}

/**
 * This tests that P_c is getting computed correctly.
 */
bool verifyClassicProjectionIntoClampsMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicProjectionIntoClampsMatrix forwardPass "
                 "returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  // Get the integrated velocities

  world->integrateVelocities();
  VectorXd integratedVelocities = world->getVelocities();
  world->setVelocities(proposedVelocities);

  // Compute the analytical constraint forces, which should match our actual
  // constraint forces

  MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix(world);
  VectorXd analyticalConstraintForces = -1 * P_c * integratedVelocities;

  // Compute the offset required from the penetration correction velocities

  VectorXd penetrationCorrectionVelocities
      = classicPtr->getPenetrationCorrectionVelocities();
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXd V_c = classicPtr->getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXd V_ub = classicPtr->getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = classicPtr->getUpperBoundMappingMatrix();
  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.size() > 0
            ? forceToVel.completeOrthogonalDecomposition().pseudoInverse()
            : Eigen::MatrixXd(0, 0);
  VectorXd penetrationOffset
      = (velToForce * penetrationCorrectionVelocities) / world->getTimeStep();

  // Sum the two constraints forces together

  VectorXd analyticalConstraintForcesCorrected
      = analyticalConstraintForces + penetrationOffset;

  // Get the actual constraint forces

  VectorXd contactConstraintForces
      = classicPtr->getContactConstraintImpluses() / world->getTimeStep();

  // The analytical constraint forces are a shorter vector than the actual
  // constraint forces, since the analytical constraint forces are only
  // computing the constraints that are clamping. So we need to check equality
  // while taking into account that mapping.

  VectorXi mappings = classicPtr->getContactConstraintMappings();
  VectorXd analyticalError = VectorXd(analyticalConstraintForces.size());
  std::size_t pointer = 0;
  for (std::size_t i = 0; i < mappings.size(); i++)
  {
    if (mappings(i) == neural::ConstraintMapping::CLAMPING)
    {
      analyticalError(pointer) = contactConstraintForces(i)
                                 - analyticalConstraintForcesCorrected(pointer);
      pointer++;
    }
  }

  // Check that the analytical error is zero

  VectorXd zero = VectorXd::Zero(analyticalError.size());
  if (!equals(analyticalError, zero, 1e-3))
  {
    std::cout << "Proposed velocities: " << std::endl
              << proposedVelocities << std::endl;
    std::cout << "Integrated velocities: " << std::endl
              << integratedVelocities << std::endl;
    std::cout << "P_c: " << std::endl << P_c << std::endl;
    std::cout << "bounce: " << std::endl
              << classicPtr->getBounceDiagonals() << std::endl;
    std::cout << "status: " << std::endl;
    for (std::size_t i = 0; i < mappings.size(); i++)
    {
      std::cout << mappings(i) << std::endl;
    }
    std::cout << "Constraint forces: " << std::endl
              << contactConstraintForces << std::endl;
    std::cout << "-(P_c * proposedVelocities) (should be the roughly same as "
                 "actual constraint forces): "
              << std::endl
              << analyticalConstraintForces << std::endl;
    std::cout << "Penetration correction velocities: " << std::endl
              << penetrationCorrectionVelocities << std::endl;
    std::cout << "(A_c^T(V_c + V_ub*E)).pinv() * correction_vels (should "
                 "account for any errors in above): "
              << std::endl
              << penetrationOffset << std::endl;
    std::cout << "Corrected analytical constraint forces (should be the same "
                 "as actual constraint forces): "
              << std::endl
              << analyticalConstraintForcesCorrected << std::endl;
    std::cout << "Analytical error (should be zero):" << std::endl
              << analyticalError << std::endl;
    return false;
  }

  return true;
}

/**
 * This tests that P_c is getting computed correctly.
 */
bool verifyMassedProjectionIntoClampsMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout
        << "verifyWorldMassedProjectionIntoClampsMatrix forwardPass returned a "
           "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
        << std::endl;
    return false;
  }

  Eigen::MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix(world);

  // Reconstruct P_c without the massed shortcut
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = classicPtr->getUpperBoundMappingMatrix();

  Eigen::MatrixXd constraintForceToImpliedTorques = A_c + (A_ub * E);
  Eigen::MatrixXd forceToVel = A_c.eval().transpose()
                               * classicPtr->getInvMassMatrix(world)
                               * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.size() > 0
            ? forceToVel.completeOrthogonalDecomposition().pseudoInverse()
            : Eigen::MatrixXd(0, 0);
  Eigen::MatrixXd bounce = classicPtr->getBounceDiagonals().asDiagonal();
  Eigen::MatrixXd P_c_recovered
      = (1.0 / world->getTimeStep()) * velToForce * bounce * A_c.transpose();

  if (!equals(P_c, P_c_recovered, 1e-4))
  {
    std::cout << "P_c massed check failed" << std::endl;
    std::cout << "P_c:" << std::endl << P_c << std::endl;
    std::cout << "P_c recovered:" << std::endl << P_c_recovered << std::endl;
    return false;
  }

  return true;
}

bool verifyVelVelJacobian(WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicVelVelJacobian forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getVelVelJacobian(world);
  MatrixXd bruteForce = classicPtr->finiteDifferenceVelVelJacobian(world);

  if (!equals(analytical, bruteForce, 1e-4))
  {
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Brute force velCJacobian:" << std::endl
              << classicPtr->getVelCJacobian(world) << std::endl;
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << classicPtr->getForceVelJacobian(world) << std::endl;
    std::cout << "Brute force velCJacobian * forceVelJacobian:" << std::endl
              << classicPtr->getVelCJacobian(world)
                     * classicPtr->getForceVelJacobian(world)
              << std::endl;
    std::cout << "Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyPosVelJacobian(WorldPtr world, VectorXd proposedVelocities)
{
  // TODO(keenon): In the presence of collisions, this is meaningless because
  // the collision correction forces dominate everything else.
  return true;

  /*
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicVelVelJacobian forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getPosVelJacobian(world);
  MatrixXd bruteForce = classicPtr->finiteDifferencePosVelJacobian(world);

  if (!equals(analytical, bruteForce, 1e-6))
  {
    std::cout << "Brute force posVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical posVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
  */
}

bool verifyForceVelJacobian(WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicForceVelJacobian forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getForceVelJacobian(world);
  MatrixXd bruteForce = classicPtr->finiteDifferenceForceVelJacobian(world);

  if (!equals(analytical, bruteForce, 1e-4))
  {
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical forceVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyVelGradients(WorldPtr world, VectorXd worldVel)
{
  return (
      verifyClassicClampingConstraintMatrix(world, worldVel)
      && verifyMassedClampingConstraintMatrix(world, worldVel)
      && verifyMassedUpperBoundConstraintMatrix(world, worldVel)
      && verifyClassicProjectionIntoClampsMatrix(world, worldVel)
      && verifyMassedProjectionIntoClampsMatrix(world, worldVel)
      && verifyVelVelJacobian(world, worldVel)
      && verifyPosVelJacobian(world, worldVel)
      && verifyForceVelJacobian(world, worldVel));
}

bool verifyPosPosJacobianApproximation(WorldPtr world, std::size_t subdivisions)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyPosPosJacobianApproximation forwardPass returned a "
                 "null BackpropSnapshotPtr!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getPosPosJacobian(world);
  MatrixXd bruteForce
      = classicPtr->finiteDifferencePosPosJacobian(world, subdivisions);

  if (!equals(analytical, bruteForce, 1e-4))
  {
    std::cout << "Brute force pos-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical pos-pos Jacobian: " << std::endl
              << analytical << std::endl;
    return false;
  }
  return true;
}

bool verifyVelPosJacobianApproximation(WorldPtr world, std::size_t subdivisions)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyVelPosJacobianApproximation forwardPass returned a "
                 "null BackpropSnapshotPtr!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getVelPosJacobian(world);
  MatrixXd bruteForce
      = classicPtr->finiteDifferenceVelPosJacobian(world, subdivisions);

  if (!equals(analytical, bruteForce, 1e-6))
  {
    std::cout << "Brute force vel-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical vel-pos Jacobian: " << std::endl
              << analytical << std::endl;
    return false;
  }
  return true;
}

bool verifyPosGradients(WorldPtr world, std::size_t subdivisions)
{
  return (
      verifyPosPosJacobianApproximation(world, subdivisions)
      && verifyVelPosJacobianApproximation(world, subdivisions));
}

bool verifyAnalyticalBackpropInstance(
    WorldPtr world,
    const neural::BackpropSnapshotPtr& classicPtr,
    const VectorXd& phaseSpace)
{
  LossGradient nextTimeStep;
  nextTimeStep.lossWrtPosition = phaseSpace.segment(0, phaseSpace.size() / 2);
  nextTimeStep.lossWrtVelocity
      = phaseSpace.segment(phaseSpace.size() / 2, phaseSpace.size() / 2);

  LossGradient thisTimeStep;
  classicPtr->backprop(world, thisTimeStep, nextTimeStep);

  // Compute "brute force" backprop using full Jacobians
  /*
  The forward computation graph looks like this:

  -------> p_t ----+-----------------------> p_t+1 ---->
            /       \
           /         \
  v_t ----+-----------+----(LCP Solver)----> v_t+1 ---->
                     /
                    /
  f_t -------------+
  */

  // p_t
  VectorXd lossWrtThisPosition =
      // p_t --> p_t+1
      (classicPtr->getPosPosJacobian(world).transpose()
       * nextTimeStep.lossWrtPosition)
      // p_t --> v_t+1
      + (classicPtr->getPosVelJacobian(world).transpose()
         * nextTimeStep.lossWrtVelocity);

  // v_t
  VectorXd lossWrtThisVelocity =
      // v_t --> v_t+1
      (classicPtr->getVelVelJacobian(world).transpose()
       * nextTimeStep.lossWrtVelocity)
      // v_t --> p_t
      + (classicPtr->getVelPosJacobian(world).transpose()
         * lossWrtThisPosition);

  // f_t
  VectorXd lossWrtThisTorque =
      // f_t --> v_t+1
      classicPtr->getForceVelJacobian(world).transpose()
      * nextTimeStep.lossWrtVelocity;

  if (!equals(lossWrtThisPosition, thisTimeStep.lossWrtPosition, 1e-5)
      || !equals(lossWrtThisVelocity, thisTimeStep.lossWrtVelocity, 1e-5)
      || !equals(lossWrtThisTorque, thisTimeStep.lossWrtTorque, 1e-5))
  {
    std::cout << "Input: loss wrt position at time t + 1:" << std::endl
              << nextTimeStep.lossWrtPosition << std::endl;
    std::cout << "Input: loss wrt velocity at time t + 1:" << std::endl
              << nextTimeStep.lossWrtVelocity << std::endl;

    std::cout << "-----" << std::endl;

    std::cout << "Brute force: loss wrt position at time t:" << std::endl
              << lossWrtThisPosition << std::endl;
    std::cout << "Analytical: loss wrt position at time t:" << std::endl
              << thisTimeStep.lossWrtPosition << std::endl;
    std::cout << "pos-vel Jacobian:" << std::endl
              << classicPtr->getPosVelJacobian(world) << std::endl;
    std::cout << "pos-C Jacobian:" << std::endl
              << classicPtr->getPosCJacobian(world) << std::endl;
    std::cout << "Brute force: pos-pos Jac:" << std::endl
              << classicPtr->getPosPosJacobian(world) << std::endl;

    std::cout << "-----" << std::endl;

    std::cout << "Brute force: loss wrt velocity at time t:" << std::endl
              << lossWrtThisVelocity << std::endl;
    std::cout << "Analytical: loss wrt velocity at time t:" << std::endl
              << thisTimeStep.lossWrtVelocity << std::endl;
    std::cout << "vel-vel Jacobian:" << std::endl
              << classicPtr->getVelVelJacobian(world) << std::endl;
    std::cout << "vel-pos Jacobian:" << std::endl
              << classicPtr->getVelPosJacobian(world) << std::endl;
    std::cout << "vel-C Jacobian:" << std::endl
              << classicPtr->getVelCJacobian(world) << std::endl;
    std::cout << "v_t --> v_t+1:" << std::endl
              << (classicPtr->getVelVelJacobian(world).transpose()
                  * nextTimeStep.lossWrtVelocity)
              << std::endl;
    std::cout << "v_t --> p_t:" << std::endl
              << (classicPtr->getVelPosJacobian(world).transpose()
                  * lossWrtThisPosition)
              << std::endl;

    std::cout << "-----" << std::endl;

    std::cout << "Brute force: loss wrt torque at time t:" << std::endl
              << lossWrtThisTorque << std::endl;
    std::cout << "Analytical: loss wrt torque at time t:" << std::endl
              << thisTimeStep.lossWrtTorque << std::endl;
    return false;
  }
  return true;
}

bool verifyAnalyticalBackprop(WorldPtr world)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyVelPosJacobianApproximation forwardPass returned a "
                 "null BackpropSnapshotPtr!"
              << std::endl;
    return false;
  }

  VectorXd phaseSpace = VectorXd::Zero(world->getNumDofs() * 2);

  // Test a "1" in each dimension of the phase space separately
  for (std::size_t i = 0; i < world->getNumDofs() * 2; i++)
  {
    phaseSpace(i) = 1;
    if (i > 0)
      phaseSpace(i - 1) = 0;
    if (!verifyAnalyticalBackpropInstance(world, classicPtr, phaseSpace))
      return false;
  }

  // Test all "0"s
  phaseSpace = VectorXd::Zero(world->getNumDofs() * 2);
  if (!verifyAnalyticalBackpropInstance(world, classicPtr, phaseSpace))
    return false;

  // Test all "1"s
  phaseSpace = VectorXd::Ones(world->getNumDofs() * 2);
  if (!verifyAnalyticalBackpropInstance(world, classicPtr, phaseSpace))
    return false;

  return true;
}

LossGradient computeBruteForceGradient(
    WorldPtr world, std::size_t timesteps, std::function<double(WorldPtr)> loss)
{
  RestorableSnapshot snapshot(world);

  std::size_t n = world->getNumDofs();
  LossGradient grad;
  grad.lossWrtPosition = Eigen::VectorXd(n);
  grad.lossWrtVelocity = Eigen::VectorXd(n);
  grad.lossWrtTorque = Eigen::VectorXd(n);

  for (std::size_t k = 0; k < timesteps; k++)
    world->step();
  double defaultLoss = loss(world);
  snapshot.restore();

  Eigen::VectorXd originalPos = world->getPositions();
  Eigen::VectorXd originalVel = world->getVelocities();
  Eigen::VectorXd originalForce = world->getForces();

  double EPSILON = 1e-7;

  for (std::size_t i = 0; i < n; i++)
  {
    Eigen::VectorXd tweakedPos = originalPos;
    tweakedPos(i) += EPSILON;

    snapshot.restore();
    world->setPositions(tweakedPos);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step();
    grad.lossWrtPosition(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXd tweakedVel = originalVel;
    tweakedVel(i) += EPSILON;

    snapshot.restore();
    world->setVelocities(tweakedVel);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step();
    grad.lossWrtVelocity(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXd tweakedForce = originalForce;
    tweakedForce(i) += EPSILON;

    snapshot.restore();
    world->setForces(tweakedForce);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step();
    grad.lossWrtTorque(i) = (loss(world) - defaultLoss) / EPSILON;
  }

  snapshot.restore();
  return grad;
}

bool verifyGradientBackprop(
    WorldPtr world, std::size_t timesteps, std::function<double(WorldPtr)> loss)
{
  // Get the brute force the compare against
  LossGradient bruteForce = computeBruteForceGradient(world, timesteps, loss);

  RestorableSnapshot snapshot(world);

  std::vector<BackpropSnapshotPtr> snapshots;
  snapshots.reserve(timesteps);
  for (std::size_t i = 0; i < timesteps; i++)
  {
    snapshots.push_back(forwardPass(world));
  }

  // Get the loss gradient at the final timestep (by brute force)
  LossGradient analytical = computeBruteForceGradient(world, 0, loss);

  std::vector<LossGradient> gradients;
  gradients.push_back(analytical);
  for (int i = timesteps - 1; i >= 0; i--)
  {
    LossGradient thisTimestep;
    snapshots[i]->backprop(world, thisTimestep, analytical);
    gradients.push_back(thisTimestep);
    analytical = thisTimestep;
  }

  // Assert that the results are the same
  if (!equals(analytical.lossWrtPosition, bruteForce.lossWrtPosition, 1e-5)
      || !equals(analytical.lossWrtVelocity, bruteForce.lossWrtVelocity, 1e-5)
      || !equals(analytical.lossWrtTorque, bruteForce.lossWrtTorque, 1e-5))
  {
    std::cout << "Analytical loss wrt position:" << std::endl
              << analytical.lossWrtPosition << std::endl;
    std::cout << "Brute force loss wrt position:" << std::endl
              << bruteForce.lossWrtPosition << std::endl;
    std::cout << "Analytical loss wrt velocity:" << std::endl
              << analytical.lossWrtVelocity << std::endl;
    std::cout << "Brute force loss wrt velocity:" << std::endl
              << bruteForce.lossWrtVelocity << std::endl;
    std::cout << "Analytical loss wrt torque:" << std::endl
              << analytical.lossWrtTorque << std::endl;
    std::cout << "Brute force loss wrt torque:" << std::endl
              << bruteForce.lossWrtTorque << std::endl;
    return false;
  }

  return true;
}

// This test is ugly and difficult to interpret, but it broke our system early
// on. It now passes, and is here to detect regression.
/******************************************************************************

This test sets up a configuration that looks like this:

          O
          |
          | +
Force --> O | <-- Fixed
          | +
          |
          O

There's a 3 link pendulum, with a force driving the middle link into a fixed
block, creating a contact.

*/
/*
TEST(GRADIENTS, PENDULUM_BLOCK)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the pendulum
  ///////////////////////////////////////////////

  SkeletonPtr pendulum = Skeleton::create("pendulum");

  std::pair<RevoluteJoint*, BodyNode*> pair;
  BodyNode *body1, *body2, *body3;
  RevoluteJoint *joint1, *joint2, *joint3;

  RevoluteJoint::Properties jointProps;
  BodyNode::Properties bodyProps;

  jointProps.mName = 'Joint_1';
  bodyProps.mName = 'Body_1';
  pair = pendulum->createJointAndBodyNodePair<RevoluteJoint>(
      nullptr, jointProps, bodyProps);
  joint1 = pair.first;
  body1 = pair.second;

  jointProps.mName = 'Joint_2';
  bodyProps.mName = 'Body_2';
  pair = body1->createChildJointAndBodyNodePair<RevoluteJoint>(
      jointProps, bodyProps);
  joint2 = pair.first;
  body2 = pair.second;

  jointProps.mName = 'Joint_3';
  bodyProps.mName = 'Body_3';
  pair = body2->createChildJointAndBodyNodePair<RevoluteJoint>(
      jointProps, bodyProps);
  joint3 = pair.first;
  body3 = pair.second;

  Eigen::Isometry3d offset(Eigen::Isometry3d::Identity());
  offset.translation().noalias() = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d axis = Eigen::Vector3d(0.0, 1.0, 0.0);

  // Joints
  joint1->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  joint1->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint1->setAxis(axis);

  joint2->setTransformFromParentBodyNode(offset);
  joint2->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint2->setAxis(axis);

  joint3->setTransformFromParentBodyNode(offset);
  joint3->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());
  joint3->setAxis(axis);

  // Add collisions to the last node of the chain
  std::shared_ptr<BoxShape> pendulumBox(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  body1->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body1->setFrictionCoeff(0);
  body2->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body2->setFrictionCoeff(0);
  body3->createShapeNodeWith<VisualAspect, CollisionAspect>(pendulumBox);
  body3->setFrictionCoeff(0);

  // The block is to the right, drive the chain into the block
  body2->setExtForce(Eigen::Vector3d(5.0, 0, 0));
  world->addSkeleton(pendulum);

  ///////////////////////////////////////////////
  // Create the block
  ///////////////////////////////////////////////

  SkeletonPtr block = Skeleton::create("block");

  // Give the floor a body
  BodyNodePtr body
      = block->createJointAndBodyNodePair<WeldJoint>(nullptr).second;

  // Give the body a shape
  std::shared_ptr<BoxShape> box(new BoxShape(Eigen::Vector3d(1.0, 0.5, 0.5)));
  auto shapeNode
      = body->createShapeNodeWith<VisualAspect, CollisionAspect>(box);
  shapeNode->getVisualAspect()->setColor(dart::Color::Black());

  // Put the body into position
  Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());
  tf.translation() = Eigen::Vector3d(0.55, 0.0, -1.0);
  body->getParentJoint()->setTransformFromParentBodyNode(tf);

  world->addSkeleton(block);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  pendulum->computeForwardDynamics();
  pendulum->integrateVelocities(world->getTimeStep());
  VectorXd timestepVel = pendulum->getVelocities();

  VectorXd worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyWorldGradients(world, worldVel));
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

          Force
            |
            v
          +---+
Force --> |   |
          +---+
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground has configurable friction in this setup.

*/
void testBlockWithFrictionCoeff(double frictionCoeff, double mass)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = box->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -1.0, 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXd timestepVel = box->getVelocities();
  VectorXd timestepWorldVel = world->getVelocities();

  VectorXd worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_1_MASS)
{
  testBlockWithFrictionCoeff(0, 1);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_2_MASS)
{
  testBlockWithFrictionCoeff(0, 2);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_NO_FRICTION_4_MASS)
{
  testBlockWithFrictionCoeff(0, 4);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_STATIC_FRICTION)
{
  testBlockWithFrictionCoeff(1e7, 1);
}

TEST(GRADIENTS, BLOCK_ON_GROUND_SLIPPING_FRICTION)
{
  testBlockWithFrictionCoeff(0.5, 1);
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

                +---+
          +---+ |   |
Force --> |   | |   | <-- Force
          +---+ |   |
                +---+

There are two blocks, each with two DOFs (X, Y). The force pushing them together
(or apart, if negative) is configurable.

The right box is larger, to prevent exact vertex-vertex collisions, which are
hard for the engine to handle.

*/
void testTwoBlocks(
    double leftPressingForce,
    double rightPressingForce,
    double frictionCoeff,
    double leftMass,
    double rightMass)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the left box
  ///////////////////////////////////////////////

  SkeletonPtr leftBox = Skeleton::create("left box");

  std::pair<PrismaticJoint*, BodyNode*> leftBoxPair
      = leftBox->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* leftBoxJoint = leftBoxPair.first;
  BodyNode* leftBoxBody = leftBoxPair.second;

  leftBoxJoint->setAxis(Eigen::Vector3d::UnitX());
  Eigen::Isometry3d leftBoxPosition = Eigen::Isometry3d::Identity();
  leftBoxPosition.translation() = Eigen::Vector3d(-0.5, 0, 0);
  leftBoxJoint->setTransformFromParentBodyNode(leftBoxPosition);
  leftBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> leftBoxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* leftBoxShapeNode
      = leftBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          leftBoxShape);
  leftBoxShapeNode->setName("Left box shape");
  leftBoxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the right
  leftBoxBody->addExtForce(Eigen::Vector3d(leftPressingForce, -1, 0));
  // Prevent the mass matrix from being Identity
  leftBoxBody->setMass(leftMass);

  world->addSkeleton(leftBox);

  ///////////////////////////////////////////////
  // Create the right box
  ///////////////////////////////////////////////

  SkeletonPtr rightBox = Skeleton::create("right box");

  std::pair<PrismaticJoint*, BodyNode*> rightBoxPair
      = rightBox->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  PrismaticJoint* rightBoxJoint = rightBoxPair.first;
  BodyNode* rightBoxBody = rightBoxPair.second;

  rightBoxJoint->setAxis(Eigen::Vector3d::UnitX());
  Eigen::Isometry3d rightBoxPosition = Eigen::Isometry3d::Identity();
  rightBoxPosition.translation() = Eigen::Vector3d(0.5, 0, 0);
  rightBoxJoint->setTransformFromParentBodyNode(rightBoxPosition);
  rightBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> rightBoxShape(
      new BoxShape(Eigen::Vector3d(1.0, 2.0, 2.0)));
  ShapeNode* rightBoxShapeNode
      = rightBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          rightBoxShape);
  rightBoxBody->setFrictionCoeff(frictionCoeff);
  rightBoxShapeNode->setName("Right box shape");

  // Add a force driving the box down into the floor, and to the left
  rightBoxBody->addExtForce(Eigen::Vector3d(-rightPressingForce, -1, 0));
  // Prevent the mass matrix from being Identity
  rightBoxBody->setMass(rightMass);

  world->addSkeleton(rightBox);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  leftBox->computeForwardDynamics();
  leftBox->integrateVelocities(world->getTimeStep());
  VectorXd leftBoxVel = leftBox->getVelocities();

  rightBox->computeForwardDynamics();
  rightBox->integrateVelocities(world->getTimeStep());
  VectorXd rightBoxVel = rightBox->getVelocities();

  VectorXd worldVel = world->getVelocities();

  /*
  world->getConstraintSolver()->solve();
  std::cout << "Contacts: " << world->getLastCollisionResult().getNumContacts()
            << std::endl;
  for (std::size_t i = 0; i < world->getLastCollisionResult().getNumContacts();
       i++)
  {
    collision::Contact contact = world->getLastCollisionResult().getContact(i);
    std::cout << "Contact " << i << " "
              << contact.collisionObject1->getShapeFrame()->getName() << "<->"
              << contact.collisionObject2->getShapeFrame()->getName() << ": "
              << contact.point << " at depth " << contact.penetrationDepth
              << std::endl;
  }
  */

  // Test the classic formulation

  world->getConstraintSolver()->setGradientEnabled(true);
  world->getConstraintSolver()->solve();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, TWO_BLOCKS_1_1_MASS)
{
  testTwoBlocks(1, 1, 0, 1, 1);
}

TEST(GRADIENTS, TWO_BLOCKS_1_2_MASS)
{
  testTwoBlocks(2, 1, 0, 1, 2);
}

TEST(GRADIENTS, TWO_BLOCKS_3_5_MASS)
{
  testTwoBlocks(2, 1, 0, 3, 5);
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

      Large Velocity
            |
            v
          +---+
Force --> |   |
          +---+
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground and the block both have coefficients of restitution of 0.5.
The ground has configurable friction in this setup.

*/
void testBouncingBlockWithFrictionCoeff(double frictionCoeff, double mass)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = box->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -1.0, 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockWithFrictionCoeff(0, 1);
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

          +---+
          | x |
          +-|-+
            |
            |
          +-|-+
Force --> | O |
          +---+
      -------------
            ^
       Fixed ground

There's a reverse pendulum sled with three DOFs, x and y axis, and angle of the
reverse pendulum, with a force driving it into the ground. The ground has
configurable friction in this setup.

*/
void testReversePendulumSledWithFrictionCoeff(double frictionCoeff)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr reversePendulumSled = Skeleton::create("reversePendulumSled");

  TranslationalJoint2D::Properties jointProps;
  BodyNode::Properties bodyProps;
  jointProps.mName = "2D Sled Translation";
  bodyProps.mName = "Sled";
  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = reversePendulumSled->createJointAndBodyNodePair<TranslationalJoint2D>(
          nullptr, jointProps, bodyProps);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));

  // Create the reverse pendulum portion

  RevoluteJoint::Properties pendulumJointProps;
  pendulumJointProps.mName = "Reverse Pendulum Joint";
  bodyProps.mName = "Reverse Pendulum Body";

  std::pair<RevoluteJoint*, BodyNode*> pendulumPair
      = boxBody->createChildJointAndBodyNodePair<RevoluteJoint>(
          pendulumJointProps, bodyProps);
  RevoluteJoint* pendulumJoint = pendulumPair.first;
  BodyNode* pendulumBody = pendulumPair.second;

  pendulumJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  Eigen::Isometry3d pendulumBodyPosition = Eigen::Isometry3d::Identity();
  pendulumBodyPosition.translation() = Eigen::Vector3d(0, 1, 0);
  pendulumJoint->setTransformFromChildBodyNode(pendulumBodyPosition);
  pendulumJoint->setAxis(Eigen::Vector3d(0, 0, 1.0));
  pendulumBody->setMass(200);

  world->addSkeleton(reversePendulumSled);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -1.0, 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  reversePendulumSled->computeForwardDynamics();
  reversePendulumSled->integrateVelocities(world->getTimeStep());
  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, SLIDING_REVERSE_PENDULUM_NO_FRICTION)
{
  testReversePendulumSledWithFrictionCoeff(0);
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

      Large Velocity
            |
            v
          +---+
Force --> |   |
          +---+
              <-- some small air gap
      -------------
            ^
       Fixed ground

There's a box with two DOFs, x and y axis, with a force driving it into the
ground. The ground and the block both have coefficients of restitution of 0.5.
The ground has configurable friction in this setup.

*/
void testBouncingBlockPosGradients(double frictionCoeff, double mass)
{
  // World
  WorldPtr world = World::create();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<TranslationalJoint2D*, BodyNode*> pair
      = box->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  boxJoint->setXYPlane();
  boxJoint->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());
  boxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -1.0, 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(1);
  floorBody->setRestitutionCoeff(1.0);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  EXPECT_TRUE(verifyPosGradients(world, 100));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, POS_BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockPosGradients(0, 1);
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

        Velocity      Velocity
            |             |
            v             v
          +---+         +---+
          |   |         |   |        * * *
          +---+         +---+
        +-------+     +-------+
        |       |     |       |
        +-------+     +-------+
            ^             ^
        Velocity      Velocity

There are "numGroups" pairs of boxes, each with force driving them together.

*/
void testMultigroup(int numGroups)
{
  // World
  WorldPtr world = World::create();

  std::vector<SkeletonPtr> topBoxes;
  std::vector<SkeletonPtr> bottomBoxes;

  for (std::size_t i = 0; i < numGroups; i++)
  {
    // This is where this group is going to be positioned along the x axis
    double xOffset = i * 10;

    // Create the top box in the pair

    SkeletonPtr topBox = Skeleton::create("topBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> topBoxPair
        = topBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* topBoxJoint = topBoxPair.first;
    BodyNode* topBoxBody = topBoxPair.second;

    topBoxJoint->setXYPlane();
    Eigen::Isometry3d topBoxPosition = Eigen::Isometry3d::Identity();
    topBoxPosition.translation() = Eigen::Vector3d(xOffset, 0.5, 0);
    topBoxJoint->setTransformFromParentBodyNode(topBoxPosition);
    topBoxJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

    std::shared_ptr<BoxShape> topBoxShape(
        new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
    topBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(topBoxShape);
    topBoxBody->setFrictionCoeff(0.5);
    topBoxBody->setExtForce(Eigen::Vector3d(0, -1.0, 0));

    topBoxes.push_back(topBox);

    // Create the bottom box in the pair

    SkeletonPtr bottomBox = Skeleton::create("bottomBox_" + std::to_string(i));

    std::pair<TranslationalJoint2D*, BodyNode*> bottomBoxPair
        = bottomBox->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
    TranslationalJoint2D* bottomBoxJoint = bottomBoxPair.first;
    BodyNode* bottomBoxBody = bottomBoxPair.second;

    bottomBoxJoint->setXYPlane();
    Eigen::Isometry3d bottomBoxPosition = Eigen::Isometry3d::Identity();
    bottomBoxPosition.translation() = Eigen::Vector3d(xOffset, -0.5, 0);
    bottomBoxJoint->setTransformFromParentBodyNode(bottomBoxPosition);
    bottomBoxJoint->setTransformFromChildBodyNode(
        Eigen::Isometry3d::Identity());

    std::shared_ptr<BoxShape> bottomBoxShape(
        new BoxShape(Eigen::Vector3d(2.0, 1.0, 2.0)));
    bottomBoxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
        bottomBoxShape);
    bottomBoxBody->setFrictionCoeff(1);
    bottomBoxBody->setExtForce(Eigen::Vector3d(0, 1.0, 0));
    // Make each group less symmetric
    bottomBoxBody->setMass(1.0 / (i + 1));

    bottomBoxes.push_back(bottomBox);

    // Add a tiny bit of velocity to the boxes
    topBox->computeForwardDynamics();
    topBox->integrateVelocities(world->getTimeStep());
    bottomBox->computeForwardDynamics();
    bottomBox->integrateVelocities(world->getTimeStep());
  }

  // Add all the top boxes first, then all the bottom boxes. This ensures that
  // our constraint group ordering doesn't match our world ordering, which will
  // help us catch bugs in matrix layout.
  for (SkeletonPtr topBox : topBoxes)
    world->addSkeleton(topBox);
  for (SkeletonPtr bottomBox : bottomBoxes)
    world->addSkeleton(bottomBox);

  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, MULTIGROUP_2)
{
  testMultigroup(2);
}

TEST(GRADIENTS, MULTIGROUP_4)
{
  testMultigroup(4);
}
*/

/******************************************************************************

This test sets up a configuration that looks something like this:

                        |
                  _____ |
                O _____O| < Fixed wall
              / /       |
             / /        |
            / /         |
            O           |
           | |          |
           | |          |
           | |          |
            O
            ^             ^
      Rotating base

It's a robot arm, with a rotating base, with "numLinks" links and
"rotationDegree" position at each link. There's also a fixed plane at the end of
the robot arm that it intersects with.
*/
void testRobotArm(std::size_t numLinks, double rotationRadians)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  for (std::size_t i = 0; i < numLinks; i++)
  {
    RevoluteJoint::Properties jointProps;
    jointProps.mName = "revolute_" + std::to_string(i);
    BodyNode::Properties bodyProps;
    bodyProps.mName = "arm_" + std::to_string(i);
    std::pair<RevoluteJoint*, BodyNode*> jointPair
        = arm->createJointAndBodyNodePair<RevoluteJoint>(
            parent, jointProps, bodyProps);
    if (parent != nullptr)
    {
      Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
      armOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
  }

  std::shared_ptr<SphereShape> endShape(new SphereShape(1.0));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);

  arm->setPositions(Eigen::VectorXd::Ones(arm->getNumDofs()) * rotationRadians);
  world->addSkeleton(arm);

  SkeletonPtr wall = Skeleton::create("wall");
  std::pair<WeldJoint*, BodyNode*> jointPair
      = wall->createJointAndBodyNodePair<WeldJoint>(nullptr);
  std::shared_ptr<BoxShape> wallShape(
      new BoxShape(Eigen::Vector3d(1.0, 10.0, 10.0)));
  ShapeNode* wallNode
      = jointPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          wallShape);
  world->addSkeleton(wall);
  // jointPair.second->setFrictionCoeff(0.0);

  Eigen::Isometry3d wallLocalOffset = Eigen::Isometry3d::Identity();
  wallLocalOffset.translation() = parent->getWorldTransform().translation()
                                  + Eigen::Vector3d(-(1.5 - 1e-7), 0.0, 0);
  jointPair.first->setTransformFromParentBodyNode(wallLocalOffset);

  /*
  // Run collision detection
  world->getConstraintSolver()->solve();

  // Check
  auto result = world->getLastCollisionResult();
  if (result.getNumContacts() > 0)
  {
    std::cout << "Num contacts: " << result.getNumContacts() << std::endl;
    std::cout << "end affector offset: " << std::endl
              << endNode->getWorldTransform().matrix() << std::endl;
    std::cout << "wall node position: " << std::endl
              << wallNode->getWorldTransform().matrix() << std::endl;
  }
  */

  // arm->computeForwardDynamics();
  // arm->integrateVelocities(world->getTimeStep());
  // -0.029 at 0.5
  // -0.323 at 5.0
  arm->setVelocities(Eigen::VectorXd::Ones(arm->getNumDofs()) * 0.001);

  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

/*
TEST(GRADIENTS, ARM_3_LINK_30_DEG)
{
  testRobotArm(3, 30.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_5_LINK_30_DEG)
{
  // This test wraps an arm around, and it's actually breaking contact, so this
  // tests unconstrained free-motion
  testRobotArm(5, 30.0 / 180 * 3.1415);
}

TEST(GRADIENTS, ARM_6_LINK_15_DEG)
{
  testRobotArm(6, 15.0 / 180 * 3.1415);
}
*/

/******************************************************************************

This test sets up a configuration that looks something like this:

           | |
           | |
           | |
    ======= O =======
            ^
      Rotating base

It's a robot arm, with a rotating base, with "numLinks" links and
"rotationDegree" position at each link. There's also a fixed plane at the end of
the robot arm that it intersects with.
*/
void testCartpole(double rotationRadians)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 0, 0));

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
  armOffset.translation() = Eigen::Vector3d(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  /*
  cartpole = dart.dynamics.Skeleton()
  cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
  cartRail.setAxis([1, 0, 0])
  cartShape = cart.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
  cartVisual = cartShape.createVisualAspect()
  cartVisual.setColor([0, 0, 0])

  poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
  poleJoint.setAxis([0, 0, 1])
  poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
  poleVisual = poleShape.createVisualAspect()
  poleVisual.setColor([0, 0, 0])

  poleOffset = dart.math.Isometry3()
  poleOffset.set_translation([0, -0.5, 0])
  poleJoint.setTransformFromChildBodyNode(poleOffset)
  */

  world->addSkeleton(cartpole);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, rotationRadians);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyGradientBackprop(world, 10, [](WorldPtr world) {
    Eigen::Vector2d pos = world->getPositions();
    Eigen::Vector2d vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  }));
}

TEST(GRADIENTS, CARTPOLE_15_DEG)
{
  testCartpole(15.0 / 180.0 * 3.1415);
}

///////////////////////////////////////////////////////////////////////////////
// Just idiot checking that the code doesn't crash on silly edge cases.
///////////////////////////////////////////////////////////////////////////////

/*
TEST(GRADIENTS, EMPTY_WORLD)
{
  WorldPtr world = World::create();
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyBackprop(world));
}

TEST(GRADIENTS, EMPTY_SKELETON)
{
  WorldPtr world = World::create();
  SkeletonPtr empty = Skeleton::create("empty");
  world->addSkeleton(empty);
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyBackprop(world));
}
*/