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

#include <dart/gui/gui.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
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

  Eigen::MatrixXd A_cInv;
  if (A_c.size() > 0)
  {
    A_cInv = A_c.completeOrthogonalDecomposition().pseudoInverse();
  }
  else
  {
    A_cInv = Eigen::MatrixXd::Zero(0, 0);
  }
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
    if (!equals(cleanContactImpulses, zero, 1e-9))
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

  if (!equals(A_c, A_c_recovered, 1e-8) || !equals(V_c, V_c_recovered, 1e-8))
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

  if (!equals(A_ub, A_ub_recovered, 1e-8)
      || !equals(V_ub, V_ub_recovered, 1e-8))
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

  bool oldPenetrationCorrection
      = world->getConstraintSolver()->getPenetrationCorrectionEnabled();
  world->getConstraintSolver()->setPenetrationCorrectionEnabled(false);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  world->getConstraintSolver()->setPenetrationCorrectionEnabled(
      oldPenetrationCorrection);

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
  Eigen::MatrixXd velToForce = Eigen::MatrixXd::Zero(0, 0);
  if (forceToVel.size() > 0)
  {
    velToForce = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  }
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
  double constraintForces = contactConstraintForces.norm();
  if (!equals(analyticalError, zero, 1e-8))
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
  Eigen::MatrixXd velToForce = Eigen::MatrixXd::Zero(0, 0);
  if (forceToVel.size() > 0)
  {
    velToForce = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  }
  Eigen::MatrixXd bounce = classicPtr->getBounceDiagonals().asDiagonal();
  Eigen::MatrixXd P_c_recovered
      = (1.0 / world->getTimeStep()) * velToForce * bounce * A_c.transpose();

  if (!equals(P_c, P_c_recovered, 1e-8))
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

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    std::cout << "Brute force velCJacobian:" << std::endl
              << classicPtr->getVelCJacobian(world) << std::endl;
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << classicPtr->getForceVelJacobian(world) << std::endl;
    std::cout << "Brute force forceVelJacobian * velCJacobian:" << std::endl
              << classicPtr->getForceVelJacobian(world)
                     * classicPtr->getVelCJacobian(world)
              << std::endl;
    return false;
  }

  return true;
}

bool verifyAnalyticalConstraintMatrixEstimates(WorldPtr world)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::MatrixXd original = classicPtr->getClampingConstraintMatrix(world);

  double EPS = 1e-5;

  for (int i = 0; i < 100; i++)
  {
    Eigen::VectorXd diff = Eigen::VectorXd::Random(world->getNumDofs()) * EPS;
    Eigen::VectorXd pos = world->getPositions() + diff;

    Eigen::MatrixXd analytical
        = classicPtr->estimateClampingConstraintMatrixAt(world, pos);
    Eigen::MatrixXd bruteForce
        = classicPtr->getClampingConstraintMatrixAt(world, pos);

    Eigen::MatrixXd analyticalDiff = (analytical - original) / EPS;
    Eigen::MatrixXd bruteForceDiff = (bruteForce - original) / EPS;

    // I'm surprised by how quickly the gradient can change
    if (!equals(analyticalDiff, bruteForceDiff, 2e-3))
    {
      std::cout << "Error in analytical A_c estimates:" << std::endl;
      std::cout << "Analytical Diff:" << std::endl
                << analyticalDiff << std::endl;
      std::cout << "Brute Force Diff:" << std::endl
                << bruteForceDiff << std::endl;
      std::cout << "Estimate Diff Error (2nd+ order effects):" << std::endl
                << analyticalDiff - bruteForceDiff << std::endl;
      std::cout << "Position Diff:" << std::endl << diff << std::endl;
      return false;
    }
  }
  return true;
}

bool verifyF_c(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);

  Eigen::MatrixXd realQ = classicPtr->getClampingAMatrix();
  Eigen::VectorXd realB = classicPtr->getClampingConstraintRelativeVels();

  Eigen::MatrixXd analyticalQ = Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  classicPtr->computeLCPConstraintMatrixClampingSubset(world, analyticalQ, A_c);
  Eigen::VectorXd analyticalB = Eigen::VectorXd(A_c.cols());
  classicPtr->computeLCPOffsetClampingSubset(world, analyticalB, A_c);

  if (!equals(realB, analyticalB, 1e-7) || !equals(realQ, analyticalQ, 2e-5))
  {
    std::cout << "Real Q:" << std::endl << realQ << std::endl;
    std::cout << "Analytical Q:" << std::endl << analyticalQ << std::endl;
    std::cout << "Diff Q:" << std::endl << (realQ - analyticalQ) << std::endl;
    std::cout << "Real B:" << std::endl << realB << std::endl;
    std::cout << "Analytical B:" << std::endl << analyticalB << std::endl;
    classicPtr->computeLCPOffsetClampingSubset(world, analyticalB, A_c);
    std::cout << "Diff B:" << std::endl << (realB - analyticalB) << std::endl;
    return false;
  }

  Eigen::VectorXd analyticalF_c
      = classicPtr->estimateClampingConstraintImpulses(world, A_c);
  Eigen::VectorXd realF_c = classicPtr->getClampingConstraintImpulses();
  if (!equals(analyticalF_c, realF_c, 2e-5))
  {
    std::cout << "Real f_c:" << std::endl << realF_c << std::endl;
    std::cout << "Analytical f_c:" << std::endl << analyticalF_c << std::endl;
    std::cout << "Diff f_c:" << std::endl
              << (realF_c - analyticalF_c) << std::endl;
    std::cout << "Real Q:" << std::endl << realQ << std::endl;
    std::cout << "Real B:" << std::endl << realB << std::endl;
    std::cout << "Real Qinv*B:" << std::endl
              << realQ.completeOrthogonalDecomposition().solve(realB)
              << std::endl;
    return false;
  }

  Eigen::MatrixXd bruteForceJac
      = classicPtr->finiteDifferenceJacobianOfConstraintForce(
          world, WithRespectTo::POSITION);
  Eigen::MatrixXd analyticalJac = classicPtr->getJacobianOfConstraintForce(
      world, WithRespectTo::POSITION);
  if (!equals(analyticalJac, bruteForceJac, 2e-5))
  {
    std::cout << "Brute force f_c Jacobian:" << std::endl
              << bruteForceJac << std::endl;
    std::cout << "Analytical f_c Jacobian:" << std::endl
              << analyticalJac << std::endl;
    std::cout << "Diff Jac:" << std::endl
              << (bruteForceJac - analyticalJac) << std::endl;
    return false;
  }

  return true;
}

struct VelocityTest
{
  Eigen::VectorXd realNextVel;
  Eigen::VectorXd realNextVelPreSolve;
  Eigen::VectorXd realNextVelDeltaVFromF;
  Eigen::VectorXd predictedNextVel;
  Eigen::VectorXd predictedNextVelPreSolve;
  Eigen::VectorXd predictedNextVelDeltaVFromF;
  Eigen::VectorXd preStepVelocity;
};

VelocityTest runVelocityTest(WorldPtr world)
{
  RestorableSnapshot snapshot(world);
  world->step(false, true);
  Eigen::VectorXd realNextVel = world->getVelocities();
  snapshot.restore();
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    skel->computeForwardDynamics();
    skel->integrateVelocities(world->getTimeStep());
  }
  Eigen::VectorXd realNextVelPreSolve = world->getVelocities();
  Eigen::VectorXd realNextVelDeltaVFromF = realNextVel - realNextVelPreSolve;
  snapshot.restore();

  Eigen::VectorXd preStepVelocity = world->getVelocities();

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = classicPtr->getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;
  Eigen::VectorXd tau = world->getForces();
  double dt = world->getTimeStep();

  Eigen::MatrixXd Minv = world->getInvMassMatrix();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXd f_c
      = classicPtr->estimateClampingConstraintImpulses(world, A_c);

  /*
  Eigen::VectorXd b = Eigen::VectorXd(A_c.cols());
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  classicPtr->computeLCPOffsetClampingSubset(world, b, A_c);
  classicPtr->computeLCPConstraintMatrixClampingSubset(world, Q, A_c);
  std::cout << "Real B: " << std::endl
            << classicPtr->getClampingConstraintRelativeVels() << std::endl;
  std::cout << "Analytical B: " << std::endl << b << std::endl;
  std::cout << "Real A: " << std::endl
            << classicPtr->mGradientMatrices[0]->mA << std::endl;
  std::cout << "Analytical A: " << std::endl << Q << std::endl;
  */

  Eigen::VectorXd allRealImpulses = classicPtr->getContactConstraintImpluses();
  /*
  Eigen::VectorXd velChange = Eigen::VectorXd::Zero(world->getNumDofs());
  for (int i = 0; i < allRealImpulses.size(); i++)
  {
    velChange += allRealImpulses(i)
                 * classicPtr->mGradientMatrices[0]->mMassedImpulseTests[i];
  }
  */
  Eigen::VectorXd velDueToIllegal
      = classicPtr->getVelocityDueToIllegalImpulses();

  Eigen::VectorXd realImpulses = classicPtr->getClampingConstraintImpulses();

  Eigen::VectorXd preSolveV = preStepVelocity + dt * Minv * (tau - C);

  Eigen::VectorXd f_cDeltaV;
  if (A_c.cols() == 0)
  {
    f_cDeltaV = Eigen::VectorXd::Zero(preSolveV.size());
  }
  else
  {
    f_cDeltaV
        = Minv * A_c_ub_E * f_c + classicPtr->getVelocityDueToIllegalImpulses();
  }
  Eigen::VectorXd realF_cDeltaV = Minv * A_c_ub_E * realImpulses;
  Eigen::VectorXd postSolveV = preSolveV + f_cDeltaV;

  /*
std::cout << "Real f_c delta V:" << std::endl << realF_cDeltaV << std::endl;
std::cout << "Analytical f_c delta V:" << std::endl << f_cDeltaV << std::endl;
*/

  VelocityTest test;
  test.predictedNextVel = postSolveV;
  test.predictedNextVelDeltaVFromF = f_cDeltaV;
  test.predictedNextVelPreSolve = preSolveV;
  test.realNextVel = realNextVel;
  test.realNextVelDeltaVFromF = realNextVelDeltaVFromF;
  test.realNextVelPreSolve = realNextVelPreSolve;
  test.preStepVelocity = preStepVelocity;

  return test;
}

bool verifyNextV(WorldPtr world)
{
  Eigen::VectorXd positions = world->getPositions();
  RestorableSnapshot snapshot(world);

  bool oldPenetrationCorrectionEnabled
      = world->getConstraintSolver()->getPenetrationCorrectionEnabled();
  world->getConstraintSolver()->setPenetrationCorrectionEnabled(false);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  // VelocityTest originalTest = runVelocityTest(world);

  const double EPSILON = 1e-3;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    Eigen::VectorXd tweakedPos = Eigen::VectorXd(positions);
    tweakedPos(i) += EPSILON;

    // snapshot.restore();
    world->setPositions(tweakedPos);

    VelocityTest perturbedTest = runVelocityTest(world);

    if (!equals(
            perturbedTest.predictedNextVel,
            perturbedTest.realNextVel,
            classicPtr->hasBounces()
                ? 1e-4 // things get sloppy when bouncing, increase tol
                : 1e-8))
    {
      std::cout << "Real v_t+1:" << std::endl
                << perturbedTest.realNextVel << std::endl;
      std::cout << "Analytical v_t+1:" << std::endl
                << perturbedTest.predictedNextVel << std::endl;
      std::cout << "Analytical pre-solve v_t+1:" << std::endl
                << perturbedTest.predictedNextVelPreSolve << std::endl;
      std::cout << "Real pre-solve v_t+1:" << std::endl
                << perturbedTest.realNextVelPreSolve << std::endl;
      std::cout << "Analytical delta V from f_c v_t+1:" << std::endl
                << perturbedTest.predictedNextVelDeltaVFromF << std::endl;
      std::cout << "Real delta V from f_c v_t+1:" << std::endl
                << perturbedTest.realNextVelDeltaVFromF << std::endl;
      std::cout << "Diff:" << std::endl
                << (perturbedTest.realNextVelDeltaVFromF
                    - perturbedTest.predictedNextVelDeltaVFromF)
                << std::endl;
      return false;
    }
  }

  world->getConstraintSolver()->setPenetrationCorrectionEnabled(
      oldPenetrationCorrectionEnabled);

  snapshot.restore();
  return true;
}

bool verifyScratch(WorldPtr world)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXd analytical = classicPtr->getScratchAnalytical(world);
  MatrixXd bruteForce = classicPtr->getScratchFiniteDifference(world);

  if (!equals(analytical, bruteForce, 1e-8) || true)
  {
    std::cout << "Brute force Scratch Jacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical Scratch Jacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    return false;
  }

  return true;
}

Eigen::Vector6d getLinearScratchScrew()
{
  Eigen::Vector6d linearScratchScrew;
  linearScratchScrew << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  return linearScratchScrew;
}

Eigen::Vector6d getLinearScratchForce()
{
  Eigen::Vector6d linearScratchForce;
  linearScratchForce << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  return linearScratchForce;
}

double linearScratch(double input)
{
  Eigen::Isometry3d transform = math::expMap(getLinearScratchScrew() * input);
  Eigen::Vector6d forceInFrame
      = math::dAdInvT(transform, getLinearScratchForce());
  std::cout << "Force in frame: " << std::endl << forceInFrame << std::endl;
  std::cout << "screw: " << std::endl << getLinearScratchScrew() << std::endl;
  std::cout << "dotted with screw: " << std::endl
            << forceInFrame.dot(getLinearScratchScrew()) << std::endl;
  return forceInFrame.dot(getLinearScratchScrew());
}

double bruteForceLinearScratch(double startingPoint)
{
  const double EPS = 1e-6;
  return (linearScratch(startingPoint + EPS) - linearScratch(startingPoint))
         / EPS;
}

double analyticalLinearScratch(double point)
{
  return 1.0;
}

bool verifyLinearScratch()
{
  double point = 0.76;
  double bruteForce = bruteForceLinearScratch(point);
  double analytical = analyticalLinearScratch(point);
  if (abs(bruteForce - analytical) > 1e-12)
  {
    std::cout << "Brute force linear scratch:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical linear scratch (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }
  return true;
}

bool verifyJacobianOfProjectionIntoClampsMatrix(
    WorldPtr world, VectorXd proposedVelocities, WithRespectTo wrt)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXd analytical = classicPtr->getJacobianOfProjectionIntoClampsMatrix(
      world, proposedVelocities * 10, wrt);
  MatrixXd bruteForce
      = classicPtr->finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
          world, proposedVelocities * 10, wrt);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force P_c Jacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical P_c Jacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyPosVelJacobian(WorldPtr world, VectorXd proposedVelocities)
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

  MatrixXd analytical = classicPtr->getPosVelJacobian(world);
  MatrixXd bruteForce = classicPtr->finiteDifferencePosVelJacobian(world);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force posVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical posVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
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

  if (!equals(analytical, bruteForce, 1e-8))
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

bool verifyRecoveredLCPConstraints(WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);
  world->getConstraintSolver()->setPenetrationCorrectionEnabled(false);
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (classicPtr->mGradientMatrices.size() > 1)
    return true;

  MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);

  if (A_c.cols() == 0)
    return true;

  MatrixXd Q = Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  classicPtr->computeLCPConstraintMatrixClampingSubset(world, Q, A_c);
  Eigen::MatrixXd realQ = classicPtr->getClampingAMatrix();

  Eigen::VectorXd b = Eigen::VectorXd::Zero(A_c.cols());
  classicPtr->computeLCPOffsetClampingSubset(world, b, A_c);
  Eigen::VectorXd realB = classicPtr->getClampingConstraintRelativeVels();
  /* + (A_c.completeOrthogonalDecomposition().solve(
      classicPtr->getVelocityDueToIllegalImpulses())); */
  Eigen::VectorXd X = Q.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXd realX = classicPtr->getClampingConstraintImpulses();

  Eigen::VectorXd partRealX
      = realQ.completeOrthogonalDecomposition().solve(realB);

  if (!equals(b, realB, 1e-8))
  {
    std::cout << "Error in verifyRecoveredLCPConstraints():" << std::endl;
    std::cout << "analytical B:" << std::endl << b << std::endl;
    std::cout << "real B:" << std::endl << realB << std::endl;
    std::cout << "vel due to illegal impulses:" << std::endl
              << classicPtr->getVelocityDueToIllegalImpulses() << std::endl;
    return false;
  }

  if (!equals(Q, realQ, 1e-8))
  {
    std::cout << "Error in verifyRecoveredLCPConstraints():" << std::endl;
    std::cout << "analytical Q:" << std::endl << Q << std::endl;
    std::cout << "real Q:" << std::endl << realQ << std::endl;
    std::cout << "diff:" << std::endl << Q - realQ << std::endl;
    return false;
  }

  if (!equals(X, realX, 1e-8))
  {
    std::cout << "Error in verifyRecoveredLCPConstraints():" << std::endl;
    std::cout << "analytical X:" << std::endl << X << std::endl;
    std::cout << "real X:" << std::endl << realX << std::endl;
    std::cout << "diff:" << std::endl << X - realX << std::endl;
    std::cout << "part real X:" << std::endl << partRealX << std::endl;
    std::cout << "diff:" << std::endl << X - partRealX << std::endl;
    return false;
  }

  return true;
}

bool verifyVelGradients(WorldPtr world, VectorXd worldVel)
{
  // return verifyScratch(world);
  // return verifyF_c(world);
  // return verifyLinearScratch();
  // return verifyNextV(world);
  return (
      verifyClassicClampingConstraintMatrix(world, worldVel)
      && verifyMassedClampingConstraintMatrix(world, worldVel)
      // && verifyScratch(world)
      && verifyMassedUpperBoundConstraintMatrix(world, worldVel)
      // These no longer matter, because we bypass P_c in the new formulation
      // && verifyClassicProjectionIntoClampsMatrix(world, worldVel)
      // && verifyMassedProjectionIntoClampsMatrix(world, worldVel)
      // && verifyJacobianOfProjectionIntoClampsMatrix(world, worldVel,
      // POSITION)
      && verifyRecoveredLCPConstraints(world, worldVel) && verifyF_c(world)
      && verifyVelVelJacobian(world, worldVel)
      && verifyPosVelJacobian(world, worldVel) && verifyNextV(world)
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

  RestorableSnapshot snapshot(world);

  world->setPositions(classicPtr->getPreStepPosition());
  world->setVelocities(classicPtr->getPreStepVelocity());

  /*
  std::cout << "Pre time step position: " << std::endl
            << classicPtr->getPreStepPosition() << std::endl;
  std::cout << "Post time step position: " << std::endl
            << classicPtr->getPostStepPosition() << std::endl;
  std::cout << "Pre time step velocity: " << std::endl
            << classicPtr->getPreStepVelocity() << std::endl;
  std::cout << "Post time step velocity: " << std::endl
            << classicPtr->getPostStepVelocity() << std::endl;
  */

  // Compute "brute force" backprop using full Jacobians

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
      // v_t --> p_t+1
      + (classicPtr->getVelPosJacobian(world).transpose()
         * nextTimeStep.lossWrtPosition);

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

    if (!equals(lossWrtThisPosition, thisTimeStep.lossWrtPosition, 1e-5))
    {
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
    }

    if (!equals(lossWrtThisVelocity, thisTimeStep.lossWrtVelocity, 1e-5))
    {
      std::cout << "-----" << std::endl;

      Eigen::MatrixXd velVelJac = classicPtr->getVelVelJacobian(world);

      Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
      Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
      Eigen::MatrixXd V_c
          = classicPtr->getMassedClampingConstraintMatrix(world);
      Eigen::MatrixXd V_ub
          = classicPtr->getMassedUpperBoundConstraintMatrix(world);
      Eigen::MatrixXd B = classicPtr->getBounceDiagonals().asDiagonal();
      Eigen::MatrixXd E = classicPtr->getUpperBoundMappingMatrix();
      Eigen::MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix(world);
      Eigen::MatrixXd Minv = classicPtr->getInvMassMatrix(world);
      Eigen::MatrixXd parts1 = A_c + A_ub * E;
      Eigen::MatrixXd parts2 = world->getTimeStep() * Minv * parts1 * P_c;

      std::cout << "Brute force A_c*z:" << std::endl
                << parts2.transpose() * nextTimeStep.lossWrtVelocity
                << std::endl;

      // Classic formulation

      Eigen::MatrixXd classicInnerPart
          = A_c.transpose().eval() * Minv * (A_c + A_ub * E);
      Eigen::MatrixXd classicInnerPartInv
          = classicInnerPart.completeOrthogonalDecomposition().pseudoInverse();
      Eigen::MatrixXd classicRightPart = B * A_c.transpose().eval();
      Eigen::MatrixXd classicLeftPart = Minv * (A_c + A_ub * E);
      Eigen::MatrixXd classicComplete
          = classicLeftPart * classicInnerPart * classicRightPart;

      std::cout << "Classic brute force A_c*z:" << std::endl
                << classicComplete.transpose() * nextTimeStep.lossWrtVelocity
                << std::endl;

      // Massed formulation

      Eigen::MatrixXd massedInnerPart
          = A_c.transpose().eval() * (V_c + V_ub * E);
      Eigen::MatrixXd massedInnerPartInv
          = massedInnerPart.completeOrthogonalDecomposition().pseudoInverse();
      Eigen::MatrixXd massedRightPart = B * A_c.transpose().eval();
      Eigen::MatrixXd massedLeftPart = V_c + V_ub * E;
      Eigen::MatrixXd massedComplete
          = massedLeftPart * massedInnerPart * massedRightPart;

      std::cout << "Massed brute force A_c*z:" << std::endl
                << massedComplete.transpose() * nextTimeStep.lossWrtVelocity
                << std::endl;

      if (!equals(massedInnerPart, classicInnerPart, 1e-8))
      {
        std::cout << "Mismatch at inner part!" << std::endl;
        std::cout << "Classic inner part:" << std::endl
                  << classicInnerPart << std::endl;
        std::cout << "Massed inner part:" << std::endl
                  << massedInnerPart << std::endl;
      }
      if (!equals(massedInnerPartInv, classicInnerPartInv, 1e-8))
      {
        std::cout << "Mismatch at inner part inv!" << std::endl;
        std::cout << "Classic inner part inv:" << std::endl
                  << classicInnerPartInv << std::endl;
        std::cout << "Massed inner part inv:" << std::endl
                  << massedInnerPartInv << std::endl;
      }
      if (!equals(massedLeftPart, classicLeftPart, 1e-8))
      {
        std::cout << "Mismatch at left part!" << std::endl;
        std::cout << "Classic left part:" << std::endl
                  << classicLeftPart << std::endl;
        std::cout << "Massed left part:" << std::endl
                  << massedLeftPart << std::endl;
      }
      if (!equals(massedRightPart, classicRightPart, 1e-8))
      {
        std::cout << "Mismatch at right part!" << std::endl;
        std::cout << "Classic right part:" << std::endl
                  << classicRightPart << std::endl;
        std::cout << "Massed right part:" << std::endl
                  << massedRightPart << std::endl;
      }
      Eigen::MatrixXd V_c_recovered = Minv * A_c;
      if (!equals(V_c_recovered, V_c, 1e-8))
      {
        std::cout << "Mismatch at V_c == Minv * A_c!" << std::endl;
        std::cout << "V_c:" << std::endl << V_c << std::endl;
        std::cout << "A_c:" << std::endl << A_c << std::endl;
        std::cout << "Minv:" << std::endl << Minv << std::endl;
        std::cout << "Minv * A_c:" << std::endl << V_c_recovered << std::endl;
      }
      Eigen::MatrixXd V_ub_recovered = Minv * A_ub;
      if (!equals(V_ub_recovered, V_ub, 1e-8))
      {
        std::cout << "Mismatch at V_ub == Minv * A_ub!" << std::endl;
        std::cout << "V_ub:" << std::endl << V_ub << std::endl;
        std::cout << "Minv * A_ub:" << std::endl << V_ub_recovered << std::endl;
      }

      /*
      std::cout << "vel-vel Jacobian:" << std::endl << velVelJac << std::endl;
      std::cout << "vel-pos Jacobian:" << std::endl
                << classicPtr->getVelPosJacobian(world) << std::endl;
      std::cout << "vel-C Jacobian:" << std::endl
                << classicPtr->getVelCJacobian(world) << std::endl;
      std::cout << "1: nextLossWrtVel:" << std::endl
                << nextTimeStep.lossWrtVelocity << std::endl;
      std::cout << "2: Intermediate:" << std::endl
                << -parts2.transpose() * nextTimeStep.lossWrtVelocity
                << std::endl;
      */
      std::cout << "2.5: (force-vel)^T * nextLossWrtVel:" << std::endl
                << -classicPtr->getForceVelJacobian(world).transpose()
                       * nextTimeStep.lossWrtVelocity
                << std::endl;
      std::cout << "3: -((force-vel) * (vel-C))^T * nextLossWrtVel:"
                << std::endl
                << -(classicPtr->getForceVelJacobian(world)
                     * classicPtr->getVelCJacobian(world))
                           .transpose()
                       * nextTimeStep.lossWrtVelocity
                << std::endl;
      /*
std::cout << "(v_t --> v_t+1) * v_t+1:" << std::endl
      << (velVelJac.transpose() * nextTimeStep.lossWrtVelocity)
      << std::endl;
std::cout << "v_t --> p_t+1:" << std::endl
      << (classicPtr->getVelPosJacobian(world).transpose()
          * lossWrtThisPosition)
      << std::endl;
      */

      std::cout << "Brute force: loss wrt velocity at time t:" << std::endl
                << lossWrtThisVelocity << std::endl;
      std::cout << "Analytical: loss wrt velocity at time t:" << std::endl
                << thisTimeStep.lossWrtVelocity << std::endl;
    }

    if (!equals(lossWrtThisTorque, thisTimeStep.lossWrtTorque, 1e-5))
    {
      std::cout << "-----" << std::endl;

      std::cout << "Brute force: loss wrt torque at time t:" << std::endl
                << lossWrtThisTorque << std::endl;
      std::cout << "Analytical: loss wrt torque at time t:" << std::endl
                << thisTimeStep.lossWrtTorque << std::endl;
      std::cout << "(f_t --> v_t+1)^T:" << std::endl
                << (classicPtr->getForceVelJacobian(world).transpose())
                << std::endl;
      std::cout << "MInv:" << std::endl
                << classicPtr->getInvMassMatrix(world) << std::endl;
      std::cout << "v_t+1:" << std::endl
                << nextTimeStep.lossWrtVelocity << std::endl;
      std::cout << "MInv * v_t+1:" << std::endl
                << (classicPtr->getInvMassMatrix(world))
                       * nextTimeStep.lossWrtVelocity
                << std::endl;
    }
    return false;
  }

  snapshot.restore();

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
  for (int i = (world->getNumDofs() * 2) - 1; i >= 0; i--)
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
      world->step(true, false);
    grad.lossWrtPosition(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXd tweakedVel = originalVel;
    tweakedVel(i) += EPSILON;

    snapshot.restore();
    world->setVelocities(tweakedVel);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step(true, false);
    grad.lossWrtVelocity(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXd tweakedForce = originalForce;
    tweakedForce(i) += EPSILON;

    snapshot.restore();
    world->setForces(tweakedForce);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step(true, false);
    grad.lossWrtTorque(i) = (loss(world) - defaultLoss) / EPSILON;
  }

  snapshot.restore();
  return grad;
}

bool verifyGradientBackprop(
    WorldPtr world, std::size_t timesteps, std::function<double(WorldPtr)> loss)
{
  RestorableSnapshot snapshot(world);

  std::vector<BackpropSnapshotPtr> backpropSnapshots;
  std::vector<RestorableSnapshot> restorableSnapshots;
  backpropSnapshots.reserve(timesteps);
  for (std::size_t i = 0; i < timesteps; i++)
  {
    restorableSnapshots.push_back(RestorableSnapshot(world));
    backpropSnapshots.push_back(forwardPass(world, false, false));
  }

  // Get the loss gradient at the final timestep (by brute force) to initialize
  // an analytical backwards pass
  LossGradient analytical = computeBruteForceGradient(world, 0, loss);

  LossGradient bruteForce = analytical;

  snapshot.restore();
  for (int i = timesteps - 1; i >= 0; i--)
  {
    LossGradient thisTimestep;
    backpropSnapshots[i]->backprop(world, thisTimestep, analytical);
    analytical = thisTimestep;

    int numSteps = timesteps - i;
    restorableSnapshots[i].restore();
    LossGradient bruteForceThisTimestep
        = computeBruteForceGradient(world, numSteps, loss);

    // p_t+1 <-- p_t
    Eigen::MatrixXd posPos = backpropSnapshots[i]->getPosPosJacobian(world);
    // v_t+1 <-- p_t
    Eigen::MatrixXd posVel = backpropSnapshots[i]->getPosVelJacobian(world);
    // p_t+1 <-- v_t
    Eigen::MatrixXd velPos = backpropSnapshots[i]->getVelPosJacobian(world);
    // v_t+1 <-- v_t
    Eigen::MatrixXd velVel = backpropSnapshots[i]->getVelVelJacobian(world);

    // p_t+1 <-- p_t
    Eigen::MatrixXd posPosFD
        = backpropSnapshots[i]->finiteDifferencePosPosJacobian(world, 1);
    // v_t+1 <-- p_t
    Eigen::MatrixXd posVelFD
        = backpropSnapshots[i]->finiteDifferencePosVelJacobian(world);
    // p_t+1 <-- v_t
    Eigen::MatrixXd velPosFD
        = backpropSnapshots[i]->finiteDifferenceVelPosJacobian(world, 1);
    // v_t+1 <-- v_t
    Eigen::MatrixXd velVelFD
        = backpropSnapshots[i]->finiteDifferenceVelVelJacobian(world);

    double diffPosPos = (posPos - posPosFD).norm();
    double diffPosVel = (posVel - posVelFD).norm();
    double diffVelPos = (velPos - velPosFD).norm();
    double diffVelVel = (velVel - velVelFD).norm();

    std::cout << "Jacobian error at step:" << numSteps << ": " << diffPosPos
              << ", " << diffPosVel << ", " << diffVelPos << ", " << diffVelVel
              << std::endl;

    LossGradient analyticalWithBruteForce;
    analyticalWithBruteForce.lossWrtPosition
        = posPos.transpose() * bruteForce.lossWrtPosition
          + posVel.transpose() * bruteForce.lossWrtVelocity;
    analyticalWithBruteForce.lossWrtVelocity
        = velPos.transpose() * bruteForce.lossWrtPosition
          + velVel.transpose() * bruteForce.lossWrtVelocity;

    bruteForce = bruteForceThisTimestep;

    std::cout
        << "Backprop error at step:" << numSteps << ": "
        << (analytical.lossWrtPosition - bruteForce.lossWrtPosition).norm()
        << ", "
        << (analytical.lossWrtVelocity - bruteForce.lossWrtVelocity).norm()
        << ", " << (analytical.lossWrtTorque - bruteForce.lossWrtTorque).norm()
        << std::endl;

    // Assert that the results are the same
    if (!equals(analytical.lossWrtPosition, bruteForce.lossWrtPosition, 1e-8)
        || !equals(analytical.lossWrtVelocity, bruteForce.lossWrtVelocity, 1e-8)
        || !equals(analytical.lossWrtTorque, bruteForce.lossWrtTorque, 1e-8))
    {
      std::cout << "Diverged at backprop steps:" << numSteps << std::endl;
      std::cout << "Analytical loss wrt position:" << std::endl
                << analytical.lossWrtPosition << std::endl;
      std::cout << "Brute force loss wrt position:" << std::endl
                << bruteForce.lossWrtPosition << std::endl;
      std::cout << "Analytical off Brute force loss wrt position:" << std::endl
                << analyticalWithBruteForce.lossWrtPosition << std::endl;
      std::cout << "Diff loss gradient wrt position:" << std::endl
                << bruteForce.lossWrtPosition - analytical.lossWrtPosition
                << std::endl;
      std::cout << "Diff analytical loss gradient wrt position:" << std::endl
                << bruteForce.lossWrtPosition
                       - analyticalWithBruteForce.lossWrtPosition
                << std::endl;
      std::cout << "Analytical loss wrt velocity:" << std::endl
                << analytical.lossWrtVelocity << std::endl;
      std::cout << "Brute force loss wrt velocity:" << std::endl
                << bruteForce.lossWrtVelocity << std::endl;
      std::cout << "Analytical off Brute force loss wrt velocity:" << std::endl
                << analyticalWithBruteForce.lossWrtVelocity << std::endl;
      std::cout << "Diff loss gradient wrt velocity:" << std::endl
                << bruteForce.lossWrtVelocity - analytical.lossWrtVelocity
                << std::endl;
      std::cout << "Diff loss analytical off brute force gradient wrt velocity:"
                << std::endl
                << bruteForce.lossWrtVelocity
                       - analyticalWithBruteForce.lossWrtVelocity
                << std::endl;
      std::cout << "Diff analytical loss gradient wrt velocity:" << std::endl
                << bruteForce.lossWrtVelocity
                       - analyticalWithBruteForce.lossWrtVelocity
                << std::endl;
      std::cout << "Analytical loss wrt torque:" << std::endl
                << analytical.lossWrtTorque << std::endl;
      std::cout << "Brute force loss wrt torque:" << std::endl
                << bruteForce.lossWrtTorque << std::endl;
      std::cout << "Diff loss gradient wrt torque:" << std::endl
                << bruteForce.lossWrtTorque - analytical.lossWrtTorque
                << std::endl;
      return false;
    }
  }

  snapshot.restore();

  return true;
}

bool verifyBulkPass(
    WorldPtr world, std::size_t timesteps, std::size_t shootingLength)
{
  int dofs = world->getNumDofs();
  int numKnotPoses = floor(static_cast<double>(timesteps) / shootingLength);

  // Run a bulk forward pass

  Eigen::MatrixXd torques = Eigen::MatrixXd::Random(dofs, timesteps);
  Eigen::MatrixXd knotPoses = Eigen::MatrixXd::Random(dofs, numKnotPoses);
  Eigen::MatrixXd knotVels = Eigen::MatrixXd::Random(dofs, numKnotPoses);
  BulkForwardPassResult forward
      = bulkForwardPass(world, torques, shootingLength, knotPoses, knotVels);

  // Run a bulk backward pass

  Eigen::MatrixXd lossWrtPoses = Eigen::MatrixXd::Random(dofs, timesteps);
  Eigen::MatrixXd lossWrtVels = Eigen::MatrixXd::Random(dofs, timesteps);
  BulkBackwardPassResult backward = bulkBackwardPass(
      world, forward.snapshots, shootingLength, lossWrtPoses, lossWrtVels);

  // This passes if it just doesn't crash
  return true;
}

bool verifyWorldSpaceToVelocitySpatial(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXd worldVel = worldVelMatrix.col(0);

  Eigen::VectorXd bruteWorldVel = Eigen::VectorXd::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() << std::endl;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector6d bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity());
      Eigen::Vector6d analyticalVel = worldVel.segment(cursor, 6);
      /*
      std::cout << "Body " << k << std::endl << bruteVel << std::endl;
      std::cout << "Analytical " << k << std::endl
                << analyticalVel << std::endl;
                */
      bruteWorldVel.segment(cursor, 6) = bruteVel;
      cursor += 6;
    }
  }

  if (!equals(worldVel, bruteWorldVel))
  {
    std::cout << "convertJointSpaceVelocitiesToWorldPositions() failed!"
              << std::endl;
    std::cout << "Analytical world vel screws: " << std::endl
              << worldVel << std::endl;
    std::cout << "Brute world vel screws: " << std::endl
              << bruteWorldVel << std::endl;
    return false;
  }
  return true;
}

bool verifyWorldSpaceToLinearVelocity(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXd worldVel = worldVelMatrix.col(0);

  Eigen::VectorXd bruteWorldVel = Eigen::VectorXd::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() << std::endl;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector3d bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity())
                .tail<3>();
      Eigen::Vector3d analyticalVel = worldVel.segment(cursor, 3);

      /*
      std::cout << "Body " << k << std::endl << bruteVel << std::endl;
      std::cout << "Analytical " << k << std::endl
                << analyticalVel << std::endl;
      */

      bruteWorldVel.segment(cursor, 3) = bruteVel;
      cursor += 3;
    }
  }

  if (!equals(worldVel, bruteWorldVel))
  {
    std::cout << "convertJointSpaceVelocitiesToWorldPositions() failed!"
              << std::endl;
    std::cout << "Analytical world vel: " << std::endl << worldVel << std::endl;
    std::cout << "Brute world vel: " << std::endl << bruteWorldVel << std::endl;
    return false;
  }
  return true;
}

bool verifyWorldSpaceToPositionCOM(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldPosMatrix = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::COM_POS, false);
  if (worldPosMatrix.cols() != 1)
    return false;
  Eigen::VectorXd worldPos = worldPosMatrix.col(0);

  Eigen::VectorXd bruteWorldCOMPos = Eigen::VectorXd::Zero(worldPos.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() << std::endl;
    Eigen::Vector3d bruteCOMPos = skel->getCOM();

    Eigen::Vector3d analyticalVel = worldPos.segment(cursor, 3);
    /*
    std::cout << "Body " << k << std::endl << bruteVel << std::endl;
    std::cout << "Analytical " << k << std::endl
              << analyticalVel << std::endl;
              */
    bruteWorldCOMPos.segment(cursor, 3) = bruteCOMPos;
    cursor += 3;
  }

  if (!equals(worldPos, bruteWorldCOMPos))
  {
    std::cout << "convertJointSpaceVelocitiesToWorldCOM() failed!" << std::endl;
    std::cout << "Analytical world pos COM: " << std::endl
              << worldPos << std::endl;
    std::cout << "Brute world pos COM: " << std::endl
              << bruteWorldCOMPos << std::endl;
    return false;
  }
  return true;
}

bool verifyWorldSpaceToVelocityCOMLinear(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXd worldVel = worldVelMatrix.col(0);

  Eigen::VectorXd bruteWorldCOMVel = Eigen::VectorXd::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() << std::endl;
    Eigen::Vector3d bruteCOMVel = Eigen::Vector3d::Zero();
    double totalMass = 0.0;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector3d bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity())
                .tail<3>();
      bruteCOMVel += bruteVel * node->getMass();
      totalMass += node->getMass();
    }
    bruteCOMVel /= totalMass;

    Eigen::Vector3d analyticalVel = worldVel.segment(cursor, 3);
    /*
    std::cout << "Body " << k << std::endl << bruteVel << std::endl;
    std::cout << "Analytical " << k << std::endl
              << analyticalVel << std::endl;
              */
    bruteWorldCOMVel.segment(cursor, 3) = bruteCOMVel;
    cursor += 3;
  }

  if (!equals(worldVel, bruteWorldCOMVel))
  {
    std::cout << "convertJointSpaceVelocitiesToWorldCOM() failed!" << std::endl;
    std::cout << "Analytical world vel COM: " << std::endl
              << worldVel << std::endl;
    std::cout << "Brute world vel COM: " << std::endl
              << bruteWorldCOMVel << std::endl;
    return false;
  }
  return true;
}

bool verifyWorldSpaceToVelocityCOMSpatial(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXd worldVel = worldVelMatrix.col(0);

  Eigen::VectorXd bruteWorldCOMVel = Eigen::VectorXd::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() << std::endl;
    Eigen::Vector6d bruteCOMVel = Eigen::Vector6d::Zero();
    double totalMass = 0.0;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector6d bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity());
      bruteCOMVel += bruteVel * node->getMass();
      totalMass += node->getMass();
    }
    bruteCOMVel /= totalMass;

    Eigen::Vector6d analyticalVel = worldVel.segment(cursor, 6);
    /*
    std::cout << "Body " << k << std::endl << bruteVel << std::endl;
    std::cout << "Analytical " << k << std::endl
              << analyticalVel << std::endl;
              */
    bruteWorldCOMVel.segment(cursor, 6) = bruteCOMVel;
    cursor += 6;
  }

  if (!equals(worldVel, bruteWorldCOMVel))
  {
    std::cout << "convertJointSpaceVelocitiesToWorldCOM() failed!" << std::endl;
    std::cout << "Analytical world vel COM: " << std::endl
              << worldVel << std::endl;
    std::cout << "Brute world vel COM: " << std::endl
              << bruteWorldCOMVel << std::endl;
    return false;
  }
  return true;
}

bool verifyBackpropWorldSpacePositionToSpatial(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();

  // Delete the 2nd body node, arbitrarily, to force some shuffling
  bodyNodes.erase(bodyNodes.begin()++);
  // Shuffle the remaining elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  Eigen::VectorXd originalWorldPos = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::POS_SPATIAL, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(position.size()) * 1e-6;
  Eigen::VectorXd perturbedPos = position + perturbation;

  Eigen::VectorXd perturbedWorldPos = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::POS_SPATIAL, false);
  Eigen::MatrixXd skelSpatialJac
      = jointToWorldSpatialJacobian(world->getSkeleton(0), bodyNodes);
  Eigen::VectorXd expectedPerturbation = skelSpatialJac * perturbation;

  /*
  std::cout << "World perturbation: " << std::endl
            << worldPerturbation << std::endl;
  std::cout << "Expected perturbation: " << std::endl
            << expectedPerturbation << std::endl;
            */

  Eigen::VectorXd worldPerturbation = perturbedWorldPos - originalWorldPos;
  Eigen::VectorXd recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::POS_SPATIAL, true);

  if (!equals(perturbation, recoveredPerturbation, 1e-8))
  {
    std::cout << "backprop() POS_SPATIAL failed!" << std::endl;
    Eigen::MatrixXd skelSpatialJac2 = jointToWorldSpatialJacobian(
        world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
    Eigen::MatrixXd perturbations(worldPerturbation.size(), 2);
    perturbations << worldPerturbation, expectedPerturbation;
    std::cout << "World perturbation | expected perturbation: " << std::endl
              << perturbations << std::endl;
    std::cout << "Recovered perturbation: " << std::endl
              << recoveredPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }
  return true;
}

bool verifyBackpropWorldSpaceVelocityToSpatial(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();

  // Delete the 2nd body node, arbitrarily, to force some shuffling
  bodyNodes.erase(bodyNodes.begin()++);
  // Shuffle the remaining elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  Eigen::VectorXd originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(position.size()) * 1e-6;
  Eigen::VectorXd perturbedVel = velocity + perturbation;

  Eigen::VectorXd perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldVel - originalWorldVel;
  Eigen::VectorXd recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::VEL_SPATIAL, true);

  if (!equals(perturbation, recoveredPerturbation, 1e-8))
  {
    std::cout << "backprop() VEL_SPATIAL failed!" << std::endl;
    std::cout << "Original vel: " << std::endl << velocity << std::endl;
    std::cout << "Perturbed vel: " << std::endl << perturbedVel << std::endl;
    std::cout << "Original world vel: " << std::endl
              << originalWorldVel << std::endl;
    std::cout << "Perturbed world vel: " << std::endl
              << perturbedWorldVel << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Recovered perturbation: " << std::endl
              << recoveredPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }
  return true;
}

bool verifyLinearJacobian(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  world->setPositions(position);
  world->setVelocities(velocity);
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    std::vector<dynamics::BodyNode*> bodyNodes
        = world->getSkeleton(i)->getBodyNodes();
    // Shuffle the elements
    std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

    Eigen::MatrixXd analytical = jointToWorldLinearJacobian(skel, bodyNodes);

    // Compute a brute force version
    Eigen::VectorXd originalPos = skel->getPositions();
    Eigen::VectorXd originalVel = skel->getVelocities();
    Eigen::VectorXd originalWorldPos = skelConvertJointSpaceToWorldSpace(
        skel, originalPos, bodyNodes, ConvertToSpace::POS_LINEAR);
    Eigen::VectorXd originalWorldVel = skelConvertJointSpaceToWorldSpace(
        skel, originalVel, bodyNodes, ConvertToSpace::VEL_LINEAR);
    Eigen::MatrixXd bruteForce
        = Eigen::MatrixXd::Zero(analytical.rows(), analytical.cols());
    const double EPS = 1e-7;
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      Eigen::VectorXd perturbedPos = originalPos;
      perturbedPos(j) += EPS;
      Eigen::VectorXd perturbedVel = originalVel;
      perturbedVel(j) += EPS;
      Eigen::VectorXd posColumn
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedPos, bodyNodes, ConvertToSpace::POS_LINEAR)
             - originalWorldPos)
            / EPS;
      Eigen::VectorXd velColumn
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedVel, bodyNodes, ConvertToSpace::VEL_LINEAR)
             - originalWorldVel)
            / EPS;

      if (!equals(posColumn, velColumn, 1e-4))
      {
        std::cout << "Check your assumptions! Column " << j << " pos:\n"
                  << posColumn << "\nvel:\n"
                  << velColumn << "\nanalytical:\n"
                  << analytical.col(j) << "\n";
        // return false;
      }
      bruteForce.block(0, j, bruteForce.rows(), 1)
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedPos, bodyNodes, ConvertToSpace::POS_LINEAR)
             - originalWorldPos)
            / EPS;
    }

    if (!equals(bruteForce, analytical, 1e-6))
    {
      std::cout << "jointToWorldLinearJacobian() is wrong!" << std::endl;
      std::cout << "Analytical Jac: " << std::endl << analytical << std::endl;
      std::cout << "Brute force Jac: " << std::endl << bruteForce << std::endl;
      std::cout << "Diff: " << std::endl
                << (analytical - bruteForce) << std::endl;
      return false;
    }
  }
  return true;
}

bool verifySpatialJacobian(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  world->setPositions(position);
  world->setVelocities(velocity);
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    std::vector<dynamics::BodyNode*> bodyNodes
        = world->getSkeleton(i)->getBodyNodes();
    // Shuffle the elements
    std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

    Eigen::MatrixXd analytical = jointToWorldSpatialJacobian(skel, bodyNodes);

    // Compute a brute force version
    Eigen::VectorXd originalPos = skel->getPositions();
    Eigen::VectorXd originalVel = skel->getVelocities();
    Eigen::VectorXd originalWorldPos = skelConvertJointSpaceToWorldSpace(
        skel, originalPos, bodyNodes, ConvertToSpace::POS_SPATIAL);
    Eigen::VectorXd originalWorldVel = skelConvertJointSpaceToWorldSpace(
        skel, originalVel, bodyNodes, ConvertToSpace::VEL_SPATIAL);
    Eigen::MatrixXd bruteForce
        = Eigen::MatrixXd::Zero(analytical.rows(), analytical.cols());
    const double EPS = 1e-7;
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      Eigen::VectorXd perturbedPos = originalPos;
      perturbedPos(j) += EPS;
      Eigen::VectorXd perturbedVel = originalVel;
      perturbedVel(j) += EPS;
      Eigen::VectorXd posColumn
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedPos, bodyNodes, ConvertToSpace::POS_SPATIAL)
             - originalWorldPos)
            / EPS;
      Eigen::VectorXd velColumn
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedVel, bodyNodes, ConvertToSpace::VEL_SPATIAL)
             - originalWorldVel)
            / EPS;

      if (!equals(posColumn, velColumn, 1e-4))
      {
        std::cout << "Check your assumptions! Column " << j << " pos:\n"
                  << posColumn << "\nvel:\n"
                  << velColumn << "\nanalytical:\n"
                  << analytical.col(j) << "\n";
        // return false;
      }
      bruteForce.block(0, j, bruteForce.rows(), 1)
          = (skelConvertJointSpaceToWorldSpace(
                 skel, perturbedPos, bodyNodes, ConvertToSpace::POS_SPATIAL)
             - originalWorldPos)
            / EPS;
    }

    if (!equals(bruteForce, analytical, 1e-6))
    {
      std::cout << "jointToWorldSpatialJacobian() is wrong!" << std::endl;
      std::cout << "Analytical Jac: " << std::endl << analytical << std::endl;
      std::cout << "Brute force Jac: " << std::endl << bruteForce << std::endl;
      std::cout << "Diff: " << std::endl
                << (analytical - bruteForce) << std::endl;
      return false;
    }
  }
  return true;
}

bool verifyBackpropWorldSpacePositionToPosition(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXd originalWorldPos = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::POS_LINEAR, false);
  Eigen::VectorXd originalWorldSpatial = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::POS_SPATIAL, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(position.size()) * 1e-4;
  Eigen::VectorXd perturbedPos = position + perturbation;

  Eigen::MatrixXd skelLinearJac = jointToWorldLinearJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXd expectedPerturbation = skelLinearJac * perturbation;

  Eigen::MatrixXd skelSpatialJac = jointToWorldSpatialJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXd expectedPerturbationSpatial = skelSpatialJac * perturbation;

  Eigen::VectorXd perturbedWorldPos = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::POS_LINEAR, false);
  Eigen::VectorXd perturbedWorldSpatial = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::POS_SPATIAL, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldPos - originalWorldPos;
  Eigen::VectorXd worldPerturbationSpatial
      = perturbedWorldSpatial - originalWorldSpatial;

  Eigen::VectorXd recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::POS_LINEAR, true);

  Eigen::VectorXd expectedPerturbationFromSpatial
      = Eigen::VectorXd(expectedPerturbationSpatial.size() / 2);
  for (int i = 0; i < expectedPerturbationSpatial.size() / 6; i++)
  {
    /*
      Eigen::Vector3d bruteVel = math::expMap(math::AdR(
                                                  node->getWorldTransform(),
                                                  node->getSpatialVelocity()))
                                     .translation();
                                     */
    expectedPerturbationFromSpatial.segment(i * 3, 3)
        = math::expMap(expectedPerturbationSpatial.segment(i * 6, 6).eval())
              .translation();
  }

  Eigen::MatrixXd perturbations
      = Eigen::MatrixXd::Zero(worldPerturbation.size(), 3);
  perturbations << worldPerturbation, expectedPerturbation,
      expectedPerturbationFromSpatial;

  Eigen::MatrixXd perturbationsSpatial
      = Eigen::MatrixXd::Zero(worldPerturbationSpatial.size(), 2);
  perturbationsSpatial << worldPerturbationSpatial, expectedPerturbationSpatial;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5))
  {
    std::cout << "backprop() POS_LINEAR failed!" << std::endl;
    std::cout << "Original pos: " << std::endl << position << std::endl;
    std::cout << "Perturbed pos: " << std::endl << perturbedPos << std::endl;
    std::cout << "Original world pos: " << std::endl
              << originalWorldPos << std::endl;
    std::cout << "Perturbed world pos: " << std::endl
              << perturbedWorldPos << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Expected world perturbation: " << std::endl
              << expectedPerturbation << std::endl;
    std::cout << "World :: Expected pos perturbation :: Expected from spatial: "
              << std::endl
              << perturbations << std::endl;
    std::cout << "World :: Expected spatial perturbation: " << std::endl
              << perturbationsSpatial << std::endl;
    std::cout << "Recovered perturbation: " << std::endl
              << recoveredPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }

  return true;
}

bool verifyBackpropWorldSpaceVelocityToPosition(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXd originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  Eigen::VectorXd originalWorldSpatial = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(velocity.size()) * 1e-4;
  Eigen::VectorXd perturbedVel = velocity + perturbation;

  Eigen::MatrixXd skelLinearJac = jointToWorldLinearJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXd expectedPerturbation = skelLinearJac * perturbation;

  Eigen::MatrixXd skelSpatialJac = jointToWorldSpatialJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXd expectedPerturbationSpatial = skelSpatialJac * perturbation;

  Eigen::VectorXd perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  Eigen::VectorXd perturbedWorldSpatial = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldVel - originalWorldVel;
  Eigen::VectorXd worldPerturbationSpatial
      = perturbedWorldSpatial - originalWorldSpatial;

  Eigen::VectorXd recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::VEL_LINEAR, true);

  Eigen::VectorXd expectedPerturbationFromSpatial
      = Eigen::VectorXd(expectedPerturbationSpatial.size() / 2);
  for (int i = 0; i < expectedPerturbationSpatial.size() / 6; i++)
  {
    expectedPerturbationFromSpatial.segment(i * 3, 3)
        = math::expMap(expectedPerturbationSpatial.segment(i * 6, 6).eval())
              .translation();
  }

  Eigen::MatrixXd perturbations
      = Eigen::MatrixXd::Zero(worldPerturbation.size(), 3);
  perturbations << worldPerturbation, expectedPerturbation,
      expectedPerturbationFromSpatial;

  Eigen::MatrixXd perturbationsSpatial
      = Eigen::MatrixXd::Zero(worldPerturbationSpatial.size(), 2);
  perturbationsSpatial << worldPerturbationSpatial, expectedPerturbationSpatial;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5))
  {
    std::cout << "backprop() VEL_LINEAR failed!" << std::endl;
    std::cout << "Original vel: " << std::endl << velocity << std::endl;
    std::cout << "Perturbed vel: " << std::endl << perturbedVel << std::endl;
    std::cout << "Original world vel: " << std::endl
              << originalWorldVel << std::endl;
    std::cout << "Perturbed world vel: " << std::endl
              << perturbedWorldVel << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Expected world perturbation: " << std::endl
              << expectedPerturbation << std::endl;
    std::cout << "World :: Expected vel perturbation :: Expected from spatial: "
              << std::endl
              << perturbations << std::endl;
    std::cout << "World :: Expected spatial perturbation: " << std::endl
              << perturbationsSpatial << std::endl;
    std::cout << "Recovered perturbation: " << std::endl
              << recoveredPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }

  return true;
}

bool verifyBackpropWorldSpacePositionToCOM(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXd originalWorldPos = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::COM_POS, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(velocity.size()) * 1e-4;
  Eigen::VectorXd perturbedPos = position + perturbation;

  Eigen::MatrixXd skelLinearJac = world->getSkeleton(0)->getCOMLinearJacobian();
  Eigen::VectorXd expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXd perturbedWorldPos = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::COM_POS, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldPos - originalWorldPos;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5))
  {
    std::cout << "backpropWorldPositionsToCOM() failed!" << std::endl;
    std::cout << "Original pos: " << std::endl << position << std::endl;
    std::cout << "Perturbed pos: " << std::endl << perturbedPos << std::endl;
    std::cout << "Original world pos: " << std::endl
              << originalWorldPos << std::endl;
    std::cout << "Perturbed world pos: " << std::endl
              << perturbedWorldPos << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Expected world perturbation: " << std::endl
              << expectedPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }

  return true;
}

bool verifyBackpropWorldSpaceVelocityToCOMLinear(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXd originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(velocity.size()) * 1e-4;
  Eigen::VectorXd perturbedVel = velocity + perturbation;

  Eigen::MatrixXd skelLinearJac = world->getSkeleton(0)->getCOMLinearJacobian();
  Eigen::VectorXd expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXd perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldVel - originalWorldVel;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5))
  {
    std::cout << "backpropWorldVelocityToCOM() failed!" << std::endl;
    std::cout << "Original vel: " << std::endl << velocity << std::endl;
    std::cout << "Perturbed vel: " << std::endl << perturbedVel << std::endl;
    std::cout << "Original world vel: " << std::endl
              << originalWorldVel << std::endl;
    std::cout << "Perturbed world vel: " << std::endl
              << perturbedWorldVel << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Expected world perturbation: " << std::endl
              << expectedPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }

  return true;
}

bool verifyBackpropWorldSpaceVelocityToCOMSpatial(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXd originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);

  Eigen::VectorXd perturbation
      = Eigen::VectorXd::Random(velocity.size()) * 1e-4;
  Eigen::VectorXd perturbedVel = velocity + perturbation;

  Eigen::MatrixXd skelLinearJac = world->getSkeleton(0)->getCOMJacobian();
  Eigen::VectorXd expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXd perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);

  Eigen::VectorXd worldPerturbation = perturbedWorldVel - originalWorldVel;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5))
  {
    std::cout << "backpropWorldVelocityToCOM() failed!" << std::endl;
    std::cout << "Original vel: " << std::endl << velocity << std::endl;
    std::cout << "Perturbed vel: " << std::endl << perturbedVel << std::endl;
    std::cout << "Original world vel: " << std::endl
              << originalWorldVel << std::endl;
    std::cout << "Perturbed world vel: " << std::endl
              << perturbedWorldVel << std::endl;
    std::cout << "World perturbation: " << std::endl
              << worldPerturbation << std::endl;
    std::cout << "Expected world perturbation: " << std::endl
              << expectedPerturbation << std::endl;
    std::cout << "Original perturbation: " << std::endl
              << perturbation << std::endl;
    return false;
  }

  return true;
}

bool verifyWorldSpaceTransformInstance(
    WorldPtr world, Eigen::VectorXd position, Eigen::VectorXd velocity)
{
  if (!verifyLinearJacobian(world, position, velocity))
    return false;
  if (!verifySpatialJacobian(world, position, velocity))
    return false;
  if (!verifyWorldSpaceToVelocitySpatial(world, position, velocity))
    return false;
  if (!verifyWorldSpaceToPositionCOM(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpacePositionToSpatial(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpaceVelocityToSpatial(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpacePositionToPosition(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpaceVelocityToPosition(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpaceVelocityToCOMLinear(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpaceVelocityToCOMSpatial(world, position, velocity))
    return false;
  if (!verifyBackpropWorldSpacePositionToCOM(world, position, velocity))
    return false;
  if (!verifyWorldSpaceToLinearVelocity(world, position, velocity))
    return false;
  if (!verifyWorldSpaceToVelocityCOMLinear(world, position, velocity))
    return false;
  if (!verifyWorldSpaceToVelocityCOMSpatial(world, position, velocity))
    return false;
  return true;
}

bool verifyWorldSpaceTransform(WorldPtr world)
{
  int timesteps = 7;
  Eigen::MatrixXd jointPoses
      = Eigen::MatrixXd::Random(world->getNumDofs(), timesteps);
  Eigen::MatrixXd jointVels
      = Eigen::MatrixXd::Random(world->getNumDofs(), timesteps);

  for (int i = 0; i < timesteps; i++)
  {
    if (!verifyWorldSpaceTransformInstance(
            world, jointPoses.col(i), jointVels.col(i)))
      return false;
  }

  // Verify that nothing crashes when we run a batch
  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXd worldPos = convertJointSpaceToWorldSpace(
      world, jointPoses, bodyNodes, ConvertToSpace::POS_LINEAR, false);
  Eigen::MatrixXd worldVel = convertJointSpaceToWorldSpace(
      world, jointVels, bodyNodes, ConvertToSpace::VEL_LINEAR, false);

  Eigen::MatrixXd backprop = convertJointSpaceToWorldSpace(
      world, worldPos * 5, bodyNodes, ConvertToSpace::POS_LINEAR, false, true);

  return true;
}

bool verifyAnalyticalA_c(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix(world);
  for (int i = 0; i < classicPtr->getNumClamping(); i++)
  {
    Eigen::VectorXd trueCol = A_c.col(i);
    Eigen::VectorXd analyticalCol = constraints[i]->getConstraintForces(world);
    if (!equals(trueCol, analyticalCol, 1e-8))
    {
      std::cout << "True A_c col: " << std::endl << trueCol << std::endl;
      std::cout << "Analytical A_c col: " << std::endl
                << analyticalCol << std::endl;
      return false;
    }
  }

  return true;
}

bool verifyAnalyticalContactPositionJacobians(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();

  for (int i = 0; i < constraints.size(); i++)
  {
    math::LinearJacobian analyticalJac
        = constraints[i]->getContactPositionJacobian(world);
    math::LinearJacobian bruteForceJac
        = constraints[i]->bruteForceContactPositionJacobian(world);

    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Analytical Contact Pos Jac:" << std::endl
                << analyticalJac << std::endl;
      std::cout << "Brute Force Contact Pos Jac:" << std::endl
                << bruteForceJac << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalJac - bruteForceJac << std::endl;
      return false;
    }
  }

  return true;
}

bool verifyAnalyticalContactNormalJacobians(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();

  for (int i = 0; i < constraints.size(); i++)
  {
    math::LinearJacobian analyticalJac
        = constraints[i]->getContactForceDirectionJacobian(world);
    math::LinearJacobian bruteForceJac
        = constraints[i]->bruteForceContactForceDirectionJacobian(world);

    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Analytical Contact Force Direction Jac:" << std::endl
                << analyticalJac << std::endl;
      std::cout << "Brute Force Contact Force Direction Jac:" << std::endl
                << bruteForceJac << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalJac - bruteForceJac << std::endl;
      return false;
    }
  }

  return true;
}

bool verifyAnalyticalContactForceJacobians(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();

  for (int i = 0; i < constraints.size(); i++)
  {
    math::Jacobian analyticalJac
        = constraints[i]->getContactForceJacobian(world);
    math::Jacobian bruteForceJac
        = constraints[i]->bruteForceContactForceJacobian(world);

    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Analytical Contact Force Jac:" << std::endl
                << analyticalJac << std::endl;
      std::cout << "Brute Force Contact Force Jac:" << std::endl
                << bruteForceJac << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalJac - bruteForceJac << std::endl;
      return false;
    }
  }

  return true;
}

bool equals(EdgeData e1, EdgeData e2, double threshold)
{
  return equals(e1.edgeAPos, e2.edgeAPos, threshold)
         && equals(e1.edgeADir, e2.edgeADir, threshold)
         && equals(e1.edgeBPos, e2.edgeBPos, threshold)
         && equals(e1.edgeBDir, e2.edgeBDir, threshold);
}

/// Looks for any edge-edge contacts, and makes sure our analytical models of
/// them are correct
bool verifyPerturbedContactEdges(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const double EPS = 1e-7;
  for (int k = 0; k < constraints.size(); k++)
  {
    if (constraints[k]->getContactType() == EDGE_EDGE)
    {
      EdgeData original = constraints[k]->getEdges();
      Eigen::Vector3d originalIntersectionPoint = math::getContactPoint(
          original.edgeAPos,
          original.edgeADir,
          original.edgeBPos,
          original.edgeBDir);

      for (int i = 0; i < world->getNumSkeletons(); i++)
      {
        auto skel = world->getSkeleton(i);
        for (int j = 0; j < skel->getNumDofs(); j++)
        {
          EdgeData bruteForce
              = constraints[k]->bruteForceEdges(world, skel, j, EPS);
          EdgeData analytical
              = constraints[k]->estimatePerturbedEdges(skel, j, EPS);

          Eigen::Vector3d bruteIntersection = math::getContactPoint(
              bruteForce.edgeAPos,
              bruteForce.edgeADir,
              bruteForce.edgeBPos,
              bruteForce.edgeBDir);
          Eigen::Vector3d analyticalIntersection = math::getContactPoint(
              bruteForce.edgeAPos,
              bruteForce.edgeADir,
              bruteForce.edgeBPos,
              bruteForce.edgeBDir);

          double estimateThreshold = 1e-8;

          // Check the intersection point first, because the actual input points
          // can be different if the collision detector decided to use a
          // different vertex on either of the edges, which will screw
          // everything up. Even if it does this, though, the intersection
          // points will remain unchanged, so check those first.
          if (!equals(
                  bruteIntersection, analyticalIntersection, estimateThreshold))
          {
            std::cout << "Got intersection wrong!" << std::endl;
            std::cout << "Skel:" << std::endl
                      << skel->getName() << " - " << j << std::endl;
            std::cout << "Contact Type:" << std::endl
                      << constraints[k]->getDofContactType(skel->getDof(j))
                      << std::endl;
            std::cout << "Brute force intersection:" << std::endl
                      << bruteIntersection << std::endl;
            std::cout << "Analytical intersection:" << std::endl
                      << analyticalIntersection << std::endl;

            // Only check the actual edge parameters if the intersections are
            // materially different, because it is valid to have the edges pick
            // different corners when finding a collision.
            if (!equals(bruteForce, analytical, estimateThreshold))
            {
              std::cout << "Got edge wrong!" << std::endl;
              std::cout << "Skel:" << std::endl
                        << skel->getName() << " - " << j << std::endl;
              std::cout << "Contact Type:" << std::endl
                        << constraints[k]->getDofContactType(skel->getDof(j))
                        << std::endl;
              if (equals(
                      bruteForce.edgeAPos,
                      analytical.edgeAPos,
                      estimateThreshold))
              {
                std::cout << "Edge A Pos correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge A Pos analytical: " << std::endl
                          << analytical.edgeAPos << std::endl;
                std::cout << "Edge A Pos brute force: " << std::endl
                          << bruteForce.edgeAPos << std::endl;
              }
              if (equals(
                      bruteForce.edgeADir,
                      analytical.edgeADir,
                      estimateThreshold))
              {
                std::cout << "Edge A Dir correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge A Dir analytical: " << std::endl
                          << analytical.edgeADir << std::endl;
                std::cout << "Edge A Dir brute force: " << std::endl
                          << bruteForce.edgeADir << std::endl;
              }
              if (equals(
                      bruteForce.edgeBPos,
                      analytical.edgeBPos,
                      estimateThreshold))
              {
                std::cout << "Edge B Pos correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge B Pos analytical: " << std::endl
                          << analytical.edgeBPos << std::endl;
                std::cout << "Edge B Pos brute force: " << std::endl
                          << bruteForce.edgeBPos << std::endl;
              }
              if (equals(
                      bruteForce.edgeBDir,
                      analytical.edgeBDir,
                      estimateThreshold))
              {
                std::cout << "Edge B Dir correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge B Dir analytical: " << std::endl
                          << analytical.edgeBDir << std::endl;
                std::cout << "Edge B Dir brute force: " << std::endl
                          << bruteForce.edgeBDir << std::endl;
              }
              return false;
            }
          }

          EdgeData analyticalGradient
              = constraints[k]->getEdgeGradient(skel->getDof(j));
          EdgeData finiteDifferenceGradient;
          finiteDifferenceGradient.edgeAPos
              = (bruteForce.edgeAPos - original.edgeAPos) / EPS;
          finiteDifferenceGradient.edgeADir
              = (bruteForce.edgeADir - original.edgeADir) / EPS;
          finiteDifferenceGradient.edgeBPos
              = (bruteForce.edgeBPos - original.edgeBPos) / EPS;
          finiteDifferenceGradient.edgeBDir
              = (bruteForce.edgeBDir - original.edgeBDir) / EPS;

          Eigen::Vector3d analyticalIntersectionGradient
              = math::getContactPointGradient(
                  original.edgeAPos,
                  analyticalGradient.edgeAPos,
                  original.edgeADir,
                  analyticalGradient.edgeADir,
                  original.edgeBPos,
                  analyticalGradient.edgeBPos,
                  original.edgeBDir,
                  analyticalGradient.edgeBDir);

          Eigen::Vector3d finiteDifferenceIntersectionGradient
              = (bruteIntersection - originalIntersectionPoint) / EPS;

          estimateThreshold = 1e-8;

          // Check the intersection point first, because the actual input points
          // can be different if the collision detector decided to use a
          // different vertex on either of the edges, which will screw
          // everything up. Even if it does this, though, the intersection
          // points will remain unchanged, so check those first.
          if (!equals(
                  finiteDifferenceIntersectionGradient,
                  analyticalIntersectionGradient,
                  estimateThreshold))
          {
            std::cout << "Got intersection gradient wrong!" << std::endl;
            std::cout << "Skel:" << std::endl
                      << skel->getName() << " - " << j << std::endl;
            std::cout << "Contact Type:" << std::endl
                      << constraints[k]->getDofContactType(skel->getDof(j))
                      << std::endl;
            std::cout << "Brute force intersection gradient:" << std::endl
                      << finiteDifferenceIntersectionGradient << std::endl;
            std::cout << "Analytical intersection gradient:" << std::endl
                      << analyticalIntersectionGradient << std::endl;

            if (!equals(
                    analyticalGradient,
                    finiteDifferenceGradient,
                    estimateThreshold))
            {
              std::cout << "Got edge gradient wrong!" << std::endl;
              std::cout << "Skel:" << std::endl
                        << skel->getName() << " - " << j << std::endl;
              std::cout << "Contact Type:" << std::endl
                        << constraints[k]->getDofContactType(skel->getDof(j))
                        << std::endl;
              // TODO: dirty hack to save retyping
              bruteForce = finiteDifferenceGradient;
              analytical = analyticalGradient;
              if (equals(
                      bruteForce.edgeAPos,
                      analytical.edgeAPos,
                      estimateThreshold))
              {
                std::cout << "Edge A Pos correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge A Pos analytical: " << std::endl
                          << analytical.edgeAPos << std::endl;
                std::cout << "Edge A Pos brute force: " << std::endl
                          << bruteForce.edgeAPos << std::endl;
              }
              if (equals(
                      bruteForce.edgeADir,
                      analytical.edgeADir,
                      estimateThreshold))
              {
                std::cout << "Edge A Dir correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge A Dir analytical: " << std::endl
                          << analytical.edgeADir << std::endl;
                std::cout << "Edge A Dir brute force: " << std::endl
                          << bruteForce.edgeADir << std::endl;
              }
              if (equals(
                      bruteForce.edgeBPos,
                      analytical.edgeBPos,
                      estimateThreshold))
              {
                std::cout << "Edge B Pos correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge B Pos analytical: " << std::endl
                          << analytical.edgeBPos << std::endl;
                std::cout << "Edge B Pos brute force: " << std::endl
                          << bruteForce.edgeBPos << std::endl;
              }
              if (equals(
                      bruteForce.edgeBDir,
                      analytical.edgeBDir,
                      estimateThreshold))
              {
                std::cout << "Edge B Dir correct!" << std::endl;
              }
              else
              {
                std::cout << "Edge B Dir analytical: " << std::endl
                          << analytical.edgeBDir << std::endl;
                std::cout << "Edge B Dir brute force: " << std::endl
                          << bruteForce.edgeBDir << std::endl;
              }
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

bool verifyPerturbedContactPositions(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const double EPS = 1e-8;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3d pos = constraints[k]->getContactWorldPosition();
        Eigen::Vector3d normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3d analytical
            = constraints[k]->estimatePerturbedContactPosition(skel, j, EPS);
        Eigen::Vector3d bruteForce
            = constraints[k]->bruteForcePerturbedContactPosition(
                world, skel, j, EPS);
        if (!equals(analytical, bruteForce, 1e-8))
        {
          std::cout << "Skel:" << std::endl
                    << skel->getName() << " - " << j << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Contact Normal:" << std::endl << normal << std::endl;
          std::cout << "Original Contact Pos:" << std::endl << pos << std::endl;
          std::cout << "Analytical Contact Pos:" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical Contact Pos Diff:" << std::endl
                    << (analytical - pos) << std::endl;
          std::cout << "Brute Force Contact Pos Diff:" << std::endl
                    << (bruteForce - pos) << std::endl;
          return false;
        }

        Eigen::Vector3d finiteDifferenceGradient = (bruteForce - pos) / EPS;
        Eigen::Vector3d analyticalGradient
            = constraints[k]->getContactPositionGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
        {
          std::cout << "Skel:" << std::endl
                    << skel->getName() << " - " << j << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Contact Normal:" << std::endl << normal << std::endl;
          std::cout << "Contact Pos:" << std::endl << pos << std::endl;
          std::cout << "Analytical Contact Pos Gradient:" << std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference Contact Pos Gradient:" << std::endl
                    << finiteDifferenceGradient << std::endl;
          std::cout << "Diff:" << std::endl
                    << analyticalGradient - finiteDifferenceGradient
                    << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

bool verifyPerturbedContactNormals(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const double EPS = 1e-7;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3d normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3d analytical
            = constraints[k]->estimatePerturbedContactNormal(skel, j, EPS);
        Eigen::Vector3d bruteForce
            = constraints[k]->bruteForcePerturbedContactNormal(
                world, skel, j, EPS);
        if (!equals(analytical, bruteForce, 1e-8))
        {
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Original Contact Normal:" << std::endl
                    << normal << std::endl;
          std::cout << "Analytical Contact Normal:" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical Contact Normal Diff:" << std::endl
                    << (analytical - normal) << std::endl;
          std::cout << "Brute Force Contact Normal Diff:" << std::endl
                    << (bruteForce - normal) << std::endl;
          return false;
        }

        Eigen::Vector3d finiteDifferenceGradient = (analytical - normal) / EPS;
        Eigen::Vector3d analyticalGradient
            = constraints[k]->getContactNormalGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
        {
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Contact Normal:" << std::endl << normal << std::endl;
          std::cout << "Analytical Contact Normal Gradient:" << std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference Contact Normal Gradient:" << std::endl
                    << finiteDifferenceGradient << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

bool verifyPerturbedContactForceDirections(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const double EPS = 1e-7;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3d normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3d dir = constraints[k]->getContactWorldForceDirection();
        Eigen::Vector3d analytical
            = constraints[k]->estimatePerturbedContactForceDirection(
                skel, j, EPS);
        Eigen::Vector3d bruteForce
            = constraints[k]->bruteForcePerturbedContactForceDirection(
                world, skel, j, EPS);
        if (!equals(analytical, bruteForce, 1e-8))
        {
          std::cout << "Constraint index:" << std::endl << k << std::endl;
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Diff wrt index:" << std::endl << j << std::endl;

          auto dof = skel->getDof(j);
          int jointIndex = dof->getIndexInJoint();
          math::Jacobian relativeJac = dof->getJoint()->getRelativeJacobian();
          dynamics::BodyNode* childNode = dof->getChildBodyNode();
          Eigen::Isometry3d transform = childNode->getWorldTransform();
          Eigen::Vector6d localTwist = relativeJac.col(jointIndex);
          Eigen::Vector6d worldTwist = math::AdT(transform, localTwist);

          std::cout << "local twist:" << std::endl << localTwist << std::endl;
          std::cout << "world twist:" << std::endl << worldTwist << std::endl;

          std::cout << "Contact type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Index:" << std::endl
                    << constraints[k]->getIndexInConstraint() << std::endl;
          std::cout << "Original Contact Normal:" << std::endl
                    << normal << std::endl;
          std::cout << "Original Contact Force Direction:" << std::endl
                    << dir << std::endl;
          std::cout << "Analytical Contact Force Direction Diff:" << std::endl
                    << (analytical - dir) << std::endl;
          std::cout << "Brute Force Contact Force Direction Diff:" << std::endl
                    << (bruteForce - dir) << std::endl;
          return false;
        }

        Eigen::Vector3d finiteDifferenceGradient = (analytical - dir) / EPS;
        Eigen::Vector3d analyticalGradient
            = constraints[k]->getContactForceGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
        {
          Eigen::Vector3d analyticalGradient
              = constraints[k]->getContactForceGradient(skel->getDof(j));
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Contact Normal:" << std::endl << normal << std::endl;
          std::cout << "Contact Force Direction:" << std::endl
                    << dir << std::endl;
          std::cout << "Analytical Contact Force Gradient:" << std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference Contact Force Gradient:" << std::endl
                    << finiteDifferenceGradient << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

bool verifyPerturbedScrewAxis(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const double EPS = 1e-6;
  std::vector<DegreeOfFreedom*> dofs = world->getDofs();
  for (int j = 0; j < dofs.size(); j++)
  {
    DegreeOfFreedom* axis = dofs[j];
    for (int k = 0; k < dofs.size(); k++)
    {
      DegreeOfFreedom* wrt = dofs[k];
      for (int q = 0; q < constraints.size(); q++)
      {
        Eigen::Vector6d original = constraints[q]->getWorldScrewAxis(axis);
        Eigen::Vector6d analytical
            = constraints[q]->estimatePerturbedScrewAxis(axis, wrt, EPS);
        Eigen::Vector6d bruteForce
            = constraints[q]->bruteForceScrewAxis(axis, wrt, EPS);

        if (!equals(analytical, bruteForce, 1e-8))
        {
          std::cout << "Axis: " << axis->getSkeleton()->getName() << " - "
                    << axis->getIndexInSkeleton() << std::endl;
          std::cout << "Rotate: " << wrt->getSkeleton()->getName() << " - "
                    << wrt->getIndexInSkeleton() << std::endl;
          std::cout << "Axis Contact Type: "
                    << constraints[q]->getDofContactType(axis) << std::endl;
          std::cout << "Rotate Contact Type: "
                    << constraints[q]->getDofContactType(wrt) << std::endl;
          std::cout << "Is parent: " << constraints[q]->isParent(wrt, axis)
                    << std::endl;
          std::cout << "Analytical World Screw:" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical World Screw Diff:" << std::endl
                    << (analytical - original) << std::endl;
          std::cout << "Brute Force World Screw Diff:" << std::endl
                    << (bruteForce - original) << std::endl;
          return false;
        }

        Eigen::Vector6d finiteDifferenceGradient
            = (bruteForce - original) / EPS;
        Eigen::Vector6d analyticalGradient
            = constraints[q]->getScrewAxisGradient(axis, wrt);
        if (!equals(analyticalGradient, finiteDifferenceGradient, EPS * 2))
        {
          std::cout << "Axis:" << std::endl
                    << axis->getSkeleton()->getName() << " - "
                    << axis->getIndexInSkeleton() << std::endl;
          std::cout << "Rotate:" << std::endl
                    << wrt->getSkeleton()->getName() << " - "
                    << wrt->getIndexInSkeleton() << std::endl;
          std::cout << "Axis Contact Type:" << std::endl
                    << constraints[q]->getDofContactType(axis) << std::endl;
          std::cout << "Rotate Contact Type:" << std::endl
                    << constraints[q]->getDofContactType(wrt) << std::endl;
          std::cout << "Analytical World Screw Gradient:" << std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference World Screw Gradient:" << std::endl
                    << finiteDifferenceGradient << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

bool verifyAnalyticalConstraintDerivatives(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  const double EPS = 1e-7;

  std::vector<DegreeOfFreedom*> dofs = world->getDofs();
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  for (int i = 0; i < constraints.size(); i++)
  {
    for (int j = 0; j < dofs.size(); j++)
    {
      for (int k = 0; k < dofs.size(); k++)
      {
        DegreeOfFreedom* rotate = dofs[j];
        DegreeOfFreedom* axis = dofs[k];
        double originalValue = constraints[i]->getConstraintForce(axis);

        double analytical
            = constraints[i]->getConstraintForceDerivative(axis, rotate);

        double originalPosition = rotate->getPosition();
        rotate->setPosition(originalPosition + EPS);
        BackpropSnapshotPtr newPtr = neural::forwardPass(world, true);
        double newValue
            = constraints[i]->getPeerConstraint(newPtr)->getConstraintForce(
                axis);

        Eigen::Vector6d gradientOfWorldForce
            = constraints[i]->getContactWorldForceGradient(rotate);
        Eigen::Vector6d worldTwist = constraints[i]->getWorldScrewAxis(axis);

        rotate->setPosition(originalPosition);

        double bruteForce = (newValue - originalValue) / EPS;

        if (abs(analytical - bruteForce) > 1e-8)
        {
          std::cout << "Rotate:" << k << " - "
                    << rotate->getSkeleton()->getName() << " - "
                    << rotate->getIndexInSkeleton() << std::endl;
          std::cout << "Axis:" << j << " - " << axis->getSkeleton()->getName()
                    << " - " << axis->getIndexInSkeleton() << std::endl;
          std::cout << "Original:" << std::endl << originalValue << std::endl;
          std::cout << "Analytical:" << std::endl << analytical << std::endl;
          std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
          std::cout << "Gradient of world force:" << std::endl
                    << gradientOfWorldForce << std::endl;
          std::cout << "World twist:" << std::endl << worldTwist << std::endl;
          double analytical
              = constraints[i]->getConstraintForceDerivative(axis, rotate);
          return false;
        }
      }
    }
  }
  return true;
}

bool verifyAnalyticalA_cJacobian(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  for (int i = 0; i < constraints.size(); i++)
  {
    Eigen::MatrixXd analytical
        = constraints[i]->getConstraintForcesJacobian(world);
    Eigen::MatrixXd bruteForce
        = constraints[i]->bruteForceConstraintForcesJacobian(world);
    Eigen::VectorXd A_cCol = constraints[i]->getConstraintForces(world);
    if (!equals(analytical, bruteForce, 1e-8))
    {
      std::cout << "A_c col:" << std::endl << A_cCol << std::endl;
      std::cout << "Analytical constraint forces Jac:" << std::endl
                << analytical << std::endl;
      std::cout << "Brute force constraint forces Jac:" << std::endl
                << bruteForce << std::endl;
      std::cout << "Constraint forces Jac diff:" << std::endl
                << (analytical - bruteForce) << std::endl;
    }

    // Check that the skeleton-by-skeleton computation works

    int col = 0;
    for (int j = 0; j < world->getNumSkeletons(); j++)
    {
      auto wrt = world->getSkeleton(j);

      // Go skeleton-by-skeleton

      int row = 0;
      for (int k = 0; k < world->getNumSkeletons(); k++)
      {
        auto skel = world->getSkeleton(k);
        Eigen::MatrixXd gold
            = analytical.block(row, col, skel->getNumDofs(), wrt->getNumDofs());
        Eigen::MatrixXd chunk
            = constraints[i]->getConstraintForcesJacobian(skel, wrt);
        if (!equals(gold, chunk, 1e-8))
        {
          std::cout << "Analytical constraint forces Jac of " << skel->getName()
                    << " wrt " << wrt->getName() << " incorrect!" << std::endl;
          std::cout << "Analytical constraint forces Jac chunk of world:"
                    << std::endl
                    << gold << std::endl;
          std::cout << "Analytical constraint forces Jac skel-by-skel:"
                    << std::endl
                    << chunk << std::endl;
        }

        row += skel->getNumDofs();
      }

      // Try a group of skeletons

      std::vector<std::shared_ptr<dynamics::Skeleton>> skels;
      for (int k = 0; k < world->getNumSkeletons(); k++)
      {
        skels.push_back(world->getSkeleton(k));
      }

      Eigen::MatrixXd gold
          = analytical.block(0, col, world->getNumDofs(), wrt->getNumDofs());
      Eigen::MatrixXd chunk
          = constraints[i]->getConstraintForcesJacobian(skels, wrt);
      if (!equals(gold, chunk, 1e-8))
      {
        std::cout << "Analytical constraint forces Jac of "
                  << "all skeletons"
                  << " wrt " << wrt->getName() << " incorrect!" << std::endl;
        std::cout << "Analytical constraint forces Jac chunk of world:"
                  << std::endl
                  << gold << std::endl;
        std::cout << "Analytical constraint forces Jac skel-by-skel:"
                  << std::endl
                  << chunk << std::endl;
      }

      col += wrt->getNumDofs();
    }

    std::vector<std::shared_ptr<dynamics::Skeleton>> skels;
    for (int j = 0; j < world->getNumSkeletons(); j++)
    {
      skels.push_back(world->getSkeleton(j));
    }

    Eigen::MatrixXd skelAnalytical
        = constraints[i]->getConstraintForcesJacobian(skels);
    if (!equals(analytical, skelAnalytical, 1e-8))
    {
      std::cout << "Analytical constraint forces Jac of "
                << "all skeletons"
                << " wrt "
                << "all skeletons"
                << " incorrect!" << std::endl;
      std::cout << "Analytical constraint forces Jac of world:" << std::endl
                << analytical << std::endl;
      std::cout << "Analytical constraint forces Jac skel-by-skel:" << std::endl
                << skelAnalytical << std::endl;
    }
  }

  return true;
}

bool verifyJacobianOfClampingConstraints(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::VectorXd f0 = Eigen::VectorXd::Random(classicPtr->getNumClamping());

  Eigen::MatrixXd analytical
      = classicPtr->getJacobianOfClampingConstraints(world, f0);
  Eigen::MatrixXd bruteForce
      = classicPtr->finiteDifferenceJacobianOfClampingConstraints(world, f0);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "getJacobianOfClampingConstraints error:" << std::endl;
    std::cout << "f0:" << std::endl << f0 << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    return false;
  }
  return true;
}

bool verifyJacobianOfClampingConstraintsTranspose(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::VectorXd v0 = Eigen::VectorXd::Random(world->getNumDofs());

  Eigen::MatrixXd analytical
      = classicPtr->getJacobianOfClampingConstraintsTranspose(world, v0);
  Eigen::MatrixXd bruteForce
      = classicPtr->finiteDifferenceJacobianOfClampingConstraintsTranspose(
          world, v0);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "getJacobianOfClampingConstraintsTranspose error:"
              << std::endl;
    std::cout << "v0:" << std::endl << v0 << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    return false;
  }
  return true;
}

bool verifyJacobianOfUpperBoundConstraints(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::VectorXd f0 = Eigen::VectorXd::Random(classicPtr->getNumUpperBound());

  Eigen::MatrixXd analytical
      = classicPtr->getJacobianOfUpperBoundConstraints(world, f0);
  Eigen::MatrixXd bruteForce
      = classicPtr->finiteDifferenceJacobianOfUpperBoundConstraints(world, f0);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "getJacobianOfUpperBoundConstraints error:" << std::endl;
    std::cout << "f0:" << std::endl << f0 << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    return false;
  }
  return true;
}

bool verifyAnalyticalJacobians(WorldPtr world)
{
  return verifyPerturbedContactEdges(world)
         && verifyPerturbedContactPositions(world)
         && verifyPerturbedContactNormals(world)
         && verifyPerturbedContactForceDirections(world)
         && verifyPerturbedScrewAxis(world)
         && verifyAnalyticalContactPositionJacobians(world)
         && verifyAnalyticalContactNormalJacobians(world)
         && verifyAnalyticalContactForceJacobians(world)
         && verifyAnalyticalA_c(world)
         && verifyAnalyticalConstraintDerivatives(world)
         && verifyAnalyticalA_cJacobian(world)
         && verifyAnalyticalConstraintMatrixEstimates(world)
         && verifyJacobianOfClampingConstraints(world)
         && verifyJacobianOfClampingConstraintsTranspose(world)
         && verifyJacobianOfUpperBoundConstraints(world);
}

class MyWindow : public dart::gui::glut::SimWindow
{
public:
  /// Constructor
  MyWindow(WorldPtr world)
  {
    setWorld(world);
  }
};

void renderWorld(WorldPtr world)
{
  // Create a window for rendering the world and handling user input
  MyWindow window(world);
  // Initialize glut, initialize the window, and begin the glut event loop
  int argc = 0;
  glutInit(&argc, nullptr);
  window.initWindow(640, 480, "Test");
  glutMainLoop();
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

// This is the margin so that finite-differencing over position doesn't break
// contacts
const double CONTACT_MARGIN = 1e-5;

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
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - CONTACT_MARGIN), 0);
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

There are two blocks, each with two DOFs (X, Y). The force pushing them
together (or apart, if negative) is configurable.

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
  rightBoxPosition.translation() = Eigen::Vector3d(0.5 - CONTACT_MARGIN, 0, 0);
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
  std::cout << "Contacts: " <<
  world->getLastCollisionResult().getNumContacts()
            << std::endl;
  for (std::size_t i = 0; i <
  world->getLastCollisionResult().getNumContacts(); i++)
  {
    collision::Contact contact =
  world->getLastCollisionResult().getContact(i); std::cout << "Contact " << i
  << " "
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
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - CONTACT_MARGIN), 0);
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

There's a reverse pendulum sled with three DOFs, x and y axis, and angle of
the reverse pendulum, with a force driving it into the ground. The ground has
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
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - 1e-6), 0);
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

  reversePendulumSled->computeForwardDynamics();
  reversePendulumSled->integrateVelocities(world->getTimeStep());
  VectorXd worldVel = world->getVelocities();

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
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
    topBoxPosition.translation()
        = Eigen::Vector3d(xOffset, 0.5 - CONTACT_MARGIN, 0);
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
  // our constraint group ordering doesn't match our world ordering, which
  // will help us catch bugs in matrix layout.
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
"rotationDegree" position at each link. There's also a fixed plane at the end
of the robot arm that it intersects with.
*/
void testRobotArm(
    std::size_t numLinks, double rotationRadians, int attachPoint = -1)
{
  // World
  WorldPtr world = World::create();
  /*
  for (auto key : collision::CollisionDetector::getFactory()->getKeys())
  {
    std::cout << "Option: " << key << std::endl;
  }
  */
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));

  std::vector<BodyNode*> nodes;

  for (std::size_t i = 0; i < numLinks; i++)
  {
    RevoluteJoint::Properties jointProps;
    jointProps.mName = "revolute_" + std::to_string(i);
    BodyNode::Properties bodyProps;
    bodyProps.mName = "arm_" + std::to_string(i);
    std::pair<RevoluteJoint*, BodyNode*> jointPair
        = arm->createJointAndBodyNodePair<RevoluteJoint>(
            parent, jointProps, bodyProps);
    nodes.push_back(jointPair.second);
    if (parent != nullptr)
    {
      Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
      armOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
    if ((attachPoint == -1 && i < numLinks - 1) || i != attachPoint)
    {
      ShapeNode* visual = parent->createShapeNodeWith<VisualAspect>(boxShape);
    }
  }

  if (attachPoint != -1)
  {
    parent = nodes[attachPoint];
  }

  std::shared_ptr<BoxShape> endShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0) * sqrt(1.0 / 3.0)));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);

  arm->setPositions(Eigen::VectorXd::Ones(arm->getNumDofs()) * rotationRadians);
  world->addSkeleton(arm);

  Eigen::Isometry3d worldTransform = parent->getWorldTransform();
  Eigen::Matrix3d rotation
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, -45) * 3.1415 / 180)
        * worldTransform.linear().transpose();
  endNode->setRelativeRotation(rotation);

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
                                  + Eigen::Vector3d(-(1.0 - 1e-2), 0.0, 0);
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
  arm->setVelocities(Eigen::VectorXd::Ones(arm->getNumDofs()) * 0.05);

  VectorXd worldVel = world->getVelocities();

  /*
  VectorXd pos = world->getPositions();
  pos(0) += 1e-4;
  world->setPositions(pos);
  */

  // renderWorld(world);
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
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

TEST(GRADIENTS, ARM_3_LINK_30_DEG_MIDDLE_ATTACH)
{
  testRobotArm(3, 30.0 / 180 * 3.1415, 1);
}
*/

/**
 * This sets up two boxes colliding with each other. Each can rotate and
 * translate freely. We set it up so that the one box's corner is contacting
 * the other box's face.
 *
 *  /\ +---+
 * /  \|   |
 * \  /|   |
 *  \/ +---+
 */
void testVertexFaceCollision(bool isSelfCollision)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box1Node
      = box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box1Shape);

  // This box is rotated by 45 degrees on the X and Y axis, so that it's
  // sqrt(3) along the X axis.
  SkeletonPtr box2 = Skeleton::create("vertex box");
  std::pair<FreeJoint*, BodyNode*> box2Pair;

  if (isSelfCollision)
  {
    box1->enableSelfCollision(true);
    box2Pair = box1Pair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    box2Pair = box2->createJointAndBodyNodePair<FreeJoint>();
  }

  double box2EdgeSize = sqrt(1.0 / 3);
  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0) * box2EdgeSize));
  ShapeNode* box2Node
      = box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box2Shape);
  FreeJoint* box2Joint = box2Pair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d box2Position = Eigen::Isometry3d::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, -45) * 3.1415 / 180);
  box2Position.translation()
      = box2Position.linear() * Eigen::Vector3d(1.0 - 2e-2, 0, 0);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
}

/*
TEST(GRADIENTS, VERTEX_FACE_COLLISION)
{
  testVertexFaceCollision(false);
}

TEST(GRADIENTS, VERTEX_FACE_SELF_COLLISION)
{
  testVertexFaceCollision(true);
}
*/

/**
 * This sets up two boxes colliding with each other. Each can rotate and
 * translate freely. We set it up so that the one box's edge is contacting the
 * other box's edge
 *
 *  /\
 * /  \
 * \  /+---+
 *  \/ |   |
 *     +---+
 */
void testEdgeEdgeCollision(bool isSelfCollision)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box1Node
      = box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box1Shape);

  // This box is rotated by 45 degrees on the X and Y axis, so that it's
  // sqrt(3) along the X axis.
  SkeletonPtr box2 = Skeleton::create("vertex box");
  std::pair<FreeJoint*, BodyNode*> box2Pair;

  if (isSelfCollision)
  {
    box1->enableSelfCollision(true);
    box2Pair = box1Pair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    box2Pair = box2->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box2Node
      = box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box2Shape);
  FreeJoint* box2Joint = box2Pair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d box2Position = Eigen::Isometry3d::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, 45) * 3.1415 / 180);
  box2Position.translation() = box2Position.linear() * Eigen::Vector3d(1, -1, 0)
                               * ((2 * sqrt(0.5) / sqrt(2)) - 2e-2);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  vels(9) += 0.1;  // +x
  vels(10) -= 0.1; // -y

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
}

/*
TEST(GRADIENTS, EDGE_EDGE_COLLISION)
{
  testEdgeEdgeCollision(false);
}

TEST(GRADIENTS, EDGE_EDGE_SELF_COLLISION)
{
  testEdgeEdgeCollision(true);
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
"rotationDegree" position at each link. There's also a fixed plane at the end
of the robot arm that it intersects with.
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
  EXPECT_TRUE(verifyGradientBackprop(world, 50, [](WorldPtr world) {
    Eigen::VectorXd pos = world->getPositions();
    Eigen::VectorXd vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  }));
  // Run this a bunch to let us catch it on top
  EXPECT_TRUE(verifyBulkPass(world, 1000, 50));
}

/*
TEST(GRADIENTS, CARTPOLE_15_DEG)
{
  testCartpole(15.0 / 180.0 * 3.1415);
}
*/

void testWorldSpace(std::size_t numLinks)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");

  std::pair<WeldJoint*, BodyNode*> rootJointPair
      = arm->createJointAndBodyNodePair<WeldJoint>(nullptr);
  Eigen::Isometry3d rootTransform = Eigen::Isometry3d::Identity();
  rootTransform.linear()
      = eulerXYZToMatrix(Eigen::Vector3d(90.0 * 3.14159 / 180.0, 0, 0));
  rootJointPair.first->setTransformFromParentBodyNode(rootTransform);

  BodyNode* parent = rootJointPair.second;

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

      Eigen::Isometry3d bodyOffset = Eigen::Isometry3d::Identity();
      bodyOffset.translation() = Eigen::Vector3d(0, -1.0, 0);
      jointPair.first->setTransformFromChildBodyNode(bodyOffset);
    }
    jointPair.second->setMass(1.0);
    jointPair.first->setAxis(Eigen::Vector3d(1, 0, 0));
    parent = jointPair.second;
  }

  world->addSkeleton(arm);

  SkeletonPtr floor = Skeleton::create("floor");
  std::pair<WeldJoint*, BodyNode*> floorJointPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  world->addSkeleton(floor);

  EXPECT_TRUE(verifyWorldSpaceTransform(world));
}

/*
TEST(GRADIENTS, WORLD_SPACE_5_LINK_ROBOT)
{
  testWorldSpace(5);
}
*/

/******************************************************************************

This test sets up a configuration that looks something like this:

   O-----O
   |     |
   |     |
   O

It's a robot arm, with a rotating base, and 3 links. Each link is of unit
length. This is supposed to make it easier to reason about geometry Jacobians.
*/
void testSimple3Link()
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");

  std::pair<RevoluteJoint*, BodyNode*> zerothJointPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  zerothJointPair.first->setAxis(Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d zerothOffset = Eigen::Isometry3d::Identity();
  zerothOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
  zerothJointPair.first->setTransformFromChildBodyNode(zerothOffset);

  std::pair<RevoluteJoint*, BodyNode*> firstJointPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(zerothJointPair.second);
  firstJointPair.first->setAxis(Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d firstOffset = Eigen::Isometry3d::Identity();
  firstOffset.translation() = Eigen::Vector3d(1.0, 0.0, 0);
  firstJointPair.first->setTransformFromChildBodyNode(firstOffset);

  std::pair<RevoluteJoint*, BodyNode*> secondJointPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(firstJointPair.second);
  secondJointPair.first->setAxis(Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d secondOffset = Eigen::Isometry3d::Identity();
  secondOffset.translation() = Eigen::Vector3d(0, -1.0, 0);
  secondJointPair.first->setTransformFromChildBodyNode(secondOffset);

  Eigen::MatrixXd expectedJac = Eigen::MatrixXd::Zero(9, 3);
  expectedJac <<
      /* Body 1 X */
      1,
      0, 0, /* Body 1 Y */ 0, 0, 0, /* Body 1 Z */ 0, 0, 0,
      /* Body 2 X */ 1, 0, 0, /* Body 2 Y */ -1, -1, 0,
      /* Body 2 Z */ 0, 0, 0,
      /* Body 3 X */ 0, -1, -1, /* Body 3 Y */ -1, -1, 0,
      /* Body 3 Z */ 0, 0, 0;
  Eigen::MatrixXd analyticalJac
      = jointToWorldLinearJacobian(arm, arm->getBodyNodes());

  if (!equals(analyticalJac, expectedJac, 1e-5))
  {
    std::cout << "Expected: \n"
              << expectedJac << "\nAnalytical: \n"
              << analyticalJac << "\nDiff: \n"
              << (expectedJac - analyticalJac) << "\n";
  }
  EXPECT_TRUE(equals(analyticalJac, expectedJac));

  world->addSkeleton(arm);

  EXPECT_TRUE(verifyLinearJacobian(
      world, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
  EXPECT_TRUE(verifySpatialJacobian(
      world, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
}

/*
TEST(GRADIENTS, WORLD_SPACE_SIMPLE_LINK)
{
  testSimple3Link();
}
*/

void testWorldSpaceWithBoxes()
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr boxes = Skeleton::create("boxes");

  // We set up the root so that it's off by 90 deg on the x-axis, establishing
  // a non-world root frame

  std::pair<WeldJoint*, BodyNode*> rootJointPair
      = boxes->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* rootJoint = rootJointPair.first;
  BodyNode* rootBody = rootJointPair.second;
  Eigen::Isometry3d rootTransform = Eigen::Isometry3d::Identity();
  rootTransform.linear()
      = eulerXYZToMatrix(Eigen::Vector3d(90.0 * 3.14159 / 180.0, 0, 0));
  rootJoint->setTransformFromParentBodyNode(rootTransform);

  // Then we add a box, on a free transform from the root. This is now a
  // different frame than the world, by 90 deg on the x axis.

  std::pair<TranslationalJoint*, BodyNode*> boxJointPair
      = boxes->createJointAndBodyNodePair<TranslationalJoint>(rootBody);

  world->addSkeleton(boxes);

  EXPECT_TRUE(verifyWorldSpaceTransform(world));
}

/*
TEST(GRADIENTS, WORLD_SPACE_BOXES)
{
  testWorldSpaceWithBoxes();
}
*/

///////////////////////////////////////////////////////////////////////////////
// Just idiot checking that the code doesn't crash on silly edge cases.
///////////////////////////////////////////////////////////////////////////////

/*
TEST(GRADIENTS, EMPTY_WORLD)
{
  WorldPtr world = World::create();
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

TEST(GRADIENTS, EMPTY_SKELETON)
{
  WorldPtr world = World::create();
  SkeletonPtr empty = Skeleton::create("empty");
  world->addSkeleton(empty);
  VectorXd worldVel = world->getVelocities();
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
}
*/

///////////////////////////////////////////////////////////////////////////////
// Checking the trajectory optimizations
///////////////////////////////////////////////////////////////////////////////

BodyNode* createTailSegment(BodyNode* parent, Eigen::Vector3d color)
{
  std::pair<RevoluteJoint*, BodyNode*> poleJointPair
      = parent->createChildJointAndBodyNodePair<RevoluteJoint>();
  RevoluteJoint* poleJoint = poleJointPair.first;
  BodyNode* pole = poleJointPair.second;
  poleJoint->setAxis(Eigen::Vector3d::UnitZ());

  std::shared_ptr<BoxShape> shape(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* poleShape
      = pole->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  poleShape->getVisualAspect()->setColor(color);
  poleJoint->setForceUpperLimit(0, 100.0);
  poleJoint->setForceLowerLimit(0, -100.0);
  poleJoint->setVelocityUpperLimit(0, 10000.0);
  poleJoint->setVelocityLowerLimit(0, -10000.0);

  Eigen::Isometry3d poleOffset = Eigen::Isometry3d::Identity();
  poleOffset.translation() = Eigen::Vector3d(0, -0.125, 0);
  poleJoint->setTransformFromChildBodyNode(poleOffset);
  poleJoint->setPosition(0, 90 * 3.1415 / 180);

  if (parent->getParentBodyNode() != nullptr)
  {
    Eigen::Isometry3d childOffset = Eigen::Isometry3d::Identity();
    childOffset.translation() = Eigen::Vector3d(0, 0.125, 0);
    poleJoint->setTransformFromParentBodyNode(childOffset);
  }

  return pole;
}

void testJumpWorm(bool offGround)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr jumpworm = Skeleton::create("jumpworm");

  std::pair<TranslationalJoint2D*, BodyNode*> rootJointPair
      = jumpworm->createJointAndBodyNodePair<TranslationalJoint2D>(nullptr);
  TranslationalJoint2D* rootJoint = rootJointPair.first;
  BodyNode* root = rootJointPair.second;

  std::shared_ptr<BoxShape> shape(new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* rootVisual
      = root->createShapeNodeWith<VisualAspect, CollisionAspect>(shape);
  Eigen::Vector3d black = Eigen::Vector3d::Zero();
  rootVisual->getVisualAspect()->setColor(black);
  rootJoint->setForceUpperLimit(0, 0);
  rootJoint->setForceLowerLimit(0, 0);
  rootJoint->setForceUpperLimit(1, 0);
  rootJoint->setForceLowerLimit(1, 0);
  rootJoint->setVelocityUpperLimit(0, 1000.0);
  rootJoint->setVelocityLowerLimit(0, -1000.0);
  rootJoint->setVelocityUpperLimit(1, 1000.0);
  rootJoint->setVelocityLowerLimit(1, -1000.0);

  BodyNode* tail1 = createTailSegment(
      root, Eigen::Vector3d(182.0 / 255, 223.0 / 255, 144.0 / 255));
  BodyNode* tail2 = createTailSegment(
      tail1, Eigen::Vector3d(223.0 / 255, 228.0 / 255, 163.0 / 255));
  BodyNode* tail3 = createTailSegment(
      tail2, Eigen::Vector3d(221.0 / 255, 193.0 / 255, 121.0 / 255));

  Eigen::VectorXd pos = Eigen::VectorXd(5);
  pos << 0, 0, 90, 90, 45;
  jumpworm->setPositions(pos * 3.1415 / 180);

  world->addSkeleton(jumpworm);

  // Floor

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorJointPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorJointPair.first;
  BodyNode* floorBody = floorJointPair.second;
  Eigen::Isometry3d floorOffset = Eigen::Isometry3d::Identity();
  floorOffset.translation() = Eigen::Vector3d(0, offGround ? -0.7 : -0.56, 0);
  floorJoint->setTransformFromParentBodyNode(floorOffset);
  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(2.5, 0.25, 0.5)));
  ShapeNode* floorVisual
      = floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(
          floorShape);
  floorBody->setFrictionCoeff(0);

  world->addSkeleton(floor);

  rootJoint->setVelocity(1, -0.1);
  Eigen::VectorXd vels = world->getVelocities();

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));

  /*
  std::function<double(WorldPtr)> loss = [](WorldPtr world) {
    Eigen::VectorXd pos = world->getPositions();
    Eigen::VectorXd vel = world->getVelocities();
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
  };
  // Test to make sure the loss lambda doesn't crash
  double l = loss(world);
  EXPECT_TRUE(verifyGradientBackprop(world, 50, loss));
  EXPECT_TRUE(verifyBulkPass(world, 1000, 50));
  */
}

TEST(GRADIENTS, JUMP_WORM)
{
  testJumpWorm(false);
}

/*
TEST(GRADIENTS, JUMP_WORM_OFF_GROUND)
{
  testJumpWorm(true);
}
*/