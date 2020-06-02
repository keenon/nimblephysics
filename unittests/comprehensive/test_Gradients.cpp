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
#include "dart/simulation/World.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

class FullSnapshot
{
public:
  FullSnapshot(WorldPtr world);
  void restore();

private:
  WorldPtr mWorld;
  std::vector<Skeleton::Configuration> mSkeletonConfigurations;
};

FullSnapshot::FullSnapshot(WorldPtr world)
{
  mWorld = world;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    mSkeletonConfigurations.push_back(world->getSkeleton(i)->getConfiguration(
        Skeleton::ConfigFlags::CONFIG_ALL));
  }
}

void FullSnapshot::restore()
{
  for (std::size_t i = 0; i < mWorld->getNumSkeletons(); i++)
  {
    mWorld->getSkeleton(i)->setConfiguration(mSkeletonConfigurations[i]);
  }
}

/**
 * Brute force compute a torque->accel Jacobian
 */
MatrixXd finiteDifferenceForceVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd velocities)
{
  FullSnapshot snapshot(world);

  MatrixXd J(skel->getNumDofs(), skel->getNumDofs());

  skel->setVelocities(velocities);
  world->step(false);

  VectorXd originalVel = skel->getVelocities();
  VectorXd originalForces = skel->getForces();

  std::cout << "skel: " << std::endl;
  std::cout << "original vel:" << std::endl << originalVel << std::endl;

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < skel->getNumDofs(); i++)
  {
    snapshot.restore();

    skel->setVelocities(velocities);

    VectorXd tweakedForces = VectorXd(originalForces);
    tweakedForces(i) += EPSILON;
    skel->setForces(tweakedForces);

    world->step(false);

    std::cout << "vel(" << i << "):" << std::endl
              << skel->getVelocities() << std::endl;

    VectorXd velChange = (skel->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();

  return J;
}

/**
 * Brute force computation of \dot{v}_{t} -> \dot{v}_{t+1}
 */
MatrixXd finiteDifferenceVelVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd velocities)
{
  FullSnapshot snapshot(world);

  MatrixXd J(skel->getNumDofs(), skel->getNumDofs());

  skel->setVelocities(velocities);
  world->step(false);

  VectorXd originalVel = skel->getVelocities();

  double EPSILON = 1e-7;
  for (auto i = 0; i < skel->getNumDofs(); i++)
  {
    snapshot.restore();

    VectorXd tweakedVel = VectorXd(velocities);
    tweakedVel(i) += EPSILON;
    skel->setVelocities(tweakedVel);
    world->step(false);

    VectorXd velChange = (skel->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();

  return J;
}

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
bool verifyWorldClassicClampingConstraintMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);

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

  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  Eigen::MatrixXd A_cInv
      = A_c.completeOrthogonalDecomposition().pseudoInverse();
  MatrixXd mapper = A_cInv.eval().transpose() * A_c.transpose();
  VectorXd violationVelocities = mapper * proposedVelocities;
  VectorXd cleanVelocities = proposedVelocities - violationVelocities;

  world->setVelocities(cleanVelocities);
  // Populate the constraint matrices, without taking a time step or integrating
  // velocities
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
                << " pos: " << std::endl
                << world->getSkeleton(i)->getPositions() << std::endl
                << "vel: " << std::endl
                << world->getSkeleton(i)->getVelocities() << std::endl;
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
bool verifyWorldMassedClampingConstraintMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);
  neural::BackpropSnapshotPtr massedPtr
      = neural::forwardPass(world, neural::GradientMode::MASSED, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClampingConstraintMatrix forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  if (!massedPtr)
  {
    std::cout << "verifyWorldClampingConstraintMatrix forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::MASSED!"
              << std::endl;
    return false;
  }

  // Verify that the massed formulation and classic formulation relate as
  // expected.

  MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  MatrixXd massedA_c = massedPtr->getClampingConstraintMatrix();
  MatrixXd recoveredMassedA_c = classicPtr->getInvMassMatrix() * A_c;

  bool success = equals(recoveredMassedA_c, massedA_c);
  if (!success)
  {
    std::cout << "Classic Clamping Constraint Matrix: " << std::endl
              << A_c << std::endl;
    std::cout << "MInv: " << std::endl
              << classicPtr->getInvMassMatrix() << std::endl;
    std::cout << "MInv * Classic Clamping Constraint Matrix: " << std::endl
              << recoveredMassedA_c << std::endl;
    std::cout << "Massed Clamping Constraint Matrix: " << std::endl
              << massedA_c << std::endl;
    return false;
  }

  return true;
}

/**
 * This tests that P_c is getting computed correctly.
 */
bool verifyWorldClassicProjectionIntoClampsMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicProjectionIntoClampsMatrix forwardPass "
                 "returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  // Compute the analytical constraint forces, which should match our actual
  // constraint forces

  MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix();
  VectorXd analyticalConstraintForces = -2 * (P_c * proposedVelocities);

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
      analyticalError(pointer)
          = contactConstraintForces(i) - analyticalConstraintForces(pointer);
      pointer++;
    }
  }

  // Check that the analytical error is zero

  VectorXd zero = VectorXd::Zero(analyticalError.size());
  if (!equals(analyticalError, zero, 1e-3))
  {
    std::cout << "Proposed velocities: " << std::endl
              << proposedVelocities << std::endl;
    std::cout << "P_c: " << std::endl << P_c << std::endl;
    std::cout << "Constraint forces: " << std::endl
              << contactConstraintForces << std::endl;
    std::cout << "-(P_c * proposedVelocities) (should be the same as above): "
              << std::endl
              << analyticalConstraintForces << std::endl;
    std::cout << "Analytical error (should be zero):" << std::endl
              << analyticalError << std::endl;
    std::cout << "Zero: " << std::endl << zero << std::endl;

    // Recompute step by step

    /*
    MatrixXd clampingConstraintMatrix
        = classicPtr->getClampingConstraintMatrix();
    MatrixXd clampingConstraintMatrixPinv
        = clampingConstraintMatrix.completeOrthogonalDecomposition()
              .pseudoInverse();
    MatrixXd mapper = clampingConstraintMatrixPinv.eval().transpose()
                      * clampingConstraintMatrix.transpose();
    VectorXd violationVelocities = mapper * proposedVelocities;
    VectorXd violationAccel = violationVelocities / world->getTimeStep();
    VectorXd violationTorques = classicPtr->getMassMatrix() * violationAccel;
    VectorXd violationContactForces
        = clampingConstraintMatrixPinv * violationTorques;
    VectorXd violationTorquesSum = VectorXd(violationTorques.size());
    violationTorquesSum(0) = (violationTorques(0) - violationTorques(1)) / 2;
    violationTorquesSum(1) = (violationTorques(1) - violationTorques(0)) / 2;
    VectorXd violationContactForcesAlt
        = clampingConstraintMatrixPinv * violationTorquesSum;

    VectorXd violationTorquesRecovered
        = clampingConstraintMatrix * violationContactForces;

    std::cout << "A_c: " << std::endl << clampingConstraintMatrix << std::endl;
    std::cout << "A_c.pinv(): " << std::endl
              << clampingConstraintMatrixPinv << std::endl;
    std::cout << "A_c * A_c.pinv(): " << std::endl
              << clampingConstraintMatrix * clampingConstraintMatrixPinv
              << std::endl;
    std::cout << "A_c.pinv().transpose(): " << std::endl
              << clampingConstraintMatrixPinv.eval().transpose() << std::endl;
    std::cout << "A_c.transpose(): " << std::endl
              << clampingConstraintMatrix.transpose() << std::endl;
    std::cout << "Mapper: " << std::endl << mapper << std::endl;
    std::cout << "Violation velocities: " << std::endl
              << violationVelocities << std::endl;
    std::cout << "Violation accel: " << std::endl
              << violationAccel << std::endl;
    std::cout << "M: " << std::endl << classicPtr->getMassMatrix() << std::endl;
    std::cout << "Violation torques: " << std::endl
              << violationTorques << std::endl;
    std::cout << "Violation contact forces: " << std::endl
              << violationContactForces << std::endl;
    std::cout << "Violation torques sum: " << std::endl
              << violationTorquesSum << std::endl;
    std::cout << "Violation contact forces alt: " << std::endl
              << violationContactForcesAlt << std::endl;
    std::cout << "Violation torques recovered: " << std::endl
              << violationTorquesRecovered << std::endl;

    // Working the P_c steps backwards with the real constraint forces

    std::cout << "Working backwards: " << std::endl;

    std::cout << "Constraint forces: " << std::endl
              << contactConstraintForces << std::endl;
    violationTorquesRecovered
        = clampingConstraintMatrix * contactConstraintForces;
    std::cout << "Violation torques recovered: " << std::endl
              << violationTorquesRecovered << std::endl;
    VectorXd violationAccelRecovered
        = classicPtr->getInvMassMatrix() * violationTorquesRecovered;
    std::cout << "Violation accel recovered: " << std::endl
              << violationAccelRecovered << std::endl;
    VectorXd violationDeltaVRecovered
        = violationAccelRecovered * world->getTimeStep();
    std::cout << "Violation delta V recovered: " << std::endl
              << violationAccelRecovered << std::endl;
    */

    return false;
  }

  return true;
}

/**
 * This tests that P_c is getting computed correctly.
 */
bool verifyWorldMassedProjectionIntoClampsMatrix(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);
  neural::BackpropSnapshotPtr massedPtr
      = neural::forwardPass(world, neural::GradientMode::MASSED, true);

  if (!classicPtr)
  {
    std::cout
        << "verifyWorldMassedProjectionIntoClampsMatrix forwardPass returned a "
           "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
        << std::endl;
    return false;
  }

  if (!massedPtr)
  {
    std::cout
        << "verifyWorldMassedProjectionIntoClampsMatrix forwardPass returned a "
           "null BackpropSnapshotPtr for GradientMode::MASSED!"
        << std::endl;
    return false;
  }

  // Check the massed version

  MatrixXd projectionIntoClampsMatrix
      = classicPtr->getProjectionIntoClampsMatrix();
  MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix();
  MatrixXd E = classicPtr->getUpperBoundMappingMatrix();
  MatrixXd massedProjectionIntoClampsMatrix
      = massedPtr->getProjectionIntoClampsMatrix();
  MatrixXd recoveredMassedProjectionIntoClampsMatrix
      = world->getInvMassMatrix() * (A_c + A_ub * E)
        * projectionIntoClampsMatrix * world->getInvMassMatrix()
        * world->getTimeStep();
  MatrixXd clampingConstraintMatrixInv
      = (A_c + A_ub * E).completeOrthogonalDecomposition().pseudoInverse();
  MatrixXd recoveredProjectionIntoClampsMatrix
      = clampingConstraintMatrixInv * world->getMassMatrix().eval()
        * massedProjectionIntoClampsMatrix * world->getMassMatrix().eval()
        * (1 / world->getTimeStep());

  bool success
      = equals(
            recoveredMassedProjectionIntoClampsMatrix,
            massedProjectionIntoClampsMatrix)
        && equals(
            recoveredProjectionIntoClampsMatrix, projectionIntoClampsMatrix);
  if (!success)
  {
    std::cout << "Classic Projection Into Clamps Matrix: " << std::endl
              << projectionIntoClampsMatrix << std::endl;
    std::cout << "Minv: " << std::endl
              << world->getInvMassMatrix() << std::endl;
    std::cout << "Minv * Clamping Constraint Matrix * Classic Projection Into "
                 "Clamps Matrix * Minv: "
              << std::endl
              << recoveredMassedProjectionIntoClampsMatrix << std::endl;
    std::cout << "Massed Projection Into Clamps Matrix: " << std::endl
              << massedProjectionIntoClampsMatrix << std::endl;
    std::cout << "Recovered Projection Into Clamps Matrix: " << std::endl
              << recoveredProjectionIntoClampsMatrix << std::endl;
    return false;
  }

  return true;
}

bool verifyWorldClassicVelVelJacobian(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicVelVelJacobian forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getVelVelJacobian();
  MatrixXd bruteForce = classicPtr->finiteDifferenceVelVelJacobian();

  if (!equals(analytical, bruteForce, 1e-1))
  {
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyWorldClassicForceVelJacobian(
    WorldPtr world, VectorXd proposedVelocities)
{
  world->setVelocities(proposedVelocities);
  neural::BackpropSnapshotPtr classicPtr
      = neural::forwardPass(world, neural::GradientMode::CLASSIC, true);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicForceVelJacobian forwardPass returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  MatrixXd analytical = classicPtr->getForceVelJacobian();
  MatrixXd bruteForce = classicPtr->finiteDifferenceForceVelJacobian();

  if (!equals(analytical, bruteForce, 1e-1))
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

////////////////////////////////////////////////////////////////////////////////
// Testing methods
////////////////////////////////////////////////////////////////////////////////

/**
 * This tests that A_c is being computed correctly, by checking that
 * mapper = A_c.pinv().transpose() * A_c.transpose() does the right thing to a
 * given set of joint velocities. Namely, it maps proposed joint velocities into
 * just the component of motion that's violating the constraints. If we subtract
 * out that components and re-run the solver, we should see no constraint
 * forces.
 */
bool verifyClampingConstraintMatrix(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  skel->setVelocities(proposedVelocities);
  // Run a solver, to compute the constraint matrix as a byproduct
  world->getConstraintSolver()->solve();
  MatrixXd clampingConstraintMatrix = skel->getClampingConstraintMatrix();
  MatrixXd clampingConstraintMatrixPinv
      = clampingConstraintMatrix.completeOrthogonalDecomposition()
            .pseudoInverse();
  MatrixXd mapper = clampingConstraintMatrixPinv.eval().transpose()
                    * clampingConstraintMatrix.transpose();
  VectorXd violationVelocities = mapper * proposedVelocities;
  VectorXd cleanVelocities = proposedVelocities - violationVelocities;
  skel->setVelocities(cleanVelocities);
  world->getConstraintSolver()->solve();
  VectorXd cleanConstraintForces = skel->getConstraintForces();

  VectorXd zero = VectorXd::Zero(cleanVelocities.size());
  if (!equals(cleanConstraintForces, zero, 1e-3))
  {
    debugDofs(skel);
    std::cout << "Original velocities: " << std::endl
              << proposedVelocities << std::endl;
    std::cout << "Original constraint forces: " << std::endl
              << skel->getConstraintForces() << std::endl;
    std::cout << "A_c: " << std::endl << clampingConstraintMatrix << std::endl;
    std::cout << "A_c.pinv(): " << std::endl
              << clampingConstraintMatrixPinv << std::endl;
    std::cout << "A_c.pinv().transpose(): " << std::endl
              << clampingConstraintMatrixPinv.eval().transpose() << std::endl;
    std::cout << "A_c.transpose(): " << std::endl
              << clampingConstraintMatrix.transpose() << std::endl;
    std::cout << "Mapper: " << std::endl << mapper << std::endl;
    std::cout << "Violation velocities: " << std::endl
              << violationVelocities << std::endl;
    std::cout << "Clean velocities: " << std::endl
              << cleanVelocities << std::endl;
    std::cout << "Clean velocity constraint forces (should be 0): " << std::endl
              << cleanConstraintForces << std::endl;
    std::cout << "Original velocities: " << std::endl
              << proposedVelocities << std::endl;
    std::cout << "Original constraint forces: " << std::endl
              << skel->getConstraintForces() << std::endl;
    std::cout << "A_c: " << std::endl << clampingConstraintMatrix << std::endl;
    std::cout << "A_c.pinv(): " << std::endl
              << clampingConstraintMatrixPinv << std::endl;
    std::cout << "A_c.pinv().transpose(): " << std::endl
              << clampingConstraintMatrixPinv.eval().transpose() << std::endl;
    std::cout << "A_c.transpose(): " << std::endl
              << clampingConstraintMatrix.transpose() << std::endl;
    std::cout << "Mapper: " << std::endl << mapper << std::endl;
    std::cout << "Violation velocities: " << std::endl
              << violationVelocities << std::endl;
    std::cout << "Clean velocities: " << std::endl
              << cleanVelocities << std::endl;
    std::cout << "Clean velocity constraint forces (should be 0): " << std::endl
              << cleanConstraintForces << std::endl;
    std::cout << "Zero: " << std::endl << zero << std::endl;
    return false;
  }

  return true;
}

bool verifyMassedClampingConstraintMatrix(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  skel->setVelocities(proposedVelocities);
  // Run a solver, to compute the constraint matrix as a byproduct
  world->getConstraintSolver()->solve();
  MatrixXd clampingConstraintMatrix = skel->getClampingConstraintMatrix();
  MatrixXd massedClampingConstraintMatrix
      = skel->getMassedClampingConstraintMatrix();
  MatrixXd recoveredMassedClampingConstraintMatrix
      = skel->getInvMassMatrix() * clampingConstraintMatrix;

  bool success = equals(
      recoveredMassedClampingConstraintMatrix, massedClampingConstraintMatrix);
  if (!success)
  {
    std::cout << "Classic Clamping Constraint Matrix: " << std::endl
              << clampingConstraintMatrix << std::endl;
    std::cout << "MInv: " << std::endl << skel->getInvMassMatrix() << std::endl;
    std::cout << "MInv * Classic Clamping Constraint Matrix: " << std::endl
              << recoveredMassedClampingConstraintMatrix << std::endl;
    std::cout << "Massed Clamping Constraint Matrix: " << std::endl
              << massedClampingConstraintMatrix << std::endl;
    return false;
  }

  return true;
}

bool verifyMassedProjectionIntoClampsMatrix(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  skel->setVelocities(proposedVelocities);
  // Run a solver, to compute the constraint matrix as a byproduct
  world->getConstraintSolver()->solve();
  MatrixXd projectionIntoClampsMatrix
      = skel->getProjectionIntoClampsMatrix(world->getTimeStep());
  MatrixXd A_c = skel->getClampingConstraintMatrix();
  MatrixXd A_ub = skel->getUpperBoundConstraintMatrix();
  MatrixXd E = skel->getUpperBoundMappingMatrix();
  MatrixXd massedProjectionIntoClampsMatrix
      = skel->getMassedProjectionIntoClampsMatrix();
  MatrixXd recoveredMassedProjectionIntoClampsMatrix
      = skel->getInvMassMatrix() * (A_c + A_ub * E) * projectionIntoClampsMatrix
        * skel->getInvMassMatrix() * world->getTimeStep();
  MatrixXd clampingConstraintMatrixInv
      = (A_c + A_ub * E).completeOrthogonalDecomposition().pseudoInverse();
  MatrixXd recoveredProjectionIntoClampsMatrix
      = clampingConstraintMatrixInv * skel->getMassMatrix().eval()
        * massedProjectionIntoClampsMatrix * skel->getMassMatrix().eval()
        * (1 / world->getTimeStep());

  bool success
      = equals(
            recoveredMassedProjectionIntoClampsMatrix,
            massedProjectionIntoClampsMatrix)
        && equals(
            recoveredProjectionIntoClampsMatrix, projectionIntoClampsMatrix);
  if (!success)
  {
    std::cout << "Classic Projection Into Clamps Matrix: " << std::endl
              << projectionIntoClampsMatrix << std::endl;
    std::cout << "Minv: " << std::endl << skel->getInvMassMatrix() << std::endl;
    std::cout << "Minv * Clamping Constraint Matrix * Classic Projection Into "
                 "Clamps Matrix * Minv: "
              << std::endl
              << recoveredMassedProjectionIntoClampsMatrix << std::endl;
    std::cout << "Massed Projection Into Clamps Matrix: " << std::endl
              << massedProjectionIntoClampsMatrix << std::endl;
    std::cout << "Recovered Projection Into Clamps Matrix: " << std::endl
              << recoveredProjectionIntoClampsMatrix << std::endl;
    return false;
  }

  return true;
}

bool verifyProjectionIntoClampsMatrix(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  skel->setVelocities(proposedVelocities);
  // Run a solver, to compute the projection matrix as a byproduct
  world->getConstraintSolver()->solve();
  MatrixXd projectionIntoClamps
      = skel->getProjectionIntoClampsMatrix(world->getTimeStep());
  VectorXi mappings = skel->getContactConstraintMappings();
  VectorXd contactConstraintForces
      = skel->getContactConstraintImpluses() / world->getTimeStep();
  VectorXd analyticalConstraintForces
      = -(projectionIntoClamps * proposedVelocities);
  VectorXd analyticalError = VectorXd(analyticalConstraintForces.size());

  // TODO(keenon): This code smells. We should probably put this somewhere else?
  std::size_t pointer = 0;
  for (std::size_t i = 0; i < mappings.size(); i++)
  {
    if (mappings(i) == neural::ConstraintMapping::CLAMPING)
    {
      analyticalError(pointer)
          = contactConstraintForces(i) - analyticalConstraintForces(pointer);
      pointer++;
    }
  }

  VectorXd zero = VectorXd::Zero(analyticalError.size());
  if (!equals(analyticalError, zero, 1e-3))
  {
    debugDofs(skel);
    std::cout << "Proposed velocities: " << std::endl
              << proposedVelocities << std::endl;
    std::cout << "P_c: " << std::endl << projectionIntoClamps << std::endl;
    std::cout << "Constraint forces: " << std::endl
              << contactConstraintForces << std::endl;
    std::cout << "-(P_c * proposedVelocities) (should be the same as above): "
              << std::endl
              << analyticalConstraintForces << std::endl;
    std::cout << "Analytical error (should be zero):" << std::endl
              << analyticalError << std::endl;
    std::cout << "Zero: " << std::endl << zero << std::endl;

    // Recompute step by step

    MatrixXd clampingConstraintMatrix = skel->getClampingConstraintMatrix();
    MatrixXd clampingConstraintMatrixPinv
        = clampingConstraintMatrix.completeOrthogonalDecomposition()
              .pseudoInverse();
    MatrixXd mapper = clampingConstraintMatrixPinv.eval().transpose()
                      * clampingConstraintMatrix.transpose();
    VectorXd violationVelocities = mapper * proposedVelocities;
    VectorXd violationAccel = violationVelocities / world->getTimeStep();
    VectorXd violationTorques = skel->getMassMatrix() * violationAccel;
    VectorXd violationContactForces
        = clampingConstraintMatrixPinv * violationTorques;

    std::cout << "A_c: " << std::endl << clampingConstraintMatrix << std::endl;
    std::cout << "A_c.pinv(): " << std::endl
              << clampingConstraintMatrixPinv << std::endl;
    std::cout << "A_c * A_c.pinv(): " << std::endl
              << clampingConstraintMatrix * clampingConstraintMatrixPinv
              << std::endl;
    std::cout << "A_c.pinv().transpose(): " << std::endl
              << clampingConstraintMatrixPinv.eval().transpose() << std::endl;
    std::cout << "A_c.transpose(): " << std::endl
              << clampingConstraintMatrix.transpose() << std::endl;
    std::cout << "Mapper: " << std::endl << mapper << std::endl;
    std::cout << "Violation velocities: " << std::endl
              << violationVelocities << std::endl;
    std::cout << "Violation accel: " << std::endl
              << violationAccel << std::endl;
    std::cout << "Violation torques: " << std::endl
              << violationTorques << std::endl;
    std::cout << "Violation contact forces: " << std::endl
              << violationContactForces << std::endl;
    return false;
  }

  return true;
}

bool verifyVelVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  MatrixXd bruteForce
      = finiteDifferenceVelVelJacobian(skel, world, proposedVelocities);

  skel->setVelocities(proposedVelocities);
  world->getConstraintSolver()->solve();
  MatrixXd analytical = skel->getVelVelJacobian(world->getTimeStep());

  if (!equals(analytical, bruteForce, 1e-1))
  {
    debugDofs(skel);
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyMassedVelVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  MatrixXd bruteForce
      = finiteDifferenceVelVelJacobian(skel, world, proposedVelocities);

  skel->setVelocities(proposedVelocities);
  world->getConstraintSolver()->solve();
  MatrixXd analytical = skel->getMassedVelVelJacobian();

  if (!equals(analytical, bruteForce, 1e-1))
  {
    debugDofs(skel);
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    MatrixXd classicAnalytical = skel->getVelVelJacobian(world->getTimeStep());
    std::cout
        << "Classic Analytical velVelJacobian (should be the same as above):"
        << std::endl
        << classicAnalytical << std::endl;
    std::cout << "Mass Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    return false;
  }

  return true;
}

bool verifyForceVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  MatrixXd bruteForce
      = finiteDifferenceForceVelJacobian(skel, world, proposedVelocities);

  skel->setVelocities(proposedVelocities);
  world->getConstraintSolver()->solve();
  MatrixXd analytical = skel->getForceVelJacobian(world->getTimeStep());

  if (!equals(analytical, bruteForce, 1e-3))
  {
    debugDofs(skel);
    std::cout << "Analytical forceVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << bruteForce << std::endl;
    return false;
  }

  return true;
}

bool verifyMassedForceVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd proposedVelocities)
{
  MatrixXd bruteForce
      = finiteDifferenceForceVelJacobian(skel, world, proposedVelocities);

  skel->setVelocities(proposedVelocities);
  world->getConstraintSolver()->solve();
  MatrixXd analytical = skel->getMassedForceVelJacobian(world->getTimeStep());

  if (!equals(analytical, bruteForce, 1e-3))
  {
    debugDofs(skel);
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << bruteForce << std::endl;
    MatrixXd classicAnalytical
        = skel->getForceVelJacobian(world->getTimeStep());
    std::cout
        << "Classic Analytical forceVelJacobian (should be the same as above):"
        << std::endl
        << classicAnalytical << std::endl;
    std::cout
        << "Mass Analytical forceVelJacobian (should be the same as above):"
        << std::endl
        << analytical << std::endl;
    return false;
  }

  return true;
}

// TODO(keenon): This test is broken, but also very difficult to interpret, so
// leaving it for now. Should ensure this test passes eventually.
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

  EXPECT_TRUE(verifyClampingConstraintMatrix(pendulum, world, timestepVel));
  EXPECT_TRUE(verifyProjectionIntoClampsMatrix(pendulum, world, timestepVel));
  EXPECT_TRUE(verifyVelVelJacobian(pendulum, world, timestepVel));
  EXPECT_TRUE(verifyForceVelJacobian(pendulum, world, timestepVel));
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
  VectorXd worldVel = world->getVelocities();
  VectorXd timestepVel = box->getVelocities();
  VectorXd timestepWorldVel = world->getVelocities();

  // Test the classic formulation
  EXPECT_TRUE(verifyWorldClassicClampingConstraintMatrix(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicProjectionIntoClampsMatrix(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicVelVelJacobian(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicForceVelJacobian(world, worldVel));
  /*
  EXPECT_TRUE(verifyClampingConstraintMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyProjectionIntoClampsMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyVelVelJacobian(box, world, timestepVel));
  EXPECT_TRUE(verifyForceVelJacobian(box, world, timestepVel));

  // Test the massed formulation
  EXPECT_TRUE(verifyMassedClampingConstraintMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedProjectionIntoClampsMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedVelVelJacobian(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedForceVelJacobian(box, world, timestepVel));
  */
}

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

  world->getConstraintSolver()->setGradientMode(neural::GradientMode::CLASSIC);
  world->getConstraintSolver()->solve();

  EXPECT_TRUE(verifyWorldClassicClampingConstraintMatrix(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicProjectionIntoClampsMatrix(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicVelVelJacobian(world, worldVel));
  EXPECT_TRUE(verifyWorldClassicForceVelJacobian(world, worldVel));
  /*
  EXPECT_TRUE(verifyWorldMassedClampingConstraintMatrix(world, worldVel));
  EXPECT_TRUE(verifyWorldMassedProjectionIntoClampsMatrix(world, worldVel));
  */
  // EXPECT_TRUE(verifyClampingConstraintMatrix(leftBox, world, leftBoxVel));

  /*
  EXPECT_TRUE(verifyProjectionIntoClampsMatrix(leftBox, world, leftBoxVel));
  EXPECT_TRUE(verifyVelVelJacobian(leftBox, world, leftBoxVel));
  EXPECT_TRUE(verifyForceVelJacobian(leftBox, world, leftBoxVel));

  // Test the massed formulation
  EXPECT_TRUE(verifyMassedClampingConstraintMatrix(leftBox, world, leftBoxVel));
  EXPECT_TRUE(
      verifyMassedProjectionIntoClampsMatrix(leftBox, world, leftBoxVel));
  EXPECT_TRUE(verifyMassedVelVelJacobian(leftBox, world, leftBoxVel));
  EXPECT_TRUE(verifyMassedForceVelJacobian(leftBox, world, leftBoxVel));
  */
}

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
  boxBody->addExtForce(Eigen::Vector3d(1, 0, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);
  boxBody->setRestitutionCoeff(0.5);
  box->setVelocity(1, -10);

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
  floorBody->setRestitutionCoeff(0.5);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  VectorXd timestepVel = box->getVelocities();

  // Test the classic formulation
  EXPECT_TRUE(verifyClampingConstraintMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyProjectionIntoClampsMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyVelVelJacobian(box, world, timestepVel));
  EXPECT_TRUE(verifyForceVelJacobian(box, world, timestepVel));

  // Test the massed formulation
  EXPECT_TRUE(verifyMassedClampingConstraintMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedProjectionIntoClampsMatrix(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedVelVelJacobian(box, world, timestepVel));
  EXPECT_TRUE(verifyMassedForceVelJacobian(box, world, timestepVel));
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
  VectorXd timestepVel = reversePendulumSled->getVelocities();

  // Test the classic formulation
  EXPECT_TRUE(
      verifyClampingConstraintMatrix(reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(verifyProjectionIntoClampsMatrix(
      reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(verifyVelVelJacobian(reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(verifyForceVelJacobian(reversePendulumSled, world, timestepVel));

  // Test the massed formulation
  EXPECT_TRUE(verifyMassedClampingConstraintMatrix(
      reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(verifyMassedProjectionIntoClampsMatrix(
      reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(
      verifyMassedVelVelJacobian(reversePendulumSled, world, timestepVel));
  EXPECT_TRUE(
      verifyMassedForceVelJacobian(reversePendulumSled, world, timestepVel));
}

/*
TEST(GRADIENTS, SLIDING_REVERSE_PENDULUM_NO_FRICTION)
{
  testReversePendulumSledWithFrictionCoeff(0);
}
*/