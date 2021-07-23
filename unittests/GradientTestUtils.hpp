#ifndef GRADIENT_TEST_UTILS
#define GRADIENT_TEST_UTILS
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
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/MappedBackpropSnapshot.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

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
    WorldPtr world, VectorXs proposedVelocities)
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

  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  if (A_c.size() == 0)
  {
    // this means that there's no clamping contacts
    return true;
  }

  Eigen::MatrixXs A_cInv;
  if (A_c.size() > 0)
  {
    A_cInv = A_c.completeOrthogonalDecomposition().pseudoInverse();
  }
  else
  {
    A_cInv = Eigen::MatrixXs::Zero(0, 0);
  }
  MatrixXs mapper = A_cInv.eval().transpose() * A_c.transpose();
  VectorXs violationVelocities = mapper * proposedVelocities;
  VectorXs cleanVelocities = proposedVelocities - violationVelocities;

  world->setVelocities(cleanVelocities);
  // Populate the constraint matrices, without taking a time step or integrating
  // velocities
  world->getConstraintSolver()->setGradientEnabled(true);
  world->getConstraintSolver()->solve(world.get());

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    std::shared_ptr<ConstrainedGroupGradientMatrices> grad
        = skel->getGradientConstraintMatrices();
    if (!grad)
      continue;

    VectorXs cleanContactImpulses = grad->getContactConstraintImpulses();
    VectorXs zero = VectorXs::Zero(cleanContactImpulses.size());
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
    WorldPtr world, VectorXs proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs V_c = classicPtr->getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXs M = classicPtr->getMassMatrix(world);
  Eigen::MatrixXs Minv = classicPtr->getInvMassMatrix(world);

  Eigen::MatrixXs A_c_recovered = M * V_c;
  Eigen::MatrixXs V_c_recovered = Minv * A_c;

  if (!equals(A_c, A_c_recovered, 1e-8) || !equals(V_c, V_c_recovered, 1e-8))
  {
    std::cout << "A_c massed check failed" << std::endl;
    std::cout << "A_c: " << std::endl << A_c << std::endl;
    std::cout << "A_c recovered = M * V_c: " << std::endl
              << A_c_recovered << std::endl;
    Eigen::MatrixXs diff = A_c - A_c_recovered;
    std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
        = classicPtr->getClampingConstraints();
    for (int i = 0; i < diff.cols(); i++)
    {
      Eigen::VectorXs diffCol = diff.col(i);
      if (diffCol.norm() > 1e-8)
      {
        std::cout << "Disagreement on column " << i << std::endl;
        std::cout << "Diff: " << std::endl << diffCol << std::endl;
        std::shared_ptr<DifferentiableContactConstraint> constraint
            = constraints[i];
        std::cout << "Contact type: " << constraint->getContactType()
                  << std::endl;
        Eigen::VectorXs worldPos = constraint->getContactWorldPosition();
        std::cout << "Contact pos: " << std::endl << worldPos << std::endl;
        Eigen::VectorXs worldNormal = constraint->getContactWorldNormal();
        std::cout << "Contact normal: " << std::endl
                  << worldNormal << std::endl;

        assert(diffCol.size() == world->getNumDofs());
        for (int j = 0; j < world->getNumDofs(); j++)
        {
          if (abs(diffCol(j)) > 1e-8)
          {
            std::cout << "Error at DOF " << j << " ("
                      << world->getDofs()[j]->getName() << "): " << diffCol(j)
                      << std::endl;
          }
        }
      }
    }
    /*
    std::cout << "V_c: " << std::endl << V_c << std::endl;
    std::cout << "V_c recovered = Minv * A_c: " << std::endl
              << V_c_recovered << std::endl;
    std::cout << "Diff: " << std::endl << V_c - V_c_recovered << std::endl;
    */
    return false;
  }

  return true;
}

/**
 * This verifies the massed formulation by verifying its relationship to the
 * classic formulation.
 */
bool verifyMassedUpperBoundConstraintMatrix(
    WorldPtr world, VectorXs proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs V_ub = classicPtr->getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs M = classicPtr->getMassMatrix(world);
  Eigen::MatrixXs Minv = classicPtr->getInvMassMatrix(world);

  Eigen::MatrixXs A_ub_recovered = M * V_ub;
  Eigen::MatrixXs V_ub_recovered = Minv * A_ub;

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
    WorldPtr world, VectorXs proposedVelocities)
{
  world->setVelocities(proposedVelocities);

  // Compute classic and massed formulation of the backprop snapshot. The "true"
  // as the last argument says do this in an idempotent way, so leave the world
  // state unchanged in computing these backprop snapshots.

  bool oldPenetrationCorrection = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  world->setPenetrationCorrectionEnabled(oldPenetrationCorrection);

  if (!classicPtr)
  {
    std::cout << "verifyWorldClassicProjectionIntoClampsMatrix forwardPass "
                 "returned a "
                 "null BackpropSnapshotPtr for GradientMode::CLASSIC!"
              << std::endl;
    return false;
  }

  // Compute the offset required from the penetration correction velocities

  VectorXs penetrationCorrectionVelocities
      = classicPtr->getPenetrationCorrectionVelocities();
  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs V_c = classicPtr->getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXs V_ub = classicPtr->getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();
  s_t dt = world->getTimeStep();
  // Eigen::MatrixXs b = classicPtr->getClampingConstraintRelativeVels();
  Eigen::MatrixXs constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXs forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXs velToForce = Eigen::MatrixXs::Zero(0, 0);
  VectorXi mappings = classicPtr->getContactConstraintMappings();
  if (forceToVel.size() > 0)
  {
    velToForce = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  }
  VectorXs penetrationOffset
      = (velToForce * penetrationCorrectionVelocities) / world->getTimeStep();

  // Get the actual constraint forces

  VectorXs fReal
      = classicPtr->getContactConstraintImpulses() / world->getTimeStep();
  VectorXs f_cReal = Eigen::VectorXs::Zero(A_c.cols());
  std::size_t pointer = 0;
  for (std::size_t i = 0; i < mappings.size(); i++)
  {
    if (mappings(i) == neural::ConstraintMapping::CLAMPING)
    {
      f_cReal(pointer) = fReal(i);
      pointer++;
    }
  }

  // Compute the analytical constraint forces, which should match our actual
  // constraint forces. We center the approximation around our old solution,
  // because A_c can be low-rank and so many solutions are possible.

  MatrixXs P_c = classicPtr->getProjectionIntoClampsMatrix(world);
  VectorXs contactVel
      = (-1.0 / dt) * A_c.transpose() * classicPtr->getPreConstraintVelocity();
  VectorXs contactVelFromForce = forceToVel * f_cReal;
  VectorXs centeredVel = contactVel - contactVelFromForce;
  VectorXs eps = -1 * velToForce * centeredVel;
  VectorXs f_c = f_cReal + eps;

  // Sum the two constraints forces together

  VectorXs f_cPrime = f_c + penetrationOffset;

  // The analytical constraint forces are a shorter vector than the actual
  // constraint forces, since the analytical constraint forces are only
  // computing the constraints that are clamping. So we need to check equality
  // while taking into account that mapping.

  VectorXs analyticalError = VectorXs(f_c.size());
  pointer = 0;
  for (std::size_t i = 0; i < mappings.size(); i++)
  {
    if (mappings(i) == neural::ConstraintMapping::CLAMPING)
    {
      analyticalError(pointer) = fReal(i) - f_cPrime(pointer);
      pointer++;
    }
  }

  // Check that the analytical error is zero

  VectorXs zero = VectorXs::Zero(analyticalError.size());
  // We can crunch down this error by making the
  // BoxedLcpConstraintSolver::makeHyperAccurateAndVerySlow() even more
  // aggressive in tightening errors on the LCP, but that's very slow.
  if (!equals(analyticalError, zero, 3e-7))
  {
    std::cout << "Error in verifyClassicProjectionIntoClampsMatrix(): "
              << std::endl;

    std::cout << "analyticalError: " << std::endl
              << analyticalError << std::endl;

    Eigen::MatrixXs Q = classicPtr->mGradientMatrices[0]->mA;
    Eigen::MatrixXs A = classicPtr->mGradientMatrices[0]->mAllConstraintMatrix;
    Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
    Eigen::VectorXs bReal = classicPtr->mGradientMatrices[0]->mB;
    // int numContacts = bReal.size();

    /////////////////////////////////////////////////////
    // Checking the forward computation
    /////////////////////////////////////////////////////

    // We expect these to be equal
    Eigen::VectorXs tau = (A_c + A_ub * E) * f_cReal;
    Eigen::VectorXs tauReal = A * fReal;

    Eigen::MatrixXs compareTaus = Eigen::MatrixXs::Zero(tau.size(), 2);
    compareTaus.col(0) = tau;
    compareTaus.col(1) = tauReal;
    std::cout << "(A_c + A_ub * E) * f_cReal  :::   A * fReal" << std::endl
              << compareTaus << std::endl;

    // We expect these to be equal
    Eigen::VectorXs v = A.transpose() * (V_c + V_ub * E) * f_cReal;
    Eigen::VectorXs vReal = Q * fReal;

    Eigen::MatrixXs compareVs = Eigen::MatrixXs::Zero(v.size(), 6);
    compareVs.col(0) = v;
    compareVs.col(1) = vReal;
    compareVs.col(2) = bReal / dt;
    compareVs.col(3) = mappings.cast<s_t>();
    compareVs.col(4) = fReal;
    compareVs.col(5) = vReal - bReal / dt;
    std::cout << "A^T * (V_c + V_ub * E) * f_cReal  :::   Q * fReal  :::   "
                 "bReal / dt  :::   indexType   ::: fReal  ::: next v"
              << std::endl
              << compareVs << std::endl;

    // Why doesn't least squares over Q find the right solution?
    Eigen::FullPivLU<Eigen::MatrixXs> lu_decompQ(Q);
    std::cout << "Dimension of Q: " << Q.rows() << "x" << Q.cols() << std::endl;
    std::cout << "Rank of Q: " << lu_decompQ.rank() << std::endl;
    s_t dist = (vReal - (bReal / dt)).norm();

    Eigen::VectorXs eps
        = Q.completeOrthogonalDecomposition().solve(bReal / dt - Q * fReal);
    Eigen::VectorXs centeredApprox = fReal + eps;
    s_t distApprox = (Q * centeredApprox - (bReal / dt)).norm();

    std::cout << "Dist of fReal: " << dist << std::endl;
    std::cout << "Dist of best approx: " << distApprox << std::endl;
    Eigen::MatrixXs compareSolves = Eigen::MatrixXs::Zero(vReal.size(), 6);
    compareSolves.col(0) = (vReal - (bReal / dt));
    compareSolves.col(1) = (Q * centeredApprox - (bReal / dt));
    compareSolves.col(2) = vReal;
    compareSolves.col(3) = Q * centeredApprox;
    compareSolves.col(4) = fReal;
    compareSolves.col(5) = centeredApprox;
    std::cout << "Q*fReal - bReal ::: Q*Q^{-1}*bReal - bReal ::: Q*fReal ::: "
                 "Q*Q^{-1}*bReal ::: fReal ::: Q^{-1}*bReal"
              << std::endl
              << compareSolves << std::endl;

    // We'd like to try to recover the original forces
    // Eigen::MatrixXs R = A.transpose() * (V_c + V_ub * E);
    Eigen::MatrixXs R = A.transpose() * (V_c + V_ub * E);
    Eigen::FullPivLU<Eigen::MatrixXs> lu_decompR(R);
    std::cout << "Dimension of R: " << R.rows() << "x" << R.cols() << std::endl;
    std::cout << "Rank of R: " << lu_decompR.rank() << std::endl;
    /*
    std::cout << "Here is a matrix whose columns form a basis of the "
                 "null-space of R:\n"
              << lu_decomp.kernel() << std::endl;
    std::cout << "Here is a matrix whose columns form a basis of the "
                 "column-space of R:\n"
              << lu_decomp.image(R) << std::endl;
    */

    Eigen::VectorXs random = Eigen::VectorXs::Random(R.cols());

    Eigen::MatrixXs errorFwd
        = random
          - R.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(R * random);
    Eigen::MatrixXs errorBack
        = random
          - R.transpose()
                * R.transpose()
                      .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                      .solve(random);

    Eigen::MatrixXs compareRecovery = Eigen::MatrixXs::Zero(errorFwd.size(), 2);
    compareRecovery.col(0) = errorBack;
    compareRecovery.col(1) = errorFwd;
    std::cout << "random - R^T*R^{-T}*random ::: random - R^{-1}*R*random"
              << std::endl
              << compareRecovery << std::endl;

    /*
    std::cout << "P_c: " << std::endl << P_c << std::endl;
    std::cout << "bounce: " << std::endl
              << classicPtr->getBounceDiagonals() << std::endl;
    std::cout << "status: " << std::endl;
    for (std::size_t i = 0; i < mappings.size(); i++)
    {
      std::cout << mappings(i) << std::endl;
    }
    */
    /*
    std::cout << "-(P_c * proposedVelocities) (should be the roughly same as
    " "actual constraint forces): "
              << std::endl
              << analyticalConstraintForces << std::endl;
    std::cout << "Penetration correction velocities: " << std::endl
              << penetrationCorrectionVelocities << std::endl;
    std::cout << "(A_c^T(V_c + V_ub*E)).pinv() * correction_vels (should "
                 "account for any errors in above): "
              << std::endl
              << penetrationOffset << std::endl;
    */

    // Show the constraint forces at each of the contact points

    /*
    Eigen::MatrixXs comparison
        = Eigen::MatrixXs::Zero(contactConstraintForces.size(), 2);
    comparison.col(0) = contactConstraintForces;
    comparison.col(1) = mappedAnalyticalConstraintForces;
    std::cout
        << "Real clamping constraint forces - Analytical constraint forces:
    "
        << std::endl
        << comparison << std::endl;

    // Show the contact accels

    Eigen::VectorXs realContactAccels
        = classicPtr->mGradientMatrices[0]->mA * contactConstraintForces;
    Eigen::VectorXs analyticalContactAccels
        = classicPtr->mGradientMatrices[0]->mA
          * mappedAnalyticalConstraintForces;
    Eigen::MatrixXs jointTorqueComparison
        = Eigen::MatrixXs::Zero(realContactAccels.size(), 9);
    s_t dt = world->getTimeStep();
    jointTorqueComparison.col(0) = realContactAccels;
    jointTorqueComparison.col(1) = analyticalContactAccels;
    jointTorqueComparison.col(2) = mappings.cast<s_t>();
    jointTorqueComparison.col(3) = classicPtr->mGradientMatrices[0]->mX;
    jointTorqueComparison.col(4) = classicPtr->mGradientMatrices[0]->mA
                                   * classicPtr->mGradientMatrices[0]->mX;
    jointTorqueComparison.col(5) = classicPtr->mGradientMatrices[0]->mA
                                       * classicPtr->mGradientMatrices[0]->mX
                                   - classicPtr->mGradientMatrices[0]->mB;
    jointTorqueComparison.col(6) = mappedAnalyticalConstraintForces * dt;
    jointTorqueComparison.col(7) = classicPtr->mGradientMatrices[0]->mA
                                   * mappedAnalyticalConstraintForces * dt;
    jointTorqueComparison.col(8) = classicPtr->mGradientMatrices[0]->mA
                                       * mappedAnalyticalConstraintForces *
    dt
                                   - classicPtr->mGradientMatrices[0]->mB;
    std::cout << "Real contact accels - Analytical contact accels - type -
    mX "
                 "- mA*mX - mA*mX - mB - aX - mA*aX - mA*aX - mB: "
              << std::endl
              << jointTorqueComparison << std::endl;
    */

    // Show the resulting joint accelerations

    /*
    Eigen::VectorXs realAccel
        = world->getInvMassMatrix() * contactConstraintForces;
    Eigen::VectorXs analyticalAccel
        = world->getInvMassMatrix() * mappedAnalyticalConstraintForces;
    Eigen::MatrixXs accelComparison
        = Eigen::MatrixXs::Zero(realAccel.size(), 2);
    accelComparison.col(0) = realAccel;
    accelComparison.col(1) = analyticalAccel;
    std::cout << "Real accel - Analytical accel: " << std::endl
              << accelComparison << std::endl;

    std::cout << "Analytical error (should be zero):" << std::endl
              << analyticalError << std::endl;
    */

    return false;
  }

  return true;
}

/**
 * This tests that P_c is getting computed correctly.
 */
bool verifyMassedProjectionIntoClampsMatrix(
    WorldPtr world, VectorXs proposedVelocities)
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

  Eigen::MatrixXs P_c = classicPtr->getProjectionIntoClampsMatrix(world);

  // Reconstruct P_c without the massed shortcut
  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();

  Eigen::MatrixXs constraintForceToImpliedTorques = A_c + (A_ub * E);
  Eigen::MatrixXs forceToVel = A_c.eval().transpose()
                               * classicPtr->getInvMassMatrix(world)
                               * constraintForceToImpliedTorques;
  Eigen::MatrixXs velToForce = Eigen::MatrixXs::Zero(0, 0);
  if (forceToVel.size() > 0)
  {
    velToForce = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  }
  Eigen::MatrixXs bounce = classicPtr->getBounceDiagonals().asDiagonal();
  Eigen::MatrixXs P_c_recovered
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

bool verifyVelVelJacobian(WorldPtr world, VectorXs proposedVelocities)
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

  MatrixXs analytical = classicPtr->getVelVelJacobian(world);
  MatrixXs bruteForce = classicPtr->finiteDifferenceVelVelJacobian(world);

  // atlas run as 1.6e-08 error
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force velVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical velVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    std::cout << "Diff range:" << std::endl
              << (analytical - bruteForce).minCoeff() << " to "
              << (analytical - bruteForce).maxCoeff() << std::endl;
    std::cout << "Brute force velCJacobian:" << std::endl
              << classicPtr->getJacobianOfC(world, WithRespectTo::VELOCITY)
              << std::endl;
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << classicPtr->getControlForceVelJacobian(world) << std::endl;
    std::cout << "Brute force forceVelJacobian * velCJacobian:" << std::endl
              << classicPtr->getControlForceVelJacobian(world)
                     * classicPtr->getJacobianOfC(
                         world, WithRespectTo::VELOCITY)
              << std::endl;
    return false;
  }

  return true;
}

bool verifyAnalyticalConstraintMatrixEstimates(WorldPtr world)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::MatrixXs original = classicPtr->getClampingConstraintMatrix(world);

  s_t EPS = 1e-5;

  for (int i = 0; i < 100; i++)
  {
    Eigen::VectorXs diff = Eigen::VectorXs::Random(world->getNumDofs());

    Eigen::MatrixXs analyticalPos;
    Eigen::MatrixXs analyticalNeg;
    Eigen::MatrixXs bruteForcePos;
    Eigen::MatrixXs bruteForceNeg;

    s_t epsPos = EPS;
    while (true)
    {
      Eigen::VectorXs pos = world->getPositions() + diff * epsPos;

      Eigen::MatrixXs analytical
          = classicPtr->estimateClampingConstraintMatrixAt(world, pos);
      Eigen::MatrixXs bruteForce
          = classicPtr->getClampingConstraintMatrixAt(world, pos);

      if (bruteForce.cols() == original.cols()
          && bruteForce.rows() == original.rows())
      {
        analyticalPos = analytical;
        bruteForcePos = bruteForce;
        break;
      }
      epsPos *= 0.5;
    }

    s_t epsNeg = EPS;
    while (true)
    {
      Eigen::VectorXs pos = world->getPositions() + diff * epsNeg;

      Eigen::MatrixXs analytical
          = classicPtr->estimateClampingConstraintMatrixAt(world, pos);
      Eigen::MatrixXs bruteForce
          = classicPtr->getClampingConstraintMatrixAt(world, pos);

      if (bruteForce.cols() == original.cols()
          && bruteForce.rows() == original.rows())
      {
        analyticalNeg = analytical;
        bruteForceNeg = bruteForce;
        break;
      }
      epsNeg *= 0.5;
    }

    Eigen::MatrixXs analyticalDiff
        = (analyticalPos - analyticalNeg) / (epsPos + epsNeg);
    Eigen::MatrixXs bruteForceDiff
        = (bruteForcePos - bruteForceNeg) / (epsPos + epsNeg);

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

// This verifies that f_c changes in predictable ways as we perturb the
// positions
bool verifyPerturbedF_c(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  // assert(classicPtr->areResultsStandardized());

  Eigen::VectorXs original = world->getPositions();
  const s_t EPS = 1e-6;
  std::vector<std::pair<int, s_t>> testPairs;
  for (int i = 0; i < world->getNumDofs(); i++)
  {
    testPairs.emplace_back(i, EPS);
    testPairs.emplace_back(i, -EPS);
  }

  // Perturb the world positions by a random amount
  for (auto pair : testPairs)
  {
    int i = pair.first;
    Eigen::VectorXs perturbedPos = original;
    s_t eps = pair.second;

    neural::BackpropSnapshotPtr perturbedPtr;

    while (true)
    {
      snapshot.restore();
      perturbedPos = original;
      perturbedPos(i) += eps;
      world->setPositions(perturbedPos);
      perturbedPtr = neural::forwardPass(world, true);
      if ((!classicPtr->areResultsStandardized()
           || perturbedPtr->areResultsStandardized())
          && perturbedPtr->getNumClamping() == classicPtr->getNumClamping()
          && perturbedPtr->getNumUpperBound() == classicPtr->getNumUpperBound())
      {
        break;
      }
      std::cout << "Perturbing joint " << i << " by " << eps
                << " crossed a discontinuity" << std::endl;
      std::cout << "Perturbing standardized: "
                << perturbedPtr->areResultsStandardized() << std::endl;
      std::cout << "Perturbing num clamping: " << perturbedPtr->getNumClamping()
                << std::endl;
      std::cout << "Original num clamping: " << classicPtr->getNumClamping()
                << std::endl;
      std::cout << "Perturbing num upper bound: "
                << perturbedPtr->getNumUpperBound() << std::endl;
      std::cout << "Original num upper bound: "
                << classicPtr->getNumUpperBound() << std::endl;
      eps *= 0.5;
      if (abs(eps) < 1e-15)
      {
        std::cout << "Couldn't find a numerically stable epsilon small enough "
                     "to satisfy criteria when finite differencing. Maybe the "
                     "test is exactly at a "
                     "non-differentiable point?"
                  << std::endl;
        return false;
      }
    }

    Eigen::MatrixXs realA_c
        = perturbedPtr->getMassMatrix(world)
          * perturbedPtr->getMassedClampingConstraintMatrix(world);
    // realA_c = perturbedPtr->getClampingConstraintMatrix(world);
    Eigen::MatrixXs realA_ub
        = perturbedPtr->getUpperBoundConstraintMatrix(world);
    Eigen::MatrixXs realE = perturbedPtr->getUpperBoundMappingMatrix();
    Eigen::MatrixXs realQ = perturbedPtr->getClampingAMatrix();
    Eigen::VectorXs realB = perturbedPtr->getClampingConstraintRelativeVels();
    Eigen::VectorXs realV = perturbedPtr->getPreConstraintVelocity();

    Eigen::MatrixXs A_c
        = classicPtr->estimateClampingConstraintMatrixAt(world, perturbedPos);
    Eigen::MatrixXs A_ub
        = classicPtr->estimateUpperBoundConstraintMatrixAt(world, perturbedPos);
    Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();

    if (A_c.cols() > 0 && !equals(realA_c, A_c, 1e-10))
    {
      /*
      std::cout << "Failed perturb " << i << " by " << eps << std::endl;
      std::cout << "Jac[0]:" << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getConstraintForcesJacobian(world)
                << std::endl;
      std::cout << "Contact force Jac[0]:" << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getContactForceJacobian(world)
                << std::endl;
      std::cout << "Contact pos Jac[0]:" << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getContactPositionJacobian(world)
                << std::endl;
      std::cout << "Contact force dir Jac[0]:" << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getContactForceDirectionJacobian(world)
                << std::endl;
      std::cout << "Contact force dir Jac[0] 10:" << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getContactForceGradient(world->getDofs()[10])
                << std::endl;
      */
      /*
      std::cout << "Contact getScrewAxisForForceGradient(2,10) for Jac[0]:"
                << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getScrewAxisForForceGradient(
                           world->getDofs()[2], world->getDofs()[10])
                << std::endl;
      std::cout << "Contact getScrewAxisForForceGradient(4,10) for Jac[0]:"
                << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getScrewAxisForForceGradient(
                           world->getDofs()[4], world->getDofs()[10])
                << std::endl;
      std::cout << "Contact getScrewAxisForForceGradient(8,10) for Jac[0]:"
                << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getScrewAxisForForceGradient(
                           world->getDofs()[8], world->getDofs()[10])
                << std::endl;
      std::cout << "Contact getScrewAxisForForceGradient(10,10) for Jac[0]:"
                << std::endl
                << classicPtr->getClampingConstraints()[0]
                       ->getScrewAxisForForceGradient(
                           world->getDofs()[10], world->getDofs()[10])
                << std::endl;
      */
      if (realA_c.cols() >= 6 && realA_c.rows() >= 6)
      {
        std::cout << "Real A_c (top-left 6x6):" << std::endl
                  << realA_c.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical A_c (top-left 6x6):" << std::endl
                  << A_c.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff A_c (top-left 6x6):" << std::endl
                  << (realA_c - A_c).block<6, 6>(0, 0) << std::endl;
      }
      else
      {
        std::cout << "Real A_c:" << std::endl << realA_c << std::endl;
        std::cout << "Analytical A_c:" << std::endl << A_c << std::endl;
        std::cout << "Diff A_c:" << std::endl << (realA_c - A_c) << std::endl;
      }
      std::cout << "Diff A_c range:" << std::endl
                << (realA_c - A_c).minCoeff() << " to "
                << (realA_c - A_c).maxCoeff() << std::endl;
      return false;
    }

    Eigen::MatrixXs analyticalQ = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
    classicPtr->computeLCPConstraintMatrixClampingSubset(
        world, analyticalQ, A_c, A_ub, E);
    Eigen::VectorXs analyticalB = Eigen::VectorXs(A_c.cols());
    classicPtr->computeLCPOffsetClampingSubset(world, analyticalB, A_c);

    if (!equals(realB, analyticalB, 1e-9))
    {
      std::cout << "Real B perturb[" << i << "] += " << pair.second << ":"
                << std::endl
                << realB << std::endl;
      std::cout << "Analytical B perturb[" << i << "] += " << pair.second << ":"
                << std::endl
                << analyticalB << std::endl;
      classicPtr->computeLCPOffsetClampingSubset(world, analyticalB, A_c);
      std::cout << "Diff B perturb[" << i << "] += " << pair.second << ":"
                << std::endl
                << (realB - analyticalB) << std::endl;
      std::cout << "Diff range:" << std::endl
                << (realB - analyticalB).minCoeff() << " to "
                << (realB - analyticalB).maxCoeff() << std::endl;
      return false;
    }

    if (A_ub.cols() == 0 && !equals(realQ, analyticalQ, 1e-10))
    {
      if (realQ.rows() >= 6)
      {
        std::cout << "Real Q (top-left 6x6):" << std::endl
                  << realQ.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical Q (top-left 6x6):" << std::endl
                  << analyticalQ.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Q (top-left 6x6):" << std::endl
                  << (realQ - analyticalQ).block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Q range:" << std::endl
                  << (realQ - analyticalQ).minCoeff() << " to "
                  << (realQ - analyticalQ).maxCoeff() << std::endl;
      }
      else
      {
        // TODO(JS): There is cases that the size of realQ is smaller than 6x6
        std::cout << "Real Q:" << std::endl << realQ << std::endl;
        std::cout << "Analytical Q:" << std::endl << analyticalQ << std::endl;
        std::cout << "Diff Q:" << std::endl
                  << (realQ - analyticalQ) << std::endl;
        std::cout << "Diff Q range:" << std::endl
                  << (realQ - analyticalQ).minCoeff() << " to "
                  << (realQ - analyticalQ).maxCoeff() << std::endl;
      }
      return false;
    }

    if (A_c.cols() > 0)
    {
      Eigen::MatrixXs analyticalQinv
          = analyticalQ.completeOrthogonalDecomposition().pseudoInverse();
      Eigen::MatrixXs realQinv
          = realQ.completeOrthogonalDecomposition().pseudoInverse();

      // The inverted Q's can have enormous terms (like 5000), which means we
      // need to scale error bounds appropriately
      if (A_ub.cols() == 0
          && !equals(
              realQinv,
              analyticalQinv,
              1e-10
                  * std::max(
                      abs(realQinv.maxCoeff()), abs(realQinv.minCoeff()))))
      {
        if (realQ.rows() >= 6)
        {
          std::cout << "Real Qinv (top-left 6x6):" << std::endl
                    << realQinv.block<6, 6>(0, 0) << std::endl;
          std::cout << "Analytical Qinv (top-left 6x6):" << std::endl
                    << analyticalQinv.block<6, 6>(0, 0) << std::endl;
          std::cout << "Diff Qinv (top-left 6x6):" << std::endl
                    << (realQinv - analyticalQinv).block<6, 6>(0, 0)
                    << std::endl;
          std::cout << "Diff Qinv range:" << std::endl
                    << (realQinv - analyticalQinv).minCoeff() << " to "
                    << (realQinv - analyticalQinv).maxCoeff() << std::endl;
          return false;
        }
        else
        {
          std::cout << "Real Qinv:" << std::endl << realQinv << std::endl;
          std::cout << "Analytical Qinv:" << std::endl
                    << analyticalQinv << std::endl;
          std::cout << "Diff Qinv:" << std::endl
                    << (realQinv - analyticalQinv) << std::endl;
          std::cout << "Diff Qinv range:" << std::endl
                    << (realQinv - analyticalQinv).minCoeff() << " to "
                    << (realQinv - analyticalQinv).maxCoeff() << std::endl;
          return false;
        }
      }

      /*
      std::cout << "Classic Vel: " << std::endl
                << classicPtr->getPreConstraintVelocity() << std::endl;
      std::cout << "Perturbed Vel: " << std::endl
                << perturbedPtr->getPreConstraintVelocity() << std::endl;
      for (auto clampingConstraint : perturbedPtr->getClampingConstraints())
      {
        std::cout << "Contact pos: " << std::endl
                  << clampingConstraint->getContactWorldPosition() << std::endl;
        std::cout << "Contact normal: " << std::endl
                  << clampingConstraint->getContactWorldNormal() << std::endl;
        std::cout << "Contact force direction: " << std::endl
                  << clampingConstraint->getContactWorldForceDirection()
                  << std::endl;
      }
      */

      Eigen::VectorXs analyticalF_c
          = analyticalQ.completeOrthogonalDecomposition().solve(analyticalB);
      Eigen::VectorXs realF_c = perturbedPtr->getClampingConstraintImpulses();
      // realQ == Q only when A_ub is empty
      if (A_ub.cols() == 0)
      {
        Eigen::VectorXs cleanRealF_c
            = realQ.completeOrthogonalDecomposition().solve(realB);
        /// These don't have to be numerically equivalent when there are
        /// multiple collision islands
        if (classicPtr->areResultsStandardized()
            && !equals(realF_c, cleanRealF_c, 1e-12))
        {
          std::cout << "f_c isn't exactly the same as a solve on Q and B!"
                    << std::endl;
          std::cout << "f_c:" << std::endl << realF_c << std::endl;
          std::cout << "Qinv*b:" << std::endl << cleanRealF_c << std::endl;
          std::cout << "Diff:" << std::endl
                    << cleanRealF_c - realF_c << std::endl;
          return false;
        }
      }
      if (!equals(analyticalF_c, realF_c, 1e-9))
      {
        Eigen::MatrixXs comparison = Eigen::MatrixXs::Zero(realF_c.size(), 4);
        comparison.col(0) = realF_c;
        comparison.col(1) = analyticalF_c;
        comparison.col(2) = (realF_c - analyticalF_c);
        comparison.col(3)
            = realQ.completeOrthogonalDecomposition().solve(realB);
        std::cout << "Diff f_c range: " << (realF_c - analyticalF_c).minCoeff()
                  << " - " << (realF_c - analyticalF_c).maxCoeff() << std::endl;
        std::cout << "Real f_c ::: Analytical f_c ::: Diff f_c ::: Real Qinv*B "
                  << std::endl
                  << comparison << std::endl;

        Eigen::MatrixXs comparisonB = Eigen::MatrixXs::Zero(realB.size(), 4);
        comparisonB.col(0) = realB;
        comparisonB.col(1) = analyticalQ * analyticalF_c;
        comparisonB.col(2) = (realB - (analyticalQ * analyticalF_c));
        comparisonB.col(3)
            = realQ
              * analyticalQ.completeOrthogonalDecomposition().solve(realB);
        std::cout << "Diff Q*f_c range: "
                  << (realB - (analyticalQ * analyticalF_c)).minCoeff() << " - "
                  << (realB - (analyticalQ * analyticalF_c)).maxCoeff()
                  << std::endl;
        std::cout << "Real Q*f_c ::: Analytical Q*f_c ::: Diff Q*f_c ::: Real "
                     "Q*(analytical Qinv*B) "
                  << std::endl
                  << comparisonB << std::endl;
        // "Real Q" only matches our Q if there are no columns in upper bounds
        if (A_ub.cols() == 0)
        {
          if (realQ.rows() >= 6 || realQ.cols() >= 6)
          {
            std::cout << "Real Q (top left 6x6):" << std::endl
                      << realQ.block<6, 6>(0, 0) << std::endl;
            std::cout << "Real B (top 6):" << std::endl
                      << realB.segment<6>(0) << std::endl;
            std::cout << "Real Qinv*B (top 6):" << std::endl
                      << realQ.completeOrthogonalDecomposition()
                             .solve(realB)
                             .segment<6>(0)
                      << std::endl;
          }
          else
          {
            std::cout << "Real Q:" << std::endl << realQ << std::endl;
            std::cout << "Real B:" << std::endl << realB << std::endl;
            std::cout << "Real Qinv*B:" << std::endl
                      << realQ.completeOrthogonalDecomposition().solve(realB)
                      << std::endl;
          }
        }
        return false;
      }
    }
  }

  snapshot.restore();
  return true;
}

bool verifyF_c(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  // assert(classicPtr->areResultsStandardized());

  /*
  auto constraints = classicPtr->getDifferentiableConstraints();
  for (auto constraint : constraints)
  {
    auto contact = constraint->getContact();
    std::cout << "Contact type=" << contact.type
              << ", depth=" << contact.penetrationDepth
              << ", clip=" << world->getContactClippingDepth() << std::endl;
  }
  */

  Eigen::VectorXs original = world->getPositions();

  // const s_t EPS = 1e-6;

  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();

  world->setPositions(original);

  Eigen::VectorXs ones = Eigen::VectorXs::Ones(A_c.cols());
  Eigen::MatrixXs analyticalA_cJac
      = classicPtr->getJacobianOfClampingConstraints(world, ones);
  Eigen::MatrixXs bruteForceA_cJac
      = classicPtr->finiteDifferenceJacobianOfClampingConstraints(world, ones);

  assert(
      world->getPositions() == classicPtr->getPreStepPosition()
      && world->getVelocities() == classicPtr->getPreStepVelocity());

  if (!equals(analyticalA_cJac, bruteForceA_cJac, 1e-8))
  {
    if (analyticalA_cJac.cols() >= 6 && analyticalA_cJac.rows() >= 6)
    {
      std::cout << "Analytical A_c Jac (top-left 6x6): " << std::endl
                << analyticalA_cJac.block<6, 6>(0, 0) << std::endl;
      std::cout << "Brute force A_c Jac (top-left 6x6): " << std::endl
                << bruteForceA_cJac.block<6, 6>(0, 0) << std::endl;
      std::cout << "Diff (" << (analyticalA_cJac - bruteForceA_cJac).minCoeff()
                << " - " << (analyticalA_cJac - bruteForceA_cJac).maxCoeff()
                << ") (top-left 6x6): " << std::endl
                << (analyticalA_cJac - bruteForceA_cJac).block<6, 6>(0, 0)
                << std::endl;
    }
    else
    {
      std::cout << "Analytical A_c Jac: " << std::endl
                << analyticalA_cJac << std::endl;
      std::cout << "Brute force A_c Jac: " << std::endl
                << bruteForceA_cJac << std::endl;
      std::cout << "Diff (" << (analyticalA_cJac - bruteForceA_cJac).minCoeff()
                << " - " << (analyticalA_cJac - bruteForceA_cJac).maxCoeff()
                << "): " << std::endl
                << analyticalA_cJac - bruteForceA_cJac << std::endl;
    }
    return false;
  }

  ones = Eigen::VectorXs::Ones(A_c.rows());
  Eigen::MatrixXs analyticalA_cTJac
      = classicPtr->getJacobianOfClampingConstraintsTranspose(world, ones);
  Eigen::MatrixXs bruteForceA_cTJac
      = classicPtr->finiteDifferenceJacobianOfClampingConstraintsTranspose(
          world, ones);

  assert(
      world->getPositions() == classicPtr->getPreStepPosition()
      && world->getVelocities() == classicPtr->getPreStepVelocity());

  if (!equals(analyticalA_cTJac, bruteForceA_cTJac, 1e-9))
  {
    if (analyticalA_cTJac.cols() >= 6 && analyticalA_cTJac.rows() >= 6)
    {
      std::cout << "Analytical A_c^T Jac (top-left 6x6): " << std::endl
                << analyticalA_cTJac.block<6, 6>(0, 0) << std::endl;
      std::cout << "Brute force A_c^T Jac (top-left 6x6): " << std::endl
                << bruteForceA_cTJac.block<6, 6>(0, 0) << std::endl;
      std::cout << "Diff ("
                << (analyticalA_cTJac - bruteForceA_cTJac).minCoeff() << " - "
                << (analyticalA_cTJac - bruteForceA_cTJac).maxCoeff()
                << ") (top-left 6x6): " << std::endl
                << (analyticalA_cTJac - bruteForceA_cTJac).block<6, 6>(0, 0)
                << std::endl;
    }
    else
    {
      std::cout << "Analytical A_c^T Jac: " << std::endl
                << analyticalA_cTJac << std::endl;
      std::cout << "Brute force A_c^T Jac: " << std::endl
                << bruteForceA_cTJac << std::endl;
      std::cout << "Diff ("
                << (analyticalA_cTJac - bruteForceA_cTJac).minCoeff() << " - "
                << (analyticalA_cTJac - bruteForceA_cTJac).maxCoeff()
                << "): " << std::endl
                << analyticalA_cTJac - bruteForceA_cTJac << std::endl;
    }
    return false;
  }

  if (A_c.cols() > 0)
  {
    Eigen::MatrixXs realQ = classicPtr->getClampingAMatrix();
    Eigen::VectorXs realB = classicPtr->getClampingConstraintRelativeVels();

    Eigen::VectorXs x
        = A_c * realQ.completeOrthogonalDecomposition().solve(realB);
    Eigen::MatrixXs analyticalMinv_fJac
        = classicPtr->getJacobianOfMinv(world, x, WithRespectTo::POSITION);
    Eigen::MatrixXs bruteForceMinv_fJac
        = classicPtr->finiteDifferenceJacobianOfMinv(
            world, x, WithRespectTo::POSITION);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());
    if (!equals(analyticalMinv_fJac, bruteForceMinv_fJac, 1e-8))
    {
      std::cout << "Brute force Minv*x (x constant) Jacobian:" << std::endl
                << bruteForceMinv_fJac << std::endl;
      std::cout << "Analytical Minv*x (x constant) Jacobian:" << std::endl
                << analyticalMinv_fJac << std::endl;
      std::cout << "Diff Jac ("
                << (bruteForceMinv_fJac - analyticalMinv_fJac).minCoeff()
                << " - "
                << (bruteForceMinv_fJac - analyticalMinv_fJac).maxCoeff()
                << "):" << std::endl
                << (bruteForceMinv_fJac - analyticalMinv_fJac) << std::endl;

      // The first step of the computation for Minv involves finding the Jac for
      // M
      Eigen::VectorXs v = classicPtr->getInvMassMatrix(world) * x;
      Eigen::MatrixXs bruteForceMvJac = classicPtr->finiteDifferenceJacobianOfM(
          world, v, WithRespectTo::POSITION);
      Eigen::MatrixXs analyticalMvJac
          = classicPtr->getJacobianOfM(world, v, WithRespectTo::POSITION);
      if (!equals(analyticalMvJac, bruteForceMvJac, 1e-8))
      {
        std::cout << "Brute force M*v (v constant) Jacobian:" << std::endl
                  << bruteForceMvJac << std::endl;
        std::cout << "Analytical M*v (v constant) Jacobian:" << std::endl
                  << analyticalMvJac << std::endl;
        std::cout << "Diff Jac ("
                  << (bruteForceMvJac - analyticalMvJac).minCoeff() << " - "
                  << (bruteForceMvJac - analyticalMvJac).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceMvJac - analyticalMvJac) << std::endl;

        // Check accuracy on component pieces

        int cursor = 0;
        for (int i = 0; i < world->getNumSkeletons(); i++)
        {
          auto skel = world->getSkeleton(i);
          int dofs = skel->getNumDofs();
          for (int j = 0; j < skel->getNumBodyNodes(); j++)
          {
            skel->getBodyNode(j)->debugJacobianOfMForward(
                WithRespectTo::POSITION, v.segment(cursor, dofs));
          }
          for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
          {
            skel->getBodyNode(j)->debugJacobianOfMBackward(
                WithRespectTo::POSITION,
                v.segment(cursor, dofs),
                bruteForceMvJac);
          }
          cursor += dofs;
        }
      }
    }

    Eigen::MatrixXs Minv = world->getInvMassMatrix();
    Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;
    Eigen::MatrixXs bruteForceQbJac = classicPtr->finiteDifferenceJacobianOfQb(
        world, realB, WithRespectTo::POSITION);
    Eigen::MatrixXs analyticalQbJac = classicPtr->dQ_WithUB(
        world, Minv, A_c, E, A_c_ub_E, realB, WithRespectTo::POSITION);

    if (!equals(analyticalQbJac, bruteForceQbJac, 4e-8))
    {
      if (analyticalQbJac.rows() >= 6 && bruteForceQbJac.cols() >= 6)
      {
        std::cout << "Brute force Q*b (b constant) Jacobian (top-left 6x6):"
                  << std::endl
                  << bruteForceQbJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical Q*b (b constant) Jacobian (top-left 6x6):"
                  << std::endl
                  << analyticalQbJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac ("
                  << (bruteForceQbJac - analyticalQbJac).minCoeff() << " - "
                  << (bruteForceQbJac - analyticalQbJac).maxCoeff()
                  << ") (top-left 6x6):" << std::endl
                  << (bruteForceQbJac - analyticalQbJac).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force Q*b (b constant) Jacobian:" << std::endl
                  << bruteForceQbJac << std::endl;
        std::cout << "Analytical Q*b (b constant) Jacobian:" << std::endl
                  << analyticalQbJac << std::endl;
        std::cout << "Diff Jac ("
                  << (bruteForceQbJac - analyticalQbJac).minCoeff() << " - "
                  << (bruteForceQbJac - analyticalQbJac).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceQbJac - analyticalQbJac) << std::endl;
      }

      Eigen::MatrixXs diff = bruteForceQbJac - analyticalQbJac;
      for (int i = 0; i < world->getNumDofs(); i++)
      {
        s_t maxError = std::max(
            abs(diff.col(i).minCoeff()), abs(diff.col(i).maxCoeff()));
        dynamics::DegreeOfFreedom* dof = world->getDofs()[i];

        std::cout << "col[" << i << "] joint \"" << dof->getJoint()->getName()
                  << "::" << dof->getJoint()->getType() << "\"["
                  << dof->getIndexInJoint() << "] max error: " << maxError
                  << std::endl;
      }

      return false;
    }

    Eigen::MatrixXs bruteForceQinvBJac
        = classicPtr
              ->finiteDifferenceJacobianOfLCPConstraintMatrixClampingSubset(
                  world, realB, WithRespectTo::POSITION);
    Eigen::MatrixXs analyticalQinvBJac
        = classicPtr->getJacobianOfLCPConstraintMatrixClampingSubset(
            world, realB, WithRespectTo::POSITION);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalQinvBJac, bruteForceQinvBJac, 1e-8))
    {
      if (analyticalQinvBJac.rows() >= 6 && analyticalQinvBJac.cols() >= 6)
      {
        std::cout << "Brute force Qinv*b (b constant) Jacobian (top-left 6x6):"
                  << std::endl
                  << bruteForceQinvBJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical Qinv*b (b constant) Jacobian (top-left 6x6):"
                  << std::endl
                  << analyticalQinvBJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac ("
                  << (bruteForceQinvBJac - analyticalQinvBJac).minCoeff()
                  << " - "
                  << (bruteForceQinvBJac - analyticalQinvBJac).maxCoeff()
                  << ") (top-left 6x6):" << std::endl
                  << (bruteForceQinvBJac - analyticalQinvBJac).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force Qinv*b (b constant) Jacobian:" << std::endl
                  << bruteForceQinvBJac << std::endl;
        std::cout << "Analytical Qinv*b (b constant) Jacobian:" << std::endl
                  << analyticalQinvBJac << std::endl;
        std::cout << "Diff Jac ("
                  << (bruteForceQinvBJac - analyticalQinvBJac).minCoeff()
                  << " - "
                  << (bruteForceQinvBJac - analyticalQinvBJac).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceQinvBJac - analyticalQinvBJac) << std::endl;
      }

      Eigen::MatrixXs diff = bruteForceQinvBJac - analyticalQinvBJac;
      for (int i = 0; i < world->getNumDofs(); i++)
      {
        s_t maxError = std::max(
            abs(diff.col(i).minCoeff()), abs(diff.col(i).maxCoeff()));
        dynamics::DegreeOfFreedom* dof = world->getDofs()[i];

        std::cout << "col[" << i << "] joint \"" << dof->getJoint()->getName()
                  << "::" << dof->getJoint()->getType() << "\"["
                  << dof->getIndexInJoint() << "] max error: " << maxError
                  << std::endl;
      }

      return false;
    }

    /*
    // Suspicious inputs to check if dB is broken:

    Eigen::MatrixXs dMinv_f = getJacobianOfMinv(world, f, wrt);
    Eigen::MatrixXs dA_c_f
        = getJacobianOfClampingConstraintsTranspose(world, v_f);
    */

    Eigen::MatrixXs bruteForceJacB
        = classicPtr->finiteDifferenceJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::POSITION);
    Eigen::MatrixXs analyticalJacB
        = classicPtr->getJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::POSITION);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJacB, bruteForceJacB, 1e-8))
    {
      std::cout << "Failed dB Jacobian for POSITION!" << std::endl;
      if (bruteForceJacB.rows() >= 6 && bruteForceJacB.cols() >= 6)
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac (top-left 6x6) ("
                  << (bruteForceJacB - analyticalJacB).minCoeff() << " - "
                  << (bruteForceJacB - analyticalJacB).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceJacB - analyticalJacB).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols() << "):" << std::endl
                  << bruteForceJacB << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols() << "):" << std::endl
                  << analyticalJacB << std::endl;
        std::cout << "Diff Jac:" << std::endl
                  << (bruteForceJacB - analyticalJacB) << std::endl;
        std::cout << "Real Qinv:" << std::endl
                  << realQ.completeOrthogonalDecomposition().pseudoInverse()
                  << std::endl;
      }
      return false;
    }

    Eigen::MatrixXs bruteForceEstimatedJac
        = classicPtr->finiteDifferenceJacobianOfEstimatedConstraintForce(
            world, WithRespectTo::POSITION);
    Eigen::MatrixXs bruteForceJac
        = classicPtr->finiteDifferenceJacobianOfConstraintForce(
            world, WithRespectTo::POSITION);
    Eigen::MatrixXs analyticalJac = classicPtr->getJacobianOfConstraintForce(
        world, WithRespectTo::POSITION);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJac, bruteForceJac, 2e-8))
    {
      std::cout << "Failed f_c Jacobian for POSITION!" << std::endl;
      if (bruteForceJac.rows() >= 6 && bruteForceJac.cols() >= 6)
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff between brute force and analytical Jac ("
                  << (bruteForceJac - analyticalJac).minCoeff() << " - "
                  << (bruteForceJac - analyticalJac).maxCoeff()
                  << ") (top-left 6x6):" << std::endl
                  << (bruteForceJac - analyticalJac).block<6, 6>(0, 0)
                  << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceEstimatedJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "dB (top-left 6x6):" << std::endl
                  << analyticalJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff between brute force estimated and analytical Jac ("
                  << (bruteForceEstimatedJac - analyticalJac).minCoeff()
                  << " - "
                  << (bruteForceEstimatedJac - analyticalJac).maxCoeff()
                  << ") (top-left 6x6):" << std::endl
                  << (bruteForceEstimatedJac - analyticalJac).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols() << "):" << std::endl
                  << bruteForceJac << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols() << "):" << std::endl
                  << analyticalJac << std::endl;
        std::cout << "Diff Jac (" << (bruteForceJac - analyticalJac).minCoeff()
                  << " - " << (bruteForceJac - analyticalJac).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceJac - analyticalJac) << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols() << "):" << std::endl
                  << bruteForceEstimatedJac << std::endl;
        std::cout << "Diff Brute force Jac ("
                  << (bruteForceEstimatedJac - analyticalJac).minCoeff()
                  << " - "
                  << (bruteForceEstimatedJac - analyticalJac).maxCoeff()
                  << "):" << std::endl
                  << (bruteForceEstimatedJac - analyticalJac) << std::endl;
      }
      return false;
    }

    bruteForceJacB
        = classicPtr->finiteDifferenceJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::VELOCITY);
    analyticalJacB = classicPtr->getJacobianOfLCPOffsetClampingSubset(
        world, WithRespectTo::VELOCITY);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJacB, bruteForceJacB, 1e-8))
    {
      std::cout << "Failed dB Jacobian for VELOCITY!" << std::endl;
      if (analyticalJacB.cols() >= 6 && analyticalJacB.rows() >= 6)
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac (top-left 6x6):" << std::endl
                  << (bruteForceJacB - analyticalJacB).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols() << "):" << std::endl
                  << bruteForceJacB << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols() << "):" << std::endl
                  << analyticalJacB << std::endl;
        std::cout << "Diff Jac:" << std::endl
                  << (bruteForceJacB - analyticalJacB) << std::endl;
      }
      std::cout << "Real Qinv:" << std::endl
                << realQ.completeOrthogonalDecomposition().pseudoInverse()
                << std::endl;
      return false;
    }

    bruteForceEstimatedJac
        = classicPtr->finiteDifferenceJacobianOfEstimatedConstraintForce(
            world, WithRespectTo::VELOCITY);
    bruteForceJac = classicPtr->finiteDifferenceJacobianOfConstraintForce(
        world, WithRespectTo::VELOCITY);
    analyticalJac = classicPtr->getJacobianOfConstraintForce(
        world, WithRespectTo::VELOCITY);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Failed f_c Jacobian for VELOCITY!" << std::endl;
      if (bruteForceJac.rows() >= 6 && bruteForceJac.cols() >= 6)
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac (top-left 6x6):" << std::endl
                  << (bruteForceJac - analyticalJac).block<6, 6>(0, 0)
                  << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceEstimatedJac.block<6, 6>(0, 0) << std::endl;
      }
      else
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols() << "):" << std::endl
                  << bruteForceJac << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols() << "):" << std::endl
                  << analyticalJac << std::endl;
        std::cout << "Diff Jac:" << std::endl
                  << (bruteForceJac - analyticalJac) << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols() << "):" << std::endl
                  << bruteForceEstimatedJac << std::endl;
      }
      return false;
    }

    bruteForceJacB
        = classicPtr->finiteDifferenceJacobianOfLCPOffsetClampingSubset(
            world, WithRespectTo::FORCE);
    analyticalJacB = classicPtr->getJacobianOfLCPOffsetClampingSubset(
        world, WithRespectTo::FORCE);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJacB, bruteForceJacB, 1e-8))
    {
      std::cout << "Failed dB Jacobian for FORCE!" << std::endl;
      if (analyticalJacB.cols() >= 6 && analyticalJacB.rows() >= 6)
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJacB.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac (top-left 6x6):" << std::endl
                  << (bruteForceJacB - analyticalJacB).block<6, 6>(0, 0)
                  << std::endl;
      }
      else
      {
        std::cout << "Brute force dB Jacobian (size " << bruteForceJacB.rows()
                  << "x" << bruteForceJacB.cols() << "):" << std::endl
                  << bruteForceJacB << std::endl;
        std::cout << "Analytical dB Jacobian (size " << analyticalJacB.rows()
                  << "x" << analyticalJacB.cols() << "):" << std::endl
                  << analyticalJacB << std::endl;
        std::cout << "Diff Jac:" << std::endl
                  << (bruteForceJacB - analyticalJacB) << std::endl;
      }
      std::cout << "Real Qinv:" << std::endl
                << realQ.completeOrthogonalDecomposition().pseudoInverse()
                << std::endl;
      return false;
    }

    bruteForceEstimatedJac
        = classicPtr->finiteDifferenceJacobianOfEstimatedConstraintForce(
            world, WithRespectTo::FORCE);
    bruteForceJac = classicPtr->finiteDifferenceJacobianOfConstraintForce(
        world, WithRespectTo::FORCE);
    analyticalJac
        = classicPtr->getJacobianOfConstraintForce(world, WithRespectTo::FORCE);

    assert(
        world->getPositions() == classicPtr->getPreStepPosition()
        && world->getVelocities() == classicPtr->getPreStepVelocity());

    if (!equals(analyticalJac, bruteForceJac, 1e-8))
    {
      std::cout << "Failed f_c Jacobian for FORCE!" << std::endl;
      if (bruteForceJac.rows() >= 6 && bruteForceJac.cols() >= 6)
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << analyticalJac.block<6, 6>(0, 0) << std::endl;
        std::cout << "Diff Jac (top-left 6x6):" << std::endl
                  << (bruteForceJac - analyticalJac).block<6, 6>(0, 0)
                  << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols()
                  << ") (top-left 6x6):" << std::endl
                  << bruteForceEstimatedJac.block<6, 6>(0, 0) << std::endl;
      }
      else
      {
        std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
                  << "x" << bruteForceJac.cols() << "):" << std::endl
                  << bruteForceJac << std::endl;
        std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows()
                  << "x" << analyticalJac.cols() << "):" << std::endl
                  << analyticalJac << std::endl;
        std::cout << "Diff Jac:" << std::endl
                  << (bruteForceJac - analyticalJac) << std::endl;
        std::cout << "Brute force f_c estimated Jacobian (size "
                  << bruteForceEstimatedJac.rows() << "x"
                  << bruteForceEstimatedJac.cols() << "):" << std::endl
                  << bruteForceEstimatedJac << std::endl;
      }
      return false;
    }
  }

  snapshot.restore();

  return true;
}

bool verifyConstraintForceJac(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();

  /*
  Eigen::MatrixXs bruteForceJac
      = classicPtr->finiteDifferenceJacobianOfConstraintForce(
          world, WithRespectTo::POSITION);
  */
  Eigen::MatrixXs bruteForceJac
      = classicPtr->finiteDifferenceJacobianOfEstimatedConstraintForce(
          world, WithRespectTo::POSITION);
  Eigen::MatrixXs analyticalJac = classicPtr->getJacobianOfConstraintForce(
      world, WithRespectTo::POSITION);
  if (!equals(analyticalJac, bruteForceJac, 1e-8))
  {
    std::cout << "Failed f_c Jacobian!" << std::endl;
    std::cout << "Brute force f_c Jacobian (size " << bruteForceJac.rows()
              << "x" << bruteForceJac.cols() << ") (top-left 6x6):" << std::endl
              << bruteForceJac.block<6, 6>(0, 0) << std::endl;
    std::cout << "Analytical f_c Jacobian (size " << analyticalJac.rows() << "x"
              << analyticalJac.cols() << ") (top-left 6x6):" << std::endl
              << analyticalJac.block<6, 6>(0, 0) << std::endl;
    std::cout << "Diff Jac (top-left 6x6):" << std::endl
              << (bruteForceJac - analyticalJac).block<6, 6>(0, 0) << std::endl;
    return false;
  }

  snapshot.restore();

  return true;
}

struct VelocityTest
{
  bool standardized;
  Eigen::VectorXs realNextVel;
  Eigen::VectorXs realNextVelPreSolve;
  Eigen::VectorXs realNextVelDeltaVFromF;
  Eigen::VectorXs predictedNextVel;
  Eigen::VectorXs predictedNextVelPreSolve;
  Eigen::VectorXs predictedNextVelDeltaVFromF;
  Eigen::VectorXs preStepVelocity;
  Eigen::VectorXs realF_c;
  Eigen::VectorXs predictedF_c;
  Eigen::MatrixXs realQ;
  Eigen::MatrixXs predictedQ;
  Eigen::VectorXs realB;
  Eigen::VectorXs predictedB;
  Eigen::VectorXs realX;
  Eigen::VectorXs predictedX;
  Eigen::MatrixXs Minv;
};

VelocityTest runVelocityTest(WorldPtr world)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  RestorableSnapshot snapshot(world);
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    skel->computeForwardDynamics();
    skel->integrateVelocities(world->getTimeStep());
  }
  Eigen::VectorXs realNextVelPreSolve = world->getVelocities();
  snapshot.restore();

  Eigen::VectorXs preStepVelocity = world->getVelocities();

  Eigen::VectorXs realNextVel = classicPtr->getPostStepVelocity();
  Eigen::VectorXs realNextVelDeltaVFromF = realNextVel - realNextVelPreSolve;

  Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;
  Eigen::VectorXs tau = world->getControlForces();
  int nDofs = world->getNumDofs();
  Eigen::VectorXs damping = Eigen::VectorXs::Zero(nDofs);
  std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
  for (int i=0;i<nDofs;i++)
  {
    damping(i) = dofs[i]->getDampingCoefficient();
  }
  s_t dt = world->getTimeStep();

  Eigen::MatrixXs Minv = world->getInvMassMatrix();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXs f_c
      = classicPtr->estimateClampingConstraintImpulses(world, A_c, A_ub, E);

  /*
  Eigen::VectorXs b = Eigen::VectorXs(A_c.cols());
  Eigen::MatrixXs Q = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
  classicPtr->computeLCPOffsetClampingSubset(world, b, A_c);
  classicPtr->computeLCPConstraintMatrixClampingSubset(world, Q, A_c);
  std::cout << "Real B: " << std::endl
            << classicPtr->getClampingConstraintRelativeVels() << std::endl;
  std::cout << "Analytical B: " << std::endl << b << std::endl;
  std::cout << "Real A: " << std::endl
            << classicPtr->mGradientMatrices[0]->mA << std::endl;
  std::cout << "Analytical A: " << std::endl << Q << std::endl;
  */

  Eigen::VectorXs allRealImpulses = classicPtr->getContactConstraintImpulses();
  /*
  Eigen::VectorXs velChange = Eigen::VectorXs::Zero(world->getNumDofs());
  for (int i = 0; i < allRealImpulses.size(); i++)
  {
    velChange += allRealImpulses(i)
                 * classicPtr->mGradientMatrices[0]->mMassedImpulseTests[i];
  }
  */
  Eigen::VectorXs velDueToIllegal
      = classicPtr->getVelocityDueToIllegalImpulses();
  velDueToIllegal.setZero();

  Eigen::VectorXs realImpulses = classicPtr->getClampingConstraintImpulses();

  Eigen::VectorXs preSolveV = preStepVelocity + dt * Minv * (tau - C - damping.asDiagonal()*preStepVelocity);

  Eigen::VectorXs f_cDeltaV;
  if (A_c.cols() == 0)
  {
    f_cDeltaV = Eigen::VectorXs::Zero(preSolveV.size());
  }
  else
  {
    f_cDeltaV = Minv * A_c_ub_E * f_c + velDueToIllegal;
  }
  Eigen::VectorXs realF_cDeltaV = Minv * A_c_ub_E * realImpulses;
  Eigen::VectorXs postSolveV = preSolveV + f_cDeltaV;

  /*
  std::cout << "Real f_c delta V:" << std::endl << realF_cDeltaV << std::endl;
  std::cout << "Analytical f_c delta V:" << std::endl << f_cDeltaV << std::endl;
  std::cout << "Diff:" << std::endl << f_cDeltaV - realF_cDeltaV << std::endl;
  */

  VelocityTest test;
  test.standardized = classicPtr->areResultsStandardized();
  test.predictedNextVel = postSolveV;
  test.predictedNextVelDeltaVFromF = f_cDeltaV;
  test.predictedNextVelPreSolve = preSolveV;
  test.realNextVel = realNextVel;
  test.realNextVelDeltaVFromF = realNextVelDeltaVFromF;
  test.realNextVelPreSolve = realNextVelPreSolve;
  test.preStepVelocity = preStepVelocity;
  test.realF_c = realImpulses;
  test.predictedF_c = f_c;
  test.realQ = classicPtr->getClampingAMatrix();
  test.predictedQ = A_c.transpose() * Minv * A_c_ub_E;
  test.predictedQ.diagonal() += classicPtr->getConstraintForceMixingDiagonal();
  test.realB = classicPtr->getClampingConstraintRelativeVels();
  test.predictedB = -A_c.transpose() * preSolveV;
  test.Minv = Minv;

  snapshot.restore();

  return test;
}

bool verifyNextV(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  // VelocityTest originalTest = runVelocityTest(world);

  Eigen::VectorXs forces = world->getControlForces();

  const s_t EPSILON = 1e-6;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    Eigen::VectorXs tweakedForce = Eigen::VectorXs(forces);
    tweakedForce(i) -= EPSILON;

    // snapshot.restore();
    world->setControlForces(tweakedForce);

    VelocityTest perturbedTest = runVelocityTest(world);

    if (!equals(
            perturbedTest.predictedNextVel,
            perturbedTest.realNextVel,
            classicPtr->hasBounces()
                ? 1e-4 // things get sloppy when bouncing, increase tol
                : 5e-9))
    {
      std::cout << "Real standardized: " << perturbedTest.standardized
                << std::endl;
      std::cout << "Real v_t+1:" << std::endl
                << perturbedTest.realNextVel << std::endl;
      std::cout << "Analytical v_t+1:" << std::endl
                << perturbedTest.predictedNextVel << std::endl;
      std::cout << "Analytical pre-solve v_t+1:" << std::endl
                << perturbedTest.predictedNextVelPreSolve << std::endl;
      std::cout << "Real pre-solve v_t+1:" << std::endl
                << perturbedTest.realNextVelPreSolve << std::endl;
      std::cout << "Pre-solve diff:" << std::endl
                << (perturbedTest.predictedNextVelPreSolve
                    - perturbedTest.realNextVelPreSolve)
                << std::endl;
      std::cout << "Analytical delta V from f_c v_t+1:" << std::endl
                << perturbedTest.predictedNextVelDeltaVFromF << std::endl;
      std::cout << "Real delta V from f_c v_t+1:" << std::endl
                << perturbedTest.realNextVelDeltaVFromF << std::endl;
      std::cout << "Diff:" << std::endl
                << (perturbedTest.realNextVelDeltaVFromF
                    - perturbedTest.predictedNextVelDeltaVFromF)
                << std::endl;
      std::cout << "Analytical f_c:" << std::endl
                << perturbedTest.predictedF_c << std::endl;
      std::cout << "Real f_c:" << std::endl
                << perturbedTest.realF_c << std::endl;
      std::cout << "Diff:" << std::endl
                << (perturbedTest.predictedF_c - perturbedTest.realF_c)
                << std::endl;
      std::cout << "Analytical Q:" << std::endl
                << perturbedTest.predictedQ << std::endl;
      std::cout << "Real Q:" << std::endl << perturbedTest.realQ << std::endl;
      std::cout << "Diff:" << std::endl
                << (perturbedTest.predictedQ - perturbedTest.realQ)
                << std::endl;
      std::cout << "Analytical b:" << std::endl
                << perturbedTest.predictedB << std::endl;
      std::cout << "Real b:" << std::endl << perturbedTest.realB << std::endl;
      std::cout << "Diff:" << std::endl
                << (perturbedTest.predictedB - perturbedTest.realB)
                << std::endl;
      return false;
    }
  }

  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);

  snapshot.restore();
  return true;
}

bool verifyScratch(WorldPtr world, WithRespectTo* wrt)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXs analytical = classicPtr->getScratchAnalytical(world, wrt);
  MatrixXs bruteForce = classicPtr->getScratchFiniteDifference(world, wrt);

  /*
  MatrixXs posVelAnalytical = classicPtr->getPosVelJacobian(world);
  MatrixXs posVelFd = classicPtr->finiteDifferencePosVelJacobian(world);
  */
  if (!equals(world->getPositions(), classicPtr->getPreStepPosition()))
  {
    std::cout << "Position not preserved!" << std::endl;
  }
  if (!equals(world->getVelocities(), classicPtr->getPreStepVelocity()))
  {
    std::cout << "Velocity not preserved!" << std::endl;
  }
  if (!equals(world->getControlForces(), classicPtr->getPreStepTorques()))
  {
    std::cout << "Force not preserved!" << std::endl;
  }

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force Scratch Jacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical Scratch Jacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    std::cout << "Diff (" << (analytical - bruteForce).minCoeff() << " - "
              << (analytical - bruteForce).maxCoeff() << "):" << std::endl;
    /*
    std::cout << "Pos-Vel Analytical:" << std::endl
              << posVelAnalytical << std::endl;
    std::cout << "Pos-Vel FD:" << std::endl << posVelFd << std::endl;
    */
    return false;
  }

  return true;
}

Eigen::Vector6s getLinearScratchScrew()
{
  Eigen::Vector6s linearScratchScrew;
  linearScratchScrew << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  return linearScratchScrew;
}

Eigen::Vector6s getLinearScratchForce()
{
  Eigen::Vector6s linearScratchForce;
  linearScratchForce << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  return linearScratchForce;
}

s_t linearScratch(s_t input)
{
  Eigen::Isometry3s transform = math::expMap(getLinearScratchScrew() * input);
  Eigen::Vector6s forceInFrame
      = math::dAdInvT(transform, getLinearScratchForce());
  std::cout << "Force in frame: " << std::endl << forceInFrame << std::endl;
  std::cout << "screw: " << std::endl << getLinearScratchScrew() << std::endl;
  std::cout << "dotted with screw: " << std::endl
            << forceInFrame.dot(getLinearScratchScrew()) << std::endl;
  return forceInFrame.dot(getLinearScratchScrew());
}

s_t bruteForceLinearScratch(s_t startingPoint)
{
  const s_t EPS = 1e-6;
  return (linearScratch(startingPoint + EPS) - linearScratch(startingPoint))
         / EPS;
}

s_t analyticalLinearScratch(s_t /* point */)
{
  return 1.0;
}

bool verifyLinearScratch()
{
  s_t point = 0.76;
  s_t bruteForce = bruteForceLinearScratch(point);
  s_t analytical = analyticalLinearScratch(point);
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
    WorldPtr world, VectorXs proposedVelocities, WithRespectTo* wrt)
{
  world->setVelocities(proposedVelocities);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXs analytical = classicPtr->getJacobianOfProjectionIntoClampsMatrix(
      world, proposedVelocities * 10, wrt);
  MatrixXs bruteForce
      = classicPtr->finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
          world, proposedVelocities * 10, wrt);

  // These individual values can be quite large, on the order of 1e+4, so we
  // normalize by size before checking for error, because 1e-8 on a 1e+4 value
  // (12 digits of precision) may be unattainable
  MatrixXs zero = MatrixXs::Zero(analytical.rows(), analytical.cols());
  MatrixXs normalizedDiff
      = (analytical - bruteForce) / (0.001 + analytical.norm());

  if (!equals(normalizedDiff, zero, 1e-8))
  {
    std::cout << "Brute force P_c Jacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical P_c Jacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << bruteForce - analytical << std::endl;
    std::cout << "Normalized Diff:" << std::endl << normalizedDiff << std::endl;
    return false;
  }

  return true;
}

bool verifyJointVelocityJacobians(WorldPtr world)
{
  s_t threshold = 1e-9;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumJoints(); j++)
    {
      Eigen::MatrixXs fd
          = skel->getJoint(i)->finiteDifferenceRelativeJacobian();
      Eigen::MatrixXs analytical = skel->getJoint(i)->getRelativeJacobian();
      if (!equals(fd, analytical, threshold))
      {
        std::cout << "Velocity jacabians disagree on skeleton \""
                  << skel->getName() << "\", joint [" << j << "]:" << std::endl;
        std::cout << "Brute force: " << std::endl << fd << std::endl;
        std::cout << "Analytical: " << std::endl << analytical << std::endl;
        std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verifyJointPositionJacobians(WorldPtr world)
{
  s_t threshold = 1e-9;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumJoints(); j++)
    {
      Eigen::MatrixXs fd
          = skel->getJoint(i)
                ->finiteDifferenceRelativeJacobianInPositionSpace();
      Eigen::MatrixXs analytical
          = skel->getJoint(i)->getRelativeJacobianInPositionSpace();
      if (!equals(fd, analytical, threshold))
      {
        std::cout << "Position jacabians disagree on skeleton \""
                  << skel->getName() << "\", joint [" << j << "]:" << std::endl;
        std::cout << "Brute force: " << std::endl << fd << std::endl;
        std::cout << "Analytical: " << std::endl << analytical << std::endl;
        std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verifyFeatherstoneJacobians(WorldPtr world)
{
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumJoints(); j++)
    {
      skel->getJoint(j)->debugRelativeJacobianInPositionSpace();
    }
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->debugJacobianOfCForward(WithRespectTo::POSITION);
    }
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->debugJacobianOfCBackward(WithRespectTo::POSITION);
    }
    Eigen::VectorXs x = Eigen::VectorXs::Random(skel->getNumDofs());
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      skel->getBodyNode(j)->debugJacobianOfMForward(WithRespectTo::POSITION, x);
    }
    Eigen::MatrixXs MinvX
        = skel->finiteDifferenceJacobianOfM(x, WithRespectTo::POSITION);
    for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
    {
      skel->getBodyNode(j)->debugJacobianOfMBackward(
          WithRespectTo::POSITION, x, MinvX);
    }
  }
  return true;
}

bool verifyPosVelJacobian(WorldPtr world, VectorXs proposedVelocities)
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

  MatrixXs analytical = classicPtr->getPosVelJacobian(world);
  MatrixXs bruteForce = classicPtr->finiteDifferencePosVelJacobian(world);

  // Everything except Atlas passes this in 1e-8, but atlas runs closer to
  // 6e-8 max error. I'm going to cheerfully assume that this is just due to
  // the size and complexity of Atlas producing finite differencing errors in
  // dC and dMinv, which end up propagating. For now, this is fine.
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force posVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical posVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff (" << (analytical - bruteForce).minCoeff() << " - "
              << (analytical - bruteForce).maxCoeff() << "):" << std::endl
              << analytical - bruteForce << std::endl;
    return false;
  }

  return true;
}

bool verifyForceVelJacobian(WorldPtr world, VectorXs proposedVelocities)
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
  if (!classicPtr->areResultsStandardized())
  {
    std::cout << "verifyForceVelJacobian() got a non-standardized result"
              << std::endl;
  }

  MatrixXs analytical = classicPtr->getControlForceVelJacobian(world);
  MatrixXs bruteForce = classicPtr->finiteDifferenceForceVelJacobian(world);

  // Atlas runs at 1.5e-8 error
  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force forceVelJacobian:" << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical forceVelJacobian (should be the same as above):"
              << std::endl
              << analytical << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    std::cout << "Diff range:" << std::endl
              << (analytical - bruteForce).minCoeff() << " to "
              << (analytical - bruteForce).maxCoeff() << std::endl;
    return false;
  }

  return true;
}

bool verifyRecoveredLCPConstraints(WorldPtr world, VectorXs proposedVelocities)
{
  world->setVelocities(proposedVelocities);
  bool oldPenetrationCorrection = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  world->setPenetrationCorrectionEnabled(oldPenetrationCorrection);

  if (classicPtr->mGradientMatrices.size() > 1)
    return true;

  MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
  MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
  MatrixXs E = classicPtr->getUpperBoundMappingMatrix();

  if (A_c.cols() == 0)
    return true;

  MatrixXs Q = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
  classicPtr->computeLCPConstraintMatrixClampingSubset(world, Q, A_c, A_ub, E);
  Eigen::MatrixXs realQ = classicPtr->getClampingAMatrix();

  Eigen::VectorXs b = Eigen::VectorXs::Zero(A_c.cols());
  classicPtr->computeLCPOffsetClampingSubset(world, b, A_c);
  Eigen::VectorXs realB = classicPtr->getClampingConstraintRelativeVels();
  /* + (A_c.completeOrthogonalDecomposition().solve(
      classicPtr->getVelocityDueToIllegalImpulses())); */

  if (!equals(b, realB, 1e-12))
  {
    std::cout << "Error in verifyRecoveredLCPConstraints():" << std::endl;
    std::cout << "analytical B:" << std::endl << b << std::endl;
    std::cout << "real B:" << std::endl << realB << std::endl;
    /*
    std::cout << "vel due to illegal impulses:" << std::endl
              << classicPtr->getVelocityDueToIllegalImpulses() << std::endl;
    */
    return false;
  }

  // The next test doesn't make sense if we have any upper bounded, because
  // the matrices become more complex
  if (classicPtr->getNumUpperBound() == 0)
  {
    if (!equals(Q, realQ, 1e-8))
    {
      std::cout << "Error in verifyRecoveredLCPConstraints Q():" << std::endl;
      std::cout << "analytical Q:" << std::endl << Q << std::endl;
      std::cout << "real Q:" << std::endl << realQ << std::endl;
      std::cout << "diff:" << std::endl << Q - realQ << std::endl;
      return false;
    }
  }

  Eigen::VectorXs realX = classicPtr->getClampingConstraintImpulses();
  // Q can be low rank, so we center the solution on "realX"
  Eigen::VectorXs X
      = Q.completeOrthogonalDecomposition().solve(b - Q * realX) + realX;

  // If we make it this far, it's possible for our computed X to have errors
  // in it due to inverting a Q matrix that isn't precisely right. But
  // partRealX is computed using the "real" Q matrix.
  if (!equals(X, realX, 1e-8))
  {
    std::cout << "Error in verifyRecoveredLCPConstraints():" << std::endl;
    std::cout << "analytical X:" << std::endl << X << std::endl;
    std::cout << "real X:" << std::endl << realX << std::endl;
    std::cout << "diff:" << std::endl << X - realX << std::endl;
    return false;
  }

  return true;
}

bool verifyVelGradients(WorldPtr world, VectorXs worldVel)
{
  // return verifyScratch(world, WithRespectTo::VELOCITY);
  // return verifyConstraintForceJac(world);
  // return verifyF_c(world);
  // return verifyScratch(world, WithRespectTo::POSITION);
  // return verifyJacobianOfProjectionIntoClampsMatrix(world, worldVel,
  // POSITION); return verifyScratch(world); return verifyF_c(world); return
  // verifyLinearScratch(); return verifyNextV(world);
  return (
      // This test assumes no upper bounded contact
      // verifyClassicClampingConstraintMatrix(world, worldVel)
      verifyMassedClampingConstraintMatrix(world, worldVel)
      && verifyMassedUpperBoundConstraintMatrix(world, worldVel)
      // We no longer use P_c or its Jacobian anywhere
      // && verifyClassicProjectionIntoClampsMatrix(world, worldVel)
      // && verifyMassedProjectionIntoClampsMatrix(world, worldVel)
      // && verifyJacobianOfProjectionIntoClampsMatrix(world, worldVel,
      // POSITION)
      && verifyRecoveredLCPConstraints(world, worldVel)
      //&& verifyPerturbedF_c(world) && verifyF_c(world)
      && verifyForceVelJacobian(world, worldVel)
      && verifyVelVelJacobian(world, worldVel)
      && verifyFeatherstoneJacobians(world)
      && verifyPosVelJacobian(world, worldVel) 
      && verifyNextV(world));
}

bool verifyPosPosJacobianApproximation(
    WorldPtr world, std::size_t subdivisions, s_t tolerance)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyPosPosJacobianApproximation forwardPass returned a "
                 "null BackpropSnapshotPtr!"
              << std::endl;
    return false;
  }

  MatrixXs analytical = classicPtr->getPosPosJacobian(world);
  MatrixXs bruteForce
      = classicPtr->finiteDifferencePosPosJacobian(world, subdivisions);

  if (!equals(analytical, bruteForce, tolerance))
  {
    std::cout << "Brute force pos-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical pos-pos Jacobian: " << std::endl
              << analytical << std::endl;
    return false;
  }
  return true;
}

bool verifyVelPosJacobianApproximation(
    WorldPtr world, std::size_t subdivisions, s_t tolerance)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  if (!classicPtr)
  {
    std::cout << "verifyVelPosJacobianApproximation forwardPass returned a "
                 "null BackpropSnapshotPtr!"
              << std::endl;
    return false;
  }

  MatrixXs analytical = classicPtr->getVelPosJacobian(world);
  MatrixXs bruteForce
      = classicPtr->finiteDifferenceVelPosJacobian(world, subdivisions);

  if (!equals(analytical, bruteForce, tolerance))
  {
    std::cout << "Brute force vel-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical vel-pos Jacobian: " << std::endl
              << analytical << std::endl;
    return false;
  }
  return true;
}

bool verifyPosGradients(WorldPtr world, std::size_t subdivisions, s_t tolerance)
{
  return (
      verifyPosPosJacobianApproximation(world, subdivisions, tolerance)
      && verifyVelPosJacobianApproximation(world, subdivisions, tolerance));
}

bool verifyConstraintGroupSubJacobians(
    WorldPtr world, const neural::BackpropSnapshotPtr& classicPtr)
{

  RestorableSnapshot snapshot(world);

  world->setPositions(classicPtr->getPreStepPosition());
  world->setVelocities(classicPtr->getPreStepVelocity());
  world->setControlForces(classicPtr->getPreStepTorques());

  // Special case, there's only one constraint group
  if (classicPtr->mGradientMatrices.size() == 1)
  {
    std::shared_ptr<ConstrainedGroupGradientMatrices> group
        = classicPtr->mGradientMatrices[0];

    Eigen::MatrixXs groupPosPos = group->getPosPosJacobian(world);
    Eigen::MatrixXs worldPosPos = classicPtr->getPosPosJacobian(world);
    if (!equals(groupPosPos, worldPosPos, 0))
    {
      std::cout << "ConstrainedGroupGradientMatrices and BackpropSnapshotPtr "
                   "don't match!"
                << std::endl;
      std::cout << "World pos-pos Jacobian: " << std::endl
                << worldPosPos << std::endl;
      std::cout << "Group pos-pos Jacobian: " << std::endl
                << groupPosPos << std::endl;
      return false;
    }

    Eigen::MatrixXs groupVelPos = group->getVelPosJacobian(world);
    Eigen::MatrixXs worldVelPos = classicPtr->getVelPosJacobian(world);
    if (!equals(groupVelPos, worldVelPos, 1e-10))
    {
      std::cout << "ConstrainedGroupGradientMatrices and BackpropSnapshotPtr "
                   "don't match!"
                << std::endl;
      std::cout << "World vel-pos Jacobian: " << std::endl
                << worldVelPos << std::endl;
      std::cout << "Group vel-pos Jacobian: " << std::endl
                << groupVelPos << std::endl;
      return false;
    }

    Eigen::MatrixXs groupPosVel = group->getPosVelJacobian(world);
    Eigen::MatrixXs worldPosVel = classicPtr->getPosVelJacobian(world);
    if (!equals(groupPosVel, worldPosVel, 0))
    {
      std::cout << "ConstrainedGroupGradientMatrices and BackpropSnapshotPtr "
                   "don't match!"
                << std::endl;
      std::cout << "World pos-vel Jacobian: " << std::endl
                << worldPosVel << std::endl;
      std::cout << "Group pos-vel Jacobian: " << std::endl
                << groupPosVel << std::endl;
      std::cout << "Diff: " << std::endl
                << worldPosVel - groupPosVel << std::endl;

      // Find the constrained group variables
      const Eigen::MatrixXs& groupA_c = group->getClampingConstraintMatrix();
      const Eigen::MatrixXs& groupA_ub = group->getUpperBoundConstraintMatrix();
      const Eigen::MatrixXs& groupE = group->getUpperBoundMappingMatrix();
      Eigen::MatrixXs groupA_c_ub_E = groupA_c + groupA_ub * groupE;
      Eigen::VectorXs groupTau = group->mPreStepTorques;
      Eigen::VectorXs groupC
          = group->getCoriolisAndGravityAndExternalForces(world);
      const Eigen::VectorXs& groupF_c = group->getClampingConstraintImpulses();
      s_t dt = world->getTimeStep();
      Eigen::MatrixXs group_dM = group->getJacobianOfMinv(
          world,
          dt * (groupTau - groupC) + groupA_c_ub_E * groupF_c,
          WithRespectTo::POSITION);
      // Do the same thing with the backprop snapshot
      const Eigen::MatrixXs& worldA_c
          = classicPtr->getClampingConstraintMatrix(world);
      const Eigen::MatrixXs& worldA_ub
          = classicPtr->getUpperBoundConstraintMatrix(world);
      const Eigen::MatrixXs& worldE = classicPtr->getUpperBoundMappingMatrix();
      Eigen::MatrixXs worldA_c_ub_E = worldA_c + worldA_ub * worldE;
      Eigen::VectorXs worldTau = classicPtr->getPreStepTorques();
      Eigen::VectorXs worldC = world->getCoriolisAndGravityAndExternalForces();
      const Eigen::VectorXs& worldF_c
          = classicPtr->getClampingConstraintImpulses();
      Eigen::MatrixXs world_dM = classicPtr->getJacobianOfMinv(
          world,
          dt * (worldTau - worldC) + worldA_c_ub_E * worldF_c,
          WithRespectTo::POSITION);

      std::cout << "World dM Jacobian: " << std::endl << world_dM << std::endl;
      std::cout << "Group dM Jacobian: " << std::endl << group_dM << std::endl;
      std::cout << "Diff: " << std::endl << world_dM - group_dM << std::endl;

      std::cout << "World f_c: " << std::endl << worldF_c << std::endl;
      std::cout << "Group f_c: " << std::endl << groupF_c << std::endl;
      std::cout << "Diff: " << std::endl << worldF_c - groupF_c << std::endl;

      Eigen::MatrixXs group_dA_c
          = group->getJacobianOfClampingConstraints(world, groupF_c);
      Eigen::MatrixXs world_dA_c
          = classicPtr->getJacobianOfClampingConstraints(world, worldF_c);

      std::cout << "World dA_c: " << std::endl << world_dA_c << std::endl;
      std::cout << "Group dA_c: " << std::endl << group_dA_c << std::endl;
      std::cout << "Diff: " << std::endl
                << world_dA_c - group_dA_c << std::endl;

      Eigen::MatrixXs group_dF_c
          = group->getJacobianOfConstraintForce(world, WithRespectTo::POSITION);
      Eigen::MatrixXs world_dF_c = classicPtr->getJacobianOfConstraintForce(
          world, WithRespectTo::POSITION);

      std::cout << "World dF_c: " << std::endl << world_dF_c << std::endl;
      std::cout << "Group dF_c: " << std::endl << group_dF_c << std::endl;
      std::cout << "Diff: " << std::endl
                << world_dF_c - group_dF_c << std::endl;

      Eigen::VectorXs group_b = group->getClampingConstraintRelativeVels();
      Eigen::MatrixXs group_dQ_b
          = group->getJacobianOfLCPConstraintMatrixClampingSubset(
              world, group_b, WithRespectTo::POSITION);

      Eigen::VectorXs world_b = classicPtr->getClampingConstraintRelativeVels();
      Eigen::MatrixXs world_dQ_b
          = classicPtr->getJacobianOfLCPConstraintMatrixClampingSubset(
              world, world_b, WithRespectTo::POSITION);

      std::cout << "World b: " << std::endl << world_b << std::endl;
      std::cout << "Group b: " << std::endl << group_b << std::endl;
      std::cout << "Diff: " << std::endl << world_b - group_b << std::endl;

      std::cout << "World dQ_b: " << std::endl << world_dQ_b << std::endl;
      std::cout << "Group dQ_b: " << std::endl << group_dQ_b << std::endl;
      std::cout << "Diff: " << std::endl
                << world_dQ_b - group_dQ_b << std::endl;

      Eigen::MatrixXs group_dB = group->getJacobianOfLCPOffsetClampingSubset(
          world, WithRespectTo::POSITION);
      Eigen::MatrixXs world_dB
          = classicPtr->getJacobianOfLCPOffsetClampingSubset(
              world, WithRespectTo::POSITION);

      std::cout << "World dB: " << std::endl << world_dB << std::endl;
      std::cout << "Group dB: " << std::endl << group_dB << std::endl;
      std::cout << "Diff: " << std::endl << world_dB - group_dB << std::endl;

      Eigen::MatrixXs group_posC
          = group->getJacobianOfC(world, WithRespectTo::POSITION);
      Eigen::MatrixXs world_posC
          = classicPtr->getJacobianOfC(world, WithRespectTo::POSITION);

      std::cout << "World pos-C: " << std::endl << world_posC << std::endl;
      std::cout << "Group pos-C: " << std::endl << group_posC << std::endl;
      std::cout << "Diff: " << std::endl
                << world_posC - group_posC << std::endl;

      return false;
    }

    Eigen::MatrixXs groupVelVel = group->getVelVelJacobian(world);
    Eigen::MatrixXs worldVelVel = classicPtr->getVelVelJacobian(world);
    if (!equals(groupVelVel, worldVelVel, 0))
    {
      std::cout << "ConstrainedGroupGradientMatrices and BackpropSnapshotPtr "
                   "don't match!"
                << std::endl;
      std::cout << "World vel-vel Jacobian: " << std::endl
                << worldVelVel << std::endl;
      std::cout << "Group vel-vel Jacobian: " << std::endl
                << groupVelVel << std::endl;
      std::cout << "Diff range:" << std::endl
                << (worldVelVel - groupVelVel).minCoeff() << " to "
                << (worldVelVel - groupVelVel).maxCoeff() << std::endl;
      return false;
    }

    Eigen::MatrixXs groupForceVel = group->getControlForceVelJacobian(world);
    Eigen::MatrixXs worldForceVel
        = classicPtr->getControlForceVelJacobian(world);
    if (!equals(groupForceVel, worldForceVel, 0))
    {
      std::cout << "ConstrainedGroupGradientMatrices and BackpropSnapshotPtr "
                   "don't match!"
                << std::endl;
      std::cout << "World force-vel Jacobian: " << std::endl
                << worldForceVel << std::endl;
      std::cout << "Group force-vel Jacobian: " << std::endl
                << groupForceVel << std::endl;
      std::cout << "Diff range:" << std::endl
                << (worldForceVel - groupForceVel).minCoeff() << " to "
                << (worldForceVel - groupForceVel).maxCoeff() << std::endl;
      return false;
    }
  }

  snapshot.restore();

  return true;
}

bool verifyAnalyticalBackpropInstance(
    WorldPtr world,
    const neural::BackpropSnapshotPtr& classicPtr,
    const VectorXs& phaseSpace)
{
  LossGradient nextTimeStep;
  nextTimeStep.lossWrtPosition = phaseSpace.segment(0, phaseSpace.size() / 2);
  nextTimeStep.lossWrtVelocity
      = phaseSpace.segment(phaseSpace.size() / 2, phaseSpace.size() / 2);

  LossGradient thisTimeStep;
  classicPtr->backprop(world, thisTimeStep, nextTimeStep);
  LossGradientHighLevelAPI thisTimestepHighLevel
      = classicPtr->backpropState(world, phaseSpace);

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
  VectorXs lossWrtThisPosition =
      // p_t --> p_t+1
      (classicPtr->getPosPosJacobian(world).transpose()
       * nextTimeStep.lossWrtPosition)
      // p_t --> v_t+1
      + (classicPtr->getPosVelJacobian(world).transpose()
         * nextTimeStep.lossWrtVelocity);

  // v_t
  VectorXs lossWrtThisVelocity =
      // v_t --> v_t+1
      (classicPtr->getVelVelJacobian(world).transpose()
       * nextTimeStep.lossWrtVelocity)
      // v_t --> p_t+1
      + (classicPtr->getVelPosJacobian(world).transpose()
         * nextTimeStep.lossWrtPosition);

  // f_t
  VectorXs lossWrtThisTorque =
      // f_t --> v_t+1
      classicPtr->getControlForceVelJacobian(world).transpose()
      * nextTimeStep.lossWrtVelocity;

  // Trim gradients to the box constraints

  Eigen::VectorXs pos = world->getPositions();
  Eigen::VectorXs posUpperLimits = world->getPositionUpperLimits();
  Eigen::VectorXs posLowerLimits = world->getPositionLowerLimits();
  Eigen::VectorXs vels = world->getVelocities();
  Eigen::VectorXs velUpperLimits = world->getVelocityUpperLimits();
  Eigen::VectorXs velLowerLimits = world->getVelocityLowerLimits();
  Eigen::VectorXs forces = world->getControlForces();
  Eigen::VectorXs forceUpperLimits = world->getControlForceUpperLimits();
  Eigen::VectorXs forceLowerLimits = world->getControlForceLowerLimits();
  for (int i = 0; i < world->getNumDofs(); i++)
  {
    // Clip position gradients

    if ((pos(i) == posLowerLimits(i)) && (lossWrtThisPosition(i) > 0))
    {
      lossWrtThisPosition(i) = 0;
    }
    if ((pos(i) == posUpperLimits(i)) && (lossWrtThisPosition(i) < 0))
    {
      lossWrtThisPosition(i) = 0;
    }

    // Clip velocity gradients

    if ((vels(i) == velLowerLimits(i)) && (lossWrtThisVelocity(i) > 0))
    {
      lossWrtThisVelocity(i) = 0;
    }
    if ((vels(i) == velUpperLimits(i)) && (lossWrtThisVelocity(i) < 0))
    {
      lossWrtThisVelocity(i) = 0;
    }

    // Clip force gradients

    if ((forces(i) == forceLowerLimits(i)) && (lossWrtThisTorque(i) > 0))
    {
      lossWrtThisTorque(i) = 0;
    }
    if ((forces(i) == forceUpperLimits(i)) && (lossWrtThisTorque(i) < 0))
    {
      lossWrtThisTorque(i) = 0;
    }
  }

  Eigen::VectorXs stateLossFromAnalytical = Eigen::VectorXs::Zero(
      thisTimeStep.lossWrtPosition.size()
      + thisTimeStep.lossWrtVelocity.size());
  stateLossFromAnalytical.head(thisTimeStep.lossWrtPosition.size())
      = thisTimeStep.lossWrtPosition;
  stateLossFromAnalytical.tail(thisTimeStep.lossWrtVelocity.size())
      = thisTimeStep.lossWrtVelocity;
  if (!equals(
          stateLossFromAnalytical, thisTimestepHighLevel.lossWrtState, 1e-10))
  {
    std::cout
        << "backpropState() result disagrees with backprop() on state grad!"
        << std::endl;
    std::cout << "backprop() version:" << std::endl
              << stateLossFromAnalytical << std::endl;
    std::cout << "backpropState() version:" << std::endl
              << thisTimestepHighLevel.lossWrtState << std::endl;
    std::cout << "diff:" << std::endl
              << stateLossFromAnalytical - thisTimestepHighLevel.lossWrtState
              << std::endl;
    return false;
  }
  if (!equals(
          thisTimeStep.lossWrtTorque,
          thisTimestepHighLevel.lossWrtAction,
          1e-10))
  {
    std::cout
        << "backpropState() result disagrees with backprop() on action grad!"
        << std::endl;
    std::cout << "backprop() version:" << std::endl
              << thisTimeStep.lossWrtTorque << std::endl;
    std::cout << "backpropState() version:" << std::endl
              << thisTimestepHighLevel.lossWrtAction << std::endl;
    std::cout << "diff:" << std::endl
              << thisTimeStep.lossWrtTorque
                     - thisTimestepHighLevel.lossWrtAction
              << std::endl;
    return false;
  }

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
                << classicPtr->getJacobianOfC(world, WithRespectTo::POSITION)
                << std::endl;
      std::cout << "Brute force: pos-pos Jac:" << std::endl
                << classicPtr->getPosPosJacobian(world) << std::endl;
    }

    if (!equals(lossWrtThisVelocity, thisTimeStep.lossWrtVelocity, 1e-5))
    {
      std::cout << "-----" << std::endl;

      Eigen::MatrixXs velVelJac = classicPtr->getVelVelJacobian(world);

      Eigen::MatrixXs A_c = classicPtr->getClampingConstraintMatrix(world);
      Eigen::MatrixXs A_ub = classicPtr->getUpperBoundConstraintMatrix(world);
      Eigen::MatrixXs V_c
          = classicPtr->getMassedClampingConstraintMatrix(world);
      Eigen::MatrixXs V_ub
          = classicPtr->getMassedUpperBoundConstraintMatrix(world);
      Eigen::MatrixXs B = classicPtr->getBounceDiagonals().asDiagonal();
      Eigen::MatrixXs E = classicPtr->getUpperBoundMappingMatrix();
      Eigen::MatrixXs P_c = classicPtr->getProjectionIntoClampsMatrix(world);
      Eigen::MatrixXs Minv = classicPtr->getInvMassMatrix(world);
      Eigen::MatrixXs parts1 = A_c + A_ub * E;
      Eigen::MatrixXs parts2 = world->getTimeStep() * Minv * parts1 * P_c;

      std::cout << "Brute force A_c*z:" << std::endl
                << parts2.transpose() * nextTimeStep.lossWrtVelocity
                << std::endl;

      // Classic formulation

      Eigen::MatrixXs classicInnerPart
          = A_c.transpose().eval() * Minv * (A_c + A_ub * E);
      if (classicInnerPart.size() > 0)
      {
        Eigen::MatrixXs classicInnerPartInv
            = classicInnerPart.completeOrthogonalDecomposition()
                  .pseudoInverse();
        Eigen::MatrixXs classicRightPart = B * A_c.transpose().eval();
        Eigen::MatrixXs classicLeftPart = Minv * (A_c + A_ub * E);
        Eigen::MatrixXs classicComplete
            = classicLeftPart * classicInnerPart * classicRightPart;

        std::cout << "Classic brute force A_c*z:" << std::endl
                  << classicComplete.transpose() * nextTimeStep.lossWrtVelocity
                  << std::endl;

        // Massed formulation

        Eigen::MatrixXs massedInnerPart
            = A_c.transpose().eval() * (V_c + V_ub * E);
        Eigen::MatrixXs massedInnerPartInv
            = massedInnerPart.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXs massedRightPart = B * A_c.transpose().eval();
        Eigen::MatrixXs massedLeftPart = V_c + V_ub * E;
        Eigen::MatrixXs massedComplete
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
      }
      Eigen::MatrixXs V_c_recovered = Minv * A_c;
      if (!equals(V_c_recovered, V_c, 1e-8))
      {
        std::cout << "Mismatch at V_c == Minv * A_c!" << std::endl;
        std::cout << "V_c:" << std::endl << V_c << std::endl;
        std::cout << "A_c:" << std::endl << A_c << std::endl;
        std::cout << "Minv:" << std::endl << Minv << std::endl;
        std::cout << "Minv * A_c:" << std::endl << V_c_recovered << std::endl;
      }
      Eigen::MatrixXs V_ub_recovered = Minv * A_ub;
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
                << -classicPtr->getControlForceVelJacobian(world).transpose()
                       * nextTimeStep.lossWrtVelocity
                << std::endl;
      std::cout << "3: -((force-vel) * (vel-C))^T * nextLossWrtVel:"
                << std::endl
                << -(classicPtr->getControlForceVelJacobian(world)
                     * classicPtr->getJacobianOfC(
                         world, WithRespectTo::VELOCITY))
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
                << (classicPtr->getControlForceVelJacobian(world).transpose())
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

  // This can often be the root of the problem, so verify that the constraint
  // group Jacobians match the Jacobians of the BackpropSnapshot.
  if (!verifyConstraintGroupSubJacobians(world, classicPtr))
    return false;

  VectorXs phaseSpace = VectorXs::Zero(world->getNumDofs() * 2);

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
  phaseSpace = VectorXs::Zero(world->getNumDofs() * 2);
  if (!verifyAnalyticalBackpropInstance(world, classicPtr, phaseSpace))
    return false;

  // Test all "1"s
  phaseSpace = VectorXs::Ones(world->getNumDofs() * 2);
  if (!verifyAnalyticalBackpropInstance(world, classicPtr, phaseSpace))
    return false;

  return true;
}

LossGradient computeBruteForceGradient(
    WorldPtr world, std::size_t timesteps, std::function<s_t(WorldPtr)> loss)
{
  RestorableSnapshot snapshot(world);

  std::size_t n = world->getNumDofs();
  LossGradient grad;
  grad.lossWrtPosition = Eigen::VectorXs(n);
  grad.lossWrtVelocity = Eigen::VectorXs(n);
  grad.lossWrtTorque = Eigen::VectorXs(n);

  for (std::size_t k = 0; k < timesteps; k++)
    world->step();
  s_t defaultLoss = loss(world);
  snapshot.restore();

  Eigen::VectorXs originalPos = world->getPositions();
  Eigen::VectorXs originalVel = world->getVelocities();
  Eigen::VectorXs originalForce = world->getControlForces();

  s_t EPSILON = 1e-7;

  for (std::size_t i = 0; i < n; i++)
  {
    Eigen::VectorXs tweakedPos = originalPos;
    tweakedPos(i) += EPSILON;

    snapshot.restore();
    world->setPositions(tweakedPos);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step(true);
    grad.lossWrtPosition(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXs tweakedVel = originalVel;
    tweakedVel(i) += EPSILON;

    snapshot.restore();
    world->setVelocities(tweakedVel);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step(true);
    grad.lossWrtVelocity(i) = (loss(world) - defaultLoss) / EPSILON;

    Eigen::VectorXs tweakedForce = originalForce;
    tweakedForce(i) += EPSILON;

    snapshot.restore();
    world->setControlForces(tweakedForce);
    for (std::size_t k = 0; k < timesteps; k++)
      world->step(true);
    grad.lossWrtTorque(i) = (loss(world) - defaultLoss) / EPSILON;
  }

  snapshot.restore();
  return grad;
}

bool verifyGradientBackprop(
    WorldPtr world, std::size_t timesteps, std::function<s_t(WorldPtr)> loss)
{
  RestorableSnapshot snapshot(world);

  std::vector<BackpropSnapshotPtr> backpropSnapshots;
  std::vector<RestorableSnapshot> restorableSnapshots;
  backpropSnapshots.reserve(timesteps);
  for (std::size_t i = 0; i < timesteps; i++)
  {
    restorableSnapshots.push_back(RestorableSnapshot(world));
    backpropSnapshots.push_back(forwardPass(world, false));
  }

  // Get the loss gradient at the final timestep (by brute force) to
  // initialize an analytical backwards pass
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
    Eigen::MatrixXs posPos = backpropSnapshots[i]->getPosPosJacobian(world);
    // v_t+1 <-- p_t
    Eigen::MatrixXs posVel = backpropSnapshots[i]->getPosVelJacobian(world);
    // p_t+1 <-- v_t
    Eigen::MatrixXs velPos = backpropSnapshots[i]->getVelPosJacobian(world);
    // v_t+1 <-- v_t
    Eigen::MatrixXs velVel = backpropSnapshots[i]->getVelVelJacobian(world);

    // p_t+1 <-- p_t
    Eigen::MatrixXs posPosFD
        = backpropSnapshots[i]->finiteDifferencePosPosJacobian(world, 1);
    // v_t+1 <-- p_t
    Eigen::MatrixXs posVelFD
        = backpropSnapshots[i]->finiteDifferencePosVelJacobian(world);
    // p_t+1 <-- v_t
    Eigen::MatrixXs velPosFD
        = backpropSnapshots[i]->finiteDifferenceVelPosJacobian(world, 1);
    // v_t+1 <-- v_t
    Eigen::MatrixXs velVelFD
        = backpropSnapshots[i]->finiteDifferenceVelVelJacobian(world);

    /*
    s_t diffPosPos = (posPos - posPosFD).norm();
    s_t diffPosVel = (posVel - posVelFD).norm();
    s_t diffVelPos = (velPos - velPosFD).norm();
    s_t diffVelVel = (velVel - velVelFD).norm();
    */

    /*
    std::cout << "Jacobian error at step:" << numSteps << ": " << diffPosPos
              << ", " << diffPosVel << ", " << diffVelPos << ", " <<
    diffVelVel
              << std::endl;
    */

    LossGradient analyticalWithBruteForce;
    backpropSnapshots[i]->backprop(world, analyticalWithBruteForce, bruteForce);
    /*
    analyticalWithBruteForce.lossWrtPosition
        = posPos.transpose() * bruteForce.lossWrtPosition
          + posVel.transpose() * bruteForce.lossWrtVelocity;
    analyticalWithBruteForce.lossWrtVelocity
        = velPos.transpose() * bruteForce.lossWrtPosition
          + velVel.transpose() * bruteForce.lossWrtVelocity;
    */

    bruteForce = bruteForceThisTimestep;

    /*
    std::cout
        << "Backprop error at step:" << numSteps << ": "
        << (analytical.lossWrtPosition - bruteForce.lossWrtPosition).norm()
        << ", "
        << (analytical.lossWrtVelocity - bruteForce.lossWrtVelocity).norm()
        << ", " << (analytical.lossWrtTorque -
    bruteForce.lossWrtTorque).norm()
        << std::endl;
    */

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

bool verifyWorldSpaceToVelocitySpatial(
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXs worldVel = worldVelMatrix.col(0);

  Eigen::VectorXs bruteWorldVel = Eigen::VectorXs::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() <<
    // std::endl;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector6s bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity());
      /*
      Eigen::Vector6s analyticalVel = worldVel.segment(cursor, 6);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXs worldVel = worldVelMatrix.col(0);

  Eigen::VectorXs bruteWorldVel = Eigen::VectorXs::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() <<
    // std::endl;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector3s bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity())
                .tail<3>();

      /*
      Eigen::Vector3s analyticalVel = worldVel.segment(cursor, 3);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldPosMatrix = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::COM_POS, false);
  if (worldPosMatrix.cols() != 1)
    return false;
  Eigen::VectorXs worldPos = worldPosMatrix.col(0);

  Eigen::VectorXs bruteWorldCOMPos = Eigen::VectorXs::Zero(worldPos.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() <<
    // std::endl;
    Eigen::Vector3s bruteCOMPos = skel->getCOM();

    /*
    Eigen::Vector3s analyticalVel = worldPos.segment(cursor, 3);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXs worldVel = worldVelMatrix.col(0);

  Eigen::VectorXs bruteWorldCOMVel = Eigen::VectorXs::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() <<
    // std::endl;
    Eigen::Vector3s bruteCOMVel = Eigen::Vector3s::Zero();
    s_t totalMass = 0.0;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector3s bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity())
                .tail<3>();
      bruteCOMVel += bruteVel * node->getMass();
      totalMass += node->getMass();
    }
    bruteCOMVel /= totalMass;

    /*
    Eigen::Vector3s analyticalVel = worldVel.segment(cursor, 3);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldVelMatrix = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);
  if (worldVelMatrix.cols() != 1)
    return false;
  Eigen::VectorXs worldVel = worldVelMatrix.col(0);

  Eigen::VectorXs bruteWorldCOMVel = Eigen::VectorXs::Zero(worldVel.size());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    // std::cout << "Vels: " << std::endl << skel->getVelocities() <<
    // std::endl;
    Eigen::Vector6s bruteCOMVel = Eigen::Vector6s::Zero();
    s_t totalMass = 0.0;
    for (std::size_t k = 0; k < skel->getNumBodyNodes(); k++)
    {
      BodyNode* node = skel->getBodyNode(k);
      Eigen::Vector6s bruteVel
          = math::AdR(node->getWorldTransform(), node->getSpatialVelocity());
      bruteCOMVel += bruteVel * node->getMass();
      totalMass += node->getMass();
    }
    bruteCOMVel /= totalMass;

    /*
    Eigen::Vector6s analyticalVel = worldVel.segment(cursor, 6);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
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

  Eigen::VectorXs originalWorldPos = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::POS_SPATIAL, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(position.size()) * 1e-6;
  Eigen::VectorXs perturbedPos = position + perturbation;

  Eigen::VectorXs perturbedWorldPos = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::POS_SPATIAL, false);
  Eigen::MatrixXs skelSpatialJac
      = jointPosToWorldSpatialJacobian(world->getSkeleton(0), bodyNodes);
  Eigen::VectorXs expectedPerturbation = skelSpatialJac * perturbation;

  /*
  std::cout << "World perturbation: " << std::endl
            << worldPerturbation << std::endl;
  std::cout << "Expected perturbation: " << std::endl
            << expectedPerturbation << std::endl;
            */

  Eigen::VectorXs worldPerturbation = perturbedWorldPos - originalWorldPos;
  Eigen::VectorXs recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::POS_SPATIAL, true);

  if (!equals(perturbation, recoveredPerturbation, 1e-8))
  {
    std::cout << "backprop() POS_SPATIAL failed!" << std::endl;
    Eigen::MatrixXs skelSpatialJac2 = jointPosToWorldSpatialJacobian(
        world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
    Eigen::MatrixXs perturbations(worldPerturbation.size(), 2);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();

  // Delete the 2nd body node, arbitrarily, to force some shuffling
  bodyNodes.erase(bodyNodes.begin()++);
  // Shuffle the remaining elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  Eigen::VectorXs originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(position.size()) * 1e-6;
  Eigen::VectorXs perturbedVel = velocity + perturbation;

  Eigen::VectorXs perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXs worldPerturbation = perturbedWorldVel - originalWorldVel;
  Eigen::VectorXs recoveredPerturbation = convertJointSpaceToWorldSpace(
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

enum MappingTestComponent
{
  POSITION,
  VELOCITY,
  FORCE
};

Eigen::VectorXs getTestComponentWorld(
    WorldPtr world, MappingTestComponent component)
{
  if (component == MappingTestComponent::POSITION)
    return world->getPositions();
  else if (component == MappingTestComponent::VELOCITY)
    return world->getVelocities();
  else if (component == MappingTestComponent::FORCE)
    return world->getControlForces();
  else
    assert(false && "Unrecognized component value in getTestComponent()");
  throw std::runtime_error{"Execution should never reach this point"};
}

void setTestComponentWorld(
    WorldPtr world, MappingTestComponent component, const Eigen::VectorXs& val)
{
  if (component == MappingTestComponent::POSITION)
    world->setPositions(val);
  else if (component == MappingTestComponent::VELOCITY)
    world->setVelocities(val);
  else if (component == MappingTestComponent::FORCE)
    world->setControlForces(val);
  else
    assert(false && "Unrecognized component value in getTestComponent()");
}

Eigen::VectorXs getTestComponentMapping(
    std::shared_ptr<Mapping> mapping,
    WorldPtr world,
    MappingTestComponent component)
{
  if (component == MappingTestComponent::POSITION)
    return mapping->getPositions(world);
  else if (component == MappingTestComponent::VELOCITY)
    return mapping->getVelocities(world);
  else if (component == MappingTestComponent::FORCE)
    return mapping->getControlForces(world);
  else
    assert(false && "Unrecognized component value in getTestComponent()");
  throw std::runtime_error{"Execution should never reach this point"};
}

int getTestComponentMappingDim(
    std::shared_ptr<Mapping> mapping,
    WorldPtr /* world */,
    MappingTestComponent component)
{
  if (component == MappingTestComponent::POSITION)
    return mapping->getPosDim();
  else if (component == MappingTestComponent::VELOCITY)
    return mapping->getVelDim();
  else if (component == MappingTestComponent::FORCE)
    return mapping->getControlForceDim();
  else
    assert(false && "Unrecognized component value in getTestComponent()");
  return 0;
}

Eigen::MatrixXs getTestComponentMappingIntoJac(
    std::shared_ptr<Mapping> mapping,
    WorldPtr world,
    MappingTestComponent component,
    MappingTestComponent wrt)
{
  if (component == MappingTestComponent::POSITION
      && wrt == MappingTestComponent::POSITION)
    return mapping->getRealPosToMappedPosJac(world);
  if (component == MappingTestComponent::POSITION
      && wrt == MappingTestComponent::VELOCITY)
    return mapping->getRealVelToMappedPosJac(world);
  else if (
      component == MappingTestComponent::VELOCITY
      && wrt == MappingTestComponent::VELOCITY)
    return mapping->getRealVelToMappedVelJac(world);
  else if (
      component == MappingTestComponent::VELOCITY
      && wrt == MappingTestComponent::POSITION)
    return mapping->getRealPosToMappedVelJac(world);
  else if (component == MappingTestComponent::FORCE)
    return mapping->getRealForceToMappedForceJac(world);
  else
    assert(false && "Unrecognized <component, wrt> pair in getTestComponent()");
  throw std::runtime_error{"Execution should never reach this point"};
}

void setTestComponentMapping(
    std::shared_ptr<Mapping> mapping,
    WorldPtr world,
    MappingTestComponent component,
    Eigen::VectorXs val)
{
  if (component == MappingTestComponent::POSITION)
    mapping->setPositions(world, val);
  else if (component == MappingTestComponent::VELOCITY)
    mapping->setVelocities(world, val);
  else if (component == MappingTestComponent::FORCE)
    mapping->setControlForces(world, val);
  else
    assert(false && "Unrecognized component value in getTestComponent()");
}

std::string getComponentName(MappingTestComponent component)
{
  if (component == MappingTestComponent::POSITION)
    return "POSITION";
  else if (component == MappingTestComponent::VELOCITY)
    return "VELOCITY";
  else if (component == MappingTestComponent::FORCE)
    return "FORCE";
  else
    assert(false && "Unrecognized component value in getTestComponent()");
  throw std::runtime_error{"Execution should never reach this point"};
}

bool verifyMappingSetGet(
    WorldPtr world,
    std::shared_ptr<Mapping> mapping,
    MappingTestComponent component)
{
  RestorableSnapshot snapshot(world);

  srand(42);

  // Pick a random target, set to it, and see if there are any near-neighbors
  // that are better
  for (int i = 0; i < 5; i++)
  {
    Eigen::VectorXs beforeTest = getTestComponentWorld(world, component);

    Eigen::VectorXs target = Eigen::VectorXs::Random(mapping->getPosDim());
    s_t originalLoss;

    setTestComponentMapping(mapping, world, component, target);
    Eigen::VectorXs original = getTestComponentWorld(world, component);
    Eigen::VectorXs originalMapped
        = getTestComponentMapping(mapping, world, component);
    originalLoss = (originalMapped - target).squaredNorm();

    // Try a bunch of near neighbor perturbations
    for (int j = 0; j < 20; j++)
    {
      Eigen::VectorXs randomPerturbations
          = Eigen::VectorXs::Random(world->getNumDofs()) * 0.001;

      setTestComponentWorld(world, component, original + randomPerturbations);
      Eigen::VectorXs newMapped
          = getTestComponentMapping(mapping, world, component);
      s_t newLoss = (newMapped - target).squaredNorm();

      if (newLoss < originalLoss)
      {
        std::cout << "Found near neighbor that's better than original IK "
                     "solution for "
                  << getComponentName(component) << "!" << std::endl;
        std::cout << "Original loss: " << originalLoss << std::endl;
        std::cout << "New loss: " << newLoss << std::endl;
        std::cout << "Diff: " << (newLoss - originalLoss) << std::endl;
        std::cout << "Original mapped: " << originalMapped << std::endl;
        std::cout << "Target: " << target << std::endl;
        std::cout << "Original world: " << original << std::endl;
        std::cout << "Perturbation: " << randomPerturbations << std::endl;
        std::cout << "New mapped: " << newMapped << std::endl;
        std::cout << "****\nTo recreate:\n"
                  << std::endl
                  << "Eigen::VectorXs beforeTest = Eigen::VectorXs("
                  << beforeTest.size() << ");\nbeforeTest << ";
        for (int q = 0; q < beforeTest.size(); q++)
        {
          if (q > 0)
            std::cout << ", ";
          std::cout << beforeTest(q);
          if (q == beforeTest.size() - 1)
            std::cout << ";";
        }
        std::cout << std::endl
                  << "Eigen::VectorXs target = Eigen::VectorXs("
                  << target.size() << ");\ntarget << ";
        for (int q = 0; q < target.size(); q++)
        {
          if (q > 0)
            std::cout << ", ";
          std::cout << target(q);
          if (q == target.size() - 1)
            std::cout << ";";
        }
        return false;
      }
    }
  }

  snapshot.restore();
  return true;
}

bool verifyMappingIntoJacobian(
    WorldPtr world,
    std::shared_ptr<Mapping> mapping,
    MappingTestComponent component,
    MappingTestComponent wrt)
{
  RestorableSnapshot snapshot(world);

  int mappedDim = getTestComponentMappingDim(mapping, world, component);
  Eigen::MatrixXs analytical
      = getTestComponentMappingIntoJac(mapping, world, component, wrt);
  Eigen::MatrixXs bruteForce
      = Eigen::MatrixXs::Zero(mappedDim, world->getNumDofs());

  Eigen::VectorXs originalWorld = getTestComponentWorld(world, wrt);
  Eigen::VectorXs originalMapped
      = getTestComponentMapping(mapping, world, component);

  const s_t EPS = 1e-5;
  for (int i = 0; i < world->getNumDofs(); i++)
  {
    Eigen::VectorXs perturbedWorld = originalWorld;
    perturbedWorld(i) += EPS;
    setTestComponentWorld(world, wrt, perturbedWorld);
    Eigen::VectorXs perturbedMappedPos
        = getTestComponentMapping(mapping, world, component);

    perturbedWorld = originalWorld;
    perturbedWorld(i) -= EPS;
    setTestComponentWorld(world, wrt, perturbedWorld);
    Eigen::VectorXs perturbedMappedNeg
        = getTestComponentMapping(mapping, world, component);

    bruteForce.col(i) = (perturbedMappedPos - perturbedMappedNeg) / (2 * EPS);
  }

  if (!equals(bruteForce, analytical, 1e-8))
  {
    std::cout << "Got a bad Into Jac for mapped " << getComponentName(component)
              << " wrt world " << getComponentName(wrt) << "!" << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "Brute Force: " << std::endl << bruteForce << std::endl;
    std::cout << "Diff: " << (analytical - bruteForce) << std::endl;
    return false;
  }

  snapshot.restore();
  return true;
}

Eigen::MatrixXs getTimestepJacobian(
    WorldPtr world,
    std::shared_ptr<MappedBackpropSnapshot> snapshot,
    MappingTestComponent inComponent,
    MappingTestComponent outComponent)
{
  if (inComponent == MappingTestComponent::POSITION
      && outComponent == MappingTestComponent::POSITION)
  {
    return snapshot->getPosPosJacobian(world);
  }
  else if (
      inComponent == MappingTestComponent::POSITION
      && outComponent == MappingTestComponent::VELOCITY)
  {
    return snapshot->getPosVelJacobian(world);
  }
  else if (
      inComponent == MappingTestComponent::VELOCITY
      && outComponent == MappingTestComponent::POSITION)
  {
    return snapshot->getVelPosJacobian(world);
  }
  else if (
      inComponent == MappingTestComponent::VELOCITY
      && outComponent == MappingTestComponent::VELOCITY)
  {
    return snapshot->getVelVelJacobian(world);
  }
  else if (
      inComponent == MappingTestComponent::FORCE
      && outComponent == MappingTestComponent::VELOCITY)
  {
    return snapshot->getControlForceVelJacobian(world);
  }
  assert(false && "Unsupported combination of inComponent and outComponent in getTimestepJacobian()!");
  throw std::runtime_error{"Execution should never reach this point"};
}

bool verifyMapping(WorldPtr world, std::shared_ptr<Mapping> mapping)
{
  return verifyMappingSetGet(world, mapping, MappingTestComponent::POSITION)
         && verifyMappingSetGet(world, mapping, MappingTestComponent::VELOCITY)
         && verifyMappingSetGet(world, mapping, MappingTestComponent::FORCE)
         && verifyMappingIntoJacobian(
             world,
             mapping,
             MappingTestComponent::POSITION,
             MappingTestComponent::POSITION)
         && verifyMappingIntoJacobian(
             world,
             mapping,
             MappingTestComponent::POSITION,
             MappingTestComponent::VELOCITY)
         && verifyMappingIntoJacobian(
             world,
             mapping,
             MappingTestComponent::VELOCITY,
             MappingTestComponent::POSITION)
         && verifyMappingIntoJacobian(
             world,
             mapping,
             MappingTestComponent::VELOCITY,
             MappingTestComponent::VELOCITY)
         && verifyMappingIntoJacobian(
             world,
             mapping,
             MappingTestComponent::FORCE,
             MappingTestComponent::FORCE);
}

bool verifyIKMappingVelocity(WorldPtr world, std::shared_ptr<IKMapping> mapping)
{
  Eigen::VectorXs vels = mapping->getVelocities(world);
  Eigen::VectorXs velsByJac
      = mapping->getRealVelToMappedVelJac(world) * world->getVelocities();
  if (!equals(vels, velsByJac, 1e-10))
  {
    std::cout << "Is our vels Jacobian wrong? J*q_dot != v" << std::endl;
    std::cout << "J*q_dot:" << std::endl << velsByJac << std::endl;
    std::cout << "v:" << std::endl << vels << std::endl;
    std::cout << "diff:" << std::endl << velsByJac - vels << std::endl;
    return false;
  }
  return true;
}

bool verifyLinearIKMapping(WorldPtr world)
{
  std::vector<dynamics::BodyNode*> bodyNodes;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    for (dynamics::BodyNode* node : skel->getBodyNodes())
      bodyNodes.push_back(node);
  }

  srand(42);

  // Shuffle the elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  std::shared_ptr<IKMapping> mapping = std::make_shared<IKMapping>(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    mapping->addLinearBodyNode(node);
  }
  mapping->setIKIterationLimit(-1);
  return verifyIKMappingVelocity(world, mapping)
         && verifyMapping(world, mapping);
}

bool verifySpatialIKMapping(WorldPtr world)
{
  std::vector<dynamics::BodyNode*> bodyNodes;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    for (dynamics::BodyNode* node : skel->getBodyNodes())
      bodyNodes.push_back(node);
  }

  srand(42);

  // Shuffle the elements
  // std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  std::shared_ptr<IKMapping> mapping = std::make_shared<IKMapping>(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    mapping->addSpatialBodyNode(node);
  }
  mapping->setIKIterationLimit(-1);
  return verifyIKMappingVelocity(world, mapping)
         && verifyMapping(world, mapping);
}

bool verifyAngularIKMapping(WorldPtr world)
{
  std::vector<dynamics::BodyNode*> bodyNodes;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    for (dynamics::BodyNode* node : skel->getBodyNodes())
      bodyNodes.push_back(node);
  }

  srand(42);

  // Shuffle the elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  std::shared_ptr<IKMapping> mapping = std::make_shared<IKMapping>(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    mapping->addAngularBodyNode(node);
  }
  mapping->setIKIterationLimit(-1);
  return verifyIKMappingVelocity(world, mapping)
         && verifyMapping(world, mapping);
}

bool verifyRandomIKMapping(WorldPtr world)
{
  std::vector<dynamics::BodyNode*> bodyNodes;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);

    for (dynamics::BodyNode* node : skel->getBodyNodes())
      bodyNodes.push_back(node);
  }

  srand(42);

  // Shuffle the elements
  std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

  std::shared_ptr<IKMapping> mapping = std::make_shared<IKMapping>(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    int option = rand() % 4;
    if (option == 0)
    {
      mapping->addAngularBodyNode(node);
    }
    else if (option == 1)
    {
      mapping->addLinearBodyNode(node);
    }
    else if (option == 2)
    {
      mapping->addSpatialBodyNode(node);
    }
    else if (option == 3)
    {
      // Don't add node
    }
  }
  mapping->setIKIterationLimit(-1);
  return verifyIKMappingVelocity(world, mapping)
         && verifyMapping(world, mapping);
}

bool verifyIKMapping(WorldPtr world)
{
  return verifyLinearIKMapping(world) && verifyAngularIKMapping(world)
         && verifySpatialIKMapping(world) && verifyRandomIKMapping(world);
}

bool verifyIdentityMapping(WorldPtr world)
{
  std::shared_ptr<IdentityMapping> mapping
      = std::make_shared<IdentityMapping>(world);
  return verifyMapping(world, mapping);
}

bool verifyClosestIKPosition(WorldPtr world, Eigen::VectorXs position)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(position);
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    snapshot.restore();

    auto skel = world->getSkeleton(i);

    std::vector<dynamics::BodyNode*> bodyNodes
        = world->getSkeleton(i)->getBodyNodes();
    // Shuffle the elements
    std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

    IKMapping mapping(world);
    for (dynamics::BodyNode* node : bodyNodes)
    {
      mapping.addLinearBodyNode(node);
    }

    Eigen::VectorXs targetPos = Eigen::VectorXs::Random(mapping.getPosDim());
    mapping.setPositions(world, targetPos);
    Eigen::VectorXs originalMapped = mapping.getPositions(world);
    Eigen::VectorXs originalWorld = world->getPositions();

    s_t originalLoss = (originalMapped - targetPos).squaredNorm();

    // Try a bunch of near neighbor perturbations
    for (int i = 0; i < 20; i++)
    {
      Eigen::VectorXs randomPerturbations
          = Eigen::VectorXs::Random(world->getNumDofs()) * 0.001;
      world->setPositions(originalWorld + randomPerturbations);
      Eigen::VectorXs newMapped = mapping.getPositions(world);
      s_t newLoss = (newMapped - targetPos).squaredNorm();

      if (newLoss < originalLoss)
      {
        std::cout << "Found near neighbor that's better than original IK "
                     "position solution!"
                  << std::endl;
        std::cout << "Original loss: " << originalLoss << std::endl;
        std::cout << "New loss: " << newLoss << std::endl;
        std::cout << "Diff: " << (newLoss - originalLoss) << std::endl;
        std::cout << "Target: " << targetPos << std::endl;
        std::cout << "Original Mapped: " << originalMapped << std::endl;
        std::cout << "Original World: " << originalWorld << std::endl;
        std::cout << "Perturbation: " << randomPerturbations << std::endl;
        std::cout << "New Mapped: " << newMapped << std::endl;
        return false;
      }
    }
  }

  snapshot.restore();
  return true;
}

bool verifyClosestIKVelocity(WorldPtr world, Eigen::VectorXs velocity)
{
  RestorableSnapshot snapshot(world);

  world->setVelocities(velocity);
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    snapshot.restore();

    auto skel = world->getSkeleton(i);

    std::vector<dynamics::BodyNode*> bodyNodes
        = world->getSkeleton(i)->getBodyNodes();
    // Shuffle the elements
    std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

    IKMapping mapping(world);
    for (dynamics::BodyNode* node : bodyNodes)
    {
      mapping.addLinearBodyNode(node);
    }

    Eigen::VectorXs targetVel = Eigen::VectorXs::Random(mapping.getVelDim());
    mapping.setVelocities(world, targetVel);

    s_t originalLoss = (mapping.getVelocities(world) - targetVel).squaredNorm();

    // Try a bunch of near neighbor perturbations
    for (int i = 0; i < 20; i++)
    {
      Eigen::VectorXs randomPerturbations
          = Eigen::VectorXs::Random(world->getNumDofs()) * 0.001;
      world->setVelocities(velocity + randomPerturbations);
      s_t newLoss = (mapping.getVelocities(world) - targetVel).squaredNorm();

      if (newLoss < originalLoss)
      {
        std::cout << "Found near neighbor that's better than original IK "
                     "velocity solution!"
                  << std::endl;
        std::cout << "Original loss: " << originalLoss << std::endl;
        std::cout << "New loss: " << newLoss << std::endl;
        std::cout << "Diff: " << (newLoss - originalLoss) << std::endl;
        return false;
      }
    }
  }

  snapshot.restore();
  return true;
}

bool verifyLinearJacobian(
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    world->setPositions(position);
    world->setVelocities(velocity);

    auto skel = world->getSkeleton(i);

    std::vector<dynamics::BodyNode*> bodyNodes
        = world->getSkeleton(i)->getBodyNodes();
    // Shuffle the elements
    std::random_shuffle(bodyNodes.begin(), bodyNodes.end());

    IKMapping mapping(world);
    for (dynamics::BodyNode* node : bodyNodes)
    {
      mapping.addLinearBodyNode(node);
    }

    Eigen::MatrixXs analytical = mapping.getRealPosToMappedPosJac(world);

    // Compute a brute force version
    Eigen::VectorXs originalPos = skel->getPositions();
    Eigen::VectorXs originalVel = skel->getVelocities();
    Eigen::VectorXs originalWorldPos = mapping.getPositions(world);
    Eigen::VectorXs originalWorldVel = mapping.getVelocities(world);
    Eigen::MatrixXs bruteForce
        = Eigen::MatrixXs::Zero(analytical.rows(), analytical.cols());
    const s_t EPS = 1e-5;
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      Eigen::VectorXs perturbedPos = originalPos;
      perturbedPos(j) += EPS;
      skel->setPositions(perturbedPos);
      Eigen::VectorXs posColumn = mapping.getPositions(world);

      Eigen::VectorXs perturbedNeg = originalPos;
      perturbedNeg(j) -= EPS;
      skel->setPositions(perturbedNeg);
      Eigen::VectorXs negColumn = mapping.getPositions(world);

      skel->setPositions(originalPos);

      bruteForce.block(0, j, bruteForce.rows(), 1)
          = (posColumn - negColumn) / (2 * EPS);
    }

    if (!equals(bruteForce, analytical, 1e-5))
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
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

    IKMapping mapping(world);
    for (dynamics::BodyNode* node : bodyNodes)
    {
      mapping.addSpatialBodyNode(node);
    }

    Eigen::MatrixXs analytical = mapping.getRealPosToMappedPosJac(world);

    // Compute a brute force version
    Eigen::VectorXs originalPos = skel->getPositions();
    Eigen::VectorXs originalVel = skel->getVelocities();
    Eigen::VectorXs originalWorldPos = mapping.getPositions(world);
    Eigen::VectorXs originalWorldVel = mapping.getVelocities(world);
    Eigen::MatrixXs bruteForce
        = Eigen::MatrixXs::Zero(analytical.rows(), analytical.cols());
    const s_t EPS = 1e-6;
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      Eigen::VectorXs perturbedPos = originalPos;
      perturbedPos(j) += EPS;
      skel->setPositions(perturbedPos);
      Eigen::VectorXs plusPos = mapping.getPositions(world);

      perturbedPos = originalPos;
      perturbedPos(j) -= EPS;
      skel->setPositions(perturbedPos);
      Eigen::VectorXs minusPos = mapping.getPositions(world);

      Eigen::VectorXs posColumn = (plusPos - minusPos) / (2 * EPS);

      skel->setPositions(originalPos);

      bruteForce.block(0, j, bruteForce.rows(), 1) = posColumn;
    }

    if (!equals(bruteForce, analytical, 1e-5))
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  IKMapping linearMapping(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    linearMapping.addLinearBodyNode(node);
  }
  IKMapping spatialMapping(world);
  for (dynamics::BodyNode* node : bodyNodes)
  {
    spatialMapping.addSpatialBodyNode(node);
  }
  Eigen::VectorXs originalWorldPos = linearMapping.getPositions(world);
  Eigen::VectorXs originalWorldSpatial = spatialMapping.getPositions(world);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(position.size()) * 1e-4;
  Eigen::VectorXs perturbedPos = position + perturbation;

  Eigen::MatrixXs skelLinearJac = linearMapping.getRealPosToMappedPosJac(world);
  Eigen::VectorXs expectedPerturbation = skelLinearJac * perturbation;

  Eigen::MatrixXs skelSpatialJac
      = spatialMapping.getRealPosToMappedPosJac(world);
  Eigen::VectorXs expectedPerturbationSpatial = skelSpatialJac * perturbation;

  world->setPositions(perturbedPos);
  Eigen::VectorXs perturbedWorldPos = linearMapping.getPositions(world);
  Eigen::VectorXs perturbedWorldSpatial = spatialMapping.getPositions(world);

  Eigen::VectorXs worldPerturbation = perturbedWorldPos - originalWorldPos;
  Eigen::VectorXs worldPerturbationSpatial
      = perturbedWorldSpatial - originalWorldSpatial;

  if (!equals(worldPerturbation, expectedPerturbation, 1e-5)
      || !equals(worldPerturbationSpatial, expectedPerturbationSpatial, 1e-5))
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
    return false;
  }

  return true;
}

bool verifyBackpropWorldSpaceVelocityToPosition(
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXs originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  Eigen::VectorXs originalWorldSpatial = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(velocity.size()) * 1e-4;
  Eigen::VectorXs perturbedVel = velocity + perturbation;

  Eigen::MatrixXs skelLinearJac = jointVelToWorldLinearJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXs expectedPerturbation = skelLinearJac * perturbation;

  Eigen::MatrixXs skelSpatialJac = jointVelToWorldSpatialJacobian(
      world->getSkeleton(0), world->getSkeleton(0)->getBodyNodes());
  Eigen::VectorXs expectedPerturbationSpatial = skelSpatialJac * perturbation;

  Eigen::VectorXs perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_LINEAR, false);
  Eigen::VectorXs perturbedWorldSpatial = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::VEL_SPATIAL, false);

  Eigen::VectorXs worldPerturbation = perturbedWorldVel - originalWorldVel;
  Eigen::VectorXs worldPerturbationSpatial
      = perturbedWorldSpatial - originalWorldSpatial;

  Eigen::VectorXs recoveredPerturbation = convertJointSpaceToWorldSpace(
      world, worldPerturbation, bodyNodes, ConvertToSpace::VEL_LINEAR, true);

  Eigen::VectorXs expectedPerturbationFromSpatial
      = Eigen::VectorXs(expectedPerturbationSpatial.size() / 2);
  for (int i = 0; i < expectedPerturbationSpatial.size() / 6; i++)
  {
    expectedPerturbationFromSpatial.segment(i * 3, 3)
        = math::expMap(expectedPerturbationSpatial.segment(i * 6, 6).eval())
              .translation();
  }

  Eigen::MatrixXs perturbations
      = Eigen::MatrixXs::Zero(worldPerturbation.size(), 3);
  perturbations << worldPerturbation, expectedPerturbation,
      expectedPerturbationFromSpatial;

  Eigen::MatrixXs perturbationsSpatial
      = Eigen::MatrixXs::Zero(worldPerturbationSpatial.size(), 2);
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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXs originalWorldPos = convertJointSpaceToWorldSpace(
      world, position, bodyNodes, ConvertToSpace::COM_POS, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(velocity.size()) * 1e-4;
  Eigen::VectorXs perturbedPos = position + perturbation;

  Eigen::MatrixXs skelLinearJac
      = world->getSkeleton(0)->getCOMPositionJacobian().bottomRows<3>();
  Eigen::VectorXs expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXs perturbedWorldPos = convertJointSpaceToWorldSpace(
      world, perturbedPos, bodyNodes, ConvertToSpace::COM_POS, false);

  Eigen::VectorXs worldPerturbation = perturbedWorldPos - originalWorldPos;

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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXs originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(velocity.size()) * 1e-4;
  Eigen::VectorXs perturbedVel = velocity + perturbation;

  Eigen::MatrixXs skelLinearJac = world->getSkeleton(0)->getCOMLinearJacobian();
  Eigen::VectorXs expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXs perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::COM_VEL_LINEAR, false);

  Eigen::VectorXs worldPerturbation = perturbedWorldVel - originalWorldVel;

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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  // Verify that the velocity conversion works properly
  world->setPositions(position);
  world->setVelocities(velocity);

  std::vector<dynamics::BodyNode*> bodyNodes
      = world->getSkeleton(0)->getBodyNodes();
  Eigen::VectorXs originalWorldVel = convertJointSpaceToWorldSpace(
      world, velocity, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);

  Eigen::VectorXs perturbation
      = Eigen::VectorXs::Random(velocity.size()) * 1e-4;
  Eigen::VectorXs perturbedVel = velocity + perturbation;

  Eigen::MatrixXs skelLinearJac = world->getSkeleton(0)->getCOMJacobian();
  Eigen::VectorXs expectedPerturbation = skelLinearJac * perturbation;

  Eigen::VectorXs perturbedWorldVel = convertJointSpaceToWorldSpace(
      world, perturbedVel, bodyNodes, ConvertToSpace::COM_VEL_SPATIAL, false);

  Eigen::VectorXs worldPerturbation = perturbedWorldVel - originalWorldVel;

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
    WorldPtr world, Eigen::VectorXs position, Eigen::VectorXs velocity)
{
  if (!verifyLinearJacobian(world, position, velocity))
    return false;
  if (!verifyClosestIKPosition(world, position))
    return false;
  if (!verifyClosestIKVelocity(world, velocity))
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
  Eigen::MatrixXs jointPoses
      = Eigen::MatrixXs::Random(world->getNumDofs(), timesteps);
  Eigen::MatrixXs jointVels
      = Eigen::MatrixXs::Random(world->getNumDofs(), timesteps);

  for (int i = 0; i < timesteps; i++)
  {
    if (!verifyWorldSpaceTransformInstance(
            world, jointPoses.col(i), jointVels.col(i)))
      return false;
  }

  // Verify that nothing crashes when we run a batch
  std::vector<dynamics::BodyNode*> bodyNodes = world->getAllBodyNodes();
  Eigen::MatrixXs worldPos = convertJointSpaceToWorldSpace(
      world, jointPoses, bodyNodes, ConvertToSpace::POS_LINEAR, false);
  Eigen::MatrixXs worldVel = convertJointSpaceToWorldSpace(
      world, jointVels, bodyNodes, ConvertToSpace::VEL_LINEAR, false);

  Eigen::MatrixXs backprop = convertJointSpaceToWorldSpace(
      world, worldPos * 5, bodyNodes, ConvertToSpace::POS_LINEAR, false, true);

  return true;
}

bool verifyAnalyticalA_c(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs truePreStep = world->getPositions();
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  Eigen::MatrixXs A_c = classicPtr->getMassMatrix(world)
                        * classicPtr->getMassedClampingConstraintMatrix(world);

  Eigen::VectorXs preStepPos = classicPtr->getPreStepPosition();
  Eigen::VectorXs postStepPos = classicPtr->getPostStepPosition();
  world->setPositions(classicPtr->getPreStepPosition());
  for (int i = 0; i < classicPtr->getNumClamping(); i++)
  {
    Eigen::VectorXs trueCol = A_c.col(i);
    Eigen::VectorXs analyticalCol
        = constraints[i]->getConstraintForces(world.get());
    if (!equals(trueCol, analyticalCol, 5e-10))
    {
      std::cout << "True A_c col: " << std::endl << trueCol << std::endl;
      std::cout << "Analytical A_c col: " << std::endl
                << analyticalCol << std::endl;
      snapshot.restore();
      return false;
    }
  }

  snapshot.restore();

  return true;
}

bool verifyAnalyticalA_ub(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs truePreStep = world->getPositions();
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getUpperBoundConstraints();
  Eigen::MatrixXs A_ub
      = classicPtr->getMassMatrix(world)
        * classicPtr->getMassedUpperBoundConstraintMatrix(world);

  Eigen::VectorXs preStepPos = classicPtr->getPreStepPosition();
  Eigen::VectorXs postStepPos = classicPtr->getPostStepPosition();
  world->setPositions(classicPtr->getPreStepPosition());
  for (int i = 0; i < classicPtr->getNumUpperBound(); i++)
  {
    Eigen::VectorXs trueCol = A_ub.col(i);
    Eigen::VectorXs analyticalCol
        = constraints[i]->getConstraintForces(world.get());
    if (!equals(trueCol, analyticalCol, 5e-9))
    {
      std::cout << "True A_ub col: " << std::endl << trueCol << std::endl;
      std::cout << "Analytical A_ub col: " << std::endl
                << analyticalCol << std::endl;
      snapshot.restore();
      return false;
    }
  }

  snapshot.restore();

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
      std::cout << "Pos Jac failed on constraint " << i << "/"
                << constraints.size() << std::endl;
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
      std::cout << "Contact Force Jac failed on constraint " << i << "/"
                << constraints.size() << std::endl;
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

bool equals(EdgeData e1, EdgeData e2, s_t threshold)
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
  const s_t EPS = 1e-5;
  for (int k = 0; k < constraints.size(); k++)
  {
    if (constraints[k]->getContactType() == EDGE_EDGE)
    {
      EdgeData original = constraints[k]->getEdges();
      /*
      Eigen::Vector3s originalIntersectionPoint = math::getContactPoint(
          original.edgeAPos,
          original.edgeADir,
          original.edgeBPos,
          original.edgeBDir);
          */

      for (int i = 0; i < world->getNumSkeletons(); i++)
      {
        auto skel = world->getSkeleton(i);
        for (int j = 0; j < skel->getNumDofs(); j++)
        {
          EdgeData bruteForce
              = constraints[k]->bruteForceEdges(world, skel, j, EPS);
          EdgeData bruteForceNeg
              = constraints[k]->bruteForceEdges(world, skel, j, -EPS);
          EdgeData analytical
              = constraints[k]->estimatePerturbedEdges(skel, j, EPS);
          EdgeData analyticalNeg
              = constraints[k]->estimatePerturbedEdges(skel, j, -EPS);

          Eigen::Vector3s bruteIntersection = math::getContactPoint(
              bruteForce.edgeAPos,
              bruteForce.edgeADir,
              bruteForce.edgeBPos,
              bruteForce.edgeBDir);
          Eigen::Vector3s bruteIntersectionNeg = math::getContactPoint(
              bruteForceNeg.edgeAPos,
              bruteForceNeg.edgeADir,
              bruteForceNeg.edgeBPos,
              bruteForceNeg.edgeBDir);
          Eigen::Vector3s analyticalIntersection = math::getContactPoint(
              bruteForce.edgeAPos,
              bruteForce.edgeADir,
              bruteForce.edgeBPos,
              bruteForce.edgeBDir);

          s_t estimateThreshold = 1e-10;

          // Check the intersection point first, because the actual input
          // points can be different if the collision detector decided to use
          // a different vertex on either of the edges, which will screw
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
            // materially different, because it is valid to have the edges
            // pick different corners when finding a collision.
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

          EdgeData finiteDifferenceGradient;
          finiteDifferenceGradient.edgeAPos
              = (bruteForce.edgeAPos - bruteForceNeg.edgeAPos) / (2 * EPS);
          finiteDifferenceGradient.edgeADir
              = (bruteForce.edgeADir - bruteForceNeg.edgeADir) / (2 * EPS);
          finiteDifferenceGradient.edgeBPos
              = (bruteForce.edgeBPos - bruteForceNeg.edgeBPos) / (2 * EPS);
          finiteDifferenceGradient.edgeBDir
              = (bruteForce.edgeBDir - bruteForceNeg.edgeBDir) / (2 * EPS);

          EdgeData finiteDifferenceAnalyticalGradient;
          finiteDifferenceAnalyticalGradient.edgeAPos
              = (analytical.edgeAPos - analyticalNeg.edgeAPos) / (2 * EPS);
          finiteDifferenceAnalyticalGradient.edgeADir
              = (analytical.edgeADir - analyticalNeg.edgeADir) / (2 * EPS);
          finiteDifferenceAnalyticalGradient.edgeBPos
              = (analytical.edgeBPos - analyticalNeg.edgeBPos) / (2 * EPS);
          finiteDifferenceAnalyticalGradient.edgeBDir
              = (analytical.edgeBDir - analyticalNeg.edgeBDir) / (2 * EPS);

          EdgeData analyticalGradient
              = constraints[k]->getEdgeGradient(skel->getDof(j));

          if (!equals(
                  finiteDifferenceGradient.edgeAPos,
                  analyticalGradient.edgeAPos,
                  estimateThreshold))
          {
            std::cout << "Edge A Pos gradient analytical: " << std::endl
                      << analyticalGradient.edgeAPos << std::endl;
            std::cout << "Edge A Pos gradient brute force: " << std::endl
                      << finiteDifferenceGradient.edgeAPos << std::endl;
            std::cout << "Edge A Pos gradient brute force over estimates: "
                      << std::endl
                      << finiteDifferenceAnalyticalGradient.edgeAPos
                      << std::endl;
            return false;
          }
          if (!equals(
                  finiteDifferenceGradient.edgeADir,
                  analyticalGradient.edgeADir,
                  estimateThreshold))
          {
            std::cout << "Edge A Dir gradient analytical: " << std::endl
                      << analyticalGradient.edgeADir << std::endl;
            std::cout << "Edge A Dir gradient brute force: " << std::endl
                      << bruteForce.edgeADir << std::endl;
            return false;
          }
          if (!equals(
                  finiteDifferenceGradient.edgeBPos,
                  analyticalGradient.edgeBPos,
                  estimateThreshold))
          {
            std::cout << "Edge B Pos gradient analytical: " << std::endl
                      << analyticalGradient.edgeBPos << std::endl;
            std::cout << "Edge B Pos gradient brute force: " << std::endl
                      << finiteDifferenceGradient.edgeBPos << std::endl;
            constraints[k]->getEdgeGradient(skel->getDof(j));
            return false;
          }
          if (!equals(
                  finiteDifferenceGradient.edgeBDir,
                  analyticalGradient.edgeBDir,
                  estimateThreshold))
          {
            std::cout << "Edge B Dir gradient analytical: " << std::endl
                      << analyticalGradient.edgeBDir << std::endl;
            std::cout << "Edge B Dir gradient brute force: " << std::endl
                      << finiteDifferenceGradient.edgeBDir << std::endl;
            return false;
          }

          Eigen::Vector3s analyticalIntersectionGradient
              = math::getContactPointGradient(
                  original.edgeAPos,
                  analyticalGradient.edgeAPos,
                  original.edgeADir,
                  analyticalGradient.edgeADir,
                  original.edgeBPos,
                  analyticalGradient.edgeBPos,
                  original.edgeBDir,
                  analyticalGradient.edgeBDir);

          Eigen::Vector3s finiteDifferenceIntersectionGradient
              = (bruteIntersection - bruteIntersectionNeg) / (2 * EPS);

          estimateThreshold = 1e-8;

          // Check the intersection point first, because the actual input
          // points can be different if the collision detector decided to use
          // a different vertex on either of the edges, which will screw
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
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

bool verifyTranlationalLCPInvariance(
    WorldPtr world, int dofIndex, s_t perturbBy)
{
  RestorableSnapshot snapshot(world);

  BackpropSnapshotPtr originalPtr = neural::forwardPass(world, true);
  dynamics::DegreeOfFreedom* dof = world->getDofs()[dofIndex];
  dof->setPosition(dof->getPosition() + perturbBy);
  BackpropSnapshotPtr perturbedPtr = neural::forwardPass(world, true);

  Eigen::MatrixXs perturbedA = perturbedPtr->mGradientMatrices[0]->mA;
  Eigen::MatrixXs originalA = originalPtr->mGradientMatrices[0]->mA;
  std::cout << "Original A:" << std::endl << originalA << std::endl;
  std::cout << "Perturbed A:" << std::endl << perturbedA << std::endl;
  std::cout << "Diff:" << std::endl << (originalA - perturbedA) << std::endl;

  Eigen::MatrixXs perturbedA_c
      = perturbedPtr->mGradientMatrices[0]->getFullConstraintMatrix(
          world.get());
  Eigen::MatrixXs originalA_c
      = originalPtr->mGradientMatrices[0]->getFullConstraintMatrix(world.get());
  std::cout << "Original A_c:" << std::endl << originalA_c << std::endl;
  std::cout << "Perturbed A_c:" << std::endl << perturbedA_c << std::endl;
  std::cout << "Diff:" << std::endl
            << (originalA_c - perturbedA_c) << std::endl;

  std::shared_ptr<DifferentiableContactConstraint> perturbedConstraint
      = perturbedPtr->mGradientMatrices[0]->getDifferentiableConstraints()[0];
  std::shared_ptr<DifferentiableContactConstraint> originalConstraint
      = originalPtr->mGradientMatrices[0]->getDifferentiableConstraints()[0];

  std::cout << "Contact world force gradient: " << std::endl
            << originalConstraint->getContactWorldForceGradient(dof)
            << std::endl;

  Eigen::VectorXs perturbedB = perturbedPtr->mGradientMatrices[0]->mB;
  Eigen::VectorXs originalB = originalPtr->mGradientMatrices[0]->mB;
  std::cout << "Original B:" << std::endl << originalB << std::endl;
  std::cout << "Perturbed B:" << std::endl << perturbedB << std::endl;
  std::cout << "Diff:" << std::endl << (originalB - perturbedB) << std::endl;

  Eigen::VectorXs perturbedX = perturbedPtr->mGradientMatrices[0]->mX;
  Eigen::VectorXs originalX = originalPtr->mGradientMatrices[0]->mX;
  std::cout << "Original X:" << std::endl << originalX << std::endl;
  std::cout << "Perturbed X:" << std::endl << perturbedX << std::endl;
  std::cout << "Diff:" << std::endl << (originalX - perturbedX) << std::endl;

  snapshot.restore();
  return true;
}

bool verifyPerturbedContactPositions(
    WorldPtr world, bool allowNoContacts = false)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  if ((constraints.size() == 0) && !allowNoContacts)
  {
    std::cout
        << "verifyPerturbedContactPositions() got no clamping contacts, and "
           "didn't pass `allowNoContacts = true`!"
        << std::endl;
    return false;
  }

  // const s_t EPS = 1e-4;
  const s_t EPS = 1e-6;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3s pos = constraints[k]->getContactWorldPosition();
        Eigen::Vector3s normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3s analytical
            = constraints[k]->estimatePerturbedContactPosition(skel, j, EPS);
        Eigen::Vector3s bruteForce
            = constraints[k]->bruteForcePerturbedContactPosition(
                world, skel, j, EPS);
        // our EPS is only 1e-7, so tolerances have to be tight on our tests
        if (false && !equals(analytical, bruteForce, 1e-9))
        {
          std::cout << "Failed perturbed contact pos!" << std::endl;
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

          Eigen::Vector6s worldTwist
              = constraints[k]->getWorldScrewAxisForPosition(skel, j);
          Eigen::Isometry3s rotation = math::expMap(worldTwist * EPS);
          Eigen::Vector3s perturbedContactPos = rotation * pos;
          std::cout << "World Screw:" << std::endl << worldTwist << std::endl;
          std::cout << "Transform:" << std::endl
                    << rotation.matrix() << std::endl;
          std::cout << "Transform * Original Pos:" << std::endl
                    << perturbedContactPos << std::endl;

          auto dof = skel->getDof(j);
          dof->setPosition(dof->getPosition() + EPS);

          std::shared_ptr<BackpropSnapshot> perturbedSnapshot
              = neural::forwardPass(world, true);
          std::vector<std::shared_ptr<DifferentiableContactConstraint>>
              perturbedConstraints
              = perturbedSnapshot->getClampingConstraints();

          std::cout << "Original contacts: " << std::endl;
          for (auto& point : constraints)
          {
            std::cout << "::" << std::endl
                      << point->getContactWorldPosition() << std::endl;
          }
          std::cout << "Perturbed contacts: " << std::endl;
          for (auto& point : perturbedConstraints)
          {
            std::cout << "::" << std::endl
                      << point->getContactWorldPosition() << std::endl;
          }

          /*
          // Uncomment for a breakpoint
          Eigen::Vector3s analytical
              = constraints[k]->estimatePerturbedContactPosition(skel, j, EPS);
          */
          return false;
        }

        Eigen::Vector3s bruteForceNeg
            = constraints[k]->bruteForcePerturbedContactPosition(
                world, skel, j, -EPS);
        Eigen::Vector3s analyticalNeg
            = constraints[k]->estimatePerturbedContactPosition(skel, j, -EPS);

        Eigen::Vector3s finiteDifferenceGradient
            = (bruteForce - bruteForceNeg) / (2 * EPS);
        Eigen::Vector3s finiteDifferenceAnalyticalGradient
            = (analytical - analyticalNeg) / (2 * EPS);
        Eigen::Vector3s analyticalGradient
            = constraints[k]->getContactPositionGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
        {
          std::cout << "Failed contact pos gradient!" << std::endl;
          std::cout << "Contact type: " << constraints[k]->getContactType()
                    << std::endl;
          constraints[k]->getContactPositionGradient(skel->getDof(j));
          constraints[k]->bruteForcePerturbedContactPosition(
              world, skel, j, -EPS);
          constraints[k]->bruteForcePerturbedContactPosition(
              world, skel, j, EPS);
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
          std::cout << "Finite Difference Analytical Contact Pos Gradient:"
                    << std::endl
                    << finiteDifferenceAnalyticalGradient << std::endl;
          constraints[k]->bruteForcePerturbedContactPosition(
              world, skel, j, -EPS);
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
  const s_t EPS = 1e-6;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3s pos = constraints[k]->getContactWorldPosition();
        Eigen::Vector3s normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3s analytical
            = constraints[k]->estimatePerturbedContactNormal(skel, j, EPS);
        Eigen::Vector3s bruteForce
            = constraints[k]->bruteForcePerturbedContactNormal(
                world, skel, j, EPS);
        if (!equals(analytical, bruteForce, 1e-9))
        {
          analytical
              = constraints[k]->estimatePerturbedContactNormal(skel, j, EPS);
          bruteForce = constraints[k]->bruteForcePerturbedContactNormal(
              world, skel, j, EPS);
          std::cout << "Failed perturbed contact normal by " << EPS << "!"
                    << std::endl;
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "DOF:" << std::endl << j << std::endl;
          std::cout << "Contact:" << std::endl << k << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Original Contact Normal:" << std::endl
                    << normal << std::endl;
          std::cout << "Original Contact Pos:" << std::endl << pos << std::endl;
          std::cout << "Analytical Contact Normal:" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical Contact Normal Diff:" << std::endl
                    << (analytical - normal) << std::endl;
          std::cout << "Brute Force Contact Normal Diff:" << std::endl
                    << (bruteForce - normal) << std::endl;
          return false;
        }

        Eigen::Vector3s bruteForceNeg
            = constraints[k]->bruteForcePerturbedContactNormal(
                world, skel, j, -EPS);

        Eigen::Vector3s finiteDifferenceGradient
            = (bruteForce - bruteForceNeg) / (2 * EPS);
        Eigen::Vector3s analyticalGradient
            = constraints[k]->getContactNormalGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-9))
        {
          Eigen::Vector3s analyticalGradient
              = constraints[k]->getContactNormalGradient(skel->getDof(j));
          std::cout << "Failed contact normal gradient!" << std::endl;
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Contact Type:" << std::endl
                    << constraints[k]->getDofContactType(skel->getDof(j))
                    << std::endl;
          std::cout << "Contact Normal:" << std::endl << normal << std::endl;
          std::cout << "Contact Pos:" << std::endl << pos << std::endl;
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
  const s_t EPS = 1e-6;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      for (int k = 0; k < constraints.size(); k++)
      {
        Eigen::Vector3s normal = constraints[k]->getContactWorldNormal();
        Eigen::Vector3s dir = constraints[k]->getContactWorldForceDirection();
        Eigen::Vector3s analytical
            = constraints[k]->estimatePerturbedContactForceDirection(
                skel, j, EPS);
        Eigen::Vector3s bruteForce
            = constraints[k]->bruteForcePerturbedContactForceDirection(
                world, skel, j, EPS);
        if (!equals(analytical, bruteForce, 1e-8))
        {
          std::cout << "Failed analytical force dir estimate!" << std::endl;
          std::cout << "Constraint index:" << std::endl << k << std::endl;
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Diff wrt index:" << std::endl << j << std::endl;

          auto dof = skel->getDof(j);
          int jointIndex = dof->getIndexInJoint();
          math::Jacobian relativeJac = dof->getJoint()->getRelativeJacobian();
          dynamics::BodyNode* childNode = dof->getChildBodyNode();
          Eigen::Isometry3s transform = childNode->getWorldTransform();
          Eigen::Vector6s localTwist = relativeJac.col(jointIndex);
          Eigen::Vector6s worldTwist = math::AdT(transform, localTwist);

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

        Eigen::Vector3s bruteForceNeg
            = constraints[k]->bruteForcePerturbedContactForceDirection(
                world, skel, j, -EPS);
        Eigen::Vector3s finiteDifferenceGradient
            = (bruteForce - bruteForceNeg) / (2 * EPS);
        Eigen::Vector3s analyticalGradient
            = constraints[k]->getContactForceGradient(skel->getDof(j));
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-9))
        {
          analyticalGradient
              = constraints[k]->getContactForceGradient(skel->getDof(j));
          bruteForceNeg
              = constraints[k]->bruteForcePerturbedContactForceDirection(
                  world, skel, j, -EPS);
          std::cout << "Failed analytical force gradient!" << std::endl;
          std::cout << "Constraint index:" << std::endl << k << std::endl;
          std::cout << "Skel:" << std::endl << skel->getName() << std::endl;
          std::cout << "Diff wrt index:" << std::endl << j << std::endl;
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

bool verifyPerturbedScrewAxisForPosition(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const s_t EPS = 5e-9;
  std::vector<DegreeOfFreedom*> dofs = world->getDofs();
  for (int j = 0; j < dofs.size(); j++)
  {
    DegreeOfFreedom* axis = dofs[j];
    for (int k = 0; k < dofs.size(); k++)
    {
      DegreeOfFreedom* wrt = dofs[k];
      for (int q = 0; q < constraints.size(); q++)
      {
        Eigen::Vector6s original
            = constraints[q]->getWorldScrewAxisForPosition(axis);
        Eigen::Vector6s analytical
            = constraints[q]->estimatePerturbedScrewAxisForPosition(
                axis, wrt, EPS);
        Eigen::Vector6s bruteForce
            = constraints[q]->bruteForceScrewAxisForPosition(axis, wrt, EPS);

        if (!equals(analytical, bruteForce, 1e-9))
        {
          std::cout << "Axis: " << axis->getSkeleton()->getName() << " - "
                    << axis->getIndexInSkeleton() << std::endl;
          std::cout << "Rotate: " << wrt->getSkeleton()->getName() << " - "
                    << wrt->getIndexInSkeleton() << std::endl;
          std::cout << "Axis Contact Type: "
                    << constraints[q]->getDofContactType(axis) << std::endl;
          std::cout << "Rotate Contact Type: "
                    << constraints[q]->getDofContactType(wrt) << std::endl;
          std::cout << "Is parent: " << wrt->isParentOf(axis) << std::endl;
          std::cout << "Analytical World Screw (for pos):" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical World Screw (for pos) Diff:" << std::endl
                    << (analytical - original) << std::endl;
          std::cout << "Brute Force World Screw (for pos) Diff:" << std::endl
                    << (bruteForce - original) << std::endl;
          return false;
        }

        ///////////////////////////////////////////////////////
        //
        // We actually never use the gradient for the positional screw, so we
        // haven't bothered to make it analytical. Instead we just use finite
        // differencing. This test can sometimes fail due to numerical issues
        // that result. For that reason, they're currenty commented out. If we
        // find ourselves needing that gradient in the future, we should
        // uncomment these tests.
        //
        ///////////////////////////////////////////////////////

        /*
        Eigen::Vector6s analyticalNeg
            = constraints[q]->estimatePerturbedScrewAxisForPosition(
                axis, wrt, -EPS);
        Eigen::Vector6s bruteForceNeg
            = constraints[q]->bruteForceScrewAxisForPosition(axis, wrt, -EPS);

        Eigen::Vector6s finiteDifferenceGradient
            = (bruteForce - bruteForceNeg) / (2 * EPS);
        Eigen::Vector6s finiteDifferenceAnalyticalGradient
            = (analytical - analyticalNeg) / (2 * EPS);
        Eigen::Vector6s analyticalGradient
            = constraints[q]->getScrewAxisForPositionGradient(axis, wrt);
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
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
          std::cout << "Analytical World Screw (for pos) Gradient:" <<
        std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference World Screw (for pos) Gradient:"
                    << std::endl
                    << finiteDifferenceGradient << std::endl;
          std::cout
              << "Finite Difference Analytical World Screw (for pos)
        Gradient:"
              << std::endl
              << finiteDifferenceAnalyticalGradient << std::endl;
          return false;
        }
        */
      }
    }
  }
  return true;
}

bool verifyPerturbedScrewAxisForForce(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getClampingConstraints();
  const s_t EPS = 1e-6;
  std::vector<DegreeOfFreedom*> dofs = world->getDofs();
  for (int j = 0; j < dofs.size(); j++)
  {
    DegreeOfFreedom* axis = dofs[j];
    for (int k = 0; k < dofs.size(); k++)
    {
      DegreeOfFreedom* wrt = dofs[k];
      for (int q = 0; q < constraints.size(); q++)
      {
        Eigen::Vector6s original
            = constraints[q]->getWorldScrewAxisForForce(axis);
        Eigen::Vector6s analytical
            = constraints[q]->estimatePerturbedScrewAxisForForce(
                axis, wrt, EPS);
        Eigen::Vector6s bruteForce
            = constraints[q]->bruteForceScrewAxisForForce(axis, wrt, EPS);

        if (!equals(analytical, bruteForce, 1e-9))
        {
          std::cout << "Axis: " << axis->getSkeleton()->getName() << " - "
                    << axis->getIndexInSkeleton() << std::endl;
          std::cout << "Rotate: " << wrt->getSkeleton()->getName() << " - "
                    << wrt->getIndexInSkeleton() << std::endl;
          std::cout << "Axis Contact Type: "
                    << constraints[q]->getDofContactType(axis) << std::endl;
          std::cout << "Rotate Contact Type: "
                    << constraints[q]->getDofContactType(wrt) << std::endl;
          std::cout << "Is parent: " << wrt->isParentOf(axis) << std::endl;
          std::cout << "Analytical World Screw (for force):" << std::endl
                    << analytical << std::endl;
          std::cout << "Analytical World Screw (for force) Diff:" << std::endl
                    << (analytical - original) << std::endl;
          std::cout << "Brute Force World Screw (for force) Diff:" << std::endl
                    << (bruteForce - original) << std::endl;

          analytical = constraints[q]->estimatePerturbedScrewAxisForForce(
              axis, wrt, EPS);
          bruteForce
              = constraints[q]->bruteForceScrewAxisForForce(axis, wrt, EPS);
          return false;
        }

        Eigen::Vector6s analyticalNeg
            = constraints[q]->estimatePerturbedScrewAxisForForce(
                axis, wrt, -EPS);
        Eigen::Vector6s bruteForceNeg
            = constraints[q]->bruteForceScrewAxisForForce(axis, wrt, -EPS);

        Eigen::Vector6s finiteDifferenceGradient
            = (bruteForce - bruteForceNeg) / (2 * EPS);
        Eigen::Vector6s finiteDifferenceAnalyticalGradient
            = (analytical - analyticalNeg) / (2 * EPS);
        Eigen::Vector6s analyticalGradient
            = constraints[q]->getScrewAxisForForceGradient(axis, wrt);
        if (!equals(analyticalGradient, finiteDifferenceGradient, 1e-8))
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
          std::cout << "Analytical World Screw (for force) Gradient:"
                    << std::endl
                    << analyticalGradient << std::endl;
          std::cout << "Finite Difference World Screw (for force) Gradient:"
                    << std::endl
                    << finiteDifferenceGradient << std::endl;
          std::cout << "Diff:" << std::endl
                    << analyticalGradient - finiteDifferenceGradient
                    << std::endl;
          std::cout << "Finite Difference Analytical World Screw (for force) "
                       "Gradient:"
                    << std::endl
                    << finiteDifferenceAnalyticalGradient << std::endl;
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

  const s_t EPS = 1e-7;

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
        s_t originalValue = constraints[i]->getConstraintForce(axis);

        s_t analytical
            = constraints[i]->getConstraintForceDerivative(axis, rotate);

        s_t originalPosition = rotate->getPosition();
        rotate->setPosition(originalPosition + EPS);
        BackpropSnapshotPtr newPtr = neural::forwardPass(world, true);
        s_t newValue
            = constraints[i]->getPeerConstraint(newPtr)->getConstraintForce(
                axis);

        rotate->setPosition(originalPosition - EPS);
        BackpropSnapshotPtr newPtrNeg = neural::forwardPass(world, true);
        s_t newValueNeg
            = constraints[i]->getPeerConstraint(newPtrNeg)->getConstraintForce(
                axis);

        rotate->setPosition(originalPosition);

        Eigen::Vector6s gradientOfWorldForce
            = constraints[i]->getContactWorldForceGradient(rotate);
        Eigen::Vector6s worldTwist
            = constraints[i]->getWorldScrewAxisForForce(axis);

        s_t bruteForce = (newValue - newValueNeg) / (2 * EPS);

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
          std::cout << "Diff:" << std::endl
                    << analytical - bruteForce << std::endl;
          std::cout << "Gradient of world force:" << std::endl
                    << gradientOfWorldForce << std::endl;
          std::cout << "World twist (for force):" << std::endl
                    << worldTwist << std::endl;
          /*
          // Uncomment for a breakpoint
          s_t analytical
              = constraints[i]->getConstraintForceDerivative(axis, rotate);
          */
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
    Eigen::MatrixXs analytical
        = constraints[i]->getConstraintForcesJacobian(world);
    Eigen::MatrixXs bruteForce
        = constraints[i]->bruteForceConstraintForcesJacobian(world);
    Eigen::VectorXs A_cCol = constraints[i]->getConstraintForces(world.get());
    if (!equals(analytical, bruteForce, 1e-8))
    {
      std::cout << "A_c col:" << std::endl << A_cCol << std::endl;
      std::cout << "Analytical constraint forces Jac:" << std::endl
                << analytical << std::endl;
      std::cout << "Brute force constraint forces Jac:" << std::endl
                << bruteForce << std::endl;
      std::cout << "Constraint forces Jac diff:" << std::endl
                << (analytical - bruteForce) << std::endl;
      return false;
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
        Eigen::MatrixXs gold
            = analytical.block(row, col, skel->getNumDofs(), wrt->getNumDofs());
        Eigen::MatrixXs chunk
            = constraints[i]->getConstraintForcesJacobian(skel, wrt);
        if (!equals(gold, chunk, 1e-9))
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

      Eigen::MatrixXs gold
          = analytical.block(0, col, world->getNumDofs(), wrt->getNumDofs());
      Eigen::MatrixXs chunk
          = constraints[i]->getConstraintForcesJacobian(skels, wrt);
      if (!equals(gold, chunk, 1e-9))
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

    Eigen::MatrixXs skelAnalytical
        = constraints[i]->getConstraintForcesJacobian(world, skels);
    if (!equals(analytical, skelAnalytical, 1e-9))
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

bool verifyAnalyticalA_ubJacobian(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = classicPtr->getUpperBoundConstraints();
  for (int i = 0; i < constraints.size(); i++)
  {
    Eigen::MatrixXs analytical
        = constraints[i]->getConstraintForcesJacobian(world);
    Eigen::MatrixXs bruteForce
        = constraints[i]->bruteForceConstraintForcesJacobian(world);
    Eigen::VectorXs A_ubCol = constraints[i]->getConstraintForces(world.get());
    if (!equals(analytical, bruteForce, 1e-8))
    {
      std::cout << "A_ub col:" << std::endl << A_ubCol << std::endl;
      std::cout << "Analytical constraint forces Jac:" << std::endl
                << analytical << std::endl;
      std::cout << "Brute force constraint forces Jac:" << std::endl
                << bruteForce << std::endl;
      std::cout << "Constraint forces Jac diff:" << std::endl
                << (analytical - bruteForce) << std::endl;
      return false;
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
        Eigen::MatrixXs gold
            = analytical.block(row, col, skel->getNumDofs(), wrt->getNumDofs());
        Eigen::MatrixXs chunk
            = constraints[i]->getConstraintForcesJacobian(skel, wrt);
        if (!equals(gold, chunk, 1e-9))
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

      Eigen::MatrixXs gold
          = analytical.block(0, col, world->getNumDofs(), wrt->getNumDofs());
      Eigen::MatrixXs chunk
          = constraints[i]->getConstraintForcesJacobian(skels, wrt);
      if (!equals(gold, chunk, 1e-9))
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

    Eigen::MatrixXs skelAnalytical
        = constraints[i]->getConstraintForcesJacobian(world, skels);
    if (!equals(analytical, skelAnalytical, 1e-9))
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

  Eigen::VectorXs f0 = Eigen::VectorXs::Random(classicPtr->getNumClamping());

  Eigen::MatrixXs analytical
      = classicPtr->getJacobianOfClampingConstraints(world, f0);
  Eigen::MatrixXs bruteForce
      = classicPtr->finiteDifferenceJacobianOfClampingConstraints(world, f0);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "getJacobianOfClampingConstraints error:" << std::endl;
    std::cout << "f0:" << std::endl << f0 << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;

    for (int i = 0; i < f0.size(); i++)
    {
      Eigen::VectorXs oneHot = Eigen::VectorXs::Zero(f0.size());
      oneHot(i) = 1.0;

      Eigen::MatrixXs analyticalOneHot
          = classicPtr->getJacobianOfClampingConstraints(world, oneHot);
      Eigen::MatrixXs bruteForceOneHot
          = classicPtr->finiteDifferenceJacobianOfClampingConstraints(
              world, oneHot);
      if (!equals(analyticalOneHot, bruteForceOneHot, 1e-8))
      {
        std::cout << "getJacobianOfClampingConstraints error [" << i
                  << "]:" << std::endl;
        std::cout << "Analytical one hot:" << std::endl
                  << analyticalOneHot << std::endl;
        std::cout << "Brute Force one hot:" << std::endl
                  << bruteForceOneHot << std::endl;
        std::cout << "Diff:" << std::endl
                  << analyticalOneHot - bruteForceOneHot << std::endl;
      }
    }

    return false;
  }
  return true;
}

bool verifyJacobianOfClampingConstraintsTranspose(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::VectorXs v0 = Eigen::VectorXs::Random(world->getNumDofs());

  Eigen::MatrixXs analytical
      = classicPtr->getJacobianOfClampingConstraintsTranspose(world, v0);
  Eigen::MatrixXs bruteForce
      = classicPtr->finiteDifferenceJacobianOfClampingConstraintsTranspose(
          world, v0);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "getJacobianOfClampingConstraintsTranspose error:"
              << std::endl;
    std::cout << "v0:" << std::endl << v0 << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << analytical - bruteForce << std::endl;
    return false;
  }
  return true;
}

bool verifyJacobianOfUpperBoundConstraints(WorldPtr world)
{
  BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  Eigen::VectorXs f0 = Eigen::VectorXs::Random(classicPtr->getNumUpperBound());

  Eigen::MatrixXs analytical
      = classicPtr->getJacobianOfUpperBoundConstraints(world, f0);
  Eigen::MatrixXs bruteForce
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

bool verifyPositionScrews(WorldPtr world)
{
  s_t EPS = 1e-4;

  std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
  for (int dofIndex = 0; dofIndex < dofs.size(); dofIndex++)
  {
    dynamics::DegreeOfFreedom* dof = dofs[dofIndex];

    // dynamics::BodyNode* node = dof->getChildBodyNode();
    // Eigen::Isometry3s originalTransform = node->getWorldTransform();

    // get world twist
    int jointIndex = dof->getIndexInJoint();
    dynamics::BodyNode* childNode = dof->getChildBodyNode();
    Eigen::Isometry3s transform = childNode->getWorldTransform();
    Eigen::Vector6s worldTwist
        = dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);

    s_t pos = dof->getPosition();
    dof->setPosition(pos + EPS);

    Eigen::Matrix4s analyticalPerturbRotation
        = (math::expMap(worldTwist * EPS) * transform).matrix();
    Eigen::Matrix4s realPerturbRotation
        = childNode->getWorldTransform().matrix();

    // Reset
    dof->setPosition(pos);

    if (!equals(analyticalPerturbRotation, realPerturbRotation, 5e-9))
    {
      std::cout << "Failed screw test!" << std::endl;
      std::cout << "Axis: " << dofIndex << std::endl;
      std::cout << "Analytical perturbations" << std::endl
                << analyticalPerturbRotation << std::endl;
      std::cout << "Real perturbations" << std::endl
                << realPerturbRotation << std::endl;
      Eigen::Matrix4s diff = analyticalPerturbRotation - realPerturbRotation;
      std::cout << "Diff" << std::endl << diff << std::endl;
      std::cout << "Screw: " << worldTwist << std::endl;
      std::cout << "Original transform" << std::endl
                << transform.matrix() << std::endl;
      std::cout << "Left multiply" << std::endl
                << math::expMap(worldTwist * EPS).matrix() << std::endl;
      std::cout << "Relative transform" << std::endl
                << dof->getJoint()->getRelativeTransform().matrix()
                << std::endl;
      worldTwist = dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);

      return false;
    }
  }

  return true;
}

bool verifyIKPositionJacobians(WorldPtr world)
{
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      BodyNode* body = skel->getBodyNode(j);
      math::Jacobian analytical = skel->getWorldPositionJacobian(body);
      math::Jacobian bruteForce
          = skel->finiteDifferenceWorldPositionJacobian(body);

      if (!equals(bruteForce, analytical, 1e-8))
      {
        std::cout << "World jac for skeleton \"" << skel->getName()
                  << "\" body " << j << " " << std::endl
                  << "Brute force jac col: " << std::endl
                  << bruteForce << std::endl
                  << "Analytical jac col: " << std::endl
                  << analytical << std::endl
                  << "Diff: " << std::endl
                  << (bruteForce - analytical) << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verifyVelocityScrews(WorldPtr world)
{
  s_t EPS = 1e-4;

  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);

    std::vector<dynamics::DegreeOfFreedom*> dofs = skel->getDofs();
    for (int dofIndex = 0; dofIndex < dofs.size(); dofIndex++)
    {
      dynamics::DegreeOfFreedom* dof = dofs[dofIndex];

      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        dynamics::BodyNode* node = skel->getBodyNode(j);

        s_t originalVel = dof->getVelocity();
        // Eigen::Vector6s originalSpatialVel = node->getSpatialVelocity();

        dof->setVelocity(originalVel + EPS);
        Eigen::Vector6s plusVel = node->getSpatialVelocity();

        dof->setVelocity(originalVel - EPS);
        Eigen::Vector6s minusVel = node->getSpatialVelocity();

        dof->setVelocity(originalVel);

        // Doing this s_t-sided should be unnecessary, because it should be
        // exactly linear (in theory).
        Eigen::Vector6s bruteForceDiff = (plusVel - minusVel) / (2 * EPS);

        Eigen::Vector6s bruteForceWorldJac
            = math::AdT(node->getWorldTransform(), bruteForceDiff);

        int jointIndex = dof->getIndexInJoint();
        Eigen::Vector6s analyticalWorldJac
            = dof->getJoint()->getWorldAxisScrewForVelocity(jointIndex);
        if (!dof->isParentOf(node))
          analyticalWorldJac.setZero();

        if (!equals(bruteForceWorldJac, analyticalWorldJac, 1e-10))
        {
          std::cout << "World jac col " << dofIndex << " against body node "
                    << j << "error: " << std::endl
                    << "Brute force jac col: " << std::endl
                    << bruteForceWorldJac << std::endl
                    << "Analytical jac col: " << std::endl
                    << analyticalWorldJac << std::endl
                    << "Diff: " << std::endl
                    << (bruteForceWorldJac - analyticalWorldJac) << std::endl;
        }
      }
    }
  }

  return true;
}

bool verifyAnalyticalJacobians(WorldPtr world, bool allowNoContacts = false)
{
  return verifyPositionScrews(world) && verifyVelocityScrews(world)
         && verifyPerturbedContactEdges(world)
         && verifyPerturbedContactPositions(world, allowNoContacts)
         && verifyPerturbedContactNormals(world)
         && verifyPerturbedContactForceDirections(world)
         && verifyPerturbedScrewAxisForPosition(world)
         && verifyPerturbedScrewAxisForForce(world)
         && verifyAnalyticalContactPositionJacobians(world)
         && verifyAnalyticalContactNormalJacobians(world)
         && verifyAnalyticalContactForceJacobians(world)
         && verifyAnalyticalA_c(world) && verifyAnalyticalA_ub(world)
         && verifyAnalyticalConstraintDerivatives(world)
         && verifyAnalyticalA_cJacobian(world)
         && verifyAnalyticalA_ubJacobian(world)
         && verifyAnalyticalConstraintMatrixEstimates(world)
         && verifyJacobianOfClampingConstraints(world)
         && verifyJacobianOfClampingConstraintsTranspose(world)
         && verifyJacobianOfUpperBoundConstraints(world);
}

bool verifyNoMultistepIntereference(WorldPtr world, int steps)
{
  RestorableSnapshot snapshot(world);

  std::vector<BackpropSnapshotPtr> snapshots;
  snapshots.reserve(steps);

  WorldPtr clean = world->clone();
  assert(steps > 1);
  for (int i = 0; i < steps - 1; i++)
  {
    snapshots.push_back(neural::forwardPass(world));
  }

  clean->setPositions(world->getPositions());
  clean->setVelocities(world->getVelocities());
  clean->setControlForces(world->getControlForces());

  BackpropSnapshotPtr dirtyPtr = neural::forwardPass(world);
  BackpropSnapshotPtr cleanPtr = neural::forwardPass(clean);

  Eigen::MatrixXs dirtyVelVel = dirtyPtr->getVelVelJacobian(world);
  Eigen::MatrixXs dirtyVelPos = dirtyPtr->getVelPosJacobian(world);
  Eigen::MatrixXs dirtyPosVel = dirtyPtr->getPosVelJacobian(world);
  Eigen::MatrixXs dirtyPosPos = dirtyPtr->getPosPosJacobian(world);

  Eigen::MatrixXs cleanVelVel = cleanPtr->getVelVelJacobian(clean);
  Eigen::MatrixXs cleanVelPos = cleanPtr->getVelPosJacobian(clean);
  Eigen::MatrixXs cleanPosVel = cleanPtr->getPosVelJacobian(clean);
  Eigen::MatrixXs cleanPosPos = cleanPtr->getPosPosJacobian(clean);

  // These Jacobians should match EXACTLY. If not, then something funky is
  // happening with memory getting reused.

  if (!equals(dirtyVelVel, cleanVelVel, 0)
      || !equals(dirtyVelPos, cleanVelPos, 0)
      || !equals(dirtyPosVel, cleanPosVel, 0)
      || !equals(dirtyPosPos, cleanPosPos, 0))
  {
    std::cout << "Multistep intereference detected!" << std::endl;

    snapshot.restore();
    return false;
  }

  snapshot.restore();
  return true;
}

bool verifyVelJacobianWrt(WorldPtr world, WithRespectTo* wrt)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXs analytical = classicPtr->getVelJacobianWrt(world, wrt);
  MatrixXs bruteForce = classicPtr->finiteDifferenceVelJacobianWrt(world, wrt);

  if (!equals(analytical, bruteForce, 5e-7))
  {
    std::cout << "Brute force wrt-vel Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical wrt-vel Jacobian: " << std::endl
              << analytical << std::endl;
    std::cout << "Diff Jacobian: " << std::endl
              << (bruteForce - analytical) << std::endl;
    return false;
  }
  return true;
}

bool verifyPosJacobianWrt(WorldPtr world, WithRespectTo* wrt)
{
  neural::BackpropSnapshotPtr classicPtr = neural::forwardPass(world, true);

  MatrixXs analytical = classicPtr->getPosJacobianWrt(world, wrt);
  MatrixXs bruteForce = classicPtr->finiteDifferencePosJacobianWrt(world, wrt);

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Brute force wrt-pos Jacobian: " << std::endl
              << bruteForce << std::endl;
    std::cout << "Analytical wrt-pos Jacobian: " << std::endl
              << analytical << std::endl;
    std::cout << "Diff Jacobian: " << std::endl
              << (bruteForce - analytical) << std::endl;
    return false;
  }
  return true;
}

bool verifyWrtMapping(WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  int dim = wrt->dim(world.get());
  if (dim == 0)
  {
    std::cout << "Got an empty WRT mapping!" << std::endl;
    return false;
  }

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs randMapping = Eigen::VectorXs::Random(dim).cwiseAbs();
    wrt->set(world.get(), randMapping);
    Eigen::VectorXs recoveredMapping = wrt->get(world.get());
    if (!equals(randMapping, recoveredMapping))
    {
      std::cout << "Didn't recover WRT mapping" << std::endl;
      std::cout << "Original mapping: " << std::endl
                << randMapping << std::endl;
      std::cout << "Recovered mapping: " << std::endl
                << recoveredMapping << std::endl;
      return false;
    }
  }

  snapshot.restore();

  return true;
}

bool verifyJacobiansWrt(WorldPtr world, WithRespectTo* wrt)
{
  return verifyWrtMapping(world, wrt) && verifyVelJacobianWrt(world, wrt)
         && verifyPosJacobianWrt(world, wrt);
}

bool verifyWrtMass(WorldPtr world)
{
  WithRespectToMass massMapping = WithRespectToMass();
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    if (skel->isMobile() && skel->getNumDofs() > 0)
    {
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        Eigen::VectorXs lowerBound = Eigen::VectorXs::Ones(1) * 0.1;
        Eigen::VectorXs upperBound = Eigen::VectorXs::Ones(1) * 1000;
        massMapping.registerNode(
            skel->getBodyNode(j), INERTIA_MASS, upperBound, lowerBound);
      }
    }
  }

  // return verifyScratch(world, &massMapping);
  // TODO: re-enable me later
  return true;

  if (!verifyJacobiansWrt(world, &massMapping))
  {
    std::cout << "Error with Jacobians on mass" << std::endl;
    return false;
  }

  WithRespectToMass inertiaMapping = WithRespectToMass();
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumBodyNodes(); j++)
    {
      Eigen::VectorXs lowerBound = Eigen::VectorXs::Ones(6) * 0.1;
      Eigen::VectorXs upperBound = Eigen::VectorXs::Ones(6) * 1000;
      inertiaMapping.registerNode(
          skel->getBodyNode(j), INERTIA_FULL, upperBound, lowerBound);
    }
  }
  verifyJacobiansWrt(world, &inertiaMapping);
  if (!verifyJacobiansWrt(world, &inertiaMapping))
  {
    std::cout << "Error with Jacobians on full" << std::endl;
    return false;
  }

  return true;
}

bool verifyTrajectory(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::MultiShot> trajectory)
{
  const trajectory::TrajectoryRollout* rollout
      = trajectory->getRolloutCache(world);
  Eigen::MatrixXs poses = rollout->getPosesConst();
  Eigen::MatrixXs vels = rollout->getVelsConst();
  Eigen::MatrixXs forces = rollout->getControlForcesConst();

  for (int i = 0; i < poses.cols(); i++)
  {
    std::cout << "Checking trajectory step " << i << "/" << poses.cols()
              << "..." << std::endl;
    world->setPositions(poses.col(i));
    world->setVelocities(vels.col(i));
    world->setControlForces(forces.col(i));

    if (!verifyAnalyticalJacobians(world, true))
    {
      std::cout << "Error in verifyAnalyticalJacobians() on trajectory step "
                << i << "/" << poses.cols() << std::endl;
      return false;
    }
    if (!verifyVelGradients(world, vels.col(i)))
    {
      std::cout << "Error in verifyVelGradients() on trajectory step " << i
                << "/" << poses.cols() << std::endl;
      return false;
    }
    if (!verifyWrtMass(world))
    {
      std::cout << "Error in verifyWrtMass() on trajectory step " << i << "/"
                << poses.cols() << std::endl;
      return false;
    }
  }

  return true;
}

/*
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
*/
#endif
