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

  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  Eigen::MatrixXd A_cInv
      = A_c.completeOrthogonalDecomposition().pseudoInverse();
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
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  Eigen::MatrixXd V_c = classicPtr->getMassedClampingConstraintMatrix();
  Eigen::MatrixXd M = classicPtr->getMassMatrix();
  Eigen::MatrixXd Minv = classicPtr->getInvMassMatrix();

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
  Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix();
  Eigen::MatrixXd V_ub = classicPtr->getMassedUpperBoundConstraintMatrix();
  Eigen::MatrixXd M = classicPtr->getMassMatrix();
  Eigen::MatrixXd Minv = classicPtr->getInvMassMatrix();

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

  MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix();
  VectorXd analyticalConstraintForces = -1 * P_c * integratedVelocities;

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
    std::cout << "Integrated velocities: " << std::endl
              << integratedVelocities << std::endl;
    std::cout << "P_c: " << std::endl << P_c << std::endl;
    std::cout << "Constraint forces: " << std::endl
              << contactConstraintForces << std::endl;
    std::cout << "bounce: " << std::endl
              << classicPtr->getBounceDiagonals() << std::endl;
    std::cout << "-(P_c * proposedVelocities) (should be the same as above): "
              << std::endl
              << analyticalConstraintForces << std::endl;
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

  Eigen::MatrixXd P_c = classicPtr->getProjectionIntoClampsMatrix();

  // Reconstruct P_c without the massed shortcut
  Eigen::MatrixXd A_c = classicPtr->getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = classicPtr->getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = classicPtr->getUpperBoundMappingMatrix();

  Eigen::MatrixXd constraintForceToImpliedTorques = A_c + (A_ub * E);
  Eigen::MatrixXd forceToVel = A_c.eval().transpose()
                               * classicPtr->getInvMassMatrix()
                               * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXd bounce = classicPtr->getBounceDiagonals().asDiagonal();
  Eigen::MatrixXd P_c_recovered
      = (1.0 / world->getTimeStep()) * velToForce * bounce * A_c.transpose();

  if (!equals(P_c, P_c_recovered, 1e-3))
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

bool verifyVelGradients(WorldPtr world, VectorXd worldVel)
{
  return (
      verifyClassicClampingConstraintMatrix(world, worldVel)
      && verifyMassedClampingConstraintMatrix(world, worldVel)
      && verifyMassedUpperBoundConstraintMatrix(world, worldVel)
      && verifyClassicProjectionIntoClampsMatrix(world, worldVel)
      && verifyMassedProjectionIntoClampsMatrix(world, worldVel)
      && verifyVelVelJacobian(world, worldVel)
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

  MatrixXd analytical = classicPtr->getPosPosJacobian();
  MatrixXd bruteForce
      = classicPtr->finiteDifferencePosPosJacobian(subdivisions);

  if (!equals(analytical, bruteForce, 1e-1))
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

  MatrixXd analytical = classicPtr->getVelPosJacobian();
  MatrixXd bruteForce
      = classicPtr->finiteDifferenceVelPosJacobian(subdivisions);

  if (!equals(analytical, bruteForce, 1e-1))
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

bool verifyBackpropInstance(
    const neural::BackpropSnapshotPtr& classicPtr, const VectorXd& phaseSpace)
{
  LossGradient nextTimeStep;
  nextTimeStep.lossWrtPosition = phaseSpace.segment(0, phaseSpace.size() / 2);
  nextTimeStep.lossWrtVelocity
      = phaseSpace.segment(phaseSpace.size() / 2, phaseSpace.size() / 2);

  LossGradient thisTimeStep;
  classicPtr->backprop(thisTimeStep, nextTimeStep);

  // Compute "brute force" backprop using full Jacobians
  /*
  The forward computation graph looks like this:

  p_t ---------------------> p_t+1 ---->
                               ^
                               |
  v_t ------> v_t+i -----------+------->
                ^
                |
  f_t ----------+
  */

  // p_t
  VectorXd lossWrtThisPosition = classicPtr->getPosPosJacobian().transpose()
                                 * nextTimeStep.lossWrtPosition;

  // v_t+1
  VectorXd lossWrtNextVelocity = nextTimeStep.lossWrtVelocity
                                 + classicPtr->getVelPosJacobian().transpose()
                                       * nextTimeStep.lossWrtPosition;

  // v_t
  VectorXd lossWrtThisVelocity = classicPtr->getVelVelJacobian().transpose()
                                 * nextTimeStep.lossWrtVelocity;

  // f_t
  VectorXd lossWrtThisTorque = classicPtr->getForceVelJacobian().transpose()
                               * nextTimeStep.lossWrtVelocity;

  if (!equals(lossWrtThisPosition, thisTimeStep.lossWrtPosition, 1e-5)
      || !equals(lossWrtThisVelocity, thisTimeStep.lossWrtVelocity, 1e-5)
      || !equals(lossWrtThisTorque, thisTimeStep.lossWrtTorque, 1e-5))
  {
    std::cout << "Input: loss wrt position at time t + 1:" << std::endl
              << nextTimeStep.lossWrtPosition << std::endl;
    std::cout << "Input: loss wrt velocity at time t + 1:" << std::endl
              << nextTimeStep.lossWrtVelocity << std::endl;
    std::cout << "Brute force: loss wrt position at time t:" << std::endl
              << lossWrtThisPosition << std::endl;
    std::cout << "Analytical: loss wrt position at time t:" << std::endl
              << thisTimeStep.lossWrtPosition << std::endl;
    std::cout << "Brute force: loss wrt velocity at time t:" << std::endl
              << lossWrtThisVelocity << std::endl;
    std::cout << "Analytical: loss wrt velocity at time t:" << std::endl
              << thisTimeStep.lossWrtVelocity << std::endl;
    std::cout << "Brute force: loss wrt torque at time t:" << std::endl
              << lossWrtThisTorque << std::endl;
    std::cout << "Analytical: loss wrt torque at time t:" << std::endl
              << thisTimeStep.lossWrtTorque << std::endl;
    return false;
  }
  return true;
}

bool verifyBackprop(WorldPtr world)
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
    if (!verifyBackpropInstance(classicPtr, phaseSpace))
      return false;
  }

  // Test all "0"s
  phaseSpace = VectorXd::Zero(world->getNumDofs() * 2);
  if (!verifyBackpropInstance(classicPtr, phaseSpace))
    return false;

  // Test all "1"s
  phaseSpace = VectorXd::Ones(world->getNumDofs() * 2);
  if (!verifyBackpropInstance(classicPtr, phaseSpace))
    return false;

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
  EXPECT_TRUE(verifyBackprop(world));
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

  world->getConstraintSolver()->setGradientEnabled(true);
  world->getConstraintSolver()->solve();

  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyBackprop(world));
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
  EXPECT_TRUE(verifyBackprop(world));
}

TEST(GRADIENTS, BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockWithFrictionCoeff(0, 1);
}

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
  EXPECT_TRUE(verifyBackprop(world));
}

TEST(GRADIENTS, SLIDING_REVERSE_PENDULUM_NO_FRICTION)
{
  testReversePendulumSledWithFrictionCoeff(0);
}

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
  EXPECT_TRUE(verifyBackprop(world));
}

TEST(GRADIENTS, POS_BLOCK_BOUNCING_OFF_GROUND_NO_FRICTION_1_MASS)
{
  testBouncingBlockPosGradients(0, 1);
}