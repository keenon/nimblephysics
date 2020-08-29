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
#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;

bool equals(TimestepJacobians a, TimestepJacobians b, double threshold)
{
  return equals(a.forcePos, b.forcePos, threshold)
         && equals(a.forceVel, b.forceVel, threshold)
         && equals(a.posPos, b.posPos, threshold)
         && equals(a.posVel, b.posVel, threshold)
         && equals(a.velPos, b.velPos, threshold)
         && equals(a.velVel, b.velVel, threshold);
}

void debugMatrices(
    Eigen::MatrixXd analytical,
    Eigen::MatrixXd bruteForce,
    double threshold,
    std::string name)
{
  if (!equals(analytical, bruteForce, threshold))
  {
    std::cout << "Error at " << name << ":" << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << (analytical - bruteForce) << std::endl;
  }
}

bool verifySingleStep(WorldPtr world)
{
  SingleShot shot(world, 1);
  TimestepJacobians analyticalJacobians
      = shot.backpropStartStateJacobians(world);
  TimestepJacobians bruteForceJacobians
      = shot.finiteDifferenceStartStateJacobians(world);
  BackpropSnapshotPtr ptr = neural::forwardPass(world);
  Eigen::MatrixXd velVelAnalytical = ptr->getVelVelJacobian(world);
  Eigen::MatrixXd velVelFD = ptr->finiteDifferenceVelVelJacobian(world);

  Eigen::MatrixXd forceVel = ptr->getForceVelJacobian(world);
  Eigen::MatrixXd forceVelFD = ptr->finiteDifferenceForceVelJacobian(world);

  Eigen::MatrixXd velCJacobian = ptr->getVelCJacobian(world);
  Eigen::MatrixXd velVelCore = ptr->getForceVelJacobian(world) * velCJacobian;
  Eigen::MatrixXd velVelCoreFD
      = Eigen::MatrixXd::Identity(world->getNumDofs(), world->getNumDofs())
        - velVelFD;

  double threshold = 1e-8;

  if (!equals(analyticalJacobians.velVel, bruteForceJacobians.velVel, threshold)
      || !equals(velVelAnalytical, velVelFD, threshold)
      || !equals(velVelCore, velVelCoreFD, threshold)
      || !equals(forceVel, forceVelFD, threshold))
  {

    std::cout << "Time series: " << std::endl;
    debugMatrices(
        analyticalJacobians.velVel,
        bruteForceJacobians.velVel,
        threshold,
        "v_t -> v_end");

    std::cout << "Jacobians: " << std::endl;
    debugMatrices(velVelAnalytical, velVelFD, threshold, "v_t -> v_end");
    debugMatrices(velVelCore, velVelCoreFD, threshold, "v_t -> v_end core");
    debugMatrices(forceVel, forceVelFD, threshold, "f_t -> v_end");
    std::cout << "Vel-C: " << std::endl
              << ptr->getVelCJacobian(world) << std::endl;

    return false;
  }
  return true;
}

bool verifySingleShot(WorldPtr world, int maxSteps)
{
  for (int i = 1; i < maxSteps; i++)
  {
    SingleShot shot(world, i);
    TimestepJacobians analyticalJacobians
        = shot.backpropStartStateJacobians(world);
    TimestepJacobians bruteForceJacobians
        = shot.finiteDifferenceStartStateJacobians(world);
    double threshold = 1e-8;
    if (!equals(analyticalJacobians, bruteForceJacobians, threshold))
    {
      std::cout << "Trajectory broke at timestep " << i << ":" << std::endl;
      debugMatrices(
          analyticalJacobians.forcePos,
          bruteForceJacobians.forcePos,
          threshold,
          "f_t -> p_end");
      debugMatrices(
          analyticalJacobians.forceVel,
          bruteForceJacobians.forceVel,
          threshold,
          "f_t -> v_end");
      debugMatrices(
          analyticalJacobians.posPos,
          bruteForceJacobians.posPos,
          threshold,
          "p_t -> p_end");
      debugMatrices(
          analyticalJacobians.posVel,
          bruteForceJacobians.posVel,
          threshold,
          "p_t -> v_end");
      debugMatrices(
          analyticalJacobians.velPos,
          bruteForceJacobians.velPos,
          threshold,
          "v_t -> p_end");
      debugMatrices(
          analyticalJacobians.velVel,
          bruteForceJacobians.velVel,
          threshold,
          "v_t -> v_end");
      return false;
    }
  }
  return true;
}

bool verifyShotJacobian(WorldPtr world, int steps)
{
  SingleShot shot(world, steps, true);
  int dim = shot.getFlatProblemDim();

  // Random initialization
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);

  Eigen::MatrixXd analyticalJacobian
      = Eigen::MatrixXd::Zero(world->getNumDofs() * 2, dim);
  shot.backpropJacobianOfFinalState(world, analyticalJacobian);
  Eigen::MatrixXd bruteForceJacobian
      = Eigen::MatrixXd::Zero(world->getNumDofs() * 2, dim);
  shot.finiteDifferenceJacobianOfFinalState(world, bruteForceJacobian);
  double threshold = 1e-8;
  if (!equals(analyticalJacobian, bruteForceJacobian, threshold))
  {
    std::cout << "Jacobians don't match!" << std::endl;
    std::cout << "Analytical:" << std::endl << analyticalJacobian << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForceJacobian << std::endl;
    std::cout << "Diff:" << std::endl
              << (analyticalJacobian - bruteForceJacobian) << std::endl;
    return false;
  }
  return true;
}

bool verifyShotGradient(WorldPtr world, int steps, TrajectoryLossFn loss)
{
  SingleShot shot(world, steps, true);
  int dim = shot.getFlatProblemDim();

  // Random initialization
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);

  Eigen::MatrixXd gradWrtPoses
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtVels
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtForces
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  shot.bruteForceGradOfLossInputs(
      world, loss, gradWrtPoses, gradWrtVels, gradWrtForces);

  Eigen::VectorXd analyticalGrad = Eigen::VectorXd::Zero(dim);
  shot.backpropGradient(
      world, gradWrtPoses, gradWrtVels, gradWrtForces, analyticalGrad);
  Eigen::VectorXd bruteForceGrad = Eigen::VectorXd::Zero(dim);
  shot.finiteDifferenceGradient(world, loss, bruteForceGrad);

  // This threshold is just barely enough for the cartpole example, but the
  // fluctuation appears due to tuning EPS values for finite differencing, which
  // means I think we're within safe ranges of correct.
  double threshold = 2e-8;
  if (!equals(analyticalGrad, bruteForceGrad, threshold))
  {
    std::cout << "Gradients don't match!" << std::endl;
    std::cout << "Analytical:" << std::endl << analyticalGrad << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForceGrad << std::endl;
    std::cout << "Diff:" << std::endl
              << (analyticalGrad - bruteForceGrad) << std::endl;
    return false;
  }
  return true;
}

bool verifyMultiShotJacobian(WorldPtr world, int steps, int shotLength)
{
  MultiShot shot(world, steps, shotLength, true);
  int dim = shot.getFlatProblemDim();
  int numConstraints = shot.getConstraintDim();

  // Random initialization
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);

  Eigen::MatrixXd analyticalJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);
  shot.backpropJacobian(world, analyticalJacobian);
  Eigen::MatrixXd bruteForceJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);
  shot.finiteDifferenceJacobian(world, bruteForceJacobian);
  double threshold = 1e-8;
  if (!equals(analyticalJacobian, bruteForceJacobian, threshold))
  {
    std::cout << "Jacobians don't match!" << std::endl;
    std::cout << "Analytical:" << std::endl << analyticalJacobian << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForceJacobian << std::endl;
    std::cout << "Diff:" << std::endl
              << (analyticalJacobian - bruteForceJacobian) << std::endl;
    return false;
  }
  return true;
}

bool verifyMultiShotGradient(
    WorldPtr world, int steps, int shotLength, TrajectoryLossFn loss)
{
  MultiShot shot(world, steps, shotLength, true);
  int dim = shot.getFlatProblemDim();

  // Random initialization
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);

  Eigen::MatrixXd gradWrtPoses
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtVels
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtForces
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  shot.bruteForceGradOfLossInputs(
      world, loss, gradWrtPoses, gradWrtVels, gradWrtForces);

  Eigen::VectorXd analyticalGrad = Eigen::VectorXd::Zero(dim);
  shot.backpropGradient(
      world, gradWrtPoses, gradWrtVels, gradWrtForces, analyticalGrad);
  Eigen::VectorXd bruteForceGrad = Eigen::VectorXd::Zero(dim);
  shot.finiteDifferenceGradient(world, loss, bruteForceGrad);

  // This threshold is just barely enough for the cartpole example, but the
  // fluctuation appears due to tuning EPS values for finite differencing, which
  // means I think we're within safe ranges of correct.
  double threshold = 2e-8;
  if (!equals(analyticalGrad, bruteForceGrad, threshold))
  {
    std::cout << "Gradients don't match!" << std::endl;
    std::cout << "Analytical:" << std::endl << analyticalGrad << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForceGrad << std::endl;
    std::cout << "Diff:" << std::endl
              << (analyticalGrad - bruteForceGrad) << std::endl;
    return false;
  }
  return true;
}

/*
TEST(TRAJECTORY, UNCONSTRAINED_BOX)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

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
  boxBody->setFrictionCoeff(0.0);

  // Add a force driving the box to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(1.0);
  boxBody->setRestitutionCoeff(0.5);
  // Set the 1th joint index to -1.0
  box->setVelocity(1, -1);

  world->addSkeleton(box);

  // Passes
  // EXPECT_TRUE(verifySingleShot(world, 100));
}

TEST(TRAJECTORY, REVOLUTE_JOINT)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr spinner = Skeleton::create("spinner");

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = spinner->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  world->addSkeleton(spinner);

  spinner->setPosition(0, 15.0 / 180.0 * 3.1415);
  spinner->computeForwardDynamics();
  spinner->integrateVelocities(world->getTimeStep());

  // Passes
  // EXPECT_TRUE(verifySingleShot(world, 100));
}

TEST(TRAJECTORY, TWO_LINK)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr arm = Skeleton::create("arm");

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(nullptr);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));

  // Add child arm

  std::pair<RevoluteJoint*, BodyNode*> elbowPair
      = arm->createJointAndBodyNodePair<RevoluteJoint>(armPair.second);
  Eigen::Isometry3d elbowOffset = Eigen::Isometry3d::Identity();
  elbowOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
  elbowPair.first->setTransformFromParentBodyNode(elbowOffset);

  world->addSkeleton(arm);

  arm->setPosition(0, 15.0 / 180.0 * 3.1415);
  arm->computeForwardDynamics();
  arm->integrateVelocities(world->getTimeStep());

  // Passes
  // EXPECT_TRUE(verifySingleShot(world, 100));
}

TEST(TRAJECTORY, PRISMATIC)
{

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 1, 0));

  world->addSkeleton(cartpole);

  cartpole->setPosition(0, 0);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  // Passes
  // EXPECT_TRUE(verifySingleShot(world, 100));
}
*/

TEST(TRAJECTORY, CARTPOLE)
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

  world->addSkeleton(cartpole);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  EXPECT_TRUE(verifySingleStep(world));
  EXPECT_TRUE(verifySingleShot(world, 40));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  TrajectoryLossFn loss = [](Eigen::Ref<Eigen::MatrixXd> poses,
                             Eigen::Ref<Eigen::MatrixXd> vels,
                             Eigen::Ref<Eigen::MatrixXd> forces) {
    return vels.col(vels.cols() - 1).squaredNorm()
           + poses.col(poses.cols() - 1).squaredNorm() + forces.squaredNorm();
  };
  EXPECT_TRUE(verifyShotGradient(world, 7, loss));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
  EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss));
}