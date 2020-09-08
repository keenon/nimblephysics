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

#include <chrono>
#include <iostream>
#include <thread>

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
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

#include "GradientTestUtils.hpp"
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

bool verifySingleStep(WorldPtr world, double EPS)
{
  SingleShot shot(world, 1);
  TimestepJacobians analyticalJacobians
      = shot.backpropStartStateJacobians(world);
  TimestepJacobians bruteForceJacobians
      = shot.finiteDifferenceStartStateJacobians(world, EPS);
  BackpropSnapshotPtr ptr = neural::forwardPass(world);
  Eigen::MatrixXd velVelAnalytical = ptr->getVelVelJacobian(world);
  Eigen::MatrixXd velVelFD = ptr->finiteDifferenceVelVelJacobian(world);

  Eigen::MatrixXd forceVel = ptr->getForceVelJacobian(world);
  Eigen::MatrixXd forceVelFD = ptr->finiteDifferenceForceVelJacobian(world);

  Eigen::MatrixXd velCJacobian = ptr->getVelCJacobian(world);

  double threshold = 1e-8;

  if (!equals(analyticalJacobians.velVel, bruteForceJacobians.velVel, threshold)
      || !equals(velVelAnalytical, velVelFD, threshold)
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
    debugMatrices(forceVel, forceVelFD, threshold, "f_t -> v_end");
    std::cout << "Vel-C: " << std::endl
              << ptr->getVelCJacobian(world) << std::endl;

    return false;
  }
  return true;
}

bool verifySingleShot(WorldPtr world, int maxSteps, double EPS, bool useFdJacs)
{
  for (int i = 1; i < maxSteps; i++)
  {
    SingleShot shot(world, i);

    double threshold = 1e-8;
    std::vector<BackpropSnapshotPtr> ptrs = shot.getSnapshots(world);
    for (int j = 0; j < ptrs.size(); j++)
    {
      if (!useFdJacs
          && (!equals(
                  ptrs[j]->getPosPosJacobian(world),
                  ptrs[j]->finiteDifferencePosPosJacobian(world, 1),
                  threshold)
              || !equals(
                  ptrs[j]->getVelPosJacobian(world),
                  ptrs[j]->finiteDifferenceVelPosJacobian(world, 1),
                  threshold)
              || !equals(
                  ptrs[j]->getPosVelJacobian(world),
                  ptrs[j]->finiteDifferencePosVelJacobian(world),
                  threshold)
              || !equals(
                  ptrs[j]->getVelVelJacobian(world),
                  ptrs[j]->finiteDifferenceVelVelJacobian(world),
                  threshold)
              || !equals(
                  ptrs[j]->getForceVelJacobian(world),
                  ptrs[j]->finiteDifferenceForceVelJacobian(world),
                  threshold)))
      {
        std::cout << "Detected Jac imprecision at step " << (j + 1) << "/" << i
                  << std::endl;
        debugMatrices(
            ptrs[j]->getPosPosJacobian(world),
            ptrs[j]->finiteDifferencePosPosJacobian(world, 1),
            threshold,
            "pos-pos jac");
        debugMatrices(
            ptrs[j]->getVelPosJacobian(world),
            ptrs[j]->finiteDifferenceVelPosJacobian(world, 1),
            threshold,
            "vel-pos jac");
        debugMatrices(
            ptrs[j]->getPosVelJacobian(world),
            ptrs[j]->finiteDifferencePosVelJacobian(world),
            threshold,
            "pos-vel jac");
        debugMatrices(
            ptrs[j]->getVelVelJacobian(world),
            ptrs[j]->finiteDifferenceVelVelJacobian(world),
            threshold,
            "vel-vel jac");
        debugMatrices(
            ptrs[j]->getForceVelJacobian(world),
            ptrs[j]->finiteDifferenceForceVelJacobian(world),
            threshold,
            "force-vel jac");

        world->setPositions(ptrs[j]->getPreStepPosition());
        world->setVelocities(ptrs[j]->getPreStepVelocity());
        world->setForces(ptrs[j]->getPreStepTorques());
        verifyVelGradients(world, ptrs[j]->getPreStepVelocity());

        return false;
      }
    }

    TimestepJacobians analyticalJacobians
        = shot.backpropStartStateJacobians(world, useFdJacs);
    TimestepJacobians bruteForceJacobians
        = shot.finiteDifferenceStartStateJacobians(world, 1e-7);
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
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

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

bool verifyShotGradient(
    WorldPtr world,
    int steps,
    TrajectoryLossFn loss,
    TrajectoryLossFnGrad lossGrad)
{
  SingleShot shot(world, steps, true);
  int dim = shot.getFlatProblemDim();

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  Eigen::MatrixXd poses = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd vels = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  shot.getStates(world, poses, vels, forces, true);

  Eigen::MatrixXd gradWrtPoses
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtVels
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtForces
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  /*
  shot.bruteForceGradOfLossInputs(
    world, loss, gradWrtPoses, gradWrtVels, gradWrtForces);
  */
  lossGrad(poses, vels, forces, gradWrtPoses, gradWrtVels, gradWrtForces);
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
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  /*
  Eigen::VectorXd pos = randomInit.segment(20, 5);
  Eigen::VectorXd vel = randomInit.segment(25, 5);
  */

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
    for (int i = 0; i < dim; i++)
    {
      Eigen::VectorXd analyticalCol = analyticalJacobian.col(i);
      Eigen::VectorXd bruteForceCol = bruteForceJacobian.col(i);
      if (!equals(analyticalCol, bruteForceCol, threshold))
      {
        std::cout << "ERROR at col " << shot.getFlatDimName(i) << " (" << i
                  << ") by " << (analyticalCol - bruteForceCol).norm()
                  << std::endl;
        /*
        std::cout << "Analytical:" << std::endl << analyticalCol << std::endl;
        std::cout << "Brute Force:" << std::endl << bruteForceCol << std::endl;
        std::cout << "Diff:" << std::endl
                  << (analyticalCol - bruteForceCol) << std::endl;
        */
      }
      else
      {
        std::cout << "Match at col " << shot.getFlatDimName(i) << " (" << i
                  << ")" << std::endl;
      }
    }
    return false;
  }
  return true;
}

bool verifySparseJacobian(WorldPtr world, int steps, int shotLength)
{
  MultiShot shot(world, steps, shotLength, true);

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  int dim = shot.getFlatProblemDim();
  int numConstraints = shot.getConstraintDim();
  Eigen::MatrixXd analyticalJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);
  shot.backpropJacobian(world, analyticalJacobian);
  Eigen::MatrixXd sparseRecoveredJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);

  int numSparse = shot.getNumberNonZeroJacobian();
  Eigen::VectorXi rows = Eigen::VectorXi::Zero(numSparse);
  Eigen::VectorXi cols = Eigen::VectorXi::Zero(numSparse);
  shot.getJacobianSparsityStructure(rows, cols);
  Eigen::VectorXd sparseValues = Eigen::VectorXd::Zero(numSparse);
  shot.getSparseJacobian(world, sparseValues);
  for (int i = 0; i < numSparse; i++)
  {
    sparseRecoveredJacobian(rows(i), cols(i)) = sparseValues(i);
  }

  double threshold = 1e-8;
  if (!equals(analyticalJacobian, sparseRecoveredJacobian, threshold))
  {
    std::cout << "Jacobians don't match!" << std::endl;
    for (int i = 0; i < dim; i++)
    {
      Eigen::VectorXd analyticalCol = analyticalJacobian.col(i);
      Eigen::VectorXd sparseRecoveredCol = sparseRecoveredJacobian.col(i);
      if (!equals(analyticalCol, sparseRecoveredCol, threshold))
      {
        std::cout << "ERROR at col " << shot.getFlatDimName(i) << " (" << i
                  << ") by " << (analyticalCol - sparseRecoveredCol).norm()
                  << std::endl;
        /*
        std::cout << "Dense:" << std::endl << analyticalCol << std::endl;
        std::cout << "Sparse:" << std::endl << sparseRecoveredCol << std::endl;
        std::cout << "Diff:" << std::endl
                  << (analyticalCol - bruteForceCol) << std::endl;
        */
      }
      else
      {
        std::cout << "Match at col " << shot.getFlatDimName(i) << " (" << i
                  << ")" << std::endl;
      }
    }
    return false;
  }
  return true;
}

bool verifyMultiShotGradient(
    WorldPtr world,
    int steps,
    int shotLength,
    TrajectoryLossFn loss,
    TrajectoryLossFnGrad lossGrad)
{
  MultiShot shot(world, steps, shotLength, true);

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  Eigen::MatrixXd poses = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd vels = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  shot.getStates(world, poses, vels, forces, true);

  int dim = shot.getFlatProblemDim();
  Eigen::MatrixXd gradWrtPoses
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtVels
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd gradWrtForces
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  /*
  shot.bruteForceGradOfLossInputs(
      world, loss, gradWrtPoses, gradWrtVels, gradWrtForces);
  */
  lossGrad(poses, vels, forces, gradWrtPoses, gradWrtVels, gradWrtForces);

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

//==============================================================================
class AbstractShotWindow : public dart::gui::glut::SimWindow
{
public:
  /// Constructor
  AbstractShotWindow(
      std::shared_ptr<simulation::World> world,
      Eigen::MatrixXd posesWithKnots,
      Eigen::MatrixXd posesWithoutKnots)
  {
    mPosesWithKnots = posesWithKnots;
    mPosesWithoutKnots = posesWithoutKnots;
    mCounter = 0;
    setWorld(world);
  }

  void timeStepping() override
  {
    // std::cout << "Time stepping " << mCounter << std::endl;
    mCounter++;
    int cols = mPosesWithKnots.cols();
    if (mCounter < cols)
      mWorld->setPositions(mPosesWithKnots.col(mCounter));
    if (mCounter >= 2 * cols && mCounter < 3 * cols)
      mWorld->setPositions(mPosesWithoutKnots.col(mCounter - 2 * cols));
    if (mCounter >= 4 * cols)
      mCounter = 0;

    // Step the simulation forward
    SimWindow::draw();
    // SimWindow::timeStepping();
  }

private:
  int mCounter = 0;
  Eigen::MatrixXd mPosesWithKnots;
  Eigen::MatrixXd mPosesWithoutKnots;
};

bool verifyMultiShotOptimization(
    WorldPtr world, int steps, int shotLength, TrajectoryLossFn loss)
{
  MultiShot shot(world, steps, shotLength, false);
  shot.setLossFunction(loss);
  IPOptOptimizer optimizer;

  optimizer.optimize(&shot);

  // Playback the trajectory

  Eigen::MatrixXd posesWithKnots
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd posesWithoutKnots
      = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd vels = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);

  // Get the version with knots
  shot.getStates(world, posesWithKnots, vels, forces, true);
  // Get the version without knots next, so that they can play in a loop
  shot.getStates(world, posesWithoutKnots, vels, forces, false);

  // Create a window for rendering the world and handling user input
  AbstractShotWindow window(world, posesWithKnots, posesWithoutKnots);

  // Initialize glut, initialize the window, and begin the glut event loop
  int argc = 0;
  glutInit(&argc, nullptr);
  window.initWindow(640, 480, "Test");
  glutMainLoop();
}

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
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
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
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
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
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
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
  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
  // EXPECT_TRUE(verifyShotGradient(world, 7, loss));
  // EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss));
}

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

  cartpole->setForceUpperLimit(0, 0);
  cartpole->setForceLowerLimit(0, 0);
  cartpole->setVelocityUpperLimit(0, 1000);
  cartpole->setVelocityLowerLimit(0, -1000);
  cartpole->setPositionUpperLimit(0, 10);
  cartpole->setPositionLowerLimit(0, -10);

  cartpole->setForceLowerLimit(1, -1000);
  cartpole->setForceUpperLimit(1, 1000);
  cartpole->setVelocityUpperLimit(1, 1000);
  cartpole->setVelocityLowerLimit(1, -1000);
  cartpole->setPositionUpperLimit(1, 10);
  cartpole->setPositionLowerLimit(1, -10);

  cartpole->setPosition(0, 0);
  cartpole->setPosition(1, 15.0 / 180.0 * 3.1415);
  cartpole->computeForwardDynamics();
  cartpole->integrateVelocities(world->getTimeStep());

  TrajectoryLossFn loss = [](const Eigen::Ref<const Eigen::MatrixXd>& poses,
                             const Eigen::Ref<const Eigen::MatrixXd>& vels,
                             const Eigen::Ref<const Eigen::MatrixXd>& forces) {
    Eigen::VectorXd lastPos = poses.col(poses.cols() - 1);
    return vels.col(vels.cols() - 1).squaredNorm() + lastPos.squaredNorm()
           + forces.squaredNorm();
  };

  TrajectoryLossFnGrad lossGrad
      = [](const Eigen::Ref<const Eigen::MatrixXd>& poses,
           const Eigen::Ref<const Eigen::MatrixXd>& vels,
           const Eigen::Ref<const Eigen::MatrixXd>& forces,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces) {
          gradWrtPoses.setZero();
          gradWrtVels.setZero();
          gradWrtForces.setZero();
          gradWrtPoses.col(poses.cols() - 1) = 2 * poses.col(poses.cols() - 1);
          gradWrtVels.col(vels.cols() - 1) = 2 * vels.col(vels.cols() - 1);
          for (int i = 0; i < forces.cols(); i++)
          {
            gradWrtForces.col(i) = 2 * forces.col(i);
          }
        };

  EXPECT_TRUE(verifySingleStep(world, 1e-7));
  EXPECT_TRUE(verifySingleShot(world, 40, 1e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 40));
  EXPECT_TRUE(verifyShotGradient(world, 7, loss, lossGrad));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 8, 2));
  EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss, lossGrad));
  // EXPECT_TRUE(verifyMultiShotOptimization(world, 50, 10, loss));
}

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
  poleJoint->setVelocityUpperLimit(0, 100.0);
  poleJoint->setVelocityLowerLimit(0, -100.0);
  poleJoint->setPositionUpperLimit(0, 270 * 3.1415 / 180);
  poleJoint->setPositionLowerLimit(0, -270 * 3.1415 / 180);

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

TEST(TRAJECTORY, JUMP_WORM)
{
  bool offGround = false;

  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  world->getConstraintSolver()->setPenetrationCorrectionEnabled(false);

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
  rootJoint->setPositionUpperLimit(0, 5);
  rootJoint->setPositionLowerLimit(0, -5);
  rootJoint->setPositionUpperLimit(1, 5);
  rootJoint->setPositionLowerLimit(1, -5);

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

  TrajectoryLossFn loss = [](const Eigen::Ref<const Eigen::MatrixXd>& poses,
                             const Eigen::Ref<const Eigen::MatrixXd>& vels,
                             const Eigen::Ref<const Eigen::MatrixXd>& forces) {
    double maxPos = -1000;
    double minPos = 1000;
    for (int i = 0; i < poses.cols(); i++)
    {
      if (poses(1, i) > maxPos)
      {
        maxPos = poses(1, i);
      }
      if (poses(1, i) < minPos)
      {
        minPos = poses(1, i);
      }
    }
    double peakPosLoss = -(maxPos * maxPos) * (maxPos > 0 ? 1.0 : -1.0);
    double minPosLoss = -(minPos * minPos) * (minPos > 0 ? 1.0 : -1.0);
    double endPos = poses(1, poses.cols() - 1);
    double endPosLoss = -(endPos * endPos) * (endPos > 0 ? 1.0 : -1.0);

    double forceLoss = forces.squaredNorm();

    // return endPosLoss * 100 + forceLoss * 1e-3;
    // return forceLoss;
    return endPosLoss; // + forceLoss;
    // return (100 * peakPosLoss) + (20 * minPosLoss) + endPosLoss;

    /*
    Eigen::VectorXd midVel = vels.col(vels.cols() / 2);
    double midVelSquaredSigned
        = -(midVel[1] * midVel[1]) * (midVel[1] > 0 ? 1.0 : -1.0);

    return posSquaredSigned + midVelSquaredSigned;
    */
    /*
    return (pos[0] * pos[0]) + (pos[1] * pos[1]) + (vel[0] * vel[0])
           + (vel[1] * vel[1]);
    */
  };

  TrajectoryLossFnGrad lossGrad
      = [](const Eigen::Ref<const Eigen::MatrixXd>& poses,
           const Eigen::Ref<const Eigen::MatrixXd>& vels,
           const Eigen::Ref<const Eigen::MatrixXd>& forces,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
           /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces) {
          gradWrtPoses.setZero();
          gradWrtVels.setZero();
          gradWrtForces.setZero();
          gradWrtPoses(1, poses.cols() - 1) = 2 * poses(1, poses.cols() - 1);
        };

  // Make a huge timestep, to try to make the gradients easier to get exactly
  // for finite differencing
  world->setTimeStep(1e-3);

  world->getConstraintSolver()->setPenetrationCorrectionEnabled(false);

  /*
  // Initial pos that creates deep inter-penetration and generates larger
  // gradient errors
  Eigen::VectorXd initialPos = Eigen::VectorXd(5);
  initialPos << 0.96352, -0.5623, -0.0912082, 0.037308, 0.147683;
  // Initial vel
  Eigen::VectorXd initialVel = Eigen::VectorXd(5);
  initialVel << 0.110462, 0.457093, 0.257748, 0.592256, 0.167432;

  world->setPositions(initialPos);
  world->setVelocities(initialVel);
  */

  /*
  EXPECT_TRUE(verifyVelGradients(world, world->getVelocities()));
  EXPECT_TRUE(verifyNoMultistepIntereference(world, 10));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  */
  // renderWorld(world);

  // EXPECT_TRUE(verifyMultiShotOptimization(world, 600, 20, loss));
  EXPECT_TRUE(verifySingleStep(world, 5e-7));
  // EXPECT_TRUE(verifySingleShot(world, 40, 5e-7, false));
  EXPECT_TRUE(verifyShotJacobian(world, 4));
  EXPECT_TRUE(verifyShotGradient(world, 7, loss, lossGrad));
  EXPECT_TRUE(verifyMultiShotJacobian(world, 6, 2));
  EXPECT_TRUE(verifySparseJacobian(world, 8, 2));
  EXPECT_TRUE(verifyMultiShotGradient(world, 8, 4, loss, lossGrad));
}