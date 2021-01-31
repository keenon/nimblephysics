#ifndef TRAJECTORY_TEST_UTILS
#define TRAJECTORY_TEST_UTILS

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
#include "dart/gui/glut/TrajectoryReplayWindow.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

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
  LossFn lossFn = LossFn();
  SingleShot shot(world, lossFn, 1);
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

bool verifySingleShot(
    WorldPtr world,
    int maxSteps,
    double /* EPS */,
    bool useFdJacs,
    std::shared_ptr<Mapping> mapping)
{
  for (int i = 1; i < maxSteps; i++)
  {
    LossFn lossFn = LossFn();
    SingleShot shot(world, lossFn, i);
    if (mapping != nullptr)
    {
      shot.addMapping("custom", mapping);
      shot.switchRepresentationMapping(world, "custom");
    }

    double threshold = 1e-8;
    std::vector<MappedBackpropSnapshotPtr> ptrs = shot.getSnapshots(world);
    /*
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
        world->setExternalForces(ptrs[j]->getPreStepTorques());
        verifyVelGradients(world, ptrs[j]->getPreStepVelocity());

        return false;
      }
    }
    */

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

bool verifyShotJacobian(
    WorldPtr world, int steps, std::shared_ptr<Mapping> mapping)
{
  LossFn lossFn = LossFn();
  SingleShot shot(world, lossFn, steps, true);
  int stateSize = world->getNumDofs() * 2;
  if (mapping != nullptr)
  {
    shot.addMapping("custom", mapping);
    shot.switchRepresentationMapping(world, "custom");
    stateSize = mapping->getPosDim() + mapping->getVelDim();
  }

  int dim = shot.getFlatProblemDim(world);

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  Eigen::MatrixXd analyticalJacobian = Eigen::MatrixXd::Zero(stateSize, dim);
  shot.backpropJacobianOfFinalState(world, analyticalJacobian);
  Eigen::MatrixXd bruteForceJacobian = Eigen::MatrixXd::Zero(stateSize, dim);
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
    TrajectoryLossFnAndGrad lossGrad)
{
  LossFn lossFn = LossFn(loss, lossGrad);
  SingleShot shot(world, lossFn, steps, true);
  int dim = shot.getFlatProblemDim(world);

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  Eigen::VectorXd analyticalGrad = Eigen::VectorXd::Zero(dim);
  shot.backpropGradient(world, analyticalGrad);
  Eigen::VectorXd bruteForceGrad = Eigen::VectorXd::Zero(dim);
  shot.finiteDifferenceGradient(world, bruteForceGrad);

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

bool verifyMultiShotJacobian(
    WorldPtr world, int steps, int shotLength, std::shared_ptr<Mapping> mapping)
{
  LossFn lossFn = LossFn();
  MultiShot shot(world, lossFn, steps, shotLength, true);
  if (mapping != nullptr)
  {
    shot.addMapping("custom", mapping);
    shot.switchRepresentationMapping(world, "custom");
  }

  int dim = shot.getFlatProblemDim(world);
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
  shot.Problem::backpropJacobian(world, analyticalJacobian);
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
        std::cout << "ERROR at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ") by " << (analyticalCol - bruteForceCol).norm()
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
        std::cout << "Match at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ")" << std::endl;
      }
    }
    return false;
  }
  return true;
}

bool verifySparseJacobian(WorldPtr world, MultiShot& shot)
{
  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  int dim = shot.getFlatProblemDim(world);
  int numConstraints = shot.getConstraintDim();
  Eigen::MatrixXd analyticalJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);
  shot.Problem::backpropJacobian(world, analyticalJacobian);
  Eigen::MatrixXd sparseRecoveredJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);

  int numSparse = shot.getNumberNonZeroJacobian(world);
  Eigen::VectorXi rows = Eigen::VectorXi::Zero(numSparse);
  Eigen::VectorXi cols = Eigen::VectorXi::Zero(numSparse);
  shot.getJacobianSparsityStructure(world, rows, cols);
  Eigen::VectorXd sparseValues = Eigen::VectorXd::Zero(numSparse);
  shot.Problem::getSparseJacobian(world, sparseValues);
  for (int i = 0; i < numSparse; i++)
  {
    sparseRecoveredJacobian(rows(i), cols(i)) = sparseValues(i);
  }

  double threshold = 0;
  if (!equals(analyticalJacobian, sparseRecoveredJacobian, threshold))
  {
    std::cout << "Sparse jacobians don't match!" << std::endl;
    // int staticDim = shot.getFlatStaticProblemDim(world);
    std::cout << "Static region size: " << shot.getFlatStaticProblemDim(world)
              << std::endl;
    std::cout << "Analytical first region: " << std::endl
              << analyticalJacobian.block(0, 0, analyticalJacobian.rows(), 10)
              << std::endl;
    std::cout << "Sparse recovered region: " << std::endl
              << sparseRecoveredJacobian.block(
                     0, 0, analyticalJacobian.rows(), 10)
              << std::endl;

    for (int i = 0; i < dim; i++)
    {
      Eigen::VectorXd analyticalCol = analyticalJacobian.col(i);
      Eigen::VectorXd sparseRecoveredCol = sparseRecoveredJacobian.col(i);
      if (!equals(analyticalCol, sparseRecoveredCol, threshold))
      {
        std::cout << "ERROR at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ") by " << (analyticalCol - sparseRecoveredCol).norm()
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
        std::cout << "Match at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ")" << std::endl;
      }
    }
    return false;
  }
  return true;
}

bool verifySparseJacobian(
    WorldPtr world, int steps, int shotLength, std::shared_ptr<Mapping> mapping)
{
  LossFn lossFn = LossFn();
  MultiShot shot(world, lossFn, steps, shotLength, true);
  if (mapping != nullptr)
  {
    shot.addMapping("custom", mapping);
    shot.switchRepresentationMapping(world, "custom");
  }
  return verifySparseJacobian(world, shot);
}

bool verifyMultiShotGradient(
    WorldPtr world,
    int steps,
    int shotLength,
    TrajectoryLossFn loss,
    TrajectoryLossFnAndGrad lossGrad)
{
  LossFn lossFn(loss, lossGrad);
  MultiShot shot(world, lossFn, steps, shotLength, true);

  // Random initialization
  /*
  srand(42);
  Eigen::VectorXd randomInit = Eigen::VectorXd::Random(dim);
  shot.unflatten(randomInit);
  */

  int dim = shot.getFlatProblemDim(world);

  Eigen::VectorXd analyticalGrad = Eigen::VectorXd::Zero(dim);
  shot.backpropGradient(world, analyticalGrad);
  Eigen::VectorXd bruteForceGrad = Eigen::VectorXd::Zero(dim);
  shot.finiteDifferenceGradient(world, bruteForceGrad);

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

bool verifyMultiShotJacobianCustomConstraint(
    WorldPtr world,
    int steps,
    int shotLength,
    TrajectoryLossFn constraint,
    TrajectoryLossFnAndGrad constraintGrad,
    double constraintValue)
{
  LossFn lossFn = LossFn();
  MultiShot shot(world, lossFn, steps, shotLength, true);

  LossFn constraintFn = LossFn(constraint, constraintGrad);
  constraintFn.setLowerBound(constraintValue);
  constraintFn.setUpperBound(constraintValue);
  shot.addConstraint(constraintFn);

  int dim = shot.getFlatProblemDim(world);
  int numConstraints = shot.getConstraintDim();

  Eigen::MatrixXd analyticalJacobian
      = Eigen::MatrixXd::Zero(numConstraints, dim);
  shot.Problem::backpropJacobian(world, analyticalJacobian);
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
        std::cout << "ERROR at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ") by " << (analyticalCol - bruteForceCol).norm()
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
        std::cout << "Match at col " << shot.getFlatDimName(world, i) << " ("
                  << i << ")" << std::endl;
      }
    }
    return false;
  }
  return true;
}

bool verifyChangeRepresentationToIK(
    WorldPtr world,
    int steps,
    int shotLength,
    std::shared_ptr<IKMapping> newRepresentation,
    bool shouldBeLosslessInto,
    bool shouldBeLosslessOut)
{
  LossFn lossFn = LossFn();
  MultiShot shot(world, lossFn, steps, shotLength, true);

  // Get the initial state
  TrajectoryRolloutReal initialIdentityRollout = TrajectoryRolloutReal(&shot);
  shot.getStates(world, &initialIdentityRollout, nullptr, true);

  shot.addMapping("custom", newRepresentation);
  // Switch to a mapped state, and get the problem state
  shot.switchRepresentationMapping(world, "custom");

  TrajectoryRolloutReal mappedRollout = TrajectoryRolloutReal(&shot);
  shot.getStates(world, &mappedRollout, nullptr, true);

  // Go back to identity maps
  shot.switchRepresentationMapping(world, "identity");

  TrajectoryRolloutReal recoveredIdentityRollout = TrajectoryRolloutReal(&shot);
  shot.getStates(world, &recoveredIdentityRollout, nullptr, true);

  double threshold = 1e-8;

  if (shouldBeLosslessInto)
  {
    for (int i = 0; i < steps; i++)
    {
      world->setPositions(initialIdentityRollout.getPoses("identity").col(i));
      world->setVelocities(initialIdentityRollout.getVels("identity").col(i));
      world->setExternalForces(
          initialIdentityRollout.getForces("identity").col(i));

      Eigen::VectorXd manualMappedPos = newRepresentation->getPositions(world);
      Eigen::VectorXd manualMappedVel = newRepresentation->getVelocities(world);
      Eigen::VectorXd manualMappedForce = newRepresentation->getForces(world);
      Eigen::VectorXd mappedPos = mappedRollout.getPoses("custom").col(i);
      Eigen::VectorXd mappedVel = mappedRollout.getVels("custom").col(i);
      Eigen::VectorXd mappedForce = mappedRollout.getForces("custom").col(i);

      if (!equals(mappedPos, manualMappedPos, threshold)
          || !equals(mappedVel, manualMappedVel, threshold)
          || !equals(mappedForce, manualMappedForce, threshold))
      {
        std::cout << "verifyChangeRepresentationToIK() failed to be lossloss "
                     "in the into mapping "
                     "when shouldBeLosslessInto=true"
                  << std::endl;
        return false;
      }
    }
  }

  if (shouldBeLosslessOut)
  {
    for (int i = 0; i < steps; i++)
    {
      Eigen::VectorXd mappedPos = mappedRollout.getPoses("custom").col(i);
      Eigen::VectorXd mappedVel = mappedRollout.getVels("custom").col(i);
      Eigen::VectorXd mappedForce = mappedRollout.getForces("custom").col(i);
      newRepresentation->setPositions(world, mappedPos);
      newRepresentation->setVelocities(world, mappedVel);
      newRepresentation->setForces(world, mappedForce);

      Eigen::VectorXd recoveredPos
          = recoveredIdentityRollout.getPoses("identity").col(i);
      Eigen::VectorXd recoveredVel
          = recoveredIdentityRollout.getVels("identity").col(i);
      Eigen::VectorXd recoveredForce
          = recoveredIdentityRollout.getForces("identity").col(i);
      Eigen::VectorXd manualRecoveredPos = world->getPositions();
      Eigen::VectorXd manualRecoveredVel = world->getVelocities();
      Eigen::VectorXd manualRecoveredForce = world->getExternalForces();

      if (!equals(recoveredPos, manualRecoveredPos, threshold)
          || !equals(recoveredVel, manualRecoveredVel, threshold)
          || !equals(recoveredForce, manualRecoveredForce, threshold))
      {
        std::cout << "verifyChangeRepresentationToIK() failed to be lossloss "
                     "in the out mapping "
                     "when shouldBeLosslessOut=true"
                  << std::endl;
        std::cout << "Step " << i << ":" << std::endl;
        if (!equals(recoveredPos, manualRecoveredPos, threshold))
        {
          std::cout << "Recovered pos:" << std::endl
                    << recoveredPos << std::endl;
          std::cout << "Manually recovered pos:" << std::endl
                    << manualRecoveredPos << std::endl;
        }
        if (!equals(recoveredVel, manualRecoveredVel, threshold))
        {
          std::cout << "Recovered vel:" << std::endl
                    << recoveredVel << std::endl;
          std::cout << "Manually recovered vel:" << std::endl
                    << manualRecoveredVel << std::endl;
        }
        if (!equals(recoveredForce, manualRecoveredForce, threshold))
        {
          std::cout << "Recovered force:" << std::endl
                    << recoveredForce << std::endl;
          std::cout << "Manually recovered force:" << std::endl
                    << manualRecoveredForce << std::endl;
        }
        return false;
      }
    }
  }

  if (shouldBeLosslessInto && shouldBeLosslessOut)
  {
    if (!equals(
            initialIdentityRollout.getPoses("identity"),
            recoveredIdentityRollout.getPoses("identity"),
            threshold)
        || !equals(
            initialIdentityRollout.getVels("identity"),
            recoveredIdentityRollout.getVels("identity"),
            threshold)
        || !equals(
            initialIdentityRollout.getForces("identity"),
            recoveredIdentityRollout.getForces("identity"),
            threshold))
    {
      std::cout << "verifyChangeRepresentationToIK() failed to be lossloss "
                   "when shouldBeLosslessInto=true and shouldBeLosslessOut=true"
                << std::endl;
      return false;
    }
  }

  return true;
}

bool verifyMultiShotOptimization(WorldPtr world, MultiShot shot)
{
  IPOptOptimizer optimizer = IPOptOptimizer();

  optimizer.setIterationLimit(1);
  std::shared_ptr<Solution> record = optimizer.optimize(&shot);
  EXPECT_TRUE(record->getNumSteps() == 2);
  EXPECT_TRUE(record->getStep(0).index == 0);
  EXPECT_TRUE(record->getStep(1).index == 1);
  EXPECT_TRUE(record->getStep(1).rollout != record->getStep(0).rollout);

  // Playback the trajectory

  TrajectoryRolloutReal withKnots = TrajectoryRolloutReal(&shot);
  TrajectoryRolloutReal withoutKnots = TrajectoryRolloutReal(&shot);

  // Get the version with knots
  shot.getStates(world, &withKnots, nullptr, true);
  // Get the version without knots next, so that they can play in a loop
  shot.getStates(world, &withoutKnots, nullptr, false);

  // Create a window for rendering the world and handling user input
  // dart::gui::glut::displayTrajectoryInGUI(world, &shot);

  return true;
}

#endif