#include "dart/neural/BackpropSnapshot.hpp"

#include <iostream>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

//==============================================================================
BackpropSnapshot::BackpropSnapshot(
    WorldPtr world,
    Eigen::VectorXd forwardPassPosition,
    Eigen::VectorXd forwardPassVelocity,
    Eigen::VectorXd forwardPassTorques)
{
  mTimeStep = world->getTimeStep();
  mForwardPassPosition = forwardPassPosition;
  mForwardPassVelocity = forwardPassVelocity;
  mForwardPassTorques = forwardPassTorques;
  mNumDOFs = 0;
  mNumConstraintDim = 0;
  mNumClamping = 0;
  mNumUpperBound = 0;
  mNumBouncing = 0;

  // Collect all the constraint groups attached to each skeleton

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    mSkeletonOffset[skel->getName()] = mNumDOFs;
    mNumDOFs += skel->getNumDofs();

    std::shared_ptr<ConstrainedGroupGradientMatrices> gradientMatrix
        = skel->getGradientConstraintMatrices();
    if (gradientMatrix
        && std::find(
               mGradientMatrices.begin(),
               mGradientMatrices.end(),
               gradientMatrix)
               == mGradientMatrices.end())
    {
      mGradientMatrices.push_back(gradientMatrix);
      mNumConstraintDim += gradientMatrix->getNumConstraintDim();
      mNumClamping += gradientMatrix->getClampingConstraintMatrix().cols();
      mNumUpperBound += gradientMatrix->getUpperBoundConstraintMatrix().cols();
      mNumBouncing += gradientMatrix->getBouncingConstraintMatrix().cols();
    }
  }
}

//==============================================================================
void BackpropSnapshot::backprop(
    WorldPtr world,
    LossGradient& thisTimestepLoss,
    const LossGradient& nextTimestepLoss)
{
  LossGradient groupThisTimestepLoss;
  LossGradient groupNextTimestepLoss;

  // Set the state of the world back to what it was during the forward pass, so
  // that implicit mass matrix computations work correctly.

  Eigen::VectorXd oldPositions = world->getPositions();
  Eigen::VectorXd oldVelocities = world->getVelocities();
  world->setPositions(mForwardPassPosition);
  world->setVelocities(mForwardPassVelocity);

  // Create the vectors for this timestep

  thisTimestepLoss.lossWrtPosition = Eigen::VectorXd(mNumDOFs);
  thisTimestepLoss.lossWrtVelocity = Eigen::VectorXd(mNumDOFs);
  thisTimestepLoss.lossWrtTorque = Eigen::VectorXd(mNumDOFs);

  // Actually run the backprop

  std::unordered_map<std::string, bool> skeletonsVisited;

  for (std::shared_ptr<ConstrainedGroupGradientMatrices> group :
       mGradientMatrices)
  {
    std::size_t groupDofs = group->getNumDOFs();

    // Instantiate the vectors with plenty of DOFs

    groupNextTimestepLoss.lossWrtPosition = Eigen::VectorXd(groupDofs);
    groupNextTimestepLoss.lossWrtVelocity = Eigen::VectorXd(groupDofs);
    groupThisTimestepLoss.lossWrtPosition = Eigen::VectorXd(groupDofs);
    groupThisTimestepLoss.lossWrtVelocity = Eigen::VectorXd(groupDofs);
    groupThisTimestepLoss.lossWrtTorque = Eigen::VectorXd(groupDofs);

    // Set up next timestep loss as a map of the real values

    std::size_t cursor = 0;
    for (std::size_t j = 0; j < group->getSkeletons().size(); j++)
    {
      SkeletonPtr skel = world->getSkeleton(group->getSkeletons()[j]);
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];
      std::size_t dofs = skel->getNumDofs();

      // Keep track of which skeletons have been covered by constraint groups
      bool skelAlreadyVisited
          = (skeletonsVisited.find(skel->getName()) != skeletonsVisited.end());
      assert(!skelAlreadyVisited);
      skeletonsVisited[skel->getName()] = true;

      groupNextTimestepLoss.lossWrtPosition.segment(cursor, dofs)
          = nextTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs);
      groupNextTimestepLoss.lossWrtVelocity.segment(cursor, dofs)
          = nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs);

      cursor += dofs;
    }

    // Now actually run the backprop

    group->backprop(world, groupThisTimestepLoss, groupNextTimestepLoss);

    // Read the values back out of the group backprop

    cursor = 0;
    for (std::size_t j = 0; j < group->getSkeletons().size(); j++)
    {
      SkeletonPtr skel = world->getSkeleton(group->getSkeletons()[j]);
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];
      std::size_t dofs = skel->getNumDofs();

      thisTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs)
          = groupThisTimestepLoss.lossWrtPosition.segment(cursor, dofs);
      thisTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
          = groupThisTimestepLoss.lossWrtVelocity.segment(cursor, dofs);
      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
          = groupThisTimestepLoss.lossWrtTorque.segment(cursor, dofs);

      cursor += dofs;
    }
  }

  // We need to go through and manually cover any skeletons that aren't covered
  // by any constraint group (because they have no active constraints). Because
  // these skeletons aren't part of a constrained group, their Jacobians are
  // quite simple.

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    bool skelAlreadyVisited
        = (skeletonsVisited.find(skel->getName()) != skeletonsVisited.end());
    if (!skelAlreadyVisited)
    {
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];
      std::size_t dofs = skel->getNumDofs();
      /*

      The forward computation graph looks like this:

      -----------> p_t ---------> p_t+1 ---->
                           ^
                           |
      v_t -----------------+----> v_t+1 ---->
                                    ^
                                    |
      f_t --------------------------+

      */

      // p_t --> p_t+1
      // pos-pos = I
      thisTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs)
          = nextTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs);

      // v_t --> p_t+1
      // vel-pos = timeStep * I
      thisTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
          = mTimeStep
            * nextTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs);

      // v_t --> v_t+1
      // vel-vel = I
      thisTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
          += nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs);

      // f_t --> v_t+1
      // force-vel = timeStep * Minv
      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
          = skel->multiplyByImplicitInvMassMatrix(
              nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs));
      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs) *= mTimeStep;
    }
  }

  // Restore the old position and velocity values before we ran backprop
  world->setPositions(oldPositions);
  world->setVelocities(oldVelocities);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getForceVelJacobian(WorldPtr world)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd Minv = getInvMassMatrix(world);

  // If there are no clamping constraints, then force-vel is just the mTimeStep
  // * Minv
  if (A_c.size() == 0)
    return mTimeStep * Minv;

  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix(world);

  if (A_ub.size() > 0 && E.size() > 0)
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
  }
  else
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * A_c * P_c * Minv);
  }
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelVelJacobian(WorldPtr world)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);

  // If there are no clamping constraints, then vel-vel is just the identity
  if (A_c.size() == 0)
    return Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs);

  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix(world);
  Eigen::MatrixXd Minv = getInvMassMatrix(world);
  Eigen::MatrixXd parts1 = A_c + A_ub * E;
  Eigen::MatrixXd parts2 = mTimeStep * Minv * parts1 * P_c;
  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  std::cout << "P_c: " << std::endl << P_c << std::endl;
  std::cout << "Minv: " << std::endl << Minv << std::endl;
  std::cout << "mTimestep: " << mTimeStep << std::endl;
  std::cout << "A_c + A_ub * E: " << std::endl << parts1 << std::endl;
  std::cout << "mTimestep * Minv * (A_c + A_ub * E) * P_c: " << std::endl
            << parts2 << std::endl;
  */
  return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) - parts2);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getPosPosJacobian(WorldPtr world)
{
  Eigen::MatrixXd A_b = getBouncingConstraintMatrix(world);

  // If there are no bounces, pos-pos is a simple identity
  if (A_b.size() == 0)
    return Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs);

  // Construct the W matrix we'll need to use to solve for our closest approx
  Eigen::MatrixXd W = Eigen::MatrixXd(A_b.cols(), A_b.rows() * A_b.rows());
  for (int i = 0; i < A_b.cols(); i++)
  {
    Eigen::VectorXd a_i = A_b.col(i);
    for (int j = 0; j < A_b.rows(); j++)
    {
      W.block(j * A_b.rows(), i, A_b.rows(), 1) = a_i(j) * a_i;
    }
  }

  // We want to center the solution around the identity matrix, and find the
  // least-squares deviation along the diagonals that gets us there.
  Eigen::VectorXd center = Eigen::VectorXd::Zero(mNumDOFs * mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    center((i * mNumDOFs) + i) = 1;
  }

  // Solve the linear system
  Eigen::VectorXd q
      = center
        - W.transpose().completeOrthogonalDecomposition().solve(
            getRestitutionDiagonals() + (W.eval().transpose() * center));

  // Recover X from the q vector
  Eigen::MatrixXd X = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
  }

  return X;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelPosJacobian(WorldPtr world)
{
  return mTimeStep * getPosPosJacobian(world);
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassPosition()
{
  return mForwardPassPosition;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassVelocity()
{
  return mForwardPassVelocity;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassTorques()
{
  return mForwardPassTorques;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getClampingConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::CLAMPING);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassedClampingConstraintMatrix(
    WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::MASSED_CLAMPING);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getUpperBoundConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::UPPER_BOUND);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassedUpperBoundConstraintMatrix(
    WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::MASSED_UPPER_BOUND);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getUpperBoundMappingMatrix()
{
  std::size_t numUpperBound = 0;
  std::size_t numClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    numUpperBound
        += mGradientMatrices[i]->getUpperBoundConstraintMatrix().cols();
    numClamping += mGradientMatrices[i]->getClampingConstraintMatrix().cols();
  }

  Eigen::MatrixXd mappingMatrix = Eigen::MatrixXd(numUpperBound, numClamping);
  mappingMatrix.setZero();

  std::size_t cursorUpperBound = 0;
  std::size_t cursorClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupMappingMatrix
        = mGradientMatrices[i]->getUpperBoundMappingMatrix();
    mappingMatrix.block(
        cursorUpperBound,
        cursorClamping,
        groupMappingMatrix.rows(),
        groupMappingMatrix.cols())
        = groupMappingMatrix;

    cursorUpperBound += groupMappingMatrix.rows();
    cursorClamping += groupMappingMatrix.cols();
  }

  return mappingMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getBouncingConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::BOUNCING);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassMatrix(WorldPtr world)
{
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  massMatrix.setZero();

  // Set the state of the world back to what it was during the forward pass, so
  // that implicit mass matrix computations work correctly.

  Eigen::VectorXd oldPositions = world->getPositions();
  Eigen::VectorXd oldVelocities = world->getVelocities();
  world->setPositions(mForwardPassPosition);
  world->setVelocities(mForwardPassVelocity);

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::size_t skelDOF = world->getSkeleton(i)->getNumDofs();
    massMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = world->getSkeleton(i)->getMassMatrix();
    cursor += skelDOF;
  }

  // Reset the position of the world to what it was before

  world->setPositions(oldPositions);
  world->setVelocities(oldVelocities);

  return massMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getInvMassMatrix(WorldPtr world)
{
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  invMassMatrix.setZero();

  // Set the state of the world back to what it was during the forward pass, so
  // that implicit mass matrix computations work correctly.

  Eigen::VectorXd oldPositions = world->getPositions();
  Eigen::VectorXd oldVelocities = world->getVelocities();
  world->setPositions(mForwardPassPosition);
  world->setVelocities(mForwardPassVelocity);

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::size_t skelDOF = world->getSkeleton(i)->getNumDofs();
    invMassMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = world->getSkeleton(i)->getInvMassMatrix();
    cursor += skelDOF;
  }

  // Reset the position of the world to what it was before

  world->setPositions(oldPositions);
  world->setVelocities(oldVelocities);

  return invMassMatrix;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getContactConstraintImpluses()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::CONTACT_CONSTRAINT_IMPULSES);
}

//==============================================================================
Eigen::VectorXi BackpropSnapshot::getContactConstraintMappings()
{
  return assembleVector<Eigen::VectorXi>(
      VectorToAssemble::CONTACT_CONSTRAINT_MAPPINGS);
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getBounceDiagonals()
{
  return assembleVector<Eigen::VectorXd>(VectorToAssemble::BOUNCE_DIAGONALS);
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getRestitutionDiagonals()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::RESTITUTION_DIAGONALS);
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPenetrationCorrectionVelocities()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::PENETRATION_VELOCITY_HACK);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelVelJacobian(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);

  world->setVelocities(mForwardPassVelocity);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd tweakedVel = Eigen::VectorXd(mForwardPassVelocity);
    tweakedVel(i) += EPSILON;
    world->setVelocities(tweakedVel);
    world->step(false);

    Eigen::VectorXd velChange
        = (world->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceForceVelJacobian(
    WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);

  world->setVelocities(mForwardPassVelocity);
  world->step(false);

  Eigen::VectorXd originalForces = world->getForces();
  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setVelocities(mForwardPassVelocity);
    Eigen::VectorXd tweakedForces = Eigen::VectorXd(originalForces);
    tweakedForces(i) += EPSILON;
    world->setForces(tweakedForces);

    world->step(false);

    Eigen::VectorXd velChange
        = (world->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferencePosPosJacobian(
    WorldPtr world, std::size_t subdivisions)
{
  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  world->setPositions(mForwardPassPosition);
  world->setVelocities(mForwardPassVelocity);
  world->setForces(mForwardPassTorques);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = world->getPositions();

  // IMPORTANT: EPSILON must be larger than the distance traveled in a single
  // subdivided timestep. Ideally much larger.
  double EPSILON = 1e-1 / subdivisions;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setVelocities(mForwardPassVelocity);
    world->setForces(mForwardPassTorques);

    Eigen::VectorXd tweakedPositions = Eigen::VectorXd(mForwardPassPosition);
    tweakedPositions(i) += EPSILON;
    world->setPositions(tweakedPositions);

    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd posChange
        = (world->getPositions() - originalPosition) / EPSILON;
    J.col(i).noalias() = posChange;
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelPosJacobian(
    WorldPtr world, std::size_t subdivisions)
{
  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  world->setPositions(mForwardPassPosition);
  world->setVelocities(mForwardPassVelocity);
  world->setForces(mForwardPassTorques);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = world->getPositions();

  double EPSILON = 1e-3 / subdivisions;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setPositions(mForwardPassPosition);
    world->setForces(mForwardPassTorques);

    Eigen::VectorXd tweakedVelocity = Eigen::VectorXd(mForwardPassVelocity);
    tweakedVelocity(i) += EPSILON;
    world->setVelocities(tweakedVelocity);

    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd posChange
        = (world->getPositions() - originalPosition) / EPSILON;
    J.col(i).noalias() = posChange;
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getProjectionIntoClampsMatrix(WorldPtr world)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  */

  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.size() > 0
            ? forceToVel.completeOrthogonalDecomposition().pseudoInverse()
            : Eigen::MatrixXd(0, 0);
  Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();
  /*
  std::cout << "forceToVel: " << std::endl << forceToVel << std::endl;
  std::cout << "forceToVel^-1: " << std::endl << velToForce << std::endl;
  std::cout << "mTimeStep: " << mTimeStep << std::endl;
  */
  return (1.0 / mTimeStep) * velToForce * bounce * A_c.transpose();
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::assembleMatrix(
    WorldPtr world, MatrixToAssemble whichMatrix)
{
  std::size_t numCols = 0;
  if (whichMatrix == MatrixToAssemble::CLAMPING
      || whichMatrix == MatrixToAssemble::MASSED_CLAMPING)
    numCols = mNumClamping;
  else if (
      whichMatrix == MatrixToAssemble::UPPER_BOUND
      || whichMatrix == MatrixToAssemble::MASSED_UPPER_BOUND)
    numCols = mNumUpperBound;
  else if (whichMatrix == MatrixToAssemble::BOUNCING)
    numCols = mNumBouncing;

  Eigen::MatrixXd matrix = Eigen::MatrixXd(mNumDOFs, numCols);
  matrix.setZero();
  std::size_t constraintCursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupMatrix;

    if (whichMatrix == MatrixToAssemble::CLAMPING)
      groupMatrix = mGradientMatrices[i]->getClampingConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::MASSED_CLAMPING)
      groupMatrix = mGradientMatrices[i]->getMassedClampingConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::UPPER_BOUND)
      groupMatrix = mGradientMatrices[i]->getUpperBoundConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::MASSED_UPPER_BOUND)
      groupMatrix = mGradientMatrices[i]->getMassedUpperBoundConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::BOUNCING)
      groupMatrix = mGradientMatrices[i]->getBouncingConstraintMatrix();

    // shuffle the clamps into the main matrix
    std::size_t dofCursorGroup = 0;
    for (std::size_t k = 0; k < mGradientMatrices[i]->getSkeletons().size();
         k++)
    {
      SkeletonPtr skel
          = world->getSkeleton(mGradientMatrices[i]->getSkeletons()[k]);
      // This maps to the row in the world matrix
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];

      // The source block in the groupClamps matrix is a row section at
      // (dofCursorGroup, 0) of full width (skel->getNumDOFs(),
      // groupClamps.cols()), which we want to copy into our unified
      // clampingConstraintMatrix.

      // The destination block in clampingConstraintMatrix is the column
      // corresponding to this constraint group's constraint set, and the row
      // corresponding to this skeleton's offset into the world at
      // (dofCursorWorld, constraintCursor).

      matrix.block(
          dofCursorWorld,
          constraintCursor,
          skel->getNumDofs(),
          groupMatrix.cols())
          = groupMatrix.block(
              dofCursorGroup, 0, skel->getNumDofs(), groupMatrix.cols());

      dofCursorGroup += skel->getNumDofs();
    }

    constraintCursor += groupMatrix.cols();
  }
  return matrix;
}

//==============================================================================
template <typename Vec>
Vec BackpropSnapshot::assembleVector(VectorToAssemble whichVector)
{
  if (mGradientMatrices.size() == 1)
  {
    return getVectorToAssemble<Vec>(mGradientMatrices[0], whichVector);
  }

  std::size_t size = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    // BOUNCE_DIAGONALS: bounce size is number of clamping contacts for each
    // group RESTITUTION_DIAGONALS: bounce size is number of bouncing contacts
    // (which is usually less than the number of clamping contacts) for each
    // group CONTACT_CONSTRAINT_IMPULSES: This is the total number of contacts,
    // including non-clamping ones CONTACT_CONSTRAINT_MAPPINGS: This is the
    // total number of contacts, including non-clamping ones
    size += getVectorToAssemble<Vec>(mGradientMatrices[0], whichVector).size();
  }

  Vec collected = Vec(size);

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    const Vec& vec
        = getVectorToAssemble<Vec>(mGradientMatrices[i], whichVector);
    collected.segment(cursor, vec.size()) = vec;
    cursor += vec.size();
  }
  return collected;
}

//==============================================================================
template <>
const Eigen::VectorXd& BackpropSnapshot::getVectorToAssemble(
    std::shared_ptr<ConstrainedGroupGradientMatrices> matrices,
    VectorToAssemble whichVector)
{
  if (whichVector == VectorToAssemble::BOUNCE_DIAGONALS)
    return matrices->getBounceDiagonals();
  if (whichVector == VectorToAssemble::RESTITUTION_DIAGONALS)
    return matrices->getRestitutionDiagonals();
  if (whichVector == VectorToAssemble::CONTACT_CONSTRAINT_IMPULSES)
    return matrices->getContactConstraintImpluses();
  if (whichVector == VectorToAssemble::PENETRATION_VELOCITY_HACK)
    return matrices->getPenetrationCorrectionVelocities();

  assert(whichVector != VectorToAssemble::CONTACT_CONSTRAINT_MAPPINGS);
  // Control will never reach this point, but this removes a warning
  throw 1;
}

template <>
const Eigen::VectorXi& BackpropSnapshot::getVectorToAssemble(
    std::shared_ptr<ConstrainedGroupGradientMatrices> matrices,
    VectorToAssemble whichVector)
{
  assert(whichVector == VectorToAssemble::CONTACT_CONSTRAINT_MAPPINGS);
  return matrices->getContactConstraintMappings();
}

} // namespace neural
} // namespace dart