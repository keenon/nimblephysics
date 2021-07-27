#include "dart/neural/BackpropSnapshot.hpp"

#include <array>
#include <chrono>
#include <iostream>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

#define LOG_PERFORMANCE_BACKPROP_SNAPSHOT

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace performance;

namespace dart {
namespace neural {

//==============================================================================
BackpropSnapshot::BackpropSnapshot(
    WorldPtr world,
    Eigen::VectorXs preStepPosition,
    Eigen::VectorXs preStepVelocity,
    Eigen::VectorXs preStepTorques,
    Eigen::VectorXs preConstraintVelocities,
    Eigen::VectorXs preStepLCPCache)
  : mUseFDOverride(world->getUseFDOverride()),
    mSlowDebugResultsAgainstFD(world->getSlowDebugResultsAgainstFD()),
    mNumDOFs(0),
    mNumConstraintDim(0),
    mNumClamping(0),
    mNumUpperBound(0),
    mNumBouncing(0)
{
  mTimeStep = world->getTimeStep();
  mPreStepPosition = preStepPosition;
  mPreStepVelocity = preStepVelocity;
  mPreStepTorques = preStepTorques;
  mPreConstraintVelocities = preConstraintVelocities;
  mPreStepLCPCache = preStepLCPCache;
  mPostStepPosition = world->getPositions();
  mPostStepVelocity = world->getVelocities();
  mPostStepTorques = world->getControlForces();

  // Reset the world to the initial state before finalizing all the gradient
  // matrices

  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  // Collect all the constraint groups attached to each skeleton

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    mSkeletonOffset[skel->getName()] = mNumDOFs;
    mSkeletonDofs[skel->getName()] = skel->getNumDofs();
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
      // Finalize the construction of the matrices
      mGradientMatrices.push_back(gradientMatrix);
      mNumConstraintDim += gradientMatrix->getNumConstraintDim();
      mNumClamping += gradientMatrix->getClampingConstraintMatrix().cols();
      mNumUpperBound += gradientMatrix->getUpperBoundConstraintMatrix().cols();
      mNumBouncing += gradientMatrix->getBouncingConstraintMatrix().cols();
    }
  }

  // snapshot.restore();

  mCachedPosPosDirty = true;
  mCachedVelPosDirty = true;
  mCachedBounceApproximationDirty = true;
  mCachedPosVelDirty = true;
  mCachedVelVelDirty = true;
  mCachedForcePosDirty = true;
  mCachedForceVelDirty = true;
  mCachedMassVelDirty = true;
  mCachedVelCDirty = true;
  mCachedPosCDirty = true;

  /*
  if (!areResultsStandardized())
  {
    std::cout << "Found an example which was impossible to standardize!"
              << std::endl;
    printReplicationInstructions(world);
    assert(false);
  }
  */
}

//==============================================================================
void BackpropSnapshot::backprop(
    WorldPtr world,
    LossGradient& thisTimestepLoss,
    const LossGradient& nextTimestepLoss,
    PerformanceLog* perfLog,
    bool exploreAlternateStrategies)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.backprop");
  }
#endif

  LossGradient groupThisTimestepLoss;
  LossGradient groupNextTimestepLoss;

  // Set the state of the world back to what it was during the forward pass, so
  // that implicit mass matrix computations work correctly.

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  // Create the vectors for this timestep

  thisTimestepLoss.lossWrtPosition = Eigen::VectorXs::Zero(mNumDOFs);
  thisTimestepLoss.lossWrtVelocity = Eigen::VectorXs::Zero(mNumDOFs);
  thisTimestepLoss.lossWrtTorque = Eigen::VectorXs::Zero(mNumDOFs);
  thisTimestepLoss.lossWrtMass = Eigen::VectorXs::Zero(world->getMassDims());

  // TODO(opt): remove me, as soon as it's faster to construct the Jacobians
  // using ConstrainedGroups directly. Currently it's redundant to construct
  // Jacobians _both_ in the ConstrainedGroups and in the BackpropSnapshot, so
  // it's better overall to just use one.
  if (exploreAlternateStrategies == false)
  {
    const Eigen::MatrixXs& posPos = getPosPosJacobian(world, thisLog);
    const Eigen::MatrixXs& posVel = getPosVelJacobian(world, thisLog);
    const Eigen::MatrixXs& velPos = getVelPosJacobian(world, thisLog);
    const Eigen::MatrixXs& velVel = getVelVelJacobian(world, thisLog);
    const Eigen::MatrixXs& forceVel
        = getControlForceVelJacobian(world, thisLog);
    const Eigen::MatrixXs& massVel = getMassVelJacobian(world, thisLog);

    thisTimestepLoss.lossWrtPosition
        = posPos.transpose() * nextTimestepLoss.lossWrtPosition
          + posVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtVelocity
        = velPos.transpose() * nextTimestepLoss.lossWrtPosition
          + velVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtTorque
        = forceVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtMass
        = massVel.transpose() * nextTimestepLoss.lossWrtVelocity;

    clipLossGradientsToBounds(
        world,
        thisTimestepLoss.lossWrtPosition,
        thisTimestepLoss.lossWrtVelocity,
        thisTimestepLoss.lossWrtTorque);

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      thisLog->end();
    }
#endif
    snapshot.restore();
    return;
  }

  //////////////////////////////////////////////////////////
  // Exploring alternate strategies requires breaking down into the individual
  // constrained groups and doing backprop there
  //////////////////////////////////////////////////////////

  // Handle mass the old fashioned way, for now

  const Eigen::MatrixXs& massVel = getMassVelJacobian(world, thisLog);
  thisTimestepLoss.lossWrtMass
      = massVel.transpose() * nextTimestepLoss.lossWrtVelocity;

  // Actually run the backprop

  std::unordered_map<std::string, bool> skeletonsVisited;

  for (std::shared_ptr<ConstrainedGroupGradientMatrices> group :
       mGradientMatrices)
  {
    std::size_t groupDofs = group->getNumDOFs();

    // Instantiate the vectors with plenty of DOFs

    groupNextTimestepLoss.lossWrtPosition = Eigen::VectorXs::Zero(groupDofs);
    groupNextTimestepLoss.lossWrtVelocity = Eigen::VectorXs::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtPosition = Eigen::VectorXs::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtVelocity = Eigen::VectorXs::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtTorque = Eigen::VectorXs::Zero(groupDofs);

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
      DART_UNUSED(skelAlreadyVisited);
      assert(!skelAlreadyVisited);
      skeletonsVisited[skel->getName()] = true;

      groupNextTimestepLoss.lossWrtPosition.segment(cursor, dofs)
          = nextTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs);
      groupNextTimestepLoss.lossWrtVelocity.segment(cursor, dofs)
          = nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs);

      cursor += dofs;
    }

    // Now actually run the backprop

    group->backprop(
        world,
        groupThisTimestepLoss,
        groupNextTimestepLoss,
        exploreAlternateStrategies);

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
    if (!skelAlreadyVisited && skel->isMobile())
    {
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];
      std::size_t dofs = skel->getNumDofs();

      /////////////////////////////////////////////////////////////////
      // Explicitly form the matrices
      /////////////////////////////////////////////////////////////////

      Eigen::MatrixXs Minv = skel->getInvMassMatrix();

      Eigen::MatrixXs forceVel = mTimeStep * Minv;
      Eigen::MatrixXs velVel
          = Eigen::MatrixXs::Identity(skel->getNumDofs(), skel->getNumDofs())
            - mTimeStep * Minv * skel->getVelCJacobian();
      Eigen::MatrixXs posVel = skel->getUnconstrainedVelJacobianWrt(
          world->getTimeStep(), WithRespectTo::POSITION);
      Eigen::MatrixXs posPos
          = Eigen::MatrixXs::Identity(skel->getNumDofs(), skel->getNumDofs());
      Eigen::MatrixXs velPos
          = mTimeStep
            * Eigen::MatrixXs::Identity(skel->getNumDofs(), skel->getNumDofs());

      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
          = forceVel.transpose()
            * nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs);
      thisTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
          = velVel.transpose()
                * nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
            + velPos.transpose()
                  * nextTimestepLoss.lossWrtPosition.segment(
                      dofCursorWorld, dofs);
      thisTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs)
          = posVel.transpose()
                * nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
            + posPos.transpose()
                  * nextTimestepLoss.lossWrtPosition.segment(
                      dofCursorWorld, dofs);

      /*

      // f_t
      // force-vel = dT * Minv
      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
          // f_t --> v_t+1
          = mTimeStep
            * skel->getInvMassMatrix() // This is symmetric, but otherwise
                                       // we would need to .transpose()
            * nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs);

      // skel->getJacobionOfMinv();
      // getUnconstrainedVelJacobianWrt
      // p_t
      // pos-pos = I
      // pos-vel = dT * Minv * d/dpos C(pos,vel) + dT * d/dpos Minv * C(pos,
      // vel) pos-vel^T = dT * d/dpos C(pos,vel)^T * Minv
      thisTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs)
          // p_t --> p_t+1
          = nextTimestepLoss.lossWrtPosition.segment(dofCursorWorld, dofs)
            // p_t --> v_t+1
            + (skel->getUnconstrainedVelJacobianWrt(
                       world->getTimeStep(), WithRespectTo::POSITION)
                   .transpose()
               * nextTimestepLoss.lossWrtVelocity.segment(
                   dofCursorWorld, dofs));

      // v_t
      // vel-vel = I - dT * Minv * d/dvel C(pos,vel)
      // vel-pos = dT * I
      thisTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
          // v_t --> v_t+1
          = nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs)
            - mTimeStep * skel->getVelCJacobian().transpose()
                  * thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
            // v_t --> p_t+1
            + mTimeStep
                  * nextTimestepLoss.lossWrtPosition.segment(
                      dofCursorWorld, dofs);
      */
    }
  }

  // Restore the old position and velocity values before we ran backprop
  snapshot.restore();

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes backprop in the high-level RL API's space, use `state` and
/// `action` as the primitives we're taking gradients wrt to.
LossGradientHighLevelAPI BackpropSnapshot::backpropState(
    simulation::WorldPtr world,
    const Eigen::VectorXs& nextTimestepStateLossGrad,
    PerformanceLog* perfLog,
    bool exploreAlternateStrategies)
{
  LossGradient thisTimestepLoss;
  LossGradient nextTimestepLoss;
  int dofs = world->getNumDofs();
  nextTimestepLoss.lossWrtPosition = nextTimestepStateLossGrad.head(dofs);
  nextTimestepLoss.lossWrtVelocity = nextTimestepStateLossGrad.tail(dofs);
  backprop(
      world,
      thisTimestepLoss,
      nextTimestepLoss,
      perfLog,
      exploreAlternateStrategies);

  LossGradientHighLevelAPI grad;
  grad.lossWrtState = Eigen::VectorXs(dofs * 2);
  grad.lossWrtState.head(dofs) = thisTimestepLoss.lossWrtPosition;
  grad.lossWrtState.tail(dofs) = thisTimestepLoss.lossWrtVelocity;
  grad.lossWrtAction = Eigen::VectorXs::Zero(world->getActionSize());
  std::vector<int> actionMapping = world->getActionSpace();
  for (int i = 0; i < actionMapping.size(); i++)
  {
    if (actionMapping[i] < 0 || actionMapping[i] >= dofs)
    {
      std::cerr << "neural::BackpropSnapshot::backpropState() discovered an "
                   "out-of-bounds element in the action state mapping. Element "
                << i << " -> " << actionMapping[i] << " is out of bounds [0,"
                << dofs << "). Ignoring." << std::endl;
      continue;
    }
    grad.lossWrtAction(i) = thisTimestepLoss.lossWrtTorque(actionMapping[i]);
  }
  grad.lossWrtMass = thisTimestepLoss.lossWrtMass;
  return grad;
}

//==============================================================================
/// This zeros out any components of the gradient that would want to push us
/// out of the box-bounds encoded in the world for pos, vel, or force.
void BackpropSnapshot::clipLossGradientsToBounds(
    simulation::WorldPtr world,
    Eigen::VectorXs& lossWrtPos,
    Eigen::VectorXs& lossWrtVel,
    Eigen::VectorXs& lossWrtForce)
{
  int cursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      // Clip position gradients

      if ((skel->getPosition(j) == skel->getPositionLowerLimit(j))
          && (lossWrtPos(cursor) > 0))
      {
        lossWrtPos(cursor) = 0;
      }
      if ((skel->getPosition(j) == skel->getPositionUpperLimit(j))
          && (lossWrtPos(cursor) < 0))
      {
        lossWrtPos(cursor) = 0;
      }

      // Clip velocity gradients

      if ((skel->getVelocity(j) == skel->getVelocityLowerLimit(j))
          && (lossWrtVel(cursor) > 0))
      {
        lossWrtVel(cursor) = 0;
      }
      if ((skel->getVelocity(j) == skel->getVelocityUpperLimit(j))
          && (lossWrtVel(cursor) < 0))
      {
        lossWrtVel(cursor) = 0;
      }

      // Clip force gradients

      if ((skel->getControlForce(j) == skel->getControlForceLowerLimit(j))
          && (lossWrtForce(cursor) > 0))
      {
        lossWrtForce(cursor) = 0;
      }
      if ((skel->getControlForce(j) == skel->getControlForceUpperLimit(j))
          && (lossWrtForce(cursor) < 0))
      {
        lossWrtForce(cursor) = 0;
      }

      cursor++;
    }
  }
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getControlForceVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getControlForceVelJacobian");
  }
#endif

  if (mCachedForceVelDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getControlForceVelJacobian#refreshCache");
    }
#endif
    if (mUseFDOverride)
    {
      mCachedForceVel = finiteDifferenceForceVelJacobian(world);
    }
    else
    {
      Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
      Eigen::MatrixXs Minv = getInvMassMatrix(world);

      // If there are no clamping constraints, then force-vel is just the
      // mTimeStep
      // * Minv
      if (A_c.size() == 0)
      {
        mCachedForceVel = mTimeStep * Minv;
      }
      else
      {
        mCachedForceVel = getVelJacobianWrt(world, WithRespectTo::FORCE);

        /*
        Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
        Eigen::MatrixXs E = getUpperBoundMappingMatrix();
        Eigen::MatrixXs P_c = getProjectionIntoClampsMatrix(world);

        if (A_ub.size() > 0 && E.size() > 0)
        {
          mCachedForceVel = mTimeStep * Minv
                            * (Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs)
                               - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
        }
        else
        {
          mCachedForceVel = mTimeStep * Minv
                            * (Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs)
                               - mTimeStep * A_c * P_c * Minv);
        }
        */
      }
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXs bruteForce = finiteDifferenceForceVelJacobian(world);
      equalsOrCrash(world, mCachedForceVel, bruteForce, "force-vel");
    }

    // mCachedForceVel = getVelJacobianWrt(world, WithRespectTo::FORCE);
    mCachedForceVelDirty = false;

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedForceVel;
}

//==============================================================================
/// This computes and returns the whole mass-vel jacobian. For backprop, you
/// don't actually need this matrix, you can compute backprop directly. This
/// is here if you want access to the full Jacobian for some reason.
const Eigen::MatrixXs& BackpropSnapshot::getMassVelJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getMassVelJacobian");
  }
#endif

  if (mCachedMassVelDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getMassVelJacobian#refreshCache");
    }
#endif

    if (mUseFDOverride)
    {
      mCachedMassVel = finiteDifferenceMassVelJacobian(world);
    }
    else
    {
      mCachedMassVel = getVelJacobianWrt(world, world->getWrtMass().get());
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXs bruteForce = finiteDifferenceMassVelJacobian(world);
      equalsOrCrash(world, mCachedMassVel, bruteForce, "mass-vel");
    }

    mCachedMassVelDirty = false;

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedMassVel;
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getVelVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getVelVelJacobian");
  }
#endif

  if (mCachedVelVelDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getVelVelJacobian#refreshCache");
    }
#endif

    if (mUseFDOverride)
    {
      mCachedVelVel = finiteDifferenceVelVelJacobian(world);
    }
    else
    {
      Eigen::VectorXs ddamp = getDampingVector(world);
      Eigen::VectorXs spring_stiffs = getSpringStiffVector(world);
      Eigen::MatrixXs Minv = getInvMassMatrix(world);
      s_t dt = world->getTimeStep();
      Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);

      // If there are no clamping constraints, then vel-vel is just the identity
      if (A_c.size() == 0)
      {
        mCachedVelVel
            = Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs) 
            - dt*Minv*ddamp.asDiagonal() - dt*dt*Minv*spring_stiffs.asDiagonal()
              - dt*Minv* getVelCJacobian(world);
      }
      else
      {
        mCachedVelVel = getVelJacobianWrt(world, WithRespectTo::VELOCITY) 
        - dt*Minv*ddamp.asDiagonal() - dt*dt*Minv*spring_stiffs.asDiagonal();
        

        /*
        Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
        Eigen::MatrixXs E = getUpperBoundMappingMatrix();
        Eigen::MatrixXs P_c = getProjectionIntoClampsMatrix(world);
        Eigen::MatrixXs Minv = getInvMassMatrix(world);
        Eigen::MatrixXs dF_c
            = getJacobianOfConstraintForce(world, WithRespectTo::VELOCITY);
        Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;
        Eigen::MatrixXs parts2 = Minv * A_c_ub_E * dF_c;

        mCachedVelVel = (Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs) + parts2)
                        - getControlForceVelJacobian(world) *
        getVelCJacobian(world);
        */

        /*
        std::cout << "A_c: " << std::endl << A_c << std::endl;
        std::cout << "A_ub: " << std::endl << A_ub << std::endl;
        std::cout << "E: " << std::endl << E << std::endl;
        std::cout << "P_c: " << std::endl << P_c << std::endl;
        std::cout << "Minv: " << std::endl << Minv << std::endl;
        std::cout << "mTimestep: " << mTimeStep << std::endl;
        std::cout << "A_c + A_ub * E: " << std::endl << parts1 << std::endl;
        */
        /*
         std::cout << "Vel-vel construction pieces: " << std::endl;
         std::cout << "1: I " << std::endl
                   << Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs) <<
         std::endl; std::cout << "2: - mTimestep * Minv * (A_c + A_ub * E) *
         P_c" << std::endl
                   << -parts2 << std::endl;
         std::cout << "2.5: velC" << std::endl << getVelCJacobian(world) <<
         std::endl; std::cout << "3: - forceVel * velC" << std::endl
                   << -getControlForceVelJacobian(world) *
         getVelCJacobian(world)
                   << std::endl;
         */
      }
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXs bruteForce = finiteDifferenceVelVelJacobian(world);
      equalsOrCrash(world, mCachedVelVel, bruteForce, "vel-vel");
    }

    mCachedVelVelDirty = false;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedVelVel;
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getPosVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getPosVelJacobian");
  }
#endif

  if (mCachedPosVelDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getPosVelJacobian#refreshCache");
    }
#endif

    if (mUseFDOverride)
    {
      mCachedPosVel = finiteDifferencePosVelJacobian(world);
    }
    else
    {
      mCachedPosVel = getVelJacobianWrt(world, WithRespectTo::POSITION);
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXs bruteForce = finiteDifferencePosVelJacobian(world);
      equalsOrCrash(world, mCachedPosVel, bruteForce, "pos-vel");
    }

    mCachedPosVelDirty = false;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedPosVel;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getAnalyticalNextV(
    simulation::WorldPtr world, bool morePreciseButSlower)
{
  Eigen::MatrixXs A_c
      = morePreciseButSlower
            ? getClampingConstraintMatrixAt(world, world->getPositions())
            : estimateClampingConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXs A_ub
      = morePreciseButSlower
            ? getUpperBoundConstraintMatrixAt(world, world->getPositions())
            : estimateUpperBoundConstraintMatrixAt(
                world, world->getPositions());
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXs Minv = world->getInvMassMatrix();
  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  s_t dt = world->getTimeStep();
  Eigen::VectorXs f_c = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

  Eigen::VectorXs preSolveV = mPreStepVelocity + dt * Minv * (tau - C);
  Eigen::VectorXs f_cDeltaV = Minv * A_c_ub_E * f_c;
  Eigen::VectorXs postSolveV = preSolveV + f_cDeltaV;
  return postSolveV;

  /*
  Eigen::VectorXs innerV = world->getVelocities() + dt * Minv * (tau - C);

  return world->getVelocities()
         + dt * Minv * (tau - C - A_c_ub_E * P_c * innerV);
  */
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getScratchAnalytical(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXs f_c = getClampingConstraintImpulses();
  s_t dt = world->getTimeStep();

  Eigen::MatrixXs dM
      = getJacobianOfMinv(world, dt * (tau - C) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXs Minv = world->getInvMassMatrix();
  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);

  Eigen::MatrixXs dF_c = getJacobianOfConstraintForce(world, wrt);

  Eigen::MatrixXs Q = A_c.transpose() * Minv * A_c_ub_E;

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXs dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

  snapshot.restore();
  /*
  return dB;
  // dQ_b is 0, so don't compute it
  return Qfac.solve(dB);
  return dF_c;
  */

  return Minv * (A_c * dF_c - dt * dC);
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::scratch(simulation::WorldPtr world)
{
  /////////////////////////////////////////////////////////////////////////
  // Compute NextV
  /////////////////////////////////////////////////////////////////////////

  Eigen::MatrixXs A_c
      = estimateClampingConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXs A_ub
      = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXs Minv = world->getInvMassMatrix();

  Eigen::MatrixXs Q = A_c.transpose() * Minv * A_c_ub_E;

  Eigen::VectorXs b = Eigen::VectorXs::Zero(A_c.cols());
  // Q = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c, A_ub, E);

  Eigen::VectorXs f_c = Q.completeOrthogonalDecomposition().solve(b);

  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  s_t dt = world->getTimeStep();

  Eigen::VectorXs nextV
      = world->getVelocities() + Minv * (dt * (tau - C) + A_c_ub_E * f_c);

  // return b;
  // return f_c;
  return nextV;
}

Eigen::MatrixXs BackpropSnapshot::getScratchFiniteDifference(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return getScratchFiniteDifferenceRidders(world, wrt);

  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);
  world->setPenetrationCorrectionEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXs original = scratch(world);

  int worldDim = wrt->dim(world.get());
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), worldDim);

  Eigen::VectorXs preStepWrt = wrt->get(world.get());

  s_t EPSILON = 1e-6;
  for (std::size_t i = 0; i < worldDim; i++)
  {
    Eigen::VectorXs tweakedWrt = preStepWrt;
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);
    Eigen::VectorXs perturbedPos = scratch(world);

    tweakedWrt = preStepWrt;
    tweakedWrt(i) -= EPSILON;
    wrt->set(world.get(), tweakedWrt);
    Eigen::VectorXs perturbedNeg = scratch(world);

    Eigen::VectorXs change = (perturbedPos - perturbedNeg) / (2 * EPSILON);
    J.col(i).noalias() = change;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);

  return J;
}

Eigen::MatrixXs BackpropSnapshot::getScratchFiniteDifferenceRidders(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->getConstraintSolver()->setGradientEnabled(false);
  world->setPenetrationCorrectionEnabled(false);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = scratch(world);

  int worldDim = wrt->dim(world.get());
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), worldDim);

  for (std::size_t i = 0; i < worldDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(world.get(), perturbed);
    Eigen::VectorXs perturbedPlus = scratch(world);

    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(world.get(), perturbed);
    Eigen::VectorXs perturbedMinus = scratch(world);

    tab[0][0] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(world.get(), perturbed);
      perturbedPlus = scratch(world);
      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(world.get(), perturbed);
      perturbedMinus = scratch(world);

      tab[0][iTab] = (perturbedPlus - perturbedMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  if (wrtDim == 0)
  {
    return Eigen::MatrixXs::Zero(world->getNumDofs(), 0);
  }
  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXs f_c = getClampingConstraintImpulses();
  s_t dt = world->getTimeStep();
  Eigen::VectorXs ddamp = getDampingVector(world);
  Eigen::VectorXs spring_stiffs = getSpringStiffVector(world);
  Eigen::VectorXs p_rest = getRestPositions(world);
  Eigen::VectorXs v_t = world->getVelocities();
  Eigen::VectorXs p_t = world->getPositions();
  Eigen::VectorXs spring_force = spring_stiffs.asDiagonal()*(p_t-p_rest+dt*v_t);
  Eigen::VectorXs damping_force = ddamp.asDiagonal()*v_t;
  Eigen::MatrixXs dM
      = getJacobianOfMinv(world, dt * (tau - C - damping_force - spring_force) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXs Minv = world->getInvMassMatrix();

  Eigen::MatrixXs dF_c = getJacobianOfConstraintForce(world, wrt);

  if (wrt == WithRespectTo::FORCE)
  {
    /*
    std::cout << "A_c_ub_E:" << std::endl << A_c_ub_E << std::endl;
    std::cout << "dF_c:" << std::endl << dF_c << std::endl;
    std::cout << "A_c_ub_E * dF_c:" << std::endl
              << A_c_ub_E * dF_c << std::endl;
    std::cout << "Minv * (A_c_ub_E * dF_c):" << std::endl
              << Minv * (A_c_ub_E * dF_c) << std::endl;
    std::cout << "Minv * (dt * I):" << std::endl
              << Minv
                     * (dt
                        * Eigen::MatrixXs::Identity(
                            world->getNumDofs(), world->getNumDofs()))
              << std::endl;
    */
    // snapshot.restore();
    return Minv
           * ((A_c_ub_E * dF_c)
              + (dt
                 * Eigen::MatrixXs::Identity(
                     world->getNumDofs(), world->getNumDofs())));
  }

  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);

  if (wrt == WithRespectTo::VELOCITY)
  {
    // snapshot.restore();
    return Eigen::MatrixXs::Identity(world->getNumDofs(), world->getNumDofs())
           + Minv * (A_c_ub_E * dF_c - dt * dC);
  }
  else if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dA_c = getJacobianOfClampingConstraints(world, f_c);
    Eigen::MatrixXs dA_ubE = getJacobianOfUpperBoundConstraints(world, E * f_c);
    // snapshot.restore();
    return dM + Minv * (A_c_ub_E * dF_c + dA_c + dA_ubE - dt * dC) - Minv*dt*spring_stiffs.asDiagonal();
  }
  else
  {
    // snapshot.restore();
    return dM + Minv * (A_c_ub_E * dF_c - dt * dC);
  }

  // std::cout << "dA_c: " << std::endl << dA_c << std::endl;

  // return dM + Minv * (dA_c + A_c_ub_E * dF_c - dt * dC);

  // Old version
  /*
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::MatrixXs dM = getJacobianOfMinv(world, tau - C, wrt);
  Eigen::MatrixXs Minv = world->getInvMassMatrix();
  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);
  s_t dt = world->getTimeStep();
  Eigen::VectorXs innerV = world->getVelocities() + dt * Minv * (tau - C);

  Eigen::MatrixXs dP_c
      = getJacobianOfProjectionIntoClampsMatrix(world, innerV, wrt);
  Eigen::MatrixXs P_c = getProjectionIntoClampsMatrix(world);
  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXs outerTau = tau - C - A_c_ub_E * P_c * innerV;
  Eigen::MatrixXs dOuterM = getJacobianOfMinv(world, outerTau, wrt);

  snapshot.restore();

  return dt
         * (dOuterM
            + Minv * (-dC - A_c_ub_E * (dP_c + P_c * dt * (dM - Minv * dC))));
}
  */
}

//==============================================================================
/// This computes and returns the whole wrt-pos jacobian. For backprop, you
/// don't actually need this matrix, you can compute backprop directly. This
/// is here if you want access to the full Jacobian for some reason.
Eigen::MatrixXs BackpropSnapshot::getPosJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  if (wrt == WithRespectTo::POSITION)
  {
    return getPosPosJacobian(world);
  }
  else if (wrt == WithRespectTo::VELOCITY)
  {
    return getVelPosJacobian(world);
  }
  else
  {
    return Eigen::MatrixXs::Zero(mNumDOFs, wrt->dim(world.get()));
  }
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getBounceApproximationJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog
        = perfLog->startRun("BackpropSnapshot.getBounceApproximationJacobian");
  }
#endif

  if (mCachedBounceApproximationDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getBounceApproximationJacobian#refreshCache");
    }
#endif

    /*
    RestorableSnapshot snapshot(world);
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    */

    Eigen::MatrixXs A_b = getBouncingConstraintMatrix(world);

    // If there are no bounces, pos-pos is a simple identity
    if (A_b.size() == 0)
    {
      mCachedBounceApproximation
          = Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs);
    }
    else
    {
      // Construct the W matrix we'll need to use to solve for our closest
      // approx
      Eigen::MatrixXs W
          = Eigen::MatrixXs::Zero(A_b.rows() * A_b.rows(), A_b.cols());
      for (int i = 0; i < A_b.cols(); i++)
      {
        Eigen::VectorXs a_i = A_b.col(i);
        for (int j = 0; j < A_b.rows(); j++)
        {
          W.block(j * A_b.rows(), i, A_b.rows(), 1) = a_i(j) * a_i;
        }
      }

      // We want to center the solution around the identity matrix, and find the
      // least-squares deviation along the diagonals that gets us there.
      Eigen::VectorXs center = Eigen::VectorXs::Zero(mNumDOFs * mNumDOFs);
      for (std::size_t i = 0; i < mNumDOFs; i++)
      {
        center((i * mNumDOFs) + i) = 1;
      }

      // Solve the linear system
      Eigen::VectorXs q
          = center
            - W.transpose().completeOrthogonalDecomposition().solve(
                getRestitutionDiagonals() + (W.eval().transpose() * center));

      // Recover X from the q vector
      Eigen::MatrixXs X = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
      for (std::size_t i = 0; i < mNumDOFs; i++)
      {
        X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
      }

      mCachedBounceApproximation = X;
    }
    // snapshot.restore();

    mCachedBounceApproximationDirty = false;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedBounceApproximation;
}

//==============================================================================
/// This returns the Jacobian for state_t -> state_{t+1}.
Eigen::MatrixXs BackpropSnapshot::getStateJacobian(simulation::WorldPtr world)
{
  // TODO: this method is untested, but is identical to the _tested_ method
  // World::getStateJacobian(). We should really test this anyways.
  int dofs = world->getNumDofs();
  Eigen::MatrixXs stateJac = Eigen::MatrixXs::Zero(2 * dofs, 2 * dofs);
  stateJac.block(0, 0, dofs, dofs) = getPosPosJacobian(world);
  stateJac.block(dofs, 0, dofs, dofs) = getPosVelJacobian(world);
  stateJac.block(0, dofs, dofs, dofs) = getVelPosJacobian(world);
  stateJac.block(dofs, dofs, dofs, dofs) = getVelVelJacobian(world);
  return stateJac;
}

//==============================================================================
/// This returns the Jacobian for action_t -> state_{t+1}.
Eigen::MatrixXs BackpropSnapshot::getActionJacobian(simulation::WorldPtr world)
{
  // TODO: this method is untested, but is identical to the _tested_ method
  // World::getActionJacobian(). We should really test this anyways.
  int dofs = world->getNumDofs();
  const Eigen::MatrixXs& forceVelJac = getControlForceVelJacobian(world);

  std::vector<int> actionSpace = world->getActionSpace();
  int actionDim = world->getActionSize();
  Eigen::MatrixXs actionJac = Eigen::MatrixXs::Zero(2 * dofs, actionDim);
  for (int i = 0; i < actionDim; i++)
  {
    actionJac.block(dofs, i, dofs, 1) = forceVelJac.col(actionSpace[i]);
  }
  return actionJac;
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getPosPosJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getPosPosJacobian");
  }
#endif

  if (mCachedPosPosDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getPosPosJacobian#refreshCache");
    }
#endif

    if (mUseFDOverride)
    {
      mCachedPosPos = finiteDifferencePosPosJacobian(world, 1);
    }
    else
    {
      /*
      RestorableSnapshot snapshot(world);
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      */

      mCachedPosPos = world->getPosPosJacobian()
                      * getBounceApproximationJacobian(world, thisLog);

      // snapshot.restore();
    }

    if (mSlowDebugResultsAgainstFD)
    {
      // TODO: this is crappy, because if we are actually bouncing we want a
      // better approximation
      Eigen::MatrixXs bruteForce = finiteDifferencePosPosJacobian(world, 1);
      equalsOrCrash(world, mCachedPosPos, bruteForce, "pos-pos");
    }

    mCachedPosPosDirty = false;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedPosPos;
}

//==============================================================================
const Eigen::MatrixXs& BackpropSnapshot::getVelPosJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getVelPosJacobian");
  }
#endif

  if (mCachedVelPosDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getVelPosJacobian#refreshCache");
    }
#endif

    if (mUseFDOverride)
    {
      mCachedVelPos = finiteDifferenceVelPosJacobian(world, 1);
    }
    else
    {
      mCachedVelPos = world->getVelPosJacobian()
                      * getBounceApproximationJacobian(world, thisLog);
    }

    if (mSlowDebugResultsAgainstFD)
    {
      // TODO: this is crappy, because if we are actually bouncing we want a
      // better approximation
      Eigen::MatrixXs bruteForce = finiteDifferenceVelPosJacobian(world, 1);
      equalsOrCrash(world, mCachedVelPos, bruteForce, "vel-pos");
    }

    mCachedVelPosDirty = false;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mCachedVelPos;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPreStepPosition()
{
  return mPreStepPosition;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPreStepVelocity()
{
  // return assembleVector<Eigen::VectorXs>(VectorToAssemble::PRE_STEP_VEL);
  return mPreStepVelocity;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPreStepTorques()
{
  // return assembleVector<Eigen::VectorXs>(VectorToAssemble::PRE_STEP_TAU);
  return mPreStepTorques;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPreConstraintVelocity()
{
  return mPreConstraintVelocities;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPostStepPosition()
{
  return mPostStepPosition;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPostStepVelocity()
{
  return mPostStepVelocity;
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPostStepTorques()
{
  return mPostStepTorques;
}

//==============================================================================
/// Returns the LCP's cached solution from before the step
const Eigen::VectorXs& BackpropSnapshot::getPreStepLCPCache()
{
  return mPreStepLCPCache;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getClampingConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::CLAMPING);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getMassedClampingConstraintMatrix(
    WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::MASSED_CLAMPING);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getUpperBoundConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::UPPER_BOUND);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getMassedUpperBoundConstraintMatrix(
    WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::MASSED_UPPER_BOUND);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getUpperBoundMappingMatrix()
{
  std::size_t numUpperBound = 0;
  std::size_t numClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    numUpperBound
        += mGradientMatrices[i]->getUpperBoundConstraintMatrix().cols();
    numClamping += mGradientMatrices[i]->getClampingConstraintMatrix().cols();
  }

  Eigen::MatrixXs mappingMatrix
      = Eigen::MatrixXs::Zero(numUpperBound, numClamping);

  std::size_t cursorUpperBound = 0;
  std::size_t cursorClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXs groupMappingMatrix
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
Eigen::MatrixXs BackpropSnapshot::getBouncingConstraintMatrix(WorldPtr world)
{
  return assembleMatrix(world, MatrixToAssemble::BOUNCING);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getMassMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  return assembleBlockDiagonalMatrix(
      world,
      BackpropSnapshot::BlockDiagonalMatrixToAssemble::MASS,
      forFiniteDifferencing);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getInvMassMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  return assembleBlockDiagonalMatrix(
      world,
      BackpropSnapshot::BlockDiagonalMatrixToAssemble::INV_MASS,
      forFiniteDifferencing);
}

Eigen::VectorXs BackpropSnapshot::getDampingVector(
  WorldPtr world
)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mNumDOFs);
  std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
  for (int i=0;i<mNumDOFs;i++)
  {
    result(i) = dofs[i]->getDampingCoefficient();
  }
  return result;
}

Eigen::VectorXs BackpropSnapshot::getSpringStiffVector(
  WorldPtr world
)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mNumDOFs);
  std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
  for (int i=0;i<mNumDOFs;i++)
  {
    result(i) = dofs[i]->getSpringStiffness();
  }
  return result;
}

Eigen::VectorXs BackpropSnapshot::getRestPositions(
  WorldPtr world
)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mNumDOFs);
  std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();
  for (int i=0;i<mNumDOFs;i++)
  {
    result(i) = dofs[i]->getRestPosition();
  }
  return result;
}
//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getClampingAMatrix()
{
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumClamping, mNumClamping);
  int cursor = 0;
  for (int i = 0; i < mGradientMatrices.size(); i++)
  {
    int size = mGradientMatrices[i]->getClampingAMatrix().rows();
    result.block(cursor, cursor, size, size)
        = mGradientMatrices[i]->getClampingAMatrix();
    cursor += size;
  }
  return result;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getPosCJacobian(simulation::WorldPtr world)
{
  return assembleBlockDiagonalMatrix(
      world, BackpropSnapshot::BlockDiagonalMatrixToAssemble::POS_C);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getVelCJacobian(simulation::WorldPtr world)
{
  return getJacobianOfC(world, WithRespectTo::VELOCITY);
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getContactConstraintImpulses()
{
  return assembleVector<Eigen::VectorXs>(
      VectorToAssemble::CONTACT_CONSTRAINT_IMPULSES);
}

//==============================================================================
Eigen::VectorXi BackpropSnapshot::getContactConstraintMappings()
{
  return assembleVector<Eigen::VectorXi>(
      VectorToAssemble::CONTACT_CONSTRAINT_MAPPINGS);
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getBounceDiagonals()
{
  return assembleVector<Eigen::VectorXs>(VectorToAssemble::BOUNCE_DIAGONALS);
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getRestitutionDiagonals()
{
  return assembleVector<Eigen::VectorXs>(
      VectorToAssemble::RESTITUTION_DIAGONALS);
}

//==============================================================================
Eigen::VectorXs BackpropSnapshot::getPenetrationCorrectionVelocities()
{
  return assembleVector<Eigen::VectorXs>(
      VectorToAssemble::PENETRATION_VELOCITY_HACK);
}

//==============================================================================
/// Returns the constraint impulses along the clamping constraints
Eigen::VectorXs BackpropSnapshot::getClampingConstraintImpulses()
{
  return assembleVector<Eigen::VectorXs>(
      VectorToAssemble::CLAMPING_CONSTRAINT_IMPULSES);
}

//==============================================================================
/// Returns the relative velocities along the clamping constraints
Eigen::VectorXs BackpropSnapshot::getClampingConstraintRelativeVels()
{
  return assembleVector<Eigen::VectorXs>(
      VectorToAssemble::CLAMPING_CONSTRAINT_RELATIVE_VELS);
}

//==============================================================================
/// Returns the velocity change caused by illegal impulses in the LCP this
/// timestep
Eigen::VectorXs BackpropSnapshot::getVelocityDueToIllegalImpulses()
{
  return assembleVector<Eigen::VectorXs>(VectorToAssemble::VEL_DUE_TO_ILLEGAL);
}

//==============================================================================
/// Returns the velocity pre-LCP
Eigen::VectorXs BackpropSnapshot::getPreLCPVelocity()
{
  return assembleVector<Eigen::VectorXs>(VectorToAssemble::PRE_LCP_VEL);
}

//==============================================================================
bool BackpropSnapshot::hasBounces()
{
  return mNumBouncing > 0;
}

//==============================================================================
std::size_t BackpropSnapshot::getNumContacts()
{
  std::size_t count = 0;
  for (auto gradientMatrices : mGradientMatrices)
  {
    count += gradientMatrices->getDifferentiableConstraints().size();
  }
  return count;
}

//==============================================================================
std::size_t BackpropSnapshot::getNumClamping()
{
  return mNumClamping;
}

//==============================================================================
std::size_t BackpropSnapshot::getNumUpperBound()
{
  return mNumUpperBound;
}

//==============================================================================
/// This is the clamping constraints from all the constrained
/// groups, concatenated together
std::vector<std::shared_ptr<DifferentiableContactConstraint>>
BackpropSnapshot::getDifferentiableConstraints()
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> vec;
  vec.reserve(mNumConstraintDim);
  for (auto gradientMatrices : mGradientMatrices)
  {
    for (auto constraint : gradientMatrices->getDifferentiableConstraints())
    {
      vec.push_back(constraint);
    }
  }
  assert(vec.size() == mNumConstraintDim);
  return vec;
}

//==============================================================================
std::vector<std::shared_ptr<DifferentiableContactConstraint>>
BackpropSnapshot::getClampingConstraints()
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> vec;
  vec.reserve(mNumClamping);
  for (auto gradientMatrices : mGradientMatrices)
  {
    for (auto constraint : gradientMatrices->getClampingConstraints())
    {
      constraint->setOffsetIntoWorld(vec.size(), false);
      vec.push_back(constraint);
    }
  }
  assert(vec.size() == mNumClamping);
  return vec;
}

//==============================================================================
std::vector<std::shared_ptr<DifferentiableContactConstraint>>
BackpropSnapshot::getUpperBoundConstraints()
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> vec;
  vec.reserve(mNumUpperBound);
  for (auto gradientMatrices : mGradientMatrices)
  {
    for (auto constraint : gradientMatrices->getUpperBoundConstraints())
    {
      constraint->setOffsetIntoWorld(vec.size(), true);
      vec.push_back(constraint);
    }
  }
  return vec;
}

//==============================================================================
/// This verifies that the two matrices are equal to some tolerance, and if
/// they're not it prints the information needed to replicated this scenario
/// and it exits the program.
void BackpropSnapshot::equalsOrCrash(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXs analytical,
    Eigen::MatrixXs bruteForce,
    std::string name)
{
  if (!areResultsStandardized())
  {
    std::cout << "Got an LCP result that couldn't be standardized!"
              << std::endl;
    printReplicationInstructions(world);
    exit(1);
  }
  Eigen::MatrixXs diff = (analytical - bruteForce).cwiseAbs();
  // TODO: this should be 1e-8, diverges slightly in KR5Trajectory
  s_t threshold = 5e-8;
  bool broken = (diff.array() > threshold).any();
  if (broken)
  {
    std::cout << "Found invalid matrix! " << name << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << diff << std::endl;
    diagnoseSubJacobianErrors(world, WithRespectTo::POSITION);
    printReplicationInstructions(world);
    exit(1);
  }
}

#define compare(analytical, bruteForce, threshold, name)                       \
  do                                                                           \
  {                                                                            \
    if (((analytical - bruteForce).cwiseAbs().array() > threshold).any())      \
    {                                                                          \
      std::cout << name << " disagrees! " << name << std::endl;                \
      std::cout << "Analytical:" << std::endl << analytical << std::endl;      \
      std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;     \
      std::cout << "Diff:" << std::endl                                        \
                << analytical - bruteForce << std::endl;                       \
    }                                                                          \
  } while (0)

//==============================================================================
/// This compares our analytical sub-Jacobians (like dMinv), to attempt to
/// diagnose where there are differences creeping in between our finite
/// differencing and our analytical results.
void BackpropSnapshot::diagnoseSubJacobianErrors(
    std::shared_ptr<simulation::World> world, WithRespectTo* wrt)
{
  s_t threshold = 3e-8;

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXs tau = world->getControlForces();
  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXs f_c = getClampingConstraintImpulses();
  s_t dt = world->getTimeStep();

  Eigen::VectorXs x = dt * (tau - C) + A_c_ub_E * f_c;
  Eigen::MatrixXs dM = getJacobianOfMinv(world, x, wrt);
  Eigen::MatrixXs dMFd = finiteDifferenceJacobianOfMinv(world, x, wrt);
  compare(dM, dMFd, threshold, "dMinv");
  if (((dM - dMFd).cwiseAbs().array() > threshold).any())
  {
    Eigen::VectorXs y = getInvMassMatrix(world) * x;
    // Internally, the dMinv calculation uses Minv*dM*Minv. So we need to check
    // the accuracy of dM.
    Eigen::MatrixXs dMyFd
        = finiteDifferenceJacobianOfM(world, y, wrt).cast<s_t>();
    int cursor = 0;
    for (int i = 0; i < world->getNumSkeletons(); i++)
    {
      auto skel = world->getSkeleton(i);
      int dofs = skel->getNumDofs();
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->debugJacobianOfMForward(
            WithRespectTo::POSITION, y.segment(cursor, dofs).cast<s_t>());
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->debugJacobianOfMBackward(
            WithRespectTo::POSITION,
            y.segment(cursor, dofs).cast<s_t>(),
            dMyFd);
      }
      cursor += dofs;
    }
  }

  Eigen::MatrixXs Minv = world->getInvMassMatrix();

  Eigen::MatrixXs dF_c = getJacobianOfConstraintForce(world, wrt);
  if (f_c.size() > 0)
  {
    Eigen::MatrixXs dF_cFd
        = finiteDifferenceJacobianOfConstraintForce(world, wrt);
    compare(dF_c, dF_cFd, threshold, "dF_c");
  }

  if (wrt == WithRespectTo::FORCE)
  {
    snapshot.restore();
  }

  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);
  Eigen::MatrixXs dCFd = finiteDifferenceJacobianOfC(world, wrt);
  compare(dC, dCFd, threshold, "dC");
  if (((dC - dCFd).cwiseAbs().array() > threshold).any())
  {
    for (int i = 0; i < world->getNumSkeletons(); i++)
    {
      auto skel = world->getSkeleton(i);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        skel->getBodyNode(j)->debugJacobianOfCForward(WithRespectTo::POSITION);
      }
      for (int j = skel->getNumBodyNodes() - 1; j >= 0; j--)
      {
        skel->getBodyNode(j)->debugJacobianOfCBackward(WithRespectTo::POSITION);
      }
    }
  }

  if (wrt == WithRespectTo::VELOCITY)
  {
    snapshot.restore();
  }
  else if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dA_c = getJacobianOfClampingConstraints(world, f_c);
    Eigen::MatrixXs dA_cFd
        = finiteDifferenceJacobianOfClampingConstraints(world, f_c);
    compare(dA_c, dA_cFd, threshold, "dA_c");

    Eigen::MatrixXs dA_ubE = getJacobianOfUpperBoundConstraints(world, E * f_c);
    Eigen::MatrixXs dA_ubEFd
        = finiteDifferenceJacobianOfUpperBoundConstraints(world, E * f_c);
    compare(dA_ubE, dA_ubEFd, threshold, "dA_ub");
    snapshot.restore();
  }
  else
  {
    snapshot.restore();
  }
}

//==============================================================================
void BackpropSnapshot::printReplicationInstructions(
    // TODO: export the world as a skel file
    std::shared_ptr<simulation::World> /* world */)
{
  std::cout << "Code to replicate:" << std::endl;
  std::cout << "--------------------" << std::endl;
  std::cout << "Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero(" << mNumDOFs
            << ");" << std::endl;
  std::cout << "brokenPos <<" << std::endl;
  for (int i = 0; i < mNumDOFs; i++)
  {
    std::cout << "  " << mPreStepPosition(i);
    if (i == mNumDOFs - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << "," << std::endl;
    }
  }
  std::cout << "Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero(" << mNumDOFs
            << ");" << std::endl;
  std::cout << "brokenVel <<" << std::endl;
  for (int i = 0; i < mNumDOFs; i++)
  {
    std::cout << "  " << mPreStepVelocity(i);
    if (i == mNumDOFs - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << "," << std::endl;
    }
  }
  std::cout << "Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero("
            << mNumDOFs << ");" << std::endl;
  std::cout << "brokenForce <<" << std::endl;
  for (int i = 0; i < mNumDOFs; i++)
  {
    std::cout << "  " << mPreStepTorques(i);
    if (i == mNumDOFs - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << "," << std::endl;
    }
  }
  std::cout << "Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero("
            << mPreStepLCPCache.size() << ");" << std::endl;
  if (mPreStepLCPCache.size() > 0)
  {
    std::cout << "brokenLCPCache <<" << std::endl;
    for (int i = 0; i < mPreStepLCPCache.size(); i++)
    {
      std::cout << "  " << mPreStepLCPCache(i);
      if (i == mPreStepLCPCache.size() - 1)
      {
        std::cout << ";" << std::endl;
      }
      else
      {
        std::cout << "," << std::endl;
      }
    }
  }

  std::cout << "world->setPositions(brokenPos);" << std::endl;
  std::cout << "world->setVelocities(brokenVel);" << std::endl;
  std::cout << "world->setControlForces(brokenForce);" << std::endl;
  std::cout << "world->setCachedLCPSolution(brokenLCPCache);" << std::endl;

  std::cout << "--------------------" << std::endl;
}

//==============================================================================
bool BackpropSnapshot::areResultsStandardized() const
{
  for (auto matrix : mGradientMatrices)
  {
    if (!matrix->areResultsStandardized())
      return false;
  }
  return true;
}

//==============================================================================
void BackpropSnapshot::setUseFDOverride(bool fdOverride)
{
  mUseFDOverride = fdOverride;
}

//==============================================================================
void BackpropSnapshot::setSlowDebugResultsAgainstFD(bool slowDebug)
{
  mSlowDebugResultsAgainstFD = slowDebug;
}

//==============================================================================
/// This does a battery of tests comparing the speeds to compute all the
/// different Jacobians, both with finite differencing and analytically, and
/// prints the results to std out.
void BackpropSnapshot::benchmarkJacobians(
    std::shared_ptr<simulation::World> world, int numSamples)
{
  long posPosFd = 0L;
  long posPosA = 0L;

  long posVelFd = 0L;
  long posVelA = 0L;

  long velPosFd = 0L;
  long velPosA = 0L;

  long velVelFd = 0L;
  long velVelA = 0L;

  long forceVelFd = 0L;
  long forceVelA = 0L;

  // Take a bunch of samples. This will not be the same speed as real runtime,
  // because of branch predictions warming up and caches getting warm, but it's
  // a reasonable indicator.
  for (int sample = 0; sample < numSamples; sample++)
  {
    using namespace std::chrono;

    for (auto contact : getClampingConstraints())
    {
      contact->mWorldConstraintJacCacheDirty = true;
    }
    for (auto contact : getUpperBoundConstraints())
    {
      contact->mWorldConstraintJacCacheDirty = true;
    }

    ////////////////////////////////////////////////////////////////////
    // Do all the analytical Jacobians one after another first
    ////////////////////////////////////////////////////////////////////

    long startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    mCachedPosPosDirty = true;
    getPosPosJacobian(world);
    long endTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    posPosA += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    mCachedPosVelDirty = true;
    getPosVelJacobian(world);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    posVelA += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    mCachedVelPosDirty = true;
    getVelPosJacobian(world);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    velPosA += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    mCachedVelVelDirty = true;
    getVelVelJacobian(world);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    velVelA += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    mCachedForceVelDirty = true;
    getControlForceVelJacobian(world);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    forceVelA += endTime - startTime;

    ////////////////////////////////////////////////////////////////////
    // Now do all the FD Jacobians one after another
    ////////////////////////////////////////////////////////////////////

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    finiteDifferencePosPosJacobian(world, 1, false);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    posPosFd += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    finiteDifferencePosVelJacobian(world, false);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    posVelFd += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    finiteDifferenceVelPosJacobian(world, 1, false);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    velPosFd += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    finiteDifferenceVelVelJacobian(world, false);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    velVelFd += endTime - startTime;

    startTime
        = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
              .count();
    finiteDifferenceForceVelJacobian(world, false);
    endTime = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                  .count();
    forceVelFd += endTime - startTime;
  }

  // Get one sample of each for accuracy testing
  Eigen::MatrixXs posPosJacA = getPosPosJacobian(world);
  Eigen::MatrixXs posVelJacA = getPosVelJacobian(world);
  Eigen::MatrixXs velPosJacA = getVelPosJacobian(world);
  Eigen::MatrixXs velVelJacA = getVelVelJacobian(world);
  Eigen::MatrixXs forceVelJacA = getControlForceVelJacobian(world);
  Eigen::MatrixXs posPosJacFD = finiteDifferencePosPosJacobian(world, 1, false);
  Eigen::MatrixXs posVelJacFD = finiteDifferencePosVelJacobian(world, false);
  Eigen::MatrixXs velPosJacFD = finiteDifferenceVelPosJacobian(world, 1, false);
  Eigen::MatrixXs velVelJacFD = finiteDifferenceVelVelJacobian(world, false);
  Eigen::MatrixXs forceVelJacFD
      = finiteDifferenceForceVelJacobian(world, false);
  Eigen::MatrixXs posPosJacR = finiteDifferencePosPosJacobian(world, 1, true);
  Eigen::MatrixXs posVelJacR = finiteDifferencePosVelJacobian(world, true);
  Eigen::MatrixXs velPosJacR = finiteDifferenceVelPosJacobian(world, 1, true);
  Eigen::MatrixXs velVelJacR = finiteDifferenceVelVelJacobian(world, true);
  Eigen::MatrixXs forceVelJacR = finiteDifferenceForceVelJacobian(world, true);

  // Now we need to form and print out a report
  std::cout << "Benchmark results:" << std::endl;

  long allA = posPosA + posVelA + velPosA + velVelA + forceVelA;
  long allFd = posPosFd + posVelFd + velPosFd + velVelFd + forceVelFd;
  s_t NANOS_TO_MILLIS = 1e-6;

  std::cout << "All Jacs:" << std::endl;
  std::cout << "   All Jacs  ANALYTICAL: "
            << ((s_t)allA * NANOS_TO_MILLIS / numSamples) << "ms" << std::endl;
  std::cout << "   All Jacs          FD: "
            << ((s_t)allFd * NANOS_TO_MILLIS / numSamples) << "ms" << std::endl;
  std::cout << "   All Jacs FD MULTIPLE: " << ((s_t)allFd / (s_t)allA)
            << "x faster" << std::endl;

  std::cout << "Pos-pos Jac:" << std::endl;
  std::cout << "   Pos-pos Jac  ANALYTICAL: "
            << ((s_t)posPosA * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Pos-pos Jac          FD: "
            << ((s_t)posPosFd * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Pos-pos Jac FD MULTIPLE: " << ((s_t)posPosFd / (s_t)posPosA)
            << "x faster" << std::endl;
  std::cout << "   Pos-pos Jac FD ACCURACY: "
            << (posPosJacFD - posPosJacR).array().abs().maxCoeff() << std::endl;
  std::cout << "   Pos-pos Jac  A ACCURACY: "
            << (posPosJacA - posPosJacR).array().abs().maxCoeff() << std::endl;

  std::cout << "Pos-vel Jac:" << std::endl;
  std::cout << "   Pos-vel Jac  ANALYTICAL: "
            << ((s_t)posVelA * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Pos-vel Jac          FD: "
            << ((s_t)posVelFd * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Pos-vel Jac FD MULTIPLE: " << ((s_t)posVelFd / (s_t)posVelA)
            << "x faster" << std::endl;
  std::cout << "   Pos-vel Jac FD ACCURACY: "
            << (posVelJacFD - posVelJacR).array().abs().maxCoeff() << std::endl;
  std::cout << "   Pos-vel Jac  A ACCURACY: "
            << (posVelJacA - posVelJacR).array().abs().maxCoeff() << std::endl;

  std::cout << "Vel-pos Jac:" << std::endl;
  std::cout << "   Vel-pos Jac  ANALYTICAL: "
            << ((s_t)velPosA * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Vel-pos Jac          FD: "
            << ((s_t)velPosFd * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Vel-pos Jac FD MULTIPLE: " << ((s_t)velPosFd / (s_t)velPosA)
            << "x faster" << std::endl;
  std::cout << "   Vel-pos Jac FD ACCURACY: "
            << (velPosJacFD - velPosJacR).array().abs().maxCoeff() << std::endl;
  std::cout << "   Vel-pos Jac  A ACCURACY: "
            << (velPosJacA - velPosJacR).array().abs().maxCoeff() << std::endl;

  std::cout << "Vel-vel Jac:" << std::endl;
  std::cout << "   Vel-vel Jac  ANALYTICAL: "
            << ((s_t)velVelA * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Vel-vel Jac          FD: "
            << ((s_t)velVelFd * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Vel-vel Jac FD MULTIPLE: " << ((s_t)velVelFd / (s_t)velVelA)
            << "x faster" << std::endl;
  std::cout << "   Vel-vel Jac FD ACCURACY: "
            << (velVelJacFD - velVelJacR).array().abs().maxCoeff() << std::endl;
  std::cout << "   Vel-vel Jac  A ACCURACY: "
            << (velVelJacA - velVelJacR).array().abs().maxCoeff() << std::endl;

  std::cout << "Force-vel Jac:" << std::endl;
  std::cout << "   Force-vel Jac  ANALYTICAL: "
            << ((s_t)forceVelA * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Force-vel Jac          FD: "
            << ((s_t)forceVelFd * NANOS_TO_MILLIS / numSamples) << "ms"
            << std::endl;
  std::cout << "   Force-vel Jac FD MULTIPLE: "
            << ((s_t)forceVelFd / (s_t)forceVelA) << "x faster" << std::endl;
  std::cout << "   Force-vel Jac FD ACCURACY: "
            << (forceVelJacFD - forceVelJacR).array().abs().maxCoeff()
            << std::endl;
  std::cout << "   Force-vel Jac  A ACCURACY: "
            << (forceVelJacA - forceVelJacR).array().abs().maxCoeff()
            << std::endl;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceVelVelJacobian(
    WorldPtr world, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersVelVelJacobian(world);

  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXs originalVel = world->getVelocities();

  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXs velPos = world->getVelocities();
    Eigen::VectorXs velNeg = world->getVelocities();

    s_t epsPos = EPSILON;
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs tweakedVel = Eigen::VectorXs(mPreStepVelocity);
      tweakedVel(i) += epsPos;
      world->setVelocities(tweakedVel);
      std::shared_ptr<neural::BackpropSnapshot> snapshot
          = neural::forwardPass(world, true);

      if ((!areResultsStandardized() || snapshot->areResultsStandardized())
          && snapshot->getNumClamping() == getNumClamping()
          && snapshot->getNumUpperBound() == getNumUpperBound())
      {
        velPos = snapshot->getPostStepVelocity();
        break;
      }
      epsPos *= 0.5;

      assert(abs(epsPos) > 1e-20);
    }

    s_t epsNeg = EPSILON;
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs tweakedVel = Eigen::VectorXs(mPreStepVelocity);
      tweakedVel(i) -= epsNeg;
      world->setVelocities(tweakedVel);
      std::shared_ptr<neural::BackpropSnapshot> snapshot
          = neural::forwardPass(world, true);

      if ((!areResultsStandardized() || snapshot->areResultsStandardized())
          && snapshot->getNumClamping() == getNumClamping()
          && snapshot->getNumUpperBound() == getNumUpperBound())
      {
        velNeg = snapshot->getPostStepVelocity();
        break;
      }
      epsNeg *= 0.5;

      assert(abs(epsNeg) > 1e-20);
    }

    Eigen::VectorXs velChange = (velPos - velNeg) / (epsPos + epsNeg);

    // TODO: remove me
    /*
#ifndef NDEBUG
    Eigen::VectorXs identityCol = Eigen::VectorXs::Zero(velChange.size());
    identityCol(i) = 1.0;
    // Sanity check
    if ((velChange - identityCol).squaredNorm() > 1e-10)
    {
      // Something is screwy here, let's investigate
      dynamics::DegreeOfFreedom* dof = world->getDofs()[i];
      std::cout << "Error on perturbing joint vel: " << dof->getName()
                << std::endl;

      snapshot.restore();
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      tweakedVel = Eigen::VectorXs::Zero(mPreStepVelocity);
      tweakedVel(i) -= EPSILON;
      world->setVelocities(tweakedVel);
      // Opportunity to put a breakpoint here
      world->step(false);
    }
#endif
    */
    // TODO: </remove>

    J.col(i).noalias() = velChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersVelVelJacobian(
    WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs velPlus;
    Eigen::VectorXs velMinus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepVelocity);
      perturbedPlus(i) += originalStepSize;
      world->setVelocities(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepVelocity);
      perturbedMinus(i) -= originalStepSize;
      world->setVelocities(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);

      if ((!areResultsStandardized() || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == getNumClamping()
          && snapshotPlus->getNumUpperBound() == getNumUpperBound()
          && (!areResultsStandardized()
              || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == getNumClamping()
          && snapshotMinus->getNumUpperBound() == getNumUpperBound())
      {
        velPlus = snapshotPlus->getPostStepVelocity();
        velMinus = snapshotMinus->getPostStepVelocity();
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    tab[0][0] = (velPlus - velMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepVelocity);
      perturbedPlus(i) += stepSize;
      world->setVelocities(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      velPlus = snapshotPlus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotPlus->areResultsStandardized())
            && snapshotPlus->getNumClamping() == getNumClamping()
            && snapshotPlus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersVelVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }
      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepVelocity);
      perturbedMinus(i) -= stepSize;
      world->setVelocities(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      velMinus = snapshotMinus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotMinus->areResultsStandardized())
            && snapshotMinus->getNumClamping() == getNumClamping()
            && snapshotMinus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersVelVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (velPlus - velMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferencePosVelJacobian(
    simulation::WorldPtr world, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersPosVelJacobian(world);

  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  world->step(false);

  Eigen::VectorXs originalVel = world->getVelocities();

  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXs perturbedVelPos = world->getVelocities();
    Eigen::VectorXs perturbedVelNeg = world->getVelocities();

    s_t epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs tweakedPos = Eigen::VectorXs(mPreStepPosition);
      tweakedPos(i) += epsPos;
      world->setPositions(tweakedPos);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelPos = ptr->getPostStepVelocity();
        break;
      }
      epsPos *= 0.5;

      if (abs(epsPos) <= 1e-20)
      {
        std::cout << "Found a non-differentiabe point in getting pos-vel Jac:"
                  << std::endl;
        printReplicationInstructions(world);
      }
      assert(abs(epsPos) > 1e-20);
    }

    s_t epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs tweakedPos = Eigen::VectorXs(mPreStepPosition);
      tweakedPos(i) -= epsNeg;
      world->setPositions(tweakedPos);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelNeg = ptr->getPostStepVelocity();
        break;
      }
      epsNeg *= 0.5;

      if (abs(epsNeg) <= 1e-20)
      {
        std::cout << "Found a non-differentiabe point in getting pos-vel Jac:"
                  << std::endl;
        printReplicationInstructions(world);
      }
      assert(abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersPosVelJacobian(
    simulation::WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

#ifdef DART_USE_ARBITRARY_PRECISION
  s_t originalStepSize = 5e-6;
#else
  s_t originalStepSize = 5e-3;
#endif
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 15;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs velPlus;
    Eigen::VectorXs velMinus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += originalStepSize;
      world->setPositions(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= originalStepSize;
      world->setPositions(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);

      if ((!areResultsStandardized() || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumContacts() == getNumContacts()
          && snapshotPlus->getNumClamping() == getNumClamping()
          && snapshotPlus->getNumUpperBound() == getNumUpperBound()
          && (!areResultsStandardized()
              || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumContacts() == getNumContacts()
          && snapshotMinus->getNumClamping() == getNumClamping()
          && snapshotMinus->getNumUpperBound() == getNumUpperBound())
      {
        velPlus = snapshotPlus->getPostStepVelocity();
        velMinus = snapshotMinus->getPostStepVelocity();
        break;
      }
      originalStepSize *= 0.5;

      if (abs(originalStepSize) <= 1e-20)
      {
        std::cout << "Found a non-differentiabe point in getting pos-vel Jac:"
                  << std::endl;
        printReplicationInstructions(world);
      }
      assert(abs(originalStepSize) > 1e-20);
    }

    tab[0][0] = (velPlus - velMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      velPlus = snapshotPlus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotPlus->areResultsStandardized())
            && snapshotPlus->getNumContacts() == getNumContacts()
            && snapshotPlus->getNumClamping() == getNumClamping()
            && snapshotPlus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersPosVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      velMinus = snapshotMinus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotMinus->areResultsStandardized())
            && snapshotMinus->getNumClamping() == getNumClamping()
            && snapshotMinus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersPosVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (velPlus - velMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceForceVelJacobian(
    WorldPtr world, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersForceVelJacobian(world);

  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXs originalForces = world->getControlForces();
  Eigen::VectorXs originalVel = world->getVelocities();

  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXs perturbedVelPos = world->getVelocities();
    Eigen::VectorXs perturbedVelNeg = world->getVelocities();

    s_t epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs tweakedForces = Eigen::VectorXs(mPreStepTorques);
      tweakedForces(i) += epsPos;
      world->setControlForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelPos = ptr->getPostStepVelocity();
        break;
      }
      epsPos *= 0.5;

      assert(abs(epsPos) > 1e-20);
    }

    s_t epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs tweakedForces = Eigen::VectorXs(mPreStepTorques);
      tweakedForces(i) -= epsNeg;
      world->setControlForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelNeg = ptr->getPostStepVelocity();
        break;
      }
      epsNeg *= 0.5;

      assert(abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  /*
  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs tweakedForcesPos = Eigen::VectorXs::Zero(originalForces);
    tweakedForcesPos(i) += EPSILON;
    world->setControlForces(tweakedForcesPos);
    world->step(false);
    Eigen::VectorXs velPos = world->getVelocities();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs tweakedForcesNeg = Eigen::VectorXs::Zero(originalForces);
    tweakedForcesNeg(i) -= EPSILON;
    world->setControlForces(tweakedForcesNeg);
    world->step(false);
    Eigen::VectorXs velNeg = world->getVelocities();

    Eigen::VectorXs velChange = (velPos - velNeg) / (2 * EPSILON);
    J.col(i).noalias() = velChange;
  }
  */

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersForceVelJacobian(
    WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs velPlus;
    Eigen::VectorXs velMinus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepTorques);
      perturbedPlus(i) += originalStepSize;
      world->setControlForces(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepTorques);
      perturbedMinus(i) -= originalStepSize;
      world->setControlForces(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);

      if ((!areResultsStandardized() || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == getNumClamping()
          && snapshotPlus->getNumUpperBound() == getNumUpperBound()
          && (!areResultsStandardized()
              || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == getNumClamping()
          && snapshotMinus->getNumUpperBound() == getNumUpperBound())
      {
        velPlus = snapshotPlus->getPostStepVelocity();
        velMinus = snapshotMinus->getPostStepVelocity();
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    tab[0][0] = (velPlus - velMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepTorques);
      perturbedPlus(i) += stepSize;
      world->setControlForces(perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      velPlus = snapshotPlus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotPlus->areResultsStandardized())
            && snapshotPlus->getNumClamping() == getNumClamping()
            && snapshotPlus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersForceVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepTorques);
      perturbedMinus(i) -= stepSize;
      world->setControlForces(perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      velMinus = snapshotMinus->getPostStepVelocity();
      if (!((!areResultsStandardized()
             || snapshotMinus->areResultsStandardized())
            && snapshotMinus->getNumClamping() == getNumClamping()
            && snapshotMinus->getNumUpperBound() == getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersForceVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (velPlus - velMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceMassVelJacobian(
    simulation::WorldPtr world, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersMassVelJacobian(world);

  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXs originalMass = world->getWrtMass()->get(world.get());
  Eigen::VectorXs originalVel = world->getVelocities();

  Eigen::MatrixXs J(mNumDOFs, originalMass.size());

  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < originalMass.size(); i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs tweakedMass = Eigen::VectorXs(originalMass);
    tweakedMass(i) += EPSILON;
    world->getWrtMass()->set(world.get(), tweakedMass);

    world->step(false);

    Eigen::VectorXs velChange
        = (world->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersMassVelJacobian(
    simulation::WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs originalMass = world->getWrtMass()->get(world.get());
  Eigen::VectorXs velPlus;
  Eigen::VectorXs velMinus;

  Eigen::MatrixXs J(mNumDOFs, originalMass.size());

  for (std::size_t i = 0; i < originalMass.size(); i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalMass);
    perturbedPlus(i) += stepSize;
    world->getWrtMass()->set(world.get(), perturbedPlus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
        = neural::forwardPass(world, true);
    velPlus = snapshotPlus->getPostStepVelocity();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalMass);
    perturbedMinus(i) -= stepSize;
    world->getWrtMass()->set(world.get(), perturbedMinus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
        = neural::forwardPass(world, true);
    velMinus = snapshotMinus->getPostStepVelocity();

    tab[0][0] = (velPlus - velMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      perturbedPlus = Eigen::VectorXs(originalMass);
      perturbedPlus(i) += stepSize;
      world->getWrtMass()->set(world.get(), perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      velPlus = snapshotPlus->getPostStepVelocity();
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      perturbedMinus = Eigen::VectorXs(originalMass);
      perturbedMinus(i) -= stepSize;
      world->getWrtMass()->set(world.get(), perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      velMinus = snapshotMinus->getPostStepVelocity();

      tab[0][iTab] = (velPlus - velMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferencePosPosJacobian(
    WorldPtr world, std::size_t subdivisions, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersPosPosJacobian(world, subdivisions);

  RestorableSnapshot snapshot(world);

  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXs originalPosition = world->getPositions();

  if (subdivisions == 1)
  {
    s_t EPSILON = 1e-6;
    for (std::size_t i = 0; i < world->getNumDofs(); i++)
    {
      snapshot.restore();

      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      Eigen::VectorXs tweakedPositions = mPreStepPosition;
      tweakedPositions(i) += EPSILON;
      world->setPositions(tweakedPositions);
      world->step(false);
      Eigen::VectorXs posChange = world->getPositions();

      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      tweakedPositions = mPreStepPosition;
      tweakedPositions(i) -= EPSILON;
      world->setPositions(tweakedPositions);
      world->step(false);
      Eigen::VectorXs negChange = world->getPositions();

      J.col(i).noalias() = (posChange - negChange) / (2 * EPSILON);
    }
  }
  else
  {
    // IMPORTANT: EPSILON must be larger than the distance traveled in a single
    // subdivided timestep. Ideally much larger.
    s_t EPSILON = 1e-2 / subdivisions;
    for (std::size_t i = 0; i < world->getNumDofs(); i++)
    {
      snapshot.restore();

      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      Eigen::VectorXs tweakedPositions = Eigen::VectorXs(mPreStepPosition);
      tweakedPositions(i) += EPSILON;
      world->setPositions(tweakedPositions);

      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);

      Eigen::VectorXs posChange
          = (world->getPositions() - originalPosition) / EPSILON;
      J.col(i).noalias() = posChange;
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersPosPosJacobian(
    WorldPtr world, std::size_t subdivisions)
{
  RestorableSnapshot snapshot(world);

  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  const s_t originalStepSize = 1e-3 / subdivisions;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
    perturbedPlus(i) += stepSize;
    world->setPositions(perturbedPlus);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXs posPlus = world->getPositions();

    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
    perturbedMinus(i) -= stepSize;
    world->setPositions(perturbedMinus);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXs posMinus = world->getPositions();

    tab[0][0] = (posPlus - posMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      posPlus = world->getPositions();

      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      posMinus = world->getPositions();

      tab[0][iTab] = (posPlus - posMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceVelPosJacobian(
    WorldPtr world, std::size_t subdivisions, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersVelPosJacobian(world, subdivisions);

  RestorableSnapshot snapshot(world);

  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXs originalPosition = world->getPositions();

  if (subdivisions == 1)
  {
    s_t EPSILON = 1e-6;
    for (std::size_t i = 0; i < world->getNumDofs(); i++)
    {
      snapshot.restore();

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      Eigen::VectorXs tweakedVelocity = mPreStepVelocity;
      tweakedVelocity(i) += EPSILON;
      world->setVelocities(tweakedVelocity);
      world->step(false);
      Eigen::VectorXs posChange = world->getPositions();

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      tweakedVelocity = mPreStepVelocity;
      tweakedVelocity(i) -= EPSILON;
      world->setVelocities(tweakedVelocity);
      world->step(false);
      Eigen::VectorXs negChange = world->getPositions();

      J.col(i).noalias() = (posChange - negChange) / (2 * EPSILON);
    }
  }
  else
  {
    s_t EPSILON = 1e-3 / subdivisions;
    for (std::size_t i = 0; i < world->getNumDofs(); i++)
    {
      snapshot.restore();

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      Eigen::VectorXs tweakedVelocity = Eigen::VectorXs(mPreStepVelocity);
      tweakedVelocity(i) += EPSILON;
      world->setVelocities(tweakedVelocity);

      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);

      Eigen::VectorXs posChange
          = (world->getPositions() - originalPosition) / EPSILON;
      J.col(i).noalias() = posChange;
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersVelPosJacobian(
    WorldPtr world, std::size_t subdivisions)
{
  RestorableSnapshot snapshot(world);

  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::MatrixXs J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  const s_t originalStepSize = 1e-3 / subdivisions;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepVelocity);
    perturbedPlus(i) += stepSize;
    world->setVelocities(perturbedPlus);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXs posPlus = world->getPositions();

    world->setPositions(mPreStepPosition);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepVelocity);
    perturbedMinus(i) -= stepSize;
    world->setVelocities(perturbedMinus);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXs posMinus = world->getPositions();

    tab[0][0] = (posPlus - posMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedPlus = Eigen::VectorXs(mPreStepVelocity);
      perturbedPlus(i) += stepSize;
      world->setVelocities(perturbedPlus);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      posPlus = world->getPositions();

      world->setPositions(mPreStepPosition);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedMinus = Eigen::VectorXs(mPreStepVelocity);
      perturbedMinus(i) -= stepSize;
      world->setVelocities(perturbedMinus);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      posMinus = world->getPositions();

      tab[0][iTab] = (posPlus - posMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the whole wrt-vel jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersVelJacobianWrt(world, wrt);

  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXs originalVel = world->getVelocities();

  s_t EPSILON = 1e-7;
  for (std::size_t i = 0; i < wrtDim; i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXs tweakedWrt = Eigen::VectorXs(originalWrt);
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);

    world->step(false);

    Eigen::VectorXs velChange
        = (world->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
/// This computes and returns the whole wrt-vel jacobian by Ridders
/// extrapolated finite differences. This is SUPER SUPER SLOW,
/// and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  for (std::size_t i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
        = neural::forwardPass(world, true);
    Eigen::VectorXs velPlus = snapshotPlus->getPostStepVelocity();

    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
        = neural::forwardPass(world, true);
    Eigen::VectorXs velMinus = snapshotMinus->getPostStepVelocity();

    tab[0][0] = (velPlus - velMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      velPlus = snapshotPlus->getPostStepVelocity();

      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      velMinus = snapshotMinus->getPostStepVelocity();

      tab[0][iTab] = (velPlus - velMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
/// This computes and returns the whole wrt-pos jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferencePosJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersPosJacobianWrt(world, wrt);

  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXs originalPos = world->getPositions();

  s_t EPSILON = 1e-6;
  for (std::size_t i = 0; i < wrtDim; i++)
  {
    snapshot.restore();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs tweakedWrt = Eigen::VectorXs(originalWrt);
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);
    world->step(false);
    Eigen::VectorXs pos = world->getPositions();

    snapshot.restore();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    tweakedWrt = Eigen::VectorXs(originalWrt);
    tweakedWrt(i) -= EPSILON;
    wrt->set(world.get(), tweakedWrt);
    world->step(false);
    Eigen::VectorXs neg = world->getPositions();

    Eigen::VectorXs posChange = (pos - neg) / (2 * EPSILON);
    J.col(i).noalias() = posChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
/// This computes and returns the whole wrt-pos jacobian by Ridders
/// extrapolated finite differences. This is SUPER SUPER SLOW,
/// and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersPosJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  for (std::size_t i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
        = neural::forwardPass(world, true);
    Eigen::VectorXs posPlus = snapshotPlus->getPostStepPosition();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
        = neural::forwardPass(world, true);
    Eigen::VectorXs posMinus = snapshotMinus->getPostStepPosition();

    tab[0][0] = (posPlus - posMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotPlus
          = neural::forwardPass(world, true);
      posPlus = snapshotPlus->getPostStepPosition();
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      std::shared_ptr<neural::BackpropSnapshot> snapshotMinus
          = neural::forwardPass(world, true);
      posMinus = snapshotMinus->getPostStepPosition();

      tab[0][0] = (posPlus - posMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

/*
//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getProjectionIntoClampsMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  Eigen::MatrixXs A_c;
  if (forFiniteDifferencing)
  {
    A_c = getClampingConstraintMatrixAt(world, world->getPositions());
  }
  else
  {
    A_c = getClampingConstraintMatrix(world);
  }
  if (A_c.size() == 0)
    return Eigen::MatrixXs::Zero(0, world->getNumDofs());

  Eigen::MatrixXs E = getUpperBoundMappingMatrix();

  Eigen::MatrixXs constraintForceToImpliedTorques;
  if (forFiniteDifferencing || true)
  {
    Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
    Eigen::MatrixXs Minv = getInvMassMatrix(world, forFiniteDifferencing);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));
  }
  else
  {
    Eigen::MatrixXs V_c = getMassedClampingConstraintMatrix(world);
    Eigen::MatrixXs V_ub = getMassedUpperBoundConstraintMatrix(world);
    constraintForceToImpliedTorques = V_c + (V_ub * E);
  }

  Eigen::MatrixXs forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXs bounce = getBounceDiagonals().asDiagonal();
  Eigen::MatrixXs rightHandSize = bounce * A_c.transpose();
  return (1.0 / mTimeStep)
         * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
}
*/

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getProjectionIntoClampsMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  Eigen::MatrixXs A_c;
  if (forFiniteDifferencing)
  {
    A_c = getClampingConstraintMatrixAt(world, world->getPositions());
  }
  else
  {
    A_c = getClampingConstraintMatrix(world);
  }
  if (A_c.size() == 0)
    return Eigen::MatrixXs::Zero(0, world->getNumDofs());

  Eigen::MatrixXs constraintForceToImpliedTorques;
  if (forFiniteDifferencing)
  {
    Eigen::MatrixXs A_ub
        = getUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs E
        = getUpperBoundMappingMatrixAt(world, world->getPositions());
    Eigen::MatrixXs Minv = getInvMassMatrix(world, true);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));

    Eigen::MatrixXs forceToVel
        = A_c.eval().transpose() * constraintForceToImpliedTorques;
    Eigen::MatrixXs bounce
        = getBounceDiagonalsAt(world, world->getPositions()).asDiagonal();
    Eigen::MatrixXs rightHandSize = bounce * A_c.transpose();
    return (1.0 / mTimeStep)
           * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
  }
  else
  {
    Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();
    Eigen::MatrixXs Minv = getInvMassMatrix(world, false);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));
    // We don't use the massed formulation anymore because it introduces slight
    // numerical instability

    // Eigen::MatrixXs V_c = getMassedClampingConstraintMatrix(world);
    // Eigen::MatrixXs V_ub = getMassedUpperBoundConstraintMatrix(world);
    // constraintForceToImpliedTorques = V_c + (V_ub * E);
    Eigen::MatrixXs forceToVel
        = A_c.eval().transpose() * constraintForceToImpliedTorques;
    Eigen::MatrixXs bounce = getBounceDiagonals().asDiagonal();
    Eigen::MatrixXs rightHandSize = bounce * A_c.transpose();
    return (1.0 / mTimeStep)
           * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
  }
}

/// This returns the result of M*x, without explicitly
/// forming M
Eigen::VectorXs BackpropSnapshot::implicitMultiplyByMassMatrix(
    simulation::WorldPtr world, const Eigen::VectorXs& x)
{
  Eigen::VectorXs result = x;
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->multiplyByImplicitMassMatrix(x.segment(cursor, dofs));
    cursor += dofs;
  }
  return result;
}

/// This return the result of Minv*x, without explicitly
/// forming Minv
Eigen::VectorXs BackpropSnapshot::implicitMultiplyByInvMassMatrix(
    simulation::WorldPtr world, const Eigen::VectorXs& x)
{
  Eigen::VectorXs result = x;
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->multiplyByImplicitInvMassMatrix(x.segment(cursor, dofs));
    cursor += dofs;
  }
  return result;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::getJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
#ifndef NDEBUG
  assert(world->getPositions() == mPreStepPosition);
  assert(world->getVelocities() == mPreStepVelocity);
  assert(world->getControlForces() == mPreStepTorques);
  assert(world->getCachedLCPSolution() == mPreStepLCPCache);
#endif

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  if (A_c.cols() == 0)
  {
    int wrtDim = wrt->dim(world.get());
    return Eigen::MatrixXs::Zero(0, wrtDim);
  }
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();

  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;
  Eigen::MatrixXs Q = A_c.transpose() * Minv * A_c_ub_E;
  Q.diagonal() += getConstraintForceMixingDiagonal();

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXs dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    // dQ_b is 0, so don't compute it
    // snapshot.restore();
    return Qfac.solve(dB);
  }

  Eigen::VectorXs b = getClampingConstraintRelativeVels();
  Eigen::MatrixXs dQ_b
      = getJacobianOfLCPConstraintMatrixClampingSubset(world, b, wrt);

  // snapshot.restore();

  return dQ_b + Qfac.solve(dB);
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::dQ_WithUB(
    simulation::WorldPtr world,
    Eigen::MatrixXs& Minv,
    Eigen::MatrixXs& A_c,
    Eigen::MatrixXs& E,
    Eigen::MatrixXs& A_c_ub_E,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = getJacobianOfClampingConstraintsTranspose(world, Minv * A_c_ub_E * rhs)
        + (A_c.transpose()
           * (getJacobianOfMinv(world, A_c_ub_E * rhs, wrt)
              + (Minv
                 * (getJacobianOfClampingConstraints(world, rhs)
                    + getJacobianOfUpperBoundConstraints(world, E * rhs)))));
  return result;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::dQT_WithUB(
    simulation::WorldPtr world,
    Eigen::MatrixXs& Minv,
    Eigen::MatrixXs& A_c,
    Eigen::MatrixXs& E,
    Eigen::MatrixXs& A_ub,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)
         + (A_c.transpose()
            * (getJacobianOfMinv(world, A_c * rhs, wrt)
               + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))
        + (E.transpose()
           * (getJacobianOfUpperBoundConstraintsTranspose(
                  world, Minv * A_c * rhs)
              + A_ub.transpose()
                    * (getJacobianOfMinv(world, A_c * rhs, wrt)
                       + (Minv
                          * (getJacobianOfClampingConstraints(world, rhs))))));
  return result;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::dQ_WithoutUB(
    simulation::WorldPtr world,
    Eigen::MatrixXs& Minv,
    Eigen::MatrixXs& A_c,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)
        + (A_c.transpose()
           * (getJacobianOfMinv(world, A_c * rhs, wrt)
              + (Minv * (getJacobianOfClampingConstraints(world, rhs)))));
  return result;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfQb(
    simulation::WorldPtr world,
    Eigen::VectorXs b,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfQb(world, b, wrt);
  }

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  if (wrt != WithRespectTo::POSITION)
  {
    return jac;
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t EPS = 1e-6;
  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    Eigen::MatrixXs A_c
        = estimateClampingConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs A_ub
        = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();
    Eigen::MatrixXs Q
        = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();

    Eigen::VectorXs bPlus = Q * b;

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    E = getUpperBoundMappingMatrix();
    Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();

    Eigen::VectorXs bMinus = Q * b;

    jac.col(i) = (bPlus - bMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  snapshot.restore();
  return jac;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersJacobianOfQb(
    simulation::WorldPtr world, Eigen::VectorXs b, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  if (wrt != WithRespectTo::POSITION)
  {
    return jac;
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 14;

  Eigen::VectorXs original = wrt->get(world.get());

  for (int i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs A_c
        = estimateClampingConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs A_ub
        = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();
    Eigen::MatrixXs Q
        = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();
    Eigen::VectorXs QbPlus = Q * b;
    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(world.get(), perturbed);
    A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    E = getUpperBoundMappingMatrix();
    Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();
    Eigen::VectorXs QbMinus = Q * b;

    tab[0][0] = (QbPlus - QbMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(world.get(), perturbed);
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
      E = getUpperBoundMappingMatrix();
      Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
      Q.diagonal() += getConstraintForceMixingDiagonal();
      QbPlus = Q * b;
      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(world.get(), perturbed);
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
      E = getUpperBoundMappingMatrix();
      Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
      Q.diagonal() += getConstraintForceMixingDiagonal();
      QbMinus = Q * b;

      tab[0][iTab] = (QbPlus - QbMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i) = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), original);
  snapshot.restore();

  return jac;
}

//==============================================================================
/// This returns the vector of constants that get added to the diagonal of Q
/// to guarantee that Q is full-rank
Eigen::VectorXs BackpropSnapshot::getConstraintForceMixingDiagonal()
{
  return assembleVector<Eigen::VectorXs>(VectorToAssemble::CFM_CONSTANTS);
}

//==============================================================================
Eigen::MatrixXs
BackpropSnapshot::getJacobianOfLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXs b, WithRespectTo* wrt)
{
  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  if (A_c.cols() == 0)
  {
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    // TODO(keenon): this was A_c.cols() columns, instead of mNumDOFs, but that
    // doesn't seem to make dimensional sense. Change this back if things break.
    return Eigen::MatrixXs::Zero(A_c.cols(), mNumDOFs);
  }

  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */
#ifndef NDEBUG
  assert(world->getPositions() == mPreStepPosition);
  assert(world->getVelocities() == mPreStepVelocity);
  assert(world->getControlForces() == mPreStepTorques);
  assert(world->getCachedLCPSolution() == mPreStepLCPCache);
#endif

  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  Eigen::MatrixXs Q = A_c.transpose() * Minv * (A_c + A_ub * E);
  Q.diagonal() += getConstraintForceMixingDiagonal();
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> Qfactored
      = Q.completeOrthogonalDecomposition();

  Eigen::VectorXs Qinv_b = Qfactored.solve(b);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs Qinv = Qfactored.pseudoInverse();
    Eigen::MatrixXs I = Eigen::MatrixXs::Identity(Q.rows(), Q.cols());

    // Position is the only term that affects A_c and A_ub. We use the full
    // gradient of the pseudoinverse, rather than approximate with the gradient
    // of the raw inverse, because Q could be rank-deficient.

    if (A_ub.cols() > 0)
    {

#define dQ(rhs)                                                                \
  (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c_ub_E * rhs)     \
   + (A_c.transpose()                                                          \
      * (getJacobianOfMinv(world, A_c_ub_E * rhs, wrt)                         \
         + (Minv                                                               \
            * (getJacobianOfClampingConstraints(world, rhs)                    \
               + getJacobianOfUpperBoundConstraints(world, E * rhs))))))

#define dQT(rhs)                                                               \
  ((getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)         \
    + (A_c.transpose()                                                         \
       * (getJacobianOfMinv(world, A_c * rhs, wrt)                             \
          + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))         \
   + (E.transpose()                                                            \
      * (getJacobianOfUpperBoundConstraintsTranspose(world, Minv * A_c * rhs)  \
         + A_ub.transpose()                                                    \
               * (getJacobianOfMinv(world, A_c * rhs, wrt)                     \
                  + (Minv                                                      \
                     * (getJacobianOfClampingConstraints(world, rhs)))))))

      // snapshot.restore();

      Eigen::MatrixXs imprecisionMap = I - Q * Qinv;

      // If we were able to precisely invert Q, then let's use the exact inverse
      // Jacobian, because it's faster to compute
      if (imprecisionMap.squaredNorm() < 1e-18)
      {
        // Note: this formula only asks for the Jacobian of Minv once, instead
        // of 3 times like the below formula. That's actually a pretty big speed
        // advantage. When we can, we should use this formula instead.
        // return -Qinv * dQ(Qinv * b);
        return -Qfactored.solve(dQ(Qfactored.solve(b)));
      }
      // Otherwise fall back to the exact Jacobian of the pseudo-inverse
      else
      {
        // This is the gradient of the pseudoinverse, see
        // https://mathoverflow.net/a/29511/163259
        return -Qfactored.solve(dQ(Qfactored.solve(b)))
               + Qfactored.solve(Qinv.transpose() * dQT(imprecisionMap * b))
               + (I - Qinv * Q) * dQT(Qinv.transpose() * Qfactored.solve(b));
      }

#undef dQ
#undef dQT
    }
    else
    {

      // A_ub = 0 here

#define dQ(rhs)                                                                \
  (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)          \
   + (A_c.transpose()                                                          \
      * (getJacobianOfMinv(world, A_c * rhs, wrt)                              \
         + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))

#define dQT(rhs) dQ(rhs)

      Eigen::MatrixXs imprecisionMap = I - Q * Qinv;

      // If we were able to precisely invert Q, then let's use the exact inverse
      // Jacobian, because it's faster to compute
      if (imprecisionMap.squaredNorm() < 1e-18)
      {
        // Note: this formula only asks for the Jacobian of Minv once, instead
        // of 3 times like the below formula. That's actually a pretty big speed
        // advantage. When we can, we should use this formula instead.
        // return -Qinv * dQ(Qinv * b);
        return -Qfactored.solve(dQ(Qfactored.solve(b)));
      }
      else
      {
        // This is the gradient of the pseudoinverse, see
        // https://mathoverflow.net/a/29511/163259
        return -Qfactored.solve(dQ(Qfactored.solve(b)))
               + Qfactored.solve(Qinv.transpose() * dQT(imprecisionMap * b))
               + (I - Qinv * Q) * dQT(Qinv.transpose() * Qfactored.solve(b));
      }

#undef dQ
#undef dQT
    }
  }
  else
  {
    // All other terms get to treat A_c as constant
    Eigen::MatrixXs innerTerms
        = A_c.transpose() * getJacobianOfMinv(world, A_c * Qinv_b, wrt);
    Eigen::MatrixXs result = -Qfactored.solve(innerTerms);

    // snapshot.restore();
    return result;
  }

  assert(false && "Execution should never reach this point.");
}

//==============================================================================
/// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
/// wrt, by finite differencing
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world,
    Eigen::VectorXs b,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfLCPConstraintMatrixClampingSubset(
        world, b, wrt);
  }

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  if (wrt != WithRespectTo::POSITION)
  {
    return jac;
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t EPS = 1e-6;
  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    Eigen::MatrixXs A_c
        = estimateClampingConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs A_ub
        = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();
    Eigen::MatrixXs Q
        = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();

    /*
    std::cout << "+" << i << ": " << A_c.cols() << " :: " << mNumClamping
              << std::endl;
              */

    Eigen::VectorXs bPlus = Q.completeOrthogonalDecomposition().solve(b);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    E = getUpperBoundMappingMatrix();
    Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();

    /*
    std::cout << "-" << i << ": " << A_c.cols() << " :: " << mNumClamping
              << std::endl;
              */

    Eigen::VectorXs bMinus = Q.completeOrthogonalDecomposition().solve(b);

    jac.col(i) = (bPlus - bMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  snapshot.restore();
  return jac;
}

//==============================================================================
/// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
/// wrt, by Ridders extrapolated finite differencing
Eigen::MatrixXs BackpropSnapshot::
    finiteDifferenceRiddersJacobianOfLCPConstraintMatrixClampingSubset(
        simulation::WorldPtr world, Eigen::VectorXs b, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  if (wrt != WithRespectTo::POSITION)
  {
    return jac;
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t originalStepSize = 1e-2;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(world.get());

  for (int i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbed = original;
    perturbed(i) += stepSize;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs A_c
        = estimateClampingConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs A_ub
        = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();
    Eigen::MatrixXs Q
        = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();
    // std::cout << "+" << i << ": " << A_c.cols() << " :: " <<
    //   mNumClamping << std::endl;
    Eigen::VectorXs QinvBPlus = Q.completeOrthogonalDecomposition().solve(b);
    perturbed = original;
    perturbed(i) -= stepSize;
    wrt->set(world.get(), perturbed);
    A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    E = getUpperBoundMappingMatrix();
    Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
    Q.diagonal() += getConstraintForceMixingDiagonal();
    // std::cout << "+" << i << ": " << A_c.cols() << " :: " <<
    //   mNumClamping << std::endl;
    Eigen::VectorXs QinvBMinus = Q.completeOrthogonalDecomposition().solve(b);

    tab[0][0] = (QinvBPlus - QinvBMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbed = original;
      perturbed(i) += stepSize;
      wrt->set(world.get(), perturbed);
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
      E = getUpperBoundMappingMatrix();
      Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
      Q.diagonal() += getConstraintForceMixingDiagonal();
      // std::cout << "+" << i << ": " << A_c.cols() << " :: " << mNumClamping
      // << std::endl;
      QinvBPlus = Q.completeOrthogonalDecomposition().solve(b);
      perturbed = original;
      perturbed(i) -= stepSize;
      wrt->set(world.get(), perturbed);
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
      E = getUpperBoundMappingMatrix();
      Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);
      Q.diagonal() += getConstraintForceMixingDiagonal();
      // std::cout << "+" << i << ": " << A_c.cols() << " :: " << mNumClamping
      // << std::endl;
      QinvBMinus = Q.completeOrthogonalDecomposition().solve(b);

      tab[0][iTab] = (QinvBPlus - QinvBMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          jac.col(i) = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), original);
  snapshot.restore();

  return jac;
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
Eigen::MatrixXs BackpropSnapshot::getJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  s_t dt = world->getTimeStep();
  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);
  Eigen::MatrixXs ddamp = getDampingVector(world).asDiagonal();
  Eigen::MatrixXs spring_stiffs = getSpringStiffVector(world).asDiagonal();
  Eigen::VectorXs p_rest = getRestPositions(world);
  if (wrt == WithRespectTo::VELOCITY)
  {
    // snapshot.restore();
    return getBounceDiagonals().asDiagonal() * -A_c.transpose()
           * (Eigen::MatrixXs::Identity(
                  world->getNumDofs(), world->getNumDofs())
              - dt * Minv * (dC+ddamp+dt*spring_stiffs));
  }
  else if (wrt == WithRespectTo::FORCE)
  {
    // snapshot.restore();
    return getBounceDiagonals().asDiagonal() * -A_c.transpose() * dt * Minv;
  }

  Eigen::VectorXs C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXs f = getPreStepTorques() - C;
  Eigen::VectorXs v_f = getPreConstraintVelocity();
  Eigen::VectorXs v_t = world->getVelocities();
  Eigen::VectorXs p_t = world->getPositions();
  Eigen::VectorXs spring_force = spring_stiffs*(p_t - p_rest+dt*v_t);
  Eigen::VectorXs damping_force = ddamp*v_t;
  f = f - damping_force - spring_force;
  Eigen::MatrixXs dMinv_f = getJacobianOfMinv(world, f, wrt);
  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dA_c_f
        = getJacobianOfClampingConstraintsTranspose(world, v_f);

    // snapshot.restore();
    return getBounceDiagonals().asDiagonal()
           * -(dA_c_f + A_c.transpose() * dt * (dMinv_f - Minv * dC - Minv*spring_stiffs));
  }
  else
  {
    // snapshot.restore();
    return getBounceDiagonals().asDiagonal()
           * -(A_c.transpose() * dt * (dMinv_f - Minv * dC));
  }
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
/// finite differencing
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfLCPOffsetClampingSubset(world, wrt);
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);

  const s_t EPS = 1e-7;
  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs bPlus;
    Eigen::VectorXs bMinus;

    s_t epsPos = EPS;
    while (true)
    {
      Eigen::VectorXs perturbed = original;
      perturbed(i) += epsPos;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr posSnapshot = neural::forwardPass(world, true);
      bPlus = posSnapshot->getClampingConstraintRelativeVels();
      if (bPlus.size() == mNumClamping)
      {
        break;
      }

      epsPos *= 0.5;
      assert(epsPos > 1e-25);
    }

    s_t epsNeg = EPS;
    while (true)
    {
      Eigen::VectorXs perturbed = original;
      perturbed(i) -= epsNeg;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr negSnapshot = neural::forwardPass(world, true);
      bMinus = negSnapshot->getClampingConstraintRelativeVels();
      if (bMinus.size() == mNumClamping)
      {
        break;
      }

      epsNeg *= 0.5;
      assert(epsNeg > 1e-25);
    }

    jac.col(i) = (bPlus - bMinus) / (epsPos + epsNeg);
  }
  wrt->set(world.get(), original);

  snapshot.restore();
  return jac;
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
/// Ridders extrapolated finite differencing
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs bPlus;
    Eigen::VectorXs bMinus;

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(original);
      perturbedPlus(i) += originalStepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      bPlus = snapshotPlus->getClampingConstraintRelativeVels();
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(original);
      perturbedMinus(i) -= originalStepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      bMinus = snapshotMinus->getClampingConstraintRelativeVels();

      if (bPlus.size() == mNumClamping && bMinus.size() == mNumClamping)
      {
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-25);
    }
    tab[0][0] = (bPlus - bMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(original);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      bPlus = snapshotPlus->getClampingConstraintRelativeVels();
      if (!(bPlus.size() == mNumClamping))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfLCPOffsetClampingSubset() "
  "caused bPlus.size() to change.");
      }
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(original);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      bMinus = snapshotMinus->getClampingConstraintRelativeVels();
      if (!(bMinus.size() == mNumClamping))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfLCPOffsetClampingSubset() "
  "caused bPlus.size() to change.");
      }

      tab[0][iTab] = (bPlus - bMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), original);

  snapshot.restore();
  return J;
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
/// finite differencing
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfLCPEstimatedOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfLCPEstimatedOffsetClampingSubset(
        world, wrt);
  }

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t EPS = 1e-8;
  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs bPlus = Eigen::VectorXs::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bPlus, A_c);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs bMinus = Eigen::VectorXs::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bMinus, A_c);

    jac.col(i) = (bPlus - bMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  snapshot.restore();

  return jac;
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
/// Ridders extrapolated finite differencing
Eigen::MatrixXs BackpropSnapshot::
    finiteDifferenceRiddersJacobianOfLCPEstimatedOffsetClampingSubset(
        simulation::WorldPtr world, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);
  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(original);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs bPlus = Eigen::VectorXs::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bPlus, A_c);

    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(original);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs bMinus = Eigen::VectorXs::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bMinus, A_c);

    tab[0][0] = (bPlus - bMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(original);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      if (wrt == WithRespectTo::POSITION)
      {
        A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      }
      bPlus = Eigen::VectorXs::Zero(mNumClamping);
      computeLCPOffsetClampingSubset(world, bPlus, A_c);

      perturbedMinus = Eigen::VectorXs(original);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      if (wrt == WithRespectTo::POSITION)
      {
        A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      }
      bMinus = Eigen::VectorXs::Zero(mNumClamping);
      computeLCPOffsetClampingSubset(world, bMinus, A_c);

      tab[0][iTab] = (bPlus - bMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), original);

  snapshot.restore();

  return J;
}

//==============================================================================
/// This returns the subset of the A matrix used by the original LCP for just
/// the clamping constraints. It relates constraint force to constraint
/// acceleration. It's a mass matrix, just in a weird frame.
void BackpropSnapshot::computeLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world,
    Eigen::MatrixXs& Q,
    const Eigen::MatrixXs& A_c,
    const Eigen::MatrixXs& A_ub,
    const Eigen::MatrixXs& E)
{
  /*
  int numClamping = A_c.cols();
  for (int i = 0; i < numClamping; i++)
  {
    Q.col(i)
        = A_c.transpose() * implicitMultiplyByInvMassMatrix(world, A_c.col(i));
  }
  */
  if (A_ub.cols() > 0)
  {
    Q = A_c.transpose() * getInvMassMatrix(world, true) * (A_c + A_ub * E);
  }
  else
  {
    Q = A_c.transpose() * getInvMassMatrix(world, true) * A_c;
  }
  Q.diagonal() += getConstraintForceMixingDiagonal();
}

//==============================================================================
/// This returns the subset of the b vector used by the original LCP for just
/// the clamping constraints. It's just the relative velocity at the clamping
/// contact points.
void BackpropSnapshot::computeLCPOffsetClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXs& b, const Eigen::MatrixXs& A_c)
{
  /*
  Eigen::VectorXs velDiff = world->getVelocities() - mPreStepVelocity;
  b = getClampingConstraintRelativeVels() + A_c.transpose() * velDiff;
  */

  /*
  b = -A_c.transpose()
      * (world->getVelocities() + getVelocityDueToIllegalImpulses());
  */

  /*
  b = -A_c.transpose()
      * (getPreConstraintVelocity() + getVelocityDueToIllegalImpulses());
  */
  s_t dt = world->getTimeStep();
  Eigen::MatrixXs damp = getDampingVector(world).asDiagonal();
  Eigen::MatrixXs spring_stiffs = getSpringStiffVector(world).asDiagonal();
  Eigen::VectorXs p_t = world->getPositions();
  Eigen::VectorXs v_t = world->getVelocities();
  Eigen::VectorXs p_rest = getRestPositions(world);
  Eigen::VectorXs damping_force = damp*world->getVelocities();
  Eigen::VectorXs spring_force = spring_stiffs*(p_t-p_rest+dt*v_t);
  b = -getBounceDiagonals().cwiseProduct(
      A_c.transpose()
      * (v_t
         + (dt
            * implicitMultiplyByInvMassMatrix(
                world,
                world->getControlForces()
                    - world->getCoriolisAndGravityAndExternalForces()
                    - damping_force - spring_force))));
}

//==============================================================================
/// This computes and returns an estimate of the constraint impulses for the
/// clamping constraints. This is based on a linear approximation of the
/// constraint impulses.
Eigen::VectorXs BackpropSnapshot::estimateClampingConstraintImpulses(
    simulation::WorldPtr world,
    const Eigen::MatrixXs& A_c,
    const Eigen::MatrixXs& A_ub,
    const Eigen::MatrixXs& E)
{
  if (A_c.cols() == 0)
  {
    return Eigen::VectorXs::Zero(0);
  }

  Eigen::VectorXs b = Eigen::VectorXs::Zero(A_c.cols());
  Eigen::MatrixXs Q = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c, A_ub, E);

  // Q can be low rank, but we resolve ambiguity during the forward pass by
  // taking the least-squares minimal solution
  return Q.completeOrthogonalDecomposition().solve(b);
}

//==============================================================================
/// This returns the jacobian of P_c * v, holding everyhing constant except
/// the value of WithRespectTo
Eigen::MatrixXs BackpropSnapshot::getJacobianOfProjectionIntoClampsMatrix(
    simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt)
{
  // return finiteDifferenceJacobianOfProjectionIntoClampsMatrix(world, v, wrt);

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  if (A_c.size() == 0)
    return Eigen::MatrixXs::Zero(0, world->getNumDofs());
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();

  Eigen::MatrixXs V_c = getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXs V_ub = getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXs A_c_ub_E = A_c + (A_ub * E);
  Eigen::MatrixXs Q = A_c.eval().transpose() * constraintForceToImpliedTorques;
  auto XFactor = Q.completeOrthogonalDecomposition();
  Eigen::MatrixXs bounce = getBounceDiagonals().asDiagonal();

  // New formulation
  if (wrt == WithRespectTo::POSITION)
  {
    // d/d Q^{-1} v = - Q^{-1} (d/d Q) Q^{-1} v
    Eigen::MatrixXs rightHandSide = bounce * A_c.transpose();
    Eigen::MatrixXs dRhs
        = bounce * getJacobianOfClampingConstraintsTranspose(world, v);
    Eigen::MatrixXs Minv = getInvMassMatrix(world);

    Eigen::MatrixXs Qinv = XFactor.pseudoInverse();
    Eigen::VectorXs Qinv_v = XFactor.solve(rightHandSide * v);
    Eigen::MatrixXs dQ
        = getJacobianOfClampingConstraintsTranspose(
              world, Minv * A_c_ub_E * Qinv_v)
          + A_c.transpose()
                * (getJacobianOfMinv(world, A_c_ub_E * Qinv_v, wrt)
                   + Minv * getJacobianOfClampingConstraints(world, Qinv_v));

    return (1 / world->getTimeStep())
           * (XFactor.solve(dRhs) - XFactor.solve(dQ));
  }
  else
  {
    // Ignore changes to A_c

    // Approximate the pseudo-inverse as just a plain inverse for the purposes
    // of derivation

    Eigen::VectorXs tau
        = A_c_ub_E * XFactor.solve(bounce * A_c.transpose() * v);

    Eigen::MatrixXs MinvJac = getJacobianOfMinv(world, tau, wrt);

    return -(1.0 / world->getTimeStep())
           * XFactor.solve(A_c.transpose() * MinvJac);
  }

  // An older approach that attempted to handle pseudoinverse distinct from
  // normal inverse

  /*
  Eigen::MatrixXs X = XFactor.pseudoInverse();
  Eigen::VectorXs A_c_T_V = bounce * A_c.transpose() * v;

  // Part 1

  Eigen::VectorXs part1Tau = A_c * X.transpose() * X * A_c_T_V;
  Eigen::MatrixXs part1MinvJac = getJacobianOfMinv(world, part1Tau, wrt);
  Eigen::MatrixXs XQ = X * Q;
  Eigen::MatrixXs part1 = (Eigen::MatrixXs::Identity(XQ.rows(), XQ.cols()) + XQ)
                          * A_c_ub_E.transpose() * part1MinvJac;

  Eigen::MatrixXs QX = Q * X;
  Eigen::VectorXs part2Tau
      = A_c * (Eigen::MatrixXs::Identity(QX.rows(), QX.cols()) - QX) * A_c_T_V;
  Eigen::MatrixXs part2MinvJac = getJacobianOfMinv(world, part2Tau, wrt);
  Eigen::MatrixXs part2
      = X * X.transpose() * A_c_ub_E.transpose() * part2MinvJac;

  Eigen::VectorXs part3Tau = A_c_ub_E * X * A_c_T_V;
  Eigen::MatrixXs part3MinvJac = getJacobianOfMinv(world, part3Tau, wrt);
  Eigen::MatrixXs part3 = X * A_c.transpose() * part3MinvJac;

  return (1.0 / mTimeStep) * (part1 + part2 - part3);
  */
}

//==============================================================================
/// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
/// everything constant except the value of WithRespectTo
Eigen::MatrixXs BackpropSnapshot::getJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXs tau, WithRespectTo* wrt)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  if (wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::VELOCITY
      || wrt == neural::WithRespectTo::FORCE)
  {
    Eigen::MatrixXs jac
        = Eigen::MatrixXs::Zero(world->getNumDofs(), wrt->dim(world.get()));
    int cursor = 0;
    for (int i = 0; i < world->getNumSkeletons(); i++)
    {
      auto skel = world->getSkeleton(i);
      int dofs = skel->getNumDofs();
      jac.block(cursor, cursor, dofs, dofs)
          = skel->getJacobianOfMinv(tau.segment(cursor, dofs), wrt);
      cursor += dofs;
    }

#ifndef NDEBUG
    assert(
        world->getPositions() == mPreStepPosition
        && world->getVelocities() == mPreStepVelocity);
#endif

    return jac;
  }
  else
  {
    return finiteDifferenceJacobianOfMinv(world, tau, wrt);
  }
}

//==============================================================================
/// This returns the jacobian of C(pos, inertia, vel), holding everything
/// constant except the value of WithRespectTo
Eigen::MatrixXs BackpropSnapshot::getJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
#ifndef NDEBUG
  assert(
      world->getPositions() == mPreStepPosition
      && world->getVelocities() == mPreStepVelocity);
#endif

  if (wrt == WithRespectTo::POSITION)
  {
    if (mCachedPosCDirty)
    {
      mCachedPosC = computeJacobianOfC(world, WithRespectTo::POSITION);
      mCachedPosCDirty = false;
    }
    return mCachedPosC;
  }
  if (wrt == WithRespectTo::VELOCITY)
  {
    if (mCachedVelCDirty)
    {
      mCachedVelC = computeJacobianOfC(world, WithRespectTo::VELOCITY);
      mCachedVelCDirty = false;
    }
    return mCachedVelC;
  }
  return computeJacobianOfC(world, wrt);
}

//==============================================================================
/// This returns the jacobian of C(pos, inertia, vel), holding everything
/// constant except the value of WithRespectTo
Eigen::MatrixXs BackpropSnapshot::computeJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t wrtDim = wrt->dim(world.get());

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumDOFs, wrtDim);

  int wrtCursor = 0;
  int dofCursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    int dofs = skel->getNumDofs();
    int skelWrtDim = wrt->dim(skel.get());
    J.block(dofCursor, wrtCursor, dofs, skelWrtDim) = skel->getJacobianOfC(wrt);
    wrtCursor += skelWrtDim;
    dofCursor += dofs;
  }

  return J;
}

/// This returns the jacobian of M^{-1}(pos, inertia) * (C(pos, inertia, vel) +
/// mPreStepTorques), holding everything constant except the value of
/// WithRespectTo
Eigen::MatrixXs BackpropSnapshot::getJacobianOfMinvC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfMinvC(world, wrt);
}

//==============================================================================
/// This returns a fast approximation to A_c in the neighborhood of the original
Eigen::MatrixXs BackpropSnapshot::estimateClampingConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  Eigen::VectorXs posDiff = pos - mPreStepPosition;
  if (posDiff.squaredNorm() == 0)
  {
    return getClampingConstraintMatrix(world);
  }
  Eigen::VectorXs oldPos = world->getPositions();
  world->setPositions(mPreStepPosition);

  auto clampingConstraints = getClampingConstraints();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumDOFs, mNumClamping);
  for (int i = 0; i < clampingConstraints.size(); i++)
  {
    auto constraint = clampingConstraints[i];
    /*
    std::cout << "Constraint forces jac " << i << "=" << std::endl
              << constraint->getConstraintForcesJacobian(world) << std::endl
              << "*" << std::endl
              << posDiff << std::endl
              << "=" << std::endl
              << (constraint->getConstraintForcesJacobian(world) * posDiff)
              << std::endl;
    */
    result.col(i) = constraint->getConstraintForces(world.get())
                    + constraint->getConstraintForcesJacobian(world) * posDiff;
  }

  world->setPositions(oldPos);
  return result;
}

//==============================================================================
/// This returns a fast approximation to A_ub in the neighborhood of the
/// original
Eigen::MatrixXs BackpropSnapshot::estimateUpperBoundConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  Eigen::VectorXs posDiff = pos - mPreStepPosition;
  if (posDiff.squaredNorm() == 0)
  {
    return getUpperBoundConstraintMatrix(world);
  }
  Eigen::VectorXs oldPos = world->getPositions();
  world->setPositions(mPreStepPosition);

  auto upperBoundConstraints = getUpperBoundConstraints();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumDOFs, mNumUpperBound);
  for (int i = 0; i < upperBoundConstraints.size(); i++)
  {
    auto constraint = upperBoundConstraints[i];
    result.col(i) = constraint->getConstraintForces(world.get())
                    + constraint->getConstraintForcesJacobian(world) * posDiff;
  }

  world->setPositions(oldPos);
  return result;
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of A_c at the
/// desired position.
Eigen::MatrixXs BackpropSnapshot::getClampingConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  Eigen::MatrixXs bruteResult = ptr->getClampingConstraintMatrix(world);
  return bruteResult;
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of A_ub at the
/// desired position.
Eigen::MatrixXs BackpropSnapshot::getUpperBoundConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getUpperBoundConstraintMatrix(world);
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of E at the
/// desired position.
Eigen::MatrixXs BackpropSnapshot::getUpperBoundMappingMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getUpperBoundMappingMatrix();
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of the bounce
/// diagonals at the desired position.
Eigen::VectorXs BackpropSnapshot::getBounceDiagonalsAt(
    simulation::WorldPtr world, Eigen::VectorXs pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getBounceDiagonals();
}

//==============================================================================
/// This computes the Jacobian of A_c*f0 with respect to `wrt` using impulse
/// tests.
Eigen::MatrixXs BackpropSnapshot::getJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0)
{
  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  int dofs = world->getNumDofs();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(dofs, dofs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(world);
  }

  // snapshot.restore();
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_c^T*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXs BackpropSnapshot::getJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0)
{
  /*
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  */

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  int dofs = world->getNumDofs();
  assert(constraints.size() == mNumClamping);
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumClamping, dofs);
  for (int i = 0; i < constraints.size(); i++)
  {
    result.row(i)
        = constraints[i]->getConstraintForcesJacobian(world).transpose() * v0;
  }

  // snapshot.restore();
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub*E*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXs BackpropSnapshot::getJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXs E_f0)
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  int dofs = world->getNumDofs();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(dofs, dofs);
  assert(constraints.size() == E_f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += E_f0(i) * constraints[i]->getConstraintForcesJacobian(world);
  }
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub^T*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXs BackpropSnapshot::getJacobianOfUpperBoundConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0)
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  int dofs = world->getNumDofs();
  assert(constraints.size() == mNumUpperBound);
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumUpperBound, dofs);
  for (int i = 0; i < constraints.size(); i++)
  {
    result.row(i)
        = constraints[i]->getConstraintForcesJacobian(world).transpose() * v0;
  }

  return result;
}

//==============================================================================
/// This computes the finite difference Jacobian of A_c*f0 with respect to
/// position
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfClampingConstraints(world, f0);

  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXs original = getClampingConstraintMatrix(world) * f0;
  Eigen::MatrixXs originalA_c = getClampingConstraintMatrix(world);

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  const s_t EPS = 5e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    s_t posEPS = EPS;
    Eigen::VectorXs perturbedResultPos = Eigen::VectorXs::Zero(0);
    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 40; j++)
    {
      snapshot.restore();
      Eigen::VectorXs perturbed = mPreStepPosition;
      perturbed(i) += posEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      Eigen::MatrixXs perturbedA_c = ptr->getClampingConstraintMatrix(world);
      if (perturbedA_c.cols() == f0.size())
      {
        perturbedResultPos = perturbedA_c * f0;
        s_t squaredNorm = (original - perturbedResultPos).squaredNorm();
        if (squaredNorm < 100 * posEPS)
        {
          break;
        }
        else
        {
          std::cout << "Result diff at " << posEPS << std::endl
                    << (original - perturbedResultPos) << std::endl;
          std::cout << "A_c original at " << posEPS << std::endl
                    << originalA_c << std::endl;
          std::cout << "A_c diff at " << posEPS << std::endl
                    << (originalA_c - perturbedA_c) << std::endl;
          assert(false && "Encountered too large a jump in finiteDifferenceJacobianOfClampingConstraints()");
        }
      }
      posEPS /= 2;
    }

    s_t negEPS = EPS;
    Eigen::VectorXs perturbedResultNeg = Eigen::VectorXs::Zero(0);
    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 40; j++)
    {
      snapshot.restore();
      Eigen::VectorXs perturbed = mPreStepPosition;
      perturbed(i) -= negEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      Eigen::MatrixXs perturbedA_c = ptr->getClampingConstraintMatrix(world);
      if (perturbedA_c.cols() == f0.size())
      {
        perturbedResultNeg = perturbedA_c * f0;
        s_t squaredNorm = (original - perturbedResultNeg).squaredNorm();
        if (squaredNorm < 100 * negEPS)
        {
          break;
        }
        else
        {
          std::cout << "Result diff at -" << negEPS << std::endl
                    << (original - perturbedResultNeg) << std::endl;
          std::cout << "A_c original at -" << negEPS << std::endl
                    << originalA_c << std::endl;
          std::cout << "A_c diff at -" << negEPS << std::endl
                    << (originalA_c - perturbedA_c) << std::endl;
          assert(false && "Encountered too large a jump in finiteDifferenceJacobianOfClampingConstraints()");
        }
      }
      negEPS /= 2;
    }

    /*
    std::cout << "i=" << i << " -> (" << posEPS << "," << negEPS << ")"
              << std::endl;
    */
    result.col(i)
        = (perturbedResultPos - perturbedResultNeg) / (posEPS + negEPS);
  }

  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes the finite difference Jacobian of A_c*f0 with respect to
/// position by Ridders extrapolated finite differences
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXs original = getClampingConstraintMatrix(world) * f0;

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs A_c_f0Plus;
    Eigen::VectorXs A_c_f0Minus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += originalStepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_cPlus
          = snapshotPlus->getClampingConstraintMatrix(world);

      Eigen::VectorXs perturbedMinus = mPreStepPosition;
      perturbedMinus(i) -= originalStepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_cMinus
          = snapshotMinus->getClampingConstraintMatrix(world);

      if (A_cPlus.cols() == f0.size() && A_cMinus.cols() == f0.size())
      {
        A_c_f0Plus = A_cPlus * f0;
        s_t squaredNormPlus = (original - A_c_f0Plus).squaredNorm();
        A_c_f0Minus = A_cMinus * f0;
        s_t squaredNormMinus = (original - A_c_f0Minus).squaredNorm();
        if (squaredNormPlus < 100 * originalStepSize
            && squaredNormMinus < 100 * originalStepSize)
        {
          break;
        }
        else
        {
          assert(false && "Encountered too large a jump in "
                 "finiteDifferenceRiddersJacobianOfClampingConstraints()");
        }
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    tab[0][0] = (A_c_f0Plus - A_c_f0Minus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_cPlus
          = snapshotPlus->getClampingConstraintMatrix(world);

      Eigen::VectorXs perturbedMinus = mPreStepPosition;
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_cMinus
          = snapshotMinus->getClampingConstraintMatrix(world);

      if (!(A_cPlus.cols() == f0.size() && A_cMinus.cols() == f0.size()))
      {
        /*
        neural::forwardPass(world, true);
        std::vector<std::shared_ptr<DifferentiableContactConstraint>>
            originalConstraints = getDifferentiableConstraints();
        for (int k = 0; k < originalConstraints.size(); k++)
        {
          std::shared_ptr<DifferentiableContactConstraint> constraint
              = originalConstraints[k];
          std::cout << "Original Constraint " << k << " type "
                    << constraint->getContactType() << "\npos:\n"
                    << constraint->getContactWorldPosition() << "\nnormal:\n"
                    << constraint->getContactWorldNormal() << std::endl;
        }
        std::vector<std::shared_ptr<DifferentiableContactConstraint>>
            plusConstraints = snapshotPlus->getDifferentiableConstraints();
        for (int k = 0; k < plusConstraints.size(); k++)
        {
          std::shared_ptr<DifferentiableContactConstraint> constraint
              = plusConstraints[k];
          std::cout << "Plus Constraint " << k << " type "
                    << constraint->getContactType() << "\npos:\n"
                    << constraint->getContactWorldPosition() << "\nnormal:\n"
                    << constraint->getContactWorldNormal() << std::endl;
        }
        std::vector<std::shared_ptr<DifferentiableContactConstraint>>
            minusConstraints = snapshotMinus->getDifferentiableConstraints();
        for (int k = 0; k < minusConstraints.size(); k++)
        {
          std::shared_ptr<DifferentiableContactConstraint> constraint
              = minusConstraints[k];
          std::cout << "Minus Constraint " << k << " type "
                    << constraint->getContactType() << "\npos:\n"
                    << constraint->getContactWorldPosition() << "\nnormal:\n"
                    << constraint->getContactWorldNormal() << std::endl;
        }
        */
        assert(false && 
        "Lowering EPS in finiteDifferenceRiddersJacobianOfClampingConstraints()"
          "caused A_c.cols() to change.");
      }
      A_c_f0Plus = A_cPlus * f0;
      A_c_f0Minus = A_cMinus * f0;
      /*
      if (i == 0 && abs(A_c_f0Minus(2) - original(2)) > stepSize)
      {
        std::cout << "Error of " << A_c_f0Minus(2) - original(2)
                  << " on [0] -= " << stepSize << std::endl;
      }
      if (i == 0 && abs(A_c_f0Plus(2) - original(2)) > stepSize)
      {
        std::cout << "*********" << std::endl;
        std::cout << "Error of " << A_c_f0Plus(2) - original(2)
                  << " on [0] += " << stepSize << std::endl;
        for (int k = 0; k < f0.size(); k++)
        {
          std::shared_ptr<DifferentiableContactConstraint> originalConstraint
              = getDifferentiableConstraints()[k];
          Eigen::Vector3s posGrad
              = originalConstraint->getContactPositionGradient(
                  world->getDofs()[0]);
          Eigen::Vector3s dirGrad
              = originalConstraint->getContactNormalGradient(
                  world->getDofs()[0]);
          collision::Contact& originalContact
              = originalConstraint->getContact();
          std::shared_ptr<DifferentiableContactConstraint> constraint
              = snapshotPlus->getDifferentiableConstraints()[k];
          collision::Contact& contact = constraint->getContact();
          // constraint->getContactWorldForceDirection()
          std::cout << "Contact: " << k << ", type: " << contact.type
                    << std::endl;
          if (contact.type != collision::ContactType::PIPE_EDGE)
          {
            std::cout << "Pos grad: " << posGrad << std::endl;
            std::cout << "Dir grad: " << dirGrad << std::endl;
            if (originalContact.type != contact.type)
            {
              std::cout << "Original type: " << originalContact.type
                        << std::endl;
              std::cout << "New type: " << contact.type << std::endl;
            }
            if (originalContact.point != contact.point)
            {
              std::cout << "Original pt: " << originalContact.point
                        << std::endl;
              std::cout << "New pt: " << contact.point << std::endl;
              std::cout << "Diff: " << contact.point - originalContact.point
                        << std::endl;
            }
            if (originalContact.normal != contact.normal)
            {
              std::cout << "Original normal: " << originalContact.normal
                        << std::endl;
              std::cout << "New normal: " << contact.normal << std::endl;
              std::cout << "Diff: " << contact.normal - originalContact.normal
                        << std::endl;
            }
          }
        }
      }
      */

      tab[0][iTab] = (A_c_f0Plus - A_c_f0Minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes the finite difference Jacobian of A_c^T*v0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfClampingConstraintsTranspose(
        world, v0);
  }

  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXs original
      = getClampingConstraintMatrix(world).transpose() * v0;

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  const s_t EPS = 5e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    s_t posEPS = EPS;
    Eigen::VectorXs perturbedResultPos;

    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 10; j++)
    {
      snapshot.restore();
      Eigen::VectorXs perturbed = mPreStepPosition;
      perturbed(i) += posEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      perturbedResultPos
          = ptr->getClampingConstraintMatrix(world).transpose() * v0;
      if (perturbedResultPos.size() == original.size())
      {
        break;
      }
      posEPS /= 2;
    }

    s_t negEPS = EPS;
    Eigen::VectorXs perturbedResultNeg;

    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 10; j++)
    {
      snapshot.restore();
      Eigen::VectorXs perturbed = mPreStepPosition;
      perturbed(i) -= negEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setControlForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      perturbedResultNeg
          = ptr->getClampingConstraintMatrix(world).transpose() * v0;
      if (perturbedResultNeg.size() == original.size())
      {
        break;
      }
      negEPS /= 2;
    }

    result.col(i)
        = (perturbedResultPos - perturbedResultNeg) / (posEPS + negEPS);
  }

  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes the finite difference Jacobian of A_c^T*v0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXs test = getClampingConstraintMatrix(world);
  Eigen::VectorXs original
      = getClampingConstraintMatrix(world).transpose() * v0;

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  if (original.size() == 0)
  {
    return J;
  }

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs A_c_T_v0Plus;
    Eigen::VectorXs A_c_T_v0Minus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += originalStepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      A_c_T_v0Plus
          = snapshotPlus->getClampingConstraintMatrix(world).transpose() * v0;

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= originalStepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      A_c_T_v0Minus
          = snapshotMinus->getClampingConstraintMatrix(world).transpose() * v0;

      if (A_c_T_v0Plus.size() == original.size()
          && A_c_T_v0Minus.size() == original.size())
      {
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    tab[0][0] = (A_c_T_v0Plus - A_c_T_v0Minus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_c_T_v0Plus
          = snapshotPlus->getClampingConstraintMatrix(world).transpose() * v0;

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_c_T_v0Minus
          = snapshotMinus->getClampingConstraintMatrix(world).transpose() * v0;

      if (!(A_c_T_v0Plus.size() == original.size()
            && A_c_T_v0Minus.size() == original.size()))
      {
        assert(false && 
    "Lowering EPS in finiteDifferenceRiddersJacobianOfClampingConstraintsTranspose()"
    "caused A_c_T_v0.size() to change.");
      }

      tab[0][iTab] = (A_c_T_v0Plus - A_c_T_v0Minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();

  return J;
}

/// This computes the finite difference Jacobian of A_ub*E*f0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfUpperBoundConstraints(world, f0);
  }

  if (mNumUpperBound == 0)
  {
    return Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  }

  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  /*
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();
  std::cout << "A_ub size: " << A_ub.rows() << "x" << A_ub.cols() << std::endl;
  std::cout << "E size: " << E.rows() << "x" << E.cols() << std::endl;
  std::cout << "f0 size: " << f0.size() << std::endl;
  */

  Eigen::VectorXs original = A_ub * f0;

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  const s_t EPS = 1e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    snapshot.restore();
    Eigen::VectorXs perturbed = mPreStepPosition;
    perturbed(i) += EPS;
    world->setPositions(perturbed);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
    Eigen::VectorXs perturbedResultPos
        = ptr->getUpperBoundConstraintMatrix(world) * f0;

    perturbed = mPreStepPosition;
    perturbed(i) -= EPS;
    world->setPositions(perturbed);
    world->setVelocities(mPreStepVelocity);
    world->setControlForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    ptr = neural::forwardPass(world, true);
    Eigen::VectorXs perturbedResultNeg
        = ptr->getUpperBoundConstraintMatrix(world) * f0;

    result.col(i) = (perturbedResultPos - perturbedResultNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

/// This computes the finite difference Jacobian of A_ub*E*f0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice. Uses Ridders method.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0)
{
  if (mNumUpperBound == 0)
  {
    return Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  }

  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXs originalA_ub = getUpperBoundConstraintMatrix(world);
  Eigen::VectorXs original = originalA_ub * f0;

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), mNumDOFs);

  s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXs A_ub_f0Plus;
    Eigen::VectorXs A_ub_f0Minus;

    // Find largest original step size which doesn't change numUpperBound
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += originalStepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_ubPlus
          = snapshotPlus->getUpperBoundConstraintMatrix(world);

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= originalStepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_ubMinus
          = snapshotMinus->getUpperBoundConstraintMatrix(world);

      if (A_ubPlus.size() == originalA_ub.size()
          && A_ubMinus.size() == originalA_ub.size())
      {
        A_ub_f0Plus = A_ubPlus * f0;
        A_ub_f0Minus = A_ubMinus * f0;
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    tab[0][0] = (A_ub_f0Plus - A_ub_f0Minus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_ubPlus
          = snapshotPlus->getUpperBoundConstraintMatrix(world);
      if (!(A_ubPlus.size() == originalA_ub.size()))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfUpperBoundConstraints() "
  "caused A_ub.size() to change.");
      }
      A_ub_f0Plus = A_ubPlus * f0;

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(mPreStepPosition);
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs A_ubMinus
          = snapshotMinus->getUpperBoundConstraintMatrix(world);
      if (!(A_ubMinus.size() == originalA_ub.size()))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfUpperBoundConstraints() "
  "caused A_ub.size() to change.");
      }
      A_ub_f0Minus = A_ubMinus * f0;

      tab[0][iTab] = (A_ub_f0Plus - A_ub_f0Minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the jacobian of P_c * v by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
    simulation::WorldPtr world,
    Eigen::VectorXs v,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfProjectionIntoClampsMatrix(
        world, v, wrt);
  }

  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXs before = wrt->get(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = getProjectionIntoClampsMatrix(world, true) * v;

  Eigen::MatrixXs originalP_c = getProjectionIntoClampsMatrix(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getDifferentiableConstraints();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  const s_t EPS = 1e-5;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    s_t posEps = EPS;

    Eigen::VectorXs newPlus;
    Eigen::VectorXs newMinus;

    while (true)
    {
      perturbed = before;
      perturbed(i) += posEps;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr plusBackptr = neural::forwardPass(world, true);
      Eigen::MatrixXs newP_c
          = plusBackptr->getProjectionIntoClampsMatrix(world);
      if (newP_c.rows() == originalP_c.rows())
      {
        newPlus = newP_c * v;
        break;
      }
      posEps *= 0.5;
    }

    perturbed = before;
    s_t negEps = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) -= negEps;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr negBackptr = neural::forwardPass(world, true);

      Eigen::MatrixXs newP_c = getProjectionIntoClampsMatrix(world, true);
      if (newP_c.rows() == originalP_c.rows())
      {
        newMinus = newP_c * v;
        break;
      }
      negEps *= 0.5;
    }

    Eigen::VectorXs diff = newPlus - newMinus;
    result.col(i) = diff / (posEps + negEps);
  }

  wrt->set(world.get(), before);
  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of P_c * v by Ridders extrapolated
/// finite differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfProjectionIntoClampsMatrix(
    simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs originalP_c_v
      = getProjectionIntoClampsMatrix(world, true) * v;
  Eigen::MatrixXs originalP_c = getProjectionIntoClampsMatrix(world, true);

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(originalP_c_v.size(), innerDim);

  s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs P_c_vPlus;
    Eigen::VectorXs P_c_vMinus;

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += originalStepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs P_cPlus
          = snapshotPlus->getProjectionIntoClampsMatrix(world);

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= originalStepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs P_cMinus
          = snapshotMinus->getProjectionIntoClampsMatrix(world);
      if (P_cPlus.rows() == originalP_c.rows()
          && P_cMinus.rows() == originalP_c.rows())
      {
        P_c_vPlus = P_cPlus * v;
        P_c_vMinus = P_cMinus * v;
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }

    tab[0][0] = (P_c_vPlus - P_c_vMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::MatrixXs P_cPlus
          = snapshotPlus->getProjectionIntoClampsMatrix(world);
      if (!(P_cPlus.rows() == originalP_c.rows()))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfProjectionIntoClampsMatrix() "
  "caused P_c.rows() to change.");
      }
      P_c_vPlus = P_cPlus * v;

      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      Eigen::MatrixXs P_cMinus
          = snapshotMinus->getProjectionIntoClampsMatrix(world);
      if (!(P_cMinus.rows() == originalP_c.rows()))
      {
        assert(false && 
  "Lowering EPS in finiteDifferenceRiddersJacobianOfProjectionIntoClampsMatrix() "
  "caused P_c.rows() to change.");
      }
      P_c_vMinus = P_cMinus * v;

      tab[0][iTab] = (P_c_vPlus - P_c_vMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// finite differences.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfMinv(
    simulation::WorldPtr world,
    Eigen::VectorXs tau,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMinv(world, tau, wrt);

  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs before = wrt->get(world.get());

  const s_t EPS = 5e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs newVPlus = implicitMultiplyByInvMassMatrix(world, tau);
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs newVMinus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXs diff = newVPlus - newVMinus;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);
  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// Ridders extrapolated finite differences.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXs tau, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    Eigen::MatrixXs MinvTauPlus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    Eigen::MatrixXs MinvTauMinus = implicitMultiplyByInvMassMatrix(world, tau);

    tab[0][0] = (MinvTauPlus - MinvTauMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      MinvTauPlus = implicitMultiplyByInvMassMatrix(world, tau);
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      MinvTauMinus = implicitMultiplyByInvMassMatrix(world, tau);

      tab[0][iTab] = (MinvTauPlus - MinvTauMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This returns the jacobian of M(pos, inertia) * v, holding
/// everything constant except the value of WithRespectTo
Eigen::MatrixXs BackpropSnapshot::getJacobianOfM(
    simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt)
{
  Eigen::MatrixXs J
      = Eigen::MatrixXs::Zero(world->getNumDofs(), wrt->dim(world.get()));
  int dofCursor = 0;
  int wrtCursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    int dofs = skel->getNumDofs();
    int wrts = wrt->dim(skel.get());
    J.block(dofCursor, wrtCursor, dofs, wrts)
        = skel->getJacobianOfM(v.segment(dofCursor, dofs), wrt);
    dofCursor += dofs;
    wrtCursor += wrts;
  }
  return J;
}

//==============================================================================
/// This computes and returns the jacobian of M(pos, inertia) * v by
/// finite differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfM(
    simulation::WorldPtr world,
    Eigen::VectorXs v,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfM(world, v, wrt);

  RestorableSnapshot snapshot(world);

  std::size_t wrtDim = wrt->dim(world.get());

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(world->getNumDofs(), wrtDim);

  Eigen::VectorXs before = wrt->get(world.get());

  const s_t EPS = 5e-7;

  for (std::size_t i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs newVPlus = world->getMassMatrix() * v;
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs newVMinus = world->getMassMatrix() * v;
    Eigen::VectorXs diff = newVPlus - newVMinus;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);
  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M(pos, inertia) * v by
/// Ridders extrapolated finite differences. This is SUPER SLOW, and is
/// only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersJacobianOfM(
    simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  std::size_t wrtDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(world->getNumDofs(), wrtDim);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    Eigen::MatrixXs plus = world->getMassMatrix() * v;

    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    Eigen::MatrixXs minus = world->getMassMatrix() * v;

    tab[0][0] = (plus - minus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      plus = world->getMassMatrix() * v;
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      minus = world->getMassMatrix() * v;

      tab[0][iTab] = (plus - minus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by finite
/// differences.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfC(world, wrt);

  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = world->getCoriolisAndGravityAndExternalForces();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs before = wrt->get(world.get());

  const s_t EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs tauPos = world->getCoriolisAndGravityAndExternalForces();
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs tauNeg = world->getCoriolisAndGravityAndExternalForces();
    Eigen::VectorXs diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);
  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by
/// Ridders extrapolated finite differences.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = world->getCoriolisAndGravityAndExternalForces();

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    Eigen::MatrixXs tauPlus = world->getCoriolisAndGravityAndExternalForces();
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    Eigen::MatrixXs tauMinus = world->getCoriolisAndGravityAndExternalForces();

    tab[0][0] = (tauPlus - tauMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      tauPlus = world->getCoriolisAndGravityAndExternalForces();
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      tauMinus = world->getCoriolisAndGravityAndExternalForces();

      tab[0][iTab] = (tauPlus - tauMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * C(pos,
/// inertia, vel) by finite differences. This is SUPER SLOW, and is only here
/// for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfMinvC(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMinvC(world, wrt);

  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(
      world, mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs before = wrt->get(world.get());

  const s_t EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs tauPos = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXs tauNeg = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
    Eigen::VectorXs diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);
  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * C(pos,
/// inertia, vel) by Ridders extrapolated finite differences. This is SUPER
/// SLOW, and is only here for testing.
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceRiddersJacobianOfMinvC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(
      world, mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    Eigen::MatrixXs MinvCPlus = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    Eigen::MatrixXs MinvCMinus = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());

    tab[0][0] = (MinvCPlus - MinvCMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      MinvCPlus = implicitMultiplyByInvMassMatrix(
          world,
          mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      MinvCMinus = implicitMultiplyByInvMassMatrix(
          world,
          mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());

      tab[0][iTab] = (MinvCPlus - MinvCMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::finiteDifferenceJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfConstraintForce(world, wrt);
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  bool oldPenetrationCorrection = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  BackpropSnapshotPtr originalPtr = neural::forwardPass(world, true);
  Eigen::VectorXs f0 = originalPtr->getClampingConstraintImpulses();
  assert(f0.size() == mNumClamping);
  assert(originalPtr->getNumUpperBound() == mNumUpperBound);
  assert(originalPtr->areResultsStandardized());

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXs before = wrt->get(world.get());

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(f0.size(), innerDim);

  const s_t EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs fPlus;
    Eigen::VectorXs fMinus;

    Eigen::VectorXs perturbed = before;
    s_t epsPos = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) += epsPos;
      wrt->set(world.get(), perturbed);
      BackpropSnapshotPtr perturbedPtr = neural::forwardPass(world, true);
      if (perturbedPtr->getNumClamping() == f0.size()
          && perturbedPtr->getNumUpperBound() == originalPtr->getNumUpperBound()
          && (!areResultsStandardized()
              || perturbedPtr->areResultsStandardized())
          && perturbedPtr->getNumContacts() == originalPtr->getNumContacts()
          && perturbedPtr->getConstraintForceMixingDiagonal()(0)
                 == originalPtr->getConstraintForceMixingDiagonal()(0))
      {
        fPlus = perturbedPtr->getClampingConstraintImpulses();
        break;
      }
      else
      {
        std::cout << "Perturb[" << i << "] += " << epsPos << " failed"
                  << std::endl;
        std::cout << "Original num contacts: " << originalPtr->getNumContacts()
                  << std::endl;
        std::cout << "Original num clamping: " << originalPtr->getNumClamping()
                  << std::endl;
        std::cout << "Original num upper bounded: "
                  << originalPtr->getNumUpperBound() << std::endl;
        std::cout << "Original LCP standardized: "
                  << originalPtr->areResultsStandardized() << std::endl;
        std::cout << "Perturbed num contacts: "
                  << perturbedPtr->getContactConstraintImpulses().size()
                  << std::endl;
        std::cout << "Perturbed num clamping: "
                  << perturbedPtr->getNumClamping() << std::endl;
        std::cout << "Perturbed num upper bounded: "
                  << perturbedPtr->getNumUpperBound() << std::endl;
        std::cout << "Perturbed LCP standardized: "
                  << perturbedPtr->areResultsStandardized() << std::endl;
      }
      epsPos *= 0.5;
      assert(epsPos > 1e-30);
    }

    s_t epsNeg = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) -= epsNeg;
      wrt->set(world.get(), perturbed);
      BackpropSnapshotPtr perturbedPtr = neural::forwardPass(world, true);
      if (perturbedPtr->getNumClamping() == f0.size()
          && perturbedPtr->getNumUpperBound() == originalPtr->getNumUpperBound()
          && perturbedPtr->getNumContacts() == originalPtr->getNumContacts()
          && (!areResultsStandardized()
              || perturbedPtr->areResultsStandardized())
          && perturbedPtr->getConstraintForceMixingDiagonal()(0)
                 == originalPtr->getConstraintForceMixingDiagonal()(0))
      {
        fMinus = perturbedPtr->getClampingConstraintImpulses();
        break;
      }
      else
      {
        std::cout << "Perturb[" << i << "] -= " << epsNeg << " failed"
                  << std::endl;
        std::cout << "Original num contacts: "
                  << originalPtr->getContactConstraintImpulses().size()
                  << std::endl;
        std::cout << "Original num clamping: " << originalPtr->getNumClamping()
                  << std::endl;
        std::cout << "Original num upper bounded: "
                  << originalPtr->getNumUpperBound() << std::endl;
        std::cout << "Original LCP standardized: "
                  << originalPtr->areResultsStandardized() << std::endl;
        std::cout << "Perturbed num contacts: "
                  << perturbedPtr->getContactConstraintImpulses().size()
                  << std::endl;
        std::cout << "Perturbed num clamping: "
                  << perturbedPtr->getNumClamping() << std::endl;
        std::cout << "Perturbed num upper bounded: "
                  << perturbedPtr->getNumUpperBound() << std::endl;
        std::cout << "Perturbed LCP standardized: "
                  << perturbedPtr->areResultsStandardized() << std::endl;
      }
      epsNeg *= 0.5;
      assert(epsNeg > 1e-30);
    }
    Eigen::VectorXs diff = fPlus - fMinus;

    if (abs(epsPos) < 1e-11 || abs(epsNeg) < 1e-11)
    {
      std::cout << "WARNING: finiteDifferenceJacobianOfConstraintForce() had "
                   "to use dangerously small EPS to get a sample with the same "
                   "number of clamping contacts. Perturb["
                << i << "]: eps_pos=" << epsPos << ", eps_neg=" << epsNeg
                << std::endl;
    }

    result.col(i) = diff / (epsPos + epsNeg);
  }

  wrt->set(world.get(), before);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrection);

  snapshot.restore();
  return result;
}

//==============================================================================
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setControlForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  bool oldPenetrationCorrection = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  BackpropSnapshotPtr originalPtr = neural::forwardPass(world, true);
  Eigen::VectorXs f0 = originalPtr->getClampingConstraintImpulses();
  assert(f0.size() == mNumClamping);
  assert(originalPtr->getNumUpperBound() == mNumUpperBound);
  assert(originalPtr->areResultsStandardized());

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXs originalWrt = wrt->get(world.get());

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(f0.size(), innerDim);

  s_t originalStepSize = 1e-2;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 15;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs fPlus;
    Eigen::VectorXs fMinus;

    Eigen::VectorXs perturbed = originalWrt;

    // Find largest original step size which doesn't change
    // numClamping/numUpperBound
    while (true)
    {
      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += originalStepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= originalStepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      if (snapshotPlus->getNumClamping() == f0.size()
          && snapshotPlus->getNumUpperBound() == originalPtr->getNumUpperBound()
          && snapshotPlus->getNumContacts() == originalPtr->getNumContacts()
          && snapshotPlus->getConstraintForceMixingDiagonal()(0)
                 == originalPtr->getConstraintForceMixingDiagonal()(0)
          && (!areResultsStandardized()
              || snapshotPlus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == f0.size()
          && snapshotMinus->getNumUpperBound()
                 == originalPtr->getNumUpperBound()
          && snapshotMinus->getNumContacts() == originalPtr->getNumContacts()
          && snapshotMinus->getConstraintForceMixingDiagonal()(0)
                 == originalPtr->getConstraintForceMixingDiagonal()(0)
          && (!areResultsStandardized()
              || snapshotMinus->areResultsStandardized()))
      {
        fPlus = snapshotPlus->getClampingConstraintImpulses();
        fMinus = snapshotMinus->getClampingConstraintImpulses();
        break;
      }
      originalStepSize *= 0.5;

      assert(abs(originalStepSize) > 1e-20);
    }
    if (abs(originalStepSize) < 1e-11)
    {
      std::cout << "WARNING: finiteDifferenceRiddersJacobianOfConstraintForce()"
                   " had to use dangerously small EPS to get a sample with the"
                   " same number of clamping contacts. Perturb["
                << i << "]: originalStepSize=" << originalStepSize << std::endl;
    }

    tab[0][0] = (fPlus - fMinus) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, true);
      fPlus = snapshotPlus->getClampingConstraintImpulses();

      if (!(snapshotPlus->getNumClamping() == f0.size()
            && snapshotPlus->getNumUpperBound()
                   == originalPtr->getNumUpperBound()
            && snapshotPlus->getNumContacts() == originalPtr->getNumContacts()
            && snapshotPlus->getConstraintForceMixingDiagonal()(0)
                   == originalPtr->getConstraintForceMixingDiagonal()(0)
            && (!areResultsStandardized()
                || snapshotPlus->areResultsStandardized())))
      {
        assert(false && 
        "Lowering EPS in finiteDifferenceRiddersJacobianOfConstraintForce "
        "caused numClamping() or numUpperBound() to change.");
      }
      Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, true);
      fMinus = snapshotMinus->getClampingConstraintImpulses();
      if (!(snapshotMinus->getNumClamping() == f0.size()
            && snapshotMinus->getNumUpperBound()
                   == originalPtr->getNumUpperBound()
            && snapshotMinus->getNumContacts() == originalPtr->getNumContacts()
            && snapshotMinus->getConstraintForceMixingDiagonal()(0)
                   == originalPtr->getConstraintForceMixingDiagonal()(0)
            && (!areResultsStandardized()
                || snapshotMinus->areResultsStandardized())))
      {
        assert(false && 
        "Lowering EPS in finiteDifferenceRiddersJacobianOfConstraintForce "
        "caused numClamping() or numUpperBound() to change.");
      }

      /*
      #ifndef NDEBUG
      if (i == 1)
      {
        if (abs(fPlus(0) - f0(0)) > 1e-12)
        {
          std::cout << "Constraint force changed by " << fPlus(0) - f0(0)
                    << " with positive y-axis bump of " << stepSize << "!"
                    << std::endl;
        }
        else
        {
          std::cout
              << "Constraint force not changed with positive y-axis bump of "
              << stepSize << "!" << std::endl;
        }
        if (abs(fMinus(0) - f0(0)) > 1e-12)
        {
          std::cout << "Constraint force changed by " << fMinus(0) - f0(0)
                    << " with negative y-axis bump of " << stepSize << "!"
                    << std::endl;
          std::shared_ptr<DifferentiableContactConstraint> originalConstraint
              = originalPtr->getDifferentiableConstraints()[0];
          collision::Contact& originalContact
              = originalConstraint->getContact();
          std::shared_ptr<DifferentiableContactConstraint> constraint
              = snapshotMinus->getDifferentiableConstraints()[0];
          collision::Contact& contact = constraint->getContact();
          // constraint->getContactWorldForceDirection()
          if (originalContact.type != contact.type)
          {
            std::cout << "Original type: " << originalContact.type << std::endl;
            std::cout << "New type: " << contact.type << std::endl;
          }
          if (originalContact.point != contact.point)
          {
            std::cout << "Original pt: " << originalContact.point << std::endl;
            std::cout << "New pt: " << contact.point << std::endl;
            std::cout << "Diff: " << contact.point - originalContact.point
                      << std::endl;
          }
          if (originalContact.normal != contact.normal)
          {
            std::cout << "Original normal: " << originalContact.normal
                      << std::endl;
            std::cout << "New normal: " << contact.normal << std::endl;
            std::cout << "Diff: " << contact.normal - originalContact.normal
                      << std::endl;
          }
        }
        else
        {
          std::cout
              << "Constraint force not changed with negative y-axis bump of "
              << stepSize << "!" << std::endl;
        }
      }
      #endif
      */

      tab[0][iTab] = (fPlus - fMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  wrt->set(world.get(), originalWrt);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrection);

  snapshot.restore();
  return J;
}

//==============================================================================
/// This returns the jacobian of estimated constraint force, without actually
/// running forward passes, holding everyhing constant except the value of
/// WithRespectTo
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceJacobianOfEstimatedConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersJacobianOfEstimatedConstraintForce(
        world, wrt);
  }

  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);

  const s_t EPS = 1e-7;
  Eigen::VectorXs original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXs perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }

    Eigen::VectorXs fPlus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }

    Eigen::VectorXs fMinus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    jac.col(i) = (fPlus - fMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);
  snapshot.restore();

  return jac;
}

//==============================================================================
/// This returns the jacobian of estimated constraint force, without actually
/// running forward passes, holding everyhing constant except the value of
/// WithRespectTo. Uses Ridders method.
Eigen::MatrixXs
BackpropSnapshot::finiteDifferenceRiddersJacobianOfEstimatedConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXs A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXs A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXs E = getUpperBoundMappingMatrix();

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumClamping, wrtDim);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  Eigen::VectorXs originalWrt = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    wrt->set(world.get(), perturbedPlus);
    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs fPlus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    wrt->set(world.get(), perturbedMinus);
    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXs fMinus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    tab[0][0] = (fPlus - fMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      wrt->set(world.get(), perturbedPlus);
      if (wrt == WithRespectTo::POSITION)
      {
        A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
        A_ub = estimateUpperBoundConstraintMatrixAt(
            world, world->getPositions());
      }
      fPlus = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      wrt->set(world.get(), perturbedMinus);
      if (wrt == WithRespectTo::POSITION)
      {
        A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
        A_ub = estimateUpperBoundConstraintMatrixAt(
            world, world->getPositions());
      }
      fMinus = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

      tab[0][iTab] = (fPlus - fMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }
  wrt->set(world.get(), originalWrt);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXs BackpropSnapshot::assembleMatrix(
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

  Eigen::MatrixXs matrix = Eigen::MatrixXs::Zero(mNumDOFs, numCols);
  std::size_t constraintCursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXs groupMatrix;

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

Eigen::MatrixXs BackpropSnapshot::assembleBlockDiagonalMatrix(
    simulation::WorldPtr world,
    BackpropSnapshot::BlockDiagonalMatrixToAssemble whichMatrix,
    bool forFiniteDifferencing)
{
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);

  // If we're not finite differencing, then set the state of the world back to
  // what it was during the forward pass, so that implicit mass matrix
  // computations work correctly.

  Eigen::VectorXs oldPositions = world->getPositions();
  Eigen::VectorXs oldVelocities = world->getVelocities();
  if (!forFiniteDifferencing)
  {
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
  }

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::size_t skelDOF = world->getSkeleton(i)->getNumDofs();
    if (whichMatrix == BackpropSnapshot::BlockDiagonalMatrixToAssemble::MASS)
    {
      J.block(cursor, cursor, skelDOF, skelDOF)
          = world->getSkeleton(i)->getMassMatrix();
    }
    else if (
        whichMatrix
        == BackpropSnapshot::BlockDiagonalMatrixToAssemble::INV_MASS)
    {
      J.block(cursor, cursor, skelDOF, skelDOF)
          = world->getSkeleton(i)->getInvMassMatrix();
    }
    else if (
        whichMatrix == BackpropSnapshot::BlockDiagonalMatrixToAssemble::POS_C)
    {
      J.block(cursor, cursor, skelDOF, skelDOF)
          = world->getSkeleton(i)->getJacobianOfC(WithRespectTo::POSITION);
    }
    else if (
        whichMatrix == BackpropSnapshot::BlockDiagonalMatrixToAssemble::VEL_C)
    {
      J.block(cursor, cursor, skelDOF, skelDOF)
          = world->getSkeleton(i)->getVelCJacobian();
    }
    cursor += skelDOF;
  }

  // If we're not finite differencing, reset the position of the world to what
  // it was before

  if (!forFiniteDifferencing)
  {
    world->setPositions(oldPositions);
    world->setVelocities(oldVelocities);
  }

  return J;
}

//==============================================================================
template <typename Vec>
Vec BackpropSnapshot::assembleVector(VectorToAssemble whichVector)
{
  // When we're assembling vectors related to contact constraints, we can put
  // them in order of constraint groups
  if (whichVector == BOUNCE_DIAGONALS || whichVector == RESTITUTION_DIAGONALS
      || whichVector == CONTACT_CONSTRAINT_IMPULSES
      || whichVector == CONTACT_CONSTRAINT_MAPPINGS
      || whichVector == PENETRATION_VELOCITY_HACK
      || whichVector == CLAMPING_CONSTRAINT_IMPULSES
      || whichVector == CLAMPING_CONSTRAINT_RELATIVE_VELS
      || whichVector == CFM_CONSTANTS)
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
      // group CONTACT_CONSTRAINT_IMPULSES: This is the total number of
      // contacts, including non-clamping ones CONTACT_CONSTRAINT_MAPPINGS: This
      // is the total number of contacts, including non-clamping ones
      size
          += getVectorToAssemble<Vec>(mGradientMatrices[i], whichVector).size();
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
  // The other types of vectors need to go in order of skeletons
  else
  {
    Vec collected = Vec(mNumDOFs);
    collected.setZero();

    for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
    {
      const Vec& vec
          = getVectorToAssemble<Vec>(mGradientMatrices[i], whichVector);
      int groupCursor = 0;
      for (auto skelName : mGradientMatrices[i]->getSkeletons())
      {
        int dofs = mSkeletonDofs[skelName];
        int worldOffset = mSkeletonOffset[skelName];
        collected.segment(worldOffset, dofs) = vec.segment(groupCursor, dofs);
        groupCursor += dofs;
      }
    }
    return collected;
  }
}

//==============================================================================
template <>
const Eigen::VectorXs& BackpropSnapshot::getVectorToAssemble(
    std::shared_ptr<ConstrainedGroupGradientMatrices> matrices,
    VectorToAssemble whichVector)
{
  if (whichVector == VectorToAssemble::BOUNCE_DIAGONALS)
    return matrices->getBounceDiagonals();
  if (whichVector == VectorToAssemble::RESTITUTION_DIAGONALS)
    return matrices->getRestitutionDiagonals();
  if (whichVector == VectorToAssemble::CONTACT_CONSTRAINT_IMPULSES)
    return matrices->getContactConstraintImpulses();
  if (whichVector == VectorToAssemble::PENETRATION_VELOCITY_HACK)
    return matrices->getPenetrationCorrectionVelocities();
  if (whichVector == VectorToAssemble::CLAMPING_CONSTRAINT_IMPULSES)
    return matrices->getClampingConstraintImpulses();
  if (whichVector == VectorToAssemble::CLAMPING_CONSTRAINT_RELATIVE_VELS)
    return matrices->getClampingConstraintRelativeVels();
  if (whichVector == VectorToAssemble::VEL_DUE_TO_ILLEGAL)
    return matrices->getVelocityDueToIllegalImpulses();
  if (whichVector == VectorToAssemble::PRE_STEP_VEL)
    return matrices->getPreStepVelocity();
  if (whichVector == VectorToAssemble::PRE_STEP_TAU)
    return matrices->getPreStepTorques();
  if (whichVector == VectorToAssemble::PRE_LCP_VEL)
    return matrices->getPreLCPVelocity();
  if (whichVector == VectorToAssemble::CFM_CONSTANTS)
    return matrices->getConstraintForceMixingDiagonal();

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
  _unused(whichVector);
  return matrices->getContactConstraintMappings();
}

} // namespace neural
} // namespace dart
