#include "dart/neural/BackpropSnapshot.hpp"

#include <iostream>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

// This will enable runtime checks where every analytical Jacobian is compared
// to the finite difference version, and we error if they're too close far
// apart:
//
#define LOG_PERFORMANCE_BACKPROP_SNAPSHOT ;

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
    Eigen::VectorXd preStepPosition,
    Eigen::VectorXd preStepVelocity,
    Eigen::VectorXd preStepTorques,
    Eigen::VectorXd preConstraintVelocities,
    Eigen::VectorXd preStepLCPCache)
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
  mPostStepTorques = world->getExternalForces();

  // Reset the world to the initial state before finalizing all the gradient
  // matrices

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

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

  snapshot.restore();

  mCachedPosPosDirty = true;
  mCachedVelPosDirty = true;
  mCachedBounceApproximationDirty = true;
  mCachedPosVelDirty = true;
  mCachedVelVelDirty = true;
  mCachedForcePosDirty = true;
  mCachedForceVelDirty = true;
  mCachedMassVelDirty = true;

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
    PerformanceLog* perfLog)
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

  Eigen::VectorXd oldPositions = world->getPositions();
  Eigen::VectorXd oldVelocities = world->getVelocities();
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);

  // Create the vectors for this timestep

  thisTimestepLoss.lossWrtPosition = Eigen::VectorXd::Zero(mNumDOFs);
  thisTimestepLoss.lossWrtVelocity = Eigen::VectorXd::Zero(mNumDOFs);
  thisTimestepLoss.lossWrtTorque = Eigen::VectorXd::Zero(mNumDOFs);

  const Eigen::MatrixXd& posPos = getPosPosJacobian(world, thisLog);
  const Eigen::MatrixXd& posVel = getPosVelJacobian(world, thisLog);
  const Eigen::MatrixXd& velPos = getVelPosJacobian(world, thisLog);
  const Eigen::MatrixXd& velVel = getVelVelJacobian(world, thisLog);
  const Eigen::MatrixXd& forceVel = getForceVelJacobian(world, thisLog);
  const Eigen::MatrixXd& massVel = getMassVelJacobian(world, thisLog);

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

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return;

  //////////////////////////////////////////////////////////
  // TODO: this is the older, potentially more efficient way to do it. We should
  // measure and verify that during optimization.

  /*
  // Actually run the backprop

  std::unordered_map<std::string, bool> skeletonsVisited;

  for (std::shared_ptr<ConstrainedGroupGradientMatrices> group :
       mGradientMatrices)
  {
    std::size_t groupDofs = group->getNumDOFs();

    // Instantiate the vectors with plenty of DOFs

    groupNextTimestepLoss.lossWrtPosition = Eigen::VectorXd::Zero(groupDofs);
    groupNextTimestepLoss.lossWrtVelocity = Eigen::VectorXd::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtPosition = Eigen::VectorXd::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtVelocity = Eigen::VectorXd::Zero(groupDofs);
    groupThisTimestepLoss.lossWrtTorque = Eigen::VectorXd::Zero(groupDofs);

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
    if (!skelAlreadyVisited && skel->isMobile())
    {
      std::size_t dofCursorWorld = mSkeletonOffset[skel->getName()];
      std::size_t dofs = skel->getNumDofs();
      // f_t
      // force-vel = dT * Minv
      thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
          // f_t --> v_t+1
          = mTimeStep
            * skel->multiplyByImplicitInvMassMatrix(
                nextTimestepLoss.lossWrtVelocity.segment(dofCursorWorld, dofs));

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
            - skel->getVelCJacobian().transpose()
                  * thisTimestepLoss.lossWrtTorque.segment(dofCursorWorld, dofs)
            // v_t --> p_t
            + mTimeStep
                  * thisTimestepLoss.lossWrtPosition.segment(
                      dofCursorWorld, dofs);
    }
  }

  // Restore the old position and velocity values before we ran backprop
  world->setPositions(oldPositions);
  world->setVelocities(oldVelocities);

#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  */
}

//==============================================================================
const Eigen::MatrixXd& BackpropSnapshot::getForceVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("BackpropSnapshot.getForceVelJacobian");
  }
#endif

  if (mCachedForceVelDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_BACKPROP_SNAPSHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun(
          "BackpropSnapshot.getForceVelJacobian#refreshCache");
    }
#endif
    if (mUseFDOverride)
    {
      mCachedForceVel = finiteDifferenceForceVelJacobian(world);
    }
    else
    {
      Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
      Eigen::MatrixXd Minv = getInvMassMatrix(world);

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
        Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
        Eigen::MatrixXd E = getUpperBoundMappingMatrix();
        Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix(world);

        if (A_ub.size() > 0 && E.size() > 0)
        {
          mCachedForceVel = mTimeStep * Minv
                            * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
                               - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
        }
        else
        {
          mCachedForceVel = mTimeStep * Minv
                            * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
                               - mTimeStep * A_c * P_c * Minv);
        }
        */
      }
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXd bruteForce = finiteDifferenceForceVelJacobian(world);
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
const Eigen::MatrixXd& BackpropSnapshot::getMassVelJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
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
      Eigen::MatrixXd bruteForce = finiteDifferenceMassVelJacobian(world);
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
const Eigen::MatrixXd& BackpropSnapshot::getVelVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
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
      Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);

      // If there are no clamping constraints, then vel-vel is just the identity
      if (A_c.size() == 0)
      {
        mCachedVelVel = Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
                        - getForceVelJacobian(world) * getVelCJacobian(world);
      }
      else
      {
        mCachedVelVel = getVelJacobianWrt(world, WithRespectTo::VELOCITY);

        /*
        Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
        Eigen::MatrixXd E = getUpperBoundMappingMatrix();
        Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix(world);
        Eigen::MatrixXd Minv = getInvMassMatrix(world);
        Eigen::MatrixXd dF_c
            = getJacobianOfConstraintForce(world, WithRespectTo::VELOCITY);
        Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;
        Eigen::MatrixXd parts2 = Minv * A_c_ub_E * dF_c;

        mCachedVelVel = (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) + parts2)
                        - getForceVelJacobian(world) * getVelCJacobian(world);
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
                   << Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) <<
         std::endl; std::cout << "2: - mTimestep * Minv * (A_c + A_ub * E) *
         P_c" << std::endl
                   << -parts2 << std::endl;
         std::cout << "2.5: velC" << std::endl << getVelCJacobian(world) <<
         std::endl; std::cout << "3: - forceVel * velC" << std::endl
                   << -getForceVelJacobian(world) * getVelCJacobian(world)
                   << std::endl;
         */
      }
    }

    if (mSlowDebugResultsAgainstFD)
    {
      Eigen::MatrixXd bruteForce = finiteDifferenceVelVelJacobian(world);
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
const Eigen::MatrixXd& BackpropSnapshot::getPosVelJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
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
      Eigen::MatrixXd bruteForce = finiteDifferencePosVelJacobian(world);
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
Eigen::VectorXd BackpropSnapshot::getAnalyticalNextV(
    simulation::WorldPtr world, bool morePreciseButSlower)
{
  Eigen::MatrixXd A_c
      = morePreciseButSlower
            ? getClampingConstraintMatrixAt(world, world->getPositions())
            : estimateClampingConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXd A_ub
      = morePreciseButSlower
            ? getUpperBoundConstraintMatrixAt(world, world->getPositions())
            : estimateUpperBoundConstraintMatrixAt(
                world, world->getPositions());
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXd Minv = world->getInvMassMatrix();
  Eigen::VectorXd tau = world->getExternalForces();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  double dt = world->getTimeStep();
  Eigen::VectorXd f_c = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

  Eigen::VectorXd preSolveV = mPreStepVelocity + dt * Minv * (tau - C);
  Eigen::VectorXd f_cDeltaV = Minv * A_c_ub_E * f_c;
  Eigen::VectorXd postSolveV = preSolveV + f_cDeltaV;
  return postSolveV;

  /*
  Eigen::VectorXd innerV = world->getVelocities() + dt * Minv * (tau - C);

  return world->getVelocities()
         + dt * Minv * (tau - C - A_c_ub_E * P_c * innerV);
  */
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getScratchAnalytical(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXd tau = world->getExternalForces();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXd f_c = getClampingConstraintImpulses();
  double dt = world->getTimeStep();

  Eigen::MatrixXd dM
      = getJacobianOfMinv(world, dt * (tau - C) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXd Minv = world->getInvMassMatrix();
  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);

  Eigen::MatrixXd dF_c = getJacobianOfConstraintForce(world, wrt);

  Eigen::MatrixXd Q = A_c.transpose() * Minv * A_c_ub_E;

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXd dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

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
Eigen::VectorXd BackpropSnapshot::scratch(simulation::WorldPtr world)
{
  /////////////////////////////////////////////////////////////////////////
  // Compute NextV
  /////////////////////////////////////////////////////////////////////////

  Eigen::MatrixXd A_c
      = estimateClampingConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXd A_ub
      = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXd Minv = world->getInvMassMatrix();

  Eigen::MatrixXd Q = A_c.transpose() * Minv * A_c_ub_E;

  Eigen::VectorXd b = Eigen::VectorXd::Zero(A_c.cols());
  // Q = Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c, A_ub, E);

  Eigen::VectorXd f_c = Q.completeOrthogonalDecomposition().solve(b);

  Eigen::VectorXd tau = world->getExternalForces();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  double dt = world->getTimeStep();

  Eigen::VectorXd nextV
      = world->getVelocities() + Minv * (dt * (tau - C) + A_c_ub_E * f_c);

  // return b;
  // return f_c;
  return nextV;
}

Eigen::MatrixXd BackpropSnapshot::getScratchFiniteDifference(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  bool oldCFM = world->getConstraintForceMixingEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);
  world->setPenetrationCorrectionEnabled(false);
  world->setConstraintForceMixingEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXd original = scratch(world);

  int worldDim = wrt->dim(world.get());
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(original.size(), worldDim);

  Eigen::VectorXd preStepWrt = wrt->get(world.get());

  double EPSILON = 1e-6;
  for (std::size_t i = 0; i < worldDim; i++)
  {
    Eigen::VectorXd tweakedWrt = preStepWrt;
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);
    Eigen::VectorXd perturbedPos = scratch(world);

    tweakedWrt = preStepWrt;
    tweakedWrt(i) -= EPSILON;
    wrt->set(world.get(), tweakedWrt);
    Eigen::VectorXd perturbedNeg = scratch(world);

    Eigen::VectorXd change = (perturbedPos - perturbedNeg) / (2 * EPSILON);
    J.col(i).noalias() = change;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);
  world->setConstraintForceMixingEnabled(oldCFM);

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  if (wrtDim == 0)
  {
    return Eigen::MatrixXd::Zero(world->getNumDofs(), 0);
  }
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXd tau = world->getExternalForces();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXd f_c = getClampingConstraintImpulses();
  double dt = world->getTimeStep();

  Eigen::MatrixXd dM
      = getJacobianOfMinv(world, dt * (tau - C) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXd Minv = world->getInvMassMatrix();

  Eigen::MatrixXd dF_c = getJacobianOfConstraintForce(world, wrt);

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
                        * Eigen::MatrixXd::Identity(
                            world->getNumDofs(), world->getNumDofs()))
              << std::endl;
    */
    snapshot.restore();
    return Minv
           * ((A_c_ub_E * dF_c)
              + (dt
                 * Eigen::MatrixXd::Identity(
                     world->getNumDofs(), world->getNumDofs())));
  }

  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);

  if (wrt == WithRespectTo::VELOCITY)
  {
    snapshot.restore();
    return Eigen::MatrixXd::Identity(world->getNumDofs(), world->getNumDofs())
           + Minv * (A_c_ub_E * dF_c - dt * dC);
  }
  else if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXd dA_c = getJacobianOfClampingConstraints(world, f_c);
    Eigen::MatrixXd dA_ubE = getJacobianOfUpperBoundConstraints(world, E * f_c);
    snapshot.restore();
    return dM + Minv * (A_c_ub_E * dF_c + dA_c + dA_ubE - dt * dC);
  }
  else
  {
    snapshot.restore();
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
  world->setForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXd tau = world->getForces();
  Eigen::VectorXd C = world->getCoriolisAndGravityAndExternalForces();
  Eigen::MatrixXd dM = getJacobianOfMinv(world, tau - C, wrt);
  Eigen::MatrixXd Minv = world->getInvMassMatrix();
  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);
  double dt = world->getTimeStep();
  Eigen::VectorXd innerV = world->getVelocities() + dt * Minv * (tau - C);

  Eigen::MatrixXd dP_c
      = getJacobianOfProjectionIntoClampsMatrix(world, innerV, wrt);
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix(world);
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXd outerTau = tau - C - A_c_ub_E * P_c * innerV;
  Eigen::MatrixXd dOuterM = getJacobianOfMinv(world, outerTau, wrt);

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
Eigen::MatrixXd BackpropSnapshot::getPosJacobianWrt(
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
    return Eigen::MatrixXd::Zero(mNumDOFs, wrt->dim(world.get()));
  }
}

//==============================================================================
const Eigen::MatrixXd& BackpropSnapshot::getBounceApproximationJacobian(
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

    RestorableSnapshot snapshot(world);
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    Eigen::MatrixXd A_b = getBouncingConstraintMatrix(world);

    // If there are no bounces, pos-pos is a simple identity
    if (A_b.size() == 0)
    {
      mCachedBounceApproximation
          = Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs);
    }
    else
    {
      // Construct the W matrix we'll need to use to solve for our closest
      // approx
      Eigen::MatrixXd W
          = Eigen::MatrixXd::Zero(A_b.rows() * A_b.rows(), A_b.cols());
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
      Eigen::MatrixXd X = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
      for (std::size_t i = 0; i < mNumDOFs; i++)
      {
        X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
      }

      mCachedBounceApproximation = X;
    }
    snapshot.restore();

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
const Eigen::MatrixXd& BackpropSnapshot::getPosPosJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
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
      RestorableSnapshot snapshot(world);
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      mCachedPosPos = world->getPosPosJacobian()
                      * getBounceApproximationJacobian(world, thisLog);

      snapshot.restore();
    }

    if (mSlowDebugResultsAgainstFD)
    {
      // TODO: this is crappy, because if we are actually bouncing we want a
      // better approximation
      Eigen::MatrixXd bruteForce = finiteDifferencePosPosJacobian(world, 1);
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
const Eigen::MatrixXd& BackpropSnapshot::getVelPosJacobian(
    WorldPtr world, PerformanceLog* perfLog)
{
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
      Eigen::MatrixXd bruteForce = finiteDifferenceVelPosJacobian(world, 1);
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
Eigen::VectorXd BackpropSnapshot::getPreStepPosition()
{
  return mPreStepPosition;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPreStepVelocity()
{
  // return assembleVector<Eigen::VectorXd>(VectorToAssemble::PRE_STEP_VEL);
  return mPreStepVelocity;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPreStepTorques()
{
  // return assembleVector<Eigen::VectorXd>(VectorToAssemble::PRE_STEP_TAU);
  return mPreStepTorques;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPreConstraintVelocity()
{
  return mPreConstraintVelocities;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPostStepPosition()
{
  return mPostStepPosition;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPostStepVelocity()
{
  return mPostStepVelocity;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getPostStepTorques()
{
  return mPostStepTorques;
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

  Eigen::MatrixXd mappingMatrix
      = Eigen::MatrixXd::Zero(numUpperBound, numClamping);

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
Eigen::MatrixXd BackpropSnapshot::getMassMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  return assembleBlockDiagonalMatrix(
      world,
      BackpropSnapshot::BlockDiagonalMatrixToAssemble::MASS,
      forFiniteDifferencing);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getInvMassMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  return assembleBlockDiagonalMatrix(
      world,
      BackpropSnapshot::BlockDiagonalMatrixToAssemble::INV_MASS,
      forFiniteDifferencing);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getClampingAMatrix()
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumClamping, mNumClamping);
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
Eigen::MatrixXd BackpropSnapshot::getPosCJacobian(simulation::WorldPtr world)
{
  return assembleBlockDiagonalMatrix(
      world, BackpropSnapshot::BlockDiagonalMatrixToAssemble::POS_C);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelCJacobian(simulation::WorldPtr world)
{
  return assembleBlockDiagonalMatrix(
      world, BackpropSnapshot::BlockDiagonalMatrixToAssemble::VEL_C);
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
/// Returns the constraint impulses along the clamping constraints
Eigen::VectorXd BackpropSnapshot::getClampingConstraintImpulses()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::CLAMPING_CONSTRAINT_IMPULSES);
}

//==============================================================================
/// Returns the relative velocities along the clamping constraints
Eigen::VectorXd BackpropSnapshot::getClampingConstraintRelativeVels()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::CLAMPING_CONSTRAINT_RELATIVE_VELS);
}

//==============================================================================
/// Returns the velocity change caused by illegal impulses in the LCP this
/// timestep
Eigen::VectorXd BackpropSnapshot::getVelocityDueToIllegalImpulses()
{
  return assembleVector<Eigen::VectorXd>(VectorToAssemble::VEL_DUE_TO_ILLEGAL);
}

//==============================================================================
/// Returns the coriolis and gravity forces pre-step
Eigen::VectorXd BackpropSnapshot::getCoriolisAndGravityAndExternalForces()
{
  return assembleVector<Eigen::VectorXd>(
      VectorToAssemble::CORIOLIS_AND_GRAVITY);
}

//==============================================================================
/// Returns the velocity pre-LCP
Eigen::VectorXd BackpropSnapshot::getPreLCPVelocity()
{
  return assembleVector<Eigen::VectorXd>(VectorToAssemble::PRE_LCP_VEL);
}

//==============================================================================
bool BackpropSnapshot::hasBounces()
{
  return mNumBouncing > 0;
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
    Eigen::MatrixXd analytical,
    Eigen::MatrixXd bruteForce,
    std::string name)
{
  if (!areResultsStandardized())
  {
    std::cout << "Got an LCP result that couldn't be standardized!"
              << std::endl;
    printReplicationInstructions(world);
    exit(1);
  }
  Eigen::MatrixXd diff = (analytical - bruteForce).cwiseAbs();
  double threshold = 1e-5;
  bool broken = (diff.array() > threshold).any();
  if (broken)
  {
    std::cout << "Found invalid matrix! " << name << std::endl;
    std::cout << "Analytical:" << std::endl << analytical << std::endl;
    std::cout << "Brute Force:" << std::endl << bruteForce << std::endl;
    std::cout << "Diff:" << std::endl << diff << std::endl;
    printReplicationInstructions(world);
    exit(1);
  }
}

//==============================================================================
void BackpropSnapshot::printReplicationInstructions(
    // TODO: export the world as a skel file
    std::shared_ptr<simulation::World> /* world */)
{
  std::cout << "Code to replicate:" << std::endl;
  std::cout << "--------------------" << std::endl;
  std::cout << "Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(" << mNumDOFs
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
  std::cout << "Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(" << mNumDOFs
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
  std::cout << "Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero("
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
  std::cout << "Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero("
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
  std::cout << "world->setExternalForces(brokenForce);" << std::endl;
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
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelVelJacobian(WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd velPos = world->getVelocities();
    Eigen::VectorXd velNeg = world->getVelocities();

    double epsPos = EPSILON;
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXd tweakedVel = Eigen::VectorXd(mPreStepVelocity);
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

      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      world->setPositions(mPreStepPosition);
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXd tweakedVel = Eigen::VectorXd(mPreStepVelocity);
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

      assert(std::abs(epsNeg) > 1e-20);
    }

    Eigen::VectorXd velChange = (velPos - velNeg) / (epsPos + epsNeg);

    // TODO: remove me
    /*
#ifndef NDEBUG
    Eigen::VectorXd identityCol = Eigen::VectorXd::Zero(velChange.size());
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
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      tweakedVel = Eigen::VectorXd::Zero(mPreStepVelocity);
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
Eigen::MatrixXd BackpropSnapshot::finiteDifferencePosVelJacobian(
    simulation::WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);
  bool oldCFM = world->getConstraintForceMixingEnabled();
  world->setConstraintForceMixingEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd perturbedVelPos = world->getVelocities();
    Eigen::VectorXd perturbedVelNeg = world->getVelocities();

    double epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXd tweakedPos = Eigen::VectorXd(mPreStepPosition);
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

      if (std::abs(epsPos) <= 1e-20)
      {
        std::cout << "Found a non-differentiabe point in getting pos-vel Jac:"
                  << std::endl;
        printReplicationInstructions(world);
      }
      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);
      world->setVelocities(mPreStepVelocity);
      Eigen::VectorXd tweakedPos = Eigen::VectorXd(mPreStepPosition);
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

      if (std::abs(epsNeg) <= 1e-20)
      {
        std::cout << "Found a non-differentiabe point in getting pos-vel Jac:"
                  << std::endl;
        printReplicationInstructions(world);
      }
      assert(std::abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);
  world->setConstraintForceMixingEnabled(oldCFM);

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceForceVelJacobian(
    WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalForces = world->getExternalForces();
  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd perturbedVelPos = world->getVelocities();
    Eigen::VectorXd perturbedVelNeg = world->getVelocities();

    double epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXd tweakedForces = Eigen::VectorXd(mPreStepTorques);
      tweakedForces(i) += epsPos;
      world->setExternalForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelPos = ptr->getPostStepVelocity();
        break;
      }
      epsPos *= 0.5;

      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      world->setPositions(mPreStepPosition);
      world->setVelocities(mPreStepVelocity);
      world->setCachedLCPSolution(mPreStepLCPCache);
      Eigen::VectorXd tweakedForces = Eigen::VectorXd(mPreStepTorques);
      tweakedForces(i) -= epsNeg;
      world->setExternalForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      if ((!areResultsStandardized() || ptr->areResultsStandardized())
          && ptr->getNumClamping() == getNumClamping()
          && ptr->getNumUpperBound() == getNumUpperBound())
      {
        perturbedVelNeg = ptr->getPostStepVelocity();
        break;
      }
      epsNeg *= 0.5;

      assert(std::abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  /*
  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXd tweakedForcesPos = Eigen::VectorXd::Zero(originalForces);
    tweakedForcesPos(i) += EPSILON;
    world->setExternalForces(tweakedForcesPos);
    world->step(false);
    Eigen::VectorXd velPos = world->getVelocities();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXd tweakedForcesNeg = Eigen::VectorXd::Zero(originalForces);
    tweakedForcesNeg(i) -= EPSILON;
    world->setExternalForces(tweakedForcesNeg);
    world->step(false);
    Eigen::VectorXd velNeg = world->getVelocities();

    Eigen::VectorXd velChange = (velPos - velNeg) / (2 * EPSILON);
    J.col(i).noalias() = velChange;
  }
  */

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceMassVelJacobian(
    simulation::WorldPtr world)
{
  RestorableSnapshot snapshot(world);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalMass = world->getWrtMass()->get(world.get());
  Eigen::VectorXd originalVel = world->getVelocities();

  Eigen::MatrixXd J(mNumDOFs, originalMass.size());

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < originalMass.size(); i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXd tweakedMass = Eigen::VectorXd(originalMass);
    tweakedMass(i) += EPSILON;
    world->getWrtMass()->set(world.get(), tweakedMass);

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
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = world->getPositions();

  // IMPORTANT: EPSILON must be larger than the distance traveled in a single
  // subdivided timestep. Ideally much larger.
  double EPSILON = (subdivisions > 1) ? (1e-2 / subdivisions) : 1e-6;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    Eigen::VectorXd tweakedPositions = Eigen::VectorXd(mPreStepPosition);
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
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = world->getPositions();

  double EPSILON = 1e-3 / subdivisions;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    Eigen::VectorXd tweakedVelocity = Eigen::VectorXd(mPreStepVelocity);
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
/// This computes and returns the whole wrt-vel jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::VectorXd originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < wrtDim; i++)
  {
    snapshot.restore();

    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    Eigen::VectorXd tweakedWrt = Eigen::VectorXd(originalWrt);
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);

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
/// This computes and returns the whole wrt-pos jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXd BackpropSnapshot::finiteDifferencePosJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd J(mNumDOFs, wrtDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  // world->getConstraintSolver()->setGradientEnabled(false);

  Eigen::VectorXd originalWrt = wrt->get(world.get());

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalPos = world->getPositions();

  double EPSILON = 1e-6;
  for (std::size_t i = 0; i < wrtDim; i++)
  {
    snapshot.restore();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    Eigen::VectorXd tweakedWrt = Eigen::VectorXd(originalWrt);
    tweakedWrt(i) += EPSILON;
    wrt->set(world.get(), tweakedWrt);
    world->step(false);
    Eigen::VectorXd pos = world->getPositions();

    snapshot.restore();
    world->setPositions(mPreStepPosition);
    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);
    tweakedWrt = Eigen::VectorXd(originalWrt);
    tweakedWrt(i) -= EPSILON;
    wrt->set(world.get(), tweakedWrt);
    world->step(false);
    Eigen::VectorXd neg = world->getPositions();

    Eigen::VectorXd posChange = (pos - neg) / (2 * EPSILON);
    J.col(i).noalias() = posChange;
  }

  snapshot.restore();
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  return J;
}

/*
//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getProjectionIntoClampsMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  Eigen::MatrixXd A_c;
  if (forFiniteDifferencing)
  {
    A_c = getClampingConstraintMatrixAt(world, world->getPositions());
  }
  else
  {
    A_c = getClampingConstraintMatrix(world);
  }
  if (A_c.size() == 0)
    return Eigen::MatrixXd::Zero(0, world->getNumDofs());

  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  Eigen::MatrixXd constraintForceToImpliedTorques;
  if (forFiniteDifferencing || true)
  {
    Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
    Eigen::MatrixXd Minv = getInvMassMatrix(world, forFiniteDifferencing);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));
  }
  else
  {
    Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix(world);
    Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix(world);
    constraintForceToImpliedTorques = V_c + (V_ub * E);
  }

  Eigen::MatrixXd forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();
  Eigen::MatrixXd rightHandSize = bounce * A_c.transpose();
  return (1.0 / mTimeStep)
         * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
}
*/

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getProjectionIntoClampsMatrix(
    WorldPtr world, bool forFiniteDifferencing)
{
  Eigen::MatrixXd A_c;
  if (forFiniteDifferencing)
  {
    A_c = getClampingConstraintMatrixAt(world, world->getPositions());
  }
  else
  {
    A_c = getClampingConstraintMatrix(world);
  }
  if (A_c.size() == 0)
    return Eigen::MatrixXd::Zero(0, world->getNumDofs());

  Eigen::MatrixXd constraintForceToImpliedTorques;
  if (forFiniteDifferencing)
  {
    Eigen::MatrixXd A_ub
        = getUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXd E
        = getUpperBoundMappingMatrixAt(world, world->getPositions());
    Eigen::MatrixXd Minv = getInvMassMatrix(world, true);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));

    Eigen::MatrixXd forceToVel
        = A_c.eval().transpose() * constraintForceToImpliedTorques;
    Eigen::MatrixXd bounce
        = getBounceDiagonalsAt(world, world->getPositions()).asDiagonal();
    Eigen::MatrixXd rightHandSize = bounce * A_c.transpose();
    return (1.0 / mTimeStep)
           * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
  }
  else
  {
    Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
    Eigen::MatrixXd E = getUpperBoundMappingMatrix();
    Eigen::MatrixXd Minv = getInvMassMatrix(world, false);
    constraintForceToImpliedTorques = Minv * (A_c + (A_ub * E));
    // We don't use the massed formulation anymore because it introduces slight
    // numerical instability

    // Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix(world);
    // Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix(world);
    // constraintForceToImpliedTorques = V_c + (V_ub * E);
    Eigen::MatrixXd forceToVel
        = A_c.eval().transpose() * constraintForceToImpliedTorques;
    Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();
    Eigen::MatrixXd rightHandSize = bounce * A_c.transpose();
    return (1.0 / mTimeStep)
           * forceToVel.completeOrthogonalDecomposition().solve(rightHandSize);
  }
}

/// This returns the result of M*x, without explicitly
/// forming M
Eigen::VectorXd BackpropSnapshot::implicitMultiplyByMassMatrix(
    simulation::WorldPtr world, const Eigen::VectorXd& x)
{
  Eigen::VectorXd result = x;
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
Eigen::VectorXd BackpropSnapshot::implicitMultiplyByInvMassMatrix(
    simulation::WorldPtr world, const Eigen::VectorXd& x)
{
  Eigen::VectorXd result = x;
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
Eigen::MatrixXd BackpropSnapshot::getJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  if (A_c.cols() == 0)
  {
    int wrtDim = wrt->dim(world.get());
    return Eigen::MatrixXd::Zero(0, wrtDim);
  }
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXd Minv = getInvMassMatrix(world);
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;
  Eigen::MatrixXd Q = A_c.transpose() * Minv * A_c_ub_E;

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXd dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    // dQ_b is 0, so don't compute it
    snapshot.restore();
    return Qfac.solve(dB);
  }

  Eigen::VectorXd b = getClampingConstraintRelativeVels();
  Eigen::MatrixXd dQ_b
      = getJacobianOfLCPConstraintMatrixClampingSubset(world, b, wrt);

  snapshot.restore();

  return dQ_b + Qfac.solve(dB);
}

//==============================================================================
Eigen::MatrixXd
BackpropSnapshot::getJacobianOfLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  if (A_c.cols() == 0)
  {
    return Eigen::MatrixXd::Zero(0, 0);
  }
  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    return Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXd Minv = getInvMassMatrix(world);
  Eigen::MatrixXd Q = A_c.transpose() * Minv * (A_c + A_ub * E);
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Qfactored
      = Q.completeOrthogonalDecomposition();

  Eigen::VectorXd Qinv_b = Qfactored.solve(b);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXd Qinv = Qfactored.pseudoInverse();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Q.rows(), Q.cols());

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

      snapshot.restore();
      // This is the gradient of the pseudoinverse, see
      // https://mathoverflow.net/a/29511/163259
      return -Qinv * dQ(Qinv * b)
             + Qinv * Qinv.transpose() * dQT((I - Q * Qinv) * b)
             + (I - Qinv * Q) * dQT(Qinv.transpose() * Qinv * b);

#undef dQ
#undef dQT

      /*
      // The old formula, approximating just the raw inverse, for posterity

      Eigen::MatrixXd innerTerms
          = getJacobianOfClampingConstraintsTranspose(
                world, Minv * A_c_ub_E * Qinv_b)
            + A_c.transpose()
                  * (getJacobianOfMinv(world, A_c_ub_E * Qinv_b, wrt)
                     + Minv
                           * (getJacobianOfClampingConstraints(world, Qinv_b)
                              + getJacobianOfUpperBoundConstraints(
                                  world, E * Qinv_b)));
      Eigen::MatrixXd result = -Qfactored.solve(innerTerms);
      return result;
      */
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

      snapshot.restore();
      // This is the gradient of the pseudoinverse, see
      // https://mathoverflow.net/a/29511/163259
      return -Qinv * dQ(Qinv * b)
             + Qinv * Qinv.transpose() * dQT((I - Q * Qinv) * b)
             + (I - Qinv * Q) * dQT(Qinv.transpose() * Qinv * b);

#undef dQ
#undef dQT

      /*
      // The old formula, approximating just the raw inverse, for posterity

      Eigen::MatrixXd innerTerms
          = getJacobianOfClampingConstraintsTranspose(
                world, Minv * A_c * Qinv_b)
            + A_c.transpose()
                  * (getJacobianOfMinv(world, A_c * Qinv_b, wrt)
                     + Minv * getJacobianOfClampingConstraints(world, Qinv_b));
      Eigen::MatrixXd result = -Qfactored.solve(innerTerms);
      return result;
      */
    }
  }
  else
  {
    // All other terms get to treat A_c as constant
    Eigen::MatrixXd innerTerms
        = A_c.transpose() * getJacobianOfMinv(world, A_c * Qinv_b, wrt);
    Eigen::MatrixXd result = -Qfactored.solve(innerTerms);

    snapshot.restore();
    return result;
  }

  assert(false && "Execution should never reach this point.");
}

//==============================================================================
/// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
/// wrt, by finite differencing
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mNumClamping, wrtDim);
  if (wrt != WithRespectTo::POSITION)
  {
    return jac;
  }

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const double EPS = 5e-7;
  Eigen::VectorXd original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXd perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    Eigen::MatrixXd A_c
        = estimateClampingConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXd A_ub
        = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    Eigen::MatrixXd E = getUpperBoundMappingMatrix();
    Eigen::MatrixXd Q
        = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);

    /*
    std::cout << "+" << i << ": " << A_c.cols() << " :: " << mNumClamping
              << std::endl;
              */

    Eigen::VectorXd bPlus = Q.completeOrthogonalDecomposition().solve(b);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    E = getUpperBoundMappingMatrix();
    Q = A_c.transpose() * world->getInvMassMatrix() * (A_c + A_ub * E);

    /*
    std::cout << "-" << i << ": " << A_c.cols() << " :: " << mNumClamping
              << std::endl;
              */

    Eigen::VectorXd bMinus = Q.completeOrthogonalDecomposition().solve(b);

    jac.col(i) = (bPlus - bMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  snapshot.restore();
  return jac;
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
Eigen::MatrixXd BackpropSnapshot::getJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  double dt = world->getTimeStep();
  Eigen::MatrixXd Minv = getInvMassMatrix(world);
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);
  if (wrt == WithRespectTo::VELOCITY)
  {
    snapshot.restore();
    return getBounceDiagonals().asDiagonal() * -A_c.transpose()
           * (Eigen::MatrixXd::Identity(
                  world->getNumDofs(), world->getNumDofs())
              - dt * Minv * dC);
  }
  else if (wrt == WithRespectTo::FORCE)
  {
    snapshot.restore();
    return getBounceDiagonals().asDiagonal() * -A_c.transpose() * dt * Minv;
  }

  Eigen::VectorXd C = getCoriolisAndGravityAndExternalForces();
  Eigen::VectorXd f = getPreStepTorques() - C;
  Eigen::MatrixXd dMinv_f = getJacobianOfMinv(world, f, wrt);
  Eigen::VectorXd v_f = getPreConstraintVelocity();

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXd dA_c_f
        = getJacobianOfClampingConstraintsTranspose(world, v_f);

    snapshot.restore();
    return getBounceDiagonals().asDiagonal()
           * -(dA_c_f + A_c.transpose() * dt * (dMinv_f - Minv * dC));
  }
  else
  {
    snapshot.restore();
    return getBounceDiagonals().asDiagonal()
           * -(A_c.transpose() * dt * (dMinv_f - Minv * dC));
  }
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
/// finite differencing
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mNumClamping, wrtDim);

  const double EPS = 1e-7;
  Eigen::VectorXd original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXd bPlus;
    Eigen::VectorXd bMinus;

    double epsPos = EPS;
    while (true)
    {
      Eigen::VectorXd perturbed = original;
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

    double epsNeg = EPS;
    while (true)
    {
      Eigen::VectorXd perturbed = original;
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
/// finite differencing
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfLCPEstimatedOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mNumClamping, wrtDim);
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);

  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  const double EPS = 1e-8;
  Eigen::VectorXd original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXd perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXd bPlus = Eigen::VectorXd::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bPlus, A_c);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
    }
    Eigen::VectorXd bMinus = Eigen::VectorXd::Zero(mNumClamping);
    computeLCPOffsetClampingSubset(world, bMinus, A_c);

    jac.col(i) = (bPlus - bMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  snapshot.restore();

  return jac;
}

//==============================================================================
/// This returns the subset of the A matrix used by the original LCP for just
/// the clamping constraints. It relates constraint force to constraint
/// acceleration. It's a mass matrix, just in a weird frame.
void BackpropSnapshot::computeLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world,
    Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& A_c,
    const Eigen::MatrixXd& A_ub,
    const Eigen::MatrixXd& E)
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
}

//==============================================================================
/// This returns the subset of the b vector used by the original LCP for just
/// the clamping constraints. It's just the relative velocity at the clamping
/// contact points.
void BackpropSnapshot::computeLCPOffsetClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXd& b, const Eigen::MatrixXd& A_c)
{
  /*
  Eigen::VectorXd velDiff = world->getVelocities() - mPreStepVelocity;
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
  b = -getBounceDiagonals().cwiseProduct(
      A_c.transpose()
      * (world->getVelocities()
         + (world->getTimeStep()
            * implicitMultiplyByInvMassMatrix(
                world,
                world->getExternalForces()
                    - world->getCoriolisAndGravityAndExternalForces()))));
}

//==============================================================================
/// This computes and returns an estimate of the constraint impulses for the
/// clamping constraints. This is based on a linear approximation of the
/// constraint impulses.
Eigen::VectorXd BackpropSnapshot::estimateClampingConstraintImpulses(
    simulation::WorldPtr world,
    const Eigen::MatrixXd& A_c,
    const Eigen::MatrixXd& A_ub,
    const Eigen::MatrixXd& E)
{
  if (A_c.cols() == 0)
  {
    return Eigen::VectorXd::Zero(0);
  }

  Eigen::VectorXd b = Eigen::VectorXd::Zero(A_c.cols());
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c, A_ub, E);

  // Q can be low rank, but we resolve ambiguity during the forward pass by
  // taking the least-squares minimal solution
  return Q.completeOrthogonalDecomposition().solve(b);
}

//==============================================================================
/// This returns the jacobian of P_c * v, holding everyhing constant except
/// the value of WithRespectTo
Eigen::MatrixXd BackpropSnapshot::getJacobianOfProjectionIntoClampsMatrix(
    simulation::WorldPtr world, Eigen::VectorXd v, WithRespectTo* wrt)
{
  // return finiteDifferenceJacobianOfProjectionIntoClampsMatrix(world, v, wrt);

  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  if (A_c.size() == 0)
    return Eigen::MatrixXd::Zero(0, world->getNumDofs());
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix(world);
  Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd A_c_ub_E = A_c + (A_ub * E);
  Eigen::MatrixXd Q = A_c.eval().transpose() * constraintForceToImpliedTorques;
  auto XFactor = Q.completeOrthogonalDecomposition();
  Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();

  // New formulation
  if (wrt == WithRespectTo::POSITION)
  {
    // d/d Q^{-1} v = - Q^{-1} (d/d Q) Q^{-1} v
    Eigen::MatrixXd rightHandSide = bounce * A_c.transpose();
    Eigen::MatrixXd dRhs
        = bounce * getJacobianOfClampingConstraintsTranspose(world, v);
    Eigen::MatrixXd Minv = getInvMassMatrix(world);

    Eigen::MatrixXd Qinv = XFactor.pseudoInverse();
    Eigen::VectorXd Qinv_v = XFactor.solve(rightHandSide * v);
    Eigen::MatrixXd dQ
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

    Eigen::VectorXd tau
        = A_c_ub_E * XFactor.solve(bounce * A_c.transpose() * v);

    Eigen::MatrixXd MinvJac = getJacobianOfMinv(world, tau, wrt);

    return -(1.0 / world->getTimeStep())
           * XFactor.solve(A_c.transpose() * MinvJac);
  }

  // An older approach that attempted to handle pseudoinverse distinct from
  // normal inverse

  /*
  Eigen::MatrixXd X = XFactor.pseudoInverse();
  Eigen::VectorXd A_c_T_V = bounce * A_c.transpose() * v;

  // Part 1

  Eigen::VectorXd part1Tau = A_c * X.transpose() * X * A_c_T_V;
  Eigen::MatrixXd part1MinvJac = getJacobianOfMinv(world, part1Tau, wrt);
  Eigen::MatrixXd XQ = X * Q;
  Eigen::MatrixXd part1 = (Eigen::MatrixXd::Identity(XQ.rows(), XQ.cols()) + XQ)
                          * A_c_ub_E.transpose() * part1MinvJac;

  Eigen::MatrixXd QX = Q * X;
  Eigen::VectorXd part2Tau
      = A_c * (Eigen::MatrixXd::Identity(QX.rows(), QX.cols()) - QX) * A_c_T_V;
  Eigen::MatrixXd part2MinvJac = getJacobianOfMinv(world, part2Tau, wrt);
  Eigen::MatrixXd part2
      = X * X.transpose() * A_c_ub_E.transpose() * part2MinvJac;

  Eigen::VectorXd part3Tau = A_c_ub_E * X * A_c_T_V;
  Eigen::MatrixXd part3MinvJac = getJacobianOfMinv(world, part3Tau, wrt);
  Eigen::MatrixXd part3 = X * A_c.transpose() * part3MinvJac;

  return (1.0 / mTimeStep) * (part1 + part2 - part3);
  */
}

//==============================================================================
/// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
/// everything constant except the value of WithRespectTo
Eigen::MatrixXd BackpropSnapshot::getJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfMinv(world, tau, wrt);
}

//==============================================================================
/// This returns the jacobian of C(pos, inertia, vel), holding everything
/// constant except the value of WithRespectTo
Eigen::MatrixXd BackpropSnapshot::getJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfC(world, wrt);
}

/// This returns the jacobian of M^{-1}(pos, inertia) * (C(pos, inertia, vel) +
/// mPreStepTorques), holding everything constant except the value of
/// WithRespectTo
Eigen::MatrixXd BackpropSnapshot::getJacobianOfMinvC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfMinvC(world, wrt);
}

//==============================================================================
/// This returns a fast approximation to A_c in the neighborhood of the original
Eigen::MatrixXd BackpropSnapshot::estimateClampingConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  Eigen::VectorXd posDiff = pos - mPreStepPosition;
  if (posDiff.squaredNorm() == 0)
  {
    return getClampingConstraintMatrix(world);
  }
  Eigen::VectorXd oldPos = world->getPositions();
  world->setPositions(mPreStepPosition);

  auto clampingConstraints = getClampingConstraints();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumDOFs, mNumClamping);
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
Eigen::MatrixXd BackpropSnapshot::estimateUpperBoundConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  Eigen::VectorXd posDiff = pos - mPreStepPosition;
  if (posDiff.squaredNorm() == 0)
  {
    return getUpperBoundConstraintMatrix(world);
  }
  Eigen::VectorXd oldPos = world->getPositions();
  world->setPositions(mPreStepPosition);

  auto upperBoundConstraints = getUpperBoundConstraints();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumDOFs, mNumUpperBound);
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
Eigen::MatrixXd BackpropSnapshot::getClampingConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  Eigen::MatrixXd bruteResult = ptr->getClampingConstraintMatrix(world);
  return bruteResult;
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of A_ub at the
/// desired position.
Eigen::MatrixXd BackpropSnapshot::getUpperBoundConstraintMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getUpperBoundConstraintMatrix(world);
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of E at the
/// desired position.
Eigen::MatrixXd BackpropSnapshot::getUpperBoundMappingMatrixAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getUpperBoundMappingMatrix();
}

//==============================================================================
/// Only for testing: VERY SLOW. This returns the actual value of the bounce
/// diagonals at the desired position.
Eigen::VectorXd BackpropSnapshot::getBounceDiagonalsAt(
    simulation::WorldPtr world, Eigen::VectorXd pos)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(pos);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  BackpropSnapshotPtr ptr = neural::forwardPass(world, true);

  snapshot.restore();

  return ptr->getBounceDiagonals();
}

//==============================================================================
/// This computes the Jacobian of A_c*f0 with respect to `wrt` using impulse
/// tests.
Eigen::MatrixXd BackpropSnapshot::getJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXd f0)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  int dofs = world->getNumDofs();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dofs, dofs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(world);
  }

  snapshot.restore();
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_c^T*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXd BackpropSnapshot::getJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXd v0)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  int dofs = world->getNumDofs();
  assert(constraints.size() == mNumClamping);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumClamping, dofs);
  for (int i = 0; i < constraints.size(); i++)
  {
    result.row(i)
        = constraints[i]->getConstraintForcesJacobian(world).transpose() * v0;
  }

  snapshot.restore();
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub*E*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXd BackpropSnapshot::getJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXd E_f0)
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  int dofs = world->getNumDofs();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dofs, dofs);
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
Eigen::MatrixXd BackpropSnapshot::getJacobianOfUpperBoundConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXd v0)
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  int dofs = world->getNumDofs();
  assert(constraints.size() == mNumUpperBound);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumUpperBound, dofs);
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
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXd f0)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXd original = getClampingConstraintMatrix(world) * f0;
  Eigen::MatrixXd originalA_c = getClampingConstraintMatrix(world);

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), mNumDOFs);

  const double EPS = 5e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    double posEPS = EPS;
    Eigen::VectorXd perturbedResultPos = Eigen::VectorXd::Zero(0);
    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 40; j++)
    {
      snapshot.restore();
      Eigen::VectorXd perturbed = mPreStepPosition;
      perturbed(i) += posEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      Eigen::MatrixXd perturbedA_c = ptr->getClampingConstraintMatrix(world);
      if (perturbedA_c.cols() == f0.size())
      {
        perturbedResultPos = perturbedA_c * f0;
        double squaredNorm = (original - perturbedResultPos).squaredNorm();
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

    double negEPS = EPS;
    Eigen::VectorXd perturbedResultNeg = Eigen::VectorXd::Zero(0);
    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 40; j++)
    {
      snapshot.restore();
      Eigen::VectorXd perturbed = mPreStepPosition;
      perturbed(i) -= negEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setExternalForces(mPreStepTorques);
      world->setCachedLCPSolution(mPreStepLCPCache);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
      Eigen::MatrixXd perturbedA_c = ptr->getClampingConstraintMatrix(world);
      if (perturbedA_c.cols() == f0.size())
      {
        perturbedResultNeg = perturbedA_c * f0;
        double squaredNorm = (original - perturbedResultNeg).squaredNorm();
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
/// This computes the finite difference Jacobian of A_c^T*v0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice.
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXd v0)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::VectorXd original
      = getClampingConstraintMatrix(world).transpose() * v0;

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), mNumDOFs);

  const double EPS = 5e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    double posEPS = EPS;
    Eigen::VectorXd perturbedResultPos;

    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 10; j++)
    {
      snapshot.restore();
      Eigen::VectorXd perturbed = mPreStepPosition;
      perturbed(i) += posEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setExternalForces(mPreStepTorques);
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

    double negEPS = EPS;
    Eigen::VectorXd perturbedResultNeg;

    // Keep scaling down the EPS until it no longer results in a different
    // number of columns
    for (int j = 0; j < 10; j++)
    {
      snapshot.restore();
      Eigen::VectorXd perturbed = mPreStepPosition;
      perturbed(i) -= negEPS;
      world->setPositions(perturbed);
      world->setVelocities(mPreStepVelocity);
      world->setExternalForces(mPreStepTorques);
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

/// This computes the finite difference Jacobian of A_ub*E*f0 with respect to
/// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
/// is way too slow to use in practice.
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXd f0)
{
  if (mNumUpperBound == 0)
  {
    return Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  }

  RestorableSnapshot snapshot(world);

  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  /*
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  std::cout << "A_ub size: " << A_ub.rows() << "x" << A_ub.cols() << std::endl;
  std::cout << "E size: " << E.rows() << "x" << E.cols() << std::endl;
  std::cout << "f0 size: " << f0.size() << std::endl;
  */

  Eigen::VectorXd original = A_ub * f0;

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), mNumDOFs);

  const double EPS = 1e-7;

  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    snapshot.restore();
    Eigen::VectorXd perturbed = mPreStepPosition;
    perturbed(i) += EPS;
    world->setPositions(perturbed);
    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    BackpropSnapshotPtr ptr = neural::forwardPass(world, true);
    Eigen::VectorXd perturbedResultPos
        = ptr->getUpperBoundConstraintMatrix(world) * f0;

    perturbed = mPreStepPosition;
    perturbed(i) -= EPS;
    world->setPositions(perturbed);
    world->setVelocities(mPreStepVelocity);
    world->setExternalForces(mPreStepTorques);
    world->setCachedLCPSolution(mPreStepLCPCache);

    ptr = neural::forwardPass(world, true);
    Eigen::VectorXd perturbedResultNeg
        = ptr->getUpperBoundConstraintMatrix(world) * f0;

    result.col(i) = (perturbedResultPos - perturbedResultNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of P_c * v by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
    simulation::WorldPtr world, Eigen::VectorXd v, WithRespectTo* wrt)
{
  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXd before = wrt->get(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = getProjectionIntoClampsMatrix(world, true) * v;

  Eigen::MatrixXd originalP_c = getProjectionIntoClampsMatrix(world, true);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getDifferentiableConstraints();

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  const double EPS = 1e-5;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    double posEps = EPS;

    Eigen::VectorXd newPlus;
    Eigen::VectorXd newMinus;

    while (true)
    {
      perturbed = before;
      perturbed(i) += posEps;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr plusBackptr = neural::forwardPass(world, true);
      Eigen::MatrixXd newP_c
          = plusBackptr->getProjectionIntoClampsMatrix(world);
      if (newP_c.rows() == originalP_c.rows())
      {
        newPlus = newP_c * v;
        break;
      }
      posEps *= 0.5;
    }

    perturbed = before;
    double negEps = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) -= negEps;
      wrt->set(world.get(), perturbed);

      BackpropSnapshotPtr negBackptr = neural::forwardPass(world, true);

      Eigen::MatrixXd newP_c = getProjectionIntoClampsMatrix(world, true);
      if (newP_c.rows() == originalP_c.rows())
      {
        newMinus = newP_c * v;
        break;
      }
      negEps *= 0.5;
    }

    Eigen::VectorXd diff = newPlus - newMinus;
    result.col(i) = diff / (posEps + negEps);
  }

  wrt->set(world.get(), before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// finite differences.
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt)
{
  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  Eigen::VectorXd before = wrt->get(world.get());

  const double EPS = 5e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd newVPlus = implicitMultiplyByInvMassMatrix(world, tau);
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd newVMinus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXd diff = newVPlus - newVMinus;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by finite
/// differences.
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = world->getCoriolisAndGravityAndExternalForces();

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  Eigen::VectorXd before = wrt->get(world.get());

  const double EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd tauPos = world->getCoriolisAndGravityAndExternalForces();
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd tauNeg = world->getCoriolisAndGravityAndExternalForces();
    Eigen::VectorXd diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * C(pos,
/// inertia, vel) by finite differences. This is SUPER SLOW, and is only here
/// for testing.
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceJacobianOfMinvC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t innerDim = wrt->dim(world.get());

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = implicitMultiplyByInvMassMatrix(
      world, mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  Eigen::VectorXd before = wrt->get(world.get());

  const double EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd tauPos = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
    perturbed = before;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);
    Eigen::MatrixXd tauNeg = implicitMultiplyByInvMassMatrix(
        world,
        mPreStepTorques - world->getCoriolisAndGravityAndExternalForces());
    Eigen::VectorXd diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  wrt->set(world.get(), before);

  return result;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  RestorableSnapshot snapshot(world);
  world->setPositions(mPreStepPosition);
  world->setVelocities(mPreStepVelocity);
  world->setExternalForces(mPreStepTorques);
  world->setCachedLCPSolution(mPreStepLCPCache);

  bool oldPenetrationCorrection = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);
  bool oldCFM = world->getConstraintForceMixingEnabled();
  world->setConstraintForceMixingEnabled(false);

  BackpropSnapshotPtr originalPtr = neural::forwardPass(world, true);
  Eigen::VectorXd f0 = originalPtr->getClampingConstraintImpulses();
  assert(f0.size() == mNumClamping);
  assert(originalPtr->getNumUpperBound() == mNumUpperBound);
  assert(originalPtr->areResultsStandardized());

  std::size_t innerDim = wrt->dim(world.get());

  Eigen::VectorXd before = wrt->get(world.get());

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(f0.size(), innerDim);

  const double EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd fPlus;
    Eigen::VectorXd fMinus;

    Eigen::VectorXd perturbed = before;
    double epsPos = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) += epsPos;
      wrt->set(world.get(), perturbed);
      BackpropSnapshotPtr perturbedPtr = neural::forwardPass(world, true);
      if (perturbedPtr->getNumClamping() == f0.size()
          && perturbedPtr->getNumUpperBound() == originalPtr->getNumUpperBound()
          && (!areResultsStandardized()
              || perturbedPtr->areResultsStandardized()))
      {
        fPlus = perturbedPtr->getClampingConstraintImpulses();
        break;
      }
      else
      {
        std::cout << "Perturb[" << i << "] += " << epsPos << " failed"
                  << std::endl;
        std::cout << "Original num contacts: "
                  << originalPtr->getContactConstraintImpluses().size()
                  << std::endl;
        std::cout << "Original num clamping: " << originalPtr->getNumClamping()
                  << std::endl;
        std::cout << "Original num upper bounded: "
                  << originalPtr->getNumUpperBound() << std::endl;
        std::cout << "Original LCP standardized: "
                  << originalPtr->areResultsStandardized() << std::endl;
        std::cout << "Perturbed num contacts: "
                  << perturbedPtr->getContactConstraintImpluses().size()
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

    double epsNeg = EPS;
    while (true)
    {
      perturbed = before;
      perturbed(i) -= epsNeg;
      wrt->set(world.get(), perturbed);
      BackpropSnapshotPtr perturbedPtr = neural::forwardPass(world, true);
      if (perturbedPtr->getNumClamping() == f0.size()
          && perturbedPtr->getNumUpperBound() == originalPtr->getNumUpperBound()
          && (!areResultsStandardized()
              || perturbedPtr->areResultsStandardized()))
      {
        fMinus = perturbedPtr->getClampingConstraintImpulses();
        break;
      }
      else
      {
        std::cout << "Perturb[" << i << "] -= " << epsNeg << " failed"
                  << std::endl;
        std::cout << "Original num contacts: "
                  << originalPtr->getContactConstraintImpluses().size()
                  << std::endl;
        std::cout << "Original num clamping: " << originalPtr->getNumClamping()
                  << std::endl;
        std::cout << "Original num upper bounded: "
                  << originalPtr->getNumUpperBound() << std::endl;
        std::cout << "Original LCP standardized: "
                  << originalPtr->areResultsStandardized() << std::endl;
        std::cout << "Perturbed num contacts: "
                  << perturbedPtr->getContactConstraintImpluses().size()
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
    Eigen::VectorXd diff = fPlus - fMinus;

    if (std::abs(epsPos) < 1e-11 || std::abs(epsNeg) < 1e-11)
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
  world->setConstraintForceMixingEnabled(oldCFM);

  snapshot.restore();
  return result;
}

//==============================================================================
/// This returns the jacobian of estimated constraint force, without actually
/// running forward passes, holding everyhing constant except the value of
/// WithRespectTo
Eigen::MatrixXd
BackpropSnapshot::finiteDifferenceJacobianOfEstimatedConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix(world);
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix(world);
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  int wrtDim = wrt->dim(world.get());
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(mNumClamping, wrtDim);

  const double EPS = 1e-7;
  Eigen::VectorXd original = wrt->get(world.get());
  for (int i = 0; i < wrtDim; i++)
  {
    Eigen::VectorXd perturbed = original;
    perturbed(i) += EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }

    Eigen::VectorXd fPlus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    perturbed = original;
    perturbed(i) -= EPS;
    wrt->set(world.get(), perturbed);

    if (wrt == WithRespectTo::POSITION)
    {
      A_c = estimateClampingConstraintMatrixAt(world, world->getPositions());
      A_ub = estimateUpperBoundConstraintMatrixAt(world, world->getPositions());
    }

    Eigen::VectorXd fMinus
        = estimateClampingConstraintImpulses(world, A_c, A_ub, E);

    jac.col(i) = (fPlus - fMinus) / (2 * EPS);
  }
  wrt->set(world.get(), original);

  return jac;
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

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(mNumDOFs, numCols);
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

Eigen::MatrixXd BackpropSnapshot::assembleBlockDiagonalMatrix(
    simulation::WorldPtr world,
    BackpropSnapshot::BlockDiagonalMatrixToAssemble whichMatrix,
    bool forFiniteDifferencing)
{
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);

  // If we're not finite differencing, then set the state of the world back to
  // what it was during the forward pass, so that implicit mass matrix
  // computations work correctly.

  Eigen::VectorXd oldPositions = world->getPositions();
  Eigen::VectorXd oldVelocities = world->getVelocities();
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
      || whichVector == CLAMPING_CONSTRAINT_RELATIVE_VELS)
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
  if (whichVector == VectorToAssemble::CORIOLIS_AND_GRAVITY)
    return matrices->getCoriolisAndGravityAndExternalForces();

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