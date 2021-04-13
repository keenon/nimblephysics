#include "dart/trajectory/SingleShot.hpp"

#include <vector>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

using namespace dart;
using namespace dynamics;
using namespace simulation;
using namespace neural;

#define LOG_PERFORMANCE_SINGLE_SHOT

namespace dart {
namespace trajectory {

//==============================================================================
SingleShot::SingleShot(
    std::shared_ptr<simulation::World> world,
    LossFn loss,
    int steps,
    bool tuneStartingState)
  : Problem(world, loss, steps)
{
  mTuneStartingState = tuneStartingState;
  mStartPos = world->getPositions();
  mStartVel = world->getVelocities();
  assert(steps > 0);
  mForces = Eigen::MatrixXs::Zero(world->getNumDofs(), steps);
  mSnapshotsCacheDirty = true;
  mPinnedForces = Eigen::MatrixXs::Zero(world->getNumDofs(), steps);
  for (int i = 0; i < steps; i++)
  {
    mForcesPinned.push_back(false);
  }
}

//==============================================================================
SingleShot::~SingleShot()
{
  // std::cout << "Freeing SingleShot: " << this << std::endl;
}

//==============================================================================
/// This prevents a force from changing in optimization, keeping it fixed at a
/// specified value.
void SingleShot::pinForce(int time, Eigen::VectorXs value)
{
  mPinnedForces.col(time) = value;
  mForcesPinned[time] = true;
}

//==============================================================================
/// This returns the pinned force value at this timestep.
Eigen::Ref<Eigen::VectorXs> SingleShot::getPinnedForce(int time)
{
  return mPinnedForces.col(time);
}

//==============================================================================
/// Returns the length of the flattened problem state
int SingleShot::getFlatDynamicProblemDim(
    std::shared_ptr<simulation::World> /* ignored */) const
{
  int dofs = mWorld->getNumDofs();
  if (mTuneStartingState)
    return (dofs * 2) // Initial state
           + (mSteps * dofs);
  return mSteps * dofs;
}

//==============================================================================
int SingleShot::getConstraintDim() const
{
  return 0;
}

//==============================================================================
/// This copies a shot down into a single flat vector
void SingleShot::flatten(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> flatStatic,
    Eigen::Ref<Eigen::VectorXs> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.flatten");
  }
#endif

  // Run the AbstractShot flattening, and set our cursors forward to ignore
  // anything flattened already
  int cursorStatic = Problem::getFlatStaticProblemDim(world);
  int cursorDynamic = Problem::getFlatDynamicProblemDim(world);
  Problem::flatten(
      world,
      flatStatic.segment(0, cursorStatic),
      flatDynamic.segment(0, cursorDynamic),
      thisLog);

  if (mTuneStartingState)
  {
    flatDynamic.segment(cursorDynamic, world->getNumDofs()) = mStartPos;
    cursorDynamic += world->getNumDofs();
    flatDynamic.segment(cursorDynamic, world->getNumDofs()) = mStartVel;
    cursorDynamic += world->getNumDofs();
  }
  int forceDim = world->getNumDofs();
  for (int i = 0; i < mSteps; i++)
  {
    if (mForcesPinned[i])
    {
      flatDynamic.segment(cursorDynamic, forceDim) = mPinnedForces.col(i);
    }
    else
    {
      flatDynamic.segment(cursorDynamic, forceDim) = mForces.col(i);
    }
    cursorDynamic += forceDim;
  }
  assert(cursorDynamic == flatDynamic.size());
  assert(cursorStatic == flatStatic.size());

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the parameters out of a flat vector
void SingleShot::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXs>& flatStatic,
    const Eigen::Ref<const Eigen::VectorXs>& flatDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.unflatten");
  }
#endif

  mRolloutCacheDirty = true;
  mSnapshotsCacheDirty = true;

  int cursorDynamic = Problem::getFlatDynamicProblemDim(world);
  int cursorStatic = Problem::getFlatStaticProblemDim(world);
  Problem::unflatten(
      world,
      flatStatic.segment(0, cursorStatic),
      flatDynamic.segment(0, cursorDynamic),
      thisLog);

  if (mTuneStartingState)
  {
    mStartPos = flatDynamic.segment(0, world->getNumDofs());
    cursorDynamic += world->getNumDofs();
    mStartVel = flatDynamic.segment(cursorDynamic, world->getNumDofs());
    cursorDynamic += world->getNumDofs();
  }
  int forceDim = world->getNumDofs();
  for (int i = 0; i < mSteps; i++)
  {
    mForces.col(i) = flatDynamic.segment(cursorDynamic, forceDim);
    cursorDynamic += forceDim;
  }

  assert(cursorDynamic == flatDynamic.size());
  assert(cursorStatic == flatStatic.size());

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void SingleShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getUpperBounds");
  }
#endif

  int cursorDynamic = Problem::getFlatDynamicProblemDim(world);
  int cursorStatic = Problem::getFlatStaticProblemDim(world);
  Problem::getUpperBounds(
      world,
      flatStatic.segment(0, cursorStatic),
      flatDynamic.segment(0, cursorDynamic),
      thisLog);

  if (mTuneStartingState)
  {
    int posDim = world->getNumDofs();
    int velDim = world->getNumDofs();
    flatDynamic.segment(0, posDim) = world->getPositionUpperLimits();
    flatDynamic.segment(posDim, velDim) = world->getVelocityUpperLimits();
    cursorDynamic = posDim + velDim;
  }
  int forceDim = world->getNumDofs();
  Eigen::VectorXs forceUpperLimits = world->getExternalForceUpperLimits();
  assert(forceDim == forceUpperLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    if (mForcesPinned[i])
    {
      flatDynamic.segment(cursorDynamic, forceDim) = mPinnedForces.col(i);
    }
    else
    {
      flatDynamic.segment(cursorDynamic, forceDim) = forceUpperLimits;
    }
    cursorDynamic += forceDim;
  }
  assert(cursorDynamic == flatDynamic.size());
  assert(cursorStatic == flatStatic.size());

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void SingleShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getLowerBounds");
  }
#endif

  int cursorDynamic = Problem::getFlatDynamicProblemDim(world);
  int cursorStatic = Problem::getFlatStaticProblemDim(world);
  Problem::getLowerBounds(
      world,
      flatStatic.segment(0, cursorStatic),
      flatDynamic.segment(0, cursorDynamic),
      thisLog);

  if (mTuneStartingState)
  {
    int posDim = world->getNumDofs();
    int velDim = world->getNumDofs();
    flatDynamic.segment(0, posDim) = world->getPositionLowerLimits();
    flatDynamic.segment(posDim, velDim) = world->getVelocityLowerLimits();
    cursorDynamic = posDim + velDim;
  }
  int forceDim = world->getNumDofs();
  Eigen::VectorXs forceLowerLimits = world->getExternalForceLowerLimits();
  assert(forceDim == forceLowerLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    if (mForcesPinned[i])
    {
      flatDynamic.segment(cursorDynamic, forceDim) = mPinnedForces.col(i);
    }
    else
    {
      flatDynamic.segment(cursorDynamic, forceDim) = forceLowerLimits;
    }
    cursorDynamic += forceDim;
  }
  assert(cursorDynamic == flatDynamic.size());
  assert(cursorStatic == flatStatic.size());

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void SingleShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getInitialGuess");
  }
#endif

  flatten(world, flatStatic, flatDynamic, thisLog);

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
void SingleShot::backpropJacobianOfFinalState(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXs> jac,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  backpropJacobianOfFinalState(
      world,
      jac.block(0, 0, jac.rows(), staticDim),
      jac.block(0, staticDim, jac.rows(), dynamicDim),
      log);
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
void SingleShot::backpropJacobianOfFinalState(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::MatrixXs> jacStatic,
    Eigen::Ref<Eigen::MatrixXs> jacDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.backpropJacobianOfFinalState");
  }
#endif

  Problem::initializeStaticJacobianOfFinalState(world, jacStatic, thisLog);

  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);

  int posDim = world->getNumDofs();
  int velDim = world->getNumDofs();
  int forceDim = world->getNumDofs();

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXs::Zero(velDim, forceDim);
  last.forcePos = Eigen::MatrixXs::Zero(posDim, forceDim);
  last.posVel = Eigen::MatrixXs::Zero(velDim, posDim);
  last.posPos = Eigen::MatrixXs::Identity(posDim, posDim);
  last.velVel = Eigen::MatrixXs::Identity(velDim, velDim);
  last.velPos = Eigen::MatrixXs::Zero(posDim, velDim);

  /*
  std::cout << "Jac dynamic: " << jacDynamic.rows() << "x" << jacDynamic.cols()
            << std::endl;
  std::cout << "Jac static: " << jacStatic.rows() << "x" << jacStatic.cols()
            << std::endl;
  */
  assert(jacDynamic.rows() == posDim + velDim);
  assert(jacStatic.rows() == posDim + velDim);

  RestorableSnapshot restoreSnapshot(world);

  int cursorDynamic = getFlatDynamicProblemDim(world);
  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;

    world->setPositions(ptr->getPreStepPosition());
    world->setVelocities(ptr->getPreStepVelocity());
    world->setControlForces(ptr->getPreStepTorques());
    world->setCachedLCPSolution(ptr->getPreStepLCPCache());

    const Eigen::MatrixXs& forceVel = ptr->getControlForceVelJacobian(world, thisLog);
    const Eigen::MatrixXs& posPos = ptr->getPosPosJacobian(world, thisLog);
    const Eigen::MatrixXs& posVel = ptr->getPosVelJacobian(world, thisLog);
    const Eigen::MatrixXs& velPos = ptr->getVelPosJacobian(world, thisLog);
    const Eigen::MatrixXs& velVel = ptr->getVelVelJacobian(world, thisLog);
    // This blows up our caches, because it does finite differencing, so put
    // this last
    const Eigen::MatrixXs& massVel = ptr->getMassVelJacobian(world, thisLog);

    // p_end <- f_t = p_end <- v_t+1 * v_t+1 <- f_t
    thisTimestep.forcePos = last.velPos * forceVel;
    // v_end <- f_t = v_end <- v_t+1 * v_t+1 <- f_t
    thisTimestep.forceVel = last.velVel * forceVel;
    // p_end <- m_t = p_end <- v_t+1 * v_t+1 <- m_t
    thisTimestep.massPos = last.velPos * massVel;
    // v_end <- m_t = v_end <- v_t+1 * v_t+1 <- m_t
    thisTimestep.massVel = last.velVel * massVel;
    // p_end <- v_t = (p_end <- p_t+1 * p_t+1 <- v_t) + (p_end <- v_t+1 * v_t+1
    // <- v_t)
    thisTimestep.velPos = last.posPos * velPos + last.velPos * velVel;
    // v_end <- v_t = (v_end <- p_t+1 * p_t+1 <- v_t) + (v_end <- v_t+1 * v_t+1
    // <- v_t)
    thisTimestep.velVel = last.posVel * velPos + last.velVel * velVel;
    // p_end <- p_t = (p_end <- p_t+1 * p_t+1 <- p_t) + (p_end <- v_t+1 * v_t+1
    // <- p_t)
    thisTimestep.posPos = last.posPos * posPos + last.velPos * posVel;
    // v_end <- p_t = (v_end <- p_t+1 * p_t+1 <- p_t) + (v_end <- v_t+1 * v_t+1
    // <- p_t)
    thisTimestep.posVel = last.posVel * posPos + last.velVel * posVel;

    cursorDynamic -= forceDim;
    jacDynamic.block(0, cursorDynamic, posDim, forceDim)
        = thisTimestep.forcePos;
    jacDynamic.block(posDim, cursorDynamic, velDim, forceDim)
        = thisTimestep.forceVel;

    if (i == 0 && mTuneStartingState)
    {
      cursorDynamic -= velDim;
      assert(cursorDynamic == posDim);
      jacDynamic.block(0, cursorDynamic, posDim, velDim) = thisTimestep.velPos;
      jacDynamic.block(posDim, cursorDynamic, velDim, velDim)
          = thisTimestep.velVel;
      cursorDynamic -= posDim;
      assert(cursorDynamic == 0);
      jacDynamic.block(0, cursorDynamic, posDim, posDim) = thisTimestep.posPos;
      jacDynamic.block(posDim, cursorDynamic, velDim, posDim)
          = thisTimestep.posVel;
    }

    Problem::accumulateStaticJacobianOfFinalState(
        world, jacStatic, thisTimestep, thisLog);

    last = thisTimestep;
  }
  assert(cursorDynamic == 0);

  restoreSnapshot.restore();

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
void SingleShot::finiteDifferenceJacobianOfFinalState(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXs> jac)
{
  Eigen::VectorXs originalEndPos = getFinalState(world, nullptr);

  int dim = getFlatProblemDim(world);
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  Eigen::VectorXs flat = Eigen::VectorXs(dim);
  flatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      nullptr);

  s_t EPS = 1e-7;

  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(
        world,
        flat.segment(0, staticDim),
        flat.segment(staticDim, dynamicDim),
        nullptr);
    flat(i) -= EPS;
    Eigen::VectorXs perturbedEndStatePos = getFinalState(world, nullptr);

    flat(i) -= EPS;
    unflatten(
        world,
        flat.segment(0, staticDim),
        flat.segment(staticDim, dynamicDim),
        nullptr);
    flat(i) += EPS;
    Eigen::VectorXs perturbedEndStateNeg = getFinalState(world, nullptr);

    jac.col(i) = (perturbedEndStatePos - perturbedEndStateNeg) / (2 * EPS);
  }

  // Restore original value
  unflatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      nullptr);
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void SingleShot::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> gradStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> gradDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.backpropGradientWrt");
  }
#endif

  Problem::initializeStaticGradient(world, gradStatic, thisLog);
  // Add any gradient we have from the loss wrt mass directly into our gradient,
  // cause we don't need to do any extra processing on that.
  gradStatic += gradWrtRollout->getMassesConst();

  int staticDims = getFlatStaticProblemDim(world);
  int dynamicDims = getFlatDynamicProblemDim(world);
  assert(gradStatic.size() == staticDims);
  _unused(staticDims);
  assert(gradDynamic.size() == dynamicDims);

  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);
  assert(snapshots.size() == mSteps);

  LossGradient nextTimestep;
  nextTimestep.lossWrtPosition = Eigen::VectorXs::Zero(world->getNumDofs());
  nextTimestep.lossWrtVelocity = Eigen::VectorXs::Zero(world->getNumDofs());
  nextTimestep.lossWrtTorque = Eigen::VectorXs::Zero(world->getNumDofs());

  int cursorDynamic = dynamicDims;
  int forceDim = world->getNumDofs();
  for (int i = mSteps - 1; i >= 0; i--)
  {
    std::unordered_map<std::string, LossGradient> mappedLosses;
    for (auto pair : mMappings)
    {
      LossGradient mappedGrad;
      mappedGrad.lossWrtPosition
          = gradWrtRollout->getPosesConst(pair.first).col(i);
      mappedGrad.lossWrtVelocity
          = gradWrtRollout->getVelsConst(pair.first).col(i);

      // Both these values are currently ignored
      mappedGrad.lossWrtTorque
          = gradWrtRollout->getControlForcesConst(pair.first).col(i);
      mappedGrad.lossWrtMass = gradWrtRollout->getMassesConst();

      mappedLosses[pair.first] = mappedGrad;
    }
    mappedLosses["identity"].lossWrtPosition += nextTimestep.lossWrtPosition;
    mappedLosses["identity"].lossWrtVelocity += nextTimestep.lossWrtVelocity;

    LossGradient thisTimestep;
    snapshots[i]->backprop(
        world,
        thisTimestep,
        mappedLosses,
        thisLog,
        mExploreAlternateStrategies);

    Problem::accumulateStaticGradient(world, gradStatic, thisTimestep, thisLog);

    cursorDynamic -= forceDim;
    gradDynamic.segment(cursorDynamic, forceDim) = thisTimestep.lossWrtTorque;
    if (i == 0 && mTuneStartingState)
    {
      int posDim = world->getNumDofs();
      int velDim = world->getNumDofs();
      assert(cursorDynamic == posDim + velDim);
      cursorDynamic -= velDim;
      gradDynamic.segment(cursorDynamic, velDim) = thisTimestep.lossWrtVelocity;
      cursorDynamic -= posDim;
      gradDynamic.segment(cursorDynamic, posDim) = thisTimestep.lossWrtPosition;
    }
    thisTimestep.lossWrtTorque += gradWrtRollout->getControlForcesConst().col(i);

    nextTimestep = thisTimestep;
  }
  assert(cursorDynamic == 0);

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the snapshots from a fresh unroll
std::vector<MappedBackpropSnapshotPtr> SingleShot::getSnapshots(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getSnapshots");
  }
#endif

  if (mSnapshotsCacheDirty)
  {
    PerformanceLog* refreshLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
    if (thisLog != nullptr)
    {
      refreshLog = thisLog->startRun("SingleShot.getSnapshots#refreshCache");
    }
#endif
    RestorableSnapshot snapshot(world);

    mSnapshotsCache.clear();
    mSnapshotsCache.reserve(mSteps);

    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);

    for (int i = 0; i < mSteps; i++)
    {
      world->setControlForces(mForces.col(i));
      mSnapshotsCache.push_back(mappedForwardPass(world, mMappings));
    }

    snapshot.restore();
    mSnapshotsCacheDirty = false;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
    if (refreshLog != nullptr)
    {
      refreshLog->end();
    }
#endif
  }

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return mSnapshotsCache;
}

//==============================================================================
/// This populates the passed in matrices with the values from this trajectory
void SingleShot::getStates(
    std::shared_ptr<simulation::World> world,
    /* OUT */ TrajectoryRollout* rollout,
    PerformanceLog* log,
    bool /* useKnots */)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getStates");
  }
#endif

  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);

  for (std::string key : rollout->getMappings())
  {
    assert(rollout->getPoses(key).cols() == mSteps);
    assert(rollout->getPoses(key).rows() == mMappings[key]->getPosDim());
    assert(rollout->getVels(key).cols() == mSteps);
    assert(rollout->getVels(key).rows() == mMappings[key]->getVelDim());
    assert(rollout->getControlForces(key).cols() == mSteps);
    assert(rollout->getControlForces(key).rows() == mMappings[key]->getControlForceDim());
    for (int i = 0; i < mSteps; i++)
    {
      rollout->getPoses(key).col(i) = snapshots[i]->getPostStepPosition(key);
      rollout->getVels(key).col(i) = snapshots[i]->getPostStepVelocity(key);
      rollout->getControlForces(key).col(i) = snapshots[i]->getPreStepTorques(key);
    }
  }
  assert(rollout->getMasses().size() == world->getMassDims());
  rollout->getMasses() = world->getMasses();
  for (auto pair : mMetadata)
  {
    rollout->setMetadata(pair.first, pair.second);
  }

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This fills our trajectory with the values from the rollout being passed in
void SingleShot::setStates(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* rollout,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.setStates");
  }
#endif

  mStartPos = rollout->getPosesConst().col(0);
  mStartVel = rollout->getVelsConst().col(0);
  mForces = rollout->getControlForcesConst();
  world->setMasses(rollout->getMassesConst());

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This sets the forces in this trajectory from the passed in matrix
void SingleShot::setControlForcesRaw(
    Eigen::MatrixXs forces, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.setControlForcesRaw");
  }
#endif

  mForces = forces;

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This moves the trajectory forward in time, setting the starting point to
/// the new given starting point, and shifting the forces over by `steps`,
/// padding the remainder with 0s
Eigen::VectorXi SingleShot::advanceSteps(
    std::shared_ptr<simulation::World> world,
    Eigen::VectorXs startPos,
    Eigen::VectorXs startVel,
    int steps)
{
  Eigen::VectorXi mapping = Eigen::VectorXi::Zero(getFlatProblemDim(world));

  mStartPos = startPos;
  mStartVel = startVel;

  Eigen::MatrixXs newForces = Eigen::MatrixXs::Zero(mForces.rows(), mSteps);
  if (steps < mSteps)
  {
    newForces.block(0, 0, mForces.rows(), mSteps - steps)
        = mForces.block(0, steps, mForces.rows(), mSteps - steps);
  }
  mForces = newForces;

  return mapping;
}

//==============================================================================
Eigen::VectorXs SingleShot::getStartState()
{
  Eigen::VectorXs state = Eigen::VectorXs::Zero(getRepresentationStateSize());
  int posDim = mWorld->getNumDofs();
  int velDim = mWorld->getNumDofs();
  state.segment(0, posDim) = mStartPos;
  state.segment(posDim, velDim) = mStartVel;
  return state;
}

//==============================================================================
/// This returns start pos
Eigen::VectorXs SingleShot::getStartPos()
{
  return mStartPos;
}

//==============================================================================
/// This returns start vel
Eigen::VectorXs SingleShot::getStartVel()
{
  return mStartVel;
}

//==============================================================================
/// This sets the start pos
void SingleShot::setStartPos(Eigen::VectorXs startPos)
{
  mStartPos = startPos;
}

//==============================================================================
/// This sets the start vel
void SingleShot::setStartVel(Eigen::VectorXs startVel)
{
  mStartVel = startVel;
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXs SingleShot::getFinalState(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getFinalState");
  }
#endif

  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);

  Eigen::VectorXs state = Eigen::VectorXs::Zero(getRepresentationStateSize());
  state.segment(0, world->getNumDofs())
      = snapshots[snapshots.size() - 1]->getPostStepPosition("identity");
  state.segment(world->getNumDofs(), world->getNumDofs())
      = snapshots[snapshots.size() - 1]->getPostStepVelocity("identity");

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return state;
}

//==============================================================================
/// This returns the debugging name of a given DOF
std::string SingleShot::getFlatDimName(
    std::shared_ptr<simulation::World> world, int dim)
{
  int staticDim = getFlatStaticProblemDim(world);
  if (dim < staticDim)
  {
    return "Static " + std::to_string(dim);
  }
  dim -= staticDim;
  if (mTuneStartingState)
  {
    if (dim < world->getNumDofs())
    {
      return "Start Pos " + std::to_string(dim);
    }
    dim -= world->getNumDofs();
    if (dim < world->getNumDofs())
    {
      return "Start Vel " + std::to_string(dim);
    }
    dim -= world->getNumDofs();
  }
  int forceDim = world->getNumDofs();
  for (int i = 0; i < mSteps; i++)
  {
    if (dim < forceDim)
    {
      return "Force[" + std::to_string(i) + "] " + std::to_string(dim);
    }
    dim -= forceDim;
  }
  return "Error OOB by " + std::to_string(dim);
}

//==============================================================================
/// This computes the Jacobians that relate each timestep to the endpoint of
/// the trajectory. For a timestep at time t, this will relate quantities like
/// v_t -> p_end, for example.
TimestepJacobians SingleShot::backpropStartStateJacobians(
    std::shared_ptr<simulation::World> world, bool useFdJacs)
{
  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, nullptr);

  int posDim = world->getNumDofs();
  int velDim = world->getNumDofs();
  int forceDim = world->getNumDofs();

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXs::Zero(velDim, forceDim);
  last.forcePos = Eigen::MatrixXs::Zero(posDim, forceDim);
  last.posVel = Eigen::MatrixXs::Zero(velDim, posDim);
  last.posPos = Eigen::MatrixXs::Identity(posDim, posDim);
  last.velVel = Eigen::MatrixXs::Identity(velDim, velDim);
  last.velPos = Eigen::MatrixXs::Zero(posDim, velDim);

  RestorableSnapshot restoreSnapshot(world);

  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;

    if (useFdJacs)
    {
      Eigen::MatrixXs forceVel = ptr->finiteDifferenceForceVelJacobian(world);
      Eigen::MatrixXs posPos = ptr->finiteDifferencePosPosJacobian(world);
      Eigen::MatrixXs posVel = ptr->finiteDifferencePosVelJacobian(world);
      Eigen::MatrixXs velPos = ptr->finiteDifferenceVelPosJacobian(world);
      Eigen::MatrixXs velVel = ptr->finiteDifferenceVelVelJacobian(world);

      // v_end <- f_t = v_end <- v_t+1 * v_t+1 <- f_t
      thisTimestep.forceVel = last.velVel * forceVel;
      // p_end <- f_t = p_end <- v_t+1 * v_t+1 <- f_t
      thisTimestep.forcePos = last.velPos * forceVel;
      // v_end <- p_t = (v_end <- p_t+1 * p_t+1 <- p_t) + (v_end <- v_t+1 *
      // v_t+1
      // <- p_t)
      thisTimestep.posVel = last.posVel * posPos + last.velVel * posVel;
      // p_end <- p_t = (p_end <- p_t+1 * p_t+1 <- p_t) + (p_end <- v_t+1 *
      // v_t+1
      // <- p_t)
      thisTimestep.posPos = last.posPos * posPos + last.velPos * posVel;
      // v_end <- v_t = (v_end <- p_t+1 * p_t+1 <- v_t) + (v_end <- v_t+1 *
      // v_t+1
      // <- v_t)
      thisTimestep.velVel = last.posVel * velPos + last.velVel * velVel;
      // p_end <- v_t = (p_end <- p_t+1 * p_t+1 <- v_t) + (p_end <- v_t+1 *
      // v_t+1
      // <- v_t)
      thisTimestep.velPos = last.posPos * velPos + last.velPos * velVel;
    }
    else
    {
      world->setPositions(ptr->getPreStepPosition());
      world->setVelocities(ptr->getPreStepVelocity());
      world->setControlForces(ptr->getPreStepTorques());
      world->setCachedLCPSolution(ptr->getPreStepLCPCache());

      const Eigen::MatrixXs& forceVel = ptr->getControlForceVelJacobian(world);
      const Eigen::MatrixXs& posPos = ptr->getPosPosJacobian(world);
      const Eigen::MatrixXs& posVel = ptr->getPosVelJacobian(world);
      const Eigen::MatrixXs& velPos = ptr->getVelPosJacobian(world);
      const Eigen::MatrixXs& velVel = ptr->getVelVelJacobian(world);

      // v_end <- f_t = v_end <- v_t+1 * v_t+1 <- f_t
      thisTimestep.forceVel = last.velVel * forceVel;
      // p_end <- f_t = p_end <- v_t+1 * v_t+1 <- f_t
      thisTimestep.forcePos = last.velPos * forceVel;
      // v_end <- p_t = (v_end <- p_t+1 * p_t+1 <- p_t) + (v_end <- v_t+1 *
      // v_t+1
      // <- p_t)
      thisTimestep.posVel = last.posVel * posPos + last.velVel * posVel;
      // p_end <- p_t = (p_end <- p_t+1 * p_t+1 <- p_t) + (p_end <- v_t+1 *
      // v_t+1
      // <- p_t)
      thisTimestep.posPos = last.posPos * posPos + last.velPos * posVel;
      // v_end <- v_t = (v_end <- p_t+1 * p_t+1 <- v_t) + (v_end <- v_t+1 *
      // v_t+1
      // <- v_t)
      thisTimestep.velVel = last.posVel * velPos + last.velVel * velVel;
      // p_end <- v_t = (p_end <- p_t+1 * p_t+1 <- v_t) + (p_end <- v_t+1 *
      // v_t+1
      // <- v_t)
      thisTimestep.velPos = last.posPos * velPos + last.velPos * velVel;
    }

    last = thisTimestep;
  }

  restoreSnapshot.restore();

  return last;
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
TimestepJacobians SingleShot::finiteDifferenceStartStateJacobians(
    std::shared_ptr<simulation::World> world, s_t EPS)
{
  RestorableSnapshot snapshot(world);

  world->setPositions(mStartPos);
  world->setVelocities(mStartVel);
  for (int i = 0; i < mSteps; i++)
  {
    world->setControlForces(mForces.col(i));
    world->step();
  }

  Eigen::VectorXs originalEndPos = world->getPositions();
  Eigen::VectorXs originalEndVel = world->getVelocities();

  TimestepJacobians result;

  int posDim = world->getNumDofs();
  int velDim = world->getNumDofs();
  int forceDim = world->getNumDofs();

  // Perturb starting position

  result.posPos = Eigen::MatrixXs::Zero(posDim, posDim);
  result.posVel = Eigen::MatrixXs::Zero(velDim, posDim);
  for (int j = 0; j < posDim; j++)
  {
    Eigen::VectorXs perturbedStartPos = mStartPos;
    perturbedStartPos(j) += EPS;
    world->setPositions(perturbedStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setControlForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXs perturbedEndPos = world->getPositions();
    Eigen::VectorXs perturbedEndVel = world->getVelocities();

    perturbedStartPos = mStartPos;
    perturbedStartPos(j) -= EPS;
    world->setPositions(perturbedStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setControlForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXs perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXs perturbedEndVelNeg = world->getVelocities();

    result.posPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.posVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting velocity

  result.velPos = Eigen::MatrixXs::Zero(posDim, velDim);
  result.velVel = Eigen::MatrixXs::Zero(velDim, velDim);
  for (int j = 0; j < velDim; j++)
  {
    Eigen::VectorXs perturbedStartVel = mStartVel;
    perturbedStartVel(j) += EPS;
    world->setPositions(mStartPos);
    world->setVelocities(perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setControlForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXs perturbedEndPos = world->getPositions();
    Eigen::VectorXs perturbedEndVel = world->getVelocities();

    perturbedStartVel = mStartVel;
    perturbedStartVel(j) -= EPS;
    world->setPositions(mStartPos);
    world->setVelocities(perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setControlForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXs perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXs perturbedEndVelNeg = world->getVelocities();

    result.velPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.velVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting force

  result.forcePos = Eigen::MatrixXs::Zero(posDim, forceDim);
  result.forceVel = Eigen::MatrixXs::Zero(velDim, forceDim);
  for (int j = 0; j < forceDim; j++)
  {
    Eigen::VectorXs perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) += EPS;
    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXs force = i == 0 ? perturbedStartForce : mForces.col(i);
      world->setControlForces(force);
      world->step();
    }

    Eigen::VectorXs perturbedEndPos = world->getPositions();
    Eigen::VectorXs perturbedEndVel = world->getVelocities();

    perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) -= EPS;
    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXs force = i == 0 ? perturbedStartForce : mForces.col(i);
      world->setControlForces(force);
      world->step();
    }

    Eigen::VectorXs perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXs perturbedEndVelNeg = world->getVelocities();

    result.forcePos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.forceVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

} // namespace trajectory
} // namespace dart