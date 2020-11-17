#include "dart/trajectory/SingleShot.hpp"

#include <vector>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

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
  mForces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  mSnapshotsCacheDirty = true;
  mPinnedForces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
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
/// This sets the mapping we're using to store the representation of the Shot.
/// WARNING: THIS IS A POTENTIALLY DESTRUCTIVE OPERATION! This will rewrite
/// the internal representation of the Shot to use the new mapping, and if the
/// new mapping is underspecified compared to the old mapping, you may lose
/// information. It's not guaranteed that you'll get back the same trajectory
/// if you switch to a different mapping, and then switch back.
///
/// This will affect the values you get back from getStates() - they'll now be
/// returned in the view given by `mapping`. That's also the represenation
/// that'll be passed to IPOPT, and updated on each gradient step. Therein
/// lies the power of changing the representation mapping: There will almost
/// certainly be mapped spaces that are easier to optimize in than native
/// joint space, at least initially.
void SingleShot::switchRepresentationMapping(
    std::shared_ptr<simulation::World> world,
    const std::string& mapping,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.switchRepresentationMapping");
  }
#endif

  RestorableSnapshot snapshot(world);

  // Rewrite the forces in the new mapping
  Eigen::MatrixXd newForces = Eigen::MatrixXd::Zero(
      mMappings[mapping]->getForceDim(), mForces.cols());
  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);
  for (int i = 0; i < snapshots.size(); i++)
  {
    // Set the state in the old mapping
    // TODO:optimize
    Eigen::VectorXd posCopy
        = snapshots[i]->getPreStepPosition(mRepresentationMapping);
    Eigen::VectorXd velCopy
        = snapshots[i]->getPreStepVelocity(mRepresentationMapping);
    Eigen::VectorXd forceCopy
        = snapshots[i]->getPreStepTorques(mRepresentationMapping);
    getRepresentation()->setPositions(world, posCopy);
    getRepresentation()->setVelocities(world, velCopy);
    getRepresentation()->setForces(world, forceCopy);

    // Read back the forces in the new mapping
    newForces.col(i) = mMappings[mapping]->getForces(world);
  }
  mForces = newForces;
  // Any pinned forces must also show up in the main forces list
  mPinnedForces = newForces;

  // Rewrite the start state in the new mapping
  getRepresentation()->setPositions(world, mStartPos);
  getRepresentation()->setVelocities(world, mStartVel);
  Eigen::VectorXd forceCopy
      = snapshots[0]->getPreStepTorques(mRepresentationMapping);
  getRepresentation()->setForces(world, forceCopy);
  mStartPos = mMappings[mapping]->getPositions(world);
  mStartVel = mMappings[mapping]->getVelocities(world);

  mSnapshotsCacheDirty = true;
  Problem::switchRepresentationMapping(world, mapping, thisLog);
  snapshot.restore();

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This prevents a force from changing in optimization, keeping it fixed at a
/// specified value.
void SingleShot::pinForce(int time, Eigen::VectorXd value)
{
  mPinnedForces.col(time) = value;
  mForcesPinned[time] = true;
}

//==============================================================================
/// This returns the pinned force value at this timestep.
Eigen::Ref<Eigen::VectorXd> SingleShot::getPinnedForce(int time)
{
  return mPinnedForces.col(time);
}

//==============================================================================
/// Returns the length of the flattened problem state
int SingleShot::getFlatDynamicProblemDim(
    std::shared_ptr<simulation::World> /* ignored */) const
{
  if (mTuneStartingState)
    return (getRepresentation()->getPosDim()
            + getRepresentation()->getVelDim()) // Initial state
           + (mSteps * getRepresentation()->getForceDim());
  return mSteps * getRepresentation()->getForceDim();
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
    Eigen::Ref<Eigen::VectorXd> flatStatic,
    Eigen::Ref<Eigen::VectorXd> flatDynamic,
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
    flatDynamic.segment(cursorDynamic, getRepresentation()->getPosDim())
        = mStartPos;
    cursorDynamic += getRepresentation()->getPosDim();
    flatDynamic.segment(cursorDynamic, getRepresentation()->getVelDim())
        = mStartVel;
    cursorDynamic += getRepresentation()->getVelDim();
  }
  int forceDim = getRepresentation()->getForceDim();
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
    const Eigen::Ref<const Eigen::VectorXd>& flatStatic,
    const Eigen::Ref<const Eigen::VectorXd>& flatDynamic,
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
    mStartPos = flatDynamic.segment(0, getRepresentation()->getPosDim());
    cursorDynamic += getRepresentation()->getPosDim();
    mStartVel
        = flatDynamic.segment(cursorDynamic, getRepresentation()->getVelDim());
    cursorDynamic += getRepresentation()->getVelDim();
  }
  int forceDim = getRepresentation()->getForceDim();
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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
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
    int posDim = getRepresentation()->getPosDim();
    int velDim = getRepresentation()->getVelDim();
    flatDynamic.segment(0, posDim)
        = getRepresentation()->getPositionUpperLimits(world);
    flatDynamic.segment(posDim, velDim)
        = getRepresentation()->getVelocityUpperLimits(world);
    cursorDynamic = posDim + velDim;
  }
  int forceDim = getRepresentation()->getForceDim();
  Eigen::VectorXd forceUpperLimits = world->getExternalForceUpperLimits();
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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
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
    int posDim = getRepresentation()->getPosDim();
    int velDim = getRepresentation()->getVelDim();
    flatDynamic.segment(0, posDim)
        = getRepresentation()->getPositionLowerLimits(world);
    flatDynamic.segment(posDim, velDim)
        = getRepresentation()->getVelocityLowerLimits(world);
    cursorDynamic = posDim + velDim;
  }
  int forceDim = getRepresentation()->getForceDim();
  Eigen::VectorXd forceLowerLimits
      = getRepresentation()->getForceLowerLimits(world);
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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
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
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac,
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
    Eigen::Ref<Eigen::MatrixXd> jacStatic,
    Eigen::Ref<Eigen::MatrixXd> jacDynamic,
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

  int posDim = getRepresentation()->getPosDim();
  int velDim = getRepresentation()->getVelDim();
  int forceDim = getRepresentation()->getForceDim();

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXd::Zero(velDim, forceDim);
  last.forcePos = Eigen::MatrixXd::Zero(posDim, forceDim);
  last.posVel = Eigen::MatrixXd::Zero(velDim, posDim);
  last.posPos = Eigen::MatrixXd::Identity(posDim, posDim);
  last.velVel = Eigen::MatrixXd::Identity(velDim, velDim);
  last.velPos = Eigen::MatrixXd::Zero(posDim, velDim);

  /*
  std::cout << "Jac dynamic: " << jacDynamic.rows() << "x" << jacDynamic.cols()
            << std::endl;
  std::cout << "Jac static: " << jacStatic.rows() << "x" << jacStatic.cols()
            << std::endl;
  */
  assert(jacDynamic.rows() == posDim + velDim);
  assert(jacStatic.rows() == posDim + velDim);

  int cursorDynamic = getFlatDynamicProblemDim(world);
  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;
    Eigen::MatrixXd forceVel
        = ptr->getForceVelJacobian(world, mRepresentationMapping, thisLog);
    Eigen::MatrixXd massVel
        = ptr->getMassVelJacobian(world, mRepresentationMapping, thisLog);
    Eigen::MatrixXd posPos
        = ptr->getPosPosJacobian(world, mRepresentationMapping, thisLog);
    Eigen::MatrixXd posVel
        = ptr->getPosVelJacobian(world, mRepresentationMapping, thisLog);
    Eigen::MatrixXd velPos
        = ptr->getVelPosJacobian(world, mRepresentationMapping, thisLog);
    Eigen::MatrixXd velVel
        = ptr->getVelVelJacobian(world, mRepresentationMapping, thisLog);

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
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  Eigen::VectorXd originalEndPos = getFinalState(world, nullptr);

  int dim = getFlatProblemDim(world);
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  Eigen::VectorXd flat = Eigen::VectorXd(dim);
  flatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      nullptr);

  double EPS = 1e-7;

  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(
        world,
        flat.segment(0, staticDim),
        flat.segment(staticDim, dynamicDim),
        nullptr);
    flat(i) -= EPS;
    Eigen::VectorXd perturbedEndStatePos = getFinalState(world, nullptr);

    flat(i) -= EPS;
    unflatten(
        world,
        flat.segment(0, staticDim),
        flat.segment(staticDim, dynamicDim),
        nullptr);
    flat(i) += EPS;
    Eigen::VectorXd perturbedEndStateNeg = getFinalState(world, nullptr);

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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradDynamic,
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
  assert(gradDynamic.size() == dynamicDims);

  std::vector<MappedBackpropSnapshotPtr> snapshots
      = getSnapshots(world, thisLog);
  assert(snapshots.size() == mSteps);

  LossGradient nextTimestep;
  nextTimestep.lossWrtPosition
      = Eigen::VectorXd::Zero(mMappings[mRepresentationMapping]->getPosDim());
  nextTimestep.lossWrtVelocity
      = Eigen::VectorXd::Zero(mMappings[mRepresentationMapping]->getVelDim());
  nextTimestep.lossWrtTorque
      = Eigen::VectorXd::Zero(mMappings[mRepresentationMapping]->getForceDim());

  int cursorDynamic = dynamicDims;
  int forceDim = mMappings[mRepresentationMapping]->getForceDim();
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
          = gradWrtRollout->getForcesConst(pair.first).col(i);
      mappedGrad.lossWrtMass = gradWrtRollout->getMassesConst();

      mappedLosses[pair.first] = mappedGrad;
    }
    mappedLosses[mRepresentationMapping].lossWrtPosition
        += nextTimestep.lossWrtPosition;
    mappedLosses[mRepresentationMapping].lossWrtVelocity
        += nextTimestep.lossWrtVelocity;

    LossGradient thisTimestep;
    snapshots[i]->backprop(world, thisTimestep, mappedLosses, thisLog);

    Problem::accumulateStaticGradient(world, gradStatic, thisTimestep, thisLog);

    cursorDynamic -= forceDim;
    gradDynamic.segment(cursorDynamic, forceDim) = thisTimestep.lossWrtTorque;
    if (i == 0 && mTuneStartingState)
    {
      assert(
          cursorDynamic
          == mMappings[mRepresentationMapping]->getPosDim()
                 + mMappings[mRepresentationMapping]->getVelDim());
      cursorDynamic -= mMappings[mRepresentationMapping]->getVelDim();
      gradDynamic.segment(
          cursorDynamic, mMappings[mRepresentationMapping]->getVelDim())
          = thisTimestep.lossWrtVelocity;
      cursorDynamic -= mMappings[mRepresentationMapping]->getPosDim();
      gradDynamic.segment(
          cursorDynamic, mMappings[mRepresentationMapping]->getPosDim())
          = thisTimestep.lossWrtPosition;
    }
    thisTimestep.lossWrtTorque
        += gradWrtRollout
               ->getForcesConst(gradWrtRollout->getRepresentationMapping())
               .col(i);

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

    getRepresentation()->setPositions(world, mStartPos);
    getRepresentation()->setVelocities(world, mStartVel);

    for (int i = 0; i < mSteps; i++)
    {
      getRepresentation()->setForces(world, mForces.col(i));
      mSnapshotsCache.push_back(
          mappedForwardPass(world, mRepresentationMapping, mMappings));
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
    assert(rollout->getForces(key).cols() == mSteps);
    assert(rollout->getForces(key).rows() == mMappings[key]->getForceDim());
    for (int i = 0; i < mSteps; i++)
    {
      rollout->getPoses(key).col(i) = snapshots[i]->getPostStepPosition(key);
      rollout->getVels(key).col(i) = snapshots[i]->getPostStepVelocity(key);
      rollout->getForces(key).col(i) = snapshots[i]->getPreStepTorques(key);
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

  mStartPos = rollout->getPosesConst(mRepresentationMapping).col(0);
  mStartVel = rollout->getVelsConst(mRepresentationMapping).col(0);
  mForces = rollout->getForcesConst(mRepresentationMapping);
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
void SingleShot::setForcesRaw(Eigen::MatrixXd forces, PerformanceLog* log)
{
  mForces = forces;
}

//==============================================================================
/// This moves the trajectory forward in time, setting the starting point to
/// the new given starting point, and shifting the forces over by `steps`,
/// padding the remainder with 0s
Eigen::VectorXi SingleShot::advanceSteps(
    std::shared_ptr<simulation::World> world,
    Eigen::VectorXd startPos,
    Eigen::VectorXd startVel,
    int steps)
{
  Eigen::VectorXi mapping = Eigen::VectorXi::Zero(getFlatProblemDim(world));

  mStartPos = startPos;
  mStartVel = startVel;

  Eigen::MatrixXd newForces = Eigen::MatrixXd::Zero(mForces.rows(), mSteps);
  if (steps < mSteps)
  {
    newForces.block(0, 0, mForces.rows(), mSteps - steps)
        = mForces.block(0, steps, mForces.rows(), mSteps - steps);
  }
  mForces = newForces;

  return mapping;
}

//==============================================================================
Eigen::VectorXd SingleShot::getStartState()
{
  Eigen::VectorXd state = Eigen::VectorXd::Zero(getRepresentationStateSize());
  state.segment(0, mMappings[mRepresentationMapping]->getPosDim()) = mStartPos;
  state.segment(
      mMappings[mRepresentationMapping]->getPosDim(),
      mMappings[mRepresentationMapping]->getVelDim())
      = mStartVel;
  return state;
}

//==============================================================================
/// This returns start pos
Eigen::VectorXd SingleShot::getStartPos()
{
  return mStartPos;
}

//==============================================================================
/// This returns start vel
Eigen::VectorXd SingleShot::getStartVel()
{
  return mStartVel;
}

//==============================================================================
/// This sets the start pos
void SingleShot::setStartPos(Eigen::VectorXd startPos)
{
  mStartPos = startPos;
}

//==============================================================================
/// This sets the start vel
void SingleShot::setStartVel(Eigen::VectorXd startVel)
{
  mStartVel = startVel;
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd SingleShot::getFinalState(
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

  Eigen::VectorXd state = Eigen::VectorXd::Zero(getRepresentationStateSize());
  state.segment(0, getRepresentation()->getPosDim())
      = snapshots[snapshots.size() - 1]->getPostStepPosition(
          mRepresentationMapping);
  state.segment(
      getRepresentation()->getPosDim(), getRepresentation()->getVelDim())
      = snapshots[snapshots.size() - 1]->getPostStepVelocity(
          mRepresentationMapping);

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
    if (dim < getRepresentation()->getPosDim())
    {
      return "Start Pos " + std::to_string(dim);
    }
    dim -= getRepresentation()->getPosDim();
    if (dim < getRepresentation()->getVelDim())
    {
      return "Start Vel " + std::to_string(dim);
    }
    dim -= getRepresentation()->getVelDim();
  }
  int forceDim = getRepresentation()->getForceDim();
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

  int posDim = getRepresentation()->getPosDim();
  int velDim = getRepresentation()->getVelDim();
  int forceDim = getRepresentation()->getForceDim();

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXd::Zero(velDim, forceDim);
  last.forcePos = Eigen::MatrixXd::Zero(posDim, forceDim);
  last.posVel = Eigen::MatrixXd::Zero(velDim, posDim);
  last.posPos = Eigen::MatrixXd::Identity(posDim, posDim);
  last.velVel = Eigen::MatrixXd::Identity(velDim, velDim);
  last.velPos = Eigen::MatrixXd::Zero(posDim, velDim);

  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;
    Eigen::MatrixXd forceVel
        = ptr->getForceVelJacobian(world, mRepresentationMapping);
    Eigen::MatrixXd posPos
        = ptr->getPosPosJacobian(world, mRepresentationMapping);
    Eigen::MatrixXd posVel
        = ptr->getPosVelJacobian(world, mRepresentationMapping);
    Eigen::MatrixXd velPos
        = ptr->getVelPosJacobian(world, mRepresentationMapping);
    Eigen::MatrixXd velVel
        = ptr->getVelVelJacobian(world, mRepresentationMapping);

    // v_end <- f_t = v_end <- v_t+1 * v_t+1 <- f_t
    thisTimestep.forceVel = last.velVel * forceVel;
    // p_end <- f_t = p_end <- v_t+1 * v_t+1 <- f_t
    thisTimestep.forcePos = last.velPos * forceVel;
    // v_end <- p_t = (v_end <- p_t+1 * p_t+1 <- p_t) + (v_end <- v_t+1 * v_t+1
    // <- p_t)
    thisTimestep.posVel = last.posVel * posPos + last.velVel * posVel;
    // p_end <- p_t = (p_end <- p_t+1 * p_t+1 <- p_t) + (p_end <- v_t+1 * v_t+1
    // <- p_t)
    thisTimestep.posPos = last.posPos * posPos + last.velPos * posVel;
    // v_end <- v_t = (v_end <- p_t+1 * p_t+1 <- v_t) + (v_end <- v_t+1 * v_t+1
    // <- v_t)
    thisTimestep.velVel = last.posVel * velPos + last.velVel * velVel;
    // p_end <- v_t = (p_end <- p_t+1 * p_t+1 <- v_t) + (p_end <- v_t+1 * v_t+1
    // <- v_t)
    thisTimestep.velPos = last.posPos * velPos + last.velPos * velVel;

    last = thisTimestep;
  }

  return last;
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
TimestepJacobians SingleShot::finiteDifferenceStartStateJacobians(
    std::shared_ptr<simulation::World> world, double EPS)
{
  RestorableSnapshot snapshot(world);

  getRepresentation()->setPositions(world, mStartPos);
  getRepresentation()->setVelocities(world, mStartVel);
  for (int i = 0; i < mSteps; i++)
  {
    getRepresentation()->setForces(world, mForces.col(i));
    world->step();
  }

  Eigen::VectorXd originalEndPos = getRepresentation()->getPositions(world);
  Eigen::VectorXd originalEndVel = getRepresentation()->getVelocities(world);

  TimestepJacobians result;

  int posDim = getRepresentation()->getPosDim();
  int velDim = getRepresentation()->getVelDim();
  int forceDim = getRepresentation()->getForceDim();

  // Perturb starting position

  result.posPos = Eigen::MatrixXd::Zero(posDim, posDim);
  result.posVel = Eigen::MatrixXd::Zero(velDim, posDim);
  for (int j = 0; j < posDim; j++)
  {
    Eigen::VectorXd perturbedStartPos = mStartPos;
    perturbedStartPos(j) += EPS;
    getRepresentation()->setPositions(world, perturbedStartPos);
    getRepresentation()->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      getRepresentation()->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVel = getRepresentation()->getVelocities(world);

    perturbedStartPos = mStartPos;
    perturbedStartPos(j) -= EPS;
    getRepresentation()->setPositions(world, perturbedStartPos);
    getRepresentation()->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      getRepresentation()->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = getRepresentation()->getVelocities(world);

    result.posPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.posVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting velocity

  result.velPos = Eigen::MatrixXd::Zero(posDim, velDim);
  result.velVel = Eigen::MatrixXd::Zero(velDim, velDim);
  for (int j = 0; j < velDim; j++)
  {
    Eigen::VectorXd perturbedStartVel = mStartVel;
    perturbedStartVel(j) += EPS;
    getRepresentation()->setPositions(world, mStartPos);
    getRepresentation()->setVelocities(world, perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      getRepresentation()->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVel = getRepresentation()->getVelocities(world);

    perturbedStartVel = mStartVel;
    perturbedStartVel(j) -= EPS;
    getRepresentation()->setPositions(world, mStartPos);
    getRepresentation()->setVelocities(world, perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      getRepresentation()->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = getRepresentation()->getVelocities(world);

    result.velPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.velVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting force

  result.forcePos = Eigen::MatrixXd::Zero(posDim, forceDim);
  result.forceVel = Eigen::MatrixXd::Zero(velDim, forceDim);
  for (int j = 0; j < forceDim; j++)
  {
    Eigen::VectorXd perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) += EPS;
    getRepresentation()->setPositions(world, mStartPos);
    getRepresentation()->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXd force = i == 0 ? perturbedStartForce : mForces.col(i);
      getRepresentation()->setForces(world, force);
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVel = getRepresentation()->getVelocities(world);

    perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) -= EPS;
    getRepresentation()->setPositions(world, mStartPos);
    getRepresentation()->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXd force = i == 0 ? perturbedStartForce : mForces.col(i);
      getRepresentation()->setForces(world, force);
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = getRepresentation()->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = getRepresentation()->getVelocities(world);

    result.forcePos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.forceVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

} // namespace trajectory
} // namespace dart