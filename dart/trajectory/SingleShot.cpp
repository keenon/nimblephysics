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
  : AbstractShot(world, loss, steps)
{
  mTuneStartingState = tuneStartingState;
  mStartPos = world->getPositions();
  mStartVel = world->getVelocities();
  assert(steps > 0);
  mForces = Eigen::MatrixXd::Zero(world->getNumDofs(), steps);
  mSnapshotsCacheDirty = true;
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

  // Rewrite the start state in the new mapping
  getRepresentation()->setPositions(world, mStartPos);
  getRepresentation()->setVelocities(world, mStartVel);
  Eigen::VectorXd forceCopy
      = snapshots[0]->getPreStepTorques(mRepresentationMapping);
  getRepresentation()->setForces(world, forceCopy);
  mStartPos = mMappings[mapping]->getPositions(world);
  mStartVel = mMappings[mapping]->getVelocities(world);

  mSnapshotsCacheDirty = true;
  AbstractShot::switchRepresentationMapping(world, mapping, thisLog);
  snapshot.restore();

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// Returns the length of the flattened problem state
int SingleShot::getFlatProblemDim() const
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
    Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.flatten");
  }
#endif

  int cursor = 0;
  if (mTuneStartingState)
  {
    flat.segment(0, getRepresentation()->getPosDim()) = mStartPos;
    cursor += getRepresentation()->getPosDim();
    flat.segment(cursor, getRepresentation()->getVelDim()) = mStartVel;
    cursor += getRepresentation()->getVelDim();
  }
  int forceDim = getRepresentation()->getForceDim();
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = mForces.col(i);
    cursor += forceDim;
  }
  assert(cursor == flat.size());

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
    const Eigen::Ref<const Eigen::VectorXd>& flat, PerformanceLog* log)
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
  int cursor = 0;
  if (mTuneStartingState)
  {
    mStartPos = flat.segment(0, getRepresentation()->getPosDim());
    cursor += getRepresentation()->getPosDim();
    mStartVel = flat.segment(cursor, getRepresentation()->getVelDim());
    cursor += getRepresentation()->getVelDim();
  }
  int forceDim = getRepresentation()->getForceDim();
  for (int i = 0; i < mSteps; i++)
  {
    mForces.col(i) = flat.segment(cursor, forceDim);
    cursor += forceDim;
  }
  assert(cursor == flat.size());

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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getUpperBounds");
  }
#endif

  int cursor = 0;
  if (mTuneStartingState)
  {
    int posDim = getRepresentation()->getPosDim();
    int velDim = getRepresentation()->getVelDim();
    flat.segment(0, posDim)
        = getRepresentation()->getPositionUpperLimits(world);
    flat.segment(posDim, velDim)
        = getRepresentation()->getVelocityUpperLimits(world);
    cursor = posDim + velDim;
  }
  int forceDim = getRepresentation()->getForceDim();
  Eigen::VectorXd forceUpperLimits = world->getForceUpperLimits();
  assert(forceDim == forceUpperLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = forceUpperLimits;
    cursor += forceDim;
  }
  assert(cursor == flat.size());

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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getLowerBounds");
  }
#endif

  int cursor = 0;
  if (mTuneStartingState)
  {
    int posDim = getRepresentation()->getPosDim();
    int velDim = getRepresentation()->getVelDim();
    flat.segment(0, posDim)
        = getRepresentation()->getPositionLowerLimits(world);
    flat.segment(posDim, velDim)
        = getRepresentation()->getVelocityLowerLimits(world);
    cursor = posDim + velDim;
  }
  int forceDim = getRepresentation()->getForceDim();
  Eigen::VectorXd forceLowerLimits
      = getRepresentation()->getForceLowerLimits(world);
  assert(forceDim == forceLowerLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = forceLowerLimits;
    cursor += forceDim;
  }
  assert(cursor == flat.size());

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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.getInitialGuess");
  }
#endif

  flatten(flat, thisLog);

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
    Eigen::Ref<Eigen::MatrixXd> jac,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.backpropJacobianOfFinalState");
  }
#endif

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

  int cursor = getFlatProblemDim();
  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;
    Eigen::MatrixXd forceVel
        = ptr->getForceVelJacobian(world, mRepresentationMapping, thisLog);
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

    cursor -= forceDim;
    jac.block(0, cursor, posDim, forceDim) = thisTimestep.forcePos;
    jac.block(posDim, cursor, velDim, forceDim) = thisTimestep.forceVel;

    if (i == 0 && mTuneStartingState)
    {
      cursor -= velDim;
      assert(cursor == posDim);
      jac.block(0, cursor, posDim, velDim) = thisTimestep.velPos;
      jac.block(posDim, cursor, velDim, velDim) = thisTimestep.velVel;
      cursor -= posDim;
      assert(cursor == 0);
      jac.block(0, cursor, posDim, posDim) = thisTimestep.posPos;
      jac.block(posDim, cursor, velDim, posDim) = thisTimestep.posVel;
    }

    last = thisTimestep;
  }
  assert(cursor == 0);

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

  int dim = getFlatProblemDim();
  Eigen::VectorXd flat = Eigen::VectorXd(dim);
  flatten(flat, nullptr);

  double EPS = 1e-7;

  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(flat, nullptr);
    flat(i) -= EPS;
    Eigen::VectorXd perturbedEndStatePos = getFinalState(world, nullptr);

    flat(i) -= EPS;
    unflatten(flat, nullptr);
    flat(i) += EPS;
    Eigen::VectorXd perturbedEndStateNeg = getFinalState(world, nullptr);

    jac.col(i) = (perturbedEndStatePos - perturbedEndStateNeg) / (2 * EPS);
  }

  // Restore original value
  unflatten(flat, nullptr);
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void SingleShot::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("SingleShot.backpropGradientWrt");
  }
#endif

  int dims = getFlatProblemDim();
  assert(grad.size() == dims);
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

  int cursor = dims;
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
      mappedGrad.lossWrtTorque
          = gradWrtRollout->getForcesConst(pair.first).col(i);
      mappedLosses[pair.first] = mappedGrad;
    }
    mappedLosses[mRepresentationMapping].lossWrtPosition
        += nextTimestep.lossWrtPosition;
    mappedLosses[mRepresentationMapping].lossWrtVelocity
        += nextTimestep.lossWrtVelocity;

    LossGradient thisTimestep;
    snapshots[i]->backprop(world, thisTimestep, mappedLosses, thisLog);
    cursor -= forceDim;
    grad.segment(cursor, forceDim) = thisTimestep.lossWrtTorque;
    if (i == 0 && mTuneStartingState)
    {
      assert(
          cursor
          == mMappings[mRepresentationMapping]->getPosDim()
                 + mMappings[mRepresentationMapping]->getVelDim());
      cursor -= mMappings[mRepresentationMapping]->getVelDim();
      grad.segment(cursor, mMappings[mRepresentationMapping]->getVelDim())
          = thisTimestep.lossWrtVelocity;
      cursor -= mMappings[mRepresentationMapping]->getPosDim();
      grad.segment(cursor, mMappings[mRepresentationMapping]->getPosDim())
          = thisTimestep.lossWrtPosition;
    }
    thisTimestep.lossWrtTorque
        += gradWrtRollout
               ->getForcesConst(gradWrtRollout->getRepresentationMapping())
               .col(i);

    nextTimestep = thisTimestep;
  }
  assert(cursor == 0);

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

#ifdef LOG_PERFORMANCE_SINGLE_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
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
std::string SingleShot::getFlatDimName(int dim)
{
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
  return "Error OOB";
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