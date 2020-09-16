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
    std::shared_ptr<neural::Mapping> mapping)
{
  RestorableSnapshot snapshot(world);

  // Rewrite the forces in the new mapping
  Eigen::MatrixXd newForces
      = Eigen::MatrixXd::Zero(mapping->getForceDim(), mForces.cols());
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);
  for (int i = 0; i < snapshots.size(); i++)
  {
    // Set the state in the old mapping
    // TODO:optimize
    Eigen::VectorXd posCopy = snapshots[i]->getPreStepPosition();
    Eigen::VectorXd velCopy = snapshots[i]->getPreStepVelocity();
    Eigen::VectorXd forceCopy = snapshots[i]->getPreStepTorques();
    mRepresentationMapping->setPositions(world, posCopy);
    mRepresentationMapping->setVelocities(world, velCopy);
    mRepresentationMapping->setForces(world, forceCopy);

    // Read back the forces in the new mapping
    newForces.col(i) = mapping->getForces(world);
  }
  mForces = newForces;

  // Rewrite the start state in the new mapping
  mRepresentationMapping->setPositions(world, mStartPos);
  mRepresentationMapping->setVelocities(world, mStartVel);
  Eigen::VectorXd forceCopy = snapshots[0]->getPreStepTorques();
  mRepresentationMapping->setForces(world, forceCopy);
  mStartPos = mapping->getPositions(world);
  mStartVel = mapping->getVelocities(world);

  mSnapshotsCacheDirty = true;
  AbstractShot::switchRepresentationMapping(world, mapping);
  snapshot.restore();
}

//==============================================================================
/// Returns the length of the flattened problem state
int SingleShot::getFlatProblemDim() const
{
  if (mTuneStartingState)
    return (mRepresentationMapping->getPosDim()
            + mRepresentationMapping->getVelDim()) // Initial state
           + (mSteps * mRepresentationMapping->getForceDim());
  return mSteps * mRepresentationMapping->getForceDim();
}

//==============================================================================
int SingleShot::getConstraintDim() const
{
  return 0;
}

//==============================================================================
/// This copies a shot down into a single flat vector
void SingleShot::flatten(Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  if (mTuneStartingState)
  {
    flat.segment(0, mRepresentationMapping->getPosDim()) = mStartPos;
    cursor += mRepresentationMapping->getPosDim();
    flat.segment(cursor, mRepresentationMapping->getVelDim()) = mStartVel;
    cursor += mRepresentationMapping->getVelDim();
  }
  int forceDim = mRepresentationMapping->getForceDim();
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = mForces.col(i);
    cursor += forceDim;
  }
  assert(cursor == flat.size());
}

//==============================================================================
/// This gets the parameters out of a flat vector
void SingleShot::unflatten(const Eigen::Ref<const Eigen::VectorXd>& flat)
{
  mSnapshotsCacheDirty = true;
  int cursor = 0;
  if (mTuneStartingState)
  {
    mStartPos = flat.segment(0, mRepresentationMapping->getPosDim());
    cursor += mRepresentationMapping->getPosDim();
    mStartVel = flat.segment(cursor, mRepresentationMapping->getVelDim());
    cursor += mRepresentationMapping->getVelDim();
  }
  int forceDim = mRepresentationMapping->getForceDim();
  for (int i = 0; i < mSteps; i++)
  {
    mForces.col(i) = flat.segment(cursor, forceDim);
    cursor += forceDim;
  }
  assert(cursor == flat.size());
}

//==============================================================================
/// This runs the shot out, and writes the positions, velocities, and forces
void SingleShot::unroll(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces)
{
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);
  for (int i = 0; i < mSteps; i++)
  {
    poses.col(i) = snapshots[i]->getPostStepPosition();
    vels.col(i) = snapshots[i]->getPostStepVelocity();
    forces.col(i) = snapshots[i]->getPostStepTorques();
  }
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void SingleShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  if (mTuneStartingState)
  {
    int posDim = mRepresentationMapping->getPosDim();
    int velDim = mRepresentationMapping->getVelDim();
    flat.segment(0, posDim)
        = mRepresentationMapping->getPositionUpperLimits(world);
    flat.segment(posDim, velDim)
        = mRepresentationMapping->getVelocityUpperLimits(world);
    cursor = posDim + velDim;
  }
  int forceDim = mRepresentationMapping->getForceDim();
  Eigen::VectorXd forceUpperLimits = world->getForceUpperLimits();
  assert(forceDim == forceUpperLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = forceUpperLimits;
    cursor += forceDim;
  }
  assert(cursor == flat.size());
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void SingleShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  if (mTuneStartingState)
  {
    int posDim = mRepresentationMapping->getPosDim();
    int velDim = mRepresentationMapping->getVelDim();
    flat.segment(0, posDim)
        = mRepresentationMapping->getPositionLowerLimits(world);
    flat.segment(posDim, velDim)
        = mRepresentationMapping->getVelocityLowerLimits(world);
    cursor = posDim + velDim;
  }
  int forceDim = mRepresentationMapping->getForceDim();
  Eigen::VectorXd forceLowerLimits
      = mRepresentationMapping->getForceLowerLimits(world);
  assert(forceDim == forceLowerLimits.size());
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, forceDim) = forceLowerLimits;
    cursor += forceDim;
  }
  assert(cursor == flat.size());
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void SingleShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  flat.setZero();
  if (mTuneStartingState)
  {
    flat.segment(0, mRepresentationMapping->getPosDim()) = mStartPos;
    flat.segment(
        mRepresentationMapping->getPosDim(),
        mRepresentationMapping->getVelDim())
        = mStartVel;
  }
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
void SingleShot::backpropJacobianOfFinalState(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);

  int posDim = mRepresentationMapping->getPosDim();
  int velDim = mRepresentationMapping->getVelDim();
  int forceDim = mRepresentationMapping->getForceDim();

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
    Eigen::MatrixXd forceVel = ptr->getForceVelJacobian(world);
    Eigen::MatrixXd posPos = ptr->getPosPosJacobian(world);
    Eigen::MatrixXd posVel = ptr->getPosVelJacobian(world);
    Eigen::MatrixXd velPos = ptr->getVelPosJacobian(world);
    Eigen::MatrixXd velVel = ptr->getVelVelJacobian(world);

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
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
void SingleShot::finiteDifferenceJacobianOfFinalState(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  Eigen::VectorXd originalEndPos = getFinalState(world);

  int dim = getFlatProblemDim();
  Eigen::VectorXd flat = Eigen::VectorXd(dim);
  flatten(flat);

  double EPS = 1e-7;

  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(flat);
    flat(i) -= EPS;
    Eigen::VectorXd perturbedEndStatePos = getFinalState(world);

    flat(i) -= EPS;
    unflatten(flat);
    flat(i) += EPS;
    Eigen::VectorXd perturbedEndStateNeg = getFinalState(world);

    jac.col(i) = (perturbedEndStatePos - perturbedEndStateNeg) / (2 * EPS);
  }

  // Restore original value
  unflatten(flat);
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void SingleShot::backpropGradient(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtPoses,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtVels,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtForces,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  int dims = getFlatProblemDim();
  assert(grad.size() == dims);
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);
  assert(snapshots.size() == mSteps);
  LossGradient nextTimestep;
  nextTimestep.lossWrtPosition
      = Eigen::VectorXd::Zero(mRepresentationMapping->getPosDim());
  nextTimestep.lossWrtVelocity
      = Eigen::VectorXd::Zero(mRepresentationMapping->getVelDim());
  nextTimestep.lossWrtTorque
      = Eigen::VectorXd::Zero(mRepresentationMapping->getForceDim());

  int cursor = dims;
  int forceDim = mRepresentationMapping->getForceDim();
  for (int i = mSteps - 1; i >= 0; i--)
  {
    nextTimestep.lossWrtPosition += gradWrtPoses.col(i);
    nextTimestep.lossWrtVelocity += gradWrtVels.col(i);
    LossGradient thisTimestep;
    snapshots[i]->backprop(world, thisTimestep, nextTimestep);
    cursor -= forceDim;
    grad.segment(cursor, forceDim) = thisTimestep.lossWrtTorque;
    if (i == 0 && mTuneStartingState)
    {
      assert(
          cursor
          == mRepresentationMapping->getPosDim()
                 + mRepresentationMapping->getVelDim());
      cursor -= mRepresentationMapping->getVelDim();
      grad.segment(cursor, mRepresentationMapping->getVelDim())
          = thisTimestep.lossWrtVelocity;
      cursor -= mRepresentationMapping->getPosDim();
      grad.segment(cursor, mRepresentationMapping->getPosDim())
          = thisTimestep.lossWrtPosition;
    }
    thisTimestep.lossWrtTorque += gradWrtForces.col(i);
    nextTimestep = thisTimestep;
  }
  assert(cursor == 0);
}

//==============================================================================
/// This returns the snapshots from a fresh unroll
std::vector<MappedBackpropSnapshotPtr> SingleShot::getSnapshots(
    std::shared_ptr<simulation::World> world)
{
  if (mSnapshotsCacheDirty)
  {
    RestorableSnapshot snapshot(world);

    mSnapshotsCache.clear();
    mSnapshotsCache.reserve(mSteps);

    mRepresentationMapping->setPositions(world, mStartPos);
    mRepresentationMapping->setVelocities(world, mStartVel);

    for (int i = 0; i < mSteps; i++)
    {
      mRepresentationMapping->setForces(world, mForces.col(i));
      mSnapshotsCache.push_back(
          forwardPass(world, mRepresentationMapping, mLossMappings));
    }

    snapshot.restore();
    mSnapshotsCacheDirty = false;
  }
  return mSnapshotsCache;
}

//==============================================================================
/// This populates the passed in matrices with the values from this trajectory
void SingleShot::getStates(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces,
    bool /* useKnots */)
{
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);
  assert(poses.cols() == mSteps);
  assert(poses.rows() == mRepresentationMapping->getPosDim());
  assert(vels.cols() == mSteps);
  assert(vels.rows() == mRepresentationMapping->getVelDim());
  assert(forces.cols() == mSteps);
  assert(forces.rows() == mRepresentationMapping->getForceDim());
  for (int i = 0; i < mSteps; i++)
  {
    poses.col(i) = snapshots[i]->getPostStepPosition();
    vels.col(i) = snapshots[i]->getPostStepVelocity();
    forces.col(i) = snapshots[i]->getPreStepTorques();
  }
}

//==============================================================================
Eigen::VectorXd SingleShot::getStartState()
{
  Eigen::VectorXd state = Eigen::VectorXd::Zero(
      mRepresentationMapping->getPosDim()
      + mRepresentationMapping->getVelDim());
  state.segment(0, mRepresentationMapping->getPosDim()) = mStartPos;
  state.segment(
      mRepresentationMapping->getPosDim(), mRepresentationMapping->getVelDim())
      = mStartVel;
  return state;
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd SingleShot::getFinalState(
    std::shared_ptr<simulation::World> world)
{
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);

  Eigen::VectorXd state = Eigen::VectorXd::Zero(
      mRepresentationMapping->getPosDim()
      + mRepresentationMapping->getVelDim());
  state.segment(0, mRepresentationMapping->getPosDim())
      = snapshots[snapshots.size() - 1]->getPostStepPosition();
  state.segment(
      mRepresentationMapping->getPosDim(), mRepresentationMapping->getVelDim())
      = snapshots[snapshots.size() - 1]->getPostStepVelocity();

  return state;
}

//==============================================================================
/// This returns the debugging name of a given DOF
std::string SingleShot::getFlatDimName(int dim)
{
  if (mTuneStartingState)
  {
    if (dim < mRepresentationMapping->getPosDim())
    {
      return "Start Pos " + std::to_string(dim);
    }
    dim -= mRepresentationMapping->getPosDim();
    if (dim < mRepresentationMapping->getVelDim())
    {
      return "Start Vel " + std::to_string(dim);
    }
    dim -= mRepresentationMapping->getVelDim();
  }
  int forceDim = mRepresentationMapping->getForceDim();
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
  std::vector<MappedBackpropSnapshotPtr> snapshots = getSnapshots(world);

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXd::Zero(
      mRepresentationMapping->getVelDim(),
      mRepresentationMapping->getForceDim());
  last.forcePos = Eigen::MatrixXd::Zero(
      mRepresentationMapping->getPosDim(),
      mRepresentationMapping->getForceDim());
  last.posVel = Eigen::MatrixXd::Zero(
      mRepresentationMapping->getVelDim(), mRepresentationMapping->getPosDim());
  last.posPos = Eigen::MatrixXd::Identity(
      mRepresentationMapping->getPosDim(), mRepresentationMapping->getPosDim());
  last.velVel = Eigen::MatrixXd::Identity(
      mRepresentationMapping->getVelDim(), mRepresentationMapping->getVelDim());
  last.velPos = Eigen::MatrixXd::Zero(
      mRepresentationMapping->getPosDim(), mRepresentationMapping->getVelDim());

  for (int i = mSteps - 1; i >= 0; i--)
  {
    MappedBackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;
    Eigen::MatrixXd forceVel = ptr->getForceVelJacobian(world);
    Eigen::MatrixXd posPos = ptr->getPosPosJacobian(world);
    Eigen::MatrixXd posVel = ptr->getPosVelJacobian(world);
    Eigen::MatrixXd velPos = ptr->getVelPosJacobian(world);
    Eigen::MatrixXd velVel = ptr->getVelVelJacobian(world);

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

  mRepresentationMapping->setPositions(world, mStartPos);
  mRepresentationMapping->setVelocities(world, mStartVel);
  for (int i = 0; i < mSteps; i++)
  {
    mRepresentationMapping->setForces(world, mForces.col(i));
    world->step();
  }

  Eigen::VectorXd originalEndPos = mRepresentationMapping->getPositions(world);
  Eigen::VectorXd originalEndVel = mRepresentationMapping->getVelocities(world);

  TimestepJacobians result;

  int posDim = mRepresentationMapping->getPosDim();
  int velDim = mRepresentationMapping->getVelDim();
  int forceDim = mRepresentationMapping->getForceDim();

  // Perturb starting position

  result.posPos = Eigen::MatrixXd::Zero(posDim, posDim);
  result.posVel = Eigen::MatrixXd::Zero(velDim, posDim);
  for (int j = 0; j < posDim; j++)
  {
    Eigen::VectorXd perturbedStartPos = mStartPos;
    perturbedStartPos(j) += EPS;
    mRepresentationMapping->setPositions(world, perturbedStartPos);
    mRepresentationMapping->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      mRepresentationMapping->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVel
        = mRepresentationMapping->getVelocities(world);

    perturbedStartPos = mStartPos;
    perturbedStartPos(j) -= EPS;
    mRepresentationMapping->setPositions(world, perturbedStartPos);
    mRepresentationMapping->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      mRepresentationMapping->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = mRepresentationMapping->getVelocities(world);

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
    mRepresentationMapping->setPositions(world, mStartPos);
    mRepresentationMapping->setVelocities(world, perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      mRepresentationMapping->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVel
        = mRepresentationMapping->getVelocities(world);

    perturbedStartVel = mStartVel;
    perturbedStartVel(j) -= EPS;
    mRepresentationMapping->setPositions(world, mStartPos);
    mRepresentationMapping->setVelocities(world, perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      mRepresentationMapping->setForces(world, mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = mRepresentationMapping->getVelocities(world);

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
    mRepresentationMapping->setPositions(world, mStartPos);
    mRepresentationMapping->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXd force = i == 0 ? perturbedStartForce : mForces.col(i);
      mRepresentationMapping->setForces(world, force);
      world->step();
    }

    Eigen::VectorXd perturbedEndPos
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVel
        = mRepresentationMapping->getVelocities(world);

    perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) -= EPS;
    mRepresentationMapping->setPositions(world, mStartPos);
    mRepresentationMapping->setVelocities(world, mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      Eigen::VectorXd force = i == 0 ? perturbedStartForce : mForces.col(i);
      mRepresentationMapping->setForces(world, force);
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg
        = mRepresentationMapping->getPositions(world);
    Eigen::VectorXd perturbedEndVelNeg
        = mRepresentationMapping->getVelocities(world);

    result.forcePos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.forceVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

} // namespace trajectory
} // namespace dart