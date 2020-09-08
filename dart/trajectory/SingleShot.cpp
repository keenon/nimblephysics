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
    std::shared_ptr<simulation::World> world, int steps, bool tuneStartingState)
  : AbstractShot(world)
{
  mTuneStartingState = tuneStartingState;
  mSteps = steps;
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
/// Returns the length of the flattened problem state
int SingleShot::getFlatProblemDim() const
{
  if (mTuneStartingState)
    return (2 * mNumDofs) + (mSteps * mNumDofs);
  return mSteps * mNumDofs;
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
    flat.segment(0, mNumDofs) = mStartPos;
    flat.segment(mNumDofs, mNumDofs) = mStartVel;
    cursor = 2 * mNumDofs;
  }
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, mNumDofs) = mForces.col(i);
    cursor += mNumDofs;
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
    mStartPos = flat.segment(0, mNumDofs);
    mStartVel = flat.segment(mNumDofs, mNumDofs);
    cursor = 2 * mNumDofs;
  }
  for (int i = 0; i < mSteps; i++)
  {
    mForces.col(i) = flat.segment(cursor, mNumDofs);
    cursor += mNumDofs;
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
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);
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
    flat.segment(0, mNumDofs) = world->getPositionUpperLimits();
    flat.segment(mNumDofs, mNumDofs) = world->getVelocityUpperLimits();
    cursor = 2 * mNumDofs;
  }
  Eigen::VectorXd forceUpperLimits = world->getForceUpperLimits();
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, mNumDofs) = forceUpperLimits;
    cursor += mNumDofs;
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
    flat.segment(0, mNumDofs) = world->getPositionLowerLimits();
    flat.segment(mNumDofs, mNumDofs) = world->getVelocityLowerLimits();
    cursor = 2 * mNumDofs;
  }
  Eigen::VectorXd forceUpperLimits = world->getForceLowerLimits();
  for (int i = 0; i < mSteps; i++)
  {
    flat.segment(cursor, mNumDofs) = forceUpperLimits;
    cursor += mNumDofs;
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
    flat.segment(0, mNumDofs) = mStartPos;
    flat.segment(mNumDofs, mNumDofs) = mStartVel;
  }
}

//==============================================================================
/// This computes the values of the constraints
void SingleShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints)
{
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the
/// constraints. This returns a matrix that's (getConstraintDim(),
/// getFlatProblemDim()).
void SingleShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac)
{
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
void SingleShot::backpropJacobianOfFinalState(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.forcePos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.posVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.posPos = Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
  last.velVel = Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
  last.velPos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);

  int cursor = getFlatProblemDim();
  for (int i = mSteps - 1; i >= 0; i--)
  {
    BackpropSnapshotPtr ptr = snapshots[i];
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

    cursor -= mNumDofs;
    jac.block(0, cursor, mNumDofs, mNumDofs) = thisTimestep.forcePos;
    jac.block(mNumDofs, cursor, mNumDofs, mNumDofs) = thisTimestep.forceVel;

    if (i == 0 && mTuneStartingState)
    {
      cursor -= mNumDofs;
      assert(cursor == mNumDofs);
      jac.block(0, cursor, mNumDofs, mNumDofs) = thisTimestep.velPos;
      jac.block(mNumDofs, cursor, mNumDofs, mNumDofs) = thisTimestep.velVel;
      cursor -= mNumDofs;
      assert(cursor == 0);
      jac.block(0, cursor, mNumDofs, mNumDofs) = thisTimestep.posPos;
      jac.block(mNumDofs, cursor, mNumDofs, mNumDofs) = thisTimestep.posVel;
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
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);
  assert(snapshots.size() == mSteps);
  LossGradient nextTimestep;
  nextTimestep.lossWrtPosition = Eigen::VectorXd::Zero(mNumDofs);
  nextTimestep.lossWrtVelocity = Eigen::VectorXd::Zero(mNumDofs);
  nextTimestep.lossWrtTorque = Eigen::VectorXd::Zero(mNumDofs);

  int cursor = dims;
  for (int i = mSteps - 1; i >= 0; i--)
  {
    nextTimestep.lossWrtPosition += gradWrtPoses.col(i);
    nextTimestep.lossWrtVelocity += gradWrtVels.col(i);
    LossGradient thisTimestep;
    snapshots[i]->backprop(world, thisTimestep, nextTimestep);
    cursor -= mNumDofs;
    grad.segment(cursor, mNumDofs) = thisTimestep.lossWrtTorque;
    if (i == 0 && mTuneStartingState)
    {
      assert(cursor == 2 * mNumDofs);
      cursor -= mNumDofs;
      grad.segment(cursor, mNumDofs) = thisTimestep.lossWrtVelocity;
      cursor -= mNumDofs;
      grad.segment(cursor, mNumDofs) = thisTimestep.lossWrtPosition;
    }
    thisTimestep.lossWrtTorque += gradWrtForces.col(i);
    nextTimestep = thisTimestep;
  }
  assert(cursor == 0);
}

//==============================================================================
/// This returns the snapshots from a fresh unroll
std::vector<BackpropSnapshotPtr> SingleShot::getSnapshots(
    std::shared_ptr<simulation::World> world)
{
  if (mSnapshotsCacheDirty)
  {
    RestorableSnapshot snapshot(world);

    mSnapshotsCache.clear();
    mSnapshotsCache.reserve(mSteps);

    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);

    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(mForces.col(i));
      mSnapshotsCache.push_back(forwardPass(world));
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
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);
  assert(poses.cols() == mSteps);
  assert(poses.rows() == mNumDofs);
  assert(vels.cols() == mSteps);
  assert(vels.rows() == mNumDofs);
  assert(forces.cols() == mSteps);
  assert(forces.rows() == mNumDofs);
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
  Eigen::VectorXd state = Eigen::VectorXd::Zero(mNumDofs * 2);
  state.segment(0, mNumDofs) = mStartPos;
  state.segment(mNumDofs, mNumDofs) = mStartVel;
  return state;
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd SingleShot::getFinalState(
    std::shared_ptr<simulation::World> world)
{
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);

  Eigen::VectorXd state = Eigen::VectorXd::Zero(mNumDofs * 2);
  state.segment(0, mNumDofs)
      = snapshots[snapshots.size() - 1]->getPostStepPosition();
  state.segment(mNumDofs, mNumDofs)
      = snapshots[snapshots.size() - 1]->getPostStepVelocity();

  return state;
}

//==============================================================================
/// This returns the debugging name of a given DOF
std::string SingleShot::getFlatDimName(int dim)
{
  if (mTuneStartingState)
  {
    if (dim < mNumDofs)
    {
      return "Start Pos " + std::to_string(dim);
    }
    dim -= mNumDofs;
    if (dim < mNumDofs)
    {
      return "Start Vel " + std::to_string(dim);
    }
    dim -= mNumDofs;
  }
  for (int i = 0; i < mSteps; i++)
  {
    if (dim < mNumDofs)
    {
      return "Force[" + std::to_string(i) + "] " + std::to_string(dim);
    }
    dim -= mNumDofs;
  }
  return "Error OOB";
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int SingleShot::getNumberNonZeroJacobian()
{
  return 0;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void SingleShot::getJacobianSparsityStructure(
    Eigen::Ref<Eigen::VectorXi> rows, Eigen::Ref<Eigen::VectorXi> cols)
{
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void SingleShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparse)
{
}

//==============================================================================
/// This computes the Jacobians that relate each timestep to the endpoint of
/// the trajectory. For a timestep at time t, this will relate quantities like
/// v_t -> p_end, for example.
TimestepJacobians SingleShot::backpropStartStateJacobians(
    std::shared_ptr<simulation::World> world, bool useFdJacs)
{
  std::vector<BackpropSnapshotPtr> snapshots = getSnapshots(world);

  TimestepJacobians last;
  last.forceVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.forcePos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.posVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  last.posPos = Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
  last.velVel = Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
  last.velPos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);

  for (int i = mSteps - 1; i >= 0; i--)
  {
    BackpropSnapshotPtr ptr = snapshots[i];
    TimestepJacobians thisTimestep;
    Eigen::MatrixXd forceVel
        = useFdJacs ? ptr->finiteDifferenceForceVelJacobian(world)
                    : ptr->getForceVelJacobian(world);
    Eigen::MatrixXd posPos = useFdJacs
                                 ? ptr->finiteDifferencePosPosJacobian(world, 1)
                                 : ptr->getPosPosJacobian(world);
    Eigen::MatrixXd posVel = useFdJacs
                                 ? ptr->finiteDifferencePosVelJacobian(world)
                                 : ptr->getPosVelJacobian(world);
    Eigen::MatrixXd velPos = useFdJacs
                                 ? ptr->finiteDifferenceVelPosJacobian(world, 1)
                                 : ptr->getVelPosJacobian(world);
    Eigen::MatrixXd velVel = useFdJacs
                                 ? ptr->finiteDifferenceVelVelJacobian(world)
                                 : ptr->getVelVelJacobian(world);

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

  world->setPositions(mStartPos);
  world->setVelocities(mStartVel);
  for (int i = 0; i < mSteps; i++)
  {
    world->setForces(mForces.col(i));
    world->step();
  }

  Eigen::VectorXd originalEndPos = world->getPositions();
  Eigen::VectorXd originalEndVel = world->getVelocities();

  TimestepJacobians result;

  // Perturb starting position

  result.posPos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  result.posVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  for (int j = 0; j < mNumDofs; j++)
  {
    Eigen::VectorXd perturbedStartPos = mStartPos;
    perturbedStartPos(j) += EPS;
    world->setPositions(perturbedStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = world->getPositions();
    Eigen::VectorXd perturbedEndVel = world->getVelocities();

    perturbedStartPos = mStartPos;
    perturbedStartPos(j) -= EPS;
    world->setPositions(perturbedStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXd perturbedEndVelNeg = world->getVelocities();

    result.posPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.posVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting velocity

  result.velPos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  result.velVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  for (int j = 0; j < mNumDofs; j++)
  {
    Eigen::VectorXd perturbedStartVel = mStartVel;
    perturbedStartVel(j) += EPS;
    world->setPositions(mStartPos);
    world->setVelocities(perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = world->getPositions();
    Eigen::VectorXd perturbedEndVel = world->getVelocities();

    perturbedStartVel = mStartVel;
    perturbedStartVel(j) -= EPS;
    world->setPositions(mStartPos);
    world->setVelocities(perturbedStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXd perturbedEndVelNeg = world->getVelocities();

    result.velPos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.velVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  // Perturb starting force

  result.forcePos = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  result.forceVel = Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
  for (int j = 0; j < mNumDofs; j++)
  {
    Eigen::VectorXd perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) += EPS;
    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(i == 0 ? perturbedStartForce : mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPos = world->getPositions();
    Eigen::VectorXd perturbedEndVel = world->getVelocities();

    perturbedStartForce = mForces.col(0);
    perturbedStartForce(j) -= EPS;
    world->setPositions(mStartPos);
    world->setVelocities(mStartVel);
    for (int i = 0; i < mSteps; i++)
    {
      world->setForces(i == 0 ? perturbedStartForce : mForces.col(i));
      world->step();
    }

    Eigen::VectorXd perturbedEndPosNeg = world->getPositions();
    Eigen::VectorXd perturbedEndVelNeg = world->getVelocities();

    result.forcePos.col(j) = (perturbedEndPos - perturbedEndPosNeg) / (2 * EPS);
    result.forceVel.col(j) = (perturbedEndVel - perturbedEndVelNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

} // namespace trajectory
} // namespace dart