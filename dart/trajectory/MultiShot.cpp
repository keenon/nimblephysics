#include "dart/trajectory/MultiShot.hpp"

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
MultiShot::MultiShot(
    std::shared_ptr<simulation::World> world,
    int steps,
    int shotLength,
    bool tuneStartingState)
  : AbstractShot(world)
{
  mSteps = steps;
  mShotLength = shotLength;
  mTuneStartingState = tuneStartingState;
  mNumDofs = world->getNumDofs();

  int stepsRemaining = steps;
  bool isFirst = true;
  while (stepsRemaining > 0)
  {
    int shot = std::min(shotLength, stepsRemaining);
    mShots.push_back(std::make_shared<SingleShot>(
        world, shot, !isFirst || tuneStartingState));
    stepsRemaining -= shot;
    isFirst = false;
  }
}

//==============================================================================
MultiShot::~MultiShot()
{
  std::cout << "Freeing MultiShot: " << this << std::endl;
}

//==============================================================================
/// Returns the length of the flattened problem state
int MultiShot::getFlatProblemDim() const
{
  int sum = 0;
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    sum += shot->getFlatProblemDim();
  }
  return sum;
}

//==============================================================================
/// Returns the length of the knot-point constraint vector
int MultiShot::getConstraintDim() const
{
  return (2 * mNumDofs) * (mShots.size() - 1);
}

//==============================================================================
/// This computes the values of the constraints
void MultiShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints)
{
  int cursor = 0;
  for (int i = 1; i < mShots.size(); i++)
  {
    constraints.segment(cursor, mNumDofs * 2)
        = mShots[i - 1]->getFinalState(world) - mShots[i]->getStartState();
    cursor += mNumDofs * 2;
  }
}

//==============================================================================
/// This copies a shot down into a single flat vector
void MultiShot::flatten(/* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->flatten(flat.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This gets the parameters out of a flat vector
void MultiShot::unflatten(const Eigen::Ref<const Eigen::VectorXd>& flat)
{
  int cursor = 0;
  for (std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->unflatten(flat.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This runs the shot out, and writes the positions, velocities, and forces
void MultiShot::unroll(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces)
{
  int cursor = 0;
  for (std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getNumSteps();
    shot->unroll(
        world,
        poses.block(0, cursor, mNumDofs, dim),
        vels.block(0, cursor, mNumDofs, dim),
        forces.block(0, cursor, mNumDofs, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void MultiShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getUpperBounds(world, flat.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void MultiShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getLowerBounds(world, flat.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void MultiShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getInitialGuess(world, flat.segment(cursor, dim));
    cursor += dim;
  }
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the
/// constraints. This returns a matrix that's (getConstraintDim(),
/// getFlatProblemDim()).
void MultiShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac)
{
  assert(jac.cols() == getFlatProblemDim());
  assert(jac.rows() == getConstraintDim());

  int rowCursor = 0;
  int colCursor = 0;
  for (int i = 1; i < mShots.size(); i++)
  {
    int dim = mShots[i - 1]->getFlatProblemDim();
    mShots[i - 1]->backpropJacobianOfFinalState(
        world, jac.block(rowCursor, colCursor, 2 * mNumDofs, dim));
    colCursor += dim;
    jac.block(rowCursor, colCursor, 2 * mNumDofs, 2 * mNumDofs)
        = -1 * Eigen::MatrixXd::Identity(2 * mNumDofs, 2 * mNumDofs);
    rowCursor += 2 * mNumDofs;
  }

  // We don't include the last shot in the constraints, cause it doesn't end in
  // a knot point
  assert(
      colCursor == jac.cols() - mShots[mShots.size() - 1]->getFlatProblemDim());
  assert(rowCursor == jac.rows());
}

//==============================================================================
/// This populates the passed in matrices with the values from this trajectory
void MultiShot::getStates(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces)
{
  assert(poses.cols() == mSteps);
  assert(poses.rows() == mNumDofs);
  assert(vels.cols() == mSteps);
  assert(vels.rows() == mNumDofs);
  assert(forces.cols() == mSteps);
  assert(forces.rows() == mNumDofs);
  int cursor = 0;
  for (int i = 0; i < mShots.size(); i++)
  {
    int steps = mShots[i]->getNumSteps();
    mShots[i]->getStates(
        world,
        poses.block(0, cursor, mNumDofs, steps),
        vels.block(0, cursor, mNumDofs, steps),
        forces.block(0, cursor, mNumDofs, steps));
    cursor += steps;
  }
}

//==============================================================================
/// This returns the concatenation of (start pos, start vel) for convenience
Eigen::VectorXd MultiShot::getStartState()
{
  return mShots[0]->getStartState();
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd MultiShot::getFinalState(
    std::shared_ptr<simulation::World> world)
{
  return mShots[mShots.size() - 1]->getFinalState(world);
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void MultiShot::backpropGradient(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtPoses,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtVels,
    const Eigen::Ref<const Eigen::MatrixXd>& gradWrtForces,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  int cursorDims = 0;
  int cursorSteps = 0;
  for (int i = 0; i < mShots.size(); i++)
  {
    int steps = mShots[i]->getNumSteps();
    int dim = mShots[i]->getFlatProblemDim();
    mShots[i]->backpropGradient(
        world,
        gradWrtPoses.block(0, cursorSteps, mNumDofs, steps),
        gradWrtVels.block(0, cursorSteps, mNumDofs, steps),
        gradWrtVels.block(0, cursorSteps, mNumDofs, steps),
        grad.segment(cursorDims, dim));
    cursorSteps += steps;
    cursorDims += dim;
  }
}

} // namespace trajectory
} // namespace dart