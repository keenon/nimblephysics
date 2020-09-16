#include "dart/trajectory/AbstractShot.hpp"

#include <iostream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpReturnCodes.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <dart/gui/gui.hpp>

#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/optimizer/ipopt/ipopt.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace trajectory {

//==============================================================================
/// Default constructor
AbstractShot::AbstractShot(
    std::shared_ptr<simulation::World> world, LossFn loss, int steps)
  : mWorld(world),
    mLoss(loss),
    mSteps(steps),
    mScratchPoses(Eigen::MatrixXd::Zero(world->getNumDofs(), steps)),
    mScratchVels(Eigen::MatrixXd::Zero(world->getNumDofs(), steps)),
    mScratchForces(Eigen::MatrixXd::Zero(world->getNumDofs(), steps)),
    mScratchGradWrtPoses(Eigen::MatrixXd::Zero(world->getNumDofs(), steps)),
    mScratchGradWrtVels(Eigen::MatrixXd::Zero(world->getNumDofs(), steps)),
    mScratchGradWrtForces(Eigen::MatrixXd::Zero(world->getNumDofs(), steps))
{
  mRepresentationMapping = std::make_shared<neural::IdentityMapping>(world);
}

//==============================================================================
AbstractShot::~AbstractShot()
{
  // std::cout << "Freeing AbstractShot: " << this << std::endl;
}

//==============================================================================
/// This updates the loss function for this trajectory
void AbstractShot::setLoss(LossFn loss)
{
  mLoss = loss;
}

//==============================================================================
/// Add a custom constraint function to the trajectory
void AbstractShot::addConstraint(LossFn loss)
{
  mConstraints.push_back(loss);
}

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
void AbstractShot::switchRepresentationMapping(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<neural::Mapping> mapping)
{
  // Resize our scratch spaces
  mScratchPoses.resize(mapping->getPosDim(), mSteps);
  mScratchVels.resize(mapping->getVelDim(), mSteps);
  mScratchForces.resize(mapping->getForceDim(), mSteps);
  mScratchGradWrtPoses.resize(mapping->getPosDim(), mSteps);
  mScratchGradWrtVels.resize(mapping->getVelDim(), mSteps);
  mScratchGradWrtForces.resize(mapping->getForceDim(), mSteps);
  // Reset the main representation mapping
  mRepresentationMapping = mapping;
}

//==============================================================================
/// This removes any mapping on the representation, meaning the representation
/// space goes back to the native joint-space.
void AbstractShot::clearRepresentationMapping(
    std::shared_ptr<simulation::World> world)
{
  // Just set the mapping to an Identity map, effectively the same as clearing
  // the mapping.
  std::shared_ptr<neural::IdentityMapping> identity
      = std::make_shared<neural::IdentityMapping>(world);
  switchRepresentationMapping(world, identity);
}

//==============================================================================
/// This adds a mapping through which the loss function can interpret the
/// output. We can have multiple loss mappings at the same time, and loss can
/// use arbitrary combinations of multiple views, as long as it can provide
/// gradients.
void AbstractShot::addLossMapping(
    std::string key, std::shared_ptr<neural::Mapping> mapping)
{
  mLossMappings[key] = mapping;
}

//==============================================================================
/// This returns true if there is a loss mapping at the specified key
bool AbstractShot::hasLossMapping(std::string key)
{
  return mLossMappings.find(key) != mLossMappings.end();
}

//==============================================================================
/// This returns the loss mapping at the specified key
std::shared_ptr<neural::Mapping> AbstractShot::getLossMapping(std::string key)
{
  return mLossMappings[key];
}

//==============================================================================
/// This removes the loss mapping at a particular key
void AbstractShot::removeLossMapping(std::string key)
{
  mLossMappings.erase(key);
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void AbstractShot::getConstraintUpperBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getUpperBound();
  }
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void AbstractShot::getConstraintLowerBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const
{
  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getLowerBound();
  }
}

//==============================================================================
int AbstractShot::getConstraintDim() const
{
  return mConstraints.size();
}

//==============================================================================
/// This computes the values of the constraints, assuming that the constraint
/// vector being passed in is only the size of mConstraints
void AbstractShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints)
{
  assert(constraints.size() == mConstraints.size());

  getStates(
      world,
      /* OUT */ mScratchPoses,
      /* OUT */ mScratchVels,
      /* OUT */ mScratchForces);
  for (int i = 0; i < mConstraints.size(); i++)
  {
    constraints(i)
        = mConstraints[i].getLoss(mScratchPoses, mScratchVels, mScratchForces);
  }
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
void AbstractShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac)
{
  assert(jac.rows() == mConstraints.size());
  assert(jac.cols() == getFlatProblemDim());

  getStates(
      world,
      /* OUT */ mScratchPoses,
      /* OUT */ mScratchVels,
      /* OUT */ mScratchForces);

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(getFlatProblemDim());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        mScratchPoses,
        mScratchVels,
        mScratchForces,
        /* OUT */ mScratchGradWrtPoses,
        /* OUT */ mScratchGradWrtVels,
        /* OUT */ mScratchGradWrtForces);
    grad.setZero();
    backpropGradient(
        world,
        mScratchGradWrtPoses,
        mScratchGradWrtVels,
        mScratchGradWrtForces,
        /* OUT */ grad);
    jac.row(i) = grad;
  }
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int AbstractShot::getNumberNonZeroJacobian()
{
  return mConstraints.size() * getFlatProblemDim();
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void AbstractShot::getJacobianSparsityStructure(
    Eigen::Ref<Eigen::VectorXi> rows, Eigen::Ref<Eigen::VectorXi> cols)
{
  assert(rows.size() == AbstractShot::getNumberNonZeroJacobian());
  assert(cols.size() == AbstractShot::getNumberNonZeroJacobian());
  int cursor = 0;
  // Do row-major ordering
  for (int j = 0; j < mConstraints.size(); j++)
  {
    for (int i = 0; i < getFlatProblemDim(); i++)
    {
      rows(cursor) = j;
      cols(cursor) = i;
      cursor++;
    }
  }
  assert(cursor == AbstractShot::getNumberNonZeroJacobian());
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void AbstractShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparse)
{
  assert(sparse.size() == AbstractShot::getNumberNonZeroJacobian());

  getStates(
      world,
      /* OUT */ mScratchPoses,
      /* OUT */ mScratchVels,
      /* OUT */ mScratchForces);

  sparse.setZero();

  int cursor = 0;
  int n = getFlatProblemDim();
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        mScratchPoses,
        mScratchVels,
        mScratchForces,
        /* OUT */ mScratchGradWrtPoses,
        /* OUT */ mScratchGradWrtVels,
        /* OUT */ mScratchGradWrtForces);
    backpropGradient(
        world,
        mScratchGradWrtPoses,
        mScratchGradWrtVels,
        mScratchGradWrtForces,
        /* OUT */ sparse.segment(cursor, n));
    cursor += n;
  }

  assert(cursor == sparse.size());
}

//==============================================================================
/// This computes the gradient in the flat problem space, automatically
/// computing the gradients of the loss function as part of the call
void AbstractShot::backpropGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  getStates(
      world,
      /* OUT */ mScratchPoses,
      /* OUT */ mScratchVels,
      /* OUT */ mScratchForces);
  mLoss.getLossAndGradient(
      mScratchPoses,
      mScratchVels,
      mScratchForces,
      /* OUT */ mScratchGradWrtPoses,
      /* OUT */ mScratchGradWrtVels,
      /* OUT */ mScratchGradWrtForces);
  backpropGradient(
      world,
      mScratchGradWrtPoses,
      mScratchGradWrtVels,
      mScratchGradWrtForces,
      /* OUT */ grad);
}

//==============================================================================
/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void AbstractShot::finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  Eigen::MatrixXd poses
      = Eigen::MatrixXd::Zero(mRepresentationMapping->getPosDim(), mSteps);
  Eigen::MatrixXd vels
      = Eigen::MatrixXd::Zero(mRepresentationMapping->getVelDim(), mSteps);
  Eigen::MatrixXd forces
      = Eigen::MatrixXd::Zero(mRepresentationMapping->getForceDim(), mSteps);
  getStates(world, poses, vels, forces);
  double originalLoss = mLoss.getLoss(poses, vels, forces);

  int dims = getFlatProblemDim();
  Eigen::VectorXd flat = Eigen::VectorXd::Zero(dims);
  flatten(flat);

  assert(grad.size() == dims);

  const double EPS = 1e-6;

  for (int i = 0; i < dims; i++)
  {
    flat(i) += EPS;
    unflatten(flat);
    getStates(world, poses, vels, forces);
    double posLoss = mLoss.getLoss(poses, vels, forces);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(flat);
    getStates(world, poses, vels, forces);
    double negLoss = mLoss.getLoss(poses, vels, forces);
    flat(i) += EPS;

    grad(i) = (posLoss - negLoss) / (2 * EPS);
  }
}

//==============================================================================
int AbstractShot::getNumSteps()
{
  return mSteps;
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
void AbstractShot::finiteDifferenceJacobian(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  int dim = getFlatProblemDim();
  int numConstraints = getConstraintDim();
  assert(jac.cols() == dim);
  assert(jac.rows() == numConstraints);

  Eigen::VectorXd originalConstraints = Eigen::VectorXd::Zero(numConstraints);
  computeConstraints(world, originalConstraints);
  Eigen::VectorXd flat = Eigen::VectorXd::Zero(dim);
  flatten(flat);

  const double EPS = 1e-7;

  Eigen::VectorXd positiveConstraints = Eigen::VectorXd::Zero(numConstraints);
  Eigen::VectorXd negativeConstraints = Eigen::VectorXd::Zero(numConstraints);
  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(flat);
    computeConstraints(world, positiveConstraints);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(flat);
    computeConstraints(world, negativeConstraints);
    flat(i) += EPS;

    jac.col(i) = (positiveConstraints - negativeConstraints) / (2 * EPS);
  }
}

} // namespace trajectory
} // namespace dart