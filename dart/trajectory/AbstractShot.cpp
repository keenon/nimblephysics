#include "dart/trajectory/AbstractShot.hpp"

#include <iostream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpReturnCodes.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <dart/gui/gui.hpp>

#include "dart/optimizer/ipopt/ipopt.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace trajectory {

//==============================================================================
/// Default constructor
AbstractShot::AbstractShot(std::shared_ptr<simulation::World> world)
{
  mNumDofs = world->getNumDofs();
  mWorld = world;
}

//==============================================================================
AbstractShot::~AbstractShot()
{
  // std::cout << "Freeing AbstractShot: " << this << std::endl;
}

//==============================================================================
void AbstractShot::setLossFunction(TrajectoryLossFn loss)
{
  mLoss = loss;
}

//==============================================================================
void AbstractShot::setLossFunctionGradient(TrajectoryLossFnGrad grad)
{
  mGrad = grad;
}

//==============================================================================
/// This computes finite difference gradients of (poses, vels, forces)
/// matrices with respect to a passed in loss function. If there aren't
/// analytical gradients of the loss, then this is a useful pre-step for
/// analytically computing the gradients for backprop.
void AbstractShot::bruteForceGradOfLossInputs(
    std::shared_ptr<simulation::World> world,
    TrajectoryLossFn loss,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces)
{
  Eigen::MatrixXd poses = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd vels = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  getStates(world, poses, vels, forces);
  double originalLoss = loss(poses, vels, forces);

  const double EPS = 1e-8;

  for (int row = 0; row < mNumDofs; row++)
  {
    for (int col = 0; col < mSteps; col++)
    {
      poses(row, col) += EPS;
      double lossPos = loss(poses, vels, forces);
      poses(row, col) -= EPS;
      gradWrtPoses(row, col) = (lossPos - originalLoss) / EPS;

      vels(row, col) += EPS;
      double lossVel = loss(poses, vels, forces);
      vels(row, col) -= EPS;
      gradWrtVels(row, col) = (lossVel - originalLoss) / EPS;

      forces(row, col) += EPS;
      double lossForce = loss(poses, vels, forces);
      forces(row, col) -= EPS;
      gradWrtForces(row, col) = (lossForce - originalLoss) / EPS;
    }
  }
}

//==============================================================================
/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void AbstractShot::finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    TrajectoryLossFn loss,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  Eigen::MatrixXd poses = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd vels = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  getStates(world, poses, vels, forces);
  double originalLoss = loss(poses, vels, forces);

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
    double posLoss = loss(poses, vels, forces);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(flat);
    getStates(world, poses, vels, forces);
    double negLoss = loss(poses, vels, forces);
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