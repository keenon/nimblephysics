#include "dart/trajectory/AbstractShot.hpp"

#include <iostream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpReturnCodes.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <dart/gui/gui.hpp>

#include "dart/optimizer/ipopt/ipopt.hpp"
#include "dart/simulation/World.hpp"

using namespace Ipopt;

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
/// This runs IPOPT to try to minimize loss, subject to our constraints
bool AbstractShot::optimize()
{
  // Create an instance of the IpoptApplication
  //
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", 1e-7);
  app->Options()->SetStringValue(
      "linear_solver",
      "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps, custom

  app->Options()->SetStringValue(
      "hessian_approximation", "limited-memory"); // limited-memory, exacty

  // Just for debugging
  /*
  app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
  app->Options()->SetStringValue("derivative_test", "first-order");
  app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);
  */

  int freq = 1;
  if (freq > 0)
  {
    app->Options()->SetNumericValue("print_frequency_iter", freq);
  }
  else
  {
    app->Options()->SetIntegerValue(
        "print_frequency_iter", std::numeric_limits<int>::infinity());
  }

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded)
  {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return (int)status;
  }

  status = app->OptimizeTNLP(this);

  if (status == Solve_Succeeded)
  {
    // Retrieve some statistics about the solve
    Index iter_count = app->Statistics()->IterationCount();
    std::cout << std::endl
              << std::endl
              << "*** The problem solved in " << iter_count << " iterations!"
              << std::endl;

    Number final_obj = app->Statistics()->FinalObjective();
    std::cout << std::endl
              << std::endl
              << "*** The final value of the objective function is "
              << final_obj << '.' << std::endl;
  }

  if (status == Ipopt::Solve_Succeeded)
    return true;
  else
    return false;
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

  const double EPS = 1e-6;

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

//==============================================================================
bool AbstractShot::get_nlp_info(
    Ipopt::Index& n,
    Ipopt::Index& m,
    Ipopt::Index& nnz_jac_g,
    Ipopt::Index& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  // Set the number of decision variables
  n = getFlatProblemDim();

  // Set the total number of constraints
  m = getConstraintDim();

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = n * m;

  // Set the number of entries in the Hessian
  nnz_h_lag = n * n;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

//==============================================================================
bool AbstractShot::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(static_cast<std::size_t>(n) == getFlatProblemDim());
  assert(static_cast<std::size_t>(m) == getConstraintDim());

  // lower and upper bounds
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  getUpperBounds(mWorld, upperBounds);
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  getLowerBounds(mWorld, lowerBounds);

  /*
  for (Ipopt::Index i = 0; i < n; i++)
  {
    x_l[i] = problem->getLowerBounds()[i];
    x_u[i] = problem->getUpperBounds()[i];
  }
  */

  // Add inequality constraint functions
  for (std::size_t i = 0; i < getConstraintDim(); ++i)
  {
    g_l[i] = g_u[i] = 0.0;

    // Ipopt interprets any number greater than nlp_upper_bound_inf as
    // infinity. The default value of nlp_upper_bound_inf and
    // nlp_lower_bound_inf is 1e+19 and can be changed through ipopt options.
    //
    // If we wanted to set an inequality constraint, we could say:
    //
    // g_l[i] = -std::numeric_limits<double>::infinity();
    // g_u[i] = 0;
  }

  return true;
}

//==============================================================================
bool AbstractShot::get_starting_point(
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number* x,
    bool init_z,
    Ipopt::Number* /*z_L*/,
    Ipopt::Number* /*z_U*/,
    Ipopt::Index /*m*/,
    bool init_lambda,
    Ipopt::Number* /*lambda*/)
{
  // If init_x is true, this method must provide an initial value for x.
  if (init_x)
  {
    Eigen::Map<Eigen::VectorXd> x_vec(x, n);
    getInitialGuess(mWorld, x_vec);
  }

  // If init_z is true, this method must provide an initial value for the bound
  // multipliers z^L and z^U
  if (init_z)
  {
    // TODO(JS): Not implemented yet.
    std::cout << "Initializing lower/upper bounds for z is not supported yet. "
              << "Ignored here.\n";
  }

  // If init_lambda is true, this method must provide an initial value for the
  // constraint multipliers, lambda.
  if (init_lambda)
  {
    // TODO(JS): Not implemented yet.
    std::cout << "Initializing lambda is not supported yet. "
              << "Ignored here.\n";
  }

  return true;
}

//==============================================================================
bool AbstractShot::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  assert(_n == getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    unflatten(flat);
  }
  Eigen::MatrixXd poses = Eigen::MatrixXd(mNumDofs, mSteps);
  Eigen::MatrixXd vels = Eigen::MatrixXd(mNumDofs, mSteps);
  Eigen::MatrixXd forces = Eigen::MatrixXd(mNumDofs, mSteps);
  unroll(mWorld, poses, vels, forces);
  _obj_value = mLoss(poses, vels, forces);

  return true;
}

//==============================================================================
bool AbstractShot::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  assert(_n == getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    unflatten(flat);
  }
  Eigen::MatrixXd gradWrtPoses = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd gradWrtVels = Eigen::MatrixXd::Zero(mNumDofs, mSteps);
  Eigen::MatrixXd gradWrtForces = Eigen::MatrixXd::Zero(mNumDofs, mSteps);

  bruteForceGradOfLossInputs(
      mWorld, mLoss, gradWrtPoses, gradWrtVels, gradWrtForces);

  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);
  backpropGradient(mWorld, gradWrtPoses, gradWrtVels, gradWrtForces, grad);

  return true;
}

//==============================================================================
bool AbstractShot::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  assert(_n == getFlatProblemDim());
  assert(_m == getConstraintDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    unflatten(flat);
  }
  Eigen::Map<Eigen::VectorXd> constraints(_g, _m);
  computeConstraints(mWorld, constraints);

  return true;
}

//==============================================================================
bool AbstractShot::eval_jac_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Index /*_nele_jac*/,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  // If the iRow and jCol arguments are not nullptr, then IPOPT wants you to
  // fill in the sparsity structure of the Jacobian (the row and column indices
  // only). At this time, the x argument and the values argument will be
  // nullptr.

  if (nullptr == _values)
  {
    // return the structure of the Jacobian
    assert(_n == getFlatProblemDim());
    assert(_m == getConstraintDim());

    // Assume the gradient is dense
    std::size_t idx = 0;
    for (int i = 0; i < _m; ++i)
    {
      for (int j = 0; j < _n; ++j)
      {
        _iRow[idx] = i;
        _jCol[idx] = j;
        ++idx;
      }
    }
  }
  else
  {
    if (_new_x && _n > 0)
    {
      Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
      unflatten(flat);
    }
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(_m, _n);
    backpropJacobian(mWorld, jac);

    std::size_t idx = 0;
    for (int i = 0; i < _m; ++i)
    {
      for (int j = 0; j < _n; ++j)
      {
        _values[idx] = jac(i, j);
        idx++;
      }
    }
  }

  return true;
}

//==============================================================================
bool AbstractShot::eval_h(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number _obj_factor,
    Ipopt::Index _m,
    const Ipopt::Number* _lambda,
    bool _new_lambda,
    Ipopt::Index _nele_hess,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  // TODO(JS): Not implemented yet.
  std::cout << "[AbstractShot::eval_h] Not implemented yet.\n";

  /*
  return TNLP::eval_h(
      _n,
      _x,
      _new_x,
      _obj_factor,
      _m,
      _lambda,
      _new_lambda,
      _nele_hess,
      _iRow,
      _jCol,
      _values);
      */
  return false;
}

//==============================================================================
void AbstractShot::finalize_solution(
    Ipopt::SolverReturn /*_status*/,
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    const Ipopt::Number* /*_z_L*/,
    const Ipopt::Number* /*_z_U*/,
    Ipopt::Index /*_m*/,
    const Ipopt::Number* /*_g*/,
    const Ipopt::Number* /*_lambda*/,
    Ipopt::Number _obj_value,
    const Ipopt::IpoptData* /*_ip_data*/,
    Ipopt::IpoptCalculatedQuantities* /*_ip_cq*/)
{
  Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
  unflatten(flat);
  /*
  const std::shared_ptr<Problem>& problem = mSolver->getProblem();

  // Store optimal and optimum values
  problem->setOptimumValue(_obj_value);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(_n);
  for (int i = 0; i < _n; ++i)
    x[i] = _x[i];
  problem->setOptimalSolution(x);
  */
}

} // namespace trajectory
} // namespace dart