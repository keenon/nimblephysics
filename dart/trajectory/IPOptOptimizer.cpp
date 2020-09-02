#include "dart/trajectory/IPOptOptimizer.hpp"

#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

using namespace dart;
using namespace simulation;

using namespace Ipopt;

namespace dart {
namespace trajectory {

//==============================================================================
IPOptOptimizer::IPOptOptimizer()
{
}

//==============================================================================
bool IPOptOptimizer::optimize(AbstractShot* shot)
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

  app->Options()->SetIntegerValue("max_iter", 40);

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

  // This will automatically free the problem object when finished,
  // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will crash.
  // If you try to leave `problem` on the stack, you'll get invalid free
  // exceptions when IPOpt attempts to free it.
  IPOptShotWrapper* problem = new IPOptShotWrapper(shot);
  SmartPtr<IPOptShotWrapper> problemPtr(problem);
  status = app->OptimizeTNLP(problemPtr);

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
IPOptShotWrapper::IPOptShotWrapper(AbstractShot* wrapped)
{
  mWrapped = wrapped;
}

//==============================================================================
IPOptShotWrapper::~IPOptShotWrapper()
{
  // std::cout << "Freeing IPOptShotWrapper: " << this << std::endl;
}

//==============================================================================
bool IPOptShotWrapper::get_nlp_info(
    Ipopt::Index& n,
    Ipopt::Index& m,
    Ipopt::Index& nnz_jac_g,
    Ipopt::Index& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  // Set the number of decision variables
  n = mWrapped->getFlatProblemDim();

  // Set the total number of constraints
  m = mWrapped->getConstraintDim();

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = n * m;

  // Set the number of entries in the Hessian
  nnz_h_lag = n * n;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

//==============================================================================
bool IPOptShotWrapper::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(static_cast<std::size_t>(n) == mWrapped->getFlatProblemDim());
  assert(static_cast<std::size_t>(m) == mWrapped->getConstraintDim());

  // lower and upper bounds
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  mWrapped->getUpperBounds(mWrapped->mWorld, upperBounds);
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  mWrapped->getLowerBounds(mWrapped->mWorld, lowerBounds);

  /*
  for (Ipopt::Index i = 0; i < n; i++)
  {
    x_l[i] = problem->getLowerBounds()[i];
    x_u[i] = problem->getUpperBounds()[i];
  }
  */

  // Add inequality constraint functions
  for (std::size_t i = 0; i < mWrapped->getConstraintDim(); ++i)
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
bool IPOptShotWrapper::get_starting_point(
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
    mWrapped->getInitialGuess(mWrapped->mWorld, x_vec);
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
bool IPOptShotWrapper::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  assert(_n == mWrapped->getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat);
  }
  Eigen::MatrixXd poses = Eigen::MatrixXd(mWrapped->mNumDofs, mWrapped->mSteps);
  Eigen::MatrixXd vels = Eigen::MatrixXd(mWrapped->mNumDofs, mWrapped->mSteps);
  Eigen::MatrixXd forces
      = Eigen::MatrixXd(mWrapped->mNumDofs, mWrapped->mSteps);
  mWrapped->unroll(mWrapped->mWorld, poses, vels, forces);
  _obj_value = mWrapped->mLoss(poses, vels, forces);

  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  assert(_n == mWrapped->getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat);
  }
  Eigen::MatrixXd gradWrtPoses
      = Eigen::MatrixXd::Zero(mWrapped->mNumDofs, mWrapped->mSteps);
  Eigen::MatrixXd gradWrtVels
      = Eigen::MatrixXd::Zero(mWrapped->mNumDofs, mWrapped->mSteps);
  Eigen::MatrixXd gradWrtForces
      = Eigen::MatrixXd::Zero(mWrapped->mNumDofs, mWrapped->mSteps);

  mWrapped->bruteForceGradOfLossInputs(
      mWrapped->mWorld,
      mWrapped->mLoss,
      gradWrtPoses,
      gradWrtVels,
      gradWrtForces);

  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);
  mWrapped->backpropGradient(
      mWrapped->mWorld, gradWrtPoses, gradWrtVels, gradWrtForces, grad);

  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  assert(_n == mWrapped->getFlatProblemDim());
  assert(_m == mWrapped->getConstraintDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat);
  }
  Eigen::Map<Eigen::VectorXd> constraints(_g, _m);
  mWrapped->computeConstraints(mWrapped->mWorld, constraints);

  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_jac_g(
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
    assert(_n == mWrapped->getFlatProblemDim());
    assert(_m == mWrapped->getConstraintDim());

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
      mWrapped->unflatten(flat);
    }
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(_m, _n);
    mWrapped->backpropJacobian(mWrapped->mWorld, jac);

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
bool IPOptShotWrapper::eval_h(
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
  std::cout << "[IPOptShotWrapper::eval_h] Not implemented yet.\n";

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
void IPOptShotWrapper::finalize_solution(
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
  mWrapped->unflatten(flat);
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