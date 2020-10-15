#include "dart/trajectory/IPOptOptimizer.hpp"

#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/performance/PerformanceLog.hpp"

#define LOG_PERFORMANCE_IPOPT

using namespace dart;
using namespace simulation;
using namespace performance;

using namespace Ipopt;

namespace dart {
namespace trajectory {

//==============================================================================
IPOptOptimizer::IPOptOptimizer()
  : mTolerance(1e-7),
    mLBFGSHistoryLength(1),
    mPrintFrequency(1),
    mIterationLimit(100),
    mCheckDerivatives(false),
    mRecordPerfLog(false)
{
}

//==============================================================================
std::shared_ptr<OptimizationRecord> IPOptOptimizer::optimize(AbstractShot* shot)
{
  // Create an instance of the IpoptApplication
  //
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", mTolerance);
  app->Options()->SetStringValue(
      "linear_solver",
      "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps, custom

  app->Options()->SetStringValue(
      "hessian_approximation", "limited-memory"); // limited-memory, exacty

  app->Options()->SetStringValue(
      "scaling_method", "none"); // none, gradient-based

  app->Options()->SetIntegerValue("max_iter", mIterationLimit);

  // Disable LBFGS history
  app->Options()->SetIntegerValue(
      "limited_memory_max_history", mLBFGSHistoryLength);

  // Just for debugging
  if (mCheckDerivatives)
  {
    app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
    app->Options()->SetStringValue("derivative_test", "first-order");
    app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);
  }

  if (mPrintFrequency > 0)
  {
    app->Options()->SetNumericValue("print_frequency_iter", mPrintFrequency);
  }
  else
  {
    app->Options()->SetIntegerValue(
        "print_frequency_iter", std::numeric_limits<int>::infinity());
  }

  std::shared_ptr<OptimizationRecord> record
      = std::make_shared<OptimizationRecord>();
  if (mRecordPerfLog)
    record->startPerfLog();

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded)
  {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return record;
  }

  // This will automatically free the problem object when finished,
  // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will crash.
  // If you try to leave `problem` on the stack, you'll get invalid free
  // exceptions when IPOpt attempts to free it.
  IPOptShotWrapper* problem = new IPOptShotWrapper(shot, record);
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

  record->setSuccess(status == Ipopt::Solve_Succeeded);

  return record;
}

//==============================================================================
void IPOptOptimizer::setIterationLimit(int iterationLimit)
{
  mIterationLimit = iterationLimit;
}

//==============================================================================
void IPOptOptimizer::setTolerance(double tolerance)
{
  mTolerance = tolerance;
}

//==============================================================================
void IPOptOptimizer::setLBFGSHistoryLength(int historyLen)
{
  mLBFGSHistoryLength = historyLen;
}

//==============================================================================
void IPOptOptimizer::setCheckDerivatives(bool checkDerivatives)
{
  mCheckDerivatives = checkDerivatives;
}

//==============================================================================
void IPOptOptimizer::setPrintFrequency(int frequency)
{
  mPrintFrequency = frequency;
}

void IPOptOptimizer::setRecordPerformanceLog(bool recordPerfLog)
{
  mRecordPerfLog = recordPerfLog;
}

//==============================================================================
IPOptShotWrapper::IPOptShotWrapper(
    AbstractShot* wrapped, std::shared_ptr<OptimizationRecord> record)
  : mBestFeasibleObjectiveValue(1e20),
    mWrapped(wrapped),
    mBestIter(0),
    mRecord(record)
{
  mBestFeasibleState = Eigen::VectorXd(mWrapped->getFlatProblemDim());
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
  nnz_jac_g = mWrapped->getNumberNonZeroJacobian();

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
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog
        = mRecord->getPerfLog()->startRun("IPOptShotWrapper.get_bound_info");
  }
#endif

  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(static_cast<std::size_t>(n) == mWrapped->getFlatProblemDim());
  assert(static_cast<std::size_t>(m) == mWrapped->getConstraintDim());

  // lower and upper bounds
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  mWrapped->getUpperBounds(mWrapped->mWorld, upperBounds, perflog);
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  mWrapped->getLowerBounds(mWrapped->mWorld, lowerBounds, perflog);

  /*
  for (Ipopt::Index i = 0; i < n; i++)
  {
    x_l[i] = problem->getLowerBounds()[i];
    x_u[i] = problem->getUpperBounds()[i];
  }
  */

  // Add inequality constraint functions
  Eigen::Map<Eigen::VectorXd> constraintUpperBounds(g_u, m);
  mWrapped->getConstraintUpperBounds(constraintUpperBounds, perflog);
  Eigen::Map<Eigen::VectorXd> constraintLowerBounds(g_l, m);
  mWrapped->getConstraintLowerBounds(constraintLowerBounds, perflog);

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
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
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun(
        "IPOptShotWrapper.get_starting_point");
  }
#endif

  // If init_x is true, this method must provide an initial value for x.
  if (init_x)
  {
    Eigen::Map<Eigen::VectorXd> x_vec(x, n);
    mWrapped->getInitialGuess(mWrapped->mWorld, x_vec, perflog);
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

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_f");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat, perflog);
  }
  _obj_value = mWrapped->getLoss(mWrapped->mWorld, perflog);

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_grad_f");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat, perflog);
  }
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);
  mWrapped->backpropGradient(mWrapped->mWorld, grad, perflog);

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
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
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_g");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim());
  assert(_m == mWrapped->getConstraintDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(flat, perflog);
  }
  Eigen::Map<Eigen::VectorXd> constraints(_g, _m);
  mWrapped->computeConstraints(mWrapped->mWorld, constraints, perflog);

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
  return true;
}

//==============================================================================
bool IPOptShotWrapper::eval_jac_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Index _nnzj,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_jac_g");
  }
#endif

  // If the iRow and jCol arguments are not nullptr, then IPOPT wants you to
  // fill in the sparsity structure of the Jacobian (the row and column indices
  // only). At this time, the x argument and the values argument will be
  // nullptr.

  if (nullptr == _values)
  {
    // return the structure of the Jacobian
    assert(_n == mWrapped->getFlatProblemDim());
    assert(_m == mWrapped->getConstraintDim());
    assert(_nnzj == mWrapped->getNumberNonZeroJacobian());

    Eigen::Map<Eigen::VectorXi> rows(_iRow, _nnzj);
    Eigen::Map<Eigen::VectorXi> cols(_jCol, _nnzj);

    mWrapped->getJacobianSparsityStructure(rows, cols, perflog);

    /*
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
    */
  }
  else
  {
    if (_new_x && _n > 0)
    {
      Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
      mWrapped->unflatten(flat, perflog);
    }
    Eigen::Map<Eigen::VectorXd> sparse(_values, _nnzj);
    mWrapped->getSparseJacobian(mWrapped->mWorld, sparse, perflog);

    /*
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
    */
  }

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
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
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog
        = mRecord->getPerfLog()->startRun("IPOptShotWrapper.finalize_solution");
  }
#endif

  Eigen::Map<const Eigen::VectorXd> flat(_x, _n);

  // TODO: we may not actually want to do this
  std::cout << "Recovering best discovered state from iter " << mBestIter
            << " with loss " << mBestFeasibleObjectiveValue << std::endl;
  mWrapped->unflatten(mBestFeasibleState, perflog);
  /*
  const std::shared_ptr<Problem>& problem = mSolver->getProblem();

  // Store optimal and optimum values
  problem->setOptimumValue(_obj_value);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(_n);
  for (int i = 0; i < _n; ++i)
    x[i] = _x[i];
  problem->setOptimalSolution(x);
  */

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
}

//==============================================================================
bool IPOptShotWrapper::intermediate_callback(
    Ipopt::AlgorithmMode mode,
    Ipopt::Index iter,
    Ipopt::Number obj_value,
    Ipopt::Number inf_pr,
    Ipopt::Number inf_du,
    Ipopt::Number mu,
    Ipopt::Number d_norm,
    Ipopt::Number regularization_size,
    Ipopt::Number alpha_du,
    Ipopt::Number alpha_pr,
    Ipopt::Index ls_trials,
    const Ipopt::IpoptData* ip_data,
    Ipopt::IpoptCalculatedQuantities* ip_cq)
{
  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun(
        "IPOptShotWrapper.intermediate_callback");
  }
#endif

  // Always record the iteration
  mRecord->registerIteration(
      iter,
      mWrapped->getRolloutCache(mWrapped->mWorld, perflog),
      obj_value,
      inf_pr);
  if (obj_value < mBestFeasibleObjectiveValue && inf_pr < 5e-4)
  {
    // std::cout << "Found new best feasible loss!" << std::endl;
    mBestIter = iter;
    // Found new best feasible objective
    mBestFeasibleObjectiveValue = obj_value;
    mWrapped->flatten(mBestFeasibleState, perflog);
  }

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
  return true;
}

} // namespace trajectory
} // namespace dart