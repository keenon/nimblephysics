#include "dart/trajectory/IPOptShotWrapper.hpp"

#include <chrono>
#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/realtime/Millis.hpp"
#include "dart/trajectory/Solution.hpp"

#define LOG_PERFORMANCE_IPOPT

using namespace dart;
using namespace simulation;
using namespace performance;

using namespace Ipopt;

namespace dart {
namespace trajectory {

//==============================================================================
IPOptShotWrapper::IPOptShotWrapper(
    Problem* wrapped,
    std::shared_ptr<Solution> record,
    bool recoverBest,
    bool recordFullDebugInfo,
    bool printIterations,
    bool recordIterations)
  : mBestFeasibleObjectiveValue(std::numeric_limits<double>::infinity()),
    mWrapped(wrapped),
    mBestIter(-1),
    mRecord(record),
    mRecoverBest(recoverBest),
    mRecordFullDebugInfo(recordFullDebugInfo),
    mPrintIterations(printIterations),
    mRecordIterations(recordIterations),
    mLastTimestep(timeSinceEpochMillis()),
    mFCalls(0),
    mGradFCalls(0),
    mGCalls(0),
    mJacGCalls(0),
    mNewXs(0)
{
  if (mRecoverBest)
  {
    mBestFeasibleState
        = Eigen::VectorXd(mWrapped->getFlatProblemDim(mWrapped->mWorld));
  }
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
  n = mWrapped->getFlatProblemDim(mWrapped->mWorld);

  // Set the total number of constraints
  m = mWrapped->getConstraintDim();

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = mWrapped->getNumberNonZeroJacobian(mWrapped->mWorld);

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
  assert(
      static_cast<std::size_t>(n)
      == mWrapped->getFlatProblemDim(mWrapped->mWorld));
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
    Ipopt::Number* z_L,
    Ipopt::Number* z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number* lambda)
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
    Eigen::Map<Eigen::VectorXd> zU_vec(z_U, n);
    Eigen::Map<Eigen::VectorXd> zL_vec(z_L, n);
    zU_vec = mSaved_zU;
    zL_vec = mSaved_zL;
    /*
    std::cout << "Initializing lower/upper bounds for z is not supported yet. "
              << "Ignored here.\n";
              */
  }

  // If init_lambda is true, this method must provide an initial value for the
  // constraint multipliers, lambda.
  if (init_lambda)
  {
    Eigen::Map<Eigen::VectorXd> lambda_vec(lambda, m);
    lambda_vec = mSaved_lambda;
    /*
    std::cout << "Initializing lambda is not supported yet. "
              << "Ignored here.\n";
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
bool IPOptShotWrapper::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  if (!can_eval_f(_new_x))
  {
    return false;
  }

  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_f");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim(mWrapped->mWorld));
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(mWrapped->mWorld, flat, perflog);
  }
  _obj_value = mWrapped->getLoss(mWrapped->mWorld, perflog);

  if (mRecordFullDebugInfo)
  {
    if (_new_x)
    {
      std::cout << "  New X" << std::endl;
      Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
      mRecord->registerX(flat);
    }
    std::cout << "Loss eval " << mRecord->getLosses().size() << std::endl;
    mRecord->registerLoss(_obj_value);
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
bool IPOptShotWrapper::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  if (!can_eval_grad_f(_new_x))
  {
    return false;
  }

  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_grad_f");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim(mWrapped->mWorld));
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(mWrapped->mWorld, flat, perflog);
  }
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);
  mWrapped->backpropGradient(mWrapped->mWorld, grad, perflog);

  if (mRecordFullDebugInfo)
  {
    if (_new_x)
    {
      std::cout << "  New X" << std::endl;
      Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
      mRecord->registerX(flat);
    }
    std::cout << "Gradient eval " << mRecord->getGradients().size()
              << std::endl;
    mRecord->registerGradient(grad);
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
bool IPOptShotWrapper::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  if (!can_eval_g(_new_x))
  {
    return false;
  }

  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun("IPOptShotWrapper.eval_g");
  }
#endif

  assert(_n == mWrapped->getFlatProblemDim(mWrapped->mWorld));
  assert(_m == mWrapped->getConstraintDim());
  if (_new_x && _n > 0)
  {
    Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
    mWrapped->unflatten(mWrapped->mWorld, flat, perflog);
  }
  Eigen::Map<Eigen::VectorXd> constraints(_g, _m);
  mWrapped->computeConstraints(mWrapped->mWorld, constraints, perflog);

  if (mRecordFullDebugInfo)
  {
    if (_new_x)
    {
      std::cout << "  New X" << std::endl;
      Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
      mRecord->registerX(flat);
    }
    std::cout << "Constraint eval " << mRecord->getConstraintValues().size()
              << std::endl;
    mRecord->registerConstraintValues(constraints);
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
  if (!can_eval_jac_g(_new_x))
  {
    return false;
  }

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
    assert(_n == mWrapped->getFlatProblemDim(mWrapped->mWorld));
    assert(_m == mWrapped->getConstraintDim());
    assert(_nnzj == mWrapped->getNumberNonZeroJacobian(mWrapped->mWorld));

    Eigen::Map<Eigen::VectorXi> rows(_iRow, _nnzj);
    Eigen::Map<Eigen::VectorXi> cols(_jCol, _nnzj);

    mWrapped->getJacobianSparsityStructure(
        mWrapped->mWorld, rows, cols, perflog);

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
      mWrapped->unflatten(mWrapped->mWorld, flat, perflog);
    }
    Eigen::Map<Eigen::VectorXd> sparse(_values, _nnzj);
    mWrapped->getSparseJacobian(mWrapped->mWorld, sparse, perflog);

    if (mRecordFullDebugInfo)
    {
      if (_new_x)
      {
        std::cout << "  New X" << std::endl;
        Eigen::Map<const Eigen::VectorXd> flat(_x, _n);
        mRecord->registerX(flat);
      }
      std::cout << "Jac eval " << mRecord->getSparseJacobians().size()
                << std::endl;
      mRecord->registerSparseJac(sparse);
    }

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
    const Ipopt::Number* _z_L,
    const Ipopt::Number* _z_U,
    Ipopt::Index _m,
    const Ipopt::Number* /*_g*/,
    const Ipopt::Number* _lambda,
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

  Eigen::Map<const Eigen::VectorXd> zU_vec(_z_U, _n);
  Eigen::Map<const Eigen::VectorXd> zL_vec(_z_L, _n);
  Eigen::Map<const Eigen::VectorXd> lambda_vec(_lambda, _m);
  mSaved_zU = zU_vec;
  mSaved_zL = zL_vec;
  mSaved_lambda = lambda_vec;

  if (mRecoverBest && mBestIter != -1)
  {
    // std::cout << "Recovering best discovered state from iter " << mBestIter
    // << " with loss " << mBestFeasibleObjectiveValue << std::endl;
    mWrapped->unflatten(mWrapped->mWorld, mBestFeasibleState, perflog);
  }
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
  // Reset call counts
  reset_iteration();

  PerformanceLog* perflog = nullptr;
#ifdef LOG_PERFORMANCE_IPOPT
  if (mRecord->getPerfLog() != nullptr)
  {
    perflog = mRecord->getPerfLog()->startRun(
        "IPOptShotWrapper.intermediate_callback");
  }
#endif

  // Always record the iteration
  if (mRecordIterations)
  {
    mRecord->registerIteration(
        iter,
        mWrapped->getRolloutCache(mWrapped->mWorld, perflog),
        obj_value,
        inf_pr);
  }

  if (mPrintIterations)
  {
    uint64_t now = timeSinceEpochMillis();

    std::cout << "(" << (now - mLastTimestep) << "ms) Loss:  " << obj_value
              << "  Viol:  " << inf_pr << std::endl;

    mLastTimestep = now;
  }

  if (mRecoverBest && obj_value < mBestFeasibleObjectiveValue && inf_pr < 5e-4)
  {
    // std::cout << "Found new best feasible loss!" << std::endl;
    mBestIter = iter;
    // Found new best feasible objective
    mBestFeasibleObjectiveValue = obj_value;
    mWrapped->flatten(mWrapped->mWorld, mBestFeasibleState, perflog);
  }

#ifdef LOG_PERFORMANCE_IPOPT
  if (perflog != nullptr)
  {
    perflog->end();
  }
#endif
  return true;
}

/// This gets called when we're about to repoptimize, to let us reset values.
void IPOptShotWrapper::prep_for_reoptimize()
{
  // Reset the objective value, so we recover properly
  mBestFeasibleObjectiveValue = std::numeric_limits<double>::infinity();
  mBestIter = -1;
}

/// This records a single call of eval_f(). If this returns false, then we
/// need to terminate this call to eval_f().
bool IPOptShotWrapper::can_eval_f(bool new_x)
{
  if (new_x)
    mNewXs++;
  mFCalls++;
  return can_continue();
}

/// This records a single call of eval_grad_f(). If this returns false, then
/// we need to terminate this call to eval_grad_f().
bool IPOptShotWrapper::can_eval_grad_f(bool new_x)
{
  if (new_x)
    mNewXs++;
  mGradFCalls++;
  return can_continue();
}

/// This records a single call of eval_g(). If this returns false, then
/// we need to terminate this call to eval_g().
bool IPOptShotWrapper::can_eval_g(bool new_x)
{
  if (new_x)
    mNewXs++;
  mGCalls++;
  return can_continue();
}

/// This records a single call of eval_jac_g(). If this returns false, then
/// we need to terminate this call to eval_jac_g().
bool IPOptShotWrapper::can_eval_jac_g(bool new_x)
{
  if (new_x)
    mNewXs++;
  mJacGCalls++;
  return can_continue();
}

/// This is a central method that evaluates if we can continue the
/// optimization
bool IPOptShotWrapper::can_continue()
{
  return true;
}

/// This resets the stored data from this iteration
void IPOptShotWrapper::reset_iteration()
{
  mFCalls = 0;
  mGradFCalls = 0;
  mGCalls = 0;
  mJacGCalls = 0;
  mNewXs = 0;
}

} // namespace trajectory
} // namespace dart