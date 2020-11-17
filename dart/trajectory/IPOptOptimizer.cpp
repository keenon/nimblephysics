#include "dart/trajectory/IPOptOptimizer.hpp"

#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/trajectory/IPOptShotWrapper.hpp"

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
    mRecordPerfLog(false),
    mRecoverBest(true),
    mRecordFullDebugInfo(false),
    mSuppressOutput(false),
    mSilenceOutput(false),
    mDisableLinesearch(false),
    mRecordIterations(true)
{
}

//==============================================================================
std::shared_ptr<Solution> IPOptOptimizer::optimize(
    Problem* shot, std::shared_ptr<Solution> reuseRecord)
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

  /*
  app->Options()->SetStringValue(
      "scaling_method", "none"); // none, gradient-based
  */

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
    app->Options()->SetIntegerValue("print_frequency_iter", mPrintFrequency);
  }
  else
  {
    app->Options()->SetIntegerValue(
        "print_frequency_iter", std::numeric_limits<int>::infinity());
  }
  if (mSuppressOutput || mSilenceOutput)
  {
    app->Options()->SetIntegerValue("print_level", 0);
  }
  if (mDisableLinesearch)
  {
    app->Options()->SetIntegerValue("max_soc", 0);
    app->Options()->SetStringValue("accept_every_trial_step", "yes");
  }
  app->Options()->SetIntegerValue("watchdog_shortened_iter_trigger", 0);

  std::shared_ptr<Solution> record
      = reuseRecord ? reuseRecord : std::make_shared<Solution>();
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
  IPOptShotWrapper* problem = new IPOptShotWrapper(
      shot,
      record,
      mRecoverBest,
      mRecordFullDebugInfo,
      mSuppressOutput && !mSilenceOutput,
      mRecordIterations);
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
  record->registerForReoptimization(app, problemPtr);

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

//==============================================================================
void IPOptOptimizer::setRecordPerformanceLog(bool recordPerfLog)
{
  mRecordPerfLog = recordPerfLog;
}

//==============================================================================
void IPOptOptimizer::setRecoverBest(bool recoverBest)
{
  mRecoverBest = recoverBest;
}

//==============================================================================
void IPOptOptimizer::setRecordFullDebugInfo(bool recordFullDebugInfo)
{
  mRecordFullDebugInfo = recordFullDebugInfo;
}

//==============================================================================
void IPOptOptimizer::setSuppressOutput(bool suppressOutput)
{
  mSuppressOutput = suppressOutput;
}

//==============================================================================
void IPOptOptimizer::setSilenceOutput(bool silenceOutput)
{
  mSilenceOutput = silenceOutput;
}

//==============================================================================
void IPOptOptimizer::setDisableLinesearch(bool disableLinesearch)
{
  mDisableLinesearch = disableLinesearch;
}

//==============================================================================
void IPOptOptimizer::setRecordIterations(bool recordIterations)
{
  mRecordIterations = recordIterations;
}

} // namespace trajectory
} // namespace dart