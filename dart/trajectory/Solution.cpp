#include "dart/trajectory/Solution.hpp"

#include <sstream>
#include <unordered_map>
#include <vector>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/server/RawJsonUtils.hpp"
#include "dart/simulation/World.hpp"

using namespace Ipopt;

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
Solution::Solution() : mSuccess(false), mPerfLog(nullptr)
{
}

//==============================================================================
/// This returns a reference to the PerformanceLog for this Optimization
void Solution::startPerfLog()
{
  mPerfLog = performance::PerformanceLog::startRoot("IPOptOptimizer.optimize");
}

//==============================================================================
/// This returns a reference to the PerformanceLog for this Optimization
performance::PerformanceLog* Solution::getPerfLog()
{
  return mPerfLog;
}

//==============================================================================
/// This gets the x's we've recorded. This will be empty unless we've
/// called optimizer.setRecordFullDebugInfo(true)
std::vector<Eigen::VectorXd>& Solution::getXs()
{
  return mXs;
}

//==============================================================================
/// This gets the losses we've recorded. This will be empty unless we've
/// called optimizer.setRecordFullDebugInfo(true)
std::vector<double>& Solution::getLosses()
{
  return mLosses;
}

//==============================================================================
/// This gets the gradients we've recorded. This will be empty unless we've
/// called optimizer.setRecordFullDebugInfo(true)
std::vector<Eigen::VectorXd>& Solution::getGradients()
{
  return mGradients;
}

//==============================================================================
/// This gets the gradients we've recorded. This will be empty unless we've
/// called optimizer.setRecordFullDebugInfo(true)
std::vector<Eigen::VectorXd>& Solution::getConstraintValues()
{
  return mConstraintValues;
}

//==============================================================================
/// This gets the gradients we've recorded. This will be empty unless we've
/// called optimizer.setRecordFullDebugInfo(true)
std::vector<Eigen::VectorXd>& Solution::getSparseJacobians()
{
  return mSparseJacobians;
}

//==============================================================================
/// This registers all the pieces we need in order to be able to re-optimize
/// this problem efficiently.
void Solution::registerForReoptimization(
    SmartPtr<Ipopt::IpoptApplication> ipopt,
    SmartPtr<IPOptShotWrapper> ipoptProblem)
{
  mIpopt = ipopt;
  mIpoptProblem = ipoptProblem;
}

//==============================================================================
/// This will attempt to run another round of optimization.
void Solution::reoptimize()
{
  std::string oldWarmStart;
  mIpoptProblem->prep_for_reoptimize();
  // mIpopt->Options()->GetStringValue("warm_start_init_point", oldWarmStart,
  // ""); mIpopt->Options()->SetStringValue("warm_start_init_point", "yes");
  ApplicationReturnStatus status = mIpopt->ReOptimizeTNLP(mIpoptProblem);
  // mIpopt->Options()->SetStringValue("warm_start_init_point", oldWarmStart);

  if (status == Solve_Succeeded)
  {
    // Retrieve some statistics about the solve
    Index iter_count = mIpopt->Statistics()->IterationCount();
    Number final_obj = mIpopt->Statistics()->FinalObjective();
  }

  this->setSuccess(status == Ipopt::Solve_Succeeded);
  this->registerForReoptimization(mIpopt, mIpoptProblem);
}

//==============================================================================
void Solution::setSuccess(bool success)
{
  if (mPerfLog != nullptr)
  {
    mPerfLog->end();
  }
  mSuccess = success;
}

//==============================================================================
void Solution::registerIteration(
    int index,
    const TrajectoryRollout* rollout,
    double loss,
    double constraintViolation)
{
  mSteps.emplace_back(index, rollout, loss, constraintViolation);
}

//==============================================================================
/// This only gets called if we're saving full debug info, but it stores every
/// x that we receive during optimization
void Solution::registerX(Eigen::VectorXd x)
{
  mXs.push_back(x);
}

//==============================================================================
/// This only gets called if we're saving full debug info, but it stores every
/// loss evaluation that we produce during optimization
void Solution::registerLoss(double loss)
{
  mLosses.push_back(loss);
}

//==============================================================================
/// This only gets called if we're saving full debug info, but it stores every
/// gradient that we produce during optimization
void Solution::registerGradient(Eigen::VectorXd grad)
{
  mGradients.push_back(grad);
}

//==============================================================================
/// This only gets called if we're saving full debug info, but it stores every
/// constraint value that we produce during optimization
void Solution::registerConstraintValues(Eigen::VectorXd g)
{
  mConstraintValues.push_back(g);
}

//==============================================================================
/// This only gets called if we're saving full debug info, but it stores every
/// jacobian that we produce during optimization
void Solution::registerSparseJac(Eigen::VectorXd jac)
{
  mSparseJacobians.push_back(jac);
}

//==============================================================================
/// Returns the number of steps that were registered
int Solution::getNumSteps()
{
  return mSteps.size();
}

//==============================================================================
/// This returns the step record for this index
const OptimizationStep& Solution::getStep(int index)
{
  return mSteps.at(index);
}

//==============================================================================
/// This converts this optimization record into a JSON blob we can display on
/// our web GUI
std::string Solution::toJson(std::shared_ptr<simulation::World> world)
{
  std::stringstream json;

  json << "{";
  json << "\"world\": ";
  json << world->toJson();

  std::vector<dynamics::BodyNode*> bodies = world->getAllBodyNodes();
  Eigen::VectorXd originalWorldPos = world->getPositions();

  json << ",\"record\": [";
  for (int i = 0; i < getNumSteps(); i++)
  {
    json << "{";

    const trajectory::OptimizationStep& step = getStep(i);
    json << "\"index\": " << step.index << ",";
    json << "\"loss\": " << step.loss << ",";
    json << "\"constraintViolation\": " << step.constraintViolation << ",";
    int timesteps = step.rollout->getPoses("identity").cols();
    json << "\"timesteps\": " << timesteps << ",";
    json << "\"trajectory\": " << step.rollout->toJson(world);

    json << "}";
    if (i < getNumSteps() - 1)
      json << ",";
  }
  json << "]";

  world->setPositions(originalWorldPos);

  json << "}";

  return json.str();
}

} // namespace trajectory
} // namespace dart