#ifndef DART_TRAJECTORY_OPTIMIZATION_RECORD_HPP_
#define DART_TRAJECTORY_OPTIMIZATION_RECORD_HPP_

#include <memory>
#include <string>
#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/trajectory/IPOptShotWrapper.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

using namespace Ipopt;

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

struct OptimizationStep
{
  int index;
  std::shared_ptr<TrajectoryRollout> rollout;
  s_t loss;
  s_t constraintViolation;

  OptimizationStep(
      int index,
      const TrajectoryRollout* rollout,
      s_t loss,
      s_t constraintViolation)
    : index(index),
      rollout(std::make_shared<TrajectoryRolloutReal>(rollout)),
      loss(loss),
      constraintViolation(constraintViolation)
  {
  }
};

class Solution
{
public:
  Solution();

  /// After optimization, register whether IPOPT thought it was a success
  void setSuccess(bool success);

  /// During optimization, register a single iteration of gradient descent
  void registerIteration(
      int index,
      const TrajectoryRollout* rollout,
      s_t loss,
      s_t constraintViolation);

  /// This only gets called if we're saving full debug info, but it stores every
  /// x that we receive during optimization
  void registerX(Eigen::VectorXs x);

  /// This only gets called if we're saving full debug info, but it stores every
  /// loss evaluation that we produce during optimization
  void registerLoss(s_t loss);

  /// This only gets called if we're saving full debug info, but it stores every
  /// gradient that we produce during optimization
  void registerGradient(Eigen::VectorXs grad);

  /// This only gets called if we're saving full debug info, but it stores every
  /// constraint value that we produce during optimization
  void registerConstraintValues(Eigen::VectorXs g);

  /// This only gets called if we're saving full debug info, but it stores every
  /// jacobian that we produce during optimization
  void registerSparseJac(Eigen::VectorXs jac);

  /// Returns the number of steps that were registered
  int getNumSteps();

  /// This returns the step record for this index
  const OptimizationStep& getStep(int index);

  /// This converts this optimization record into a JSON blob we can display on
  /// our web GUI
  std::string toJson(std::shared_ptr<simulation::World> world);

  /// This gets called by the optimizer, if we're recording performance per
  /// optimization
  void startPerfLog();

  /// This returns a reference to the PerformanceLog for this Optimization
  performance::PerformanceLog* getPerfLog();

  /// This gets the x's we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXs>& getXs();

  /// This gets the losses we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<s_t>& getLosses();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXs>& getGradients();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXs>& getConstraintValues();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXs>& getSparseJacobians();

  /// This registers all the pieces we need in order to be able to re-optimize
  /// this problem efficiently.
  void registerForReoptimization(
      SmartPtr<Ipopt::IpoptApplication> ipopt,
      SmartPtr<trajectory::IPOptShotWrapper> ipoptProblem);

  /// This will attempt to run another round of optimization.
  void reoptimize();

protected:
  bool mSuccess;
  std::vector<OptimizationStep> mSteps;
  performance::PerformanceLog* mPerfLog;
  std::vector<Eigen::VectorXs> mXs;
  std::vector<s_t> mLosses;
  std::vector<Eigen::VectorXs> mGradients;
  std::vector<Eigen::VectorXs> mConstraintValues;
  std::vector<Eigen::VectorXs> mSparseJacobians;
  // In order to re-optimize
  SmartPtr<Ipopt::IpoptApplication> mIpopt;
  SmartPtr<trajectory::IPOptShotWrapper> mIpoptProblem;
};

} // namespace trajectory
} // namespace dart

#endif