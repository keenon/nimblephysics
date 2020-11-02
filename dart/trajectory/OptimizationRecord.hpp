#ifndef DART_TRAJECTORY_OPTIMIZATION_RECORD_HPP_
#define DART_TRAJECTORY_OPTIMIZATION_RECORD_HPP_

#include <memory>
#include <string>
#include <vector>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

struct OptimizationStep
{
  int index;
  std::shared_ptr<TrajectoryRollout> rollout;
  double loss;
  double constraintViolation;

  OptimizationStep(
      int index,
      const TrajectoryRollout* rollout,
      double loss,
      double constraintViolation)
    : index(index),
      rollout(std::make_shared<TrajectoryRolloutReal>(rollout)),
      loss(loss),
      constraintViolation(constraintViolation)
  {
  }
};

class OptimizationRecord
{
public:
  OptimizationRecord();

  /// After optimization, register whether IPOPT thought it was a success
  void setSuccess(bool success);

  /// During optimization, register a single iteration of gradient descent
  void registerIteration(
      int index,
      const TrajectoryRollout* rollout,
      double loss,
      double constraintViolation);

  /// This only gets called if we're saving full debug info, but it stores every
  /// x that we receive during optimization
  void registerX(Eigen::VectorXd x);

  /// This only gets called if we're saving full debug info, but it stores every
  /// loss evaluation that we produce during optimization
  void registerLoss(double loss);

  /// This only gets called if we're saving full debug info, but it stores every
  /// gradient that we produce during optimization
  void registerGradient(Eigen::VectorXd grad);

  /// This only gets called if we're saving full debug info, but it stores every
  /// constraint value that we produce during optimization
  void registerConstraintValues(Eigen::VectorXd g);

  /// This only gets called if we're saving full debug info, but it stores every
  /// jacobian that we produce during optimization
  void registerSparseJac(Eigen::VectorXd jac);

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
  std::vector<Eigen::VectorXd>& getXs();

  /// This gets the losses we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<double>& getLosses();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXd>& getGradients();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXd>& getConstraintValues();

  /// This gets the gradients we've recorded. This will be empty unless we've
  /// called optimizer.setRecordFullDebugInfo(true)
  std::vector<Eigen::VectorXd>& getSparseJacobians();

protected:
  bool mSuccess;
  std::vector<OptimizationStep> mSteps;
  performance::PerformanceLog* mPerfLog;
  std::vector<Eigen::VectorXd> mXs;
  std::vector<double> mLosses;
  std::vector<Eigen::VectorXd> mGradients;
  std::vector<Eigen::VectorXd> mConstraintValues;
  std::vector<Eigen::VectorXd> mSparseJacobians;
};

} // namespace trajectory
} // namespace dart

#endif