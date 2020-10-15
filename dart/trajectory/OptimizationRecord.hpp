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

protected:
  bool mSuccess;
  std::vector<OptimizationStep> mSteps;
  performance::PerformanceLog* mPerfLog;
};

} // namespace trajectory
} // namespace dart

#endif