#ifndef DART_TRAJECTORY_LOSS_FUNCTION_HPP_
#define DART_TRAJECTORY_LOSS_FUNCTION_HPP_

#include <memory>

#include <Eigen/Dense>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/tl_optional.hpp"

namespace dart {

using namespace performance;

namespace trajectory {

typedef std::function<s_t(const TrajectoryRollout* rollout)>
    TrajectoryLossFn;

typedef std::function<s_t(
    const TrajectoryRollout* rollout,
    /* OUT */ TrajectoryRollout* gradWrtRollout)>
    TrajectoryLossFnAndGrad;

class LossFn
{
public:
  LossFn();

  LossFn(TrajectoryLossFn loss);

  LossFn(TrajectoryLossFn loss, TrajectoryLossFnAndGrad lossAndGrad);

  virtual ~LossFn();

  virtual s_t getLoss(
      const TrajectoryRollout* rollout, PerformanceLog* perflog = nullptr);

  virtual s_t getLossAndGradient(
      const TrajectoryRollout* rollout,
      /* OUT */ TrajectoryRollout* gradWrtRollout,
      PerformanceLog* perflog = nullptr);

  /// If this LossFn is being used as a constraint, this gets the lower bound
  /// it's allowed to reach
  s_t getLowerBound() const;

  /// If this LossFn is being used as a constraint, this sets the lower bound
  /// it's allowed to reach
  void setLowerBound(s_t lowerBound);

  /// If this LossFn is being used as a constraint, this gets the upper bound
  /// it's allowed to reach
  s_t getUpperBound() const;

  /// If this LossFn is being used as a constraint, this sets the upper bound
  /// it's allowed to reach
  void setUpperBound(s_t upperBound);

protected:
  tl::optional<TrajectoryLossFn> mLoss;
  tl::optional<TrajectoryLossFnAndGrad> mLossAndGrad;
  // If this loss function is being used as a constraint, this is the lower
  // bound it's allowed to reach
  s_t mLowerBound;
  // If this loss function is being used as a constraint, this is the upper
  // bound it's allowed to reach
  s_t mUpperBound;
};

} // namespace trajectory
} // namespace dart

#endif