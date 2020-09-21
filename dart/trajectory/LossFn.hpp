#ifndef DART_TRAJECTORY_LOSS_FUNCTION_HPP_
#define DART_TRAJECTORY_LOSS_FUNCTION_HPP_

#include <memory>

#include <Eigen/Dense>

#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/tl_optional.hpp"

namespace dart {

namespace trajectory {

typedef std::function<double(const TrajectoryRollout* rollout)>
    TrajectoryLossFn;

typedef std::function<double(
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

  virtual double getLoss(const TrajectoryRollout* rollout);

  virtual double getLossAndGradient(
      const TrajectoryRollout* rollout,
      /* OUT */ TrajectoryRollout* gradWrtRollout);

  /// If this LossFn is being used as a constraint, this gets the lower bound
  /// it's allowed to reach
  double getLowerBound() const;

  /// If this LossFn is being used as a constraint, this sets the lower bound
  /// it's allowed to reach
  void setLowerBound(double lowerBound);

  /// If this LossFn is being used as a constraint, this gets the upper bound
  /// it's allowed to reach
  double getUpperBound() const;

  /// If this LossFn is being used as a constraint, this sets the upper bound
  /// it's allowed to reach
  void setUpperBound(double upperBound);

protected:
  tl::optional<TrajectoryLossFn> mLoss;
  tl::optional<TrajectoryLossFnAndGrad> mLossAndGrad;
  // If this loss function is being used as a constraint, this is the lower
  // bound it's allowed to reach
  double mLowerBound;
  // If this loss function is being used as a constraint, this is the upper
  // bound it's allowed to reach
  double mUpperBound;
};

} // namespace trajectory
} // namespace dart

#endif