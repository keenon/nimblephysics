#ifndef DART_TRAJECTORY_LOSS_FUNCTION_HPP_
#define DART_TRAJECTORY_LOSS_FUNCTION_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace trajectory {

typedef std::function<double(
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces)>
    TrajectoryLossFn;

typedef std::function<double(
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces)>
    TrajectoryLossFnAndGrad;

class LossFn
{
public:
  LossFn();

  LossFn(TrajectoryLossFn loss);

  LossFn(TrajectoryLossFn loss, TrajectoryLossFnAndGrad lossAndGrad);

  virtual ~LossFn();

  virtual double getLoss(
      const Eigen::Ref<const Eigen::MatrixXd>& poses,
      const Eigen::Ref<const Eigen::MatrixXd>& vels,
      const Eigen::Ref<const Eigen::MatrixXd>& forces);

  virtual double getLossAndGradient(
      const Eigen::Ref<const Eigen::MatrixXd>& poses,
      const Eigen::Ref<const Eigen::MatrixXd>& vels,
      const Eigen::Ref<const Eigen::MatrixXd>& forces,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces);

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
  std::optional<TrajectoryLossFn> mLoss;
  std::optional<TrajectoryLossFnAndGrad> mLossAndGrad;
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