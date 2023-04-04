#ifndef UNSCENTED_KALMAN_FILTER_H
#define UNSCENTED_KALMAN_FILTER_H

#include <functional>

#include <Eigen/Dense>

namespace dart {
namespace realtime {
class UnscentedKalmanFilter
{
public:
  UnscentedKalmanFilter(
      std::function<
          Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
          stateTransitionFunction,
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
          measurementFunction,
      Eigen::VectorXd initialState,
      Eigen::MatrixXd initialCovariance,
      Eigen::MatrixXd processNoiseCovariance,
      Eigen::MatrixXd measurementNoiseCovariance,
      double alpha = 1e-3,
      double beta = 2,
      double kappa = 0);

  void predict(const Eigen::VectorXd& controlInput);
  void update(const Eigen::VectorXd& measurement);

  const Eigen::VectorXd& state() const;
  const Eigen::MatrixXd& stateCovariance() const;

private:
  std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
      mStateTransitionFunction;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> mMeasurementFunction;
  Eigen::VectorXd mState;
  Eigen::MatrixXd mStateCovariance;
  Eigen::MatrixXd mProcessNoiseCovariance;
  Eigen::MatrixXd mMeasurementNoiseCovariance;
  Eigen::VectorXd mWeightsMean;
  Eigen::VectorXd mWeightsCovariance;
  double mScalingFactorAlpha;
  double mPriorKnowledgeFactorBeta;
  double mSecondaryScalingFactorKappa;

  Eigen::MatrixXd generateSigmaPoints() const;
  void initWeights();
};
} // namespace realtime
} // namespace dart

#endif // UNSCENTED_KALMAN_FILTER_H