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
      Eigen::MatrixXd measurementNoiseCovariance);

  void predict(const Eigen::VectorXd& controlInput);
  void update(const Eigen::VectorXd& measurement);

  const Eigen::VectorXd& state() const;
  const Eigen::MatrixXd& stateCovariance() const;

private:
  std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
      fx_;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hx_;
  Eigen::VectorXd state_;
  Eigen::MatrixXd stateCovariance_;
  Eigen::MatrixXd processNoiseCovariance_;
  Eigen::MatrixXd measurementNoiseCovariance_;
  Eigen::VectorXd weightsMean_;
  Eigen::VectorXd weightsCovariance_;

  Eigen::MatrixXd generateSigmaPoints() const;
  void initWeights();
};
} // namespace realtime
} // namespace dart

#endif // UNSCENTED_KALMAN_FILTER_H