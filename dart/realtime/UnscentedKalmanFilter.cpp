#include "dart/realtime/UnscentedKalmanFilter.hpp"

#include <functional>
#include <iostream>

#include <Eigen/Dense>

namespace dart {
namespace realtime {

UnscentedKalmanFilter::UnscentedKalmanFilter(
    std::function<
        Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
        stateTransitionFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> measurementFunction,
    Eigen::VectorXd initialState,
    Eigen::MatrixXd initialCovariance,
    Eigen::MatrixXd processNoiseCovariance,
    Eigen::MatrixXd measurementNoiseCovariance)
  : fx_(stateTransitionFunction),
    hx_(measurementFunction),
    state_(initialState),
    stateCovariance_(initialCovariance),
    processNoiseCovariance_(processNoiseCovariance),
    measurementNoiseCovariance_(measurementNoiseCovariance)
{
  initWeights();
}

void UnscentedKalmanFilter::predict(const Eigen::VectorXd& controlInput)
{
  Eigen::MatrixXd sigmaPoints = generateSigmaPoints();
  Eigen::MatrixXd predictedSigmaPoints(state_.rows(), sigmaPoints.cols());

  for (int i = 0; i < sigmaPoints.cols(); ++i)
  {
    predictedSigmaPoints.col(i) = fx_(sigmaPoints.col(i), controlInput);
  }

  state_ = predictedSigmaPoints * weightsMean_;
  stateCovariance_ = (predictedSigmaPoints.colwise() - state_)
                         * weightsCovariance_.asDiagonal()
                         * (predictedSigmaPoints.colwise() - state_).transpose()
                     + processNoiseCovariance_;
}

void UnscentedKalmanFilter::update(const Eigen::VectorXd& measurement)
{
  Eigen::MatrixXd sigmaPoints = generateSigmaPoints();
  Eigen::MatrixXd predictedMeasurements(state_.rows(), sigmaPoints.cols());

  for (int i = 0; i < sigmaPoints.cols(); ++i)
  {
    predictedMeasurements.col(i) = hx_(sigmaPoints.col(i));
  }

  Eigen::VectorXd measurementPrediction = predictedMeasurements * weightsMean_;
  Eigen::MatrixXd innovationCovariance
      = (predictedMeasurements.colwise() - measurementPrediction)
            * weightsCovariance_.asDiagonal()
            * (predictedMeasurements.colwise() - measurementPrediction)
                  .transpose()
        + measurementNoiseCovariance_;
  Eigen::MatrixXd crossCovariance
      = (sigmaPoints.colwise() - state_) * weightsCovariance_.asDiagonal()
        * (predictedMeasurements.colwise() - measurementPrediction).transpose();
  Eigen::MatrixXd kalmanGain = crossCovariance * innovationCovariance.inverse();

  state_ += kalmanGain * (measurement - measurementPrediction);
  stateCovariance_
      -= kalmanGain * innovationCovariance * kalmanGain.transpose();
}

const Eigen::VectorXd& UnscentedKalmanFilter::state() const
{
  return state_;
}

const Eigen::MatrixXd& UnscentedKalmanFilter::stateCovariance() const
{
  return stateCovariance_;
}

Eigen::MatrixXd UnscentedKalmanFilter::generateSigmaPoints() const
{
  int stateSize = state_.size();
  double lambda = 3.0 - stateSize;
  Eigen::MatrixXd sigmaPoints(stateSize, 2 * stateSize + 1);

  Eigen::LLT<Eigen::MatrixXd> cholesky(stateCovariance_);
  Eigen::MatrixXd lowerTriangle = cholesky.matrixL();

  sigmaPoints.col(0) = state_;
  for (int i = 0; i < stateSize; ++i)
  {
    sigmaPoints.col(i + 1)
        = state_ + sqrt(stateSize + lambda) * lowerTriangle.col(i);
    sigmaPoints.col(i + 1 + stateSize)
        = state_ - sqrt(stateSize + lambda) * lowerTriangle.col(i);
  }

  return sigmaPoints;
}

void UnscentedKalmanFilter::initWeights()
{
  int stateSize = state_.size();
  double lambda = 3.0 - stateSize;

  weightsMean_ = Eigen::VectorXd(2 * stateSize + 1);
  weightsCovariance_ = Eigen::VectorXd(2 * stateSize + 1);

  weightsMean_(0) = lambda / (stateSize + lambda);
  weightsCovariance_(0) = lambda / (stateSize + lambda) + (1 - 1e-3 + 1e-3);

  for (int i = 1; i < 2 * stateSize + 1; ++i)
  {
    weightsMean_(i) = 1.0 / (2 * (stateSize + lambda));
    weightsCovariance_(i) = 1.0 / (2 * (stateSize + lambda));
  }
}

} // namespace realtime
} // namespace dart