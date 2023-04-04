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
    Eigen::MatrixXd measurementNoiseCovariance,
    double scalingFactorAlpha,
    double priorKnowledgeFactorBeta,
    double secondaryScalingFactorKappa)
  : mStateTransitionFunction(stateTransitionFunction),
    mMeasurementFunction(measurementFunction),
    mState(initialState),
    mStateCovariance(initialCovariance),
    mProcessNoiseCovariance(processNoiseCovariance),
    mMeasurementNoiseCovariance(measurementNoiseCovariance),
    mScalingFactorAlpha(scalingFactorAlpha),
    mPriorKnowledgeFactorBeta(priorKnowledgeFactorBeta),
    mSecondaryScalingFactorKappa(secondaryScalingFactorKappa)
{
  // Check dimensions
  int stateSize = mState.size();
  assert(
      initialCovariance.rows() == stateSize
      && initialCovariance.cols() == stateSize);
  assert(
      processNoiseCovariance.rows() == stateSize
      && processNoiseCovariance.cols() == stateSize);
  assert(
      measurementNoiseCovariance.rows() == measurementNoiseCovariance.cols());

  initWeights();
}

void UnscentedKalmanFilter::predict(const Eigen::VectorXd& controlInput)
{
  Eigen::MatrixXd sigmaPoints = generateSigmaPoints();
  assert(!sigmaPoints.hasNaN());
  Eigen::MatrixXd predictedSigmaPoints(mState.rows(), sigmaPoints.cols());
  assert(!predictedSigmaPoints.hasNaN());

  for (int i = 0; i < sigmaPoints.cols(); ++i)
  {
    predictedSigmaPoints.col(i)
        = mStateTransitionFunction(sigmaPoints.col(i), controlInput);
    assert(!predictedSigmaPoints.col(i).hasNaN());
  }

  mState = predictedSigmaPoints * mWeightsMean;
  assert(!mState.hasNaN());
  mStateCovariance = (predictedSigmaPoints.colwise() - mState)
                         * mWeightsCovariance.asDiagonal()
                         * (predictedSigmaPoints.colwise() - mState).transpose()
                     + mProcessNoiseCovariance;
  assert(!mStateCovariance.hasNaN());
}

void UnscentedKalmanFilter::update(const Eigen::VectorXd& measurement)
{
  Eigen::MatrixXd sigmaPoints = generateSigmaPoints();
  assert(!sigmaPoints.hasNaN());
  Eigen::MatrixXd predictedMeasurements(
      mMeasurementNoiseCovariance.rows(), sigmaPoints.cols());
  assert(!predictedMeasurements.hasNaN());

  for (int i = 0; i < sigmaPoints.cols(); ++i)
  {
    predictedMeasurements.col(i) = mMeasurementFunction(sigmaPoints.col(i));
    assert(!predictedMeasurements.col(i).hasNaN());
  }

  Eigen::VectorXd measurementPrediction = predictedMeasurements * mWeightsMean;
  Eigen::MatrixXd innovationCovariance
      = (predictedMeasurements.colwise() - measurementPrediction)
            * mWeightsCovariance.asDiagonal()
            * (predictedMeasurements.colwise() - measurementPrediction)
                  .transpose()
        + mMeasurementNoiseCovariance;
  Eigen::MatrixXd crossCovariance
      = (sigmaPoints.colwise() - mState) * mWeightsCovariance.asDiagonal()
        * (predictedMeasurements.colwise() - measurementPrediction).transpose();
  Eigen::MatrixXd kalmanGain = crossCovariance * innovationCovariance.inverse();

  mState += kalmanGain * (measurement - measurementPrediction);
  mStateCovariance
      -= kalmanGain * innovationCovariance * kalmanGain.transpose();
}

const Eigen::VectorXd& UnscentedKalmanFilter::state() const
{
  return mState;
}

const Eigen::MatrixXd& UnscentedKalmanFilter::stateCovariance() const
{
  return mStateCovariance;
}

Eigen::MatrixXd UnscentedKalmanFilter::generateSigmaPoints() const
{
  int stateSize = mState.size();
  double lambda = 3.0 - stateSize;
  Eigen::MatrixXd sigmaPoints(stateSize, 2 * stateSize + 1);

  Eigen::LLT<Eigen::MatrixXd> cholesky(mStateCovariance);
  Eigen::MatrixXd lowerTriangle = cholesky.matrixL();

  sigmaPoints.col(0) = mState;
  for (int i = 0; i < stateSize; ++i)
  {
    sigmaPoints.col(i + 1)
        = mState + sqrt(stateSize + lambda) * lowerTriangle.col(i);
    sigmaPoints.col(i + 1 + stateSize)
        = mState - sqrt(stateSize + lambda) * lowerTriangle.col(i);
  }

  return sigmaPoints;
}

void UnscentedKalmanFilter::initWeights()
{
  int stateSize = mState.size();
  double lambda = mScalingFactorAlpha * mScalingFactorAlpha
                      * (stateSize + mSecondaryScalingFactorKappa)
                  - stateSize;

  mWeightsMean = Eigen::VectorXd::Zero(2 * stateSize + 1);
  mWeightsCovariance = Eigen::VectorXd::Zero(2 * stateSize + 1);

  mWeightsMean(0) = lambda / (stateSize + lambda);
  mWeightsCovariance(0) = lambda / (stateSize + lambda)
                          + (1 - mScalingFactorAlpha * mScalingFactorAlpha
                             + mPriorKnowledgeFactorBeta);

  for (int i = 1; i < 2 * stateSize + 1; ++i)
  {
    mWeightsMean(i) = 1.0 / (2 * (stateSize + lambda));
    mWeightsCovariance(i) = 1.0 / (2 * (stateSize + lambda));
  }
}

} // namespace realtime
} // namespace dart