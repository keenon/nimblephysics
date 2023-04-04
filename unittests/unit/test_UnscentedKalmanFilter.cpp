#include <gtest/gtest.h>

#include "dart/realtime/UnscentedKalmanFilter.hpp" // Make sure to include the header file for the UnscentedKalmanFilter class

using namespace dart;
using namespace realtime;

// State transition function for the constant velocity model
Eigen::VectorXd constantVelocityModel(
    const Eigen::VectorXd& state, const Eigen::VectorXd& controlInput)
{
  Eigen::VectorXd newState(2);
  newState(0) = state(0) + state(1) * controlInput(0);
  newState(1) = state(1);
  return newState;
}

// Measurement function for a linear model
Eigen::VectorXd linearMeasurementModel(const Eigen::VectorXd& state)
{
  Eigen::VectorXd measurement(1);
  measurement(0) = state(0);
  return measurement;
}

TEST(UnscentedKalmanFilterTest, ConstantVelocityModel)
{
  // Initialize the UnscentedKalmanFilter
  Eigen::VectorXd initialState(2);
  initialState << 0.0, 1.0;
  Eigen::MatrixXd initialCovariance = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd processNoiseCovariance
      = 0.1 * Eigen::MatrixXd::Identity(2, 2);

  Eigen::MatrixXd measurementNoiseCovariance(1, 1);
  measurementNoiseCovariance << 0.1;

  UnscentedKalmanFilter ukf(
      constantVelocityModel,
      linearMeasurementModel,
      initialState,
      initialCovariance,
      processNoiseCovariance,
      measurementNoiseCovariance);

  // Apply a series of predict and update steps
  Eigen::VectorXd controlInput(2);
  controlInput << 1.0, 0.0;

  for (int i = 0; i < 10; ++i)
  {
    ukf.predict(controlInput);

    Eigen::VectorXd measurement(1);
    measurement << i + 1.0 + 0.1 * (rand() % 1000) / 1000.0;
    ukf.update(measurement);
  }

  // Check the final state estimate
  Eigen::VectorXd expectedState(2);
  expectedState << 10.0, 1.0;
  std::cout << "State: " << ukf.state().transpose() << std::endl;
  EXPECT_TRUE(ukf.state().isApprox(expectedState, 1e-1));

  // Check the final state covariance
  Eigen::MatrixXd expectedCovariance = Eigen::MatrixXd::Zero(2, 2);
  expectedCovariance(0, 0) = 0.1;
  expectedCovariance(1, 1) = 0.1;
  std::cout << "Covariance: " << std::endl
            << ukf.stateCovariance() << std::endl;
  EXPECT_TRUE(ukf.stateCovariance().isApprox(expectedCovariance, 1e-1));
}