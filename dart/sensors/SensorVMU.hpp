#ifndef DART_SENSOR_VMU_HPP_
#define DART_SENSOR_VMU_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/sensors/Sensor.hpp"

namespace dart {
namespace sensors {

class SensorVMU : public Sensor
{
public:
  SensorVMU(
      std::string sensorName,
      std::string bodyName,
      Eigen::Isometry3s localTransform);

  virtual ~SensorVMU();

  /// This method returns the dimension of the sensor output
  virtual int outputDim() override;

  /// This method returns the dimension of the state of the sensor, if there is
  /// any. Most sensors will be "stateless" and so return a 0 here. If, however,
  /// you have a sensor who's location or orientation on a body is unknown, for
  /// example, then that could be the state of the sensor.
  virtual int stateDim() override;

  /// This method will return the predicted outputs of the sensor
  virtual Eigen::VectorXs observationFunction(
      std::shared_ptr<dynamics::Skeleton> skel,
      Eigen::VectorXs sensorState) override;

  /// This method will compute the total Jacobian of the observation function,
  /// with respect to `wrt`
  virtual Eigen::MatrixXs observationJacobianWrt(
      std::shared_ptr<dynamics::Skeleton> skel,
      Eigen::VectorXs sensorState,
      neural::WithRespectTo* wrt) override;

  /// This method will compute the total Jacobian of the observation function,
  /// with respect to the sensor state vector
  virtual Eigen::MatrixXs observationJacobianWrtSensorState(
      std::shared_ptr<dynamics::Skeleton> skel,
      Eigen::VectorXs sensorState) override;

  /// This returns the noise of the sensor. This can be a constant value, but is
  /// also allowed to vary based on the state of the skeleton
  virtual Eigen::MatrixXs noiseCovariance(
      std::shared_ptr<dynamics::Skeleton> skel,
      Eigen::VectorXs sensorState) override;

protected:
  std::string mSensorName;
  std::string mBodyName;
  Eigen::Isometry3s mLocalTransform;
};

} // namespace sensors
} // namespace dart

#endif