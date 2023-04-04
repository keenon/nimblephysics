#include "dart/sensors/SensorVMU.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace sensors {

SensorVMU::SensorVMU(
    std::string sensorName,
    std::string bodyName,
    Eigen::Isometry3s localTransform)
  : Sensor(sensorName), mBodyName(bodyName), mLocalTransform(localTransform)
{
}

SensorVMU::~SensorVMU()
{
}

/// This method returns the dimension of the sensor output
int SensorVMU::outputDim()
{
  return 9;
}

/// This method returns the dimension of the state of the sensor, if there is
/// any. Most sensors will be "stateless" and so return a 0 here. If, however,
/// you have a sensor who's location or orientation on a body is unknown, for
/// example, then that could be the state of the sensor.
int SensorVMU::stateDim()
{
  return 0;
}

/// This method will return the predicted outputs of the sensor
Eigen::VectorXs SensorVMU::observationFunction(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs sensorState)
{
  (void)sensorState; // TODO: Someday we want to implement "uncertainty about
                     // VMU location/orientation". For now, we don't use this
                     // parameter.
  dynamics::BodyNode* body = skel->getBodyNode(mBodyName);
  if (body == nullptr)
    return Eigen::Vector9s::Zero();

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> sensorList;
  sensorList.emplace_back(body, mLocalTransform);

  // Make the measurement
  Eigen::Vector9s result = Eigen::Vector9s::Zero();
  result.head<3>() = skel->getGyroReadings(sensorList);
  result.segment<3>(3) = skel->getRotationalAccelerometerReadings(sensorList);
  result.segment<3>(6) = skel->getAccelerometerReadings(sensorList);

  return result;
}

/// This method will compute the total Jacobian of the observation function,
/// with respect to `wrt`
Eigen::MatrixXs SensorVMU::observationJacobianWrt(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXs sensorState,
    neural::WithRespectTo* wrt)
{
  (void)sensorState; // TODO: Someday we want to implement "uncertainty about
                     // VMU location/orientation". For now, we don't use this
                     // parameter.
  dynamics::BodyNode* body = skel->getBodyNode(mBodyName);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(9, wrt->dim(skel.get()));
  if (body == nullptr)
    return J;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> sensorList;
  sensorList.emplace_back(body, mLocalTransform);

  // Make the measurement
  J.block(0, 0, 3, J.cols())
      = skel->getGyroReadingsJacobianWrt(sensorList, wrt);
  J.block(3, 0, 3, J.cols())
      = skel->getRotationalAccelerometerReadingsJacobianWrt(sensorList, wrt);
  J.block(6, 0, 3, J.cols())
      = skel->getRotationalAccelerometerReadingsJacobianWrt(sensorList, wrt);

  return J;
}

/// This method will compute the total Jacobian of the observation function,
/// with respect to the sensor state vector
Eigen::MatrixXs SensorVMU::observationJacobianWrtSensorState(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs sensorState)
{
  (void)sensorState; // TODO: Someday we want to implement "uncertainty about
                     // VMU location/orientation". For now, we don't use this
                     // parameter.
  (void)skel;
  return Eigen::MatrixXs::Zero(0, 0);
}

/// This returns the noise of the sensor. This can be a constant value, but is
/// also allowed to vary based on the state of the skeleton
Eigen::MatrixXs SensorVMU::noiseCovariance(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs sensorState)
{
  (void)sensorState; // TODO: Someday we want to implement "uncertainty about
                     // VMU location/orientation". For now, we don't use this
                     // parameter.
  (void)skel;
  return Eigen::MatrixXs::Identity(9, 9);
}

} // namespace sensors
} // namespace dart