#ifndef DART_SENSOR_SET_HPP_
#define DART_SENSOR_SET_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/sensors/Sensor.hpp"

namespace dart {
namespace sensors {

class FilterState
{
public:
  FilterState();

  /// This is a list of names of the joints we will be tracking in the state.
  /// Any joints not in this list will not be tracked.
  void setIncludedJoints(std::vector<std::string> joints);

  /// If true, we include the acceleration of the joints in the state.
  void setIncludeAcceleration(bool useAcc);

  /// This returns the joints states that are in our `mIncludedJoints` list.
  Eigen::VectorXs getState(std::shared_ptr<dynamics::Skeleton> skel);

  /// This sets the joints states that are in our `mIncludedJoints` list.
  void setState(std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x);

  /// This method will return the predicted outputs of all the sensors in the
  /// set, concatenated together.
  Eigen::VectorXs observationFunction(
      std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x);

  /// This method will compute the total Jacobian of the observation function
  Eigen::MatrixXs observationJacobian(
      std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x);

  /// This method will compute the transition function on the given skeleton,
  /// and return the new state.
  Eigen::VectorXs transitionFunction(
      std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x, s_t dt);

  /// This method will compute the Jacobian of the transition function
  Eigen::VectorXs transitionJacobian(
      std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x, s_t dt);

  /// This adds a sensor to the list
  void addSensor(std::shared_ptr<Sensor> sensor);

protected:
  bool mUseAcceleration;
  std::vector<std::string> mIncludedJoints;
  std::vector<std::shared_ptr<Sensor>> mSensors;
};

} // namespace sensors
} // namespace dart

#endif