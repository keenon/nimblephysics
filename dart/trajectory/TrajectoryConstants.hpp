#ifndef DART_TRAJECTORY_CONSTANTS_HPP_
#define DART_TRAJECTORY_CONSTANTS_HPP_

#include <functional>

#include <Eigen/Dense>

namespace dart {
namespace trajectory {

struct TimestepJacobians
{
  Eigen::MatrixXs posPos;
  Eigen::MatrixXs velPos;
  Eigen::MatrixXs forcePos;
  Eigen::MatrixXs massPos;
  Eigen::MatrixXs posVel;
  Eigen::MatrixXs velVel;
  Eigen::MatrixXs forceVel;
  Eigen::MatrixXs massVel;
};

} // namespace trajectory
} // namespace dart

#endif