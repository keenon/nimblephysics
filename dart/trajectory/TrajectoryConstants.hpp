#ifndef DART_TRAJECTORY_CONSTANTS_HPP_
#define DART_TRAJECTORY_CONSTANTS_HPP_

#include <functional>

#include <Eigen/Dense>

namespace dart {
namespace trajectory {

struct TimestepJacobians
{
  Eigen::MatrixXd posPos;
  Eigen::MatrixXd velPos;
  Eigen::MatrixXd forcePos;
  Eigen::MatrixXd massPos;
  Eigen::MatrixXd posVel;
  Eigen::MatrixXd velVel;
  Eigen::MatrixXd forceVel;
  Eigen::MatrixXd massVel;
};

} // namespace trajectory
} // namespace dart

#endif