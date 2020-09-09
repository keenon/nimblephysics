#ifndef DART_TRAJECTORY_CONSTANTS_HPP_
#define DART_TRAJECTORY_CONSTANTS_HPP_

#include <functional>

namespace dart {
namespace trajectory {

struct TimestepJacobians
{
  Eigen::MatrixXd posPos;
  Eigen::MatrixXd velPos;
  Eigen::MatrixXd forcePos;
  Eigen::MatrixXd posVel;
  Eigen::MatrixXd velVel;
  Eigen::MatrixXd forceVel;
};

} // namespace trajectory
} // namespace dart

#endif