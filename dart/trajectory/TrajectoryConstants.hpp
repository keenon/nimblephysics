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

typedef std::function<double(
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces)>
    TrajectoryLossFn;

typedef std::function<void(
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces)>
    TrajectoryLossFnGrad;

} // namespace trajectory
} // namespace dart

#endif