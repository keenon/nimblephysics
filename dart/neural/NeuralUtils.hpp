#ifndef DART_NEURAL_UTILS_HPP_
#define DART_NEURAL_UTILS_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace constraint {
class ConstrainedGroup;
}
namespace simulation {
class World;
}

namespace neural {

struct LossGradient
{
  Eigen::VectorXd lossWrtPosition;
  Eigen::VectorXd lossWrtVelocity;
  Eigen::VectorXd lossWrtTorque;
};

// We don't issue a full import here, because we want this file to be safe to
// import from anywhere else in DART
class ConstrainedGroupGradientMatrices;
class BackpropSnapshot;

std::shared_ptr<ConstrainedGroupGradientMatrices> createGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep);

/// Takes a step in the world, and returns a backprop snapshot which can be used
/// to backpropagate gradients and compute Jacobians
std::shared_ptr<BackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world, bool idempotent = false);

/// This unrolls a trajectory with multiple knot points by exploiting the
/// available parallelism by running each knot on its own thread.
/// This is implemented in C++ with the explicit purpose of calling it from
/// Python.
std::vector<std::shared_ptr<BackpropSnapshot>> bulkForwardPass(
    std::shared_ptr<simulation::World> world,
    std::vector<Eigen::VectorXd> torques,
    std::size_t shootingLength,
    std::vector<Eigen::VectorXd> knotPoses,
    std::vector<Eigen::VectorXd> knotVels);

} // namespace neural
} // namespace dart

#endif