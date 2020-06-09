#ifndef DART_NEURAL_UTILS_HPP_
#define DART_NEURAL_UTILS_HPP_

#include <memory>

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

} // namespace neural
} // namespace dart

#endif