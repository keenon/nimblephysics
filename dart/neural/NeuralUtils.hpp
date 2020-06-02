#ifndef DART_NEURAL_UTILS_HPP_
#define DART_NEURAL_UTILS_HPP_

#include <memory>

namespace dart {
namespace constraint {
class ConstrainedGroup;
}
namespace simulation {
class World;
}

namespace neural {

// We don't issue a full import here, because we want this file to be safe to
// import from anywhere else in DART
class ConstrainedGroupGradientMatrices;
class BackpropSnapshot;

enum GradientMode
{
  NONE,
  MASSED,
  CLASSIC
};

std::shared_ptr<ConstrainedGroupGradientMatrices> createGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep, GradientMode mode);

/// Takes a step in the world, and returns a backprop snapshot which can be used
/// to backpropagate gradients and compute Jacobians
std::shared_ptr<BackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world,
    GradientMode mode,
    bool idempotent = false);

} // namespace neural
} // namespace dart

#endif