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

struct BulkForwardPassResult
{
  std::vector<std::shared_ptr<BackpropSnapshot>> snapshots;
  Eigen::MatrixXd postStepPoses;
  Eigen::MatrixXd postStepVels;
};

/// This unrolls a trajectory with multiple knot points by exploiting the
/// available parallelism by running each knot on its own thread.
/// This is implemented in C++ with the explicit purpose of calling it from
/// Python.
BulkForwardPassResult bulkForwardPass(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd torques,
    std::size_t shootingLength,
    Eigen::MatrixXd knotPoses,
    Eigen::MatrixXd knotVels);

struct KnotJacobian
{
  Eigen::MatrixXd knotPosEndPos;
  Eigen::MatrixXd knotVelEndPos;
  Eigen::MatrixXd knotPosEndVel;
  Eigen::MatrixXd knotVelEndVel;
  std::vector<Eigen::MatrixXd> torquesEndPos;
  std::vector<Eigen::MatrixXd> torquesEndVel;
};

struct BulkBackwardPassResult
{
  Eigen::MatrixXd gradWrtPreStepKnotPoses;
  Eigen::MatrixXd gradWrtPreStepKnotVels;
  Eigen::MatrixXd gradWrtPreStepTorques;
  std::vector<KnotJacobian> knotJacobians;
};

/// This is the companion to bulkForwardPass(), and runs the gradients back
/// up the stack in parallel, by exploiting the fact that gradients across
/// knots are independent.
/// This is implemented in C++ with the explicit purpose of calling it from
/// Python.
BulkBackwardPassResult bulkBackwardPass(
    std::shared_ptr<simulation::World> world,
    std::vector<std::shared_ptr<BackpropSnapshot>> snapshots,
    std::size_t shootingLength,
    Eigen::MatrixXd gradWrtPoses,
    Eigen::MatrixXd gradWrtVels,
    bool computeJacobians = true);

} // namespace neural
} // namespace dart

#endif