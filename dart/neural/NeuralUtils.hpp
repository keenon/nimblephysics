#ifndef DART_NEURAL_UTILS_HPP_
#define DART_NEURAL_UTILS_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

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

//////////////////////////////////////////////////////////////////////////////
// Geometry helpers
//////////////////////////////////////////////////////////////////////////////

/// This converts a batch of joint space positions (one per column) to a batch
/// of world screws (one per column).
Eigen::MatrixXd convertJointSpacePositionsToWorldSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions);

/// This converts a batch of joint space velocities (one per column) to a batch
/// of world screws (one per column).
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocity);

/// This turns a batch of losses in world screws into a batch of losses in joint
/// space.
Eigen::MatrixXd backpropWorldSpatialToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd bodySpaceLoss,
    bool useTranspose = false);

/// This converts a batch of joint space positions (one per column) to a batch
/// of world positions (one per column).
Eigen::MatrixXd convertJointSpacePositionsToWorldLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions);

/// This converts a batch of joint space velocities (one per column) to a batch
/// of world velocities (one per column).
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocity);

/// This turns a batch of losses in world space into a batch of losses in joint
/// space.
Eigen::MatrixXd backpropWorldLinearToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd lossWrtWorldLinear,
    bool useTranspose = false);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::MatrixXd convertJointSpacePositionsToWorldCOM(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldCOMLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities);

/// This turns a batch of losses in world space wrt center of masses into a
/// batch of losses in joint space.
Eigen::MatrixXd backpropWorldCOMLinearToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd lossWrtWorldCOM,
    bool useTranspose = false);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldCOMSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities);

/// This turns a batch of losses in world space wrt center of masses into a
/// batch of losses in joint space.
Eigen::MatrixXd backpropWorldCOMSpatialToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd lossWrtWorldCOM,
    bool useTranspose = false);

//////////////////////////////////////////////
// Similar to above, but just for Skeletons //
//////////////////////////////////////////////

/// Computes a Jacobian that transforms changes in joint angle to changes in
/// body positions (expressed in log space).
Eigen::MatrixXd jointToWorldSpatialJacobian(
    std::shared_ptr<dynamics::Skeleton> skel);

/// Computes a Jacobian that transforms changes in joint angle to changes in
/// body positions (expressed in linear space).
Eigen::MatrixXd jointToWorldLinearJacobian(
    std::shared_ptr<dynamics::Skeleton> skel);

/// Convert a set of joint positions to a vector of body positions in world
/// space (expressed in log space).
Eigen::VectorXd skelConvertJointSpacePositionsToWorldSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions);

/// Convert a set of joint velocities to a vector of body velocities in world
/// space (expressed in log space).
Eigen::VectorXd skelConvertJointSpaceVelocitiesToWorldSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions);

/// Turns losses in terms of body space into losses in terms of joint space
Eigen::VectorXd skelBackpropWorldSpatialToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd bodySpace,
    bool useTranspose = false);

/// Convert a set of joint positions to a vector of body positions in world
/// space (expressed in log space).
Eigen::VectorXd skelConvertJointSpacePositionsToWorldLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions);

/// Convert a set of joint velocities to a vector of body velocities in world
/// space (expressed in log space).
Eigen::VectorXd skelConvertJointSpaceVelocitiesToWorldLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions);

/// Turns losses in terms of body space into losses in terms of joint space
Eigen::VectorXd skelBackpropWorldLinearToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd lossWrtWorldLinear,
    bool useTranspose = false);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::Vector3d skelConvertJointSpacePositionsToWorldCOM(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions);

/// This turns a batch of losses in world space wrt center of masses into a
/// batch of losses in joint space.
Eigen::VectorXd skelBackpropWorldCOMLinearToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector3d lossWrtWorldCOM,
    bool useTranspose = false);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::Vector3d skelConvertJointSpaceVelocitiesToWorldCOMLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVelocities);

/// This converts a batch of joint space positions (one per column) to the
/// center of mass positions in world coordinates.
Eigen::Vector6d skelConvertJointSpaceVelocitiesToWorldCOMSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVelocities);

/// This turns a batch of losses in world space wrt center of masses into a
/// batch of losses in joint space.
Eigen::VectorXd skelBackpropWorldCOMSpatialToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector6d lossWrtWorldCOM,
    bool useTranspose = false);

} // namespace neural
} // namespace dart

#endif