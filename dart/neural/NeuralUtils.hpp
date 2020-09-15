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
class Mapping;
class MappedBackpropSnapshot;

std::shared_ptr<ConstrainedGroupGradientMatrices> createGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep);

/// Takes a step in the world, and returns a backprop snapshot which can be used
/// to backpropagate gradients and compute Jacobians
std::shared_ptr<BackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world, bool idempotent = false);

/// Takes a step in the world, and returns a mapped snapshot which can be used
/// to backpropagate gradients and compute Jacobians in the mapped space
std::shared_ptr<MappedBackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<Mapping> representationMapping,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> lossMappings,
    bool idempotent = false);

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

enum ConvertToSpace
{
  POS_SPATIAL,
  POS_LINEAR,
  VEL_SPATIAL,
  VEL_LINEAR,
  COM_POS,
  COM_VEL_SPATIAL,
  COM_VEL_LINEAR
};

/// Convert a set of joint positions to a vector of body positions in world
/// space (expressed in log space).
Eigen::MatrixXd convertJointSpaceToWorldSpace(
    const std::shared_ptr<simulation::World>& world,
    const Eigen::MatrixXd& in, /* These can be velocities or positions,
                                    depending on the value of `space` */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space,
    bool backprop = false,
    bool useIK = true /* Only relevant for backprop */);

//////////////////////////////////////////////
// Similar to above, but just for Skeletons //
//////////////////////////////////////////////

/// Computes a Jacobian that transforms changes in joint angle to changes in
/// body positions (expressed in log space).
Eigen::MatrixXd jointToWorldSpatialJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Computes a Jacobian that transforms changes in joint angle to changes in
/// body positions (expressed in linear space).
Eigen::MatrixXd jointToWorldLinearJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Convert a set of joint positions to a vector of body positions in world
/// space (expressed in log space).
Eigen::VectorXd skelConvertJointSpaceToWorldSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXd& jointValues, /* These can be velocities or positions,
                                    depending on the value of `space` */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space);

/// Turns losses in terms of body space into losses in terms of joint space
Eigen::VectorXd skelBackpropWorldSpaceToJointSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXd& bodySpace, /* This is the gradient in body space */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space, /* This is the source space for our gradient */
    bool useIK = true);

} // namespace neural
} // namespace dart

#endif