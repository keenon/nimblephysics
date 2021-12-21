#ifndef DART_NEURAL_UTILS_HPP_
#define DART_NEURAL_UTILS_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dart/include_eigen.hpp"

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
  Eigen::VectorXs lossWrtPosition;
  Eigen::VectorXs lossWrtVelocity;
  Eigen::VectorXs lossWrtTorque;
  Eigen::VectorXs lossWrtMass;
};

struct LossGradientHighLevelAPI
{
  Eigen::VectorXs lossWrtState;
  Eigen::VectorXs lossWrtAction;
  Eigen::VectorXs lossWrtMass;
};

// We don't issue a full import here, because we want this file to be safe to
// import from anywhere else in DART
class ConstrainedGroupGradientMatrices;
class BackpropSnapshot;
class Mapping;
class MappedBackpropSnapshot;

std::shared_ptr<ConstrainedGroupGradientMatrices> createGradientMatrices(
    constraint::ConstrainedGroup& group, s_t timeStep);

/// Takes a step in the world, and returns a backprop snapshot which can be used
/// to backpropagate gradients and compute Jacobians
std::shared_ptr<BackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world, bool idempotent = false);

/// Takes a step in the world, and returns a mapped snapshot which can be used
/// to backpropagate gradients and compute Jacobians in the mapped space
std::shared_ptr<MappedBackpropSnapshot> mappedForwardPass(
    std::shared_ptr<simulation::World> world,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> mappings,
    bool idempotent = false);

struct KnotJacobian
{
  Eigen::MatrixXs knotPosEndPos;
  Eigen::MatrixXs knotVelEndPos;
  Eigen::MatrixXs knotPosEndVel;
  Eigen::MatrixXs knotVelEndVel;
  std::vector<Eigen::MatrixXs> torquesEndPos;
  std::vector<Eigen::MatrixXs> torquesEndVel;
};

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
Eigen::MatrixXs convertJointSpaceToWorldSpace(
    const std::shared_ptr<simulation::World>& world,
    const Eigen::MatrixXs& in, /* These can be velocities or positions,
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
Eigen::MatrixXs jointPosToWorldSpatialJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Computes a Jacobian that transforms changes in joint angle to changes in
/// body positions (expressed in linear space).
Eigen::MatrixXs jointPosToWorldLinearJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Computes a Jacobian that transforms changes in joint velocity to changes in
/// body velocity (expressed in log space).
Eigen::MatrixXs jointVelToWorldSpatialJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Computes a Jacobian that transforms changes in joint velocity to changes in
/// body velocity (expressed in linear space).
Eigen::MatrixXs jointVelToWorldLinearJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes);

/// Convert a set of joint positions to a vector of body positions in world
/// space (expressed in log space).
Eigen::VectorXs skelConvertJointSpaceToWorldSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXs& jointValues, /* These can be velocities or positions,
                                    depending on the value of `space` */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space);

/// Turns losses in terms of body space into losses in terms of joint space
Eigen::VectorXs skelBackpropWorldSpaceToJointSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXs& bodySpace, /* This is the gradient in body space */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space, /* This is the source space for our gradient */
    bool useIK = true);

} // namespace neural
} // namespace dart

#endif