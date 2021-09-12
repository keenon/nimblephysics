#ifndef DART_BIOMECH_MARKERFITTER_HPP_
#define DART_BIOMECH_MARKERFITTER_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct MarkerFitResult
{
  Eigen::MatrixXs poses;
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      adjustedMarkers;
};

class MarkerFitter
{
public:
  MarkerFitter(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
          markers);

  /// This method will set `skeleton` to the configuration given by the vectors
  /// of jointPositions and groupScales. It will also compute and return the
  /// list of markers given by markerDiffs.
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
  setConfiguration(
      std::shared_ptr<dynamics::Skeleton>& skeleton,
      Eigen::VectorXs jointPositions,
      Eigen::VectorXs groupScales,
      Eigen::VectorXs markerDiffs);

  /// This computes a vector of concatenated differences between where markers
  /// are and where the observed markers are. Unobserved markers are assumed to
  /// have a difference of zero.
  Eigen::VectorXs getMarkerError(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the overall objective term for the MarkerFitter for a single
  /// timestep. The MarkerFitter is trying to do a bilevel optimization to
  /// minimize this term.
  s_t computeLoss(Eigen::VectorXs markerError);

  //////////////////////////////////////////////////////////////////////////
  // First order gradients
  //////////////////////////////////////////////////////////////////////////

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs getLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs finiteDifferenceLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs getLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs finiteDifferenceLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs getLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs finiteDifferenceLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  //////////////////////////////////////////////////////////////////////////
  // Jacobians of the gradient wrt joints (for bilevel optimization)
  //////////////////////////////////////////////////////////////////////////

  /// Get the marker indices that are not visible
  std::vector<int> getSparsityMap(
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs getMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs getMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs getMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
};

} // namespace biomechanics

} // namespace dart

#endif