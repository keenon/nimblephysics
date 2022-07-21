#ifndef DART_UTILS_SKELCONVERTER_HPP_
#define DART_UTILS_SKELCONVERTER_HPP_

#include <string>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace biomechanics {

class SkeletonConverter
{
public:
  /// We're going to try to move and rescale "source", so that it more closely
  /// matches "target"
  SkeletonConverter(dynamics::SkeletonPtr source, dynamics::SkeletonPtr target);

  /// This will register two joints as representing the same real underlying
  /// joint on both skeletons. These joints may be of different types (like a
  /// BallJoint vs a CustomJoint) and we'll do our best to match it up so the
  /// rotations are as close as possible.
  void linkJoints(dynamics::Joint* sourceJoint, dynamics::Joint* targetJoint);

  /// This assumes that both skeletons are already scaled and aligned, and just
  /// goes through and adds fake markers
  void createVirtualMarkers(
      int addFakeMarkers = 3, s_t weightFakeMarkers = 0.1);

  /// This will do its best to map the target onto the source skeleton
  void rescaleAndPrepTarget(
      int addFakeMarkers = 3,
      s_t weightFakeMarkers = 0.1,
      // IK options - default to a fairly close fit since we only do this once
      // and this will effect all downstream steps
      s_t convergenceThreshold = 1e-15,
      int maxStepCount = 1000,
      s_t leastSquaresDamping = 0.01,
      bool lineSearch = true,
      bool logOutput = false);

  /// This will try to get the source skeleton configured to match the target as
  /// closely as possible
  s_t fitSourceToTarget(
      s_t convergenceThreshold = 1e-7,
      int maxStepCount = 100,
      s_t leastSquaresDamping = 0.01,
      bool lineSearch = true,
      bool logOutput = false);

  /// This will try to get the target skeleton configured to match the source as
  /// closely as possible. This is mostly just here for debugging, in general
  /// there isn't much point to trying to fit the target skeleton back to the
  /// source.
  s_t fitTargetToSource(
      s_t convergenceThreshold = 1e-7,
      int maxStepCount = 100,
      s_t leastSquaresDamping = 0.01,
      bool lineSearch = true,
      bool logOutput = false);

  /// This converts a motion from the target skeleton to the source skeleton
  Eigen::MatrixXs convertMotion(
      Eigen::MatrixXs targetMotion,
      bool logProgress = true,
      // IK Options
      s_t convergenceThreshold = 1e-7,
      int maxStepCount = 100,
      s_t leastSquaresDamping = 0.01,
      bool lineSearch = true,
      bool logIKOutput = false);

  /// This returns the concatenated 3-vectors for world positions of each joint
  /// in 3D world space, for the registered target joints.
  Eigen::VectorXs getSourceJointWorldPositions();

  /// This returns the concatenated 3-vectors for world positions of each "fake"
  /// marker in 3D world space, for the registered target joints.
  Eigen::VectorXs getSourceMarkerWorldPositions();

  /// This returns the concatenated 3-vectors for world positions of each marker
  /// in 3D world space, for the registered target joints.
  Eigen::VectorXs getTargetJointWorldPositions();

  /// This returns the concatenated 3-vectors for world positions of each "fake"
  /// marker in 3D world space, for the registered target joints.
  Eigen::VectorXs getTargetMarkerWorldPositions();

  /// This will display the state of the linkages between the two skeletons into
  /// the provided GUI.
  void debugToGUI(std::shared_ptr<server::GUIWebsocketServer> server);

  const std::vector<dynamics::Joint*>& getSourceJoints() const;

  const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
  getSourceMarkers() const;

  const std::vector<dynamics::Joint*>& getTargetJoints() const;

  const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
  getTargetMarkers() const;

protected:
  dynamics::SkeletonPtr mSourceSkeleton;
  dynamics::SkeletonPtr mSourceSkeletonBallJoints;
  dynamics::SkeletonPtr mTargetSkeleton;

  std::vector<dynamics::Joint*> mSourceJoints;
  std::vector<dynamics::Joint*> mSourceJointsWithBalls;
  std::vector<dynamics::Joint*> mTargetJoints;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mSourceMarkers;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      mSourceMarkersBallJoints;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mTargetMarkers;
  Eigen::VectorXs mMarkerWeights;
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif