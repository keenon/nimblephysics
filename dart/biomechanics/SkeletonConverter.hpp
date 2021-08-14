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
  void linkJoints(
      const dynamics::Joint* sourceJoint, const dynamics::Joint* targetJoint);

  /// This will do its best to map the target onto the source skeleton
  void rescaleAndPrepTarget();

  /// This converts a motion from the target skeleton to the source skeleton
  Eigen::MatrixXs convertMotion(Eigen::MatrixXs targetMotion);

  /// This returns the concatenated 3-vectors for world positions of each joint
  /// in 3D world space, for the registered target joints.
  Eigen::VectorXs getSourceJointWorldPositions();

  /// This returns the concatenated 3-vectors for world positions of each joint
  /// in 3D world space, for the registered target joints.
  Eigen::VectorXs getTargetJointWorldPositions();

  /// This will display the state of the linkages between the two skeletons into
  /// the provided GUI.
  void debugToGUI(std::shared_ptr<server::GUIWebsocketServer> server);

  const std::vector<const dynamics::Joint*>& getSourceJoints() const;

  const std::vector<const dynamics::Joint*>& getTargetJoints() const;

protected:
  dynamics::SkeletonPtr mSourceSkeleton;
  dynamics::SkeletonPtr mTargetSkeleton;

  std::vector<const dynamics::Joint*> mSourceJoints;
  std::vector<const dynamics::Joint*> mTargetJoints;
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif