#include "dart/biomechanics/SkeletonConverter.hpp"

#include <unordered_map>
#include <vector>

#include "dart/dynamics/DegreeOfFreedom.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
SkeletonConverter::SkeletonConverter(
    dynamics::SkeletonPtr source, dynamics::SkeletonPtr target)
  : mSourceSkeleton(source), mTargetSkeleton(target)
{
}

//==============================================================================
/// This will register two joints as representing the same real underlying
/// joint on both skeletons. These joints may be of different types (like a
/// BallJoint vs a CustomJoint) and we'll do our best to match it up so the
/// rotations are as close as possible.
void SkeletonConverter::linkJoints(
    const dynamics::Joint* sourceJoint, const dynamics::Joint* targetJoint)
{
  mSourceJoints.push_back(sourceJoint);
  mTargetJoints.push_back(targetJoint);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered source joints.
Eigen::VectorXs SkeletonConverter::getSourceJointWorldPositions()
{
  return mSourceSkeleton->getJointWorldPositions(mSourceJoints);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getTargetJointWorldPositions()
{
  return mTargetSkeleton->getJointWorldPositions(mTargetJoints);
}

//==============================================================================
/// This will do its best to map the target onto the source skeleton
void SkeletonConverter::rescaleAndPrepTarget()
{
  mSourceSkeleton->fitJointsToWorldPositions(
      mSourceJoints, getTargetJointWorldPositions(), true);
}

//==============================================================================
/// This converts a motion from the target skeleton to the source skeleton
Eigen::MatrixXs SkeletonConverter::convertMotion(Eigen::MatrixXs targetMotion)
{
  Eigen::MatrixXs sourceMotion = Eigen::MatrixXs::Zero(
      mSourceSkeleton->getNumDofs(), targetMotion.cols());

  Eigen::VectorXs originalSource = mSourceSkeleton->getPositions();
  Eigen::VectorXs originalTarget = mTargetSkeleton->getPositions();

  // Take a few hundred iterations of IK to get a really good fit on the first
  // frame
  mTargetSkeleton->setPositions(targetMotion.col(0));
  mSourceSkeleton->fitJointsToWorldPositions(
      mSourceJoints, getTargetJointWorldPositions(), false, 300);

  for (int i = 0; i < targetMotion.cols(); i++)
  {
    // Each subsequent frame after the first one doesn't need as many steps of
    // IK
    mTargetSkeleton->setPositions(targetMotion.col(i));
    mSourceSkeleton->fitJointsToWorldPositions(
        mSourceJoints, getTargetJointWorldPositions(), false, 30);

    sourceMotion.col(i) = mSourceSkeleton->getPositions();
  }

  mSourceSkeleton->setPositions(originalSource);
  mTargetSkeleton->setPositions(originalTarget);

  return sourceMotion;
}

//==============================================================================
/// This will display the state of the linkages between the two skeletons into
/// the provided GUI.
void SkeletonConverter::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server)
{
  Eigen::VectorXs sourcePositions = getSourceJointWorldPositions();
  Eigen::VectorXs targetPositions = getTargetJointWorldPositions();
  for (int i = 0; i < mSourceJoints.size(); i++)
  {
    Eigen::Vector3s sourcePos = sourcePositions.segment<3>(i * 3);
    Eigen::Vector3s targetPos = targetPositions.segment<3>(i * 3);
    std::vector<Eigen::Vector3s> line;
    line.push_back(sourcePos);
    line.push_back(targetPos);
    server->createLine(
        "SkeletonConverter_link_line_" + std::to_string(i),
        line,
        Eigen::Vector3s::UnitX());
  }
}

//==============================================================================
const std::vector<const dynamics::Joint*>& SkeletonConverter::getSourceJoints()
    const
{
  return mSourceJoints;
}

//==============================================================================
const std::vector<const dynamics::Joint*>& SkeletonConverter::getTargetJoints()
    const
{
  return mTargetJoints;
}

} // namespace biomechanics
} // namespace dart