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
  mSourceSkeletonBallJoints = mSourceSkeleton->convertSkeletonToBallJoints();
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
  for (int i = 0; i < mSourceSkeleton->getNumJoints(); i++)
  {
    if (sourceJoint == mSourceSkeleton->getJoint(i))
    {
      mSourceJointsWithBalls.push_back(mSourceSkeletonBallJoints->getJoint(i));
      break;
    }
  }
  assert(mSourceJointsWithBalls.size() == mSourceJoints.size());
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
/// This returns the concatenated 3-vectors for world positions of each "fake"
/// marker in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getSourceMarkerWorldPositions()
{
  return mSourceSkeleton->getMarkerWorldPositions(mSourceMarkers);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getTargetJointWorldPositions()
{
  return mTargetSkeleton->getJointWorldPositions(mTargetJoints);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each "fake"
/// marker in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getTargetMarkerWorldPositions()
{
  return mTargetSkeleton->getMarkerWorldPositions(mTargetMarkers);
}

//==============================================================================
/// This will do its best to map the target onto the source skeleton
void SkeletonConverter::rescaleAndPrepTarget(
    int addFakeMarkers,
    s_t weightFakeMarkers,
    ////// IK options
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  if (addFakeMarkers < 0)
  {
    std::cout
        << "rescaleAndPrepTarget() expects addFakeMarkers between 0 and 3. Got "
        << addFakeMarkers << ", so clamping to 0" << std::endl;
    addFakeMarkers = 0;
  }
  if (addFakeMarkers > 3)
  {
    std::cout
        << "rescaleAndPrepTarget() expects addFakeMarkers between 0 and 3. Got "
        << addFakeMarkers << ", so clamping to 3" << std::endl;
    addFakeMarkers = 3;
  }
  mSourceSkeleton->fitJointsToWorldPositions(
      mSourceJoints,
      getTargetJointWorldPositions(),
      true,
      convergenceThreshold,
      maxStepCount,
      leastSquaresDamping,
      lineSearch,
      logOutput);
  for (int i = 0; i < mSourceSkeleton->getNumBodyNodes(); i++)
  {
    mSourceSkeletonBallJoints->getBodyNode(i)->setScale(
        mSourceSkeleton->getBodyNode(i)->getScale());
  }
#ifndef NDEBUG
  for (int i = 0; i < mSourceSkeleton->getNumJoints(); i++)
  {
    dynamics::Joint* sourceJoint = mSourceSkeleton->getJoint(i);
    dynamics::Joint* sourceJointWithBalls
        = mSourceSkeletonBallJoints->getJoint(i);
    Eigen::Matrix4s originalChild
        = sourceJoint->getTransformFromChildBodyNode().matrix();
    Eigen::Matrix4s originalParent
        = sourceJoint->getTransformFromParentBodyNode().matrix();
    Eigen::Matrix4s convertedChild
        = sourceJointWithBalls->getTransformFromChildBodyNode().matrix();
    Eigen::Matrix4s convertedParent
        = sourceJointWithBalls->getTransformFromParentBodyNode().matrix();
    assert(originalChild == convertedChild);
    assert(originalParent == convertedParent);
  }
#endif
  mMarkerWeights = Eigen::VectorXs::Ones(mTargetJoints.size() * 4);
  int cursor = 0;
  // Go through and create a bunch of "fake" 3D markers that register pairs of
  // child bodies, which will help preserve rotation information
  for (int i = 0; i < mTargetJoints.size(); i++)
  {
    const dynamics::BodyNode* targetBody = mTargetJoints[i]->getChildBodyNode();
    const dynamics::BodyNode* sourceBody = mSourceJoints[i]->getChildBodyNode();
    const dynamics::BodyNode* sourceBodyWithBalls
        = mSourceJointsWithBalls[i]->getChildBodyNode();
    for (int j = 0; j <= addFakeMarkers; j++)
    {
      /*
      // Define the unit vectors in the source body space
      Eigen::Vector3s sourceOffset;
      if (j > 0)
      {
        sourceOffset = Eigen::Vector3s::Unit(j - 1) * 0.1;
        mMarkerWeights(cursor) = 1.0;
      }
      else
      {
        sourceOffset = Eigen::Vector3s::Zero();
        mMarkerWeights(cursor) = weightFakeMarkers;
      }
      cursor++;
      mSourceMarkers.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              sourceBody, sourceOffset));
      mSourceMarkersBallJoints.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              sourceBodyWithBalls, sourceOffset));
      Eigen::Vector3s targetOffset = targetBody->getWorldTransform().inverse()
                                     * sourceBody->getWorldTransform()
                                     * sourceOffset;
      // Always align the joints to each other directly, without offset
      if (j == 0)
        targetOffset.setZero();
      mTargetMarkers.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              targetBody, targetOffset));
      */
      // Define the unit vectors in the target body space
      Eigen::Vector3s targetOffset;
      if (j > 0)
      {
        targetOffset = Eigen::Vector3s::Unit(j - 1) * 0.1;
        mMarkerWeights(cursor) = 1.0;
      }
      else
      {
        targetOffset = Eigen::Vector3s::Zero();
        mMarkerWeights(cursor) = weightFakeMarkers;
      }
      cursor++;
      mTargetMarkers.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              targetBody, targetOffset.cwiseQuotient(targetBody->getScale())));
      Eigen::Vector3s sourceOffset
          = (sourceBody->getWorldTransform().inverse()
             * targetBody->getWorldTransform() * targetOffset)
                .cwiseQuotient(sourceBody->getScale());
      // Always align the joints to each other directly, without offset
      if (j == 0)
        sourceOffset.setZero();
      mSourceMarkers.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              sourceBody, sourceOffset));
      mSourceMarkersBallJoints.push_back(
          std::pair<const dynamics::BodyNode*, Eigen::Vector3s>(
              sourceBodyWithBalls, sourceOffset));
    }
  }
#ifndef NDEBUG
  Eigen::VectorXs targetPos = getTargetMarkerWorldPositions();
  Eigen::VectorXs sourcePos = getSourceMarkerWorldPositions();
  Eigen::VectorXs diff = targetPos - sourcePos;
  for (int i = 0; i < diff.size() / 3; i++)
  {
    // Every 4th entry is a "joint" marker, which may have some error and that's
    // ok
    if (i % 4 == 0)
    {
      diff.segment<3>(i * 3).setZero();
    }
  }
  s_t error = diff.squaredNorm();
  assert(error < 1e-16);

  mSourceSkeletonBallJoints->setPositions(
      mSourceSkeleton->convertPositionsToBallSpace(
          mSourceSkeleton->getPositions()));
  Eigen::VectorXs sourcePosWithBalls
      = mSourceSkeletonBallJoints->getMarkerWorldPositions(
          mSourceMarkersBallJoints);
  diff = (sourcePos - sourcePosWithBalls);
  s_t errorFromBalls = diff.squaredNorm();
  assert(errorFromBalls < 1e-16);
#endif
}

//==============================================================================
/// This will try to get the source skeleton configured to match the target as
/// closely as possible
s_t SkeletonConverter::fitSourceToTarget(
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  // We can do this gradient descent in a
  // gimbal-lock-free version of the skeleton, and then convert back when we're
  // done. This might jump around in joint space, but it's more robust.

  mSourceSkeletonBallJoints->setPositions(
      mSourceSkeleton->convertPositionsToBallSpace(
          mSourceSkeleton->getPositions()));
  s_t error = mSourceSkeletonBallJoints->fitMarkersToWorldPositions(
      mSourceMarkersBallJoints,
      getTargetMarkerWorldPositions(),
      mMarkerWeights,
      convergenceThreshold,
      maxStepCount,
      leastSquaresDamping,
      lineSearch,
      logOutput);
  mSourceSkeleton->setPositions(mSourceSkeleton->convertPositionsFromBallSpace(
      mSourceSkeletonBallJoints->getPositions()));
  return error;
}

//==============================================================================
/// This will try to get the target skeleton configured to match the source as
/// closely as possible
s_t SkeletonConverter::fitTargetToSource(
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  s_t error = mTargetSkeleton->fitMarkersToWorldPositions(
      mTargetMarkers,
      getSourceMarkerWorldPositions(),
      mMarkerWeights,
      convergenceThreshold,
      maxStepCount,
      leastSquaresDamping,
      lineSearch,
      logOutput);
  return error;
}

//==============================================================================
/// This converts a motion from the target skeleton to the source skeleton
Eigen::MatrixXs SkeletonConverter::convertMotion(
    Eigen::MatrixXs targetMotion,
    bool logProgress,
    ////// IK options
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  std::cout << "Converting " << targetMotion.cols() << " timesteps..."
            << std::endl;

  Eigen::MatrixXs sourceMotion = Eigen::MatrixXs::Zero(
      mSourceSkeleton->getNumDofs(), targetMotion.cols());

  Eigen::VectorXs originalSource = mSourceSkeleton->getPositions();
  Eigen::VectorXs originalTarget = mTargetSkeleton->getPositions();

  // Take a few hundred iterations of IK to get a really good fit on the first
  // frame
  mTargetSkeleton->setPositions(targetMotion.col(0));
  fitSourceToTarget(
      convergenceThreshold,
      maxStepCount,
      leastSquaresDamping,
      lineSearch,
      logOutput);

  for (int i = 0; i < targetMotion.cols(); i++)
  {
    if (logProgress && (i % 20 == 0))
    {
      std::cout << "Converted " << i << "/" << targetMotion.cols() << std::endl;
    }
    // Each subsequent frame after the first one doesn't need as many steps of
    // IK
    mTargetSkeleton->setPositions(targetMotion.col(i));
    Eigen::VectorXs originalPos = mSourceSkeleton->getPositions();
    s_t bestError = fitSourceToTarget(
        convergenceThreshold,
        maxStepCount,
        leastSquaresDamping,
        lineSearch,
        logOutput);
    if (bestError > 0.1)
    {
      std::cout << "ERROR: Had a terrible fit! Got a best error " << bestError
                << " > 0.1. Breaking "
                   "early, because this run will be garbage. Here's the "
                   "original pos of the source skeleton, for debugging:"
                << std::endl;
      std::cout << "Eigen::VectorXs originalPos = Eigen::VectorXs("
                << originalPos.size() << ");" << std::endl;
      std::cout << "originalPos << ";
      for (int i = 0; i < originalPos.size(); i++)
      {
        std::cout << originalPos(i);
        if (i == originalPos.size() - 1)
        {
          std::cout << ";" << std::endl;
        }
        else
        {
          std::cout << "," << std::endl;
        }
      }
      Eigen::VectorXs targetPos = mTargetSkeleton->getPositions();
      std::cout << "Eigen::VectorXs targetPos = Eigen::VectorXs("
                << targetPos.size() << ");" << std::endl;
      std::cout << "targetPos << ";
      for (int i = 0; i < targetPos.size(); i++)
      {
        std::cout << targetPos(i);
        if (i == targetPos.size() - 1)
        {
          std::cout << ";" << std::endl;
        }
        else
        {
          std::cout << "," << std::endl;
        }
      }
      // Re-run the fit with logs on, to see what happened (and provide
      // breakpoints)
      mSourceSkeleton->setPositions(originalPos);
      // Just angle
      /*
      std::cout << "Fitting angles" << std::endl;
      mSourceSkeleton->fitJointsToWorldPositions(
          std::vector<const dynamics::Joint*>(),
          Eigen::VectorXs::Zero(0),
          mSourceJoints,
          getTargetJointWorldAngles(),
          false,
          100,
          true,
          true);
      */
      // Just position
      std::cout << "Fitting position" << std::endl;
      mSourceSkeleton->fitJointsToWorldPositions(
          mSourceJoints,
          getTargetJointWorldPositions(),
          false,
          100,
          true,
          true);

      return sourceMotion;
    }

    sourceMotion.col(i) = mSourceSkeleton->getPositions();
  }

  mSourceSkeleton->setPositions(originalSource);
  mTargetSkeleton->setPositions(originalTarget);

  std::cout << "Finished converting " << targetMotion.cols() << " timesteps!"
            << std::endl;

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

  Eigen::VectorXs sourceMarkers = getSourceMarkerWorldPositions();
  Eigen::VectorXs targetMarkers = getTargetMarkerWorldPositions();
  for (int i = 0; i < mSourceMarkers.size(); i++)
  {
    Eigen::Vector3s sourcePos = sourceMarkers.segment<3>(i * 3);
    Eigen::Vector3s targetPos = targetMarkers.segment<3>(i * 3);
    std::vector<Eigen::Vector3s> line;
    line.push_back(sourcePos);
    line.push_back(targetPos);
    server->createLine(
        "SkeletonConverter_marker_line_" + std::to_string(i),
        line,
        Eigen::Vector3s::UnitY());
    server->createSphere(
        "SkeletonConverter_marker_target_" + std::to_string(i),
        0.01,
        targetPos,
        Eigen::Vector3s::UnitY());
  }
}

//==============================================================================
const std::vector<const dynamics::Joint*>& SkeletonConverter::getSourceJoints()
    const
{
  return mSourceJoints;
}

//==============================================================================
const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
SkeletonConverter::getSourceMarkers() const
{
  return mSourceMarkers;
}

//==============================================================================
const std::vector<const dynamics::Joint*>& SkeletonConverter::getTargetJoints()
    const
{
  return mTargetJoints;
}

//==============================================================================
const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
SkeletonConverter::getTargetMarkers() const
{
  return mTargetMarkers;
}

} // namespace biomechanics
} // namespace dart