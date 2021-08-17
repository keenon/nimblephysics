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
  mAngleOffsets.push_back(Eigen::Matrix3s::Identity());
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
Eigen::VectorXs SkeletonConverter::getSourceJointWorldAngles()
{
  return mSourceSkeleton->getJointWorldAngles(mSourceJoints);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world positions of each joint
/// in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getTargetJointWorldPositions()
{
  return mTargetSkeleton->getJointWorldPositions(mTargetJoints);
}

//==============================================================================
/// This returns the concatenated 3-vectors for world angles of each joint
/// in 3D world space, for the registered target joints.
Eigen::VectorXs SkeletonConverter::getTargetJointWorldAngles(bool adjusted)
{
  Eigen::VectorXs angles = mTargetSkeleton->getJointWorldAngles(mTargetJoints);
  if (adjusted)
  {
    for (int i = 0; i < angles.size() / 3; i++)
    {
#ifndef NDEBUG
      Eigen::Vector3s recovered = math::logMap(
          Eigen::Matrix3s::Identity()
          * math::expMapRot(angles.segment<3>(i * 3)));
      Eigen::Vector3s diff = recovered - angles.segment<3>(i * 3);
      if (diff.squaredNorm() > 1e-10)
      {
        std::cout << "Error! logMap(expMapRot()) not recovered!" << std::endl;
        std::cout << "Original: " << std::endl
                  << angles.segment<3>(i * 3) << std::endl;
        std::cout << "Recovered: " << std::endl << recovered << std::endl;
        std::cout << "Diff (" << diff.squaredNorm() << "): " << std::endl
                  << diff << std::endl;
      }
      assert(diff.squaredNorm() < 1e-10);
#endif

      angles.segment<3>(i * 3) = math::logMap(
          math::expMapRot(angles.segment<3>(i * 3)) * mAngleOffsets[i]);
    }
  }
  return angles;
}

//==============================================================================
/// This will do its best to map the target onto the source skeleton
void SkeletonConverter::rescaleAndPrepTarget()
{
  mSourceSkeleton->fitJointsToWorldPositions(
      mSourceJoints,
      getTargetJointWorldPositions(),
      std::vector<const dynamics::Joint*>(),
      Eigen::VectorXs::Zero(0),
      true,
      500,
      true,
      false);
  // Go through and register angle offsets from the target skeleton, which
  // we'll do our best to preserve
  for (int i = 0; i < mTargetJoints.size(); i++)
  {
    const dynamics::Joint* targetJoint = mTargetJoints[i];
    const dynamics::Joint* sourceJoint = mSourceJoints[i];

    Eigen::Matrix3s targetR
        = targetJoint->getChildBodyNode()->getWorldTransform().linear();
    Eigen::Matrix3s sourceR
        = sourceJoint->getChildBodyNode()->getWorldTransform().linear();

    mAngleOffsets[i] = targetR.transpose() * sourceR;

#ifndef NDEBUG
    Eigen::Matrix3s recovered = targetR * mAngleOffsets[i];
    assert(recovered == sourceR);
#endif
  }
}

//==============================================================================
/// This will try to get the source skeleton configured to match the target as
/// closely as possible
s_t SkeletonConverter::fitTarget(int maxFitSteps, s_t convergenceThreshold)
{
  const bool log = false;
  if (maxFitSteps == -1)
  {
    int attempt = 0;
    int stepsPerIteration = 100;
    s_t error = std::numeric_limits<s_t>::infinity();
    for (int i = 0; i < 7; i++)
    {
      // Just angle
      mSourceSkeleton->fitJointsToWorldPositions(
          std::vector<const dynamics::Joint*>(),
          Eigen::VectorXs::Zero(0),
          mSourceJoints,
          getTargetJointWorldAngles(),
          false,
          stepsPerIteration,
          true,
          log);
      // Just position
      s_t errorAfterPos = mSourceSkeleton->fitJointsToWorldPositions(
          mSourceJoints,
          getTargetJointWorldPositions(),
          std::vector<const dynamics::Joint*>(),
          Eigen::VectorXs::Zero(0),
          false,
          stepsPerIteration,
          true,
          log);
      // We've converged
      if (errorAfterPos < convergenceThreshold)
      {
        return errorAfterPos;
      }
      attempt++;
      std::cout << "> Error after attempt " << attempt << " to reach error |"
                << convergenceThreshold << "|: " << errorAfterPos << std::endl;
      stepsPerIteration *= 2;
      error = errorAfterPos;
    }
    return error;
  }
  else
  {
    // Just angle
    mSourceSkeleton->fitJointsToWorldPositions(
        std::vector<const dynamics::Joint*>(),
        Eigen::VectorXs::Zero(0),
        mSourceJoints,
        getTargetJointWorldAngles(),
        false,
        maxFitSteps / 2,
        true,
        log);
    // Just position
    return mSourceSkeleton->fitJointsToWorldPositions(
        mSourceJoints,
        getTargetJointWorldPositions(),
        std::vector<const dynamics::Joint*>(),
        Eigen::VectorXs::Zero(0),
        false,
        maxFitSteps / 2,
        true,
        log);
  }
}

//==============================================================================
/// This converts a motion from the target skeleton to the source skeleton
Eigen::MatrixXs SkeletonConverter::convertMotion(
    Eigen::MatrixXs targetMotion,
    bool logProgress,
    int maxFitStepsPerTimestep,
    s_t convergenceThreshold)
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
  fitTarget(maxFitStepsPerTimestep, convergenceThreshold);

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
    s_t bestError = fitTarget(maxFitStepsPerTimestep, convergenceThreshold);
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
          std::vector<const dynamics::Joint*>(),
          Eigen::VectorXs::Zero(0),
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

  Eigen::VectorXs sourceAngles = getSourceJointWorldAngles();
  Eigen::VectorXs targetAngles = getTargetJointWorldAngles();
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

    Eigen::Vector3s sourceEuler = math::matrixToEulerXYZ(
        math::expMapRot(sourceAngles.segment<3>(i * 3)));
    Eigen::Vector3s targetEuler = math::matrixToEulerXYZ(
        math::expMapRot(targetAngles.segment<3>(i * 3)));

    server->renderBasis(
        0.1,
        "SkeletonConverter_link_source_" + std::to_string(i),
        sourcePos,
        sourceEuler);

    server->renderBasis(
        0.05,
        "SkeletonConverter_link_target_" + std::to_string(i),
        targetPos,
        targetEuler);
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