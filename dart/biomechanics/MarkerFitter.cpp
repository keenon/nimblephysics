#include "dart/biomechanics/MarkerFitter.hpp"

#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/biomechanics/IKInitializer.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/MetaSkeleton.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/server/GUIStateMachine.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

namespace dart {

namespace biomechanics {

using namespace Ipopt;

//==============================================================================
/// This unflattens an input vector, given some information about the problm
MarkerFitterState::MarkerFitterState(
    const Eigen::VectorXs& flat,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    std::vector<dynamics::Joint*> joints,
    Eigen::MatrixXs jointCenters,
    Eigen::VectorXs jointWeights,
    Eigen::MatrixXs jointAxis,
    Eigen::VectorXs axisWeights,
    MarkerFitter* fitter)
  : markerOrder(fitter->mMarkerNames),
    skeleton(fitter->mSkeleton),
    markerObservations(markerObservations),
    joints(joints),
    jointCenters(jointCenters),
    jointWeights(jointWeights),
    jointAxis(jointAxis),
    axisWeights(axisWeights),
    fitter(fitter)
{
  for (auto joint : joints)
  {
    jointOrder.push_back(joint->getName());
  }

  // group scale
  int groupScaleDim = skeleton->getGroupScaleDim();
  // marker offsets
  int markerOffsetDim = fitter->mMarkers.size() * 3;
  // poses
  int posesDim = skeleton->getNumDofs() * markerObservations.size();
  // the root position for the static calibration pose
  int staticRootDim = 6;

  (void)posesDim;      // Don't break the compile when we strip out asserts
  (void)staticRootDim; // Don't break the compile when we strip out asserts
  assert(
      flat.size()
      == groupScaleDim + markerOffsetDim + posesDim + staticRootDim);

  /*
  std::map<std::string, Eigen::Vector3s> bodyScales;
  std::map<std::string, Eigen::Vector3s> markerOffsets;
  std::vector<std::string> markerOrder;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerErrorsAtTimesteps;
  std::vector<Eigen::VectorXs> posesAtTimesteps;
  */

  // Read the body scales

  Eigen::VectorXs originalScales = skeleton->getGroupScales();
  skeleton->setGroupScales(flat.segment(0, groupScaleDim));
  bodyScales = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
      3, skeleton->getNumBodyNodes());
  bodyScalesGrad = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
      3, skeleton->getNumBodyNodes());
  for (int i = 0; i < skeleton->getNumBodyNodes(); i++)
  {
    bodyNames.push_back(skeleton->getBodyNode(i)->getName());
    bodyScales.col(i) = skeleton->getBodyNode(i)->getScale();
  }

  // Read marker offsets

  markerOffsets
      = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, markerOrder.size());
  markerOffsetsGrad
      = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, markerOrder.size());
  for (int i = 0; i < markerOrder.size(); i++)
  {
    markerOffsets.col(i) = flat.segment<3>(groupScaleDim + i * 3);
  }

  // Read poses and marker errors

  Eigen::VectorXs originalPos = skeleton->getPositions();

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = fitter->setConfiguration(
          skeleton,
          flat.segment(groupScaleDim + markerOffsetDim, skeleton->getNumDofs()),
          flat.segment(0, groupScaleDim),
          flat.segment(groupScaleDim, markerOffsetDim));
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      markerMap;
  for (int i = 0; i < fitter->mMarkerNames.size(); i++)
  {
    markerMap[fitter->mMarkerNames[i]] = markers[i];
  }

  posesAtTimesteps = Eigen::MatrixXs::Zero(
      skeleton->getNumDofs(), markerObservations.size());
  posesAtTimestepsGrad = Eigen::MatrixXs::Zero(
      skeleton->getNumDofs(), markerObservations.size());

  markerErrorsAtTimesteps = Eigen::MatrixXs::Zero(
      markerObservations.size() * 3, markerOrder.size());
  markerErrorsAtTimestepsGrad = Eigen::MatrixXs::Zero(
      markerObservations.size() * 3, markerOrder.size());

  jointErrorsAtTimesteps
      = Eigen::MatrixXs::Zero(joints.size() * 3, markerObservations.size());
  jointErrorsAtTimestepsGrad
      = Eigen::MatrixXs::Zero(joints.size() * 3, markerObservations.size());

  axisErrorsAtTimesteps
      = Eigen::MatrixXs::Zero(joints.size() * 3, markerObservations.size());
  axisErrorsAtTimestepsGrad
      = Eigen::MatrixXs::Zero(joints.size() * 3, markerObservations.size());

  for (int i = 0; i < markerObservations.size(); i++)
  {
    Eigen::VectorXs pos = flat.segment(
        groupScaleDim + markerOffsetDim + (skeleton->getNumDofs() * i),
        skeleton->getNumDofs());
    posesAtTimesteps.col(i) = pos;

    // Compute marker errors at each timestep

    skeleton->setPositions(pos);
    std::map<std::string, Eigen::Vector3s> currentMarkerPoses
        = skeleton->getMarkerMapWorldPositions(markerMap);
    std::map<std::string, Eigen::Vector3s> desiredMarkerPoses
        = markerObservations[i];

    for (int j = 0; j < markerOrder.size(); j++)
    {
      std::string markerName = markerOrder[j];
      if (desiredMarkerPoses.count(markerName))
      {
        markerErrorsAtTimesteps.block<3, 1>(i * 3, j)
            = currentMarkerPoses[markerName] - desiredMarkerPoses[markerName];
      }
    }

    // Compute the joint errors at each timestep

    Eigen::VectorXs jointPoses = skeleton->getJointWorldPositions(joints);
    jointErrorsAtTimesteps.col(i) = jointPoses - jointCenters.col(i);
    if (jointWeights.size() > 0)
    {
      for (int j = 0; j < joints.size(); j++)
      {
        jointErrorsAtTimesteps.col(i).segment<3>(j * 3) *= jointWeights(j);
      }
    }

    // Compute the axis errors at each timestep

    if (jointAxis.size() > 0)
    {
      for (int j = 0; j < joints.size(); j++)
      {
        Eigen::Vector3s jointPos = jointPoses.segment<3>(j * 3);
        Eigen::Vector3s axisCenter = jointAxis.block<3, 1>(j * 6, i);
        Eigen::Vector3s axisDir
            = jointAxis.block<3, 1>(j * 6 + 3, i).normalized();

        Eigen::Vector3s diff = jointPos - axisCenter;
        // Subtract out the component of `diff` that's parallel to the axisDir
        diff -= diff.dot(axisDir) * axisDir;
        // Now our measured diff is only the distance perpendicular to axisDir
        // (ie the shortest path to the axis)
        axisErrorsAtTimesteps.block<3, 1>(j * 3, i) = diff;
        if (axisWeights.size() > 0)
        {
          axisErrorsAtTimesteps.block<3, 1>(j * 3, i) *= axisWeights(j);
        }
      }
    }
  }

  // Grab the root position for the static calibration pose

  staticPoseRoot = flat.segment<6>(
      groupScaleDim + markerOffsetDim
      + (skeleton->getNumDofs() * posesAtTimesteps.cols()));
  staticPoseRootGrad = Eigen::Vector6s::Zero();

  skeleton->setPositions(originalPos);
  skeleton->setGroupScales(originalScales);
}

//==============================================================================
/// This returns a single flat vector representing this whole problem state
Eigen::VectorXs MarkerFitterState::flattenState()
{
  // group scale
  int groupScaleDim = skeleton->getGroupScaleDim();
  // marker offsets
  int markerOffsetDim = markerOrder.size() * 3;
  // poses
  int posesDim = skeleton->getNumDofs() * posesAtTimesteps.cols();
  // the root position for the static calibration pose
  int staticRootDim = 6;

  Eigen::VectorXs flat = Eigen::VectorXs::Zero(
      groupScaleDim + markerOffsetDim + posesDim + staticRootDim);

  // Collapse body scales into group scales

  Eigen::VectorXs originalScales = skeleton->getGroupScales();
  for (int i = 0; i < skeleton->getNumBodyNodes(); i++)
  {
    skeleton->getBodyNode(i)->setScale(bodyScales.col(i));
  }
  flat.segment(0, groupScaleDim) = skeleton->getGroupScales();
  skeleton->setGroupScales(originalScales);

  // Write marker offsets

  for (int i = 0; i < markerOrder.size(); i++)
  {
    flat.segment<3>(groupScaleDim + i * 3) = markerOffsets.col(i);
  }

  // Write poses

  for (int i = 0; i < posesAtTimesteps.cols(); i++)
  {
    flat.segment(
        groupScaleDim + markerOffsetDim + (skeleton->getNumDofs() * i),
        skeleton->getNumDofs())
        = posesAtTimesteps.col(i);
  }

  // Write the static calibration pose root position

  flat.segment<6>(
      groupScaleDim + markerOffsetDim
      + (skeleton->getNumDofs() * posesAtTimesteps.cols()))
      = staticPoseRoot;

  return flat;
}

//==============================================================================
/// This returns a single flat vector representing the gradient of this whole
/// problem state
Eigen::VectorXs MarkerFitterState::flattenGradient()
{
  // group scale
  int groupScaleDim = skeleton->getGroupScaleDim();
  // marker offsets
  int markerOffsetDim = markerOrder.size() * 3;
  // poses
  int posesDim = skeleton->getNumDofs() * posesAtTimesteps.cols();
  // the root position for the static calibration pose
  int staticRootDim = 6;

  // 1. Write scale grad

  Eigen::VectorXs grad = Eigen::VectorXs::Zero(
      groupScaleDim + markerOffsetDim + posesDim + staticRootDim);

  std::map<std::string, Eigen::Vector3s> bodyScalesGradMap;
  for (int i = 0; i < skeleton->getNumBodyNodes(); i++)
  {
    bodyScalesGradMap[skeleton->getBodyNode(i)->getName()]
        = bodyScalesGrad.col(i);
  }
  grad.segment(0, groupScaleDim)
      = skeleton->getGroupScaleGradientsFromMap(bodyScalesGradMap);

  // 2. Write marker offsets grad

  for (int i = 0; i < markerOrder.size(); i++)
  {
    grad.segment<3>(groupScaleDim + (i * 3)) = markerOffsetsGrad.col(i);
  }

  // 3. Write poses grad

  for (int i = 0; i < posesAtTimesteps.cols(); i++)
  {
    grad.segment(
        groupScaleDim + markerOffsetDim + (skeleton->getNumDofs() * i),
        skeleton->getNumDofs())
        = posesAtTimestepsGrad.col(i);
  }

  // 4. Incorporate marker and joint error grads

  // 4.1. Recover original skeleton and marker state

  Eigen::VectorXs originalPos = skeleton->getPositions();
  Eigen::VectorXs originalScales = skeleton->getGroupScales();

  for (int i = 0; i < skeleton->getNumBodyNodes(); i++)
  {
    skeleton->getBodyNode(i)->setScale(bodyScales.col(i));
  }
  Eigen::VectorXs groupScales = skeleton->getGroupScales();
  Eigen::VectorXs markerOffsetsFlat
      = Eigen::VectorXs::Zero(markerOrder.size() * 3);
  for (int i = 0; i < markerOrder.size(); i++)
  {
    markerOffsetsFlat.segment<3>(i * 3) = markerOffsets.col(i);
  }
  Eigen::VectorXs firstPose = posesAtTimesteps.col(0);

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = fitter->setConfiguration(
          skeleton, firstPose, groupScales, markerOffsetsFlat);

  // 4.2. Go through each observation and accumulate gradient where appropriate

  for (int i = 0; i < markerObservations.size(); i++)
  {
    int offset = groupScaleDim + markerOffsetDim + (i * skeleton->getNumDofs());
    Eigen::VectorXs pose = posesAtTimesteps.col(i);
    skeleton->setPositions(pose);

    Eigen::VectorXs markerErrorGrad
        = Eigen::VectorXs::Zero(markerOrder.size() * 3);
    for (int j = 0; j < markerOrder.size(); j++)
    {
      markerErrorGrad.segment<3>(j * 3)
          = markerErrorsAtTimestepsGrad.block<3, 1>(3 * i, j);
    }
    Eigen::VectorXs jointErrorGrad = jointErrorsAtTimestepsGrad.col(i);
    Eigen::VectorXs axisErrorGrad = axisErrorsAtTimestepsGrad.col(i);
    Eigen::VectorXs combinedJointGrad = jointErrorGrad + axisErrorGrad;

    // Get loss wrt joint positions
    grad.segment(offset, skeleton->getNumDofs())
        += fitter->getMarkerLossGradientWrtJoints(
            skeleton, markers, markerErrorGrad);
    grad.segment(offset, skeleton->getNumDofs())
        += skeleton->getJointWorldPositionsJacobianWrtJointPositions(joints)
               .transpose()
           * combinedJointGrad;

    // Acculumulate loss wrt the global scale groups
    grad.segment(0, groupScaleDim)
        += fitter->getMarkerLossGradientWrtGroupScales(
            skeleton, markers, markerErrorGrad);
    grad.segment(0, groupScaleDim)
        += skeleton->getJointWorldPositionsJacobianWrtGroupScales(joints)
               .transpose()
           * combinedJointGrad;

    // Acculumulate loss wrt the global marker offsets (this is 0 for joints,
    // since marker offsets don't change joint locations)
    grad.segment(groupScaleDim, markerOffsetDim)
        += fitter->getMarkerLossGradientWrtMarkerOffsets(
            skeleton, markers, markerErrorGrad);
  }

  skeleton->setGroupScales(originalScales);
  skeleton->setPositions(originalPos);

  // Write the static calibration pose root position

  grad.segment<6>(
      groupScaleDim + markerOffsetDim
      + (skeleton->getNumDofs() * posesAtTimesteps.cols()))
      = staticPoseRootGrad;

  return grad;
}

//==============================================================================
BilevelFitResult::BilevelFitResult() : success(false){};

//==============================================================================
InitialMarkerFitParams::InitialMarkerFitParams()
  : numBlocks(12),
    numIKTries(12),
    dontRescaleBodies(false),
    dontMoveMarkers(false),
    maxTrialsToUseForMultiTrialScaling(5),
    maxTimestepsToUseForMultiTrialScaling(800),
    initPoses(Eigen::MatrixXs::Zero(0, 0)),
    groupScales(Eigen::VectorXs::Zero(0)),
    jointCenters(Eigen::MatrixXs::Zero(0, 0)),
    jointWeights(Eigen::VectorXs::Zero(0)),
    jointAxis(Eigen::MatrixXs::Zero(0, 0)),
    axisWeights(Eigen::VectorXs::Zero(0))
{
}

//==============================================================================
InitialMarkerFitParams::InitialMarkerFitParams(
    const InitialMarkerFitParams& other)
  : markerWeights(other.markerWeights),
    joints(other.joints),
    jointCenters(other.jointCenters),
    jointWeights(other.jointWeights),
    jointAxis(other.jointAxis),
    jointAdjacentMarkers(other.jointAdjacentMarkers),
    axisWeights(other.axisWeights),
    numBlocks(other.numBlocks),
    numIKTries(other.numIKTries),
    initPoses(other.initPoses),
    markerOffsets(other.markerOffsets),
    groupScales(other.groupScales),
    dontRescaleBodies(other.dontRescaleBodies),
    dontMoveMarkers(other.dontMoveMarkers),
    maxTrialsToUseForMultiTrialScaling(
        other.maxTrialsToUseForMultiTrialScaling),
    maxTimestepsToUseForMultiTrialScaling(
        other.maxTimestepsToUseForMultiTrialScaling)
{
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setMarkerWeights(
    std::map<std::string, s_t> markerWeights)
{
  this->markerWeights = markerWeights;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setJointCenters(
    std::vector<dynamics::Joint*> joints,
    Eigen::MatrixXs jointCenters,
    std::vector<std::vector<std::string>> jointAdjacentMarkers)
{
  this->joints = joints;
  this->jointCenters = jointCenters;
  this->jointAdjacentMarkers = jointAdjacentMarkers;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setJointCentersAndWeights(
    std::vector<dynamics::Joint*> joints,
    Eigen::MatrixXs jointCenters,
    std::vector<std::vector<std::string>> jointAdjacentMarkers,
    Eigen::VectorXs jointWeights)
{
  this->joints = joints;
  this->jointCenters = jointCenters;
  this->jointWeights = jointWeights;
  this->jointAdjacentMarkers = jointAdjacentMarkers;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setJointAxisAndWeights(
    Eigen::MatrixXs jointAxis, Eigen::VectorXs axisWeights)
{
  this->jointAxis = jointAxis;
  this->axisWeights = axisWeights;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setNumBlocks(int numBlocks)
{
  this->numBlocks = numBlocks;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setNumIKTries(int retries)
{
  this->numIKTries = retries;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setInitPoses(
    Eigen::MatrixXs initPoses)
{
  this->initPoses = initPoses;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setDontRescaleBodies(
    bool dontRescaleBodies)
{
  this->dontRescaleBodies = dontRescaleBodies;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setDontMoveMarkers(
    bool dontMoveMarkers)
{
  this->dontMoveMarkers = dontMoveMarkers;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setMarkerOffsets(
    std::map<std::string, Eigen::Vector3s> markerOffsets)
{
  this->markerOffsets = markerOffsets;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setGroupScales(
    Eigen::VectorXs groupScales)
{
  this->groupScales = groupScales;
  return *this;
}

//==============================================================================
InitialMarkerFitParams&
InitialMarkerFitParams::setMaxTrialsToUseForMultiTrialScaling(int numTrials)
{
  this->maxTrialsToUseForMultiTrialScaling = numTrials;
  return *this;
}

//==============================================================================
InitialMarkerFitParams&
InitialMarkerFitParams::setMaxTimestepsToUseForMultiTrialScaling(
    int numTimesteps)
{
  this->maxTimestepsToUseForMultiTrialScaling = numTimesteps;
  return *this;
}

//==============================================================================
MarkerFitter::MarkerFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    dynamics::MarkerMap markers,
    bool ignoreVirtualJointCenterMarkers)
  : mSkeleton(skeleton),
    mAnthropometrics(nullptr),
    mAnthropometricWeight(0.001),
    mHeightPrior(1.68),
    mHeightPriorWeight(0.0),
    mInitialIKSatisfactoryLoss(0.003),
    mInitialIKMaxRestarts(100),
    mDebugLoss(false),
    mIgnoreJointLimits(false),
    mMaxMarkerOffset(0.2),
    mMinVarianceCutoff(3.0),
    mMinSphereFitScore(0.01),
    mMinAxisFitScore(0.001),
    // mMinSphereFitScore(6e-5),
    // mMinAxisFitScore(1.2e-4),
    mMaxJointWeight(1.0),
    mMaxAxisWeight(0.5),
    mDebugJointVariability(false),
    mRegularizeTrackingMarkerOffsets(0.05),
    mRegularizeAnatomicalMarkerOffsets(10.0),
    mRegularizeIndividualBodyScales(10.0),
    mRegularizeAllBodyScales(0.2),
    mRegularizeJointBounds(0),
    mTolerance(1e-8),
    mIterationLimit(500),
    mLBFGSHistoryLength(8),
    mJointSphereFitSGDIterations(5000),
    mJointAxisFitSGDIterations(10000),
    mCheckDerivatives(false),
    mUseParallelIKWarps(false),
    mPrintFrequency(1),
    mSilenceOutput(false),
    mDisableLinesearch(false),
    mAnatomicalMarkerDefaultWeight(1.0),
    mTrackingMarkerDefaultWeight(0.2),
    mStaticTrialWeight(50.0),
    mJointForceFieldThresholdDistance(0.0),
    mJointForceFieldSoftness(0.2),
    // mJointForceFieldSoftness(20.0),
    mStaticTrialEnabled(false),
    mPostprocessTrackingMarkerOffsets(true),
    mPostprocessAnatomicalMarkerOffsets(false)
{
  mImuMap = std::
      map<std::string, std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>();
  mSkeletonBallJoints = mSkeleton->convertSkeletonToBallJoints();

  // Pre-filter the markers to get rid of artificial joint centers
  dynamics::MarkerMap filteredMarkers;
  for (auto pair : markers)
  {
    bool isOnJoint = false;

    if (ignoreVirtualJointCenterMarkers)
    {
      auto* bodyNode = pair.second.first;

      if ((pair.second.second
           - bodyNode->getParentJoint()
                 ->getTransformFromChildBodyNode()
                 .translation())
              .squaredNorm()
          < 1e-9)
      {
        isOnJoint = true;
      };
      for (int i = 0; i < bodyNode->getNumChildJoints(); i++)
      {
        if ((pair.second.second
             - bodyNode->getChildJoint(i)
                   ->getTransformFromParentBodyNode()
                   .translation())
                .squaredNorm()
            < 1e-9)
        {
          isOnJoint = true;
          break;
        }
      }
    }

    if (!isOnJoint)
    {
      filteredMarkers.emplace(pair);
    }
  }

  int offset = 0;
  for (auto pair : filteredMarkers)
  {
    mMarkerIndices[pair.first] = offset;
    mMarkerNames.push_back(pair.first);
    mMarkerIsTracking.push_back(false);
    offset++;
    mMarkers.push_back(pair.second);
    mMarkersBallJoints.emplace_back(
        mSkeletonBallJoints->getBodyNode(pair.second.first->getName()),
        Eigen::Vector3s(pair.second.second));
  }
  mMarkerMap = filteredMarkers;

  // Default to a least-squares loss over just the marker errors
  mLossAndGrad = [this](MarkerFitterState* state) {
    int numTimesteps = state->posesAtTimesteps.cols();

    const s_t markerErrors = state->markerErrorsAtTimesteps.squaredNorm();
    const s_t jointErrors = state->jointErrorsAtTimesteps.squaredNorm();
    const s_t axisErrors = state->axisErrorsAtTimesteps.squaredNorm();

    // 1. Compute loss as a simple squared norm of marker and joint errors
    s_t loss = markerErrors + jointErrors + axisErrors;
    // 2. Compute the gradient of squared norm
    state->markerErrorsAtTimestepsGrad = 2 * state->markerErrorsAtTimesteps;
    state->jointErrorsAtTimestepsGrad = 2 * state->jointErrorsAtTimesteps;
    state->axisErrorsAtTimestepsGrad = 2 * state->axisErrorsAtTimesteps;

    // Regularize tracking vs anatomical differently
    state->markerOffsetsGrad = 2 * numTimesteps * state->markerOffsets;
    s_t markerRegularization = 0.0;
    for (int i = 0; i < this->mMarkerIsTracking.size(); i++)
    {
      s_t multiple
          = (this->mMarkerIsTracking[i]
                 ? this->mRegularizeTrackingMarkerOffsets
                 : this->mRegularizeAnatomicalMarkerOffsets);
      markerRegularization += numTimesteps
                              * state->markerOffsets.col(i).squaredNorm()
                              * multiple;
      state->markerOffsetsGrad.col(i) *= multiple;
    }
    loss += markerRegularization;

    state->bodyScalesGrad.setZero();

    // 3. If we've got an anthropometrics prior, use it
    s_t anthroRegularization = 0.0;
    if (this->mAnthropometrics)
    {
      Eigen::VectorXs oldBodyScales = this->mSkeleton->getBodyScales();
      // 3.1. Translate body scales from matrix form, which is how they show up
      // on the state object
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        this->mSkeleton->getBodyNode(i)->setScale(state->bodyScales.col(i));
      }
      // 3.2. Actually compute the loss
      anthroRegularization -= this->mAnthropometrics->getLogPDF(this->mSkeleton)
                              * this->mAnthropometricWeight;
      // 3.3. Translate gradients from vector back to matrix form for the state
      // object
      Eigen::VectorXs bodyScalesGradVector
          = this->mAnthropometrics->getGradientOfLogPDFWrtBodyScales(
                this->mSkeleton)
            * (-1 * this->mAnthropometricWeight);
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        state->bodyScalesGrad.col(i) += bodyScalesGradVector.segment<3>(i * 3);
      }
      // 3.4. Reset
      this->mSkeleton->setBodyScales(oldBodyScales);
    }
    loss += anthroRegularization;

    // 4. Regularize the height of the body, if we've got a height prior
    s_t heightRegularization = 0.0;
    if (mHeightPriorWeight > 0.0)
    {
      Eigen::VectorXs pose
          = Eigen::VectorXs::Zero(this->mSkeleton->getNumDofs());
      s_t height = this->mSkeleton->getHeight(pose);
      Eigen::VectorXs heightGrad
          = this->mSkeleton->getGradientOfHeightWrtBodyScales(pose);

      s_t diff = height - mHeightPrior;
      heightRegularization = mHeightPriorWeight * diff * diff;

      // 4.1. Translate gradients from vector back to matrix form for the state
      // object
      Eigen::VectorXs bodyScalesGradVector
          = 2 * mHeightPriorWeight * diff * heightGrad;
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        state->bodyScalesGrad.col(i) += bodyScalesGradVector.segment<3>(i * 3);
      }
    }
    loss += heightRegularization;

    // 4. Regularize the body sizes to try to be even in every direction
    s_t bodyEvenRegularization = 0.0;
    for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
    {
      Eigen::Vector3s scales = state->bodyScales.col(i);
      s_t avgScale = (scales(0) + scales(1) + scales(2)) / 3;
      Eigen::Vector3s diffVec = scales - (Eigen::Vector3s::Ones() * avgScale);
      bodyEvenRegularization
          += diffVec.squaredNorm() * mRegularizeIndividualBodyScales;

      s_t gradScale = (2.0 / 3) * mRegularizeIndividualBodyScales;
      state->bodyScalesGrad(0, i)
          += gradScale * (2 * scales(0) - scales(1) - scales(2));
      state->bodyScalesGrad(1, i)
          += gradScale * (2 * scales(1) - scales(0) - scales(2));
      state->bodyScalesGrad(2, i)
          += gradScale * (2 * scales(2) - scales(0) - scales(1));
    }
    loss += bodyEvenRegularization;

    // 5. Regularize all the body scales, to try to make them all match the
    // average
    Eigen::Vector3s avgScale = Eigen::Vector3s::Zero();
    for (int i = 0; i < state->bodyScales.cols(); i++)
    {
      avgScale += state->bodyScales.col(i);
    }
    s_t k = state->bodyScales.cols();
    avgScale /= k;

    // 6. If we're ignoring joint bounds, then at least provide a soft loss
    // incentive to stay inside the bounds
    s_t jointBoundsRegularization = 0.0;
    if (mRegularizeJointBounds > 0)
    {
      Eigen::VectorXs upperBound = mSkeleton->getPositionUpperLimits();
      Eigen::VectorXs lowerBound = mSkeleton->getPositionLowerLimits();
      for (int t = 0; t < state->posesAtTimesteps.cols(); t++)
      {
        for (int i = 0; i < upperBound.size(); i++)
        {
          s_t pos = state->posesAtTimesteps(i, t);
          if (pos > upperBound(i))
          {
            s_t diff = pos - upperBound(i);
            jointBoundsRegularization += mRegularizeJointBounds * diff * diff;
            state->posesAtTimestepsGrad(i, t)
                += 2 * mRegularizeJointBounds * diff;
          }
          if (pos < lowerBound(i))
          {
            s_t diff = lowerBound(i) - pos;
            jointBoundsRegularization += mRegularizeJointBounds * diff * diff;
            state->posesAtTimestepsGrad(i, t)
                -= 2 * mRegularizeJointBounds * diff;
          }
        }
      }
    }
    loss += jointBoundsRegularization;

    s_t virtualJointSpringRegularization = 0.0;
    for (auto& pair : this->mJointVirtualSpringRegularizerWeight)
    {
      auto* joint = this->mSkeleton->getJoint(pair.first);
      if (joint != nullptr)
      {
        for (int i = 0; i < joint->getNumDofs(); i++)
        {
          s_t weight = pair.second;
          int dof = joint->getDof(i)->getIndexInSkeleton();
          Eigen::VectorXs poses = state->posesAtTimesteps.row(dof);
          virtualJointSpringRegularization
              += poses.squaredNorm() * weight / poses.size();
          state->posesAtTimestepsGrad.row(dof)
              += (2 * weight / poses.size()) * poses;
        }
      }
    }
    loss += virtualJointSpringRegularization;

    // avg = (sum)/K
    // d/dx = (K-1)*2 / (K^2) * (K*x - sum)
    // d/dx = (K-1)*2 / (K) * (x - avg)

    // 2*(k-1)/(k*k) * (kx - x - q - y - z)
    // 2*(k-1)/k * (x - (x + q + y + z)/k)
    // 2*(k-1)/k*(x - avg)

    /*
    int pelvisIndex = mSkeleton->getBodyNode("pelvis")->getIndexInSkeleton();
    Eigen::Vector3s diffPelvis
        = state->bodyScales.col(pelvisIndex) - (Eigen::Vector3s::Ones());
    loss += 100 * diffPelvis.squaredNorm();
    state->bodyScalesGrad.col(pelvisIndex) += 100 * 2 * diffPelvis;
    */

    // s_t gradScale = (k - 1.0) * 2.0 / k;
    s_t bodyScaleReg = 0.0;
    for (int i = 0; i < state->bodyScales.cols(); i++)
    {
      Eigen::Vector3s diff = state->bodyScales.col(i) - avgScale;
      bodyScaleReg += mRegularizeAllBodyScales * diff.squaredNorm();

      // state->bodyScalesGrad.col(i) += 2.0 * ((k - 1) / k) * diff;
      // state->bodyScalesGrad.col(i) += 2.0 * diff;
      state->bodyScalesGrad.col(i) += mRegularizeAllBodyScales * 2.0 * diff;
      /*
      state->bodyScalesGrad.col(i)
          += gradScale * (state->bodyScales.col(i) - avgScale);
      */
    }
    loss += bodyScaleReg;

    // 7. If we've included a static pose, then we want to compute the loss at
    // the static pose for our current scaling and marker offsets.
    s_t staticPoseRegularization = 0.0;
    if (mStaticTrialEnabled)
    {
      Eigen::VectorXs oldBodyScales = this->mSkeleton->getBodyScales();
      Eigen::VectorXs oldPose = this->mSkeleton->getPositions();

      // 7.1. Set the skeleton to the static pose and the current scales in the
      // problem.
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        this->mSkeleton->getBodyNode(i)->setScale(state->bodyScales.col(i));
      }
      Eigen::VectorXs staticPose = mStaticTrialPose;
      staticPose.head<6>() = state->staticPoseRoot;
      this->mSkeleton->setPositions(staticPose);

      // 7.2. Construct the most up-to-date list of markers for the static trial
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
          staticTrialMarkers;
      std::vector<int> staticMarkerIndexes;
      for (int i = 0; i < mStaticTrialMarkerNames.size(); i++)
      {
        bool foundMarker = false;
        for (int j = 0; j < mMarkerNames.size(); j++)
        {
          if (mStaticTrialMarkerNames[i] == mMarkerNames[j])
          {
            staticMarkerIndexes.push_back(j);
            staticTrialMarkers.push_back(std::make_pair(
                mMarkers[j].first,
                mMarkers[j].second + state->markerOffsets.col(j)));
            foundMarker = true;
            break;
          }
        }
        assert(foundMarker);
        if (!foundMarker)
        {
          std::cout << "ERROR: static trial marker "
                    << mStaticTrialMarkerNames[i]
                    << " did not appear in the markerset. This is a runtime "
                       "error, because it should have been filtered out in "
                       "MarkerFitter::setStaticTrial(). Exit 1"
                    << std::endl;
          exit(1);
        }
      }

      // 7.3. Actually compute the loss, which is just over the marker positions
      // in the static pose.
      Eigen::VectorXs computedStaticMarkerPosition
          = this->mSkeleton->getMarkerWorldPositions(staticTrialMarkers);
      Eigen::VectorXs staticMarkerError
          = computedStaticMarkerPosition - this->mStaticTrialMarkerPositions;
      staticPoseRegularization
          = mStaticTrialWeight * staticMarkerError.squaredNorm();

      // 7.4. Translate gradients from vector back to matrix form for the state
      // object
      Eigen::VectorXs staticMarkerErrorGrad
          = mStaticTrialWeight * 2 * staticMarkerError;

      // 7.4.1. Body scales
      Eigen::VectorXs bodyScalesGradVector
          = this->mSkeleton
                ->getMarkerWorldPositionsJacobianWrtBodyScales(
                    staticTrialMarkers)
                .transpose()
            * staticMarkerErrorGrad;
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        state->bodyScalesGrad.col(i) += bodyScalesGradVector.segment<3>(i * 3);
      }
      // 7.4.2. Marker offsets
      Eigen::VectorXs markerOffsetsGradVector
          = this->mSkeleton
                ->getMarkerWorldPositionsJacobianWrtMarkerOffsets(
                    staticTrialMarkers)
                .transpose()
            * staticMarkerErrorGrad;
      for (int i = 0; i < staticMarkerIndexes.size(); i++)
      {
        state->markerOffsetsGrad.col(staticMarkerIndexes[i])
            += markerOffsetsGradVector.segment<3>(i * 3);
      }
      // 7.4.3. Root translation
      Eigen::VectorXs jointGradVector
          = this->mSkeleton
                ->getMarkerWorldPositionsJacobianWrtJointPositions(
                    staticTrialMarkers)
                .transpose()
            * staticMarkerErrorGrad;
      state->staticPoseRootGrad = jointGradVector.head<6>();

      // 7.5. Reset
      this->mSkeleton->setPositions(oldPose);
      this->mSkeleton->setBodyScales(oldBodyScales);
    }
    else
    {
      state->staticPoseRootGrad.setZero();
    }
    loss += staticPoseRegularization;

    // 8. Run the force field loss
    s_t jointForceFieldLoss = 0.0;
    if (mJointForceFieldThresholdDistance > 0)
    {
      Eigen::VectorXs oldBodyScales = this->mSkeleton->getBodyScales();
      Eigen::VectorXs oldPose = this->mSkeleton->getPositions();

      // 8.1. Set the skeleton to the current scales in the problem.
      for (int i = 0; i < this->mSkeleton->getNumBodyNodes(); i++)
      {
        this->mSkeleton->getBodyNode(i)->setScale(state->bodyScales.col(i));
      }

      // 8.2. Collect joint force field loss at each timestep.
      for (int t = 0; t < state->posesAtTimesteps.cols(); t++)
      {
        mSkeleton->setPositions(state->posesAtTimesteps.col(t));

        for (int i = 0; i < mSkeleton->getNumJoints(); i++)
        {
          jointForceFieldLoss += mSkeleton->getJointForceFieldToOtherJoints(
              mSkeleton->getJoints(),
              i,
              mJointForceFieldThresholdDistance,
              mJointForceFieldSoftness);
          state->posesAtTimestepsGrad.col(t)
              += mSkeleton->getJointForceFieldToOtherJointsGradient(
                  mSkeleton->getJoints(),
                  i,
                  mJointForceFieldThresholdDistance,
                  mJointForceFieldSoftness,
                  neural::WithRespectTo::POSITION);

          Eigen::VectorXs bodyScalesGrad
              = mSkeleton->getJointForceFieldToOtherJointsGradientWrtBodyScales(
                  mSkeleton->getJoints(),
                  i,
                  mJointForceFieldThresholdDistance,
                  mJointForceFieldSoftness);
          for (int b = 0; b < mSkeleton->getNumBodyNodes(); b++)
          {
            state->bodyScalesGrad.col(b) += bodyScalesGrad.segment<3>(b * 3);
          }
        }
      }

      // 8.3. Reset
      this->mSkeleton->setPositions(oldPose);
      this->mSkeleton->setBodyScales(oldBodyScales);
    }
    loss += jointForceFieldLoss;

    if (mDebugLoss)
    {
      std::cout << "[mkr=" << markerErrors << ",jnt=" << jointErrors
                << ",axs=" << axisErrors << ",mkrR=" << markerRegularization
                << ",anthR=" << anthroRegularization
                << ",heightR=" << heightRegularization
                << ",bodyEvenR=" << bodyEvenRegularization
                << ",bodyR=" << bodyScaleReg
                << ",boundsR=" << jointBoundsRegularization
                << ",jointSpringR=" << virtualJointSpringRegularization
                << ",staticR=" << staticPoseRegularization
                << ",jointForceField=" << jointForceFieldLoss << "]"
                << std::endl;
    }

    return loss;
  };
}

//==============================================================================
/// Returns the skeleton pointer we're fitting against
std::shared_ptr<dynamics::Skeleton> MarkerFitter::getSkeleton()
{
  return mSkeleton;
}

//==============================================================================
MarkerInitialization::MarkerInitialization()
  : staticPoseRoot(Eigen::Vector6s::Zero())
{
}

//==============================================================================
/// This just checks if there are enough markers in the data with the names
/// expected by the model. Returns true if there are enough, and false
/// otherwise.
bool MarkerFitter::checkForEnoughMarkers(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // Generate a list of the markers we observe in this clip

  std::map<std::string, int> observedMarkersMap;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    for (auto& pair : markerObservations[i])
    {
      observedMarkersMap[pair.first] = 1;
    }
  }
  std::vector<std::string> intersectionMarkers;
  for (auto& pair : observedMarkersMap)
  {
    if (mMarkerMap.count(pair.first))
    {
      intersectionMarkers.push_back(pair.first);
    }
  }

  if (intersectionMarkers.size() < 8)
  {
    return false;
  }
  return true;
}

//==============================================================================
/// This will go through original marker data and attempt to detect common
/// anomalies, generate warnings to help the user fix their own issues, and
/// produce fixes where possible.
std::shared_ptr<MarkersErrorReport> MarkerFitter::generateDataErrorsReport(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        immutableMarkerObservations,
    s_t dt)
{
  std::shared_ptr<MarkersErrorReport> report
      = MarkerFixer::generateDataErrorsReport(immutableMarkerObservations, dt);

  // 1. Generate a list of the markers we observe in this clip

  std::map<std::string, int> observedMarkersMap;
  for (int i = 0; i < immutableMarkerObservations.size(); i++)
  {
    for (auto& pair : immutableMarkerObservations[i])
    {
      observedMarkersMap[pair.first] = 1;
    }
  }
  std::vector<std::string> observedMarkers;
  for (auto& pair : observedMarkersMap)
  {
    observedMarkers.push_back(pair.first);
  }

  // 2. Generate a warning if there aren't enough fixed markers on the pelvis
  bool hasPelvis = false;
  for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
  {
    std::string name = mSkeleton->getBodyNode(i)->getName();
    if (name == "pelvis" || name == "Pelvis" || name == "PELVIS")
    {
      hasPelvis = true;
      break;
    }
  }
  if (hasPelvis)
  {
    int numFixedOnPelvis = 0;
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      if (mMarkers[i].first->getName() == "pelvis"
          || mMarkers[i].first->getName() == "Pelvis"
          || mMarkers[i].first->getName() == "PELVIS")
      {
        if (!mMarkerIsTracking[i] && observedMarkersMap.count(mMarkerNames[i]))
        {
          numFixedOnPelvis++;
        }
      }
    }
    if (numFixedOnPelvis < 3)
    {
      report->warnings.push_back(
          "Need more fixed markers on the pelvis! This trial only has "
          + std::to_string(numFixedOnPelvis)
          + ". We recommend 3 (or more) for good results.");
    }
  }

  // 3. Generate a warning about mismatched markers between the model and the
  // data.

  // The only reason this could be a problem is if the overlap is too small
  std::vector<std::string> markersInBoth;
  // This could be suspicious, and should generate a warning
  std::vector<std::string> markersInJustMocap;
  // These are not a big deal, and generally indicate virtual markers
  std::vector<std::string> markersInJustModel;

  for (std::string& marker : observedMarkers)
  {
    if (mMarkerMap.count(marker) > 0)
    {
      markersInBoth.push_back(marker);
    }
    else
    {
      markersInJustMocap.push_back(marker);
      report->info.push_back(
          "The marker \"" + marker
          + "\" appears in the mocap data, but not on the model");
    }
  }
  for (std::string& marker : mMarkerNames)
  {
    if (observedMarkersMap.count(marker) == 0)
    {
      markersInJustModel.push_back(marker);
      report->info.push_back("The marker \""+marker+"\" appears in the model, but not in the mocap data. This generally happens because the marker is a virtual marker. We ignore virtual markers, so this is not a problem, just a heads up.");
    }
  }

  if (markersInBoth.size() < 8)
  {
    report->warnings.push_back("We only found " + std::to_string(markersInBoth.size()) + " model markers that also appear in the motion capture! This will probably lead to suboptimal results, and likely means this file doesn't match the model you uploaded!");
  }

  return report;
}

//==============================================================================
/// After we've finished our initialization, it may become clear that markers
/// in some of the files should be reversed. This method will do that check,
/// and if it finds that the markers should be reversed it does the swap,
/// re-runs IK, and records the result in a new init. If it doesn't find
/// anything, this is a no-op.
bool MarkerFitter::checkForFlippedMarkers(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        immutableMarkerObservations,
    MarkerInitialization& init,
    std::shared_ptr<MarkersErrorReport> report)
{
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations
      = immutableMarkerObservations;

  std::cout << "Checking for marker swaps..." << std::endl;

  // 0. Generate a list of the markers we observe in this clip

  std::map<std::string, int> observedMarkersMap;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    for (auto& pair : markerObservations[i])
    {
      observedMarkersMap[pair.first] = 1;
    }
  }
  std::vector<std::string> observedMarkers;
  for (auto& pair : observedMarkersMap)
  {
    observedMarkers.push_back(pair.first);
  }

  // Attempt to detect markers that were mistakenly assigned to each other
  // during labelling. This is tricky, but the motivating example is that in
  // half of Scott Uhlrich's OpenCap trials the foot markers are on one side of
  // the foot, and in the other half of the trials the foot markers are flipped.

  Eigen::VectorXs originalPose = mSkeleton->getPositions();
  Eigen::VectorXs originalScales = mSkeleton->getGroupScales();
  std::map<std::string, std::map<std::string, s_t>> totalDistances;
  std::map<std::string, std::map<std::string, int>> totalObservations;
  for (std::string& marker : mMarkerNames)
  {
    for (std::string& innerMarker : observedMarkers)
    {
      totalDistances[marker][innerMarker] = 0.0;
    }
  }

  mSkeleton->setGroupScales(init.groupScales);
  for (int i = 0; i < init.poses.cols(); i++)
  {
    mSkeleton->setPositions(init.poses.col(i));
    std::map<std::string, Eigen::Vector3s> markerPos
        = mSkeleton->getMarkerMapWorldPositions(init.updatedMarkerMap);
    for (auto& pair : markerPos)
    {
      for (auto& innerPair : markerObservations[i])
      {
        totalDistances[pair.first][innerPair.first]
            += (pair.second - innerPair.second).norm();
        totalObservations[pair.first][innerPair.first]++;
      }
    }
  }
  mSkeleton->setPositions(originalPose);
  mSkeleton->setGroupScales(originalScales);

  std::map<std::string, std::string> closestMarkers;
  for (std::string& marker : mMarkerNames)
  {
    std::string closestMarker = marker;
    s_t closestMarkerDistance = totalDistances[marker][marker]
                                / (totalObservations[marker][marker] > 0
                                       ? totalObservations[marker][marker]
                                       : 1.0);

    for (std::string& innerMarker : observedMarkers)
    {
      int observed = totalObservations[marker][innerMarker];
      if (observed > 0)
      {
        s_t avgDist = totalDistances[marker][innerMarker] / observed;
        if (avgDist < closestMarkerDistance)
        {
          closestMarker = innerMarker;
          closestMarkerDistance = avgDist;
        }
      }
    }

    closestMarkers[marker] = closestMarker;
  }

  std::map<std::string, std::string> swapped;
  bool anySwapped = false;

  for (std::string& marker : mMarkerNames)
  {
    // If we're not closest to ourselves, and the thing we're closest to is
    // closest to us, then this is a good indication that we've found a marker
    // that shouldn't have been swapped! We also check that it hasn't already
    // been swapped with us, cause we don't want to double-swap and end up
    // swapping back.
    if (closestMarkers[marker] != marker
        && closestMarkers[closestMarkers[marker]] == marker
        && swapped.count(closestMarkers[marker]) == 0)
    {
      report->warnings.push_back(
          "Marker \"" + marker + "\" seems it was swapped with \""
          + closestMarkers[marker] + "\" for the whole clip.");
      swapped[marker] = closestMarkers[marker];
      swapped[closestMarkers[marker]] = marker;
      anySwapped = true;

      for (int i = 0; i < markerObservations.size(); i++)
      {
        if (markerObservations[i].count(marker) > 0
            && markerObservations[i].count(closestMarkers[marker]))
        {
          Eigen::Vector3s tmp
              = markerObservations[i].at(closestMarkers[marker]);
          markerObservations[i][closestMarkers[marker]]
              = markerObservations[i].at(marker);
          markerObservations[i][marker] = tmp;
        }
        else if (markerObservations[i].count(marker) > 0)
        {
          markerObservations[i][closestMarkers[marker]]
              = markerObservations[i].at(marker);
          markerObservations[i].erase(marker);
        }
        else if (markerObservations[i].count(closestMarkers[marker]) > 0)
        {
          markerObservations[i][marker]
              = markerObservations[i].at(closestMarkers[marker]);
          markerObservations[i].erase(closestMarkers[marker]);
        }
      }
    }
  }
  if (anySwapped)
  {
    report->markerObservationsAttemptedFixed = markerObservations;
  }

  return anySwapped;
}

//==============================================================================
/// Run the whole pipeline of optimization problems to fit the data as closely
/// as we can, working on multiple trials at once
std::vector<MarkerInitialization> MarkerFitter::runMultiTrialKinematicsPipeline(
    const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
        markerObservationTrials,
    InitialMarkerFitParams params,
    int numSamples)
{
  // 1. Check if we need to downsample the input for performance reasons
  bool needToDownsample = false;
  if (markerObservationTrials.size()
      > params.maxTrialsToUseForMultiTrialScaling)
  {
    needToDownsample = true;
  }
  else
  {
    int numTimesteps = 0;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      numTimesteps += markerObservationTrials[i].size();
    }
    if (numTimesteps > params.maxTimestepsToUseForMultiTrialScaling)
    {
      needToDownsample = true;
    }
  }

  // 2. Now we have completely separate codepaths depending on whether we're
  // downsampling or not
  if (needToDownsample)
  {
    // 3. Sort the trials by the amount of joint variability in each one

    // 3.1. First, get the joint inits for all the trials, since we need to be
    // able to sort clips by joint variability
    std::vector<s_t> markerVariabilities;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      std::cout << "Running joint init pipeline for trial " << i << "/"
                << markerObservationTrials.size() << std::endl;
      markerVariabilities.push_back(
          computeMarkerDistanceMatrixVariability(markerObservationTrials[i]));
    }

    // 3.2. Sort the trials by the amount of joint variability in each one
    std::vector<int> orderedByJointVariability;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      orderedByJointVariability.push_back(i);
    }
    std::sort(
        orderedByJointVariability.begin(),
        orderedByJointVariability.end(),
        [&](int a, int b) {
          // Sort by joint marker variability
          return markerVariabilities[a] > markerVariabilities[b];
        });
    std::vector<int> inverseOrderedByJointVariability;
    inverseOrderedByJointVariability.resize(orderedByJointVariability.size());
    for (int i = 0; i < orderedByJointVariability.size(); i++)
    {
      inverseOrderedByJointVariability[orderedByJointVariability[i]] = i;
    }

    // 3.3. Debug the sorted trials variability
    std::cout << "Sorted trials by the variation of markers around the joints:"
              << std::endl;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      std::cout << std::to_string(i) << ": " << orderedByJointVariability[i]
                << " marker variability norm = "
                << markerVariabilities[orderedByJointVariability[i]]
                << std::endl;
    }

    // 4. We'll subsample the data down here, to avoid having any accidental
    // performance tanking if people upload 100 trials for one subject

    // Sample at most N trials
    int numTrialsToSample = markerObservationTrials.size();
    if (numTrialsToSample > params.maxTrialsToUseForMultiTrialScaling)
    {
      numTrialsToSample = params.maxTrialsToUseForMultiTrialScaling;
    }

    std::vector<int> trialSampledAtIndex;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      trialSampledAtIndex.push_back(-1);
    }

    int cursor = 0;
    std::cout << "Downsampling the input for scaling performance!" << std::endl;
    std::cout << "Sampling " << numTrialsToSample << "/"
              << markerObservationTrials.size() << std::endl;
    for (int i = 0; i < numTrialsToSample; i++)
    {
      std::cout << "Trial " << orderedByJointVariability[i] << " length "
                << markerObservationTrials[orderedByJointVariability[i]].size()
                << std::endl;
      trialSampledAtIndex[orderedByJointVariability[i]] = cursor;
      cursor += markerObservationTrials[orderedByJointVariability[i]].size();
    }
    std::cout << "Total timesteps to use for scaling: " << cursor << std::endl;

    // 5. Construct a merged dataset, including the merged joint and axis data
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
    std::vector<bool> newClip;
    for (int i = 0; i < numTrialsToSample; i++)
    {
      for (int j = 0;
           j < markerObservationTrials[orderedByJointVariability[i]].size();
           j++)
      {
        markerObservations.emplace_back(
            markerObservationTrials[orderedByJointVariability[i]][j]);
        newClip.push_back(j == 0);
      }
    }

    // 6. Run the kinematics pipeline on the merged dataset
    MarkerInitialization overallInit = runKinematicsPipeline(
        markerObservations, newClip, params, numSamples, false);

    std::cout << "Done scaling and computing marker offsets! Now we'll do IK "
                 "on all the trials."
              << std::endl;

    // 7. Use the scaling from overallInit to do IK on each skeleton
    std::vector<MarkerInitialization> separateInits;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      std::cout << "## IK on trial " << i << "/"
                << markerObservationTrials.size() << std::endl;

      if (trialSampledAtIndex[i] != -1)
      {
        int cursor = trialSampledAtIndex[i];
        int size = markerObservationTrials[i].size();

        MarkerInitialization result;
        result.groupScales = overallInit.groupScales;
        result.markerOffsets = overallInit.markerOffsets;
        result.updatedMarkerMap = overallInit.updatedMarkerMap;
        result.joints = overallInit.joints;
        result.jointMarkerVariability = overallInit.jointMarkerVariability;
        result.jointsAdjacentMarkers = overallInit.jointsAdjacentMarkers;
        result.jointLoss = overallInit.jointLoss;
        result.jointWeights = overallInit.jointWeights;
        result.axisWeights = overallInit.axisWeights;
        result.axisLoss = overallInit.axisLoss;
        result.poses = overallInit.poses.block(
            0, cursor, overallInit.poses.rows(), size);
        result.poseScores = overallInit.poseScores.segment(cursor, size);
        result.jointCenters = overallInit.jointCenters.block(
            0, cursor, overallInit.jointCenters.rows(), size);
        result.jointAxis = overallInit.jointAxis.block(
            0, cursor, overallInit.jointAxis.rows(), size);
        separateInits.push_back(result);
      }
      else
      {
        separateInits.push_back(runPrescaledPipeline(
            markerObservationTrials[i],
            InitialMarkerFitParams(params)
                .setGroupScales(overallInit.groupScales)
                .setMarkerOffsets(overallInit.markerOffsets)));
        // separateInits.push_back(fineTuneIK(
        //     markerObservationTrials[i],
        //     params.numBlocks,
        //     params.markerWeights,
        //     jointInits[i]));
      }
    }
    std::cout << "Finished IKs" << std::endl;
    return separateInits;
  }
  else
  {
    // 3. Construct a merged dataset, including the merged joint and axis data
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
    std::vector<bool> newClip;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      for (int j = 0; j < markerObservationTrials[i].size(); j++)
      {
        markerObservations.emplace_back(markerObservationTrials[i][j]);
        newClip.push_back(j == 0);
      }
    }
    // 4. Run the kinematics pipeline on the merged dataset
    MarkerInitialization overallInit = runKinematicsPipeline(
        markerObservations, newClip, params, numSamples, false);

    std::cout << "Recovering output from kinematics pipeline" << std::endl;

    // 5. Separate out the individual trials from the merged data
    std::vector<MarkerInitialization> separateInits;
    int cursor = 0;
    for (int i = 0; i < markerObservationTrials.size(); i++)
    {
      int size = markerObservationTrials[i].size();
      separateInits.emplace_back();

      separateInits[i].poses
          = overallInit.poses.block(0, cursor, overallInit.poses.rows(), size);
      separateInits[i].poseScores
          = overallInit.poseScores.segment(cursor, size);
      separateInits[i].groupScales = overallInit.groupScales;
      separateInits[i].markerOffsets = overallInit.markerOffsets;
      separateInits[i].updatedMarkerMap = overallInit.updatedMarkerMap;

      separateInits[i].joints = overallInit.joints;
      separateInits[i].observedJoints = overallInit.observedJoints;
      separateInits[i].unobservedJoints = overallInit.unobservedJoints;
      separateInits[i].jointMarkerVariability
          = overallInit.jointMarkerVariability;
      separateInits[i].jointsAdjacentMarkers
          = overallInit.jointsAdjacentMarkers;
      separateInits[i].jointLoss = overallInit.jointLoss;
      separateInits[i].jointWeights = overallInit.jointWeights;
      separateInits[i].jointCenters = overallInit.jointCenters.block(
          0, cursor, overallInit.jointCenters.rows(), size);
      separateInits[i].axisWeights = overallInit.axisWeights;
      separateInits[i].axisLoss = overallInit.axisLoss;
      separateInits[i].jointAxis = overallInit.jointAxis.block(
          0, cursor, overallInit.jointAxis.rows(), size);

      /*
      separateInits[i] = fineTuneIK(
          markerObservationTrials[i],
          params.numBlocks,
          params.markerWeights,
          separateInits[i]);
      */

      cursor += size;
    }
    return separateInits;
  }
}

//==============================================================================
/// Run the whole pipeline of optimization problems to fit the data as closely
/// as we can
MarkerInitialization MarkerFitter::runKinematicsPipeline(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<bool>& newClip,
    InitialMarkerFitParams params,
    int numSamples,
    bool skipFinalIK)
{
  // 1. Find the initial scaling + IK
  MarkerInitialization init = getInitialization(
      markerObservations, newClip, InitialMarkerFitParams(params));
  mSkeleton->setGroupScales(init.groupScales);

  // 2. Find the joint centers
  findJointCenters(init, newClip, markerObservations);
  assert(init.joints.size() * 3 == init.jointCenters.rows());
  findAllJointAxis(init, newClip, markerObservations);
  computeJointConfidences(init, markerObservations);
  assert(init.joints.size() * 3 == init.jointCenters.rows());

  // 3. Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = getInitialization(
      markerObservations,
      newClip,
      InitialMarkerFitParams(params)
          .setJointCentersAndWeights(
              init.joints,
              init.jointCenters,
              init.jointsAdjacentMarkers,
              init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  // 4. Run bilevel optimization
  std::shared_ptr<BilevelFitResult> bilevelFit
      = optimizeBilevel(markerObservations, reinit, numSamples);

  // 5. Fine-tune IK and re-fit all the points
  mSkeleton->setGroupScales(bilevelFit->groupScales);
  if (!skipFinalIK)
  {
    MarkerInitialization finalKinematicInit = completeBilevelResult(
        markerObservations,
        newClip,
        bilevelFit,
        init.observedJoints,
        InitialMarkerFitParams(params)
            .setJointCentersAndWeights(
                init.joints,
                init.jointCenters,
                init.jointsAdjacentMarkers,
                init.jointWeights)
            .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
            .setInitPoses(reinit.poses)
            .setDontRescaleBodies(true)
            .setGroupScales(bilevelFit->groupScales)
            .setMarkerOffsets(bilevelFit->markerOffsets));
    return finalKinematicInit;
  }
  else
  {
    std::cout << "Skipping completing bilevel IK, because we're operating on a "
                 "subsampled dataset"
              << std::endl;

    reinit.groupScales = bilevelFit->groupScales;
    reinit.markerOffsets = bilevelFit->markerOffsets;
    return reinit;
  }
}

//==============================================================================
/// This just finds the joint centers and axis over time.
MarkerInitialization MarkerFitter::runJointsPipeline(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    InitialMarkerFitParams params)
{
  std::vector<bool> newClip;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    newClip.push_back(false);
  }

  // 1. Find the initial scaling + IK
  MarkerInitialization init = getInitialization(
      markerObservations, newClip, InitialMarkerFitParams(params));
  mSkeleton->setGroupScales(init.groupScales);
  // 2. Find the joint centers
  findJointCenters(init, newClip, markerObservations);
  findAllJointAxis(init, newClip, markerObservations);
  computeJointConfidences(init, markerObservations);
  return init;
}

//==============================================================================
/// This just runs the IK pipeline steps over the given marker observations,
/// assuming we've got a pre-scaled model. This finds the joint centers and
/// axis over time, then uses those to run multithreaded IK.
MarkerInitialization MarkerFitter::runPrescaledPipeline(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    InitialMarkerFitParams params)
{
  std::vector<bool> newClip;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    newClip.push_back(i == 0);
  }

  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      updatedMarkerMap;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    std::string name = mMarkerNames[i];
    Eigen::Vector3s offset = Eigen::Vector3s::Zero();
    if (params.markerOffsets.count(name))
    {
      offset = params.markerOffsets[name];
    }
    updatedMarkerMap[name] = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
        mMarkerMap[name].first, mMarkerMap[name].second + offset);
  }
  mSkeleton->setGroupScales(params.groupScales);

  std::map<std::string, bool> markerIsAnatomical;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    markerIsAnatomical[mMarkerNames[i]] = !mMarkerIsTracking[i];
  }
  IKInitializer initializer(
      mSkeleton,
      updatedMarkerMap,
      markerIsAnatomical,
      markerObservations,
      newClip,
      -1.0,
      true);
  initializer.runFullPipeline();

  MarkerInitialization result;
  // For now, we're just going to use the poses and scales, and then leave the
  // joint centers to the non-convex optimization pipeline. We could (and
  // maybe should?) use the joint centers that the least-squares algo finds
  // directly. That would require some refactoring, since those joint centers
  // are for "stacked joints" which may be multiple joints in the skeleton at
  // once.
  result.poses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());
  result.groupScales = params.groupScales;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    result.poses.col(i) = initializer.getPoses()[i];
  }

  return result;
}

//==============================================================================
/// This is a convenience method to display just some manually labeled gold
/// data, without having to first run the optimizer.
void MarkerFitter::debugGoldTrajectoryAndMarkersToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    C3D* c3d,
    const OpenSimFile* goldOsim,
    const Eigen::MatrixXs goldPoses)
{
  Eigen::Vector4s goldColor
      = Eigen::Vector4s(59.0 / 255, 184.0 / 255, 92.0 / 255, 0.7);
  server->renderSkeleton(
      goldOsim->skeleton, "gold_", goldColor, "Gold Skeleton");
  // Render the plates as red rectangles
  for (int i = 0; i < c3d->forcePlates.size(); i++)
  {
    std::vector<Eigen::Vector3s> points;
    for (int j = 0; j < c3d->forcePlates[i].corners.size(); j++)
    {
      points.push_back(c3d->forcePlates[i].corners[j]);
    }
    points.push_back(c3d->forcePlates[i].corners[0]);

    server->createLine(
        "plate_" + std::to_string(i),
        points,
        Eigen::Vector4s(1.0, 0., 0., 1.0),
        "Force Plates");
  }

  s_t secondsPerTick = 1.0 / 50;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(secondsPerTick);
  ticker->registerTickListener(
      [c3d, server, secondsPerTick, goldOsim, goldPoses, goldColor](long t) {
        long tick = (long)round((s_t)t / (secondsPerTick * 1000));
        int timestep
            = tick
              % std::min(
                  (long)c3d->markerTimesteps.size(), (long)goldPoses.cols());

        goldOsim->skeleton->setPositions(goldPoses.col(timestep));
        server->renderSkeleton(
            goldOsim->skeleton, "gold_", goldColor, "Manual Skeleton");

        for (auto pair : c3d->markerTimesteps[timestep])
        {
          if (goldOsim->markersMap.count(pair.first) > 0)
          {
            Eigen::Vector3s worldObserved = pair.second;
            Eigen::Vector3s worldInferred
                = goldOsim->markersMap.at(pair.first).first->getWorldTransform()
                  * (goldOsim->markersMap.at(pair.first)
                         .second.cwiseProduct(
                             goldOsim->markersMap.at(pair.first)
                                 .first->getScale()));
            std::vector<Eigen::Vector3s> points;
            points.push_back(worldObserved);
            points.push_back(worldInferred);
            server->createLine(
                "gold_marker_error_" + pair.first,
                points,
                Eigen::Vector4s(1, 0, 0, 1),
                "Manual Skeleton");
          }
        }

        for (int i = 0; i < c3d->forcePlates.size(); i++)
        {
          server->deleteObject("force_" + std::to_string(i));
          if (c3d->forcePlates[i].forces[timestep].squaredNorm() > 0)
          {
            std::vector<Eigen::Vector3s> forcePoints;
            forcePoints.push_back(
                c3d->forcePlates[i].centersOfPressure[timestep]);
            forcePoints.push_back(
                c3d->forcePlates[i].centersOfPressure[timestep]
                + (c3d->forcePlates[i].forces[timestep] * 0.001));
            server->createLine(
                "force_" + std::to_string(i),
                forcePoints,
                Eigen::Vector4s(1.0, 0, 0, 1.),
                "Force Plates");
          }
        }
      });
  server->registerConnectionListener([ticker]() { ticker->start(); });
}

//==============================================================================
void MarkerFitter::debugTrajectoryAndMarkersToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    MarkerInitialization init,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    std::vector<ForcePlate> forcePlates,
    const OpenSimFile* goldOsim,
    const Eigen::MatrixXs goldPoses)
{
  std::string autoFitLayerName = "Auto-fit Skeleton";
  Eigen::Vector4s autoFitLayerColor = Eigen::Vector4s(0.8, 0.8, 0.8, 1.0);
  std::string trackingMarkersLayerName = "Tracking Markers";
  Eigen::Vector4s trackingMarkersLayerColor
      = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0);
  std::string anatomicalMarkersLayerName = "Anatomical Markers";
  Eigen::Vector4s anatomicalMarkersLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  std::string unusedMarkersLayerName = "Unused Markers";
  Eigen::Vector4s unusedMarkersLayerColor = Eigen::Vector4s(1.0, 0.5, 0.5, 1.0);
  std::string virtualMarkersLayerName = "Virtual Markers";
  Eigen::Vector4s virtualMarkersLayerColor
      = Eigen::Vector4s(0.5, 1.0, 0.5, 1.0);
  std::string markersErrorLayerName = "Marker Errors";
  Eigen::Vector4s markersErrorLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string forcePlateLayerName = "Force Plates";
  Eigen::Vector4s forcePlateLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);

  std::string functionalJointCenterLayerName = "Functional Joint Centers";
  Eigen::Vector4s functionalJointCenterLayerColor
      = Eigen::Vector4s(0.0, 1.0, 0.0, 1.0);
  std::string originalMarkerLocationsLayerName
      = "Original Anatomical Marker Locations";
  Eigen::Vector4s originalMarkerLocationsLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 0.3);
  std::string manualLayerName = "Manual Skeleton";
  Eigen::Vector4s manualLayerColor
      = Eigen::Vector4s(59.0 / 255, 184.0 / 255, 92.0 / 255, 0.7);

  server->createLayer(autoFitLayerName, autoFitLayerColor, true);
  server->createLayer(
      trackingMarkersLayerName, trackingMarkersLayerColor, true);
  server->createLayer(
      anatomicalMarkersLayerName, anatomicalMarkersLayerColor, true);
  server->createLayer(unusedMarkersLayerName, unusedMarkersLayerColor, true);
  server->createLayer(virtualMarkersLayerName, virtualMarkersLayerColor, false);
  server->createLayer(markersErrorLayerName, markersErrorLayerColor, true);
  server->createLayer(forcePlateLayerName, forcePlateLayerColor, true);

  server->createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, false);
  server->createLayer(
      originalMarkerLocationsLayerName,
      originalMarkerLocationsLayerColor,
      false);
  server->renderSkeleton(
      mSkeleton, "auto_", Eigen::Vector4s::Ones() * -1, autoFitLayerName);

  if (goldOsim && goldPoses.size() > 0)
  {
    server->createLayer(manualLayerName, manualLayerColor, false);
    server->renderSkeleton(
        goldOsim->skeleton, "gold_", manualLayerColor, manualLayerName);
  }

  int numJoints = init.jointCenters.rows() / 3;
  server->createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, true);
  for (int i = 0; i < numJoints; i++)
  {
    if (init.jointWeights(i) > 0)
    {
      server->setObjectTooltip(
          "joint_center_" + std::to_string(i),
          "Joint center: " + init.joints[i]->getName());
      server->createSphere(
          "joint_center_" + std::to_string(i),
          0.01 * min(3.0, (1.0 / init.jointWeights(i))),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init.jointWeights(i)),
          functionalJointCenterLayerName);
    }
  }
  int numAxis = init.jointAxis.rows() / 6;
  for (int i = 0; i < numAxis; i++)
  {
    if (init.axisWeights(i) > 0)
    {
      server->createCapsule(
          "joint_axis_" + std::to_string(i),
          0.003 * min(3.0, (1.0 / init.axisWeights(i))),
          0.1,
          Eigen::Vector3s::Zero(),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init.axisWeights(i)),
          functionalJointCenterLayerName);
    }
  }

  // Render the plates as red rectangles
  for (int i = 0; i < forcePlates.size(); i++)
  {
    if (forcePlates[i].corners.size() > 0)
    {
      std::vector<Eigen::Vector3s> points;
      for (int j = 0; j < forcePlates[i].corners.size(); j++)
      {
        points.push_back(forcePlates[i].corners[j]);
      }
      points.push_back(forcePlates[i].corners[0]);

      server->createLine(
          "plate_" + std::to_string(i),
          points,
          forcePlateLayerColor,
          forcePlateLayerName);
    }
  }

  std::set<std::string> observedMarkers;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    for (auto pair : markerObservations[i])
    {
      observedMarkers.emplace(pair.first);
    }
  }

  for (std::string marker : observedMarkers)
  {
    if (mMarkerMap.count(marker) == 0)
    {
      server->setObjectTooltip(
          "marker_unused_" + marker, "Unused Marker: " + marker);
    }
  }

  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (observedMarkers.count(mMarkerNames[i]) == 0)
    {
      server->setObjectTooltip(
          "marker_virtual_" + mMarkerNames[i],
          "Virtual Marker: " + mMarkerNames[i]);
    }
    else
    {
      bool isTracking = mMarkerIsTracking[i];
      server->setObjectTooltip(
          "marker_real_" + mMarkerNames[i],
          (isTracking ? "Tracking Marker: " : "Anatomical Marker: ")
              + mMarkerNames[i]);
    }
  }

  s_t secondsPerTick = 1.0 / 50;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(secondsPerTick);
  ticker->registerTickListener([forcePlates,
                                server,
                                init,
                                markerObservations,
                                numJoints,
                                numAxis,
                                secondsPerTick,
                                goldOsim,
                                goldPoses,
                                manualLayerName,
                                manualLayerColor,
                                autoFitLayerName,
                                trackingMarkersLayerName,
                                trackingMarkersLayerColor,
                                anatomicalMarkersLayerName,
                                anatomicalMarkersLayerColor,
                                unusedMarkersLayerName,
                                unusedMarkersLayerColor,
                                virtualMarkersLayerName,
                                virtualMarkersLayerColor,
                                observedMarkers,
                                markersErrorLayerColor,
                                markersErrorLayerName,
                                originalMarkerLocationsLayerName,
                                originalMarkerLocationsLayerColor,
                                functionalJointCenterLayerName,
                                functionalJointCenterLayerColor,
                                forcePlateLayerName,
                                forcePlateLayerColor,
                                this](long t) {
    long tick = (long)round((s_t)t / (secondsPerTick * 1000));
    int timestep = tick % init.poses.cols();
    mSkeleton->setPositions(init.poses.col(timestep));
    server->renderSkeleton(
        mSkeleton, "auto_", Eigen::Vector4s::Ones() * -1, autoFitLayerName);

    std::map<std::string, Eigen::Vector3s> markerWorldPositions
        = markerObservations[timestep];
    server->deleteObjectsByPrefix("marker_error_");
    for (auto pair : markerWorldPositions)
    {
      if (init.updatedMarkerMap.count(pair.first) > 0)
      {
        Eigen::Vector3s worldObserved = pair.second;
        Eigen::Vector3s worldInferred
            = init.updatedMarkerMap.at(pair.first).first->getWorldTransform()
              * (init.updatedMarkerMap.at(pair.first)
                     .second.cwiseProduct(init.updatedMarkerMap.at(pair.first)
                                              .first->getScale()));
        bool isTracking = mMarkerIsTracking[mMarkerIndices[pair.first]];
        Eigen::Vector4s color = isTracking ? trackingMarkersLayerColor
                                           : anatomicalMarkersLayerColor;
        std::string label = isTracking ? trackingMarkersLayerName
                                       : anatomicalMarkersLayerName;
        server->createBox(
            "marker_real_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            worldObserved,
            Eigen::Vector3s::Zero(),
            color,
            label);
        /*
        server->createBox(
            "marker_inferred_" + pair.first,
            Eigen::Vector3s::Ones() * 0.007,
            worldInferred,
            Eigen::Vector3s::Zero(),
            color,
            label);
        */

        std::vector<Eigen::Vector3s> points;
        points.push_back(worldObserved);
        points.push_back(worldInferred);
        server->createLine(
            "marker_error_" + pair.first,
            points,
            markersErrorLayerColor,
            markersErrorLayerName);

        if (!mMarkerIsTracking.at(mMarkerIndices.at(pair.first)))
        {
          Eigen::Vector3s worldOriginal
              = init.updatedMarkerMap.at(pair.first).first->getWorldTransform()
                * ((init.updatedMarkerMap.at(pair.first).second
                    - init.markerOffsets.at(pair.first))
                       .cwiseProduct(init.updatedMarkerMap.at(pair.first)
                                         .first->getScale()));
          server->createBox(
              "marker_original_" + pair.first,
              Eigen::Vector3s::Ones() * 0.005,
              worldOriginal,
              Eigen::Vector3s::Zero(),
              originalMarkerLocationsLayerColor,
              originalMarkerLocationsLayerName);
          std::vector<Eigen::Vector3s> offsetPoints;
          offsetPoints.push_back(worldOriginal);
          offsetPoints.push_back(worldInferred);
          server->createLine(
              "marker_offset_" + pair.first,
              offsetPoints,
              originalMarkerLocationsLayerColor,
              originalMarkerLocationsLayerName);
        }
      }
      else
      {
        server->createBox(
            "marker_unused_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            pair.second,
            Eigen::Vector3s::Zero(),
            unusedMarkersLayerColor,
            unusedMarkersLayerName);
      }
    }

    std::map<std::string, Eigen::Vector3s> markerWorldPosition
        = mSkeleton->getMarkerMapWorldPositions(mMarkerMap);
    for (auto pair : markerWorldPosition)
    {
      if (markerWorldPositions.count(pair.first) == 0)
      {
        if (init.updatedMarkerMap.count(pair.first) > 0)
        {
          server->deleteObject("marker_original_" + pair.first);
        }
        else
        {
          server->deleteObject("marker_unused_" + pair.first);
        }
      }
      if (observedMarkers.count(pair.first) == 0)
      {
        // This is a virtual marker:
        server->createBox(
            "marker_virtual_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            pair.second,
            Eigen::Vector3s::Zero(),
            virtualMarkersLayerColor,
            virtualMarkersLayerName);
      }
    }

    if (goldOsim && goldPoses.size() > 0)
    {
      goldOsim->skeleton->setPositions(goldPoses.col(timestep));
      server->renderSkeleton(
          goldOsim->skeleton, "gold_", manualLayerColor, manualLayerName);

      for (auto pair : markerWorldPositions)
      {
        if (goldOsim->markersMap.count(pair.first) > 0)
        {
          Eigen::Vector3s worldObserved = pair.second;
          Eigen::Vector3s worldInferred
              = goldOsim->markersMap.at(pair.first).first->getWorldTransform()
                * (goldOsim->markersMap.at(pair.first)
                       .second.cwiseProduct(goldOsim->markersMap.at(pair.first)
                                                .first->getScale()));
          std::vector<Eigen::Vector3s> points;
          points.push_back(worldObserved);
          points.push_back(worldInferred);
          server->createLine(
              "gold_marker_error_" + pair.first,
              points,
              manualLayerColor,
              manualLayerName);
        }
      }
    }

    for (int i = 0; i < forcePlates.size(); i++)
    {
      server->deleteObject("force_" + std::to_string(i));
      if (forcePlates[i].forces[timestep].squaredNorm() > 0)
      {
        std::vector<Eigen::Vector3s> forcePoints;
        forcePoints.push_back(forcePlates[i].centersOfPressure[timestep]);
        forcePoints.push_back(
            forcePlates[i].centersOfPressure[timestep]
            + (forcePlates[i].forces[timestep] * 0.001));
        server->createLine(
            "force_" + std::to_string(i),
            forcePoints,
            forcePlateLayerColor,
            forcePlateLayerName);
      }
    }

    for (int i = 0; i < numJoints; i++)
    {
      if (init.jointWeights(i) > 0)
      {
        Eigen::Vector3s inferredJointCenter
            = init.jointCenters.block<3, 1>(i * 3, timestep);
        server->setObjectPosition(
            "joint_center_" + std::to_string(i), inferredJointCenter);
        if (i < init.jointsAdjacentMarkers.size())
        {
          for (std::string marker : init.jointsAdjacentMarkers[i])
          {
            if (markerWorldPositions.count(marker))
            {
              std::vector<Eigen::Vector3s> centerToMarker;
              centerToMarker.push_back(inferredJointCenter);
              centerToMarker.push_back(markerWorldPositions[marker]);
              server->createLine(
                  "joint_center_" + std::to_string(i) + "_to_marker_" + marker,
                  centerToMarker,
                  functionalJointCenterLayerColor,
                  functionalJointCenterLayerName);
            }
          }
        }
      }
    }
    for (int i = 0; i < numAxis; i++)
    {
      if (init.axisWeights(i) > 0)
      {
        // Render an axis capsule
        server->setObjectPosition(
            "joint_axis_" + std::to_string(i),
            init.jointAxis.block<3, 1>(i * 6, timestep));
        Eigen::Vector3s dir = init.jointAxis.block<3, 1>(i * 6 + 3, timestep);
        Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
        R.col(2) = dir;
        R.col(1) = Eigen::Vector3s::UnitZ().cross(dir);
        R.col(0) = R.col(1).cross(R.col(2));
        server->setObjectRotation(
            "joint_axis_" + std::to_string(i), math::matrixToEulerXYZ(R));
      }
    }
  });
  server->registerConnectionListener([ticker]() { ticker->start(); });

  server->registerKeydownListener([ticker](std::string key) {
    if (key == " ")
    {
      ticker->toggle();
    }
  });
  // TODO: it'd be nice if this method didn't block forever, but we need to hold
  // onto a bunch of resources otherwise
  // server->blockWhileServing();
}

//==============================================================================
void MarkerFitter::saveTrajectoryAndMarkersToGUI(
    std::string path,
    MarkerInitialization init,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    int framesPerSecond,
    std::vector<ForcePlate> forcePlates,
    const OpenSimFile* goldOsim,
    const Eigen::MatrixXs goldPoses)
{
  std::string autoFitLayerName = "Auto-fit Skeleton";
  Eigen::Vector4s autoFitLayerColor = Eigen::Vector4s(0.8, 0.8, 0.8, 1.0);
  std::string trackingMarkersLayerName = "Tracking Markers";
  Eigen::Vector4s trackingMarkersLayerColor
      = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0);
  std::string anatomicalMarkersLayerName = "Anatomical Markers";
  Eigen::Vector4s anatomicalMarkersLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  std::string unusedMarkersLayerName = "Unused Markers";
  Eigen::Vector4s unusedMarkersLayerColor = Eigen::Vector4s(1.0, 0.5, 0.5, 1.0);
  std::string virtualMarkersLayerName = "Virtual Markers";
  Eigen::Vector4s virtualMarkersLayerColor
      = Eigen::Vector4s(0.5, 1.0, 0.5, 1.0);
  std::string markersErrorLayerName = "Marker Errors";
  Eigen::Vector4s markersErrorLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string forcePlateLayerName = "Force Plates";
  Eigen::Vector4s forcePlateLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);

  std::string functionalJointCenterLayerName = "Functional Joint Centers";
  Eigen::Vector4s functionalJointCenterLayerColor
      = Eigen::Vector4s(0.0, 1.0, 0.0, 1.0);
  std::string originalMarkerLocationsLayerName
      = "Original Anatomical Marker Locations";
  Eigen::Vector4s originalMarkerLocationsLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 0.3);
  std::string manualLayerName = "Manual Skeleton";
  Eigen::Vector4s manualLayerColor
      = Eigen::Vector4s(59.0 / 255, 184.0 / 255, 92.0 / 255, 0.7);

  std::string imuLayerName = "IMUs";
  Eigen::Vector4s imuLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);

  server::GUIRecording server;
  server.setFramesPerSecond(framesPerSecond);
  server.createLayer(autoFitLayerName, autoFitLayerColor, true);
  server.createLayer(trackingMarkersLayerName, trackingMarkersLayerColor, true);
  server.createLayer(
      anatomicalMarkersLayerName, anatomicalMarkersLayerColor, true);
  server.createLayer(unusedMarkersLayerName, unusedMarkersLayerColor, true);
  server.createLayer(virtualMarkersLayerName, virtualMarkersLayerColor, false);
  server.createLayer(markersErrorLayerName, markersErrorLayerColor, true);
  server.createLayer(forcePlateLayerName, forcePlateLayerColor, true);
  server.createLayer(imuLayerName, imuLayerColor, true);

  server.createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, false);
  server.createLayer(
      originalMarkerLocationsLayerName,
      originalMarkerLocationsLayerColor,
      false);
  server.renderSkeleton(
      mSkeleton, "auto_", Eigen::Vector4s::Ones() * -1, autoFitLayerName);
  if (goldOsim && goldPoses.size() > 0)
  {
    server.createLayer(manualLayerName, manualLayerColor, false);
    server.renderSkeleton(
        goldOsim->skeleton, "gold_", manualLayerColor, manualLayerName);
  }

  int numJoints = init.jointCenters.rows() / 3;
  server.createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, true);
  for (int i = 0; i < numJoints; i++)
  {
    if (init.jointWeights(i) > 0)
    {
      server.setObjectTooltip(
          "joint_center_" + std::to_string(i),
          "Joint center: " + init.joints[i]->getName());
      server.createSphere(
          "joint_center_" + std::to_string(i),
          0.01 * min(3.0, (1.0 / init.jointWeights(i))),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init.jointWeights(i)),
          functionalJointCenterLayerName);
    }
  }
  int numAxis = init.jointAxis.rows() / 6;
  for (int i = 0; i < numAxis; i++)
  {
    if (init.axisWeights(i) > 0)
    {
      server.createCapsule(
          "joint_axis_" + std::to_string(i),
          0.003 * min(3.0, (1.0 / init.axisWeights(i))),
          0.1,
          Eigen::Vector3s::Zero(),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init.axisWeights(i)),
          functionalJointCenterLayerName);
    }
  }

  // Render the plates as red rectangles
  for (int i = 0; i < forcePlates.size(); i++)
  {
    if (forcePlates[i].corners.size() > 0)
    {
      std::vector<Eigen::Vector3s> points;
      for (int j = 0; j < forcePlates[i].corners.size(); j++)
      {
        points.push_back(forcePlates[i].corners[j]);
      }
      points.push_back(forcePlates[i].corners[0]);

      server.createLine(
          "plate_" + std::to_string(i),
          points,
          forcePlateLayerColor,
          forcePlateLayerName);
    }
  }

  std::set<std::string> observedMarkers;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    for (auto pair : markerObservations[i])
    {
      observedMarkers.emplace(pair.first);
    }
  }

  for (std::string marker : observedMarkers)
  {
    if (mMarkerMap.count(marker) == 0)
    {
      server.setObjectTooltip(
          "marker_unused_" + marker, "Unused Marker: " + marker);
    }
  }

  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (observedMarkers.count(mMarkerNames[i]) == 0)
    {
      server.setObjectTooltip(
          "marker_virtual_" + mMarkerNames[i],
          "Virtual Marker: " + mMarkerNames[i]);
    }
    else
    {
      bool isTracking = mMarkerIsTracking[i];
      server.setObjectTooltip(
          "marker_real_" + mMarkerNames[i],
          (isTracking ? "Tracking Marker: " : "Anatomical Marker: ")
              + mMarkerNames[i]);
    }
  }
  std::map<std::string, Eigen::Isometry3s> sensorWorldMap
      = mSkeleton->getSensorWorldPositions(mImuMap);
  for (auto& pair : sensorWorldMap)
  {
    std::cout << "Rendering IMU: " << pair.first << std::endl;
    server.createBox(
        "imu_" + pair.first,
        Eigen::Vector3s::Ones() * 0.03,
        pair.second.translation(),
        math::matrixToEulerXYZ(pair.second.linear()),
        imuLayerColor);
    server.setObjectTooltip("imu_" + pair.first, "IMU: " + pair.first);
  }

  for (int timestep = 0; timestep < init.poses.cols(); timestep++)
  {
    mSkeleton->setPositions(init.poses.col(timestep));
    server.renderSkeleton(
        mSkeleton, "auto_", Eigen::Vector4s::Ones() * -1, autoFitLayerName);

    std::map<std::string, Eigen::Vector3s> markerWorldPositions
        = markerObservations[timestep];
    server.deleteObjectsByPrefix("marker_error_");
    for (auto pair : markerWorldPositions)
    {
      if (init.updatedMarkerMap.count(pair.first) > 0)
      {
        Eigen::Vector3s worldObserved = pair.second;
        Eigen::Vector3s worldInferred
            = init.updatedMarkerMap.at(pair.first).first->getWorldTransform()
              * (init.updatedMarkerMap.at(pair.first)
                     .second.cwiseProduct(init.updatedMarkerMap.at(pair.first)
                                              .first->getScale()));
        bool isTracking = mMarkerIsTracking[mMarkerIndices[pair.first]];
        Eigen::Vector4s color = isTracking ? trackingMarkersLayerColor
                                           : anatomicalMarkersLayerColor;
        std::string label = isTracking ? trackingMarkersLayerName
                                       : anatomicalMarkersLayerName;
        server.createBox(
            "marker_real_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            worldObserved,
            Eigen::Vector3s::Zero(),
            color,
            label);
        /*
        server->createBox(
            "marker_inferred_" + pair.first,
            Eigen::Vector3s::Ones() * 0.007,
            worldInferred,
            Eigen::Vector3s::Zero(),
            color,
            label);
        */

        std::vector<Eigen::Vector3s> points;
        points.push_back(worldObserved);
        points.push_back(worldInferred);
        server.createLine(
            "marker_error_" + pair.first,
            points,
            markersErrorLayerColor,
            markersErrorLayerName);

        if (!mMarkerIsTracking.at(mMarkerIndices.at(pair.first)))
        {
          Eigen::Vector3s worldOriginal
              = init.updatedMarkerMap.at(pair.first).first->getWorldTransform()
                * ((init.updatedMarkerMap.at(pair.first).second
                    - init.markerOffsets.at(pair.first))
                       .cwiseProduct(init.updatedMarkerMap.at(pair.first)
                                         .first->getScale()));
          server.createBox(
              "marker_original_" + pair.first,
              Eigen::Vector3s::Ones() * 0.005,
              worldOriginal,
              Eigen::Vector3s::Zero(),
              originalMarkerLocationsLayerColor,
              originalMarkerLocationsLayerName);
          std::vector<Eigen::Vector3s> offsetPoints;
          offsetPoints.push_back(worldOriginal);
          offsetPoints.push_back(worldInferred);
          server.createLine(
              "marker_offset_" + pair.first,
              offsetPoints,
              originalMarkerLocationsLayerColor,
              originalMarkerLocationsLayerName);
        }
      }
      else
      {
        server.createBox(
            "marker_unused_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            pair.second,
            Eigen::Vector3s::Zero(),
            unusedMarkersLayerColor,
            unusedMarkersLayerName);
      }
    }

    std::map<std::string, Eigen::Vector3s> markerWorldPosition
        = mSkeleton->getMarkerMapWorldPositions(mMarkerMap);
    for (auto pair : markerWorldPosition)
    {
      if (markerWorldPositions.count(pair.first) == 0)
      {
        if (init.updatedMarkerMap.count(pair.first) > 0)
        {
          server.deleteObject("marker_original_" + pair.first);
        }
        else
        {
          server.deleteObject("marker_unused_" + pair.first);
        }
      }
      if (observedMarkers.count(pair.first) == 0)
      {
        // This is a virtual marker:
        server.createBox(
            "marker_virtual_" + pair.first,
            Eigen::Vector3s::Ones() * 0.01,
            pair.second,
            Eigen::Vector3s::Zero(),
            virtualMarkersLayerColor,
            virtualMarkersLayerName);
      }
    }

    std::map<std::string, Eigen::Isometry3s> sensorWorldTransforms
        = mSkeleton->getSensorWorldPositions(mImuMap);
    for (auto& pair : sensorWorldTransforms)
    {
      server.setObjectPosition("imu_" + pair.first, pair.second.translation());
      server.setObjectRotation(
          "imu_" + pair.first, math::matrixToEulerXYZ(pair.second.linear()));
    }

    if (goldOsim && goldPoses.size() > 0)
    {
      goldOsim->skeleton->setPositions(goldPoses.col(timestep));
      server.renderSkeleton(
          goldOsim->skeleton, "gold_", manualLayerColor, manualLayerName);

      for (auto pair : markerWorldPositions)
      {
        if (goldOsim->markersMap.count(pair.first) > 0)
        {
          Eigen::Vector3s worldObserved = pair.second;
          Eigen::Vector3s worldInferred
              = goldOsim->markersMap.at(pair.first).first->getWorldTransform()
                * (goldOsim->markersMap.at(pair.first)
                       .second.cwiseProduct(goldOsim->markersMap.at(pair.first)
                                                .first->getScale()));
          std::vector<Eigen::Vector3s> points;
          points.push_back(worldObserved);
          points.push_back(worldInferred);
          server.createLine(
              "gold_marker_error_" + pair.first,
              points,
              manualLayerColor,
              manualLayerName);
        }
      }
    }

    for (int i = 0; i < forcePlates.size(); i++)
    {
      server.deleteObject("force_" + std::to_string(i));
      if (forcePlates[i].forces[timestep].squaredNorm() > 0)
      {
        std::vector<Eigen::Vector3s> forcePoints;
        forcePoints.push_back(forcePlates[i].centersOfPressure[timestep]);
        forcePoints.push_back(
            forcePlates[i].centersOfPressure[timestep]
            + (forcePlates[i].forces[timestep] * 0.001));
        server.createLine(
            "force_" + std::to_string(i),
            forcePoints,
            forcePlateLayerColor,
            forcePlateLayerName);
      }
    }
    for (int i = 0; i < numJoints; i++)
    {
      if (init.jointWeights(i) > 0)
      {
        Eigen::Vector3s inferredJointCenter
            = init.jointCenters.block<3, 1>(i * 3, timestep);
        server.setObjectPosition(
            "joint_center_" + std::to_string(i), inferredJointCenter);
        if (i < init.jointsAdjacentMarkers.size())
        {
          for (std::string marker : init.jointsAdjacentMarkers[i])
          {
            if (markerWorldPositions.count(marker))
            {
              std::vector<Eigen::Vector3s> centerToMarker;
              centerToMarker.push_back(inferredJointCenter);
              centerToMarker.push_back(markerWorldPositions[marker]);
              server.createLine(
                  "joint_center_" + std::to_string(i) + "_to_marker_" + marker,
                  centerToMarker,
                  functionalJointCenterLayerColor,
                  functionalJointCenterLayerName);
            }
          }
        }
      }
    }
    for (int i = 0; i < numAxis; i++)
    {
      if (init.axisWeights(i) > 0)
      {
        // Render an axis capsule
        server.setObjectPosition(
            "joint_axis_" + std::to_string(i),
            init.jointAxis.block<3, 1>(i * 6, timestep));
        Eigen::Vector3s dir = init.jointAxis.block<3, 1>(i * 6 + 3, timestep);
        Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
        R.col(2) = dir;
        R.col(1) = Eigen::Vector3s::UnitZ().cross(dir);
        R.col(0) = R.col(1).cross(R.col(2));
        server.setObjectRotation(
            "joint_axis_" + std::to_string(i), math::matrixToEulerXYZ(R));
      }
    }

    s_t dt = 1.0 / framesPerSecond;
    if (timestep > 0 && timestep < init.poses.cols() - 1)
    {
      Eigen::VectorXs q = init.poses.col(timestep);
      Eigen::VectorXs dq
          = mSkeleton->getPositionDifferences(
                init.poses.col(timestep), init.poses.col(timestep - 1))
            / dt;
      Eigen::VectorXs ddq
          = (mSkeleton->getPositionDifferences(
                 init.poses.col(timestep + 1), init.poses.col(timestep))
             - mSkeleton->getPositionDifferences(
                 init.poses.col(timestep), init.poses.col(timestep - 1)))
            / (dt * dt);
      mSkeleton->setPositions(q);
      mSkeleton->setVelocities(dq);
      mSkeleton->setAccelerations(ddq);

      std::map<std::string, Eigen::Vector3s> virtualAccMap
          = mSkeleton->getAccMapReadings(mImuMap);
      std::map<std::string, Eigen::Vector3s> realAccMap;
      if (accObservations.size() > timestep)
      {
        realAccMap = accObservations[timestep];
      }
      else
      {
        realAccMap = std::map<std::string, Eigen::Vector3s>();
      }
      std::map<std::string, Eigen::Vector3s> virtualGyroMap
          = mSkeleton->getGyroMapReadings(mImuMap);
      std::map<std::string, Eigen::Vector3s> realGyroMap;
      if (gyroObservations.size() > timestep)
      {
        realGyroMap = gyroObservations[timestep];
      }

      s_t renderScale = 0.2;
      for (auto& pair : sensorWorldTransforms)
      {
        std::vector<Eigen::Vector3s> virtualAccLine;
        virtualAccLine.push_back(pair.second.translation());
        Eigen::Vector3s sensorAcc
            = renderScale * pair.second.linear() * virtualAccMap.at(pair.first);
        virtualAccLine.push_back(pair.second.translation() + sensorAcc);
        server.createLine(
            "virtual_acc_" + pair.first,
            virtualAccLine,
            imuLayerColor,
            imuLayerName);

        std::vector<Eigen::Vector3s> realAccLine;
        realAccLine.push_back(pair.second.translation());
        Eigen::Vector3s realAcc
            = renderScale * pair.second.linear() * realAccMap.at(pair.first);
        realAccLine.push_back(pair.second.translation() + realAcc);
        server.createLine(
            "real_acc_" + pair.first,
            realAccLine,
            Eigen::Vector4s(0, 0, 1, 1),
            imuLayerName);

        /*
        std::vector<Eigen::Vector3s> virtualGyroLine;
        virtualGyroLine.push_back(pair.second.translation());
        Eigen::Vector3s sensorGyro
            = pair.second.linear() * virtualGyroMap.at(pair.first);
        virtualGyroLine.push_back(pair.second.translation() + sensorGyro);
        server.createLine(
            "virtual_gyro_" + pair.first,
            virtualGyroLine,
            imuLayerColor,
            imuLayerName);

        std::vector<Eigen::Vector3s> realGyroLine;
        realGyroLine.push_back(pair.second.translation());
        Eigen::Vector3s realGyro
            = pair.second.linear() * realGyroMap.at(pair.first);
        realGyroLine.push_back(pair.second.translation() + realGyro);
        server.createLine(
            "real_gyro_" + pair.first,
            realGyroLine,
            imuLayerColor,
            imuLayerName);
        */
      }
    }
    else
    {
      for (auto& pair : sensorWorldTransforms)
      {
        server.deleteObject("virtual_acc_" + pair.first);
        server.deleteObject("real_acc_" + pair.first);
      }
    }

    server.saveFrame();
  }

  server.writeFramesJson(path);
}

//==============================================================================
/// This automatically finds the "probably correct" rotation for the C3D data
/// that has no force plate data and rotates the C3D data to match it. This is
/// determined by which orientation for the data has the skeleton torso
/// pointed generally upwards most of the time. While this is usually a safe
/// assumption, it could break down with breakdancing or some other strang
/// motions, so it should be an option to turn it off.
void MarkerFitter::autorotateC3D(C3D* c3d)
{
  if (c3d->forcePlates.size() > 0)
  {
    std::cout << "Attempted to call MarkerFitter::autorotateC3D() on a c3d "
                 "file with force plates! This is redundant, because c3d files "
                 "containing forceplates are automatically rotated when "
                 "they're loaded, and so the call is being ignored."
              << std::endl;
    return;
  }
  std::vector<bool> newClip;
  for (int i = 0; i < c3d->markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  std::vector<Eigen::Matrix3s> rotationsToTry;
  rotationsToTry.push_back(Eigen::Matrix3s::Identity());
  // People generally just can't agree on which axis means "up", Y or Z
  rotationsToTry.push_back(
      math::eulerXYZToMatrix(Eigen::Vector3s(-M_PI / 2, 0, 0)));
  rotationsToTry.push_back(
      math::eulerXYZToMatrix(Eigen::Vector3s(M_PI / 2, 0, 0)));

  s_t smallestMag = std::numeric_limits<double>::infinity();
  Eigen::Matrix3s smallestMagR = Eigen::Matrix3s::Identity();

  for (Eigen::Matrix3s& R : rotationsToTry)
  {
    std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
    for (int i = 0; i < c3d->markerTimesteps.size(); i++)
    {
      std::map<std::string, Eigen::Vector3s> rotatedMarkers;
      for (auto& pair : c3d->markerTimesteps[i])
      {
        rotatedMarkers[pair.first] = R * pair.second;
      }
      markerTimesteps.push_back(rotatedMarkers);
    }

    MarkerInitialization init = getInitialization(
        markerTimesteps,
        newClip,
        InitialMarkerFitParams().setNumIKTries(1).setDontRescaleBodies(true));

    Eigen::VectorXs originalPose = mSkeleton->getPositions();
    s_t avgMag = 0.0;
    for (int i = 0; i < init.poses.cols(); i++)
    {
      mSkeleton->setPositions(init.poses.col(i));
      s_t mag = math::logMap(
                    mSkeleton->getJoint(0)->getRelativeTransform().linear())
                    .norm();
      avgMag += mag;
    }
    avgMag /= init.poses.cols();
    mSkeleton->setPositions(originalPose);

    if (avgMag < smallestMag)
    {
      smallestMag = avgMag;
      smallestMagR = R;
    }
  }

  std::cout << "Picked best rotation for c3d data: " << std::endl
            << smallestMagR << std::endl;

  c3d->dataRotation = smallestMagR * c3d->dataRotation;
  for (int i = 0; i < c3d->markerTimesteps.size(); i++)
  {
    for (auto& pair : c3d->markerTimesteps[i])
    {
      pair.second = smallestMagR * pair.second;
    }
  }
}

//==============================================================================
/// This solves an optimization problem, trying to get the Skeleton to match
/// the markers as closely as possible.
std::shared_ptr<BilevelFitResult> MarkerFitter::optimizeBilevel(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    MarkerInitialization& initialization,
    int numSamples,
    bool applyInnerProblemGradientConstraints)
{
  // Before using Eigen in a multi-threaded environment, we need to explicitly
  // call this (at least prior to Eigen 3.3)
  Eigen::initParallel();

  // Create an instance of the IpoptApplication
  //
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", static_cast<double>(mTolerance));
  app->Options()->SetStringValue(
      "linear_solver",
      "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps, custom

  app->Options()->SetStringValue(
      "hessian_approximation", "limited-memory"); // limited-memory, exacty

  /*
  app->Options()->SetStringValue(
      "scaling_method", "none"); // none, gradient-based
  */

  app->Options()->SetIntegerValue("max_iter", mIterationLimit);

  // Disable LBFGS history
  app->Options()->SetIntegerValue(
      "limited_memory_max_history", mLBFGSHistoryLength);

  // Just for debugging
  if (mCheckDerivatives)
  {
    app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
    app->Options()->SetStringValue("derivative_test", "first-order");
    app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);
  }

  if (mPrintFrequency > 0)
  {
    app->Options()->SetIntegerValue("print_frequency_iter", mPrintFrequency);
  }
  else
  {
    app->Options()->SetIntegerValue(
        "print_frequency_iter", std::numeric_limits<int>::infinity());
  }
  if (mSilenceOutput)
  {
    app->Options()->SetIntegerValue("print_level", 0);
  }
  if (mDisableLinesearch)
  {
    app->Options()->SetIntegerValue("max_soc", 0);
    app->Options()->SetStringValue("accept_every_trial_step", "yes");
  }
  app->Options()->SetIntegerValue("watchdog_shortened_iter_trigger", 0);

  std::shared_ptr<BilevelFitResult> result
      = std::make_shared<BilevelFitResult>();

  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded)
  {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return result;
  }

  // This will automatically free the problem object when finished,
  // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will crash.
  // If you try to leave `problem` on the stack, you'll get invalid free
  // exceptions when IPOpt attempts to free it.
  BilevelFitProblem* problem = new BilevelFitProblem(
      this,
      markerObservations,
      initialization,
      numSamples,
      applyInnerProblemGradientConstraints,
      result);
  result->sampleIndices = problem->getSampleIndices();

  SmartPtr<BilevelFitProblem> problemPtr(problem);
  status = app->OptimizeTNLP(problemPtr);

  if (status == Solve_Succeeded)
  {
    // Retrieve some statistics about the solve
    Index iter_count = app->Statistics()->IterationCount();
    std::cout << std::endl
              << std::endl
              << "*** The problem solved in " << iter_count << " iterations!"
              << std::endl;

    Number final_obj = app->Statistics()->FinalObjective();
    std::cout << std::endl
              << std::endl
              << "*** The final value of the objective function is "
              << final_obj << '.' << std::endl;
  }

  result->success = (status == Ipopt::Solve_Succeeded);
  std::cout << "Number of results: " << result->poses.size() << std::endl;
  result->posesMatrix
      = Eigen::MatrixXs::Zero(result->poses[0].size(), result->poses.size());
  for (int i = 0; i < result->poses.size(); i++)
  {
    result->posesMatrix.col(i) = result->poses[i];
  }

  return result;
}

//==============================================================================
/// The bilevel optimization only picks a subset of poses to fine tune. This
/// method takes those poses as a starting point, and extends each pose
/// forward and backwards (half the distance to the next pose to fine tune)
/// with IK.
MarkerInitialization MarkerFitter::completeBilevelResult(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<bool>& newClip,
    std::shared_ptr<BilevelFitResult> solution,
    std::vector<dynamics::Joint*> initObservedJoints,
    InitialMarkerFitParams params)
{
  assert(initObservedJoints.size() > 0);
  // Before using Eigen in a multi-threaded environment, we need to explicitly
  // call this (at least prior to Eigen 3.3)
  Eigen::initParallel();

  MarkerInitialization result;

  assert(
      params.jointCenters.cols() == 0
      || params.jointCenters.cols() == markerObservations.size());

  // 1. Initialize datastructures to hold the results
  result.poses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());
  result.poseScores = Eigen::VectorXs::Zero(markerObservations.size());

  std::cout << "Completing bilevel fit result using IK on the timesteps "
               "between our sampled indices..."
            << std::endl;

  std::vector<std::future<void>> blockFitFutures;

  // 2. Do a forward pass starting at each sample index and guessing forward to
  // the next index
  Eigen::MatrixXs forwardPoses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());
  Eigen::VectorXs forwardScores
      = Eigen::VectorXs::Ones(markerObservations.size())
        * std::numeric_limits<s_t>::max();
  for (int i = 0; i < solution->sampleIndices.size(); i++)
  {
    int thisIndex = solution->sampleIndices[i];
    int nextIndexExclusive = markerObservations.size();
    if (i < solution->sampleIndices.size() - 1)
    {
      nextIndexExclusive = solution->sampleIndices[i + 1];
    }
    // Don't extend though clip boundaries
    for (int j = thisIndex + 1; j < nextIndexExclusive; j++)
    {
      if (newClip[j])
      {
        nextIndexExclusive = j;
      }
    }

    int segmentLength = nextIndexExclusive - thisIndex;

    std::vector<std::map<std::string, Eigen::Vector3s>>
        segmentMarkerObservations;
    std::vector<Eigen::VectorXs> jointCenterArr;
    std::vector<Eigen::VectorXs> jointAxisArr;
    for (int i = 0; i < segmentLength; i++)
    {
      int index = thisIndex + i;
      segmentMarkerObservations.emplace_back();
      for (int i = 0; i < mMarkerNames.size(); i++)
      {
        std::string& name = mMarkerNames[i];
        if (mMarkerMap.count(name) && markerObservations[index].count(name))
        {
          segmentMarkerObservations[segmentMarkerObservations.size() - 1][name]
              = markerObservations[index].at(name);
        }
      }
      jointCenterArr.push_back(params.jointCenters.col(index));
      jointAxisArr.push_back(params.jointAxis.col(index));
    }

    /*
    fitTrajectory(
        this,
        solution->groupScales,
        solution->poses[i],
        segmentMarkerObservations,
        params.markerWeights,
        params.markerOffsets,
        params.joints,
        jointCenterArr,
        params.jointWeights,
        jointAxisArr,
        params.axisWeights,
        forwardPoses.block(
            0, thisIndex, mSkeleton->getNumDofs(), segmentLength),
        forwardScores.segment(thisIndex, segmentLength),
        false);
        */
    blockFitFutures.push_back(std::async(
        &MarkerFitter::fitTrajectory,
        this,
        solution->groupScales,
        solution->poses[i],
        segmentMarkerObservations,
        params.markerWeights,
        params.markerOffsets,
        params.joints,
        jointCenterArr,
        params.jointWeights,
        jointAxisArr,
        params.axisWeights,
        initObservedJoints,
        forwardPoses.block(
            0, thisIndex, mSkeleton->getNumDofs(), segmentLength),
        forwardScores.segment(thisIndex, segmentLength),
        false));
  }

  // 3. Do a backward pass starting at each sample index and guessing backward
  // to the previous index
  Eigen::MatrixXs backwardPoses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());
  Eigen::VectorXs backwardScores
      = Eigen::VectorXs::Ones(markerObservations.size())
        * std::numeric_limits<s_t>::max();
  for (int i = 0; i < solution->sampleIndices.size(); i++)
  {
    int thisIndex = solution->sampleIndices[i];
    int prevIndexExclusive = -1;
    if (i > 0)
    {
      prevIndexExclusive = solution->sampleIndices[i - 1];
    }
    // Don't extend though clip boundaries
    for (int j = thisIndex - 1; j > prevIndexExclusive; j--)
    {
      if (newClip[j])
      {
        prevIndexExclusive = j - 1;
      }
    }

    int segmentLength = thisIndex - prevIndexExclusive;

    std::vector<std::map<std::string, Eigen::Vector3s>>
        segmentMarkerObservations;
    std::vector<Eigen::VectorXs> jointCenterArr;
    std::vector<Eigen::VectorXs> jointAxisArr;
    for (int i = 0; i < segmentLength; i++)
    {
      int index = prevIndexExclusive + i + 1;
      segmentMarkerObservations.emplace_back();
      for (int i = 0; i < mMarkerNames.size(); i++)
      {
        std::string& name = mMarkerNames[i];
        if (mMarkerMap.count(name) && markerObservations[index].count(name))
        {
          segmentMarkerObservations[segmentMarkerObservations.size() - 1][name]
              = markerObservations[index].at(name);
        }
      }
      jointCenterArr.push_back(params.jointCenters.col(index));
      jointAxisArr.push_back(params.jointAxis.col(index));
    }

    /*
    fitTrajectory(
        this,
        solution->groupScales,
        solution->poses[i],
        segmentMarkerObservations,
        params.markerWeights,
        params.markerOffsets,
        params.joints,
        jointCenterArr,
        params.jointWeights,
        jointAxisArr,
        params.axisWeights,
        backwardPoses.block(
            0, thisIndex, mSkeleton->getNumDofs(), segmentLength),
        backwardScores.segment(thisIndex, segmentLength),
        true);
        */
    blockFitFutures.push_back(std::async(
        &MarkerFitter::fitTrajectory,
        this,
        solution->groupScales,
        solution->poses[i],
        segmentMarkerObservations,
        params.markerWeights,
        params.markerOffsets,
        params.joints,
        jointCenterArr,
        params.jointWeights,
        jointAxisArr,
        params.axisWeights,
        initObservedJoints,
        backwardPoses.block(
            0, prevIndexExclusive + 1, mSkeleton->getNumDofs(), segmentLength),
        backwardScores.segment(prevIndexExclusive + 1, segmentLength),
        true));
  }

  // 4. Wait for all the threads to finish
  for (int i = 0; i < blockFitFutures.size(); i++)
  {
    blockFitFutures[i].get();
  }

  // 5. Merge the pose guesses by taking the best guess from forward and
  // backwards passes
  for (int i = 0; i < markerObservations.size(); i++)
  {
    if (forwardScores(i) < backwardScores(i))
    {
      result.poses.col(i) = forwardPoses.col(i);
      result.poseScores(i) = forwardScores(i);
    }
    else
    {
      result.poses.col(i) = backwardPoses.col(i);
      result.poseScores(i) = backwardScores(i);
    }
    // Set the unobserved joints to default angles
    for (auto* joint : result.unobservedJoints)
    {
      if (joint->getNumDofs() > 0)
      {
        result.poses.col(i).segment(
            joint->getDof(0)->getIndexInSkeleton(), joint->getNumDofs())
            = joint->getInitialPositions();
      }
    }
  }
  // Overwrite with the poses from the solution
  /*
  for (int i = 0; i < solution->sampleIndices.size(); i++)
  {
    result.poses.col(solution->sampleIndices[i]) = solution->poses[i];
  }
  */

  result.groupScales = params.groupScales;
  result.joints = params.joints;
  result.jointCenters = params.jointCenters;
  result.jointsAdjacentMarkers = params.jointAdjacentMarkers;
  result.jointWeights = params.jointWeights;

  result.jointAxis = params.jointAxis;
  result.axisWeights = params.axisWeights;
  result.markerOffsets = solution->markerOffsets;

  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    std::string name = mMarkerNames[i];
    if (solution->markerOffsets.count(name))
    {
      result.updatedMarkerMap[name]
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              mMarkerMap[name].first,
              mMarkerMap[name].second + solution->markerOffsets[name]);
    }
  }

  // 6. Get the average marker error, and do a final marker offset to cancel it
  // as best we can

  std::map<std::string, Eigen::Vector3s> markerObservationsSum;
  std::map<std::string, int> markerNumObservations;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    std::string name = mMarkerNames[i];
    markerObservationsSum[name] = Eigen::Vector3s::Zero();
    markerNumObservations[name] = 0;
  }

  // 6.2. Run through every pose and accumulate error at that point

  Eigen::VectorXs originalPos = mSkeleton->getPositions();
  for (int i = 0; i < result.poses.cols(); i++)
  {
    mSkeleton->setPositions(result.poses.col(i));
    // Accumulate observations for the tracking markers
    for (auto& pair : markerObservations[i])
    {
      std::string name = pair.first;
      int index = mMarkerIndices[name];
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> marker = mMarkers[index];
      Eigen::Vector3s worldPosition = pair.second;
      Eigen::Vector3s localOffset
          = (marker.first->getWorldTransform().inverse() * worldPosition)
                .cwiseQuotient(marker.first->getScale());
      Eigen::Vector3s netOffset = localOffset - marker.second;

      markerObservationsSum[name] += netOffset;
      markerNumObservations[name]++;
    }
  }
  mSkeleton->setPositions(originalPos);

  // 6.3. Average out the result

  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    std::string name = mMarkerNames[i];
    if (mMarkerIsTracking[i] && !mPostprocessTrackingMarkerOffsets)
    {
      std::cout << "Skipping adjusting marker " << name
                << " because we are not adjusting tracking markers after the "
                   "main optimization"
                << std::endl;
    }
    else if (!mMarkerIsTracking[i] && !mPostprocessAnatomicalMarkerOffsets)
    {
      std::cout << "Skipping adjusting marker " << name
                << " because we are not adjusting anatomical markers after the "
                   "main optimization"
                << std::endl;
    }
    // Avoid divide-by-zero edge case
    else if (markerNumObservations[name] > 0)
    {
      result.markerOffsets[name]
          = markerObservationsSum[name] / markerNumObservations[name];
      Eigen::Vector3s updatedMarkers
          = mMarkerMap[name].second + result.markerOffsets[name];
      result.updatedMarkerMap[name]
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              mMarkerMap[name].first,
              mMarkerMap[name].second + result.markerOffsets[name]);
      std::cout << "Adjusting marker " << name << ": [" << updatedMarkers(0)
                << "," << updatedMarkers(1) << "," << updatedMarkers(2)
                << "] = original [" << mMarkerMap[name].second(0) << ","
                << mMarkerMap[name].second(1) << ","
                << mMarkerMap[name].second(2) << "] + offset ["
                << result.markerOffsets[name](0) << ","
                << result.markerOffsets[name](1) << ","
                << result.markerOffsets[name](2) << "]" << std::endl;
    }
    else
    {
      std::cout << "Skipping adjusting marker " << name
                << " with 0 observations" << std::endl;
    }
  }

  // 7. Do one final "polishing pass" on all the IK
  std::cout << "Done completing bilevel fit!" << std::endl;

  std::cout << "Running final smoothing IK" << std::endl;
  return smoothOutIK(markerObservations, newClip, result);

  // return result;
}

//==============================================================================
/// For the multi-trial pipeline, this takes our finished body scales and
/// marker offsets, and fine tunes on the IK initialized in the early joint
/// centering process.
MarkerInitialization MarkerFitter::fineTuneIK(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    int numBlocks,
    std::map<std::string, s_t> markerWeights,
    MarkerInitialization& initialization)
{
  MarkerInitialization result(initialization);
  result.observedJoints = initialization.observedJoints;
  result.unobservedJoints = initialization.unobservedJoints;

  assert(
      initialization.jointCenters.cols() == 0
      || initialization.jointCenters.cols() == markerObservations.size());
  assert(
      initialization.jointCenters.rows() == initialization.joints.size() * 3);

  // 0. Prep configuration variables we'll use for the rest of the algo
  // Upper bound the number of blocks at the number of observations
  if (numBlocks > markerObservations.size())
  {
    numBlocks = markerObservations.size();
  }
  int blockLen = markerObservations.size() / numBlocks;
  std::vector<std::string> anatomicalMarkerNames;
  for (int j = 0; j < mMarkerNames.size(); j++)
  {
    // if (!mMarkerIsTracking[j])
    anatomicalMarkerNames.push_back(mMarkerNames[j]);
  }

  // 1. Divide the marker observations into N sequential blocks.
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>> blocks;
  std::vector<Eigen::VectorXs> firstGuessPoses;
  std::vector<std::vector<Eigen::VectorXs>> jointCenterBlocks;
  std::vector<std::vector<Eigen::VectorXs>> jointAxisBlocks;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    // This means we've just started a new clip, so we need a new block
    if (i % blockLen == 0)
    {
      blocks.emplace_back();
      jointCenterBlocks.emplace_back();
      jointAxisBlocks.emplace_back();
      assert(
          initialization.poses.cols() == 0 || i < initialization.poses.cols());
      if (i < initialization.poses.cols())
      {
        firstGuessPoses.emplace_back(initialization.poses.col(i));
      }
      else
      {
        firstGuessPoses.emplace_back(mSkeleton->getPositions());
      }
    }
    // Append our state to whatever the current block is
    int mapIndex = blocks[blocks.size() - 1].size();
    blocks[blocks.size() - 1].emplace_back();
    for (std::string& marker : anatomicalMarkerNames)
    {
      if (markerObservations[i].count(marker) > 0)
      {
        assert(markerObservations[i].count(marker));
        blocks[blocks.size() - 1][mapIndex].emplace(
            marker, markerObservations[i].at(marker));
      }
    }
    assert(
        initialization.jointCenters.cols() == 0
        || initialization.jointCenters.cols() > i);
    if (initialization.jointCenters.cols() > i)
    {
      assert(
          initialization.jointCenters.rows()
          == initialization.joints.size() * 3);
      jointCenterBlocks[jointCenterBlocks.size() - 1].emplace_back(
          initialization.jointCenters.col(i));
    }
    else
    {
      assert(initialization.jointCenters.cols() == 0);
      jointCenterBlocks[jointCenterBlocks.size() - 1].emplace_back(
          Eigen::VectorXs::Zero(0));
    }
    if (initialization.jointAxis.cols() > i)
    {
      jointAxisBlocks[jointAxisBlocks.size() - 1].emplace_back(
          initialization.jointAxis.col(i));
    }
    else
    {
      jointAxisBlocks[jointAxisBlocks.size() - 1].emplace_back(
          Eigen::VectorXs::Zero(0));
    }
  }

  assert(blocks.size() >= numBlocks);
  numBlocks = blocks.size();

  std::vector<int> blockStartIndices;
  std::vector<int> blockSizeIndices;
  int cursor = 0;
  for (int i = 0; i < blocks.size(); i++)
  {
    blockStartIndices.push_back(cursor);
    blockSizeIndices.push_back(blocks[i].size());
    cursor += blocks[i].size();
  }

  // 3. Average the scalings for each block together
  assert(initialization.groupScales.size() > 0);
  result.groupScales = initialization.groupScales;

  // 4. Go through and run IK on each block
  result.poses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());
  result.poseScores = Eigen::VectorXs::Zero(markerObservations.size());

  std::vector<std::future<void>> blockFitFutures;
  for (int i = 0; i < numBlocks; i++)
  {
    std::cout << "Starting fit for whole block " << i << "/" << numBlocks
              << std::endl;

    blockFitFutures.push_back(std::async(
        &MarkerFitter::fitTrajectory,
        this,
        result.groupScales,
        firstGuessPoses[i],
        blocks[i],
        markerWeights,
        initialization.markerOffsets,
        initialization.joints,
        jointCenterBlocks[i],
        initialization.jointWeights,
        jointAxisBlocks[i],
        initialization.axisWeights,
        result.observedJoints,
        result.poses.block(
            0,
            blockStartIndices[i],
            mSkeleton->getNumDofs(),
            blockSizeIndices[i]),
        result.poseScores.segment(blockStartIndices[i], blockSizeIndices[i]),
        false));
  }
  for (int i = 0; i < numBlocks; i++)
  {
    blockFitFutures[i].get();
    std::cout << "Finished fit for whole block " << i << "/" << numBlocks
              << std::endl;
  }

  return result;
}

//==============================================================================
/// When our parallel-thread IK finishes, sometimes we can have a bit of
/// jitter in some of the joints, often around the wrists because the upper
/// body in general is poorly modelled in OpenSim. This will go through and
/// smooth out the frame-by-frame jitter.
MarkerInitialization MarkerFitter::smoothOutIK(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<bool>& newClip,
    MarkerInitialization& initialization)
{
  MarkerInitialization smoothed(initialization);
  smoothed.observedJoints = initialization.observedJoints;
  smoothed.unobservedJoints = initialization.unobservedJoints;

  std::vector<std::future<void>> ikFutures;

  // Smooth each trial
  int trialStart = 0;
  std::vector<int> trialStarts;
  std::vector<int> trialSizes;
  for (int t = 0; t < newClip.size(); t++)
  {
    if (newClip[t])
    {
      int size = (t - trialStart) + 1;
      if (size > 1)
      {
        trialStarts.push_back(trialStart);
        trialSizes.push_back(size);
      }
      trialStart = t;
    }
  }
  int size = newClip.size() - trialStart;
  if (size > 1)
  {
    trialStarts.push_back(trialStart);
    trialSizes.push_back(size);
  }

  (void)markerObservations;
  // for (int j = 0; j < trialStarts.size(); j++)
  // {
  //   std::cout << "Single-threaded fitting trial poses for trial " << j << "/"
  //             << trialStarts.size() << std::endl;

  //   std::shared_ptr<dynamics::Skeleton> skelClone =
  //   mSkeleton->cloneSkeleton(); int size = trialSizes[j]; int thisCursor =
  //   trialStarts[j];

  //   ikFutures.push_back(std::async([&, size, thisCursor, skelClone]() {
  //     for (int i = thisCursor; i < thisCursor + size; i++)
  //     {
  //       std::vector<std::string> observedMarkerNames;
  //       std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
  //           observedMarkers;

  //       for (auto& pair : markerObservations[i])
  //       {
  //         if (initialization.updatedMarkerMap.count(pair.first))
  //         {
  //           observedMarkerNames.push_back(pair.first);
  //           auto markerPair = initialization.updatedMarkerMap.at(pair.first);
  //           observedMarkers.push_back(std::make_pair(
  //               skelClone->getBodyNode(markerPair.first->getName()),
  //               markerPair.second));
  //         }
  //       }

  //       Eigen::VectorXs markerPoses
  //           = Eigen::VectorXs::Zero(observedMarkerNames.size() * 3);

  //       for (int k = 0; k < observedMarkerNames.size(); k++)
  //       {
  //         markerPoses.segment<3>(k * 3)
  //             = markerObservations[i].at(observedMarkerNames[k]);
  //       }

  //       const bool ignoreJointLimits = mIgnoreJointLimits;
  //       s_t finalLoss = math::solveIK(
  //           // Initial guess
  //           initialization.poses.col(i),
  //           skelClone->getPositionUpperLimits(),
  //           skelClone->getPositionLowerLimits(),
  //           // Output dimension
  //           observedMarkerNames.size() * 3,
  //           // Set positions
  //           [&skelClone, ignoreJointLimits](
  //               /* in*/ const Eigen::VectorXs pos, bool clamp) {
  //             skelClone->setPositions(pos);

  //             if (clamp && !ignoreJointLimits)
  //             {
  //               skelClone->clampPositionsToLimits();
  //             }

  //             // Return the clamped position
  //             return skelClone->getPositions();
  //           },
  //           [&observedMarkers, &markerPoses, &skelClone](
  //               /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
  //               /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
  //             diff.segment(0, markerPoses.size())
  //                 = skelClone->getMarkerWorldPositions(observedMarkers)
  //                   - markerPoses;
  //             jac.block(0, 0, jac.rows(), jac.cols())
  //                 =
  //                 skelClone->getMarkerWorldPositionsJacobianWrtJointPositions(
  //                     observedMarkers);
  //           },
  //           [](Eigen::Ref<Eigen::VectorXs> val) {
  //             (void)val;
  //             assert(false && "This should never be called");
  //           },
  //           math::IKConfig()
  //               .setMaxStepCount(50)
  //               .setConvergenceThreshold(1e-6)
  //               .setDontExitTranspose(true)
  //               .setLossLowerBound(1e-8)
  //               .setMaxRestarts(1)
  //               .setStartClamped(true)
  //               .setLogOutput(false));

  //       // 2.3. Record this outcome
  //       smoothed.poses.col(i) = skelClone->getPositions();
  //       smoothed.poseScores(i) = finalLoss;
  //     }
  //   }));
  // }

  // Join all the futures
  for (int i = 0; i < ikFutures.size(); i++)
  {
    ikFutures[i].get();
  }

  // Set the unobserved joints to default angles
  for (int i = 0; i < smoothed.poses.cols(); i++)
  {
    for (auto* joint : smoothed.unobservedJoints)
    {
      if (joint->getNumDofs() > 0)
      {
        smoothed.poses.col(i).segment(
            joint->getDof(0)->getIndexInSkeleton(), joint->getNumDofs())
            = joint->getInitialPositions();
      }
    }
  }

  for (int i = 0; i < trialStarts.size(); i++)
  {
    std::cout << "Smoothing trial poses for trial " << i << "/"
              << trialStarts.size() << std::endl;
    int start = trialStarts[i];
    int size = trialSizes[i];
    AccelerationSmoother smoother(size, 1.0, 0.001);
    smoother.smooth(
        smoothed.poses.block(0, start, smoothed.poses.rows(), size));
  }

  return smoothed;
}

//==============================================================================
/// This sets the map of IMUs to their locations on body segments
void MarkerFitter::setImuMap(dynamics::SensorMap imuMap)
{
  mImuMap = imuMap;
  mImuNames.clear();
  mImus.clear();
  for (auto& pair : mImuMap)
  {
    mImuNames.push_back(pair.first);
    mImus.push_back(pair.second);
  }
}

//==============================================================================
dynamics::SensorMap MarkerFitter::getImuMap()
{
  return mImuMap;
}

//==============================================================================
/// This returns (a copy of) the list of IMUs to their locations on body
/// segments
std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>
MarkerFitter::getImuList()
{
  return mImus;
}

//==============================================================================
/// This returns (a copy of) the list of IMU names, corresponding to the
/// getImuList() IMUs
std::vector<std::string> MarkerFitter::getImuNames()
{
  return mImuNames;
}

//==============================================================================
/// This calculates the best-fit rotation for the IMUs that were passed in, to
/// fit the target data.
void MarkerFitter::rotateIMUs(
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    const std::vector<bool>& newClip,
    MarkerInitialization& initialization,
    s_t dt)
{
  std::vector<std::string> accNameList;
  std::vector<int> accIndexList;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> accList;
  std::vector<std::string> gyroNameList;
  std::vector<int> gyroIndexList;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> gyroList;

  std::vector<Eigen::Matrix3s> imuUnnormalizedRotations;

  for (int imu = 0; imu < mImuNames.size(); imu++)
  {
    std::string imuName = mImuNames[imu];
    if (accObservations[0].count(imuName))
    {
      accNameList.push_back(imuName);
      accList.push_back(mImus[imu]);
      accIndexList.push_back(imu);
    }
    if (gyroObservations[0].count(imuName))
    {
      gyroNameList.push_back(imuName);
      gyroList.push_back(mImus[imu]);
      gyroIndexList.push_back(imu);
    }

    imuUnnormalizedRotations.push_back(Eigen::Matrix3s::Identity());
  }

  for (int t = 0; t < newClip.size(); t++)
  {
    if (t > 0 && t < newClip.size() - 1 && !newClip[t] && !newClip[t + 1])
    {
      Eigen::VectorXs q = initialization.poses.col(t);
      Eigen::VectorXs dq
          = mSkeleton->getPositionDifferences(
                initialization.poses.col(t), initialization.poses.col(t - 1))
            / dt;
      Eigen::VectorXs ddq
          = (mSkeleton->getPositionDifferences(
                 initialization.poses.col(t + 1), initialization.poses.col(t))
             - mSkeleton->getPositionDifferences(
                 initialization.poses.col(t), initialization.poses.col(t - 1)))
            / (dt * dt);
      mSkeleton->setPositions(q);
      mSkeleton->setVelocities(dq);
      mSkeleton->setAccelerations(ddq);

      Eigen::VectorXs simulatedAccReadings
          = mSkeleton->getAccelerometerReadings(accList);
      for (int i = 0; i < accList.size(); i++)
      {
        Eigen::Vector3s acc = accObservations[t].at(accNameList[i]);
        Eigen::Vector3s simulatedAcc = simulatedAccReadings.segment<3>(3 * i);
        // The idea here is that we're adding a matrix that measures an input
        // vector `x` along `simulatedAcc` (by doing `simulatedAcc.dot(x)`) and
        // then projects that value along `acc`. You can think of this like a
        // soft rotation from `simulatedAcc` to `acc`. By summing a bunch of
        // these together, we get an "unnormalized rotation matrix". Then when
        // we normalize later with SVD, we get the best-fit rotation.
        //
        // You can read more about this on the internet if you google the "SVD
        // method for Wahba's problem".
        imuUnnormalizedRotations[accIndexList[i]]
            += acc * simulatedAcc.transpose();
      }

      Eigen::VectorXs simulatedGyroReadings
          = mSkeleton->getGyroReadings(gyroList);
      for (int i = 0; i < gyroList.size(); i++)
      {
        Eigen::Vector3s gyro = gyroObservations[t].at(gyroNameList[i]);
        Eigen::Vector3s simulatedGyro = simulatedGyroReadings.segment<3>(3 * i);
        // See the comment above for the accelerometer case for an explanation.
        imuUnnormalizedRotations[gyroIndexList[i]]
            += gyro * simulatedGyro.transpose();
      }
    }
  }

  for (int i = 0; i < mImuNames.size(); i++)
  {
    Eigen::JacobiSVD<Eigen::Matrix3s> svd(
        imuUnnormalizedRotations[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3s U = svd.matrixU();
    Eigen::Matrix3s V = svd.matrixV();
    Eigen::Matrix3s R = U * V.transpose();
    // We want the rotation to be a right-handed (standard) rotation, so if the
    // determinant is negative that's no good, and we have to flip one of the
    // axis.
    if (R.determinant() < 0)
    {
      U.col(2) *= -1;
      R = U * V.transpose();
    }
    mImus[i].second.linear() = mImus[i].second.linear() * R.transpose();
    mImuMap[mImuNames[i]].second = mImus[i].second;
  }
}

//==============================================================================
/// This calculates the RMS of accelerometer data, to see how well the motion
/// matches the accelerometers.
s_t MarkerFitter::measureAccelerometerRMS(
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<bool>& newClip,
    MarkerInitialization& initialization,
    s_t dt)
{
  std::vector<std::string> accNameList;
  std::vector<int> accIndexList;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> accList;

  for (int imu = 0; imu < mImuNames.size(); imu++)
  {
    std::string imuName = mImuNames[imu];
    if (accObservations[0].count(imuName))
    {
      accNameList.push_back(imuName);
      accList.push_back(mImus[imu]);
      accIndexList.push_back(imu);
    }
  }

  s_t error = 0.0;
  int numFramesCounted = 0;

  for (int t = 0; t < newClip.size(); t++)
  {
    if (t > 0 && t < newClip.size() - 1 && !newClip[t] && !newClip[t + 1])
    {
      Eigen::VectorXs q = initialization.poses.col(t);
      Eigen::VectorXs dq
          = mSkeleton->getPositionDifferences(
                initialization.poses.col(t), initialization.poses.col(t - 1))
            / dt;
      Eigen::VectorXs ddq
          = (mSkeleton->getPositionDifferences(
                 initialization.poses.col(t + 1), initialization.poses.col(t))
             - mSkeleton->getPositionDifferences(
                 initialization.poses.col(t), initialization.poses.col(t - 1)))
            / (dt * dt);
      mSkeleton->setPositions(q);
      mSkeleton->setVelocities(dq);
      mSkeleton->setAccelerations(ddq);

      Eigen::VectorXs simulatedAccReadings
          = mSkeleton->getAccelerometerReadings(accList);
      for (int i = 0; i < accList.size(); i++)
      {
        Eigen::Vector3s acc = accObservations[t].at(accNameList[i]);
        Eigen::Vector3s simulatedAcc = simulatedAccReadings.segment<3>(3 * i);
        error += (acc - simulatedAcc).norm();
      }
      numFramesCounted++;
    }
  }

  return error / (numFramesCounted * accList.size());
}

//==============================================================================
/// This calculates the RMS of gyro data, to see how well the motion matches
/// the gyroscopes.
s_t MarkerFitter::measureGyroRMS(
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    const std::vector<bool>& newClip,
    MarkerInitialization& initialization,
    s_t dt)
{
  std::vector<std::string> gyroNameList;
  std::vector<int> gyroIndexList;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> gyroList;

  for (int imu = 0; imu < mImuNames.size(); imu++)
  {
    std::string imuName = mImuNames[imu];
    if (gyroObservations[0].count(imuName))
    {
      gyroNameList.push_back(imuName);
      gyroList.push_back(mImus[imu]);
      gyroIndexList.push_back(imu);
    }
  }

  s_t error = 0.0;
  int numFramesCounted = 0;

  for (int t = 0; t < newClip.size(); t++)
  {
    if (t > 0 && t < newClip.size() - 1 && !newClip[t] && !newClip[t + 1])
    {
      Eigen::VectorXs q = initialization.poses.col(t);
      Eigen::VectorXs dq
          = mSkeleton->getPositionDifferences(
                initialization.poses.col(t), initialization.poses.col(t - 1))
            / dt;
      Eigen::VectorXs ddq
          = (mSkeleton->getPositionDifferences(
                 initialization.poses.col(t + 1), initialization.poses.col(t))
             - mSkeleton->getPositionDifferences(
                 initialization.poses.col(t), initialization.poses.col(t - 1)))
            / (dt * dt);
      mSkeleton->setPositions(q);
      mSkeleton->setVelocities(dq);
      mSkeleton->setAccelerations(ddq);

      Eigen::VectorXs simulatedGyroReadings
          = mSkeleton->getGyroReadings(gyroList);
      for (int i = 0; i < gyroList.size(); i++)
      {
        Eigen::Vector3s gyro = gyroObservations[t].at(gyroNameList[i]);
        Eigen::Vector3s simulatedGyro = simulatedGyroReadings.segment<3>(3 * i);
        error += (gyro - simulatedGyro).norm();
      }
      numFramesCounted++;
    }
  }

  return error / (numFramesCounted * gyroList.size());
}

//==============================================================================
/// This returns an object that can be used to run an optimization to fine
/// tune the joint angles to match IMU data.
std::shared_ptr<biomechanics::IMUFineTuneProblem>
MarkerFitter::getIMUFineTuneProblem(
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    MarkerInitialization& initialization,
    s_t dt,
    int start,
    int end)
{
  return std::make_shared<IMUFineTuneProblem>(
      this,
      accObservations,
      gyroObservations,
      markerObservations,
      dt,
      initialization,
      start,
      end);
}

//==============================================================================
/// This adjusts the joint accelerations to match the IMU data, and offsets the
/// IMU locations + rotations.
MarkerInitialization MarkerFitter::fineTuneWithIMU(
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<bool>& newClip,
    MarkerInitialization& initialization,
    s_t dt,
    s_t weightAccs,
    s_t weightGyros,
    s_t weightMarkers,
    s_t regularizePoses,
    bool useIPOPT,
    int iterations,
    int lbfgsMemory)
{
  std::vector<std::pair<int, int>> clipBoundaries;
  int start = 0;
  for (int i = 1; i < newClip.size(); i++)
  {
    if (newClip[i])
    {
      clipBoundaries.emplace_back(start, i);
      start = i;
    }
  }
  clipBoundaries.emplace_back(start, newClip.size());

  for (auto& pair : clipBoundaries)
  {
    if (useIPOPT)
    {
      // Before using Eigen in a multi-threaded environment, we need to
      // explicitly call this (at least prior to Eigen 3.3)
      Eigen::initParallel();

      // Create an instance of the IpoptApplication
      //
      // We are using the factory, since this allows us to compile this
      // example with an Ipopt Windows DLL
      SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

      // Change some options
      // Note: The following choices are only examples, they might not be
      //       suitable for your optimization problem.
      app->Options()->SetNumericValue("tol", static_cast<double>(mTolerance));
      app->Options()->SetStringValue(
          "linear_solver",
          "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps,
                    // custom

      app->Options()->SetStringValue(
          "hessian_approximation", "limited-memory"); // limited-memory, exacty

      /*
      app->Options()->SetStringValue(
          "scaling_method", "none"); // none, gradient-based
      */

      app->Options()->SetIntegerValue("max_iter", iterations);

      // Disable LBFGS history
      app->Options()->SetIntegerValue(
          "limited_memory_max_history", lbfgsMemory);

      // Just for debugging
      if (mCheckDerivatives)
      {
        app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
        app->Options()->SetStringValue("derivative_test", "first-order");
        app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);
      }

      if (mPrintFrequency > 0)
      {
        app->Options()->SetIntegerValue(
            "print_frequency_iter", mPrintFrequency);
      }
      else
      {
        app->Options()->SetIntegerValue(
            "print_frequency_iter", std::numeric_limits<int>::infinity());
      }
      if (mSilenceOutput)
      {
        app->Options()->SetIntegerValue("print_level", 0);
      }
      if (mDisableLinesearch)
      {
        app->Options()->SetIntegerValue("max_soc", 0);
        app->Options()->SetStringValue("accept_every_trial_step", "yes");
      }
      app->Options()->SetIntegerValue("watchdog_shortened_iter_trigger", 0);

      // Initialize the IpoptApplication and process the options
      Ipopt::ApplicationReturnStatus status;
      status = app->Initialize();
      if (status != Solve_Succeeded)
      {
        std::cout << std::endl
                  << std::endl
                  << "*** Error during initialization!" << std::endl;
      }
      else
      {
        // This will automatically free the problem object when finished,
        // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will
        // crash. If you try to leave `problem` on the stack, you'll get invalid
        // free exceptions when IPOpt attempts to free it.
        IMUFineTuneProblem* problem = new IMUFineTuneProblem(
            this,
            accObservations,
            gyroObservations,
            markerObservations,
            dt,
            initialization,
            pair.first,
            pair.second);
        problem->setWeightAccs(weightAccs);
        problem->setWeightGyros(weightGyros);
        problem->setWeightMarkers(weightMarkers);
        problem->setRegularizePoses(regularizePoses);

        SmartPtr<IMUFineTuneProblem> problemPtr(problem);
        status = app->OptimizeTNLP(problemPtr);

        if (status == Solve_Succeeded)
        {
          // Retrieve some statistics about the solve
          Index iter_count = app->Statistics()->IterationCount();
          std::cout << std::endl
                    << std::endl
                    << "*** The problem solved in " << iter_count
                    << " iterations!" << std::endl;

          Number final_obj = app->Statistics()->FinalObjective();
          std::cout << std::endl
                    << std::endl
                    << "*** The final value of the objective function is "
                    << final_obj << '.' << std::endl;
        }
      }
    }
    else
    {
      IMUFineTuneProblem problem(
          this,
          accObservations,
          gyroObservations,
          markerObservations,
          dt,
          initialization,
          pair.first,
          pair.second);
      problem.setWeightAccs(weightAccs);
      problem.setWeightGyros(weightGyros);
      problem.setWeightMarkers(weightMarkers);
      problem.setRegularizePoses(regularizePoses);

      // Run SGD

      Eigen::VectorXs x = problem.flatten();
      s_t decay = 0.95;
      Eigen::VectorXs adagradSum = Eigen::VectorXs::Ones(x.size());
      problem.unflatten(x);
      s_t loss = problem.getLoss();
      s_t learningRate = 0.5;

      for (int i = 0; i < iterations; i++)
      {
        Eigen::VectorXs grad = problem.getGrad();
        adagradSum
            = (1.0 - decay) * grad.cwiseProduct(grad) + decay * adagradSum;

        while (true)
        {
          Eigen::VectorXs adagradRoot = adagradSum.cwiseSqrt();
          Eigen::VectorXs proposedX
              = x - grad.cwiseQuotient(adagradRoot) * learningRate;
          problem.unflatten(proposedX);
          s_t newLoss = problem.getLoss();
          std::cout << "Iteration " << i << " [lr=" << learningRate
                    << "]: " << newLoss << " (prev best=" << loss
                    << ", diff=" << (newLoss - loss) << ")" << std::endl;
          if (newLoss < loss)
          {
            x = proposedX;
            loss = newLoss;
            learningRate *= 1.5;
            break;
          }
          else
          {
            learningRate *= 0.5;
            // If our learning rate has gotten too small, update x and continue
            // anyways
            if (learningRate < 1e-12)
            {
              x = proposedX;
              break;
            }
          }
        }
      }

      // Write the results back to the initialization

      Eigen::MatrixXs& foundPoses = problem.getPoses();
      initialization.poses.block(
          0, pair.first, initialization.poses.rows(), pair.second - pair.first)
          = foundPoses;
    }
  }

  return initialization;
}

//==============================================================================
/// This finds an initial guess for the body scales and poses, holding
/// anatomical marker offsets at 0, that we can use for downstream tasks.
///
/// This can multithread over `numBlocks` independent sets of problems.
MarkerInitialization MarkerFitter::getInitialization(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    const std::vector<bool>& newClip,
    InitialMarkerFitParams params)
{
  // Before using Eigen in a multi-threaded environment, we need to explicitly
  // call this (at least prior to Eigen 3.3)
  Eigen::initParallel();

  MarkerInitialization result;

  assert(
      params.jointCenters.cols() == 0
      || params.jointCenters.cols() == markerObservations.size());
  assert(params.jointCenters.rows() == params.joints.size() * 3);

  // 0. Prep configuration variables we'll use for the rest of the algo
  int numBlocks = params.numBlocks;
  // Upper bound the number of blocks at the number of observations
  if (numBlocks > markerObservations.size())
  {
    numBlocks = markerObservations.size();
  }
  int blockLen = markerObservations.size() / numBlocks;

  // Construct the observedMarkers list, for markers that appear in both the
  // data and the model
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (auto& pair : markerObservations[t])
    {
      if (mMarkerMap.count(pair.first) > 0)
      {
        if (std::find(
                result.observedMarkers.begin(),
                result.observedMarkers.end(),
                pair.first)
            == result.observedMarkers.end())
        {
          result.observedMarkers.push_back(pair.first);
        }
      }
    }
  }

  for (auto& pair : mMarkerMap)
  {
    if (std::find(
            result.observedMarkers.begin(),
            result.observedMarkers.end(),
            pair.first)
        != result.observedMarkers.end())
    {
      // Traverse up the parent list looking for all the joints that effect this
      // node
      dynamics::BodyNode* cursor = pair.second.first;
      while (cursor != nullptr)
      {
        dynamics::Joint* parentJoint = cursor->getParentJoint();
        assert(parentJoint != nullptr);
        if (std::find(
                result.observedJoints.begin(),
                result.observedJoints.end(),
                parentJoint)
            == result.observedJoints.end())
        {
          result.observedJoints.push_back(parentJoint);
        }
        cursor = parentJoint->getParentBodyNode();
      }
    }
  }

  for (int i = 0; i < mSkeleton->getNumJoints(); i++)
  {
    if (std::find(
            result.observedJoints.begin(),
            result.observedJoints.end(),
            mSkeleton->getJoint(i))
        == result.observedJoints.end())
    {
      result.unobservedJoints.push_back(mSkeleton->getJoint(i));
      std::cout << "Unobserved Joint: " << mSkeleton->getJoint(i)->getName()
                << std::endl;
    }
  }
  std::cout << "Size of observed joints: " << result.observedJoints.size()
            << std::endl;
  std::cout << "Size of unobserved joints: " << result.unobservedJoints.size()
            << std::endl;
  assert(
      result.observedJoints.size() + result.unobservedJoints.size()
      == mSkeleton->getNumJoints());

  std::cout << " joints: " << params.joints.size() << std::endl;
  if (params.joints.size() == 0)
  {
    // 1. If we don't have any joint center information yet, don't just run
    // non-convex IK and hope for the best. Instead, run the least-squares IK
    // initializer.

    std::map<std::string, bool> markerIsAnatomical;
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      markerIsAnatomical[mMarkerNames[i]] = !mMarkerIsTracking[i];
    }

    IKInitializer initializer(
        mSkeleton,
        mMarkerMap,
        markerIsAnatomical,
        markerObservations,
        newClip,
        mHeightPrior);
    initializer.runFullPipeline();

    // For now, we're just going to use the poses and scales, and then leave the
    // joint centers to the non-convex optimization pipeline. We could (and
    // maybe should?) use the joint centers that the least-squares algo finds
    // directly. That would require some refactoring, since those joint centers
    // are for "stacked joints" which may be multiple joints in the skeleton at
    // once.
    result.poses = Eigen::MatrixXs::Zero(
        mSkeleton->getNumDofs(), markerObservations.size());
    result.groupScales = initializer.getGroupScales();
    for (int i = 0; i < markerObservations.size(); i++)
    {
      result.poses.col(i) = initializer.getPoses()[i];
    }
  }
  else
  {
    // 1. Divide the marker observations into N sequential blocks.
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>> blocks;
    std::vector<Eigen::VectorXs> firstGuessPoses;
    std::vector<std::vector<Eigen::VectorXs>> jointCenterBlocks;
    std::vector<std::vector<Eigen::VectorXs>> jointAxisBlocks;
    for (int i = 0; i < markerObservations.size(); i++)
    {
      // This means we've just started a new clip, so we need a new block
      if ((i % blockLen == 0) || newClip[i])
      {
        blocks.emplace_back();
        jointCenterBlocks.emplace_back();
        jointAxisBlocks.emplace_back();
        assert(params.initPoses.cols() == 0 || i < params.initPoses.cols());
        if (i < params.initPoses.cols())
        {
          firstGuessPoses.emplace_back(params.initPoses.col(i));
        }
        else
        {
          firstGuessPoses.emplace_back(mSkeleton->getPositions());
        }
      }
      // Append our state to whatever the current block is
      int mapIndex = blocks[blocks.size() - 1].size();
      blocks[blocks.size() - 1].emplace_back();
      for (std::string& marker : mMarkerNames)
      {
        if (markerObservations[i].count(marker) > 0)
        {
          assert(markerObservations[i].count(marker));
          blocks[blocks.size() - 1][mapIndex].emplace(
              marker, markerObservations[i].at(marker));
        }
      }
      assert(params.jointCenters.cols() == 0 || params.jointCenters.cols() > i);
      if (params.jointCenters.cols() > i)
      {
        assert(params.jointCenters.rows() == params.joints.size() * 3);
        jointCenterBlocks[jointCenterBlocks.size() - 1].push_back(
            params.jointCenters.col(i));
      }
      else
      {
        assert(params.jointCenters.cols() == 0);
        jointCenterBlocks[jointCenterBlocks.size() - 1].push_back(
            Eigen::VectorXs::Zero(0));
      }
      if (params.jointAxis.cols() > i)
      {
        jointAxisBlocks[jointAxisBlocks.size() - 1].push_back(
            params.jointAxis.col(i));
      }
      else
      {
        jointAxisBlocks[jointAxisBlocks.size() - 1].push_back(
            Eigen::VectorXs::Zero(0));
      }
    }

    assert(blocks.size() >= numBlocks);
    numBlocks = blocks.size();

    std::vector<int> blockStartIndices;
    std::vector<int> blockSizeIndices;
    int cursor = 0;
    for (int i = 0; i < blocks.size(); i++)
    {
      blockStartIndices.push_back(cursor);
      blockSizeIndices.push_back(blocks[i].size());
      cursor += blocks[i].size();
    }

    // Divide everything up into trials, instead of blocks.

    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>> trials;
    std::vector<std::vector<Eigen::VectorXs>> jointCenterTrials;
    std::vector<std::vector<Eigen::VectorXs>> jointAxisTrials;
    for (int i = 0; i < markerObservations.size(); i++)
    {
      // This means we've just started a new clip, so we need a new block
      if (newClip[i] || i == 0)
      {
        trials.emplace_back();
        jointCenterTrials.emplace_back();
        jointAxisTrials.emplace_back();
      }
      // Append our state to whatever the current block is
      int mapIndex = trials[trials.size() - 1].size();
      trials[trials.size() - 1].emplace_back();
      for (std::string& marker : mMarkerNames)
      {
        if (markerObservations[i].count(marker) > 0)
        {
          assert(markerObservations[i].count(marker));
          trials[trials.size() - 1][mapIndex].emplace(
              marker, markerObservations[i].at(marker));
        }
      }
      assert(params.jointCenters.cols() == 0 || params.jointCenters.cols() > i);
      if (params.jointCenters.cols() > i)
      {
        assert(params.jointCenters.rows() == params.joints.size() * 3);
        jointCenterTrials[jointCenterTrials.size() - 1].emplace_back(
            params.jointCenters.col(i));
      }
      else
      {
        assert(params.jointCenters.cols() == 0);
        jointCenterTrials[jointCenterTrials.size() - 1].emplace_back(
            Eigen::VectorXs::Zero(0));
      }
      if (params.jointAxis.cols() > i)
      {
        jointAxisTrials[jointAxisTrials.size() - 1].emplace_back(
            params.jointAxis.col(i));
      }
      else
      {
        jointAxisTrials[jointAxisTrials.size() - 1].emplace_back(
            Eigen::VectorXs::Zero(0));
      }
    }

    int numTrials = trials.size();

    std::vector<int> trialStartIndices;
    std::vector<int> trialSizeIndices;
    int trialCursor = 0;
    for (int i = 0; i < trials.size(); i++)
    {
      trialStartIndices.push_back(trialCursor);
      trialSizeIndices.push_back(trials[i].size());
      trialCursor += trials[i].size();
    }

    // 2. Find IK+scaling for the beginning of each block independently
    std::vector<ScaleAndFitResult> posesAndScales;
    std::vector<std::future<ScaleAndFitResult>> posesAndScalesFutures;

    if (params.groupScales.size() > 0)
    {
      mSkeleton->setGroupScales(params.groupScales);
    }
    for (int i = 0; i < numBlocks; i++)
    {
      if (params.dontRescaleBodies)
      {
        std::cout << "Starting initial fit for first timestep of block " << i
                  << "/" << numBlocks << std::endl;
      }
      else
      {
        std::cout << "Starting initial scale+fit for first timestep of block "
                  << i << "/" << numBlocks << std::endl;
      }
      /*
      posesAndScales.push_back(scaleAndFit(
          this,
          blocks[i][0],
          firstGuessPoses[i],
          params.markerWeights,
          params.markerOffsets,
          params.joints,
          jointCenterBlocks[i][0],
          params.jointWeights,
          jointAxisBlocks[i][0],
          params.axisWeights,
          result.observedJoints,
          params.dontRescaleBodies,
          i,
          true,
          true));
      exit(1);
      */
      posesAndScalesFutures.push_back(std::async(
          &MarkerFitter::scaleAndFit,
          this,
          blocks[i][0],
          firstGuessPoses[i],
          params.markerWeights,
          params.markerOffsets,
          params.joints,
          jointCenterBlocks[i][0],
          params.jointWeights,
          jointAxisBlocks[i][0],
          params.axisWeights,
          result.observedJoints,
          params.dontRescaleBodies,
          i,
          false,
          false));
    }

    for (int i = 0; i < numBlocks; i++)
    {
      ScaleAndFitResult result = posesAndScalesFutures[i].get();

      // Do some error checking on the results
      if (!mIgnoreJointLimits)
      {
        Eigen::VectorXs posLowerLimits = mSkeleton->getPositionLowerLimits();
        Eigen::VectorXs posUpperLimits = mSkeleton->getPositionUpperLimits();
        for (int j = 0; j < mSkeleton->getNumDofs(); j++)
        {
          s_t posJ = result.pose(j);
          if (posJ < posLowerLimits(j))
          {
            std::cout << "DOF " << j << " (" << mSkeleton->getDof(j)->getName()
                      << ") below lower limit! " << posJ << " < "
                      << posLowerLimits(j) << std::endl;
          }
          if (posJ > posUpperLimits(j))
          {
            std::cout << "DOF " << j << " (" << mSkeleton->getDof(j)->getName()
                      << ") above upper limit! " << posJ << " > "
                      << posUpperLimits(j) << std::endl;
          }
        }
      }
      // End: error checking

      posesAndScales.push_back(result);
      if (params.dontRescaleBodies)
      {
        std::cout << "Finished initial fit for first timestep of block " << i
                  << "/" << numBlocks << std::endl;
      }
      else
      {
        std::cout << "Finished initial scale+fit for first timestep of block "
                  << i << "/" << numBlocks << std::endl;
      }
    }

    // 3. Average the scalings for each block together
    if (params.dontRescaleBodies && params.groupScales.size() > 0)
    {
      result.groupScales = params.groupScales;
      mSkeleton->setGroupScales(result.groupScales);
    }
    else
    {
      Eigen::VectorXs averageGroupScales
          = Eigen::VectorXs::Zero(mSkeleton->getGroupScaleDim());
      s_t totalWeight = 0.0;
      for (int i = 0; i < numBlocks; i++)
      {
        s_t weight = 1.0 / posesAndScales[i].score;
        averageGroupScales += weight * posesAndScales[i].scale;
        totalWeight += weight;
      }
      averageGroupScales /= totalWeight;
      result.groupScales = averageGroupScales;
      mSkeleton->setGroupScales(result.groupScales);
    }

    // 4. Go through and run IK on each block
    result.poses = Eigen::MatrixXs::Zero(
        mSkeleton->getNumDofs(), markerObservations.size());
    result.poseScores = Eigen::VectorXs::Zero(markerObservations.size());

    for (int i = 0; i < numBlocks; i++)
    {
      result.poses.col(blockStartIndices[i]) = posesAndScales[i].pose;
    }
    std::vector<bool> shouldProcessBlock;
    std::vector<s_t> lastBlockLoss;
    for (int i = 0; i < numBlocks; i++)
    {
      shouldProcessBlock.push_back(true);
      lastBlockLoss.push_back(std::numeric_limits<double>::infinity());
    }

    // We re-run parallel processing over all the blocks until convergence, or
    // at most numBlocks times
    for (int k = 0; k < params.numIKTries; k++)
    {
      std::vector<std::future<void>> blockFitFutures;
      for (int i = 0; i < numBlocks; i++)
      {
        std::cout << "Starting fit for whole block " << i << "/" << numBlocks
                  << std::endl;

        if (shouldProcessBlock[i])
        {
          blockFitFutures.push_back(std::async(
              &MarkerFitter::fitTrajectory,
              this,
              result.groupScales,
              result.poses.col(blockStartIndices[i]),
              blocks[i],
              params.markerWeights,
              params.markerOffsets,
              params.joints,
              jointCenterBlocks[i],
              params.jointWeights,
              jointAxisBlocks[i],
              params.axisWeights,
              result.observedJoints,
              result.poses.block(
                  0,
                  blockStartIndices[i],
                  mSkeleton->getNumDofs(),
                  blockSizeIndices[i]),
              result.poseScores.segment(
                  blockStartIndices[i], blockSizeIndices[i]),
              false));
        }
        else
        {
          blockFitFutures.push_back(std::async([]() {}));
        }
      }
      for (int i = 0; i < numBlocks; i++)
      {
        blockFitFutures[i].get();
        std::cout << "Finished fit for whole block " << i << "/" << numBlocks
                  << std::endl;
      }

      bool foundGap = false;
      for (int i = 1; i < numBlocks; i++)
      {
        s_t blockStartLoss = result.poseScores(blockStartIndices[i]);
        if (blockStartLoss > result.poseScores(blockStartIndices[i] - 1) * 3
            && blockStartLoss > 0.01 && !newClip[blockStartIndices[i]])
        {
          s_t lastBlockStartLoss = lastBlockLoss[i];
          if (abs(blockStartLoss - lastBlockStartLoss) < 1e-3)
          {
            std::cout << "Bad block start at " << i << ": prev block ended at "
                      << result.poseScores(blockStartIndices[i] - 1)
                      << ", but this block started at "
                      << result.poseScores(blockStartIndices[i])
                      << ". However, no improvement since last iteration, so "
                         "we're not going to retry this block."
                      << std::endl;
          }
          else
          {
            lastBlockLoss[i] = blockStartLoss;
            shouldProcessBlock[i] = true;
            std::cout << "Found bad block start at " << i
                      << ": prev block ended at "
                      << result.poseScores(blockStartIndices[i] - 1)
                      << ", but this block started at "
                      << result.poseScores(blockStartIndices[i]) << std::endl;
            result.poses(blockStartIndices[i])
                = result.poses(blockStartIndices[i] - 1);
            foundGap = true;
          }
        }
        else
        {
          shouldProcessBlock[i] = false;
        }
      }

      /*
      Eigen::MatrixXs blockVsScore
          = Eigen::MatrixXs::Zero(result.poseScores.size(), 2);
      for (int i = 0; i < numBlocks; i++)
      {
        blockVsScore.block(blockStartIndices[i], 0, blockSizeIndices[i], 1)
            .setConstant(i);
      }
      blockVsScore.col(1) = result.poseScores;
      std::cout << "Block vs score: " << blockVsScore << std::endl;
      */

      if (!foundGap)
      {
        break;
      }
    }

    bool shouldResmooth = false;
    for (int t = 1; t < result.poses.cols(); t++)
    {
      if (!newClip[t])
      {
        s_t diff = (result.poses.col(t) - result.poses.col(t - 1)).norm();
        if (diff > 0.1)
        {
          shouldResmooth = true;
          break;
        }
      }
    }

    if (shouldResmooth)
    {
      std::cout << "Fixing jitters by running a round of single-threaded IK"
                << std::endl;

      std::vector<std::future<void>> trialFitFutures;
      for (int i = 0; i < numTrials; i++)
      {
        std::cout << "Starting fit for whole trial " << i << "/" << numTrials
                  << std::endl;

        trialFitFutures.push_back(std::async(
            &MarkerFitter::fitTrajectory,
            this,
            result.groupScales,
            result.poses.col(trialStartIndices[i]),
            trials[i],
            params.markerWeights,
            params.markerOffsets,
            params.joints,
            jointCenterTrials[i],
            params.jointWeights,
            jointAxisTrials[i],
            params.axisWeights,
            result.observedJoints,
            result.poses.block(
                0,
                trialStartIndices[i],
                mSkeleton->getNumDofs(),
                trialSizeIndices[i]),
            result.poseScores.segment(
                trialStartIndices[i], trialSizeIndices[i]),
            false));
      }
      for (int i = 0; i < numTrials; i++)
      {
        trialFitFutures[i].get();
        std::cout << "Finished fit for whole trial " << i << "/" << numTrials
                  << std::endl;
      }
    }
  }

  // 5. Find the local offsets for the tracking markers as a simple
  // average

  if (params.dontMoveMarkers == false)
  {
    bool offsetAnatomicalMarkersToo = false;

    // 5.1. Initialize empty maps to accumulate sums into

    std::map<std::string, Eigen::Vector3s> trackingMarkerObservationsSum;
    std::map<std::string, int> trackingMarkerNumObservations;
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      if (mMarkerIsTracking[i] || offsetAnatomicalMarkersToo)
      {
        std::string name = mMarkerNames[i];
        trackingMarkerObservationsSum[name] = Eigen::Vector3s::Zero();
        trackingMarkerNumObservations[name] = 0;
      }
    }

    // 5.2. Run through every pose in the solve, and accumulate error at that
    // point

    Eigen::VectorXs originalPos = mSkeleton->getPositions();
    for (int i = 0; i < result.poses.cols(); i++)
    {
      mSkeleton->setPositions(result.poses.col(i));
      // Accumulate observations for the tracking markers
      for (auto& pair : markerObservations[i])
      {
        std::string name = pair.first;
        int index = mMarkerIndices[name];
        if (mMarkerIsTracking[index] || offsetAnatomicalMarkersToo)
        {
          std::pair<dynamics::BodyNode*, Eigen::Vector3s> trackingMarker
              = mMarkers[index];
          Eigen::Vector3s worldPosition = pair.second;
          Eigen::Vector3s localOffset
              = (trackingMarker.first->getWorldTransform().inverse()
                 * worldPosition)
                    .cwiseQuotient(trackingMarker.first->getScale());
          Eigen::Vector3s netOffset = localOffset - trackingMarker.second;

          trackingMarkerObservationsSum[name] += netOffset;
          trackingMarkerNumObservations[name]++;
        }
      }
    }
    mSkeleton->setPositions(originalPos);

    // 5.3. Average out the result

    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      std::string name = mMarkerNames[i];
      if (params.markerOffsets.count(name) > 0)
      {
        assert(params.markerOffsets.count(name));
        if (mMarkerIsTracking[i])
        {
          result.markerOffsets[name] = params.markerOffsets.at(name);
          result.updatedMarkerMap[name]
              = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
                  mMarkerMap[name].first,
                  mMarkerMap[name].second + params.markerOffsets.at(name));
        }
      }
      else
      {
        if (mMarkerIsTracking[i] || offsetAnatomicalMarkersToo)
        {
          // Avoid divide-by-zero edge case
          if (trackingMarkerNumObservations[name] == 0)
          {
            result.markerOffsets[name] = Eigen::Vector3s::Zero();
          }
          else
          {
            result.markerOffsets[name] = trackingMarkerObservationsSum[name]
                                         / trackingMarkerNumObservations[name];
          }
          result.updatedMarkerMap[name]
              = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
                  mMarkerMap[name].first,
                  mMarkerMap[name].second + result.markerOffsets[name]);
        }
        else
        {
          result.markerOffsets[name] = Eigen::Vector3s::Zero();
          result.updatedMarkerMap[name] = mMarkerMap[name];
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      std::string name = mMarkerNames[i];
      if (params.markerOffsets.count(name) > 0)
      {
        result.markerOffsets[name] = params.markerOffsets.at(name);
        result.updatedMarkerMap[name]
            = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
                mMarkerMap[name].first,
                mMarkerMap[name].second + params.markerOffsets.at(name));
      }
      else
      {
        result.markerOffsets[name] = Eigen::Vector3s::Zero();
        result.updatedMarkerMap[name] = mMarkerMap[name];
      }
    }
  }

  // 6. Find a good initial value for the root position for the static pose,
  // given all the other configuration we've found.

  if (mStaticTrialEnabled)
  {
    Eigen::VectorXs originalPositions = mSkeleton->getPositions();

    // 6.1. Construct the most up-to-date list of markers for the static trial
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        staticTrialMarkers;
    for (int i = 0; i < mStaticTrialMarkerNames.size(); i++)
    {
      bool foundMarker = false;
      for (int j = 0; j < mMarkerNames.size(); j++)
      {
        if (mStaticTrialMarkerNames[i] == mMarkerNames[j])
        {
          staticTrialMarkers.push_back(
              result.updatedMarkerMap[mMarkerNames[j]]);
          foundMarker = true;
          break;
        }
      }
      assert(foundMarker);
      if (!foundMarker)
      {
        std::cout << "ERROR: static trial marker " << mStaticTrialMarkerNames[i]
                  << " did not appear in the markerset. This is a runtime "
                     "error, because it should have been filtered out in "
                     "MarkerFitter::setStaticTrial(). Exit 1"
                  << std::endl;
        exit(1);
      }
    }

    Eigen::VectorXs rootPos = mStaticTrialPose.head(6);
    math::solveIK(
        rootPos,
        mSkeleton->getPositionUpperLimits().head(6),
        mSkeleton->getPositionLowerLimits().head(6),
        (staticTrialMarkers.size() * 3),
        // Set positions
        [this](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          (void)clamp;
          // Complete the root position to the rest of the skeleton
          Eigen::VectorXs fullPos = mStaticTrialPose;
          fullPos.head(6) = pos;
          mSkeleton->setPositions(fullPos);
          // Return the root position
          return pos;
        },
        // Compute the Jacobian
        [this, &staticTrialMarkers](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff.segment(0, staticTrialMarkers.size())
              = mSkeleton->getMarkerWorldPositions(staticTrialMarkers)
                - mStaticTrialMarkerPositions;
          assert(jac.cols() == 6);
          assert(jac.rows() == (staticTrialMarkers.size() * 3));
          jac.setZero();
          jac.block(0, 0, staticTrialMarkers.size() * 3, 6)
              = mSkeleton
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        staticTrialMarkers)
                    .block(0, 0, staticTrialMarkers.size() * 3, 6);
        },
        // Generate a random restart position
        [this](Eigen::Ref<Eigen::VectorXs> val) {
          val = mSkeleton->getRandomPose().head(6);
        },
        math::IKConfig()
            .setMaxStepCount(150)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLogOutput(true));
    result.staticPoseRoot = mSkeleton->getPositions().head<6>();
    mSkeleton->setPositions(originalPositions);
  }

  result.joints = params.joints;
  result.jointCenters = params.jointCenters;
  result.jointWeights = params.jointWeights;
  result.jointsAdjacentMarkers = params.jointAdjacentMarkers;

  result.jointAxis = params.jointAxis;
  result.axisWeights = params.axisWeights;

  return result;
}

//==============================================================================
/// For when we need to subsample trials, because the user uploaded a bunch of
/// trials and it would overwhelm the bilevel optimizer, we can subsample them
/// down to a smaller number of trials by ranking on this quantity. This tells
/// us how much the markers for a trial move with respect to each other over
/// the course of the trial. More motion is better.
s_t MarkerFitter::computeMarkerDistanceMatrixVariability(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  std::vector<std::string> markerNames;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (auto& pair : markerObservations[t])
    {
      if (std::find(markerNames.begin(), markerNames.end(), pair.first)
          == markerNames.end())
      {
        markerNames.push_back(pair.first);
      }
    }
  }

  // 1. Go through and find the mean distances

  Eigen::MatrixXs pairMeans
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  Eigen::MatrixXs pairObservationCounts
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (int i = 0; i < markerNames.size() - 1; i++)
    {
      if (markerObservations[t].count(markerNames[i]))
      {
        for (int j = i + 1; j < markerNames.size(); j++)
        {
          if (markerObservations[t].count(markerNames[j]))
          {
            s_t dist = (markerObservations[t].at(markerNames[i])
                        - markerObservations[t].at(markerNames[j]))
                           .norm();
            pairMeans(i, j) += dist;
            pairObservationCounts(i, j) += 1;
          }
        }
      }
    }
  }

  pairMeans = pairMeans.cwiseQuotient(pairObservationCounts);

  // 2. Compute the variance

  Eigen::MatrixXs pairVariance
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (int i = 0; i < markerNames.size() - 1; i++)
    {
      if (markerObservations[t].count(markerNames[i]))
      {
        for (int j = i + 1; j < markerNames.size(); j++)
        {
          if (markerObservations[t].count(markerNames[j]))
          {
            s_t dist = (markerObservations[t].at(markerNames[i])
                        - markerObservations[t].at(markerNames[j]))
                           .norm();
            s_t diff = dist - pairMeans(i, j);
            pairVariance(i, j) += diff * diff;
          }
        }
      }
    }
  }

  pairVariance.cwiseQuotient(pairObservationCounts);

  // 3. Go through and compute the sum normalized RMSE

  s_t sum = 0.0;
  for (int i = 0; i < markerNames.size(); i++)
  {
    for (int j = 0; j < markerNames.size(); j++)
    {
      if (pairObservationCounts(i, j) > 0)
      {
        sum += sqrt(pairVariance(i, j)) / pairMeans(i, j);
      }
    }
  }

  return sum;
}

//==============================================================================
/// This computes the IK diff for joint positions, given a bunch of weighted
/// joint centers and also a bunch of weighted joint axis.
void MarkerFitter::computeJointIKDiff(
    Eigen::Ref<Eigen::VectorXs> diff,
    const Eigen::VectorXs& jointPoses,
    const Eigen::VectorXs& jointCenters,
    const Eigen::VectorXs& jointWeights,
    const Eigen::VectorXs& jointAxis,
    const Eigen::VectorXs& axisWeights)
{
  diff = jointPoses - jointCenters;
  for (int i = 0; i < jointWeights.size(); i++)
  {
    diff.segment<3>(i * 3) *= jointWeights(i);
  }
  for (int i = 0; i < axisWeights.size(); i++)
  {
    Eigen::Vector3s axisCenter = jointAxis.segment<3>(i * 6);
    Eigen::Vector3s axisDir = jointAxis.segment<3>(i * 6 + 3).normalized();
    Eigen::Vector3s actualJointPos = jointPoses.segment<3>(i * 3);

    // Subtract out any component parallel to the axis
    Eigen::Vector3s jointDiff = actualJointPos - axisCenter;
    jointDiff -= jointDiff.dot(axisDir) * axisDir;

    diff.segment<3>(i * 3) += jointDiff * axisWeights(i);
  }
}

//==============================================================================
/// This takes a Jacobian of joint world positions (with respect to anything),
/// and rescales and reshapes to reflect the weights on joint and axis losses,
/// as well as the direction for axis losses.
void MarkerFitter::rescaleIKJacobianForWeightsAndAxis(
    Eigen::Ref<Eigen::MatrixXs> jac,
    const Eigen::VectorXs& jointWeights,
    const Eigen::VectorXs& jointAxis,
    const Eigen::VectorXs& axisWeights)
{
  for (int i = 0; i < jointWeights.size(); i++)
  {
    Eigen::Vector3s axisDir = jointAxis.segment<3>(i * 6 + 3).normalized();
    Eigen::Matrix3s axisDirT
        = Eigen::Matrix3s::Identity() - axisDir * axisDir.transpose();
    Eigen::Matrix3s overallT = (jointWeights(i) * Eigen::Matrix3s::Identity())
                               + (axisWeights(i) * axisDirT);

    for (int j = 0; j < jac.cols(); j++)
    {
      jac.block<3, 1>(i * 3, j) = overallT * jac.block<3, 1>(i * 3, j);
    }
  }
}

//==============================================================================
/// This scales the skeleton and IK fits to the marker observations. It
/// returns a pair, with (pose, group scales) from the fit.
ScaleAndFitResult MarkerFitter::scaleAndFit(
    const MarkerFitter* fitter,
    std::map<std::string, Eigen::Vector3s> markerObservations,
    Eigen::VectorXs firstGuessPose,
    std::map<std::string, s_t> markerWeights,
    std::map<std::string, Eigen::Vector3s> markerOffsets,
    std::vector<dynamics::Joint*> joints,
    Eigen::VectorXs jointCenters,
    Eigen::VectorXs jointWeights,
    Eigen::VectorXs jointAxis,
    Eigen::VectorXs axisWeights,
    std::vector<dynamics::Joint*> initObservedJoints,
    bool dontScale,
    int debugIndex,
    bool debug,
    bool saveToGUI)
{
  assert(initObservedJoints.size() > 0);

  // 0. To make this thread safe, we're going to clone the fitter skeleton
  std::shared_ptr<dynamics::Skeleton> skeleton;
  {
    const std::lock_guard<std::mutex> lock(
        *(const_cast<std::mutex*>(&fitter->mGlobalLock)));
    skeleton = fitter->mSkeleton->clone();
  }
  skeleton->setPositions(firstGuessPose);

  // Restrict the scale bounds more when we have less information. Don't allow
  // scaling that's too far beyond the default height for any individual body
  // when we have no joint center information to help constrain the problem
  // (e.g. during our first initialization pass).
  bool useMoreRestrictiveScaleBounds = joints.size() == 0;
  std::vector<bool> originalUniformScaling;
  if (useMoreRestrictiveScaleBounds)
  {
    for (auto& group : skeleton->getBodyScaleGroups())
    {
      originalUniformScaling.push_back(group.uniformScaling);
      skeleton->setScaleGroupUniformScaling(group.nodes[0], true);
    }
    skeleton->setGroupScales(
        Eigen::VectorXs::Ones(skeleton->getGroupScaleDim()));
  }
  s_t defaultHeight
      = skeleton->getHeight(Eigen::VectorXs::Zero(skeleton->getNumDofs()));
  s_t defaultScale = fitter->mHeightPrior / defaultHeight;
  s_t defaultScaleLowerBound = defaultScale * 0.9;
  s_t defaultScaleUpperBound = defaultScale * 1.1;

  // 0.1. Translate over the observedJoints array to the cloned skeleton
  std::vector<dynamics::Joint*> observedJoints;
  for (auto joint : initObservedJoints)
  {
    observedJoints.push_back(skeleton->getJoint(joint->getName()));
  }

  // Because we have no initialization, we should do the slow thing and
  // try really hard to fit the IK well

  // 1. We're going to enforce the joint limits in the Eulerian space, but do
  // our actual gradient descient in SO3 space so we can avoid gimbal
  // lock. That requires a bit of careful book-keeping.

  // 1.1. Convert the skeleton to have any Euler joints as ball joints
  std::shared_ptr<dynamics::Skeleton> skeletonBallJoints
      = skeleton->convertSkeletonToBallJoints();
  std::vector<dynamics::Joint*> jointsForSkeletonBallJoints;
  for (auto joint : joints)
  {
    jointsForSkeletonBallJoints.push_back(
        skeletonBallJoints->getJoint(joint->getName()));
  }

  // 1.2. Linearize the marker names and marker observations
  Eigen::VectorXs markerPoses
      = Eigen::VectorXs::Zero(markerObservations.size() * 3);
  Eigen::VectorXs markerWeightsVector
      = Eigen::VectorXs::Ones(markerObservations.size());
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerVector;
  std::vector<std::string> outputNames;
  for (std::pair<std::string, Eigen::Vector3s> pair : markerObservations)
  {
    markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
    if (markerWeights.count(pair.first))
    {
      assert(markerWeights.count(pair.first));
      markerWeightsVector(markerVector.size()) = markerWeights.at(pair.first);
    }
    else
    {
      markerWeightsVector(markerVector.size())
          = fitter->mMarkerIsTracking.at(fitter->mMarkerIndices.at(pair.first))
                ? fitter->mTrackingMarkerDefaultWeight
                : fitter->mAnatomicalMarkerDefaultWeight;
    }
    assert(fitter->mMarkerMap.count(pair.first));
    const std::pair<dynamics::BodyNode*, Eigen::Vector3s>& originalMarker
        = fitter->mMarkerMap.at(pair.first);
    Eigen::Vector3s offset = Eigen::Vector3s::Zero();
    if (markerOffsets.count(pair.first))
    {
      assert(markerOffsets.count(pair.first));
      offset = markerOffsets.at(pair.first);
    }
    markerVector.emplace_back(
        skeletonBallJoints->getBodyNode(originalMarker.first->getName()),
        originalMarker.second + offset);
    Eigen::Vector3s markerPos = originalMarker.second + offset;
    std::string markerPrefix = "marker " + originalMarker.first->getName()
                               + " (" + std::to_string((double)markerPos(0))
                               + "," + std::to_string((double)markerPos(1))
                               + "," + std::to_string((double)markerPos(2))
                               + ")";
    outputNames.push_back(markerPrefix + " X");
    outputNames.push_back(markerPrefix + " Y");
    outputNames.push_back(markerPrefix + " Z");
  }
  for (int i = 0; i < joints.size(); i++)
  {
    std::string jointPrefix = "joint " + joints[i]->getName();
    outputNames.push_back(jointPrefix + " X");
    outputNames.push_back(jointPrefix + " Y");
    outputNames.push_back(jointPrefix + " Z");
  }

  // 1.3. Get ready to enforce virtual joint springs that will try to drive
  // "virtual spring joints" to zero
  std::vector<int> virtualSpringDofs;
  std::vector<s_t> virtualSpringWeights;
  for (auto* joint : skeletonBallJoints->getJoints())
  {
    if (fitter->mJointVirtualSpringRegularizerWeight.count(joint->getName()))
    {
      for (int i = 0; i < joint->getNumDofs(); i++)
      {
        virtualSpringDofs.push_back(joint->getDof(i)->getIndexInSkeleton());
        virtualSpringWeights.push_back(
            fitter->mJointVirtualSpringRegularizerWeight.at(joint->getName()));
        outputNames.push_back("Spring dof " + joint->getDof(i)->getName());
      }
    }
  }

  std::vector<std::string> inputNames;
  for (int i = 0; i < skeletonBallJoints->getNumDofs(); i++)
  {
    inputNames.push_back("dof " + skeletonBallJoints->getDof(i)->getName());
  }

  server::GUIRecording server;
  if (saveToGUI)
  {
    server.setFramesPerSecond(10);
  }

  ScaleAndFitResult result;
  if (dontScale)
  {
    // 1.3. Calculate problem size
    int problemDim = skeletonBallJoints->getNumDofs();

    // 1.4. Set our initial guess for IK to whatever the current pose of the
    // skeleton is
    Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);
    initialPos = skeletonBallJoints->convertPositionsToBallSpace(
        skeletonBallJoints->getPositions());

    if (debug)
    {
      std::cout << "Beginning IK (no scaling) for block " << debugIndex
                << std::endl;
      std::cout << "Joint weights: " << jointWeights << std::endl;
      std::cout << "Axis weights: " << axisWeights << std::endl;
      std::cout << "First guess pose: " << firstGuessPose << std::endl;
#ifndef NDEBUG
      std::vector<double> timestamps;
      timestamps.push_back(0.0);
      std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
      markerTimesteps.push_back(markerObservations);
      OpenSimParser::saveTRC(
          "./replicateBadIK.trc", timestamps, markerTimesteps);
#endif
    }

    const bool ignoreJointLimits = fitter->mIgnoreJointLimits;
    // 2. Actually solve the IK
    result.score = math::solveIK(
        initialPos,
        skeletonBallJoints->getPositionUpperLimits(),
        skeletonBallJoints->getPositionLowerLimits(),
        (markerObservations.size() * 3) + (joints.size() * 3)
            + virtualSpringDofs.size(),
        // Set positions
        [skeletonBallJoints,
         skeleton,
         saveToGUI,
         &server,
         markerPoses,
         markerVector,
         jointsForSkeletonBallJoints,
         jointCenters,
         jointAxis,
         ignoreJointLimits](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(pos);
          if (clamp)
          {
            // 1. Map the position back into eulerian space
            skeleton->setPositions(
                skeleton->convertPositionsFromBallSpace(pos));
            if (!ignoreJointLimits)
            {
              // 2. Clamp the position to limits
              skeleton->clampPositionsToLimits();
              // 3. Map the position back into SO3 space
              skeletonBallJoints->setPositions(
                  skeleton->convertPositionsToBallSpace(
                      skeleton->getPositions()));
            }
          }
          if (saveToGUI)
          {
            server.renderSkeleton(skeletonBallJoints);

            Eigen::VectorXs currentMarkerPoses
                = skeletonBallJoints->getMarkerWorldPositions(markerVector);
            for (int i = 0; i < markerVector.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              points.push_back(currentMarkerPoses.segment<3>(i * 3));
              points.push_back(markerPoses.segment<3>(i * 3));
              server.createLine(
                  "marker_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(1, 0, 0, 1));
            }
            Eigen::VectorXs currentJointPoses
                = skeletonBallJoints->getJointWorldPositions(
                    jointsForSkeletonBallJoints);
            for (int i = 0; i < jointsForSkeletonBallJoints.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              points.push_back(currentJointPoses.segment<3>(i * 3));
              points.push_back(jointCenters.segment<3>(i * 3));
              server.createLine(
                  "joint_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(0, 1, 0, 1));
            }
            for (int i = 0; i < jointsForSkeletonBallJoints.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              points.push_back(currentJointPoses.segment<3>(i * 3));
              points.push_back(jointAxis.segment<3>(i * 6));
              server.createLine(
                  "joint_axis_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(0, 0, 1, 1));
            }

            server.saveFrame();
          }

          // Return the clamped position
          return skeletonBallJoints->getPositions();
        },
        // Compute the Jacobian
        [skeletonBallJoints,
         markerPoses,
         markerVector,
         markerWeightsVector,
         jointsForSkeletonBallJoints,
         jointCenters,
         jointWeights,
         jointAxis,
         axisWeights,
         virtualSpringDofs,
         virtualSpringWeights](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          diff.segment(0, markerPoses.size())
              = skeletonBallJoints->getMarkerWorldPositions(markerVector)
                - markerPoses;
          Eigen::VectorXs jointPoses
              = skeletonBallJoints->getJointWorldPositions(
                  jointsForSkeletonBallJoints);
          computeJointIKDiff(
              diff.segment(markerPoses.size(), jointCenters.size()),
              jointPoses,
              jointCenters,
              jointWeights,
              jointAxis,
              axisWeights);

          assert(jac.cols() == skeletonBallJoints->getNumDofs());
          assert(
              jac.rows()
              == (markerVector.size() * 3)
                     + (jointsForSkeletonBallJoints.size() * 3));
          jac.setZero();
          jac.block(
              0, 0, markerVector.size() * 3, skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        markerVector);
          jac.block(
              markerVector.size() * 3,
              0,
              jointsForSkeletonBallJoints.size() * 3,
              skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtJointPositions(
                        jointsForSkeletonBallJoints);
          for (int i = 0; i < markerWeightsVector.size(); i++)
          {
            diff.segment<3>(i * 3) *= markerWeightsVector(i);
            jac.block(i * 3, 0, 3, jac.cols()) *= markerWeightsVector(i);
          }
          rescaleIKJacobianForWeightsAndAxis(
              jac.block(
                  markerVector.size() * 3,
                  0,
                  jointsForSkeletonBallJoints.size() * 3,
                  skeletonBallJoints->getNumDofs()),
              jointWeights,
              jointAxis,
              axisWeights);
          for (int i = 0; i < virtualSpringDofs.size(); i++)
          {
            diff(markerPoses.size() + jointPoses.size() + i)
                = skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * virtualSpringWeights[i];
            jac(markerPoses.size() + jointPoses.size() + i,
                virtualSpringDofs[i])
                = 2 * skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * virtualSpringWeights[i];
          }
        },
        // Generate a random restart position
        [&skeleton, &observedJoints, markerPoses](
            Eigen::Ref<Eigen::VectorXs> val) {
          val = skeleton->convertPositionsToBallSpace(
              skeleton->getRandomPoseForJoints(observedJoints));

          // Set the root translation to within a fairly narrow range of the
          // average marker cloud
          Eigen::Vector3s avgMarkerPos = Eigen::Vector3s::Zero();
          for (int i = 0; i < markerPoses.size() / 3; i++)
          {
            avgMarkerPos += markerPoses.segment<3>(i * 3);
          }
          avgMarkerPos /= (s_t)(markerPoses.size() / 3);
          val.segment<3>(3) = avgMarkerPos + (Eigen::Vector3s::Random() * 0.2);
        },
        math::IKConfig()
            .setMaxStepCount(350)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(fitter->mInitialIKSatisfactoryLoss)
            .setMaxRestarts(fitter->mInitialIKMaxRestarts)
            .setLogOutput(debug)
            .setInputNames(inputNames)
            .setOutputNames(outputNames));
  }
  else
  {
    // 1.3. Calculate problem size
    int problemDim = skeletonBallJoints->getNumDofs()
                     + skeletonBallJoints->getGroupScaleDim();

    // Record debug names for the scale groups
    for (int i = 0; i < skeletonBallJoints->getNumScaleGroups(); i++)
    {
      auto group = skeletonBallJoints->getBodyScaleGroup(i);
      std::string prefix = "scale group [";
      for (auto* body : group.nodes)
      {
        prefix += " " + body->getName();
      }
      prefix += " ] ";
      if (group.uniformScaling)
      {
        inputNames.push_back(prefix + " scale");
      }
      else
      {
        inputNames.push_back(prefix + " X");
        inputNames.push_back(prefix + " Y");
        inputNames.push_back(prefix + " Z");
      }
    }

    // Get the neutral pose of the skeleton where we'll measure the height
    Eigen::VectorXs skeletonHeightMeasurementPose
        = Eigen::VectorXs::Zero(skeleton->getNumDofs());
    Eigen::VectorXs skeletonBallJointsHeightMeasurementPose
        = skeleton->convertPositionsToBallSpace(skeletonHeightMeasurementPose);

    // 1.4. Set our initial guess for IK to whatever the current pose of the
    // skeleton is
    Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);
    Eigen::VectorXs lowerBound = Eigen::VectorXs::Zero(problemDim);
    Eigen::VectorXs upperBound = Eigen::VectorXs::Zero(problemDim);
    initialPos.segment(0, skeletonBallJoints->getNumDofs())
        = skeletonBallJoints->convertPositionsToBallSpace(
            skeleton->getPositions());
    lowerBound.segment(0, skeletonBallJoints->getNumDofs())
        = skeletonBallJoints->getPositionLowerLimits();
    upperBound.segment(0, skeletonBallJoints->getNumDofs())
        = skeletonBallJoints->getPositionUpperLimits();
    initialPos.segment(
        skeletonBallJoints->getNumDofs(),
        skeletonBallJoints->getGroupScaleDim())
        = skeletonBallJoints->getGroupScales();
    if (useMoreRestrictiveScaleBounds)
    {
      lowerBound
          .segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim())
          .setConstant(defaultScaleLowerBound);
      upperBound
          .segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim())
          .setConstant(defaultScaleUpperBound);
    }
    else
    {
      lowerBound.segment(
          skeletonBallJoints->getNumDofs(),
          skeletonBallJoints->getGroupScaleDim())
          = skeletonBallJoints->getGroupScalesLowerBound();
      upperBound.segment(
          skeletonBallJoints->getNumDofs(),
          skeletonBallJoints->getGroupScaleDim())
          = skeletonBallJoints->getGroupScalesUpperBound();
    }

    if (debug)
    {
      std::cout << "Beginning IK (with scaling) for block " << debugIndex
                << std::endl;
      std::cout << "Joint weights: " << jointWeights << std::endl;
      std::cout << "Axis weights: " << axisWeights << std::endl;
      std::cout << "First guess pose: " << firstGuessPose << std::endl;
#ifndef NDEBUG
      std::vector<double> timestamps;
      timestamps.push_back(0.0);
      std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
      markerTimesteps.push_back(markerObservations);
      OpenSimParser::saveTRC(
          "./replicateBadIK.trc", timestamps, markerTimesteps);
#endif
    }

    // The extra dimension is for the anthropometric penalty term
    int outputDim = (markerObservations.size() * 3) + (joints.size() * 3)
                    + virtualSpringDofs.size();
    if (fitter->mAnthropometrics != nullptr)
    {
      outputDim += 1;
      outputNames.push_back("Sigmoid of log(anthropometric prior)");
    }
    bool useHeightPrior = fitter->mHeightPriorWeight > 0;
    if (fitter->mSkeleton->getHeight(
            Eigen::VectorXs::Zero(fitter->mSkeleton->getNumDofs()))
        == 0.0)
    {
      useHeightPrior = false;
    }
    if (useHeightPrior)
    {
      outputDim += 1;
      outputNames.push_back("Height prior");
    }

    const bool ignoreJointLimits = fitter->mIgnoreJointLimits;
    // 2. Actually solve the IK
    result.score = math::solveIK(
        initialPos,
        upperBound,
        lowerBound,
        outputDim,
        // Set positions
        [skeletonBallJoints,
         skeleton,
         saveToGUI,
         &server,
         markerPoses,
         markerVector,
         markerWeightsVector,
         jointsForSkeletonBallJoints,
         jointCenters,
         jointAxis,
         useMoreRestrictiveScaleBounds,
         defaultScaleLowerBound,
         defaultScaleUpperBound,
         ignoreJointLimits](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(
              pos.segment(0, skeletonBallJoints->getNumDofs()));

          /*
          // Verify the translation is lossless
          Eigen::VectorXs eulerPos = skeleton->convertPositionsFromBallSpace(
              pos.segment(0, skeletonBallJoints->getNumDofs()));
          Eigen::VectorXs recovered
              = skeleton->convertPositionsToBallSpace(eulerPos);
          Eigen::VectorXs bodyPoses = Eigen::VectorXs::Zero(
              skeletonBallJoints->getNumBodyNodes() * 3);
          for (int i = 0; i < skeletonBallJoints->getNumBodyNodes(); i++)
          {
            bodyPoses.segment<3>(i * 3) = skeletonBallJoints->getBodyNode(i)
                                              ->getWorldTransform()
                                              .translation();
          }
          skeletonBallJoints->setPositions(recovered);
          Eigen::VectorXs recoveredBodyPoses = Eigen::VectorXs::Zero(
              skeletonBallJoints->getNumBodyNodes() * 3);
          for (int i = 0; i < skeletonBallJoints->getNumBodyNodes(); i++)
          {
            recoveredBodyPoses.segment<3>(i * 3)
                = skeletonBallJoints->getBodyNode(i)
                      ->getWorldTransform()
                      .translation();
          }
          if ((recoveredBodyPoses - bodyPoses).norm() > 1e-3)
          {
            std::cout << "!!!!!! Got a recovery error of "
                      << (recoveredBodyPoses - bodyPoses).norm() << std::endl;
            std::cout << "Eigen::VectorXs pos = Eigen::VectorXs(" << pos.size()
                      << ");" << std::endl;
            std::cout << "pos << ";
            for (int i = 0; i < pos.size(); i++)
            {
              if (i > 0)
                std::cout << ", ";
              std::cout << pos(i);
            }
            std::cout << ";" << std::endl;
          }
          // End: Verify the translation is lossless
          */

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            skeleton->setPositions(skeleton->convertPositionsFromBallSpace(
                pos.segment(0, skeletonBallJoints->getNumDofs())));
            if (!ignoreJointLimits)
            {
              // 2. Clamp the position to limits
              skeleton->clampPositionsToLimits();
              // 3. Map the position back into SO3 space
              skeletonBallJoints->setPositions(
                  skeleton->convertPositionsToBallSpace(
                      skeleton->getPositions()));
            }
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = skeletonBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = skeletonBallJoints->getGroupScalesLowerBound();
          if (useMoreRestrictiveScaleBounds)
          {
            scalesLowerBound.setConstant(defaultScaleLowerBound);
            scalesUpperBound.setConstant(defaultScaleUpperBound);
          }
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          skeleton->setGroupScales(newScales);
          skeletonBallJoints->setGroupScales(newScales);

          if (saveToGUI)
          {
            server.renderSkeleton(skeletonBallJoints);

            Eigen::VectorXs currentMarkerPoses
                = skeletonBallJoints->getMarkerWorldPositions(markerVector);
            for (int i = 0; i < markerVector.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              s_t markerWeightPercentage
                  = markerWeightsVector(i) / markerWeightsVector.maxCoeff();
              points.push_back(currentMarkerPoses.segment<3>(i * 3));
              points.push_back(markerPoses.segment<3>(i * 3));
              server.createLine(
                  "marker_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(1, 0, 0, markerWeightPercentage));
            }
            Eigen::VectorXs currentJointPoses
                = skeletonBallJoints->getJointWorldPositions(
                    jointsForSkeletonBallJoints);
            for (int i = 0; i < jointsForSkeletonBallJoints.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              points.push_back(currentJointPoses.segment<3>(i * 3));
              points.push_back(jointCenters.segment<3>(i * 3));
              server.createLine(
                  "joint_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(0, 1, 0, 1));
            }
            for (int i = 0; i < jointsForSkeletonBallJoints.size(); i++)
            {
              std::vector<Eigen::Vector3s> points;
              points.push_back(currentJointPoses.segment<3>(i * 3));
              points.push_back(jointAxis.segment<3>(i * 6));
              server.createLine(
                  "joint_axis_" + std::to_string(i),
                  points,
                  Eigen::Vector4s(0, 0, 1, 1));
            }

            server.saveFrame();
          }

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, skeletonBallJoints->getNumDofs())
              = skeletonBallJoints->getPositions();
          clampedPos.segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim())
              = newScales;
          return clampedPos;
        },
        // Compute the Jacobian
        [skeletonBallJoints,
         fitter,
         markerPoses,
         markerVector,
         markerWeightsVector,
         jointsForSkeletonBallJoints,
         skeletonBallJointsHeightMeasurementPose,
         jointCenters,
         jointWeights,
         jointAxis,
         axisWeights,
         outputDim,
         virtualSpringDofs,
         virtualSpringWeights,
         useHeightPrior](
            /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
            /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
          /// Compute the output diff (error)

          diff.segment(0, markerPoses.size())
              = skeletonBallJoints->getMarkerWorldPositions(markerVector)
                - markerPoses;
          Eigen::VectorXs jointPoses
              = skeletonBallJoints->getJointWorldPositions(
                  jointsForSkeletonBallJoints);
          computeJointIKDiff(
              diff.segment(markerPoses.size(), jointCenters.size()),
              jointPoses,
              jointCenters,
              jointWeights,
              jointAxis,
              axisWeights);

          assert(
              jac.cols()
              == skeletonBallJoints->getNumDofs()
                     + skeletonBallJoints->getGroupScaleDim());
          (void)outputDim;
          assert(jac.rows() == outputDim);

          jac.setZero();

          int outputCursor = markerPoses.size() + jointCenters.size();
          for (int i = 0; i < virtualSpringDofs.size(); i++)
          {
            diff(outputCursor)
                = skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * virtualSpringWeights[i];
            jac(outputCursor, virtualSpringDofs[i])
                = 2 * skeletonBallJoints->getPosition(virtualSpringDofs[i])
                  * virtualSpringWeights[i];
            outputCursor++;
          }

          s_t alphaScale = 0.0;
          s_t expNegLogPdf = 0.0;
          if (fitter->mAnthropometrics != nullptr)
          {
            s_t logPdf
                = fitter->mAnthropometrics->getLogPDF(skeletonBallJoints);
            // Do the negative exponent of the logPdf.
            // The benefit here is that as the PDF gets to be a
            // larger and larger value (more probable),
            // this penalty will fall to zero.
            alphaScale = 0.01;
            expNegLogPdf = exp(-alphaScale * logPdf);
            diff(outputCursor) = expNegLogPdf;
            outputCursor++;
          }

          if (useHeightPrior)
          {
            s_t height = skeletonBallJoints->getHeight(
                skeletonBallJointsHeightMeasurementPose);
            s_t heightDiff = height - fitter->mHeightPrior;
            diff(outputCursor) = 10 * heightDiff * heightDiff;
            outputCursor++;
          }

          jac.block(
              0, 0, markerVector.size() * 3, skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        markerVector);
          jac.block(
              0,
              skeletonBallJoints->getNumDofs(),
              markerVector.size() * 3,
              skeletonBallJoints->getGroupScaleDim())
              = skeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtGroupScales(
                        markerVector);
          for (int i = 0; i < markerWeightsVector.size(); i++)
          {
            diff.segment<3>(i * 3) *= markerWeightsVector(i);
            jac.block(i * 3, 0, 3, jac.cols()) *= markerWeightsVector(i);
          }

          jac.block(
              markerVector.size() * 3,
              0,
              jointsForSkeletonBallJoints.size() * 3,
              skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtJointPositions(
                        jointsForSkeletonBallJoints);
          rescaleIKJacobianForWeightsAndAxis(
              jac.block(
                  markerVector.size() * 3,
                  0,
                  jointsForSkeletonBallJoints.size() * 3,
                  skeletonBallJoints->getNumDofs()),
              jointWeights,
              jointAxis,
              axisWeights);
          jac.block(
              markerVector.size() * 3,
              skeletonBallJoints->getNumDofs(),
              jointsForSkeletonBallJoints.size() * 3,
              skeletonBallJoints->getGroupScaleDim())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtGroupScales(
                        jointsForSkeletonBallJoints);
          rescaleIKJacobianForWeightsAndAxis(
              jac.block(
                  markerVector.size() * 3,
                  skeletonBallJoints->getNumDofs(),
                  jointsForSkeletonBallJoints.size() * 3,
                  skeletonBallJoints->getGroupScaleDim()),
              jointWeights,
              jointAxis,
              axisWeights);

          // Add group scales grad
          int outputRowCursor = (markerVector.size() * 3)
                                + (jointsForSkeletonBallJoints.size() * 3)
                                + virtualSpringDofs.size();
          if (fitter->mAnthropometrics != nullptr)
          {
            Eigen::VectorXs expNegLogPdfGradient
                = -alphaScale * expNegLogPdf
                  * fitter->mAnthropometrics->getGradientOfLogPDFWrtGroupScales(
                      skeletonBallJoints);
            int groupScaleDim = skeletonBallJoints->getGroupScaleDim();
            assert(groupScaleDim == expNegLogPdfGradient.size());
            assert(
                jac.cols() == skeletonBallJoints->getNumDofs() + groupScaleDim);
            assert(outputRowCursor < jac.rows());
            jac.block(
                outputRowCursor,
                skeletonBallJoints->getNumDofs(),
                1,
                groupScaleDim)
                = expNegLogPdfGradient.transpose();
            outputRowCursor++;
          }
          if (useHeightPrior)
          {
            s_t height = skeletonBallJoints->getHeight(
                skeletonBallJointsHeightMeasurementPose);
            s_t heightDiff = height - fitter->mHeightPrior;
            int groupScaleDim = skeletonBallJoints->getGroupScaleDim();
            (void)groupScaleDim;
            assert(outputRowCursor < jac.rows());
            assert(
                jac.cols() == skeletonBallJoints->getNumDofs() + groupScaleDim);
            Eigen::VectorXs gradientOfHeightWrtGroupScales
                = 2 * heightDiff * 10
                  * skeletonBallJoints->getGradientOfHeightWrtGroupScales(
                      skeletonBallJointsHeightMeasurementPose);
            jac.block(
                outputRowCursor,
                skeletonBallJoints->getNumDofs(),
                1,
                skeletonBallJoints->getGroupScaleDim())
                = gradientOfHeightWrtGroupScales.transpose();
            outputRowCursor++;
          }
          assert(jac.rows() == outputDim);
        },
        // Generate a random restart position
        [&skeletonBallJoints,
         &skeleton,
         &observedJoints,
         markerPoses,
         defaultScale](Eigen::Ref<Eigen::VectorXs> val) {
          val.segment(0, skeletonBallJoints->getNumDofs())
              = skeleton
                    ->convertPositionsToBallSpace(
                        skeleton->getRandomPoseForJoints(observedJoints))
                    .segment(0, skeletonBallJoints->getNumDofs());

          // Set the root translation to within a fairly narrow range of the
          // average marker cloud
          Eigen::Vector3s avgMarkerPos = Eigen::Vector3s::Zero();
          for (int i = 0; i < markerPoses.size() / 3; i++)
          {
            avgMarkerPos += markerPoses.segment<3>(i * 3);
          }
          avgMarkerPos /= (s_t)(markerPoses.size() / 3);
          val.segment<3>(3) = avgMarkerPos + (Eigen::Vector3s::Random() * 0.2);

          val.segment(
                 skeletonBallJoints->getNumDofs(),
                 skeletonBallJoints->getGroupScaleDim())
              .setConstant(defaultScale);
        },
        math::IKConfig()
            .setMaxStepCount(350)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(fitter->mInitialIKSatisfactoryLoss)
            .setMaxRestarts(fitter->mInitialIKMaxRestarts)
            .setLogOutput(debug)
            .setInputNames(inputNames)
            .setOutputNames(outputNames));
  }

  // 3. Return the result from the best fit we had
  result.pose = skeleton->getPositions();
  // If we used a more restrictive "uniform scaling for all the bodies" reset
  // that here, so that the dimensions of getGroupScales() below is correct.
  if (useMoreRestrictiveScaleBounds)
  {
    for (int i = 0; i < skeleton->getNumScaleGroups(); i++)
    {
      skeleton->setScaleGroupUniformScaling(
          skeleton->getBodyScaleGroup(i).nodes[0], originalUniformScaling[i]);
    }
  }
  result.scale = skeleton->getGroupScales();
  std::cout << "Best result: " << result.score << std::endl;

  if (saveToGUI)
  {
    server.writeFramesJson("../../../javascript/src/data/movement2.bin");
  }

  /*
  if (result.score > 0.15)
  {
    // This is very bad
    std::cout << "Very bad result! Recursing in the hope of doing better:"
              << std::endl;
    return scaleAndFit(
        fitter,
        markerObservations,
        firstGuessPose,
        markerWeights,
        markerOffsets,
        joints,
        jointCenters,
        jointWeights,
        jointAxis,
        axisWeights,
        dontScale,
        debugIndex,
        true);
  }
  */

  return result;
}

//==============================================================================
/// This fits IK to the given trajectory, without scaling
void MarkerFitter::fitTrajectory(
    const MarkerFitter* fitter,
    Eigen::VectorXs groupScales,
    Eigen::VectorXs firstPoseGuess,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    std::map<std::string, s_t> markerWeights,
    std::map<std::string, Eigen::Vector3s> markerOffsets,
    std::vector<dynamics::Joint*> joints,
    std::vector<Eigen::VectorXs> jointCenters,
    Eigen::VectorXs jointWeights,
    std::vector<Eigen::VectorXs> jointAxis,
    Eigen::VectorXs axisWeights,
    std::vector<dynamics::Joint*> initObservedJoints,
    Eigen::Ref<Eigen::MatrixXs> result,
    Eigen::Ref<Eigen::VectorXs> resultScores,
    bool backwards)
{
  assert(initObservedJoints.size() > 0);

  // 0. To make this thread safe, we're going to clone the fitter skeleton
  std::shared_ptr<dynamics::Skeleton> skeleton;
  {
    const std::lock_guard<std::mutex> lock(
        *(const_cast<std::mutex*>(&fitter->mGlobalLock)));
    skeleton = fitter->mSkeleton->clone();
  }
  skeleton->setGroupScales(groupScales);

  // 0.1. Translate over the observedJoints array to the cloned skeleton
  std::vector<dynamics::Joint*> observedJoints;
  for (auto joint : initObservedJoints)
  {
    observedJoints.push_back(skeleton->getJoint(joint->getName()));
  }

  bool useBallJoints = true;

  if (useBallJoints)
  {
    // 1.1. Convert the skeleton to have any Euler joints as ball joints
    std::shared_ptr<dynamics::Skeleton> skeletonBallJoints
        = skeleton->convertSkeletonToBallJoints();
    skeletonBallJoints->setGroupScales(groupScales);
    std::vector<dynamics::Joint*> jointsForSkeletonBallJoints;
    for (auto joint : joints)
    {
      jointsForSkeletonBallJoints.push_back(
          skeletonBallJoints->getJoint(joint->getName()));
    }

    // 1.2. The initial guess will be carried from timestep to timestep. The
    // solution of the previous timestep will form the initial guess for the
    // next timestep.
    Eigen::VectorXs initialGuess
        = skeleton->convertPositionsToBallSpace(firstPoseGuess);

    // 1.3. Verify the results matrix
    assert(result.rows() == skeleton->getNumDofs());
    assert(result.cols() == markerObservations.size());

    if (fitter->mUseParallelIKWarps)
    {
      int numThreads = 32;
      int numWarps = ceil((s_t)markerObservations.size() / numThreads);

      // Create copies of the skeleton we'll use for multi-threaded IK, since
      // each thread will be re-posing the skeleton independently and in
      // parallel, and we don't want them stepping on each other's toes.
      std::vector<std::shared_ptr<dynamics::Skeleton>> threadSkeleton;
      std::vector<std::shared_ptr<dynamics::Skeleton>> threadSkeletonBallJoints;
      std::vector<std::vector<dynamics::Joint*>>
          threadJointsForSkeletonBallJoints;
      for (int t = 0; t < numThreads; t++)
      {
        threadSkeleton.push_back(skeleton->cloneSkeleton());
        threadSkeletonBallJoints.push_back(skeletonBallJoints->cloneSkeleton());
        std::vector<dynamics::Joint*> jointsForThreadSkeletonBallJoints;
        for (auto joint : joints)
        {
          jointsForThreadSkeletonBallJoints.push_back(
              threadSkeletonBallJoints[threadSkeletonBallJoints.size() - 1]
                  ->getJoint(joint->getName()));
        }
        threadJointsForSkeletonBallJoints.push_back(
            jointsForThreadSkeletonBallJoints);
      }

      for (int warp = 0; warp < numWarps; warp++)
      {
        int warpStart = warp * numThreads;
        int warpEndExclusive
            = min((int)(warp + 1) * numThreads, (int)markerObservations.size());

        std::vector<std::future<Eigen::VectorXs>> warpFutures;
        // 2. Run through each observation in sequence, and do a best fit
        for (int j = warpStart; j < warpEndExclusive; j++)
        {
          int threadIdx = j - warpStart;
          int i = j;
          if (backwards)
          {
            i = markerObservations.size() - 1 - j;
          }

          warpFutures.push_back(std::async([i,
                                            &markerObservations,
                                            &jointCenters,
                                            &jointWeights,
                                            &jointAxis,
                                            &axisWeights,
                                            &markerWeights,
                                            &markerOffsets,
                                            &fitter,
                                            &joints,
                                            &result,
                                            &resultScores,
                                            initialGuess,
                                            threadIdx,
                                            &threadSkeleton,
                                            &threadSkeletonBallJoints,
                                            &threadJointsForSkeletonBallJoints] {
            // 2.0. Grab the skeleton copies for this thread
            std::shared_ptr<dynamics::Skeleton> skeleton
                = threadSkeleton[threadIdx];
            std::shared_ptr<dynamics::Skeleton> skeletonBallJoints
                = threadSkeletonBallJoints[threadIdx];
            std::vector<dynamics::Joint*> jointsForSkeletonBallJoints
                = threadJointsForSkeletonBallJoints[threadIdx];

            // 2.1. Linearize the marker names and marker observations. This
            // needs to be done at each step, because the observed markers can
            // be different at different steps.
            Eigen::VectorXs markerPoses
                = Eigen::VectorXs::Zero(markerObservations[i].size() * 3);
            Eigen::VectorXs markerWeightsVector
                = Eigen::VectorXs::Ones(markerObservations[i].size());
            Eigen::VectorXs centerPoses = jointCenters[i];
            Eigen::VectorXs axisPoses = jointAxis[i];
            std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
                markerVector;
            std::vector<std::string> outputNames;
            for (std::pair<std::string, Eigen::Vector3s> pair :
                 markerObservations[i])
            {
              markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
              if (markerWeights.count(pair.first))
              {
                assert(markerWeights.count(pair.first));
                markerWeightsVector(markerVector.size())
                    = markerWeights.at(pair.first);
              }
              else
              {
                markerWeightsVector(markerVector.size())
                    = fitter->mMarkerIsTracking.at(
                          fitter->mMarkerIndices.at(pair.first))
                          ? fitter->mTrackingMarkerDefaultWeight
                          : fitter->mAnatomicalMarkerDefaultWeight;
              }
              assert(fitter->mMarkerMap.count(pair.first));
              const std::pair<dynamics::BodyNode*, Eigen::Vector3s>&
                  originalMarker
                  = fitter->mMarkerMap.at(pair.first);
              Eigen::Vector3s offset = Eigen::Vector3s::Zero();
              if (markerOffsets.count(pair.first))
              {
                assert(markerOffsets.count(pair.first));
                offset = markerOffsets.at(pair.first);
              }
              markerVector.emplace_back(
                  skeletonBallJoints->getBodyNode(
                      originalMarker.first->getName()),
                  originalMarker.second + offset);
              Eigen::Vector3s markerPos = originalMarker.second + offset;
              std::string markerPrefix
                  = "marker " + originalMarker.first->getName() + " ("
                    + std::to_string((double)markerPos(0)) + ","
                    + std::to_string((double)markerPos(1)) + ","
                    + std::to_string((double)markerPos(2)) + ")";
              outputNames.push_back(markerPrefix + " X");
              outputNames.push_back(markerPrefix + " Y");
              outputNames.push_back(markerPrefix + " Z");
            }
            for (int i = 0; i < joints.size(); i++)
            {
              std::string jointPrefix = "joint " + joints[i]->getName();
              outputNames.push_back(jointPrefix + " X");
              outputNames.push_back(jointPrefix + " Y");
              outputNames.push_back(jointPrefix + " Z");
            }
            std::vector<std::string> inputNames;
            for (int i = 0; i < skeletonBallJoints->getNumDofs(); i++)
            {
              inputNames.push_back(
                  "dof " + skeletonBallJoints->getDof(i)->getName());
            }

            assert(markerPoses.size() == markerVector.size() * 3);
            assert(centerPoses.size() == joints.size() * 3);

            // 2.2. Actually run the IK solver
            // Initialize at the old config

            const bool ignoreJointLimits = fitter->mIgnoreJointLimits;
            s_t finalLoss = math::solveIK(
                initialGuess,
                skeletonBallJoints->getPositionUpperLimits(),
                skeletonBallJoints->getPositionLowerLimits(),
                (markerVector.size() * 3) + (joints.size() * 3),
                // Set positions
                [skeletonBallJoints, skeleton, ignoreJointLimits](
                    /* in*/ const Eigen::VectorXs pos, bool clamp) {
                  skeletonBallJoints->setPositions(pos);
                  if (clamp)
                  {
                    // 1. Map the position back into eulerian space
                    skeleton->setPositions(
                        skeleton->convertPositionsFromBallSpace(pos));
                    if (!ignoreJointLimits)
                    {
                      // 2. Clamp the position to limits
                      skeleton->clampPositionsToLimits();
                      // 3. Map the position back into SO3 space
                      skeletonBallJoints->setPositions(
                          skeleton->convertPositionsToBallSpace(
                              skeleton->getPositions()));
                    }
                  }

                  // Return the clamped position
                  return skeletonBallJoints->getPositions();
                },
                [skeletonBallJoints,
                 markerPoses,
                 markerVector,
                 markerWeightsVector,
                 jointsForSkeletonBallJoints,
                 centerPoses,
                 jointWeights,
                 axisPoses,
                 axisWeights](
                    /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
                    /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
                  assert(
                      diff.size() == markerPoses.size() + centerPoses.size());

                  diff.segment(0, markerPoses.size())
                      = skeletonBallJoints->getMarkerWorldPositions(
                            markerVector)
                        - markerPoses;
                  Eigen::VectorXs jointPoses
                      = skeletonBallJoints->getJointWorldPositions(
                          jointsForSkeletonBallJoints);
                  computeJointIKDiff(
                      diff.segment(markerPoses.size(), centerPoses.size()),
                      jointPoses,
                      centerPoses,
                      jointWeights,
                      axisPoses,
                      axisWeights);

                  assert(jac.cols() == skeletonBallJoints->getNumDofs());
                  assert(
                      jac.rows()
                      == (markerVector.size() * 3)
                             + (jointsForSkeletonBallJoints.size() * 3));
                  jac.block(
                      0,
                      0,
                      markerVector.size() * 3,
                      skeletonBallJoints->getNumDofs())
                      = skeletonBallJoints
                            ->getMarkerWorldPositionsJacobianWrtJointPositions(
                                markerVector);
                  jac.block(
                      markerVector.size() * 3,
                      0,
                      jointsForSkeletonBallJoints.size() * 3,
                      skeletonBallJoints->getNumDofs())
                      = skeletonBallJoints
                            ->getJointWorldPositionsJacobianWrtJointPositions(
                                jointsForSkeletonBallJoints);
                  for (int i = 0; i < markerWeightsVector.size(); i++)
                  {
                    diff.segment<3>(i * 3) *= markerWeightsVector(i);
                    jac.block(i * 3, 0, 3, jac.cols())
                        *= markerWeightsVector(i);
                  }
                  rescaleIKJacobianForWeightsAndAxis(
                      jac.block(
                          markerVector.size() * 3,
                          0,
                          jointsForSkeletonBallJoints.size() * 3,
                          skeletonBallJoints->getNumDofs()),
                      jointWeights,
                      axisPoses,
                      axisWeights);
                },
                [initialGuess](Eigen::Ref<Eigen::VectorXs> val) {
                  assert(false);
                  val = initialGuess;
                },
                math::IKConfig()
                    .setMaxStepCount(500)
                    .setConvergenceThreshold(1e-6)
                    .setDontExitTranspose(true)
                    .setLossLowerBound(1e-8)
                    .setMaxRestarts(1)
                    .setStartClamped(true)
                    .setLogOutput(false)
                    .setInputNames(inputNames)
                    .setOutputNames(outputNames));

            // 2.3. Record this outcome
            result.col(i) = skeleton->getPositions();
            resultScores(i) = finalLoss;

            // 2.4. Set up for the next iteration, by setting the initial guess
            // to the current solve
            return skeletonBallJoints->getPositions();
          }));
        }

        // Block until all these warps have finished
        for (auto& warpFuture : warpFutures)
        {
          initialGuess = warpFuture.get();
        }
      }
    }
    else
    {
      // 2. Run through each observation in sequence, and do a best fit
      for (int j = 0; j < markerObservations.size(); j++)
      {
        int i = j;
        if (backwards)
        {
          i = markerObservations.size() - 1 - j;
        }

        /*
        std::cout << "> Fit timestep " << i << "/" << markerObservations.size()
                  << std::endl;
        */

        // 2.1. Linearize the marker names and marker observations. This needs
        // to be done at each step, because the observed markers can be
        // different at different steps.
        Eigen::VectorXs markerPoses
            = Eigen::VectorXs::Zero(markerObservations[i].size() * 3);
        Eigen::VectorXs markerWeightsVector
            = Eigen::VectorXs::Ones(markerObservations[i].size());
        Eigen::VectorXs centerPoses = jointCenters[i];
        Eigen::VectorXs axisPoses = jointAxis[i];
        std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
            markerVector;
        std::vector<std::string> outputNames;
        for (std::pair<std::string, Eigen::Vector3s> pair :
             markerObservations[i])
        {
          markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
          if (markerWeights.count(pair.first))
          {
            assert(markerWeights.count(pair.first));
            markerWeightsVector(markerVector.size())
                = markerWeights.at(pair.first);
          }
          else
          {
            markerWeightsVector(markerVector.size())
                = fitter->mMarkerIsTracking.at(
                      fitter->mMarkerIndices.at(pair.first))
                      ? fitter->mTrackingMarkerDefaultWeight
                      : fitter->mAnatomicalMarkerDefaultWeight;
          }
          assert(fitter->mMarkerMap.count(pair.first));
          const std::pair<dynamics::BodyNode*, Eigen::Vector3s>& originalMarker
              = fitter->mMarkerMap.at(pair.first);
          Eigen::Vector3s offset = Eigen::Vector3s::Zero();
          if (markerOffsets.count(pair.first))
          {
            assert(markerOffsets.count(pair.first));
            offset = markerOffsets.at(pair.first);
          }
          markerVector.emplace_back(
              skeletonBallJoints->getBodyNode(originalMarker.first->getName()),
              originalMarker.second + offset);
          Eigen::Vector3s markerPos = originalMarker.second + offset;
          std::string markerPrefix
              = "marker " + originalMarker.first->getName() + " ("
                + std::to_string((double)markerPos(0)) + ","
                + std::to_string((double)markerPos(1)) + ","
                + std::to_string((double)markerPos(2)) + ")";
          outputNames.push_back(markerPrefix + " X");
          outputNames.push_back(markerPrefix + " Y");
          outputNames.push_back(markerPrefix + " Z");
        }
        for (int i = 0; i < joints.size(); i++)
        {
          std::string jointPrefix = "joint " + joints[i]->getName();
          outputNames.push_back(jointPrefix + " X");
          outputNames.push_back(jointPrefix + " Y");
          outputNames.push_back(jointPrefix + " Z");
        }
        std::vector<std::string> inputNames;
        for (int i = 0; i < skeletonBallJoints->getNumDofs(); i++)
        {
          inputNames.push_back(
              "dof " + skeletonBallJoints->getDof(i)->getName());
        }

        assert(markerPoses.size() == markerVector.size() * 3);
        assert(centerPoses.size() == joints.size() * 3);

        // 2.2. Actually run the IK solver
        // Initialize at the old config

        const bool ignoreJointLimits = fitter->mIgnoreJointLimits;
        s_t finalLoss = math::solveIK(
            initialGuess,
            skeletonBallJoints->getPositionUpperLimits(),
            skeletonBallJoints->getPositionLowerLimits(),
            (markerVector.size() * 3) + (joints.size() * 3),
            // Set positions
            [skeletonBallJoints, skeleton, ignoreJointLimits](
                /* in*/ const Eigen::VectorXs pos, bool clamp) {
              skeletonBallJoints->setPositions(pos);
              if (clamp)
              {
                // 1. Map the position back into eulerian space
                skeleton->setPositions(
                    skeleton->convertPositionsFromBallSpace(pos));
                if (!ignoreJointLimits)
                {
                  // 2. Clamp the position to limits
                  skeleton->clampPositionsToLimits();
                  // 3. Map the position back into SO3 space
                  skeletonBallJoints->setPositions(
                      skeleton->convertPositionsToBallSpace(
                          skeleton->getPositions()));
                }
              }

              // Return the clamped position
              return skeletonBallJoints->getPositions();
            },
            [skeletonBallJoints,
             markerPoses,
             markerVector,
             markerWeightsVector,
             jointsForSkeletonBallJoints,
             centerPoses,
             jointWeights,
             axisPoses,
             axisWeights](
                /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
                /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
              assert(diff.size() == markerPoses.size() + centerPoses.size());

              diff.segment(0, markerPoses.size())
                  = skeletonBallJoints->getMarkerWorldPositions(markerVector)
                    - markerPoses;
              Eigen::VectorXs jointPoses
                  = skeletonBallJoints->getJointWorldPositions(
                      jointsForSkeletonBallJoints);
              computeJointIKDiff(
                  diff.segment(markerPoses.size(), centerPoses.size()),
                  jointPoses,
                  centerPoses,
                  jointWeights,
                  axisPoses,
                  axisWeights);

              assert(jac.cols() == skeletonBallJoints->getNumDofs());
              assert(
                  jac.rows()
                  == (markerVector.size() * 3)
                         + (jointsForSkeletonBallJoints.size() * 3));
              jac.block(
                  0,
                  0,
                  markerVector.size() * 3,
                  skeletonBallJoints->getNumDofs())
                  = skeletonBallJoints
                        ->getMarkerWorldPositionsJacobianWrtJointPositions(
                            markerVector);
              jac.block(
                  markerVector.size() * 3,
                  0,
                  jointsForSkeletonBallJoints.size() * 3,
                  skeletonBallJoints->getNumDofs())
                  = skeletonBallJoints
                        ->getJointWorldPositionsJacobianWrtJointPositions(
                            jointsForSkeletonBallJoints);
              for (int i = 0; i < markerWeightsVector.size(); i++)
              {
                diff.segment<3>(i * 3) *= markerWeightsVector(i);
                jac.block(i * 3, 0, 3, jac.cols()) *= markerWeightsVector(i);
              }
              rescaleIKJacobianForWeightsAndAxis(
                  jac.block(
                      markerVector.size() * 3,
                      0,
                      jointsForSkeletonBallJoints.size() * 3,
                      skeletonBallJoints->getNumDofs()),
                  jointWeights,
                  axisPoses,
                  axisWeights);
            },
            [initialGuess](Eigen::Ref<Eigen::VectorXs> val) {
              assert(false);
              val = initialGuess;
            },
            math::IKConfig()
                .setMaxStepCount(500)
                .setConvergenceThreshold(1e-6)
                .setDontExitTranspose(true)
                .setLossLowerBound(1e-8)
                .setMaxRestarts(1)
                .setStartClamped(true)
                .setLogOutput(false)
                .setInputNames(inputNames)
                .setOutputNames(outputNames));

        // 2.3. Record this outcome
        result.col(i) = skeleton->getPositions();
        resultScores(i) = finalLoss;

        // 2.4. Set up for the next iteration, by setting the initial guess to
        // the current solve
        initialGuess = skeletonBallJoints->getPositions();
      }
    }
  }
  else
  {
    // 1.2. The initial guess will be carried from timestep to timestep. The
    // solution of the previous timestep will form the initial guess for the
    // next timestep.
    Eigen::VectorXs initialGuess = firstPoseGuess;

    // 1.3. Verify the results matrix
    assert(result.rows() == skeleton->getNumDofs());
    assert(result.cols() == markerObservations.size());

    // 2. Run through each observation in sequence, and do a best fit
    for (int j = 0; j < markerObservations.size(); j++)
    {
      int i = j;
      if (backwards)
      {
        i = markerObservations.size() - 1 - j;
      }

      /*
      std::cout << "> Fit timestep " << i << "/" << markerObservations.size()
                << std::endl;
      */

      // 2.1. Linearize the marker names and marker observations. This needs to
      // be done at each step, because the observed markers can be different at
      // different steps.
      Eigen::VectorXs markerPoses
          = Eigen::VectorXs::Zero(markerObservations[i].size() * 3);
      Eigen::VectorXs markerWeightsVector
          = Eigen::VectorXs::Ones(markerObservations[i].size());
      Eigen::VectorXs centerPoses = jointCenters[i];
      Eigen::VectorXs axisPoses = jointAxis[i];
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerVector;
      for (std::pair<std::string, Eigen::Vector3s> pair : markerObservations[i])
      {
        markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
        if (markerWeights.count(pair.first))
        {
          assert(markerWeights.count(pair.first));
          markerWeightsVector(markerVector.size())
              = markerWeights.at(pair.first);
        }
        else
        {
          markerWeightsVector(markerVector.size())
              = fitter->mMarkerIsTracking.at(
                    fitter->mMarkerIndices.at(pair.first))
                    ? fitter->mTrackingMarkerDefaultWeight
                    : fitter->mAnatomicalMarkerDefaultWeight;
        }
        assert(fitter->mMarkerMap.count(pair.first));
        const std::pair<dynamics::BodyNode*, Eigen::Vector3s>& originalMarker
            = fitter->mMarkerMap.at(pair.first);
        Eigen::Vector3s offset = Eigen::Vector3s::Zero();
        if (markerOffsets.count(pair.first))
        {
          assert(markerOffsets.count(pair.first));
          offset = markerOffsets.at(pair.first);
        }
        markerVector.emplace_back(
            skeleton->getBodyNode(originalMarker.first->getName()),
            originalMarker.second + offset);
      }

      // 2.2. Actually run the IK solver
      // Initialize at the old config

      const bool ignoreJointLimits = fitter->mIgnoreJointLimits;
      s_t finalLoss = math::solveIK(
          initialGuess,
          skeleton->getPositionUpperLimits(),
          skeleton->getPositionLowerLimits(),
          (markerVector.size() * 3) + (joints.size() * 3),
          // Set positions
          [skeleton, ignoreJointLimits](
              /* in*/ const Eigen::VectorXs pos, bool clamp) {
            skeleton->setPositions(pos);

            if (clamp && !ignoreJointLimits)
            {
              // Clamp the position to limits
              skeleton->clampPositionsToLimits();
            }

            // Return the clamped position
            return skeleton->getPositions();
          },
          [skeleton,
           joints,
           markerPoses,
           markerVector,
           markerWeightsVector,
           centerPoses,
           jointWeights,
           axisPoses,
           axisWeights](
              /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
              /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
            diff.segment(0, markerPoses.size())
                = skeleton->getMarkerWorldPositions(markerVector) - markerPoses;
            Eigen::VectorXs jointPoses
                = skeleton->getJointWorldPositions(joints);
            computeJointIKDiff(
                diff.segment(markerPoses.size(), centerPoses.size()),
                jointPoses,
                centerPoses,
                jointWeights,
                axisPoses,
                axisWeights);

            assert(jac.cols() == skeleton->getNumDofs());
            assert(
                jac.rows() == (markerVector.size() * 3) + (joints.size() * 3));
            jac.block(0, 0, markerVector.size() * 3, skeleton->getNumDofs())
                = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(
                    markerVector);
            jac.block(
                markerVector.size() * 3,
                0,
                joints.size() * 3,
                skeleton->getNumDofs())
                = skeleton->getJointWorldPositionsJacobianWrtJointPositions(
                    joints);
            for (int i = 0; i < markerWeightsVector.size(); i++)
            {
              diff.segment<3>(i * 3) *= markerWeightsVector(i);
              jac.block(i * 3, 0, 3, jac.cols()) *= markerWeightsVector(i);
            }
            rescaleIKJacobianForWeightsAndAxis(
                jac.block(
                    markerVector.size() * 3,
                    0,
                    joints.size() * 3,
                    skeleton->getNumDofs()),
                jointWeights,
                axisPoses,
                axisWeights);
          },
          [initialGuess](Eigen::Ref<Eigen::VectorXs> val) {
            assert(false);
            val = initialGuess;
          },
          math::IKConfig()
              .setMaxStepCount(500)
              .setConvergenceThreshold(1e-6)
              .setDontExitTranspose(true)
              .setLossLowerBound(1e-8)
              .setMaxRestarts(1)
              .setStartClamped(true)
              .setLogOutput(false));

      // 2.3. Record this outcome
      result.col(i) = skeleton->getPositions();
      resultScores(i) = finalLoss;

      // 2.4. Set up for the next iteration, by setting the initial guess to the
      // current solve
      initialGuess = skeleton->getPositions();
    }
  }
}

//==============================================================================
/// This solves a bunch of optimization problems, one per joint, to find and
/// track the joint centers over time. It puts the results back into
/// `initialization`
void MarkerFitter::findJointCenters(
    MarkerInitialization& initialization,
    const std::vector<bool>& newClip,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // 1. Figure out which joints to find centers for
  initialization.joints.clear();
  for (int i = 0; i < mSkeleton->getNumJoints(); i++)
  {
    if (SphereFitJointCenterProblem::canFitJoint(
            this, mSkeleton->getJoint(i), markerObservations))
    {
      initialization.joints.push_back(mSkeleton->getJoint(i));
    }
  }
  initialization.jointCenters = Eigen::MatrixXs::Zero(
      initialization.joints.size() * 3, markerObservations.size());
  assert(
      initialization.joints.size() * 3 == initialization.jointCenters.rows());

  initialization.jointWeights
      = Eigen::VectorXs::Ones(initialization.joints.size());
  initialization.jointLoss
      = Eigen::VectorXs::Ones(initialization.joints.size());

  /*
  // 2. Actually compute the joint centers (single threaded)
  for (int i = 0; i < initialization.joints.size(); i++)
  {
    std::cout << "Computing joint center for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;

    std::shared_ptr<SphereFitJointCenterProblem> problemPtr
        = std::make_shared<SphereFitJointCenterProblem>(
            this,
            markerObservations,
            initialization.poses,
            initialization.joints[i],
            initialization.jointCenters.block(
                i * 3, 0, 3, markerObservations.size()));

    findJointCenter(problemPtr)->saveSolutionBackToInitialization();
  }
  */

  if (initialization.poses.hasNaN())
  {
    std::cout << "IK initialization in findJointCenters() has NaNs!"
              << std::endl;
    exit(1);
  }

  // 2. Actually compute the joint centers (multi threaded)
  std::vector<std::future<std::shared_ptr<SphereFitJointCenterProblem>>>
      futures;
  for (int i = 0; i < initialization.joints.size(); i++)
  {
    std::cout << "Computing joint center for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;

    std::shared_ptr<SphereFitJointCenterProblem> problemPtr
        = std::make_shared<SphereFitJointCenterProblem>(
            this,
            markerObservations,
            initialization.poses,
            initialization.joints[i],
            newClip,
            initialization.jointCenters.block(
                i * 3, 0, 3, markerObservations.size()));
    initialization.jointsAdjacentMarkers.push_back(problemPtr->mActiveMarkers);

    futures.push_back(std::async(
        [this, problemPtr] { return this->findJointCenter(problemPtr); }));
  }
  for (int i = 0; i < futures.size(); i++)
  {
    s_t loss = futures[i].get()->saveSolutionBackToInitialization();
    initialization.jointLoss(i) = loss / markerObservations.size();
    std::cout << "Finished computing joint center for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;
  }
  std::cout << "Finished computing all joint centers!" << std::endl;
}

//==============================================================================
/// This finds the trajectory for a single specified joint center over time
std::shared_ptr<SphereFitJointCenterProblem> MarkerFitter::findJointCenter(
    std::shared_ptr<SphereFitJointCenterProblem> problemPtr, bool logSteps)
{
  SphereFitJointCenterProblem* problem = problemPtr.get();

  s_t lr = 1.0;
  s_t decay = 0.99;
  Eigen::VectorXs x = problem->flatten();
  Eigen::VectorXs accum = Eigen::VectorXs::Ones(x.size()) * 1.0;
#ifndef NDEBUG
  if (x.hasNaN() || accum.hasNaN())
  {
    std::cout << "Got a NaN before init!" << std::endl;
    std::cout << "x.hasNaN(): " << x.hasNaN() << std::endl;
    std::cout << "accum.hasNaN(): " << accum.hasNaN() << std::endl;
    exit(1);
  }
#endif
  s_t loss = problem->getLoss();
  s_t initialLoss = loss;
  for (int i = 0; i < mJointSphereFitSGDIterations; i++)
  {
    Eigen::VectorXs grad = problem->getGradient();
    // The use of `decay` here is AdaDelta, to solve the normal problem in
    // Adagrad of the unbounded accumulation of a Hessian approximation, which
    // ends up eventually driving your gradients to 0 and terminating
    // optimization, even if you're not at the optimum.
    Eigen::VectorXs newAccum
        = decay * accum + (1.0 - decay) * grad.cwiseProduct(grad);
    Eigen::VectorXs newX = x - grad.cwiseQuotient(newAccum) * lr;
#ifndef NDEBUG
    if (newX.hasNaN())
    {
      std::cout << "Got a NaN on iteration " << i << std::endl;
      std::cout << "x.hasNaN(): " << x.hasNaN() << std::endl;
      std::cout << "grad.hasNaN(): " << grad.hasNaN() << std::endl;
      std::cout << "accum.hasNaN(): " << accum.hasNaN() << std::endl;
      std::cout << "lr: " << lr << std::endl;
      exit(1);
    }
#endif
    problem->unflatten(newX);
    s_t newLoss = problem->getLoss();
    if (newLoss < loss)
    {
      loss = newLoss;
      x = newX;
      accum = newAccum;
      if (logSteps)
      {
        std::cout << "[lr=" << lr << "] " << i << ": " << newLoss << std::endl;
      }
      lr *= 1.1;
    }
    else
    {
      if (logSteps)
      {
        std::cout << "[bad step, lr=" << lr << "] " << i << ": " << newLoss
                  << std::endl;
      }
      // backtrack
      problem->unflatten(x);
      lr *= 0.5;
    }
  }
  std::cout << "Sphere-fitting \"" << problemPtr->mJointName << "\""
            << ": initial loss=" << (initialLoss / problemPtr->mNumTimesteps)
            << ", final loss=" << (loss / problemPtr->mNumTimesteps)
            << std::endl;
  return problemPtr;
}

//==============================================================================
/// This solves a bunch of optimization problems, one per joint, to find and
/// track the joint axis over time. It puts the results back into
/// `initialization`
void MarkerFitter::findAllJointAxis(
    MarkerInitialization& initialization,
    const std::vector<bool>& newClip,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // 1. Initialize the matrices we'll fill up
  initialization.jointAxis = Eigen::MatrixXs::Zero(
      initialization.joints.size() * 6, markerObservations.size());
  initialization.axisWeights
      = Eigen::VectorXs::Ones(initialization.joints.size());
  initialization.axisLoss = Eigen::VectorXs::Ones(initialization.joints.size());

  /*
  // 2. Actually compute the joint centers (single threaded)
  for (int i = 0; i < initialization.joints.size(); i++)
  {
    std::cout << "Computing joint center for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;

    std::shared_ptr<SphereFitJointCenterProblem> problemPtr
        = std::make_shared<SphereFitJointCenterProblem>(
            this,
            markerObservations,
            initialization.poses,
            initialization.joints[i],
            initialization.jointCenters.block(
                i * 3, 0, 3, markerObservations.size()));

    findJointCenter(problemPtr)->saveSolutionBackToInitialization();
  }
  */

  // 2. Actually compute the joint centers (multi threaded)
  std::vector<std::future<std::shared_ptr<CylinderFitJointAxisProblem>>>
      futures;
  for (int i = 0; i < initialization.joints.size(); i++)
  {
    std::cout << "Computing joint axis for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;

    std::shared_ptr<CylinderFitJointAxisProblem> problemPtr
        = std::make_shared<CylinderFitJointAxisProblem>(
            this,
            markerObservations,
            initialization.poses,
            initialization.joints[i],
            initialization.jointCenters.block(
                i * 3, 0, 3, markerObservations.size()),
            newClip,
            initialization.jointAxis.block(
                i * 6, 0, 6, markerObservations.size()));

    futures.push_back(std::async(
        [this, problemPtr] { return this->findJointAxis(problemPtr); }));
  }
  for (int i = 0; i < futures.size(); i++)
  {
    s_t loss = futures[i].get()->saveSolutionBackToInitialization();
    initialization.axisLoss(i) = loss / markerObservations.size();

    std::cout << "Finished computing joint axis for " << i << "/"
              << initialization.joints.size() << ": \""
              << initialization.joints[i]->getName() << "\"" << std::endl;
  }
  std::cout << "Finished computing all joint axis!" << std::endl;
}

//==============================================================================
/// This finds the trajectory for a single specified joint axis over time
std::shared_ptr<CylinderFitJointAxisProblem> MarkerFitter::findJointAxis(
    std::shared_ptr<CylinderFitJointAxisProblem> problemPtr, bool logSteps)
{
  CylinderFitJointAxisProblem* problem = problemPtr.get();

  s_t lr = 1.0;
  Eigen::VectorXs x = problem->flatten();

  s_t decay = 0.99;
  Eigen::VectorXs accum = Eigen::VectorXs::Ones(x.size()) * 0.001;

  s_t loss = problem->getLoss();
  s_t initialLoss = loss;
  for (int i = 0; i < mJointAxisFitSGDIterations; i++)
  {
    Eigen::VectorXs grad = problem->getGradient();
    accum = decay * accum + (1.0 - decay) * grad.cwiseProduct(grad);
    x = problem->flatten();
    Eigen::VectorXs newX = x - grad.cwiseQuotient(accum) * lr;
    problem->unflatten(newX);
    s_t newLoss = problem->getLoss();
    if (newLoss < loss)
    {
      loss = newLoss;
      x = newX;
      if (logSteps)
      {
        std::cout << "[lr=" << lr << "] " << i << ": " << newLoss << std::endl;
      }
      lr *= 1.1;
    }
    else
    {
      if (logSteps)
      {
        std::cout << "[bad step, lr=" << lr << "] " << i << ": " << newLoss
                  << std::endl;
      }
      // backtrack
      problem->unflatten(x);
      lr *= 0.5;
    }
  }
  std::cout << "Cylinder fitting \"" << problemPtr->mJointName << "\""
            << ": initial loss=" << (initialLoss / problemPtr->mNumTimesteps)
            << ", final loss=" << (loss / problemPtr->mNumTimesteps)
            << std::endl;
  return problemPtr;
}

//==============================================================================
/// This computes several metrics, including the variation in the marker
/// movement for each joint, which then go into computing how much weight we
/// should put on each joint center / joint axis.
void MarkerFitter::computeJointConfidences(
    MarkerInitialization& initialization,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  initialization.jointMarkerVariability
      = Eigen::VectorXs::Zero(initialization.joints.size());
  for (int i = 0; i < initialization.joints.size(); i++)
  {
    s_t variability
        = computeJointVariability(initialization.joints[i], markerObservations);
    initialization.jointMarkerVariability(i) = variability;

    // If we've got small variability, then the joint axis won't be
    // very accurate, cause pretty much any axis could fit a bunch of points
    // that don't move relative to each other. Likewise, pretty much any center
    // could fit the points, by just changing the relative radii of each marker.
    // So don't pass along a joint axis, and put joint center at low weight.
    if (variability < mMinVarianceCutoff)
    {
      initialization.axisWeights(i) = 0;
      // We want to map low loss (0.005 ish) to 0.1, and we want to map
      // increasing loss to a decreasing weight
      initialization.jointWeights(i)
          = 0.1 * (mMinSphereFitScore / initialization.jointLoss(i));
      // We cap the weight at 0.1
      if (initialization.jointWeights(i) > 0.1)
      {
        initialization.jointWeights(i) = 0.1;
      }
      std::cout << "Joint " << initialization.joints[i]->getName()
                << " variability " << variability
                << " -> Too low, we won't include joint axis "
                   "restrictions on this joint"
                << "\n\tjoint loss: " << initialization.jointLoss(i)
                << "\n\tjoint weight: " << initialization.jointWeights(i)
                << "\n\taxis loss: " << initialization.axisLoss(i)
                << "\n\taxis weight: " << initialization.axisWeights(i)
                << std::endl;
    }
    else
    {
      // We want to map low loss (0.01 ish) to 1.0, and we want to map
      // increasing loss to a decreasing weight
      initialization.jointWeights(i)
          = mMinSphereFitScore / initialization.jointLoss(i);
      // We cap the weight at 1.0
      if (initialization.jointWeights(i) > mMaxJointWeight)
      {
        initialization.jointWeights(i) = mMaxJointWeight;
      }

      // We want to map low loss (0.001 ish) to 1.0, and we want to map
      // increasing loss to a decreasing weight
      initialization.axisWeights(i)
          = mMinAxisFitScore / initialization.axisLoss(i);
      // We cap the weight at 1.0
      if (initialization.axisWeights(i) > mMaxAxisWeight)
      {
        initialization.axisWeights(i) = mMaxAxisWeight;
      }

      // We'd rather use the axis to fit the joints, because it's less likely to
      // get us into trouble than a bad joint center. So if the axis fit is
      // below a certain absolute loss (indicating that the axis was fairly
      // accurate, even if the sphere fit was better), or if the axis fit is
      // below a certain multiple of the joint fit, then we'll use the axis fit.
      if (initialization.axisLoss(i) < 0.005
          || (initialization.axisLoss(i) < 5 * initialization.jointLoss(i)))
      {
        initialization.jointWeights(i) = 0;
      }
      // If we get lower loss with a ball joint, then set the axis weight to 0
      else
      {
        initialization.axisWeights(i) = 0;
      }
      std::cout << "Joint " << initialization.joints[i]->getName()
                << " variability " << variability
                << "\n\tjoint loss: " << initialization.jointLoss(i)
                << "\n\tjoint weight: " << initialization.jointWeights(i)
                << "\n\taxis loss: " << initialization.axisLoss(i)
                << "\n\taxis weight: " << initialization.axisWeights(i)
                << std::endl;
    }
  }
}

//==============================================================================
/// This sets the minimum joint variance allowed before
/// computeJointConfidences() will cut off a joint as having too low variance
void MarkerFitter::setMinJointVarianceCutoff(s_t cutoff)
{
  mMinVarianceCutoff = cutoff;
}

//==============================================================================
/// This sets the value used to compute sphere fit weights
void MarkerFitter::setMinSphereFitScore(s_t score)
{
  mMinSphereFitScore = score;
}

//==============================================================================
/// This sets the value used to compute axis fit weights
void MarkerFitter::setMinAxisFitScore(s_t score)
{
  mMinAxisFitScore = score;
}

//==============================================================================
/// This sets the maximum value that we can weight a joint center in IK
void MarkerFitter::setMaxJointWeight(s_t weight)
{
  mMaxJointWeight = weight;
}

//==============================================================================
/// This sets the maximum value that we can weight a joint axis in IK
void MarkerFitter::setMaxAxisWeight(s_t weight)
{
  mMaxAxisWeight = weight;
}

//==============================================================================
/// This sets the value weight used to regularize tracking marker offsets from
/// where the model thinks they should be
void MarkerFitter::setRegularizeTrackingMarkerOffsets(s_t weight)
{
  mRegularizeTrackingMarkerOffsets = weight;
}

//==============================================================================
/// This sets the value weight used to regularize anatomical marker offsets from
/// where the model thinks they should be
void MarkerFitter::setRegularizeAnatomicalMarkerOffsets(s_t weight)
{
  mRegularizeAnatomicalMarkerOffsets = weight;
}

//==============================================================================
/// This sets the value weight used to regularize body scales, to penalize
/// scalings that result in bodies that are very different along the 3 axis,
/// like bones that become "fat" in order to not pay a marker regularization
/// penalty, despite having the correct length.
void MarkerFitter::setRegularizeIndividualBodyScales(s_t weight)
{
  mRegularizeIndividualBodyScales = weight;
}

//==============================================================================
/// This tries to make all bones in the body have the same scale, punishing
/// outliers.
void MarkerFitter::setRegularizeAllBodyScales(s_t weight)
{
  mRegularizeAllBodyScales = weight;
}

//==============================================================================
/// If we've disabled joint limits, this provides the option for a soft penalty
/// instead
void MarkerFitter::setRegularizeJointBounds(s_t weight)
{
  mRegularizeJointBounds = weight;
}

//==============================================================================
/// If set to true, we print the pair observation counts and data for
/// computing joint variability.
void MarkerFitter::setDebugJointVariability(bool debug)
{
  mDebugJointVariability = debug;
}

//==============================================================================
/// This is the default weight that gets assigned to anatomical markers during
/// IK, if nothing marker-specific gets assigned.
void MarkerFitter::setAnatomicalMarkerDefaultWeight(s_t weight)
{
  mAnatomicalMarkerDefaultWeight = weight;
}

//==============================================================================
/// This is the default weight that gets assigned to tracking markers during
/// IK, if nothing marker-specific gets assigned.
void MarkerFitter::setTrackingMarkerDefaultWeight(s_t weight)
{
  mTrackingMarkerDefaultWeight = weight;
}

//==============================================================================
/// This allows us to encourage the pelvis to align in a way that minimizes
/// the total joint angles for the hips and the lower back, which are
/// otherwise very ambiguous. This is presented as a general API in case we
/// want to apply the same logic to other joints.
void MarkerFitter::setRegularizeJointWithVirtualSpring(
    std::string jointName, s_t weight)
{
  mJointVirtualSpringRegularizerWeight[jointName] = weight;
}

//==============================================================================
/// This will heuristically detect the pelvis, and attached joints, and mark
/// them all as needing to be regularized with a virtual spring during IK.
void MarkerFitter::setRegularizePelvisJointsWithVirtualSpring(s_t weight)
{
  std::vector<std::string> namesToLookFor;
  namesToLookFor.push_back("pelvis");

  for (auto* body : mSkeleton->getBodyNodes())
  {
    // 1. Check if the body name contains the name we're looking for
    bool isMatch = false;
    for (std::string nameToLookFor : namesToLookFor)
    {
      if (body->getName().find(nameToLookFor) != std::string::npos)
      {
        isMatch = true;
        break;
      }
    }

    // 2. Check if the attached mesh names contains the name we're looking for
    for (auto* shapeNode : body->getShapeNodes())
    {
      dynamics::ShapePtr shape = shapeNode->getShape();
      if (shape->getType() == dynamics::MeshShape::getStaticType())
      {
        auto mesh = std::static_pointer_cast<dynamics::MeshShape>(shape);

        for (std::string nameToLookFor : namesToLookFor)
        {
          if (mesh->getMeshPath().find(nameToLookFor) != std::string::npos)
          {
            isMatch = true;
            break;
          }
        }
      }
      if (isMatch)
      {
        break;
      }
    }

    // 3. If we think we've found the pelvis...
    if (isMatch)
    {
      // 4. ... then we add all the joints attached to it to the list of joints
      // we want to regularize with a virtual spring
      if (body->getParentJoint() != mSkeleton->getRootJoint())
      {
        setRegularizeJointWithVirtualSpring(
            body->getParentJoint()->getName(), weight);
      }
      for (int i = 0; i < body->getNumChildJoints(); i++)
      {
        setRegularizeJointWithVirtualSpring(
            body->getChildJoint(i)->getName(), weight);
      }
    }
  }
}

//==============================================================================
/// This returns a score summarizing how much the markers attached to this
/// joint move relative to one another.
s_t MarkerFitter::computeJointVariability(
    dynamics::Joint* joint,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  std::vector<std::string> markerNames;
  for (auto pair : mMarkerMap)
  {
    if (joint->getParentBodyNode()
        && (pair.second.first->getName()
                == joint->getParentBodyNode()->getName()
            || pair.second.first->getName()
                   == joint->getChildBodyNode()->getName()))
    {
      markerNames.push_back(pair.first);
    }
  }

  // 1. Go through and find the mean distances

  Eigen::MatrixXs pairMeans
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  Eigen::MatrixXs pairObservationCounts
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (int i = 0; i < markerNames.size() - 1; i++)
    {
      if (markerObservations[t].count(markerNames[i]))
      {
        for (int j = i + 1; j < markerNames.size(); j++)
        {
          if (markerObservations[t].count(markerNames[j]))
          {
            s_t dist = (markerObservations[t].at(markerNames[i])
                        - markerObservations[t].at(markerNames[j]))
                           .norm();
            pairMeans(i, j) += dist;
            pairObservationCounts(i, j) += 1;
          }
        }
      }
    }
  }

  pairMeans = pairMeans.cwiseQuotient(pairObservationCounts);

  // 2. Compute the variance

  Eigen::MatrixXs pairVariance
      = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (int i = 0; i < markerNames.size() - 1; i++)
    {
      if (markerObservations[t].count(markerNames[i]))
      {
        for (int j = i + 1; j < markerNames.size(); j++)
        {
          if (markerObservations[t].count(markerNames[j]))
          {
            s_t dist = (markerObservations[t].at(markerNames[i])
                        - markerObservations[t].at(markerNames[j]))
                           .norm();
            s_t diff = dist - pairMeans(i, j);
            pairVariance(i, j) += diff * diff;
          }
        }
      }
    }
  }

  pairVariance.cwiseQuotient(pairObservationCounts);

  if (mDebugJointVariability)
  {
    std::cout << "Computing joint variability for \"" << joint->getName()
              << "\"" << std::endl
              << "Pair means: " << std::endl
              << pairMeans << std::endl
              << "Pair observation counts: " << std::endl
              << pairObservationCounts << std::endl
              << "Pair variance: " << std::endl
              << pairVariance << std::endl;
  }

  // 3. Go through and compute the sum normalized RMSE

  s_t sum = 0.0;
  for (int i = 0; i < markerNames.size(); i++)
  {
    for (int j = 0; j < markerNames.size(); j++)
    {
      if (pairObservationCounts(i, j) > 0)
      {
        sum += sqrt(pairVariance(i, j)) / pairMeans(i, j);
      }
    }
  }

  return sum;
}

//==============================================================================
/// This lets us pick a subset of the marker observations, to cap the size of
/// the optimization problem.
std::vector<std::map<std::string, Eigen::Vector3s>> MarkerFitter::pickSubset(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    int maxSize)
{
  if (maxSize >= markerObservations.size())
  {
    return markerObservations;
  }

  // Create a vector of indices, random shuffle them, then use them to select
  // the elements we want
  std::vector<unsigned int> indices(markerObservations.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  std::vector<std::map<std::string, Eigen::Vector3s>> result;
  for (int i = 0; i < maxSize; i++)
  {
    result.push_back(markerObservations[indices[i]]);
  }

  return result;
}

//==============================================================================
/// All markers are either "anatomical" or "tracking". Markers are presumed to
/// be anamotical markers unless otherwise specified. Tracking markers are
/// treated differently - they're not used in the initial scaling and fitting,
/// and their initial positions are not trusted at all. Instead, during
/// initialization, we guess their offset based on where the markers are
/// observed to be.
void MarkerFitter::setMarkerIsTracking(std::string marker, bool isTracking)
{
  mMarkerIsTracking[mMarkerIndices[marker]] = isTracking;
}

//==============================================================================
/// This returns true if the given marker is "tracking", otherwise it's
/// "anatomical"
bool MarkerFitter::getMarkerIsTracking(std::string marker)
{
  return mMarkerIsTracking[mMarkerIndices[marker]];
}

//==============================================================================
/// This auto-labels any markers whose names end with '1', '2', or '3' as
/// tracking markers, on the assumption that they're tracking triads.
void MarkerFitter::setTriadsToTracking()
{
  std::unordered_map<std::string, int> prefixCount;
  for (int i = 0; i < getNumMarkers(); i++)
  {
    std::string markerName = getMarkerNameAtIndex(i);
    char lastChar = markerName[markerName.size() - 1];
    if (lastChar == '1' || lastChar == '2' || lastChar == '3' || lastChar == '4'
        || lastChar == '5' || lastChar == '6' || lastChar == '7')
    {
      std::string prefix = markerName.substr(0, markerName.size() - 1);
      if (prefixCount.count(prefix) == 0)
      {
        prefixCount[prefix] = 0;
      }
      prefixCount[prefix]++;
    }
  }
  for (int i = 0; i < getNumMarkers(); i++)
  {
    std::string markerName = getMarkerNameAtIndex(i);
    std::string prefix = markerName.substr(0, markerName.size() - 1);
    if (prefixCount.count(prefix) > 0 && prefixCount[prefix] > 1)
    {
      setMarkerIsTracking(markerName);
    }
  }
}

//==============================================================================
/// If we load a list of tracking markers from the OpenSim file, we can
void MarkerFitter::setTrackingMarkers(const std::vector<std::string>& tracking)
{
  for (int i = 0; i < getNumMarkers(); i++)
  {
    std::string markerName = getMarkerNameAtIndex(i);
    bool isTracking = std::find(tracking.begin(), tracking.end(), markerName)
                      != tracking.end();
    setMarkerIsTracking(markerName, isTracking);
  }
}

//==============================================================================
/// Gets the total number of markers we've got in this Fitter
int MarkerFitter::getNumMarkers()
{
  return mMarkerNames.size();
}

//==============================================================================
/// Internally all the markers are concatenated together, so each index has a
/// name.
std::string MarkerFitter::getMarkerNameAtIndex(int index)
{
  return mMarkerNames[index];
}

//==============================================================================
/// Internally all the markers are concatenated together, so each index has a
/// name.
int MarkerFitter::getMarkerIndex(std::string name)
{
  return mMarkerIndices[name];
}

//==============================================================================
/// This method will set `skeleton` to the configuration given by the vectors
/// of jointPositions and groupScales. It will also compute and return the
/// list of markers given by markerDiffs.
std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
MarkerFitter::setConfiguration(
    std::shared_ptr<dynamics::Skeleton>& skeleton,
    Eigen::VectorXs jointPositions,
    Eigen::VectorXs groupScales,
    Eigen::VectorXs markerDiffs)
{
  skeleton->setPositions(jointPositions);
  skeleton->setGroupScales(groupScales);
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> adjustedMarkers;
  for (int i = 0; i < mMarkers.size(); i++)
  {
    adjustedMarkers.push_back(
        std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
            skeleton->getBodyNode(mMarkers[i].first->getName()),
            mMarkers[i].second + markerDiffs.segment<3>(i * 3)));
  }
  return adjustedMarkers;
}

//==============================================================================
/// This computes a vector of concatenated differences between where markers
/// are and where the observed markers are. Unobserved markers are assumed to
/// have a difference of zero.
Eigen::VectorXs MarkerFitter::getMarkerError(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::VectorXs observedMarkerDiffs
      = Eigen::VectorXs::Zero(mMarkers.size() * 3);

  Eigen::VectorXs adjustedMarkerWorldPoses
      = skeleton->getMarkerWorldPositions(markers);

  for (auto pair : visibleMarkerWorldPoses)
  {
    observedMarkerDiffs.segment<3>(pair.first * 3)
        = adjustedMarkerWorldPoses.segment<3>(pair.first * 3) - pair.second;
  }

  return observedMarkerDiffs;
}

//==============================================================================
/// This gets the overall objective term for the MarkerFitter for a single
/// timestep. The MarkerFitter is trying to do a bilevel optimization to
/// minimize this term.
s_t MarkerFitter::computeIKLoss(Eigen::VectorXs markerError)
{
  return markerError.squaredNorm();
}

//==============================================================================
/// This returns the gradient for the simple IK loss term
Eigen::VectorXs MarkerFitter::getIKLossGradWrtMarkerError(
    Eigen::VectorXs markerError)
{
  return 2 * markerError;
}

//==============================================================================
/// This lets us print the components of the loss, to allow easier tuning of
/// different weights
void MarkerFitter::setDebugLoss(bool debug)
{
  mDebugLoss = debug;
}

//==============================================================================
/// During random-restarts on IK, when we find solutions below this loss we'll
/// stop doing restarts early, to speed up the process.
void MarkerFitter::setInitialIKSatisfactoryLoss(s_t loss)
{
  mInitialIKSatisfactoryLoss = loss;
}

//==============================================================================
/// This sets the maximum number of restarts allowed for the initial IK solver
void MarkerFitter::setInitialIKMaxRestarts(int restarts)
{
  mInitialIKMaxRestarts = restarts;
}

//==============================================================================
/// This gives us a configuration option to ignore the joint limits in the
/// uploaded model, and then set them after the fit.
void MarkerFitter::setIgnoreJointLimits(bool ignore)
{
  mIgnoreJointLimits = ignore;
}

//==============================================================================
/// Sets the maximum that we'll allow markers to move from their original
/// position, in meters
void MarkerFitter::setMaxMarkerOffset(s_t offset)
{
  mMaxMarkerOffset = offset;
}

//==============================================================================
/// Sets the maximum number of iterations for IPOPT
void MarkerFitter::setIterationLimit(int limit)
{
  mIterationLimit = limit;
}

//==============================================================================
/// Sets the number of SGD iterations to run when fitting joint center
/// problems
void MarkerFitter::setJointSphereFitSGDIterations(int iters)
{
  mJointSphereFitSGDIterations = iters;
}

//==============================================================================
/// Sets the number of SGD iterations to run when fitting joint axis
/// problems
void MarkerFitter::setJointAxisFitSGDIterations(int iters)
{
  mJointAxisFitSGDIterations = iters;
}

//==============================================================================
/// This sets an anthropometric prior which is used by the default loss. If
/// you've called `setCustomLossAndGrad` then this has no effect.
void MarkerFitter::setAnthropometricPrior(
    std::shared_ptr<biomechanics::Anthropometrics> prior, s_t weight)
{
  mAnthropometrics = prior;
  mAnthropometricWeight = weight;
}

//==============================================================================
/// This sets the height (in meters) that the model should be scaled to, and
/// the weight we should use when enforcing that scaling. This is in some
/// sense redundant to the anthropometric prior, but it's useful to have a
/// separate term for this, because it allows us to weight the height
/// constraint more heavily, because it's closer to a "hard constraint" than
/// the other anthropometric priors.
void MarkerFitter::setExplicitHeightPrior(s_t height, s_t weight)
{
  mHeightPrior = height;
  mHeightPriorWeight = weight;
}

//==============================================================================
/// This sets the data from a static trial, which we can use to resolve some
/// forms of pelvis and foot ambiguity.
void MarkerFitter::setStaticTrial(
    std::map<std::string, Eigen::Vector3s> markerObservations,
    Eigen::VectorXs pose)
{
  mStaticTrialEnabled = true;
  mStaticTrialPose = pose;

  mStaticTrialMarkerNames.clear();
  for (auto& pair : mMarkerMap)
  {
    if (markerObservations.count(pair.first))
    {
      mStaticTrialMarkerNames.push_back(pair.first);
    }
  }
  mStaticTrialMarkerPositions
      = Eigen::VectorXs::Zero(mStaticTrialMarkerNames.size() * 3);
  for (int i = 0; i < mStaticTrialMarkerNames.size(); i++)
  {
    mStaticTrialMarkerPositions.segment<3>(i * 3)
        = markerObservations.at(mStaticTrialMarkerNames[i]);
  }
}

//==============================================================================
/// This sets how heavily to weight the static trial in our optimization,
/// compared to other terms.
void MarkerFitter::setStaticTrialWeight(s_t weight)
{
  mStaticTrialWeight = weight;
}

//==============================================================================
/// This sets the minimum distance joints have to be apart in order to get zero
/// "force field" loss. Any joints closer than this (in world space) will incur
/// a penalty.
void MarkerFitter::setJointForceFieldThresholdDistance(s_t minDistance)
{
  mJointForceFieldThresholdDistance = minDistance;
}

//==============================================================================
/// Larger values will increase the softness of the threshold penalty. Smaller
/// values, as they approach zero, will have an almost perfectly vertical
/// penality for going below the threshold distance. That would be hard to
/// optimize, so don't make it too small.
void MarkerFitter::setJointForceFieldSoftness(s_t softness)
{
  mJointForceFieldSoftness = softness;
}

//==============================================================================
/// If we set this to true, then after the main optimization completes we will
/// do a final step to "center" the error of the anatomical markers. This
/// minimizes marker RMSE, but does NOT respect the weights about how far
/// markers should be allowed to move.
void MarkerFitter::setPostprocessAnatomicalMarkerOffsets(bool postprocess)
{
  mPostprocessAnatomicalMarkerOffsets = postprocess;
}

//==============================================================================
/// If we set this to true, then after the main optimization completes we will
/// do a final step to "center" the error of the tracking markers. This
/// minimizes marker RMSE, but does NOT respect the weights about how far
/// markers should be allowed to move.
void MarkerFitter::setPostprocessTrackingMarkerOffsets(bool postprocess)
{
  mPostprocessTrackingMarkerOffsets = postprocess;
}

//==============================================================================
/// Sets the loss and gradient function
void MarkerFitter::setCustomLossAndGrad(
    std::function<s_t(MarkerFitterState*)> customLossAndGrad)
{
  mLossAndGrad = customLossAndGrad;
}

//==============================================================================
/// This adds a custom function as an equality constraint to the problem. The
/// constraint has to equal 0.
void MarkerFitter::addZeroConstraint(
    std::string name,
    std::function<s_t(MarkerFitterState*)> customConstraintAndGrad)
{
  mZeroConstraints[name] = customConstraintAndGrad;
}

//==============================================================================
/// This removes an equality constraint by name
void MarkerFitter::removeZeroConstraint(std::string name)
{
  mZeroConstraints.erase(name);
}

//==============================================================================
/// This writes a unified CSV with columns describing the results for the trial
  /// associated with this MarkerInitialization.
void MarkerFitter::writeCSVData(
    std::string path,
    const MarkerInitialization& init,
    const std::vector<s_t>& rmsMarkerErrors,
    const std::vector<s_t>& maxMarkerErrors,
    const std::vector<s_t>& timestamps)
{
  // Convenience function.
  auto writeVectorToCSV = [](std::ofstream& csvFile, Eigen::VectorXs& vec) {
    for (int i = 0; i < vec.size(); i++) {
      csvFile << "," << vec(i);
    }
  };

  std::ofstream csvFile;
  csvFile.open(path);

  csvFile << "time";
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",pos_" << mSkeleton->getDof(i)->getName();
  }
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",vel_" << mSkeleton->getDof(i)->getName();
  }
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",acc_" << mSkeleton->getDof(i)->getName();
  }
  csvFile << ",marker_error_rms,marker_error_max";

  (void)init;
  int nrows = init.poses.cols();
  s_t dt = timestamps[1] - timestamps[0];
  assert(rmsMarkerErrors.size() == nrows);
  assert(maxMarkerErrors.size() == nrows);
  for (int t = 0; t < nrows; t++)
  {
    csvFile << std::endl;
    csvFile << timestamps[t];
    Eigen::VectorXs q = init.poses.col(t);
    Eigen::VectorXs dq = Eigen::VectorXs::Zero(q.size());
    Eigen::VectorXs ddq = Eigen::VectorXs::Zero(q.size());
    if (t > 0 && t < nrows - 1)
    {
      dq = (init.poses.col(t) - init.poses.col(t - 1)) / dt;
      ddq = (init.poses.col(t + 1) - 2 * init.poses.col(t)
             + init.poses.col(t - 1)) / (dt * dt);
    }

    writeVectorToCSV(csvFile, q);
    writeVectorToCSV(csvFile, dq);
    writeVectorToCSV(csvFile, ddq);

    // Marker errors.
    csvFile << "," << rmsMarkerErrors[t] << "," << maxMarkerErrors[t];
  }

  csvFile.close();
}

//==============================================================================
/// This gets the gradient of the objective wrt the joint positions
Eigen::VectorXs MarkerFitter::getMarkerLossGradientWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs lossGradWrtMarkerError)
{
  return skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers)
             .transpose()
         * lossGradWrtMarkerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the joint positions
Eigen::VectorXs
MarkerFitter::finiteDifferenceSquaredMarkerLossGradientWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::VectorXs originalPos = skeleton->getPositions();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(originalPos.size());

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalPos;
  for (int i = 0; i < originalPos.size(); i++)
  {
    perturbed = originalPos;
    perturbed(i) += EPS;
    skeleton->setPositions(perturbed);

    s_t plus = computeIKLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalPos;
    perturbed(i) -= EPS;
    skeleton->setPositions(perturbed);

    s_t minus = computeIKLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    grad(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setPositions(originalPos);

  return grad;
}

//==============================================================================
/// This gets the gradient of the objective wrt the group scales
Eigen::VectorXs MarkerFitter::getMarkerLossGradientWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs lossGradWrtMarkerError)
{
  return skeleton->getMarkerWorldPositionsJacobianWrtGroupScales(markers)
             .transpose()
         * lossGradWrtMarkerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the group scales
Eigen::VectorXs
MarkerFitter::finiteDifferenceSquaredMarkerLossGradientWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::VectorXs originalGroupScales = skeleton->getGroupScales();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(originalGroupScales.size());

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalGroupScales;
  for (int i = 0; i < originalGroupScales.size(); i++)
  {
    perturbed = originalGroupScales;
    perturbed(i) += EPS;
    skeleton->setGroupScales(perturbed);

    s_t plus = computeIKLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalGroupScales;
    perturbed(i) -= EPS;
    skeleton->setGroupScales(perturbed);

    s_t minus = computeIKLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    grad(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setGroupScales(originalGroupScales);

  return grad;
}

//==============================================================================
/// This gets the gradient of the objective wrt the marker offsets
Eigen::VectorXs MarkerFitter::getMarkerLossGradientWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    Eigen::VectorXs lossGradWrtMarkerError)
{
  return skeleton->getMarkerWorldPositionsJacobianWrtMarkerOffsets(markers)
             .transpose()
         * lossGradWrtMarkerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the marker offsets
Eigen::VectorXs
MarkerFitter::finiteDifferenceSquaredMarkerLossGradientWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
        &(*markers[i].first), Eigen::Vector3s(markers[i].second)));
  }

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      markersCopy[i].second(axis) = markers[i].second(axis) + EPS;

      s_t plus = computeIKLoss(
          getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses));

      markersCopy[i].second(axis) = markers[i].second(axis) - EPS;

      s_t minus = computeIKLoss(
          getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses));

      markersCopy[i].second(axis) = markers[i].second(axis);

      grad(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return grad;
}

//==============================================================================
/// Get the marker indices that are not visible
std::vector<int> MarkerFitter::getSparsityMap(
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  std::vector<int> sparsityMap;
  for (int i = 0; i < markers.size(); i++)
  {
    sparsityMap.push_back(i);
  }
  for (auto pair : visibleMarkerWorldPoses)
  {
    // Find the position of the observed marker index in sparsityMap
    std::vector<int>::iterator position
        = std::find(sparsityMap.begin(), sparsityMap.end(), pair.first);
    if (position
        != sparsityMap
               .end()) // == sparsityMap.end() means the element was not found
      sparsityMap.erase(position);
  }
  return sparsityMap;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the joints
Eigen::MatrixXs MarkerFitter::getMarkerErrorJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  // Clear out the sections of the Jacobian that were not observed, since
  // those won't change the error
  for (int i : sparsityMap)
  {
    jac.block(i * 3, 0, 3, jac.cols()).setZero();
  }

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the joints
Eigen::MatrixXs MarkerFitter::finiteDifferenceMarkerErrorJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, skeleton->getNumDofs());

  Eigen::VectorXs originalPos = skeleton->getPositions();

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalPos;
  for (int i = 0; i < originalPos.size(); i++)
  {
    perturbed = originalPos;
    perturbed(i) += EPS;
    skeleton->setPositions(perturbed);

    Eigen::VectorXs plus
        = getMarkerError(skeleton, markers, visibleMarkerWorldPoses);

    perturbed = originalPos;
    perturbed(i) -= EPS;
    skeleton->setPositions(perturbed);

    Eigen::VectorXs minus
        = getMarkerError(skeleton, markers, visibleMarkerWorldPoses);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the joint positions
Eigen::MatrixXs MarkerFitter::getIKLossGradientWrtJointsJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const Eigen::VectorXs& markerError,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs firstOrderJac
      = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  // First order grad:
  // 2 * markerError.transpose() * firstOrderJac

  // We'll use the product rule to get a derivative

  Eigen::MatrixXs secondOrderJac
      = skeleton
            ->getMarkerWorldPositionsSecondJacobianWrtJointWrtJointPositions(
                markers, markerError);

  // (d/dq markerError) is getMarkerErrorJacobianWrtJoints(...)
  // markerError.transpose() * (d/dq firstOrderJac) is secondOrderJac

  return 2
         * ((firstOrderJac.transpose()
             * getMarkerErrorJacobianWrtJoints(skeleton, markers, sparsityMap))
            + secondOrderJac);
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the joint positions
Eigen::MatrixXs
MarkerFitter::finiteDifferenceIKLossGradientWrtJointsJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(skeleton->getNumDofs(), skeleton->getNumDofs());

  Eigen::VectorXs originalPos = skeleton->getPositions();

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalPos;
  for (int i = 0; i < originalPos.size(); i++)
  {
    perturbed = originalPos;
    perturbed(i) += EPS;
    skeleton->setPositions(perturbed);

    Eigen::VectorXs plus = getMarkerLossGradientWrtJoints(
        skeleton,
        markers,
        getIKLossGradWrtMarkerError(
            getMarkerError(skeleton, markers, visibleMarkerWorldPoses)));

    perturbed = originalPos;
    perturbed(i) -= EPS;
    skeleton->setPositions(perturbed);

    Eigen::VectorXs minus = getMarkerLossGradientWrtJoints(
        skeleton,
        markers,
        getIKLossGradWrtMarkerError(
            getMarkerError(skeleton, markers, visibleMarkerWorldPoses)));

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the group scales
Eigen::MatrixXs MarkerFitter::getMarkerErrorJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtGroupScales(markers);

  // Clear out the sections of the Jacobian that were not observed, since
  // those won't change the error
  for (int i : sparsityMap)
  {
    jac.block(i * 3, 0, 3, jac.cols()).setZero();
  }

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the group scales
Eigen::MatrixXs MarkerFitter::finiteDifferenceMarkerErrorJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, skeleton->getGroupScaleDim());

  Eigen::VectorXs originalGroupScales = skeleton->getGroupScales();

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalGroupScales;
  for (int i = 0; i < originalGroupScales.size(); i++)
  {
    perturbed = originalGroupScales;
    perturbed(i) += EPS;
    skeleton->setGroupScales(perturbed);

    Eigen::VectorXs plus
        = getMarkerError(skeleton, markers, visibleMarkerWorldPoses);

    perturbed = originalGroupScales;
    perturbed(i) -= EPS;
    skeleton->setGroupScales(perturbed);

    Eigen::VectorXs minus
        = getMarkerError(skeleton, markers, visibleMarkerWorldPoses);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setGroupScales(originalGroupScales);

  return jac;
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the group scales
Eigen::MatrixXs MarkerFitter::getIKLossGradientWrtJointsJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const Eigen::VectorXs& markerError,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs firstOrderJac
      = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  // First order grad:
  // 2 * markerError.transpose() * firstOrderJac

  // We'll use the product rule to get a derivative

  Eigen::MatrixXs secondOrderJac
      = skeleton->getMarkerWorldPositionsSecondJacobianWrtJointWrtGroupScales(
          markers, markerError);

  // (d/dq markerError) is getMarkerErrorJacobianWrtJoints(...)
  // markerError.transpose() * (d/dq firstOrderJac) is secondOrderJac

  return 2
         * ((firstOrderJac.transpose()
             * getMarkerErrorJacobianWrtGroupScales(
                 skeleton, markers, sparsityMap))
            + secondOrderJac);
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the group scales
Eigen::MatrixXs
MarkerFitter::finiteDifferenceIKLossGradientWrtJointsJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(
      skeleton->getNumDofs(), skeleton->getGroupScaleDim());

  Eigen::VectorXs originalGroupScales = skeleton->getGroupScales();

  const s_t EPS = 1e-7;

  Eigen::VectorXs perturbed = originalGroupScales;
  for (int i = 0; i < originalGroupScales.size(); i++)
  {
    perturbed = originalGroupScales;
    perturbed(i) += EPS;
    skeleton->setGroupScales(perturbed);

    Eigen::VectorXs plus = getMarkerLossGradientWrtJoints(
        skeleton,
        markers,
        getIKLossGradWrtMarkerError(
            getMarkerError(skeleton, markers, visibleMarkerWorldPoses)));

    perturbed = originalGroupScales;
    perturbed(i) -= EPS;
    skeleton->setGroupScales(perturbed);

    Eigen::VectorXs minus = getMarkerLossGradientWrtJoints(
        skeleton,
        markers,
        getIKLossGradWrtMarkerError(
            getMarkerError(skeleton, markers, visibleMarkerWorldPoses)));

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setGroupScales(originalGroupScales);

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the marker offsets
Eigen::MatrixXs MarkerFitter::getMarkerErrorJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtMarkerOffsets(markers);

  // Clear out the sections of the Jacobian that were not observed, since
  // those won't change the error
  for (int i : sparsityMap)
  {
    jac.block(i * 3, 0, 3, jac.cols()).setZero();
  }

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the marker offsets
Eigen::MatrixXs
MarkerFitter::finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
        &(*markers[i].first), Eigen::Vector3s(markers[i].second)));
  }

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      markersCopy[i].second(axis) = markers[i].second(axis) + EPS;

      Eigen::VectorXs plus
          = getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses);

      markersCopy[i].second(axis) = markers[i].second(axis) - EPS;

      Eigen::VectorXs minus
          = getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses);

      markersCopy[i].second(axis) = markers[i].second(axis);

      jac.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return jac;
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the marker offsets
Eigen::MatrixXs
MarkerFitter::getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const Eigen::VectorXs& markerError,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs firstOrderJac
      = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  // First order grad:
  // 2 * markerError.transpose() * firstOrderJac

  // We'll use the product rule to get a derivative

  Eigen::MatrixXs secondOrderJac
      = skeleton->getMarkerWorldPositionsSecondJacobianWrtJointWrtMarkerOffsets(
          markers, markerError);

  // (d/dq markerError) is getMarkerErrorJacobianWrtJoints(...)
  // markerError.transpose() * (d/dq firstOrderJac) is secondOrderJac

  return 2
         * ((firstOrderJac.transpose()
             * getMarkerErrorJacobianWrtMarkerOffsets(
                 skeleton, markers, sparsityMap))
            + secondOrderJac);
}

//==============================================================================
/// This gets the jacobian of (the gradient of the objective wrt the joint
/// positions) wrt the marker offsets
Eigen::MatrixXs
MarkerFitter::finiteDifferenceIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(skeleton->getNumDofs(), markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(std::make_pair<dynamics::BodyNode*, Eigen::Vector3s>(
        &(*markers[i].first), Eigen::Vector3s(markers[i].second)));
  }

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      markersCopy[i].second(axis) = markers[i].second(axis) + EPS;

      Eigen::VectorXs plus = getMarkerLossGradientWrtJoints(
          skeleton,
          markersCopy,
          getIKLossGradWrtMarkerError(
              getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses)));

      markersCopy[i].second(axis) = markers[i].second(axis) - EPS;

      Eigen::VectorXs minus = getMarkerLossGradientWrtJoints(
          skeleton,
          markersCopy,
          getIKLossGradWrtMarkerError(
              getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses)));

      markersCopy[i].second(axis) = markers[i].second(axis);

      jac.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return jac;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
void MarkerFitter::setLBFGSHistory(int hist)
{
  mLBFGSHistoryLength = hist;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
void MarkerFitter::setCheckDerivatives(bool checkDerivatives)
{
  mCheckDerivatives = checkDerivatives;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
void MarkerFitter::setParallelIKWarps(bool parallelWarps)
{
  mUseParallelIKWarps = parallelWarps;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// The SphereFitJointCenterProblem, which maps the sphere-fitting joint-center
// problem onto a differentiable format.
///////////////////////////////////////////////////////////////////////////////////////////////////////

//==============================================================================
SphereFitJointCenterProblem::SphereFitJointCenterProblem(
    MarkerFitter* fitter,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    Eigen::MatrixXs ikPoses,
    dynamics::Joint* joint,
    const std::vector<bool>& newClip,
    Eigen::Ref<Eigen::MatrixXs> out)
  : mFitter(fitter),
    mMarkerObservations(markerObservations),
    mOut(out),
    mJointName(joint->getName()),
    mNewClip(newClip),
    mSmoothingLoss(
        0.1) // just to tie break when there's nothing better available
{
  mNumTimesteps = markerObservations.size();

  // 1. Figure out which markers are on BodyNode's adjacent to the joint

  for (auto pair : fitter->mMarkerMap)
  {
    if (isDynamicParentOfJoint(pair.second.first->getName(), joint)
        || isDynamicChildOfJoint(pair.second.first->getName(), joint))
    {
      // Only add the markers if we see them observed at least once in the
      // dataset. If it's never observed, then we'll end up having all sorts of
      // divide by zeros
      for (int i = 0; i < mNumTimesteps; i++)
      {
        if (mMarkerObservations[i].count(pair.first) > 0)
        {
          mActiveMarkers.push_back(pair.first);
          break;
        }
      }
    }
  }

  // 1.1. If there aren't enough markers, throw a warning and return

  if (mActiveMarkers.size() < 3)
  {
    std::cout << "WARNING! Trying to instantiate a "
                 "SphereFitJointCenterProblem, but only have "
              << mActiveMarkers.size()
              << " markers on BodyNode's adjacent to chosen Joint \""
              << joint->getName() << "\"" << std::endl;
    return;
  }

  // 2. Go through and initialize the problem

  mMarkerPositions
      = Eigen::MatrixXs::Zero(mActiveMarkers.size() * 3, mNumTimesteps);
  mMarkerObserved = Eigen::MatrixXi::Zero(mActiveMarkers.size(), mNumTimesteps);
  mRadii = Eigen::VectorXs::Zero(mActiveMarkers.size());
  mCenterPoints = Eigen::VectorXs::Zero(3 * mNumTimesteps);

  Eigen::VectorXi numRadiiObservations
      = Eigen::VectorXi::Zero(mActiveMarkers.size());

  Eigen::VectorXs originalPosition = mFitter->mSkeleton->getPositions();
  std::vector<dynamics::Joint*> jointVec;
  jointVec.push_back(joint);

  for (int i = 0; i < mNumTimesteps; i++)
  {
    mFitter->mSkeleton->setPositions(ikPoses.col(i));
    mCenterPoints.segment<3>(i * 3)
        = mFitter->mSkeleton->getJointWorldPositions(jointVec);
    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      std::string name = mActiveMarkers[j];
      if (mMarkerObservations[i].count(name) > 0)
      {
#ifndef NDEBUG
        if (mMarkerObservations[i][name].hasNaN())
        {
          std::cout << "MARKER NaN DETECTED!! timestep " << i << " name "
                    << name << ": " << mMarkerObservations[i][name]
                    << std::endl;
          exit(1);
        }
#endif
        mMarkerPositions.block<3, 1>(j * 3, i) = mMarkerObservations[i][name];
        mMarkerObserved(j, i) = 1;
        mRadii(j)
            += (mCenterPoints.segment<3>(i * 3) - mMarkerObservations[i][name])
                   .norm();
        numRadiiObservations(j)++;
      }
    }
  }

  mFitter->mSkeleton->setPositions(originalPosition);

  for (int j = 0; j < mActiveMarkers.size(); j++)
  {
    if (numRadiiObservations(j) > 0)
    {
      mRadii(j) /= numRadiiObservations(j);
    }
  }

#ifndef NDEBUG
  if (mRadii.hasNaN())
  {
    std::cout << "mRadii.hasNaN(): " << mRadii.hasNaN() << std::endl;
    std::cout << "mCenterPoints.hasNaN(): " << mCenterPoints.hasNaN()
              << std::endl;
    std::cout << "mRadii: " << mRadii << std::endl;
    std::cout << "numRadiiObservations: " << numRadiiObservations << std::endl;
    exit(1);
  }
#endif
}

//==============================================================================
/// This returns true if the given body is the parent of the joint OR if
/// there's a hierarchy of fixed joints that connect it to the parent
bool SphereFitJointCenterProblem::isDynamicParentOfJoint(
    std::string bodyName, dynamics::Joint* joint)
{
  while (true)
  {
    if (joint->getParentBodyNode() == nullptr)
      return false;
    if (bodyName == joint->getParentBodyNode()->getName())
    {
      return true;
    }
    // Recurse up the chain, as long as we're traversing only fixed joints
    if (joint->getParentBodyNode()->getParentJoint() != nullptr
        && joint->getParentBodyNode()->getParentJoint()->isFixed())
    {
      joint = joint->getParentBodyNode()->getParentJoint();
    }
    else
    {
      return false;
    }
  }
}

//==============================================================================
/// This returns true if the given body is the child of the joint OR if
/// there's a hierarchy of fixed joints that connect it to the child
bool SphereFitJointCenterProblem::isDynamicChildOfJoint(
    std::string bodyName, dynamics::Joint* joint)
{
  while (true)
  {
    if (joint->getChildBodyNode() == nullptr)
      return false;
    if (bodyName == joint->getChildBodyNode()->getName())
    {
      return true;
    }
    // Recurse down the chain, as long as we're traversing only fixed joints
    if (joint->getChildBodyNode()->getNumChildJoints() == 1
        && joint->getChildBodyNode()->getChildJoint(0)->isFixed())
    {
      joint = joint->getChildBodyNode()->getChildJoint(0);
    }
    else
    {
      return false;
    }
  }
}

//==============================================================================
bool SphereFitJointCenterProblem::canFitJoint(
    MarkerFitter* fitter,
    dynamics::Joint* joint,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // We can't fit locked joints
  if (joint->isFixed())
  {
    return false;
  }

  // We can't fit root joints
  if (joint->getParentBodyNode() == nullptr)
  {
    return false;
  }

  int numActiveParents = 0;
  int numActiveChildren = 0;

  for (auto pair : fitter->mMarkerMap)
  {
    if (isDynamicParentOfJoint(pair.second.first->getName(), joint))
    {
      // Only add the markers if we see them observed at least once in the
      // dataset. If it's never observed, then we'll end up having all sorts of
      // divide by zeros
      for (int i = 0; i < markerObservations.size(); i++)
      {
        if (markerObservations[i].count(pair.first) > 0)
        {
          numActiveParents++;
          break;
        }
      }
    }
    if (isDynamicChildOfJoint(pair.second.first->getName(), joint))
    {
      // Only add the markers if we see them observed at least once in the
      // dataset. If it's never observed, then we'll end up having all sorts of
      // divide by zeros
      for (int i = 0; i < markerObservations.size(); i++)
      {
        if (markerObservations[i].count(pair.first) > 0)
        {
          numActiveChildren++;
          break;
        }
      }
    }
  }
  return numActiveParents > 0 && numActiveChildren > 0
         && (numActiveParents + numActiveChildren >= 3);
}

//==============================================================================
int SphereFitJointCenterProblem::getProblemDim()
{
  return mRadii.size() + mCenterPoints.size();
}

//==============================================================================
Eigen::VectorXs SphereFitJointCenterProblem::flatten()
{
  Eigen::VectorXs flat
      = Eigen::VectorXs::Zero(mRadii.size() + mCenterPoints.size());
  flat.segment(0, mRadii.size()) = mRadii;
  flat.segment(mRadii.size(), mCenterPoints.size()) = mCenterPoints;
  return flat;
}

//==============================================================================
void SphereFitJointCenterProblem::unflatten(Eigen::VectorXs x)
{
  mRadii = x.segment(0, mRadii.size());
  mCenterPoints = x.segment(mRadii.size(), mCenterPoints.size());
}

//==============================================================================
s_t SphereFitJointCenterProblem::getLoss()
{
  s_t loss = 0.0;

  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
    {
      if (!mNewClip[i])
      {
        loss += mSmoothingLoss
                * (mCenterPoints.segment<3>(i * 3)
                   - mCenterPoints.segment<3>((i - 1) * 3))
                      .squaredNorm();
      }
    }
    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      if (mMarkerObserved(j, i))
      {
        s_t diff = mRadii(j) * mRadii(j)
                   - (mCenterPoints.segment<3>(i * 3)
                      - mMarkerPositions.block<3, 1>(j * 3, i))
                         .squaredNorm();
        loss += diff * diff;
      }
    }
  }

  return loss;
}

//==============================================================================
Eigen::VectorXs SphereFitJointCenterProblem::getGradient()
{
  Eigen::VectorXs grad
      = Eigen::VectorXs::Zero(mRadii.size() + mCenterPoints.size());

  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
    {
      if (!mNewClip[i])
      {
        grad.segment<3>(mRadii.size() + i * 3)
            += 2 * mSmoothingLoss
               * (mCenterPoints.segment<3>(i * 3)
                  - mCenterPoints.segment<3>((i - 1) * 3));
        grad.segment<3>(mRadii.size() + (i - 1) * 3)
            -= 2 * mSmoothingLoss
               * (mCenterPoints.segment<3>(i * 3)
                  - mCenterPoints.segment<3>((i - 1) * 3));
      }
    }
    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      if (mMarkerObserved(j, i))
      {
        s_t diff = mRadii(j) * mRadii(j)
                   - (mCenterPoints.segment<3>(i * 3)
                      - mMarkerPositions.block<3, 1>(j * 3, i))
                         .squaredNorm();
        grad(j) += (2 * diff) * (2 * mRadii(j));
        grad.segment<3>(mRadii.size() + i * 3)
            += (2 * diff)
               * (-2
                  * (mCenterPoints.segment<3>(i * 3)
                     - mMarkerPositions.block<3, 1>(j * 3, i)));
      }
    }
  }

  return grad;
}

//==============================================================================
Eigen::VectorXs SphereFitJointCenterProblem::finiteDifferenceGradient()
{
  Eigen::VectorXs x = flatten();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());

  math::finiteDifference(
      [this, &x](
          /* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs tweaked = x;
        tweaked(dof) += eps;
        unflatten(tweaked);
        perturbed = getLoss();
        return true;
      },
      grad,
      1e-2,
      true);

  unflatten(x);

  return grad;
}

//==============================================================================
s_t SphereFitJointCenterProblem::saveSolutionBackToInitialization()
{
  for (int i = 0; i < mNumTimesteps; i++)
  {
    mOut.col(i) = mCenterPoints.segment<3>(i * 3);
  }
  return getLoss();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// The CylinderFitJointAxisProblem, which maps the sphere-fitting joint-center
// problem onto a differentiable format.
///////////////////////////////////////////////////////////////////////////////////////////////////////

//==============================================================================
CylinderFitJointAxisProblem::CylinderFitJointAxisProblem(
    MarkerFitter* fitter,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    Eigen::MatrixXs ikPoses,
    dynamics::Joint* joint,
    Eigen::MatrixXs centers,
    const std::vector<bool>& newClip,
    Eigen::Ref<Eigen::MatrixXs> out)
  : mFitter(fitter),
    mMarkerObservations(markerObservations),
    mOut(out),
    mJointName(joint->getName()),
    mJointCenters(centers),
    mNewClip(newClip),
    mKeepCenterLoss(0.01),
    mSmoothingCenterLoss(0.0),
    mSmoothingAxisLoss(1.0)
{
  mNumTimesteps = markerObservations.size();

  // 1. Figure out which markers are on BodyNode's adjacent to the joint

  for (auto pair : fitter->mMarkerMap)
  {
    if (SphereFitJointCenterProblem::isDynamicParentOfJoint(
            pair.second.first->getName(), joint)
        || SphereFitJointCenterProblem::isDynamicChildOfJoint(
            pair.second.first->getName(), joint))
    {
      // Only add the markers if we see them observed at least once in the
      // dataset. If it's never observed, then we'll end up having all sorts of
      // divide by zeros
      for (int i = 0; i < mNumTimesteps; i++)
      {
        if (mMarkerObservations[i].count(pair.first) > 0)
        {
          mActiveMarkers.push_back(pair.first);
          break;
        }
      }
    }
  }

  // 1.1. If there aren't enough markers, throw a warning and return

  if (mActiveMarkers.size() < 2)
  {
    std::cout << "WARNING! Trying to instantiate a "
                 "CylinderFitJointAxisProblem, but only have "
              << mActiveMarkers.size()
              << " markers on BodyNode's adjacent to chosen Joint \""
              << joint->getName() << "\"" << std::endl;
    return;
  }

  // 2. Go through and initialize the problem

  mMarkerPositions
      = Eigen::MatrixXs::Zero(mActiveMarkers.size() * 3, mNumTimesteps);
  mMarkerObserved = Eigen::MatrixXi::Zero(mActiveMarkers.size(), mNumTimesteps);
  mPerpendicularRadii = Eigen::VectorXs::Zero(mActiveMarkers.size());
  mParallelRadii = Eigen::VectorXs::Zero(mActiveMarkers.size());
  mAxisLines = Eigen::VectorXs::Zero(6 * mNumTimesteps);

  Eigen::VectorXi numRadiiObservations
      = Eigen::VectorXi::Zero(mActiveMarkers.size());

  Eigen::VectorXs originalPosition = mFitter->mSkeleton->getPositions();
  std::vector<dynamics::Joint*> jointVec;
  jointVec.push_back(joint);

  for (int i = 0; i < mNumTimesteps; i++)
  {
    mFitter->mSkeleton->setPositions(ikPoses.col(i));

    // Find the center points
    mAxisLines.segment<3>(i * 6)
        = mFitter->mSkeleton->getJointWorldPositions(jointVec);

    // Find the axis by the angle being formed by this joint
    Eigen::Vector3s angleAxis
        = math::logMap(joint->getRelativeTransform().linear());

    if (angleAxis.squaredNorm() < 0.01)
    {
      // Default to the previous axis, if the current axis is too small
      if (i > 0)
      {
        mAxisLines.segment<3>(i * 6 + 3)
            = mAxisLines.segment<3>((i - 1) * 6 + 3);
      }
      else
      {
        // Just pick an axis at random, if we're on the first timestep
        mAxisLines.segment<3>(i * 6 + 3) = Eigen::Vector3s::UnitX();
      }
    }
    else
    {
      mAxisLines.segment<3>(i * 6 + 3) = angleAxis.normalized();

      // Our axis can point in symmetrically in either direction, default to
      // pointing closer to the direction at the previous timestep to make
      // smoothing easier.
      if (i > 0)
      {
        Eigen::Vector3s thisTimestepNormal = mAxisLines.segment<3>(i * 6 + 3);
        Eigen::Vector3s lastTimestepNormal
            = mAxisLines.segment<3>((i - 1) * 6 + 3);
        s_t dist = (thisTimestepNormal - lastTimestepNormal).squaredNorm();
        s_t distFlipped
            = ((thisTimestepNormal * -1) - lastTimestepNormal).squaredNorm();
        if (distFlipped < dist)
        {
          mAxisLines.segment<3>(i * 6 + 3) = -1 * thisTimestepNormal;
        }
      }
    }

    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      std::string name = mActiveMarkers[j];
      if (mMarkerObservations[i].count(name) > 0)
      {
        mMarkerPositions.block<3, 1>(j * 3, i) = mMarkerObservations[i][name];
        mMarkerObserved(j, i) = 1;
        Eigen::Vector3s diff
            = mAxisLines.segment<3>(i * 6) - mMarkerObservations[i][name];
        // The radius is our distance to the cylinder at the nearest point
        mPerpendicularRadii(j) += (diff
                                   - (diff.dot(mAxisLines.segment<3>(i * 6 + 3))
                                      * mAxisLines.segment<3>(i * 6 + 3)))
                                      .norm();
        mParallelRadii(j) += (diff.dot(mAxisLines.segment<3>(i * 6 + 3))
                              * mAxisLines.segment<3>(i * 6 + 3))
                                 .norm();
        numRadiiObservations(j)++;
      }
    }
  }

  mFitter->mSkeleton->setPositions(originalPosition);

  for (int j = 0; j < mActiveMarkers.size(); j++)
  {
    if (numRadiiObservations(j) > 0)
    {
      mPerpendicularRadii(j) /= numRadiiObservations(j);
      mParallelRadii(j) /= numRadiiObservations(j);
    }
  }
}

//==============================================================================
int CylinderFitJointAxisProblem::getProblemDim()
{
  return mPerpendicularRadii.size() + mParallelRadii.size() + mAxisLines.size();
}

//==============================================================================
Eigen::VectorXs CylinderFitJointAxisProblem::flatten()
{
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(
      mPerpendicularRadii.size() + mParallelRadii.size() + mAxisLines.size());
  flat.segment(0, mPerpendicularRadii.size()) = mPerpendicularRadii;
  flat.segment(mPerpendicularRadii.size(), mParallelRadii.size())
      = mParallelRadii;
  flat.segment(
      mPerpendicularRadii.size() + mParallelRadii.size(), mAxisLines.size())
      = mAxisLines;
  return flat;
}

//==============================================================================
void CylinderFitJointAxisProblem::unflatten(Eigen::VectorXs x)
{
  mPerpendicularRadii = x.segment(0, mPerpendicularRadii.size());
  mParallelRadii = x.segment(mPerpendicularRadii.size(), mParallelRadii.size());
  mAxisLines = x.segment(
      mPerpendicularRadii.size() + mParallelRadii.size(), mAxisLines.size());
  // Ensure all the axis directions are normalized
  for (int i = 0; i < mNumTimesteps; i++)
  {
    mAxisLines.segment<3>(i * 6 + 3).normalize();
  }
}

//==============================================================================
s_t CylinderFitJointAxisProblem::getLoss()
{
  s_t loss = 0.0;

  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
    {
      loss += mKeepCenterLoss
              * (mAxisLines.segment<3>(i * 6) - mJointCenters.col(i))
                    .squaredNorm();
      if (!mNewClip[i])
      {
        loss += mSmoothingCenterLoss
                * (mAxisLines.segment<3>(i * 6)
                   - mAxisLines.segment<3>((i - 1) * 6))
                      .squaredNorm();
        loss += mSmoothingAxisLoss
                * (mAxisLines.segment<3>(i * 6 + 3)
                   - mAxisLines.segment<3>((i - 1) * 6 + 3))
                      .squaredNorm();
      }
    }
    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      if (mMarkerObserved(j, i))
      {
        Eigen::Vector3s center = mAxisLines.segment<3>(i * 6);
        Eigen::Vector3s axis = mAxisLines.segment<3>(i * 6 + 3);
        Eigen::Vector3s jointToCenter
            = (center - mMarkerPositions.block<3, 1>(j * 3, i));
        s_t diff = mPerpendicularRadii(j) * mPerpendicularRadii(j)
                   - (jointToCenter - (jointToCenter.dot(axis) * axis))
                         .squaredNorm();
        (void)diff;
        loss += diff * diff;
        s_t parallelDiff = mParallelRadii(j) * mParallelRadii(j)
                           - (jointToCenter.dot(axis) * axis).squaredNorm();
        (void)parallelDiff;
        loss += parallelDiff * parallelDiff;

        /*
        s_t dist = (jointToCenter.dot(axis) * axis).squaredNorm();
        s_t dist2 = (axis * axis.transpose() * jointToCenter).squaredNorm();
        s_t dist3 = (jointToCenter.transpose() * axis * axis.transpose())
                    * (axis * axis.transpose() * jointToCenter);
        s_t dist4 = jointToCenter.transpose()
                    * ((axis * axis.transpose()) * (axis * axis.transpose()))
                    * jointToCenter;
        s_t dist5 = axis.dot(axis) * jointToCenter.dot(axis)
                    * jointToCenter.dot(axis);
        if (abs(dist - dist2) > 1e-14)
        {
          std::cout << "Dist1 != dist2: " << dist << ", " << dist2 << std::endl;
        }
        if (abs(dist - dist3) > 1e-14)
        {
          std::cout << "Dist != dist3: " << dist << ", " << dist3 << std::endl;
        }
        if (abs(dist - dist4) > 1e-14)
        {
          std::cout << "Dist != dist4: " << dist << ", " << dist4 << std::endl;
        }
        if (abs(dist - dist5) > 1e-14)
        {
          std::cout << "Dist != dist5: " << dist << ", " << dist5 << std::endl;
        }
        // loss += dist * dist;

        // loss += dist4 * dist4;
        // loss += -2 * jointToCenter.dot(axis) * jointToCenter.dot(axis);
        */
      }
    }
  }

  return loss;
}

//==============================================================================
Eigen::VectorXs CylinderFitJointAxisProblem::getGradient()
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(
      mPerpendicularRadii.size() + mParallelRadii.size() + mAxisLines.size());

  int offset = mPerpendicularRadii.size() + mParallelRadii.size();
  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
    {
      grad.segment<3>(offset + i * 6)
          += 2 * mKeepCenterLoss
             * (mAxisLines.segment<3>(i * 6) - mJointCenters.col(i));

      if (!mNewClip[i])
      {
        grad.segment<3>(offset + i * 6)
            += 2 * mSmoothingCenterLoss
               * (mAxisLines.segment<3>(i * 6)
                  - mAxisLines.segment<3>((i - 1) * 6));
        grad.segment<3>(offset + (i - 1) * 6)
            -= 2 * mSmoothingCenterLoss
               * (mAxisLines.segment<3>(i * 6)
                  - mAxisLines.segment<3>((i - 1) * 6));
        grad.segment<3>(offset + i * 6 + 3)
            += 2 * mSmoothingAxisLoss
               * (mAxisLines.segment<3>(i * 6 + 3)
                  - mAxisLines.segment<3>((i - 1) * 6 + 3));
        grad.segment<3>(offset + (i - 1) * 6 + 3)
            -= 2 * mSmoothingAxisLoss
               * (mAxisLines.segment<3>(i * 6 + 3)
                  - mAxisLines.segment<3>((i - 1) * 6 + 3));
      }
    }

    const Eigen::Vector3s center = mAxisLines.segment<3>(i * 6);
    const Eigen::Vector3s axis = mAxisLines.segment<3>(i * 6 + 3);
    const s_t axisDotAxis = axis.dot(axis);
    for (int j = 0; j < mActiveMarkers.size(); j++)
    {
      if (mMarkerObserved(j, i))
      {
        const Eigen::Vector3s jointToCenter
            = (center - mMarkerPositions.block<3, 1>(j * 3, i));
        const s_t jointToCenterDotAxis = jointToCenter.dot(axis);
        const Eigen::Vector3s jointToCenterAlongAxis
            = jointToCenterDotAxis * axis;
        const s_t diff
            = mPerpendicularRadii(j) * mPerpendicularRadii(j)
              - (jointToCenter - (jointToCenterAlongAxis)).squaredNorm();
        const s_t parallelDiff = mParallelRadii(j) * mParallelRadii(j)
                                 - (jointToCenterAlongAxis).squaredNorm();
        // Gradient wrt perpendicular radii
        grad(j) += (2 * diff) * (2 * mPerpendicularRadii(j));
        // Gradient wrt parallel radii
        grad(mParallelRadii.size() + j)
            += (2 * parallelDiff) * (2 * mParallelRadii(j));

        // Gradient wrt the axis center of perpendicular term
        grad.segment<3>(offset + i * 6)
            += (2 * diff) * -2
               * (jointToCenter - axisDotAxis * jointToCenterAlongAxis);
        // Gradient wrt the axis of perpendicular term
        grad.segment<3>(offset + i * 6 + 3)
            += 2 * diff * -1
               * ((-4 + 2 * axisDotAxis) * jointToCenterDotAxis * jointToCenter
                  + 2 * jointToCenterDotAxis * jointToCenterDotAxis * axis);
        // Gradient wrt the axis center of parallel term
        grad.segment<3>(offset + i * 6)
            += -2 * parallelDiff * 2 * (axisDotAxis * jointToCenterAlongAxis);
        // Gradient wrt the axis of parallel term
        grad.segment<3>(offset + i * 6 + 3)
            += -2 * parallelDiff
               * (2 * jointToCenterAlongAxis * jointToCenterDotAxis
                  + axisDotAxis * 2 * jointToCenterDotAxis * jointToCenter);
      }
    }

    // Keep only the portion of the gradient wrt the normal vector that's
    // perpendicular to the current normal
    // Operate on the last timestep gradient, since that's now complete
    if (i > 0)
    {
      Eigen::Vector3s axisDir
          = mAxisLines.segment<3>((i - 1) * 6 + 3).normalized();
      s_t dot = grad.segment<3>(offset + (i - 1) * 6 + 3).dot(axisDir);
      grad.segment<3>(offset + (i - 1) * 6 + 3) -= axisDir * dot;
    }
  }

  // Keep only the portion of the gradient wrt the normal vector that's
  // perpendicular to the current normal
  // Patch the last timestep gradient, since that never gets patched in the
  // normal loop
  Eigen::Vector3s axisDir
      = mAxisLines.segment<3>((mNumTimesteps - 1) * 6 + 3).normalized();
  s_t dot = grad.segment<3>(offset + (mNumTimesteps - 1) * 6 + 3).dot(axisDir);
  grad.segment<3>(offset + (mNumTimesteps - 1) * 6 + 3) -= axisDir * dot;

  return grad;
}

//==============================================================================
Eigen::VectorXs CylinderFitJointAxisProblem::finiteDifferenceGradient()
{
  Eigen::VectorXs x = flatten();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());

  math::finiteDifference(
      [this, &x](
          /* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs tweaked = x;
        tweaked(dof) += eps;
        unflatten(tweaked);
        perturbed = getLoss();
        return true;
      },
      grad,
      1e-2,
      true);

  unflatten(x);

  return grad;
}

//==============================================================================
s_t CylinderFitJointAxisProblem::saveSolutionBackToInitialization()
{
  for (int i = 0; i < mNumTimesteps; i++)
  {
    mOut.col(i) = mAxisLines.segment<6>(i * 6);
  }
  return getLoss();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// The BilevelFitProblem, which maps the problem onto a format that IPOpt can
// work with.
///////////////////////////////////////////////////////////////////////////////////////////////////////

//==============================================================================
/// This creates a problem object. We take as arguments:
/// @param skeleton the skeleton we're going to use to scale + fit the data
/// @param markerSet the marker set we're using, with default offsets from the
/// skeleton
/// @param markerObservations a list of timesteps, where each timestep
/// observes some subset of the markers at some points in 3D space.
BilevelFitProblem::BilevelFitProblem(
    MarkerFitter* fitter,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    MarkerInitialization& initialization,
    int numSamples,
    bool applyInnerProblemGradientConstraints,
    std::shared_ptr<BilevelFitResult>& outResult)
  : mFitter(fitter),
    mOutResult(outResult),
    mInitialization(initialization),
    mApplyInnerProblemGradientConstraints(applyInnerProblemGradientConstraints),
    mBestObjectiveValue(std::numeric_limits<s_t>::infinity())
{
  // 1. Select the random indices we'll be using for this problem
  mSampleIndices
      = math::evenlySpacedTimesteps(markerObservations.size(), numSamples);

  // if (numSamples >= markerObservations.size())
  // {
  //   numSamples = markerObservations.size();
  //   for (int i = 0; i < numSamples; i++)
  //   {
  //     mSampleIndices.push_back(i);
  //   }
  // }
  // else
  // {
  //   double stride = (double)markerObservations.size() / numSamples;

  //   for (int i = 0; i < numSamples; i++)
  //   {
  //     int index = (int)round(stride * ((double)i + 0.5));
  //     if (index > markerObservations.size() - 1)
  //       index = markerObservations.size() - 1;
  //     if (index < 0)
  //       index = 0;

  //     mSampleIndices.push_back(index);
  //   }
  // }

  // TODO: <remove>
  std::cout << "Picked " << numSamples << " evenly spaced in [0,"
            << markerObservations.size() << "]: " << std::endl
            << "[";
  for (int i : mSampleIndices)
  {
    std::cout << i << ",";
  }
  std::cout << "]" << std::endl;
  // TODO: </remove>

  mJointCenters = Eigen::MatrixXs::Zero(
      initialization.jointCenters.rows(), mSampleIndices.size());
  mJointWeights = initialization.jointWeights;
  mJointAxis = Eigen::MatrixXs::Zero(
      initialization.jointAxis.rows(), mSampleIndices.size());
  mAxisWeights = initialization.axisWeights;

  // 2. Select the observations from the randomly chosen indices we'll be using
  // for this problem
  for (int i : mSampleIndices)
  {
    auto observation = markerObservations[i];
    if (initialization.jointCenters.rows() > 0)
    {
      mJointCenters.col(mMarkerMapObservations.size())
          = initialization.jointCenters.col(i);
    }
    if (initialization.jointAxis.rows() > 0)
    {
      mJointAxis.col(mMarkerMapObservations.size())
          = initialization.jointAxis.col(i);
    }
    mMarkerMapObservations.push_back(observation);
    std::vector<std::pair<int, Eigen::Vector3s>> translated;
    for (auto pair : observation)
    {
      translated.emplace_back(
          mFitter->mMarkerIndices[pair.first], Eigen::Vector3s(pair.second));
    }
    mMarkerObservations.push_back(translated);
  }

  mObservationWeights = Eigen::VectorXs::Ones(mSampleIndices.size());

  // 3. Create threads
  mNumThreads = 12;
  std::vector<int> samplesPerThread;
  for (int i = 0; i < mNumThreads; i++)
  {
    mPerThreadSkeletons.push_back(mFitter->mSkeleton->clone());
    samplesPerThread.push_back(0);
  }

  // 4. Allocate sample counts to threads
  int samplesToAllocate = mSampleIndices.size();
  int cursor = 0;
  while (samplesToAllocate > 0)
  {
    samplesPerThread[cursor]++;
    samplesToAllocate--;
    cursor++;
    if (cursor >= samplesPerThread.size())
      cursor = 0;
  }

  // 5. Allocate the indices to thread buckets
  cursor = 0;
  for (int i = 0; i < mNumThreads; i++)
  {
    std::vector<int> cursorIndices;
    for (int j = 0; j < samplesPerThread[i]; j++)
    {
      cursorIndices.push_back(cursor);
      cursor++;
    }
    mPerThreadCursor.push_back(cursorIndices);
  }
  assert(cursor == mSampleIndices.size());
}

//==============================================================================
BilevelFitProblem::~BilevelFitProblem()
{
}

//==============================================================================
int BilevelFitProblem::getProblemSize()
{
  int scaleGroupDims = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDims = mFitter->mMarkers.size() * 3;
  int poseDims = mFitter->mSkeleton->getNumDofs() * mSampleIndices.size();
  int staticPoseRootDims = 6;
  return scaleGroupDims + markerOffsetDims + poseDims + staticPoseRootDims;
}

//==============================================================================
/// This gets a decent initial guess for the problem. We can guess scaling and
/// joint positions from the first marker observation, and then use that
/// scaling to get joint positions for all the other entries. This initially
/// satisfies the constraint that we remain at optimal positional IK
/// throughout optimization.
Eigen::VectorXs BilevelFitProblem::getInitialization()
{
  int groupScaleDim = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDim = mFitter->mMarkers.size() * 3;
  int dofs = mFitter->mSkeleton->getNumDofs();

  Eigen::VectorXs init = Eigen::VectorXs::Zero(
      groupScaleDim + markerOffsetDim + (mSampleIndices.size() * dofs) + 6);

  // Copy group scales
  init.segment(0, groupScaleDim) = mInitialization.groupScales;

  // Copy marker offsets
  init.segment(groupScaleDim, markerOffsetDim).setZero();
  for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
  {
    if (mInitialization.markerOffsets.count(mFitter->mMarkerNames[i]))
    {
      assert(mInitialization.markerOffsets.count(mFitter->mMarkerNames[i]));
      init.segment<3>(groupScaleDim + i * 3)
          = mInitialization.markerOffsets.at(mFitter->mMarkerNames[i]);
    }
  }

  // Copy positions
  for (int i = 0; i < mSampleIndices.size(); i++)
  {
    init.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs)
        = mInitialization.poses.col(mSampleIndices[i]);
  }

  // Copy static pose root
  init.segment<6>(
      groupScaleDim + markerOffsetDim + (mSampleIndices.size() * dofs))
      = mInitialization.staticPoseRoot;

  // std::cout << "Init group scales: " << std::endl
  //           << init.segment(0, groupScaleDim) << std::endl;
  // std::cout << "Init marker offsets: " << std::endl
  //           << init.segment(groupScaleDim, markerOffsetDim) << std::endl;
  // std::cout << "Init first pose: " << std::endl
  //           << init.segment(groupScaleDim + markerOffsetDim, dofs) <<
  //           std::endl;

  return init;
}

//==============================================================================
/// This evaluates our loss function given a concatenated vector of all the
/// problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
s_t BilevelFitProblem::getLoss(Eigen::VectorXs x)
{
  mLastX = x;

  // int groupScaleDim = mFitter->mSkeleton->getGroupScaleDim();
  // int markerOffsetDim = mFitter->mMarkers.size() * 3;
  // int dofs = mFitter->mSkeleton->getNumDofs();
  // std::cout << "Getting loss at group scales: " << std::endl
  //           << x.segment(0, groupScaleDim) << std::endl;
  // std::cout << "Getting loss at marker offsets: " << std::endl
  //           << x.segment(groupScaleDim, markerOffsetDim) << std::endl;
  // std::cout << "Getting loss at first pose: " << std::endl
  //           << x.segment(groupScaleDim + markerOffsetDim, dofs) << std::endl;

  MarkerFitterState state(
      x,
      mMarkerMapObservations,
      mInitialization.joints,
      mJointCenters,
      mJointWeights,
      mJointAxis,
      mAxisWeights,
      mFitter);
  return mFitter->mLossAndGrad(&state);
}

//==============================================================================
/// This evaluates our gradient of loss given a concatenated vector of all
/// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
Eigen::VectorXs BilevelFitProblem::getGradient(Eigen::VectorXs x)
{
  mLastX = x;
  MarkerFitterState state(
      x,
      mMarkerMapObservations,
      mInitialization.joints,
      mJointCenters,
      mJointWeights,
      mJointAxis,
      mAxisWeights,
      mFitter);
  mFitter->mLossAndGrad(&state);
  return state.flattenGradient();
}

//==============================================================================
/// This evaluates our gradient of loss given a concatenated vector of all
/// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
Eigen::VectorXs BilevelFitProblem::finiteDifferenceGradient(Eigen::VectorXs x)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());
  const s_t EPS = 1e-7;
  for (int i = 0; i < x.size(); i++)
  {
    Eigen::VectorXs perturbed = x;
    perturbed(i) += EPS;
    s_t plus = getLoss(perturbed);
    perturbed = x;
    perturbed(i) -= EPS;
    s_t minus = getLoss(perturbed);

    grad(i) = (plus - minus) / (2 * EPS);
  }
  return grad;
}

//==============================================================================
/// This evaluates our constraint vector given a concatenated vector of all
/// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
Eigen::VectorXs BilevelFitProblem::getConstraints(Eigen::VectorXs x)
{
  int scaleGroupDims = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDims = mFitter->mMarkers.size() * 3;
  Eigen::VectorXs groupScales = x.segment(0, scaleGroupDims);
  Eigen::VectorXs markerOffsets = x.segment(scaleGroupDims, markerOffsetDims);
  Eigen::VectorXs firstPose = x.segment(
      scaleGroupDims + markerOffsetDims, mFitter->mSkeleton->getNumDofs());

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  Eigen::VectorXs ikGrad
      = Eigen::VectorXs::Zero(mFitter->mSkeleton->getNumDofs());

  if (mApplyInnerProblemGradientConstraints)
  {
    bool multiThreaded = true;
    if (multiThreaded)
    {
      std::vector<std::future<Eigen::VectorXs>> futures;
      for (int k = 0; k < mNumThreads; k++)
      {
        std::vector<int> threadCursors = mPerThreadCursor[k];
        std::shared_ptr<dynamics::Skeleton> threadSkeleton
            = mPerThreadSkeletons[k];
        threadSkeleton->setGroupScales(mFitter->mSkeleton->getGroupScales());

        std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
            threadMarkers;
        for (auto pair : markers)
        {
          threadMarkers.emplace_back(
              threadSkeleton->getBodyNode(pair.first->getName()), pair.second);
        }

        futures.push_back(
            std::async([&, threadCursors, threadSkeleton, threadMarkers]() {
              Eigen::VectorXs ikGradLocal
                  = Eigen::VectorXs::Zero(threadSkeleton->getNumDofs());

              for (int i : threadCursors)
              {
                int offset = scaleGroupDims + markerOffsetDims
                             + (i * threadSkeleton->getNumDofs());
                Eigen::VectorXs pose
                    = x.segment(offset, threadSkeleton->getNumDofs());
                threadSkeleton->setPositions(pose);

                // Get loss wrt joint positions
                ikGradLocal += mFitter->getMarkerLossGradientWrtJoints(
                                   threadSkeleton,
                                   threadMarkers,
                                   mFitter->getIKLossGradWrtMarkerError(
                                       mFitter->getMarkerError(
                                           threadSkeleton,
                                           threadMarkers,
                                           mMarkerObservations[i])))
                               * mObservationWeights(i);
              }

              return ikGradLocal;
            }));
      }
      for (int k = 0; k < mNumThreads; k++)
      {
        ikGrad += futures[k].get();
      }
    }
    else
    {
      for (int i = 0; i < mMarkerObservations.size(); i++)
      {
        int offset = scaleGroupDims + markerOffsetDims
                     + (i * mFitter->mSkeleton->getNumDofs());
        Eigen::VectorXs pose
            = x.segment(offset, mFitter->mSkeleton->getNumDofs());
        mFitter->mSkeleton->setPositions(pose);

        // Get loss wrt joint positions
        ikGrad
            += mFitter->getMarkerLossGradientWrtJoints(
                   mFitter->mSkeleton,
                   markers,
                   mFitter->getIKLossGradWrtMarkerError(mFitter->getMarkerError(
                       mFitter->mSkeleton, markers, mMarkerObservations[i])))
               * mObservationWeights(i);
      }
    }
  }

  if (mFitter->mZeroConstraints.size() > 0)
  {
    MarkerFitterState state(
        x,
        mMarkerMapObservations,
        mInitialization.joints,
        mJointCenters,
        mJointWeights,
        mJointAxis,
        mAxisWeights,
        mFitter);

    Eigen::VectorXs concatenatedConstraints = Eigen::VectorXs::Zero(
        ikGrad.size() + mFitter->mZeroConstraints.size());
    concatenatedConstraints.segment(0, ikGrad.size()) = ikGrad;
    int cursor = ikGrad.size();
    for (auto pair : mFitter->mZeroConstraints)
    {
      concatenatedConstraints(cursor) = pair.second(&state);
      cursor++;
    }
    return concatenatedConstraints;
  }
  else
  {
    return ikGrad;
  }
}

//==============================================================================
/// This evaluates the Jacobian of our constraint vector wrt x given a
/// concatenated vector of all the problem state: [groupSizes, markerOffsets,
/// q_0, ..., q_N]
Eigen::MatrixXs BilevelFitProblem::getConstraintsJacobian(Eigen::VectorXs x)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(mFitter->mSkeleton->getNumDofs(), x.size());

  int scaleGroupDims = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDims = mFitter->mMarkers.size() * 3;
  Eigen::VectorXs groupScales = x.segment(0, scaleGroupDims);
  Eigen::VectorXs markerOffsets = x.segment(scaleGroupDims, markerOffsetDims);
  Eigen::VectorXs firstPose = x.segment(
      scaleGroupDims + markerOffsetDims, mFitter->mSkeleton->getNumDofs());

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  if (mApplyInnerProblemGradientConstraints)
  {
    bool multiThreaded = true;
    if (multiThreaded)
    {
      std::vector<std::future<Eigen::MatrixXs>> futures;
      for (int k = 0; k < mNumThreads; k++)
      {
        std::vector<int> threadCursors = mPerThreadCursor[k];
        std::shared_ptr<dynamics::Skeleton> threadSkeleton
            = mPerThreadSkeletons[k];
        threadSkeleton->setGroupScales(mFitter->mSkeleton->getGroupScales());

        std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
            threadMarkers;
        for (auto pair : markers)
        {
          threadMarkers.emplace_back(
              threadSkeleton->getBodyNode(pair.first->getName()), pair.second);
        }

        futures.push_back(std::async([&,
                                      threadCursors,
                                      threadSkeleton,
                                      threadMarkers]() {
          Eigen::MatrixXs markersAndScalesLocalJac = Eigen::MatrixXs::Zero(
              threadSkeleton->getNumDofs(), scaleGroupDims + markerOffsetDims);

          for (int i : threadCursors)
          {
            int offset = scaleGroupDims + markerOffsetDims
                         + (i * threadSkeleton->getNumDofs());
            Eigen::VectorXs pose
                = x.segment(offset, threadSkeleton->getNumDofs());
            threadSkeleton->setPositions(pose);

            Eigen::VectorXs markerError = mFitter->getMarkerError(
                threadSkeleton, threadMarkers, mMarkerObservations[i]);
            std::vector<int> sparsityMap = mFitter->getSparsityMap(
                threadMarkers, mMarkerObservations[i]);

            // Get loss wrt joint positions
            jac.block(
                0,
                offset,
                threadSkeleton->getNumDofs(),
                threadSkeleton->getNumDofs())
                = mFitter->getIKLossGradientWrtJointsJacobianWrtJoints(
                      threadSkeleton, threadMarkers, markerError, sparsityMap)
                  * mObservationWeights(i);

            // Acculumulate loss wrt the global scale groups
            markersAndScalesLocalJac.block(
                0, 0, threadSkeleton->getNumDofs(), scaleGroupDims)
                += mFitter->getIKLossGradientWrtJointsJacobianWrtGroupScales(
                       threadSkeleton, threadMarkers, markerError, sparsityMap)
                   * mObservationWeights(i);
            // Acculumulate loss wrt the global marker offsets
            markersAndScalesLocalJac.block(
                0,
                scaleGroupDims,
                threadSkeleton->getNumDofs(),
                markerOffsetDims)
                += mFitter->getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
                       threadSkeleton, threadMarkers, markerError, sparsityMap)
                   * mObservationWeights(i);
          }

          return markersAndScalesLocalJac;
        }));
      }
      for (int k = 0; k < mNumThreads; k++)
      {
        jac.block(
            0,
            0,
            mFitter->mSkeleton->getNumDofs(),
            scaleGroupDims + markerOffsetDims)
            += futures[k].get();
      }
    }
    else
    {
      for (int i = 0; i < mMarkerObservations.size(); i++)
      {
        int offset = scaleGroupDims + markerOffsetDims
                     + (i * mFitter->mSkeleton->getNumDofs());
        Eigen::VectorXs pose
            = x.segment(offset, mFitter->mSkeleton->getNumDofs());
        mFitter->mSkeleton->setPositions(pose);

        Eigen::VectorXs markerError = mFitter->getMarkerError(
            mFitter->mSkeleton, markers, mMarkerObservations[i]);
        std::vector<int> sparsityMap
            = mFitter->getSparsityMap(markers, mMarkerObservations[i]);

        // Get loss wrt joint positions
        jac.block(
            0,
            offset,
            mFitter->mSkeleton->getNumDofs(),
            mFitter->mSkeleton->getNumDofs())
            = mFitter->getIKLossGradientWrtJointsJacobianWrtJoints(
                  mFitter->mSkeleton, markers, markerError, sparsityMap)
              * mObservationWeights(i);

        // Acculumulate loss wrt the global scale groups
        jac.block(0, 0, mFitter->mSkeleton->getNumDofs(), scaleGroupDims)
            += mFitter->getIKLossGradientWrtJointsJacobianWrtGroupScales(
                   mFitter->mSkeleton, markers, markerError, sparsityMap)
               * mObservationWeights(i);
        // Acculumulate loss wrt the global marker offsets
        jac.block(
            0,
            scaleGroupDims,
            mFitter->mSkeleton->getNumDofs(),
            markerOffsetDims)
            += mFitter->getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
                   mFitter->mSkeleton, markers, markerError, sparsityMap)
               * mObservationWeights(i);
      }
    }
  }

  if (mFitter->mZeroConstraints.size() > 0)
  {
    MarkerFitterState state(
        x,
        mMarkerMapObservations,
        mInitialization.joints,
        mJointCenters,
        mJointWeights,
        mJointAxis,
        mAxisWeights,
        mFitter);

    Eigen::MatrixXs concatenatedJac = Eigen::MatrixXs::Zero(
        jac.rows() + mFitter->mZeroConstraints.size(), jac.cols());
    concatenatedJac.block(0, 0, jac.rows(), jac.cols()) = jac;
    int cursor = jac.rows();
    for (auto pair : mFitter->mZeroConstraints)
    {
      pair.second(&state);
      concatenatedJac.row(cursor) = state.flattenGradient();
      cursor++;
    }
    return concatenatedJac;
  }
  else
  {
    return jac;
  }
}

//==============================================================================
/// This evaluates the Jacobian of our constraint vector wrt x given a
/// concatenated vector of all the problem state: [groupSizes, markerOffsets,
/// q_0, ..., q_N]
Eigen::MatrixXs BilevelFitProblem::finiteDifferenceConstraintsJacobian(
    Eigen::VectorXs x)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(
      mFitter->mSkeleton->getNumDofs() + mFitter->mZeroConstraints.size(),
      x.size());
  const s_t EPS = 1e-7;
  for (int i = 0; i < x.size(); i++)
  {
    Eigen::VectorXs perturbed = x;
    perturbed(i) += EPS;
    Eigen::VectorXs plus = getConstraints(perturbed);

    perturbed = x;
    perturbed(i) -= EPS;
    Eigen::VectorXs minus = getConstraints(perturbed);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }
  return jac;
}

//==============================================================================
/// This returns the indices that this problem is using to specify the problem
const std::vector<int>& BilevelFitProblem::getSampleIndices()
{
  return mSampleIndices;
}

//==============================================================================
/// This returns the marker map observations that this problem is using to
/// specify the problem
const std::vector<std::map<std::string, Eigen::Vector3s>>&
BilevelFitProblem::getMarkerMapObservations()
{
  return mMarkerMapObservations;
}

//==============================================================================
/// This returns the marker observations that this problem is using to specify
/// the problem
const std::vector<std::vector<std::pair<int, Eigen::Vector3s>>>&
BilevelFitProblem::getMarkerObservations()
{
  return mMarkerObservations;
}

//==============================================================================
/// This returns the subset of joint centers, for the selected timestep samples
const Eigen::MatrixXs& BilevelFitProblem::getJointCenters()
{
  return mJointCenters;
}

//==============================================================================
//------------------------- Ipopt::TNLP --------------------------------------
/// \brief Method to return some info about the nlp
bool BilevelFitProblem::get_nlp_info(
    Ipopt::Index& n,         // number of variables in "x"
    Ipopt::Index& m,         // number of variables in constraint
    Ipopt::Index& nnz_jac_g, // number of non-zero values in the Jacobian of
                             // the constraint
    Ipopt::Index& nnz_h_lag, // number of non-zero values in the Hessian of
                             // the Lagrangian (we'll use LBFGS instead)
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  // Set the number of decision variables
  n = mFitter->mSkeleton->getGroupScaleDim() + mFitter->mMarkers.size() * 3
      + (mFitter->mSkeleton->getNumDofs() * mMarkerObservations.size()) + 6;

  // Set the total number of constraints
  m = mFitter->mSkeleton->getNumDofs() + mFitter->mZeroConstraints.size();

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = m * n;

  // Set the number of entries in the Hessian
  nnz_h_lag = n * n;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

/// \brief Method to return the bounds for my problem
bool BilevelFitProblem::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  // We default bounds to a large finite value, rather than infinity, to prevent
  // NaNs from creeping into computations where they're not wanted.
  s_t LARGE_FINITE_VALUE = 10000000;
  // Lower and upper bounds on X
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  upperBounds.setConstant(LARGE_FINITE_VALUE);
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  lowerBounds.setConstant(-LARGE_FINITE_VALUE);

  int scaleGroupDim = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDim = mFitter->mMarkers.size() * 3;
  int dofs = mFitter->mSkeleton->getNumDofs();

  assert(
      n
      == scaleGroupDim + markerOffsetDim + (mMarkerObservations.size() * dofs)
             + 6);

  upperBounds.segment(0, scaleGroupDim)
      = mFitter->mSkeleton->getGroupScalesUpperBound().cast<double>();
  lowerBounds.segment(0, scaleGroupDim)
      = mFitter->mSkeleton->getGroupScalesLowerBound().cast<double>();
  upperBounds.segment(scaleGroupDim, markerOffsetDim)
      .setConstant((double)mFitter->mMaxMarkerOffset);
  lowerBounds.segment(scaleGroupDim, markerOffsetDim)
      .setConstant((double)-mFitter->mMaxMarkerOffset);
  if (!mFitter->mIgnoreJointLimits)
  {
    for (int i = 0; i < mMarkerObservations.size(); i++)
    {
      upperBounds.segment(scaleGroupDim + markerOffsetDim + (i * dofs), dofs)
          = mFitter->mSkeleton->getPositionUpperLimits().cast<double>();
      lowerBounds.segment(scaleGroupDim + markerOffsetDim + (i * dofs), dofs)
          = mFitter->mSkeleton->getPositionLowerLimits().cast<double>();
    }
  }
  // Add bounds for the static pose value
  upperBounds.segment(
      scaleGroupDim + markerOffsetDim + (mMarkerMapObservations.size() * dofs),
      6)
      = mFitter->mSkeleton->getPositionUpperLimits().head<6>().cast<double>();
  lowerBounds.segment(
      scaleGroupDim + markerOffsetDim + (mMarkerMapObservations.size() * dofs),
      6)
      = mFitter->mSkeleton->getPositionLowerLimits().head<6>().cast<double>();

  // std::cout << "Group scales upper bound: " << std::endl
  //           << upperBounds.segment(0, scaleGroupDim) << std::endl;
  // std::cout << "Marker offset upper bound: " << std::endl
  //           << upperBounds.segment(scaleGroupDim, markerOffsetDim) <<
  //           std::endl;
  // std::cout << "Positions upper bound: " << std::endl
  //           << upperBounds.segment(scaleGroupDim + markerOffsetDim, dofs)
  //           << std::endl;

  // std::cout << "Group scales lower bound: " << std::endl
  //           << lowerBounds.segment(0, scaleGroupDim) << std::endl;
  // std::cout << "Marker offset lower bound: " << std::endl
  //           << lowerBounds.segment(scaleGroupDim, markerOffsetDim) <<
  //           std::endl;
  // std::cout << "Positions lower bound: " << std::endl
  //           << lowerBounds.segment(scaleGroupDim + markerOffsetDim, dofs)
  //           << std::endl;

  // Our constraint function has to be 0
  Eigen::Map<Eigen::VectorXd> constraintUpperBounds(g_u, m);
  constraintUpperBounds.setZero();
  Eigen::Map<Eigen::VectorXd> constraintLowerBounds(g_l, m);
  constraintLowerBounds.setZero();

  return true;
}

/// \brief Method to return the starting point for the algorithm
bool BilevelFitProblem::get_starting_point(
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number* _x,
    bool init_z,
    Ipopt::Number* z_L,
    Ipopt::Number* z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number* lambda)
{
  // Here, we assume we only have starting values for x
  (void)init_x;
  assert(init_x == true);
  (void)init_z;
  assert(init_z == false);
  (void)init_lambda;
  assert(init_lambda == false);
  // We don't set the lagrange multipliers
  (void)z_L;
  (void)z_U;
  (void)m;
  (void)lambda;

  if (init_x)
  {
    Eigen::Map<Eigen::VectorXd> x(_x, n);
    x = getInitialization().cast<double>();
  }

  return true;
}

/// \brief Method to return the objective value
bool BilevelFitProblem::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);

  _obj_value = (double)getLoss(x.cast<s_t>());

  return true;
}

/// \brief Method to return the gradient of the objective
bool BilevelFitProblem::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);

  grad = getGradient(x.cast<s_t>()).cast<double>();

  return true;
}

/// \brief Method to return the constraint residuals
bool BilevelFitProblem::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  Eigen::Map<Eigen::VectorXd> g(_g, _m);

  g = getConstraints(x.cast<s_t>()).cast<double>();

  return true;
}

/// \brief Method to return:
///        1) The structure of the jacobian (if "values" is nullptr)
///        2) The values of the jacobian (if "values" is not nullptr)
bool BilevelFitProblem::eval_jac_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Index _nnzj,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_new_x;
  (void)_m;
  // If the iRow and jCol arguments are not nullptr, then IPOPT wants you to
  // fill in the sparsity structure of the Jacobian (the row and column
  // indices only). At this time, the x argument and the values argument will
  // be nullptr.

  if (nullptr == _x)
  {
    Eigen::Map<Eigen::VectorXi> rows(_iRow, _nnzj);
    Eigen::Map<Eigen::VectorXi> cols(_jCol, _nnzj);
    int cursor = 0;
    for (int col = 0; col < _n; col++)
    {
      for (int row = 0; row < _m; row++)
      {
        rows(cursor) = row;
        cols(cursor) = col;
        cursor++;
      }
    }
    assert(cursor == _nnzj);
  }
  else
  {
    // Return the concatenated gradient of everything
    Eigen::Map<const Eigen::VectorXd> x(_x, _n);
    Eigen::Map<Eigen::VectorXd> vals(_values, _nnzj);

    Eigen::MatrixXs jac = getConstraintsJacobian(x.cast<s_t>());

    int cursor = 0;
    for (int col = 0; col < jac.cols(); col++)
    {
      for (int row = 0; row < jac.rows(); row++)
      {
        vals(cursor) = (double)jac(row, col);
        cursor++;
      }
    }
    assert(cursor == _nnzj);
  }

  return true;
}

/// \brief Method to return:
///        1) The structure of the hessian of the lagrangian (if "values" is
///           nullptr)
///        2) The values of the hessian of the lagrangian (if "values" is not
///           nullptr)
bool BilevelFitProblem::eval_h(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number _obj_factor,
    Ipopt::Index _m,
    const Ipopt::Number* _lambda,
    bool _new_lambda,
    Ipopt::Index _nele_hess,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_n;
  (void)_x;
  (void)_new_x;
  (void)_obj_factor;
  (void)_m;
  (void)_lambda;
  (void)_new_lambda;
  (void)_nele_hess;
  (void)_iRow;
  (void)_jCol;
  (void)_values;
  return false;
}

/// \brief This method is called when the algorithm is complete so the TNLP
///        can store/write the solution
void BilevelFitProblem::finalize_solution(
    Ipopt::SolverReturn _status,
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    const Ipopt::Number* _z_L,
    const Ipopt::Number* _z_U,
    Ipopt::Index _m,
    const Ipopt::Number* _g,
    const Ipopt::Number* _lambda,
    Ipopt::Number _obj_value,
    const Ipopt::IpoptData* _ip_data,
    Ipopt::IpoptCalculatedQuantities* _ip_cq)
{
  (void)_status;
  (void)_n;
  (void)_x;
  (void)_z_L;
  (void)_z_U;
  (void)_m;
  (void)_g;
  (void)_lambda;
  (void)_obj_value;
  (void)_ip_data;
  (void)_ip_cq;
  // Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  std::cout << "Recovering state with best loss: iteration "
            << mBestObjectiveValueIteration << " with " << mBestObjectiveValue
            << std::endl;
  Eigen::VectorXs x = mBestObjectiveValueState;

  int groupScaleDim = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDim = mFitter->mMarkers.size() * 3;
  int dofs = mFitter->mSkeleton->getNumDofs();

  mOutResult->groupScales = x.segment(0, groupScaleDim);
  mOutResult->rawMarkerOffsets = x.segment(groupScaleDim, markerOffsetDim);
  for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
  {
    mOutResult->markerOffsets[mFitter->mMarkerNames[i]]
        = mOutResult->rawMarkerOffsets.segment<3>(i * 3);
  }
  std::cout << "Saving " << mMarkerObservations.size() << " results"
            << std::endl;
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    mOutResult->poses.push_back(
        x.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs));
  }
}

bool BilevelFitProblem::intermediate_callback(
    Ipopt::AlgorithmMode mode,
    Ipopt::Index iter,
    Ipopt::Number obj_value,
    Ipopt::Number inf_pr,
    Ipopt::Number inf_du,
    Ipopt::Number mu,
    Ipopt::Number d_norm,
    Ipopt::Number regularization_size,
    Ipopt::Number alpha_du,
    Ipopt::Number alpha_pr,
    Ipopt::Index ls_trials,
    const Ipopt::IpoptData* ip_data,
    Ipopt::IpoptCalculatedQuantities* ip_cq)
{
  (void)mode;
  (void)iter;
  (void)obj_value;
  (void)inf_pr;
  (void)inf_du;
  (void)mu;
  (void)d_norm;
  (void)regularization_size;
  (void)alpha_du;
  (void)alpha_pr;
  (void)ls_trials;
  (void)ip_data;
  (void)ip_cq;

  if (obj_value < mBestObjectiveValue && abs(inf_pr) < 1.0)
  {
    mBestObjectiveValueIteration = iter;
    mBestObjectiveValue = obj_value;
    mBestObjectiveValueState = mLastX;
  }

  return true;
}

IMUFineTuneProblem::IMUFineTuneProblem(
    MarkerFitter* fitter,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& accObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& gyroObservations,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    s_t dt,
    MarkerInitialization& initialization,
    int start,
    int end)
  : mFitter(fitter),
    mInitialization(initialization),
    mTimeStep(dt),
    mStart(start),
    mEnd(end),
    mWeightGyros(1.0),
    mWeightAccs(1.0),
    mWeightMarkers(100.0),
    mRegularizePoses(1.0),
    mMarkerObservations(markerObservations),
    mBestObjectiveValueIteration(-1),
    mBestObjectiveValue(std::numeric_limits<s_t>::infinity()),
    mUseMultiThreading(true)
{
  mOriginalPoses = initialization.poses.block(
      0, start, initialization.poses.rows(), end - start);
  mPoses = mOriginalPoses;
  mVels = Eigen::MatrixXs::Zero(mPoses.rows(), mPoses.cols());
  mAccs = Eigen::MatrixXs::Zero(mPoses.rows(), mPoses.cols());
  mLength = mEnd - mStart;

  // Use finite differencing to initially fill in columns in mVels and mAccs
  for (int t = 0; t < mPoses.cols(); t++)
  {
    if (t > 0)
    {
      mVels.col(t) = mFitter->mSkeleton->getPositionDifferences(
                         mPoses.col(t), mPoses.col(t - 1))
                     / mTimeStep;
    }
    if (t > 0 && t < mPoses.cols() - 1)
    {
      mAccs.col(t) = (mFitter->mSkeleton->getPositionDifferences(
                          mPoses.col(t + 1), mPoses.col(t))
                      - mFitter->mSkeleton->getPositionDifferences(
                          mPoses.col(t), mPoses.col(t - 1)))
                     / (mTimeStep * mTimeStep);
    }
  }
  mVels.col(0) = mVels.col(1);

  // Set up our list of observed accelerometers and gyroscopes, in order
  for (auto& pair : mFitter->mImuMap)
  {
    if (accObservations[0].count(pair.first))
    {
      mAccelerometerNames.push_back(pair.first);
      mAccelerometers.push_back(pair.second);
    }
    if (gyroObservations[0].count(pair.first))
    {
      mGyroNames.push_back(pair.first);
      mGyros.push_back(pair.second);
    }
  }

  // Copy the accelerometer observations into a matrix
  mAccelerometerObservations
      = Eigen::MatrixXs::Zero(mAccelerometerNames.size() * 3, mLength);
  for (int t = 0; t < mLength; t++)
  {
    for (int i = 0; i < mAccelerometerNames.size(); i++)
    {
      mAccelerometerObservations.block<3, 1>(i * 3, t)
          = accObservations[mStart + t].at(mAccelerometerNames[i]);
    }
  }

  // Copy the gyro observations into a matrix
  mGyroObservations = Eigen::MatrixXs::Zero(mGyroNames.size() * 3, mLength);
  for (int t = 0; t < mLength; t++)
  {
    for (int i = 0; i < mGyroNames.size(); i++)
    {
      mGyroObservations.block<3, 1>(i * 3, t)
          = gyroObservations[mStart + t].at(mGyroNames[i]);
    }
  }

  // Establish copies of everything that we will need for our threading
  mNumThreads = std::thread::hardware_concurrency();
  int cursor = 0;
  int chunkSize = mLength / mNumThreads;

  for (int t = 0; t < mNumThreads; t++)
  {
    std::shared_ptr<dynamics::Skeleton> threadSkel
        = mFitter->mSkeleton->clone();
    mThreadSkeletons.push_back(threadSkel);

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> threadMarkers;
    for (auto& pair : mFitter->mMarkers)
    {
      dynamics::BodyNode* threadBody
          = threadSkel->getBodyNode(pair.first->getName());
      assert(threadBody != nullptr);
      assert(threadBody->getIndexInSkeleton() < threadSkel->getNumBodyNodes());
      threadMarkers.push_back(std::make_pair(threadBody, pair.second));
    }
    mThreadMarkers.push_back(threadMarkers);

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>
        threadAccelerometers;
    for (auto& pair : mAccelerometers)
    {
      dynamics::BodyNode* threadBody
          = threadSkel->getBodyNode(pair.first->getName());
      assert(threadBody != nullptr);
      assert(threadBody->getIndexInSkeleton() < threadSkel->getNumBodyNodes());
      threadAccelerometers.push_back(std::make_pair(threadBody, pair.second));
    }
    mThreadAccelerometers.push_back(threadAccelerometers);

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>> threadGyros;
    for (auto& pair : mGyros)
    {
      dynamics::BodyNode* threadBody
          = threadSkel->getBodyNode(pair.first->getName());
      assert(threadBody != nullptr);
      assert(threadBody->getIndexInSkeleton() < threadSkel->getNumBodyNodes());
      threadGyros.push_back(std::make_pair(threadBody, pair.second));
    }
    mThreadGyros.push_back(threadGyros);

    if (t == mNumThreads - 1)
      mThreadRanges.emplace_back(cursor, mLength);
    else
      mThreadRanges.emplace_back(cursor, cursor + chunkSize);
    cursor += chunkSize;
  }
}

/// Get the size of the x vector for the problem
int IMUFineTuneProblem::getProblemSize()
{
  int dofs = 0;

  dofs += mFitter->mSkeleton->getNumDofs() * (mLength + 2);

  return dofs;
}

void IMUFineTuneProblem::setWeightGyros(s_t weight)
{
  mWeightGyros = weight;
}

void IMUFineTuneProblem::setWeightAccs(s_t weight)
{
  mWeightAccs = weight;
}

void IMUFineTuneProblem::setWeightMarkers(s_t weight)
{
  mWeightMarkers = weight;
}

void IMUFineTuneProblem::setRegularizePoses(s_t weight)
{
  mRegularizePoses = weight;
}

void IMUFineTuneProblem::setUseMultithreading(bool useMultithreading)
{
  mUseMultiThreading = useMultithreading;
}

/// Returns the current x vector
Eigen::VectorXs IMUFineTuneProblem::flatten()
{
  Eigen::VectorXs x = Eigen::VectorXs::Zero(getProblemSize());
  int cursor = 0;

  for (int t = 0; t < mLength; t++)
  {
    // The first timestep gets a position, velocity, and acceleration
    if (t == 0)
    {
      x.segment(cursor, mFitter->mSkeleton->getNumDofs()) = mPoses.col(t);
      cursor += mFitter->mSkeleton->getNumDofs();
      x.segment(cursor, mFitter->mSkeleton->getNumDofs()) = mVels.col(t);
      cursor += mFitter->mSkeleton->getNumDofs();
      x.segment(cursor, mFitter->mSkeleton->getNumDofs()) = mAccs.col(t);
      cursor += mFitter->mSkeleton->getNumDofs();
    }
    // All other timesteps get just an acceleration
    else
    {
      x.segment(cursor, mFitter->mSkeleton->getNumDofs()) = mAccs.col(t);
      cursor += mFitter->mSkeleton->getNumDofs();
    }
  }

  return x;
}

//==============================================================================
/// Sets the state of the IMU problem to the given x vector
void IMUFineTuneProblem::unflatten(Eigen::VectorXs x)
{
  Eigen::VectorXs q = x.segment(0, mFitter->mSkeleton->getNumDofs());
  Eigen::VectorXs dq = x.segment(
      mFitter->mSkeleton->getNumDofs(), mFitter->mSkeleton->getNumDofs());
  int cursor = 2 * mFitter->mSkeleton->getNumDofs();
  for (int t = 0; t < mLength; t++)
  {
    Eigen::VectorXs ddq = x.segment(cursor, mFitter->mSkeleton->getNumDofs());

    cursor += mFitter->mSkeleton->getNumDofs();
    mAccs.col(t) = ddq;
    mVels.col(t) = dq;
    mPoses.col(t) = q;

    dq += ddq * mTimeStep;
    q += dq * mTimeStep;
  }
}

//==============================================================================
/// This computes the loss for the passed in x vector
s_t IMUFineTuneProblem::getLoss()
{
  s_t sum = 0.0;
  if (mUseMultiThreading)
  {
    std::vector<std::future<s_t>> futures;
    for (int threadIdx = 0; threadIdx < mNumThreads; threadIdx++)
    {
      futures.push_back(std::async([&, threadIdx]() {
        s_t threadSum = 0.0;
        std::shared_ptr<dynamics::Skeleton>& skel = mThreadSkeletons[threadIdx];
        for (int t = mThreadRanges[threadIdx].first;
             t < mThreadRanges[threadIdx].second;
             t++)
        {
          skel->setPositions(mPoses.col(t));
          skel->setVelocities(mVels.col(t));
          skel->setAccelerations(mAccs.col(t));

          Eigen::VectorXs accError = mAccelerometerObservations.col(t)
                                     - skel->getAccelerometerReadings(
                                         mThreadAccelerometers[threadIdx]);
          threadSum += mWeightAccs * accError.squaredNorm();

          Eigen::VectorXs gyroError
              = mGyroObservations.col(t)
                - skel->getGyroReadings(mThreadGyros[threadIdx]);
          threadSum += mWeightGyros * gyroError.squaredNorm();

          std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
          for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
          {
            if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
            {
              markers.push_back(mThreadMarkers[threadIdx][i]);
            }
          }
          Eigen::VectorXs markerPositions
              = Eigen::VectorXs::Zero(markers.size() * 3);
          int cursor = 0;
          for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
          {
            if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
            {
              markerPositions.segment<3>(cursor)
                  = mMarkerObservations[t].at(mFitter->mMarkerNames[i]);
              cursor += 3;
            }
          }
          Eigen::VectorXs markerError
              = skel->getMarkerWorldPositions(markers) - markerPositions;
          threadSum += mWeightMarkers * markerError.squaredNorm();

          Eigen::VectorXs poseChange = mPoses.col(t) - mOriginalPoses.col(t);
          threadSum += mRegularizePoses * poseChange.squaredNorm();
        }
        return threadSum;
      }));
    }
    for (int threadIdx = 0; threadIdx < mNumThreads; threadIdx++)
    {
      sum += futures[threadIdx].get();
    }
  }
  else
  {
    for (int t = 0; t < mLength; t++)
    {
      mFitter->mSkeleton->setPositions(mPoses.col(t));
      mFitter->mSkeleton->setVelocities(mVels.col(t));
      mFitter->mSkeleton->setAccelerations(mAccs.col(t));

      Eigen::VectorXs accError
          = mAccelerometerObservations.col(t)
            - mFitter->mSkeleton->getAccelerometerReadings(mAccelerometers);
      sum += mWeightAccs * accError.squaredNorm();

      Eigen::VectorXs gyroError = mGyroObservations.col(t)
                                  - mFitter->mSkeleton->getGyroReadings(mGyros);
      sum += mWeightGyros * gyroError.squaredNorm();

      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
      for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
      {
        if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
        {
          markers.push_back(mFitter->mMarkers[i]);
        }
      }
      Eigen::VectorXs markerPositions
          = Eigen::VectorXs::Zero(markers.size() * 3);
      int cursor = 0;
      for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
      {
        if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
        {
          markerPositions.segment<3>(cursor)
              = mMarkerObservations[t].at(mFitter->mMarkerNames[i]);
          cursor += 3;
        }
      }
      Eigen::VectorXs markerError
          = mFitter->mSkeleton->getMarkerWorldPositions(markers)
            - markerPositions;
      sum += mWeightMarkers * markerError.squaredNorm();

      Eigen::VectorXs poseChange = mPoses.col(t) - mOriginalPoses.col(t);
      sum += mRegularizePoses * poseChange.squaredNorm();
    }
  }

  return sum;
}

//==============================================================================
/// Returns the pose data over time for the current problem state
Eigen::MatrixXs& IMUFineTuneProblem::getPoses()
{
  return mPoses;
}

//==============================================================================
/// Returns the vel data over time for the current problem state
Eigen::MatrixXs& IMUFineTuneProblem::getVels()
{
  return mVels;
}

//==============================================================================
/// Returns the acceleration data over time for the current problem state
Eigen::MatrixXs& IMUFineTuneProblem::getAccs()
{
  return mAccs;
}

//==============================================================================
/// This computes the gradient for the passed in x vector
Eigen::VectorXs IMUFineTuneProblem::getGrad()
{
  Eigen::VectorXs x = flatten();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());

  Eigen::VectorXs posGrad
      = getGradientWrtFlattenedState(neural::WithRespectTo::POSITION);
  grad += getJacobianFromXToFlattenedState(neural::WithRespectTo::POSITION)
              .transpose()
          * posGrad;

  Eigen::VectorXs velGrad
      = getGradientWrtFlattenedState(neural::WithRespectTo::VELOCITY);
  grad += getJacobianFromXToFlattenedState(neural::WithRespectTo::VELOCITY)
              .transpose()
          * velGrad;

  Eigen::VectorXs accGrad
      = getGradientWrtFlattenedState(neural::WithRespectTo::ACCELERATION);
  grad += getJacobianFromXToFlattenedState(neural::WithRespectTo::ACCELERATION)
              .transpose()
          * accGrad;

  return grad;
}

//==============================================================================
/// This is the brute force version of the getGrad() computation
Eigen::VectorXs IMUFineTuneProblem::finiteDifferenceGrad()
{
  Eigen::VectorXs x = flatten();
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());

  math::finiteDifference(
      [this, &x](
          /* in*/ s_t eps,
          /* in*/ int dof,

          /*out*/ s_t& perturbed) {
        Eigen::VectorXs tweaked = x;
        tweaked(dof) += eps;
        unflatten(tweaked);
        perturbed = getLoss();
        return true;
      },
      grad,
      1e-2,
      true);

  unflatten(x);

  return grad;
}

//==============================================================================
/// This computes the gradient with respect to one of the state types.
Eigen::VectorXs IMUFineTuneProblem::getGradientWrtFlattenedState(
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(mPoses.rows() * mPoses.cols());
  if (mUseMultiThreading)
  {
    std::vector<std::future<void>> futures;
    for (int threadIdx = 0; threadIdx < mNumThreads; threadIdx++)
    {
      futures.push_back(std::async([&, threadIdx]() {
        std::shared_ptr<dynamics::Skeleton>& skel = mThreadSkeletons[threadIdx];
        for (int t = mThreadRanges[threadIdx].first;
             t < mThreadRanges[threadIdx].second;
             t++)
        {
          skel->setPositions(mPoses.col(t));
          skel->setVelocities(mVels.col(t));
          skel->setAccelerations(mAccs.col(t));

          Eigen::VectorXs accError
              = skel->getAccelerometerReadings(mThreadAccelerometers[threadIdx])
                - mAccelerometerObservations.col(t);
          grad.segment(t * skel->getNumDofs(), skel->getNumDofs())
              += mWeightAccs * 2
                 * skel->getAccelerometerReadingsJacobianWrt(
                           mThreadAccelerometers[threadIdx], wrt)
                       .transpose()
                 * accError;

          Eigen::VectorXs gyroError
              = skel->getGyroReadings(mThreadGyros[threadIdx])
                - mGyroObservations.col(t);
          grad.segment(t * skel->getNumDofs(), skel->getNumDofs())
              += mWeightGyros * 2
                 * skel->getGyroReadingsJacobianWrt(
                           mThreadGyros[threadIdx], wrt)
                       .transpose()
                 * gyroError;

          if (wrt == neural::WithRespectTo::POSITION)
          {
            std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
                markers;
            for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
            {
              if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
              {
                markers.push_back(mThreadMarkers[threadIdx][i]);
              }
            }
            Eigen::VectorXs markerPositions
                = Eigen::VectorXs::Zero(markers.size() * 3);
            int cursor = 0;
            for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
            {
              if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
              {
                markerPositions.segment<3>(cursor)
                    = mMarkerObservations[t].at(mFitter->mMarkerNames[i]);
                cursor += 3;
              }
            }
            Eigen::VectorXs markerError
                = skel->getMarkerWorldPositions(markers) - markerPositions;
            Eigen::MatrixXs markerJac
                = skel->getMarkerWorldPositionsJacobianWrtJointPositions(
                    markers);
            grad.segment(t * skel->getNumDofs(), skel->getNumDofs())
                += 2 * mWeightMarkers * markerJac.transpose() * markerError;

            Eigen::VectorXs poseChange = mPoses.col(t) - mOriginalPoses.col(t);
            grad.segment(t * skel->getNumDofs(), skel->getNumDofs())
                += 2 * mRegularizePoses * poseChange;
          }
        }
      }));
    }
    for (int threadIdx = 0; threadIdx < mNumThreads; threadIdx++)
    {
      futures[threadIdx].get();
    }
  }
  else
  {
    for (int t = 0; t < mPoses.cols(); t++)
    {
      mFitter->mSkeleton->setPositions(mPoses.col(t));
      mFitter->mSkeleton->setVelocities(mVels.col(t));
      mFitter->mSkeleton->setAccelerations(mAccs.col(t));

      Eigen::VectorXs accError
          = mFitter->mSkeleton->getAccelerometerReadings(mAccelerometers)
            - mAccelerometerObservations.col(t);
      grad.segment(
          t * mFitter->mSkeleton->getNumDofs(),
          mFitter->mSkeleton->getNumDofs())
          += mWeightAccs * 2
             * mFitter->mSkeleton
                   ->getAccelerometerReadingsJacobianWrt(mAccelerometers, wrt)
                   .transpose()
             * accError;

      Eigen::VectorXs gyroError = mFitter->mSkeleton->getGyroReadings(mGyros)
                                  - mGyroObservations.col(t);
      grad.segment(
          t * mFitter->mSkeleton->getNumDofs(),
          mFitter->mSkeleton->getNumDofs())
          += mWeightGyros * 2
             * mFitter->mSkeleton->getGyroReadingsJacobianWrt(mGyros, wrt)
                   .transpose()
             * gyroError;

      if (wrt == neural::WithRespectTo::POSITION)
      {
        std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
        for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
        {
          if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
          {
            markers.push_back(mFitter->mMarkers[i]);
          }
        }
        Eigen::VectorXs markerPositions
            = Eigen::VectorXs::Zero(markers.size() * 3);
        int cursor = 0;
        for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
        {
          if (mMarkerObservations[t].count(mFitter->mMarkerNames[i]))
          {
            markerPositions.segment<3>(cursor)
                = mMarkerObservations[t].at(mFitter->mMarkerNames[i]);
            cursor += 3;
          }
        }
        Eigen::VectorXs markerError
            = mFitter->mSkeleton->getMarkerWorldPositions(markers)
              - markerPositions;
        Eigen::MatrixXs markerJac
            = mFitter->mSkeleton
                  ->getMarkerWorldPositionsJacobianWrtJointPositions(markers);
        grad.segment(
            t * mFitter->mSkeleton->getNumDofs(),
            mFitter->mSkeleton->getNumDofs())
            += 2 * mWeightMarkers * markerJac.transpose() * markerError;

        Eigen::VectorXs poseChange = mPoses.col(t) - mOriginalPoses.col(t);
        grad.segment(
            t * mFitter->mSkeleton->getNumDofs(),
            mFitter->mSkeleton->getNumDofs())
            += 2 * mRegularizePoses * poseChange;
      }
    }
  }

  return grad;
}

//==============================================================================
/// This computes the gradient with respect to one of the state types.
Eigen::VectorXs IMUFineTuneProblem::finiteDifferenceGradientWrtFlattenedState(
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs grad
      = Eigen::VectorXs::Zero(mFitter->mSkeleton->getNumDofs() * mLength);

  Eigen::MatrixXs original = mPoses;
  if (wrt == neural::WithRespectTo::POSITION)
  {
    original = mPoses;
  }
  if (wrt == neural::WithRespectTo::VELOCITY)
  {
    original = mVels;
  }
  else if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    original = mAccs;
  }

  math::finiteDifference(
      [this, wrt, original](
          /* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        int row = dof % mFitter->mSkeleton->getNumDofs();
        int col = (int)floor((s_t)dof / mFitter->mSkeleton->getNumDofs());
        if (wrt == neural::WithRespectTo::POSITION)
        {
          mPoses = original;
          mPoses(row, col) += eps;
        }
        if (wrt == neural::WithRespectTo::VELOCITY)
        {
          mVels = original;
          mVels(row, col) += eps;
        }
        else if (wrt == neural::WithRespectTo::ACCELERATION)
        {
          mAccs = original;
          mAccs(row, col) += eps;
        }
        perturbed = getLoss();
        return true;
      },
      grad,
      1e-2,
      true);

  if (wrt == neural::WithRespectTo::POSITION)
  {
    mPoses = original;
  }
  if (wrt == neural::WithRespectTo::VELOCITY)
  {
    mVels = original;
  }
  else if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    mAccs = original;
  }

  return grad;
}

//==============================================================================
/// This returns the jacobian relating changes in the flattened state vector
/// to changes in X.
Eigen::SparseMatrix<s_t>& IMUFineTuneProblem::getJacobianFromXToFlattenedState(
    neural::WithRespectTo* wrt)
{
  const int dofs = mFitter->mSkeleton->getNumDofs();
  if (!mCachedJacobianFromXToFlattenedState.count(wrt->name()))
  {
    Eigen::MatrixXs result
        = Eigen::MatrixXs::Zero(dofs * mLength, getProblemSize());

    if (wrt == neural::WithRespectTo::POSITION)
    {
      // Changes to the initial position have an identity effect on all
      // subsequent positions
      for (int t = 0; t < mLength; t++)
      {
        result.block(t * dofs, 0, dofs, dofs)
            = Eigen::MatrixXs::Identity(dofs, dofs);
      }
      // Changes to the initial velocity have a linear effect on all subsequent
      // positions
      for (int t = 0; t < mLength; t++)
      {
        result.block(t * dofs, dofs, dofs, dofs)
            = mTimeStep * t * Eigen::MatrixXs::Identity(dofs, dofs);
      }
      // Changes to acceleration have a linear effect on all subsequent
      // positions
      for (int accT = 0; accT < mLength; accT++)
      {
        for (int t = accT; t < mLength; t++)
        {
          result.block(t * dofs, (accT + 2) * dofs, dofs, dofs)
              = mTimeStep * mTimeStep * (t - accT)
                * Eigen::MatrixXs::Identity(dofs, dofs);
        }
      }
    }
    else if (wrt == neural::WithRespectTo::VELOCITY)
    {
      // Changes to the initial velocity have an identity effect on all
      // subsequent velocities
      for (int t = 0; t < mLength; t++)
      {
        result.block(t * dofs, dofs, dofs, dofs)
            = Eigen::MatrixXs::Identity(dofs, dofs);
      }
      // Changes to acceleration have a linear effect on all subsequent
      // velocities
      for (int accT = 0; accT < mLength; accT++)
      {
        for (int t = accT + 1; t < mLength; t++)
        {
          result.block(t * dofs, (accT + 2) * dofs, dofs, dofs)
              = mTimeStep * Eigen::MatrixXs::Identity(dofs, dofs);
        }
      }
    }
    else if (wrt == neural::WithRespectTo::ACCELERATION)
    {
      // Changes to acceleration have an identity effect on that timestep's
      // acceleration
      for (int accT = 0; accT < mLength; accT++)
      {
        result.block(accT * dofs, (accT + 2) * dofs, dofs, dofs)
            = Eigen::MatrixXs::Identity(dofs, dofs);
      }
    }

    mCachedJacobianFromXToFlattenedState[wrt->name()] = result.sparseView();
  }

  return mCachedJacobianFromXToFlattenedState.at(wrt->name());
}

//==============================================================================
/// This returns the jacobian relating changes in the flattened state vector
/// to changes in X.
Eigen::MatrixXs
IMUFineTuneProblem::finiteDifferenceJacobianFromXToFlattenedState(
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs original = flatten();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(
      mFitter->mSkeleton->getNumDofs() * mLength, original.size());

  const bool useRidders = true;
  s_t eps = useRidders ? 1e-3 : 1e-6;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        unflatten(tweaked);

        perturbed
            = Eigen::VectorXs::Zero(mFitter->mSkeleton->getNumDofs() * mLength);
        for (int i = 0; i < mLength; i++)
        {
          if (wrt == neural::WithRespectTo::POSITION)
          {
            perturbed.segment(
                i * mFitter->mSkeleton->getNumDofs(),
                mFitter->mSkeleton->getNumDofs())
                = mPoses.col(i);
          }
          else if (wrt == neural::WithRespectTo::VELOCITY)
          {
            perturbed.segment(
                i * mFitter->mSkeleton->getNumDofs(),
                mFitter->mSkeleton->getNumDofs())
                = mVels.col(i);
          }
          else if (wrt == neural::WithRespectTo::ACCELERATION)
          {
            perturbed.segment(
                i * mFitter->mSkeleton->getNumDofs(),
                mFitter->mSkeleton->getNumDofs())
                = mAccs.col(i);
          }
        }
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//------------------------- Ipopt::TNLP --------------------------------------
//==============================================================================
/// \brief Method to return some info about the nlp
bool IMUFineTuneProblem::get_nlp_info(
    Ipopt::Index& n,
    Ipopt::Index& m,
    Ipopt::Index& nnz_jac_g,
    Ipopt::Index& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  n = getProblemSize();
  m = 0;
  nnz_jac_g = 0;
  nnz_h_lag = 0;
  (void)index_style;

  return true;
}

//==============================================================================
/// \brief Method to return the bounds for my problem
bool IMUFineTuneProblem::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  (void)n;
  assert(n == getProblemSize());

  // We default bounds to a large finite value, rather than infinity, to prevent
  // NaNs from creeping into computations where they're not wanted.
  s_t LARGE_FINITE_VALUE = 10000000;
  // Lower and upper bounds on X
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  upperBounds.setConstant(LARGE_FINITE_VALUE);
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  lowerBounds.setConstant(-LARGE_FINITE_VALUE);

  (void)m;
  assert(m == 0);
  (void)g_l;
  (void)g_u;

  return true;
}

//==============================================================================
/// \brief Method to return the starting point for the algorithm
bool IMUFineTuneProblem::get_starting_point(
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number* _x,
    bool init_z,
    Ipopt::Number* z_L,
    Ipopt::Number* z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number* lambda)
{
  // Here, we assume we only have starting values for x
  (void)init_x;
  assert(init_x == true);
  (void)init_z;
  assert(init_z == false);
  (void)init_lambda;
  assert(init_lambda == false);
  // We don't set the lagrange multipliers
  (void)z_L;
  (void)z_U;
  (void)m;
  (void)lambda;

  if (init_x)
  {
    Eigen::Map<Eigen::VectorXd> x(_x, n);
    x = flatten().cast<double>();
  }

  return true;
}

//==============================================================================
/// \brief Method to return the objective value
bool IMUFineTuneProblem::eval_f(
    Ipopt::Index n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  if (_new_x)
  {
    Eigen::Map<const Eigen::VectorXd> x(_x, n);
    unflatten(x.cast<s_t>());
  }
  _obj_value = getLoss();
  return true;
}

//==============================================================================
/// \brief Method to return the gradient of the objective
bool IMUFineTuneProblem::eval_grad_f(
    Ipopt::Index n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  if (_new_x)
  {
    Eigen::Map<const Eigen::VectorXd> x(_x, n);
    unflatten(x.cast<s_t>());
  }
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, n);
  grad = getGrad().cast<s_t>();
  return true;
}

//==============================================================================
/// \brief Method to return the constraint residuals
bool IMUFineTuneProblem::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  (void)_n;
  (void)_x;
  (void)_new_x;
  (void)_m;
  (void)_g;
  return true;
}

//==============================================================================
/// \brief Method to return:
///        1) The structure of the jacobian (if "values" is nullptr)
///        2) The values of the jacobian (if "values" is not nullptr)
bool IMUFineTuneProblem::eval_jac_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Index _nele_jac,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_n;
  (void)_x;
  (void)_new_x;
  (void)_m;
  (void)_nele_jac;
  (void)_iRow;
  (void)_jCol;
  (void)_values;
  return true;
}

//==============================================================================
/// \brief Method to return:
///        1) The structure of the hessian of the lagrangian (if "values" is
///           nullptr)
///        2) The values of the hessian of the lagrangian (if "values" is not
///           nullptr)
bool IMUFineTuneProblem::eval_h(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number _obj_factor,
    Ipopt::Index _m,
    const Ipopt::Number* _lambda,
    bool _new_lambda,
    Ipopt::Index _nele_hess,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_n;
  (void)_x;
  (void)_new_x;
  (void)_obj_factor;
  (void)_m;
  (void)_lambda;
  (void)_new_lambda;
  (void)_nele_hess;
  (void)_iRow;
  (void)_jCol;
  (void)_values;
  return true;
}

//==============================================================================
/// \brief This method is called when the algorithm is complete so the TNLP
///        can store/write the solution
void IMUFineTuneProblem::finalize_solution(
    Ipopt::SolverReturn _status,
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    const Ipopt::Number* _z_L,
    const Ipopt::Number* _z_U,
    Ipopt::Index _m,
    const Ipopt::Number* _g,
    const Ipopt::Number* _lambda,
    Ipopt::Number _obj_value,
    const Ipopt::IpoptData* _ip_data,
    Ipopt::IpoptCalculatedQuantities* _ip_cq)
{
  (void)_status;
  (void)_n;
  (void)_z_L;
  (void)_z_U;
  (void)_m;
  (void)_g;
  (void)_lambda;
  (void)_obj_value;
  (void)_ip_data;
  (void)_ip_cq;
  if (mBestObjectiveValueIteration != -1)
  {
    std::cout << "Recovering iteration " << mBestObjectiveValueIteration
              << " with value " << mBestObjectiveValue << std::endl;
    unflatten(mBestObjectiveValueState);
  }
  else
  {
    Eigen::Map<const Eigen::VectorXd> x(_x, _n);
    unflatten(x.cast<s_t>());
  }
  // Write the results back to the initialization
  mInitialization.poses.block(0, mStart, mInitialization.poses.rows(), mLength)
      = mPoses;
}

//==============================================================================
bool IMUFineTuneProblem::intermediate_callback(
    Ipopt::AlgorithmMode mode,
    Ipopt::Index iter,
    Ipopt::Number obj_value,
    Ipopt::Number inf_pr,
    Ipopt::Number inf_du,
    Ipopt::Number mu,
    Ipopt::Number d_norm,
    Ipopt::Number regularization_size,
    Ipopt::Number alpha_du,
    Ipopt::Number alpha_pr,
    Ipopt::Index ls_trials,
    const Ipopt::IpoptData* ip_data,
    Ipopt::IpoptCalculatedQuantities* ip_cq)
{
  (void)mode;
  (void)iter;
  (void)obj_value;
  (void)inf_pr;
  (void)inf_du;
  (void)mu;
  (void)d_norm;
  (void)regularization_size;
  (void)alpha_du;
  (void)alpha_pr;
  (void)ls_trials;
  (void)ip_data;
  (void)ip_cq;

  if (obj_value < mBestObjectiveValue && abs(inf_pr) < 1.0)
  {
    mBestObjectiveValueIteration = iter;
    mBestObjectiveValue = obj_value;
    mBestObjectiveValueState = flatten();
  }

  return true;
}

} // namespace biomechanics
} // namespace dart