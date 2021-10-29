#include "dart/biomechanics/MarkerFitter.hpp"

#include <future>
#include <mutex>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

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
    MarkerFitter* fitter)
  : markerOrder(fitter->mMarkerNames),
    skeleton(fitter->mSkeleton),
    markerObservations(markerObservations),
    joints(joints),
    jointCenters(jointCenters),
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

  (void)posesDim; // Don't break the compile when we strip out asserts
  assert(flat.size() == groupScaleDim + markerOffsetDim + posesDim);

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
  }

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

  Eigen::VectorXs flat
      = Eigen::VectorXs::Zero(groupScaleDim + markerOffsetDim + posesDim);

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

  // 1. Write scale grad

  Eigen::VectorXs grad
      = Eigen::VectorXs::Zero(groupScaleDim + markerOffsetDim + posesDim);

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

    // Get loss wrt joint positions
    grad.segment(offset, skeleton->getNumDofs())
        += fitter->getMarkerLossGradientWrtJoints(
            skeleton, markers, markerErrorGrad);
    grad.segment(offset, skeleton->getNumDofs())
        += skeleton->getJointWorldPositionsJacobianWrtJointPositions(joints)
               .transpose()
           * jointErrorGrad;

    // Acculumulate loss wrt the global scale groups
    grad.segment(0, groupScaleDim)
        += fitter->getMarkerLossGradientWrtGroupScales(
            skeleton, markers, markerErrorGrad);
    grad.segment(0, groupScaleDim)
        += skeleton->getJointWorldPositionsJacobianWrtGroupScales(joints)
               .transpose()
           * jointErrorGrad;

    // Acculumulate loss wrt the global marker offsets (this is 0 for joints,
    // since marker offsets don't change joint locations)
    grad.segment(groupScaleDim, markerOffsetDim)
        += fitter->getMarkerLossGradientWrtMarkerOffsets(
            skeleton, markers, markerErrorGrad);
  }

  skeleton->setGroupScales(originalScales);
  skeleton->setPositions(originalPos);

  return grad;
}

//==============================================================================
BilevelFitResult::BilevelFitResult() : success(false){};

//==============================================================================
InitialMarkerFitParams::InitialMarkerFitParams()
  : numBlocks(12), dontRescaleBodies(false)
{
}

//==============================================================================
InitialMarkerFitParams::InitialMarkerFitParams(
    const InitialMarkerFitParams& other)
  : markerWeights(other.markerWeights),
    joints(other.joints),
    jointCenters(other.jointCenters),
    jointWeights(other.jointWeights),
    numBlocks(other.numBlocks),
    initPoses(other.initPoses),
    markerOffsets(other.markerOffsets),
    groupScales(other.groupScales),
    dontRescaleBodies(other.dontRescaleBodies)
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
    std::vector<dynamics::Joint*> joints, Eigen::MatrixXs jointCenters)
{
  this->joints = joints;
  this->jointCenters = jointCenters;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setJointCentersAndWeights(
    std::vector<dynamics::Joint*> joints,
    Eigen::MatrixXs jointCenters,
    Eigen::VectorXs jointWeights)
{
  this->joints = joints;
  this->jointCenters = jointCenters;
  this->jointWeights = jointWeights;
  return *this;
}

//==============================================================================
InitialMarkerFitParams& InitialMarkerFitParams::setNumBlocks(int numBlocks)
{
  this->numBlocks = numBlocks;
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
MarkerFitter::MarkerFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton, dynamics::MarkerMap markers)
  : mSkeleton(skeleton),
    mMarkerMap(markers),
    mTolerance(1e-8),
    mIterationLimit(500),
    mLBFGSHistoryLength(8),
    mCheckDerivatives(false),
    mPrintFrequency(1),
    mSilenceOutput(false),
    mDisableLinesearch(false),
    mInitialIKSatisfactoryLoss(0.003),
    mInitialIKMaxRestarts(100),
    mMaxMarkerOffset(0.2)
{
  mSkeletonBallJoints = mSkeleton->convertSkeletonToBallJoints();
  int offset = 0;
  for (auto pair : markers)
  {
    mMarkerIndices[pair.first] = offset;
    mMarkerNames.push_back(pair.first);
    mMarkerIsTracking.push_back(false);
    offset++;
    mMarkers.push_back(pair.second);
    mMarkersBallJoints.emplace_back(
        mSkeletonBallJoints->getBodyNode(pair.second.first->getName()),
        Eigen::Vector3s(pair.second.second));

    // Traverse up the parent list looking for all the joints that effect this
    // node
    dynamics::BodyNode* cursor = pair.second.first;
    while (cursor != nullptr)
    {
      dynamics::Joint* parentJoint = cursor->getParentJoint();
      assert(parentJoint != nullptr);
      if (std::find(mObservedJoints.begin(), mObservedJoints.end(), parentJoint)
          == mObservedJoints.end())
      {
        mObservedJoints.push_back(parentJoint);
      }
      cursor = parentJoint->getParentBodyNode();
    }
  }

  // Default to a least-squares loss over just the marker errors
  mLossAndGrad = [](MarkerFitterState* state) {
    s_t loss = state->markerErrorsAtTimesteps.squaredNorm()
               + state->jointErrorsAtTimesteps.squaredNorm();
    state->markerErrorsAtTimestepsGrad = 2 * state->markerErrorsAtTimesteps;
    state->jointErrorsAtTimestepsGrad = 2 * state->jointErrorsAtTimesteps;
    return loss;
  };
}

//==============================================================================
/// Run the whole pipeline of optimization problems to fit the data as closely
/// as we can
MarkerInitialization MarkerFitter::runKinematicsPipeline(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    InitialMarkerFitParams params)
{
  // 1. Find the initial scaling + IK
  MarkerInitialization init = getInitialization(markerObservations, params);
  mSkeleton->setGroupScales(init.groupScales);

  // 2. Find the joint centers
  findJointCenters(init, markerObservations);

  // 3. Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = getInitialization(
      markerObservations,
      InitialMarkerFitParams(params)
          .setJointCenters(init.joints, init.jointCenters)
          .setInitPoses(init.poses));

  // 4. Run bilevel optimization
  std::shared_ptr<BilevelFitResult> bilevelFit
      = optimizeBilevel(markerObservations, reinit, 20);

  // 5. Fine-tune IK and re-fit all the points
  mSkeleton->setGroupScales(bilevelFit->groupScales);
  MarkerInitialization finalKinematicInit = getInitialization(
      markerObservations,
      InitialMarkerFitParams(params)
          .setJointCenters(init.joints, init.jointCenters)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));
  return finalKinematicInit;
}

//==============================================================================
/// This solves an optimization problem, trying to get the Skeleton to match
/// the markers as closely as possible.
std::shared_ptr<BilevelFitResult> MarkerFitter::optimizeBilevel(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    MarkerInitialization& initialization,
    int numSamples)
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
      this, markerObservations, initialization, numSamples, result);

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
  result->posesMatrix
      = Eigen::MatrixXs::Zero(result->poses[0].size(), result->poses.size());
  for (int i = 0; i < result->poses.size(); i++)
  {
    result->posesMatrix.col(i) = result->poses[i];
  }

  return result;
}

//==============================================================================
/// This finds an initial guess for the body scales and poses, holding
/// anatomical marker offsets at 0, that we can use for downstream tasks.
///
/// This can multithread over `numBlocks` independent sets of problems.
MarkerInitialization MarkerFitter::getInitialization(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    InitialMarkerFitParams params)
{
  // Before using Eigen in a multi-threaded environment, we need to explicitly
  // call this (at least prior to Eigen 3.3)
  Eigen::initParallel();

  MarkerInitialization result;

  assert(
      params.jointCenters.cols() == 0
      || params.jointCenters.cols() == markerObservations.size());

  // 0. Prep configuration variables we'll use for the rest of the algo
  int numBlocks = params.numBlocks;
  // Upper bound the number of blocks at the number of observations
  if (numBlocks > markerObservations.size())
  {
    numBlocks = markerObservations.size();
  }
  int blockLen = markerObservations.size() / numBlocks;
  int lastBlockLen = markerObservations.size() - (blockLen * (numBlocks - 1));
  std::vector<std::string> anatomicalMarkerNames;
  for (int j = 0; j < mMarkerNames.size(); j++)
  {
    if (!mMarkerIsTracking[j])
      anatomicalMarkerNames.push_back(mMarkerNames[j]);
  }

  // 1. Divide the marker observations into N sequential blocks.
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>> blocks;
  std::vector<Eigen::VectorXs> firstGuessPoses;
  std::vector<std::vector<Eigen::VectorXs>> jointCenterBlocks;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    if ((i % blockLen == 0) && (blocks.size() < numBlocks))
    {
      blocks.emplace_back();
      jointCenterBlocks.emplace_back();
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
    int mapIndex = blocks[blocks.size() - 1].size();
    blocks[blocks.size() - 1].emplace_back();
    for (std::string& marker : anatomicalMarkerNames)
    {
      if (markerObservations[i].count(marker) > 0)
      {
        blocks[blocks.size() - 1][mapIndex].emplace(
            marker, markerObservations[i].at(marker));
      }
    }
    assert(params.jointCenters.cols() == 0 || params.jointCenters.cols() > i);
    if (params.jointCenters.cols() > i)
    {
      jointCenterBlocks[jointCenterBlocks.size() - 1].emplace_back(
          params.jointCenters.col(i));
    }
    else
    {
      assert(params.jointCenters.cols() == 0);
      jointCenterBlocks[jointCenterBlocks.size() - 1].emplace_back(
          Eigen::VectorXs::Zero(0));
    }
  }

  for (int i = 0; i < numBlocks; i++)
  {
    assert(blocks[i].size() == (i == numBlocks - 1 ? lastBlockLen : blockLen));
  }

  // 2. Find IK+scaling for the beginning of each block independently
  std::vector<std::future<std::pair<Eigen::VectorXs, Eigen::VectorXs>>>
      posesAndScalesFutures;

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
    // posesAndScales.push_back(scaleAndFit(this, blocks[i][0]));
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
        params.dontRescaleBodies));
  }

  std::vector<std::pair<Eigen::VectorXs, Eigen::VectorXs>> posesAndScales;
  for (int i = 0; i < numBlocks; i++)
  {
    posesAndScales.push_back(posesAndScalesFutures[i].get());
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
  }
  else
  {
    Eigen::VectorXs averageGroupScales
        = Eigen::VectorXs::Zero(mSkeleton->getGroupScaleDim());
    for (int i = 0; i < numBlocks; i++)
    {
      averageGroupScales += posesAndScales[i].second;
    }
    averageGroupScales /= numBlocks;
    result.groupScales = averageGroupScales;
  }

  // 4. Go through and run IK on each block
  result.poses = Eigen::MatrixXs::Zero(
      mSkeleton->getNumDofs(), markerObservations.size());

  std::vector<std::future<void>> blockFitFutures;
  for (int i = 0; i < numBlocks; i++)
  {
    std::cout << "Starting fit for whole block " << i << "/" << numBlocks
              << std::endl;

    blockFitFutures.push_back(std::async(
        &MarkerFitter::fitTrajectory,
        this,
        result.groupScales,
        posesAndScales[i].first,
        blocks[i],
        params.markerWeights,
        params.markerOffsets,
        params.joints,
        jointCenterBlocks[i],
        params.jointWeights,
        result.poses.block(
            0,
            i * blockLen,
            mSkeleton->getNumDofs(),
            i == numBlocks - 1 ? lastBlockLen : blockLen)));
  }
  for (int i = 0; i < numBlocks; i++)
  {
    blockFitFutures[i].get();
    std::cout << "Finished fit for whole block " << i << "/" << numBlocks
              << std::endl;
  }

  // 5. Find the local offsets for the anthropometric markers as a simple
  // average

  // 5.1. Initialize empty maps to accumulate sums into

  std::map<std::string, Eigen::Vector3s> trackingMarkerObservationsSum;
  std::map<std::string, int> trackingMarkerNumObservations;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mMarkerIsTracking[i])
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
      if (mMarkerIsTracking[index] || true)
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
      result.markerOffsets[name] = params.markerOffsets.at(name);
      result.updatedMarkerMap[name]
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              mMarkerMap[name].first,
              mMarkerMap[name].second + params.markerOffsets.at(name));
    }
    else
    {
      if (mMarkerIsTracking[i] || true)
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

  result.joints = params.joints;
  result.jointCenters = params.jointCenters;

  return result;
}

//==============================================================================
/// This scales the skeleton and IK fits to the marker observations. It
/// returns a pair, with (pose, group scales) from the fit.
std::pair<Eigen::VectorXs, Eigen::VectorXs> MarkerFitter::scaleAndFit(
    const MarkerFitter* fitter,
    std::map<std::string, Eigen::Vector3s> markerObservations,
    Eigen::VectorXs firstGuessPose,
    std::map<std::string, s_t> markerWeights,
    std::map<std::string, Eigen::Vector3s> markerOffsets,
    std::vector<dynamics::Joint*> joints,
    Eigen::VectorXs jointCenters,
    Eigen::VectorXs jointWeights,
    bool dontScale)
{
  // 0. To make this thread safe, we're going to clone the fitter skeleton
  std::shared_ptr<dynamics::Skeleton> skeleton;
  {
    const std::lock_guard<std::mutex> lock(
        *(const_cast<std::mutex*>(&fitter->mGlobalLock)));
    skeleton = fitter->mSkeleton->clone();
  }
  skeleton->setPositions(firstGuessPose);

  // 0.1. Translate over the observedJoints array to the cloned skeleton
  std::vector<dynamics::Joint*> observedJoints;
  for (auto joint : fitter->mObservedJoints)
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
  for (std::pair<std::string, Eigen::Vector3s> pair : markerObservations)
  {
    markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
    if (markerWeights.count(pair.first))
    {
      markerWeightsVector(markerVector.size()) = markerWeights.at(pair.first);
    }
    const std::pair<dynamics::BodyNode*, Eigen::Vector3s>& originalMarker
        = fitter->mMarkerMap.at(pair.first);
    Eigen::Vector3s offset = Eigen::Vector3s::Zero();
    if (markerOffsets.count(pair.first))
    {
      offset = markerOffsets.at(pair.first);
    }
    markerVector.emplace_back(
        skeletonBallJoints->getBodyNode(originalMarker.first->getName()),
        originalMarker.second + offset);
  }

  if (dontScale)
  {
    // 1.3. Calculate problem size
    int problemDim = skeletonBallJoints->getNumDofs();

    // 1.4. Set our initial guess for IK to whatever the current pose of the
    // skeleton is
    Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);
    initialPos = skeletonBallJoints->convertPositionsToBallSpace(
        skeletonBallJoints->getPositions());

    // 2. Actually solve the IK
    math::solveIK(
        initialPos,
        (markerObservations.size() * 3) + (joints.size() * 3),
        // Set positions
        [&skeletonBallJoints, &skeleton](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(pos);

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            skeleton->setPositions(
                skeleton->convertPositionsFromBallSpace(pos));
            // 2. Clamp the position to limits
            skeleton->clampPositionsToLimits();
            // 3. Map the position back into SO3 space
            skeletonBallJoints->setPositions(
                skeleton->convertPositionsToBallSpace(
                    skeleton->getPositions()));
          }

          // Return the clamped position
          return skeletonBallJoints->getPositions();
        },
        // Compute the Jacobian
        [&skeletonBallJoints,
         &markerPoses,
         &markerVector,
         &markerWeightsVector,
         &jointsForSkeletonBallJoints,
         &jointCenters,
         &jointWeights](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff.segment(0, markerPoses.size())
              = markerPoses
                - skeletonBallJoints->getMarkerWorldPositions(markerVector);
          for (int i = 0; i < markerWeightsVector.size(); i++)
          {
            diff.segment<3>(i * 3) *= markerWeightsVector(i);
          }
          diff.segment(markerPoses.size(), jointCenters.size())
              = jointCenters
                - skeletonBallJoints->getJointWorldPositions(
                    jointsForSkeletonBallJoints);
          for (int i = 0; i < jointWeights.size(); i++)
          {
            diff.segment<3>(markerPoses.size() + (i * 3)) *= jointWeights(i);
          }

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
        },
        // Generate a random restart position
        [&skeleton, &observedJoints](Eigen::VectorXs& val) {
          val = skeleton->convertPositionsToBallSpace(
              skeleton->getRandomPoseForJoints(observedJoints));
        },
        math::IKConfig()
            .setMaxStepCount(150)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(fitter->mInitialIKSatisfactoryLoss)
            .setMaxRestarts(fitter->mInitialIKMaxRestarts)
            .setLogOutput(false));
  }
  else
  {
    // 1.3. Calculate problem size
    int problemDim = skeletonBallJoints->getNumDofs()
                     + skeletonBallJoints->getGroupScaleDim();

    // 1.4. Set our initial guess for IK to whatever the current pose of the
    // skeleton is
    Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);
    initialPos.segment(0, skeletonBallJoints->getNumDofs())
        = skeletonBallJoints->convertPositionsToBallSpace(
            skeletonBallJoints->getPositions());
    initialPos.segment(
        skeletonBallJoints->getNumDofs(),
        skeletonBallJoints->getGroupScaleDim())
        = skeletonBallJoints->getGroupScales();

    // 2. Actually solve the IK
    math::solveIK(
        initialPos,
        (markerObservations.size() * 3) + (joints.size() * 3),
        // Set positions
        [&skeletonBallJoints, &skeleton](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(
              pos.segment(0, skeletonBallJoints->getNumDofs()));

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            skeleton->setPositions(skeleton->convertPositionsFromBallSpace(
                pos.segment(0, skeletonBallJoints->getNumDofs())));
            // 2. Clamp the position to limits
            skeleton->clampPositionsToLimits();
            // 3. Map the position back into SO3 space
            skeletonBallJoints->setPositions(
                skeleton->convertPositionsToBallSpace(
                    skeleton->getPositions()));
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = skeletonBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = skeletonBallJoints->getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          skeleton->setGroupScales(newScales);
          skeletonBallJoints->setGroupScales(newScales);

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
        [&skeletonBallJoints,
         &markerPoses,
         &markerVector,
         &markerWeightsVector,
         &jointsForSkeletonBallJoints,
         &jointCenters,
         &jointWeights](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff.segment(0, markerPoses.size())
              = markerPoses
                - skeletonBallJoints->getMarkerWorldPositions(markerVector);
          for (int i = 0; i < markerWeightsVector.size(); i++)
          {
            diff.segment<3>(i * 3) *= markerWeightsVector(i);
          }
          diff.segment(markerPoses.size(), jointCenters.size())
              = jointCenters
                - skeletonBallJoints->getJointWorldPositions(
                    jointsForSkeletonBallJoints);
          for (int i = 0; i < jointWeights.size(); i++)
          {
            diff.segment<3>(markerPoses.size() + (i * 3)) *= jointWeights(i);
          }

          assert(
              jac.cols()
              == skeletonBallJoints->getNumDofs()
                     + skeletonBallJoints->getGroupScaleDim());
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
              0,
              skeletonBallJoints->getNumDofs(),
              markerVector.size() * 3,
              skeletonBallJoints->getGroupScaleDim())
              = skeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtGroupScales(
                        markerVector);

          jac.block(
              markerVector.size() * 3,
              0,
              jointsForSkeletonBallJoints.size() * 3,
              skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtJointPositions(
                        jointsForSkeletonBallJoints);
          jac.block(
              markerVector.size() * 3,
              skeletonBallJoints->getNumDofs(),
              jointsForSkeletonBallJoints.size() * 3,
              skeletonBallJoints->getGroupScaleDim())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtGroupScales(
                        jointsForSkeletonBallJoints);
        },
        // Generate a random restart position
        [&skeletonBallJoints, &skeleton, &observedJoints](
            Eigen::VectorXs& val) {
          val.segment(0, skeletonBallJoints->getNumDofs())
              = skeleton->convertPositionsToBallSpace(
                  skeleton->getRandomPoseForJoints(observedJoints));
          val.segment(
                 skeletonBallJoints->getNumDofs(),
                 skeletonBallJoints->getGroupScaleDim())
              .setConstant(1.0);
        },
        math::IKConfig()
            .setMaxStepCount(150)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(fitter->mInitialIKSatisfactoryLoss)
            .setMaxRestarts(fitter->mInitialIKMaxRestarts)
            .setLogOutput(false));
  }

  // 3. Return the result from the best fit we had

  return std::make_pair<Eigen::VectorXs, Eigen::VectorXs>(
      skeleton->getPositions(), skeleton->getGroupScales());
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
    Eigen::Ref<Eigen::MatrixXs> result)
{
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
  for (auto joint : fitter->mObservedJoints)
  {
    observedJoints.push_back(skeleton->getJoint(joint->getName()));
  }

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

  // 2. Run through each observation in sequence, and do a best fit
  for (int i = 0; i < markerObservations.size(); i++)
  {
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
    Eigen::VectorXs jointPoses = jointCenters[i];
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerVector;
    for (std::pair<std::string, Eigen::Vector3s> pair : markerObservations[i])
    {
      markerPoses.segment<3>(markerVector.size() * 3) = pair.second;
      if (markerWeights.count(pair.first))
      {
        markerWeightsVector(markerVector.size()) = markerWeights.at(pair.first);
      }
      const std::pair<dynamics::BodyNode*, Eigen::Vector3s>& originalMarker
          = fitter->mMarkerMap.at(pair.first);
      Eigen::Vector3s offset = Eigen::Vector3s::Zero();
      if (markerOffsets.count(pair.first))
      {
        offset = markerOffsets.at(pair.first);
      }
      markerVector.emplace_back(
          skeletonBallJoints->getBodyNode(originalMarker.first->getName()),
          originalMarker.second + offset);
    }

    // 2.2. Actually run the IK solver
    // Initialize at the old config

    math::solveIK(
        initialGuess,
        (markerVector.size() * 3) + (joints.size() * 3),
        // Set positions
        [&skeletonBallJoints, &skeleton](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(pos);

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            skeleton->setPositions(
                skeleton->convertPositionsFromBallSpace(pos));
            // 2. Clamp the position to limits
            skeleton->clampPositionsToLimits();
            // 3. Map the position back into SO3 space
            skeletonBallJoints->setPositions(
                skeleton->convertPositionsToBallSpace(
                    skeleton->getPositions()));
          }

          // Return the clamped position
          return skeletonBallJoints->getPositions();
        },
        [&skeletonBallJoints,
         &markerPoses,
         &markerVector,
         &markerWeightsVector,
         &jointsForSkeletonBallJoints,
         &jointPoses,
         &jointWeights](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff.segment(0, markerPoses.size())
              = markerPoses
                - skeletonBallJoints->getMarkerWorldPositions(markerVector);
          for (int i = 0; i < markerWeightsVector.size(); i++)
          {
            diff.segment<3>(i * 3) *= markerWeightsVector(i);
          }
          diff.segment(markerPoses.size(), jointPoses.size())
              = jointPoses
                - skeletonBallJoints->getJointWorldPositions(
                    jointsForSkeletonBallJoints);
          for (int i = 0; i < jointWeights.size(); i++)
          {
            diff.segment<3>(markerPoses.size() + (i * 3)) *= jointWeights(i);
          }

          assert(jac.cols() == skeletonBallJoints->getNumDofs());
          assert(
              jac.rows()
              == (markerVector.size() * 3)
                     + (jointsForSkeletonBallJoints.size() * 3));
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
        },
        [&skeleton, &observedJoints](Eigen::VectorXs& val) {
          val = skeleton->convertPositionsToBallSpace(
              skeleton->getRandomPoseForJoints(observedJoints));
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

    // 2.4. Set up for the next iteration, by setting the initial guess to the
    // current solve
    initialGuess = skeletonBallJoints->getPositions();
  }
}

//==============================================================================
/// This solves a bunch of optimization problems, one per joint, to find and
/// track the joint centers over time. It puts the results back into
/// `initialization`
void MarkerFitter::findJointCenters(
    MarkerInitialization& initialization,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // 1. Figure out which joints to find centers for
  initialization.joints.clear();
  for (int i = 0; i < mSkeleton->getNumJoints(); i++)
  {
    if (SphereFitJointCenterProblem::canFitJoint(this, mSkeleton->getJoint(i)))
    {
      initialization.joints.push_back(mSkeleton->getJoint(i));
    }
  }
  initialization.jointCenters = Eigen::MatrixXs::Zero(
      initialization.joints.size() * 3, markerObservations.size());

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
            initialization.jointCenters.block(
                i * 3, 0, 3, markerObservations.size()));

    futures.push_back(std::async(
        [this, problemPtr] { return this->findJointCenter(problemPtr); }));
  }
  for (int i = 0; i < futures.size(); i++)
  {
    futures[i].get()->saveSolutionBackToInitialization();
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
  Eigen::VectorXs x = problem->flatten();
  s_t loss = problem->getLoss();
  s_t initialLoss = loss;
  for (int i = 0; i < 20000; i++)
  {
    Eigen::VectorXs grad = problem->getGradient();
    Eigen::VectorXs newX = x - grad * lr;
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
  std::cout << "Sphere-fitting"
            << ": initial loss=" << initialLoss << ", final loss=" << loss
            << std::endl;
  return problemPtr;
}

//==============================================================================
/// This finds the trajectory for a single specified joint center over time
void MarkerFitter::findJointCenterLBFGS(
    int joint,
    MarkerInitialization& initialization,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  // Create an instance of the IpoptApplication
  //
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", static_cast<double>(1e-5));
  app->Options()->SetStringValue(
      "linear_solver",
      "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps, custom

  app->Options()->SetStringValue(
      "hessian_approximation", "limited-memory"); // limited-memory, exacty

  /*
  app->Options()->SetStringValue(
      "scaling_method", "none"); // none, gradient-based
  */

  app->Options()->SetIntegerValue("max_iter", 300);

  // Disable LBFGS history
  app->Options()->SetIntegerValue("limited_memory_max_history", 12);

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

  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded)
  {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return;
  }

  // This will automatically free the problem object when finished,
  // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will crash.
  // If you try to leave `problem` on the stack, you'll get invalid free
  // exceptions when IPOpt attempts to free it.
  SphereFitJointCenterProblem* problem = new SphereFitJointCenterProblem(
      this,
      markerObservations,
      initialization.poses,
      initialization.joints[joint],
      initialization.jointCenters.block(
          joint * 3, 0, 3, markerObservations.size()));
  SmartPtr<SphereFitJointCenterProblem> problemPtr(problem);
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
  for (int i = 0; i < getNumMarkers(); i++)
  {
    std::string markerName = getMarkerNameAtIndex(i);
    char lastChar = markerName[markerName.size() - 1];
    if (lastChar == '1' || lastChar == '2' || lastChar == '3')
    {
      setMarkerIsTracking(markerName);
    }
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
// The SphereFitJointCenterProblem, which maps the sphere-fitting joint-center
// problem onto a format that IPOpt can work with.
///////////////////////////////////////////////////////////////////////////////////////////////////////

//==============================================================================
SphereFitJointCenterProblem::SphereFitJointCenterProblem(
    MarkerFitter* fitter,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    Eigen::MatrixXs ikPoses,
    dynamics::Joint* joint,
    Eigen::Ref<Eigen::MatrixXs> out)
  : mFitter(fitter),
    mMarkerObservations(markerObservations),
    mIkPoses(ikPoses),
    mJoint(joint),
    mOut(out),
    mSmoothingLoss(
        0.1) // just to tie break when there's nothing better available
{
  // 1. Figure out which markers are on BodyNode's adjacent to the joint

  for (auto pair : fitter->mMarkerMap)
  {
    if (pair.second.first->getName() == joint->getParentBodyNode()->getName()
        || pair.second.first->getName() == joint->getChildBodyNode()->getName())
    {
      mActiveMarkers.push_back(pair.first);
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

  mNumTimesteps = markerObservations.size();

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

  // 3. Work out the thread splits

  int numThreads = 4;
  int sizeOfThreadSplits = mNumTimesteps / numThreads;
  int lastThreadSplit = mNumTimesteps - (sizeOfThreadSplits * (numThreads - 1));

  int cursor = 0;
  for (int i = 0; i < numThreads; i++)
  {
    int sizeOfThread
        = (i == numThreads - 1) ? lastThreadSplit : sizeOfThreadSplits;

    mThreadSplits.emplace_back(cursor, cursor + sizeOfThread);
    cursor += sizeOfThread;
  }
  assert(cursor == mNumTimesteps);
}

//==============================================================================
bool SphereFitJointCenterProblem::canFitJoint(
    MarkerFitter* fitter, dynamics::Joint* joint)
{
  int numActive = 0;
  for (auto pair : fitter->mMarkerMap)
  {
    if (joint->getParentBodyNode()
        && (pair.second.first->getName()
                == joint->getParentBodyNode()->getName()
            || pair.second.first->getName()
                   == joint->getChildBodyNode()->getName()))
    {
      numActive++;
    }
  }
  return numActive >= 3;
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

  /*
  std::vector<std::future<s_t>> futures;
  for (int i = 0; i < mThreadSplits.size(); i++)
  {
    int threadNum = i;
    futures.push_back(std::async([this, threadNum] {
      s_t loss = 0.0;

      int threadStart = this->mThreadSplits[threadNum].first;
      int threadEndExclusive = this->mThreadSplits[threadNum].second;
      for (int i = threadStart; i < threadEndExclusive; i++)
      {
        if (i > 0)
        {
          loss += mSmoothingLoss
                  * (mCenterPoints.segment<3>(i * 3)
                     - mCenterPoints.segment<3>((i - 1) * 3))
                        .squaredNorm();
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
    }));
  }

  for (int i = 0; i < futures.size(); i++)
  {
    loss += futures[i].get();
  }
  */

  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
    {
      loss += mSmoothingLoss
              * (mCenterPoints.segment<3>(i * 3)
                 - mCenterPoints.segment<3>((i - 1) * 3))
                    .squaredNorm();
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

  /*
  std::vector<std::future<Eigen::VectorXs>> futures;
  for (int i = 0; i < mThreadSplits.size(); i++)
  {
    int threadNum = i;
    futures.push_back(std::async([this, threadNum] {
      Eigen::VectorXs grad
          = Eigen::VectorXs::Zero(mRadii.size() + mCenterPoints.size());

      int threadStart = this->mThreadSplits[threadNum].first;
      int threadEndExclusive = this->mThreadSplits[threadNum].second;
      for (int i = threadStart; i < threadEndExclusive; i++)
      {
        if (i > 0)
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
    }));
  }

  for (int i = 0; i < futures.size(); i++)
  {
    grad += futures[i].get();
  }
  */

  for (int i = 0; i < mNumTimesteps; i++)
  {
    if (i > 0)
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

  const s_t EPS = 1e-7;
  for (int i = 0; i < x.size(); i++)
  {
    Eigen::VectorXs perturbed = x;
    perturbed(i) += EPS;
    unflatten(perturbed);
    s_t plus = getLoss();
    perturbed = x;
    perturbed(i) -= EPS;
    unflatten(perturbed);
    s_t minus = getLoss();

    grad(i) = (plus - minus) / (2 * EPS);
  }
  unflatten(x);

  return grad;
}

//------------------------- Ipopt::TNLP --------------------------------------
//==============================================================================
/// \brief Method to return some info about the nlp
bool SphereFitJointCenterProblem::get_nlp_info(
    Ipopt::Index& n,
    Ipopt::Index& m,
    Ipopt::Index& nnz_jac_g,
    Ipopt::Index& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  // Set the number of decision variables
  n = getProblemDim();

  // Set the total number of constraints
  m = 0;

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = 0;

  // Set the number of entries in the Hessian
  nnz_h_lag = n * n;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

//==============================================================================
/// \brief Method to return the bounds for my problem
bool SphereFitJointCenterProblem::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  (void)m;
  (void)g_l;
  (void)g_u;
  // Lower and upper bounds on X
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  upperBounds.setConstant(std::numeric_limits<double>::infinity());
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  lowerBounds.setConstant(-1 * std::numeric_limits<double>::infinity());
  return true;
}

//==============================================================================
/// \brief Method to return the starting point for the algorithm
bool SphereFitJointCenterProblem::get_starting_point(
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
    x = flatten();
  }

  return true;
}

//==============================================================================
/// \brief Method to return the objective value
bool SphereFitJointCenterProblem::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  if (_new_x)
  {
    Eigen::Map<const Eigen::VectorXd> x(_x, _n);
    unflatten(x);
  }
  _obj_value = getLoss();
  return true;
}

//==============================================================================
/// \brief Method to return the gradient of the objective
bool SphereFitJointCenterProblem::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  if (_new_x)
  {
    Eigen::Map<const Eigen::VectorXd> x(_x, _n);
    unflatten(x);
  }
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);
  grad = getGradient();
  return true;
}

//==============================================================================
/// \brief Method to return the constraint residuals
bool SphereFitJointCenterProblem::eval_g(
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
bool SphereFitJointCenterProblem::eval_jac_g(
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
bool SphereFitJointCenterProblem::eval_h(
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

void SphereFitJointCenterProblem::saveSolutionBackToInitialization()
{
  for (int i = 0; i < mNumTimesteps; i++)
  {
    mOut.col(i) = mCenterPoints.segment<3>(i * 3);
  }
}

//==============================================================================
/// \brief This method is called when the algorithm is complete so the TNLP
///        can store/write the solution
void SphereFitJointCenterProblem::finalize_solution(
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

  Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  unflatten(x);

  saveSolutionBackToInitialization();
}

//==============================================================================
bool SphereFitJointCenterProblem::intermediate_callback(
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
  return true;
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
    std::shared_ptr<BilevelFitResult>& outResult)
  : mFitter(fitter), mOutResult(outResult), mInitialization(initialization)
{
  // 1. Select the random indices we'll be using for this problem
  if (numSamples >= markerObservations.size())
  {
    numSamples = markerObservations.size();
    for (int i = 0; i < numSamples; i++)
    {
      mSampleIndices.push_back(i);
    }
  }
  else
  {
    double stride = (double)markerObservations.size() / numSamples;

    for (int i = 0; i < numSamples; i++)
    {
      int index = (int)round(stride * ((double)i + 0.5));
      if (index > markerObservations.size() - 1)
        index = markerObservations.size() - 1;
      if (index < 0)
        index = 0;

      mSampleIndices.push_back(index);
    }
  }

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
  // 2. Select the observations from the randomly chosen indices we'll be using
  // for this problem
  for (int i : mSampleIndices)
  {
    auto observation = markerObservations[i];
    mJointCenters.col(mMarkerMapObservations.size())
        = initialization.jointCenters.col(i);
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
  return scaleGroupDims + markerOffsetDims + poseDims;
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
      groupScaleDim + markerOffsetDim + (mSampleIndices.size() * dofs));

  // Copy group scales
  init.segment(0, groupScaleDim) = mInitialization.groupScales;

  // Copy marker offsets
  init.segment(groupScaleDim, markerOffsetDim).setZero();
  for (int i = 0; i < mFitter->mMarkerNames.size(); i++)
  {
    if (mInitialization.markerOffsets.count(mFitter->mMarkerNames[i]))
    {
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

  return init;
}

//==============================================================================
/// This evaluates our loss function given a concatenated vector of all the
/// problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
s_t BilevelFitProblem::getLoss(Eigen::VectorXs x)
{
  MarkerFitterState state(
      x,
      mMarkerMapObservations,
      mInitialization.joints,
      mJointCenters,
      mFitter);
  return mFitter->mLossAndGrad(&state);
}

//==============================================================================
/// This evaluates our gradient of loss given a concatenated vector of all
/// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
Eigen::VectorXs BilevelFitProblem::getGradient(Eigen::VectorXs x)
{
  MarkerFitterState state(
      x,
      mMarkerMapObservations,
      mInitialization.joints,
      mJointCenters,
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

  if (mFitter->mZeroConstraints.size() > 0)
  {
    MarkerFitterState state(
        x,
        mMarkerMapObservations,
        mInitialization.joints,
        mJointCenters,
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
          std::vector<int> sparsityMap
              = mFitter->getSparsityMap(threadMarkers, mMarkerObservations[i]);

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
              0, scaleGroupDims, threadSkeleton->getNumDofs(), markerOffsetDims)
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
          0, scaleGroupDims, mFitter->mSkeleton->getNumDofs(), markerOffsetDims)
          += mFitter->getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
                 mFitter->mSkeleton, markers, markerError, sparsityMap)
             * mObservationWeights(i);
    }
  }

  if (mFitter->mZeroConstraints.size() > 0)
  {
    MarkerFitterState state(
        x,
        mMarkerMapObservations,
        mInitialization.joints,
        mJointCenters,
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
      + (mFitter->mSkeleton->getNumDofs() * mMarkerObservations.size());

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
  // Lower and upper bounds on X
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  upperBounds.setConstant(std::numeric_limits<double>::infinity());
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  lowerBounds.setConstant(-1 * std::numeric_limits<double>::infinity());

  int scaleGroupDim = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDim = mFitter->mMarkers.size() * 3;
  int dofs = mFitter->mSkeleton->getNumDofs();

  assert(
      n
      == scaleGroupDim + markerOffsetDim + (mMarkerObservations.size() * dofs));

  upperBounds.segment(0, scaleGroupDim)
      = mFitter->mSkeleton->getGroupScalesUpperBound();
  lowerBounds.segment(0, scaleGroupDim)
      = mFitter->mSkeleton->getGroupScalesLowerBound();
  upperBounds.segment(scaleGroupDim, markerOffsetDim)
      .setConstant(mFitter->mMaxMarkerOffset);
  lowerBounds.segment(scaleGroupDim, markerOffsetDim)
      .setConstant(-mFitter->mMaxMarkerOffset);
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    upperBounds.segment(scaleGroupDim + markerOffsetDim + (i * dofs), dofs)
        = mFitter->mSkeleton->getPositionUpperLimits();
    lowerBounds.segment(scaleGroupDim + markerOffsetDim + (i * dofs), dofs)
        = mFitter->mSkeleton->getPositionLowerLimits();
  }

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
    x = getInitialization();
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

  _obj_value = getLoss(x);

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

  grad = getGradient(x);

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

  g = getConstraints(x);

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

    Eigen::MatrixXs jac = getConstraintsJacobian(x);

    int cursor = 0;
    for (int col = 0; col < jac.cols(); col++)
    {
      for (int row = 0; row < jac.rows(); row++)
      {
        vals(cursor) = jac(row, col);
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
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);

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
  return true;
}

} // namespace biomechanics
} // namespace dart