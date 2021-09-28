#include "dart/biomechanics/MarkerFitter.hpp"

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

namespace dart {

namespace biomechanics {

using namespace Ipopt;

MarkerFitResult::MarkerFitResult() : success(false){};

MarkerFitter::MarkerFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton, dynamics::MarkerMap markers)
  : mSkeleton(skeleton),
    mTolerance(1e-8),
    mIterationLimit(500),
    mLBFGSHistoryLength(15),
    mCheckDerivatives(false),
    mPrintFrequency(1),
    mSilenceOutput(false),
    mDisableLinesearch(false),
    mInitialIKSatisfactoryLoss(0.003),
    mInitialIKMaxRestarts(100),
    mMaxMarkerOffset(0.1)
{
  mSkeletonBallJoints = mSkeleton->convertSkeletonToBallJoints();
  int offset = 0;
  for (auto pair : markers)
  {
    mMarkerIndices[pair.first] = offset;
    mMarkerNames.push_back(pair.first);
    offset++;
    mMarkers.push_back(pair.second);
    mMarkersBallJoints.emplace_back(
        mSkeletonBallJoints->getBodyNode(pair.second.first->getName()),
        Eigen::Vector3s(pair.second.second));
  }
}

//==============================================================================
/// This solves an optimization problem, trying to get the Skeleton to match
/// the markers as closely as possible.
std::shared_ptr<MarkerFitResult> MarkerFitter::optimize(
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

  std::shared_ptr<MarkerFitResult> result = std::make_shared<MarkerFitResult>();

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
  BilevelFitProblem* problem
      = new BilevelFitProblem(this, markerObservations, result);
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

  return result;
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
/// Internally all the markers are concatenated together, so each index has a
/// name.
std::string MarkerFitter::getMarkerNameAtIndex(int index)
{
  return mMarkerNames[index];
}

//==============================================================================
/// This method will set `skeleton` to the configuration given by the vectors
/// of jointPositions and groupScales. It will also compute and return the
/// list of markers given by markerDiffs.
std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
MarkerFitter::setConfiguration(
    std::shared_ptr<dynamics::Skeleton>& skeleton,
    Eigen::VectorXs jointPositions,
    Eigen::VectorXs groupScales,
    Eigen::VectorXs markerDiffs)
{
  skeleton->setPositions(jointPositions);
  skeleton->setGroupScales(groupScales);
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      adjustedMarkers;
  for (int i = 0; i < mMarkers.size(); i++)
  {
    adjustedMarkers.push_back(
        std::make_pair<const dynamics::BodyNode*, Eigen::Vector3s>(
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
s_t MarkerFitter::computeLoss(Eigen::VectorXs markerError)
{
  return markerError.squaredNorm();
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
/// This gets the gradient of the objective wrt the joint positions
Eigen::VectorXs MarkerFitter::getLossGradientWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    Eigen::VectorXs markerError)
{
  return 2
         * skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers)
               .transpose()
         * markerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the joint positions
Eigen::VectorXs MarkerFitter::finiteDifferenceLossGradientWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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

    s_t plus = computeLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalPos;
    perturbed(i) -= EPS;
    skeleton->setPositions(perturbed);

    s_t minus = computeLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    grad(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setPositions(originalPos);

  return grad;
}

//==============================================================================
/// This gets the gradient of the objective wrt the group scales
Eigen::VectorXs MarkerFitter::getLossGradientWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    Eigen::VectorXs markerError)
{
  return 2
         * skeleton->getMarkerWorldPositionsJacobianWrtGroupScales(markers)
               .transpose()
         * markerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the group scales
Eigen::VectorXs MarkerFitter::finiteDifferenceLossGradientWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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

    s_t plus = computeLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalGroupScales;
    perturbed(i) -= EPS;
    skeleton->setGroupScales(perturbed);

    s_t minus = computeLoss(
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    grad(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setGroupScales(originalGroupScales);

  return grad;
}

//==============================================================================
/// This gets the gradient of the objective wrt the marker offsets
Eigen::VectorXs MarkerFitter::getLossGradientWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    Eigen::VectorXs markerError)
{
  return 2
         * skeleton->getMarkerWorldPositionsJacobianWrtMarkerOffsets(markers)
               .transpose()
         * markerError;
}

//==============================================================================
/// This gets the gradient of the objective wrt the marker offsets
Eigen::VectorXs MarkerFitter::finiteDifferenceLossGradientWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(
        std::make_pair<const dynamics::BodyNode*, Eigen::Vector3s>(
            &(*markers[i].first), Eigen::Vector3s(markers[i].second)));
  }

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      markersCopy[i].second(axis) = markers[i].second(axis) + EPS;

      s_t plus = computeLoss(
          getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses));

      markersCopy[i].second(axis) = markers[i].second(axis) - EPS;

      s_t minus = computeLoss(
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtJointPositions(markers);

  // Clear out the sections of the Jacobian that were not observed, since those
  // won't change the error
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
Eigen::MatrixXs MarkerFitter::getLossGradientWrtJointsJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
MarkerFitter::finiteDifferenceLossGradientWrtJointsJacobianWrtJoints(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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

    Eigen::VectorXs plus = getLossGradientWrtJoints(
        skeleton,
        markers,
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalPos;
    perturbed(i) -= EPS;
    skeleton->setPositions(perturbed);

    Eigen::VectorXs minus = getLossGradientWrtJoints(
        skeleton,
        markers,
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setPositions(originalPos);

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the group scales
Eigen::MatrixXs MarkerFitter::getMarkerErrorJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtGroupScales(markers);

  // Clear out the sections of the Jacobian that were not observed, since those
  // won't change the error
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
Eigen::MatrixXs MarkerFitter::getLossGradientWrtJointsJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
MarkerFitter::finiteDifferenceLossGradientWrtJointsJacobianWrtGroupScales(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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

    Eigen::VectorXs plus = getLossGradientWrtJoints(
        skeleton,
        markers,
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    perturbed = originalGroupScales;
    perturbed(i) -= EPS;
    skeleton->setGroupScales(perturbed);

    Eigen::VectorXs minus = getLossGradientWrtJoints(
        skeleton,
        markers,
        getMarkerError(skeleton, markers, visibleMarkerWorldPoses));

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  skeleton->setGroupScales(originalGroupScales);

  return jac;
}

//==============================================================================
/// This gets the jacobian of the marker error wrt the marker offsets
Eigen::MatrixXs MarkerFitter::getMarkerErrorJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<int>& sparsityMap)
{
  Eigen::MatrixXs jac
      = skeleton->getMarkerWorldPositionsJacobianWrtMarkerOffsets(markers);

  // Clear out the sections of the Jacobian that were not observed, since those
  // won't change the error
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(markers.size() * 3, markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(
        std::make_pair<const dynamics::BodyNode*, Eigen::Vector3s>(
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
Eigen::MatrixXs MarkerFitter::getLossGradientWrtJointsJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
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
MarkerFitter::finiteDifferenceLossGradientWrtJointsJacobianWrtMarkerOffsets(
    const std::shared_ptr<dynamics::Skeleton>& skeleton,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& visibleMarkerWorldPoses)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(skeleton->getNumDofs(), markers.size() * 3);

  const s_t EPS = 1e-7;

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      markersCopy;
  for (int i = 0; i < markers.size(); i++)
  {
    markersCopy.push_back(
        std::make_pair<const dynamics::BodyNode*, Eigen::Vector3s>(
            &(*markers[i].first), Eigen::Vector3s(markers[i].second)));
  }

  for (int i = 0; i < markers.size(); i++)
  {
    for (int axis = 0; axis < 3; axis++)
    {
      markersCopy[i].second(axis) = markers[i].second(axis) + EPS;

      Eigen::VectorXs plus = getLossGradientWrtJoints(
          skeleton,
          markersCopy,
          getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses));

      markersCopy[i].second(axis) = markers[i].second(axis) - EPS;

      Eigen::VectorXs minus = getLossGradientWrtJoints(
          skeleton,
          markersCopy,
          getMarkerError(skeleton, markersCopy, visibleMarkerWorldPoses));

      markersCopy[i].second(axis) = markers[i].second(axis);

      jac.col(i * 3 + axis) = (plus - minus) / (2 * EPS);
    }
  }

  return jac;
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
    std::shared_ptr<MarkerFitResult>& outResult)
  : mFitter(fitter), mOutResult(outResult), mInitializationCached(false)
{
  // Translate the markers into the format expected by the problem
  for (auto observation : markerObservations)
  {
    std::vector<std::pair<int, Eigen::Vector3s>> translated;
    for (auto pair : observation)
    {
      translated.emplace_back(
          mFitter->mMarkerIndices[pair.first], Eigen::Vector3s(pair.second));
    }
    mMarkerObservations.push_back(translated);
  }

  mObservationWeights = Eigen::VectorXs::Ones(mMarkerObservations.size());
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
  int poseDims = mFitter->mSkeleton->getNumDofs() * mMarkerObservations.size();
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
  if (mInitializationCached)
  {
    return mCachedInitialization;
  }
  int groupScaleDim = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDim = mFitter->mMarkers.size() * 3;
  int dofs = mFitter->mSkeleton->getNumDofs();

  Eigen::VectorXs init = Eigen::VectorXs::Zero(
      groupScaleDim + markerOffsetDim + (mMarkerObservations.size() * dofs));

  // Initialize with a zero marker offset
  init.segment(groupScaleDim, markerOffsetDim).setZero();

  Eigen::VectorXs scalesAvg
      = Eigen::VectorXs::Zero(mFitter->mSkeletonBallJoints->getGroupScaleDim());
  s_t individualLossSum = 0.0;
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    // Translate observations to markers
    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
        observedMarkers;
    Eigen::VectorXs markerPoses
        = Eigen::VectorXs::Zero(mMarkerObservations[i].size() * 3);
    for (int j = 0; j < mMarkerObservations[i].size(); j++)
    {
      auto pair = mMarkerObservations[i][j];
      observedMarkers.push_back(mFitter->mMarkersBallJoints[pair.first]);
      markerPoses.segment<3>(j * 3) = pair.second;
    }

    // Because we have no initialization, we should do the slow thing and try
    // really hard to fit the IK well

    // We're going to enforce the joint limits in the Eulerian space, but do our
    // actual gradient descient in SO3 space so we can avoid gimbal lock. That
    // requires a bit of careful book-keeping.

    int problemDim = mFitter->mSkeletonBallJoints->getNumDofs()
                     + mFitter->mSkeletonBallJoints->getGroupScaleDim();
    Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);

    // Set our initial guess for IK to whatever the current pose of the skeleton
    // is

    initialPos.segment(0, mFitter->mSkeletonBallJoints->getNumDofs())
        = mFitter->mSkeleton->convertPositionsToBallSpace(
            mFitter->mSkeleton->getPositions());
    initialPos.segment(
        mFitter->mSkeletonBallJoints->getNumDofs(),
        mFitter->mSkeletonBallJoints->getGroupScaleDim())
        = mFitter->mSkeletonBallJoints->getGroupScales();

    s_t loss = math::solveIK(
        initialPos,
        observedMarkers.size() * 3,
        [this, observedMarkers](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          mFitter->mSkeletonBallJoints->setPositions(
              pos.segment(0, mFitter->mSkeletonBallJoints->getNumDofs()));

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            mFitter->mSkeleton->setPositions(
                mFitter->mSkeleton->convertPositionsFromBallSpace(pos.segment(
                    0, mFitter->mSkeletonBallJoints->getNumDofs())));
            // 2. Clamp the position to limits
            mFitter->mSkeleton->clampPositionsToLimits();
            // 1. Map the position back into SO3 space
            mFitter->mSkeletonBallJoints->setPositions(
                mFitter->mSkeleton->convertPositionsToBallSpace(
                    mFitter->mSkeleton->getPositions()));
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              mFitter->mSkeletonBallJoints->getNumDofs(),
              mFitter->mSkeletonBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = mFitter->mSkeletonBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = mFitter->mSkeletonBallJoints->getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          mFitter->mSkeleton->setGroupScales(newScales);
          mFitter->mSkeletonBallJoints->setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, mFitter->mSkeletonBallJoints->getNumDofs())
              = mFitter->mSkeletonBallJoints->getPositions();
          clampedPos.segment(
              mFitter->mSkeletonBallJoints->getNumDofs(),
              mFitter->mSkeletonBallJoints->getGroupScaleDim())
              = newScales;
          return clampedPos;
        },
        [this, observedMarkers, markerPoses](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = markerPoses
                 - mFitter->mSkeletonBallJoints->getMarkerWorldPositions(
                     observedMarkers);
          assert(
              jac.cols()
              == mFitter->mSkeletonBallJoints->getNumDofs()
                     + mFitter->mSkeletonBallJoints->getGroupScaleDim());
          assert(jac.rows() == observedMarkers.size() * 3);
          jac.setZero();
          jac.block(
              0,
              0,
              observedMarkers.size() * 3,
              mFitter->mSkeletonBallJoints->getNumDofs())
              = mFitter->mSkeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        observedMarkers);
          jac.block(
              0,
              mFitter->mSkeletonBallJoints->getNumDofs(),
              observedMarkers.size() * 3,
              mFitter->mSkeletonBallJoints->getGroupScaleDim())
              = mFitter->mSkeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtGroupScales(
                        observedMarkers);
        },
        [this](Eigen::VectorXs& val) {
          val.segment(0, mFitter->mSkeletonBallJoints->getNumDofs())
              = mFitter->mSkeleton->convertPositionsToBallSpace(
                  mFitter->mSkeleton->getRandomPose());
          val.segment(
                 mFitter->mSkeletonBallJoints->getNumDofs(),
                 mFitter->mSkeletonBallJoints->getGroupScaleDim())
              .setConstant(1.0);
        },
        math::IKConfig()
            .setMaxStepCount(150)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(mFitter->mInitialIKSatisfactoryLoss)
            .setMaxRestarts(mFitter->mInitialIKMaxRestarts)
            .setLogOutput(false));

    if (loss > mFitter->mInitialIKSatisfactoryLoss)
    {
      mObservationWeights(i) = 0.0;
      std::cout << "Observation Loss [" << i
                << "] (with scaling) TOO HIGH, will be excluded from fit: "
                << loss << std::endl;
    }
    else
    {
      std::cout << "Observation Loss [" << i << "] (with scaling): " << loss
                << std::endl;
      individualLossSum += loss;
    }

    init.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs)
        = mFitter->mSkeleton->convertPositionsFromBallSpace(
            mFitter->mSkeletonBallJoints->getPositions());
    scalesAvg += mFitter->mSkeletonBallJoints->getGroupScales();
  }
  scalesAvg /= mMarkerObservations.size();
  mFitter->mSkeletonBallJoints->setGroupScales(scalesAvg);
  mFitter->mSkeleton->setGroupScales(scalesAvg);

  // Take the average scale we found, and use that as the initial guess
  init.segment(0, groupScaleDim) = scalesAvg;

  s_t noScaleLossSum = 0.0;
  // Go through and fit the markers as best we can, while holding scales
  // constant
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    // Translate observations to markers
    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
        observedMarkers;
    Eigen::VectorXs markerPoses
        = Eigen::VectorXs::Zero(mMarkerObservations[i].size() * 3);
    for (int j = 0; j < mMarkerObservations[i].size(); j++)
    {
      auto pair = mMarkerObservations[i][j];
      observedMarkers.push_back(mFitter->mMarkersBallJoints[pair.first]);
      markerPoses.segment<3>(j * 3) = pair.second;
    }

    // Initialize at the old config
    mFitter->mSkeletonBallJoints->setPositions(
        mFitter->mSkeleton->convertPositionsToBallSpace(
            init.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs)));

    s_t loss = math::solveIK(
        mFitter->mSkeletonBallJoints->getPositions(),
        observedMarkers.size() * 3,
        [this, observedMarkers](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          mFitter->mSkeletonBallJoints->setPositions(pos);

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            mFitter->mSkeleton->setPositions(
                mFitter->mSkeleton->convertPositionsFromBallSpace(pos));
            // 2. Clamp the position to limits
            mFitter->mSkeleton->clampPositionsToLimits();
            // 1. Map the position back into SO3 space
            mFitter->mSkeletonBallJoints->setPositions(
                mFitter->mSkeleton->convertPositionsToBallSpace(
                    mFitter->mSkeleton->getPositions()));
          }

          // Return the clamped position
          return mFitter->mSkeletonBallJoints->getPositions();
        },
        [this, observedMarkers, markerPoses](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = markerPoses
                 - mFitter->mSkeletonBallJoints->getMarkerWorldPositions(
                     observedMarkers);
          assert(jac.cols() == mFitter->mSkeletonBallJoints->getNumDofs());
          assert(jac.rows() == observedMarkers.size() * 3);
          jac = mFitter->mSkeletonBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        observedMarkers);
        },
        [this](Eigen::VectorXs& val) {
          val = mFitter->mSkeleton->convertPositionsToBallSpace(
              mFitter->mSkeleton->getRandomPose());
        },
        math::IKConfig()
            .setMaxStepCount(500)
            .setConvergenceThreshold(1e-10)
            .setDontExitTranspose(true)
            .setLossLowerBound(1e-8)
            .setMaxRestarts(1)
            .setStartClamped(true)
            .setLogOutput(false));
    noScaleLossSum += loss * mObservationWeights(i);

    init.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs)
        = mFitter->mSkeleton->convertPositionsFromBallSpace(
            mFitter->mSkeletonBallJoints->getPositions());
  }

  std::cout << "Total sum of individual losses (with scaling): "
            << individualLossSum << std::endl;
  std::cout << "Total sum of individual losses (holding scale at avg): "
            << noScaleLossSum << std::endl;

  // Try to recreate this manually
  // TODO: <remove>

  s_t manualError = 0.0;
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    // Translate observations to markers
    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
        observedMarkers;
    Eigen::VectorXs markerPoses
        = Eigen::VectorXs::Zero(mMarkerObservations[i].size() * 3);
    for (int j = 0; j < mMarkerObservations[i].size(); j++)
    {
      auto pair = mMarkerObservations[i][j];
      observedMarkers.push_back(mFitter->mMarkers[pair.first]);
      markerPoses.segment<3>(j * 3) = pair.second;
    }

    mFitter->mSkeleton->setPositions(
        init.segment(groupScaleDim + markerOffsetDim + (i * dofs), dofs));
    Eigen::VectorXs markerWorldPoses
        = mFitter->mSkeleton->getMarkerWorldPositions(observedMarkers);
    manualError += (markerPoses - markerWorldPoses).squaredNorm()
                   * mObservationWeights(i);
  }
  std::cout << "Manually calculated total loss: " << manualError << std::endl;

  // TODO: </remove>

  s_t problemLoss = getLoss(init);
  std::cout << "Got total loss: " << problemLoss << std::endl;

  mCachedInitialization = init;
  mInitializationCached = true;

  return init;
}

//==============================================================================
/// This evaluates our loss function given a concatenated vector of all the
/// problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
s_t BilevelFitProblem::getLoss(Eigen::VectorXs x)
{
  int scaleGroupDims = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDims = mFitter->mMarkers.size() * 3;
  Eigen::VectorXs groupScales = x.segment(0, scaleGroupDims);
  Eigen::VectorXs markerOffsets = x.segment(scaleGroupDims, markerOffsetDims);
  Eigen::VectorXs firstPose = x.segment(
      scaleGroupDims + markerOffsetDims, mFitter->mSkeleton->getNumDofs());

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  s_t lossSum = 0.0;
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    Eigen::VectorXs pose = x.segment(
        scaleGroupDims + markerOffsetDims
            + (i * mFitter->mSkeleton->getNumDofs()),
        mFitter->mSkeleton->getNumDofs());
    mFitter->mSkeleton->setPositions(pose);
    lossSum += mFitter->computeLoss(mFitter->getMarkerError(
                   mFitter->mSkeleton, markers, mMarkerObservations[i]))
               * mObservationWeights(i);
  }
  return lossSum;
}

//==============================================================================
/// This evaluates our gradient of loss given a concatenated vector of all
/// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
Eigen::VectorXs BilevelFitProblem::getGradient(Eigen::VectorXs x)
{
  int scaleGroupDims = mFitter->mSkeleton->getGroupScaleDim();
  int markerOffsetDims = mFitter->mMarkers.size() * 3;
  Eigen::VectorXs groupScales = x.segment(0, scaleGroupDims);
  Eigen::VectorXs markerOffsets = x.segment(scaleGroupDims, markerOffsetDims);
  Eigen::VectorXs firstPose = x.segment(
      scaleGroupDims + markerOffsetDims, mFitter->mSkeleton->getNumDofs());

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  Eigen::VectorXs grad = Eigen::VectorXs::Zero(x.size());
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    int offset = scaleGroupDims + markerOffsetDims
                 + (i * mFitter->mSkeleton->getNumDofs());
    Eigen::VectorXs pose = x.segment(offset, mFitter->mSkeleton->getNumDofs());
    mFitter->mSkeleton->setPositions(pose);
    Eigen::VectorXs markerError = mFitter->getMarkerError(
        mFitter->mSkeleton, markers, mMarkerObservations[i]);

    // Get loss wrt joint positions
    grad.segment(offset, mFitter->mSkeleton->getNumDofs())
        = mFitter->getLossGradientWrtJoints(
              mFitter->mSkeleton, markers, markerError)
          * mObservationWeights(i);

    // Acculumulate loss wrt the global scale groups
    grad.segment(0, scaleGroupDims)
        += mFitter->getLossGradientWrtGroupScales(
               mFitter->mSkeleton, markers, markerError)
           * mObservationWeights(i);
    // Acculumulate loss wrt the global marker offsets
    grad.segment(scaleGroupDims, markerOffsetDims)
        += mFitter->getLossGradientWrtMarkerOffsets(
               mFitter->mSkeleton, markers, markerError)
           * mObservationWeights(i);
  }
  return grad;
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

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  Eigen::VectorXs ikGrad
      = Eigen::VectorXs::Zero(mFitter->mSkeleton->getNumDofs());
  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    int offset = scaleGroupDims + markerOffsetDims
                 + (i * mFitter->mSkeleton->getNumDofs());
    Eigen::VectorXs pose = x.segment(offset, mFitter->mSkeleton->getNumDofs());
    mFitter->mSkeleton->setPositions(pose);

    // Get loss wrt joint positions
    ikGrad += mFitter->getLossGradientWrtJoints(
                  mFitter->mSkeleton,
                  markers,
                  mFitter->getMarkerError(
                      mFitter->mSkeleton, markers, mMarkerObservations[i]))
              * mObservationWeights(i);
  }

  return ikGrad;
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

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
      = mFitter->setConfiguration(
          mFitter->mSkeleton, firstPose, groupScales, markerOffsets);

  for (int i = 0; i < mMarkerObservations.size(); i++)
  {
    int offset = scaleGroupDims + markerOffsetDims
                 + (i * mFitter->mSkeleton->getNumDofs());
    Eigen::VectorXs pose = x.segment(offset, mFitter->mSkeleton->getNumDofs());
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
        = mFitter->getLossGradientWrtJointsJacobianWrtJoints(
              mFitter->mSkeleton, markers, markerError, sparsityMap)
          * mObservationWeights(i);

    // Acculumulate loss wrt the global scale groups
    jac.block(0, 0, mFitter->mSkeleton->getNumDofs(), scaleGroupDims)
        += mFitter->getLossGradientWrtJointsJacobianWrtGroupScales(
               mFitter->mSkeleton, markers, markerError, sparsityMap)
           * mObservationWeights(i);
    // Acculumulate loss wrt the global marker offsets
    jac.block(
        0, scaleGroupDims, mFitter->mSkeleton->getNumDofs(), markerOffsetDims)
        += mFitter->getLossGradientWrtJointsJacobianWrtMarkerOffsets(
               mFitter->mSkeleton, markers, markerError, sparsityMap)
           * mObservationWeights(i);
  }

  return jac;
}

//==============================================================================
/// This evaluates the Jacobian of our constraint vector wrt x given a
/// concatenated vector of all the problem state: [groupSizes, markerOffsets,
/// q_0, ..., q_N]
Eigen::MatrixXs BilevelFitProblem::finiteDifferenceConstraintsJacobian(
    Eigen::VectorXs x)
{
  Eigen::MatrixXs jac
      = Eigen::MatrixXs::Zero(mFitter->mSkeleton->getNumDofs(), x.size());
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
  m = mFitter->mSkeleton->getNumDofs();

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
  // fill in the sparsity structure of the Jacobian (the row and column indices
  // only). At this time, the x argument and the values argument will be
  // nullptr.

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
    mOutResult->markerErrors[mFitter->mMarkerNames[i]]
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