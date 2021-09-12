#include "dart/biomechanics/MarkerFitter.hpp"

namespace dart {

namespace biomechanics {

MarkerFitter::MarkerFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers)
  : mSkeleton(skeleton), mMarkers(markers)
{
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
            &(*mMarkers[i].first),
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
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(
      markers.size() * 3, skeleton->getNumScaleGroups() * 3);

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
      skeleton->getNumDofs(), skeleton->getNumScaleGroups() * 3);

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

} // namespace biomechanics
} // namespace dart