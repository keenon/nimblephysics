#include "dart/biomechanics/IKInitializer.hpp"

#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
/// This is a helper struct that is used to simplify the code in
/// estimatePosesAndGroupScales()
typedef struct IKWeldedBodyGroup
{
  std::vector<dynamics::BodyNode*> bodies;
  std::vector<Eigen::Isometry3s> relativeTransforms;
  std::vector<std::string> adjacentJoints;
  std::vector<Eigen::Vector3s> adjacentJointCenters;
  std::vector<std::string> adjacentMarkers;
  std::vector<Eigen::Vector3s> adjacentMarkerCenters;
} IKWeldedBodyGroup;

//==============================================================================
/// This returns true if the given body is the parent of the joint OR if
/// there's a hierarchy of fixed joints that connect it to the parent
bool isDynamicParentOfJoint(std::string bodyName, dynamics::Joint* joint)
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
bool isDynamicChildOfJoint(std::string bodyName, dynamics::Joint* joint)
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
bool isCoplanar(const std::vector<Eigen::Vector3s>& points)
{
  // Ensure there are at least 4 points
  if (points.size() < 4)
  {
    return true;
  }

  // Choose the first three points and form two vectors
  const Eigen::Vector3s& p1 = points[0];
  const Eigen::Vector3s& p2 = points[1];
  const Eigen::Vector3s& p3 = points[2];
  const Eigen::Vector3s v1 = p2 - p1;
  const Eigen::Vector3s v2 = p3 - p1;

  // Calculate the cross-product of the two vectors to get a normal vector
  const Eigen::Vector3s normal = v1.cross(v2).normalized();

  // Check if all remaining points are coplanar with the first three points
  for (size_t i = 3; i < points.size(); ++i)
  {
    // Form a vector from one of the original three points to the current point
    const Eigen::Vector3s& point = points[i];
    const Eigen::Vector3s v = point - p1;

    // Calculate the dot product of the vector with the normal vector
    const double dot_product = v.dot(normal);

    // Check if the dot product is greater zero, and if so we have found a
    // non-coplanar point
    const s_t threshold = 1e-3;
    if (std::abs(dot_product) > threshold)
    {
      return false;
    }
  }

  return true;
}

//==============================================================================
Eigen::Vector3s ensureOnSameSideOfPlane(
    const std::vector<Eigen::Vector3s>& neutralPoints,
    Eigen::Vector3s neutralGoal,
    std::vector<Eigen::Vector3s> actualPoints,
    Eigen::Vector3s ambiguousReconstructionFromActualPoints)
{
  if (neutralPoints.size() < 3 || actualPoints.size() < 3)
  {
    return ambiguousReconstructionFromActualPoints;
  }

  // Choose the first three points and form two vectors
  const Eigen::Vector3s neutralNormal
      = ((neutralPoints[1] - neutralPoints[0])
             .cross(neutralPoints[2] - neutralPoints[0]))
            .normalized();
  s_t neutralGoalDistanceFromNormal
      = (neutralGoal - neutralPoints[0]).dot(neutralNormal);

  const Eigen::Vector3s actualNormal
      = ((actualPoints[1] - actualPoints[0])
             .cross(actualPoints[2] - actualPoints[0]))
            .normalized();
  s_t ambiguousReconstructionDistanceFromNormal
      = (ambiguousReconstructionFromActualPoints - actualPoints[0])
            .dot(actualNormal);

  if (neutralGoalDistanceFromNormal * ambiguousReconstructionDistanceFromNormal
      < 0)
  {
    // If they're on opposite sides of the plane, flip the `actualGoal` to be on
    // the same side of the plane as the `neutralGoal`.
    return ambiguousReconstructionFromActualPoints
           - 2 * ambiguousReconstructionDistanceFromNormal * actualNormal;
  }
  else
  {
    // If they're already on the same side of the plane, don't worry
    return ambiguousReconstructionFromActualPoints;
  }
}

//==============================================================================
IKInitializer::IKInitializer(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        markers,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    s_t modelHeightM)
  : mSkel(skel),
    mMarkerObservations(markerObservations),
    mModelHeightM(modelHeightM)
{
  // 1. Convert the marker map to an ordered list
  for (auto& pair : markers)
  {
    mMarkerNames.push_back(pair.first);
    mMarkers.push_back(pair.second);
  }

  Eigen::VectorXs oldPositions = mSkel->getPositions();
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
  Eigen::VectorXs oldScales = mSkel->getBodyScales();
  if (modelHeightM > 0)
  {
    mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
    s_t defaultHeight
        = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
    if (defaultHeight > 0)
    {
      s_t ratio = modelHeightM / defaultHeight;
      mSkel->setBodyScales(mSkel->getBodyScales() * ratio);
      s_t newHeight
          = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
      (void)newHeight;
      assert(abs(newHeight - modelHeightM) < 1e-6);
    }
  }

  // 2. Find all the joints that are connected to at least three markers and
  // measure the distances between that joint center and the adjacent markers
  Eigen::VectorXs jointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs markerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    dynamics::Joint* joint = mSkel->getJoint(i);

    Eigen::Vector3s jointWorldCenter = jointWorldPositions.segment<3>(i * 3);

    std::map<std::string, s_t> jointToMarkerSquaredDistances;
    for (int j = 0; j < mMarkers.size(); j++)
    {
      if (isDynamicChildOfJoint(mMarkers[j].first->getName(), joint)
          || isDynamicParentOfJoint(mMarkers[j].first->getName(), joint))
      {
        Eigen::Vector3s markerWorldCenter
            = markerWorldPositions.segment<3>(j * 3);

        if ((markerWorldCenter - jointWorldCenter).squaredNorm() > 0)
        {
          jointToMarkerSquaredDistances[mMarkerNames[j]]
              = (markerWorldCenter - jointWorldCenter).squaredNorm();
          assert(jointToMarkerSquaredDistances[mMarkerNames[j]] >= 0);
        }
      }
    }

    if (jointToMarkerSquaredDistances.size() >= 3)
    {
      mJoints.push_back(mSkel->getJoint(i));
      mJointToMarkerSquaredDistances[joint->getName()]
          = jointToMarkerSquaredDistances;
    }
  }

  // 3. See if there are any connected pairs of joints, and if so measure the
  // distance between them on the skeleton.
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    dynamics::Joint* joint1 = mSkel->getJoint(i);
    Eigen::Vector3s joint1WorldCenter = jointWorldPositions.segment<3>(i * 3);
    if (std::find(mJoints.begin(), mJoints.end(), joint1) == mJoints.end())
    {
      continue;
    }
    for (int j = i + 1; j < mSkel->getNumJoints(); j++)
    {
      dynamics::Joint* joint2 = mSkel->getJoint(j);
      if (std::find(mJoints.begin(), mJoints.end(), joint2) == mJoints.end())
      {
        continue;
      }
      Eigen::Vector3s joint2WorldCenter = jointWorldPositions.segment<3>(j * 3);

      // 3.1. If the joints are connected by a body (or a series of bodies
      // connected with fixed joints), then record them
      if ((joint1->getParentBodyNode() != nullptr
           && isDynamicChildOfJoint(
               joint1->getParentBodyNode()->getName(), joint2))
          || (joint2->getParentBodyNode() != nullptr
              && isDynamicChildOfJoint(
                  joint2->getParentBodyNode()->getName(), joint1)))
      {
        Eigen::Vector3s joint1ToJoint2 = joint2WorldCenter - joint1WorldCenter;
        s_t joint1ToJoint2SquaredDistance = joint1ToJoint2.squaredNorm();
        assert(joint1ToJoint2SquaredDistance > 0);

        mJointToJointSquaredDistances[joint1->getName()][joint2->getName()]
            = joint1ToJoint2SquaredDistance;
        mJointToJointSquaredDistances[joint2->getName()][joint1->getName()]
            = joint1ToJoint2SquaredDistance;
      }
    }
  }

  mSkel->setPositions(oldPositions);
  mSkel->setBodyScales(oldScales);
}

//==============================================================================
/// This runs the full IK initialization algorithm, and leaves the answers in
/// the public fields of this class
void IKInitializer::runFullPipeline(bool logOutput)
{
  // Use the pivot finding, where there is the huge wealth of marker information
  // (3+ markers on adjacent body segments) to make it possible
  closedFormPivotFindingJointCenterSolver();
  reestimateDistancesFromJointCenters();
  // Use the MDS solver, where there is less information (only 1-2 markers on
  // adjacent body segments) to make it possible
  closedFormMDSJointCenterSolver(logOutput);
  // Fill in the parts of the body scales and poses that we can in closed form
  estimatePosesAndGroupScalesInClosedForm();

  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    if (logOutput || t % 20 == 0)
    {
      std::cout << "Completing IK for " << t << "/"
                << mMarkerObservations.size() << " timesteps" << std::endl;
    }
    completeIKIteratively(t, mSkel);
    fineTuneIKIteratively(t, mSkel);
  }
}

//==============================================================================
/// For each timestep, and then for each joint, this sets up and runs an MDS
/// algorithm to triangulate the joint center location from the known marker
/// locations of the markers attached to the joint's two body segments, and
/// then known distance from the joint center to each marker.
s_t IKInitializer::closedFormMDSJointCenterSolver(bool logOutput)
{
  // 0. Save the default pose and scale's version of the joint centers and
  // marker locations, so that we can use it to detect and resolve coplanar
  // ambiguity in subsequent steps.
  Eigen::VectorXs oldPositions = mSkel->getPositions();
  Eigen::VectorXs oldScales = mSkel->getBodyScales();
  Eigen::VectorXs neutralSkelJointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs neutralSkelMarkerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  mSkel->setPositions(oldPositions);
  mSkel->setBodyScales(oldScales);
  std::map<std::string, Eigen::Vector3s>
      neutralSkelJointCenterWorldPositionsMap;
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    neutralSkelJointCenterWorldPositionsMap[mSkel->getJoint(i)->getName()]
        = neutralSkelJointWorldPositions.segment<3>(i * 3);
  }
  std::map<std::string, Eigen::Vector3s> neutralSkelMarkerWorldPositionsMap;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    neutralSkelMarkerWorldPositionsMap[mMarkerNames[i]]
        = neutralSkelMarkerWorldPositions.segment<3>(i * 3);
  }

  s_t totalMarkerError = 0.0;
  int count = 0;
  mJointCenters.clear();
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    std::vector<dynamics::Joint*> joints = getVisibleJoints(t);
    std::map<std::string, Eigen::Vector3s> lastSolvedJointCenters;
    std::map<std::string, Eigen::Vector3s> solvedJointCenters;
    while (true)
    {
      for (dynamics::Joint* joint : joints)
      {
        // If we already solved this joint, no need to go again
        if (lastSolvedJointCenters.count(joint->getName()))
          continue;

        std::vector<Eigen::Vector3s> adjacentPointLocations;
        std::vector<s_t> adjacentPointSquaredDistances;
        // These are the locations of each of the adjacent points, but on the
        // unscaled skeleton in the neutral pose, which we use to detect and
        // resolve coplanar ambiguity during reconstruction.
        std::vector<Eigen::Vector3s> adjacentPointLocationsInNeutralSkel;

        // 1. Find the markers that are adjacent to this joint and visible on
        // this frame
        for (auto& pair : mJointToMarkerSquaredDistances[joint->getName()])
        {
          if (mMarkerObservations[t].count(pair.first))
          {
            adjacentPointLocations.push_back(
                mMarkerObservations[t][pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
            adjacentPointLocationsInNeutralSkel.push_back(
                neutralSkelMarkerWorldPositionsMap[pair.first]);
          }
        }

        for (auto& pair : mJointToJointSquaredDistances[joint->getName()])
        {
          if (lastSolvedJointCenters.count(pair.first))
          {
            adjacentPointLocations.push_back(
                lastSolvedJointCenters[pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
            adjacentPointLocationsInNeutralSkel.push_back(
                neutralSkelJointCenterWorldPositionsMap[pair.first]);
          }
        }

        if (adjacentPointLocations.size() < 3)
          continue;

        // 2. Setup the squared-distance matrix
        int dim = adjacentPointLocations.size() + 1;
        Eigen::MatrixXs D = Eigen::MatrixXs::Zero(dim, dim);
        for (int i = 0; i < adjacentPointLocations.size(); i++)
        {
          for (int j = i + 1; j < adjacentPointLocations.size(); j++)
          {
            D(i, j) = (adjacentPointLocations[i] - adjacentPointLocations[j])
                          .squaredNorm();
            // The distance matrix is symmetric
            D(j, i) = D(i, j);
          }
        }
        for (int i = 0; i < adjacentPointLocations.size(); i++)
        {
          D(i, dim - 1) = adjacentPointSquaredDistances[i];
          D(dim - 1, i) = D(i, dim - 1);
        }

        // 3. Solve the distance matrix
        Eigen::MatrixXs pointCloud = getPointCloudFromDistanceMatrix(D);
        assert(!pointCloud.hasNaN());
        Eigen::MatrixXs transformed
            = mapPointCloudToData(pointCloud, adjacentPointLocations);

        // 4. Collect error statistics
        s_t pointCloudError = 0.0;
        for (int j = 0; j < adjacentPointLocations.size(); j++)
        {
          pointCloudError
              += (adjacentPointLocations[j] - transformed.col(j)).norm();
        }
        pointCloudError /= adjacentPointLocations.size();
        if (logOutput)
        {
          std::cout << "Joint center " << joint->getName()
                    << " point cloud reconstruction error: " << pointCloudError
                    << "m" << std::endl;
        }
        totalMarkerError += pointCloudError;
        count++;

        // 5. If the point cloud is co-planar (or very close to it), then
        // there's ambiguity about which side of the plane to place the joint
        // center. We'll check co-planarity on the neutral skeleton also, to
        // avoid having noise in the real marker data fool us into thinking
        // there isn't a coplanar ambiguity. Then we can also use the neutral
        // skeleton to resolve which side of the plane to place the joint
        // center.
        Eigen::Vector3s jointCenter = transformed.col(dim - 1);
        if (isCoplanar(adjacentPointLocationsInNeutralSkel)
            || isCoplanar(adjacentPointLocations))
        {
          if (logOutput)
          {
            std::cout << "Joint " << joint->getName()
                      << " has coplanar support, and is therefore ambiguous! "
                         "Resolving ambiguity."
                      << std::endl;
          }
          // If the point cloud is coplanar, then we need to resolve the
          // ambiguity about which side of the plane to place the joint center.
          // We do this by checking which side of the plane the joint center is
          // on in the neutral pose, and ensuring that the joint center is on
          // the same side of the plane in the reconstructed pose.
          jointCenter = ensureOnSameSideOfPlane(
              adjacentPointLocationsInNeutralSkel,
              neutralSkelJointCenterWorldPositionsMap[joint->getName()],
              adjacentPointLocations,
              jointCenter);
        }

        solvedJointCenters[joint->getName()] = jointCenter;
      }
      if (solvedJointCenters.size() == lastSolvedJointCenters.size())
        break;
      lastSolvedJointCenters = solvedJointCenters;
    }
    mJointCenters.push_back(lastSolvedJointCenters);
  }
  return totalMarkerError / count;
}

//==============================================================================
/// This first finds an approximate rigid body trajectory for each body
/// that has at least 3 markers on it, and then sets up and solves a linear
/// system of equations for each joint to find the pair of offsets in the
/// adjacent body nodes that results in the least offset between the joint
/// centers.
s_t IKInitializer::closedFormPivotFindingJointCenterSolver(bool logOutput)
{
  // 1. Find all the bodies with at least 3 markers on them
  std::map<std::string, int> bodyMarkerCounts;
  for (auto& marker : mMarkers)
  {
    if (bodyMarkerCounts.count(marker.first->getName()) == 0)
      bodyMarkerCounts[marker.first->getName()] = 0;
    bodyMarkerCounts[marker.first->getName()]++;
  }

  // 2. Find all the joints with bodies on both sides that have 3 markers on
  // them, and keep track of all adjacent bodies.
  std::vector<dynamics::Joint*> jointsToSolve;
  std::vector<std::string> bodiesAdjacentToJoints;
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    dynamics::Joint* joint = mSkel->getJoint(i);
    if (joint->getParentBodyNode() == nullptr)
      continue;
    if (bodyMarkerCounts[joint->getParentBodyNode()->getName()] >= 3
        && bodyMarkerCounts[joint->getChildBodyNode()->getName()] >= 3)
    {
      jointsToSolve.push_back(joint);
      // Keep track of adjacent bodies, so we can solve for their positions
      if (std::find(
              bodiesAdjacentToJoints.begin(),
              bodiesAdjacentToJoints.end(),
              joint->getParentBodyNode()->getName())
          == bodiesAdjacentToJoints.end())
      {
        bodiesAdjacentToJoints.push_back(joint->getParentBodyNode()->getName());
      }
      if (std::find(
              bodiesAdjacentToJoints.begin(),
              bodiesAdjacentToJoints.end(),
              joint->getChildBodyNode()->getName())
          == bodiesAdjacentToJoints.end())
      {
        bodiesAdjacentToJoints.push_back(joint->getChildBodyNode()->getName());
      }
    }
  }

  (void)logOutput;
  // 3. Find the approximate rigid body trajectory for each body. The original
  // transform doesn't matter, since we're just looking for the relative
  // transform between the two bodies. So, arbitrarily, we choose the first
  // timestep where 3 markers are observed as the identity transform.
  std::map<std::string, std::map<int, Eigen::Isometry3s>> bodyTrajectories;
  for (std::string bodyName : bodiesAdjacentToJoints)
  {
    // 3.1. Collect the names of the markers we're attached to
    std::vector<std::string> attachedMarkers;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      auto marker = mMarkers[i];
      if (marker.first->getName() == bodyName)
      {
        attachedMarkers.push_back(mMarkerNames[i]);
      }
    }
    // Go through all the timesteps, solving for relative transforms on the
    // timesteps that we can.
    std::map<int, Eigen::Isometry3s> bodyTrajectory;
    std::vector<std::string> visibleMarkersCloud;
    std::vector<Eigen::Vector3s> visibleMarkerCloudIdentityTransform;
    bool foundFirstFrame = false;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      // 3.2. We first search through all the frames to find the first frame
      // where there are at least 3 markers visible on the body. Call this the
      // identity transform.
      if (!foundFirstFrame)
      {
        visibleMarkersCloud.clear();
        visibleMarkerCloudIdentityTransform.clear();
        for (std::string marker : attachedMarkers)
        {
          if (mMarkerObservations[t].count(marker))
          {
            visibleMarkersCloud.push_back(marker);
            visibleMarkerCloudIdentityTransform.push_back(
                mMarkerObservations[t][marker]);
          }
        }
        if (visibleMarkersCloud.size() >= 3)
        {
          foundFirstFrame = true;
          bodyTrajectory[t] = Eigen::Isometry3s::Identity();
        }
      }
      // 3.3. Once we have the identity transform, we can solve for the relative
      // transform of subsequent frames, if enough markers are visible.
      else
      {
        std::vector<Eigen::Vector3s> identityMarkerCloud;
        std::vector<Eigen::Vector3s> currentMarkerCloud;
        std::vector<s_t> weights;
        for (int i = 0; i < visibleMarkersCloud.size(); i++)
        {
          std::string marker = visibleMarkersCloud[i];
          if (mMarkerObservations[t].count(marker))
          {
            identityMarkerCloud.push_back(
                visibleMarkerCloudIdentityTransform[i]);
            currentMarkerCloud.push_back(mMarkerObservations[t][marker]);
            weights.push_back(1.0);
          }
        }

        // 3.4. If we have enough markers, we can solve for the relative
        // transform at this frame
        if (identityMarkerCloud.size() >= 3)
        {
          bodyTrajectory[t] = getPointCloudToPointCloudTransform(
              identityMarkerCloud, currentMarkerCloud, weights);
        }
      }
    }
    bodyTrajectories[bodyName] = bodyTrajectory;
  }

  // 4. Now we can solve for the relative transforms of each joint, by setting
  // up a linear system of equations where the unknowns are the relative
  // transforms from each body to the joint center, and the constraints are that
  // those map to the same location in world space on each timestep.
  s_t avgJointCenterError = 0.0;
  for (dynamics::Joint* joint : jointsToSolve)
  {
    // At this point, the parent body node shouldn't be null
    assert(joint->getParentBodyNode() != nullptr);
    std::string parentBodyName = joint->getParentBodyNode()->getName();
    std::string childBodyName = joint->getChildBodyNode()->getName();

    // 4.1. Collect all the usable timesteps where transforms of parent and
    // child are both known
    std::vector<int> usableTimesteps;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      if (bodyTrajectories[parentBodyName].count(t)
          && bodyTrajectories[childBodyName].count(t))
      {
        usableTimesteps.push_back(t);
      }
    }

    // 4.2. Subsample timesteps if necessary to keep problem size reasonable
    const int maxSamples = 500;
    if (usableTimesteps.size() > maxSamples)
    {
      std::vector<int> sampledIndices
          = math::evenlySpacedTimesteps(usableTimesteps.size(), maxSamples);
      std::vector<int> sampledTimesteps;
      for (int index : sampledIndices)
      {
        sampledTimesteps.push_back(usableTimesteps[index]);
      }
      usableTimesteps = sampledTimesteps;
    }

    // 4.3. Setup the linear system of equations
    int cols = 6;
    int rows = usableTimesteps.size() * 3;

    Eigen::MatrixXs A = Eigen::MatrixXs::Zero(rows, cols);
    Eigen::VectorXs b = Eigen::VectorXs::Zero(rows);
    for (int i = 0; i < usableTimesteps.size(); i++)
    {
      Eigen::Isometry3s parentTransform
          = bodyTrajectories[parentBodyName][usableTimesteps[i]];
      Eigen::Isometry3s childTransform
          = bodyTrajectories[childBodyName][usableTimesteps[i]];
      A.block<3, 3>(i * 3, 0) = parentTransform.linear();
      A.block<3, 3>(i * 3, 3) = -1 * childTransform.linear();
      b.segment<3>(i * 3)
          = parentTransform.translation() - childTransform.translation();
    }

    Eigen::VectorXs target = Eigen::VectorXs::Zero(rows);
    Eigen::VectorXs offsets = A.householderQr().solve(target - b);
    assert(offsets.size() == 6);

    // 4.4. Decode the results of our solution into average values
    Eigen::Vector3s parentOffset = offsets.segment<3>(0);
    Eigen::Vector3s childOffset = offsets.segment<3>(3);

    // Ensure that we've got enough space in our joint centers vector
    while (mJointCenters.size() < mMarkerObservations.size())
    {
      mJointCenters.push_back(std::map<std::string, Eigen::Vector3s>());
    }

    s_t error = 0.0;
    for (int i = 0; i < usableTimesteps.size(); i++)
    {
      Eigen::Isometry3s parentTransform = bodyTrajectories[parentBodyName][i];
      Eigen::Isometry3s childTransform = bodyTrajectories[childBodyName][i];
      Eigen::Vector3s parentWorldCenter = parentTransform * parentOffset;
      Eigen::Vector3s childWorldCenter = childTransform * childOffset;
      Eigen::Vector3s jointCenter
          = (parentWorldCenter + childWorldCenter) / 2.0;
      error += (parentWorldCenter - jointCenter).norm();
      error += (childWorldCenter - jointCenter).norm();

      // 4.4.1. Record the result
      mJointCenters[usableTimesteps[i]][joint->getName()] = jointCenter;
    }

    error /= usableTimesteps.size() * 2;
    avgJointCenterError += error;
  }
  avgJointCenterError /= jointsToSolve.size();

  return avgJointCenterError;
}

//==============================================================================
/// This uses the current guesses for the joint centers to re-estimate the
/// bone sizes (based on distance between joint centers) and then use that to
/// get the group scale vector. This also uses the joint centers to estimate
/// the body positions.
s_t IKInitializer::estimatePosesAndGroupScalesInClosedForm(bool log)
{
  Eigen::VectorXs originalPose = mSkel->getPositions();
  Eigen::VectorXs originalBodyScales = mSkel->getBodyScales();
  mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));

  Eigen::VectorXs jointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());

  // 1. Estimate the bone sizes from the joint centers

  // 1.1. Get estimated joint to joint distances from our current joint center
  // guesses
  std::map<std::string, std::map<std::string, s_t>>
      estimatedJointToJointDistances = estimateJointToJointDistances();
  // 1.2. Find a scale for all the bodies that we can
  std::map<std::string, Eigen::Vector3s> bodyScales;
  std::map<std::string, Eigen::Vector3s> bodyScaleWeights;
  for (dynamics::BodyNode* bodyNode : mSkel->getBodyNodes())
  {
    // 1.2.1. Collect joints adjacent to this body that we can potentially use
    // to help scale
    std::vector<dynamics::Joint*> adjacentJoints;
    adjacentJoints.push_back(bodyNode->getParentJoint());
    for (int i = 0; i < bodyNode->getNumChildJoints(); i++)
    {
      adjacentJoints.push_back(bodyNode->getChildJoint(i));
    }

    // 1.2.2. Find which pairs of joints (expressed as locations in the local
    // frame of the body), if any, have estimated distances between them
    std::vector<std::tuple<std::string, std::string, s_t>> jointPairs;
    std::map<std::string, Eigen::Vector3s> jointsInLocalSpace;
    for (int i = 0; i < adjacentJoints.size(); i++)
    {
      for (int j = i + 1; j < adjacentJoints.size(); j++)
      {
        dynamics::Joint* joint1 = adjacentJoints[i];
        dynamics::Joint* joint2 = adjacentJoints[j];
        if (estimatedJointToJointDistances.count(joint1->getName()) > 0
            && estimatedJointToJointDistances[joint1->getName()].count(
                   joint2->getName())
                   > 0)
        {
          s_t dist = estimatedJointToJointDistances[joint1->getName()]
                                                   [joint2->getName()];
          Eigen::Vector3s joint1WorldPos = jointWorldPositions.segment<3>(
              joint1->getJointIndexInSkeleton() * 3);
          Eigen::Vector3s joint2WorldPos = jointWorldPositions.segment<3>(
              joint2->getJointIndexInSkeleton() * 3);
          Eigen::Vector3s joint1LocalPos
              = bodyNode->getWorldTransform().inverse() * joint1WorldPos;
          jointsInLocalSpace[joint1->getName()] = joint1LocalPos;
          Eigen::Vector3s joint2LocalPos
              = bodyNode->getWorldTransform().inverse() * joint2WorldPos;
          jointsInLocalSpace[joint2->getName()] = joint2LocalPos;
          jointPairs.emplace_back(joint1->getName(), joint2->getName(), dist);
        }
      }
    }

    // 1.2.3. If we have a single joint pair, then we can use that to scale the
    // body
    if (jointPairs.size() == 1)
    {
      std::string joint1Name = std::get<0>(jointPairs[0]);
      std::string joint2Name = std::get<1>(jointPairs[0]);
      s_t desiredDist = std::get<2>(jointPairs[0]);
      s_t currentDist
          = (jointsInLocalSpace[joint1Name] - jointsInLocalSpace[joint2Name])
                .norm();
      s_t scale = desiredDist / currentDist;
      if (log)
      {
        std::cout << "Scaling " << bodyNode->getName() << " by " << scale
                  << " based on " << joint1Name << " and " << joint2Name
                  << std::endl;
      }
      bodyScales[bodyNode->getName()] = Eigen::Vector3s(scale, scale, scale);
      bodyScaleWeights[bodyNode->getName()]
          = Eigen::Vector3s::Ones() * currentDist * currentDist;
    }
    // 1.2.4. If we have multiple joint pairs, then we can use those to scale
    // along multiple axis at once.
    else if (jointPairs.size() > 1)
    {
      assert(jointsInLocalSpace.size() > 0);
      std::map<std::string, int> jointToIndex;
      std::vector<Eigen::Vector3s> jointsInLocalSpaceVector;
      for (auto& pair : jointsInLocalSpace)
      {
        jointToIndex[pair.first] = jointToIndex.size();
        jointsInLocalSpaceVector.push_back(pair.second);
      }
      assert(jointsInLocalSpaceVector.size() > 0);

      // Default the square distances to the current distances, in case we're
      // missing some pairs
      Eigen::MatrixXs squaredDistances = Eigen::MatrixXs::Zero(
          jointsInLocalSpace.size(), jointsInLocalSpace.size());
      for (auto& pair1 : jointsInLocalSpace)
      {
        for (auto& pair2 : jointsInLocalSpace)
        {
          squaredDistances(jointToIndex[pair1.first], jointToIndex[pair2.first])
              = (pair1.second - pair2.second).squaredNorm();
        }
      }

      // Now overwrite the distances we have with the estimated distances
      for (auto& pair : jointPairs)
      {
        std::string joint1Name = std::get<0>(pair);
        std::string joint2Name = std::get<1>(pair);
        s_t desiredDist = std::get<2>(pair);
        squaredDistances(jointToIndex[joint1Name], jointToIndex[joint2Name])
            = desiredDist * desiredDist;
        squaredDistances(jointToIndex[joint2Name], jointToIndex[joint1Name])
            = desiredDist * desiredDist;
      }

      // Now we can run MDS to get the point cloud
      Eigen::MatrixXs rawPointCloud
          = getPointCloudFromDistanceMatrix(squaredDistances);

      // Now we can map the point cloud back to the original locations
      Eigen::MatrixXs pointCloud
          = mapPointCloudToData(rawPointCloud, jointsInLocalSpaceVector);

      // Compute axis scales as the average ratio of the columns of `pointCloud`
      // to `jointsInLocalSpaceVector`
      Eigen::Vector3s axisScales = Eigen::Vector3s::Zero();
      Eigen::Vector3s axisLengthSum = Eigen::Vector3s::Zero();
      for (int i = 0; i < jointsInLocalSpaceVector.size(); i++)
      {
        axisLengthSum += jointsInLocalSpaceVector[i].cwiseProduct(
            jointsInLocalSpaceVector[i]);
        for (int axis = 0; axis < 3; axis++)
        {
          if (jointsInLocalSpaceVector[i](axis) > 0)
          {
            axisScales(axis)
                += pointCloud(axis, i) / jointsInLocalSpaceVector[i](axis);
          }
          else
          {
            // We have info about this axis, so assume the scale is perfectly
            // fine
            axisScales(axis) += 1.0;
          }
        }
      }
      axisScales /= jointsInLocalSpaceVector.size();

      if (log)
      {
        std::cout << "Scaling " << bodyNode->getName() << " by "
                  << axisScales.transpose() << " based on " << jointPairs.size()
                  << " pairs of joints" << std::endl;
        for (int i = 0; i < jointsInLocalSpaceVector.size(); i++)
        {
          Eigen::Vector3s target = pointCloud.col(i);
          Eigen::Vector3s scaled
              = jointsInLocalSpaceVector[i].cwiseProduct(axisScales);
          std::cout << "  " << i
                    << " reconstruction error = " << (target - scaled).norm()
                    << ": target " << target.transpose() << " vs scaled "
                    << scaled.transpose() << std::endl;
        }
      }

      bodyScales[bodyNode->getName()] = axisScales;
      bodyScaleWeights[bodyNode->getName()] = axisLengthSum;
    }
  }
  // 1.3. Scale any remaining bodies to a default scale based on the height, if
  // known
  Eigen::Vector3s defaultScales = Eigen::Vector3s::Ones();
  Eigen::Vector3s defaultScaleWeights = Eigen::Vector3s::Ones() * 0.01;

  if (mModelHeightM > 0)
  {
    s_t defaultHeight
        = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
    if (defaultHeight > 0)
    {
      s_t ratio = mModelHeightM / defaultHeight;
      defaultScales = Eigen::Vector3s(ratio, ratio, ratio);
    }
  }

  for (dynamics::BodyNode* bodyNode : mSkel->getBodyNodes())
  {
    if (bodyScales.count(bodyNode->getName()) == 0)
    {
      bodyScales[bodyNode->getName()] = defaultScales;
      bodyScaleWeights[bodyNode->getName()] = defaultScaleWeights;
    }
  }
  // 1.4. Apply all the scalings to bodies, taking weighted averages over scale
  // groups
  for (auto& group : mSkel->getBodyScaleGroups())
  {
    Eigen::Vector3s scaleAvg = defaultScales.cwiseProduct(defaultScaleWeights);
    Eigen::Vector3s weightsSum = defaultScaleWeights;

    std::cout << "Scaling group:" << std::endl;
    for (auto* body : group.nodes)
    {
      // Prevent divide-by-zero errors
      Eigen::Vector3s weight = bodyScaleWeights[body->getName()];
      for (int axis = 0; axis < 3; axis++)
      {
        if (weight(axis) <= 0)
          weight(axis) = 1e-5;
      }

      std::cout << "  " << body->getName() << " = "
                << bodyScales[body->getName()].transpose() << " with weight "
                << weight.transpose() << std::endl;
      scaleAvg += bodyScales[body->getName()].cwiseProduct(weight);
      weightsSum += weight;
    }
    scaleAvg = scaleAvg.cwiseQuotient(weightsSum);
    std::cout << "  -> weighted avg = " << scaleAvg.transpose() << std::endl;
    for (auto* body : group.nodes)
    {
      body->setScale(scaleAvg);
    }
  }
  // 1.5. Ensure that the scaling is symmetric across groups, by condensing into
  // the group scales vector and then re-setting the body scales from that
  // vector.
  mGroupScales = mSkel->getGroupScales();
  mSkel->setGroupScales(mGroupScales);
  // Recompute the joint world positions
  jointWorldPositions = mSkel->getJointWorldPositions(mSkel->getJoints());

  // 2. We'll need to go by groups of bodies connected by weld joints in the
  // subsequent code, so compute and store those outside of the time loop.
  std::vector<IKWeldedBodyGroup> weldedGroups;
  for (auto* body : mSkel->getBodyNodes())
  {
    weldedGroups.emplace_back();
    weldedGroups[weldedGroups.size() - 1].bodies.push_back(body);
  }
  for (auto* joint : mSkel->getJoints())
  {
    if (joint->getNumDofs() == 0 && joint->getParentBodyNode() != nullptr)
    {
      int parentGroupIndex = -1;
      int childGroupIndex = -1;
      for (int i = 0; i < weldedGroups.size(); i++)
      {
        if (std::find(
                weldedGroups[i].bodies.begin(),
                weldedGroups[i].bodies.end(),
                joint->getParentBodyNode())
            != weldedGroups[i].bodies.end())
        {
          parentGroupIndex = i;
        }
        if (std::find(
                weldedGroups[i].bodies.begin(),
                weldedGroups[i].bodies.end(),
                joint->getChildBodyNode())
            != weldedGroups[i].bodies.end())
        {
          childGroupIndex = i;
        }
      }
      assert(parentGroupIndex != -1);
      assert(childGroupIndex != -1);
      if (parentGroupIndex != childGroupIndex)
      {
        weldedGroups[parentGroupIndex].bodies.insert(
            weldedGroups[parentGroupIndex].bodies.end(),
            weldedGroups[childGroupIndex].bodies.begin(),
            weldedGroups[childGroupIndex].bodies.end());
        weldedGroups.erase(weldedGroups.begin() + childGroupIndex);
      }
    }
  }
  // 2.1. Compute all the related information for each welded group, everything
  // in the frame of the first (root) body in the welded group. We compute the
  // transforms of any other bodies, the joint centers, and the marker centers.

  // 2.1.1. Compute body relative transforms
  for (int i = 0; i < weldedGroups.size(); i++)
  {
    for (int j = 0; j < weldedGroups[i].bodies.size(); j++)
    {
      weldedGroups[i].relativeTransforms.push_back(
          weldedGroups[i].bodies[0]->getWorldTransform().inverse()
          * weldedGroups[i].bodies[j]->getWorldTransform());
    }
  }
  // 2.1.2. Compute adjacent joints and their relative transforms
  for (auto* joint : mSkel->getJoints())
  {
    for (auto& weldedGroup : weldedGroups)
    {
      if (std::find(
              weldedGroup.bodies.begin(),
              weldedGroup.bodies.end(),
              joint->getParentBodyNode())
              != weldedGroup.bodies.end()
          || std::find(
                 weldedGroup.bodies.begin(),
                 weldedGroup.bodies.end(),
                 joint->getChildBodyNode())
                 != weldedGroup.bodies.end())
      {
        weldedGroup.adjacentJoints.push_back(joint->getName());
        Eigen::Vector3s jointCenter = jointWorldPositions.segment<3>(
            joint->getJointIndexInSkeleton() * 3);
        weldedGroup.adjacentJointCenters.push_back(
            weldedGroup.bodies[0]->getWorldTransform().inverse() * jointCenter);
      }
    }
  }
  // 2.1.3. Compute adjacent markers and their relative transforms
  Eigen::VectorXs markerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  for (int i = 0; i < mMarkers.size(); i++)
  {
    auto& marker = mMarkers[i];
    for (auto& weldedGroup : weldedGroups)
    {
      if (std::find(
              weldedGroup.bodies.begin(),
              weldedGroup.bodies.end(),
              marker.first)
          != weldedGroup.bodies.end())
      {
        weldedGroup.adjacentMarkers.push_back(mMarkerNames[i]);
        Eigen::Vector3s markerCenter = markerWorldPositions.segment<3>(i * 3);
        weldedGroup.adjacentMarkerCenters.push_back(
            weldedGroup.bodies[0]->getWorldTransform().inverse()
            * markerCenter);
      }
    }
  }

  // 3. Now we can go through each timestep and do "closed form IK", by first
  // estimating the body positions and then estimating the joint angles to
  // achieve those body positions.
  mPoses.clear();
  mBodyTransforms.clear();
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    // 3.1. Estimate the body positions in world space of each welded group from
    // the joint estimates and marker estimates, when they're available.
    std::map<std::string, Eigen::Isometry3s> estimatedBodyWorldTransforms;
    for (auto& weldGroup : weldedGroups)
    {
      // 3.1.1. Collect all the visible points in world space that we can use to
      // estimate the body transform
      std::vector<Eigen::Vector3s> visibleAdjacentPointsInWorldSpace;
      std::vector<std::string> visibleAdjacentPointNames;
      std::vector<Eigen::Vector3s> visibleAdjacentPointsInLocalSpace;
      std::vector<s_t> visibleAdjacentPointWeights;
      for (int j = 0; j < weldGroup.adjacentJoints.size(); j++)
      {
        if (mJointCenters[t].count(weldGroup.adjacentJoints[j]) > 0)
        {
          visibleAdjacentPointsInWorldSpace.push_back(
              mJointCenters[t][weldGroup.adjacentJoints[j]]);
          visibleAdjacentPointsInLocalSpace.push_back(
              weldGroup.adjacentJointCenters[j]);
          visibleAdjacentPointNames.push_back(
              "Joint " + weldGroup.adjacentJoints[j]);
          visibleAdjacentPointWeights.push_back(1.0);
        }
      }
      for (int j = 0; j < weldGroup.adjacentMarkers.size(); j++)
      {
        if (mMarkerObservations[t].count(weldGroup.adjacentMarkers[j]) > 0)
        {
          visibleAdjacentPointsInWorldSpace.push_back(
              mMarkerObservations[t][weldGroup.adjacentMarkers[j]]);
          visibleAdjacentPointsInLocalSpace.push_back(
              weldGroup.adjacentMarkerCenters[j]);
          visibleAdjacentPointNames.push_back(
              "Marker " + weldGroup.adjacentMarkers[j]);
          visibleAdjacentPointWeights.push_back(1.0);
        }
      }
      assert(
          visibleAdjacentPointsInLocalSpace.size()
          == visibleAdjacentPointsInWorldSpace.size());

      if (visibleAdjacentPointsInLocalSpace.size() >= 4)
      {
        // 3.1.2. Compute the root body transform from the visible points, and
        // then use that to compute the other body transforms
        Eigen::Isometry3s rootBodyWorldTransform
            = getPointCloudToPointCloudTransform(
                visibleAdjacentPointsInLocalSpace,
                visibleAdjacentPointsInWorldSpace,
                visibleAdjacentPointWeights);
        for (int j = 0; j < weldGroup.bodies.size(); j++)
        {
          estimatedBodyWorldTransforms[weldGroup.bodies[j]->getName()]
              = rootBodyWorldTransform * weldGroup.relativeTransforms[j];
        }

        s_t error = 0.0;
        for (int j = 0; j < visibleAdjacentPointsInLocalSpace.size(); j++)
        {
          Eigen::Vector3s reconstructedPoint
              = rootBodyWorldTransform * visibleAdjacentPointsInLocalSpace[j];
          error += (visibleAdjacentPointsInWorldSpace[j] - reconstructedPoint)
                       .norm();
        }
        error /= visibleAdjacentPointsInLocalSpace.size();

        if (log)
        {
          std::cout << "Estimated bodies ";
          for (int j = 0; j < weldGroup.bodies.size(); j++)
          {
            std::cout << "\"" << weldGroup.bodies[j]->getName() << "\" ";
          }
          std::cout << " at time " << t << " with error " << error << "m from "
                    << visibleAdjacentPointsInLocalSpace.size()
                    << " points: " << std::endl;
          for (int j = 0; j < visibleAdjacentPointsInLocalSpace.size(); j++)
          {
            std::cout << "  " << visibleAdjacentPointNames[j] << ": "
                      << visibleAdjacentPointsInWorldSpace[j].transpose()
                      << " reconstructed as "
                      << (rootBodyWorldTransform
                          * visibleAdjacentPointsInLocalSpace[j])
                             .transpose()
                      << std::endl;
          }
        }
      }
    }
    mBodyTransforms.push_back(estimatedBodyWorldTransforms);

    // 3.2. Now that we have the estimated body positions, we can estimate the
    // joint angles.
    Eigen::VectorXs jointAngles = Eigen::VectorXs::Zero(mSkel->getNumDofs());
    Eigen::VectorXi jointAnglesClosedFormEstimate
        = Eigen::VectorXi::Zero(mSkel->getNumDofs());
    for (auto* joint : mSkel->getJoints())
    {
      // Only estimate joint angles for joints connecting two bodies that we
      // have estimated locations for
      if ((joint->getParentBodyNode() == nullptr
           || estimatedBodyWorldTransforms.count(
               joint->getParentBodyNode()->getName()))
          && estimatedBodyWorldTransforms.count(
              joint->getChildBodyNode()->getName()))
      {
        // 3.2.1. Get the relative transformation we estimate is taking place
        // over the joint
        Eigen::Isometry3s parentTransform = Eigen::Isometry3s::Identity();
        if (joint->getParentBodyNode() != nullptr)
        {
          parentTransform
              = estimatedBodyWorldTransforms[joint->getParentBodyNode()
                                                 ->getName()];
        }
        Eigen::Isometry3s childTransform
            = estimatedBodyWorldTransforms[joint->getChildBodyNode()
                                               ->getName()];
        Eigen::Isometry3s jointTransform
            = parentTransform.inverse() * childTransform;

        // 3.2.2. Convert that estimated tranformation into joint coordinates
        Eigen::VectorXs pos = joint->getNearestPositionToDesiredRotation(
            jointTransform.linear());
        joint->setPositions(pos);
        Eigen::Isometry3s recoveredJointTransform
            = joint->getRelativeTransform();
        if (joint->getType() == dynamics::FreeJoint::getStaticType()
            || joint->getType() == dynamics::EulerFreeJoint::getStaticType())
        {
          Eigen::Vector3s translationOffset
              = (recoveredJointTransform.translation()
                 - jointTransform.translation());
          pos.tail<3>() -= translationOffset;
          joint->setPositions(pos);
          recoveredJointTransform = joint->getRelativeTransform();
        }

        if (log)
        {
          s_t translationError = (jointTransform.translation()
                                  - recoveredJointTransform.translation())
                                     .norm();
          s_t rotationError
              = (jointTransform.linear() - recoveredJointTransform.linear())
                    .norm();
          std::cout << "Estimating joint " << joint->getName()
                    << " from adjacent body transforms with error "
                    << translationError << "m and " << rotationError
                    << " on rotation" << std::endl;
        }

        // 3.2.3. Save our estimate back to our joint angles
        jointAngles.segment(joint->getIndexInSkeleton(0), joint->getNumDofs())
            = pos;
        jointAnglesClosedFormEstimate
            .segment(joint->getIndexInSkeleton(0), joint->getNumDofs())
            .setConstant(1.0);
      }
    }
    mPoses.push_back(jointAngles);
    mPosesClosedFormEstimateAvailable.push_back(jointAnglesClosedFormEstimate);
  }

  mSkel->setBodyScales(originalBodyScales);
  mSkel->setPositions(originalPose);

  // TODO: compute marker reconstruction error and return that
  return 0.0;
}

/// This solves the remaining DOFs that couldn't be found in closed form using
/// an iterative IK solver. This portion of the solver is the only non-convex
/// portion. It uses random-restarts, and so is not as unit-testable as the
/// other portions of the algorithm, so it should hopefully only impact less
/// important joints.
s_t IKInitializer::completeIKIteratively(
    int timestep, std::shared_ptr<dynamics::Skeleton> threadsafeSkel)
{
  Eigen::VectorXs analyticalPose = mPoses[timestep];
  Eigen::VectorXi analyticalPoseClosedFormEstimate
      = mPosesClosedFormEstimateAvailable[timestep];

  int numClosedForm = 0;
  for (int i = 0; i < analyticalPoseClosedFormEstimate.size(); i++)
  {
    if (analyticalPoseClosedFormEstimate(i) == 1)
    {
      numClosedForm++;
    }
  }
  int numDofsToSolve = analyticalPose.size() - numClosedForm;
  if (numDofsToSolve == 0)
    return 0.0;

  Eigen::VectorXs initialGuess = Eigen::VectorXs::Zero(numDofsToSolve);
  Eigen::VectorXs upperLimit = Eigen::VectorXs::Zero(numDofsToSolve);
  Eigen::VectorXs lowerLimit = Eigen::VectorXs::Zero(numDofsToSolve);
  int cursor = 0;
  for (int i = 0; i < analyticalPose.size(); i++)
  {
    if (analyticalPoseClosedFormEstimate(i) == 0)
    {
      upperLimit(cursor) = threadsafeSkel->getDof(i)->getPositionUpperLimit();
      lowerLimit(cursor) = threadsafeSkel->getDof(i)->getPositionLowerLimit();
      cursor++;
    }
  }

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = getVisibleMarkers(timestep);
  std::vector<std::string> markerNames = getVisibleMarkerNames(timestep);
  std::vector<dynamics::Joint*> otherSkelJoints = getVisibleJoints(timestep);
  std::vector<dynamics::Joint*> joints;
  for (auto* j : otherSkelJoints)
  {
    joints.push_back(threadsafeSkel->getJoint(j->getName()));
  }

  Eigen::VectorXs goal
      = Eigen::VectorXs::Zero(markers.size() * 3 + joints.size() * 3);
  for (int i = 0; i < markerNames.size(); i++)
  {
    goal.segment<3>(i * 3) = mMarkerObservations[timestep][markerNames[i]];
  }
  for (int i = 0; i < joints.size(); i++)
  {
    goal.segment<3>(markerNames.size() * 3 + i * 3)
        = mJointCenters[timestep][joints[i]->getName()];
  }

  math::solveIK(
      initialGuess,
      upperLimit,
      lowerLimit,
      goal.size(),
      // Set positions
      [threadsafeSkel, analyticalPose, analyticalPoseClosedFormEstimate](
          /* in*/ const Eigen::VectorXs pos, bool clamp) {
        (void)clamp;
        // Complete the root position to the rest of the skeleton
        Eigen::VectorXs fullPos = analyticalPose;
        int cursor = 0;
        for (int i = 0; i < analyticalPoseClosedFormEstimate.size(); i++)
        {
          if (analyticalPoseClosedFormEstimate(i) == 0)
          {
            fullPos(i) = pos(cursor);
            cursor++;
          }
        }
        threadsafeSkel->setPositions(fullPos);

        // Return the root position
        return pos;
      },
      // Compute the Jacobian
      [threadsafeSkel,
       analyticalPose,
       analyticalPoseClosedFormEstimate,
       goal,
       markers,
       joints,
       numDofsToSolve](
          /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
          /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
        diff.segment(0, markers.size() * 3)
            = threadsafeSkel->getMarkerWorldPositions(markers)
              - goal.segment(0, markers.size() * 3);
        diff.segment(markers.size() * 3, joints.size() * 3)
            = threadsafeSkel->getJointWorldPositions(joints)
              - goal.segment(markers.size() * 3, joints.size() * 3);

        (void)numDofsToSolve;
        assert(jac.cols() == numDofsToSolve);
        assert(jac.rows() == goal.size());
        jac.setZero();

        Eigen::MatrixXs markerJac
            = threadsafeSkel->getMarkerWorldPositionsJacobianWrtJointPositions(
                markers);
        Eigen::MatrixXs jointJac
            = threadsafeSkel->getJointWorldPositionsJacobianWrtJointPositions(
                joints);

        int cursor = 0;
        for (int i = 0; i < analyticalPoseClosedFormEstimate.size(); i++)
        {
          if (analyticalPoseClosedFormEstimate(i) == 0)
          {
            jac.block(0, cursor, markerJac.rows(), 1)
                = markerJac.block(0, i, markerJac.rows(), 1);
            jac.block(markerJac.rows(), cursor, jointJac.rows(), 1)
                = jointJac.block(0, i, jointJac.rows(), 1);
            cursor++;
          }
        }
      },
      // Generate a random restart position
      [threadsafeSkel, analyticalPoseClosedFormEstimate, numDofsToSolve](
          Eigen::Ref<Eigen::VectorXs> val) {
        Eigen::VectorXs fullRandom = threadsafeSkel->getRandomPose();

        Eigen::VectorXs random = Eigen::VectorXs::Zero(numDofsToSolve);
        int cursor = 0;
        for (int i = 0; i < analyticalPoseClosedFormEstimate.size(); i++)
        {
          if (analyticalPoseClosedFormEstimate(i) == 0)
          {
            random(cursor) = fullRandom(i);
            cursor++;
          }
        }
        val = random;
      },
      math::IKConfig()
          .setMaxStepCount(150)
          .setConvergenceThreshold(1e-10)
          .setMaxRestarts(3)
          .setLogOutput(false));

  mPoses[timestep] = threadsafeSkel->getPositions();
  return 0.0;
}

/// This solves ALL the DOFs, including the ones that were found in closed
/// form, to fine tune loss.
s_t IKInitializer::fineTuneIKIteratively(
    int timestep, std::shared_ptr<dynamics::Skeleton> threadsafeSkel)
{
  Eigen::VectorXs analyticalPose = mPoses[timestep];

  Eigen::VectorXs initialGuess
      = Eigen::VectorXs::Zero(threadsafeSkel->getNumDofs());
  Eigen::VectorXs upperLimit
      = Eigen::VectorXs::Zero(threadsafeSkel->getNumDofs());
  Eigen::VectorXs lowerLimit
      = Eigen::VectorXs::Zero(threadsafeSkel->getNumDofs());

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
      = getVisibleMarkers(timestep);
  std::vector<std::string> markerNames = getVisibleMarkerNames(timestep);
  std::vector<dynamics::Joint*> otherSkelJoints = getVisibleJoints(timestep);
  std::vector<dynamics::Joint*> joints;
  for (auto* j : otherSkelJoints)
  {
    joints.push_back(threadsafeSkel->getJoint(j->getName()));
  }

  Eigen::VectorXs goal
      = Eigen::VectorXs::Zero(markers.size() * 3 + joints.size() * 3);
  for (int i = 0; i < markerNames.size(); i++)
  {
    goal.segment<3>(i * 3) = mMarkerObservations[timestep][markerNames[i]];
  }
  for (int i = 0; i < joints.size(); i++)
  {
    goal.segment<3>(markerNames.size() * 3 + i * 3)
        = mJointCenters[timestep][joints[i]->getName()];
  }

  math::solveIK(
      initialGuess,
      threadsafeSkel->getPositionUpperLimits(),
      threadsafeSkel->getPositionLowerLimits(),
      goal.size(),
      // Set positions
      [threadsafeSkel, analyticalPose](
          /* in*/ const Eigen::VectorXs pos, bool clamp) {
        Eigen::VectorXs clampedPos = pos;
        if (clamp)
        {
          clampedPos
              = clampedPos.cwiseMax(threadsafeSkel->getPositionLowerLimits());
          clampedPos
              = clampedPos.cwiseMin(threadsafeSkel->getPositionUpperLimits());
        }
        // Complete the root position to the rest of the skeleton
        threadsafeSkel->setPositions(clampedPos);
        // Return the root position
        return pos;
      },
      // Compute the Jacobian
      [threadsafeSkel, analyticalPose, goal, markers, joints](
          /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
          /*out*/ Eigen::Ref<Eigen::MatrixXs> jac) {
        diff.segment(0, markers.size() * 3)
            = threadsafeSkel->getMarkerWorldPositions(markers)
              - goal.segment(0, markers.size() * 3);
        diff.segment(markers.size() * 3, joints.size() * 3)
            = threadsafeSkel->getJointWorldPositions(joints)
              - goal.segment(markers.size() * 3, joints.size() * 3);

        assert(jac.cols() == threadsafeSkel->getNumDofs());
        assert(jac.rows() == goal.size());
        jac.setZero();

        Eigen::MatrixXs markerJac
            = threadsafeSkel->getMarkerWorldPositionsJacobianWrtJointPositions(
                markers);
        Eigen::MatrixXs jointJac
            = threadsafeSkel->getJointWorldPositionsJacobianWrtJointPositions(
                joints);
        jac.block(0, 0, markerJac.rows(), markerJac.cols()) = markerJac;
        jac.block(markerJac.rows(), 0, jointJac.rows(), jointJac.cols())
            = jointJac;
      },
      // Generate a random restart position
      [threadsafeSkel](Eigen::Ref<Eigen::VectorXs> val) {
        Eigen::VectorXs fullRandom = threadsafeSkel->getRandomPose();
        val = fullRandom;
      },
      math::IKConfig()
          .setMaxStepCount(150)
          .setConvergenceThreshold(1e-10)
          .setMaxRestarts(1)
          .setLogOutput(false));

  mPoses[timestep] = threadsafeSkel->getPositions();
  return 0.0;
}

/// This uses the current guesses for the joint centers to re-estimate the
/// distances between the markers and the joint centers, and the distances
/// between the adjacent joints.
void IKInitializer::reestimateDistancesFromJointCenters()
{
  // 1. Recompute the joint to joint squared distances

  std::map<std::string, std::map<std::string, s_t>>
      jointToJointSquaredDistances;
  std::map<std::string, std::map<std::string, int>>
      jointToJointSquaredDistancesCount;
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (auto& pair : mJointToJointSquaredDistances)
    {
      std::string joint1Name = pair.first;
      for (auto& pair2 : pair.second)
      {
        std::string joint2Name = pair2.first;
        if (joint1Name == joint2Name)
        {
          continue;
        }
        if (mJointCenters[t].count(joint1Name) == 0)
        {
          continue;
        }
        if (mJointCenters[t].count(joint2Name) == 0)
        {
          continue;
        }
        Eigen::Vector3s joint1Center = mJointCenters[t][joint1Name];
        Eigen::Vector3s joint2Center = mJointCenters[t][joint2Name];
        s_t squaredDistance = (joint1Center - joint2Center).squaredNorm();
        if (jointToJointSquaredDistances[joint1Name].count(joint2Name) == 0)
        {
          jointToJointSquaredDistances[joint1Name][joint2Name] = 0;
          jointToJointSquaredDistancesCount[joint1Name][joint2Name] = 0;
        }
        jointToJointSquaredDistances[joint1Name][joint2Name] += squaredDistance;
        jointToJointSquaredDistancesCount[joint1Name][joint2Name]++;
      }
    }
  }
  for (auto& pair : jointToJointSquaredDistances)
  {
    std::string joint1Name = pair.first;
    for (auto& pair2 : pair.second)
    {
      std::string joint2Name = pair2.first;
      s_t squaredDistance
          = pair2.second
            / jointToJointSquaredDistancesCount[joint1Name][joint2Name];
      mJointToJointSquaredDistances[joint1Name][joint2Name] = squaredDistance;
      mJointToJointSquaredDistances[joint2Name][joint1Name] = squaredDistance;
    }
  }

  // 2. Do the same thing, but for joint to marker distances

  std::map<std::string, std::map<std::string, s_t>>
      jointToMarkerSquaredDistances;
  std::map<std::string, std::map<std::string, int>>
      jointToMarkerSquaredDistancesCount;
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (auto& pair : mJointToMarkerSquaredDistances)
    {
      std::string jointName = pair.first;
      for (auto& pair2 : pair.second)
      {
        std::string markerName = pair2.first;
        if (mJointCenters[t].count(jointName) == 0)
        {
          continue;
        }
        if (mMarkerObservations[t].count(markerName) == 0)
        {
          continue;
        }
        Eigen::Vector3s jointCenter = mJointCenters[t][jointName];
        Eigen::Vector3s marker = mMarkerObservations[t][markerName];
        s_t squaredDistance = (jointCenter - marker).squaredNorm();
        if (jointToMarkerSquaredDistances[jointName].count(markerName) == 0)
        {
          jointToMarkerSquaredDistances[jointName][markerName] = 0;
          jointToMarkerSquaredDistancesCount[jointName][markerName] = 0;
        }
        jointToMarkerSquaredDistances[jointName][markerName] += squaredDistance;
        jointToMarkerSquaredDistancesCount[jointName][markerName]++;
      }
    }
  }
  for (auto& pair : jointToMarkerSquaredDistances)
  {
    std::string jointName = pair.first;
    for (auto& pair2 : pair.second)
    {
      std::string markerName = pair2.first;
      s_t squaredDistance
          = pair2.second
            / jointToMarkerSquaredDistancesCount[jointName][markerName];
      mJointToMarkerSquaredDistances[jointName][markerName] = squaredDistance;
      mJointToMarkerSquaredDistances[jointName][markerName] = squaredDistance;
    }
  }
}

//==============================================================================
/// This gets the average distance between adjacent joint centers in our
/// current joint center estimates.
std::map<std::string, std::map<std::string, s_t>>
IKInitializer::estimateJointToJointDistances()
{
  // 1. Recompute the joint to joint squared distances

  std::map<std::string, std::map<std::string, s_t>> jointToJointAvgDistances;
  std::map<std::string, std::map<std::string, int>>
      jointToJointAvgDistancesCount;
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (auto& pair : mJointToJointSquaredDistances)
    {
      std::string joint1Name = pair.first;
      for (auto& pair2 : pair.second)
      {
        std::string joint2Name = pair2.first;
        if (joint1Name == joint2Name)
        {
          continue;
        }
        if (mJointCenters[t].count(joint1Name) == 0)
        {
          continue;
        }
        if (mJointCenters[t].count(joint2Name) == 0)
        {
          continue;
        }
        Eigen::Vector3s joint1Center = mJointCenters[t][joint1Name];
        Eigen::Vector3s joint2Center = mJointCenters[t][joint2Name];
        s_t distance = (joint1Center - joint2Center).norm();
        if (jointToJointAvgDistances[joint1Name].count(joint2Name) == 0)
        {
          jointToJointAvgDistances[joint1Name][joint2Name] = 0;
          jointToJointAvgDistancesCount[joint1Name][joint2Name] = 0;
        }
        jointToJointAvgDistances[joint1Name][joint2Name] += distance;
        jointToJointAvgDistancesCount[joint1Name][joint2Name]++;
      }
    }
  }
  for (auto& pair : jointToJointAvgDistances)
  {
    std::string joint1Name = pair.first;
    for (auto& pair2 : pair.second)
    {
      std::string joint2Name = pair2.first;
      jointToJointAvgDistances[joint1Name][joint2Name]
          /= jointToJointAvgDistancesCount[joint1Name][joint2Name];
    }
  }
  return jointToJointAvgDistances;
}

//==============================================================================
/// This gets the subset of markers that are visible at a given timestep
std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
IKInitializer::getVisibleMarkers(int t)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mMarkerObservations[t].count(mMarkerNames[i]))
    {
      markers.push_back(mMarkers[i]);
    }
  }
  return markers;
}

//==============================================================================
/// This gets the subset of markers that are visible at a given timestep
std::vector<std::string> IKInitializer::getVisibleMarkerNames(int t)
{
  std::vector<std::string> markerNames;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mMarkerObservations[t].count(mMarkerNames[i]))
    {
      markerNames.push_back(mMarkerNames[i]);
    }
  }
  return markerNames;
}

//==============================================================================
/// This gets the subset of joints that are attached to markers that are
/// visible at a given timestep
std::vector<dynamics::Joint*> IKInitializer::getVisibleJoints(int t)
{
  std::vector<dynamics::Joint*> joints;
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    dynamics::Joint* joint = mSkel->getJoint(i);
    if (mJointToMarkerSquaredDistances.count(joint->getName()) == 0)
    {
      continue;
    }
    for (auto& pair : mJointToMarkerSquaredDistances[joint->getName()])
    {
      if (mMarkerObservations[t].count(pair.first))
      {
        joints.push_back(joint);
        break;
      }
    }
  }
  return joints;
}

//==============================================================================
/// This gets the world center estimates for joints that are attached to
/// markers that are visible at this timestep.
std::map<std::string, Eigen::Vector3s> IKInitializer::getVisibleJointCenters(
    int t)
{
  return mJointCenters[t];
}

//==============================================================================
/// This gets the squared distance between a joint and a marker on an adjacent
/// body segment.
std::map<std::string, s_t> IKInitializer::getJointToMarkerSquaredDistances(
    std::string jointName)
{
  return mJointToMarkerSquaredDistances[jointName];
}

//==============================================================================
/// This does a rank-N completion of the distance matrix, which provides some
/// guarantees about reconstruction quality. See "Distance Matrix
/// Reconstruction from Incomplete Distance Information for Sensor Network
/// Localization"
Eigen::MatrixXs IKInitializer::rankNDistanceMatrix(
    const Eigen::MatrixXs& distances, int n)
{
  Eigen::JacobiSVD<Eigen::MatrixXs> svd(
      distances, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXs U = svd.matrixU();
  Eigen::VectorXs S = svd.singularValues();
  Eigen::MatrixXs V = svd.matrixV();

  Eigen::MatrixXs Ur = U.leftCols(n);
  Eigen::MatrixXs Vr = V.leftCols(n);
  Eigen::VectorXs Sr = S.head(n);

  Eigen::MatrixXs rank_five_approximation
      = Ur * Sr.asDiagonal() * Vr.transpose();
  return rank_five_approximation;
}

//==============================================================================
/// This will reconstruct a centered Euclidean point cloud from a distance
/// matrix.
Eigen::MatrixXs IKInitializer::getPointCloudFromDistanceMatrix(
    const Eigen::MatrixXs& distances)
{
  int n = distances.rows();
  Eigen::MatrixXs centering_matrix = Eigen::MatrixXs::Identity(n, n)
                                     - Eigen::MatrixXs::Constant(n, n, 1.0 / n);
  Eigen::MatrixXs double_centering_matrix
      = -0.5 * centering_matrix * distances * centering_matrix;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXs> eigensolver(
      double_centering_matrix);
  Eigen::VectorXs eigenvalues = eigensolver.eigenvalues();
  Eigen::MatrixXs eigenvectors = eigensolver.eigenvectors();
  assert(eigensolver.info() == Eigen::Success);
  assert(!eigenvalues.hasNaN());
  assert(!eigenvectors.hasNaN());

  int k = 3;
  Eigen::MatrixXs k_eigenvectors = eigenvectors.rightCols(k);
  Eigen::VectorXs k_eigenvalues
      = eigenvalues.tail(k).cwiseMax(1e-16).cwiseSqrt();
  assert(!k_eigenvalues.hasNaN());
  assert(!k_eigenvectors.hasNaN());

  return k_eigenvalues.asDiagonal() * k_eigenvectors.transpose();
}

//==============================================================================
/// This will rotate and translate a point cloud to match the first N points
/// as closely as possible to the passed in matrix
Eigen::MatrixXs IKInitializer::mapPointCloudToData(
    const Eigen::MatrixXs& pointCloud,
    std::vector<Eigen::Vector3s> firstNPoints)
{
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> targetPointCloud
      = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, firstNPoints.size());
  for (int i = 0; i < firstNPoints.size(); i++)
  {
    targetPointCloud.col(i) = firstNPoints[i];
  }
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> sourcePointCloud
      = pointCloud.block(0, 0, 3, firstNPoints.size());

  assert(sourcePointCloud.cols() == targetPointCloud.cols());

  // Compute the centroids of the source and target points
  Eigen::Vector3s sourceCentroid = sourcePointCloud.rowwise().mean();
  Eigen::Vector3s targetCentroid = targetPointCloud.rowwise().mean();

#ifndef NDEBUG
  Eigen::Vector3s sourceAvg = Eigen::Vector3s::Zero();
  Eigen::Vector3s targetAvg = Eigen::Vector3s::Zero();
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    sourceAvg += sourcePointCloud.col(i);
    targetAvg += targetPointCloud.col(i);
  }
  sourceAvg /= sourcePointCloud.cols();
  targetAvg /= targetPointCloud.cols();
  assert((sourceAvg - sourceCentroid).norm() < 1e-12);
  assert((targetAvg - targetCentroid).norm() < 1e-12);
#endif

  // Compute the centered source and target points
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredSourcePoints
      = sourcePointCloud.colwise() - sourceCentroid;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredTargetPoints
      = targetPointCloud.colwise() - targetCentroid;

#ifndef NDEBUG
  assert(std::abs(centeredSourcePoints.rowwise().mean().norm()) < 1e-8);
  assert(std::abs(centeredTargetPoints.rowwise().mean().norm()) < 1e-8);
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    Eigen::Vector3s expectedCenteredSourcePoints
        = sourcePointCloud.col(i) - sourceAvg;
    assert(
        (centeredSourcePoints.col(i) - expectedCenteredSourcePoints).norm()
        < 1e-12);
    Eigen::Vector3s expectedCenteredTargetPoints
        = targetPointCloud.col(i) - targetAvg;
    assert(
        (centeredTargetPoints.col(i) - expectedCenteredTargetPoints).norm()
        < 1e-12);
  }
#endif

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix
      = centeredTargetPoints * centeredSourcePoints.transpose();

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  // Normally, we would want to check the determinant of R here, to ensure that
  // we're only doing right-handed rotations. HOWEVER, because we're using a
  // point cloud, we may actually have to flip the data along an axis to get it
  // to match up, so we skip the determinant check.

  // Transform the source point cloud to the target point cloud
  Eigen::MatrixXs transformed = Eigen::MatrixXs::Zero(3, pointCloud.cols());
  for (int i = 0; i < pointCloud.cols(); i++)
  {
    transformed.col(i)
        = R * (pointCloud.col(i).head<3>() - sourceCentroid) + targetCentroid;
  }
  return transformed;
}

/// This will give the world transform necessary to apply to the local points
/// (worldT * p[i] for all localPoints) to get the local points to match the
/// world points as closely as possible.
Eigen::Isometry3s IKInitializer::getPointCloudToPointCloudTransform(
    std::vector<Eigen::Vector3s> localPoints,
    std::vector<Eigen::Vector3s> worldPoints,
    std::vector<s_t> weights)
{
  assert(localPoints.size() > 0);
  assert(worldPoints.size() > 0);
  assert(localPoints.size() == worldPoints.size());

  // Compute the centroids of the local and world points
  Eigen::Vector3s localCentroid = Eigen::Vector3s::Zero();
  for (Eigen::Vector3s& point : localPoints)
  {
    localCentroid += point;
  }
  localCentroid /= localPoints.size();
  Eigen::Vector3s worldCentroid = Eigen::Vector3s::Zero();
  for (Eigen::Vector3s& point : worldPoints)
  {
    worldCentroid += point;
  }
  worldCentroid /= worldPoints.size();

  // Compute the centered local and world points
  std::vector<Eigen::Vector3s> centeredLocalPoints;
  std::vector<Eigen::Vector3s> centeredWorldPoints;
  for (int i = 0; i < localPoints.size(); i++)
  {
    centeredLocalPoints.push_back(localPoints[i] - localCentroid);
    centeredWorldPoints.push_back(worldPoints[i] - worldCentroid);
  }

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix = Eigen::Matrix3s::Zero();
  for (int i = 0; i < localPoints.size(); i++)
  {
    covarianceMatrix += weights[i] * centeredWorldPoints[i]
                        * centeredLocalPoints[i].transpose();
  }

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  if (R.determinant() < 0)
  {
    Eigen::Matrix3s scales = Eigen::Matrix3s::Identity();
    scales(2, 2) = -1;
    R = U * scales * V.transpose();
  }
  Eigen::Vector3s translation = worldCentroid - R * localCentroid;

  Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
  transform.linear() = R;
  transform.translation() = translation;
  return transform;
}

} // namespace biomechanics
} // namespace dart