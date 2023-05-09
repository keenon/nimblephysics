#include "dart/biomechanics/IKInitializer.hpp"

#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

IKInitializer::IKInitializer(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        markers,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    s_t modelHeightM)
  : mSkel(skel), mMarkerObservations(markerObservations)
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
      assert(abs(newHeight - modelHeightM) < 1e-6);
    }
  }

  // 2. Find all the joints that are connected to at least one marker, and
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
      if (mMarkers[j].first->getParentJoint() == joint
          || joint->getParentBodyNode() == mMarkers[j].first)
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

    if (jointToMarkerSquaredDistances.size() > 0)
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

      if ((joint1->getParentBodyNode() != nullptr
           && joint1->getParentBodyNode()->getParentJoint() == joint2)
          || (joint2->getParentBodyNode() != nullptr
              && joint2->getParentBodyNode()->getParentJoint() == joint1))
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

/// For each timestep, and then for each joint, this sets up and runs an MDS
/// algorithm to triangulate the joint center location from the known marker
/// locations of the markers attached to the joint's two body segments, and
/// then known distance from the joint center to each marker.
s_t IKInitializer::islandJointCenterSolver()
{
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
        // if (lastSolvedJointCenters.count(joint->getName()))
        //   continue;

        std::vector<Eigen::Vector3s> adjacentPointLocations;
        std::vector<s_t> adjacentPointSquaredDistances;

        // 1. Find the markers that are adjacent to this joint and visible on
        // this frame
        for (auto& pair : mJointToMarkerSquaredDistances[joint->getName()])
        {
          if (mMarkerObservations[t].count(pair.first))
          {
            adjacentPointLocations.push_back(
                mMarkerObservations[t][pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
          }
        }

        for (auto& pair : mJointToJointSquaredDistances[joint->getName()])
        {
          if (lastSolvedJointCenters.count(pair.first))
          {
            adjacentPointLocations.push_back(
                lastSolvedJointCenters[pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
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
        totalMarkerError += pointCloudError;
        count++;

        // 5. Record the newly found joint center
        Eigen::Vector3s jointCenter = transformed.col(dim - 1);
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

/// This gets the world center estimates for joints that are attached to
/// markers that are visible at this timestep.
std::map<std::string, Eigen::Vector3s> IKInitializer::getVisibleJointCenters(
    int t)
{
  return mJointCenters[t];
}

/// This gets the squared distance between a joint and a marker on an adjacent
/// body segment.
std::map<std::string, s_t> IKInitializer::getJointToMarkerSquaredDistances(
    std::string jointName)
{
  return mJointToMarkerSquaredDistances[jointName];
}

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
  Eigen::Vector3s trans = targetCentroid;

  // Compute the transformation as an Isometry3s
  Eigen::Isometry3s transformation = Eigen::Isometry3s::Identity();
  transformation.linear() = R;
  transformation.translation() = trans;

  // Transform the source point cloud to the target point cloud
  Eigen::MatrixXs transformed = Eigen::MatrixXs::Zero(3, pointCloud.cols());
  for (int i = 0; i < pointCloud.cols(); i++)
  {
    transformed.col(i)
        = transformation * (pointCloud.col(i).head<3>() - sourceCentroid);
  }
  return transformed;
}

} // namespace biomechanics
} // namespace dart