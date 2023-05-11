#ifndef DART_BIOMECH_CONVEX_IK_INIT
#define DART_BIOMECH_CONVEX_IK_INIT

#include <memory>
#include <tuple>
#include <vector>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"

namespace dart {
namespace biomechanics {

/**
 * This class implements a closed-form optimization problem for initializing the
 * joint center locations in world space over time, and then using that to
 * compute the body scales and positions. The core idea is that we assume that
 * the distances between optical markers and joint centers are known and fixed
 * quantities.
 *
 * There are several iterative "polishing" steps that can also be done to
 * improve the closed form solution in the presence of noise.
 */
class IKInitializer
{
public:
  IKInitializer(
      std::shared_ptr<dynamics::Skeleton> skel,
      std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
          markers,
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
      s_t modelHeightM = -1.0);

  /// This runs the full IK initialization algorithm, and leaves the answers in
  /// the public fields of this class
  void runFullPipeline(bool logOutput = false);

  /// For each timestep, and then for each joint, this sets up and runs an MDS
  /// algorithm to triangulate the joint center location from the known marker
  /// locations of the markers attached to the joint's two body segments, and
  /// then known distance from the joint center to each marker.
  s_t closedFormMDSJointCenterSolver(bool logOutput = false);

  /// This first finds an approximate rigid body trajectory for each body
  /// that has at least 3 markers on it, and then sets up and solves a linear
  /// system of equations for each joint to find the pair of offsets in the
  /// adjacent body nodes that results in the least offset between the joint
  /// centers.
  s_t closedFormPivotFindingJointCenterSolver(bool logOutput = false);

  /// This uses the current guesses for the joint centers to re-estimate the
  /// bone sizes (based on distance between joint centers) and then use that to
  /// get the group scale vector. This also uses the joint centers to estimate
  /// the body positions.
  s_t estimatePosesAndGroupScalesInClosedForm(bool logOutput = false);

  /// This solves the remaining DOFs that couldn't be found in closed form using
  /// an iterative IK solver. This portion of the solver is the only requirement
  /// that is a non-convex portion. It uses random-restarts, and so is not as
  /// unit-testable as the other portions of the algorithm, so it should
  /// hopefully only impact less important joints.
  s_t completeIKIteratively(
      int timestep, std::shared_ptr<dynamics::Skeleton> threadsafeSkel);

  /// This solves ALL the DOFs, including the ones that were found in closed
  /// form, to fine tune loss.
  s_t fineTuneIKIteratively(
      int timestep, std::shared_ptr<dynamics::Skeleton> threadsafeSkel);

  /// This uses the current guesses for the joint centers to re-estimate the
  /// distances between the markers and the joint centers, and the distances
  /// between the adjacent joints.
  void reestimateDistancesFromJointCenters();

  /// This gets the average distance between adjacent joint centers in our
  /// current joint center estimates.
  std::map<std::string, std::map<std::string, s_t>>
  estimateJointToJointDistances();

  /// This gets the subset of markers that are visible at a given timestep
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
  getVisibleMarkers(int t);

  /// This gets the subset of markers that are visible at a given timestep
  std::vector<std::string> getVisibleMarkerNames(int t);

  /// This gets the subset of joints that are attached to markers that are
  /// visible at a given timestep
  std::vector<dynamics::Joint*> getVisibleJoints(int t);

  /// This gets the world center estimates for joints that are attached to
  /// markers that are visible at this timestep.
  std::map<std::string, Eigen::Vector3s> getVisibleJointCenters(int t);

  /// This gets the squared distance between a joint and a marker on an adjacent
  /// body segment.
  std::map<std::string, s_t> getJointToMarkerSquaredDistances(
      std::string jointName);

  /// This does a rank-N completion of the distance matrix, which provides some
  /// guarantees about reconstruction quality. See "Distance Matrix
  /// Reconstruction from Incomplete Distance Information for Sensor Network
  /// Localization"
  static Eigen::MatrixXs rankNDistanceMatrix(
      const Eigen::MatrixXs& distances, int n = 3);

  /// This will reconstruct a centered Euclidean point cloud from a distance
  /// matrix.
  static Eigen::MatrixXs getPointCloudFromDistanceMatrix(
      const Eigen::MatrixXs& distances);

  /// This will rotate and translate a point cloud to match the first N points
  /// as closely as possible to the passed in matrix
  static Eigen::MatrixXs mapPointCloudToData(
      const Eigen::MatrixXs& pointCloud,
      std::vector<Eigen::Vector3s> firstNPoints);

  /// This will give the world transform necessary to apply to the local points
  /// (worldT * p[i] for all localPoints) to get the local points to match the
  /// world points as closely as possible.
  static Eigen::Isometry3s getPointCloudToPointCloudTransform(
      std::vector<Eigen::Vector3s> localPoints,
      std::vector<Eigen::Vector3s> worldPoints,
      std::vector<s_t> weights);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkel;
  s_t mModelHeightM;
  std::vector<std::string> mMarkerNames;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerObservations;

  std::vector<dynamics::Joint*> mJoints;
  std::vector<std::map<std::string, Eigen::Vector3s>> mJointCenters;

  std::map<std::string, std::map<std::string, s_t>>
      mJointToMarkerSquaredDistances;
  std::map<std::string, std::map<std::string, s_t>>
      mJointToJointSquaredDistances;
  std::map<std::string, std::map<std::string, std::vector<dynamics::BodyNode*>>>
      mJointToJointBodyNodesPath;

public:
  // Results from the IK solver
  std::vector<std::map<std::string, Eigen::Isometry3s>> mBodyTransforms;
  Eigen::VectorXs mGroupScales;
  std::vector<Eigen::VectorXs> mPoses;
  // These are vectors of integers (as booleans), where a 1 in index i means
  // that DOF [i] was found with a closed form estimate, and a 0 means that it
  // wasn't.
  std::vector<Eigen::VectorXi> mPosesClosedFormEstimateAvailable;
};

} // namespace biomechanics
} // namespace dart

#endif