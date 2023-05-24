#ifndef DART_BIOMECH_CONVEX_IK_INIT
#define DART_BIOMECH_CONVEX_IK_INIT

#include <memory>
#include <tuple>
#include <vector>

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

/// Several bookkeeping things need to happen in order to handle edge cases in
/// arbitrary uploaded OpenSim files elegantly.
///
/// 1. Joints that are stacked on top of each other with negligible offsets
/// between them (effectively using many low-DOF joints to represent one
/// high-DOF joint) must be somehow merged and treated as a unit, so that we can
/// use as many adjacent markers as possible and solve for a single shared joint
/// center across all the "stacked" joints.
///
/// 2. Welded joints must be collapsed, and the bodies they connect must be
/// treated as a single body.
///
/// What we ultimately want is to be able to rewrite the IK algorithms in terms
/// of this SimplifiedSkeleton.
///
/// Really, this means that we're forming a new topology out of lists of bodies
/// and lists of joints.
struct StackedJoint;
struct StackedBody;
struct StackedJoint
{
  std::string name;
  std::vector<dynamics::Joint*> joints;
  std::shared_ptr<struct StackedBody> parentBody;
  std::shared_ptr<struct StackedBody> childBody;
};
struct StackedBody
{
  std::string name;
  std::vector<dynamics::BodyNode*> bodies;
  std::shared_ptr<struct StackedJoint> parentJoint;
  std::vector<std::shared_ptr<struct StackedJoint>> childJoints;
  double numMarkers;
};

enum JointCenterEstimateSource
{
  MDS = 0,
  LEAST_SQUARES_EXACT = 1,
  LEAST_SQUARES_AXIS = 2,
};

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
      std::map<std::string, bool> markerIsAnatomical,
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
      std::vector<bool> newClip,
      s_t modelHeightM = -1.0);

  //////////////////////////////////////////////////////////////////////////////
  // Core entrypoint
  //////////////////////////////////////////////////////////////////////////////

  /// This runs the full IK initialization algorithm, and leaves the answers in
  /// the public fields of this class.
  ///
  /// The key results are
  /// - mGroupScales: call skeleton->setGroupScales(mGroupScales) to apply
  /// - mPoses: each column of mPoses represents one timestep
  ///
  /// The other fields that may be of use are the values for joint locations,
  /// which can be used for loss terms:
  /// - mJointCenters: each entry of mJointCenters is a map of joint_name ->
  /// joint_location, if available
  /// - mJointAxisDirs: each entry of mJointAxisDirs is a map of joint_name ->
  /// joint_axis_dir, if available
  /// - mStackedJoints: each entry of mStackedJoints is a StackedJoint, which
  /// contains a list of joints whose average center is the joint center that
  /// the other entries are talking about
  void runFullPipeline(bool logOutput = false);

  //////////////////////////////////////////////////////////////////////////////
  // Steps of the pipeline
  //////////////////////////////////////////////////////////////////////////////

  /// This takes advantage of the fact that we assume anatomical markers have
  /// pretty much known locations on their body segment. That means that two or
  /// more anatomical markers on the same body segment can tell you about the
  /// body segment scaling, which can then help inform the distances between
  /// those markers and the joints, which helps all the subsequent steps in the
  /// pipeline be more accurate.
  s_t prescaleBasedOnAnatomicalMarkers(bool logOutput = false);

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

  /// For joints where we believe they're floating along an axis, and we have at
  /// least one of either the parent or child joint has a known joint center
  /// WITHOUT the axis amiguity, we can slide the joint center along the axis
  /// until the angle formed by the axis and the known adjacent joint matches
  /// what is in the skeleton.
  s_t recenterAxisJointsBasedOnBoneAngles(bool logOutput = false);

  /// This uses the current guesses for the joint centers to re-estimate the
  /// bone sizes (based on distance between joint centers) and then use that to
  /// get the group scale vector.
  void estimateGroupScalesClosedForm(bool logOutput = false);

  /// This takes the current guesses for the joint centers and the group scales,
  /// and just runs straightforward IK to get the body positions.
  s_t estimatePosesWithIK(bool logOutput = false);

  /// WARNING: You must have already called estimateGroupScalesClosedForm()!
  /// This uses the joint centers to estimate the body positions.
  s_t estimatePosesClosedForm(bool logOutput = false);

  //////////////////////////////////////////////////////////////////////////////
  // Utilities
  //////////////////////////////////////////////////////////////////////////////

  /// This gets the average distance between adjacent joint centers in our
  /// current joint center estimates.
  std::map<std::string, std::map<std::string, s_t>>
  estimateJointToJointDistances();

  /// This gets the subset of markers that are visible at a given timestep
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
  getObservedMarkers(int t);

  /// This gets the subset of markers that are visible at a given timestep
  std::vector<std::string> getObservedMarkerNames(int t);

  /// This gets the subset of joints that are attached to markers that are
  /// visible at a given timestep
  std::vector<std::shared_ptr<struct StackedJoint>>
  getJointsAttachedToObservedMarkers(int t);

  /// This gets the world center estimates for joints that are attached to
  /// markers that are visible at this timestep.
  std::map<std::string, Eigen::Vector3s>
  getJointsAttachedToObservedMarkersCenters(int t);

  /// This gets the squared distance between a joint and a marker on an adjacent
  /// body segment.
  std::map<std::string, s_t> getJointToMarkerSquaredDistances(
      std::string jointName);

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

  /// This tries to solve the least-squares problem to get the local scales for
  /// a body, such that the distances between the local points match the input
  /// distances as closely as possible, with a weighted preference.
  static Eigen::Vector3s getLocalScale(
      std::vector<Eigen::Vector3s> localPoints,
      std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights,
      s_t defaultAxisScale = 1.0,
      bool logOutput = false);

  /// This implements the method in "Constrained least-squares optimization for
  /// robust estimation of center of rotation" by Chang and Pollard, 2006. This
  /// is the simpler version, that only supports a single marker.
  ///
  /// This assumes we're operating in a frame where the joint center is fixed in
  /// place, and `markerTrace` is a list of marker positions over time in that
  /// frame. So `markerTrace[t]` is the location of the marker on its t'th
  /// observation. This method will return the joint center in that frame.
  ///
  /// The benefit of using this instead of the
  /// `closedFormPivotFindingJointCenterSolver` is that we can run it on pairs
  /// of bodies where only one side of the pair has 3 markers on it (commonly,
  /// the ankle).
  ///
  /// This method has various singularities, such as if there is a marker that
  /// is zero noise in the marker data (because it's synthetic), or if the
  /// marker is exactly on the joint center. In these cases, the method will
  /// fall back to the leastSquaresConcentricSphereFit() below.
  static Eigen::Vector3s getChangPollard2006JointCenterMultiMarker(
      std::vector<std::vector<Eigen::Vector3s>> markerTrace, bool log = false);

  /// This implements a simple least-squares problem to find the center of a
  /// sphere of unknown radius, with samples along the hull given by `points`.
  /// This tolerates noise less well than the ChangPollard2006 method, because
  /// it biases the radius of the sphere towards zero in the presence of
  /// ambiguity, because it is part of the least-squares terms. However, this
  /// method will work even on data with zero noise, whereas ChangPollard2006
  /// will fail when there is zero noise (for example, on synthetic datasets).
  static Eigen::Vector3s leastSquaresConcentricSphereFit(
      std::vector<std::vector<Eigen::Vector3s>> traces, bool logOutput = false);

  /// This implements the least-squares method in Gamage and Lasenby 2002, "New
  /// least squares solutions for estimating the average centre of rotation and
  /// the axis of rotation"
  static std::pair<Eigen::Vector3s, s_t> gamageLasenby2002AxisFit(
      std::vector<std::vector<Eigen::Vector3s>> traces);

  /// This uses Cardano's method to solve a cubic equation in closed form. It
  /// return a list of roots, real and complex.
  static std::vector<double> findCubicRealRoots(
      double a, double b, double c, double d);

  /// This method will find the best point on the line given by f(x) = (c + a*x)
  /// that minimizes: Sum_i weights[i] * (f(x) -
  /// pointsAndRadii[i].first).squaredNorm() - pointsAndRadii[i].second^2)^2
  static Eigen::Vector3s centerPointOnAxis(
      Eigen::Vector3s c,
      Eigen::Vector3s a,
      std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii,
      std::vector<s_t> weights = std::vector<s_t>());

  /// Here's a bunch of simple getters:

  std::vector<std::map<std::string, Eigen::Vector3s>>& getJointCenters();

  std::vector<std::map<std::string, Eigen::Vector3s>>& getJointAxisDirs();

  std::map<std::string, std::map<std::string, s_t>>&
  getJointToMarkerSquaredDistances();

  std::map<std::string, std::map<std::string, s_t>>&
  getJointToJointSquaredDistances();

  std::vector<std::shared_ptr<struct StackedBody>>& getStackedBodies();

  std::vector<std::shared_ptr<struct StackedJoint>>& getStackedJoints();

  std::vector<std::map<std::string, Eigen::Isometry3s>>& getBodyTransforms();

  Eigen::VectorXs getGroupScales();

  std::vector<Eigen::VectorXs> getPoses();

  std::vector<Eigen::VectorXi> getPosesClosedFormEstimateAvailable();

  std::vector<std::map<std::string, Eigen::Vector3s>>&
  getDebugKnownSyntheticJointCenters();

protected:
  std::shared_ptr<dynamics::Skeleton> mSkel;
  s_t mModelHeightM;
  std::vector<std::string> mMarkerNames;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerObservations;
  std::vector<bool> mNewClip;
  std::vector<bool> mMarkerIsAnatomical;
  std::map<std::string, int> mMarkerNameToIndex;

  std::vector<std::map<std::string, Eigen::Vector3s>> mJointCenters;
  std::vector<std::map<std::string, JointCenterEstimateSource>>
      mJointCentersEstimateSource;
  std::vector<std::map<std::string, Eigen::Vector3s>> mJointAxisDirs;

  std::map<std::string, std::map<std::string, s_t>>
      mJointToMarkerSquaredDistances;
  std::map<std::string, std::map<std::string, s_t>>
      mJointToJointSquaredDistances;

  // This holds the simplified skeleton, which is a list of (possibly stacked)
  // bodies and joints. This is public to faccilitate unit testing.
  std::vector<std::shared_ptr<struct StackedBody>> mStackedBodies;
  std::vector<std::shared_ptr<struct StackedJoint>> mStackedJoints;

  // Results from the IK solver
  std::vector<std::map<std::string, Eigen::Isometry3s>> mBodyTransforms;
  Eigen::VectorXs mGroupScales;
  std::vector<Eigen::VectorXs> mPoses;
  // These are vectors of integers (as booleans), where a 1 in index i means
  // that DOF [i] was found with a closed form estimate, and a 0 means that it
  // wasn't.
  std::vector<Eigen::VectorXi> mPosesClosedFormEstimateAvailable;

  // If we have synthetic joint centers with zero error, we can store them here
  // so that we can compare our estimates to them with asserts mid-function
  std::vector<std::map<std::string, Eigen::Vector3s>>
      mDebugKnownSyntheticJointCenters;
};

} // namespace biomechanics
} // namespace dart

#endif