#ifndef DART_BIOMECH_MARKERFITTER_HPP_
#define DART_BIOMECH_MARKERFITTER_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <mutex>
#include <vector>

#include <Eigen/Dense>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

class MarkerFitter;
struct MarkerFitterState;

struct BilevelFitResult
{
  bool success;

  Eigen::VectorXs groupScales;
  std::vector<Eigen::VectorXs> poses;
  Eigen::MatrixXs posesMatrix;
  std::vector<int> sampleIndices;
  std::map<std::string, Eigen::Vector3s> markerOffsets;

  Eigen::VectorXs rawMarkerOffsets;

  BilevelFitResult();
};

/**
 * This gives us a legible version of what's going on inside the optimization
 * problem, so it's easier to unpack specific parts of the loss in useful ways.
 */
struct MarkerFitterState
{
  // It's very expensive to bind maps over to Python (for some reason I don't
  // understand), so we instead send everything as arrays, and reconstruct maps
  // in the Python API

  // The current state
  std::vector<std::string> bodyNames;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> bodyScales;
  Eigen::VectorXs jointWeights;
  Eigen::VectorXs axisWeights;

  std::vector<std::string> markerOrder;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> markerOffsets;
  Eigen::MatrixXs markerErrorsAtTimesteps;
  Eigen::MatrixXs posesAtTimesteps;
  std::vector<std::string> jointOrder;
  Eigen::MatrixXs jointErrorsAtTimesteps;
  Eigen::MatrixXs axisErrorsAtTimesteps;
  Eigen::Vector6s staticPoseRoot;

  // The gradient of the current state, which is not always used, but can help
  // shuttling information back and forth from friendly PyTorch APIs.
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> bodyScalesGrad;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> markerOffsetsGrad;
  Eigen::MatrixXs markerErrorsAtTimestepsGrad;
  Eigen::MatrixXs posesAtTimestepsGrad;
  Eigen::MatrixXs jointErrorsAtTimestepsGrad;
  Eigen::MatrixXs axisErrorsAtTimestepsGrad;
  Eigen::Vector6s staticPoseRootGrad;

  /// This unflattens an input vector, given some information about the problm
  MarkerFitterState(
      const Eigen::VectorXs& flat,
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
      std::vector<dynamics::Joint*> joints,
      Eigen::MatrixXs jointCenters,
      Eigen::VectorXs jointWeights,
      Eigen::MatrixXs jointAxis,
      Eigen::VectorXs axisWeights,
      MarkerFitter* fitter);

  /// This returns a single flat vector representing this whole problem state
  Eigen::VectorXs flattenState();

  /// This returns a single flat vector representing the gradient of this whole
  /// problem state
  Eigen::VectorXs flattenGradient();

protected:
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::shared_ptr<dynamics::Skeleton> skeleton;
  std::vector<dynamics::Joint*> joints;
  Eigen::MatrixXs jointCenters;
  Eigen::MatrixXs jointAxis;
  MarkerFitter* fitter;
};

/**
 * We create a single initialization object, and pass it around to optimization
 * problems to re-use, because it's not super cheap to construct.
 */
struct MarkerInitialization
{
  Eigen::MatrixXs poses;
  Eigen::VectorXs
      poseScores; // These are the loss values for IK at each timestep
  Eigen::VectorXs groupScales;
  std::map<std::string, Eigen::Vector3s> markerOffsets;
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      updatedMarkerMap;
  Eigen::Vector6s staticPoseRoot;

  std::vector<dynamics::Joint*> joints;
  std::vector<std::vector<std::string>> jointsAdjacentMarkers;
  Eigen::VectorXs jointMarkerVariability;
  Eigen::VectorXs jointLoss;
  Eigen::VectorXs jointWeights;
  Eigen::MatrixXs jointCenters;
  Eigen::VectorXs axisWeights;
  Eigen::VectorXs axisLoss;
  Eigen::MatrixXs jointAxis;

  std::vector<std::string> observedMarkers;
  std::vector<dynamics::Joint*> observedJoints;
  std::vector<dynamics::Joint*> unobservedJoints;

  MarkerInitialization();
};

/**
 * This sets up and finds the joint centers using a non-convex sphere-fitting
 * method.
 */
class SphereFitJointCenterProblem
{
public:
  SphereFitJointCenterProblem(
      MarkerFitter* fitter,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      Eigen::MatrixXs ikPoses,
      dynamics::Joint* joint,
      const std::vector<bool>& newClip,
      Eigen::Ref<Eigen::MatrixXs> out);

  /// This returns true if the given body is the parent of the joint OR if
  /// there's a hierarchy of fixed joints that connect it to the parent
  static bool isDynamicParentOfJoint(
      std::string bodyName, dynamics::Joint* joint);

  /// This returns true if the given body is the child of the joint OR if
  /// there's a hierarchy of fixed joints that connect it to the child
  static bool isDynamicChildOfJoint(
      std::string bodyName, dynamics::Joint* joint);

  static bool canFitJoint(
      MarkerFitter* fitter,
      dynamics::Joint* joint,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  int getProblemDim();

  Eigen::VectorXs flatten();

  void unflatten(Eigen::VectorXs x);

  s_t getLoss();

  Eigen::VectorXs getGradient();

  Eigen::VectorXs finiteDifferenceGradient();

  /// This writes the solution back to the output matrix reference passed in
  /// during initialization. This also returns a loss we achieved, which can be
  /// used as a confidence for downstream tasks.
  s_t saveSolutionBackToInitialization();

protected:
  MarkerFitter* mFitter;
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerObservations;
  Eigen::Ref<Eigen::MatrixXs> mOut;
  s_t mSmoothingLoss;

public:
  std::vector<std::string> mActiveMarkers;

  int mNumTimesteps;
  Eigen::MatrixXs mMarkerPositions;
  Eigen::MatrixXi mMarkerObserved;
  Eigen::VectorXs mRadii;
  Eigen::VectorXs mCenterPoints;
  std::string mJointName;
  std::vector<bool> mNewClip;
};

/**
 * This sets up and finds the joint centers using a non-convex sphere-fitting
 * method.
 */
class CylinderFitJointAxisProblem
{
public:
  CylinderFitJointAxisProblem(
      MarkerFitter* fitter,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      Eigen::MatrixXs ikPoses,
      dynamics::Joint* joint,
      Eigen::MatrixXs centers,
      const std::vector<bool>& newClip,
      Eigen::Ref<Eigen::MatrixXs> out);

  int getProblemDim();

  Eigen::VectorXs flatten();

  void unflatten(Eigen::VectorXs x);

  s_t getLoss();

  Eigen::VectorXs getGradient();

  Eigen::VectorXs finiteDifferenceGradient();

  /// This writes the solution back to the output matrix reference passed in
  /// during initialization. This also returns a loss we achieved, which can be
  /// used as a confidence for downstream tasks.
  s_t saveSolutionBackToInitialization();

protected:
  MarkerFitter* mFitter;
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerObservations;
  Eigen::Ref<Eigen::MatrixXs> mOut;
  s_t mKeepCenterLoss;
  s_t mSmoothingCenterLoss;
  s_t mSmoothingAxisLoss;

  std::vector<std::pair<int, int>> mThreadSplits;

public:
  std::vector<std::string> mActiveMarkers;

  int mNumTimesteps;
  Eigen::MatrixXs mJointCenters;
  Eigen::MatrixXs mMarkerPositions;
  Eigen::MatrixXi mMarkerObserved;
  Eigen::VectorXs mPerpendicularRadii;
  Eigen::VectorXs mParallelRadii;
  Eigen::VectorXs mAxisLines;
  const std::vector<bool> mNewClip;
  std::string mJointName;
};

/**
 * Here's some extra, optional params for controlling how initializations
 * happen.
 */
struct InitialMarkerFitParams
{
  std::map<std::string, s_t> markerWeights;
  std::vector<dynamics::Joint*> joints;
  Eigen::MatrixXs jointCenters;
  std::vector<std::vector<std::string>> jointAdjacentMarkers;
  Eigen::VectorXs jointWeights;

  Eigen::MatrixXs jointAxis;
  Eigen::VectorXs axisWeights;

  int numBlocks;
  int numIKTries;
  Eigen::MatrixXs initPoses;

  std::map<std::string, Eigen::Vector3s> markerOffsets;
  Eigen::VectorXs groupScales;
  bool dontRescaleBodies;
  bool dontMoveMarkers;

  int maxTrialsToUseForMultiTrialScaling;
  int maxTimestepsToUseForMultiTrialScaling;

  InitialMarkerFitParams();
  InitialMarkerFitParams(const InitialMarkerFitParams& other);
  InitialMarkerFitParams& setMarkerWeights(
      std::map<std::string, s_t> markerWeights);
  InitialMarkerFitParams& setJointCenters(
      std::vector<dynamics::Joint*> joints,
      Eigen::MatrixXs jointCenters,
      std::vector<std::vector<std::string>> jointAdjacentMarkers);
  InitialMarkerFitParams& setJointCentersAndWeights(
      std::vector<dynamics::Joint*> joints,
      Eigen::MatrixXs jointCenters,
      std::vector<std::vector<std::string>> jointAdjacentMarkers,
      Eigen::VectorXs jointWeights);
  InitialMarkerFitParams& setJointAxisAndWeights(
      Eigen::MatrixXs jointAxis, Eigen::VectorXs axisWeights);
  InitialMarkerFitParams& setNumBlocks(int numBlocks);
  InitialMarkerFitParams& setNumIKTries(int retries);
  InitialMarkerFitParams& setInitPoses(Eigen::MatrixXs initPoses);
  InitialMarkerFitParams& setDontRescaleBodies(bool dontRescaleBodies);
  InitialMarkerFitParams& setDontMoveMarkers(bool dontMoveMarkers);
  InitialMarkerFitParams& setMarkerOffsets(
      std::map<std::string, Eigen::Vector3s> markerOffsets);
  InitialMarkerFitParams& setGroupScales(Eigen::VectorXs groupScales);
  InitialMarkerFitParams& setMaxTrialsToUseForMultiTrialScaling(int numTrials);
  InitialMarkerFitParams& setMaxTimestepsToUseForMultiTrialScaling(
      int numTimesteps);
};

struct ScaleAndFitResult
{
  Eigen::VectorXs pose;
  Eigen::VectorXs scale;
  s_t score;
};

/**
 * This is the high level object that handles fitting skeletons to mocap data.
 *
 * It's supposed to take labeled point trajectories, and a known "marker set"
 * (where markers are, roughly, attached to the body), and use that to figure
 * out the body scales and the marker offsets (marker positions are never
 * perfect) that allow the best IK fit of the data.
 */
class MarkerFitter
{
public:
  MarkerFitter(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markers,
      bool ignoreVirtualJointCenterMarkers = false);

  /// This just checks if there are enough markers in the data with the names
  /// expected by the model. Returns true if there are enough, and false
  /// otherwise.
  bool checkForEnoughMarkers(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  /// This will go through original marker data and attempt to detect common
  /// anomalies, generate warnings to help the user fix their own issues, and
  /// produce fixes where possible.
  std::shared_ptr<MarkersErrorReport> generateDataErrorsReport(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      s_t dt);

  /// After we've finished our initialization, it may become clear that markers
  /// in some of the files should be reversed. This method will do that check,
  /// and if it finds that the markers should be reversed it does the swap,
  /// re-runs IK, and records the result in a new init. If it doesn't find
  /// anything, this is a no-op.
  bool checkForFlippedMarkers(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      MarkerInitialization& initialization,
      std::shared_ptr<MarkersErrorReport> report);

  /// Run the whole pipeline of optimization problems to fit the data as closely
  /// as we can, working on multiple trials at once
  std::vector<MarkerInitialization> runMultiTrialKinematicsPipeline(
      const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
          markerObservationTrials,
      InitialMarkerFitParams params = InitialMarkerFitParams(),
      int numSamples = 20);

  /// Run the whole pipeline of optimization problems to fit the data as closely
  /// as we can
  MarkerInitialization runKinematicsPipeline(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      const std::vector<bool>& newClip,
      InitialMarkerFitParams params = InitialMarkerFitParams(),
      int numSamples = 20,
      bool skipFinalIK = false);

  /// This just finds the joint centers and axis over time.
  MarkerInitialization runJointsPipeline(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      InitialMarkerFitParams params = InitialMarkerFitParams());

  /// This just runs the IK pipeline steps over the given marker observations,
  /// assuming we've got a pre-scaled model. This finds the joint centers and
  /// axis over time, then uses those to run multithreaded IK.
  MarkerInitialization runPrescaledPipeline(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      InitialMarkerFitParams params = InitialMarkerFitParams());

  /// This is a convenience method to display just some manually labeled gold
  /// data, without having to first run the optimizer.
  static void debugGoldTrajectoryAndMarkersToGUI(
      std::shared_ptr<server::GUIWebsocketServer> server,
      C3D* c3d,
      const OpenSimFile* goldSkeleton,
      const Eigen::MatrixXs goldPoses);

  /// This runs a server to display the detailed trajectory information, along
  /// with fit data
  void debugTrajectoryAndMarkersToGUI(
      std::shared_ptr<server::GUIWebsocketServer> server,
      MarkerInitialization init,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      std::vector<ForcePlate> forcePlates = std::vector<ForcePlate>(),
      const OpenSimFile* goldSkeleton = nullptr,
      const Eigen::MatrixXs goldPoses = Eigen::MatrixXs::Zero(0, 0));

  /// This saves a GUI state machine log to display detailed trajectory
  /// information
  void saveTrajectoryAndMarkersToGUI(
      std::string path,
      MarkerInitialization init,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerTrajectories,
      int framesPerSecond,
      std::vector<ForcePlate> forcePlates = std::vector<ForcePlate>(),
      const OpenSimFile* goldSkeleton = nullptr,
      const Eigen::MatrixXs goldPoses = Eigen::MatrixXs::Zero(0, 0));

  /// This automatically finds the "probably correct" rotation for the C3D data
  /// that has no force plate data and rotates the C3D data to match it. This is
  /// determined by which orientation for the data has the skeleton torso
  /// pointed generally upwards most of the time. While this is usually a safe
  /// assumption, it could break down with breakdancing or some other strang
  /// motions, so it should be an option to turn it off.
  void autorotateC3D(C3D* c3d);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 1 and 3: (Re)Initialize scaling+IK
  ///////////////////////////////////////////////////////////////////////////

  /// This finds an initial guess for the body scales and poses, holding
  /// anatomical marker offsets at 0, that we can use for downstream tasks.
  ///
  /// This can multithread over `numBlocks` independent sets of problems.
  MarkerInitialization getInitialization(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      const std::vector<bool>& newClip,
      InitialMarkerFitParams params);

  /// This computes the IK diff for joint positions, given a bunch of weighted
  /// joint centers and also a bunch of weighted joint axis.
  static void computeJointIKDiff(
      Eigen::Ref<Eigen::VectorXs> diff,
      const Eigen::VectorXs& jointPoses,
      const Eigen::VectorXs& jointCenters,
      const Eigen::VectorXs& jointWeights,
      const Eigen::VectorXs& jointAxis,
      const Eigen::VectorXs& axisWeights);

  /// This takes a Jacobian of joint world positions (with respect to anything),
  /// and rescales and reshapes to reflect the weights on joint and axis losses,
  /// as well as the direction for axis losses.
  static void rescaleIKJacobianForWeightsAndAxis(
      Eigen::Ref<Eigen::MatrixXs> jac,
      const Eigen::VectorXs& jointWeights,
      const Eigen::VectorXs& jointAxis,
      const Eigen::VectorXs& axisWeights);

  /// This scales the skeleton and IK fits to the marker observations. It
  /// returns a pair, with (pose, group scales) from the fit.
  static ScaleAndFitResult scaleAndFit(
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
      bool dontScale = false,
      int debugIndex = 0,
      bool debug = false,
      bool saveToGUI = false);

  /// Pipeline step 1, 3, and 5:
  /// This fits IK to the given trajectory, without scaling
  static void fitTrajectory(
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
      bool backwards = false);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 2: Find joint centers
  ///////////////////////////////////////////////////////////////////////////

  /// This solves a bunch of optimization problems, one per joint, to find and
  /// track the joint centers over time. It puts the results back into
  /// `initialization`
  void findJointCenters(
      MarkerInitialization& initialization,
      const std::vector<bool>& newClip,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  /// This finds the trajectory for a single specified joint center over time
  std::shared_ptr<SphereFitJointCenterProblem> findJointCenter(
      std::shared_ptr<SphereFitJointCenterProblem> problem,
      bool logSteps = false);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 3: Find joint axis
  ///////////////////////////////////////////////////////////////////////////

  /// This solves a bunch of optimization problems, one per joint, to find and
  /// track the joint centers over time. It puts the results back into
  /// `initialization`
  void findAllJointAxis(
      MarkerInitialization& initialization,
      const std::vector<bool>& newClip,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  /// This finds the trajectory for a single specified joint axis over time
  std::shared_ptr<CylinderFitJointAxisProblem> findJointAxis(
      std::shared_ptr<CylinderFitJointAxisProblem> problem,
      bool logSteps = false);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 3.5: Weight the joint centers + joint axis
  ///////////////////////////////////////////////////////////////////////////

  /// This computes several metrics, including the variation in the marker
  /// movement for each joint, which then go into computing how much weight we
  /// should put on each joint center / joint axis.
  void computeJointConfidences(
      MarkerInitialization& initialization,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  /// This sets the minimum joint variance allowed before
  /// computeJointConfidences() will cut off a joint as having too low variance
  void setMinJointVarianceCutoff(s_t cutoff);

  /// This sets the value used to compute sphere fit weights
  void setMinSphereFitScore(s_t score);

  /// This sets the value used to compute axis fit weights
  void setMinAxisFitScore(s_t score);

  /// This sets the maximum value that we can weight a joint center in IK
  void setMaxJointWeight(s_t weight);

  /// This sets the maximum value that we can weight a joint axis in IK
  void setMaxAxisWeight(s_t weight);

  /// This sets the value weight used to regularize tracking marker offsets from
  /// where the model thinks they should be
  void setRegularizeTrackingMarkerOffsets(s_t weight);

  /// This sets the value weight used to regularize anatomical marker offsets
  /// from where the model thinks they should be
  void setRegularizeAnatomicalMarkerOffsets(s_t weight);

  /// This sets the value weight used to regularize body scales, to penalize
  /// scalings that result in bodies that are very different along the 3 axis,
  /// like bones that become "fat" in order to not pay a marker regularization
  /// penalty, despite having the correct length.
  void setRegularizeIndividualBodyScales(s_t weight);

  /// This tries to make all bones in the body have the same scale, punishing
  /// outliers.
  void setRegularizeAllBodyScales(s_t weight);

  /// If we've disabled joint limits, this provides the option for a soft
  /// penalty instead
  void setRegularizeJointBounds(s_t weight);

  /// If set to true, we print the pair observation counts and data for
  /// computing joint variability.
  void setDebugJointVariability(bool debug);

  /// This is the default weight that gets assigned to anatomical markers during
  /// IK, if nothing marker-specific gets assigned.
  void setAnatomicalMarkerDefaultWeight(s_t weight);

  /// This is the default weight that gets assigned to tracking markers during
  /// IK, if nothing marker-specific gets assigned.
  void setTrackingMarkerDefaultWeight(s_t weight);

  /// This returns a score summarizing how much the markers attached to this
  /// joint move relative to one another.
  s_t computeJointVariability(
      dynamics::Joint* joint,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 4: Jointly scale+fit+marker offsets, with joint centers as
  // part of the loss term
  ///////////////////////////////////////////////////////////////////////////

  /// This solves an optimization problem, trying to get the Skeleton to match
  /// the markers as closely as possible.
  std::shared_ptr<BilevelFitResult> optimizeBilevel(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      MarkerInitialization& initialization,
      int numSamples,
      bool applyInnerProblemGradientConstraints = true);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 5: Complete the intermittent pose information of the
  // BilevelFitResult by running IK to extend each section.
  ///////////////////////////////////////////////////////////////////////////

  /// The bilevel optimization only picks a subset of poses to fine tune. This
  /// method takes those poses as a starting point, and extends each pose
  /// forward and backwards (half the distance to the next pose to fine tune)
  /// with IK.
  MarkerInitialization completeBilevelResult(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      const std::vector<bool>& newClip,
      std::shared_ptr<BilevelFitResult> result,
      std::vector<dynamics::Joint*> initObservedJoints,
      InitialMarkerFitParams params);

  /// For the multi-trial pipeline, this takes our finished body scales and
  /// marker offsets, and fine tunes on the IK initialized in the early joint
  /// centering process.
  MarkerInitialization fineTuneIK(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      int numBlocks,
      std::map<std::string, s_t> markerWeights,
      MarkerInitialization& initialization);

  /// When our parallel-thread IK finishes, sometimes we can have a bit of
  /// jitter in some of the joints, often around the wrists because the upper
  /// body in general is poorly modelled in OpenSim. This will go through and
  /// smooth out the frame-by-frame jitter.
  MarkerInitialization smoothOutIK(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      const std::vector<bool>& newClip,
      MarkerInitialization& initialization);

  ///////////////////////////////////////////////////////////////////////////
  // Pipeline step 6: Run through and do a linear initialization of masses based
  // on link masses.
  ///////////////////////////////////////////////////////////////////////////

  /// This sets up a bunch of linear constraints based on the motion of each
  /// body, and attempts to solve all the equations with least-squares.
  void initializeMasses(MarkerInitialization& initialization);

  ///////////////////////////////////////////////////////////////////////////
  // Supporting methods
  ///////////////////////////////////////////////////////////////////////////

  /// This lets us pick a subset of the marker observations, to cap the size of
  /// the optimization problem.
  static std::vector<std::map<std::string, Eigen::Vector3s>> pickSubset(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      int maxSize);

  /// All markers are either "anatomical" or "tracking". Markers are presumed to
  /// be anamotical markers unless otherwise specified. Tracking markers are
  /// treated differently - they're not used in the initial scaling and fitting,
  /// and their initial positions are not trusted at all. Instead, during
  /// initialization, we guess their offset based on where the markers are
  /// observed to be.
  void setMarkerIsTracking(std::string marker, bool isTracking = true);

  /// This returns true if the given marker is "tracking", otherwise it's
  /// "anatomical"
  bool getMarkerIsTracking(std::string marker);

  /// This auto-labels any markers whose names end with '1', '2', or '3' as
  /// tracking markers, on the assumption that they're tracking triads.
  void setTriadsToTracking();

  /// If we load a list of tracking markers from the OpenSim file, we can
  void setTrackingMarkers(const std::vector<std::string>& tracking);

  /// Gets the total number of markers we've got in this Fitter
  int getNumMarkers();

  /// Internally all the markers are concatenated together, so each index has a
  /// name.
  std::string getMarkerNameAtIndex(int index);

  /// Internally all the markers are concatenated together, so each index has a
  /// name.
  int getMarkerIndex(std::string name);

  /// This method will set `skeleton` to the configuration given by the vectors
  /// of jointPositions and groupScales. It will also compute and return the
  /// list of markers given by markerDiffs.
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> setConfiguration(
      std::shared_ptr<dynamics::Skeleton>& skeleton,
      Eigen::VectorXs jointPositions,
      Eigen::VectorXs groupScales,
      Eigen::VectorXs markerDiffs);

  /// This computes a vector of concatenated differences between where markers
  /// are and where the observed markers are. Unobserved markers are assumed to
  /// have a difference of zero.
  Eigen::VectorXs getMarkerError(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the overall objective term for the MarkerFitter for a single
  /// timestep. The MarkerFitter is trying to do a bilevel optimization to
  /// minimize this term.
  s_t computeIKLoss(Eigen::VectorXs markerError);

  /// This returns the gradient for the simple IK loss term
  Eigen::VectorXs getIKLossGradWrtMarkerError(Eigen::VectorXs markerError);

  /// This lets us print the components of the loss, to allow easier tuning of
  /// different weights
  void setDebugLoss(bool debug);

  /// During random-restarts on IK, when we find solutions below this loss we'll
  /// stop doing restarts early, to speed up the process.
  void setInitialIKSatisfactoryLoss(s_t loss);

  /// This sets the maximum number of restarts allowed for the initial IK solver
  void setInitialIKMaxRestarts(int restarts);

  /// If true, this processes "single threaded" IK tasks 32 timesteps at a time
  /// (a "warp"), in parallel, using the first timestep of the warp as the
  /// initialization for the whole warp. Defaults to false.
  void setParallelIKWarps(bool parallelWarps);

  /// This gives us a configuration option to ignore the joint limits in the
  /// uploaded model, and then set them after the fit.
  void setIgnoreJointLimits(bool ignore);

  /// Sets the maximum that we'll allow markers to move from their original
  /// position, in meters
  void setMaxMarkerOffset(s_t offset);

  /// Sets the maximum number of iterations for IPOPT
  void setIterationLimit(int limit);

  /// Sets the number of SGD iterations to run when fitting joint center
  /// problems
  void setJointSphereFitSGDIterations(int iters);

  /// Sets the number of SGD iterations to run when fitting joint axis
  /// problems
  void setJointAxisFitSGDIterations(int iters);

  /// This sets an anthropometric prior which is used by the default loss. If
  /// you've called `setCustomLossAndGrad` then this has no effect.
  void setAnthropometricPrior(
      std::shared_ptr<biomechanics::Anthropometrics> prior, s_t weight = 0.001);

  /// This sets the height (in meters) that the model should be scaled to, and
  /// the weight we should use when enforcing that scaling. This is in some
  /// sense redundant to the anthropometric prior, but it's useful to have a
  /// separate term for this, because it allows us to weight the height
  /// constraint more heavily, because it's closer to a "hard constraint" than
  /// the other anthropometric priors.
  void setExplicitHeightPrior(s_t height, s_t weight = 0.001);

  /// This sets the data from a static trial, which we can use to resolve some
  /// forms of pelvis and foot ambiguity.
  void setStaticTrial(
      std::map<std::string, Eigen::Vector3s> markerObservations,
      Eigen::VectorXs pose);

  /// This sets how heavily to weight the static trial in our optimization,
  /// compared to other terms.
  void setStaticTrialWeight(s_t weight);

  /// This sets the minimum distance joints have to be apart in order to get
  /// zero "force field" loss. Any joints closer than this (in world space) will
  /// incur a penalty.
  void setJointForceFieldThresholdDistance(s_t minDistance);

  /// Larger values will increase the softness of the threshold penalty. Smaller
  /// values, as they approach zero, will have an almost perfectly vertical
  /// penality for going below the threshold distance. That would be hard to
  /// optimize, so don't make it too small.
  void setJointForceFieldSoftness(s_t softness);

  /// Sets the loss and gradient function
  void setCustomLossAndGrad(
      std::function<s_t(MarkerFitterState*)> customLossAndGrad);

  /// This adds a custom function as an equality constraint to the problem. The
  /// constraint has to equal 0.
  void addZeroConstraint(
      std::string name,
      std::function<s_t(MarkerFitterState*)> customConstraintAndGrad);

  /// This removes an equality constraint by name
  void removeZeroConstraint(std::string name);

  //////////////////////////////////////////////////////////////////////////
  // First order gradients
  //////////////////////////////////////////////////////////////////////////

  /// This gets the gradient of the objective wrt the joint positions
  static Eigen::VectorXs getMarkerLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the group scales
  static Eigen::VectorXs getMarkerLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the marker offsets
  static Eigen::VectorXs getMarkerLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  //////////////////////////////////////////////////////////////////////////
  // Jacobians of the gradient wrt joints (for bilevel optimization)
  //////////////////////////////////////////////////////////////////////////

  /// Get the marker indices that are not visible
  std::vector<int> getSparsityMap(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs getMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs finiteDifferenceIKLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs getMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs finiteDifferenceIKLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs getMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs
  finiteDifferenceIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  void setLBFGSHistory(int hist);
  void setCheckDerivatives(bool checkDerivatives);

  friend class BilevelFitProblem;
  friend class SphereFitJointCenterProblem;
  friend class CylinderFitJointAxisProblem;
  friend struct MarkerFitterState;

protected:
  std::map<std::string, int> mMarkerIndices;
  std::vector<std::string> mMarkerNames;
  std::vector<bool> mMarkerIsTracking;

  std::mutex mGlobalLock;
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
  dynamics::MarkerMap mMarkerMap;

  std::shared_ptr<dynamics::Skeleton> mSkeletonBallJoints;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      mMarkersBallJoints;

  std::function<s_t(MarkerFitterState*)> mLossAndGrad;
  std::map<std::string, std::function<s_t(MarkerFitterState*)>>
      mZeroConstraints;

  /// This is an optional prior to use when computing default loss, which can
  /// add its log-PDF to standard loss
  std::shared_ptr<biomechanics::Anthropometrics> mAnthropometrics;
  s_t mAnthropometricWeight;

  /// This is an optional prior to use when scaling the skeleton, to ensure that
  /// the height matches what the user expects.
  s_t mHeightPrior;
  s_t mHeightPriorWeight;

  /// This is an optional prior for a static pose trial, which can be used to
  /// help address ambiguity about feet and pelvis offsets
  bool mStaticTrialEnabled;
  std::vector<std::string> mStaticTrialMarkerNames;
  Eigen::VectorXs mStaticTrialMarkerPositions;
  Eigen::VectorXs mStaticTrialPose;

  bool mDebugLoss;
  s_t mInitialIKSatisfactoryLoss;
  int mInitialIKMaxRestarts;
  bool mIgnoreJointLimits;
  s_t mMaxMarkerOffset;
  bool mUseParallelIKWarps;

  // Parameters for joint weighting
  s_t mMinVarianceCutoff;
  s_t mMinSphereFitScore;
  s_t mMinAxisFitScore;
  s_t mMaxJointWeight;
  s_t mMaxAxisWeight;
  bool mDebugJointVariability;
  s_t mRegularizeTrackingMarkerOffsets;
  s_t mRegularizeAnatomicalMarkerOffsets;
  s_t mRegularizeIndividualBodyScales;
  s_t mRegularizeAllBodyScales;
  s_t mRegularizeJointBounds;
  s_t mAnatomicalMarkerDefaultWeight;
  s_t mTrackingMarkerDefaultWeight;
  s_t mStaticTrialWeight;
  s_t mJointForceFieldThresholdDistance;
  s_t mJointForceFieldSoftness;

  // These are IPOPT settings
  double mTolerance;
  int mIterationLimit;
  int mLBFGSHistoryLength;
  bool mCheckDerivatives;
  int mPrintFrequency;
  bool mSilenceOutput;
  bool mDisableLinesearch;

  int mJointSphereFitSGDIterations;
  int mJointAxisFitSGDIterations;
};

/*
 * Reminder: IPOPT will want to free this object when it's done with
 * optimization. This is responsible for actually transcribing the problem into
 * a format IPOpt can work with.
 */
class BilevelFitProblem : public Ipopt::TNLP
{
public:
  /// This creates a problem object. We take as arguments:
  /// @param skeleton the skeleton we're going to use to scale + fit the data
  /// @param markerSet the marker set we're using, with default offsets from the
  /// skeleton
  /// @param markerObservations a list of timesteps, where each timestep
  /// observes some subset of the markers at some points in 3D space.
  BilevelFitProblem(
      MarkerFitter* fitter,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      MarkerInitialization& initialization,
      int numSamples,
      bool applyInnerProblemGradientConstraints,
      std::shared_ptr<BilevelFitResult>& outResult);

  ~BilevelFitProblem();

  int getProblemSize();

  /// This gets a decent initial guess for the problem. We can guess scaling and
  /// joint positions from the first marker observation, and then use that
  /// scaling to get joint positions for all the other entries. This initially
  /// satisfies the constraint that we remain at optimal positional IK
  /// throughout optimization.
  Eigen::VectorXs getInitialization();

  /// This evaluates our loss function given a concatenated vector of all the
  /// problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  s_t getLoss(Eigen::VectorXs x);

  /// This evaluates our gradient of loss given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs getGradient(Eigen::VectorXs x);

  /// This evaluates our gradient of loss given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs finiteDifferenceGradient(Eigen::VectorXs x);

  /// This evaluates our constraint vector given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs getConstraints(Eigen::VectorXs x);

  /// This evaluates the Jacobian of our constraint vector wrt x given a
  /// concatenated vector of all the problem state: [groupSizes, markerOffsets,
  /// q_0, ..., q_N]
  Eigen::MatrixXs getConstraintsJacobian(Eigen::VectorXs x);

  /// This evaluates the Jacobian of our constraint vector wrt x given a
  /// concatenated vector of all the problem state: [groupSizes, markerOffsets,
  /// q_0, ..., q_N]
  Eigen::MatrixXs finiteDifferenceConstraintsJacobian(Eigen::VectorXs x);

  /// This returns the indices that this problem is using to specify the problem
  const std::vector<int>& getSampleIndices();

  /// This returns the marker map observations that this problem is using to
  /// specify the problem
  const std::vector<std::map<std::string, Eigen::Vector3s>>&
  getMarkerMapObservations();

  /// This returns the marker observations that this problem is using to specify
  /// the problem
  const std::vector<std::vector<std::pair<int, Eigen::Vector3s>>>&
  getMarkerObservations();

  /// This returns the subset of joint centers, for the selected timestep
  /// samples
  const Eigen::MatrixXs& getJointCenters();

  //------------------------- Ipopt::TNLP --------------------------------------
  /// \brief Method to return some info about the nlp
  bool get_nlp_info(
      Ipopt::Index& n,
      Ipopt::Index& m,
      Ipopt::Index& nnz_jac_g,
      Ipopt::Index& nnz_h_lag,
      Ipopt::TNLP::IndexStyleEnum& index_style) override;

  /// \brief Method to return the bounds for my problem
  bool get_bounds_info(
      Ipopt::Index n,
      Ipopt::Number* x_l,
      Ipopt::Number* x_u,
      Ipopt::Index m,
      Ipopt::Number* g_l,
      Ipopt::Number* g_u) override;

  /// \brief Method to return the starting point for the algorithm
  bool get_starting_point(
      Ipopt::Index n,
      bool init_x,
      Ipopt::Number* x,
      bool init_z,
      Ipopt::Number* z_L,
      Ipopt::Number* z_U,
      Ipopt::Index m,
      bool init_lambda,
      Ipopt::Number* lambda) override;

  /// \brief Method to return the objective value
  bool eval_f(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Number& _obj_value) override;

  /// \brief Method to return the gradient of the objective
  bool eval_grad_f(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Number* _grad_f) override;

  /// \brief Method to return the constraint residuals
  bool eval_g(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Index _m,
      Ipopt::Number* _g) override;

  /// \brief Method to return:
  ///        1) The structure of the jacobian (if "values" is nullptr)
  ///        2) The values of the jacobian (if "values" is not nullptr)
  bool eval_jac_g(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Index _m,
      Ipopt::Index _nele_jac,
      Ipopt::Index* _iRow,
      Ipopt::Index* _jCol,
      Ipopt::Number* _values) override;

  /// \brief Method to return:
  ///        1) The structure of the hessian of the lagrangian (if "values" is
  ///           nullptr)
  ///        2) The values of the hessian of the lagrangian (if "values" is not
  ///           nullptr)
  bool eval_h(
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
      Ipopt::Number* _values) override;

  /// \brief This method is called when the algorithm is complete so the TNLP
  ///        can store/write the solution
  void finalize_solution(
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
      Ipopt::IpoptCalculatedQuantities* _ip_cq) override;

  bool intermediate_callback(
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
      Ipopt::IpoptCalculatedQuantities* ip_cq) override;

protected:
  MarkerFitter* mFitter;
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerMapObservations;
  std::vector<std::vector<std::pair<int, Eigen::Vector3s>>> mMarkerObservations;
  Eigen::MatrixXs mJointCenters;
  Eigen::VectorXs mJointWeights;
  Eigen::MatrixXs mJointAxis;
  Eigen::VectorXs mAxisWeights;
  std::vector<int> mSampleIndices;
  MarkerInitialization& mInitialization;
  bool mApplyInnerProblemGradientConstraints;
  Eigen::VectorXs mObservationWeights;
  std::shared_ptr<BilevelFitResult>& mOutResult;

  int mBestObjectiveValueIteration;
  s_t mBestObjectiveValue;
  Eigen::VectorXs mLastX;
  Eigen::VectorXs mBestObjectiveValueState;

  // Thread state

  int mNumThreads;
  std::vector<std::vector<int>> mPerThreadIndices;
  std::vector<std::vector<int>> mPerThreadCursor;
  std::vector<std::shared_ptr<dynamics::Skeleton>> mPerThreadSkeletons;
};

} // namespace biomechanics

} // namespace dart

#endif