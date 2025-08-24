#ifndef BIOMECH_SUBJECT_ON_DISK
#define BIOMECH_SUBJECT_ON_DISK

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/proto/SubjectOnDisk.pb.h"

namespace dart {
namespace biomechanics {

class SubjectOnDiskHeader;

struct FramePass
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ProcessingPassType type;
  s_t markerRMS;
  s_t markerMax;
  s_t linearResidual;
  s_t angularResidual;
  Eigen::VectorXd pos;
  Eigen::VectorXd vel;
  Eigen::VectorXd acc;
  Eigen::VectorXd tau;
  // These are boolean values (0 or 1) for each contact body indicating whether
  // or not it's in contact
  Eigen::VectorXi contact;
  // These are each 6-vector of contact body wrenches, all concatenated together
  Eigen::VectorXd groundContactWrenches;
  // These are each 3-vector for each contact body, concatenated together
  Eigen::VectorXd groundContactCenterOfPressure;
  Eigen::VectorXd groundContactTorque;
  Eigen::VectorXd groundContactForce;
  // These are the center of mass kinematics
  Eigen::Vector3s comPos;
  Eigen::Vector3s comVel;
  Eigen::Vector3s comAcc;
  Eigen::Vector3s comAccInRootFrame;
  // These are each 6-vectors of the contact wrench of each body, but expressed
  // in the world frame, all concatenated together
  Eigen::VectorXd groundContactWrenchesInRootFrame;
  // These are each 3-vector for each contact body, concatenated together
  Eigen::VectorXd groundContactCenterOfPressureInRootFrame;
  Eigen::VectorXd groundContactTorqueInRootFrame;
  Eigen::VectorXd groundContactForceInRootFrame;
  // This is the residual, expressed as a wrench in the root body (probably the
  // pelvis) frame
  Eigen::Vector6d residualWrenchInRootFrame;
  // These are the joint centers, expressed in the world frame
  Eigen::VectorXd jointCenters;
  // These are the joint centers, expressed in the root body (probably the
  // pelvis) frame
  Eigen::VectorXd jointCentersInRootFrame;
  // These are masks for which DOFs are observed
  Eigen::VectorXi posObserved;
  // These are masks for which DOFs have been finite differenced (if they
  // haven't been finite differenced, they're from real sensors and therefore
  // more trustworthy)
  Eigen::VectorXi velFiniteDifferenced;
  Eigen::VectorXi accFiniteDifferenced;

  Eigen::Vector3s rootLinearVelInRootFrame;
  Eigen::Vector3s rootAngularVelInRootFrame;
  Eigen::Vector3s rootLinearAccInRootFrame;
  Eigen::Vector3s rootAngularAccInRootFrame;
  Eigen::VectorXs rootPosHistoryInRootFrame;
  Eigen::VectorXs rootEulerHistoryInRootFrame;

  void readFromProto(
      dart::proto::SubjectOnDiskProcessingPassFrame* proto,
      const SubjectOnDiskHeader& header,
      int trial,
      int t,
      int pass,
      s_t contactThreshold);
};

struct Frame
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int trial;
  int t;
  MissingGRFReason missingGRFReason;
  // Each processing pass has its own set of kinematics and dynamics, as the
  // model and trajectories are adjusted
  std::vector<std::shared_ptr<FramePass>> processingPasses;

  ///////////////////////////////////////////////////////////////////////////
  // Raw sensor data
  ///////////////////////////////////////////////////////////////////////////

  // We include this to allow the binary format to store/load a bunch of new
  // types of values while remaining backwards compatible.
  std::vector<std::pair<std::string, Eigen::VectorXd>> customValues;
  // These are the marker observations
  std::vector<std::pair<std::string, Eigen::Vector3s>> markerObservations;
  // These are the accelerometer observations
  std::vector<std::pair<std::string, Eigen::Vector3s>> accObservations;
  // These are the gyroscope observations
  std::vector<std::pair<std::string, Eigen::Vector3s>> gyroObservations;
  // These are the raw values recorded from the force plates, without any
  // post-processing or aggregation applied
  std::vector<Eigen::Vector3s> rawForcePlateCenterOfPressures;
  std::vector<Eigen::Vector3s> rawForcePlateTorques;
  std::vector<Eigen::Vector3s> rawForcePlateForces;
  // These are the EMG signals, where the signal is represented as a vector that
  // can vary in length depending on how much faster the EMG sampling was than
  // the motion capture sampling.
  std::vector<std::pair<std::string, Eigen::VectorXs>> emgSignals;
  // These are the torques from the exo, along with the index of the DOFs
  // they are applied to
  std::vector<std::pair<int, s_t>> exoTorques;

  void readSensorsFromProto(
      dart::proto::SubjectOnDiskSensorFrame* proto,
      const SubjectOnDiskHeader& header,
      int trial,
      int t);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Builders, to create a SubjectOnDisk from scratch
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SubjectOnDiskTrialPass
{
public:
  SubjectOnDiskTrialPass();
  void setType(ProcessingPassType type);
  ProcessingPassType getType();
  void setDofPositionsObserved(std::vector<bool> dofPositionsObserved);
  void setDofVelocitiesFiniteDifferenced(
      std::vector<bool> dofVelocitiesFiniteDifferenced);
  void setDofAccelerationFiniteDifferenced(
      std::vector<bool> dofAccelerationFiniteDifference);
  void setMarkerRMS(std::vector<s_t> markerRMS);
  std::vector<s_t> getMarkerRMS();
  void setMarkerMax(std::vector<s_t> markerMax);
  std::vector<s_t> getMarkerMax();
  // If we're doing a lowpass filter on this pass, then what was the cutoff
  // frequency of that filter?
  void setLowpassCutoffFrequency(s_t freq);
  // If we're doing a lowpass filter on this pass, then what was the order of
  // that (Butterworth) filter?
  void setLowpassFilterOrder(int order);
  // If we filtered the force plates, then what was the cutoff frequency of that
  // filtering?
  void setForcePlateCutoffs(std::vector<s_t> cutoffs);
  // If we filtered the position data with an acceleration minimizing filter,
  // then what was the regularization weight that tracked the original position.
  void setAccelerationMinimizingRegularization(s_t regularization);
  // If we filtered the position data with an acceleration minimizing filter,
  // then what was the regularization weight that tracked the original force
  // data
  void setAccelerationMinimizingForceRegularization(s_t weight);

  // This is for allowing the user to set all the values of a pass at once,
  // without having to manually compute them in Python, which turns out to be
  // slow and difficult to test.
  void computeValues(
      std::shared_ptr<dynamics::Skeleton> skel,
      s_t timestep,
      Eigen::MatrixXs poses,
      std::vector<std::string> footBodies,
      // If we've already assigned the force plates to feet
      Eigen::MatrixXs forces,
      Eigen::MatrixXs moments,
      Eigen::MatrixXs cops,
      // How much history to use for the root position and orientation
      int rootHistoryLen = 5,
      int rootHistoryStride = 1);

  // This is for allowing the user to set all the kinematic values of a pass at
  // once. All dynamics values are set to zero.
  void computeKinematicValues(
      std::shared_ptr<dynamics::Skeleton> skel,
      s_t timestep,
      Eigen::MatrixXs poses,
      // How much history to use for the root position and orientation
      int rootHistoryLen = 5,
      int rootHistoryStride = 1,
      Eigen::MatrixXs explicitVels = Eigen::MatrixXs::Zero(0, 0),
      Eigen::MatrixXs explicitAccs = Eigen::MatrixXs::Zero(0, 0));

  // This is for allowing the user to set all the values of a pass at once,
  // without having to manually compute them in Python, which turns out to be
  // slow and difficult to test.
  void computeValuesFromForcePlates(
      std::shared_ptr<dynamics::Skeleton> skel,
      s_t timestep,
      Eigen::MatrixXs poses,
      std::vector<std::string> footBodies,
      std::vector<ForcePlate> forcePlates,
      // How much history to use for the root position and orientation
      int rootHistoryLen = 5,
      int rootHistoryStride = 1,
      Eigen::MatrixXs explicitVels = Eigen::MatrixXs::Zero(0, 0),
      Eigen::MatrixXs explicitAccs = Eigen::MatrixXs::Zero(0, 0),
      s_t forcePlateZeroThresholdNewtons = 3.0);

  // Manual setters (and getters) that compete with computeValues()
  void setLinearResidual(std::vector<s_t> linearResidual);
  std::vector<s_t> getLinearResidual();
  void setAngularResidual(std::vector<s_t> angularResidual);
  std::vector<s_t> getAngularResidual();
  void setPoses(Eigen::MatrixXs poses);
  Eigen::MatrixXs getPoses();
  void setVels(Eigen::MatrixXs vels);
  Eigen::MatrixXs getVels();
  void setAccs(Eigen::MatrixXs accs);
  Eigen::MatrixXs getAccs();
  void setTaus(Eigen::MatrixXs taus);
  Eigen::MatrixXs getTaus();
  void setGroundBodyWrenches(Eigen::MatrixXs wrenches);
  Eigen::MatrixXs getGroundBodyWrenches();
  void setGroundBodyCopTorqueForce(Eigen::MatrixXs copTorqueForces);
  Eigen::MatrixXs getGroundBodyCopTorqueForce();
  void setComPoses(Eigen::MatrixXs poses);
  Eigen::MatrixXs getComPoses();
  void setComVels(Eigen::MatrixXs vels);
  Eigen::MatrixXs getComVels();
  void setComAccs(Eigen::MatrixXs accs);
  Eigen::MatrixXs getComAccs();
  void setComAccsInRootFrame(Eigen::MatrixXs accs);
  Eigen::MatrixXs getComAccsInRootFrame();
  void setResidualWrenchInRootFrame(Eigen::MatrixXs wrenches);
  Eigen::MatrixXs getResidualWrenchInRootFrame();
  void setGroundBodyWrenchesInRootFrame(Eigen::MatrixXs wrenches);
  Eigen::MatrixXs getGroundBodyWrenchesInRootFrame();
  void setGroundBodyCopTorqueForceInRootFrame(Eigen::MatrixXs copTorqueForces);
  Eigen::MatrixXs getGroundBodyCopTorqueForceInRootFrame();
  void setJointCenters(Eigen::MatrixXs centers);
  Eigen::MatrixXs getJointCenters();
  void setJointCentersInRootFrame(Eigen::MatrixXs centers);
  Eigen::MatrixXs getJointCentersInRootFrame();
  void setRootSpatialVelInRootFrame(Eigen::MatrixXs spatialVel);
  Eigen::MatrixXs getRootSpatialVelInRootFrame();
  void setRootSpatialAccInRootFrame(Eigen::MatrixXs spatialAcc);
  Eigen::MatrixXs getRootSpatialAccInRootFrame();
  void setRootPosHistoryInRootFrame(Eigen::MatrixXs rootHistory);
  Eigen::MatrixXs getRootPosHistoryInRootFrame();
  void setRootEulerHistoryInRootFrame(Eigen::MatrixXs rootHistory);
  Eigen::MatrixXs getRootEulerHistoryInRootFrame();

  // This gets the data from `getGroundBodyCopTorqueForce()` in the form of
  // ForcePlate objects, which are easier to work with.
  std::vector<ForcePlate> getProcessedForcePlates();

  // This will return a matrix where every one of our properties with setters is
  // stacked together vertically. Each column represents time, and each row is a
  // different property of interest. The point here is not to introspect into
  // the individual rows, but to have a convenient object that we can resample
  // to a new timestep length, and possibly lowpass filter.
  Eigen::MatrixXs getResamplingMatrix();
  // This is the setter for the matrix you get from `getResamplingMatrix()`,
  // after you've finished modifying it.
  void setResamplingMatrix(Eigen::MatrixXs matrix);

  void read(const proto::SubjectOnDiskTrialProcessingPassHeader& proto);
  void write(proto::SubjectOnDiskTrialProcessingPassHeader* proto);

  void copyValuesFrom(std::shared_ptr<SubjectOnDiskTrialPass> other);

protected:
  // This data is included in the header
  ProcessingPassType mType;
  std::vector<bool> mDofPositionsObserved;
  std::vector<bool> mDofVelocitiesFiniteDifferenced;
  std::vector<bool> mDofAccelerationFiniteDifferenced;
  std::vector<s_t> mMarkerRMS;
  std::vector<s_t> mMarkerMax;
  // If we're doing a lowpass filter on this pass, then what was the cutoff
  // frequency of that filter?
  s_t mLowpassCutoffFrequency;
  // If we're doing a lowpass filter on this pass, then what was the order of
  // that (Butterworth) filter?
  int mLowpassFilterOrder;
  // If we filtered position with an acceleration minimizing filter, then this
  // is the regularization weight that tracked the original position.
  s_t mAccelerationMinimizingRegularization;
  // If we filtered position with an acceleration minimizing filter, then this
  // is the regularization weight that tracked the original force data
  s_t mAccelerationMinimizingForceRegularization;
  // If we filtered the force plates, then what was the cutoff frequency of that
  // filtering?
  std::vector<s_t> mForcePlateCutoffs;
  std::vector<s_t> mLinearResidual;
  std::vector<s_t> mAngularResidual;
  // This data is in each separate Frame, and so won't be loaded from the proto
  Eigen::MatrixXs mPos;
  Eigen::MatrixXs mVel;
  Eigen::MatrixXs mAcc;
  Eigen::MatrixXs mTaus;
  Eigen::MatrixXs mGroundBodyWrenches;
  Eigen::MatrixXs mGroundBodyCopTorqueForce;
  Eigen::MatrixXs mComPoses;
  Eigen::MatrixXs mComVels;
  Eigen::MatrixXs mComAccs;
  Eigen::MatrixXs mComAccsInRootFrame;
  Eigen::MatrixXs mResidualWrenchInRootFrame;
  Eigen::MatrixXs mGroundBodyWrenchesInRootFrame;
  Eigen::MatrixXs mGroundBodyCopTorqueForceInRootFrame;
  Eigen::MatrixXs mJointCenters;
  Eigen::MatrixXs mJointCentersInRootFrame;
  Eigen::MatrixXs mRootSpatialVelInRootFrame;
  Eigen::MatrixXs mRootSpatialAccInRootFrame;
  Eigen::MatrixXs mRootPosHistoryInRootFrame;
  Eigen::MatrixXs mRootEulerHistoryInRootFrame;
  // This is for allowing the user to pre-filter out data where joint velocities
  // are above a certain "unreasonable limit", like 50 rad/s or so
  std::vector<s_t> mJointsMaxVelocity;

  friend struct Frame;
  friend struct FramePass;
  friend class SubjectOnDisk;
  friend class SubjectOnDiskHeader;
};

class SubjectOnDiskTrial
{
public:
  SubjectOnDiskTrial();
  void setName(const std::string& name);
  const std::string& getName() const;
  void setTimestep(s_t timestep);
  s_t getTimestep();
  void setTrialLength(int length);
  int getTrialLength();
  void setTrialTags(std::vector<std::string> trialTags);
  std::string getOriginalTrialName();
  void setOriginalTrialName(const std::string& name);
  int getSplitIndex();
  void setSplitIndex(int split);
  int getOriginalTrialStartFrame();
  void setOriginalTrialStartFrame(int startFrame);
  int getOriginalTrialEndFrame();
  void setOriginalTrialEndFrame(int endFrame);
  s_t getOriginalTrialStartTime();
  void setOriginalTrialStartTime(s_t startTime);
  s_t getOriginalTrialEndTime();
  void setOriginalTrialEndTime(s_t endTime);
  std::vector<MissingGRFReason> getMissingGRFReason();
  void setMissingGRFReason(std::vector<MissingGRFReason> missingGRFReason);
  std::vector<bool> getHasManualGRFAnnotation();
  void setHasManualGRFAnnotation(std::vector<bool> hasManualGRFAnnotation);
  void setCustomValues(std::vector<Eigen::MatrixXs> customValues);
  void setMarkerNamesGuessed(bool markersGuessed);
  std::vector<std::map<std::string, Eigen::Vector3s>> getMarkerObservations();
  void setMarkerObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations);
  void setAccObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> accObservations);
  void setGyroObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> gyroObservations);
  void setEmgObservations(
      std::vector<std::map<std::string, Eigen::VectorXs>> emgObservations);
  void setExoTorques(std::map<int, Eigen::VectorXs> exoTorques);
  void setForcePlates(std::vector<ForcePlate> forcePlates);
  std::vector<ForcePlate> getForcePlates();
  void setBasicTrialType(BasicTrialType type);
  BasicTrialType getBasicTrialType();
  void setDetectedTrialFeatures(std::vector<DetectedTrialFeature> features);
  std::vector<DetectedTrialFeature> getDetectedTrialFeatures();
  std::shared_ptr<SubjectOnDiskTrialPass> addPass();
  std::vector<std::shared_ptr<SubjectOnDiskTrialPass>> getPasses();
  void read(const proto::SubjectOnDiskTrialHeader& proto);
  void write(proto::SubjectOnDiskTrialHeader* proto);

protected:
  std::string mName;
  s_t mTimestep;
  int mLength;
  std::vector<std::string> mTrialTags;
  std::vector<std::shared_ptr<SubjectOnDiskTrialPass>> mTrialPasses;
  std::vector<MissingGRFReason> mMissingGRFReason;
  std::vector<bool> mHasManualGRFAnnotation;
  // This is true if we guessed the marker names, and false if we got them from
  // the uploaded user's file, which implies that they got them from human
  // observations.
  bool mMarkerNamesGuessed;
  std::string mOriginalTrialName;
  int mSplitIndex;

  int mOriginalTrialStartFrame;
  int mOriginalTrialEndFrame;
  s_t mOriginalTrialStartTime;
  s_t mOriginalTrialEndTime;

  BasicTrialType mBasicTrialType;
  std::vector<DetectedTrialFeature> mDetectedTrialFeatures;

  ///////////////////////////////////////////////////////////////////////////
  // Recovered proto summaries, for incremental loading of Frames
  ///////////////////////////////////////////////////////////////////////////

  int mNumForcePlates;
  std::vector<std::vector<Eigen::Vector3s>> mForcePlateCorners;

  ///////////////////////////////////////////////////////////////////////////
  // Raw sensor observations to write out frame by frame
  ///////////////////////////////////////////////////////////////////////////

  std::vector<Eigen::MatrixXs> mCustomValues;
  // These are the markers, gyros, accelerometers
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerObservations;
  std::vector<std::map<std::string, Eigen::Vector3s>> mAccObservations;
  std::vector<std::map<std::string, Eigen::Vector3s>> mGyroObservations;
  // These are EMG observations, with potentially many samples per frame
  std::vector<std::map<std::string, Eigen::VectorXs>> mEmgObservations;
  // These are the torques applied by an exoskeleton, if any, per DOF they are
  // applied to
  std::map<int, Eigen::VectorXs> mExoTorques;
  // This is raw force plate data
  std::vector<ForcePlate> mForcePlates;

  friend class SubjectOnDiskHeader;
  friend class SubjectOnDisk;
  friend struct Frame;
  friend struct FramePass;
};

class SubjectOnDiskPassHeader
{
public:
  SubjectOnDiskPassHeader();
  void setProcessingPassType(ProcessingPassType type);
  ProcessingPassType getProcessingPassType();
  void setOpenSimFileText(const std::string& openSimFileText);
  std::string getOpenSimFileText();
  void write(dart::proto::SubjectOnDiskPass* proto);
  void read(const dart::proto::SubjectOnDiskPass& proto);

protected:
  ProcessingPassType mType;
  // The OpenSim file XML gets copied into our binary bundle, along with
  // any necessary Geometry files
  std::string mOpenSimFileText;

  friend class SubjectOnDisk;
  friend struct FramePass;
};

class SubjectOnDiskHeader
{
public:
  SubjectOnDiskHeader();
  SubjectOnDiskHeader& setNumDofs(int dofs);
  SubjectOnDiskHeader& setNumJoints(int joints);
  SubjectOnDiskHeader& setGroundForceBodies(
      std::vector<std::string> groundForceBodies);
  SubjectOnDiskHeader& setCustomValueNames(
      std::vector<std::string> customValueNames);
  SubjectOnDiskHeader& setBiologicalSex(const std::string& biologicalSex);
  SubjectOnDiskHeader& setHeightM(double heightM);
  SubjectOnDiskHeader& setMassKg(double massKg);
  SubjectOnDiskHeader& setAgeYears(int ageYears);
  SubjectOnDiskHeader& setSubjectTags(std::vector<std::string> subjectTags);
  SubjectOnDiskHeader& setHref(const std::string& sourceHref);
  SubjectOnDiskHeader& setNotes(const std::string& notes);
  SubjectOnDiskHeader& setQuality(DataQuality quality);
  DataQuality getQuality();
  std::shared_ptr<SubjectOnDiskPassHeader> addProcessingPass();
  std::vector<std::shared_ptr<SubjectOnDiskPassHeader>> getProcessingPasses();
  std::shared_ptr<SubjectOnDiskTrial> addTrial();
  std::vector<std::shared_ptr<SubjectOnDiskTrial>> getTrials();
  void filterTrials(std::vector<bool> keepTrials);
  void trimToProcessingPasses(int numPasses);
  void setTrials(std::vector<std::shared_ptr<SubjectOnDiskTrial>> trials);
  void recomputeColumnNames();
  void write(dart::proto::SubjectOnDiskHeader* proto);
  void read(const dart::proto::SubjectOnDiskHeader& proto);
  void writeSensorsFrame(
      dart::proto::SubjectOnDiskSensorFrame* proto,
      int trial,
      int t,
      int maxNumForcePlates);
  void writeProcessingPassFrame(
      dart::proto::SubjectOnDiskProcessingPassFrame* proto,
      int trial,
      int t,
      int pass);

protected:
  // How many DOFs are in the skeleton
  int mNumDofs;
  // How many joints are in the skeleton
  int mNumJoints;
  // The passes we applied to this data, along with the result skeletons that
  // were generated by each pass.
  std::vector<std::shared_ptr<SubjectOnDiskPassHeader>> mPasses;
  // These are generalized 6-dof wrenches applied to arbitrary bodies
  // (generally by foot-ground contact, though other things too)
  std::vector<std::string> mGroundContactBodies;
  // We include this to allow the binary format to store/load a bunch of new
  // types of values while remaining backwards compatible.
  std::vector<std::string> mCustomValueNames;
  std::vector<int> mCustomValueLengths;
  // This is the subject info
  std::string mBiologicalSex;
  double mHeightM;
  double mMassKg;
  int mAgeYears;
  // The provenance info, optional, for investigating where training data
  // came from after its been aggregated
  std::vector<std::string> mSubjectTags;
  std::string mHref = "";
  std::string mNotes = "";
  // These are the trials, which contain the actual data
  std::vector<std::shared_ptr<SubjectOnDiskTrial>> mTrials;

  // These are the marker, accelerometer and gyroscope names
  std::vector<std::string> mMarkerNames;
  std::vector<std::string> mAccNames;
  std::vector<std::string> mGyroNames;
  // This is EMG data
  std::vector<std::string> mEmgNames;
  int mEmgDim;
  // This is exoskeleton data
  std::vector<int> mExoDofIndices;

  // This is the user supplied quality of the data
  DataQuality mDataQuality;

  friend class SubjectOnDisk;
  friend struct Frame;
  friend struct FramePass;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the SubjectOnDisk object
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This is for doing ML and large-scale data analysis. The idea here is to
 * create a lazy-loadable view of a subject, where everything remains on disk
 * until asked for. That way we can instantiate thousands of these in memory,
 * and not worry about OOM'ing a machine.
 */
class SubjectOnDisk
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SubjectOnDisk(const std::string& path);

  SubjectOnDisk(std::shared_ptr<SubjectOnDiskHeader> header);

  /// This will write a B3D file to disk
  static void writeB3D(
      const std::string& path, std::shared_ptr<SubjectOnDiskHeader> header);

  /// This loads all the frames of data, and fills in the processing pass data
  /// matrices in the proto header classes.
  void loadAllFrames(bool doNotStandardizeForcePlateData = false);

  bool hasLoadedAllFrames();

  /// This returns the raw proto header for this subject, which can be used to
  /// write out a new B3D file
  std::shared_ptr<SubjectOnDiskHeader> getHeaderProto();

  /// This reads all the raw sensor data for this trial, and constructs
  /// force plates.
  std::vector<ForcePlate> readForcePlates(int trial);

  /// This will read the skeleton from the binary, and optionally use the passed
  /// in Geometry folder.
  std::shared_ptr<dynamics::Skeleton> readSkel(
      int processingPass,
      std::string geometryFolder = "",
      bool ignoreGeometry = false);

  /// If you want the skeleton _and the markerset_ from the binary, use this
  /// instead of readSkel() to get full parsed OpenSimFile object, which
  /// includes the markerset.
  OpenSimFile readOpenSimFile(
      int processingPass,
      std::string geometryFolder = "",
      bool ignoreGeometry = false);

  /// This will read the raw OpenSim XML file text out of the binary, and return
  /// it as a string
  std::string getOpensimFileText(int processingPass);

  // If we're doing a lowpass filter on this pass, then what was the cutoff
  // frequency of that filter?
  s_t getLowpassCutoffFrequency(int trial, int processingPass);

  // If we're doing a lowpass filter on this pass, then what was the order of
  // that (Butterworth) filter?
  int getLowpassFilterOrder(int trial, int processingPass);

  // If we reprocessed the force plates with a cutoff, then these are the cutoff
  // values we used.
  std::vector<s_t> getForceplateCutoffs(int trial, int processingPass);

  /// This will read from disk and allocate a number of Frame objects.
  /// These Frame objects are assumed to be
  /// short-lived, to save working memory.
  ///
  /// On OOB access, prints an error and returns an empty vector.
  std::vector<std::shared_ptr<Frame>> readFrames(
      int trial,
      int startFrame,
      int numFramesToRead = 1,
      bool includeSensorData = true,
      bool includeProcessingPasses = true,
      int stride = 1,
      s_t contactThreshold = 1.0);

  /// This returns the number of trials on the subject
  int getNumTrials();

  /// This returns the length of the trial
  int getTrialLength(int trial);

  /// This returns the original name of the trial before it was (potentially)
  /// split into multiple pieces
  std::string getTrialOriginalName(int trial);

  /// This returns the index of the split, if this trial was the result of
  /// splitting an original trial into multiple pieces
  int getTrialSplitIndex(int trial);

  /// This returns the number of processing passes in the trial
  int getTrialNumProcessingPasses(int trial);

  /// This returns the timestep size for the trial
  s_t getTrialTimestep(int trial);

  /// This returns the number of DOFs for the model on this Subject
  int getNumDofs();

  /// This returns the number of joints for the model on this Subject
  int getNumJoints();

  /// This returns the vector of enums of type 'MissingGRFReason', which can
  /// include `notMissingGRF`.
  std::vector<MissingGRFReason> getMissingGRF(int trial);

  /// This returns the user supplied enum of type 'DataQuality'
  DataQuality getQuality();

  int getNumProcessingPasses();

  ProcessingPassType getProcessingPassType(int processingPass);

  std::vector<bool> getDofPositionsObserved(int trial, int processingPass);

  std::vector<bool> getDofVelocitiesFiniteDifferenced(
      int trial, int processingPass);

  std::vector<bool> getDofAccelerationsFiniteDifferenced(
      int trial, int processingPass);

  std::vector<s_t> getTrialLinearResidualNorms(int trial, int processingPass);
  std::vector<s_t> getTrialAngularResidualNorms(int trial, int processingPass);
  std::vector<s_t> getTrialMarkerRMSs(int trial, int processingPass);
  std::vector<s_t> getTrialMarkerMaxs(int trial, int processingPass);

  /// This returns the maximum absolute velocity of any DOF at each timestep for
  /// a given trial
  std::vector<s_t> getTrialMaxJointVelocity(int trial, int processingPass);

  /// This returns the list of contact body names for this Subject
  std::vector<std::string> getGroundForceBodies();

  /// This returns the list of custom value names stored in this subject
  std::vector<std::string> getCustomValues();

  /// This returns the dimension of the custom value specified by `valueName`
  int getCustomValueDim(std::string valueName);

  /// The name of the trial, if provided, or else an empty string
  std::string getTrialName(int trial);

  std::string getBiologicalSex();

  double getHeightM();

  double getMassKg();

  /// This gets the tags associated with the subject, if there are any.
  std::vector<std::string> getSubjectTags();

  /// This gets the tags associated with the trial, if there are any.
  std::vector<std::string> getTrialTags(int trial);

  int getAgeYears();

  /// This returns the number of raw force plates that were used to generate the
  /// data, for this trial
  int getNumForcePlates(int trial);

  /// This returns the corners (in 3D space) of the selected force plate, for
  /// this trial. Empty arrays on out of bounds.
  std::vector<Eigen::Vector3s> getForcePlateCorners(int trial, int forcePlate);

  /// This gets the href link associated with the subject, if there is one.
  std::string getHref();

  /// This gets the notes associated with the subject, if there are any.
  std::string getNotes();

protected:
  std::string mPath;
  // We cache some very basic data about the accessible bounds of on-disk data,
  // so we don't have to look that up every time.
  long mDataSectionStart;
  long mSensorFrameSize;
  long mProcessingPassFrameSize;
  bool mLoadedAllFrames;

  std::shared_ptr<SubjectOnDiskHeader> mHeader;
};

} // namespace biomechanics
} // namespace dart

#endif