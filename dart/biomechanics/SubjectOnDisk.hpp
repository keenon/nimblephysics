#ifndef BIOMECH_SUBJECT_ON_DISK
#define BIOMECH_SUBJECT_ON_DISK

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/proto/SubjectOnDisk.pb.h"

namespace dart {
namespace biomechanics {

class SubjectOnDiskHeader;

struct FramePass
{
  std::string passName;
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
  // These are masks for which DOFs are observed
  Eigen::VectorXi posObserved;
  // These are masks for which DOFs have been finite differenced (if they
  // haven't been finite differenced, they're from real sensors and therefore
  // more trustworthy)
  Eigen::VectorXi velFiniteDifferenced;
  Eigen::VectorXi accFiniteDifferenced;
};

struct Frame
{
  int trial;
  int t;
  bool probablyMissingGRF;
  MissingGRFReason missingGRFReason;
  // Each processing pass has its own set of kinematics and dynamics, as the
  // model and trajectories are adjusted
  std::vector<FramePass> processingPasses;

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

  void readFromProto(
      dart::proto::SubjectOnDiskFrame* proto,
      const SubjectOnDiskHeader& header,
      int trial,
      int t,
      s_t contactThreshold);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Builders, to create a SubjectOnDisk from scratch
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SubjectOnDiskTrialPass
{
public:
  SubjectOnDiskTrialPass();
  SubjectOnDiskTrialPass& setName(const std::string& name);
  SubjectOnDiskTrialPass& setDofPositionsObserved(
      std::vector<bool> dofPositionsObserved);
  SubjectOnDiskTrialPass& setDofVelocitiesFiniteDifferenced(
      std::vector<bool> dofVelocitiesFiniteDifferenced);
  SubjectOnDiskTrialPass& setDofAccelerationFiniteDifferenced(
      std::vector<bool> dofAccelerationFiniteDifference);
  SubjectOnDiskTrialPass& setMarkerRMS(std::vector<s_t> markerRMS);
  SubjectOnDiskTrialPass& setMarkerMax(std::vector<s_t> markerMax);
  SubjectOnDiskTrialPass& setLinearResidual(std::vector<s_t> linearResidual);
  SubjectOnDiskTrialPass& setAngularResidual(std::vector<s_t> angularResidual);
  SubjectOnDiskTrialPass& setPoses(Eigen::MatrixXs poses);
  SubjectOnDiskTrialPass& setVels(Eigen::MatrixXs vels);
  SubjectOnDiskTrialPass& setAccs(Eigen::MatrixXs accs);
  SubjectOnDiskTrialPass& setTaus(Eigen::MatrixXs taus);
  SubjectOnDiskTrialPass& setComPoses(Eigen::MatrixXs poses);
  SubjectOnDiskTrialPass& setComVels(Eigen::MatrixXs vels);
  SubjectOnDiskTrialPass& setComAccs(Eigen::MatrixXs accs);
  void read(const proto::SubjectOnDiskTrialProcessingPassHeader& proto);
  void write(proto::SubjectOnDiskTrialProcessingPassHeader* proto);

protected:
  // This data is included in the header
  std::string mName;
  std::vector<bool> mDofPositionsObserved;
  std::vector<bool> mDofVelocitiesFiniteDifferenced;
  std::vector<bool> mDofAccelerationFiniteDifferenced;
  std::vector<s_t> mMarkerRMS;
  std::vector<s_t> mMarkerMax;
  std::vector<s_t> mLinearResidual;
  std::vector<s_t> mAngularResidual;
  // This data is in each separate Frame, and so won't be loaded from the proto
  Eigen::MatrixXs mPos;
  Eigen::MatrixXs mVel;
  Eigen::MatrixXs mAcc;
  Eigen::MatrixXs mTaus;
  Eigen::MatrixXs mComPoses;
  Eigen::MatrixXs mComVels;
  Eigen::MatrixXs mComAccs;
  Eigen::MatrixXs mTrialGroundBodyWrenches;
  Eigen::MatrixXs mTrialGroundBodyCopTorqueForce;

  friend struct Frame;
  friend class SubjectOnDisk;
};

class SubjectOnDiskTrial
{
public:
  SubjectOnDiskTrial();
  SubjectOnDiskTrial& setName(const std::string& name);
  SubjectOnDiskTrial& setTimestep(s_t timestep);
  SubjectOnDiskTrial& setTrialTags(std::vector<std::string> trialTags);
  SubjectOnDiskTrial& setProbablyMissingGRF(
      std::vector<bool> probablyMissingGRF);
  SubjectOnDiskTrial& setMissingGRFReason(
      std::vector<MissingGRFReason> missingGRFReason);
  SubjectOnDiskTrial& setCustomValues(
      std::vector<Eigen::MatrixXs> customValues);
  SubjectOnDiskTrial& setMarkerNamesGuessed(bool markersGuessed);
  SubjectOnDiskTrial& setMarkerObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations);
  SubjectOnDiskTrial& setAccObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> accObservations);
  SubjectOnDiskTrial& setGyroObservations(
      std::vector<std::map<std::string, Eigen::Vector3s>> gyroObservations);
  SubjectOnDiskTrial& setEmgObservations(
      std::vector<std::map<std::string, Eigen::VectorXs>> emgObservations);
  SubjectOnDiskTrial& setExoTorques(
      std::map<std::string, Eigen::VectorXs> exoTorques);
  SubjectOnDiskTrial& setForcePlates(std::vector<ForcePlate> forcePlates);
  SubjectOnDiskTrialPass& addPass();
  void read(const proto::SubjectOnDiskTrialHeader& proto);
  void write(proto::SubjectOnDiskTrialHeader* proto);

protected:
  std::string mName;
  s_t mTimestep;
  int mLength;
  std::vector<std::string> mTrialTags;
  std::vector<SubjectOnDiskTrialPass> mPasses;
  std::vector<bool> mProbablyMissingGRF;
  std::vector<MissingGRFReason> mMissingGRFReason;
  // This is for allowing the user to pre-filter out data where joint velocities
  // are above a certain "unreasonable limit", like 50 rad/s or so
  std::vector<s_t> mJointsMaxVelocity;
  // This is true if we guessed the marker names, and false if we got them from
  // the uploaded user's file, which implies that they got them from human
  // observations.
  bool mMarkerNamesGuessed;

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
  std::map<std::string, Eigen::VectorXs> mExoTorques;
  // This is raw force plate data
  std::vector<ForcePlate> mForcePlates;

  friend class SubjectOnDiskHeader;
  friend class SubjectOnDisk;
  friend struct Frame;
};

class SubjectOnDiskPassHeader
{
public:
  SubjectOnDiskPassHeader();
  SubjectOnDiskPassHeader& setProcessingPassType(ProcessingPassType type);
  SubjectOnDiskPassHeader& setOpenSimFileText(
      const std::string& openSimFileText);
  void write(dart::proto::SubjectOnDiskPass* proto);
  void read(const dart::proto::SubjectOnDiskPass& proto);

protected:
  ProcessingPassType mType;
  // The OpenSim file XML gets copied into our binary bundle, along with
  // any necessary Geometry files
  std::string mOpenSimFileText;

  friend class SubjectOnDisk;
};

class SubjectOnDiskHeader
{
public:
  SubjectOnDiskHeader();
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
  SubjectOnDiskPassHeader& addProcessingPass();
  SubjectOnDiskTrial& addTrial();
  void recomputeColumnNames();
  void write(dart::proto::SubjectOnDiskHeader* proto);
  void read(const dart::proto::SubjectOnDiskHeader& proto);
  void writeFrame(dart::proto::SubjectOnDiskFrame* proto, int trial, int t);

protected:
  // How many DOFs are in the skeleton
  int mNumDofs;
  // The passes we applied to this data, along with the result skeletons that
  // were generated by each pass.
  std::vector<SubjectOnDiskPassHeader> mPasses;
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
  std::vector<SubjectOnDiskTrial> mTrials;

  // These are the marker, accelerometer and gyroscope names
  std::vector<std::string> mMarkerNames;
  std::vector<std::string> mAccNames;
  std::vector<std::string> mGyroNames;
  // This is EMG data
  std::vector<std::string> mEmgNames;
  int mEmgDim;
  // This is exoskeleton data
  std::vector<std::string> mExoDofNames;

  friend class SubjectOnDisk;
  friend struct Frame;
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
  SubjectOnDisk(const std::string& path);

  /// This will write a B3D file to disk
  static void writeB3D(const std::string& path, SubjectOnDiskHeader& header);

  /// This will read the skeleton from the binary, and optionally use the passed
  /// in Geometry folder.
  std::shared_ptr<dynamics::Skeleton> readSkel(
      int processingPass, std::string geometryFolder = "");

  /// This will read the raw OpenSim XML file text out of the binary, and return
  /// it as a string
  std::string readRawOsimFileText(int processingPass);

  /// This will read from disk and allocate a number of Frame objects.
  /// These Frame objects are assumed to be
  /// short-lived, to save working memory.
  ///
  /// On OOB access, prints an error and returns an empty vector.
  std::vector<std::shared_ptr<Frame>> readFrames(
      int trial,
      int startFrame,
      int numFramesToRead = 1,
      int stride = 1,
      s_t contactThreshold = 1.0);

  /// This returns the number of trials on the subject
  int getNumTrials();

  /// This returns the length of the trial
  int getTrialLength(int trial);

  /// This returns the timestep size for the trial
  s_t getTrialTimestep(int trial);

  /// This returns the number of DOFs for the model on this Subject
  int getNumDofs();

  /// This returns the vector of booleans for whether or not each timestep is
  /// heuristically detected to be missing external forces (which means that the
  /// inverse dynamics cannot be trusted).
  std::vector<bool> getProbablyMissingGRF(int trial);

  /// This returns the vector of enums of type 'MissingGRFReason', which labels
  /// why each time step was identified as 'probablyMissingGRF'.
  std::vector<MissingGRFReason> getMissingGRFReason(int trial);

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
  std::vector<s_t> getTrialMaxJointVelocity(int trial);

  /// This returns the list of contact body names for this Subject
  std::vector<std::string> getGroundContactBodies();

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
  int mDataSectionStart;
  int mFrameSize;

  SubjectOnDiskHeader mHeader;
};

} // namespace biomechanics
} // namespace dart

#endif