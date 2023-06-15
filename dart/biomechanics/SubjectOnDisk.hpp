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

namespace dart {
namespace biomechanics {

struct Frame
{
  int trial;
  int t;
  s_t residual;
  bool probablyMissingGRF;
  MissingGRFReason missingGRFReason;

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
};

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

  /// This will read the skeleton from the binary, and optionally use the passed
  /// in Geometry folder.
  std::shared_ptr<dynamics::Skeleton> readSkel(std::string geometryFolder = "");

  /// This will read from disk and allocate a number of Frame objects.
  /// These Frame objects are assumed to be
  /// short-lived, to save working memory.
  ///
  /// On OOB access, prints an error and returns an empty vector.
  std::vector<std::shared_ptr<Frame>> readFrames(
      int trial,
      int startFrame,
      int numFramesToRead = 1,
      s_t contactThreshold = 1.0);

  /// This writes a subject out to disk in a compressed and random-seekable
  /// binary format.
  static void writeSubject(
      const std::string& outputPath,
      // The OpenSim file XML gets copied into our binary bundle, along with
      // any necessary Geometry files
      const std::string& openSimFilePath,
      // The per-trial motion data
      std::vector<s_t> trialTimesteps,
      std::vector<Eigen::MatrixXs>& trialPoses,
      std::vector<Eigen::MatrixXs>& trialVels,
      std::vector<Eigen::MatrixXs>& trialAccs,
      std::vector<std::vector<bool>>& probablyMissingGRF,
      std::vector<std::vector<MissingGRFReason>>& missingGRFReason,
      std::vector<std::vector<bool>>& dofPositionsObserved,
      std::vector<std::vector<bool>>& dofVelocitiesFiniteDifferenced,
      std::vector<std::vector<bool>>& dofAccelerationFiniteDifferenced,
      std::vector<Eigen::MatrixXs>& trialTaus,
      std::vector<Eigen::MatrixXs>& trialComPoses,
      std::vector<Eigen::MatrixXs>& trialComVels,
      std::vector<Eigen::MatrixXs>& trialComAccs,
      std::vector<std::vector<s_t>> trialResidualNorms,
      // These are generalized 6-dof wrenches applied to arbitrary bodies
      // (generally by foot-ground contact, though other things too)
      std::vector<std::string>& groundForceBodies,
      std::vector<Eigen::MatrixXs>& trialGroundBodyWrenches,
      std::vector<Eigen::MatrixXs>& trialGroundBodyCopTorqueForce,
      // We include this to allow the binary format to store/load a bunch of new
      // types of values while remaining backwards compatible.
      std::vector<std::string>& customValueNames,
      std::vector<std::vector<Eigen::MatrixXs>> customValues,
      // These are the markers, gyros and accelerometers
      const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
          markerObservations,
      const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
          accObservations,
      const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
          gyroObservations,
      const std::vector<std::vector<std::map<std::string, Eigen::VectorXs>>>&
          emgObservations,
      // This is raw force plate data
      std::vector<std::vector<ForcePlate>>& forcePlates,
      // This is the subject info
      const std::string& biologicalSex,
      double heightM,
      double massKg,
      int ageYears,
      // The provenance info, optional, for investigating where training data
      // came from after its been aggregated
      std::vector<std::string> trialNames,
      std::vector<std::string> subjectTags,
      std::vector<std::vector<std::string>> trialTags,
      const std::string& sourceHref = "",
      const std::string& notes = "");

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

  std::vector<bool> getDofPositionsObserved(int trial);

  std::vector<bool> getDofVelocitiesFiniteDifferenced(int trial);

  std::vector<bool> getDofAccelerationsFiniteDifferenced(int trial);

  std::vector<s_t> getTrialResidualNorms(int trial);

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
  int mNumDofs;
  int mNumTrials;
  std::vector<std::string> mGroundContactBodies;
  std::vector<int> mTrialLength;
  std::vector<s_t> mTrialTimesteps;
  std::vector<std::string> mCustomValues;
  std::vector<int> mCustomValueLengths;
  std::vector<int> mTrialNumForcePlates;
  std::vector<std::vector<std::vector<Eigen::Vector3s>>>
      mTrialForcePlateCorners;
  int mDataSectionStart;
  int mFrameSize;
  // If we're projecting a lower-body-only dataset onto a full-body model, then
  // there will be DOFs that we don't get to observe. Downstream applications
  // will want to ignore these DOFs.
  std::vector<std::vector<bool>> mDofPositionsObserved;
  // If we didn't use gyros to measure rotational velocity directly, then the
  // velocity on this joint is likely to be noisy. If that's true, downstream
  // applications won't want to try to predict the velocity on these DOFs
  // directly.
  std::vector<std::vector<bool>> mDofVelocitiesFiniteDifferenced;
  // If we didn't use accelerometers to measure acceleration directly, then the
  // acceleration on this joint is likely to be noisy. If that's true,
  // downstream applications won't want to try to predict the acceleration on
  // these DOFs directly.
  std::vector<std::vector<bool>> mDofAccelerationFiniteDifferenced;
  // These are the norms of the residuals on each frame, for each trial
  std::vector<std::vector<s_t>> mTrialResidualNorms;
  // These are the maximum absolute DOF velocity on each frame, for each trial
  std::vector<std::vector<s_t>> mTrialMaxJointVelocity;

  // This is the only array that has the potential to be somewhat large in
  // memory, but we really want to know this information when randomly picking
  // frames from the subject to sample.
  std::vector<std::vector<bool>> mProbablyMissingGRF;
  std::vector<std::vector<MissingGRFReason>> mMissingGRFReason;
  // The trial names, if provided, or empty strings
  std::vector<std::string> mTrialNames;
  // An optional link to the web where this subject came from
  std::string mHref;
  // Any text-based notes on the subject data, like citations etc
  std::string mNotes;
  // These are details about the subject's input parameters
  std::string mBiologicalSex;
  double mHeightM;
  double mMassKg;
  int mAgeYears;
  std::vector<std::string> mSubjectTags;
  std::vector<std::vector<std::string>> mTrialTags;
  // These are the marker, accelerometer and gyroscope names
  std::vector<std::string> mMarkerNames;
  std::vector<std::string> mAccNames;
  std::vector<std::string> mGyroNames;
  // This is EMG data
  std::vector<std::string> mEmgNames;
  int mEmgDim;
};

} // namespace biomechanics
} // namespace dart

#endif