#ifndef BIOMECH_SUBJECT_ON_DISK
#define BIOMECH_SUBJECT_ON_DISK

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

struct Frame
{
  int trial;
  int t;
  bool probablyMissingGRF;

  Eigen::VectorXd pos;
  Eigen::VectorXd vel;
  Eigen::VectorXd acc;
  Eigen::VectorXd tau;
  // We don't use `std::map` here because somehow the Pybind wrapper for it is
  // excruciatingly slow
  std::vector<std::pair<std::string, Eigen::Vector6d>> externalWrenches;
  // The timestep we used during this trial
  s_t dt;
  // We include this to allow the binary format to store/load a bunch of new
  // types of values while remaining backwards compatible.
  std::vector<std::pair<std::string, Eigen::VectorXd>> customValues;
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
  SubjectOnDisk(const std::string& path, bool printDebuggingDetails = false);

  /// This will read the skeleton from the binary, and optionally use the passed
  /// in Geometry folder.
  std::shared_ptr<dynamics::Skeleton> readSkel(std::string geometryFolder = "");

  /// This will read from disk and allocate a number of Frame objects.
  /// These Frame objects are assumed to be
  /// short-lived, to save working memory.
  ///
  /// On OOB access, prints an error and returns an empty vector.
  std::vector<std::shared_ptr<Frame>> readFrames(
      int trial, int startFrame, int numFramesToRead = 1);

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
      std::vector<Eigen::MatrixXs>& trialTaus,
      // These are generalized 6-dof wrenches applied to arbitrary bodies
      // (generally by foot-ground contact, though other things too)
      std::vector<std::string>& externalForceBodies,
      std::vector<Eigen::MatrixXs>& trialExternalBodyWrenches,
      // We include this to allow the binary format to store/load a bunch of new
      // types of values while remaining backwards compatible.
      std::vector<std::string>& customValueNames,
      std::vector<std::vector<Eigen::MatrixXs>> customValues,
      // The provenance info, optional, for investigating where training data
      // came from after its been aggregated
      const std::string& sourceHref = "",
      const std::string& notes = "");

  /// This returns the number of trials on the subject
  int getNumTrials();

  /// This returns the length of the trial
  int getTrialLength(int trial);

  /// This returns the number of DOFs for the model on this Subject
  int getNumDofs();

  /// This returns the vector of booleans for whether or not each timestep is
  /// heuristically detected to be missing external forces (which means that the
  /// inverse dynamics cannot be trusted).
  std::vector<bool> getProbablyMissingGRF(int trial);

  /// This returns the list of contact body names for this Subject
  std::vector<std::string> getContactBodies();

  /// This returns the list of custom value names stored in this subject
  std::vector<std::string> getCustomValues();

  /// This returns the dimension of the custom value specified by `valueName`
  int getCustomValueDim(std::string valueName);

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
  std::vector<std::string> mContactBodies;
  std::vector<int> mTrialLength;
  std::vector<s_t> mTrialTimesteps;
  std::vector<std::string> mCustomValues;
  std::vector<int> mCustomValueLengths;
  int mModelLength;
  int mModelSectionStart;
  int mDataSectionStart;
  int mFrameSize;
  // This is the only array that has the potential to be somewhat large in
  // memory, but we really want to know this information when randomly picking
  // frames from the subject to sample.
  std::vector<std::vector<bool>> mProbablyMissingGRF;
  // An optional link to the web where this subject came from
  std::string mHref;
  // Any text-based notes on the subject data, like citations etc
  std::string mNotes;
};

} // namespace biomechanics
} // namespace dart

#endif