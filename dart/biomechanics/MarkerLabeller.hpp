#ifndef DART_BIOMECH_MARKERLABELLER_HPP_
#define DART_BIOMECH_MARKERLABELLER_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct LabelledMarkers
{
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>> markerOffsets;
  std::vector<std::map<std::string, Eigen::Vector3s>> jointCenterGuesses;
};

class MarkerTrace
{
public:
  int mMinTime;
  int mMaxTime;
  std::vector<int> mTimes;
  std::vector<Eigen::Vector3s> mPoints;
  std::vector<std::map<std::string, Eigen::Vector2s>> mJointFingerprints;
  std::string mMarkerLabel;

  /// This is the constructor for when a MarkerTrace is first created, before we
  /// begin adding points to it
  MarkerTrace(int time, Eigen::Vector3s firstPoint);

protected:
  /// This is the direct constructor
  MarkerTrace(
      std::vector<int> times,
      std::vector<Eigen::Vector3s> points,
      std::vector<std::map<std::string, Eigen::Vector2s>> jointFingerprints);

public:
  /// Add a point to the end of the marker trace
  void appendPoint(int time, Eigen::Vector3s point);

  /// This gives the distance from the last point (or an extrapolation at this
  /// timestep of the last point, of order up to 2)
  s_t pointToAppendDistance(int time, Eigen::Vector3s point, bool extrapolate);

  /// This merges point clouds over time, to create a set of raw MarkerTraces
  /// over time. These traces can then be intelligently merged using any desired
  /// algorithm.
  static std::vector<MarkerTrace> createRawTraces(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
      s_t mergeDistance = 0.001,
      int mergeFrames = 5);

  /// This will create fingerprints from the joint history
  void computeJointFingerprints(
      std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime,
      std::map<std::string, std::string> jointParents);

  /// Returns true if these traces overlap in time
  bool overlap(MarkerTrace& toAppend);

  /// This merges two MarkerTrace's together, to create a new trace object
  MarkerTrace concat(MarkerTrace& toAppend);

  /// This returns when this MarkerTrace begins (inclusive)
  int firstTimestep();

  /// This returns when this MarkerTrace ends (inclusive)
  int lastTimestep();

  /// This gets the mean and variance of all the joint fingerprints.
  std::map<std::string, std::tuple<Eigen::Vector2s, s_t>>
  getJointFingerprintStats();
};

class MarkerLabeller
{
public:
  virtual std::vector<std::map<std::string, Eigen::Vector3s>>
  guessJointLocations(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds)
      = 0;

  virtual std::map<std::string, std::string> getJointParents() = 0;

  LabelledMarkers labelPointClouds(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
      s_t mergeMarkersThreshold = 0.01);

  void setSkeleton(std::shared_ptr<dynamics::Skeleton> skeleton);

  void matchUpJointToSkeletonJoint(
      std::string jointName, std::string skeletonJointName);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::map<std::string, std::string> mJointToSkelJointNames;
};

class MarkerLabellerMock : public MarkerLabeller
{
public:
  virtual std::vector<std::map<std::string, Eigen::Vector3s>>
  guessJointLocations(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds);

  virtual std::map<std::string, std::string> getJointParents();

  void setMockJointLocations(
      std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime,
      std::map<std::string, std::string> jointParents);

  /// This takes in a set of labeled point clouds over time, and runs the
  /// labeller over unlabeled copies of those point clouds, and then scores the
  /// reconstruction accuracy.
  void evaluate(
      const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
          markerOffsets,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          labeledPointClouds);

protected:
  std::vector<std::map<std::string, Eigen::Vector3s>> mJointsOverTime;
  std::map<std::string, std::string> mJointParents;
};

} // namespace biomechanics
} // namespace dart

#endif