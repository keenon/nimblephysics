#ifndef DART_BIOMECH_MARKERLABELLER_HPP_
#define DART_BIOMECH_MARKERLABELLER_HPP_

#include <functional>
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

class MarkerTrace
{
public:
  int mMinTime;
  int mMaxTime;
  std::vector<int> mTimes;
  std::vector<Eigen::Vector3s> mPoints;

  std::string mMarkerLabel;

  // This marker trace could be paired with any body to create a marker, so we
  // compute what each (marker, body) pairing would look like.
  std::map<std::string, Eigen::Vector3s> mBodyMarkerOffsets;
  std::map<std::string, s_t> mBodyMarkerOffsetVariance;
  std::map<std::string, s_t> mBodyRootJointDistVariance;
  std::map<std::string, s_t> mBodyClosestPointDistance;

  /// This is the constructor for when a MarkerTrace is first created, before we
  /// begin adding points to it
  MarkerTrace(int time, Eigen::Vector3s firstPoint);

protected:
  /// This is the direct constructor
  MarkerTrace(std::vector<int> times, std::vector<Eigen::Vector3s> points);

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
      s_t mergeDistance = 0.01,
      int mergeFrames = 5);

  /// Each possible combination of (trace, body) can create a marker. So we can
  /// compute some summary statistics for each body we could assign this trace
  /// to.
  void computeBodyMarkerStats(
      std::shared_ptr<dynamics::Skeleton> skel,
      std::vector<Eigen::VectorXs> posesOverTime,
      std::vector<Eigen::VectorXs> scalesOverTime);

  /// Each possible combination of (trace, body) can create a marker. This
  /// returns a score for a given body, for how "good" of a marker that body
  /// would create when combined with this trace. Lower is better.
  s_t computeBodyMarkerLoss(std::string bodyName);

  /// This finds the best body to pair this trace with (using the stats from
  /// computeBodyMarkerStats()) and returns the best marker
  std::pair<std::string, Eigen::Vector3s> getBestMarker();

  /// Returns true if these traces overlap in time
  bool overlap(MarkerTrace& toAppend);

  /// This merges two MarkerTrace's together, to create a new trace object
  MarkerTrace concat(MarkerTrace& toAppend);

  /// This returns when this MarkerTrace begins (inclusive)
  int firstTimestep();

  /// This returns when this MarkerTrace ends (inclusive)
  int lastTimestep();
};

struct LabelledMarkers
{
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>> markerOffsets;
  std::vector<std::map<std::string, Eigen::Vector3s>> jointCenterGuesses;
  // Just for debugging
  std::vector<MarkerTrace> traces;
};

class MarkerLabeller
{
public:
  virtual std::vector<std::map<std::string, Eigen::Vector3s>>
  guessJointLocations(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds)
      = 0;

  /// This labels a sequence of unlabeled point clouds, using our joint center
  /// prediction.
  LabelledMarkers labelPointClouds(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
      s_t mergeMarkersThreshold = 0.01);

  void setSkeleton(std::shared_ptr<dynamics::Skeleton> skeleton);

  void matchUpJointToSkeletonJoint(
      std::string jointName, std::string skeletonJointName);

  /// This takes in a set of labeled point clouds over time, and runs the
  /// labeller over unlabeled copies of those point clouds, and then scores the
  /// reconstruction accuracy.
  void evaluate(
      const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
          markerOffsets,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          labeledPointClouds);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::map<std::string, std::string> mJointToSkelJointNames;
};

class NeuralMarkerLabeller : public MarkerLabeller
{
public:
  NeuralMarkerLabeller(
      std::function<std::vector<std::map<std::string, Eigen::Vector3s>>(
          const std::vector<std::vector<Eigen::Vector3s>>&)>
          jointCenterPredictor);

  virtual std::vector<std::map<std::string, Eigen::Vector3s>>
  guessJointLocations(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds);

protected:
  std::function<std::vector<std::map<std::string, Eigen::Vector3s>>(
      const std::vector<std::vector<Eigen::Vector3s>>&)>
      mJointCenterPredictor;
};

class MarkerLabellerMock : public MarkerLabeller
{
public:
  virtual std::vector<std::map<std::string, Eigen::Vector3s>>
  guessJointLocations(
      const std::vector<std::vector<Eigen::Vector3s>>& pointClouds);

  void setMockJointLocations(
      std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime);

protected:
  std::vector<std::map<std::string, Eigen::Vector3s>> mJointsOverTime;
};

} // namespace biomechanics
} // namespace dart

#endif