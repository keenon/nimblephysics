#ifndef DART_BIOMECH_MARKERFIXER_HPP_
#define DART_BIOMECH_MARKERFIXER_HPP_

#include <functional>
#include <memory>
// #include <unordered_map>
#include <map>
#include <mutex>
#include <string>
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

class LabeledMarkerTrace
{
public:
  LabeledMarkerTrace();

  /// This is the constructor for when a MarkerTrace is first created, before we
  /// begin adding points to it
  LabeledMarkerTrace(
      int time, Eigen::Vector3s firstPoint, std::string markerLabel);

protected:
  /// This is the direct constructor
  LabeledMarkerTrace(
      std::vector<int> times,
      std::vector<Eigen::Vector3s> points,
      std::vector<std::string> markerLabels);

public:
  /// Add a point to the end of the marker trace
  void appendPoint(int time, Eigen::Vector3s point, std::string label);

  /// This gives the distance from the last point (or an extrapolation at this
  /// timestep of the last point, of order up to 2)
  s_t pointToAppendDistance(int time, Eigen::Vector3s point, bool extrapolate);

  /// This merges point clouds over time, to create a set of raw MarkerTraces
  /// over time. These traces can then be intelligently merged using any desired
  /// algorithm.
  static std::vector<LabeledMarkerTrace> createRawTraces(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      s_t mergeDistance = 0.01,
      int mergeFrames = 5);

  /// Returns true if these traces overlap in time
  bool overlap(LabeledMarkerTrace& toAppend);

  /// This merges two MarkerTrace's together, to create a new trace object
  LabeledMarkerTrace concat(LabeledMarkerTrace& toAppend);

  /// This returns when this MarkerTrace begins (inclusive)
  int firstTimestep();

  /// This returns when this MarkerTrace ends (inclusive)
  int lastTimestep();

  /// Returns true if this timestep is in this trace
  bool hasTimestep(int t);

  /// Returns the index in this trace for the specified timestep
  int getIndexForTimestep(int t);

  /// Pick the best label for this trace
  std::string getBestLabel(std::vector<std::string> alreadyTaken);

  /// Generate warnings about how we changed the labels of markers to keep them
  /// continuous
  std::vector<std::string> emitWarningsAboutLabelChange(std::string finalLabel);

  /// This computes the timesteps to drop, based on which points have too much
  /// acceleration.
  void filterTimestepsBasedOnAcc(s_t dt, s_t accThreshold);

public:
  int mMinTime;
  int mMaxTime;
  std::vector<int> mTimes;
  std::vector<Eigen::Vector3s> mPoints;
  std::vector<s_t> mAccNorm;
  std::vector<bool> mDropPoint;
  std::vector<std::string> mMarkerLabels;
};

struct MarkersErrorReport
{
  std::vector<std::string> warnings;
  std::vector<std::string> info;
  std::vector<std::map<std::string, Eigen::Vector3s>>
      markerObservationsAttemptedFixed;
};

class RippleReductionProblem
{
public:
  RippleReductionProblem(
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
      s_t dt);

  int dropSuspiciousPoints(MarkersErrorReport* report = nullptr);

  void interpolateMissingPoints();

  std::vector<std::map<std::string, Eigen::Vector3s>> smooth(
      MarkersErrorReport* report = nullptr);

  void saveToGUI(std::string markerName, std::string path);

public:
  s_t mDt;
  std::vector<std::string> mMarkerNames;
  std::map<std::string, Eigen::VectorXs> mObserved;
  std::map<std::string, Eigen::Matrix<s_t, 3, Eigen::Dynamic>> mOriginalMarkers;
  std::map<std::string, Eigen::Matrix<s_t, 3, Eigen::Dynamic>> mMarkers;
  std::map<std::string, Eigen::Matrix<s_t, 3, Eigen::Dynamic>> mSupportPlanes;
};

class MarkerFixer
{
public:
  static MarkersErrorReport generateDataErrorsReport(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      s_t dt);
};

} // namespace biomechanics
} // namespace dart

#endif