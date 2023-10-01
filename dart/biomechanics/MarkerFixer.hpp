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
      s_t mergeDistance = 0.01);

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

  /// Returns the label of the point at the last timestep, or an empty string if
  /// the length of the trace is 0
  std::string getLastLabel();

  /// Pick the best label for this trace
  std::string getBestLabel(std::vector<std::string> alreadyTaken);

  /// Generate warnings about how we changed the labels of markers to keep them
  /// continuous
  std::vector<std::string> emitWarningsAboutLabelChange(std::string finalLabel);

  /// This computes the timesteps to drop, based on which points have too much
  /// acceleration.
  void filterTimestepsBasedOnAcc(s_t dt, s_t accThreshold);

  /// If a marker is below a certain velocity for a certain number of timesteps
  /// (or more) then mark all the timesteps where the marker is still as
  /// filtered out.
  void filterTimestepsBasedOnProlongedStillness(
      s_t dt, s_t velThreshold, int numTimesteps);

  /// This is a useful measurement to test if the marker just never moves from
  /// its starting point (generally if your optical setup accidentally captured
  /// a shiny object that is fixed in place as a marker).
  s_t getMaxMarkerMovementFromStart();

public:
  int mMinTime;
  int mMaxTime;
  std::vector<int> mTimes;
  std::vector<Eigen::Vector3s> mPoints;
  std::vector<s_t> mAccNorm;
  std::vector<bool> mDropPointForAcc;
  std::vector<s_t> mVelNorm;
  std::vector<bool> mDropPointForStillness;
  std::vector<std::string> mMarkerLabels;
};

struct MarkersErrorReport
{
  std::vector<std::string> warnings;
  std::vector<std::string> info;
  std::vector<std::map<std::string, Eigen::Vector3s>>
      markerObservationsAttemptedFixed;
  // This is a list of all the 3D warnings for markers that were dropped
  std::vector<
      std::vector<std::tuple<std::string, Eigen::Vector3s, std::string>>>
      droppedMarkerWarnings;
  // This is a list of all the warnings for markers whose names were swapped
  std::vector<std::vector<std::pair<std::string, std::string>>>
      markersRenamedFromTo;
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
      MarkersErrorReport* report = nullptr, bool useSparse = true, bool useIterativeSolver = true, int solverIterations = 1e5);

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
  static std::shared_ptr<MarkersErrorReport> generateDataErrorsReport(
      std::vector<std::map<std::string, Eigen::Vector3s>>
          markerObservations,
      s_t dt,
      bool dropProlongedStillness = false,
      bool rippleReduce = true,
      bool rippleReduceUseSparse = true,
      bool rippleReduceUseIterativeSolver = true,
      int rippleReduceSolverIterations = 1e5);
};

} // namespace biomechanics
} // namespace dart

#endif