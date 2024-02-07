#ifndef DART_BIOMECH_STREAMING_TRACE
#define DART_BIOMECH_STREAMING_TRACE

#include <memory>
#include <tuple>
#include <vector>

#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIStateMachine.hpp"

namespace dart {
namespace biomechanics {

class Trace
{
public:
  Trace(
      const Eigen::Vector3s& first_point,
      long trace_time,
      int num_classes,
      int num_bodies = 0);

  void add_point(const Eigen::Vector3s& point, long time);

  Eigen::Vector3s project_to(long time) const;

  s_t dist_to(const Eigen::Vector3s& other) const;

  long time_since_last_point(long now) const;

  long last_time() const;

  long start_time() const;

  long get_duration() const;

  int get_predicted_class() const;

  std::vector<Eigen::Vector4s> get_points_at_intervals(
      long end, long interval, int windows) const;

  void render_to_gui(std::shared_ptr<server::GUIStateMachine> gui);

  void drop_from_gui(std::shared_ptr<server::GUIStateMachine> gui);

public:
  int uuid;
  std::vector<Eigen::Vector3s> points;
  std::vector<long> times;
  Eigen::VectorXs logits;
  int num_bodies;
};

/**
 * This class implements the (hopefully somewhat efficient) real-time
 * computation of IK from point clouds.
 */
class StreamingMarkerTraces
{
public:
  StreamingMarkerTraces(int numClasses, int numBodies);

  /// This method takes in a set of markers, and returns a vector of the
  /// predicted classes for each marker, based on classes we have predicted for
  /// previous markers, and continuity assumptions. It also returns a "trace
  /// tag" for each marker, that can be used to associate it with previous
  /// continuous observations of the same marker. The returned vector will be
  /// the same length and order as the input `markers` vector.
  std::pair<std::vector<int>, std::vector<int>> observeMarkers(
      const std::vector<Eigen::Vector3s>& markers, long timestamp);

  /// This method returns the features that we used to predict the classes of
  /// the markers. The first element of the pair is the features (which are
  /// trace points concatenated with the time, as measured in integer units of
  /// "windowDuration", backwards from now), and the second is the trace ID for
  /// each point, so that we can correctly assign logit outputs back to the
  /// traces.
  std::pair<Eigen::MatrixXs, Eigen::VectorXi> getTraceFeatures(
      int numWindows, long windowDuration, bool center = true);

  /// This method takes in the logits for each point, and the trace IDs for each
  /// point, and updates the internal state of the trace classifier to reflect
  /// the new information.
  void observeTraceLogits(
      const Eigen::MatrixXs& logits, const Eigen::VectorXi& traceIDs);

  /// This method sets the maximum distance that can exist between the last head
  /// of a trace, and a new marker position. Markers that are within this
  /// distance from a trace are not guaranteed to be merged (they must be the
  /// closest to the trace), but markers that are further than this distance are
  /// guaranteed to be split into a new trace.
  void setMaxJoinDistance(s_t maxJoinDistance);

  /// This method sets the timeout for traces. If a trace has not been updated
  /// for this many milliseconds, it will be removed from the trace list.
  void setTraceTimeoutMillis(long traceTimeoutMillis);

  /// This sets the maximum number of milliseconds that we will tolerate between
  /// a stride and a point we are going to accept as being at that stride.
  void setFeatureMaxStrideTolerance(long strideTolerance);

  /// This resets all traces to empty
  void reset();

  /// This method returns the number of traces we have active. This is mostly
  /// here for debugging and testing, because the number of active traces should
  /// not be a useful metric for downstream tasks that are just interested in
  /// labeled marker clouds.
  int getNumTraces();

  /// This renders the traces we have in our state to the GUI
  void renderTracesToGUI(std::shared_ptr<server::GUIStateMachine> gui);

protected:
  int mNumClasses;
  int mNumBodies;
  std::vector<Trace> mTraces;
  std::vector<int> tracesToRemoveFromGUI;

  // This is configuration to handle how we group points into traces
  long mTraceTimeoutMillis;
  s_t mTraceMaxJoinDistance;
  int mFeatureMaxStrideToleranceMillis;
};

} // namespace biomechanics
} // namespace dart

#endif