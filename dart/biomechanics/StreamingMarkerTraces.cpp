#include "dart/biomechanics/StreamingMarkerTraces.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

Trace::Trace(
    const Eigen::Vector3s& first_point,
    long trace_time,
    int num_classes,
    int num_bodies)
  : num_bodies(num_bodies), logits(Eigen::VectorXs::Zero(num_classes))
{
  points.push_back(first_point);
  times.push_back(trace_time);

  uuid = rand();
}

void Trace::add_point(const Eigen::Vector3s& point, long time)
{
  points.push_back(point);
  times.push_back(time);
}

Eigen::Vector3s Trace::project_to(long time) const
{
  if (points.size() > 1)
  {
    Eigen::Vector3s last_vel = (points.back() - points[points.size() - 2])
                               / (times.back() - times[times.size() - 2]);
    return points.back() + last_vel * (time - times.back());
  }
  else if (points.size() == 1)
  {
    return points.back();
  }
  else
  {
    return Eigen::Vector3s::Zero();
  }
}

s_t Trace::dist_to(const Eigen::Vector3s& other) const
{
  return (points.back() - other).norm();
}

long Trace::time_since_last_point(long now) const
{
  return now - times.back();
}

long Trace::last_time() const
{
  return times.back();
}

long Trace::start_time() const
{
  return times.front();
}

long Trace::get_duration() const
{
  return times.back() - times.front();
}

int Trace::get_predicted_class() const
{
  Eigen::Index maxLogit;
  logits.maxCoeff(&maxLogit);
  return maxLogit;
}

std::vector<Eigen::Vector4s> Trace::get_points_at_intervals(
    long end, long interval, int windows) const
{
  std::vector<Eigen::Vector4s> result_points;
  if (times.empty())
    return result_points;

  long start = end - interval * (windows - 1);
  Eigen::Matrix<long, Eigen::Dynamic, 1> evenly_spaced_times
      = Eigen::Matrix<long, Eigen::Dynamic, 1>::LinSpaced(windows, start, end);

  size_t points_cursor = 0;
  long threshold = 100;

  for (int i = 0; i < evenly_spaced_times.size(); ++i)
  {
    long target_time = evenly_spaced_times[i];
    while (points_cursor < times.size()
           && std::abs(times[points_cursor] - target_time) > threshold
           && times[points_cursor] < target_time)
    {
      ++points_cursor;
    }
    if (points_cursor >= times.size())
      break;

    if (std::abs(times[points_cursor] - target_time) <= threshold)
    {
      Eigen::Vector4s augmented_point;
      augmented_point.head<3>() = points[points_cursor];
      augmented_point[3] = static_cast<s_t>(
          i); // Assigning the interval index to the fourth dimension
      result_points.push_back(augmented_point);
      ++points_cursor;
    }
  }

  return result_points;
}

void Trace::render_to_gui(std::shared_ptr<server::GUIStateMachine> gui)
{
  if (!gui)
  {
    std::cout << "NO GUI!!!" << std::endl;
    return;
  }

  std::vector<Eigen::Vector3s> line_points;
  const size_t num_points = 20;

  // Ensure there are enough points, or repeat the first point as necessary
  if (points.size() >= num_points)
  {
    line_points.insert(
        line_points.end(), points.end() - num_points, points.end());
  }
  else
  {
    line_points.insert(line_points.end(), points.begin(), points.end());
    while (line_points.size() < num_points)
    {
      line_points.push_back(points.front());
    }
  }

  // Assuming logits is an Eigen::VectorXf
  int max_logit_index = std::distance(
      logits.data(),
      std::max_element(logits.data(), logits.data() + logits.size()));
  bool is_nothing = max_logit_index == logits.size() - 1;
  bool is_anatomical = max_logit_index > num_bodies;

  Eigen::Vector4s color
      = is_nothing ? Eigen::Vector4s(0.5, 0.5, 0.5, 1.0)
                   : (is_anatomical ? Eigen::Vector4s(0.0, 0.0, 1.0, 1.0)
                                    : Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));

  // Convert line_points to the required format if needed
  gui->createLine(std::to_string(uuid), line_points, color);
}

void Trace::drop_from_gui(std::shared_ptr<server::GUIStateMachine> gui)
{
  gui->deleteObject(std::to_string(uuid));
}

//==============================================================================
StreamingMarkerTraces::StreamingMarkerTraces(int numClasses, int numBodies)
  : mNumBodies(numBodies),
    mNumClasses(numClasses),
    mTraceMaxJoinDistance(0.05),
    mTraceTimeoutMillis(300),
    mFeatureMaxStrideToleranceMillis(10)
{
}

//==============================================================================
/// This method sets the maximum distance that can exist between the last head
/// of a trace, and a new marker position. Markers that are within this
/// distance from a trace are not guaranteed to be merged (they must be the
/// closest to the trace), but markers that are further than this distance are
/// guaranteed to be split into a new trace.
void StreamingMarkerTraces::setMaxJoinDistance(s_t maxJoinDistance)
{
  mTraceMaxJoinDistance = maxJoinDistance;
}

//==============================================================================
/// This method sets the timeout for traces. If a trace has not been updated
/// for this many milliseconds, it will be removed from the trace list.
void StreamingMarkerTraces::setTraceTimeoutMillis(long traceTimeoutMillis)
{
  mTraceTimeoutMillis = traceTimeoutMillis;
}

//==============================================================================
/// This sets the maximum number of milliseconds that we will tolerate between
/// a stride and a point we are going to accept as being at that stride.
void StreamingMarkerTraces::setFeatureMaxStrideTolerance(long strideTolerance)
{
  mFeatureMaxStrideToleranceMillis = strideTolerance;
}

//==============================================================================
/// This resets all traces to empty
void StreamingMarkerTraces::reset()
{
  for (auto& trace : mTraces)
  {
    tracesToRemoveFromGUI.push_back(trace.uuid);
  }
  mTraces.clear();
}

//==============================================================================
/// This method takes in a set of markers, and returns a vector of the
/// predicted classes for each marker, based on classes we have predicted for
/// previous markers, and continuity assumptions. The returned vector will be
/// the same length and order as the input `markers` vector.
std::pair<std::vector<int>, std::vector<int>>
StreamingMarkerTraces::observeMarkers(
    const std::vector<Eigen::Vector3s>& markers, long now)
{
  std::vector<int> resultClasses = std::vector<int>(markers.size(), -1);
  std::vector<int> resultTraceTags = std::vector<int>(markers.size(), -1);

  for (auto& trace : mTraces)
  {
    if (trace.time_since_last_point(now) >= this->mTraceTimeoutMillis)
    {
      tracesToRemoveFromGUI.push_back(trace.uuid);
    }
  }

  // 1. Trim the old traces
  auto it = std::remove_if(
      mTraces.begin(), mTraces.end(), [this, now](const Trace& trace) {
        return trace.time_since_last_point(now) >= this->mTraceTimeoutMillis;
      });
  mTraces.erase(it, mTraces.end());

  // 2. Assign markers to traces
  std::vector<bool> markers_assigned(markers.size(), false);
  Eigen::MatrixXs dists = Eigen::MatrixXs::Zero(mTraces.size(), markers.size());
  for (size_t i = 0; i < mTraces.size(); ++i)
  {
    auto projected_marker = mTraces[i].project_to(now);
    for (size_t j = 0; j < markers.size(); ++j)
    {
      dists(i, j) = (projected_marker - markers[j]).norm();
    }
  }
  for (size_t k = 0; k < std::min(markers.size(), mTraces.size()); ++k)
  {
    // Find the closest pair
    Eigen::MatrixXf::Index minRow, minCol;
    s_t minDist = dists.minCoeff(&minRow, &minCol);
    if (minDist > 0.1f)
    {
      break;
    }
    markers_assigned[minCol] = true;
    mTraces[minRow].add_point(markers[minCol], now);
    resultTraceTags[minCol] = mTraces[minRow].uuid;
    resultClasses[minCol] = mTraces[minRow].get_predicted_class();
    dists.row(minRow).setConstant(std::numeric_limits<s_t>::infinity());
    dists.col(minCol).setConstant(std::numeric_limits<s_t>::infinity());
  }

  // 3. Add any remaining markers as new traces
  for (size_t j = 0; j < markers.size(); ++j)
  {
    if (!markers_assigned[j])
    {
      mTraces.emplace_back(markers[j], now, mNumClasses, mNumBodies);
    }
  }

  return std::make_pair(resultClasses, resultTraceTags);
}

//==============================================================================
/// This method returns the features that we used to predict the classes of
/// the markers. The first element of the pair is the features (which are
/// trace points concatenated with the time, as measured in integer units of
/// "windowDuration", backwards from now), and the second is the trace ID for
/// each point, so that we can correctly assign logit outputs back to the
/// traces.
std::pair<Eigen::MatrixXs, Eigen::VectorXi>
StreamingMarkerTraces::getTraceFeatures(
    int numWindows, long windowDuration, bool center)
{
  //  # Get the traces that are long enough to run the model on
  // if len(self.traces) == 0:
  //     return
  if (mTraces.size() == 0)
  {
    return std::make_pair(
        Eigen::MatrixXs::Zero(4, 0), Eigen::VectorXi::Zero(0));
  }
  // expected_duration = self.window * self.stride
  long expectedDuration = numWindows * windowDuration;
  // now: float = max([trace.last_time() for trace in self.traces])
  long now = 0;
  for (int i = 0; i < mTraces.size(); i++)
  {
    const long last = mTraces[i].last_time();
    if (last > now)
    {
      now = last;
    }
  }
  // start: float = min([trace.start_time() for trace in self.traces])
  long start = LONG_MAX;
  for (int i = 0; i < mTraces.size(); i++)
  {
    const long traceStart = mTraces[i].start_time();
    if (traceStart < start)
    {
      start = traceStart;
    }
  }
  // if now - start < expected_duration:
  //     return
  if (now - start < expectedDuration)
  {
    return std::make_pair(
        Eigen::MatrixXs::Zero(4, 0), Eigen::VectorXi::Zero(0));
  }

  // input_points_list: List[np.ndarray] = []
  std::vector<Eigen::Vector4s> inputPointsList;
  // points_to_trace_uuids: List[str] = []
  std::vector<int> traceIDsList;

  // for i in range(len(self.traces)):
  //     trace = self.traces[i]
  //     points = trace.get_points_at_intervals(now, self.stride, self.window)
  //     input_points_list.extend(points)
  //     points_to_trace_uuids.extend([trace.uuid for _ in range(len(points))])
  for (int i = 0; i < mTraces.size(); i++)
  {
    if (mTraces[i].get_duration() < expectedDuration)
      continue;
    std::vector<Eigen::Vector4s> points
        = mTraces[i].get_points_at_intervals(now, windowDuration, numWindows);
    for (auto& point : points)
    {
      inputPointsList.push_back(point);
      traceIDsList.push_back(mTraces[i].uuid);
    }
  }

  // x = np.stack(input_points_list)
  // # Center the first 3 rows
  // x[:, :3] -= x[:, :3].mean(axis=0)
  Eigen::MatrixXs features = Eigen::MatrixXs(4, inputPointsList.size());
  Eigen::VectorXi traceIDs = Eigen::VectorXi(inputPointsList.size());
  for (int i = 0; i < inputPointsList.size(); i++)
  {
    features.col(i) = inputPointsList[i];
    traceIDs(i) = traceIDsList[i];
  }

  if (center)
  {
    features.block(0, 0, 3, features.cols())
        = features.block(0, 0, 3, features.cols()).colwise()
          - features.block(0, 0, 3, features.cols()).rowwise().mean();
  }

  return std::make_pair(features, traceIDs);
}

//==============================================================================
/// This method takes in the logits for each point, and the trace IDs for each
/// point, and updates the internal state of the trace classifier to reflect
/// the new information.
void StreamingMarkerTraces::observeTraceLogits(
    const Eigen::MatrixXs& logits, const Eigen::VectorXi& traceIDs)
{
  for (int i = 0; i < traceIDs.size(); i++)
  {
    int traceID = traceIDs(i);
    for (int j = 0; j < mTraces.size(); j++)
    {
      if (mTraces[j].uuid == traceID)
      {
        mTraces[j].logits += logits.col(i);
        break;
      }
    }
  }
}

//==============================================================================
/// This method returns the number of traces we have active. This is mostly
/// here for debugging and testing, because the number of active traces should
/// not be a useful metric for downstream tasks that are just interested in
/// labeled marker clouds.
int StreamingMarkerTraces::getNumTraces()
{
  return mTraces.size();
}

//==============================================================================
/// This renders the traces we have in our state to the GUI
void StreamingMarkerTraces::renderTracesToGUI(
    std::shared_ptr<server::GUIStateMachine> gui)
{
  for (int toRemove : tracesToRemoveFromGUI)
  {
    gui->deleteObject(std::to_string(toRemove));
  }
  tracesToRemoveFromGUI.clear();
  for (auto& trace : mTraces)
  {
    if (trace.get_duration() > 1500)
    {
      trace.render_to_gui(gui);
    }
  }
}

} // namespace biomechanics
} // namespace dart
