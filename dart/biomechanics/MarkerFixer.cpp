#include "dart/biomechanics/MarkerFixer.hpp"

#include <algorithm>

#include "dart/math/AssignmentMatcher.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {

namespace biomechanics {

//==============================================================================
LabeledMarkerTrace::LabeledMarkerTrace()
{
}

//==============================================================================
/// This constructor will compute jointFingerprints from the joints passed in
LabeledMarkerTrace::LabeledMarkerTrace(
    int time, Eigen::Vector3s firstPoint, std::string markerLabel)
{
  mTimes.push_back(time);
  mPoints.push_back(firstPoint);
  mMarkerLabels.push_back(markerLabel);
  mMinTime = time;
  mMaxTime = time;
}

//==============================================================================
/// This is the direct constructor
LabeledMarkerTrace::LabeledMarkerTrace(
    std::vector<int> times,
    std::vector<Eigen::Vector3s> points,
    std::vector<std::string> markerLabels)
  : mTimes(times), mPoints(points), mMarkerLabels(markerLabels)
{
  mMinTime = INT_MAX;
  mMaxTime = INT_MIN;

  for (int t : mTimes)
  {
    if (t < mMinTime)
    {
      mMinTime = t;
    }
    if (t > mMaxTime)
    {
      mMaxTime = t;
    }
  }
}

//==============================================================================
/// Add a point to the end of the marker trace
void LabeledMarkerTrace::appendPoint(
    int time, Eigen::Vector3s point, std::string label)
{
  mTimes.push_back(time);
  mPoints.push_back(point);
  mMarkerLabels.push_back(label);
  if (time < mMinTime)
  {
    mMinTime = time;
  }
  if (time > mMaxTime)
  {
    mMaxTime = time;
  }
}

//==============================================================================
/// This gives the distance from the last point (or an extrapolation at this
/// timestep of the last point, of order up to 2)
///
/// This always returns 0 if the current trace is empty.
s_t LabeledMarkerTrace::pointToAppendDistance(
    int time, Eigen::Vector3s point, bool extrapolate)
{
  if (mPoints.size() == 0)
  {
    return 0.0;
  }
  Eigen::Vector3s& lastPoint = mPoints.at(mPoints.size() - 1);
  if (extrapolate && mPoints.size() > 1)
  {
    int lastTime = mTimes.at(mTimes.size() - 1);
    Eigen::Vector3s v = (lastPoint - mPoints.at(mPoints.size() - 2))
                        / (lastTime - mTimes.at(mTimes.size() - 2));
    Eigen::Vector3s projected = lastPoint + (v * (time - lastTime));
    return (point - projected).norm();
  }
  else
  {
    return (point - lastPoint).norm();
  }
}

//==============================================================================
/// Returns true if these traces don't overlap
bool LabeledMarkerTrace::overlap(LabeledMarkerTrace& toAppend)
{
  //// AAAAAAAAAA
  //// BBBBBBBB

  //// AAAAAAAAAA
  ////         BBBBBBBB

  ////   AAAA
  //// BBBBBBBB

  // We can guarantee that they don't overlap if we have either bound mismatched
  bool dontOverlap
      = (mMaxTime < toAppend.mMinTime) || (toAppend.mMaxTime < mMinTime);
  return !dontOverlap;
}

//==============================================================================
/// This merges two MarkerTrace's together, to create a new trace object
LabeledMarkerTrace LabeledMarkerTrace::concat(LabeledMarkerTrace& toAppend)
{
  std::vector<int> times = std::vector<int>(mTimes);
  std::vector<Eigen::Vector3s> points = std::vector<Eigen::Vector3s>(mPoints);
  std::vector<std::string> labels = std::vector<std::string>(mMarkerLabels);
  for (int t : toAppend.mTimes)
  {
    times.push_back(t);
  }
  for (Eigen::Vector3s p : toAppend.mPoints)
  {
    points.push_back(p);
  }
  for (std::string p : toAppend.mMarkerLabels)
  {
    labels.push_back(p);
  }

  return LabeledMarkerTrace(times, points, labels);
}

//==============================================================================
/// This returns when this MarkerTrace begins (inclusive)
int LabeledMarkerTrace::firstTimestep()
{
  return mMinTime;
}

//==============================================================================
/// This returns when this MarkerTrace ends (inclusive)
int LabeledMarkerTrace::lastTimestep()
{
  return mMaxTime;
}

//==============================================================================
/// Returns the index in this trace for the specified timestep
int LabeledMarkerTrace::getIndexForTimestep(int t)
{
  if (t < mMinTime || t > mMaxTime)
    return -1;
  auto res = std::find(mTimes.begin(), mTimes.end(), t);
  if (res == mTimes.end())
    return -1;
  else
    return res - mTimes.begin();
}

//==============================================================================
/// Pick the best label for this trace
std::string LabeledMarkerTrace::getBestLabel(
    std::vector<std::string> alreadyTaken)
{
  std::map<std::string, int> counts;
  for (std::string& label : mMarkerLabels)
  {
    if (counts.count(label) == 0)
    {
      counts[label] = 0;
    }
    counts[label]++;
  }

  std::string bestLabel = "";
  int bestLabelScore = 0;
  for (auto& pair : counts)
  {
    if (std::find(alreadyTaken.begin(), alreadyTaken.end(), pair.first)
        == alreadyTaken.end())
    {
      if (pair.second > bestLabelScore)
      {
        bestLabel = pair.first;
        bestLabelScore = pair.second;
      }
    }
  }
  // assert(bestLabel != "");

  return bestLabel;
}

//==============================================================================
/// Generate warnings about how we changed the labels of markers to keep them
/// continuous
std::vector<std::string> LabeledMarkerTrace::emitWarningsAboutLabelChange(
    std::string finalLabel)
{
  std::vector<std::string> warnings;
  if (mMarkerLabels.size() == 0)
    return warnings;

  std::string lastMarkerLabel = mMarkerLabels[0];
  int startSegment = 0;
  bool lastInChangedSegment = (lastMarkerLabel != finalLabel);
  for (int i = 1; i < mMarkerLabels.size(); i++)
  {
    // Any time the label changes
    bool inChangedSegment = (mMarkerLabels[i] != finalLabel);
    if (mMarkerLabels[i] != lastMarkerLabel)
    {
      if ((lastInChangedSegment && !inChangedSegment)
          || (lastInChangedSegment && inChangedSegment))
      {
        // We reached the end of a changed segment, so we can emit a warning
        // about it
        if (finalLabel == "")
        {
          if (mTimes[startSegment] == mTimes[i - 1])
          {
            warnings.push_back(
                "Ignored flickering marker " + lastMarkerLabel + " on frame "
                + std::to_string(mTimes[startSegment]));
          }
          else
          {
            warnings.push_back(
                "Ignored flickering marker " + lastMarkerLabel + " from frames "
                + std::to_string(mTimes[startSegment]) + " to "
                + std::to_string(mTimes[i - 1]));
          }
        }
        else
        {
          if (mTimes[startSegment] == mTimes[i - 1])
          {
            warnings.push_back(
                "Relabeled " + lastMarkerLabel + " as " + finalLabel
                + " on frame " + std::to_string(mTimes[startSegment])
                + " to preserve smooth movement");
          }
          else
          {
            warnings.push_back(
                "Relabeled " + lastMarkerLabel + " as " + finalLabel
                + " from frames " + std::to_string(mTimes[startSegment])
                + " to " + std::to_string(mTimes[i - 1])
                + " to preserve smooth movement");
          }
        }
      }
      if ((!lastInChangedSegment && inChangedSegment)
          || (lastInChangedSegment && inChangedSegment))
      {
        // We're starting a change segment
        startSegment = i;
      }
      // It should be impossible to have them both be the original label, and
      // yet be different
      assert(!(!lastInChangedSegment && !inChangedSegment));
    }
    lastInChangedSegment = inChangedSegment;
    lastMarkerLabel = mMarkerLabels[i];
  }

  return warnings;
}

//==============================================================================
/// This merges point clouds over time, to create a set of raw MarkerTraces
/// over time. These traces can then be intelligently merged using any desired
/// algorithm.
std::vector<LabeledMarkerTrace> LabeledMarkerTrace::createRawTraces(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    s_t mergeDistance,
    int mergeFrames)
{
  std::vector<LabeledMarkerTrace> traces;
  std::vector<int> activeTraces;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    // 1. Only count as "active" the traces that are within `mergeFrames` of now
    std::vector<int> tracesToRemove;
    for (int i = 0; i < activeTraces.size(); i++)
    {
      if (traces[activeTraces[i]].lastTimestep() < t - mergeFrames)
      {
        // This needs to be deactivated
        tracesToRemove.push_back(i);
      }
    }
    for (int j = tracesToRemove.size() - 1; j >= 0; j--)
    {
      activeTraces.erase(activeTraces.begin() + tracesToRemove[j]);
    }

    // Bail early on empty frames
    if (markerObservations[t].size() == 0)
    {
      continue;
    }

    std::vector<std::string> markerNames;
    for (auto& pair : markerObservations[t])
    {
      markerNames.push_back(pair.first);
    }

    // 2. Compute affinity scores between active traces and points
    Eigen::MatrixXs weights
        = Eigen::MatrixXs(markerObservations[t].size(), activeTraces.size());
    for (int i = 0; i < markerNames.size(); i++)
    {
      for (int j = 0; j < activeTraces.size(); j++)
      {
        s_t dist = traces[activeTraces[j]].pointToAppendDistance(
            t, markerObservations[t].at(markerNames[i]), true);
        if (dist > mergeDistance)
        {
          weights(i, j) = -1 * std::numeric_limits<double>::infinity();
        }
        else
        {
          weights(i, j) = 1.0 / dist;
        }
      }
    }

    // 3. Assign points to active traces, or create new traces for unassigned
    // points
    Eigen::VectorXi map = math::AssignmentMatcher::assignRowsToColumns(weights);
    for (int i = 0; i < map.size(); i++)
    {
      if (map(i) == -1)
      {
        traces.emplace_back(
            t, markerObservations[t].at(markerNames[i]), markerNames[i]);
        assert(traces.at(traces.size() - 1).mPoints.size() == 1);
        activeTraces.push_back(traces.size() - 1);
        assert(
            traces.at(activeTraces.at(activeTraces.size() - 1)).mPoints.size()
            == 1);
      }
      else
      {
        traces[activeTraces[map(i)]].appendPoint(
            t, markerObservations[t].at(markerNames[i]), markerNames[i]);
      }
    }
  }

  return traces;
}

//==============================================================================
/// This will go through original marker data and attempt to detect common
/// anomalies, generate warnings to help the user fix their own issues, and
/// produce fixes where possible.
MarkersErrorReport MarkerFixer::generateDataErrorsReport(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        immutableMarkerObservations)
{
  MarkersErrorReport report;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations
      = immutableMarkerObservations;

  // 1. Attempt to detect marker flips that occur partway through the trajectory

  // 1.1. Collect markers into continuous traces
  std::vector<LabeledMarkerTrace> traces
      = LabeledMarkerTrace::createRawTraces(immutableMarkerObservations);

  // 1.2. Label the traces based on their majority label during the trace
  std::vector<std::string> traceLabels;
  for (int i = 0; i < traces.size(); i++)
  {
    std::vector<std::string> alreadyTakenLabels;
    for (int j = 0; j < i; j++)
    {
      if (traces[j].overlap(traces[i]))
      {
        alreadyTakenLabels.push_back(traceLabels[j]);
      }
    }
    std::string bestLabel = traces[i].getBestLabel(alreadyTakenLabels);
    traceLabels.push_back(bestLabel);
  }

  // 2. Emit the corrected marker observations

  std::vector<std::map<std::string, Eigen::Vector3s>> correctedObservations;
  for (int t = 0; t < immutableMarkerObservations.size(); t++)
  {
    std::map<std::string, Eigen::Vector3s> frame;
    for (int j = 0; j < traces.size(); j++)
    {
      int index = traces[j].getIndexForTimestep(t);
      if (index != -1)
      {
        Eigen::Vector3s point = traces[j].mPoints[index];
        frame[traceLabels[j]] = point;
      }
    }
    correctedObservations.push_back(frame);
  }

  // 3. Emit warnings based on any traces that include markers that are not
  // labelled correctly during part of the trace

  // 3.1. Give marker jump warnings
  std::set<std::string> markerNames;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    for (auto& pair : markerObservations[i])
    {
      if (markerNames.count(pair.first) == 0)
      {
        markerNames.insert(pair.first);
      }
    }
  }

  for (auto& marker : markerNames)
  {
    int firstTimestep = -1;
    for (int i = 0; i < markerObservations.size(); i++)
    {
      if (markerObservations[i].count(marker))
      {
        firstTimestep = i;
        break;
      }
    }

    if (firstTimestep != -1)
    {
      LabeledMarkerTrace trace(
          firstTimestep, markerObservations[firstTimestep].at(marker), marker);
      for (int i = firstTimestep + 1; i < markerObservations.size(); i++)
      {
        if (markerObservations[i].count(marker))
        {
          Eigen::Vector3s point = markerObservations[i].at(marker);
          s_t dist = trace.pointToAppendDistance(i, point, true);
          if (dist > 0.2)
          {
            report.warnings.push_back(
                "Marker " + marker + " jumps a suspiciously large "
                + std::to_string((double)dist) + "m on frame "
                + std::to_string(i));
          }
          trace.appendPoint(i, point, marker);
        }
      }
    }
  }

  // 3.2. Give trace swap warnings
  std::map<std::string, std::vector<LabeledMarkerTrace>> consolidatedTraces;
  for (int i = 0; i < traces.size(); i++)
  {
    std::string label = traceLabels[i];
    consolidatedTraces[label].push_back(traces[i]);
  }

  for (auto& pair : consolidatedTraces)
  {
    std::vector<LabeledMarkerTrace> tracesGroup = pair.second;
    if (pair.first != "")
    {
      for (int i = 0; i < tracesGroup.size(); i++)
      {
        for (int j = i + 1; j < tracesGroup.size(); j++)
        {
          assert(!tracesGroup[i].overlap(tracesGroup[j]));
          assert(!tracesGroup[j].overlap(tracesGroup[i]));
        }
      }
    }
    std::sort(
        tracesGroup.begin(),
        tracesGroup.end(),
        [](const LabeledMarkerTrace& a, const LabeledMarkerTrace& b) {
          return a.mMaxTime < b.mMaxTime;
        });

    LabeledMarkerTrace merged = tracesGroup[0];
    for (int i = 1; i < tracesGroup.size(); i++)
    {
      if (pair.first != "")
      {
        assert(merged.mMaxTime < tracesGroup[i].mMinTime);
        merged = merged.concat(tracesGroup[i]);
      }
      else if (merged.mMaxTime < tracesGroup[i].mMinTime)
      {
        merged = merged.concat(tracesGroup[i]);
      }
    }

    std::vector<std::string> warnings
        = merged.emitWarningsAboutLabelChange(pair.first);
    for (int j = 0; j < warnings.size(); j++)
    {
      report.warnings.push_back(warnings[j]);
    }
  }

  report.markerObservationsAttemptedFixed = correctedObservations;
  return report;
}

} // namespace biomechanics
} // namespace dart
