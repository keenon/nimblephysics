#include "dart/biomechanics/MarkerFixer.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "dart/math/AssignmentMatcher.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

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
/// Returns the label of the point at the last timestep, or an empty string if
/// the length of the trace is 0
std::string LabeledMarkerTrace::getLastLabel()
{
  if (mMarkerLabels.size() == 0)
  {
    return "";
  }
  else
  {
    return mMarkerLabels[mMarkerLabels.size() - 1];
  }
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
/// This computes the timesteps to drop, based on which points have too much
/// acceleration.
void LabeledMarkerTrace::filterTimestepsBasedOnAcc(s_t dt, s_t accThreshold)
{
  for (int i = 0; i < mPoints.size(); i++)
  {
    bool shouldDrop = false;

    if (i > 0 && i < mPoints.size() - 1)
    {
      Eigen::Vector3s acc
          = (mPoints[i + 1] - 2 * mPoints[i] + mPoints[i - 1]) / (dt * dt);
      s_t accNorm = acc.norm();
      mAccNorm.push_back(accNorm);
      shouldDrop = accNorm > accThreshold;
    }
    else
    {
      mAccNorm.push_back(0);
    }

    mDropPointForAcc.push_back(shouldDrop);
  }
}

//==============================================================================
/// If a marker is below a certain velocity for a certain number of timesteps
/// (or more) then mark all the timesteps where the marker is still as
/// filtered out.
void LabeledMarkerTrace::filterTimestepsBasedOnProlongedStillness(
    s_t dt, s_t velThreshold, int numTimesteps)
{
  for (int i = 0; i < mPoints.size(); i++)
    mDropPointForStillness.push_back(false);

  int startDrop = -1;
  for (int i = 0; i < mPoints.size(); i++)
  {
    if (i > 0)
    {
      Eigen::Vector3s vel = (mPoints[i] - mPoints[i - 1]) / dt;
      s_t velNorm = vel.norm();
      mVelNorm.push_back(velNorm);
      if (velNorm < velThreshold)
      {
        if (startDrop == -1)
        {
          startDrop = i;
        }
      }
      else if (startDrop != -1)
      {
        int duration = i - startDrop;
        if (duration >= numTimesteps)
        {
          for (int j = startDrop; j < i; j++)
          {
            mDropPointForStillness[j] = true;
          }
        }
        startDrop = -1;
      }
    }
    else
    {
      mVelNorm.push_back(0);
    }
  }
  if (startDrop != -1)
  {
    int duration = mPoints.size() - startDrop;
    if (duration >= numTimesteps)
    {
      for (int j = startDrop; j < mPoints.size(); j++)
      {
        mDropPointForStillness[j] = true;
      }
    }
    startDrop = -1;
  }
}

//==============================================================================
/// This is a useful measurement to test if the marker just never moves from
/// its starting point (generally if your optical setup accidentally captured
/// a shiny object that is fixed in place as a marker).
s_t LabeledMarkerTrace::getMaxMarkerMovementFromStart()
{
  if (mPoints.size() < 2)
  {
    return 0.0;
  }

  s_t maxDist = 0.0;
  Eigen::Vector3s startPoint = mPoints[0];
  for (int i = 1; i < mPoints.size(); i++)
  {
    s_t dist = (mPoints[i] - startPoint).norm();
    if (dist > maxDist)
    {
      maxDist = dist;
    }
  }
  return maxDist;
}

//==============================================================================
/// This merges point clouds over time, to create a set of raw MarkerTraces
/// over time. These traces can then be intelligently merged using any desired
/// algorithm.
std::vector<LabeledMarkerTrace> LabeledMarkerTrace::createRawTraces(
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations,
    s_t mergeDistance)
{
  std::vector<LabeledMarkerTrace> traces;
  std::vector<int> activeTraces;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    // 1. Only count as "active" the traces that had a point on the last frame
    std::vector<int> tracesToRemove;
    for (int i = 0; i < activeTraces.size(); i++)
    {
      if (traces[activeTraces[i]].lastTimestep() < t - 1)
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
        if (traces[activeTraces[j]].getLastLabel() != markerNames[i])
        {
          dist += 0.04;
        }
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
int MarkersErrorReport::getNumTimesteps()
{
  return markerObservationsAttemptedFixed.size();
}

//==============================================================================
std::map<std::string, Eigen::Vector3s>
MarkersErrorReport::getMarkerMapOnTimestep(int t)
{
  return markerObservationsAttemptedFixed[t];
}

//==============================================================================
std::vector<std::string> MarkersErrorReport::getMarkerNamesOnTimestep(int t)
{
  std::vector<std::string> keys;
  for (auto& pair : markerObservationsAttemptedFixed[t])
  {
    keys.push_back(pair.first);
  }
  return keys;
}

//==============================================================================
Eigen::Vector3s MarkersErrorReport::getMarkerPositionOnTimestep(
    int t, std::string marker)
{
  return markerObservationsAttemptedFixed[t][marker];
}

//==============================================================================
RippleReductionProblem::RippleReductionProblem(
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    s_t dt)
  : mDt(dt)
{
  for (int t = 0; t < markerObservations.size(); t++)
  {
    for (auto& pair : markerObservations[t])
    {
      if (std::find(mMarkerNames.begin(), mMarkerNames.end(), pair.first)
          == mMarkerNames.end())
      {
        mMarkerNames.push_back(pair.first);
      }
    }
  }

  for (std::string& markerName : mMarkerNames)
  {
    mObserved[markerName] = Eigen::VectorXs::Zero(markerObservations.size());
    mOriginalMarkers[markerName] = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
        3, markerObservations.size());
    mMarkers[markerName] = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
        3, markerObservations.size());

    for (int t = 0; t < markerObservations.size(); t++)
    {
      if (markerObservations[t].count(markerName))
      {
        mObserved[markerName](t) = 1.0;
        mOriginalMarkers[markerName].col(t)
            = markerObservations[t].at(markerName);
        mMarkers[markerName].col(t) = markerObservations[t].at(markerName);
      }
    }

    // Start with all support planes facing upwards
    mSupportPlanes[markerName] = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
        3, markerObservations.size());
    mSupportPlanes[markerName].row(2).setConstant(1.0);
  }
}

//==============================================================================
int RippleReductionProblem::dropSuspiciousPoints(MarkersErrorReport* report)
{
  int dropped = 0;
  for (std::string markerName : mMarkerNames)
  {
    std::vector<int> observedTimesteps;
    for (int t = 0; t < mMarkers[markerName].cols(); t++)
    {
      if (mObserved[markerName](t) == 1)
      {
        observedTimesteps.push_back(t);
      }
    }
    // Keep the first two points no matter what
    int lastObserved = 1;
    Eigen::Vector3s vLast
        = (mMarkers[markerName].col(observedTimesteps[1])
           - mMarkers[markerName].col(observedTimesteps[0]))
          / (mDt * (observedTimesteps[1] - observedTimesteps[0]));
    // Go through and only accept points that still "fit" the existing
    // trajectory, starting from the first two observed points
    for (int i = 2; i < observedTimesteps.size(); i++)
    {
      // Compute the velocity from the last observed to the current point
      Eigen::Vector3s vNow
          = (mMarkers[markerName].col(observedTimesteps[i])
             - mMarkers[markerName].col(observedTimesteps[lastObserved]))
            / (mDt * (observedTimesteps[i] - observedTimesteps[lastObserved]));

      Eigen::Vector3s acc
          = (vNow - vLast)
            / (mDt * (observedTimesteps[i] - observedTimesteps[lastObserved]));
      if (acc.norm() > 1000.0)
      {
        // Drop this frame
        mObserved[markerName](observedTimesteps[i]) = 0;
        if (report != nullptr)
        {
          report->warnings.push_back(
              "Dropping marker " + markerName + " for suspicious acceleration ("
              + std::to_string(acc.norm()) + "m/s^2) on frame "
              + std::to_string(observedTimesteps[i]));
        }
        dropped++;
      }
      else
      {
        lastObserved = i;
        vLast = vNow;
      }
    }

    /*
    for (int t = 0; t < mMarkers.cols() - 2; t++)
    {
      if (mObserved(t) && mObserved(t + 1) && mObserved(t + 2))
      {
        Eigen::Vector3s vNow = mMarkers.col(t + 1) - mMarkers.col(t);
        Eigen::Vector3s vNext = mMarkers.col(t + 2) - mMarkers.col(t + 1);
        if ((vNext - vNow).norm() > vNow.norm())
        {
          // Drop this frame
          mObserved(t) = 0;
        }
      }
    }
    // Drop looking backward
    for (int t = 2; t < mMarkers.cols(); t++)
    {
      if (mObserved(t) && mObserved(t - 1) && mObserved(t - 2))
      {
        Eigen::Vector3s vNow = mMarkers.col(t) - mMarkers.col(t - 1);
        Eigen::Vector3s vLast = mMarkers.col(t - 1) - mMarkers.col(t - 2);
        if ((vNow - vLast).norm() > vNow.norm())
        {
          // Drop this frame
          mObserved(t) = 0;
        }
      }
    }
    */
  }
  return dropped;
}

//==============================================================================
void RippleReductionProblem::interpolateMissingPoints()
{
  for (std::string markerName : mMarkerNames)
  {
    int startUnobserved = -1;
    for (int t = 0; t < mMarkers[markerName].cols(); t++)
    {
      if (mObserved[markerName](t) == 0 && startUnobserved == -1)
      {
        startUnobserved = t;
      }
      else if (mObserved[markerName](t) == 1 && startUnobserved != -1)
      {
        if (startUnobserved > 0)
        {
          int beforeStartTimestep = startUnobserved - 1;
          Eigen::Vector3s start = mMarkers[markerName].col(beforeStartTimestep);
          int afterEndTimestep = t;
          Eigen::Vector3s end = mMarkers[markerName].col(afterEndTimestep);

          int duration = afterEndTimestep - beforeStartTimestep;

          for (int j = 1; j < duration; j++)
          {
            Eigen::Vector3s blend = start + (end - start) * ((s_t)j / duration);
            mMarkers[markerName].col(beforeStartTimestep + j) = blend;
          }
        }
        startUnobserved = -1;
      }
    }
  }
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>
RippleReductionProblem::smooth(
    MarkersErrorReport* report,
    bool useSparse,
    bool useIterativeSolver,
    int solverIterations)
{
  dropSuspiciousPoints(report);
  interpolateMissingPoints();
  for (std::string markerName : mMarkerNames)
  {
    // Figure out where the marker observations begin and end
    int firstObserved = -1;
    int lastObserved = 0;
    for (int t = 0; t < mMarkers[markerName].cols(); t++)
    {
      if (mObserved[markerName](t) == 1)
      {
        if (firstObserved == -1)
        {
          firstObserved = t;
        }
        if (t > lastObserved)
        {
          lastObserved = t;
        }
      }
    }
    if (firstObserved == -1)
      firstObserved = 0;
    int duration = (lastObserved - firstObserved) + 1;

    // Smooth only the window during which we observed the marker (don't smooth
    // to the 0's on frames where we weren't observing the marker)
    AccelerationSmoother smoother(
        duration, 0.3, 1.0, useSparse, useIterativeSolver);
    smoother.setIterations(solverIterations);
    mMarkers[markerName].block(0, firstObserved, 3, duration) = smoother.smooth(
        mMarkers[markerName].block(0, firstObserved, 3, duration));
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> markers;
  if (mMarkerNames.size() > 0)
  {
    for (int t = 0; t < mMarkers[mMarkerNames[0]].cols(); t++)
    {
      markers.emplace_back();
      for (std::string& markerName : mMarkerNames)
      {
        // markers[markers.size() - 1][markerName] =
        // mMarkers[markerName].col(t);
        if (mObserved[markerName](t) == 1)
        {
          markers[markers.size() - 1][markerName] = mMarkers[markerName].col(t);
        }
      }
    }
  }
  return markers;
}

//==============================================================================
void RippleReductionProblem::saveToGUI(std::string markerName, std::string path)
{
  server::GUIRecording server;

  std::string originalLayerName = "Original Path";
  Eigen::Vector4s originalLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  server.createLayer(originalLayerName, originalLayerColor);
  std::string smoothedLayerName = "De-rippled Path";
  Eigen::Vector4s smoothedLayerColor = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  server.createLayer(smoothedLayerName, smoothedLayerColor);
  std::string planesLayerName = "Support Planes";
  Eigen::Vector4s planesLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  server.createLayer(planesLayerName, planesLayerColor, false);
  std::string timestampLayerName = "Timestamps";
  Eigen::Vector4s timestampLayerColor = Eigen::Vector4s(0.7, 0.7, 0.7, 1.0);
  server.createLayer(timestampLayerName, timestampLayerColor, false);
  std::string smoothTimestampLayerName = "Smooth Timestamps";
  Eigen::Vector4s smoothTimestampLayerColor
      = Eigen::Vector4s(0.7, 0.7, 0.7, 1.0);
  server.createLayer(
      smoothTimestampLayerName, smoothTimestampLayerColor, false);

  std::vector<Eigen::Vector3s> originalPath;
  std::vector<Eigen::Vector3s> smoothedPath;
  for (int t = 0; t < mMarkers[markerName].cols(); t++)
  {
    smoothedPath.push_back(mMarkers[markerName].col(t));
    server.createBox(
        "smooth_timestamp_" + std::to_string(t),
        0.02 * Eigen::Vector3s::Ones(),
        mMarkers[markerName].col(t),
        Eigen::Vector3s::Zero(),
        smoothTimestampLayerColor,
        smoothTimestampLayerName);
    server.setObjectTooltip(
        "smooth_timestamp_" + std::to_string(t), std::to_string(t));

    if (mObserved[markerName](t) == 1)
    {
      originalPath.push_back(mOriginalMarkers[markerName].col(t));

      std::vector<Eigen::Vector3s> planePoints;
      planePoints.push_back(mOriginalMarkers[markerName].col(t));
      planePoints.push_back(
          mOriginalMarkers[markerName].col(t)
          + mSupportPlanes[markerName].col(t) * 0.03);
      server.createLine(
          "support_at_" + std::to_string(t),
          planePoints,
          planesLayerColor,
          planesLayerName);

      server.createBox(
          "timestamp_" + std::to_string(t),
          0.02 * Eigen::Vector3s::Ones(),
          mOriginalMarkers[markerName].col(t),
          Eigen::Vector3s::Zero(),
          timestampLayerColor,
          timestampLayerName);
      server.setObjectTooltip(
          "timestamp_" + std::to_string(t), std::to_string(t));
    }
    else
    {
      if (originalPath.size() > 0)
      {
        server.createLine(
            "original_end_at_" + std::to_string(t),
            originalPath,
            originalLayerColor,
            originalLayerName);
        originalPath.clear();
      }
      /*
      if (smoothedPath.size() > 0)
      {
        server.createLine(
            "smoothed_end_at_" + std::to_string(t),
            smoothedPath,
            smoothedLayerColor,
            smoothedLayerName);
        smoothedPath.clear();
      }
      */
    }
  }
  if (originalPath.size() > 0)
  {
    server.createLine(
        "original_end", originalPath, originalLayerColor, originalLayerName);
    originalPath.clear();
  }
  if (smoothedPath.size() > 0)
  {
    server.createLine(
        "smoothed_end", smoothedPath, smoothedLayerColor, smoothedLayerName);
    smoothedPath.clear();
  }
  server.saveFrame();

  server.writeFramesJson(path);
}

//==============================================================================
/// This will go through original marker data and attempt to detect common
/// anomalies, generate warnings to help the user fix their own issues, and
/// produce fixes where possible.
std::shared_ptr<MarkersErrorReport> MarkerFixer::generateDataErrorsReport(
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    s_t dt,
    bool dropProlongedStillness,
    bool rippleReduce,
    bool rippleReduceUseSparse,
    bool rippleReduceUseIterativeSolver,
    int rippleReduceSolverIterations)
{
  std::shared_ptr<MarkersErrorReport> report
      = std::make_shared<MarkersErrorReport>();

  // 1. Attempt to detect marker flips that occur partway through the trajectory

  // 1.1. Collect markers into continuous traces. Break marker traces that imply
  // a velocity greater than 20 m/s
  std::vector<LabeledMarkerTrace> traces
      = LabeledMarkerTrace::createRawTraces(markerObservations, dt * 20.0);

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
    traces[i].filterTimestepsBasedOnAcc(dt, 1000.0);
    if (dropProlongedStillness)
    {
      traces[i].filterTimestepsBasedOnProlongedStillness(dt, 0.001, 10);
    }
    traceLabels.push_back(bestLabel);
  }

  // 2. Emit the corrected marker observations

  std::vector<std::map<std::string, Eigen::Vector3s>> correctedObservations;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    std::vector<std::tuple<std::string, Eigen::Vector3s, std::string>>
        droppedMarkerWarningsFrame;

    std::map<std::string, Eigen::Vector3s> frame;
    for (int j = 0; j < traces.size(); j++)
    {
      int index = traces[j].getIndexForTimestep(t);
      if (index != -1)
      {
        // Ignore points that we specifically asked to drop
        if (index < traces[j].mDropPointForAcc.size()
            && traces[j].mDropPointForAcc[index])
        {
          report->warnings.push_back(
              "Marker " + traceLabels[j]
              + " dropped for accelerating too fast ("
              + std::to_string((double)traces[j].mAccNorm[index])
              + " m/s^2) on frame " + std::to_string(t));
          droppedMarkerWarningsFrame.emplace_back(
              traces[j].mMarkerLabels[index],
              traces[j].mPoints[index],
              "accelerating too fast ("
                  + std::to_string((double)traces[j].mAccNorm[index])
                  + " m/s^2)");
        }
        else if (
            index < traces[j].mDropPointForStillness.size()
            && traces[j].mDropPointForStillness[index])
        {
          report->warnings.push_back(
              "Marker " + traceLabels[j] + " dropped for velocity too slow ("
              + std::to_string((double)traces[j].mVelNorm[index])
              + " m/s) on sequential frame " + std::to_string(t));
          droppedMarkerWarningsFrame.emplace_back(
              traces[j].mMarkerLabels[index],
              traces[j].mPoints[index],
              "velocity too slow ("
                  + std::to_string((double)traces[j].mVelNorm[index])
                  + " m/s^2)");
        }
        else if (index >= 0 && index < traces[j].mPoints.size())
        {
          Eigen::Vector3s point = traces[j].mPoints[index];
          if (abs(point(0)) > 1e+6 || abs(point(1)) > 1e+6 || abs(point(2)) > 1e+6) {
            report->warnings.push_back(
                "Marker " + traceLabels[j] + " dropped for being too far away ("
                + std::to_string((double)point.norm())
                + " m) on frame " + std::to_string(t));
            droppedMarkerWarningsFrame.emplace_back(
                traces[j].mMarkerLabels[index],
                traces[j].mPoints[index],
                "being too far away ("
                    + std::to_string((double)point.norm())
                    + " m)");
            continue;
          }

          frame[traceLabels[j]] = point;
        }
        else {
          std::cout << "ERROR(MarkerFixed): index " << index << " into traces[j].mPoints out of bounds "
                    << traces[j].mPoints.size() << std::endl;
        }
      }
    }
    correctedObservations.push_back(frame);
    report->droppedMarkerWarnings.push_back(droppedMarkerWarningsFrame);
  }

  // 2.1. Count the number of observations of each marker name
  std::map<std::string, int> observationCount;
  for (const auto& obs : correctedObservations)
  {
    for (const auto& pair : obs)
    {
      if (observationCount.count(pair.first) == 0)
      {
        observationCount[pair.first] = 0;
      }
      observationCount[pair.first]++;
    }
  }

  // 2.2. Drop any marker observations that occur too infrequently.
  std::vector<std::string> markersToDrop;
  std::map<std::string, s_t> markerDropPercentage;
  for (auto& pair : observationCount)
  {
    s_t percentage = (s_t)pair.second / (s_t)markerObservations.size();
    if (percentage < 0.1)
    {
      report->warnings.push_back(
          "Dropping marker \"" + pair.first + "\", only on "
          + std::to_string(percentage * 100) + " percent of frames");
      markersToDrop.push_back(pair.first);
      markerDropPercentage[pair.first] = percentage;
    }
  }
  for (int t = 0; t < correctedObservations.size(); t++)
  {
    std::map<std::string, Eigen::Vector3s>& obs = correctedObservations[t];
    for (std::string drop : markersToDrop)
    {
      if (obs.count(drop) > 0)
      {
        report->droppedMarkerWarnings[t].emplace_back(
            drop,
            obs.at(drop),
            "dropped for appearing only on "
                + std::to_string(markerDropPercentage[drop] * 100)
                + " percent of frames");
        obs.erase(drop);
      }
    }
  }

  if (rippleReduce)
  {
    RippleReductionProblem rippleReduction(correctedObservations, dt);
    correctedObservations = rippleReduction.smooth(
        report.get(),
        rippleReduceUseSparse,
        rippleReduceUseIterativeSolver,
        rippleReduceSolverIterations);
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
            report->warnings.push_back(
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

  for (int t = 0; t < correctedObservations.size(); t++)
  {
    std::vector<std::pair<std::string, std::string>> markersRenamedFromToFrame;
    report->markersRenamedFromTo.push_back(markersRenamedFromToFrame);
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

    for (int i = 0; i < merged.mTimes.size(); i++)
    {
      int t = merged.mTimes[i];
      if (merged.mMarkerLabels[i] != pair.first)
      {
        report->markersRenamedFromTo[t].emplace_back(
            merged.mMarkerLabels[i], pair.first);
      }
    }

    std::vector<std::string> warnings
        = merged.emitWarningsAboutLabelChange(pair.first);
    for (int j = 0; j < warnings.size(); j++)
    {
      report->warnings.push_back(warnings[j]);
    }
  }

  report->markerObservationsAttemptedFixed = correctedObservations;

  // 4. Go through and check for NaNs and out-of-bounds values
  for (int t = 0; t < report->markerObservationsAttemptedFixed.size(); t++)
  {
    for (auto& pair : report->markerObservationsAttemptedFixed[t])
    {
      if (pair.second.hasNaN())
      {
        std::cout << "ERROR(MarkerFixer): Marker " << pair.first
                  << " has NaN on frame " << t << std::endl;
      }
      if (abs(pair.second(0)) > 1e+6 || abs(pair.second(1)) > 1e+6
          || abs(pair.second(2)) > 1e+6)
      {
        std::cout << "ERROR(MarkerFixer): Marker " << pair.first
                  << " has suspiciously large value on frame " << t
                  << std::endl;
      }
    }
  }

  return report;
}

} // namespace biomechanics
} // namespace dart
