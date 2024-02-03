#include "dart/biomechanics/StreamingMarkerTraces.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
StreamingMarkerTraces::StreamingMarkerTraces(int totalClasses, int bufferSize)
  : mTotalClasses(totalClasses),
    mBufferSize(bufferSize),
    mPointBuffer(Eigen::Matrix<s_t, 3, Eigen::Dynamic>(3, bufferSize)),
    mBufferCursor(0),
    mPointBufferTrace(Eigen::VectorXi::Ones(bufferSize) * -1),
    mPointBufferTime(std::vector<long>(bufferSize, -1)),
    mTraceTagCursor(0),
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
  mPointBuffer.setZero();
  mBufferCursor = 0;
  mPointBufferTrace.setConstant(-1);
  for (int i = 0; i < mPointBufferTime.size(); i++)
  {
    mPointBufferTime[i] = -1;
  }
  mTraceTagCursor = 0;
  mTraceHeads.clear();
  mTraceVelocities.clear();
  mTraceLastSeenTimestamp.clear();
  mTraceTag.clear();
  mTraceLogits.clear();
}

//==============================================================================
/// This method takes in a set of markers, and returns a vector of the
/// predicted classes for each marker, based on classes we have predicted for
/// previous markers, and continuity assumptions. The returned vector will be
/// the same length and order as the input `markers` vector.
std::pair<std::vector<int>, std::vector<int>>
StreamingMarkerTraces::observeMarkers(
    const std::vector<Eigen::Vector3s>& markers, long timestamp)
{
  std::vector<int> resultClasses(markers.size(), -1);
  std::vector<int> resultTraceTags(markers.size(), -1);

  // Do each write separately, so they can be more easily collapsed by the
  // compiler
  for (int i = 0; i < markers.size(); i++)
  {
    mPointBuffer.col((mBufferCursor + i) % mBufferSize) = markers[i];
  }
  for (int i = 0; i < markers.size(); i++)
  {
    mPointBufferTime[(mBufferCursor + i) % mBufferSize] = timestamp;
  }
  for (int i = 0; i < markers.size(); i++)
  {
    mPointBufferTrace[(mBufferCursor + i) % mBufferSize] = -1;
  }

  // First collapse the traces that have timed out, since that saves space in
  // subsequent operations
  for (int tr = 0; tr < mTraceHeads.size(); tr++)
  {
    long elapsedMillis = timestamp - mTraceLastSeenTimestamp[tr];
    if (elapsedMillis > mTraceTimeoutMillis)
    {
      // This trace has timed out, so we need to remove it
      mTraceHeads.erase(mTraceHeads.begin() + tr);
      mTraceVelocities.erase(mTraceVelocities.begin() + tr);
      mTraceLastSeenTimestamp.erase(mTraceLastSeenTimestamp.begin() + tr);
      mTraceTag.erase(mTraceTag.begin() + tr);
      mTraceLogits.erase(mTraceLogits.begin() + tr);
      tr--;
    }
  }

  int pointsAssigned = 0;
  if (mTraceHeads.size() > 0)
  {
    // Now we want to fill in the distances between the points we just received
    // and the projected trace heads
    Eigen::MatrixXs pointsToTraces
        = Eigen::MatrixXs::Zero(markers.size(), mTraceHeads.size());
    for (int tr = 0; tr < mTraceHeads.size(); tr++)
    {
      long elapsedMillis = timestamp - mTraceLastSeenTimestamp[tr];
      // This trace is still active, so we need to project it forward by
      // velocity
      double elapsedSeconds = (double)elapsedMillis / 1000.0;
      (void)elapsedSeconds;
      const Eigen::Vector3s traceHeadProjected
          = mTraceHeads[tr] + (mTraceVelocities[tr] * elapsedSeconds);

      for (int i = 0; i < markers.size(); i++)
      {
        int index = (mBufferCursor + i) % mBufferSize;
        pointsToTraces(i, tr)
            = (traceHeadProjected - mPointBuffer.col(index)).norm();
      }
    }

    // Finally, we want to collapse the points into the traces
    while (pointsAssigned < markers.size())
    {
      // Find the closest point to any trace
      Eigen::MatrixXf::Index minMarker, minTrace;
      s_t minDistance = pointsToTraces.minCoeff(&minMarker, &minTrace);
      if (minDistance > mTraceMaxJoinDistance)
      {
        break;
      }

      // We found a point that's close enough to a trace, so we want to assign
      // it to that trace

      int markerIndex = (mBufferCursor + minMarker) % mBufferSize;
      mPointBufferTrace[markerIndex] = minTrace;

      // We can now say that this point gets the same class as the trace.

      resultClasses[minMarker] = mTracePredictedClass[minTrace];
      resultTraceTags[minMarker] = mTraceTag[minTrace];

      // Now we want to update the trace head and velocity

      // Before we update anything we need to figure out the time delta, and
      // distance
      const long elapsedMillis = timestamp - mTraceLastSeenTimestamp[minTrace];
      const double elapsedSeconds = (double)elapsedMillis / 1000.0;
      const Eigen::Vector3s dist
          = mTraceHeads[minTrace] - mPointBuffer.col(markerIndex);

      // Now we can update the trace data
      mTraceHeads[minTrace] = mPointBuffer.col(markerIndex);
      mTraceVelocities[minTrace] = dist / elapsedSeconds;
      mTraceLastSeenTimestamp[minTrace] = timestamp;

      // Finally, we want to mark the distances as infinity, so we don't assign
      // this point to any other trace, or this trace to any other point
      pointsToTraces.col(minTrace).setConstant(
          std::numeric_limits<s_t>::infinity());
      pointsToTraces.row(minMarker).setConstant(
          std::numeric_limits<s_t>::infinity());
    }
  }

  // Now we want to assign any unassigned points to new traces
  for (int i = 0; i < markers.size(); i++)
  {
    int markerIndex = (mBufferCursor + i) % mBufferSize;
    if (mPointBufferTrace[markerIndex] == -1)
    {
      // This point is unassigned, so we want to assign it to a new trace
      mTraceHeads.push_back(mPointBuffer.col(markerIndex));
      mTraceVelocities.push_back(Eigen::Vector3s::Zero());
      mTraceLastSeenTimestamp.push_back(timestamp);
      mTraceTag.push_back(mTraceTagCursor);
      mTraceLogits.push_back(Eigen::VectorXs::Zero(mTotalClasses));
      mPointBufferTrace[markerIndex] = mTraceTagCursor;
      // The unknown class is the last class
      mTracePredictedClass.push_back(mTotalClasses - 1);
      resultClasses[i] = mTotalClasses - 1;
      resultTraceTags[i] = mTraceTagCursor;
      mTraceTagCursor++;
    }
  }

  mBufferCursor = (mBufferCursor + markers.size()) % mBufferSize;

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
    int numWindows, long windowDuration, long now, bool center)
{
  // 1. First, we do a linear scan through our timestamp data to determine which
  // of the points in the buffer are within the window we care about, so we can
  // size the matrix correctly.
  int lastEntryCursor = mBufferCursor;
  int numEntries = 0;
  int safetyLatch = 0;
  for (int windowCursor = 0; windowCursor < numWindows; windowCursor++)
  {
    long windowCursorTime = now - (windowCursor * windowDuration);

    int lastDistance
        = abs(mPointBufferTime[lastEntryCursor] - windowCursorTime);
    // 1.1. Scan backwards through the buffer to find the first point that's
    // increasing in time distance from our window cursor
    while (true)
    {
      int newCursor = (lastEntryCursor + (mBufferSize - 1)) % mBufferSize;
      int newDistance = abs(mPointBufferTime[newCursor] - windowCursorTime);
      if (newDistance > lastDistance)
      {
        break;
      }
      lastDistance = newDistance;
      lastEntryCursor = newCursor;
      if (safetyLatch++ > mBufferSize
          || (safetyLatch > 0 && mPointBufferTime[lastEntryCursor] == -1))
      {
        // We've looped all the way around the buffer, so we're done
        break;
      }
    }
    if (safetyLatch > mBufferSize
        || (safetyLatch > 0 && mPointBufferTime[lastEntryCursor] == -1))
    {
      // We've looped all the way around the buffer, so we're done
      break;
    }

    // 1.2. Now we check if this minimum distance point is within our acceptable
    // tolerance.
    int distance = abs(mPointBufferTime[lastEntryCursor] - windowCursorTime);
    if (distance > mFeatureMaxStrideToleranceMillis)
    {
      continue;
    }

    // 1.3. Now we want to scan forward from this point to find all the points
    // with the same timestamp.
    long timestamp = mPointBufferTime[lastEntryCursor];
    int forwardCursor = lastEntryCursor;
    int forwardSafetyLatch = 0;
    while (mPointBufferTime[forwardCursor] == timestamp)
    {
      numEntries++;
      forwardCursor = (forwardCursor + 1) % mBufferSize;
      if (forwardSafetyLatch++ > mBufferSize)
      {
        // We've looped all the way around the buffer, so we're done
        break;
      }
    }

    // Consume one more token (our start token), to step backwards
    lastEntryCursor = (lastEntryCursor + (mBufferSize - 1)) % mBufferSize;
    safetyLatch++;
  }

  Eigen::MatrixXs features = Eigen::MatrixXs::Zero(4, numEntries);
  Eigen::VectorXi traceIDs = Eigen::VectorXi::Zero(numEntries);

  // 2. Now we want to fill in the features, which requires the same scan as
  // before.
  lastEntryCursor = mBufferCursor;
  int entryCursor = 0;
  safetyLatch = 0;
  for (int windowCursor = 0; windowCursor < numWindows; windowCursor++)
  {
    long windowCursorTime = now - (windowCursor * windowDuration);

    int lastDistance
        = abs(mPointBufferTime[lastEntryCursor] - windowCursorTime);
    // 2.1. Scan backwards through the buffer to find the first point that's
    // increasing in time distance from our window cursor
    while (true)
    {
      int newCursor = (lastEntryCursor + (mBufferSize - 1)) % mBufferSize;
      int newDistance = abs(mPointBufferTime[newCursor] - windowCursorTime);
      if (newDistance > lastDistance)
      {
        break;
      }
      lastDistance = newDistance;
      lastEntryCursor = newCursor;
      if (safetyLatch++ > mBufferSize
          || (safetyLatch > 0 && mPointBufferTime[lastEntryCursor] == -1))
      {
        // We've looped all the way around the buffer, so we're done
        break;
      }
    }
    if (safetyLatch > mBufferSize
        || (safetyLatch > 0 && mPointBufferTime[lastEntryCursor] == -1))
    {
      // We've looped all the way around the buffer, so we're done
      break;
    }

    // 2.2. Now we check if this minimum distance point is within our acceptable
    // tolerance.
    int distance = abs(mPointBufferTime[lastEntryCursor] - windowCursorTime);
    if (distance > mFeatureMaxStrideToleranceMillis)
    {
      continue;
    }

    // 2.3. Now we want to scan forward from this point to find all the points
    // with the same timestamp.
    long timestamp = mPointBufferTime[lastEntryCursor];
    int forwardCursor = lastEntryCursor;
    int forwardSafetyLatch = 0;
    while (mPointBufferTime[forwardCursor] == timestamp)
    {
      features.col(entryCursor).head<3>() = mPointBuffer.col(forwardCursor);
      features(3, entryCursor) = (numWindows - 1) - windowCursor;
      traceIDs(entryCursor) = mPointBufferTrace(forwardCursor);
      entryCursor++;

      forwardCursor = (forwardCursor + 1) % mBufferSize;
      if (forwardSafetyLatch++ > mBufferSize)
      {
        // We've looped all the way around the buffer, so we're done
        break;
      }
    }

    // Consume one more token (our start token), to step backwards
    lastEntryCursor = (lastEntryCursor + (mBufferSize - 1)) % mBufferSize;
    safetyLatch++;
  }

  if (center)
  {
    features.block(0, 0, 3, numEntries)
        = features.block(0, 0, 3, numEntries).colwise()
          - features.block(0, 0, 3, numEntries).rowwise().mean();
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
    for (int j = 0; j < mTraceTag.size(); j++)
    {
      if (mTraceTag[j] == traceID)
      {
        mTraceLogits[j] += logits.col(i);
        break;
      }
    }
  }

  for (int i = 0; i < mTraceLogits.size(); i++)
  {
    Eigen::MatrixXf::Index minIndex;
    mTraceLogits[i].maxCoeff(&minIndex);
    mTracePredictedClass[i] = minIndex;
  }
}

//==============================================================================
/// This method returns the number of traces we have active. This is mostly
/// here for debugging and testing, because the number of active traces should
/// not be a useful metric for downstream tasks that are just interested in
/// labeled marker clouds.
int StreamingMarkerTraces::getNumTraces()
{
  return mTraceHeads.size();
}

//==============================================================================
/// This is just for testing, and returns our internal points buffer cursor
int StreamingMarkerTraces::getRawPointsBufferCursor()
{
  return mBufferCursor;
}

//==============================================================================
/// This is just for testing, and returns our internal points buffer
Eigen::MatrixXs StreamingMarkerTraces::getRawPointsBuffer()
{
  return mPointBuffer;
}

} // namespace biomechanics
} // namespace dart
