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
StreamingMarkerTraces::StreamingMarkerTraces(
    int totalClasses, int numWindows, int stride, int maxMarkersPerTimestep)
  : mTotalClasses(totalClasses),
    mNumWindows(numWindows),
    mStride(stride),
    mStrideCursor(0),
    mMaxMarkersPerTimestep(maxMarkersPerTimestep),
    mWindowFeatures(Eigen::Matrix<s_t, 4, Eigen::Dynamic>::Random(
        4, numWindows * maxMarkersPerTimestep)),
    mWindowFeaturesTrace(
        Eigen::VectorXi::Ones(numWindows * maxMarkersPerTimestep) * -1),
    mWindowCursor(0),
    mTraceTagCursor(0),
    mTraceMaxJoinDistance(0.05),
    mTraceTimeoutMillis(300),
    mFeatureMaxStrideToleranceMillis(10)
{
  mWindowFeatures.row(3).setZero();
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
  mWindowFeatures = Eigen::Matrix<s_t, 4, Eigen::Dynamic>::Random(
      4, mNumWindows * mMaxMarkersPerTimestep);
  mWindowFeatures.row(3).setZero();
  mWindowFeaturesTrace
      = Eigen::VectorXi::Ones(mNumWindows * mMaxMarkersPerTimestep) * -1;
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

  ///////////////////////////////////////////////////////////
  // 1. Assign all the markers to traces
  ///////////////////////////////////////////////////////////

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
        pointsToTraces(i, tr) = (traceHeadProjected - markers[i]).norm();
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

      // We can now say that this point gets the same class as the trace.

      resultClasses[minMarker] = mTracePredictedClass[minTrace];
      resultTraceTags[minMarker] = mTraceTag[minTrace];

      // Now we want to update the trace head and velocity

      // Before we update anything we need to figure out the time delta, and
      // distance
      const long elapsedMillis = timestamp - mTraceLastSeenTimestamp[minTrace];
      const double elapsedSeconds = (double)elapsedMillis / 1000.0;
      const Eigen::Vector3s dist = mTraceHeads[minTrace] - markers[minMarker];

      // Now we can update the trace data
      mTraceHeads[minTrace] = markers[minMarker];
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
    if (resultTraceTags[i] == -1)
    {
      // This point is unassigned, so we want to assign it to a new trace
      mTraceHeads.push_back(markers[i]);
      mTraceVelocities.push_back(Eigen::Vector3s::Zero());
      mTraceLastSeenTimestamp.push_back(timestamp);
      mTraceTag.push_back(mTraceTagCursor);
      mTraceLogits.push_back(Eigen::VectorXs::Zero(mTotalClasses));
      // The unknown class is the last class
      mTracePredictedClass.push_back(mTotalClasses - 1);
      resultClasses[i] = mTotalClasses - 1;
      resultTraceTags[i] = mTraceTagCursor;
      mTraceTagCursor++;
    }
  }

  ///////////////////////////////////////////////////////////
  // 2. On the appropriate strides
  ///////////////////////////////////////////////////////////

  mStrideCursor--;
  if (mStrideCursor <= 0)
  {
    // Decrement all the timestep features on the observed data
    mWindowFeatures.row(3) -= Eigen::VectorXs::Ones(mWindowFeatures.cols());

    int observeMarkers = std::min((int)markers.size(), mMaxMarkersPerTimestep);

    // Write all the marker data
    for (int i = 0; i < observeMarkers; i++)
    {
      // Write the features matrix with the marker positions
      mWindowFeatures.col((mWindowCursor + i) % mWindowFeatures.cols())
          .head<3>()
          = markers[i];
      // Start the timestamps on these entries as the full window count
      mWindowFeatures(3, (mWindowCursor + i) % mWindowFeatures.cols())
          = mNumWindows - 1;
      // Write the trace tags we just found
      mWindowFeaturesTrace[(mWindowCursor + i) % mWindowFeatures.cols()]
          = resultTraceTags[i];
    }

    // Reset the stride counter
    mStrideCursor = mStride;
    // Increment the window cursor by the appropriate amount
    mWindowCursor = (mWindowCursor + observeMarkers) % mWindowFeatures.cols();
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
StreamingMarkerTraces::getTraceFeatures(bool center)
{
  // Sub-select just those elements that haven't timed out
  int numFeatures = 0;
  for (int i = 0; i < mWindowFeatures.cols(); i++)
  {
    if (mWindowFeatures(3, i) >= 0)
    {
      numFeatures++;
    }
  }
  Eigen::MatrixXs features = Eigen::MatrixXs(4, numFeatures);
  Eigen::VectorXi traces = Eigen::VectorXi(numFeatures);
  int cursor = 0;
  for (int i = 0; i < mWindowFeatures.cols(); i++)
  {
    if (mWindowFeatures(3, i) >= 0)
    {
      features.col(cursor) = mWindowFeatures.col(i);
      traces(cursor) = mWindowFeaturesTrace(i);
      cursor++;
    }
  }

  // If we're centering the point cloud, then do that
  if (center)
  {
    features.block(0, 0, 3, features.cols())
        = features.block(0, 0, 3, features.cols()).colwise()
          - features.block(0, 0, 3, features.cols()).rowwise().mean();
  }

  return std::make_pair(features, traces);
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
        mTraceLogits[j] = logits.col(i);
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

} // namespace biomechanics
} // namespace dart
