#include "dart/biomechanics/MarkerLabeller.hpp"

#include "dart/math/AssignmentMatcher.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
/// This constructor will compute jointFingerprints from the joints passed in
MarkerTrace::MarkerTrace(int time, Eigen::Vector3s firstPoint)
{
  mTimes.push_back(time);
  mPoints.push_back(firstPoint);
  mMinTime = time;
  mMaxTime = time;
}

//==============================================================================
/// This is the direct constructor
MarkerTrace::MarkerTrace(
    std::vector<int> times,
    std::vector<Eigen::Vector3s> points,
    std::vector<std::map<std::string, Eigen::Vector2s>> jointFingerprints)
  : mTimes(times), mPoints(points), mJointFingerprints(jointFingerprints)
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
void MarkerTrace::appendPoint(int time, Eigen::Vector3s point)
{
  mTimes.push_back(time);
  mPoints.push_back(point);
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
s_t MarkerTrace::pointToAppendDistance(
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
    Eigen::Vector3d v = (lastPoint - mPoints.at(mPoints.size() - 2))
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
/// This merges point clouds over time, to create a set of raw MarkerTraces
/// over time. These traces can then be intelligently merged using any desired
/// algorithm.
std::vector<MarkerTrace> MarkerTrace::createRawTraces(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
    s_t mergeDistance,
    int mergeFrames)
{
  std::vector<MarkerTrace> traces;
  std::vector<MarkerTrace*> activeTraces;
  for (int t = 0; t < pointClouds.size(); t++)
  {
    // 1. Only count as "active" the traces that are within `mergeFrames` of now
    std::vector<int> tracesToRemove;
    for (int i = 0; i < activeTraces.size(); i++)
    {
      if (activeTraces[i]->lastTimestep() < t - mergeFrames)
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
    if (pointClouds[t].size() == 0)
    {
      continue;
    }

    // 2. Compute affinity scores between active traces and points
    Eigen::MatrixXs weights
        = Eigen::MatrixXs(pointClouds[t].size(), activeTraces.size());
    for (int i = 0; i < pointClouds[t].size(); i++)
    {
      for (int j = 0; j < activeTraces.size(); j++)
      {
        s_t dist = activeTraces[j]->pointToAppendDistance(
            t, pointClouds[t][i], true);
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
        traces.emplace_back(t, pointClouds[t][i]);
        assert(traces.at(traces.size() - 1).mPoints.size() == 1);
        activeTraces.push_back(&traces.at(traces.size() - 1));
        assert(activeTraces.at(activeTraces.size() - 1)->mPoints.size() == 1);
      }
      else
      {
        activeTraces[map(i)]->appendPoint(t, pointClouds[t][i]);
      }
    }
  }

  return traces;
}

//==============================================================================
/// This will create fingerprints from the joint history
void MarkerTrace::computeJointFingerprints(
    std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime,
    std::map<std::string, std::string> jointParents)
{
  for (int i = 0; i < mPoints.size(); i++)
  {
    Eigen::Vector3s p = mPoints[i];
    int t = mTimes[i];

    assert(t >= 0 && t < jointsOverTime.size());
    std::map<std::string, Eigen::Vector3s>& jointCenters = jointsOverTime[t];

    mJointFingerprints.emplace_back();
    for (auto& pair : jointCenters)
    {
      // Don't count the root joint, since it has no parent
      if (jointParents.count(pair.first) == 0)
      {
        continue;
      }

      // compute distance along joint, and distance from joint axis

      std::string parentJointName = jointParents.at(pair.first);
      std::string bodyName = parentJointName + "::" + pair.first;

      Eigen::Vector3s& parentJoint = jointCenters.at(parentJointName);
      Eigen::Vector3s& childJoint = pair.second;

      Eigen::Vector3s dir = (childJoint - parentJoint).normalized();

      // This is the distance parallel to the body, along the body
      s_t x = p.dot(dir) - parentJoint.dot(dir);

      // This is the radial distance to the nearest point on the body line
      Eigen::Vector3s diff = p - parentJoint;
      diff -= diff.dot(dir) * dir;
      s_t r = diff.norm();

      mJointFingerprints.at(mJointFingerprints.size() - 1)[bodyName]
          = Eigen::Vector2s(x, r);
    }
  }
}

//==============================================================================
/// This merges two MarkerTrace's together, to create a new trace object
MarkerTrace MarkerTrace::concat(MarkerTrace& toAppend)
{
  std::vector<int> times = std::vector<int>(mTimes);
  std::vector<Eigen::Vector3s> points = std::vector<Eigen::Vector3s>(mPoints);
  std::vector<std::map<std::string, Eigen::Vector2s>> jointFingerprints
      = std::vector<std::map<std::string, Eigen::Vector2s>>(mJointFingerprints);

  for (int t : toAppend.mTimes)
  {
    times.push_back(t);
  }
  for (Eigen::Vector3s p : toAppend.mPoints)
  {
    points.push_back(p);
  }
  for (auto& fingerprint : toAppend.mJointFingerprints)
  {
    jointFingerprints.push_back(fingerprint);
  }

  return MarkerTrace(times, points, jointFingerprints);
}

//==============================================================================
/// This returns when this MarkerTrace begins (inclusive)
int MarkerTrace::firstTimestep()
{
  return mMinTime;
}

//==============================================================================
/// This returns when this MarkerTrace ends (inclusive)
int MarkerTrace::lastTimestep()
{
  return mMaxTime;
}

//==============================================================================
/// This gets the variance of all the joint fingerprints, and returns the
/// lowest one. This is used for scoring beam search alternatives.
s_t MarkerTrace::getBestJointFingerprintVariance()
{
  assert(false && "Not implemented yet!");
  return 0.0;
}

//==============================================================================
LabelledMarkers MarkerLabeller::labelPointClouds(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds)
{
  LabelledMarkers result;

  // 1. Divide the markers into continuous trajectory segments
  std::vector<MarkerTrace> traces = MarkerTrace::createRawTraces(pointClouds);

  // 2. Get the joint centers
  std::vector<std::map<std::string, Eigen::Vector3s>> jointCenters
      = guessJointLocations(pointClouds);
  std::map<std::string, std::string> jointParents = getJointParents();

  // 3. Compute the local joint fingerprints
  for (MarkerTrace& trace : traces)
  {
    trace.computeJointFingerprints(jointCenters, jointParents);
  }

  // 4. Now beam search through combinations of traces
  assert(false && "Not implemented yet!");

  return result;
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>
MarkerLabellerMock::guessJointLocations(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds)
{
  (void)pointClouds;
  // This is a mock, so we just return the mocked value
  return mJointsOverTime;
}

//==============================================================================
std::map<std::string, std::string> MarkerLabellerMock::getJointParents()
{
  return mJointParents;
}

//==============================================================================
void MarkerLabellerMock::setMockJointLocations(
    std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime,
    std::map<std::string, std::string> jointParents)
{
  mJointsOverTime = jointsOverTime;
  mJointParents = jointParents;
}

//==============================================================================
/// This takes in a set of labeled point clouds over time, and runs the
/// labeller over unlabeled copies of those point clouds, and then scores the
/// reconstruction accuracy.
void MarkerLabellerMock::evaluate(
    const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
        markerOffsets,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        labeledPointClouds)
{
  // 1. Create an unlabeled copy of the point clouds
  std::vector<std::vector<Eigen::Vector3s>> pointClouds;
  for (int t = 0; t < labeledPointClouds.size(); t++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    for (auto& pair : labeledPointClouds[t])
    {
      pointCloud.push_back(pair.second);
    }
    pointClouds.push_back(pointCloud);
  }

  // 2. Run our labeller
  LabelledMarkers output = labelPointClouds(pointClouds);

  // 3. Now score the reconstruction accuracy

  // 3.1. List out the names of the markers we're going to map between
  std::vector<std::string> realMarkerNames;
  std::vector<std::string> reconstructedMarkerNames;

  for (auto& pair : markerOffsets)
  {
    realMarkerNames.push_back(pair.first);
  }
  for (auto& pair : output.markerOffsets)
  {
    reconstructedMarkerNames.push_back(pair.first);
  }

  // 3.2. Let's figure out which reconstructed markers correspond to which
  // actual markers. There may be different numbers of reconstructed vs actual
  // markers.
  std::map<std::string, std::string> reconToReal
      = math::AssignmentMatcher::assignKeysToKeys(
          reconstructedMarkerNames,
          realMarkerNames,
          [&](std::string reconstructed, std::string real) {
            const std::string& reconstructedBody
                = output.markerOffsets.at(reconstructed).first;
            const std::string& realBody = markerOffsets.at(real).first;
            if (reconstructedBody != realBody)
            {
              return -1 * std::numeric_limits<s_t>::infinity();
            }

            const Eigen::Vector3s& reconstructedOffset
                = output.markerOffsets.at(reconstructed).second;
            const Eigen::Vector3s& realOffset = markerOffsets.at(real).second;

            s_t dist = (reconstructedOffset - realOffset).squaredNorm();
            return 1.0 / dist;
          });

  // 3.3. Now we can score how many markers were detected, and how many were not
  std::cout << "Matched " << reconToReal.size() << " of "
            << realMarkerNames.size() << " real markers" << std::endl;
  if (reconstructedMarkerNames.size() > reconToReal.size())
  {
    std::cout << "Detected "
              << (reconstructedMarkerNames.size() - reconToReal.size())
              << " markers which didn't match anything" << std::endl;
  }
  s_t avgDist = 0.0;
  s_t maxDist = 0.0;
  for (auto& pair : reconToReal)
  {
    s_t dist = (output.markerOffsets.at(pair.first).second
                - markerOffsets.at(pair.second).second)
                   .norm();
    avgDist += dist;
    if (dist > maxDist)
      maxDist = dist;
  }
  avgDist /= reconToReal.size();
  std::cout << "Average distance between markers: " << avgDist << std::endl;

  // 3.4. Now we can go through and check the accuracy of the labels on each
  // frame.
  s_t avgAccuracy = 0.0;
  s_t maxAccuracy = 0.0;
  s_t minAccuracy = 1.0;

  for (int i = 0; i < labeledPointClouds.size(); i++)
  {
    std::map<std::string, Eigen::Vector3s> goldLabels = labeledPointClouds[i];
    std::map<std::string, Eigen::Vector3s> reconLabels
        = output.markerObservations[i];

    int matches = 0;
    for (auto& goldPair : goldLabels)
    {
      for (auto& reconPair : reconLabels)
      {
        // When the 3d points match up precisely, we've found the original point
        if (goldPair.second == reconPair.second)
        {
          // Check whether the labels match
          if (reconToReal.count(reconPair.first) > 0
              && reconToReal.at(reconPair.first) == goldPair.first)
          {
            // We match!
            matches++;
          }
        }
      }
    }

    s_t frameAccuracy = matches / goldLabels.size();
    if (frameAccuracy > maxAccuracy)
    {
      maxAccuracy = frameAccuracy;
    }
    if (frameAccuracy < minAccuracy)
    {
      minAccuracy = frameAccuracy;
    }
    avgAccuracy += frameAccuracy;
  }
  avgAccuracy /= labeledPointClouds.size();

  std::cout << "Average label accuracy on " << labeledPointClouds.size()
            << " frames: " << avgAccuracy << " (min=" << minAccuracy
            << ",max=" << maxAccuracy << ")" << std::endl;
}

} // namespace biomechanics
} // namespace dart