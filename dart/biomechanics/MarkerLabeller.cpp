#include "dart/biomechanics/MarkerLabeller.hpp"

#include <limits>
#include <string>

#include "dart/dynamics/Joint.hpp"
#include "dart/math/AssignmentMatcher.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
/// This constructor will compute jointFingerprints from the joints passed in
MarkerTrace::MarkerTrace(int time, Eigen::Vector3s firstPoint)
  : mMarkerLabel("")
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
  : mTimes(times),
    mPoints(points),
    mJointFingerprints(jointFingerprints),
    mMarkerLabel("")
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
  std::vector<int> activeTraces;
  for (int t = 0; t < pointClouds.size(); t++)
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
        s_t dist = traces[activeTraces[j]].pointToAppendDistance(
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
        activeTraces.push_back(traces.size() - 1);
        assert(
            traces.at(activeTraces.at(activeTraces.size() - 1)).mPoints.size()
            == 1);
      }
      else
      {
        traces[activeTraces[map(i)]].appendPoint(t, pointClouds[t][i]);
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
/// Returns true if these traces don't overlap
bool MarkerTrace::overlap(MarkerTrace& toAppend)
{
  return !((mMaxTime < toAppend.mMinTime) || (toAppend.mMaxTime < mMinTime));
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
/// This gets the mean and variance of all the joint fingerprints.
std::map<std::string, std::tuple<Eigen::Vector2s, s_t>>
MarkerTrace::getJointFingerprintStats()
{
  // 1. Compute the center point for each joint
  std::map<std::string, int> numSamples;
  std::map<std::string, Eigen::Vector2s> centerPoint;
  for (auto& map : mJointFingerprints)
  {
    for (auto& pair : map)
    {
      if (centerPoint.count(pair.first) == 0)
      {
        numSamples[pair.first] = 0;
        centerPoint[pair.first] = Eigen::Vector2s::Zero();
      }
      numSamples[pair.first]++;
      centerPoint[pair.first] += pair.second;
    }
  }
  for (auto& pair : centerPoint)
  {
    centerPoint[pair.first] /= numSamples.at(pair.first);
  }

  // 2. Compute the squared average distances from the center point for each
  // joint
  std::map<std::string, s_t> jointVariance;
  for (auto& map : mJointFingerprints)
  {
    for (auto& pair : map)
    {
      if (jointVariance.count(pair.first) == 0)
      {
        jointVariance[pair.first] = 0;
      }
      jointVariance[pair.first]
          += (pair.second - centerPoint[pair.first]).squaredNorm();
    }
  }

  std::map<std::string, std::tuple<Eigen::Vector2s, s_t>> result;
  for (auto& pair : centerPoint)
  {
    jointVariance[pair.first] /= numSamples.at(pair.first);
    result[pair.first] = std::tuple<Eigen::Vector2s, s_t>(
        centerPoint[pair.first], jointVariance.at(pair.first));
  }
  return result;
}

//==============================================================================
void MarkerLabeller::setSkeleton(std::shared_ptr<dynamics::Skeleton> skeleton)
{
  mSkeleton = skeleton;
}

//==============================================================================
void MarkerLabeller::matchUpJointToSkeletonJoint(
    std::string jointName, std::string skeletonJointName)
{
  mJointToSkelJointNames[jointName] = skeletonJointName;
}

/*
//==============================================================================
LabelledMarkers MarkerLabeller::labelPointClouds(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
    s_t mergeMarkersThreshold)
{
  LabelledMarkers result;

  // 1. Divide the markers into continuous trajectory segments
  std::vector<MarkerTrace> traces = MarkerTrace::createRawTraces(pointClouds);

  // 2. Get the joint centers
  std::vector<std::map<std::string, Eigen::Vector3s>> jointCenters
      = guessJointLocations(pointClouds);

  // 3. Run IK+scaling on each timestep of the trajectory to match the joint
  // centers at that frame.
  std::vector<Eigen::VectorXs> poses;
  std::vector<Eigen::VectorXs> scales;

  // 3.1. Convert the skeleton to have any Euler joints as ball joints
  std::shared_ptr<dynamics::Skeleton> skeletonBallJoints
      = mSkeleton->convertSkeletonToBallJoints();

  // 3.2. Calculate problem size
  int problemDim = skeletonBallJoints->getNumDofs()
                   + skeletonBallJoints->getGroupScaleDim();

  // 3.3. Set our initial guess for IK to whatever the current pose of the
  // skeleton is
  Eigen::VectorXs initialPos = Eigen::VectorXs::Ones(problemDim);
  initialPos.segment(0, skeletonBallJoints->getNumDofs())
      = skeletonBallJoints->convertPositionsToBallSpace(
          skeletonBallJoints->getPositions());
  initialPos.segment(
      skeletonBallJoints->getNumDofs(), skeletonBallJoints->getGroupScaleDim())
      = skeletonBallJoints->getGroupScales();

  // 3.4. Linearize the joint center guesses
  Eigen::VectorXs jointCenterVec
      = Eigen::VectorXs::Zero(mJointToSkelJointNames.size() * 3);
  std::vector<std::string> jointNames;
  std::vector<dynamics::Joint*> ballSkelJoints;
  std::vector<dynamics::Joint*> skelJoints;
  for (auto& pair : mJointToSkelJointNames)
  {
    jointNames.push_back(pair.first);
    skelJoints.push_back(mSkeleton->getJoint(pair.second));
    ballSkelJoints.push_back(skeletonBallJoints->getJoint(pair.second));
  }

  // 3.5. Perform IK on each timestep
  for (int i = 0; i < jointCenters.size(); i++)
  {
    // 3.5.1. Copy this frame's joint center guesses over
    for (int j = 0; j < jointNames.size(); j++)
    {
      jointCenterVec.segment<3>(j * 3) = jointCenters[i][jointNames[j]];
    }

    // 3.5. Actually solve the IK
    s_t result = math::solveIK(
        initialPos,
        jointCenterVec.size(),
        // Set positions
        [&skeletonBallJoints, this](
            const Eigen::VectorXs pos, bool clamp) {
          skeletonBallJoints->setPositions(
              pos.segment(0, skeletonBallJoints->getNumDofs()));

          if (clamp)
          {
            // 1. Map the position back into eulerian space
            mSkeleton->setPositions(mSkeleton->convertPositionsFromBallSpace(
                pos.segment(0, skeletonBallJoints->getNumDofs())));
            // 2. Clamp the position to limits
            mSkeleton->clampPositionsToLimits();
            // 3. Map the position back into SO3 space
            skeletonBallJoints->setPositions(
                mSkeleton->convertPositionsToBallSpace(
                    mSkeleton->getPositions()));
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = skeletonBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = skeletonBallJoints->getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          mSkeleton->setGroupScales(newScales);
          skeletonBallJoints->setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, skeletonBallJoints->getNumDofs())
              = skeletonBallJoints->getPositions();
          clampedPos.segment(
              skeletonBallJoints->getNumDofs(),
              skeletonBallJoints->getGroupScaleDim())
              = newScales;
          return clampedPos;
        },
        // Compute the Jacobian
        [&skeletonBallJoints, &jointCenterVec, &ballSkelJoints](
            Eigen::VectorXs& diff,
            Eigen::MatrixXs& jac)
{
  Eigen::VectorXs jointPoses
      = skeletonBallJoints->getJointWorldPositions(ballSkelJoints);
  diff = jointCenterVec - jointPoses;

  assert(
      jac.cols()
      == skeletonBallJoints->getNumDofs()
             + skeletonBallJoints->getGroupScaleDim());
  assert(jac.rows() == jointCenterVec.size());
  jac.setZero();

  jac.block(0, 0, jointCenterVec.size(), skeletonBallJoints->getNumDofs())
      = skeletonBallJoints->getJointWorldPositionsJacobianWrtJointPositions(
          ballSkelJoints);
  jac.block(
      0,
      skeletonBallJoints->getNumDofs(),
      jointCenterVec.size(),
      skeletonBallJoints->getGroupScaleDim())
      = skeletonBallJoints->getJointWorldPositionsJacobianWrtGroupScales(
          ballSkelJoints);
},
        // Generate a random restart position
        [&skeletonBallJoints, &skelJoints, this](Eigen::VectorXs& val) {
  val.segment(0, skeletonBallJoints->getNumDofs())
      = mSkeleton->convertPositionsToBallSpace(
          mSkeleton->getRandomPoseForJoints(skelJoints));
  val.segment(
         skeletonBallJoints->getNumDofs(),
         skeletonBallJoints->getGroupScaleDim())
      .setConstant(1.0);
        },
        math::IKConfig()
            .setMaxStepCount(150)
            .setConvergenceThreshold(1e-10)
            // .setLossLowerBound(1e-8)
            .setLossLowerBound(0.01)
            .setMaxRestarts(10)
            .setLogOutput(false));
std::cout << "Best result: " << result << std::endl;

poses.push_back(mSkeleton->getPositions());
scales.push_back(mSkeleton->getGroupScales());
}

// 4. Compute fingerprints on each body, based on the results of IK
for (MarkerTrace& trace : traces)
{
  trace.computeJointFingerprints(jointCenters, jointParents);
}

// 5. Now work through the traces, and create/assign markers for each trace

// Note: These markers are defined in as (x, r): distance along the joint
// axis, and radial distance from the axis. At this stage we're deliberately
// leaving the angle of attachment to the joint ambiguous, because we don't
// yet have enough information to determine that.
std::map<std::string, std::pair<std::string, Eigen::Vector2s>> markers;

for (MarkerTrace& trace : traces)
{
  std::map<std::string, std::tuple<Eigen::Vector2s, s_t>> stats
      = trace.getJointFingerprintStats();

  // 4.1. Find the best available joint, in terms of minimum variance wrt to
  // this trace. For now, we'll just always attach to this joint. A more
  // sophisticated mechanism might look at multiple good joints, and choose
  // between them somehow.
  s_t minVariance = std::numeric_limits<s_t>::infinity();
  std::string minVarianceJoint = "";
  Eigen::Vector2s minVarianceOffset = Eigen::Vector2s::Zero();
  for (auto jointStat : stats)
  {
    s_t jointVariance = std::get<1>(jointStat.second);
    if (jointVariance < minVariance)
    {
      minVariance = jointVariance;
      minVarianceJoint = jointStat.first;
      minVarianceOffset = std::get<0>(jointStat.second);
    }
  }

  // 4.2. Check if there's already a marker for this joint and offset, and if
  // so, if that marker is already assigned during our trace
  bool foundMarker = false;

  for (auto markerPair : markers)
  {
    std::string markerName = markerPair.first;
    std::string jointName = std::get<0>(markerPair.second);
    if (jointName == minVarianceJoint)
    {
      Eigen::Vector2s offset = std::get<1>(markerPair.second);
      s_t dist = (offset - minVarianceOffset).norm();
      if (dist < mergeMarkersThreshold)
      {
        // 4.2.1. We've got a hit! Check whether this marker is already
        // assigned to any traces that overlap this one
        bool anyOverlap = false;
        for (MarkerTrace& otherTrace : traces)
        {
          if (&trace == &otherTrace)
            continue;
          if (otherTrace.mMarkerLabel == markerName
              && trace.overlap(otherTrace))
          {
            anyOverlap = true;
            break;
          }
        }

        // 4.2.2. We're clear to merge this marker in!
        if (!anyOverlap)
        {
          foundMarker = true;
          trace.mMarkerLabel = markerName;
          break;
        }
      }
    }
  }

  if (!foundMarker)
  {
    // Create a new marker
    std::string markerName = std::to_string(markers.size());
    markers[markerName] = std::tuple<std::string, Eigen::Vector2s>(
        minVarianceJoint, minVarianceOffset);
    trace.mMarkerLabel = markerName;
  }
}

// 5. Now we can ditch the traces, and reconstruct labeled point clouds
std::vector<std::map<std::string, Eigen::Vector3s>> labeledPointClouds;

// 5.1. Create all the blank timestep objects we need
int maxTimestep = 0;
for (MarkerTrace& trace : traces)
{
  if (trace.mMaxTime > maxTimestep)
  {
    maxTimestep = trace.mMaxTime;
  }
}
for (int i = 0; i < maxTimestep; i++)
{
  labeledPointClouds.emplace_back();
}

// 5.2. Populate the labeled point clouds
for (MarkerTrace& trace : traces)
{
  for (int i = 0; i < trace.mTimes.size(); i++)
  {
    labeledPointClouds[trace.mTimes[i]][trace.mMarkerLabel] = trace.mPoints[i];
  }
}

// TODO: add guessed hand and foot 3D orientations
// (SO3) to the neural network output. That disambiguates the IK problem.

// Maybe we just adapt the MarkerFitter code to take this as an entry point!
// Joint center guesses

// 6. Optimize joint centers to smooth out joint center trajectories
// TODO:keenon
std::vector<std::map<std::string, Eigen::Vector3s>> smoothedJointCenters
    = jointCenters;
// 7. Do IK with target skeleton?
// TODO:keenon

// 8. Find more exact marker offsets by taking an average.

// 9. Run bilevel optimization from MarkerFitter?

result.markerObservations = labeledPointClouds;
result.markerOffsetsXR = markers;
result.jointCenterGuesses = jointCenters;
return result;
}
*/

//==============================================================================
LabelledMarkers MarkerLabeller::labelPointClouds(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds,
    s_t mergeMarkersThreshold)
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

  // 4. Now work through the traces, and create/assign markers for each trace
  // Note: These markers are defined in as (x, r): distance along the joint
  // axis, and radial distance from the axis. At this stage we're deliberately
  // leaving the angle of attachment to the joint ambiguous, because we don't
  // yet have enough information to determine that.
  std::map<std::string, std::pair<std::string, Eigen::Vector2s>> markers;

  for (MarkerTrace& trace : traces)
  {
    std::map<std::string, std::tuple<Eigen::Vector2s, s_t>> stats
        = trace.getJointFingerprintStats();

    // 4.1. Find the best available joint, in terms of minimum variance wrt to
    // this trace. For now, we'll just always attach to this joint. A more
    // sophisticated mechanism might look at multiple good joints, and choose
    // between them somehow.
    s_t minVariance = std::numeric_limits<s_t>::infinity();
    std::string minVarianceJoint = "";
    Eigen::Vector2s minVarianceOffset = Eigen::Vector2s::Zero();
    for (auto jointStat : stats)
    {
      s_t jointVariance = std::get<1>(jointStat.second);
      if (jointVariance < minVariance)
      {
        minVariance = jointVariance;
        minVarianceJoint = jointStat.first;
        minVarianceOffset = std::get<0>(jointStat.second);
      }
    }

    // 4.2. Check if there's already a marker for this joint and offset, and if
    // so, if that marker is already assigned during our trace
    bool foundMarker = false;

    for (auto markerPair : markers)
    {
      std::string markerName = markerPair.first;
      std::string jointName = std::get<0>(markerPair.second);
      if (jointName == minVarianceJoint)
      {
        Eigen::Vector2s offset = std::get<1>(markerPair.second);
        s_t dist = (offset - minVarianceOffset).norm();
        if (dist < mergeMarkersThreshold)
        {
          // 4.2.1. We've got a hit! Check whether this marker is already
          // assigned to any traces that overlap this one
          bool anyOverlap = false;
          for (MarkerTrace& otherTrace : traces)
          {
            if (&trace == &otherTrace)
              continue;
            if (otherTrace.mMarkerLabel == markerName
                && trace.overlap(otherTrace))
            {
              anyOverlap = true;
              break;
            }
          }

          // 4.2.2. We're clear to merge this marker in!
          if (!anyOverlap)
          {
            foundMarker = true;
            trace.mMarkerLabel = markerName;
            break;
          }
        }
      }
    }

    if (!foundMarker)
    {
      // Create a new marker
      std::string markerName = std::to_string(markers.size());
      markers[markerName] = std::tuple<std::string, Eigen::Vector2s>(
          minVarianceJoint, minVarianceOffset);
      trace.mMarkerLabel = markerName;
    }
  }

  // 5. Now we can ditch the traces, and reconstruct labeled point clouds
  std::vector<std::map<std::string, Eigen::Vector3s>> labeledPointClouds;

  // 5.1. Create all the blank timestep objects we need
  int maxTimestep = 0;
  for (MarkerTrace& trace : traces)
  {
    if (trace.mMaxTime > maxTimestep)
    {
      maxTimestep = trace.mMaxTime;
    }
  }
  for (int i = 0; i < maxTimestep; i++)
  {
    labeledPointClouds.emplace_back();
  }

  // 5.2. Populate the labeled point clouds
  for (MarkerTrace& trace : traces)
  {
    for (int i = 0; i < trace.mTimes.size(); i++)
    {
      labeledPointClouds[trace.mTimes[i]][trace.mMarkerLabel]
          = trace.mPoints[i];
    }
  }

  // TODO: add guessed hand and foot 3D orientations
  // (SO3) to the neural network output. That disambiguates the IK problem.

  // Maybe we just adapt the MarkerFitter code to take this as an entry point!
  // Joint center guesses

  // 6. Optimize joint centers to smooth out joint center trajectories
  // TODO:keenon
  std::vector<std::map<std::string, Eigen::Vector3s>> smoothedJointCenters
      = jointCenters;
  // 7. Do IK with target skeleton?
  // TODO:keenon

  // 8. Find more exact marker offsets by taking an average.

  // 9. Run bilevel optimization from MarkerFitter?

  result.markerObservations = labeledPointClouds;
  // result.markerOffsets = markers;
  result.jointCenterGuesses = jointCenters;
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