#include "dart/biomechanics/MarkerLabeller.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/math/AssignmentMatcher.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

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
    std::vector<int> times, std::vector<Eigen::Vector3s> points)
  : mTimes(times), mPoints(points), mMarkerLabel("")
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
/// Each possible combination of (trace, body) can create a marker. So we can
/// compute some summary statistics for each body we could assign this trace
/// to.
void MarkerTrace::computeBodyMarkerStats(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<Eigen::VectorXs> posesOverTime,
    std::vector<Eigen::VectorXs> scalesOverTime)
{
  // 1. Go through all the timestamps, and compute local coordinates for the
  // trace in each body.

  std::vector<std::map<std::string, Eigen::Vector3s>> bodyFingerprints;
  std::vector<std::map<std::string, s_t>> bodyRootDist;
  std::vector<std::map<std::string, s_t>> bodyClosestPointDist;
  for (int i = 0; i < mPoints.size(); i++)
  {
    Eigen::Vector3s p = mPoints[i];
    int t = mTimes[i];

    assert(t >= 0 && t < posesOverTime.size());
    skel->setPositions(posesOverTime[t]);
    skel->setGroupScales(scalesOverTime[t]);

    std::map<std::string, Eigen::Vector3s> jointWorldPositions
        = skel->getJointWorldPositionsMap();

    bodyFingerprints.emplace_back();
    bodyRootDist.emplace_back();
    bodyClosestPointDist.emplace_back();

    for (int b = 0; b < skel->getNumBodyNodes(); b++)
    {
      auto* body = skel->getBodyNode(b);

      // Compute local coordinates in the body space for this body
      bodyFingerprints.at(bodyFingerprints.size() - 1)[body->getName()]
          = (body->getWorldTransform().inverse() * p)
                .cwiseQuotient(body->getScale());

      // Compute the distance from the parent joint for this body
      Eigen::Vector3s parentJointWorldPosition
          = jointWorldPositions[body->getParentJoint()->getName()];
      bodyRootDist.at(bodyRootDist.size() - 1)[body->getName()]
          = (parentJointWorldPosition - p).norm();

      // Compute the polygon defined by all the joints attached to this body
      if (body->getNumChildJoints() > 0)
      {
        std::vector<Eigen::Vector3s> corner;
        corner.push_back(parentJointWorldPosition);
        for (int j = 0; j < body->getNumChildJoints(); j++)
        {
          corner.push_back(
              jointWorldPositions[body->getChildJoint(j)->getName()]);
        }

        // Compute the distance to the nearest point on the polygon
        s_t dist = std::numeric_limits<s_t>::infinity();
        for (int x = 0; x < corner.size(); x++)
        {
          for (int y = x + 1; y < corner.size(); y++)
          {
            Eigen::Vector3s start = corner[x];
            Eigen::Vector3s end = corner[y];
            s_t d = math::distanceToSegment(start, end, p);
            if (d < dist)
            {
              dist = d;
            }
          }
        }
        bodyClosestPointDist.at(
            bodyClosestPointDist.size() - 1)[body->getName()]
            = dist;
      }
      else
      {
        // Just default to distance from the root joint
        bodyClosestPointDist.at(
            bodyClosestPointDist.size() - 1)[body->getName()]
            = bodyRootDist.at(bodyRootDist.size() - 1)[body->getName()];
      }
    }
  }

  std::vector<std::string> bodyNames;
  for (int b = 0; b < skel->getNumBodyNodes(); b++)
  {
    bodyNames.push_back(skel->getBodyNode(b)->getName());
  }

  // 2. Compute the summary statistics for each body's marker pos

  // 2.1. Compute the center points
  std::map<std::string, s_t> avgRootJointDist;
  for (std::string& bodyName : bodyNames)
  {
    avgRootJointDist[bodyName] = 0.0;
    mBodyMarkerOffsets[bodyName] = Eigen::Vector3s::Zero();
    mBodyClosestPointDistance[bodyName] = 0.0;

    mBodyRootJointDistVariance[bodyName] = 0.0;
    mBodyMarkerOffsetVariance[bodyName] = 0.0;
  }

  for (int i = 0; i < bodyFingerprints.size(); i++)
  {
    for (std::string& bodyName : bodyNames)
    {
      avgRootJointDist[bodyName] += bodyRootDist.at(i).at(bodyName);
      mBodyMarkerOffsets[bodyName] += bodyFingerprints.at(i).at(bodyName);
      mBodyClosestPointDistance[bodyName]
          += bodyClosestPointDist.at(i).at(bodyName);
    }
  }
  for (std::string& bodyName : bodyNames)
  {
    avgRootJointDist[bodyName] /= bodyFingerprints.size();
    mBodyMarkerOffsets[bodyName] /= bodyFingerprints.size();
    mBodyClosestPointDistance[bodyName] /= bodyFingerprints.size();
  }

  // 2.2. Compute the squared average distances from the center points
  for (int i = 0; i < bodyFingerprints.size(); i++)
  {
    for (std::string& bodyName : bodyNames)
    {
      s_t x = bodyRootDist.at(i).at(bodyName) - avgRootJointDist.at(bodyName);
      mBodyRootJointDistVariance[bodyName] += x * x;
      mBodyMarkerOffsetVariance[bodyName]
          += (bodyFingerprints.at(i).at(bodyName)
              - mBodyMarkerOffsets.at(bodyName))
                 .squaredNorm();
    }
  }

  // 3. Normalize all the maps
  s_t maxMarkerOffsetVariance = 0.0;
  for (auto pair : mBodyMarkerOffsetVariance)
  {
    if (pair.second > maxMarkerOffsetVariance)
    {
      maxMarkerOffsetVariance = pair.second;
    }
  }
  if (maxMarkerOffsetVariance > 0)
  {
    for (std::string body : bodyNames)
    {
      mBodyMarkerOffsetVariance[body] /= maxMarkerOffsetVariance;
    }
  }

  s_t maxRootDistVariance = 0.0;
  for (auto pair : mBodyRootJointDistVariance)
  {
    if (pair.second > maxRootDistVariance)
    {
      maxRootDistVariance = pair.second;
    }
  }
  if (maxRootDistVariance > 0)
  {
    for (std::string body : bodyNames)
    {
      mBodyRootJointDistVariance[body] /= maxRootDistVariance;
    }
  }

  s_t maxClosestPointDist = 0.0;
  for (auto pair : mBodyClosestPointDistance)
  {
    if (pair.second > maxClosestPointDist)
    {
      maxClosestPointDist = pair.second;
    }
  }
  if (maxClosestPointDist > 0)
  {
    for (std::string body : bodyNames)
    {
      mBodyClosestPointDistance[body] /= maxClosestPointDist;
    }
  }
}

//==============================================================================
/// Each possible combination of (trace, body) can create a marker. This
/// returns a score for a given body, for how "good" of a marker that body
/// would create when combined with this trace. Lower is better.
s_t MarkerTrace::computeBodyMarkerLoss(std::string bodyName)
{
  if (mBodyMarkerOffsetVariance.count(bodyName) == 0)
    return std::numeric_limits<double>::infinity();

  return mBodyMarkerOffsetVariance.at(bodyName)
         + mBodyRootJointDistVariance.at(bodyName)
         + mBodyClosestPointDistance.at(bodyName);
}

//==============================================================================
/// This finds the best body to pair this trace with (using the stats from
/// computeBodyMarkerStats()) and returns the best marker
std::pair<std::string, Eigen::Vector3s> MarkerTrace::getBestMarker()
{
  std::pair<std::string, Eigen::Vector3s> marker
      = std::pair<std::string, Eigen::Vector3s>("", Eigen::Vector3s::Zero());
  s_t bestLoss = std::numeric_limits<double>::infinity();
  for (auto& pair : mBodyMarkerOffsets)
  {
    std::string bodyName = pair.first;
    s_t loss = computeBodyMarkerLoss(bodyName);
    if (loss < bestLoss)
    {
      marker = pair;
      bestLoss = loss;
    }
  }
  return marker;
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
  for (int t : toAppend.mTimes)
  {
    times.push_back(t);
  }
  for (Eigen::Vector3s p : toAppend.mPoints)
  {
    points.push_back(p);
  }

  return MarkerTrace(times, points);
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

//==============================================================================
MarkerLabeller::~MarkerLabeller()
{
}

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
      = mSkeleton->convertPositionsToBallSpace(mSkeleton->getPositions());
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
  for (int t = 0; t < jointCenters.size(); t++)
  {
    // 3.5.1. Copy this frame's joint center guesses over
    for (int j = 0; j < jointNames.size(); j++)
    {
      jointCenterVec.segment<3>(j * 3) = jointCenters[t][jointNames[j]];
    }

    // 3.5. Actually solve the IK
    s_t result = math::solveIK(
        initialPos,
        jointCenterVec.size(),
        // Set positions
        [&skeletonBallJoints, this](const Eigen::VectorXs pos, bool clamp) {
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
            Eigen::VectorXs& diff, Eigen::MatrixXs& jac) {
          Eigen::VectorXs jointPoses
              = skeletonBallJoints->getJointWorldPositions(ballSkelJoints);
          diff = jointCenterVec - jointPoses;

          assert(
              jac.cols()
              == skeletonBallJoints->getNumDofs()
                     + skeletonBallJoints->getGroupScaleDim());
          assert(jac.rows() == jointCenterVec.size());
          jac.setZero();

          jac.block(
              0, 0, jointCenterVec.size(), skeletonBallJoints->getNumDofs())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtJointPositions(
                        ballSkelJoints);
          jac.block(
              0,
              skeletonBallJoints->getNumDofs(),
              jointCenterVec.size(),
              skeletonBallJoints->getGroupScaleDim())
              = skeletonBallJoints
                    ->getJointWorldPositionsJacobianWrtGroupScales(
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
            .setMaxStepCount(t == 0 ? 150 : 50)
            .setConvergenceThreshold(1e-10)
            .setLossLowerBound(1e-8)
            // .setLossLowerBound(0.001)
            .setMaxRestarts(t == 0 ? 10 : 1)
            .setLogOutput(false));
    (void)result;
    // std::cout << "Best result: " << result << std::endl;

    poses.push_back(mSkeleton->getPositions());
    scales.push_back(mSkeleton->getGroupScales());

    initialPos.segment(0, skeletonBallJoints->getNumDofs())
        = mSkeleton->convertPositionsToBallSpace(mSkeleton->getPositions());
    initialPos.segment(
        skeletonBallJoints->getNumDofs(),
        skeletonBallJoints->getGroupScaleDim())
        = mSkeleton->getGroupScales();
  }

  // 4. Compute fingerprints on each body, based on the results of IK
  for (MarkerTrace& trace : traces)
  {
    trace.computeBodyMarkerStats(mSkeleton, poses, scales);
  }

  // 5. Now work through the traces, and create/assign markers for each trace

  // Note: These markers are defined in as (x, r): distance along the joint
  // axis, and radial distance from the axis. At this stage we're deliberately
  // leaving the angle of attachment to the joint ambiguous, because we don't
  // yet have enough information to determine that.
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>> markers;

  for (MarkerTrace& trace : traces)
  {
    std::pair<std::string, Eigen::Vector3s> bestMarker = trace.getBestMarker();

    // 4.2. Check if there's already a marker for this body and offset, and
    // if so, if that marker is already assigned during our trace
    bool foundMarker = false;

    for (auto markerPair : markers)
    {
      std::string markerName = markerPair.first;
      std::string bodyName = markerPair.second.first;
      if (bodyName == bestMarker.first)
      {
        Eigen::Vector3s offset = markerPair.second.second;
        s_t dist = (offset - bestMarker.second).norm();
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
      markers[markerName] = bestMarker;
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
  for (int i = 0; i <= maxTimestep; i++)
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

  result.markerObservations = labeledPointClouds;
  result.markerOffsets = markers;
  result.jointCenterGuesses = jointCenters;
  result.traces = traces;
  return result;
}

//==============================================================================
/// This takes in a set of labeled point clouds over time, and runs the
/// labeller over unlabeled copies of those point clouds, and then scores the
/// reconstruction accuracy.
void MarkerLabeller::evaluate(
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

  /// The thing we really care about in scoring is for each frame, how accurate
  /// is the attachment and offset of the marker.

  int totalBodies = 0;
  int totalCorrectBodies = 0;
  s_t totalAverageError = 0.0;

  for (MarkerTrace& trace : output.traces)
  {
    std::string guessBody = output.markerOffsets[trace.mMarkerLabel].first;
    Eigen::Vector3s guessOffset
        = output.markerOffsets[trace.mMarkerLabel].second;

    int correctBody = 0;
    std::map<std::string, int> correctMarkerCount;
    s_t averageError = 0.0;

    for (int i = 0; i < trace.mTimes.size(); i++)
    {
      int t = trace.mTimes[i];
      Eigen::Vector3s p = trace.mPoints[i];

      // Find the true label and offset
      std::string markerName = "";
      std::string trueBody = "";
      Eigen::Vector3s trueOffset = Eigen::Vector3s::Zero();
      bool foundPoint = false;
      for (auto& pair : labeledPointClouds[t])
      {
        if ((pair.second - p).squaredNorm() < 1e-9)
        {
          markerName = pair.first;
          trueBody = markerOffsets.at(pair.first).first;
          trueOffset = markerOffsets.at(pair.first).second;
          foundPoint = true;
          break;
        }
      }
      (void)foundPoint;
      assert(foundPoint);

      if (trueBody == guessBody)
      {
        correctBody++;
        averageError += (trueOffset - guessOffset).norm();
      }

      if (correctMarkerCount.count(markerName) == 0)
      {
        correctMarkerCount[markerName] = 0;
      }
      correctMarkerCount[markerName]++;
    }

    totalCorrectBodies += correctBody;
    totalBodies += trace.mTimes.size();
    totalAverageError += averageError;

    if (correctBody > 0)
    {
      averageError /= correctBody;
    }

    // If the trace is entirely correct
    if (correctBody == trace.mTimes.size())
    {
      std::cout << "CORRECT: " << correctBody << " with error " << averageError
                << std::endl;
    }
    // If the trace is wrong
    else
    {
      // If there's more than one true-body, then this trace shouldn't be a
      // single trace
      if (correctMarkerCount.size() > 1)
      {
        std::cout << "This trace falsely collapsed multiple marker traces!"
                  << std::endl;
        // TODO: more debugging info. This seems like a rare case, so won't type
        // that out yet.
      }
      else if (correctMarkerCount.size() == 0)
      {
        // This shouldn't happen, because that would imply that this is an empty
        // trace
        assert(false && "No empty traces");
      }
      else
      {
        std::string correctMarkerName = correctMarkerCount.begin()->first;
        std::string correctBodyName = markerOffsets.at(correctMarkerName).first;

        std::cout << "Trace picked the wrong body! Correct marker "
                  << correctMarkerName << " is attached to body "
                  << correctBodyName << ", but chose " << guessBody
                  << std::endl;
        std::cout << "Correct body " << correctBodyName
                  << " stats:" << std::endl;
        std::cout << "\tMarker variance: "
                  << trace.mBodyMarkerOffsetVariance.at(correctBodyName)
                  << std::endl;
        std::cout << "\tJoint dist variance: "
                  << trace.mBodyRootJointDistVariance.at(correctBodyName)
                  << std::endl;
        std::cout << "\tClosest point distance: "
                  << trace.mBodyClosestPointDistance.at(correctBodyName)
                  << std::endl;
        std::cout << "\tLoss: " << trace.computeBodyMarkerLoss(correctBodyName)
                  << std::endl;
        std::cout << "Incorrect guessed body " << guessBody
                  << " stats:" << std::endl;
        std::cout << "\tMarker variance: "
                  << trace.mBodyMarkerOffsetVariance.at(guessBody) << std::endl;
        std::cout << "\tJoint dist variance: "
                  << trace.mBodyRootJointDistVariance.at(guessBody)
                  << std::endl;
        std::cout << "\tClosest point distance: "
                  << trace.mBodyClosestPointDistance.at(guessBody) << std::endl;
        std::cout << "\tLoss: " << trace.computeBodyMarkerLoss(guessBody)
                  << std::endl;
      }
    }
  }

  if (totalCorrectBodies > 0)
  {
    totalAverageError /= totalCorrectBodies;
  }

  std::cout << "**************" << std::endl;
  std::cout << "Body Labelling Accuracy: "
            << ((s_t)totalCorrectBodies / totalBodies) * 100 << " percent"
            << std::endl;
  std::cout << "Of correctly labeled, average offset difference: "
            << totalAverageError * 100 << " cm" << std::endl;
  std::cout << "**************" << std::endl;

  /*
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

  if (realMarkerNames.size() == reconstructedMarkerNames.size())
  {
    std::map<std::string, std::string> realToRecon
        = math::AssignmentMatcher::assignKeysToKeys(
            realMarkerNames,
            reconstructedMarkerNames,
            [&](std::string real, std::string reconstructed) {
              s_t dist = 0.0;

              for (int t = 0; t < labeledPointClouds.size(); t++)
              {
                Eigen::Vector3s reconPoint
                    = output.markerObservations[t].at(reconstructed);
                Eigen::Vector3s realPoint = labeledPointClouds[t].at(real);
                dist += (reconPoint - realPoint).squaredNorm();
              }

              if (dist == 0)
              {
                return std::numeric_limits<s_t>::max();
              }
              return 1.0 / dist;
            });

    std::cout << "Reconstructed an identical number of markers to the ground "
                 "truth data"
              << std::endl;

    for (std::string& real : realMarkerNames)
    {
      std::string realBody = markerOffsets.at(real).first;
      Eigen::Vector3s realOffset = markerOffsets.at(real).second;

      std::string reconstructed = realToRecon[real];
      std::string reconBody = output.markerOffsets.at(reconstructed).first;
      Eigen::Vector3s reconOffset
          = output.markerOffsets.at(reconstructed).second;

      std::cout << "Marker \"" << real << "\":" << std::endl;
      if (realBody == reconBody)
      {
        std::cout << " - Correct body: \"" << realBody
                  << "\", offset difference: "
                  << (realOffset - reconOffset).norm() << std::endl;
      }
      else
      {
        std::cout << " - Real body: \"" << realBody << "\""
                  << ", reconstructed body: \"" << reconBody << "\""
                  << std::endl;
      }
    }
  }
  else
  {
    int totalPoints = 0;
    int matchedBodyPoints = 0;
    s_t totalOffsetError = 0.0;

    for (int i = 0; i < labeledPointClouds.size(); i++)
    {
      for (auto& pair : labeledPointClouds[i])
      {
        std::pair<std::string, Eigen::Vector3s> goldMarker
            = markerOffsets.at(pair.first);
        Eigen::Vector3s point = pair.second;

        for (auto& reconPair : output.markerObservations[i])
        {
          if ((point - reconPair.second).squaredNorm() < 1e-7)
          {
            std::pair<std::string, Eigen::Vector3s> reconMarker
                = output.markerOffsets.at(reconPair.first);
            totalPoints++;
            if (reconMarker.first == goldMarker.first)
            {
              matchedBodyPoints++;
              totalOffsetError
                  += (reconMarker.second - goldMarker.second).norm();
            }
            break;
          }
        }
      }
    }

    if (matchedBodyPoints > 0)
      totalOffsetError /= matchedBodyPoints;
    s_t bodyAccuracy = (s_t)totalPoints / matchedBodyPoints;

    std::cout << "On " << totalPoints << " points, got body right "
              << (bodyAccuracy * 100) << " percent of the time." << std::endl;
    std::cout << "On the correctly assigned points, marker offset error was "
              << totalOffsetError << " on average" << std::endl;
  }
  */
}

//==============================================================================
NeuralMarkerLabeller::NeuralMarkerLabeller(
    std::function<std::vector<std::map<std::string, Eigen::Vector3s>>(
        const std::vector<std::vector<Eigen::Vector3s>>&)> jointCenterPredictor)
  : mJointCenterPredictor(jointCenterPredictor)
{
}

//==============================================================================
NeuralMarkerLabeller::~NeuralMarkerLabeller()
{
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>
NeuralMarkerLabeller::guessJointLocations(
    const std::vector<std::vector<Eigen::Vector3s>>& pointClouds)
{
  return mJointCenterPredictor(pointClouds);
}

//==============================================================================
MarkerLabellerMock::~MarkerLabellerMock()
{
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
void MarkerLabellerMock::setMockJointLocations(
    std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime)
{
  mJointsOverTime = jointsOverTime;
}

} // namespace biomechanics
} // namespace dart