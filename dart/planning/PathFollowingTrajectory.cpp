/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

// Algorithm details and publications: http://www.golems.org/node/1570

#include "dart/planning/PathFollowingTrajectory.hpp"

#include <fstream>
#include <iostream>
#include <limits>

using namespace Eigen;
using namespace std;

namespace dart {
namespace planning {

const double PathFollowingTrajectory::timeStep = 1.0e-3;
const double PathFollowingTrajectory::eps = 1.0e-6;

//==============================================================================
PathFollowingTrajectory::PathFollowingTrajectory(
    const Path& path,
    const VectorXd& maxVelocity,
    const VectorXd& maxAcceleration)
  : path(path),
    maxVelocity(maxVelocity),
    maxAcceleration(maxAcceleration),
    n(maxVelocity.size()),
    valid(true),
    cachedTime(numeric_limits<double>::max())
{
  // debug
  //{
  // ofstream file("maxVelocity.txt");
  // for(double s = 0.0; s < path.getLength(); s += 0.0001) {
  //	double maxVelocity = getAccelerationMaxPathVelocity(s);
  //	if(maxVelocity == numeric_limits<double>::infinity())
  //		maxVelocity = 10.0;
  //	file << s << "  " << maxVelocity << "  " << getVelocityMaxPathVelocity(s)
  //<< endl;
  //}
  // file.close();
  //}

  list<TrajectoryStep> startTrajectory;
  startTrajectory.push_back(TrajectoryStep(0.0, 0.0));
  double afterAcceleration = getMinMaxPathAcceleration(0.0, 0.0, true);
  while (!integrateForward(startTrajectory, afterAcceleration) && valid)
  {
    double beforeAcceleration;
    TrajectoryStep switchingPoint;
    if (getNextSwitchingPoint(
            startTrajectory.back().mPathPos,
            switchingPoint,
            beforeAcceleration,
            afterAcceleration))
    {
      break;
    }
    // cout << "set arrow from " << switchingPoint.pathPos << ", " <<
    // switchingPoint.pathVel - 0.8 << " to " << switchingPoint.pathPos << ", "
    // << switchingPoint.pathVel - 0.3 << endl;
    list<TrajectoryStep> trajectory;
    trajectory.push_back(switchingPoint);
    integrateBackward(trajectory, startTrajectory, beforeAcceleration);
  }

  list<TrajectoryStep> endTrajectory;
  endTrajectory.push_front(TrajectoryStep(path.getLength(), 0.0));
  double beforeAcceleration
      = getMinMaxPathAcceleration(path.getLength(), 0.0, false);
  integrateBackward(endTrajectory, startTrajectory, beforeAcceleration);

  this->trajectory = startTrajectory;

  // calculate timing
  list<TrajectoryStep>::iterator previous = trajectory.begin();
  list<TrajectoryStep>::iterator it = previous;
  it->mTime = 0.0;
  ++it;
  while (it != trajectory.end())
  {
    it->mTime = previous->mTime
                + (it->mPathPos - previous->mPathPos)
                      / ((it->mPathVel + previous->mPathVel) / 2.0);
    previous = it;
    ++it;
  }

  // debug
  // ofstream file("trajectory.txt");
  // for(list<TrajectoryStep>::iterator it = trajectory.begin(); it !=
  // trajectory.end(); it++) { 	file << it->mPathPos << "  " << it->mPathVel <<
  // endl;
  //}
  // file.close();
}

//==============================================================================
// returns true if end of path is reached.
bool PathFollowingTrajectory::getNextSwitchingPoint(
    double pathPos,
    TrajectoryStep& nextSwitchingPoint,
    double& beforeAcceleration,
    double& afterAcceleration)
{
  TrajectoryStep accelerationSwitchingPoint(pathPos, 0.0);
  double accelerationBeforeAcceleration, accelerationAfterAcceleration;
  bool accelerationReachedEnd;
  do
  {
    accelerationReachedEnd = getNextAccelerationSwitchingPoint(
        accelerationSwitchingPoint.mPathPos,
        accelerationSwitchingPoint,
        accelerationBeforeAcceleration,
        accelerationAfterAcceleration);
    // double test =
    // getVelocityMaxPathVelocity(accelerationSwitchingPoint.pathPos);
  } while (
      !accelerationReachedEnd
      && accelerationSwitchingPoint.mPathVel
             > getVelocityMaxPathVelocity(accelerationSwitchingPoint.mPathPos));

  TrajectoryStep velocitySwitchingPoint(pathPos, 0.0);
  double velocityBeforeAcceleration, velocityAfterAcceleration;
  bool velocityReachedEnd;
  do
  {
    velocityReachedEnd = getNextVelocitySwitchingPoint(
        velocitySwitchingPoint.mPathPos,
        velocitySwitchingPoint,
        velocityBeforeAcceleration,
        velocityAfterAcceleration);
  } while (!velocityReachedEnd
           && velocitySwitchingPoint.mPathPos
                  <= accelerationSwitchingPoint.mPathPos
           && (velocitySwitchingPoint.mPathVel
                   > getAccelerationMaxPathVelocity(
                         velocitySwitchingPoint.mPathPos - eps)
               || velocitySwitchingPoint.mPathVel
                      > getAccelerationMaxPathVelocity(
                            velocitySwitchingPoint.mPathPos + eps)));

  if (accelerationReachedEnd && velocityReachedEnd)
  {
    return true;
  }
  else if (
      !accelerationReachedEnd
      && (velocityReachedEnd
          || accelerationSwitchingPoint.mPathPos
                 <= velocitySwitchingPoint.mPathPos))
  {
    nextSwitchingPoint = accelerationSwitchingPoint;
    beforeAcceleration = accelerationBeforeAcceleration;
    afterAcceleration = accelerationAfterAcceleration;
    return false;
  }
  else
  {
    nextSwitchingPoint = velocitySwitchingPoint;
    beforeAcceleration = velocityBeforeAcceleration;
    afterAcceleration = velocityAfterAcceleration;
    return false;
  }
}

//==============================================================================
bool PathFollowingTrajectory::getNextAccelerationSwitchingPoint(
    double pathPos,
    TrajectoryStep& nextSwitchingPoint,
    double& beforeAcceleration,
    double& afterAcceleration)
{
  double switchingPathPos = pathPos;
  double switchingPathVel;
  while (true)
  {
    bool discontinuity;
    switchingPathPos
        = path.getNextSwitchingPoint(switchingPathPos, discontinuity);

    if (switchingPathPos > path.getLength() - eps)
    {
      return true;
    }

    if (discontinuity)
    {
      const double beforePathVel
          = getAccelerationMaxPathVelocity(switchingPathPos - eps);
      const double afterPathVel
          = getAccelerationMaxPathVelocity(switchingPathPos + eps);
      switchingPathVel = min(beforePathVel, afterPathVel);
      beforeAcceleration = getMinMaxPathAcceleration(
          switchingPathPos - eps, switchingPathVel, false);
      afterAcceleration = getMinMaxPathAcceleration(
          switchingPathPos + eps, switchingPathVel, true);

      if ((beforePathVel > afterPathVel
           || getMinMaxPhaseSlope(
                  switchingPathPos - eps, switchingPathVel, false)
                  > getAccelerationMaxPathVelocityDeriv(
                        switchingPathPos - 2.0 * eps))
          && (beforePathVel < afterPathVel
              || getMinMaxPhaseSlope(
                     switchingPathPos + eps, switchingPathVel, true)
                     < getAccelerationMaxPathVelocityDeriv(
                           switchingPathPos + 2.0 * eps)))
      {
        break;
      }
    }
    else
    {
      switchingPathVel = getAccelerationMaxPathVelocity(switchingPathPos);
      beforeAcceleration = 0.0;
      afterAcceleration = 0.0;

      if (getAccelerationMaxPathVelocityDeriv(switchingPathPos - eps) < 0.0
          && getAccelerationMaxPathVelocityDeriv(switchingPathPos + eps) > 0.0)
      {
        break;
      }
    }
  }

  nextSwitchingPoint = TrajectoryStep(switchingPathPos, switchingPathVel);
  return false;
}

//==============================================================================
bool PathFollowingTrajectory::getNextVelocitySwitchingPoint(
    double pathPos,
    TrajectoryStep& nextSwitchingPoint,
    double& beforeAcceleration,
    double& afterAcceleration)
{
  const double stepSize = 0.001;
  const double accuracy = 0.000001;

  bool start = false;
  pathPos -= stepSize;
  do
  {
    pathPos += stepSize;

    if (getMinMaxPhaseSlope(pathPos, getVelocityMaxPathVelocity(pathPos), false)
        >= getVelocityMaxPathVelocityDeriv(pathPos))
    {
      start = true;
    }
  } while ((!start
            || getMinMaxPhaseSlope(
                   pathPos, getVelocityMaxPathVelocity(pathPos), false)
                   > getVelocityMaxPathVelocityDeriv(pathPos))
           && pathPos < path.getLength());

  if (pathPos >= path.getLength())
  {
    return true; // end of trajectory reached
  }

  double beforePathPos = pathPos - stepSize;
  double afterPathPos = pathPos;
  while (afterPathPos - beforePathPos > accuracy)
  {
    pathPos = (beforePathPos + afterPathPos) / 2.0;
    if (getMinMaxPhaseSlope(pathPos, getVelocityMaxPathVelocity(pathPos), false)
        > getVelocityMaxPathVelocityDeriv(pathPos))
    {
      beforePathPos = pathPos;
    }
    else
    {
      afterPathPos = pathPos;
    }
  }

  beforeAcceleration = getMinMaxPathAcceleration(
      beforePathPos, getVelocityMaxPathVelocity(beforePathPos), false);
  afterAcceleration = getMinMaxPathAcceleration(
      afterPathPos, getVelocityMaxPathVelocity(afterPathPos), true);
  nextSwitchingPoint
      = TrajectoryStep(afterPathPos, getVelocityMaxPathVelocity(afterPathPos));
  return false;
}

//==============================================================================
bool PathFollowingTrajectory::integrateForward(
    list<TrajectoryStep>& trajectory, double acceleration)
{

  double pathPos = trajectory.back().mPathPos;
  double pathVel = trajectory.back().mPathVel;

  list<pair<double, bool> > switchingPoints = path.getSwitchingPoints();
  list<pair<double, bool> >::iterator nextDiscontinuity
      = switchingPoints.begin();

  while (true)
  {
    if (pathPos > 1.304)
    {
      // int test = 48;
    }

    while (
        nextDiscontinuity != switchingPoints.end()
        && (nextDiscontinuity->first <= pathPos || !nextDiscontinuity->second))
    {
      ++nextDiscontinuity;
    }

    double oldPathPos = pathPos;
    double oldPathVel = pathVel;

    pathVel += timeStep * acceleration;
    pathPos += timeStep * 0.5 * (oldPathVel + pathVel);

    if (nextDiscontinuity != switchingPoints.end()
        && pathPos > nextDiscontinuity->first)
    {
      pathVel = oldPathVel
                + (nextDiscontinuity->first + eps - oldPathPos)
                      * (pathVel - oldPathVel) / (pathPos - oldPathPos);
      pathPos = nextDiscontinuity->first + eps;
    }

    // pathVel += timeStep * acceleration;
    // pathPos += timeStep * 0.5 * (oldPathVel + pathVel);

    if (pathPos > path.getLength())
    {
      return true;
    }
    else if (pathVel < 0.0)
    {
      valid = false;
      cout << "error" << endl;
      return true;
    }

    // double test1 = getMinMaxPhaseSlope(oldPathPos,
    // getVelocityMaxPathVelocity(oldPathPos), false); double test2 =
    // getVelocityMaxPathVelocityDeriv(oldPathPos);

    if (pathVel > getVelocityMaxPathVelocity(pathPos)
        && getMinMaxPhaseSlope(
               oldPathPos, getVelocityMaxPathVelocity(oldPathPos), false)
               <= getVelocityMaxPathVelocityDeriv(oldPathPos))
    {
      pathVel = getVelocityMaxPathVelocity(pathPos);
    }

    trajectory.push_back(TrajectoryStep(pathPos, pathVel));
    acceleration = getMinMaxPathAcceleration(pathPos, pathVel, true);

    if (pathVel > getAccelerationMaxPathVelocity(pathPos)
        || pathVel > getVelocityMaxPathVelocity(pathPos))
    {
      // find more accurate intersection with max-velocity curve using bisection
      TrajectoryStep overshoot = trajectory.back();
      trajectory.pop_back();
      double slope = getSlope(trajectory.back(), overshoot);
      double before = trajectory.back().mPathPos;
      double after = overshoot.mPathPos;
      while (after - before > 0.00001)
      {
        const double midpoint = 0.5 * (before + after);
        double midpointPathVel
            = trajectory.back().mPathVel
              + slope * (midpoint - trajectory.back().mPathPos);

        if (midpointPathVel > getVelocityMaxPathVelocity(midpoint)
            && getMinMaxPhaseSlope(
                   before, getVelocityMaxPathVelocity(before), false)
                   <= getVelocityMaxPathVelocityDeriv(before))
        {
          midpointPathVel = getVelocityMaxPathVelocity(midpoint);
        }

        if (midpointPathVel > getAccelerationMaxPathVelocity(midpoint)
            || midpointPathVel > getVelocityMaxPathVelocity(midpoint))
          after = midpoint;
        else
          before = midpoint;
      }
      trajectory.push_back(TrajectoryStep(
          before,
          trajectory.back().mPathVel
              + slope * (before - trajectory.back().mPathPos)));

      if (getAccelerationMaxPathVelocity(after)
          < getVelocityMaxPathVelocity(after))
      {
        if (after > nextDiscontinuity->first)
        {
          return false;
        }
        else if (
            getMinMaxPhaseSlope(
                trajectory.back().mPathPos, trajectory.back().mPathVel, true)
            > getAccelerationMaxPathVelocityDeriv(trajectory.back().mPathPos))
        {
          return false;
        }
      }
      else
      {
        if (getMinMaxPhaseSlope(
                trajectory.back().mPathPos, trajectory.back().mPathVel, false)
            > getVelocityMaxPathVelocityDeriv(trajectory.back().mPathPos))
        {
          return false;
        }
      }
    }
  }
}

//==============================================================================
void PathFollowingTrajectory::integrateBackward(
    list<TrajectoryStep>& trajectory,
    list<TrajectoryStep>& startTrajectory,
    double acceleration)
{
  list<TrajectoryStep>::reverse_iterator before = startTrajectory.rbegin();
  double pathPos = trajectory.front().mPathPos;
  double pathVel = trajectory.front().mPathVel;

  while (true)
  {
    // pathPos -= timeStep * pathVel;
    // pathVel -= timeStep * acceleration;

    double oldPathVel = pathVel;
    pathVel -= timeStep * acceleration;
    pathPos -= timeStep * 0.5 * (oldPathVel + pathVel);

    trajectory.push_front(TrajectoryStep(pathPos, pathVel));
    acceleration = getMinMaxPathAcceleration(pathPos, pathVel, false);

    if (pathVel < 0.0 || pathPos < 0.0)
    {
      valid = false;
      cout << "error " << pathPos << " " << pathVel << endl;
      return;
    }

    while (before != startTrajectory.rend() && before->mPathPos > pathPos)
    {
      ++before;
    }

    bool error = false;

    if (before != startTrajectory.rbegin()
        && pathVel
               >= before->mPathVel
                      + getSlope(before.base()) * (pathPos - before->mPathPos))
    {
      TrajectoryStep overshoot = trajectory.front();
      trajectory.pop_front();
      list<TrajectoryStep>::iterator after = before.base();
      TrajectoryStep intersection = getIntersection(
          startTrajectory, after, overshoot, trajectory.front());
      // cout << "set arrow from " << intersection.pathPos << ", " <<
      // intersection.pathVel - 0.8 << " to " << intersection.pathPos << ", " <<
      // intersection.pathVel - 0.3 << endl;

      if (after != startTrajectory.end())
      {
        startTrajectory.erase(after, startTrajectory.end());
        startTrajectory.push_back(intersection);
      }
      startTrajectory.splice(startTrajectory.end(), trajectory);

      return;
    }
    else if (
        pathVel > getAccelerationMaxPathVelocity(pathPos) + eps
        || pathVel > getVelocityMaxPathVelocity(pathPos) + eps)
    {
      // find more accurate intersection with max-velocity curve using bisection
      TrajectoryStep overshoot = trajectory.front();
      trajectory.pop_front();
      double slope = getSlope(overshoot, trajectory.front());
      double before = overshoot.mPathPos;
      double after = trajectory.front().mPathPos;
      while (after - before > 0.00001)
      {
        const double midpoint = 0.5 * (before + after);
        double midpointPathVel
            = overshoot.mPathVel + slope * (midpoint - overshoot.mPathPos);

        if (midpointPathVel > getAccelerationMaxPathVelocity(midpoint)
            || midpointPathVel > getVelocityMaxPathVelocity(midpoint))
          before = midpoint;
        else
          after = midpoint;
      }
      trajectory.push_front(TrajectoryStep(
          after, overshoot.mPathVel + slope * (after - overshoot.mPathPos)));

      if (getAccelerationMaxPathVelocity(before)
          < getVelocityMaxPathVelocity(before))
      {
        if (trajectory.front().mPathVel
            > getAccelerationMaxPathVelocity(before) + 0.0001)
        {
          error = true;
        }
        else if (
            getMinMaxPhaseSlope(
                trajectory.front().mPathPos, trajectory.front().mPathVel, false)
            < getAccelerationMaxPathVelocityDeriv(trajectory.front().mPathPos))
        {
          error = true;
        }
      }
      else
      {
        if (getMinMaxPhaseSlope(
                trajectory.back().mPathPos, trajectory.back().mPathVel, false)
            < getVelocityMaxPathVelocityDeriv(trajectory.back().mPathPos))
        {
          error = true;
        }
      }
    }

    if (error)
    {
      ofstream file("trajectory.txt");
      for (list<TrajectoryStep>::iterator it = startTrajectory.begin();
           it != startTrajectory.end();
           ++it)
      {
        file << it->mPathPos << "  " << it->mPathVel << endl;
      }
      for (list<TrajectoryStep>::iterator it = trajectory.begin();
           it != trajectory.end();
           ++it)
      {
        file << it->mPathPos << "  " << it->mPathVel << endl;
      }
      file.close();
      cout << "error" << endl;
      valid = false;
      return;
    }
  }
}

//==============================================================================
double PathFollowingTrajectory::getSlope(
    const TrajectoryStep& point1, const TrajectoryStep& point2)
{
  return (point2.mPathVel - point1.mPathVel)
         / (point2.mPathPos - point1.mPathPos);
}

//==============================================================================
double PathFollowingTrajectory::getSlope(
    list<TrajectoryStep>::const_iterator lineEnd)
{
  list<TrajectoryStep>::const_iterator lineStart = lineEnd;
  --lineStart;
  return getSlope(*lineStart, *lineEnd);
}

//==============================================================================
PathFollowingTrajectory::TrajectoryStep
PathFollowingTrajectory::getIntersection(
    const list<TrajectoryStep>& trajectory,
    list<TrajectoryStep>::iterator& it,
    const TrajectoryStep& linePoint1,
    const TrajectoryStep& linePoint2)
{
  const double lineSlope = getSlope(linePoint1, linePoint2);
  it--;

  double factor = 1.0;
  if (it->mPathVel
      > linePoint1.mPathVel + lineSlope * (it->mPathPos - linePoint1.mPathPos))
    factor = -1.0;
  it++;

  while (it != trajectory.end()
         && factor * it->mPathVel
                < factor
                      * (linePoint1.mPathVel
                         + lineSlope * (it->mPathPos - linePoint1.mPathPos)))
  {
    it++;
  }

  if (it == trajectory.end())
  {
    return TrajectoryStep(0.0, 0.0);
  }
  else
  {
    const double trajectorySlope = getSlope(it);
    const double intersectionPathPos
        = (it->mPathVel - linePoint1.mPathVel + lineSlope * linePoint1.mPathPos
           - trajectorySlope * it->mPathPos)
          / (lineSlope - trajectorySlope);
    const double intersectionPathVel
        = linePoint1.mPathVel
          + lineSlope * (intersectionPathPos - linePoint1.mPathPos);
    return TrajectoryStep(intersectionPathPos, intersectionPathVel);
  }
}

//==============================================================================
double PathFollowingTrajectory::getMinMaxPathAcceleration(
    double pathPos, double pathVel, bool max)
{
  VectorXd configDeriv = path.getTangent(pathPos);
  VectorXd configDeriv2 = path.getCurvature(pathPos);
  double factor = max ? 1.0 : -1.0;
  double maxPathAcceleration = numeric_limits<double>::max();
  for (unsigned int i = 0; i < n; i++)
  {
    if (configDeriv[i] != 0.0)
    {
      maxPathAcceleration = min(
          maxPathAcceleration,
          maxAcceleration[i] / std::abs(configDeriv[i])
              - factor * configDeriv2[i] * pathVel * pathVel / configDeriv[i]);
    }
  }
  return factor * maxPathAcceleration;
}

//==============================================================================
double PathFollowingTrajectory::getMinMaxPhaseSlope(
    double pathPos, double pathVel, bool max)
{
  return getMinMaxPathAcceleration(pathPos, pathVel, max) / pathVel;
}

//==============================================================================
double PathFollowingTrajectory::getAccelerationMaxPathVelocity(double pathPos)
{
  double maxPathVelocity = numeric_limits<double>::infinity();
  const VectorXd configDeriv = path.getTangent(pathPos);
  const VectorXd configDeriv2 = path.getCurvature(pathPos);
  for (unsigned int i = 0; i < n; i++)
  {
    if (configDeriv[i] != 0.0)
    {
      for (unsigned int j = i + 1; j < n; j++)
      {
        if (configDeriv[j] != 0.0)
        {
          double A_ij = configDeriv2[i] / configDeriv[i]
                        - configDeriv2[j] / configDeriv[j];
          if (A_ij != 0.0)
          {
            maxPathVelocity = min(
                maxPathVelocity,
                sqrt(
                    (maxAcceleration[i] / std::abs(configDeriv[i])
                     + maxAcceleration[j] / std::abs(configDeriv[j]))
                    / std::abs(A_ij)));
          }
        }
      }
    }
    else if (configDeriv2[i] != 0.0)
    {
      maxPathVelocity = min(
          maxPathVelocity,
          sqrt(maxAcceleration[i] / std::abs(configDeriv2[i])));
    }
  }
  return maxPathVelocity;
}

//==============================================================================
double PathFollowingTrajectory::getVelocityMaxPathVelocity(double pathPos)
{
  const VectorXd tangent = path.getTangent(pathPos);
  double maxPathVelocity = numeric_limits<double>::max();
  for (unsigned int i = 0; i < n; i++)
  {
    maxPathVelocity
        = min(maxPathVelocity, maxVelocity[i] / std::abs(tangent[i]));
  }
  return maxPathVelocity;
}

//==============================================================================
double PathFollowingTrajectory::getAccelerationMaxPathVelocityDeriv(
    double pathPos)
{
  return (getAccelerationMaxPathVelocity(pathPos + eps)
          - getAccelerationMaxPathVelocity(pathPos - eps))
         / (2.0 * eps);
}

//==============================================================================
double PathFollowingTrajectory::getVelocityMaxPathVelocityDeriv(double pathPos)
{
  const VectorXd tangent = path.getTangent(pathPos);
  double maxPathVelocity = numeric_limits<double>::max();
  unsigned int activeConstraint = 0;
  for (unsigned int i = 0; i < n; i++)
  {
    const double thisMaxPathVelocity = maxVelocity[i] / std::abs(tangent[i]);
    if (thisMaxPathVelocity < maxPathVelocity)
    {
      maxPathVelocity = thisMaxPathVelocity;
      activeConstraint = i;
    }
  }
  return -(maxVelocity[activeConstraint]
           * path.getCurvature(pathPos)[activeConstraint])
         / (tangent[activeConstraint] * std::abs(tangent[activeConstraint]));
}

//==============================================================================
bool PathFollowingTrajectory::isValid() const
{
  return valid;
}

//==============================================================================
double PathFollowingTrajectory::getDuration() const
{
  return trajectory.back().mTime;
}

//==============================================================================
list<PathFollowingTrajectory::TrajectoryStep>::const_iterator
PathFollowingTrajectory::getTrajectorySegment(double time) const
{
  if (time >= trajectory.back().mTime)
  {
    list<TrajectoryStep>::const_iterator last = trajectory.end();
    --last;
    return last;
  }
  else
  {
    if (time < cachedTime)
    {
      cachedTrajectorySegment = trajectory.begin();
    }
    while (time >= cachedTrajectorySegment->mTime)
    {
      ++cachedTrajectorySegment;
    }
    cachedTime = time;
    return cachedTrajectorySegment;
  }
}

//==============================================================================
VectorXd PathFollowingTrajectory::getPosition(double time) const
{
  list<TrajectoryStep>::const_iterator it = getTrajectorySegment(time);
  list<TrajectoryStep>::const_iterator previous = it;
  --previous;

  // const double pathPos = previous->mPathPos + (time - previous->mTime) *
  // (previous->mPathVel + it->mPathVel) / 2.0;

  double timeStep = it->mTime - previous->mTime;
  const double acceleration
      = (it->mPathPos - previous->mPathPos - timeStep * previous->mPathVel)
        / (timeStep * timeStep);

  timeStep = time - previous->mTime;
  const double pathPos = previous->mPathPos + timeStep * previous->mPathVel
                         + timeStep * timeStep * acceleration;

  return path.getConfig(pathPos);
}

//==============================================================================
VectorXd PathFollowingTrajectory::getVelocity(double time) const
{
  list<TrajectoryStep>::const_iterator it = getTrajectorySegment(time);
  list<TrajectoryStep>::const_iterator previous = it;
  --previous;

  // const double pathPos = previous->mPathPos + (time - previous->mTime) *
  // (previous->mPathVel + it->mPathVel) / 2.0;

  double timeStep = it->mTime - previous->mTime;
  const double acceleration
      = (it->mPathPos - previous->mPathPos - timeStep * previous->mPathVel)
        / (timeStep * timeStep);

  timeStep = time - previous->mTime;
  const double pathPos = previous->mPathPos + timeStep * previous->mPathVel
                         + timeStep * timeStep * acceleration;
  const double pathVel = previous->mPathVel + timeStep * acceleration;

  return path.getTangent(pathPos) * pathVel;
}

//==============================================================================
double PathFollowingTrajectory::getMaxAccelerationError()
{
  double maxAccelerationError = 0.0;

  for (double time = 0.0; time < getDuration(); time += 0.000001)
  {
    list<TrajectoryStep>::const_iterator it = getTrajectorySegment(time);
    list<TrajectoryStep>::const_iterator previous = it;
    --previous;

    double timeStep = it->mTime - previous->mTime;
    const double pathAcceleration
        = (it->mPathPos - previous->mPathPos - timeStep * previous->mPathVel)
          / (timeStep * timeStep);

    timeStep = time - previous->mTime;
    const double pathPos = previous->mPathPos + timeStep * previous->mPathVel
                           + timeStep * timeStep * pathAcceleration;

    const double pathVel = previous->mPathVel + timeStep * pathAcceleration;

    VectorXd acceleration = path.getTangent(pathPos) * pathAcceleration
                            + path.getCurvature(pathPos) * pathVel * pathVel;

    for (int i = 0; i < acceleration.size(); i++)
    {
      if (std::abs(acceleration[i]) > maxAcceleration[i])
      {
        maxAccelerationError = max(
            maxAccelerationError,
            std::abs(acceleration[i]) / maxAcceleration[i]);
      }
    }
  }

  return maxAccelerationError;
}

} // namespace planning
} // namespace dart
