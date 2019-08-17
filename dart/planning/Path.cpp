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

#include "dart/planning/Path.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Geometry>

#include "dart/common/Console.hpp"
#include "dart/math/Constants.hpp"

namespace dart {
namespace planning {

class LinearPathSegment : public PathSegment
{
public:
  LinearPathSegment(const Eigen::VectorXd& start, const Eigen::VectorXd& end)
    : PathSegment((end - start).norm()), mStart(start), mEnd(end)
  {
    // Do nothing
  }

  Eigen::VectorXd getConfig(double s) const override
  {
    s /= mLength;
    s = std::max(0.0, std::min(1.0, s));
    return (1.0 - s) * mStart + s * mEnd;
  }

  Eigen::VectorXd getTangent(double /* s */) const override
  {
    return (mEnd - mStart) / mLength;
  }

  Eigen::VectorXd getCurvature(double /* s */) const override
  {
    return Eigen::VectorXd::Zero(mStart.size());
  }

  std::list<double> getSwitchingPoints() const override
  {
    return std::list<double>();
  }

  std::shared_ptr<PathSegment> clone() const override
  {
    return std::make_shared<LinearPathSegment>(*this);
  }

private:
  Eigen::VectorXd mStart;
  Eigen::VectorXd mEnd;
};

class CircularPathSegment : public PathSegment
{
public:
  CircularPathSegment(
      const Eigen::VectorXd& start,
      const Eigen::VectorXd& intersection,
      const Eigen::VectorXd& end,
      double maxDeviation)
  {
    if ((intersection - start).norm() < 0.000001
        || (end - intersection).norm() < 0.000001)
    {
      mLength = 0.0;
      mRadius = 1.0;
      mCenter = intersection;
      mX = Eigen::VectorXd::Zero(start.size());
      mY = Eigen::VectorXd::Zero(start.size());
      return;
    }

    const Eigen::VectorXd startDirection = (intersection - start).normalized();
    const Eigen::VectorXd endDirection = (end - intersection).normalized();

    if ((startDirection - endDirection).norm() < 0.000001)
    {
      mLength = 0.0;
      mRadius = 1.0;
      mCenter = intersection;
      mX = Eigen::VectorXd::Zero(start.size());
      mY = Eigen::VectorXd::Zero(start.size());
      return;
    }

    // const double startDistance = (start - intersection).norm();
    // const double endDistance = (end - intersection).norm();

    double distance
        = std::min((start - intersection).norm(), (end - intersection).norm());
    const double angle = acos(startDirection.dot(endDirection));

    distance = std::min(
        distance,
        maxDeviation * sin(0.5 * angle)
            / (1.0 - cos(0.5 * angle))); // enforce max deviation

    mRadius = distance / tan(0.5 * angle);
    mLength = angle * mRadius;

    mCenter = intersection
              + (endDirection - startDirection).normalized() * mRadius
                    / cos(0.5 * angle);
    mX = (intersection - distance * startDirection - mCenter).normalized();
    mY = startDirection;

    // debug
    double dotStart
        = startDirection.dot((intersection - getConfig(0.0)).normalized());
    double dotEnd
        = endDirection.dot((getConfig(mLength) - intersection).normalized());
    if (std::abs(dotStart - 1.0) > 0.0001 || std::abs(dotEnd - 1.0) > 0.0001)
    {
      std::cout << "Error\n";
    }
  }

  Eigen::VectorXd getConfig(double s) const override
  {
    const double angle = s / mRadius;
    return mCenter + mRadius * (mX * cos(angle) + mY * sin(angle));
  }

  Eigen::VectorXd getTangent(double s) const override
  {
    const double angle = s / mRadius;
    return -mX * sin(angle) + mY * cos(angle);
  }

  Eigen::VectorXd getCurvature(double s) const override
  {
    const double angle = s / mRadius;
    return -1.0 / mRadius * (mX * cos(angle) + mY * sin(angle));
  }

  std::list<double> getSwitchingPoints() const override
  {
    const double pi = math::constantsd::pi();
    std::list<double> switchingPoints;
    const double dim = mX.size();
    for (unsigned int i = 0; i < dim; i++)
    {
      double switchingAngle = atan2(mY[i], mX[i]);
      if (switchingAngle < 0.0)
      {
        switchingAngle += pi;
      }
      const double switchingPoint = switchingAngle * mRadius;
      if (switchingPoint < mLength)
      {
        switchingPoints.push_back(switchingPoint);
      }
    }
    switchingPoints.sort();
    return switchingPoints;
  }

  std::shared_ptr<PathSegment> clone() const override
  {
    return std::make_shared<CircularPathSegment>(*this);
  }

private:
  double mRadius;
  Eigen::VectorXd mCenter;
  Eigen::VectorXd mX;
  Eigen::VectorXd mY;
};

//==============================================================================
PathSegment::PathSegment(double length) : mPosition(0.0), mLength(length)
{
  // Do nothing
}

//==============================================================================
void PathSegment::setPosition(double position)
{
  mPosition = position;
}

//==============================================================================
double PathSegment::getPosition() const
{
  return mPosition;
}

//==============================================================================
double PathSegment::getLength() const
{
  return mLength;
}

//==============================================================================
bool validatePath(
    const Eigen::VectorXd& startConfig,
    const Eigen::VectorXd& endConfig,
    const Eigen::VectorXd& config1,
    const Eigen::VectorXd& config2,
    const Eigen::VectorXd& config3)
{
  const double tol = 1.0e-6;

  const auto distE1 = (endConfig - config1).norm();
  const auto dist2E = (config2 - endConfig).norm();
  const auto vecE1Norm = (endConfig - config1).normalized();
  const auto vec2ENorm = (config2 - endConfig).normalized();
  if (distE1 > tol && dist2E > tol
      && std::abs(vecE1Norm.dot(vec2ENorm) - 1.0) > tol)
  {
    return false;
  }

  const auto distS2 = (startConfig - config2).norm();
  const auto dist3S = (config2 - startConfig).norm();
  const auto vecS2Norm = (startConfig - config2).normalized();
  const auto vec3SNorm = (config3 - startConfig).normalized();
  if (distS2 > tol && dist3S > tol
      && std::abs(vecS2Norm.dot(vec3SNorm) - 1.0) > tol)
  {
    return false;
  }

  return true;
}

//==============================================================================
Path::Path(const std::list<Eigen::VectorXd>& path, double maxDeviation)
  : mLength(0.0)
{
  if (path.size() < 2)
    return;

  auto config1 = path.cbegin();
  auto config2 = config1;
  ++config2;
  std::list<Eigen::VectorXd>::const_iterator config3;
  Eigen::VectorXd startConfig = *config1;
  while (config2 != path.end())
  {
    config3 = config2;
    ++config3;
    if (maxDeviation > 0.0 && config3 != path.cend())
    {
      auto blendSegment = std::make_shared<CircularPathSegment>(
          0.5 * (*config1 + *config2),
          *config2,
          0.5 * (*config2 + *config3),
          maxDeviation);
      Eigen::VectorXd endConfig = blendSegment->getConfig(0.0);
      if ((endConfig - startConfig).norm() > 0.000001)
      {
        mPathSegments.push_back(
            std::make_shared<LinearPathSegment>(startConfig, endConfig));
      }
      mPathSegments.push_back(blendSegment);

      startConfig = blendSegment->getConfig(blendSegment->getLength());

      if (!validatePath(startConfig, endConfig, *config1, *config2, *config3))
      {
        dterr << "[Path] The input path is not valid.\n";
      }
    }
    else
    {
      mPathSegments.push_back(
          std::make_shared<LinearPathSegment>(startConfig, *config2));
      startConfig = *config2;
    }
    config1 = config2;
    ++config2;
  }

  // create list of switching point candidates, calculate total path length and
  // absolute positions of path segments
  for (auto& segment : mPathSegments)
  {
    segment->setPosition(mLength);
    std::list<double> localSwitchingPoints = segment->getSwitchingPoints();
    for (const auto& point : localSwitchingPoints)
    {
      mSwitchingPoints.push_back(std::make_pair(mLength + point, false));
    }
    mLength += segment->getLength();
    mSwitchingPoints.push_back(std::make_pair(mLength, true));
  }
  mSwitchingPoints.pop_back();
}

//==============================================================================
Path::Path(const Path& other)
  : mLength(other.mLength), mSwitchingPoints(other.mSwitchingPoints)
{
  for (const auto& pathSegment : mPathSegments)
    mPathSegments.push_back(pathSegment->clone());
}

//==============================================================================
double Path::getLength() const
{
  return mLength;
}

//==============================================================================
PathSegment* Path::getPathSegment(double& s) const
{
  auto it = mPathSegments.cbegin();
  auto next = it;
  ++next;
  while (next != mPathSegments.cend() && s >= (*next)->getPosition())
  {
    it = next;
    ++next;
  }
  s -= (*it)->getPosition();
  return it->get();
}

//==============================================================================
Eigen::VectorXd Path::getConfig(double s) const
{
  const PathSegment* pathSegment = getPathSegment(s);
  return pathSegment->getConfig(s);
}

//==============================================================================
Eigen::VectorXd Path::getTangent(double s) const
{
  const PathSegment* pathSegment = getPathSegment(s);
  return pathSegment->getTangent(s);
}

//==============================================================================
Eigen::VectorXd Path::getCurvature(double s) const
{
  const PathSegment* pathSegment = getPathSegment(s);
  return pathSegment->getCurvature(s);
}

//==============================================================================
double Path::getNextSwitchingPoint(double s, bool& discontinuity) const
{
  auto it = mSwitchingPoints.cbegin();
  while (it != mSwitchingPoints.cend() && it->first <= s)
  {
    ++it;
  }

  if (it == mSwitchingPoints.cend())
  {
    discontinuity = true;
    return mLength;
  }
  else
  {
    discontinuity = it->second;
    return it->first;
  }
}

//==============================================================================
std::list<std::pair<double, bool>> Path::getSwitchingPoints() const
{
  return mSwitchingPoints;
}

} // namespace planning
} // namespace dart
