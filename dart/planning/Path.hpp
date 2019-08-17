/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:a
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

#ifndef DART_PLANNING_PATH_HPP_
#define DART_PLANNING_PATH_HPP_

#include <list>
#include <memory>

#include <Eigen/Core>

namespace dart {
namespace planning {

class PathSegment
{
public:
  /// Constructur
  PathSegment(double mLength = 0.0);

  /// Destructor
  virtual ~PathSegment() = default;

  /// Sets the path segment position
  void setPosition(double position);

  /// Returns the path segment position
  double getPosition() const;

  /// Returns path segment length
  double getLength() const;

  virtual Eigen::VectorXd getConfig(double s) const = 0;
  virtual Eigen::VectorXd getTangent(double s) const = 0;
  virtual Eigen::VectorXd getCurvature(double s) const = 0;
  virtual std::list<double> getSwitchingPoints() const = 0;
  virtual std::shared_ptr<PathSegment> clone() const = 0;

protected:
  double mPosition;
  double mLength;
};

class Path final
{
public:
  /// Constructor
  Path(const std::list<Eigen::VectorXd>& path, double maxDeviation = 0.0);

  /// Copy constructor
  Path(const Path& other);

  /// Destructor
  ~Path() = default;

  /// Returns the path length
  double getLength() const;

  /// Returns the configuration on the path at parameter \c s.
  Eigen::VectorXd getConfig(double s) const;

  /// Returns the tangent on the path at parameter \c s.
  Eigen::VectorXd getTangent(double s) const;

  /// Returns the curvature on the path at parameter \c s.
  Eigen::VectorXd getCurvature(double s) const;

  /// Returns the next switching point.
  ///
  /// \param[in] s The path parameter
  /// \param[out] discontinuity
  double getNextSwitchingPoint(double s, bool& discontinuity) const;

  /// Returns all the switching point.
  std::list<std::pair<double, bool>> getSwitchingPoints() const;

private:
  /// Returns the path segment that contains parameter \c s.
  ///
  /// \param[in,out] s The path parameter. Returns the local parameter in the
  /// result path segment.
  PathSegment* getPathSegment(double& s) const;

  double mLength;
  std::list<std::pair<double, bool>> mSwitchingPoints;
  std::list<std::shared_ptr<PathSegment>> mPathSegments;
};

} // namespace planning
} // namespace dart

#endif // DART_PLANNING_PATH_HPP_
