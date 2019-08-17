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

/**
 * @file RRT.h
 * @author Tobias Kunz, Can Erdogan
 * @date Jan 31, 2013
 * @brief The generic RRT implementation. It can be inherited for modifications
 * to collision checking, sampling and etc.
 */

#ifndef DART_PLANNING_RRT_HPP_
#define DART_PLANNING_RRT_HPP_

#include <list>
#include <vector>
#include <Eigen/Core>

#include "dart/dynamics/SmartPointer.hpp"
#include "dart/simulation/World.hpp"

namespace flann {
template <class A>
class L2;
template <class A>
class Index;
} // namespace flann

namespace dart {

namespace simulation {
class World;
}
namespace dynamics {
class Skeleton;
}

namespace planning {

/// The rapidly-expanding random tree implementation
class RRT
{
public:
  /// To get byte-aligned Eigen vectors
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// The result of attempting to create a new node to reach a target node
  enum StepResult
  {
    /// Collided with obstacle. No node added.
    STEP_COLLISION,

    /// The target node is closer than step size (so reached). No node added.
    STEP_REACHED,

    /// One node added.
    STEP_PROGRESS
  };

public:
  // Initialization constants and search variables

  /// Number of dof we can manipulate (may be less than robot's)
  const int ndim;

  /// Step size at each node creation
  const double stepSize;

  /// Last added node or the nearest node found after a search
  int activeNode;

  /// The ith node in configVector has parent with index pV[i]
  std::vector<int> parentVector;

  /// All visited configs
  // NOTE We are using pointers for the VectorXd's because flann copies the
  // pointers for the data points and we give it the copies made in the heap
  std::vector<const Eigen::VectorXd*> configVector;

public:
  //// Constructor with a single root
  RRT(dart::simulation::WorldPtr mWorld,
      dart::dynamics::SkeletonPtr mRobot,
      const std::vector<std::size_t>& mDofs,
      const Eigen::VectorXd& root,
      double stepSize = 0.02);

  /// Constructor with multiple roots (so, multiple trees)
  RRT(simulation::WorldPtr mWorld,
      dynamics::SkeletonPtr mRobot,
      const std::vector<std::size_t>& mDofs,
      const std::vector<Eigen::VectorXd>& roots,
      double stepSize = 0.02);

  /// Destructor
  virtual ~RRT() = default;

  /// Reaches for a random node by repeatedly extending nodes from the nearest
  /// neighbor in the tree. Stop if there is a collision.
  bool connect();

  /// Reaches for a target by repeatedly extending nodes from the nearest
  /// neighbor. Stop if collide.
  bool connect(const Eigen::VectorXd& target);

  /// Tries a single step with the given "stepSize" to a random configuration.
  /// Fail if collide.
  StepResult tryStep();

  /// Tries a single step with the given "stepSize" to the given configuration.
  /// Fail if collide.
  StepResult tryStep(const Eigen::VectorXd& qtry);

  /// Tries to extend tree towards provided sample
  virtual StepResult tryStepFromNode(const Eigen::VectorXd& qtry, int NNidx);

  /// Checks if the given new configuration is in collision with an obstacle.
  /// Moreover, it is a an opportunity for child classes to change the new
  /// configuration if there is a need. For instance, task constrained planners
  /// might want to sample around this point and replace it with a better (less
  /// erroroneous due to constraint) node.
  virtual bool newConfig(
      std::list<Eigen::VectorXd>& intermediatePoints,
      Eigen::VectorXd& qnew,
      const Eigen::VectorXd& qnear,
      const Eigen::VectorXd& qtarget);

  /// Returns the distance between the current active node and the given node.
  /// TODO This might mislead the users to thinking returning the distance
  /// between the given target and the nearest neighbor.
  double getGap(const Eigen::VectorXd& target);

  /// Traces the path from some node to the initConfig node - useful in creating
  /// the full path after the goal is reached.
  void tracePath(
      int node, std::list<Eigen::VectorXd>& path, bool reverse = false);

  /// Returns the number of nodes in the tree.
  std::size_t getSize() const;

  /// Implementation-specific function for checking collisions
  virtual bool checkCollisions(const Eigen::VectorXd& c);

  /// Returns a random configuration with the specified node IDs
  virtual Eigen::VectorXd getRandomConfig();

protected:
  /// Returns a random value between the given minimum and maximum value
  double randomInRange(double min, double max);

  /// Returns the nearest neighbor to query point
  virtual int getNearestNeighbor(const Eigen::VectorXd& qsamp);

  /// Adds a new node to the tree
  virtual int addNode(const Eigen::VectorXd& qnew, int parentId);

  /// The world that the robot is in
  simulation::WorldPtr mWorld;

  /// The ID of the robot for which a plan is generated
  dynamics::SkeletonPtr mRobot;

  /// The dofs of the robot the planner can manipulate
  std::vector<std::size_t> mDofs;

  /// The underlying flann data structure for fast nearest neighbor searches
  flann::Index<flann::L2<double> >* mIndex;
};

} // namespace planning
} // namespace dart

#endif // DART_PLANNING_RRT_HPP_
