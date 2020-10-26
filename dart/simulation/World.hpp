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
 *   * This code incorporates portions of Open Dynamics Engine
 *     (Copyright (c) 2001-2004, Russell L. Smith. All rights
 *     reserved.) and portions of FCL (Copyright (c) 2011, Willow
 *     Garage, Inc. All rights reserved.), which were released under
 *     the same BSD license as below
 *
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

#ifndef DART_SIMULATION_WORLD_HPP_
#define DART_SIMULATION_WORLD_HPP_

#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dart/collision/CollisionOption.hpp"
#include "dart/common/NameManager.hpp"
#include "dart/common/SmartPointer.hpp"
#include "dart/common/Subject.hpp"
#include "dart/common/Timer.hpp"
#include "dart/constraint/SmartPointer.hpp"
#include "dart/dynamics/SimpleFrame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/Recording.hpp"
#include "dart/simulation/SmartPointer.hpp"

namespace dart {

namespace integration {
class Integrator;
} // namespace integration

namespace dynamics {
class Skeleton;
class DegreeOfFreedom;
} // namespace dynamics

namespace constraint {
class ConstraintSolver;
} // namespace constraint

namespace collision {
class CollisionResult;
} // namespace collision

namespace simulation {

DART_COMMON_DECLARE_SHARED_WEAK(World)

/// class World
class World : public virtual common::Subject
{
public:
  using NameChangedSignal = common::Signal<void(
      const std::string& _oldName, const std::string& _newName)>;

  /// Creates World as shared_ptr
  template <typename... Args>
  static WorldPtr create(Args&&... args);

  //--------------------------------------------------------------------------
  // Constructor and Destructor
  //--------------------------------------------------------------------------

  /// Creates a World
  static std::shared_ptr<World> create(const std::string& name = "world");

  /// Constructor
  World(const std::string& _name = "world");

  /// Destructor
  virtual ~World();

  /// Create a clone of this World. All Skeletons and SimpleFrames that are held
  /// by this World will be copied over.
  std::shared_ptr<World> clone() const;

  //--------------------------------------------------------------------------
  // Properties
  //--------------------------------------------------------------------------

  /// Set the name of this World
  const std::string& setName(const std::string& _newName);

  /// Get the name of this World
  const std::string& getName() const;

  /// Set gravity
  void setGravity(const Eigen::Vector3d& _gravity);

  /// Get gravity
  const Eigen::Vector3d& getGravity() const;

  /// Set time step
  void setTimeStep(double _timeStep);

  /// Get time step
  double getTimeStep() const;

  //--------------------------------------------------------------------------
  // Structural Properties
  //--------------------------------------------------------------------------

  /// Get the indexed skeleton
  dynamics::SkeletonPtr getSkeleton(std::size_t _index) const;

  /// Find a Skeleton by name
  /// \param[in] _name The name of the Skeleton you are looking for.
  /// \return If the skeleton does not exist then return nullptr.
  dynamics::SkeletonPtr getSkeleton(const std::string& _name) const;

  /// Get the index into the total DOFs of the world for this skeleton
  std::size_t getSkeletonDofOffset(
      const dynamics::SkeletonPtr& _skeleton) const;

  /// Get all the bodies attached to all the skeletons in this world
  std::vector<dynamics::BodyNode*> getAllBodyNodes();

  /// Get the number of skeletons
  std::size_t getNumSkeletons() const;

  /// Add a skeleton to this world
  std::string addSkeleton(const dynamics::SkeletonPtr& _skeleton);

  /// Remove a skeleton from this world
  void removeSkeleton(const dynamics::SkeletonPtr& _skeleton);

  /// Remove all the skeletons in this world, and return a set of shared
  /// pointers to them, in case you want to recycle them
  std::set<dynamics::SkeletonPtr> removeAllSkeletons();

  /// Returns wether this World contains a Skeleton.
  bool hasSkeleton(const dynamics::ConstSkeletonPtr& skeleton) const;

  /// Get the dof index for the indexed skeleton
  int getIndex(int _index) const;

  /// Get the indexed Entity
  dynamics::SimpleFramePtr getSimpleFrame(std::size_t _index) const;

  /// Find an Entity by name
  dynamics::SimpleFramePtr getSimpleFrame(const std::string& _name) const;

  /// Get the number of Entities
  std::size_t getNumSimpleFrames() const;

  /// Add an Entity to this world
  std::string addSimpleFrame(const dynamics::SimpleFramePtr& _frame);

  /// Remove a SimpleFrame from this world
  void removeSimpleFrame(const dynamics::SimpleFramePtr& _frame);

  /// Remove all SimpleFrames in this world, and return a set of shared
  /// pointers to them, in case you want to recycle them
  std::set<dynamics::SimpleFramePtr> removeAllSimpleFrames();

  //--------------------------------------------------------------------------
  // World state
  //--------------------------------------------------------------------------

  /// Returns the sum of all the dofs of all the skeletons in this world
  std::size_t getNumDofs();

  /// Returns a vector of all the degrees of freedom of all the skeletons in the
  /// world concatenated
  std::vector<dynamics::DegreeOfFreedom*> getDofs();

  /// Returns the size of the getLinkCOMs() vector
  std::size_t getLinkCOMDims();

  /// Returns the size of the getLinkMoments() vector
  std::size_t getLinkMOIDims();

  /// Returns the size of the getMasses() vector
  std::size_t getLinkMassesDims();

  /// Returns the size of the getLinkMasses() vector
  std::size_t getNumBodyNodes();

  /// Gets the position of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXd getPositions();

  /// Gets the velocity of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXd getVelocities();

  /// Gets the acceleration of all the skeletons in the world concatenated
  /// together as a single vector
  Eigen::VectorXd getAccelerations();

  /// Gets the torques of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXd getForces();

  // This gives the vector of force upper limits for all the DOFs in this
  // world
  Eigen::VectorXd getForceUpperLimits();

  // This gives the vector of force lower limits for all the DOFs in this
  // world
  Eigen::VectorXd getForceLowerLimits();

  // This gives the vector of position upper limits for all the DOFs in this
  // world
  Eigen::VectorXd getPositionUpperLimits();

  // This gives the vector of position lower limits for all the DOFs in this
  // world
  Eigen::VectorXd getPositionLowerLimits();

  // This gives the vector of position upper limits for all the DOFs in this
  // world
  Eigen::VectorXd getVelocityUpperLimits();

  // This gives the vector of position lower limits for all the DOFs in this
  // world
  Eigen::VectorXd getVelocityLowerLimits();

  // This gets all the inertia matrices for all the links in all the skeletons
  // in the world mapped into a flat vector.
  Eigen::VectorXd getLinkCOMs();

  // This gets all the inertia moment-of-inertia paremeters for all the links in
  // all the skeletons in this world concatenated together
  Eigen::VectorXd getLinkMOIs();

  // This returns a vector of all the link masses for all the skeletons in the
  // world concatenated into a flat vector.
  Eigen::VectorXd getLinkMasses();

  /// Sets the position of all the skeletons in the world from a single
  /// concatenated state vector
  void setPositions(Eigen::VectorXd position);

  /// Sets the velocities of all the skeletons in the world from a single
  /// concatenated state vector
  void setVelocities(Eigen::VectorXd velocity);

  /// Sets the accelerations of all the skeletons in the world from a single
  /// concatenated state vector
  void setAccelerations(Eigen::VectorXd acceleration);

  /// Sets the forces of all the skeletons in the world from a single
  /// concatenated state vector
  void setForces(Eigen::VectorXd torques);

  // Sets the upper limits of all the joints from a single vector
  void setForceUpperLimits(Eigen::VectorXd limits);

  // Sets the lower limits of all the joints from a single vector
  void setForceLowerLimits(Eigen::VectorXd limits);

  // Sets the upper limits of all the joints from a single vector
  void setPositionUpperLimits(Eigen::VectorXd limits);

  // Sets the lower limits of all the joints from a single vector
  void setPositionLowerLimits(Eigen::VectorXd limits);

  // Sets the upper limits of all the joints from a single vector
  void setVelocityUpperLimits(Eigen::VectorXd limits);

  // Sets the lower limits of all the joints from a single vector
  void setVelocityLowerLimits(Eigen::VectorXd limits);

  // This sets all the inertia matrices for all the links in all the skeletons
  // in the world mapped into a flat vector.
  void setLinkCOMs(Eigen::VectorXd coms);

  // This sets all the inertia moment-of-inertia paremeters for all the links in
  // all the skeletons in this world concatenated together
  void setLinkMOIs(Eigen::VectorXd mois);

  // This returns a vector of all the link masses for all the skeletons in the
  // world concatenated into a flat vector.
  void setLinkMasses(Eigen::VectorXd masses);

  /// This gives the C(pos, vel) vector for all the skeletons in the world,
  /// without accounting for the external forces
  Eigen::VectorXd getCoriolisAndGravityForces();

  /// This gives the C(pos, vel) vector for all the skeletons in the world
  Eigen::VectorXd getCoriolisAndGravityAndExternalForces();

  /// This constructs a mass matrix for the whole world, by creating a
  /// block-diagonal concatenation of each skeleton's mass matrix.
  Eigen::MatrixXd getMassMatrix();

  /// This constructs an inverse mass matrix for the whole world, by creating a
  /// block-diagonal concatenation of each skeleton's inverse mass matrix.
  Eigen::MatrixXd getInvMassMatrix();

  //--------------------------------------------------------------------------
  // Collision checking
  //--------------------------------------------------------------------------

  /// Deprecated. Please use checkCollision(~) instead.
  DART_DEPRECATED(6.0)
  bool checkCollision(bool checkAllCollisions);

  /// Perform collision checking with 'option' over all the feasible collision
  /// pairs in this World, and the result will be stored 'result'. If no
  /// argument is passed in then it will return just whether there is collision
  /// or not without the contact information such as contact point, normal, and
  /// penetration depth.
  bool checkCollision(
      const collision::CollisionOption& option
      = collision::CollisionOption(false, 1u, nullptr),
      collision::CollisionResult* result = nullptr);

  /// Return the collision checking result of the last simulation step. If this
  /// world hasn't stepped forward yet, then the result would be empty. Note
  /// that this function does not return the collision checking result of
  /// World::checkCollision().
  const collision::CollisionResult& getLastCollisionResult() const;

  //--------------------------------------------------------------------------
  // Simulation
  //--------------------------------------------------------------------------

  /// Reset the time, frame counter and recorded histories
  void reset();

  /// Calculate the dynamics and integrate the world for one step
  /// \param[in] _resetCommand True if you want to reset to zero the joint
  /// command after simulation step.
  void step(bool _resetCommand = true);

  void integrateVelocities();

  /// Set current time
  void setTime(double _time);

  /// Get current time
  double getTime() const;

  /// Get the number of simulated frames
  ///
  /// TODO(MXG): I think the name of this function is much too similar to
  /// getSimpleFrame()
  int getSimFrames() const;

  //--------------------------------------------------------------------------
  // Constraint
  //--------------------------------------------------------------------------

  /// Sets the constraint solver
  ///
  /// Note that the internal properties of \c solver will be overwritten by this
  /// World.
  void setConstraintSolver(constraint::UniqueConstraintSolverPtr solver);

  /// Get the constraint solver
  constraint::ConstraintSolver* getConstraintSolver();

  /// Get the constraint solver
  const constraint::ConstraintSolver* getConstraintSolver() const;

  /// Bake simulated current state and store it into mRecording
  void bake();

  /// Get recording
  Recording* getRecording();

  /// Get the unconstrained velocities that we found in the last timestep,
  /// before we solved the LCP for constraints
  const Eigen::VectorXd& getLastPreConstraintVelocity() const;

  /// True by default. Sets whether or not to apply artifical "penetration
  /// correction" forces to objects that inter-penetrate.
  void setPenetrationCorrectionEnabled(bool enable);

  bool getPenetrationCorrectionEnabled();

  /// This adds tiny positive values to the diagonal before solving the LCP,
  /// which makes our gradients slightly inaccurate, but does increase stability
  /// of our solutions.
  ///
  /// Defaults to false
  void setConstraintForceMixingEnabled(bool enable);

  bool getConstraintForceMixingEnabled();

protected:
  /// Register when a Skeleton's name is changed
  void handleSkeletonNameChange(
      const dynamics::ConstMetaSkeletonPtr& _skeleton);

  /// Register when a SimpleFrame's name is changed
  void handleSimpleFrameNameChange(const dynamics::Entity* _entity);

  /// Name of this World
  std::string mName;

  /// Skeletons in this world
  std::vector<dynamics::SkeletonPtr> mSkeletons;

  std::map<dynamics::ConstMetaSkeletonPtr, dynamics::SkeletonPtr>
      mMapForSkeletons;

  /// Connections for noticing changes in Skeleton names
  /// TODO(MXG): Consider putting this functionality into NameManager
  std::vector<common::Connection> mNameConnectionsForSkeletons;

  /// NameManager for keeping track of Skeletons
  dart::common::NameManager<dynamics::SkeletonPtr> mNameMgrForSkeletons;

  /// Entities in this world
  std::vector<dynamics::SimpleFramePtr> mSimpleFrames;

  /// Connections for noticing changes in Frame names
  /// TODO(MXG): Consider putting this functionality into NameManager
  std::vector<common::Connection> mNameConnectionsForSimpleFrames;

  /// Map from raw SimpleFrame pointers to their shared_ptrs
  std::map<const dynamics::SimpleFrame*, dynamics::SimpleFramePtr>
      mSimpleFrameToShared;

  /// NameManager for keeping track of Entities
  dart::common::NameManager<dynamics::SimpleFramePtr> mNameMgrForSimpleFrames;

  /// The first indeices of each skeleton's dof in mDofs
  ///
  /// For example, if this world has three skeletons and their dof are
  /// 6, 1 and 2 then the mIndices goes like this: [0 6 7].
  std::vector<int> mIndices;

  /// The total number of degrees of freedom of all the skeletons in this world.
  std::size_t mDofs;

  /// Gravity
  Eigen::Vector3d mGravity;

  /// Simulation time step
  double mTimeStep;

  /// Current simulation time
  double mTime;

  /// Current simulation frame number
  int mFrame;

  /// Constraint solver
  std::unique_ptr<constraint::ConstraintSolver> mConstraintSolver;

  ///
  Recording* mRecording;

  /// This holds the unconstrained velocities that we found in the last
  /// timestep, before we solved the LCP for constraints
  Eigen::VectorXd mLastPreConstraintVelocity;

  /// True if we want to enable artificial penetration correction forces
  bool mPenetrationCorrectionEnabled;

  /// True if we want to enable adding tiny positive values to the diagonal
  /// of the A matrix before solving our LCP.
  bool mConstraintForceMixingEnabled;

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  NameChangedSignal mNameChangedSignal;

public:
  //--------------------------------------------------------------------------
  // Slot registers
  //--------------------------------------------------------------------------
  common::SlotRegister<NameChangedSignal> onNameChanged;
};

} // namespace simulation
} // namespace dart

#include "dart/simulation/detail/World-impl.hpp"

#endif // DART_SIMULATION_WORLD_HPP_
