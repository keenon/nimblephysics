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

#ifndef DART_CONSTRAINT_CONSTRAINTSOVER_HPP_
#define DART_CONSTRAINT_CONSTRAINTSOVER_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/collision/CollisionDetector.hpp"
#include "dart/common/Deprecated.hpp"
#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/SmartPointer.hpp"
#include "dart/neural/NeuralUtils.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
class ShapeNodeCollisionObject;
} // namespace dynamics

namespace constraint {

/// ConstraintSolver manages constraints and computes constraint impulses
class ConstraintSolver
{
public:
  using solveCallback = std::function<void(void)>;

  /// Constructor
  ///
  /// \deprecated Deprecated in DART 6.8. Please use other constructors that
  /// doesn't take timespte. Timestep should be set by the owner of this solver
  /// such as dart::simulation::World when the solver added.
  DART_DEPRECATED(6.8)
  explicit ConstraintSolver(s_t timeStep);

  // TODO(JS): Remove timeStep. The timestep can be set by world when a
  // constraint solver is assigned to a world.
  // Deprecate

  /// Default constructor
  ConstraintSolver();

  /// Copy constructor
  // TODO: implement copy constructor since this class contains a pointer to
  // allocated memory.
  ConstraintSolver(const ConstraintSolver& other) = delete;

  /// Destructor
  virtual ~ConstraintSolver() = default;

  /// Add single skeleton
  void addSkeleton(const dynamics::SkeletonPtr& skeleton);

  /// Add mutiple skeletons
  void addSkeletons(const std::vector<dynamics::SkeletonPtr>& skeletons);

  /// Returns all the skeletons added to this ConstraintSolver.
  const std::vector<dynamics::SkeletonPtr>& getSkeletons() const;

  /// Remove single skeleton
  void removeSkeleton(const dynamics::SkeletonPtr& skeleton);

  /// Remove multiple skeletons
  void removeSkeletons(const std::vector<dynamics::SkeletonPtr>& skeletons);

  /// Remove all skeletons in this constraint solver
  void removeAllSkeletons();

  /// Add a constraint
  void addConstraint(const ConstraintBasePtr& constraint);

  /// Remove a constraint
  void removeConstraint(const ConstraintBasePtr& constraint);

  /// Remove all constraints
  void removeAllConstraints();

  /// Returns the number of constraints that was manually added to this
  /// ConstraintSolver.
  std::size_t getNumConstraints() const;

  /// Returns a constraint by index.
  constraint::ConstraintBasePtr getConstraint(std::size_t index);

  /// Returns a constraint by index.
  constraint::ConstConstraintBasePtr getConstraint(std::size_t index) const;

  /// Returns all the constraints added to this ConstraintSolver.
  std::vector<constraint::ConstraintBasePtr> getConstraints();

  /// Returns all the constraints added to this ConstraintSolver.
  std::vector<constraint::ConstConstraintBasePtr> getConstraints() const;

  /// Clears the last collision result
  void clearLastCollisionResult();

  /// Set time step
  virtual void setTimeStep(s_t _timeStep);

  /// Get time step
  s_t getTimeStep() const;

  /// Set collision detector. This function acquires ownership of the
  /// CollisionDetector passed as an argument. This method is deprecated in
  /// favor of the overload that accepts a std::shared_ptr.
  DART_DEPRECATED(6.0)
  void setCollisionDetector(collision::CollisionDetector* collisionDetector);

  /// Set collision detector
  void setCollisionDetector(
      const std::shared_ptr<collision::CollisionDetector>& collisionDetector);

  /// Get collision detector
  collision::CollisionDetectorPtr getCollisionDetector();

  /// Get (const) collision detector
  collision::ConstCollisionDetectorPtr getCollisionDetector() const;

  /// Return collision group of collision objects that are added to this
  /// ConstraintSolver
  collision::CollisionGroupPtr getCollisionGroup();

  /// Return (const) collision group of collision objects that are added to this
  /// ConstraintSolver
  collision::ConstCollisionGroupPtr getCollisionGroup() const;

  /// Returns collision option that is used for collision checkings in this
  /// ConstraintSolver to generate contact constraints.
  collision::CollisionOption& getCollisionOption();

  /// Returns collision option that is used for collision checkings in this
  /// ConstraintSolver to generate contact constraints.
  const collision::CollisionOption& getCollisionOption() const;

  /// Return the last collision checking result
  collision::CollisionResult& getLastCollisionResult();

  /// Return the last collision checking result
  const collision::CollisionResult& getLastCollisionResult() const;

  /// Set LCP solver
  DART_DEPRECATED(6.7)
  void setLCPSolver(std::unique_ptr<LCPSolver> lcpSolver);

  /// Get LCP solver
  DART_DEPRECATED(6.7)
  LCPSolver* getLCPSolver() const;

  /// Solve constraint impulses and apply them to the skeletons
  void solve();

  /// Solve callback function that uses LCP.
  void lcpSolveCallback();

  /// Replace the default solve callback function.
  void replaceSolveCallback(const solveCallback& f);

  /// Update constraints
  void updateConstraints();

  /// Build constrained groupsContact
  void buildConstrainedGroups();

  /// Solve constrained groups
  void solveConstrainedGroups();

  // Solve for constraint impulses to apply to each constraint in group.
  virtual std::vector<s_t*> solveConstrainedGroup(ConstrainedGroup& group) = 0;

  /// Apply constraint impulses to each constraint.
  void applyConstraintImpulses(
      std::vector<ConstraintBasePtr> constraints, std::vector<s_t*> impulses);

  /// Get constrained groups.
  const std::vector<ConstrainedGroup>& getConstrainedGroups() const;

  /// Get number of constrained groups.
  std::size_t getNumConstrainedGroups() const;

  /// Sets this constraint solver using other constraint solver. All the
  /// properties and registered skeletons and constraints will be copied over.
  virtual void setFromOtherConstraintSolver(const ConstraintSolver& other);

  /// Sets the formulation used to compute the gradients.
  void setGradientEnabled(bool enabled);

  bool getGradientEnabled();

  /// True by default. Sets whether or not to apply artifical "penetration
  /// correction" forces to objects that inter-penetrate.
  void setPenetrationCorrectionEnabled(bool enable);

  bool getPenetrationCorrectionEnabled();

  /// We add this value to the diagonal entries of A, ONLY IF our initial LCP
  /// solution fails, to help prevent A from being low-rank. This both increases
  /// the stability of the forward LCP solution, and it also helps prevent cases
  /// where a low-rank A means that the least-squares stabilization of A has
  /// illegal negative force values. This corresponds to slightly softening the
  /// hard contact constraint.
  ///
  /// This needs to be a fairly large value (compared to normal CFM), like 1e-3,
  /// to prevent numerical accuracy issues during the backprop computations. We
  /// don't use this on most timesteps, so a relatively large CFM constant
  /// shouldn't affect simulation accuracy.
  void setFallbackConstraintForceMixingConstant(s_t constant);

  /// This gets the cached LCP solution, which is useful to be able to get/set
  /// because it can effect the forward solutions of physics problems because of
  /// our optimistic LCP-stabilization-to-acceptance approach.
  virtual Eigen::VectorXs getCachedLCPSolution();

  /// This gets the cached LCP solution, which is useful to be able to get/set
  /// because it can effect the forward solutions of physics problems because of
  /// our optimistic LCP-stabilization-to-acceptance approach.
  virtual void setCachedLCPSolution(Eigen::VectorXs X);

  /// Contacts whose penetrationDepth is deeper than this depth will be ignored.
  /// This is a simple solution to avoid extremely nasty situations with
  /// impossibly deep inter-penetration during multiple shooting optimization.
  void setContactClippingDepth(s_t depth);

  /// Contacts whose penetrationDepth is deeper than this depth will be ignored.
  /// This is a simple solution to avoid extremely nasty situations with
  /// impossibly deep inter-penetration during multiple shooting optimization.
  s_t getContactClippingDepth();

protected:
  /// Check if the skeleton is contained in this solver
  bool containSkeleton(const dynamics::ConstSkeletonPtr& skeleton) const;

  /// Add skeleton if the constraint is not contained in this solver
  bool checkAndAddSkeleton(const dynamics::SkeletonPtr& skeleton);

  /// Check if the constraint is contained in this solver
  bool containConstraint(const ConstConstraintBasePtr& constraint) const;

  /// Add constraint if the constraint is not contained in this solver
  bool checkAndAddConstraint(const ConstraintBasePtr& constraint);

  /// Return true if at least one of colliding body is soft body
  bool isSoftContact(const collision::Contact& contact) const;

  using CollisionDetector = collision::CollisionDetector;

  /// Collision detector
  collision::CollisionDetectorPtr mCollisionDetector;

  /// Collision group
  collision::CollisionGroupPtr mCollisionGroup;

  /// Collision detection option
  collision::CollisionOption mCollisionOption;

  /// Last collision checking result
  collision::CollisionResult mCollisionResult;

  /// Time step
  s_t mTimeStep;

  /// Skeleton list
  std::vector<dynamics::SkeletonPtr> mSkeletons;

  /// Contact constraints those are automatically created
  std::vector<ContactConstraintPtr> mContactConstraints;

  /// Soft contact constraints those are automatically created
  std::vector<SoftContactConstraintPtr> mSoftContactConstraints;

  /// Joint limit constraints those are automatically created
  std::vector<JointLimitConstraintPtr> mJointLimitConstraints;

  /// Servo motor constraints those are automatically created
  std::vector<ServoMotorConstraintPtr> mServoMotorConstraints;

  /// Mimic motor constraints those are automatically created
  std::vector<MimicMotorConstraintPtr> mMimicMotorConstraints;

  /// Joint Coulomb friction constraints those are automatically created
  std::vector<JointCoulombFrictionConstraintPtr>
      mJointCoulombFrictionConstraints;

  /// Constraints that manually added
  std::vector<ConstraintBasePtr> mManualConstraints;

  /// Active constraints
  std::vector<ConstraintBasePtr> mActiveConstraints;

  /// Constraint group list
  std::vector<ConstrainedGroup> mConstrainedGroups;

  /// The type of gradients we want to use for backprop
  bool mGradientEnabled;

  /// True if we want to enable artificial penetration correction forces
  bool mPenetrationCorrectionEnabled;

  /// We add this value to the diagonal entries of A, ONLY IF our initial LCP
  /// solution fails, to help prevent A from being low-rank. This both increases
  /// the stability of the forward LCP solution, and it also helps prevent cases
  /// where a low-rank A means that the least-squares stabilization of A has
  /// illegal negative force values. This corresponds to slightly softening the
  /// hard contact constraint.
  ///
  /// This needs to be a fairly large value (compared to normal CFM), like 1e-3,
  /// to prevent numerical accuracy issues during the backprop computations. We
  /// don't use this on most timesteps, so a relatively large CFM constant
  /// shouldn't affect simulation accuracy.
  s_t mFallbackConstraintForceMixingConstant;

  /// Contacts whose penetrationDepth is deeper than this depth will be ignored.
  /// This is a simple solution to avoid extremely nasty situations with
  /// impossibly deep inter-penetration during multiple shooting optimization.
  s_t mContactClippingDepth;

  /// Solve constraints callback function.
  solveCallback mSolveCallback;
};

} // namespace constraint
} // namespace dart

#endif // DART_CONSTRAINT_CONSTRAINTSOVER_HPP_
