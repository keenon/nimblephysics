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
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/neural/WithRespectToDamping.hpp"
#include "dart/neural/WithRespectToSpring.hpp"
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

namespace neural {
class WithRespectToMass;
class WithRespectToDamping;
class WithRespectToSpring;
class BackpropSnapshot;
} // namespace neural

namespace simulation {

DART_COMMON_DECLARE_SHARED_WEAK(World)

/// class World
class World : public virtual common::Subject,
              public std::enable_shared_from_this<World>
{
public:
  using NameChangedSignal = common::Signal<void(
      const std::string& _oldName, const std::string& _newName)>;

  using constraintEngineFnType = std::function<void(bool)>;

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
  void setGravity(const Eigen::Vector3s& _gravity);

  /// Get gravity
  const Eigen::Vector3s& getGravity() const;

  /// Set time step
  void setTimeStep(s_t _timeStep);

  /// Get time step
  s_t getTimeStep() const;

  //--------------------------------------------------------------------------
  // Structural Properties
  //--------------------------------------------------------------------------

  /// Get the indexed skeleton
  dynamics::SkeletonPtr getSkeleton(std::size_t _index) const;

  /// Get the indexed skeleton
  const dynamics::SkeletonPtr& getSkeletonRef(std::size_t _index) const;

  /// Find a Skeleton by name
  /// \param[in] _name The name of the Skeleton you are looking for.
  /// \return If the skeleton does not exist then return nullptr.
  dynamics::SkeletonPtr getSkeleton(const std::string& _name) const;

  /// Get the index into the total DOFs of the world for this skeleton
  std::size_t getSkeletonDofOffset(
      const dynamics::SkeletonPtr& _skeleton) const;

  /// Get all the bodies attached to all the skeletons in this world
  std::vector<dynamics::BodyNode*> getAllBodyNodes();

  dynamics::BodyNode* getBodyNodeByIndex(size_t index);

  dynamics::Joint* getJointIndex(size_t index);

  /// Get rest position of a particular dof
  s_t getRestPositionIndex(size_t index);

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
  std::size_t getNumDofs() const;

  /// Returns a vector of all the degrees of freedom of all the skeletons in the
  /// world concatenated
  std::vector<dynamics::DegreeOfFreedom*> getDofs();

  /// Returns the size of the getMasses() vector
  std::size_t getMassDims();

  /// Returns the size of the getDampings() vector
  std::size_t getDampingDims();

  std::size_t getSpringDims();

  /// This will prevent mass from being tuned
  void clearTunableMassThisInstance();

  void clearTunableDampingThisInstance();

  void clearTunableSpringThisInstance();

  /// This registers that we'd like to keep track of this BodyNode's mass in a
  /// specified way in differentiation
  void tuneMass(
      dynamics::BodyNode* node,
      neural::WrtMassBodyNodeEntryType type,
      Eigen::VectorXs upperBound,
      Eigen::VectorXs lowerBound);

  void tuneDamping(
      dynamics::Joint* joint,
      neural::WrtDampingJointEntryType type,
      Eigen::VectorXi dofs_index,
      Eigen::VectorXs upperBound,
      Eigen::VectorXs lowerBound);

  void tuneSpring(dynamics::Joint* joint,
      neural::WrtSpringJointEntryType type,
      Eigen::VectorXi dofs_index,
      Eigen::VectorXs upperBound,
      Eigen::VectorXs lowerBound);

  /// Returns the size of the getLinkMasses() vector
  std::size_t getNumBodyNodes();

  /// Gets the position of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXs getPositions();

  /// Gets the velocity of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXs getVelocities();

  /// Gets the acceleration of all the skeletons in the world concatenated
  /// together as a single vector
  Eigen::VectorXs getAccelerations();

  /// Gets the torques of all the skeletons in the world concatenated together
  /// as a single vector
  Eigen::VectorXs getControlForces();

  /// Gets the masses of all the nodes in the world concatenated together as a
  /// single vector
  Eigen::VectorXs getMasses();

  Eigen::VectorXs getDampings();

  Eigen::VectorXs getSprings();

  Eigen::VectorXi getDampingDofsMapping();

  Eigen::VectorXi getSpringDofsMapping();
  //Eigen::VectorXs getLinkMasses();


  size_t getLinkMassesDims();

  // This gives the vector of force upper limits for all the DOFs in this
  // world
  Eigen::VectorXs getControlForceUpperLimits();

  // This gives the vector of force lower limits for all the DOFs in this
  // world
  Eigen::VectorXs getControlForceLowerLimits();

  // This gives the vector of position upper limits for all the DOFs in this
  // world
  Eigen::VectorXs getPositionUpperLimits();

  // This gives the vector of position lower limits for all the DOFs in this
  // world
  Eigen::VectorXs getPositionLowerLimits();

  // This gives the vector of position upper limits for all the DOFs in this
  // world
  Eigen::VectorXs getVelocityUpperLimits();

  // This gives the vector of position lower limits for all the DOFs in this
  // world
  Eigen::VectorXs getVelocityLowerLimits();

  // This gives the vector of mass upper limits for all the registered bodies in
  // this world
  Eigen::VectorXs getMassUpperLimits();

  // This gives the vector of mass lower limits for all the registered bodies in
  // this world
  Eigen::VectorXs getMassLowerLimits();

  // This gives the vector of mass upper limits for all the registered bodies in
  // this world
  Eigen::VectorXs getDampingUpperLimits();

  // This gives the vector of mass lower limits for all the registered bodies in
  // this world
  Eigen::VectorXs getDampingLowerLimits();

  // This gives the vector of spring coeff lower limits for all the registered bodies in
  // this world
  Eigen::VectorXs getSpringLowerLimits();

  // This gives the vector of spring coeff upper limits for all the registered bodies in
  // this world
  Eigen::VectorXs getSpringUpperLimits();

  // This gets all the inertia matrices for all the links in all the skeletons
  // in the world mapped into a flat vector.

  Eigen::VectorXs getLinkMUs();

  s_t getLinkMUIndex(size_t index);

  Eigen::VectorXs getLinkCOMs();

  Eigen::Vector3s getLinkCOMIndex(size_t index);

  // This gets all the inertia moment-of-inertia paremeters for all the links in
  // all the skeletons in this world concatenated together
  Eigen::VectorXs getLinkMOIs();

  Eigen::Vector6s getLinkMOIIndex(size_t index);

  // This gets all links betas which is used for COM SSID

  Eigen::VectorXs getLinkBetas();

  Eigen::Vector3s getLinkBetaIndex(size_t index);

  Eigen::MatrixXs getLinkAkMatrixIndex(size_t index);

  Eigen::VectorXs getLinkDiagIs();

  Eigen::Vector3s getLinkDiagIIndex(size_t index);

  // This returns a vector of all the link masses for all the skeletons in the
  // world concatenated into a flat vector.
  Eigen::VectorXs getLinkMasses();

  s_t getLinkMassIndex(size_t index);

  Eigen::VectorXs getJointDampingCoeffs();

  Eigen::VectorXs getJointDampingCoeffIndex(size_t index);

  Eigen::VectorXs getJointSpringStiffs();

  Eigen::VectorXs getJointSpringStiffIndex(size_t index);

  /// Sets the position of all the skeletons in the world from a single
  /// concatenated state vector
  void setPositions(Eigen::VectorXs position);

  /// Sets the velocities of all the skeletons in the world from a single
  /// concatenated state vector
  void setVelocities(Eigen::VectorXs velocity);

  /// Sets the accelerations of all the skeletons in the world from a single
  /// concatenated state vector
  void setAccelerations(Eigen::VectorXs acceleration);

  /// Sets the forces of all the skeletons in the world from a single
  /// concatenated state vector
  void setControlForces(Eigen::VectorXs torques);

  // Sets the upper limits of all the joints from a single vector
  void setControlForceUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setControlForceLowerLimits(Eigen::VectorXs limits);

  // Sets the upper limits of all the joints from a single vector
  void setPositionUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setPositionLowerLimits(Eigen::VectorXs limits);

  // Sets the upper limits of all the joints from a single vector
  void setVelocityUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setVelocityLowerLimits(Eigen::VectorXs limits);

  // This sets all the masses for all the registered bodies in the world
  void setMasses(Eigen::VectorXs masses);

  void setDampings(Eigen::VectorXs dampings);

  void setSprings(Eigen::VectorXs springs);

  void setJointDampingCoeffs(Eigen::VectorXs damp_coeffs);

  void setJointDampingCoeffIndex(Eigen::VectorXs damp_coeff, size_t index);

  void setJointSpringStiffs(Eigen::VectorXs spring_coeffs);

  void setJointSpringStiffIndex(Eigen::VectorXs spring_stiff, size_t index);

  void setLinkMasses(Eigen::VectorXs masses);

  void setLinkMassIndex(s_t mass, size_t index);

  void setLinkCOMs(Eigen::VectorXs coms);

  void setLinkMOIs(Eigen::VectorXs mois);

  void setLinkMUs(Eigen::VectorXs mus);

  void setLinkMUIndex(s_t mu, size_t index);

  void setLinkBetas(Eigen::VectorXs betas);

  void setLinkDiagIs(Eigen::VectorXs diag_Is);

  void setLinkBetaIndex(Eigen::Vector3s beta, size_t index);

  void setLinkCOMIndex(Eigen::Vector3s com, size_t index);

  void setLinkMOIIndex(Eigen::Vector6s com, size_t index);

  void setLinkDiagIIndex(Eigen::Vector3s diag_I, size_t index);

  /// This gives the C(pos, vel) vector for all the skeletons in the world,
  /// without accounting for the external forces
  Eigen::VectorXs getCoriolisAndGravityForces();

  /// This gives the C(pos, vel) vector for all the skeletons in the world
  Eigen::VectorXs getCoriolisAndGravityAndExternalForces();

  /// This constructs a mass matrix for the whole world, by creating a
  /// block-diagonal concatenation of each skeleton's mass matrix.
  Eigen::MatrixXs getMassMatrix();

  /// This constructs an inverse mass matrix for the whole world, by creating a
  /// block-diagonal concatenation of each skeleton's inverse mass matrix.
  Eigen::MatrixXs getInvMassMatrix();

  void clampPositionsToLimits();
  //--------------------------------------------------------------------------
  // High Level ("Reinforcement Learning style") API
  //
  // This allows high-level algorithms to think of the physics engine in terms
  // of "states" and "actions", which are familiar terms to RL people.
  //
  // It also provides an intelligent API to efficiently construct Jacobians for
  // state_t -> state_{t+1} (`getStateJacobian()`) and action_t -> state_{t+1}
  // (`getActionJacobian()`).
  //--------------------------------------------------------------------------

  // The state is [pos, vel] concatenated, so this return 2*getNumDofs()
  int getStateSize();
  // This takes a single state vector and calls setPositions() and
  // setVelocities() on the head and tail, respectively
  void setState(Eigen::VectorXs state);
  // This return the concatenation of [pos, vel]
  Eigen::VectorXs getState();

  // The action dim is given by the size of the action mapping. This defaults to
  // a 1-1 map onto control forces, but can be configured to be just a subset of
  // the control forces, if there are several DOFs that are uncontrolled.
  int getActionSize();
  // This sets the control forces, using the action mapping to decide how to map
  // the passed in vector to control forces. Unmapped control forces are set to
  // 0.
  void setAction(Eigen::VectorXs action);
  // This reads the control forces and runs them through the action mapping to
  // construct a vector for the currently set action.
  Eigen::VectorXs getAction();

  // This sets the mapping that will be used for the action. Each index of
  // `mapping` is an integer corresponding to an index in the control forces
  // vector
  void setActionSpace(std::vector<int> mapping);
  // This returns the action mapping set by `setActionMapping()`. Each index of
  // the returned mapping is an integer corresponding to an index in the control
  // forces vector.
  std::vector<int> getActionSpace();
  // This is a shorthand method to remove a DOF from the action vector. No-op if
  // the dof is already not in the action vector.
  void removeDofFromActionSpace(int index);
  // This is a shorthand method to add a DOF from the action vector, at the end
  // of the mapping space. No-op if the dof is already in the action vector.
  void addDofToActionSpace(int index);

  // This returns the Jacobian for state_t -> state_{t+1}.
  Eigen::MatrixXs getStateJacobian();

  Eigen::MatrixXs getContactFreeStateJacobian();
  // This returns the Jacobian for action_t -> state_{t+1}.
  Eigen::MatrixXs getActionJacobian();

  // This function map any forces to actions
  Eigen::MatrixXs mapToActionSpace(Eigen::MatrixXs forces);

  Eigen::VectorXs mapToActionSpaceVector(Eigen::VectorXs force);

  // This function map any actions to to full dof control forces
  Eigen::MatrixXs mapToForceSpace(Eigen::MatrixXs actions);

  Eigen::VectorXs mapToForceSpaceVector(Eigen::VectorXs action);
  Eigen::MatrixXs getContactFreeActionJacobian();

  Eigen::MatrixXs finiteDifferenceStateJacobian();
  Eigen::MatrixXs finiteDifferenceActionJacobian();

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

  /// Integrate non-constraint forces.
  void integrateVelocities();

  /// Run the constraint engine which solves for constraint impulses and
  /// integrates velocities given these constraint impulses.
  void runConstraintEngine(bool _resetCommand);

  /// The default constraint engine which runs an LCP.
  void runLcpConstraintEngine(bool _resetCommand);

  /// Replace the default constraint engine with a custom one.
  void replaceConstraintEngineFn(const constraintEngineFnType& engineFn);

  /// Integrate velocities given impulses.
  void integrateVelocitiesFromImpulses(bool _resetCommand = true);

  /// Integrate positions.
  void integratePositions(Eigen::VectorXs initialVelocity);

  /// Set current time
  void setTime(s_t _time);

  /// Get current time
  s_t getTime() const;

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

  //--------------------------------------------------------------------------
  // Gradients
  //--------------------------------------------------------------------------

  /// Get the unconstrained velocities that we found in the last timestep,
  /// before we solved the LCP for constraints
  const Eigen::VectorXs& getLastPreConstraintVelocity() const;

  /// This gets the Jacobian relating how changing our current position will
  /// change our next position after a step. Intuitively, you'd expect this to
  /// just be an identity matrix, and often it is, but if we have any FreeJoints
  /// or BallJoints things get more complicated, because they actually use a
  /// complicated function to integrate to the next position.
  Eigen::MatrixXs getPosPosJacobian() const;

  /// This gets the Jacobian relating how changing our current velocity will
  /// change our next position after a step. Intuitively, you'd expect this to
  /// just be an identity matrix * dt, and often it is, but if we have any
  /// FreeJoints or BallJoints things get more complicated, because they
  /// actually use a complicated function to integrate to the next position.
  Eigen::MatrixXs getVelPosJacobian() const;

  /// True if we want to update p_{t+1} as f(p_t, v_t), rather than the old
  /// f(p_t, v_{t+1}). This makes it much easier to reason about
  /// backpropagation, but it can introduce simulation instability in some
  /// environments. True by default.
  void setParallelVelocityAndPositionUpdates(bool enable);

  bool getParallelVelocityAndPositionUpdates();

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

  s_t getFallbackConstraintForceMixingConstant();

  /// Contacts whose penetrationDepth is deeper than this depth will be ignored.
  /// This is a simple solution to avoid extremely nasty situations with
  /// impossibly deep inter-penetration during multiple shooting optimization.
  void setContactClippingDepth(s_t depth);

  /// Contacts whose penetrationDepth is deeper than this depth will be ignored.
  /// This is a simple solution to avoid extremely nasty situations with
  /// impossibly deep inter-penetration during multiple shooting optimization.
  s_t getContactClippingDepth();

  /// This returns the object that we're using to keep track of which objects in
  /// the world need gradients through which kinds of mass.
  std::shared_ptr<neural::WithRespectToMass> getWrtMass();

  std::shared_ptr<neural::WithRespectToDamping> getWrtDamping();

  std::shared_ptr<neural::WithRespectToSpring> getWrtSpring();

  /// This returns the world state as a JSON blob that we can render
  std::string toJson();

  /// This returns just the positions as a JSON blob that can be rendered if we
  /// already have the original world loaded. Good for real-time viewing.
  std::string positionsToJson();

  /// This returns the colors as a JSON blob that can be rendered if we
  /// already have the original world loaded. Good for real-time viewing.
  std::string colorsToJson();

  /// This gets the cached LCP solution, which is useful to be able to get/set
  /// because it can effect the forward solutions of physics problems because of
  /// our optimistic LCP-stabilization-to-acceptance approach.
  Eigen::VectorXs getCachedLCPSolution();

  /// This gets the cached LCP solution, which is useful to be able to get/set
  /// because it can effect the forward solutions of physics problems because of
  /// our optimistic LCP-stabilization-to-acceptance approach.
  void setCachedLCPSolution(Eigen::VectorXs X);

  /// If this is true, we use finite-differencing to compute all of the
  /// requested Jacobians. This override can be useful to verify if there's a
  /// bug in the analytical Jacobians that's causing learning to not converge.
  void setUseFDOverride(bool override);

  bool getUseFDOverride();

  /// If this is true, we check all Jacobians against their finite-differencing
  /// counterparts at runtime. If they aren't sufficiently close, we immediately
  /// crash the program and print what went wrong and some simple replication
  /// instructions.
  void setSlowDebugResultsAgainstFD(bool slowDebug);

  bool getSlowDebugResultsAgainstFD();

  void DisableWrtMass();

protected:
  /// If this is true, we use finite-differencing to compute all of the
  /// requested Jacobians. This override can be useful to verify if there's a
  /// bug in the analytical Jacobians that's causing learning to not converge.
  bool mUseFDOverride;

  /// If this is true, we check all Jacobians against their finite-differencing
  /// counterparts at runtime. If they aren't sufficiently close, we immediately
  /// crash the program and print what went wrong and some simple replication
  /// instructions.
  bool mSlowDebugResultsAgainstFD;

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
  Eigen::Vector3s mGravity;

  /// Simulation time step
  s_t mTimeStep;

  /// Current simulation time
  s_t mTime;

  /// Current simulation frame number
  int mFrame;

  /// Constraint solver
  std::unique_ptr<constraint::ConstraintSolver> mConstraintSolver;

  ///
  Recording* mRecording;

  /// This holds the unconstrained velocities that we found in the last
  /// timestep, before we solved the LCP for constraints
  Eigen::VectorXs mLastPreConstraintVelocity;

  /// Constraint engine which solves for constraint impulses and integrates
  /// velocities according to the given impulses.
  constraintEngineFnType mConstraintEngineFn;

  /// True if we want to update p_{t+1} as f(p_t, v_t), rather than the old
  /// f(p_t, v_{t+1}). This makes it much easier to reason about
  /// backpropagation, but it can introduce simulation instability in some
  /// environments. True by default.
  bool mParallelVelocityAndPositionUpdates;

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

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  NameChangedSignal mNameChangedSignal;

  //--------------------------------------------------------------------------
  // Gradients
  //--------------------------------------------------------------------------

  std::shared_ptr<neural::WithRespectToMass> mWrtMass;

  std::shared_ptr<neural::WithRespectToDamping> mWrtDamping;

  std::shared_ptr<neural::WithRespectToSpring> mWrtSpring; 

  //--------------------------------------------------------------------------
  // High-level RL-style API
  //--------------------------------------------------------------------------
  std::vector<int> mActionSpace;

  /// This gets a backprop snapshot for the current state, (re)computing if
  /// necessary
  std::shared_ptr<neural::BackpropSnapshot> getCachedBackpropSnapshot();

  std::shared_ptr<neural::BackpropSnapshot> mCachedSnapshotPtr;
  Eigen::VectorXs mCachedSnapshotPos;
  Eigen::VectorXs mCachedSnapshotVel;
  Eigen::VectorXs mCachedSnapshotForce;

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
