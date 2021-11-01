#ifndef DART_TRAJECTORY_ABSTRACT_SHOT_HPP_
#define DART_TRAJECTORY_ABSTRACT_SHOT_HPP_

#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/MappedBackpropSnapshot.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

namespace dart {

using namespace performance;

namespace simulation {
class World;
}

namespace trajectory {

class Problem
{
public:
  friend class IPOptShotWrapper;
  friend class SGDOptimizer;

  /// Default constructor
  Problem(std::shared_ptr<simulation::World> world, LossFn loss, int steps);

  /// Abstract destructor
  virtual ~Problem();

  /// This prevents a force from changing in optimization, keeping it fixed at a
  /// specified value.
  virtual void pinForce(int time, Eigen::VectorXs value) = 0;

  /// This returns the pinned force value at this timestep.
  virtual Eigen::Ref<Eigen::VectorXs> getPinnedForce(int time) = 0;

  /// This updates the loss function for this trajectory
  void setLoss(LossFn loss);

  /// Add a custom constraint function to the trajectory
  void addConstraint(LossFn loss);

  /// Register constant metadata, which will be passed along to the loss
  /// function, but will not be backpropagated into.
  void setMetadata(std::string key, Eigen::MatrixXs value);

  /// If set to true, our backprop for finding the gradient may use alternative
  /// contact strategies, other than the ones that are technically "correct".
  void setExploreAlternateStrategies(bool flag);

  /// If set to true, our backprop for finding the gradient may use alternative
  /// contact strategies, other than the ones that are technically "correct".
  bool getExploreAlternateStrategies();

  /// This returns the whole map for metadata
  std::unordered_map<std::string, Eigen::MatrixXs>& getMetadataMap();

  /// This adds a mapping through which the loss function can interpret the
  /// output. We can have multiple loss mappings at the same time, and loss can
  /// use arbitrary combinations of multiple views, as long as it can provide
  /// gradients.
  virtual void addMapping(
      const std::string& key, std::shared_ptr<neural::Mapping> mapping);

  /// This returns true if there is a loss mapping at the specified key
  bool hasMapping(const std::string& key);

  /// This returns the loss mapping at the specified key
  std::shared_ptr<neural::Mapping> getMapping(const std::string& key);

  /// This returns a reference to all the mappings in this shot
  std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>&
  getMappings();

  /// This removes the loss mapping at a particular key
  virtual void removeMapping(const std::string& key);

  /// Returns the sum of posDim() + velDim() for the current representation
  /// mapping
  int getRepresentationStateSize() const;

  /// Returns the length of the flattened problem state
  int getFlatProblemDim(std::shared_ptr<simulation::World> world) const;

  /// This returns the dimension of the non-temporal portion of the problem
  /// (masses, global variables, etc)
  virtual int getFlatStaticProblemDim(
      std::shared_ptr<simulation::World> world) const;

  /// This returns the dimension of the temporal part of our problem (torques,
  /// knot-points, etc)
  virtual int getFlatDynamicProblemDim(
      std::shared_ptr<simulation::World> world) const;

  /// Returns the length of the knot-point constraint vector
  virtual int getConstraintDim() const;

  /// This copies a shot down into a single flat vector
  void flatten(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This gets the parameters out of a flat vector
  void unflatten(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<const Eigen::VectorXs>& flat,
      PerformanceLog* log = nullptr);

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  virtual void getConstraintUpperBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  virtual void getConstraintLowerBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  virtual void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const;

  /// This computes the values of the constraints
  virtual void computeConstraints(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> constraints,
      PerformanceLog* log = nullptr);

  /// This is used for reset the problem for SSID

  virtual void resetDirty();

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
  void backpropJacobian(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jac,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, automatically
  /// computing the gradients of the loss function as part of the call
  void backpropGradient(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> grad,
      PerformanceLog* log = nullptr);

  /// Get the loss for the rollout
  s_t getLoss(
      std::shared_ptr<simulation::World> world, PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> grad,
      PerformanceLog* log = nullptr);

  /// This populates the passed in matrices with the values from this trajectory
  virtual void getStates(
      std::shared_ptr<simulation::World> world,
      /* OUT */ TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr,
      bool useKnots = true)
      = 0;

  /// This fills our trajectory with the values from the rollout being passed in
  virtual void setStates(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr)
      = 0;

  /// This sets the forces in this trajectory from the passed in matrix. This
  /// doesn't update the knot points, or change the starting position. If you'd
  /// like to do that, use `updateWithForces()` instead.
  virtual void setControlForcesRaw(
      Eigen::MatrixXs forces, PerformanceLog* log = nullptr)
      = 0;

  /// This moves the trajectory forward in time, setting the starting point to
  /// the new given starting point, and shifting the forces over by `steps`,
  /// padding the remainder with 0s
  virtual Eigen::VectorXi advanceSteps(
      std::shared_ptr<simulation::World> world,
      Eigen::VectorXs startPos,
      Eigen::VectorXs startVel,
      int steps)
      = 0;

  /// This sets the forces in this trajectory from the passed in matrix. This
  /// updates the knot points as part of the update, and returns a mapping of
  /// where indices moved around to, so that we can update lagrange multipliers
  /// etc.
  virtual Eigen::VectorXi updateWithForces(
      std::shared_ptr<simulation::World> world,
      Eigen::MatrixXs forces,
      PerformanceLog* log = nullptr);

  const TrajectoryRollout* getRolloutCache(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr,
      bool useKnots = true);

  TrajectoryRollout* getGradientWrtRolloutCache(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr,
      bool useKnots = true);

  /// This returns the concatenation of (start pos, start vel) for convenience
  virtual Eigen::VectorXs getStartState() = 0;

  /// This returns start pos
  virtual Eigen::VectorXs getStartPos() = 0;

  /// This returns start vel
  virtual Eigen::VectorXs getStartVel() = 0;

  /// This sets the start pos
  virtual void setStartPos(Eigen::VectorXs startPos) = 0;

  /// This sets the start vel
  virtual void setStartVel(Eigen::VectorXs startVel) = 0;

  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  virtual Eigen::VectorXs getFinalState(
      std::shared_ptr<simulation::World> world, PerformanceLog* log = nullptr)
      = 0;

  int getNumSteps();

  /// Returns the dimension of the mass vector
  int getMassDims();

  /// This returns the debugging name of a given DOF
  virtual std::string getFlatDimName(
      std::shared_ptr<simulation::World> world, int dim)
      = 0;

  /// This gets the total number of non-zero entries in the Jacobian
  int getNumberNonZeroJacobian(std::shared_ptr<simulation::World> world);

  /// This gets the number of non-zero entries in the Jacobian
  virtual int getNumberNonZeroJacobianStatic(
      std::shared_ptr<simulation::World> world);

  /// This gets the number of non-zero entries in the Jacobian
  virtual int getNumberNonZeroJacobianDynamic(
      std::shared_ptr<simulation::World> world);

  /// This gets the structure of the non-zero entries in the Jacobian
  virtual void getJacobianSparsityStructure(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXi> rows,
      Eigen::Ref<Eigen::VectorXi> cols,
      PerformanceLog* log = nullptr);

  /// This writes the Jacobian to a sparse vector
  virtual void getSparseJacobian(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXs> sparse,
      PerformanceLog* log = nullptr);

  /// This returns the snapshots from a fresh unroll
  virtual std::vector<neural::MappedBackpropSnapshotPtr> getSnapshots(
      std::shared_ptr<simulation::World> world, PerformanceLog* log = nullptr)
      = 0;

  //////////////////////////////////////////////////////////////////////////////
  // For Testing
  //////////////////////////////////////////////////////////////////////////////

  /// This computes finite difference Jacobians analagous to backpropJacobians()
  void finiteDifferenceJacobian(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::MatrixXs> jac);

  /// This computes finite difference Jacobians analagous to
  /// backpropGradient()
  void finiteDifferenceGradient(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> grad);

  /// This computes the Jacobians that relate each timestep to the endpoint of
  /// the trajectory. For a timestep at time t, this will relate quantities like
  /// v_t -> p_end, for example.
  TimestepJacobians backpropStartStateJacobians(
      std::shared_ptr<simulation::World> world);

  /// This computes finite difference Jacobians analagous to
  /// backpropStartStateJacobians()
  TimestepJacobians finiteDifferenceStartStateJacobians(
      std::shared_ptr<simulation::World> world, s_t EPS);

  //////////////////////////////////////////////////////////////////////////////
  // With a Static/Dynamic distinction
  //////////////////////////////////////////////////////////////////////////////

protected:
  /// This copies a shot down into a single flat vector
  virtual void flatten(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const;

  /// This gets the parameters out of a flat vector
  virtual void unflatten(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<const Eigen::VectorXs>& flatStatic,
      const Eigen::Ref<const Eigen::VectorXs>& flatDynamic,
      PerformanceLog* log = nullptr);

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  virtual void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  virtual void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const;

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
  virtual void backpropJacobian(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacDynamic,
      PerformanceLog* log = nullptr);

  /// This gets called at the beginning of backpropGradientWrt(), as an
  /// opportunity to zero out any static gradient values being managed by
  /// AbstractShot.
  void initializeStaticGradient(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXs> gradStatic,
      PerformanceLog* log = nullptr);

  /// This adds anything to the static gradient that we need to. It needs to be
  /// called for every timestep during backpropGradientWrt().
  void accumulateStaticGradient(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXs> gradStatic,
      neural::LossGradient& thisTimestep,
      PerformanceLog* log = nullptr);

  /// This gets called at the beginning of backpropJacobianOfFinalState() in
  /// SingleShot, as an opportunity to zero out any static jacobian values being
  /// managed by AbstractShot.
  void initializeStaticJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::MatrixXs> jacStatic,
      PerformanceLog* log = nullptr);

  /// This adds anything to the static gradient that we need to. It needs to be
  /// called for every timestep during backpropJacobianOfFinalState() in
  /// SingleShot.
  void accumulateStaticJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::MatrixXs> jacStatic,
      TimestepJacobians& thisTimestep,
      PerformanceLog* log = nullptr);

  /// This returns the initial guess for the values of X when running an
  /// optimization
  virtual void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const;

  /// This gets the structure of the non-zero entries in the Jacobian
  virtual void getJacobianSparsityStructureStatic(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXi> rows,
      Eigen::Ref<Eigen::VectorXi> cols,
      PerformanceLog* log = nullptr);

  /// This gets the structure of the non-zero entries in the Jacobian
  virtual void getJacobianSparsityStructureDynamic(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXi> rows,
      Eigen::Ref<Eigen::VectorXi> cols,
      PerformanceLog* log = nullptr);

  /// This writes the Jacobian to a pair of sparse vectors, separating static
  /// and dynamic regions
  virtual void getSparseJacobian(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXs> sparseStatic,
      Eigen::Ref<Eigen::VectorXs> sparseDynamic,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  virtual void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradDynamic,
      PerformanceLog* log = nullptr)
      = 0;

protected:
  std::shared_ptr<simulation::World> mWorld;
  LossFn mLoss;
  int mSteps;
  bool mTuneStartingState;
  bool mExploreAlternateStrategies;
  std::vector<LossFn> mConstraints;
  std::unordered_map<std::string, std::shared_ptr<neural::Mapping>> mMappings;
  bool mRolloutCacheDirty;
  std::shared_ptr<TrajectoryRolloutReal> mRolloutCache;
  std::shared_ptr<TrajectoryRolloutReal> mGradWrtRolloutCache;
  std::unordered_map<std::string, Eigen::MatrixXs> mMetadata;
};

} // namespace trajectory
} // namespace dart

#endif