#ifndef DART_NEURAL_SINGLE_SHOT_HPP_
#define DART_NEURAL_SINGLE_SHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/MappedBackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class SingleShot : public Problem
{
public:
  SingleShot(
      std::shared_ptr<simulation::World> world,
      LossFn loss,
      int steps,
      bool tuneStartingState = true);

  friend class MultiShot;

  /// Destructor
  virtual ~SingleShot() override;

  /// This prevents a force from changing in optimization, keeping it fixed at a
  /// specified value.
  void pinForce(int time, Eigen::VectorXs value) override;

  /// This returns the pinned force value at this timestep.
  Eigen::Ref<Eigen::VectorXs> getPinnedForce(int time) override;

  /// Returns the length of the flattened problem state
  int getFlatDynamicProblemDim(
      std::shared_ptr<simulation::World> world) const override;

  /// Returns the length of the knot-point constraint vector
  int getConstraintDim() const override;

  /// This copies a shot down into a single flat vector
  void flatten(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This gets the parameters out of a flat vector
  void unflatten(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<const Eigen::VectorXs>& flatStatic,
      const Eigen::Ref<const Eigen::VectorXs>& flatDynamic,
      PerformanceLog* log = nullptr) override;

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
  void backpropJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jac,
      PerformanceLog* log = nullptr);

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
  void backpropJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacDynamic,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradDynamic,
      PerformanceLog* log = nullptr) override;

  /// This returns the snapshots from a fresh unroll
  std::vector<neural::MappedBackpropSnapshotPtr> getSnapshots(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr) override;

  /// This populates the passed in matrices with the values from this trajectory
  void getStates(
      std::shared_ptr<simulation::World> world,
      /* OUT */ TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr,
      bool useKnots = true) override;

  /// This fills our trajectory with the values from the rollout being passed in
  void setStates(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr) override;

  /// This sets the forces in this trajectory from the passed in matrix
  void setControlForcesRaw(
      Eigen::MatrixXs forces, PerformanceLog* log = nullptr) override;

  /// This moves the trajectory forward in time, setting the starting point to
  /// the new given starting point, and shifting the forces over by `steps`,
  /// padding the remainder with 0s
  Eigen::VectorXi advanceSteps(
      std::shared_ptr<simulation::World> world,
      Eigen::VectorXs startPos,
      Eigen::VectorXs startVel,
      int steps) override;

  /// This returns the concatenation of (start pos, start vel) for convenience
  Eigen::VectorXs getStartState() override;

  /// This returns start pos
  Eigen::VectorXs getStartPos() override;

  /// This returns start vel
  Eigen::VectorXs getStartVel() override;

  /// This sets the start pos
  void setStartPos(Eigen::VectorXs startPos) override;

  /// This sets the start vel
  void setStartVel(Eigen::VectorXs startVel) override;

  void resetDirty() override;
  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  Eigen::VectorXs getFinalState(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr) override;

  /// This returns the debugging name of a given DOF
  std::string getFlatDimName(
      std::shared_ptr<simulation::World> world, int dim) override;

  //////////////////////////////////////////////////////////////////////////////
  // For Testing
  //////////////////////////////////////////////////////////////////////////////

  /// This computes finite difference Jacobians analagous to backpropJacobians()
  void finiteDifferenceJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::MatrixXs> jac);

  /// This computes the Jacobians that relate each timestep to the endpoint of
  /// the trajectory. For a timestep at time t, this will relate quantities like
  /// v_t -> p_end, for example.
  TimestepJacobians backpropStartStateJacobians(
      std::shared_ptr<simulation::World> world, bool useFdJacs = false);

  /// This computes finite difference Jacobians analagous to
  /// backpropStartStateJacobians()
  TimestepJacobians finiteDifferenceStartStateJacobians(
      std::shared_ptr<simulation::World> world, s_t EPS);

private:
  Eigen::VectorXs mStartPos;
  Eigen::VectorXs mStartVel;
  Eigen::MatrixXs mForces;
  std::vector<bool> mForcesPinned;
  Eigen::MatrixXs mPinnedForces;

  bool mSnapshotsCacheDirty;
  std::vector<neural::MappedBackpropSnapshotPtr> mSnapshotsCache;
};

} // namespace trajectory

} // namespace dart

#endif