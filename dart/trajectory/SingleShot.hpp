#ifndef DART_NEURAL_SINGLE_SHOT_HPP_
#define DART_NEURAL_SINGLE_SHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/MappedBackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class SingleShot : public AbstractShot
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

  /// This sets the mapping we're using to store the representation of the Shot.
  /// WARNING: THIS IS A POTENTIALLY DESTRUCTIVE OPERATION! This will rewrite
  /// the internal representation of the Shot to use the new mapping, and if the
  /// new mapping is underspecified compared to the old mapping, you may lose
  /// information. It's not guaranteed that you'll get back the same trajectory
  /// if you switch to a different mapping, and then switch back.
  ///
  /// This will affect the values you get back from getStates() - they'll now be
  /// returned in the view given by `mapping`. That's also the represenation
  /// that'll be passed to IPOPT, and updated on each gradient step. Therein
  /// lies the power of changing the representation mapping: There will almost
  /// certainly be mapped spaces that are easier to optimize in than native
  /// joint space, at least initially.
  void switchRepresentationMapping(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* log = nullptr) override;

  /// Returns the length of the flattened problem state
  int getFlatProblemDim() const override;

  /// Returns the length of the knot-point constraint vector
  int getConstraintDim() const override;

  /// This copies a shot down into a single flat vector
  void flatten(
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This gets the parameters out of a flat vector
  void unflatten(
      const Eigen::Ref<const Eigen::VectorXd>& flat,
      PerformanceLog* log = nullptr) override;

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
  void backpropJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> grad,
      PerformanceLog* log = nullptr) override;

  /// This returns the snapshots from a fresh unroll
  std::vector<neural::MappedBackpropSnapshotPtr> getSnapshots(
      std::shared_ptr<simulation::World> world, PerformanceLog* log = nullptr);

  /// This populates the passed in matrices with the values from this trajectory
  void getStates(
      std::shared_ptr<simulation::World> world,
      /* OUT */ TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr,
      bool useKnots = true) override;

  /// This returns the concatenation of (start pos, start vel) for convenience
  Eigen::VectorXd getStartState() override;

  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  Eigen::VectorXd getFinalState(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr) override;

  /// This returns the debugging name of a given DOF
  std::string getFlatDimName(int dim) override;

  //////////////////////////////////////////////////////////////////////////////
  // For Testing
  //////////////////////////////////////////////////////////////////////////////

  /// This computes finite difference Jacobians analagous to backpropJacobians()
  void finiteDifferenceJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::MatrixXd> jac);

  /// This computes the Jacobians that relate each timestep to the endpoint of
  /// the trajectory. For a timestep at time t, this will relate quantities like
  /// v_t -> p_end, for example.
  TimestepJacobians backpropStartStateJacobians(
      std::shared_ptr<simulation::World> world, bool useFdJacs = false);

  /// This computes finite difference Jacobians analagous to
  /// backpropStartStateJacobians()
  TimestepJacobians finiteDifferenceStartStateJacobians(
      std::shared_ptr<simulation::World> world, double EPS);

private:
  Eigen::VectorXd mStartPos;
  Eigen::VectorXd mStartVel;
  Eigen::MatrixXd mForces;

  bool mSnapshotsCacheDirty;
  std::vector<neural::MappedBackpropSnapshotPtr> mSnapshotsCache;
};

} // namespace trajectory

} // namespace dart

#endif