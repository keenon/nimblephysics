#ifndef DART_NEURAL_SINGLE_SHOT_HPP_
#define DART_NEURAL_SINGLE_SHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
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

  /// Returns the length of the flattened problem state
  int getFlatProblemDim() const override;

  /// Returns the length of the knot-point constraint vector
  int getConstraintDim() const override;

  /// This copies a shot down into a single flat vector
  void flatten(/* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const override;

  /// This gets the parameters out of a flat vector
  void unflatten(const Eigen::Ref<const Eigen::VectorXd>& flat) override;

  /// This runs the shot out, and writes the positions, velocities, and forces
  void unroll(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces) override;

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const override;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const override;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat) const override;

  /// This computes the Jacobian that relates the flat problem to the end state.
  /// This returns a matrix that's (2 * mNumDofs, getFlatProblemDim()).
  void backpropJacobianOfFinalState(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradient(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<const Eigen::MatrixXd>& gradWrtPoses,
      const Eigen::Ref<const Eigen::MatrixXd>& gradWrtVels,
      const Eigen::Ref<const Eigen::MatrixXd>& gradWrtForces,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> grad) override;

  /// This returns the snapshots from a fresh unroll
  std::vector<neural::BackpropSnapshotPtr> getSnapshots(
      std::shared_ptr<simulation::World> world);

  /// This populates the passed in matrices with the values from this trajectory
  void getStates(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> poses,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> vels,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> forces,
      bool useKnots) override;

  /// This returns the concatenation of (start pos, start vel) for convenience
  Eigen::VectorXd getStartState() override;

  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  Eigen::VectorXd getFinalState(
      std::shared_ptr<simulation::World> world) override;

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
  std::vector<neural::BackpropSnapshotPtr> mSnapshotsCache;
};

} // namespace trajectory

} // namespace dart

#endif