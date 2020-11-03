#ifndef DART_NEURAL_MULTI_SHOT_HPP_
#define DART_NEURAL_MULTI_SHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class MultiShot : public AbstractShot
{
public:
  MultiShot(
      std::shared_ptr<simulation::World> world,
      LossFn loss,
      int steps,
      int shotLength,
      bool tuneStartingState = true);

  /// Destructor
  virtual ~MultiShot() override;

  /// If TRUE, this will use multiple independent threads to compute each
  /// SingleShot's values internally. Currently defaults to FALSE. This should
  /// be considered EXPERIMENTAL! Expect bugs.
  void setParallelOperationsEnabled(bool enabled);

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

  /// This adds a mapping through which the loss function can interpret the
  /// output. We can have multiple loss mappings at the same time, and loss can
  /// use arbitrary combinations of multiple views, as long as it can provide
  /// gradients.
  void addMapping(
      const std::string& key,
      std::shared_ptr<neural::Mapping> mapping) override;

  /// This removes the loss mapping at a particular key
  void removeMapping(const std::string& key) override;

  /// This prevents a force from changing in optimization, keeping it fixed at a
  /// specified value.
  void pinForce(int time, Eigen::VectorXd value) override;

  /// This returns the pinned force value at this timestep.
  Eigen::Ref<Eigen::VectorXd> getPinnedForce(int time) override;

  /// Returns the length of the flattened problem stat
  int getFlatDynamicProblemDim(
      std::shared_ptr<simulation::World> world) const override;

  /// Returns the length of the knot-point constraint vector
  int getConstraintDim() const override;

  /// This copies a shot down into a single flat vector
  void flatten(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This gets the parameters out of a flat vector
  void unflatten(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<const Eigen::VectorXd>& flatStatic,
      const Eigen::Ref<const Eigen::VectorXd>& flatDynamic,
      PerformanceLog* log = nullptr) override;

  /// This gets the fixed upper bounds for a flat vector, used during
  /// optimization
  void getUpperBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This gets the fixed lower bounds for a flat vector, used during
  /// optimization
  void getLowerBounds(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  void getConstraintUpperBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  void getConstraintLowerBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
      PerformanceLog* log = nullptr) const override;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This computes the values of the constraints
  void computeConstraints(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
      PerformanceLog* log = nullptr) override;

  /// This is a single async call for computing constraints in parallel, if
  /// we're using multithreading
  void asyncPartComputeConstraints(
      int index,
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
      int cursor,
      PerformanceLog* log = nullptr);

  /// This computes the Jacobian that relates the flat problem to the
  /// constraints. This returns a matrix that's (getConstraintDim(),
  /// getFlatProblemDim()).
  void backpropJacobian(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacDynamic,
      PerformanceLog* log = nullptr) override;

  /// This is a single async call for computing constraints in parallel, if
  /// we're using multithreading
  void asyncPartBackpropJacobian(
      int index,
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacDynamic,
      int rowCursor,
      int colCursor,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> gradDynamic,
      PerformanceLog* log = nullptr) override;

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void asyncPartBackpropGradientWrt(
      int index,
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> gradDynamic,
      int cursorDims,
      int cursorSteps,
      PerformanceLog* log = nullptr);

  /// This populates the passed in matrices with the values from this trajectory
  void getStates(
      std::shared_ptr<simulation::World> world,
      /* OUT */ TrajectoryRollout* rollout,
      PerformanceLog* log = nullptr,
      bool useKnots = true) override;

  void asyncPartGetStates(
      int index,
      std::shared_ptr<simulation::World> world,
      /* OUT */ TrajectoryRollout* rollout,
      int cursor,
      int steps,
      PerformanceLog* log = nullptr);

  /// This returns the concatenation of (start pos, start vel) for
  /// convenience
  Eigen::VectorXd getStartState() override;

  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  Eigen::VectorXd getFinalState(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr) override;

  /// This returns the debugging name of a given DOF
  std::string getFlatDimName(
      std::shared_ptr<simulation::World> world, int dim) override;

  /// This gets the number of non-zero entries in the Jacobian
  int getNumberNonZeroJacobianDynamic(
      std::shared_ptr<simulation::World> world) override;

  /// This gets the number of non-zero entries in the Jacobian
  int getNumberNonZeroJacobianStatic(
      std::shared_ptr<simulation::World> world) override;

  /// This gets the structure of the non-zero entries in the Jacobian
  void getJacobianSparsityStructureDynamic(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXi> rows,
      Eigen::Ref<Eigen::VectorXi> cols,
      PerformanceLog* log = nullptr) override;

  /// This gets the structure of the non-zero entries in the Jacobian
  void getJacobianSparsityStructureStatic(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXi> rows,
      Eigen::Ref<Eigen::VectorXi> cols,
      PerformanceLog* log = nullptr) override;

  /// This writes the Jacobian to a sparse vector
  void getSparseJacobian(
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXd> sparseStatic,
      Eigen::Ref<Eigen::VectorXd> sparseDynamic,
      PerformanceLog* log = nullptr) override;

  /// This writes the Jacobian to a sparse vector
  void asyncPartGetSparseJacobian(
      int index,
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXd> sparseStatic,
      Eigen::Ref<Eigen::VectorXd> sparseDynamic,
      int cursorStatic,
      int cursorDynamic,
      PerformanceLog* log = nullptr);

  /// This returns the snapshots from a fresh unroll
  std::vector<neural::MappedBackpropSnapshotPtr> getSnapshots(
      std::shared_ptr<simulation::World> world,
      PerformanceLog* log = nullptr) override;

  //////////////////////////////////////////////////////////////////////////////
  // For Testing
  //////////////////////////////////////////////////////////////////////////////

private:
  std::vector<std::shared_ptr<SingleShot>> mShots;
  std::vector<simulation::WorldPtr> mParallelWorlds;
  int mShotLength;
  bool mParallelOperationsEnabled;
};

} // namespace trajectory
} // namespace dart

#endif