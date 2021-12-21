#ifndef DART_NEURAL_MULTI_SHOT_HPP_
#define DART_NEURAL_MULTI_SHOT_HPP_

#include <memory>
#include <vector>

#include "dart/include_eigen.hpp"

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class MultiShot : public Problem
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
  void pinForce(int time, Eigen::VectorXs value) override;

  /// This returns the pinned force value at this timestep.
  Eigen::Ref<Eigen::VectorXs> getPinnedForce(int time) override;

  /// Returns the length of the flattened problem stat
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

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  void getConstraintUpperBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const override;

  /// This gets the bounds on the constraint functions (both knot points and any
  /// custom constraints)
  void getConstraintLowerBounds(
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
      PerformanceLog* log = nullptr) const override;

  /// This returns the initial guess for the values of X when running an
  /// optimization
  void getInitialGuess(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> flatDynamic,
      PerformanceLog* log = nullptr) const override;

  /// This computes the values of the constraints
  void computeConstraints(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> constraints,
      PerformanceLog* log = nullptr) override;

  /// This is a single async call for computing constraints in parallel, if
  /// we're using multithreading
  void asyncPartComputeConstraints(
      int index,
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> constraints,
      int cursor,
      PerformanceLog* log = nullptr);

  /// This computes the Jacobian that relates the flat problem to the
  /// constraints. This returns a matrix that's (getConstraintDim(),
  /// getFlatProblemDim()).
  void backpropJacobian(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacDynamic,
      PerformanceLog* log = nullptr) override;

  /// This is a single async call for computing constraints in parallel, if
  /// we're using multithreading
  void asyncPartBackpropJacobian(
      int index,
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacStatic,
      /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacDynamic,
      int rowCursor,
      int colCursor,
      PerformanceLog* log = nullptr);

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void backpropGradientWrt(
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradDynamic,
      PerformanceLog* log = nullptr) override;

  /// This computes the gradient in the flat problem space, taking into accounts
  /// incoming gradients with respect to any of the shot's values.
  void asyncPartBackpropGradientWrt(
      int index,
      std::shared_ptr<simulation::World> world,
      const TrajectoryRollout* gradWrtRollout,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradStatic,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> gradDynamic,
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

  /// This returns the concatenation of (start pos, start vel) for
  /// convenience
  Eigen::VectorXs getStartState() override;

  /// This returns start pos
  Eigen::VectorXs getStartPos() override;

  /// This returns start vel
  Eigen::VectorXs getStartVel() override;

  /// This sets the start pos
  void setStartPos(Eigen::VectorXs startPos) override;

  /// This sets the start vel
  void setStartVel(Eigen::VectorXs startVel) override;

  /// This unrolls the shot, and returns the (pos, vel) state concatenated at
  /// the end of the shot
  Eigen::VectorXs getFinalState(
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
      Eigen::Ref<Eigen::VectorXs> sparseStatic,
      Eigen::Ref<Eigen::VectorXs> sparseDynamic,
      PerformanceLog* log = nullptr) override;

  /// This writes the Jacobian to a sparse vector
  void asyncPartGetSparseJacobian(
      int index,
      std::shared_ptr<simulation::World> world,
      Eigen::Ref<Eigen::VectorXs> sparseStatic,
      Eigen::Ref<Eigen::VectorXs> sparseDynamic,
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