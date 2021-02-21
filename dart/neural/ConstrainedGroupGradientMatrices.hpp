#ifndef DART_NEURAL_CONSTRAINT_MATRICES_HPP_
#define DART_NEURAL_CONSTRAINT_MATRICES_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace constraint {
class ConstrainedGroup;
class ConstraintBase;
} // namespace constraint

namespace dynamics {
class Skeleton;
} // namespace dynamics

using namespace performance;

namespace neural {

enum ConstraintMapping
{
  CLAMPING = -1,
  NOT_CLAMPING = -2,
  ILLEGAL = -3,
  IRRELEVANT = -4
};

/// This class pairs with a ConstrainedGroup, to save all the constraint
/// matrices and related info for that ConstrainedGroup, so that we can
/// construct full Jacobian matrices or run backprop later.
class ConstrainedGroupGradientMatrices
{
public:
  ConstrainedGroupGradientMatrices(
      constraint::ConstrainedGroup& group, double timeStep);

  /// This is a constructor for test mocks
  ConstrainedGroupGradientMatrices(
      int numDofs, int numConstraintDim, double timeStep);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// at each constraint. This must be called before constructMatrices(), and
  /// must be called exactly once for each constraint.
  void registerConstraint(
      const std::shared_ptr<constraint::ConstraintBase>& constraint);

  /// This mocks registering a constaint. Useful for testing.
  void mockRegisterConstraint(
      double restitutionCoeff, double penetrationHackVel);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// at each constraint's dimension. It gets called _after_ the system has
  /// already applied a measurement impulse to that constraint dimension, and
  /// measured some velocity changes. This must be called before
  /// constructMatrices(), and must be called exactly once for each constraint's
  /// dimension.
  void measureConstraintImpulse(
      const std::shared_ptr<constraint::ConstraintBase>& constraint,
      std::size_t constraintIndex);

  /// This will attempt to quickly solve an LCP by exploiting locality in the
  /// solution. Assuming we were initialized at the last solution, there's
  /// actually a good chance that we're still in all the same force categories.
  /// This gets called before constructMatrices(), so it assumes no matrices
  /// exist yet. Returns true if successful, false otherwise.
  bool attemptFastSolveLCP(
      simulation::World* world,
      Eigen::VectorXd& mX,
      const Eigen::VectorXd& A,
      const Eigen::VectorXd& mHi,
      const Eigen::VectorXd& mLo,
      const Eigen::VectorXd& mB,
      const Eigen::VectorXd& mFIndex);

  /// This mocks measuring a constraint impulse. Useful for testing.
  void mockMeasureConstraintImpulse(Eigen::VectorXd massedImpulseTest);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// after the LCP has run, with the result from the LCP solver.
  void registerLCPResults(
      Eigen::VectorXd mX,
      Eigen::VectorXd hi,
      Eigen::VectorXd lo,
      Eigen::VectorXi fIndex,
      Eigen::VectorXd b,
      Eigen::VectorXd aColNorms,
      Eigen::MatrixXd A,
      bool deliberatelyIgnoreFriction);

  /// If possible (because A is rank-deficient), this changes mX to be the
  /// least-squares minimal solution. This makes mX unique for a given set of
  /// inputs, rather than leaving the exact solution undefined. This can also be
  /// used to short-circuit an LCP solve before it even needs to start, by using
  /// the previous LCP solution as a "close enough" guess that can then be
  /// cleaned up by this method and made exact. To faccilitate that use case,
  /// this method returns true if it's found a valid solution, whether it
  /// changed anything or not, and false if the solution is invalid.
  bool opportunisticallyStandardizeResults(
      simulation::World* world, Eigen::VectorXd& mX);

  /// This returns true if the proposed mX is consistent with our recorded LCP
  /// construction
  bool isSolutionValid(const Eigen::VectorXd& mX);

  /// This gets called by constructMatrices()
  void deduplicateConstraints();

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// after registerLCPResults(). This can only
  /// be called once, and after this is called you cannot call
  /// measureConstraintImpulse() again!
  void constructMatrices(
      simulation::World* world,
      Eigen::VectorXi overrideClasses = Eigen::VectorXi::Zero(0));

  /// This computes and returns the whole vel-vel jacobian for this group. For
  /// backprop, you don't actually need this matrix, you can compute backprop
  /// directly. This is here if you want access to the full Jacobian for some
  /// reason.
  Eigen::MatrixXd getVelVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getPosVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole force-vel jacobian for this group. For
  /// backprop, you don't actually need this matrix, you can compute backprop
  /// directly. This is here if you want access to the full Jacobian for some
  /// reason.
  Eigen::MatrixXd getForceVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-pos jacobian for this group. For
  /// backprop, you don't actually need this matrix, you can compute backprop
  /// directly. This is here if you want access to the full Jacobian for some
  /// reason.
  Eigen::MatrixXd getPosPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole vel-pos jacobian for this group. For
  /// backprop, you don't actually need this matrix, you can compute backprop
  /// directly. This is here if you want access to the full Jacobian for some
  /// reason.
  Eigen::MatrixXd getVelPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This returns the [dC(pos,vel)/dpos] for the group, a block diagonal
  /// concatenation of the skeleton [dC(pos,vel)/dpos] matrices.
  Eigen::MatrixXd getPosCJacobian(simulation::WorldPtr world);

  /// This returns the [dC(pos,vel)/dvel] for the group, a block diagonal
  /// concatenation of the skeleton [dC(pos,vel)/dvel] matrices.
  Eigen::MatrixXd getVelCJacobian(simulation::WorldPtr world);

  /// This returns the mass matrix for the group, a block diagonal
  /// concatenation of the skeleton mass matrices.
  Eigen::MatrixXd getMassMatrix(simulation::WorldPtr world);

  /// This returns the inverse mass matrix for the group, a block diagonal
  /// concatenation of the skeleton inverse mass matrices.
  Eigen::MatrixXd getInvMassMatrix(simulation::WorldPtr world);

  /// This returns the block diagonal matrix where each skeleton's joints
  /// integration scheme is reflected.
  Eigen::MatrixXd getJointsPosPosJacobian(simulation::WorldPtr world);

  /// This returns the block diagonal matrix where each skeleton's joints
  /// integration scheme is reflected.
  Eigen::MatrixXd getJointsVelPosJacobian(simulation::WorldPtr world);

  /// This computes and returns the component of the pos-pos and pos-vel
  /// jacobians due to bounce approximation. For backprop, you don't actually
  /// need this matrix, you can compute backprop directly. This is here if you
  /// want access to the full Jacobian for some reason.
  Eigen::MatrixXd getBounceApproximationJacobian(PerformanceLog* perfLog);

  /// This computes and returns the whole pos-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of constraint force, holding everyhing constant
  /// except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
  /// wrt
  Eigen::MatrixXd getJacobianOfLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
  Eigen::MatrixXd getJacobianOfLCPOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the subset of the A matrix used by the original LCP for just
  /// the clamping constraints. It relates constraint force to constraint
  /// acceleration. It's a mass matrix, just in a weird frame.
  void computeLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world,
      Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& A_c);

  /// This returns the subset of the b vector used by the original LCP for just
  /// the clamping constraints. It's just the relative velocity at the clamping
  /// contact points.
  void computeLCPOffsetClampingSubset(
      simulation::WorldPtr world,
      Eigen::VectorXd& b,
      const Eigen::MatrixXd& A_c);

  /// This computes and returns an estimate of the constraint impulses for the
  /// clamping constraints. This is based on a linear approximation of the
  /// constraint impulses.
  Eigen::VectorXd estimateClampingConstraintImpulses(
      simulation::WorldPtr world, const Eigen::MatrixXd& A_c);

  /// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
  /// everything constant except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfMinv(
      simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt);

  /// This returns the jacobian of C(pos, inertia, vel), holding everything
  /// constant except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes the Jacobian of A_c*f0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfClampingConstraints(
      simulation::WorldPtr world, Eigen::VectorXd f0);

  /// This computes the Jacobian of A_c^T*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfClampingConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXd v0);

  /// This computes the Jacobian of A_ub*E*f0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfUpperBoundConstraints(
      simulation::WorldPtr world, Eigen::VectorXd f0);

  /// This computes the Jacobian of A_ub^T*E*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfUpperBoundConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXd v0);

  /// This computes the implicit backprop without forming intermediate
  /// Jacobians. It takes a LossGradient with the position and velocity vectors
  /// filled it, though the loss with respect to torque is ignored and can be
  /// null. It returns a LossGradient with all three values filled in, position,
  /// velocity, and torque.
  void backprop(
      simulation::WorldPtr world,
      LossGradient& thisTimestepLoss,
      const LossGradient& nextTimestepLoss,
      bool exploreAlternateStrategies = false);

  /// This zeros out any components of the gradient that would want to push us
  /// out of the box-bounds encoded in the world for pos, vel, or force.
  void clipLossGradientsToBounds(
      simulation::WorldPtr world,
      Eigen::VectorXd& lossWrtPos,
      Eigen::VectorXd& lossWrtVel,
      Eigen::VectorXd& lossWrtForce);

  /// This replaces x with the result of M*x in place, without explicitly
  /// forming M
  Eigen::VectorXd implicitMultiplyByMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXd& x);

  /// This replaces x with the result of Minv*x in place, without explicitly
  /// forming Minv
  Eigen::VectorXd implicitMultiplyByInvMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXd& x);

  const Eigen::MatrixXd& getAllConstraintMatrix() const;

  const Eigen::MatrixXd& getClampingConstraintMatrix() const;

  const Eigen::MatrixXd& getMassedClampingConstraintMatrix() const;

  const Eigen::MatrixXd& getUpperBoundConstraintMatrix() const;

  const Eigen::MatrixXd& getMassedUpperBoundConstraintMatrix() const;

  const Eigen::MatrixXd& getUpperBoundMappingMatrix() const;

  const Eigen::MatrixXd& getBouncingConstraintMatrix() const;

  /// These was the mX() vector used to construct this. Pretty much only here
  /// for testing.
  const Eigen::VectorXd& getContactConstraintImpluses() const;

  /// These was the fIndex() vector used to construct this. Pretty much only
  /// here for testing.
  const Eigen::VectorXi& getContactConstraintMappings() const;

  /// Returns the restitution coefficiennts at each clamping contact point.
  const Eigen::VectorXd& getBounceDiagonals() const;

  /// Returns the contact distances at each clamping contact point.
  const Eigen::VectorXd& getRestitutionDiagonals() const;

  /// Returns the penetration correction hack "bounce" (or 0 if the contact is
  /// not inter-penetrating or is actively bouncing) at each contact point.
  const Eigen::VectorXd& getPenetrationCorrectionVelocities() const;

  /// This is the subset of the A matrix from the original LCP that corresponds
  /// to clamping indices.
  const Eigen::MatrixXd& getClampingAMatrix() const;

  /// Returns the constraint impulses along the clamping constraints
  const Eigen::VectorXd& getClampingConstraintImpulses() const;

  /// Returns the relative velocities along the clamping constraints
  const Eigen::VectorXd& getClampingConstraintRelativeVels() const;

  /// Returns the velocity change caused by the illegal impulses from the LCP
  const Eigen::VectorXd& getVelocityDueToIllegalImpulses() const;

  /// Returns the torques applied pre-step
  const Eigen::VectorXd& getPreStepTorques() const;

  /// Returns the velocity pre-step
  const Eigen::VectorXd& getPreStepVelocity() const;

  /// Returns the velocity pre-LCP
  const Eigen::VectorXd& getPreLCPVelocity() const;

  /// Returns the M^{-1} matrix from pre-step
  const Eigen::MatrixXd& getMinv() const;

  /// Get the coriolis and gravity forces
  const Eigen::VectorXd getCoriolisAndGravityAndExternalForces(
      simulation::WorldPtr world) const;

  /// This is like `getClampingConstraintMatrix()` or
  /// `getUpperBoundConstraintMatrix()`, except that it returns all the columns
  /// instead of just a subset.
  Eigen::MatrixXd getFullConstraintMatrix(simulation::World* world) const;

  std::size_t getNumDOFs() const;

  std::size_t getNumConstraintDim() const;

  const std::vector<std::string>& getSkeletons() const;

  const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
  getDifferentiableConstraints() const;

  const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
  getClampingConstraints() const;

  const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
  getUpperBoundConstraints() const;

  /// Returns true if we were able to standardize our LCP results, false if we
  /// weren't
  bool areResultsStandardized() const;

  /// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
  /// finite differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfMinv(
      simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt);

  /// This computes and returns the jacobian of C(pos, inertia, vel) by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

private:
  std::size_t getWrtDim(simulation::WorldPtr world, WithRespectTo* wrt);

  Eigen::VectorXd getWrt(simulation::WorldPtr world, WithRespectTo* wrt);

  void setWrt(
      simulation::WorldPtr world, WithRespectTo* wrt, Eigen::VectorXd v);

  /// Gets the skeletons associated with this constrained group in vector form
  std::vector<std::shared_ptr<dynamics::Skeleton>> getSkeletons(
      simulation::WorldPtr world);

public:
  /// This is only true after we've called constructMatrices(). It's a useful
  /// flag to ensure we don't call it twice.
  bool mFinalized;

  /// This flag gets set if we needed to ignore the friction indices in order to
  /// solve the LCP. This can happen because boxed LCPs that we use to solve
  /// friction aren't guaranteed to be solvable.
  bool mDeliberatelyIgnoreFriction;

  /// Impulse test matrix for all the constraints (only initialized in debug
  /// mode)
  Eigen::MatrixXd mAllConstraintMatrix;

  /// Impulse test matrix for the clamping constraints
  Eigen::MatrixXd mClampingConstraintMatrix;

  /// Massed impulse test matrix for the clamping constraints
  Eigen::MatrixXd mMassedClampingConstraintMatrix;

  /// Impulse test matrix for the upper bound constraints
  Eigen::MatrixXd mUpperBoundConstraintMatrix;

  /// Massed impulse test matrix for the upper bound constraints
  Eigen::MatrixXd mMassedUpperBoundConstraintMatrix;

  /// Mapping matrix for upper bound constraints
  Eigen::MatrixXd mUpperBoundMappingMatrix;

  /// Impulse test matrix for the bouncing constraints
  Eigen::MatrixXd mBouncingConstraintMatrix;

  /// This is the vector of the coefficients on the diagonal of the bounce
  /// matrix. These are 1+restitutionCoeff[i]
  Eigen::VectorXd mBounceDiagonals;

  /// This is the vector of the coefficients sized for just the bounces.
  Eigen::VectorXd mRestitutionDiagonals;

  /// This is the vector of velocity changes due to any impulses from the LCP
  /// solver that are illegal (out of their legal bounds).
  Eigen::VectorXd mVelocityDueToIllegalImpulses;

  /// This is the vector of constraint impulses for all the clamping
  /// constraints. It's key for computing Jacobians through quantities that
  /// change the mass matrix.
  Eigen::VectorXd mClampingConstraintImpulses;

  /// This is just useful for testing the gradient computations
  Eigen::VectorXd mClampingConstraintRelativeVels;

  /// This is just useful for testing the gradient computations
  Eigen::VectorXi mContactConstraintMappings;

  /// This is just useful for testing the gradient computations
  Eigen::VectorXd mPenetrationCorrectionVelocitiesVec;

  /// This is the subset of the A matrix from the original LCP that corresponds
  /// to clamping indices.
  Eigen::MatrixXd mClampingAMatrix;

  /// This is the inverse mass matrix computed in the constuctor
  Eigen::MatrixXd mMinv;

  /// These are the torques being applied, computed in the constuctor
  Eigen::VectorXd mPreStepTorques;

  /// These are the pre-step velocities, computed in the constuctor
  Eigen::VectorXd mPreStepVelocities;

  /// These are the pre-LCP velocities, computed in the constuctor
  Eigen::VectorXd mPreLCPVelocities;

  /// These are the names of skeletons that are covered by this constraint group
  std::vector<std::string> mSkeletons;

  /// For each index in the original force vector, this either points to an
  /// index in the clamping vector, or it contains -1 to indicate the index was
  /// not clamping.
  std::vector<int> mClampingIndex;

  /// For each index in the original force vector, this either points to an
  /// index in the upper bound vector, or it contains -1 to indicate the index
  /// was not clamping.
  std::vector<int> mUpperBoundIndex;

  /// This is the global timestep length. This is included here because it shows
  /// up as a constant in some of the matrices.
  double mTimeStep;

  /// This is the total DOFs for this ConstrainedGroup
  std::size_t mNumDOFs;

  /// This is the number of total dimensions on all the constraints
  std::size_t mNumConstraintDim;

  /// These are the offsets into the total degrees of freedom for each skeleton
  std::unordered_map<std::string, std::size_t> mSkeletonOffset;

  /// This is all the constraints, in order that they were registered
  std::vector<std::shared_ptr<constraint::ConstraintBase>> mConstraints;

  /// This gives the index into the constraint at mConstraints[i] that
  /// constraint i represents
  std::vector<int> mConstraintIndices;

  /// These are all the constraints
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
      mDifferentiableConstraints;

  /// These are just the clamping constraints
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
      mClampingConstraints;

  /// These are just the upper bound constraints
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
      mUpperBoundConstraints;

  /// This is set to true when we were able to standardize the LCP output. It's
  /// false if we were invalid for some reason.
  bool mStandardizedResults;

  /// These are public to enable unit testing
public:
  /// This holds the coefficient of restitution for each constraint on this
  /// group.
  std::vector<double> mRestitutionCoeffs;

  /// This holds the penetration correction velocities for each constraint in
  /// this group.
  std::vector<double> mPenetrationCorrectionVelocities;

  /// These are all the values from the original LCP
  Eigen::VectorXd mX;
  Eigen::VectorXd mHi;
  Eigen::VectorXd mLo;
  Eigen::VectorXi mFIndex;
  Eigen::VectorXd mB;
  Eigen::VectorXd mAColNorms;
  Eigen::MatrixXd mA;

  /// This holds the outputs of the impulse tests we run to create the
  /// constraint matrices. We shuffle these vectors into the columns of
  /// mClampingConstraintMatrix and mUpperBoundConstraintMatrix depending on the
  /// values of the LCP solution. We also discard many of these vectors.
  ///
  /// mImpulseTests[k] holds the k'th constraint's impulse test, which is
  /// a concatenated vector of the results for each skeleton in the group.
  std::vector<Eigen::VectorXd> mMassedImpulseTests;
};

} // namespace neural
} // namespace dart

#endif