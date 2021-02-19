#ifndef DART_NEURAL_SNAPSHOT_HPP_
#define DART_NEURAL_SNAPSHOT_HPP_

#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

using namespace performance;

namespace neural {

class BackpropSnapshot
{
  friend class MappedBackpropSnapshot;

public:
  /// This saves a snapshot from a forward pass, with all the info we need in
  /// order to efficiently compute a backwards pass. Crucially, the positions
  /// must all be snapshots from before the timestep, yet this constructor must
  /// be called after the timestep.
  BackpropSnapshot(
      simulation::WorldPtr world,
      Eigen::VectorXd preStepPosition,
      Eigen::VectorXd preStepVelocity,
      Eigen::VectorXd preStepTorques,
      Eigen::VectorXd preConstraintVelocities,
      Eigen::VectorXd preStepLCPCache);

  /// This computes the implicit backprop without forming intermediate
  /// Jacobians. It takes a LossGradient with the position and velocity vectors
  /// filled it, though the loss with respect to torque is ignored and can be
  /// null. It returns a LossGradient with all three values filled in, position,
  /// velocity, and torque.
  void backprop(
      simulation::WorldPtr world,
      LossGradient& thisTimestepLoss,
      const LossGradient& nextTimestepLoss,
      PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole vel-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getVelVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getPosVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole force-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getForceVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole mass-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getMassVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getPosPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole vel-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getVelPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the component of the pos-pos and pos-vel
  /// jacobians due to bounce approximation. For backprop, you don't actually
  /// need this matrix, you can compute backprop directly. This is here if you
  /// want access to the full Jacobian for some reason.
  const Eigen::MatrixXd& getBounceApproximationJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  Eigen::VectorXd getPreStepPosition();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  Eigen::VectorXd getPreStepVelocity();

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, BEFORE the timestep.
  Eigen::VectorXd getPreStepTorques();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER integrating forward dynamics but BEFORE
  /// running the LCP.
  Eigen::VectorXd getPreConstraintVelocity();

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  Eigen::VectorXd getPostStepPosition();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  Eigen::VectorXd getPostStepVelocity();

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, AFTER the timestep.
  Eigen::VectorXd getPostStepTorques();

  /////////////////////////////////////////////////////////////////////////////
  /// Just public for testing
  /////////////////////////////////////////////////////////////////////////////

  /// This returns the A_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getClampingConstraintMatrix(simulation::WorldPtr world);

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getMassedClampingConstraintMatrix(simulation::WorldPtr world);

  /// This returns the A_ub matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getUpperBoundConstraintMatrix(simulation::WorldPtr world);

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getMassedUpperBoundConstraintMatrix(
      simulation::WorldPtr world);

  /// This returns the E matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getUpperBoundMappingMatrix();

  /// This returns the B matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getBouncingConstraintMatrix(simulation::WorldPtr world);

  /// This returns the mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton mass matrices.
  Eigen::MatrixXd getMassMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This returns the inverse mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton inverse mass matrices.
  Eigen::MatrixXd getInvMassMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This is the subset of the A matrix from the original LCP that corresponds
  /// to clamping indices.
  Eigen::MatrixXd getClampingAMatrix();

  /// This returns the pos-C(pos,vel) Jacobian for the whole world, a block
  /// diagonal concatenation of the skeleton pos-C(pos,vel) Jacobians.
  Eigen::MatrixXd getPosCJacobian(simulation::WorldPtr world);

  /// This returns the vel-C(pos,vel) Jacobian for the whole world, a block
  /// diagonal concatenation of the skeleton vel-C(pos,vel) Jacobians.
  Eigen::MatrixXd getVelCJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceVelVelJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole pos-C(pos,vel) jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferencePosVelJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceForceVelJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole mass-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceMassVelJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferencePosPosJacobian(
      simulation::WorldPtr world, std::size_t subdivisions = 20);

  /// This computes and returns the whole vel-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceVelPosJacobian(
      simulation::WorldPtr world, std::size_t subdivisions = 20);

  /// This computes and returns the whole wrt-vel jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the whole wrt-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferencePosJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the P_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getProjectionIntoClampsMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This replaces x with the result of M*x in place, without explicitly
  /// forming M
  Eigen::VectorXd implicitMultiplyByMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXd& x);

  /// This replaces x with the result of Minv*x in place, without explicitly
  /// forming Minv
  Eigen::VectorXd implicitMultiplyByInvMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXd& x);

  /// TODO(keenon): Remove me
  Eigen::MatrixXd getScratchAnalytical(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// TODO(keenon): Remove me
  Eigen::MatrixXd getScratchFiniteDifference(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This predicts what the next velocity will be using our linear algebra
  /// formula. This is only here for testing, to compare it against the actual
  /// result of a timestep.
  ///
  /// The `morePreciseButSlower` flag tells this function to do brute force
  /// steps to get constraint matrices A_c and A_ub, rather than use 1st order
  /// approximations. This is important because when we're doing
  /// finite-differencing over tiny EPS (1e-9) then tiny errors in the 1st order
  /// approximations blow up to become huge errors in gradients.
  Eigen::VectorXd getAnalyticalNextV(
      simulation::WorldPtr world, bool morePreciseButSlower = false);

  /// This computes and returns the whole wrt-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the whole wrt-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getPosJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of constraint force, holding everyhing constant
  /// except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
  /// wrt
  Eigen::MatrixXd getJacobianOfLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt);

  /// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
  /// wrt, by finite differencing
  Eigen::MatrixXd finiteDifferenceJacobianOfLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
  Eigen::MatrixXd getJacobianOfLCPOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
  /// finite differencing
  Eigen::MatrixXd finiteDifferenceJacobianOfLCPEstimatedOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
  /// finite differencing
  Eigen::MatrixXd finiteDifferenceJacobianOfLCPOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the subset of the A matrix used by the original LCP for just
  /// the clamping constraints. It relates constraint force to constraint
  /// acceleration. It's a mass matrix, just in a weird frame.
  void computeLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world,
      Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& A_c,
      const Eigen::MatrixXd& A_ub,
      const Eigen::MatrixXd& E);

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
      simulation::WorldPtr world,
      const Eigen::MatrixXd& A_c,
      const Eigen::MatrixXd& A_ub,
      const Eigen::MatrixXd& E);

  /// This returns the jacobian of P_c * v, holding everyhing constant except
  /// the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfProjectionIntoClampsMatrix(
      simulation::WorldPtr world, Eigen::VectorXd v, WithRespectTo* wrt);

  /// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
  /// everything constant except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfMinv(
      simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt);

  /// This returns the jacobian of C(pos, inertia, vel), holding everything
  /// constant except the value of WithRespectTo
  Eigen::MatrixXd getJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of M^{-1}(pos, inertia) * (C(pos, inertia, vel)
  /// + mPreStepTorques), holding everything constant except the value of
  /// WithRespectTo
  Eigen::MatrixXd getJacobianOfMinvC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns a fast approximation to A_c in the neighborhood of the
  /// original
  Eigen::MatrixXd estimateClampingConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// This returns a fast approximation to A_ub in the neighborhood of the
  /// original
  Eigen::MatrixXd estimateUpperBoundConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// Only for testing: VERY SLOW. This returns the actual value of A_c at the
  /// desired position.
  Eigen::MatrixXd getClampingConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// Only for testing: VERY SLOW. This returns the actual value of A_ub at the
  /// desired position.
  Eigen::MatrixXd getUpperBoundConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// Only for testing: VERY SLOW. This returns the actual value of E at the
  /// desired position.
  Eigen::MatrixXd getUpperBoundMappingMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// Only for testing: VERY SLOW. This returns the actual value of the bounce
  /// diagonals at the desired position.
  Eigen::VectorXd getBounceDiagonalsAt(
      simulation::WorldPtr world, Eigen::VectorXd pos);

  /// This computes the Jacobian of A_c*f0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfClampingConstraints(
      simulation::WorldPtr world, Eigen::VectorXd f0);

  /// This computes the Jacobian of A_c^T*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfClampingConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXd v0);

  /// This computes the Jacobian of A_ub*(E*f0) with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfUpperBoundConstraints(
      simulation::WorldPtr world, Eigen::VectorXd E_f0);

  /// This computes the Jacobian of A_ub^T*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXd getJacobianOfUpperBoundConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXd v0);

  /// This computes the finite difference Jacobian of A_c*f0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXd finiteDifferenceJacobianOfClampingConstraints(
      simulation::WorldPtr world, Eigen::VectorXd f0);

  /// This computes the finite difference Jacobian of A_c^T*v0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXd finiteDifferenceJacobianOfClampingConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXd v0);

  /// This computes the finite difference Jacobian of A_ub*E*f0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXd finiteDifferenceJacobianOfUpperBoundConstraints(
      simulation::WorldPtr world, Eigen::VectorXd f0);

  /// This computes and returns the jacobian of P_c * v by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
      simulation::WorldPtr world, Eigen::VectorXd v, WithRespectTo* wrt);

  /// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
  /// finite differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfMinv(
      simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt);

  /// This computes and returns the jacobian of C(pos, inertia, vel) by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the jacobian of M^{-1}(pos, inertia) * C(pos,
  /// inertia, vel) by finite differences. This is SUPER SLOW, and is only here
  /// for testing.
  Eigen::MatrixXd finiteDifferenceJacobianOfMinvC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of constraint force, holding everyhing constant
  /// except the value of WithRespectTo
  Eigen::MatrixXd finiteDifferenceJacobianOfConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of estimated constraint force, without actually
  /// running forward passes, holding everyhing constant except the value of
  /// WithRespectTo
  Eigen::MatrixXd finiteDifferenceJacobianOfEstimatedConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// These was the mX() vector used to construct this. Pretty much only here
  /// for testing.
  Eigen::VectorXd getContactConstraintImpluses();

  /// These was the fIndex() vector used to construct this. Pretty much only
  /// here for testing.
  Eigen::VectorXi getContactConstraintMappings();

  /// Returns the vector of the coefficients on the diagonal of the bounce
  /// matrix. These are 1+restitutionCoeff[i].
  Eigen::VectorXd getBounceDiagonals();

  /// Returns the vector of the restitution coeffs, sized for the number of
  /// bouncing collisions.
  Eigen::VectorXd getRestitutionDiagonals();

  /// Returns the penetration correction hack "bounce" (or 0 if the contact is
  /// not inter-penetrating or is actively bouncing) at each contact point.
  Eigen::VectorXd getPenetrationCorrectionVelocities();

  /// Returns the constraint impulses along the clamping constraints
  Eigen::VectorXd getClampingConstraintImpulses();

  /// Returns the relative velocities along the clamping constraints
  Eigen::VectorXd getClampingConstraintRelativeVels();

  /// Returns the velocity change caused by illegal impulses in the LCP this
  /// timestep
  Eigen::VectorXd getVelocityDueToIllegalImpulses();

  /// Returns the velocity pre-LCP
  Eigen::VectorXd getPreLCPVelocity();

  /// Returns true if there were any bounces in this snapshot.
  bool hasBounces();

  /// Returns the number of clamping contacts in this snapshot.
  std::size_t getNumClamping();

  /// Returns the number of upper bound contacts in this snapshot.
  std::size_t getNumUpperBound();

  /// These are the gradient constraint matrices from the LCP solver
  std::vector<std::shared_ptr<ConstrainedGroupGradientMatrices>>
      mGradientMatrices;

  /// This is the clamping constraints from all the constrained
  /// groups, concatenated together
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
  getDifferentiableConstraints();

  /// This is the clamping constraints from all the constrained
  /// groups, concatenated together
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
  getClampingConstraints();

  /// This is the upper bound constraints from all the constrained
  /// groups, concatenated together
  std::vector<std::shared_ptr<DifferentiableContactConstraint>>
  getUpperBoundConstraints();

  /// This verifies that the two matrices are equal to some tolerance, and if
  /// they're not it prints the information needed to replicated this scenario
  /// and it exits the program.
  void equalsOrCrash(
      std::shared_ptr<simulation::World> world,
      Eigen::MatrixXd analytical,
      Eigen::MatrixXd bruteForce,
      std::string name);

  /// This prints code to the console to replicate a scenario for testing.
  void printReplicationInstructions(std::shared_ptr<simulation::World> world);

  /// Returns true if we were able to standardize our LCP results, false if we
  /// weren't
  bool areResultsStandardized() const;

  /// If this is true, we use finite-differencing to compute all of the
  /// requested Jacobians. This override can be useful to verify if there's a
  /// bug in the analytical Jacobians that's causing learning to not converge.
  void setUseFDOverride(bool override);

  /// If this is true, we check all Jacobians against their finite-differencing
  /// counterparts at runtime. If they aren't sufficiently close, we immediately
  /// crash the program and print what went wrong and some simple replication
  /// instructions.
  void setSlowDebugResultsAgainstFD(bool slowDebug);

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

  /// This is the global timestep length. This is included here because it shows
  /// up as a constant in some of the matrices.
  double mTimeStep;

  /// This is the total DOFs for this World
  std::size_t mNumDOFs;

  /// This is the number of total dimensions on all the constraints active in
  /// the world
  std::size_t mNumConstraintDim;

  /// This is the number of total constraint dimensions that are clamping
  std::size_t mNumClamping;

  /// This is the number of total constraint dimensions that are upper bounded
  std::size_t mNumUpperBound;

  /// This is the number of total constraint dimensions that are upper bounded
  std::size_t mNumBouncing;

  /// These are the offsets into the total degrees of freedom for each skeleton
  std::unordered_map<std::string, std::size_t> mSkeletonOffset;

  /// These are the number of degrees of freedom for each skeleton
  std::unordered_map<std::string, std::size_t> mSkeletonDofs;

  /// The position of all the DOFs of the world BEFORE the timestep
  Eigen::VectorXd mPreStepPosition;

  /// The velocities of all the DOFs of the world BEFORE the timestep
  Eigen::VectorXd mPreStepVelocity;

  /// The torques on all the DOFs of the world BEFORE the timestep
  Eigen::VectorXd mPreStepTorques;

  /// The LCP's initial cached value BEFORE the timestep
  Eigen::VectorXd mPreStepLCPCache;

  /// The velocities of all the DOFs of the world AFTER an unconstrained forward
  /// step, but BEFORE the LCP runs
  Eigen::VectorXd mPreConstraintVelocities;

  /// The position of all the DOFs of the world AFTER the timestep
  Eigen::VectorXd mPostStepPosition;

  /// The velocities of all the DOFs of the world AFTER the timestep
  /// created
  Eigen::VectorXd mPostStepVelocity;

  /// The torques on all the DOFs of the world AFTER the timestep
  Eigen::VectorXd mPostStepTorques;

private:
  /// These are mCached versions of the various Jacobians
  bool mCachedPosPosDirty;
  Eigen::MatrixXd mCachedPosPos;
  bool mCachedPosVelDirty;
  Eigen::MatrixXd mCachedPosVel;
  bool mCachedBounceApproximationDirty;
  Eigen::MatrixXd mCachedBounceApproximation;
  bool mCachedVelPosDirty;
  Eigen::MatrixXd mCachedVelPos;
  bool mCachedVelVelDirty;
  Eigen::MatrixXd mCachedVelVel;
  bool mCachedForcePosDirty;
  Eigen::MatrixXd mCachedForcePos;
  bool mCachedForceVelDirty;
  Eigen::MatrixXd mCachedForceVel;
  bool mCachedMassVelDirty;
  Eigen::MatrixXd mCachedMassVel;

  Eigen::VectorXd scratch(simulation::WorldPtr world);

  enum MatrixToAssemble
  {
    CLAMPING,
    MASSED_CLAMPING,
    UPPER_BOUND,
    MASSED_UPPER_BOUND,
    BOUNCING
  };

  Eigen::MatrixXd assembleMatrix(
      simulation::WorldPtr world, MatrixToAssemble whichMatrix);

  enum BlockDiagonalMatrixToAssemble
  {
    MASS,
    INV_MASS,
    POS_C,
    VEL_C
  };

  Eigen::MatrixXd assembleBlockDiagonalMatrix(
      simulation::WorldPtr world,
      BlockDiagonalMatrixToAssemble whichMatrix,
      bool forFiniteDifferencing = false);

  enum VectorToAssemble
  {
    CONTACT_CONSTRAINT_IMPULSES,
    CONTACT_CONSTRAINT_MAPPINGS,
    BOUNCE_DIAGONALS,
    RESTITUTION_DIAGONALS,
    PENETRATION_VELOCITY_HACK,
    CLAMPING_CONSTRAINT_IMPULSES,
    CLAMPING_CONSTRAINT_RELATIVE_VELS,
    VEL_DUE_TO_ILLEGAL,
    PRE_STEP_VEL,
    PRE_STEP_TAU,
    PRE_LCP_VEL
  };
  template <typename Vec>
  Vec assembleVector(VectorToAssemble whichVector);

  template <typename Vec>
  const Vec& getVectorToAssemble(
      std::shared_ptr<ConstrainedGroupGradientMatrices> matrices,
      VectorToAssemble whichVector);
};

using BackpropSnapshotPtr = std::shared_ptr<BackpropSnapshot>;

} // namespace neural
} // namespace dart

#endif