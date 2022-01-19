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
      Eigen::VectorXs preStepPosition,
      Eigen::VectorXs preStepVelocity,
      Eigen::VectorXs preStepTorques,
      Eigen::VectorXs preConstraintVelocities,
      Eigen::VectorXs preStepLCPCache);

  /// This computes the implicit backprop without forming intermediate
  /// Jacobians. It takes a LossGradient with the position and velocity vectors
  /// filled it, though the loss with respect to torque is ignored and can be
  /// null. It returns a LossGradient with all three values filled in, position,
  /// velocity, and torque.
  void backprop(
      simulation::WorldPtr world,
      LossGradient& thisTimestepLoss,
      const LossGradient& nextTimestepLoss,
      PerformanceLog* perfLog = nullptr,
      bool exploreAlternateStrategies = false);

  /// This computes backprop in the high-level RL API's space, use `state` and
  /// `action` as the primitives we're taking gradients wrt to.
  LossGradientHighLevelAPI backpropState(
      simulation::WorldPtr world,
      const Eigen::VectorXs& nextTimestepStateLossGrad,
      PerformanceLog* perfLog = nullptr,
      bool exploreAlternateStrategies = false);

  /// This zeros out any components of the gradient that would want to push us
  /// out of the box-bounds encoded in the world for pos, vel, or force.
  void clipLossGradientsToBounds(
      simulation::WorldPtr world,
      Eigen::VectorXs& lossWrtPos,
      Eigen::VectorXs& lossWrtVel,
      Eigen::VectorXs& lossWrtForce);

  /// This computes and returns the whole vel-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getVelVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactFreeVelVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// Index should specify which contact will be kept
  const Eigen::MatrixXs& getContactReducedVelVelJacobian(
      simulation::WorldPtr world, Eigen::VectorXs indexs, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getPosVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactFreePosVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactReducedPosVelJacobian(
      simulation::WorldPtr world, Eigen::VectorXs indexs, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole force-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getControlForceVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactFreeControlForceVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactReducedCControlForceVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole mass-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getMassVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole damping coeffient-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getDampingVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole spring stiffness-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getSpringVelJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole pos-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getPosPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactFreePosPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the whole vel-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getVelPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  const Eigen::MatrixXs& getContactFreeVelPosJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This computes and returns the component of the pos-pos and pos-vel
  /// jacobians due to bounce approximation. For backprop, you don't actually
  /// need this matrix, you can compute backprop directly. This is here if you
  /// want access to the full Jacobian for some reason.
  const Eigen::MatrixXs& getBounceApproximationJacobian(
      simulation::WorldPtr world, PerformanceLog* perfLog = nullptr);

  /// This returns the Jacobian for state_t -> state_{t+1}.
  Eigen::MatrixXs getStateJacobian(simulation::WorldPtr world);

  Eigen::MatrixXs getContactFreeStateJacobian(simulation::WorldPtr world);

  Eigen::MatrixXs getContactReducedStateJacobian(
      simulation::WorldPtr world,
      Eigen::VectorXs indexs);

  /// This returns the Jacobian for action_t -> state_{t+1}.
  Eigen::MatrixXs getActionJacobian(simulation::WorldPtr world);

  Eigen::MatrixXs getContactFreeActionJacobian(simulation::WorldPtr world);

  Eigen::MatrixXs getContactReducedActionJacobian(
      simulation::WorldPtr world,
      Eigen::VectorXs indexs);
  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  Eigen::VectorXs getPreStepPosition();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  Eigen::VectorXs getPreStepVelocity();

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, BEFORE the timestep.
  Eigen::VectorXs getPreStepTorques();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER integrating forward dynamics but BEFORE
  /// running the LCP.
  Eigen::VectorXs getPreConstraintVelocity();

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  Eigen::VectorXs getPostStepPosition();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  Eigen::VectorXs getPostStepVelocity();

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, AFTER the timestep.
  Eigen::VectorXs getPostStepTorques();

  /// Returns the LCP's cached solution from before the step
  const Eigen::VectorXs& getPreStepLCPCache();

  /////////////////////////////////////////////////////////////////////////////
  /// Just public for testing
  /////////////////////////////////////////////////////////////////////////////

  /// This returns the A_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getClampingConstraintMatrix(simulation::WorldPtr world);

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getMassedClampingConstraintMatrix(simulation::WorldPtr world);

  /// This returns the A_ub matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getUpperBoundConstraintMatrix(simulation::WorldPtr world);

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getMassedUpperBoundConstraintMatrix(
      simulation::WorldPtr world);

  /// This returns the E matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getUpperBoundMappingMatrix();

  /// This returns the B matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getBouncingConstraintMatrix(simulation::WorldPtr world);

  /// This return the entire constraint matrix which may be used to provide heuristic
  /// For iLQR planning through contact
  Eigen::MatrixXs getAllConstraintMatrix(simulation::WorldPtr world);

  /// This returns the mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton mass matrices.
  Eigen::MatrixXs getMassMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This returns the inverse mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton inverse mass matrices.
  Eigen::MatrixXs getInvMassMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This return the diagonal matrix representing Coefficients of damping
  Eigen::VectorXs getDampingVector(simulation::WorldPtr world);

  /// This return the diagonal matrix(vector) representing spring stiffness
  Eigen::VectorXs getSpringStiffVector(simulation::WorldPtr world);

  /// This returns the rest position vector of each degree of freedom
  Eigen::VectorXs getRestPositions(simulation::WorldPtr world);

  /// This is the subset of the A matrix from the original LCP that corresponds
  /// to clamping indices.
  Eigen::MatrixXs getClampingAMatrix();

  /// This returns the pos-C(pos,vel) Jacobian for the whole world, a block
  /// diagonal concatenation of the skeleton pos-C(pos,vel) Jacobians.
  Eigen::MatrixXs getPosCJacobian(simulation::WorldPtr world);

  /// This returns the vel-C(pos,vel) Jacobian for the whole world, a block
  /// diagonal concatenation of the skeleton vel-C(pos,vel) Jacobians.
  Eigen::MatrixXs getVelCJacobian(simulation::WorldPtr world);

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceVelVelJacobian(
      simulation::WorldPtr world, bool useRidders = true);

  /// This computes and returns the whole pos-C(pos,vel) jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferencePosVelJacobian(
      simulation::WorldPtr world, bool useRidders = true);

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceForceVelJacobian(
      simulation::WorldPtr world, bool useRidders = true);

  /// This computes and returns the whole mass-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceMassVelJacobian(
      simulation::WorldPtr world, bool useRidders = true);

  /// This computes and returns the whole pos-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferencePosPosJacobian(
      simulation::WorldPtr world,
      std::size_t subdivisions = 20,
      bool useRidders = true);

  /// This computes and returns the whole vel-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceVelPosJacobian(
      simulation::WorldPtr world,
      std::size_t subdivisions = 20,
      bool useRidders = true);

  /// This computes and returns the whole wrt-vel jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This computes and returns the whole wrt-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferencePosJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This returns the P_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXs getProjectionIntoClampsMatrix(
      simulation::WorldPtr world, bool forFiniteDifferencing = false);

  /// This replaces x with the result of M*x in place, without explicitly
  /// forming M
  Eigen::VectorXs implicitMultiplyByMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXs& x);

  /// This replaces x with the result of Minv*x in place, without explicitly
  /// forming Minv
  Eigen::VectorXs implicitMultiplyByInvMassMatrix(
      simulation::WorldPtr world, const Eigen::VectorXs& x);

  /// TODO(keenon): Remove me
  Eigen::MatrixXs getScratchAnalytical(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// TODO(keenon): Remove me
  Eigen::MatrixXs getScratchFiniteDifference(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This predicts what the next velocity will be using our linear algebra
  /// formula. This is only here for testing, to compare it against the actual
  /// result of a timestep.
  ///
  /// The `morePreciseButSlower` flag tells this function to do brute force
  /// steps to get constraint matrices A_c and A_ub, rather than use 1st order
  /// approximations. This is important because when we're doing
  /// finite-differencing over tiny EPS (1e-9) then tiny errors in the 1st order
  /// approximations blow up to become huge errors in gradients.
  Eigen::VectorXs getAnalyticalNextV(
      simulation::WorldPtr world, bool morePreciseButSlower = false);

  /// This computes and returns the whole wrt-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXs getVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the whole wrt-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason
  Eigen::MatrixXs getContactReducedVelJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt, Eigen::VectorXs index);

  /// This computes and returns the whole wrt-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXs getPosJacobianWrt(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of constraint force, holding everyhing constant
  /// except the value of WithRespectTo
  Eigen::MatrixXs getJacobianOfConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the analytical expression for the Jacobian of Q*b, holding b
  /// constant, if there are some upper-bound indices
  Eigen::MatrixXs dQ_WithUB(
      simulation::WorldPtr world,
      Eigen::MatrixXs& Minv,
      Eigen::MatrixXs& A_c,
      Eigen::MatrixXs& E,
      Eigen::MatrixXs& A_c_ub_E,
      Eigen::VectorXs rhs,
      WithRespectTo* wrt);
  /// This returns the analytical expression for the Jacobian of Q^T*b, holding
  /// b constant, if there are some upper-bound indices
  Eigen::MatrixXs dQT_WithUB(
      simulation::WorldPtr world,
      Eigen::MatrixXs& Minv,
      Eigen::MatrixXs& A_c,
      Eigen::MatrixXs& E,
      Eigen::MatrixXs& A_ub,
      Eigen::VectorXs rhs,
      WithRespectTo* wrt);
  /// This returns the analytical expression for the Jacobian of Q*b, holding b
  /// constant, if there are no upper-bound indices
  Eigen::MatrixXs dQ_WithoutUB(
      simulation::WorldPtr world,
      Eigen::MatrixXs& Minv,
      Eigen::MatrixXs& A_c,
      Eigen::VectorXs rhs,
      WithRespectTo* wrt);

  /// This returns the jacobian of Qb, holding b constant, with respect to
  /// wrt, by finite differencing
  Eigen::MatrixXs finiteDifferenceJacobianOfQb(
      simulation::WorldPtr world,
      Eigen::VectorXs b,
      WithRespectTo* wrt,
      bool useRidders = true);

  /// This returns the vector of constants that get added to the diagonal of Q
  /// to guarantee that Q is full-rank
  Eigen::VectorXs getConstraintForceMixingDiagonal();

  /// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
  /// wrt
  Eigen::MatrixXs getJacobianOfLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world, Eigen::VectorXs b, WithRespectTo* wrt);

  /// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
  /// wrt, by finite differencing
  Eigen::MatrixXs finiteDifferenceJacobianOfLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world,
      Eigen::VectorXs b,
      WithRespectTo* wrt,
      bool useRidders = true);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
  Eigen::MatrixXs getJacobianOfLCPOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
  /// finite differencing
  Eigen::MatrixXs finiteDifferenceJacobianOfLCPOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This returns the jacobian of b (from Q^{-1}b) with respect to wrt, by
  /// finite differencing
  Eigen::MatrixXs finiteDifferenceJacobianOfLCPEstimatedOffsetClampingSubset(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This returns the subset of the A matrix used by the original LCP for just
  /// the clamping constraints. It relates constraint force to constraint
  /// acceleration. It's a mass matrix, just in a weird frame.
  void computeLCPConstraintMatrixClampingSubset(
      simulation::WorldPtr world,
      Eigen::MatrixXs& Q,
      const Eigen::MatrixXs& A_c,
      const Eigen::MatrixXs& A_ub,
      const Eigen::MatrixXs& E);

  /// This returns the subset of the b vector used by the original LCP for just
  /// the clamping constraints. It's just the relative velocity at the clamping
  /// contact points.
  void computeLCPOffsetClampingSubset(
      simulation::WorldPtr world,
      Eigen::VectorXs& b,
      const Eigen::MatrixXs& A_c);

  /// This computes and returns an estimate of the constraint impulses for the
  /// clamping constraints. This is based on a linear approximation of the
  /// constraint impulses.
  Eigen::VectorXs estimateClampingConstraintImpulses(
      simulation::WorldPtr world,
      const Eigen::MatrixXs& A_c,
      const Eigen::MatrixXs& A_ub,
      const Eigen::MatrixXs& E);

  /// This returns the jacobian of P_c * v, holding everyhing constant except
  /// the value of WithRespectTo
  Eigen::MatrixXs getJacobianOfProjectionIntoClampsMatrix(
      simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt);

  /// This computes and returns the jacobian of P_c * v by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceJacobianOfProjectionIntoClampsMatrix(
      simulation::WorldPtr world,
      Eigen::VectorXs v,
      WithRespectTo* wrt,
      bool useRidders = true);

  /// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
  /// everything constant except the value of WithRespectTo
  Eigen::MatrixXs getJacobianOfMinv(
      simulation::WorldPtr world, Eigen::VectorXs tau, WithRespectTo* wrt);

  /// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
  /// finite differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceJacobianOfMinv(
      simulation::WorldPtr world,
      Eigen::VectorXs tau,
      WithRespectTo* wrt,
      bool useRidders = true);

  /// This returns the jacobian of M(pos, inertia) * v, holding
  /// everything constant except the value of WithRespectTo
  Eigen::MatrixXs getJacobianOfM(
      simulation::WorldPtr world, Eigen::VectorXs v, WithRespectTo* wrt);

  /// This computes and returns the jacobian of M(pos, inertia) * v by
  /// finite differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceJacobianOfM(
      simulation::WorldPtr world,
      Eigen::VectorXs v,
      WithRespectTo* wrt,
      bool useRidders = true);

  /// This returns the jacobian of C(pos, inertia, vel), holding everything
  /// constant except the value of WithRespectTo
  Eigen::MatrixXs getJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This returns the jacobian of C(pos, inertia, vel), holding everything
  /// constant except the value of WithRespectTo
  Eigen::MatrixXs computeJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the jacobian of C(pos, inertia, vel) by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceJacobianOfC(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This returns the jacobian of M^{-1}(pos, inertia) * (C(pos, inertia, vel)
  /// + mPreStepTorques), holding everything constant except the value of
  /// WithRespectTo
  Eigen::MatrixXs getJacobianOfMinvC(
      simulation::WorldPtr world, WithRespectTo* wrt);

  /// This computes and returns the jacobian of M^{-1}(pos, inertia) * C(pos,
  /// inertia, vel) by finite differences. This is SUPER SLOW, and is only here
  /// for testing.
  Eigen::MatrixXs finiteDifferenceJacobianOfMinvC(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = false);

  /// This returns a fast approximation to A_c in the neighborhood of the
  /// original
  Eigen::MatrixXs estimateClampingConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// This returns a fast approximation to A_ub in the neighborhood of the
  /// original
  Eigen::MatrixXs estimateUpperBoundConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// Only for testing: VERY SLOW. This returns the actual value of A_c at the
  /// desired position.
  Eigen::MatrixXs getClampingConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// Only for testing: VERY SLOW. This returns the actual value of A_ub at the
  /// desired position.
  Eigen::MatrixXs getUpperBoundConstraintMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// Only for testing: VERY SLOW. This returns the actual value of E at the
  /// desired position.
  Eigen::MatrixXs getUpperBoundMappingMatrixAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// Only for testing: VERY SLOW. This returns the actual value of the bounce
  /// diagonals at the desired position.
  Eigen::VectorXs getBounceDiagonalsAt(
      simulation::WorldPtr world, Eigen::VectorXs pos);

  /// This computes the Jacobian of A_c*f0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXs getJacobianOfClampingConstraints(
      simulation::WorldPtr world, Eigen::VectorXs f0);

  /// This computes the finite difference Jacobian of A_c*f0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXs finiteDifferenceJacobianOfClampingConstraints(
      simulation::WorldPtr world, Eigen::VectorXs f0, bool useRidders = true);

  /// This computes the Jacobian of A_c^T*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXs getJacobianOfClampingConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXs v0);

  /// This computes the finite difference Jacobian of A_c^T*v0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXs finiteDifferenceJacobianOfClampingConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXs v0, bool useRidders = true);

  /// This computes the Jacobian of A_ub*(E*f0) with respect to position using
  /// impulse tests.
  Eigen::MatrixXs getJacobianOfUpperBoundConstraints(
      simulation::WorldPtr world, Eigen::VectorXs E_f0);

  /// This computes the finite difference Jacobian of A_ub*E*f0 with respect to
  /// position. This is AS SLOW AS FINITE DIFFERENCING THE WHOLE ENGINE, which
  /// is way too slow to use in practice.
  Eigen::MatrixXs finiteDifferenceJacobianOfUpperBoundConstraints(
      simulation::WorldPtr world, Eigen::VectorXs f0, bool useRidders = true);

  /// This computes the Jacobian of A_ub^T*v0 with respect to position using
  /// impulse tests.
  Eigen::MatrixXs getJacobianOfUpperBoundConstraintsTranspose(
      simulation::WorldPtr world, Eigen::VectorXs v0);

  /// This returns the jacobian of constraint force, holding everything constant
  /// except the value of WithRespectTo
  Eigen::MatrixXs finiteDifferenceJacobianOfConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// This returns the jacobian of estimated constraint force, without actually
  /// running forward passes, holding everyhing constant except the value of
  /// WithRespectTo
  Eigen::MatrixXs finiteDifferenceJacobianOfEstimatedConstraintForce(
      simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders = true);

  /// These was the mX() vector used to construct this. Pretty much only here
  /// for testing.
  Eigen::VectorXs getContactConstraintImpulses();

  /// These was the fIndex() vector used to construct this. Pretty much only
  /// here for testing.
  Eigen::VectorXi getContactConstraintMappings();

  /// Returns the vector of the coefficients on the diagonal of the bounce
  /// matrix. These are 1+restitutionCoeff[i].
  Eigen::VectorXs getBounceDiagonals();

  /// Returns the vector of the restitution coeffs, sized for the number of
  /// bouncing collisions.
  Eigen::VectorXs getRestitutionDiagonals();

  /// Returns the penetration correction hack "bounce" (or 0 if the contact is
  /// not inter-penetrating or is actively bouncing) at each contact point.
  Eigen::VectorXs getPenetrationCorrectionVelocities();

  /// Returns the constraint impulses along the clamping constraints
  Eigen::VectorXs getClampingConstraintImpulses();

  /// Returns the relative velocities along the clamping constraints
  Eigen::VectorXs getClampingConstraintRelativeVels();

  /// Returns the velocity change caused by illegal impulses in the LCP this
  /// timestep
  Eigen::VectorXs getVelocityDueToIllegalImpulses();

  /// Returns the velocity pre-LCP
  Eigen::VectorXs getPreLCPVelocity();

  /// Returns true if there were any bounces in this snapshot.
  bool hasBounces();

  /// Returns true if we had to deliberately ignore friction on any of our sub-groups in order to solve.
  bool getDeliberatelyIgnoreFriction();

  /// Returns the number of contacts (regardless of state) in this snapshot.
  std::size_t getNumContacts();

  /// Returns the number of clamping contacts in this snapshot.
  std::size_t getNumClamping();

  /// Returns the number of upper bound contacts in this snapshot.
  std::size_t getNumUpperBound();

  // Returns number of constraint dimension
  std::size_t getNumConstraintDim();

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
      Eigen::MatrixXs analytical,
      Eigen::MatrixXs bruteForce,
      std::string name);

  /// This compares our analytical sub-Jacobians (like dMinv), to attempt to
  /// diagnose where there are differences creeping in between our finite
  /// differencing and our analytical results.
  void diagnoseSubJacobianErrors(
      std::shared_ptr<simulation::World> world, WithRespectTo* wrt);

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

  /// This does a battery of tests comparing the speeds to compute all the
  /// different Jacobians, both with finite differencing and analytically, and
  /// prints the results to std out.
  void benchmarkJacobians(
      std::shared_ptr<simulation::World> world, int numSamples);

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
  s_t mTimeStep;

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
  Eigen::VectorXs mPreStepPosition;

  /// The velocities of all the DOFs of the world BEFORE the timestep
  Eigen::VectorXs mPreStepVelocity;

  /// The torques on all the DOFs of the world BEFORE the timestep
  Eigen::VectorXs mPreStepTorques;

  /// The LCP's initial cached value BEFORE the timestep
  Eigen::VectorXs mPreStepLCPCache;

  /// The velocities of all the DOFs of the world AFTER an unconstrained forward
  /// step, but BEFORE the LCP runs
  Eigen::VectorXs mPreConstraintVelocities;

  /// The position of all the DOFs of the world AFTER the timestep
  Eigen::VectorXs mPostStepPosition;

  /// The velocities of all the DOFs of the world AFTER the timestep
  /// created
  Eigen::VectorXs mPostStepVelocity;

  /// The torques on all the DOFs of the world AFTER the timestep
  Eigen::VectorXs mPostStepTorques;

private:
  /// These are mCached versions of the various Jacobians
  bool mCachedPosPosDirty;
  Eigen::MatrixXs mCachedPosPos;
  bool mCachedContactFreePosPosDirty;
  Eigen::MatrixXs mCachedContactFreePosPos;
  bool mCachedPosVelDirty;
  Eigen::MatrixXs mCachedPosVel;
  bool mCachedContactFreePosVelDirty;
  Eigen::MatrixXs mCachedContactFreePosVel;
  bool mCachedBounceApproximationDirty;
  Eigen::MatrixXs mCachedBounceApproximation;
  bool mCachedVelPosDirty;
  Eigen::MatrixXs mCachedVelPos;
  bool mCachedContactFreeVelPosDirty;
  Eigen::MatrixXs mCachedContactFreeVelPos;
  bool mCachedVelVelDirty;
  Eigen::MatrixXs mCachedVelVel;
  bool mCachedContactFreeVelVelDirty;
  Eigen::MatrixXs mCachedContactFreeVelVel;
  bool mCachedForcePosDirty;
  Eigen::MatrixXs mCachedForcePos;
  bool mCachedContactFreeForcePosDirty;
  Eigen::MatrixXs mCachedContactFreeForcePos;
  bool mCachedForceVelDirty;
  Eigen::MatrixXs mCachedForceVel;
  bool mCachedContactFreeForceVelDirty;
  Eigen::MatrixXs mCachedContactFreeForceVel;
  bool mCachedMassVelDirty;
  Eigen::MatrixXs mCachedMassVel;
  bool mCachedPosCDirty;
  Eigen::MatrixXs mCachedPosC;
  bool mCachedVelCDirty;
  Eigen::MatrixXs mCachedVelC;
  bool mCachedDampingVelDirty;
  Eigen::MatrixXs mCachedDampingVel;
  bool mCachedSpringVelDirty;
  Eigen::MatrixXs mCachedSpringVel;

  Eigen::VectorXs scratch(simulation::WorldPtr world);

  enum MatrixToAssemble
  {
    CLAMPING,
    MASSED_CLAMPING,
    UPPER_BOUND,
    MASSED_UPPER_BOUND,
    BOUNCING,
    ALL
  };

  Eigen::MatrixXs assembleMatrix(
      simulation::WorldPtr world, MatrixToAssemble whichMatrix);

  enum BlockDiagonalMatrixToAssemble
  {
    MASS,
    INV_MASS,
    POS_C,
    VEL_C
  };

  Eigen::MatrixXs assembleBlockDiagonalMatrix(
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
    PRE_LCP_VEL,
    CFM_CONSTANTS
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