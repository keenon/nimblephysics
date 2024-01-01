#ifndef EXO_SOLVER_PINNED_CONTACT
#define EXO_SOLVER_PINNED_CONTACT

#include <memory>

#include <Eigen/Dense>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace exo {

class ExoSolverPinnedContact
{
public:
  /// \brief Constructor
  /// Both the real and virtual skeletons must be identical in their number of
  /// DOFs and their structure (names of bodies, etc). The key difference is the
  /// masses, inertias, gravity, and spring forces of the virtual skeleton.
  ExoSolverPinnedContact(
      std::shared_ptr<dynamics::Skeleton> realSkel,
      std::shared_ptr<dynamics::Skeleton> virtualSkel);

  void addMotorDof(int dofIndex);

  void setPositions(Eigen::VectorXs q);

  Eigen::MatrixXs getExoToJointTorquesJacobian();

  /// Set the contact points that we will use when solving inverse dynamics.
  void setContactPins(std::vector<std::pair<int, Eigen::Vector3s>> pins);

  /// Get the Jacobian relating world space velocity of the contact points to
  /// joint velocities.
  Eigen::MatrixXs getContactJacobian();

  /// This is only used for testing: Get the Jacobian relating world space
  /// velocity of the contact points to joint velocities, by finite
  /// differencing.
  Eigen::MatrixXs finiteDifferenceContactJacobian();

  /// This is only used for testing, to allow us to compare the analytical
  /// solution to the numerical solution.
  Eigen::VectorXs analyticalForwardDynamics(
      Eigen::VectorXs dq,
      Eigen::VectorXs tau,
      Eigen::VectorXs exoTorques,
      Eigen::VectorXs contactForces);

  /// This is only used for testing, to allow us to compare the analytical
  /// solution to the numerical solution.
  Eigen::VectorXs implicitForwardDynamics(
      Eigen::VectorXs dq,
      Eigen::VectorXs tau,
      Eigen::VectorXs exoTorques,
      Eigen::VectorXs contactForces);

  /// This is part of the main exoskeleton solver. It takes in the current
  /// joint velocities and accelerations, and the last exoskeleton torques, and
  /// returns the estimated human pilot joint torques.
  Eigen::VectorXs estimateHumanTorques(
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs contactForces,
      Eigen::VectorXs lastExoTorques);

  /// This is part of the main exoskeleton solver. It takes in the current
  /// joint velocities and accelerations, and returns the estimated total
  /// joint torques for the human + exoskeleton system.
  Eigen::VectorXs estimateTotalTorques(
      Eigen::VectorXs dq, Eigen::VectorXs ddq, Eigen::VectorXs contactForces);

  /// This is part of the main exoskeleton solver. It takes in the current
  /// estimated human pilot joint torques, and computes the accelerations we
  /// would see on the virtual skeleton if we applied those same torques, with
  /// the contacts pinned at the CoPs.
  std::pair<Eigen::VectorXs, Eigen::VectorXs> getPinnedVirtualDynamics(
      Eigen::VectorXs dq, Eigen::VectorXs tau);

  /// This does the same thing as getPinndVirtualDynamics, but returns the Ax +
  /// b values A and b such that Ax + b = ddq, accounting for the pin
  /// constraints.
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> getPinnedVirtualDynamicsLinearMap(
      Eigen::VectorXs dq);

  /// This is not part of the main exoskeleton solver, but is useful for the
  /// inverse problem of analyzing the human pilot's joint torques under
  /// different assistance strategies.
  std::pair<Eigen::VectorXs, Eigen::VectorXs> getPinnedRealDynamics(
      Eigen::VectorXs dq, Eigen::VectorXs tau);

  /// This does the same thing as getPinndRealDynamics, but returns the Ax +
  /// b values A and b such that Ax + b = ddq, accounting for the pin
  /// constraints.
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> getPinnedRealDynamicsLinearMap(
      Eigen::VectorXs dq);

  /// This is part of the main exoskeleton solver. It takes in how the digital
  /// twin of the exo pilot is accelerating, and attempts to solve for the
  /// torques that the exo needs to apply to get as close to that as possible.
  /// It resolves ambiguities by minimizing the change in total system torques.
  std::pair<Eigen::VectorXs, Eigen::VectorXs> getPinnedTotalTorques(
      Eigen::VectorXs dq,
      Eigen::VectorXs ddqDesired,
      Eigen::VectorXs centeringTau,
      Eigen::VectorXs centeringForces);

  /// This does the same thing as getPinnedTotalTorques, but returns the Ax +
  /// b values A and b such that Ax + b = tau, accounting for the pin
  /// constraints.
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> getPinnedTotalTorquesLinearMap(
      Eigen::VectorXs dq);

  /// This is part of the main exoskeleton solver. It takes in the desired
  /// torques for the exoskeleton, and returns the torques on the actuated
  /// DOFs that can be used to drive the exoskeleton.
  Eigen::VectorXs projectTorquesToExoControlSpace(Eigen::VectorXs torques);

  /// This does the same thing as projectTorquesToExoControlSpace, but returns
  /// the matrix to multiply by the torques to get the exo torques.
  Eigen::MatrixXs projectTorquesToExoControlSpaceLinearMap();

  /// Often our estimates for `dq` and `ddq` violate the pin constraints. That
  /// leads to exo torques that do not tend to zero as the virtual human exactly
  /// matches the real human+exo system. To solve this problem, we can solve a
  /// set of least-squares equations to find the best set of ddq values to
  /// satisfy the constraint.
  Eigen::VectorXs getClosestRealAccelerationConsistentWithPinsAndContactForces(
      Eigen::VectorXs dq, Eigen::VectorXs ddq, Eigen::VectorXs contactForces);

  /// This runs the entire exoskeleton solver pipeline, spitting out the
  /// torques to apply to the exoskeleton actuators.
  Eigen::VectorXs solveFromAccelerations(
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs lastExoTorques,
      Eigen::VectorXs contactForces);

  /// This is a subset of the steps in solveFromAccelerations, which can take
  /// the biological joint torques directly, and solve for the exo torques.
  Eigen::VectorXs solveFromBiologicalTorques(
      Eigen::VectorXs dq,
      Eigen::VectorXs humanTau,
      Eigen::VectorXs centeringTau,
      Eigen::VectorXs centeringForces);

  /// This is the same as solveFromBiologicalTorques, but returns the Ax + b
  /// values A and b such that Ax + b = exo_tau, accounting for the pin
  /// constraints.
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> getExoTorquesLinearMap(
      Eigen::VectorXs dq);

  /// This does a simple forward dynamics step, given the current human joint
  /// torques, factoring in how the exoskeleton will respond to those torques.
  /// This returns the `ddq` that we would see on the human, and the contact
  /// forces we see at the pin constraints, and the exo torques we would get.
  std::tuple<Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs>
  getPinnedForwardDynamicsForExoAndHuman(
      Eigen::VectorXs dq, Eigen::VectorXs humanTau);

  /// This does the same thing as getPinnedForwardDynamicsForExoAndHuman, but
  /// returns the Ax + b values A and b such that Ax + b = ddq, accounting for
  /// the pin constraints.
  std::pair<Eigen::MatrixXs, Eigen::VectorXs>
  getPinnedForwardDynamicsForExoAndHumanLinearMap(Eigen::VectorXs dq);

  /// Given the desired end-kinematics, after the human and exoskeleton have
  /// finished "negotiating" how they will collaborate, this computes the
  /// resulting human and exoskeleton torques.
  std::pair<Eigen::VectorXs, Eigen::VectorXs> getHumanAndExoTorques(
      Eigen::VectorXs dq, Eigen::VectorXs ddq);

protected:
  std::shared_ptr<dynamics::Skeleton> mRealSkel;
  std::shared_ptr<dynamics::Skeleton> mVirtualSkel;

  std::vector<int> mMotorDofs;
  std::vector<std::pair<int, Eigen::Vector3s>> mPins;
};

} // namespace exo
} // namespace dart

#endif