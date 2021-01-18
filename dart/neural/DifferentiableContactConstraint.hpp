#ifndef DART_NEURAL_DIFF_CONSTRAINT_HPP_
#define DART_NEURAL_DIFF_CONSTRAINT_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/collision/Contact.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace constraint {
class ConstrainedGroup;
class ConstraintBase;
class ContactConstraint;
} // namespace constraint

namespace dynamics {
class Skeleton;
}

namespace simulation {
class World;
}

namespace neural {
class BackpropSnapshot;

enum DofContactType
{
  UNSUPPORTED = 0,
  NONE = 1,
  VERTEX = 2,
  FACE = 3,
  EDGE_A = 4,
  EDGE_B = 5,
  SELF_COLLISION = 7,
  SPHERE_TO_BOX = 8,
  BOX_TO_SPHERE = 9,
  SPHERE_A = 10,
  SPHERE_B = 11,
  SPHERE_TO_FACE = 12,
  FACE_TO_SPHERE = 13,
  SPHERE_TO_EDGE = 14,
  EDGE_TO_SPHERE = 15,
  SPHERE_TO_VERTEX = 16,
  VERTEX_TO_SPHERE = 17,
  PIPE_TO_SPHERE = 18,
  SPHERE_TO_PIPE = 19,
  PIPE_A = 20,
  PIPE_B = 21,
  PIPE_TO_VERTEX = 22,
  VERTEX_TO_PIPE = 23,
  PIPE_TO_EDGE = 24,
  EDGE_TO_PIPE = 25
};

struct EdgeData
{
  Eigen::Vector3d edgeAPos;
  Eigen::Vector3d edgeADir;
  Eigen::Vector3d edgeBPos;
  Eigen::Vector3d edgeBDir;
};

class DifferentiableContactConstraint
{

public:
  DifferentiableContactConstraint(
      std::shared_ptr<constraint::ConstraintBase> constraint,
      int index,
      double constraintForce);

  Eigen::Vector3d getContactWorldPosition();

  /// This returns the normal of the contact, pointing from A to B. This IS NOT
  /// NECESSARILY THE DIRECTION OF FORCE! If this contact constraint is a
  /// friction constraint, then this returns the normal of the contact, and
  /// getContactWorldForceDirection() returns the direction of the force.
  Eigen::Vector3d getContactWorldNormal();

  Eigen::Vector3d getContactWorldForceDirection();

  /// This computes the force, in world space exponential coordinates, that this
  /// contact generates
  Eigen::Vector6d getWorldForce();

  /// This returns the nature of the contact, whether it's a face-vertex, or a
  /// vertex-face, or an edge-edge, or something else. This is relevant because
  /// we need to know in order to accurately estimate how the contact position
  /// and normal will change as we perturb skeletons by small amounts.
  collision::ContactType getContactType();

  /// This figures out what type of contact this skeleton is involved in.
  DofContactType getDofContactType(dynamics::DegreeOfFreedom* dof);

  /// This analytically computes a column of the A_c matrix just for this
  /// skeleton.
  Eigen::VectorXd getConstraintForces(std::shared_ptr<dynamics::Skeleton> skel);

  /// This analytically computes a column of the A_c matrix for this set of
  /// skeletons.
  Eigen::VectorXd getConstraintForces(
      simulation::World* world, std::vector<std::string> skelNames);

  /// This analytically computes a column of the A_c matrix, for this contact
  /// constraint, across the whole world by concatenating the result for each
  /// skeleton together into a single vector.
  Eigen::VectorXd getConstraintForces(simulation::World* world);

  /// Returns the gradient of the contact position with respect to the
  /// specified dof of this skeleton
  Eigen::Vector3d getContactPositionGradient(dynamics::DegreeOfFreedom* dof);

  /// Returns the gradient of the contact normal with respect to the
  /// specified dof of this skeleton
  Eigen::Vector3d getContactNormalGradient(dynamics::DegreeOfFreedom* dof);

  /// Returns the gradient of the contact force with respect to the
  /// specified dof of this skeleton
  Eigen::Vector3d getContactForceGradient(dynamics::DegreeOfFreedom* dof);

  /// Returns the gradient of the full 6d twist force
  Eigen::Vector6d getContactWorldForceGradient(dynamics::DegreeOfFreedom* dof);

  /// Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6d getScrewAxisForPositionGradient(
      dynamics::DegreeOfFreedom* screwDof,
      dynamics::DegreeOfFreedom* rotateDof);

  /// Returns the gradient of the screw axis with respect to the rotate dof
  Eigen::Vector6d getScrewAxisForForceGradient(
      dynamics::DegreeOfFreedom* screwDof,
      dynamics::DegreeOfFreedom* rotateDof);

  /// This is the analytical Jacobian for the contact position
  math::LinearJacobian getContactPositionJacobian(
      std::shared_ptr<simulation::World> world);

  /// This is the analytical Jacobian for the contact position
  math::LinearJacobian getContactPositionJacobian(
      std::shared_ptr<dynamics::Skeleton> skel);

  /// This is the analytical Jacobian for the contact normal
  math::LinearJacobian getContactForceDirectionJacobian(
      std::shared_ptr<simulation::World> world);

  /// This is the analytical Jacobian for the contact normal
  math::LinearJacobian getContactForceDirectionJacobian(
      std::shared_ptr<dynamics::Skeleton> skel);

  /// This is the analytical Jacobian for the force (in exponential coordinates,
  /// in the world frame) generated by this contact. This is measuring how the
  /// force (in world space) changes as a result of moving the contact position
  /// and normal as a result of moving the joints of the skeletons in the world.
  math::Jacobian getContactForceJacobian(
      std::shared_ptr<simulation::World> world);

  /// This is the analytical Jacobian for the force (in exponential coordinates,
  /// in the world frame) generated by this contact. This is measuring how the
  /// force (in world space) changes as a result of moving the contact position
  /// and normal as a result of moving the joints of the skeletons passed in.
  math::Jacobian getContactForceJacobian(
      std::shared_ptr<dynamics::Skeleton> skel);

  /// This gets the constraint force for a given DOF
  double getConstraintForce(dynamics::DegreeOfFreedom* dof);

  /// This gets the gradient of constraint force at this joint with respect to
  /// another joint
  double getConstraintForceDerivative(
      dynamics::DegreeOfFreedom* dof, dynamics::DegreeOfFreedom* wrt);

  /// This returns an analytical Jacobian relating the skeletons that this
  /// contact touches.
  Eigen::MatrixXd getConstraintForcesJacobian(
      std::shared_ptr<simulation::World> world);

  /// This computes and returns the analytical Jacobian relating how changes in
  /// the positions of wrt's DOFs changes the constraint forces on skel.
  Eigen::MatrixXd getConstraintForcesJacobian(
      std::shared_ptr<dynamics::Skeleton> skel,
      std::shared_ptr<dynamics::Skeleton> wrt);

  /// This computes and returns the analytical Jacobian relating how changes in
  /// the positions of wrt's DOFs changes the constraint forces on all the
  /// skels.
  Eigen::MatrixXd getConstraintForcesJacobian(
      std::vector<std::shared_ptr<dynamics::Skeleton>> skels,
      std::shared_ptr<dynamics::Skeleton> wrt);

  /// This computes and returns the analytical Jacobian relating how changes in
  /// the positions of any of the DOFs changes the constraint forces on all the
  /// skels.
  Eigen::MatrixXd getConstraintForcesJacobian(
      std::vector<std::shared_ptr<dynamics::Skeleton>> skels);

  /// This returns the skeletons that this contact constraint interacts with.
  const std::vector<std::shared_ptr<dynamics::Skeleton>>& getSkeletons();

  /////////////////////////////////////////////////////////////////////////////////////
  // Testing
  /////////////////////////////////////////////////////////////////////////////////////

  /// The linear Jacobian for the contact position
  math::LinearJacobian bruteForceContactPositionJacobian(
      std::shared_ptr<simulation::World> world);

  /// The linear Jacobian for the contact normal
  math::LinearJacobian bruteForceContactForceDirectionJacobian(
      std::shared_ptr<simulation::World> world);

  /// This is the brute force version of getWorldForceJacobian()
  math::Jacobian bruteForceContactForceJacobian(
      std::shared_ptr<simulation::World> world);

  /// This is the brute force version of getConstraintForcesJacobian()
  Eigen::MatrixXd bruteForceConstraintForcesJacobian(
      std::shared_ptr<simulation::World> world);

  /// Just for testing: This analytically estimates the way the contact position
  /// will change if we perturb the `dofIndex`'th DOF of `skel` by `eps`.
  Eigen::Vector3d estimatePerturbedContactPosition(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, double eps);

  /// Just for testing: This analytically estimates the way the contact normal
  /// will change if we perturb the `dofIndex`'th DOF of `skel` by `eps`.
  Eigen::Vector3d estimatePerturbedContactNormal(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, double eps);

  /// Just for testing: This analytically estimates the way the contact normal
  /// will change if we perturb the `dofIndex`'th DOF of `skel` by `eps`.
  Eigen::Vector3d estimatePerturbedContactForceDirection(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, double eps);

  /// Just for testing: This analytically estimates how edges will move under a
  /// perturbation
  EdgeData estimatePerturbedEdges(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, double eps);

  /// Just for testing: returns the edges, if this is an edge-edge collision,
  /// otherwise 0s
  EdgeData getEdges();

  /// Just for testing: Returns the gradient of the edge data for this collision
  /// (0s if this isn't an edge-edge collision)
  EdgeData getEdgeGradient(dynamics::DegreeOfFreedom* dof);

  /// Just for testing: This analytically estimates how a screw axis will move
  /// when rotated by another screw.
  Eigen::Vector6d estimatePerturbedScrewAxisForPosition(
      dynamics::DegreeOfFreedom* axis,
      dynamics::DegreeOfFreedom* rotate,
      double eps);

  /// Just for testing: This analytically estimates how a screw axis will move
  /// when rotated by another screw.
  Eigen::Vector6d estimatePerturbedScrewAxisForForce(
      dynamics::DegreeOfFreedom* axis,
      dynamics::DegreeOfFreedom* rotate,
      double eps);

  /// Just for testing: This lets the world record what index this
  /// constraint is at, so that we can recover the analagous constraint from
  /// another forward pass for finite-differencing.
  void setOffsetIntoWorld(int offset, bool isUpperBoundConstraint);

  /// Just for testing: This runs a full timestep to get the way the contact
  /// position will change if we perturb the `dofIndex`'th DOF of `skel` by
  /// `eps`.
  Eigen::Vector3d bruteForcePerturbedContactPosition(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<dynamics::Skeleton> skel,
      int dofIndex,
      double eps);

  /// Just for testing: This runs a full timestep to get the way the contact
  /// normal will change if we perturb the `dofIndex`'th DOF of `skel` by
  /// `eps`.
  Eigen::Vector3d bruteForcePerturbedContactNormal(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<dynamics::Skeleton> skel,
      int dofIndex,
      double eps);

  /// Just for testing: This runs a full timestep to get the way the contact
  /// force direction will change if we perturb the `dofIndex`'th DOF of `skel`
  /// by `eps`.
  Eigen::Vector3d bruteForcePerturbedContactForceDirection(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<dynamics::Skeleton> skel,
      int dofIndex,
      double eps);

  /// Just for testing: This perturbs the world position of a skeleton to read a
  /// screw axis will move when rotated by another screw.
  Eigen::Vector6d bruteForceScrewAxisForPosition(
      dynamics::DegreeOfFreedom* axis,
      dynamics::DegreeOfFreedom* rotate,
      double eps);

  /// Just for testing: This perturbs the world position of a skeleton to read a
  /// screw axis will move when rotated by another screw.
  Eigen::Vector6d bruteForceScrewAxisForForce(
      dynamics::DegreeOfFreedom* axis,
      dynamics::DegreeOfFreedom* rotate,
      double eps);

  /// Just for testing: This perturbs the world position of a skeleton  to read
  /// how edges will move.
  EdgeData bruteForceEdges(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<dynamics::Skeleton> skel,
      int dofIndex,
      double eps);

  /// Return the index into the contact that this constraint represents. If it's
  /// >0, then this is a frictional constraint.
  int getIndexInConstraint();

  /// This returns the axis for the specified dof index
  static Eigen::Vector6d getWorldScrewAxisForPosition(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex);

  /// This returns the axis for the specified dof index
  static Eigen::Vector6d getWorldScrewAxisForPosition(
      dynamics::DegreeOfFreedom* dof);

  /// This returns the axis for the specified dof index which, when dotted with
  /// force, calculates the torque on this joint. THIS IS NOT THE SAME AS
  /// getWorldScrewAxisForPosition(), because of FreeJoints and BallJoints.
  static Eigen::Vector6d getWorldScrewAxisForForce(
      std::shared_ptr<dynamics::Skeleton> skel, int dofIndex);

  /// This returns the axis for the specified dof index which, when dotted with
  /// force, calculates the torque on this joint. THIS IS NOT THE SAME AS
  /// getWorldScrewAxisForPosition(), because of FreeJoints and BallJoints.
  static Eigen::Vector6d getWorldScrewAxisForForce(
      dynamics::DegreeOfFreedom* dof);

  /// This returns the constraint that's at our same location in the snapshot.
  /// This assumes that `mOffsetIntoWorld` and `mIsUpperBoundConstraint` are
  /// set.
  std::shared_ptr<DifferentiableContactConstraint> getPeerConstraint(
      std::shared_ptr<neural::BackpropSnapshot> snapshot);

  /// This returns 1.0 by default, 0.0 if this constraint doesn't effect the
  /// specified DOF, and -1.0 if the constraint effects this dof negatively.
  /// Pretty much only public for testing
  double getForceMultiple(dynamics::DegreeOfFreedom* dof);

public:
  /// Returns true if this dof moves this body node
  bool isParent(
      const dynamics::DegreeOfFreedom* dof, const dynamics::BodyNode* node);

  /// Returns true if this dof moves the other dof's screw axis
  bool isParent(
      const dynamics::DegreeOfFreedom* parent,
      const dynamics::DegreeOfFreedom* child);

protected:
  std::shared_ptr<constraint::ConstraintBase> mConstraint;
  std::shared_ptr<constraint::ContactConstraint> mContactConstraint;
  std::shared_ptr<collision::Contact> mContact;
  std::vector<std::string> mSkeletons;
  std::vector<Eigen::VectorXd> mSkeletonOriginalPositions;
  double mConstraintForce;

  int mIndex;

  /// This allows us to locate this constraint in the world arrays. This value
  /// is not guaranteed to be set, but must be set before calling any of the
  /// brute force methods!
  int mOffsetIntoWorld;
  /// This allows us to locate this constraint in the world arrays. If true,
  /// we're in the upper bound array. Otherwise, we're clamping. This value
  /// is not guaranteed to be set, but must be set before calling any of the
  /// brute force methods!
  bool mIsUpperBoundConstraint;
};
} // namespace neural
} // namespace dart

#endif