#ifndef DART_NEURAL_SNAPSHOT_HPP_
#define DART_NEURAL_SNAPSHOT_HPP_

#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

class BackpropSnapshot
{
public:
  /// This saves a snapshot from a forward pass, with all the info we need in
  /// order to efficiently compute a backwards pass. Crucially, the positions
  /// must all be snapshots from before the timestep, yet this constructor must
  /// be called after the timestep.
  BackpropSnapshot(
      simulation::WorldPtr world,
      Eigen::VectorXd forwardPassPosition,
      Eigen::VectorXd forwardPassVelocity,
      Eigen::VectorXd forwardPassTorques);

  /// This computes and returns the whole vel-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getVelVelJacobian();

  /// This computes and returns the whole force-vel jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getForceVelJacobian();

  /// This computes and returns the whole pos-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getPosPosJacobian();

  /// This computes and returns the whole vel-pos jacobian. For backprop, you
  /// don't actually need this matrix, you can compute backprop directly. This
  /// is here if you want access to the full Jacobian for some reason.
  Eigen::MatrixXd getVelPosJacobian();

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them.
  Eigen::VectorXd getForwardPassPosition();

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them.
  Eigen::VectorXd getForwardPassVelocity();

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass.
  Eigen::VectorXd getForwardPassTorques();

  /////////////////////////////////////////////////////////////////////////////
  /// Just public for testing
  /////////////////////////////////////////////////////////////////////////////

  /// This returns the A_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getClampingConstraintMatrix();

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getMassedClampingConstraintMatrix();

  /// This returns the A_ub matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getUpperBoundConstraintMatrix();

  /// This returns the V_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getMassedUpperBoundConstraintMatrix();

  /// This returns the E matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getUpperBoundMappingMatrix();

  /// This returns the B matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getBouncingConstraintMatrix();

  /// This returns the mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton mass matrices.
  Eigen::MatrixXd getMassMatrix();

  /// This returns the inverse mass matrix for the whole world, a block diagonal
  /// concatenation of the skeleton inverse mass matrices.
  Eigen::MatrixXd getInvMassMatrix();

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceVelVelJacobian();

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceForceVelJacobian();

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferencePosPosJacobian(std::size_t subdivisions = 20);

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXd finiteDifferenceVelPosJacobian(std::size_t subdivisions = 20);

  /// This returns the P_c matrix. You shouldn't ever need this matrix, it's
  /// just here to enable testing.
  Eigen::MatrixXd getProjectionIntoClampsMatrix();

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

  ~BackpropSnapshot();

protected:
  /// A handle to the world that we're taking a snapshot of. This world can
  /// change configurations underneath us, but this whole neural.* package
  /// assumes that no skeletons are added or removed once training begins.
  simulation::WorldPtr mWorld;

  /// These are the skeletons we're interested in.
  std::vector<dynamics::SkeletonPtr> mSkeletons;

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

  /// These are the gradient constraint matrices from the LCP solver
  std::vector<std::shared_ptr<ConstrainedGroupGradientMatrices>>
      mGradientMatrices;

  /// The position of all the DOFs of the world, when this snapshot was created
  Eigen::VectorXd mForwardPassPosition;

  /// The velocities of all the DOFs of the world, when this snapshot was
  /// created
  Eigen::VectorXd mForwardPassVelocity;

  /// The torques on all the DOFs of the world, when this snapshot was created
  Eigen::VectorXd mForwardPassTorques;

private:
  enum MatrixToAssemble
  {
    CLAMPING,
    MASSED_CLAMPING,
    UPPER_BOUND,
    MASSED_UPPER_BOUND,
    BOUNCING
  };

  Eigen::MatrixXd assembleMatrix(MatrixToAssemble whichMatrix);
};

using BackpropSnapshotPtr = std::shared_ptr<BackpropSnapshot>;

} // namespace neural
} // namespace dart

#endif