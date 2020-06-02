
#include "dart/neural/ClassicBackpropSnapshot.hpp"

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

//==============================================================================
ClassicBackpropSnapshot::ClassicBackpropSnapshot(
    simulation::WorldPtr world,
    Eigen::VectorXd forwardPassPosition,
    Eigen::VectorXd forwardPassVelocity,
    Eigen::VectorXd forwardPassTorques)
  : BackpropSnapshot(
      world, forwardPassPosition, forwardPassVelocity, forwardPassTorques)
{
}

Eigen::MatrixXd ClassicBackpropSnapshot::getProjectionIntoClampsMatrix()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  */

  Eigen::MatrixXd constraintForceToImpliedTorques = A_c + (A_ub * E);
  Eigen::MatrixXd forceToVel = A_c.eval().transpose() * getInvMassMatrix()
                               * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  /*
  std::cout << "forceToVel: " << std::endl << forceToVel << std::endl;
  std::cout << "forceToVel^-1: " << std::endl << velToForce << std::endl;
  std::cout << "mTimeStep: " << mTimeStep << std::endl;
  */
  return (1.0 / mTimeStep) * velToForce * A_c.transpose();
}

Eigen::MatrixXd ClassicBackpropSnapshot::getForceVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();

  if (A_ub.size() > 0 && E.size() > 0)
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
  }
  else
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * A_c * P_c * Minv);
  }
}

Eigen::MatrixXd ClassicBackpropSnapshot::getVelVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(
      mNumDOFs, mNumDOFs); // TODO(keenon): B needs to be set properly.
  Eigen::MatrixXd parts1 = A_c + A_ub * E;
  Eigen::MatrixXd parts2 = mTimeStep * Minv * parts1 * P_c;
  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  std::cout << "P_c: " << std::endl << P_c << std::endl;
  std::cout << "Minv: " << std::endl << Minv << std::endl;
  std::cout << "mTimestep: " << mTimeStep << std::endl;
  std::cout << "A_c + A_ub * E: " << std::endl << parts1 << std::endl;
  std::cout << "mTimestep * Minv * (A_c + A_ub * E) * P_c: " << std::endl
            << parts2 << std::endl;
            */
  return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) - parts2) * B;
}

} // namespace neural
} // namespace dart