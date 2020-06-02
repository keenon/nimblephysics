#include "dart/neural/ClassicConstrainedGroupGradientMatrices.hpp"

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

//==============================================================================
ClassicConstrainedGroupGradientMatrices::
    ClassicConstrainedGroupGradientMatrices(
        constraint::ConstrainedGroup& group, double timeStep)
  : ConstrainedGroupGradientMatrices(group, timeStep)
{
}

/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// at each constraint's dimension. It gets called _after_ the system has
/// already applied a measurement impulse to that constraint dimension, and
/// measured some velocity changes. This must be called before
/// constructMatrices(), and must be called exactly once for each constraint's
/// dimension.
void ClassicConstrainedGroupGradientMatrices::measureConstraintImpulse(
    const constraint::ConstraintBasePtr& constraint,
    std::size_t constraintIndex)
{
  // For gradient comptutations: clear constraint impulses
  for (SkeletonPtr skel : constraint->getSkeletons())
  {
    skel->clearConstraintImpulses();
  }

  double* impulses = new double[constraint->getDimension()];
  for (std::size_t k = 0; k < constraint->getDimension(); ++k)
    impulses[k] = (k == constraintIndex) ? 1 : 0;
  constraint->applyImpulse(impulses);
  delete impulses;

  // For gradient computations: record the torque changes for each
  // skeleton for the unit impulse on this constraint.
  Eigen::VectorXd impulseTest = Eigen::VectorXd::Zero(mNumDOFs);
  for (SkeletonPtr skel : constraint->getSkeletons())
  {
    std::size_t offset = mSkeletonOffset.find(skel->getName())->second;
    std::size_t dofs = skel->getNumDofs();

    impulseTest.segment(offset, dofs) = skel->getConstraintForces() * mTimeStep;
  }

  mImpulseTests.push_back(impulseTest);

  for (SkeletonPtr skel : constraint->getSkeletons())
  {
    skel->clearConstraintImpulses();
  }
}

Eigen::MatrixXd
ClassicConstrainedGroupGradientMatrices::getProjectionIntoClampsMatrix()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();

  /*
  std::cout << "Computing P_c:" << std::endl;
  std::cout << "A_c size: " << A_c.size() << std::endl;
  std::cout << "A_ub size: " << A_ub.size() << std::endl;
  std::cout << "E size: " << E.size() << std::endl;
  std::cout << "M size: " << getMassMatrix().size() << std::endl;
  */

  /*
  if (A_ub.size() > 0 && E.size() > 0)
  {
    std::cout << "Doing P_c computation with A_ub and E" << std::endl;
    std::cout << "A_c: " << std::endl << A_c << std::endl;
    std::cout << "M: " << std::endl << getMassMatrix() << std::endl;
    std::cout << "E: " << std::endl << E << std::endl;
    std::cout << "A_ub: " << std::endl << A_ub << std::endl;
    Eigen::MatrixXd A_cA_ub = (A_c + A_ub * E);
    Eigen::MatrixXd A_cA_ubInv
        = (A_c + A_ub * E).completeOrthogonalDecomposition().pseudoInverse();
    Eigen::MatrixXd A_cInv
        = A_c.completeOrthogonalDecomposition().pseudoInverse();
    std::cout << "A_cInv: " << std::endl << A_cInv << std::endl;
    std::cout << "A_c + A_ub*E: " << std::endl << A_cA_ub << std::endl;
    std::cout << "(A_c + A_ub*E)Inv: " << std::endl << A_cA_ubInv << std::endl;
    return (1.0 / dt) * A_cA_ubInv.eval() * getMassMatrix()
           * A_cInv.eval().transpose() * A_c.transpose();
  }
  else
  {
  */
  Eigen::MatrixXd A_cInv
      = A_c.completeOrthogonalDecomposition().pseudoInverse();
  /*
  std::cout << "Doing P_c computation without A_ub and E" << std::endl;
  std::cout << "A_c: " << A_c << std::endl;
  std::cout << "M: " << getMassMatrix() << std::endl;
  std::cout << "A_cInv: " << A_cInv << std::endl;
  */
  return (1.0 / mTimeStep) * A_cInv.eval() * getMassMatrix()
         * A_cInv.eval().transpose() * A_c.eval().transpose();
  //}
}

Eigen::MatrixXd ClassicConstrainedGroupGradientMatrices::getForceVelJacobian()
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

Eigen::MatrixXd ClassicConstrainedGroupGradientMatrices::getVelVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(
      mNumDOFs, mNumDOFs); // TODO(keenon): B needs to be set properly.
  if (A_ub.size() > 0 && E.size() > 0)
  {
    return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
            - mTimeStep * Minv * (A_c + A_ub * E) * P_c)
           * B;
  }
  else
  {
    return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
            - mTimeStep * Minv * (A_c + A_ub * E) * P_c)
           * B;
  }
}

} // namespace neural
} // namespace dart