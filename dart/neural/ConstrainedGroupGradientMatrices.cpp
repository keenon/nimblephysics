#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

//==============================================================================
ConstrainedGroupGradientMatrices::ConstrainedGroupGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep)
{
  mTimeStep = timeStep;

  // Collect all the skeletons attached to the constraints

  mNumDOFs = 0;
  mNumConstraintDim = 0;
  for (std::size_t i = 0; i < group.getNumConstraints(); i++)
  {
    constraint::ConstraintBasePtr constraint = group.getConstraint(i);
    mNumConstraintDim += constraint->getDimension();
    std::vector<SkeletonPtr> constraintSkeletons = constraint->getSkeletons();
    for (std::size_t j = 0; j < constraintSkeletons.size(); j++)
    {
      SkeletonPtr skel = constraintSkeletons[j];

      // Only add this skeleton to our list if it's not already present

      if (std::find(mSkeletons.begin(), mSkeletons.end(), skel)
          == mSkeletons.end())
      {
        mSkeletons.push_back(skel);
        mSkeletonOffset.insert(std::make_pair(skel->getName(), mNumDOFs));
        mNumDOFs += skel->getNumDofs();
      }
    }
  }

  mImpulseTests.reserve(mNumConstraintDim);
}

ConstrainedGroupGradientMatrices::~ConstrainedGroupGradientMatrices()
{
  // Do nothing, for now
}

/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// at each constraint's dimension. It gets called _after_ the system has
/// already applied a measurement impulse to that constraint dimension, and
/// measured some velocity changes. This must be called before
/// constructMatrices(), and must be called exactly once for each constraint's
/// dimension.
void ConstrainedGroupGradientMatrices::measureConstraintImpulse(
    const constraint::ConstraintBasePtr& constraint,
    std::size_t constraintIndex)
{
  // For gradient computations: record the velocity changes for each
  // skeleton for the unit impulse on this constraint. We are guaranteed that
  // this is called after the outer code has already applied the impulse, so we
  // can just read off velocity changes.
  Eigen::VectorXd massedImpulseTest = Eigen::VectorXd::Zero(mNumDOFs);
  for (SkeletonPtr skel : constraint->getSkeletons())
  {
    std::size_t offset = mSkeletonOffset.find(skel->getName())->second;
    std::size_t dofs = skel->getNumDofs();

    massedImpulseTest.segment(offset, dofs) = skel->getVelocityChanges();
  }
  mMassedImpulseTests.push_back(massedImpulseTest);

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
ConstrainedGroupGradientMatrices::getProjectionIntoClampsMatrix()
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

Eigen::MatrixXd ConstrainedGroupGradientMatrices::getForceVelJacobian()
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

Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelVelJacobian()
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

/// This creates a block-diagonal matrix that concatenates the mass matrices
/// of the skeletons that are part of this ConstrainedGroup.
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getMassMatrix() const
{
  if (mSkeletons.size() == 1)
    return mSkeletons[0]->getMassMatrix();
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getNumDofs();
    massMatrix.block(offset, offset, dims, dims)
        = mSkeletons[i]->getMassMatrix();
    offset += dims;
  }
  return massMatrix;
}

/// This creates a block-diagonal matrix that concatenates the inverse mass
/// matrices of the skeletons that are part of this ConstrainedGroup.
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getInvMassMatrix() const
{
  if (mSkeletons.size() == 1)
    return mSkeletons[0]->getMassMatrix();
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dims = mSkeletons[i]->getNumDofs();
    invMassMatrix.block(offset, offset, dims, dims)
        = mSkeletons[i]->getMassMatrix();
    offset += dims;
  }
  return invMassMatrix;
}

/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// after the LCP has run, with the result from the LCP solver. This can only
/// be called once, and after this is called you cannot call
/// measureConstraintImpulse() again!
void ConstrainedGroupGradientMatrices::constructMatrices(
    Eigen::VectorXd mX,
    Eigen::VectorXd hi,
    Eigen::VectorXd lo,
    Eigen::VectorXi fIndex)
{
  mContactConstraintImpulses = mX;
  mContactConstraintMappings = fIndex;
  // Group the constraints based on their solution values into three buckets:
  //
  // - "Clamping": These are constraints that have non-zero constraint forces
  //               being applied, which means they have a zero constraint
  //               velocity, and aren't dependent on any other forces (ie have
  //               an fIndex = -1)
  // - "Upper Bound": These are sliding-friction constraints that have hit their
  //                  upper OR lower bounds, and so are tied to the strength of
  //                  the corresponding force (fIndex != -1)
  // - "Not Clamping": These are constraints with a zero constraint force. These
  //                   don't actually get used anywhere in the gradient
  //                   computation, and so can be safely ignored.

  // Declare a shared array to re-use for mapping info for each skeleton.
  // Semantics are as follows:
  // - If mappings[j] >= 0, constraint "j" is "Upper Bound".
  // - If mappings[j] == CLAMPING, constraint "j" is "Clamping".
  // - If mappings[j] == NOT_CLAMPING, constraint "j" is "Not Clamping".
  // - If mappings[j] == IRRELEVANT, constraint "j" is doesn't effect this
  //   skeleton, and so can be safely ignored.

  Eigen::VectorXi contactConstraintMappings
      = Eigen::VectorXi(mNumConstraintDim);
  int* clampingIndex = new int[mNumConstraintDim];
  int* upperBoundIndex = new int[mNumConstraintDim];

  int numClamping = 0;
  int numUpperBound = 0;
  // Fill in mappings[] with the correct values, overwriting previous data
  for (std::size_t j = 0; j < mNumConstraintDim; j++)
  {
    // If the Eigen::VectorXd representing the impulse test is of length 0,
    // that means that constraint "j" doesn't effect skeleton "i".
    // TODO(keenon): This has gone away in the format where we're doing this
    // across a whole constraint group.
    /*
    if (mSkeletonsImpulseTests[i][j].size() == 0
        || mSkeletonsImpulseTests[i][j].isZero())
    {
      contactConstraintMappings(j) = neural::ConstraintMapping::IRRELEVANT;
      continue;
    }
    */
    const double constraintForce = mX(j);

    // If constraintForce is zero, this means "j" is in "Not Clamping"
    if (std::abs(constraintForce) < 1e-9)
    {
      contactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
      continue;
    }

    double upperBound = hi(j);
    double lowerBound = lo(j);
    const int fIndexPointer = fIndex(j);
    if (fIndexPointer != -1)
    {
      upperBound *= mX(fIndexPointer);
      lowerBound *= mX(fIndexPointer);
    }

    // This means "j" is in "Clamping"
    if (mX(j) > lowerBound && mX(j) < upperBound)
    {
      contactConstraintMappings(j) = neural::ConstraintMapping::CLAMPING;
      clampingIndex[j] = numClamping;
      numClamping++;
    }
    // Otherwise, if fIndex != -1, "j" is in "Upper Bound"
    // Note, this could also mean "j" is at it's lower bound, but we call the
    // group of all "j"'s that have reached their dependent bound "Upper
    // Bound"
    else if (fIndexPointer != -1)
    {
      /*
      std::cout << "Listing " << j << " as UB: mX=" << mX(j)
                << ", fIndex=" << fIndex << ", mX(fIndex)=" << mX(fIndex)
                << ", hiBackup=" << hiGradientBackup(j)
                << ", loBackup=" << loGradientBackup(j)
                << ", upperBound=" << upperBound
                << ", lowerBound=" << lowerBound << std::endl;
      */
      contactConstraintMappings(j) = fIndexPointer;
      upperBoundIndex[j] = numUpperBound;
      numUpperBound++;
    }
    // If fIndex == -1, and we're at a bound, then we're actually "Not
    // Clamping", cause the velocity can change freely without the force
    // changing to compensate.
    else
    {
      contactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
    }
  }

  // Create the matrices we want to pass along for this skeleton:
  mClampingConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numClamping);
  mMassedClampingConstraintMatrix
      = Eigen::MatrixXd::Zero(mNumDOFs, numClamping);
  mUpperBoundConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numUpperBound);
  mMassedUpperBoundConstraintMatrix
      = Eigen::MatrixXd::Zero(mNumDOFs, numUpperBound);
  mUpperBoundMappingMatrix = Eigen::MatrixXd::Zero(numUpperBound, numClamping);

  // Copy impulse tests into the matrices
  for (std::size_t i = 0; i < mSkeletons.size(); ++i)
  {
    // Copy values into our new matrices
    for (size_t j = 0; j < mNumConstraintDim; j++)
    {
      if (contactConstraintMappings(j) == neural::ConstraintMapping::CLAMPING)
      {
        assert(numClamping > clampingIndex[j]);
        mClampingConstraintMatrix.col(clampingIndex[j]) = mImpulseTests[j];
        mMassedClampingConstraintMatrix.col(clampingIndex[j])
            = mMassedImpulseTests[j];
      }
      else if (contactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
      {
        assert(numUpperBound > upperBoundIndex[j]);
        mUpperBoundConstraintMatrix.col(upperBoundIndex[j]) = mImpulseTests[j];
        mMassedUpperBoundConstraintMatrix.col(upperBoundIndex[j])
            = mMassedImpulseTests[j];
      }
    }
  }

  // Set up mUpperboundMappingMatrix (aka E)
  for (size_t j = 0; j < mNumConstraintDim; j++)
  {
    if (contactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
    {
      const int fIndexPointer = contactConstraintMappings(j);
      const double upperBound = mX(fIndexPointer) * hi(j);
      const double lowerBound = mX(fIndexPointer) * lo(j);

      // If we're clamped at the upper bound
      if (std::abs(mX(j) - upperBound) < std::abs(mX(j) - lowerBound))
      {
        if (std::abs(mX(j) - upperBound) > 1e-5)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j): " << hi(j) << std::endl;
          std::cout << "mLo(j): " << lo(j) << std::endl;
          std::cout << "mX(j): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(std::abs(mX(j) - upperBound) < 1e-5);
        mUpperBoundMappingMatrix(
            upperBoundIndex[j], clampingIndex[fIndexPointer])
            = hi(j);
      }
      // If we're clamped at the lower bound
      else
      {
        if (std::abs(mX(j) - lowerBound) > 1e-5)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j): " << hi(j) << std::endl;
          std::cout << "mLo(j): " << lo(j) << std::endl;
          std::cout << "mX(j): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(std::abs(mX(j) - lowerBound) < 1e-5);
        mUpperBoundMappingMatrix(
            upperBoundIndex[j], clampingIndex[fIndexPointer])
            = lo(j);
      }
    }
  }
}

const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getClampingConstraintMatrix() const
{
  return mClampingConstraintMatrix;
}

const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getMassedClampingConstraintMatrix() const
{
  return mMassedClampingConstraintMatrix;
}

const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getUpperBoundConstraintMatrix() const
{
  return mUpperBoundConstraintMatrix;
}

const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getMassedUpperBoundConstraintMatrix() const
{
  return mMassedUpperBoundConstraintMatrix;
}

const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getUpperBoundMappingMatrix() const
{
  return mUpperBoundMappingMatrix;
}

/// These was the mX() vector used to construct this. Pretty much only here
/// for testing.
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getContactConstraintImpluses() const
{
  return mContactConstraintImpulses;
}

/// These was the fIndex() vector used to construct this. Pretty much only
/// here for testing.
const Eigen::VectorXi&
ConstrainedGroupGradientMatrices::getContactConstraintMappings() const
{
  return mContactConstraintMappings;
}

std::size_t ConstrainedGroupGradientMatrices::getNumDOFs() const
{
  return mNumDOFs;
}

std::size_t ConstrainedGroupGradientMatrices::getNumConstraintDim() const
{
  return mNumConstraintDim;
}

const std::vector<std::shared_ptr<dynamics::Skeleton>>&
ConstrainedGroupGradientMatrices::getSkeletons() const
{
  return mSkeletons;
}

} // namespace neural
} // namespace dart