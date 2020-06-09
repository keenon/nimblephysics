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

//==============================================================================
ConstrainedGroupGradientMatrices::~ConstrainedGroupGradientMatrices()
{
  // Do nothing, for now
}

//==============================================================================
/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// at each constraint. This must be called before constructMatrices(), and
/// must be called exactly once for each constraint.
void ConstrainedGroupGradientMatrices::registerConstraint(
    const std::shared_ptr<constraint::ConstraintBase>& constraint)
{
  double coeff = constraint->getCoefficientOfRestitution();
  mRestitutionCoeffs.push_back(coeff);
  // Pad with 0s, since these values always apply to the first dimension of the
  // constraint even if there are more (ex. friction) dimensions
  for (std::size_t i = 1; i < constraint->getDimension(); i++)
  {
    mRestitutionCoeffs.push_back(0);
  }
}

//==============================================================================
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

//==============================================================================
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
  int* bouncingIndex = new int[mNumConstraintDim];

  int numClamping = 0;
  int numUpperBound = 0;
  int numBouncing = 0;
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
      if (mRestitutionCoeffs[j] > 0)
      {
        bouncingIndex[j] = numBouncing;
        numBouncing++;
      }
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
  mBounceDiagonals = Eigen::VectorXd(numClamping);
  mBouncingConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numBouncing);
  mRestitutionDiagonals = Eigen::VectorXd(numBouncing);

  // Copy impulse tests into the matrices
  for (size_t j = 0; j < mNumConstraintDim; j++)
  {
    if (contactConstraintMappings(j) == neural::ConstraintMapping::CLAMPING)
    {
      assert(numClamping > clampingIndex[j]);
      mClampingConstraintMatrix.col(clampingIndex[j]) = mImpulseTests[j];
      mMassedClampingConstraintMatrix.col(clampingIndex[j])
          = mMassedImpulseTests[j];
      mBounceDiagonals(clampingIndex[j]) = 1 + mRestitutionCoeffs[j];
      if (mRestitutionCoeffs[j] > 0)
      {
        mBouncingConstraintMatrix.col(bouncingIndex[j]) = mImpulseTests[j];
        mRestitutionDiagonals(bouncingIndex[j]) = mRestitutionCoeffs[j];
      }
    }
    else if (contactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
    {
      assert(numUpperBound > upperBoundIndex[j]);
      mUpperBoundConstraintMatrix.col(upperBoundIndex[j]) = mImpulseTests[j];
      mMassedUpperBoundConstraintMatrix.col(upperBoundIndex[j])
          = mMassedImpulseTests[j];
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

//==============================================================================
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

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();
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
  return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) - parts2);
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getPosPosJacobian()
{
  Eigen::MatrixXd A_b = getBouncingConstraintMatrix();

  // Check if we don't have any bounces this frame, and if so this is just the
  // identity
  if (A_b.size() == 0)
  {
    return Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs);
  }

  // Construct the W matrix we'll need to use to solve for our closest approx
  Eigen::MatrixXd W = Eigen::MatrixXd(A_b.cols(), A_b.rows() * A_b.rows());
  for (std::size_t i = 0; i < A_b.cols(); i++)
  {
    Eigen::VectorXd a_i = A_b.col(i);
    for (std::size_t j = 0; j < A_b.rows(); j++)
    {
      W.block(j * A_b.rows(), i, A_b.rows(), 1) = a_i(j) * a_i;
    }
  }

  // We want to center the solution around the identity matrix, and find the
  // least-squares deviation along the diagonals that gets us there.
  Eigen::VectorXd center = Eigen::VectorXd::Zero(mNumDOFs * mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    center((i * mNumDOFs) + i) = 1;
  }

  // Solve the linear system
  Eigen::VectorXd q
      = center
        - W.transpose().completeOrthogonalDecomposition().solve(
            getRestitutionDiagonals() + (W.eval().transpose() * center));

  // Recover X from the q vector
  Eigen::MatrixXd X = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
  }

  return X;
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelPosJacobian()
{
  return mTimeStep * getPosPosJacobian();
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getMassMatrix()
{
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  massMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t skelDOF = mSkeletons[i]->getNumDofs();
    massMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = mSkeletons[i]->getMassMatrix();
    cursor += skelDOF;
  }
  return massMatrix;
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getInvMassMatrix()
{
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  invMassMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t skelDOF = mSkeletons[i]->getNumDofs();
    invMassMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = mSkeletons[i]->getInvMassMatrix();
    cursor += skelDOF;
  }
  return invMassMatrix;
}

//==============================================================================
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::getProjectionIntoClampsMatrix()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix();
  Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();
  return (1.0 / mTimeStep) * velToForce * bounce * A_c.transpose();
}

//==============================================================================
/// This replaces x with the result of M*x in place, without explicitly forming
/// M
void ConstrainedGroupGradientMatrices::implicitMultiplyByMassMatrix(
    Eigen::VectorXd& x)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    x.segment(cursor, dofs)
        = mSkeletons[i]->multiplyByImplicitMassMatrix(x.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
/// This replaces x with the result of Minv*x in place, without explicitly
/// forming Minv
void ConstrainedGroupGradientMatrices::implicitMultiplyByInvMassMatrix(
    Eigen::VectorXd& x)
{
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t dofs = mSkeletons[i]->getNumDofs();
    x.segment(cursor, dofs) = mSkeletons[i]->multiplyByImplicitInvMassMatrix(
        x.segment(cursor, dofs));
    cursor += dofs;
  }
}

//==============================================================================
/// Multiply by the vel-vel jacobian, without forming it explicitly
void ConstrainedGroupGradientMatrices::backprop(
    LossGradient& thisTimestepLoss, const LossGradient& nextTimestepLoss)
{
  /*
  The forward computation graph looks like this:

  p_t ---------------------> p_t+1 ---->
                               ^
                               |
  v_t ------> v_t+1 -----------+------->
                ^
                |
  f_t ----------+
  */

  // Compute loss wrt p_t:
  if (mBouncingConstraintMatrix.size() == 0)
  {
    thisTimestepLoss.lossWrtPosition = nextTimestepLoss.lossWrtPosition;
  }
  else
  {
    Eigen::MatrixXd X = getPosPosJacobian().transpose();
    thisTimestepLoss.lossWrtPosition = X * nextTimestepLoss.lossWrtPosition;
  }

  // Compute loss wrt v_t+1: This is just p_t * mTimeStep
  thisTimestepLoss.lossWrtVelocity
      = mTimeStep * thisTimestepLoss.lossWrtPosition;

  // Compute common precursor to loss wrt v_t and f_t:
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix();
  Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd forceToVelTranspose
      = constraintForceToImpliedTorques.transpose() * A_c;
  auto velToForceTranspose
      = forceToVelTranspose.completeOrthogonalDecomposition();

  Eigen::VectorXd z = velToForceTranspose.solve(
      constraintForceToImpliedTorques.transpose()
      * nextTimestepLoss.lossWrtVelocity);
  z = mBounceDiagonals.cwiseProduct(z);

  // Compute loss wrt v_t:
  thisTimestepLoss.lossWrtVelocity = nextTimestepLoss.lossWrtVelocity - A_c * z;

  // Compute loss wrt f_t:
  thisTimestepLoss.lossWrtTorque = nextTimestepLoss.lossWrtVelocity;
  implicitMultiplyByInvMassMatrix(thisTimestepLoss.lossWrtTorque);
  thisTimestepLoss.lossWrtTorque -= V_c * z;
  thisTimestepLoss.lossWrtTorque *= mTimeStep;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getClampingConstraintMatrix() const
{
  return mClampingConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getMassedClampingConstraintMatrix() const
{
  return mMassedClampingConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getUpperBoundConstraintMatrix() const
{
  return mUpperBoundConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getMassedUpperBoundConstraintMatrix() const
{
  return mMassedUpperBoundConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getUpperBoundMappingMatrix() const
{
  return mUpperBoundMappingMatrix;
}

//==============================================================================
const Eigen::MatrixXd&
ConstrainedGroupGradientMatrices::getBouncingConstraintMatrix() const
{
  return mBouncingConstraintMatrix;
}

//==============================================================================
/// These was the mX() vector used to construct this. Pretty much only here
/// for testing.
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getContactConstraintImpluses() const
{
  return mContactConstraintImpulses;
}

//==============================================================================
const Eigen::VectorXd& ConstrainedGroupGradientMatrices::getBounceDiagonals()
    const
{
  return mBounceDiagonals;
}

//==============================================================================
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getRestitutionDiagonals() const
{
  return mRestitutionDiagonals;
}

//==============================================================================
/// These was the fIndex() vector used to construct this. Pretty much only
/// here for testing.
const Eigen::VectorXi&
ConstrainedGroupGradientMatrices::getContactConstraintMappings() const
{
  return mContactConstraintMappings;
}

//==============================================================================
std::size_t ConstrainedGroupGradientMatrices::getNumDOFs() const
{
  return mNumDOFs;
}

//==============================================================================
std::size_t ConstrainedGroupGradientMatrices::getNumConstraintDim() const
{
  return mNumConstraintDim;
}

//==============================================================================
const std::vector<std::shared_ptr<dynamics::Skeleton>>&
ConstrainedGroupGradientMatrices::getSkeletons() const
{
  return mSkeletons;
}

} // namespace neural
} // namespace dart