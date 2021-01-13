#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

#define CLAMPING_THRESHOLD 1e-4

namespace dart {
namespace neural {

//==============================================================================
ConstrainedGroupGradientMatrices::ConstrainedGroupGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep)
  : mFinalized(false)
{
  mTimeStep = timeStep;
  assert(mClampingConstraints.size() == 0);
  assert(mUpperBoundConstraints.size() == 0);

  // Collect all the skeletons attached to the constraints

  std::vector<SkeletonPtr> skeletons;

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

      if (std::find(mSkeletons.begin(), mSkeletons.end(), skel->getName())
          == mSkeletons.end())
      {
        mSkeletons.push_back(skel->getName());
        mSkeletonOffset[skel->getName()] = mNumDOFs;
        mNumDOFs += skel->getNumDofs();
        skeletons.push_back(skel);
      }
    }
  }

  mMassedImpulseTests.reserve(mNumConstraintDim);

  // Cache an inverse mass matrix for later use
  mMinv = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  mCoriolisAndGravityForces = Eigen::VectorXd(mNumDOFs);
  mPreStepTorques = Eigen::VectorXd(mNumDOFs);
  mPreStepVelocities = Eigen::VectorXd(mNumDOFs);
  mPreLCPVelocities = Eigen::VectorXd(mNumDOFs);
  int cursor = 0;
  for (auto skel : skeletons)
  {
    int dofs = skel->getNumDofs();
    mMinv.block(cursor, cursor, dofs, dofs) = skel->getInvMassMatrix();
    // TODO: does this break everything?
    mCoriolisAndGravityForces.segment(cursor, dofs)
        = skel->getCoriolisAndGravityForces() - skel->getExternalForces();
    mPreStepTorques.segment(cursor, dofs) = skel->getForces();
    mPreLCPVelocities.segment(cursor, dofs) = skel->getVelocities();
    mPreStepVelocities.segment(cursor, dofs)
        = skel->getVelocities() - (timeStep * skel->getAccelerations());
    cursor += dofs;
  }
}

//==============================================================================
ConstrainedGroupGradientMatrices::ConstrainedGroupGradientMatrices(
    int numDofs, int numConstraintDim, double timeStep)
{
  mNumDOFs = numDofs;
  mNumConstraintDim = numConstraintDim;
  mTimeStep = timeStep;

  mMassedImpulseTests.reserve(mNumConstraintDim);
}

//==============================================================================
/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// at each constraint. This must be called before constructMatrices(), and
/// must be called exactly once for each constraint.
void ConstrainedGroupGradientMatrices::registerConstraint(
    const std::shared_ptr<constraint::ConstraintBase>& constraint)
{
  double coeff = constraint->getCoefficientOfRestitution();
  double penetrationHack = constraint->getPenetrationCorrectionVelocity();
  mRestitutionCoeffs.push_back(coeff);
  mPenetrationCorrectionVelocities.push_back(penetrationHack);
  mConstraints.push_back(constraint);
  mConstraintIndices.push_back(0);
  // Pad with 0s, since these values always apply to the first dimension of the
  // constraint even if there are more (ex. friction) dimensions
  for (std::size_t i = 1; i < constraint->getDimension(); i++)
  {
    mRestitutionCoeffs.push_back(0);
    mPenetrationCorrectionVelocities.push_back(0);
    // Put a reference to this constraint at each of its dimensions, to make the
    // later logistics easier
    mConstraints.push_back(constraint);
    mConstraintIndices.push_back(i);
  }
}

//==============================================================================
void ConstrainedGroupGradientMatrices::mockRegisterConstraint(
    double restitutionCoeff, double penetrationHackVel)
{
  mRestitutionCoeffs.push_back(restitutionCoeff);
  mPenetrationCorrectionVelocities.push_back(penetrationHackVel);
  mConstraints.push_back(nullptr);
  mConstraintIndices.push_back(0);
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
    std::size_t offset = mSkeletonOffset[skel->getName()];
    std::size_t dofs = skel->getNumDofs();

    massedImpulseTest.segment(offset, dofs) = skel->getVelocityChanges();
  }
  mMassedImpulseTests.push_back(massedImpulseTest);
}

//==============================================================================
void ConstrainedGroupGradientMatrices::mockMeasureConstraintImpulse(
    Eigen::VectorXd impulseTest, Eigen::VectorXd massedImpulseTest)
{
  mMassedImpulseTests.push_back(massedImpulseTest);
}

//==============================================================================
void ConstrainedGroupGradientMatrices::registerLCPResults(
    Eigen::VectorXd X,
    Eigen::VectorXd hi,
    Eigen::VectorXd lo,
    Eigen::VectorXi fIndex,
    Eigen::VectorXd b,
    Eigen::VectorXd aColNorms,
    Eigen::MatrixXd A)
{
  mX = X;
  mHi = hi;
  mLo = lo;
  mFIndex = fIndex;
  mB = b;
  mAColNorms = aColNorms;
  mA = A;
}

//==============================================================================
/// This returns true if the proposed mX is consistent with our recorded LCP
/// construction
bool ConstrainedGroupGradientMatrices::isSolutionValid(
    const Eigen::VectorXd& mX)
{
  Eigen::VectorXd v = mA * mX - mB;
  for (int i = 0; i < mX.size(); i++)
  {
    double upperLimit = mHi(i);
    double lowerLimit = mLo(i);
    if (mFIndex(i) != -1)
    {
      upperLimit *= mX(mFIndex(i));
      lowerLimit *= mX(mFIndex(i));
    }

    const double tol = 1e-3;

    /// Solves constriant impulses for a constrained group. The LCP formulation
    /// setting that this function solve is A*x = b + w where each x[i], w[i]
    /// satisfies one of
    ///   (1) x = lo, w >= 0
    ///   (2) x = hi, w <= 0
    ///   (3) lo < x < hi, w = 0

    // If force has a zero bound, and we're at a zero bound (this is common with
    // friction being upper-bounded by a near-zero normal force) then allow
    // velocity in either direction.
    if (std::abs(lowerLimit) < tol && std::abs(upperLimit) < tol
        && std::abs(mX(i)) < tol)
    {
      // This is always allowed
    }
    // If force is at the lower bound, velocity must be >= 0
    else if (std::abs(mX(i) - lowerLimit) < tol)
    {
      if (v(i) < -tol)
        return false;
    }
    // If force is at the upper bound, velocity must be <= 0
    else if (std::abs(mX(i) - upperLimit) < tol)
    {
      if (v(i) > tol)
        return false;
    }
    // If force is within bounds, then velocity must be zero
    else if (mX(i) > lowerLimit && mX(i) < upperLimit)
    {
      if (std::abs(v(i)) > tol)
        return false;
    }
    // If force is out of bounds, we're always illegal
    else
    {
      return false;
    }
  }
  // If we make it here, the solution is fine
  return true;
}

//==============================================================================
/// If possible (because A is rank-deficient), this changes mX to be the
/// least-squares minimal solution. This makes mX unique for a given set of
/// inputs, rather than leaving the exact solution undefined. This can also be
/// used to short-circuit an LCP solve before it even needs to start, by using
/// the previous LCP solution as a "close enough" guess that can then be
/// cleaned up by this method and made exact. To faccilitate that use case,
/// this method returns true if it's found a valid solution, whether it
/// changed anything or not, and false if the solution is invalid.
bool ConstrainedGroupGradientMatrices::opportunisticallyStandardizeResults(
    simulation::World* world, Eigen::VectorXd& mX)
{
  mStandardizedResults = true;
  if (mX.size() == 0)
  {
    // a 0-size solution is always valid, as long as it's not trivially broken
    // (wrong dimensions)
    return mX.size() == mA.rows();
  }
  const Eigen::MatrixXd& A_c = getClampingConstraintMatrix();
  if (A_c.cols() == 0)
  {
    // this means that our guessing formula won't work, so just return whether
    // the solution is already valid.
    return isSolutionValid(mX);
  }
  const Eigen::MatrixXd& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXd& E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXd Q = A_c.transpose() * mMinv * A_c_ub_E;
  Eigen::VectorXd b = getClampingConstraintRelativeVels();

  if (A_ub.cols() == 0)
  {
    Eigen::MatrixXd realQ = getClampingAMatrix();
#ifndef NDEBUG
    // Sanity check
    Eigen::MatrixXd diff = Q - realQ;
    assert(std::abs(diff.maxCoeff()) < 1e-11);
    assert(std::abs(diff.minCoeff()) < 1e-11);
#endif
    Q = realQ;
  }
  /*
    else
    {
  #ifndef NDEBUG
      // Sanity check
      Eigen::MatrixXd realQ = getClampingAMatrix();
      Eigen::MatrixXd diff = Q - realQ;
      // These should usually not be equal, but it's not a requirement
      assert(
          std::abs(diff.maxCoeff()) > 1e-11 || std::abs(diff.minCoeff()) >
  1e-11); #endif
    }
  */

  Eigen::VectorXd f_c = Q.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXd originalF_c = getClampingConstraintImpulses();

  bool anyNewlyNotClamping = false;

  Eigen::VectorXd newX = Eigen::VectorXd::Zero(mX.size());
  for (int i = 0; i < newX.size(); i++)
  {
    int clampingIndex = mClampingIndex[i];
    int upperBoundIndex = mUpperBoundIndex[i];
    if (clampingIndex != -1)
    {
      // If we're clamping
      assert(upperBoundIndex == -1);
      newX(i) = f_c(clampingIndex);

      if (std::abs(f_c(clampingIndex)) < CLAMPING_THRESHOLD
          && std::abs(mX(i)) > CLAMPING_THRESHOLD)
      {
        // Only mark stuff as "newly not clamping" if it's not a friction
        // coordinate, since those will be tie broken as clamping anyways
        if (mFIndex(i) == -1)
        {
          anyNewlyNotClamping = true;
        }
      }
    }
    if (upperBoundIndex != -1)
    {
      assert(clampingIndex == -1);
      int fIndex = mFIndex[i];
      assert(mClampingIndex[fIndex] != -1);
      double originalMultiple = mX(mClampingIndex[fIndex]) / mX(i);
      double cleanMultiple = (std::abs(originalMultiple - mHi(i))
                              < std::abs(originalMultiple - mLo(i)))
                                 ? mHi(i)
                                 : mLo(i);
      newX(i) = f_c(mClampingIndex[fIndex]) * cleanMultiple;
    }
  }

  if (isSolutionValid(newX))
  {
    mX = newX;
    ConstrainedGroupGradientMatrices::mX = newX;
    mContactConstraintImpulses = newX;
    mClampingConstraintImpulses = f_c;
    if (anyNewlyNotClamping)
    {
      // If any previously clamping indices have become "not clamping" then
      // we need to reconstruct our matrices
      constructMatrices(world);
    }
    return true;
  }
  else
  {
    mStandardizedResults = false;
    return false;
  }
}

//==============================================================================
void ConstrainedGroupGradientMatrices::deduplicateConstraints()
{
  // Build the merge groups

  const double MERGE_THRESHOLD = 0.01;

  std::vector<int> mergeGroup;
  for (int i = 0; i < mNumConstraintDim; i++)
  {
    mergeGroup.push_back(-1);
  }

  int groupCursor = 0;

  for (int i = 0; i < mNumConstraintDim; i++)
  {
    // Merge with the first neighbor who is near enough
    for (int j = 0; j < i; j++)
    {
      double distance
          = (mMassedImpulseTests[i] - mMassedImpulseTests[j]).squaredNorm();
      if (distance < MERGE_THRESHOLD)
      {
        mergeGroup[i] = mergeGroup[j];
        break;
      }
    }

    // If we didn't find a group to merge into, then form a new group

    if (mergeGroup[i] == -1)
    {
      mergeGroup[i] = groupCursor;
      groupCursor++;
    }
  }

  // We have:
  //
  // mNumConstraintDim - the number of constraints we currently have
  //
  // mX - impulses at each constraint
  // mHi - upper bound at each constraint
  // mLo - lower bound at each constraint
  // mFIndex - pointer for each constraint for friction bound, or -1
  // mB - offset at each constraint
  // mAColNorms - norms of each constraints' col of A
  //
  // mMassedImpulseTests - Minv * impulse test for each constraint
  // mPenetrationCorrectionVelocities - Penetration hack val for each constraint
  // mRestitutionCoeffs - Restitution coefficients for each constraint

  int newNumConstraintDim = groupCursor;
  Eigen::VectorXd newX = Eigen::VectorXd::Zero(newNumConstraintDim);
  Eigen::VectorXd newHi = Eigen::VectorXd::Zero(newNumConstraintDim);
  Eigen::VectorXd newLo = Eigen::VectorXd::Zero(newNumConstraintDim);
  Eigen::VectorXd newB = Eigen::VectorXd::Zero(newNumConstraintDim);
  Eigen::VectorXd newAColNorms = Eigen::VectorXd::Zero(newNumConstraintDim);

  Eigen::VectorXi newFIndex = Eigen::VectorXi::Zero(newNumConstraintDim);

  std::vector<Eigen::VectorXd> newMassedImpulseTests;
  std::vector<double> newPenetrationCorrectionVelocities;
  std::vector<double> newRestitutionCoeffs;
  std::vector<std::shared_ptr<constraint::ConstraintBase>> newConstraints;
  std::vector<int> newConstraintIndices;

  for (int i = 0; i < newNumConstraintDim; i++)
  {
    int groupCount = 0;

    Eigen::VectorXd massedImpulseTest = Eigen::VectorXd::Zero(mNumDOFs);
    double penetrationCorrectionVelocity = 0.0;
    double restitutionCoeff = 0.0;
    std::shared_ptr<constraint::ConstraintBase> constraint = nullptr;
    int constraintIndex = 0;

    for (int j = 0; j < mNumConstraintDim; j++)
    {
      if (mergeGroup[j] == i)
      {
        groupCount++;
        newX(i) += mX(j);
        newHi(i) += mHi(j);
        newLo(i) += mLo(j);
        newB(i) += mB(j);
        newAColNorms(i) += mAColNorms(j);

        newFIndex(i) = mFIndex(j);
        if (newFIndex(i) >= 0)
        {
          newFIndex(i) = mergeGroup[newFIndex(i)];
        }

        massedImpulseTest += mMassedImpulseTests[j];

        penetrationCorrectionVelocity += mPenetrationCorrectionVelocities[j];
        restitutionCoeff += mRestitutionCoeffs[j];

        constraint = mConstraints[j];
        constraintIndex = mConstraintIndices[j];
      }
    }

    newB(i) /= groupCount;
    newHi(i) /= groupCount;
    newLo(i) /= groupCount;
    newAColNorms(i) /= groupCount;
    massedImpulseTest /= groupCount;
    penetrationCorrectionVelocity /= groupCount;
    restitutionCoeff /= groupCount;

    newMassedImpulseTests.push_back(massedImpulseTest);
    newPenetrationCorrectionVelocities.push_back(penetrationCorrectionVelocity);
    newRestitutionCoeffs.push_back(restitutionCoeff);
    newConstraints.push_back(constraint);
    newConstraintIndices.push_back(constraintIndex);
  }

  // Set our new values

  mNumConstraintDim = newNumConstraintDim;
  mX = newX;
  mHi = newHi;
  mLo = newLo;
  mB = newB;
  mAColNorms = newAColNorms;
  mFIndex = newFIndex;
  mMassedImpulseTests = newMassedImpulseTests;
  mPenetrationCorrectionVelocities = newPenetrationCorrectionVelocities;
  mRestitutionCoeffs = newRestitutionCoeffs;
  mConstraints = newConstraints;
  mConstraintIndices = newConstraintIndices;
}

//==============================================================================
/// This gets called during the setup of the ConstrainedGroupGradientMatrices
/// after the LCP has run, with the result from the LCP solver. This can only
/// be called once, and after this is called you cannot call
/// measureConstraintImpulse() again!
void ConstrainedGroupGradientMatrices::constructMatrices(
    simulation::World* world)
{
  // in the new world, we can actually call this multiple times
  // assert(!mFinalized);
  // mFinalized = true;

  // TODO: this actually results in very wrong values when we've got deep
  // inter-penetration, so we're leaving it disabled.
  // deduplicateConstraints();

  mContactConstraintImpulses = mX;
  mContactConstraintMappings = mFIndex;
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
  //
  // There's a special case where the gradient technically doesn't exist, and
  // we're going to break ties about which way we want the gradient to go. If
  // relative velocity is 0, and so is relative force, then we're actually going
  // to call that constraint "Clamping" rather than "Not Clamping", though both
  // could apply. Our motivation is for when a contact is at rest, but has
  // static friction, we don't want to discard the friction constraints just
  // because they have 0 force and 0 velocity.

  // Declare a shared array to re-use for mapping info for each skeleton.
  // Semantics are as follows:
  // - If mappings[j] >= 0, constraint "j" is "Upper Bound".
  // - If mappings[j] == CLAMPING, constraint "j" is "Clamping".
  // - If mappings[j] == NOT_CLAMPING, constraint "j" is "Not Clamping".
  // - If mappings[j] == IRRELEVANT, constraint "j" is doesn't effect this
  //   skeleton, and so can be safely ignored.

  mClampingIndex.clear();
  mUpperBoundIndex.clear();
  std::vector<int> bouncingIndex;
  for (int i = 0; i < mNumConstraintDim; i++)
  {
    mClampingIndex.push_back(-1);
    mUpperBoundIndex.push_back(-1);
    bouncingIndex.push_back(-1);
  }

  int numClamping = 0;
  int numUpperBound = 0;
  int numBouncing = 0;
  int numIllegal = 0;
  // Fill in mappings[] with the correct values, overwriting previous data
  for (std::size_t j = 0; j < mNumConstraintDim; j++)
  {
    // This is the squared l2 norm of the column of A corresponding to this
    // constraint. If it's too small, the optimizer will freely set the force on
    // this constraint because it has negligible effect, which can lead to weird
    // effects like calling a constraint force that's perpendicular to the
    // degrees of freedom of the skeleton getting set to UPPER_BOUND or
    // CLAMPING.
    const double constraintActionNorm = mAColNorms(j);
    if (constraintActionNorm < 1e-9)
    {
      mContactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
      /*
      std::cout << "Labeled " << j
                << " as NOT_CLAMPING because constraintActionNorm is "
                << constraintActionNorm << std::endl;
                */
      continue;
    }

    const double constraintForce = mX(j);
    const double relativeVelocity = mB(j);

    double upperBound = mHi(j);
    double lowerBound = mLo(j);
    const int fIndexPointer = mFIndex(j);
    if (fIndexPointer != -1)
    {
      upperBound *= mX(fIndexPointer);
      lowerBound *= mX(fIndexPointer);
    }

    // If constraintForce is zero, this means "j" is in "Not Clamping" unless
    // relative velocity is also zero
    if (std::abs(constraintForce) < CLAMPING_THRESHOLD)
    {
      if (std::abs(relativeVelocity) < CLAMPING_THRESHOLD)
      {
        // This is technically a TIE! Therefore, technically non-differentiable.
        //
        // We tie break following a heuristic to guess about the behavior of
        // this constraint. This is all mathematically incorrect, since the
        // gradient doesn't exist, but we're trying to pick something that is
        // useful for downstream applications.

        // If this is a frictional contact, we need to check how the
        // corresponding normal constraint is doing. If the normal constraint
        // has a 0 magnitude, then this clamping constraint can't increase to
        // fight friction.
        if (fIndexPointer != -1)
        {
          double normalForce = mX(fIndexPointer);
          // If the normal force is 0, then we're NOT_CLAMPING, because we can't
          // increase our friction force and stay in bounds.
          if (std::abs(normalForce) < CLAMPING_THRESHOLD)
          {
            mContactConstraintMappings(j)
                = neural::ConstraintMapping::NOT_CLAMPING;
          }
          // Otherwise, this is CLAMPING, because as we attempt to move the
          // contact the friction force will stop us.
          else
          {
            mContactConstraintMappings(j) = neural::ConstraintMapping::CLAMPING;
            mClampingIndex[j] = numClamping;
            numClamping++;
          }
        }
        // If this is a normal contact, then we arbitrarily call it
        // NOT_CLAMPING. These generally occur if there are lots of redundant
        // vertices supporting a mesh, and some of them require 0 force during a
        // solve. Leaving these out of CLAMPING also increases stability of our
        // linear algebra later, like in opportunisticallyStandardizeResults()
        else
        {
          mContactConstraintMappings(j)
              = neural::ConstraintMapping::NOT_CLAMPING;
        }
      }
      else
      {
        mContactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
      }
      continue;
    }

    // This means "j" is in "Clamping"
    if ((mX(j) > lowerBound && mX(j) < upperBound)                  // Clamping
        || (lowerBound - mX(j) > 1e-2 || mX(j) - upperBound > 1e-2) // Illegal
    )
    {
      mContactConstraintMappings(j) = neural::ConstraintMapping::CLAMPING;
      mClampingIndex[j] = numClamping;
      numClamping++;
      if (mRestitutionCoeffs[j] > 0)
      {
        bouncingIndex[j] = numBouncing;
        numBouncing++;
      }
    }
    // This means we're out of bounds. This actually happens depressingly often
    // with the current LCP solver, and it means we just have to handle it.
    else if (lowerBound - mX(j) > 1e-2 || mX(j) - upperBound > 1e-2)
    {
      mContactConstraintMappings(j) = neural::ConstraintMapping::ILLEGAL;
      numIllegal++;
    }
    // Otherwise, if fIndex != -1, "j" is in "Upper Bound"
    // Note, this could also mean "j" is at it's lower bound, but we call the
    // group of all "j"'s that have reached their dependent bound "Upper
    // Bound". The only exception to this rule is if the fIndex pointer is
    // pointing at an index that's not clamping, in which case this is also not
    // clamping.
    else if (fIndexPointer != -1 && std::abs(mX(fIndexPointer)) > 1e-9)
    {
      /*
      std::cout << "Listing " << j << " as UB: mX=" << mX(j)
                << ", fIndex=" << fIndex << ", mX(fIndex)=" << mX(fIndex)
                << ", hiBackup=" << hiGradientBackup(j)
                << ", loBackup=" << loGradientBackup(j)
                << ", upperBound=" << upperBound
                << ", lowerBound=" << lowerBound << std::endl;
      */
      mContactConstraintMappings(j) = fIndexPointer;
      mUpperBoundIndex[j] = numUpperBound;
      numUpperBound++;
    }
    // If fIndex == -1, and we're at a bound, then we're actually "Not
    // Clamping", cause the velocity can change freely without the force
    // changing to compensate.
    else
    {
      mContactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
    }
  }

  // Create the matrices we want to pass along for this skeleton:
  mClampingConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numClamping);
#ifndef NDEBUG
  mAllConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, mNumConstraintDim);
#endif
  mMassedClampingConstraintMatrix
      = Eigen::MatrixXd::Zero(mNumDOFs, numClamping);
  mUpperBoundConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numUpperBound);
  mMassedUpperBoundConstraintMatrix
      = Eigen::MatrixXd::Zero(mNumDOFs, numUpperBound);
  mUpperBoundMappingMatrix = Eigen::MatrixXd::Zero(numUpperBound, numClamping);
  mBounceDiagonals = Eigen::VectorXd::Zero(numClamping);
  mPenetrationCorrectionVelocitiesVec = Eigen::VectorXd::Zero(numClamping);
  mBouncingConstraintMatrix = Eigen::MatrixXd::Zero(mNumDOFs, numBouncing);
  mRestitutionDiagonals = Eigen::VectorXd::Zero(numBouncing);
  mClampingConstraintImpulses = Eigen::VectorXd::Zero(numClamping);
  mClampingConstraintRelativeVels = Eigen::VectorXd::Zero(numClamping);
  mDifferentiableConstraints.clear();
  mDifferentiableConstraints.reserve(mNumConstraintDim);
  assert(mDifferentiableConstraints.size() == 0);
  mClampingConstraints.clear();
  mClampingConstraints.reserve(numClamping);
  assert(mClampingConstraints.size() == 0);
  mUpperBoundConstraints.clear();
  mUpperBoundConstraints.reserve(numUpperBound);
  assert(mUpperBoundConstraints.size() == 0);
  mVelocityDueToIllegalImpulses = Eigen::VectorXd::Zero(mNumDOFs);
  mClampingAMatrix = Eigen::MatrixXd::Zero(numClamping, numClamping);

  /*
  std::cout << "numClamping: " << numClamping << std::endl;
  std::cout << "numUpperBound: " << numUpperBound << std::endl;
  std::cout << "numBouncing: " << numBouncing << std::endl;
  */

  // Copy impulse tests into the matrices
  for (size_t j = 0; j < mNumConstraintDim; j++)
  {
    std::shared_ptr<DifferentiableContactConstraint> constraint
        = std::make_shared<DifferentiableContactConstraint>(
            mConstraints[j], mConstraintIndices[j], mX(j));
    mDifferentiableConstraints.push_back(constraint);

#ifndef NDEBUG
    mAllConstraintMatrix.col(j)
        = constraint->getConstraintForces(world, mSkeletons);
#endif

    if (mContactConstraintMappings(j) == neural::ConstraintMapping::CLAMPING)
    {
      assert(numClamping > mClampingIndex[j]);

      mClampingConstraints.push_back(constraint);

      Eigen::VectorXd analyticalImpulse
          = constraint->getConstraintForces(world, mSkeletons);

      mClampingConstraintMatrix.col(mClampingIndex[j]) = analyticalImpulse;
      mMassedClampingConstraintMatrix.col(mClampingIndex[j])
          = mMassedImpulseTests[j];
      mBounceDiagonals(mClampingIndex[j]) = 1 + mRestitutionCoeffs[j];
      mPenetrationCorrectionVelocitiesVec(mClampingIndex[j])
          = mPenetrationCorrectionVelocities[j];
      if (mRestitutionCoeffs[j] > 0)
      {
        mBouncingConstraintMatrix.col(bouncingIndex[j]) = analyticalImpulse;
        mRestitutionDiagonals(bouncingIndex[j]) = mRestitutionCoeffs[j];
      }
      mClampingConstraintImpulses(mClampingIndex[j]) = mX(j);
      mClampingConstraintRelativeVels(mClampingIndex[j]) = mB(j);
    }
    else if (
        mContactConstraintMappings(j) == neural::ConstraintMapping::ILLEGAL)
    {
      mVelocityDueToIllegalImpulses += mX(j) * mMassedImpulseTests[j];
    }
    else if (mContactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
    {
      assert(numUpperBound > mUpperBoundIndex[j]);

      mUpperBoundConstraints.push_back(constraint);
      mUpperBoundConstraintMatrix.col(mUpperBoundIndex[j])
          = constraint->getConstraintForces(world);
      mMassedUpperBoundConstraintMatrix.col(mUpperBoundIndex[j])
          = mMassedImpulseTests[j];
    }
  }

  assert(mClampingConstraints.size() == numClamping);
  assert(mUpperBoundConstraints.size() == numUpperBound);

  // Set up mUpperboundMappingMatrix (aka E)
  for (size_t j = 0; j < mNumConstraintDim; j++)
  {
    if (mContactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
    {
      const int fIndexPointer = mContactConstraintMappings(j);
      const double upperBound = mX(fIndexPointer) * mHi(j);
      const double lowerBound = mX(fIndexPointer) * mLo(j);

      assert(
          mContactConstraintMappings(fIndexPointer)
          == neural::ConstraintMapping::CLAMPING);

      // If we're clamped at the upper bound
      if (std::abs(mX(j) - upperBound) < std::abs(mX(j) - lowerBound))
      {
        if (std::abs(mX(j) - upperBound) > 1e-2)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j = " << j << "): " << mHi(j) << std::endl;
          std::cout << "mLo(j = " << j << "): " << mLo(j) << std::endl;
          std::cout << "mX(j = " << j << "): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(std::abs(mX(j) - upperBound) < 1e-2);
        mUpperBoundMappingMatrix(
            mUpperBoundIndex[j], mClampingIndex[fIndexPointer])
            = mHi(j);
      }
      // If we're clamped at the lower bound
      else
      {
        if (std::abs(mX(j) - lowerBound) > 1e-2)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j = " << j << "): " << mHi(j) << std::endl;
          std::cout << "mLo(j = " << j << "): " << mLo(j) << std::endl;
          std::cout << "mX(j = " << j << "): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(std::abs(mX(j) - lowerBound) < 1e-2);
        mUpperBoundMappingMatrix(
            mUpperBoundIndex[j], mClampingIndex[fIndexPointer])
            = mLo(j);
      }
    }
  }

  // Fill in the clamping A matrix
  for (size_t row = 0; row < mNumConstraintDim; row++)
  {
    if (mContactConstraintMappings(row) == neural::ConstraintMapping::CLAMPING)
    {
      int clampingRow = mClampingIndex[row];
      for (size_t col = 0; col < mNumConstraintDim; col++)
      {
        if (mContactConstraintMappings(col)
            == neural::ConstraintMapping::CLAMPING)
        {
          int clampingCol = mClampingIndex[col];
          mClampingAMatrix(clampingRow, clampingCol) = mA(row, col);
        }
      }
    }
  }
  // If possible (if A is rank-deficient), change to an equivalent
  // least-squares solution that also satisfies the LCP
  opportunisticallyStandardizeResults(world, mX);
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getForceVelJacobian(
    WorldPtr world)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix(world);

  if (A_ub.size() > 0 && E.size() > 0)
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
  }
  else if (A_c.size() > 0)
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * A_c * P_c * Minv);
  }
  else
  {
    return mTimeStep * Minv;
  }
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelVelJacobian(
    WorldPtr world)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  if (A_c.cols() == 0 && A_ub.cols() == 0)
  {
    return Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs);
  }
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix(world);
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
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getPosVelJacobian(
    simulation::WorldPtr world)
{
  return getVelJacobianWrt(world, WithRespectTo::POSITION);
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
  Eigen::MatrixXd W
      = Eigen::MatrixXd::Zero(A_b.rows() * A_b.rows(), A_b.cols());
  for (int i = 0; i < A_b.cols(); i++)
  {
    Eigen::VectorXd a_i = A_b.col(i);
    for (int j = 0; j < A_b.rows(); j++)
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
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
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
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getPosCJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXd posCJac = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  posCJac.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t skelDOF = skel->getNumDofs();
    posCJac.block(cursor, cursor, skelDOF, skelDOF)
        = skel->getJacobianOfC(WithRespectTo::POSITION);
    cursor += skelDOF;
  }
  return posCJac;
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelCJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXd velCJac = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  velCJac.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t skelDOF = skel->getNumDofs();
    velCJac.block(cursor, cursor, skelDOF, skelDOF) = skel->getVelCJacobian();
    cursor += skelDOF;
  }
  return velCJac;
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getMassMatrix(WorldPtr world)
{
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  massMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t skelDOF = skel->getNumDofs();
    massMatrix.block(cursor, cursor, skelDOF, skelDOF) = skel->getMassMatrix();
    cursor += skelDOF;
  }
  return massMatrix;
}

//==============================================================================
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getInvMassMatrix(
    WorldPtr world)
{
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  invMassMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t skelDOF = skel->getNumDofs();
    invMassMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = skel->getInvMassMatrix();
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

  if (A_c.cols() == 0 && V_ub.cols() == 0)
  {
    return Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  }

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
/// This computes and returns the whole pos-vel jacobian. For backprop, you
/// don't actually need this matrix, you can compute backprop directly. This
/// is here if you want access to the full Jacobian for some reason.
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  const Eigen::MatrixXd& A_c = getClampingConstraintMatrix();
  const Eigen::MatrixXd& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXd& E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXd tau = mPreStepTorques;
  Eigen::VectorXd C = mCoriolisAndGravityForces;
  Eigen::VectorXd f_c = getClampingConstraintImpulses();
  double dt = world->getTimeStep();

  Eigen::MatrixXd dM
      = getJacobianOfMinv(world, dt * (tau - C) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXd Minv = world->getInvMassMatrix();
  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);

  Eigen::MatrixXd dF_c = getJacobianOfConstraintForce(world, wrt);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXd dA_c = getJacobianOfClampingConstraints(world, f_c);
    Eigen::MatrixXd dA_ubE = getJacobianOfUpperBoundConstraints(world, E * f_c);
    return dM + Minv * (A_c_ub_E * dF_c + dA_c + dA_ubE - dt * dC);
  }
  else
  {
    return dM + Minv * (A_c_ub_E * dF_c - dt * dC);
  }
}

//==============================================================================
/// This returns the jacobian of constraint force, holding everyhing constant
/// except the value of WithRespectTo
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  if (A_c.cols() == 0)
  {
    int wrtDim = getWrtDim(world, wrt);
    return Eigen::MatrixXd::Zero(0, wrtDim);
  }

  Eigen::MatrixXd Q = getClampingAMatrix();
  Eigen::VectorXd b = getClampingConstraintRelativeVels();

  Eigen::MatrixXd dQ_b
      = getJacobianOfLCPConstraintMatrixClampingSubset(world, b, wrt);

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXd dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

  return dQ_b + Qfac.solve(dB);
}

//==============================================================================
/// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
/// wrt
Eigen::MatrixXd ConstrainedGroupGradientMatrices::
    getJacobianOfLCPConstraintMatrixClampingSubset(
        simulation::WorldPtr world, Eigen::VectorXd b, WithRespectTo* wrt)
{
  const Eigen::MatrixXd& A_c = mClampingConstraintMatrix;
  if (A_c.cols() == 0)
  {
    return Eigen::MatrixXd::Zero(0, 0);
  }
  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    return Eigen::MatrixXd::Zero(A_c.cols(), A_c.cols());
  }
  const Eigen::MatrixXd& A_ub = mUpperBoundConstraintMatrix;
  const Eigen::MatrixXd& E = mUpperBoundMappingMatrix;
  const Eigen::MatrixXd& Minv = getInvMassMatrix(world);
  Eigen::MatrixXd A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXd Q = A_c.transpose() * Minv * A_c_ub_E;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Qfactored
      = Q.completeOrthogonalDecomposition();

  // This is a proxy for f_c
  Eigen::VectorXd Qinv_b = Qfactored.solve(b);

  if (wrt == WithRespectTo::POSITION)
  {
    // Position is the only term that affects A_c and A_ub
    if (mUpperBoundConstraints.size() > 0)
    {
      Eigen::MatrixXd innerTerms
          = getJacobianOfClampingConstraintsTranspose(
                world, mMinv * A_c_ub_E * Qinv_b)
            + A_c.transpose()
                  * (getJacobianOfMinv(world, A_c_ub_E * Qinv_b, wrt)
                     + mMinv
                           * (getJacobianOfClampingConstraints(world, Qinv_b)
                              + getJacobianOfUpperBoundConstraints(
                                  world, E * Qinv_b)));
      Eigen::MatrixXd result = -Qfactored.solve(innerTerms);
      return result;
    }
    else
    {
      // If A_ub is empty, this doesn't matter
      Eigen::MatrixXd innerTerms
          = getJacobianOfClampingConstraintsTranspose(
                world, mMinv * A_c * Qinv_b)
            + A_c.transpose()
                  * (getJacobianOfMinv(world, A_c * Qinv_b, wrt)
                     + mMinv * getJacobianOfClampingConstraints(world, Qinv_b));
      Eigen::MatrixXd result = -Qfactored.solve(innerTerms);
      return result;
    }
  }
  else
  {
    // All other terms get to treat A_c and A_ub as constant
    Eigen::MatrixXd innerTerms
        = A_c.transpose() * getJacobianOfMinv(world, A_c * Qinv_b, wrt);
    Eigen::MatrixXd result = -Qfactored.solve(innerTerms);
    return result;
  }
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::getJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  double dt = world->getTimeStep();
  const Eigen::MatrixXd& A_c = mClampingConstraintMatrix;
  Eigen::MatrixXd dC = getJacobianOfC(world, wrt);
  if (wrt == WithRespectTo::VELOCITY)
  {
    return -A_c.transpose()
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) + dt * mMinv * dC);
  }
  else if (wrt == WithRespectTo::FORCE)
  {
    return -A_c.transpose() * dt * mMinv;
  }

  const Eigen::VectorXd& C = mCoriolisAndGravityForces;
  Eigen::VectorXd f = mPreStepTorques - C;
  Eigen::MatrixXd dMinv_f = getJacobianOfMinv(world, f, wrt);
  Eigen::VectorXd v_f = mPreStepVelocities + (world->getTimeStep() * mMinv * f);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXd dA_c_f
        = getJacobianOfClampingConstraintsTranspose(world, v_f);

    return -(dA_c_f + A_c.transpose() * dt * (dMinv_f - mMinv * dC));
  }
  else
  {
    return -(A_c.transpose() * dt * (dMinv_f - mMinv * dC));
  }
}

//==============================================================================
/// This returns the subset of the A matrix used by the original LCP for just
/// the clamping constraints. It relates constraint force to constraint
/// acceleration. It's a mass matrix, just in a weird frame.
void ConstrainedGroupGradientMatrices::computeLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world, Eigen::MatrixXd& Q, const Eigen::MatrixXd& A_c)
{
  int numClamping = A_c.cols();
  for (int i = 0; i < numClamping; i++)
  {
    Q.col(i)
        = A_c.transpose() * implicitMultiplyByInvMassMatrix(world, A_c.col(i));
  }
}

//==============================================================================
/// This returns the subset of the b vector used by the original LCP for just
/// the clamping constraints. It's just the relative velocity at the clamping
/// contact points.
void ConstrainedGroupGradientMatrices::computeLCPOffsetClampingSubset(
    simulation::WorldPtr world, Eigen::VectorXd& b, const Eigen::MatrixXd& A_c)
{
  b = -A_c.transpose()
      * (world->getVelocities()
         + (world->getTimeStep()
            * implicitMultiplyByInvMassMatrix(
                world,
                world->getExternalForces()
                    - world->getCoriolisAndGravityAndExternalForces())));
}

//==============================================================================
/// This computes and returns an estimate of the constraint impulses for the
/// clamping constraints. This is based on a linear approximation of the
/// constraint impulses.
Eigen::VectorXd
ConstrainedGroupGradientMatrices::estimateClampingConstraintImpulses(
    simulation::WorldPtr world, const Eigen::MatrixXd& A_c)
{
  if (A_c.cols() == 0)
  {
    return Eigen::VectorXd::Zero(0);
  }

  Eigen::VectorXd b = Eigen::VectorXd(A_c.cols());
  Eigen::MatrixXd Q = Eigen::MatrixXd(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c);

  return Q.completeOrthogonalDecomposition().solve(b);
}

//==============================================================================
/// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
/// everything constant except the value of WithRespectTo
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfMinv(world, tau, wrt);
}

//==============================================================================
/// This returns the jacobian of C(pos, inertia, vel), holding everything
/// constant except the value of WithRespectTo
Eigen::MatrixXd ConstrainedGroupGradientMatrices::getJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  return finiteDifferenceJacobianOfC(world, wrt);
}

//==============================================================================
/// This computes the Jacobian of A_c*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::getJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXd f0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(skels);
  }

  return result;
}

//==============================================================================
/// This computes the Jacobian of A_c^T*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::getJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXd v0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(constraints.size(), mNumDOFs);
  for (int i = 0; i < constraints.size(); i++)
  {
    result.row(i)
        = constraints[i]->getConstraintForcesJacobian(world).transpose() * v0;
  }

  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub*E*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::getJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXd f0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mNumDOFs, mNumDOFs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(skels);
  }
  return result;
}

//==============================================================================
/// This replaces x with the result of M*x in place, without explicitly forming
/// M
Eigen::VectorXd ConstrainedGroupGradientMatrices::implicitMultiplyByMassMatrix(
    WorldPtr world, const Eigen::VectorXd& x)
{
  Eigen::VectorXd result = Eigen::VectorXd::Zero(getNumDOFs());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->multiplyByImplicitMassMatrix(x.segment(cursor, dofs));
    cursor += dofs;
  }
  return result;
}

//==============================================================================
/// This replaces x with the result of Minv*x in place, without explicitly
/// forming Minv
Eigen::VectorXd
ConstrainedGroupGradientMatrices::implicitMultiplyByInvMassMatrix(
    WorldPtr world, const Eigen::VectorXd& x)
{
  Eigen::VectorXd result = Eigen::VectorXd::Zero(getNumDOFs());
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::size_t dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->multiplyByImplicitInvMassMatrix(x.segment(cursor, dofs));
    cursor += dofs;
  }
  return result;
}

//==============================================================================
/// Multiply by the vel-vel jacobian, without forming it explicitly
void ConstrainedGroupGradientMatrices::backprop(
    WorldPtr world,
    LossGradient& thisTimestepLoss,
    const LossGradient& nextTimestepLoss)
{
  Eigen::MatrixXd forceVelJacobian = getForceVelJacobian(world);
  // p_t+1 <-- v_t
  Eigen::MatrixXd posVelJacobian = getPosVelJacobian(world);
  // v_t+1 <-- v_t
  Eigen::MatrixXd velVelJacobian = getVelVelJacobian(world);
  // v_t+1 <-- p_t
  Eigen::MatrixXd velPosJacobian = getVelPosJacobian();
  // p_t+1 <-- p_t
  Eigen::MatrixXd posPosJacobian = getPosPosJacobian();

  thisTimestepLoss.lossWrtPosition
      = posVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
        + posPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtVelocity
      = velVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
        + velPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtTorque
      = forceVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity;

  /*
  // Compute intermediate variable "b", described in the doc
  Eigen::VectorXd b = nextTimestepLoss.lossWrtPosition;
  if (mBouncingConstraintMatrix.size() > 0)
  {
    Eigen::MatrixXd X = getPosPosJacobian().transpose();
    b = X * nextTimestepLoss.lossWrtPosition;
  }

  // Compute intermediate variable "x", described in the doc
  Eigen::VectorXd x = implicitMultiplyByInvMassMatrix(
      world, nextTimestepLoss.lossWrtVelocity);

  // Get the massed clamping constraints
  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix();

  // Initialize the intermediate variable "A_c * z", described in the doc
  Eigen::VectorXd A_c_z = Eigen::VectorXd::Zero(x.size());

  // If there are clamping constraints:
  if (V_c.size() > 0)
  {
    Eigen::MatrixXd A_c = getClampingConstraintMatrix();
    Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix();
    Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix();
    Eigen::MatrixXd E = getUpperBoundMappingMatrix();

    Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
    Eigen::MatrixXd forceToVelTranspose
        = constraintForceToImpliedTorques.transpose() * A_c;
    auto velToForceTranspose
        = forceToVelTranspose.completeOrthogonalDecomposition();

    // (V_c + (V_ub * E))^T * nextTimestepLoss.lossWrtVelocity
    Eigen::VectorXd tmp = constraintForceToImpliedTorques.transpose()
                          * nextTimestepLoss.lossWrtVelocity;
    // (A_c^T(V_c + (V_ub * E))^T).pinv() * tmp
    Eigen::VectorXd z = velToForceTranspose.solve(tmp);
    // z = B * z
    z = mBounceDiagonals.cwiseProduct(z);

    x -= V_c * z;

    // Cache the result of "A_c * z"
    A_c_z = A_c * z;
  }
  x *= mTimeStep;

  Eigen::MatrixXd posCJacobianTranspose = getPosCJacobian(world).transpose();
  Eigen::MatrixXd velCJacobianTranspose = getVelCJacobian(world).transpose();

  thisTimestepLoss.lossWrtTorque = x;
  thisTimestepLoss.lossWrtPosition = b - (posCJacobianTranspose * x);
  thisTimestepLoss.lossWrtVelocity = (mTimeStep * b)
                                     + nextTimestepLoss.lossWrtVelocity - A_c_z
                                     - (velCJacobianTranspose * x);
                                     */
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
/// These was the vector of neural::ConstraintMapping mappings used to construct
/// this, where >= 0 indicates UPPER_BOUND. Pretty much only here for testing.
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
const std::vector<std::string>& ConstrainedGroupGradientMatrices::getSkeletons()
    const
{
  return mSkeletons;
}

//==============================================================================
const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
ConstrainedGroupGradientMatrices::getDifferentiableConstraints() const
{
  return mDifferentiableConstraints;
}

//==============================================================================
const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
ConstrainedGroupGradientMatrices::getClampingConstraints() const
{
  return mClampingConstraints;
}

//==============================================================================
const std::vector<std::shared_ptr<DifferentiableContactConstraint>>&
ConstrainedGroupGradientMatrices::getUpperBoundConstraints() const
{
  return mUpperBoundConstraints;
}

//==============================================================================
bool ConstrainedGroupGradientMatrices::areResultsStandardized() const
{
  return mStandardizedResults;
}

//==============================================================================
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getPenetrationCorrectionVelocities() const
{
  return mPenetrationCorrectionVelocitiesVec;
}

//==============================================================================
/// This is the subset of the A matrix from the original LCP that corresponds
/// to clamping indices.
const Eigen::MatrixXd& ConstrainedGroupGradientMatrices::getClampingAMatrix()
    const
{
  return mClampingAMatrix;
}

//==============================================================================
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getClampingConstraintImpulses() const
{
  return mClampingConstraintImpulses;
}

//==============================================================================
/// Returns the relative velocities along the clamping constraints
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getClampingConstraintRelativeVels() const
{
  return mClampingConstraintRelativeVels;
}

//==============================================================================
/// Returns the velocity change caused by the illegal impulses from the LCP
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getVelocityDueToIllegalImpulses() const
{
  return mVelocityDueToIllegalImpulses;
}

//==============================================================================
/// Returns the coriolis and gravity forces pre-step
const Eigen::VectorXd&
ConstrainedGroupGradientMatrices::getCoriolisAndGravityAndExternalForces() const
{
  return mCoriolisAndGravityForces;
}

//==============================================================================
/// Returns the torques applied pre-step
const Eigen::VectorXd& ConstrainedGroupGradientMatrices::getPreStepTorques()
    const
{
  return mPreStepTorques;
}

//==============================================================================
/// Returns the velocity pre-step
const Eigen::VectorXd& ConstrainedGroupGradientMatrices::getPreStepVelocity()
    const
{
  return mPreStepVelocities;
}

//==============================================================================
/// Returns the velocity pre-LCP
const Eigen::VectorXd& ConstrainedGroupGradientMatrices::getPreLCPVelocity()
    const
{
  return mPreLCPVelocities;
}

//==============================================================================
/// Returns the M^{-1} matrix from pre-step
const Eigen::VectorXd& ConstrainedGroupGradientMatrices::getMinv() const
{
  return mMinv;
}

//==============================================================================
/// Get the coriolis and gravity forces
const Eigen::VectorXd
ConstrainedGroupGradientMatrices::getCoriolisAndGravityAndExternalForces(
    simulation::WorldPtr world) const
{
  Eigen::VectorXd result = Eigen::VectorXd(mNumDOFs);
  int cursor = 0;
  for (std::string skelName : mSkeletons)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(skelName);
    int dofs = skel->getNumDofs();
    result.segment(cursor, dofs) = skel->getCoriolisAndGravityForces();
    cursor += dofs;
  }
  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// finite differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd
ConstrainedGroupGradientMatrices::finiteDifferenceJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXd tau, WithRespectTo* wrt)
{
  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  Eigen::VectorXd before = getWrt(world, wrt);

  const double EPS = 1e-8;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    perturbed(i) += EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXd newVPlus = implicitMultiplyByInvMassMatrix(world, tau);
    perturbed = before;
    perturbed(i) -= EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXd newVMinus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXd diff = newVPlus - newVMinus;
    result.col(i) = diff / (2 * EPS);
  }

  setWrt(world, wrt, before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by finite
/// differences.
Eigen::MatrixXd ConstrainedGroupGradientMatrices::finiteDifferenceJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXd original = world->getCoriolisAndGravityAndExternalForces();

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(original.size(), innerDim);

  Eigen::VectorXd before = getWrt(world, wrt);

  const double EPS = 1e-8;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXd perturbed = before;
    perturbed(i) += EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXd tauPos = world->getCoriolisAndGravityAndExternalForces();
    perturbed = before;
    perturbed(i) -= EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXd tauNeg = world->getCoriolisAndGravityAndExternalForces();
    Eigen::VectorXd diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  setWrt(world, wrt, before);

  return result;
}

//==============================================================================
std::size_t ConstrainedGroupGradientMatrices::getWrtDim(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  int sum = 0;
  for (auto skel : mSkeletons)
  {
    sum += wrt->dim(world->getSkeleton(skel).get());
  }
  return sum;
}

//==============================================================================
Eigen::VectorXd ConstrainedGroupGradientMatrices::getWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  Eigen::VectorXd result = Eigen::VectorXd(getWrtDim(world, wrt));
  int cursor = 0;
  for (auto skelName : mSkeletons)
  {
    dynamics::Skeleton* skel = world->getSkeleton(skelName).get();
    int dims = wrt->dim(skel);
    result.segment(cursor, dims) = wrt->get(skel);
    cursor += dims;
  }
  return result;
}

//==============================================================================
void ConstrainedGroupGradientMatrices::setWrt(
    simulation::WorldPtr world, WithRespectTo* wrt, Eigen::VectorXd v)
{
  int cursor = 0;
  for (auto skelName : mSkeletons)
  {
    dynamics::Skeleton* skel = world->getSkeleton(skelName).get();
    int dims = wrt->dim(skel);
    wrt->set(skel, v.segment(cursor, dims));
    cursor += dims;
  }
}

//==============================================================================
/// Gets the skeletons associated with this constrained group in vector form
std::vector<std::shared_ptr<dynamics::Skeleton>>
ConstrainedGroupGradientMatrices::getSkeletons(simulation::WorldPtr world)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels;
  for (auto skelName : mSkeletons)
  {
    skels.push_back(world->getSkeleton(skelName));
  }
  return skels;
}

} // namespace neural
} // namespace dart