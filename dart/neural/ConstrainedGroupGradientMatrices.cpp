#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "dart/constraint/ConstrainedGroup.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/LCPUtils.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

#define CLAMPING_THRESHOLD 1e-6
#define LOG_PERFORMANCE_CONSTRAINED_GROUP

namespace dart {
namespace neural {

//==============================================================================
ConstrainedGroupGradientMatrices::ConstrainedGroupGradientMatrices(
    constraint::ConstrainedGroup& group, s_t timeStep)
  : mFinalized(false), mDeliberatelyIgnoreFriction(false)
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
  mMinv = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  mPreStepTorques = Eigen::VectorXs::Zero(mNumDOFs);
  mPreStepVelocities = Eigen::VectorXs::Zero(mNumDOFs);
  mPreLCPVelocities = Eigen::VectorXs::Zero(mNumDOFs);
  int cursor = 0;
  for (auto skel : skeletons)
  {
    int dofs = skel->getNumDofs();
    mMinv.block(cursor, cursor, dofs, dofs) = skel->getInvMassMatrix();
    mPreStepTorques.segment(cursor, dofs) = skel->getControlForces();
    mPreLCPVelocities.segment(cursor, dofs) = skel->getVelocities();
    mPreStepVelocities.segment(cursor, dofs)
        = skel->getVelocities() - (timeStep * skel->getAccelerations());
    cursor += dofs;
  }
}

//==============================================================================
ConstrainedGroupGradientMatrices::ConstrainedGroupGradientMatrices(
    int numDofs, int numConstraintDim, s_t timeStep)
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
  s_t coeff = constraint->getCoefficientOfRestitution();
  s_t penetrationHack = constraint->getPenetrationCorrectionVelocity();
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
    s_t restitutionCoeff, s_t penetrationHackVel)
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
    // we currently just rely on the order of the calls to sequence our massed
    // impuls test vector correctly, so we don't use constraintIndex
    std::size_t /* constraintIndex */)
{
  // For gradient computations: record the velocity changes for each
  // skeleton for the unit impulse on this constraint. We are guaranteed that
  // this is called after the outer code has already applied the impulse, so we
  // can just read off velocity changes.
  Eigen::VectorXs massedImpulseTest = Eigen::VectorXs::Zero(mNumDOFs);
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
    Eigen::VectorXs massedImpulseTest)
{
  mMassedImpulseTests.push_back(massedImpulseTest);
}

//==============================================================================
void ConstrainedGroupGradientMatrices::registerLCPResults(
    Eigen::VectorXs X,
    Eigen::VectorXs hi,
    Eigen::VectorXs lo,
    Eigen::VectorXi fIndex,
    Eigen::VectorXs b,
    Eigen::VectorXs aColNorms,
    Eigen::MatrixXs A,
    s_t constraintForceMixingConstant,
    bool deliberatelyIgnoreFriction)
{
  mX = X;
  mHi = hi;
  mLo = lo;
  mFIndex = fIndex;
  mB = b;
  mAColNorms = aColNorms;
  mA = A;
  mConstraintForceMixingConstant = constraintForceMixingConstant;
  mDeliberatelyIgnoreFriction = deliberatelyIgnoreFriction;
}

//==============================================================================
/// This returns true if the proposed mX is consistent with our recorded LCP
/// construction
bool ConstrainedGroupGradientMatrices::isSolutionValid(
    const Eigen::VectorXs& mX)
{
  return constraint::LCPUtils::isLCPSolutionValid(
      mA, mX, mB, mHi, mLo, mFIndex, mDeliberatelyIgnoreFriction);
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
    simulation::World* world, Eigen::VectorXs& mX)
{
  mStandardizedResults = true;
  if (mX.size() == 0)
  {
    // a 0-size solution is always valid, as long as it's not trivially broken
    // (wrong dimensions)
    return mX.size() == mA.rows();
  }
  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  if (A_c.cols() == 0)
  {
    // this means that we should apply 0 force all around, so check if that's
    // right. If it isn't, then something is pretty badly wrong...
    Eigen::VectorXs zero = Eigen::VectorXs::Zero(mX.size());
    if (isSolutionValid(zero))
    {
      mX = zero;
      return true;
    }
    else
    {
      mStandardizedResults = false;
      return false;
    }
  }
  const Eigen::MatrixXs& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXs& E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXs Q = A_c.transpose() * mMinv * A_c_ub_E;
  Q.diagonal() += getConstraintForceMixingDiagonal();
  Eigen::VectorXs b = getClampingConstraintRelativeVels();

  if (A_ub.cols() == 0)
  {
    Eigen::MatrixXs realQ = getClampingAMatrix();
#ifndef NDEBUG
    // Sanity check
    Eigen::MatrixXs diff = Q - realQ;
    assert(abs(diff.maxCoeff()) < 1e-11);
    assert(abs(diff.minCoeff()) < 1e-11);
#endif
    Q = realQ;
  }
  /*
    else
    {
  #ifndef NDEBUG
      // Sanity check
      Eigen::MatrixXs realQ = getClampingAMatrix();
      Eigen::MatrixXs diff = Q - realQ;
      // These should usually not be equal, but it's not a requirement
      assert(
          abs(diff.maxCoeff()) > 1e-11 || abs(diff.minCoeff()) >
  1e-11); #endif
    }
  */
  // TODO: <remove>
  mStabilizationPos = world->getPositions();
  mStabilizationVel = world->getVelocities();
  // TODO: </remove>
  mStabilizationQ = Q;
  mStabilizationB = b;

  Eigen::VectorXs f_c = Q.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXs originalF_c = getClampingConstraintImpulses();

  bool anyNewlyNotClamping = false;

  Eigen::VectorXs newX = Eigen::VectorXs::Zero(mX.size());
  for (int i = 0; i < newX.size(); i++)
  {
    int clampingIndex = mClampingIndex[i];
    int upperBoundIndex = mUpperBoundIndex[i];
    if (clampingIndex != -1)
    {
      // If we're clamping
      assert(upperBoundIndex == -1);
      newX(i) = f_c(clampingIndex);

      if (abs(f_c(clampingIndex)) < CLAMPING_THRESHOLD
          && abs(mX(i)) > CLAMPING_THRESHOLD)
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
      s_t originalMultiple = originalF_c(mClampingIndex[fIndex]) / mX(i);
      s_t cleanMultiple
          = (abs(originalMultiple - mHi(i)) < abs(originalMultiple - mLo(i)))
                ? mHi(i)
                : mLo(i);
      newX(i) = f_c(mClampingIndex[fIndex]) * cleanMultiple;
    }
  }

  if (isSolutionValid(newX))
  {
    mX = newX;
    ConstrainedGroupGradientMatrices::mX = newX;
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

  const s_t MERGE_THRESHOLD = 0.01;

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
      s_t distance
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
  Eigen::VectorXs newX = Eigen::VectorXs::Zero(newNumConstraintDim);
  Eigen::VectorXs newHi = Eigen::VectorXs::Zero(newNumConstraintDim);
  Eigen::VectorXs newLo = Eigen::VectorXs::Zero(newNumConstraintDim);
  Eigen::VectorXs newB = Eigen::VectorXs::Zero(newNumConstraintDim);
  Eigen::VectorXs newAColNorms = Eigen::VectorXs::Zero(newNumConstraintDim);

  Eigen::VectorXi newFIndex = Eigen::VectorXi::Zero(newNumConstraintDim);

  std::vector<Eigen::VectorXs> newMassedImpulseTests;
  std::vector<s_t> newPenetrationCorrectionVelocities;
  std::vector<s_t> newRestitutionCoeffs;
  std::vector<std::shared_ptr<constraint::ConstraintBase>> newConstraints;
  std::vector<int> newConstraintIndices;

  for (int i = 0; i < newNumConstraintDim; i++)
  {
    int groupCount = 0;

    Eigen::VectorXs massedImpulseTest = Eigen::VectorXs::Zero(mNumDOFs);
    s_t penetrationCorrectionVelocity = 0.0;
    s_t restitutionCoeff = 0.0;
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
    simulation::World* world, Eigen::VectorXi overrideClasses)
{
  // in the new world, we can actually call this multiple times
  // assert(!mFinalized);
  // mFinalized = true;

  // TODO: this actually results in very wrong values when we've got deep
  // inter-penetration, so we're leaving it disabled.
  // deduplicateConstraints();

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
    // If we passed in an `overrideClasses` vector then we can use that to set
    // the classes instead.
    if (overrideClasses.size() == mNumConstraintDim)
    {
      int overrideClass = overrideClasses(j);
      mContactConstraintMappings(j) = overrideClass;
      if (overrideClass == neural::ConstraintMapping::NOT_CLAMPING)
      {
        // Do nothing
      }
      else if (overrideClass == neural::ConstraintMapping::CLAMPING)
      {
        mClampingIndex[j] = numClamping;
        numClamping++;
      }
      else
      {
        mUpperBoundIndex[j] = numUpperBound;
        numUpperBound++;
      }
      continue;
    }

    // This is the squared l2 norm of the column of A corresponding to this
    // constraint. If it's too small, the optimizer will freely set the force on
    // this constraint because it has negligible effect, which can lead to weird
    // effects like calling a constraint force that's perpendicular to the
    // degrees of freedom of the skeleton getting set to UPPER_BOUND or
    // CLAMPING.
    const s_t constraintActionNorm = mAColNorms(j);
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

    const s_t constraintForce = mX(j);
    const s_t relativeVelocity = mB(j);

    s_t upperBound = mHi(j);
    s_t lowerBound = mLo(j);
    const int fIndexPointer = mFIndex(j);
    if (fIndexPointer != -1)
    {
      upperBound *= mX(fIndexPointer);
      lowerBound *= mX(fIndexPointer);
    }

    // If constraintForce is zero, this means "j" is in "Not Clamping" unless
    // relative velocity is also zero
    if (abs(constraintForce) < CLAMPING_THRESHOLD)
    {
      // If this is a frictional contact, we need to check how the
      // corresponding normal constraint is doing. If the normal constraint
      // has a 0 magnitude, then this clamping constraint can't increase to
      // fight friction, and it's not UPPER_BOUND in this special case because
      // the constraint force is at 0. If the normal force isn't at 0, then we
      // just say friction forces are always CLAMPING.
      if (fIndexPointer != -1)
      {
        s_t normalForce = mX(fIndexPointer);
        // If the corresponding normal force is 0, then we're NOT_CLAMPING,
        // because we can't increase our friction force and stay in bounds.
        if (abs(normalForce) < CLAMPING_THRESHOLD)
        {
          mContactConstraintMappings(j)
              = neural::ConstraintMapping::NOT_CLAMPING;
        }
        // Otherwise, this is CLAMPING, because as we attempt to move the
        // contact the friction force will stop us. If we're deliberately
        // ignoring friction then this is NOT_CLAMPING, because we'll always set
        // the values to 0.
        else
        {
          if (mDeliberatelyIgnoreFriction)
          {
            mContactConstraintMappings(j)
                = neural::ConstraintMapping::NOT_CLAMPING;
          }
          else
          {
            mContactConstraintMappings(j) = neural::ConstraintMapping::CLAMPING;
            mClampingIndex[j] = numClamping;
            numClamping++;
          }
        }
      }
      else
      {
        mContactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
      }

      if (abs(relativeVelocity) < CLAMPING_THRESHOLD)
      {
        // This is technically a TIE! Therefore, technically non-differentiable.
        //
        // For now, we just default to the above classification as tie-breaking.
        //
        // If this is a normal contact, then we arbitrarily still call it
        // NOT_CLAMPING. These generally occur if there are lots of redundant
        // vertices supporting a mesh, and some of them require 0 force during a
        // solve. Leaving these out of CLAMPING also increases stability of our
        // linear algebra later, like in opportunisticallyStandardizeResults()
        //
        // If it's a friction pointer, then the above classification is right.
      }

      continue;
    }

    // If we're within a very small distance to the upper bound, we want to tie
    // break in favor of upper bounding
    s_t tieBreakToUpperBound = 1e-5;

    // This means "j" is in "Clamping"
    if ((mX(j) > lowerBound + tieBreakToUpperBound
         && mX(j) < upperBound - tieBreakToUpperBound)              // Clamping
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
    else if (
        fIndexPointer != -1 && abs(mX(fIndexPointer)) > 1e-9
        && mAColNorms(fIndexPointer) > 1e-9
        && ((fIndexPointer > j)
            || mContactConstraintMappings(fIndexPointer) == CLAMPING))
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
  mClampingConstraintMatrix = Eigen::MatrixXs::Zero(mNumDOFs, numClamping);
  mAllConstraintMatrix = Eigen::MatrixXs::Zero(mNumDOFs, mNumConstraintDim);
  mMassedClampingConstraintMatrix
      = Eigen::MatrixXs::Zero(mNumDOFs, numClamping);
  mUpperBoundConstraintMatrix = Eigen::MatrixXs::Zero(mNumDOFs, numUpperBound);
  mMassedUpperBoundConstraintMatrix
      = Eigen::MatrixXs::Zero(mNumDOFs, numUpperBound);
  mUpperBoundMappingMatrix = Eigen::MatrixXs::Zero(numUpperBound, numClamping);
  mBounceDiagonals = Eigen::VectorXs::Zero(numClamping);
  mPenetrationCorrectionVelocitiesVec = Eigen::VectorXs::Zero(numClamping);
  mBouncingConstraintMatrix = Eigen::MatrixXs::Zero(mNumDOFs, numBouncing);
  mRestitutionDiagonals = Eigen::VectorXs::Zero(numBouncing);
  mClampingConstraintImpulses = Eigen::VectorXs::Zero(numClamping);
  mClampingConstraintRelativeVels = Eigen::VectorXs::Zero(numClamping);
  mDifferentiableConstraints.clear();
  mDifferentiableConstraints.reserve(mNumConstraintDim);
  assert(mDifferentiableConstraints.size() == 0);
  mClampingConstraints.clear();
  mClampingConstraints.reserve(numClamping);
  assert(mClampingConstraints.size() == 0);
  mUpperBoundConstraints.clear();
  mUpperBoundConstraints.reserve(numUpperBound);
  assert(mUpperBoundConstraints.size() == 0);
  mVelocityDueToIllegalImpulses = Eigen::VectorXs::Zero(mNumDOFs);
  mClampingAMatrix = Eigen::MatrixXs::Zero(numClamping, numClamping);
  mConstraintForceMixingDiagonal
      = Eigen::VectorXs::Ones(numClamping) * mConstraintForceMixingConstant;

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

    mAllConstraintMatrix.col(j)
        = constraint->getConstraintForces(world, mSkeletons);

    if (mContactConstraintMappings(j) == neural::ConstraintMapping::CLAMPING)
    {
      assert(numClamping > mClampingIndex[j]);

      mClampingConstraints.push_back(constraint);

      mClampingConstraintMatrix.col(mClampingIndex[j])
          = mAllConstraintMatrix.col(j);
      mMassedClampingConstraintMatrix.col(mClampingIndex[j])
          = mMassedImpulseTests[j];
      mBounceDiagonals(mClampingIndex[j]) = 1 + mRestitutionCoeffs[j];
      mPenetrationCorrectionVelocitiesVec(mClampingIndex[j])
          = mPenetrationCorrectionVelocities[j];
      if (mRestitutionCoeffs[j] > 0)
      {
        mBouncingConstraintMatrix.col(bouncingIndex[j])
            = mAllConstraintMatrix.col(j);
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
          = constraint->getConstraintForces(world, mSkeletons);
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
      const s_t upperBound = mX(fIndexPointer) * mHi(j);
      const s_t lowerBound = mX(fIndexPointer) * mLo(j);

      assert(
          mContactConstraintMappings(fIndexPointer)
          == neural::ConstraintMapping::CLAMPING);

      // If we're clamped at the upper bound
      if (abs(mX(j) - upperBound) < abs(mX(j) - lowerBound))
      {
        if (abs(mX(j) - upperBound) > 1e-2)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j = " << j << "): " << mHi(j) << std::endl;
          std::cout << "mLo(j = " << j << "): " << mLo(j) << std::endl;
          std::cout << "mX(j = " << j << "): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(abs(mX(j) - upperBound) < 1e-2);
        mUpperBoundMappingMatrix(
            mUpperBoundIndex[j], mClampingIndex[fIndexPointer])
            = mHi(j);
      }
      // If we're clamped at the lower bound
      else
      {
        if (abs(mX(j) - lowerBound) > 1e-2)
        {
          std::cout << "Lower bound: " << lowerBound << std::endl;
          std::cout << "Upper bound: " << upperBound << std::endl;
          std::cout << "mHi(j = " << j << "): " << mHi(j) << std::endl;
          std::cout << "mLo(j = " << j << "): " << mLo(j) << std::endl;
          std::cout << "mX(j = " << j << "): " << mX(j) << std::endl;
          std::cout << "fIndex: " << fIndexPointer << std::endl;
        }
        assert(abs(mX(j) - lowerBound) < 1e-2);
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
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getControlForceVelJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getControlForceVelJacobian");
  }
#endif

  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  Eigen::MatrixXs Minv = getInvMassMatrix(world);

  Eigen::MatrixXs jac;

  // If there are no clamping constraints, then force-vel is just the
  // mTimeStep
  // * Minv
  if (A_c.size() == 0)
  {
    jac = mTimeStep * Minv;
  }
  else
  {
    jac = getVelJacobianWrt(world, WithRespectTo::FORCE);
  }

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getVelVelJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getVelVelJacobian");
  }
#endif

  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  Eigen::MatrixXs jac;

  // If there are no clamping constraints, then vel-vel is just the identity
  Eigen::VectorXs ddamp = getDampingVector(world);
  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  s_t dt = world->getTimeStep();
  if (A_c.size() == 0)
  {
    jac = Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs)
          - getControlForceVelJacobian(world) * getVelCJacobian(world)
          - dt*Minv*ddamp.asDiagonal();
  }
  else
  {
    jac = getVelJacobianWrt(world, WithRespectTo::VELOCITY) -dt*Minv*ddamp.asDiagonal();
  }

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getPosVelJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getPosVelJacobian");
  }
#endif

  Eigen::MatrixXs jac = getVelJacobianWrt(world, WithRespectTo::POSITION);

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getPosPosJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getPosPosJacobian");
  }
#endif

  Eigen::MatrixXs jac = getJointsPosPosJacobian(world)
                        * getBounceApproximationJacobian(thisLog);

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getVelPosJacobian(
    simulation::WorldPtr world, PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getVelPosJacobian");
  }
#endif

  Eigen::MatrixXs jac = getJointsVelPosJacobian(world)
                        * getBounceApproximationJacobian(thisLog);

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getPosCJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXs posCJac = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
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
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getVelCJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXs velCJac = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
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
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getMassMatrix(WorldPtr world)
{
  Eigen::MatrixXs massMatrix = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
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
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getInvMassMatrix(
    WorldPtr world)
{
  Eigen::MatrixXs invMassMatrix = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
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

Eigen::VectorXs ConstrainedGroupGradientMatrices::getDampingVector(
  WorldPtr world
)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mNumDOFs);
  std::size_t cursor = 0;
  for(std::size_t i=0;i<mSkeletons.size();i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::vector<dynamics::DegreeOfFreedom*> skeldofs = skel->getDofs();
    std::size_t nDofs = skel->getNumDofs();
    for (int j=0;j<nDofs;j++)
    {
      result(cursor) = skeldofs[j]->getDampingCoefficient();
      cursor ++;
    }
  }
  return result;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getJointsPosPosJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  int cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    int dofs = skel->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = skel->getPosPosJac(
        skel->getPositions(), skel->getVelocities(), mTimeStep);
    cursor += dofs;
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getJointsVelPosJacobian(
    simulation::WorldPtr world)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  int cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    int dofs = skel->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs) = skel->getVelPosJac(
        skel->getPositions(), skel->getVelocities(), mTimeStep);
    cursor += dofs;
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getBounceApproximationJacobian(
    PerformanceLog* perfLog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun(
        "ConstrainedGroupGradientMatrices.getBounceApproximationJacobian");
  }
#endif

  const Eigen::MatrixXs& A_b = getBouncingConstraintMatrix();
  Eigen::MatrixXs jac;

  // If there are no bounces, pos-pos is a simple identity
  if (A_b.size() == 0)
  {
    jac = Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs);
  }
  else
  {
    // Construct the W matrix we'll need to use to solve for our closest
    // approx
    Eigen::MatrixXs W
        = Eigen::MatrixXs::Zero(A_b.rows() * A_b.rows(), A_b.cols());
    for (int i = 0; i < A_b.cols(); i++)
    {
      Eigen::VectorXs a_i = A_b.col(i);
      for (int j = 0; j < A_b.rows(); j++)
      {
        W.block(j * A_b.rows(), i, A_b.rows(), 1) = a_i(j) * a_i;
      }
    }

    // We want to center the solution around the identity matrix, and find the
    // least-squares deviation along the diagonals that gets us there.
    Eigen::VectorXs center = Eigen::VectorXs::Zero(mNumDOFs * mNumDOFs);
    for (std::size_t i = 0; i < mNumDOFs; i++)
    {
      center((i * mNumDOFs) + i) = 1;
    }

    // Solve the linear system
    Eigen::VectorXs q
        = center
          - W.transpose().completeOrthogonalDecomposition().solve(
              getRestitutionDiagonals() + (W.eval().transpose() * center));

    // Recover X from the q vector
    Eigen::MatrixXs X = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
    for (std::size_t i = 0; i < mNumDOFs; i++)
    {
      X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
    }

    jac = X;
  }

#ifdef LOG_PERFORMANCE_CONSTRAINED_GROUP
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
  return jac;
}

//==============================================================================
/// This computes and returns the whole pos-vel jacobian. For backprop, you
/// don't actually need this matrix, you can compute backprop directly. This
/// is here if you want access to the full Jacobian for some reason.
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getVelJacobianWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  int wrtDim = 0;
  int dofs = 0;
  for (const std::string& skelName : mSkeletons)
  {
    auto skel = world->getSkeleton(skelName);
    wrtDim += wrt->dim(skel.get());
    dofs += skel->getNumDofs();
  }
  if (wrtDim == 0)
  {
    return Eigen::MatrixXs::Zero(dofs, 0);
  }
  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  const Eigen::MatrixXs& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXs& E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::VectorXs tau = mPreStepTorques;
  Eigen::VectorXs C = getCoriolisAndGravityAndExternalForces(world);
  const Eigen::VectorXs& f_c = getClampingConstraintImpulses();
  s_t dt = world->getTimeStep();
  Eigen::VectorXs ddamp = getDampingVector(world);
  Eigen::VectorXs v_t = Eigen::VectorXs::Zero(dofs);
  int cursor = 0;
  for(std::size_t i=0;i<mSkeletons.size();i++)
  {
    SkeletonPtr skel = world->getSkeleton(mSkeletons[i]);
    std::vector<dynamics::DegreeOfFreedom*> skeldofs = skel->getDofs();
    std::size_t nDofs = skel->getNumDofs();
    for (int j=0;j<nDofs;j++)
    {
      v_t(cursor) = skeldofs[j]->getVelocity();
      cursor ++;
    }
  }
  Eigen::MatrixXs dM
      = getJacobianOfMinv(world, dt * (tau - C - ddamp.asDiagonal()*v_t) + A_c_ub_E * f_c, wrt);

  Eigen::MatrixXs Minv = getInvMassMatrix(world);

  Eigen::MatrixXs dF_c = getJacobianOfConstraintForce(world, wrt);

  if (wrt == WithRespectTo::FORCE)
  {
    return Minv
           * ((A_c_ub_E * dF_c)
              + (dt * Eigen::MatrixXs::Identity(dofs, wrtDim)));
  }

  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);

  if (wrt == WithRespectTo::VELOCITY)
  {
    return Eigen::MatrixXs::Identity(dofs, wrtDim)
           + Minv * (A_c_ub_E * dF_c - dt * dC);
  }
  else if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dA_c = getJacobianOfClampingConstraints(world, f_c);
    Eigen::MatrixXs dA_ubE = getJacobianOfUpperBoundConstraints(world, E * f_c);
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
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getJacobianOfConstraintForce(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  if (A_c.cols() == 0)
  {
    int wrtDim = wrt->dim(world.get());
    return Eigen::MatrixXs::Zero(0, wrtDim);
  }
  const Eigen::MatrixXs& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXs& E = getUpperBoundMappingMatrix();

  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;
  Eigen::MatrixXs Q = A_c.transpose() * Minv * A_c_ub_E;
  Q.diagonal() += getConstraintForceMixingDiagonal();

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> Qfac
      = Q.completeOrthogonalDecomposition();

  Eigen::MatrixXs dB = getJacobianOfLCPOffsetClampingSubset(world, wrt);

  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    // dQ_b is 0, so don't compute it
    return Qfac.solve(dB);
  }

  Eigen::VectorXs b = getClampingConstraintRelativeVels();
  Eigen::MatrixXs dQ_b
      = getJacobianOfLCPConstraintMatrixClampingSubset(world, b, wrt);

  return dQ_b + Qfac.solve(dB);
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::dQ_WithUB(
    simulation::WorldPtr world,
    const Eigen::MatrixXs& Minv,
    const Eigen::MatrixXs& A_c,
    const Eigen::MatrixXs& E,
    const Eigen::MatrixXs& A_c_ub_E,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = getJacobianOfClampingConstraintsTranspose(world, Minv * A_c_ub_E * rhs)
        + (A_c.transpose()
           * (getJacobianOfMinv(world, A_c_ub_E * rhs, wrt)
              + (Minv
                 * (getJacobianOfClampingConstraints(world, rhs)
                    + getJacobianOfUpperBoundConstraints(world, E * rhs)))));
  return result;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::dQT_WithUB(
    simulation::WorldPtr world,
    const Eigen::MatrixXs& Minv,
    const Eigen::MatrixXs& A_c,
    const Eigen::MatrixXs& E,
    const Eigen::MatrixXs& A_ub,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)
         + (A_c.transpose()
            * (getJacobianOfMinv(world, A_c * rhs, wrt)
               + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))
        + (E.transpose()
           * (getJacobianOfUpperBoundConstraintsTranspose(
                  world, Minv * A_c * rhs)
              + A_ub.transpose()
                    * (getJacobianOfMinv(world, A_c * rhs, wrt)
                       + (Minv
                          * (getJacobianOfClampingConstraints(world, rhs))))));
  return result;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::dQ_WithoutUB(
    simulation::WorldPtr world,
    const Eigen::MatrixXs& Minv,
    const Eigen::MatrixXs& A_c,
    Eigen::VectorXs rhs,
    WithRespectTo* wrt)
{
  Eigen::MatrixXs result
      = getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)
        + (A_c.transpose()
           * (getJacobianOfMinv(world, A_c * rhs, wrt)
              + (Minv * (getJacobianOfClampingConstraints(world, rhs)))));
  return result;
}

//==============================================================================
/// This returns the vector of constants that get added to the diagonal of Q
/// to guarantee that Q is full-rank
Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getConstraintForceMixingDiagonal()
{
  return mConstraintForceMixingDiagonal;
}

//==============================================================================
/// This returns the jacobian of Q^{-1}b, holding b constant, with respect to
/// wrt
Eigen::MatrixXs ConstrainedGroupGradientMatrices::
    getJacobianOfLCPConstraintMatrixClampingSubset(
        simulation::WorldPtr world, Eigen::VectorXs b, WithRespectTo* wrt)
{
  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  if (A_c.cols() == 0)
  {
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (wrt == WithRespectTo::VELOCITY || wrt == WithRespectTo::FORCE)
  {
    return Eigen::MatrixXs::Zero(A_c.cols(), mNumDOFs);
  }

  const Eigen::MatrixXs& A_ub = getUpperBoundConstraintMatrix();
  const Eigen::MatrixXs& E = getUpperBoundMappingMatrix();
  Eigen::MatrixXs A_c_ub_E = A_c + A_ub * E;

  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  Eigen::MatrixXs Q = A_c.transpose() * Minv * (A_c + A_ub * E);
  Q.diagonal() += getConstraintForceMixingDiagonal();
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXs> Qfactored
      = Q.completeOrthogonalDecomposition();

  Eigen::VectorXs Qinv_b = Qfactored.solve(b);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs Qinv = Qfactored.pseudoInverse();
    Eigen::MatrixXs I = Eigen::MatrixXs::Identity(Q.rows(), Q.cols());

    // Position is the only term that affects A_c and A_ub. We use the full
    // gradient of the pseudoinverse, rather than approximate with the gradient
    // of the raw inverse, because Q could be rank-deficient.

    if (A_ub.cols() > 0)
    {

#define dQ(rhs)                                                                \
  (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c_ub_E * rhs)     \
   + (A_c.transpose()                                                          \
      * (getJacobianOfMinv(world, A_c_ub_E * rhs, wrt)                         \
         + (Minv                                                               \
            * (getJacobianOfClampingConstraints(world, rhs)                    \
               + getJacobianOfUpperBoundConstraints(world, E * rhs))))))

#define dQT(rhs)                                                               \
  ((getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)         \
    + (A_c.transpose()                                                         \
       * (getJacobianOfMinv(world, A_c * rhs, wrt)                             \
          + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))         \
   + (E.transpose()                                                            \
      * (getJacobianOfUpperBoundConstraintsTranspose(world, Minv * A_c * rhs)  \
         + A_ub.transpose()                                                    \
               * (getJacobianOfMinv(world, A_c * rhs, wrt)                     \
                  + (Minv                                                      \
                     * (getJacobianOfClampingConstraints(world, rhs)))))))

      Eigen::MatrixXs imprecisionMap = I - Q * Qinv;

      // If we were able to precisely invert Q, then let's use the exact inverse
      // Jacobian, because it's faster to compute
      if (imprecisionMap.squaredNorm() < 1e-18)
      {
        // Note: this formula only asks for the Jacobian of Minv once, instead
        // of 3 times like the below formula. That's actually a pretty big speed
        // advantage. When we can, we should use this formula instead.
        return -Qfactored.solve(dQ(Qfactored.solve(b)));
      }
      // Otherwise fall back to the exact Jacobian of the pseudo-inverse
      else
      {
        // This is the gradient of the pseudoinverse, see
        // https://mathoverflow.net/a/29511/163259
        return -Qfactored.solve(dQ(Qfactored.solve(b)))
               + Qfactored.solve(Qinv.transpose() * dQT(imprecisionMap * b))
               + (I - Qinv * Q) * dQT(Qinv.transpose() * Qfactored.solve(b));
      }

#undef dQ
#undef dQT
    }
    else
    {

      // A_ub = 0 here

#define dQ(rhs)                                                                \
  (getJacobianOfClampingConstraintsTranspose(world, Minv * A_c * rhs)          \
   + (A_c.transpose()                                                          \
      * (getJacobianOfMinv(world, A_c * rhs, wrt)                              \
         + (Minv * (getJacobianOfClampingConstraints(world, rhs))))))

#define dQT(rhs) dQ(rhs)

      Eigen::MatrixXs imprecisionMap = I - Q * Qinv;

      // If we were able to precisely invert Q, then let's use the exact inverse
      // Jacobian, because it's faster to compute
      if (imprecisionMap.squaredNorm() < 1e-18)
      {
        // Note: this formula only asks for the Jacobian of Minv once, instead
        // of 3 times like the below formula. That's actually a pretty big speed
        // advantage. When we can, we should use this formula instead.
        // return -Qinv * dQ(Qinv * b);
        return -Qfactored.solve(dQ(Qfactored.solve(b)));
      }
      else
      {
        // This is the gradient of the pseudoinverse, see
        // https://mathoverflow.net/a/29511/163259
        /*
        return -Qinv * dQ(Qinv * b)
               + Qinv * Qinv.transpose() * dQT(imprecisionMap * b)
               + (I - Qinv * Q) * dQT(Qinv.transpose() * Qinv * b);
        */
        return -Qfactored.solve(dQ(Qfactored.solve(b)))
               + Qfactored.solve(Qinv.transpose() * dQT(imprecisionMap * b))
               + (I - Qinv * Q) * dQT(Qinv.transpose() * Qfactored.solve(b));
      }

#undef dQ
#undef dQT
    }
  }
  else
  {
    // All other terms get to treat A_c as constant
    Eigen::MatrixXs innerTerms
        = A_c.transpose() * getJacobianOfMinv(world, A_c * Qinv_b, wrt);
    Eigen::MatrixXs result = -Qfactored.solve(innerTerms);

    return result;
  }

  assert(false && "Execution should never reach this point.");
}

//==============================================================================
/// This returns the jacobian of b (from Q^{-1}b) with respect to wrt
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getJacobianOfLCPOffsetClampingSubset(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  s_t dt = world->getTimeStep();
  Eigen::MatrixXs Minv = getInvMassMatrix(world);
  const Eigen::MatrixXs& A_c = getClampingConstraintMatrix();
  Eigen::MatrixXs dC = getJacobianOfC(world, wrt);
  if (wrt == WithRespectTo::VELOCITY)
  {
    Eigen::MatrixXs ddamp = getDampingVector(world).asDiagonal();
    return getBounceDiagonals().asDiagonal() * -A_c.transpose()
           * (Eigen::MatrixXs::Identity(mNumDOFs, mNumDOFs)
           - dt*Minv*(dC+ddamp));
  }
  else if (wrt == WithRespectTo::FORCE)
  {
    return getBounceDiagonals().asDiagonal() * -A_c.transpose() * dt * Minv;
  }

  Eigen::VectorXs C = getCoriolisAndGravityAndExternalForces(world);
  Eigen::VectorXs f = getPreStepTorques() - C;
  Eigen::VectorXs v_f = mPreLCPVelocities;
  Eigen::VectorXs v_t = mPreStepVelocities;
  Eigen::MatrixXs ddamp = getDampingVector(world).asDiagonal();
  f = f - ddamp*v_t;
  Eigen::MatrixXs dMinv_f = getJacobianOfMinv(world, f, wrt);

  if (wrt == WithRespectTo::POSITION)
  {
    Eigen::MatrixXs dA_c_f
        = getJacobianOfClampingConstraintsTranspose(world, v_f);

    return getBounceDiagonals().asDiagonal()
           * -(dA_c_f + A_c.transpose() * dt * (dMinv_f - Minv * dC));
  }
  else
  {
    return getBounceDiagonals().asDiagonal()
           * -(A_c.transpose() * dt * (dMinv_f - Minv * dC));
  }
}

//==============================================================================
/// This returns the subset of the A matrix used by the original LCP for just
/// the clamping constraints. It relates constraint force to constraint
/// acceleration. It's a mass matrix, just in a weird frame.
void ConstrainedGroupGradientMatrices::computeLCPConstraintMatrixClampingSubset(
    simulation::WorldPtr world, Eigen::MatrixXs& Q, const Eigen::MatrixXs& A_c)
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
    simulation::WorldPtr world, Eigen::VectorXs& b, const Eigen::MatrixXs& A_c)
{
  int nDofs = world->getNumDofs();
  Eigen::MatrixXs damp = Eigen::MatrixXs::Zero(nDofs,nDofs);
  for (int i=0;i<nDofs;i++)
  {
    damp(i,i) = world->getDofs()[i]->getDampingCoefficient();
  }
  //std::cout<<"Damping Matrix: \n"<<damp<<std::endl;
  b = -A_c.transpose()
      * (world->getVelocities()
         + (world->getTimeStep()
            * implicitMultiplyByInvMassMatrix(
                world,
                world->getControlForces()
                    - world->getCoriolisAndGravityAndExternalForces() - damp*world->getVelocities())));
}

//==============================================================================
/// This computes and returns an estimate of the constraint impulses for the
/// clamping constraints. This is based on a linear approximation of the
/// constraint impulses.
Eigen::VectorXs
ConstrainedGroupGradientMatrices::estimateClampingConstraintImpulses(
    simulation::WorldPtr world, const Eigen::MatrixXs& A_c)
{
  if (A_c.cols() == 0)
  {
    return Eigen::VectorXs::Zero(0);
  }

  Eigen::VectorXs b = Eigen::VectorXs::Zero(A_c.cols());
  Eigen::MatrixXs Q = Eigen::MatrixXs::Zero(A_c.cols(), A_c.cols());
  computeLCPOffsetClampingSubset(world, b, A_c);
  computeLCPConstraintMatrixClampingSubset(world, Q, A_c);

  return Q.completeOrthogonalDecomposition().solve(b);
}

//==============================================================================
/// This returns the jacobian of M^{-1}(pos, inertia) * tau, holding
/// everything constant except the value of WithRespectTo
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXs tau, WithRespectTo* wrt)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  int cursor = 0;
  for (int i = 0; i < mSkeletons.size(); i++)
  {
    auto skel = world->getSkeleton(mSkeletons[i]);
    int dofs = skel->getNumDofs();
    jac.block(cursor, cursor, dofs, dofs)
        = skel->getJacobianOfMinv(tau.segment(cursor, dofs), wrt);
    cursor += dofs;
  }
  return jac;
  return finiteDifferenceJacobianOfMinv(world, tau, wrt);
}

//==============================================================================
/// This returns the jacobian of C(pos, inertia, vel), holding everything
/// constant except the value of WithRespectTo
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t wrtDim = getWrtDim(world, wrt);

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mNumDOFs, wrtDim);

  int wrtCursor = 0;
  int dofCursor = 0;
  for (const std::string& skelName : mSkeletons)
  {
    auto skel = world->getSkeleton(skelName);
    int dofs = skel->getNumDofs();
    int skelWrtDim = wrt->dim(skel.get());
    J.block(dofCursor, wrtCursor, dofs, skelWrtDim) = skel->getJacobianOfC(wrt);
    wrtCursor += skelWrtDim;
    dofCursor += dofs;
  }

  return J;
}

//==============================================================================
/// This computes the Jacobian of A_c*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getJacobianOfClampingConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(world, skels);
  }

  return result;
}

//==============================================================================
/// This computes the Jacobian of A_c^T*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getJacobianOfClampingConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getClampingConstraints();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(constraints.size(), mNumDOFs);
  for (int i = 0; i < constraints.size(); i++)
  {
    result.row(i)
        = constraints[i]->getConstraintForcesJacobian(world, skels).transpose()
          * v0;
  }

  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub*E*f0 with respect to position using
/// impulse tests.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getJacobianOfUpperBoundConstraints(
    simulation::WorldPtr world, Eigen::VectorXs f0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  std::vector<std::shared_ptr<DifferentiableContactConstraint>> constraints
      = getUpperBoundConstraints();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(mNumDOFs, mNumDOFs);
  assert(constraints.size() == f0.size());
  for (int i = 0; i < constraints.size(); i++)
  {
    result += f0(i) * constraints[i]->getConstraintForcesJacobian(world, skels);
  }
  return result;
}

//==============================================================================
/// This computes the Jacobian of A_ub*E*v0 with respect to position using
/// impulse tests.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::getJacobianOfUpperBoundConstraintsTranspose(
    simulation::WorldPtr world, Eigen::VectorXs v0)
{
  std::vector<std::shared_ptr<dynamics::Skeleton>> skels = getSkeletons(world);

  int dofs = world->getNumDofs();
  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(mUpperBoundConstraints.size(), dofs);
  for (int i = 0; i < mUpperBoundConstraints.size(); i++)
  {
    result.row(i) = mUpperBoundConstraints[i]
                        ->getConstraintForcesJacobian(world)
                        .transpose()
                    * v0;
  }

  return result;
}

//==============================================================================
/// This replaces x with the result of M*x in place, without explicitly forming
/// M
Eigen::VectorXs ConstrainedGroupGradientMatrices::implicitMultiplyByMassMatrix(
    WorldPtr world, const Eigen::VectorXs& x)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getNumDOFs());
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
Eigen::VectorXs
ConstrainedGroupGradientMatrices::implicitMultiplyByInvMassMatrix(
    WorldPtr world, const Eigen::VectorXs& x)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getNumDOFs());
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
    const LossGradient& nextTimestepLoss,
    bool exploreAlternateStrategies)
{
  // First, we compute the correct gradients through backprop

  Eigen::MatrixXs forceVelJacobian = getControlForceVelJacobian(world);
  // p_t+1 <-- v_t
  Eigen::MatrixXs posVelJacobian = getPosVelJacobian(world);
  // v_t+1 <-- v_t
  Eigen::MatrixXs velVelJacobian = getVelVelJacobian(world);
  // v_t+1 <-- p_t
  Eigen::MatrixXs velPosJacobian = getVelPosJacobian(world);
  // p_t+1 <-- p_t
  Eigen::MatrixXs posPosJacobian = getPosPosJacobian(world);

  thisTimestepLoss.lossWrtPosition
      = posVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
        + posPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtVelocity
      = velVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
        + velPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtTorque
      = forceVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity;

  clipLossGradientsToBounds(
      world,
      thisTimestepLoss.lossWrtPosition,
      thisTimestepLoss.lossWrtVelocity,
      thisTimestepLoss.lossWrtTorque);

  if (exploreAlternateStrategies)
  {
    // Now that we have correct gradients, we can run our algorithm for
    // intelligently "unclamping" contacts as a heuristic to allow us to escape
    // from some saddle points. This will produce gradients that are
    // mathematically incorrect, but can provide a good search direction.

    // First, we need to compute the Jacobian of v_t+1 wrt contact force. This
    // is straightforward, M^{-1}*A. Then backprop with the transpose method.

    /*
    Eigen::MatrixXs jac = getMinv() * getAllConstraintMatrix();
    Eigen::VectorXs lossWrtContactForce
        = jac.transpose() * nextTimestepLoss.lossWrtVelocity;
    */

    Eigen::MatrixXs jac = getAllConstraintMatrix();
    Eigen::VectorXs lossWrtContactVels
        = jac.transpose() * nextTimestepLoss.lossWrtVelocity;

    // Now we're going to try using the `lossWrtContactForce` gradient to set
    // the state of each contact to clamping or separating, instead of the real
    // state of the world. This may help us escape saddle points, though it's
    // technically incorrect.

    Eigen::VectorXi overrideClasses
        = Eigen::VectorXi::Zero(lossWrtContactVels.size());
    for (int i = 0; i < lossWrtContactVels.size(); i++)
    {
      // If this is a frictional contact force
      if (mFIndex(i) != -1)
      {
        overrideClasses(i) = neural::ConstraintMapping::NOT_CLAMPING;
      }
      // If this is a normal contact force
      else
      {
        // If we want to increase contact vels (pulling further apart), let's
        // try this as not clamping
        if (lossWrtContactVels(i) < 0)
        {
          overrideClasses(i) = neural::ConstraintMapping::NOT_CLAMPING;
        }
        // If we want to decrease contact vels (pushing closer together), let's
        // try clamping
        else
        {
          overrideClasses(i) = neural::ConstraintMapping::CLAMPING;
        }
      }
    }
    constructMatrices(world.get(), overrideClasses);

    Eigen::MatrixXs stratForceVelJacobian = getControlForceVelJacobian(world);
    // p_t+1 <-- v_t
    Eigen::MatrixXs stratPosVelJacobian = getPosVelJacobian(world);
    // v_t+1 <-- v_t
    Eigen::MatrixXs stratVelVelJacobian = getVelVelJacobian(world);
    // v_t+1 <-- p_t
    Eigen::MatrixXs stratVelPosJacobian = getVelPosJacobian(world);
    // p_t+1 <-- p_t
    Eigen::MatrixXs stratPosPosJacobian = getPosPosJacobian(world);

    Eigen::VectorXs stratLossWrtPos
        = stratPosVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
          + stratPosPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
    Eigen::VectorXs stratLossWrtVel
        = stratVelVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity
          + stratVelPosJacobian.transpose() * nextTimestepLoss.lossWrtPosition;
    Eigen::VectorXs stratLossWrtTorque
        = stratForceVelJacobian.transpose() * nextTimestepLoss.lossWrtVelocity;

    clipLossGradientsToBounds(
        world, stratLossWrtPos, stratLossWrtVel, stratLossWrtTorque);

    // If this is a more promising direction to pursue, measured by the rate of
    // improvement in the loss, then let's use this search direction instead.
    s_t stratMagnitude = stratLossWrtVel.norm()
                         + (stratLossWrtTorque.norm() / world->getTimeStep());
    s_t realMagnitude
        = thisTimestepLoss.lossWrtVelocity.norm()
          + (thisTimestepLoss.lossWrtTorque.norm() / world->getTimeStep());
    if (stratMagnitude > realMagnitude)
    {
      thisTimestepLoss.lossWrtPosition = stratLossWrtPos;
      thisTimestepLoss.lossWrtVelocity = stratLossWrtVel;
      thisTimestepLoss.lossWrtTorque = stratLossWrtTorque;
    }

    // Reset the matrices, just in case people ask for Jacobians or something
    // later, don't want to give them the wrong one.
    constructMatrices(world.get());
  }

  /*
  // Compute intermediate variable "b", described in the doc
  Eigen::VectorXs b = nextTimestepLoss.lossWrtPosition;
  if (mBouncingConstraintMatrix.size() > 0)
  {
    Eigen::MatrixXs X = getPosPosJacobian().transpose();
    b = X * nextTimestepLoss.lossWrtPosition;
  }

  // Compute intermediate variable "x", described in the doc
  Eigen::VectorXs x = implicitMultiplyByInvMassMatrix(
      world, nextTimestepLoss.lossWrtVelocity);

  // Get the massed clamping constraints
  Eigen::MatrixXs V_c = getMassedClampingConstraintMatrix();

  // Initialize the intermediate variable "A_c * z", described in the doc
  Eigen::VectorXs A_c_z = Eigen::VectorXs::Zero(x.size());

  // If there are clamping constraints:
  if (V_c.size() > 0)
  {
    Eigen::MatrixXs A_c = getClampingConstraintMatrix();
    Eigen::MatrixXs V_c = getMassedClampingConstraintMatrix();
    Eigen::MatrixXs V_ub = getMassedUpperBoundConstraintMatrix();
    Eigen::MatrixXs E = getUpperBoundMappingMatrix();

    Eigen::MatrixXs constraintForceToImpliedTorques = V_c + (V_ub * E);
    Eigen::MatrixXs forceToVelTranspose
        = constraintForceToImpliedTorques.transpose() * A_c;
    auto velToForceTranspose
        = forceToVelTranspose.completeOrthogonalDecomposition();

    // (V_c + (V_ub * E))^T * nextTimestepLoss.lossWrtVelocity
    Eigen::VectorXs tmp = constraintForceToImpliedTorques.transpose()
                          * nextTimestepLoss.lossWrtVelocity;
    // (A_c^T(V_c + (V_ub * E))^T).pinv() * tmp
    Eigen::VectorXs z = velToForceTranspose.solve(tmp);
    // z = B * z
    z = mBounceDiagonals.cwiseProduct(z);

    x -= V_c * z;

    // Cache the result of "A_c * z"
    A_c_z = A_c * z;
  }
  x *= mTimeStep;

  Eigen::MatrixXs posCJacobianTranspose = getPosCJacobian(world).transpose();
  Eigen::MatrixXs velCJacobianTranspose = getVelCJacobian(world).transpose();

  thisTimestepLoss.lossWrtTorque = x;
  thisTimestepLoss.lossWrtPosition = b - (posCJacobianTranspose * x);
  thisTimestepLoss.lossWrtVelocity = (mTimeStep * b)
                                     + nextTimestepLoss.lossWrtVelocity - A_c_z
                                     - (velCJacobianTranspose * x);
                                     */
}

//==============================================================================
void ConstrainedGroupGradientMatrices::clipLossGradientsToBounds(
    simulation::WorldPtr world,
    Eigen::VectorXs& lossWrtPos,
    Eigen::VectorXs& lossWrtVel,
    Eigen::VectorXs& lossWrtForce)
{
  int cursor = 0;
  for (int i = 0; i < mSkeletons.size(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel
        = world->getSkeleton(mSkeletons[i]);
    for (int j = 0; j < skel->getNumDofs(); j++)
    {
      // Clip position gradients

      if ((skel->getPosition(j) == skel->getPositionLowerLimit(j))
          && (lossWrtPos(cursor) > 0))
      {
        lossWrtPos(cursor) = 0;
      }
      if ((skel->getPosition(j) == skel->getPositionUpperLimit(j))
          && (lossWrtPos(cursor) < 0))
      {
        lossWrtPos(cursor) = 0;
      }

      // Clip velocity gradients

      if ((skel->getVelocity(j) == skel->getVelocityLowerLimit(j))
          && (lossWrtVel(cursor) > 0))
      {
        lossWrtVel(cursor) = 0;
      }
      if ((skel->getVelocity(j) == skel->getVelocityUpperLimit(j))
          && (lossWrtVel(cursor) < 0))
      {
        lossWrtVel(cursor) = 0;
      }

      // Clip force gradients

      if ((skel->getControlForce(j) == skel->getControlForceLowerLimit(j))
          && (lossWrtForce(cursor) > 0))
      {
        lossWrtForce(cursor) = 0;
      }
      if ((skel->getControlForce(j) == skel->getControlForceUpperLimit(j))
          && (lossWrtForce(cursor) < 0))
      {
        lossWrtForce(cursor) = 0;
      }

      cursor++;
    }
  }
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getAllConstraintMatrix() const
{
  return mAllConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getClampingConstraintMatrix() const
{
  return mClampingConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getMassedClampingConstraintMatrix() const
{
  return mMassedClampingConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getUpperBoundConstraintMatrix() const
{
  return mUpperBoundConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getMassedUpperBoundConstraintMatrix() const
{
  return mMassedUpperBoundConstraintMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getUpperBoundMappingMatrix() const
{
  return mUpperBoundMappingMatrix;
}

//==============================================================================
const Eigen::MatrixXs&
ConstrainedGroupGradientMatrices::getBouncingConstraintMatrix() const
{
  return mBouncingConstraintMatrix;
}

//==============================================================================
/// These was the mX() vector used to construct this. Pretty much only here
/// for testing.
const Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getContactConstraintImpulses() const
{
  return mX;
}

//==============================================================================
const Eigen::VectorXs& ConstrainedGroupGradientMatrices::getBounceDiagonals()
    const
{
  return mBounceDiagonals;
}

//==============================================================================
const Eigen::VectorXs&
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
const Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getPenetrationCorrectionVelocities() const
{
  return mPenetrationCorrectionVelocitiesVec;
}

//==============================================================================
/// This is the subset of the A matrix from the original LCP that corresponds
/// to clamping indices.
const Eigen::MatrixXs& ConstrainedGroupGradientMatrices::getClampingAMatrix()
    const
{
  return mClampingAMatrix;
}

//==============================================================================
const Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getClampingConstraintImpulses() const
{
  return mClampingConstraintImpulses;
}

//==============================================================================
/// Returns the relative velocities along the clamping constraints
const Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getClampingConstraintRelativeVels() const
{
  return mClampingConstraintRelativeVels;
}

//==============================================================================
/// Returns the velocity change caused by the illegal impulses from the LCP
const Eigen::VectorXs&
ConstrainedGroupGradientMatrices::getVelocityDueToIllegalImpulses() const
{
  return mVelocityDueToIllegalImpulses;
}

//==============================================================================
/// Returns the torques applied pre-step
const Eigen::VectorXs& ConstrainedGroupGradientMatrices::getPreStepTorques()
    const
{
  return mPreStepTorques;
}

//==============================================================================
/// Returns the velocity pre-step
const Eigen::VectorXs& ConstrainedGroupGradientMatrices::getPreStepVelocity()
    const
{
  return mPreStepVelocities;
}

//==============================================================================
/// Returns the velocity pre-LCP
const Eigen::VectorXs& ConstrainedGroupGradientMatrices::getPreLCPVelocity()
    const
{
  return mPreLCPVelocities;
}

//==============================================================================
/// Returns the M^{-1} matrix from pre-step
const Eigen::MatrixXs& ConstrainedGroupGradientMatrices::getMinv() const
{
  return mMinv;
}

//==============================================================================
/// Get the coriolis and gravity forces
const Eigen::VectorXs
ConstrainedGroupGradientMatrices::getCoriolisAndGravityAndExternalForces(
    simulation::WorldPtr world) const
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mNumDOFs);
  int cursor = 0;
  for (std::string skelName : mSkeletons)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(skelName);
    int dofs = skel->getNumDofs();
    result.segment(cursor, dofs)
        = skel->getCoriolisAndGravityForces() - skel->getExternalForces();
    cursor += dofs;
  }
  return result;
}

//==============================================================================
Eigen::MatrixXs ConstrainedGroupGradientMatrices::getFullConstraintMatrix(
    simulation::World* world) const
{
  Eigen::MatrixXs impulses = Eigen::MatrixXs::Zero(
      world->getNumDofs(), mDifferentiableConstraints.size());
  for (int i = 0; i < mDifferentiableConstraints.size(); i++)
  {
    impulses.col(i) = mDifferentiableConstraints[i]->getConstraintForces(world);
  }
  return impulses;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// finite differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::finiteDifferenceJacobianOfMinv(
    simulation::WorldPtr world,
    Eigen::VectorXs tau,
    WithRespectTo* wrt,
    bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfMinv(world, tau, wrt);

  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs before = getWrt(world, wrt);

  const s_t EPS = 5e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXs newVPlus = implicitMultiplyByInvMassMatrix(world, tau);
    perturbed = before;
    perturbed(i) -= EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXs newVMinus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXs diff = newVPlus - newVMinus;
    result.col(i) = diff / (2 * EPS);
  }

  setWrt(world, wrt, before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of M^{-1}(pos, inertia) * tau by
/// finite differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::finiteDifferenceRiddersJacobianOfMinv(
    simulation::WorldPtr world, Eigen::VectorXs tau, WithRespectTo* wrt)
{
  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = implicitMultiplyByInvMassMatrix(world, tau);

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs originalWrt = getWrt(world, wrt);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    setWrt(world, wrt, perturbedPlus);
    Eigen::MatrixXs MinvTauPlus = implicitMultiplyByInvMassMatrix(world, tau);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    setWrt(world, wrt, perturbedMinus);
    Eigen::MatrixXs MinvTauMinus = implicitMultiplyByInvMassMatrix(world, tau);

    tab[0][0] = (MinvTauPlus - MinvTauMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      setWrt(world, wrt, perturbedPlus);
      MinvTauPlus = implicitMultiplyByInvMassMatrix(world, tau);
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      setWrt(world, wrt, perturbedMinus);
      MinvTauMinus = implicitMultiplyByInvMassMatrix(world, tau);

      tab[0][iTab] = (MinvTauPlus - MinvTauMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  setWrt(world, wrt, originalWrt);

  return J;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by finite
/// differences.
Eigen::MatrixXs ConstrainedGroupGradientMatrices::finiteDifferenceJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt, bool useRidders)
{
  if (useRidders)
    return finiteDifferenceRiddersJacobianOfC(world, wrt);

  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = getCoriolisAndGravityAndExternalForces(world);

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs before = getWrt(world, wrt);

  const s_t EPS = 1e-7;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    Eigen::VectorXs perturbed = before;
    perturbed(i) += EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXs tauPos = getCoriolisAndGravityAndExternalForces(world);
    perturbed = before;
    perturbed(i) -= EPS;
    setWrt(world, wrt, perturbed);
    Eigen::MatrixXs tauNeg = getCoriolisAndGravityAndExternalForces(world);
    Eigen::VectorXs diff = tauPos - tauNeg;
    result.col(i) = diff / (2 * EPS);
  }

  setWrt(world, wrt, before);

  return result;
}

//==============================================================================
/// This computes and returns the jacobian of C(pos, inertia, vel) by finite
/// differences.
Eigen::MatrixXs
ConstrainedGroupGradientMatrices::finiteDifferenceRiddersJacobianOfC(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  std::size_t innerDim = getWrtDim(world, wrt);

  // These are predicted contact forces at the clamping contacts
  Eigen::VectorXs original = getCoriolisAndGravityAndExternalForces(world);

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(original.size(), innerDim);

  Eigen::VectorXs originalWrt = getWrt(world, wrt);

  const s_t originalStepSize = 1e-3;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < innerDim; i++)
  {
    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    Eigen::VectorXs perturbedPlus = Eigen::VectorXs(originalWrt);
    perturbedPlus(i) += stepSize;
    setWrt(world, wrt, perturbedPlus);
    Eigen::MatrixXs tauPlus = getCoriolisAndGravityAndExternalForces(world);
    Eigen::VectorXs perturbedMinus = Eigen::VectorXs(originalWrt);
    perturbedMinus(i) -= stepSize;
    setWrt(world, wrt, perturbedMinus);
    Eigen::MatrixXs tauMinus = getCoriolisAndGravityAndExternalForces(world);

    tab[0][0] = (tauPlus - tauMinus) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      perturbedPlus = Eigen::VectorXs(originalWrt);
      perturbedPlus(i) += stepSize;
      setWrt(world, wrt, perturbedPlus);
      tauPlus = getCoriolisAndGravityAndExternalForces(world);
      perturbedMinus = Eigen::VectorXs(originalWrt);
      perturbedMinus(i) -= stepSize;
      setWrt(world, wrt, perturbedMinus);
      tauMinus = getCoriolisAndGravityAndExternalForces(world);

      tab[0][iTab] = (tauPlus - tauMinus) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  setWrt(world, wrt, originalWrt);

  return J;
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
Eigen::VectorXs ConstrainedGroupGradientMatrices::getWrt(
    simulation::WorldPtr world, WithRespectTo* wrt)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getWrtDim(world, wrt));
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
    simulation::WorldPtr world, WithRespectTo* wrt, Eigen::VectorXs v)
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