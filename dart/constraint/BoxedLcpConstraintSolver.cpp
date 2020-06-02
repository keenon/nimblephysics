/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "dart/constraint/BoxedLcpConstraintSolver.hpp"

#include <cassert>
#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include "dart/common/Console.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/DantzigBoxedLcpSolver.hpp"
#include "dart/constraint/PgsBoxedLcpSolver.hpp"
#include "dart/external/odelcpsolver/lcp.h"
#include "dart/lcpsolver/Lemke.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

namespace dart {
namespace constraint {

//==============================================================================
BoxedLcpConstraintSolver::BoxedLcpConstraintSolver(
    double timeStep,
    BoxedLcpSolverPtr boxedLcpSolver,
    BoxedLcpSolverPtr secondaryBoxedLcpSolver)
  : BoxedLcpConstraintSolver(
      std::move(boxedLcpSolver), std::move(secondaryBoxedLcpSolver))
{
  setTimeStep(timeStep);
}

//==============================================================================
BoxedLcpConstraintSolver::BoxedLcpConstraintSolver()
  : BoxedLcpConstraintSolver(std::make_shared<DantzigBoxedLcpSolver>())
{
  // Do nothing
}

//==============================================================================
BoxedLcpConstraintSolver::BoxedLcpConstraintSolver(
    BoxedLcpSolverPtr boxedLcpSolver)
  : BoxedLcpConstraintSolver(
      std::move(boxedLcpSolver), std::make_shared<PgsBoxedLcpSolver>())
{
  // Do nothing
}

//==============================================================================
BoxedLcpConstraintSolver::BoxedLcpConstraintSolver(
    BoxedLcpSolverPtr boxedLcpSolver, BoxedLcpSolverPtr secondaryBoxedLcpSolver)
  : ConstraintSolver()
{
  if (boxedLcpSolver)
  {
    setBoxedLcpSolver(std::move(boxedLcpSolver));
  }
  else
  {
    dtwarn << "[BoxedLcpConstraintSolver] Attempting to construct with nullptr "
           << "LCP solver, which is not allowed. Using Dantzig solver "
           << "instead.\n";
    setBoxedLcpSolver(std::make_shared<DantzigBoxedLcpSolver>());
  }

  setSecondaryBoxedLcpSolver(std::move(secondaryBoxedLcpSolver));
}

//==============================================================================
void BoxedLcpConstraintSolver::setBoxedLcpSolver(BoxedLcpSolverPtr lcpSolver)
{
  if (!lcpSolver)
  {
    dtwarn << "[BoxedLcpConstraintSolver::setBoxedLcpSolver] "
           << "nullptr for boxed LCP solver is not allowed.";
    return;
  }

  if (lcpSolver == mSecondaryBoxedLcpSolver)
  {
    dtwarn << "[BoxedLcpConstraintSolver::setBoxedLcpSolver] Attempting to set "
           << "a primary LCP solver that is the same with the secondary LCP "
           << "solver, which is discouraged. Ignoring this request.\n";
  }

  mBoxedLcpSolver = std::move(lcpSolver);
}

//==============================================================================
ConstBoxedLcpSolverPtr BoxedLcpConstraintSolver::getBoxedLcpSolver() const
{
  return mBoxedLcpSolver;
}

//==============================================================================
void BoxedLcpConstraintSolver::setSecondaryBoxedLcpSolver(
    BoxedLcpSolverPtr lcpSolver)
{
  if (lcpSolver == mBoxedLcpSolver)
  {
    dtwarn << "[BoxedLcpConstraintSolver::setBoxedLcpSolver] Attempting to set "
           << "the secondary LCP solver that is identical to the primary LCP "
           << "solver, which is redundant. Please use different solvers or set "
           << "the secondary LCP solver to nullptr.\n";
  }

  mSecondaryBoxedLcpSolver = std::move(lcpSolver);
}

//==============================================================================
ConstBoxedLcpSolverPtr BoxedLcpConstraintSolver::getSecondaryBoxedLcpSolver()
    const
{
  return mSecondaryBoxedLcpSolver;
}

//==============================================================================
void BoxedLcpConstraintSolver::solveConstrainedGroup(ConstrainedGroup& group)
{
  // Build LCP terms by aggregating them from constraints
  const std::size_t numConstraints = group.getNumConstraints();
  const std::size_t n = group.getTotalDimension();

  // If there is no constraint, then just return.
  if (0u == n)
    return;

  const int nSkip = dPAD(n); // nSkip = n + (n % 4);
#ifdef NDEBUG                // release
  mA.resize(n, nSkip);
#else // debug
  mA.setZero(n, nSkip); // rows = n, cols = n + (n % 4)
#endif
  mX.resize(n);
  mB.resize(n);
  mW.setZero(n); // set w to 0
  mLo.resize(n);
  mHi.resize(n);
  mFIndex.setConstant(n, -1); // set findex to -1

  // Compute offset indices
  mOffset.resize(n);
  mOffset[0] = 0;
  for (std::size_t i = 1; i < numConstraints; ++i)
  {
    const ConstraintBasePtr& constraint = group.getConstraint(i - 1);
    assert(constraint->getDimension() > 0);
    mOffset[i] = mOffset[i - 1] + constraint->getDimension();
  }

  // For gradient computations:
  // Create the data structure to save our impules test results. The
  // index into skeletonsImpulseTests[k] is a vector of Eigen::VectorXd
  // for the k'th skeleton, corresponding to each dimension of the constraint
  // group getting a unit impules applied.
  std::vector<std::vector<Eigen::VectorXd> > skeletonsImpulseTorqueTests;
  skeletonsImpulseTorqueTests.reserve(mSkeletons.size());
  for (auto skeleton : mSkeletons)
  {
    std::vector<Eigen::VectorXd> skeletonImpulseTorqueTests;
    skeletonImpulseTorqueTests.reserve(n);
    skeletonsImpulseTorqueTests.push_back(skeletonImpulseTorqueTests);
  }

  // For gradient computations (massed formulation):
  // Create the data structure to save our impules test results. The
  // index into skeletonsImpulseTests[k] is a vector of Eigen::VectorXd
  // for the k'th skeleton, corresponding to each dimension of the constraint
  // group getting a unit impules applied.
  std::vector<std::vector<Eigen::VectorXd> > skeletonsImpulseVelocityTests;
  skeletonsImpulseVelocityTests.reserve(mSkeletons.size());
  for (auto skeleton : mSkeletons)
  {
    std::vector<Eigen::VectorXd> skeletonImpulseVelocityTests;
    skeletonImpulseVelocityTests.reserve(n);
    skeletonsImpulseVelocityTests.push_back(skeletonImpulseVelocityTests);
  }

  // For each constraint
  ConstraintInfo constInfo;
  constInfo.invTimeStep = 1.0 / mTimeStep;
  for (std::size_t i = 0; i < numConstraints; ++i)
  {
    const ConstraintBasePtr& constraint = group.getConstraint(i);

    constInfo.x = mX.data() + mOffset[i];
    constInfo.lo = mLo.data() + mOffset[i];
    constInfo.hi = mHi.data() + mOffset[i];
    constInfo.b = mB.data() + mOffset[i];
    constInfo.findex = mFIndex.data() + mOffset[i];
    constInfo.w = mW.data() + mOffset[i];

    // Fill vectors: lo, hi, b, w
    constraint->getInformation(&constInfo);

    // Fill a matrix by impulse tests: A
    constraint->excite();

    double* impulses = new double[constraint->getDimension()];
    for (std::size_t j = 0; j < constraint->getDimension(); ++j)
    {
      // Adjust findex for global index
      if (mFIndex[mOffset[i] + j] >= 0)
        mFIndex[mOffset[i] + j] += mOffset[i];

      //////////////////////////////////////////////////////////////
      // Gradients - Classic formulation
      //////////////////////////////////////////////////////////////

      // For gradient comptutations: clear constraint impulses
      for (std::size_t k = 0; k < mSkeletons.size(); k++)
      {
        mSkeletons[k]->clearConstraintImpulses();
      }
      for (std::size_t k = 0; k < constraint->getDimension(); ++k)
        impulses[k] = (k == j) ? 1 : 0;
      constraint->applyImpulse(impulses);
      // For gradient computations: record the torque changes for each
      // skeleton for the unit impulse on this constraint.
      // TODO(keenon): In large scenes with many skeletons, most of these
      // vectors will be 0 for any given constraint, so we should filter them
      // out.
      for (std::size_t k = 0; k < mSkeletons.size(); k++)
      {
        Eigen::VectorXd impulseTorqueChange
            = mSkeletons[k]->getConstraintForces() * mTimeStep;
        skeletonsImpulseTorqueTests[k].push_back(impulseTorqueChange);
        mSkeletons[k]->clearConstraintImpulses();
      }

      //////////////////////////////////////////////////////////////
      // END: Gradients - classic formulation
      //////////////////////////////////////////////////////////////

      // Apply impulse for mipulse test
      constraint->applyUnitImpulse(j);

      //////////////////////////////////////////////////////////////
      // Gradients - Massed formulation
      //////////////////////////////////////////////////////////////

      for (std::size_t k = 0; k < mSkeletons.size(); k++)
      {
        // TODO(keenon): In large scenes with many skeletons, most of these
        // vectors will be 0 for any given constraint, so we should filter them
        // out.
        skeletonsImpulseVelocityTests[k].push_back(
            mSkeletons[k]->getVelocityChanges());
      }

      //////////////////////////////////////////////////////////////
      // END Gradients - Massed formulation
      //////////////////////////////////////////////////////////////

      // Fill upper triangle blocks of A matrix

      // mA is row-major order, n rows by nSkip cols
      // nSkip * (mOffset[i] + j) takes us to the (mOffset[i] + j)'th row
      // mOffset[i] into that row
      //
      // -------------------------------
      //                                |
      //                   nSkip * (mOffset[i] + j)
      //                                |
      //                                v
      // ------- mOffset[i] ----------> xxxxxxxxxx
      //
      // This whole loop fills the entire row (mOffset[i] + j) of A with
      // the effect that the unit impluse on constraint for j for this
      // constraint has on the relative velocities for each constraint
      // force direction.
      //
      // As an efficiency tweak, since you know that A is symmetric, only
      // bother to actually compute half of the velocity changes (upper
      // triangle, arbitrarily) and then just copy that into the other half.

      // Create a 3x3 square from A(mOffset[i], mOffset[i]) iterating over j
      // This iteration fill in row j
      int index = nSkip * (mOffset[i] + j) + mOffset[i];
      constraint->getVelocityChange(mA.data() + index, true);

      for (std::size_t k = i + 1; k < numConstraints; ++k)
      {
        // Create a 3x3 square from A(mOffset[i], mOffset[k]), iterating over j
        // This iteration fill in row j
        // Probably mostly 0s
        index = nSkip * (mOffset[i] + j) + mOffset[k];
        group.getConstraint(k)->getVelocityChange(mA.data() + index, false);
      }

      // Filling symmetric part of A matrix
      for (std::size_t k = 0; k < i; ++k)
      {
        const int indexI = mOffset[i] + j;
        for (std::size_t l = 0; l < group.getConstraint(k)->getDimension(); ++l)
        {
          const int indexJ = mOffset[k] + l;
          // We've already calculate the velocity of
          // mA(column for this constraint, previous constraint row) =
          //     mA(previous constraint row, column for this constraint)
          mA(indexI, indexJ) = mA(indexJ, indexI);
        }
      }

      if (group.getGradientConstraintMatrices())
      {
        group.getGradientConstraintMatrices()->measureConstraintImpulse(
            constraint, j);
      }
    }
    delete impulses;

    assert(isSymmetric(
        n, mA.data(), mOffset[i], mOffset[i] + constraint->getDimension() - 1));

    constraint->unexcite();
  }

  assert(isSymmetric(n, mA.data()));

  // Print LCP formulation
  /*
  dtdbg << "Before solve:" << std::endl;
  print(
      n,
      mA.data(),
      mX.data(),
      mLo.data(),
      mHi.data(),
      mB.data(),
      mW.data(),
      mFIndex.data());
  std::cout << std::endl;
  */

  // Solve LCP using the primary solver and fallback to secondary solver when
  // the parimary solver failed.
  if (mSecondaryBoxedLcpSolver)
  {
    // Make backups for the secondary LCP solver because the primary solver
    // modifies the original terms.
    mABackup = mA;
    mXBackup = mX;
    mBBackup = mB;
    mLoBackup = mLo;
    mHiBackup = mHi;
    mFIndexBackup = mFIndex;
  }
  // Always make backups of these variables, regardless of whether we're using
  // a secondary solver, because we need them for gradients
  Eigen::VectorXd loGradientBackup = mLo;
  Eigen::VectorXd hiGradientBackup = mHi;
  Eigen::VectorXi fIndexGradientBackup = mFIndex;

  const bool earlyTermination = (mSecondaryBoxedLcpSolver != nullptr);
  assert(mBoxedLcpSolver);
  bool success = mBoxedLcpSolver->solve(
      n,
      mA.data(),
      mX.data(),
      mB.data(),
      0,
      mLo.data(),
      mHi.data(),
      mFIndex.data(),
      earlyTermination);

  // Sanity check. LCP solvers should not report success with nan values, but
  // it could happen. So we set the sucees to false for nan values.
  if (success && mX.hasNaN())
    success = false;

  if (!success && mSecondaryBoxedLcpSolver)
  {
    mSecondaryBoxedLcpSolver->solve(
        n,
        mABackup.data(),
        mXBackup.data(),
        mBBackup.data(),
        0,
        mLoBackup.data(),
        mHiBackup.data(),
        mFIndexBackup.data(),
        false);
    mX = mXBackup;
  }

  if (mX.hasNaN())
  {
    dterr << "[BoxedLcpConstraintSolver] The solution of LCP includes NAN "
          << "values: " << mX.transpose() << ". We're setting it zero for "
          << "safety. Consider using more robust solver such as PGS as a "
          << "secondary solver. If this happens even with PGS solver, please "
          << "report this as a bug.\n";
    mX.setZero();
  }

  // Print LCP formulation
  //  dtdbg << "After solve:" << std::endl;
  //  print(n, A, x, lo, hi, b, w, findex);
  //  std::cout << std::endl;

  // Apply constraint impulses
  for (std::size_t i = 0; i < numConstraints; ++i)
  {
    const ConstraintBasePtr& constraint = group.getConstraint(i);
    constraint->applyImpulse(mX.data() + mOffset[i]);
    constraint->excite();
  }

  if (group.getGradientConstraintMatrices())
  {
    group.getGradientConstraintMatrices()->constructMatrices(
        mX, hiGradientBackup, loGradientBackup, fIndexGradientBackup);
  }

  // For gradient computations:
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

  Eigen::VectorXi contactConstraintMappings = Eigen::VectorXi(n);
  int* clampingIndex = new int[n];
  int* upperBoundIndex = new int[n];

  for (std::size_t i = 0; i < mSkeletons.size(); ++i)
  {
    int numClamping = 0;
    int numUpperBound = 0;
    // Fill in mappings[] with the correct values, overwriting previous data
    for (std::size_t j = 0; j < n; j++)
    {
      // If the Eigen::VectorXd representing the impulse test is of length 0,
      // that means that constraint "j" doesn't effect skeleton "i".
      if (skeletonsImpulseTorqueTests[i][j].size() == 0
          || skeletonsImpulseTorqueTests[i][j].isZero())
      {
        contactConstraintMappings(j) = neural::ConstraintMapping::IRRELEVANT;
        continue;
      }
      const double constraintForce = mX(j);

      // If constraintForce is zero, this means "j" is in "Not Clamping"
      if (std::abs(constraintForce) < 1e-9)
      {
        contactConstraintMappings(j) = neural::ConstraintMapping::NOT_CLAMPING;
        continue;
      }

      double upperBound = hiGradientBackup(j);
      double lowerBound = loGradientBackup(j);
      const int fIndex = fIndexGradientBackup(j);
      if (fIndex != -1)
      {
        upperBound *= mX(fIndex);
        lowerBound *= mX(fIndex);
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
      else if (fIndex != -1)
      {
        /*
        std::cout << "Listing " << j << " as UB: mX=" << mX(j)
                  << ", fIndex=" << fIndex << ", mX(fIndex)=" << mX(fIndex)
                  << ", hiBackup=" << hiGradientBackup(j)
                  << ", loBackup=" << loGradientBackup(j)
                  << ", upperBound=" << upperBound
                  << ", lowerBound=" << lowerBound << std::endl;
        */
        contactConstraintMappings(j) = fIndex;
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

    int dofs = mSkeletons[i]->getNumDofs();
    // Create the matrices we want to pass along for this skeleton:
    Eigen::MatrixXd clampingConstraintMatrix(dofs, numClamping);
    Eigen::MatrixXd upperBoundConstraintMatrix(dofs, numUpperBound);
    Eigen::MatrixXd upperBoundMappingMatrix(numUpperBound, numClamping);
    // Massed formulation
    Eigen::MatrixXd massedClampingConstraintMatrix(dofs, numClamping);
    Eigen::MatrixXd massedUpperBoundConstraintMatrix(dofs, numUpperBound);

    // We only need to zero out the mapping matrix, the other two will get
    // completely overwritten in the next loop
    upperBoundMappingMatrix.setZero();

    // Copy values into our new matrices
    for (size_t j = 0; j < n; j++)
    {
      if (contactConstraintMappings(j) == neural::ConstraintMapping::CLAMPING)
      {
        assert(numClamping > clampingIndex[j]);
        clampingConstraintMatrix.col(
            clampingIndex[j]) // .block(0, clampingIndex[j], dofs, 1)
            = skeletonsImpulseTorqueTests[i][j];
        massedClampingConstraintMatrix.col(clampingIndex[j])
            = skeletonsImpulseVelocityTests[i][j];
      }
      else if (contactConstraintMappings(j) >= 0) // means we're an UPPER_BOUND
      {
        assert(numUpperBound > upperBoundIndex[j]);
        upperBoundConstraintMatrix.col(
            upperBoundIndex[j]) // .block(0, upperBoundIndex[j], dofs, 1)
            = skeletonsImpulseTorqueTests[i][j];
        massedUpperBoundConstraintMatrix.col(upperBoundIndex[j])
            = skeletonsImpulseVelocityTests[i][j];

        // Figure out, and write the correct coefficient to E

        const int fIndex = contactConstraintMappings(j);
        const double upperBound = mX(fIndex) * hiGradientBackup(j);
        const double lowerBound = mX(fIndex) * loGradientBackup(j);

        // If we're clamped at the upper bound
        if (std::abs(mX(j) - upperBound) < std::abs(mX(j) - lowerBound))
        {
          if (std::abs(mX(j) - upperBound) > 1e-5)
          {
            std::cout << "Lower bound: " << lowerBound << std::endl;
            std::cout << "Upper bound: " << upperBound << std::endl;
            std::cout << "mHi(j): " << hiGradientBackup(j) << std::endl;
            std::cout << "mLo(j): " << loGradientBackup(j) << std::endl;
            std::cout << "mX(j): " << mX(j) << std::endl;
            std::cout << "fIndex: " << fIndex << std::endl;
          }
          assert(std::abs(mX(j) - upperBound) < 1e-5);
          upperBoundMappingMatrix(upperBoundIndex[j], clampingIndex[fIndex])
              = hiGradientBackup(j);
        }
        // If we're clamped at the lower bound
        else
        {
          if (std::abs(mX(j) - lowerBound) > 1e-5)
          {
            std::cout << "Lower bound: " << lowerBound << std::endl;
            std::cout << "Upper bound: " << upperBound << std::endl;
            std::cout << "mHi(j): " << hiGradientBackup(j) << std::endl;
            std::cout << "mLo(j): " << loGradientBackup(j) << std::endl;
            std::cout << "mX(j): " << mX(j) << std::endl;
            std::cout << "fIndex: " << fIndex << std::endl;
          }
          assert(std::abs(mX(j) - lowerBound) < 1e-5);
          upperBoundMappingMatrix(upperBoundIndex[j], clampingIndex[fIndex])
              = loGradientBackup(j);
        }
      }
    }

    mSkeletons[i]->setConstraintMatricesForGradient(
        clampingConstraintMatrix,
        upperBoundConstraintMatrix,
        upperBoundMappingMatrix,
        contactConstraintMappings,
        mX);

    mSkeletons[i]->setMassedConstraintMatricesForGradient(
        massedClampingConstraintMatrix,
        massedUpperBoundConstraintMatrix,
        upperBoundMappingMatrix,
        contactConstraintMappings,
        mX);
  }

  delete clampingIndex;
  delete upperBoundIndex;
}

//==============================================================================
#ifndef NDEBUG
bool BoxedLcpConstraintSolver::isSymmetric(std::size_t n, double* A)
{
  std::size_t nSkip = dPAD(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = 0; j < n; ++j)
    {
      if (std::abs(A[nSkip * i + j] - A[nSkip * j + i]) > 1e-6)
      {
        std::cout << "A: " << std::endl;
        for (std::size_t k = 0; k < n; ++k)
        {
          for (std::size_t l = 0; l < nSkip; ++l)
          {
            std::cout << std::setprecision(4) << A[k * nSkip + l] << " ";
          }
          std::cout << std::endl;
        }

        std::cout << "A(" << i << ", " << j << "): " << A[nSkip * i + j]
                  << std::endl;
        std::cout << "A(" << j << ", " << i << "): " << A[nSkip * j + i]
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

//==============================================================================
bool BoxedLcpConstraintSolver::isSymmetric(
    std::size_t n, double* A, std::size_t begin, std::size_t end)
{
  std::size_t nSkip = dPAD(n);
  for (std::size_t i = begin; i <= end; ++i)
  {
    for (std::size_t j = begin; j <= end; ++j)
    {
      if (std::abs(A[nSkip * i + j] - A[nSkip * j + i]) > 1e-6)
      {
        std::cout << "A: " << std::endl;
        for (std::size_t k = 0; k < n; ++k)
        {
          for (std::size_t l = 0; l < nSkip; ++l)
          {
            std::cout << std::setprecision(4) << A[k * nSkip + l] << " ";
          }
          std::cout << std::endl;
        }

        std::cout << "A(" << i << ", " << j << "): " << A[nSkip * i + j]
                  << std::endl;
        std::cout << "A(" << j << ", " << i << "): " << A[nSkip * j + i]
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

//==============================================================================
void BoxedLcpConstraintSolver::print(
    std::size_t n,
    double* A,
    double* x,
    double* /*lo*/,
    double* /*hi*/,
    double* b,
    double* w,
    int* findex)
{
  std::size_t nSkip = dPAD(n);
  std::cout << "A: " << std::endl;
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = 0; j < nSkip; ++j)
    {
      std::cout << std::setprecision(4) << A[i * nSkip + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "b: ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << std::setprecision(4) << b[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "w: ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << w[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "x: ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;

  //  std::cout << "lb: ";
  //  for (int i = 0; i < dim; ++i)
  //  {
  //    std::cout << lb[i] << " ";
  //  }
  //  std::cout << std::endl;

  //  std::cout << "ub: ";
  //  for (int i = 0; i < dim; ++i)
  //  {
  //    std::cout << ub[i] << " ";
  //  }
  //  std::cout << std::endl;

  std::cout << "frictionIndex: ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << findex[i] << " ";
  }
  std::cout << std::endl;

  double* Ax = new double[n];

  for (std::size_t i = 0; i < n; ++i)
  {
    Ax[i] = 0.0;
  }

  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = 0; j < n; ++j)
    {
      Ax[i] += A[i * nSkip + j] * x[j];
    }
  }

  std::cout << "Ax   : ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << Ax[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "b + w: ";
  for (std::size_t i = 0; i < n; ++i)
  {
    std::cout << b[i] + w[i] << " ";
  }
  std::cout << std::endl;

  delete[] Ax;
}
#endif

} // namespace constraint
} // namespace dart
