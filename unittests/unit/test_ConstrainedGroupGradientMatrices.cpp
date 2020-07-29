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

#include <iostream>

#include <gtest/gtest.h>

#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dart::neural;

TEST(ConstrainedGroupGradientMatrices, SIMPLE_MERGE)
{
  int numDofs = 2;
  int numConstraints = 4;
  double timestep = 0.001;

  ConstrainedGroupGradientMatrices matrices(numDofs, numConstraints, timestep);

  // Register 4 constraints with no bounce, and no penetration correction vel
  matrices.mockRegisterConstraint(0.0, 0.0);
  matrices.mockRegisterConstraint(0.0, 0.0);
  matrices.mockRegisterConstraint(0.0, 0.0);
  matrices.mockRegisterConstraint(0.0, 0.0);

  // Register 2 constraints with similar impulse tests
  Eigen::VectorXd i1(2);
  i1 << 0.0, 1.0;
  matrices.mockMeasureConstraintImpulse(i1, i1);
  Eigen::VectorXd i2(2);
  i2 << 0.01, 0.99;
  matrices.mockMeasureConstraintImpulse(i2, i2);

  // Register another 2 constraints with similar impulse tests
  Eigen::VectorXd i3(2);
  i3 << 1.0, 0.0;
  matrices.mockMeasureConstraintImpulse(i3, i3);
  Eigen::VectorXd i4(2);
  i4 << 0.99, 0.01;
  matrices.mockMeasureConstraintImpulse(i4, i4);

  // Regester mock LCP results
  Eigen::VectorXd X(4);
  X << 0.99, 1, 0.99, 1;
  Eigen::VectorXd lo(4);
  lo << 0.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd hi(4);
  hi << 1000, 1000, 1000, 1000;
  Eigen::VectorXi fIndex(4);
  fIndex << -1, -1, 0, 1;
  Eigen::VectorXd b(4);
  b << 0.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd aColNorms(4);
  aColNorms << 1.0, 1.0, 1.0, 1.0;
  matrices.registerLCPResults(X, lo, hi, fIndex, b, aColNorms);

  // Cut down the number of constraints
  matrices.deduplicateConstraints();

  EXPECT_TRUE(matrices.mX.size() == 2);
  EXPECT_TRUE(matrices.mHi.size() == 2);
  EXPECT_TRUE(matrices.mLo.size() == 2);
  EXPECT_TRUE(matrices.mFIndex.size() == 2);
  EXPECT_TRUE(matrices.mB.size() == 2);
  EXPECT_TRUE(matrices.mAColNorms.size() == 2);
  EXPECT_TRUE(matrices.mImpulseTests.size() == 2);
  EXPECT_TRUE(matrices.mMassedImpulseTests.size() == 2);
  EXPECT_TRUE(matrices.mRestitutionCoeffs.size() == 2);
  EXPECT_TRUE(matrices.mPenetrationCorrectionVelocities.size() == 2);

  Eigen::VectorXd c1 = (i1 + i2) / 2;
  Eigen::VectorXd c2 = (i3 + i4) / 2;
  EXPECT_TRUE(equals(c1, matrices.mImpulseTests[0]));
  EXPECT_TRUE(equals(c1, matrices.mMassedImpulseTests[0]));
  EXPECT_TRUE(equals(c2, matrices.mImpulseTests[1]));
  EXPECT_TRUE(equals(c2, matrices.mMassedImpulseTests[1]));

  Eigen::VectorXd newX(2);
  newX << 1.99, 1.99;
  EXPECT_TRUE(equals(newX, matrices.mX));
  Eigen::VectorXd newLo(2);
  newLo << 0.0, 0.0;
  EXPECT_TRUE(equals(newLo, matrices.mLo));
  Eigen::VectorXd newHi(2);
  newHi << 1000.0, 1000.0;
  EXPECT_TRUE(equals(newHi, matrices.mHi));
  Eigen::VectorXi newFIndex(2);
  newFIndex << -1, 0;
  EXPECT_TRUE(equals(newFIndex, matrices.mFIndex));
  Eigen::VectorXd newB(2);
  newB << 0.0, 0.0;
  EXPECT_TRUE(equals(newB, matrices.mB));
  Eigen::VectorXd newACols(2);
  newACols << 1.0, 1.0;
  EXPECT_TRUE(equals(newACols, matrices.mAColNorms));
}