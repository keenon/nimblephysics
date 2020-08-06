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

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/constraint/LCPUtils.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dart::constraint;

TEST(LCPUtils, SIMPLE_CLEAN)
{
  // The idea here is pretty simple: we've got constraint impulses that are
  // pretty close, but not exactly correct, because the LCP solver is a sloppy,
  // imprecise beast. We'll clean up those results to be precise.

  Eigen::MatrixXd A(2, 2);
  A << 1, 0, 0, 1;
  Eigen::VectorXd X(2);
  X << 1.05, 0.95;
  Eigen::VectorXd b(2);
  b << -1, -1;
  Eigen::VectorXd hi(2);
  hi << 1000, 1000;
  Eigen::VectorXd lo(2);
  lo << 0, 0;
  Eigen::VectorXi fIndex(2);
  fIndex << -1, -1;

  LCPUtils::cleanUpResults(A, X, b, hi, lo, fIndex);

  Eigen::VectorXd expected_X(2);
  expected_X << 1, 1;

  EXPECT_TRUE(equals(X, expected_X));
}

TEST(LCPUtils, HALF_CLAMPED)
{
  Eigen::MatrixXd A(2, 2);
  A << 1, 0, 0, 1;
  Eigen::VectorXd X(2);
  X << 1.05, 0.0;
  Eigen::VectorXd b(2);
  b << -1, 0;
  Eigen::VectorXd hi(2);
  hi << 1000, 1000;
  Eigen::VectorXd lo(2);
  lo << 0, 0;
  Eigen::VectorXi fIndex(2);
  fIndex << -1, -1;

  LCPUtils::cleanUpResults(A, X, b, hi, lo, fIndex);

  Eigen::VectorXd expected_X(2);
  expected_X << 1, 0;

  EXPECT_TRUE(equals(X, expected_X));
}

TEST(LCPUtils, HALF_UPPER_BOUND)
{
  Eigen::MatrixXd A(2, 2);
  A << 1, 0, 0, 1;
  Eigen::VectorXd X(2);
  X << 1.05, 0.1;
  Eigen::VectorXd b(2);
  b << -1, 0;
  Eigen::VectorXd hi(2);
  hi << 1000, 0.01;
  Eigen::VectorXd lo(2);
  lo << 0, 0;
  Eigen::VectorXi fIndex(2);
  fIndex << -1, 0;

  LCPUtils::cleanUpResults(A, X, b, hi, lo, fIndex);

  Eigen::VectorXd expected_X(2);
  expected_X << 1, 0.1;

  EXPECT_TRUE(equals(X, expected_X));
}