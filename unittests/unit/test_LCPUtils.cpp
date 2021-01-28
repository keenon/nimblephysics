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

#include "dart/constraint/DantzigBoxedLcpSolver.hpp"
#include "dart/constraint/LCPUtils.hpp"
#include "dart/constraint/PgsBoxedLcpSolver.hpp"
#include "dart/external/odelcpsolver/lcp.h"

#include "TestHelpers.hpp"

using namespace dart;
using namespace dart::constraint;

#ifdef ALL_TESTS
TEST(LCP_UTILS, MERGE_COLS)
{
  Eigen::VectorXd aFacBlock = Eigen::VectorXd::Random(2);
  Eigen::VectorXd aFac = Eigen::VectorXd::Zero(4);
  aFac.segment(0, 2) = aFacBlock;
  aFac.segment(2, 2) = aFacBlock;
  Eigen::MatrixXd A = aFac * aFac.transpose();
  Eigen::VectorXd x = Eigen::VectorXd::Random(4);
  Eigen::VectorXd b = Eigen::VectorXd::Random(4);
  b.segment(2, 2) = b.segment(0, 2);
  Eigen::VectorXd hi = Eigen::VectorXd::Ones(4) * 1000;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(4);
  Eigen::VectorXi fIndex = Eigen::VectorXi::Ones(4) * -1;
  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(4, 4);
  // set up friction forces
  hi(1) = 1.0;
  hi(3) = 1.0;
  lo(1) = -1.0;
  lo(3) = -1.0;
  fIndex(1) = 0;
  fIndex(3) = 2;

  Eigen::MatrixXd oldA = A;
  Eigen::VectorXd oldX = x;
  Eigen::VectorXd oldB = b;
  Eigen::VectorXd oldHi = hi;
  Eigen::VectorXd oldLo = lo;
  Eigen::VectorXi oldFIndex = fIndex;
  Eigen::MatrixXd oldMapOut = mapOut;

  DantzigBoxedLcpSolver lcpSolver;
  /*
  bool successBig = lcpSolver.solve(
      2,
      A.data(),
      x.data(),
      b.data(),
      0,
      lo.data(),
      hi.data(),
      fIndex.data(),
      true);
  EXPECT_TRUE(successBig);
  */

  std::cout << "Original A: " << std::endl << A << std::endl;
  LCPUtils::mergeLCPColumns(0, 2, A, x, b, hi, lo, fIndex, mapOut);
  std::cout << "map out: " << std::endl << mapOut << std::endl;
  LCPUtils::printReplicationCode(A, x, lo, hi, b, fIndex);
  // Merge friction
  LCPUtils::mergeLCPColumns(1, 2, A, x, b, hi, lo, fIndex, mapOut);

  std::cout << "map out: " << std::endl << mapOut << std::endl;
  LCPUtils::printReplicationCode(A, x, lo, hi, b, fIndex);

  bool successSmall = lcpSolver.solve(
      2,
      A.data(),
      x.data(),
      b.data(),
      0,
      lo.data(),
      hi.data(),
      fIndex.data(),
      true);
  EXPECT_TRUE(successSmall);
  bool valid = LCPUtils::isLCPSolutionValid(
      oldA, mapOut * x, oldB, oldHi, oldLo, oldFIndex);
  EXPECT_TRUE(valid);
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, SOLVE_MERGED)
{
  srand(42);
  Eigen::VectorXd aFacBlock = Eigen::VectorXd::Random(2);
  Eigen::VectorXd aFac = Eigen::VectorXd::Zero(4);
  aFac.segment(0, 2) = aFacBlock;
  aFac.segment(2, 2) = aFacBlock;
  Eigen::MatrixXd A = aFac * aFac.transpose();
  Eigen::VectorXd x = Eigen::VectorXd::Random(4);
  x.segment(2, 2) = x.segment(0, 2);
  Eigen::VectorXd b = A * x;
  x = Eigen::VectorXd::Random(4);
  Eigen::VectorXd hi = Eigen::VectorXd::Ones(4) * 1000;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(4);
  Eigen::VectorXi fIndex = Eigen::VectorXi::Ones(4) * -1;
  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(4, 4);
  // set up friction forces
  hi(1) = 1.0;
  hi(3) = 1.0;
  lo(1) = -1.0;
  lo(3) = -1.0;
  fIndex(1) = 0;
  fIndex(3) = 2;

  Eigen::MatrixXd oldA = A;
  Eigen::VectorXd oldX = x;
  Eigen::VectorXd oldB = b;
  Eigen::VectorXd oldHi = hi;
  Eigen::VectorXd oldLo = lo;
  Eigen::VectorXi oldFIndex = fIndex;
  Eigen::MatrixXd oldMapOut = mapOut;

  std::shared_ptr<BoxedLcpSolver> lcpSolver
      = std::make_shared<DantzigBoxedLcpSolver>();

  /*
  std::cout << "Original A: " << std::endl << A << std::endl;
  LCPUtils::mergeLCPColumns(0, 2, A, x, b, hi, lo, fIndex, mapOut);
  std::cout << "map out: " << std::endl << mapOut << std::endl;
  LCPUtils::printReplicationCode(A, x, lo, hi, b, fIndex);
  // Merge friction
  LCPUtils::mergeLCPColumns(1, 2, A, x, b, hi, lo, fIndex, mapOut);

  std::cout << "map out: " << std::endl << mapOut << std::endl;
  LCPUtils::printReplicationCode(A, x, lo, hi, b, fIndex);

  bool successSmall = lcpSolver->solve(
      2,
      A.data(),
      x.data(),
      b.data(),
      0,
      lo.data(),
      hi.data(),
      fIndex.data(),
      true);
  std::cout << "x: " << std::endl << x << std::endl;
  EXPECT_TRUE(successSmall);
  bool valid = LCPUtils::isLCPSolutionValid(
      oldA, mapOut * x, oldB, oldHi, oldLo, oldFIndex);
  EXPECT_TRUE(valid);
  */
  x = A.completeOrthogonalDecomposition().solve(b);

  bool success
      = LCPUtils::solveDeduplicated(lcpSolver, A, x, b, hi, lo, fIndex);
  EXPECT_TRUE(success);
  bool valid = LCPUtils::isLCPSolutionValid(A, x, b, hi, lo, fIndex);
  EXPECT_TRUE(valid);
  std::cout << "Solve got X: " << std::endl << x << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_1)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
  // clang-format off
  A <<
  0.0424296,  -0.0139791,  0,  0.0424296,  -0.0139791,  0,
  -0.0139791,  0.0698999,  0,  -0.0139791,  0.0698999,  0,
  0,  0,  0,  0,  0,  0,
  0.0424296,  -0.0139791,  0,  0.0424296,  -0.0139791,  0,
  -0.0139791,  0.0698999,  0,  -0.0139791,  0.0698999,  0,
  0,  0,  0,  0,  0,  0;
  // clang-format on
  Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
  x << 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(6);
  lo << 0, -1, -1, 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(6);
  hi << std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
  b << 1.67162, 2.08376, 0, 1.67162, 2.08376, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(6);
  fIndex << -1, 0, 0, -1, 3, 3;

  Eigen::MatrixXd reducedA = A;
  Eigen::VectorXd reducedX = x;
  Eigen::VectorXd reducedLo = lo;
  Eigen::VectorXd reducedHi = hi;
  Eigen::VectorXd reducedB = b;
  Eigen::VectorXi reducedFIndex = fIndex;
  Eigen::MatrixXd mapOutOrig = LCPUtils::reduce(
      reducedA, reducedX, reducedLo, reducedHi, reducedB, reducedFIndex);

  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(A.rows(), A.cols());

  LCPUtils::mergeLCPColumns(0, 3, A, x, b, hi, lo, fIndex, mapOut);
  LCPUtils::mergeLCPColumns(1, 3, A, x, b, hi, lo, fIndex, mapOut);
  LCPUtils::mergeLCPColumns(2, 3, A, x, b, hi, lo, fIndex, mapOut);

  EXPECT_TRUE(equals(A, reducedA, 1e-8));
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_2)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
  // clang-format off
  A <<
  1,  -0.0279582,  0,
  -0.329466,  0.1398,  0,
  0,  0,  0;
  // clang-format on
  Eigen::VectorXd x = Eigen::VectorXd::Zero(3);
  x << 19.6988, 0, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(3);
  lo << 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(3);
  hi << std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
  b << 19.6988, 2.08376, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(3);
  fIndex << -1, 0, 0;
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_3)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
  // clang-format off
  A <<
  0.0424296,  -0.0139791,  0,  0.0424296,  -0.0139791,  0,
  -0.0139791,  0.0698999,  0,  -0.0139791,  0.0698999,  0,
  0,  0,  0,  0,  0,  0,
  0.0424296,  -0.0139791,  0,  0.0424296,  -0.0139791,  0,
  -0.0139791,  0.0698999,  0,  -0.0139791,  0.0698999,  0,
  0,  0,  0,  0,  0,  0;
  // clang-format on
  Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
  x << 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(6);
  lo << 0, -1, -1, 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(6);
  hi << std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
  b << 1.67162, 2.08376, 0, 1.67162, 2.08376, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(6);
  fIndex << -1, 0, 0, -1, 3, 3;

  Eigen::MatrixXd reducedA = A;
  Eigen::VectorXd reducedX = x;
  Eigen::VectorXd reducedLo = lo;
  Eigen::VectorXd reducedHi = hi;
  Eigen::VectorXd reducedB = b;
  Eigen::VectorXi reducedFIndex = fIndex;
  Eigen::MatrixXd mapOut = LCPUtils::reduce(
      reducedA, reducedX, reducedLo, reducedHi, reducedB, reducedFIndex);
  DantzigBoxedLcpSolver lcpSolver;
  int reducedN = reducedX.size();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      reducedAPadded = Eigen::MatrixXd::Zero(reducedN, dPAD(reducedN));
  reducedAPadded.block(0, 0, reducedN, reducedN) = reducedA;

  Eigen::VectorXd proposedX
      = reducedA.completeOrthogonalDecomposition().solve(reducedB);
  Eigen::VectorXd diff = reducedA * proposedX - reducedB;

  Eigen::VectorXd reducedBSaved = reducedB;

  lcpSolver.solve(
      reducedN,
      reducedAPadded.data(),
      reducedX.data(),
      reducedB.data(),
      0,
      reducedLo.data(),
      reducedHi.data(),
      reducedFIndex.data(),
      false);

  Eigen::VectorXd smallV = reducedA * reducedX - reducedBSaved;
  Eigen::VectorXd fullResult = mapOut * reducedX;
  Eigen::VectorXd bigV = A * fullResult - b;

  EXPECT_TRUE(LCPUtils::isLCPSolutionValid(A, fullResult, b, hi, lo, fIndex));
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_4)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
  // clang-format off
  A <<
  0.0923023,  0.0247589,  0,  0.0923023,  0.0247589,  0,
  0.0247589,  0.0137374,  0,  0.0247589,  0.0137374,  0,
  0,  0,  0,  0,  0,  0,
  0.0923023,  0.0247589,  0,  0.0923023,  0.0247589,  0,
  0.0247589,  0.0137374,  0,  0.0247589,  0.0137374,  0,
  0,  0,  0,  0,  0,  0;
  // clang-format on
  /*
  Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
  x << 0.0251303, -0.0198151, 0, 0.0251303, -0.0198151, 0;
  */
  Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
  x << 0.0270786, -0.0270786, 0, 0.0270786, -0.0270786, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(6);
  lo << 0, -1, -1, 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(6);
  hi << std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
  b << 0.00365796, 0.000140769, 0, 0.00365796, 0.000140769, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(6);
  fIndex << -1, 0, 0, -1, 3, 3;
  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(6, 6);

  bool isSolutionValid = LCPUtils::isLCPSolutionValid(A, x, b, hi, lo, fIndex);
  std::cout << "is solution valid: " << isSolutionValid << std::endl;

  LCPUtils::mergeLCPColumns(0, 3, A, x, b, hi, lo, fIndex, mapOut);
  LCPUtils::mergeLCPColumns(1, 3, A, x, b, hi, lo, fIndex, mapOut);
  LCPUtils::mergeLCPColumns(2, 3, A, x, b, hi, lo, fIndex, mapOut);
  LCPUtils::printReplicationCode(A, x, lo, hi, b, fIndex);

  Eigen::VectorXd out = A.completeOrthogonalDecomposition().solve(b);
  Eigen::VectorXd outBig = mapOut * out;

  std::cout << "solved X: " << std::endl << out << std::endl;
  std::cout << "solved full X: " << std::endl << outBig << std::endl;
  isSolutionValid = LCPUtils::isLCPSolutionValid(A, out, b, hi, lo, fIndex);
  std::cout << "is solution valid: " << isSolutionValid << std::endl;
  std::cout << "original X: " << std::endl << x << std::endl;
  std::cout << "original full X: " << std::endl << mapOut * x << std::endl;
  Eigen::VectorXd diff = A * x - b;
  std::cout << "original LCP diff: " << std::endl << diff << std::endl;
  std::cout << "solved LCP diff: " << std::endl << A * out - b << std::endl;
  Eigen::VectorXd outWithGap
      = A.completeOrthogonalDecomposition().solve(b + diff);
  std::cout << "solved LCP with gap: " << std::endl
            << A * outWithGap - b << std::endl;
  std::cout << "original full X: " << std::endl
            << mapOut * outWithGap << std::endl;

  Eigen::VectorXd sum = Eigen::VectorXd::Zero(3);
  sum += A.col(0);
  sum += A.col(1);
  Eigen::VectorXd res = sum.completeOrthogonalDecomposition().solve(b);
  std::cout << "res: " << std::endl << res << std::endl;

  Eigen::Vector3d boundedResult = Eigen::Vector3d(res(0), res(0), 0);
  std::cout << "bounded result LCP diff: " << std::endl
            << A * boundedResult - b << std::endl;

  std::cout << "sum: " << std::endl << sum << std::endl;
  std::cout << "b: " << std::endl << b << std::endl;

  std::shared_ptr<PgsBoxedLcpSolver> accurateAndSlowSolver
      = std::make_shared<PgsBoxedLcpSolver>();
  accurateAndSlowSolver->setOption(
      PgsBoxedLcpSolver::Option(50000, 1e-15, 1e-12, 1e-10, false));
  int reducedN = x.size();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      reducedAPadded = Eigen::MatrixXd::Zero(reducedN, dPAD(reducedN));
  reducedAPadded.block(0, 0, reducedN, reducedN) = A;
  accurateAndSlowSolver->solve(
      reducedN,
      reducedAPadded.data(),
      x.data(),
      b.data(),
      0,
      lo.data(),
      hi.data(),
      fIndex.data(),
      false);
  std::cout << "new solved LCP: " << std::endl << x << std::endl;
  std::cout << "new solved LCP diff: " << std::endl << A * x - b << std::endl;

  isSolutionValid = LCPUtils::isLCPSolutionValid(A, x, b, hi, lo, fIndex);
  std::cout << "is solution valid: " << isSolutionValid << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_5)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(12, 12);
  // clang-format off
  A <<
  1.0591,  -0.0531116,  0,  1.0591,  -0.0531116,  0,  1.06715,  -0.0532007,  0,  1.06715,  -0.0532007,  0,
  -0.0531116,  1.05186,  0,  -0.0531116,  1.05186,  0,  -0.0548259,  1.05188,  0,  -0.0548259,  1.05188,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  1.0591,  -0.0531116,  0,  1.0591,  -0.0531116,  0,  1.06715,  -0.0532007,  0,  1.06715,  -0.0532007,  0,
  -0.0531116,  1.05186,  0,  -0.0531116,  1.05186,  0,  -0.0548259,  1.05188,  0,  -0.0548259,  1.05188,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  1.06715,  -0.0548259,  0,  1.06715,  -0.0548259,  0,  1.08506,  -0.0550241,  0,  1.08506,  -0.0550241,  0,
  -0.0532007,  1.05188,  0,  -0.0532007,  1.05188,  0,  -0.0550241,  1.0519,  0,  -0.0550241,  1.0519,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  1.06715,  -0.0548259,  0,  1.06715,  -0.0548259,  0,  1.08506,  -0.0550241,  0,  1.08506,  -0.0550241,  0,
  -0.0532007,  1.05188,  0,  -0.0532007,  1.05188,  0,  -0.0550241,  1.0519,  0,  -0.0550241,  1.0519,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0;
  // clang-format on
  Eigen::VectorXd x = Eigen::VectorXd::Zero(12);
  x << 0.00426985, 7.49341e-05, 0, 0.00426985, 7.49341e-05, 0, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(12);
  lo << 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(12);
  hi << std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(12);
  b << 0.0090364, -0.000295916, 0, 0.0090364, -0.000295916, 0, -0.0139171,
      -4.19424e-05, 0, -0.0139171, -4.19424e-05, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(12);
  fIndex << -1, 0, 0, -1, 3, 3, -1, 6, 6, -1, 9, 9;
}
#endif

// #ifdef ALL_TESTS
TEST(LCP_UTILS, REAL_LIFE_FAILURE_6)
{
}
// #endif

#ifdef ALL_TESTS
TEST(LCP_UTILS, BLOCK_SYMMETRIC_CASE)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
  // clang-format off
  A <<
  0.0923029,  0.0247581,  0,  0.0923029,  0.0247581,  0,
  0.0247581,  0.0137368,  0,  0.0247581,  0.0137368,  0,
  0,  0,  0,  0,  0,  0,
  0.0923029,  0.0247581,  0,  0.0923029,  0.0247581,  0,
  0.0247581,  0.0137368,  0,  0.0247581,  0.0137368,  0,
  0,  0,  0,  0,  0,  0;
  // clang-format on
  Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
  x << 0.0491903, 0.00921924, 0, 0, 0, 0;
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(6);
  lo << 0, -1, -1, 0, -1, -1;
  Eigen::VectorXd hi = Eigen::VectorXd::Zero(6);
  hi << std::numeric_limits<double>::infinity(), 1, 1,
      std::numeric_limits<double>::infinity(), 1, 1;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
  b << 0.00365797, 0.000140734, 0, 0.00365797, 0.000140734, 0;
  Eigen::VectorXi fIndex = Eigen::VectorXi::Zero(6);
  fIndex << -1, 0, 0, -1, 3, 3;

  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
  // A = U^T * S * V
  std::cout << "Singular values" << std::endl
            << svd.singularValues() << std::endl;
  std::cout << "U:" << std::endl << svd.matrixU() << std::endl;
  std::cout << "V:" << std::endl << svd.matrixV() << std::endl;
  // If A is symmetric, then U must equal V (at least for columns with non-zero
  // singular values)

  // So then A = V^T * S * V

  // Where V can be rectangular

  Eigen::VectorXd Vx = svd.matrixV() * b;
  std::cout << "V*b:" << std::endl << Vx << std::endl;
  Eigen::VectorXd SVx = svd.singularValues().asDiagonal() * Vx;
  std::cout << "S*V*b:" << std::endl << SVx << std::endl;
  Eigen::VectorXd UTSVx = svd.matrixU().transpose() * SVx;
  std::cout << "U^T*S*V*b:" << std::endl << UTSVx << std::endl;

  Eigen::VectorXd filter = Eigen::VectorXd::Zero(svd.singularValues().size());
  for (int i = 0; i < svd.singularValues().size(); i++)
  {
    if (std::abs(svd.singularValues()(i)) > 1e-4)
    {
      filter(i) = 1.0;
    }
  }

  // If A was full rank, this would be the identity matrix
  Eigen::MatrixXd fA
      = svd.matrixU().transpose() * filter.asDiagonal() * svd.matrixU();
  Eigen::VectorXd fx = fA * x;
  std::cout << "x:" << std::endl << x << std::endl;
  std::cout << "A * x:" << std::endl << A * x << std::endl;
  std::cout << "filtered x:" << std::endl << fx << std::endl;
  std::cout << "A * fx:" << std::endl << A * fx << std::endl;
}
#endif