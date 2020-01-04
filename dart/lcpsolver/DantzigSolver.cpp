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

#include "dart/lcpsolver/DantzigSolver.hpp"

#include <cstdio>

#include "dart/common/StlHelpers.hpp"
#include "dart/lcpsolver/Lemke.hpp"
#include "dart/math/Constants.hpp"

namespace dart {
namespace lcpsolver {

//==============================================================================
void solve_l1(const double* L, double* B, int n, int lskip1)
{
  // Declare variables - Z matrix, p and q vectors, etc
  double Z11;
  double m11;
  double Z21;
  double m21;
  double p1;
  double q1;
  double p2;
  double* ex;
  const double* ell;
  int i;
  int j;

  // Compute all 2 x 1 blocks of X
  for (i = 0; i < n; i += 2)
  {
    // Compute all 2 x 1 block of X, from rows i..i+2-1

    // Set the Z matrix to 0
    Z11 = 0;
    Z21 = 0;
    ell = L + i * lskip1;
    ex = B;

    // The inner loop that computes outer products and adds them to Z
    for (j = i - 2; j >= 0; j -= 2)
    {
      p1 = ell[0];
      q1 = ex[0];
      m11 = p1 * q1;
      p2 = ell[lskip1];
      m21 = p2 * q1;
      Z11 += m11;
      Z21 += m21;

      // Compute outer product and add it to the Z matrix
      p1 = ell[1];
      q1 = ex[1];
      m11 = p1 * q1;
      p2 = ell[1 + lskip1];
      m21 = p2 * q1;

      // Advance pointers
      ell += 2;
      ex += 2;
      Z11 += m11;
      Z21 += m21;
    }

    // Compute left-over iterations
    j += 2;
    for (; j > 0; j--)
    {
      // Compute outer product and add it to the Z matrix
      p1 = ell[0];
      q1 = ex[0];
      m11 = p1 * q1;
      p2 = ell[lskip1];
      m21 = p2 * q1;

      // Advance pointers
      ell += 1;
      ex += 1;
      Z11 += m11;
      Z21 += m21;
    }

    // Finish computing the X(i) block
    Z11 = ex[0] - Z11;
    ex[0] = Z11;
    p1 = ell[lskip1];
    Z21 = ex[1] - Z21 - p1 * Z11;
    ex[1] = Z21;
  }
}

//==============================================================================
DantzigLcp::DantzigLcp(Eigen::MatrixXd& A, Eigen::VectorXd& b)
  : mA(A), mb(b), mn(static_cast<int>(A.rows()))
{
  // Do nothing (yet)
}

//==============================================================================
void DantzigLcp::swapRowsAndCols(int index1, int index2)
{
  mA.col(index1).swap(mA.col(index2));
  mA.row(index1).swap(mA.row(index2));
}

//==============================================================================
void DantzigLcp::transfer_i_to_C(int i)
{
  if (mNumC > 0)
  {
    {
      // Do something
    }

    mLdlt.D[mNumC] = 1.0 / (mA(i, i) - mell.head(mNumC).dot(mDell.head(mNumC)));
  }
  else
  {
    mLdlt.D[0] = 1.0 / mA(i, i);
  }
}

//==============================================================================
void DantzigLcp::transfer_i_to_N(int /*i*/)
{
  mNumN++;
}

//==============================================================================
DantzigLcpOptimized::DantzigLcpOptimized(Eigen::MatrixXd& A, Eigen::VectorXd& b)
  : mA(A), mb(b), mn(static_cast<int>(A.rows()))
{
  mARows.resize(static_cast<size_t>(mn));
  double* ptr = mA.data();
  for (auto i = 0; i < mn; ++i)
  {
    mARows[static_cast<size_t>(i)] = ptr;
    ptr += mn;
  }
}

//==============================================================================
int DantzigLcpOptimized::indexC(int i) const
{
  return i;
}

//==============================================================================
int DantzigLcpOptimized::indexN(int i) const
{
  return i + mNumC;
}

//==============================================================================
double DantzigLcpOptimized::Aii(size_t i) const
{
  return mARows[i][i];
}

//==============================================================================
double DantzigLcpOptimized::AiC_times_qC(
    size_t i, const Eigen::VectorXd& x) const
{
  const auto& AC = mA.row(mRowIndex[static_cast<int>(i)]).head(mNumC);
  const auto& xC = x.head(mNumC);
  return AC.dot(xC);
}

//==============================================================================
double DantzigLcpOptimized::AiN_times_qN(
    size_t i, const Eigen::VectorXd& x) const
{
  const auto& AN = mA.row(mRowIndex[static_cast<int>(i)]).segment(mNumC, mNumN);
  const auto& xN = x.segment(mNumC, mNumN);
  return AN.dot(xN);
}

//==============================================================================
void DantzigLcpOptimized::ANC_times_qC(
    Eigen::VectorXd& p, const Eigen::VectorXd& q)
{
  for (auto i = 0; i < mNumN; ++i)
  {
    p[mNumC + i] = mA.row(mRowIndex[mNumC + static_cast<int>(i)])
                       .head(mNumC)
                       .dot(q.head(mNumC));
  }
}

//==============================================================================
void DantzigLcpOptimized::solve1(double* a, int i, int dir, int only_transfer)
{
  if (mNumC > 0)
  {
  }
}

//==============================================================================
DantzigSolver::DantzigSolver()
{
  // Do nothing
}

//==============================================================================
DantzigSolver::~DantzigSolver()
{
  // Do nothing
}

//==============================================================================
void compute_f_direction(
    int n, int index, const Eigen::MatrixXd& A, Eigen::VectorXd& df)
{
  df.setZero(n);
  df[index] = 1;
}

//==============================================================================
bool DantzigSolver::solve(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    Eigen::VectorXd& x,
    int numContacts,
    double mu,
    int numDir)
{
  assert(numDir >= 4);
  DART_UNUSED(numDir);

  const auto n = A.rows();

  x.setZero(n);
  Eigen::VectorXd w = b;

  Eigen::VectorXi type = Eigen::VectorXi::Zero(0);

  for (auto i = 0; i < n; ++i)
  {
  }

  //  Eigen::VectorXd lo = Eigen::VectorXd::Zero(n);
  //  Eigen::VectorXd hi = Eigen::VectorXd::Constant(n,
  //  math::constantsd::inf()); Eigen::VectorXi fIndex =
  //  Eigen::VectorXi::Constant(n, -1);

  //  for (int i = 0; i < numContacts; ++i)
  //  {
  //    const auto index1 = numContacts + i * 2 + 0;
  //    const auto index2 = numContacts + i * 2 + 0;

  //    fIndex[index1] = i;
  //    fIndex[index2] = i;

  //    lo[index1] = -mu;
  //    lo[index2] = -mu;

  //    hi[index1] = mu;
  //    hi[index2] = mu;
  //  }

  return true;
}

} // namespace lcpsolver
} // namespace dart
