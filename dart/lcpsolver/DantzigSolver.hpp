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

#ifndef DART_LCPSOLVER_DANTZIGSOLVER_HPP_
#define DART_LCPSOLVER_DANTZIGSOLVER_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace lcpsolver {

// Solves L*X=B, with B containing 1 right hand sides where:
//   L is an n*n lower triangular matrix with ones on the diagonal.
//   L is stored by rows and its leading dimension is lskip.
//   B is an n*1 matrix that contains the right hand sides.
//   B is stored by columns and its leading dimension is also lskip.
//   B is overwritten with X.
// This processes blocks of 2*2.
// if this is in the factorizer source file, n must be a multiple of 2.
void solve_l1(const double* L, double* B, int n, int lskip1);

struct LdltFactorization
{
  Eigen::MatrixXd L;
  Eigen::VectorXd D;
};

class DantzigLcp
{
public:
  DantzigLcp(Eigen::MatrixXd& A, Eigen::VectorXd& b);

private:
  void swapRowsAndCols(int index1, int index2);
  void transfer_i_to_C(int i);
  void transfer_i_to_N(int i);

  Eigen::MatrixXd& mA;
  Eigen::VectorXd& mb;

  const int mn;
  int mNumC{0};
  int mNumN{0};

  Eigen::VectorXd mDell;
  Eigen::VectorXd mell;
  Eigen::VectorXd mtmp;

  LdltFactorization mLdlt;
};

class DantzigLcpOptimized
{
public:
  DantzigLcpOptimized(Eigen::MatrixXd& A, Eigen::VectorXd& b);

private:
  void swapRowsAndCols(int index1, int index2);
  void transfer_i_to_C(int i);
  void transfer_i_to_N(int i);

  int indexC(int i) const;
  int indexN(int i) const;
  double Aii(size_t i) const;
  double AiC_times_qC(size_t i, const Eigen::VectorXd& x) const;
  double AiN_times_qN(size_t i, const Eigen::VectorXd& x) const;
  void ANC_times_qC(Eigen::VectorXd& p, const Eigen::VectorXd& q);
  void solve1(double* a, int i, int dir, int only_transfer);

  Eigen::MatrixXd& mA;
  Eigen::VectorXd& mb;

  const int mn;
  int mNumC{0};
  int mNumN{0};

  Eigen::VectorXd mDell;
  Eigen::VectorXd mell;
  Eigen::VectorXd mtmp;

  LdltFactorization mLdlt;

  std::vector<double*> mARows;
  Eigen::VectorXi mRowIndex;

  int* mC;
};

class DantzigSolver
{
public:
  DantzigSolver();

  virtual ~DantzigSolver();

  static bool solve(
      const Eigen::MatrixXd& A,
      const Eigen::VectorXd& b,
      Eigen::VectorXd& x,
      int numContacts = 0,
      double mu = 0,
      int numDir = 4);

private:
};

} // namespace lcpsolver
} // namespace dart

#endif // DART_LCPSOLVER_DANTZIGSOLVER_HPP_
