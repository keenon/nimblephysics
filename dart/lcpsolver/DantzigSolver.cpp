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
bool DantzigSolver::solve(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    Eigen::VectorXd* x,
    int numContacts,
    double mu,
    int numDir)
{
  assert(numDir >= 4);
  DART_UNUSED(numDir);

  const auto n = A.rows();

  x->resize(n);
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd lo = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd hi = Eigen::VectorXd::Constant(n, math::constantsd::inf());
  Eigen::VectorXi fIndex = Eigen::VectorXi::Constant(n, -1);

  for (int i = 0; i < numContacts; ++i)
  {
    const auto index1 = numContacts + i * 2 + 0;
    const auto index2 = numContacts + i * 2 + 0;

    fIndex[index1] = i;
    fIndex[index2] = i;

    lo[index1] = -mu;
    lo[index2] = -mu;

    hi[index1] = mu;
    hi[index2] = mu;
  }

  return true;
}

} // namespace lcpsolver
} // namespace dart
