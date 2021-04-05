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

#include "dart/constraint/DantzigBoxedLcpSolver.hpp"

#include "dart/external/odelcpsolver/lcp.h"

namespace dart {
namespace constraint {

//==============================================================================
const std::string& DantzigBoxedLcpSolver::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& DantzigBoxedLcpSolver::getStaticType()
{
  static const std::string type = "DantzigBoxedLcpSolver";
  return type;
}

//==============================================================================
bool DantzigBoxedLcpSolver::solve(
    int n,
    s_t* A,
    s_t* x,
    s_t* b,
    int /*nub*/,
    s_t* lo,
    s_t* hi,
    int* findex,
    bool earlyTermination)
{
  try
  {
#ifdef DART_USE_ARBITRARY_PRECISION
    int nSkip = dPAD(n);
    double* A_d = new double[n * nSkip];
    double* x_d = new double[n];
    double* b_d = new double[n];
    double* lo_d = new double[n];
    double* hi_d = new double[n];
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < nSkip; j++)
      {
        A_d[i * nSkip + j] = static_cast<double>(A[i * nSkip + j]);
      }
      x_d[i] = static_cast<double>(x[i]);
      b_d[i] = static_cast<double>(b[i]);
      lo_d[i] = static_cast<double>(lo[i]);
      hi_d[i] = static_cast<double>(hi[i]);
    }
    bool ret = dSolveLCP(
        n, A_d, x_d, b_d, nullptr, 0, lo_d, hi_d, findex, earlyTermination);
    for (int i = 0; i < n; i++)
    {
      x[i] = static_cast<s_t>(x_d[i]);
    }
    delete[] A_d;
    delete[] x_d;
    delete[] b_d;
    delete[] lo_d;
    delete[] hi_d;
    return ret;
#else
    return dSolveLCP(n, A, x, b, nullptr, 0, lo, hi, findex, earlyTermination);
#endif
  }
  catch (...)
  {
    // In case of unspecified error, don't crash the whole program, just return
    // failure so we can fall back to another LCP solver.
    return false;
  }
}

#ifndef NDEBUG
//==============================================================================
bool DantzigBoxedLcpSolver::canSolve(int /*n*/, const s_t* /*A*/)
{
  // TODO(JS): Not implemented.
  return true;
}
#endif

} // namespace constraint
} // namespace dart
