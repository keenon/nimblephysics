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

#include "dart/constraint/PgsBoxedLcpSolver.hpp"

#include <cmath>
#include <cstring>

#include <Eigen/Dense>

#include "dart/external/odelcpsolver/matrix.h"
#include "dart/external/odelcpsolver/misc.h"
#include "dart/math/Constants.hpp"

#define PGS_EPSILON 10e-9

namespace dart {
namespace constraint {

//==============================================================================
PgsBoxedLcpSolver::Option::Option(
    int maxIteration,
    s_t deltaXTolerance,
    s_t relativeDeltaXTolerance,
    s_t epsilonForDivision,
    bool randomizeConstraintOrder)
  : mMaxIteration(maxIteration),
    mDeltaXThreshold(deltaXTolerance),
    mRelativeDeltaXTolerance(relativeDeltaXTolerance),
    mEpsilonForDivision(epsilonForDivision),
    mRandomizeConstraintOrder(randomizeConstraintOrder)
{
  // Do nothing
}

//==============================================================================
const std::string& PgsBoxedLcpSolver::getType() const
{
  return getStaticType();
}

//==============================================================================
const std::string& PgsBoxedLcpSolver::getStaticType()
{
  static const std::string type = "PgsBoxedLcpSolver";
  return type;
}

//==============================================================================
bool PgsBoxedLcpSolver::solve(
    int n,
    s_t* A,
    s_t* x,
    s_t* b,
    int nub,
    s_t* lo,
    s_t* hi,
    int* findex,
    bool /*earlyTermination*/)
{
  const int nskip = dPAD(n);

  // If all the variables are unbounded then we can just factor, solve, and
  // return.R
  if (nub >= n)
  {
    double* cache = new double[n];
    memset(cache, 0, n);
    double* A_d = new double[n * nskip];
    double* b_d = new double[n];
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < nskip; j++)
      {
        A_d[i * nskip + j] = static_cast<double>(A[i * nskip + j]);
      }
      b_d[i] = static_cast<double>(b[i]);
    }

    dFactorLDLT(A_d, cache, n, nskip);
    dSolveLDLT(A_d, cache, b_d, n, nskip);

    for (int i = 0; i < n; i++)
    {
      x[i] = static_cast<s_t>(b_d[i]);
    }
    delete[] cache;
    delete[] A_d;
    delete[] b_d;

    return true;
  }

  mCacheOrder.clear();
  mCacheOrder.reserve(n);

  bool possibleToTerminate = true;
  for (int i = 0; i < n; ++i)
  {
    // mOrderCacheing
    if (A[nskip * i + i] < mOption.mEpsilonForDivision)
    {
      x[i] = 0.0;
      continue;
    }

    mCacheOrder.push_back(i);

    // Initial loop
    const s_t* A_ptr = A + nskip * i;
    const s_t old_x = x[i];
    assert(!isnan(old_x));

    s_t new_x = b[i];

    for (int j = 0; j < i; ++j)
      new_x -= A_ptr[j] * x[j];

    for (int j = i + 1; j < n; ++j)
      new_x -= A_ptr[j] * x[j];

    assert(!isnan(new_x));
    assert(A[nskip * i + i] != 0);
    new_x /= A[nskip * i + i];
    assert(!isnan(new_x));

    if (findex[i] >= 0)
    {
      const s_t hi_tmp = hi[i] * x[findex[i]];
      const s_t lo_tmp = -hi_tmp;

      if (new_x > hi_tmp)
        x[i] = hi_tmp;
      else if (new_x < lo_tmp)
        x[i] = lo_tmp;
      else
        x[i] = new_x;
    }
    else
    {
      if (new_x > hi[i])
        x[i] = hi[i];
      else if (new_x < lo[i])
        x[i] = lo[i];
      else
        x[i] = new_x;
    }
    assert(!isnan(x[i]));

    // Test
    if (possibleToTerminate)
    {
      const s_t deltaX = abs(x[i] - old_x);
      if (deltaX > mOption.mDeltaXThreshold)
        possibleToTerminate = false;
    }
  }

  if (possibleToTerminate)
  {
    return true;
  }

  // Normalizing
  for (const auto& index : mCacheOrder)
  {
    const s_t dummy = 1.0 / A[nskip * index + index];
    b[index] *= dummy;
    for (int j = 0; j < n; ++j)
      A[nskip * index + j] *= dummy;
  }

  for (int iter = 1; iter < mOption.mMaxIteration; ++iter)
  {
    if (mOption.mRandomizeConstraintOrder)
    {
      if ((iter & 7) == 0)
      {
        for (std::size_t i = 1; i < mCacheOrder.size(); ++i)
        {
          const int tmp = mCacheOrder[i];
          const int swapi = dRandInt(i + 1);
          mCacheOrder[i] = mCacheOrder[swapi];
          mCacheOrder[swapi] = tmp;
        }
      }
    }

    possibleToTerminate = true;

    // Single loop
    for (const auto& index : mCacheOrder)
    {
      const s_t* A_ptr = A + nskip * index;
      s_t new_x = b[index];
      const s_t old_x = x[index];

      for (int j = 0; j < index; j++)
        new_x -= A_ptr[j] * x[j];

      for (int j = index + 1; j < n; j++)
        new_x -= A_ptr[j] * x[j];

      if (findex[index] >= 0)
      {
        const s_t hi_tmp = hi[index] * x[findex[index]];
        const s_t lo_tmp = -hi_tmp;

        if (new_x > hi_tmp)
          x[index] = hi_tmp;
        else if (new_x < lo_tmp)
          x[index] = lo_tmp;
        else
          x[index] = new_x;
      }
      else
      {
        if (new_x > hi[index])
          x[index] = hi[index];
        else if (new_x < lo[index])
          x[index] = lo[index];
        else
          x[index] = new_x;
      }

      if (possibleToTerminate && abs(x[index]) > mOption.mEpsilonForDivision)
      {
        const s_t relativeDeltaX = abs((x[index] - old_x) / x[index]);
        if (relativeDeltaX > mOption.mRelativeDeltaXTolerance)
          possibleToTerminate = false;
      }
    }

    if (possibleToTerminate)
      break;
  }

  return possibleToTerminate;
}

#ifndef NDEBUG
//==============================================================================
bool PgsBoxedLcpSolver::canSolve(int n, const s_t* A)
{
  const int nskip = dPAD(n);

  // Return false if A has zero-diagonal or A is nonsymmetric matrix
  for (auto i = 0; i < n; ++i)
  {
    if (A[nskip * i + i] < PGS_EPSILON)
      return false;

    for (auto j = 0; j < n; ++j)
    {
      if (abs(A[nskip * i + j] - A[nskip * j + i]) > PGS_EPSILON)
        return false;
    }
  }

  return true;
}
#endif

//==============================================================================
void PgsBoxedLcpSolver::setOption(const PgsBoxedLcpSolver::Option& option)
{
  mOption = option;
}

//==============================================================================
const PgsBoxedLcpSolver::Option& PgsBoxedLcpSolver::getOption() const
{
  return mOption;
}

} // namespace constraint
} // namespace dart
