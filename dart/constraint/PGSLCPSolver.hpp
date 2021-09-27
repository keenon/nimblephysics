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

#ifndef DART_CONSTRAINT_PGSLCPSOLVER_HPP_
#define DART_CONSTRAINT_PGSLCPSOLVER_HPP_

#include <cstddef>

#include "dart/config.hpp"
#include "dart/constraint/LCPSolver.hpp"

namespace dart {
namespace constraint {

/// \deprecated This header has been deprecated in DART 6.7. Please include
/// PgsBoxedLcpSolver.hpp intead.
///
/// PGSLCPSolver
class PGSLCPSolver : public LCPSolver
{
public:
  /// Constructor
  explicit PGSLCPSolver(s_t _timestep);

  /// Constructor
  virtual ~PGSLCPSolver();

  // Documentation inherited
  void solve(ConstrainedGroup* _group) override;

#ifndef NDEBUG
private:
  /// Return true if the matrix is symmetric
  bool isSymmetric(std::size_t _n, s_t* _A);

  /// Return true if the diagonla block of matrix is symmetric
  bool isSymmetric(std::size_t _n, s_t* _A, std::size_t _begin, std::size_t _end);

  /// Print debug information
  void print(std::size_t _n, s_t* _A, s_t* _x, s_t* _lo, s_t* _hi,
             s_t* _b, s_t* w, int* _findex);
#endif
};

struct PGSOption
{
  int itermax;
  s_t sor_w;
  s_t eps_ea;
  s_t eps_res;
  s_t eps_div;

  void setDefault();
};

bool solvePGS(int n, int nskip, int /*nub*/, s_t* A,
                            s_t* x, s_t * b,
                            s_t * lo, s_t * hi, int * findex,
                            PGSOption * option);


} // namespace constraint
} // namespace dart

#endif  // DART_CONSTRAINT_PGSLCPSOLVER_HPP_

