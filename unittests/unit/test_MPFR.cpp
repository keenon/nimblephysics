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
#include <unsupported/Eigen/MPRealSupport>

#include "TestHelpers.hpp"
#include "mpreal.h"

TEST(MPFR, BASICS)
{
  using mpfr::mpreal;
  using std::cout;
  using std::endl;

  // Required precision of computations in decimal digits
  // Play with it to check different precisions
  const int digits = 50;

  // Setup default precision for all subsequent computations
  // MPFR accepts precision in bits - so we do the conversion
  mpreal::set_default_prec(mpfr::digits2bits(digits));

  // Compute all the vital characteristics of mpreal (in current precision)
  // Analogous to lamch from LAPACK
  const mpreal one = 1.0;
  const mpreal zero = 0.0;
  const mpreal eps = std::numeric_limits<mpreal>::epsilon();
  const int base = std::numeric_limits<mpreal>::radix;
  const mpreal prec = eps * base;
  const int bindigits = std::numeric_limits<
      mpreal>::digits(); // eqv. to mpfr::mpreal::get_default_prec();
  const mpreal rnd = std::numeric_limits<mpreal>::round_error();
  const mpreal maxval = std::numeric_limits<mpreal>::max();
  const mpreal minval = std::numeric_limits<mpreal>::min();
  const mpreal small = one / maxval;
  const mpreal sfmin = (small > minval) ? small * (one + eps) : minval;
  const mpreal round = std::numeric_limits<mpreal>::round_style();
  const int min_exp = std::numeric_limits<mpreal>::min_exponent;
  const mpreal underflow = std::numeric_limits<mpreal>::min();
  const int max_exp = std::numeric_limits<mpreal>::max_exponent;
  const mpreal overflow = std::numeric_limits<mpreal>::max();

  // Additionally compute pi with required accuracy - just for fun :)
  const mpreal pi = mpfr::const_pi();

  cout.precision(digits); // Show all the digits
  cout << "pi         =    " << pi << endl;
  cout << "eps        =    " << eps << endl;
  cout << "base       =    " << base << endl;
  cout << "prec       =    " << prec << endl;
  cout << "b.digits   =    " << bindigits << endl;
  cout << "rnd        =    " << rnd << endl;
  cout << "maxval     =    " << maxval << endl;
  cout << "minval     =    " << minval << endl;
  cout << "small      =    " << small << endl;
  cout << "sfmin      =    " << sfmin << endl;
  cout << "1/sfmin    =    " << 1 / sfmin << endl;
  cout << "round      =    " << round << endl;
  cout << "max_exp    =    " << max_exp << endl;
  cout << "min_exp    =    " << min_exp << endl;
  cout << "underflow  =    " << underflow << endl;
  cout << "overflow   =    " << overflow << endl;
}

TEST(MPFR, EIGEN)
{
  using namespace Eigen;
  using mpfr::mpreal;
  using std::cout;
  using std::endl;

  // set precision to 256 bits (double has only 53 bits)
  mpreal::set_default_prec(256);
  // Declare matrix and vector types with multi-precision scalar type
  typedef Matrix<mpreal, Dynamic, Dynamic> MatrixXmp;
  typedef Matrix<mpreal, Dynamic, 1> VectorXmp;

  MatrixXmp A = MatrixXmp::Random(100, 100);
  VectorXmp b = VectorXmp::Random(100);

  // Solve Ax=b using LU
  VectorXmp x = A.lu().solve(b);
  std::cout << "relative error: " << (A * x - b).norm() / b.norm() << std::endl;
}

TEST(MPFR, EIGEN_DOUBLE_AND_BACK)
{
  using namespace Eigen;
  using mpfr::mpreal;
  using std::cout;
  using std::endl;

  // set precision to 256 bits (double has only 53 bits)
  mpreal::set_default_prec(256);
  // Declare matrix and vector types with multi-precision scalar type
  typedef Matrix<mpreal, Dynamic, Dynamic> MatrixXmp;
  typedef Matrix<mpreal, Dynamic, 1> VectorXmp;

  MatrixXmp A = MatrixXmp::Random(100, 100);
  VectorXmp b = VectorXmp::Random(100);

  // Solve Ax=b using LU
  VectorXmp x = A.lu().solve(b);
  std::cout << "relative error: " << (A * x - b).norm() / b.norm() << std::endl;

  Eigen::MatrixXd A_lb = A.cast<double>();
  Eigen::VectorXd b_lp = b.cast<double>();
  Eigen::VectorXd x_lp = x.cast<double>();
  Eigen::VectorXd x_lp_solve = A_lb.lu().solve(b_lp);

  VectorXmp diff = x_lp_solve.cast<mpreal>() - x;
  std::cout << "comparative diff: " << diff.norm() << std::endl;
}