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

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "TestHelpers.hpp"

#include "dart/common/Composite.hpp"
#include "dart/common/EmbeddedAspect.hpp"
#include "dart/common/SpecializedForAspect.hpp"
#include "dart/common/Subject.hpp"
#include "dart/common/sub_ptr.hpp"

#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/EulerJoint.hpp"

using namespace af;

TEST(ArrayFire, HelloWorld)
{
  try {
    // Select a device and display arrayfire info
    int device = 0;
    af::setDevice(device);
    af::info();

    printf("Create a 5-by-3 matrix of random floats on the GPU\n");
    array A = randu(5,3, f32);
    af_print(A);

    printf("Element-wise arithmetic\n");
    array B = sin(A) + 1.5;
    af_print(B);

    printf("Negate the first three elements of second column\n");
    B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
    af_print(B);

    printf("Fourier transform the result\n");
    array C = fft(B);
    af_print(C);

    printf("Grab last row\n");
    array c = C.row(end);
    af_print(c);

    printf("Scan Test\n");
    dim4 dims(16, 4, 1, 1);
    array r = constant(2, dims);
    af_print(r);

//    printf("Scan\n");
//    array S = af::scan(r, 0, AF_BINARY_MUL);
//    af_print(S);

    printf("Create 2-by-3 matrix from host data\n");
    float d[] = { 1, 2, 3, 4, 5, 6 };
    array D(2, 3, d, afHost);
    af_print(D);

    printf("Copy last column onto first\n");
    D.col(0) = D.col(end);
    af_print(D);

    // Sort A
    printf("Sort A and print sorted array and corresponding indices\n");
    array vals, inds;
    sort(vals, inds, A);
    af_print(vals);
    af_print(inds);
  } catch (af::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw;
  }
}
