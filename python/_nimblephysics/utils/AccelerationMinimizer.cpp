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

#include <dart/utils/AccelerationMinimizer.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void AccelerationMinimizer(py::module& m)
{
  ::py::class_<dart::utils::AccelerationMinimizer>(m, "AccelerationMinimizer")
      .def(
          ::py::init<int, s_t, s_t, s_t, s_t, s_t, s_t, int>(),
          ::py::arg("numTimesteps"),
          ::py::arg("smoothingWeight") = 1.0,
          ::py::arg("regularizationWeight") = 0.01,
          ::py::arg("startPositionZeroWeight") = 0.0,
          ::py::arg("endPositionZeroWeight") = 0.0,
          ::py::arg("startVelocityZeroWeight") = 0.0,
          ::py::arg("endVelocityZeroWeight") = 0.0,
          ::py::arg("numIterations") = 10000)
      .def(
          "minimize",
          &dart::utils::AccelerationMinimizer::minimize,
          ::py::arg("series"))
      .def(
          "setDebugIterationBackoff",
          &dart::utils::AccelerationMinimizer::setDebugIterationBackoff,
          ::py::arg("iterations"))
      .def(
          "setNumIterationsBackoff",
          &dart::utils::AccelerationMinimizer::setNumIterationsBackoff,
          ::py::arg("series"))
      .def(
          "setConvergenceTolerance",
          &dart::utils::AccelerationMinimizer::setConvergenceTolerance,
          ::py::arg("tolerance"));
}

} // namespace python
} // namespace dart