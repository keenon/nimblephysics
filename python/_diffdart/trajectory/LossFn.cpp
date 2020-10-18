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

#include <dart/trajectory/AbstractShot.hpp>
#include <dart/trajectory/MultiShot.hpp>
#include <dart/trajectory/TrajectoryConstants.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void LossFn(py::module& m)
{
  ::py::class_<dart::trajectory::LossFn>(m, "LossFn")
      .def(::py::init<>())
      .def(::py::init<dart::trajectory::TrajectoryLossFn>(), ::py::arg("loss"))
      .def(
          ::py::init<
              dart::trajectory::TrajectoryLossFn,
              dart::trajectory::TrajectoryLossFnAndGrad>(),
          ::py::arg("loss"),
          ::py::arg("lossFnAndGrad"))
      .def(
          "getLoss",
          &dart::trajectory::LossFn::getLoss,
          ::py::arg("rollout"),
          ::py::arg("perfLog") = nullptr)
      .def(
          "getLossAndGradient",
          &dart::trajectory::LossFn::getLossAndGradient,
          ::py::arg("rollout"),
          ::py::arg("gradWrtRollout"),
          ::py::arg("perfLog") = nullptr)
      .def(
          "setUpperBound",
          &dart::trajectory::LossFn::setUpperBound,
          ::py::arg("upperBound"))
      .def("getUpperBound", &dart::trajectory::LossFn::getUpperBound)
      .def(
          "setLowerBound",
          &dart::trajectory::LossFn::setLowerBound,
          ::py::arg("lowerBound"))
      .def("getLowerBound", &dart::trajectory::LossFn::getLowerBound);
}

} // namespace python
} // namespace dart
