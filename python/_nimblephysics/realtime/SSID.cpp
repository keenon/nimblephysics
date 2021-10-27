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

#include <dart/realtime/SSID.hpp>
#include <dart/simulation/World.hpp>
#include <dart/trajectory/LossFn.hpp>
#include <dart/trajectory/Optimizer.hpp>
#include <dart/trajectory/Solution.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void SSID(py::module& m)
{
  ::py::class_<dart::realtime::SSID>(m, "SSID")
      .def(
          ::py::init<
              std::shared_ptr<dart::simulation::World>,
              std::shared_ptr<dart::trajectory::LossFn>,
              int,
              int>(),
          ::py::arg("world"),
          ::py::arg("loss"),
          ::py::arg("planningHorizonMillis"),
          ::py::arg("sensorDim"))
      .def("setLoss", &dart::realtime::SSID::setLoss, ::py::arg("loss"))
      .def(
          "setOptimizer",
          &dart::realtime::SSID::setOptimizer,
          ::py::arg("optimizer"))
      .def(
          "setProblem", &dart::realtime::SSID::setProblem, ::py::arg("problem"))
      .def("getProblem", &dart::realtime::SSID::getProblem)
      .def("getOptimizer", &dart::realtime::SSID::getOptimizer)
      .def(
          "setInitialPosEstimator",
          &dart::realtime::SSID::setInitialPosEstimator,
          ::py::arg("initialPosEstimator"))
      .def(
          "registerSensorsNow",
          &dart::realtime::SSID::registerSensorsNow,
          ::py::arg("sensors"))
      .def(
          "registerControlsNow",
          &dart::realtime::SSID::registerControlsNow,
          ::py::arg("controls"))
      .def(
          "registerSensors",
          &dart::realtime::SSID::registerSensors,
          ::py::arg("now"),
          ::py::arg("sensors"))
      .def(
          "registerControls",
          &dart::realtime::SSID::registerControls,
          ::py::arg("now"),
          ::py::arg("controls"))
      .def(
          "start",
          &dart::realtime::SSID::start,
          ::py::call_guard<py::gil_scoped_release>())
      .def("stop", &dart::realtime::SSID::stop)
      .def(
          "runInference",
          &dart::realtime::SSID::runInference,
          ::py::arg("startTime"))
      .def(
          "registerInferListener",
          &dart::realtime::SSID::registerInferListener,
          ::py::arg("inferListener"));
}

} // namespace python
} // namespace dart
