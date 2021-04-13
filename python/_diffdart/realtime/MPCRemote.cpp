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

#include <dart/realtime/MPC.hpp>
#include <dart/realtime/MPCRemote.hpp>
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

void MPCRemote(py::module& m)
{
  ::py::class_<
      dart::realtime::MPCRemote,
      dart::realtime::MPC,
      std::shared_ptr<dart::realtime::MPCRemote>>(m, "MPCRemote")
      .def(
          ::py::init<std::string, int, int, int, int>(),
          ::py::arg("host"),
          ::py::arg("port"),
          ::py::arg("dofs"),
          ::py::arg("steps"),
          ::py::arg("millisPerStep"))
      .def(
          ::py::init<dart::realtime::MPCLocal&, int>(),
          ::py::arg("local"),
          ::py::arg("ignored") = 0)
      .def(
          "getRemainingPlanBufferMillis",
          &dart::realtime::MPCRemote::getRemainingPlanBufferMillis)
      .def(
          "recordGroundTruthState",
          &dart::realtime::MPCRemote::recordGroundTruthState,
          ::py::arg("time"),
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("mass"))
      .def(
          "recordGroundTruthStateNow",
          &dart::realtime::MPCRemote::recordGroundTruthStateNow,
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("mass"))
      .def("getControlForce", &dart::realtime::MPCRemote::getControlForce, ::py::arg("now"))
      .def(
          "start",
          &dart::realtime::MPCRemote::start,
          ::py::call_guard<py::gil_scoped_release>())
      .def("stop", &dart::realtime::MPCRemote::stop)
      .def(
          "registerReplaningListener",
          &dart::realtime::MPCRemote::registerReplanningListener,
          ::py::arg("replanListener"));
}

} // namespace python
} // namespace dart
