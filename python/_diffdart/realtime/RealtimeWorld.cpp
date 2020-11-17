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

#include <dart/realtime/RealtimeWorld.hpp>
#include <dart/simulation/World.hpp>
#include <dart/trajectory/TrajectoryRollout.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void RealtimeWorld(py::module& m)
{
  ::py::class_<dart::realtime::RealtimeWorld>(m, "RealtimeWorld")
      .def(
          ::py::init<
              std::shared_ptr<simulation::World>,
              std::function<Eigen::VectorXd()>,
              std::function<void(
                  Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)> >(),
          ::py::arg("world"),
          ::py::arg("getForces"),
          ::py::arg("recordState"))
      .def("start", &dart::realtime::RealtimeWorld::start)
      .def("stop", &dart::realtime::RealtimeWorld::stop)
      .def("serve", &dart::realtime::RealtimeWorld::serve, ::py::arg("port"))
      .def("stopServing", &dart::realtime::RealtimeWorld::stopServing)
      .def(
          "registerConnectionListener",
          &dart::realtime::RealtimeWorld::registerConnectionListener,
          ::py::arg("listener"))
      .def(
          "registerShutdownListener",
          &dart::realtime::RealtimeWorld::registerShutdownListener,
          ::py::arg("listener"))
      .def(
          "registerKeydownListener",
          &dart::realtime::RealtimeWorld::registerKeydownListener,
          ::py::arg("listener"))
      .def(
          "registerKeyupListener",
          &dart::realtime::RealtimeWorld::registerKeyupListener,
          ::py::arg("listener"))
      .def(
          "registerPreStepListener",
          &dart::realtime::RealtimeWorld::registerPreStepListener,
          ::py::arg("listener"))
      .def(
          "displayMPCPlan",
          &dart::realtime::RealtimeWorld::displayMPCPlan,
          ::py::arg("rollout"))
      .def(
          "registerTiming",
          &dart::realtime::RealtimeWorld::registerTiming,
          ::py::arg("key"),
          ::py::arg("value"),
          ::py::arg("units"))
      .def("timingsToJson", &dart::realtime::RealtimeWorld::timingsToJson);
}

} // namespace python
} // namespace dart
