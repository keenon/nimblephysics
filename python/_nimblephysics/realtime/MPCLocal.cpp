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
#include <dart/realtime/MPCLocal.hpp>
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

void MPCLocal(py::module& m)
{
  ::py::class_<
      dart::realtime::MPCLocal,
      dart::realtime::MPC,
      std::shared_ptr<dart::realtime::MPCLocal>>(m, "MPCLocal")
      .def(
          ::py::init<
              std::shared_ptr<dart::simulation::World>,
              std::shared_ptr<dart::trajectory::LossFn>,
              int>(),
          ::py::arg("world"),
          ::py::arg("loss"),
          ::py::arg("planningHorizonMillis"))
      .def("setLoss", &dart::realtime::MPCLocal::setLoss, ::py::arg("loss"))
      .def(
          "setOptimizer",
          &dart::realtime::MPCLocal::setOptimizer,
          ::py::arg("optimizer"))
      .def(
          "setProblem",
          &dart::realtime::MPCLocal::setProblem,
          ::py::arg("problem"))
      .def("getProblem", &dart::realtime::MPCLocal::getProblem)
      .def("getOptimizer", &dart::realtime::MPCLocal::getOptimizer)
      .def(
          "getRemainingPlanBufferMillis",
          &dart::realtime::MPCLocal::getRemainingPlanBufferMillis)
      .def(
          "setSilent",
          &dart::realtime::MPCLocal::setSilent,
          ::py::arg("silent"))
      .def(
          "setEnableLineSearch",
          &dart::realtime::MPCLocal::setEnableLineSearch,
          ::py::arg("enabled"))
      .def(
          "setEnableOptimizationGuards",
          &dart::realtime::MPCLocal::setEnableOptimizationGuards,
          ::py::arg("enabled"))
      .def(
          "setRecordIterations",
          &dart::realtime::MPCLocal::setRecordIterations,
          ::py::arg("enabled"))
      .def("getMaxIterations", &dart::realtime::MPCLocal::getMaxIterations)
      .def(
          "setMaxIterations",
          &dart::realtime::MPCLocal::setMaxIterations,
          ::py::arg("maxIterations"))
      .def(
          "recordGroundTruthState",
          &dart::realtime::MPCLocal::recordGroundTruthState,
          ::py::arg("time"),
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("mass"))
      .def(
          "recordGroundTruthStateNow",
          &dart::realtime::MPCLocal::recordGroundTruthStateNow,
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("mass"))
      .def(
          "optimizePlan",
          &dart::realtime::MPCLocal::optimizePlan,
          ::py::arg("now"))
      .def(
          "adjustPerformance",
          &dart::realtime::MPCLocal::adjustPerformance,
          ::py::arg("lastOptimizationTimeMillis"))
      .def("start", &dart::realtime::MPCLocal::start)
      .def("stop", &dart::realtime::MPCLocal::stop)
      .def(
          "serve",
          &dart::realtime::MPCLocal::serve,
          ::py::arg("port"),
          "A blocking call - this starts a gRPC server that clients can "
          "connect to to get MPC computations done remotely",
          ::py::call_guard<py::gil_scoped_release>())
      .def("getCurrentSolution", &dart::realtime::MPCLocal::getCurrentSolution)
      .def(
          "registerReplaningListener",
          &dart::realtime::MPCLocal::registerReplanningListener,
          ::py::arg("replanListener"));
}

} // namespace python
} // namespace dart
