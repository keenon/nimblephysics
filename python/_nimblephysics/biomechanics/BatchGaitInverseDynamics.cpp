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

#include <Eigen/Dense>
#include <dart/biomechanics/BatchGaitInverseDynamics.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void BatchGaitInverseDynamics(py::module& m)
{
  ::py::class_<
      dart::biomechanics::BatchGaitInverseDynamics,
      std::shared_ptr<dart::biomechanics::BatchGaitInverseDynamics>>(
      m, "BatchGaitInverseDynamics")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              Eigen::MatrixXs,
              std::vector<const dynamics::BodyNode*>,
              Eigen::Vector3s,
              double,
              int,
              double,
              double,
              double,
              double,
              double>(),
          ::py::arg("skeleton"),
          ::py::arg("poses"),
          ::py::arg("groundContactBodies"),
          ::py::arg("groundNormal"),
          ::py::arg("tileSize"),
          ::py::arg("maxSectionLength") = 100,
          ::py::arg("smoothingWeight") = 1.0,
          ::py::arg("minTorqueWeight") = 1.0,
          ::py::arg("prevContactWeight") = 0.1,
          ::py::arg("blendWeight") = 1.0,
          ::py::arg("blendSteepness") = 10.0)
      .def(
          "numTimesteps",
          &dart::biomechanics::BatchGaitInverseDynamics::numTimesteps)
      .def(
          "getSectionForTimestep",
          &dart::biomechanics::BatchGaitInverseDynamics::getSectionForTimestep,
          ::py::arg("timestep"))
      .def(
          "getContactBodiesAtTimestep",
          &dart::biomechanics::BatchGaitInverseDynamics::
              getContactBodiesAtTimestep,
          ::py::arg("timestep"))
      .def(
          "getContactWrenchesAtTimestep",
          &dart::biomechanics::BatchGaitInverseDynamics::
              getContactWrenchesAtTimestep,
          ::py::arg("timestep"))
      .def(
          "debugLilypadToGUI",
          &dart::biomechanics::BatchGaitInverseDynamics::debugLilypadToGUI,
          ::py::arg("gui"),
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "debugTimestepToGUI",
          &dart::biomechanics::BatchGaitInverseDynamics::debugTimestepToGUI,
          ::py::arg("gui"),
          ::py::arg("timesteps"),
          ::py::call_guard<py::gil_scoped_release>());
}

} // namespace python
} // namespace dart
