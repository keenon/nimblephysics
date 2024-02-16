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

#include "dart/biomechanics/StreamingIK.hpp"

#include <memory>

#include <Eigen/Dense>
#include <dart/biomechanics/StreamingMarkerTraces.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void StreamingIK(py::module& m)
{
  ::py::class_<
      dart::biomechanics::StreamingIK,
      std::shared_ptr<dart::biomechanics::StreamingIK>>(m, "StreamingIK")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"))
      .def(
          "startSolverThread",
          &dart::biomechanics::StreamingIK::startSolverThread,
          "This method starts the thread that runs the IK continuously.")
      .def(
          "startGUIThread",
          &dart::biomechanics::StreamingIK::startGUIThread,
          ::py::arg("gui"),
          "This method starts a thread that periodically updates a GUI server "
          "state, though at a much lower framerate than the IK solver.")
      .def(
          "observeMarkers",
          &dart::biomechanics::StreamingIK::observeMarkers,
          ::py::arg("markers"),
          ::py::arg("classes"),
          ::py::arg("timestamp"),
          ::py::arg("copTorqueForces") = std::vector<Eigen::Vector9s>(),
          "This method takes in a set of markers, along with their assigned "
          "classes, and updates the targets for the IK to match the observed "
          "markers.")
      .def(
          "setAnthropometricPrior",
          &dart::biomechanics::StreamingIK::setAnthropometricPrior,
          ::py::arg("prior"),
          ::py::arg("priorWeight") = 1.0,
          "This sets an anthropometric prior used to help condition the body "
          "to keep reasonable scalings.")
      .def(
          "estimateState",
          &dart::biomechanics::StreamingIK::estimateState,
          ::py::arg("now"),
          ::py::arg("numHistory") = 20,
          ::py::arg("polynomialDegree") = 3)
      .def(
          "reset",
          &dart::biomechanics::StreamingIK::reset,
          "This method allows tests to manually input a set of markers, rather "
          "than waiting for Cortex to send them.");
}

} // namespace python
} // namespace dart
