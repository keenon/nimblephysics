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

#include <memory>

#include <Eigen/Dense>
#include <dart/biomechanics/StreamingMocapLab.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/biomechanics/StreamingIK.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void StreamingMocapLab(py::module& m)
{
  ::py::class_<
      dart::biomechanics::StreamingMocapLab,
      std::shared_ptr<dart::biomechanics::StreamingMocapLab>>(
      m, "StreamingMocapLab")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"))
      .def(
          "startSolverThread",
          &dart::biomechanics::StreamingMocapLab::startSolverThread,
          "This method starts the thread that runs the IK continuously.")
      .def(
          "startGUIThread",
          &dart::biomechanics::StreamingMocapLab::startGUIThread,
          ::py::arg("gui"),
          "This method starts a thread that periodically updates a GUI server "
          "state, though at a much lower framerate than the IK solver.")
      .def(
          "setAnthropometricPrior",
          &dart::biomechanics::StreamingMocapLab::setAnthropometricPrior,
          ::py::arg("prior"),
          ::py::arg("priorWeight") = 1.0,
          "This sets an anthropometric prior used to help condition the body "
          "to keep reasonable scalings.")
      .def(
          "listenToCortex",
          &dart::biomechanics::StreamingMocapLab::listenToCortex,
          ::py::arg("host"),
          ::py::arg("port"),
          "This method establishes a link to Cortex, and listens for real-time "
          "observations of markers and force plate data.")
      .def(
          "manuallyObserveMarkers",
          &dart::biomechanics::StreamingMocapLab::manuallyObserveMarkers,
          ::py::arg("markers"),
          ::py::arg("timestamp"),
          "This method allows tests to manually input a set of markers, rather "
          "than waiting for Cortex to send them.")
      .def(
          "getTraceFeatures",
          &dart::biomechanics::StreamingMocapLab::getTraceFeatures,
          ::py::arg("numWindows"),
          ::py::arg("windowDuration"),
          "This method returns the features that we used to predict the "
          "classes "
          "of the markers. The first element of the pair is the features "
          "(which "
          "are trace points concatenated with the time, as measured in integer "
          "units of 'windowDuration', backwards from now), and the second is "
          "the "
          "trace ID for each point, so that we can correctly assign logit "
          "outputs back to the traces.")
      .def(
          "observeTraceLogits",
          &dart::biomechanics::StreamingMocapLab::observeTraceLogits,
          ::py::arg("logits"),
          ::py::arg("traceIDs"),
          "This method takes in the logits for each point, and the trace IDs "
          "for each point, and updates the internal state of the trace "
          "classifier "
          "to reflect the new information.")
      .def(
          "reset",
          &dart::biomechanics::StreamingMocapLab::reset,
          "This method resets the state of the mocap lab, including the IK "
          "and the marker traces.")
      .def(
          "getMarkerTraces",
          &dart::biomechanics::StreamingMocapLab::getMarkerTraces)
      .def("getIK", &dart::biomechanics::StreamingMocapLab::getMarkerTraces);
}

} // namespace python
} // namespace dart
