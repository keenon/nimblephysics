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
#include <dart/biomechanics/StreamingMarkerTraces.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void StreamingMarkerTraces(py::module& m)
{
  ::py::class_<
      dart::biomechanics::StreamingMarkerTraces,
      std::shared_ptr<dart::biomechanics::StreamingMarkerTraces>>(
      m, "StreamingMarkerTraces")
      .def(
          ::py::init<int, int, int, int>(),
          ::py::arg("totalClasses"),
          ::py::arg("numWindows"),
          ::py::arg("stride"),
          ::py::arg("maxMarkersPerTimestep"))
      .def(
          "observeMarkers",
          &dart::biomechanics::StreamingMarkerTraces::observeMarkers,
          ::py::arg("markers"),
          ::py::arg("timestamp"),
          "This method takes in a set of markers, and returns a vector of the "
          "predicted classes for each marker, based on classes we have "
          "predicted for previous markers, and continuity assumptions. It also "
          "returns a 'trace tag' for each marker, that can be used to "
          "associate it with previous continuous observations of the same "
          "marker. The returned vector will be the same length and order as "
          "the input `markers` vector.")
      .def(
          "getTraceFeatures",
          &dart::biomechanics::StreamingMarkerTraces::getTraceFeatures,
          ::py::arg("center") = true,
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
          &dart::biomechanics::StreamingMarkerTraces::observeTraceLogits,
          ::py::arg("logits"),
          ::py::arg("traceIDs"),
          "This method takes in the logits for each point, and the trace IDs "
          "for each point, and updates the internal state of the trace "
          "classifier to reflect the new information.")
      .def(
          "setMaxJoinDistance",
          &dart::biomechanics::StreamingMarkerTraces::setMaxJoinDistance,
          ::py::arg("distance"),
          "This method sets the maximum distance that "
          "can exist between the last head of a trace, and a new marker "
          "position. Markers that are within this distance from a trace are "
          "not guaranteed to be merged (they must be the closest to the "
          "trace), but markers that are further than this distance are "
          "guaranteed to be split into a new trace.")
      .def(
          "setTraceTimeoutMillis",
          &dart::biomechanics::StreamingMarkerTraces::setTraceTimeoutMillis,
          ::py::arg("timeout"),
          "This method sets the timeout for traces. If a "
          "trace has not been updated for this many milliseconds, it will be "
          "removed from the trace list.")
      .def(
          "setFeatureMaxStrideTolerance",
          &dart::biomechanics::StreamingMarkerTraces::
              setFeatureMaxStrideTolerance,
          ::py::arg("tolerance"),
          "This sets the maximum number of milliseconds "
          "that we will tolerate between a stride and a point we are going to "
          "accept as being at that stride.")
      .def(
          "reset",
          &dart::biomechanics::StreamingMarkerTraces::reset,
          "This resets all traces to empty");
}

} // namespace python
} // namespace dart
