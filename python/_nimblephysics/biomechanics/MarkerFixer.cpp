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
#include <dart/biomechanics/MarkerFixer.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerFixer(py::module& m)
{
  ::py::class_<
      dart::biomechanics::MarkersErrorReport,
      std::shared_ptr<dart::biomechanics::MarkersErrorReport>>(
      m, "MarkersErrorReport")
      .def_readwrite(
          "warnings", &dart::biomechanics::MarkersErrorReport::warnings)
      .def_readwrite("info", &dart::biomechanics::MarkersErrorReport::info)
      .def_readwrite(
          "markerObservationsAttemptedFixed",
          &dart::biomechanics::MarkersErrorReport::
              markerObservationsAttemptedFixed)
      .def_readwrite(
          "droppedMarkerWarnings",
          &dart::biomechanics::MarkersErrorReport::droppedMarkerWarnings)
      .def_readwrite(
          "markersRenamedFromTo",
          &dart::biomechanics::MarkersErrorReport::markersRenamedFromTo)
      .def(
          "getNumTimesteps",
          &dart::biomechanics::MarkersErrorReport::getNumTimesteps)
      .def(
          "getMarkerMapOnTimestep",
          &dart::biomechanics::MarkersErrorReport::getMarkerMapOnTimestep,
          ::py::arg("t"))
      .def(
          "getMarkerNamesOnTimestep",
          &dart::biomechanics::MarkersErrorReport::getMarkerNamesOnTimestep,
          ::py::arg("t"))
      .def(
          "getMarkerPositionOnTimestep",
          &dart::biomechanics::MarkersErrorReport::getMarkerPositionOnTimestep,
          ::py::arg("t"),
          ::py::arg("marker"));

  ::py::class_<dart::biomechanics::MarkerFixer>(m, "MarkerFixer")
      .def_static(
          "generateDataErrorsReport",
          &dart::biomechanics::MarkerFixer::generateDataErrorsReport,
          ::py::arg("immutableMarkerObservations"),
          ::py::arg("dt"),
          ::py::arg("dropProlongedStillness") = false,
          ::py::arg("rippleReduce") = true,
          ::py::arg("rippleReduceUseSparse") = true,
          ::py::arg("rippleReduceUseIterativeSolver") = true,
          ::py::arg("rippleReduceSolverIterations") = 1e5);
}

} // namespace python
} // namespace dart
