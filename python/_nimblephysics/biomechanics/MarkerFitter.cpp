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
#include <dart/biomechanics/MarkerFitter.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerFitter(py::module& m)
{
  ::py::class_<dart::biomechanics::MarkerFitterState>(m, "MarkerFitterState")
      .def_readwrite(
          "bodyScales", &dart::biomechanics::MarkerFitterState::bodyScales)
      .def_readwrite(
          "markerOffsets",
          &dart::biomechanics::MarkerFitterState::markerOffsets)
      .def_readwrite(
          "markerErrorsAtTimesteps",
          &dart::biomechanics::MarkerFitterState::markerErrorsAtTimesteps)
      .def_readwrite(
          "posesAtTimesteps",
          &dart::biomechanics::MarkerFitterState::posesAtTimesteps)
      // Gradient of current state
      .def_readwrite(
          "bodyScalesGrad",
          &dart::biomechanics::MarkerFitterState::bodyScalesGrad)
      .def_readwrite(
          "markerOffsetsGrad",
          &dart::biomechanics::MarkerFitterState::markerOffsetsGrad)
      .def_readwrite(
          "markerErrorsAtTimestepsGrad",
          &dart::biomechanics::MarkerFitterState::markerErrorsAtTimestepsGrad)
      .def_readwrite(
          "posesAtTimestepsGrad",
          &dart::biomechanics::MarkerFitterState::posesAtTimestepsGrad);

  ::py::class_<
      dart::biomechanics::MarkerFitResult,
      std::shared_ptr<dart::biomechanics::MarkerFitResult>>(
      m, "MarkerFitResult")
      .def_readwrite("success", &dart::biomechanics::MarkerFitResult::success)
      .def_readwrite(
          "groupScales", &dart::biomechanics::MarkerFitResult::groupScales)
      .def_readwrite(
          "markerErrors", &dart::biomechanics::MarkerFitResult::markerErrors)
      .def_readwrite("poses", &dart::biomechanics::MarkerFitResult::poses)
      .def_readwrite(
          "posesMatrix", &dart::biomechanics::MarkerFitResult::posesMatrix)
      .def_readwrite(
          "rawMarkerOffsets",
          &dart::biomechanics::MarkerFitResult::rawMarkerOffsets);

  ::py::class_<
      dart::biomechanics::MarkerFitter,
      std::shared_ptr<dart::biomechanics::MarkerFitter>>(m, "MarkerFitter")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::map<
                  std::string,
                  std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"))
      .def(
          "setInitialIKSatisfactoryLoss",
          &dart::biomechanics::MarkerFitter::setInitialIKSatisfactoryLoss,
          ::py::arg("loss"))
      .def(
          "setInitialIKMaxRestarts",
          &dart::biomechanics::MarkerFitter::setInitialIKMaxRestarts,
          ::py::arg("starts"))
      .def(
          "setMaxMarkerOffset",
          &dart::biomechanics::MarkerFitter::setMaxMarkerOffset,
          ::py::arg("offset"))
      .def(
          "setIterationLimit",
          &dart::biomechanics::MarkerFitter::setIterationLimit,
          ::py::arg("iters"))
      .def(
          "setCustomLossAndGrad",
          &dart::biomechanics::MarkerFitter::setCustomLossAndGrad,
          ::py::arg("loss"))
      .def(
          "addZeroConstraint",
          &dart::biomechanics::MarkerFitter::addZeroConstraint,
          ::py::arg("name"),
          ::py::arg("loss"))
      .def(
          "removeZeroConstraint",
          &dart::biomechanics::MarkerFitter::removeZeroConstraint,
          ::py::arg("name"))
      .def(
          "optimize",
          &dart::biomechanics::MarkerFitter::optimize,
          ::py::arg("markerObservations"))
      .def_static(
          "pickSubset",
          &dart::biomechanics::MarkerFitter::pickSubset,
          ::py::arg("markerObservations"),
          ::py::arg("subsetSize"))
      .def(
          "setMarkerIsTracking",
          &dart::biomechanics::MarkerFitter::setMarkerIsTracking,
          ::py::arg("marker"),
          ::py::arg("isTracking") = true)
      .def(
          "getMarkerIsTracking",
          &dart::biomechanics::MarkerFitter::getMarkerIsTracking,
          ::py::arg("marker"))
      .def(
          "setTriadsToTracking",
          &dart::biomechanics::MarkerFitter::setTriadsToTracking)
      .def("getNumMarkers", &dart::biomechanics::MarkerFitter::getNumMarkers);
}

} // namespace python
} // namespace dart
