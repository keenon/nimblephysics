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
#include <dart/biomechanics/Anthropometrics.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void Anthropometrics(py::module& m)
{
  ::py::class_<
      dart::biomechanics::Anthropometrics,
      std::shared_ptr<dart::biomechanics::Anthropometrics>>(
      m, "Anthropometrics")
      .def_static(
          "loadFromFile",
          [](std::string path) {
            return dart::biomechanics::Anthropometrics::loadFromFile(path);
          },
          ::py::arg("uri"))
      .def(
          "debugToGUI",
          &dart::biomechanics::Anthropometrics::debugToGUI,
          ::py::arg("server"),
          ::py::arg("skel"))
      .def(
          "addMetric",
          &dart::biomechanics::Anthropometrics::addMetric,
          ::py::arg("name"),
          ::py::arg("bodyPose"),
          ::py::arg("bodyA"),
          ::py::arg("offsetA"),
          ::py::arg("bodyB"),
          ::py::arg("offsetB"),
          ::py::arg("axis") = Eigen::Vector3s::Zero())
      .def(
          "getMetricNames",
          &dart::biomechanics::Anthropometrics::getMetricNames)
      .def(
          "setDistribution",
          &dart::biomechanics::Anthropometrics::setDistribution,
          ::py::arg("dist"))
      .def(
          "getDistribution",
          &dart::biomechanics::Anthropometrics::getDistribution)
      .def(
          "condition",
          &dart::biomechanics::Anthropometrics::condition,
          ::py::arg("observedValues"))
      .def(
          "measure",
          &dart::biomechanics::Anthropometrics::measure,
          ::py::arg("skel"))
      .def(
          "getPDF",
          &dart::biomechanics::Anthropometrics::getPDF,
          ::py::arg("skel"))
      .def(
          "getLogPDF",
          &dart::biomechanics::Anthropometrics::getLogPDF,
          ::py::arg("skel"),
          ::py::arg("normalized") = true)
      .def(
          "getGradientOfLogPDFWrtBodyScales",
          &dart::biomechanics::Anthropometrics::
              getGradientOfLogPDFWrtBodyScales,
          ::py::arg("skel"))
      .def(
          "getGradientOfLogPDFWrtGroupScales",
          &dart::biomechanics::Anthropometrics::
              getGradientOfLogPDFWrtGroupScales,
          ::py::arg("skel"));
}

} // namespace python
} // namespace dart
