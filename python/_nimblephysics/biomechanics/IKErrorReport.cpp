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
#include <dart/biomechanics/IKErrorReport.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

/*
  IKErrorReport(
      std::shared_ptr<dynamics::Skeleton> skel,
      dynamics::MarkerMap markers,
      Eigen::MatrixXs poses,
      std::vector<std::map<std::string, Eigen::Vector3s>> observations);

  std::vector<s_t> sumSquaredError;
  std::vector<s_t> rootMeanSquaredError;
  std::vector<s_t> maxError;
  s_t averageRootMeanSquaredError;
  s_t averageSumSquaredError;
  s_t averageMaxError;
*/

void IKErrorReport(py::module& m)
{
  ::py::class_<dart::biomechanics::IKErrorReport>(m, "IKErrorReport")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              dynamics::MarkerMap,
              Eigen::MatrixXs,
              std::vector<std::map<std::string, Eigen::Vector3s>>>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"),
          ::py::arg("poses"),
          ::py::arg("observations"))
      .def(
          "printReport",
          &dart::biomechanics::IKErrorReport::printReport,
          ::py::arg("limitTimesteps") = -1)
      .def(
          "saveCSVMarkerErrorReport",
          &dart::biomechanics::IKErrorReport::saveCSVMarkerErrorReport,
          ::py::arg("path"))
      .def(
          "getSortedMarkerRMSE",
          &dart::biomechanics::IKErrorReport::getSortedMarkerRMSE)
      .def_readwrite(
          "sumSquaredError",
          &dart::biomechanics::IKErrorReport::sumSquaredError)
      .def_readwrite(
          "rootMeanSquaredError",
          &dart::biomechanics::IKErrorReport::rootMeanSquaredError)
      .def_readwrite("maxError", &dart::biomechanics::IKErrorReport::maxError)
      .def_readwrite(
          "averageRootMeanSquaredError",
          &dart::biomechanics::IKErrorReport::averageRootMeanSquaredError)
      .def_readwrite(
          "averageSumSquaredError",
          &dart::biomechanics::IKErrorReport::averageSumSquaredError)
      .def_readwrite(
          "averageMaxError",
          &dart::biomechanics::IKErrorReport::averageMaxError);
}

} // namespace python
} // namespace dart
