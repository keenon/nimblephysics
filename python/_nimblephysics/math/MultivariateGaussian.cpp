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

#include "dart/include_eigen.hpp"
#include <dart/math/MultivariateGaussian.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MultivariateGaussian(py::module& m)
{
  ::py::class_<
      dart::math::MultivariateGaussian,
      std::shared_ptr<dart::math::MultivariateGaussian>>(
      m, "MultivariateGaussian")
      .def(
          ::py::init<
              std::vector<std::string>,
              Eigen::VectorXs,
              Eigen::MatrixXs>(),
          ::py::arg("variables"),
          ::py::arg("mu"),
          ::py::arg("cov"))
      .def("debugToStdout", &dart::math::MultivariateGaussian::debugToStdout)
      .def("getMu", &dart::math::MultivariateGaussian::getMu)
      .def("getCov", &dart::math::MultivariateGaussian::getCov)
      .def(
          "getLogNormalizationConstant",
          &dart::math::MultivariateGaussian::getLogNormalizationConstant)
      .def(
          "getMean",
          &dart::math::MultivariateGaussian::getMean,
          ::py::arg("variable"))
      .def(
          "convertFromMap",
          &dart::math::MultivariateGaussian::convertFromMap,
          ::py::arg("values"))
      .def(
          "convertToMap",
          &dart::math::MultivariateGaussian::convertToMap,
          ::py::arg("values"))
      .def(
          "computePDF",
          &dart::math::MultivariateGaussian::computePDF,
          ::py::arg("values"))
      .def(
          "computeLogPDF",
          &dart::math::MultivariateGaussian::computeLogPDF,
          ::py::arg("values"),
          ::py::arg("normalized") = true)
      .def(
          "computeLogPDFGrad",
          &dart::math::MultivariateGaussian::computeLogPDFGrad,
          ::py::arg("x"))
      .def(
          "getVariableNameAtIndex",
          &dart::math::MultivariateGaussian::getVariableNameAtIndex,
          ::py::arg("i"))
      .def(
          "getVariableNames",
          &dart::math::MultivariateGaussian::getVariableNames)
      .def(
          "condition",
          &dart::math::MultivariateGaussian::condition,
          ::py::arg("observedValues"))
      .def(
          "getObservedIndices",
          &dart::math::MultivariateGaussian::getObservedIndices,
          ::py::arg("observedValues"))
      .def(
          "getUnobservedIndices",
          &dart::math::MultivariateGaussian::getUnobservedIndices,
          ::py::arg("observedValues"))
      .def(
          "getMuSubset",
          &dart::math::MultivariateGaussian::getMuSubset,
          ::py::arg("indices"))
      .def(
          "getCovSubset",
          &dart::math::MultivariateGaussian::getCovSubset,
          ::py::arg("rowIndices"),
          ::py::arg("colIndices"))
      .def_static(
          "loadFromCSV",
          &dart::math::MultivariateGaussian::loadFromCSV,
          ::py::arg("file"),
          ::py::arg("columns"),
          ::py::arg("units") = 1.0);
}

} // namespace python
} // namespace dart
