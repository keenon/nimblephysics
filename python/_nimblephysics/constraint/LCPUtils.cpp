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

#include <dart/constraint/BoxedLcpSolver.hpp>
#include <dart/constraint/LCPUtils.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void LCPUtils(py::module& m)
{
  m.def(
      "isLCPSolutionValid",
      &dart::constraint::LCPUtils::isLCPSolutionValid,
      ::py::arg("mA"),
      ::py::arg("mX"),
      ::py::arg("mB"),
      ::py::arg("mHi"),
      ::py::arg("mLo"),
      ::py::arg("mFIndex"),
      ::py::arg("ignoreFrictionIndices"));
  m.def(
      "getLCPSolutionTypes",
      &dart::constraint::LCPUtils::getLCPSolutionTypes,
      ::py::arg("mA"),
      ::py::arg("mX"),
      ::py::arg("mB"),
      ::py::arg("mHi"),
      ::py::arg("mLo"),
      ::py::arg("mFIndex"),
      ::py::arg("ignoreFrictionIndices"));
  m.def(
      "getLCPSolutionType",
      &dart::constraint::LCPUtils::getLCPSolutionType,
      ::py::arg("i"),
      ::py::arg("mA"),
      ::py::arg("mX"),
      ::py::arg("mB"),
      ::py::arg("mHi"),
      ::py::arg("mLo"),
      ::py::arg("mFIndex"),
      ::py::arg("ignoreFrictionIndices"));
  ::py::enum_<dart::constraint::LCPSolutionType>(m, "LCPSolutionType")
      .value("SUCCESS", dart::constraint::LCPSolutionType::SUCCESS)
      .value(
          "FAILURE_IGNORE_FRICTION",
          dart::constraint::LCPSolutionType::FAILURE_IGNORE_FRICTION)
      .value(
          "FAILURE_LOWER_BOUND",
          dart::constraint::LCPSolutionType::FAILURE_LOWER_BOUND)
      .value(
          "FAILURE_UPPER_BOUND",
          dart::constraint::LCPSolutionType::FAILURE_UPPER_BOUND)
      .value(
          "FAILURE_WITHIN_BOUNDS",
          dart::constraint::LCPSolutionType::FAILURE_WITHIN_BOUNDS)
      .value(
          "FAILURE_OUT_OF_BOUNDS",
          dart::constraint::LCPSolutionType::FAILURE_OUT_OF_BOUNDS)
      .export_values();
}

} // namespace python
} // namespace dart
