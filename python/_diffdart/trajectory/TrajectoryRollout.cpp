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

#include <dart/simulation/World.hpp>
#include <dart/trajectory/TrajectoryConstants.hpp>
#include <dart/trajectory/TrajectoryRollout.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {
/*
  virtual const std::string& getRepresentationMapping() const = 0;
  virtual const std::vector<std::string>& getMappings() const = 0;

  virtual Eigen::Ref<Eigen::MatrixXs> getPoses(const std::string& mapping) = 0;
  virtual Eigen::Ref<Eigen::MatrixXs> getVels(const std::string& mapping) = 0;
  virtual Eigen::Ref<Eigen::MatrixXs> getForces(const std::string& mapping) = 0;
  */

void TrajectoryRollout(py::module& m)
{
  ::py::class_<dart::trajectory::TrajectoryRollout>(m, "TrajectoryRollout")
      .def("getMappings", &dart::trajectory::TrajectoryRollout::getMappings)
      .def(
          "getPoses",
          &dart::trajectory::TrajectoryRollout::getPoses,
          ::py::arg("mapping") = "identity")
      .def(
          "getVels",
          &dart::trajectory::TrajectoryRollout::getVels,
          ::py::arg("mapping") = "identity")
      .def(
          "getForces",
          &dart::trajectory::TrajectoryRollout::getForces,
          ::py::arg("mapping") = "identity")
      .def("getMasses", &dart::trajectory::TrajectoryRollout::getMasses)
      .def(
          "toJson",
          &dart::trajectory::TrajectoryRollout::toJson,
          ::py::arg("world"))
      .def(
          "copy",
          &dart::trajectory::TrajectoryRollout::copy,
          ::py::return_value_policy::automatic);
}

} // namespace python
} // namespace dart
