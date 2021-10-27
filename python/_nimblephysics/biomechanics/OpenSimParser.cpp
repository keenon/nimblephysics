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

#include <dart/biomechanics/OpenSimParser.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/MeshShape.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void OpenSimParser(py::module& m)
{
  ::py::class_<dart::biomechanics::OpenSimFile>(m, "OpenSimFile")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::map<
                  std::string,
                  std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"))
      .def_readwrite("skeleton", &dart::biomechanics::OpenSimFile::skeleton)
      .def_readwrite(
          "markersMap", &dart::biomechanics::OpenSimFile::markersMap);

  ::py::class_<dart::biomechanics::OpenSimMot>(m, "OpenSimMot")
      .def_readwrite("poses", &dart::biomechanics::OpenSimMot::poses)
      .def_readwrite("timestamps", &dart::biomechanics::OpenSimMot::timestamps);

  ::py::class_<dart::biomechanics::OpenSimTRC>(m, "OpenSimTRC")
      .def_readwrite(
          "markerTimesteps", &dart::biomechanics::OpenSimTRC::markerTimesteps)
      .def_readwrite(
          "markerLines", &dart::biomechanics::OpenSimTRC::markerLines)
      .def_readwrite("timestamps", &dart::biomechanics::OpenSimTRC::timestamps);

  ::py::class_<dart::biomechanics::OpenSimScaleAndMarkerOffsets>(
      m, "OpenSimScaleAndMarkerOffsets")
      .def_readwrite(
          "success", &dart::biomechanics::OpenSimScaleAndMarkerOffsets::success)
      .def_readwrite(
          "bodyScales",
          &dart::biomechanics::OpenSimScaleAndMarkerOffsets::bodyScales)
      .def_readwrite(
          "markerOffsets",
          &dart::biomechanics::OpenSimScaleAndMarkerOffsets::markerOffsets)
      .def_readwrite(
          "markers",
          &dart::biomechanics::OpenSimScaleAndMarkerOffsets::markers);

  auto sm = m.def_submodule("OpenSimParser");
  sm.def(
      "parseOsim",
      +[](const std::string& path) {
        return dart::biomechanics::OpenSimParser::parseOsim(path);
      },
      ::py::arg("path"));

  sm.def(
      "loadTRC",
      +[](const std::string& path) {
        return dart::biomechanics::OpenSimParser::loadTRC(path);
      },
      ::py::arg("path"));

  sm.def(
      "loadMot",
      +[](std::shared_ptr<dynamics::Skeleton> skel, const std::string& path) {
        return dart::biomechanics::OpenSimParser::loadMot(skel, path);
      },
      ::py::arg("skel"),
      ::py::arg("path"));

  sm.def(
      "getScaleAndMarkerOffsets",
      &dart::biomechanics::OpenSimParser::getScaleAndMarkerOffsets,
      ::py::arg("standardSkeleton"),
      ::py::arg("scaledSkeleton"));
}

} // namespace python
} // namespace dart
