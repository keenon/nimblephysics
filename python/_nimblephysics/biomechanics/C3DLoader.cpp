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
#include <dart/biomechanics/C3DLoader.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void C3DLoader(py::module& m)
{
  ::py::class_<dart::biomechanics::C3D>(m, "C3D")
      .def_readwrite("timestamps", &dart::biomechanics::C3D::timestamps)
      .def_readwrite("markers", &dart::biomechanics::C3D::markers)
      .def_readwrite(
          "markerTimesteps", &dart::biomechanics::C3D::markerTimesteps)
      .def_readwrite("forcePlates", &dart::biomechanics::C3D::forcePlates)
      .def_readwrite(
          "shuffledMarkersMatrix",
          &dart::biomechanics::C3D::shuffledMarkersMatrix)
      .def_readwrite(
          "shuffledMarkersMatrixMask",
          &dart::biomechanics::C3D::shuffledMarkersMatrixMask);

  ::py::class_<dart::biomechanics::ForcePlate>(m, "ForcePlate")
      .def_readwrite(
          "worldOrigin", &dart::biomechanics::ForcePlate::worldOrigin)
      .def_readwrite("corners", &dart::biomechanics::ForcePlate::corners)
      .def_readwrite(
          "centersOfPressure",
          &dart::biomechanics::ForcePlate::centersOfPressure)
      .def_readwrite("moments", &dart::biomechanics::ForcePlate::moments)
      .def_readwrite("forces", &dart::biomechanics::ForcePlate::forces);

  ::py::class_<dart::biomechanics::C3DLoader>(m, "C3DLoader")
      .def_static(
          "loadC3D", &dart::biomechanics::C3DLoader::loadC3D, ::py::arg("uri"))
      .def_static(
          "debugToGUI",
          &dart::biomechanics::C3DLoader::debugToGUI,
          ::py::arg("file"),
          ::py::arg("server"));
}

} // namespace python
} // namespace dart
