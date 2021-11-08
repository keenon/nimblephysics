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

#include <Python.h>
#include <dart/server/GUIRecording.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void GUIRecording(py::module& m)
{
  ::py::class_<
      dart::server::GUIRecording,
      dart::server::GUIStateMachine,
      std::shared_ptr<dart::server::GUIRecording>>(m, "GUIRecording")
      .def(::py::init<>())
      .def("saveFrame", &dart::server::GUIRecording::saveFrame)
      .def("getNumFrames", &dart::server::GUIRecording::getNumFrames)
      .def(
          "getFramesJson",
          &dart::server::GUIRecording::getFramesJson,
          ::py::arg("startFrame") = 0)
      .def(
          "getFrameJson",
          &dart::server::GUIRecording::getFrameJson,
          ::py::arg("frame"))
      .def(
          "writeFramesJson",
          &dart::server::GUIRecording::writeFramesJson,
          ::py::arg("path"),
          ::py::arg("startFrame") = 0)
      .def(
          "writeFrameJson",
          &dart::server::GUIRecording::writeFrameJson,
          ::py::arg("path"),
          ::py::arg("frame"));
}

} // namespace python
} // namespace dart
