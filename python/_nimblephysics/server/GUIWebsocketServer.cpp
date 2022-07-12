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
#include <dart/server/GUIWebsocketServer.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void GUIWebsocketServer(py::module& m)
{
  ::py::class_<
      dart::server::GUIWebsocketServer,
      dart::server::GUIStateMachine,
      std::shared_ptr<dart::server::GUIWebsocketServer>>(
      m, "GUIWebsocketServer")
      .def(::py::init<>())
      .def(
          "serve",
          &dart::server::GUIWebsocketServer::serve,
          ::py::arg("port"),
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "stopServing",
          &dart::server::GUIWebsocketServer::stopServing,
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "blockWhileServing",
          +[](dart::server::GUIWebsocketServer* self) -> void {
            self->blockWhileServing([]() {
              /* Acquire GIL before calling Python code */
              py::gil_scoped_acquire acquire;

              if (PyErr_CheckSignals() != 0)
                throw py::error_already_set();
            });
          },
          ::py::call_guard<py::gil_scoped_release>())
      .def("isServing", &dart::server::GUIWebsocketServer::isServing)
      .def("getScreenSize", &dart::server::GUIWebsocketServer::getScreenSize)
      .def("getKeysDown", &dart::server::GUIWebsocketServer::getKeysDown)
      .def(
          "isKeyDown",
          &dart::server::GUIWebsocketServer::isKeyDown,
          ::py::arg("key"))
      .def("clear", &dart::server::GUIWebsocketServer::clear)
      .def("flush", &dart::server::GUIWebsocketServer::flush)
      .def(
          "registerConnectionListener",
          &dart::server::GUIWebsocketServer::registerConnectionListener,
          ::py::arg("listener"))
      .def(
          "registerShutdownListener",
          &dart::server::GUIWebsocketServer::registerShutdownListener,
          ::py::arg("listener"))
      .def(
          "registerKeydownListener",
          &dart::server::GUIWebsocketServer::registerKeydownListener,
          ::py::arg("listener"))
      .def(
          "registerKeyupListener",
          &dart::server::GUIWebsocketServer::registerKeyupListener,
          ::py::arg("listener"))
      .def(
          "registerScreenResizeListener",
          &dart::server::GUIWebsocketServer::registerScreenResizeListener,
          ::py::arg("listener"))
      .def(
          "registerDragListener",
          &dart::server::GUIWebsocketServer::registerDragListener,
          ::py::arg("key"),
          ::py::arg("listener"),
          ::py::arg("endDrag"))
      .def(
          "registerTooltipChangeListener",
          &dart::server::GUIWebsocketServer::registerTooltipChangeListener,
          ::py::arg("key"),
          ::py::arg("listener"));
}

} // namespace python
} // namespace dart
