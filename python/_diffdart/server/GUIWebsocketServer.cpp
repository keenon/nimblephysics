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
      .def("isServing", &dart::server::GUIWebsocketServer::isServing)
      .def("getScreenSize", &dart::server::GUIWebsocketServer::getScreenSize)
      .def("getKeysDown", &dart::server::GUIWebsocketServer::getKeysDown)
      .def(
          "isKeyDown",
          &dart::server::GUIWebsocketServer::isKeyDown,
          ::py::arg("key"))
      .def("clear", &dart::server::GUIWebsocketServer::clear)
      .def(
          "createBox",
          &dart::server::GUIWebsocketServer::createBox,
          ::py::arg("key"),
          ::py::arg("size"),
          ::py::arg("pos"),
          ::py::arg("euler"),
          ::py::arg("color"),
          ::py::arg("castShadows"),
          ::py::arg("receiveShadows"))
      .def(
          "createSphere",
          &dart::server::GUIWebsocketServer::createSphere,
          ::py::arg("key"),
          ::py::arg("radius"),
          ::py::arg("pos"),
          ::py::arg("color"),
          ::py::arg("castShadows"),
          ::py::arg("receiveShadows"))
      .def(
          "createLine",
          &dart::server::GUIWebsocketServer::createLine,
          ::py::arg("key"),
          ::py::arg("points"),
          ::py::arg("color"))
      .def(
          "getObjectPosition",
          &dart::server::GUIWebsocketServer::getObjectPosition,
          ::py::arg("key"))
      .def(
          "getObjectRotation",
          &dart::server::GUIWebsocketServer::getObjectRotation,
          ::py::arg("key"))
      .def(
          "getObjectColor",
          &dart::server::GUIWebsocketServer::getObjectColor,
          ::py::arg("key"))
      .def(
          "setObjectPosition",
          &dart::server::GUIWebsocketServer::setObjectPosition,
          ::py::arg("key"),
          ::py::arg("position"),
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "setObjectRotation",
          &dart::server::GUIWebsocketServer::setObjectRotation,
          ::py::arg("key"),
          ::py::arg("euler"))
      .def(
          "setObjectColor",
          &dart::server::GUIWebsocketServer::setObjectColor,
          ::py::arg("key"),
          ::py::arg("color"))
      .def(
          "setAutoflush",
          &dart::server::GUIWebsocketServer::setAutoflush,
          ::py::arg("autoflush"))
      .def("flush", &dart::server::GUIWebsocketServer::flush)
      .def(
          "deleteObject",
          &dart::server::GUIWebsocketServer::deleteObject,
          ::py::arg("key"))
      .def(
          "createText",
          &dart::server::GUIWebsocketServer::createText,
          ::py::arg("key"),
          ::py::arg("contents"),
          ::py::arg("fromTopLeft"),
          ::py::arg("size"))
      .def(
          "createButton",
          &dart::server::GUIWebsocketServer::createButton,
          ::py::arg("key"),
          ::py::arg("label"),
          ::py::arg("fromTopLeft"),
          ::py::arg("size"),
          ::py::arg("onClick"))
      .def(
          "createSlider",
          &dart::server::GUIWebsocketServer::createSlider,
          ::py::arg("key"),
          ::py::arg("fromTopLeft"),
          ::py::arg("size"),
          ::py::arg("min"),
          ::py::arg("max"),
          ::py::arg("value"),
          ::py::arg("onlyInts"),
          ::py::arg("horizontal"),
          ::py::arg("onChange"))
      .def(
          "createPlot",
          &dart::server::GUIWebsocketServer::createPlot,
          ::py::arg("key"),
          ::py::arg("fromTopLeft"),
          ::py::arg("size"),
          ::py::arg("xs"),
          ::py::arg("minX"),
          ::py::arg("maxX"),
          ::py::arg("ys"),
          ::py::arg("minY"),
          ::py::arg("maxY"),
          ::py::arg("plotType"))
      .def(
          "setUIElementPosition",
          &dart::server::GUIWebsocketServer::setUIElementPosition,
          ::py::arg("key"),
          ::py::arg("position"))
      .def(
          "setUIElementSize",
          &dart::server::GUIWebsocketServer::setUIElementSize,
          ::py::arg("key"),
          ::py::arg("size"))
      .def(
          "deleteUIElement",
          &dart::server::GUIWebsocketServer::deleteUIElement,
          ::py::arg("key"))
      .def(
          "setTextContents",
          &dart::server::GUIWebsocketServer::setTextContents,
          ::py::arg("key"),
          ::py::arg("contents"))
      .def(
          "setButtonLabel",
          &dart::server::GUIWebsocketServer::setButtonLabel,
          ::py::arg("key"),
          ::py::arg("label"))
      .def(
          "setSliderValue",
          &dart::server::GUIWebsocketServer::setSliderValue,
          ::py::arg("key"),
          ::py::arg("value"))
      .def(
          "setSliderMin",
          &dart::server::GUIWebsocketServer::setSliderMin,
          ::py::arg("key"),
          ::py::arg("value"))
      .def(
          "setSliderMax",
          &dart::server::GUIWebsocketServer::setSliderMax,
          ::py::arg("key"),
          ::py::arg("value"))
      .def(
          "setPlotData",
          &dart::server::GUIWebsocketServer::setPlotData,
          ::py::arg("key"),
          ::py::arg("xs"),
          ::py::arg("minX"),
          ::py::arg("maxX"),
          ::py::arg("ys"),
          ::py::arg("minY"),
          ::py::arg("maxY"))
      .def(
          "renderWorld",
          &dart::server::GUIWebsocketServer::renderWorld,
          ::py::arg("world"),
          ::py::arg("prefix") = "world",
          ::py::arg("renderForces") = true,
          ::py::arg("renderForceMagnitudes") = true,
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "renderBasis",
          &dart::server::GUIWebsocketServer::renderBasis,
          ::py::arg("scale") = 10.0,
          ::py::arg("prefix") = "basis",
          ::py::arg("pos") = Eigen::Vector3d::Zero(),
          ::py::arg("euler") = Eigen::Vector3d::Zero(),
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "renderSkeleton",
          &dart::server::GUIWebsocketServer::renderSkeleton,
          ::py::arg("skeleton"),
          ::py::arg("prefix") = "world",
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "renderTrajectoryLines",
          &dart::server::GUIWebsocketServer::renderTrajectoryLines,
          ::py::arg("world"),
          ::py::arg("positions"),
          ::py::arg("prefix") = "trajectory",
          ::py::call_guard<py::gil_scoped_release>())
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
          ::py::arg("listener"));
}

} // namespace python
} // namespace dart
