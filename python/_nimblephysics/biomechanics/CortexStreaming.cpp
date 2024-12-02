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

#include "dart/biomechanics/CortexStreaming.hpp"

#include <memory>

#include <dart/biomechanics/OpenSimParser.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/MeshShape.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void CortexStreaming(py::module& m)
{
  ::py::class_<dart::biomechanics::CortexStreaming>(m, "CortexStreaming")
      .def(
          ::py::init<std::string, int, int>(),
          ::py::arg("cortexNicAddress"),
          ::py::arg("cortexMulticastPort") = 1001,
          ::py::arg("cortexRequestsPort") = 1510)
      .def(
          "setFrameHandler",
          &dart::biomechanics::CortexStreaming::setFrameHandler,
          ::py::arg("handler"),
          "This is the callback that gets called when a frame of data is "
          "received")
      .def(
          "mockServerSetData",
          &dart::biomechanics::CortexStreaming::mockServerSetData,
          ::py::arg("markerNames"),
          ::py::arg("markers"),
          ::py::arg("copTorqueForces"),
          "This is used for mocking the Cortex API server for local testing. "
          "This sets the current body defs and frame of data to send back to "
          "the client.")
      .def(
          "initialize",
          &dart::biomechanics::CortexStreaming::initialize,
          "This connects to Cortex, and requests the body defs and a frame of "
          "data")
      .def(
          "connect",
          &dart::biomechanics::CortexStreaming::connect,
          "This creates a UDP socket and starts listening for packets from "
          "Cortex")
      .def(
          "startMockServer",
          &dart::biomechanics::CortexStreaming::startMockServer,
          "This starts a UDP server that mimicks the Cortex API, so we can "
          "test locally without having to run Cortex. This is an alternative "
          "to connect(), and cannot run in the same process as connect().")
      .def(
          "disconnect",
          &dart::biomechanics::CortexStreaming::disconnect,
          "This closes the UDP socket and stops listening for packets from "
          "Cortex")
      .def(
          "mockServerSendFrameMulticast",
          &dart::biomechanics::CortexStreaming::mockServerSendFrameMulticast,
          "This sends a UDP packet out on the multicast address, to tell "
          "everyone about the current frame");
}

} // namespace python
} // namespace dart
