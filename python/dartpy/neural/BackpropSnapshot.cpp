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

#include <dart/dart.hpp>
#include <dart/neural/BackpropSnapshot.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void BackpropSnapshot(py::module& m)
{
  ::py::class_<
      dart::neural::BackpropSnapshot,
      std::shared_ptr<dart::neural::BackpropSnapshot>>(m, "BackpropSnapshot")
      .def(
          ::py::init<
              dart::simulation::WorldPtr,
              Eigen::VectorXd,
              Eigen::VectorXd,
              Eigen::VectorXd>(),
          ::py::arg("world"),
          ::py::arg("forwardPassPosition"),
          ::py::arg("forwardPassVelocity"),
          ::py::arg("forwardPassTorques"))
      .def(
          "backprop",
          &dart::neural::BackpropSnapshot::backprop,
          ::py::arg("world"),
          ::py::arg("thisTimestepLoss"),
          ::py::arg("nextTimestepLoss"))
      .def(
          "getVelVelJacobian",
          &dart::neural::BackpropSnapshot::getVelVelJacobian,
          ::py::arg("world"))
      .def(
          "getForceVelJacobian",
          &dart::neural::BackpropSnapshot::getForceVelJacobian,
          ::py::arg("world"))
      .def(
          "getPosPosJacobian",
          &dart::neural::BackpropSnapshot::getPosPosJacobian,
          ::py::arg("world"))
      .def(
          "getVelPosJacobian",
          &dart::neural::BackpropSnapshot::getVelPosJacobian,
          ::py::arg("world"))
      .def(
          "getPosVelJacobian",
          &dart::neural::BackpropSnapshot::getPosVelJacobian,
          ::py::arg("world"))
      .def(
          "getForwardPassPosition",
          &dart::neural::BackpropSnapshot::getForwardPassPosition)
      .def(
          "getForwardPassVelocity",
          &dart::neural::BackpropSnapshot::getForwardPassVelocity)
      .def(
          "getForwardPassTorques",
          &dart::neural::BackpropSnapshot::getForwardPassTorques)
      .def("getMassMatrix", &dart::neural::BackpropSnapshot::getMassMatrix)
      .def(
          "getInvMassMatrix", &dart::neural::BackpropSnapshot::getInvMassMatrix)
      .def(
          "finiteDifferenceVelVelJacobian",
          &dart::neural::BackpropSnapshot::finiteDifferenceVelVelJacobian,
          ::py::arg("world"))
      .def(
          "finiteDifferenceForceVelJacobian",
          &dart::neural::BackpropSnapshot::finiteDifferenceForceVelJacobian,
          ::py::arg("world"))
      .def(
          "finiteDifferencePosPosJacobian",
          &dart::neural::BackpropSnapshot::finiteDifferencePosPosJacobian,
          ::py::arg("world"),
          ::py::arg("subdivisions"))
      .def(
          "finiteDifferenceVelPosJacobian",
          &dart::neural::BackpropSnapshot::finiteDifferenceVelPosJacobian,
          ::py::arg("world"),
          ::py::arg("subdivisions"));
}

} // namespace python
} // namespace dart
