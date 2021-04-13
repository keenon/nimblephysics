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

#include <dart/neural/IKMapping.hpp>
#include <dart/neural/IdentityMapping.hpp>
#include <dart/neural/Mapping.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void Mapping(py::module& m)
{
  ::py::class_<dart::neural::Mapping, std::shared_ptr<dart::neural::Mapping>>(
      m, "Mapping")
      .def(
          "getPosDim",
          &dart::neural::Mapping::getPosDim,
          "Gets the dimension of the Position space in this mapping. This will "
          "be the length of the getPositions() vector, and the length of the "
          "vector expected by setPositions().")
      .def(
          "getVelDim",
          &dart::neural::Mapping::getVelDim,
          "Gets the dimension of the Velocity space in this mapping. This will "
          "be the length of the getVelocities() vector, and the length of the "
          "vector expected by setVelocities().")
      .def(
          "getControlForceDim",
          &dart::neural::Mapping::getControlForceDim,
          "Gets the dimension of the Force space in this mapping. This will "
          "be the length of the getControlForces() vector, and the length of the "
          "vector expected by setControlForces().")
      .def(
          "setPositions",
          &dart::neural::Mapping::setPositions,
          ::py::arg("world"),
          ::py::arg("positions"))
      .def(
          "setVelocities",
          &dart::neural::Mapping::setVelocities,
          ::py::arg("world"),
          ::py::arg("velocities"))
      .def(
          "setControlForces",
          &dart::neural::Mapping::setControlForces,
          ::py::arg("world"),
          ::py::arg("forces"))
      .def(
          "getPositions",
          &dart::neural::Mapping::getPositions,
          ::py::arg("world"))
      .def(
          "getVelocities",
          &dart::neural::Mapping::getVelocities,
          ::py::arg("world"))
      .def("getControlForces", &dart::neural::Mapping::getControlForces, ::py::arg("world"))
      .def(
          "getRealPosToMappedPosJac",
          &dart::neural::Mapping::getRealPosToMappedPosJac,
          ::py::arg("world"),
          "This returns a Jacobian that transforms the rate of change of the "
          "position in the 'real' space given by the world to the rate of "
          "change of the position in mapped space.")
      .def(
          "getRealPosToMappedVelJac",
          &dart::neural::Mapping::getRealPosToMappedVelJac,
          ::py::arg("world"),
          "This returns a Jacobian that transforms the rate of change of the "
          "position in the 'real' space given by the world to the rate of "
          "change of the velocity in mapped space.")
      .def(
          "getRealVelToMappedVelJac",
          &dart::neural::Mapping::getRealVelToMappedVelJac,
          ::py::arg("world"),
          "This returns a Jacobian that transforms the rate of change of the "
          "velocity in the 'real' space given by the world to the rate of "
          "change of the velocity in mapped space.")
      .def(
          "getRealVelToMappedPosJac",
          &dart::neural::Mapping::getRealVelToMappedPosJac,
          ::py::arg("world"),
          "This returns a Jacobian that transforms the rate of change of the "
          "velocity in the 'real' space given by the world to the rate of "
          "change of the position in mapped space.")
      .def(
          "getRealForceToMappedForceJac",
          &dart::neural::Mapping::getRealForceToMappedForceJac,
          ::py::arg("world"),
          "This returns a Jacobian that transforms the rate of change of the "
          "force in the 'real' space given by the world to the rate of "
          "change of the force in mapped space.")
      .def(
          "getPositionLowerLimits",
          &dart::neural::Mapping::getPositionLowerLimits,
          ::py::arg("world"))
      .def(
          "getPositionUpperLimits",
          &dart::neural::Mapping::getPositionUpperLimits,
          ::py::arg("world"))
      .def(
          "getVelocityLowerLimits",
          &dart::neural::Mapping::getVelocityLowerLimits,
          ::py::arg("world"))
      .def(
          "getVelocityUpperLimits",
          &dart::neural::Mapping::getVelocityUpperLimits,
          ::py::arg("world"))
      .def(
          "getControlForceLowerLimits",
          &dart::neural::Mapping::getControlForceLowerLimits,
          ::py::arg("world"))
      .def(
          "getControlForceUpperLimits",
          &dart::neural::Mapping::getControlForceUpperLimits,
          ::py::arg("world"));
}

} // namespace python
} // namespace dart
