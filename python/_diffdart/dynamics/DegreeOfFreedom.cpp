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

#include <dart/dynamics/DegreeOfFreedom.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/pybind11.h>

#include "eigen_geometry_pybind.h"
#include "eigen_pybind.h"

namespace py = pybind11;

namespace dart {
namespace python {

void DegreeOfFreedom(py::module& m)
{
  ::py::class_<
      dart::dynamics::DegreeOfFreedom,
      dart::common::Subject,
      std::shared_ptr<dart::dynamics::DegreeOfFreedom>>(m, "DegreeOfFreedom")
      .def(
          "setName",
          +[](dart::dynamics::DegreeOfFreedom* self, const std::string& _name)
              -> const std::string& { return self->setName(_name); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("name"))
      .def(
          "setName",
          +[](dart::dynamics::DegreeOfFreedom* self,
              const std::string& _name,
              bool _preserveName) -> const std::string& {
            return self->setName(_name, _preserveName);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("name"),
          ::py::arg("preserveName"))
      .def(
          "getName",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> const std::string& { return self->getName(); },
          ::py::return_value_policy::reference_internal)
      .def(
          "preserveName",
          +[](dart::dynamics::DegreeOfFreedom* self, bool _preserve) {
            self->preserveName(_preserve);
          },
          ::py::arg("preserve"))
      .def(
          "isNamePreserved",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> bool {
            return self->isNamePreserved();
          })
      .def(
          "getIndexInSkeleton",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> std::size_t {
            return self->getIndexInSkeleton();
          })
      .def(
          "getIndexInTree",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> std::size_t {
            return self->getIndexInTree();
          })
      .def(
          "getIndexInJoint",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> std::size_t {
            return self->getIndexInJoint();
          })
      .def(
          "getTreeIndex",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> std::size_t {
            return self->getTreeIndex();
          })
      .def(
          "setCommand",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _command) {
            self->setCommand(_command);
          },
          ::py::arg("command"))
      .def(
          "getCommand",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getCommand();
          })
      .def(
          "resetCommand",
          +[](dart::dynamics::DegreeOfFreedom* self) { self->resetCommand(); })
      .def(
          "setPosition",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _position) {
            self->setPosition(_position);
          },
          ::py::arg("position"))
      .def(
          "getPosition",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getPosition();
          })
      .def(
          "setPositionLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              s_t _lowerLimit,
              s_t _upperLimit) {
            self->setPositionLimits(_lowerLimit, _upperLimit);
          },
          ::py::arg("lowerLimit"),
          ::py::arg("upperLimit"))
      .def(
          "setPositionLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              const std::pair<s_t, s_t>& _limits) {
            self->setPositionLimits(_limits);
          },
          ::py::arg("limits"))
      .def(
          "getPositionLimits",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> std::pair<s_t, s_t> {
            return self->getPositionLimits();
          })
      .def(
          "setPositionLowerLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setPositionLowerLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getPositionLowerLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getPositionLowerLimit();
          })
      .def(
          "setPositionUpperLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setPositionUpperLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getPositionUpperLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getPositionUpperLimit();
          })
      .def(
          "isCyclic",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> bool {
            return self->isCyclic();
          })
      .def(
          "hasPositionLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> bool {
            return self->hasPositionLimit();
          })
      .def(
          "resetPosition",
          +[](dart::dynamics::DegreeOfFreedom* self) { self->resetPosition(); })
      .def(
          "setInitialPosition",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _initial) {
            self->setInitialPosition(_initial);
          },
          ::py::arg("initial"))
      .def(
          "getInitialPosition",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getInitialPosition();
          })
      .def(
          "setVelocity",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _velocity) {
            self->setVelocity(_velocity);
          },
          ::py::arg("velocity"))
      .def(
          "getVelocity",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getVelocity();
          })
      .def(
          "setVelocityLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              s_t _lowerLimit,
              s_t _upperLimit) {
            self->setVelocityLimits(_lowerLimit, _upperLimit);
          },
          ::py::arg("lowerLimit"),
          ::py::arg("upperLimit"))
      .def(
          "setVelocityLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              const std::pair<s_t, s_t>& _limits) {
            self->setVelocityLimits(_limits);
          },
          ::py::arg("limits"))
      .def(
          "getVelocityLimits",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> std::pair<s_t, s_t> {
            return self->getVelocityLimits();
          })
      .def(
          "setVelocityLowerLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setVelocityLowerLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getVelocityLowerLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getVelocityLowerLimit();
          })
      .def(
          "setVelocityUpperLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setVelocityUpperLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getVelocityUpperLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getVelocityUpperLimit();
          })
      .def(
          "setPositionUpperLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setPositionUpperLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getPositionUpperLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getPositionUpperLimit();
          })
      .def(
          "setPositionLowerLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setPositionLowerLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getPositionLowerLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getPositionLowerLimit();
          })
      .def(
          "resetVelocity",
          +[](dart::dynamics::DegreeOfFreedom* self) { self->resetVelocity(); })
      .def(
          "setInitialVelocity",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _initial) {
            self->setInitialVelocity(_initial);
          },
          ::py::arg("initial"))
      .def(
          "getInitialVelocity",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getInitialVelocity();
          })
      .def(
          "setAcceleration",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _acceleration) {
            self->setAcceleration(_acceleration);
          },
          ::py::arg("acceleration"))
      .def(
          "getAcceleration",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getAcceleration();
          })
      .def(
          "resetAcceleration",
          +[](dart::dynamics::DegreeOfFreedom* self) {
            self->resetAcceleration();
          })
      .def(
          "setAccelerationLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              s_t _lowerLimit,
              s_t _upperLimit) {
            self->setAccelerationLimits(_lowerLimit, _upperLimit);
          },
          ::py::arg("lowerLimit"),
          ::py::arg("upperLimit"))
      .def(
          "setAccelerationLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              const std::pair<s_t, s_t>& _limits) {
            self->setAccelerationLimits(_limits);
          },
          ::py::arg("limits"))
      .def(
          "getAccelerationLimits",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> std::pair<s_t, s_t> {
            return self->getAccelerationLimits();
          })
      .def(
          "setAccelerationLowerLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setAccelerationLowerLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getAccelerationLowerLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getAccelerationLowerLimit();
          })
      .def(
          "setAccelerationUpperLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setAccelerationUpperLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getAccelerationUpperLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getAccelerationUpperLimit();
          })
      .def(
          "setControlForce",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _force) {
            self->setControlForce(_force);
          },
          ::py::arg("force"))
      .def(
          "getControlForce",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getControlForce();
          })
      .def(
          "resetControlForce",
          +[](dart::dynamics::DegreeOfFreedom* self) { self->resetControlForce(); })
      .def(
          "setControlForceLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              s_t _lowerLimit,
              s_t _upperLimit) {
            self->setControlForceLimits(_lowerLimit, _upperLimit);
          },
          ::py::arg("lowerLimit"),
          ::py::arg("upperLimit"))
      .def(
          "setControlForceLimits",
          +[](dart::dynamics::DegreeOfFreedom* self,
              const std::pair<s_t, s_t>& _limits) {
            self->setControlForceLimits(_limits);
          },
          ::py::arg("limits"))
      .def(
          "getControlForceLimits",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> std::pair<s_t, s_t> { return self->getControlForceLimits(); })
      .def(
          "setControlForceLowerLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setControlForceLowerLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getControlForceLowerLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getControlForceLowerLimit();
          })
      .def(
          "setControlForceUpperLimit",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _limit) {
            self->setControlForceUpperLimit(_limit);
          },
          ::py::arg("limit"))
      .def(
          "getControlForceUpperLimit",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getControlForceUpperLimit();
          })
      .def(
          "setVelocityChange",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _velocityChange) {
            self->setVelocityChange(_velocityChange);
          },
          ::py::arg("velocityChange"))
      .def(
          "getVelocityChange",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getVelocityChange();
          })
      .def(
          "resetVelocityChange",
          +[](dart::dynamics::DegreeOfFreedom* self) {
            self->resetVelocityChange();
          })
      .def(
          "setConstraintImpulse",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _impulse) {
            self->setConstraintImpulse(_impulse);
          },
          ::py::arg("impulse"))
      .def(
          "getConstraintImpulse",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getConstraintImpulse();
          })
      .def(
          "resetConstraintImpulse",
          +[](dart::dynamics::DegreeOfFreedom* self) {
            self->resetConstraintImpulse();
          })
      .def(
          "setSpringStiffness",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _k) {
            self->setSpringStiffness(_k);
          },
          ::py::arg("k"))
      .def(
          "getSpringStiffness",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getSpringStiffness();
          })
      .def(
          "setRestPosition",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _q0) {
            self->setRestPosition(_q0);
          },
          ::py::arg("q0"))
      .def(
          "getRestPosition",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getRestPosition();
          })
      .def(
          "setDampingCoefficient",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _coeff) {
            self->setDampingCoefficient(_coeff);
          },
          ::py::arg("coeff"))
      .def(
          "getDampingCoefficient",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getDampingCoefficient();
          })
      .def(
          "setCoulombFriction",
          +[](dart::dynamics::DegreeOfFreedom* self, s_t _friction) {
            self->setCoulombFriction(_friction);
          },
          ::py::arg("friction"))
      .def(
          "getCoulombFriction",
          +[](const dart::dynamics::DegreeOfFreedom* self) -> s_t {
            return self->getCoulombFriction();
          })
      .def(
          "getSkeleton",
          +[](dart::dynamics::DegreeOfFreedom* self)
              -> dart::dynamics::SkeletonPtr { return self->getSkeleton(); })
      .def(
          "getSkeleton",
          +[](const dart::dynamics::DegreeOfFreedom* self)
              -> dart::dynamics::ConstSkeletonPtr {
            return self->getSkeleton();
          });
}

} // namespace python
} // namespace dart
