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

#include <dart/dynamics/ZeroDofJoint.hpp>
#include <eigen_geometry_pybind.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Joint.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void ZeroDofJoint(py::module& m)
{
  ::py::class_<dart::dynamics::ZeroDofJoint::Properties>(
      m, "ZeroDofJointProperties")
      .def(::py::init<>())
      .def(
          ::py::init<const dart::dynamics::Joint::Properties&>(),
          ::py::arg("properties"));

  ::py::class_<
      dart::dynamics::ZeroDofJoint,
      dart::dynamics::Joint,
      std::shared_ptr<dart::dynamics::ZeroDofJoint> >(m, "ZeroDofJoint")
      .def(
          "getZeroDofJointProperties",
          +[](const dart::dynamics::ZeroDofJoint* self)
              -> dart::dynamics::ZeroDofJoint::Properties {
            return self->getZeroDofJointProperties();
          })
      .def(
          "setDofName",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _arg0_,
              const std::string& _arg1_,
              bool _arg2_) -> const std::string& {
            return self->setDofName(_arg0_, _arg1_, _arg2_);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("arg0_"),
          ::py::arg("arg1_"),
          ::py::arg("arg2_"))
      .def(
          "preserveDofName",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _arg0_,
              bool _arg1_) { self->preserveDofName(_arg0_, _arg1_); },
          ::py::arg("arg0_"),
          ::py::arg("arg1_"))
      .def(
          "isDofNamePreserved",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _arg0_)
              -> bool { return self->isDofNamePreserved(_arg0_); },
          ::py::arg("arg0_"))
      .def(
          "getDofName",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _arg0_)
              -> const std::string& { return self->getDofName(_arg0_); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("arg0_"))
      .def(
          "getNumDofs",
          +[](const dart::dynamics::ZeroDofJoint* self) -> std::size_t {
            return self->getNumDofs();
          })
      .def(
          "getIndexInSkeleton",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> std::size_t { return self->getIndexInSkeleton(_index); },
          ::py::arg("index"))
      .def(
          "getIndexInTree",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> std::size_t { return self->getIndexInTree(_index); },
          ::py::arg("index"))
      .def(
          "setCommand",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _command) { self->setCommand(_index, _command); },
          ::py::arg("index"),
          ::py::arg("command"))
      .def(
          "getCommand",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getCommand(_index); },
          ::py::arg("index"))
      .def(
          "setCommands",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _commands) {
            self->setCommands(_commands);
          },
          ::py::arg("commands"))
      .def(
          "getCommands",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getCommands();
          })
      .def(
          "resetCommands",
          +[](dart::dynamics::ZeroDofJoint* self) { self->resetCommands(); })
      .def(
          "setPosition",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _arg0_,
              s_t _arg1_) { self->setPosition(_arg0_, _arg1_); },
          ::py::arg("arg0_"),
          ::py::arg("arg1_"))
      .def(
          "getPosition",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getPosition(_index); },
          ::py::arg("index"))
      .def(
          "setPositions",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _positions) {
            self->setPositions(_positions);
          },
          ::py::arg("positions"))
      .def(
          "getPositions",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getPositions();
          })
      .def(
          "setPositionLowerLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _position) {
            self->setPositionLowerLimit(_index, _position);
          },
          ::py::arg("index"),
          ::py::arg("position"))
      .def(
          "getPositionLowerLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getPositionLowerLimit(_index); },
          ::py::arg("index"))
      .def(
          "setPositionLowerLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& lowerLimits) {
            self->setPositionLowerLimits(lowerLimits);
          },
          ::py::arg("lowerLimits"))
      .def(
          "getPositionLowerLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getPositionLowerLimits();
          })
      .def(
          "setPositionUpperLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t index,
              s_t position) {
            self->setPositionUpperLimit(index, position);
          },
          ::py::arg("index"),
          ::py::arg("position"))
      .def(
          "getPositionUpperLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t index)
              -> s_t { return self->getPositionUpperLimit(index); },
          ::py::arg("index"))
      .def(
          "setPositionUpperLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& upperLimits) {
            self->setPositionUpperLimits(upperLimits);
          },
          ::py::arg("upperLimits"))
      .def(
          "getPositionUpperLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getPositionUpperLimits();
          })
      .def(
          "hasPositionLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> bool { return self->hasPositionLimit(_index); },
          ::py::arg("index"))
      .def(
          "resetPosition",
          +[](dart::dynamics::ZeroDofJoint* self, std::size_t _index) {
            self->resetPosition(_index);
          },
          ::py::arg("index"))
      .def(
          "resetPositions",
          +[](dart::dynamics::ZeroDofJoint* self) { self->resetPositions(); })
      .def(
          "setInitialPosition",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _initial) { self->setInitialPosition(_index, _initial); },
          ::py::arg("index"),
          ::py::arg("initial"))
      .def(
          "getInitialPosition",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getInitialPosition(_index); },
          ::py::arg("index"))
      .def(
          "setInitialPositions",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _initial) {
            self->setInitialPositions(_initial);
          },
          ::py::arg("initial"))
      .def(
          "getInitialPositions",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getInitialPositions();
          })
      .def(
          "setVelocity",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _velocity) { self->setVelocity(_index, _velocity); },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocity",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getVelocity(_index); },
          ::py::arg("index"))
      .def(
          "setVelocities",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _velocities) {
            self->setVelocities(_velocities);
          },
          ::py::arg("velocities"))
      .def(
          "getVelocities",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getVelocities();
          })
      .def(
          "setVelocityLowerLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _velocity) {
            self->setVelocityLowerLimit(_index, _velocity);
          },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocityLowerLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getVelocityLowerLimit(_index); },
          ::py::arg("index"))
      .def(
          "setVelocityLowerLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& lowerLimits) {
            self->setVelocityLowerLimits(lowerLimits);
          },
          ::py::arg("lowerLimits"))
      .def(
          "getVelocityLowerLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getVelocityLowerLimits();
          })
      .def(
          "setVelocityUpperLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _velocity) {
            self->setVelocityUpperLimit(_index, _velocity);
          },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocityUpperLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getVelocityUpperLimit(_index); },
          ::py::arg("index"))
      .def(
          "setVelocityUpperLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& upperLimits) {
            self->setVelocityUpperLimits(upperLimits);
          },
          ::py::arg("upperLimits"))
      .def(
          "getVelocityUpperLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getVelocityUpperLimits();
          })
      .def(
          "resetVelocity",
          +[](dart::dynamics::ZeroDofJoint* self, std::size_t _index) {
            self->resetVelocity(_index);
          },
          ::py::arg("index"))
      .def(
          "resetVelocities",
          +[](dart::dynamics::ZeroDofJoint* self) { self->resetVelocities(); })
      .def(
          "setInitialVelocity",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _initial) { self->setInitialVelocity(_index, _initial); },
          ::py::arg("index"),
          ::py::arg("initial"))
      .def(
          "getInitialVelocity",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getInitialVelocity(_index); },
          ::py::arg("index"))
      .def(
          "setInitialVelocities",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _initial) {
            self->setInitialVelocities(_initial);
          },
          ::py::arg("initial"))
      .def(
          "getInitialVelocities",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getInitialVelocities();
          })
      .def(
          "setAcceleration",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _acceleration) {
            self->setAcceleration(_index, _acceleration);
          },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAcceleration",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getAcceleration(_index); },
          ::py::arg("index"))
      .def(
          "setAccelerations",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _accelerations) {
            self->setAccelerations(_accelerations);
          },
          ::py::arg("accelerations"))
      .def(
          "getAccelerations",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getAccelerations();
          })
      .def(
          "resetAccelerations",
          +[](dart::dynamics::ZeroDofJoint* self) {
            self->resetAccelerations();
          })
      .def(
          "setAccelerationLowerLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _acceleration) {
            self->setAccelerationLowerLimit(_index, _acceleration);
          },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAccelerationLowerLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getAccelerationLowerLimit(_index); },
          ::py::arg("index"))
      .def(
          "setAccelerationLowerLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& lowerLimits) {
            self->setAccelerationLowerLimits(lowerLimits);
          },
          ::py::arg("lowerLimits"))
      .def(
          "getAccelerationLowerLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getAccelerationLowerLimits();
          })
      .def(
          "setAccelerationUpperLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _acceleration) {
            self->setAccelerationUpperLimit(_index, _acceleration);
          },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAccelerationUpperLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getAccelerationUpperLimit(_index); },
          ::py::arg("index"))
      .def(
          "setAccelerationUpperLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& upperLimits) {
            self->setAccelerationUpperLimits(upperLimits);
          },
          ::py::arg("upperLimits"))
      .def(
          "getAccelerationUpperLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getAccelerationUpperLimits();
          })
      .def(
          "setControlForce",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _force) { self->setControlForce(_index, _force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForce",
          +[](const dart::dynamics::ZeroDofJoint* self,
              std::size_t _index) -> s_t { return self->getControlForce(_index); },
          ::py::arg("index"))
      .def(
          "setControlForces",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _forces) { self->setControlForces(_forces); },
          ::py::arg("forces"))
      .def(
          "getControlForces",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getControlForces();
          })
      .def(
          "resetControlForces",
          +[](dart::dynamics::ZeroDofJoint* self) { self->resetControlForces(); })
      .def(
          "setControlForceLowerLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _force) { self->setControlForceLowerLimit(_index, _force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForceLowerLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getControlForceLowerLimit(_index); },
          ::py::arg("index"))
      .def(
          "setControlForceLowerLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& lowerLimits) {
            self->setControlForceLowerLimits(lowerLimits);
          },
          ::py::arg("lowerLimits"))
      .def(
          "getControlForceLowerLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getControlForceLowerLimits();
          })
      .def(
          "setControlForceUpperLimit",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _force) { self->setControlForceUpperLimit(_index, _force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForceUpperLimit",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getControlForceUpperLimit(_index); },
          ::py::arg("index"))
      .def(
          "setControlForceUpperLimits",
          +[](dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& upperLimits) {
            self->setControlForceUpperLimits(upperLimits);
          },
          ::py::arg("upperLimits"))
      .def(
          "getControlForceUpperLimits",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::VectorXs {
            return self->getControlForceUpperLimits();
          })
      .def(
          "setVelocityChange",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _velocityChange) {
            self->setVelocityChange(_index, _velocityChange);
          },
          ::py::arg("index"),
          ::py::arg("velocityChange"))
      .def(
          "getVelocityChange",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getVelocityChange(_index); },
          ::py::arg("index"))
      .def(
          "resetVelocityChanges",
          +[](dart::dynamics::ZeroDofJoint*
                  self) { self->resetVelocityChanges(); })
      .def(
          "setConstraintImpulse",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _impulse) {
            self->setConstraintImpulse(_index, _impulse);
          },
          ::py::arg("index"),
          ::py::arg("impulse"))
      .def(
          "getConstraintImpulse",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getConstraintImpulse(_index); },
          ::py::arg("index"))
      .def(
          "resetConstraintImpulses",
          +[](dart::dynamics::ZeroDofJoint*
                  self) { self->resetConstraintImpulses(); })
      .def(
          "integratePositions",
          +[](dart::dynamics::ZeroDofJoint* self, s_t _dt) {
            self->integratePositions(_dt);
          },
          ::py::arg("dt"))
      .def(
          "integrateVelocities",
          +[](dart::dynamics::ZeroDofJoint* self, s_t _dt) {
            self->integrateVelocities(_dt);
          },
          ::py::arg("dt"))
      .def(
          "getPositionDifferences",
          +[](const dart::dynamics::ZeroDofJoint* self,
              const Eigen::VectorXs& _q2,
              const Eigen::VectorXs& _q1) -> Eigen::VectorXs {
            return self->getPositionDifferences(_q2, _q1);
          },
          ::py::arg("q2"),
          ::py::arg("q1"))
      .def(
          "setSpringStiffness",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _k) { self->setSpringStiffness(_index, _k); },
          ::py::arg("index"),
          ::py::arg("k"))
      .def(
          "getSpringStiffness",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getSpringStiffness(_index); },
          ::py::arg("index"))
      .def(
          "setRestPosition",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _q0) { self->setRestPosition(_index, _q0); },
          ::py::arg("index"),
          ::py::arg("q0"))
      .def(
          "getRestPosition",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getRestPosition(_index); },
          ::py::arg("index"))
      .def(
          "setDampingCoefficient",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _d) { self->setDampingCoefficient(_index, _d); },
          ::py::arg("index"),
          ::py::arg("d"))
      .def(
          "getDampingCoefficient",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getDampingCoefficient(_index); },
          ::py::arg("index"))
      .def(
          "setCoulombFriction",
          +[](dart::dynamics::ZeroDofJoint* self,
              std::size_t _index,
              s_t _friction) {
            self->setCoulombFriction(_index, _friction);
          },
          ::py::arg("index"),
          ::py::arg("friction"))
      .def(
          "getCoulombFriction",
          +[](const dart::dynamics::ZeroDofJoint* self, std::size_t _index)
              -> s_t { return self->getCoulombFriction(_index); },
          ::py::arg("index"))
      .def(
          "computePotentialEnergy",
          +[](const dart::dynamics::ZeroDofJoint* self) -> s_t {
            return self->computePotentialEnergy();
          })
      .def(
          "getBodyConstraintWrench",
          +[](const dart::dynamics::ZeroDofJoint* self) -> Eigen::Vector6s {
            return self->getBodyConstraintWrench();
          });
}

} // namespace python
} // namespace dart
