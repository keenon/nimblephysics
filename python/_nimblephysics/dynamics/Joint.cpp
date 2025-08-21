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

#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Joint.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <eigen_geometry_pybind.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void Joint(
    py::module& m,
    ::py::class_<dart::dynamics::detail::JointProperties>& jointProps,
    ::py::class_<dart::dynamics::Joint>& joint)
{
  auto attr = m.attr("Joint");

  ::py::enum_<dart::dynamics::detail::ActuatorType>(attr, "ActuatorType")
      .value("FORCE", dart::dynamics::detail::ActuatorType::FORCE)
      .value("PASSIVE", dart::dynamics::detail::ActuatorType::PASSIVE)
      .value("SERVO", dart::dynamics::detail::ActuatorType::SERVO)
      .value("MIMIC", dart::dynamics::detail::ActuatorType::MIMIC)
      .value("ACCELERATION", dart::dynamics::detail::ActuatorType::ACCELERATION)
      .value("VELOCITY", dart::dynamics::detail::ActuatorType::VELOCITY)
      .value("LOCKED", dart::dynamics::detail::ActuatorType::LOCKED);

  (void)m;
  jointProps.def(::py::init<>())
      .def(::py::init<const std::string&>(), ::py::arg("name"))
      .def_readwrite("mName", &dart::dynamics::detail::JointProperties::mName)
      .def_readwrite(
          "mT_ParentBodyToJoint",
          &dart::dynamics::detail::JointProperties::mT_ParentBodyToJoint)
      .def_readwrite(
          "mT_ChildBodyToJoint",
          &dart::dynamics::detail::JointProperties::mT_ChildBodyToJoint)
      .def_readwrite(
          "mIsPositionLimitEnforced",
          &dart::dynamics::detail::JointProperties::mIsPositionLimitEnforced)
      .def_readwrite(
          "mActuatorType",
          &dart::dynamics::detail::JointProperties::mActuatorType)
      .def_readwrite(
          "mMimicJoint", &dart::dynamics::detail::JointProperties::mMimicJoint)
      .def_readwrite(
          "mMimicMultiplier",
          &dart::dynamics::detail::JointProperties::mMimicMultiplier)
      .def_readwrite(
          "mMimicOffset",
          &dart::dynamics::detail::JointProperties::mMimicOffset);

  joint
      .def(
          "removeJointAspect",
          +[](dart::dynamics::Joint* self) -> void {
            return self->removeJointAspect();
          })
      .def(
          "setProperties",
          +[](dart::dynamics::Joint* self,
              const dart::dynamics::Joint::Properties& properties) -> void {
            return self->setProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "copy",
          +[](dart::dynamics::Joint* self,
              const dart::dynamics::Joint& otherJoint) -> void {
            return self->copy(otherJoint);
          },
          ::py::arg("otherJoint"))
      .def(
          "copy",
          +[](dart::dynamics::Joint* self,
              const dart::dynamics::Joint* otherJoint) -> void {
            return self->copy(otherJoint);
          },
          ::py::arg("otherJoint"))
      .def(
          "setName",
          +[](dart::dynamics::Joint* self, const std::string& name) -> void {
            self->setName(name);
          },
          ::py::arg("name"))
      .def(
          "setName",
          +[](dart::dynamics::Joint* self,
              const std::string& name,
              bool renameDofs) -> void { self->setName(name, renameDofs); },
          ::py::arg("name"),
          ::py::arg("renameDofs"))
      .def(
          "getName",
          +[](const dart::dynamics::Joint* self) -> std::string {
            return self->getName();
          })
      .def(
          "getType",
          +[](const dart::dynamics::Joint* self) -> std::string {
            return self->getType();
          })
      .def(
          "setActuatorType",
          +[](dart::dynamics::Joint* self,
              dart::dynamics::Joint::ActuatorType actuatorType) -> void {
            return self->setActuatorType(actuatorType);
          },
          ::py::arg("actuatorType"))
      .def(
          "getActuatorType",
          +[](const dart::dynamics::Joint* self)
              -> dart::dynamics::Joint::ActuatorType {
            return self->getActuatorType();
          })
      .def(
          "isKinematic",
          +[](const dart::dynamics::Joint* self) -> bool {
            return self->isKinematic();
          })
      .def(
          "isDynamic",
          +[](const dart::dynamics::Joint* self) -> bool {
            return self->isDynamic();
          })
      .def(
          "getChildBodyNode",
          +[](dart::dynamics::Joint* self) -> dart::dynamics::BodyNode* {
            return self->getChildBodyNode();
          },
          ::py::return_value_policy::reference)
      .def(
          "getParentBodyNode",
          +[](dart::dynamics::Joint* self) -> dart::dynamics::BodyNode* {
            return self->getParentBodyNode();
          },
          ::py::return_value_policy::reference)
      .def(
          "getSkeleton",
          +[](dart::dynamics::Joint* self) -> dart::dynamics::SkeletonPtr {
            return self->getSkeleton();
          })
      .def(
          "getSkeleton",
          +[](const dart::dynamics::Joint* self)
              -> std::shared_ptr<const dart::dynamics::Skeleton> {
            return self->getSkeleton();
          })
      .def(
          "setTransformFromParentBodyNode",
          +[](dart::dynamics::Joint* self, const Eigen::Isometry3s& T) -> void {
            return self->setTransformFromParentBodyNode(T);
          },
          ::py::arg("T"))
      .def(
          "setTransformFromChildBodyNode",
          +[](dart::dynamics::Joint* self, const Eigen::Isometry3s& T) -> void {
            return self->setTransformFromChildBodyNode(T);
          },
          ::py::arg("T"))
      .def(
          "setPositionLimitEnforced",
          +[](dart::dynamics::Joint* self,
              bool isPositionLimitEnforced) -> void {
            return self->setPositionLimitEnforced(isPositionLimitEnforced);
          },
          ::py::arg("isPositionLimitEnforced"))
      .def(
          "isPositionLimitEnforced",
          +[](const dart::dynamics::Joint* self) -> bool {
            return self->isPositionLimitEnforced();
          })
      .def(
          "getIndexInSkeleton",
          +[](const dart::dynamics::Joint* self, std::size_t index)
              -> std::size_t { return self->getIndexInSkeleton(index); },
          ::py::arg("index"))
      .def(
          "getIndexInTree",
          +[](const dart::dynamics::Joint* self, std::size_t index)
              -> std::size_t { return self->getIndexInTree(index); },
          ::py::arg("index"))
      .def(
          "getJointIndexInSkeleton",
          +[](const dart::dynamics::Joint* self) -> std::size_t {
            return self->getJointIndexInSkeleton();
          })
      .def(
          "getJointIndexInTree",
          +[](const dart::dynamics::Joint* self) -> std::size_t {
            return self->getJointIndexInTree();
          })
      .def(
          "getTreeIndex",
          +[](const dart::dynamics::Joint* self) -> std::size_t {
            return self->getTreeIndex();
          })
      .def(
          "setDofName",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              const std::string& name) -> void {
            self->setDofName(index, name);
          },
          ::py::arg("index"),
          ::py::arg("name"))
      .def(
          "setDofName",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              const std::string& name,
              bool preserveName) -> void {
            self->setDofName(index, name, preserveName);
          },
          ::py::arg("index"),
          ::py::arg("name"),
          ::py::arg("preserveName"))
      .def(
          "preserveDofName",
          +[](dart::dynamics::Joint* self, std::size_t index, bool preserve)
              -> void { return self->preserveDofName(index, preserve); },
          ::py::arg("index"),
          ::py::arg("preserve"))
      .def(
          "isDofNamePreserved",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> bool {
            return self->isDofNamePreserved(index);
          },
          ::py::arg("index"))
      .def(
          "getDofName",
          +[](const dart::dynamics::Joint* self, std::size_t index)
              -> std::string { return self->getDofName(index); },
          ::py::arg("index"))
      .def(
          "getNumDofs",
          +[](const dart::dynamics::Joint* self) -> std::size_t {
            return self->getNumDofs();
          })
      .def(
          "setCommand",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              s_t command) -> void { return self->setCommand(index, command); },
          ::py::arg("index"),
          ::py::arg("command"))
      .def(
          "getCommand",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getCommand(index);
          },
          ::py::arg("index"))
      .def(
          "setCommands",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& commands)
              -> void { return self->setCommands(commands); },
          ::py::arg("commands"))
      .def(
          "getCommands",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getCommands();
          })
      .def(
          "resetCommands",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetCommands();
          })
      .def(
          "setPosition",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t position)
              -> void { return self->setPosition(index, position); },
          ::py::arg("index"),
          ::py::arg("position"))
      .def(
          "getPosition",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getPosition(index);
          },
          ::py::arg("index"))
      .def(
          "setPositions",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& positions)
              -> void { return self->setPositions(positions); },
          ::py::arg("positions"))
      .def(
          "getPositions",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getPositions();
          })
      .def(
          "setPositionLowerLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t position)
              -> void { return self->setPositionLowerLimit(index, position); },
          ::py::arg("index"),
          ::py::arg("position"))
      .def(
          "getPositionLowerLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getPositionLowerLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setPositionLowerLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& lowerLimits)
              -> void { return self->setPositionLowerLimits(lowerLimits); },
          ::py::arg("lowerLimits"))
      .def(
          "getPositionLowerLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getPositionLowerLimits();
          })
      .def(
          "setPositionUpperLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t position)
              -> void { return self->setPositionUpperLimit(index, position); },
          ::py::arg("index"),
          ::py::arg("position"))
      .def(
          "getPositionUpperLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getPositionUpperLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setPositionUpperLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& upperLimits)
              -> void { return self->setPositionUpperLimits(upperLimits); },
          ::py::arg("upperLimits"))
      .def(
          "getPositionUpperLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getPositionUpperLimits();
          })
      .def(
          "isCyclic",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> bool {
            return self->isCyclic(index);
          },
          ::py::arg("index"))
      .def(
          "hasPositionLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> bool {
            return self->hasPositionLimit(index);
          },
          ::py::arg("index"))
      .def(
          "resetPosition",
          +[](dart::dynamics::Joint* self, std::size_t index) -> void {
            return self->resetPosition(index);
          },
          ::py::arg("index"))
      .def(
          "resetPositions",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetPositions();
          })
      .def(
          "setInitialPosition",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t initial)
              -> void { return self->setInitialPosition(index, initial); },
          ::py::arg("index"),
          ::py::arg("initial"))
      .def(
          "getInitialPosition",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getInitialPosition(index);
          },
          ::py::arg("index"))
      .def(
          "setInitialPositions",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& initial)
              -> void { return self->setInitialPositions(initial); },
          ::py::arg("initial"))
      .def(
          "getInitialPositions",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getInitialPositions();
          })
      .def(
          "setVelocity",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t velocity)
              -> void { return self->setVelocity(index, velocity); },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocity",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getVelocity(index);
          },
          ::py::arg("index"))
      .def(
          "setVelocities",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& velocities)
              -> void { return self->setVelocities(velocities); },
          ::py::arg("velocities"))
      .def(
          "getVelocities",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getVelocities();
          })
      .def(
          "setVelocityLowerLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t velocity)
              -> void { return self->setVelocityLowerLimit(index, velocity); },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocityLowerLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getVelocityLowerLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setVelocityLowerLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& lowerLimits)
              -> void { return self->setVelocityLowerLimits(lowerLimits); },
          ::py::arg("lowerLimits"))
      .def(
          "getVelocityLowerLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getVelocityLowerLimits();
          })
      .def(
          "setVelocityUpperLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t velocity)
              -> void { return self->setVelocityUpperLimit(index, velocity); },
          ::py::arg("index"),
          ::py::arg("velocity"))
      .def(
          "getVelocityUpperLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getVelocityUpperLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setVelocityUpperLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& upperLimits)
              -> void { return self->setVelocityUpperLimits(upperLimits); },
          ::py::arg("upperLimits"))
      .def(
          "getVelocityUpperLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getVelocityUpperLimits();
          })
      .def(
          "resetVelocity",
          +[](dart::dynamics::Joint* self, std::size_t index) -> void {
            return self->resetVelocity(index);
          },
          ::py::arg("index"))
      .def(
          "resetVelocities",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetVelocities();
          })
      .def(
          "setInitialVelocity",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t initial)
              -> void { return self->setInitialVelocity(index, initial); },
          ::py::arg("index"),
          ::py::arg("initial"))
      .def(
          "getInitialVelocity",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getInitialVelocity(index);
          },
          ::py::arg("index"))
      .def(
          "setInitialVelocities",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& initial)
              -> void { return self->setInitialVelocities(initial); },
          ::py::arg("initial"))
      .def(
          "getInitialVelocities",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getInitialVelocities();
          })
      .def(
          "setAcceleration",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t acceleration)
              -> void { return self->setAcceleration(index, acceleration); },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAcceleration",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getAcceleration(index);
          },
          ::py::arg("index"))
      .def(
          "setAccelerations",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& accelerations)
              -> void { return self->setAccelerations(accelerations); },
          ::py::arg("accelerations"))
      .def(
          "getAccelerations",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getAccelerations();
          })
      .def(
          "resetAccelerations",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetAccelerations();
          })
      .def(
          "setAccelerationLowerLimit",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              s_t acceleration) -> void {
            return self->setAccelerationLowerLimit(index, acceleration);
          },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAccelerationLowerLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getAccelerationLowerLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setAccelerationLowerLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& lowerLimits)
              -> void { return self->setAccelerationLowerLimits(lowerLimits); },
          ::py::arg("lowerLimits"))
      .def(
          "getAccelerationLowerLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getAccelerationLowerLimits();
          })
      .def(
          "setAccelerationUpperLimit",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              s_t acceleration) -> void {
            return self->setAccelerationUpperLimit(index, acceleration);
          },
          ::py::arg("index"),
          ::py::arg("acceleration"))
      .def(
          "getAccelerationUpperLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getAccelerationUpperLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setAccelerationUpperLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& upperLimits)
              -> void { return self->setAccelerationUpperLimits(upperLimits); },
          ::py::arg("upperLimits"))
      .def(
          "getAccelerationUpperLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getAccelerationUpperLimits();
          })
      .def(
          "setControlForce",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t force)
              -> void { return self->setControlForce(index, force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForce",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getControlForce(index);
          },
          ::py::arg("index"))
      .def(
          "setControlForces",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& forces)
              -> void { return self->setControlForces(forces); },
          ::py::arg("forces"))
      .def(
          "getControlForces",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getControlForces();
          })
      .def(
          "resetControlForces",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetControlForces();
          })
      .def(
          "setControlForceLowerLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t force)
              -> void { return self->setControlForceLowerLimit(index, force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForceLowerLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getControlForceLowerLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setControlForceLowerLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& lowerLimits)
              -> void { return self->setControlForceLowerLimits(lowerLimits); },
          ::py::arg("lowerLimits"))
      .def(
          "getControlForceLowerLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getControlForceLowerLimits();
          })
      .def(
          "setControlForceUpperLimit",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t force)
              -> void { return self->setControlForceUpperLimit(index, force); },
          ::py::arg("index"),
          ::py::arg("force"))
      .def(
          "getControlForceUpperLimit",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getControlForceUpperLimit(index);
          },
          ::py::arg("index"))
      .def(
          "setControlForceUpperLimits",
          +[](dart::dynamics::Joint* self, const Eigen::VectorXs& upperLimits)
              -> void { return self->setControlForceUpperLimits(upperLimits); },
          ::py::arg("upperLimits"))
      .def(
          "getControlForceUpperLimits",
          +[](const dart::dynamics::Joint* self) -> Eigen::VectorXs {
            return self->getControlForceUpperLimits();
          })
      .def(
          "checkSanity",
          +[](const dart::dynamics::Joint* self) -> bool {
            return self->checkSanity();
          })
      .def(
          "checkSanity",
          +[](const dart::dynamics::Joint* self, bool printWarnings) -> bool {
            return self->checkSanity(printWarnings);
          },
          ::py::arg("printWarnings"))
      .def(
          "setVelocityChange",
          +[](dart::dynamics::Joint* self,
              std::size_t index,
              s_t velocityChange) -> void {
            return self->setVelocityChange(index, velocityChange);
          },
          ::py::arg("index"),
          ::py::arg("velocityChange"))
      .def(
          "getVelocityChange",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getVelocityChange(index);
          },
          ::py::arg("index"))
      .def(
          "resetVelocityChanges",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetVelocityChanges();
          })
      .def(
          "setConstraintImpulse",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t impulse)
              -> void { return self->setConstraintImpulse(index, impulse); },
          ::py::arg("index"),
          ::py::arg("impulse"))
      .def(
          "getConstraintImpulse",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getConstraintImpulse(index);
          },
          ::py::arg("index"))
      .def(
          "resetConstraintImpulses",
          +[](dart::dynamics::Joint* self) -> void {
            return self->resetConstraintImpulses();
          })
      .def(
          "integratePositions",
          +[](dart::dynamics::Joint* self, s_t dt) -> void {
            return self->integratePositions(dt);
          },
          ::py::arg("dt"))
      .def(
          "integrateVelocities",
          +[](dart::dynamics::Joint* self, s_t dt) -> void {
            return self->integrateVelocities(dt);
          },
          ::py::arg("dt"))
      .def(
          "getPositionDifferences",
          +[](const dart::dynamics::Joint* self,
              const Eigen::VectorXs& q2,
              const Eigen::VectorXs& q1) -> Eigen::VectorXs {
            return self->getPositionDifferences(q2, q1);
          },
          ::py::arg("q2"),
          ::py::arg("q1"))
      .def(
          "setSpringStiffness",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t k) -> void {
            return self->setSpringStiffness(index, k);
          },
          ::py::arg("index"),
          ::py::arg("k"))
      .def(
          "getSpringStiffness",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getSpringStiffness(index);
          },
          ::py::arg("index"))
      .def(
          "setRestPosition",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t q0) -> void {
            return self->setRestPosition(index, q0);
          },
          ::py::arg("index"),
          ::py::arg("q0"))
      .def(
          "getRestPosition",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getRestPosition(index);
          },
          ::py::arg("index"))
      .def(
          "setDampingCoefficient",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t coeff)
              -> void { return self->setDampingCoefficient(index, coeff); },
          ::py::arg("index"),
          ::py::arg("coeff"))
      .def(
          "getDampingCoefficient",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getDampingCoefficient(index);
          },
          ::py::arg("index"))
      .def(
          "setCoulombFriction",
          +[](dart::dynamics::Joint* self, std::size_t index, s_t friction)
              -> void { return self->setCoulombFriction(index, friction); },
          ::py::arg("index"),
          ::py::arg("friction"))
      .def(
          "getCoulombFriction",
          +[](const dart::dynamics::Joint* self, std::size_t index) -> s_t {
            return self->getCoulombFriction(index);
          },
          ::py::arg("index"))
      .def(
          "computePotentialEnergy",
          +[](const dart::dynamics::Joint* self) -> s_t {
            return self->computePotentialEnergy();
          })
      .def(
          "getTransformFromChildBodyNode",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Isometry3s& {
            return self->getTransformFromChildBodyNode();
          })
      .def(
          "getTransformFromParentBodyNode",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Isometry3s& {
            return self->getTransformFromParentBodyNode();
          })
      .def(
          "getRelativeTransform",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Isometry3s& {
            return self->getRelativeTransform();
          })
      .def(
          "getWorldAxisScrewForPosition",
          &dart::dynamics::Joint::getWorldAxisScrewForPosition,
          ::py::arg("dof"))
      .def(
          "getWorldAxisScrewForVelocity",
          &dart::dynamics::Joint::getWorldAxisScrewForVelocity,
          ::py::arg("dof"))
      .def(
          "getRelativeSpatialVelocity",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Vector6s& {
            return self->getRelativeSpatialVelocity();
          })
      .def(
          "getRelativeSpatialAcceleration",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Vector6s& {
            return self->getRelativeSpatialAcceleration();
          })
      .def(
          "getRelativePrimaryAcceleration",
          +[](const dart::dynamics::Joint* self) -> const Eigen::Vector6s& {
            return self->getRelativePrimaryAcceleration();
          })
      .def(
          "getRelativeJacobian",
          +[](const dart::dynamics::Joint* self) -> const dart::math::Jacobian {
            return self->getRelativeJacobian();
          })
      .def(
          "getRelativeJacobian",
          +[](const dart::dynamics::Joint* self,
              const Eigen::VectorXs& positions) -> dart::math::Jacobian {
            return self->getRelativeJacobian(positions);
          },
          ::py::arg("positions"))
      .def(
          "getRelativeJacobianTimeDeriv",
          +[](const dart::dynamics::Joint* self) -> const dart::math::Jacobian {
            return self->getRelativeJacobianTimeDeriv();
          })
      .def(
          "getBodyConstraintWrench",
          +[](const dart::dynamics::Joint* self) -> Eigen::Vector6s {
            return self->getBodyConstraintWrench();
          })
      .def(
          "notifyPositionUpdated",
          +[](dart::dynamics::Joint* self) -> void {
            return self->notifyPositionUpdated();
          })
      .def(
          "notifyVelocityUpdated",
          +[](dart::dynamics::Joint* self) -> void {
            return self->notifyVelocityUpdated();
          })
      .def(
          "notifyAccelerationUpdated",
          +[](dart::dynamics::Joint* self) -> void {
            return self->notifyAccelerationUpdated();
          })
      .def(
          "getNearestPositionToDesiredRotation",
          +[](dart::dynamics::Joint* self,
              const Eigen::Matrix3s& relativeRotation) -> Eigen::VectorXs {
            return self->getNearestPositionToDesiredRotation(relativeRotation);
          })
      .def_readonly_static("FORCE", &dart::dynamics::Joint::FORCE)
      .def_readonly_static("PASSIVE", &dart::dynamics::Joint::PASSIVE)
      .def_readonly_static("SERVO", &dart::dynamics::Joint::SERVO)
      .def_readonly_static("ACCELERATION", &dart::dynamics::Joint::ACCELERATION)
      .def_readonly_static("VELOCITY", &dart::dynamics::Joint::VELOCITY)
      .def_readonly_static("LOCKED", &dart::dynamics::Joint::LOCKED)
      .def_readonly_static(
          "DefaultActuatorType", &dart::dynamics::Joint::DefaultActuatorType);
}

} // namespace python
} // namespace dart
