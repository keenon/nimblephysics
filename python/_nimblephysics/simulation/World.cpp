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

#include <dart/collision/CollisionResult.hpp>
#include <dart/constraint/ConstraintSolver.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/neural/WithRespectToMass.hpp>
#include <dart/simulation/World.hpp>
#include <dart/utils/UniversalLoader.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void World(py::module& m)
{
  ::py::class_<
      dart::simulation::World,
      std::shared_ptr<dart::simulation::World>>(m, "World")
      .def(::py::init<>())
      .def(::py::init<const std::string&>(), ::py::arg("name"))
      .def(::py::init(+[]() -> dart::simulation::WorldPtr {
        return dart::simulation::World::create();
      }))
      .def(::py::init(
          +[](const std::string& name) -> dart::simulation::WorldPtr {
            return dart::simulation::World::create(name);
          }))
      .def(
          "clone",
          +[](const dart::simulation::World* self)
              -> std::shared_ptr<dart::simulation::World> {
            return self->clone();
          })
      .def(
          "setName",
          +[](dart::simulation::World* self, const std::string& _newName)
              -> const std::string& { return self->setName(_newName); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("newName"))
      .def(
          "getName",
          +[](const dart::simulation::World* self) -> const std::string& {
            return self->getName();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "setGravity",
          +[](dart::simulation::World* self, const Eigen::Vector3s _gravity)
              -> void { return self->setGravity(_gravity); },
          ::py::arg("gravity"))
      .def(
          "getGravity",
          +[](const dart::simulation::World* self) -> const Eigen::Vector3s& {
            return self->getGravity();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "setTimeStep",
          +[](dart::simulation::World* self, s_t _timeStep) -> void {
            return self->setTimeStep(_timeStep);
          },
          ::py::arg("timeStep"))
      .def(
          "getTimeStep",
          +[](const dart::simulation::World* self) -> s_t {
            return self->getTimeStep();
          })
      .def(
          "getSkeleton",
          +[](const dart::simulation::World* self,
              std::size_t _index) -> dart::dynamics::SkeletonPtr {
            return self->getSkeleton(_index);
          },
          ::py::arg("index"))
      .def(
          "getSkeleton",
          +[](const dart::simulation::World* self,
              const std::string& _name) -> dart::dynamics::SkeletonPtr {
            return self->getSkeleton(_name);
          },
          ::py::arg("name"))
      .def(
          "getNumSkeletons",
          +[](const dart::simulation::World* self) -> std::size_t {
            return self->getNumSkeletons();
          })
      .def(
          "addSkeleton",
          +[](dart::simulation::World* self,
              const dart::dynamics::SkeletonPtr& _skeleton) -> std::string {
            return self->addSkeleton(_skeleton);
          },
          ::py::arg("skeleton"))
      .def(
          "loadSkeleton",
          +[](dart::simulation::World* self,
              const std::string& path,
              Eigen::Vector3s basePosition,
              Eigen::Vector3s baseEulerAnglesXYZ)
              -> dart::dynamics::SkeletonPtr {
            return dart::utils::UniversalLoader::loadSkeleton(
                self, path, basePosition, baseEulerAnglesXYZ);
          },
          ::py::arg("path"),
          ::py::arg("basePosition") = Eigen::Vector3s::Zero(),
          ::py::arg("baseEulerAnglesXYZ") = Eigen::Vector3s::Zero())
      .def_static(
          "loadFrom",
          +[](const std::string& path)
              -> std::shared_ptr<dart::simulation::World> {
            return dart::utils::UniversalLoader::loadWorld(path);
          })
      .def(
          "removeSkeleton",
          +[](dart::simulation::World* self,
              const dart::dynamics::SkeletonPtr& _skeleton) -> void {
            return self->removeSkeleton(_skeleton);
          },
          ::py::arg("skeleton"))
      .def(
          "removeAllSkeletons",
          +[](dart::simulation::World* self)
              -> std::set<dart::dynamics::SkeletonPtr> {
            return self->removeAllSkeletons();
          })
      .def(
          "hasSkeleton",
          +[](const dart::simulation::World* self,
              const dart::dynamics::ConstSkeletonPtr& skeleton) -> bool {
            return self->hasSkeleton(skeleton);
          },
          ::py::arg("skeleton"))
      .def(
          "getIndex",
          +[](const dart::simulation::World* self, int _index) -> int {
            return self->getIndex(_index);
          },
          ::py::arg("index"))
      .def(
          "getNumBodyNodes",
          +[](dart::simulation::World* self) -> std::size_t {
            return self->getNumBodyNodes();
          })
      .def(
          "getBodyNodeIndex",
          +[](dart::simulation::World* self,
              std::size_t _index) -> dart::dynamics::BodyNode* {
            return self->getBodyNodeIndex(_index);
          })
      .def(
          "getSimpleFrame",
          +[](const dart::simulation::World* self,
              std::size_t _index) -> dart::dynamics::SimpleFramePtr {
            return self->getSimpleFrame(_index);
          },
          ::py::arg("index"))
      .def(
          "getSimpleFrame",
          +[](const dart::simulation::World* self,
              const std::string& _name) -> dart::dynamics::SimpleFramePtr {
            return self->getSimpleFrame(_name);
          },
          ::py::arg("name"))
      .def(
          "getNumSimpleFrames",
          +[](const dart::simulation::World* self) -> std::size_t {
            return self->getNumSimpleFrames();
          })
      .def(
          "addSimpleFrame",
          +[](dart::simulation::World* self,
              const dart::dynamics::SimpleFramePtr& _frame) -> std::string {
            return self->addSimpleFrame(_frame);
          },
          ::py::arg("frame"))
      .def(
          "removeSimpleFrame",
          +[](dart::simulation::World* self,
              const dart::dynamics::SimpleFramePtr& _frame) -> void {
            return self->removeSimpleFrame(_frame);
          },
          ::py::arg("frame"))
      .def(
          "removeAllSimpleFrames",
          +[](dart::simulation::World* self)
              -> std::set<dart::dynamics::SimpleFramePtr> {
            return self->removeAllSimpleFrames();
          })
      .def(
          "checkCollision",
          +[](dart::simulation::World* self) -> bool {
            return self->checkCollision();
          })
      .def(
          "checkCollision",
          +[](dart::simulation::World* self,
              const dart::collision::CollisionOption& option) -> bool {
            return self->checkCollision(option);
          },
          ::py::arg("option"))
      .def(
          "checkCollision",
          +[](dart::simulation::World* self,
              const dart::collision::CollisionOption& option,
              dart::collision::CollisionResult* result) -> bool {
            return self->checkCollision(option, result);
          },
          ::py::arg("option"),
          ::py::arg("result"))
      .def(
          "getLastCollisionResult",
          +[](dart::simulation::World* self)
              -> const collision::CollisionResult& {
            return self->getLastCollisionResult();
          })
      .def(
          "reset",
          +[](dart::simulation::World* self) -> void { return self->reset(); })
      .def(
          "step",
          +[](dart::simulation::World* self) -> void { return self->step(); })
      .def(
          "step",
          +[](dart::simulation::World* self, bool _resetCommand) -> void {
            return self->step(_resetCommand);
          },
          ::py::arg("resetCommand"))
      .def(
          "integratePositions",
          +[](dart::simulation::World* self, Eigen::VectorXs initialVelocity)
              -> void { return self->integratePositions(initialVelocity); },
          ::py::arg("initialVelocity"))
      .def(
          "integrateVelocitiesFromImpulses",
          +[](dart::simulation::World* self, bool _resetCommand) -> void {
            return self->integrateVelocitiesFromImpulses(_resetCommand);
          },
          ::py::arg("resetCommand") = true)
      .def(
          "setTime",
          +[](dart::simulation::World* self, s_t _time) -> void {
            return self->setTime(_time);
          },
          ::py::arg("time"))
      .def(
          "getTime",
          +[](const dart::simulation::World* self) -> s_t {
            return self->getTime();
          })
      .def(
          "getSimFrames",
          +[](const dart::simulation::World* self) -> int {
            return self->getSimFrames();
          })
      .def(
          "getConstraintSolver",
          +[](dart::simulation::World* self) -> constraint::ConstraintSolver* {
            return self->getConstraintSolver();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "runRegisteredConstraintEngine",
          +[](dart::simulation::World* self, bool _resetCommand) -> void {
            return self->runRegisteredConstraintEngine(_resetCommand);
          },
          ::py::arg("resetCommand") = true)
      .def(
          "runLcpConstraintEngine",
          +[](dart::simulation::World* self, bool _resetCommand) -> void {
            return self->runLcpConstraintEngine(_resetCommand);
          },
          ::py::arg("resetCommand") = true)
      .def(
          "runFrictionlessLcpConstraintEngine",
          +[](dart::simulation::World* self, bool _resetCommand) -> void {
            return self->runFrictionlessLcpConstraintEngine(_resetCommand);
          },
          ::py::arg("resetCommand") = true)
      .def(
          "replaceConstraintEngineFn",
          &dart::simulation::World::replaceConstraintEngineFn)
      .def(
          "bake",
          +[](dart::simulation::World* self) -> void { return self->bake(); })
      .def(
          "tuneMass",
          +[](dart::simulation::World* self,
              dynamics::BodyNode* node,
              neural::WrtMassBodyNodeEntryType type,
              Eigen::VectorXs upperBound,
              Eigen::VectorXs lowerBound) -> void {
            self->tuneMass(node, type, upperBound, lowerBound);
          })
      .def(
          "getNumDofs",
          +[](dart::simulation::World* self) -> std::size_t {
            return self->getNumDofs();
          })
      .def(
          "getPositions",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getPositions();
          })
      .def(
          "getVelocities",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getVelocities();
          })
      .def(
          "getControlForces",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getControlForces();
          })
      .def(
          "getMasses",
          +[](dart::simulation::World* self)
              -> Eigen::VectorXs { return self->getMasses(); })
      .def(
          "getControlForceUpperLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getControlForceUpperLimits();
          })
      .def(
          "getControlForceLowerLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getControlForceLowerLimits();
          })
      .def(
          "getPositionLowerLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getPositionLowerLimits();
          })
      .def(
          "getPositionUpperLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getPositionUpperLimits();
          })
      .def(
          "getVelocityLowerLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getVelocityLowerLimits();
          })
      .def(
          "getVelocityUpperLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getVelocityUpperLimits();
          })
      .def(
          "getMassLowerLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getMassLowerLimits();
          })
      .def(
          "getMassUpperLimits",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getMassUpperLimits();
          })
      .def(
          "setPositions",
          +[](dart::simulation::World* self, Eigen::VectorXs positions)
              -> void { self->setPositions(positions); })
      .def(
          "setVelocities",
          +[](dart::simulation::World* self, Eigen::VectorXs velocities)
              -> void { self->setVelocities(velocities); })
      .def(
          "setControlForces",
          +[](dart::simulation::World* self, Eigen::VectorXs forces) -> void {
            self->setControlForces(forces);
          })
      .def(
          "setMasses",
          +[](dart::simulation::World* self, Eigen::VectorXs forces) -> void {
            self->setMasses(forces);
          })
      .def(
          "setControlForcesUpperLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setControlForceUpperLimits(limits);
          })
      .def(
          "setControlForcesLowerLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setControlForceLowerLimits(limits);
          })
      .def(
          "setPositionUpperLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setPositionUpperLimits(limits);
          })
      .def(
          "setPositionLowerLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setPositionLowerLimits(limits);
          })
      .def(
          "setVelocityUpperLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setVelocityUpperLimits(limits);
          })
      .def(
          "setVelocityLowerLimits",
          +[](dart::simulation::World* self, Eigen::VectorXs limits) -> void {
            self->setVelocityLowerLimits(limits);
          })
      .def(
          "getCoriolisAndGravityForces",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getCoriolisAndGravityForces();
          })
      .def(
          "getCoriolisAndGravityAndExternalForces",
          +[](dart::simulation::World* self) -> Eigen::VectorXs {
            return self->getCoriolisAndGravityAndExternalForces();
          })
      .def(
          "getMassMatrix",
          +[](dart::simulation::World* self) -> Eigen::MatrixXs {
            return self->getMassMatrix();
          })
      .def(
          "getInvMassMatrix",
          +[](dart::simulation::World* self) -> Eigen::MatrixXs {
            return self->getInvMassMatrix();
          })
      .def(
          "getParallelVelocityAndPositionUpdates",
          &dart::simulation::World::getParallelVelocityAndPositionUpdates)
      .def(
          "setParallelVelocityAndPositionUpdates",
          &dart::simulation::World::setParallelVelocityAndPositionUpdates,
          ::py::arg("enabled"))
      .def(
          "getPenetrationCorrectionEnabled",
          &dart::simulation::World::getPenetrationCorrectionEnabled)
      .def(
          "setPenetrationCorrectionEnabled",
          &dart::simulation::World::setPenetrationCorrectionEnabled,
          ::py::arg("enabled"))
      .def(
          "getContactClippingDepth",
          &dart::simulation::World::getContactClippingDepth)
      .def(
          "getFallbackConstraintForceMixingConstant",
          &dart::simulation::World::getFallbackConstraintForceMixingConstant)
      .def(
          "setFallbackConstraintForceMixingConstant",
          &dart::simulation::World::setFallbackConstraintForceMixingConstant,
          ::py::arg("constant") = 1e-3)
      .def("getWrtMass", &dart::simulation::World::getWrtMass)
      .def("toJson", &dart::simulation::World::toJson)
      .def("positionsToJson", &dart::simulation::World::positionsToJson)
      .def("colorsToJson", &dart::simulation::World::colorsToJson)
      .def(
          "setUseFDOverride",
          &dart::simulation::World::setUseFDOverride,
          ::py::arg("useFDOverride"))
      .def("getUseFDOverride", &dart::simulation::World::getUseFDOverride)
      .def(
          "getCachedLCPSolution",
          &dart::simulation::World::getCachedLCPSolution)
      .def(
          "setCachedLCPSolution",
          &dart::simulation::World::setCachedLCPSolution,
          ::py::arg("cachedLCPSolution"))
      .def(
          "setSlowDebugResultsAgainstFD",
          &dart::simulation::World::setSlowDebugResultsAgainstFD,
          ::py::arg("setSlowDebugResultsAgainstFD"))
      .def(
          "setSlowDebugResultsAgainstFD",
          &dart::simulation::World::setSlowDebugResultsAgainstFD)
      .def("getStateSize", &dart::simulation::World::getStateSize)
      .def("setState", &dart::simulation::World::setState, ::py::arg("state"))
      .def("getState", &dart::simulation::World::getState)
      .def("getActionSize", &dart::simulation::World::getActionSize)
      .def(
          "setAction", &dart::simulation::World::setAction, ::py::arg("action"))
      .def("getAction", &dart::simulation::World::getAction)
      .def(
          "setActionSpace",
          &dart::simulation::World::setActionSpace,
          ::py::arg("actionSpaceMapping"))
      .def("getActionSpace", &dart::simulation::World::getActionSpace)
      .def(
          "removeDofFromActionSpace",
          &dart::simulation::World::removeDofFromActionSpace,
          ::py::arg("dofIndex"))
      .def(
          "addDofToActionSpace",
          &dart::simulation::World::addDofToActionSpace,
          ::py::arg("dofIndex"))
      .def("getStateJacobian", &dart::simulation::World::getStateJacobian)
      .def("getActionJacobian", &dart::simulation::World::getActionJacobian)
      .def_readonly("onNameChanged", &dart::simulation::World::onNameChanged);
}

} // namespace python
} // namespace dart
