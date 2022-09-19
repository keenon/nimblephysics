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

#include <dart/collision/CollisionDetector.hpp>
#include <dart/collision/CollisionGroup.hpp>
#include <dart/constraint/ConstrainedGroup.hpp>
#include <dart/constraint/ConstraintSolver.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void ConstraintSolver(py::module& m)
{
  ::py::class_<
      dart::constraint::ConstraintSolver,
      std::shared_ptr<dart::constraint::ConstraintSolver>>(
      m, "ConstraintSolver")
      .def(
          "addSkeleton",
          +[](dart::constraint::ConstraintSolver* self,
              const dart::dynamics::SkeletonPtr& skeleton) {
            self->addSkeleton(skeleton);
          },
          ::py::arg("skeleton"))
      .def(
          "addSkeletons",
          +[](dart::constraint::ConstraintSolver* self,
              const std::vector<dart::dynamics::SkeletonPtr>& skeletons) {
            self->addSkeletons(skeletons);
          },
          ::py::arg("skeletons"))
      .def(
          "removeSkeleton",
          +[](dart::constraint::ConstraintSolver* self,
              const dart::dynamics::SkeletonPtr& skeleton) {
            self->removeSkeleton(skeleton);
          },
          ::py::arg("skeleton"))
      .def(
          "removeSkeletons",
          +[](dart::constraint::ConstraintSolver* self,
              const std::vector<dart::dynamics::SkeletonPtr>& skeletons) {
            self->removeSkeletons(skeletons);
          },
          ::py::arg("skeletons"))
      .def(
          "removeAllSkeletons",
          +[](dart::constraint::ConstraintSolver* self) {
            self->removeAllSkeletons();
          })
      .def(
          "addConstraint",
          +[](dart::constraint::ConstraintSolver* self,
              const dart::constraint::ConstraintBasePtr& constraint) {
            self->addConstraint(constraint);
          },
          ::py::arg("constraint"))
      .def(
          "removeConstraint",
          +[](dart::constraint::ConstraintSolver* self,
              const dart::constraint::ConstraintBasePtr& constraint) {
            self->removeConstraint(constraint);
          },
          ::py::arg("constraint"))
      .def(
          "removeAllConstraints",
          +[](dart::constraint::ConstraintSolver* self) {
            self->removeAllConstraints();
          })
      .def(
          "clearLastCollisionResult",
          +[](dart::constraint::ConstraintSolver* self) {
            self->clearLastCollisionResult();
          })
      .def(
          "setTimeStep",
          +[](dart::constraint::ConstraintSolver* self, s_t _timeStep) {
            self->setTimeStep(_timeStep);
          },
          ::py::arg("timeStep"))
      .def(
          "getTimeStep",
          +[](const dart::constraint::ConstraintSolver* self) -> s_t {
            return self->getTimeStep();
          })
      .def(
          "setCollisionDetector",
          +[](dart::constraint::ConstraintSolver* self,
              const std::shared_ptr<dart::collision::CollisionDetector>&
                  collisionDetector) {
            self->setCollisionDetector(collisionDetector);
          },
          ::py::arg("collisionDetector"))
      .def(
          "getCollisionDetector",
          +[](dart::constraint::ConstraintSolver* self)
              -> dart::collision::CollisionDetectorPtr {
            return self->getCollisionDetector();
          })
      .def(
          "getCollisionDetector",
          +[](const dart::constraint::ConstraintSolver* self)
              -> dart::collision::ConstCollisionDetectorPtr {
            return self->getCollisionDetector();
          })
      .def(
          "getCollisionGroup",
          +[](dart::constraint::ConstraintSolver* self)
              -> dart::collision::CollisionGroupPtr {
            return self->getCollisionGroup();
          })
      .def(
          "getCollisionGroup",
          +[](const dart::constraint::ConstraintSolver* self)
              -> dart::collision::ConstCollisionGroupPtr {
            return self->getCollisionGroup();
          })
      .def(
          "getGradientEnabled",
          +[](dart::constraint::ConstraintSolver* self) -> bool {
            return self->getGradientEnabled();
          })
      .def(
          "setGradientEnabled",
          +[](dart::constraint::ConstraintSolver* self, bool enabled) -> void {
            return self->setGradientEnabled(enabled);
          })
      .def(
          "setPenetrationCorrectionEnabled",
          +[](dart::constraint::ConstraintSolver* self, bool enable) -> void {
            return self->setPenetrationCorrectionEnabled(enable);
          })
      .def(
          "setContactClippingDepth",
          +[](dart::constraint::ConstraintSolver* self, s_t depth) -> void {
            return self->setContactClippingDepth(depth);
          })
      .def(
          "updateConstraints",
          +[](dart::constraint::ConstraintSolver* self) {
            self->updateConstraints();
          })
      .def(
          "getConstraints",
          +[](dart::constraint::ConstraintSolver* self)
              -> std::vector<constraint::ConstraintBasePtr> {
            return self->getConstraints();
          })
      .def(
          "getConstrainedGroups",
          +[](dart::constraint::ConstraintSolver* self)
              -> std::vector<constraint::ConstrainedGroup> {
            return self->getConstrainedGroups();
          })
      .def(
          "buildConstrainedGroups",
          +[](dart::constraint::ConstraintSolver* self) {
            self->buildConstrainedGroups();
          })
      .def(
          "solveConstrainedGroups",
          +[](dart::constraint::ConstraintSolver* self) {
            self->solveConstrainedGroups();
          })
      .def(
          "applyConstraintImpulses",
          +[](dart::constraint::ConstraintSolver* self,
              std::vector<dart::constraint::ConstraintBasePtr> constraints,
              Eigen::MatrixXs impulses) {
            self->applyConstraintImpulses(constraints, impulses);
          })
      .def(
          "solve",
          +[](dart::constraint::ConstraintSolver* self) { self->solve(); })
      .def(
          "enforceContactAndJointAndCustomConstraintsWithLcp",
          +[](dart::constraint::ConstraintSolver* self) {
            self->enforceContactAndJointAndCustomConstraintsWithLcp();
          })
      .def(
          "replaceEnforceContactAndJointAndCustomConstraintsFn",
          &dart::constraint::ConstraintSolver::
              replaceEnforceContactAndJointAndCustomConstraintsFn);
}

} // namespace python
} // namespace dart
