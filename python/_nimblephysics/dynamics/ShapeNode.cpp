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

#include <dart/dynamics/ShapeNode.hpp>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

#include "eigen_geometry_pybind.h"
#include "eigen_pybind.h"

namespace py = pybind11;

namespace dart {
namespace python {

void ShapeNode(py::module& m)
{
  ::py::class_<dart::dynamics::ShapeNode::Properties>(m, "ShapeNodeProperties");

  ::py::class_<dart::dynamics::ShapeNode, dart::dynamics::ShapeFrame>(
      m, "ShapeNode")
      .def(
          "setProperties",
          +[](dart::dynamics::ShapeNode* self,
              const dart::dynamics::ShapeNode::Properties& properties) {
            self->setProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "getShapeNodeProperties",
          +[](const dart::dynamics::ShapeNode* self)
              -> const dart::dynamics::ShapeNode::Properties {
            return self->getShapeNodeProperties();
          })
      .def(
          "copy",
          +[](dart::dynamics::ShapeNode* self,
              const dart::dynamics::ShapeNode* other) { self->copy(other); },
          ::py::arg("other"))
      .def(
          "setRelativeTransform",
          +[](dart::dynamics::ShapeNode* self,
              const Eigen::Isometry3s& transform) {
            self->setRelativeTransform(transform);
          },
          ::py::arg("transform"))
      .def(
          "setRelativeRotation",
          +[](dart::dynamics::ShapeNode* self,
              const Eigen::Matrix3s& rotation) {
            self->setRelativeRotation(rotation);
          },
          ::py::arg("rotation"))
      .def(
          "getRelativeRotation",
          +[](const dart::dynamics::ShapeNode* self) -> Eigen::Matrix3s {
            return self->getRelativeRotation();
          })
      .def(
          "setRelativeTranslation",
          +[](dart::dynamics::ShapeNode* self,
              const Eigen::Vector3s& translation) {
            self->setRelativeTranslation(translation);
          },
          ::py::arg("translation"))
      .def(
          "setOffset",
          +[](dart::dynamics::ShapeNode* self, const Eigen::Vector3s& offset) {
            self->setOffset(offset);
          },
          ::py::arg("offset"))
      .def(
          "getRelativeTranslation",
          +[](const dart::dynamics::ShapeNode* self) -> Eigen::Vector3s {
            return self->getRelativeTranslation();
          })
      .def(
          "getOffset",
          +[](const dart::dynamics::ShapeNode* self) -> Eigen::Vector3s {
            return self->getOffset();
          })
      .def(
          "getShape",
          +[](const dart::dynamics::ShapeNode* self)
              -> std::shared_ptr<const dart::dynamics::Shape> {
            return self->getShape();
          },
          py::return_value_policy::reference_internal)
      .def(
          "getWorldTransform",
          +[](const dart::dynamics::ShapeNode* self) -> Eigen::Isometry3s {
            return self->getWorldTransform();
          })
      .def("getName", &dart::dynamics::ShapeNode::getName)
      .def(
          "createVisualAspect",
          &dart::dynamics::ShapeNode::createVisualAspect<>,
          py::return_value_policy::reference_internal)
      .def(
          "createCollisionAspect",
          &dart::dynamics::ShapeNode::createCollisionAspect<>,
          py::return_value_policy::reference_internal)
      .def(
          "createDynamicsAspect",
          &dart::dynamics::ShapeNode::createDynamicsAspect<>,
          py::return_value_policy::reference_internal);
}

} // namespace python
} // namespace dart
