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

#include <dart/dynamics/ShapeFrame.hpp>
#include <pybind11/pybind11.h>

#include "eigen_geometry_pybind.h"
#include "eigen_pybind.h"

namespace py = pybind11;

#define DARTPY_DEFINE_SPECIALIZED_ASPECT(name)                                 \
  .def(                                                                        \
      "has" #name,                                                             \
      +[](const dart::dynamics::ShapeFrame* self) -> bool {                    \
        return self->has##name();                                              \
      })                                                                       \
      .def(                                                                    \
          "get" #name,                                                         \
          +[](dart::dynamics::ShapeFrame* self) -> dart::dynamics::name* {     \
            return self->get##name();                                          \
          },                                                                   \
          py::return_value_policy::reference_internal)                         \
      .def(                                                                    \
          "get" #name,                                                         \
          +[](dart::dynamics::ShapeFrame* self,                                \
              bool createIfNull) -> dart::dynamics::name* {                    \
            return self->get##name(createIfNull);                              \
          },                                                                   \
          py::return_value_policy::reference_internal,                         \
          ::py::arg("createIfNull"))                                           \
      .def(                                                                    \
          "set" #name,                                                         \
          +[](dart::dynamics::ShapeFrame* self,                                \
              const dart::dynamics::name* aspect) {                            \
            self->set##name(aspect);                                           \
          },                                                                   \
          ::py::arg("aspect"))                                                 \
      .def(                                                                    \
          "create" #name,                                                      \
          +[](dart::dynamics::ShapeFrame* self) -> dart::dynamics::name* {     \
            return self->create##name();                                       \
          },                                                                   \
          py::return_value_policy::reference_internal)                         \
      .def(                                                                    \
          "remove" #name,                                                      \
          +[](dart::dynamics::ShapeFrame* self) { self->remove##name(); })     \
      .def(                                                                    \
          "release" #name,                                                     \
          +[](dart::dynamics::ShapeFrame* self)                                \
              -> std::unique_ptr<dart::dynamics::name> {                       \
            return self->release##name();                                      \
          })

namespace dart {
namespace python {

void ShapeFrame(py::module& m)
{
  auto visualAspect
      = ::py::class_<dart::dynamics::VisualAspect>(m, "VisualAspect");
  auto collisionAspect
      = ::py::class_<dart::dynamics::CollisionAspect>(m, "CollisionAspect");
  auto dynamicsAspect
      = ::py::class_<dart::dynamics::DynamicsAspect>(m, "DynamicsAspect");

  ::py::class_<dart::dynamics::ShapeFrame::Properties>(
      m, "ShapeFrameProperties");

  ::py::class_<
      dart::dynamics::ShapeFrame,
      // dart::common::EmbedPropertiesOnTopOf<
      //     dart::dynamics::ShapeFrame,
      //     dart::dynamics::detail::ShapeFrameProperties,
      //     dart::common::SpecializedForAspect<
      //         dart::dynamics::VisualAspect,
      //         dart::dynamics::CollisionAspect,
      //         dart::dynamics::DynamicsAspect> >,
      dart::dynamics::Frame,
      std::shared_ptr<dart::dynamics::ShapeFrame>>(m, "ShapeFrame")
      .def(
          "setProperties",
          +[](dart::dynamics::ShapeFrame* self,
              const dart::dynamics::ShapeFrame::UniqueProperties& properties) {
            self->setProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "setShape",
          +[](dart::dynamics::ShapeFrame* self,
              const dart::dynamics::ShapePtr& shape) { self->setShape(shape); },
          ::py::arg("shape"))
      .def(
          "getShape",
          +[](dart::dynamics::ShapeFrame* self) -> dart::dynamics::ShapePtr {
            return self->getShape();
          })
      .def(
          "getShape",
          +[](const dart::dynamics::ShapeFrame* self)
              -> dart::dynamics::ConstShapePtr { return self->getShape(); })
      // clang-format off
      DARTPY_DEFINE_SPECIALIZED_ASPECT(VisualAspect)
      DARTPY_DEFINE_SPECIALIZED_ASPECT(CollisionAspect)
      DARTPY_DEFINE_SPECIALIZED_ASPECT(DynamicsAspect)
      // clang-format on
      .def(
          "isShapeNode", +[](const dart::dynamics::ShapeFrame* self) -> bool {
            return self->isShapeNode();
          });

  ::py::class_<dart::dynamics::detail::VisualAspectProperties>(
      m, "VisualAspectProperties");

  visualAspect.def(::py::init<>())
      .def(
          ::py::init<const dart::common::detail::AspectWithVersionedProperties<
              dart::common::CompositeTrackingAspect<dart::dynamics::ShapeFrame>,
              dart::dynamics::VisualAspect,
              dart::dynamics::detail::VisualAspectProperties,
              dart::dynamics::ShapeFrame,
              &dart::common::detail::NoOp>::PropertiesData&>(),
          ::py::arg("properties"))
      .def(
          "setRGBA",
          +[](dart::dynamics::VisualAspect* self,
              const Eigen::Vector4s& color) { self->setRGBA(color); },
          ::py::arg("color"))
      .def(
          "getRGBA",
          +[](dart::dynamics::VisualAspect* self) -> const Eigen::Vector4s& {
            return self->getRGBA();
          })
      .def(
          "setHidden",
          +[](dart::dynamics::VisualAspect* self, const bool& value) {
            self->setHidden(value);
          },
          ::py::arg("value"))
      .def(
          "getHidden",
          +[](dart::dynamics::VisualAspect* self) -> bool {
            return self->getHidden();
          })
      .def(
          "setCastShadows",
          +[](dart::dynamics::VisualAspect* self, const bool& value) {
            self->setCastShadows(value);
          },
          ::py::arg("value"))
      .def(
          "getCastShadows",
          +[](dart::dynamics::VisualAspect* self) -> bool {
            return self->getCastShadows();
          })
      .def(
          "setReceiveShadows",
          +[](dart::dynamics::VisualAspect* self, const bool& value) {
            self->setReceiveShadows(value);
          },
          ::py::arg("value"))
      .def(
          "getReceiveShadows",
          +[](dart::dynamics::VisualAspect* self) -> bool {
            return self->getReceiveShadows();
          })
      .def(
          "setColor",
          +[](dart::dynamics::VisualAspect* self,
              const Eigen::Vector3s& color) { self->setColor(color); },
          ::py::arg("color"))
      .def(
          "setColor",
          +[](dart::dynamics::VisualAspect* self,
              const Eigen::Vector4s& color) { self->setColor(color); },
          ::py::arg("color"))
      .def(
          "setRGB",
          +[](dart::dynamics::VisualAspect* self, const Eigen::Vector3s& rgb) {
            self->setRGB(rgb);
          },
          ::py::arg("rgb"))
      .def(
          "setAlpha",
          +[](dart::dynamics::VisualAspect* self, const s_t alpha) {
            self->setAlpha(alpha);
          },
          ::py::arg("alpha"))
      .def(
          "getColor",
          +[](const dart::dynamics::VisualAspect* self) -> Eigen::Vector3s {
            return self->getColor();
          })
      .def(
          "getRGB",
          +[](const dart::dynamics::VisualAspect* self) -> Eigen::Vector3s {
            return self->getRGB();
          })
      .def(
          "getAlpha",
          +[](const dart::dynamics::VisualAspect* self) -> s_t {
            return self->getAlpha();
          })
      .def(
          "hide", +[](dart::dynamics::VisualAspect* self) { self->hide(); })
      .def(
          "show", +[](dart::dynamics::VisualAspect* self) { self->show(); })
      .def(
          "isHidden", +[](const dart::dynamics::VisualAspect* self) -> bool {
            return self->isHidden();
          });

  ::py::class_<dart::dynamics::detail::CollisionAspectProperties>(
      m, "CollisionAspectProperties");

  collisionAspect.def(::py::init<>())
      .def(
          ::py::init<const dart::common::detail::AspectWithVersionedProperties<
              dart::common::CompositeTrackingAspect<dart::dynamics::ShapeFrame>,
              dart::dynamics::CollisionAspect,
              dart::dynamics::detail::CollisionAspectProperties,
              dart::dynamics::ShapeFrame,
              &dart::common::detail::NoOp>::PropertiesData&>(),
          ::py::arg("properties"))
      .def(
          "setCollidable",
          +[](dart::dynamics::CollisionAspect* self, const bool& value) {
            self->setCollidable(value);
          },
          ::py::arg("value"))
      .def(
          "getCollidable",
          +[](const dart::dynamics::CollisionAspect* self) -> bool {
            return self->getCollidable();
          })
      .def(
          "isCollidable",
          +[](const dart::dynamics::CollisionAspect* self) -> bool {
            return self->isCollidable();
          });

  ::py::class_<dart::dynamics::detail::DynamicsAspectProperties>(
      m, "DynamicsAspectProperties");

  dynamicsAspect.def(::py::init<>())
      .def(
          ::py::init<const dart::common::detail::AspectWithVersionedProperties<
              dart::common::CompositeTrackingAspect<dart::dynamics::ShapeFrame>,
              dart::dynamics::DynamicsAspect,
              dart::dynamics::detail::DynamicsAspectProperties,
              dart::dynamics::ShapeFrame,
              &dart::common::detail::NoOp>::PropertiesData&>(),
          ::py::arg("properties"))
      .def(
          "setFrictionCoeff",
          +[](dart::dynamics::DynamicsAspect* self, const s_t& value) {
            self->setFrictionCoeff(value);
          },
          ::py::arg("value"))
      .def(
          "getFrictionCoeff",
          +[](const dart::dynamics::DynamicsAspect* self) -> s_t {
            return self->getFrictionCoeff();
          })
      .def(
          "setRestitutionCoeff",
          +[](dart::dynamics::DynamicsAspect* self, const s_t& value) {
            self->setRestitutionCoeff(value);
          },
          ::py::arg("value"))
      .def(
          "getRestitutionCoeff",
          +[](const dart::dynamics::DynamicsAspect* self) -> s_t {
            return self->getRestitutionCoeff();
          });
}

} // namespace python
} // namespace dart
