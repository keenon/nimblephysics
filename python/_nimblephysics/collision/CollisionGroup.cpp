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

// #include <dart/collision/CollisionDetector.hpp>
#include <dart/collision/CollisionGroup.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/collision/DistanceFilter.hpp"
#include "dart/collision/DistanceOption.hpp"
#include "dart/collision/DistanceResult.hpp"
#include "dart/collision/RaycastResult.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void CollisionGroup(py::module& m)
{
  ::py::class_<dart::collision::DistanceOption>(m, "DistanceOption")
      .def_readwrite(
          "enableNearestPoints",
          &dart::collision::DistanceOption::enableNearestPoints)
      .def_readwrite(
          "distanceLowerBound",
          &dart::collision::DistanceOption::distanceLowerBound);

  ::py::class_<dart::collision::DistanceResult>(m, "DistanceResult")
      .def_readwrite(
          "minDistance", &dart::collision::DistanceResult::minDistance)
      .def_readwrite(
          "unclampedMinDistance",
          &dart::collision::DistanceResult::unclampedMinDistance)
      .def_readwrite(
          "nearestPoint1", &dart::collision::DistanceResult::nearestPoint1)
      .def_readwrite(
          "nearestPoint2", &dart::collision::DistanceResult::nearestPoint2)
      .def("clear", &dart::collision::DistanceResult::clear)
      .def("found", &dart::collision::DistanceResult::found)
      .def(
          "isMinDistanceClamped",
          &dart::collision::DistanceResult::isMinDistanceClamped);

  ::py::class_<dart::collision::RaycastOption>(m, "RaycastOption")
      .def_readwrite(
          "mEnableAllHits", &dart::collision::RaycastOption::mEnableAllHits)
      .def_readwrite(
          "mSortByClosest", &dart::collision::RaycastOption::mSortByClosest);

  ::py::class_<dart::collision::RayHit>(m, "RayHit")
      .def(::py::init<>())
      .def_readwrite(
          "mNormal",
          &dart::collision::RayHit::mNormal,
          "The normal at the hit point in the world coordinates")
      .def_readwrite(
          "mPoint",
          &dart::collision::RayHit::mPoint,
          "The hit point in the world coordinates")
      .def_readwrite(
          "mFraction",
          &dart::collision::RayHit::mFraction,
          "The fraction from `from` point to `to` point");

  ::py::class_<dart::collision::RaycastResult>(m, "RaycastResult")
      .def(::py::init<>())
      .def("hasHit", &dart::collision::RaycastResult::hasHit)
      .def("clear", &dart::collision::RaycastResult::clear)
      .def_readwrite("mRayHits", &dart::collision::RaycastResult::mRayHits);

  ::py::class_<
      dart::collision::CollisionGroup,
      std::shared_ptr<dart::collision::CollisionGroup>>(m, "CollisionGroup")
      /*
      .def(
          "getCollisionDetector",
          +[](dart::collision::CollisionGroup* self)
              -> dart::collision::CollisionDetectorPtr {
            return self->getCollisionDetector();
          })
      .def(
          "getCollisionDetector",
          +[](const dart::collision::CollisionGroup* self)
              -> dart::collision::ConstCollisionDetectorPtr {
            return self->getCollisionDetector();
          })
      */
      .def(
          "addShapeFrame",
          +[](dart::collision::CollisionGroup* self,
              const dart::dynamics::ShapeFrame* shapeFrame) {
            self->addShapeFrame(shapeFrame);
          },
          ::py::arg("shapeFrame"))
      .def(
          "addShapeFrames",
          +[](dart::collision::CollisionGroup* self,
              const std::vector<std::shared_ptr<dart::dynamics::ShapeFrame>>&
                  shapeFrames) {
            std::vector<const dart::dynamics::ShapeFrame*> converted;
            for (auto& frame : shapeFrames)
            {
              converted.push_back(frame.get());
            }
            self->addShapeFrames(converted);
          },
          ::py::arg("shapeFrames"))
      .def(
          "addShapeFramesOf",
          +[](dart::collision::CollisionGroup* self) {
            self->addShapeFramesOf();
          })
      .def(
          "addShapeFramesOf",
          +[](dart::collision::CollisionGroup* self,
              dart::dynamics::BodyNode* bodyNode) {
            self->addShapeFramesOf(bodyNode);
          })
      .def(
          "subscribeTo",
          +[](dart::collision::CollisionGroup* self) { self->subscribeTo(); })
      .def(
          "removeShapeFrame",
          +[](dart::collision::CollisionGroup* self,
              const dart::dynamics::ShapeFrame* shapeFrame) {
            self->removeShapeFrame(shapeFrame);
          },
          ::py::arg("shapeFrame"))
      .def(
          "removeShapeFrames",
          +[](dart::collision::CollisionGroup* self,
              const std::vector<const dart::dynamics::ShapeFrame*>&
                  shapeFrames) { self->removeShapeFrames(shapeFrames); },
          ::py::arg("shapeFrames"))
      .def(
          "removeShapeFramesOf",
          +[](dart::collision::CollisionGroup* self) {
            self->removeShapeFramesOf();
          })
      .def(
          "removeShapeFramesOf",
          +[](dart::collision::CollisionGroup* self,
              dart::dynamics::BodyNode* bodyNode) {
            self->removeShapeFramesOf(bodyNode);
          })
      .def(
          "removeAllShapeFrames",
          +[](dart::collision::CollisionGroup* self) {
            self->removeAllShapeFrames();
          })
      .def(
          "hasShapeFrame",
          +[](const dart::collision::CollisionGroup* self,
              const dart::dynamics::ShapeFrame* shapeFrame) -> bool {
            return self->hasShapeFrame(shapeFrame);
          },
          ::py::arg("shapeFrame"))
      .def(
          "getNumShapeFrames",
          +[](const dart::collision::CollisionGroup* self) -> std::size_t {
            return self->getNumShapeFrames();
          })
      .def(
          "collide",
          +[](dart::collision::CollisionGroup* self) -> bool {
            return self->collide();
          })
      .def(
          "collide",
          +[](dart::collision::CollisionGroup* self,
              const dart::collision::CollisionOption& option) -> bool {
            return self->collide(option);
          },
          ::py::arg("option"))
      .def(
          "collide",
          +[](dart::collision::CollisionGroup* self,
              const dart::collision::CollisionOption& option,
              dart::collision::CollisionResult* result) -> bool {
            return self->collide(option, result);
          },
          ::py::arg("option"),
          ::py::arg("result"))
      .def(
          "distance",
          +[](dart::collision::CollisionGroup* self) -> s_t {
            return self->distance();
          })
      .def(
          "distance",
          +[](dart::collision::CollisionGroup* self,
              const dart::collision::DistanceOption& option) -> s_t {
            return self->distance(option);
          },
          ::py::arg("option"))
      .def(
          "distance",
          +[](dart::collision::CollisionGroup* self,
              const dart::collision::DistanceOption& option,
              dart::collision::DistanceResult* result) -> s_t {
            return self->distance(option, result);
          },
          ::py::arg("option"),
          ::py::arg("result"))
      .def(
          "raycast",
          +[](dart::collision::CollisionGroup* self,
              const Eigen::Vector3s& from,
              const Eigen::Vector3s& to) -> bool {
            return self->raycast(from, to);
          },
          ::py::arg("from_point"),
          ::py::arg("to_point"))
      .def(
          "raycast",
          +[](dart::collision::CollisionGroup* self,
              const Eigen::Vector3s& from,
              const Eigen::Vector3s& to,
              const dart::collision::RaycastOption& option) -> bool {
            return self->raycast(from, to, option);
          },
          ::py::arg("from_point"),
          ::py::arg("to_point"),
          ::py::arg("option"))
      .def(
          "raycast",
          +[](dart::collision::CollisionGroup* self,
              const Eigen::Vector3s& from,
              const Eigen::Vector3s& to,
              const dart::collision::RaycastOption& option,
              dart::collision::RaycastResult* result) -> bool {
            return self->raycast(from, to, option, result);
          },
          ::py::arg("from_point"),
          ::py::arg("to_point"),
          ::py::arg("option"),
          ::py::arg("result"))
      .def(
          "setAutomaticUpdate",
          +[](dart::collision::CollisionGroup* self) {
            self->setAutomaticUpdate();
          })
      .def(
          "setAutomaticUpdate",
          +[](dart::collision::CollisionGroup* self, bool automatic) {
            self->setAutomaticUpdate(automatic);
          },
          ::py::arg("automatic"))
      .def(
          "getAutomaticUpdate",
          +[](const dart::collision::CollisionGroup* self) -> bool {
            return self->getAutomaticUpdate();
          })
      .def(
          "update",
          +[](dart::collision::CollisionGroup* self) { self->update(); })
      .def(
          "removeDeletedShapeFrames",
          +[](dart::collision::CollisionGroup* self) {
            self->removeDeletedShapeFrames();
          });
}

} // namespace python
} // namespace dart
