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
#include <dart/dynamics/Skeleton.hpp>
#include <dart/neural/WithRespectTo.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/math/MathTypes.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void WithRespectTo(py::module& m)
{
  ::py::class_<dart::neural::WithRespectTo>(m, "WithRespectTo")
      .def("name", &dart::neural::WithRespectTo::name)
      .def(
          "get",
          +[](dart::neural::WithRespectTo* self, simulation::World* world)
              -> Eigen::VectorXs { return self->get(world); },
          ::py::arg("world"))
      .def(
          "get",
          +[](dart::neural::WithRespectTo* self, dynamics::Skeleton* skel)
              -> Eigen::VectorXs { return self->get(skel); },
          ::py::arg("skel"))
      .def(
          "set",
          +[](dart::neural::WithRespectTo* self,
              simulation::World* world,
              Eigen::VectorXs value) { self->set(world, value); },
          ::py::arg("world"),
          ::py::arg("value"))
      .def(
          "set",
          +[](dart::neural::WithRespectTo* self,
              dynamics::Skeleton* skel,
              Eigen::VectorXs value) { self->set(skel, value); },
          ::py::arg("skel"),
          ::py::arg("value"))
      .def(
          "dim",
          +[](dart::neural::WithRespectTo* self,
              simulation::World* world) -> int { return self->dim(world); },
          ::py::arg("world"))
      .def(
          "dim",
          +[](dart::neural::WithRespectTo* self,
              dynamics::Skeleton* skel) -> int { return self->dim(skel); },
          ::py::arg("skel"))
      .def(
          "upperBound",
          &dart::neural::WithRespectTo::upperBound,
          ::py::arg("world"))
      .def(
          "lowerBound",
          &dart::neural::WithRespectTo::lowerBound,
          ::py::arg("world"));

  /*
    static WithRespectToPosition* POSITION;
    static WithRespectToVelocity* VELOCITY;
    static WithRespectToForce* FORCE;
    static WithRespectToAcceleration* ACCELERATION;
    static WithRespectToGroupScales* GROUP_SCALES;
    static WithRespectToGroupMasses* GROUP_MASSES;
    static WithRespectToLinearizedMasses* LINEARIZED_MASSES;
    static WithRespectToGroupCOMs* GROUP_COMS;
    static WithRespectToGroupInertias* GROUP_INERTIAS;
    */

  py::class_<dart::neural::WithRespectToPosition, dart::neural::WithRespectTo>(
      m, "WithRespectToPosition");
  py::class_<dart::neural::WithRespectToVelocity, dart::neural::WithRespectTo>(
      m, "WithRespectToVelocity");
  py::class_<dart::neural::WithRespectToForce, dart::neural::WithRespectTo>(
      m, "WithRespectToForce");
  py::class_<
      dart::neural::WithRespectToAcceleration,
      dart::neural::WithRespectTo>(m, "WithRespectToAcceleration");
  py::class_<
      dart::neural::WithRespectToGroupScales,
      dart::neural::WithRespectTo>(m, "WithRespectToGroupScales");
  py::class_<
      dart::neural::WithRespectToGroupMasses,
      dart::neural::WithRespectTo>(m, "WithRespectToGroupMasses");
  py::class_<
      dart::neural::WithRespectToLinearizedMasses,
      dart::neural::WithRespectTo>(m, "WithRespectToLinearizedMasses");
  py::class_<dart::neural::WithRespectToGroupCOMs, dart::neural::WithRespectTo>(
      m, "WithRespectToGroupCOMs");
  py::class_<
      dart::neural::WithRespectToGroupInertias,
      dart::neural::WithRespectTo>(m, "WithRespectToGroupInertias");

  m.attr("WRT_POSITION") = dart::neural::WithRespectTo::POSITION;
  m.attr("WRT_VELOCITY") = dart::neural::WithRespectTo::VELOCITY;
  m.attr("WRT_FORCE") = dart::neural::WithRespectTo::FORCE;
  m.attr("WRT_ACCELERATION") = dart::neural::WithRespectTo::ACCELERATION;
  m.attr("WRT_GROUP_SCALES") = dart::neural::WithRespectTo::GROUP_SCALES;
  m.attr("WRT_GROUP_MASSES") = dart::neural::WithRespectTo::GROUP_MASSES;
  m.attr("WRT_LINEARIZED_MASSES")
      = dart::neural::WithRespectTo::LINEARIZED_MASSES;
  m.attr("WRT_GROUP_INERTIAS") = dart::neural::WithRespectTo::GROUP_INERTIAS;
}

} // namespace python
} // namespace dart
