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

#include <dart/dynamics/ScapulathoracicJoint.hpp>
#include <eigen_geometry_pybind.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Joint.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void ScapulathoracicJoint(py::module& m)
{
  /*
void setAxisOrder(EulerJoint::AxisOrder _order, bool _renameDofs = true);

/// Return the axis order
EulerJoint::AxisOrder getAxisOrder() const;

/// This takes a vector of 1's and -1's to indicate which entries to flip, if
/// any
void setFlipAxisMap(Eigen::Vector4s map);

Eigen::Vector4s getFlipAxisMap() const;

void setEllipsoidRadii(Eigen::Vector3s radii);

Eigen::Vector3s getEllipsoidRadii();

void setWingingAxisOffset(Eigen::Vector2s offset);

Eigen::Vector2s getWingingAxisOffset();

void setWingingAxisDirection(s_t radians);

s_t getWingingAxisDirection();
*/
  ::py::class_<
      dart::dynamics::ScapulathoracicJoint,
      dart::dynamics::GenericJoint<dart::math::R4Space>,
      std::shared_ptr<dart::dynamics::ScapulathoracicJoint>>(
      m, "ScapulothoracicJoint")
      .def(
          "getType",
          &dart::dynamics::ScapulathoracicJoint::getType,
          ::py::return_value_policy::reference_internal)
      .def(
          "isCyclic",
          &dart::dynamics::ScapulathoracicJoint::isCyclic,
          ::py::arg("index"))
      .def(
          "setFlipAxisMap",
          &dart::dynamics::ScapulathoracicJoint::setFlipAxisMap,
          ::py::arg("flipMap"))
      .def(
          "getFlipAxisMap",
          &dart::dynamics::ScapulathoracicJoint::getFlipAxisMap)
      .def(
          "getEllipsoidRadii",
          &dart::dynamics::ScapulathoracicJoint::getEllipsoidRadii)
      .def(
          "setEllipsoidRadii",
          &dart::dynamics::ScapulathoracicJoint::setEllipsoidRadii,
          ::py::arg("radii"))
      .def(
          "getWingingAxisDirection",
          &dart::dynamics::ScapulathoracicJoint::getWingingAxisDirection)
      .def(
          "setWingingAxisDirection",
          &dart::dynamics::ScapulathoracicJoint::setWingingAxisDirection,
          ::py::arg("direction"))
      .def(
          "getWingingAxisOffset",
          &dart::dynamics::ScapulathoracicJoint::getWingingAxisOffset)
      .def(
          "setWingingAxisOffset",
          &dart::dynamics::ScapulathoracicJoint::setWingingAxisOffset,
          ::py::arg("offset"))
      .def(
          "setAxisOrder",
          &dart::dynamics::ScapulathoracicJoint::setAxisOrder,
          ::py::arg("order"),
          ::py::arg("renameDofs"))
      .def("getAxisOrder", &dart::dynamics::ScapulathoracicJoint::getAxisOrder)
      .def(
          "getRelativeJacobianStatic",
          &dart::dynamics::ScapulathoracicJoint::getRelativeJacobianStatic,
          ::py::arg("positions"))
      .def_static(
          "getStaticType",
          +[]() -> const std::string& {
            return dart::dynamics::ScapulathoracicJoint::getStaticType();
          },
          ::py::return_value_policy::reference_internal);
}

} // namespace python
} // namespace dart
