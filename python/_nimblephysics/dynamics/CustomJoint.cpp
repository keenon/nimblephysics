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

#include <dart/dynamics/CustomJoint.hpp>
#include <eigen_geometry_pybind.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Joint.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void CustomJoint(py::module& m)
{
  ::py::class_<
      dart::dynamics::CustomJoint<1>,
      dart::dynamics::GenericJoint<dart::math::R1Space>,
      std::shared_ptr<dart::dynamics::CustomJoint<1>>>(m, "CustomJoint1")
      .def(
          "getType",
          &dart::dynamics::CustomJoint<1>::getType,
          ::py::return_value_policy::reference_internal)
      .def(
          "isCyclic",
          &dart::dynamics::CustomJoint<1>::isCyclic,
          ::py::arg("index"))
      .def(
          "setFlipAxisMap",
          &dart::dynamics::CustomJoint<1>::setFlipAxisMap,
          ::py::arg("flipMap"))
      .def("getFlipAxisMap", &dart::dynamics::CustomJoint<1>::getFlipAxisMap)
      .def(
          "setAxisOrder",
          &dart::dynamics::CustomJoint<1>::setAxisOrder,
          ::py::arg("order"),
          ::py::arg("renameDofs"))
      .def("getAxisOrder", &dart::dynamics::CustomJoint<1>::getAxisOrder)
      .def(
          "getCustomFunctionPositions",
          &dart::dynamics::CustomJoint<1>::getCustomFunctionPositions,
          ::py::arg("x"))
      .def(
          "getCustomFunctionGradientAt",
          &dart::dynamics::CustomJoint<1>::getCustomFunctionGradientAt,
          ::py::arg("x"))
      .def(
          "getCustomFunctionSecondGradientAt",
          &dart::dynamics::CustomJoint<1>::getCustomFunctionSecondGradientAt,
          ::py::arg("x"))
      .def(
          "getRelativeJacobianStatic",
          &dart::dynamics::CustomJoint<1>::getRelativeJacobianStatic,
          ::py::arg("positions"))
      .def_static(
          "getStaticType",
          +[]() -> const std::string& {
            return dart::dynamics::CustomJoint<1>::getStaticType();
          },
          ::py::return_value_policy::reference_internal);

  ::py::class_<
      dart::dynamics::CustomJoint<2>,
      dart::dynamics::GenericJoint<dart::math::R2Space>,
      std::shared_ptr<dart::dynamics::CustomJoint<2>>>(m, "CustomJoint2")
      .def(
          "getType",
          &dart::dynamics::CustomJoint<2>::getType,
          ::py::return_value_policy::reference_internal)
      .def(
          "isCyclic",
          &dart::dynamics::CustomJoint<2>::isCyclic,
          ::py::arg("index"))
      .def(
          "setFlipAxisMap",
          &dart::dynamics::CustomJoint<2>::setFlipAxisMap,
          ::py::arg("flipMap"))
      .def("getFlipAxisMap", &dart::dynamics::CustomJoint<2>::getFlipAxisMap)
      .def(
          "setAxisOrder",
          &dart::dynamics::CustomJoint<2>::setAxisOrder,
          ::py::arg("order"),
          ::py::arg("renameDofs"))
      .def("getAxisOrder", &dart::dynamics::CustomJoint<2>::getAxisOrder)
      .def(
          "getCustomFunctionPositions",
          &dart::dynamics::CustomJoint<2>::getCustomFunctionPositions,
          ::py::arg("x"))
      .def(
          "getCustomFunctionGradientAt",
          &dart::dynamics::CustomJoint<2>::getCustomFunctionGradientAt,
          ::py::arg("x"))
      .def(
          "getCustomFunctionSecondGradientAt",
          &dart::dynamics::CustomJoint<2>::getCustomFunctionSecondGradientAt,
          ::py::arg("x"))
      .def(
          "getRelativeJacobianStatic",
          &dart::dynamics::CustomJoint<2>::getRelativeJacobianStatic,
          ::py::arg("positions"))
      .def_static(
          "getStaticType",
          +[]() -> const std::string& {
            return dart::dynamics::CustomJoint<2>::getStaticType();
          },
          ::py::return_value_policy::reference_internal);
}

} // namespace python
} // namespace dart
