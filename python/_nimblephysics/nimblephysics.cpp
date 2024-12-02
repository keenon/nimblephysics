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

#include <dart/config.hpp>
#include <pybind11/pybind11.h>

#include "dart/neural/WithRespectTo.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void eigen_geometry(py::module& m);

void dart_common(py::module& m);
py::module dart_math(py::module& m);
void dart_euler_math(py::module& m);
void dart_dynamics(py::module& m);
void dart_collision(py::module& m);
void dart_constraint(py::module& m);
void dart_simulation(py::module& m);
void dart_utils(py::module& m);
void dart_simulation_and_neural(
    py::module& m,
    py::module& neural,
    ::py::class_<dart::neural::WithRespectTo>& withRespectTo);
void dart_trajectory(py::module& m);
void dart_performance(py::module& m);
void dart_realtime(py::module& m);
void dart_server(py::module& m);
void dart_biomechanics(py::module& m);
void dart_exo(py::module& m);

PYBIND11_MODULE(_nimblephysics, m)
{
  m.doc() = "nimblephysics: Python API of Nimble";

  auto neural = m.def_submodule("neural");
  auto withRespectTo
      = ::py::class_<dart::neural::WithRespectTo>(neural, "WithRespectTo");

  eigen_geometry(m);

  dart_common(m);
  py::module math_module = dart_math(m);
  dart_performance(m);
  dart_dynamics(m);
  dart_euler_math(math_module);
  dart_collision(m);
  dart_constraint(m);
  dart_simulation_and_neural(m, neural, withRespectTo);
  dart_utils(m);
  dart_trajectory(m);
  dart_realtime(m);
  dart_server(m);
  dart_biomechanics(m);
  dart_exo(m);
}

} // namespace python
} // namespace dart
