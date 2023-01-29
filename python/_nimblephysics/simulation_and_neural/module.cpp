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

#include <Eigen/Dense>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

// Neural
void WithRespectTo(py::module& sm);
void WithRespectToMass(py::module& sm);
void NeuralUtils(py::module& sm);
void NeuralGlobalMethods(py::module& sm);
void Mapping(py::module& sm);
void IKMapping(py::module& sm);
void IdentityMapping(py::module& sm);
void BackpropSnapshot(py::module& sm);
void MappedBackpropSnapshot(py::module& sm);

// Simulation
void World(
    py::module& sm,
    ::py::class_<
        dart::simulation::World,
        std::shared_ptr<dart::simulation::World>>& world);

void dart_simulation_and_neural(py::module& m)
{
  auto simulation = m.def_submodule("simulation");
  auto neural = m.def_submodule("neural");

  neural.doc()
      = "This provides gradients to DART, with an eye on embedding DART as a "
        "non-linearity in neural networks.";

  auto world = ::py::
      class_<dart::simulation::World, std::shared_ptr<dart::simulation::World>>(
          simulation, "World");

  NeuralUtils(neural);
  WithRespectTo(neural);
  WithRespectToMass(neural);
  Mapping(neural);
  IKMapping(neural);
  IdentityMapping(neural);
  BackpropSnapshot(neural);
  MappedBackpropSnapshot(neural);
  NeuralGlobalMethods(neural);

  World(simulation, world);
}

} // namespace python
} // namespace dart
