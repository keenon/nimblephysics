#include <Eigen/Dense>
#include <dart/math/GraphFlowDiscretizer.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void GraphFlowDiscretizer(py::module& m)
{
  ::py::class_<dart::math::ParticlePath>(m, "ParticlePath")
      .def_readwrite("startTime", &dart::math::ParticlePath::startTime)
      .def_readwrite("energyValue", &dart::math::ParticlePath::energyValue)
      .def_readwrite("nodeHistory", &dart::math::ParticlePath::nodeHistory);

  ::py::class_<dart::math::GraphFlowDiscretizer>(m, "GraphFlowDiscretizer")
      .def(
          ::py::
              init<int, std::vector<std::pair<int, int>>, std::vector<bool>>(),
          ::py::arg("numNodes"),
          ::py::arg("arcs"),
          ::py::arg("nodeAttachedToSink"))
      .def(
          "cleanUpArcRates",
          &dart::math::GraphFlowDiscretizer::cleanUpArcRates,
          ::py::arg("energyLevels"),
          ::py::arg("arcRates"),
          "This will find the least-squares closest rates of transfer across "
          "the arcs to end up with the energy levels at each node we got over "
          "time. The idea here is that arc rates may not perfectly reflect the "
          "observed changes in energy levels.")
      .def(
          "discretize",
          &dart::math::GraphFlowDiscretizer::discretize,
          ::py::arg("maxSimultaneousParticles"),
          ::py::arg("energyLevels"),
          ::py::arg("arcRates"),
          "This will attempt to create a set of ParticlePath objects that map "
          "the recorded graph node levels and flows as closely as possible. "
          "The particles can be created and destroyed within the arcs.");
}

} // namespace python
} // namespace dart
