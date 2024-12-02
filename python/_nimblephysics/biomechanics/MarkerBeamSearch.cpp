#include <memory>

#include <Eigen/Dense>
#include <dart/biomechanics/MarkerBeamSearch.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerBeamSearch(py::module& m)
{
  // Binding for the Beam class
  py::class_<
      dart::biomechanics::Beam,
      std::shared_ptr<dart::biomechanics::Beam>>(m, "Beam")
      .def(
          py::init<
              const std::string&,
              double,
              bool,
              const Eigen::Vector3d&,
              double,
              const Eigen::Vector3d&,
              std::shared_ptr<dart::biomechanics::Beam>>(),
          py::arg("label"),
          py::arg("cost"),
          py::arg("observed_this_timestep"),
          py::arg("last_observed_point"),
          py::arg("last_observed_timestamp"),
          py::arg("last_observed_velocity"),
          py::arg("parent"))
      .def_readonly("label", &dart::biomechanics::Beam::label)
      .def_readonly("cost", &dart::biomechanics::Beam::cost)
      .def_readonly(
          "observed_this_timestep",
          &dart::biomechanics::Beam::observed_this_timestep)
      .def_readonly(
          "last_observed_point", &dart::biomechanics::Beam::last_observed_point)
      .def_readonly(
          "last_observed_timestamp",
          &dart::biomechanics::Beam::last_observed_timestamp)
      .def_readonly(
          "last_observed_velocity",
          &dart::biomechanics::Beam::last_observed_velocity)
      .def_readonly("parent", &dart::biomechanics::Beam::parent);

  // Binding for the MarkerBeamSearch class
  py::class_<dart::biomechanics::MarkerBeamSearch>(m, "MarkerBeamSearch")
      .def(
          py::init<
              const Eigen::Vector3d&,
              double,
              const std::string&,
              double,
              double>(),
          py::arg("seed_point"),
          py::arg("seed_timestamp"),
          py::arg("seed_label"),
          py::arg("vel_threshold") = 7.0,
          py::arg("acc_threshold") = 2000.0)
      .def(
          "make_next_generation",
          &dart::biomechanics::MarkerBeamSearch::make_next_generation,
          py::arg("markers"),
          py::arg("timestamp"))
      .def(
          "prune_beams",
          &dart::biomechanics::MarkerBeamSearch::prune_beams,
          py::arg("beam_width"))
      .def_static(
          "convert_to_trace",
          &dart::biomechanics::MarkerBeamSearch::convert_to_trace,
          py::arg("beam"))
      .def_readonly("beams", &dart::biomechanics::MarkerBeamSearch::beams)
      .def_static(
          "search",
          &dart::biomechanics::MarkerBeamSearch::search,
          py::arg("label"),
          py::arg("marker_observations"),
          py::arg("timestamps"),
          py::arg("beam_width") = 20,
          py::arg("vel_threshold") = 7.0,
          py::arg("acc_threshold") = 2000.0,
          py::return_value_policy::automatic);
}

} // namespace python
} // namespace dart
