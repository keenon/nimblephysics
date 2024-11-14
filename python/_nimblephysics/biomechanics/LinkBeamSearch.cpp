#include "dart/biomechanics/LinkBeamSearch.hpp"

#include <memory>

#include <Eigen/Dense>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {

using namespace biomechanics;

namespace python {

void LinkBeamSearch(py::module& m)
{
  // Binding for the LinkBeam class
  py::class_<LinkBeam, std::shared_ptr<LinkBeam>>(m, "LinkBeam")
      .def(
          py::init<
              double,
              const std::string&,
              bool,
              const Eigen::VectorXd&,
              double,
              const Eigen::VectorXd&,
              const std::string&,
              bool,
              const Eigen::VectorXd&,
              double,
              const Eigen::VectorXd&,
              std::shared_ptr<LinkBeam>>(),
          py::arg("cost"),
          py::arg("a_label"),
          py::arg("a_observed_this_timestep"),
          py::arg("a_last_observed_point"),
          py::arg("a_last_observed_timestamp"),
          py::arg("a_last_observed_velocity"),
          py::arg("b_label"),
          py::arg("b_observed_this_timestep"),
          py::arg("b_last_observed_point"),
          py::arg("b_last_observed_timestamp"),
          py::arg("b_last_observed_velocity"),
          py::arg("parent") = nullptr)
      .def_readonly("a_label", &LinkBeam::a_label)
      .def_readonly("b_label", &LinkBeam::b_label)
      .def_readonly(
          "a_observed_this_timestep", &LinkBeam::a_observed_this_timestep)
      .def_readonly("a_last_observed_point", &LinkBeam::a_last_observed_point)
      .def_readonly(
          "a_last_observed_timestamp", &LinkBeam::a_last_observed_timestamp)
      .def_readonly(
          "a_last_observed_velocity", &LinkBeam::a_last_observed_velocity)
      .def_readonly(
          "b_observed_this_timestep", &LinkBeam::b_observed_this_timestep)
      .def_readonly("b_last_observed_point", &LinkBeam::b_last_observed_point)
      .def_readonly(
          "b_last_observed_timestamp", &LinkBeam::b_last_observed_timestamp)
      .def_readonly(
          "b_last_observed_velocity", &LinkBeam::b_last_observed_velocity)
      .def_readonly("cost", &LinkBeam::cost)
      .def_readonly("parent", &LinkBeam::parent);

  // Binding for the LinkBeamSearch class
  py::class_<dart::biomechanics::LinkBeamSearch>(m, "LinkBeamSearch")
      .def(
          py::init<
              const Eigen::VectorXd&,
              const std::string&,
              const Eigen::VectorXd&,
              const std::string&,
              double,
              double,
              double,
              double,
              double,
              double,
              double,
              double>(),
          py::arg("seed_a_point"),
          py::arg("seed_a_label"),
          py::arg("seed_b_point"),
          py::arg("seed_b_label"),
          py::arg("seed_timestamp"),
          py::arg("pair_dist"),
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.01,
          py::arg("vel_weight") = 1.0,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.001,
          py::arg("acc_threshold") = 1000.0)
      .def(
          "make_next_generation",
          &LinkBeamSearch::make_next_generation,
          py::arg("markers"),
          py::arg("timestamp"),
          py::arg("beam_width"))
      .def("prune_beams", &LinkBeamSearch::prune_beams, py::arg("beam_width"))
      .def_readonly("beams", &LinkBeamSearch::beams)
      .def_static(
          "convert_to_traces",
          &LinkBeamSearch::convert_to_traces,
          py::arg("beam"))
      .def_static(
          "search",
          &LinkBeamSearch::search,
          py::arg("a_label"),
          py::arg("b_label"),
          py::arg("marker_observations"),
          py::arg("timestamps"),
          py::arg("beam_width") = 5,
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.01,
          py::arg("vel_weight") = 1.0,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.001,
          py::arg("acc_threshold") = 1000.0,
          py::arg("print_updates") = true,
          py::return_value_policy::automatic)
      .def_static(
          "process_markers",
          &LinkBeamSearch::process_markers,
          py::arg("label_pairs"),
          py::arg("marker_observations"),
          py::arg("timestamps"),
          py::arg("beam_width") = 5,
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.01,
          py::arg("vel_weight") = 0.1,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.001,
          py::arg("acc_threshold") = 1000.0,
          py::arg("print_updates") = true,
          py::arg("multithread") = true,
          py::return_value_policy::automatic);
}

} // namespace python
} // namespace dart