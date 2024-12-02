// MultiBeamSearch_bindings.cpp

#include <memory>

#include <Eigen/Dense>
#include <dart/biomechanics/MarkerMultiBeamSearch.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerMultiBeamSearch(py::module& m)
{
  // Binding for the TraceHead class
  py::class_<
      dart::biomechanics::TraceHead,
      std::shared_ptr<dart::biomechanics::TraceHead>>(m, "TraceHead")
      .def(
          py::init<
              const std::string&,
              bool,
              const Eigen::Vector3d&,
              double,
              int,
              const Eigen::Vector3d&,
              std::shared_ptr<dart::biomechanics::TraceHead>>(),
          py::arg("label"),
          py::arg("observed_this_timestep"),
          py::arg("last_observed_point"),
          py::arg("last_observed_timestamp"),
          py::arg("last_observed_index"),
          py::arg("last_observed_velocity"),
          py::arg("parent") = nullptr)
      .def_readonly("label", &dart::biomechanics::TraceHead::label)
      .def_readonly(
          "observed_this_timestep",
          &dart::biomechanics::TraceHead::observed_this_timestep)
      .def_readonly(
          "last_observed_point",
          &dart::biomechanics::TraceHead::last_observed_point)
      .def_readonly(
          "last_observed_timestamp",
          &dart::biomechanics::TraceHead::last_observed_timestamp)
      .def_readonly(
          "last_observed_index",
          &dart::biomechanics::TraceHead::last_observed_index)
      .def_readonly(
          "last_observed_velocity",
          &dart::biomechanics::TraceHead::last_observed_velocity)
      .def_readonly("parent", &dart::biomechanics::TraceHead::parent);

  // Binding for the MultiBeam class
  py::class_<
      dart::biomechanics::MultiBeam,
      std::shared_ptr<dart::biomechanics::MultiBeam>>(m, "MultiBeam")
      .def(
          py::init<
              double,
              const std::vector<
                  std::shared_ptr<dart::biomechanics::TraceHead>>&,
              const std::set<std::string>&>(),
          py::arg("cost"),
          py::arg("trace_heads"),
          py::arg("timestep_used_markers"))
      .def(
          "get_child_trace_heads",
          &dart::biomechanics::MultiBeam::get_child_trace_heads,
          py::arg("trace_head"),
          py::arg("index"))
      .def_readonly("cost", &dart::biomechanics::MultiBeam::cost)
      .def_readonly("trace_heads", &dart::biomechanics::MultiBeam::trace_heads)
      .def_readonly(
          "timestep_used_markers",
          &dart::biomechanics::MultiBeam::timestep_used_markers);

  // Binding for the MarkerMultiBeamSearch class
  py::class_<dart::biomechanics::MarkerMultiBeamSearch>(
      m, "MarkerMultiBeamSearch")
      .def(
          py::init<
              const std::vector<Eigen::Vector3d>&,
              const std::vector<std::string>&,
              double,
              int,
              Eigen::MatrixXd,
              double,
              double,
              double,
              double,
              double,
              double>(),
          py::arg("seed_points"),
          py::arg("seed_labels"),
          py::arg("seed_timestamp"),
          py::arg("seed_index"),
          py::arg("pairwise_distances"),
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.01,
          py::arg("vel_weight") = 1.0,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.01,
          py::arg("acc_threshold") = 1000.0)
      .def(
          "make_next_generation",
          &dart::biomechanics::MarkerMultiBeamSearch::make_next_generation,
          py::arg("markers"),
          py::arg("timestamp"),
          py::arg("index"),
          py::arg("trace_head_to_attach"),
          py::arg("beam_width"))
      .def(
          "prune_beams",
          &dart::biomechanics::MarkerMultiBeamSearch::prune_beams,
          py::arg("beam_width"))
      .def(
          "crysatilize_beams",
          &dart::biomechanics::MarkerMultiBeamSearch::crystallize_beams,
          py::arg("include_last") = true)
      .def_readonly("beams", &dart::biomechanics::MarkerMultiBeamSearch::beams)
      .def_readonly(
          "vel_threshold",
          &dart::biomechanics::MarkerMultiBeamSearch::vel_threshold)
      .def_readonly(
          "acc_threshold",
          &dart::biomechanics::MarkerMultiBeamSearch::acc_threshold)
      .def_readonly(
          "pair_weight",
          &dart::biomechanics::MarkerMultiBeamSearch::pair_weight)
      .def_static(
          "convert_to_traces",
          &dart::biomechanics::MarkerMultiBeamSearch::convert_to_traces,
          py::arg("beam"))
      .def_static(
          "get_median_70_percent_mean_distance",
          [](std::string label_1,
             std::string label_2,
             const std::vector<std::map<std::string, Eigen::Vector3d>>&
                 marker_observations) {
            return dart::biomechanics::MarkerMultiBeamSearch::
                get_median_70_percent_mean_distance(
                    label_1, label_2, marker_observations);
          })
      .def_static(
          "search",
          &dart::biomechanics::MarkerMultiBeamSearch::search,
          py::arg("labels"),
          py::arg("marker_observations"),
          py::arg("timestamps"),
          py::arg("beam_width") = 20,
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.01,
          py::arg("vel_weight") = 1.0,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.01,
          py::arg("acc_threshold") = 1000.0,
          py::arg("print_interval") = 1000,
          py::arg("crysatilize_interval") = 1000)
      .def_static(
          "process_markers",
          &dart::biomechanics::MarkerMultiBeamSearch::process_markers,
          py::arg("label_groups"),
          py::arg("marker_observations"),
          py::arg("timestamps"),
          py::arg("beam_width") = 20,
          py::arg("pair_weight") = 100.0,
          py::arg("pair_threshold") = 0.001,
          py::arg("vel_weight") = 0.1,
          py::arg("vel_threshold") = 5.0,
          py::arg("acc_weight") = 0.001,
          py::arg("acc_threshold") = 500.0,
          py::arg("print_interval") = 1000,
          py::arg("crysatilize_interval") = 1000,
          py::arg("multithread") = true);
}

} // namespace python
} // namespace dart