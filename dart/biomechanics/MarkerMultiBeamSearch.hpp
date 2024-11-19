// MarkerMultiBeamSearch.h

#ifndef MULTIBEAMSEARCH_H
#define MULTIBEAMSEARCH_H

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace biomechanics {

class TraceHead
{
public:
  std::string label;
  bool observed_this_timestep;
  Eigen::Vector3d last_observed_point;
  double last_observed_timestamp;
  int last_observed_index;
  Eigen::Vector3d last_observed_velocity;
  std::weak_ptr<TraceHead> parent;

  TraceHead(
      const std::string& label,
      bool observed_this_timestep,
      const Eigen::Vector3d& last_observed_point,
      double last_observed_timestamp,
      int last_observed_index,
      const Eigen::Vector3d& last_observed_velocity,
      std::shared_ptr<TraceHead> parent = nullptr);
};

class MultiBeam
{
public:
  double cost;
  std::vector<std::shared_ptr<TraceHead>> trace_heads;
  std::set<std::string> timestep_used_markers;

  MultiBeam(
      double cost,
      const std::vector<std::shared_ptr<TraceHead>>& trace_heads,
      const std::set<std::string>& timestep_used_markers);

  std::vector<std::shared_ptr<TraceHead>> get_child_trace_heads(
      const std::shared_ptr<TraceHead>& trace_head, int index) const;
};

class MarkerMultiBeamSearch
{
public:
  std::vector<std::shared_ptr<MultiBeam>> beams;
  Eigen::MatrixXd pairwise_distances;

  double pair_weight;
  double pair_threshold;
  double vel_weight;
  double vel_threshold;
  double acc_weight;
  double acc_threshold;

  // We try to prune the beams as we go, and store the finished marker
  // observations.
  std::vector<std::map<std::string, Eigen::Vector3d>> marker_observations;
  std::vector<double> timestamps;

  // For each timestep, store the beams that were alive at that timestep with a
  // shared_ptr to keep them alive
  std::vector<std::vector<std::shared_ptr<MultiBeam>>> past_beams;

  MarkerMultiBeamSearch(
      const std::vector<Eigen::Vector3d>& seed_points,
      const std::vector<std::string>& seed_labels,
      double seed_timestamp,
      int seed_index,
      Eigen::MatrixXd pairwise_distances,
      double pair_weight = 100.0,
      double pair_threshold = 0.01,
      double vel_weight = 1.0,
      double vel_threshold = 5.0,
      double acc_weight = 0.01,
      double acc_threshold = 1000.0);

  void make_next_generation(
      const std::map<std::string, Eigen::Vector3d>& markers,
      double timestamp,
      int index,
      int trace_head_to_attach,
      int beam_width);

  void prune_beams(int beam_width);

  static std::pair<
      std::vector<std::map<std::string, Eigen::Vector3d>>,
      std::vector<double>>
  convert_to_traces(const std::shared_ptr<MultiBeam>& beam);

  void crystallize_beams(bool include_last = true);

  static double get_median_70_percent_mean_distance(
      std::string label_1,
      std::string label_2,
      const std::vector<std::map<std::string, Eigen::Vector3d>>&
          marker_observations);

  static std::pair<
      std::vector<std::map<std::string, Eigen::Vector3d>>,
      std::vector<double>>
  search(
      const std::vector<std::string>& labels,
      const std::vector<std::map<std::string, Eigen::Vector3d>>&
          marker_observations,
      const std::vector<double>& timestamps,
      int beam_width = 20,
      double pair_weight = 100.0,
      double pair_threshold = 0.01,
      double vel_weight = 1.0,
      double vel_threshold = 5.0,
      double acc_weight = 0.01,
      double acc_threshold = 1000.0,
      int print_interval = 1000,
      int crysatilize_interval = 1000);

  static std::tuple<
      std::vector<std::map<std::string, Eigen::Vector3d>>,
      std::vector<double>>
  process_markers(
      const std::vector<std::vector<std::string>>& label_groups,
      const std::vector<std::map<std::string, Eigen::Vector3d>>&
          marker_observations,
      const std::vector<double>& timestamps,
      size_t beam_width = 20,
      double pair_weight = 100.0,
      double pair_threshold = 0.001,
      double vel_weight = 0.1,
      double vel_threshold = 5.0,
      double acc_weight = 0.001,
      double acc_threshold = 500.0,
      int print_interval = 1000,
      int crysatilize_interval = 1000,
      bool multithread = true);
};

} // namespace biomechanics
} // namespace dart

#endif // MULTIBEAMSEARCH_H
