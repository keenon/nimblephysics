#ifndef LINK_BEAM_H
#define LINK_BEAM_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace biomechanics {

class LinkBeam
{
public:
  std::string a_label;
  std::string b_label;

  bool a_observed_this_timestep;
  Eigen::VectorXd a_last_observed_point;
  double a_last_observed_timestamp;
  Eigen::VectorXd a_last_observed_velocity;

  bool b_observed_this_timestep;
  Eigen::VectorXd b_last_observed_point;
  double b_last_observed_timestamp;
  Eigen::VectorXd b_last_observed_velocity;

  double cost;
  std::weak_ptr<LinkBeam> parent;

  LinkBeam(
      double cost,
      const std::string& a_label,
      bool a_observed_this_timestep,
      const Eigen::VectorXd& a_last_observed_point,
      double a_last_observed_timestamp,
      const Eigen::VectorXd& a_last_observed_velocity,
      const std::string& b_label,
      bool b_observed_this_timestep,
      const Eigen::VectorXd& b_last_observed_point,
      double b_last_observed_timestamp,
      const Eigen::VectorXd& b_last_observed_velocity,
      std::shared_ptr<LinkBeam> parent = nullptr);
};

class LinkBeamSearch
{
public:
  std::vector<std::shared_ptr<LinkBeam>> beams;
  std::vector<std::vector<std::shared_ptr<LinkBeam>>> old_beams;
  double pair_dist;
  double pair_weight;
  double pair_threshold;
  double vel_weight;
  double vel_threshold;
  double acc_weight;
  double acc_threshold;
  // Total durations of different parts of the computation
  std::chrono::duration<double> pair_distances_cost;
  std::chrono::duration<double> create_options_cost;
  std::chrono::duration<double> create_beams_cost;
  std::chrono::duration<double> prune_beams_cost;

  LinkBeamSearch(
      const Eigen::VectorXd& seed_a_point,
      const std::string& seed_a_label,
      const Eigen::VectorXd& seed_b_point,
      const std::string& seed_b_label,
      double seed_timestamp,
      double pair_dist,
      double pair_weight = 100.0,
      double pair_threshold = 0.01,
      double vel_weight = 1.0,
      double vel_threshold = 5.0,
      double acc_weight = 0.01,
      double acc_threshold = 1000.0);

  void make_next_generation(
      const std::map<std::string, Eigen::VectorXd>& markers,
      double timestamp,
      size_t beam_width);
  void prune_beams(size_t beam_width);

  static std::tuple<
      std::vector<Eigen::VectorXd>,
      std::vector<double>,
      std::string,
      std::vector<Eigen::VectorXd>,
      std::vector<double>,
      std::string>
  convert_to_traces(const std::shared_ptr<LinkBeam>& beam);

  static std::tuple<
      std::vector<Eigen::VectorXd>,
      std::vector<double>,
      std::string,
      std::vector<Eigen::VectorXd>,
      std::vector<double>,
      std::string>
  search(
      const std::string& a_label,
      const std::string& b_label,
      const std::vector<std::map<std::string, Eigen::VectorXd>>&
          marker_observations,
      const std::vector<double>& timestamps,
      size_t beam_width = 20,
      double pair_weight = 100.0,
      double pair_threshold = 0.001,
      double vel_weight = 0.1,
      double vel_threshold = 5.0,
      double acc_weight = 0.001,
      double acc_threshold = 500.0,
      bool print_updates = true);

  static std::tuple<
      std::vector<std::map<std::string, Eigen::VectorXd>>,
      std::vector<double>>
  process_markers(
      const std::vector<std::pair<std::string, std::string>>& label_pairs,
      const std::vector<std::map<std::string, Eigen::VectorXd>>&
          marker_observations,
      const std::vector<double>& timestamps,
      size_t beam_width = 20,
      double pair_weight = 100.0,
      double pair_threshold = 0.001,
      double vel_weight = 0.1,
      double vel_threshold = 5.0,
      double acc_weight = 0.001,
      double acc_threshold = 500.0,
      bool print_updates = true,
      bool multithread = true);
};

} // namespace biomechanics
} // namespace dart

#endif // LINK_BEAM_H
