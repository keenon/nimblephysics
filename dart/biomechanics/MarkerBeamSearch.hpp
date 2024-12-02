#ifndef DART_BIOMECH_MARKERBEAMSEARCH_HPP_
#define DART_BIOMECH_MARKERBEAMSEARCH_HPP_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace biomechanics {

class Beam
{
public:
  std::string label;
  double cost;
  bool observed_this_timestep;
  Eigen::Vector3d last_observed_point;
  double last_observed_timestamp;
  Eigen::Vector3d last_observed_velocity;
  std::shared_ptr<Beam> parent;

  Beam(
      const std::string& label,
      double cost,
      bool observed_this_timestep,
      const Eigen::Vector3d& last_observed_point,
      double last_observed_timestamp,
      const Eigen::Vector3d& last_observed_velocity,
      std::shared_ptr<Beam> parent);
};

class MarkerBeamSearch
{
public:
  std::vector<std::shared_ptr<Beam>> beams;
  double vel_threshold;
  double acc_threshold;

  MarkerBeamSearch(
      const Eigen::Vector3d& seed_point,
      double seed_timestamp,
      const std::string& seed_label,
      double vel_threshold = 5.0,
      double acc_threshold = 175.0);

  void make_next_generation(
      const std::map<std::string, Eigen::Vector3d>& markers, double timestamp);

  void prune_beams(int beam_width);

  static std::
      tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::string>
      convert_to_trace(std::shared_ptr<Beam> beam);

  static std::
      tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::string>
      search(
          const std::string& label,
          const std::vector<std::map<std::string, Eigen::Vector3d>>&
              marker_observations,
          const std::vector<double>& timestamps,
          int beam_width = 20,
          double vel_threshold = 7.0,
          double acc_threshold = 2000.0);
};

} // namespace biomechanics
} // namespace dart

#endif