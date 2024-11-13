#include "dart/biomechanics/LinkBeamSearch.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

namespace dart {
namespace biomechanics {

LinkBeam::LinkBeam(
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
    std::shared_ptr<LinkBeam> parent)
  : cost(cost),
    a_label(a_label),
    b_label(b_label),
    a_observed_this_timestep(a_observed_this_timestep),
    a_last_observed_point(a_last_observed_point),
    a_last_observed_timestamp(a_last_observed_timestamp),
    a_last_observed_velocity(a_last_observed_velocity),
    b_observed_this_timestep(b_observed_this_timestep),
    b_last_observed_point(b_last_observed_point),
    b_last_observed_timestamp(b_last_observed_timestamp),
    b_last_observed_velocity(b_last_observed_velocity),
    parent(parent)
{
}

LinkBeamSearch::LinkBeamSearch(
    const Eigen::VectorXd& seed_a_point,
    const std::string& seed_a_label,
    const Eigen::VectorXd& seed_b_point,
    const std::string& seed_b_label,
    double seed_timestamp,
    double pair_dist,
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
    double acc_threshold)
  : pair_dist(pair_dist),
    pair_weight(pair_weight),
    pair_threshold(pair_threshold),
    vel_weight(vel_weight),
    vel_threshold(vel_threshold),
    acc_weight(acc_weight),
    acc_threshold(acc_threshold)
{
  beams.emplace_back(std::make_shared<LinkBeam>(
      0.0,
      seed_a_label,
      true,
      seed_a_point,
      seed_timestamp,
      Eigen::VectorXd::Zero(seed_a_point.size()),
      seed_b_label,
      true,
      seed_b_point,
      seed_timestamp,
      Eigen::VectorXd::Zero(seed_b_point.size()),
      nullptr));
}

void LinkBeamSearch::make_next_generation(
    const std::map<std::string, Eigen::VectorXd>& markers, double timestamp)
{
  std::vector<std::shared_ptr<LinkBeam>> new_beams;
  for (const auto& beam : beams)
  {
    std::vector<std::pair<std::string, double>> a_point_options;
    std::vector<std::pair<std::string, double>> b_point_options;

    // Initialize options with None equivalent
    a_point_options.emplace_back(
        "", vel_threshold * vel_weight + acc_threshold * acc_weight);
    b_point_options.emplace_back(
        "", vel_threshold * vel_weight + acc_threshold * acc_weight);

    // 1. Find the options for the next point for each marker
    for (const auto& marker_pair : markers)
    {
      const std::string& label = marker_pair.first;
      const Eigen::VectorXd& point = marker_pair.second;

      // For 'a' marker
      Eigen::VectorXd a_velocity
          = (point - beam->a_last_observed_point)
            / (timestamp - beam->a_last_observed_timestamp);
      Eigen::VectorXd a_acc = (a_velocity - beam->a_last_observed_velocity)
                              / (timestamp - beam->a_last_observed_timestamp);
      double a_vel_mag = a_velocity.norm();

      if (a_vel_mag < 2 * vel_threshold)
      {
        double acc_mag = a_acc.norm();
        double cost = a_vel_mag * vel_weight + acc_mag * acc_weight;
        a_point_options.emplace_back(label, cost);
      }

      // For 'b' marker
      Eigen::VectorXd b_velocity
          = (point - beam->b_last_observed_point)
            / (timestamp - beam->b_last_observed_timestamp);
      Eigen::VectorXd b_acc = (b_velocity - beam->b_last_observed_velocity)
                              / (timestamp - beam->b_last_observed_timestamp);
      double b_vel_mag = b_velocity.norm();

      if (b_vel_mag < 2 * vel_threshold)
      {
        double acc_mag = b_acc.norm();
        double cost = b_vel_mag * vel_weight + acc_mag * acc_weight;
        b_point_options.emplace_back(label, cost);
      }
    }

    // 2. Create new beams for each pair of options
    for (const auto& a_option : a_point_options)
    {
      const std::string& a_label = a_option.first;
      double a_cost = a_option.second;

      for (const auto& b_option : b_point_options)
      {
        const std::string& b_label = b_option.first;

        if (b_label == a_label && !a_label.empty())
          continue;

        double b_cost = b_option.second;

        double pair_cost = pair_threshold * pair_weight;
        Eigen::VectorXd new_a_point = beam->a_last_observed_point;
        double new_a_timestamp = beam->a_last_observed_timestamp;
        Eigen::VectorXd new_a_velocity = beam->a_last_observed_velocity;
        bool a_observed = false;

        if (!a_label.empty())
        {
          new_a_point = markers.at(a_label);
          new_a_timestamp = timestamp;
          new_a_velocity = (new_a_point - beam->a_last_observed_point)
                           / (timestamp - beam->a_last_observed_timestamp);
          a_observed = true;
        }

        Eigen::VectorXd new_b_point = beam->b_last_observed_point;
        double new_b_timestamp = beam->b_last_observed_timestamp;
        Eigen::VectorXd new_b_velocity = beam->b_last_observed_velocity;
        bool b_observed = false;

        if (!b_label.empty())
        {
          new_b_point = markers.at(b_label);
          new_b_timestamp = timestamp;
          new_b_velocity = (new_b_point - beam->b_last_observed_point)
                           / (timestamp - beam->b_last_observed_timestamp);
          b_observed = true;
        }

        if (!a_label.empty() && !b_label.empty())
        {
          double real_dist = (new_a_point - new_b_point).norm();
          pair_cost = std::abs(real_dist - pair_dist) * pair_weight;
        }

        double total_cost = beam->cost + a_cost + b_cost + pair_cost;

        new_beams.emplace_back(std::make_shared<LinkBeam>(
            total_cost,
            a_label.empty() ? beam->a_label : a_label,
            a_observed,
            new_a_point,
            new_a_timestamp,
            new_a_velocity,
            b_label.empty() ? beam->b_label : b_label,
            b_observed,
            new_b_point,
            new_b_timestamp,
            new_b_velocity,
            beam));
      }
    }
  }
  old_beams.push_back(beams);
  beams = new_beams;
}

void LinkBeamSearch::prune_beams(size_t beam_width)
{
  std::sort(
      beams.begin(),
      beams.end(),
      [](const std::shared_ptr<LinkBeam>& a,
         const std::shared_ptr<LinkBeam>& b) { return a->cost < b->cost; });
  if (beams.size() > beam_width)
  {
    beams.resize(beam_width);
  }
}

std::tuple<
    std::vector<Eigen::VectorXd>,
    std::vector<double>,
    std::string,
    std::vector<Eigen::VectorXd>,
    std::vector<double>,
    std::string>
LinkBeamSearch::convert_to_traces(const std::shared_ptr<LinkBeam>& beam)
{
  std::vector<Eigen::VectorXd> a_points;
  std::vector<double> a_timestamps;
  std::map<std::string, int> a_label_count;

  std::vector<Eigen::VectorXd> b_points;
  std::vector<double> b_timestamps;
  std::map<std::string, int> b_label_count;

  auto current_beam = beam;
  while (current_beam)
  {
    if (current_beam->a_observed_this_timestep)
    {
      a_points.push_back(current_beam->a_last_observed_point);
      a_timestamps.push_back(current_beam->a_last_observed_timestamp);
      a_label_count[current_beam->a_label]++;
    }
    if (current_beam->b_observed_this_timestep)
    {
      b_points.push_back(current_beam->b_last_observed_point);
      b_timestamps.push_back(current_beam->b_last_observed_timestamp);
      b_label_count[current_beam->b_label]++;
    }
    current_beam = current_beam->parent.lock();
  }

  // Reverse the vectors to get the correct order
  std::reverse(a_points.begin(), a_points.end());
  std::reverse(a_timestamps.begin(), a_timestamps.end());
  std::reverse(b_points.begin(), b_points.end());
  std::reverse(b_timestamps.begin(), b_timestamps.end());

  // Find the label with the maximum votes
  auto a_max_vote_label = std::max_element(
                              a_label_count.begin(),
                              a_label_count.end(),
                              [](const std::pair<std::string, int>& a,
                                 const std::pair<std::string, int>& b) {
                                return a.second < b.second;
                              })
                              ->first;

  auto b_max_vote_label = std::max_element(
                              b_label_count.begin(),
                              b_label_count.end(),
                              [](const std::pair<std::string, int>& a,
                                 const std::pair<std::string, int>& b) {
                                return a.second < b.second;
                              })
                              ->first;

  return std::make_tuple(
      a_points,
      a_timestamps,
      a_max_vote_label,
      b_points,
      b_timestamps,
      b_max_vote_label);
}

std::tuple<
    std::vector<Eigen::VectorXd>,
    std::vector<double>,
    std::string,
    std::vector<Eigen::VectorXd>,
    std::vector<double>,
    std::string>
LinkBeamSearch::search(
    const std::string& a_label,
    const std::string& b_label,
    const std::vector<std::map<std::string, Eigen::VectorXd>>&
        marker_observations,
    const std::vector<double>& timestamps,
    size_t beam_width,
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
    double acc_threshold)
{
  // 1. Scan through to find the first observation of both labels
  int first_observation_index = -1;
  for (size_t i = 0; i < marker_observations.size(); ++i)
  {
    const auto& markers = marker_observations[i];
    if (markers.find(a_label) != markers.end()
        && markers.find(b_label) != markers.end())
    {
      first_observation_index = static_cast<int>(i);
      break;
    }
  }

  // 2. If the label is not observed, return an empty pair of traces
  if (first_observation_index == -1)
  {
    return std::make_tuple(
        std::vector<Eigen::VectorXd>(),
        std::vector<double>(),
        a_label,
        std::vector<Eigen::VectorXd>(),
        std::vector<double>(),
        b_label);
  }

  // 3. Figure out the distance between the markers
  std::vector<double> observed_distances;
  for (size_t t = 0; t < marker_observations.size(); ++t)
  {
    const auto& marker_timestep = marker_observations[t];
    if (marker_timestep.find(a_label) != marker_timestep.end()
        && marker_timestep.find(b_label) != marker_timestep.end())
    {
      double dist
          = (marker_timestep.at(a_label) - marker_timestep.at(b_label)).norm();
      observed_distances.push_back(dist);
    }
  }

  // Calculate the median
  std::vector<double> sorted_distances = observed_distances;
  std::sort(sorted_distances.begin(), sorted_distances.end());
  size_t n = sorted_distances.size();
  double median
      = (n % 2 == 0)
            ? (sorted_distances[n / 2 - 1] + sorted_distances[n / 2]) / 2.0
            : sorted_distances[n / 2];

  // Calculate absolute distances from the median
  std::vector<double> distance_to_median;
  for (double dist : observed_distances)
  {
    distance_to_median.push_back(std::abs(dist - median));
  }

  // Select the 70% of data closest to the median
  std::vector<size_t> indices(distance_to_median.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(),
      indices.end(),
      [&distance_to_median](size_t i1, size_t i2) {
        return distance_to_median[i1] < distance_to_median[i2];
      });

  size_t threshold_index = static_cast<size_t>(observed_distances.size() * 0.7);
  std::vector<double> closest_70_percent;
  for (size_t i = 0; i < threshold_index; ++i)
  {
    closest_70_percent.push_back(observed_distances[indices[i]]);
  }

  // Calculate the mean of the 70% of data closest to the median
  double mean_70
      = std::accumulate(
            closest_70_percent.begin(), closest_70_percent.end(), 0.0)
        / closest_70_percent.size();

  // 3. Otherwise, start the beam search
  const Eigen::VectorXd& a_seed_point
      = marker_observations[first_observation_index].at(a_label);
  const Eigen::VectorXd& b_seed_point
      = marker_observations[first_observation_index].at(b_label);
  double seed_timestamp = timestamps[first_observation_index];

  LinkBeamSearch beam_search(
      a_seed_point,
      a_label,
      b_seed_point,
      b_label,
      seed_timestamp,
      mean_70,
      pair_weight,
      pair_threshold,
      vel_weight,
      vel_threshold,
      acc_weight,
      acc_threshold);

  for (size_t i = first_observation_index + 1; i < marker_observations.size();
       ++i)
  {
    if (i % 1000 == 0)
    {
      std::cout << "Beam searching timestep: " << i << "/"
                << marker_observations.size()
                << ", num beams: " << beam_search.beams.size() << std::endl;
    }
    beam_search.make_next_generation(marker_observations[i], timestamps[i]);
    beam_search.prune_beams(beam_width);
  }

  return LinkBeamSearch::convert_to_traces(beam_search.beams.front());
}

} // namespace biomechanics
} // namespace dart