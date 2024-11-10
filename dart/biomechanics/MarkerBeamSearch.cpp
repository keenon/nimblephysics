#include "dart/biomechanics/MarkerBeamSearch.hpp"

namespace dart {
namespace biomechanics {

Beam::Beam(
    const std::string& label,
    double cost,
    bool observed_this_timestep,
    const Eigen::Vector3d& last_observed_point,
    double last_observed_timestamp,
    const Eigen::Vector3d& last_observed_velocity,
    std::shared_ptr<Beam> parent)
  : label(label),
    cost(cost),
    observed_this_timestep(observed_this_timestep),
    last_observed_point(last_observed_point),
    last_observed_timestamp(last_observed_timestamp),
    last_observed_velocity(last_observed_velocity),
    parent(parent)
{
}

MarkerBeamSearch::MarkerBeamSearch(
    const Eigen::Vector3d& seed_point,
    double seed_timestamp,
    const std::string& seed_label,
    double vel_threshold,
    double acc_threshold)
  : vel_threshold(vel_threshold), acc_threshold(acc_threshold)
{
  beams.emplace_back(std::make_shared<Beam>(
      seed_label,
      0.0,
      true,
      seed_point,
      seed_timestamp,
      Eigen::Vector3d::Zero(),
      nullptr));
}

void MarkerBeamSearch::make_next_generation(
    const std::map<std::string, Eigen::Vector3d>& markers, double timestamp)
{
  std::vector<std::shared_ptr<Beam>> new_beams;
  for (const auto& beam : beams)
  {
    // 1. Always append a beam option where we don't add a marker
    double skip_cost = beam->cost + vel_threshold + acc_threshold;
    auto new_beam = std::make_shared<Beam>(
        beam->label,
        skip_cost,
        false,
        beam->last_observed_point,
        timestamp,
        beam->last_observed_velocity,
        beam);
    new_beams.push_back(new_beam);

    // 2. For each marker, add a beam option where we add that marker
    for (auto it = markers.begin(); it != markers.end(); ++it)
    {
      const std::string& label = it->first;
      const Eigen::Vector3d& point = it->second;
      double dt = timestamp - beam->last_observed_timestamp;
      if (dt == 0)
        continue; // Avoid division by zero
      Eigen::Vector3d velocity = (point - beam->last_observed_point) / dt;
      Eigen::Vector3d acc = (velocity - beam->last_observed_velocity) / dt;

      double vel_mag = velocity.norm();
      if (vel_mag < 2 * vel_threshold)
      {
        double acc_mag = acc.norm();
        double cost = beam->cost + vel_mag + acc_mag;
        auto new_beam = std::make_shared<Beam>(
            label, cost, true, point, timestamp, velocity, beam);
        new_beams.push_back(new_beam);
      }
    }
  }
  beams = std::move(new_beams);
}

void MarkerBeamSearch::prune_beams(int beam_width)
{
  std::sort(
      beams.begin(),
      beams.end(),
      [](const std::shared_ptr<Beam>& a, const std::shared_ptr<Beam>& b) {
        return a->cost < b->cost;
      });
  if (beams.size() > beam_width)
  {
    beams.resize(beam_width);
  }
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::string>
MarkerBeamSearch::convert_to_trace(std::shared_ptr<Beam> beam)
{
  std::vector<Eigen::Vector3d> points;
  std::vector<double> timestamps;
  std::map<std::string, int> label_count;

  while (beam != nullptr)
  {
    if (beam->observed_this_timestep)
    {
      points.push_back(beam->last_observed_point);
      timestamps.push_back(beam->last_observed_timestamp);
      label_count[beam->label]++;
    }
    beam = beam->parent;
  }

  // Find the label with the maximum count
  std::string max_vote_label;
  int max_votes = 0;
  for (auto it = label_count.begin(); it != label_count.end(); ++it)
  {
    const std::string& label = it->first;
    int count = it->second;
    if (count > max_votes)
    {
      max_votes = count;
      max_vote_label = label;
    }
  }

  // Reverse the points and timestamps to get them in chronological order
  std::reverse(points.begin(), points.end());
  std::reverse(timestamps.begin(), timestamps.end());

  return std::make_tuple(points, timestamps, max_vote_label);
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::string>
MarkerBeamSearch::search(
    const std::string& label,
    const std::vector<std::map<std::string, Eigen::Vector3d>>&
        marker_observations,
    const std::vector<double>& timestamps,
    int beam_width,
    double vel_threshold,
    double acc_threshold)
{
  // 1. Find the first observation of the label
  int first_observation_index = -1;
  for (size_t i = 0; i < marker_observations.size(); ++i)
  {
    if (marker_observations[i].find(label) != marker_observations[i].end())
    {
      first_observation_index = static_cast<int>(i);
      break;
    }
  }

  // 2. If the label is not observed, return an empty trace
  if (first_observation_index == -1)
  {
    return std::make_tuple(
        std::vector<Eigen::Vector3d>(), std::vector<double>(), label);
  }

  // 3. Start the beam search
  Eigen::Vector3d seed_point
      = marker_observations[first_observation_index].at(label);
  MarkerBeamSearch beam_search(
      seed_point,
      timestamps[first_observation_index],
      label,
      vel_threshold,
      acc_threshold);

  for (size_t i = first_observation_index + 1; i < marker_observations.size();
       ++i)
  {
    // if (i % 1000 == 0)
    // {
    //   std::cout << "Beam searching timestep: " << i << "/"
    //             << marker_observations.size()
    //             << ", num beams: " << beam_search.beams.size() << std::endl;
    // }
    beam_search.make_next_generation(marker_observations[i], timestamps[i]);
    beam_search.prune_beams(beam_width);
  }

  return convert_to_trace(beam_search.beams[0]);
}

} // namespace biomechanics
} // namespace dart