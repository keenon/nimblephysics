// MarkerMultiBeamSearch.cpp

#include "dart/biomechanics/MarkerMultiBeamSearch.hpp"

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace dart {
namespace biomechanics {

//==============================================================================
TraceHead::TraceHead(
    const std::string& label,
    bool observed_this_timestep,
    const Eigen::Vector3d& last_observed_point,
    double last_observed_timestamp,
    const Eigen::Vector3d& last_observed_velocity,
    Eigen::VectorXd distances_to_other_traces,
    std::shared_ptr<TraceHead> parent)
  : label(label),
    observed_this_timestep(observed_this_timestep),
    last_observed_point(last_observed_point),
    last_observed_timestamp(last_observed_timestamp),
    last_observed_velocity(last_observed_velocity),
    parent(parent)
{
  if (parent != nullptr)
  {
    if (parent->num_distance_samples > 0)
    {
      num_distance_samples = parent->num_distance_samples + 1;
      Eigen::VectorXd delta
          = distances_to_other_traces - parent->distances_to_other_traces_mean;
      distances_to_other_traces_m2
          = parent->distances_to_other_traces_m2
            + delta.cwiseProduct(
                distances_to_other_traces
                - parent->distances_to_other_traces_mean);
      distances_to_other_traces_mean = parent->distances_to_other_traces_mean
                                       + delta / (double)num_distance_samples;
    }
    else
    {
      num_distance_samples = 1;
      distances_to_other_traces_mean = distances_to_other_traces;
      distances_to_other_traces_m2
          = Eigen::VectorXd::Zero(distances_to_other_traces.size());
    }
  }
  else
  {
    num_distance_samples = 1;
    distances_to_other_traces_mean = distances_to_other_traces;
    distances_to_other_traces_m2
        = Eigen::VectorXd::Zero(distances_to_other_traces.size());
  }
}

//==============================================================================
TraceHead::TraceHead(
    const std::string& label,
    bool observed_this_timestep,
    const Eigen::Vector3d& last_observed_point,
    double last_observed_timestamp,
    const Eigen::Vector3d& last_observed_velocity,
    std::shared_ptr<TraceHead> parent)
  : label(label),
    observed_this_timestep(observed_this_timestep),
    last_observed_point(last_observed_point),
    last_observed_timestamp(last_observed_timestamp),
    last_observed_velocity(last_observed_velocity),
    parent(parent)
{
  if (parent != nullptr)
  {
    num_distance_samples = parent->num_distance_samples;
    distances_to_other_traces_mean = parent->distances_to_other_traces_mean;
    distances_to_other_traces_m2 = parent->distances_to_other_traces_m2;
  }
  else
  {
    num_distance_samples = 0;
    distances_to_other_traces_mean = Eigen::VectorXd::Zero(0);
    distances_to_other_traces_m2 = Eigen::VectorXd::Zero(0);
  }
}

//==============================================================================
MultiBeam::MultiBeam(
    double cost,
    const std::vector<std::shared_ptr<TraceHead>>& trace_heads,
    const std::set<std::string>& timestep_used_markers)
  : cost(cost),
    timestep_used_markers(timestep_used_markers),
    trace_heads(trace_heads)
{
}

//==============================================================================
std::vector<std::shared_ptr<TraceHead>> MultiBeam::get_child_trace_heads(
    const std::shared_ptr<TraceHead>& trace_head, int index) const
{
  std::vector<std::shared_ptr<TraceHead>> new_trace_heads = trace_heads;
  if (index >= 0 && index < static_cast<int>(new_trace_heads.size()))
  {
    new_trace_heads[index] = trace_head;
  }
  return new_trace_heads;
}

//==============================================================================
MarkerMultiBeamSearch::MarkerMultiBeamSearch(
    const std::vector<Eigen::Vector3d>& seed_points,
    const std::vector<std::string>& seed_labels,
    double seed_timestamp,
    double vel_threshold,
    double acc_threshold)
  : vel_threshold(vel_threshold), acc_threshold(acc_threshold)
{
  std::vector<std::shared_ptr<TraceHead>> trace_heads;
  for (size_t i = 0; i < seed_points.size(); ++i)
  {
    Eigen::Vector3d zero_velocity = Eigen::Vector3d::Zero();
    std::shared_ptr<TraceHead> trace_head = std::make_shared<TraceHead>(
        seed_labels[i],
        true,
        seed_points[i],
        seed_timestamp,
        zero_velocity,
        nullptr);
    trace_heads.push_back(trace_head);
  }
  beams.emplace_back(
      std::make_shared<MultiBeam>(0.0, trace_heads, std::set<std::string>()));
}

//==============================================================================
void MarkerMultiBeamSearch::make_next_generation(
    const std::map<std::string, Eigen::Vector3d>& markers,
    double timestamp,
    int trace_head_to_attach)
{
  std::vector<std::shared_ptr<MultiBeam>> new_beams;
  for (const auto& beam : beams)
  {
    const std::shared_ptr<TraceHead>& trace_head
        = beam->trace_heads[trace_head_to_attach];

    std::set<std::string> timestep_used_markers = beam->timestep_used_markers;
    if (trace_head_to_attach == 0)
    {
      timestep_used_markers.clear();
    }

    // Option 1: Skip adding a marker
    double skip_cost = beam->cost + vel_threshold + acc_threshold;
    std::shared_ptr<TraceHead> skip_trace_head = std::make_shared<TraceHead>(
        trace_head->label,
        false,
        trace_head->last_observed_point,
        trace_head->last_observed_timestamp,
        trace_head->last_observed_velocity,
        trace_head);
    std::vector<std::shared_ptr<TraceHead>> child_trace_heads
        = beam->get_child_trace_heads(skip_trace_head, trace_head_to_attach);
    auto new_beam_ptr = std::make_shared<MultiBeam>(
        skip_cost, child_trace_heads, timestep_used_markers);
    new_beams.push_back(new_beam_ptr);

    // Option 2: Add each possible marker
    for (const auto& marker : markers)
    {
      const std::string& label = marker.first;
      const Eigen::Vector3d& point = marker.second;

      if (timestep_used_markers.find(label) != timestep_used_markers.end())
      {
        continue;
      }

      double delta_time = timestamp - trace_head->last_observed_timestamp;
      Eigen::Vector3d velocity
          = (point - trace_head->last_observed_point) / delta_time;
      Eigen::Vector3d acc
          = (velocity - trace_head->last_observed_velocity) / delta_time;

      double vel_mag = velocity.norm();
      if (vel_mag < 2 * vel_threshold)
      {
        double acc_mag = acc.norm();
        double cost = beam->cost + vel_mag + acc_mag;

        Eigen::VectorXd distances_to_other_traces
            = Eigen::VectorXd::Zero(beam->trace_heads.size());
        for (size_t i = 0; i < beam->trace_heads.size(); ++i)
        {
          if (trace_head->num_distance_samples > 100
              && trace_head->distances_to_other_traces_mean[i] > 0.2)
          {
            distances_to_other_traces[i]
                = trace_head->distances_to_other_traces_mean[i];
          }
          else
          {
            distances_to_other_traces[i]
                = (beam->trace_heads[i]->last_observed_point - point).norm();
          }
        }
        if (trace_head->num_distance_samples > 100)
        {
          for (int i = 0; i < distances_to_other_traces.size(); ++i)
          {
            // Only penalize marker pairs that are closer than 20cm on mean
            if (trace_head->distances_to_other_traces_mean[i] > 0.2)
            {
              continue;
            }
            double stddev = std::sqrt(
                trace_head->distances_to_other_traces_m2[i]
                / (trace_head->num_distance_samples - 1));
            double error = std::abs(
                               distances_to_other_traces[i]
                               - trace_head->distances_to_other_traces_mean[i])
                           / stddev;
            if (error > 2.0)
            {
              cost += error * 10000.0;
            }
          }
        }

        std::shared_ptr<TraceHead> new_trace_head = std::make_shared<TraceHead>(
            label,
            true,
            point,
            timestamp,
            velocity,
            distances_to_other_traces,
            trace_head);
        std::set<std::string> new_timestep_used_markers = timestep_used_markers;
        new_timestep_used_markers.insert(label);

        std::vector<std::shared_ptr<TraceHead>> child_trace_heads
            = beam->get_child_trace_heads(new_trace_head, trace_head_to_attach);
        auto new_beam_ptr = std::make_shared<MultiBeam>(
            cost, child_trace_heads, new_timestep_used_markers);
        new_beams.push_back(new_beam_ptr);
      }
    }
  }
  // Keep the old beam references alive with non-recursive shared_ptrs
  past_beams.push_back(beams);

  beams = new_beams;
}

//==============================================================================
void MarkerMultiBeamSearch::prune_beams(int beam_width)
{
  std::sort(
      beams.begin(),
      beams.end(),
      [](const std::shared_ptr<MultiBeam>& a,
         const std::shared_ptr<MultiBeam>& b) { return a->cost < b->cost; });
  if (static_cast<int>(beams.size()) > beam_width)
  {
    beams.resize(beam_width);
  }
}

//==============================================================================
std::pair<
    std::vector<std::map<std::string, Eigen::Vector3d>>,
    std::vector<double>>
MarkerMultiBeamSearch::convert_to_traces(
    const std::shared_ptr<MultiBeam>& beam_ptr)
{
  std::map<double, std::map<std::string, Eigen::Vector3d>> observed_timesteps;

  for (const auto& trace_head : beam_ptr->trace_heads)
  {
    std::vector<Eigen::Vector3d> points;
    std::vector<double> timestamps;
    std::map<std::string, int> label_count;
    std::shared_ptr<TraceHead> current_trace_head = trace_head;
    std::string first_label = current_trace_head->label;
    while (current_trace_head != nullptr)
    {
      if (current_trace_head->observed_this_timestep)
      {
        points.push_back(current_trace_head->last_observed_point);
        timestamps.push_back(current_trace_head->last_observed_timestamp);
        label_count[current_trace_head->label]++;
        first_label = current_trace_head->label;
      }
      current_trace_head = current_trace_head->parent.lock();
    }
    // Determine the label with the most votes
    std::string max_vote_label = "";
    int max_votes = -1;
    for (const auto& lc : label_count)
    {
      if (lc.second > max_votes)
      {
        max_votes = lc.second;
        max_vote_label = lc.first;
      }
    }

    // std::cout << " -> Reconstructing starting label: " << first_label
    //           << " with max vote " << max_vote_label << " with "
    //           << points.size() << " observations." << std::endl;

    for (size_t p = 0; p < points.size(); ++p)
    {
      double t = timestamps[p];
      observed_timesteps[t][first_label] = points[p];
    }
  }

  std::vector<double> sorted_timestamps;
  for (const auto& ot : observed_timesteps)
  {
    sorted_timestamps.push_back(ot.first);
  }
  std::sort(sorted_timestamps.begin(), sorted_timestamps.end());

  std::vector<std::map<std::string, Eigen::Vector3d>> trace;
  for (double t : sorted_timestamps)
  {
    trace.push_back(observed_timesteps[t]);
  }

  return std::make_pair(trace, sorted_timestamps);
}

//==============================================================================
void MarkerMultiBeamSearch::crystallize_beams(bool include_last)
{
  // Convert the best beam to a set of marker traces
  std::pair<
      std::vector<std::map<std::string, Eigen::Vector3d>>,
      std::vector<double>>
      result = convert_to_traces(beams[0]);
  // Append the result to the marker_observations and timestamps
  marker_observations.insert(
      marker_observations.end(),
      result.first.begin(),
      result.first.end() - (include_last ? 0 : 1));
  timestamps.insert(
      timestamps.end(),
      result.second.begin(),
      result.second.end() - (include_last ? 0 : 1));
  // Clear the beams
  past_beams.clear();
  beams.resize(1);
}

//==============================================================================
std::pair<
    std::vector<std::map<std::string, Eigen::Vector3d>>,
    std::vector<double>>
MarkerMultiBeamSearch::search(
    const std::vector<std::string>& labels,
    const std::vector<std::map<std::string, Eigen::Vector3d>>&
        marker_observations,
    const std::vector<double>& timestamps,
    int beam_width,
    double vel_threshold,
    double acc_threshold,
    int print_interval,
    int crysatilize_interval)
{
  // 1. Find first observation of all labels
  int first_observation_index = -1;
  for (size_t i = 0; i < marker_observations.size(); ++i)
  {
    bool all_labels_present = true;
    for (const auto& label : labels)
    {
      if (marker_observations[i].find(label) == marker_observations[i].end())
      {
        all_labels_present = false;
        break;
      }
    }
    if (all_labels_present)
    {
      first_observation_index = static_cast<int>(i);
      break;
    }
  }

  // 2. If not found, return empty trace
  if (first_observation_index == -1)
  {
    return std::make_pair(
        std::vector<std::map<std::string, Eigen::Vector3d>>(),
        std::vector<double>());
  }

  // 3. Start the beam search
  std::vector<Eigen::Vector3d> seed_points;
  for (const auto& label : labels)
  {
    seed_points.push_back(
        marker_observations[first_observation_index].at(label));
  }
  MarkerMultiBeamSearch beam_search(
      seed_points,
      labels,
      timestamps[first_observation_index],
      vel_threshold,
      acc_threshold);
  beam_search.past_beams.reserve(marker_observations.size() * labels.size());

  for (size_t i = first_observation_index + 1; i < marker_observations.size();
       ++i)
  {
    if (i % print_interval == 0)
    {
      std::cout << "Beam searching timestep: " << i << "/"
                << marker_observations.size()
                << ", num beams: " << beam_search.beams.size() << std::endl;
    }
    // At each timestep, take one decision for each trace
    for (size_t j = 0; j < labels.size(); ++j)
    {
      beam_search.make_next_generation(
          marker_observations[i], timestamps[i], static_cast<int>(j));
      beam_search.prune_beams(beam_width);
    }

    if (i % crysatilize_interval == 0)
    {
      std::cout << "Crystallizing beams at timestep: " << i << "/"
                << marker_observations.size() << std::endl;
      beam_search.crystallize_beams(false);
    }
  }

  beam_search.crystallize_beams();
  return std::make_pair(
      beam_search.marker_observations, beam_search.timestamps);
}

} // namespace biomechanics
} // namespace dart
