// MarkerMultiBeamSearch.cpp

#include "dart/biomechanics/MarkerMultiBeamSearch.hpp"

#include <algorithm>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_set>

namespace dart {
namespace biomechanics {

//==============================================================================
TraceHead::TraceHead(
    const std::string& label,
    bool observed_this_timestep,
    const Eigen::Vector3d& last_observed_point,
    double last_observed_timestamp,
    int last_observed_index,
    const Eigen::Vector3d& last_observed_velocity,
    std::shared_ptr<TraceHead> parent)
  : label(label),
    observed_this_timestep(observed_this_timestep),
    last_observed_point(last_observed_point),
    last_observed_timestamp(last_observed_timestamp),
    last_observed_index(last_observed_index),
    last_observed_velocity(last_observed_velocity),
    parent(parent)
{
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
    int seed_index,
    Eigen::MatrixXd pairwise_distances,
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
    double acc_threshold)
  : pairwise_distances(pairwise_distances),
    pair_weight(pair_weight),
    pair_threshold(pair_threshold),
    vel_weight(vel_weight),
    vel_threshold(vel_threshold),
    acc_weight(acc_weight),
    acc_threshold(acc_threshold)
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
        seed_index,
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
    int index,
    int trace_head_to_attach,
    int beam_width)
{
  std::vector<std::shared_ptr<MultiBeam>> new_beams;
  for (const auto& beam : beams)
  {
    const std::shared_ptr<TraceHead>& trace_head
        = beam->trace_heads[trace_head_to_attach];
    const double delta_time = timestamp - trace_head->last_observed_timestamp;

    std::set<std::string> timestep_used_markers = beam->timestep_used_markers;
    if (trace_head_to_attach == 0)
    {
      timestep_used_markers.clear();
    }

    // Option 1: Skip adding a marker
    double skip_cost
        = beam->cost + (vel_threshold * vel_weight)
          + (acc_threshold * acc_weight)
          // Measure your distance to all previous trace_heads, but not yourself
          // or subsequent trace_heads that have not been chosen yet. Hence,
          // this is `trace_head_to_attach` penalties.
          + (pair_threshold * pair_weight * trace_head_to_attach);
    if (new_beams.size() < beam_width || skip_cost < new_beams.end()[-1]->cost)
    {
      std::shared_ptr<TraceHead> skip_trace_head = std::make_shared<TraceHead>(
          trace_head->label,
          false,
          trace_head->last_observed_point,
          trace_head->last_observed_timestamp,
          trace_head->last_observed_index,
          trace_head->last_observed_velocity,
          trace_head);
      std::vector<std::shared_ptr<TraceHead>> child_trace_heads
          = beam->get_child_trace_heads(skip_trace_head, trace_head_to_attach);
      auto new_beam_ptr = std::make_shared<MultiBeam>(
          skip_cost, child_trace_heads, timestep_used_markers);
      new_beams.push_back(new_beam_ptr);
      std::sort(
          new_beams.begin(),
          new_beams.end(),
          [](const std::shared_ptr<MultiBeam>& a,
             const std::shared_ptr<MultiBeam>& b) {
            return a->cost < b->cost;
          });
      if (new_beams.size() > beam_width)
      {
        new_beams.resize(beam_width);
      }
    }

    // Option 2: Add each possible marker
    for (const auto& marker : markers)
    {
      const std::string& label = marker.first;
      const Eigen::Vector3d& point = marker.second;

      if (timestep_used_markers.find(label) != timestep_used_markers.end())
      {
        continue;
      }

      Eigen::Vector3d velocity
          = (point - trace_head->last_observed_point) / delta_time;
      Eigen::Vector3d acc
          = (velocity - trace_head->last_observed_velocity) / delta_time;

      double vel_mag = velocity.norm();
      double acc_mag = acc.norm();
      double cost
          = beam->cost + (vel_mag * vel_weight) + (acc_mag * acc_weight);
      if (new_beams.size() == beam_width && cost > new_beams.end()[-1]->cost)
      {
        continue;
      }

      // Compare our distances to all previous trace_heads that have already
      // (potentially) been attached.
      for (size_t i = 0; i < trace_head_to_attach; ++i)
      {
        // If this trace head was attached this frame, take a penalty on the
        // observed distance
        if (beam->trace_heads[i]->last_observed_index == index)
        {
          double distance
              = (beam->trace_heads[i]->last_observed_point - point).norm();
          cost += pair_weight
                  * std::abs(
                      pairwise_distances(i, trace_head_to_attach) - distance);
        }
        else
        {
          // Otherwise just take a penalty on the threshold distance
          cost += pair_threshold * pair_weight;
        }
      }
      if (new_beams.size() == beam_width && cost > new_beams.end()[-1]->cost)
      {
        continue;
      }

      std::shared_ptr<TraceHead> new_trace_head = std::make_shared<TraceHead>(
          label, true, point, timestamp, index, velocity, trace_head);
      std::set<std::string> new_timestep_used_markers = timestep_used_markers;
      new_timestep_used_markers.insert(label);

      std::vector<std::shared_ptr<TraceHead>> child_trace_heads
          = beam->get_child_trace_heads(new_trace_head, trace_head_to_attach);
      new_beams.emplace_back(std::make_shared<MultiBeam>(
          cost, child_trace_heads, new_timestep_used_markers));
      std::sort(
          new_beams.begin(),
          new_beams.end(),
          [](const std::shared_ptr<MultiBeam>& a,
             const std::shared_ptr<MultiBeam>& b) {
            return a->cost < b->cost;
          });
      if (new_beams.size() > beam_width)
      {
        new_beams.resize(beam_width);
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
double MarkerMultiBeamSearch::get_median_70_percent_mean_distance(
    std::string a_label,
    std::string b_label,
    const std::vector<std::map<std::string, Eigen::Vector3d>>&
        marker_observations)
{
  // Figure out the distance between the markers
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

  if (closest_70_percent.size() == 0)
  {
    return 0.0;
  }

  // Calculate the mean of the 70% of data closest to the median
  double mean_70
      = std::accumulate(
            closest_70_percent.begin(), closest_70_percent.end(), 0.0)
        / closest_70_percent.size();

  return mean_70;
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
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
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
    std::cout
        << "Could not find first observation of all labels on the same frame"
        << std::endl;
    std::map<std::string, int> label_counts;
    for (size_t i = 0; i < marker_observations.size(); ++i)
    {
      for (const auto& label : labels)
      {
        if (marker_observations[i].find(label) != marker_observations[i].end())
        {
          label_counts[label]++;
        }
      }
    }
    for (const auto& lc : label_counts)
    {
      std::cout << "Label " << lc.first << " found " << lc.second << " times"
                << std::endl;
    }
    return std::make_pair(
        std::vector<std::map<std::string, Eigen::Vector3d>>(),
        std::vector<double>());
  }

  // 3. Get the starting points for the beam search
  std::vector<Eigen::Vector3d> seed_points;
  for (const auto& label : labels)
  {
    seed_points.push_back(
        marker_observations[first_observation_index].at(label));
  }

  // 4. Collect all the pairwise distance statistics for the given labels
  Eigen::MatrixXd pairwise_distances(labels.size(), labels.size());
  for (size_t i = 0; i < labels.size(); ++i)
  {
    for (size_t j = i + 1; j < labels.size(); ++j)
    {
      double first_timestep_distance = (seed_points[i] - seed_points[j]).norm();
      double median_distance = get_median_70_percent_mean_distance(
          labels[i], labels[j], marker_observations);
      std::cout << "Distance between " << labels[i] << " and " << labels[j]
                << " is " << first_timestep_distance << ", with median "
                << median_distance << std::endl;
      pairwise_distances(i, j) = first_timestep_distance;
      pairwise_distances(j, i) = first_timestep_distance;
    }
  }

  MarkerMultiBeamSearch beam_search(
      seed_points,
      labels,
      timestamps[first_observation_index],
      first_observation_index,
      pairwise_distances,
      pair_weight,
      pair_threshold,
      vel_weight,
      vel_threshold,
      acc_weight,
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
          marker_observations[i],
          timestamps[i],
          i,
          static_cast<int>(j),
          beam_width);
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

std::tuple<
    std::vector<std::map<std::string, Eigen::Vector3d>>,
    std::vector<double>>
MarkerMultiBeamSearch::process_markers(
    const std::vector<std::vector<std::string>>& label_groups,
    const std::vector<std::map<std::string, Eigen::Vector3d>>&
        marker_observations,
    const std::vector<double>& timestamps,
    size_t beam_width,
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
    double acc_threshold,
    int print_interval,
    int crysatilize_interval,
    bool multithread)
{
  std::map<std::string, int> marker_observation_counts;
  for (const auto& marker_observation : marker_observations)
  {
    for (const auto& marker : marker_observation)
    {
      marker_observation_counts[marker.first]++;
    }
  }

  std::cout << "Building marker traces for " << label_groups.size() << " groups"
            << std::endl;
  std::vector<std::vector<std::string>> filtered_groups;
  for (int i = 0; i < label_groups.size(); ++i)
  {
    std::cout << "Group " << i << ":" << std::endl;

    std::vector<std::string> filtered_group;
    std::vector<std::string> rejected_group;

    for (const auto& label : label_groups[i])
    {
      if (marker_observation_counts[label] > 0)
      {
        filtered_group.push_back(label);
      }
      else
      {
        rejected_group.push_back(label);
      }
    }
    if (filtered_group.size() > 0)
    {
      filtered_groups.push_back(filtered_group);
      std::cout << "  Found Markers: ";
      for (const auto& label : filtered_group)
      {
        std::cout << label << " ";
      }
      std::cout << std::endl;
    }
    if (rejected_group.size() > 0)
    {
      std::cout << "  Did Not Find Markers: ";
      for (const auto& label : rejected_group)
      {
        std::cout << label << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::map<double, std::map<std::string, Eigen::Vector3d>> trace_output_map;

  if (multithread)
  {
    std::mutex mutex;
    std::vector<std::future<void>> futures;
    for (const std::vector<std::string>& label_group : filtered_groups)
    {
      futures.push_back(std::async([&] {
        auto result = MarkerMultiBeamSearch::search(
            label_group,
            marker_observations,
            timestamps,
            beam_width,
            pair_weight,
            pair_threshold,
            vel_weight,
            vel_threshold,
            acc_weight,
            acc_threshold,
            print_interval,
            crysatilize_interval);

        {
          std::lock_guard<std::mutex> lock(mutex);
          std::vector<std::map<std::string, Eigen::Vector3d>> outputTraces
              = std::get<0>(result);
          std::vector<double> outputTimesteps = std::get<1>(result);

          for (size_t i = 0; i < outputTraces.size(); ++i)
          {
            for (const auto& label_point : outputTraces[i])
            {
              trace_output_map[outputTimesteps[i]][label_point.first]
                  = label_point.second;
            }
          }
        }
      }));
    }

    for (auto& future : futures)
    {
      future.wait();
    }
  }
  else
  {
    for (int g = 0; g < filtered_groups.size(); g++)
    {
      std::cout << "Processing group " << g << "/" << filtered_groups.size()
                << ": ";
      const std::vector<std::string>& label_group = filtered_groups.at(g);
      for (const auto& label : label_group)
      {
        std::cout << label << " ";
      }
      std::cout << std::endl;

      auto result = MarkerMultiBeamSearch::search(
          label_group,
          marker_observations,
          timestamps,
          beam_width,
          pair_weight,
          pair_threshold,
          vel_weight,
          vel_threshold,
          acc_weight,
          acc_threshold,
          print_interval,
          crysatilize_interval);
      std::vector<std::map<std::string, Eigen::Vector3d>> outputTraces
          = std::get<0>(result);
      std::vector<double> outputTimesteps = std::get<1>(result);

      for (size_t i = 0; i < outputTraces.size(); ++i)
      {
        for (const auto& label_point : outputTraces[i])
        {
          trace_output_map[outputTimesteps[i]][label_point.first]
              = label_point.second;
        }
      }
    }
  }

  std::cout << "Finished building marker traces" << std::endl;

  std::vector<std::map<std::string, Eigen::Vector3d>> traces;
  std::vector<double> timesteps;
  for (const auto& trace : trace_output_map)
  {
    traces.push_back(trace.second);
    timesteps.push_back(trace.first);
  }

  return std::make_tuple(traces, timesteps);
}

} // namespace biomechanics
} // namespace dart
