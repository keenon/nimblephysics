#include "dart/biomechanics/LinkBeamSearch.hpp"

#include <algorithm>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

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
    const std::map<std::string, Eigen::VectorXd>& markers,
    double timestamp,
    size_t beam_width)
{
  // Store the start time so that we can keep track of where the performance is
  // going
  auto start_pair_distances_time = std::chrono::high_resolution_clock::now();

  // Precompute the distances between all pairs of markers
  Eigen::MatrixXd markerPairDistances
      = Eigen::MatrixXd::Zero(markers.size(), markers.size());
  std::vector<std::string> markerLabels;
  for (const auto& marker_key_value : markers)
  {
    markerLabels.push_back(marker_key_value.first);
  }
  for (size_t i = 0; i < markers.size(); ++i)
  {
    const Eigen::Vector3d& marker_a = markers.at(markerLabels.at(i));
    for (size_t j = i + 1; j < markers.size(); ++j)
    {
      markerPairDistances(i, j)
          = (marker_a - markers.at(markerLabels.at(j))).norm();
      markerPairDistances(j, i) = markerPairDistances(i, j);
    }
  }

  // Measure the duration of the precomputation
  auto end_pair_distances = std::chrono::high_resolution_clock::now();
  pair_distances_cost += end_pair_distances - start_pair_distances_time;

  std::vector<std::shared_ptr<LinkBeam>> new_beams;
  for (const auto& beam : beams)
  {
    // Store the start time so that we can keep track of where the performance
    // is going
    auto start_create_options_time = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, double>> a_point_options;
    std::vector<std::pair<std::string, double>> b_point_options;

    // Initialize options with None equivalent
    a_point_options.emplace_back(
        "", vel_threshold * vel_weight + acc_threshold * acc_weight);
    b_point_options.emplace_back(
        "", vel_threshold * vel_weight + acc_threshold * acc_weight);

    // 1. Consider only points with appropriate distance to other points
    std::map<std::string, bool> points_valid;
    for (int i = 0; i < markerLabels.size(); i++)
    {
      for (int j = i + 1; j < markerLabels.size(); j++)
      {
        double dist = markerPairDistances(i, j);
        double dist_diff = std::abs(dist - pair_dist);
        if (dist_diff < pair_threshold * 2)
        {
          points_valid[markerLabels.at(i)] = true;
          points_valid[markerLabels.at(j)] = true;
        }
      }
    }

    // 2. Find the options for the next point for each marker individually
    for (const auto& marker_key_value : markers)
    {
      const std::string& label = marker_key_value.first;
      const Eigen::VectorXd& point = marker_key_value.second;

      bool point_valid = points_valid.find(label) != points_valid.end();

      // For 'a' marker
      Eigen::VectorXd a_velocity
          = (point - beam->a_last_observed_point)
            / (timestamp - beam->a_last_observed_timestamp);
      double a_vel_mag = a_velocity.norm();

      if (point_valid || (a_vel_mag < vel_threshold * 2))
      {
        Eigen::VectorXd a_acc = (a_velocity - beam->a_last_observed_velocity)
                                / (timestamp - beam->a_last_observed_timestamp);
        double a_acc_mag = a_acc.norm();
        double a_cost = a_vel_mag * vel_weight + a_acc_mag * acc_weight;
        a_point_options.emplace_back(label, a_cost);
      }

      // For 'b' marker
      Eigen::VectorXd b_velocity
          = (point - beam->b_last_observed_point)
            / (timestamp - beam->b_last_observed_timestamp);
      double b_vel_mag = b_velocity.norm();

      if (point_valid || (b_vel_mag < vel_threshold * 2))
      {
        Eigen::VectorXd b_acc = (b_velocity - beam->b_last_observed_velocity)
                                / (timestamp - beam->b_last_observed_timestamp);
        double b_acc_mag = b_acc.norm();
        double b_cost = b_vel_mag * vel_weight + b_acc_mag * acc_weight;
        b_point_options.emplace_back(label, b_cost);
      }
    }

    // Measure the duration of the options creation
    auto end_create_options = std::chrono::high_resolution_clock::now();
    create_options_cost += end_create_options - start_create_options_time;

    // Store the start time so that we can keep track of where the performance
    // is going
    auto start_create_beams_time = std::chrono::high_resolution_clock::now();

    new_beams.reserve(beam_width + 1);

    // 3. Create new beams for each pair of options
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
        double total_cost = beam->cost + a_cost + b_cost + pair_cost;

        if (new_beams.size() == 0 || total_cost < new_beams.end()[-1]->cost)
        {
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
          std::sort(
              new_beams.begin(),
              new_beams.end(),
              [](const std::shared_ptr<LinkBeam>& a,
                 const std::shared_ptr<LinkBeam>& b) {
                return a->cost < b->cost;
              });
          if (new_beams.size() > beam_width)
          {
            new_beams.pop_back();
          }
        }
      }
    }

    // Measure the duration of the beam creation
    auto end_create_beams = std::chrono::high_resolution_clock::now();
    create_beams_cost += end_create_beams - start_create_beams_time;
  }
  old_beams.push_back(beams);
  beams = new_beams;
}

void LinkBeamSearch::prune_beams(size_t beam_width)
{
  // Store the start time so that we can keep track of where the performance is
  // going
  auto start_prune_beams_time = std::chrono::high_resolution_clock::now();

  std::sort(
      beams.begin(),
      beams.end(),
      [](const std::shared_ptr<LinkBeam>& a,
         const std::shared_ptr<LinkBeam>& b) { return a->cost < b->cost; });
  if (beams.size() > beam_width)
  {
    beams.resize(beam_width);
  }

  // Measure the duration of the beam pruning
  auto end_prune_beams = std::chrono::high_resolution_clock::now();
  prune_beams_cost += end_prune_beams - start_prune_beams_time;
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
    double acc_threshold,
    bool print_updates)
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
    if (i % 1000 == 0 && print_updates)
    {
      std::cout << "Beam searching timestep: " << i << "/"
                << marker_observations.size()
                << ", num beams: " << beam_search.beams.size() << std::endl;
    }
    beam_search.make_next_generation(
        marker_observations[i], timestamps[i], beam_width);
    // beam_search.prune_beams(beam_width);
  }

  std::cout << "Performance report:" << std::endl;
  std::cout << "  Pair distances cost: "
            << beam_search.pair_distances_cost.count() << "s" << std::endl;
  std::cout << "  Create options cost: "
            << beam_search.create_options_cost.count() << "s" << std::endl;
  std::cout << "  Create beams cost: " << beam_search.create_beams_cost.count()
            << "s" << std::endl;
  std::cout << "  Prune beams cost: " << beam_search.prune_beams_cost.count()
            << "s" << std::endl;

  return LinkBeamSearch::convert_to_traces(beam_search.beams.front());
}

std::tuple<
    std::vector<std::map<std::string, Eigen::VectorXd>>,
    std::vector<double>>
LinkBeamSearch::process_markers(
    const std::vector<std::pair<std::string, std::string>>& label_pairs,
    const std::vector<std::map<std::string, Eigen::VectorXd>>&
        marker_observations,
    const std::vector<double>& timestamps,
    size_t beam_width,
    double pair_weight,
    double pair_threshold,
    double vel_weight,
    double vel_threshold,
    double acc_weight,
    double acc_threshold,
    bool print_updates,
    bool multithread)
{
  // This is a map from timestep -> label -> label pair's guess -> point
  std::
      map<double, std::map<std::string, std::map<std::string, Eigen::VectorXd>>>
          trace_votes;

  if (multithread)
  {
    std::mutex mutex;
    std::vector<std::future<void>> futures;
    for (const auto& label_pair : label_pairs)
    {
      futures.push_back(std::async([&] {
        const std::string& a_label = label_pair.first;
        const std::string& b_label = label_pair.second;
        const std::string pair_name = a_label + " - " + b_label;

        auto result = LinkBeamSearch::search(
            a_label,
            b_label,
            marker_observations,
            timestamps,
            beam_width,
            pair_weight,
            pair_threshold,
            vel_weight,
            vel_threshold,
            acc_weight,
            acc_threshold,
            print_updates);

        {
          std::lock_guard<std::mutex> lock(mutex);
          const auto& a_points = std::get<0>(result);
          const auto& a_timestamps = std::get<1>(result);
          for (size_t i = 0; i < a_points.size(); ++i)
          {
            trace_votes[a_timestamps[i]][a_label][pair_name] = a_points[i];
          }

          const auto& b_points = std::get<3>(result);
          const auto& b_timestamps = std::get<4>(result);
          for (size_t i = 0; i < b_points.size(); ++i)
          {
            trace_votes[b_timestamps[i]][b_label][pair_name] = b_points[i];
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
    for (const auto& label_pair : label_pairs)
    {
      const std::string& a_label = label_pair.first;
      const std::string& b_label = label_pair.second;
      const std::string pair_name = a_label + " - " + b_label;
      std::cout << "Processing label pair: " << pair_name << std::endl;

      auto result = LinkBeamSearch::search(
          a_label,
          b_label,
          marker_observations,
          timestamps,
          beam_width,
          pair_weight,
          pair_threshold,
          vel_weight,
          vel_threshold,
          acc_weight,
          acc_threshold,
          print_updates);

      const auto& a_points = std::get<0>(result);
      const auto& a_timestamps = std::get<1>(result);
      for (size_t i = 0; i < a_points.size(); ++i)
      {
        trace_votes[a_timestamps[i]][a_label][pair_name] = a_points[i];
      }

      const auto& b_points = std::get<3>(result);
      const auto& b_timestamps = std::get<4>(result);
      for (size_t i = 0; i < b_points.size(); ++i)
      {
        trace_votes[b_timestamps[i]][b_label][pair_name] = b_points[i];
      }
    }
  }

  std::cout << "Finished processing all label pairs" << std::endl;
  std::cout << "Counting trace agreements..." << std::endl;

  // Count trace agreement, which is a measure of credibility for that trace
  std::map<std::string, int> trace_agreements;
  for (const auto& marker_timestep : trace_votes)
  {
    for (const auto& marker : marker_timestep.second)
    {
      const auto& label_votes = marker.second;
      for (const auto& label_vote_1 : label_votes)
      {
        for (const auto& label_vote_2 : label_votes)
        {
          if (label_vote_1.first == label_vote_2.first)
            continue;

          if (label_vote_1.second.isApprox(label_vote_2.second, 0.0001))
          {
            trace_agreements[label_vote_1.first]++;
            trace_agreements[label_vote_2.first]++;
          }
        }
      }
    }
  }

  std::cout << "Pair agreement counts:" << std::endl;
  for (const auto& agreement : trace_agreements)
  {
    std::cout << "  " << agreement.first << ": " << agreement.second
              << std::endl;
  }

  std::cout << "Building marker traces from pair votes with the highest "
               "agreement count..."
            << std::endl;
  std::vector<double> trace_timestamps;
  std::vector<std::map<std::string, Eigen::VectorXd>> traces_vector;
  for (const auto& trace : trace_votes)
  {
    trace_timestamps.push_back(trace.first);
    std::map<std::string, Eigen::VectorXd> trace_map;
    for (const auto& label_votes : trace.second)
    {
      const auto& label = label_votes.first;
      const auto& votes = label_votes.second;
      const auto& max_vote = std::max_element(
          votes.begin(),
          votes.end(),
          [&trace_agreements](
              const std::pair<std::string, Eigen::VectorXd>& a,
              const std::pair<std::string, Eigen::VectorXd>& b) {
            return trace_agreements.at(a.first) > trace_agreements.at(b.first);
          });
      trace_map[label] = max_vote->second;
    }
    traces_vector.push_back(trace_map);
  }

  std::cout << "Finished building marker traces" << std::endl;

  return std::make_tuple(traces_vector, trace_timestamps);
}

} // namespace biomechanics
} // namespace dart