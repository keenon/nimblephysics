#include <memory>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/biomechanics/MarkerMultiBeamSearch.hpp"

using namespace std;
using namespace dart;
using namespace biomechanics;

#define ALL_TESTS

// Test case for MarkerMultiBeamSearch constructor
#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, ConstructorInitialization)
{
  std::vector<Eigen::Vector3d> seed_points
      = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::vector<std::string> seed_labels = {"A", "B"};
  double seed_timestamp = 0.0;
  int seed_index = 0;
  Eigen::MatrixXd pairwise_distances = Eigen::MatrixXd::Zero(2, 2);

  MarkerMultiBeamSearch beam_search(
      seed_points, seed_labels, seed_timestamp, seed_index, pairwise_distances);

  ASSERT_EQ(beam_search.beams.size(), 1);
  EXPECT_EQ(beam_search.beams[0]->trace_heads[0]->label, "A");
  EXPECT_EQ(beam_search.beams[0]->trace_heads[1]->label, "B");
  EXPECT_EQ(beam_search.beams[0]->cost, 0.0);
}
#endif

// Test case for make_next_generation
#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, MakeNextGeneration)
{
  std::vector<Eigen::Vector3d> seed_points
      = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()};
  std::vector<std::string> seed_labels = {"A", "B"};
  double seed_timestamp = 0.0;
  int seed_index = 0;
  Eigen::MatrixXd pairwise_distances = Eigen::MatrixXd::Zero(2, 2);

  MarkerMultiBeamSearch beam_search(
      seed_points, seed_labels, seed_timestamp, seed_index, pairwise_distances);

  std::map<std::string, Eigen::Vector3d> markers;
  markers["new"] = 0.5 * Eigen::Vector3d::Ones();

  beam_search.make_next_generation(markers, 1.0, 1, 0, 10);
  beam_search.make_next_generation(markers, 1.0, 1, 1, 10);

  ASSERT_GT(beam_search.beams.size(), 0);
  for (const auto& beam : beam_search.beams)
  {
    EXPECT_TRUE(
        (beam->trace_heads[0]->label == "A"
         && beam->trace_heads[1]->label == "B")
        || (beam->trace_heads[0]->label == "A"
            && beam->trace_heads[1]->label == "new")
        || (beam->trace_heads[0]->label == "new"
            && beam->trace_heads[1]->label == "B"));
  }
}
#endif

// Test case for prune_beams
#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, PruneBeams)
{
  std::vector<Eigen::Vector3d> seed_points
      = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::vector<std::string> seed_labels = {"A", "B"};
  double seed_timestamp = 0.0;
  int seed_index = 0;
  Eigen::MatrixXd pairwise_distances = Eigen::MatrixXd::Zero(2, 2);

  MarkerMultiBeamSearch beam_search(
      seed_points, seed_labels, seed_timestamp, seed_index, pairwise_distances);

  // Create multiple beams with different costs
  for (int i = 0; i < 10; ++i)
  {
    auto trace_head = std::make_shared<TraceHead>(
        "A",
        true,
        Eigen::Vector3d::Random(),
        seed_timestamp,
        seed_index,
        Eigen::Vector3d::Zero());
    auto beam = std::make_shared<MultiBeam>(
        i,
        std::vector<std::shared_ptr<TraceHead>>{trace_head},
        std::set<std::string>());
    beam_search.beams.push_back(beam);
  }

  // Prune to 5 beams
  beam_search.prune_beams(5);

  EXPECT_EQ(beam_search.beams.size(), 5);

  // Ensure beams are sorted by cost
  for (size_t i = 0; i < beam_search.beams.size() - 1; ++i)
  {
    EXPECT_LE(beam_search.beams[i]->cost, beam_search.beams[i + 1]->cost);
  }
}
#endif

// Test case for convert_to_traces
#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, ConvertToTraces)
{
  auto trace_head = std::make_shared<TraceHead>(
      "A", true, Eigen::Vector3d::Zero(), 0.0, 0, Eigen::Vector3d::Zero());

  auto beam = std::make_shared<MultiBeam>(
      0.5,
      std::vector<std::shared_ptr<TraceHead>>{trace_head},
      std::set<std::string>());

  auto result = MarkerMultiBeamSearch::convert_to_traces(beam);

  const auto& marker_observations = result.first;
  const auto& timestamps = result.second;

  EXPECT_EQ(marker_observations.size(), 1);
  EXPECT_EQ(timestamps.size(), 1);
}
#endif

// Test case for the full search method
#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, FullSearch)
{
  std::vector<std::map<std::string, Eigen::Vector3d>> marker_observations
      = {{{"A", Eigen::Vector3d::Zero()}, {"B", Eigen::Vector3d::Ones()}},
         {{"A", Eigen::Vector3d::Ones()}, {"B", 2 * Eigen::Vector3d::Ones()}}};

  std::vector<double> timestamps = {0.0, 1.0};

  auto result = MarkerMultiBeamSearch::search(
      {"A", "B"}, marker_observations, timestamps);

  const auto& marker_traces = result.first;
  const auto& trace_timestamps = result.second;

  EXPECT_EQ(marker_traces.size(), 2);
  EXPECT_EQ(trace_timestamps.size(), 2);
}
#endif

#ifdef ALL_TESTS
TEST(MarkerMultiBeamSearchTest, SearchWithMissingLabels)
{
  std::vector<std::map<std::string, Eigen::Vector3d>> marker_observations
      = {{{"A", Eigen::Vector3d::Zero()}, {"B", Eigen::Vector3d::Ones()}},
         {{"A", Eigen::Vector3d::Ones()}}};

  std::vector<double> timestamps = {0.0, 1.0};

  auto result = MarkerMultiBeamSearch::search(
      {"A", "B"}, marker_observations, timestamps);

  const auto& marker_traces = result.first;
  const auto& trace_timestamps = result.second;

  EXPECT_EQ(marker_traces.size(), 2);
  EXPECT_EQ(trace_timestamps.size(), 2);
}
#endif