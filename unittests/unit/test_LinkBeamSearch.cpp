#include <memory>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/biomechanics/LinkBeamSearch.hpp"

using namespace std;
using namespace dart;
using namespace biomechanics;

// Test case for LinkBeam
TEST(LinkBeamTest, ConstructorInitialization)
{
  Eigen::VectorXd point = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(3);
  auto parent = std::make_shared<LinkBeam>(
      0.5,
      "A",
      true,
      point,
      1.0,
      velocity,
      "B",
      true,
      point,
      1.0,
      velocity,
      nullptr);

  LinkBeam beam(
      0.75,
      "A",
      true,
      point,
      2.0,
      velocity,
      "B",
      false,
      point,
      2.0,
      velocity,
      parent);

  EXPECT_EQ(beam.cost, 0.75);
  EXPECT_EQ(beam.a_label, "A");
  EXPECT_EQ(beam.b_label, "B");
  EXPECT_TRUE(beam.a_observed_this_timestep);
  EXPECT_FALSE(beam.b_observed_this_timestep);
  EXPECT_EQ(beam.parent.lock(), parent);
}

// Test case for LinkBeamSearch constructor
TEST(LinkBeamSearchTest, ConstructorInitialization)
{
  Eigen::VectorXd seed_a_point = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd seed_b_point = Eigen::VectorXd::Zero(3);
  double seed_timestamp = 0.0;

  LinkBeamSearch beam_search(
      seed_a_point, "A", seed_b_point, "B", seed_timestamp, 1.0);

  ASSERT_EQ(beam_search.beams.size(), 1);
  EXPECT_EQ(beam_search.beams[0]->a_label, "A");
  EXPECT_EQ(beam_search.beams[0]->b_label, "B");
  EXPECT_EQ(beam_search.beams[0]->cost, 0.0);
}

// Test case for make_next_generation
TEST(LinkBeamSearchTest, MakeNextGeneration)
{
  Eigen::VectorXd seed_a_point(3);
  seed_a_point << 0.0, 0.0, 0.0;

  Eigen::VectorXd seed_b_point(3);
  seed_b_point << 1.0, 1.0, 1.0;

  double seed_timestamp = 0.0;
  LinkBeamSearch beam_search(
      seed_a_point, "A", seed_b_point, "B", seed_timestamp, 1.0);

  std::map<std::string, Eigen::VectorXd> markers;
  Eigen::VectorXd new_point(3);
  new_point << 0.5, 0.5, 0.5;
  markers["new"] = new_point;

  beam_search.make_next_generation(markers, 1.0);

  ASSERT_GT(beam_search.beams.size(), 0);
  for (const auto& beam : beam_search.beams)
  {
    EXPECT_TRUE(
        (beam->a_label == "A" && beam->b_label == "B")
        || (beam->a_label == "A" && beam->b_label == "new")
        || (beam->a_label == "new" && beam->b_label == "B"));
  }
}

// Test case for prune_beams
TEST(LinkBeamSearchTest, PruneBeams)
{
  Eigen::VectorXd seed_a_point = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd seed_b_point = Eigen::VectorXd::Zero(3);
  LinkBeamSearch beam_search(seed_a_point, "A", seed_b_point, "B", 0.0, 1.0);

  // Create multiple beams with different costs
  for (int i = 0; i < 10; ++i)
  {
    Eigen::VectorXd new_point = Eigen::VectorXd::Random(3);
    auto beam = std::make_shared<LinkBeam>(
        i,
        "A",
        true,
        new_point,
        1.0,
        Eigen::VectorXd::Zero(3),
        "B",
        true,
        new_point,
        1.0,
        Eigen::VectorXd::Zero(3),
        nullptr);
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

// Test case for convert_to_traces
TEST(LinkBeamSearchTest, ConvertToTraces)
{
  Eigen::VectorXd point = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(3);
  auto beam = std::make_shared<LinkBeam>(
      0.5,
      "A",
      true,
      point,
      1.0,
      velocity,
      "B",
      true,
      point,
      1.0,
      velocity,
      nullptr);

  auto result = LinkBeamSearch::convert_to_traces(beam);

  const auto& a_points = std::get<0>(result);
  const auto& a_timestamps = std::get<1>(result);
  const std::string& a_label = std::get<2>(result);
  const auto& b_points = std::get<3>(result);
  const auto& b_timestamps = std::get<4>(result);
  const std::string& b_label = std::get<5>(result);

  EXPECT_EQ(a_points.size(), 1);
  EXPECT_EQ(a_timestamps.size(), 1);
  EXPECT_EQ(a_label, "A");
  EXPECT_EQ(b_points.size(), 1);
  EXPECT_EQ(b_timestamps.size(), 1);
  EXPECT_EQ(b_label, "B");
}

// Test case for the full search method
TEST(LinkBeamSearchTest, FullSearch)
{
  std::vector<std::map<std::string, Eigen::VectorXd>> marker_observations;
  std::vector<double> timestamps = {0.0, 1.0, 2.0};

  Eigen::VectorXd point_a(3);
  point_a << 0.0, 0.0, 0.0;

  Eigen::VectorXd point_b(3);
  point_b << 1.0, 1.0, 1.0;

  std::map<std::string, Eigen::VectorXd> obs1
      = {{"A", point_a}, {"B", point_b}};
  std::map<std::string, Eigen::VectorXd> obs2
      = {{"A", point_a + Eigen::VectorXd::Ones(3)},
         {"B", point_b + Eigen::VectorXd::Ones(3)}};
  std::map<std::string, Eigen::VectorXd> obs3
      = {{"A", point_a + 2 * Eigen::VectorXd::Ones(3)},
         {"B", point_b + 2 * Eigen::VectorXd::Ones(3)}};

  marker_observations.push_back(obs1);
  marker_observations.push_back(obs2);
  marker_observations.push_back(obs3);

  auto result
      = LinkBeamSearch::search("A", "B", marker_observations, timestamps);

  const auto& a_points = std::get<0>(result);
  const auto& a_timestamps = std::get<1>(result);
  const std::string& a_label = std::get<2>(result);
  const auto& b_points = std::get<3>(result);
  const auto& b_timestamps = std::get<4>(result);
  const std::string& b_label = std::get<5>(result);

  EXPECT_EQ(a_points.size(), 3);
  EXPECT_EQ(a_timestamps.size(), 3);
  EXPECT_EQ(a_label, "A");
  EXPECT_EQ(b_points.size(), 3);
  EXPECT_EQ(b_timestamps.size(), 3);
  EXPECT_EQ(b_label, "B");
}
