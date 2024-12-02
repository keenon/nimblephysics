#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/math/AssignmentMatcher.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;

// #define ALL_TESTS

TEST(C3D, IDENTITY_MATCH)
{
  Eigen::MatrixXs weights = Eigen::MatrixXs::Identity(5, 5);
  Eigen::VectorXi map = math::AssignmentMatcher::assignRowsToColumns(weights);
  EXPECT_EQ(map(0), 0);
  EXPECT_EQ(map(1), 1);
  EXPECT_EQ(map(2), 2);
  EXPECT_EQ(map(3), 3);
  EXPECT_EQ(map(4), 4);
}

TEST(C3D, IDENTITY_MATCH_MORE_ROWS)
{
  Eigen::MatrixXs weights = Eigen::MatrixXs::Identity(5, 3);
  Eigen::VectorXi map = math::AssignmentMatcher::assignRowsToColumns(weights);
  EXPECT_EQ(map(0), 0);
  EXPECT_EQ(map(1), 1);
  EXPECT_EQ(map(2), 2);
  EXPECT_EQ(map(3), -1);
  EXPECT_EQ(map(4), -1);
}

TEST(C3D, IDENTITY_MATCH_MORE_COLS)
{
  Eigen::MatrixXs weights = Eigen::MatrixXs::Identity(3, 5);
  Eigen::VectorXi map = math::AssignmentMatcher::assignRowsToColumns(weights);
  EXPECT_EQ(map(0), 0);
  EXPECT_EQ(map(1), 1);
  EXPECT_EQ(map(2), 2);
}

TEST(C3D, TRANSPOSE_MATCH)
{
  Eigen::MatrixXs weights = Eigen::MatrixXs::Zero(5, 5);
  weights(4, 0) = 1.0;
  weights(3, 1) = 1.0;
  weights(2, 2) = 1.0;
  weights(1, 3) = 1.0;
  weights(0, 4) = 1.0;
  Eigen::VectorXi map = math::AssignmentMatcher::assignRowsToColumns(weights);
  EXPECT_EQ(map(0), 4);
  EXPECT_EQ(map(1), 3);
  EXPECT_EQ(map(2), 2);
  EXPECT_EQ(map(3), 1);
  EXPECT_EQ(map(4), 0);
}

TEST(C3D, MAPPING_STR)
{
  std::vector<std::string> source;
  std::vector<std::string> target;
  for (int i = 0; i < 5; i++)
  {
    source.push_back(std::to_string(i));
    target.push_back(std::to_string(i));
  }

  Eigen::MatrixXs weights = Eigen::MatrixXs::Random(5, 5);
  Eigen::VectorXi mapVec
      = math::AssignmentMatcher::assignRowsToColumns(weights);
  std::map<std::string, std::string> mapStr
      = math::AssignmentMatcher::assignKeysToKeys(
          source, target, [&](std::string s, std::string t) {
            return weights(std::atoi(s.c_str()), std::atoi(t.c_str()));
          });

  for (int i = 0; i < 5; i++)
  {
    std::string s = std::to_string(i);
    std::string t = std::to_string(mapVec[i]);

    EXPECT_EQ(mapStr[s], t);
  }
}

TEST(C3D, MAPPING_STR_MORE_SOURCE)
{
  std::vector<std::string> source;
  std::vector<std::string> target;
  for (int i = 0; i < 5; i++)
  {
    source.push_back(std::to_string(i));
  }
  for (int i = 0; i < 3; i++)
  {
    target.push_back(std::to_string(i));
  }

  Eigen::MatrixXs weights = Eigen::MatrixXs::Random(5, 3);
  Eigen::VectorXi mapVec
      = math::AssignmentMatcher::assignRowsToColumns(weights);
  std::map<std::string, std::string> mapStr
      = math::AssignmentMatcher::assignKeysToKeys(
          source, target, [&](std::string s, std::string t) {
            return weights(std::atoi(s.c_str()), std::atoi(t.c_str()));
          });

  EXPECT_EQ(mapStr.size(), 3);
  for (int i = 0; i < 3; i++)
  {
    std::string s = std::to_string(i);
    if (mapStr.count(s) > 0)
    {
      EXPECT_NE(mapVec[i], -1);
      std::string t = std::to_string(mapVec[i]);

      EXPECT_EQ(mapStr[s], t);
    }
    else
    {
      EXPECT_EQ(mapVec[i], -1);
    }
  }
}

TEST(C3D, MAPPING_STR_MORE_TARGET)
{
  std::vector<std::string> source;
  std::vector<std::string> target;
  for (int i = 0; i < 3; i++)
  {
    source.push_back(std::to_string(i));
  }
  for (int i = 0; i < 5; i++)
  {
    target.push_back(std::to_string(i));
  }

  Eigen::MatrixXs weights = Eigen::MatrixXs::Random(3, 5);
  Eigen::VectorXi mapVec
      = math::AssignmentMatcher::assignRowsToColumns(weights);
  std::map<std::string, std::string> mapStr
      = math::AssignmentMatcher::assignKeysToKeys(
          source, target, [&](std::string s, std::string t) {
            return weights(std::atoi(s.c_str()), std::atoi(t.c_str()));
          });

  EXPECT_EQ(mapStr.size(), 3);
  for (int i = 0; i < 3; i++)
  {
    std::string s = std::to_string(i);
    EXPECT_NE(mapVec[i], -1);
    EXPECT_TRUE(mapStr.count(s) > 0);
    std::string t = std::to_string(mapVec[i]);
    EXPECT_EQ(mapStr[s], t);
  }
}