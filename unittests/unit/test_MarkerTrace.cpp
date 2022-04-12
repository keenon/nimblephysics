#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/MarkerLabeller.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(LABELLER, MAKE_TRACES_SINGLE_TRACE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  for (int i = 0; i < 500; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));

    rawPoints.push_back(pointCloud);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 1);
}
#endif

#ifdef ALL_TESTS
TEST(LABELLER, MAKE_TRACES_MULTIPLE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  for (int i = 0; i < 50; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    if (i % 10 > 5)
    {
      pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));
    }

    rawPoints.push_back(pointCloud);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 5);
}
#endif

TEST(LABELLER, MAKE_TRACES_PARALLEL_TRACES)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  const int TIMESTEPS = 10;

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  for (int i = 0; i < TIMESTEPS; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 1.0));

    rawPoints.push_back(pointCloud);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 2);
  EXPECT_EQ(traces[0].mPoints.size(), TIMESTEPS);
  EXPECT_EQ(traces[1].mPoints.size(), TIMESTEPS);
}