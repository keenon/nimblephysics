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

TEST(LABELLER, MAKE_TRACES_SINGLE_TRACE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  for (int i = 0; i < 500; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.001, 0));

    rawPoints.push_back(pointCloud);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 1);
}

TEST(LABELLER, MAKE_TRACES_MULTIPLE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  std::cout << "Points: " << std::endl;
  for (int i = 0; i < 50; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    if (i % 10 > 5)
    {
      std::cout << i << std::endl;
      pointCloud.push_back(Eigen::Vector3s(0, i * 0.001, 0));
    }

    rawPoints.push_back(pointCloud);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  std::cout << "Traces: " << std::endl;
  for (auto trace : traces)
  {
    std::cout << "Trace: " << std::endl;
    for (int t : trace.mTimes)
    {
      std::cout << t << std::endl;
    }
  }

  EXPECT_EQ(traces.size(), 5);
}