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

#define ALL_TESTS

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

#ifdef ALL_TESTS
TEST(LABELLER, TRACE_VARIANCE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  const int TIMESTEPS = 10;
  std::vector<std::vector<Eigen::Vector3s>> rawPoints;
  for (int i = 0; i < TIMESTEPS; i++)
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

  for (int i = 0; i < traces[0].mPoints.size() - 1; i++)
  {
    EXPECT_EQ(traces[0].mPoints[i][2], traces[0].mPoints[i][2]);
  }
  for (int i = 0; i < traces[1].mPoints.size() - 1; i++)
  {
    EXPECT_EQ(traces[1].mPoints[i][2], traces[1].mPoints[i][2]);
  }
  EXPECT_EQ(traces[0].mMinTime, traces[1].mMinTime);
  EXPECT_EQ(traces[0].mMaxTime, traces[1].mMaxTime);
}
#endif

#ifdef ALL_TESTS
TEST(LABELLER, COMPUTE_JOINT_FINGERPRINTS)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  const int TIMESTEPS = 10;

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;

  std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime;
  std::map<std::string, std::string> jointParents;

  jointParents["CHILD"] = "PARENT";

  for (int i = 0; i < TIMESTEPS; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));

    rawPoints.push_back(pointCloud);

    std::map<std::string, Eigen::Vector3s> jointCenters;
    jointCenters["PARENT"] = Eigen::Vector3s(1.0, i * 0.0005, 0);
    jointCenters["CHILD"] = Eigen::Vector3s(0.0, i * 0.0005, 0);
    jointsOverTime.push_back(jointCenters);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 1);
  EXPECT_EQ(traces[0].mPoints.size(), TIMESTEPS);

  traces[0].computeJointFingerprints(jointsOverTime, jointParents);

  EXPECT_EQ(traces[0].mJointFingerprints.size(), traces[0].mPoints.size());
  for (int i = 0; i < TIMESTEPS; i++)
  {
    // Along axis distance
    EXPECT_EQ(traces[0].mJointFingerprints[i].at("PARENT::CHILD")(0), 1.0);
    // Off axis distance
    EXPECT_EQ(traces[0].mJointFingerprints[i].at("PARENT::CHILD")(1), 0.0);
  }
}
#endif

#ifdef ALL_TESTS
TEST(LABELLER, CONCAT_WITH_FINGERPRINTS)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  const int TIMESTEPS = 30;

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;

  std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime;
  std::map<std::string, std::string> jointParents;

  jointParents["CHILD"] = "PARENT";

  for (int i = 0; i < TIMESTEPS; i++)
  {
    if (i > 10 && i < 20)
    {
      continue;
    }

    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));

    rawPoints.push_back(pointCloud);

    std::map<std::string, Eigen::Vector3s> jointCenters;
    jointCenters["PARENT"] = Eigen::Vector3s(1.0, i * 0.0005, 0);
    jointCenters["CHILD"] = Eigen::Vector3s(0.0, i * 0.0005, 0);
    jointsOverTime.push_back(jointCenters);
  }

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 2);
  traces[0].computeJointFingerprints(jointsOverTime, jointParents);
  traces[1].computeJointFingerprints(jointsOverTime, jointParents);

  biomechanics::MarkerTrace merged = traces[0].concat(traces[1]);

  EXPECT_NE(traces[1].mMinTime, traces[0].mMinTime);
  EXPECT_NE(traces[1].mMaxTime, traces[0].mMaxTime);

  EXPECT_EQ(merged.mMinTime, traces[0].mMinTime);
  EXPECT_EQ(merged.mMaxTime, traces[1].mMaxTime);

  EXPECT_NE(traces[1].mMinTime, traces[0].mMinTime);
  EXPECT_NE(traces[1].mMaxTime, traces[0].mMaxTime);
}
#endif

#ifdef ALL_TESTS
TEST(LABELLER, COMPUTE_FINGERPRINTS_VARIANCE)
{
  biomechanics::MarkerLabellerMock labeller
      = biomechanics::MarkerLabellerMock();

  const int TIMESTEPS = 10;

  std::vector<std::vector<Eigen::Vector3s>> rawPoints;

  std::vector<std::map<std::string, Eigen::Vector3s>> jointsOverTime;
  std::map<std::string, std::string> jointParents;

  jointParents["CHILD"] = "PARENT";

  s_t avgX = 0.0;
  for (int i = 0; i < TIMESTEPS; i++)
  {
    std::vector<Eigen::Vector3s> pointCloud;
    pointCloud.push_back(Eigen::Vector3s(0, i * 0.0005, 0));
    rawPoints.push_back(pointCloud);

    std::map<std::string, Eigen::Vector3s> jointCenters;
    jointCenters["PARENT"] = Eigen::Vector3s(i + 1.0, i * 0.0005, 0);
    jointCenters["CHILD"] = Eigen::Vector3s(0.0, i * 0.0005, 0);
    jointsOverTime.push_back(jointCenters);
    avgX += jointCenters.at("PARENT")(0);
  }
  avgX /= TIMESTEPS;

  s_t varX = 0.0;
  for (int i = 0; i < TIMESTEPS; i++)
  {
    varX += (jointsOverTime.at(i).at("PARENT")(0) - avgX)
            * (jointsOverTime.at(i).at("PARENT")(0) - avgX);
  }
  varX /= TIMESTEPS;

  std::vector<biomechanics::MarkerTrace> traces
      = biomechanics::MarkerTrace::createRawTraces(rawPoints);

  EXPECT_EQ(traces.size(), 1);
  EXPECT_EQ(traces[0].mPoints.size(), TIMESTEPS);

  traces[0].computeJointFingerprints(jointsOverTime, jointParents);
}
#endif