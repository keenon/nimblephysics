#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/StreamingMarkerTraces.hpp"
#include "dart/math/MathTypes.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(MARKER_TRACES_BASICS, SIMPLE_TRACE_CONSTRUCTION)
{
  int numClasses = 5;
  int numWindows = 3;
  int stride = 1;
  int maxMarkersPerTimestep = 1;
  StreamingMarkerTraces markerTraces(
      numClasses, numWindows, stride, maxMarkersPerTimestep);

  for (int i = 0; i < 10; i++)
  {
    std::vector<Eigen::Vector3s> markers;
    markers.push_back(Eigen::Vector3s::Ones() * 0.01 * i);
    std::vector<int> classes = markerTraces.observeMarkers(markers, i).first;
    EXPECT_EQ(classes.size(), 1);
    EXPECT_EQ(classes[0], numClasses - 1);
    EXPECT_EQ(markerTraces.getNumTraces(), 1);
  }
}
#endif

#ifdef ALL_TESTS
TEST(MARKER_TRACES_BASICS, HANDLING_LOGITS)
{
  int numClasses = 5;
  int numWindows = 3;
  int stride = 1;
  int maxMarkersPerTimestep = 1;
  StreamingMarkerTraces markerTraces(
      numClasses, numWindows, stride, maxMarkersPerTimestep);

  std::vector<Eigen::Vector3s> originalMarkers;
  for (int i = 0; i <= 10; i++)
  {
    std::vector<Eigen::Vector3s> markers;
    markers.push_back(Eigen::Vector3s::Ones() * 0.01 * i);
    originalMarkers.push_back(markers[0]);
    std::vector<int> classes = markerTraces.observeMarkers(markers, i).first;
    if (i > 0)
    {
      EXPECT_EQ(classes[0], 3);
    }

    Eigen::MatrixXs logits = Eigen::MatrixXs::Zero(numClasses, 1);
    logits(3, 0) = 1;
    Eigen::VectorXi traceIDs = Eigen::VectorXi::Zero(1);
    markerTraces.observeTraceLogits(logits, traceIDs);
  }
}
#endif

// #ifdef ALL_TESTS
TEST(MARKER_TRACES_BASICS, MAKING_FEATURES)
{
  int numClasses = 5;
  int numWindows = 5;
  int stride = 2;
  int maxMarkersPerTimestep = 1;
  StreamingMarkerTraces markerTraces(
      numClasses, numWindows, stride, maxMarkersPerTimestep);

  std::vector<Eigen::Vector3s> originalMarkers;
  for (int i = 0; i <= 10; i++)
  {
    std::vector<Eigen::Vector3s> markers;
    markers.push_back(Eigen::Vector3s::Ones() * 0.01 * i);
    originalMarkers.push_back(markers[0]);
    std::vector<int> classes = markerTraces.observeMarkers(markers, i).first;
  }

  auto pair = markerTraces.getTraceFeatures(false);
  Eigen::MatrixXs features = pair.first;
  Eigen::VectorXi traceIDs = pair.second;
  EXPECT_EQ(features.cols(), 5);
  EXPECT_EQ(features.rows(), 4);
  for (int i = 0; i < features.cols(); i++)
  {
    EXPECT_EQ(features(3, i), (i + 4) % features.cols());
    EXPECT_EQ(traceIDs(i), 0);
    int originalCol = i * 2;
    // At the end, we wrap around and begin overwriting the buffer
    if (i == 0)
    {
      originalCol = 10;
    }
    Eigen::Vector3s centering = Eigen::Vector3s::Zero() * 0.04;
    Eigen::Vector3s originalMarker = originalMarkers[originalCol] - centering;
    Eigen::Vector3s feature = features.block<3, 1>(0, i);
    if ((originalMarker - feature).norm() > 1e-8)
    {
      Eigen::Matrix3s compare;
      compare.col(0) = originalMarker;
      compare.col(1) = feature;
      compare.col(2) = originalMarker - feature;
      std::cout << "Original - feature - diff: " << std::endl
                << compare << std::endl;
    }
    EXPECT_TRUE((originalMarker - feature).norm() < 1e-8);
  }
}
// #endif

#ifdef ALL_TESTS
TEST(MARKER_TRACES_BASICS, MAKING_MIXED_FEATURES)
{
  int numClasses = 5;
  int numWindows = 5;
  int stride = 2;
  int maxMarkersPerTimestep = 10;
  StreamingMarkerTraces markerTraces(
      numClasses, numWindows, stride, maxMarkersPerTimestep);

  int numMarkers = 5;

  std::vector<std::vector<Eigen::Vector3s>> originalMarkers;
  for (int m = 0; m < numMarkers; m++)
  {
    originalMarkers.push_back(std::vector<Eigen::Vector3s>());
  }

  for (int i = 0; i <= 10; i++)
  {
    std::vector<Eigen::Vector3s> markers;
    for (int m = 0; m < numMarkers; m++)
    {
      Eigen::Vector3s point = Eigen::Vector3s(0.01 * i, 0.1 * m, 0.0);
      originalMarkers[m].push_back(point);
      markers.push_back(point);
    }
    std::vector<int> classes = markerTraces.observeMarkers(markers, i).first;
  }

  auto pair = markerTraces.getTraceFeatures(false);
  Eigen::MatrixXs features = pair.first;
  Eigen::VectorXi traceIDs = pair.second;
  EXPECT_EQ(features.cols(), 5 * numMarkers);
  EXPECT_EQ(features.rows(), 4);
  // std::cout << "Features: " << std::endl << features.transpose() <<
  // std::endl; std::cout << "Trace IDs: " << std::endl << traceIDs <<
  // std::endl;
}
#endif