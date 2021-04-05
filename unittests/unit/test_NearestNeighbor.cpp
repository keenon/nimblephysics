/**
 * @file rrts02-nearestNeighbors.cpp
 * @author Can Erdogan
 * @date Feb 04, 2013
 * @brief Checks if the nearest neighbor computation done by flann is correct.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <dart/dart.hpp>
#if HAVE_FLANN
#include <flann/flann.hpp>
#endif // HAVE_FLANN
#include "TestHelpers.hpp"

/* ********************************************************************************************* */
#if HAVE_FLANN
TEST(NEAREST_NEIGHBOR, 2D) {

    // Build the index with the first node
    flann::Index<flann::L2<s_t> > index (flann::KDTreeSingleIndexParams(10, true));
    Eigen::VectorXs p1 (2);
    p1 << -3.04159, -3.04159;
    index.buildIndex(flann::Matrix<s_t>((s_t*)p1.data(), 1, p1.size()));

    // Add two more points
    Eigen::Vector2s p2 (-2.96751, -2.97443), p3 (-2.91946, -2.88672);
    index.addPoints(flann::Matrix<s_t>((s_t*)p2.data(), 1, p2.size()));
    index.addPoints(flann::Matrix<s_t>((s_t*)p3.data(), 1, p3.size()));

    // Check the size of the tree
    EXPECT_EQ(3, (int)index.size());

    // Get the nearest neighbor index for a sample point
    Eigen::Vector2s sample (-2.26654, 2.2874);
    int nearest;
    s_t distance;
    const flann::Matrix<s_t> queryMatrix((s_t*)sample.data(), 1, sample.size());
    flann::Matrix<int> nearestMatrix(&nearest, 1, 1);
    flann::Matrix<s_t> distanceMatrix(flann::Matrix<s_t>(&distance, 1, 1));
    index.knnSearch(queryMatrix, nearestMatrix, distanceMatrix, 1,
        flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
    EXPECT_EQ(2, nearest);

    // Get the nearest neighbor
    s_t* point = index.getPoint(nearest);
    bool equality = equals(Vector2s(point[0], point[1]), p3, 1e-3);
    EXPECT_TRUE(equality);
}
#endif // HAVE_FLANN
