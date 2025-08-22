#ifndef DART_BIOMECH_FORCE_PLATE_HPP_
#define DART_BIOMECH_FORCE_PLATE_HPP_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"

namespace dart {

namespace biomechanics {

struct ForcePlate
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3s worldOrigin;
  std::vector<s_t> timestamps;
  std::vector<Eigen::Vector3s> corners;
  std::vector<Eigen::Vector3s> centersOfPressure;
  std::vector<Eigen::Vector3s> moments;
  std::vector<Eigen::Vector3s> forces;

  // Default constructors and assignment operators:
  ForcePlate() = default;
  ForcePlate(const ForcePlate&) = default;
  ForcePlate(ForcePlate&&) = default;
  ForcePlate& operator=(const ForcePlate&) = default;
  ForcePlate& operator=(ForcePlate&&) = default;

  void autodetectNoiseThresholdAndClip(
      s_t percentOfMaxToDetectThumb = 0.25,
      s_t percentOfMaxToCheckThumbRightEdge = 0.35);

  void detectAndFixCopMomentConvention(int trial = -1, int i = -1);

  void trim(s_t newStartTime, s_t newEndTime);

  void trimToIndexes(int start, int end);

  std::pair<Eigen::MatrixXs, Eigen::VectorXs>
  getResamplingMatrixAndGroundHeights();

  void setResamplingMatrixAndGroundHeights(
      Eigen::MatrixXs matrix, Eigen::VectorXs groundHeights);

  static ForcePlate copyForcePlate(const ForcePlate& plate);
};

} // namespace biomechanics
} // namespace dart

#endif