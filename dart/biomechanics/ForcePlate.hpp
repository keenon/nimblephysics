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
  Eigen::Vector3s worldOrigin;
  std::vector<s_t> timestamps;
  std::vector<Eigen::Vector3s> corners;
  std::vector<Eigen::Vector3s> centersOfPressure;
  std::vector<Eigen::Vector3s> moments;
  std::vector<Eigen::Vector3s> forces;

  void autodetectNoiseThresholdAndClip(
      s_t percentOfMaxToDetectThumb = 0.25,
      s_t percentOfMaxToCheckThumbRightEdge = 0.35);

  void detectAndFixCopMomentConvention(int trial = -1, int i = -1);

  void trim(s_t newStartTime, s_t newEndTime);

  void trimToIndexes(int start, int end);

  static ForcePlate copyForcePlate(const ForcePlate& plate);
};

} // namespace biomechanics
} // namespace dart

#endif