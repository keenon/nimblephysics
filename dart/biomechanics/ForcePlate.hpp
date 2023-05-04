#ifndef DART_BIOMECH_FORCE_PLATE_HPP_
#define DART_BIOMECH_FORCE_PLATE_HPP_

#include <vector>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"
#include <iostream>

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

  void trimToIndices(int newStartIndex, int newEndIndex) {
    std::cout << "DEBUG original timestamps: " << timestamps.size() << std::endl;
    std::cout << "DEBUG trim 1" << std::endl;
    timestamps.erase(timestamps.begin(), timestamps.begin() + newStartIndex);
    centersOfPressure.erase(centersOfPressure.begin(),
                            centersOfPressure.begin() + newStartIndex);
    moments.erase(moments.begin(), moments.begin() + newStartIndex);
    forces.erase(forces.begin(), forces.begin() + newStartIndex);

    std::cout << "DEBUG trim 2" << std::endl;
    int adjustedEndIndex = newEndIndex - newStartIndex + 1;
    timestamps.erase(timestamps.begin() + adjustedEndIndex, timestamps.end());
    centersOfPressure.erase(centersOfPressure.begin() + adjustedEndIndex,
                            centersOfPressure.end());
    moments.erase(moments.begin() + adjustedEndIndex, moments.end());
    forces.erase(forces.begin() + adjustedEndIndex, forces.end());
    std::cout << "DEBUG trim 3" << std::endl;
    std::cout << "DEBUG new timestamps: " << timestamps.size() << std::endl;

  }

  void trim(s_t newStartTime, s_t newEndTime) {
    assert(newStartTime >= timestamps[0]);
    assert(newEndTime <= timestamps[timestamps.size()-1]);

    // Find new start index.
    auto lower = std::lower_bound(timestamps.begin(), timestamps.end(),
                                  newStartTime);
    int newStartIndex = (int)std::distance(timestamps.begin(), lower);

    // Find new end index.
    auto upper = std::upper_bound(timestamps.begin(), timestamps.end(),
                                  newEndTime);
    int newEndIndex = (int)std::distance(timestamps.begin(), upper);

    // Trim the data.
    trimToIndices(newStartIndex, newEndIndex);
  }

  static ForcePlate copyForcePlate(const ForcePlate& plate) {
    return plate;
  }
};

} // namespace biomechanics
} // namespace dart

#endif