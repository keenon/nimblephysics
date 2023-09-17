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

  void trim(s_t newStartTime, s_t newEndTime)
  {
    assert(newStartTime >= timestamps[0]);
    assert(newEndTime <= timestamps[timestamps.size() - 1]);
    // Find new start index.
    auto lower
        = std::lower_bound(timestamps.begin(), timestamps.end(), newStartTime);
    int newStartIndex = (int)std::distance(timestamps.begin(), lower);

    // Find new end index.
    auto upper
        = std::upper_bound(timestamps.begin(), timestamps.end(), newEndTime);
    int newEndIndex = (int)std::distance(timestamps.begin(), upper);
    // Actually do the trimming
    trimToIndexes(newStartIndex, newEndIndex);
  }

  void trimToIndexes(int start, int end)
  {
    if (end < timestamps.size())
    {
      // Erase the data from the new end index to the end.
      timestamps.erase(timestamps.begin() + end, timestamps.end());
      centersOfPressure.erase(
          centersOfPressure.begin() + end, centersOfPressure.end());
      moments.erase(moments.begin() + end, moments.end());
      forces.erase(forces.begin() + end, forces.end());
    }
    else
    {
      std::cout << "Warning: trimToIndexes() called with end index " << end
                << " larger than the size of the data (" << timestamps.size()
                << ")." << std::endl;
    }
    if (start < timestamps.size())
    {
      // Erase the data up until the new start index.
      timestamps.erase(timestamps.begin(), timestamps.begin() + start);
      centersOfPressure.erase(
          centersOfPressure.begin(), centersOfPressure.begin() + start);
      moments.erase(moments.begin(), moments.begin() + start);
      forces.erase(forces.begin(), forces.begin() + start);
    }
    else
    {
      std::cout << "Warning: trimToIndexes() called with start index " << end
                << " larger than the size of the data (" << timestamps.size()
                << ")." << std::endl;
    }
  }

  static ForcePlate copyForcePlate(const ForcePlate& plate)
  {
    return plate;
  }
};

} // namespace biomechanics
} // namespace dart

#endif