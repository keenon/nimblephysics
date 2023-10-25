#include "dart/biomechanics/ForcePlate.hpp"

#include <fstream>
#include <ostream>
#include <string>

#include "dart/math/MathTypes.hpp"
#include "dart/utils/StringUtils.hpp"

namespace dart {
using namespace utils;
namespace biomechanics {

void ForcePlate::autodetectNoiseThresholdAndClip(
    s_t percentOfMaxToCut, s_t percentOfMaxToCheckThumbRightEdge)
{
  int trialLen = forces.size();
  Eigen::VectorXs forceNorms(trialLen);
  int numExactlyZero = 0;
  for (int i = 0; i < trialLen; i++)
  {
    forceNorms(i) = forces[i].norm();
    if (forceNorms[i] == 0.0)
    {
      numExactlyZero++;
    }
  }
  if (numExactlyZero > 5)
  {
    // If there are more than a few exactly zero force readings (so more
    // than just the first and last frame, for example), then we assume
    // that someone has already clipped this trial to a cutoff threshold.
    std::cout << "not clipping force plate because it appears to already have "
                 "been clipped"
              << std::endl;
    return;
  }

  int numBins = 200;
  // Don't go higher than 200N as the max force, because above that we start
  // selecting cutoff levels that are frighteningly high
  double maxForce = std::min(200.0, forceNorms.maxCoeff());
  std::vector<int> hist(numBins, 0);

  for (int i = 0; i < trialLen; i++)
  {
    int bin = std::min(
        static_cast<int>((forceNorms(i) / maxForce) * numBins), numBins - 1);
    hist[bin]++;
  }

  int avg_bin_value = trialLen / numBins;
  auto max_it = std::max_element(hist.begin(), hist.end());
  int histMaxIndex = std::distance(hist.begin(), max_it);

  if (histMaxIndex < (int)((s_t)numBins * percentOfMaxToCut))
  {
    int rightBound = histMaxIndex;
    for (int j = histMaxIndex; j < numBins; j++)
    {
      if (hist[j] < avg_bin_value)
      {
        rightBound = j;
        break;
      }
    }

    if (rightBound > (int)((s_t)numBins * percentOfMaxToCheckThumbRightEdge))
    {
      std::cout << "Not clipping force plate because it has no obvious thumb "
                   "in the histogram. Right bound = "
                << rightBound << " > threshold = "
                << (int)((s_t)numBins * percentOfMaxToCheckThumbRightEdge)
                << std::endl;
      int maxBin = 0;
      for (int i = 0; i < numBins; i++)
      {
        if (hist[i] > maxBin)
        {
          maxBin = hist[i];
        }
      }
      std::cout << numBins << " bins (from 0N to " << maxForce
                << "N):" << std::endl;
      for (int i = 0; i < numBins; i++)
      {
        std::cout << (i == histMaxIndex ? "[Max] " : "")
                  << (i == rightBound ? "[Right] " : "") << i << ": ";
        s_t percent = (s_t)hist[i] / maxBin;
        for (int j = 0; j < int(std::ceil(percent * 20)); j++)
        {
          std::cout << "*";
        }
        std::cout << std::endl;
      }
    }
    else
    {
      double zeroThreshold
          = (static_cast<double>(rightBound) / numBins) * maxForce;
      std::cout << "clip force plate at " << zeroThreshold << " N" << std::endl;
      for (int i = 0; i < trialLen; i++)
      {
        if (forceNorms(i) < zeroThreshold)
        {
          forces[i] = Eigen::Vector3s::Zero();
          moments[i] = Eigen::Vector3s::Zero();
          centersOfPressure[i] = Eigen::Vector3s::Zero();
        }
      }
    }
  }
}

void ForcePlate::detectAndFixCopMomentConvention(int trial, int i)
{
  // Try to figure out the coordinate system of the moments data on this
  // force plate

  int numParallelMoments = 0;
  int numVerticalMoments = 0;
  int numWorldMoments = 0;
  int numOffsetMoments = 0;
  std::vector<Eigen::Vector3s> copOffsets;
  for (int t = 0; t < forces.size(); t++)
  {
    Eigen::Vector3s f = forces[t];
    Eigen::Vector3s m = moments[t];
    Eigen::Vector3s cop = centersOfPressure[t];
    // Check for the section of the moment that is not parallel with force
    Eigen::Vector3s parallelComponent = f.normalized().dot(m) * f.normalized();
    Eigen::Vector3s antiParallelComponent = m - parallelComponent;
    s_t percentageAntiparallel
        = antiParallelComponent.norm() / parallelComponent.norm();
    // Check for the "moment is in world space" interpretation
    Eigen::Vector3s tau = cop.cross(f);
    Eigen::Vector3s worldM = m - tau;
    if (f.norm() < 3 || cop.isZero())
    {
      // Zero out this data
      // plate.forces[t].setZero();
      // plate.moments[t].setZero();
    }
    else if (percentageAntiparallel < 0.1)
    {
      // This is parallel GRF data
      numParallelMoments++;
    }
    else if (abs(m(0)) < 1e-5 && abs(m(2)) < 1e-5)
    {
      // This is vertical GRF data
      numVerticalMoments++;
    }
    else if (
        worldM.norm() < m.norm() * 0.3
        || (worldM.norm() < 15 && worldM.norm() < m.norm()))
    {
      // This is parallel GRF data if the moment is interpreted as a world
      // frame moment
      numWorldMoments++;
    }
    else
    {
      numOffsetMoments++;

      Eigen::Vector3s recoveredCop
          = Eigen::Vector3s(m(2) / f(1), 0, -m(0) / f(1));
      Eigen::Vector3s copDiff = recoveredCop - cop;
      copOffsets.push_back(copDiff);
    }
  }

  // Now we try to decide how to interpret this force plate
  if (numParallelMoments >= numOffsetMoments
      && numParallelMoments >= numVerticalMoments
      && numParallelMoments >= numWorldMoments)
  {
    std::cout << "Interpreting force plate " << i << " in trial " << trial
              << " as having centers-of-pressure that are calculated so "
                 "that remaining moment is PARALLEL to forces."
              << std::endl;
  }
  else if (
      numVerticalMoments >= numOffsetMoments
      && numVerticalMoments >= numParallelMoments
      && numVerticalMoments >= numWorldMoments)
  {
    std::cout << "Interpreting force plate " << i << " in trial " << trial
              << " as having centers-of-pressure that are calculated so "
                 "that remaining moment is VERTICAL."
              << std::endl;
  }
  else if (
      numWorldMoments >= numOffsetMoments
      && numWorldMoments >= numParallelMoments
      && numWorldMoments >= numVerticalMoments)
  {
    std::cout
        << "Interpreting force plate " << i << " in trial " << trial
        << " as having moments expressed in WORLD SPACE. We will recompute "
           "the moments to be expressed at the center of pressure."
        << std::endl;
    for (int t = 0; t < forces.size(); t++)
    {
      Eigen::Vector3s f = forces[t];
      Eigen::Vector3s m = moments[t];
      Eigen::Vector3s cop = centersOfPressure[t];
      Eigen::Vector3s tau = cop.cross(f);
      Eigen::Vector3s worldM = m - tau;
      if (f.norm() < 3 || cop.isZero())
        continue;
      moments[t] = worldM;
    }
  }
  else if (
      numOffsetMoments >= numWorldMoments
      && numOffsetMoments >= numParallelMoments
      && numOffsetMoments >= numVerticalMoments)
  {
    // Before we pronouce a judgement, we need to compute some statistics
    // about the CoP offset. If it's consistent around a particular offset,
    // then we can be confident that's where the force plate is supposed to
    // offset to. If not, then this is an error case and needs to get zero'd
    // out.
    Eigen::Vector3s averageOffset = Eigen::Vector3s::Zero();
    for (Eigen::Vector3s& v : copOffsets)
    {
      averageOffset += v;
    }
    averageOffset /= copOffsets.size();

    Eigen::Vector3s offsetVariance = Eigen::Vector3s::Zero();
    for (Eigen::Vector3s& v : copOffsets)
    {
      offsetVariance += (v - averageOffset).cwiseProduct(v - averageOffset);
    }
    offsetVariance /= copOffsets.size();

    if (offsetVariance.norm() < 0.2)
    {
      std::cout
          << "Interpreting force plate " << i << " in trial " << trial
          << " as having moments expressed in OFFSET WORLD SPACE (offset = "
          << averageOffset(0) << "," << averageOffset(1) << ","
          << averageOffset(2) << "; variance = " << offsetVariance.norm()
          << "). We will recompute the "
             "moments to be expressed at the center of pressure."
          << std::endl;

      for (int t = 0; t < forces.size(); t++)
      {
        Eigen::Vector3s f = forces[t];
        Eigen::Vector3s m = moments[t];
        Eigen::Vector3s cop = centersOfPressure[t];
        cop += averageOffset;
        Eigen::Vector3s tau = cop.cross(f);
        Eigen::Vector3s worldM = m - tau;
        if (f.norm() < 3 || cop.isZero())
          continue;
        moments[t] = worldM;
      }
    }
    else
    {
      std::cout << "BAD INPUT DATA DETECTED!! Could not find an interpretation "
                   "for "
                   "force plate "
                << i << " in trial " << trial
                << " where the moments, forces, and centers of pressure are "
                   "consistent. Continuing anyways, but EXPECT BAD RESULTS!"
                << std::endl;
    }
  }
  else
  {
    assert(false && "this should be impossible to reach");
  }
}

void ForcePlate::trim(s_t newStartTime, s_t newEndTime)
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

void ForcePlate::trimToIndexes(int start, int end)
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
  else if (end > timestamps.size())
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

ForcePlate ForcePlate::copyForcePlate(const ForcePlate& plate)
{
  return plate;
}

} // namespace biomechanics
} // namespace dart