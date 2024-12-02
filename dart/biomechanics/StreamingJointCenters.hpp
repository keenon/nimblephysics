#ifndef DART_BIOMECH_STREAMING_JOINT_CENTERS
#define DART_BIOMECH_STREAMING_JOINT_CENTERS

#include <tuple>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

/**
 * If we can predict joint centers, it allows us to leverage the body tracking
 * markers on realtime data.
 */
class StreamingJointCenters
{
public:
  StreamingJointCenters();

  /// This sets the list of class tags that we will associate with markers on
  /// the first body
  void setFirstBodyClasses(std::vector<int> firstBodyClasses);

  /// This sets the list of class tags that we will associate with markers on
  /// the second body
  void setSecondBodyClasses(std::vector<int> secondBodyClasses);

  /// This method takes in a set of markers (already annotated with classes and
  /// unique trace IDs), and updates the internal state of the joint center
  /// predictor to reflect the new information, returning a new joint center
  /// estimate in world space.
  Eigen::Vector3s observeMarkers(
      const std::vector<Eigen::Vector3s>& markers,
      const std::vector<int>& classes,
      const std::vector<int>& traces);

protected:
  int mTotalClasses;
  std::vector<int> mFirstBodyClasses;
  std::vector<int> mSecondBodyClasses;

  // Things we want to track:
  // - History of relative transforms estimated for each body, when those are
  // available.
  // - History of world points observed for each body.

  std::vector<int> mActiveTraceTags;

  // We have two kinds of useful information:
  // - The history of observed points
  // - The body, and point in local space of that body, where the points can be
  // coming from
  // - The mapping between the two

  // If we can get it, can use relative distances from points and PGS to
  // estimate joint center in world space. This is as a gaussian, so it's
  // got a mean and variance for each trace tag.
  Eigen::Matrix<s_t, 2, Eigen::Dynamic>
      mTraceTagToJointCenterDistanceMeanVariance;

  Eigen::MatrixXs mPointBuffer;
};

} // namespace biomechanics
} // namespace dart

#endif