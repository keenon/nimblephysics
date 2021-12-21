#ifndef DART_NEURAL_FULL_SNAPSHOT_HPP_
#define DART_NEURAL_FULL_SNAPSHOT_HPP_

#include <memory>
#include <vector>

#include "dart/include_eigen.hpp"

#include "dart/dynamics/Skeleton.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace neural {

class RestorableSnapshot
{
public:
  RestorableSnapshot(std::shared_ptr<simulation::World> world);
  void restore();
  /// Returns true if the world is already in the preserved state
  bool isPreserved();

private:
  std::shared_ptr<simulation::World> mWorld;
  Eigen::VectorXs mLCPCache;
  std::vector<dynamics::Skeleton::Configuration> mSkeletonConfigurations;
};

} // namespace neural

} // namespace dart

#endif