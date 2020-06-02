#include "dart/neural/RestorableSnapshot.hpp"

#include <vector>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

RestorableSnapshot::RestorableSnapshot(std::shared_ptr<World> world)
{
  mWorld = world;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    mSkeletonConfigurations.push_back(world->getSkeleton(i)->getConfiguration(
        Skeleton::ConfigFlags::CONFIG_ALL));
  }
}

void RestorableSnapshot::restore()
{
  for (std::size_t i = 0; i < mWorld->getNumSkeletons(); i++)
  {
    mWorld->getSkeleton(i)->setConfiguration(mSkeletonConfigurations[i]);
  }
}

} // namespace neural
} // namespace dart