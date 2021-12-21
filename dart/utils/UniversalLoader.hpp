#ifndef DART_UNIVERSAL_LOADER
#define DART_UNIVERSAL_LOADER

#include <memory>
#include <string>

#include "dart/include_eigen.hpp"

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/MeshShape.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
}

namespace utils {
namespace UniversalLoader {

/// This loads a whole world from a skel file
std::shared_ptr<simulation::World> loadWorld(const std::string& path);

/// This loads a skeleton from a path, attempting to decide which loader to use
/// based on the path suffix. From Python's perspective, this becomes a method
/// on World, but since in C++ this depends on linking in utils in order to
/// compile, we're leaving it in its own file in utils.
std::shared_ptr<dynamics::Skeleton> loadSkeleton(
    simulation::World* world,
    std::string path,
    Eigen::Vector3s basePosition = Eigen::Vector3s::Zero(),
    Eigen::Vector3s baseEulerAnglesXYZ = Eigen::Vector3s::Zero());

/// This loads a mesh from a file
std::shared_ptr<dynamics::MeshShape> loadMeshShape(std::string path);

} // namespace UniversalLoader
} // namespace utils
} // namespace dart

#endif