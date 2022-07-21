#ifndef UTILS_MJCF_HPP_
#define UTILS_MJCF_HPP_

#include <memory>

#include <dart/dynamics/Skeleton.hpp>

namespace dart {

namespace utils {

/// \brief class FileInfoWorld
class MJCFExporter
{
public:
  static void writeSkeleton(
      const std::string& path, std::shared_ptr<dynamics::Skeleton> skel);
};

} // namespace utils
} // namespace dart

#endif // DART_UTILS_FILEINFOWORLD_HPP_
