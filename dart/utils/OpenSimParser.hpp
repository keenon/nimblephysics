#ifndef DART_UTILS_OSIMPARSER_HPP_
#define DART_UTILS_OSIMPARSER_HPP_

#include <string>
#include "dart/common/Uri.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/simulation/World.hpp"
#include "dart/dynamics/Skeleton.hpp"

namespace dart {
namespace utils {

namespace OpenSimParser {
  /// Read Skeleton from osim file
  dynamics::SkeletonPtr readSkeleton(
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever = nullptr);
};

}
}

#endif