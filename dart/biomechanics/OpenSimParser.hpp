#ifndef DART_UTILS_OSIMPARSER_HPP_
#define DART_UTILS_OSIMPARSER_HPP_

#include <string>

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/XmlHelpers.hpp"

namespace dart {
using namespace utils;

namespace biomechanics {

class OpenSimParser
{
public:
  /// Read Skeleton from osim file
  static dynamics::SkeletonPtr readSkeleton(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

protected:
  static dynamics::SkeletonPtr readOsim30(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
  static dynamics::SkeletonPtr readOsim40(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif