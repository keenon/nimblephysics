#ifndef DART_UTILS_OSIMPARSER_HPP_
#define DART_UTILS_OSIMPARSER_HPP_

#include <string>
#include <map>

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/XmlHelpers.hpp"

namespace dart {
using namespace utils;

namespace biomechanics {

struct OpenSimFile
{
  dynamics::SkeletonPtr skeleton;
  // Markers map
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markersMap;
  // TODO: eventually we'll want to record muscles here
};

class OpenSimParser
{
public:
  /// Read Skeleton from osim file
  static OpenSimFile parseOsim(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

protected:
  static OpenSimFile readOsim30(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
  static OpenSimFile readOsim40(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif