#ifndef DART_UTILS_CSVPARSER_HPP_
#define DART_UTILS_CSVPARSER_HPP_

#include <map>
#include <string>
#include <vector>

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace utils {

/// SkelParser
namespace CSVParser {

/// Read World from skel file
std::vector<std::map<std::string, std::string>> parseFile(
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever = nullptr);

} // namespace CSVParser

} // namespace utils
} // namespace dart

#endif // #ifndef DART_UTILS_SKELPARSER_HPP_
