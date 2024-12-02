#ifndef DART_UTILS_STR_HPP_
#define DART_UTILS_STR_HPP_

#include <string>

namespace dart {
namespace utils {

std::string ltrim(const std::string& s);

std::string rtrim(const std::string& s);

std::string trim(const std::string& s);

}
}

#endif