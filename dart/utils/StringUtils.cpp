#include "dart/utils/StringUtils.hpp"

namespace dart {
namespace utils {

const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string& s)
{
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string& s)
{
  size_t end = s.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string& s)
{
  return rtrim(ltrim(s));
}

}
}