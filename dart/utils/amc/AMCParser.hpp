#ifndef DART_MOCAP_PARSER
#define DART_MOCAP_PARSER

#include <memory>
#include <utility>

#include "dart/include_eigen.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {

namespace dynamics {
class Skeleton;
}

namespace utils {
namespace amc {

class AMCParser
{
public:
  std::pair<std::shared_ptr<dynamics::Skeleton>, Eigen::MatrixXs> loadAMC(
      const std::string& asfPath, const std::string& amcPath);
};

} // namespace amc
} // namespace utils
} // namespace dart

#endif