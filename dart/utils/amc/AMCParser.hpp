#ifndef DART_MOCAP_PARSER
#define DART_MOCAP_PARSER

#include <memory>
#include <utility>

#include <Eigen/Dense>

namespace dart {

namespace dynamics {
class Skeleton;
}

namespace utils {
namespace amc {

class AMCParser
{
public:
  std::pair<std::shared_ptr<dynamics::Skeleton>, Eigen::MatrixXd> loadAMC(
      const std::string& asfPath, const std::string& amcPath);
};

} // namespace amc
} // namespace utils
} // namespace dart

#endif