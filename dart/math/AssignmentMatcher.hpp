#ifndef MATH_LINEARFN_H_
#define MATH_LINEARFN_H_

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class AssignmentMatcher
{
public:
  /// This maps the rows to columns. If there are fewer columns than rows,
  /// unassigned rows get assigned to -1
  static Eigen::VectorXi assignRowsToColumns(const Eigen::MatrixXs& weights);

  static std::map<std::string, std::string> assignKeysToKeys(
      std::vector<std::string> source,
      std::vector<std::string> target,
      std::function<double(std::string, std::string)> weight);

protected:
};

} // namespace math
} // namespace dart

#endif