#ifndef DART_CONSTRAINT_LCPUTILS_HPP_
#define DART_CONSTRAINT_LCPUTILS_HPP_

#include <Eigen/Dense>

namespace dart {
namespace constraint {

class LCPUtils
{
public:
  static void cleanUpResults(
      const Eigen::MatrixXd& A,
      Eigen::VectorXd& X,
      const Eigen::VectorXd& b,
      const Eigen::VectorXd& hi,
      const Eigen::VectorXd& lo,
      const Eigen::VectorXi& fIndex);
};

} // namespace constraint
} // namespace dart

#endif