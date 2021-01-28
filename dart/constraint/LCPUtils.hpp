#ifndef DART_CONSTRAINT_LCPUTILS_HPP_
#define DART_CONSTRAINT_LCPUTILS_HPP_

#include <memory>

#include <Eigen/Dense>

#include "dart/constraint/BoxedLcpSolver.hpp"

namespace dart {
namespace constraint {

class LCPUtils
{
public:
  static bool isLCPSolutionValid(
      const Eigen::MatrixXd& mA,
      const Eigen::VectorXd& mX,
      const Eigen::VectorXd& mB,
      const Eigen::VectorXd& mHi,
      const Eigen::VectorXd& mLo,
      const Eigen::VectorXi& mFIndex);

  /// This reduces an LCP problem by merging any near-identical contact points.
  /// It returns a mapOut matrix, such that if you solve this LCP and then
  /// multiply the resulting x as mapOut*x, you'll get the solution to the
  /// original LCP.
  static Eigen::MatrixXd reduce(
      Eigen::MatrixXd& A,
      Eigen::VectorXd& X,
      Eigen::VectorXd& b,
      Eigen::VectorXd& hi,
      Eigen::VectorXd& lo,
      Eigen::VectorXi& fIndex);

  /// This solves the LCP problem by first automatically de-duplicating columns
  /// to create a reduced version of an equivalent problem, ideally with a
  /// full-rank A. Then the solution to the original LCP is recovered by
  /// re-inflating the solution to the reduced problem.
  static bool solveDeduplicated(
      std::shared_ptr<BoxedLcpSolver>& solver,
      const Eigen::MatrixXd& A,
      Eigen::VectorXd& X,
      const Eigen::VectorXd& b,
      const Eigen::VectorXd& hi,
      const Eigen::VectorXd& lo,
      const Eigen::VectorXi& fIndex);

  /// This will modify the LCP problem formulation to merge two columns
  /// together, and rewrite and resize all the matrices appropriately. It will
  /// also yell and scream (throw asserts) if the columns shouldn't be merged.
  /// It'll also update the mapOut matrix, so that it's possible to simply
  /// multiply mapOut*x on the reduced problem to get a valid solution to the
  /// larger problem.
  static void mergeLCPColumns(
      int colA,
      int colB,
      Eigen::MatrixXd& A,
      Eigen::VectorXd& X,
      Eigen::VectorXd& b,
      Eigen::VectorXd& hi,
      Eigen::VectorXd& lo,
      Eigen::VectorXi& fIndex,
      Eigen::MatrixXd& mapOut);

  /// Print replication code info
  static void printReplicationCode(
      Eigen::MatrixXd A,
      Eigen::VectorXd x,
      Eigen::VectorXd lo,
      Eigen::VectorXd hi,
      Eigen::VectorXd b,
      Eigen::VectorXi fIndex);
};

} // namespace constraint
} // namespace dart

#endif