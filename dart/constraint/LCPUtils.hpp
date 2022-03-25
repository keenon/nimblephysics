#ifndef DART_CONSTRAINT_LCPUTILS_HPP_
#define DART_CONSTRAINT_LCPUTILS_HPP_

#include <memory>

#include <Eigen/Dense>

#include "dart/constraint/BoxedLcpSolver.hpp"

namespace dart {
namespace constraint {

enum LCPSolutionType
{
  SUCCESS,
  FAILURE_IGNORE_FRICTION,
  FAILURE_LOWER_BOUND,
  FAILURE_UPPER_BOUND,
  FAILURE_WITHIN_BOUNDS,
  FAILURE_OUT_OF_BOUNDS
};

class LCPUtils
{
public:
  // This checks whether a solution to an LCP problem is valid.
  static bool isLCPSolutionValid(
      const Eigen::MatrixXs& mA,
      const Eigen::VectorXs& mX,
      const Eigen::VectorXs& mB,
      const Eigen::VectorXs& mHi,
      const Eigen::VectorXs& mLo,
      const Eigen::VectorXi& mFIndex,
      bool ignoreFrictionIndices);

  // This determines the solution types of an LCP problem.
  static LCPSolutionType getLCPSolutionTypes(
      const Eigen::MatrixXs& mA,
      const Eigen::VectorXs& mX,
      const Eigen::VectorXs& mB,
      const Eigen::VectorXs& mHi,
      const Eigen::VectorXs& mLo,
      const Eigen::VectorXi& mFIndex,
      bool ignoreFrictionIndices);

  // This determines the type of a solution to an LCP problem.
  static LCPSolutionType getLCPSolutionType(
      int i,
      const Eigen::MatrixXs& mA,
      const Eigen::VectorXs& mX,
      const Eigen::VectorXs& mB,
      const Eigen::VectorXs& mHi,
      const Eigen::VectorXs& mLo,
      const Eigen::VectorXi& mFIndex,
      bool ignoreFrictionIndices);

  /// This applies a simple algorithm to guess the solution to the LCP problem.
  /// It's not guaranteed to be correct, but it often can be if there is no
  /// sliding friction on this timestep.
  static Eigen::VectorXs guessSolution(
      const Eigen::MatrixXs& mA,
      const Eigen::VectorXs& mB,
      const Eigen::VectorXs& mHi,
      const Eigen::VectorXs& mLo,
      const Eigen::VectorXi& mFIndex);

  /// This reduces an LCP problem by merging any near-identical contact points.
  /// It returns a mapOut matrix, such that if you solve this LCP and then
  /// multiply the resulting x as mapOut*x, you'll get the solution to the
  /// original LCP.
  static Eigen::MatrixXs reduce(
      Eigen::MatrixXs& A,
      Eigen::VectorXs& X,
      Eigen::VectorXs& b,
      Eigen::VectorXs& hi,
      Eigen::VectorXs& lo,
      Eigen::VectorXi& fIndex);

  /// This cuts a problem down to just the normal forces, ignoring friction.
  /// It returns a mapOut matrix, such that if you solve this LCP and then
  /// multiply the resulting x as mapOut*x, you'll get the solution to the
  /// original LCP, but with friction forces all 0.
  static Eigen::MatrixXs removeFriction(
      Eigen::MatrixXs& A,
      Eigen::VectorXs& X,
      Eigen::VectorXs& b,
      Eigen::VectorXs& hi,
      Eigen::VectorXs& lo,
      Eigen::VectorXi& fIndex);

  /// This solves the LCP problem by first automatically de-duplicating columns
  /// to create a reduced version of an equivalent problem, ideally with a
  /// full-rank A. Then the solution to the original LCP is recovered by
  /// re-inflating the solution to the reduced problem.
  static bool solveDeduplicated(
      std::shared_ptr<BoxedLcpSolver>& solver,
      const Eigen::MatrixXs& A,
      Eigen::VectorXs& X,
      const Eigen::VectorXs& b,
      const Eigen::VectorXs& hi,
      const Eigen::VectorXs& lo,
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
      Eigen::MatrixXs& A,
      Eigen::VectorXs& X,
      Eigen::VectorXs& b,
      Eigen::VectorXs& hi,
      Eigen::VectorXs& lo,
      Eigen::VectorXi& fIndex,
      Eigen::MatrixXs& mapOut);

  /// This will modify the LCP problem formulation to drop a column
  /// and rewrite and resize all the matrices appropriately.
  /// It'll also update the mapOut matrix, so that it's possible to simply
  /// multiply mapOut*x on the reduced problem to get a valid solution to the
  /// larger problem.
  static void dropLCPColumn(
      int col,
      Eigen::MatrixXs& A,
      Eigen::VectorXs& X,
      Eigen::VectorXs& b,
      Eigen::VectorXs& hi,
      Eigen::VectorXs& lo,
      Eigen::VectorXi& fIndex,
      Eigen::MatrixXs& mapOut);

  /// Print replication code info
  static void printReplicationCode(
      Eigen::MatrixXs A,
      Eigen::VectorXs x,
      Eigen::VectorXs lo,
      Eigen::VectorXs hi,
      Eigen::VectorXs b,
      Eigen::VectorXi fIndex);
};

} // namespace constraint
} // namespace dart

#endif