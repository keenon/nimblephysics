#include "dart/constraint/LCPUtils.hpp"

#include <iostream>
#include <vector>

#define MERGE_THRESHOLD 1e-4

namespace dart {
namespace constraint {

bool LCPUtils::isLCPSolutionValid(
    const Eigen::MatrixXd& mA,
    const Eigen::VectorXd& mX,
    const Eigen::VectorXd& mB,
    const Eigen::VectorXd& mHi,
    const Eigen::VectorXd& mLo,
    const Eigen::VectorXi& mFIndex)
{
  Eigen::VectorXd v = mA * mX - mB;
  for (int i = 0; i < mX.size(); i++)
  {
    double upperLimit = mHi(i);
    double lowerLimit = mLo(i);
    if (mFIndex(i) != -1)
    {
      upperLimit *= mX(mFIndex(i));
      lowerLimit *= mX(mFIndex(i));
    }

    const double tol = 1e-5;

    /// Solves constriant impulses for a constrained group. The LCP formulation
    /// setting that this function solve is A*x = b + w where each x[i], w[i]
    /// satisfies one of
    ///   (1) x = lo, w >= 0
    ///   (2) x = hi, w <= 0
    ///   (3) lo < x < hi, w = 0

    // If force has a zero bound, and we're at a zero bound (this is common with
    // friction being upper-bounded by a near-zero normal force) then allow
    // velocity in either direction.
    if (std::abs(lowerLimit) < tol && std::abs(upperLimit) < tol
        && std::abs(mX(i)) < tol)
    {
      // This is always allowed
    }
    // If force is at the lower bound, velocity must be >= 0
    else if (std::abs(mX(i) - lowerLimit) < tol)
    {
      if (v(i) < -tol)
        return false;
    }
    // If force is at the upper bound, velocity must be <= 0
    else if (std::abs(mX(i) - upperLimit) < tol)
    {
      if (v(i) > tol)
        return false;
    }
    // If force is within bounds, then velocity must be zero
    else if (mX(i) > lowerLimit && mX(i) < upperLimit)
    {
      if (std::abs(v(i)) > tol)
        return false;
    }
    // If force is out of bounds, we're always illegal
    else
    {
      return false;
    }
  }
  // If we make it here, the solution is fine
  return true;
}

/// This applies a simple algorithm to guess the solution to the LCP problem.
/// It's not guaranteed to be correct, but it often can be if there is no
/// sliding friction on this timestep.
Eigen::VectorXd LCPUtils::guessSolution(
    const Eigen::MatrixXd& mA,
    const Eigen::VectorXd& mB,
    const Eigen::VectorXd& /* mHi */,
    const Eigen::VectorXd& /* mLo */,
    const Eigen::VectorXi& mFIndex)
{
  std::vector<int> clampingIndices;
  for (int i = 0; i < mB.size(); i++)
  {
    if (mFIndex[i] == -1)
    {
      // Normal forces are only clamping if mB[i] > 0
      if (mB[i] > 0)
        clampingIndices.push_back(i);
    }
    else
    {
      // Always treat friction as clamping
      clampingIndices.push_back(i);
    }
  }

  int numClamping = clampingIndices.size();
  // Special case, everything is clamping
  if (numClamping == mB.size())
  {
    return mA.completeOrthogonalDecomposition().solve(mB);
  }
  // Otherwise we have to map down to the clamping subset
  Eigen::MatrixXd reducedA = Eigen::MatrixXd::Zero(numClamping, numClamping);
  Eigen::VectorXd reducedB = Eigen::VectorXd::Zero(numClamping);
  for (int row = 0; row < numClamping; row++)
  {
    reducedB(row) = mB(clampingIndices[row]);
    for (int col = 0; col < numClamping; col++)
    {
      reducedA(row, col) = mA(clampingIndices[row], clampingIndices[col]);
    }
  }
  // Then we solve the clamping subset, and map back out to the full problem
  // size
  Eigen::VectorXd reducedX
      = reducedA.completeOrthogonalDecomposition().solve(reducedB);
  Eigen::VectorXd fullX = Eigen::VectorXd::Zero(mB.size());
  for (int i = 0; i < numClamping; i++)
  {
    fullX(clampingIndices[i]) = reducedX(i);
  }
  return fullX;
}

/// This reduces an LCP problem by merging any near-identical contact points.
Eigen::MatrixXd LCPUtils::reduce(
    Eigen::MatrixXd& A,
    Eigen::VectorXd& X,
    Eigen::VectorXd& b,
    Eigen::VectorXd& hi,
    Eigen::VectorXd& lo,
    Eigen::VectorXi& fIndex)
{
  Eigen::MatrixXd reducedA = A;
  Eigen::VectorXd reducedX = X;
  Eigen::VectorXd reducedB = b;
  Eigen::VectorXd reducedHi = hi;
  Eigen::VectorXd reducedLo = lo;
  Eigen::VectorXi reducedFIndex = fIndex;
  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  // Step 1. Merge any duplicate columns, as long as we keep finding ones we can
  // merge
  while (true)
  {
    int n = reducedA.cols();
    bool foundDuplicates = false;
    for (int a = 0; a < n - 1; a++)
    {
      for (int b = a + 1; b < n; b++)
      {
        if ((reducedA.col(a) - reducedA.col(b)).squaredNorm() < MERGE_THRESHOLD
            && (std::abs(reducedB(a) - reducedB(b)) < MERGE_THRESHOLD)
            && (reducedFIndex(a) == reducedFIndex(b))
            && (reducedHi(a) == reducedHi(b)) && (reducedLo(a) == reducedLo(b)))
        {
          foundDuplicates = true;
          mergeLCPColumns(
              a,
              b,
              reducedA,
              reducedX,
              reducedB,
              reducedHi,
              reducedLo,
              reducedFIndex,
              mapOut);
          break;
        }
      }
      if (foundDuplicates)
        break;
    }
    if (!foundDuplicates)
      break;
  }
  A = reducedA;
  X = reducedX;
  b = reducedB;
  hi = reducedHi;
  lo = reducedLo;
  fIndex = reducedFIndex;
  return mapOut;
}

bool LCPUtils::solveDeduplicated(
    std::shared_ptr<BoxedLcpSolver>& solver,
    const Eigen::MatrixXd& A,
    Eigen::VectorXd& X,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& hi,
    const Eigen::VectorXd& lo,
    const Eigen::VectorXi& fIndex)
{
  Eigen::MatrixXd reducedA = A;
  Eigen::VectorXd reducedX = X;
  Eigen::VectorXd reducedB = b;
  Eigen::VectorXd reducedHi = hi;
  Eigen::VectorXd reducedLo = lo;
  Eigen::VectorXi reducedFIndex = fIndex;
  Eigen::MatrixXd mapOut = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  // Step 1. Merge any duplicate columns, as long as we keep finding ones we can
  // merge
  while (true)
  {
    int n = reducedA.cols();
    bool foundDuplicates = false;
    for (int a = 0; a < n - 1; a++)
    {
      for (int b = a + 1; b < n; b++)
      {
        if ((reducedA.col(a) - reducedA.col(b)).squaredNorm() < MERGE_THRESHOLD
            && (std::abs(reducedB(a) - reducedB(b)) < MERGE_THRESHOLD)
            && (reducedFIndex(a) == reducedFIndex(b))
            && (reducedHi(a) == reducedHi(b)) && (reducedLo(a) == reducedLo(b)))
        {
          foundDuplicates = true;
          mergeLCPColumns(
              a,
              b,
              reducedA,
              reducedX,
              reducedB,
              reducedHi,
              reducedLo,
              reducedFIndex,
              mapOut);
          break;
        }
      }
      if (foundDuplicates)
        break;
    }
    if (!foundDuplicates)
      break;
  }
  Eigen::MatrixXd oldA = reducedA;
  Eigen::VectorXd oldX = reducedX;
  Eigen::VectorXd oldLo = reducedLo;
  Eigen::VectorXd oldHi = reducedHi;
  Eigen::VectorXd oldB = reducedB;
  Eigen::VectorXi oldFIndex = reducedFIndex;

  // Step 2. Solve
  bool success = solver->solve(
      reducedX.size(),
      reducedA.data(),
      reducedX.data(),
      reducedB.data(),
      0,
      reducedLo.data(),
      reducedHi.data(),
      reducedFIndex.data(),
      false);

  bool valid = LCPUtils::isLCPSolutionValid(
      oldA, reducedX, oldB, oldHi, oldLo, oldFIndex);

  // Step 3. Map out results
  if (success && valid)
  {
    X = mapOut * reducedX;
  }
  else
  {
    std::cout << "LCPUtils::solveDeduplicated() failed to find solution to "
                 "reduced LCP"
              << std::endl;
    LCPUtils::printReplicationCode(
        reducedA, reducedX, reducedLo, reducedHi, reducedB, reducedFIndex);
  }
  return success;
}

/// This will modify the LCP problem formulation to merge two columns
/// together, and rewrite and resize all the matrices appropriately. It will
/// also yell and scream (throw asserts) if the columns shouldn't be merged.
/// It'll also update the mapOut matrix, so that it's possible to simply
/// multiply mapOut*x on the reduced problem to get a valid solution to the
/// larger problem.
void LCPUtils::mergeLCPColumns(
    int colA,
    int colB,
    Eigen::MatrixXd& A,
    Eigen::VectorXd& X,
    Eigen::VectorXd& b,
    Eigen::VectorXd& hi,
    Eigen::VectorXd& lo,
    Eigen::VectorXi& fIndex,
    Eigen::MatrixXd& mapOut)
{
  assert(colA < colB);
  assert((A.col(colA) - A.col(colB)).squaredNorm() < MERGE_THRESHOLD);
  assert(std::abs(b(colA) - b(colB)) < MERGE_THRESHOLD);
  assert(fIndex(colA) == fIndex(colB));
  assert(hi(colA) == hi(colB));
  assert(lo(colA) == lo(colB));

  int n = A.cols();
  Eigen::MatrixXd newACols = Eigen::MatrixXd::Zero(n, n - 1);
  Eigen::VectorXd newX = Eigen::VectorXd::Zero(n - 1);
  Eigen::VectorXd newB = Eigen::VectorXd::Zero(n - 1);
  Eigen::VectorXd newHi = Eigen::VectorXd::Zero(n - 1);
  Eigen::VectorXd newLo = Eigen::VectorXd::Zero(n - 1);
  Eigen::VectorXi newFIndex = Eigen::VectorXi::Zero(n - 1);
  Eigen::MatrixXd newMapOut = Eigen::MatrixXd::Zero(mapOut.rows(), n - 1);

  // Map columns down
  for (int i = 0; i < n; i++)
  {
    if (i == colB)
    {
      newMapOut.col(colA) += mapOut.col(i);
    }
    else
    {
      int newIndex = i;
      if (i > colB)
      {
        newIndex--;
      }

      newACols.col(newIndex) = A.col(i);
      if (i == colA)
      {
        newACols.col(newIndex) *= 2.0;
      }
      newX(newIndex) = X(i);
      newB(newIndex) = b(i);
      newHi(newIndex) = hi(i);
      newLo(newIndex) = lo(i);
      if (fIndex(i) < colB)
      {
        newFIndex(newIndex) = fIndex(i);
      }
      else if (fIndex(i) == colB)
      {
        newFIndex(newIndex) = colA;
      }
      else if (fIndex(i) > colB)
      {
        newFIndex(newIndex) = fIndex(i) - 1;
      }
      else
      {
        // This code should never run
        assert(false);
      }

      newMapOut.col(newIndex) += mapOut.col(i);
    }
  }

  // Map rows down for A
  Eigen::MatrixXd newA = Eigen::MatrixXd::Zero(n - 1, n - 1);
  for (int i = 0; i < n; i++)
  {
    if (i == colB)
    {
      // Do nothing, this is being deleted
    }
    else
    {
      int newIndex = i;
      if (i > colB)
        newIndex--;
      newA.row(newIndex) = newACols.row(i);
    }
  }

  // Write out results
  A = newA;
  X = newX;
  b = newB;
  hi = newHi;
  lo = newLo;
  fIndex = newFIndex;
  mapOut = newMapOut;
}

//==============================================================================
void printDoubleAsCode(double d)
{
  if (d == std::numeric_limits<double>::infinity())
  {
    std::cout << "std::numeric_limits<double>::infinity()";
  }
  else if (d == -std::numeric_limits<double>::infinity())
  {
    std::cout << "-std::numeric_limits<double>::infinity()";
  }
  else
  {
    std::cout << d;
  }
}

void LCPUtils::printReplicationCode(
    Eigen::MatrixXd A,
    Eigen::VectorXd x,
    Eigen::VectorXd lo,
    Eigen::VectorXd hi,
    Eigen::VectorXd b,
    Eigen::VectorXi fIndex)
{
  std::cout << "Code to replicate:" << std::endl;
  std::cout << "--------------------" << std::endl;
  std::cout << "Eigen::MatrixXd A = Eigen::MatrixXd::Zero(" << A.rows() << ","
            << A.cols() << ");" << std::endl;
  std::cout << "// clang-format off" << std::endl;
  std::cout << "A <<" << std::endl;
  for (int row = 0; row < A.rows(); row++)
  {
    for (int col = 0; col < A.cols(); col++)
    {
      std::cout << "  " << A(row, col);
      if (col == A.cols() - 1 && row == A.rows() - 1)
      {
        std::cout << ";";
      }
      else
      {
        std::cout << ",";
      }
      if (col == A.cols() - 1)
      {
        std::cout << std::endl;
      }
    }
  }
  std::cout << "// clang-format on" << std::endl;
  std::cout << "Eigen::VectorXd x = Eigen::VectorXd::Zero(" << x.size() << ");"
            << std::endl;
  std::cout << "x <<" << std::endl;
  for (int i = 0; i < x.size(); i++)
  {
    std::cout << "  ";
    printDoubleAsCode(x(i));
    if (i == x.size() - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
  std::cout << "Eigen::VectorXd lo = Eigen::VectorXd::Zero(" << lo.size()
            << ");" << std::endl;
  std::cout << "lo <<" << std::endl;
  for (int i = 0; i < lo.size(); i++)
  {
    std::cout << "  ";
    printDoubleAsCode(lo(i));
    if (i == lo.size() - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
  std::cout << "Eigen::VectorXd hi = Eigen::VectorXd::Zero(" << hi.size()
            << ");" << std::endl;
  std::cout << "hi <<" << std::endl;
  for (int i = 0; i < hi.size(); i++)
  {
    std::cout << "  ";
    printDoubleAsCode(hi(i));
    if (i == hi.size() - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
  std::cout << "Eigen::VectorXd b = Eigen::VectorXd::Zero(" << b.size() << ");"
            << std::endl;
  std::cout << "b <<" << std::endl;
  for (int i = 0; i < b.size(); i++)
  {
    std::cout << "  ";
    printDoubleAsCode(b(i));
    if (i == b.size() - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
  std::cout << "Eigen::VectorXi fIndex = Eigen::VectorXi::Zero("
            << fIndex.size() << ");" << std::endl;
  std::cout << "fIndex <<" << std::endl;
  for (int i = 0; i < fIndex.size(); i++)
  {
    std::cout << "  ";
    printDoubleAsCode(fIndex(i));
    if (i == fIndex.size() - 1)
    {
      std::cout << ";" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
}

} // namespace constraint
} // namespace dart