#include "dart/constraint/LCPUtils.hpp"

#include <iostream>
#include <vector>

namespace dart {
namespace constraint {

void LCPUtils::cleanUpResults(
    const Eigen::MatrixXd& A,
    Eigen::VectorXd& X,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& hi,
    const Eigen::VectorXd& lo,
    const Eigen::VectorXi& fIndex)
{
  std::vector<int> clampingIndices;
  for (int i = 0; i < X.size(); i++)
  {
    double upperBound = hi(i);
    double lowerBound = lo(i);
    const int fIndexPointer = fIndex(i);
    if (fIndexPointer != -1)
    {
      upperBound *= X(fIndexPointer);
      lowerBound *= X(fIndexPointer);
    }

    if (std::abs(X(i)) < 1e-9)
    {
      // Not clamping
    }
    else if (X(i) > upperBound || X(i) < lowerBound)
    {
      // Upper bounded
    }
    else
    {
      // Clamping
      clampingIndices.push_back(i);
    }
  }

  int numClamping = clampingIndices.size();
  if (numClamping == 0)
    return;

  Eigen::MatrixXd reducedA = Eigen::MatrixXd(numClamping, numClamping);
  Eigen::VectorXd reducedB = Eigen::VectorXd(numClamping);

  for (int row = 0; row < numClamping; row++)
  {
    int clampingRow = clampingIndices[row];

    reducedB(row) = b(clampingRow);

    for (int col = 0; col < numClamping; col++)
    {
      int clampingCol = clampingIndices[col];
      reducedA(row, col) = A(clampingRow, clampingCol);
    }
  }

  Eigen::VectorXd reducedX
      = reducedA.completeOrthogonalDecomposition().solve(reducedB);

  for (int row = 0; row < numClamping; row++)
  {
    int clampingRow = clampingIndices[row];
    X(clampingRow) = reducedX(row);
  }
}

} // namespace constraint
} // namespace dart