#include "dart/math/AssignmentMatcher.hpp"

namespace dart {
namespace math {

/// This maps the rows to columns. If there are fewer columns than rows,
/// unassigned rows get assigned to -1
Eigen::VectorXi AssignmentMatcher::assignRowsToColumns(
    const Eigen::MatrixXs& weights)
{
  std::vector<int> rowsNeedAssignments;
  std::vector<int> colsNeedAssignments;

  for (int i = 0; i < weights.rows(); i++)
  {
    rowsNeedAssignments.push_back(i);
  }
  for (int i = 0; i < weights.cols(); i++)
  {
    colsNeedAssignments.push_back(i);
  }

  Eigen::VectorXi mapping = -1 * Eigen::VectorXi::Ones(weights.rows());

  // TODO: this is a greedy algorithm that does not return optimal assignments.
  // We should eventually implement the Hungarian method here.
  while (rowsNeedAssignments.size() > 0 && colsNeedAssignments.size() > 0)
  {
    int maxRowIndex = -1;
    int maxColIndex = -1;
    s_t maxScore = -1 * std::numeric_limits<double>::infinity();

    for (int row = 0; row < rowsNeedAssignments.size(); row++)
    {
      for (int col = 0; col < colsNeedAssignments.size(); col++)
      {
        s_t score = weights(rowsNeedAssignments[row], colsNeedAssignments[col]);
        if (score > maxScore)
        {
          maxScore = score;
          maxRowIndex = row;
          maxColIndex = col;
        }
      }
    }

    if (maxRowIndex == -1 || maxColIndex == -1)
    {
      break;
    }

    int row = rowsNeedAssignments[maxRowIndex];
    int col = colsNeedAssignments[maxColIndex];
    mapping(row) = col;

    rowsNeedAssignments.erase(rowsNeedAssignments.begin() + maxRowIndex);
    colsNeedAssignments.erase(colsNeedAssignments.begin() + maxColIndex);
  }

  return mapping;
}

std::map<std::string, std::string> AssignmentMatcher::assignKeysToKeys(
    std::vector<std::string> source,
    std::vector<std::string> target,
    std::function<double(std::string, std::string)> weight)
{
  Eigen::MatrixXs weights = Eigen::MatrixXs::Zero(source.size(), target.size());
  for (int i = 0; i < source.size(); i++)
  {
    for (int j = 0; j < target.size(); j++)
    {
      weights(i, j) = weight(source[i], target[j]);
    }
  }

  Eigen::VectorXi assignment = assignRowsToColumns(weights);

  std::map<std::string, std::string> result;
  for (int i = 0; i < assignment.size(); i++)
  {
    if (assignment[i] != -1)
    {
      result[source[i]] = target[assignment[i]];
    }
  }

  return result;
}

} // namespace math
} // namespace dart