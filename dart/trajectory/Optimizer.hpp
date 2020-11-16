#ifndef DART_TRAJECTORY_OPTIMIZER
#define DART_TRAJECTORY_OPTIMIZER

#include <memory>

namespace dart {
namespace trajectory {

class Problem;
class Solution;

class Optimizer
{
  virtual std::shared_ptr<Solution> optimize(
      Problem* shot, std::shared_ptr<Solution> warmStart = nullptr)
      = 0;
};

} // namespace trajectory
} // namespace dart

#endif