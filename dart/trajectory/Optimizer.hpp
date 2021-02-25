#ifndef DART_TRAJECTORY_OPTIMIZER
#define DART_TRAJECTORY_OPTIMIZER

#include <functional>
#include <memory>
#include <vector>

namespace dart {
namespace trajectory {

class Problem;
class Solution;

class Optimizer
{
public:
  virtual std::shared_ptr<Solution> optimize(
      Problem* shot, std::shared_ptr<Solution> warmStart = nullptr)
      = 0;

  /// This registers an intermediate callback, to get called by IPOPT after each
  /// step of optimization. If any callback returns false on a given step, then
  /// the optimizer will terminate early.
  void registerIntermediateCallback(
      std::function<bool(Problem* problem, int, double primal, double dual)>
          callback);

protected:
  std::vector<
      std::function<bool(Problem* problem, int, double primal, double dual)>>
      mIntermediateCallbacks;
};

} // namespace trajectory
} // namespace dart

#endif