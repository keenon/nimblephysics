#include "dart/trajectory/Optimizer.hpp"

namespace dart {
namespace trajectory {

//==============================================================================
void Optimizer::registerIntermediateCallback(
    std::function<bool(Problem* problem, int, double primal, double dual)>
        callback)
{
  mIntermediateCallbacks.push_back(callback);
}

} // namespace trajectory
} // namespace dart