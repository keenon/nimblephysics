#include "dart/trajectory/Optimizer.hpp"

namespace dart {
namespace trajectory {

//==============================================================================
void Optimizer::registerIntermediateCallback(
    std::function<bool(Problem* problem, int, s_t primal, s_t dual)> callback)
{
  mIntermediateCallbacks.push_back(callback);
}

} // namespace trajectory
} // namespace dart