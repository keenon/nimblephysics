#include "dart/realtime/MPC.hpp"

#include "dart/realtime/Millis.hpp"

namespace dart {
namespace realtime {

/// This calls getControlForce() with the current system clock as the time parameter
Eigen::VectorXs MPC::getControlForceNow()
{
  return getControlForce(timeSinceEpochMillis());
}

/// This calls recordGroundTruthState() with the current system clock as the
/// time parameter
void MPC::recordGroundTruthStateNow(
    Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass)
{
  recordGroundTruthState(timeSinceEpochMillis(), pos, vel, mass);
}

} // namespace realtime
} // namespace dart