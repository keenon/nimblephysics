#import "dart/realtime/MPC.hpp"

#include "dart/realtime/Millis.hpp"

namespace dart {
namespace realtime {

/// This calls getForce() with the current system clock as the time parameter
Eigen::VectorXd MPC::getForceNow()
{
  return getForce(timeSinceEpochMillis());
}

/// This calls recordGroundTruthState() with the current system clock as the
/// time parameter
void MPC::recordGroundTruthStateNow(
    Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass)
{
  recordGroundTruthState(timeSinceEpochMillis(), pos, vel, mass);
}

} // namespace realtime
} // namespace dart