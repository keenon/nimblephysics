#ifndef BIOMECHANICS_ENUMS_HPP
#define BIOMECHANICS_ENUMS_HPP

namespace dart {
namespace biomechanics {

enum MissingGRFReason
{
  notMissingGRF,
  measuredGrfZeroWhenAccelerationNonZero,
  unmeasuredExternalForceDetected,
  torqueDiscrepancy,
  forceDiscrepancy,
  notOverForcePlate,
  missingImpact,
  missingBlip,
  shiftGRF,
  interpolatedClippedGRF
};

enum ProcessingPassType
{
  kinematics,
  dynamics,
  lowPassFilter
};

} // namespace biomechanics
} // namespace dart

#endif // BIOMECHANICS_ENUMS_HPP
