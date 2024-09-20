#ifndef BIOMECHANICS_ENUMS_HPP
#define BIOMECHANICS_ENUMS_HPP

namespace dart {
namespace biomechanics {

enum MissingGRFReason
{
  notMissingGRF,
  measuredGrfZeroWhenAccelerationNonZero,
  unmeasuredExternalForceDetected,
  footContactDetectedButNoForce,
  torqueDiscrepancy,
  forceDiscrepancy,
  notOverForcePlate,
  missingImpact,
  missingBlip,
  shiftGRF,
  manualReview,
  interpolatedClippedGRF
};

enum MissingGRFStatus
{
  no = 0,      // no will cast to `false`
  unknown = 1, // unknown will cast to `true`
  yes = 2,     // yes will cast to `true`
};

enum ProcessingPassType
{
  kinematics,
  dynamics,
  lowPassFilter,
  accMinimizingFilter
};

} // namespace biomechanics
} // namespace dart

#endif // BIOMECHANICS_ENUMS_HPP
