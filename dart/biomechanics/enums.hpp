#ifndef BIOMECHANICS_ENUMS_HPP
#define BIOMECHANICS_ENUMS_HPP

namespace dart {
namespace biomechanics {

enum MissingGRFReason { notMissingGRF,
                        measuredGrfZeroWhenAccelerationNonZero,
                        unmeasuredExternalForceDetected,
                        torqueDiscrepancy,
                        forceDiscrepancy,
                        notOverForcePlate,
                        missingImpact,
                        missingBlip,
                        shiftGRF};

}
}

#endif // BIOMECHANICS_ENUMS_HPP
