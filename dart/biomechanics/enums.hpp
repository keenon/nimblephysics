#ifndef BIOMECHANICS_ENUMS_HPP
#define BIOMECHANICS_ENUMS_HPP

namespace dart {
namespace biomechanics {

enum MissingGRFReason { notMissingGRF,
                        missingGRF,
                        unmeasuredForce,
                        unmeasuredTorque,
                        forceDiscrepancy,
                        notOverForcePlate,
                        missingImpact,
                        missingBlip,
                        shiftGRF};

}
}

#endif // BIOMECHANICS_ENUMS_HPP
