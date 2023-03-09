#ifndef BIOMECHANICS_ENUMS_HPP
#define BIOMECHANICS_ENUMS_HPP

namespace dart {
namespace biomechanics {

enum MissingGRFReason { notMissingGRF = 0,
                        missingGRF = 1,
                        unmeasuredForce = 2,
                        unmeasuredTorque = 3,
                        forceDiscrepancy = 4,
                        notOverForcePlate = 5,
                        missingImpact = 6,
                        missingBlip = 7,
                        shiftGRF = 8};

}
}

#endif // BIOMECHANICS_ENUMS_HPP
