#ifndef BIOMECHANICS_MACROS_HPP
#define BIOMECHANICS_MACROS_HPP

#include <stdexcept>

namespace dart {
namespace biomechanics {

#define NIMBLE_THROW(message) \
    throw std::runtime_error(message); \

#define NIMBLE_THROW_IF(condition, message) \
    do { \
        if (condition) { \
            throw std::runtime_error(message); \
        } \
    } while (false)

}
}

#endif // BIOMECHANICS_MACROS_HPP
