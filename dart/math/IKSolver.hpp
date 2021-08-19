#ifndef DART_MATH_IK_SOLVER_HPP_
#define DART_MATH_IK_SOLVER_HPP_

#include <functional>

#include <Eigen/Dense>

#include "dart/common/Deprecated.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

s_t solveIK(
    Eigen::VectorXs initialPos,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs pos)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::VectorXs& diff,
        /*out*/ Eigen::MatrixXs& jac)> eval,
    s_t convergenceThreshold = 1e-7,
    int maxStepCount = 100,
    s_t leastSquaresDamping = 0.01,
    bool lineSearch = true,
    bool logOutput = false);

}
} // namespace dart

#endif