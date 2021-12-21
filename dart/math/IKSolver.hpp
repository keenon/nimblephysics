#ifndef DART_MATH_IK_SOLVER_HPP_
#define DART_MATH_IK_SOLVER_HPP_

#include <functional>

#include "dart/include_eigen.hpp"

#include "dart/common/Deprecated.hpp"
#include "dart/math/Constants.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

struct IKConfig
{
  IKConfig();

  IKConfig& setConvergenceThreshold(s_t v);
  IKConfig& setMaxStepCount(int v);
  IKConfig& setLeastSquaresDamping(s_t v);
  IKConfig& setMaxRestarts(int v);
  IKConfig& setLossLowerBound(s_t v);
  IKConfig& setStartClamped(bool v);
  IKConfig& setDontExitTranspose(bool v);
  IKConfig& setLineSearch(bool v);
  IKConfig& setLogOutput(bool v);

  s_t convergenceThreshold = 1e-7;
  int maxStepCount = 100;
  s_t leastSquaresDamping = 0.01;
  int maxRestarts = 5;
  s_t lossLowerBound = 0;
  bool startClamped = false;
  bool dontExitTranspose = false;
  bool lineSearch = true;
  bool logOutput = false;
};

s_t solveIK(
    Eigen::VectorXs initialPos,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::VectorXs& diff,
        /*out*/ Eigen::MatrixXs& jac)> eval,
    std::function<void(/*out*/ Eigen::VectorXs& pos)> getRandomRestart,
    IKConfig config = IKConfig());

} // namespace math
} // namespace dart

#endif