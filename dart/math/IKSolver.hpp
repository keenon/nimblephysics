#ifndef DART_MATH_IK_SOLVER_HPP_
#define DART_MATH_IK_SOLVER_HPP_

#include <functional>

#include <Eigen/Dense>

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
  IKConfig& setInputNames(const std::vector<std::string>& inputNames);
  IKConfig& setOutputNames(const std::vector<std::string>& outputNames);

  s_t convergenceThreshold = 1e-7;
  int maxStepCount = 100;
  s_t leastSquaresDamping = 0.01;
  int maxRestarts = 5;
  s_t lossLowerBound = 0;
  bool startClamped = false;
  bool dontExitTranspose = false;
  bool lineSearch = true;
  bool logOutput = false;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
};

struct IKResult
{
  s_t loss;
  Eigen::VectorXs pos;
  bool clamped;
};

void verifyJacobian(
    Eigen::VectorXs atPos,
    Eigen::VectorXs upperBound,
    Eigen::VectorXs lowerBound,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs& pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
        /*out*/ Eigen::Ref<Eigen::MatrixXs> jac)> eval,
    IKConfig config = IKConfig());

s_t solveIK(
    const Eigen::VectorXs& initialPos,
    const Eigen::VectorXs& upperBound,
    const Eigen::VectorXs& lowerBound,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs& pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
        /*out*/ Eigen::Ref<Eigen::MatrixXs> jac)> eval,
    std::function<void(/*out*/ Eigen::Ref<Eigen::VectorXs> pos)>
        getRandomRestart,
    IKConfig config = IKConfig());

IKResult refineIK(
    const Eigen::VectorXs& initialPos,
    const Eigen::VectorXs& upperBound,
    const Eigen::VectorXs& lowerBound,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs& pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
        /*out*/ Eigen::Ref<Eigen::MatrixXs> jac)> eval,
    IKConfig config = IKConfig());

} // namespace math
} // namespace dart

#endif