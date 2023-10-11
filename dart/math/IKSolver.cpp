#include "dart/math/IKSolver.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

namespace dart {
namespace math {

IKConfig::IKConfig()
  : convergenceThreshold(1e-7),
    maxStepCount(100),
    leastSquaresDamping(0.01),
    maxRestarts(5),
    lossLowerBound(1e-10),
    startClamped(false),
    lineSearch(true),
    logOutput(false)
{
}

IKConfig& IKConfig::setConvergenceThreshold(s_t v)
{
  convergenceThreshold = v;
  return *this;
}

IKConfig& IKConfig::setMaxStepCount(int v)
{
  maxStepCount = v;
  return *this;
}

IKConfig& IKConfig::setLeastSquaresDamping(s_t v)
{
  leastSquaresDamping = v;
  return *this;
}

IKConfig& IKConfig::setMaxRestarts(int v)
{
  maxRestarts = v;
  return *this;
}

IKConfig& IKConfig::setLossLowerBound(s_t v)
{
  lossLowerBound = v;
  return *this;
}

IKConfig& IKConfig::setStartClamped(bool v)
{
  startClamped = v;
  return *this;
}

IKConfig& IKConfig::setDontExitTranspose(bool v)
{
  dontExitTranspose = v;
  return *this;
}

IKConfig& IKConfig::setLineSearch(bool v)
{
  lineSearch = v;
  return *this;
}

IKConfig& IKConfig::setLogOutput(bool v)
{
  logOutput = v;
  return *this;
}

IKConfig& IKConfig::setInputNames(const std::vector<std::string>& v)
{
  inputNames = v;
  return *this;
}

IKConfig& IKConfig::setOutputNames(const std::vector<std::string>& v)
{
  outputNames = v;
  return *this;
}

void verifyJacobian(
    const Eigen::VectorXs& originalPos,
    const Eigen::VectorXs& upperBound,
    const Eigen::VectorXs& lowerBound,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs& pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::Ref<Eigen::VectorXs> diff,
        /*out*/ Eigen::Ref<Eigen::MatrixXs> jac)> eval,
    IKConfig config)
{
  // Get the original analytical result
  Eigen::MatrixXs analyticalJac(targetSize, originalPos.size());
  Eigen::VectorXs analyticalDiff(targetSize);
  setPosAndClamp(originalPos, false);
  eval(analyticalDiff, analyticalJac);

  // Finite difference the same Jacobian
  Eigen::MatrixXs result(targetSize, originalPos.size());
  Eigen::MatrixXs scratch(targetSize, originalPos.size());
  s_t eps = 1e-4;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweakedPos = originalPos;
        tweakedPos(dof) += eps;
        setPosAndClamp(tweakedPos, false);
        Eigen::VectorXs out(targetSize);
        eval(out, scratch);
        perturbed = out;
        return true;
      },
      result,
      eps,
      true);

  for (int i = 0; i < originalPos.size(); i++)
  {
    if (originalPos(i) >= upperBound(i) - eps)
    {
      analyticalJac.col(i).setZero();
      result.col(i).setZero();
    }
    if (originalPos(i) <= lowerBound(i) + eps)
    {
      analyticalJac.col(i).setZero();
      result.col(i).setZero();
    }
  }

  // Reset everything how we left it
  setPosAndClamp(originalPos, false);

  if ((result - analyticalJac).norm() > 1e-8)
  {
    std::cout << "Error! Jacobians do not match!! Diff="
              << (result - analyticalJac).norm() << "" << std::endl;
    std::cout << "Analytical: " << std::endl
              << analyticalJac.block(0, 0, 7, 7) << std::endl;
    std::cout << "Brute force: " << std::endl
              << result.block(0, 0, 7, 7) << std::endl;
    Eigen::MatrixXs diff = analyticalJac - result;
    std::cout << "Diff: " << std::endl << diff.block(0, 0, 7, 7) << std::endl;
    std::cout << "Input names size vs Actual input size (should be the same): "
              << config.inputNames.size() << " - " << originalPos.size()
              << std::endl;
    std::cout
        << "Output names size vs Actual output size (should be the same): "
        << config.outputNames.size() << " - " << targetSize << std::endl;
    for (int i = 0; i < diff.rows(); i++)
    {
      for (int j = 0; j < diff.cols(); j++)
      {
        if (abs(diff(i, j)) > 1e-8)
        {
          std::cout << "d (output "
                    << (config.outputNames.size() > i ? config.outputNames[i]
                                                      : std::to_string(i))
                    << " with val " << analyticalDiff(i) << ") wrt d (input "
                    << (config.inputNames.size() > j ? config.inputNames[j]
                                                     : std::to_string(j))
                    << " with val " << originalPos(j)
                    << ") error: " << diff(i, j) << ", analytical "
                    << analyticalJac(i, j) << ", FD " << result(i, j)
                    << std::endl;
        }
      }
    }
    throw std::runtime_error(
        "Error! IK analytical Jacobians do not match finite differenced "
        "version!");
  }
}

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
    IKConfig config)
{
#ifndef NDEBUG
  // For now, disable this, though it DOES fire sometimes which is alarming

  // verifyJacobian(
  //     initialPos,
  //     upperBound,
  //     lowerBound,
  //     targetSize,
  //     setPosAndClamp,
  //     eval,
  //     config);
#endif

  s_t bestError = std::numeric_limits<s_t>::infinity();
  Eigen::VectorXs bestResult = initialPos;

  Eigen::VectorXs pos = setPosAndClamp(initialPos, config.startClamped);

  // For each of the restarts, only do 20 steps, to gauge which one seems most
  // promising
  for (int k = 0; k < config.maxRestarts; k++)
  {
    if (k > 0)
    {
      getRandomRestart(pos);
      pos = setPosAndClamp(pos, true);
      if (config.logOutput)
      {
        std::cout << "## IK random restart " << k << " [best = " << bestError
                  << "]" << std::endl;
      }
    }

    IKResult result = refineIK(
        pos,
        upperBound,
        lowerBound,
        targetSize,
        setPosAndClamp,
        eval,
        IKConfig(config).setMaxStepCount(20));

    if (result.loss < bestError && (result.clamped || !isfinite(bestError)))
    {
      bestError = result.loss;
      bestResult = result.pos;
      if (result.loss <= config.lossLowerBound)
      {
        if (config.logOutput)
        {
          std::cout
              << "Terminating random restarts early, because we found an loss "
              << bestError << " <= " << config.lossLowerBound
              << " that satisfies or exceeds the loss lower-bound we "
                 "were expecting."
              << std::endl;
        }
        break;
      }
    }
  }

  setPosAndClamp(bestResult, true);

  // For the best restart, run the remainder of the steps to further refine the
  // IK solution
  IKResult result = refineIK(
      bestResult,
      upperBound,
      lowerBound,
      targetSize,
      setPosAndClamp,
      eval,
      config);

  if (config.logOutput)
  {
    std::cout << "Finished IK search with loss: " << bestError << std::endl;
  }
  return bestError;
}

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
    IKConfig config)
{
  (void)upperBound;
  (void)lowerBound;

  Eigen::VectorXs pos = initialPos;

  // Allocate these values once, to re-use in the inner loop
  Eigen::VectorXs diff = Eigen::VectorXs::Zero(targetSize);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(targetSize, pos.size());

  s_t lastError = std::numeric_limits<s_t>::infinity();
  s_t lr = 1.0;
  bool useTranspose = false;
  bool clamp = config.startClamped;

  pos = setPosAndClamp(pos, clamp);

  Eigen::VectorXs lastPos = pos;

  for (int i = 0; i < config.maxStepCount; i++)
  {
    // Force clamping on the last 5 steps of IK, even if we wouldn't have
    // otherwise clamped. This means that each run results in _something_
    // valid, even if we hit our maxStepCount before we hit our convergence
    // threshold.
    if (i > config.maxStepCount - 5)
    {
      clamp = true;
    }

    /////////////////////////////////////////////////////////////////////////////
    // Get a current error
    /////////////////////////////////////////////////////////////////////////////

    eval(diff, J);
    assert(!diff.hasNaN());
    assert(!J.hasNaN());
    s_t currentError = diff.squaredNorm();

    /////////////////////////////////////////////////////////////////////////////
    // Measure change from last timestep
    /////////////////////////////////////////////////////////////////////////////

    if (i > 0)
    {
      s_t errorChange = currentError - lastError;

      if (config.logOutput)
      {
        std::cout << "IK "
                  << " iteration " << i - 1
                  << " step: " << (useTranspose ? "transpose" : "DLS")
                  << " clamp: " << clamp << " lr: " << lr
                  << " loss: " << currentError << " change: " << errorChange
                  << std::endl;
      }

      if (currentError < 1e-21)
      {
        if (config.logOutput)
        {
          std::cout << "Terminating IK search after " << i
                    << " iterations with loss: " << currentError << std::endl;
        }
        lastError = currentError;
        break;
      }
      if (errorChange > 0)
      {
        lr *= 0.5;
        if (lr < 1e-4)
        {
          useTranspose = true;
        }
        else if (!config.dontExitTranspose)
        {
          useTranspose = false;
        }

        if (config.lineSearch)
        {
          pos = setPosAndClamp(lastPos, clamp);
        }
        if (lr < 1e-10)
        {
          if (config.logOutput)
          {
            std::cout << "Terminating IK search after " << i
                      << " iterations because learning rate is vanishing, "
                         "with loss: "
                      << currentError << std::endl;
          }
          lastError = currentError;
          break;
        }
      }
      else if (errorChange > -config.convergenceThreshold)
      {
        if (!useTranspose)
        {
          // Go ahead and tighten down the solution with the transpose as far
          // as we can
          if (lr > 5e-5)
          {
            lr = 5e-5;
          }
          useTranspose = true;
        }
        else
        {
          if (!clamp)
          {
            clamp = true;
          }
          else
          {
            // Terminate after we've reached the limit _and_ we're clamping
            // properly
            if (config.logOutput)
            {
              std::cout << "Terminating IK search after " << i
                        << " iterations with optimal loss: " << currentError
                        << std::endl;
            }
            break;
          }
        }
      }
      else
      {
        // Slowly grow LR while we're safely decreasing loss
        lr *= 1.1;
        lastError = currentError;
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    // Do the actual IK update
    /////////////////////////////////////////////////////////////////////////////

    Eigen::VectorXs delta;
    if (useTranspose)
    {
      delta = J.transpose() * diff;
      assert(!delta.hasNaN());
    }
    else
    {
      // Do damped-least-squares
      if (config.leastSquaresDamping == 0)
      {
        delta = J.completeOrthogonalDecomposition().solve(diff);
        assert(!delta.hasNaN());
      }
      else
      {
        if (J.cols() < J.rows())
        {
          Eigen::MatrixXs toInvert
              = J * J.transpose()
                + config.leastSquaresDamping
                      * Eigen::MatrixXs::Identity(J.rows(), J.rows());
          delta = J.transpose() * toInvert.llt().solve(diff);
          assert(!delta.hasNaN());
        }
        else
        {
          Eigen::MatrixXs toInvert
              = J.transpose() * J
                + config.leastSquaresDamping
                      * Eigen::MatrixXs::Identity(J.cols(), J.cols());
          delta = toInvert.llt().solve(J.transpose() * diff);
          assert(!delta.hasNaN());
        }
      }
    }
    lastPos = pos;
    pos = setPosAndClamp(pos - (lr * delta), clamp);
  }

  if (config.logOutput)
  {
    std::cout << "Finished IK search with loss: " << lastError << std::endl;
  }

  IKResult result;
  result.pos = pos;
  result.loss = lastError;
  result.clamped = clamp;

  return result;
}

} // namespace math
} // namespace dart
