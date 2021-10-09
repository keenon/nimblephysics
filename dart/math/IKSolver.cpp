#include "dart/math/IKSolver.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dart/math/Helpers.hpp"

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

s_t solveIK(
    Eigen::VectorXs initialPos,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs pos, bool clamp)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::VectorXs& diff,
        /*out*/ Eigen::MatrixXs& jac)> eval,
    std::function<void(/*out*/ Eigen::VectorXs& pos)> getRandomRestart,
    IKConfig config)
{
  s_t bestError = std::numeric_limits<s_t>::infinity();
  Eigen::VectorXs bestResult = initialPos;

  Eigen::VectorXs pos = initialPos;

  for (int k = 0; k < config.maxRestarts; k++)
  {
    s_t lastError = std::numeric_limits<s_t>::infinity();
    s_t lr = 1.0;
    bool useTranspose = false;
    bool clamp = config.startClamped;

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
    pos = setPosAndClamp(pos, clamp);
    Eigen::VectorXs diff = Eigen::VectorXs::Zero(targetSize);
    Eigen::MatrixXs J = Eigen::MatrixXs::Zero(targetSize, pos.size());

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
      s_t currentError = diff.squaredNorm();

      /////////////////////////////////////////////////////////////////////////////
      // Measure change from last timestep
      /////////////////////////////////////////////////////////////////////////////

      if (i > 0)
      {
        s_t errorChange = currentError - lastError;

        if (config.logOutput)
        {
          std::cout << "[best = " << bestError << "] IK pass " << k
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
        }
      }

      lastError = currentError;

      /////////////////////////////////////////////////////////////////////////////
      // Do the actual IK update
      /////////////////////////////////////////////////////////////////////////////

      Eigen::VectorXs delta;
      if (useTranspose)
      {
        delta = J.transpose() * diff;
      }
      else
      {
        // Do damped-least-squares
        if (config.leastSquaresDamping == 0)
        {
          delta = J.completeOrthogonalDecomposition().solve(diff);
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
          }
          else
          {
            Eigen::MatrixXs toInvert
                = J.transpose() * J
                  + config.leastSquaresDamping
                        * Eigen::MatrixXs::Identity(J.cols(), J.cols());
            delta = toInvert.llt().solve(J.transpose() * diff);
          }
        }
      }
      lastPos = pos;
      pos = setPosAndClamp(pos + (lr * delta), clamp);
    }
    if (lastError < bestError && clamp)
    {
      bestError = lastError;
      bestResult = pos;
      if (lastError <= config.lossLowerBound)
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
  if (config.logOutput)
  {
    std::cout << "Finished IK search with loss: " << bestError << std::endl;
  }
  return bestError;
}

} // namespace math
} // namespace dart
