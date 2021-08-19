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

s_t solveIK(
    Eigen::VectorXs initialPos,
    int targetSize,
    std::function<Eigen::VectorXs(
        /* in*/ const Eigen::VectorXs pos)> setPosAndClamp,
    std::function<void(
        /*out*/ Eigen::VectorXs& diff,
        /*out*/ Eigen::MatrixXs& jac)> eval,
    s_t convergenceThreshold,
    int maxStepCount,
    s_t leastSquaresDamping,
    bool lineSearch,
    bool logOutput)
{
  s_t lastError = std::numeric_limits<s_t>::infinity();
  s_t lr = 1.0;
  bool useTranspose = false;

  Eigen::VectorXs pos = setPosAndClamp(initialPos);
  Eigen::VectorXs diff = Eigen::VectorXs::Zero(targetSize);
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(targetSize, pos.size());

  Eigen::VectorXs lastPos = pos;

  for (int i = 0; i < maxStepCount; i++)
  {
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

      if (logOutput)
      {
        std::cout << "IK iteration " << i - 1
                  << " step: " << (useTranspose ? "transpose" : "DLS")
                  << " lr: " << lr << " loss: " << currentError
                  << " change: " << errorChange << std::endl;
      }

      if (currentError < 1e-21)
      {
        if (logOutput)
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
        if (lineSearch)
        {
          pos = setPosAndClamp(lastPos);
        }
        if (lr < 1e-10)
        {
          if (logOutput)
          {
            std::cout
                << "Terminating IK search after " << i
                << " iterations because learning rate is vanishing, with loss: "
                << currentError << std::endl;
          }
          break;
        }
      }
      else if (errorChange > -convergenceThreshold)
      {
        if (logOutput)
        {
          std::cout << "Terminating IK search after " << i
                    << " iterations with optimal loss: " << currentError
                    << std::endl;
        }
        break;
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
      if (leastSquaresDamping == 0)
      {
        delta = J.completeOrthogonalDecomposition().solve(diff);
      }
      else
      {
        if (J.cols() < J.rows())
        {
          Eigen::MatrixXs toInvert
              = J * J.transpose()
                + leastSquaresDamping
                      * Eigen::MatrixXs::Identity(J.rows(), J.rows());
          delta = J.transpose() * toInvert.llt().solve(diff);
        }
        else
        {
          Eigen::MatrixXs toInvert
              = J.transpose() * J
                + leastSquaresDamping
                      * Eigen::MatrixXs::Identity(J.cols(), J.cols());
          delta = toInvert.llt().solve(J.transpose() * diff);
        }
      }
    }
    lastPos = pos;
    pos = setPosAndClamp(pos + (lr * delta));
  }
  if (logOutput)
  {
    std::cout << "Finished IK search with loss: " << lastError << std::endl;
  }
  return lastError;
}

} // namespace math
} // namespace dart
