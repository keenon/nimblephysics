#include "dart/trajectory/SGDOptimizer.hpp"

#include <vector>

#define LOG_PERFORMANCE_SGD

using namespace dart;
using namespace simulation;
using namespace performance;

namespace dart {
namespace trajectory {

//==============================================================================
SGDOptimizer::SGDOptimizer()
  : mIterationLimit(100), mTolerance(0), mLearningRate(1e-2)
{
}

//==============================================================================
std::shared_ptr<Solution> SGDOptimizer::optimize(
    Problem* shot, std::shared_ptr<Solution> reuseRecord)
{
  std::shared_ptr<Solution> record
      = reuseRecord ? reuseRecord : std::make_shared<Solution>();

  int n = shot->getFlatProblemDim(shot->mWorld);
  Eigen::VectorXs x = Eigen::VectorXs::Zero(n);
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(n);
  shot->flatten(shot->mWorld, x);
  s_t loss = shot->getLoss(shot->mWorld);

  for (int i = 0; i < mIterationLimit; i++)
  {
    s_t newLoss = shot->getLoss(shot->mWorld);
    s_t improvement = loss - newLoss;
    if (improvement > 0 && improvement < mTolerance)
    {
      std::cout << "Improvement less than tolerance, converged." << std::endl;
      break;
    }
    loss = newLoss;
    std::cout << "Iter " << i << ": " << newLoss << std::endl;
    shot->getGradientWrtRolloutCache(shot->mWorld);
    shot->backpropGradient(shot->mWorld, grad);
    x -= grad * mLearningRate;
    shot->unflatten(shot->mWorld, x);

    for (auto callback : mIntermediateCallbacks)
    {
      callback(shot, i, loss, 0.0);
    }
  }

  return record;
}

//==============================================================================
void SGDOptimizer::setIterationLimit(int iterationLimit)
{
  mIterationLimit = iterationLimit;
}

//==============================================================================
void SGDOptimizer::setTolerance(s_t tolerance)
{
  mTolerance = tolerance;
}

//==============================================================================
void SGDOptimizer::setLearningRate(s_t learningRate)
{
  mLearningRate = learningRate;
}

} // namespace trajectory
} // namespace dart