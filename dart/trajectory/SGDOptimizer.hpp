#ifndef DART_NEURAL_SGD_OPTIMIZER_HPP_
#define DART_NEURAL_SGD_OPTIMIZER_HPP_

#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/trajectory/Optimizer.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

/*
 * IPOPT wants to own the trajectories it's trying to optimize, so we need a way
 * to create a buffer that's possible for IPOPT to own without freeing the
 * underlying trajectory when it's done.
 */
class SGDOptimizer : public Optimizer
{
public:
  SGDOptimizer();

  virtual ~SGDOptimizer() = default;

  std::shared_ptr<Solution> optimize(
      Problem* shot, std::shared_ptr<Solution> warmStart = nullptr) override;

  void setIterationLimit(int iterationLimit);

  void setTolerance(double tolerance);

  void setLearningRate(double learningRate);

protected:
  int mIterationLimit;
  double mTolerance;
  double mLearningRate;
};

} // namespace trajectory
} // namespace dart

#endif