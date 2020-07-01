#include "dart/neural/NeuralUtils.hpp"

#include <memory>
#include <thread>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

//==============================================================================
std::shared_ptr<ConstrainedGroupGradientMatrices> createGradientMatrices(
    constraint::ConstrainedGroup& group, double timeStep)
{
  return std::make_shared<ConstrainedGroupGradientMatrices>(group, timeStep);
}

//==============================================================================
std::shared_ptr<BackpropSnapshot> forwardPass(
    simulation::WorldPtr world, bool idempotent)
{
  std::shared_ptr<RestorableSnapshot> restorableSnapshot;
  if (idempotent)
  {
    restorableSnapshot = std::make_shared<RestorableSnapshot>(world);
  }

  // Record the current input vector
  Eigen::VectorXd preStepPosition = world->getPositions();
  Eigen::VectorXd preStepVelocity = world->getVelocities();
  Eigen::VectorXd preStepTorques = world->getForces();

  // Set the gradient mode we're going to use to calculate gradients
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  // Actually take a world step. As a byproduct, this will generate gradients
  world->step(!idempotent);

  // Reset the old gradient mode, so we don't have any side effects other than
  // taking a timestep.
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);

  // Actually construct and return the snapshot
  std::shared_ptr<BackpropSnapshot> snapshot
      = std::make_shared<BackpropSnapshot>(
          world, preStepPosition, preStepVelocity, preStepTorques);

  if (idempotent)
    restorableSnapshot->restore();

  return snapshot;
}

//==============================================================================
std::vector<std::shared_ptr<BackpropSnapshot>> bulkForwardPass(
    std::shared_ptr<simulation::World> world,
    std::vector<Eigen::VectorXd> torques,
    std::size_t shootingLength,
    std::vector<Eigen::VectorXd> knotPoses,
    std::vector<Eigen::VectorXd> knotVels)
{
  std::vector<std::shared_ptr<BackpropSnapshot>> ret;
  ret.reserve(torques.size());

  auto shoot = [&world, &torques, &knotPoses, &knotVels, shootingLength, &ret](
                   std::size_t knotIndex) {
    std::shared_ptr<simulation::World> threadWorld = world->clone();
    threadWorld->setPositions(knotPoses[knotIndex]);
    threadWorld->setVelocities(knotVels[knotIndex]);
    std::size_t offset = knotIndex * shootingLength;
    std::size_t end = offset + shootingLength;
    if (end > torques.size())
      end = torques.size();
    for (std::size_t i = offset; i < end; i++)
    {
      threadWorld->setForces(torques[i]);
      ret[i] = forwardPass(threadWorld);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(knotPoses.size());

  // Launch all the threads, naively one per knot
  for (std::size_t i = 0; i < knotPoses.size(); i++)
  {
    threads[i] = std::thread(shoot, i);
  }

  // Wait for everything to finish
  for (std::size_t i = 0; i < knotPoses.size(); i++)
  {
    threads[i].join();
  }

  return ret;
}

} // namespace neural
} // namespace dart