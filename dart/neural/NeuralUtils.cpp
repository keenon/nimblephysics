#include "dart/neural/NeuralUtils.hpp"

#include <memory>

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

  // Record the current state of the world
  std::size_t numDOFs = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    numDOFs += world->getSkeleton(i)->getNumDofs();
  }
  Eigen::VectorXd forwardPassPosition = Eigen::VectorXd(numDOFs);
  Eigen::VectorXd forwardPassVelocity = Eigen::VectorXd(numDOFs);
  Eigen::VectorXd forwardPassTorques = Eigen::VectorXd(numDOFs);

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::size_t skelDOF = world->getSkeleton(i)->getNumDofs();
    forwardPassPosition.segment(cursor, skelDOF)
        = world->getSkeleton(i)->getPositions();
    forwardPassVelocity.segment(cursor, skelDOF)
        = world->getSkeleton(i)->getVelocities();
    forwardPassTorques.segment(cursor, skelDOF)
        = world->getSkeleton(i)->getForces();
  }

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
          world, forwardPassPosition, forwardPassVelocity, forwardPassTorques);

  if (idempotent)
    restorableSnapshot->restore();

  return snapshot;
}

} // namespace neural
} // namespace dart