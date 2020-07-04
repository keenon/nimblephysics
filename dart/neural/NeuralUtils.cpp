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
BulkForwardPassResult bulkForwardPass(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd torques,
    std::size_t shootingLength,
    Eigen::MatrixXd knotPoses,
    Eigen::MatrixXd knotVels)
{
  int dofs = world->getNumDofs();

  BulkForwardPassResult result;
  result.postStepPoses = Eigen::MatrixXd(dofs, torques.size());
  result.postStepVels = Eigen::MatrixXd(dofs, torques.size());
  result.snapshots.reserve(torques.cols());
  for (int i = 0; i < torques.cols(); i++)
  {
    result.snapshots.push_back(nullptr);
  }

  // This is the lambda we'll pass to our threads
  // It runs a single shot out to a knot point
  auto shoot
      = [&world, &torques, &knotPoses, &knotVels, shootingLength, &result](
            std::size_t knotIndex) {
          std::shared_ptr<simulation::World> threadWorld = world->clone();
          threadWorld->setPositions(knotPoses.col(knotIndex));
          threadWorld->setVelocities(knotVels.col(knotIndex));
          std::size_t offset = knotIndex * shootingLength;
          std::size_t end = offset + shootingLength;
          if (end > torques.size())
            end = torques.size();
          for (std::size_t i = offset; i < end; i++)
          {
            threadWorld->setForces(torques.col(i));
            result.snapshots[i] = forwardPass(threadWorld);
            result.postStepPoses.col(i)
                = result.snapshots[i]->getPostStepPosition();
            result.postStepVels.col(i)
                = result.snapshots[i]->getPostStepVelocity();
          }
        };

  // This runs a bundle of shots on a single thread, since we generally
  // have many more shots than CPU cores, and want to
  auto shootSeveral
      = [&shoot](std::size_t startKnot, std::size_t endKnotExclusive) {
          for (std::size_t i = startKnot; i < endKnotExclusive; i++)
          {
            shoot(i);
          }
        };

  // Figure out how many threads we'll need
  std::size_t numThreads = std::thread::hardware_concurrency();
  std::size_t numKnots = knotPoses.cols();
  std::size_t knotsPerThread = ceil(static_cast<double>(numKnots) / numThreads);
  // We may have more threads than knots, in which case we need to cut back on
  // the number of threads
  numThreads = ceil(static_cast<double>(numKnots) / knotsPerThread);

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  // Launch all the threads
  std::size_t start = 0;
  for (std::size_t i = 0; i < numThreads; i++)
  {
    std::size_t end = start + knotsPerThread;
    if (end > numKnots)
      end = numKnots;
    threads.push_back(std::thread(shootSeveral, start, end));
    start = end;
  }

  // Wait for everything to finish
  for (std::size_t i = 0; i < numThreads; i++)
  {
    threads[i].join();
  }

  return result;
}

//==============================================================================
BulkBackwardPassResult bulkBackwardPass(
    std::shared_ptr<simulation::World> world,
    std::vector<std::shared_ptr<BackpropSnapshot>> snapshots,
    std::size_t shootingLength,
    Eigen::MatrixXd gradWrtPoses,
    Eigen::MatrixXd gradWrtVels,
    bool computeJacobians)
{
  int dofs = world->getNumDofs();
  std::size_t numKnots
      = floor(static_cast<double>(snapshots.size()) / shootingLength);

  BulkBackwardPassResult result;
  result.gradWrtPreStepKnotPoses = Eigen::MatrixXd(dofs, numKnots);
  result.gradWrtPreStepKnotVels = Eigen::MatrixXd(dofs, numKnots);
  result.gradWrtPreStepTorques = Eigen::MatrixXd(dofs, snapshots.size());
  if (computeJacobians)
  {
    // Initialize enough Jacobians to handle all the knots
    result.knotJacobians.reserve(numKnots);
    for (std::size_t i = 0; i < numKnots; i++)
    {
      KnotJacobian J;
      result.knotJacobians.push_back(J);
    }
  }

  // This is the lambda we'll pass to our threads
  // It runs backprop for a single shot back from the knot point
  auto shootBackwards = [&world,
                         &snapshots,
                         &gradWrtPoses,
                         &gradWrtVels,
                         &result,
                         shootingLength,
                         computeJacobians](std::size_t knotIndex) {
    std::shared_ptr<simulation::World> threadWorld = world->clone();

    int begin = knotIndex * shootingLength;
    int end = begin + shootingLength;
    if (end > static_cast<int>(snapshots.size()))
      end = snapshots.size();

    std::size_t dofs = world->getNumDofs();

    /*
    For reference during implementation in C++:

    This computes a full Jacobian relating the whole shot to the error at this
knot-point (last_knot_index + 1)

    For our purposes here (forward Jacobians), the forward computation
    graph looks like this:

    p_t -------------+--------------------------------> p_t+1 ---->
                      \                                   /
                        \                                 /
    v_t ----------------+----(LCP Solver)----> v_t+1 ---+---->
                        /
                      /
    f_t -------------+
    */

    Eigen::VectorXd gradWrtNextStepPos = Eigen::VectorXd::Zero(dofs);
    Eigen::VectorXd gradWrtNextStepVel = Eigen::VectorXd::Zero(dofs);

    // p_end <-- p_t+1
    Eigen::MatrixXd posend_posnext = Eigen::MatrixXd::Identity(dofs, dofs);
    // p_end <-- v_t+1
    Eigen::MatrixXd posend_velnext = Eigen::MatrixXd::Zero(dofs, dofs);
    // v_end <-- p_t+1
    Eigen::MatrixXd velend_posnext = Eigen::MatrixXd::Zero(dofs, dofs);
    // v_end <-- v_t+1
    Eigen::MatrixXd velend_velnext = Eigen::MatrixXd::Identity(dofs, dofs);

    KnotJacobian& J = result.knotJacobians.at(knotIndex);
    if (computeJacobians)
    {
      J.torquesEndPos.reserve(end - begin);
      J.torquesEndVel.reserve(end - begin);
    }

    for (int i = end - 1; i >= begin; i--)
    {
      /////////////////////////////////////////////////////////////////////////
      // Backpropagate gradient
      /////////////////////////////////////////////////////////////////////////

      LossGradient lossWrtPostStep;
      lossWrtPostStep.lossWrtPosition
          = gradWrtPoses.col(i) + gradWrtNextStepPos;
      lossWrtPostStep.lossWrtVelocity = gradWrtVels.col(i) + gradWrtNextStepVel;

      LossGradient lossWrtPreStep;
      snapshots[i]->backprop(
          threadWorld, lossWrtPreStep /* OUT */, lossWrtPostStep);

      gradWrtNextStepPos = lossWrtPreStep.lossWrtPosition;
      gradWrtNextStepVel = lossWrtPreStep.lossWrtVelocity;
      result.gradWrtPreStepTorques.col(i) = lossWrtPreStep.lossWrtTorque;

      if (i == begin)
      {
        result.gradWrtPreStepKnotPoses.col(knotIndex)
            = lossWrtPreStep.lossWrtPosition;
        result.gradWrtPreStepKnotVels.col(knotIndex)
            = lossWrtPreStep.lossWrtVelocity;
      }

      /////////////////////////////////////////////////////////////////////////
      // Backpropagate Jacobians
      /////////////////////////////////////////////////////////////////////////

      if (computeJacobians)
      {
        // p_t+1 <-- v_t+1
        Eigen::MatrixXd posnext_velnext = Eigen::MatrixXd::Identity(dofs, dofs)
                                          * threadWorld->getTimeStep();

        // v_t+1 <-- p_t
        Eigen::MatrixXd velnext_pos
            = snapshots[i]->getPosVelJacobian(threadWorld);
        // v_t+1 <-- v_t
        Eigen::MatrixXd velnext_vel
            = snapshots[i]->getVelVelJacobian(threadWorld);
        // v_t+1 <-- f_t
        Eigen::MatrixXd velnext_force
            = snapshots[i]->getForceVelJacobian(threadWorld);
        // p_t+1 <-- p_t
        Eigen::MatrixXd posnext_pos
            = snapshots[i]->getPosPosJacobian(threadWorld);
        // p_t+1 <-- v_t = (p_t+1 <-- v_t+1) * (v_t+1 <-- v_t)
        Eigen::MatrixXd posnext_vel = posnext_velnext * velnext_vel;
        // p_t+1 <-- f_t = (p_t+1 <-- v_t+1) * (v_t+1 <-- f_t)
        Eigen::MatrixXd posnext_force = posnext_velnext * velnext_force;

        // p_end <-- f_t = ((p_end <-- p_t+1) * (p_t+1 <-- f_t)) + ((p_end <--
        // v_t+1) * (v_t+1 <-- f_t))
        Eigen::MatrixXd posend_force = (posend_posnext * posnext_force)
                                       + (posend_velnext * velnext_force);
        // v_end <-- f_t ...
        Eigen::MatrixXd velend_force = (velend_posnext * posnext_force)
                                       + (velend_velnext * velnext_force);

        // Write our torques into the appropriate spot in the torques block
        J.torquesEndPos.push_back(posend_force);
        J.torquesEndVel.push_back(velend_force);

        // Update p_end <-- p_t+1 = ((p_end <-- p_t+1) * (p_t+1 <-- p_t)) +
        // ((p_end <-- v_t+1) * (v_t+1 <-- p_t))
        posend_posnext
            = (posend_posnext * posnext_pos) + (posend_velnext * velnext_pos);
        // Update v_end <-- p_t+1 ...
        velend_posnext
            = (velend_posnext * posnext_pos) + (velend_velnext * velnext_pos);
        // Update v_t+1 --> p_end ...
        posend_velnext
            = (posend_posnext * posnext_vel) + (posend_velnext * velnext_vel);
        // Update v_t+1 --> v_end ...
        velend_velnext
            = (velend_posnext * posnext_vel) + (velend_velnext * velnext_vel);
      }
    }

    if (computeJacobians)
    {
      std::reverse(J.torquesEndPos.begin(), J.torquesEndPos.end());
      std::reverse(J.torquesEndVel.begin(), J.torquesEndVel.end());
      J.knotPosEndPos = posend_posnext;
      J.knotVelEndPos = posend_velnext;
      J.knotPosEndVel = velend_posnext;
      J.knotVelEndVel = velend_velnext;
    }

    /*
    For reference during implementation in C++:

def insertFullJacobian(last_knot_index: int, cursor: int, includePhasePhase:
bool = True):
    """
    This computes a full Jacobian relating the whole shot to the error at this
knot-point (last_knot_index + 1)
    """
    start_index = last_knot_index * self.shooting_length
    end_index_exclusive = (last_knot_index + 1) * self.shooting_length
    """
    For our purposes here (forward Jacobians), the forward computation
    graph looks like this:

    p_t -------------+--------------------------------> p_t+1 ---->
                      \                                   /
                        \                                 /
    v_t ----------------+----(LCP Solver)----> v_t+1 ---+---->
                        /
                      /
    f_t -------------+
    """

    # p_end <-- p_t+1
    posend_posnext = np.identity(world_dofs)
    # p_end <-- v_t+1
    # np.zeros((world_dofs, world_dofs))
    # -dt * dt * np.identity(world_dofs)
    posend_velnext = np.zeros((world_dofs, world_dofs))
    # v_end <-- p_t+1
    velend_posnext = np.zeros((world_dofs, world_dofs))
    # v_end <-- v_t+1
    velend_velnext = np.identity(world_dofs)

    for i in reversed(range(start_index, end_index_exclusive)):
        snapshot: dart.neural.BackpropSnapshot = self.snapshots[i]

        # p_t+1 <-- v_t+1
        posnext_velnext = dt

        # v_t+1 <-- p_t
        velnext_pos = snapshot.getPosVelJacobian(self.world)
        # v_t+1 <-- v_t
        velnext_vel = snapshot.getVelVelJacobian(self.world)
        # v_t+1 <-- f_t
        velnext_force = snapshot.getForceVelJacobian(self.world)

        # p_t+1 <-- p_t = (p_t+1 <-- p_t) + ((p_t+1 <-- v_t+1) * (v_t+1 <--
p_t)) posnext_pos = snapshot.getPosPosJacobian( self.world)  # +
(posnext_velnext * velnext_pos) # p_t+1 <-- v_t = (p_t+1 <-- v_t+1) * (v_t+1 <--
v_t) posnext_vel = posnext_velnext * velnext_vel # p_t+1 <-- f_t = (p_t+1 <--
v_t+1) * (v_t+1 <-- f_t) posnext_force = posnext_velnext * velnext_force

        # p_end <-- f_t = ((p_end <-- p_t+1) * (p_t+1 <-- f_t)) + ((p_end <--
v_t+1) * (v_t+1 <-- f_t)) posend_force = np.matmul( posend_posnext,
posnext_force) + np.matmul(posend_velnext, velnext_force) # v_end <-- f_t ...
        velend_force = np.matmul(
            velend_posnext, posnext_force) + np.matmul(velend_velnext,
velnext_force)

        ########## HERE

        # Write our force_pos_end and force_vel_end into the output, row by row
        for row in range(world_dofs):
            for col in range(world_dofs):
                out[cursor] = posend_force[row][col]
                cursor += 1
        for row in range(world_dofs):
            for col in range(world_dofs):
                out[cursor] = velend_force[row][col]
                cursor += 1

        if i == end_index_exclusive - 1 and False:
            print('v_t+1 <-- p_t:\n'+str(velnext_pos))
            print('v_t+1 <-- v_t:\n'+str(velnext_vel))
            print('v_t+1 <-- f_t:\n'+str(velnext_force))
            print('(p_t+1 <-- v_t+1) * (v_t+1 <-- p_t):\n' +
                  str(posnext_velnext * velnext_pos))
            print('p_t+1 <-- p_t:\n'+str(posnext_pos))
            print('p_t+1 <-- v_t:\n'+str(posnext_vel))
            print('p_t+1 <-- f_t:\n'+str(posnext_force))
            print('p_end <-- f_t:\n'+str(posend_force))
            print('v_end <-- f_t:\n'+str(velend_force))

        # Update p_end <-- p_t+1 = ((p_end <-- p_t+1) * (p_t+1 <-- p_t)) +
((p_end <-- v_t+1) * (v_t+1 <-- p_t)) posend_posnext = np.matmul(
            posend_posnext, posnext_pos) + np.matmul(posend_velnext,
velnext_pos) # Update v_end <-- p_t+1 ... velend_posnext = np.matmul(
            velend_posnext, posnext_pos) + np.matmul(velend_velnext,
velnext_pos) # Update v_t+1 --> p_end ... posend_velnext = np.matmul(
            posend_posnext, posnext_vel) + np.matmul(posend_velnext,
velnext_vel) # Update v_t+1 --> v_end ... velend_velnext = np.matmul(
            velend_posnext, posnext_vel) + np.matmul(velend_velnext,
velnext_vel)

    if includePhasePhase:
        # Put these so the rows correspond to a whole phase vector
        posend_phase = np.concatenate([posend_posnext, posend_velnext], axis=1)
        velend_phase = np.concatenate([velend_posnext, velend_velnext], axis=1)
        for row in range(world_dofs):
            for col in range(world_dofs * 2):
                out[cursor] = posend_phase[row][col]
                cursor += 1
        for row in range(world_dofs):
            for col in range(world_dofs * 2):
                out[cursor] = velend_phase[row][col]
                cursor += 1
    return cursor
    */
  };

  // This runs a bundle of shots on a single thread, since we generally
  // have many more shots than CPU cores, and want to
  auto shootSeveralBackwards
      = [&shootBackwards](std::size_t startKnot, std::size_t endKnotExclusive) {
          for (std::size_t i = startKnot; i < endKnotExclusive; i++)
          {
            shootBackwards(i);
          }
        };

  // Figure out how many threads we'll need
  std::size_t numThreads = std::thread::hardware_concurrency();
  std::size_t knotsPerThread = ceil(static_cast<double>(numKnots) / numThreads);
  // We may have more threads than knots, in which case we need to cut back on
  // the number of threads
  numThreads = ceil(static_cast<double>(numKnots) / knotsPerThread);

  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  // Launch all the threads
  std::size_t start = 0;
  for (std::size_t i = 0; i < numThreads; i++)
  {
    std::size_t end = start + knotsPerThread;
    if (end > numKnots)
      end = numKnots;
    threads.push_back(std::thread(shootSeveralBackwards, start, end));
    start = end;
  }

  // Wait for everything to finish
  for (std::size_t i = 0; i < numThreads; i++)
  {
    threads[i].join();
  }

  return result;
}

} // namespace neural
} // namespace dart