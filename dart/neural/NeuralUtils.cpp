#include "dart/neural/NeuralUtils.hpp"

#include <memory>
#include <thread>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/math/Geometry.hpp"
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

//==============================================================================
Eigen::MatrixXd jointToWorldSpatialJacobian(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  int dofs = skel->getNumDofs();
  int bodies = skel->getNumBodyNodes();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(bodies * 6, dofs);
  for (int i = 0; i < bodies; i++)
  {
    jac.block(i * 6, 0, 6, dofs) = skel->getWorldJacobian(skel->getBodyNode(i));
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd jointToWorldLinearJacobian(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::MatrixXd jac = jointToWorldSpatialJacobian(skel);
  Eigen::MatrixXd reduced = Eigen::MatrixXd(jac.rows() / 2, jac.cols());
  for (int i = 0; i < jac.rows() / 6; i++)
  {
    reduced.block(i * 3, 0, 3, jac.cols())
        = jac.block((i * 6) + 3, 0, 3, jac.cols());
  }
  return reduced;
}

//==============================================================================
Eigen::VectorXd skelConvertJointSpacePositionsToWorldSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions)
{
  Eigen::VectorXd oldPositions = skel->getPositions();
  skel->setPositions(jointPositions);
  Eigen::VectorXd out = Eigen::VectorXd::Zero(skel->getNumBodyNodes() * 6);
  for (std::size_t i = 0; i < skel->getNumBodyNodes(); i++)
  {
    out.segment(i * 6, 6)
        = math::logMap(skel->getBodyNode(i)->getWorldTransform());
  }
  skel->setPositions(oldPositions);
  return out;
}

//==============================================================================
Eigen::VectorXd skelConvertJointSpaceVelocitiesToWorldSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVels)
{
  Eigen::MatrixXd jac = jointToWorldSpatialJacobian(skel);
  return jac * jointVels;
}

//==============================================================================
Eigen::VectorXd skelBackpropWorldSpatialToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd bodySpace,
    bool useTranspose)
{
  Eigen::MatrixXd jac = jointToWorldSpatialJacobian(skel);
  if (useTranspose)
  {
    return jac.transpose() * bodySpace;
  }
  else
  {
    return jac.completeOrthogonalDecomposition().solve(bodySpace);
  }
}

//==============================================================================
Eigen::VectorXd skelConvertJointSpacePositionsToWorldLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions)
{
  Eigen::VectorXd oldPositions = skel->getPositions();
  skel->setPositions(jointPositions);
  Eigen::VectorXd out = Eigen::VectorXd::Zero(skel->getNumBodyNodes() * 3);
  for (std::size_t i = 0; i < skel->getNumBodyNodes(); i++)
  {
    out.segment(i * 3, 3)
        = skel->getBodyNode(i)->getWorldTransform().translation();
  }
  skel->setPositions(oldPositions);
  return out;
}

//==============================================================================
Eigen::VectorXd skelConvertJointSpaceVelocitiesToWorldLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVelocities)
{
  Eigen::VectorXd screws
      = skelConvertJointSpaceVelocitiesToWorldSpatial(skel, jointVelocities);
  Eigen::VectorXd positions = Eigen::VectorXd::Zero(screws.size() / 2);
  for (std::size_t i = 0; i < skel->getNumBodyNodes(); i++)
  {
    /*
    std::cout << "Expmap->linear: \n"
              << math::expMap(screws.segment(i * 6, 6)).translation()
              << "\nTail segment: \n"
              << screws.segment((i * 6) + 3, 3) << "\n";
              */
    // positions.segment(i * 3, 3)
    // = math::expMap(screws.segment(i * 6, 6)).translation();
    positions.segment(i * 3, 3) = screws.segment((i * 6) + 3, 3);
  }
  return positions;
}

//==============================================================================
Eigen::VectorXd skelBackpropWorldLinearToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd worldSpace,
    bool useTranspose)
{
  Eigen::MatrixXd jac = jointToWorldSpatialJacobian(skel);
  Eigen::MatrixXd reduced = Eigen::MatrixXd(jac.rows() / 2, jac.cols());
  for (int i = 0; i < jac.rows() / 6; i++)
  {
    for (int j = 0; j < jac.cols(); j++)
    {
      reduced.block(i * 3, j, 3, 1)
          = math::expMap(jac.block((i * 6), j, 6, 1)).translation();
    }
  }
  if (useTranspose)
  {
    return reduced.transpose() * worldSpace;
  }
  else
  {
    return reduced.completeOrthogonalDecomposition().solve(worldSpace);
  }
}

//==============================================================================
Eigen::Vector3d skelConvertJointSpacePositionsToWorldCOM(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointPositions)
{
  Eigen::VectorXd oldPositions = skel->getPositions();
  skel->setPositions(jointPositions);

  Eigen::Vector3d comPos = skel->getCOM();

  skel->setPositions(oldPositions);

  return comPos;
}

//==============================================================================
Eigen::Vector3d skelConvertJointSpaceVelocitiesToWorldCOMLinear(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVelocities)
{
  Eigen::VectorXd bodyVels
      = skelConvertJointSpaceVelocitiesToWorldLinear(skel, jointVelocities);

  Eigen::Vector3d comVel = Eigen::Vector3d::Zero();
  double totalMass = 0.0;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    comVel += skel->getBodyNode(i)->getMass() * bodyVels.segment(i * 3, 3);
    totalMass += skel->getBodyNode(i)->getMass();
  }
  comVel /= totalMass;

  return comVel;
}

//==============================================================================
Eigen::VectorXd skelBackpropWorldCOMLinearToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector3d lossWrtWorldCOM,
    bool useTranspose)
{
  if (useTranspose)
  {
    return skel->getCOMLinearJacobian().transpose() * lossWrtWorldCOM;
  }
  else
  {
    return skel->getCOMLinearJacobian().completeOrthogonalDecomposition().solve(
        lossWrtWorldCOM);
  }
}

//==============================================================================
Eigen::Vector6d skelConvertJointSpaceVelocitiesToWorldCOMSpatial(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXd jointVelocities)
{
  Eigen::VectorXd bodyVels
      = skelConvertJointSpaceVelocitiesToWorldSpatial(skel, jointVelocities);

  Eigen::Vector6d comVel = Eigen::Vector6d::Zero();
  double totalMass = 0.0;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    comVel += skel->getBodyNode(i)->getMass() * bodyVels.segment(i * 6, 6);
    totalMass += skel->getBodyNode(i)->getMass();
  }
  comVel /= totalMass;

  return comVel;
}

//==============================================================================
Eigen::VectorXd skelBackpropWorldCOMSpatialToJointSpace(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector6d lossWrtWorldCOM,
    bool useTranspose)
{
  if (useTranspose)
  {
    return skel->getCOMJacobian().transpose() * lossWrtWorldCOM;
  }
  else
  {
    return skel->getCOMJacobian().completeOrthogonalDecomposition().solve(
        lossWrtWorldCOM);
  }
}

//==============================================================================
Eigen::MatrixXd convertJointSpacePositionsToWorldSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions)
{
  int dofs = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    dofs += skel->getNumBodyNodes() * 6;
  }

  Eigen::MatrixXd ret = Eigen::MatrixXd(dofs, jointPositions.cols());
  for (int i = 0; i < jointPositions.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (int k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      Eigen::VectorXd worldSpace = skelConvertJointSpacePositionsToWorldSpatial(
          world->getSkeleton(k),
          jointPositions.block(jointSpaceCursor, i, skelDofs, 1));
      ret.block(worldSpaceCursor, i, worldSpace.size(), 1) = worldSpace;
      worldSpaceCursor += worldSpace.size();
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities)
{
  int dofs = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    dofs += skel->getNumBodyNodes() * 6;
  }

  Eigen::MatrixXd ret = Eigen::MatrixXd(dofs, jointVelocities.cols());
  for (int i = 0; i < jointVelocities.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      Eigen::VectorXd worldSpace
          = skelConvertJointSpaceVelocitiesToWorldSpatial(
              world->getSkeleton(k),
              jointVelocities.block(jointSpaceCursor, i, skelDofs, 1));
      ret.block(worldSpaceCursor, i, worldSpace.size(), 1) = worldSpace;
      worldSpaceCursor += worldSpace.size();
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd backpropWorldSpatialToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd worldSpaceLoss,
    bool useTranspose)
{
  Eigen::MatrixXd ret
      = Eigen::MatrixXd(world->getNumDofs(), worldSpaceLoss.cols());
  for (int i = 0; i < worldSpaceLoss.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      int skelBodyDofs = skel->getNumBodyNodes() * 6;
      Eigen::VectorXd jointSpace = skelBackpropWorldSpatialToJointSpace(
          world->getSkeleton(k),
          worldSpaceLoss.block(worldSpaceCursor, i, skelBodyDofs, 1),
          useTranspose);
      ret.block(jointSpaceCursor, i, skelDofs, 1) = jointSpace;
      worldSpaceCursor += skelBodyDofs;
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd convertJointSpacePositionsToWorldLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions)
{
  int dofs = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    dofs += skel->getNumBodyNodes() * 3;
  }

  Eigen::MatrixXd ret = Eigen::MatrixXd(dofs, jointPositions.cols());
  for (int i = 0; i < jointPositions.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (int k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      Eigen::VectorXd worldSpace = skelConvertJointSpacePositionsToWorldLinear(
          world->getSkeleton(k),
          jointPositions.block(jointSpaceCursor, i, skelDofs, 1));
      ret.block(worldSpaceCursor, i, worldSpace.size(), 1) = worldSpace;
      worldSpaceCursor += worldSpace.size();
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities)
{
  int dofs = 0;
  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
    dofs += skel->getNumBodyNodes() * 3;
  }

  Eigen::MatrixXd ret = Eigen::MatrixXd(dofs, jointVelocities.cols());
  for (int i = 0; i < jointVelocities.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      Eigen::VectorXd worldSpace = skelConvertJointSpaceVelocitiesToWorldLinear(
          world->getSkeleton(k),
          jointVelocities.block(jointSpaceCursor, i, skelDofs, 1));
      ret.block(worldSpaceCursor, i, worldSpace.size(), 1) = worldSpace;
      worldSpaceCursor += worldSpace.size();
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd backpropWorldLinearToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd worldSpaceLoss,
    bool useTranspose)
{
  Eigen::MatrixXd ret
      = Eigen::MatrixXd(world->getNumDofs(), worldSpaceLoss.cols());
  for (int i = 0; i < worldSpaceLoss.cols(); i++)
  {
    int worldSpaceCursor = 0;
    int jointSpaceCursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(k);
      int skelDofs = skel->getNumDofs();
      int skelBodyDofs = skel->getNumBodyNodes() * 3;
      Eigen::VectorXd jointSpace = skelBackpropWorldLinearToJointSpace(
          world->getSkeleton(k),
          worldSpaceLoss.block(worldSpaceCursor, i, skelBodyDofs, 1),
          useTranspose);
      ret.block(jointSpaceCursor, i, skelDofs, 1) = jointSpace;
      worldSpaceCursor += skelBodyDofs;
      jointSpaceCursor += skelDofs;
    }
  }
  return ret;
}

//==============================================================================
Eigen::MatrixXd convertJointSpacePositionsToWorldCOM(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointPositions)
{
  Eigen::MatrixXd res
      = Eigen::MatrixXd(world->getNumSkeletons() * 3, jointPositions.cols());
  for (std::size_t i = 0; i < jointPositions.cols(); i++)
  {
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::size_t dofs = world->getSkeleton(i)->getNumDofs();
      res.block(k * 3, i, 3, 1) = skelConvertJointSpacePositionsToWorldCOM(
          world->getSkeleton(k), jointPositions.block(cursor, i, dofs, 1));
      cursor += dofs;
    }
  }
  return res;
}

//==============================================================================
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldCOMLinear(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities)
{
  Eigen::MatrixXd res
      = Eigen::MatrixXd(world->getNumSkeletons() * 3, jointVelocities.cols());
  for (std::size_t i = 0; i < jointVelocities.cols(); i++)
  {
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::size_t dofs = world->getSkeleton(i)->getNumDofs();
      res.block(k * 3, i, 3, 1)
          = skelConvertJointSpaceVelocitiesToWorldCOMLinear(
              world->getSkeleton(k), jointVelocities.block(cursor, i, dofs, 1));
      cursor += dofs;
    }
  }
  return res;
}

//==============================================================================
Eigen::MatrixXd backpropWorldCOMLinearToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd lossWrtWorldCOM,
    bool useTranspose)
{
  Eigen::MatrixXd res
      = Eigen::MatrixXd(lossWrtWorldCOM.cols(), world->getNumDofs());
  for (std::size_t i = 0; i < lossWrtWorldCOM.cols(); i++)
  {
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::size_t dofs = world->getSkeleton(i)->getNumDofs();
      res.block(cursor, i, dofs, 1) = skelBackpropWorldCOMLinearToJointSpace(
          world->getSkeleton(k),
          lossWrtWorldCOM.block(k * 3, i, 3, 1),
          useTranspose);
      cursor += dofs;
    }
  }
  return res;
}

//==============================================================================
Eigen::MatrixXd convertJointSpaceVelocitiesToWorldCOMSpatial(
    std::shared_ptr<simulation::World> world, Eigen::MatrixXd jointVelocities)
{
  Eigen::MatrixXd res
      = Eigen::MatrixXd(world->getNumSkeletons() * 6, jointVelocities.cols());
  for (std::size_t i = 0; i < jointVelocities.cols(); i++)
  {
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::size_t dofs = world->getSkeleton(i)->getNumDofs();
      res.block(k * 6, i, 6, 1)
          = skelConvertJointSpaceVelocitiesToWorldCOMSpatial(
              world->getSkeleton(k), jointVelocities.block(cursor, i, dofs, 1));
      cursor += dofs;
    }
  }
  return res;
}

//==============================================================================
Eigen::MatrixXd backpropWorldCOMSpatialToJointSpace(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd lossWrtWorldCOM,
    bool useTranspose)
{
  Eigen::MatrixXd res
      = Eigen::MatrixXd(lossWrtWorldCOM.cols(), world->getNumDofs());
  for (std::size_t i = 0; i < lossWrtWorldCOM.cols(); i++)
  {
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < world->getNumSkeletons(); k++)
    {
      std::size_t dofs = world->getSkeleton(i)->getNumDofs();
      res.block(cursor, i, dofs, 1) = skelBackpropWorldCOMSpatialToJointSpace(
          world->getSkeleton(k),
          lossWrtWorldCOM.block(k * 6, i, 6, 1),
          useTranspose);
      cursor += dofs;
    }
  }
  return res;
}

} // namespace neural
} // namespace dart