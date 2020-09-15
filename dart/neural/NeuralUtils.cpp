#include "dart/neural/NeuralUtils.hpp"

#include <memory>
#include <thread>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/MappedBackpropSnapshot.hpp"
#include "dart/neural/Mapping.hpp"
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
          world,
          preStepPosition,
          preStepVelocity,
          preStepTorques,
          world->getLastPreConstraintVelocity());

  if (idempotent)
    restorableSnapshot->restore();

  return snapshot;
}

//==============================================================================
/// Takes a step in the world, and returns a mapped snapshot which can be used
/// to backpropagate gradients and compute Jacobians in the mapped space
std::shared_ptr<MappedBackpropSnapshot> forwardPass(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<Mapping> representationMapping,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> lossMappings,
    bool idempotent)
{
  std::shared_ptr<RestorableSnapshot> restorableSnapshot;
  if (idempotent)
  {
    restorableSnapshot = std::make_shared<RestorableSnapshot>(world);
  }

  // Record the current input vector in mapped space
  Eigen::VectorXd preStepPosition = world->getPositions();
  Eigen::VectorXd preStepVelocity = world->getVelocities();
  Eigen::VectorXd preStepTorques = world->getForces();

  // Record the Jacobians for mapping out from mapped space to world space
  // pre-step
  PreStepMapping preStepRepresentation
      = PreStepMapping(world, representationMapping);
  std::unordered_map<std::string, PreStepMapping> preStepLosses;
  for (std::pair<std::string, std::shared_ptr<Mapping>> lossMap : lossMappings)
  {
    PreStepMapping pre(world, lossMap.second);
    preStepLosses[lossMap.first] = pre;
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
          world,
          preStepPosition,
          preStepVelocity,
          preStepTorques,
          world->getLastPreConstraintVelocity());

  // Record the Jacobians for mapping out from mapped space to world space
  // pre-step
  PostStepMapping postStepRepresentation
      = PostStepMapping(world, representationMapping);
  std::unordered_map<std::string, PostStepMapping> postStepLosses;
  for (std::pair<std::string, std::shared_ptr<Mapping>> lossMap : lossMappings)
  {
    PostStepMapping post(world, lossMap.second);
    postStepLosses[lossMap.first] = post;
  }

  if (idempotent)
    restorableSnapshot->restore();

  return std::make_shared<MappedBackpropSnapshot>(
      snapshot,
      preStepRepresentation,
      postStepRepresentation,
      preStepLosses,
      postStepLosses);
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
  result.postStepPoses = Eigen::MatrixXd::Zero(dofs, torques.size());
  result.postStepVels = Eigen::MatrixXd::Zero(dofs, torques.size());
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
  result.gradWrtPreStepKnotPoses = Eigen::MatrixXd::Zero(dofs, numKnots);
  result.gradWrtPreStepKnotVels = Eigen::MatrixXd::Zero(dofs, numKnots);
  result.gradWrtPreStepTorques = Eigen::MatrixXd::Zero(dofs, snapshots.size());
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
    graph looks like two separate processes:

    p_t -------------+
                      \
                       \
    v_t ----------------+----(LCP Solver)----> v_t+1 ---->
                       /
                      /
    f_t -------------+

    v_t -------------+
                      \
                       \
    p_t ----------------+--------------------> p_t+1 ---->
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
        // This is not always identity, because of bounces
        Eigen::MatrixXd posnext_pos
            = snapshots[i]->getPosPosJacobian(threadWorld);
        // p_t+1 <-- v_t
        // This is not always dt * identity, because of bounces
        Eigen::MatrixXd posnext_vel
            = snapshots[i]->getVelPosJacobian(threadWorld);

        // p_end <-- f_t = (p_end <-- v_t+1) * (v_t+1 <-- f_t)
        Eigen::MatrixXd posend_force = (posend_velnext * velnext_force);
        // v_end <-- f_t = (v_end <-- v_t+1) * (v_t+1 <-- f_t)
        Eigen::MatrixXd velend_force = (velend_velnext * velnext_force);

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
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nodes.size() * 6, dofs);
  for (int i = 0; i < nodes.size(); i++)
  {
    jac.block(i * 6, 0, 6, dofs) = skel->getWorldJacobian(
        skel->getBodyNode(nodes[i]->getIndexInSkeleton()));
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd jointToWorldLinearJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nodes.size() * 3, dofs);
  for (int i = 0; i < nodes.size(); i++)
  {
    jac.block(i * 3, 0, 3, dofs)
        = skel->getWorldJacobian(
                  skel->getBodyNode(nodes[i]->getIndexInSkeleton()))
              .block(3, 0, 3, jac.cols());
  }
  return jac;
}

//==============================================================================
Eigen::VectorXd skelConvertJointSpaceToWorldSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXd& jointValues, /* These can be velocities or positions,
                                    depending on the value of `space` */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space)
{
  Eigen::VectorXd out;

  if (space == ConvertToSpace::POS_LINEAR
      || space == ConvertToSpace::POS_SPATIAL
      || space == ConvertToSpace::COM_POS)
  {
    Eigen::VectorXd oldPositions = skel->getPositions();
    skel->setPositions(jointValues);

    if (space == ConvertToSpace::POS_SPATIAL)
    {
      out = Eigen::VectorXd::Zero(nodes.size() * 6);
      for (std::size_t i = 0; i < nodes.size(); i++)
      {
        dynamics::BodyNode* node
            = skel->getBodyNode(nodes[i]->getIndexInSkeleton());
        out.segment(i * 6, 3)
            = math::logMap(node->getWorldTransform()).head<3>();
        out.segment((i * 6) + 3, 3) = node->getWorldTransform().translation();
      }
    }
    else if (space == ConvertToSpace::POS_LINEAR)
    {
      out = Eigen::VectorXd::Zero(nodes.size() * 3);
      for (std::size_t i = 0; i < nodes.size(); i++)
      {
        dynamics::BodyNode* node
            = skel->getBodyNode(nodes[i]->getIndexInSkeleton());
        out.segment(i * 3, 3) = node->getWorldTransform().translation();
      }
    }
    else if (space == ConvertToSpace::COM_POS)
    {
      out = skel->getCOM();
    }

    skel->setPositions(oldPositions);
  }
  else if (
      space == ConvertToSpace::VEL_LINEAR
      || space == ConvertToSpace::VEL_SPATIAL
      || space == ConvertToSpace::COM_VEL_LINEAR
      || space == ConvertToSpace::COM_VEL_SPATIAL)
  {
    Eigen::MatrixXd jac = jointToWorldSpatialJacobian(skel, nodes);
    Eigen::VectorXd spatialVel = jac * jointValues;

    if (space == ConvertToSpace::VEL_SPATIAL)
    {
      out = spatialVel;
    }
    else if (space == ConvertToSpace::VEL_LINEAR)
    {
      out = Eigen::VectorXd::Zero(spatialVel.size() / 2);
      for (std::size_t i = 0; i < nodes.size(); i++)
      {
        out.segment(i * 3, 3) = spatialVel.segment((i * 6) + 3, 3);
      }
    }
    else if (space == ConvertToSpace::COM_VEL_SPATIAL)
    {
      out = Eigen::VectorXd::Zero(6);
      double totalMass = 0.0;
      for (int i = 0; i < nodes.size(); i++)
      {
        out += nodes[i]->getMass() * spatialVel.segment(i * 6, 6);
        totalMass += nodes[i]->getMass();
      }
      out /= totalMass;
    }
    else if (space == ConvertToSpace::COM_VEL_LINEAR)
    {
      out = Eigen::VectorXd::Zero(3);
      double totalMass = 0.0;
      for (int i = 0; i < nodes.size(); i++)
      {
        out += nodes[i]->getMass() * spatialVel.segment((i * 6) + 3, 3);
        totalMass += nodes[i]->getMass();
      }
      out /= totalMass;
    }
  }
  else
  {
    assert(
        false
        && "Unrecognized space passed to skelConvertJointSpaceToWorldSpace()");
  }

  return out;
}

//==============================================================================
Eigen::VectorXd skelBackpropWorldSpaceToJointSpace(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const Eigen::VectorXd& bodySpace, /* This is the gradient in body space */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space, /* This is the source space for our gradient */
    bool useIK)
{
  Eigen::MatrixXd jac;
  if (space == ConvertToSpace::POS_LINEAR
      || space == ConvertToSpace::VEL_LINEAR)
  {
    jac = jointToWorldLinearJacobian(skel, nodes);
  }
  else if (
      space == ConvertToSpace::POS_SPATIAL
      || space == ConvertToSpace::VEL_SPATIAL)
  {
    jac = jointToWorldSpatialJacobian(skel, nodes);
  }
  else if (
      space == ConvertToSpace::COM_POS
      || space == ConvertToSpace::COM_VEL_LINEAR)
  {
    jac = skel->getCOMLinearJacobian();
  }
  else if (space == ConvertToSpace::COM_VEL_SPATIAL)
  {
    jac = skel->getCOMJacobian();
  }
  else
  {
    assert(
        false
        && "Unrecognized space passed to skelBackpropWorldSpaceToJointSpace()");
  }

  // Short circuit if we're being asked to map through an empty matrix
  if (jac.size() == 0)
  {
    return Eigen::VectorXd::Zero(0);
  }

  if (useIK)
  {
    return jac.completeOrthogonalDecomposition().solve(bodySpace);
  }
  else
  {
    return jac.transpose() * bodySpace;
  }
}

//==============================================================================
Eigen::MatrixXd convertJointSpaceToWorldSpace(
    const std::shared_ptr<simulation::World>& world,
    const Eigen::MatrixXd& in, /* These can be velocities or positions,
                                    depending on the value of `space` */
    const std::vector<dynamics::BodyNode*>& nodes,
    ConvertToSpace space,
    bool backprop,
    bool useIK /* Only relevant for backprop */)
{
  // Build a list of skeletons in order of mentions
  std::vector<dynamics::SkeletonPtr> skeletons;
  for (dynamics::BodyNode* body : nodes)
  {
    // We need to get the skeleton out of the world, instead of just trusting
    // that the body points at the right skeleton, cause we might've been
    // passed a clone() of the world in a threaded context.
    dynamics::SkeletonPtr skel
        = world->getSkeleton(body->getSkeleton()->getName());
    if (skeletons.end() == std::find(skeletons.begin(), skeletons.end(), skel))
    {
      skeletons.push_back(skel);
    }
  }
  int data_size = (space == ConvertToSpace::COM_VEL_SPATIAL
                   || space == ConvertToSpace::POS_SPATIAL
                   || space == ConvertToSpace::VEL_SPATIAL)
                      ? 6
                      : 3;
  // Create the return matrix of the correct size
  bool isCOM
      = (space == ConvertToSpace::COM_POS
         || space == ConvertToSpace::COM_VEL_LINEAR
         || space == ConvertToSpace::COM_VEL_SPATIAL);
  int rows = backprop ? world->getNumDofs()
                      : (isCOM ? skeletons.size() * data_size
                               : nodes.size() * data_size);
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(rows, in.cols());

  for (int i = 0; i < skeletons.size(); i++)
  {
    dynamics::SkeletonPtr skel = skeletons[i];
    std::size_t dofs = skeletons[i]->getNumDofs();
    std::size_t dofOffset = world->getSkeletonDofOffset(skeletons[i]);

    // Collect the nodes for this skeleton

    std::vector<dynamics::BodyNode*> skelNodes;
    std::vector<int> skelNodesOffsets;
    for (int j = 0; j < nodes.size(); j++)
    {
      if (nodes[j]->getSkeleton()->getName() == skel->getName())
      {
        skelNodes.push_back(nodes[j]);
        skelNodesOffsets.push_back(j);
      }
    }
    // The center-of-mass calculations only care about which skeletons have been
    // mentioned
    if (space == ConvertToSpace::COM_POS
        || space == ConvertToSpace::COM_VEL_LINEAR
        || space == ConvertToSpace::COM_VEL_SPATIAL)
    {
      for (std::size_t t = 0; t < in.cols(); t++)
      {
        if (backprop)
        {
          out.block(dofOffset, t, dofs, 1) = skelBackpropWorldSpaceToJointSpace(
              skeletons[i],
              in.block(i * data_size, t, data_size, 1),
              skelNodes,
              space,
              useIK);
        }
        else
        {
          out.block(i * data_size, t, data_size, 1)
              = skelConvertJointSpaceToWorldSpace(
                  skeletons[i],
                  in.block(dofOffset, t, dofs, 1),
                  skelNodes,
                  space);
        }
      }
    }
    else if (
        space == ConvertToSpace::POS_LINEAR
        || space == ConvertToSpace::POS_SPATIAL
        || space == ConvertToSpace::VEL_LINEAR
        || space == ConvertToSpace::VEL_SPATIAL)
    {
      for (std::size_t t = 0; t < in.cols(); t++)
      {
        if (backprop)
        {
          Eigen::VectorXd skelIn
              = Eigen::VectorXd::Zero(skelNodes.size() * data_size);
          for (int k = 0; k < skelNodes.size(); k++)
          {
            skelIn.segment(k * data_size, data_size)
                = in.block(skelNodesOffsets[k] * data_size, t, data_size, 1);
          }

          out.block(dofOffset, t, dofs, 1) = skelBackpropWorldSpaceToJointSpace(
              skeletons[i], skelIn, skelNodes, space, useIK);
        }
        else
        {
          Eigen::VectorXd skelOut = skelConvertJointSpaceToWorldSpace(
              skeletons[i], in.block(dofOffset, t, dofs, 1), skelNodes, space);
          for (int k = 0; k < skelNodes.size(); k++)
          {
            out.block(skelNodesOffsets[k] * data_size, t, data_size, 1)
                = skelOut.segment(k * data_size, data_size);
          }
        }
      }
    }
    else
    {
      assert(
          false
          && "Unrecognized space passed to convertJointSpaceToWorldSpace()");
    }
  }
  return out;
}

} // namespace neural
} // namespace dart