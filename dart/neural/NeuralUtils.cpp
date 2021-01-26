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
  Eigen::VectorXd preStepTorques = world->getExternalForces();
  Eigen::VectorXd preStepLCPCache = world->getCachedLCPSolution();

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
          world->getLastPreConstraintVelocity(),
          preStepLCPCache);

  if (idempotent)
    restorableSnapshot->restore();

  return snapshot;
}

//==============================================================================
/// Takes a step in the world, and returns a mapped snapshot which can be used
/// to backpropagate gradients and compute Jacobians in the mapped space
std::shared_ptr<MappedBackpropSnapshot> mappedForwardPass(
    std::shared_ptr<simulation::World> world,
    std::string representation,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> mappings,
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
  Eigen::VectorXd preStepTorques = world->getExternalForces();
  Eigen::VectorXd preStepLCPCache = world->getCachedLCPSolution();

  // Record the Jacobians for mapping out from mapped space to world space
  // pre-step
  std::unordered_map<std::string, PreStepMapping> preStepMappings;
  for (std::pair<std::string, std::shared_ptr<Mapping>> lossMap : mappings)
  {
    PreStepMapping pre(world, lossMap.second);
    preStepMappings[lossMap.first] = pre;
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
          world->getLastPreConstraintVelocity(),
          preStepLCPCache);

  // Record the Jacobians for mapping out from mapped space to world space
  // pre-step
  std::unordered_map<std::string, PostStepMapping> postStepMappings;
  for (std::pair<std::string, std::shared_ptr<Mapping>> lossMap : mappings)
  {
    PostStepMapping post(world, lossMap.second);
    postStepMappings[lossMap.first] = post;
  }

  if (idempotent)
    restorableSnapshot->restore();

  return std::make_shared<MappedBackpropSnapshot>(
      snapshot, representation, preStepMappings, postStepMappings);
}

//==============================================================================
Eigen::MatrixXd jointPosToWorldSpatialJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nodes.size() * 6, dofs);
  for (int i = 0; i < nodes.size(); i++)
  {
    jac.block(i * 6, 0, 6, dofs) = skel->getWorldPositionJacobian(
        skel->getBodyNode(nodes[i]->getIndexInSkeleton()));
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd jointPosToWorldLinearJacobian(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<dynamics::BodyNode*>& nodes)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nodes.size() * 3, dofs);
  for (int i = 0; i < nodes.size(); i++)
  {
    jac.block(i * 3, 0, 3, dofs)
        = skel->getWorldPositionJacobian(
                  skel->getBodyNode(nodes[i]->getIndexInSkeleton()))
              .block(3, 0, 3, jac.cols());
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd jointVelToWorldSpatialJacobian(
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
Eigen::MatrixXd jointVelToWorldLinearJacobian(
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
    Eigen::MatrixXd jac = jointVelToWorldSpatialJacobian(skel, nodes);
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
  if (space == ConvertToSpace::POS_LINEAR)
  {
    jac = jointPosToWorldLinearJacobian(skel, nodes);
  }
  else if (space == ConvertToSpace::VEL_LINEAR)
  {
    jac = jointVelToWorldLinearJacobian(skel, nodes);
  }
  else if (space == ConvertToSpace::POS_SPATIAL)
  {
    jac = jointPosToWorldSpatialJacobian(skel, nodes);
  }
  else if (space == ConvertToSpace::VEL_SPATIAL)
  {
    jac = jointVelToWorldSpatialJacobian(skel, nodes);
  }
  else if (space == ConvertToSpace::COM_POS)
  {
    jac = skel->getCOMPositionJacobian().bottomRows<3>();
  }
  else if (space == ConvertToSpace::COM_VEL_LINEAR)
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