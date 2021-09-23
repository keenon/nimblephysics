#include "dart/neural/IKMapping.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Frame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
IKMapping::IKMapping(std::shared_ptr<simulation::World> world)
  : mIKIterationLimit(100)
{
  mMassDim = world->getMassDims();
}

//==============================================================================
void IKMapping::setIKIterationLimit(int limit)
{
  if (limit == -1)
  {
    mIKIterationLimit = 100000;
  }
  else
  {
    mIKIterationLimit = limit;
  }
}

//==============================================================================
int IKMapping::getIKIterationLimit()
{
  return mIKIterationLimit;
}

//==============================================================================
void IKMapping::addSpatialBodyNode(dynamics::BodyNode* node)
{
  mEntries.push_back(IKMappingEntry(IKMappingEntryType::NODE_SPATIAL, node));
}

//==============================================================================
void IKMapping::addLinearBodyNode(dynamics::BodyNode* node)
{
  mEntries.push_back(IKMappingEntry(IKMappingEntryType::NODE_LINEAR, node));
}

//==============================================================================
void IKMapping::addAngularBodyNode(dynamics::BodyNode* node)
{
  mEntries.push_back(IKMappingEntry(IKMappingEntryType::NODE_ANGULAR, node));
}

//==============================================================================
int IKMapping::getPosDim()
{
  return getDim();
}

//==============================================================================
int IKMapping::getVelDim()
{
  return getDim();
}

//==============================================================================
int IKMapping::getControlForceDim()
{
  return getDim();
}

//==============================================================================
int IKMapping::getMassDim()
{
  return mMassDim;
}

//==============================================================================
// #define DART_NEURAL_LOG_IK_OUTPUT
void IKMapping::setPositions(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& positions)
{
  // Reset to 0, so that solutions are always deterministic even if IK is
  // under/over specified
  world->setPositions(Eigen::VectorXs::Zero(world->getNumDofs()));

  math::solveIK(
      Eigen::VectorXs::Zero(world->getNumDofs()),
      positions.size(),
      [world](Eigen::VectorXs pos, bool clamp) {
        world->setPositions(pos);
        if (clamp)
        {
          world->clampPositionsToLimits();
          return world->getPositions();
        }
        return pos;
      },
      [this, world, positions](Eigen::VectorXs& diff, Eigen::MatrixXs& J) {
        diff = positions - getPositions(world);
        J = getPosJacobian(world);
      },
      [](Eigen::VectorXs& pos) {
        // Don't random restart here
        (void)pos;
        assert(false);
      },
      math::IKConfig().setMaxStepCount(500).setMaxRestarts(1));
}

//==============================================================================
void IKMapping::setVelocities(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& velocities)
{
  world->setVelocities(getVelJacobianInverse(world) * velocities);
}

//==============================================================================
void IKMapping::setControlForces(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& forces)
{
  world->setControlForces(getVelJacobian(world).transpose() * forces);
}

//==============================================================================
void IKMapping::setMasses(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& masses)
{
  world->setMasses(masses);
}

//==============================================================================
void IKMapping::getPositionsInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> positions)
{
  assert(positions.size() == getPosDim());
  int cursor = 0;
  for (IKMappingEntry& entry : mEntries)
  {
    auto skel = world->getSkeleton(entry.skelName);

    if (entry.type == NODE_SPATIAL || entry.type == NODE_LINEAR
        || entry.type == NODE_ANGULAR)
    {
      // Get the body node in this world
      dynamics::BodyNode* node = skel->getBodyNode(entry.bodyNodeOffset);
      Eigen::Isometry3s T = node->getWorldTransform();
      if (entry.type == NODE_ANGULAR || entry.type == NODE_SPATIAL)
      {
        positions.segment<3>(cursor) = math::logMap(T.linear());
        cursor += 3;
      }
      if (entry.type == NODE_LINEAR || entry.type == NODE_SPATIAL)
      {
        positions.segment<3>(cursor) = T.translation();
        cursor += 3;
      }
    }
    else if (entry.type == COM)
    {
      positions.segment(cursor, 3) = skel->getCOM();
      cursor += 3;
    }
  }
  assert(cursor == getPosDim());
}

//==============================================================================
void IKMapping::getVelocitiesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> velocities)
{
  assert(velocities.size() == getVelDim());
  int cursor = 0;
  for (IKMappingEntry& entry : mEntries)
  {
    auto skel = world->getSkeleton(entry.skelName);

    if (entry.type == NODE_SPATIAL || entry.type == NODE_LINEAR
        || entry.type == NODE_ANGULAR)
    {
      // Get the body node in this world
      dynamics::BodyNode* node = skel->getBodyNode(entry.bodyNodeOffset);
      if (entry.type == NODE_SPATIAL)
      {
        /*
        velocities.segment<6>(cursor)
            = skel->getWorldJacobian(node) * skel->getVelocities();
            */
        velocities.segment<6>(cursor) = node->getSpatialVelocity(
            dynamics::Frame::World(), dynamics::Frame::World());
        cursor += 6;
      }
      else if (entry.type == NODE_ANGULAR)
      {
        velocities.segment<3>(cursor) = node->getAngularVelocity(
            dynamics::Frame::World(), dynamics::Frame::World());
        cursor += 3;
      }
      else if (entry.type == NODE_LINEAR)
      {
        velocities.segment<3>(cursor) = node->getLinearVelocity(
            dynamics::Frame::World(), dynamics::Frame::World());
        cursor += 3;
      }
      else
      {
        assert(false && "Execution should never reach here");
      }
    }
    else if (entry.type == COM)
    {
      velocities.segment<3>(cursor) = skel->getCOMLinearVelocity();
      cursor += 3;
    }
    else
    {
      assert(false && "Unrecognized entry type");
    }
  }
  assert(cursor == getVelDim());
}

//==============================================================================
void IKMapping::getControlForcesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> forces)
{
  assert(forces.size() == getControlForceDim());
  forces = getRealForceToMappedForceJac(world) * world->getControlForces();
}

//==============================================================================
void IKMapping::getMassesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> masses)
{
  assert(masses.size() == getMassDim());
  masses = world->getMasses();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner positions (the
/// "real" positions) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXs IKMapping::getRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  Eigen::MatrixXs J = getPosJacobian(world);
  if (world->getSlowDebugResultsAgainstFD())
  {
    equalsOrCrash(
        J,
        finiteDifferenceRealPosToMappedPosJac(world),
        world,
        "real pos - mapped pos");
  }
  return J;
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocities (the
/// "real" velocities) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXs IKMapping::getRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(getDim(), world->getNumDofs());
  if (world->getSlowDebugResultsAgainstFD())
  {
    equalsOrCrash(
        J,
        finiteDifferenceRealVelToMappedPosJac(world),
        world,
        "real vel - mapped pos");
  }
  return J;
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocity (the
/// "real" velocity) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXs IKMapping::getRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  Eigen::MatrixXs J = getVelJacobian(world);
  if (world->getSlowDebugResultsAgainstFD())
  {
    equalsOrCrash(
        J,
        finiteDifferenceRealVelToMappedVelJac(world),
        world,
        "real vel - mapped vel");
  }
  return J;
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner position (the
/// "real" position) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXs IKMapping::getRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  // Original formula:
  // mappedVel = getJacobian(world) * velocities()
  Eigen::MatrixXs J = getJacobianOfJacVelWrtPosition(world);
  if (world->getSlowDebugResultsAgainstFD())
  {
    equalsOrCrash(
        J,
        finiteDifferenceRealPosToMappedVelJac(world),
        world,
        "real pos - mapped vel");
  }
  return J;
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner force (the
/// "real" force) to the corresponding outer force (the "mapped"
/// force)
Eigen::MatrixXs IKMapping::getRealForceToMappedForceJac(
    std::shared_ptr<simulation::World> world)
{
  return getVelJacobianInverse(world).transpose();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner mass (the
/// "real" mass) to the corresponding outer mass (the "mapped"
/// mass)
Eigen::MatrixXs IKMapping::getRealMassToMappedMassJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Identity(mMassDim, mMassDim);
}

//==============================================================================
int IKMapping::getDim()
{
  int dim = 0;
  for (IKMappingEntry& entry : mEntries)
  {
    if (entry.type == NODE_SPATIAL)
      dim += 6;
    else
      dim += 3;
  }
  return dim;
}

//==============================================================================
/// Computes a Jacobian that transforms changes in joint angle to changes in
/// IK body positions (expressed in log space).
Eigen::MatrixXs IKMapping::getPosJacobian(
    std::shared_ptr<simulation::World> world)
{
  int rows = getDim();
  int cols = world->getNumDofs();
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(rows, cols);
  int cursor = 0;
  for (IKMappingEntry& entry : mEntries)
  {
    auto skel = world->getSkeleton(entry.skelName);
    int offset = world->getSkeletonDofOffset(skel);
    int dofs = skel->getNumDofs();

    if (entry.type == NODE_SPATIAL || entry.type == NODE_LINEAR
        || entry.type == NODE_ANGULAR)
    {
      // Get the body node in this world
      dynamics::BodyNode* node = skel->getBodyNode(entry.bodyNodeOffset);

      if (entry.type == NODE_SPATIAL)
      {
        jac.block(cursor, offset, 6, dofs)
            = skel->getWorldPositionJacobian(node);
        cursor += 6;
      }
      if (entry.type == NODE_ANGULAR)
      {
        jac.block(cursor, offset, 3, dofs)
            = skel->getWorldPositionJacobian(node).block(0, 0, 3, dofs);
        cursor += 3;
      }
      if (entry.type == NODE_LINEAR)
      {
        jac.block(cursor, offset, 3, dofs)
            = skel->getWorldPositionJacobian(node).block(3, 0, 3, dofs);
        cursor += 3;
      }
    }
    else if (entry.type == COM)
    {
      jac.block(cursor, offset, 3, dofs) = skel->getCOMLinearJacobian();
      cursor += 3;
    }
  }
  return jac;
}

/// Computes the pseudo-inverse of the Jacobian
Eigen::MatrixXs IKMapping::getPosJacobianInverse(
    std::shared_ptr<simulation::World> world)
{
  Eigen::MatrixXs J = getPosJacobian(world);
  // return math::clippedSingularsPinv(J);
  return J.completeOrthogonalDecomposition().pseudoInverse();
}

/// Computes a Jacobian that transforms changes in joint vel to changes in
/// IK body vels (expressed in log space).
Eigen::MatrixXs IKMapping::getVelJacobian(
    std::shared_ptr<simulation::World> world)
{
  int rows = getDim();
  int cols = world->getNumDofs();
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(rows, cols);
  int cursor = 0;
  for (IKMappingEntry& entry : mEntries)
  {
    auto skel = world->getSkeleton(entry.skelName);
    int offset = world->getSkeletonDofOffset(skel);
    int dofs = skel->getNumDofs();

    if (entry.type == NODE_SPATIAL || entry.type == NODE_LINEAR
        || entry.type == NODE_ANGULAR)
    {
      // Get the body node in this world
      dynamics::BodyNode* node = skel->getBodyNode(entry.bodyNodeOffset);

      if (entry.type == NODE_SPATIAL)
      {
        jac.block(cursor, offset, 6, dofs) = skel->getWorldJacobian(node);
        cursor += 6;
      }
      if (entry.type == NODE_ANGULAR)
      {
        jac.block(cursor, offset, 3, dofs)
            = skel->getWorldJacobian(node).block(0, 0, 3, dofs);
        cursor += 3;
      }
      if (entry.type == NODE_LINEAR)
      {
        jac.block(cursor, offset, 3, dofs)
            = skel->getWorldJacobian(node).block(3, 0, 3, dofs);
        cursor += 3;
      }
    }
    else if (entry.type == COM)
    {
      jac.block(cursor, offset, 3, dofs) = skel->getCOMLinearJacobian();
      cursor += 3;
    }
  }
  return jac;
}

/// Computes the pseudo-inverse of the vel Jacobian
Eigen::MatrixXs IKMapping::getVelJacobianInverse(
    std::shared_ptr<simulation::World> world)
{
  return getVelJacobian(world)
      .completeOrthogonalDecomposition()
      .pseudoInverse();
}

/// Computes a Jacobian of J(x)*vel wrt pos
Eigen::MatrixXs IKMapping::getJacobianOfJacVelWrtPosition(
    std::shared_ptr<simulation::World> world)
{
  // TODO: replace me with an analytical solution
  return bruteForceJacobianOfJacVelWrtPosition(world);
}

/// The brute force version of getJacobianOfJacobianWrtPosition()
Eigen::MatrixXs IKMapping::bruteForceJacobianOfJacVelWrtPosition(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalPosition = world->getPositions();
  Eigen::VectorXs originalVel = world->getVelocities();
  const s_t EPS = 1e-6;
  int n = world->getNumDofs();
  int m = getDim();
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(m, n);

  for (int i = 0; i < n; i++)
  {
    Eigen::VectorXs perturbedPosition = originalPosition;
    perturbedPosition(i) += EPS;
    world->setPositions(perturbedPosition);
    Eigen::VectorXs pos = getVelJacobian(world) * originalVel;

    perturbedPosition = originalPosition;
    perturbedPosition(i) -= EPS;
    world->setPositions(perturbedPosition);
    Eigen::VectorXs neg = getVelJacobian(world) * originalVel;

    jac.col(i) = (pos - neg) / (2 * EPS);
  }

  snapshot.restore();
  return jac;
}

//==============================================================================
Eigen::VectorXs IKMapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getControlForceLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getControlForceDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getControlForceUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getControlForceDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs IKMapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getMassLowerLimits();
}

//==============================================================================
Eigen::VectorXs IKMapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getMassUpperLimits();
}

} // namespace neural
} // namespace dart