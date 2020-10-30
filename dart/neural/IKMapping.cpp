#include "dart/neural/IKMapping.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Frame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
IKMapping::IKMapping(std::shared_ptr<simulation::World> world)
{
  mMassDim = world->getWrtMass()->dim(world);
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
int IKMapping::getForceDim()
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
    const Eigen::Ref<Eigen::VectorXd>& positions)
{
  // Reset to 0, so that solutions are always deterministic even if IK is
  // under/over specified
  world->setPositions(Eigen::VectorXd::Zero(world->getNumDofs()));
  // Run simple IK to try to get as close as possible. Completely possible that
  // the requested positions are infeasible, in which case we'll just do a best
  // guess.
  double error = std::numeric_limits<double>::infinity();
  double lr = 1.0;
  for (int i = 0; i < 30; i++)
  {
    Eigen::VectorXd diff = positions - getPositions(world);
    double newError = diff.squaredNorm();
    double errorChange = newError - error;
    error = newError;
#ifdef DART_NEURAL_LOG_IK_OUTPUT
    std::cout << "IK iteration " << i << " loss: " << error
              << " change: " << errorChange << std::endl;
#endif
    if (error < 1e-21)
    {
#ifdef DART_NEURAL_LOG_IK_OUTPUT
      std::cout << "Terminating IK search after " << i
                << " iterations with loss: " << error << std::endl;
#endif
      break;
    }
    if (errorChange > 0)
    {
      lr *= 0.5;
    }
    else if (errorChange > -1e-22)
    {
#ifdef DART_NEURAL_LOG_IK_OUTPUT
      std::cout << "Terminating IK search after " << i
                << " iterations with optimal loss: " << error << std::endl;
#endif
      break;
    }
    Eigen::VectorXd delta
        = getJacobian(world).completeOrthogonalDecomposition().solve(diff);
    world->setPositions(world->getPositions() + (lr * delta));
  }
#ifdef DART_NEURAL_LOG_IK_OUTPUT
  std::cout << "Finished IK search with loss: " << error << std::endl;
#endif
}

//==============================================================================
void IKMapping::setVelocities(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& velocities)
{
  world->setVelocities(getMappedVelToRealVelJac(world) * velocities);
}

//==============================================================================
void IKMapping::setForces(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& forces)
{
  world->setForces(getMappedForceToRealForceJac(world) * forces);
}

//==============================================================================
void IKMapping::setMasses(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& masses)
{
  world->getWrtMass()->set(world, masses);
}

//==============================================================================
void IKMapping::getPositionsInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> positions)
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
      if (entry.type == NODE_ANGULAR || entry.type == NODE_SPATIAL)
      {
        positions.segment<3>(cursor)
            = math::logMap(node->getWorldTransform()).head<3>();
        cursor += 3;
      }
      if (entry.type == NODE_LINEAR || entry.type == NODE_SPATIAL)
      {
        positions.segment<3>(cursor) = node->getWorldTransform().translation();
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
    /* OUT */ Eigen::Ref<Eigen::VectorXd> velocities)
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
void IKMapping::getForcesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> forces)
{
  assert(forces.size() == getForceDim());
  forces = getRealForceToMappedForceJac(world) * world->getForces();
}

//==============================================================================
void IKMapping::getMassesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> masses)
{
  assert(masses.size() == getMassDim());
  masses = world->getWrtMass()->get(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer positions (the
/// "mapped" positions) to inner positions (the "real" positions)
Eigen::MatrixXd IKMapping::getMappedPosToRealPosJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobianInverse(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner positions (the
/// "real" positions) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXd IKMapping::getRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobian(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocities (the
/// "real" velocities) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXd IKMapping::getRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Zero(getDim(), world->getNumDofs());
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer velocity (the
/// "mapped" velocity) to inner velocity (the "real" velocity)
Eigen::MatrixXd IKMapping::getMappedVelToRealVelJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobianInverse(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocity (the
/// "real" velocity) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXd IKMapping::getRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobian(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner position (the
/// "real" position) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXd IKMapping::getRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  // Original formula:
  // mappedVel = getJacobian(world) * velocities()
  return getJacobianOfJacVelWrtPosition(world);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer force (the
/// "mapped" force) to inner force (the "real" force)
Eigen::MatrixXd IKMapping::getMappedForceToRealForceJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobian(world).transpose();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner force (the
/// "real" force) to the corresponding outer force (the "mapped"
/// force)
Eigen::MatrixXd IKMapping::getRealForceToMappedForceJac(
    std::shared_ptr<simulation::World> world)
{
  return getJacobianInverse(world).transpose();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer mass (the
/// "mapped" mass) to inner mass (the "real" mass)
Eigen::MatrixXd IKMapping::getMappedMassToRealMassJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mMassDim, mMassDim);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner mass (the
/// "real" mass) to the corresponding outer mass (the "mapped"
/// mass)
Eigen::MatrixXd IKMapping::getRealMassToMappedMassJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mMassDim, mMassDim);
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
Eigen::MatrixXd IKMapping::getJacobian(std::shared_ptr<simulation::World> world)
{
  int rows = getDim();
  int cols = world->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(rows, cols);
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

/// Computes the pseudo-inverse of the Jacobian
Eigen::MatrixXd IKMapping::getJacobianInverse(
    std::shared_ptr<simulation::World> world)
{
  return getJacobian(world).completeOrthogonalDecomposition().pseudoInverse();
}

/// Computes a Jacobian of J(x)*vel wrt pos
Eigen::MatrixXd IKMapping::getJacobianOfJacVelWrtPosition(
    std::shared_ptr<simulation::World> world)
{
  // TODO: replace me with an analytical solution
  return bruteForceJacobianOfJacVelWrtPosition(world);
}

/// The brute force version of getJacobianOfJacobianWrtPosition()
Eigen::MatrixXd IKMapping::bruteForceJacobianOfJacVelWrtPosition(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXd originalPosition = world->getPositions();
  Eigen::VectorXd originalVel = world->getVelocities();
  const double EPS = 1e-6;
  int n = world->getNumDofs();
  int m = getDim();
  Eigen::MatrixXd jac = Eigen::MatrixXd(m, n);

  for (int i = 0; i < n; i++)
  {
    Eigen::VectorXd perturbedPosition = originalPosition;
    perturbedPosition(i) += EPS;
    world->setPositions(perturbedPosition);
    Eigen::VectorXd pos = getJacobian(world) * originalVel;

    perturbedPosition = originalPosition;
    perturbedPosition(i) -= EPS;
    world->setPositions(perturbedPosition);
    Eigen::VectorXd neg = getJacobian(world) * originalVel;

    jac.col(i) = (pos - neg) / (2 * EPS);
  }

  snapshot.restore();
  return jac;
}

//==============================================================================
Eigen::VectorXd IKMapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getPosDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getPosDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getVelDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getVelDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getForceLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getForceDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getForceUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::VectorXd::Ones(getForceDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd IKMapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getWrtMass()->lowerBound(world);
}

//==============================================================================
Eigen::VectorXd IKMapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getWrtMass()->upperBound(world);
}

} // namespace neural
} // namespace dart