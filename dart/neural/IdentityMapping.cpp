#include "dart/neural/IdentityMapping.hpp"

#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
IdentityMapping::IdentityMapping(std::shared_ptr<simulation::World> world)
{
  mNumDofs = world->getNumDofs();
}

//==============================================================================
int IdentityMapping::getPosDim()
{
  return mNumDofs;
}

//==============================================================================
int IdentityMapping::getVelDim()
{
  return mNumDofs;
}

//==============================================================================
int IdentityMapping::getForceDim()
{
  return mNumDofs;
}

//==============================================================================
void IdentityMapping::setPositions(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& positions)
{
  world->setPositions(positions);
}

//==============================================================================
void IdentityMapping::setVelocities(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& velocities)
{
  world->setVelocities(velocities);
}

//==============================================================================
void IdentityMapping::setForces(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXd>& forces)
{
  world->setForces(forces);
}

//==============================================================================
void IdentityMapping::getPositionsInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> positions)
{
  positions = world->getPositions();
}

//==============================================================================
void IdentityMapping::getVelocitiesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> velocities)
{
  velocities = world->getVelocities();
}

//==============================================================================
void IdentityMapping::getForcesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> forces)
{
  forces = world->getForces();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer positions (the
/// "mapped" positions) to inner positions (the "real" positions)
Eigen::MatrixXd IdentityMapping::getMappedPosToRealPosJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner positions (the
/// "real" positions) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXd IdentityMapping::getRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocities (the
/// "real" velocities) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXd IdentityMapping::getRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer velocity (the
/// "mapped" velocity) to inner velocity (the "real" velocity)
Eigen::MatrixXd IdentityMapping::getMappedVelToRealVelJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocity (the
/// "real" velocity) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXd IdentityMapping::getRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner position (the
/// "real" position) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXd IdentityMapping::getRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Zero(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the outer force (the
/// "mapped" force) to inner force (the "real" force)
Eigen::MatrixXd IdentityMapping::getMappedForceToRealForceJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner force (the
/// "real" force) to the corresponding outer force (the "mapped"
/// force)
Eigen::MatrixXd IdentityMapping::getRealForceToMappedForceJac(
    std::shared_ptr<simulation::World> world)
{
  return Eigen::MatrixXd::Identity(mNumDofs, mNumDofs);
}

} // namespace neural
} // namespace dart