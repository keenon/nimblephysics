#include "dart/neural/IdentityMapping.hpp"

#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
IdentityMapping::IdentityMapping(std::shared_ptr<simulation::World> world)
{
  mNumDofs = world->getNumDofs();
  mMassDim = world->getMassDims();
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
int IdentityMapping::getControlForceDim()
{
  return mNumDofs;
}

//==============================================================================
int IdentityMapping::getMassDim()
{
  return mMassDim;
}

//==============================================================================
void IdentityMapping::setPositions(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& positions)
{
  world->setPositions(positions);
}

//==============================================================================
void IdentityMapping::setVelocities(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& velocities)
{
  world->setVelocities(velocities);
}

//==============================================================================
void IdentityMapping::setControlForces(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& forces)
{
  world->setControlForces(forces);
}

//==============================================================================
void IdentityMapping::setMasses(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<Eigen::VectorXs>& masses)
{
  world->setMasses(masses);
}

//==============================================================================
void IdentityMapping::getPositionsInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> positions)
{
  positions = world->getPositions();
}

//==============================================================================
void IdentityMapping::getVelocitiesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> velocities)
{
  velocities = world->getVelocities();
}

//==============================================================================
void IdentityMapping::getControlForcesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> forces)
{
  forces = world->getExternalForces();
}

//==============================================================================
void IdentityMapping::getMassesInPlace(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> masses)
{
  masses = world->getMasses();
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner positions (the
/// "real" positions) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXs IdentityMapping::getRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocities (the
/// "real" velocities) to the corresponding outer positions (the "mapped"
/// positions)
Eigen::MatrixXs IdentityMapping::getRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Zero(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner velocity (the
/// "real" velocity) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXs IdentityMapping::getRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner position (the
/// "real" position) to the corresponding outer velocity (the "mapped"
/// velocity)
Eigen::MatrixXs IdentityMapping::getRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Zero(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner force (the
/// "real" force) to the corresponding outer force (the "mapped"
/// force)
Eigen::MatrixXs IdentityMapping::getRealForceToMappedForceJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Identity(mNumDofs, mNumDofs);
}

//==============================================================================
/// This gets a Jacobian relating the changes in the inner force (the
/// "real" force) to the corresponding outer force (the "mapped"
/// force)
Eigen::MatrixXs IdentityMapping::getRealMassToMappedMassJac(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::MatrixXs::Identity(mMassDim, mMassDim);
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getPositionLowerLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getPositionUpperLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getVelocityLowerLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getVelocityUpperLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getControlForceLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getExternalForceLowerLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getControlForceUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getExternalForceUpperLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getMassLowerLimits();
}

//==============================================================================
Eigen::VectorXs IdentityMapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> world)
{
  return world->getMassUpperLimits();
}

} // namespace neural
} // namespace dart