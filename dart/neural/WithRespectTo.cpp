#include "dart/neural/WithRespectTo.hpp"

#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

WithRespectToPosition* WithRespectTo::POSITION = new WithRespectToPosition();
WithRespectToVelocity* WithRespectTo::VELOCITY = new WithRespectToVelocity();
WithRespectToForce* WithRespectTo::FORCE = new WithRespectToForce();
WithRespectToAcceleration* WithRespectTo::ACCELERATION
    = new WithRespectToAcceleration();
WithRespectToGroupScales* WithRespectTo::GROUP_SCALES
    = new WithRespectToGroupScales();
WithRespectToGroupMasses* WithRespectTo::GROUP_MASSES
    = new WithRespectToGroupMasses();
WithRespectToLinearizedMasses* WithRespectTo::LINEARIZED_MASSES
    = new WithRespectToLinearizedMasses();
WithRespectToGroupCOMs* WithRespectTo::GROUP_COMS
    = new WithRespectToGroupCOMs();
WithRespectToGroupInertias* WithRespectTo::GROUP_INERTIAS
    = new WithRespectToGroupInertias();

/// Basic destructor
WithRespectTo::~WithRespectTo()
{
}

/// Basic constructor
WithRespectToPosition::WithRespectToPosition()
{
}

/// A printable name for this WRT object
std::string WithRespectToPosition::name()
{
  return "POSITION";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToPosition::get(simulation::World* world)
{
  return world->getPositions();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToPosition::get(dynamics::Skeleton* skel)
{
  return skel->getPositions();
}

/// This sets the world's state based on our WRT
void WithRespectToPosition::set(simulation::World* world, Eigen::VectorXs value)
{
  world->setPositions(value);
}

/// This sets the world's state based on our WRT
void WithRespectToPosition::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setPositions(value);
}

/// This gives the dimensions of the WRT
int WithRespectToPosition::dim(simulation::World* world)
{
  return world->getNumDofs();
}

/// This gives the dimensions of the WRT
int WithRespectToPosition::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToPosition::upperBound(simulation::World* world)
{
  return world->getPositionUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToPosition::lowerBound(simulation::World* world)
{
  return world->getPositionLowerLimits();
}

/// Basic constructor
WithRespectToVelocity::WithRespectToVelocity()
{
}

/// A printable name for this WRT object
std::string WithRespectToVelocity::name()
{
  return "VELOCITY";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToVelocity::get(simulation::World* world)
{
  return world->getVelocities();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToVelocity::get(dynamics::Skeleton* skel)
{
  return skel->getVelocities();
}

/// This sets the world's state based on our WRT
void WithRespectToVelocity::set(simulation::World* world, Eigen::VectorXs value)
{
  world->setVelocities(value);
}

/// This sets the world's state based on our WRT
void WithRespectToVelocity::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setVelocities(value);
}

/// This gives the dimensions of the WRT
int WithRespectToVelocity::dim(simulation::World* world)
{
  return world->getNumDofs();
}

/// This sets the world's state based on our WRT
int WithRespectToVelocity::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToVelocity::upperBound(simulation::World* world)
{
  return world->getVelocityUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToVelocity::lowerBound(simulation::World* world)
{
  return world->getVelocityLowerLimits();
}

/// Basic constructor
WithRespectToAcceleration::WithRespectToAcceleration()
{
}

/// A printable name for this WRT object
std::string WithRespectToAcceleration::name()
{
  return "ACCELERATION";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToAcceleration::get(simulation::World* world)
{
  return world->getAccelerations();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToAcceleration::get(dynamics::Skeleton* skel)
{
  return skel->getAccelerations();
}

/// This sets the world's state based on our WRT
void WithRespectToAcceleration::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setAccelerations(value);
}

/// This sets the world's state based on our WRT
void WithRespectToAcceleration::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setAccelerations(value);
}

/// This gives the dimensions of the WRT
int WithRespectToAcceleration::dim(simulation::World* world)
{
  return world->getNumDofs();
}

/// This sets the world's state based on our WRT
int WithRespectToAcceleration::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToAcceleration::upperBound(simulation::World* world)
{
  return world->getAccelerationUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToAcceleration::lowerBound(simulation::World* world)
{
  return world->getAccelerationLowerLimits();
}

/// Basic constructor
WithRespectToGroupScales::WithRespectToGroupScales()
{
}

/// A printable name for this WRT object
std::string WithRespectToGroupScales::name()
{
  return "GROUP_SCALES";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupScales::get(simulation::World* world)
{
  return world->getGroupScales();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupScales::get(dynamics::Skeleton* skel)
{
  return skel->getGroupScales();
}

/// This sets the world's state based on our WRT
void WithRespectToGroupScales::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setAccelerations(value);
}

/// This sets the world's state based on our WRT
void WithRespectToGroupScales::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setGroupScales(value);
}

/// This gives the dimensions of the WRT
int WithRespectToGroupScales::dim(simulation::World* world)
{
  return world->getGroupScaleDim();
}

/// This sets the world's state based on our WRT
int WithRespectToGroupScales::dim(dynamics::Skeleton* skel)
{
  return skel->getGroupScaleDim();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupScales::upperBound(simulation::World* world)
{
  return world->getGroupScalesUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupScales::lowerBound(simulation::World* world)
{
  return world->getGroupScalesLowerLimits();
}

/// Basic constructor
WithRespectToGroupMasses::WithRespectToGroupMasses()
{
}

/// A printable name for this WRT object
std::string WithRespectToGroupMasses::name()
{
  return "GROUP_MASSES";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupMasses::get(simulation::World* world)
{
  return world->getGroupMasses();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupMasses::get(dynamics::Skeleton* skel)
{
  return skel->getGroupMasses();
}

/// This sets the world's state based on our WRT
void WithRespectToGroupMasses::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setGroupMasses(value);
}

/// This sets the world's state based on our WRT
void WithRespectToGroupMasses::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setGroupMasses(value);
}

/// This gives the dimensions of the WRT
int WithRespectToGroupMasses::dim(simulation::World* world)
{
  return world->getNumScaleGroups();
}

/// This sets the world's state based on our WRT
int WithRespectToGroupMasses::dim(dynamics::Skeleton* skel)
{
  return skel->getNumScaleGroups();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupMasses::upperBound(simulation::World* world)
{
  return world->getGroupMassesUpperBound();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupMasses::lowerBound(simulation::World* world)
{
  return world->getGroupMassesLowerBound();
}

/// Basic constructor
WithRespectToLinearizedMasses::WithRespectToLinearizedMasses()
{
}

/// A printable name for this WRT object
std::string WithRespectToLinearizedMasses::name()
{
  return "LINEARIZED_MASSES";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToLinearizedMasses::get(simulation::World* world)
{
  return world->getLinearizedMasses();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToLinearizedMasses::get(dynamics::Skeleton* skel)
{
  return skel->getLinearizedMasses();
}

/// This sets the world's state based on our WRT
void WithRespectToLinearizedMasses::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setLinearizedMasses(value);
}

/// This sets the world's state based on our WRT
void WithRespectToLinearizedMasses::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setLinearizedMasses(value);
}

/// This gives the dimensions of the WRT
int WithRespectToLinearizedMasses::dim(simulation::World* world)
{
  return world->getNumScaleGroups() + world->getNumSkeletons();
}

/// This sets the world's state based on our WRT
int WithRespectToLinearizedMasses::dim(dynamics::Skeleton* skel)
{
  return skel->getNumScaleGroups() + 1;
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToLinearizedMasses::upperBound(
    simulation::World* world)
{
  return world->getLinearizedMassesUpperBound();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToLinearizedMasses::lowerBound(
    simulation::World* world)
{
  return world->getLinearizedMassesLowerBound();
}

/// Basic constructor
WithRespectToGroupCOMs::WithRespectToGroupCOMs()
{
}

/// A printable name for this WRT object
std::string WithRespectToGroupCOMs::name()
{
  return "GROUP_COMS";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupCOMs::get(simulation::World* world)
{
  return world->getGroupCOMs();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupCOMs::get(dynamics::Skeleton* skel)
{
  return skel->getGroupCOMs();
}

/// This sets the world's state based on our WRT
void WithRespectToGroupCOMs::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setGroupCOMs(value);
}

/// This sets the world's state based on our WRT
void WithRespectToGroupCOMs::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setGroupCOMs(value);
}

/// This gives the dimensions of the WRT
int WithRespectToGroupCOMs::dim(simulation::World* world)
{
  return world->getNumScaleGroups() * 3;
}

/// This sets the world's state based on our WRT
int WithRespectToGroupCOMs::dim(dynamics::Skeleton* skel)
{
  return skel->getNumScaleGroups() * 3;
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupCOMs::upperBound(simulation::World* world)
{
  return world->getGroupCOMUpperBound();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupCOMs::lowerBound(simulation::World* world)
{
  return world->getGroupCOMLowerBound();
}

/// Basic constructor
WithRespectToGroupInertias::WithRespectToGroupInertias()
{
}

/// A printable name for this WRT object
std::string WithRespectToGroupInertias::name()
{
  return "GROUP_INERTIAS";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupInertias::get(simulation::World* world)
{
  return world->getGroupInertias();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToGroupInertias::get(dynamics::Skeleton* skel)
{
  return skel->getGroupInertias();
}

/// This sets the world's state based on our WRT
void WithRespectToGroupInertias::set(
    simulation::World* world, Eigen::VectorXs value)
{
  world->setGroupInertias(value);
}

/// This sets the world's state based on our WRT
void WithRespectToGroupInertias::set(
    dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setGroupInertias(value);
}

/// This gives the dimensions of the WRT
int WithRespectToGroupInertias::dim(simulation::World* world)
{
  return world->getNumScaleGroups() * 6;
}

/// This sets the world's state based on our WRT
int WithRespectToGroupInertias::dim(dynamics::Skeleton* skel)
{
  return skel->getNumScaleGroups() * 6;
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupInertias::upperBound(simulation::World* world)
{
  return world->getGroupInertiasUpperBound();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToGroupInertias::lowerBound(simulation::World* world)
{
  return world->getGroupInertiasLowerBound();
}

/// Basic constructor
WithRespectToForce::WithRespectToForce()
{
}

/// A printable name for this WRT object
std::string WithRespectToForce::name()
{
  return "FORCE";
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToForce::get(simulation::World* world)
{
  return world->getControlForces();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToForce::get(dynamics::Skeleton* skel)
{
  return skel->getControlForces();
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(simulation::World* world, Eigen::VectorXs value)
{
  world->setControlForces(value);
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setControlForces(value);
}

/// This gives the dimensions of the WRT
int WithRespectToForce::dim(simulation::World* world)
{
  return world->getNumDofs();
}

/// This gives the dimensions of the WRT
int WithRespectToForce::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToForce::upperBound(simulation::World* world)
{
  return world->getControlForceUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToForce::lowerBound(simulation::World* world)
{
  return world->getControlForceLowerLimits();
}

} // namespace neural
} // namespace dart