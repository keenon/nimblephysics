#include "dart/neural/WithRespectTo.hpp"

#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

WithRespectToPosition* WithRespectTo::POSITION = new WithRespectToPosition();
WithRespectToVelocity* WithRespectTo::VELOCITY = new WithRespectToVelocity();
WithRespectToForce* WithRespectTo::FORCE = new WithRespectToForce();

/// Basic destructor
WithRespectTo::~WithRespectTo()
{
}

/// Basic constructor
WithRespectToPosition::WithRespectToPosition()
{
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
WithRespectToForce::WithRespectToForce()
{
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToForce::get(simulation::World* world)
{
  return world->getExternalForces();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToForce::get(dynamics::Skeleton* skel)
{
  return skel->getForces();
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(simulation::World* world, Eigen::VectorXs value)
{
  world->setExternalForces(value);
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  skel->setForces(value);
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
  return world->getExternalForceUpperLimits();
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToForce::lowerBound(simulation::World* world)
{
  return world->getExternalForceLowerLimits();
}

} // namespace neural
} // namespace dart