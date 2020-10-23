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
Eigen::VectorXd WithRespectToPosition::get(
    std::shared_ptr<simulation::World> world)
{
  return world->getPositions();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXd WithRespectToPosition::get(dynamics::Skeleton* skel)
{
  return skel->getPositions();
}

/// This sets the world's state based on our WRT
void WithRespectToPosition::set(
    std::shared_ptr<simulation::World> world, Eigen::VectorXd value)
{
  world->setPositions(value);
}

/// This sets the world's state based on our WRT
void WithRespectToPosition::set(dynamics::Skeleton* skel, Eigen::VectorXd value)
{
  skel->setPositions(value);
}

/// This gives the dimensions of the WRT
int WithRespectToPosition::dim(std::shared_ptr<simulation::World> world)
{
  return world->getNumDofs();
}

/// This gives the dimensions of the WRT
int WithRespectToPosition::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// Basic constructor
WithRespectToVelocity::WithRespectToVelocity()
{
}

/// This returns this WRT from the world as a vector
Eigen::VectorXd WithRespectToVelocity::get(
    std::shared_ptr<simulation::World> world)
{
  return world->getVelocities();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXd WithRespectToVelocity::get(dynamics::Skeleton* skel)
{
  return skel->getVelocities();
}

/// This sets the world's state based on our WRT
void WithRespectToVelocity::set(
    std::shared_ptr<simulation::World> world, Eigen::VectorXd value)
{
  world->setVelocities(value);
}

/// This sets the world's state based on our WRT
void WithRespectToVelocity::set(dynamics::Skeleton* skel, Eigen::VectorXd value)
{
  skel->setVelocities(value);
}

/// This gives the dimensions of the WRT
int WithRespectToVelocity::dim(std::shared_ptr<simulation::World> world)
{
  return world->getNumDofs();
}

/// This sets the world's state based on our WRT
int WithRespectToVelocity::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

/// Basic constructor
WithRespectToForce::WithRespectToForce()
{
}

/// This returns this WRT from the world as a vector
Eigen::VectorXd WithRespectToForce::get(
    std::shared_ptr<simulation::World> world)
{
  return world->getForces();
}

/// This returns this WRT from the world as a vector
Eigen::VectorXd WithRespectToForce::get(dynamics::Skeleton* skel)
{
  return skel->getForces();
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(
    std::shared_ptr<simulation::World> world, Eigen::VectorXd value)
{
  world->setForces(value);
}

/// This sets the world's state based on our WRT
void WithRespectToForce::set(dynamics::Skeleton* skel, Eigen::VectorXd value)
{
  skel->setForces(value);
}

/// This gives the dimensions of the WRT
int WithRespectToForce::dim(std::shared_ptr<simulation::World> world)
{
  return world->getNumDofs();
}

/// This gives the dimensions of the WRT
int WithRespectToForce::dim(dynamics::Skeleton* skel)
{
  return skel->getNumDofs();
}

} // namespace neural
} // namespace dart