#ifndef DART_NEURAL_MAPPED_BACKPROP_SNAPSHOT_HPP_
#define DART_NEURAL_MAPPED_BACKPROP_SNAPSHOT_HPP_

#include <memory>
#include <optional>
#include <unordered_map>

#include <Eigen/Dense>

#include "dart/neural/Mapping.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace neural {

class BackpropSnapshot;

// Before we take a step, we need to map "out" of the mapped space and back into
// world space. Then we can take our step in world space, and map back "in" to
// the mapped space.
struct PreStepMapping
{
  Eigen::VectorXd pos;
  Eigen::MatrixXd posOutJac;

  Eigen::VectorXd vel;
  Eigen::MatrixXd velOutJac;

  Eigen::VectorXd force;
  Eigen::MatrixXd forceOutJac;
  Eigen::MatrixXd forceInJac;

  PreStepMapping(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<Mapping> mapping)
  {
    pos = mapping->getPositions(world);
    posOutJac = mapping->getMappedPosToRealPosJac(world);

    vel = mapping->getVelocities(world);
    velOutJac = mapping->getMappedVelToRealVelJac(world);

    force = mapping->getForces(world);
    forceOutJac = mapping->getMappedForceToRealForceJac(world);
    forceInJac = mapping->getRealForceToMappedForceJac(world);
  }

  PreStepMapping(){};
};

// After we take a step, we need to map "in" to the mapped space, from world
// space where we took the step.
struct PostStepMapping
{
  Eigen::VectorXd pos;
  Eigen::MatrixXd posInJacWrtPos;
  Eigen::MatrixXd posInJacWrtVel;

  Eigen::VectorXd vel;
  Eigen::MatrixXd velInJacWrtPos;
  Eigen::MatrixXd velInJacWrtVel;

  PostStepMapping(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<Mapping> mapping)
  {
    pos = mapping->getPositions(world);
    posInJacWrtPos = mapping->getRealPosToMappedPosJac(world);
    posInJacWrtVel = mapping->getRealVelToMappedPosJac(world);

    vel = mapping->getVelocities(world);
    velInJacWrtPos = mapping->getRealPosToMappedVelJac(world);
    velInJacWrtVel = mapping->getRealVelToMappedVelJac(world);
  }

  PostStepMapping(){};
};

class MappedBackpropSnapshot
{
public:
  MappedBackpropSnapshot(
      std::shared_ptr<BackpropSnapshot> backpropSnapshot,
      PreStepMapping preStepRepresentation,
      PostStepMapping postStepRepresentation,
      std::unordered_map<std::string, PreStepMapping> preStepLosses,
      std::unordered_map<std::string, PostStepMapping> postStepLosses);

  Eigen::MatrixXd getPosPosJacobian(std::shared_ptr<simulation::World> world);
  Eigen::MatrixXd getPosVelJacobian(std::shared_ptr<simulation::World> world);
  Eigen::MatrixXd getVelPosJacobian(std::shared_ptr<simulation::World> world);
  Eigen::MatrixXd getVelVelJacobian(std::shared_ptr<simulation::World> world);
  Eigen::MatrixXd getForceVelJacobian(std::shared_ptr<simulation::World> world);

protected:
  std::shared_ptr<BackpropSnapshot> mBackpropSnapshot;
  PreStepMapping mPreStepRepresentation;
  PostStepMapping mPostStepRepresentation;
  std::unordered_map<std::string, PreStepMapping> mPreStepLosses;
  std::unordered_map<std::string, PostStepMapping> mPostStepLosses;
};

} // namespace neural
} // namespace dart

#endif