#ifndef DART_NEURAL_MAPPED_BACKPROP_SNAPSHOT_HPP_
#define DART_NEURAL_MAPPED_BACKPROP_SNAPSHOT_HPP_

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/performance/PerformanceLog.hpp"

namespace dart {

using namespace performance;

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
      std::string representation,
      std::unordered_map<std::string, PreStepMapping> preStepMappings,
      std::unordered_map<std::string, PostStepMapping> postStepMappings);

  const std::vector<std::string>& getMappings();
  const std::string& getRepresentation();

  Eigen::MatrixXd getPosPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXd getPosVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXd getVelPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXd getVelVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXd getForceVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapping,
      PerformanceLog* perfLog = nullptr);

  /// This computes the implicit backprop without forming intermediate
  /// Jacobians. It takes a LossGradient with the position and velocity vectors
  /// filled it, though the loss with respect to torque is ignored and can be
  /// null. It returns a LossGradient with all three values filled in, position,
  /// velocity, and torque.
  void backprop(
      simulation::WorldPtr world,
      LossGradient& thisTimestepLoss,
      const std::unordered_map<std::string, LossGradient> nextTimestepLosses,
      PerformanceLog* perfLog = nullptr);

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  const Eigen::VectorXd& getPreStepPosition(const std::string& mapping);

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  const Eigen::VectorXd& getPreStepVelocity(const std::string& mapping);

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, BEFORE the timestep.
  const Eigen::VectorXd& getPreStepTorques(const std::string& mapping);

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  const Eigen::VectorXd& getPostStepPosition(const std::string& mapping);

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  const Eigen::VectorXd& getPostStepVelocity(const std::string& mapping);

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, AFTER the timestep.
  const Eigen::VectorXd& getPostStepTorques(const std::string& mapping);

protected:
  std::shared_ptr<BackpropSnapshot> mBackpropSnapshot;
  std::string mRepresentation;
  std::vector<std::string> mMappings;
  std::unordered_map<std::string, PreStepMapping> mPreStepMappings;
  std::unordered_map<std::string, PostStepMapping> mPostStepMappings;
};

using MappedBackpropSnapshotPtr = std::shared_ptr<MappedBackpropSnapshot>;

} // namespace neural
} // namespace dart

#endif