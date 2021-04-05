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
  Eigen::VectorXs pos;
  Eigen::MatrixXs posOutJac;

  Eigen::VectorXs vel;
  Eigen::MatrixXs velOutJac;

  Eigen::VectorXs force;
  Eigen::MatrixXs forceOutJac;
  Eigen::MatrixXs forceInJac;

  Eigen::VectorXs mass;
  Eigen::MatrixXs massOutJac;

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

    mass = mapping->getMasses(world);
    massOutJac = mapping->getMappedMassToRealMassJac(world);
  }

  PreStepMapping(){};
};

// After we take a step, we need to map "in" to the mapped space, from world
// space where we took the step.
struct PostStepMapping
{
  Eigen::VectorXs pos;
  Eigen::MatrixXs posInJacWrtPos;
  Eigen::MatrixXs posInJacWrtVel;

  Eigen::VectorXs vel;
  Eigen::MatrixXs velInJacWrtPos;
  Eigen::MatrixXs velInJacWrtVel;

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
      std::unordered_map<std::string, std::shared_ptr<Mapping>> mappings,
      std::unordered_map<std::string, PreStepMapping> preStepMappings,
      std::unordered_map<std::string, PostStepMapping> postStepMappings);

  const std::vector<std::string>& getMappings();
  const std::string& getRepresentation();

  Eigen::MatrixXs getPosPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXs getPosVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXs getVelPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXs getVelVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXs getForceVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      PerformanceLog* perfLog = nullptr);
  Eigen::MatrixXs getMassVelJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
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
      PerformanceLog* perfLog = nullptr,
      bool exploreAlternateStrategies = false);

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  const Eigen::VectorXs& getPreStepPosition(const std::string& mapping = "identity");

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, BEFORE the timestep.
  const Eigen::VectorXs& getPreStepVelocity(const std::string& mapping = "identity");

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, BEFORE the timestep.
  const Eigen::VectorXs& getPreStepTorques(const std::string& mapping = "identity");

  /// Returns the LCP's cached solution from before the step
  const Eigen::VectorXs& getPreStepLCPCache();

  /// Returns a concatenated vector of all the Skeletons' position()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  const Eigen::VectorXs& getPostStepPosition(const std::string& mapping);

  /// Returns a concatenated vector of all the Skeletons' velocity()'s in the
  /// World, in order in which the Skeletons appear in the World's
  /// getSkeleton(i) returns them, AFTER the timestep.
  const Eigen::VectorXs& getPostStepVelocity(const std::string& mapping);

  /// Returns a concatenated vector of all the joint torques that were applied
  /// during the forward pass, AFTER the timestep.
  const Eigen::VectorXs& getPostStepTorques(const std::string& mapping);

  /// Returns the underlying BackpropSnapshot, without the mappings
  std::shared_ptr<BackpropSnapshot> getUnderlyingSnapshot();

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceVelVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      bool useRidders = true);

  /// This computes and returns the whole vel-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceRiddersVelVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter);      

  /// This computes and returns the whole pos-C(pos,vel) jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferencePosVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      bool useRidders = true);

  /// This computes and returns the whole pos-C(pos,vel) jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceRiddersPosVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter);

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceForceVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      bool useRidders = true);

  /// This computes and returns the whole force-vel jacobian by finite
  /// differences. This is SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceRiddersForceVelJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter);

  /// This computes a finite differenced Jacobian for pos_t->mapped_pos_{t+1}
  Eigen::MatrixXs finiteDifferencePosPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      std::size_t subdivisions = 20,
      bool useRidders = true);

    /// This computes a finite differenced Jacobian for pos_t->mapped_pos_{t+1}
  Eigen::MatrixXs finiteDifferenceRiddersPosPosJacobian(
      std::shared_ptr<simulation::World> world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      std::size_t subdivisions = 20);

  /// This computes and returns the whole vel-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceVelPosJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      std::size_t subdivisions = 20,
      bool useRidders = true);

  /// This computes and returns the whole vel-pos jacobian by finite
  /// differences. This is SUPER SUPER SLOW, and is only here for testing.
  Eigen::MatrixXs finiteDifferenceRiddersVelPosJacobian(
      simulation::WorldPtr world,
      const std::string& mapBefore,
      const std::string& mapAfter,
      std::size_t subdivisions = 20);

protected:
  std::shared_ptr<BackpropSnapshot> mBackpropSnapshot;
  std::string mRepresentation;
  std::vector<std::string> mMappingsSet;
  std::unordered_map<std::string, std::shared_ptr<Mapping>> mMappings;
  std::unordered_map<std::string, PreStepMapping> mPreStepMappings;
  std::unordered_map<std::string, PostStepMapping> mPostStepMappings;
};

using MappedBackpropSnapshotPtr = std::shared_ptr<MappedBackpropSnapshot>;

} // namespace neural
} // namespace dart

#endif