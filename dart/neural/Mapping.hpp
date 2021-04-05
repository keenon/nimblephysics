#ifndef DART_NEURAL_MAPPING_HPP_
#define DART_NEURAL_MAPPING_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace neural {

class Mapping
{
public:
  Mapping();

  virtual ~Mapping();

  virtual int getPosDim() = 0;
  virtual int getVelDim() = 0;
  virtual int getForceDim() = 0;
  virtual int getMassDim() = 0;

  virtual void setPositions(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& positions)
      = 0;
  virtual void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& velocities)
      = 0;
  virtual void setForces(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& forces)
      = 0;
  virtual void setMasses(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& masses)
      = 0;

  virtual void getPositionsInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> positions)
      = 0;
  virtual void getVelocitiesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> velocities)
      = 0;
  virtual void getForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> forces)
      = 0;
  virtual void getMassesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> masses)
      = 0;

  Eigen::VectorXs getPositions(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getVelocities(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getForces(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getMasses(std::shared_ptr<simulation::World> world);

  /// This gets a Jacobian relating the changes in the outer positions (the
  /// "mapped" positions) to inner positions (the "real" positions)
  virtual Eigen::MatrixXs getMappedPosToRealPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXs getRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner velocities (the
  /// "real" velocities) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXs getRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  virtual Eigen::MatrixXs getMappedVelToRealVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXs getRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner position (the
  /// "real" position) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXs getRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  virtual Eigen::MatrixXs getMappedForceToRealForceJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  virtual Eigen::MatrixXs getRealForceToMappedForceJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer mass (the
  /// "mapped" mass) to inner mass (the "real" mass)
  virtual Eigen::MatrixXs getMappedMassToRealMassJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner mass (the
  /// "real" mass) to the corresponding outer mass (the "mapped"
  /// mass)
  virtual Eigen::MatrixXs getRealMassToMappedMassJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  virtual Eigen::VectorXs getPositionLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getPositionUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getVelocityLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getVelocityUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getForceLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getForceUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getMassLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getMassUpperLimits(
      std::shared_ptr<simulation::World> world);
};

} // namespace neural
} // namespace dart

#endif