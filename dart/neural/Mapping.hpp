#ifndef DART_NEURAL_MAPPING_HPP_
#define DART_NEURAL_MAPPING_HPP_

#include <memory>
#include <optional>

#include "dart/include_eigen.hpp"

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
  virtual int getControlForceDim() = 0;
  virtual int getMassDim() = 0;

  virtual void setPositions(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& positions)
      = 0;
  virtual void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& velocities)
      = 0;
  virtual void setControlForces(
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
  virtual void getControlForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> forces)
      = 0;
  virtual void getMassesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> masses)
      = 0;

  Eigen::VectorXs getPositions(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getVelocities(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getControlForces(std::shared_ptr<simulation::World> world);
  Eigen::VectorXs getMasses(std::shared_ptr<simulation::World> world);

  /// Check if a Jacobian is equal
  void equalsOrCrash(
      Eigen::MatrixXs analytical,
      Eigen::MatrixXs bruteForce,
      std::shared_ptr<simulation::World> world,
      const std::string& name);

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXs getRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  Eigen::MatrixXs finiteDifferenceRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world, bool useRidders = true);

  /// This gets a Jacobian relating the changes in the inner velocities (the
  /// "real" velocities) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXs getRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  Eigen::MatrixXs finiteDifferenceRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world, bool useRidders = true);

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXs getRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  Eigen::MatrixXs finiteDifferenceRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world, bool useRidders = true);

  /// This gets a Jacobian relating the changes in the inner position (the
  /// "real" position) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXs getRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  Eigen::MatrixXs finiteDifferenceRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world, bool useRidders = true);

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  virtual Eigen::MatrixXs getRealForceToMappedForceJac(
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
  virtual Eigen::VectorXs getControlForceLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getControlForceUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getMassLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXs getMassUpperLimits(
      std::shared_ptr<simulation::World> world);
};

} // namespace neural
} // namespace dart

#endif