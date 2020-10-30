#ifndef DART_NEURAL_MAPPING_HPP_
#define DART_NEURAL_MAPPING_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

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
      const Eigen::Ref<Eigen::VectorXd>& positions)
      = 0;
  virtual void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& velocities)
      = 0;
  virtual void setForces(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& forces)
      = 0;
  virtual void setMasses(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& masses)
      = 0;

  virtual void getPositionsInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> positions)
      = 0;
  virtual void getVelocitiesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> velocities)
      = 0;
  virtual void getForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> forces)
      = 0;
  virtual void getMassesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> masses)
      = 0;

  Eigen::VectorXd getPositions(std::shared_ptr<simulation::World> world);
  Eigen::VectorXd getVelocities(std::shared_ptr<simulation::World> world);
  Eigen::VectorXd getForces(std::shared_ptr<simulation::World> world);
  Eigen::VectorXd getMasses(std::shared_ptr<simulation::World> world);

  /// This gets a Jacobian relating the changes in the outer positions (the
  /// "mapped" positions) to inner positions (the "real" positions)
  virtual Eigen::MatrixXd getMappedPosToRealPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXd getRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner velocities (the
  /// "real" velocities) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXd getRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  virtual Eigen::MatrixXd getMappedVelToRealVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXd getRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner position (the
  /// "real" position) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXd getRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  virtual Eigen::MatrixXd getMappedForceToRealForceJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  virtual Eigen::MatrixXd getRealForceToMappedForceJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer mass (the
  /// "mapped" mass) to inner mass (the "real" mass)
  virtual Eigen::MatrixXd getMappedMassToRealMassJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner mass (the
  /// "real" mass) to the corresponding outer mass (the "mapped"
  /// mass)
  virtual Eigen::MatrixXd getRealMassToMappedMassJac(
      std::shared_ptr<simulation::World> world)
      = 0;

  virtual Eigen::VectorXd getPositionLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getPositionUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getVelocityLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getVelocityUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getForceLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getForceUpperLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getMassLowerLimits(
      std::shared_ptr<simulation::World> world);
  virtual Eigen::VectorXd getMassUpperLimits(
      std::shared_ptr<simulation::World> world);
};

} // namespace neural
} // namespace dart

#endif