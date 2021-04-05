#ifndef DART_NEURAL_IDENTITY_MAPPING_HPP_
#define DART_NEURAL_IDENTITY_MAPPING_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

#include "dart/neural/Mapping.hpp"

namespace dart {

namespace neural {

class IdentityMapping : public Mapping
{
public:
  IdentityMapping(std::shared_ptr<simulation::World> world);

  int getPosDim() override;
  int getVelDim() override;
  int getForceDim() override;
  int getMassDim() override;

  void setPositions(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& positions) override;
  void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& velocities) override;
  void setForces(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& forces) override;
  void setMasses(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& masses) override;

  void getPositionsInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> positions) override;
  void getVelocitiesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> velocities) override;
  void getForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> forces) override;
  void getMassesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> masses) override;

  /// This gets a Jacobian relating the changes in the outer positions (the
  /// "mapped" positions) to inner positions (the "real" positions)
  Eigen::MatrixXs getMappedPosToRealPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  Eigen::MatrixXs getRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner velocities (the
  /// "real" velocities) to the corresponding outer positions (the "mapped"
  /// positions)
  Eigen::MatrixXs getRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  Eigen::MatrixXs getMappedVelToRealVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  Eigen::MatrixXs getRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner position (the
  /// "real" position) to the corresponding outer velocity (the "mapped"
  /// velocity)
  Eigen::MatrixXs getRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  Eigen::MatrixXs getMappedForceToRealForceJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  Eigen::MatrixXs getRealForceToMappedForceJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  Eigen::MatrixXs getMappedMassToRealMassJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  Eigen::MatrixXs getRealMassToMappedMassJac(
      std::shared_ptr<simulation::World> world) override;

  Eigen::VectorXs getPositionLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getPositionUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getVelocityLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getVelocityUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getForceLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getForceUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getMassLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getMassUpperLimits(
      std::shared_ptr<simulation::World> world) override;

protected:
  int mNumDofs;
  int mMassDim;
};

} // namespace neural
} // namespace dart

#endif