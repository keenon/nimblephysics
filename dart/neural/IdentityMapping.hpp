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

  void setPositions(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& positions) override;
  void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& velocities) override;
  void setForces(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXd>& forces) override;

  void getPositionsInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> positions) override;
  void getVelocitiesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> velocities) override;
  void getForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXd> forces) override;

  /// This gets a Jacobian relating the changes in the outer positions (the
  /// "mapped" positions) to inner positions (the "real" positions)
  Eigen::MatrixXd getMappedPosToRealPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  Eigen::MatrixXd getRealPosToMappedPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner velocities (the
  /// "real" velocities) to the corresponding outer positions (the "mapped"
  /// positions)
  Eigen::MatrixXd getRealVelToMappedPosJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  Eigen::MatrixXd getMappedVelToRealVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  Eigen::MatrixXd getRealVelToMappedVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner position (the
  /// "real" position) to the corresponding outer velocity (the "mapped"
  /// velocity)
  Eigen::MatrixXd getRealPosToMappedVelJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  Eigen::MatrixXd getMappedForceToRealForceJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  Eigen::MatrixXd getRealForceToMappedForceJac(
      std::shared_ptr<simulation::World> world) override;

  Eigen::VectorXd getPositionLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXd getPositionUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXd getVelocityLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXd getVelocityUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXd getForceLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXd getForceUpperLimits(
      std::shared_ptr<simulation::World> world) override;

protected:
  int mNumDofs;
};

} // namespace neural
} // namespace dart

#endif