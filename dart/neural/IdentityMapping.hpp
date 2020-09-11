#ifndef DART_NEURAL_IDENTITY_MAPPING_HPP_
#define DART_NEURAL_IDENTITY_MAPPING_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

#include "dart/neural/Mapping.hpp"

namespace dart {

namespace neural {

class IdentityMapping : Mapping
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
  Eigen::MatrixXd getMappedPosToRealPos(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  Eigen::MatrixXd getRealPosToMappedPos(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  Eigen::MatrixXd getMappedVelToRealVel(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  Eigen::MatrixXd getRealVelToMappedVel(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  Eigen::MatrixXd getMappedForceToRealForce(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  Eigen::MatrixXd getRealForceToMappedForce(
      std::shared_ptr<simulation::World> world) override;

protected:
  int mNumDofs;
};

} // namespace neural
} // namespace dart

#endif