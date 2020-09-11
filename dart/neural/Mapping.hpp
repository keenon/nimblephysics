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

  Eigen::VectorXd getPositions(std::shared_ptr<simulation::World> world);
  Eigen::VectorXd getVelocities(std::shared_ptr<simulation::World> world);
  Eigen::VectorXd getForces(std::shared_ptr<simulation::World> world);

  /// This gets a Jacobian relating the changes in the outer positions (the
  /// "mapped" positions) to inner positions (the "real" positions)
  virtual Eigen::MatrixXd getMappedPosToRealPos(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner positions (the
  /// "real" positions) to the corresponding outer positions (the "mapped"
  /// positions)
  virtual Eigen::MatrixXd getRealPosToMappedPos(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer velocity (the
  /// "mapped" velocity) to inner velocity (the "real" velocity)
  virtual Eigen::MatrixXd getMappedVelToRealVel(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner velocity (the
  /// "real" velocity) to the corresponding outer velocity (the "mapped"
  /// velocity)
  virtual Eigen::MatrixXd getRealVelToMappedVel(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the outer force (the
  /// "mapped" force) to inner force (the "real" force)
  virtual Eigen::MatrixXd getMappedForceToRealForce(
      std::shared_ptr<simulation::World> world)
      = 0;

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  virtual Eigen::MatrixXd getRealForceToMappedForce(
      std::shared_ptr<simulation::World> world)
      = 0;
};

} // namespace neural
} // namespace dart

#endif