#ifndef DART_NEURAL_IK_MAPPING_HPP_
#define DART_NEURAL_IK_MAPPING_HPP_

#include <memory>
#include <optional>
#include <string>

#include "dart/include_eigen.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/Mapping.hpp"

namespace dart {

namespace neural {

enum IKMappingEntryType
{
  NODE_SPATIAL,
  NODE_LINEAR,
  NODE_ANGULAR,
  COM
};

struct IKMappingEntry
{
  IKMappingEntryType type;
  std::string skelName;
  int bodyNodeOffset; // can be -1 for COM entries

  IKMappingEntry(IKMappingEntryType type, std::string skelName)
    : type(type), skelName(skelName), bodyNodeOffset(-1){};

  IKMappingEntry(IKMappingEntryType type, dynamics::BodyNode* node)
    : type(type),
      skelName(node->getSkeleton()->getName()),
      bodyNodeOffset(node->getIndexInSkeleton()){};
};

class IKMapping : public Mapping
{
public:
  IKMapping(std::shared_ptr<simulation::World> world);

  /// When we called setPosition(), we need to run an IK solve. This
  /// sets the limit on the number of iterations of our solver to run.
  void setIKIterationLimit(int limit);
  int getIKIterationLimit();

  /// This adds the spatial (6D) coordinates of a body node to the list,
  /// increasing Dim size by 6
  void addSpatialBodyNode(dynamics::BodyNode* node);

  /// This adds the linear (3D) coordinates of a body node to the list,
  /// increasing Dim size by 3
  void addLinearBodyNode(dynamics::BodyNode* node);

  /// This adds the angular (3D) coordinates of a body node to the list,
  /// increasing Dim size by 3
  void addAngularBodyNode(dynamics::BodyNode* node);

  int getPosDim() override;
  int getVelDim() override;
  int getControlForceDim() override;
  int getMassDim() override;

  void setPositions(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& positions) override;
  void setVelocities(
      std::shared_ptr<simulation::World> world,
      const Eigen::Ref<Eigen::VectorXs>& velocities) override;
  void setControlForces(
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
  void getControlForcesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> forces) override;
  void getMassesInPlace(
      std::shared_ptr<simulation::World> world,
      /* OUT */ Eigen::Ref<Eigen::VectorXs> masses) override;

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

  /// This gets a Jacobian relating the changes in the inner force (the
  /// "real" force) to the corresponding outer force (the "mapped"
  /// force)
  Eigen::MatrixXs getRealForceToMappedForceJac(
      std::shared_ptr<simulation::World> world) override;

  /// This gets a Jacobian relating the changes in the inner mass (the
  /// "real" mass) to the corresponding outer mass (the "mapped"
  /// mass)
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
  Eigen::VectorXs getControlForceLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getControlForceUpperLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getMassLowerLimits(
      std::shared_ptr<simulation::World> world) override;
  Eigen::VectorXs getMassUpperLimits(
      std::shared_ptr<simulation::World> world) override;

protected:
  /// This returns the number of dimensions that the IK mapping represents.
  int getDim();

  /// Computes a Jacobian that transforms changes in joint angle to changes in
  /// IK body positions (expressed in log space).
  Eigen::MatrixXs getPosJacobian(std::shared_ptr<simulation::World> world);

  /// Computes the pseudo-inverse of the pos Jacobian
  Eigen::MatrixXs getPosJacobianInverse(
      std::shared_ptr<simulation::World> world);

  /// Computes a Jacobian that transforms changes in joint vel to changes in
  /// IK body vels (expressed in log space).
  Eigen::MatrixXs getVelJacobian(std::shared_ptr<simulation::World> world);

  /// Computes the pseudo-inverse of the vel Jacobian
  Eigen::MatrixXs getVelJacobianInverse(
      std::shared_ptr<simulation::World> world);

  /// Computes a Jacobian of J(x)*vel wrt pos
  Eigen::MatrixXs getJacobianOfJacVelWrtPosition(
      std::shared_ptr<simulation::World> world);

  /// The brute force version of getJacobianOfJacVelWrtPosition()
  Eigen::MatrixXs bruteForceJacobianOfJacVelWrtPosition(
      std::shared_ptr<simulation::World> world);

  std::vector<IKMappingEntry> mEntries;

  int mMassDim;
  int mIKIterationLimit;
};

} // namespace neural
} // namespace dart

#endif