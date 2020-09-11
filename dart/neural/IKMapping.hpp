#ifndef DART_NEURAL_IDENTITY_MAPPING_HPP_
#define DART_NEURAL_IDENTITY_MAPPING_HPP_

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>

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
  /// This returns the number of dimensions that the IK mapping represents.
  int getDim();

  /// Computes a Jacobian that transforms changes in joint angle to changes in
  /// IK body positions (expressed in log space).
  Eigen::MatrixXd getJacobian(std::shared_ptr<simulation::World> world);

  /// Computes the pseudo-inverse of the Jacobian
  Eigen::MatrixXd getJacobianInverse(std::shared_ptr<simulation::World> world);

  std::vector<IKMappingEntry> mEntries;
};

} // namespace neural
} // namespace dart

#endif