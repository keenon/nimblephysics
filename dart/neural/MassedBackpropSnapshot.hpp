#ifndef DART_NEURAL_MASSED_BACKPROP_SNAPSHOT_HPP_
#define DART_NEURAL_MASSED_BACKPROP_SNAPSHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/BackpropSnapshot.hpp"

namespace dart {
namespace neural {

class MassedBackpropSnapshot : public BackpropSnapshot
{
public:
  MassedBackpropSnapshot(
      simulation::WorldPtr world,
      Eigen::VectorXd forwardPassPosition,
      Eigen::VectorXd forwardPassVelocity,
      Eigen::VectorXd forwardPassTorques);

  Eigen::MatrixXd getProjectionIntoClampsMatrix() override;

  Eigen::MatrixXd getForceVelJacobian() override;

  Eigen::MatrixXd getVelVelJacobian() override;
};

} // namespace neural
} // namespace dart

#endif