#ifndef DART_NEURAL_CLASSIC_BACKPROP_SNAPSHOT_HPP_
#define DART_NEURAL_CLASSIC_BACKPROP_SNAPSHOT_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/BackpropSnapshot.hpp"

namespace dart {
namespace neural {

class ClassicBackpropSnapshot : public BackpropSnapshot
{
public:
  ClassicBackpropSnapshot(
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