#ifndef DART_NEURAL_CLASSIC_CONSTRAINT_MATRICES_HPP_
#define DART_NEURAL_CLASSIC_CONSTRAINT_MATRICES_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"

namespace dart {
namespace neural {

class ClassicConstrainedGroupGradientMatrices
  : public ConstrainedGroupGradientMatrices
{
public:
  ClassicConstrainedGroupGradientMatrices(
      constraint::ConstrainedGroup& group, double timestep);

  static std::shared_ptr<ClassicConstrainedGroupGradientMatrices> create(
      constraint::ConstrainedGroup& group, double timeStep);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// at each constraint's dimension. It gets called _after_ the system has
  /// already applied a measurement impulse to that constraint dimension, and
  /// measured some velocity changes. This must be called before
  /// constructMatrices(), and must be called exactly once for each constraint's
  /// dimension.
  void measureConstraintImpulse(
      const std::shared_ptr<constraint::ConstraintBase>& constraint,
      std::size_t constraintIndex) override;

  Eigen::MatrixXd getProjectionIntoClampsMatrix() override;

  Eigen::MatrixXd getForceVelJacobian() override;

  Eigen::MatrixXd getVelVelJacobian() override;

private:
};

} // namespace neural
} // namespace dart

#endif