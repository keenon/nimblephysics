#ifndef DART_NEURAL_DIFF_EXTERNAL_HPP_
#define DART_NEURAL_DIFF_EXTERNAL_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/collision/Contact.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace dynamics {
class Skeleton;
}

namespace neural {

class DifferentiableExternalForce
{

public:
  DifferentiableExternalForce(
      std::shared_ptr<dynamics::Skeleton> skel, int appliedToBody);

  /// This analytically computes the torques that this world wrench applies to
  /// this skeleton.
  Eigen::VectorXs computeTau(Eigen::Vector6s worldWrench);

  /// This computes the Jacobian relating changes in `wrt` to changes in the
  /// output of `computeTau()`.
  Eigen::MatrixXs getJacobianOfTauWrt(
      Eigen::Vector6s worldWrench, neural::WithRespectTo* wrt);

  /// This computes the Jacobian relating changes in `wrt` to changes in the
  /// output of `computeTau()`.
  Eigen::MatrixXs finiteDifferenceJacobianOfTauWrt(
      Eigen::Vector6s worldWrench, neural::WithRespectTo* wrt);

  /// This computes the Jacobian relating changes in world torques to changes in
  /// the output of `computeTau()`.
  Eigen::MatrixXs getJacobianOfTauWrtWorldWrench(Eigen::Vector6s worldWrench);

  /// This computes the Jacobian relating changes in world torques to changes in
  /// the output of `computeTau()`.
  Eigen::MatrixXs finiteDifferenceJacobianOfTauWrtWorldWrench(
      Eigen::Vector6s worldWrench);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkel;
  int mBodyIndex;
  std::vector<int> activeDofs;
};

} // namespace neural
} // namespace dart

#endif