#ifndef DART_NEURAL_CONSTRAINT_MATRICES_HPP_
#define DART_NEURAL_CONSTRAINT_MATRICES_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace dart {

namespace constraint {
class ConstrainedGroup;
class ConstraintBase;
} // namespace constraint

namespace dynamics {
class Skeleton;
} // namespace dynamics

namespace neural {

enum ConstraintMapping
{
  CLAMPING = -1,
  NOT_CLAMPING = -2,
  IRRELEVANT = -3
};

/// This class pairs with a ConstrainedGroup, to save all the constraint
/// matrices and related info for that ConstrainedGroup, so that we can
/// construct full Jacobian matrices or run backprop later.
class ConstrainedGroupGradientMatrices
{
public:
  ConstrainedGroupGradientMatrices(
      constraint::ConstrainedGroup& group, double timeStep);
  ~ConstrainedGroupGradientMatrices();

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// at each constraint. This must be called before constructMatrices(), and
  /// must be called exactly once for each constraint.
  void registerConstraint(
      const std::shared_ptr<constraint::ConstraintBase>& constraint);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// at each constraint's dimension. It gets called _after_ the system has
  /// already applied a measurement impulse to that constraint dimension, and
  /// measured some velocity changes. This must be called before
  /// constructMatrices(), and must be called exactly once for each constraint's
  /// dimension.
  void measureConstraintImpulse(
      const std::shared_ptr<constraint::ConstraintBase>& constraint,
      std::size_t constraintIndex);

  /// This gets called during the setup of the ConstrainedGroupGradientMatrices
  /// after the LCP has run, with the result from the LCP solver. This can only
  /// be called once, and after this is called you cannot call
  /// measureConstraintImpulse() again!
  void constructMatrices(
      Eigen::VectorXd mX,
      Eigen::VectorXd hi,
      Eigen::VectorXd lo,
      Eigen::VectorXi fIndex);

  const Eigen::MatrixXd& getClampingConstraintMatrix() const;

  const Eigen::MatrixXd& getMassedClampingConstraintMatrix() const;

  const Eigen::MatrixXd& getUpperBoundConstraintMatrix() const;

  const Eigen::MatrixXd& getMassedUpperBoundConstraintMatrix() const;

  const Eigen::MatrixXd& getUpperBoundMappingMatrix() const;

  const Eigen::MatrixXd& getBouncingConstraintMatrix() const;

  /// These was the mX() vector used to construct this. Pretty much only here
  /// for testing.
  const Eigen::VectorXd& getContactConstraintImpluses() const;

  /// These was the fIndex() vector used to construct this. Pretty much only
  /// here for testing.
  const Eigen::VectorXi& getContactConstraintMappings() const;

  /// Returns the restitution coefficiennts at each clamping contact point.
  const Eigen::VectorXd& getBounceDiagonals() const;

  /// Returns the contact distances at each clamping contact point.
  const Eigen::VectorXd& getRestitutionDiagonals() const;

  std::size_t getNumDOFs() const;

  std::size_t getNumConstraintDim() const;

  const std::vector<std::shared_ptr<dynamics::Skeleton>>& getSkeletons() const;

protected:
  /// Impulse test matrix for the clamping constraints
  Eigen::MatrixXd mClampingConstraintMatrix;

  /// Massed impulse test matrix for the clamping constraints
  Eigen::MatrixXd mMassedClampingConstraintMatrix;

  /// Impulse test matrix for the upper bound constraints
  Eigen::MatrixXd mUpperBoundConstraintMatrix;

  /// Massed impulse test matrix for the upper bound constraints
  Eigen::MatrixXd mMassedUpperBoundConstraintMatrix;

  /// Mapping matrix for upper bound constraints
  Eigen::MatrixXd mUpperBoundMappingMatrix;

  /// Impulse test matrix for the bouncing constraints
  Eigen::MatrixXd mBouncingConstraintMatrix;

  /// This is the vector of the coefficients on the diagonal of the bounce
  /// matrix. These are 1+restitutionCoeff[i]
  Eigen::VectorXd mBounceDiagonals;

  /// This is the vector of the coefficients sized for just the bounces.
  Eigen::VectorXd mRestitutionDiagonals;

  /// This is just useful for testing the gradient computations
  Eigen::VectorXi mContactConstraintMappings;

  /// This is just useful for testing the gradient computations
  Eigen::VectorXd mContactConstraintImpulses;

  /// These are the skeletons that are covered by this constraint group
  std::vector<std::shared_ptr<dynamics::Skeleton>> mSkeletons;

  /// This is the global timestep length. This is included here because it shows
  /// up as a constant in some of the matrices.
  double mTimeStep;

  /// This is the total DOFs for this ConstrainedGroup
  std::size_t mNumDOFs;

  /// This is the number of total dimensions on all the constraints
  std::size_t mNumConstraintDim;

  /// These are the offsets into the total degrees of freedom for each skeleton
  std::unordered_map<std::string, std::size_t> mSkeletonOffset;

  /// This holds the coefficient of restitution for each constraint on this
  /// group.
  std::vector<double> mRestitutionCoeffs;

  /// This holds the outputs of the impulse tests we run to create the
  /// constraint matrices. We shuffle these vectors into the columns of
  /// mClampingConstraintMatrix and mUpperBoundConstraintMatrix depending on the
  /// values of the LCP solution. We also discard many of these vectors.
  ///
  /// mImpulseTests[k] holds the k'th constraint's impulse test, which is
  /// a concatenated vector of the results for each skeleton in the group.
  std::vector<Eigen::VectorXd> mImpulseTests;

  /// This holds the outputs of the impulse tests we run to create the
  /// constraint matrices. We shuffle these vectors into the columns of
  /// mClampingConstraintMatrix and mUpperBoundConstraintMatrix depending on the
  /// values of the LCP solution. We also discard many of these vectors.
  ///
  /// mImpulseTests[k] holds the k'th constraint's impulse test, which is
  /// a concatenated vector of the results for each skeleton in the group.
  std::vector<Eigen::VectorXd> mMassedImpulseTests;
};

} // namespace neural
} // namespace dart

#endif