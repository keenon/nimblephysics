#ifndef DART_BIOMECH_DYNAMICS_FITTER_HPP_
#define DART_BIOMECH_DYNAMICS_FITTER_HPP_

#include <memory>
#include <vector>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"

namespace dart {
namespace biomechanics {

/**
 * This class factors out the code to deal with calculating residual forces, and
 * the associated Jacobians of residual force with respect to lots of different
 * inputs.
 */
class ResidualForceHelper
{
public:
  ResidualForceHelper(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      std::vector<int> forceBodies);

  ///////////////////////////////////////////
  // Computes the residual for a specific timestep
  Eigen::Vector6s calculateResidual(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat);

  ///////////////////////////////////////////
  // Computes the residual norm for a specific timestep
  s_t calculateResidualNorm(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat);

  ///////////////////////////////////////////
  // Computes the Jacobian of the residual with respect to `wrt`
  Eigen::MatrixXs calculateResidualJacobianWrt(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat,
      neural::WithRespectTo* wrt);

  ///////////////////////////////////////////
  // Computes the Jacobian of the residual with respect to `wrt`
  Eigen::MatrixXs finiteDifferenceResidualJacobianWrt(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat,
      neural::WithRespectTo* wrt);

  ///////////////////////////////////////////
  // Computes the gradient of the residual norm with respect to `wrt`
  Eigen::VectorXs calculateResidualNormGradientWrt(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat,
      neural::WithRespectTo* wrt);

  ///////////////////////////////////////////
  // Computes the gradient of the residual norm with respect to `wrt`
  Eigen::VectorXs finiteDifferenceResidualNormGradientWrt(
      Eigen::VectorXs q,
      Eigen::VectorXs dq,
      Eigen::VectorXs ddq,
      Eigen::VectorXs forcesConcat,
      neural::WithRespectTo* wrt);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkel;
  std::vector<neural::DifferentiableExternalForce> mForces;
};

/**
 * We create a single initialization object, and pass it around to optimization
 * problems to re-use, because it's not super cheap to construct.
 */
struct DynamicsInitialization
{
  ///////////////////////////////////////////
  // Inputs
  std::vector<std::vector<ForcePlate>> forcePlateTrials;
  std::vector<Eigen::MatrixXs> originalPoseTrials;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<s_t> trialTimesteps;

  ///////////////////////////////////////////
  // Pure dynamics values
  Eigen::VectorXs bodyMasses;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> bodyCom;
  Eigen::Matrix<s_t, 6, Eigen::Dynamic> bodyInertia;

  ///////////////////////////////////////////
  // Relevant when trying to get dynamics to agree with movement
  std::vector<Eigen::MatrixXs> poseTrials;
  Eigen::VectorXs groupScales;
  std::map<std::string, Eigen::Vector3s> markerOffsets;

  ///////////////////////////////////////////
  // Convenience objects
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      updatedMarkerMap;
};

class DynamicsFitProblem
{
public:
  DynamicsFitProblem(
      std::shared_ptr<DynamicsInitialization> init,
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markerMap,
      std::vector<dynamics::BodyNode*> footNodes);

  // This returns the dimension of the decision variables (the length of the
  // flatten() vector), which depends on which variables we choose to include in
  // the optimization problem.
  int getProblemSize();

  // This writes the problem state into a flat vector
  Eigen::VectorXs flatten();

  // This writes the upper bounds into a flat vector
  Eigen::VectorXs flattenUpperBound();

  // This writes the upper bounds into a flat vector
  Eigen::VectorXs flattenLowerBound();

  // This reads the problem state out of a flat vector, and into the init object
  void unflatten(Eigen::VectorXs x);

  // This gets the value of the loss function, as a weighted sum of the
  // discrepancy between measured and expected GRF data and other regularization
  // terms.
  s_t computeLoss(Eigen::VectorXs x);

  // This gets the gradient of the loss function
  Eigen::VectorXs computeGradient(Eigen::VectorXs x);

  // This gets the gradient of the loss function
  Eigen::VectorXs finiteDifferenceGradient(Eigen::VectorXs x);

  // Print out the errors in a gradient vector in human readable form
  void debugErrors(Eigen::VectorXs fd, Eigen::VectorXs analytical, s_t tol);

  DynamicsFitProblem& setIncludeMasses(bool value);
  DynamicsFitProblem& setIncludeCOMs(bool value);
  DynamicsFitProblem& setIncludeInertias(bool value);
  DynamicsFitProblem& setIncludePoses(bool value);
  DynamicsFitProblem& setIncludeMarkerOffsets(bool value);
  DynamicsFitProblem& setIncludeBodyScales(bool value);

public:
  bool mIncludeMasses;
  bool mIncludeCOMs;
  bool mIncludeInertias;
  bool mIncludeBodyScales;
  bool mIncludePoses;
  bool mIncludeMarkerOffsets;
  std::shared_ptr<DynamicsInitialization> mInit;
  std::shared_ptr<dynamics::Skeleton> mSkeleton;

  std::vector<Eigen::MatrixXs> mPoses;
  std::vector<Eigen::MatrixXs> mVels;
  std::vector<Eigen::MatrixXs> mAccs;
  std::vector<Eigen::MatrixXs> mGRFs;
  std::vector<dynamics::BodyNode*> mGRFBodyNodes;

  dynamics::MarkerMap mMarkerMap;
  std::vector<std::string> mMarkerNames;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;

  std::vector<dynamics::BodyNode*> mFootNodes;
  std::vector<int> mForceBodyIndices;
  std::shared_ptr<ResidualForceHelper> mResidualHelper;
};

class DynamicsFitter
{
public:
  DynamicsFitter(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markerMap);

  // This bundles together the objects we need in order to track a dynamics
  // problem around through multiple steps of optimization
  static std::shared_ptr<DynamicsInitialization> createInitialization(
      std::shared_ptr<dynamics::Skeleton> skel,
      std::vector<std::vector<ForcePlate>> forcePlateTrials,
      std::vector<Eigen::MatrixXs> poseTrials,
      std::vector<int> framesPerSecond,
      std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
          markerObservationTrials);

  // This computes and returns the positions of the center of mass at each
  // frame
  std::vector<Eigen::Vector3s> comPositions(
      std::shared_ptr<DynamicsInitialization> init, int trial);

  // This computes and returns the acceleration of the center of mass at each
  // frame
  std::vector<Eigen::Vector3s> comAccelerations(
      std::shared_ptr<DynamicsInitialization> init, int trial);

  // This computes and returns a list of the net forces on the center of mass,
  // given the motion and link masses
  std::vector<Eigen::Vector3s> impliedCOMForces(
      std::shared_ptr<DynamicsInitialization> init,
      int trial,
      bool includeGravity = true);

  // This returns a list of the total GRF force on the body at each timestep
  std::vector<Eigen::Vector3s> measuredGRFForces(
      std::shared_ptr<DynamicsInitialization> init, int trial);

  // 1. Scale the total mass of the body (keeping the ratios of body links
  // constant) to get it as close as possible to GRF gravity forces.
  void scaleLinkMassesFromGravity(std::shared_ptr<DynamicsInitialization> init);

  // 2. Estimate just link masses, while holding the positions, COMs, and
  // inertias constant
  void estimateLinkMassesFromAcceleration(
      std::shared_ptr<DynamicsInitialization> init,
      s_t regularizationWeight = 50.0);

  // 3. Estimate link masses and COMs
  void estimateLinkMassesAndCOMs(std::shared_ptr<DynamicsInitialization> init);

  // 4. Estimate inertia properties

  // 5. Run a giant bilevel optimization where we're allowed to tweak poses,
  // marker offsets, inertias, etc

  // This debugs the current state, along with visualizations of errors where
  // the dynamics do not match the force plate data
  void saveDynamicsToGUI(
      const std::string& path,
      std::shared_ptr<DynamicsInitialization> init,
      int trialIndex,
      int framesPerSecond);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  dynamics::MarkerMap mMarkerMap;
};

}; // namespace biomechanics
}; // namespace dart

#endif