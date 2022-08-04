#ifndef DART_BIOMECH_DYNAMICS_FITTER_HPP_
#define DART_BIOMECH_DYNAMICS_FITTER_HPP_

#include <memory>
#include <vector>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

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
      dynamics::MarkerMap markerMap);

  // This returns the dimension of the decision variables (the length of the
  // flatten() vector), which depends on which variables we choose to include in
  // the optimization problem.
  int getProblemSize();

  // This writes the problem state into a flat vector
  Eigen::VectorXs flatten();

  // This reads the problem state out of a flat vector, and into the init object
  void unflatten(Eigen::VectorXs x);

  // This gets the value of the loss function, as a weighted sum of the
  // discrepancy between measured and expected GRF data and other regularization
  // terms.
  s_t computeLoss(Eigen::VectorXs x);

  // This gets the gradient of the loss function
  Eigen::VectorXs computeGradient(Eigen::VectorXs x);

  ///////////////////////////////////////////////////////////////////////
  // Notes:
  //
  // Just the linear component:
  //
  // Each body acceleration wrt (COMs, poses, body scales)
  // Then f_implied = sum(m[i]*a[i])
  // Loss is (f_implied - f_measured).norm()
  // Vars are: masses, COMs, poses, body scales
  //
  // Diff:
  // 2*(f_implied - f_measured).norm()*(f_implied - f_measured) * d_f_implied
  // Where d_f_implied:
  // - wrt Mass: just a[i]
  // - wrt COMs: m[i]*d_a_wrt_COM[i]
  //   - relies on d_pos_wrt_COM[i]
  // - etc
  //
  // Just the rotational component;
  //
  // Each body rotational acceleration wrt (poses)
  // Then tau_implied = sum(I[i]*omega[i])
  // Loss is (tau_implied - tau_measured).norm()
  ///////////////////////////////////////////////////////////////////////

  // Get the concatenated body center-of-mass positions over time. The 3-vec
  // position of each body is stacked, according to their ordering in the
  // skeleton->getBodyNode(i), to create each column of the returned matrix.
  // There is one column per timestep.
  Eigen::MatrixXs getBodyCOMPositions(Eigen::VectorXs x, int trial);

  // This gets a [3 x T] matrix, which describes the gradient of the body's COM
  // world position with respect to its local COM position.
  Eigen::MatrixXs getBodyCOMPositionsGradientWrtCOMs(
      Eigen::VectorXs x, int trial, int body, int axis);

  DynamicsFitProblem& setIncludeMasses(bool value);
  DynamicsFitProblem& setIncludeCOMs(bool value);
  DynamicsFitProblem& setIncludeInertias(bool value);
  DynamicsFitProblem& setIncludePoses(bool value);
  DynamicsFitProblem& setIncludeMarkerOffsets(bool value);
  DynamicsFitProblem& setIncludeBodyScales(bool value);

protected:
  bool mIncludeMasses;
  bool mIncludeCOMs;
  bool mIncludeInertias;
  bool mIncludePoses;
  bool mIncludeMarkerOffsets;
  bool mIncludeBodyScales;
  std::shared_ptr<DynamicsInitialization> mInit;
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  dynamics::MarkerMap mMarkerMap;
};

class DynamicsFitter
{
public:
  DynamicsFitter(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markerMap);

  // This bundles together the objects we need in order to track a dynamics
  // problem around through multiple steps of optimization
  std::shared_ptr<DynamicsInitialization> createInitialization(
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