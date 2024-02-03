#ifndef DART_BIOMECH_STREAMING_IK
#define DART_BIOMECH_STREAMING_IK

#include <future>
#include <memory>
#include <tuple>
#include <vector>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIStateMachine.hpp"

namespace dart {
namespace biomechanics {

/**
 * This class runs real time continuous IK on a skeleton, using a stream of
 * real time observations of anatomical markers.
 */
class StreamingIK
{
public:
  /// This class manages a thread that runs the IK continuously, and updates as
  /// we get new marker/joint observations.
  StreamingIK(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers);

  /// This cleans up the thread and any other resources used by the StreamingIK
  ~StreamingIK();

  /// This method starts the thread that runs the IK continuously.
  void startSolverThread();

  /// This method starts a thread that periodically updates a GUI server state,
  /// though at a much lower framerate than the IK solver.
  void startGUIThread(std::shared_ptr<server::GUIStateMachine> gui);

  /// This method takes in a set of markers, along with their assigned classes,
  /// and updates the targets for the IK to match the observed markers.
  void observeMarkers(
      std::vector<Eigen::Vector3s>& markers, std::vector<int> classes);

  /// This sets an anthropometric prior used to help condition the body to
  /// keep reasonable scalings.
  void setAnthropometricPrior(
      std::shared_ptr<biomechanics::Anthropometrics> prior,
      s_t priorWeight = 1.0);

  /// This method allows tests to manually input a set of markers, rather than
  /// waiting for Cortex to send them
  void reset();

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
  int mNumBodyNodes;

  std::shared_ptr<dynamics::Skeleton> mSkeletonBallJoints;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      mMarkersBallJoints;

  bool mSolverThreadRunning;
  std::future<void> mSolverThread;

  bool mGUIThreadRunning;
  std::future<void> mGUIThread;

  /// This is the last set of marker observations we got, always the same size
  /// regardless of if we observed all the markers or not. Now the top entry is
  /// the weight for that entry (0 if unobserved, non-zero otherwise).
  Eigen::VectorXs mLastMarkerObservations;
  Eigen::VectorXs mLastMarkerObservationWeights;

  /// This is the last set of joint observations we got, always the same size
  /// regardless of if we observed all the joints or not. Now the top entry is
  /// the weight for that entry (0 if unobserved, non-zero otherwise).
  // Eigen::Matrix<s_t, 4, Eigen::Dynamic> mLastJointObservations;
  // TODO: ^ implement!

  s_t mAnthropometricPriorWeight;
  std::shared_ptr<biomechanics::Anthropometrics> mAnthropometrics;
};

} // namespace biomechanics
} // namespace dart

#endif