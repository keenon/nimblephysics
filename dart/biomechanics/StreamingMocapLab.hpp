#ifndef DART_BIOMECH_STREAMING_MOCAP_LAB
#define DART_BIOMECH_STREAMING_MOCAP_LAB

#include <future>
#include <memory>
#include <tuple>
#include <vector>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/CortexStreaming.hpp"
#include "dart/biomechanics/StreamingIK.hpp"
#include "dart/biomechanics/StreamingMarkerTraces.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIStateMachine.hpp"

namespace dart {
namespace biomechanics {

/**
 * This class integrates the StreamingIK and StreamingMarkerTraces classes with
 * CortexStreaming to create a nice C++ interface for running real-time mocap
 * labs, without holding the GIL in Python.
 */
class StreamingMocapLab
{
public:
  /// This class manages a thread that runs the IK continuously, and updates as
  /// we get new marker/joint observations.
  StreamingMocapLab(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers);

  /// This cleans up the thread and any other resources used by the StreamingIK
  ~StreamingMocapLab();

  /// This method starts the thread that runs the IK continuously.
  void startSolverThread();

  /// This method starts a thread that periodically updates a GUI server state,
  /// though at a much lower framerate than the IK solver.
  void startGUIThread(std::shared_ptr<server::GUIStateMachine> gui);

  /// This sets an anthropometric prior used to help condition the body to
  /// keep reasonable scalings.
  void setAnthropometricPrior(
      std::shared_ptr<biomechanics::Anthropometrics> prior,
      s_t priorWeight = 1.0);

  /// This method establishes a link to Cortex, and listens for real-time
  /// observations of markers and force plate data
  void listenToCortex(
      std::string host,
      int cortexMulticastPort = 1001,
      int cortexRequestsPort = 1510);

  /// This method allows tests to manually input a set of markers, rather than
  /// waiting for Cortex to send them
  void manuallyObserveMarkers(
      std::vector<Eigen::Vector3s>& markers,
      long timestamp,
      std::vector<Eigen::Vector9s>& copTorqueForces);

  /// This method returns the features that we used to predict the classes of
  /// the markers. The first element of the pair is the features (which are
  /// trace points concatenated with the time, as measured in integer units of
  /// "windowDuration", backwards from now), and the second is the trace ID for
  /// each point, so that we can correctly assign logit outputs back to the
  /// traces.
  std::pair<Eigen::MatrixXs, Eigen::VectorXi> getTraceFeatures(
      int numWindows, long windowDuration);

  /// This method takes in the logits for each point, and the trace IDs for each
  /// point, and updates the internal state of the trace classifier to reflect
  /// the new information.
  void observeTraceLogits(
      const Eigen::MatrixXs& logits, const Eigen::VectorXi& traceIDs);

  /// This method resets the state of the mocap lab, including the IK and the
  /// marker traces
  void reset(std::shared_ptr<server::GUIStateMachine> gui);

  /// This method returns the IK solver that this mocap lab is using
  std::shared_ptr<StreamingIK> getIK();

  /// This method returns the marker traces that this mocap lab is using
  std::shared_ptr<StreamingMarkerTraces> getMarkerTraces();

  /// This method uses the recent history of poses to estimate the current state
  /// of the skeleton, including velocity and acceleration. The skeleton is set
  /// to the correct position, velocity, and acceleration.
  void estimateState(long now, int numHistory = 20, int polynomialDegree = 3);

protected:
  std::shared_ptr<StreamingMarkerTraces> mMarkerTraces;
  std::shared_ptr<StreamingIK> mIK;
  std::shared_ptr<CortexStreaming> mCortex;
  std::shared_ptr<server::GUIStateMachine> mGui;
};

} // namespace biomechanics
} // namespace dart

#endif