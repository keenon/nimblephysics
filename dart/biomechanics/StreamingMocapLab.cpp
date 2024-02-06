#include "dart/biomechanics/StreamingMocapLab.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

/// This class manages a thread that runs the IK continuously, and updates as
/// we get new marker/joint observations.
StreamingMocapLab::StreamingMocapLab(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers)
{
  int totalClasses = skeleton->getNumBodyNodes() + markers.size() + 1;
  int numBodies = skeleton->getNumBodyNodes();
  mMarkerTraces
      = std::make_shared<StreamingMarkerTraces>(totalClasses, numBodies);
  mIK = std::make_shared<StreamingIK>(skeleton, markers);
}

/// This cleans up the thread and any other resources used by the StreamingIK
StreamingMocapLab::~StreamingMocapLab()
{
}

/// This method starts the thread that runs the IK continuously.
void StreamingMocapLab::startSolverThread()
{
  mIK->startSolverThread();
}

/// This method starts a thread that periodically updates a GUI server state,
/// though at a much lower framerate than the IK solver.
void StreamingMocapLab::startGUIThread(
    std::shared_ptr<server::GUIStateMachine> gui)
{
  mIK->startGUIThread(gui);
}

/// This sets an anthropometric prior used to help condition the body to
/// keep reasonable scalings.
void StreamingMocapLab::setAnthropometricPrior(
    std::shared_ptr<biomechanics::Anthropometrics> prior, s_t priorWeight)
{
  mIK->setAnthropometricPrior(prior, priorWeight);
}

/// This method establishes a link to Cortex, and listens for real-time
/// observations of markers and force plate data
void StreamingMocapLab::listenToCortex(std::string host, int port)
{
  mCortex = std::make_shared<CortexStreaming>(host, port);
  mCortex->setFrameHandler([&](std::vector<std::string> markerNames,
                               std::vector<Eigen::Vector3s> markers,
                               std::vector<Eigen::MatrixXs> copTorqueForces) {
    (void)markerNames;
    (void)copTorqueForces;
    long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    manuallyObserveMarkers(markers, timestamp);
  });
}

/// This method allows tests to manually input a set of markers, rather than
/// waiting for Cortex to send them
void StreamingMocapLab::manuallyObserveMarkers(
    std::vector<Eigen::Vector3s>& markers, long timestamp)
{
  auto pair = mMarkerTraces->observeMarkers(markers, timestamp);
  mIK->observeMarkers(markers, pair.first);
}

/// This method returns the features that we used to predict the classes of
/// the markers. The first element of the pair is the features (which are
/// trace points concatenated with the time, as measured in integer units of
/// "windowDuration", backwards from now), and the second is the trace ID for
/// each point, so that we can correctly assign logit outputs back to the
/// traces.
std::pair<Eigen::MatrixXs, Eigen::VectorXi> StreamingMocapLab::getTraceFeatures(
    int numWindows, long windowDuration)
{
  return mMarkerTraces->getTraceFeatures(numWindows, windowDuration);
}

/// This method takes in the logits for each point, and the trace IDs for each
/// point, and updates the internal state of the trace classifier to reflect
/// the new information.
void StreamingMocapLab::observeTraceLogits(
    const Eigen::MatrixXs& logits, const Eigen::VectorXi& traceIDs)
{
  mMarkerTraces->observeTraceLogits(logits, traceIDs);
}

/// This method resets the state of the mocap lab, including the IK and the
/// marker traces
void StreamingMocapLab::reset()
{
  mIK->reset();
  mMarkerTraces->reset();
}

/// This method returns the IK solver that this mocap lab is using
std::shared_ptr<StreamingIK> StreamingMocapLab::getIK()
{
  return mIK;
}

/// This method returns the marker traces that this mocap lab is using
std::shared_ptr<StreamingMarkerTraces> StreamingMocapLab::getMarkerTraces()
{
  return mMarkerTraces;
}

} // namespace biomechanics
} // namespace dart