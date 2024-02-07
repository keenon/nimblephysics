#include "dart/biomechanics/StreamingIK.hpp"

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
StreamingIK::StreamingIK(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers)
  : mSkeleton(skeleton),
    mMarkers(markers),
    mSolverThreadRunning(false),
    mNumBodyNodes(skeleton->getNumBodyNodes())
{
  mLastMarkerObservations = Eigen::VectorXs(mMarkers.size() * 3);
  mLastMarkerObservationWeights = Eigen::VectorXs(mMarkers.size() * 3);

  mSkeletonBallJoints = skeleton->convertSkeletonToBallJoints();
  for (auto pair : markers)
  {
    mMarkersBallJoints.push_back(std::make_pair(
        mSkeletonBallJoints->getBodyNode(pair.first->getName()), pair.second));
  }
}

/// This cleans up the thread and any other resources used by the StreamingIK
StreamingIK::~StreamingIK()
{
  mSolverThreadRunning = false;
  if (mSolverThread.valid())
  {
    mSolverThread.get();
  }
  mGUIThreadRunning = false;
  if (mGUIThread.valid())
  {
    mGUIThread.get();
  }
}

/// This method starts the thread that runs the IK continuously.
void StreamingIK::startSolverThread()
{
  if (mSolverThreadRunning)
    return;
  mSolverThreadRunning = true;
  mSolverThread = std::async(std::launch::async, [&]() {
    Eigen::VectorXs scalesUpperBound
        = mSkeletonBallJoints->getGroupScalesUpperBound();
    Eigen::VectorXs scalesLowerBound
        = mSkeletonBallJoints->getGroupScalesLowerBound();

    Eigen::VectorXs x = Eigen::VectorXs::Zero(
        mSkeletonBallJoints->getNumDofs()
        + mSkeletonBallJoints->getGroupScaleDim());
    x.segment(0, mSkeletonBallJoints->getNumDofs())
        = mSkeletonBallJoints->getPositions();
    x.segment(
        mSkeletonBallJoints->getNumDofs(),
        mSkeletonBallJoints->getGroupScaleDim())
        = mSkeletonBallJoints->getGroupScales();

    Eigen::MatrixXs J = Eigen::MatrixXs::Zero(
        (mMarkers.size() * 3),
        mSkeletonBallJoints->getNumDofs()
            + mSkeletonBallJoints->getGroupScaleDim());

    s_t lastError = std::numeric_limits<s_t>::infinity();
    s_t lr = 1e-3;

    while (mSolverThreadRunning)
    {
      J.block(0, 0, mMarkers.size() * 3, mSkeletonBallJoints->getNumDofs())
          = mSkeletonBallJoints
                ->getMarkerWorldPositionsJacobianWrtJointPositions(
                    mMarkersBallJoints);
      J.block(
          0,
          mSkeletonBallJoints->getNumDofs(),
          mMarkers.size() * 3,
          mSkeletonBallJoints->getGroupScaleDim())
          = mSkeletonBallJoints->getMarkerWorldPositionsJacobianWrtGroupScales(
              mMarkersBallJoints);
      Eigen::VectorXs diff
          = (mSkeletonBallJoints->getMarkerWorldPositions(mMarkersBallJoints)
             - mLastMarkerObservations)
                .cwiseProduct(mLastMarkerObservationWeights);
      if (mLastMarkerObservationWeights.isZero())
      {
        continue;
      }

      // TODO(perf): These calls reset the skeleton pose, which then crushes our
      // cache, which is slow. Would be better to not do that...

      s_t expNegLogPdf;
      s_t anthropometricDiff;
      Eigen::VectorXs expNegLogPdfGradient;
      if (mAnthropometrics)
      {
        s_t logPdf = mAnthropometrics->getLogPDF(mSkeletonBallJoints);
        // Do the negative exponent of the logPdf.
        // The benefit here is that as the PDF gets to be a
        // larger and larger value (more probable),
        // this penalty will fall to zero.
        s_t alphaScale = 0.01;
        expNegLogPdf = mAnthropometricPriorWeight * exp(-alphaScale * logPdf);
        anthropometricDiff = expNegLogPdf * expNegLogPdf;
        expNegLogPdfGradient
            = mAnthropometricPriorWeight * -alphaScale * expNegLogPdf
              * mAnthropometrics->getGradientOfLogPDFWrtGroupScales(
                  mSkeletonBallJoints);
      }
      else
      {
        anthropometricDiff = 0;
        expNegLogPdfGradient
            = Eigen::VectorXs::Zero(mSkeletonBallJoints->getGroupScaleDim());
      }

      // Set up our adaptive learning rate
      s_t currentError = diff.squaredNorm() + anthropometricDiff;
      s_t errorChange = currentError - lastError;
      if (isnan(currentError) || (currentError > 1000 && errorChange > 100))
      {
        x.segment(0, mSkeletonBallJoints->getNumDofs()).setZero();
        x.segment(
             mSkeletonBallJoints->getNumDofs(), mSkeleton->getGroupScaleDim())
            .setOnes();
        mSkeletonBallJoints->setPositions(
            x.segment(0, mSkeletonBallJoints->getNumDofs()));
        mSkeletonBallJoints->setGroupScales(x.segment(
            mSkeletonBallJoints->getNumDofs(),
            mSkeletonBallJoints->getGroupScaleDim()));
        continue;
      }
      if (errorChange > 0)
      {
        lr *= 0.5;
        if (lr < 1e-2)
        {
          lr = 1e-2;
        }
      }
      else
      {
        // Slowly grow LR while we're safely decreasing loss
        lr *= 1.1;
      }
      lastError = currentError;

      // Actually do the update
      Eigen::VectorXs update = J.transpose() * diff;
      // Eigen::VectorXs update =
      // J.completeOrthogonalDecomposition().solve(diff);
      update.segment(
          mSkeletonBallJoints->getNumDofs(), mSkeleton->getGroupScaleDim())
          += (expNegLogPdfGradient * expNegLogPdf);
      x -= lr * update;

      // Clamp the scales
      x.segment(
          mSkeletonBallJoints->getNumDofs(),
          mSkeletonBallJoints->getGroupScaleDim())
          = x.segment(
                 mSkeletonBallJoints->getNumDofs(),
                 mSkeletonBallJoints->getGroupScaleDim())
                .cwiseMax(scalesLowerBound)
                .cwiseMin(scalesUpperBound);

      // Set the new positions
      mSkeletonBallJoints->setPositions(
          x.segment(0, mSkeletonBallJoints->getNumDofs()));
      mSkeletonBallJoints->setGroupScales(x.segment(
          mSkeletonBallJoints->getNumDofs(),
          mSkeletonBallJoints->getGroupScaleDim()));
      mSkeleton->setPositions(mSkeleton->convertPositionsFromBallSpace(
          x.segment(0, mSkeletonBallJoints->getNumDofs())));
      mSkeleton->setGroupScales(x.segment(
          mSkeletonBallJoints->getNumDofs(),
          mSkeletonBallJoints->getGroupScaleDim()));
    }
  });
}

/// This method starts a thread that periodically updates a GUI server state,
/// though at a much lower framerate than the IK solver.
void StreamingIK::startGUIThread(std::shared_ptr<server::GUIStateMachine> gui)
{
  if (mGUIThreadRunning)
    return;

  for (int i = 0; i < mMarkers.size(); i++)
  {
    gui->createBox(
        std::to_string(i),
        Eigen::Vector3s::Ones() * 0.015,
        Eigen::Vector3s::Zero(),
        Eigen::Vector3s::Zero(),
        Eigen::Vector4s(1.0, 0.0, 1.0, 1.0));
  }

  mGUIThreadRunning = true;
  mGUIThread = std::async(std::launch::async, [&, gui]() {
    while (mGUIThreadRunning)
    {
      gui->renderSkeleton(mSkeleton);
      Eigen::VectorXs virtualMarkers
          = mSkeleton->getMarkerWorldPositions(mMarkers);

      for (int i = 0; i < mMarkers.size(); i++)
      {
        if (mLastMarkerObservationWeights(i * 3) == 0)
        {
          gui->setObjectPosition(std::to_string(i), Eigen::Vector3s::Zero());
          continue;
        }
        gui->setObjectPosition(
            std::to_string(i), mLastMarkerObservations.segment<3>(i * 3));

        std::vector<Eigen::Vector3s> line;
        line.push_back(mLastMarkerObservations.segment<3>(i * 3));
        line.push_back(virtualMarkers.segment<3>(i * 3));
        gui->createLine(
            "ik_line_" + std::to_string(i), line, Eigen::Vector4s(1, 0, 0, 1));
      }
      // Don't go faster than 20fps
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
}

/// This method takes in a set of markers, along with their assigned classes,
/// and updates the targets for the IK to match the observed markers.
void StreamingIK::observeMarkers(
    std::vector<Eigen::Vector3s>& markers, std::vector<int> classes)
{
  // To go lock free, we first, before messing with any marker observations, set
  // all their weights to 0.
  mLastMarkerObservationWeights.setZero();

  // Now we can go through and update the the markers we observed, and set their
  // weights back to non-zero values.
  for (int i = 0; i < markers.size(); i++)
  {
    if (classes[i] > mNumBodyNodes
        && classes[i] < mNumBodyNodes + mMarkers.size())
    {
      int markerIndex = classes[i] - mNumBodyNodes;
      // Because we're trying to go lock free, we need to do this in a way that
      // never leaves the matrix in an inconsistent state.
      mLastMarkerObservations.segment<3>(markerIndex * 3) = markers[i];
      mLastMarkerObservationWeights.segment<3>(markerIndex * 3).setOnes();
    }
  }
}

/// This sets an anthropometric prior used to help condition the body to
/// keep reasonable scalings.
void StreamingIK::setAnthropometricPrior(
    std::shared_ptr<biomechanics::Anthropometrics> prior, s_t priorWeight)
{
  mAnthropometrics = prior;
  mAnthropometricPriorWeight = priorWeight;
}

/// This method allows tests to manually input a set of markers, rather than
/// waiting for Cortex to send them
void StreamingIK::reset(std::shared_ptr<server::GUIStateMachine> gui)
{
  mSolverThreadRunning = false;
  if (mSolverThread.valid())
  {
    mSolverThread.get();
  }

  mSkeleton->setPositions(Eigen::VectorXs::Zero(mSkeleton->getNumDofs()));
  mSkeleton->setGroupScales(
      Eigen::VectorXs::Ones(mSkeleton->getGroupScaleDim()));
  mSkeletonBallJoints->setPositions(
      Eigen::VectorXs::Zero(mSkeletonBallJoints->getNumDofs()));
  mSkeletonBallJoints->setGroupScales(
      Eigen::VectorXs::Ones(mSkeletonBallJoints->getGroupScaleDim()));

  if (gui)
  {
    // gui->deleteObjectsByPrefix("ik_line_");
    gui->renderSkeleton(mSkeleton);
  }

  startSolverThread();
}

} // namespace biomechanics
} // namespace dart