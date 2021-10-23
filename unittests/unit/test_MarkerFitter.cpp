#include <gtest/gtest.h>

#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

bool testFitterGradients(
    MarkerFitter& fitter,
    std::shared_ptr<dynamics::Skeleton>& skel,
    const std::map<
        std::string,
        std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markersMap,
    const std::map<std::string, Eigen::Vector3s>& observedMarkersMap)
{
  const s_t THRESHOLD = 2e-7;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  std::vector<std::pair<int, Eigen::Vector3s>> observedMarkers;
  int offset = 0;
  std::map<std::string, int> markerOffsets;
  for (auto pair : markersMap)
  {
    markerOffsets[pair.first] = offset;
    offset++;
    markers.push_back(pair.second);
  }
  for (auto pair : observedMarkersMap)
  {
    observedMarkers.emplace_back(markerOffsets[pair.first], pair.second);
  }

  Eigen::VectorXs gradWrtJoints = fitter.getMarkerLossGradientWrtJoints(
      skel,
      markers,
      fitter.getIKLossGradWrtMarkerError(
          fitter.getMarkerError(skel, markers, observedMarkers)));
  Eigen::VectorXs gradWrtJoints_fd
      = fitter.finiteDifferenceSquaredMarkerLossGradientWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(gradWrtJoints, gradWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt joints" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJoints << std::endl
              << "FD:" << std::endl
              << gradWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJoints - gradWrtJoints_fd << std::endl;
    return false;
  }

  Eigen::VectorXs gradWrtScales = fitter.getMarkerLossGradientWrtGroupScales(
      skel,
      markers,
      fitter.getIKLossGradWrtMarkerError(
          fitter.getMarkerError(skel, markers, observedMarkers)));
  Eigen::VectorXs gradWrtScales_fd
      = fitter.finiteDifferenceSquaredMarkerLossGradientWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(gradWrtScales, gradWrtScales_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtScales << std::endl
              << "FD:" << std::endl
              << gradWrtScales_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtScales - gradWrtScales_fd << std::endl;
    return false;
  }

  Eigen::VectorXs gradWrtMarkerOffsets
      = fitter.getMarkerLossGradientWrtMarkerOffsets(
          skel,
          markers,
          fitter.getIKLossGradWrtMarkerError(
              fitter.getMarkerError(skel, markers, observedMarkers)));
  Eigen::VectorXs gradWrtMarkerOffsets_fd
      = fitter.finiteDifferenceSquaredMarkerLossGradientWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(gradWrtMarkerOffsets, gradWrtMarkerOffsets_fd, THRESHOLD))
  {
    std::cout << "Error on grad wrt marker offsets" << std::endl
              << "Analytical:" << std::endl
              << gradWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << gradWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtMarkerOffsets - gradWrtMarkerOffsets_fd << std::endl;
    return false;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Second order Jacobians (and their input components)
  /////////////////////////////////////////////////////////////////////////////

  std::vector<int> sparsityMap
      = fitter.getSparsityMap(markers, observedMarkers);

  Eigen::MatrixXs markerErrorJacWrtJoints
      = fitter.getMarkerErrorJacobianWrtJoints(skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtJoints_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(markerErrorJacWrtJoints, markerErrorJacWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt joints" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtJoints << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtJoints - markerErrorJacWrtJoints_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtJoints
      = fitter.getIKLossGradientWrtJointsJacobianWrtJoints(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtJoints_fd
      = fitter.finiteDifferenceIKLossGradientWrtJointsJacobianWrtJoints(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtJoints, gradWrtJointsJacWrtJoints_fd, THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt joints" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtJoints << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtJoints_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtJoints - gradWrtJointsJacWrtJoints_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs markerErrorJacWrtGroupScales
      = fitter.getMarkerErrorJacobianWrtGroupScales(skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtGroupScales_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(
          markerErrorJacWrtGroupScales,
          markerErrorJacWrtGroupScales_fd,
          THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtGroupScales << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtGroupScales_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtGroupScales - markerErrorJacWrtGroupScales_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtGroupScales
      = fitter.getIKLossGradientWrtJointsJacobianWrtGroupScales(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtGroupScales_fd
      = fitter.finiteDifferenceIKLossGradientWrtJointsJacobianWrtGroupScales(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtGroupScales,
          gradWrtJointsJacWrtGroupScales_fd,
          THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtGroupScales << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtGroupScales_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtGroupScales
                     - gradWrtJointsJacWrtGroupScales_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs markerErrorJacWrtMarkerOffsets
      = fitter.getMarkerErrorJacobianWrtMarkerOffsets(
          skel, markers, sparsityMap);
  Eigen::MatrixXs markerErrorJacWrtMarkerOffsets_fd
      = fitter.finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(
          markerErrorJacWrtMarkerOffsets,
          markerErrorJacWrtMarkerOffsets_fd,
          THRESHOLD))
  {
    std::cout << "Error on marker error jac wrt marker offsets" << std::endl
              << "Analytical:" << std::endl
              << markerErrorJacWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << markerErrorJacWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << markerErrorJacWrtMarkerOffsets
                     - markerErrorJacWrtMarkerOffsets_fd
              << std::endl;
    return false;
  }

  Eigen::MatrixXs gradWrtJointsJacWrtMarkerOffsets
      = fitter.getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtMarkerOffsets_fd
      = fitter.finiteDifferenceIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
          skel, markers, observedMarkers);

  if (!equals(
          gradWrtJointsJacWrtMarkerOffsets,
          gradWrtJointsJacWrtMarkerOffsets_fd,
          THRESHOLD))
  {
    std::cout << "Error on (grad wrt joints) jac wrt group scales" << std::endl
              << "Analytical:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets << std::endl
              << "FD:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets_fd << std::endl
              << "Diff:" << std::endl
              << gradWrtJointsJacWrtMarkerOffsets
                     - gradWrtJointsJacWrtMarkerOffsets_fd
              << std::endl;
    return false;
  }

  return true;
}

bool testBilevelFitProblemGradients(
    MarkerFitter& fitter,
    int numPoses,
    double markerDropProb,
    std::shared_ptr<dynamics::Skeleton>& skel,
    std::vector<dynamics::Joint*> joints,
    const std::map<
        std::string,
        std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markersMap)
{
  const s_t THRESHOLD = 5e-8;

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  for (auto pair : markersMap)
  {
    markers.push_back(pair.second);
  }

  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  srand(42);

  // 1. Generate a bunch of marker data for the skeleton in random
  // configurations
  Eigen::VectorXs goldGroupScales
      = Eigen::VectorXs::Ones(originalGroupScales.size())
        + Eigen::VectorXs::Random(originalGroupScales.size()) * 0.07;
  Eigen::VectorXs goldMarkerOffsets
      = Eigen::VectorXs::Random(markers.size() * 3) * 0.05;
  Eigen::MatrixXs goldPoses
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), numPoses);
  std::vector<std::map<std::string, Eigen::Vector3s>> observations;
  Eigen::MatrixXs goldJointCenters
      = Eigen::MatrixXs::Zero(joints.size() * 3, numPoses);

  for (int i = 0; i < numPoses; i++)
  {
    Eigen::VectorXs goldPose = Eigen::VectorXs::Random(skel->getNumDofs());
    goldPoses.col(i) = goldPose;
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
        = fitter.setConfiguration(
            skel, goldPose, goldGroupScales, goldMarkerOffsets);
    Eigen::VectorXs markerWorldPoses = skel->getMarkerWorldPositions(markers);
    goldJointCenters.col(i) = skel->getJointWorldPositions(joints);
    // Take an observation
    std::map<std::string, Eigen::Vector3s> obs;
    for (int j = 0; j < markers.size(); j++)
    {
      if (((double)rand() / RAND_MAX) > markerDropProb)
      {
        obs[fitter.getMarkerNameAtIndex(j)]
            = Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3));
      }
    }
    observations.push_back(obs);
  }

  // 2. Reset the skeleton configuration
  skel->setPositions(Eigen::VectorXs::Zero(skel->getNumDofs()));
  skel->setGroupScales(originalGroupScales);

  // 3. Create a BilevelFitProblem
  std::shared_ptr<BilevelFitResult> tmpResult
      = std::make_shared<BilevelFitResult>();

  MarkerInitialization init;
  init.poses = goldPoses
               + Eigen::MatrixXs::Random(skel->getNumDofs(), numPoses) * 0.07;
  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    init.markerOffsets[fitter.getMarkerNameAtIndex(i)]
        = goldMarkerOffsets.segment<3>(i * 3)
          + Eigen::Vector3s::Random() * 0.001;
  }
  init.joints = joints;
  init.jointCenters
      = goldJointCenters
        + Eigen::MatrixXs::Random(joints.size() * 3, numPoses) * 0.07;
  init.groupScales = originalGroupScales;

  BilevelFitProblem problem(&fitter, observations, init, numPoses, tmpResult);

  Eigen::VectorXs x = problem.getInitialization();

  Eigen::VectorXs grad = problem.getGradient(x);
  Eigen::VectorXs grad_fd = problem.finiteDifferenceGradient(x);

  if (!equals(grad, grad_fd, THRESHOLD))
  {
    std::cout << "Error on BilevelFitProblem grad" << std::endl
              << "Analytical:" << std::endl
              << grad << std::endl
              << "FD:" << std::endl
              << grad_fd << std::endl
              << "Diff:" << std::endl
              << grad - grad_fd << std::endl;

    Eigen::VectorXs diff = grad - grad_fd;
    int scaleDim = skel->getGroupScaleDim();
    int markerDim = markersMap.size() * 3;

    Eigen::VectorXs scale = grad.segment(0, scaleDim);
    Eigen::VectorXs scale_fd = grad_fd.segment(0, scaleDim);
    if (!equals(scale, scale_fd, THRESHOLD))
    {
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(scale.size(), 3);
      compare.col(0) = scale;
      compare.col(1) = scale_fd;
      compare.col(2) = scale - scale_fd;
      std::cout << "Error on BilevelFitProblem scales grad" << std::endl
                << "Analytical - FD - Diff" << std::endl
                << compare << std::endl;
    }

    Eigen::VectorXs markers = grad.segment(scaleDim, markerDim);
    Eigen::VectorXs markers_fd = grad_fd.segment(scaleDim, markerDim);
    if (!equals(markers, markers_fd, THRESHOLD))
    {
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(markers.size(), 3);
      compare.col(0) = markers;
      compare.col(1) = markers_fd;
      compare.col(2) = markers - markers_fd;
      std::cout << "Error on BilevelFitProblem marker offsets grad" << std::endl
                << "Analytical - FD - Diff" << std::endl
                << compare << std::endl;
    }

    return false;
  }

  Eigen::MatrixXs jac = problem.getConstraintsJacobian(x);
  Eigen::MatrixXs jac_fd = problem.finiteDifferenceConstraintsJacobian(x);

  if (!equals(jac, jac_fd, THRESHOLD))
  {
    Eigen::MatrixXs jacScales
        = jac.block(0, 0, jac.rows(), skel->getGroupScaleDim());
    Eigen::MatrixXs jacScales_fd
        = jac_fd.block(0, 0, jac.rows(), skel->getGroupScaleDim());
    if (!equals(jacScales, jacScales_fd, THRESHOLD))
    {
      std::cout << "Error on BilevelFitProblem constraint jac, scales block"
                << std::endl
                << "Analytical:" << std::endl
                << jacScales << std::endl
                << "FD:" << std::endl
                << jacScales_fd << std::endl
                << "Diff:" << std::endl
                << jacScales - jacScales_fd << std::endl;
    }

    Eigen::MatrixXs jacMarkers = jac.block(
        0, skel->getGroupScaleDim(), jac.rows(), markers.size() * 3);
    Eigen::MatrixXs jacMarkers_fd = jac_fd.block(
        0, skel->getGroupScaleDim(), jac.rows(), markers.size() * 3);
    if (!equals(jacMarkers, jacMarkers_fd, THRESHOLD))
    {
      std::cout << "Error on BilevelFitProblem constraint jac, markers block"
                << std::endl
                << "Analytical:" << std::endl
                << jacMarkers << std::endl
                << "FD:" << std::endl
                << jacMarkers_fd << std::endl
                << "Diff:" << std::endl
                << jacMarkers - jacMarkers_fd << std::endl;
    }

    int offset = (skel->getGroupScaleDim()) + (markers.size() * 3);
    for (int i = 0; i < numPoses; i++)
    {
      Eigen::MatrixXs jacPos = jac.block(
          0, offset + (skel->getNumDofs() * i), jac.rows(), skel->getNumDofs());
      Eigen::MatrixXs jacPos_fd = jac_fd.block(
          0, offset + (skel->getNumDofs() * i), jac.rows(), skel->getNumDofs());
      if (!equals(jacPos, jacPos_fd, THRESHOLD))
      {
        std::cout << "Error on BilevelFitProblem constraint jac, pos " << i
                  << " block" << std::endl
                  << "Analytical:" << std::endl
                  << jacPos << std::endl
                  << "FD:" << std::endl
                  << jacPos_fd << std::endl
                  << "Diff:" << std::endl
                  << jacPos - jacPos_fd << std::endl;
      }
    }

    return false;
  }

  return true;
}

bool testSolveBilevelFitProblem(
    std::shared_ptr<dynamics::Skeleton>& skel,
    int numPoses,
    double markerDropProb,
    double markerErrorBounds = 0.001,
    double bodyScaleBounds = 0.1)
{
  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  // Provide three markers per body, to give enough data to make things
  // unambiguos
  dynamics::MarkerMap markers;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    markers[std::to_string(i) + "_0"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitX() * 0.05);
    markers[std::to_string(i) + "_1"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitY() * 0.05);
    markers[std::to_string(i) + "_2"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitZ() * 0.05);
  }

  MarkerFitter fitter(skel, markers);

  srand(42);

  // 1. Generate a bunch of marker data for the skeleton in random
  // configurations
  Eigen::VectorXs goldGroupScales
      = Eigen::VectorXs::Ones(originalGroupScales.size())
        + Eigen::VectorXs::Random(originalGroupScales.size()) * bodyScaleBounds;
  Eigen::VectorXs goldMarkerOffsets
      = Eigen::VectorXs::Random(markers.size() * 3) * markerErrorBounds;
  std::vector<Eigen::VectorXs> goldPoses;
  std::vector<std::map<std::string, Eigen::Vector3s>> observations;

  Eigen::VectorXs goldX = Eigen::VectorXs::Zero(
      goldGroupScales.size() + goldMarkerOffsets.size()
      + (skel->getNumDofs() * numPoses));
  goldX.segment(0, goldGroupScales.size()) = goldGroupScales;
  goldX.segment(goldGroupScales.size(), goldMarkerOffsets.size())
      = goldMarkerOffsets;

  for (int i = 0; i < numPoses; i++)
  {
    Eigen::VectorXs goldPose = skel->getRandomPose();

    std::cout << "Pose " << i << ":" << std::endl;
    std::cout << "Eigen::VectorXs pose = Eigen::VectorXs(" << goldPose.size()
              << ");" << std::endl;
    std::cout << "pose << ";
    for (int j = 0; j < goldPose.size(); j++)
    {
      if (j > 0)
        std::cout << ", ";
      std::cout << goldPose(j);
    }
    std::cout << std::endl;

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers
        = fitter.setConfiguration(
            skel, goldPose, goldGroupScales, goldMarkerOffsets);
    Eigen::VectorXs markerWorldPoses = skel->getMarkerWorldPositions(markers);
    // Take an observation
    std::map<std::string, Eigen::Vector3s> obs;
    for (int j = 0; j < markers.size(); j++)
    {
      if (((double)rand() / RAND_MAX) > markerDropProb)
      {
        obs[fitter.getMarkerNameAtIndex(j)]
            = Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3));
      }
    }
    observations.push_back(obs);

    goldX.segment(
        goldGroupScales.size() + goldMarkerOffsets.size()
            + (skel->getNumDofs() * i),
        skel->getNumDofs())
        = goldPose;
  }

  // 2. Reset the skeleton configuration
  skel->setPositions(Eigen::VectorXs::Zero(skel->getNumDofs()));
  skel->setGroupScales(originalGroupScales);

  // 3. Create a BilevelFitProblem
  std::shared_ptr<BilevelFitResult> tmpResult
      = std::make_shared<BilevelFitResult>();

  MarkerInitialization init;
  init.poses = Eigen::MatrixXs::Zero(skel->getNumDofs(), numPoses);
  for (int i = 0; i < numPoses; i++)
  {
    init.poses.col(i)
        = goldPoses[i] + Eigen::VectorXs::Random(skel->getNumDofs()) * 0.07;
  }
  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    init.markerOffsets[fitter.getMarkerNameAtIndex(i)]
        = goldMarkerOffsets.segment<3>(i * 3)
          + Eigen::Vector3s::Random() * 0.001;
  }

  BilevelFitProblem problem(&fitter, observations, init, numPoses, tmpResult);

  s_t lossAtGold = problem.getLoss(goldX);
  if (lossAtGold != 0)
  {
    std::cout << "Loss at gold: " << lossAtGold << std::endl;
    return false;
  }

  Eigen::VectorXs initialGuess = problem.getInitialization();
  s_t initialLoss = problem.getLoss(initialGuess);
  if (initialLoss > 0.2)
  {
    std::cout << "Initial guess was bad. Expected a loss < 0.2, but got "
              << initialLoss << std::endl;
    return false;
  }

  // Try running IPOPT
  std::shared_ptr<BilevelFitResult> result
      = fitter.optimizeBilevel(observations, init, numPoses);

  Eigen::VectorXs groupScaleError = result->groupScales - goldGroupScales;
  Eigen::MatrixXs groupScaleCols = Eigen::MatrixXs(groupScaleError.size(), 3);
  groupScaleCols.col(0) = goldGroupScales;
  groupScaleCols.col(1) = result->groupScales;
  groupScaleCols.col(2) = groupScaleError;

  Eigen::VectorXs markerOffsetError
      = result->rawMarkerOffsets - goldMarkerOffsets;
  Eigen::MatrixXs markerOffsetCols
      = Eigen::MatrixXs(markerOffsetError.size(), 3);
  markerOffsetCols.col(0) = goldMarkerOffsets;
  markerOffsetCols.col(1) = result->rawMarkerOffsets;
  markerOffsetCols.col(2) = markerOffsetError;

  std::cout << "Gold group scales - Recovered - Error: " << std::endl
            << groupScaleCols << std::endl;
  std::cout << "Gold marker offsets - Recovered - Error: " << std::endl
            << markerOffsetCols << std::endl;

  return true;
}

bool debugIKInitializationToGUI(
    std::shared_ptr<dynamics::Skeleton>& skel,
    Eigen::VectorXs pos,
    double markerDropProb)
{
  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);
  server.setAutoflush(false);

  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  // Provide three markers per body, to give enough data to make things
  // unambiguos
  dynamics::MarkerMap markers;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    markers[std::to_string(i) + "_0"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitX() * 0.05);
    markers[std::to_string(i) + "_1"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitY() * 0.05);
    markers[std::to_string(i) + "_2"]
        = std::make_pair(skel->getBodyNode(i), Eigen::Vector3s::UnitZ() * 0.05);
  }

  MarkerFitter fitter(skel, markers);

  srand(42);

  // 1. Generate a bunch of marker data for the skeleton in random
  // configurations
  Eigen::VectorXs goldGroupScales
      = originalGroupScales
        + Eigen::VectorXs::Random(originalGroupScales.size()) * 0.2;
  Eigen::VectorXs goldMarkerOffsets
      = Eigen::VectorXs::Random(markers.size() * 3) * 0.0;

  Eigen::VectorXs goldPose = pos;
  goldPose.segment<3>(3).setZero();
  /*
  goldPose(skel->getJoint("walker_knee_r")->getDof(0)->getIndexInSkeleton())
      = 0.0;
  goldPose(skel->getJoint("walker_knee_l")->getDof(0)->getIndexInSkeleton())
      = 0.0;
  */
  /*
  goldPose(skel->getJoint("ankle_r")->getDof(0)->getIndexInSkeleton()) = 0.0;
  goldPose(skel->getJoint("ankle_l")->getDof(0)->getIndexInSkeleton()) = 0.0;
  */

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      adjustedMarkersSkel = fitter.setConfiguration(
          skel, goldPose, goldGroupScales, goldMarkerOffsets);

  // std::shared_ptr<dynamics::Skeleton> goldTarget = skel->clone();
  // goldTarget->setPositions(goldPose);
  server.renderSkeleton(skel, "goldSkel");

  std::shared_ptr<dynamics::Skeleton> skelBallJoints
      = skel->convertSkeletonToBallJoints();
  skelBallJoints->setPositions(
      skel->convertPositionsToBallSpace(skel->getPositions()));

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> adjustedMarkers;
  for (auto pair : adjustedMarkersSkel)
  {
    adjustedMarkers.emplace_back(
        skelBallJoints->getBodyNode(pair.first->getName()),
        Eigen::Vector3s(pair.second));
  }

  Eigen::VectorXs markerWorldPoses
      = skelBallJoints->getMarkerWorldPositions(adjustedMarkers);
  // Take an observation
  std::map<std::string, Eigen::Vector3s> obs;
  for (int j = 0; j < markers.size(); j++)
  {
    if (((double)rand() / RAND_MAX) > markerDropProb)
    {
      obs[fitter.getMarkerNameAtIndex(j)]
          = Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3));
    }
  }

  while (true)
  {
    // 2. Reset the skeleton configuration
    skel->setPositions(skel->getRandomPose());
    skelBallJoints->setPositions(
        skel->convertPositionsToBallSpace(skel->getPositions()));
    skel->setGroupScales(originalGroupScales);
    skelBallJoints->setGroupScales(originalGroupScales);
    // skel->setPositions(goldPose);
    // skel->setGroupScales(goldGroupScales);

    Eigen::VectorXs initialPos = Eigen::VectorXs::Zero(
        skelBallJoints->getNumDofs() + skelBallJoints->getGroupScaleDim());
    /*
    initialPos.segment(0, skelBallJoints->getNumDofs())
        = skelBallJoints->getPositions();
    initialPos.segment(
        skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim())
        = skelBallJoints->getGroupScales();
        Eigen::VectorXs initialGuess = Eigen::VectorXs(97);
    */
    initialPos << -0.329172, -0.238389, -1.51384, -1.95121, 1.01627, -2.14055,
        0.479686, -0.390676, 0.825582, 1.20487, 0.0764906, 0.117637, -0.28067,
        -0.692632, 0.443975, 0.388073, 1.78741, 0.288679, -0.097154, 0.184243,
        0.268635, 1.0733, 1.59988, -0.737188, -1.03813, -0.236178, 0.823967,
        1.32797, 0.495846, 0.406708, -0.442464, 0.698941, -0.317397, 2.18822,
        0.0254939, 1.11635, 0.588954, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    math::solveIK(
        initialPos,
        adjustedMarkers.size() * 3,
        [skel,
         skelBallJoints,
         adjustedMarkers,
         adjustedMarkersSkel,
         markerWorldPoses,
         &server](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          skelBallJoints->setPositions(
              pos.segment(0, skelBallJoints->getNumDofs()));
          skel->setPositions(skel->convertPositionsFromBallSpace(
              pos.segment(0, skelBallJoints->getNumDofs())));
          if (clamp)
          {
            std::cout << "Eigen::VectorXs val = Eigen::VectorXs("
                      << pos.segment(0, skelBallJoints->getNumDofs()).size()
                      << ");" << std::endl;
            std::cout << "ballJointsPos << ";
            for (int j = 0; j < skelBallJoints->getNumDofs(); j++)
            {
              if (j > 0)
                std::cout << ", ";
              std::cout << skelBallJoints->getPositions()(j);
            }
            std::cout << std::endl;

            skel->clampPositionsToLimits();
            skelBallJoints->setPositions(
                skel->convertPositionsToBallSpace(skel->getPositions()));
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = skelBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = skelBallJoints->getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          skel->setGroupScales(newScales);
          skelBallJoints->setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, skelBallJoints->getNumDofs())
              = skelBallJoints->getPositions();
          clampedPos.segment(
              skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim())
              = newScales;

          // Debug to a GUI
          server.renderSkeleton(skel);
          Eigen::VectorXs currentMarkerPoses
              = skel->getMarkerWorldPositions(adjustedMarkersSkel);
          for (int i = 0; i < adjustedMarkersSkel.size(); i++)
          {
            Eigen::Vector3s source = currentMarkerPoses.segment<3>(i * 3);
            Eigen::Vector3s goal = markerWorldPoses.segment<3>(i * 3);
            std::vector<Eigen::Vector3s> line;
            line.push_back(source);
            line.push_back(goal);

            server.createLine(
                "marker_error_" + std::to_string(i),
                line,
                Eigen::Vector3s::UnitX());
          }
          server.flush();

          return clampedPos;
        },
        [skelBallJoints, markerWorldPoses, adjustedMarkers](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = markerWorldPoses
                 - skelBallJoints->getMarkerWorldPositions(adjustedMarkers);
          assert(
              jac.cols()
              == skelBallJoints->getNumDofs()
                     + skelBallJoints->getGroupScaleDim());
          assert(jac.rows() == adjustedMarkers.size() * 3);
          jac.setZero();
          jac.block(
              0, 0, adjustedMarkers.size() * 3, skelBallJoints->getNumDofs())
              = skelBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        adjustedMarkers);
          jac.block(
              0,
              skelBallJoints->getNumDofs(),
              adjustedMarkers.size() * 3,
              skelBallJoints->getGroupScaleDim())
              = skelBallJoints->getMarkerWorldPositionsJacobianWrtGroupScales(
                  adjustedMarkers);
        },
        [skel, skelBallJoints](Eigen::VectorXs& val) {
          val.segment(0, skelBallJoints->getNumDofs())
              = skel->convertPositionsToBallSpace(skel->getRandomPose());
          val.segment(
                 skelBallJoints->getNumDofs(),
                 skelBallJoints->getGroupScaleDim())
              .setConstant(1.0);

          std::cout << "Eigen::VectorXs initialGuess = Eigen::VectorXs("
                    << val.size() << ");" << std::endl;
          std::cout << "initialGuess << ";
          for (int j = 0; j < val.size(); j++)
          {
            if (j > 0)
              std::cout << ", ";
            std::cout << val(j);
          }
          std::cout << std::endl;
        },
        math::IKConfig().setLogOutput(true).setMaxRestarts(1));

    Eigen::VectorXs finalMarkers
        = skel->getMarkerWorldPositions(adjustedMarkers);
    Eigen::VectorXs diff = markerWorldPoses - finalMarkers;
    for (int i = 0; i < adjustedMarkers.size(); i++)
    {
      auto pair = adjustedMarkers[i];
      Eigen::Vector3s markerDiff = diff.segment<3>(i * 3);
      s_t markerError = markerDiff.squaredNorm();
      std::cout << "Error on marker (" << pair.first->getName() << ", ["
                << pair.second(0) << "," << pair.second(1) << ","
                << pair.second(2) << "]): " << markerError << std::endl;
    }
  }

  server.blockWhileServing();

  return true;
}

bool debugFitToGUI(
    std::shared_ptr<dynamics::Skeleton>& skel,
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        adjustedMarkersSkel,
    Eigen::VectorXs markerWorldPoses,
    std::shared_ptr<dynamics::Skeleton>& goldSkel,
    Eigen::VectorXs goldTarget)
{
  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);
  server.setAutoflush(false);

  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  // std::shared_ptr<dynamics::Skeleton> goldTarget = skel->clone();
  // goldTarget->setPositions(goldPose);
  goldSkel->setPositions(goldTarget);
  server.renderSkeleton(goldSkel, "gold");

  std::shared_ptr<dynamics::Skeleton> skelBallJoints
      = skel->convertSkeletonToBallJoints();
  skelBallJoints->setPositions(
      skel->convertPositionsToBallSpace(skel->getPositions()));

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> adjustedMarkers;
  for (auto pair : adjustedMarkersSkel)
  {
    adjustedMarkers.emplace_back(
        skelBallJoints->getBodyNode(pair.first->getName()),
        Eigen::Vector3s(pair.second));
  }

  while (true)
  {
    // 2. Reset the skeleton configuration
    skel->setPositions(skel->getRandomPose());
    // skel->setPositions(goldTarget);
    skelBallJoints->setPositions(
        skel->convertPositionsToBallSpace(skel->getPositions()));
    skel->setGroupScales(originalGroupScales);
    skelBallJoints->setGroupScales(originalGroupScales);
    // skel->setPositions(goldPose);
    // skel->setGroupScales(goldGroupScales);

    Eigen::VectorXs initialPos = Eigen::VectorXs::Zero(
        skelBallJoints->getNumDofs() + skelBallJoints->getGroupScaleDim());
    initialPos.segment(0, skelBallJoints->getNumDofs())
        = skelBallJoints->getPositions();
    initialPos.segment(
        skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim())
        = skelBallJoints->getGroupScales();
    Eigen::VectorXs initialGuess = Eigen::VectorXs(97);

    math::solveIK(
        initialPos,
        adjustedMarkers.size() * 3,
        [skel,
         skelBallJoints,
         adjustedMarkers,
         adjustedMarkersSkel,
         markerWorldPoses,
         &server](
            /* in*/ const Eigen::VectorXs pos, bool clamp) {
          // Set positions
          skelBallJoints->setPositions(
              pos.segment(0, skelBallJoints->getNumDofs()));
          skel->setPositions(skel->convertPositionsFromBallSpace(
              pos.segment(0, skelBallJoints->getNumDofs())));
          if (clamp)
          {
            std::cout << "Eigen::VectorXs val = Eigen::VectorXs("
                      << pos.segment(0, skelBallJoints->getNumDofs()).size()
                      << ");" << std::endl;
            std::cout << "ballJointsPos << ";
            for (int j = 0; j < skelBallJoints->getNumDofs(); j++)
            {
              if (j > 0)
                std::cout << ", ";
              std::cout << skelBallJoints->getPositions()(j);
            }
            std::cout << std::endl;

            skel->clampPositionsToLimits();
            skelBallJoints->setPositions(
                skel->convertPositionsToBallSpace(skel->getPositions()));
          }

          // Set scales
          Eigen::VectorXs newScales = pos.segment(
              skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim());
          Eigen::VectorXs scalesUpperBound
              = skelBallJoints->getGroupScalesUpperBound();
          Eigen::VectorXs scalesLowerBound
              = skelBallJoints->getGroupScalesLowerBound();
          newScales = newScales.cwiseMax(scalesLowerBound);
          newScales = newScales.cwiseMin(scalesUpperBound);
          // This effectively disables scaling
          // newScales.setConstant(1.0);

          skel->setGroupScales(newScales);
          skelBallJoints->setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, skelBallJoints->getNumDofs())
              = skelBallJoints->getPositions();
          clampedPos.segment(
              skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim())
              = newScales;

          // Debug to a GUI
          server.renderSkeleton(skel);
          Eigen::VectorXs currentMarkerPoses
              = skel->getMarkerWorldPositions(adjustedMarkersSkel);
          for (int i = 0; i < adjustedMarkersSkel.size(); i++)
          {
            Eigen::Vector3s source = currentMarkerPoses.segment<3>(i * 3);
            Eigen::Vector3s goal = markerWorldPoses.segment<3>(i * 3);
            std::vector<Eigen::Vector3s> line;
            line.push_back(source);
            line.push_back(goal);

            server.createLine(
                "marker_error_" + std::to_string(i),
                line,
                Eigen::Vector3s::UnitX());
          }
          server.flush();

          return clampedPos;
        },
        [skelBallJoints, markerWorldPoses, adjustedMarkers](
            /*out*/ Eigen::VectorXs& diff,
            /*out*/ Eigen::MatrixXs& jac) {
          diff = markerWorldPoses
                 - skelBallJoints->getMarkerWorldPositions(adjustedMarkers);
          assert(
              jac.cols()
              == skelBallJoints->getNumDofs()
                     + skelBallJoints->getGroupScaleDim());
          assert(jac.rows() == adjustedMarkers.size() * 3);
          jac.setZero();
          jac.block(
              0, 0, adjustedMarkers.size() * 3, skelBallJoints->getNumDofs())
              = skelBallJoints
                    ->getMarkerWorldPositionsJacobianWrtJointPositions(
                        adjustedMarkers);
          jac.block(
              0,
              skelBallJoints->getNumDofs(),
              adjustedMarkers.size() * 3,
              skelBallJoints->getGroupScaleDim())
              = skelBallJoints->getMarkerWorldPositionsJacobianWrtGroupScales(
                  adjustedMarkers);
        },
        [skel, skelBallJoints](Eigen::VectorXs& val) {
          val.segment(0, skelBallJoints->getNumDofs())
              = skel->convertPositionsToBallSpace(skel->getRandomPose());
          val.segment(
                 skelBallJoints->getNumDofs(),
                 skelBallJoints->getGroupScaleDim())
              .setConstant(1.0);

          std::cout << "Eigen::VectorXs initialGuess = Eigen::VectorXs("
                    << val.size() << ");" << std::endl;
          std::cout << "initialGuess << ";
          for (int j = 0; j < val.size(); j++)
          {
            if (j > 0)
              std::cout << ", ";
            std::cout << val(j);
          }
          std::cout << std::endl;
        },
        math::IKConfig().setLogOutput(true).setMaxRestarts(1));

    Eigen::VectorXs finalMarkers
        = skel->getMarkerWorldPositions(adjustedMarkers);
    Eigen::VectorXs diff = markerWorldPoses - finalMarkers;
    for (int i = 0; i < adjustedMarkers.size(); i++)
    {
      auto pair = adjustedMarkers[i];
      Eigen::Vector3s markerDiff = diff.segment<3>(i * 3);
      s_t markerError = markerDiff.squaredNorm();
      std::cout << "Error on marker (" << pair.first->getName() << ", ["
                << pair.second(0) << "," << pair.second(1) << ","
                << pair.second(2) << "]): " << markerError << std::endl;
    }
  }

  server.blockWhileServing();

  return true;
}

void debugTrajectoryAndMarkersToGUI(
    std::shared_ptr<dynamics::Skeleton>& skel,
    std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        markers,
    Eigen::MatrixXs poses,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerTrajectories,
    Eigen::MatrixXs jointCenters = Eigen::MatrixXs::Zero(0, 0))
{
  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(skel);
  server.setAutoflush(false);

  int numJoints = jointCenters.rows() / 3;
  for (int i = 0; i < numJoints; i++)
  {
    server.createSphere(
        "joint_center_" + i,
        0.02,
        Eigen::Vector3s::Zero(),
        Eigen::Vector3s(0, 0, 1));
  }

  int timestep = 0;
  Ticker ticker(1.0 / 50);
  ticker.registerTickListener([&](long) {
    skel->setPositions(poses.col(timestep));
    server.renderSkeleton(skel);

    std::map<std::string, Eigen::Vector3s> markerWorldPositions
        = markerTrajectories[timestep];
    server.deleteObjectsByPrefix("marker_error_");
    for (auto pair : markerWorldPositions)
    {
      Eigen::Vector3s worldObserved = pair.second;
      Eigen::Vector3s worldInferred
          = markers[pair.first].first->getWorldTransform()
            * (markers[pair.first].second.cwiseProduct(
                markers[pair.first].first->getScale()));
      std::vector<Eigen::Vector3s> points;
      points.push_back(worldObserved);
      points.push_back(worldInferred);
      server.createLine(
          "marker_error_" + pair.first, points, Eigen::Vector3s::UnitX());
    }

    for (int i = 0; i < numJoints; i++)
    {
      server.setObjectPosition(
          "joint_center_" + i, jointCenters.block<3, 1>(i * 3, timestep));
    }

    server.flush();

    timestep++;
    if (timestep >= poses.cols())
    {
      timestep = 0;
    }
  });
  server.registerConnectionListener([&]() { ticker.start(); });
  server.blockWhileServing();
}

#ifdef ALL_TESTS
TEST(MarkerFitter, ROTATE_IN_BOUNDS)
{
  /////////////////////////////////////////////////////////////////////
  // The problem here is that when we convert a rotation matrix back to euler
  // joints, there are infinitely many possible coordinate values we could take,
  // but we've got to respect the joint limits. If we get a converted result
  // that's outside of joint limits, then when we trim to joint limits we break
  // everything and the IK solver fails to find a good solution. This is not
  // acceptable, so we need to find a way to convert while respecting joint
  // bounds.
  /////////////////////////////////////////////////////////////////////

  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector3s angles = Eigen::Vector3s::Random() * 2 * M_PI;

    Eigen::Vector3s alternateAngles
        = Eigen::Vector3s(angles(0) + M_PI, M_PI - angles(1), angles(2) + M_PI);

    Eigen::Matrix3s R = math::eulerZXYToMatrix(angles);
    Eigen::Matrix3s Ralt = math::eulerZXYToMatrix(alternateAngles);

    if (!equals(R, Ralt, 1e-12))
    {
      std::cout << "Failed alternate strategy flip ZXY!\n"
                << angles << "\nAlternate:\n"
                << alternateAngles << std::endl;
      EXPECT_TRUE(equals(R, Ralt, 1e-12));
      return;
    }

    R = math::eulerZYXToMatrix(angles);
    Ralt = math::eulerZYXToMatrix(alternateAngles);

    if (!equals(R, Ralt, 1e-12))
    {
      std::cout << "Failed alternate strategy flip ZYX!\n"
                << angles << "\nAlternate:\n"
                << alternateAngles << std::endl;
      EXPECT_TRUE(equals(R, Ralt, 1e-12));
      return;
    }

    R = math::eulerXZYToMatrix(angles);
    Ralt = math::eulerXZYToMatrix(alternateAngles);

    if (!equals(R, Ralt, 1e-12))
    {
      std::cout << "Failed alternate strategy flip XZY!\n"
                << angles << "\nAlternate:\n"
                << alternateAngles << std::endl;
      EXPECT_TRUE(equals(R, Ralt, 1e-12));
      return;
    }

    R = math::eulerXYZToMatrix(angles);
    Ralt = math::eulerXYZToMatrix(alternateAngles);

    if (!equals(R, Ralt, 1e-12))
    {
      std::cout << "Failed alternate strategy flip XYZ!\n"
                << angles << "\nAlternate:\n"
                << alternateAngles << std::endl;
      EXPECT_TRUE(equals(R, Ralt, 1e-12));
      return;
    }
  }

  Eigen::Matrix3s R;
  R << -0.155332, -0.393741, 0.906002, -0.915744, -0.286598, -0.281556,
      0.370519, -0.873401, -0.316048;
  Eigen::Vector3s target;
  target << -0.942666, -2.07926, 0.864076;
  Eigen::Matrix3s Rtarget = math::eulerZXYToMatrix(target);
  std::cout << "Rtarget:\n" << Rtarget << std::endl;

  Eigen::Vector3s result = math::matrixToEulerZXY(R);
  Eigen::Vector3s expectedResult;
  expectedResult << 2.19999, -1.06214, -2.27702;

  EXPECT_TRUE(equals(result, expectedResult));

  expectedResult(0) -= M_PI;
  expectedResult(2) += M_PI;

  Eigen::Vector3s upperBound;
  upperBound << 1.5708, 1.5708, 1.5708;
  Eigen::Vector3s lowerBound;
  lowerBound << -1.5708, -2.0944, -1.5708;

  std::cout << "2PI: " << M_PI * 2 << std::endl;

  Eigen::Vector3s clamped
      = math::attemptToClampEulerAnglesToBounds(result, upperBound, lowerBound);
  std::cout << "clamped:\n" << clamped << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, CLAMP_WEIRDNESS)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  Eigen::VectorXs goldPos = Eigen::VectorXs(37);
  goldPos << 1.28514, 0.599806, 0.709412, 0, 0, 0, -0.363238, -0.504149,
      0.373617, 0.501634, 0.057355, -0.0944415, -0.13486, 1.04532, -0.434462,
      -0.298656, 1.19568, -0.0831732, 0.262512, -0.165874, 0.20991, 0.354704,
      0.826636, -0.942666, -2.07926, 0.864076, 1.32979, 0.0207049, -0.0381907,
      -0.231821, 1.32794, -0.652347, 1.21334, 1.6978, 0.883303, 0.915727,
      0.175323;

  Eigen::VectorXs ballJointsPos = Eigen::VectorXs(37);
  ballJointsPos << -0.138082, -2.61125, -3.74691, -3.17235e-05, -5.80963e-05,
      9.75787e-05, 3.05992, -3.25932, 3.23122, 0.500549, 0.0575203, -0.0957626,
      -0.132274, -0.239682, -0.495634, 1.08299, 1.19578, -0.083162, 0.262519,
      -0.165871, -1.39899, -4.77917, -1.91233, -1.64117, 1.48488, -1.4475,
      1.32985, 0.0208289, -0.0380994, -0.231936, 1.19135, -0.580035, 0.740875,
      1.69782, 0.883267, 0.915731, 0.175692;

  osimBallJoints->setPositions(ballJointsPos);
  osim->setPositions(osim->convertPositionsFromBallSpace(ballJointsPos));
  Eigen::VectorXs unclamped
      = osim->convertPositionsToBallSpace(osim->getPositions());

  Eigen::VectorXs recovered = osim->getPositions();

  // acromial_r is the trouble spot
  /*
  dynamics::EulerJoint* acromial_r
      = static_cast<dynamics::EulerJoint*>(osim->getJoint("acromial_r"));
  Eigen::Matrix3s R = math::expMapRot(
      ballJointsPos.segment<3>(acromial_r->getDof(0)->getIndexInSkeleton()));
  if (acromial_r->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
  {
    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "Axis order ZXY" << std::endl;
    std::cout << "Recovered axis: " << math::matrixToEulerZXY(R) << std::endl;
  }
  */

  if (!equals(unclamped, ballJointsPos, 1e-12))
  {
    std::cout << "Unclamped doesn't match original" << std::endl;
    std::cout << "Diff:" << std::endl << unclamped - ballJointsPos << std::endl;

    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
  }

  osim->clampPositionsToLimits();
  Eigen::VectorXs clamped
      = osim->convertPositionsToBallSpace(osim->getPositions());
  if (!equals(clamped, unclamped, 1e-12))
  {
    std::cout << "Clamped doesn't match original" << std::endl;
    Eigen::VectorXs diff = clamped - unclamped;
    std::cout << "Diff:" << std::endl << diff << std::endl;
    int cursor = 0;
    for (int i = 0; i < osimBallJoints->getNumJoints(); i++)
    {
      dynamics::Joint* joint = osimBallJoints->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs diffSubset = diff.segment(cursor, dofs);
      if (diffSubset.squaredNorm() > 1e-8)
      {
        std::cout << "Diff at joint " << joint->getName() << std::endl
                  << diffSubset << std::endl;
      }
      cursor += dofs;
    }

    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, CLAMP_WEIRDNESS_2)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  Eigen::VectorXs goldPos = Eigen::VectorXs::Zero(37);
  goldPos << -0.0476159, 0.42647, 1.56991, 0.555199, 1.00915, -0.977329,
      -0.399547, -0.00193299, 0.2908, 0.225648, 0.403207, 0, 0, -0.51773,
      0.0772674, -0.320767, 0.00382977, 0.165412, 0, 0, -0.599194, -0.13623,
      0.508717, 0.110365, -0.23571, 0.144373, 0.61493, 1.55246, 0, 0, 0.103503,
      -0.020086, -0.166672, 0.634983, 1.54352, 0, 0;

  /*
  Eigen::VectorXs standardBodyScales = Eigen::VectorXs::Zero(60);
  standardBodyScales << 1.07452, 1.04143, 1.46617, 1.18352, 0.969945, 1.18352,
      0.943467, 1.20593, 0.943467, 0.970271, 1, 0.943467, 0.970271, 1, 0.943467,
      0.970271, 1, 0.943467, 1.06301, 0.955219, 1.06301, 0.973821, 1.20515,
      0.973821, 0.971309, 1, 0.973821, 0.971309, 1, 0.973821, 0.971309, 1,
      0.973821, 1.31629, 1.04819, 1.2481, 1.47983, 1.03373, 1.47983, 0.809716,
      1.1894, 0.809716, 0.809716, 1.1894, 0.809716, 0.85, 0.85, 0.85, 1.41502,
      0.997312, 1.41502, 0.804157, 1.19371, 0.804157, 0.804157, 1.19371,
      0.804157, 0.85, 0.85, 0.85;
  */

  Eigen::VectorXs ballJointsPos = Eigen::VectorXs(37);
  ballJointsPos << 0.385497, 1.92537, 0.173129, 0.526872, 1.00141, -0.991841,
      -1.79229, -3.7335, -5.46329, 0.0399868, 0.15965, 0.145338, 0.509427,
      -0.0132649, 0.0595007, -0.436285, -0.0382189, 0.0938329, 0.5266,
      -0.122631, -0.0870851, 0.171899, -0.498203, -5.24101, -3.02705, 2.37908,
      6.95225, 3.34415, 0.469331, 0.0271157, 0.247161, 0.0453717, 0.0429319,
      0.850612, 2.96457, -0.337401, -0.117685;

  osimBallJoints->setPositions(ballJointsPos);
  osim->setPositions(osim->convertPositionsFromBallSpace(ballJointsPos));
  Eigen::VectorXs unclamped
      = osim->convertPositionsToBallSpace(osim->getPositions());

  Eigen::VectorXs recovered = osim->getPositions();

  // acromial_r is the trouble spot
  /*
  dynamics::EulerJoint* acromial_r
      = static_cast<dynamics::EulerJoint*>(osim->getJoint("acromial_r"));
  Eigen::Matrix3s R = math::expMapRot(
      ballJointsPos.segment<3>(acromial_r->getDof(0)->getIndexInSkeleton()));
  if (acromial_r->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
  {
    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "Axis order ZXY" << std::endl;
    std::cout << "Recovered axis: " << math::matrixToEulerZXY(R) << std::endl;
  }
  */

  if (!equals(unclamped, ballJointsPos, 1e-12))
  {
    std::cout << "Unclamped doesn't match original" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclamped.size(), 3);
    compare.col(0) = ballJointsPos;
    compare.col(1) = unclamped;
    compare.col(2) = unclamped - ballJointsPos;
    std::cout << "Original - Balls(Euler(Original)) - Diff:" << std::endl
              << compare << std::endl;

    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
  }

  Eigen::VectorXs unclampedEuler = osim->getPositions();
  osim->clampPositionsToLimits();
  Eigen::VectorXs clampedEuler = osim->getPositions();
  if (!equals(clampedEuler, unclampedEuler, 1e-12))
  {
    std::cout << "Clamped doesn't match original in Euler space!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclampedEuler.size(), 5);
    compare.col(0) = unclampedEuler;
    compare.col(1) = osim->getPositionLowerLimits();
    compare.col(2) = osim->getPositionUpperLimits();
    compare.col(3) = clampedEuler;
    compare.col(4) = clampedEuler - unclampedEuler;
    std::cout
        << "Unclamped Euler - Lower Limit - Upper Limit - Clamped Euler - Diff:"
        << std::endl
        << compare << std::endl;

    Eigen::VectorXs diff = clampedEuler - unclampedEuler;
    int cursor = 0;
    for (int i = 0; i < osim->getNumJoints(); i++)
    {
      dynamics::Joint* joint = osim->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs diffSubset = diff.segment(cursor, dofs);
      if (diffSubset.squaredNorm() > 1e-8)
      {
        std::cout << "Euler diff at joint " << joint->getName() << std::endl
                  << diffSubset << std::endl;
      }
      cursor += dofs;
    }
  }

  Eigen::VectorXs clamped
      = osim->convertPositionsToBallSpace(osim->getPositions());
  if (!equals(clamped, unclamped, 1e-12))
  {
    std::cout << "Clamped doesn't match original" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclamped.size(), 3);
    compare.col(0) = unclamped;
    compare.col(1) = clamped;
    compare.col(2) = clamped - unclamped;
    std::cout << "Unclamped - Clamped - Diff:" << std::endl
              << compare << std::endl;

    Eigen::VectorXs diff = clamped - unclamped;
    int cursor = 0;
    for (int i = 0; i < osimBallJoints->getNumJoints(); i++)
    {
      dynamics::Joint* joint = osimBallJoints->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs diffSubset = diff.segment(cursor, dofs);
      if (diffSubset.squaredNorm() > 1e-8)
      {
        std::cout << "Diff at joint " << joint->getName() << std::endl
                  << diffSubset << std::endl;
      }
      cursor += dofs;
    }

    /*
    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
    */
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, CLAMP_WEIRDNESS_3)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  Eigen::VectorXs goldPos = Eigen::VectorXs::Zero(37);
  goldPos << 0.0749049, -0.00491559, 1.32853, 0.467201, 0.98761, -0.990219,
      -0.521694, -0.330605, -0.0701683, 0.555501, 0.337712, 0, 0, -0.076734,
      0.0938467, -0.210891, 0.00541465, 0.298908, 0, 0, -0.143646, -0.00811661,
      0.335166, -0.0826361, -0.141152, -0.36731, 1.07779, 1.54974, 0, 0,
      -0.190997, -0.0918618, -0.131114, 1.33685, 1.54543, 0, 0;

  /*
  Eigen::VectorXs standardBodyScales = Eigen::VectorXs::Zero(60);
  standardBodyScales << 1.07452, 1.04143, 1.46617, 1.18352, 0.969945, 1.18352,
      0.943467, 1.20593, 0.943467, 0.970271, 1, 0.943467, 0.970271, 1, 0.943467,
      0.970271, 1, 0.943467, 1.06301, 0.955219, 1.06301, 0.973821, 1.20515,
      0.973821, 0.971309, 1, 0.973821, 0.971309, 1, 0.973821, 0.971309, 1,
      0.973821, 1.31629, 1.04819, 1.2481, 1.47983, 1.03373, 1.47983, 0.809716,
      1.1894, 0.809716, 0.809716, 1.1894, 0.809716, 0.85, 0.85, 0.85, 1.41502,
      0.997312, 1.41502, 0.804157, 1.19371, 0.804157, 0.804157, 1.19371,
      0.804157, 0.85, 0.85, 0.85;
  */

  Eigen::VectorXs ballJointsPos = Eigen::VectorXs(37);
  ballJointsPos << 0.0152175, 1.3869, 0.0771069, 0.471897, 0.996231, -0.997029,
      -0.391901, -0.0459652, -0.759009, 0.21759, 0.0853194, 0.315544, 0.369044,
      -0.066452, 0.168316, -0.194789, -0.0744426, 0.101366, 0.136293, -0.255278,
      -0.0411979, 0.241774, -0.17405, -0.283901, -0.202385, -0.0557436, 1.22807,
      2.9512, -0.381279, -0.0594017, 0.333266, 0.139513, -0.231853, 1.67011,
      3.27791, 0.508881, 0.422523;

  osimBallJoints->setPositions(ballJointsPos);
  osim->setPositions(osim->convertPositionsFromBallSpace(ballJointsPos));
  Eigen::VectorXs unclamped
      = osim->convertPositionsToBallSpace(osim->getPositions());

  Eigen::VectorXs recovered = osim->getPositions();

  // acromial_r is the trouble spot
  /*
  dynamics::EulerJoint* acromial_r
      = static_cast<dynamics::EulerJoint*>(osim->getJoint("acromial_r"));
  Eigen::Matrix3s R = math::expMapRot(
      ballJointsPos.segment<3>(acromial_r->getDof(0)->getIndexInSkeleton()));
  if (acromial_r->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
  {
    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "Axis order ZXY" << std::endl;
    std::cout << "Recovered axis: " << math::matrixToEulerZXY(R) << std::endl;
  }
  */

  if (!equals(unclamped, ballJointsPos, 1e-12))
  {
    std::cout << "Unclamped doesn't match original" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclamped.size(), 3);
    compare.col(0) = ballJointsPos;
    compare.col(1) = unclamped;
    compare.col(2) = unclamped - ballJointsPos;
    std::cout << "Original - Balls(Euler(Original)) - Diff:" << std::endl
              << compare << std::endl;

    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
  }

  Eigen::VectorXs unclampedEuler = osim->getPositions();
  osim->clampPositionsToLimits();
  Eigen::VectorXs clampedEuler = osim->getPositions();
  if (!equals(clampedEuler, unclampedEuler, 1e-12))
  {
    std::cout << "Clamped doesn't match original in Euler space!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclampedEuler.size(), 5);
    compare.col(0) = unclampedEuler;
    compare.col(1) = osim->getPositionLowerLimits();
    compare.col(2) = osim->getPositionUpperLimits();
    compare.col(3) = clampedEuler;
    compare.col(4) = clampedEuler - unclampedEuler;
    std::cout
        << "Unclamped Euler - Lower Limit - Upper Limit - Clamped Euler - Diff:"
        << std::endl
        << compare << std::endl;

    Eigen::VectorXs diff = clampedEuler - unclampedEuler;
    int cursor = 0;
    for (int i = 0; i < osim->getNumJoints(); i++)
    {
      dynamics::Joint* joint = osim->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs diffSubset = diff.segment(cursor, dofs);
      if (diffSubset.squaredNorm() > 1e-8)
      {
        std::cout << "Euler diff at joint " << joint->getName() << std::endl
                  << diffSubset << std::endl;
      }
      cursor += dofs;
    }
  }

  Eigen::VectorXs clamped
      = osim->convertPositionsToBallSpace(osim->getPositions());
  if (!equals(clamped, unclamped, 1e-12))
  {
    std::cout << "Clamped doesn't match original" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(unclamped.size(), 3);
    compare.col(0) = unclamped;
    compare.col(1) = clamped;
    compare.col(2) = clamped - unclamped;
    std::cout << "Unclamped - Clamped - Diff:" << std::endl
              << compare << std::endl;

    Eigen::VectorXs diff = clamped - unclamped;
    int cursor = 0;
    for (int i = 0; i < osimBallJoints->getNumJoints(); i++)
    {
      dynamics::Joint* joint = osimBallJoints->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs diffSubset = diff.segment(cursor, dofs);
      if (diffSubset.squaredNorm() > 1e-8)
      {
        std::cout << "Diff at joint " << joint->getName() << std::endl
                  << diffSubset << std::endl;
      }
      cursor += dofs;
    }

    /*
    for (int i = 0; i < osim->getNumBodyNodes(); i++)
    {
      Eigen::Matrix4s osimBallJointsBodyPos
          = osimBallJoints->getBodyNode(i)->getWorldTransform().matrix();
      Eigen::Matrix4s osimBodyPos
          = osim->getBodyNode(i)->getWorldTransform().matrix();
      if (!equals(osimBallJointsBodyPos, osimBodyPos, 1e-12))
      {
        std::cout << "Body node " << osim->getBodyNode(i)->getName()
                  << " doesn't match:" << std::endl;
        std::cout << "Ball joints:" << std::endl
                  << osimBallJointsBodyPos << std::endl;
        std::cout << "Euler joints:" << std::endl << osimBodyPos << std::endl;
        std::cout << "Diff:" << std::endl
                  << osimBodyPos - osimBallJointsBodyPos << std::endl;
      }
    }
    */
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, DERIVATIVES)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  // osim->autogroupSymmetricSuffixes();

  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      markers;
  markers["0"] = std::make_pair(
      osim->getBodyNode("radius_l"), Eigen::Vector3s::Random());
  markers["1"] = std::make_pair(
      osim->getBodyNode("radius_r"), Eigen::Vector3s::Random());
  markers["2"]
      = std::make_pair(osim->getBodyNode("tibia_l"), Eigen::Vector3s::Random());
  markers["3"]
      = std::make_pair(osim->getBodyNode("tibia_r"), Eigen::Vector3s::Random());
  markers["4"]
      = std::make_pair(osim->getBodyNode("ulna_l"), Eigen::Vector3s::Random());
  markers["5"]
      = std::make_pair(osim->getBodyNode("ulna_r"), Eigen::Vector3s::Random());

  MarkerFitter fitter(osim, markers);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(2);
  fitter.addZeroConstraint("trivial", [&](MarkerFitterState* state) {
    (void)state;
    return 0.0;
  });

  srand(42);

  std::map<std::string, Eigen::Vector3s> observedMarkers;
  observedMarkers["0"] = Eigen::Vector3s::Random();
  observedMarkers["1"] = Eigen::Vector3s::Random();
  // Skip 2
  observedMarkers["3"] = Eigen::Vector3s::Random();
  observedMarkers["4"] = Eigen::Vector3s::Random();
  observedMarkers["5"] = Eigen::Vector3s::Random();

  /*
  Eigen::VectorXs pose = Eigen::VectorXs(37);
  pose << 1.28514, 0.599806, 0.709412, -3.31116, 1.96564, 2.61346, -0.363238,
      -0.504149, 0.373617, 0.501634, 0.057355, -0.0944415, -0.13486, 1.04532,
      -0.434462, -0.298656, 1.19568, -0.0831732, 0.262512, -0.165874, 0.20991,
      0.354704, 0.826636, -0.942666, -2.07926, 0.864076, 1.32979, 0.0207049,
      -0.0381907, -0.231821, 1.32794, -0.652347, 1.21334, 1.6978, 0.883303,
      0.915727, 0.175323;
  debugIKInitializationToGUI(osim, pose, 0.0);
  */

  std::vector<dynamics::Joint*> joints;
  joints.push_back(osim->getJoint("walker_knee_l"));
  joints.push_back(osim->getJoint("walker_knee_r"));

  EXPECT_TRUE(
      testBilevelFitProblemGradients(fitter, 3, 0.02, osim, joints, markers));

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));

  // EXPECT_TRUE(testSolveBilevelFitProblem(osim, 20, 0.01, 0.001, 0.1));
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, DERIVATIVES_BALL_JOINTS)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);
  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      markers;
  markers["0"] = std::make_pair(
      osimBallJoints->getBodyNode("radius_l"), Eigen::Vector3s::Random());
  markers["1"] = std::make_pair(
      osimBallJoints->getBodyNode("radius_r"), Eigen::Vector3s::Random());
  markers["2"] = std::make_pair(
      osimBallJoints->getBodyNode("tibia_l"), Eigen::Vector3s::Random());
  markers["3"] = std::make_pair(
      osimBallJoints->getBodyNode("tibia_r"), Eigen::Vector3s::Random());
  markers["4"] = std::make_pair(
      osimBallJoints->getBodyNode("ulna_l"), Eigen::Vector3s::Random());
  markers["5"] = std::make_pair(
      osimBallJoints->getBodyNode("ulna_r"), Eigen::Vector3s::Random());

  MarkerFitter fitter(osimBallJoints, markers);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(2);
  fitter.addZeroConstraint("trivial", [&](MarkerFitterState* state) {
    (void)state;
    return 0.0;
  });

  srand(42);

  std::map<std::string, Eigen::Vector3s> observedMarkers;
  observedMarkers["0"] = Eigen::Vector3s::Random();
  observedMarkers["1"] = Eigen::Vector3s::Random();
  // Skip 2
  observedMarkers["3"] = Eigen::Vector3s::Random();
  observedMarkers["4"] = Eigen::Vector3s::Random();
  observedMarkers["5"] = Eigen::Vector3s::Random();

  std::vector<dynamics::Joint*> joints;
  joints.push_back(osimBallJoints->getJoint("walker_knee_l"));
  joints.push_back(osimBallJoints->getJoint("walker_knee_r"));

  EXPECT_TRUE(
      testFitterGradients(fitter, osimBallJoints, markers, observedMarkers));

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, osimBallJoints, joints, markers));
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, INITIALIZATION)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  // Get the gold data scales in `config`
  OpenSimFile moddedBase = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(moddedBase.markersMap);
  standard.markersMap = convertedMarkers;

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;
  (void)poses;

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  for (int i = 0; i < 300; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }

  MarkerInitialization init = fitter.getInitialization(subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  /*
  // Target markers
  debugTrajectoryAndMarkersToGUI(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps);
  */
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, SPHERE_FIT_GRAD)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  // Get the gold data scales in `config`
  OpenSimFile moddedBase = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(moddedBase.markersMap);
  standard.markersMap = convertedMarkers;

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;
  (void)poses;

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;

  MarkerInitialization init
      = fitter.getInitialization(subsetTimesteps, InitialMarkerFitParams());

  standard.skeleton->setGroupScales(init.groupScales);

  Eigen::MatrixXs out;
  SphereFitJointCenterProblem sphereProblem(
      &fitter,
      subsetTimesteps,
      init.poses,
      standard.skeleton->getJoint("walker_knee_r"),
      out);

  Eigen::VectorXs analytical = sphereProblem.getGradient();
  Eigen::VectorXs bruteForce = sphereProblem.finiteDifferenceGradient();

  if (!equals(analytical, bruteForce, 1e-8))
  {
    std::cout << "Error on SphereFitJointCenterProblem grad " << std::endl
              << "Analytical:" << std::endl
              << analytical << std::endl
              << "FD:" << std::endl
              << bruteForce << std::endl
              << "Diff:" << std::endl
              << analytical - bruteForce << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-8));
  }

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);
  fitter.findJointCenters(init, subsetTimesteps);
}
#endif

// #ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  // Get the gold data scales in `config`
  OpenSimFile moddedBase = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(moddedBase.markersMap);
  standard.markersMap = convertedMarkers;

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs goldPoses = mot.poses;
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      markerTrajectories.markerTimesteps);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;

  MarkerInitialization init
      = fitter.getInitialization(subsetTimesteps, InitialMarkerFitParams());

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);
  fitter.findJointCenters(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      InitialMarkerFitParams()
          .setJointCenters(init.joints, init.jointCenters)
          .setInitPoses(init.poses));

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  /*
  ////////////////////////////////////////////////////////////////////////
  std::shared_ptr<BilevelFitResult> tmpResult
      = std::make_shared<BilevelFitResult>();

  BilevelFitProblem problem(&fitter, subsetTimesteps, reinit, 10, tmpResult);
  std::cout << "Loss at initialization from getLoss(): "
            << problem.getLoss(problem.getInitialization()) << std::endl;

  MarkerFitterState state(
      problem.getInitialization(),
      problem.getMarkerMapObservations(),
      reinit.joints,
      problem.getJointCenters(),
      &fitter);

  std::vector<int> indices = problem.getSampleIndices();
  s_t totalLoss = 0.0;
  for (int i = 0; i < indices.size(); i++)
  {
    int index = indices[i];
    standard.skeleton->setPositions(reinit.poses.col(index));
    standard.skeleton->setGroupScales(reinit.groupScales);

    std::map<std::string, Eigen::Vector3s> goldMarkers = subsetTimesteps[index];
    std::map<std::string, Eigen::Vector3s> ourMarkers
        = standard.skeleton->getMarkerMapWorldPositions(
            reinit.updatedMarkerMap);

    s_t markerLoss = 0.0;
    for (auto pair : goldMarkers)
    {
      markerLoss += (ourMarkers[pair.first] - pair.second).squaredNorm();
    }

    Eigen::VectorXs goldJointCenter = reinit.jointCenters.col(index);
    Eigen::VectorXs ourJointCenter
        = standard.skeleton->getJointWorldPositions(reinit.joints);

    s_t jointLoss = (goldJointCenter - ourJointCenter).squaredNorm();

    std::cout << "Timestep " << i << ": (marker=" << markerLoss
              << ",joint=" << jointLoss << ") = " << markerLoss + jointLoss
              << std::endl;
    totalLoss += markerLoss + jointLoss;
  }
  std::cout << "Manually calculated total loss: " << totalLoss << std::endl;

  return;
  ////////////////////////////////////////////////////////////////////////
  */

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.getInitialization(
      subsetTimesteps,
      InitialMarkerFitParams()
          .setJointCenters(reinit.joints, reinit.jointCenters)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Michael's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  debugTrajectoryAndMarkersToGUI(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      finalKinematicInit.jointCenters);
}
// #endif

// #ifdef FULL_EVAL
#ifdef ALL_TESTS
TEST(MarkerFitter, EVAL_PERFORMANCE)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  // Get the gold data scales in `config`
  OpenSimFile moddedBase = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  dynamics::MarkerMap convertedMarkers
      = standard.skeleton->convertMarkerMap(moddedBase.markersMap);
  standard.markersMap = convertedMarkers;
  OpenSimScaleAndMarkerOffsets config
      = OpenSimParser::getScaleAndMarkerOffsets(standard, scaled);
  EXPECT_TRUE(config.success);

  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");
  Eigen::MatrixXs poses = mot.poses;
  (void)poses;

  // Check our marker maps

  std::vector<
      std::pair<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>>
      moddedMarkerOffsets;
  for (auto pair : standard.markersMap)
  {
    moddedMarkerOffsets.push_back(pair);
  }
  for (int i = 0; i < moddedMarkerOffsets.size(); i++)
  {
    for (int j = 0; j < moddedMarkerOffsets.size(); j++)
    {
      if (i == j)
        continue;
      // Just don't have duplicate markers
      if (moddedMarkerOffsets[i].second.first
              == moddedMarkerOffsets[j].second.first
          && moddedMarkerOffsets[i].second.second
                 == moddedMarkerOffsets[j].second.second)
      {
        std::cout << "Found duplicate markers, " << i << " and " << j << ": "
                  << moddedMarkerOffsets[i].first << " and "
                  << moddedMarkerOffsets[j].first << " on "
                  << moddedMarkerOffsets[i].second.first->getName()
                  << std::endl;
      }
    }
  }

  // Check that timestamps match up
  if (mot.timestamps.size() != markerTrajectories.timestamps.size())
  {
    std::cout << "Got a different number of timestamps. Mot: "
              << mot.timestamps.size()
              << " != Trc: " << markerTrajectories.timestamps.size()
              << std::endl;
    EXPECT_EQ(mot.timestamps.size(), markerTrajectories.timestamps.size());
    std::cout << "First 10 timesteps:" << std::endl;
    for (int k = 0; k < 10; k++)
    {
      std::cout << k << ": Mot=" << mot.timestamps[k]
                << " != Trc=" << markerTrajectories.timestamps[k] << std::endl;
    }
  }
  else
  {
    for (int i = 0; i < mot.timestamps.size(); i++)
    {
      if (abs(mot.timestamps[i] - markerTrajectories.timestamps[i]) > 1e-10)
      {
        std::cout << "Different timestamps at step " << i
                  << ": Mot: " << mot.timestamps[i]
                  << " != Trc: " << markerTrajectories.timestamps[i]
                  << std::endl;
        EXPECT_NEAR(mot.timestamps[i], markerTrajectories.timestamps[i], 1e-10);
        break;
      }
    }
  }

  /*
  scaled.skeleton->setPositions(poses.col(0));
  std::cout << scaled.skeleton->getJoint(0)->getRelativeTransform().matrix()
            << std::endl;

  std::shared_ptr<dynamics::Skeleton> clone = scaled.skeleton->clone();
  clone->setPositions(poses.col(0));
  std::cout << clone->getJoint(0)->getRelativeTransform().matrix() << std::endl;

  // Target markers
  debugTrajectoryAndMarkersToGUI(
      scaled.skeleton,
      scaled.markersMap,
      poses,
      markerTrajectories.markerTimesteps);
  */

  int timestep = 1264; // 103, 1851

  std::map<std::string, Eigen::Vector3s> goldMarkers
      = markerTrajectories.markerTimesteps[timestep];
  Eigen::VectorXs goldPose = poses.col(timestep);

  /*
  // Try to convert the goldPose to the standard skeleton

  Eigen::VectorXs standardGoldPose
      = Eigen::VectorXs::Zero(standard.skeleton->getNumDofs());
  for (int i = 0; i < standard.skeleton->getNumDofs(); i++)
  {
    standardGoldPose(i) = goldPose(
        scaled.skeleton->getDof(standard.skeleton->getDof(i)->getName())
            ->getIndexInSkeleton());
  }
  std::cout << "Eigen::VectorXs standardGoldPose = Eigen::VectorXs::Zero("
            << standardGoldPose.size() << ");" << std::endl;
  std::cout << "standardGoldPose << ";
  for (int i = 0; i < standardGoldPose.size(); i++)
  {
    if (i > 0)
    {
      std::cout << ", ";
    }
    std::cout << standardGoldPose(i);
  }
  std::cout << ";" << std::endl;

  Eigen::VectorXs standardBodyScales = config.bodyScales;
  std::cout << "Eigen::VectorXs standardBodyScales = Eigen::VectorXs::Zero("
            << standardBodyScales.size() << ");" << std::endl;
  std::cout << "standardBodyScales << ";
  for (int i = 0; i < standardBodyScales.size(); i++)
  {
    if (i > 0)
    {
      std::cout << ", ";
    }
    std::cout << standardBodyScales(i);
  }
  std::cout << ";" << std::endl;
  */

  // Try to fit the skeleton to

  /*
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  Eigen::VectorXs targetPoses = Eigen::VectorXs::Zero(goldMarkers.size() * 3);
  for (auto pair : goldMarkers)
  {
    std::cout << "Marker: " << pair.first << std::endl;
    targetPoses.segment<3>(markers.size() * 3) = pair.second;
    markers.push_back(standard.markersMap[pair.first]);
  }
  Eigen::VectorXs markerWeights = Eigen::VectorXs::Ones(markers.size());
  debugFitToGUI(
      standard.skeleton, markers, targetPoses, scaled.skeleton, goldPose);
  */

  // Get a random subset of the data

  srand(25);
  /*
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations
      = MarkerFitter::pickSubset(markerTrajectories.markerTimesteps, 40);
  */

  /*
  std::vector<unsigned int> indices(markerTrajectories.markerTimesteps.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  for (int i = 0; i < 2; i++)
  {
    std::cout << "Shuffled " << i << "->" << indices[i] << std::endl;
    markerObservations.push_back(
        markerTrajectories.markerTimesteps[indices[i]]);
    for (auto pair : markerTrajectories.markerTimesteps[indices[i]])
    {
      std::cout << pair.first << ": " << pair.second << std::endl;
    }
  }
  */

  std::cout << "Original skel pos: " << standard.skeleton->getPositions()
            << std::endl;

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  markerObservations.push_back(markerTrajectories.markerTimesteps[0]);
  for (auto pair : markerObservations[0])
  {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }

  std::cout << "Marker map:" << std::endl;
  for (auto pair : standard.markersMap)
  {
    std::cout << pair.first << ": (" << pair.second.first->getName() << ", "
              << pair.second.second << ")" << std::endl;
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::shared_ptr<BilevelFitResult> result
      = fitter.optimize(markerObservations);
  standard.skeleton->setGroupScales(result->groupScales);
  Eigen::VectorXs bodyScales = standard.skeleton->getBodyScales();

  std::cout << "Result scales: " << bodyScales << std::endl;

  Eigen::VectorXs groupScaleError = bodyScales - config.bodyScales;
  Eigen::MatrixXs groupScaleCols = Eigen::MatrixXs(groupScaleError.size(), 4);
  groupScaleCols.col(0) = config.bodyScales;
  groupScaleCols.col(1) = bodyScales;
  groupScaleCols.col(2) = groupScaleError;
  groupScaleCols.col(3) = groupScaleError.cwiseQuotient(config.bodyScales);
  std::cout << "gold scales - result scales - error - error %" << std::endl
            << groupScaleCols << std::endl;
}
#endif
// #endif