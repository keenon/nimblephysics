#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>

#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

#define FUNCTIONAL_TESTS

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
    Eigen::MatrixXs comp = Eigen::MatrixXs::Zero(gradWrtJoints.size(), 3);
    comp.col(0) = gradWrtJoints;
    comp.col(1) = gradWrtJoints_fd;
    comp.col(2) = gradWrtJoints - gradWrtJoints_fd;
    std::cout << "Error on grad wrt joints" << std::endl
              << "Analytical - FD - Diff:" << std::endl
              << comp << std::endl;
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
    Eigen::MatrixXs comp = Eigen::MatrixXs::Zero(gradWrtScales.size(), 3);
    comp.col(0) = gradWrtScales;
    comp.col(1) = gradWrtScales_fd;
    comp.col(2) = gradWrtScales - gradWrtScales_fd;
    std::cout << "Error on grad wrt scales" << std::endl
              << "Analytical - FD - Diff:" << std::endl
              << comp << std::endl;
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
    bool applyInnerProblemGradientConstraints,
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

  BilevelFitProblem problem(
      &fitter,
      observations,
      init,
      numPoses,
      applyInnerProblemGradientConstraints,
      tmpResult);

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

  std::vector<dynamics::Joint*> joints;
  joints.push_back(skel->getJoint(3));
  joints.push_back(skel->getJoint(7));

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
    goldPoses.push_back(goldPose);

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
  init.groupScales = goldGroupScales
                     + Eigen::VectorXs::Random(goldGroupScales.size()) * 0.01;

  fitter.setCheckDerivatives(true);
  BilevelFitProblem problem(
      &fitter, observations, init, numPoses, true, tmpResult);

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
    // return false;
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
    initialPos.segment(0, skelBallJoints->getNumDofs())
        = skelBallJoints->getPositions();
    initialPos.segment(
        skelBallJoints->getNumDofs(), skelBallJoints->getGroupScaleDim())
        = skelBallJoints->getGroupScales();
    /*
        Eigen::VectorXs initialGuess = Eigen::VectorXs(97);
    initialPos << -0.329172, -0.238389, -1.51384, -1.95121, 1.01627, -2.14055,
        0.479686, -0.390676, 0.825582, 1.20487, 0.0764906, 0.117637, -0.28067,
        -0.692632, 0.443975, 0.388073, 1.78741, 0.288679, -0.097154, 0.184243,
        0.268635, 1.0733, 1.59988, -0.737188, -1.03813, -0.236178, 0.823967,
        1.32797, 0.495846, 0.406708, -0.442464, 0.698941, -0.317397, 2.18822,
        0.0254939, 1.11635, 0.588954, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    */

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
            /*
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
          */

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
          server.renderSkeleton(
              skelBallJoints, "ball", Eigen::Vector4s(1.0, 0, 0, 1.0));

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
                Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
          }

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

    std::cout << "********** Joints World **********" << std::endl;

    for (int i = 0; i < skel->getNumJoints(); i++)
    {
      std::string jointName = skel->getJoint(i)->getName();

      Eigen::Matrix4s skelT
          = skel->getJoint(i)->getRelativeTransform().matrix();
      Eigen::Matrix4s ballT
          = skelBallJoints->getJoint(i)->getRelativeTransform().matrix();
      if ((skelT - ballT).norm() > 1e-8)
      {
        std::cout << "Skel and Ball joints transforms don't match for joint \""
                  << jointName << "\":" << std::endl;
        std::cout << "Skel:" << std::endl << skelT << std::endl;
        std::cout << "Ball:" << std::endl << ballT << std::endl;

        auto* joint = skel->getJoint(i);
        auto* ballJoint = skelBallJoints->getJoint(i);
        Eigen::VectorXs finalJointPos = joint->getPositions();
        Eigen::VectorXs ballJointPos = ballJoint->getPositions();

        Eigen::MatrixXs comp = Eigen::MatrixXs::Zero(finalJointPos.size(), 3);
        comp.col(0) = finalJointPos;
        comp.col(1) = ballJointPos;
        comp.col(2) = finalJointPos - ballJointPos;
        std::cout << "Skel - Ball - Diff" << std::endl << comp << std::endl;
      }
    }

    // Check the positions of the joints
    std::cout << "********** Joints **********" << std::endl;

    Eigen::VectorXs finalJoints = skel->getPositions();
    int cursor = 0;
    for (int i = 0; i < skel->getNumJoints(); i++)
    {
      auto* joint = skel->getJoint(i);
      auto* ballJoint = skelBallJoints->getJoint(i);
      int dofs = joint->getNumDofs();
      Eigen::VectorXs finalJointPos = joint->getPositions();
      Eigen::VectorXs ballJointPos = ballJoint->getPositions();
      Eigen::VectorXs goldJointPos = goldPose.segment(cursor, dofs);
      cursor += dofs;

      if ((finalJointPos - goldJointPos).norm() > 1e-2)
      {
        std::cout << "Joint " << joint->getName() << " did not recover gold!"
                  << std::endl;
        Eigen::MatrixXs comp = Eigen::MatrixXs::Zero(finalJointPos.size(), 4);
        comp.col(0) = finalJointPos;
        comp.col(1) = ballJointPos;
        comp.col(2) = goldJointPos;
        comp.col(3) = finalJointPos - goldJointPos;
        std::cout << "IK - Ball - Gold - Diff" << std::endl
                  << comp << std::endl;
      }
    }

    std::cout << "********** Adjusted Marker **********" << std::endl;

    Eigen::VectorXs finalMarkers
        = skelBallJoints->getMarkerWorldPositions(adjustedMarkers);
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

    std::cout << "********** Adjusted Marker Skel **********" << std::endl;

    Eigen::VectorXs finalMarkersSkel
        = skel->getMarkerWorldPositions(adjustedMarkersSkel);
    Eigen::VectorXs diffSkel = markerWorldPoses - finalMarkersSkel;
    for (int i = 0; i < adjustedMarkersSkel.size(); i++)
    {
      auto pair = adjustedMarkersSkel[i];
      Eigen::Vector3s markerDiff = diffSkel.segment<3>(i * 3);
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
                Eigen::Vector4s(1, 0, 0, 1));
          }

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

void drawTimeSeriesFigureToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    std::shared_ptr<dynamics::Skeleton> skeleton,
    MarkerInitialization init,
    const std::vector<std::map<std::string, Eigen::Vector3s>>&
        markerObservations)
{
  int setSize = 30;

  int cursor = 0;
  while (cursor < 250)
  {
    skeleton->setPositions(init.poses.col(cursor));
    server->renderSkeleton(skeleton, "skel_" + std::to_string(cursor));

    for (auto pair : markerObservations[cursor])
    {
      server->createSphere(
          "marker_" + pair.first + "_" + std::to_string(cursor),
          0.01,
          pair.second,
          Eigen::Vector4s(237.0 / 255, 118.0 / 255, 114.0 / 255, 1));
    }

    cursor += setSize;
  }
}

#ifdef FUNCTIONAL_TESTS
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

  srand(42);
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

#ifdef FUNCTIONAL_TESTS
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

  srand(42);

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

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, true, osim, joints, markers));

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osim, joints, markers));

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));

  // EXPECT_TRUE(testSolveBilevelFitProblem(osim, 20, 0.01, 0.001, 0.1));
}
#endif

#ifdef FUNCTIONAL_TESTS
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

  srand(42);
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
      fitter, 3, 0.02, true, osimBallJoints, joints, markers));
  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osimBallJoints, joints, markers));
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, DERIVATIVES_ARNOLD)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/LaiArnoldSubject6/"
            "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim")
            .skeleton;
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  // osim->autogroupSymmetricSuffixes();

  srand(42);
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

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, true, osim, joints, markers));
  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osim, joints, markers));

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));

  // EXPECT_TRUE(testSolveBilevelFitProblem(osim, 20, 0.01, 0.001, 0.1));
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, DERIVATIVES_ARNOLD_BALL_JOINTS)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/LaiArnoldSubject6/"
            "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim")
            .skeleton;
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);
  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  srand(42);
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

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

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
      fitter, 3, 0.02, true, osimBallJoints, joints, markers));
  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osimBallJoints, joints, markers));
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, AUTOGROUP_COMPLEX_KNEE)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/ComplexKnee/"
            "gait2392_frontHingeKnee_dem.osim")
            .skeleton;
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->autogroupSymmetricSuffixes();
  EXPECT_EQ(
      osim->getScaleGroupIndex(osim->getBodyNode("femur_r")),
      osim->getScaleGroupIndex(osim->getBodyNode("femur_l")));
  EXPECT_EQ(
      osim->getScaleGroupIndex(osim->getBodyNode("tibia_r")),
      osim->getScaleGroupIndex(osim->getBodyNode("tibia_l")));

  EXPECT_LT(osim->getGroupScales().size(), osim->getBodyScales().size());
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, DERIVATIVES_COMPLEX_KNEE)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/ComplexKnee/"
            "gait2392_frontHingeKnee_dem.osim")
            .skeleton;
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  // osim->autogroupSymmetricSuffixes();

  srand(42);
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      markers;
  markers["0"]
      = std::make_pair(osim->getBodyNode("toes_l"), Eigen::Vector3s::Random());
  markers["1"]
      = std::make_pair(osim->getBodyNode("toes_r"), Eigen::Vector3s::Random());
  markers["2"]
      = std::make_pair(osim->getBodyNode("tibia_l"), Eigen::Vector3s::Random());
  markers["3"]
      = std::make_pair(osim->getBodyNode("tibia_r"), Eigen::Vector3s::Random());
  markers["4"] = std::make_pair(
      osim->getBodyNode("sagittal_knee_body_l"), Eigen::Vector3s::Random());
  markers["5"] = std::make_pair(
      osim->getBodyNode("sagittal_knee_body_r"), Eigen::Vector3s::Random());

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
  joints.push_back(osim->getJoint("knee_l"));
  joints.push_back(osim->getJoint("knee_r"));

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, true, osim, joints, markers));
  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osim, joints, markers));

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));

  // EXPECT_TRUE(testSolveBilevelFitProblem(osim, 20, 0.01, 0.001, 0.1));
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, DERIVATIVES_COMPLEX_KNEE_BALL_JOINTS)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/ComplexKnee/"
            "gait2392_frontHingeKnee_dem.osim")
            .skeleton;
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);
  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));
  std::shared_ptr<dynamics::Skeleton> osimBallJoints
      = osim->convertSkeletonToBallJoints();

  srand(42);
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      markers;
  markers["0"] = std::make_pair(
      osimBallJoints->getBodyNode("toes_l"), Eigen::Vector3s::Random());
  markers["1"] = std::make_pair(
      osimBallJoints->getBodyNode("toes_r"), Eigen::Vector3s::Random());
  markers["2"] = std::make_pair(
      osimBallJoints->getBodyNode("tibia_l"), Eigen::Vector3s::Random());
  markers["3"] = std::make_pair(
      osimBallJoints->getBodyNode("tibia_r"), Eigen::Vector3s::Random());
  markers["4"] = std::make_pair(
      osimBallJoints->getBodyNode("sagittal_knee_body_l"),
      Eigen::Vector3s::Random());
  markers["5"] = std::make_pair(
      osimBallJoints->getBodyNode("sagittal_knee_body_r"),
      Eigen::Vector3s::Random());

  MarkerFitter fitter(osimBallJoints, markers);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(2);
  fitter.addZeroConstraint("trivial", [&](MarkerFitterState* state) {
    (void)state;
    return 0.0;
  });

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  srand(42);

  std::map<std::string, Eigen::Vector3s> observedMarkers;
  observedMarkers["0"] = Eigen::Vector3s::Random();
  observedMarkers["1"] = Eigen::Vector3s::Random();
  // Skip 2
  observedMarkers["3"] = Eigen::Vector3s::Random();
  observedMarkers["4"] = Eigen::Vector3s::Random();
  observedMarkers["5"] = Eigen::Vector3s::Random();

  std::vector<dynamics::Joint*> joints;
  joints.push_back(osimBallJoints->getJoint("knee_l"));
  joints.push_back(osimBallJoints->getJoint("knee_r"));

  EXPECT_TRUE(
      testFitterGradients(fitter, osimBallJoints, markers, observedMarkers));

  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, true, osimBallJoints, joints, markers));
  EXPECT_TRUE(testBilevelFitProblemGradients(
      fitter, 3, 0.02, false, osimBallJoints, joints, markers));
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

#ifdef FUNCTIONAL_TESTS
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
  std::vector<bool> newClip;
  for (int i = 0; i < markerTrajectories.markerTimesteps.size(); i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  standard.skeleton->setGroupScales(init.groupScales);

  Eigen::MatrixXs out;
  SphereFitJointCenterProblem sphereProblem(
      &fitter,
      subsetTimesteps,
      init.poses,
      standard.skeleton->getJoint("walker_knee_r"),
      newClip,
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
    EXPECT_TRUE(equals(analytical, bruteForce, 5e-8));
  }

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);
  fitter.findJointCenters(init, newClip, subsetTimesteps);
}
#endif

#ifdef FUNCTIONAL_TESTS
TEST(MarkerFitter, AXIS_FIT_GRAD)
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
  fitter.setInitialIKMaxRestarts(1);
  fitter.setIterationLimit(10);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
    newClip.push_back(false);
  }
  // subsetTimesteps = markerTrajectories.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  standard.skeleton->setGroupScales(init.groupScales);

  srand(42);
  Eigen::MatrixXs out;
  CylinderFitJointAxisProblem cylinderProblem(
      &fitter,
      subsetTimesteps,
      init.poses,
      standard.skeleton->getJoint("walker_knee_r"),
      Eigen::MatrixXs::Random(3, init.poses.cols()),
      newClip,
      out);

  Eigen::VectorXs analytical = cylinderProblem.getGradient();
  Eigen::VectorXs bruteForce = cylinderProblem.finiteDifferenceGradient();

  if (!equals(analytical, bruteForce, 5e-8) || true)
  {
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(analytical.size(), 4);
    compare.col(0) = cylinderProblem.flatten();
    compare.col(1) = analytical;
    compare.col(2) = bruteForce;
    compare.col(3) = analytical - bruteForce;
    std::cout << "Error on CylinderFitJointAxisProblem grad " << std::endl
              << "X - Analytical - FD - Diff:" << std::endl
              << compare << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 5e-8));
  }

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);
  // fitter.findAllJointAxis(init, subsetTimesteps);
}
#endif

#ifdef ALL_TESTS
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

  saveTrajectoryAndMarkersToGUI(
      "./michael.json",
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      finalKinematicInit.jointCenters);

  debugTrajectoryAndMarkersToGUI(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      finalKinematicInit.jointCenters);
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_LAI_ARNOLD)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/LaiArnoldSubject6/"
      "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  /*
  // TODO: <remove>
  std::shared_ptr<dynamics::Skeleton> ballJoints
      = standard.skeleton->convertSkeletonToBallJoints();
  std::cout
      << "Original knee position upper limits: "
      << standard.skeleton->getJoint("walker_knee_r")->getPositionUpperLimits()
      << std::endl;
  std::cout
      << "Original knee position lower limits: "
      << standard.skeleton->getJoint("walker_knee_r")->getPositionLowerLimits()
      << std::endl;
  Eigen::VectorXs pose = standard.skeleton->convertPositionsToBallSpace(
      standard.skeleton->getRandomPose());
  std::cout << "Knee position upper limits: "
            << ballJoints->getJoint("walker_knee_r")->getPositionUpperLimits()
            << std::endl;
  std::cout << "Knee position lower limits: "
            << ballJoints->getJoint("walker_knee_r")->getPositionLowerLimits()
            << std::endl;
  debugIKInitializationToGUI(ballJoints, pose, 0.05);
  // TODO: </remove>
  */

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/LaiArnoldSubject6/walking1.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/LaiArnoldSubject6/"
      "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/LaiArnoldSubject6/walking1.mot");
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

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

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

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

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

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

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

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

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

  saveTrajectoryAndMarkersToGUI(
      "./laiArnold.json",
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      finalKinematicInit.jointCenters);

  // Target markers
  debugTrajectoryAndMarkersToGUI(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      finalKinematicInit.jointCenters);
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_LAI_ARNOLD_2)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/LaiArnoldSubject5/"
      "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories
      = OpenSimParser::loadTRC("dart://sample/osim/LaiArnoldSubject5/DJ1.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/LaiArnoldSubject5/"
      "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/LaiArnoldSubject5/DJ1.mot");

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    subMarkerTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
    newClip.push_back(false);
  }
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      subMarkerTimesteps,
      anthropometrics);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Experts's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_SPRINTER)
{
  OpenSimFile standard
      = OpenSimParser::parseOsim("dart://sample/osim/Sprinter/sprinter.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories
      = OpenSimParser::loadTRC("dart://sample/osim/Sprinter/run0500cms.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Sprinter/sprinter_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/Sprinter/run0500cms.mot");

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    subMarkerTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  IKErrorReport goldReport(
      scaled.skeleton, scaled.markersMap, goldPoses, subMarkerTimesteps);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < markerTrajectories.markerTimesteps.size(); i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Experts's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  OpenSimParser::saveMot(
      standard.skeleton,
      "./auto_run0500cms.mot",
      markerTrajectories.timestamps,
      finalKinematicInit.poses);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_SPRINTER_2)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/sprinter_v2/unscaled_generic_w_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories
      = OpenSimParser::loadTRC("dart://sample/osim/sprinter_v2/markers.trc");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = markerTrajectories.markerTimesteps;

  std::vector<bool> newClip;
  for (int i = 0; i < markerTrajectories.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_SPRINTER_C3D)
{
  OpenSimFile standard
      = OpenSimParser::parseOsim("dart://sample/osim/Sprinter/sprinter.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D("dart://sample/c3d/JA1Gait35.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  std::vector<bool> newClip;
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = c3d.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(200);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  OpenSimParser::saveMot(
      standard.skeleton,
      "./autoIK.mot",
      c3d.timestamps,
      finalKinematicInit.poses);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  drawTimeSeriesFigureToGUI(
      server, standard.skeleton, finalKinematicInit, subsetTimesteps);
  /*
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps, &c3d);
*/
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, CLONE_KNEE_JOINT_LIMITS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim");
  auto skeleton = standard.skeleton->cloneSkeleton();
  if (skeleton->getPositionUpperLimits()
      != standard.skeleton->getPositionUpperLimits())
  {
    Eigen::MatrixXs comp = Eigen::MatrixXs(skeleton->getNumDofs(), 3);
    comp.col(0) = standard.skeleton->getPositionUpperLimits();
    comp.col(1) = skeleton->getPositionUpperLimits();
    comp.col(2) = standard.skeleton->getPositionUpperLimits()
                  - skeleton->getPositionUpperLimits();
    int cursor = 0;
    for (int i = 0; i < standard.skeleton->getNumJoints(); i++)
    {
      auto* joint = standard.skeleton->getJoint(i);
      int size = joint->getNumDofs();
      if (comp.col(2).segment(cursor, size).norm() > 0)
      {
        std::cout << "Joint " << i << " (of type " << joint->getType()
                  << ") upper limits do not match!" << std::endl;
        std::cout << "Original - Cloned - Diff" << std::endl
                  << comp.block(cursor, 0, size, 3) << std::endl;
        EXPECT_EQ(comp.col(2).segment(cursor, size).norm(), 0);
      }
      cursor += size;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, WHICH_EULER_ANGLES_ARE_NEGATIVE_GIMBAL_LOCK)
{
  Eigen::Vector3s locked = Eigen::Vector3s(0.1, M_PI / 2, -0.1);
  Eigen::Vector3s lockedPos = Eigen::Vector3s(0.1, M_PI / 2, 0.1);
  std::cout << "XYZ R-: " << std::endl
            << math::eulerXYZToMatrix(locked) << std::endl;
  std::cout << "XYZ R+: " << std::endl
            << math::eulerXYZToMatrix(lockedPos) << std::endl;
  // Conlude XYZ: a + c = constant

  std::cout << "ZYX R-: " << std::endl
            << math::eulerZYXToMatrix(locked) << std::endl;
  std::cout << "ZYX R+: " << std::endl
            << math::eulerZYXToMatrix(lockedPos) << std::endl;
  // Conclude ZYX: a - c = constant

  std::cout << "ZXY R-: " << std::endl
            << math::eulerZXYToMatrix(locked) << std::endl;
  std::cout << "ZXY R+: " << std::endl
            << math::eulerZXYToMatrix(lockedPos) << std::endl;
  // Conclude ZXY: a + c = constant

  std::cout << "XZY R-: " << std::endl
            << math::eulerXZYToMatrix(locked) << std::endl;
  std::cout << "XZY R+: " << std::endl
            << math::eulerXZYToMatrix(lockedPos) << std::endl;
  // Conclude XZY: a - c = constant
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, TORSO_GIMBAL_LOCK)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim");
  std::shared_ptr<dynamics::Skeleton> skeletonBallJoints
      = standard.skeleton->convertSkeletonToBallJoints();

  Eigen::Vector3s ballCoords = Eigen::Vector3s(1.16096, 1.28402, 1.28242);
  dynamics::EulerFreeJoint* eulerFreeJoint
      = static_cast<dynamics::EulerFreeJoint*>(standard.skeleton->getJoint(0));

  Eigen::Matrix3s R = math::expMapRot(ballCoords);
  Eigen::Vector3s euler = Eigen::Vector3s::Zero();
  if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::XYZ)
  {
    euler = math::matrixToEulerXYZ(R).cwiseProduct(
        eulerFreeJoint->getFlipAxisMap());
  }
  else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::XZY)
  {
    euler = math::matrixToEulerXZY(R).cwiseProduct(
        eulerFreeJoint->getFlipAxisMap());
  }
  else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::ZXY)
  {
    std::cout << "ZXY" << std::endl;
    euler = math::matrixToEulerZXY(R).cwiseProduct(
        eulerFreeJoint->getFlipAxisMap());
  }
  else if (eulerFreeJoint->getAxisOrder() == EulerJoint::AxisOrder::ZYX)
  {
    euler = math::matrixToEulerZYX(R).cwiseProduct(
        eulerFreeJoint->getFlipAxisMap());
  }
  else
  {
    assert(false && "Unsupported AxisOrder when decoding EulerFreeJoint");
  }
  // Do our best to pick an equivalent set of EulerAngles that's within
  // joint bounds, if one exists
  Eigen::Vector3s eulerClamped = math::attemptToClampEulerAnglesToBounds(
      euler,
      eulerFreeJoint->getPositionUpperLimits().head<3>(),
      eulerFreeJoint->getPositionLowerLimits().head<3>());

  // ZXY

  Eigen::Vector3s recoveredBallCoords
      = math::logMap(EulerJoint::convertToTransform(
                         eulerClamped,
                         eulerFreeJoint->getAxisOrder(),
                         eulerFreeJoint->getFlipAxisMap())
                         .linear());

  Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(3, 5);
  diff.col(0) = ballCoords;
  diff.col(1) = euler;
  diff.col(2) = eulerClamped;
  diff.col(3) = recoveredBallCoords;
  diff.col(4) = eulerFreeJoint->getFlipAxisMap();

  std::cout << "Diff = " << (recoveredBallCoords - ballCoords).norm()
            << std::endl;
  std::cout << "Axis order: " << (int)eulerFreeJoint->getAxisOrder()
            << std::endl;
  std::cout << "Ball - Euler - Clamped euler - Recovered - Flips" << std::endl
            << diff << std::endl;

  Eigen::Matrix3s recoveredR = math::expMapRot(recoveredBallCoords);
  Eigen::Matrix3s diffR = R - recoveredR;
  std::cout << "||diffR|| = " << diffR.norm() << std::endl;
  std::cout << "diffR = " << std::endl << diffR << std::endl;

  /*
  Eigen::VectorXs pos = Eigen::VectorXs(56);
  pos << 1.16096, 1.28402, 1.28242, 0.614382, 1.01753, 0.907596, -0.307936,
      1.94313, 0.569772, 0.405994, -0.323718, 0.120494, 0.705249, 1.46973,
      -1.72269, 0, 0.00133662, 0.369738, -0.329586, 1.1396, 0.123183, -0.456081,
      0.33667, -0.696457, 0.192617, 0, -0.0479091, -0.0115866, -0.15661,
      1.10424, 1.27731, 0.901787, 1, 1, 1, 1, 1.06463, 0.933611, 1, 1, 1.01391,
      1, 1.31495, 0.839322, 1.35261, 0.741261, 1.07313, 0.896375, 1.08346,
      1.1542, 1, 1, 1, 1.04301, 0.926338, 1.29632;

  // Verify the translation is lossless
  Eigen::VectorXs eulerPos
      = standard.skeleton->convertPositionsFromBallSpace(pos);
  Eigen::VectorXs recovered
      = standard.skeleton->convertPositionsToBallSpace(eulerPos);

  Eigen::MatrixXs diff = Eigen::MatrixXs::Zero(pos.size(), 3);
  diff.col(0) = pos;
  diff.col(1) = recovered;
  diff.col(2) = pos - recovered;
  std::cout << "Pos - Recovered - Diff" << std::endl << diff << std::endl;

  Eigen::VectorXs bodyPoses
      = Eigen::VectorXs::Zero(skeletonBallJoints->getNumBodyNodes() * 3);
  skeletonBallJoints->setPositions(pos);
  for (int i = 0; i < skeletonBallJoints->getNumBodyNodes(); i++)
  {
    bodyPoses.segment<3>(i * 3)
        = skeletonBallJoints->getBodyNode(i)->getWorldTransform().translation();
  }
  skeletonBallJoints->setPositions(recovered);
  Eigen::VectorXs recoveredBodyPoses
      = Eigen::VectorXs::Zero(skeletonBallJoints->getNumBodyNodes() * 3);
  for (int i = 0; i < skeletonBallJoints->getNumBodyNodes(); i++)
  {
    recoveredBodyPoses.segment<3>(i * 3)
        = skeletonBallJoints->getBodyNode(i)->getWorldTransform().translation();
  }
  std::cout << "!!!!!! Got a recovery error of "
            << (recoveredBodyPoses - bodyPoses).norm() << std::endl;
  */
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, COMPLEX_KNEE_OFFSETS_ISSUE)
{
  /*
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/"
      "gait2392_frontHingeKnee_dem_rescaled.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->getBodyNode("femur_l")->setScale(
      Eigen::Vector3s(1.0, 0.75, 1.0));
  OpenSimParser::saveOsimScalingXMLFile(
      "test",
      standard.skeleton,
      50,
      1.6,
      "gait2392_frontHingeKnee_dem.osim",
      "gait2392_frontHingeKnee_dem_rescaled.osim",
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/ComplexKnee/"
      "rescale.xml");
  */

  /*
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderSkeleton(
      standard.skeleton, "skel", Eigen::Vector4s(0.5, 0.5, 0.5, 1.0));

  std::map<std::string, Eigen::Vector3s> jointPoses
      = standard.skeleton->getJointWorldPositionsMap();

  auto* body = standard.skeleton->getBodyNode("femur_l");
  // body->setScale(Eigen::Vector3s(1, 0.75, 1));
  std::cout << "Body scale: " << body->getScale() << std::endl;
  std::cout << "Parent joint name: " << body->getParentJoint()->getName()
            << std::endl;
  std::cout << "Parent T:" << std::endl
            << body->getParentJoint()->getTransformFromChildBodyNode().matrix()
            << std::endl;
  std::cout << "Parent relative T:" << std::endl
            << body->getParentJoint()->getRelativeTransform().matrix()
            << std::endl;
  std::cout << "Child T:" << std::endl
            << body->getChildJoint(0)->getTransformFromParentBodyNode().matrix()
            << std::endl;
  std::cout << "Child relative T:" << std::endl
            << body->getChildJoint(0)->getRelativeTransform().matrix()
            << std::endl;

  std::vector<Eigen::Vector3s> line1;
  line1.push_back(body->getParentBodyNode()->getWorldTransform().translation());
  line1.push_back(jointPoses[body->getParentJoint()->getName()]);

  std::vector<Eigen::Vector3s> line2;
  line2.push_back(jointPoses[body->getParentJoint()->getName()]);
  line2.push_back(body->getWorldTransform().translation());

  std::vector<Eigen::Vector3s> line3;
  line3.push_back(body->getWorldTransform().translation());
  line3.push_back(jointPoses[body->getChildJoint(0)->getName()]);

  std::vector<Eigen::Vector3s> line4;
  line4.push_back(jointPoses[body->getChildJoint(0)->getName()]);
  line4.push_back(body->getChildBodyNode(0)->getWorldTransform().translation());
  server->createLine("line_1", line1, Eigen::Vector4s(1, 0, 0, 1));
  server->createLine("line_2", line2, Eigen::Vector4s(0, 1, 0, 1));
  server->createLine("line_3", line3, Eigen::Vector4s(0, 0, 1, 1));
  server->createLine("line_4", line4, Eigen::Vector4s(0, 1, 1, 1));

  server->blockWhileServing();
  */

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/"
      "gait2392_frontHingeKnee_dem.osim");
  standard.skeleton->zeroTranslationInCustomFunctions();

  std::string bodyNode = "femur_l";

  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderSkeleton(standard.skeleton);
  server->createSphere(
      "x", 0.01, Eigen::Vector3s::Zero(), Eigen::Vector4s(1, 0, 0, 1));
  server->createSphere(
      "y", 0.01, Eigen::Vector3s::Zero(), Eigen::Vector4s(0, 1, 0, 1));
  server->createSphere(
      "z", 0.01, Eigen::Vector3s::Zero(), Eigen::Vector4s(0, 0, 1, 1));

  Ticker ticker = Ticker(0.01);
  ticker.registerTickListener([&](long t) {
    long PERIOD = 5000;
    long offset = t % PERIOD;
    s_t percentage = (s_t)offset / PERIOD;

    int axis = (int)(std::floor((s_t)(t % (3 * PERIOD)) / (PERIOD))) % 3;

    auto* body = standard.skeleton->getBodyNode(bodyNode);
    body->setScaleLowerBound(Eigen::Vector3s::Zero());

    Eigen::Vector3s scale = Eigen::Vector3s::Ones();
    scale(axis) = 1.0 - (percentage * 0.55);
    body->setScale(scale);

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
    markers.push_back(std::make_pair(body, Eigen::Vector3s::UnitX()));
    markers.push_back(std::make_pair(body, Eigen::Vector3s::UnitY()));
    markers.push_back(std::make_pair(body, Eigen::Vector3s::UnitZ()));
    Eigen::VectorXs markerWorldPos
        = standard.skeleton->getMarkerWorldPositions(markers);
    server->setObjectPosition("x", markerWorldPos.segment<3>(0));
    server->setObjectPosition("y", markerWorldPos.segment<3>(3));
    server->setObjectPosition("z", markerWorldPos.segment<3>(6));

    server->renderSkeleton(standard.skeleton);
  });

  server->registerConnectionListener([&]() { ticker.start(); });
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_COMPLEX_KNEE_C3D)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_Rajagopal_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/ComplexKnee/gait2392_frontHingeKnee_dem.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->zeroTranslationInCustomFunctions();

  /*
  debugIKInitializationToGUI(
      standard.skeleton,
      Eigen::VectorXs::Zero(standard.skeleton->getNumDofs()),
      0.05);
  */

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d
      = C3DLoader::loadC3D("dart://sample/osim/ComplexKnee/2022_01_0403.c3d");
  C3DLoader::fixupMarkerFlips(&c3d);
  /*
  Eigen::Matrix3s R = math::eulerXYZToMatrix(Eigen::Vector3s(-M_PI / 2, 0, 0));
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    for (auto& pair : c3d.markerTimesteps[i])
    {
      pair.second = R * pair.second;
    }
  }
  */

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.005);
  fitter.setInitialIKMaxRestarts(200);
  fitter.setIterationLimit(300);
  fitter.setMaxJointWeight(1.0);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);
  // fitter.setRegularizeAllBodyScales(0.0);
  // fitter.setRegularizeIndividualBodyScales(0.0);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTrackingMarkers(standard.trackingMarkers);
  // fitter.setTriadsToTracking();

  fitter.autorotateC3D(&c3d);
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  markerObservationTrials.push_back(c3d.markerTimesteps);

  /*
  std::vector<bool> newClip;
  for (int i = 0; i < markerObservationTrials[0].size(); i++)
  {
    newClip.push_back(false);
  }
  MarkerInitialization finalKinematicInit = fitter.getInitialization(
      markerObservationTrials[0], newClip, InitialMarkerFitParams());
  */

  std::vector<MarkerInitialization> inits
      = fitter.runMultiTrialKinematicsPipeline(
          markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(5)
              .setMaxTimestepsToUseForMultiTrialScaling(4000),
          150);
  MarkerInitialization finalKinematicInit = inits[0];

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      c3d.markerTimesteps);

  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  std::cout << "Pelvis scaling: "
            << standard.skeleton->getBodyNode("pelvis")->getScale()
            << std::endl;
  std::cout << "Torso scaling: "
            << standard.skeleton->getBodyNode("torso")->getScale() << std::endl;
  std::cout << "femur_l scaling: "
            << standard.skeleton->getBodyNode("femur_l")->getScale()
            << std::endl;
  std::cout
      << "sagittal_knee_body_l scaling:"
      << standard.skeleton->getBodyNode("sagittal_knee_body_l")->getScale()
      << std::endl;
  std::cout
      << "aux_intercond_body_l scaling: "
      << standard.skeleton->getBodyNode("aux_intercond_body_l")->getScale()
      << std::endl;
  std::cout << "tibia_l scaling: "
            << standard.skeleton->getBodyNode("tibia_l")->getScale()
            << std::endl;

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, c3d.markerTimesteps, &c3d);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_APRIL4_C3D)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/April4/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D("dart://sample/osim/April4/April406.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  std::vector<bool> newClip;
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = c3d.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(200);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  /*
OpenSimParser::saveMot(
    standard.skeleton,
    "./autoIK.mot",
    c3d.timestamps,
    finalKinematicInit.poses);
    */

  std::cout << "Body scales:" << std::endl;
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    std::cout << standard.skeleton->getBodyNode(i)->getName() << ":"
              << std::endl;
    std::cout << standard.skeleton->getBodyNode(i)->getScale() << std::endl;
  }

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  /*
  drawTimeSeriesFigureToGUI(
      server, standard.skeleton, finalKinematicInit, subsetTimesteps);
*/
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps, &c3d);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, NAN_C3D_PROBLEMS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/NaNSubject/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D("dart://sample/osim/NaNSubject/markers.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  std::vector<bool> newClip;
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = c3d.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton, init.updatedMarkerMap, init.poses, subsetTimesteps);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(200);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  OpenSimParser::saveMot(
      standard.skeleton,
      "./autoIK.mot",
      c3d.timestamps,
      finalKinematicInit.poses);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps, &c3d);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, RAJAGOPAL_C3D)
{
  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D(
      "dart://sample/osim/MichaelTest/results/C3D/S02DN101.c3d");

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderBasis();
  std::cout << "Data rotation: " << std::endl << c3d.dataRotation << std::endl;
  C3DLoader::debugToGUI(c3d, server);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, WELK_C3D)
{
  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D(
      "dart://sample/osim/welk007/c3d_Trimmed_running_exotendon3.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  // Get the gold data
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/welk007/manually_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMotAtLowestMarkerRMSERotation(
      scaled,
      "dart://sample/osim/welk007/"
      "c3d_Trimmed_running_exotendon3_manual_scaling_ik.mot",
      c3d);
  Eigen::MatrixXs goldPoses = mot.poses;

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderBasis();
  std::cout << "Data rotation: " << std::endl << c3d.dataRotation << std::endl;
  MarkerFitter::debugGoldTrajectoryAndMarkersToGUI(
      server, &c3d, &scaled, goldPoses);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_STACK_WELK)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/welk002/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D("dart://sample/osim/welk002/markers.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  // Get the gold data
  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/welk002/manually_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMotAtLowestMarkerRMSERotation(
      scaled, "dart://sample/osim/welk002/manual_ik.mot", c3d);
  Eigen::MatrixXs goldPoses = mot.poses;
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      subMarkerTimesteps,
      anthropometrics);

  std::vector<bool> newClip;
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = c3d.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(200);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Manual error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderBasis();
  std::cout << "Data rotation: " << std::endl << c3d.dataRotation << std::endl;
  // , &scaled, goldPoses
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps, &c3d, &scaled, goldPoses);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, MULTI_TRIAL_SPRINTER)
{
  bool isDebug = false;
#ifndef NDEBUG
  isDebug = true;
#endif

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard
      = OpenSimParser::parseOsim("dart://sample/osim/Sprinter/sprinter.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  int numCopiesOfData = 1;

  // Get the raw marker trajectory data
  OpenSimTRC traj350
      = OpenSimParser::loadTRC("dart://sample/osim/Sprinter/run0350cms.trc");
  OpenSimTRC traj500
      = OpenSimParser::loadTRC("dart://sample/osim/Sprinter/run0500cms.trc");
  OpenSimTRC traj700
      = OpenSimParser::loadTRC("dart://sample/osim/Sprinter/run0700cms.trc");
  OpenSimTRC traj900
      = OpenSimParser::loadTRC("dart://sample/osim/Sprinter/run0900cms.trc");
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  for (int i = 0; i < numCopiesOfData; i++)
  {
    markerObservationTrials.push_back(traj350.markerTimesteps);
    markerObservationTrials.push_back(traj500.markerTimesteps);
    markerObservationTrials.push_back(traj700.markerTimesteps);
    markerObservationTrials.push_back(traj900.markerTimesteps);
  }

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Sprinter/sprinter_scaled.osim");
  OpenSimMot traj350Mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/Sprinter/run0350cms.mot");
  OpenSimMot traj500Mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/Sprinter/run0500cms.mot");
  OpenSimMot traj700Mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/Sprinter/run0700cms.mot");
  OpenSimMot traj900Mot = OpenSimParser::loadMot(
      scaled.skeleton, "dart://sample/osim/Sprinter/run0900cms.mot");
  std::vector<Eigen::MatrixXs> goldPoses;
  for (int i = 0; i < numCopiesOfData; i++)
  {
    goldPoses.push_back(traj350Mot.poses);
    goldPoses.push_back(traj500Mot.poses);
    goldPoses.push_back(traj700Mot.poses);
    goldPoses.push_back(traj900Mot.poses);
  }

  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      shorterTrials;
  std::vector<Eigen::MatrixXs> shorterGoldPoses;
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    shorterTrials.emplace_back();
    std::vector<std::map<std::string, Eigen::Vector3s>>& shortTrial
        = shorterTrials[shorterTrials.size() - 1];
    int shorterSize = (i + 3) * 2;
    for (int k = 0; k < shorterSize; k++)
    {
      shortTrial.push_back(markerObservationTrials[i][k]);
    }
    shorterGoldPoses.push_back(
        goldPoses[i].block(0, 0, goldPoses[i].rows(), shorterSize));
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(isDebug ? 1 : 150);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(isDebug ? 5 : 400);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<MarkerInitialization> inits
      = fitter.runMultiTrialKinematicsPipeline(
          isDebug ? shorterTrials : markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(2)
              .setMaxTimestepsToUseForMultiTrialScaling(2000),
          50);

  standard.skeleton->setGroupScales(inits[0].groupScales);
  for (int i = 0; i < 4; i++)
  {
    std::cout << "Manual error report " << i << ":" << std::endl;
    IKErrorReport manualKinematicsReport(
        scaled.skeleton,
        scaled.markersMap,
        isDebug ? shorterGoldPoses[i] : goldPoses[i],
        isDebug ? shorterTrials[i] : markerObservationTrials[i],
        anthropometrics);
    manualKinematicsReport.printReport(5);

    std::cout << "Auto error report " << i << ":" << std::endl;
    IKErrorReport finalKinematicsReport(
        standard.skeleton,
        inits[0].updatedMarkerMap,
        inits[i].poses,
        isDebug ? shorterTrials[i] : markerObservationTrials[i],
        anthropometrics);
    finalKinematicsReport.printReport(5);
  }

  /*
for (int i = 0; i < 4; i++)
{
  fitter.saveTrajectoryAndMarkersToGUI(
      "./test" + std::to_string(i) + ".json",
      inits[i],
      isDebug ? shorterTrials[i] : markerObservationTrials[i]);
}
*/

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, inits[0], markerObservationTrials[0]);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, SINGLE_TRIAL_MICHAEL)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_Rajagopal_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest/results/Models/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  C3D c3d = C3DLoader::loadC3D(
      "dart://sample/osim/MichaelTest/results/C3D/S02DN101.c3d");

  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps
      = c3d.markerTimesteps;

  std::vector<bool> newClip;
  for (int i = 0; i < c3d.markerTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = c3d.markerTimesteps;

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  // , &scaled, goldPoses
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps, &c3d);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, MULTI_TRIAL_MICHAEL)
{
  bool isDebug = false;
#ifndef NDEBUG
  isDebug = true;
#endif

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_Rajagopal_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest/results/Models/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  for (int i = 3; i <= 5; i++) // 96
  {
    std::string number;
    if (i < 10)
    {
      number = "0" + std::to_string(i);
    }
    else
    {
      number = std::to_string(i);
    }
    C3D c3d = C3DLoader::loadC3D(
        "dart://sample/osim/MichaelTest/results/C3D/S02DN1" + number + ".c3d");
    markerObservationTrials.push_back(c3d.markerTimesteps);
  }

  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      shorterTrials;
  std::vector<Eigen::MatrixXs> shorterGoldPoses;
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    shorterTrials.emplace_back();
    std::vector<std::map<std::string, Eigen::Vector3s>>& shortTrial
        = shorterTrials[shorterTrials.size() - 1];
    int shorterSize = (i + 3) * 2;
    for (int k = 0; k < shorterSize; k++)
    {
      shortTrial.push_back(markerObservationTrials[i][k]);
    }
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(isDebug ? 1 : 50);
  fitter.setRegularizeAnatomicalMarkerOffsets(10.0);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(isDebug ? 5 : 400);

  // Set all the triads to be tracking markers, instead of anatomical
  // fitter.setTriadsToTracking();
  fitter.setTrackingMarkers(standard.trackingMarkers);

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<MarkerInitialization> inits
      = fitter.runMultiTrialKinematicsPipeline(
          isDebug ? shorterTrials : markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(5)
              .setMaxTimestepsToUseForMultiTrialScaling(4000),
          150);

  standard.skeleton->setGroupScales(inits[0].groupScales);
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    std::cout << "Auto error report " << i << ":" << std::endl;
    IKErrorReport finalKinematicsReport(
        standard.skeleton,
        inits[0].updatedMarkerMap,
        inits[i].poses,
        isDebug ? shorterTrials[i] : markerObservationTrials[i],
        anthropometrics);
    finalKinematicsReport.printReport(5);
  }

  /*
for (int i = 0; i < 4; i++)
{
  fitter.saveTrajectoryAndMarkersToGUI(
      "./test" + std::to_string(i) + ".json",
      inits[i],
      isDebug ? shorterTrials[i] : markerObservationTrials[i]);
}
*/

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, inits[0], markerObservationTrials[0]);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, MULTI_TRIAL_MICHAEL_2)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = (59 * 2.204) * 0.001;
  observedValues["Heightin"] = (1.72 * 39.370) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/MichaelTest2/Models/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  C3D c3d
      = C3DLoader::loadC3D("dart://sample/osim/MichaelTest2/C3D/standing.c3d");
  markerObservationTrials.push_back(c3d.markerTimesteps);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.1);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(500);

  fitter.setRegularizeAnatomicalMarkerOffsets(10.0);
  fitter.setRegularizeTrackingMarkerOffsets(0.05);
  /*
  fitter.setMinSphereFitScore(0.01);
  fitter.setMinAxisFitScore(0.001);
  */

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Set all the triads to be tracking markers, instead of anatomical
  if (standard.anatomicalMarkers.size() > 0)
  {
    fitter.setTrackingMarkers(standard.trackingMarkers);
  }
  else
  {
    fitter.setTriadsToTracking();
  }

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  /*
//////////////////////////////
std::vector<std::map<std::string, Eigen::Vector3s>> dummyClip;
std::vector<bool> newClip;
for (int i = 0; i < 5; i++)
{
  dummyClip.push_back(markerObservationTrials[0][i]);
  newClip.push_back(false);
}

MarkerInitialization init = fitter.getInitialization(
    dummyClip,
    newClip,
    InitialMarkerFitParams()
        .setMaxTrialsToUseForMultiTrialScaling(5)
        .setMaxTimestepsToUseForMultiTrialScaling(4000));
MarkerInitialization otherInit = fitter.smoothOutIK(dummyClip, init);
return;

//////////////////////////////
*/

  std::vector<MarkerInitialization> inits
      = fitter.runMultiTrialKinematicsPipeline(
          markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(5)
              .setMaxTimestepsToUseForMultiTrialScaling(4000),
          150);

  standard.skeleton->setGroupScales(inits[0].groupScales);
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    std::cout << "Auto error report " << i << ":" << std::endl;
    IKErrorReport finalKinematicsReport(
        standard.skeleton,
        inits[0].updatedMarkerMap,
        inits[i].poses,
        markerObservationTrials[i],
        anthropometrics);
    finalKinematicsReport.printReport(5);
  }

  /*
for (int i = 0; i < 4; i++)
{
  fitter.saveTrajectoryAndMarkersToGUI(
      "./test" + std::to_string(i) + ".json",
      inits[i],
      isDebug ? shorterTrials[i] : markerObservationTrials[i]);
}
*/
  std::cout << "Pelvis scaling: "
            << standard.skeleton->getBodyNode("pelvis")->getScale()
            << std::endl;
  std::cout << "Torso scaling: "
            << standard.skeleton->getBodyNode("torso")->getScale() << std::endl;

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, inits[0], markerObservationTrials[0]);
  server->blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, MONOLEVEL_VS_BILEVEL)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 150 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 10) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    subMarkerTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      subMarkerTimesteps,
      anthropometrics);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < subsetTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(2000);

  ////////////////////////////////////////////////////
  // Perform the monolevel fit with constraints
  ////////////////////////////////////////////////////
  std::shared_ptr<BilevelFitResult> monolevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50, false);
  // Fine-tune IK and re-fit all the points
  MarkerInitialization monolevelFinalKinematicInit
      = fitter.completeBilevelResult(
          subsetTimesteps,
          newClip,
          monolevelFit,
          InitialMarkerFitParams()
              .setJointCentersAndWeights(
                  reinit.joints, reinit.jointCenters, reinit.jointWeights)
              .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
              .setInitPoses(reinit.poses)
              .setDontRescaleBodies(true)
              .setGroupScales(monolevelFit->groupScales)
              .setMarkerOffsets(monolevelFit->markerOffsets));
  for (auto pair : monolevelFinalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }
  IKErrorReport monolevelFinalKinematicsReport(
      standard.skeleton,
      monolevelFinalKinematicInit.updatedMarkerMap,
      monolevelFinalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);
  ////////////////////////////////////////////////////
  // Perform the bilevel fit with constraints
  ////////////////////////////////////////////////////
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);
  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));
  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }
  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Experts's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Monolevel Final kinematic fit report:" << std::endl;
  monolevelFinalKinematicsReport.printReport(5);
  std::cout << "Bilevel Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);
}
#endif

#ifdef OPENSIM_TESTS
TEST(MarkerFitter, SIMPLE_SCALING_TEST)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  Eigen::Vector3s scaling = Eigen::Vector3s::Random().cwiseAbs();
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    standard.skeleton->getBodyNode(i)->setScaleUpperBound(scaling);
    standard.skeleton->getBodyNode(i)->setScale(scaling);
  }

  std::vector<std::string> markerNames;
  for (auto& pair : standard.markersMap)
  {
    markerNames.push_back(pair.first);
  }
  for (std::string marker : markerNames)
  {
    standard.markersMap[marker].second = Eigen::Vector3s::Ones();
  }

  OpenSimMot mot = OpenSimParser::loadMot(
      standard.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/S01DN603_ik.mot");
  OpenSimParser::saveMot(
      standard.skeleton,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/in_rad.mot",
      mot.timestamps,
      mot.poses);
  OpenSimParser::saveBodyLocationsMot(
      standard.skeleton,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/body_world.mot",
      mot.timestamps,
      mot.poses);
  OpenSimParser::saveMarkerLocationsMot(
      standard.skeleton,
      standard.markersMap,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/marker_world.mot",
      mot.timestamps,
      mot.poses);

  /// Save out the scaling file

  OpenSimParser::saveOsimScalingXMLFile(
      standard.skeleton,
      10.0,
      1.3,
      "Rajagopal2015_passiveCal_hipAbdMoved.osim",
      "rescaled.osim",
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/scaling_instructions.xml");

  std::cout << "Run:" << std::endl;
  std::cout << "cd "
               "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
               "Rajagopal2015_v3_scaled/"
            << std::endl;
  std::cout << "opensim-cmd run-tool "
               "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
               "Rajagopal2015_v3_scaled/scaling_instructions.xml"
            << std::endl;
  /*
  chdir(
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/");
  execl(
      "opensim-cmd",
      "run-tool",
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/scaling_instructions.xml");
  */

  /// Modify the current rescaled OpenSim file by moving the markers around

  std::map<std::string, Eigen::Vector3s> bodyScalesMap;
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    bodyScalesMap[standard.skeleton->getBodyNode(i)->getName()]
        = standard.skeleton->getBodyNode(i)->getScale();
  }
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>>
      markerOffsetsMap;
  for (std::string& markerName : markerNames)
  {
    markerOffsetsMap[markerName] = std::make_pair(
        standard.markersMap[markerName].first->getName(),
        standard.markersMap[markerName].second);
  }

  standard.skeleton->getBodyScales();

  OpenSimParser::moveOsimMarkers(
      "dart://sample/osim/Rajagopal2015_v3_scaled/rescaled.osim",
      bodyScalesMap,
      markerOffsetsMap,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/moved_markers.osim");

  OpenSimFile recovered = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/rescaled.osim");

  recovered.skeleton->setPositions(standard.skeleton->getPositions());

  MarkerMap recoveredMarkers;
  for (auto& pair : standard.markersMap)
  {
    recoveredMarkers[pair.first] = std::make_pair(
        recovered.skeleton->getBodyNode(pair.second.first->getName()),
        pair.second.second);
  }
  std::map<std::string, Eigen::Vector3s> standardMap
      = standard.skeleton->getJointWorldPositionsMap();
  std::map<std::string, Eigen::Vector3s> recoveredMap
      = recovered.skeleton->getJointWorldPositionsMap();

  for (auto& pair : standardMap)
  {
    Eigen::Vector3s recoveredVec = recoveredMap[pair.first];
    Eigen::Vector3s diff = pair.second - recoveredVec;
    s_t dist = diff.norm();
    std::cout << "Joint " << pair.first << " dist: " << dist << std::endl;
  }

  /*
// Target markers
std::shared_ptr<server::GUIWebsocketServer> server
    = std::make_shared<server::GUIWebsocketServer>();
server->serve(8070);
server->renderSkeleton(standard.skeleton);
server->renderSkeleton(
    recovered.skeleton, "recovered", Eigen::Vector4s(1, 0, 0, 1));
server->blockWhileServing();
*/
}
#endif

#ifdef OPENSIM_TESTS
TEST(MarkerFitter, OPENSIM_COMPAT_TEST_1)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 150 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 10) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    subMarkerTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      subMarkerTimesteps,
      anthropometrics);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < subsetTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(100);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 50);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Experts's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  ////////////////////////////////////////////////////////////
  // Write out the OSIM file along with our verification data
  ////////////////////////////////////////////////////////////

  std::vector<std::string> markerNames;
  for (auto& pair : standard.markersMap)
  {
    markerNames.push_back(pair.first);
  }
  for (std::string marker : markerNames)
  {
    standard.markersMap[marker].second = Eigen::Vector3s::Ones();
  }

  OpenSimParser::saveMot(
      standard.skeleton,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/in_rad.mot",
      mot.timestamps,
      finalKinematicInit.poses);
  OpenSimParser::saveBodyLocationsMot(
      standard.skeleton,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/body_world.mot",
      mot.timestamps,
      finalKinematicInit.poses);
  OpenSimParser::saveMarkerLocationsMot(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/marker_world.mot",
      mot.timestamps,
      finalKinematicInit.poses);

  /// Save out the scaling file

  OpenSimParser::saveOsimScalingXMLFile(
      standard.skeleton,
      10.0,
      1.3,
      "Rajagopal2015_passiveCal_hipAbdMoved.osim",
      "rescaled.osim",
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/scaling_instructions.xml");

  std::cout << "Run:" << std::endl;
  std::cout << "cd "
               "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
               "Rajagopal2015_v3_scaled/"
            << std::endl;
  std::cout << "opensim-cmd run-tool "
               "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
               "Rajagopal2015_v3_scaled/scaling_instructions.xml"
            << std::endl;
  // Wait for user input
  std::cout << "Press enter when finished: " << std::endl;
  getchar();
  /*
  chdir(
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/");
  execl(
      "opensim-cmd",
      "run-tool",
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/scaling_instructions.xml");
  */

  /// Modify the current rescaled OpenSim file by moving the markers around

  std::cout << "Moving the markers" << std::endl;

  std::map<std::string, Eigen::Vector3s> bodyScalesMap;
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    bodyScalesMap[standard.skeleton->getBodyNode(i)->getName()]
        = standard.skeleton->getBodyNode(i)->getScale();
  }
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>>
      markerOffsetsMap;
  for (std::string& markerName : markerNames)
  {
    markerOffsetsMap[markerName] = std::make_pair(
        finalKinematicInit.updatedMarkerMap[markerName].first->getName(),
        finalKinematicInit.updatedMarkerMap[markerName].second);
  }

  standard.skeleton->getBodyScales();

  OpenSimParser::moveOsimMarkers(
      "dart://sample/osim/Rajagopal2015_v3_scaled/rescaled.osim",
      bodyScalesMap,
      markerOffsetsMap,
      "/Users/keenonwerling/Desktop/dev/nimblephysics/data/osim/"
      "Rajagopal2015_v3_scaled/moved_markers.osim");

  std::cout << "Markers moved, and written out to moved_markers.osim"
            << std::endl;

  /*
// Target markers
std::shared_ptr<server::GUIWebsocketServer> server
    = std::make_shared<server::GUIWebsocketServer>();
server->serve(8070);
server->renderSkeleton(standard.skeleton);
server->renderSkeleton(
    recovered.skeleton, "recovered", Eigen::Vector4s(1, 0, 0, 1));
server->blockWhileServing();
*/
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, FULL_KINEMATIC_RAJAGOPAL)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 150 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 10) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603.trc");

  OpenSimFile scaled = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim");
  OpenSimMot mot = OpenSimParser::loadMot(
      scaled.skeleton,
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "S01DN603_ik.mot");

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    subMarkerTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  IKErrorReport goldReport(
      scaled.skeleton,
      scaled.markersMap,
      goldPoses,
      subMarkerTimesteps,
      anthropometrics);

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(100);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTrackingMarkers(standard.trackingMarkers);
  fitter.setRegularizeAnatomicalMarkerOffsets(10.0);

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = markerTrajectories.markerTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < subsetTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints, init.jointCenters, init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  fitter.setIterationLimit(400);
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 150);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints, reinit.jointCenters, reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Experts's data error report:" << std::endl;
  goldReport.printReport(5);
  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  // Target markers
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  fitter.debugTrajectoryAndMarkersToGUI(
      server, finalKinematicInit, subsetTimesteps);
  server->blockWhileServing();
}
#endif

void evaluateOnSyntheticData(
    std::string standardModelPath,
    std::string standardModelAnthropometrics,
    std::string scaledModelPath,
    std::string motPath)
{
  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(standardModelAnthropometrics);

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = 150 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 10) * 0.001;

  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);

  OpenSimFile standard = OpenSimParser::parseOsim(
      standardModelPath); // Rajagopal2015_passiveCal_hipAbdMoved.osim
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->setScaleGroupUniformScaling(
      standard.skeleton->getBodyNode("hand_r"));

  for (auto pair : standard.markersMap)
  {
    assert(pair.second.first != nullptr);
  }

  OpenSimFile scaled = OpenSimParser::parseOsim(scaledModelPath);
  OpenSimMot mot = OpenSimParser::loadMot(scaled.skeleton, motPath);

  Eigen::MatrixXs goldPoses = mot.poses;
  std::vector<std::map<std::string, Eigen::Vector3s>> subMarkerTimesteps;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    scaled.skeleton->setPositions(goldPoses.col(i));
    subMarkerTimesteps.push_back(
        scaled.skeleton->getMarkerMapWorldPositions(scaled.markersMap));
  }

  // Create a marker fitter

  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.05);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(500);

  // Set all the triads to be tracking markers, instead of anatomical
  fitter.setTriadsToTracking();

  for (int i = 0; i < fitter.getNumMarkers(); i++)
  {
    std::string name = fitter.getMarkerNameAtIndex(i);
    std::cout << name << " is tracking: " << fitter.getMarkerIsTracking(name)
              << std::endl;
  }

  std::vector<std::map<std::string, Eigen::Vector3s>> subsetTimesteps;
  /*
  for (int i = 0; i < 10; i++)
  {
    subsetTimesteps.push_back(markerTrajectories.markerTimesteps[i]);
  }
  */
  subsetTimesteps = subMarkerTimesteps;
  std::vector<bool> newClip;
  for (int i = 0; i < subsetTimesteps.size(); i++)
  {
    newClip.push_back(false);
  }

  MarkerInitialization init = fitter.getInitialization(
      subsetTimesteps, newClip, InitialMarkerFitParams());

  for (auto pair : init.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport initReport(
      standard.skeleton,
      init.updatedMarkerMap,
      init.poses,
      subsetTimesteps,
      anthropometrics);

  standard.skeleton->setGroupScales(init.groupScales);

  // init.joints.push_back(standard.skeleton->getJoint("walker_knee_r"));
  // init.jointCenters = Eigen::MatrixXs::Zero(3, init.poses.cols());
  // fitter.findJointCenter(0, init, subsetTimesteps);

  fitter.findJointCenters(init, newClip, subsetTimesteps);
  fitter.findAllJointAxis(init, newClip, subsetTimesteps);
  fitter.computeJointConfidences(init, subsetTimesteps);

  for (int i = 0; i < init.joints.size(); i++)
  {
    s_t diff = 0.0;
    for (int t = 0; t < goldPoses.cols(); t++)
    {
      scaled.skeleton->setPositions(goldPoses.col(t));

      Eigen::Vector3s jointCenter = init.jointCenters.block<3, 1>(i * 3, t);
      Eigen::Vector3s goldJointCenter
          = scaled.skeleton
                ->getJointWorldPositionsMap()[init.joints[i]->getName()];
      diff += (jointCenter - goldJointCenter).norm();
    }
    diff /= goldPoses.cols();
    std::cout << "Joint " << init.joints[i]->getName() << " avg error: " << diff
              << "m" << std::endl;
  }

  // Re-initialize the problem, but pass in the joint centers we just found
  MarkerInitialization reinit = fitter.getInitialization(
      subsetTimesteps,
      newClip,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              init.joints,
              init.jointCenters,
              init.jointsAdjacentMarkers,
              init.jointWeights)
          .setJointAxisAndWeights(init.jointAxis, init.axisWeights)
          .setInitPoses(init.poses));

  for (auto pair : reinit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport afterJointCentersReport(
      standard.skeleton,
      reinit.updatedMarkerMap,
      reinit.poses,
      subsetTimesteps,
      anthropometrics);

  // fitter.setAnthropometricPrior(anthropometrics, 0.1);

  // Bilevel optimization
  std::shared_ptr<BilevelFitResult> bilevelFit
      = fitter.optimizeBilevel(subsetTimesteps, reinit, 200);

  // Fine-tune IK and re-fit all the points
  MarkerInitialization finalKinematicInit = fitter.completeBilevelResult(
      subsetTimesteps,
      newClip,
      bilevelFit,
      InitialMarkerFitParams()
          .setJointCentersAndWeights(
              reinit.joints,
              reinit.jointCenters,
              reinit.jointsAdjacentMarkers,
              reinit.jointWeights)
          .setJointAxisAndWeights(reinit.jointAxis, reinit.axisWeights)
          .setInitPoses(reinit.poses)
          .setDontRescaleBodies(true)
          .setGroupScales(bilevelFit->groupScales)
          .setMarkerOffsets(bilevelFit->markerOffsets));

  for (auto pair : finalKinematicInit.updatedMarkerMap)
  {
    assert(pair.second.first != nullptr);
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      finalKinematicInit.updatedMarkerMap,
      finalKinematicInit.poses,
      subsetTimesteps,
      anthropometrics);

  std::cout << "Initial error report:" << std::endl;
  initReport.printReport(5);
  std::cout << "After joint centers report:" << std::endl;
  afterJointCentersReport.printReport(5);
  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  s_t avgDiff = 0.0;
  for (int i = 0; i < goldPoses.cols(); i++)
  {
    Eigen::VectorXs goldPose = goldPoses.col(i);
    Eigen::VectorXs guessPose = finalKinematicInit.poses.col(i);
    avgDiff += (goldPose - guessPose).norm();
    if (i % 100 == 0)
    {
      std::cout << "**** Frame " << i << std::endl;
      for (int j = 0; j < standard.skeleton->getNumDofs(); j++)
      {
        std::cout << standard.skeleton->getDof(j)->getName() << ": "
                  << goldPose(j) << ", recovered " << guessPose(j) << std::endl;
      }
    }
  }
  avgDiff /= goldPoses.cols() * standard.skeleton->getNumDofs();
  std::cout << "Average joint error (rad): " << avgDiff << std::endl;
  std::cout << "Average joint error (deg): " << (avgDiff / M_PI) * 180
            << std::endl;

  /*
// Target markers
std::shared_ptr<server::GUIWebsocketServer> server
    = std::make_shared<server::GUIWebsocketServer>();
server->serve(8070);
fitter.debugTrajectoryAndMarkersToGUI(
    server, finalKinematicInit, subsetTimesteps);
server->blockWhileServing();
*/
}

#ifdef ALL_TESTS
TEST(MarkerFitter, RECOVER_SYNTHETIC_DATA_END_TO_END_CALIBRATION)
{
  evaluateOnSyntheticData(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim",
      "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml",
      "dart://sample/osim/Rajagopal2015_v3_scaled/Rajagopal_scaled.osim",
      "dart://sample/osim/Rajagopal2015_v3_scaled/S01DN603_ik.mot");
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, RECOVER_SYNTHETIC_DATA_END_TO_END_SPRINTING)
{
  evaluateOnSyntheticData(
      "dart://sample/osim/Sprinter/sprinter.osim",
      "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml",
      "dart://sample/osim/Sprinter/sprinter_scaled.osim",
      "dart://sample/osim/Sprinter/run0500cms.mot");
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, RECOVER_SYNTHETIC_DATA_END_TO_END_DJ2)
{
  evaluateOnSyntheticData(
      "dart://sample/osim/LaiArnoldSubject5/"
      "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim",
      "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml",
      "dart://sample/osim/LaiArnoldSubject5/"
      "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim",
      "dart://sample/osim/LaiArnoldSubject5/DJ2.mot");
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFitter, RECOVER_SYNTHETIC_DATA_END_TO_END_WALK)
{
  evaluateOnSyntheticData(
      "dart://sample/osim/LaiArnoldSubject6/"
      "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim",
      "dart://sample/osim/ANSUR/ANSUR_LaiArnold_metrics.xml",
      "dart://sample/osim/LaiArnoldSubject6/"
      "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim",
      "dart://sample/osim/LaiArnoldSubject6/walking1.mot");
}
#endif

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