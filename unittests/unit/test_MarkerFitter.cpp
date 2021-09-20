#include <gtest/gtest.h>

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

#define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

bool testFitterGradients(
    MarkerFitter& fitter,
    std::shared_ptr<dynamics::Skeleton>& skel,
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers,
    const std::vector<std::pair<int, Eigen::Vector3s>>& observedMarkers)
{
  const s_t THRESHOLD = 2e-7;

  Eigen::VectorXs gradWrtJoints = fitter.getLossGradientWrtJoints(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtJoints_fd
      = fitter.finiteDifferenceLossGradientWrtJoints(
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

  Eigen::VectorXs gradWrtScales = fitter.getLossGradientWrtGroupScales(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtScales_fd
      = fitter.finiteDifferenceLossGradientWrtGroupScales(
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

  Eigen::VectorXs gradWrtMarkerOffsets = fitter.getLossGradientWrtMarkerOffsets(
      skel, markers, fitter.getMarkerError(skel, markers, observedMarkers));
  Eigen::VectorXs gradWrtMarkerOffsets_fd
      = fitter.finiteDifferenceLossGradientWrtMarkerOffsets(
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
      = fitter.getLossGradientWrtJointsJacobianWrtJoints(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtJoints_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtJoints(
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
      = fitter.getLossGradientWrtJointsJacobianWrtGroupScales(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtGroupScales_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtGroupScales(
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
      = fitter.getLossGradientWrtJointsJacobianWrtMarkerOffsets(
          skel,
          markers,
          fitter.getMarkerError(skel, markers, observedMarkers),
          sparsityMap);
  Eigen::MatrixXs gradWrtJointsJacWrtMarkerOffsets_fd
      = fitter.finiteDifferenceLossGradientWrtJointsJacobianWrtMarkerOffsets(
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
    const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
        markers)
{
  const s_t THRESHOLD = 5e-8;

  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  srand(42);

  // 1. Generate a bunch of marker data for the skeleton in random
  // configurations
  Eigen::VectorXs goldGroupScales
      = originalGroupScales
        + Eigen::VectorXs::Random(originalGroupScales.size()) * 0.1;
  Eigen::VectorXs goldMarkerOffsets
      = Eigen::VectorXs::Random(markers.size() * 3) * 0.05;
  std::vector<Eigen::VectorXs> goldPoses;
  std::vector<std::vector<std::pair<int, Eigen::Vector3s>>> observations;

  for (int i = 0; i < numPoses; i++)
  {
    Eigen::VectorXs goldPose = Eigen::VectorXs::Random(skel->getNumDofs());
    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
        = fitter.setConfiguration(
            skel, goldPose, goldGroupScales, goldMarkerOffsets);
    Eigen::VectorXs markerWorldPoses = skel->getMarkerWorldPositions(markers);
    // Take an observation
    std::vector<std::pair<int, Eigen::Vector3s>> obs;
    for (int j = 0; j < markers.size(); j++)
    {
      if (((double)rand() / RAND_MAX) > markerDropProb)
      {
        obs.emplace_back(
            j, Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3)));
      }
    }
    observations.push_back(obs);
  }

  // 2. Reset the skeleton configuration
  skel->setPositions(Eigen::VectorXs::Zero(skel->getNumDofs()));
  skel->setGroupScales(originalGroupScales);

  // 3. Create a BilevelFitProblem
  std::shared_ptr<MarkerFitResult> tmpResult
      = std::make_shared<MarkerFitResult>();
  BilevelFitProblem problem(&fitter, observations, tmpResult);

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
    return false;
  }

  Eigen::MatrixXs jac = problem.getConstraintsJacobian(x);
  Eigen::MatrixXs jac_fd = problem.finiteDifferenceConstraintsJacobian(x);

  if (!equals(jac, jac_fd, THRESHOLD))
  {
    Eigen::MatrixXs jacScales
        = jac.block(0, 0, jac.rows(), skel->getNumScaleGroups() * 3);
    Eigen::MatrixXs jacScales_fd
        = jac_fd.block(0, 0, jac.rows(), skel->getNumScaleGroups() * 3);
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
        0, skel->getNumScaleGroups() * 3, jac.rows(), markers.size() * 3);
    Eigen::MatrixXs jacMarkers_fd = jac_fd.block(
        0, skel->getNumScaleGroups() * 3, jac.rows(), markers.size() * 3);
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

    int offset = (skel->getNumScaleGroups() * 3) + (markers.size() * 3);
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
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitX() * 0.05);
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitY() * 0.05);
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitZ() * 0.05);
  }

  MarkerFitter fitter(skel, markers);

  srand(42);

  // 1. Generate a bunch of marker data for the skeleton in random
  // configurations
  Eigen::VectorXs goldGroupScales
      = originalGroupScales
        + Eigen::VectorXs::Random(originalGroupScales.size()) * bodyScaleBounds;
  Eigen::VectorXs goldMarkerOffsets
      = Eigen::VectorXs::Random(markers.size() * 3) * markerErrorBounds;
  std::vector<Eigen::VectorXs> goldPoses;
  std::vector<std::vector<std::pair<int, Eigen::Vector3s>>> observations;

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

    std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers
        = fitter.setConfiguration(
            skel, goldPose, goldGroupScales, goldMarkerOffsets);
    Eigen::VectorXs markerWorldPoses = skel->getMarkerWorldPositions(markers);
    // Take an observation
    std::vector<std::pair<int, Eigen::Vector3s>> obs;
    for (int j = 0; j < markers.size(); j++)
    {
      if (((double)rand() / RAND_MAX) > markerDropProb)
      {
        obs.emplace_back(
            j, Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3)));
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
  std::shared_ptr<MarkerFitResult> tmpResult
      = std::make_shared<MarkerFitResult>();
  BilevelFitProblem problem(&fitter, observations, tmpResult);

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
  std::shared_ptr<MarkerFitResult> result = fitter.optimize(observations);

  Eigen::VectorXs groupScaleError = result->groupScales - goldGroupScales;
  Eigen::MatrixXs groupScaleCols = Eigen::MatrixXs(groupScaleError.size(), 3);
  groupScaleCols.col(0) = goldGroupScales;
  groupScaleCols.col(1) = result->groupScales;
  groupScaleCols.col(2) = groupScaleError;

  Eigen::VectorXs markerOffsetError = result->markerOffsets - goldMarkerOffsets;
  Eigen::MatrixXs markerOffsetCols
      = Eigen::MatrixXs(markerOffsetError.size(), 3);
  markerOffsetCols.col(0) = goldMarkerOffsets;
  markerOffsetCols.col(1) = result->markerOffsets;
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
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitX() * 0.05);
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitY() * 0.05);
    markers.emplace_back(skel->getBodyNode(i), Eigen::Vector3s::UnitZ() * 0.05);
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

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      adjustedMarkersSkel = fitter.setConfiguration(
          skel, goldPose, goldGroupScales, goldMarkerOffsets);

  // std::shared_ptr<dynamics::Skeleton> goldTarget = skel->clone();
  // goldTarget->setPositions(goldPose);
  server.renderSkeleton(skel, "goldSkel");

  std::shared_ptr<dynamics::Skeleton> skelBallJoints
      = skel->convertSkeletonToBallJoints();
  skelBallJoints->setPositions(
      skel->convertPositionsToBallSpace(skel->getPositions()));

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      adjustedMarkers;
  for (auto pair : adjustedMarkersSkel)
  {
    adjustedMarkers.emplace_back(
        skelBallJoints->getBodyNode(pair.first->getName()),
        Eigen::Vector3s(pair.second));
  }

  Eigen::VectorXs markerWorldPoses
      = skelBallJoints->getMarkerWorldPositions(adjustedMarkers);
  // Take an observation
  std::vector<std::pair<int, Eigen::Vector3s>> obs;
  for (int j = 0; j < adjustedMarkers.size(); j++)
  {
    if (((double)rand() / RAND_MAX) > markerDropProb)
    {
      obs.emplace_back(j, Eigen::Vector3s(markerWorldPoses.segment<3>(j * 3)));
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
        skelBallJoints->getNumDofs() + skelBallJoints->getNumScaleGroups() * 3);
    /*
    initialPos.segment(0, skelBallJoints->getNumDofs())
        = skelBallJoints->getPositions();
    initialPos.segment(
        skelBallJoints->getNumDofs(), skelBallJoints->getNumScaleGroups() * 3)
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
              skelBallJoints->getNumDofs(),
              skelBallJoints->getNumScaleGroups() * 3);
          for (int i = 0; i < skelBallJoints->getNumScaleGroups(); i++)
          {
            for (int axis = 0; axis < 3; axis++)
            {
              if (newScales(i * 3 + axis)
                  > skelBallJoints->getScaleGroupUpperBound(i)(axis))
              {
                newScales(i * 3 + axis)
                    = skelBallJoints->getScaleGroupUpperBound(i)(axis);
              }
              if (newScales(i * 3 + axis)
                  < skelBallJoints->getScaleGroupLowerBound(i)(axis))
              {
                newScales(i * 3 + axis)
                    = skelBallJoints->getScaleGroupLowerBound(i)(axis);
              }
            }
          }
          skel->setGroupScales(newScales);
          skelBallJoints->setGroupScales(newScales);

          // Return the clamped position
          Eigen::VectorXs clampedPos = Eigen::VectorXs::Zero(pos.size());
          clampedPos.segment(0, skelBallJoints->getNumDofs())
              = skelBallJoints->getPositions();
          clampedPos.segment(
              skelBallJoints->getNumDofs(),
              skelBallJoints->getNumScaleGroups() * 3)
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
                     + skelBallJoints->getNumScaleGroups() * 3);
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
              skelBallJoints->getNumScaleGroups() * 3)
              = skelBallJoints->getMarkerWorldPositionsJacobianWrtGroupScales(
                  adjustedMarkers);
        },
        [skel, skelBallJoints](Eigen::VectorXs& val) {
          val.segment(0, skelBallJoints->getNumDofs())
              = skel->convertPositionsToBallSpace(skel->getRandomPose());
          val.segment(
                 skelBallJoints->getNumDofs(),
                 skelBallJoints->getNumScaleGroups() * 3)
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

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(
      std::make_pair(osim->getBodyNode("radius_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("radius_r"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("tibia_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("tibia_r"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("ulna_l"), Eigen::Vector3s::Random()));
  markers.push_back(
      std::make_pair(osim->getBodyNode("ulna_r"), Eigen::Vector3s::Random()));

  MarkerFitter fitter(osim, markers);

  std::vector<std::pair<int, Eigen::Vector3s>> observedMarkers;
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(0, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(1, Eigen::Vector3s::Random()));
  // Skip 2
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(3, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(4, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(5, Eigen::Vector3s::Random()));

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

  EXPECT_TRUE(testFitterGradients(fitter, osim, markers, observedMarkers));

  EXPECT_TRUE(testBilevelFitProblemGradients(fitter, 3, 0.02, osim, markers));

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

  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> markers;
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("radius_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("radius_r"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("tibia_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("tibia_r"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("ulna_l"), Eigen::Vector3s::Random()));
  markers.push_back(std::make_pair(
      osimBallJoints->getBodyNode("ulna_r"), Eigen::Vector3s::Random()));

  MarkerFitter fitter(osimBallJoints, markers);

  std::vector<std::pair<int, Eigen::Vector3s>> observedMarkers;
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(0, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(1, Eigen::Vector3s::Random()));
  // Skip 2
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(3, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(4, Eigen::Vector3s::Random()));
  observedMarkers.push_back(
      std::make_pair<int, Eigen::Vector3s>(5, Eigen::Vector3s::Random()));

  EXPECT_TRUE(
      testFitterGradients(fitter, osimBallJoints, markers, observedMarkers));
  EXPECT_TRUE(
      testBilevelFitProblemGradients(fitter, 3, 0.02, osimBallJoints, markers));
}
#endif