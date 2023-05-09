#include <string>

#include <gtest/gtest.h>

#include "dart/biomechanics/IKInitializer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;

void runOnRealOsim(
    std::string openSimPath, std::vector<std::string> trcFiles, s_t heightM)
{
  auto osim = OpenSimParser::parseOsim(openSimPath);
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  for (std::string trcFile : trcFiles)
  {
    auto trc = OpenSimParser::loadTRC(trcFile);
    // Add the marker observations to our list
    for (auto& obs : trc.markerTimesteps)
    {
      markerObservations.push_back(obs);
    }
  }
  (void)heightM;

  IKInitializer initializer(
      osim.skeleton, osim.markersMap, markerObservations, heightM);
  s_t error = initializer.islandJointCenterSolver();
  std::cout << "Marker error on joint island solver: " << error << std::endl;
}

bool runOnSyntheticOsim(std::string openSimPath, bool saveToGUI = false)
{
  auto osim = OpenSimParser::parseOsim(openSimPath);
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::vector<Eigen::VectorXs> poses;
  std::vector<Eigen::VectorXs> jointCenters;
  for (int t = 0; t < 1; t++)
  {
    Eigen::VectorXs pose = osim.skeleton->getRandomPose();
    poses.push_back(pose);
    osim.skeleton->setPositions(pose);
    markerObservations.push_back(
        osim.skeleton->getMarkerMapWorldPositions(osim.markersMap));
    jointCenters.push_back(
        osim.skeleton->getJointWorldPositions(osim.skeleton->getJoints()));
  }

  server::GUIRecording server;
  server.setFramesPerSecond(10);
  server.renderSkeleton(osim.skeleton);

  IKInitializer initializer(osim.skeleton, osim.markersMap, markerObservations);

  s_t markerError = initializer.islandJointCenterSolver();

  s_t avgJointError = 0.0;
  int numJointsCounted = 0;
  for (int t = 0; t < jointCenters.size(); t++)
  {
    auto jointMap = initializer.getVisibleJointCenters(t);
    for (auto& pair : jointMap)
    {
      Eigen::Vector3s jointWorld = jointCenters[t].segment<3>(
          osim.skeleton->getJoint(pair.first)->getJointIndexInSkeleton() * 3);
      Eigen::Vector3s jointRecovered = pair.second;
      s_t jointError = (jointWorld - jointRecovered).norm();
      // std::cout << "Joint " << pair.first << " error: " << jointError
      //           << std::endl;
      avgJointError += jointError;
      numJointsCounted++;

      if (saveToGUI)
      {
        std::vector<Eigen::Vector3s> points;
        points.push_back(jointWorld);
        points.push_back(jointRecovered);
        server.createLine(pair.first, points, Eigen::Vector4s(1, 0, 0, 1));
      }

      std::map<std::string, s_t> jointToMarkerSquaredDistances
          = initializer.getJointToMarkerSquaredDistances(pair.first);
      for (std::string& markerName : initializer.getVisibleMarkerNames(t))
      {
        Eigen::Vector3s markerWorld = markerObservations[t][markerName];
        // s_t markerSquaredDist = (markerWorld -
        // jointRecovered).squaredNorm();
        Eigen::Vector4s color(0.5, 0.5, 0.5, 1.0);
        if (jointToMarkerSquaredDistances.count(markerName))
        {
          color = Eigen::Vector4s(0, 0, 1, 1);
          // s_t goalSquaredDist =
          // jointToMarkerSquaredDistances.at(markerName); std::cout << "
          // Marker " << markerName
          //           << " goal: " << goalSquaredDist
          //           << ", actual: " << markerSquaredDist << std::endl;
          std::vector<Eigen::Vector3s> markerTetherPoints;
          markerTetherPoints.push_back(markerWorld);
          markerTetherPoints.push_back(jointRecovered);
          server.createLine(
              pair.first + ":" + markerName, markerTetherPoints, color);
        }
      }
    }
  }
  server.saveFrame();
  if (numJointsCounted > 0)
  {
    avgJointError /= numJointsCounted;
  }
  std::cout << "Marker error: " << markerError
            << "m, joint error: " << avgJointError << "m" << std::endl;
  if (avgJointError > 0.05)
  {
    std::cout << "Joint error is too high on synthetic data!" << std::endl;
    return false;
  }

  if (saveToGUI)
  {
    server.writeFramesJson("../../../javascript/src/data/movement2.bin");
  }

  return true;
}

bool verifyMarkerReconstructionOnOsim(
    std::string openSimPath, std::vector<std::string> trcFiles, s_t heightM)
{
  (void)heightM;
  auto osim = OpenSimParser::parseOsim(openSimPath);
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  for (std::string trcFile : trcFiles)
  {
    auto trc = OpenSimParser::loadTRC(trcFile);
    // Add the marker observations to our list
    for (auto& obs : trc.markerTimesteps)
    {
      markerObservations.push_back(obs);
    }
  }

  IKInitializer initializer(
      osim.skeleton, osim.markersMap, markerObservations, heightM);
  for (int t = 0; t < markerObservations.size(); t++)
  {
    std::vector<std::string> markerNames = initializer.getVisibleMarkerNames(t);
    Eigen::MatrixXs D
        = Eigen::MatrixXs::Zero(markerNames.size(), markerNames.size());
    for (int i = 0; i < markerNames.size(); i++)
    {
      for (int j = 0; j < markerNames.size(); j++)
      {
        D(i, j) = (markerObservations[t][markerNames[i]]
                   - markerObservations[t][markerNames[j]])
                      .squaredNorm();
      }
    }
    Eigen::MatrixXs D_rank5 = IKInitializer::rankNDistanceMatrix(D, 5);
    Eigen::MatrixXs P = IKInitializer::getPointCloudFromDistanceMatrix(D);
    for (int i = 0; i < markerNames.size(); i++)
    {
      for (int j = 0; j < markerNames.size(); j++)
      {
        s_t recoveredDist = (P.col(i) - P.col(j)).squaredNorm();
        s_t originalDist = D_rank5(i, j);
        if (std::abs(recoveredDist - originalDist) > 1e-6)
        {
          std::cout << "Failed to reconstruct distance in P between "
                    << markerNames[i] << " and " << markerNames[j] << ", got "
                    << recoveredDist << " but should be " << originalDist
                    << std::endl;
          return false;
        }
      }
    }
    std::vector<Eigen::Vector3s> markerPoints;
    for (std::string markerName : markerNames)
    {
      markerPoints.push_back(markerObservations[t][markerName]);
    }
    Eigen::MatrixXs mapped
        = IKInitializer::mapPointCloudToData(P, markerPoints);
    for (int i = 0; i < markerPoints.size(); i++)
    {
      s_t reconstructionError = (mapped.col(i) - markerPoints[i]).norm();
      if (reconstructionError > 1e-6)
      {
        std::cout << "Failed to reconstruct marker \"" << markerNames[i]
                  << "\", got error " << reconstructionError << std::endl;
        return false;
      }
    }
  }
  return true;
}

#ifdef ALL_TESTS
TEST(IKInitializer, RECONSTRUCT_EXAMPLE_CLOUD)
{
  std::vector<Eigen::Vector3s> points;
  for (int i = 0; i < 10; i++)
  {
    points.push_back(Eigen::Vector3s::Random());
  }
  Eigen::MatrixXs D = Eigen::MatrixXs(points.size(), points.size());
  for (int i = 0; i < points.size(); i++)
  {
    for (int j = 0; j < points.size(); j++)
    {
      D(i, j) = (points[i] - points[j]).squaredNorm();
    }
  }
  Eigen::MatrixXs D_rank5 = IKInitializer::rankNDistanceMatrix(D, 5);
  Eigen::MatrixXs P = IKInitializer::getPointCloudFromDistanceMatrix(D_rank5);

  for (int i = 0; i < points.size(); i++)
  {
    for (int j = 0; j < points.size(); j++)
    {
      s_t recoveredDist = (P.col(i) - P.col(j)).squaredNorm();
      s_t originalDist = (points[i] - points[j]).squaredNorm();
      EXPECT_NEAR(recoveredDist, originalDist, 1e-6);
    }
  }

  std::vector<Eigen::Vector3s> first3Points;
  first3Points.push_back(points[0]);
  first3Points.push_back(points[1]);
  first3Points.push_back(points[2]);

  Eigen::MatrixXs mapped = IKInitializer::mapPointCloudToData(P, first3Points);
  for (int i = 0; i < points.size(); i++)
  {
    s_t reconstructionError = (mapped.col(i) - points[i]).norm();
    std::cout << "Reconstruction error[" << i << "]: " << reconstructionError
              << std::endl;
    EXPECT_NEAR(reconstructionError, 0.0, 1e-6);
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, SYNTHETIC_OSIM)
{
  EXPECT_TRUE(runOnSyntheticOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim",
      true));
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, MARKER_RECONSTRUCTION)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/grf/subject18_synthetic/trials/walk2/markers.trc");

  EXPECT_TRUE(verifyMarkerReconstructionOnOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.775));
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, EVAL_PERFORMANCE)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/grf/subject18_synthetic/trials/walk2/markers.trc");

  runOnRealOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.775);
}
#endif