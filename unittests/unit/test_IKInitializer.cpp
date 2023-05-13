#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dart/biomechanics/IKInitializer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/Skeleton.hpp"
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
    std::string openSimPath,
    std::vector<std::string> trcFiles,
    s_t heightM,
    bool saveToGUI = false)
{
  auto osim = OpenSimParser::parseOsim(openSimPath);
  osim.skeleton->zeroTranslationInCustomFunctions();
  osim.skeleton->autogroupSymmetricSuffixes();
  osim.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
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
  initializer.runFullPipeline(true);

  if (saveToGUI)
  {
    osim.skeleton->setGroupScales(initializer.mGroupScales);

    server::GUIRecording server;
    server.setFramesPerSecond(30);
    server.renderSkeleton(osim.skeleton);
    for (int t = 0; t < markerObservations.size(); t++)
    {
      osim.skeleton->setPositions(initializer.mPoses[t]);
      server.renderSkeleton(osim.skeleton);

      server.deleteObjectsByPrefix("line_");
      auto jointMap = initializer.getVisibleJointCenters(t);
      for (auto& pair : jointMap)
      {
        Eigen::Vector3s jointCenterEstimated = pair.second;
        std::map<std::string, s_t> jointToMarkerSquaredDistances
            = initializer.getJointToMarkerSquaredDistances(pair.first);
        for (std::string& markerName : initializer.getVisibleMarkerNames(t))
        {
          Eigen::Vector3s markerWorld = markerObservations[t][markerName];
          Eigen::Vector4s color(0.5, 0.5, 0.5, 1.0);
          if (jointToMarkerSquaredDistances.count(markerName))
          {
            color = Eigen::Vector4s(0, 0, 1, 1);
            std::vector<Eigen::Vector3s> markerTetherPoints;
            markerTetherPoints.push_back(markerWorld);
            markerTetherPoints.push_back(jointCenterEstimated);
            server.createLine(
                "line_" + pair.first + ":" + markerName,
                markerTetherPoints,
                color);
          }
        }
      }

      server.saveFrame();
    }

    server.writeFramesJson("../../../javascript/src/data/movement2.bin");
  }
}

bool verifyReconstructionOnSyntheticRandomPosesOsim(
    std::string openSimPath, bool saveToGUI = false)
{
  auto osim = OpenSimParser::parseOsim(openSimPath);
  s_t targetHeight = 1.8;
  s_t currentHeight = osim.skeleton->getHeight(
      Eigen::VectorXs::Zero(osim.skeleton->getNumDofs()));
  osim.skeleton->setBodyScales(
      Eigen::VectorXs::Ones(osim.skeleton->getNumBodyNodes() * 3)
      * (targetHeight / currentHeight));
  Eigen::VectorXs goldGroupScales = osim.skeleton->getGroupScales();

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::vector<Eigen::VectorXs> poses;
  std::vector<Eigen::VectorXs> jointCenters;
  for (int t = 0; t < 5; t++)
  {
    Eigen::VectorXs pose = osim.skeleton->getRandomPose();
    poses.push_back(pose);
    osim.skeleton->setPositions(pose);
    markerObservations.push_back(
        osim.skeleton->getMarkerMapWorldPositions(osim.markersMap));
    jointCenters.push_back(
        osim.skeleton->getJointWorldPositions(osim.skeleton->getJoints()));
  }

  // Reset to neutral scale
  osim.skeleton->setBodyScales(
      Eigen::VectorXs::Ones(osim.skeleton->getNumBodyNodes() * 3));

  server::GUIRecording server;
  server.setFramesPerSecond(10);
  server.renderSkeleton(osim.skeleton);
  std::shared_ptr<dynamics::Skeleton> recoveredSkel
      = osim.skeleton->cloneSkeleton();

  IKInitializer initializer(
      osim.skeleton, osim.markersMap, markerObservations, targetHeight);

  std::map<std::string, std::map<std::string, s_t>>
      originalJointToJointDistances;
  std::map<std::string, int> jointStackSize;
  for (auto& joint1 : initializer.mStackedJoints)
  {
    Eigen::Vector3s joint1Center = Eigen::Vector3s::Zero();
    for (auto* j : joint1->joints)
    {
      joint1Center
          += jointCenters[0].segment<3>(j->getJointIndexInSkeleton() * 3);
    }
    joint1Center /= joint1->joints.size();

    jointStackSize[joint1->name] = joint1->joints.size();

    for (auto& joint2 : initializer.mStackedJoints)
    {
      Eigen::Vector3s joint2Center = Eigen::Vector3s::Zero();
      for (auto* j : joint2->joints)
      {
        joint2Center
            += jointCenters[0].segment<3>(j->getJointIndexInSkeleton() * 3);
      }
      joint2Center /= joint2->joints.size();

      originalJointToJointDistances[joint1->name][joint2->name]
          = (joint1Center - joint2Center).norm();
    }
  }

  s_t markerError = 0.0;
  markerError = initializer.closedFormMDSJointCenterSolver(true);
  s_t pivotError = initializer.closedFormPivotFindingJointCenterSolver();
  std::cout << "Pivot error avg: " << pivotError << "m" << std::endl;
  initializer.reestimateDistancesFromJointCenters();
  markerError = initializer.closedFormMDSJointCenterSolver(true);
  s_t markerErrorFromIK
      = initializer.estimatePosesAndGroupScalesInClosedForm(true);
  (void)markerErrorFromIK;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    initializer.completeIKIteratively(t, osim.skeleton);
  }

  std::vector<Eigen::VectorXs> recoveredPoses = initializer.mPoses;
  Eigen::VectorXs groupScales = initializer.mGroupScales;
  std::vector<std::map<std::string, Eigen::Isometry3s>> estimatedBodyTransforms
      = initializer.mBodyTransforms;

  // Check the quality of joint to joint distances
  std::map<std::string, std::map<std::string, s_t>>
      estimatedJointToJointDistances
      = initializer.estimateJointToJointDistances();
  for (auto& pair1 : estimatedJointToJointDistances)
  {
    for (auto& pair2 : pair1.second)
    {
      s_t originalDist
          = originalJointToJointDistances[pair1.first][pair2.first];
      s_t estimatedDist = pair2.second;
      std::cout << "Joint to joint distance between " << pair1.first << " and "
                << pair2.first << ": estimated " << estimatedDist
                << ", original: " << originalDist
                << ", error: " << (originalDist - estimatedDist) << std::endl;
      s_t threshold = 1e-8;
      // If the joint isn't a perfectly rotational joint (either because it's
      // actually a compound joint, or because it's a knee), we have much looser
      // bounds on correctness, because the algorithms' core assumptions don't
      // hold.
      if (jointStackSize[pair1.first] > 1 || jointStackSize[pair2.first] > 1
          || pair1.first.find("walker_knee") != std::string::npos
          || pair2.first.find("walker_knee") != std::string::npos)
      {
        threshold = 0.02;
      }
      if (std::abs(originalDist - estimatedDist) > threshold)
      {
        std::cout << "Failed to reconstruct joint to joint distance between "
                  << pair1.first << " and " << pair2.first
                  << "! Error should be less than " << threshold << ", got "
                  << std::abs(originalDist - estimatedDist) << std::endl;
        return false;
      }
    }
  }

  if (recoveredPoses.size() > 0)
  {
    recoveredSkel->setGroupScales(groupScales);
  }
  osim.skeleton->setGroupScales(goldGroupScales);

  s_t avgJointCenterEstimateError = 0.0;
  int numJointsCenterEstimatesCounted = 0;
  s_t avgJointAngleError = 0.0;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    if (recoveredPoses.size() > t)
    {
      Eigen::VectorXs pose = poses[t];
      Eigen::VectorXs recoveredPose = recoveredPoses[t];
      recoveredSkel->setPositions(recoveredPose);
      server.renderSkeleton(
          recoveredSkel, "recovered", Eigen::Vector4s(1, 0, 0, 0.7));
      osim.skeleton->setPositions(pose);
      server.renderSkeleton(osim.skeleton);

      for (auto& pair : estimatedBodyTransforms[t])
      {
        dynamics::BodyNode* body = osim.skeleton->getBodyNode(pair.first);
        for (int i = 0; i < body->getNumShapeNodes(); i++)
        {
          dynamics::ShapeNode* shape = body->getShapeNode(i);
          if (shape->hasVisualAspect())
          {
            std::shared_ptr<dynamics::Shape> shapePtr = shape->getShape();
            if (shapePtr->getType() == MeshShape::getStaticType())
            {
              std::shared_ptr<dynamics::MeshShape> meshShape
                  = std::static_pointer_cast<dynamics::MeshShape>(shapePtr);
              Eigen::Isometry3s shapeTransform
                  = pair.second * shape->getRelativeTransform();

              server.createMeshFromShape(
                  body->getName() + "-" + std::to_string(i),
                  meshShape,
                  shapeTransform.translation(),
                  math::matrixToEulerXYZ(shapeTransform.linear()),
                  body->getScale(),
                  Eigen::Vector4s(0, 1, 0, 0.7));
              server.setObjectTooltip(
                  body->getName() + "-" + std::to_string(i),
                  "Estimated " + body->getName());
            }
          }
        }
      }

      for (int i = 0; i < osim.skeleton->getNumDofs(); i++)
      {
        std::cout << "Dof " << osim.skeleton->getDof(i)->getName()
                  << " error: " << pose(i) - recoveredPose(i) << std::endl;
      }
      avgJointAngleError += (pose - recoveredPose).norm();
    }
    auto jointMap = initializer.getVisibleJointCenters(t);
    for (auto& pair : jointMap)
    {
      Eigen::Vector3s jointWorld = jointCenters[t].segment<3>(
          osim.skeleton->getJoint(pair.first)->getJointIndexInSkeleton() * 3);
      Eigen::Vector3s jointCenterEstimated = pair.second;
      s_t jointError = (jointWorld - jointCenterEstimated).norm();
      std::cout << "Joint center " << pair.first << " error: " << jointError
                << std::endl;
      avgJointCenterEstimateError += jointError;
      numJointsCenterEstimatesCounted++;

      if (saveToGUI)
      {
        std::vector<Eigen::Vector3s> points;
        points.push_back(jointWorld);
        points.push_back(jointCenterEstimated);
        server.createLine(pair.first, points, Eigen::Vector4s(1, 0, 0, 1));
      }

      std::map<std::string, s_t> jointToMarkerSquaredDistances
          = initializer.getJointToMarkerSquaredDistances(pair.first);
      for (std::string& markerName : initializer.getVisibleMarkerNames(t))
      {
        Eigen::Vector3s markerWorld = markerObservations[t][markerName];
        Eigen::Vector4s color(0.5, 0.5, 0.5, 1.0);
        if (jointToMarkerSquaredDistances.count(markerName))
        {
          color = Eigen::Vector4s(0, 0, 1, 1);
          std::vector<Eigen::Vector3s> markerTetherPoints;
          markerTetherPoints.push_back(markerWorld);
          markerTetherPoints.push_back(jointCenterEstimated);
          server.createLine(
              pair.first + ":" + markerName, markerTetherPoints, color);
        }
      }
    }

    server.saveFrame();
  }
  avgJointAngleError /= recoveredPoses.size();

  if (numJointsCenterEstimatesCounted > 0)
  {
    avgJointCenterEstimateError /= numJointsCenterEstimatesCounted;
  }
  std::cout << "Marker error: " << markerError
            << "m, joint center estimate error: " << avgJointCenterEstimateError
            << "m" << std::endl;
  std::cout << "Joint angle error: " << avgJointAngleError << std::endl;
  if (avgJointCenterEstimateError > 0.05)
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
TEST(IKInitializer, POINT_CLOUD_TO_CLOUD_TRANSFORM)
{
  Eigen::Isometry3s T_wb = Eigen::Isometry3s::Identity();
  T_wb.translation() = Eigen::Vector3s::Random();
  T_wb.linear() = math::expMapRot(Eigen::Vector3s::Random());

  int numPoints = 5;

  std::vector<Eigen::Vector3s> points_local;
  for (int i = 0; i < numPoints; i++)
  {
    points_local.push_back(Eigen::Vector3s::Random());
  }

  std::vector<Eigen::Vector3s> points_world;
  for (int i = 0; i < numPoints; i++)
  {
    points_world.push_back(T_wb * points_local[i]);
  }

  Eigen::Isometry3s T_wb_recovered
      = IKInitializer::getPointCloudToPointCloudTransform(
          points_local, points_world, std::vector<s_t>(numPoints, 1.0));

  std::vector<Eigen::Vector3s> points_world_recovered;
  for (int i = 0; i < numPoints; i++)
  {
    points_world_recovered.push_back(T_wb_recovered * points_local[i]);
  }

  s_t error = 0.0;
  for (int i = 0; i < numPoints; i++)
  {
    error += (points_world[i] - points_world_recovered[i]).norm();
  }
  EXPECT_NEAR(error, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_SIMPLE_FORWARD_BASIS)
{
  for (int i = 0; i < 5; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();
    Eigen::Vector3s center = Eigen::Vector3s::Random();
    s_t radius = 1.0 + ((s_t)rand() / RAND_MAX);

    s_t cost = (point - center).squaredNorm() - (radius * radius);

    s_t a = 2.0;

    Eigen::Vector5s basis = Eigen::Vector5s(
        point.squaredNorm(), point(0), point(1), point(2), 1.0);
    Eigen::Vector5s u = Eigen::Vector5s(
        a,
        -2 * a * center(0),
        -2 * a * center(1),
        -2 * a * center(2),
        (center.squaredNorm() - radius * radius) * a);
    s_t recoveredCost = basis.dot(u) / a;

    Eigen::Vector3s recoveredCenter = u.segment<3>(1) / (-2.0 * a);
    EXPECT_TRUE(recoveredCenter.isApprox(center, 1e-8));
    if (!recoveredCenter.isApprox(center, 1e-8))
    {
      return;
    }
    s_t recoveredRadiusSquared = abs(u(4) / a - recoveredCenter.squaredNorm());
    EXPECT_NEAR(recoveredRadiusSquared, radius * radius, 1e-8);

    EXPECT_NEAR(cost, recoveredCost, 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_SIMPLE_REVERSE_BASIS)
{
  for (int i = 0; i < 5; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();

    Eigen::Vector5s u = Eigen::Vector5s::Random();

    Eigen::Vector3s center = u.segment<3>(1) / (-2.0 * u(0));
    s_t radiusSquared = abs((u(4) / u(0) - center.squaredNorm()));
    s_t recoveredCost = (point - center).squaredNorm() - radiusSquared;

    Eigen::Vector5s recoveredU = Eigen::Vector5s(
        1.0,
        -2 * center(0),
        -2 * center(1),
        -2 * center(2),
        center.squaredNorm() - radiusSquared);
    recoveredU *= u(0);
    EXPECT_TRUE(recoveredU.segment<3>(1).isApprox(u.segment<3>(1), 1e-8));
    EXPECT_NEAR(recoveredU(4), u(4), 1e-8);

    Eigen::Vector5s basis = Eigen::Vector5s(
        point.squaredNorm(), point(0), point(1), point(2), 1.0);
    s_t cost = basis.dot(u) / u(0);

    EXPECT_NEAR(cost, recoveredCost, 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, SPHERE_FIT_LEAST_SQUARES)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<Eigen::Vector3s> markerObservations;
  s_t radius = 2.0;
  markerObservations.push_back(center + Eigen::Vector3s::UnitX() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitX() * radius);
  markerObservations.push_back(center + Eigen::Vector3s::UnitY() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitY() * radius);
  markerObservations.push_back(center + Eigen::Vector3s::UnitZ() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitZ() * radius);

  Eigen::Vector3s center_raw
      = IKInitializer::leastSquaresSphereFit(markerObservations);
  s_t error_raw = (center_raw - center).norm();
  EXPECT_NEAR(error_raw, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_JOINT_TEST_NO_NOISE)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<Eigen::Vector3s> markerObservations;
  s_t radius = 2.0;
  markerObservations.push_back(center + Eigen::Vector3s::UnitX() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitX() * radius);
  markerObservations.push_back(center + Eigen::Vector3s::UnitY() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitY() * radius);
  markerObservations.push_back(center + Eigen::Vector3s::UnitZ() * radius);
  markerObservations.push_back(center - Eigen::Vector3s::UnitZ() * radius);

  Eigen::Vector3s center_recovered
      = IKInitializer::getChangPollard2006JointCenterSingleMarker(
          markerObservations);

  s_t error = (center_recovered - center).norm();
  EXPECT_NEAR(error, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_JOINT_TEST_WITH_NOISE)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<Eigen::Vector3s> markerObservations;
  s_t radius = 2.0;
  for (int i = 0; i < 500; i++)
  {
    Eigen::Matrix3s R = math::expMapRot(Eigen::Vector3s::Random());
    markerObservations.push_back(
        center + (R * Eigen::Vector3s::UnitX() * radius));
  }

  s_t noise = 0.01;
  for (int i = 0; i < markerObservations[i].size(); i++)
  {
    markerObservations[i] += Eigen::Vector3s::Random() * noise;
  }

  Eigen::Vector3s center_recovered
      = IKInitializer::getChangPollard2006JointCenterSingleMarker(
          markerObservations);

  s_t error = (center_recovered - center).norm();
  EXPECT_NEAR(error, 0.0, 1e-4);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, SPHERE_FIT_MULTI_JOINT_LEAST_SQUARES)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<Eigen::Vector3s> markerObservations1;
  s_t radius1 = 2.0;
  markerObservations1.push_back(center + Eigen::Vector3s::UnitX() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitX() * radius1);
  markerObservations1.push_back(center + Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center + Eigen::Vector3s::UnitZ() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitZ() * radius1);
  std::vector<Eigen::Vector3s> markerObservations2;
  s_t radius2 = 3.0;
  markerObservations2.push_back(center + Eigen::Vector3s::UnitX() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitX() * radius2);
  markerObservations2.push_back(center + Eigen::Vector3s::UnitY() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitY() * radius2);
  markerObservations2.push_back(center + Eigen::Vector3s::UnitZ() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitZ() * radius2);

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  markerTraces.push_back(markerObservations1);
  markerTraces.push_back(markerObservations2);

  Eigen::Vector3s center_raw
      = IKInitializer::leastSquaresConcentricSphereFit(markerTraces);
  s_t error_raw = (center_raw - center).norm();
  EXPECT_NEAR(error_raw, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_MULTI_MARKER_NO_NOISE)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<Eigen::Vector3s> markerObservations1;
  s_t radius1 = 2.0;
  markerObservations1.push_back(center + Eigen::Vector3s::UnitX() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitX() * radius1);
  markerObservations1.push_back(center + Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center + Eigen::Vector3s::UnitZ() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitZ() * radius1);
  std::vector<Eigen::Vector3s> markerObservations2;
  s_t radius2 = 3.0;
  markerObservations2.push_back(center + Eigen::Vector3s::UnitX() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitX() * radius2);
  markerObservations2.push_back(center + Eigen::Vector3s::UnitY() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitY() * radius2);
  markerObservations2.push_back(center + Eigen::Vector3s::UnitZ() * radius2);
  markerObservations2.push_back(center - Eigen::Vector3s::UnitZ() * radius2);

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  markerTraces.push_back(markerObservations1);
  markerTraces.push_back(markerObservations2);

  Eigen::Vector3s center_raw
      = IKInitializer::getChangPollard2006JointCenterMultiMarker(markerTraces);
  s_t error_raw = (center_raw - center).norm();
  EXPECT_NEAR(error_raw, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_MULTI_MARKER_TEST_WITH_NOISE)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX();
  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  for (int i = 0; i < 2; i++)
  {
    std::vector<Eigen::Vector3s> markerObservations;
    s_t radius = 1.0 * i;
    for (int i = 0; i < 500; i++)
    {
      Eigen::Matrix3s R = math::expMapRot(Eigen::Vector3s::Random());
      markerObservations.push_back(
          center + (R * Eigen::Vector3s::UnitX() * radius));
    }

    s_t noise = 0.01;
    for (int i = 0; i < markerObservations[i].size(); i++)
    {
      markerObservations[i] += Eigen::Vector3s::Random() * noise;
    }
    markerTraces.push_back(markerObservations);
  }

  Eigen::Vector3s center_recovered
      = IKInitializer::getChangPollard2006JointCenterMultiMarker(markerTraces);

  s_t error = (center_recovered - center).norm();
  EXPECT_NEAR(error, 0.0, 2e-4);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, SYNTHETIC_OSIM)
{
  EXPECT_TRUE(verifyReconstructionOnSyntheticRandomPosesOsim(
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

/*
TEST(IKInitializer, VISUALIZE_RESULTS)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/grf/subject18_synthetic/trials/walk2/markers.trc");

  runOnRealOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.68,
      // 1.775,
      true);
}
*/