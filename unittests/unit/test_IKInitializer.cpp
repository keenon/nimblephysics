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
#include "dart/math/Geometry.hpp"
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
  std::vector<bool> newClip;
  for (std::string trcFile : trcFiles)
  {
    auto trc = OpenSimParser::loadTRC(trcFile);
    // Add the marker observations to our list
    for (int i = 0; i < trc.markerTimesteps.size(); i++)
    {
      auto& obs = trc.markerTimesteps[i];
      markerObservations.push_back(obs);
      newClip.push_back(i == 0);
    }
  }

  std::map<std::string, bool> markerIsAnatomical;
  for (auto& pair : osim.markersMap)
  {
    markerIsAnatomical[pair.first] = false;
  }
  for (std::string& marker : osim.anatomicalMarkers)
  {
    markerIsAnatomical[marker] = true;
  }

  IKInitializer initializer(
      osim.skeleton,
      osim.markersMap,
      markerIsAnatomical,
      markerObservations,
      newClip,
      heightM);
  initializer.runFullPipeline(true);

  if (saveToGUI)
  {
    osim.skeleton->setGroupScales(initializer.getGroupScales());

    server::GUIRecording server;
    server.setFramesPerSecond(30);
    server.renderSkeleton(osim.skeleton);

    std::vector<std::map<std::string, Eigen::Isometry3s>>
        estimatedBodyTransforms = initializer.getBodyTransforms();
    std::map<std::string, bool> createdBodies;
    for (int t = 0; t < markerObservations.size(); t++)
    {
      osim.skeleton->setPositions(initializer.getPoses()[t]);
      server.renderSkeleton(osim.skeleton);

      if (t < estimatedBodyTransforms.size())
      {
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

                std::string shapeName
                    = body->getName() + "-" + std::to_string(i);
                if (createdBodies.count(shapeName) == 0)
                {
                  server.createMeshFromShape(
                      shapeName,
                      meshShape,
                      shapeTransform.translation(),
                      math::matrixToEulerXYZ(shapeTransform.linear()),
                      body->getScale(),
                      Eigen::Vector4s(0, 1, 0, 0.7));
                  server.setObjectTooltip(
                      shapeName, "Estimated " + body->getName());
                  createdBodies[shapeName] = true;
                }
                else
                {
                  server.setObjectPosition(
                      shapeName, shapeTransform.translation());
                  server.setObjectRotation(
                      shapeName,
                      math::matrixToEulerXYZ(shapeTransform.linear()));
                }
              }
            }
          }
        }
      }

      server.deleteObjectsByPrefix("line_");
      auto jointMap = initializer.getJointsAttachedToObservedMarkersCenters(t);
      for (auto& pair : jointMap)
      {
        Eigen::Vector3s jointCenterEstimated = pair.second;
        std::map<std::string, s_t> jointToMarkerSquaredDistances
            = initializer.getJointToMarkerSquaredDistances(pair.first);
        for (std::string& markerName : initializer.getObservedMarkerNames(t))
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
    std::string openSimPath,
    bool saveToGUI = false,
    bool givePerfectScaleInfoToInitializer = false)
{
  srand(42);

  auto osim = OpenSimParser::parseOsim(openSimPath);
  s_t targetHeight = 1.8;
  s_t currentHeight = osim.skeleton->getHeight(
      Eigen::VectorXs::Zero(osim.skeleton->getNumDofs()));
  osim.skeleton->setBodyScales(
      Eigen::VectorXs::Ones(osim.skeleton->getNumBodyNodes() * 3)
      * (targetHeight / currentHeight));
  Eigen::VectorXs goldGroupScales = osim.skeleton->getGroupScales();

  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::vector<bool> newClip;
  std::vector<Eigen::VectorXs> poses;
  std::vector<Eigen::VectorXs> jointCenters;
  for (int t = 0; t < 10; t++)
  {
    Eigen::VectorXs pose
        = t < 2 ? Eigen::VectorXs::Zero(osim.skeleton->getNumDofs())
                : osim.skeleton->getRandomPose();
    poses.push_back(pose);
    osim.skeleton->setPositions(pose);
    markerObservations.push_back(
        osim.skeleton->getMarkerMapWorldPositions(osim.markersMap));
    jointCenters.push_back(
        osim.skeleton->getJointWorldPositions(osim.skeleton->getJoints()));
    // There's no continuity between any of our random poses, so we always need
    // to completely restart the IK from timestep to timestep
    newClip.push_back(true);
  }

  // Reset to neutral scale, unless we're testing the IKInitializer with perfect
  // scaling information already given.
  if (!givePerfectScaleInfoToInitializer)
  {
    osim.skeleton->setBodyScales(
        Eigen::VectorXs::Ones(osim.skeleton->getNumBodyNodes() * 3));
  }

  server::GUIRecording server;
  server.setFramesPerSecond(10);
  server.renderSkeleton(osim.skeleton);
  std::shared_ptr<dynamics::Skeleton> recoveredSkel
      = osim.skeleton->cloneSkeleton();

  std::map<std::string, bool> markerIsAnatomical;
  for (auto& pair : osim.markersMap)
  {
    markerIsAnatomical[pair.first] = false;
  }
  for (std::string& marker : osim.anatomicalMarkers)
  {
    markerIsAnatomical[marker] = true;
  }

  IKInitializer initializer(
      osim.skeleton,
      osim.markersMap,
      markerIsAnatomical,
      markerObservations,
      newClip,
      targetHeight);

  std::map<std::string, std::map<std::string, s_t>>
      originalJointToJointDistances;
  std::map<std::string, int> jointStackSize;
  for (auto& joint1 : initializer.getStackedJoints())
  {
    Eigen::Vector3s joint1Center = Eigen::Vector3s::Zero();
    for (auto* j : joint1->joints)
    {
      joint1Center
          += jointCenters[0].segment<3>(j->getJointIndexInSkeleton() * 3);
    }
    joint1Center /= joint1->joints.size();

    jointStackSize[joint1->name] = joint1->joints.size();

    for (auto& joint2 : initializer.getStackedJoints())
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
  s_t pivotError = initializer.closedFormPivotFindingJointCenterSolver(true);
  initializer.recenterAxisJointsBasedOnBoneAngles(true);
  std::cout << "Pivot error avg: " << pivotError << "m" << std::endl;

  // initializer.reestimateDistancesFromJointCenters();
  // markerError = initializer.closedFormMDSJointCenterSolver(true);
  initializer.estimateGroupScalesClosedForm(false);
  s_t markerErrorFromIK = initializer.estimatePosesWithIK(false);
  (void)markerErrorFromIK;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    // initializer.completeIKIteratively(t, osim.skeleton);
    // initializer.fineTuneIKIteratively(t, osim.skeleton);
  }

  std::vector<Eigen::VectorXs> recoveredPoses = initializer.getPoses();
  Eigen::VectorXs groupScales = initializer.getGroupScales();
  std::vector<std::map<std::string, Eigen::Isometry3s>> estimatedBodyTransforms
      = initializer.getBodyTransforms();

  // Check the quality of joint to joint distances
  std::map<std::string, std::map<std::string, s_t>>
      estimatedJointToJointDistances
      = initializer.estimateJointToJointDistances();
  bool jointFailure = false;
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
      s_t threshold = 1e-5;
      // If the joint isn't a perfectly rotational joint (either because it's
      // actually a compound joint, or because it's a knee), we have much looser
      // bounds on correctness, because the algorithms' core assumptions don't
      // hold.
      if (jointStackSize[pair1.first] > 1 || jointStackSize[pair2.first] > 1
          || pair1.first.find("walker_knee") != std::string::npos
          || pair2.first.find("walker_knee") != std::string::npos)
      {
        threshold = 0.04;
      }
      if (std::abs(originalDist - estimatedDist) > threshold)
      {
        std::cout << "Failed to reconstruct joint to joint distance between "
                  << pair1.first << " and " << pair2.first
                  << "! Error should be less than " << threshold << ", got "
                  << std::abs(originalDist - estimatedDist) << std::endl;
        jointFailure = true;
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

      Eigen::VectorXs jointWorldCenters
          = osim.skeleton->getJointWorldPositions(osim.skeleton->getJoints());
      for (int j = 0; j < osim.skeleton->getNumJoints(); j++)
      {
        dynamics::Joint* joint = osim.skeleton->getJoint(j);
        if (joint->getNumDofs() == 1)
        {
          Eigen::Vector3s jointWorld = jointWorldCenters.segment<3>(
              joint->getJointIndexInSkeleton() * 3);
          Eigen::Vector3s axis
              = joint->getWorldAxisScrewForVelocity(0).head<3>();
          std::vector<Eigen::Vector3s> jointAxisLine;
          jointAxisLine.push_back(jointWorld);
          jointAxisLine.push_back(jointWorld + axis * 0.5);
          server.createLine(
              "joint_" + std::to_string(j),
              jointAxisLine,
              Eigen::Vector4s(1, 0, 0, 1));
        }
      }

      if (initializer.getJointAxisDirs().size() > t)
      {
        for (auto& pair : initializer.getJointAxisDirs()[t])
        {
          if (initializer.getJointAxisDirs()[t].count(pair.first))
          {
            std::vector<Eigen::Vector3s> guessedAxisLine;
            guessedAxisLine.push_back(pair.second);
            guessedAxisLine.push_back(
                pair.second
                + initializer.getJointAxisDirs()[t][pair.first] * 0.5);
            server.createLine(
                "guessed_joint" + pair.first,
                guessedAxisLine,
                Eigen::Vector4s(0, 1, 1, 1));
          }
        }
      }

      if (estimatedBodyTransforms.size() > t)
      {
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
      }

      for (int i = 0; i < osim.skeleton->getNumDofs(); i++)
      {
        std::cout << "Dof " << osim.skeleton->getDof(i)->getName()
                  << " error: " << pose(i) - recoveredPose(i) << std::endl;
      }
      avgJointAngleError += (pose - recoveredPose).norm();
    }
    auto jointMap = initializer.getJointsAttachedToObservedMarkersCenters(t);
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
        server.createSphere(
            "joint_" + pair.first,
            0.01,
            jointCenterEstimated,
            Eigen::Vector4s(0.5, 0.5, 0, 1));
        server.setObjectTooltip(
            "joint_" + pair.first, pair.first + " Estimated");

        std::vector<Eigen::Vector3s> points;
        points.push_back(jointWorld);
        points.push_back(jointCenterEstimated);
        server.createLine(pair.first, points, Eigen::Vector4s(1, 0, 0, 1));
      }

      std::map<std::string, s_t> jointToMarkerSquaredDistances
          = initializer.getJointToMarkerSquaredDistances(pair.first);
      for (std::string& markerName : initializer.getObservedMarkerNames(t))
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

  if (saveToGUI)
  {
    server.writeFramesJson("../../../javascript/src/data/movement2.bin");
  }

  if (jointFailure)
  {
    return false;
  }
  if (avgJointCenterEstimateError > 0.05)
  {
    std::cout << "Joint error is too high on synthetic data!" << std::endl;
    return false;
  }
  return true;
}

bool verifyJointCenterReconstructionOnSyntheticRandomPosesOsim(
    std::string openSimPath)
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
  std::vector<bool> newClip;
  std::vector<Eigen::VectorXs> poses;
  std::vector<Eigen::VectorXs> jointCenters;
  for (int t = 0; t < 50; t++)
  {
    Eigen::VectorXs pose = osim.skeleton->getRandomPose();
    poses.push_back(pose);
    osim.skeleton->setPositions(pose);
    markerObservations.push_back(
        osim.skeleton->getMarkerMapWorldPositions(osim.markersMap));
    jointCenters.push_back(
        osim.skeleton->getJointWorldPositions(osim.skeleton->getJoints()));
    // There's no continuity between any of our random poses, so we always need
    // to completely restart the IK from timestep to timestep
    newClip.push_back(true);
  }

  // Reset to neutral scale
  osim.skeleton->setBodyScales(
      Eigen::VectorXs::Ones(osim.skeleton->getNumBodyNodes() * 3));

  std::map<std::string, bool> markerIsAnatomical;
  for (auto& pair : osim.markersMap)
  {
    markerIsAnatomical[pair.first] = false;
  }
  for (std::string& marker : osim.anatomicalMarkers)
  {
    markerIsAnatomical[marker] = true;
  }

  IKInitializer initializer(
      osim.skeleton,
      osim.markersMap,
      markerIsAnatomical,
      markerObservations,
      newClip,
      targetHeight);

  // Initial guesses using MDS to center our subsequent least-squares on
  initializer.closedFormMDSJointCenterSolver();
  s_t pivotError = initializer.closedFormPivotFindingJointCenterSolver(true);
  std::cout << "Pivot error avg: " << pivotError << "m" << std::endl;
  initializer.recenterAxisJointsBasedOnBoneAngles(true);

  s_t avgJointCenterEstimateError = 0.0;
  int numJointsCenterEstimatesCounted = 0;
  std::map<std::string, s_t> avgJointCenterEstimateErrorByJoint;
  for (int t = 0; t < markerObservations.size(); t++)
  {
    auto jointMap = initializer.getJointsAttachedToObservedMarkersCenters(t);
    for (auto& pair : jointMap)
    {
      Eigen::Vector3s jointWorld = jointCenters[t].segment<3>(
          osim.skeleton->getJoint(pair.first)->getJointIndexInSkeleton() * 3);
      Eigen::Vector3s jointCenterEstimated = pair.second;
      s_t jointError = (jointWorld - jointCenterEstimated).norm();
      avgJointCenterEstimateErrorByJoint[pair.first] += jointError;
      avgJointCenterEstimateError += jointError;
      numJointsCenterEstimatesCounted++;

      std::map<std::string, s_t> jointToMarkerSquaredDistances
          = initializer.getJointToMarkerSquaredDistances(pair.first);
    }
  }
  for (auto& pair : avgJointCenterEstimateErrorByJoint)
  {
    pair.second /= markerObservations.size();
    std::cout << "Joint center " << pair.first << " error: " << pair.second
              << std::endl;
  }

  if (numJointsCenterEstimatesCounted > 0)
  {
    avgJointCenterEstimateError /= numJointsCenterEstimatesCounted;
  }
  std::cout << "Joint center estimate error: " << avgJointCenterEstimateError
            << "m" << std::endl;
  if (avgJointCenterEstimateError > 0.015)
  {
    std::cout << "Joint center estimate error is too high on synthetic data!"
              << std::endl;
    return false;
  }

  return true;
}

bool verifyMarkerReconstructionOnOsim(
    std::string openSimPath, std::vector<std::string> trcFiles, s_t heightM)
{
  (void)heightM;
  auto osim = OpenSimParser::parseOsim(openSimPath);
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::vector<bool> newClip;
  for (std::string trcFile : trcFiles)
  {
    auto trc = OpenSimParser::loadTRC(trcFile);
    // Add the marker observations to our list
    for (int t = 0; t < trc.markerTimesteps.size(); t++)
    {
      auto& obs = trc.markerTimesteps[t];
      markerObservations.push_back(obs);
      newClip.push_back(t == 0);
    }
  }
  std::map<std::string, bool> markerIsAnatomical;
  for (auto& pair : osim.markersMap)
  {
    markerIsAnatomical[pair.first] = false;
  }
  for (std::string& marker : osim.anatomicalMarkers)
  {
    markerIsAnatomical[marker] = true;
  }

  IKInitializer initializer(
      osim.skeleton,
      osim.markersMap,
      markerIsAnatomical,
      markerObservations,
      newClip,
      heightM);
  for (int t = 0; t < markerObservations.size(); t++)
  {
    std::vector<std::string> markerNames
        = initializer.getObservedMarkerNames(t);
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
    Eigen::MatrixXs P = IKInitializer::getPointCloudFromDistanceMatrix(D);
    for (int i = 0; i < markerNames.size(); i++)
    {
      for (int j = 0; j < markerNames.size(); j++)
      {
        s_t recoveredDist = (P.col(i) - P.col(j)).squaredNorm();
        s_t originalDist = D(i, j);
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
    Eigen::MatrixXs mapped = math::mapPointCloudToData(P, markerPoints);
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
  Eigen::MatrixXs P = IKInitializer::getPointCloudFromDistanceMatrix(D);

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

  Eigen::MatrixXs mapped = math::mapPointCloudToData(P, first3Points);
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

  Eigen::Isometry3s T_wb_recovered = math::getPointCloudToPointCloudTransform(
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
TEST(IKInitializer, RESCALE_POINT_CLOUD_FULLY_CONSTRAINED)
{
  srand(42);
  int numPoints = 5;
  Eigen::Vector3s scale
      = (Eigen::Vector3s::Random() * 0.25) + Eigen::Vector3s::Constant(1.0);
  std::vector<Eigen::Vector3s> points_local;
  for (int i = 0; i < numPoints; i++)
  {
    points_local.push_back(Eigen::Vector3s::Random());
  }

  std::vector<Eigen::Vector3s> points_scaled;
  for (int i = 0; i < numPoints; i++)
  {
    points_scaled.push_back(points_local[i].cwiseProduct(scale));
  }

  std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
  for (int i = 0; i < numPoints; i++)
  {
    for (int j = i + 1; j < numPoints; j++)
    {
      s_t weight = 1.0;
      s_t distance = (points_scaled[i] - points_scaled[j]).norm();
      pairDistancesWithWeights.push_back(
          std::make_tuple(i, j, distance, weight));
    }
  }

  Eigen::Vector3s scaleRecovered
      = IKInitializer::getLocalScale(points_local, pairDistancesWithWeights);

  EXPECT_TRUE(scaleRecovered.isApprox(scale, 1e-8));
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, RESCALE_POINT_CLOUD_ONE_AXIS_UNDER_CONSTRAINED)
{
  srand(42);
  int numPoints = 5;
  Eigen::Vector3s scale
      = (Eigen::Vector3s::Random() * 0.25) + Eigen::Vector3s::Constant(1.0);
  std::vector<Eigen::Vector3s> points_local;
  for (int i = 0; i < numPoints; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();
    point(1) = 2.0;
    points_local.push_back(point);
  }

  std::vector<Eigen::Vector3s> points_scaled;
  for (int i = 0; i < numPoints; i++)
  {
    points_scaled.push_back(points_local[i].cwiseProduct(scale));
  }

  std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
  for (int i = 0; i < numPoints; i++)
  {
    for (int j = i + 1; j < numPoints; j++)
    {
      s_t weight = 1.0;
      s_t distance = (points_scaled[i] - points_scaled[j]).norm();
      pairDistancesWithWeights.push_back(
          std::make_tuple(i, j, distance, weight));
    }
  }

  const s_t DEFAULT_SCALE = 1.0;
  Eigen::Vector3s scaleExpected = scale;
  scaleExpected(1) = DEFAULT_SCALE;

  Eigen::Vector3s scaleRecovered = IKInitializer::getLocalScale(
      points_local, pairDistancesWithWeights, DEFAULT_SCALE);

  EXPECT_TRUE(scaleRecovered.isApprox(scaleExpected, 1e-8));
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, RESCALE_POINT_CLOUD_TWO_AXIS_UNDER_CONSTRAINED)
{
  srand(42);
  int numPoints = 5;
  Eigen::Vector3s scale
      = (Eigen::Vector3s::Random() * 0.25) + Eigen::Vector3s::Constant(1.0);
  std::vector<Eigen::Vector3s> points_local;
  for (int i = 0; i < numPoints; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();
    point(1) = 2.0;
    point(2) = 2.0;
    points_local.push_back(point);
  }

  std::vector<Eigen::Vector3s> points_scaled;
  for (int i = 0; i < numPoints; i++)
  {
    points_scaled.push_back(points_local[i].cwiseProduct(scale));
  }

  std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
  for (int i = 0; i < numPoints; i++)
  {
    for (int j = i + 1; j < numPoints; j++)
    {
      s_t weight = 1.0;
      s_t distance = (points_scaled[i] - points_scaled[j]).norm();
      pairDistancesWithWeights.push_back(
          std::make_tuple(i, j, distance, weight));
    }
  }

  const s_t DEFAULT_SCALE = 1.0;
  Eigen::Vector3s scaleExpected = scale;
  scaleExpected(1) = DEFAULT_SCALE;
  scaleExpected(2) = DEFAULT_SCALE;

  Eigen::Vector3s scaleRecovered = IKInitializer::getLocalScale(
      points_local, pairDistancesWithWeights, DEFAULT_SCALE);

  EXPECT_TRUE(scaleRecovered.isApprox(scaleExpected, 1e-8));
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, RESCALE_POINT_CLOUD_ONE_AXIS_TOO_SMALL)
{
  srand(42);
  int numPoints = 5;
  Eigen::Vector3s scale
      = (Eigen::Vector3s::Random() * 0.25) + Eigen::Vector3s::Constant(1.0);
  std::vector<Eigen::Vector3s> points_local;
  for (int i = 0; i < numPoints; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();
    point(1) *= 0.01;
    points_local.push_back(point);
  }

  std::vector<Eigen::Vector3s> points_scaled;
  for (int i = 0; i < numPoints; i++)
  {
    points_scaled.push_back(points_local[i].cwiseProduct(scale));
  }

  std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
  for (int i = 0; i < numPoints; i++)
  {
    for (int j = i + 1; j < numPoints; j++)
    {
      s_t weight = 1.0;
      s_t distance = (points_scaled[i] - points_scaled[j]).norm();
      pairDistancesWithWeights.push_back(
          std::make_tuple(i, j, distance, weight));
    }
  }

  const s_t DEFAULT_SCALE = 1.0;
  Eigen::Vector3s scaleExpected = scale;
  scaleExpected(1) = DEFAULT_SCALE;

  Eigen::Vector3s scaleRecovered = IKInitializer::getLocalScale(
      points_local, pairDistancesWithWeights, DEFAULT_SCALE);

  EXPECT_TRUE(scaleRecovered.isApprox(scaleExpected, 1e-8));
}
#endif

#ifdef ALL_TESTS
/// This test is just for exploring the math in the Chang Pollard paper. This
/// doesn't actually test anything in IKInitializer, it's just validating that
/// the formula I'm using for mapping into/out-of the 5-dim polynomial basis
/// functions they use is correct.
TEST(IKInitializer, CHANG_POLLARD_SIMPLE_FORWARD_BASIS)
{
  srand(42);
  for (int i = 0; i < 5000; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();
    Eigen::Vector3s center = Eigen::Vector3s::Random();
    s_t radius = -5.0 + 10.0 * ((s_t)rand() / RAND_MAX);

    s_t cost = (point - center).squaredNorm() - (radius * radius);

    s_t a = 2.0;

    Eigen::Vector5s basis;
    basis << point.squaredNorm(), point(0), point(1), point(2), 1.0;
    Eigen::Vector5s u;
    u << a, -2 * a * center(0), -2 * a * center(1), -2 * a * center(2),
        (center.squaredNorm() - radius * radius) * a;
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
/// This test is just for exploring the math in the Chang Pollard paper. This
/// doesn't actually test anything in IKInitializer, it's just validating that
/// the formula I'm using for mapping into/out-of the 5-dim polynomial basis
/// functions they use is correct.
TEST(IKInitializer, CHANG_POLLARD_SIMPLE_REVERSE_BASIS)
{
  srand(42);
  for (int i = 0; i < 5000; i++)
  {
    Eigen::Vector3s point = Eigen::Vector3s::Random();

    Eigen::Vector5s u = Eigen::Vector5s::Random();

    Eigen::Vector3s center = u.segment<3>(1) / (-2.0 * u(0));
    s_t radiusSquared = center.squaredNorm() - (u(4) / u(0));
    s_t recoveredCost = (point - center).squaredNorm() - radiusSquared;

    Eigen::Vector5s recoveredU;
    recoveredU << 1.0, -2 * center(0), -2 * center(1), -2 * center(2),
        center.squaredNorm() - radiusSquared;
    recoveredU *= u(0);
    EXPECT_TRUE(recoveredU.segment<3>(1).isApprox(u.segment<3>(1), 1e-8));
    EXPECT_NEAR(recoveredU(4), u(4), 1e-8);

    Eigen::Vector5s basis;
    basis << point.squaredNorm(), point(0), point(1), point(2), 1.0;
    s_t cost = basis.dot(u) / u(0);

    EXPECT_NEAR(cost, recoveredCost, 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_SIMPLE_REVERSE_BASIS_REGRESSION_1)
{
  Eigen::Vector5s u;
  u << 0.257748, 0.592256, 0.167432, -0.450467, 0.659196;

  Eigen::Vector3s center = u.segment<3>(1) / (-2.0 * u(0));
  s_t radiusSquared = center.squaredNorm() - (u(4) / u(0));

  Eigen::Vector5s recoveredU;
  recoveredU << 1.0, -2 * center(0), -2 * center(1), -2 * center(2),
      center.squaredNorm() - radiusSquared;
  recoveredU *= u(0);
  EXPECT_TRUE(recoveredU.segment<3>(1).isApprox(u.segment<3>(1), 1e-8));
  EXPECT_NEAR(recoveredU(4), u(4), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_POLYNOMIAL_REGRESSION_1)
{
  srand(42);

  Eigen::Vector3s dataPoint = Eigen::Vector3s(-0.0910221, 0.179806, 0.172988);

  Eigen::Vector5s polynomial = Eigen::Vector5s::Zero();
  polynomial(0) = dataPoint.squaredNorm();
  polynomial(1) = dataPoint(0);
  polynomial(2) = dataPoint(1);
  polynomial(3) = dataPoint(2);
  polynomial(4) = 1.0;

  // Now we'll evaluate the cost in two different formats on the forward version
  for (int i = 0; i < 1000; i++)
  {
    // First generate a random point to evaluate
    s_t radius = 0.1 + ((s_t)rand() / RAND_MAX) * 0.9;
    Eigen::Vector3s center = Eigen::Vector3s::Random();

    // Now we'll evaluate the cost in the original definition
    s_t originalLoss = (dataPoint - center).squaredNorm() - radius * radius;

    // Next, we'll construct the algebraic version of the loss
    // From the original formula, we have:
    // (dataPoint(0) - center(0))^2 + (dataPoint(1) - center(1))^2 +
    // (dataPoint(2) - center(2))^2 - radius * radius
    // We can expand this out to:
    // dataPoint(0)^2 - 2 * dataPoint(0) * center(0) + center(0)^2 +
    // dataPoint(1)^2 - 2 * dataPoint(1) * center(1) + center(1)^2 +
    // dataPoint(2)^2 - 2 * dataPoint(2) * center(2) + center(2)^2 -
    // radius * radius
    // We can then reorganize the terms, and we get:
    // (dataPoint.squaredNorm())*(1.0)
    // (dataPoint(0))*(-2 * center(0))
    // (dataPoint(1))*(-2 * center(1))
    // (dataPoint(2))*(-2 * center(2))
    // (1.0)*(center.squaredNorm() - radius * radius)

    Eigen::Vector5s u = Eigen::Vector5s::Zero();
    u(0) = 1.0;
    u(1) = -2 * center(0);
    u(2) = -2 * center(1);
    u(3) = -2 * center(2);
    u(4) = center.squaredNorm() - radius * radius;

    s_t algebraicLoss = polynomial.dot(u);

    EXPECT_NEAR(originalLoss, algebraicLoss, 1e-10);
  }

  Eigen::Matrix5s C = Eigen::Matrix5s::Zero();
  C(1, 1) = 1.0;
  C(2, 2) = 1.0;
  C(3, 3) = 1.0;
  C(4, 0) = -2.0;
  C(0, 4) = -2.0;

  // Now evaluate the cost in two different formats on the backward version
  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector5s u = Eigen::Vector5s::Random();
    s_t rawConstraint = u.dot(C * u);
    if (rawConstraint < 0)
    {
      continue;
    }
    u /= sqrt(rawConstraint);

    // First evaluate the algebraic loss
    s_t algebraicLoss = polynomial.dot(u) / u(0);

    // Now we need to recover the center point and the radius
    u /= u(0);
    Eigen::Vector3s center = u.segment<3>(1) / -2.0;
    s_t radiusSquared = (center.squaredNorm() - u(4));
    // u(4) = center.squaredNorm() - radiusSquared;
    // u(4) - center.squaredNorm() = - radiusSquared;
    // center.squaredNorm() - u(4) = radiusSquared;

    s_t originalLoss = (dataPoint - center).squaredNorm() - radiusSquared;

    EXPECT_NEAR(originalLoss, algebraicLoss, 1e-9);
  }

  // Problem U: -13.0184   -1.356  4.62209  5.64295 -1.03786
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
TEST(IKInitializer, CHANG_POLLARD_SINGLE_MARKER_NO_NOISE)
{
  Eigen::Vector3s center = Eigen::Vector3s::UnitX() * 0;
  std::vector<Eigen::Vector3s> markerObservations1;
  s_t radius1 = 1.0;
  markerObservations1.push_back(
      center + Eigen::Vector3s::UnitX() * (radius1 + 0.01));
  markerObservations1.push_back(
      center - Eigen::Vector3s::UnitX() * (radius1 + 0.01));
  markerObservations1.push_back(center + Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitY() * radius1);
  markerObservations1.push_back(center + Eigen::Vector3s::UnitZ() * radius1);
  markerObservations1.push_back(center - Eigen::Vector3s::UnitZ() * radius1);

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  markerTraces.push_back(markerObservations1);

  Eigen::Vector3s center_raw
      = IKInitializer::getChangPollard2006JointCenterMultiMarker(
          markerTraces, true);
  s_t error_raw = (center_raw - center).norm();
  EXPECT_NEAR(error_raw, 0.0, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, CHANG_POLLARD_REGRESSION_1)
{
  std::vector<Eigen::Vector3s> markerObservations;
  markerObservations.push_back(Eigen::Vector3s(-0.0910221, 0.179806, 0.172988));
  markerObservations.push_back(Eigen::Vector3s(-0.0701303, 0.178589, 0.191935));
  markerObservations.push_back(Eigen::Vector3s(-0.0691176, 0.178504, 0.206454));
  markerObservations.push_back(Eigen::Vector3s(-0.0723091, 0.178722, 0.187018));
  markerObservations.push_back(Eigen::Vector3s(-0.0712103, 0.178656, 0.189191));
  markerObservations.push_back(Eigen::Vector3s(-0.0704132, 0.178607, 0.191132));
  markerObservations.push_back(Eigen::Vector3s(-0.0717303, 0.178687, 0.188106));
  markerObservations.push_back(Eigen::Vector3s(-0.0707039, 0.178625, 0.190376));
  markerObservations.push_back(Eigen::Vector3s(-0.0686261, 0.178486, 0.201041));

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  markerTraces.push_back(markerObservations);

  Eigen::Vector3s least_squares_center
      = IKInitializer::leastSquaresConcentricSphereFit(markerTraces);
  std::cout << "Least squares center: " << std::endl
            << least_squares_center << std::endl;

  Eigen::Vector3s center_raw
      = IKInitializer::getChangPollard2006JointCenterMultiMarker(
          markerTraces, true);

  std::vector<s_t> radii;
  std::vector<s_t> radiiLS;
  s_t avgRadius = 0.0;
  s_t avgRadiusLS = 0.0;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    // Do chang-pollard
    s_t radius = (markerObservations[i] - center_raw).norm();
    radii.push_back(radius);
    avgRadius += radius;

    // Do least squares
    s_t radiusLS = (markerObservations[i] - least_squares_center).norm();
    radiiLS.push_back(radiusLS);
    avgRadiusLS += radiusLS;
  }
  avgRadius /= markerObservations.size();
  avgRadiusLS /= markerObservations.size();

  s_t radiusVariance = 0.0;
  s_t radiusVarianceLS = 0.0;
  for (int i = 0; i < markerObservations.size(); i++)
  {
    radiusVariance += (radii[i] - avgRadius) * (radii[i] - avgRadius);
    radiusVarianceLS += (radiiLS[i] - avgRadiusLS) * (radiiLS[i] - avgRadiusLS);
  }
  std::cout << "Radius variance (Chang Pollard): " << radiusVariance
            << std::endl;
  std::cout << "Radius variance (Least Squares): " << radiusVarianceLS
            << std::endl;
  EXPECT_LE(radiusVariance, 1e-12);
  EXPECT_LE(radiusVarianceLS, 1e-12);
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
TEST(IKInitializer, AXIS_FIT_NO_NOISE)
{
  Eigen::Vector3s axis = Eigen::Vector3s::UnitX();
  Eigen::Vector3s center = Eigen::Vector3s::UnitZ();

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3s markerPos = Eigen::Vector3s::Random();
    std::vector<Eigen::Vector3s> markerObservations;
    for (int t = 0; t < 50; t++)
    {
      Eigen::Matrix3s R = math::expMapRot(axis * 0.01 * t);
      markerObservations.push_back(center + (R * markerPos));
    }
    markerTraces.push_back(markerObservations);
  }

  auto pair = IKInitializer::gamageLasenby2002AxisFit(markerTraces);
  s_t conditionNumber = pair.second;
  Eigen::Vector3s axisRecovered = pair.first;
  EXPECT_GT(std::abs(conditionNumber), 1e5);
  EXPECT_TRUE((axisRecovered - axis).norm() < 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, AXIS_FIT_SOME_NOISE)
{
  Eigen::Vector3s axis = Eigen::Vector3s::Random().normalized();
  Eigen::Vector3s center = Eigen::Vector3s::Random();

  std::vector<std::vector<Eigen::Vector3s>> markerTraces;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3s markerPos = Eigen::Vector3s::Random();
    std::vector<Eigen::Vector3s> markerObservations;
    for (int t = 0; t < 500; t++)
    {
      Eigen::Matrix3s R
          = math::expMapRot(axis * ((s_t)rand() / RAND_MAX) * 3.14159 * 2);
      Eigen::Vector3s noise = Eigen::Vector3s::Random() * 0.01;
      markerObservations.push_back(center + (R * markerPos) + noise);
    }
    markerTraces.push_back(markerObservations);
  }

  // auto pair = IKInitializer::svdAxisFit(markerTraces, center);
  auto pair = IKInitializer::gamageLasenby2002AxisFit(markerTraces);
  s_t conditionNumber = pair.second;
  Eigen::Vector3s axisRecovered = pair.first;
  // Deal with sign ambiguity
  if (((axisRecovered * -1) - axis).norm() < (axisRecovered - axis).norm())
  {
    axisRecovered *= -1;
  }
  EXPECT_GT(std::abs(conditionNumber), 1e3);
  if ((axisRecovered - axis).norm() >= 1e-3)
  {
    Eigen::Matrix3s compare;
    compare.col(0) = axis;
    compare.col(1) = axisRecovered;
    compare.col(2) = axis - axisRecovered;
    std::cout << "Axis error of " << (axisRecovered - axis).norm()
              << " is too high!" << std::endl;
    std::cout << "Axis - Recovered - Error: " << std::endl
              << compare << std::endl;
  }
  EXPECT_TRUE((axisRecovered - axis).norm() < 1e-3);
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_REAL_ROOTS)
{
  std::vector<double> roots = IKInitializer::findCubicRealRoots(1, -6, 11, -6);
  std::vector<double> expected = {1, 2, 3};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());

  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_COMPLEX_ROOTS)
{
  std::vector<double> roots = IKInitializer::findCubicRealRoots(1, -3, 3, -1);
  std::vector<double> expected = {1};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_ROOT_REGRESSION_1)
{
  std::vector<double> roots
      = IKInitializer::findCubicRealRoots(28, -3.36611, 0.220185, -0.00323973);
  std::vector<double> expected = {0.0196532};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-6);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_ROOT_REGRESSION_2)
{
  std::vector<double> roots
      = IKInitializer::findCubicRealRoots(24, 1.3544, 0.17324, 0.00158158);
  std::vector<double> expected = {-0.00974348};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-6);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_ROOT_REGRESSION_3)
{
  std::vector<double> roots = IKInitializer::findCubicRealRoots(
      18, 0.374666, 0.00595641, 6.77338e-07);
  std::vector<double> expected = {-0.000114536};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-6);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CUBIC_CONJUGATE_ROOT_PAIR)
{
  std::vector<double> roots = IKInitializer::findCubicRealRoots(1, 0, 1, 0);
  std::vector<double> expected = {0};
  std::sort(roots.begin(), roots.end());
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(roots.size(), expected.size());
  for (int i = 0; i < roots.size(); ++i)
  {
    EXPECT_NEAR(roots[i], expected[i], 1e-8);
  }
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CENTER_POINT_ON_AXIS_ONE_SUPPORT_POINT)
{
  std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii;
  pointsAndRadii.emplace_back(Eigen::Vector3s(1, 1, 0), 1.0);

  Eigen::Vector3s center = Eigen::Vector3s::Zero();
  Eigen::Vector3s axis = Eigen::Vector3s::UnitX();

  Eigen::Vector3s expectedCenter = Eigen::Vector3s::UnitX();
  Eigen::Vector3s recoveredCenter
      = IKInitializer::centerPointOnAxis(center, axis, pointsAndRadii);

  if ((recoveredCenter - expectedCenter).norm() >= 1e-8)
  {
    std::cout << "Axis centering failed." << std::endl;
    Eigen::Matrix3s compare;
    compare.col(0) = recoveredCenter;
    compare.col(1) = expectedCenter;
    compare.col(2) = recoveredCenter - expectedCenter;
    std::cout << "Recovered - expected - diff:" << std::endl
              << compare << std::endl;
  }
  EXPECT_TRUE((recoveredCenter - expectedCenter).norm() < 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CENTER_POINT_ON_AXIS_TWO_SYMMETRIC_SUPPORT_POINTS)
{
  std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii;
  pointsAndRadii.emplace_back(Eigen::Vector3s(-1.5, 1, 0), 1.0);
  pointsAndRadii.emplace_back(Eigen::Vector3s(-0.5, 1, 0), 1.0);

  Eigen::Vector3s center = Eigen::Vector3s::Zero();
  Eigen::Vector3s axis = Eigen::Vector3s::UnitX();

  Eigen::Vector3s expectedCenter = -Eigen::Vector3s::UnitX();
  Eigen::Vector3s recoveredCenter
      = IKInitializer::centerPointOnAxis(center, axis, pointsAndRadii);

  EXPECT_TRUE((recoveredCenter - expectedCenter).norm() < 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(SolveCubicTest, CENTER_POINT_ON_AXIS_RANDOM_POINTS)
{
  srand(42);

  // Generate a random problem definition
  std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii;
  for (int i = 0; i < 10; i++)
  {
    pointsAndRadii.emplace_back(
        Eigen::Vector3s::Random(), ((s_t)rand() / RAND_MAX) * 3.0 + 0.01);
  }
  Eigen::Vector3s center = Eigen::Vector3s::Zero();
  Eigen::Vector3s axis = Eigen::Vector3s::Random().normalized();

  // Solve it
  Eigen::Vector3s optimalCenter
      = IKInitializer::centerPointOnAxis(center, axis, pointsAndRadii);

  // Test a bunch of random points to see if we can guess a better value
  for (int i = 0; i < 1000; i++)
  {
    s_t randomDistance = ((s_t)rand() / RAND_MAX) * 3.0 - 1.5;
    Eigen::Vector3s trialPoint = center + (axis * randomDistance);

    s_t optimalLoss = 0.0;
    s_t trialPointLoss = 0.0;
    for (int t = 0; t < pointsAndRadii.size(); t++)
    {
      const s_t optimalLinearErrorOnSquaredDistances
          = ((pointsAndRadii[t].first - optimalCenter).squaredNorm()
             - pointsAndRadii[t].second * pointsAndRadii[t].second);
      optimalLoss += optimalLinearErrorOnSquaredDistances
                     * optimalLinearErrorOnSquaredDistances;

      const s_t trialLinearErrorOnSquaredDistances
          = ((pointsAndRadii[t].first - trialPoint).squaredNorm()
             - pointsAndRadii[t].second * pointsAndRadii[t].second);
      trialPointLoss += trialLinearErrorOnSquaredDistances
                        * trialLinearErrorOnSquaredDistances;
    }
    EXPECT_LE(optimalLoss, trialPointLoss);
    if (optimalLoss > trialPointLoss)
    {
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKInitializer, SYNTHETIC_OSIM_JUST_JOINT_CENTERS)
{
  EXPECT_TRUE(verifyJointCenterReconstructionOnSyntheticRandomPosesOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim"));
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
TEST(IKInitializer, SYNTHETIC_OSIM_SCALES_GIVEN)
{
  EXPECT_TRUE(verifyReconstructionOnSyntheticRandomPosesOsim(
      "dart://sample/grf/subject18_synthetic/"
      "unscaled_generic.osim",
      true,
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
      1.775,
      true);
}
*/

/*
TEST(IKInitializer, VISUALIZE_RESULTS_REGRESSION_SUBJECT02)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/regression/Arnold2013Synthetic/subject02/trials/walk2/"
      "markers.trc");

  runOnRealOsim(
      "dart://sample/regression/Arnold2013Synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.853,
      true);
}
*/

/*
TEST(IKInitializer, VISUALIZE_RESULTS_REGRESSION_SUBJECT18)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/regression/Arnold2013Synthetic/subject18/trials/walk2/"
      "markers.trc");

  runOnRealOsim(
      "dart://sample/regression/Arnold2013Synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.775,
      true);
}
*/

/*
TEST(IKInitializer, VISUALIZE_RESULTS_REGRESSION_SUBJECT19)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/regression/Arnold2013Synthetic/subject19/trials/walk2/"
      "markers.trc");

  runOnRealOsim(
      "dart://sample/regression/Arnold2013Synthetic/"
      "unscaled_generic.osim",
      trcFiles,
      1.79,
      true);
}
*/

/*
TEST(IKInitializer, VISUALIZE_RESULTS_REGRESSION_ANTOINE_BUG)
{
  std::vector<std::string> trcFiles;
  trcFiles.push_back(
      "dart://sample/osim/Antoine_Subj03_input/trials/Mocap0001/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/Antoine_Subj03_input/trials/Mocap0002/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/Antoine_Subj03_input/trials/Mocap0003/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/Antoine_Subj03_input/trials/Mocap0004/markers.trc");

  runOnRealOsim(
      "dart://sample/osim/Antoine_Subj03_input/unscaled_generic.osim",
      trcFiles,
      1.8,
      true);
}
*/