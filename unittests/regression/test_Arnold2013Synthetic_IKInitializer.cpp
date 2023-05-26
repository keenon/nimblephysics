#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/IKInitializer.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/math/MathTypes.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace math;

void testSubject(
    const std::string& subject, const s_t& height, bool knownScales = false)
{

  std::cout << "Testing " << subject << "..." << std::endl;

  // Run IKInitializer
  // -----------------
  std::string prefix = "dart://sample/regression/Arnold2013Synthetic/";
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;
  trcFiles.push_back(prefix + subject + "/trials/walk2/markers.trc");
  grfFiles.push_back(prefix + subject + "/trials/walk2/grf.mot");
  (void)height;
  std::string modelPath = knownScales
                              ? (prefix + subject + "/" + subject + ".osim")
                              : (prefix + "unscaled_generic.osim");
  s_t heightM = knownScales ? -1.0 : height;
  bool knownScalesInAdvance = knownScales;

  // Initialize data structures.
  // ---------------------------
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<std::vector<bool>> newClipTrials;
  std::vector<int> framesPerSecond;

  // Fill marker and GRF data structures.
  // ------------------------------------
  for (int itrial = 0; itrial < (int)trcFiles.size(); itrial++)
  {
    // Markers.
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[itrial]);
    framesPerSecond.push_back(trc.framesPerSecond);
    markerObservationTrials.push_back(trc.markerTimesteps);
    std::vector<bool> newClip;
    for (int i = 0; i < trc.markerTimesteps.size(); i++)
    {
      newClip.push_back(i == 0);
    }
    newClipTrials.push_back(newClip);
  }

  // Load the model.
  // ---------------
  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  standard.skeleton->zeroTranslationInCustomFunctions();
  standard.skeleton->autogroupSymmetricSuffixes();
  if (standard.skeleton->getBodyNode("hand_r") != nullptr)
  {
    standard.skeleton->setScaleGroupUniformScaling(
        standard.skeleton->getBodyNode("hand_r"));
  }
  standard.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
  standard.skeleton->setPositionLowerLimit(0, -M_PI);
  standard.skeleton->setPositionUpperLimit(0, M_PI);
  standard.skeleton->setPositionLowerLimit(1, -M_PI);
  standard.skeleton->setPositionUpperLimit(1, M_PI);
  standard.skeleton->setPositionLowerLimit(2, -M_PI);
  standard.skeleton->setPositionUpperLimit(2, M_PI);

  // Populate the markers list.
  // --------------------------
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerList;
  std::map<std::string, bool> markerIsAnatomical;
  for (auto& pair : standard.markersMap)
  {
    markerList.push_back(pair.second);
  }

  for (auto marker : standard.anatomicalMarkers)
  {
    std::cout << "Anatomical marker: " << marker << std::endl;
    markerIsAnatomical[marker] = true;
  }

  for (auto marker : standard.trackingMarkers)
  {
    std::cout << "Tracking marker: " << marker << std::endl;
    markerIsAnatomical[marker] = false;
  }

  for (auto marker : standard.markersMap)
  {
    std::cout << "Marker: " << marker.first << std::endl;
    std::cout << "  Body: " << marker.second.first->getName() << std::endl;
    std::cout << "  Offset: " << marker.second.second.transpose() << std::endl;
  }

  // Create and run the IKInitializer
  // ------------------------
  IKInitializer initializer(
      standard.skeleton,
      standard.markersMap,
      markerIsAnatomical,
      markerObservationTrials[0],
      newClipTrials[0],
      heightM);
  if (knownScalesInAdvance)
  {
    initializer.closedFormMDSJointCenterSolver();
    initializer.estimateGroupScalesClosedForm();
    initializer.estimatePosesWithIK();
  }
  else
  {
    initializer.runFullPipeline(false);
  }

  std::vector<Eigen::VectorXs> poses = initializer.getPoses();
  std::vector<Eigen::VectorXi> posesClosedForeEstimateAvailable
      = initializer.getPosesClosedFormEstimateAvailable();
  Eigen::VectorXs groupScales = initializer.getGroupScales();
  OpenSimFile osimFile = standard;

  // Load ground-truth results.
  // --------------------------
  auto goldOsim
      = OpenSimParser::parseOsim(prefix + subject + "/" + subject + ".osim");
  auto goldIK = OpenSimParser::loadMot(
      goldOsim.skeleton, prefix + subject + "/coordinates.sto");

  // Compare poses.
  // --------------
  auto goldPoses = goldIK.poses;
  s_t totalError = 0.0;
  Eigen::VectorXs averagePerDofError = Eigen::VectorXs::Zero(poses[0].size());
  for (int i = 0; i < poses.size(); i++)
  {
    Eigen::VectorXs diff = poses[i] - goldPoses.col(i);
    // Only could those joint with a closed form estimate available
    diff = diff.cwiseProduct(posesClosedForeEstimateAvailable[i].cast<s_t>());
    totalError += diff.cwiseAbs().sum() / diff.size();
    averagePerDofError += diff.cwiseAbs();
  }
  s_t averagePoseError = totalError / (s_t)poses.size();
  averagePerDofError /= (s_t)poses.size();
  s_t threshold = knownScales ? 0.03 : 0.18;
  if (averagePoseError >= threshold)
  {
    std::cout << "Average pose error norm " << averagePoseError << " > 0.01"
              << std::endl;
    for (int i = 0; i < averagePerDofError.size(); i++)
    {
      std::cout << "  " << osimFile.skeleton->getDof(i)->getName() << ": "
                << averagePerDofError(i) << std::endl;
    }
  }
  EXPECT_LE(averagePoseError, threshold);

  auto goldSkeleton = goldOsim.skeleton;
  auto goldJoints = goldSkeleton->getJoints();

  // Marker errors.
  // --------------
  // EXPECT_TRUE(finalKinematicsReport.averageRootMeanSquaredError < 0.01);
  // EXPECT_TRUE(finalKinematicsReport.averageMaxError < 0.02);

  // Virtual joint centers.
  // --------------
  std::vector<std::shared_ptr<struct StackedJoint>>& stackedJoints
      = initializer.getStackedJoints();
  std::vector<std::map<std::string, Eigen::Vector3s>>& jointCenters
      = initializer.getJointCenters();

  std::map<std::string, s_t> avgAnalyticalJointError;
  std::map<std::string, int> analyticalJointObservationCount;
  std::map<std::string, std::map<std::string, int>> analyticalJointSourceCount;
  for (int i = 0; i < stackedJoints.size(); i++)
  {
    avgAnalyticalJointError[stackedJoints[i]->name] = 0.0;
    analyticalJointObservationCount[stackedJoints[i]->name] = 0;
    analyticalJointSourceCount[stackedJoints[i]->name]
        = std::map<std::string, int>();
  }
  for (int t = 0; t < jointCenters.size(); t++)
  {
    goldSkeleton->setPositions(goldPoses.col(t));
    Eigen::VectorXs goldJointPoses
        = goldSkeleton->getJointWorldPositions(goldJoints);
    for (int i = 0; i < stackedJoints.size(); i++)
    {
      std::shared_ptr<struct StackedJoint>& stackedJoint = stackedJoints[i];
      if (jointCenters[t].count(stackedJoint->name))
      {
        Eigen::Vector3s goldJointCenter = Eigen::Vector3s::Zero();
        for (int k = 0; k < stackedJoint->joints.size(); k++)
        {
          goldJointCenter += goldJointPoses.segment<3>(
              3 * stackedJoint->joints[k]->getJointIndexInSkeleton());
        }
        goldJointCenter /= stackedJoint->joints.size();
        s_t jointDist
            = (jointCenters[t][stackedJoint->name] - goldJointCenter).norm();
        avgAnalyticalJointError[stackedJoint->name] += jointDist;
        analyticalJointObservationCount[stackedJoint->name]++;

        std::string debugJointEstimateSource
            = initializer.debugJointEstimateSource(t, stackedJoint->name);
        if (analyticalJointSourceCount[stackedJoint->name].count(
                debugJointEstimateSource)
            == 0)
        {
          analyticalJointSourceCount[stackedJoint->name]
                                    [debugJointEstimateSource]
              = 0;
        }
        analyticalJointSourceCount[stackedJoint->name]
                                  [debugJointEstimateSource]++;
      }
    }
  }
  for (int i = 0; i < stackedJoints.size(); i++)
  {
    avgAnalyticalJointError[stackedJoints[i]->name]
        /= analyticalJointObservationCount[stackedJoints[i]->name];
    std::cout << "Analytical Joint \"" << stackedJoints[i]->name
              << "\" got avg error "
              << avgAnalyticalJointError[stackedJoints[i]->name] << "m on "
              << analyticalJointObservationCount[stackedJoints[i]->name]
              << " observations" << std::endl;
    for (auto& pair : analyticalJointSourceCount[stackedJoints[i]->name])
    {
      std::cout << "   source \"" << pair.first << "\" on " << pair.second
                << " frames" << std::endl;
    }
  }

  // Real joint centers.
  // --------------
  auto skeleton = osimFile.skeleton;
  auto joints = skeleton->getJoints();
  Eigen::VectorXs avgJointError
      = Eigen::VectorXs::Zero(skeleton->getNumJoints());
  for (int i = 0; i < poses.size(); i++)
  {
    skeleton->setPositions(poses[i]);
    goldSkeleton->setPositions(goldPoses.col(i));
    Eigen::VectorXs jointPoses = skeleton->getJointWorldPositions(joints);
    Eigen::VectorXs goldJointPoses
        = goldSkeleton->getJointWorldPositions(goldJoints);
    Eigen::VectorXs diff = jointPoses - goldJointPoses;

    for (int j = 0; j < skeleton->getNumJoints(); j++)
    {
      s_t jointDist = diff.segment<3>(3 * j).norm();
      avgJointError(j) += jointDist;
    }
  }
  avgJointError /= poses.size();
  for (int j = 0; j < skeleton->getNumJoints(); j++)
  {
    std::cout << "Joint " << skeleton->getJoint(j)->getName()
              << " average center-estimate error: " << avgJointError(j) << "m"
              << std::endl;
  }
  s_t averageJointCenterError = avgJointError.mean();
  EXPECT_LE(averageJointCenterError, knownScales ? 0.02 : 0.08);

  // Get at body scaling: Measure absolute distances between joint centers in
  // the neutral pose
  // ------------
  skeleton->setPositions(Eigen::VectorXs::Zero(skeleton->getNumDofs()));
  goldSkeleton->setPositions(Eigen::VectorXs::Zero(goldSkeleton->getNumDofs()));

  Eigen::VectorXs jointPoses = skeleton->getJointWorldPositions(joints);
  Eigen::VectorXs goldJointPoses
      = goldSkeleton->getJointWorldPositions(goldJoints);

  s_t avgRelativeError = 0.0;
  for (int i = 0; i < joints.size(); i++)
  {
    for (int j = 0; j < joints.size(); j++)
    {
      s_t dist = (jointPoses.segment<3>(3 * i) - jointPoses.segment<3>(3 * j))
                     .norm();
      s_t goldDist = (goldJointPoses.segment<3>(3 * i)
                      - goldJointPoses.segment<3>(3 * j))
                         .norm();
      s_t error = std::abs(dist - goldDist);
      s_t relativeError = error;
      if (goldDist > 0)
      {
        relativeError /= goldDist;
      }
      avgRelativeError += relativeError;
    }
  }
  avgRelativeError /= joints.size() * joints.size();
  EXPECT_LE(avgRelativeError, knownScales ? 0.01 : 0.035);

  /*
  // Relative body scales
  // ------------
  auto bodies = skeleton->getBodyNodes();
  auto goldBodies = goldSkeleton->getBodyNodes();
  Eigen::VectorXs bodyScaleErrors = Eigen::VectorXs::Zero((int)bodies.size());
  for (int i = 0; i < (int)bodies.size(); i++)
  {
    Eigen::Vector3s bodyScale = bodies[i]->getScale();
    Eigen::Vector3s goldBodyScale = goldBodies[i]->getScale();

    Eigen::Vector3s meshScale = bodyScale;
    Eigen::Vector3s goldMeshScale = goldBodyScale;
    for (int k = 0; k < bodies[i]->getNumShapeNodes(); k++)
    {
      auto* shapeNode = bodies[i]->getShapeNode(k);
      auto* goldShapeNode = goldBodies[i]->getShapeNode(k);
      if (shapeNode->getShape()->getType()
          == dynamics::MeshShape::getStaticType())
      {
        auto* meshShape
            = static_cast<dynamics::MeshShape*>(shapeNode->getShape().get());
        auto* goldMeshShape = static_cast<dynamics::MeshShape*>(
            goldShapeNode->getShape().get());
        meshScale = meshShape->getScale();
        goldMeshScale = goldMeshShape->getScale();
      }
    }

    bodyScaleErrors[i] = (meshScale - goldMeshScale).norm();

    Eigen::Matrix3s compare;
    compare.col(0) = bodyScale;
    compare.col(1) = goldBodyScale;
    compare.col(2) = bodyScale - goldBodyScale;
    std::cout << "Body " << bodies[i]->getName()
              << " error norm: " << bodyScaleErrors[i] << std::endl
              << "guess - gold - diff: " << std::endl
              << compare << std::endl;
  }
  */
}

//==============================================================================
TEST(Arnold2013Synthetic, KNOWN_SCALES_IN_ADVANCE_SUBJECT_01)
{
  testSubject("subject01", 1.808, true);
}

//==============================================================================
TEST(Arnold2013Synthetic, KNOWN_SCALES_IN_ADVANCE_SUBJECT_02)
{
  testSubject("subject02", 1.853, true);
}

//==============================================================================
TEST(Arnold2013Synthetic, KNOWN_SCALES_IN_ADVANCE_SUBJECT_04)
{
  testSubject("subject04", 1.801, true);
}

//==============================================================================
TEST(Arnold2013Synthetic, KNOWN_SCALES_IN_ADVANCE_SUBJECT_18)
{
  testSubject("subject18", 1.775, true);
}

//==============================================================================
TEST(Arnold2013Synthetic, KNOWN_SCALES_IN_ADVANCE_SUBJECT_19)
{
  testSubject("subject19", 1.79, true);
}

//==============================================================================
TEST(Arnold2013Synthetic, UNKNOWN_SCALES_IN_ADVANCE_SUBJECT_01)
{
  testSubject("subject01", 1.808, false);
}

//==============================================================================
TEST(Arnold2013Synthetic, UNKNOWN_SCALES_IN_ADVANCE_SUBJECT_02)
{
  testSubject("subject02", 1.853, false);
}

//==============================================================================
TEST(Arnold2013Synthetic, UNKNOWN_SCALES_IN_ADVANCE_SUBJECT_04)
{
  testSubject("subject04", 1.801, false);
}

//==============================================================================
TEST(Arnold2013Synthetic, UNKNOWN_SCALES_IN_ADVANCE_SUBJECT_18)
{
  testSubject("subject18", 1.775, false);
}

//==============================================================================
TEST(Arnold2013Synthetic, UNKNOWN_SCALES_IN_ADVANCE_SUBJECT_19)
{
  testSubject("subject19", 1.79, false);
}