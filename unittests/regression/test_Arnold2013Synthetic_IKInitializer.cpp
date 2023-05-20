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
#include "dart/math/MathTypes.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace math;

//==============================================================================
std::tuple<
    std::vector<Eigen::VectorXs>,
    std::vector<Eigen::VectorXi>,
    Eigen::VectorXs,
    OpenSimFile>
runIKInitializer(
    std::string modelPath,
    std::vector<std::string> trcFiles,
    s_t heightM,
    bool knownScalesInAdvance = false)
{

  // Initialize data structures.
  // ---------------------------
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;

  // Fill marker and GRF data structures.
  // ------------------------------------
  for (int itrial = 0; itrial < (int)trcFiles.size(); itrial++)
  {
    // Markers.
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[itrial]);
    framesPerSecond.push_back(trc.framesPerSecond);
    markerObservationTrials.push_back(trc.markerTimesteps);
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
  for (auto& pair : standard.markersMap)
  {
    markerList.push_back(pair.second);
  }

  for (auto marker : standard.anatomicalMarkers)
  {
    std::cout << "Anatomical marker: " << marker << std::endl;
  }

  for (auto marker : standard.trackingMarkers)
  {
    std::cout << "Tracking marker: " << marker << std::endl;
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
      markerObservationTrials[0],
      heightM,
      knownScalesInAdvance);
  if (knownScalesInAdvance)
  {
    initializer.closedFormMDSJointCenterSolver();
    initializer.estimateGroupScalesClosedForm();
    initializer.estimatePosesClosedForm();
  }
  else
  {
    initializer.runFullPipeline(true);
  }

  return std::make_tuple(
      initializer.getPoses(),
      initializer.getPosesClosedFormEstimateAvailable(),
      initializer.getGroupScales(),
      standard);
}

void testSubject(
    const std::string& subject, const s_t& height, bool knownScales = false)
{

  std::cout << "Testing " << subject << "..." << std::endl;

  // Run MarkerFitter.
  // -----------------
  std::string prefix = "dart://sample/regression/Arnold2013Synthetic/";
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;
  trcFiles.push_back(prefix + subject + "/trials/walk2/markers.trc");
  grfFiles.push_back(prefix + subject + "/trials/walk2/grf.mot");
  (void)height;
  const auto tuple = runIKInitializer(
      knownScales ? (prefix + subject + "/" + subject + ".osim")
                  : (prefix + "unscaled_generic.osim"),
      trcFiles,
      knownScales ? -1.0 : height,
      knownScales);
  std::vector<Eigen::VectorXs> poses = std::get<0>(tuple);
  std::vector<Eigen::VectorXi> posesClosedForeEstimateAvailable
      = std::get<1>(tuple);
  Eigen::VectorXs groupScales = std::get<2>(tuple);
  OpenSimFile osimFile = std::get<3>(tuple);

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

  // Marker errors.
  // --------------
  // EXPECT_TRUE(finalKinematicsReport.averageRootMeanSquaredError < 0.01);
  // EXPECT_TRUE(finalKinematicsReport.averageMaxError < 0.02);

  // Joint centers.
  // --------------
  auto skeleton = osimFile.skeleton;
  auto joints = skeleton->getJoints();
  auto goldSkeleton = goldOsim.skeleton;
  auto goldJoints = goldSkeleton->getJoints();
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

  // Body scales.
  // ------------
  auto bodies = skeleton->getBodyNodes();
  auto goldBodies = goldSkeleton->getBodyNodes();
  Eigen::VectorXs bodyScaleErrors = Eigen::VectorXs::Zero((int)bodies.size());
  for (int i = 0; i < (int)bodies.size(); i++)
  {
    Eigen::Vector3s bodyScale = bodies[i]->getScale();
    Eigen::Vector3s goldBodyScale = goldBodies[i]->getScale();
    bodyScaleErrors[i] = (bodyScale - goldBodyScale).norm();
  }
  s_t averageBodyScaleError = bodyScaleErrors.mean();
  EXPECT_LE(averageBodyScaleError, knownScales ? 0.01 : 0.03);
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