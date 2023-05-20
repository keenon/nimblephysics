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
std::tuple<MarkerInitialization, IKErrorReport, OpenSimFile> runMarkerFitter(
    std::string modelPath,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    s_t massKg,
    s_t heightM,
    std::string sex,
    bool saveGUI = false)
{

  // Initialize data structures.
  // ---------------------------
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlates;

  // Fill marker and GRF data structures.
  // ------------------------------------
  assert(trcFiles.size() == grfFiles.size());
  for (int itrial = 0; itrial < (int)trcFiles.size(); itrial++)
  {
    // Markers.
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[itrial]);
    framesPerSecond.push_back(trc.framesPerSecond);
    markerObservationTrials.push_back(trc.markerTimesteps);
    // Ground reaction forces.
    std::vector<ForcePlate> grf
        = OpenSimParser::loadGRF(grfFiles[itrial], trc.framesPerSecond);
    forcePlates.push_back(grf);
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

  // Create the MarkerFitter.
  // ------------------------
  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(1e-5);
  fitter.setInitialIKMaxRestarts(150);
  fitter.setIterationLimit(400);
  fitter.setRegularizeAnatomicalMarkerOffsets(10.0);
  fitter.setRegularizeTrackingMarkerOffsets(0.05);
  fitter.setMinSphereFitScore(0.01);
  fitter.setMinAxisFitScore(0.001);
  fitter.setMaxJointWeight(1.0);
  fitter.setRegularizePelvisJointsWithVirtualSpring(0.0);
  fitter.setTrackingMarkers(standard.trackingMarkers);

  // Create the Anthropometric prior.
  // --------------------------------
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_metrics.xml");
  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("weightkg");
  std::shared_ptr<MultivariateGaussian> gauss;
  if (sex == "male")
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
        cols,
        0.001); // mm -> m
  }
  else if (sex == "female")
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
        cols,
        0.001); // mm -> m
  }
  else
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_BOTH_Public.csv",
        cols,
        0.001); // mm -> m
  }
  std::map<std::string, s_t> observedValues;
  std::cout << "Anthro before conditioning:" << std::endl;
  gauss->debugToStdout();
  observedValues["weightkg"] = massKg * 0.01;
  observedValues["stature"] = heightM;
  gauss = gauss->condition(observedValues);
  std::cout << "Anthro after conditioning:" << std::endl;
  gauss->debugToStdout();
  anthropometrics->setDistribution(gauss);

  // Set the anthropometric prior.
  fitter.setAnthropometricPrior(anthropometrics, 0.0);
  fitter.setExplicitHeightPrior(heightM, 0.0);

  // Create the error reports.
  // -------------------------
  std::vector<std::shared_ptr<MarkersErrorReport>> reports;
  for (int i = 0; i < (int)markerObservationTrials.size(); i++)
  {
    std::shared_ptr<MarkersErrorReport> report
        = fitter.generateDataErrorsReport(
            markerObservationTrials[i], 1.0 / (s_t)framesPerSecond[i]);
    for (std::string& warning : report->warnings)
    {
      std::cout << "DATA WARNING: " << warning << std::endl;
    }
    markerObservationTrials[i] = report->markerObservationsAttemptedFixed;
    reports.push_back(report);
  }

  // Run the MarkerFitter.
  // ---------------------
  std::vector<MarkerInitialization> markerInit
      = fitter.runMultiTrialKinematicsPipeline(
          markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(5)
              .setMaxTimestepsToUseForMultiTrialScaling(4000),
          150);

  // Create the final kinematics report.
  // -----------------------------------
  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      markerInit[0].updatedMarkerMap,
      markerInit[0].poses,
      markerObservationTrials[0],
      anthropometrics);

  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  std::cout << "Final anthropometric values: " << std::endl;
  anthropometrics->debugValues(standard.skeleton);

  // Save all the results to files.
  // -----------------------------
  std::vector<std::vector<s_t>> timestamps;
  for (int i = 0; i < (int)markerInit.size(); i++)
  {
    timestamps.emplace_back();
    for (int t = 0; t < markerInit[i].poses.cols(); t++)
    {
      timestamps[i].push_back((s_t)t * (1.0 / framesPerSecond[i]));
    }
  }

  for (int i = 0; i < (int)markerInit.size(); i++)
  {
    std::cout << "Saving IK Mot " << i << std::endl;
    OpenSimParser::saveMot(
        standard.skeleton,
        "./_ik" + std::to_string(i) + ".mot",
        timestamps[i],
        markerInit[i].poses);
    std::cout << "Saving GRF Mot " << i << std::endl;
    OpenSimParser::saveRawGRFMot(
        "./_grf" + std::to_string(i) + ".mot", timestamps[i], forcePlates[i]);
    std::cout << "Saving TRC " << i << std::endl;
    std::cout << "timestamps[i]: " << timestamps[i].size() << std::endl;
    std::cout << "markerObservationTrials[i]: "
              << markerObservationTrials[i].size() << std::endl;
    OpenSimParser::saveTRC(
        "./_markers" + std::to_string(i) + ".trc",
        timestamps[i],
        markerObservationTrials[i]);
  }

  std::vector<std::string> markerNames;
  for (auto& pair : standard.markersMap)
  {
    markerNames.push_back(pair.first);
  }
  std::cout << "Saving OpenSim IK XML" << std::endl;
  OpenSimParser::saveOsimInverseKinematicsXMLFile(
      "trial",
      markerNames,
      "Models/optimized_scale_and_markers.osim",
      "./test.trc",
      "_ik_by_opensim.mot",
      "./_ik_setup.xml");
  // TODO: remove me
  for (int i = 0; i < (int)markerInit.size(); i++)
  {
    std::cout << "Saving OpenSim ID Forces " << i << " XML" << std::endl;
    OpenSimParser::saveOsimInverseDynamicsRawForcesXMLFile(
        "test_name",
        standard.skeleton,
        markerInit[i].poses,
        forcePlates[i],
        "name_grf.mot",
        "./_external_forces.xml");
  }
  std::cout << "Saving OpenSim ID XML" << std::endl;
  OpenSimParser::saveOsimInverseDynamicsXMLFile(
      "trial",
      "Models/optimized_scale_and_markers.osim",
      "./_ik.mot",
      "./_external_forces.xml",
      "_id.sto",
      "_id_body_forces.sto",
      "./_id_setup.xml",
      0,
      2);

  Eigen::VectorXs pose = Eigen::VectorXs::Zero(standard.skeleton->getNumDofs());
  s_t gotHeight = standard.skeleton->getHeight(pose);
  std::cout << "Target height: " << heightM << "m" << std::endl;
  std::cout << "Final height: " << gotHeight << "m" << std::endl;

  if (saveGUI)
  {
    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "Frames per second: " << framesPerSecond[0] << std::endl;
    std::cout << "Number of force plates: " << forcePlates.size() << std::endl;
    std::vector<std::map<std::string, Eigen::Vector3s>> accObs;
    std::vector<std::map<std::string, Eigen::Vector3s>> gyroObs;
    fitter.saveTrajectoryAndMarkersToGUI(
        "../../../javascript/src/data/movement2.bin",
        markerInit[0],
        markerObservationTrials[0],
        accObs,
        gyroObs,
        framesPerSecond[0],
        forcePlates[0]);
  }

  auto results = std::tuple<MarkerInitialization, IKErrorReport, OpenSimFile>(
      markerInit[0], finalKinematicsReport, standard);
  return results;
}

void testSubject(const std::string& subject, const s_t& height, const s_t& mass)
{

  std::cout << "Testing " << subject << "..." << std::endl;

  // Run MarkerFitter.
  // -----------------
  std::string prefix = "dart://sample/regression/Arnold2013Synthetic/";
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;
  trcFiles.push_back(prefix + subject + "/trials/walk2/markers.trc");
  grfFiles.push_back(prefix + subject + "/trials/walk2/grf.mot");
  const auto tuple = runMarkerFitter(
      prefix + "unscaled_generic.osim",
      trcFiles,
      grfFiles,
      mass,
      height,
      "male",
      true);
  const auto markerInit = std::get<0>(tuple);
  const auto finalKinematicsReport = std::get<1>(tuple);
  const auto osimFile = std::get<2>(tuple);

  // Load ground-truth results.
  // --------------------------
  auto goldOsim
      = OpenSimParser::parseOsim(prefix + subject + "/" + subject + ".osim");
  auto goldIK = OpenSimParser::loadMot(
      goldOsim.skeleton, prefix + subject + "/coordinates.sto");

  // Compare poses.
  // --------------
  auto poses = markerInit.poses;
  auto goldPoses = goldIK.poses;
  s_t totalError = 0.0;
  Eigen::VectorXs averagePerDofError = Eigen::VectorXs::Zero(poses.rows());
  for (int i = 0; i < poses.cols(); i++)
  {
    Eigen::VectorXs diff = poses.col(i) - goldPoses.col(i);
    totalError += diff.cwiseAbs().sum() / diff.size();
    averagePerDofError += diff.cwiseAbs();
  }
  s_t averagePoseError = totalError / (s_t)poses.cols();
  averagePerDofError /= (s_t)poses.rows();
  if (averagePoseError >= 0.03)
  {
    std::cout << "Average pose error norm " << averagePoseError << " > 0.01"
              << std::endl;
    for (int i = 0; i < averagePerDofError.size(); i++)
    {
      std::cout << "  " << osimFile.skeleton->getDof(i)->getName() << ": "
                << averagePerDofError(i) << std::endl;
    }
  }
  EXPECT_LE(averagePoseError, 0.03);

  // Marker errors.
  // --------------
  EXPECT_LE(finalKinematicsReport.averageRootMeanSquaredError, 0.01);
  EXPECT_LE(finalKinematicsReport.averageMaxError, 0.02);

  // Joint centers.
  // --------------
  auto skeleton = osimFile.skeleton;
  auto joints = skeleton->getJoints();
  auto goldSkeleton = goldOsim.skeleton;
  auto goldJoints = goldSkeleton->getJoints();
  Eigen::VectorXs avgJointError
      = Eigen::VectorXs::Zero(skeleton->getNumJoints());
  for (int i = 0; i < poses.cols(); i++)
  {
    skeleton->setPositions(poses.col(i));
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
  avgJointError /= poses.cols();
  for (int j = 0; j < skeleton->getNumJoints(); j++)
  {
    std::cout << "Joint " << skeleton->getJoint(j)->getName()
              << " average center-estimate error: " << avgJointError(j) << "m"
              << std::endl;
  }
  s_t averageJointCenterError = avgJointError.mean();
  EXPECT_LE(averageJointCenterError, 0.02);

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
  EXPECT_LE(averageBodyScaleError, 0.01);
}

// Currently, these tests are timing out CI, which is quite unfortunate. For
// now, disabling these tests. We should re-enable them once we have a CI system
// that can handle longer tests.

// #define SLOW_REGRESSION_TESTS

//==============================================================================
#ifdef SLOW_REGRESSION_TESTS
TEST(Arnold2013Synthetic, SUBJECT_01)
{
  testSubject("subject01", 1.808, 72.84);
}
#endif

//==============================================================================
#ifdef SLOW_REGRESSION_TESTS
TEST(Arnold2013Synthetic, SUBJECT_02)
{
  testSubject("subject02", 1.853, 76.48);
}
#endif

//==============================================================================
#ifdef SLOW_REGRESSION_TESTS
TEST(Arnold2013Synthetic, SUBJECT_04)
{
  testSubject("subject04", 1.801, 80.3);
}
#endif

//==============================================================================
#ifdef SLOW_REGRESSION_TESTS
TEST(Arnold2013Synthetic, SUBJECT_18)
{
  testSubject("subject18", 1.775, 64.09);
}
#endif

//==============================================================================
#ifdef SLOW_REGRESSION_TESTS
TEST(Arnold2013Synthetic, SUBJECT_19)
{
  testSubject("subject19", 1.79, 68.5);
}
#endif