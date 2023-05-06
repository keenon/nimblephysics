#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>
#include <tinyxml2.h>

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
std::vector<MarkerInitialization> runMarkerFitter(
    std::string modelPath,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    s_t massKg,
    s_t heightM,
    std::string sex,
    bool saveGUI = false) {

  // Initialize data structures.
  // ---------------------------
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlates;

  // Fill marker and GRF data structures.
  // ------------------------------------
  assert(trcFiles.size() == grfFiles.size());
  for (int itrial = 0; itrial < (int)trcFiles.size(); itrial++) {
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
  if (standard.skeleton->getBodyNode("hand_r") != nullptr) {
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
  for (auto& pair : standard.markersMap) {
    markerList.push_back(pair.second);
  }

  for (auto marker : standard.anatomicalMarkers) {
    std::cout << "Anatomical marker: " << marker << std::endl;
  }

  for (auto marker : standard.trackingMarkers) {
    std::cout << "Tracking marker: " << marker << std::endl;
  }

  for (auto marker : standard.markersMap) {
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
  if (sex == "male") {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
        cols,
        0.001); // mm -> m
  } else if (sex == "female") {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
        cols,
        0.001); // mm -> m
  } else {
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
  for (int i = 0; i < (int)markerObservationTrials.size(); i++) {
    std::shared_ptr<MarkersErrorReport> report
        = fitter.generateDataErrorsReport(
            markerObservationTrials[i], 1.0 / (s_t)framesPerSecond[i]);
    for (std::string& warning : report->warnings) {
      std::cout << "DATA WARNING: " << warning << std::endl;
    }
    markerObservationTrials[i] = report->markerObservationsAttemptedFixed;
    reports.push_back(report);
  }

  // Run the MarkerFitter.
  // ---------------------
  std::vector<MarkerInitialization> results
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
      results[0].updatedMarkerMap,
      results[0].poses,
      markerObservationTrials[0],
      anthropometrics);

  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  std::cout << "Final anthropometric values: " << std::endl;
  anthropometrics->debugValues(standard.skeleton);

  std::cout << "Saving marker error report..." << std::endl;
  finalKinematicsReport.saveCSVMarkerErrorReport(
      "./ik_marker_errors.csv");

  // Save all the results to files.
  // -----------------------------
  std::vector<std::vector<s_t>> timestamps;
  for (int i = 0; i < (int)results.size(); i++) {
    timestamps.emplace_back();
    for (int t = 0; t < results[i].poses.cols(); t++) {
      timestamps[i].push_back((s_t)t * (1.0 / framesPerSecond[i]));
    }
  }

  for (int i = 0; i < (int)results.size(); i++) {
    std::cout << "Saving IK Mot " << i << std::endl;
    OpenSimParser::saveMot(
        standard.skeleton,
        "./_ik" + std::to_string(i) + ".mot",
        timestamps[i],
        results[i].poses);
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
  for (auto& pair : standard.markersMap) {
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
  for (int i = 0; i < (int)results.size(); i++) {
    std::cout << "Saving OpenSim ID Forces " << i << " XML" << std::endl;
    OpenSimParser::saveOsimInverseDynamicsRawForcesXMLFile(
        "test_name",
        standard.skeleton,
        results[i].poses,
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
        results[0],
        markerObservationTrials[0],
        accObs,
        gyroObs,
        framesPerSecond[0],
        forcePlates[0]);
  }

  return results;
}

//==============================================================================
TEST(Arnold2013Synthetic, JointCenters) {

  std::string prefix = "dart://sample/regression/Arnold2013Synthetic/";
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;
  trcFiles.push_back(prefix + "subject01/trials/walk2/markers.trc");
  grfFiles.push_back(prefix + "subject01/trials/walk2/grf.mot");

  auto results = runMarkerFitter(
      prefix + "unscaled_generic.osim",
      trcFiles,
      grfFiles,
      72.84,
      1.808,
      "male",
      true);

  MarkerInitialization result = results[0];
  auto poses = result.poses;
  std::cout << "Final poses size: [" << poses.rows() << ", " << poses.cols()
            << "]" << std::endl;

  // Load ground-truth results.
  // --------------------------
  auto goldOsim = OpenSimParser::parseOsim(
      prefix + "subject01/subject01.osim");
  auto goldIK = OpenSimParser::loadMot(
      goldOsim.skeleton, prefix + "subject01/coordinates.sto");
  auto goldPoses = goldIK.poses;
  std::cout << "Gold poses size: [" << goldPoses.rows() << ", "
            << goldPoses.cols() << "]" << std::endl;

  // Compute the RMS difference between the columns in poses and goldPoses
  // (which are the same size)
  s_t totalError = 0.0;
  for (int i = 0; i < poses.cols(); i++)
  {
    Eigen::VectorXs diff = poses.col(i) - goldPoses.col(i);
    totalError += diff.squaredNorm();
  }
  s_t rmsError = std::sqrt(totalError / poses.cols());
  std::cout << "RMS Error: " << rmsError << std::endl;
  EXPECT_TRUE(rmsError < 0.01);

}

//==============================================================================
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
