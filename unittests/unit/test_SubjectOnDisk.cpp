#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/biomechanics/SubjectOnDisk.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/AccelerationSmoother.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/MJCFExporter.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

bool testWriteSubjectToDisk(std::string outputFilePath)
{
  srand(42);

  // Global config for the test
  int dofs = 24;
  int numTrials = 4;

  ///////////////////////////////////////////////////////////////
  // 1. Declare the data we'll need
  ///////////////////////////////////////////////////////////////

  // 1.1. Header data
  std::vector<ProcessingPassType> processingPasses;
  std::vector<s_t> processingPassCutoffs;
  std::vector<int> processingPassOrders;
  std::vector<std::string> openSimFileTexts;
  std::vector<std::string> customValueNames;
  std::vector<std::string> groundForceBodies;
  std::string originalHref
      = "https://dev.addbiomechanics.org/data/"
        "35e1c7ca-cc58-457e-bfc5-f6161cc7278b/SprinterTest";
  std::string originalNotes
      = "A sprinter originally recorded by blah blah blah. Please cite bibtex "
        "{} blah blah blah if you use this data specifically.";
  std::string biologicalSex = "unknown";
  s_t heightM = 1.8;
  s_t massKg = 80;
  int age = 30;
  std::vector<std::string> markerNames;

  // 1.2. Raw sensor data, per trial
  std::vector<s_t> trialTimesteps;
  std::vector<int> trialLengths;
  std::vector<std::string> trialNames;
  std::vector<std::string> trialOriginalNames;
  std::vector<int> trialSplitIndex;
  std::vector<int> trialNumPasses;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservations;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      accObservations;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      gyroObservations;
  std::vector<std::vector<std::map<std::string, Eigen::VectorXs>>>
      emgObservations;
  std::vector<std::map<int, Eigen::VectorXs>> exoObservations;
  std::vector<std::vector<ForcePlate>> forcePlateTrials;
  std::vector<std::vector<MissingGRFReason>> missingGRFReasonTrials;
  std::vector<std::vector<Eigen::MatrixXs>> customValueTrials;

  // 1.3. Per pass header data
  std::vector<std::vector<std::vector<bool>>> dofPositionObservedTrialPasses;
  std::vector<std::vector<std::vector<bool>>>
      dofVelocityFiniteDifferencedTrialPasses;
  std::vector<std::vector<std::vector<bool>>>
      dofAccelerationFiniteDifferencedTrialPasses;
  // 1.4. Per pass frame data
  std::vector<std::vector<Eigen::MatrixXs>> poseTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> velTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> accTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> tauTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> groundBodyWrenchTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> groundBodyCopTorqueForceTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> comPosesTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> comVelsTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> comAccsTrialPasses;
  // 1.5. Per pass results data
  std::vector<std::vector<std::vector<s_t>>> linearResidualTrialPasses;
  std::vector<std::vector<std::vector<s_t>>> angularResidualTrialPasses;
  std::vector<std::vector<std::vector<s_t>>> markerRMSTrialPasses;
  std::vector<std::vector<std::vector<s_t>>> markerMaxTrialPasses;

  ///////////////////////////////////////////////////////////////
  // 2. Generate some fake data
  ///////////////////////////////////////////////////////////////

  // 2.1. Header data
  processingPasses.push_back(ProcessingPassType::kinematics);
  processingPassCutoffs.push_back(-1);
  processingPassOrders.push_back(-1);
  processingPasses.push_back(ProcessingPassType::dynamics);
  processingPassCutoffs.push_back(-1);
  processingPassOrders.push_back(-1);
  processingPasses.push_back(ProcessingPassType::lowPassFilter);
  processingPassCutoffs.push_back(25);
  processingPassOrders.push_back(3);

  openSimFileTexts.push_back("Kinematics_test");
  openSimFileTexts.push_back("Dynamics_test");
  openSimFileTexts.push_back("Butterworth_test");

  customValueNames.push_back("stretch_sensor");

  groundForceBodies.push_back("calcn_r");
  groundForceBodies.push_back("calcn_l");

  for (int i = 0; i < 10; i++)
  {
    markerNames.push_back("marker_" + std::to_string(i));
  }

  std::vector<int> exoDofs;
  for (int j = 4; j < 6; j++)
  {
    exoDofs.push_back(j);
  }

  // 2.2. Raw sensor data, per trial
  for (int trial = 0; trial < numTrials; trial++)
  {
    trialNames.push_back("trial_" + std::to_string(trial));
    trialOriginalNames.push_back(
        "trial_" + std::to_string(trial) + "_original");
    trialSplitIndex.push_back(trial % 2);
    trialLengths.push_back(100 + (rand() % 50));
    trialTimesteps.push_back(0.01);
    trialNumPasses.push_back((trial % (processingPasses.size() - 1)) + 1);

    std::vector<std::map<std::string, Eigen::Vector3s>> markerTrial;
    std::vector<std::map<std::string, Eigen::Vector3s>> accTrial;
    std::vector<std::map<std::string, Eigen::Vector3s>> gyroTrial;
    std::vector<std::map<std::string, Eigen::VectorXs>> emgTrial;
    for (int t = 0; t < trialLengths[trial]; t++)
    {
      std::map<std::string, Eigen::Vector3s> markers;
      std::map<std::string, Eigen::Vector3s> accs;
      std::map<std::string, Eigen::Vector3s> gyros;
      std::map<std::string, Eigen::VectorXs> emg;
      for (int j = 0; j < t % markerNames.size(); j++)
      {
        markers[markerNames[j]] = Eigen::Vector3s::Random();
        accs[markerNames[j]] = Eigen::Vector3s::Random();
        gyros[markerNames[j]] = Eigen::Vector3s::Random();
        emg[markerNames[j]] = Eigen::VectorXs::Random(5);
      }
      markerTrial.push_back(markers);
      accTrial.push_back(accs);
      gyroTrial.push_back(gyros);
      emgTrial.push_back(emg);
    }
    markerObservations.push_back(markerTrial);
    accObservations.push_back(accTrial);
    gyroObservations.push_back(gyroTrial);
    emgObservations.push_back(emgTrial);

    std::map<int, Eigen::VectorXs> exoTorques;
    if (trial % 2 == 0)
    {
      for (int j : exoDofs)
      {
        exoTorques[j] = Eigen::VectorXs::Random(trialLengths[trial]);
      }
    }
    exoObservations.push_back(exoTorques);

    std::vector<ForcePlate> forcePlates;
    for (int j = 0; j < 4; j++)
    {
      ForcePlate newForcePlate = ForcePlate();
      newForcePlate.worldOrigin = Eigen::Vector3s::Random();
      for (int t = 0; t < trialLengths[trial]; t++)
      {
        newForcePlate.timestamps.push_back(t * trialTimesteps[trial]);
        newForcePlate.centersOfPressure.push_back(Eigen::Vector3s::Random());
        newForcePlate.moments.push_back(Eigen::Vector3s::Random());
        newForcePlate.forces.push_back(Eigen::Vector3s::Random());
      }
      for (int c = 0; c < 4; c++)
      {
        newForcePlate.corners.push_back(Eigen::Vector3s::Random());
      }

      forcePlates.push_back(newForcePlate);
    }
    forcePlateTrials.push_back(forcePlates);

    std::vector<MissingGRFReason> grfReason;
    for (int t = 0; t < trialLengths[trial]; t++)
    {
      if (t % 10 == 0)
      {
        grfReason.push_back(
            MissingGRFReason::measuredGrfZeroWhenAccelerationNonZero);
      }
      else
      {
        grfReason.push_back(MissingGRFReason::notMissingGRF);
      }
    }
    missingGRFReasonTrials.push_back(grfReason);
  }

  // 2.3. Per pass IK data
  for (int trial = 0; trial < numTrials; trial++)
  {
    poseTrialPasses.emplace_back();
    velTrialPasses.emplace_back();
    accTrialPasses.emplace_back();
    tauTrialPasses.emplace_back();
    groundBodyWrenchTrialPasses.emplace_back();
    groundBodyCopTorqueForceTrialPasses.emplace_back();
    comPosesTrialPasses.emplace_back();
    comVelsTrialPasses.emplace_back();
    comAccsTrialPasses.emplace_back();
    linearResidualTrialPasses.emplace_back();
    angularResidualTrialPasses.emplace_back();
    markerRMSTrialPasses.emplace_back();
    markerMaxTrialPasses.emplace_back();
    dofPositionObservedTrialPasses.emplace_back();
    dofVelocityFiniteDifferencedTrialPasses.emplace_back();
    dofAccelerationFiniteDifferencedTrialPasses.emplace_back();

    for (int pass = 0; pass < trialNumPasses[trial]; pass++)
    {
      // 2.3. Per pass header data
      std::vector<bool> positionObserved;
      std::vector<bool> velocityFiniteDifferenced;
      std::vector<bool> accelerationFiniteDifferenced;
      for (int i = 0; i < dofs; i++)
      {
        positionObserved.push_back(i % 2 == 0);
        velocityFiniteDifferenced.push_back(i % 3 == 0);
        accelerationFiniteDifferenced.push_back(i % 4 == 0);
      }
      dofPositionObservedTrialPasses[trial].push_back(positionObserved);
      dofVelocityFiniteDifferencedTrialPasses[trial].push_back(
          velocityFiniteDifferenced);
      dofAccelerationFiniteDifferencedTrialPasses[trial].push_back(
          accelerationFiniteDifferenced);

      // 2.4. Per pass frame data
      poseTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(dofs, trialLengths[trial]));
      velTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(dofs, trialLengths[trial]));
      accTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(dofs, trialLengths[trial]));
      tauTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(dofs, trialLengths[trial]));
      groundBodyWrenchTrialPasses[trial].push_back(Eigen::MatrixXs::Random(
          6 * groundForceBodies.size(), trialLengths[trial]));
      groundBodyCopTorqueForceTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(
              9 * groundForceBodies.size(), trialLengths[trial]));
      comPosesTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(3, trialLengths[trial]));
      comVelsTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(3, trialLengths[trial]));
      comAccsTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(3, trialLengths[trial]));

      // 2.5. Per pass results data
      std::vector<s_t> linearResiduals;
      std::vector<s_t> angularResiduals;
      std::vector<s_t> markerRMS;
      std::vector<s_t> markerMax;
      for (int t = 0; t < trialLengths[trial]; t++)
      {
        linearResiduals.push_back(rand() % 1000);
        angularResiduals.push_back(rand() % 1000);
        markerRMS.push_back(rand() % 1000);
        markerMax.push_back(rand() % 1000);
      }
      linearResidualTrialPasses[trial].push_back(linearResiduals);
      angularResidualTrialPasses[trial].push_back(angularResiduals);
      markerRMSTrialPasses[trial].push_back(markerRMS);
      markerMaxTrialPasses[trial].push_back(markerMax);
    }

    std::vector<Eigen::MatrixXs> trialCustomValues;
    trialCustomValues.push_back(
        Eigen::MatrixXs::Random(customValueNames.size(), trialLengths[trial]));
    customValueTrials.push_back(trialCustomValues);
  }

  std::vector<std::string> subjectTags;
  for (int i = 0; i < 10; i++)
  {
    subjectTags.push_back("subject_tag_" + std::to_string(i));
  }
  std::vector<std::vector<std::string>> trialTags;
  for (int trial = 0; trial < poseTrialPasses.size(); trial++)
  {
    std::vector<std::string> trialTag;
    for (int i = 0; i < trial + 3; i++)
    {
      trialTag.push_back(
          "trial_" + std::to_string(trial) + "_tag_" + std::to_string(i));
    }
    trialTags.push_back(trialTag);
  }

  ///////////////////////////////////////////////////////////////
  // 3. Write the data to the SubjectOnDisk B3D file
  ///////////////////////////////////////////////////////////////

  // 3.1. Header data
  SubjectOnDiskHeader header;
  for (int i = 0; i < processingPasses.size(); i++)
  {
    header.addProcessingPass()
        .setProcessingPassType(processingPasses[i])
        .setLowpassCutoffFrequency(processingPassCutoffs[i])
        .setLowpassFilterOrder(processingPassOrders[i])
        .setOpenSimFileText(openSimFileTexts[i]);
  }
  header.setNumDofs(dofs);
  header.setCustomValueNames(customValueNames);
  header.setGroundForceBodies(groundForceBodies);
  header.setHref(originalHref);
  header.setNotes(originalNotes);
  header.setBiologicalSex(biologicalSex);
  header.setHeightM(heightM);
  header.setMassKg(massKg);
  header.setAgeYears(age);
  header.setSubjectTags(subjectTags);

  // 3.2. Per trial data
  for (int trial = 0; trial < numTrials; trial++)
  {
    auto& trialData = header.addTrial();
    trialData.setTimestep(trialTimesteps[trial]);
    trialData.setName(trialNames[trial]);
    trialData.setOriginalTrialName(trialOriginalNames[trial]);
    trialData.setSplitIndex(trialSplitIndex[trial]);
    trialData.setMarkerObservations(markerObservations[trial]);
    if (accObservations.size() > trial)
    {
      trialData.setAccObservations(accObservations[trial]);
    }
    if (gyroObservations.size() > trial)
    {
      trialData.setGyroObservations(gyroObservations[trial]);
    }
    if (emgObservations.size() > trial)
    {
      trialData.setEmgObservations(emgObservations[trial]);
    }
    if (exoObservations.size() > trial)
    {
      trialData.setExoTorques(exoObservations[trial]);
    }
    trialData.setMissingGRFReason(missingGRFReasonTrials[trial]);
    if (customValueTrials.size() > trial)
    {
      trialData.setCustomValues(customValueTrials[trial]);
    }
    if (trialTags.size() > trial)
    {
      trialData.setTrialTags(trialTags[trial]);
    }
    if (forcePlateTrials.size() > trial)
    {
      trialData.setForcePlates(forcePlateTrials[trial]);
    }

    for (int pass = 0; pass < trialNumPasses[trial]; pass++)
    {
      auto& passData = trialData.addPass();

      // 3.3. Per pass header data
      passData.setDofPositionsObserved(
          dofPositionObservedTrialPasses[trial][pass]);
      passData.setDofVelocitiesFiniteDifferenced(
          dofVelocityFiniteDifferencedTrialPasses[trial][pass]);
      passData.setDofAccelerationFiniteDifferenced(
          dofAccelerationFiniteDifferencedTrialPasses[trial][pass]);

      // 3.4. Per pass frame data
      passData.setPoses(poseTrialPasses[trial][pass]);
      passData.setVels(velTrialPasses[trial][pass]);
      passData.setAccs(accTrialPasses[trial][pass]);
      passData.setTaus(tauTrialPasses[trial][pass]);
      passData.setGroundBodyWrenches(groundBodyWrenchTrialPasses[trial][pass]);
      passData.setGroundBodyCopTorqueForce(
          groundBodyCopTorqueForceTrialPasses[trial][pass]);
      passData.setComPoses(comPosesTrialPasses[trial][pass]);
      passData.setComVels(comVelsTrialPasses[trial][pass]);
      passData.setComAccs(comAccsTrialPasses[trial][pass]);

      // 3.5. Per pass results data
      passData.setLinearResidual(linearResidualTrialPasses[trial][pass]);
      passData.setAngularResidual(angularResidualTrialPasses[trial][pass]);
      passData.setMarkerRMS(markerRMSTrialPasses[trial][pass]);
      passData.setMarkerMax(markerMaxTrialPasses[trial][pass]);
    }
  }

  // 3.6. Actually write out the file!
  SubjectOnDisk::writeB3D(outputFilePath, header);

  ////////////////////////////////////////
  // 4. Test reading the subject back in
  ////////////////////////////////////////

  SubjectOnDisk subject(outputFilePath);

  // 4.1. Header data
  if (subject.getNumProcessingPasses() != processingPasses.size())
  {
    std::cout << "Recovered incorrect number of processing passes!"
              << std::endl;
    return false;
  }
  for (int i = 0; i < processingPasses.size(); i++)
  {
    ProcessingPassType type = subject.getProcessingPassType(i);
    if (type != processingPasses[i])
    {
      std::cout << "Failed to recover correct processing pass type!"
                << std::endl;
      return false;
    }

    s_t frequency = subject.getLowpassCutoffFrequency(i);
    if (frequency != processingPassCutoffs[i])
    {
      std::cout << "Failed to recover correct lowpass cutoff frequency!"
                << std::endl;
      return false;
    }

    int order = subject.getLowpassFilterOrder(i);
    if (order != processingPassOrders[i])
    {
      std::cout << "Failed to recover correct lowpass filter order!"
                << std::endl;
      return false;
    }

    std::string recoveredModel = subject.getOpensimFileText(i);
    if (recoveredModel != openSimFileTexts[i])
    {
      std::cout << "Failed to recover correct OpenSim model!" << std::endl;
      return false;
    }
  }
  if (subject.getNumDofs() != dofs)
  {
    std::cout << "Failed to recover correct number of DOFs!" << std::endl;
    return false;
  }
  if (subject.getCustomValues().size() != customValueNames.size())
  {
    std::cout << "Failed to recover correct number of custom value names!"
              << std::endl;
    return false;
  }
  for (int i = 0; i < subject.getCustomValues().size(); i++)
  {
    if (subject.getCustomValues()[i] != customValueNames[i])
    {
      std::cout << "Failed to recover correct custom value name!" << std::endl;
      return false;
    }
  }
  if (subject.getGroundForceBodies().size() != groundForceBodies.size())
  {
    std::cout << "Failed to recover correct number of ground force bodies!"
              << std::endl;
    return false;
  }
  for (int i = 0; i < subject.getGroundForceBodies().size(); i++)
  {
    if (subject.getGroundForceBodies()[i] != groundForceBodies[i])
    {
      std::cout << "Failed to recover correct ground force body!" << std::endl;
      return false;
    }
  }
  if (subject.getHref() != originalHref)
  {
    std::cout << "Failed to recover href!" << std::endl;
    return false;
  }
  if (subject.getNotes() != originalNotes)
  {
    std::cout << "Failed to recover notes!" << std::endl;
    return false;
  }
  if (subject.getBiologicalSex() != biologicalSex)
  {
    std::cout << "Failed to recover biological sex" << std::endl;
    return false;
  }
  if (subject.getHeightM() != heightM)
  {
    std::cout << "Failed to recover height!" << std::endl;
    return false;
  }
  if (subject.getMassKg() != massKg)
  {
    std::cout << "Failed to recover mass!" << std::endl;
    return false;
  }
  if (subject.getAgeYears() != age)
  {
    std::cout << "Failed to recover age!" << std::endl;
    return false;
  }

  for (int trial = 0; trial < numTrials; trial++)
  {
    // 4.2. Per trial data

    if (subject.getTrialTimestep(trial) != trialTimesteps[trial])
    {
      std::cout << "Failed to recover trial timestep!" << std::endl;
      return false;
    }
    if (subject.getTrialName(trial) != trialNames[trial])
    {
      std::cout << "Failed to recover trial name!" << std::endl;
      return false;
    }
    if (subject.getTrialOriginalName(trial) != trialOriginalNames[trial])
    {
      std::cout << "Failed to recover pre-split name!" << std::endl;
      return false;
    }
    if (subject.getTrialSplitIndex(trial) != trialSplitIndex[trial])
    {
      std::cout << "Failed to recover split index!" << std::endl;
      return false;
    }
    if (subject.getNumForcePlates(trial) != forcePlateTrials[trial].size())
    {
      std::cout << "Failed to recover correct number of force plates!"
                << std::endl;
      return false;
    }
    for (int plate = 0; plate < forcePlateTrials[trial].size(); plate++)
    {
      ForcePlate& original = forcePlateTrials[trial][plate];
      std::vector<Eigen::Vector3s> corners
          = subject.getForcePlateCorners(trial, plate);
      if (original.corners.size() != corners.size())
      {
        std::cout << "Failed to recover correct number of force plate corners!"
                  << std::endl;
        return false;
      }
      for (int c = 0; c < original.corners.size(); c++)
      {
        if (original.corners[c] != corners[c])
        {
          std::cout << "Failed to recover correct force plate corner!"
                    << std::endl;
          return false;
        }
      }
    }

    // 4.2.1. Per trial per frame sensor data

    for (int t = 0; t < trialLengths[trial]; t++)
    {
      std::vector<std::shared_ptr<biomechanics::Frame>> sensorsFrames
          = subject.readFrames(trial, t, 1, true, false);
      if (sensorsFrames.size() < 1)
      {
        std::cout << "Failed to recover sensor frame!" << std::endl;
        return false;
      }
      std::shared_ptr<biomechanics::Frame> sensorsFrame = sensorsFrames[0];

      if (sensorsFrame->markerObservations.size()
          != markerObservations[trial][t].size())
      {
        std::cout << "Failed to recover correct number of marker observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : sensorsFrame->markerObservations)
      {
        if (markerObservations[trial][t].at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct marker observation!"
                    << std::endl;
          return false;
        }
      }

      if (sensorsFrame->accObservations.size()
          != accObservations[trial][t].size())
      {
        std::cout
            << "Failed to recover correct number of accelerometer observations!"
            << std::endl;
        return false;
      }
      for (auto& pair : sensorsFrame->accObservations)
      {
        if (accObservations[trial][t].at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct accelerometer observation!"
                    << std::endl;
          return false;
        }
      }

      if (sensorsFrame->gyroObservations.size()
          != gyroObservations[trial][t].size())
      {
        std::cout << "Failed to recover correct number of gyro observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : sensorsFrame->gyroObservations)
      {
        if (gyroObservations[trial][t].at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct gyro observation!"
                    << std::endl;
          return false;
        }
      }

      if (sensorsFrame->emgSignals.size() != emgObservations[trial][t].size())
      {
        std::cout << "Failed to recover correct number of EMG observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : sensorsFrame->emgSignals)
      {
        if (emgObservations[trial][t].at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct EMG observation!"
                    << std::endl;
          return false;
        }
      }

      if (sensorsFrame->exoTorques.size() != exoDofs.size())
      {
        std::cout << "Failed to recover correct number of exo observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : sensorsFrame->exoTorques)
      {
        // Default to NaN
        s_t exoValue = std::nan("");
        if (exoObservations[trial].count(pair.first)
            && exoObservations[trial].at(pair.first).size() > t)
        {
          exoValue = exoObservations[trial].at(pair.first)(t);
        }
        if ((exoValue == pair.second)
            || (std::isnan(exoValue) && std::isnan(pair.second)))
        {
          // All good
        }
        else
        {
          std::cout << "Failed to recover correct exo observation!"
                    << std::endl;
          return false;
        }
      }

      if (sensorsFrame->missingGRFReason != missingGRFReasonTrials[trial][t])
      {
        std::cout << "Failed to recover correct missing GRF reason!"
                  << std::endl;
        return false;
      }

      if (sensorsFrame->customValues.size() != customValueNames.size())
      {
        std::cout << "Failed to recover correct number of custom values!"
                  << std::endl;
        return false;
      }
      for (int i = 0; i < customValueNames.size(); i++)
      {
        if (sensorsFrame->customValues[i].second
            != customValueTrials[trial][i].col(t))
        {
          std::cout << "Failed to recover correct custom value!" << std::endl;
          return false;
        }
      }
    }
  }

  std::vector<std::string> recoveredTags = subject.getSubjectTags();
  if (recoveredTags.size() != subjectTags.size())
  {
    std::cout << "Failed to recover correct number of subject tags!"
              << std::endl;
    return false;
  }
  for (int i = 0; i < recoveredTags.size(); i++)
  {
    if (recoveredTags[i] != subjectTags[i])
    {
      std::cout << "Failed to recover correct subject tag!" << std::endl;
      return false;
    }
  }
  for (int trial = 0; trial < trialTags.size(); trial++)
  {
    std::vector<std::string> recoveredTags = subject.getTrialTags(trial);
    if (recoveredTags.size() != trialTags[trial].size())
    {
      std::cout << "Failed to recover correct number of trial tags!"
                << std::endl;
      return false;
    }
    for (int i = 0; i < recoveredTags.size(); i++)
    {
      if (recoveredTags[i] != trialTags[trial][i])
      {
        std::cout << "Failed to recover correct trial tag!" << std::endl;
        return false;
      }
    }
  }
  for (int trial = 0; trial < velTrialPasses.size(); trial++)
  {
    for (int pass = 0; pass < trialNumPasses[trial]; pass++)
    {
      std::vector<s_t> maxVels = subject.getTrialMaxJointVelocity(trial, pass);
      for (int t = 0; t < velTrialPasses[trial][pass].cols(); t++)
      {
        EXPECT_NEAR(
            maxVels[t],
            velTrialPasses[trial][pass].col(t).cwiseAbs().maxCoeff(),
            1e-6);
      }
    }
  }

  for (int i = 0; i < 500; i++)
  {
    int trial = rand() % subject.getNumTrials();
    int frame = rand() % subject.getTrialLength(trial);
    int stride = rand() % 4 + 1;

    std::vector<std::shared_ptr<biomechanics::Frame>> readResult
        = subject.readFrames(trial, frame, 5, true, true, stride);

    for (int j = 0; j < readResult.size(); j++)
    {
      int timestep = frame + j * stride;
      std::map<std::string, Eigen::Vector3s> originalMarkers
          = markerObservations[trial][timestep];
      if (readResult[j]->markerObservations.size() != originalMarkers.size())
      {
        std::cout << "Failed to recover correct number of marker observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : readResult[j]->markerObservations)
      {
        if (originalMarkers.at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct marker observation!"
                    << std::endl;
          return false;
        }
      }
      std::map<std::string, Eigen::Vector3s> originalGyros
          = gyroObservations[trial][timestep];
      if (readResult[j]->gyroObservations.size() != originalGyros.size())
      {
        std::cout << "Failed to recover correct number of gyro observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : readResult[j]->gyroObservations)
      {
        if (originalGyros.at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct gyro observation!"
                    << std::endl;
          return false;
        }
      }
      std::map<std::string, Eigen::Vector3s> originalAccs
          = accObservations[trial][timestep];
      if (readResult[j]->accObservations.size() != originalAccs.size())
      {
        std::cout
            << "Failed to recover correct number of accelerometer observations!"
            << std::endl;
        return false;
      }
      for (auto& pair : readResult[j]->accObservations)
      {
        if (originalAccs.at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct accelerometer observation!"
                    << std::endl;
          return false;
        }
      }
      std::map<std::string, Eigen::VectorXs> originalEmg
          = emgObservations[trial][timestep];
      if (readResult[j]->emgSignals.size() != originalEmg.size())
      {
        std::cout << "Failed to recover correct number of EMG observations!"
                  << std::endl;
        return false;
      }
      for (auto& pair : readResult[j]->emgSignals)
      {
        if (originalEmg.at(pair.first) != pair.second)
        {
          std::cout << "Failed to recover correct EMG observation!"
                    << std::endl;
          return false;
        }
      }
      if (forcePlateTrials[trial].size()
          != readResult[j]->rawForcePlateForces.size())
      {
        std::cout << "Failed to recover correct number of raw force values!"
                  << std::endl;
        return false;
      }
      for (int f = 0; f < forcePlateTrials[trial].size(); f++)
      {
        if (forcePlateTrials[trial][f].forces[timestep]
            != readResult[j]->rawForcePlateForces[f])
        {
          std::cout << "Failed to recover correct raw force value!"
                    << std::endl;
          return false;
        }
        if (forcePlateTrials[trial][f].moments[timestep]
            != readResult[j]->rawForcePlateTorques[f])
        {
          std::cout << "Failed to recover correct raw torque value!"
                    << std::endl;
          return false;
        }
        if (forcePlateTrials[trial][f].centersOfPressure[timestep]
            != readResult[j]->rawForcePlateCenterOfPressures[f])
        {
          std::cout << "Failed to recover correct raw center of pressure value!"
                    << std::endl;
          return false;
        }
      }
    }

    for (auto& frame : readResult)
    {
      std::cout << "Checking frame " << frame->trial << ":" << frame->t
                << std::endl;

      if (abs(subject.getTrialTimestep(trial) - trialTimesteps[frame->trial])
          > 1e-8)
      {
        std::cout << "dt not recovered" << std::endl;
        return false;
      }

      if (frame->missingGRFReason != missingGRFReasonTrials[trial][frame->t])
      {
        std::cout << "missing GRF reason not recovered" << std::endl;
        return false;
      }

      if (frame->processingPasses.size() != trialNumPasses[trial])
      {
        std::cout << "Wrong number of processing passes recovered" << std::endl;
        return false;
      }

      for (int pass = 0; pass < trialNumPasses[trial]; pass++)
      {
        Eigen::VectorXs originalPos
            = poseTrialPasses[frame->trial][pass].col(frame->t);
        if (!equals(originalPos, frame->processingPasses[pass].pos, 1e-8))
        {
          std::cout << "Pos not recovered" << std::endl;
          return false;
        }
        Eigen::VectorXs originalVel
            = velTrialPasses[frame->trial][pass].col(frame->t);
        if (!equals(originalVel, frame->processingPasses[pass].vel, 1e-8))
        {
          std::cout << "Vel not recovered" << std::endl;
          return false;
        }
        Eigen::VectorXs originalAcc
            = accTrialPasses[frame->trial][pass].col(frame->t);
        if (!equals(originalAcc, frame->processingPasses[pass].acc, 1e-8))
        {
          std::cout << "Acc not recovered" << std::endl;
          return false;
        }
        Eigen::VectorXs originalTau
            = tauTrialPasses[frame->trial][pass].col(frame->t);
        if (!equals(originalTau, frame->processingPasses[pass].tau, 1e-8))
        {
          std::cout << "Tau not recovered" << std::endl;
          return false;
        }
        for (int b = 0; b < subject.getGroundForceBodies().size(); b++)
        {
          Eigen::Vector6s originalWrench
              = groundBodyWrenchTrialPasses[frame->trial][pass]
                    .col(frame->t)
                    .segment<6>(b * 6);
          Eigen::Vector6s recoveredWrench
              = frame->processingPasses[pass].groundContactWrenches.segment<6>(
                  b * 6);
          if (!equals(originalWrench, recoveredWrench, 1e-8))
          {
            std::cout << "Body wrench not recovered" << std::endl;
            return false;
          }
          Eigen::Vector3s originalCoP
              = groundBodyCopTorqueForceTrialPasses[frame->trial][pass]
                    .col(frame->t)
                    .segment<3>(b * 9);
          Eigen::Vector3s recoveredCoP
              = frame->processingPasses[pass]
                    .groundContactCenterOfPressure.segment<3>(b * 3);
          if (!equals(originalCoP, recoveredCoP, 1e-8))
          {
            std::cout << "GRF CoP not recovered" << std::endl;
            return false;
          }
          Eigen::Vector3s originalTau
              = groundBodyCopTorqueForceTrialPasses[frame->trial][pass]
                    .col(frame->t)
                    .segment<3>((b * 9) + 3);
          Eigen::Vector3s recoveredTau
              = frame->processingPasses[pass].groundContactTorque.segment<3>(
                  b * 3);
          if (!equals(originalTau, recoveredTau, 1e-8))
          {
            std::cout << "GRF Tau not recovered" << std::endl;
            return false;
          }
          Eigen::Vector3s originalF
              = groundBodyCopTorqueForceTrialPasses[frame->trial][pass]
                    .col(frame->t)
                    .segment<3>((b * 9) + 6);
          Eigen::Vector3s recoveredF
              = frame->processingPasses[pass].groundContactForce.segment<3>(
                  b * 3);
          if (!equals(originalF, recoveredF, 1e-8))
          {
            std::cout << "GRF Force not recovered" << std::endl;
            return false;
          }
        }
      }
    }
  }

  return true;
}

#ifdef ALL_TESTS
TEST(SubjectOnDisk, WRITE_THEN_READ)
{
  std::string path = "./testSubject.bin";

  srand(42);

  for (int i = 0; i < 10; i++)
  {
    bool success = testWriteSubjectToDisk(path);
    EXPECT_TRUE(success);

    if (!success)
    {
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(SubjectOnDisk, MINIMAL_WRITE_READ)
{
  srand(42);

  std::vector<std::string> markerNames;
  for (int i = 0; i < 1; i++)
  {
    markerNames.push_back("marker_" + std::to_string(i));
  }

  std::string path = "./testSubject.bin";
  SubjectOnDiskHeader header;
  header.setAgeYears(30);
  for (int trial = 0; trial < 2; trial++)
  {
    auto& trialData = header.addTrial();
    // trialData.setName("test");

    std::vector<std::map<std::string, Eigen::Vector3s>> markerTrial;
    for (int t = 0; t < 2; t++)
    {
      std::map<std::string, Eigen::Vector3s> markers;
      for (int j = 0; j < markerNames.size(); j++)
      {
        markers[markerNames[j]] = Eigen::Vector3s::Random();
      }
      markerTrial.push_back(markers);
    }
    trialData.setMarkerObservations(markerTrial);

    // for (int pass = 0; pass < 2; pass++)
    // {
    //   auto& passData = trialData.addPass();
    //   passData.setPoses(Eigen::MatrixXs::Random(3, 3));
    //   passData.setVels(Eigen::MatrixXs::Random(3, 3));
    //   passData.setAccs(Eigen::MatrixXs::Random(3, 3));
    // }
  }
  // Write
  SubjectOnDisk::writeB3D(path, header);
  // Read
  SubjectOnDisk subject(path);
}
#endif