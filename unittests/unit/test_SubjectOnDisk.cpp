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

bool testWriteSubjectToDisk(
    std::string outputFilePath,
    std::string openSimFilePath,
    std::vector<std::string> motFiles,
    std::vector<std::string> grfFiles,
    int limitTrialSizes = -1,
    int trialStartOffset = 0)
{
  srand(42);

  std::vector<std::vector<Eigen::MatrixXs>> poseTrialPasses;
  std::vector<std::vector<std::vector<ForcePlate>>> forcePlateTrialPasses;
  std::vector<s_t> timesteps;
  std::vector<std::vector<Eigen::MatrixXs>> velTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> accTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> tauTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> groundBodyWrenchTrialPasses;
  std::vector<std::vector<Eigen::MatrixXs>> groundBodyCopTorqueForceTrialPasses;
  std::vector<std::vector<bool>> probablyMissingGRFData;
  std::vector<std::vector<MissingGRFReason>> missingGRFReason;
  std::vector<std::vector<bool>> passDofPositionObserved;
  std::vector<std::vector<bool>> passDofVelocityFiniteDifferenced;
  std::vector<std::vector<bool>> passDofAccelerationFiniteDifferenced;
  std::vector<std::vector<s_t>> residualNorms;
  std::vector<std::vector<Eigen::MatrixXs>> trialPassComPoses;
  std::vector<std::vector<Eigen::MatrixXs>> trialPassComVelocities;
  std::vector<std::vector<Eigen::MatrixXs>> trialPassComAccelerations;
  std::vector<std::string> customValueNames;
  std::vector<std::vector<Eigen::MatrixXs>> customValueTrials;
  std::vector<ProcessingPassType> processingPasses;
  processingPasses.push_back(ProcessingPassType::kinematics);
  processingPasses.push_back(ProcessingPassType::dynamics);
  processingPasses.push_back(ProcessingPassType::lowPassFilter);

  customValueNames.push_back("exo_tau");

  auto opensimFile = OpenSimParser::parseOsim(openSimFilePath);

  for (int i = 0; i < motFiles.size(); i++)
  {
    OpenSimMot mot = OpenSimParser::loadMot(opensimFile.skeleton, motFiles[i]);
    int framesPerSecond
        = (mot.timestamps.size() / mot.timestamps[mot.timestamps.size() - 1]);
    s_t dt = 1.0 / (s_t)framesPerSecond;
    timesteps.push_back(dt);

    std::vector<Eigen::MatrixXs> posePasses;
    std::vector<std::vector<ForcePlate>> grfPasses;
    std::vector<ForcePlate> grf
        = OpenSimParser::loadGRF(grfFiles[i], mot.timestamps);
    for (int pass = 0; pass < processingPasses.size(); pass++)
    {
      posePasses.push_back(mot.poses);
      grfPasses.push_back(grf);
    }
    poseTrialPasses.push_back(posePasses);
    forcePlateTrialPasses.push_back(grfPasses);
  }

  // This code trims all the timesteps down, if we asked for that
  if (limitTrialSizes > 0 || trialStartOffset > 0)
  {
    std::vector<std::vector<Eigen::MatrixXs>> trimmedPoseTrialPasses;
    std::vector<std::vector<std::vector<ForcePlate>>>
        trimmedForcePlateTrialPasses;

    for (int trial = 0; trial < poseTrialPasses.size(); trial++)
    {
      trimmedPoseTrialPasses.emplace_back();
      trimmedForcePlateTrialPasses.emplace_back();

      for (int pass = 0; pass < poseTrialPasses[trial].size(); pass++)
      {

        // TODO: handle edge cases

        trimmedPoseTrialPasses[trial].push_back(
            poseTrialPasses[trial][pass].block(
                0,
                trialStartOffset,
                poseTrialPasses[trial][pass].rows(),
                limitTrialSizes));

        std::vector<ForcePlate> trimmedPlates;
        for (int i = 0; i < forcePlateTrialPasses[trial][pass].size(); i++)
        {
          ForcePlate& toCopy = forcePlateTrialPasses[trial][pass][i];
          ForcePlate trimmedPlate;

          trimmedPlate.corners = toCopy.corners;
          trimmedPlate.worldOrigin = toCopy.worldOrigin;

          for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
               t++)
          {
            trimmedPlate.centersOfPressure.push_back(
                toCopy.centersOfPressure[t]);
            trimmedPlate.forces.push_back(toCopy.forces[t]);
            trimmedPlate.moments.push_back(toCopy.moments[t]);
          }

          trimmedPlates.push_back(trimmedPlate);
        }
        trimmedForcePlateTrialPasses[trimmedForcePlateTrialPasses.size() - 1]
            .push_back(trimmedPlates);
      }
    }

    poseTrialPasses = trimmedPoseTrialPasses;
    forcePlateTrialPasses = trimmedForcePlateTrialPasses;
  }

  std::vector<std::string> groundForceBodies;
  groundForceBodies.push_back("calcn_r");
  groundForceBodies.push_back("calcn_l");

  for (int trial = 0; trial < poseTrialPasses.size(); trial++)
  {
    velTrialPasses.emplace_back();
    accTrialPasses.emplace_back();
    tauTrialPasses.emplace_back();
    groundBodyWrenchTrialPasses.emplace_back();
    groundBodyCopTorqueForceTrialPasses.emplace_back();
    for (int pass = 0; pass < processingPasses.size(); pass++)
    {
      velTrialPasses[trial].push_back(Eigen::MatrixXs::Random(
          poseTrialPasses[trial][pass].rows(),
          poseTrialPasses[trial][pass].cols()));
      accTrialPasses[trial].push_back(Eigen::MatrixXs::Random(
          poseTrialPasses[trial][pass].rows(),
          poseTrialPasses[trial][pass].cols()));
      tauTrialPasses[trial].push_back(Eigen::MatrixXs::Random(
          poseTrialPasses[trial][pass].rows(),
          poseTrialPasses[trial][pass].cols()));
      groundBodyWrenchTrialPasses[trial].push_back(Eigen::MatrixXs::Random(
          6 * groundForceBodies.size(), poseTrialPasses[trial][pass].cols()));
      groundBodyCopTorqueForceTrialPasses[trial].push_back(
          Eigen::MatrixXs::Random(
              9 * groundForceBodies.size(),
              poseTrialPasses[trial][pass].cols()));
    }
    std::vector<Eigen::MatrixXs> trialCustomValues;
    trialCustomValues.push_back(Eigen::MatrixXs::Random(
        poseTrialPasses[trial][0].rows(), poseTrialPasses[trial][0].cols()));
    customValueTrials.push_back(trialCustomValues);

    std::vector<bool> missingGRF;
    std::vector<MissingGRFReason> grfReason;
    std::vector<s_t> residuals;
    for (int t = 0; t < poseTrialPasses[trial][0].cols(); t++)
    {
      missingGRF.push_back(t % 10 == 0);
      if (t % 10 == 0)
      {
        grfReason.push_back(
            MissingGRFReason::measuredGrfZeroWhenAccelerationNonZero);
      }
      else
      {
        grfReason.push_back(MissingGRFReason::notMissingGRF);
      }
      residuals.push_back(t);
    }
    missingGRFReason.push_back(grfReason);
    probablyMissingGRFData.push_back(missingGRF);
    residualNorms.push_back(residuals);

    for (int pass = 0; pass < processingPasses.size(); pass++)
    {
      std::vector<bool> positionObserved;
      std::vector<bool> velocityFiniteDifferenced;
      std::vector<bool> accelerationFiniteDifferenced;
      for (int i = 0; i < poseTrialPasses[trial].rows(); i++)
      {
        positionObserved.push_back(i % 2 == 0);
        velocityFiniteDifferenced.push_back(i % 3 == 0);
        accelerationFiniteDifferenced.push_back(i % 4 == 0);
      }
      passDofPositionObserved.push_back(positionObserved);
      passDofVelocityFiniteDifferenced.push_back(velocityFiniteDifferenced);
      passDofAccelerationFiniteDifferenced.push_back(
          accelerationFiniteDifferenced);
    }

    trialComPoses.push_back(
        Eigen::MatrixXs::Random(3, poseTrialPasses[trial].cols()));
    trialComVelocities.push_back(
        Eigen::MatrixXs::Random(3, poseTrialPasses[trial].cols()));
    trialComAccelerations.push_back(
        Eigen::MatrixXs::Random(3, poseTrialPasses[trial].cols()));
  }

  std::vector<std::string> trialNames;
  for (int i = 0; i < poseTrialPasses.size(); i++)
  {
    trialNames.push_back("trial_" + std::to_string(i));
  }
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
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservations;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      accObservations;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      gyroObservations;
  std::vector<std::vector<std::map<std::string, Eigen::VectorXs>>>
      emgObservations;

  std::vector<std::string> markerNames;
  for (int i = 0; i < 10; i++)
  {
    markerNames.push_back("marker_" + std::to_string(i));
  }
  for (int trial = 0; trial < poseTrialPasses.size(); trial++)
  {
    std::vector<std::map<std::string, Eigen::Vector3s>> markerTrial;
    std::vector<std::map<std::string, Eigen::Vector3s>> accTrial;
    std::vector<std::map<std::string, Eigen::Vector3s>> gyroTrial;
    std::vector<std::map<std::string, Eigen::VectorXs>> emgTrial;
    for (int t = 0; t < poseTrialPasses[trial].cols(); t++)
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

  SubjectOnDiskHeader header;
  header.setHeightM(heightM);
  header.setMassKg(massKg);
  header.setAgeYears(age);
  header.addProcessingPass()
      .setProcessingPassType(ProcessingPassType::kinematics)
      .setOpenSimFileText(openSimFilePath);
  SubjectOnDisk::writeB3D(outputFilePath, header);

  // SubjectOnDisk::writeSubject(
  //     outputFilePath,
  //     openSimFilePath,
  //     timesteps,
  //     poseTrials,
  //     velTrials,
  //     accTrials,
  //     probablyMissingGRFData,
  //     missingGRFReason,
  //     dofPositionObserved,
  //     dofVelocityFiniteDifferenced,
  //     dofAccelerationFiniteDifferenced,
  //     tauTrials,
  //     trialComPoses,
  //     trialComVelocities,
  //     trialComAccelerations,
  //     residualNorms,
  //     groundForceBodies,
  //     groundBodyWrenchTrials,
  //     groundBodyCopTorqueForceTrials,
  //     customValueNames,
  //     customValueTrials,
  //     markerObservations,
  //     accObservations,
  //     gyroObservations,
  //     emgObservations,
  //     forcePlateTrials,
  //     biologicalSex,
  //     heightM,
  //     massKg,
  //     age,
  //     trialNames,
  //     subjectTags,
  //     trialTags,
  //     originalHref,
  //     originalNotes);

  ////////////////////////////////////////
  // Test reading the subject back in
  ////////////////////////////////////////

  SubjectOnDisk subject(outputFilePath);

  std::shared_ptr<dynamics::Skeleton> skel = subject.readSkel(
      0, "dart://sample/osim/OpenCapTest/Subject4/Models/Geometry/");
  if (skel == nullptr)
  {
    std::cout << "Failed to read skeleton back in!" << std::endl;
    return false;
  }

  for (int i = 0; i < trialNames.size(); i++)
  {
    if (subject.getTrialName(i) != trialNames[i])
    {
      std::cout << "Failed to recover trial name!" << std::endl;
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

  for (int trial = 0; trial < forcePlateTrials.size(); trial++)
  {
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
    for (int pass = 0; pass < processingPasses.size(); pass++)
    {
      std::vector<s_t> maxVels = subject.getTrialMaxJointVelocity(trial);
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

    std::vector<std::shared_ptr<biomechanics::Frame>> readResult
        = subject.readFrames(trial, frame, 5);

    for (int j = 0; j < readResult.size(); j++)
    {
      int timestep = frame + j;
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

      if (abs(subject.getTrialTimestep(trial) - timesteps[frame->trial]) > 1e-8)
      {
        std::cout << "dt not recovered" << std::endl;
        return false;
      }
      if (frame->probablyMissingGRF != probablyMissingGRFData[trial][frame->t])
      {
        std::cout << "missing GRF not recovered" << std::endl;
        return false;
      }

      if (frame->missingGRFReason != missingGRFReason[trial][frame->t])
      {
        std::cout << "missing GRF reason not recovered" << std::endl;
        return false;
      }

      Eigen::VectorXs originalPos = poseTrialPasses[frame->trial].col(frame->t);
      if (!equals(originalPos, frame->processingPasses[0].pos, 1e-8))
      {
        std::cout << "Pos not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalVel = velTrials[frame->trial].col(frame->t);
      if (!equals(originalVel, frame->processingPasses[0].vel, 1e-8))
      {
        std::cout << "Vel not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalAcc = accTrials[frame->trial].col(frame->t);
      if (!equals(originalAcc, frame->processingPasses[0].acc, 1e-8))
      {
        std::cout << "Acc not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalTau = tauTrials[frame->trial].col(frame->t);
      if (!equals(originalTau, frame->processingPasses[0].tau, 1e-8))
      {
        std::cout << "Tau not recovered" << std::endl;
        return false;
      }
      for (int b = 0; b < subject.getGroundContactBodies().size(); b++)
      {
        Eigen::Vector6s originalWrench
            = groundBodyWrenchTrials[frame->trial].col(frame->t).segment<6>(
                b * 6);
        Eigen::Vector6s recoveredWrench
            = frame->processingPasses[0].groundContactWrenches.segment<6>(
                b * 6);
        if (!equals(originalWrench, recoveredWrench, 1e-8))
        {
          std::cout << "Body wrench not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalCoP
            = groundBodyCopTorqueForceTrials[frame->trial]
                  .col(frame->t)
                  .segment<3>(b * 9);
        Eigen::Vector3s recoveredCoP
            = frame->processingPasses[0]
                  .groundContactCenterOfPressure.segment<3>(b * 3);
        if (!equals(originalCoP, recoveredCoP, 1e-8))
        {
          std::cout << "GRF CoP not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalTau
            = groundBodyCopTorqueForceTrials[frame->trial]
                  .col(frame->t)
                  .segment<3>((b * 9) + 3);
        Eigen::Vector3s recoveredTau
            = frame->processingPasses[0].groundContactTorque.segment<3>(b * 3);
        if (!equals(originalTau, recoveredTau, 1e-8))
        {
          std::cout << "GRF Tau not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalF = groundBodyCopTorqueForceTrials[frame->trial]
                                        .col(frame->t)
                                        .segment<3>((b * 9) + 6);
        Eigen::Vector3s recoveredF
            = frame->processingPasses[0].groundContactForce.segment<3>(b * 3);
        if (!equals(originalF, recoveredF, 1e-8))
        {
          std::cout << "GRF Force not recovered" << std::endl;
          return false;
        }
      }
      for (int b = 0; b < frame->customValues.size(); b++)
      {
        Eigen::VectorXs originalValue
            = customValueTrials[frame->trial][b].col(frame->t);
        if (!equals(originalValue, frame->customValues[b].second, 1e-8))
        {
          std::cout << "Custom value not recovered" << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

#ifdef ALL_TESTS
TEST(SubjectOnDisk, WRITE_THEN_READ)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("DJ5");
  trialNames.push_back("walking2");
  trialNames.push_back("DJ1");
  trialNames.push_back("walking3");
  trialNames.push_back("walking4");
  trialNames.push_back("DJ4");

  std::vector<std::string> motFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/grf/OpenCapUnfiltered/IK/" + name + "_ik.mot");
    grfFiles.push_back(
        "dart://sample/grf/OpenCapUnfiltered/ID/" + name + "_grf.mot");
  }

  std::string path = "./testSubject.bin";

  for (int i = 0; i < 10; i++)
  {
    EXPECT_TRUE(testWriteSubjectToDisk(
        path,
        "dart://sample/osim/OpenCapTest/Subject4/Models/"
        "unscaled_generic.osim",
        motFiles,
        grfFiles));
  }
}
#endif