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

// #define ALL_TESTS

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

  std::vector<Eigen::MatrixXs> poseTrials;
  std::vector<std::vector<ForcePlate>> forcePlateTrials;
  std::vector<s_t> timesteps;
  std::vector<Eigen::MatrixXs> velTrials;
  std::vector<Eigen::MatrixXs> accTrials;
  std::vector<Eigen::MatrixXs> tauTrials;
  std::vector<Eigen::MatrixXs> groundBodyWrenchTrials;
  std::vector<Eigen::MatrixXs> groundBodyCopTorqueForceTrials;
  std::vector<std::vector<bool>> probablyMissingGRFData;
  std::vector<std::string> customValueNames;
  std::vector<std::vector<Eigen::MatrixXs>> customValueTrials;

  customValueNames.push_back("exo_tau");

  auto opensimFile = OpenSimParser::parseOsim(openSimFilePath);

  for (int i = 0; i < motFiles.size(); i++)
  {
    OpenSimMot mot = OpenSimParser::loadMot(opensimFile.skeleton, motFiles[i]);
    poseTrials.push_back(mot.poses);
    int framesPerSecond
        = (mot.timestamps.size() / mot.timestamps[mot.timestamps.size() - 1]);
    s_t dt = 1.0 / (s_t)framesPerSecond;
    timesteps.push_back(dt);

    for (int i = 0; i < grfFiles.size(); i++)
    {
      std::vector<ForcePlate> grf
          = OpenSimParser::loadGRF(grfFiles[i], framesPerSecond);
      forcePlateTrials.push_back(grf);
    }
  }

  // This code trims all the timesteps down, if we asked for that
  if (limitTrialSizes > 0 || trialStartOffset > 0)
  {
    std::vector<Eigen::MatrixXs> trimmedPoseTrials;
    std::vector<std::vector<ForcePlate>> trimmedForcePlateTrials;

    for (int trial = 0; trial < poseTrials.size(); trial++)
    {
      // TODO: handle edge cases

      trimmedPoseTrials.push_back(poseTrials[trial].block(
          0, trialStartOffset, poseTrials[trial].rows(), limitTrialSizes));

      std::vector<ForcePlate> trimmedPlates;
      for (int i = 0; i < forcePlateTrials[trial].size(); i++)
      {
        ForcePlate& toCopy = forcePlateTrials[trial][i];
        ForcePlate trimmedPlate;

        trimmedPlate.corners = toCopy.corners;
        trimmedPlate.worldOrigin = toCopy.worldOrigin;

        for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
             t++)
        {
          trimmedPlate.centersOfPressure.push_back(toCopy.centersOfPressure[t]);
          trimmedPlate.forces.push_back(toCopy.forces[t]);
          trimmedPlate.moments.push_back(toCopy.moments[t]);
        }

        trimmedPlates.push_back(trimmedPlate);
      }
      trimmedForcePlateTrials.push_back(trimmedPlates);
    }

    poseTrials = trimmedPoseTrials;
    forcePlateTrials = trimmedForcePlateTrials;
  }

  std::vector<std::string> groundForceBodies;
  groundForceBodies.push_back("calcn_r");
  groundForceBodies.push_back("calcn_l");

  for (int trial = 0; trial < poseTrials.size(); trial++)
  {
    velTrials.push_back(Eigen::MatrixXs::Random(
        poseTrials[trial].rows(), poseTrials[trial].cols()));
    accTrials.push_back(Eigen::MatrixXs::Random(
        poseTrials[trial].rows(), poseTrials[trial].cols()));
    tauTrials.push_back(Eigen::MatrixXs::Random(
        poseTrials[trial].rows(), poseTrials[trial].cols()));
    groundBodyWrenchTrials.push_back(Eigen::MatrixXs::Random(
        6 * groundForceBodies.size(), poseTrials[trial].cols()));
    groundBodyCopTorqueForceTrials.push_back(Eigen::MatrixXs::Random(
        9 * groundForceBodies.size(), poseTrials[trial].cols()));
    std::vector<Eigen::MatrixXs> trialCustomValues;
    trialCustomValues.push_back(Eigen::MatrixXs::Random(
        poseTrials[trial].rows(), poseTrials[trial].cols()));
    customValueTrials.push_back(trialCustomValues);

    std::vector<bool> missingGRF;
    for (int t = 0; t < poseTrials[trial].cols(); t++)
    {
      missingGRF.push_back(t % 10 == 0);
    }
    probablyMissingGRFData.push_back(missingGRF);
  }

  std::vector<std::string> trialNames;
  for (int i = 0; i < poseTrials.size(); i++)
  {
    trialNames.push_back("trial_" + std::to_string(i));
  }
  std::string originalHref
      = "https://dev.addbiomechanics.org/data/"
        "35e1c7ca-cc58-457e-bfc5-f6161cc7278b/SprinterTest";
  std::string originalNotes
      = "A sprinter originally recorded by blah blah blah. Please cite bibtex "
        "{} blah blah blah if you use this data specifically.";

  SubjectOnDisk::writeSubject(
      outputFilePath,
      openSimFilePath,
      timesteps,
      poseTrials,
      velTrials,
      accTrials,
      probablyMissingGRFData,
      tauTrials,
      groundForceBodies,
      groundBodyWrenchTrials,
      groundBodyCopTorqueForceTrials,
      customValueNames,
      customValueTrials,
      trialNames,
      originalHref,
      originalNotes);

  ////////////////////////////////////////
  // Test reading the subject back in
  ////////////////////////////////////////

  SubjectOnDisk subject(outputFilePath);

  std::shared_ptr<dynamics::Skeleton> skel = subject.readSkel(
      "dart://sample/osim/OpenCapTest/Subject4/Models/Geometry/");
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

  for (int i = 0; i < 500; i++)
  {
    int trial = rand() % subject.getNumTrials();
    int frame = rand() % subject.getTrialLength(trial);

    std::vector<std::shared_ptr<biomechanics::Frame>> readResult
        = subject.readFrames(trial, frame, 5);

    for (auto& frame : readResult)
    {
      std::cout << "Checking frame " << frame->trial << ":" << frame->t
                << std::endl;

      if (abs(frame->dt - timesteps[frame->trial]) > 1e-8)
      {
        std::cout << "dt not recovered" << std::endl;
        return false;
      }
      if (frame->probablyMissingGRF != probablyMissingGRFData[trial][frame->t])
      {
        std::cout << "missing GRF not recovered" << std::endl;
        return false;
      }

      Eigen::VectorXs originalPos = poseTrials[frame->trial].col(frame->t);
      if (!equals(originalPos, frame->pos, 1e-8))
      {
        std::cout << "Pos not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalVel = velTrials[frame->trial].col(frame->t);
      if (!equals(originalVel, frame->vel, 1e-8))
      {
        std::cout << "Vel not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalAcc = accTrials[frame->trial].col(frame->t);
      if (!equals(originalAcc, frame->acc, 1e-8))
      {
        std::cout << "Acc not recovered" << std::endl;
        return false;
      }
      Eigen::VectorXs originalTau = tauTrials[frame->trial].col(frame->t);
      if (!equals(originalTau, frame->tau, 1e-8))
      {
        std::cout << "Tau not recovered" << std::endl;
        return false;
      }
      for (int b = 0; b < frame->groundContactWrenches.size(); b++)
      {
        Eigen::Vector6s originalWrench
            = groundBodyWrenchTrials[frame->trial].col(frame->t).segment<6>(
                b * 6);
        if (!equals(
                originalWrench, frame->groundContactWrenches[b].second, 1e-8))
        {
          std::cout << "Body wrench not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalCoP
            = groundBodyCopTorqueForceTrials[frame->trial]
                  .col(frame->t)
                  .segment<3>(b * 9);
        if (!equals(
                originalCoP,
                frame->groundContactCenterOfPressure[b].second,
                1e-8))
        {
          std::cout << "GRF CoP not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalTau
            = groundBodyCopTorqueForceTrials[frame->trial]
                  .col(frame->t)
                  .segment<3>((b * 9) + 3);
        if (!equals(originalTau, frame->groundContactTorque[b].second, 1e-8))
        {
          std::cout << "GRF Tau not recovered" << std::endl;
          return false;
        }
        Eigen::Vector3s originalF = groundBodyCopTorqueForceTrials[frame->trial]
                                        .col(frame->t)
                                        .segment<3>((b * 9) + 6);
        if (!equals(originalF, frame->groundContactForce[b].second, 1e-8))
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

  EXPECT_TRUE(testWriteSubjectToDisk(
      path,
      "dart://sample/osim/OpenCapTest/Subject4/Models/"
      "unscaled_generic.osim",
      motFiles,
      grfFiles));
}