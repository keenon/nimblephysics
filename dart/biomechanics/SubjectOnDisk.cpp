#include "dart/biomechanics/SubjectOnDisk.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <stdio.h>
#include <tinyxml2.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/proto/SubjectOnDisk.pb.h"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

using namespace std;

#define float64_t double

namespace dart {
namespace biomechanics {

extern "C" {
struct FileHeader
{
  int32_t magic;
  int32_t version;
  int32_t numDofs;
  int32_t numTrials;
  int32_t numGroundContactBodies;
  int32_t numCustomValues;
};
}

proto::ProcessingPassType passTypeToProto(ProcessingPassType type)
{
  switch (type)
  {
    case kinematics:
      return proto::ProcessingPassType::kinematics;
    case dynamics:
      return proto::ProcessingPassType::dynamics;
    case lowPassFilter:
      return proto::ProcessingPassType::lowPassFilter;
  }
  return proto::ProcessingPassType::kinematics;
}

ProcessingPassType passTypeFromProto(proto::ProcessingPassType type)
{
  switch (type)
  {
    case proto::ProcessingPassType::kinematics:
      return kinematics;
    case proto::ProcessingPassType::dynamics:
      return dynamics;
    case proto::ProcessingPassType::lowPassFilter:
      return lowPassFilter;
      // These are just here to keep Clang from complaining
    case proto::ProcessingPassType_INT_MIN_SENTINEL_DO_NOT_USE_:
      return kinematics;
      break;
    case proto::ProcessingPassType_INT_MAX_SENTINEL_DO_NOT_USE_:
      return kinematics;
      break;
  }
  return kinematics;
}

proto::MissingGRFReason missingGRFReasonToProto(MissingGRFReason reason)
{
  switch (reason)
  {
    case notMissingGRF:
      return proto::MissingGRFReason::notMissingGRF;
    case measuredGrfZeroWhenAccelerationNonZero:
      return proto::MissingGRFReason::measuredGrfZeroWhenAccelerationNonZero;
    case unmeasuredExternalForceDetected:
      return proto::MissingGRFReason::unmeasuredExternalForceDetected;
    case torqueDiscrepancy:
      return proto::MissingGRFReason::torqueDiscrepancy;
    case forceDiscrepancy:
      return proto::MissingGRFReason::forceDiscrepancy;
    case notOverForcePlate:
      return proto::MissingGRFReason::notOverForcePlate;
    case missingImpact:
      return proto::MissingGRFReason::missingImpact;
    case missingBlip:
      return proto::MissingGRFReason::missingBlip;
    case shiftGRF:
      return proto::MissingGRFReason::shiftGRF;
    case interpolatedClippedGRF:
      return proto::MissingGRFReason::interpolatedClippedGRF;
  }
  return proto::MissingGRFReason::notMissingGRF;
}

MissingGRFReason missingGRFReasonFromProto(proto::MissingGRFReason reason)
{
  switch (reason)
  {
    case proto::MissingGRFReason::notMissingGRF:
      return notMissingGRF;
    case proto::MissingGRFReason::measuredGrfZeroWhenAccelerationNonZero:
      return measuredGrfZeroWhenAccelerationNonZero;
    case proto::MissingGRFReason::unmeasuredExternalForceDetected:
      return unmeasuredExternalForceDetected;
    case proto::MissingGRFReason::torqueDiscrepancy:
      return torqueDiscrepancy;
    case proto::MissingGRFReason::forceDiscrepancy:
      return forceDiscrepancy;
    case proto::MissingGRFReason::notOverForcePlate:
      return notOverForcePlate;
    case proto::MissingGRFReason::missingImpact:
      return missingImpact;
    case proto::MissingGRFReason::missingBlip:
      return missingBlip;
    case proto::MissingGRFReason::shiftGRF:
      return shiftGRF;
    case proto::MissingGRFReason::interpolatedClippedGRF:
      return interpolatedClippedGRF;
      // These are just here to keep Clang from complaining
    case proto::MissingGRFReason_INT_MIN_SENTINEL_DO_NOT_USE_:
      return notMissingGRF;
      break;
    case proto::MissingGRFReason_INT_MAX_SENTINEL_DO_NOT_USE_:
      return notMissingGRF;
      break;
  }
  return notMissingGRF;
}

SubjectOnDisk::SubjectOnDisk(const std::string& path)
  : mPath(path), mLoadedAllFrames(false)
{
  // 1. Open the file
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk attempting to open file that deos not exist: "
              << path << std::endl;
    throw new std::exception();
  }
  // 2. Read the length of the message from the integer header
  int64_t headerSize = -1;
  int64_t elementsRead = fread(&headerSize, sizeof(int64_t), 1, file);
  if (elementsRead != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path
              << ": was unable to read header size, probably because the file "
                 "is length 0?"
              << std::endl;
    throw new std::exception();
  }
  // 3. Allocate a buffer to hold the serialized data
  std::vector<char> serializedHeader(headerSize);

  // 4. Read the serialized data from the file
  int64_t bytesRead
      = fread(serializedHeader.data(), sizeof(char), headerSize, file);

  if (bytesRead != headerSize)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path << ": was unable to read full requested header size "
              << headerSize << ", instead only got " << bytesRead << " bytes."
              << std::endl;
    throw new std::exception();
  }

  // 5. Deserialize the data into a Protobuf object
  proto::SubjectOnDiskHeader header;
  bool parseSuccess
      = header.ParseFromArray(serializedHeader.data(), serializedHeader.size());
  if (!parseSuccess)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path
              << ": got an error parsing the protobuf file header. Size = "
              << serializedHeader.size() << "\nParsed Partial Message: "
              << header.DebugString() // Print the partially parsed message
              << std::endl;
    throw std::exception();
  }

  // Check if all required fields in the message are set
  if (!header.IsInitialized())
  {
    std::cout << "SubjectOnDisk protobuf message is missing required fields at "
              << path << ": "
              << header.InitializationErrorString() // Indicate which required
                                                    // fields are missing
              << std::endl;
    throw std::exception();
  }
  // 6. Get the data out of the protobuf object
  mHeader = std::make_shared<SubjectOnDiskHeader>();
  mHeader->read(header);
  mSensorFrameSize = header.raw_sensor_frame_size();
  mProcessingPassFrameSize = header.processing_pass_frame_size();
  mDataSectionStart = sizeof(int64_t) + headerSize;

  fclose(file);
}

/// This will write a B3D file to disk
void SubjectOnDisk::writeB3D(
    const std::string& outputPath, std::shared_ptr<SubjectOnDiskHeader> header)
{
  // 0. Open the file
  FILE* file = fopen(outputPath.c_str(), "wb");
  if (file == nullptr)
  {
    std::cout
        << "SubjectOnDiskBuilder::writeB3D() failed to open output file at "
        << outputPath << ". Do you have permissions to write that file?"
        << std::endl;
    return;
  }

  // Create the header proto

  proto::SubjectOnDiskHeader headerProto;
  header->write(&headerProto);

  // 1.3. Continues in next section, after we know the size of the first
  // serialized frame...

  /////////////////////////////////////////////////////////////////////////////
  // 2. Serialize and write the frames to the file
  /////////////////////////////////////////////////////////////////////////////

  int maxNumForcePlates = 0;
  for (int trial = 0; trial < header->mTrials.size(); trial++)
  {
    if (header->mTrials[trial]->mForcePlates.size() > maxNumForcePlates)
    {
      maxNumForcePlates = header->mTrials[trial]->mForcePlates.size();
    }
  }

  bool firstTrial = true;
  int64_t sensorFrameSize = 0;
  int64_t passFrameSize = 0;

  for (int trial = 0; trial < header->mTrials.size(); trial++)
  {
    for (int t = 0; t < header->mTrials[trial]->mMarkerObservations.size(); t++)
    {
      // 2.1. Populate the protobuf frame object in memory
      proto::SubjectOnDiskSensorFrame sensorsFrameProto;
      header->writeSensorsFrame(
          &sensorsFrameProto, trial, t, maxNumForcePlates);
      // 2.2. Serialize the protobuf header object
      std::string sensorFrameSerialized = "";
      sensorsFrameProto.SerializeToString(&sensorFrameSerialized);

      // 2.3. Get the length of the serialized message
      if (sensorFrameSize != 0)
      {
        assert(sensorFrameSize == sensorFrameSerialized.size());
      }
      sensorFrameSize = sensorFrameSerialized.size();

      // 2.4. Serialize the processing passes
      std::vector<std::string> passFramesSerialized;

      for (int pass = 0; pass < header->mTrials[trial]->mTrialPasses.size();
           pass++)
      {
        proto::SubjectOnDiskProcessingPassFrame passFrameProto;
        header->writeProcessingPassFrame(&passFrameProto, trial, t, pass);
        std::string passFrameSerialized = "";
        passFrameProto.SerializeToString(&passFrameSerialized);
        if (passFrameSize != 0)
        {
          assert(passFrameSize == passFrameSerialized.size());
        }
        passFrameSize = passFrameSerialized.size();
        passFramesSerialized.push_back(passFrameSerialized);
      }

      // If this is the first trial, we need to finish the header with
      // information about the size of serialized frame objects and write the
      // header first
      if (firstTrial)
      {
        // 1.3. Continued from previous section... Write the size of the frame
        // binaries
        headerProto.set_raw_sensor_frame_size(sensorFrameSize);
        headerProto.set_processing_pass_frame_size(passFrameSize);
        if (!headerProto.IsInitialized())
        {
          std::cerr << "All required fields are not set:\n"
                    << headerProto.InitializationErrorString() << std::endl;
          fclose(file);
          return;
        }

        // 1.4. Serialize the protobuf header object
        std::string headerSerialized = "";
        bool success = headerProto.SerializeToString(&headerSerialized);
        if (!success)
        {
          std::cerr << "Failed to serialize the protobuf message." << std::endl;
          fclose(file);
          return;
        }

        // 1.5. Write the length of the message as an integer header
        int64_t headerSize = headerSerialized.size();
        fwrite(&headerSize, sizeof(int64_t), 1, file);

        // 1.6. Write the serialized data to the file
        fwrite(headerSerialized.c_str(), sizeof(char), headerSize, file);

        firstTrial = false;
      }

      // 2.4. Write the serialized data to the file
      fwrite(
          sensorFrameSerialized.c_str(), sizeof(char), sensorFrameSize, file);
      for (int pass = 0; pass < header->mTrials[trial]->mTrialPasses.size();
           pass++)
      {
        fwrite(
            passFramesSerialized[pass].c_str(),
            sizeof(char),
            passFrameSize,
            file);
      }
    }
  }

  if (firstTrial == true)
  {
    std::cout << "SubjectOnDiskBuilder::writeB3D() failed to write any frames "
                 "of data."
              << std::endl;
    std::cout << "Debug info: " << std::endl;
    std::cout << "header.mTrials.size(): " << header->mTrials.size()
              << std::endl;
    for (int i = 0; i < header->mTrials.size(); i++)
    {
      std::cout << "header.mTrials[" << i << "].mMarkerObservations.size(): "
                << header->mTrials[i]->mMarkerObservations.size() << std::endl;
    }
    return;
  }

  fclose(file);
}

/// This loads all the frames of data, and fills in the processing pass data
/// matrices in the proto header classes.
void SubjectOnDisk::loadAllFrames(bool doNotStandardizeForcePlateData)
{
  if (mLoadedAllFrames)
  {
    return;
  }
  mLoadedAllFrames = true;

  const int dofs = getNumDofs();
  const int numContactBodies = getGroundForceBodies().size();

  for (int trial = 0; trial < getNumTrials(); trial++)
  {
    const int len = getTrialLength(trial);
    std::vector<std::shared_ptr<Frame>> frames
        = readFrames(trial, 0, len, true, true, 1, 0.0);

    for (int t = 0; t < frames.size(); t++)
    {
      std::map<std::string, Eigen::Vector3s> markerMap;
      for (auto& pair : frames[t]->markerObservations)
      {
        markerMap[pair.first] = pair.second;
      }
      mHeader->mTrials[trial]->mMarkerObservations.push_back(markerMap);
    }

    const int numPlates = getNumForcePlates(trial);
    for (int plate = 0; plate < numPlates; plate++)
    {
      mHeader->mTrials[trial]->mForcePlates.push_back(ForcePlate());
      mHeader->mTrials[trial]->mForcePlates[plate].corners
          = getForcePlateCorners(trial, plate);
    }

    for (int t = 0; t < frames.size(); t++)
    {
      for (int plate = 0; plate < numPlates; plate++)
      {
        mHeader->mTrials[trial]
            ->mForcePlates[plate]
            .centersOfPressure.push_back(
                frames[t]->rawForcePlateCenterOfPressures[plate]);
        mHeader->mTrials[trial]->mForcePlates[plate].forces.push_back(
            frames[t]->rawForcePlateForces[plate]);
        mHeader->mTrials[trial]->mForcePlates[plate].moments.push_back(
            frames[t]->rawForcePlateTorques[plate]);
        mHeader->mTrials[trial]->mForcePlates[plate].timestamps.push_back(
            frames[t]->t * getTrialTimestep(trial));
      }
    }
    if (!doNotStandardizeForcePlateData)
    {
      for (int plate = 0; plate < numPlates; plate++)
      {
        mHeader->mTrials[trial]
            ->mForcePlates[plate]
            .autodetectNoiseThresholdAndClip();
        mHeader->mTrials[trial]
            ->mForcePlates[plate]
            .detectAndFixCopMomentConvention(trial, plate);
      }
    }

    for (int pass = 0; pass < getTrialNumProcessingPasses(trial); pass++)
    {
      std::shared_ptr<SubjectOnDiskTrialPass> passProto
          = mHeader->mTrials[trial]->mTrialPasses[pass];

      passProto->mMarkerRMS.resize(len);
      passProto->mMarkerMax.resize(len);
      passProto->mLinearResidual.resize(len);
      passProto->mAngularResidual.resize(len);
      passProto->mPos = Eigen::MatrixXs::Zero(dofs, len);
      passProto->mVel = Eigen::MatrixXs::Zero(dofs, len);
      passProto->mAcc = Eigen::MatrixXs::Zero(dofs, len);
      passProto->mTaus = Eigen::MatrixXs::Zero(dofs, len);
      passProto->mGroundBodyWrenches
          = Eigen::MatrixXs::Zero(numContactBodies * 6, len);
      passProto->mGroundBodyCopTorqueForce
          = Eigen::MatrixXs::Zero(numContactBodies * 9, len);
      passProto->mComPoses = Eigen::MatrixXs::Zero(3, len);
      passProto->mComVels = Eigen::MatrixXs::Zero(3, len);
      passProto->mComAccs = Eigen::MatrixXs::Zero(3, len);
      passProto->mComAccsInRootFrame = Eigen::MatrixXs::Zero(3, len);
      passProto->mResidualWrenchInRootFrame = Eigen::MatrixXs::Zero(6, len);
      passProto->mGroundBodyWrenchesInRootFrame
          = Eigen::MatrixXs::Zero(numContactBodies * 6, len);
      passProto->mGroundBodyCopTorqueForceInRootFrame
          = Eigen::MatrixXs::Zero(numContactBodies * 9, len);
      const int jointCenterDim
          = frames.size() > 0
                ? frames[0]->processingPasses.size() > 0
                      ? frames[0]->processingPasses[0].jointCenters.size()
                      : 0
                : 0;
      passProto->mJointCenters = Eigen::MatrixXs::Zero(jointCenterDim, len);
      passProto->mJointCentersInRootFrame
          = Eigen::MatrixXs::Zero(jointCenterDim, len);
      passProto->mRootSpatialVelInRootFrame = Eigen::MatrixXs::Zero(6, len);
      passProto->mRootSpatialAccInRootFrame = Eigen::MatrixXs::Zero(6, len);
      int historyDim = frames.size() > 0
                           ? frames[0]->processingPasses.size() > 0
                                 ? frames[0]
                                       ->processingPasses[0]
                                       .rootEulerHistoryInRootFrame.size()
                                 : 0
                           : 0;
      passProto->mRootPosHistoryInRootFrame
          = Eigen::MatrixXs::Zero(historyDim, len);
      passProto->mRootEulerHistoryInRootFrame
          = Eigen::MatrixXs::Zero(historyDim, len);

      for (int t = 0; t < frames.size(); t++)
      {
        passProto->mPos.col(t) = frames[t]->processingPasses[pass].pos;
        passProto->mVel.col(t) = frames[t]->processingPasses[pass].vel;
        passProto->mAcc.col(t) = frames[t]->processingPasses[pass].acc;
        passProto->mTaus.col(t) = frames[t]->processingPasses[pass].tau;
        passProto->mGroundBodyWrenches.col(t)
            = frames[t]->processingPasses[pass].groundContactWrenches;
        passProto->mComPoses.col(t) = frames[t]->processingPasses[pass].comPos;
        passProto->mComVels.col(t) = frames[t]->processingPasses[pass].comVel;
        passProto->mComAccs.col(t) = frames[t]->processingPasses[pass].comAcc;
        passProto->mComAccsInRootFrame.col(t)
            = frames[t]->processingPasses[pass].comAccInRootFrame;
        passProto->mResidualWrenchInRootFrame.col(t)
            = frames[t]->processingPasses[pass].residualWrenchInRootFrame;
        passProto->mGroundBodyWrenchesInRootFrame.col(t)
            = frames[t]
                  ->processingPasses[pass]
                  .groundContactWrenchesInRootFrame;

        for (int body = 0; body < numContactBodies; body++)
        {
          passProto->mGroundBodyCopTorqueForce.block<3, 1>(body * 9, t)
              = frames[t]
                    ->processingPasses[pass]
                    .groundContactCenterOfPressure.segment<3>(body * 3);
          passProto->mGroundBodyCopTorqueForce.block<3, 1>(body * 9 + 3, t)
              = frames[t]
                    ->processingPasses[pass]
                    .groundContactTorque.segment<3>(body * 3);
          passProto->mGroundBodyCopTorqueForce.block<3, 1>(body * 9 + 6, t)
              = frames[t]->processingPasses[pass].groundContactForce.segment<3>(
                  body * 3);

          passProto->mGroundBodyCopTorqueForceInRootFrame.block<3, 1>(
              body * 9, t)
              = frames[t]
                    ->processingPasses[pass]
                    .groundContactCenterOfPressureInRootFrame.segment<3>(
                        body * 3);
          passProto->mGroundBodyCopTorqueForceInRootFrame.block<3, 1>(
              body * 9 + 3, t)
              = frames[t]
                    ->processingPasses[pass]
                    .groundContactTorqueInRootFrame.segment<3>(body * 3);
          passProto->mGroundBodyCopTorqueForceInRootFrame.block<3, 1>(
              body * 9 + 6, t)
              = frames[t]
                    ->processingPasses[pass]
                    .groundContactForceInRootFrame.segment<3>(body * 3);
        }

        passProto->mJointCenters.col(t)
            = frames[t]->processingPasses[pass].jointCenters;
        passProto->mJointCentersInRootFrame.col(t)
            = frames[t]->processingPasses[pass].jointCentersInRootFrame;
        passProto->mRootSpatialVelInRootFrame.col(t).head<3>()
            = frames[t]->processingPasses[pass].rootAngularVelInRootFrame;
        passProto->mRootSpatialVelInRootFrame.col(t).tail<3>()
            = frames[t]->processingPasses[pass].rootLinearVelInRootFrame;
        passProto->mRootSpatialAccInRootFrame.col(t).head<3>()
            = frames[t]->processingPasses[pass].rootAngularAccInRootFrame;
        passProto->mRootSpatialAccInRootFrame.col(t).tail<3>()
            = frames[t]->processingPasses[pass].rootLinearAccInRootFrame;
        passProto->mRootPosHistoryInRootFrame.col(t)
            = frames[t]->processingPasses[pass].rootPosHistoryInRootFrame;
        passProto->mRootEulerHistoryInRootFrame.col(t)
            = frames[t]->processingPasses[pass].rootEulerHistoryInRootFrame;
      }
    }
  }
}

/// This returns the raw proto header for this subject, which can be used to
/// write out a new B3D file
std::shared_ptr<SubjectOnDiskHeader> SubjectOnDisk::getHeaderProto()
{
  return mHeader;
}

/// This reads all the raw sensor data for this trial, and constructs
/// force plates.
std::vector<ForcePlate> SubjectOnDisk::readForcePlates(int trial)
{
  std::vector<ForcePlate> plates;

  const int len = getTrialLength(trial);
  const int numPlates = getNumForcePlates(trial);
  for (int plate = 0; plate < numPlates; plate++)
  {
    plates.push_back(ForcePlate());
    plates[plate].corners = getForcePlateCorners(trial, plate);
  }

  std::vector<std::shared_ptr<Frame>> frames
      = readFrames(trial, 0, len, true, false, 1, 0.0);
  for (int t = 0; t < frames.size(); t++)
  {
    for (int plate = 0; plate < numPlates; plate++)
    {
      plates[plate].centersOfPressure.push_back(
          frames[t]->rawForcePlateCenterOfPressures[plate]);
      plates[plate].forces.push_back(frames[t]->rawForcePlateForces[plate]);
      plates[plate].moments.push_back(frames[t]->rawForcePlateTorques[plate]);
      plates[plate].timestamps.push_back(
          frames[t]->t * getTrialTimestep(trial));
    }
  }
  for (int plate = 0; plate < numPlates; plate++)
  {
    plates[plate].autodetectNoiseThresholdAndClip();
    plates[plate].detectAndFixCopMomentConvention(trial, plate);
  }

  return plates;
}

/// This will read the skeleton from the binary, and optionally use the passed
/// in Geometry folder.
std::shared_ptr<dynamics::Skeleton> SubjectOnDisk::readSkel(
    int passNumberToLoad, std::string geometryFolder)
{
  if (geometryFolder == "")
  {
    // Guess that the Geometry folder is relative to the binary, if none is
    // provided
    geometryFolder = common::Uri::createFromRelativeUri(mPath, "./Geometry/")
                         .getFilesystemPath();
  }

  tinyxml2::XMLDocument osimFile;
  osimFile.Parse(mHeader->mPasses[passNumberToLoad]->mOpenSimFileText.c_str());
  OpenSimFile osimParsed
      = OpenSimParser::parseOsim(osimFile, mPath, geometryFolder);
  if (!(osimParsed.skeleton))
  {
    std::cout << "Failed to parse Osim XML: \""
              << mHeader->mPasses[passNumberToLoad]->mOpenSimFileText << "\""
              << std::endl;
    return nullptr;
  }
  osimParsed.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  return osimParsed.skeleton;
}

/// This will read the raw OpenSim XML file text out of the binary, and return
/// it as a string
std::string SubjectOnDisk::getOpensimFileText(int passNumberToLoad)
{
  return mHeader->mPasses[passNumberToLoad]->mOpenSimFileText;
}

// If we're doing a lowpass filter on this pass, then what was the cutoff
// frequency of that filter?
s_t SubjectOnDisk::getLowpassCutoffFrequency(int trial, int processingPass)
{
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mLowpassCutoffFrequency;
}

// If we're doing a lowpass filter on this pass, then what was the order of
// that (Butterworth) filter?
int SubjectOnDisk::getLowpassFilterOrder(int trial, int processingPass)
{
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mLowpassFilterOrder;
}

// If we reprocessed the force plates with a cutoff, then these are the cutoff
// values we used.
std::vector<s_t> SubjectOnDisk::getForceplateCutoffs(
    int trial, int processingPass)
{
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mForcePlateCutoffs;
}

/// This will read from disk and allocate a number of Frame objects,
/// optionally sharing the same Skeleton pointer for efficiency if
/// `shareSkeletonPtr` is true, (though that means it won't be threadsafe to
/// use the Frame objects in parallel). These Frame objects are assumed to be
/// short-lived, to save working memory.
///
/// On OOB access, prints an error and returns an empty vector.
std::vector<std::shared_ptr<Frame>> SubjectOnDisk::readFrames(
    int trial,
    int startFrame,
    int numFramesToRead,
    bool includeSensorData,
    bool includeProcessingPasses,
    int stride,
    s_t contactThreshold)
{
  (void)trial;
  (void)startFrame;
  (void)stride;
  (void)numFramesToRead;
  (void)includeSensorData;
  (void)includeProcessingPasses;

  std::vector<std::shared_ptr<Frame>> result;

  // 1. Open the file
  FILE* file = fopen(mPath.c_str(), "r");

  long linearFrameStart = 0;
  for (int i = 0; i < trial; i++)
  {
    const int pastTrialNumPasses = getTrialNumProcessingPasses(i);
    const int pastTrialFrameSize
        = (mSensorFrameSize + (pastTrialNumPasses * mProcessingPassFrameSize));
    linearFrameStart += getTrialLength(i) * pastTrialFrameSize;
  }
  const int numPasses = getTrialNumProcessingPasses(trial);
  const long frameSize
      = (mSensorFrameSize + (numPasses * mProcessingPassFrameSize));
  linearFrameStart += startFrame * frameSize;

  int remainingFrames = getTrialLength(trial) - startFrame;
  if (remainingFrames < numFramesToRead * stride)
  {
    numFramesToRead = (int)floor((s_t)remainingFrames / stride);
  }

  if (numFramesToRead <= 0)
  {
    // return an empty result
    fclose(file);
    return result;
  }

  for (int i = 0; i < numFramesToRead; i++)
  {
    // 2. Seek to the right place in the file to read this frame
    long offsetBytes
        = mDataSectionStart + (linearFrameStart + (i * stride * frameSize));

    std::shared_ptr<Frame> frame = std::make_shared<Frame>();
    if (includeSensorData)
    {
      fseek(file, offsetBytes, SEEK_SET);

      // 3. Allocate a buffer to hold the serialized data
      std::vector<char> serializedFrame(mSensorFrameSize);

      // 4. Read the serialized data from the file
      int64_t bytesRead
          = fread(serializedFrame.data(), sizeof(char), mSensorFrameSize, file);
      if (bytesRead != mSensorFrameSize)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": was unable to read full requested frame size "
            << mSensorFrameSize << " at offset " << (offsetBytes)
            << ", corresponding to sensor data frame for trial " << trial
            << " and frame " << startFrame + (i * stride) << " (" << i * stride
            << " into a " << numFramesToRead
            << " frame read), instead only got " << bytesRead << " bytes."
            << std::endl;
        throw new std::exception();
      }
      // 5. Deserialize the data into a protobuf object
      proto::SubjectOnDiskSensorFrame proto;
      bool parseSuccess = proto.ParseFromArray(
          serializedFrame.data(), serializedFrame.size());
      if (!parseSuccess)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": got an error parsing frame at offset " << offsetBytes
            << ", corresponding to sensor data frame for trial " << trial
            << " and frame " << startFrame + (i * stride) << " (" << i * stride
            << " into a " << numFramesToRead << " frame read)." << std::endl;
        throw new std::exception();
      }

      // 6. Copy the results out into a frame
      frame->readSensorsFromProto(
          &proto, *mHeader.get(), trial, startFrame + (i * stride));
    }
    if (includeProcessingPasses)
    {
      fseek(file, offsetBytes + mSensorFrameSize, SEEK_SET);

      // 3. Allocate a buffer to hold the serialized data
      std::vector<char> serializedPassFrame(mProcessingPassFrameSize);

      for (int pass = 0; pass < getTrialNumProcessingPasses(trial); pass++)
      {
        int64_t bytesRead = fread(
            serializedPassFrame.data(),
            sizeof(char),
            mProcessingPassFrameSize,
            file);
        if (bytesRead != mProcessingPassFrameSize)
        {
          std::cout
              << "SubjectOnDisk attempting to read a corrupted binary file at "
              << mPath << ": was unable to read full requested frame size "
              << mProcessingPassFrameSize << " at offset "
              << (offsetBytes + mSensorFrameSize
                  + (i * stride * mProcessingPassFrameSize))
              << ", corresponding to trial " << trial
              << " and processing pass frame " << startFrame + (i * stride)
              << " (" << i * stride << " into a " << numFramesToRead
              << " frame read), processing pass " << pass
              << ", instead only got " << bytesRead << " bytes." << std::endl;
          throw new std::exception();
        }
        // 5. Deserialize the data into a protobuf object
        proto::SubjectOnDiskProcessingPassFrame proto;
        bool parseSuccess = proto.ParseFromArray(
            serializedPassFrame.data(), serializedPassFrame.size());
        if (!parseSuccess)
        {
          std::cout
              << "SubjectOnDisk attempting to read a corrupted binary file at "
              << mPath
              << ": got an error parsing processing pass frame at offset "
              << (offsetBytes + i * stride * mSensorFrameSize)
              << ", corresponding to trial " << trial << " and frame "
              << startFrame + (i * stride) << " (" << i * stride << " into a "
              << numFramesToRead << " frame read), processing pass " << pass
              << "." << std::endl;
          throw new std::exception();
        }

        // 6. Copy the results out into a frame
        frame->processingPasses.emplace_back();
        frame->processingPasses[pass].readFromProto(
            &proto,
            *mHeader.get(),
            trial,
            startFrame + (i * stride),
            pass,
            contactThreshold);
      }
    }
    // TODO: read the processing passes separately

    result.push_back(frame);
  }

  fclose(file);

  return result;
}

void Frame::readSensorsFromProto(
    dart::proto::SubjectOnDiskSensorFrame* proto,
    const SubjectOnDiskHeader& header,
    int trial,
    int t)
{
  this->trial = trial;
  this->t = t;

  this->missingGRFReason = header.mTrials[trial]->mMissingGRFReason[t];

  // 7. Read out the marker, accelerometer, and gyro info as pairs
  for (int i = 0; i < header.mMarkerNames.size(); i++)
  {
    Eigen::Vector3s marker(
        proto->marker_obs(i * 3 + 0),
        proto->marker_obs(i * 3 + 1),
        proto->marker_obs(i * 3 + 2));
    if (!marker.hasNaN())
    {
      markerObservations.emplace_back(header.mMarkerNames[i], marker);
    }
  }
  for (int i = 0; i < header.mAccNames.size(); i++)
  {
    Eigen::Vector3s acc(
        proto->acc_obs(i * 3 + 0),
        proto->acc_obs(i * 3 + 1),
        proto->acc_obs(i * 3 + 2));
    if (!acc.hasNaN())
    {
      accObservations.emplace_back(header.mAccNames[i], acc);
    }
  }
  for (int i = 0; i < header.mGyroNames.size(); i++)
  {
    Eigen::Vector3s gyro(
        proto->gyro_obs(i * 3 + 0),
        proto->gyro_obs(i * 3 + 1),
        proto->gyro_obs(i * 3 + 2));
    if (!gyro.hasNaN())
    {
      gyroObservations.emplace_back(header.mGyroNames[i], gyro);
    }
  }
  for (int i = 0; i < header.mEmgNames.size(); i++)
  {
    Eigen::VectorXs emgSequence(header.mEmgDim);
    for (int j = 0; j < header.mEmgDim; j++)
    {
      emgSequence(j) = proto->emg_obs(i * header.mEmgDim + j);
    }
    if (!emgSequence.hasNaN())
    {
      emgSignals.emplace_back(header.mEmgNames[i], emgSequence);
    }
  }
  for (int i = 0; i < header.mExoDofIndices.size(); i++)
  {
    exoTorques.emplace_back(header.mExoDofIndices[i], proto->exo_obs(i));
  }
  int customValueCursor = 0;
  for (int i = 0; i < header.mCustomValueNames.size(); i++)
  {
    Eigen::VectorXs customValue
        = Eigen::VectorXs::Zero(header.mCustomValueLengths[i]);
    for (int d = 0; d < header.mCustomValueLengths[i]; d++)
    {
      customValue(d) = proto->custom_values(customValueCursor);
      customValueCursor++;
    }
    customValues.emplace_back(header.mCustomValueNames[i], customValue);
  }
  int maxNumForcePlates = proto->raw_force_plate_cop_size() / 3;
  for (int i = 0; i < maxNumForcePlates; i++)
  {
    Eigen::Vector3s forceCop(
        proto->raw_force_plate_cop(i * 3 + 0),
        proto->raw_force_plate_cop(i * 3 + 1),
        proto->raw_force_plate_cop(i * 3 + 2));
    Eigen::Vector3s forceTorques(
        proto->raw_force_plate_torque(i * 3 + 0),
        proto->raw_force_plate_torque(i * 3 + 1),
        proto->raw_force_plate_torque(i * 3 + 2));
    Eigen::Vector3s force(
        proto->raw_force_plate_force(i * 3 + 0),
        proto->raw_force_plate_force(i * 3 + 1),
        proto->raw_force_plate_force(i * 3 + 2));
    if (!forceCop.hasNaN() && !forceTorques.hasNaN() && !force.hasNaN())
    {
      this->rawForcePlateCenterOfPressures.push_back(forceCop);
      this->rawForcePlateTorques.push_back(forceTorques);
      this->rawForcePlateForces.push_back(force);
    }
  }
}

void FramePass::readFromProto(
    dart::proto::SubjectOnDiskProcessingPassFrame* proto,
    const SubjectOnDiskHeader& header,
    int trial,
    int t,
    int pass,
    s_t contactThreshold)
{
  (void)proto;
  (void)header;
  (void)trial;
  (void)t;
  (void)contactThreshold;

  // ProcessingPassType type;
  type = header.mPasses[pass]->mType;
  // s_t markerRMS;
  markerRMS = header.mTrials[trial]->mTrialPasses[pass]->mMarkerRMS[t];
  // s_t markerMax;
  markerMax = header.mTrials[trial]->mTrialPasses[pass]->mMarkerMax[t];
  // s_t linearResidual;
  linearResidual
      = header.mTrials[trial]->mTrialPasses[pass]->mLinearResidual[t];
  // s_t angularResidual;
  angularResidual
      = header.mTrials[trial]->mTrialPasses[pass]->mAngularResidual[t];
  // Eigen::VectorXd pos;
  pos = Eigen::VectorXs::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    pos(i) = proto->pos(i);
  }
  // Eigen::VectorXd vel;
  vel = Eigen::VectorXs::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    vel(i) = proto->vel(i);
  }
  // Eigen::VectorXd acc;
  acc = Eigen::VectorXs::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    acc(i) = proto->acc(i);
  }
  // Eigen::VectorXd tau;
  tau = Eigen::VectorXs::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    tau(i) = proto->tau(i);
  }

  int numContactBodies = header.mGroundContactBodies.size();
  // // These are boolean values (0 or 1) for each contact body indicating
  // whether
  // // or not it's in contact
  // Eigen::VectorXi contact;
  contact = Eigen::VectorXi::Zero(numContactBodies);

  // // These are each 6-vector of contact body wrenches, all concatenated
  // together
  // Eigen::VectorXd groundContactWrenches;
  groundContactWrenches = Eigen::VectorXs::Zero(numContactBodies * 6);
  for (int i = 0; i < groundContactWrenches.size(); i++)
  {
    groundContactWrenches(i) = proto->ground_contact_wrench(i);
  }

  // // These are each 3-vector for each contact body, concatenated together
  // Eigen::VectorXd groundContactCenterOfPressure;
  groundContactCenterOfPressure = Eigen::VectorXs::Zero(numContactBodies * 3);
  for (int i = 0; i < groundContactCenterOfPressure.size(); i++)
  {
    groundContactCenterOfPressure(i)
        = proto->ground_contact_center_of_pressure(i);
  }

  // Eigen::VectorXd groundContactTorque;
  groundContactTorque = Eigen::VectorXs::Zero(numContactBodies * 3);
  for (int i = 0; i < groundContactTorque.size(); i++)
  {
    groundContactTorque(i) = proto->ground_contact_torque(i);
  }

  // Eigen::VectorXd groundContactForce;
  groundContactForce = Eigen::VectorXs::Zero(numContactBodies * 3);
  for (int i = 0; i < groundContactForce.size(); i++)
  {
    groundContactForce(i) = proto->ground_contact_force(i);
  }

  // Set the contact values
  for (int i = 0; i < numContactBodies; i++)
  {
    contact(i) = groundContactForce.segment<3>(i * 3).norm() > contactThreshold;
  }

  // // These are the center of mass kinematics
  // Eigen::Vector3s comPos;
  comPos(0) = proto->com_pos(0);
  comPos(1) = proto->com_pos(1);
  comPos(2) = proto->com_pos(2);

  // Eigen::Vector3s comVel;
  comVel(0) = proto->com_vel(0);
  comVel(1) = proto->com_vel(1);
  comVel(2) = proto->com_vel(2);

  // Eigen::Vector3s comAcc;
  comAcc(0) = proto->com_acc(0);
  comAcc(1) = proto->com_acc(1);
  comAcc(2) = proto->com_acc(2);

  // Eigen::Vector3s comAccInRootFrame;
  if (proto->root_frame_com_acc_size() == 3)
  {
    comAccInRootFrame(0) = proto->root_frame_com_acc(0);
    comAccInRootFrame(1) = proto->root_frame_com_acc(1);
    comAccInRootFrame(2) = proto->root_frame_com_acc(2);
  }

  // // These are each 6-vectors of the contact wrench of each body, but
  // expressed
  // // in the world frame, all concatenated together
  // Eigen::VectorXd groundContactWrenchesInRootFrame;
  groundContactWrenchesInRootFrame
      = Eigen::VectorXs::Zero(numContactBodies * 6);
  if (proto->root_frame_ground_contact_wrench_size() == numContactBodies * 6)
  {
    for (int i = 0; i < groundContactWrenchesInRootFrame.size(); i++)
    {
      groundContactWrenchesInRootFrame(i)
          = proto->root_frame_ground_contact_wrench(i);
    }
  }

  // // These are each 3-vector for each contact body, concatenated together
  // Eigen::VectorXd groundContactCenterOfPressure;
  groundContactCenterOfPressureInRootFrame
      = Eigen::VectorXs::Zero(numContactBodies * 3);
  if (proto->root_frame_ground_contact_center_of_pressure_size()
      == numContactBodies * 3)
  {
    for (int i = 0; i < groundContactCenterOfPressureInRootFrame.size(); i++)
    {
      groundContactCenterOfPressureInRootFrame(i)
          = proto->root_frame_ground_contact_center_of_pressure(i);
    }
  }

  // Eigen::VectorXd groundContactTorque;
  groundContactTorqueInRootFrame = Eigen::VectorXs::Zero(numContactBodies * 3);
  if (proto->root_frame_ground_contact_torques_size() == numContactBodies * 3)
  {
    for (int i = 0; i < groundContactTorqueInRootFrame.size(); i++)
    {
      groundContactTorqueInRootFrame(i)
          = proto->root_frame_ground_contact_torques(i);
    }
  }

  // Eigen::VectorXd groundContactForce;
  groundContactForceInRootFrame = Eigen::VectorXs::Zero(numContactBodies * 3);
  if (proto->root_frame_ground_contact_force_size() == numContactBodies * 3)
  {
    for (int i = 0; i < groundContactForceInRootFrame.size(); i++)
    {
      groundContactForceInRootFrame(i)
          = proto->root_frame_ground_contact_force(i);
    }
  }

  // // This is the residual, expressed as a wrench in the root body (probably
  // the
  // // pelvis) frame
  // Eigen::Vector6d residualWrenchInRootFrame;
  residualWrenchInRootFrame = Eigen::Vector6s::Zero();
  if (proto->root_frame_residual_size() == 6)
  {
    for (int i = 0; i < 6; i++)
    {
      residualWrenchInRootFrame(i) = proto->root_frame_residual(i);
    }
  }

  rootLinearVelInRootFrame = Eigen::Vector3s::Zero();
  rootAngularVelInRootFrame = Eigen::Vector3s::Zero();
  if (proto->root_frame_spatial_velocity_size() == 6)
  {
    rootAngularVelInRootFrame(0) = proto->root_frame_spatial_velocity(0);
    rootAngularVelInRootFrame(1) = proto->root_frame_spatial_velocity(1);
    rootAngularVelInRootFrame(2) = proto->root_frame_spatial_velocity(2);
    rootLinearVelInRootFrame(0) = proto->root_frame_spatial_velocity(3);
    rootLinearVelInRootFrame(1) = proto->root_frame_spatial_velocity(4);
    rootLinearVelInRootFrame(2) = proto->root_frame_spatial_velocity(5);
  }

  rootLinearAccInRootFrame = Eigen::Vector3s::Zero();
  rootAngularAccInRootFrame = Eigen::Vector3s::Zero();
  if (proto->root_frame_spatial_acceleration_size() == 6)
  {
    rootAngularAccInRootFrame(0) = proto->root_frame_spatial_acceleration(0);
    rootAngularAccInRootFrame(1) = proto->root_frame_spatial_acceleration(1);
    rootAngularAccInRootFrame(2) = proto->root_frame_spatial_acceleration(2);
    rootLinearAccInRootFrame(0) = proto->root_frame_spatial_acceleration(3);
    rootLinearAccInRootFrame(1) = proto->root_frame_spatial_acceleration(4);
    rootLinearAccInRootFrame(2) = proto->root_frame_spatial_acceleration(5);
  }

  rootPosHistoryInRootFrame
      = Eigen::VectorXs::Zero(proto->root_frame_root_pos_history_size());
  for (int i = 0; i < proto->root_frame_root_pos_history_size(); i++)
  {
    rootPosHistoryInRootFrame(i) = proto->root_frame_root_pos_history(i);
  }
  rootEulerHistoryInRootFrame
      = Eigen::VectorXs::Zero(proto->root_frame_root_euler_history_size());
  for (int i = 0; i < proto->root_frame_root_euler_history_size(); i++)
  {
    rootEulerHistoryInRootFrame(i) = proto->root_frame_root_euler_history(i);
  }

  // // These are the joint centers, expressed in the world frame
  // Eigen::VectorXd jointCenters;
  jointCenters = Eigen::VectorXd::Zero(proto->world_frame_joint_centers_size());
  for (int i = 0; i < proto->world_frame_joint_centers_size(); i++)
  {
    jointCenters(i) = proto->world_frame_joint_centers(i);
  }

  // // These are the joint centers, expressed in the root body (probably the
  // // pelvis) frame
  // Eigen::VectorXd jointCentersInRootFrame;
  jointCentersInRootFrame
      = Eigen::VectorXd::Zero(proto->root_frame_joint_centers_size());
  for (int i = 0; i < proto->root_frame_joint_centers_size(); i++)
  {
    jointCentersInRootFrame(i) = proto->root_frame_joint_centers(i);
  }

  // // These are masks for which DOFs are observed

  // Eigen::VectorXi posObserved;
  posObserved = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    posObserved(i)
        = header.mTrials[trial]->mTrialPasses[pass]->mDofPositionsObserved[i];
  }

  // // These are masks for which DOFs have been finite differenced (if they
  // // haven't been finite differenced, they're from real sensors and therefore
  // // more trustworthy)
  // Eigen::VectorXi velFiniteDifferenced;
  velFiniteDifferenced = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    velFiniteDifferenced(i) = header.mTrials[trial]
                                  ->mTrialPasses[pass]
                                  ->mDofVelocitiesFiniteDifferenced[i];
  }

  // Eigen::VectorXi accFiniteDifferenced;
  accFiniteDifferenced = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    accFiniteDifferenced(i) = header.mTrials[trial]
                                  ->mTrialPasses[pass]
                                  ->mDofAccelerationFiniteDifferenced[i];
  }
}

/// This returns the number of trials on the subject
int SubjectOnDisk::getNumTrials()
{
  return mHeader->mTrials.size();
}

/// This returns the length of the trial
int SubjectOnDisk::getTrialLength(int trial)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return 0;
  }
  return mHeader->mTrials[trial]->mLength;
}

/// This returns the original name of the trial before it was (potentially)
/// split into multiple pieces
std::string SubjectOnDisk::getTrialOriginalName(int trial)
{
  return mHeader->mTrials[trial]->mOriginalTrialName;
}

/// This returns the index of the split, if this trial was the result of
/// splitting an original trial into multiple pieces
int SubjectOnDisk::getTrialSplitIndex(int trial)
{
  return mHeader->mTrials[trial]->mSplitIndex;
}

/// This returns the number of processing passes in the trial
int SubjectOnDisk::getTrialNumProcessingPasses(int trial)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return 0;
  }
  return mHeader->mTrials[trial]->mTrialPasses.size();
}

/// This returns the timestep size for the trial
s_t SubjectOnDisk::getTrialTimestep(int trial)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return 0.01;
  }
  return mHeader->mTrials[trial]->mTimestep;
}

/// This returns the number of DOFs for the model on this Subject
int SubjectOnDisk::getNumDofs()
{
  return mHeader->mNumDofs;
}

/// This returns the number of joints for the model on this Subject
int SubjectOnDisk::getNumJoints()
{
  return mHeader->mNumJoints;
}

/// This returns the vector of enums of type 'MissingGRFReason', which can
/// include `notMissingGRF`.
std::vector<MissingGRFReason> SubjectOnDisk::getMissingGRF(int trial)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<MissingGRFReason>();
  }
  return mHeader->mTrials[trial]->mMissingGRFReason;
}

int SubjectOnDisk::getNumProcessingPasses()
{
  return mHeader->mPasses.size();
}

ProcessingPassType SubjectOnDisk::getProcessingPassType(int processingPass)
{
  return mHeader->mPasses[processingPass]->mType;
}

std::vector<bool> SubjectOnDisk::getDofPositionsObserved(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mDofPositionsObserved;
}

std::vector<bool> SubjectOnDisk::getDofVelocitiesFiniteDifferenced(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mDofVelocitiesFiniteDifferenced;
}

std::vector<bool> SubjectOnDisk::getDofAccelerationsFiniteDifferenced(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mDofAccelerationFiniteDifferenced;
}

std::vector<s_t> SubjectOnDisk::getTrialLinearResidualNorms(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader->mTrials[trial]->mTrialPasses[processingPass]->mLinearResidual;
}

std::vector<s_t> SubjectOnDisk::getTrialAngularResidualNorms(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mAngularResidual;
}

std::vector<s_t> SubjectOnDisk::getTrialMarkerRMSs(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader->mTrials[trial]->mTrialPasses[processingPass]->mMarkerRMS;
}

std::vector<s_t> SubjectOnDisk::getTrialMarkerMaxs(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader->mTrials[trial]->mTrialPasses[processingPass]->mMarkerMax;
}

/// This returns the maximum absolute velocity of any DOF at each timestep for a
/// given trial
std::vector<s_t> SubjectOnDisk::getTrialMaxJointVelocity(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader->mTrials[trial]
      ->mTrialPasses[processingPass]
      ->mJointsMaxVelocity;
}

/// This returns the list of contact body names for this Subject
std::vector<std::string> SubjectOnDisk::getGroundForceBodies()
{
  return mHeader->mGroundContactBodies;
}

/// This returns the list of custom value names stored in this subject
std::vector<std::string> SubjectOnDisk::getCustomValues()
{
  return mHeader->mCustomValueNames;
}

/// This returns the dimension of the custom value specified by `valueName`
int SubjectOnDisk::getCustomValueDim(std::string valueName)
{
  for (int i = 0; i < mHeader->mCustomValueNames.size(); i++)
  {
    if (mHeader->mCustomValueNames[i] == valueName)
    {
      return mHeader->mCustomValueLengths[i];
    }
  }
  std::cout << "WARNING: Requested getCustomValueDim() for value \""
            << valueName
            << "\", which is not in this SubjectOnDisk. Options are: [";
  for (int i = 0; i < mHeader->mCustomValueNames.size(); i++)
  {
    std::cout << " \"" << mHeader->mCustomValueNames[i] << "\" ";
  }
  std::cout << "]. Returning 0." << std::endl;
  return 0;
}

/// The name of the trial, if provided, or else an empty string
std::string SubjectOnDisk::getTrialName(int trial)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return "";
  }
  return mHeader->mTrials[trial]->mName;
}

std::string SubjectOnDisk::getBiologicalSex()
{
  return mHeader->mBiologicalSex;
}

double SubjectOnDisk::getHeightM()
{
  return mHeader->mHeightM;
}

double SubjectOnDisk::getMassKg()
{
  return mHeader->mMassKg;
}

/// This gets the tags associated with the subject, if there are any.
std::vector<std::string> SubjectOnDisk::getSubjectTags()
{
  return mHeader->mSubjectTags;
}

/// This gets the tags associated with the trial, if there are any.
std::vector<std::string> SubjectOnDisk::getTrialTags(int trial)
{
  if (trial >= 0 && trial < mHeader->mTrials.size())
  {
    return mHeader->mTrials[trial]->mTrialTags;
  }
  else
  {
    return std::vector<std::string>();
  }
}

int SubjectOnDisk::getAgeYears()
{
  return mHeader->mAgeYears;
}

/// This returns the number of raw force plates that were used to generate the
/// data
int SubjectOnDisk::getNumForcePlates(int trial)
{
  if (trial >= 0 && trial < mHeader->mTrials.size())
    return mHeader->mTrials[trial]->mNumForcePlates;
  return 0;
}

/// This returns the corners (in 3D space) of the selected force plate, for
/// this trial. Empty arrays on out of bounds.
std::vector<Eigen::Vector3s> SubjectOnDisk::getForcePlateCorners(
    int trial, int forcePlate)
{
  if (trial < 0 || trial >= mHeader->mTrials.size())
  {
    return std::vector<Eigen::Vector3s>();
  }
  if (forcePlate < 0 || forcePlate >= mHeader->mTrials[trial]->mNumForcePlates)
  {
    return std::vector<Eigen::Vector3s>();
  }
  return mHeader->mTrials[trial]->mForcePlateCorners[forcePlate];
}

/// This gets the href link associated with the subject, if there is one.
std::string SubjectOnDisk::getHref()
{
  return mHeader->mHref;
}

/// This gets the notes associated with the subject, if there are any.
std::string SubjectOnDisk::getNotes()
{
  return mHeader->mNotes;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Builders, to create a SubjectOnDisk from scratch
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SubjectOnDiskTrialPass::SubjectOnDiskTrialPass()
{
}

void SubjectOnDiskTrialPass::setType(ProcessingPassType type)
{
  mType = type;
}

void SubjectOnDiskTrialPass::setPoses(Eigen::MatrixXs poses)
{
  mPos = poses;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getPoses()
{
  return mPos;
}

void SubjectOnDiskTrialPass::setVels(Eigen::MatrixXs vels)
{
  mVel = vels;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getVels()
{
  return mVel;
}

void SubjectOnDiskTrialPass::setAccs(Eigen::MatrixXs accs)
{
  mAcc = accs;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getAccs()
{
  return mAcc;
}

void SubjectOnDiskTrialPass::setTaus(Eigen::MatrixXs taus)
{
  mTaus = taus;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getTaus()
{
  return mTaus;
}

void SubjectOnDiskTrialPass::setGroundBodyWrenches(Eigen::MatrixXs wrenches)
{
  mGroundBodyWrenches = wrenches;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getGroundBodyWrenches()
{
  return mGroundBodyWrenches;
}

void SubjectOnDiskTrialPass::setGroundBodyCopTorqueForce(
    Eigen::MatrixXs copTorqueForces)
{
  mGroundBodyCopTorqueForce = copTorqueForces;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getGroundBodyCopTorqueForce()
{
  return mGroundBodyCopTorqueForce;
}

void SubjectOnDiskTrialPass::setComPoses(Eigen::MatrixXs poses)
{
  mComPoses = poses;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getComPoses()
{
  return mComPoses;
}

void SubjectOnDiskTrialPass::setComVels(Eigen::MatrixXs vels)
{
  mComVels = vels;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getComVels()
{
  return mComVels;
}

void SubjectOnDiskTrialPass::setComAccs(Eigen::MatrixXs accs)
{
  mComAccs = accs;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getComAccs()
{
  return mComAccs;
}

void SubjectOnDiskTrialPass::setComAccsInRootFrame(Eigen::MatrixXs accs)
{
  mComAccsInRootFrame = accs;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getComAccsInRootFrame()
{
  return mComAccsInRootFrame;
}

void SubjectOnDiskTrialPass::setResidualWrenchInRootFrame(
    Eigen::MatrixXs wrenches)
{
  mResidualWrenchInRootFrame = wrenches;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getResidualWrenchInRootFrame()
{
  return mResidualWrenchInRootFrame;
}

void SubjectOnDiskTrialPass::setGroundBodyWrenchesInRootFrame(
    Eigen::MatrixXs wrenches)
{
  mGroundBodyWrenchesInRootFrame = wrenches;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getGroundBodyWrenchesInRootFrame()
{
  return mGroundBodyWrenchesInRootFrame;
}

void SubjectOnDiskTrialPass::setGroundBodyCopTorqueForceInRootFrame(
    Eigen::MatrixXs copTorqueForces)
{
  mGroundBodyCopTorqueForceInRootFrame = copTorqueForces;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getGroundBodyCopTorqueForceInRootFrame()
{
  return mGroundBodyCopTorqueForceInRootFrame;
}

void SubjectOnDiskTrialPass::setJointCenters(Eigen::MatrixXs centers)
{
  mJointCenters = centers;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getJointCenters()
{
  return mJointCenters;
}

void SubjectOnDiskTrialPass::setJointCentersInRootFrame(Eigen::MatrixXs centers)
{
  mJointCentersInRootFrame = centers;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getJointCentersInRootFrame()
{
  return mJointCentersInRootFrame;
}

void SubjectOnDiskTrialPass::setRootSpatialVelInRootFrame(
    Eigen::MatrixXs spatialVel)
{
  mRootSpatialVelInRootFrame = spatialVel;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getRootSpatialVelInRootFrame()
{
  return mRootSpatialVelInRootFrame;
}

void SubjectOnDiskTrialPass::setRootSpatialAccInRootFrame(
    Eigen::MatrixXs spatialAcc)
{
  mRootSpatialAccInRootFrame = spatialAcc;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getRootSpatialAccInRootFrame()
{
  return mRootSpatialAccInRootFrame;
}

void SubjectOnDiskTrialPass::setRootPosHistoryInRootFrame(
    Eigen::MatrixXs rootHistory)
{
  mRootPosHistoryInRootFrame = rootHistory;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getRootPosHistoryInRootFrame()
{
  return mRootPosHistoryInRootFrame;
}

void SubjectOnDiskTrialPass::setRootEulerHistoryInRootFrame(
    Eigen::MatrixXs rootHistory)
{
  mRootEulerHistoryInRootFrame = rootHistory;
}

Eigen::MatrixXs SubjectOnDiskTrialPass::getRootEulerHistoryInRootFrame()
{
  return mRootEulerHistoryInRootFrame;
}

// This will return a matrix where every one of our properties with setters is
// stacked together vertically. Each column represents time, and each row is a
// different property of interest. The point here is not to introspect into
// the individual rows, but to have a convenient object that we can resample
// to a new timestep length, and possibly lowpass filter.
Eigen::MatrixXs SubjectOnDiskTrialPass::getResamplingMatrix()
{
  int rows = 4 + mPos.rows() + mVel.rows() + mAcc.rows() + mTaus.rows()
             + mGroundBodyWrenches.rows() + mGroundBodyCopTorqueForce.rows()
             + mComPoses.rows() + mComVels.rows() + mComAccs.rows()
             + mComAccsInRootFrame.rows() + mResidualWrenchInRootFrame.rows()
             + mGroundBodyWrenchesInRootFrame.rows()
             + mGroundBodyCopTorqueForceInRootFrame.rows()
             + mJointCenters.rows() + mJointCentersInRootFrame.rows();
  int timesteps = mPos.cols();
  if (mMarkerRMS.size() != timesteps)
  {
    std::cout << "ERROR: mMarkerRMS.size() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mMarkerMax.size() != timesteps)
  {
    std::cout << "ERROR: mMarkerMax.size() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mLinearResidual.size() != timesteps)
  {
    std::cout << "ERROR: mLinearResidual.size() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mAngularResidual.size() != timesteps)
  {
    std::cout << "ERROR: mAngularResidual.size() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mVel.cols() != timesteps)
  {
    std::cout << "ERROR: mVel.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mAcc.cols() != timesteps)
  {
    std::cout << "ERROR: mAcc.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mTaus.cols() != timesteps)
  {
    std::cout << "ERROR: mTaus.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mGroundBodyWrenches.cols() != timesteps)
  {
    std::cout << "ERROR: mGroundBodyWrenches.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mGroundBodyCopTorqueForce.cols() != timesteps)
  {
    std::cout << "ERROR: mGroundBodyCopTorqueForce.cols() != timesteps"
              << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mComPoses.cols() != timesteps)
  {
    std::cout << "ERROR: mComPoses.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mComVels.cols() != timesteps)
  {
    std::cout << "ERROR: mComVels.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mComAccs.cols() != timesteps)
  {
    std::cout << "ERROR: mComAccs.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mComAccsInRootFrame.cols() != timesteps)
  {
    std::cout << "ERROR: mComAccsInRootFrame.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mResidualWrenchInRootFrame.cols() != timesteps)
  {
    std::cout << "ERROR: mResidualWrenchInRootFrame.cols() != timesteps"
              << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mGroundBodyWrenchesInRootFrame.cols() != timesteps)
  {
    std::cout << "ERROR: mGroundBodyWrenchesInRootFrame.cols() != timesteps"
              << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mGroundBodyCopTorqueForceInRootFrame.cols() != timesteps)
  {
    std::cout << "ERROR: mGroundBodyCopTorqueForceInRootFrame.cols() != "
                 "timesteps"
              << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mJointCenters.cols() != timesteps)
  {
    std::cout << "ERROR: mJointCenters.cols() != timesteps" << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  if (mJointCentersInRootFrame.cols() != timesteps)
  {
    std::cout << "ERROR: mJointCentersInRootFrame.cols() != timesteps"
              << std::endl;
    return Eigen::MatrixXs::Zero(0, 0);
  }
  Eigen::MatrixXs resamplingMatrix = Eigen::MatrixXs::Zero(rows, timesteps);
  int cursor = 0;
  for (int t = 0; t < mMarkerRMS.size(); t++)
  {
    resamplingMatrix(cursor, t) = mMarkerRMS[t];
  }
  cursor++;
  for (int t = 0; t < mMarkerMax.size(); t++)
  {
    resamplingMatrix(cursor, t) = mMarkerMax[t];
  }
  cursor++;
  for (int t = 0; t < mLinearResidual.size(); t++)
  {
    resamplingMatrix(cursor, t) = mLinearResidual[t];
  }
  cursor++;
  for (int t = 0; t < mAngularResidual.size(); t++)
  {
    resamplingMatrix(cursor, t) = mAngularResidual[t];
  }
  cursor++;
  resamplingMatrix.block(cursor, 0, mPos.rows(), timesteps) = mPos;
  cursor += mPos.rows();
  resamplingMatrix.block(cursor, 0, mVel.rows(), timesteps) = mVel;
  cursor += mVel.rows();
  resamplingMatrix.block(cursor, 0, mAcc.rows(), timesteps) = mAcc;
  cursor += mAcc.rows();
  resamplingMatrix.block(cursor, 0, mTaus.rows(), timesteps) = mTaus;
  cursor += mTaus.rows();
  resamplingMatrix.block(cursor, 0, mGroundBodyWrenches.rows(), timesteps)
      = mGroundBodyWrenches;
  cursor += mGroundBodyWrenches.rows();
  resamplingMatrix.block(cursor, 0, mGroundBodyCopTorqueForce.rows(), timesteps)
      = mGroundBodyCopTorqueForce;
  cursor += mGroundBodyCopTorqueForce.rows();
  resamplingMatrix.block(cursor, 0, mComPoses.rows(), timesteps) = mComPoses;
  cursor += mComPoses.rows();
  resamplingMatrix.block(cursor, 0, mComVels.rows(), timesteps) = mComVels;
  cursor += mComVels.rows();
  resamplingMatrix.block(cursor, 0, mComAccs.rows(), timesteps) = mComAccs;
  cursor += mComAccs.rows();
  resamplingMatrix.block(cursor, 0, mComAccsInRootFrame.rows(), timesteps)
      = mComAccsInRootFrame;
  cursor += mComAccsInRootFrame.rows();
  resamplingMatrix.block(
      cursor, 0, mResidualWrenchInRootFrame.rows(), timesteps)
      = mResidualWrenchInRootFrame;
  cursor += mResidualWrenchInRootFrame.rows();
  resamplingMatrix.block(
      cursor, 0, mGroundBodyWrenchesInRootFrame.rows(), timesteps)
      = mGroundBodyWrenchesInRootFrame;
  cursor += mGroundBodyWrenchesInRootFrame.rows();
  resamplingMatrix.block(
      cursor, 0, mGroundBodyCopTorqueForceInRootFrame.rows(), timesteps)
      = mGroundBodyCopTorqueForceInRootFrame;
  cursor += mGroundBodyCopTorqueForceInRootFrame.rows();
  resamplingMatrix.block(cursor, 0, mJointCenters.rows(), timesteps)
      = mJointCenters;
  cursor += mJointCenters.rows();
  resamplingMatrix.block(cursor, 0, mJointCentersInRootFrame.rows(), timesteps)
      = mJointCentersInRootFrame;
  cursor += mJointCentersInRootFrame.rows();
  return resamplingMatrix;
}

// This is the setter for the matrix you get from `getResamplingMatrix()`,
// after you've finished modifying it.
void SubjectOnDiskTrialPass::setResamplingMatrix(Eigen::MatrixXs matrix)
{
  int rows = 4 + mPos.rows() + mVel.rows() + mAcc.rows() + mTaus.rows()
             + mGroundBodyWrenches.rows() + mGroundBodyCopTorqueForce.rows()
             + mComPoses.rows() + mComVels.rows() + mComAccs.rows()
             + mComAccsInRootFrame.rows() + mResidualWrenchInRootFrame.rows()
             + mGroundBodyWrenchesInRootFrame.rows()
             + mGroundBodyCopTorqueForceInRootFrame.rows()
             + mJointCenters.rows() + mJointCentersInRootFrame.rows();
  if (matrix.rows() != rows)
  {
    std::cout << "ERROR: matrix.rows() != expected number of rows" << std::endl;
    return;
  }

  int cursor = 0;

  mMarkerRMS.resize(matrix.cols());
  for (int t = 0; t < mMarkerRMS.size(); t++)
  {
    mMarkerRMS[t] = matrix(cursor, t);
  }
  cursor++;

  mMarkerMax.resize(matrix.cols());
  for (int t = 0; t < mMarkerMax.size(); t++)
  {
    mMarkerMax[t] = matrix(cursor, t);
  }
  cursor++;

  mLinearResidual.resize(matrix.cols());
  for (int t = 0; t < mLinearResidual.size(); t++)
  {
    mLinearResidual[t] = matrix(cursor, t);
  }
  cursor++;

  mAngularResidual.resize(matrix.cols());
  for (int t = 0; t < mAngularResidual.size(); t++)
  {
    mAngularResidual[t] = matrix(cursor, t);
  }
  cursor++;

  mPos = matrix.block(cursor, 0, mPos.rows(), matrix.cols());
  cursor += mPos.rows();

  mVel = matrix.block(cursor, 0, mVel.rows(), matrix.cols());
  cursor += mVel.rows();

  mAcc = matrix.block(cursor, 0, mAcc.rows(), matrix.cols());
  cursor += mAcc.rows();

  mTaus = matrix.block(cursor, 0, mTaus.rows(), matrix.cols());
  cursor += mTaus.rows();

  mGroundBodyWrenches
      = matrix.block(cursor, 0, mGroundBodyWrenches.rows(), matrix.cols());
  cursor += mGroundBodyWrenches.rows();

  mGroundBodyCopTorqueForce = matrix.block(
      cursor, 0, mGroundBodyCopTorqueForce.rows(), matrix.cols());
  cursor += mGroundBodyCopTorqueForce.rows();

  mComPoses = matrix.block(cursor, 0, mComPoses.rows(), matrix.cols());
  cursor += mComPoses.rows();

  mComVels = matrix.block(cursor, 0, mComVels.rows(), matrix.cols());
  cursor += mComVels.rows();

  mComAccs = matrix.block(cursor, 0, mComAccs.rows(), matrix.cols());
  cursor += mComAccs.rows();

  mComAccsInRootFrame
      = matrix.block(cursor, 0, mComAccsInRootFrame.rows(), matrix.cols());
  cursor += mComAccsInRootFrame.rows();

  mResidualWrenchInRootFrame = matrix.block(
      cursor, 0, mResidualWrenchInRootFrame.rows(), matrix.cols());
  cursor += mResidualWrenchInRootFrame.rows();

  mGroundBodyWrenchesInRootFrame = matrix.block(
      cursor, 0, mGroundBodyWrenchesInRootFrame.rows(), matrix.cols());
  cursor += mGroundBodyWrenchesInRootFrame.rows();

  mGroundBodyCopTorqueForceInRootFrame = matrix.block(
      cursor, 0, mGroundBodyCopTorqueForceInRootFrame.rows(), matrix.cols());
  cursor += mGroundBodyCopTorqueForceInRootFrame.rows();

  mJointCenters = matrix.block(cursor, 0, mJointCenters.rows(), matrix.cols());
  cursor += mJointCenters.rows();

  mJointCentersInRootFrame
      = matrix.block(cursor, 0, mJointCentersInRootFrame.rows(), matrix.cols());
  cursor += mJointCentersInRootFrame.rows();
}

void SubjectOnDiskTrialPass::setDofPositionsObserved(
    std::vector<bool> dofPositionsObserved)
{
  mDofPositionsObserved = dofPositionsObserved;
}

void SubjectOnDiskTrialPass::setDofVelocitiesFiniteDifferenced(
    std::vector<bool> dofVelocitiesFiniteDifferenced)
{
  mDofVelocitiesFiniteDifferenced = dofVelocitiesFiniteDifferenced;
}

void SubjectOnDiskTrialPass::setDofAccelerationFiniteDifferenced(
    std::vector<bool> dofAccelerationFiniteDifference)
{
  mDofAccelerationFiniteDifferenced = dofAccelerationFiniteDifference;
}

// This is for allowing the user to set all the values of a pass at once,
// without having to manually compute them in Python, which turns out to be
// slow and difficult to test.
void SubjectOnDiskTrialPass::computeValues(
    std::shared_ptr<dynamics::Skeleton> skel,
    s_t timestep,
    Eigen::MatrixXs poses,
    std::vector<std::string> footBodyNames,
    Eigen::MatrixXs forces,
    Eigen::MatrixXs moments,
    Eigen::MatrixXs cops,
    int rootHistoryLen,
    int rootHistoryStride)
{
  std::vector<ForcePlate> forcePlates;
  int numForcePlates = forces.rows() / 3;
  for (int i = 0; i < numForcePlates; i++)
  {
    forcePlates.emplace_back();
    for (int t = 0; t < forces.cols(); t++)
    {
      forcePlates[forcePlates.size() - 1].forces.push_back(
          forces.block<3, 1>(i * 3, t));
      forcePlates[forcePlates.size() - 1].centersOfPressure.push_back(
          cops.block<3, 1>(i * 3, t));
      forcePlates[forcePlates.size() - 1].moments.push_back(
          moments.block<3, 1>(i * 3, t));
    }
  }

  computeValuesFromForcePlates(
      skel,
      timestep,
      poses,
      footBodyNames,
      forcePlates,
      rootHistoryLen,
      rootHistoryStride);
}

// This is for allowing the user to set all the values of a pass at once,
// without having to manually compute them in Python, which turns out to be
// slow and difficult to test.
void SubjectOnDiskTrialPass::computeValuesFromForcePlates(
    std::shared_ptr<dynamics::Skeleton> skel,
    s_t timestep,
    Eigen::MatrixXs poses,
    std::vector<std::string> footBodyNames,
    std::vector<ForcePlate> forcePlates,
    int rootHistoryLen,
    int rootHistoryStride,
    Eigen::MatrixXs explicitVels,
    Eigen::MatrixXs explicitAccs)
{
  Eigen::MatrixXs grfTrial
      = Eigen::MatrixXs::Zero(6 * footBodyNames.size(), poses.cols());
  Eigen::MatrixXs copTorqueForceTrial
      = Eigen::MatrixXs::Zero(9 * footBodyNames.size(), poses.cols());
  Eigen::MatrixXs copTorqueForceTrialInRootFrame
      = Eigen::MatrixXs::Zero(9 * footBodyNames.size(), poses.cols());

  // 1. We need to assign the force plates to feet, and compute the total
  // wrenches applied to each foot

  std::vector<int> footIndices;
  std::vector<dynamics::BodyNode*> footBodies;
  for (std::string footName : footBodyNames)
  {
    dynamics::BodyNode* footBody = skel->getBodyNode(footName);
    footBodies.push_back(footBody);
    footIndices.push_back(footBody->getIndexInSkeleton());
  }
  std::vector<std::vector<int>> forcePlatesAssignedToContactBody;
  for (int i = 0; i < forcePlates.size(); i++)
  {
    forcePlatesAssignedToContactBody.emplace_back();
    for (int t = 0; t < poses.cols(); t++)
    {
      forcePlatesAssignedToContactBody
          [forcePlatesAssignedToContactBody.size() - 1]
              .push_back(0);
    }
  }
  DynamicsFitter::recomputeGRFs(
      forcePlates,
      poses,
      footBodies,
      std::vector<int>(),
      forcePlatesAssignedToContactBody,
      grfTrial,
      skel);

  // 2. We need to actually run through all the frames and compute the aggregate
  // values

  std::vector<s_t> linearResiduals;
  std::vector<s_t> angularResiduals;
  Eigen::MatrixXs vels
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), poses.cols());
  if (explicitVels.cols() == poses.cols()
      && explicitVels.rows() == poses.rows())
  {
    vels = explicitVels;
  }
  Eigen::MatrixXs accs
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), poses.cols());
  if (explicitAccs.cols() == poses.cols()
      && explicitAccs.rows() == poses.rows())
  {
    accs = explicitAccs;
  }
  Eigen::MatrixXs taus
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), poses.cols());
  Eigen::MatrixXs comPoses = Eigen::MatrixXs::Zero(3, poses.cols());
  Eigen::MatrixXs comVels = Eigen::MatrixXs::Zero(3, poses.cols());
  Eigen::MatrixXs comAccs = Eigen::MatrixXs::Zero(3, poses.cols());
  Eigen::MatrixXs comAccsInRootFrame = Eigen::MatrixXs::Zero(3, poses.cols());
  Eigen::MatrixXs residualWrenchInRootFrame
      = Eigen::MatrixXs::Zero(6, poses.cols());
  Eigen::MatrixXs groundBodyWrenchesInRootFrame
      = Eigen::MatrixXs::Zero(6 * footBodyNames.size(), poses.cols());
  Eigen::MatrixXs jointCenters
      = Eigen::MatrixXs::Zero(skel->getNumJoints() * 3, poses.cols());
  Eigen::MatrixXs jointCentersInRootFrame
      = Eigen::MatrixXs::Zero(skel->getNumJoints() * 3, poses.cols());
  Eigen::MatrixXs rootSpatialVelInRootFrame
      = Eigen::MatrixXs::Zero(6, poses.cols());
  Eigen::MatrixXs rootSpatialAccInRootFrame
      = Eigen::MatrixXs::Zero(6, poses.cols());
  Eigen::MatrixXs rootPosHistoryInRootFrame
      = Eigen::MatrixXs::Zero(3 * rootHistoryLen, poses.cols());
  Eigen::MatrixXs rootEulerHistoryInRootFrame
      = Eigen::MatrixXs::Zero(3 * rootHistoryLen, poses.cols());

  ResidualForceHelper helper(skel, footIndices);
  s_t dt = timestep;
  std::vector<Eigen::Isometry3s> rootTransforms;
  for (int t = 0; t < poses.cols(); t++)
  {
    Eigen::VectorXs q = poses.col(t);
    skel->setPositions(q);
    comPoses.col(t) = skel->getCOM();
    Eigen::Isometry3s T_wr = skel->getRootBodyNode()->getWorldTransform();
    rootTransforms.push_back(T_wr);

    Eigen::VectorXs worldCenters
        = skel->getJointWorldPositions(skel->getJoints());
    jointCenters.col(t) = worldCenters;
    for (int j = 0; j < skel->getNumJoints(); j++)
    {
      jointCentersInRootFrame.block<3, 1>(j * 3, t)
          = T_wr.inverse() * worldCenters.segment<3>(j * 3);
    }

    s_t linearResidual = 0.0;
    s_t angularResidual = 0.0;
    if (t > 0)
    {
      if (explicitVels.cols() == poses.cols()
          && explicitVels.rows() == poses.rows())
      {
        // Do nothing
      }
      else
      {
        vels.col(t)
            = skel->getPositionDifferences(poses.col(t), poses.col(t - 1)) / dt;
      }
      Eigen::VectorXs dq = vels.col(t);
      skel->setVelocities(dq);
      comVels.col(t) = skel->getCOMLinearVelocity();
      if (skel->getRootJoint() != nullptr
          && skel->getRootJoint()->getNumDofs() == 6)
      {
        const Eigen::Vector6s rootSpatialVel
            = skel->getRootJoint()->getRelativeJacobian() * dq.head<6>();
        const Eigen::Vector3s rootAngVel = rootSpatialVel.head<3>();
        rootSpatialVelInRootFrame.col(t).head<3>() = rootAngVel;
        const Eigen::Vector3s rootLinVel = rootSpatialVel.tail<3>();
        rootSpatialVelInRootFrame.col(t).tail<3>() = rootLinVel;
      }

      if (t < poses.cols() - 1)
      {
        Eigen::VectorXs ddq;
        if (explicitAccs.cols() == poses.cols()
            && explicitAccs.rows() == poses.rows())
        {
          ddq = explicitAccs.col(t);
        }
        else
        {
          ddq = (skel->getPositionDifferences(poses.col(t + 1), poses.col(t))
                 - skel->getPositionDifferences(poses.col(t), poses.col(t - 1)))
                / (dt * dt);
        }
        Eigen::VectorXs tau
            = helper.calculateInverseDynamics(q, dq, ddq, grfTrial.col(t));
        Eigen::Vector6s residual = tau.head<6>();
        angularResidual = residual.head<3>().norm();
        linearResidual = residual.tail<3>().norm();

        accs.col(t) = ddq;
        taus.col(t) = tau;

        skel->setAccelerations(ddq);
        comAccs.col(t) = skel->getCOMLinearAcceleration() - skel->getGravity();
        comAccsInRootFrame.col(t) = T_wr.linear().transpose() * comAccs.col(t);

        if (skel->getRootJoint()->getNumDofs() == 6)
        {
          const Eigen::MatrixXs rootJac
              = skel->getRootJoint()->getRelativeJacobian();
          const Eigen::Vector6s rootSpatialAcc = rootJac * ddq.head<6>();
          const Eigen::Vector3s rootAngAcc = rootSpatialAcc.head<3>();
          rootSpatialAccInRootFrame.col(t).head<3>() = rootAngAcc;
          const Eigen::Vector3s rootLinAcc
              = rootSpatialAcc.tail<3>()
                - (T_wr.linear().transpose() * skel->getGravity());
          rootSpatialAccInRootFrame.col(t).tail<3>() = rootLinAcc;
          Eigen::Matrix6s rootJacobianTransposeInverse
              = skel->getRootJoint()
                    ->getRelativeJacobian()
                    .transpose()
                    .completeOrthogonalDecomposition()
                    .pseudoInverse();
          residualWrenchInRootFrame.col(t)
              = rootJacobianTransposeInverse * residual;
        }

        for (int i = 0; i < footIndices.size(); i++)
        {
          // Estimate ground height from recorded CoPs, for later CoP
          // calculations
          s_t groundHeight = 0.0;
          if (forcePlates.size() > 0)
          {
            for (int f = 0; f < forcePlates.size(); f++)
            {
              if (forcePlates.at(f).centersOfPressure.size() > t
                  && !forcePlates.at(f).centersOfPressure.at(t).hasNaN()
                  && forcePlates.at(f).forces.size() > t
                  && forcePlates.at(f).forces.at(t).norm() > 1e-8
                  // We want this force plate to be assigned to this body at
                  // this frame, or else it doesn't count
                  && forcePlatesAssignedToContactBody.at(f).at(t) == i)
              {
                groundHeight = forcePlates.at(f).centersOfPressure.at(t)(1);
              }
            }
          }

          Eigen::Vector6s worldWrench = grfTrial.block<6, 1>(i * 6, t);
          Eigen::Vector9s copWrench
              = math::projectWrenchToCoP(worldWrench, groundHeight, 1);
          copTorqueForceTrial.block<9, 1>(i * 9, t) = copWrench;
          Eigen::Vector6s rootWrench
              = math::dAdInvT(T_wr.inverse(), worldWrench);
          groundBodyWrenchesInRootFrame.block<6, 1>(i * 6, t) = rootWrench;

          Eigen::Vector3s copWorld = copWrench.head<3>();
          Eigen::Vector3s torqueWorld = copWrench.segment<3>(3);
          Eigen::Vector3s forceWorld = copWrench.tail<3>();
          Eigen::Vector3s copRoot = T_wr.inverse() * copWorld;
          Eigen::Vector3s torqueRoot = T_wr.linear().transpose() * torqueWorld;
          Eigen::Vector3s forceRoot = T_wr.linear().transpose() * forceWorld;
          copTorqueForceTrialInRootFrame.block<3, 1>(i * 9, t) = copRoot;
          copTorqueForceTrialInRootFrame.block<3, 1>(i * 9 + 3, t) = torqueRoot;
          copTorqueForceTrialInRootFrame.block<3, 1>(i * 9 + 6, t) = forceRoot;
        }

#ifndef NDEBUG
        // Check that inverse dynamics given these inputs produces the expected
        // joint torques

        std::vector<Eigen::Vector6s> rootFrameContactWrenches;
        for (int b = 0; b < footBodies.size(); b++)
        {
          rootFrameContactWrenches.push_back(
              groundBodyWrenchesInRootFrame.block<6, 1>(b * 6, t));
        }
        Eigen::VectorXs recoveredTau = skel->getInverseDynamicsFromPredictions(
            ddq,
            footBodies,
            rootFrameContactWrenches,
            residualWrenchInRootFrame.col(t));
        tau.head<6>().setZero();
        if ((recoveredTau - tau).norm() > 1e-8)
        {
          std::cout << "Inverse dynamics failed to recover the expected torques"
                    << std::endl;
          std::cout << "Expected: " << tau.transpose() << std::endl;
          std::cout << "Recovered: " << recoveredTau.transpose() << std::endl;
          std::cout << "Difference: " << (recoveredTau - tau).transpose()
                    << std::endl;
          assert(false);
        }
#endif
      }
    }
    linearResiduals.push_back(linearResidual);
    angularResiduals.push_back(angularResidual);
  }

  assert(poses.cols() == rootTransforms.size());
  for (int t = 0; t < rootTransforms.size(); t++)
  {
    Eigen::Isometry3s T_wr = rootTransforms[t];

    for (int reachBack = 0; reachBack < rootHistoryLen; reachBack++)
    {
      int reachBackToT = t - rootHistoryStride * reachBack;
      if (reachBackToT >= 0)
      {
        Eigen::Isometry3s T_wb = rootTransforms[reachBackToT];

        Eigen::Isometry3s T_rb = T_wr.inverse() * T_wb;
        Eigen::Vector3s rootPos = T_rb.translation();
        Eigen::Vector3s rootEuler = math::matrixToEulerXYZ(T_rb.linear());
        rootPosHistoryInRootFrame.block<3, 1>(reachBack * 3, t) = rootPos;
        rootEulerHistoryInRootFrame.block<3, 1>(reachBack * 3, t) = rootEuler;
      }
    }
  }

  setLinearResidual(linearResiduals);
  setAngularResidual(angularResiduals);
  setPoses(poses);
  setVels(vels);
  setAccs(accs);
  setTaus(taus);
  setGroundBodyWrenches(grfTrial);
  setComPoses(comPoses);
  setComVels(comVels);
  setComAccs(comAccs);
  setComAccsInRootFrame(comAccsInRootFrame);
  setResidualWrenchInRootFrame(residualWrenchInRootFrame);
  setGroundBodyWrenchesInRootFrame(groundBodyWrenchesInRootFrame);
  setGroundBodyCopTorqueForce(copTorqueForceTrial);
  setGroundBodyCopTorqueForceInRootFrame(copTorqueForceTrialInRootFrame);
  setJointCenters(jointCenters);
  setJointCentersInRootFrame(jointCentersInRootFrame);
  setRootSpatialVelInRootFrame(rootSpatialVelInRootFrame);
  setRootSpatialAccInRootFrame(rootSpatialAccInRootFrame);
  setRootPosHistoryInRootFrame(rootPosHistoryInRootFrame);
  setRootEulerHistoryInRootFrame(rootEulerHistoryInRootFrame);
}

// Manual setters that compete with computeValues()
void SubjectOnDiskTrialPass::setMarkerRMS(std::vector<s_t> markerRMS)
{
  mMarkerRMS = markerRMS;
}

std::vector<s_t> SubjectOnDiskTrialPass::getMarkerRMS()
{
  return mMarkerRMS;
}

void SubjectOnDiskTrialPass::setMarkerMax(std::vector<s_t> markerMax)
{
  mMarkerMax = markerMax;
}

std::vector<s_t> SubjectOnDiskTrialPass::getMarkerMax()
{
  return mMarkerMax;
}

// If we're doing a lowpass filter on this pass, then what was the cutoff
// frequency of that filter?
void SubjectOnDiskTrialPass::setLowpassCutoffFrequency(s_t freq)
{
  mLowpassCutoffFrequency = freq;
}

// If we're doing a lowpass filter on this pass, then what was the order of
// that (Butterworth) filter?
void SubjectOnDiskTrialPass::setLowpassFilterOrder(int order)
{
  mLowpassFilterOrder = order;
}

// If we filtered the force plates, then what was the cutoff frequency of that
// filtering?
void SubjectOnDiskTrialPass::setForcePlateCutoffs(std::vector<s_t> cutoffs)
{
  mForcePlateCutoffs = cutoffs;
}

void SubjectOnDiskTrialPass::setLinearResidual(std::vector<s_t> linearResidual)
{
  mLinearResidual = linearResidual;
}

std::vector<s_t> SubjectOnDiskTrialPass::getLinearResidual()
{
  return mLinearResidual;
}

void SubjectOnDiskTrialPass::setAngularResidual(
    std::vector<s_t> angularResidual)
{
  mAngularResidual = angularResidual;
}

std::vector<s_t> SubjectOnDiskTrialPass::getAngularResidual()
{
  return mAngularResidual;
}

void SubjectOnDiskTrialPass::read(
    const proto::SubjectOnDiskTrialProcessingPassHeader& proto)
{
  // // This data is included in the header
  // ProcessingPassType mType;
  mType = passTypeFromProto(proto.type());

  // std::vector<bool> mDofPositionsObserved;
  mDofPositionsObserved.clear();
  for (int i = 0; i < proto.dof_positions_observed_size(); i++)
  {
    mDofPositionsObserved.push_back(proto.dof_positions_observed(i));
  }

  // std::vector<bool> mDofVelocitiesFiniteDifferenced;
  mDofVelocitiesFiniteDifferenced.clear();
  for (int i = 0; i < proto.dof_velocities_finite_differenced_size(); i++)
  {
    mDofVelocitiesFiniteDifferenced.push_back(
        proto.dof_velocities_finite_differenced(i));
  }

  // std::vector<bool> mDofAccelerationFiniteDifferenced;
  mDofAccelerationFiniteDifferenced.clear();
  for (int i = 0; i < proto.dof_acceleration_finite_differenced_size(); i++)
  {
    mDofAccelerationFiniteDifferenced.push_back(
        proto.dof_acceleration_finite_differenced(i));
  }

  // std::vector<s_t> mMarkerRMS;
  mMarkerRMS.clear();
  for (int i = 0; i < proto.marker_rms_size(); i++)
  {
    mMarkerRMS.push_back(proto.marker_rms(i));
  }

  // std::vector<s_t> mMarkerMax;
  mMarkerMax.clear();
  for (int i = 0; i < proto.marker_max_size(); i++)
  {
    mMarkerMax.push_back(proto.marker_max(i));
  }

  // std::vector<s_t> mLinearResidual;
  mLinearResidual.clear();
  for (int i = 0; i < proto.linear_residual_size(); i++)
  {
    mLinearResidual.push_back(proto.linear_residual(i));
  }

  // std::vector<s_t> mAngularResidual;
  mAngularResidual.clear();
  for (int i = 0; i < proto.angular_residual_size(); i++)
  {
    mAngularResidual.push_back(proto.angular_residual(i));
  }

  // // This is for allowing the user to pre-filter out data where joint
  // velocities
  // // are above a certain "unreasonable limit", like 50 rad/s or so
  // std::vector<s_t> mJointsMaxVelocity;
  mJointsMaxVelocity.clear();
  for (int i = 0; i < proto.joints_max_velocity_size(); i++)
  {
    mJointsMaxVelocity.push_back(proto.joints_max_velocity(i));
  }

  mLowpassCutoffFrequency = proto.lowpass_cutoff_frequency();
  mLowpassFilterOrder = proto.lowpass_filter_order();
  mForcePlateCutoffs.clear();
  for (int i = 0; i < proto.force_plate_cutoff_size(); i++)
  {
    mForcePlateCutoffs.push_back(proto.force_plate_cutoff(i));
  }
}

void SubjectOnDiskTrialPass::write(
    proto::SubjectOnDiskTrialProcessingPassHeader* proto)
{
  // // This data is included in the header
  // std::string mName;
  proto->set_type(passTypeToProto(mType));
  // std::vector<bool> mDofPositionsObserved;
  for (int i = 0; i < mDofPositionsObserved.size(); i++)
  {
    proto->add_dof_positions_observed(mDofPositionsObserved[i]);
  }
  // std::vector<bool> mDofVelocitiesFiniteDifferenced;
  for (int i = 0; i < mDofVelocitiesFiniteDifferenced.size(); i++)
  {
    proto->add_dof_velocities_finite_differenced(
        mDofVelocitiesFiniteDifferenced[i]);
  }
  // std::vector<bool> mDofAccelerationFiniteDifferenced;
  for (int i = 0; i < mDofAccelerationFiniteDifferenced.size(); i++)
  {
    proto->add_dof_acceleration_finite_differenced(
        mDofAccelerationFiniteDifferenced[i]);
  }
  // std::vector<s_t> mMarkerRMS;
  for (int i = 0; i < mMarkerRMS.size(); i++)
  {
    proto->add_marker_rms(mMarkerRMS[i]);
  }
  // std::vector<s_t> mMarkerMax;
  for (int i = 0; i < mMarkerMax.size(); i++)
  {
    proto->add_marker_max(mMarkerMax[i]);
  }
  // std::vector<s_t> mLinearResidual;
  for (int i = 0; i < mLinearResidual.size(); i++)
  {
    proto->add_linear_residual(mLinearResidual[i]);
  }
  // std::vector<s_t> mAngularResidual;
  for (int i = 0; i < mAngularResidual.size(); i++)
  {
    proto->add_angular_residual(mAngularResidual[i]);
  }
  for (int i = 0; i < mVel.cols(); i++)
  {
    proto->add_joints_max_velocity(mVel.col(i).cwiseAbs().maxCoeff());
  }

  proto->set_lowpass_cutoff_frequency(mLowpassCutoffFrequency);
  proto->set_lowpass_filter_order(mLowpassFilterOrder);
  for (int i = 0; i < mForcePlateCutoffs.size(); i++)
  {
    proto->add_force_plate_cutoff(mForcePlateCutoffs[i]);
  }

  if (!proto->IsInitialized())
  {
    std::cerr << "WARNING: All required fields are not set on "
                 "SubjectOnDiskTrialProcessingPassHeader proto:\n"
              << proto->InitializationErrorString() << std::endl;
  }
}

void SubjectOnDiskTrialPass::copyValuesFrom(
    std::shared_ptr<SubjectOnDiskTrialPass> other)
{
  mType = other->mType;
  mDofPositionsObserved = other->mDofPositionsObserved;
  mDofVelocitiesFiniteDifferenced = other->mDofVelocitiesFiniteDifferenced;
  mDofAccelerationFiniteDifferenced = other->mDofAccelerationFiniteDifferenced;
  mMarkerRMS = other->mMarkerRMS;
  mMarkerMax = other->mMarkerMax;
  mLowpassCutoffFrequency = other->mLowpassCutoffFrequency;
  mLowpassFilterOrder = other->mLowpassFilterOrder;
  mForcePlateCutoffs = other->mForcePlateCutoffs;
  mLinearResidual = other->mLinearResidual;
  mAngularResidual = other->mAngularResidual;
  mPos = other->mPos;
  mVel = other->mVel;
  mAcc = other->mAcc;
  mTaus = other->mTaus;
  mGroundBodyWrenches = other->mGroundBodyWrenches;
  mGroundBodyCopTorqueForce = other->mGroundBodyCopTorqueForce;
  mComPoses = other->mComPoses;
  mComVels = other->mComVels;
  mComAccs = other->mComAccs;
  mComAccsInRootFrame = other->mComAccsInRootFrame;
  mResidualWrenchInRootFrame = other->mResidualWrenchInRootFrame;
  mGroundBodyWrenchesInRootFrame = other->mGroundBodyWrenchesInRootFrame;
  mGroundBodyCopTorqueForceInRootFrame
      = other->mGroundBodyCopTorqueForceInRootFrame;
  mJointCenters = other->mJointCenters;
  mJointCentersInRootFrame = other->mJointCentersInRootFrame;
  mRootSpatialVelInRootFrame = other->mRootSpatialVelInRootFrame;
  mRootSpatialAccInRootFrame = other->mRootSpatialAccInRootFrame;
  mRootPosHistoryInRootFrame = other->mRootPosHistoryInRootFrame;
  mRootEulerHistoryInRootFrame = other->mRootEulerHistoryInRootFrame;
  mJointsMaxVelocity = other->mJointsMaxVelocity;
}

SubjectOnDiskTrial::SubjectOnDiskTrial()
  : mName(""),
    mTimestep(0.01),
    mLength(0),
    mMarkerNamesGuessed(false),
    mOriginalTrialName(""),
    mSplitIndex(0),
    mNumForcePlates(0)
{
}

void SubjectOnDiskTrial::setName(const std::string& name)
{
  mName = name;
}

void SubjectOnDiskTrial::setTimestep(s_t timestep)
{
  mTimestep = timestep;
}

s_t SubjectOnDiskTrial::getTimestep()
{
  return mTimestep;
}

void SubjectOnDiskTrial::setTrialTags(std::vector<std::string> trialTags)
{
  mTrialTags = trialTags;
}

void SubjectOnDiskTrial::setOriginalTrialName(const std::string& name)
{
  mOriginalTrialName = name;
}

void SubjectOnDiskTrial::setSplitIndex(int split)
{
  mSplitIndex = split;
}

std::vector<MissingGRFReason> SubjectOnDiskTrial::getMissingGRFReason()
{
  return mMissingGRFReason;
}

void SubjectOnDiskTrial::setMissingGRFReason(
    std::vector<MissingGRFReason> missingGRFReason)
{
  mMissingGRFReason = missingGRFReason;
}

void SubjectOnDiskTrial::setCustomValues(
    std::vector<Eigen::MatrixXs> customValues)
{
  mCustomValues = customValues;
}

void SubjectOnDiskTrial::setMarkerNamesGuessed(bool markersGuessed)
{
  mMarkerNamesGuessed = markersGuessed;
}

void SubjectOnDiskTrial::setMarkerObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations)
{
  mMarkerObservations = markerObservations;
}

void SubjectOnDiskTrial::setAccObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> accObservations)
{
  mAccObservations = accObservations;
}

void SubjectOnDiskTrial::setGyroObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> gyroObservations)
{
  mGyroObservations = gyroObservations;
}

void SubjectOnDiskTrial::setEmgObservations(
    std::vector<std::map<std::string, Eigen::VectorXs>> emgObservations)
{
  mEmgObservations = emgObservations;
}

void SubjectOnDiskTrial::setExoTorques(
    std::map<int, Eigen::VectorXs> exoTorques)
{
  mExoTorques = exoTorques;
}

void SubjectOnDiskTrial::setForcePlates(std::vector<ForcePlate> forcePlates)
{
  mForcePlates = forcePlates;
}

std::vector<ForcePlate> SubjectOnDiskTrial::getForcePlates()
{
  return mForcePlates;
}

std::shared_ptr<SubjectOnDiskTrialPass> SubjectOnDiskTrial::addPass()
{
  mTrialPasses.push_back(std::make_shared<SubjectOnDiskTrialPass>());
  return mTrialPasses.back();
}

std::vector<std::shared_ptr<SubjectOnDiskTrialPass>>
SubjectOnDiskTrial::getPasses()
{
  return mTrialPasses;
}

void SubjectOnDiskTrial::read(const proto::SubjectOnDiskTrialHeader& proto)
{
  // std::string mName;
  mName = proto.name();

  // s_t mTimestep;
  mTimestep = proto.trial_timestep();

  // int mLength;
  mLength = proto.trial_length();

  // std::vector<std::string> mTrialTags;
  mTrialTags.clear();
  for (int i = 0; i < proto.trial_tag_size(); i++)
  {
    mTrialTags.push_back(proto.trial_tag(i));
  }

  mOriginalTrialName = proto.original_name();

  mSplitIndex = proto.split_index();

  mTrialPasses.clear();
  for (int i = 0; i < proto.processing_pass_header_size(); i++)
  {
    std::shared_ptr<SubjectOnDiskTrialPass> pass
        = std::make_shared<SubjectOnDiskTrialPass>();
    pass->read(proto.processing_pass_header(i));
    mTrialPasses.push_back(pass);
  }

  // std::vector<MissingGRFReason> mMissingGRFReason;
  mMissingGRFReason.clear();
  for (int i = 0; i < proto.missing_grf_reason_size(); i++)
  {
    mMissingGRFReason.push_back(
        missingGRFReasonFromProto(proto.missing_grf_reason(i)));
  }

  // ///////////////////////////////////////////////////////////////////////////
  // // Raw sensor observations, which are shared across processing passes
  // ///////////////////////////////////////////////////////////////////////////

  mMarkerNamesGuessed = proto.marker_names_guessed();
  mNumForcePlates = proto.num_force_plates();
  mForcePlateCorners.clear();
  if (proto.force_plate_corners_size() == 0)
  {
    // This is fine, ignore the corners, we don't always have that info
  }
  else if (proto.force_plate_corners_size() != mNumForcePlates * 12)
  {
    std::cout << "WARNING: force_plate_corners_size() is not "
                 "num_force_plates * 12, it is "
              << proto.force_plate_corners_size()
              << ". As a result, we will not read any force plate corners, "
                 "because it is untrustworthy data."
              << std::endl;
  }
  else
  {
    for (int i = 0; i < mNumForcePlates; i++)
    {
      std::vector<Eigen::Vector3s> corners;
      for (int c = 0; c < 4; c++)
      {
        Eigen::Vector3s corner;
        corner(0) = proto.force_plate_corners(i * 12 + c * 3 + 0);
        corner(1) = proto.force_plate_corners(i * 12 + c * 3 + 1);
        corner(2) = proto.force_plate_corners(i * 12 + c * 3 + 2);
        corners.push_back(corner);
      }
      mForcePlateCorners.push_back(corners);
    }
  }
}

void SubjectOnDiskTrial::write(proto::SubjectOnDiskTrialHeader* proto)
{
  // std::string mName;
  proto->set_name(mName);
  // s_t mTimestep;
  proto->set_trial_timestep(mTimestep);
  // Set the length of the trial to whatever the length of our marker
  // observations is
  proto->set_trial_length(mMarkerObservations.size());

  proto->set_original_name(mOriginalTrialName);
  proto->set_split_index(mSplitIndex);

  // std::vector<std::string> mTrialTags;
  for (int i = 0; i < mTrialTags.size(); i++)
  {
    proto->add_trial_tag(mTrialTags[i]);
  }
  for (int i = 0; i < mTrialPasses.size(); i++)
  {
    proto::SubjectOnDiskTrialProcessingPassHeader* passProto
        = proto->add_processing_pass_header();
    mTrialPasses[i]->write(passProto);
  }
  // std::vector<MissingGRFReason> mMissingGRFReason;
  for (int i = 0; i < mMissingGRFReason.size(); i++)
  {
    proto->add_missing_grf_reason(
        missingGRFReasonToProto(mMissingGRFReason[i]));
  }

  // ///////////////////////////////////////////////////////////////////////////
  // // Raw sensor observations, which are shared across processing passes
  // ///////////////////////////////////////////////////////////////////////////

  // std::vector<Eigen::MatrixXs> mCustomValues;
  // // This is true if we guessed the marker names, and false if we got them
  // from
  // // the uploaded user's file, which implies that they got them from human
  // // observations.
  // bool mMarkerNamesGuessed;
  proto->set_marker_names_guessed(mMarkerNamesGuessed);

  // // This is raw force plate data
  // std::vector<std::vector<ForcePlate>> mForcePlates;
  proto->set_num_force_plates(mForcePlates.size());
  for (auto& forcePlate : mForcePlates)
  {
    if (forcePlate.corners.size() != 4)
    {
      for (int i = 0; i < 12; i++)
      {
        proto->add_force_plate_corners(0);
      }
    }
    for (Eigen::Vector3s& corners : forcePlate.corners)
    {
      for (int i = 0; i < 3; i++)
      {
        proto->add_force_plate_corners(corners(i));
      }
    }
  }

  if (!proto->IsInitialized())
  {
    std::cerr << "WARNING: All required fields are not set on "
                 "SubjectOnDiskTrialHeader proto:\n"
              << proto->InitializationErrorString() << std::endl;
  }
}

SubjectOnDiskPassHeader::SubjectOnDiskPassHeader()
{
  // Do nothing
}

void SubjectOnDiskPassHeader::setProcessingPassType(ProcessingPassType type)
{
  mType = type;
}

ProcessingPassType SubjectOnDiskPassHeader::getProcessingPassType()
{
  return mType;
}

void SubjectOnDiskPassHeader::setOpenSimFileText(
    const std::string& openSimFileText)
{
  mOpenSimFileText = openSimFileText;
}

std::string SubjectOnDiskPassHeader::getOpenSimFileText()
{
  return mOpenSimFileText;
}

void SubjectOnDiskPassHeader::write(dart::proto::SubjectOnDiskPass* proto)
{
  proto->set_pass_type(passTypeToProto(mType));
  proto->set_model_osim_text(mOpenSimFileText);

  if (!proto->IsInitialized())
  {
    std::cerr << "WARNING: All required fields are not set on "
                 "SubjectOnDiskPass proto:\n"
              << proto->InitializationErrorString() << std::endl;
  }
}

void SubjectOnDiskPassHeader::read(const dart::proto::SubjectOnDiskPass& proto)
{
  mType = passTypeFromProto(proto.pass_type());
  mOpenSimFileText = proto.model_osim_text();
}

SubjectOnDiskHeader::SubjectOnDiskHeader()
  : mNumDofs(0),
    mBiologicalSex("unknown"),
    mHeightM(0),
    mMassKg(0),
    mAgeYears(-1),
    mHref(""),
    mNotes("")
{
  // Do nothing
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setNumDofs(int dofs)
{
  mNumDofs = dofs;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setNumJoints(int joints)
{
  mNumJoints = joints;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setGroundForceBodies(
    std::vector<std::string> groundForceBodies)
{
  mGroundContactBodies = groundForceBodies;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setCustomValueNames(
    std::vector<std::string> customValueNames)
{
  mCustomValueNames = customValueNames;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setBiologicalSex(
    const std::string& biologicalSex)
{
  mBiologicalSex = biologicalSex;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setHeightM(double heightM)
{
  mHeightM = heightM;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setMassKg(double massKg)
{
  mMassKg = massKg;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setAgeYears(int ageYears)
{
  mAgeYears = ageYears;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setSubjectTags(
    std::vector<std::string> subjectTags)
{
  mSubjectTags = subjectTags;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setHref(const std::string& href)
{
  mHref = href;
  return *this;
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setNotes(const std::string& notes)
{
  mNotes = notes;
  return *this;
}

std::shared_ptr<SubjectOnDiskPassHeader>
SubjectOnDiskHeader::addProcessingPass()
{
  mPasses.push_back(std::make_shared<SubjectOnDiskPassHeader>());
  return mPasses.back();
}

std::vector<std::shared_ptr<SubjectOnDiskPassHeader>>
SubjectOnDiskHeader::getProcessingPasses()
{
  return mPasses;
}

std::shared_ptr<SubjectOnDiskTrial> SubjectOnDiskHeader::addTrial()
{
  mTrials.push_back(std::make_shared<SubjectOnDiskTrial>());
  return mTrials.back();
}

std::vector<std::shared_ptr<SubjectOnDiskTrial>>
SubjectOnDiskHeader::getTrials()
{
  return mTrials;
}

void SubjectOnDiskHeader::setTrials(
    std::vector<std::shared_ptr<SubjectOnDiskTrial>> trials)
{
  mTrials = trials;
}

void SubjectOnDiskHeader::recomputeColumnNames()
{
  mMarkerNames.clear();
  mAccNames.clear();
  mGyroNames.clear();
  mEmgNames.clear();
  mExoDofIndices.clear();
  for (int trial = 0; trial < mTrials.size(); trial++)
  {
    for (int t = 0; t < mTrials[trial]->mMarkerObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial]->mMarkerObservations[t])
      {
        if (std::find(mMarkerNames.begin(), mMarkerNames.end(), pair.first)
            == mMarkerNames.end())
        {
          mMarkerNames.push_back(pair.first);
        }
      }
    }
  }
  for (int trial = 0; trial < mTrials.size(); trial++)
  {
    for (int t = 0; t < mTrials[trial]->mAccObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial]->mAccObservations[t])
      {
        if (std::find(mAccNames.begin(), mAccNames.end(), pair.first)
            == mAccNames.end())
        {
          mAccNames.push_back(pair.first);
        }
      }
    }
  }
  for (int trial = 0; trial < mTrials.size(); trial++)
  {
    for (int t = 0; t < mTrials[trial]->mGyroObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial]->mGyroObservations[t])
      {
        if (std::find(mGyroNames.begin(), mGyroNames.end(), pair.first)
            == mGyroNames.end())
        {
          mGyroNames.push_back(pair.first);
        }
      }
    }
  }
  for (int trial = 0; trial < mTrials.size(); trial++)
  {
    for (auto& pair : mTrials[trial]->mExoTorques)
    {
      if (std::find(mExoDofIndices.begin(), mExoDofIndices.end(), pair.first)
          == mExoDofIndices.end())
      {
        mExoDofIndices.push_back(pair.first);
      }
    }
  }
  mEmgDim = 0;
  for (int trial = 0; trial < mTrials.size(); trial++)
  {
    for (int t = 0; t < mTrials[trial]->mEmgObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial]->mEmgObservations[t])
      {
        if (std::find(mEmgNames.begin(), mEmgNames.end(), pair.first)
            == mEmgNames.end())
        {
          mEmgNames.push_back(pair.first);
        }
        if (mEmgDim == 0)
        {
          mEmgDim = pair.second.size();
        }
        else if (mEmgDim != pair.second.size())
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "emgObservations have inconsistent dimensions for trial "
                    << mTrials[trial]->mName << " timestep " << t << " for emg "
                    << pair.first << ". Expected " << mEmgDim << " but got "
                    << pair.second.size() << std::endl;
        }
      }
    }
  }
}

void SubjectOnDiskHeader::write(dart::proto::SubjectOnDiskHeader* header)
{
  // // How many DOFs are in the skeleton
  // int mNumDofs;
  header->set_num_dofs(mNumDofs);
  // int mNumJoints;
  header->set_num_joints(mNumJoints);
  // // The passes we applied to this data, along with the result skeletons that
  // // were generated by each pass.
  // std::vector<SubjectOnDiskPassHeader> mPasses;
  for (std::shared_ptr<SubjectOnDiskPassHeader> pass : mPasses)
  {
    proto::SubjectOnDiskPass* passProto = header->add_passes();
    pass->write(passProto);
  }
  // // These are generalized 6-dof wrenches applied to arbitrary bodies
  // // (generally by foot-ground contact, though other things too)
  // std::vector<std::string> mGroundContactBodies;
  for (std::string& body : mGroundContactBodies)
  {
    header->add_ground_contact_body(body);
  }
  // // We include this to allow the binary format to store/load a bunch of new
  // // types of values while remaining backwards compatible.
  // std::vector<std::string> mCustomValueNames;
  for (int i = 0; i < mCustomValueNames.size(); i++)
  {
    header->add_custom_value_name(mCustomValueNames[i]);
    header->add_custom_value_length(
        mTrials.size() > 0 && mTrials[0]->mCustomValues.size() > 0
            ? mTrials[0]->mCustomValues[0].rows()
            : 0);
  }
  // // This is the subject info
  // std::string mBiologicalSex;
  header->set_biological_sex(mBiologicalSex);
  // double mHeightM;
  header->set_height_m(mHeightM);
  // double mMassKg;
  header->set_mass_kg(mMassKg);
  // int mAgeYears;
  header->set_age_years(mAgeYears);
  // // The provenance info, optional, for investigating where training data
  // // came from after its been aggregated
  // std::vector<std::string> mSubjectTags;
  for (std::string& tag : mSubjectTags)
  {
    header->add_subject_tag(tag);
  }
  // std::string mHref = "";
  header->set_href(mHref);
  // std::string notes = "";
  header->set_notes(mNotes);
  header->set_version(4);

  // // These are the trials, which contain the actual data
  // std::vector<SubjectOnDiskTrialBuilder> mTrials;
  for (auto& trial : mTrials)
  {
    proto::SubjectOnDiskTrialHeader* trialHeader = header->add_trial_header();
    trial->write(trialHeader);
  }

  recomputeColumnNames();
  for (std::string& name : mMarkerNames)
  {
    header->add_marker_name(name);
  }
  for (std::string& name : mAccNames)
  {
    header->add_acc_name(name);
  }
  for (std::string& name : mGyroNames)
  {
    header->add_gyro_name(name);
  }
  for (std::string& name : mEmgNames)
  {
    header->add_emg_name(name);
  }
  header->set_emg_dim(mEmgDim);
  for (int index : mExoDofIndices)
  {
    header->add_exo_dof_index(index);
  }

  if (!header->IsInitialized())
  {
    std::cerr << "WARNING: All required fields are not set on "
                 "SubjectOnDiskHeader proto:\n"
              << header->InitializationErrorString() << std::endl;
  }
}

void SubjectOnDiskHeader::read(const dart::proto::SubjectOnDiskHeader& proto)
{
  if (proto.version() > 4)
  {
    throw std::runtime_error(
        "SubjectOnDiskHeader::read() can't read file version "
        + std::to_string(proto.version())
        + ". Please upgrade your nimblephysics version to the latest version.");
  }

  // // How many DOFs are in the skeleton
  // int mNumDofs;
  mNumDofs = proto.num_dofs();

  // How many joints are in the skeleton
  mNumJoints = proto.num_joints();

  // // The passes we applied to this data, along with the result skeletons that
  // // were generated by each pass.
  // std::vector<SubjectOnDiskPassHeader> mPasses;
  mPasses.clear();
  for (int i = 0; i < proto.passes_size(); i++)
  {
    std::shared_ptr<SubjectOnDiskPassHeader> pass
        = std::make_shared<SubjectOnDiskPassHeader>();
    pass->read(proto.passes(i));
    mPasses.push_back(pass);
  }

  // // These are generalized 6-dof wrenches applied to arbitrary bodies
  // // (generally by foot-ground contact, though other things too)
  // std::vector<std::string> mGroundContactBodies;
  mGroundContactBodies.clear();
  for (int i = 0; i < proto.ground_contact_body_size(); i++)
  {
    mGroundContactBodies.push_back(proto.ground_contact_body(i));
  }

  // // We include this to allow the binary format to store/load a bunch of new
  // // types of values while remaining backwards compatible.
  // std::vector<std::string> mCustomValueNames;
  mCustomValueNames.clear();
  for (int i = 0; i < proto.custom_value_name_size(); i++)
  {
    mCustomValueNames.push_back(proto.custom_value_name(i));
  }

  // std::vector<int> mCustomValueLengths;
  mCustomValueLengths.clear();
  for (int i = 0; i < proto.custom_value_length_size(); i++)
  {
    mCustomValueLengths.push_back(proto.custom_value_length(i));
  }

  assert(mCustomValueNames.size() == mCustomValueLengths.size());

  // // This is the subject info
  // std::string mBiologicalSex;
  mBiologicalSex = proto.biological_sex();

  // double mHeightM;
  mHeightM = proto.height_m();

  // double mMassKg;
  mMassKg = proto.mass_kg();

  // int mAgeYears;
  mAgeYears = proto.age_years();

  // // The provenance info, optional, for investigating where training data
  // // came from after its been aggregated
  // std::vector<std::string> mSubjectTags;
  mSubjectTags.clear();
  for (int i = 0; i < proto.subject_tag_size(); i++)
  {
    mSubjectTags.push_back(proto.subject_tag(i));
  }

  // std::string mHref = "";
  mHref = proto.href();

  // std::string mNotes = "";
  mNotes = proto.notes();

  // // These are the trials, which contain the actual data
  // std::vector<SubjectOnDiskTrial> mTrials;
  mTrials.clear();
  for (int i = 0; i < proto.trial_header_size(); i++)
  {
    std::shared_ptr<SubjectOnDiskTrial> trial
        = std::make_shared<SubjectOnDiskTrial>();
    trial->read(proto.trial_header(i));
    mTrials.push_back(trial);
  }

  // // These are the marker, accelerometer and gyroscope names
  // std::vector<std::string> mMarkerNames;
  mMarkerNames.clear();
  for (int i = 0; i < proto.marker_name_size(); i++)
  {
    mMarkerNames.push_back(proto.marker_name(i));
  }

  // std::vector<std::string> mAccNames;
  mAccNames.clear();
  for (int i = 0; i < proto.acc_name_size(); i++)
  {
    mAccNames.push_back(proto.acc_name(i));
  }

  // std::vector<std::string> mGyroNames;
  mGyroNames.clear();
  for (int i = 0; i < proto.gyro_name_size(); i++)
  {
    mGyroNames.push_back(proto.gyro_name(i));
  }

  // // This is EMG data
  // std::vector<std::string> mEmgNames;
  mEmgNames.clear();
  for (int i = 0; i < proto.emg_name_size(); i++)
  {
    mEmgNames.push_back(proto.emg_name(i));
  }

  // int mEmgDim;
  mEmgDim = proto.emg_dim();

  // // This is exoskeleton data
  // std::vector<std::string> mExoDofNames;
  mExoDofIndices.clear();
  for (int i = 0; i < proto.exo_dof_index_size(); i++)
  {
    mExoDofIndices.push_back(proto.exo_dof_index(i));
  }
}

void SubjectOnDiskHeader::writeSensorsFrame(
    dart::proto::SubjectOnDiskSensorFrame* proto,
    int trial,
    int t,
    int maxNumForcePlates)
{
  // // We include this to allow the binary format to store/load a bunch of new
  // // types of values while remaining backwards compatible.
  // repeated double custom_values = 1;
  if (mTrials.size() > trial)
  {
    for (int i = 0; i < mTrials[trial]->mCustomValues.size(); i++)
    {
      for (int j = 0; j < mTrials[trial]->mCustomValues[i].rows(); j++)
      {
        proto->add_custom_values(mTrials[trial]->mCustomValues[i](j, t));
      }
    }
  }

  // // These are marker observations on this frame, with all NaNs indicating
  // that that marker was not observed on this frame repeated double marker_obs
  // = 2;
  if (mTrials.size() > trial && mTrials[trial]->mMarkerObservations.size() > t)
  {
    for (std::string& name : mMarkerNames)
    {
      if (mTrials[trial]->mMarkerObservations[t].find(name)
          != mTrials[trial]->mMarkerObservations[t].end())
      {
        proto->add_marker_obs(
            mTrials[trial]->mMarkerObservations[t].at(name)(0));
        proto->add_marker_obs(
            mTrials[trial]->mMarkerObservations[t].at(name)(1));
        proto->add_marker_obs(
            mTrials[trial]->mMarkerObservations[t].at(name)(2));
      }
      else
      {
        proto->add_marker_obs(nan(""));
        proto->add_marker_obs(nan(""));
        proto->add_marker_obs(nan(""));
      }
    }
  }
  else
  {
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      proto->add_marker_obs(nan(""));
      proto->add_marker_obs(nan(""));
      proto->add_marker_obs(nan(""));
    }
  }
  // // These are IMU observations on this frame, with all NaNs indicating that
  // that imu was not observed on this frame repeated double acc_obs = 3;
  if (mTrials.size() > trial && mTrials[trial]->mAccObservations.size() > t)
  {
    for (std::string& name : mAccNames)
    {
      if (mTrials[trial]->mAccObservations[t].find(name)
          != mTrials[trial]->mAccObservations[t].end())
      {
        proto->add_acc_obs(mTrials[trial]->mAccObservations[t].at(name)(0));
        proto->add_acc_obs(mTrials[trial]->mAccObservations[t].at(name)(1));
        proto->add_acc_obs(mTrials[trial]->mAccObservations[t].at(name)(2));
      }
      else
      {
        proto->add_acc_obs(nan(""));
        proto->add_acc_obs(nan(""));
        proto->add_acc_obs(nan(""));
      }
    }
  }
  else
  {
    for (int i = 0; i < mAccNames.size(); i++)
    {
      proto->add_acc_obs(nan(""));
      proto->add_acc_obs(nan(""));
      proto->add_acc_obs(nan(""));
    }
  }
  // repeated double gyro_obs = 4;
  // // These are the EMG observations on this frame
  if (mTrials.size() > trial && mTrials[trial]->mGyroObservations.size() > t)
  {
    for (std::string& name : mGyroNames)
    {
      if (mTrials[trial]->mGyroObservations[t].find(name)
          != mTrials[trial]->mGyroObservations[t].end())
      {
        proto->add_gyro_obs(mTrials[trial]->mGyroObservations[t].at(name)(0));
        proto->add_gyro_obs(mTrials[trial]->mGyroObservations[t].at(name)(1));
        proto->add_gyro_obs(mTrials[trial]->mGyroObservations[t].at(name)(2));
      }
      else
      {
        proto->add_gyro_obs(nan(""));
        proto->add_gyro_obs(nan(""));
        proto->add_gyro_obs(nan(""));
      }
    }
  }
  else
  {
    for (int i = 0; i < mGyroNames.size(); i++)
    {
      proto->add_gyro_obs(nan(""));
      proto->add_gyro_obs(nan(""));
      proto->add_gyro_obs(nan(""));
    }
  }
  // repeated double emg_obs = 5;
  if (mTrials.size() > trial && mTrials[trial]->mEmgObservations.size() > t)
  {
    for (std::string& name : mEmgNames)
    {
      if (mTrials[trial]->mEmgObservations[t].find(name)
          != mTrials[trial]->mEmgObservations[t].end())
      {
        for (int i = 0; i < mEmgDim; i++)
        {
          proto->add_emg_obs(mTrials[trial]->mEmgObservations[t].at(name)(i));
        }
      }
      else
      {
        for (int i = 0; i < mEmgDim; i++)
        {
          proto->add_emg_obs(nan(""));
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < mEmgNames.size(); i++)
    {
      for (int j = 0; j < mEmgDim; j++)
      {
        proto->add_emg_obs(nan(""));
      }
    }
  }

  // // These are the exo observations on this frame
  // repeated double exo_obs = 6;
  if (mTrials.size() > trial)
  {
    for (int dof : mExoDofIndices)
    {
      if (mTrials[trial]->mExoTorques.find(dof)
              != mTrials[trial]->mExoTorques.end()
          && mTrials[trial]->mExoTorques.at(dof).size() > t)
      {
        proto->add_exo_obs(mTrials[trial]->mExoTorques.at(dof)(t));
      }
      else
      {
        proto->add_exo_obs(nan(""));
      }
    }
  }

  // // These are the raw force plate readings, per force plate, without any
  // assignment to feet or any post-processing repeated double
  // raw_force_plate_cop = 7; repeated double raw_force_plate_torque = 8;
  // repeated double raw_force_plate_force = 9;

  for (int forcePlateIdx = 0; forcePlateIdx < maxNumForcePlates;
       forcePlateIdx++)
  {
    if (trial < mTrials.size()
        && forcePlateIdx < mTrials[trial]->mForcePlates.size()
        && mTrials[trial]->mForcePlates[forcePlateIdx].centersOfPressure.size()
               > t
        && mTrials[trial]->mForcePlates[forcePlateIdx].forces.size() > t
        && mTrials[trial]->mForcePlates[forcePlateIdx].moments.size() > t)
    {
      for (int i = 0; i < 3; i++)
      {
        proto->add_raw_force_plate_cop(
            mTrials[trial]->mForcePlates[forcePlateIdx].centersOfPressure[t](
                i));
        proto->add_raw_force_plate_torque(
            mTrials[trial]->mForcePlates[forcePlateIdx].moments[t](i));
        proto->add_raw_force_plate_force(
            mTrials[trial]->mForcePlates[forcePlateIdx].forces[t](i));
      }
    }
    else
    {
      for (int i = 0; i < 3; i++)
      {
        proto->add_raw_force_plate_cop(std::nan(""));
        proto->add_raw_force_plate_torque(std::nan(""));
        proto->add_raw_force_plate_force(std::nan(""));
      }
    }
  }
}

void SubjectOnDiskHeader::writeProcessingPassFrame(
    dart::proto::SubjectOnDiskProcessingPassFrame* proto,
    int trial,
    int t,
    int pass)
{
  // // The values for all the DOFs
  // repeated double pos = 1;
  // repeated double vel = 2;
  // repeated double acc = 3;
  // repeated double tau = 4;
  if (mTrials.size() > trial && mTrials[trial]->mTrialPasses.size() > pass
      && mTrials[trial]->mTrialPasses[pass]->mPos.size() > t
      && mTrials[trial]->mTrialPasses[pass]->mVel.size() > t
      && mTrials[trial]->mTrialPasses[pass]->mAcc.size() > t
      && mTrials[trial]->mTrialPasses[pass]->mTaus.size() > t)
  {
    for (int i = 0; i < mNumDofs; i++)
    {
      proto->add_pos(mTrials[trial]->mTrialPasses[pass]->mPos(i, t));
      proto->add_vel(mTrials[trial]->mTrialPasses[pass]->mVel(i, t));
      proto->add_acc(mTrials[trial]->mTrialPasses[pass]->mAcc(i, t));
      proto->add_tau(mTrials[trial]->mTrialPasses[pass]->mTaus(i, t));
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: trialPoses, "
                 "trialVels, or trialAccs out-of-bounds for trial "
              << trial << " frame " << t << std::endl;
  }

  // // This is an array of 6-vectors, one per ground contact body
  // repeated double ground_contact_wrench = 5;
  // // These are the original force-plate data in world space, one per ground
  // contact body repeated double ground_contact_center_of_pressure = 6;
  // repeated double ground_contact_torque = 7;
  // repeated double ground_contact_force = 8;

  if (mTrials.size() > trial && mTrials[trial]->mTrialPasses.size() > pass
      && mTrials[trial]->mTrialPasses[pass]->mGroundBodyWrenches.cols() > t
      && mTrials[trial]->mTrialPasses[pass]->mGroundBodyWrenches.rows()
             == 6 * mGroundContactBodies.size()
      && mTrials[trial]
                 ->mTrialPasses[pass]
                 ->mGroundBodyWrenchesInRootFrame.cols()
             > t
      && mTrials[trial]
                 ->mTrialPasses[pass]
                 ->mGroundBodyWrenchesInRootFrame.rows()
             == 6 * mGroundContactBodies.size()
      && mTrials[trial]->mTrialPasses[pass]->mResidualWrenchInRootFrame.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mResidualWrenchInRootFrame.rows()
             == 6
      && mTrials[trial]->mTrialPasses[pass]->mGroundBodyCopTorqueForce.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mGroundBodyCopTorqueForce.rows()
             == 9 * mGroundContactBodies.size()
      && mTrials[trial]
                 ->mTrialPasses[pass]
                 ->mGroundBodyCopTorqueForceInRootFrame.cols()
             > t
      && mTrials[trial]
                 ->mTrialPasses[pass]
                 ->mGroundBodyCopTorqueForceInRootFrame.rows()
             == 9 * mGroundContactBodies.size())
  {
    for (int j = 0; j < 6; j++)
    {
      proto->add_root_frame_residual(
          mTrials[trial]->mTrialPasses[pass]->mResidualWrenchInRootFrame(j, t));
    }
    for (int i = 0; i < mGroundContactBodies.size(); i++)
    {
      for (int j = 0; j < 6; j++)
      {
        proto->add_ground_contact_wrench(
            mTrials[trial]->mTrialPasses[pass]->mGroundBodyWrenches(
                i * 6 + j, t));
        proto->add_root_frame_ground_contact_wrench(
            mTrials[trial]->mTrialPasses[pass]->mGroundBodyWrenchesInRootFrame(
                i * 6 + j, t));
      }
      for (int j = 0; j < 3; j++)
      {
        proto->add_ground_contact_center_of_pressure(
            mTrials[trial]->mTrialPasses[pass]->mGroundBodyCopTorqueForce(
                i * 9 + j, t));
        proto->add_ground_contact_torque(
            mTrials[trial]->mTrialPasses[pass]->mGroundBodyCopTorqueForce(
                i * 9 + 3 + j, t));
        proto->add_ground_contact_force(
            mTrials[trial]->mTrialPasses[pass]->mGroundBodyCopTorqueForce(
                i * 9 + 6 + j, t));

        proto->add_root_frame_ground_contact_center_of_pressure(
            mTrials[trial]
                ->mTrialPasses[pass]
                ->mGroundBodyCopTorqueForceInRootFrame(i * 9 + j, t));
        proto->add_root_frame_ground_contact_torques(
            mTrials[trial]
                ->mTrialPasses[pass]
                ->mGroundBodyCopTorqueForceInRootFrame(i * 9 + 3 + j, t));
        proto->add_root_frame_ground_contact_force(
            mTrials[trial]
                ->mTrialPasses[pass]
                ->mGroundBodyCopTorqueForceInRootFrame(i * 9 + 6 + j, t));
      }
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                 "trialGroundBodyWrenches or trialGroundBodyCopTorqueForce or "
                 "mResidualWrenchInRootFrame or mGroundBodyCopTorqueForce "
                 "out-of-bounds for trial "
              << trial << " frame " << t << std::endl;
  }

  // // These are the center of mass kinematics
  // repeated double com_pos = 9;
  // repeated double com_vel = 10;
  // repeated double com_acc = 11;
  // repeated double root_frame_com_acc = 17;
  for (int i = 0; i < 3; i++)
  {
    if (trial < mTrials.size() && pass < mTrials[trial]->mTrialPasses.size()
        && t < mTrials[trial]->mTrialPasses[pass]->mComPoses.cols())
    {
      proto->add_com_pos(mTrials[trial]->mTrialPasses[pass]->mComPoses(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComPoses out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_pos(std::nan(""));
    }
    if (trial < mTrials.size() && pass < mTrials[trial]->mTrialPasses.size()
        && t < mTrials[trial]->mTrialPasses[pass]->mComVels.cols())
    {
      proto->add_com_vel(mTrials[trial]->mTrialPasses[pass]->mComVels(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComVels out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_vel(std::nan(""));
    }
    if (trial < mTrials.size() && pass < mTrials[trial]->mTrialPasses.size()
        && t < mTrials[trial]->mTrialPasses[pass]->mComAccs.cols())
    {
      proto->add_com_acc(mTrials[trial]->mTrialPasses[pass]->mComAccs(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComAccs out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_acc(std::nan(""));
    }
    if (trial < mTrials.size() && pass < mTrials[trial]->mTrialPasses.size()
        && t < mTrials[trial]->mTrialPasses[pass]->mComAccsInRootFrame.cols())
    {
      proto->add_root_frame_com_acc(
          mTrials[trial]->mTrialPasses[pass]->mComAccsInRootFrame(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "mComAccsInRootFrame out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_pos(std::nan(""));
    }
  }

  // // One 3-vec per joint
  // repeated double root_frame_joint_centers = 15;
  // // One 3-vec per joint
  // repeated double world_frame_joint_centers = 16;
  if (mTrials.size() > trial && mTrials[trial]->mTrialPasses.size() > pass
      && mTrials[trial]->mTrialPasses[pass]->mJointCenters.cols() > t
      && mTrials[trial]->mTrialPasses[pass]->mJointCentersInRootFrame.size()
             > t)
  {
    for (int i = 0;
         i < mTrials[trial]->mTrialPasses[pass]->mJointCenters.rows();
         i++)
    {
      proto->add_world_frame_joint_centers(
          mTrials[trial]->mTrialPasses[pass]->mJointCenters(i, t));
    }
    for (int i = 0;
         i
         < mTrials[trial]->mTrialPasses[pass]->mJointCentersInRootFrame.rows();
         i++)
    {
      proto->add_root_frame_joint_centers(
          mTrials[trial]->mTrialPasses[pass]->mJointCentersInRootFrame(i, t));
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: trialPoses, "
                 "joint centers, or joint centers in root frame out-of-bounds "
                 "for trial "
              << trial << " frame " << t << std::endl;
  }

  if (mTrials.size() > trial && mTrials[trial]->mTrialPasses.size() > pass
      && mTrials[trial]->mTrialPasses[pass]->mRootSpatialVelInRootFrame.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mRootSpatialVelInRootFrame.rows()
             == 6
      && mTrials[trial]->mTrialPasses[pass]->mRootSpatialAccInRootFrame.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mRootSpatialAccInRootFrame.rows()
             == 6)
  {
    for (int row = 0; row < 6; row++)
    {
      proto->add_root_frame_spatial_velocity(
          mTrials[trial]->mTrialPasses[pass]->mRootSpatialVelInRootFrame(
              row, t));
    }
    for (int row = 0; row < 6; row++)
    {
      proto->add_root_frame_spatial_acceleration(
          mTrials[trial]->mTrialPasses[pass]->mRootSpatialAccInRootFrame(
              row, t));
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: root spatial "
                 "vel or root spatial acc out-of-bounds "
                 "for trial "
              << trial << " frame " << t << std::endl;
  }

  if (mTrials.size() > trial && mTrials[trial]->mTrialPasses.size() > pass
      && mTrials[trial]->mTrialPasses[pass]->mRootPosHistoryInRootFrame.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mRootPosHistoryInRootFrame.rows()
                 % 3
             == 0
      && mTrials[trial]->mTrialPasses[pass]->mRootEulerHistoryInRootFrame.cols()
             > t
      && mTrials[trial]->mTrialPasses[pass]->mRootEulerHistoryInRootFrame.rows()
                 % 3
             == 0)
  {
    for (int row = 0; row < mTrials[trial]
                                ->mTrialPasses[pass]
                                ->mRootPosHistoryInRootFrame.rows();
         row++)
    {
      proto->add_root_frame_root_pos_history(
          mTrials[trial]->mTrialPasses[pass]->mRootPosHistoryInRootFrame(
              row, t));
    }
    for (int row = 0; row < mTrials[trial]
                                ->mTrialPasses[pass]
                                ->mRootEulerHistoryInRootFrame.rows();
         row++)
    {
      proto->add_root_frame_root_euler_history(
          mTrials[trial]->mTrialPasses[pass]->mRootEulerHistoryInRootFrame(
              row, t));
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: root spatial "
                 "vel or root spatial acc out-of-bounds "
                 "for trial "
              << trial << " frame " << t << std::endl;
  }
}

} // namespace biomechanics
} // namespace dart