#include "dart/biomechanics/SubjectOnDisk.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <stdio.h>
#include <sys/_types/_int64_t.h>
#include <tinyxml2.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/enums.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
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

SubjectOnDisk::SubjectOnDisk(const std::string& path) : mPath(path)
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
              << path << ": got an error parsing the protobuf file header."
              << std::endl;
    throw new std::exception();
  }
  // 6. Get the data out of the protobuf object
  mHeader.read(header);
  mSensorFrameSize = header.raw_sensor_frame_size();
  mProcessingPassFrameSize = header.processing_pass_frame_size();
  mDataSectionStart = sizeof(int64_t) + headerSize;

  fclose(file);
}

/// This will write a B3D file to disk
void SubjectOnDisk::writeB3D(
    const std::string& outputPath, SubjectOnDiskHeader& header)
{
  // 0. Open the file
  FILE* file = fopen(outputPath.c_str(), "w");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDiskBuilder::write() failed to open output file at "
              << outputPath << ". Do you have permissions to write that file?"
              << std::endl;
    return;
  }

  // Create the header proto

  proto::SubjectOnDiskHeader headerProto;
  header.write(&headerProto);

  // 1.3. Continues in next section, after we know the size of the first
  // serialized frame...

  /////////////////////////////////////////////////////////////////////////////
  // 2. Serialize and write the frames to the file
  /////////////////////////////////////////////////////////////////////////////

  int maxNumForcePlates = 0;
  for (int trial = 0; trial < header.mTrials.size(); trial++)
  {
    if (header.mTrials[trial].mForcePlates.size() > maxNumForcePlates)
    {
      maxNumForcePlates = header.mTrials[trial].mForcePlates.size();
    }
  }

  bool firstTrial = true;
  int64_t sensorFrameSize = 0;
  int64_t passFrameSize = 0;

  for (int trial = 0; trial < header.mTrials.size(); trial++)
  {
    for (int t = 0; t < header.mTrials[trial].mMarkerObservations.size(); t++)
    {
      // 2.1. Populate the protobuf frame object in memory
      proto::SubjectOnDiskSensorFrame sensorsFrameProto;
      header.writeSensorsFrame(&sensorsFrameProto, trial, t, maxNumForcePlates);
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

      for (int pass = 0; pass < header.mTrials[trial].mTrialPasses.size();
           pass++)
      {
        proto::SubjectOnDiskProcessingPassFrame passFrameProto;
        header.writeProcessingPassFrame(&passFrameProto, trial, t, pass);
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

        // 1.4. Serialize the protobuf header object
        std::string headerSerialized = "";
        headerProto.SerializeToString(&headerSerialized);

        // 1.5. Write the length of the message as an integer header
        int64_t headerSize = headerSerialized.size();
        fwrite(&headerSize, sizeof(int64_t), 1, file);

        // 1.6. Write the serialized data to the file
        fwrite(headerSerialized.c_str(), sizeof(char), headerSize, file);

        firstTrial = false;
      }

      // 2.4. Write the serialized data to the file
      std::cout << "Writing the sensor frame for trial " << trial << " and t "
                << t << " of size " << sensorFrameSize << " at bytes "
                << ftell(file) << std::endl;
      fwrite(
          sensorFrameSerialized.c_str(), sizeof(char), sensorFrameSize, file);
      for (int pass = 0; pass < header.mTrials[trial].mTrialPasses.size();
           pass++)
      {
        std::cout << "Writing the pass " << pass << " frame for trial " << trial
                  << " and t " << t << " of size " << sensorFrameSize
                  << " at bytes " << ftell(file) << std::endl;
        fwrite(
            passFramesSerialized[pass].c_str(),
            sizeof(char),
            passFrameSize,
            file);
      }
    }
  }

  fclose(file);
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

  // 1. Open the file
  FILE* file = fopen(mPath.c_str(), "r");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk attempting to open file that deos not exist: "
              << mPath << std::endl;
    throw new std::exception();
  }
  // 2. Read the length of the message from the integer header
  int64_t headerSize = -1;
  int64_t elementsRead = fread(&headerSize, sizeof(int64_t), 1, file);
  if (elementsRead != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << mPath
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
              << mPath << ": was unable to read full requested header size "
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
              << mPath << ": got an error parsing the protobuf file header."
              << std::endl;
    throw new std::exception();
  }

  if (passNumberToLoad < 0 || passNumberToLoad >= header.passes_size())
  {
    std::cout << "SubjectOnDisk attempting to read an out of bounds skeleton, "
                 "requested skeleton from processing pass "
              << passNumberToLoad << "/" << header.passes_size() << "."
              << std::endl;
    throw new std::exception();
  }

  tinyxml2::XMLDocument osimFile;
  osimFile.Parse(header.passes(passNumberToLoad).model_osim_text().c_str());
  OpenSimFile osimParsed
      = OpenSimParser::parseOsim(osimFile, mPath, geometryFolder);

  fclose(file);

  return osimParsed.skeleton;
}

/// This will read the raw OpenSim XML file text out of the binary, and return
/// it as a string
std::string SubjectOnDisk::getOpensimFileText(int passNumberToLoad)
{
  // 1. Open the file
  FILE* file = fopen(mPath.c_str(), "r");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk attempting to open file that deos not exist: "
              << mPath << std::endl;
    throw new std::exception();
  }
  // 2. Read the length of the message from the integer header
  int64_t headerSize = -1;
  int64_t elementsRead = fread(&headerSize, sizeof(int64_t), 1, file);
  if (elementsRead != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << mPath
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
              << mPath << ": was unable to read full requested header size "
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
              << mPath << ": got an error parsing the protobuf file header."
              << std::endl;
    throw new std::exception();
  }

  if (passNumberToLoad < 0 || passNumberToLoad >= header.passes_size())
  {
    std::cout << "SubjectOnDisk attempting to read an out of bounds skeleton, "
                 "requested skeleton from processing pass "
              << passNumberToLoad << "/" << header.passes_size() << "."
              << std::endl;
    throw new std::exception();
  }

  return header.passes(passNumberToLoad).model_osim_text();
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

  int linearFrameStart = 0;
  for (int i = 0; i < trial; i++)
  {
    const int pastTrialNumPasses = getTrialNumProcessingPasses(i);
    const int pastTrialFrameSize
        = (mSensorFrameSize + (pastTrialNumPasses * mProcessingPassFrameSize));
    linearFrameStart += getTrialLength(i) * pastTrialFrameSize;
  }
  const int numPasses = getTrialNumProcessingPasses(trial);
  const int frameSize
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
    int offsetBytes
        = mDataSectionStart + (linearFrameStart + (i * stride * frameSize));

    std::shared_ptr<Frame> frame = std::make_shared<Frame>();
    if (includeSensorData)
    {
      std::cout << "Reading sensor frame " << i << " of " << numFramesToRead
                << " for trial " << trial << " at offset " << offsetBytes
                << std::endl;

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
          &proto, mHeader, trial, startFrame + (i * stride));
    }
    if (includeProcessingPasses)
    {
      std::cout << "Reading passes for frame " << i << " of " << numFramesToRead
                << " for trial " << trial << " at offset " << offsetBytes
                << std::endl;
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
            mHeader,
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

  this->missingGRFReason = header.mTrials[trial].mMissingGRFReason[t];

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
  int numForcePlates = 0;
  if (header.mTrials.size() > trial)
  {
    numForcePlates = header.mTrials[trial].mNumForcePlates;
  }
  for (int i = 0; i < numForcePlates; i++)
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
      rawForcePlateCenterOfPressures.push_back(forceCop);
      rawForcePlateTorques.push_back(forceTorques);
      rawForcePlateForces.push_back(force);
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
  type = header.mPasses[pass].mType;
  // s_t markerRMS;
  markerRMS = header.mTrials[trial].mTrialPasses[pass].mMarkerRMS[t];
  // s_t markerMax;
  markerMax = header.mTrials[trial].mTrialPasses[pass].mMarkerMax[t];
  // s_t linearResidual;
  linearResidual = header.mTrials[trial].mTrialPasses[pass].mLinearResidual[t];
  // s_t angularResidual;
  angularResidual
      = header.mTrials[trial].mTrialPasses[pass].mAngularResidual[t];
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

  // // These are masks for which DOFs are observed

  // Eigen::VectorXi posObserved;
  posObserved = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    posObserved(i)
        = header.mTrials[trial].mTrialPasses[pass].mDofPositionsObserved[i];
  }

  // // These are masks for which DOFs have been finite differenced (if they
  // // haven't been finite differenced, they're from real sensors and therefore
  // // more trustworthy)
  // Eigen::VectorXi velFiniteDifferenced;
  velFiniteDifferenced = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    velFiniteDifferenced(i) = header.mTrials[trial]
                                  .mTrialPasses[pass]
                                  .mDofVelocitiesFiniteDifferenced[i];
  }

  // Eigen::VectorXi accFiniteDifferenced;
  accFiniteDifferenced = Eigen::VectorXi::Zero(header.mNumDofs);
  for (int i = 0; i < header.mNumDofs; i++)
  {
    accFiniteDifferenced(i) = header.mTrials[trial]
                                  .mTrialPasses[pass]
                                  .mDofAccelerationFiniteDifferenced[i];
  }
}

/// This returns the number of trials on the subject
int SubjectOnDisk::getNumTrials()
{
  return mHeader.mTrials.size();
}

/// This returns the length of the trial
int SubjectOnDisk::getTrialLength(int trial)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return 0;
  }
  return mHeader.mTrials[trial].mLength;
}

/// This returns the number of processing passes in the trial
int SubjectOnDisk::getTrialNumProcessingPasses(int trial)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return 0;
  }
  return mHeader.mTrials[trial].mTrialPasses.size();
}

/// This returns the timestep size for the trial
s_t SubjectOnDisk::getTrialTimestep(int trial)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return 0.01;
  }
  return mHeader.mTrials[trial].mTimestep;
}

/// This returns the number of DOFs for the model on this Subject
int SubjectOnDisk::getNumDofs()
{
  return mHeader.mNumDofs;
}

/// This returns the vector of enums of type 'MissingGRFReason', which labels
/// why each time step was identified as 'probablyMissingGRF'.
std::vector<MissingGRFReason> SubjectOnDisk::getMissingGRFReason(int trial)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<MissingGRFReason>();
  }
  return mHeader.mTrials[trial].mMissingGRFReason;
}

int SubjectOnDisk::getNumProcessingPasses()
{
  return mHeader.mPasses.size();
}

ProcessingPassType SubjectOnDisk::getProcessingPassType(int processingPass)
{
  return mHeader.mPasses[processingPass].mType;
}

std::vector<bool> SubjectOnDisk::getDofPositionsObserved(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader.mTrials[trial]
      .mTrialPasses[processingPass]
      .mDofPositionsObserved;
}

std::vector<bool> SubjectOnDisk::getDofVelocitiesFiniteDifferenced(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader.mTrials[trial]
      .mTrialPasses[processingPass]
      .mDofVelocitiesFiniteDifferenced;
}

std::vector<bool> SubjectOnDisk::getDofAccelerationsFiniteDifferenced(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<bool>();
  }
  return mHeader.mTrials[trial]
      .mTrialPasses[processingPass]
      .mDofAccelerationFiniteDifferenced;
}

std::vector<s_t> SubjectOnDisk::getTrialLinearResidualNorms(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader.mTrials[trial].mTrialPasses[processingPass].mLinearResidual;
}

std::vector<s_t> SubjectOnDisk::getTrialAngularResidualNorms(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader.mTrials[trial].mTrialPasses[processingPass].mAngularResidual;
}

std::vector<s_t> SubjectOnDisk::getTrialMarkerRMSs(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader.mTrials[trial].mTrialPasses[processingPass].mMarkerRMS;
}

std::vector<s_t> SubjectOnDisk::getTrialMarkerMaxs(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader.mTrials[trial].mTrialPasses[processingPass].mMarkerMax;
}

/// This returns the maximum absolute velocity of any DOF at each timestep for a
/// given trial
std::vector<s_t> SubjectOnDisk::getTrialMaxJointVelocity(
    int trial, int processingPass)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<s_t>();
  }
  return mHeader.mTrials[trial].mTrialPasses[processingPass].mJointsMaxVelocity;
}

/// This returns the list of contact body names for this Subject
std::vector<std::string> SubjectOnDisk::getGroundForceBodies()
{
  return mHeader.mGroundContactBodies;
}

/// This returns the list of custom value names stored in this subject
std::vector<std::string> SubjectOnDisk::getCustomValues()
{
  return mHeader.mCustomValueNames;
}

/// This returns the dimension of the custom value specified by `valueName`
int SubjectOnDisk::getCustomValueDim(std::string valueName)
{
  for (int i = 0; i < mHeader.mCustomValueNames.size(); i++)
  {
    if (mHeader.mCustomValueNames[i] == valueName)
    {
      return mHeader.mCustomValueLengths[i];
    }
  }
  std::cout << "WARNING: Requested getCustomValueDim() for value \""
            << valueName
            << "\", which is not in this SubjectOnDisk. Options are: [";
  for (int i = 0; i < mHeader.mCustomValueNames.size(); i++)
  {
    std::cout << " \"" << mHeader.mCustomValueNames[i] << "\" ";
  }
  std::cout << "]. Returning 0." << std::endl;
  return 0;
}

/// The name of the trial, if provided, or else an empty string
std::string SubjectOnDisk::getTrialName(int trial)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return "";
  }
  return mHeader.mTrials[trial].mName;
}

std::string SubjectOnDisk::getBiologicalSex()
{
  return mHeader.mBiologicalSex;
}

double SubjectOnDisk::getHeightM()
{
  return mHeader.mHeightM;
}

double SubjectOnDisk::getMassKg()
{
  return mHeader.mMassKg;
}

/// This gets the tags associated with the subject, if there are any.
std::vector<std::string> SubjectOnDisk::getSubjectTags()
{
  return mHeader.mSubjectTags;
}

/// This gets the tags associated with the trial, if there are any.
std::vector<std::string> SubjectOnDisk::getTrialTags(int trial)
{
  if (trial >= 0 && trial < mHeader.mTrials.size())
  {
    return mHeader.mTrials[trial].mTrialTags;
  }
  else
  {
    return std::vector<std::string>();
  }
}

int SubjectOnDisk::getAgeYears()
{
  return mHeader.mAgeYears;
}

/// This returns the number of raw force plates that were used to generate the
/// data
int SubjectOnDisk::getNumForcePlates(int trial)
{
  if (trial >= 0 && trial < mHeader.mTrials.size())
    return mHeader.mTrials[trial].mNumForcePlates;
  return 0;
}

/// This returns the corners (in 3D space) of the selected force plate, for
/// this trial. Empty arrays on out of bounds.
std::vector<Eigen::Vector3s> SubjectOnDisk::getForcePlateCorners(
    int trial, int forcePlate)
{
  if (trial < 0 || trial >= mHeader.mTrials.size())
  {
    return std::vector<Eigen::Vector3s>();
  }
  if (forcePlate < 0 || forcePlate >= mHeader.mTrials[trial].mNumForcePlates)
  {
    return std::vector<Eigen::Vector3s>();
  }
  return mHeader.mTrials[trial].mForcePlateCorners[forcePlate];
}

/// This gets the href link associated with the subject, if there is one.
std::string SubjectOnDisk::getHref()
{
  return mHeader.mHref;
}

/// This gets the notes associated with the subject, if there are any.
std::string SubjectOnDisk::getNotes()
{
  return mHeader.mNotes;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Builders, to create a SubjectOnDisk from scratch
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SubjectOnDiskTrialPass::SubjectOnDiskTrialPass()
{
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setType(ProcessingPassType type)
{
  mType = type;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setPoses(Eigen::MatrixXs poses)
{
  mPos = poses;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setVels(Eigen::MatrixXs vels)
{
  mVel = vels;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setAccs(Eigen::MatrixXs accs)
{
  mAcc = accs;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setTaus(Eigen::MatrixXs taus)
{
  mTaus = taus;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setGroundBodyWrenches(
    Eigen::MatrixXs wrenches)
{
  mGroundBodyWrenches = wrenches;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setGroundBodyCopTorqueForce(
    Eigen::MatrixXs copTorqueForces)
{
  mGroundBodyCopTorqueForce = copTorqueForces;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setComPoses(
    Eigen::MatrixXs poses)
{
  mComPoses = poses;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setComVels(Eigen::MatrixXs vels)
{
  mComVels = vels;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setComAccs(Eigen::MatrixXs accs)
{
  mComAccs = accs;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setDofPositionsObserved(
    std::vector<bool> dofPositionsObserved)
{
  mDofPositionsObserved = dofPositionsObserved;
  return *this;
}

SubjectOnDiskTrialPass&
SubjectOnDiskTrialPass::setDofVelocitiesFiniteDifferenced(
    std::vector<bool> dofVelocitiesFiniteDifferenced)
{
  mDofVelocitiesFiniteDifferenced = dofVelocitiesFiniteDifferenced;
  return *this;
}

SubjectOnDiskTrialPass&
SubjectOnDiskTrialPass::setDofAccelerationFiniteDifferenced(
    std::vector<bool> dofAccelerationFiniteDifference)
{
  mDofAccelerationFiniteDifferenced = dofAccelerationFiniteDifference;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setMarkerRMS(
    std::vector<s_t> markerRMS)
{
  mMarkerRMS = markerRMS;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setMarkerMax(
    std::vector<s_t> markerMax)
{
  mMarkerMax = markerMax;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setLinearResidual(
    std::vector<s_t> linearResidual)
{
  mLinearResidual = linearResidual;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrialPass::setAngularResidual(
    std::vector<s_t> angularResidual)
{
  mAngularResidual = angularResidual;
  return *this;
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
}

SubjectOnDiskTrial::SubjectOnDiskTrial()
{
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setName(const std::string& name)
{
  mName = name;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setTimestep(s_t timestep)
{
  mTimestep = timestep;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setTrialTags(
    std::vector<std::string> trialTags)
{
  mTrialTags = trialTags;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setMissingGRFReason(
    std::vector<MissingGRFReason> missingGRFReason)
{
  mMissingGRFReason = missingGRFReason;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setCustomValues(
    std::vector<Eigen::MatrixXs> customValues)
{
  mCustomValues = customValues;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setMarkerNamesGuessed(
    bool markersGuessed)
{
  mMarkerNamesGuessed = markersGuessed;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setMarkerObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations)
{
  mMarkerObservations = markerObservations;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setAccObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> accObservations)
{
  mAccObservations = accObservations;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setGyroObservations(
    std::vector<std::map<std::string, Eigen::Vector3s>> gyroObservations)
{
  mGyroObservations = gyroObservations;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setEmgObservations(
    std::vector<std::map<std::string, Eigen::VectorXs>> emgObservations)
{
  mEmgObservations = emgObservations;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setExoTorques(
    std::map<int, Eigen::VectorXs> exoTorques)
{
  mExoTorques = exoTorques;
  return *this;
}

SubjectOnDiskTrial& SubjectOnDiskTrial::setForcePlates(
    std::vector<ForcePlate> forcePlates)
{
  mForcePlates = forcePlates;
  return *this;
}

SubjectOnDiskTrialPass& SubjectOnDiskTrial::addPass()
{
  mTrialPasses.push_back(SubjectOnDiskTrialPass());
  return mTrialPasses.back();
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

  mTrialPasses.clear();
  for (int i = 0; i < proto.processing_pass_header_size(); i++)
  {
    SubjectOnDiskTrialPass pass;
    pass.read(proto.processing_pass_header(i));
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

void SubjectOnDiskTrial::write(proto::SubjectOnDiskTrialHeader* proto)
{
  // std::string mName;
  proto->set_name(mName);
  // s_t mTimestep;
  proto->set_trial_timestep(mTimestep);
  // Set the length of the trial to whatever the length of our marker
  // observations is
  proto->set_trial_length(mMarkerObservations.size());
  // std::vector<std::string> mTrialTags;
  for (int i = 0; i < mTrialTags.size(); i++)
  {
    proto->add_trial_tag(mTrialTags[i]);
  }
  for (int i = 0; i < mTrialPasses.size(); i++)
  {
    proto::SubjectOnDiskTrialProcessingPassHeader* passProto
        = proto->add_processing_pass_header();
    mTrialPasses[i].write(passProto);
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
    for (Eigen::Vector3s& corners : forcePlate.corners)
    {
      for (int i = 0; i < 3; i++)
      {
        proto->add_force_plate_corners(corners(i));
      }
    }
  }
}

SubjectOnDiskPassHeader::SubjectOnDiskPassHeader()
{
  // Do nothing
}

SubjectOnDiskPassHeader& SubjectOnDiskPassHeader::setProcessingPassType(
    ProcessingPassType type)
{
  mType = type;
  return *this;
}

SubjectOnDiskPassHeader& SubjectOnDiskPassHeader::setOpenSimFileText(
    const std::string& openSimFileText)
{
  mOpenSimFileText = openSimFileText;
  return *this;
}

void SubjectOnDiskPassHeader::write(dart::proto::SubjectOnDiskPass* proto)
{
  proto->set_pass_type(passTypeToProto(mType));
  proto->set_model_osim_text(mOpenSimFileText);
}

void SubjectOnDiskPassHeader::read(const dart::proto::SubjectOnDiskPass& proto)
{
  mType = passTypeFromProto(proto.pass_type());
  mOpenSimFileText = proto.model_osim_text();
}

SubjectOnDiskHeader::SubjectOnDiskHeader()
{
  // Do nothing
}

SubjectOnDiskHeader& SubjectOnDiskHeader::setNumDofs(int dofs)
{
  mNumDofs = dofs;
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

SubjectOnDiskPassHeader& SubjectOnDiskHeader::addProcessingPass()
{
  mPasses.push_back(SubjectOnDiskPassHeader());
  return mPasses.back();
}

SubjectOnDiskTrial& SubjectOnDiskHeader::addTrial()
{
  mTrials.push_back(SubjectOnDiskTrial());
  return mTrials.back();
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
    for (int t = 0; t < mTrials[trial].mMarkerObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial].mMarkerObservations[t])
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
    for (int t = 0; t < mTrials[trial].mAccObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial].mAccObservations[t])
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
    for (int t = 0; t < mTrials[trial].mGyroObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial].mGyroObservations[t])
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
    for (auto& pair : mTrials[trial].mExoTorques)
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
    for (int t = 0; t < mTrials[trial].mEmgObservations.size(); t++)
    {
      for (auto& pair : mTrials[trial].mEmgObservations[t])
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
                    << mTrials[trial].mName << " timestep " << t << " for emg "
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
  // // The passes we applied to this data, along with the result skeletons that
  // // were generated by each pass.
  // std::vector<SubjectOnDiskPassHeader> mPasses;
  for (SubjectOnDiskPassHeader& pass : mPasses)
  {
    proto::SubjectOnDiskPass* passProto = header->add_passes();
    pass.write(passProto);
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
        mTrials.size() > 0 && mTrials[0].mCustomValues.size() > 0
            ? mTrials[0].mCustomValues[0].rows()
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
  header->set_version(3);

  // // These are the trials, which contain the actual data
  // std::vector<SubjectOnDiskTrialBuilder> mTrials;
  for (auto& trial : mTrials)
  {
    proto::SubjectOnDiskTrialHeader* trialHeader = header->add_trial_header();
    trial.write(trialHeader);
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
}

void SubjectOnDiskHeader::read(const dart::proto::SubjectOnDiskHeader& proto)
{
  // // How many DOFs are in the skeleton
  // int mNumDofs;
  mNumDofs = proto.num_dofs();

  // // The passes we applied to this data, along with the result skeletons that
  // // were generated by each pass.
  // std::vector<SubjectOnDiskPassHeader> mPasses;
  mPasses.clear();
  for (int i = 0; i < proto.passes_size(); i++)
  {
    SubjectOnDiskPassHeader pass;
    pass.read(proto.passes(i));
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
    SubjectOnDiskTrial trial;
    trial.read(proto.trial_header(i));
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
    for (int i = 0; i < mTrials[trial].mCustomValues.size(); i++)
    {
      for (int j = 0; j < mTrials[trial].mCustomValues[i].rows(); j++)
      {
        proto->add_custom_values(mTrials[trial].mCustomValues[i](j, t));
      }
    }
  }

  // // These are marker observations on this frame, with all NaNs indicating
  // that that marker was not observed on this frame repeated double marker_obs
  // = 2;
  if (mTrials.size() > trial && mTrials[trial].mMarkerObservations.size() > t)
  {
    for (std::string& name : mMarkerNames)
    {
      if (mTrials[trial].mMarkerObservations[t].find(name)
          != mTrials[trial].mMarkerObservations[t].end())
      {
        proto->add_marker_obs(
            mTrials[trial].mMarkerObservations[t].at(name)(0));
        proto->add_marker_obs(
            mTrials[trial].mMarkerObservations[t].at(name)(1));
        proto->add_marker_obs(
            mTrials[trial].mMarkerObservations[t].at(name)(2));
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
  if (mTrials.size() > trial && mTrials[trial].mAccObservations.size() > t)
  {
    for (std::string& name : mAccNames)
    {
      if (mTrials[trial].mAccObservations[t].find(name)
          != mTrials[trial].mAccObservations[t].end())
      {
        proto->add_acc_obs(mTrials[trial].mAccObservations[t].at(name)(0));
        proto->add_acc_obs(mTrials[trial].mAccObservations[t].at(name)(1));
        proto->add_acc_obs(mTrials[trial].mAccObservations[t].at(name)(2));
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
  if (mTrials.size() > trial && mTrials[trial].mGyroObservations.size() > t)
  {
    for (std::string& name : mGyroNames)
    {
      if (mTrials[trial].mGyroObservations[t].find(name)
          != mTrials[trial].mGyroObservations[t].end())
      {
        proto->add_gyro_obs(mTrials[trial].mGyroObservations[t].at(name)(0));
        proto->add_gyro_obs(mTrials[trial].mGyroObservations[t].at(name)(1));
        proto->add_gyro_obs(mTrials[trial].mGyroObservations[t].at(name)(2));
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
  if (mTrials.size() > trial && mTrials[trial].mEmgObservations.size() > t)
  {
    for (std::string& name : mEmgNames)
    {
      if (mTrials[trial].mEmgObservations[t].find(name)
          != mTrials[trial].mEmgObservations[t].end())
      {
        for (int i = 0; i < mEmgDim; i++)
        {
          proto->add_emg_obs(mTrials[trial].mEmgObservations[t].at(name)(i));
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
      if (mTrials[trial].mExoTorques.find(dof)
              != mTrials[trial].mExoTorques.end()
          && mTrials[trial].mExoTorques.at(dof).size() > t)
      {
        proto->add_exo_obs(mTrials[trial].mExoTorques.at(dof)(t));
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
        && forcePlateIdx < mTrials[trial].mForcePlates.size()
        && mTrials[trial].mForcePlates[forcePlateIdx].centersOfPressure.size()
               > t
        && mTrials[trial].mForcePlates[forcePlateIdx].forces.size() > t
        && mTrials[trial].mForcePlates[forcePlateIdx].moments.size() > t)
    {
      for (int i = 0; i < 3; i++)
      {
        proto->add_raw_force_plate_cop(
            mTrials[trial].mForcePlates[forcePlateIdx].centersOfPressure[t](i));
        proto->add_raw_force_plate_torque(
            mTrials[trial].mForcePlates[forcePlateIdx].moments[t](i));
        proto->add_raw_force_plate_force(
            mTrials[trial].mForcePlates[forcePlateIdx].forces[t](i));
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
  if (mTrials.size() > trial && mTrials[trial].mTrialPasses.size() > pass
      && mTrials[trial].mTrialPasses[pass].mPos.size() > t
      && mTrials[trial].mTrialPasses[pass].mVel.size() > t
      && mTrials[trial].mTrialPasses[pass].mAcc.size() > t
      && mTrials[trial].mTrialPasses[pass].mTaus.size() > t)
  {
    for (int i = 0; i < mNumDofs; i++)
    {
      proto->add_pos(mTrials[trial].mTrialPasses[pass].mPos(i, t));
      proto->add_vel(mTrials[trial].mTrialPasses[pass].mVel(i, t));
      proto->add_acc(mTrials[trial].mTrialPasses[pass].mAcc(i, t));
      proto->add_tau(mTrials[trial].mTrialPasses[pass].mTaus(i, t));
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

  if (mTrials.size() > trial && mTrials[trial].mTrialPasses.size() > pass
      && mTrials[trial].mTrialPasses[pass].mGroundBodyWrenches.cols() > t
      && mTrials[trial].mTrialPasses[pass].mGroundBodyWrenches.rows()
             == 6 * mGroundContactBodies.size()
      && mTrials[trial].mTrialPasses[pass].mGroundBodyCopTorqueForce.cols() > t
      && mTrials[trial].mTrialPasses[pass].mGroundBodyCopTorqueForce.rows()
             == 9 * mGroundContactBodies.size())
  {
    for (int i = 0; i < mGroundContactBodies.size(); i++)
    {
      for (int j = 0; j < 6; j++)
      {
        proto->add_ground_contact_wrench(
            mTrials[trial].mTrialPasses[pass].mGroundBodyWrenches(
                i * 6 + j, t));
      }
      for (int j = 0; j < 3; j++)
      {
        proto->add_ground_contact_center_of_pressure(
            mTrials[trial].mTrialPasses[pass].mGroundBodyCopTorqueForce(
                i * 9 + j, t));
        proto->add_ground_contact_torque(
            mTrials[trial].mTrialPasses[pass].mGroundBodyCopTorqueForce(
                i * 9 + 3 + j, t));
        proto->add_ground_contact_force(
            mTrials[trial].mTrialPasses[pass].mGroundBodyCopTorqueForce(
                i * 9 + 6 + j, t));
      }
    }
  }
  else
  {
    std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                 "trialGroundBodyWrenches or trialGroundBodyCopTorqueForce "
                 "out-of-bounds for trial "
              << trial << " frame " << t << std::endl;
  }

  // // These are the center of mass kinematics
  // repeated double com_pos = 9;
  // repeated double com_vel = 10;
  // repeated double com_acc = 11;
  for (int i = 0; i < 3; i++)
  {
    if (trial < mTrials.size() && pass < mTrials[trial].mTrialPasses.size()
        && t < mTrials[trial].mTrialPasses[pass].mComPoses.cols())
    {
      proto->add_com_pos(mTrials[trial].mTrialPasses[pass].mComPoses(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComPoses out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_pos(std::nan(""));
    }
    if (trial < mTrials.size() && pass < mTrials[trial].mTrialPasses.size()
        && t < mTrials[trial].mTrialPasses[pass].mComVels.cols())
    {
      proto->add_com_vel(mTrials[trial].mTrialPasses[pass].mComVels(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComVels out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_vel(std::nan(""));
    }
    if (trial < mTrials.size() && pass < mTrials[trial].mTrialPasses.size()
        && t < mTrials[trial].mTrialPasses[pass].mComAccs.cols())
    {
      proto->add_com_acc(mTrials[trial].mTrialPasses[pass].mComAccs(i, t));
    }
    else
    {
      std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                   "trialComAccs out-of-bounds for trial "
                << trial << std::endl;
      proto->add_com_acc(std::nan(""));
    }
  }
}

} // namespace biomechanics
} // namespace dart