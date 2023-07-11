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
  int version = header.version();

  mNumDofs = header.num_dofs();
  mNumTrials = header.num_trials();
  for (int i = 0; i < header.ground_contact_body_size(); i++)
  {
    mGroundContactBodies.push_back(header.ground_contact_body(i));
  }
  for (int i = 0; i < header.custom_value_name_size(); i++)
  {
    mCustomValues.push_back(header.custom_value_name(i));
    mCustomValueLengths.push_back(header.custom_value_length(i));
  }
  for (int i = 0; i < header.trial_name_size(); i++)
  {
    auto& trialHeader = header.trial_header(i);
    std::vector<bool> missingGRF;
    std::vector<MissingGRFReason> missingGRFReason;
    std::vector<s_t> residualNorms;
    std::vector<s_t> maxJointVelocity;
    for (int t = 0; t < trialHeader.missing_grf_size(); t++)
    {
      missingGRF.push_back(trialHeader.missing_grf(t));
      missingGRFReason.push_back(
          missingGRFReasonFromProto(trialHeader.missing_grf_reason(t)));
      residualNorms.push_back(trialHeader.residual(t));
      if (version > 1 && trialHeader.joints_max_velocity_size() > t)
      {
        maxJointVelocity.push_back(trialHeader.joints_max_velocity(t));
      }
    }
    mProbablyMissingGRF.push_back(missingGRF);
    mMissingGRFReason.push_back(missingGRFReason);
    mTrialResidualNorms.push_back(residualNorms);
    mTrialMaxJointVelocity.push_back(maxJointVelocity);
    mTrialLength.push_back(trialHeader.trial_length());
    mTrialTimesteps.push_back(trialHeader.trial_timestep());
    std::vector<bool> dofPositionsObserved;
    std::vector<bool> dofVelocitiesFiniteDifferenced;
    std::vector<bool> dofAccelerationFiniteDifferenced;
    for (int i = 0; i < trialHeader.dof_positions_observed_size(); i++)
    {
      dofPositionsObserved.push_back(trialHeader.dof_positions_observed(i));
      dofVelocitiesFiniteDifferenced.push_back(
          trialHeader.dof_velocities_finite_differenced(i));
      dofAccelerationFiniteDifferenced.push_back(
          trialHeader.dof_acceleration_finite_differenced(i));
    }
    mDofPositionsObserved.push_back(dofPositionsObserved);
    mDofVelocitiesFiniteDifferenced.push_back(dofVelocitiesFiniteDifferenced);
    mDofAccelerationFiniteDifferenced.push_back(
        dofAccelerationFiniteDifferenced);
    if (version > 1)
    {
      std::vector<std::string> trialTags;
      for (int i = 0; i < trialHeader.trial_tag_size(); i++)
      {
        trialTags.push_back(trialHeader.trial_tag(i));
      }
      mTrialTags.push_back(trialTags);
    }
    int numForcePlates = trialHeader.num_force_plates();
    mTrialNumForcePlates.push_back(numForcePlates);
    std::vector<std::vector<Eigen::Vector3s>> forcePlateCorners;
    for (int i = 0; i < numForcePlates; i++)
    {
      std::vector<Eigen::Vector3s> corners;
      for (int corner = 0; corner < trialHeader.force_plate_corners_size() / 3;
           corner++)
      {
        Eigen::Vector3s pt(
            trialHeader.force_plate_corners(i * 3 + 0),
            trialHeader.force_plate_corners(i * 3 + 1),
            trialHeader.force_plate_corners(i * 3 + 2));
        corners.push_back(pt);
      }
      forcePlateCorners.push_back(corners);
    }
    mTrialForcePlateCorners.push_back(forcePlateCorners);
    mTrialNames.push_back(header.trial_name(i));
  }

  if (version > 1)
  {
    mBiologicalSex = header.biological_sex();
    mMassKg = header.mass_kg();
    mHeightM = header.height_m();
    mAgeYears = header.age_years();
    for (int i = 0; i < header.subject_tag_size(); i++)
    {
      mSubjectTags.push_back(header.subject_tag(i));
    }
    mMarkerNames.resize(header.marker_name_size());
    for (int i = 0; i < header.marker_name_size(); i++)
    {
      mMarkerNames[i] = header.marker_name(i);
    }
    mAccNames.resize(header.acc_name_size());
    for (int i = 0; i < header.acc_name_size(); i++)
    {
      mAccNames[i] = header.acc_name(i);
    }
    mGyroNames.resize(header.gyro_name_size());
    for (int i = 0; i < header.gyro_name_size(); i++)
    {
      mGyroNames[i] = header.gyro_name(i);
    }
    mEmgNames.resize(header.emg_name_size());
    for (int i = 0; i < header.emg_name_size(); i++)
    {
      mEmgNames[i] = header.emg_name(i);
    }
    mEmgDim = header.emg_dim();
  }

  // int32 num_dofs = 1;
  // int32 num_trials = 2;
  // repeated string ground_contact_body = 3;
  // repeated string custom_value_name = 6;
  // repeated int32 custom_value_length = 7;
  // string model_osim_text = 8;
  // repeated SubjectOnDiskTrialHeader trial_header = 9;
  // // The trial names, if provided, or empty strings
  // repeated string trial_name = 10;
  // // An optional link to the web where this subject came from
  // string href = 11;
  // // Any text-based notes on the subject data, like citations etc
  // string notes = 12;
  // // The version number for this file format
  // int32 version = 13;
  // // This is the size of each frame in bytes. This should be constant across
  // all frames in the file, to allow easy seeking. int32 frame_size = 14;
  mHref = header.href();
  mNotes = header.notes();
  mFrameSize = header.frame_size();
  mDataSectionStart = sizeof(int64_t) + headerSize;

  fclose(file);
}

/// This will read the skeleton from the binary, and optionally use the passed
/// in Geometry folder.
std::shared_ptr<dynamics::Skeleton> SubjectOnDisk::readSkel(
    std::string geometryFolder)
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

  tinyxml2::XMLDocument osimFile;
  osimFile.Parse(header.model_osim_text().c_str());
  OpenSimFile osimParsed
      = OpenSimParser::parseOsim(osimFile, mPath, geometryFolder);

  fclose(file);

  return osimParsed.skeleton;
}

/// This will read the raw OpenSim XML file text out of the binary, and return
/// it as a string
std::string SubjectOnDisk::readRawOsimFileText()
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

  return header.model_osim_text();
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
    int stride,
    s_t contactThreshold)
{
  (void)trial;
  (void)startFrame;
  (void)stride;
  (void)numFramesToRead;

  std::vector<std::shared_ptr<Frame>> result;

  // 1. Open the file
  FILE* file = fopen(mPath.c_str(), "r");

  int linearFrameStart = 0;
  for (int i = 0; i < trial; i++)
  {
    linearFrameStart += mTrialLength[i];
  }
  linearFrameStart += startFrame;

  int remainingFrames = mTrialLength[trial] - startFrame;
  if (remainingFrames < numFramesToRead)
  {
    numFramesToRead = remainingFrames;
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
    int offsetBytes = mDataSectionStart + (mFrameSize * linearFrameStart)
                      + (i * stride * mFrameSize);
    fseek(file, offsetBytes, SEEK_SET);

    // 3. Allocate a buffer to hold the serialized data
    std::vector<char> serializedFrame(mFrameSize);

    // 4. Read the serialized data from the file
    int64_t bytesRead
        = fread(serializedFrame.data(), sizeof(char), mFrameSize, file);

    if (bytesRead != mFrameSize)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": was unable to read full requested frame size "
          << mFrameSize << " at offset "
          << (offsetBytes + i * stride * mFrameSize)
          << ", corresponding to trial " << trial << " and frame "
          << startFrame + (i * stride) << " (" << i * stride << " into a "
          << numFramesToRead << " frame read), instead only got " << bytesRead
          << " bytes." << std::endl;
      throw new std::exception();
    }

    // 5. Deserialize the data into a protobuf object
    proto::SubjectOnDiskFrame proto;
    bool parseSuccess
        = proto.ParseFromArray(serializedFrame.data(), serializedFrame.size());
    if (!parseSuccess)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": got an error parsing frame at offset "
          << (offsetBytes + i * stride * mFrameSize)
          << ", corresponding to trial " << trial << " and frame "
          << startFrame + (i * stride) << " (" << i * stride << " into a "
          << numFramesToRead << " frame read)." << std::endl;
      throw new std::exception();
    }

    // 6. Copy the results out into a frame
    std::shared_ptr<Frame> frame = std::make_shared<Frame>();
    frame->trial = trial;
    frame->t = startFrame + (i * stride);
    frame->residual = mTrialResidualNorms[trial][frame->t];
    frame->probablyMissingGRF = mProbablyMissingGRF[trial][frame->t];
    frame->missingGRFReason = mMissingGRFReason[trial][frame->t];
    frame->pos = Eigen::VectorXd(mNumDofs);
    frame->vel = Eigen::VectorXd(mNumDofs);
    frame->acc = Eigen::VectorXd(mNumDofs);
    frame->tau = Eigen::VectorXd(mNumDofs);
    for (int i = 0; i < mNumDofs; i++)
    {
      frame->pos(i) = proto.pos(i);
      frame->vel(i) = proto.vel(i);
      frame->acc(i) = proto.acc(i);
      frame->tau(i) = proto.tau(i);
    }
    // These are boolean values (0 or 1) for each contact body indicating
    // whether or not it's in contact
    frame->contact = Eigen::VectorXi::Zero(mGroundContactBodies.size());
    // These are each 6-vector of contact body wrenches, all concatenated
    // together
    frame->groundContactWrenches
        = Eigen::VectorXd::Zero(mGroundContactBodies.size() * 6);
    // These are each 3-vector for each contact body, concatenated together
    frame->groundContactCenterOfPressure
        = Eigen::VectorXd::Zero(mGroundContactBodies.size() * 3);
    frame->groundContactTorque
        = Eigen::VectorXd::Zero(mGroundContactBodies.size() * 3);
    frame->groundContactForce
        = Eigen::VectorXd::Zero(mGroundContactBodies.size() * 3);
    for (int i = 0; i < mGroundContactBodies.size(); i++)
    {
      for (int j = 0; j < 6; j++)
      {
        frame->groundContactWrenches(i * 6 + j)
            = proto.ground_contact_wrench(i * 6 + j);
      }
      s_t contactNorm = frame->groundContactWrenches.segment<6>(i * 6).norm();
      if (contactNorm > contactThreshold)
      {
        frame->contact(i) = 1;
      }
      for (int j = 0; j < 3; j++)
      {
        frame->groundContactCenterOfPressure(i * 3 + j)
            = proto.ground_contact_center_of_pressure(i * 3 + j);
        frame->groundContactTorque(i * 3 + j)
            = proto.ground_contact_torque(i * 3 + j);
        frame->groundContactForce(i * 3 + j)
            = proto.ground_contact_force(i * 3 + j);
      }
    }

    frame->comPos = Eigen::Vector3s::Zero();
    frame->comVel = Eigen::Vector3s::Zero();
    frame->comAcc = Eigen::Vector3s::Zero();
    for (int i = 0; i < 3; i++)
    {
      frame->comPos(i) = proto.com_pos(i);
      frame->comVel(i) = proto.com_vel(i);
      frame->comAcc(i) = proto.com_acc(i);
    }

    frame->posObserved = Eigen::VectorXi::Zero(mNumDofs);
    frame->velFiniteDifferenced = Eigen::VectorXi::Zero(mNumDofs);
    frame->accFiniteDifferenced = Eigen::VectorXi::Zero(mNumDofs);
    for (int i = 0; i < mNumDofs; i++)
    {
      frame->posObserved(i) = mDofPositionsObserved[trial][i];
      frame->velFiniteDifferenced(i)
          = mDofVelocitiesFiniteDifferenced[trial][i];
      frame->accFiniteDifferenced(i)
          = mDofAccelerationFiniteDifferenced[trial][i];
    }
    // 7. Read out the marker, accelerometer, and gyro info as pairs
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      Eigen::Vector3s marker(
          proto.marker_obs(i * 3 + 0),
          proto.marker_obs(i * 3 + 1),
          proto.marker_obs(i * 3 + 2));
      if (!marker.hasNaN())
      {
        frame->markerObservations.emplace_back(mMarkerNames[i], marker);
      }
    }
    for (int i = 0; i < mAccNames.size(); i++)
    {
      Eigen::Vector3s acc(
          proto.acc_obs(i * 3 + 0),
          proto.acc_obs(i * 3 + 1),
          proto.acc_obs(i * 3 + 2));
      if (!acc.hasNaN())
      {
        frame->accObservations.emplace_back(mAccNames[i], acc);
      }
    }
    for (int i = 0; i < mGyroNames.size(); i++)
    {
      Eigen::Vector3s gyro(
          proto.gyro_obs(i * 3 + 0),
          proto.gyro_obs(i * 3 + 1),
          proto.gyro_obs(i * 3 + 2));
      if (!gyro.hasNaN())
      {
        frame->gyroObservations.emplace_back(mGyroNames[i], gyro);
      }
    }
    for (int i = 0; i < mEmgNames.size(); i++)
    {
      Eigen::VectorXs emgSequence(mEmgDim);
      for (int j = 0; j < mEmgDim; j++)
      {
        emgSequence(j) = proto.emg_obs(i * mEmgDim + j);
      }
      if (!emgSequence.hasNaN())
      {
        frame->emgSignals.emplace_back(mEmgNames[i], emgSequence);
      }
    }
    int numForcePlates = 0;
    if (mTrialNumForcePlates.size() > trial)
    {
      numForcePlates = mTrialNumForcePlates[trial];
    }
    for (int i = 0; i < numForcePlates; i++)
    {
      Eigen::Vector3s forceCop(
          proto.raw_force_plate_cop(i * 3 + 0),
          proto.raw_force_plate_cop(i * 3 + 1),
          proto.raw_force_plate_cop(i * 3 + 2));
      Eigen::Vector3s forceTorques(
          proto.raw_force_plate_torque(i * 3 + 0),
          proto.raw_force_plate_torque(i * 3 + 1),
          proto.raw_force_plate_torque(i * 3 + 2));
      Eigen::Vector3s force(
          proto.raw_force_plate_force(i * 3 + 0),
          proto.raw_force_plate_force(i * 3 + 1),
          proto.raw_force_plate_force(i * 3 + 2));
      if (!forceCop.hasNaN() && !forceTorques.hasNaN() && !force.hasNaN())
      {
        frame->rawForcePlateCenterOfPressures.push_back(forceCop);
        frame->rawForcePlateTorques.push_back(forceTorques);
        frame->rawForcePlateForces.push_back(force);
      }
    }

    result.push_back(frame);
  }

  fclose(file);

  return result;
}

/// This writes a subject out to disk in a compressed and random-seekable
/// binary format.
void SubjectOnDisk::writeSubject(
    const std::string& outputPath,
    // The OpenSim file XML gets copied into our binary bundle, along with
    // any necessary Geometry files
    const std::string& openSimFilePath,
    // The per-trial motion data
    std::vector<s_t> trialTimesteps,
    std::vector<Eigen::MatrixXs>& trialPoses,
    std::vector<Eigen::MatrixXs>& trialVels,
    std::vector<Eigen::MatrixXs>& trialAccs,
    std::vector<std::vector<bool>>& probablyMissingGRF,
    std::vector<std::vector<MissingGRFReason>>& missingGRFReason,
    std::vector<std::vector<bool>>& dofPositionsObserved,
    std::vector<std::vector<bool>>& dofVelocitiesFiniteDifferenced,
    std::vector<std::vector<bool>>& dofAccelerationFiniteDifferenced,
    std::vector<Eigen::MatrixXs>& trialTaus,
    std::vector<Eigen::MatrixXs>& trialComPoses,
    std::vector<Eigen::MatrixXs>& trialComVels,
    std::vector<Eigen::MatrixXs>& trialComAccs,
    std::vector<std::vector<s_t>> trialResidualNorms,
    // These are generalized 6-dof wrenches applied to arbitrary bodies
    // (generally by foot-ground contact, though other things too)
    std::vector<std::string>& groundForceBodies,
    std::vector<Eigen::MatrixXs>& trialGroundBodyWrenches,
    std::vector<Eigen::MatrixXs>& trialGroundBodyCopTorqueForce,
    // We include this to allow the binary format to store/load a bunch of new
    // types of values while remaining backwards compatible.
    std::vector<std::string>& customValueNames,
    std::vector<std::vector<Eigen::MatrixXs>> customValues,
    // These are the markers, gyros and accelerometers
    const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
        markerObservations,
    const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
        accObservations,
    const std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>&
        gyroObservations,
    const std::vector<std::vector<std::map<std::string, Eigen::VectorXs>>>&
        emgObservations,
    // This is raw force plate data
    std::vector<std::vector<ForcePlate>>& forcePlates,
    // This is the subject info
    const std::string& biologicalSex,
    double heightM,
    double massKg,
    int ageYears,
    // The provenance info, optional, for investigating where training data
    // came from after its been aggregated
    std::vector<std::string> trialNames,
    std::vector<std::string> subjectTags,
    std::vector<std::vector<std::string>> trialTags,
    const std::string& sourceHref,
    const std::string& notes)
{
  // 0. Open the file
  FILE* file = fopen(outputPath.c_str(), "w");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk::writeSubject() failed" << std::endl;
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // 1. Serialize and write the header to the file
  /////////////////////////////////////////////////////////////////////////////

  // 1.1. Read the whole OpenSim file in as a string
  auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
  newRetriever->addSchemaRetriever(
      "file", std::make_shared<common::LocalResourceRetriever>());
  newRetriever->addSchemaRetriever(
      "dart", utils::DartResourceRetriever::create());
  const std::string openSimRawXML = newRetriever->readAll(openSimFilePath);

  // 1.2. Populate the protobuf header object in memory
  proto::SubjectOnDiskHeader header;
  header.set_num_dofs(trialPoses.size() > 0 ? trialPoses[0].rows() : 0);
  header.set_num_trials(trialPoses.size());
  for (std::string& body : groundForceBodies)
  {
    header.add_ground_contact_body(body);
  }
  for (int i = 0; i < customValueNames.size(); i++)
  {
    header.add_custom_value_name(customValueNames[i]);
    header.add_custom_value_length(
        customValues.size() > 0 ? customValues[0][i].rows() : 0);
  }
  header.set_model_osim_text(openSimRawXML);
  int maxNumForcePlates = 0;
  for (int trial = 0; trial < trialNames.size(); trial++)
  {
    auto* trialHeader = header.add_trial_header();
    if (probablyMissingGRF.size() > trial && missingGRFReason.size() > trial
        && trialResidualNorms.size() > trial)
    {
      for (int t = 0; t < probablyMissingGRF[trial].size(); t++)
      {
        trialHeader->add_missing_grf(probablyMissingGRF[trial][t]);
        if (missingGRFReason[trial].size() > t)
        {
          trialHeader->add_missing_grf_reason(
              missingGRFReasonToProto(missingGRFReason[trial][t]));
        }
        else
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "missingGRFReason out-of-bounds for trial "
                    << trial << " at time " << t
                    << ", defaulting to notMissingGRF" << std::endl;
          trialHeader->add_missing_grf_reason(
              missingGRFReasonToProto(notMissingGRF));
        }
        if (trialResidualNorms[trial].size() > t)
        {
          trialHeader->add_residual(trialResidualNorms[trial][t]);
        }
        else
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "residual out-of-bounds for trial "
                    << trial << " at time " << t << ", defaulting to 0"
                    << std::endl;
          trialHeader->add_residual(0);
        }
      }
    }
    else
    {
      std::cout
          << "SubjectOnDisk::writeSubject() passed bad info: "
             "probablyMissingGRF, missingGRFReason, or trialResidualNorms "
             "out-of-bounds for trial "
          << trial << std::endl;
    }
    trialHeader->set_trial_timestep(trialTimesteps[trial]);
    trialHeader->set_trial_length(trialPoses[trial].cols());
    for (int j = 0; j < dofPositionsObserved[trial].size(); j++)
    {
      trialHeader->add_dof_positions_observed(dofPositionsObserved[trial][j]);
      trialHeader->add_dof_velocities_finite_differenced(
          dofVelocitiesFiniteDifferenced[trial][j]);
      trialHeader->add_dof_acceleration_finite_differenced(
          dofAccelerationFiniteDifferenced[trial][j]);
    }
    if (trialTags.size() > trial)
    {
      for (std::string& tag : trialTags[trial])
      {
        trialHeader->add_trial_tag(tag);
      }
    }
    trialHeader->set_num_force_plates(forcePlates[trial].size());
    for (auto& trialPlates : forcePlates[trial])
    {
      for (Eigen::Vector3s& corner : trialPlates.corners)
      {
        for (int i = 0; i < 3; i++)
        {
          trialHeader->add_force_plate_corners(corner(i));
        }
      }
    }
    // Keep track of the largest number of force plates in any trial, because we
    // need to pad all trials to this number with NaNs, in order to keep the
    // frames the same size to support random disk seeks.
    if (forcePlates[trial].size() > maxNumForcePlates)
    {
      maxNumForcePlates = forcePlates[trial].size();
    }

    for (int t = 0; t < trialVels[trial].cols(); t++)
    {
      s_t maxVel = trialVels[trial].col(t).cwiseAbs().maxCoeff();
      trialHeader->add_joints_max_velocity(maxVel);
    }

    header.add_trial_name(trialNames[trial]);
  }
  header.set_href(sourceHref);
  header.set_notes(notes);
  header.set_version(2);

  header.set_biological_sex(biologicalSex);
  header.set_mass_kg(massKg);
  header.set_height_m(heightM);
  header.set_age_years(ageYears);
  for (std::string& tag : subjectTags)
  {
    header.add_subject_tag(tag);
  }

  std::vector<std::string> markerNames;
  std::vector<std::string> accNames;
  std::vector<std::string> gyroNames;
  std::vector<std::string> emgNames;
  for (int trial = 0; trial < markerObservations.size(); trial++)
  {
    for (int t = 0; t < markerObservations[trial].size(); t++)
    {
      for (auto& pair : markerObservations[trial][t])
      {
        if (std::find(markerNames.begin(), markerNames.end(), pair.first)
            == markerNames.end())
        {
          markerNames.push_back(pair.first);
        }
      }
    }
  }
  for (int trial = 0; trial < accObservations.size(); trial++)
  {
    for (int t = 0; t < accObservations[trial].size(); t++)
    {
      for (auto& pair : accObservations[trial][t])
      {
        if (std::find(accNames.begin(), accNames.end(), pair.first)
            == accNames.end())
        {
          accNames.push_back(pair.first);
        }
      }
    }
  }
  for (int trial = 0; trial < gyroObservations.size(); trial++)
  {
    for (int t = 0; t < gyroObservations[trial].size(); t++)
    {
      for (auto& pair : gyroObservations[trial][t])
      {
        if (std::find(gyroNames.begin(), gyroNames.end(), pair.first)
            == gyroNames.end())
        {
          gyroNames.push_back(pair.first);
        }
      }
    }
  }
  int emgDim = 0;
  for (int trial = 0; trial < emgObservations.size(); trial++)
  {
    for (int t = 0; t < emgObservations[trial].size(); t++)
    {
      for (auto& pair : emgObservations[trial][t])
      {
        if (std::find(emgNames.begin(), emgNames.end(), pair.first)
            == emgNames.end())
        {
          emgNames.push_back(pair.first);
        }
        if (emgDim == 0)
        {
          emgDim = pair.second.size();
        }
        else if (emgDim != pair.second.size())
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "emgObservations have inconsistent dimensions for trial "
                    << trial << " timestep " << t << " for emg " << pair.first
                    << ". Expected " << emgDim << " but got "
                    << pair.second.size() << std::endl;
        }
      }
    }
  }

  for (std::string& name : markerNames)
  {
    header.add_marker_name(name);
  }
  for (std::string& name : accNames)
  {
    header.add_acc_name(name);
  }
  for (std::string& name : gyroNames)
  {
    header.add_gyro_name(name);
  }
  for (std::string& name : emgNames)
  {
    header.add_emg_name(name);
  }
  header.set_emg_dim(emgDim);

  // 1.3. Continues in next section, after we know the size of the first
  // serialized frame...

  /////////////////////////////////////////////////////////////////////////////
  // 2. Serialize and write the frames to the file
  /////////////////////////////////////////////////////////////////////////////

  bool firstTrial = true;
  int expectedFrameSize = -1;
  (void)expectedFrameSize;
  for (int trial = 0; trial < trialPoses.size(); trial++)
  {
    for (int t = 0; t < trialPoses[trial].cols(); t++)
    {
      // 2.1. Populate the protobuf frame object in memory
      proto::SubjectOnDiskFrame frame;
      if (trialPoses.size() > trial && trialPoses[trial].cols() > t
          && trialVels.size() > trial && trialVels[trial].cols() > t
          && trialAccs.size() > trial && trialAccs[trial].cols() > t)
      {
        for (int i = 0; i < trialPoses[trial].rows(); i++)
        {
          frame.add_pos(trialPoses[trial](i, t));
          frame.add_vel(trialVels[trial](i, t));
          frame.add_acc(trialAccs[trial](i, t));
          frame.add_tau(trialTaus[trial](i, t));
        }
      }
      else
      {
        std::cout
            << "SubjectOnDisk::writeSubject() passed bad info: trialPoses, "
               "trialVels, or trialAccs out-of-bounds for trial "
            << trial << " frame " << t << std::endl;
      }

      if (trialGroundBodyWrenches.size() > trial
          && trialGroundBodyWrenches[trial].cols() > t
          && trialGroundBodyWrenches[trial].rows()
                 == 6 * groundForceBodies.size()
          && trialGroundBodyCopTorqueForce.size() > trial
          && trialGroundBodyCopTorqueForce[trial].cols() > t
          && trialGroundBodyCopTorqueForce[trial].rows()
                 == 9 * groundForceBodies.size())
      {
        for (int i = 0; i < groundForceBodies.size(); i++)
        {
          for (int j = 0; j < 6; j++)
          {
            frame.add_ground_contact_wrench(
                trialGroundBodyWrenches[trial](i * 6 + j, t));
          }
          for (int j = 0; j < 3; j++)
          {
            frame.add_ground_contact_center_of_pressure(
                trialGroundBodyCopTorqueForce[trial](i * 9 + j, t));
            frame.add_ground_contact_torque(
                trialGroundBodyCopTorqueForce[trial](i * 9 + 3 + j, t));
            frame.add_ground_contact_force(
                trialGroundBodyCopTorqueForce[trial](i * 9 + 6 + j, t));
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
      for (int i = 0; i < 3; i++)
      {
        if (trialComPoses.size() < trial || trialComPoses[trial].cols() < t)
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "trialComPoses out-of-bounds for trial "
                    << trial << std::endl;
          frame.add_com_pos(0);
        }
        else
        {
          frame.add_com_pos(trialComPoses[trial](i, t));
        }
        if (trialComVels.size() < trial || trialComVels[trial].cols() < t)
        {
          std::cout << "SubjectOnDisk::writeSubject() passed bad info: "
                       "trialComVels out-of-bounds for trial "
                    << trial << std::endl;
          frame.add_com_vel(0);
        }
        else
        {
          frame.add_com_vel(trialComVels[trial](i, t));
        }
        if (trialComAccs.size() < trial || trialComAccs[trial].cols() < t)
        {
          frame.add_com_acc(0);
        }
        else
        {
          frame.add_com_acc(trialComAccs[trial](i, t));
        }
      }
      if (customValues.size() > trial)
      {
        for (int i = 0; i < customValues[trial].size(); i++)
        {
          for (int j = 0; j < customValues[trial][i].rows(); j++)
          {
            frame.add_custom_values(customValues[trial][i](j, t));
          }
        }
      }
      if (markerObservations.size() > trial
          && markerObservations[trial].size() > t)
      {
        for (std::string& name : markerNames)
        {
          if (markerObservations[trial][t].find(name)
              != markerObservations[trial][t].end())
          {
            frame.add_marker_obs(markerObservations[trial][t].at(name)(0));
            frame.add_marker_obs(markerObservations[trial][t].at(name)(1));
            frame.add_marker_obs(markerObservations[trial][t].at(name)(2));
          }
          else
          {
            frame.add_marker_obs(nan(""));
            frame.add_marker_obs(nan(""));
            frame.add_marker_obs(nan(""));
          }
        }
      }
      else
      {
        for (int i = 0; i < markerNames.size(); i++)
        {
          frame.add_marker_obs(nan(""));
          frame.add_marker_obs(nan(""));
          frame.add_marker_obs(nan(""));
        }
      }
      if (accObservations.size() > trial && accObservations[trial].size() > t)
      {
        for (std::string& name : accNames)
        {
          if (accObservations[trial][t].find(name)
              != accObservations[trial][t].end())
          {
            frame.add_acc_obs(accObservations[trial][t].at(name)(0));
            frame.add_acc_obs(accObservations[trial][t].at(name)(1));
            frame.add_acc_obs(accObservations[trial][t].at(name)(2));
          }
          else
          {
            frame.add_acc_obs(nan(""));
            frame.add_acc_obs(nan(""));
            frame.add_acc_obs(nan(""));
          }
        }
      }
      else
      {
        for (int i = 0; i < accNames.size(); i++)
        {
          frame.add_acc_obs(nan(""));
          frame.add_acc_obs(nan(""));
          frame.add_acc_obs(nan(""));
        }
      }
      if (gyroObservations.size() > trial && gyroObservations[trial].size() > t)
      {
        for (std::string& name : gyroNames)
        {
          if (gyroObservations[trial][t].find(name)
              != gyroObservations[trial][t].end())
          {
            frame.add_gyro_obs(gyroObservations[trial][t].at(name)(0));
            frame.add_gyro_obs(gyroObservations[trial][t].at(name)(1));
            frame.add_gyro_obs(gyroObservations[trial][t].at(name)(2));
          }
          else
          {
            frame.add_gyro_obs(nan(""));
            frame.add_gyro_obs(nan(""));
            frame.add_gyro_obs(nan(""));
          }
        }
      }
      else
      {
        for (int i = 0; i < gyroNames.size(); i++)
        {
          frame.add_gyro_obs(nan(""));
          frame.add_gyro_obs(nan(""));
          frame.add_gyro_obs(nan(""));
        }
      }
      if (emgObservations.size() > trial && emgObservations[trial].size() > t)
      {
        for (std::string& name : emgNames)
        {
          if (emgObservations[trial][t].find(name)
              != emgObservations[trial][t].end())
          {
            for (int i = 0; i < emgDim; i++)
            {
              frame.add_emg_obs(emgObservations[trial][t].at(name)(i));
            }
          }
          else
          {
            for (int i = 0; i < emgDim; i++)
            {
              frame.add_emg_obs(nan(""));
            }
          }
        }
      }
      else
      {
        for (int i = 0; i < emgNames.size(); i++)
        {
          for (int j = 0; j < emgDim; j++)
          {
            frame.add_emg_obs(nan(""));
          }
        }
      }

      for (int forcePlateIdx = 0; forcePlateIdx < maxNumForcePlates;
           forcePlateIdx++)
      {
        if (trial < forcePlates.size()
            && forcePlateIdx < forcePlates[trial].size()
            && forcePlates[trial][forcePlateIdx].centersOfPressure.size() > t
            && forcePlates[trial][forcePlateIdx].forces.size() > t
            && forcePlates[trial][forcePlateIdx].moments.size() > t)
        {
          for (int i = 0; i < 3; i++)
          {
            frame.add_raw_force_plate_cop(
                forcePlates[trial][forcePlateIdx].centersOfPressure[t](i));
            frame.add_raw_force_plate_torque(
                forcePlates[trial][forcePlateIdx].moments[t](i));
            frame.add_raw_force_plate_force(
                forcePlates[trial][forcePlateIdx].forces[t](i));
          }
        }
        else
        {
          for (int i = 0; i < 3; i++)
          {
            frame.add_raw_force_plate_cop(std::nan(""));
            frame.add_raw_force_plate_torque(std::nan(""));
            frame.add_raw_force_plate_force(std::nan(""));
          }
        }
      }

      // 2.2. Serialize the protobuf header object
      std::string frameSerialized = "";
      frame.SerializeToString(&frameSerialized);

      // 2.3. Get the length of the serialized message
      int64_t messageSize = frameSerialized.size();

      // If this is the first trial, we need to finish the header with
      // information about the size of serialized frame objects and write the
      // header first
      if (firstTrial)
      {
        // 1.3. Continued from previous section... Write the size of the frame
        // binaries
        expectedFrameSize = messageSize;
        header.set_frame_size(messageSize);

        // 1.4. Serialize the protobuf header object
        std::string headerSerialized = "";
        header.SerializeToString(&headerSerialized);

        // 1.5. Write the length of the message as an integer header
        int64_t headerSize = headerSerialized.size();
        fwrite(&headerSize, sizeof(int64_t), 1, file);

        // 1.6. Write the serialized data to the file
        fwrite(headerSerialized.c_str(), sizeof(char), headerSize, file);

        firstTrial = false;
      }
      // We need all frames to be exactly the same size, in order to support
      // random seeking in the file
      assert(messageSize == expectedFrameSize);

      // 2.4. Write the serialized data to the file
      fwrite(frameSerialized.c_str(), sizeof(char), messageSize, file);
    }
  }

  fclose(file);
}

/// This returns the number of trials on the subject
int SubjectOnDisk::getNumTrials()
{
  return mNumTrials;
}

/// This returns the length of the trial
int SubjectOnDisk::getTrialLength(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return 0;
  }
  return mTrialLength[trial];
}

/// This returns the timestep size for the trial
s_t SubjectOnDisk::getTrialTimestep(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return 0.01;
  }
  return mTrialTimesteps[trial];
}

/// This returns the number of DOFs for the model on this Subject
int SubjectOnDisk::getNumDofs()
{
  return mNumDofs;
}

/// This returns the vector of booleans for whether or not each timestep is
/// heuristically detected to be missing external forces (which means that the
/// inverse dynamics cannot be trusted).
std::vector<bool> SubjectOnDisk::getProbablyMissingGRF(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<bool>();
  }
  return mProbablyMissingGRF[trial];
}

/// This returns the vector of enums of type 'MissingGRFReason', which labels
/// why each time step was identified as 'probablyMissingGRF'.
std::vector<MissingGRFReason> SubjectOnDisk::getMissingGRFReason(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<MissingGRFReason>();
  }
  return mMissingGRFReason[trial];
}

std::vector<bool> SubjectOnDisk::getDofPositionsObserved(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<bool>();
  }
  return mDofPositionsObserved[trial];
}

std::vector<bool> SubjectOnDisk::getDofVelocitiesFiniteDifferenced(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<bool>();
  }
  return mDofVelocitiesFiniteDifferenced[trial];
}

std::vector<bool> SubjectOnDisk::getDofAccelerationsFiniteDifferenced(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<bool>();
  }
  return mDofAccelerationFiniteDifferenced[trial];
}

std::vector<s_t> SubjectOnDisk::getTrialResidualNorms(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<s_t>();
  }
  return mTrialResidualNorms[trial];
}

/// This returns the maximum absolute velocity of any DOF at each timestep for a
/// given trial
std::vector<s_t> SubjectOnDisk::getTrialMaxJointVelocity(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<s_t>();
  }
  return mTrialMaxJointVelocity[trial];
}

/// This returns the list of contact body names for this Subject
std::vector<std::string> SubjectOnDisk::getGroundContactBodies()
{
  return mGroundContactBodies;
}

/// This returns the list of custom value names stored in this subject
std::vector<std::string> SubjectOnDisk::getCustomValues()
{
  return mCustomValues;
}

/// This returns the dimension of the custom value specified by `valueName`
int SubjectOnDisk::getCustomValueDim(std::string valueName)
{
  for (int i = 0; i < mCustomValues.size(); i++)
  {
    if (mCustomValues[i] == valueName)
    {
      return mCustomValueLengths[i];
    }
  }
  std::cout << "WARNING: Requested getCustomValueDim() for value \""
            << valueName
            << "\", which is not in this SubjectOnDisk. Options are: [";
  for (int i = 0; i < mCustomValues.size(); i++)
  {
    std::cout << " \"" << mCustomValues[i] << "\" ";
  }
  std::cout << "]. Returning 0." << std::endl;
  return 0;
}

/// The name of the trial, if provided, or else an empty string
std::string SubjectOnDisk::getTrialName(int trial)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return "";
  }
  return mTrialNames[trial];
}

std::string SubjectOnDisk::getBiologicalSex()
{
  return mBiologicalSex;
}

double SubjectOnDisk::getHeightM()
{
  return mHeightM;
}

double SubjectOnDisk::getMassKg()
{
  return mMassKg;
}

/// This gets the tags associated with the subject, if there are any.
std::vector<std::string> SubjectOnDisk::getSubjectTags()
{
  return mSubjectTags;
}

/// This gets the tags associated with the trial, if there are any.
std::vector<std::string> SubjectOnDisk::getTrialTags(int trial)
{
  if (trial < mTrialTags.size())
  {
    return mTrialTags[trial];
  }
  else
  {
    return std::vector<std::string>();
  }
}

int SubjectOnDisk::getAgeYears()
{
  return mAgeYears;
}

/// This returns the number of raw force plates that were used to generate the
/// data
int SubjectOnDisk::getNumForcePlates(int trial)
{
  return mTrialNumForcePlates[trial];
}

/// This returns the corners (in 3D space) of the selected force plate, for
/// this trial. Empty arrays on out of bounds.
std::vector<Eigen::Vector3s> SubjectOnDisk::getForcePlateCorners(
    int trial, int forcePlate)
{
  if (trial < 0 || trial >= mNumTrials)
  {
    return std::vector<Eigen::Vector3s>();
  }
  if (forcePlate < 0 || forcePlate >= mTrialNumForcePlates[trial])
  {
    return std::vector<Eigen::Vector3s>();
  }
  return mTrialForcePlateCorners[trial][forcePlate];
}

/// This gets the href link associated with the subject, if there is one.
std::string SubjectOnDisk::getHref()
{
  return mHref;
}

/// This gets the notes associated with the subject, if there are any.
std::string SubjectOnDisk::getNotes()
{
  return mNotes;
}

} // namespace biomechanics
} // namespace dart