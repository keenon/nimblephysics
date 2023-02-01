#include "dart/biomechanics/SubjectOnDisk.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <stdio.h>
#include <tinyxml2.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/math/MathTypes.hpp"
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

SubjectOnDisk::SubjectOnDisk(
    const std::string& path, bool printDebuggingDetails)
  : mPath(path)
{
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk attempting to open file that deos not exist: "
              << path << std::endl;
    throw new std::exception();
  }

  struct FileHeader header;
  int read_items = fread(&header, sizeof(struct FileHeader), 1, file);
  if (header.magic != 424242 || read_items != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path << ": bad header.magic = " << header.magic << std::endl;
    throw new std::exception();
  }
  if (header.version != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a binary file with "
                 "unsupported version "
              << header.version << " (currently only support version 1)"
              << std::endl;
    throw new std::exception();
  }

  mNumDofs = header.numDofs;
  mNumTrials = header.numTrials;

  // Read the href
  int32_t hrefLen;
  read_items = fread(&hrefLen, sizeof(int32_t), 1, file);
  if (hrefLen < 0 || hrefLen > 2000 || read_items != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path << ": bad string len for href = " << hrefLen << std::endl;
    throw new std::exception();
  }
  char* hrefRaw = (char*)malloc(sizeof(char) * (hrefLen + 1));
  read_items = fread(hrefRaw, sizeof(char), hrefLen, file);
  (void)read_items;
  hrefRaw[hrefLen] = 0;
  mHref = std::string(hrefRaw);
  free(hrefRaw);

  // Read the notes
  int32_t notesLen;
  read_items = fread(&notesLen, sizeof(int32_t), 1, file);
  if (notesLen < 0 || notesLen > 100000 || read_items != 1)
  {
    std::cout << "SubjectOnDisk attempting to read a corrupted binary file at "
              << path << ": bad string len for notes = " << notesLen
              << std::endl;
    throw new std::exception();
  }
  char* notesRaw = (char*)malloc(sizeof(char) * (notesLen + 1));
  read_items = fread(notesRaw, sizeof(char), notesLen, file);
  (void)read_items;
  notesRaw[notesLen] = 0;
  mNotes = std::string(notesRaw);
  free(notesRaw);

  // Read trial names
  for (int i = 0; i < mNumTrials; i++)
  {
    int32_t len;
    read_items = fread(&len, sizeof(int32_t), 1, file);
    if (len < 0 || len > 255 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad string len for trial [" << i << "] name = " << len
          << std::endl;
      throw new std::exception();
    }
    char* rawTrialName = (char*)malloc(sizeof(char) * (len + 1));
    read_items = fread(rawTrialName, sizeof(char), len, file);
    if (read_items != len) {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad string len for trial [" << i << "] name = " << len
          << ". Only read " << read_items << "/" << len << " chars" << std::endl;
      throw new std::exception();
    }
    rawTrialName[len] = 0;
    std::string trialName(rawTrialName);
    mTrialNames.push_back(trialName);
    if (printDebuggingDetails)
    {
      std::cout << "Read trial name: " << trialName << std::endl;
    }
    free(rawTrialName);
  }

  // Read contact body names
  int32_t numContactBodies = header.numGroundContactBodies;
  for (int i = 0; i < numContactBodies; i++)
  {
    int32_t len;
    read_items = fread(&len, sizeof(int32_t), 1, file);
    if (len < 0 || len > 255 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad string len for contact body [" << i
          << "] name = " << len << std::endl;
      throw new std::exception();
    }
    char* body = (char*)malloc(sizeof(char) * (len + 1));
    read_items = fread(body, sizeof(char), len, file);
    (void)read_items;
    body[len] = 0;
    std::string bodyName(body);
    mGroundContactBodies.push_back(bodyName);
    if (printDebuggingDetails)
    {
      std::cout << "Read contact body name: " << bodyName << std::endl;
    }
    free(body);
  }

  // Read custom value names and sizes
  int32_t numCustomValues = header.numCustomValues;
  int customValuesTotalDim = 0;
  for (int i = 0; i < numCustomValues; i++)
  {
    int32_t dataLen;
    read_items = fread(&dataLen, sizeof(int32_t), 1, file);
    if (dataLen < 0 || dataLen > 100000 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad data length for custom value [" << i
          << "] name = " << dataLen << std::endl;
      throw new std::exception();
    }
    customValuesTotalDim += dataLen;
    int32_t strLen;
    read_items = fread(&strLen, sizeof(int32_t), 1, file);
    if (strLen < 0 || strLen > 512 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad string length for custom value [" << i
          << "] name = " << strLen << std::endl;
      throw new std::exception();
    }
    char* custom = (char*)malloc(sizeof(char) * (strLen + 1));
    read_items = fread(custom, sizeof(char), strLen, file);
    (void)read_items;
    custom[strLen] = 0;
    std::string customValue(custom);
    mCustomValues.push_back(customValue);
    mCustomValueLengths.push_back(dataLen);
    if (printDebuggingDetails)
    {
      std::cout << "Read custom value: " << customValue
                << " with length = " << dataLen << std::endl;
    }
    free(custom);
  }

  // Read trial lengths
  for (int i = 0; i < mNumTrials; i++)
  {
    int32_t len;
    read_items = fread(&len, sizeof(int32_t), 1, file);
    if (len < 0 || len > 10000000 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": bad trial len for trial [" << i << "] = " << len
          << std::endl;
      throw new std::exception();
    }
    if (printDebuggingDetails)
    {
      std::cout << "Read trial " << i << " len = " << len << std::endl;
    }
    mTrialLength.push_back((int)len);
  }

  // Read trial timesteps
  for (int i = 0; i < mNumTrials; i++)
  {
    double timestep;
    read_items = fread(&timestep, sizeof(double), 1, file);
    if (read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": trial timesteps suddenly reached EOF" << std::endl;
      throw new std::exception();
    }
    (void)read_items;
    if (printDebuggingDetails)
    {
      std::cout << "Read trial " << i << " timestep = " << timestep
                << std::endl;
    }
    mTrialTimesteps.push_back(timestep);
  }

  // Read the `probablyMissingGrf` data
  for (int i = 0; i < mNumTrials; i++)
  {
    // Read a magic header, to make sure we don't get lost
    int32_t magic;
    read_items = fread(&magic, sizeof(int32_t), 1, file);
    if (magic != 424242 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << path << ": before the probablyMissingGRF array for trial " << i
          << ", got bad magic = " << magic << std::endl;
      throw new std::exception();
    }

    std::vector<bool> probablyMissingGRF;
    for (int t = 0; t < mTrialLength[i]; t++)
    {
      int8_t b;
      read_items = fread(&b, sizeof(int8_t), 1, file);
      if (read_items != 1)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << path << ": probablyMissingGRF section suddenly reached EOF" << std::endl;
        throw new std::exception();
      }
      probablyMissingGRF.push_back(b);
    }
    mProbablyMissingGRF.push_back(probablyMissingGRF);
  }

  int32_t modelLen;
  read_items = fread(&modelLen, sizeof(int32_t), 1, file);
  if (read_items != 1)
  {
    std::cout
        << "SubjectOnDisk attempting to read a corrupted binary file at "
        << path << ": model length section suddenly reached EOF" << std::endl;
    throw new std::exception();
  }
  mModelLength = modelLen;
  mModelSectionStart = ftell(file);
  mDataSectionStart = mModelSectionStart + modelLen;

  // Magic number, pos, vel, acc, tau, contact wrenches, and custom values
  mFrameSize = sizeof(int32_t)
               + (((mNumDofs * 4) + (mGroundContactBodies.size() * 6)
                   + (mGroundContactBodies.size() * 9) + customValuesTotalDim)
                  * sizeof(float64_t));

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

  FILE* file = fopen(mPath.c_str(), "r");

  fseek(file, mModelSectionStart, SEEK_SET);
  char* rawOsimContents = (char*)malloc(sizeof(char) * (mModelLength + 1));
  int read_items = fread(rawOsimContents, sizeof(char), mModelLength, file);
  if (read_items != mModelLength)
  {
    std::cout
        << "SubjectOnDisk attempting to read a corrupted binary file at "
        << mPath << ": model file text contained unexpected EOF" << std::endl;
    throw new std::exception();
  }
  rawOsimContents[mModelLength - 1] = 0;
  tinyxml2::XMLDocument osimFile;
  osimFile.Parse(rawOsimContents);
  OpenSimFile osimParsed
      = OpenSimParser::parseOsim(osimFile, mPath, geometryFolder);

  free(rawOsimContents);

  fclose(file);

  return osimParsed.skeleton;
}

/// This will read from disk and allocate a number of Frame objects,
/// optionally sharing the same Skeleton pointer for efficiency if
/// `shareSkeletonPtr` is true, (though that means it won't be threadsafe to
/// use the Frame objects in parallel). These Frame objects are assumed to be
/// short-lived, to save working memory.
///
/// On OOB access, prints an error and returns an empty vector.
std::vector<std::shared_ptr<Frame>> SubjectOnDisk::readFrames(
    int trial, int startFrame, int numFramesToRead)
{
  (void)trial;
  (void)startFrame;
  (void)numFramesToRead;

  std::vector<std::shared_ptr<Frame>> result;

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

  fseek(file, mDataSectionStart + (mFrameSize * linearFrameStart), SEEK_SET);

  int read_items;

  for (int i = 0; i < numFramesToRead; i++)
  {
    int32_t magic;
    read_items = fread(&magic, sizeof(int32_t), 1, file);
    if (magic != 424242 || read_items != 1)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": before frame " << linearFrameStart + i << " (trial "
          << trial << " frame " << startFrame + i
          << "), got bad magic = " << magic << std::endl;
      throw new std::exception();
    }

    result.push_back(std::make_shared<Frame>());
    std::shared_ptr<Frame>& frame = result.at(result.size() - 1);
    frame->trial = trial;
    frame->t = startFrame + i;
    frame->dt = mTrialTimesteps[trial];
    frame->probablyMissingGRF = mProbablyMissingGRF[trial][frame->t];
    frame->pos = Eigen::VectorXd(mNumDofs);
    frame->vel = Eigen::VectorXd(mNumDofs);
    frame->acc = Eigen::VectorXd(mNumDofs);
    frame->tau = Eigen::VectorXd(mNumDofs);
    read_items = fread(frame->pos.data(), sizeof(float64_t), mNumDofs, file);
    if (read_items != mNumDofs)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
      throw new std::exception();
    }
    read_items = fread(frame->vel.data(), sizeof(float64_t), mNumDofs, file);
    if (read_items != mNumDofs)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
      throw new std::exception();
    }
    read_items = fread(frame->acc.data(), sizeof(float64_t), mNumDofs, file);
    if (read_items != mNumDofs)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
      throw new std::exception();
    }
    read_items = fread(frame->tau.data(), sizeof(float64_t), mNumDofs, file);
    if (read_items != mNumDofs)
    {
      std::cout
          << "SubjectOnDisk attempting to read a corrupted binary file at "
          << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
      throw new std::exception();
    }

    for (int b = 0; b < mGroundContactBodies.size(); b++)
    {
      frame->groundContactWrenches.emplace_back(
          mGroundContactBodies[b], Eigen::Vector6d());
      read_items = fread(
          frame->groundContactWrenches[b].second.data(),
          sizeof(float64_t),
          6,
          file);
      if (read_items != 6)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
        throw new std::exception();
      }
    }
    for (int b = 0; b < mGroundContactBodies.size(); b++)
    {
      frame->groundContactCenterOfPressure.emplace_back(
          mGroundContactBodies[b], Eigen::Vector3d());
      read_items = fread(
          frame->groundContactCenterOfPressure[b].second.data(),
          sizeof(float64_t),
          3,
          file);
      if (read_items != 3)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
        throw new std::exception();
      }
      frame->groundContactTorque.emplace_back(
          mGroundContactBodies[b], Eigen::Vector3d());
      read_items = fread(
          frame->groundContactTorque[b].second.data(),
          sizeof(float64_t),
          3,
          file);
      if (read_items != 3)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
        throw new std::exception();
      }
      frame->groundContactForce.emplace_back(
          mGroundContactBodies[b], Eigen::Vector3d());
      read_items = fread(
          frame->groundContactForce[b].second.data(),
          sizeof(float64_t),
          3,
          file);
      if (read_items != 3)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
        throw new std::exception();
      }
    }
    for (int b = 0; b < mCustomValues.size(); b++)
    {
      int len = mCustomValueLengths[b];
      frame->customValues.emplace_back(mCustomValues[b], Eigen::VectorXd(len));
      read_items = fread(frame->customValues[b].second.data(), sizeof(float64_t), len, file);
      if (read_items != len)
      {
        std::cout
            << "SubjectOnDisk attempting to read a corrupted binary file at "
            << mPath << ": frame data for frame " << (startFrame + i) << " had unexpected EOF" << std::endl;
        throw new std::exception();
      }
    }
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
    std::vector<Eigen::MatrixXs>& trialTaus,
    // These are generalized 6-dof wrenches applied to arbitrary bodies
    // (generally by foot-ground contact, though other things too)
    std::vector<std::string>& groundForceBodies,
    std::vector<Eigen::MatrixXs>& trialGroundBodyWrenches,
    std::vector<Eigen::MatrixXs>& trialGroundBodyCopTorqueForce,
    // We include this to allow the binary format to store/load a bunch of new
    // types of values while remaining backwards compatible.
    std::vector<std::string>& customValueNames,
    std::vector<std::vector<Eigen::MatrixXs>> customValues,
    // The provenance info, optional, for investigating where training data
    // came from after its been aggregated
    std::vector<std::string> trialNames,
    const std::string& sourceHref,
    const std::string& notes)
{
  (void)outputPath;
  (void)openSimFilePath;
  (void)trialPoses;
  (void)trialVels;
  (void)trialAccs;
  (void)trialTaus;
  (void)groundForceBodies;
  (void)trialGroundBodyWrenches;
  (void)trialGroundBodyCopTorqueForce;
  (void)trialTimesteps;
  (void)probablyMissingGRF;
  (void)customValueNames;
  (void)customValues;

  // Read the whole OpenSim file in as a string
  auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
  newRetriever->addSchemaRetriever(
      "file", std::make_shared<common::LocalResourceRetriever>());
  newRetriever->addSchemaRetriever(
      "dart", utils::DartResourceRetriever::create());
  const std::string openSimRawXML = newRetriever->readAll(openSimFilePath);

  FILE* file = fopen(outputPath.c_str(), "w");
  if (file == nullptr)
  {
    std::cout << "SubjectOnDisk::writeSubject() failed" << std::endl;
    return;
  }

  // Write the header
  struct FileHeader header;
  header.magic = 424242;
  header.version = 1;
  header.numDofs = trialPoses[0].rows();
  header.numTrials = trialPoses.size();
  header.numGroundContactBodies = groundForceBodies.size();
  header.numCustomValues = customValueNames.size();
  fwrite(&header, sizeof(struct FileHeader), 1, file);

  // Write the href
  int32_t hrefLen = sourceHref.length();
  fwrite(&hrefLen, sizeof(int32_t), 1, file);
  fwrite(sourceHref.c_str(), sizeof(char), hrefLen, file);

  // Write the notes
  int32_t notesLen = notes.length();
  fwrite(&notesLen, sizeof(int32_t), 1, file);
  fwrite(notes.c_str(), sizeof(char), notesLen, file);

  // Write trial names
  for (int i = 0; i < header.numTrials; i++)
  {
    std::string trialName = "";
    if (trialNames.size() > i)
    {
      trialName = trialNames[i];
    }
    int32_t len = trialName.length();
    fwrite(&len, sizeof(int32_t), 1, file);
    fwrite(trialName.c_str(), sizeof(char), len, file);
  }

  // Write contact body names
  for (int i = 0; i < groundForceBodies.size(); i++)
  {
    int32_t len = groundForceBodies[i].length();
    fwrite(&len, sizeof(int32_t), 1, file);
    fwrite(groundForceBodies[i].c_str(), sizeof(char), len, file);
  }

  // Write custom value names and lengths
  for (int i = 0; i < customValueNames.size(); i++)
  {
    int32_t dataLen = customValues[0][i].rows();
    int32_t strLen = customValueNames[i].size();
    fwrite(&dataLen, sizeof(int32_t), 1, file);
    fwrite(&strLen, sizeof(int32_t), 1, file);
    fwrite(customValueNames[i].c_str(), sizeof(char), strLen, file);
  }

  // Write trial lengths
  for (int i = 0; i < trialPoses.size(); i++)
  {
    int32_t len = trialPoses[i].cols();
    fwrite(&len, sizeof(int32_t), 1, file);
  }

  // Write trial timesteps
  for (int i = 0; i < trialPoses.size(); i++)
  {
    double timestep = trialTimesteps[i];
    fwrite(&timestep, sizeof(double), 1, file);
  }

  // Write the `probablyMissingGrf` data
  for (int i = 0; i < trialPoses.size(); i++)
  {
    // Write a magic header, to make sure we don't get lost
    int32_t magic = 424242;
    fwrite(&magic, sizeof(int32_t), 1, file);

    for (int t = 0; t < probablyMissingGRF[i].size(); t++)
    {
      int8_t b = probablyMissingGRF[i][t];
      fwrite(&b, sizeof(int8_t), 1, file);
    }
  }

  // Embed the model file XML directly in the binary
  int32_t modelLen = openSimRawXML.size();
  fwrite(&modelLen, sizeof(int32_t), 1, file);
  fwrite(openSimRawXML.c_str(), sizeof(char), modelLen, file);

  const int dofs = trialPoses[0].rows();

  // Write out all the frames
  for (int trial = 0; trial < trialPoses.size(); trial++)
  {
    for (int t = 0; t < trialPoses[trial].cols(); t++)
    {
      // Always begin each from with a magic number, to allow error checking on
      // retrieval
      int32_t magic = 424242;
      fwrite(&magic, sizeof(int32_t), 1, file);

      // Then write, in order: pos, vel, acc, tau, external wrenches, and
      // finally custom values

      // pos
      fwrite(
          trialPoses[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          dofs,
          file);
      // vel
      fwrite(
          trialVels[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          dofs,
          file);
      // acc
      fwrite(
          trialAccs[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          dofs,
          file);
      // tau
      fwrite(
          trialTaus[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          dofs,
          file);
      // external wrenches
      assert(
          trialGroundBodyWrenches[trial].rows()
          == groundForceBodies.size() * 6);
      fwrite(
          trialGroundBodyWrenches[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          trialGroundBodyWrenches[trial].rows(),
          file);
      // GRF data is COP-Torque-Force
      assert(
          trialGroundBodyCopTorqueForce[trial].rows()
          == groundForceBodies.size() * 9);
      fwrite(
          trialGroundBodyCopTorqueForce[trial].col(t).cast<float64_t>().data(),
          sizeof(float64_t),
          trialGroundBodyCopTorqueForce[trial].rows(),
          file);
      // custom values
      for (int i = 0; i < customValueNames.size(); i++)
      {
        fwrite(
            customValues[trial][i].col(t).cast<float64_t>().data(),
            sizeof(float64_t),
            customValues[trial][i].rows(),
            file);
      }
    }
  }

  // size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements,
  // FILE *a_file); size_t fwrite(const void *ptr, size_t size_of_elements,
  // size_t number_of_elements, FILE *a_file);
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