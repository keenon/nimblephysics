#ifndef DART_UTILS_OSIMPARSER_HPP_
#define DART_UTILS_OSIMPARSER_HPP_

#include <map>
#include <memory>
#include <string>

#include <tinyxml2.h>

#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/XmlHelpers.hpp"

namespace dart {
using namespace utils;

namespace biomechanics {

struct OpenSimFile
{
  dynamics::SkeletonPtr skeleton;
  // Markers map
  dynamics::MarkerMap markersMap;
  std::vector<std::string> anatomicalMarkers;
  std::vector<std::string> trackingMarkers;

  // Body scale map
  std::map<std::string, Eigen::Vector3s> bodyScales;

  // IMU map
  std::map<std::string, std::pair<std::string, Eigen::Isometry3s>> imuMap;

  std::vector<std::string> warnings;
  std::vector<std::string> ignoredBodies;
  std::vector<std::pair<std::string, std::string>> jointsDrivenBy;

  OpenSimFile();
  OpenSimFile(dynamics::SkeletonPtr skeleton, dynamics::MarkerMap markersMap);
};

struct OpenSimScaleAndMarkerOffsets
{
  bool success;
  Eigen::VectorXs bodyScales;
  dynamics::MarkerMap markers;
  std::map<std::string, Eigen::Vector3s> markerOffsets;
};

/// This holds marker trajectory information from an OpenSim TRC file
struct OpenSimTRC
{
  std::vector<double> timestamps;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
  std::map<std::string, std::vector<Eigen::Vector3s>> markerLines;
  int framesPerSecond;
};

struct OpenSimMot
{
  std::vector<double> timestamps;
  Eigen::MatrixXs poses;
};

struct OpenSimGRF
{
  std::vector<double> timestamps;
  std::vector<Eigen::Matrix<s_t, 3, Eigen::Dynamic>> plateCOPs;
  std::vector<Eigen::Matrix<s_t, 6, Eigen::Dynamic>> plateGRFs;
};

struct OpenSimIMUData
{
  std::vector<double> timestamps;
  std::vector<std::map<std::string, Eigen::Vector3s>> gyroReadings;
  std::vector<std::map<std::string, Eigen::Vector3s>> accReadings;
};

struct OpenSimMocoTrajectory
{
  std::vector<double> timestamps;
  Eigen::MatrixXs excitations;
  Eigen::MatrixXs activations;
  std::vector<std::string> excitationNames;
  std::vector<std::string> activationNames;
};

class OpenSimParser
{
public:
  /// Read Skeleton from *.osim file
  static OpenSimFile parseOsim(
      const common::Uri& uri,
      const std::string geometryFolder = "",
      bool ignoreGeometry = false,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Read Skeleton from *.osim file
  static OpenSimFile parseOsim(
      tinyxml2::XMLDocument& osimFile,
      const std::string fileNameForErrorDisplay = "",
      const std::string geometryFolder = "",
      bool ignoreGeometry = false,
      const common::ResourceRetrieverPtr& geometryRetriever = nullptr);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// scaling tool to rescale a skeleton
  static void saveOsimScalingXMLFile(
      const std::string& subjectName,
      std::shared_ptr<dynamics::Skeleton> skel,
      double massKg,
      double heightM,
      const std::string& osimInputPath,
      const std::string& osimInputMarkersPath,
      const std::string& osimOutputPath,
      const std::string& scalingInstructionsOutputPath);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// IK tool to recreate / validate the results of IK created from this tool
  static void saveOsimInverseKinematicsXMLFile(
      const std::string& subjectName,
      std::vector<std::string> markerNames,
      const std::string& osimInputModelPath,
      const std::string& osimInputTrcPath,
      const std::string& osimOutputMotPath,
      const std::string& ikInstructionsOutputPath);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// ID tool to recreate / validate the results of ID created from this tool
  static void saveOsimInverseDynamicsRawForcesXMLFile(
      const std::string& subjectName,
      std::shared_ptr<dynamics::Skeleton> skel,
      const Eigen::MatrixXs& poses,
      const std::vector<biomechanics::ForcePlate> forcePlates,
      const std::string& grfForcesPath,
      const std::string& forcesOutputPath);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// ID tool to recreate / validate the results of ID created from this tool
  static void saveOsimInverseDynamicsProcessedForcesXMLFile(
      const std::string& subjectName,
      const std::vector<dynamics::BodyNode*> contactBodies,
      const std::string& grfForcesPath,
      const std::string& forcesOutputPath);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// ID tool to recreate / validate the results of ID created from this tool
  static void saveOsimInverseDynamicsXMLFile(
      const std::string& subjectName,
      const std::string& osimInputModelPath,
      const std::string& osimInputMotPath,
      const std::string& osimForcesXmlPath,
      const std::string& osimOutputStoPath,
      const std::string& osimOutputBodyForcesStoPath,
      const std::string& idInstructionsOutputPath,
      const s_t startTime,
      const s_t endTime);

  /// This gets called by rationalizeJoints()
  static void updateRootJointLimits(
      tinyxml2::XMLElement* element, dynamics::EulerFreeJoint* joint);

  /// This gets called by rationalizeJoints()
  template <std::size_t Dimension>
  static void updateCustomJointXML(
      tinyxml2::XMLElement* element, dynamics::CustomJoint<Dimension>* joint);

  /// Read an *.osim file, move any transforms saved in Custom function
  /// translation elements into the joint offsets, and write it out to a new
  /// *.osim file. If there are no "irrational" CustomJoints, then this will
  /// just save a copy of the original skeleton.
  static void rationalizeJoints(
      const common::Uri& uri,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Read an *.osim file, overwrite all the markers, and write it out
  /// to a new *.osim file
  static void replaceOsimMarkers(
      const common::Uri& uri,
      const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
          markers,
      const std::map<std::string, bool> isAnatomical,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Read an *.osim file, move the markers to new locations, and write it out
  /// to a new *.osim file
  static void moveOsimMarkers(
      const common::Uri& uri,
      const std::map<std::string, Eigen::Vector3s>& bodyScales,
      const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
          markerOffsets,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Read an original *.osim file (which contains a marker set), and a target
  /// *.osim file, and translate the markers from the original to the target,
  /// and write it out to a new *.osim file
  ///
  /// This method returns a pair of lists, (guessedMarkers, missingMarkers).
  /// The guessedMarkers array contains the markers which were placed on the
  /// target model using heuristics. The missingMarkers array contains the
  /// markers which were not placed on the target model, because they could not
  /// be matched to any body even after heuristics were applied (this can often
  /// happen because the markers are on the arms, but the target model has no
  /// arms, for example). These should be reviewed by a person, to verify that
  /// the results look reasonable.
  static std::pair<std::vector<std::string>, std::vector<std::string>>
  translateOsimMarkers(
      const common::Uri& originalModel,
      const common::Uri& targetModel,
      const std::string& outputPath,
      bool verbose = false);

  /// This method will use several heuristics, including the names of joints,
  /// meshes, and bones to determine if this body on this skeleton should be
  /// considered an "arm," which means basically anything after an articulated
  /// shoulder joint. Importantly, a torso with fixed meshes normally associated
  /// with the arms attached to the torso is NOT considered an arm.
  static bool isArmBodyHeuristic(
      std::shared_ptr<dynamics::Skeleton> skel, const std::string& bodyName);

  /// This method will use several heuristics, including the names of joints,
  /// meshes, and bones to determine if this body on this skeleton should be
  /// considered a "torso"
  static bool isTorsoBodyHeuristic(
      std::shared_ptr<dynamics::Skeleton> skel, const std::string& bodyName);

  /// This method will return true if this skeleton has any bodies that return
  /// true for isArmBodyHeuristic().
  static bool hasArms(std::shared_ptr<dynamics::Skeleton> skel);

  /// This method will return true if this skeleton has any bodies that return
  /// true for isTorsoBodyHeuristic().
  static bool hasTorso(std::shared_ptr<dynamics::Skeleton> skel);

  /// Read an *.osim file, change the mass/COM/MOI for everything, and write it
  /// out to a new *.osim file
  static void replaceOsimInertia(
      const common::Uri& uri,
      const std::shared_ptr<dynamics::Skeleton> skel,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Read an *.osim file, then save just the markers to a new *.osim file
  static void filterJustMarkers(
      const common::Uri& uri,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This grabs the marker trajectories from a TRC file
  static OpenSimTRC loadTRC(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This saves the *.trc file from a motion for the skeleton
  static void saveTRC(
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerTimesteps);

  /// This grabs the joint angles from a *.mot file
  static OpenSimMot loadMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const common::Uri& uri,
      Eigen::Matrix3s rotateBy = Eigen::Matrix3s::Identity(),
      int downsampleByFactor = 1,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This tries a number of rotations as it's loading a .mot file, and returns
  /// the one with the lowest marker error, since that's likely to be the
  /// correct orientation.
  static OpenSimMot loadMotAtLowestMarkerRMSERotation(
      OpenSimFile& osim,
      const common::Uri& uri,
      C3D& c3d,
      int downsampleByFactor = 1,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This saves the *.mot file from a motion for the skeleton
  static void saveMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const Eigen::MatrixXs& poses);

  /// This saves the *.mot file from the inverse dynamics solved for the
  /// skeleton
  static void saveIDMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const Eigen::MatrixXs& controlForces);

  /// This saves the *.mot file for the ground reaction forces we've read from
  /// a C3D file
  static void saveRawGRFMot(
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const std::vector<biomechanics::ForcePlate> forcePlates);

  /// This saves the *.mot file for the ground reaction forces we've processed
  /// through our dynamics fitter.
  static void saveProcessedGRFMot(
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const std::vector<dynamics::BodyNode*> contactBodies,
      std::shared_ptr<dynamics::Skeleton> skel,
      const Eigen::MatrixXs& poses,
      const std::vector<biomechanics::ForcePlate>& forcePlates,
      const Eigen::MatrixXs wrenches);

  /// This saves the *.mot file with 3 columns for each body. This is
  /// basically only used for verifying consistency between Nimble and OpenSim.
  static void saveBodyLocationsMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const Eigen::MatrixXs& poses);

  /// This saves the *.mot file with 3 columns for each marker. This is
  /// basically only used for verifying consistency between Nimble and OpenSim.
  static void saveMarkerLocationsMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const dynamics::MarkerMap& markers,
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const Eigen::MatrixXs& poses);

  /// This grabs the GRF forces from a *.mot file
  static std::vector<ForcePlate> loadGRF(
      const common::Uri& uri,
      const std::vector<double>& targetTimestamps = {},
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This loads IMU data from a CSV file, where the headers of columns are
  /// names of IMUs with axis suffixes (e.g. "IMU1_Accel_X", "IMU1_Accel_Y",
  /// "IMU1_Accel_Z", "IMU1_Gyro_X", "IMU1_Gyro_Y", "IMU1_Gyro_Z")
  static OpenSimIMUData loadIMUFromCSV(
      const common::Uri& uri,
      bool isAccelInG = true,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Load excitations and activations from a MocoTrajectory *.sto file.
  static OpenSimMocoTrajectory loadMocoTrajectory(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// Append excitations and activations from a MocoTrajectory to a CSV file and
  /// save it.
  static void appendMocoTrajectoryAndSaveCSV(
      const common::Uri& uri,
      const OpenSimMocoTrajectory& mocoTraj,
      std::string path,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// When people finish preparing their model in OpenSim, they save a *.osim
  /// file with all the scales and offsets baked in. This is a utility to go
  /// through and get out the scales and offsets in terms of a standard
  /// skeleton, so that we can include their values in standard datasets, or
  /// compare their values to the results obtained from our automatic tools.
  static OpenSimScaleAndMarkerOffsets getScaleAndMarkerOffsets(
      const OpenSimFile& standardSkeleton, const OpenSimFile& scaledSkeleton);

  /// This does its best to convert a *.osim file to an SDF file. It will
  /// simplify the skeleton by merging any bodies that are requested, and
  /// deleting any joints linking those bodies.
  static bool convertOsimToSDF(
      const common::Uri& uri,
      const std::string& outputPath,
      std::map<std::string, std::string> mergeBodiesInto);

  /// This does its best to convert a *.osim file to an MJCF file. It will
  /// simplify the skeleton by merging any bodies that are requested, and
  /// deleting any joints linking those bodies.
  static bool convertOsimToMJCF(
      const common::Uri& uri,
      const std::string& outputPath,
      std::map<std::string, std::string> mergeBodiesInto);

protected:
  static OpenSimFile readOsim30(
      tinyxml2::XMLElement* docElement,
      const std::string fileNameForErrorDisplay,
      const std::string geometryFolder,
      const common::ResourceRetrieverPtr& geometryRetriever,
      bool ignoreGeometry);
  static OpenSimFile readOsim40(
      tinyxml2::XMLElement* docElement,
      const std::string fileNameForErrorDisplay,
      const std::string geometryFolder,
      const common::ResourceRetrieverPtr& geometryRetriever,
      bool ignoreGeometry);
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif