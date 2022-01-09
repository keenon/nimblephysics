#ifndef DART_UTILS_OSIMPARSER_HPP_
#define DART_UTILS_OSIMPARSER_HPP_

#include <map>
#include <string>

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/Skeleton.hpp"
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
  // TODO: use these
  std::vector<std::string> anatomicalMarkers;
  std::vector<std::string> trackingMarkers;
  // TODO: eventually we'll want to record muscles here

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

class OpenSimParser
{
public:
  /// Read Skeleton from *.osim file
  static OpenSimFile parseOsim(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This creates an XML configuration file, which you can pass to the OpenSim
  /// scaling tool to rescale a skeleton
  static void saveOsimScalingXMLFile(
      std::shared_ptr<dynamics::Skeleton> skel,
      double massKg,
      double heightM,
      const std::string& osimInputPath,
      const std::string& osimOutputPath,
      const std::string& scalingInstructionsOutputPath);

  /// Read an *.osim file, move the markers to new locations, and write it out
  /// to a new *.osim file
  static void moveOsimMarkers(
      const common::Uri& uri,
      const std::map<std::string, Eigen::Vector3s>& bodyScales,
      const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
          markerOffsets,
      const std::string& outputPath,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This grabs the marker trajectories from a TRC file
  static OpenSimTRC loadTRC(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This grabs the joint angles from a *.mot file
  static OpenSimMot loadMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const common::Uri& uri,
      int downsampleByFactor = 1,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// This saves the *.mot file from a motion for the skeleton
  static void saveMot(
      std::shared_ptr<dynamics::Skeleton> skel,
      const std::string& outputPath,
      const std::vector<double>& timestamps,
      const Eigen::MatrixXs& poses);

  /// This grabs the GRF forces from a *.mot file
  static OpenSimGRF loadGRF(
      const common::Uri& uri,
      int downsampleByFactor = 1,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  /// When people finish preparing their model in OpenSim, they save a *.osim
  /// file with all the scales and offsets baked in. This is a utility to go
  /// through and get out the scales and offsets in terms of a standard
  /// skeleton, so that we can include their values in standard datasets, or
  /// compare their values to the results obtained from our automatic tools.
  static OpenSimScaleAndMarkerOffsets getScaleAndMarkerOffsets(
      const OpenSimFile& standardSkeleton, const OpenSimFile& scaledSkeleton);

protected:
  static OpenSimFile readOsim30(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
  static OpenSimFile readOsim40(
      const common::Uri& uri,
      tinyxml2::XMLElement* docElement,
      const common::ResourceRetrieverPtr& retriever);
}; // namespace OpenSimParser

} // namespace biomechanics
} // namespace dart

#endif