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
  // TODO: eventually we'll want to record muscles here
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
};

struct OpenSimMot
{
  std::vector<double> timestamps;
  Eigen::MatrixXs poses;
};

class OpenSimParser
{
public:
  /// Read Skeleton from *.osim file
  static OpenSimFile parseOsim(
      const common::Uri& uri,
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