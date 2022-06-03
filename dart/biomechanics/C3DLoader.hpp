#ifndef DART_BIOMECH_C3D_HPP_
#define DART_BIOMECH_C3D_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

/// This holds marker trajectory information from an OpenSim TRC file
struct C3D
{
  int framesPerSecond;
  std::vector<double> timestamps;
  std::vector<std::string> markers;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
  std::vector<ForcePlate> forcePlates;
  // These are useful for faster access to the marker data in certain situations
  Eigen::MatrixXs shuffledMarkersMatrix;
  Eigen::MatrixXs shuffledMarkersMatrixMask;
  // This is the rotation applied to the loaded data, if any
  Eigen::Matrix3s dataRotation;

  /// This computes the weighted distance from each CoP to its nearest marker on
  /// that timestep. This is used as part of the heuristic to guess which
  /// convention a C3D file is using for storing its GRF data.
  s_t getWeightedDistFromCoPToNearestMarker();
};

class C3DLoader
{
public:
  static C3D loadC3D(const std::string& uri);

  static C3D loadC3DWithGRFConvention(const std::string& uri, int convention);

  /// This will check if markers
  /// obviously "flip" during the trajectory, and unflip them.
  static void fixupMarkerFlips(C3D* c3d);

  static void debugToGUI(
      C3D& file, std::shared_ptr<server::GUIWebsocketServer> server);
};

} // namespace biomechanics
} // namespace dart

#endif