#ifndef DART_BIOMECH_C3D_HPP_
#define DART_BIOMECH_C3D_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct ForcePlate
{
  Eigen::Vector3s worldOrigin;
  std::vector<Eigen::Vector3s> corners;
  std::vector<Eigen::Vector3s> centersOfPressure;
  std::vector<Eigen::Vector3s> moments;
  std::vector<Eigen::Vector3s> forces;
};

/// This holds marker trajectory information from an OpenSim TRC file
struct C3D
{
  std::vector<double> timestamps;
  std::vector<std::string> markers;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerTimesteps;
  std::vector<ForcePlate> forcePlates;
  // These are useful for faster access to the marker data in certain situations
  Eigen::MatrixXs shuffledMarkersMatrix;
  Eigen::MatrixXs shuffledMarkersMatrixMask;
};

class C3DLoader
{
public:
  static C3D loadC3D(const std::string& uri);

  static void debugToGUI(
      C3D& file, std::shared_ptr<server::GUIWebsocketServer> server);
};

} // namespace biomechanics
} // namespace dart

#endif