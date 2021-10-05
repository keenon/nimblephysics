#ifndef DART_BIOMECH_IK_ERRORS_HPP_
#define DART_BIOMECH_IK_ERRORS_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/LilypadSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

class IKErrorReport
{
public:
  IKErrorReport(
      std::shared_ptr<dynamics::Skeleton> skel,
      dynamics::MarkerMap markers,
      Eigen::MatrixXs poses,
      std::vector<std::map<std::string, Eigen::Vector3s>> observations,
      std::vector<std::string> activeMarkers);

  void printReport(int limitTimesteps = -1);

  std::vector<std::string> worstMarkers;
  std::vector<Eigen::Vector3s> worstMarkerErrors;
  std::vector<s_t> sumSquaredError;
  std::vector<s_t> rootMeanSquaredError;
  std::vector<s_t> maxError;
  s_t averageRootMeanSquaredError;
  s_t averageSumSquaredError;
  s_t averageMaxError;
};

} // namespace biomechanics
} // namespace dart

#endif