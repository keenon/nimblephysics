#ifndef DART_BIOMECH_IK_ERRORS_HPP_
#define DART_BIOMECH_IK_ERRORS_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/Anthropometrics.hpp"
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
      std::shared_ptr<Anthropometrics> anthropometrics = nullptr);

  void printReport(int limitTimesteps = -1);

  void saveCSVMarkerErrorReport(const std::string& path);

  std::vector<std::pair<std::string, s_t>> getSortedMarkerRMSE();

  std::vector<std::string> worstMarkers;
  std::vector<Eigen::Vector3s> worstMarkerErrors;
  std::vector<Eigen::Vector3s> worstMarkerReals;
  std::vector<Eigen::Vector3s> worstMarkerPredicteds;
  std::vector<s_t> sumSquaredError;
  std::vector<s_t> rootMeanSquaredError;
  std::vector<s_t> maxError;
  s_t averageRootMeanSquaredError;
  s_t averageSumSquaredError;
  s_t averageMaxError;
  s_t anthroPDF;
  std::vector<std::string> markerNames;
  std::map<std::string, int> numMarkerObservations;
  std::map<std::string, s_t> rmseMarkerErrors;
  std::vector<std::map<std::string, s_t>> markerErrorTimesteps;
};

} // namespace biomechanics
} // namespace dart

#endif