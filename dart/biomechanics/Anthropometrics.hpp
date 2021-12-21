#ifndef DART_BIOMECH_ANTHROPOMETRICS_HPP_
#define DART_BIOMECH_ANTHROPOMETRICS_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include "dart/include_eigen.hpp"

#include "dart/biomechanics/LilypadSolver.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/MultivariateGaussian.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct AnthroMetric
{
  std::string name;
  Eigen::VectorXs bodyPose;
  std::string bodyA;
  Eigen::Vector3s offsetA;
  std::string bodyB;
  Eigen::Vector3s offsetB;
  Eigen::Vector3s axis;

  AnthroMetric(
      std::string name,
      Eigen::VectorXs bodyPose,
      std::string bodyA,
      Eigen::Vector3s offsetA,
      std::string bodyB,
      Eigen::Vector3s offsetB,
      Eigen::Vector3s axis = Eigen::Vector3s::Zero());
};

class Anthropometrics
{
public:
  Anthropometrics();

  static Anthropometrics loadFromFile(
      const common::Uri& uri,
      const common::ResourceRetrieverPtr& retriever = nullptr);

  void debugToGUI(
      std::shared_ptr<server::GUIWebsocketServer> server,
      std::shared_ptr<dynamics::Skeleton> skel);

  void addMetric(
      std::string name,
      Eigen::VectorXs bodyPose,
      std::string bodyA,
      Eigen::Vector3s offsetA,
      std::string bodyB,
      Eigen::Vector3s offsetB,
      Eigen::Vector3s axis = Eigen::Vector3s::Zero());

  std::vector<std::string> getMetricNames();

  void setDistribution(std::shared_ptr<math::MultivariateGaussian> dist);

  std::shared_ptr<math::MultivariateGaussian> getDistribution();

  std::shared_ptr<Anthropometrics> condition(
      const std::map<std::string, s_t>& observedValues);

  std::map<std::string, s_t> measure(std::shared_ptr<dynamics::Skeleton> skel);

  s_t getPDF(std::shared_ptr<dynamics::Skeleton> skel);

  s_t getLogPDF(
      std::shared_ptr<dynamics::Skeleton> skel, bool normalized = true);

  void setSkelToMetricPose(
      std::shared_ptr<dynamics::Skeleton> skel, const AnthroMetric& metric);

  Eigen::VectorXs getGradientOfLogPDFWrtBodyScales(
      std::shared_ptr<dynamics::Skeleton> skel);

  Eigen::VectorXs finiteDifferenceGradientOfLogPDFWrtBodyScales(
      std::shared_ptr<dynamics::Skeleton> skel);

  Eigen::VectorXs getGradientOfLogPDFWrtGroupScales(
      std::shared_ptr<dynamics::Skeleton> skel);

  Eigen::VectorXs finiteDifferenceGradientOfLogPDFWrtGroupScales(
      std::shared_ptr<dynamics::Skeleton> skel);

protected:
  std::vector<AnthroMetric> mMetrics;
  std::shared_ptr<math::MultivariateGaussian> mDist;
};

} // namespace biomechanics
} // namespace dart

#endif